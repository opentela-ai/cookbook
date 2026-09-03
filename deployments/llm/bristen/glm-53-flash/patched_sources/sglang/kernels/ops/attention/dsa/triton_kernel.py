import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def _act_quant_sm80(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SM80 (A100) torch equivalent of the fp8e4nv Triton ``act_quant`` kernel.

    GLM53-SM80-PATCH: ``_act_quant_kernel`` stores ``y`` through ``*fp8e4nv``
    pointers, which Triton refuses to JIT on SM80 ("type fp8e4nv not supported
    in this architecture" -- the decode wall that killed jobs 82822/83091 in
    ``dsa_indexer_kpool`` -> ``forward_absorb_prepare``). torch's software
    float<->float8_e4m3fn casts carry no arch requirement (validated on A100,
    torch 2.13), so replicate the kernel semantics exactly with torch ops:

        amax  = max|x| per (row, block), clamped to >= 1e-4   (fp32 math)
        scale = amax/448; or 2^ceil(log2(amax/448)) if scale_fmt (round_scale)
        y     = clamp(x/scale, -448, 448) -> float8_e4m3fn (RTNE, as the HW cvt)
        s     = scale, fp32, shape x.shape[:-1] + (N // block_size,)

    Same "FP8 storage, fp32/bf16 compute" contract as the other SM80 patches
    (see engine.sh): consumers dequantize via ``y * s``, so a pow2 boundary
    difference in ceil(log2(.)) vs tl.log2 only shifts one block's (y, s)
    inversely and leaves ``y * s`` (hence the logits) unchanged.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    N = x.size(-1)
    n_blocks = N // block_size
    x_flat = x.view(-1, N)
    M = x_flat.size(0)

    # fp32 math, exactly like the Triton kernel (loads .to(tl.float32))
    xb = x_flat.to(torch.float32).view(M, n_blocks, block_size)
    amax = xb.abs().amax(dim=-1)
    amax = torch.clamp(amax, min=1e-4)  # same zero guard as the kernel
    if scale_fmt is not None:
        # round_scale: pow2 ceiling, same formula as the Triton kernel
        scale = torch.exp2(torch.ceil(torch.log2(amax * (1.0 / 448.0))))
    else:
        scale = amax * (1.0 / 448.0)

    y = torch.clamp(xb / scale.unsqueeze(-1), -448.0, 448.0)
    y = y.to(torch.float8_e4m3fn).view(x.shape)
    s = scale.view(*x.size()[:-1], n_blocks)
    return y, s


# Triton implementation
@triton.jit
def _act_quant_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for activation quantization.

    Each block processes BLOCK_M rows and group_size columns.
    """
    # Get block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # FP8 constants
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    # Calculate row and column offsets
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size

    # Create offset arrays
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # Mask for valid rows and columns
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    # Load input data
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute absolute max along columns (group_size dimension) for each row
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=1)  # Shape: (BLOCK_M,)

    # Clamp amax to avoid division by zero
    amax = tl.maximum(amax, 1e-4)

    # Compute scale
    if round_scale:
        # Fast round scale using bit manipulation approximation
        # This is a simplified version - the exact bit manipulation is harder in Triton
        # Using log2 + ceil + pow2 as approximation
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)

    # Store quantized output
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    # Store scales
    s_cols = pid_n
    s_ptrs = S_ptr + rows * (N // group_size) + s_cols
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with Triton.

    GLM53-SM80-PATCH: on SM80 the Triton kernel below cannot JIT (fp8e4nv
    pointer stores are Hopper+ only), so dispatch to the torch equivalent
    ``_act_quant_sm80`` (same fp8+scale bytes, same shapes). Disable with
    ``SGLANG_SM80_ACT_QUANT_DISABLE=1`` for diagnosis.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    if (
        x.is_cuda
        and os.environ.get("SGLANG_SM80_ACT_QUANT_DISABLE", "") != "1"
        and torch.cuda.get_device_capability(x.device)[0] == 8
    ):
        return _act_quant_sm80(x, block_size, scale_fmt)

    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # Flatten all dims except last
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)

    # Allocate output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    y_flat = y.view(-1, N)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    s_flat = s.view(-1, N // block_size)

    # Launch kernel
    BLOCK_M = 32
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None

    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=0 if round_scale else 2,
    )

    return y, s


@triton.jit
def _get_valid_kv_indices_kernel(
    page_table_ptr,  # [bs, topk]
    kv_indptr_ptr,  # [bs + 1]
    kv_indices_ptr,  # [bs * topk] output buffer
    bs: tl.constexpr,
    topk: tl.constexpr,
):
    """
    Extract valid indices (non -1) from page_table into kv_indices.
    Each program handles one batch.
    """
    batch_id = tl.program_id(0)

    # Get the start position for this batch in kv_indices
    dst_start = tl.load(kv_indptr_ptr + batch_id)

    # Load all topk indices for this batch
    src_offset = batch_id * topk
    offsets = tl.arange(0, topk)
    indices = tl.load(page_table_ptr + src_offset + offsets)

    # Count valid indices and compact them
    mask = indices != -1

    # Use prefix sum to compute destination positions for valid elements
    # For each position, count how many valid elements are before it
    prefix_sum = tl.cumsum(mask.to(tl.int32), axis=0) - 1

    # Store valid indices to their compacted positions
    dst_positions = dst_start + prefix_sum
    tl.store(kv_indices_ptr + dst_positions, indices, mask=mask)


def get_valid_kv_indices(
    page_table_1: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    bs: int,
):
    """
    Extract valid indices from page_table_1 into kv_indices buffer.

    Args:
        page_table_1: [bs, topk] page table with -1 as invalid
        kv_indptr: [bs + 1] cumulative count of valid indices per batch
        kv_indices: [bs * topk] pre-allocated output buffer
        bs: batch size
    """
    topk = page_table_1.shape[1]
    grid = (bs,)
    _get_valid_kv_indices_kernel[grid](
        page_table_1,
        kv_indptr,
        kv_indices,
        bs,
        topk,
    )
