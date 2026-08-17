"""vLLM MoE backend shim: VkernelFusedExperts.

Calls vkernels' HIP C ABI (libvkernels_hip.so) for MXFP4 fused MoE on
gfx942 (MI300A). Replaces the broken AITER/Triton MoE path for Kimi-K3.

The C ABI library is loaded via ctypes. All weight/scale/bias tensors are
passed as raw device pointers. The kernel does the full computation:
gate-up GEMM -> activation (SwiGLU/SiTU) -> down GEMM -> routing weight
application -> top-k summation. Output is fp32, converted to bf16.

moe_align_block_size is done on CPU (small cost for routing metadata,
O(M*top_k) tokens sorted by expert). This can be replaced with a GPU
implementation later for performance.

Validated on MI300A (gfx942) — all 8 C++ GPU tests + 4 Python ctypes
tests pass (job 596227). See deployments/llm/beverin/vkernels/README.md.
"""

import ctypes
import glob
import os

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.config import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
    UnfusedOAITritonExperts,
)
from vllm.platforms import current_platform


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_lib_cache = {}


def _find_libvkernels_hip():
    """Find libvkernels_hip.so, checking env var and common paths."""
    env_path = os.environ.get("VKERNELS_LIB")
    if env_path and os.path.exists(env_path):
        return env_path

    k3 = os.environ.get("K3", "")
    if k3:
        k3_path = os.path.join(k3, "home/pylib/libvkernels_hip.so")
        if os.path.exists(k3_path):
            return k3_path

    vdir = os.environ.get("VKERNELS_DIR", "/capstor/scratch/cscs/xyao/vkernels")
    cands = sorted(
        glob.glob(
            os.path.join(vdir, "build", "hip", "**", "libvkernels_hip.so"),
            recursive=True,
        )
    )
    if cands:
        return cands[0]

    return None


def _get_lib():
    """Load and cache libvkernels_hip.so."""
    if "lib" not in _lib_cache:
        path = _find_libvkernels_hip()
        if path is None:
            raise RuntimeError(
                "libvkernels_hip.so not found. Set VKERNELS_LIB or "
                "VKERNELS_DIR, or place it in $K3/home/pylib/"
            )
        _lib_cache["lib"] = ctypes.CDLL(path)
    return _lib_cache["lib"]


# ---------------------------------------------------------------------------
# moe_align_block_size (CPU reference, matching vkernels' expected format)
# ---------------------------------------------------------------------------


def _moe_align_block_size_cpu(
    topk_ids_flat, num_experts, block_size, expert_map=None
):
    """Pure-CPU moe_align_block_size, matching vkernels' expected format.

    Args:
        topk_ids_flat: [M*top_k] numpy int32 (global expert IDs)
        num_experts: global number of experts
        block_size: GEMM block size (16 decode, 64 prefill)
        expert_map: [num_experts] numpy int32 (global->local, -1=skip)

    Returns:
        sorted_ids: [EM] int32 (token indices, padded with N=M*top_k)
        expert_ids: [EM//block_size] int32 (local expert per block, -1=pad)
        EM: int (total padded token count)
    """
    N = len(topk_ids_flat)
    per_expert = {}
    for i in range(N):
        global_e = int(topk_ids_flat[i])
        if expert_map is not None:
            local_e = int(expert_map[global_e])
            if local_e == -1:
                continue
        else:
            local_e = global_e
        per_expert.setdefault(local_e, []).append(i)

    local_experts = sorted(per_expert.keys())
    EM = sum(
        ((len(per_expert[e]) + block_size - 1) // block_size) * block_size
        for e in local_experts
    )
    # Ensure EM is at least block_size (avoids zero-length arrays)
    if EM == 0:
        EM = block_size
    sids = np.full(EM, N, dtype=np.int32)
    eids = np.full(EM // block_size, -1, dtype=np.int32)

    idx = 0
    blk_idx = 0
    for e in local_experts:
        tokens = per_expert[e]
        for i in tokens:
            sids[idx] = i
            idx += 1
        padded_nt = ((len(tokens) + block_size - 1) // block_size) * block_size
        num_blocks = padded_nt // block_size
        for b in range(num_blocks):
            if b * block_size < len(tokens):
                eids[blk_idx] = e
            blk_idx += 1
        for _ in range(len(tokens), padded_nt):
            sids[idx] = N
            idx += 1

    return sids, eids, EM


# ---------------------------------------------------------------------------
# VkernelFusedExperts
# ---------------------------------------------------------------------------


class VkernelFusedExperts(UnfusedOAITritonExperts):
    """vkernels HIP C ABI backend for MXFP4 MoE on gfx942 (MI300A).

    Calls vk_fused_moe_mxfp4 via ctypes. The kernel does the full MoE
    computation: gate-up GEMM -> activation -> down GEMM -> routing weight
    application -> top-k summation. Output is fp32, converted to bf16.

    moe_align_block_size is done on CPU (small cost for routing metadata).
    Weight format: [E, 2*ispp, hidden/2] uint8 (w13), [E, hidden, ispp/2]
    uint8 (w2) — matches vLLM's Mxfp4MoEMethod.create_weights() exactly.
    """

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_rocm():
            return False
        # The library was compiled for gfx942 (MI300A).  Check the
        # actual device architecture to avoid a segfault on gfx90a.
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                gcn = getattr(props, "gcnArchName", "")
                return "gfx942" in gcn
        except Exception:
            pass
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SITU,  # Kimi-K3 SiTU
        ]

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        """Allocate workspace for act_scratch [EM_max, ispp] bf16.

        The framework allocates workspace tensors with dtype from
        workspace_dtype() (bf16 by default). We use workspace13 for
        act_scratch and allocate out_fp32 separately.
        """
        ispp = N // 2
        block_size = 16 if M <= 32 else 64
        # EM_max: padded(M*topk + local_num_experts) — matches the C ABI's
        # moe_align_block_size output size.
        EM_max = ((M * topk + local_num_experts + block_size - 1)
                  // block_size) * block_size
        # workspace13: act_scratch [EM_max, ispp] bf16
        ws13 = (EM_max, ispp)
        # workspace2: not used (we allocate out_fp32 separately)
        ws2 = (0,)
        output = (M, K)
        return (ws13, ws2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta,
        apply_router_weight_on_input: bool,
    ) -> None:
        lib = _get_lib()
        dev = hidden_states.device

        M, K = hidden_states.shape
        E, N, _ = w1.shape  # E=local experts, N=2*ispp
        ispp = N // 2
        top_k = topk_ids.size(1)

        if global_num_experts == -1:
            global_num_experts = E

        # Block size: 16 for decode (small M), 64 for prefill (large M)
        block_size = 16 if M <= 32 else 64

        # moe_align_block_size (CPU, then copy to GPU)
        topk_ids_flat = (
            topk_ids.contiguous().view(-1).cpu().numpy().astype(np.int32)
        )
        expert_map_np = (
            expert_map.cpu().numpy().astype(np.int32)
            if expert_map is not None
            else None
        )
        sids_np, eids_np, EM = _moe_align_block_size_cpu(
            topk_ids_flat, global_num_experts, block_size, expert_map_np
        )

        d_sids = torch.from_numpy(sids_np).to(dev)
        d_eids = torch.from_numpy(eids_np).to(dev)

        # Routing weights: if already applied on input, use ones
        if apply_router_weight_on_input:
            topk_w = torch.ones_like(topk_weights).contiguous().view(-1)
        else:
            topk_w = topk_weights.contiguous().view(-1)

        topk_ids_dev = topk_ids.contiguous().view(-1)

        # Activation parameters
        if activation == MoEActivation.SITU:
            act_code = 1  # SiTU
            beta_raw = getattr(self.moe_config, "activation_situ_beta", None)
            beta = float(beta_raw) if beta_raw is not None else 1.0
            linear_beta_raw = getattr(
                self.moe_config, "activation_situ_linear_beta", None
            )
            linear_beta = float(linear_beta_raw) if linear_beta_raw is not None else 25.0
        else:
            act_code = 0  # SwiGLU
            beta = 0.0
            linear_beta = 0.0

        swiglu_limit_raw = getattr(self, "gemm1_clamp_limit", None)
        swiglu_limit = float(swiglu_limit_raw) if swiglu_limit_raw is not None else 4.0

        # Scales and biases (from quant_config properties)
        w13_scale = self.w1_scale
        w2_scale = self.w2_scale
        b13 = self.w1_bias
        b2 = self.w2_bias

        # act_scratch [EM, ispp] bf16 — from workspace13
        if workspace13.numel() >= EM * ispp:
            act_scratch = workspace13[: EM * ispp].view(EM, ispp)
        else:
            act_scratch = torch.empty(
                EM * ispp, dtype=torch.bfloat16, device=dev
            )

        # Temp fp32 output [M*K] — allocated separately (fp32, not bf16)
        out_fp32 = torch.empty(M * K, dtype=torch.float32, device=dev)

        lib.vk_fused_moe_mxfp4(
            ctypes.c_void_p(hidden_states.data_ptr()),
            ctypes.c_void_p(w1.data_ptr()),
            ctypes.c_void_p(w13_scale.data_ptr()),
            ctypes.c_void_p(w2.data_ptr()),
            ctypes.c_void_p(w2_scale.data_ptr()),
            ctypes.c_void_p(topk_ids_dev.data_ptr()),
            ctypes.c_void_p(topk_w.data_ptr()),
            ctypes.c_void_p(act_scratch.data_ptr()),
            ctypes.c_void_p(out_fp32.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(K),
            ctypes.c_int(ispp),
            ctypes.c_int(top_k),
            ctypes.c_void_p(d_sids.data_ptr()),
            ctypes.c_void_p(d_eids.data_ptr()),
            ctypes.c_int(EM),
            ctypes.c_float(swiglu_limit),
            ctypes.c_int(act_code),
            ctypes.c_float(beta),
            ctypes.c_float(linear_beta),
            ctypes.c_void_p(b13.data_ptr()) if b13 is not None else None,
            ctypes.c_void_p(b2.data_ptr()) if b2 is not None else None,
            ctypes.c_int(block_size),
        )
        torch.cuda.synchronize()

        # Convert fp32 output to bf16 and write to vLLM's output tensor
        output.copy_(out_fp32.view(M, K).to(torch.bfloat16))
