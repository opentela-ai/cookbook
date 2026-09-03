"""SM80 in-kernel e4m3 decode for fused_moe: parity + perf tests.

Background: the SM80 fp8 path used to upcast the ENTIRE expert-weight tensor
B (~1-2 GB fp8 -> 2-4 GB bf16, a 2.25 GiB transient) on every MoE invocation
-- twice per layer, every decode step. The job-83152 profile measured those
`unrolled_elementwise_kernel` copies at ~88% of per-step GPU time (537 of
610 ms over 3 steps). The fix reinterprets B as raw uint8 (a legal pointer
type on SM80) and decodes e4m3 in-register (_decode_e4m3_u8), which is exact
(bit-injected 2^k, <=4 mantissa bits) and therefore bit-identical to the old
host-side B.to(compute_dtype).

Run inside the sglang container with PYTHONPATH=$DEPLOY_DIR/patches_full:

    python3 test_moe_e4m3_udecode.py

T1  decode helper vs torch fp8->bf16 upcast: bitwise over ALL 256 byte
    values (denormals, -0, NaN included).
T2  invoke_fused_moe_kernel end-to-end: new path (B fp8 -> uint8 view) vs
    the old behavior reproduced with B pre-upcast to bf16, identical
    quantized A / scales / routing: outputs must be BITWISE equal.
T3  perf: per-call wall time old vs new on a realistic expert-weight shape.
"""
import sys
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    _decode_e4m3_u8,
    invoke_fused_moe_kernel,
)

DEV = "cuda"
GROUP = 128
FAILURES = []


def check(name, ok, extra=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}{(' -- ' + extra) if extra else ''}")
    if not ok:
        FAILURES.append(name)


def quant_groups(x_bf16, group=GROUP):
    """Per-last-dim-group e4m3 quant: returns (fp8, fp32 scales)."""
    shape = x_bf16.shape
    assert shape[-1] % group == 0
    xg = x_bf16.float().reshape(-1, group)
    amax = xg.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / 448.0
    q = (xg / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(torch.float8_e4m3fn).view(shape), scale.view(
        *shape[:-1], shape[-1] // group
    )


def quant_blocks_2d(w_bf16, group=GROUP):
    """Per (out_blk, in_blk) 128x128-block e4m3 quant for weights [N, K]."""
    N, K = w_bf16.shape
    assert N % group == 0 and K % group == 0
    w = (
        w_bf16.float()
        .view(N // group, group, K // group, group)
        .permute(0, 2, 1, 3)
        .reshape(-1, group * group)
    )
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / 448.0
    q = (w / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    q = q.view(N // group, K // group, group, group).permute(0, 2, 1, 3).reshape(N, K)
    scale = scale.view(N // group, K // group)
    return q.view(torch.float8_e4m3fn), scale.float()


@triton.jit
def _probe_decode(u_ptr, o_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    u = tl.load(u_ptr + offs, mask=offs < N)
    tl.store(o_ptr + offs, _decode_e4m3_u8(u, tl.bfloat16), mask=offs < N)


def t1_helper_bitwise():
    all_bytes = torch.arange(256, dtype=torch.int32, device=DEV).to(torch.uint8)
    ref = all_bytes.view(torch.float8_e4m3fn).to(torch.bfloat16)
    out = torch.empty(256, dtype=torch.bfloat16, device=DEV)
    _probe_decode[(1,)](all_bytes, out, 256, BLOCK=256)
    torch.cuda.synchronize()
    # NaN payloads may differ (both are NaN); everything else must be bitwise.
    ob, rb = out.view(torch.int16), ref.view(torch.int16)
    bit_same = (ob == rb) | (out.isnan() & ref.isnan())
    nan_xor = (out.isnan() ^ ref.isnan()).sum().item()
    n_bad = (~bit_same).sum().item()
    same = n_bad == 0 and nan_xor == 0
    check("T1 decode helper bitwise vs torch (all 256 values)", same,
          f"mismatched={n_bad} nan_xored={nan_xor}")
    if not same:
        for i in (~bit_same).nonzero()[:8].flatten().tolist():
            print(f"    byte=0x{i:02x} got=0x{int(ob[i]):04x} ref=0x{int(rb[i]):04x}")


def align_block_size_manual(topk_ids, block):
    """Reproduce moe_align_block_size: sort tokens by expert, pad each
    segment to a multiple of `block`."""
    E = int(topk_ids.max().item()) + 1
    flat = topk_ids.view(-1).to(torch.int32)
    n = flat.numel()
    order = torch.argsort(flat, stable=True).to(torch.int32)
    counts = torch.bincount(flat, minlength=E)
    segs, expert_ids = [], []
    for e in range(E):
        c = int(counts[e].item())
        padded = ((c + block - 1) // block) * block
        seg = order[flat[order] == e] if c else order[:0]
        pad = torch.full((padded - c,), n, dtype=torch.int32, device=flat.device)
        segs.append(torch.cat([seg, pad]))
        expert_ids.append(
            torch.full((padded // block,), e, dtype=torch.int32, device=flat.device)
        )
    sorted_token_ids = torch.cat(segs).contiguous()
    expert_ids = torch.cat(expert_ids).contiguous()
    num_tokens_post_padded = torch.tensor(
        [sorted_token_ids.numel()], dtype=torch.int32, device=flat.device
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def run_invoke(A, B, A_scale, B_scale, topk_weights, topk_ids, routed, C):
    M, TOPK = topk_ids.shape
    N, K = B.shape[1], B.shape[2]
    sorted_token_ids, expert_ids, num_tokens_post_padded = align_block_size_manual(
        topk_ids, 16
    )
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": GROUP,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    }
    invoke_fused_moe_kernel(
        A,
        B,
        None,
        C,
        A_scale,
        B_scale,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=TOPK,
        config=config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=[GROUP, GROUP],
    )
    torch.cuda.synchronize()
    return C


def t2_invoke_bitwise():
    torch.manual_seed(7)
    E, N, K, M, TOPK = 4, 256, 512, 8, 2
    w_bf16 = torch.randn(E, N, K, device=DEV, dtype=torch.float32) * 0.3
    a_bf16 = torch.randn(M, K, device=DEV, dtype=torch.float32) * 0.5

    B_fp8 = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=DEV)
    B_scale = torch.empty(E, N // GROUP, K // GROUP, device=DEV)
    for e in range(E):
        q, s = quant_blocks_2d(w_bf16[e])
        B_fp8[e] = q
        B_scale[e] = s
    A_fp8, A_scale = quant_groups(a_bf16)

    topk_ids = torch.randint(0, E, (M, TOPK), device=DEV, dtype=torch.int32)
    topk_weights = torch.rand(M, TOPK, device=DEV, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

    C_new = torch.zeros(M, TOPK, N, device=DEV, dtype=torch.bfloat16)
    run_invoke(A_fp8, B_fp8, A_scale, B_scale, topk_weights, topk_ids, True, C_new)

    # Old behavior: host-side upcast of the quantized weights, kernel no-op.
    B_bf16 = B_fp8.to(torch.bfloat16)
    C_old = torch.zeros(M, TOPK, N, device=DEV, dtype=torch.bfloat16)
    run_invoke(A_fp8, B_bf16, A_scale, B_scale, topk_weights, topk_ids, True, C_old)

    same = torch.equal(C_new.view(torch.int16), C_old.view(torch.int16))
    diff = (C_new.float() - C_old.float()).abs().max().item()
    check("T2 invoke end-to-end bitwise (new in-kernel vs old host upcast)", same, f"maxabs={diff:.3e}")
    if not same:
        n_bad = (C_new.view(torch.int16) != C_old.view(torch.int16)).sum().item()
        print(f"    mismatched elements: {n_bad} / {C_new.numel()}")

    # Reference dequant matmul (tolerance): fp32 emulation of the fp8 math.
    ref = torch.zeros(M, TOPK, N, device=DEV, dtype=torch.float32)
    A_dq = A_fp8.to(torch.float32) * A_scale.repeat_interleave(GROUP, dim=1)
    for m in range(M):
        for t in range(TOPK):
            e = int(topk_ids[m, t].item())
            B_dq = B_fp8[e].to(torch.float32) * B_scale[e].repeat_interleave(
                GROUP, dim=0
            ).repeat_interleave(GROUP, dim=1)
            ref[m, t] = A_dq[m] @ B_dq.T
    ref = (ref * topk_weights.unsqueeze(-1)).to(torch.bfloat16)
    err = (C_new.float() - ref.float()).abs()
    rel = err.max().item() / max(ref.float().abs().max().item(), 1e-9)
    check(
        "T2b invoke vs fp32 dequant reference (rel)",
        rel < 0.02,
        f"max_rel={rel:.4f}",
    )


def quant_blocks_3d(w, group=GROUP):
    """Per (out_blk, in_blk) 128x128-block e4m3 quant for weights [E, N, K]."""
    E, N, K = w.shape
    assert N % group == 0 and K % group == 0
    wv = (
        w.float()
        .view(E, N // group, group, K // group, group)
        .permute(0, 1, 3, 2, 4)
        .reshape(E, -1, group * group)
    )
    amax = wv.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
    scale = amax / 448.0
    q = (wv / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    q = (
        q.view(E, N // group, K // group, group, group)
        .permute(0, 1, 3, 2, 4)
        .reshape(E, N, K)
    )
    return q, scale.view(E, N // group, K // group)


def run_invoke_pre(A, B, A_scale, B_scale, topk_weights, topk_ids,
                   sorted_token_ids, expert_ids, num_tokens_post_padded, C):
    """Timed region: invoke only (alignment precomputed outside)."""
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": GROUP,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    }
    invoke_fused_moe_kernel(
        A,
        B,
        None,
        C,
        A_scale,
        B_scale,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=topk_ids.shape[1],
        config=config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=[GROUP, GROUP],
    )


def t3_perf():
    free_b, total_b = torch.cuda.mem_get_info()
    free_gb = free_b / 1e9
    E, N, K, M, TOPK = 72, 2048, 2048, 8, 2
    need_gb = (E * N * K * (1 + 2)) / 1e9 + 2.0
    if free_gb < need_gb:
        check("T3 perf", True, f"skipped: {free_gb:.1f} GB free < {need_gb:.1f} GB needed (server co-located)")
        return
    print(f"    (free={free_gb:.1f} GB)")
    torch.manual_seed(3)
    w = torch.randn(E, N, K, device=DEV, dtype=torch.float32).mul(0.3).clamp(-3, 3)
    B_fp8, B_scale = quant_blocks_3d(w)
    a_bf16 = torch.randn(M, K, device=DEV, dtype=torch.float32)
    A_fp8, A_scale = quant_groups(a_bf16)
    topk_ids = torch.randint(0, E, (M, TOPK), device=DEV, dtype=torch.int32)
    topk_weights = (torch.rand(M, TOPK, device=DEV) / TOPK).float()
    C = torch.zeros(M, TOPK, N, device=DEV, dtype=torch.bfloat16)
    B_bf16 = B_fp8.to(torch.bfloat16)
    sorted_token_ids, expert_ids, num_tokens_post_padded = align_block_size_manual(
        topk_ids, 16
    )

    def bench(fn, iters=20):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    # Sanity: precomputed-alignment path is still bitwise identical.
    C.zero_(); run_invoke_pre(A_fp8, B_fp8, A_scale, B_scale, topk_weights, topk_ids,
                              sorted_token_ids, expert_ids, num_tokens_post_padded, C)
    C_new = C.clone()
    C.zero_(); run_invoke_pre(A_fp8, B_bf16, A_scale, B_scale, topk_weights, topk_ids,
                              sorted_token_ids, expert_ids, num_tokens_post_padded, C)
    bit_same = torch.equal(C_new.view(torch.int16), C.view(torch.int16))

    t_up = bench(lambda: B_fp8.to(torch.bfloat16))
    C.zero_()
    t_new = bench(lambda: run_invoke_pre(A_fp8, B_fp8, A_scale, B_scale, topk_weights,
                                         topk_ids, sorted_token_ids, expert_ids,
                                         num_tokens_post_padded, C))
    C.zero_()
    t_old = bench(lambda: run_invoke_pre(A_fp8, B_bf16, A_scale, B_scale, topk_weights,
                                         topk_ids, sorted_token_ids, expert_ids,
                                         num_tokens_post_padded, C))
    check(
        "T3 bitwise with precomputed alignment",
        bit_same,
        f"bitwise_equal={bit_same}",
    )
    # Honest serving comparison: the old path pays the full-tensor upcast on
    # EVERY call; the new path never does.
    t_old_total = t_old + t_up
    check(
        "T3 perf (per fused_moe call, E=72 N=K=2048)",
        t_new <= t_old_total,
        f"old(kernel {t_old:.3f} + upcast {t_up:.3f})={t_old_total:.3f} ms"
        f"  new={t_new:.3f} ms  speedup={t_old_total / max(t_new, 1e-9):.1f}x",
    )


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a GPU"
    print(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} triton={triton.__version__}")
    t1_helper_bitwise()
    t2_invoke_bitwise()
    t3_perf()
    print()
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL PASS")
