"""A/B correctness: current CPU routing (actual EM) vs Option C
(GPU vLLM moe_align_block_size + max_EM host constant).

Both paths feed the SAME C ABI (vk_hip_fused_moe_mxfp4, 24 args) on GPU
tensors. Expectation: outputs match within FP tolerance (the GPU path may
group tokens by a different expert block ORDER -> ~1e-5 non-associativity).

Run in-container (1 GPU, kimi-k3-vllm env):
  srun --partition=mi300 --nodes=1 --ntasks=1 --gpus-per-node=1 \
       --time=00:05:00 --account=a-infra02 --environment=kimi-k3-vllm \
       --chdir=<this dir> python3 moe_routing_ab.py
"""
import ctypes
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.environ.get("K3", ""), "home/pylib"))
from vkernels_experts import (  # noqa: E402
    _find_libvkernels_hip,
    _moe_align_block_size_cpu,
    _resolve_moe_fn,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size as gpu_moe_align,
)

DEV = "cuda:0"


def _f2bf(v):
    bits = int.from_bytes(np.float32(v).tobytes(), "little")
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFFFFFF
    return np.uint16((bits >> 16) & 0xFFFF)


def _e2m1_nibble(f):
    neg = f < 0
    af = abs(f)
    if af == 0.0 or np.isnan(af):
        return 0x8 if neg else 0x0
    if np.isinf(af):
        return 0xE if neg else 0x6
    vals = [0.25, 1.0, 1.5, 2.0, 3.0]
    nibs = [1, 2, 3, 4, 5]
    best_d = abs(af - vals[0]); best_n = nibs[0]
    for v, n in zip(vals[1:], nibs[1:]):
        d = abs(af - v)
        if d < best_d:
            best_d = d; best_n = n
    return (best_n | 0x8) if neg else best_n


def _pack_e2m1_pair(v0, v1):
    return _e2m1_nibble(v0) | (_e2m1_nibble(v1) << 4)


def _ones_weight_bytes(*shape):
    # e2m1 pairs = 1.0 (uint8, two nibbles per byte) -- matches test_tiny_sanity
    return np.full(shape, _pack_e2m1_pair(1.0, 1.0), dtype=np.uint8)


def round_up(x, m):
    return ((x + m - 1) // m) * m


def run_cabi(moe_fn, A, w13, w13_scale, w2, w2_scale, topk_ids, topk_w,
             sorted_ids, expert_ids, EM, ispp, top_k, block_size, dev):
    M, K = A.shape
    out = torch.empty(M * K, dtype=torch.float32, device=dev)
    act = torch.empty(EM * ispp, dtype=torch.bfloat16, device=dev)
    stream = torch.cuda.current_stream().cuda_stream
    moe_fn(
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(w13.data_ptr()),
        ctypes.c_void_p(w13_scale.data_ptr()),
        ctypes.c_void_p(w2.data_ptr()),
        ctypes.c_void_p(w2_scale.data_ptr()),
        ctypes.c_void_p(topk_ids.contiguous().view(-1).data_ptr()),
        ctypes.c_void_p(topk_w.contiguous().view(-1).data_ptr()),
        ctypes.c_void_p(act.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(K),
        ctypes.c_int(ispp),
        ctypes.c_int(top_k),
        ctypes.c_void_p(sorted_ids.data_ptr()),
        ctypes.c_void_p(expert_ids.data_ptr()),
        ctypes.c_int(EM),
        ctypes.c_float(4.0),                # swiglu_limit
        ctypes.c_int(0),                    # activation (SwiGLU)
        ctypes.c_float(0.0), ctypes.c_float(0.0),  # beta, linear_beta
        None, None,                         # b13, b2
        ctypes.c_int(block_size),
        ctypes.c_void_p(stream),
    )
    return out.view(M, K)


def main():
    # SIMPLE first: the exact test_tiny_sanity config (validated).
    # top_k=1, single expert, NO expert_map. Add complexity once this passes.
    M, hidden, ispp = 16, 128, 64
    top_k = 1
    num_experts = 1
    block_size = 16
    expert_map = None

    topk_ids = torch.zeros((M, top_k), dtype=torch.int32, device=DEV)
    topk_w = torch.ones((M, top_k), dtype=torch.float32, device=DEV)
    A = torch.from_numpy(np.full((M, hidden), _f2bf(1.0), dtype=np.uint16)).to(DEV)
    E_local = 1
    w13 = torch.from_numpy(_ones_weight_bytes(E_local, 2 * ispp, hidden // 2)).to(DEV)
    w13_scale = torch.full((E_local, 2 * ispp, hidden // 32), 127, dtype=torch.uint8, device=DEV)
    w2 = torch.from_numpy(_ones_weight_bytes(E_local, hidden, ispp // 2)).to(DEV)
    w2_scale = torch.full((E_local, hidden, ispp // 32), 127, dtype=torch.uint8, device=DEV)

    lib = ctypes.CDLL(_find_libvkernels_hip())
    moe_fn = _resolve_moe_fn(lib)

    # ---- Path A: current CPU routing (actual EM) ----
    topk_ids_flat = topk_ids.contiguous().view(-1).cpu().numpy().astype(np.int32)
    sids_a, eids_a, EM_a = _moe_align_block_size_cpu(
        topk_ids_flat, num_experts, block_size, None)
    d_sids_a = torch.from_numpy(sids_a).to(DEV)
    d_eids_a = torch.from_numpy(eids_a).to(DEV)
    print(f"[A] sids.numel={d_sids_a.numel()} eids.numel={d_eids_a.numel()} EM={EM_a}")
    out_a = run_cabi(moe_fn, A, w13, w13_scale, w2, w2_scale, topk_ids,
                     topk_w, d_sids_a, d_eids_a, EM_a, ispp, top_k, block_size, DEV)
    torch.cuda.synchronize()
    print(f"[A] OK out[0,:4]={out_a[0,:4].cpu().tolist()}")

    # ---- Path B: Option C -- GPU routing + max_EM (host constant) ----
    max_tp = topk_ids.numel() + num_experts * (block_size - 1)
    if topk_ids.numel() < num_experts:
        max_tp = min(topk_ids.numel() * block_size, max_tp)
    max_EM = round_up(max_tp, block_size)
    d_sids_b, d_eids_b, ntp = gpu_moe_align(
        topk_ids, num_experts, block_size, expert_map,
        pad_sorted_ids=True, ignore_invalid_experts=False)
    EM_b_actual = int(ntp.item())
    print(f"[B] sids.numel={d_sids_b.numel()} eids.numel={d_eids_b.numel()} "
          f"max_EM={max_EM} actual(ntp)={EM_b_actual} eids[:8]={d_eids_b[:8].cpu().tolist()}")
    if d_eids_b.numel() < max_EM // block_size:
        print(f"[B] !! eids too small: {d_eids_b.numel()} < {max_EM // block_size}; abort")
        sys.exit(2)
    out_b = run_cabi(moe_fn, A, w13, w13_scale, w2, w2_scale, topk_ids,
                     topk_w, d_sids_b, d_eids_b, max_EM, ispp, top_k, block_size, DEV)
    torch.cuda.synchronize()
    print(f"[B] OK out[0,:4]={out_b[0,:4].cpu().tolist()}")

    same = torch.allclose(out_a, out_b, atol=1e-3, rtol=1e-3)
    max_abs = (out_a - out_b).abs().max().item()
    print(f"\nRESULT: allclose(atol=1e-3,rtol=1e-3)={same}  max_abs_diff={max_abs:.2e}")
    print("PASS" if same else "FAIL")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
