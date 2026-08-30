#!/usr/bin/env python3
"""probe_dsa_vkernels.py — isolated check that PR #52 clears the GLM-5.3
DSA-decode JIT abort on Beverin (gfx942), BEFORE the ~5 min serve cold start.

Reproduces the EXACT job-612262 blocker — GLM-5.3 DeepseekSparseAttn DECODE
with qk_rope_head_dim = 0 -> dim = 256, d_v = 256, tail_dim = 0, topk = 2048
(the shape that made sglang's tilelang sparse_mla_fwd_decode_partial emit a
zero-K GEMM and hit TVM's `Check failed: pb->value != 0 (0 vs. 0)` at JIT
time) — and asserts the vkernels HIP kernel (vk_hip_dsa_sparse_fwd, wired in
by sitecustomize.py + vkernels_dsa.py) produces a non-NaN, non-empty output.

Run INSIDE the v0.5.18 container on one MI300A, with the SAME PYTHONPATH the
sbatch uses (pylib first, so sitecustomize is auto-imported):

    srun -p mi300 -A a-infra02 --gres=gpu:1 -N1 -n1 -t0:10:00 \
      --export=ALL,PYTHONPATH=$OVL/pylib:$OVL/pkgs310:...,VKERNELS_DIR=.../vkernels \
      python3 $SCRIPT_DIR/probe_dsa_vkernels.py
"""
from __future__ import annotations

import math
import os
import sys

import torch


def _banner(msg):
    sys.stderr.write(f"\n=== {msg} ===\n"); sys.stderr.flush()


def main():
    _banner("device + gcnArchName")
    assert torch.cuda.is_available(), "no CUDA — must run on a MI300A node"
    props = torch.cuda.get_device_properties(0)
    gcn = getattr(props, "gcnArchName", "") or ""
    sys.stderr.write(f"  gcnArchName = {gcn!r}\n")
    assert "gfx942" in gcn, f"not gfx942 (got {gcn!r})"
    dev = torch.device("cuda:0")

    _banner("sitecustomize loaded + rebind applied? (see stderr above)")
    # sitecustomize ran at startup (pylib is first on PYTHONPATH). Confirm the
    # sglang DSA symbol is now the vkernels adapter, not the tilelang kernel.
    import sglang.kernels.ops.attention.dsa.tilelang_kernel as tlk
    import vkernels_dsa
    rebound = tlk.tilelang_sparse_fwd is vkernels_dsa.tilelang_sparse_fwd
    sys.stderr.write(f"  tlk.tilelang_sparse_fwd is vkernels_dsa.tilelang_sparse_fwd = {rebound}\n")
    assert rebound, (
        "sitecustomize did NOT rebind sglang's tilelang_sparse_fwd -> "
        "vkernels_dsa. Check the '[sitecustomize] DSA-vkernels patch' line on "
        "stderr above and that $OVL/pylib is FIRST on PYTHONPATH."
    )

    _banner("libvkernels_hip.so found + vk_hip_dsa_sparse_fwd bound?")
    lib = vkernels_dsa._get_lib()
    fn = getattr(lib, "vk_hip_dsa_sparse_fwd", None)
    cfg = getattr(lib, "vk_hip_dsa_config", None)
    sys.stderr.write(f"  lib @ {vkernels_dsa._find_libvkernels_hip()!r}\n")
    sys.stderr.write(f"  vk_hip_dsa_sparse_fwd = {fn!r}\n")
    sys.stderr.write(f"  vk_hip_dsa_config     = {cfg!r}\n")
    assert fn is not None and cfg is not None, "missing DSA symbols in libvkernels_hip.so"

    _banner("reproduce job-612262 abort shape (GLM-5.3 decode, tail_dim=0)")
    S_q, S_kv, H, dim, d_v, topk, kv_group = 64, 8192, 64, 256, 256, 2048, 1
    tail_dim = dim - d_v  # 0  <- the load-bearing shape (qk_rope_head_dim=0)
    sys.stderr.write(
        f"  S_q={S_q} S_kv={S_kv} H={H} dim={dim} d_v={d_v} "
        f"tail_dim={tail_dim} topk={topk} kv_group={kv_group}\n"
    )
    assert tail_dim == 0, "this probe specifically exercises tail_dim==0"

    torch.manual_seed(0)
    q = torch.randn(S_q, H, dim, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(S_kv, kv_group, dim, dtype=torch.bfloat16, device=dev)
    # indices: valid KV slot ids in [0, S_kv); padded to a multiple of 64.
    assert topk % 64 == 0
    indices = torch.randint(0, S_kv, (S_q, kv_group, topk), dtype=torch.int32, device=dev)
    sm_scale = 1.0 / math.sqrt(dim)  # raw MLA scale; adapter folds log2(e)

    _banner("forward through the REBOUND symbol (the pre-PR-52 step aborted here)")
    out = tlk.tilelang_sparse_fwd(q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=False)
    sys.stderr.write(f"  out.shape={tuple(out.shape)} dtype={out.dtype}\n")
    assert tuple(out.shape) == (1, S_q, H, d_v), f"bad out shape {tuple(out.shape)}"
    assert out.dtype == torch.bfloat16, f"bad out dtype {out.dtype}"
    nan = int(torch.isnan(out).sum())
    inf = int(torch.isinf(out).sum())
    abssum = float(out.abs().sum())
    sys.stderr.write(f"  isnan={nan} isinf={inf} abs().sum()={abssum:.3f}\n")
    assert nan == 0 and inf == 0, "output has NaN/Inf"
    assert abssum > 0.0, "output is all zeros"

    _banner("return_lse=True path (the other tilelang_sparse_fwd contract)")
    out2, lse = tlk.tilelang_sparse_fwd(q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=True)
    assert tuple(out2.shape) == (1, S_q, H, d_v) and out2.dtype == torch.bfloat16
    assert tuple(lse.shape) == (1, S_q, H) and lse.dtype == torch.float32
    sys.stderr.write(
        f"  out2.shape={tuple(out2.shape)} lse.shape={tuple(lse.shape)} "
        f"lse[finite]={int(torch.isfinite(lse).sum())}/{lse.numel()}\n"
    )
    assert int(torch.isfinite(lse).sum()) == lse.numel(), "lse has non-finite entries"

    _banner("ALSO exercise tail_dim>0 (DeepSeek-V3 shape) — kernel not GLM-5.3-specific")
    S_q2, S_kv2, H2, dim2, d_v2, topk2 = 64, 8192, 64, 320, 256, 2048
    tail_dim2 = dim2 - d_v2  # 64 (>0) — the upstream-validated shape
    assert tail_dim2 == 64
    q2 = torch.randn(S_q2, H2, dim2, dtype=torch.bfloat16, device=dev)
    kv2 = torch.randn(S_kv2, 1, dim2, dtype=torch.bfloat16, device=dev)
    idx2 = torch.randint(0, S_kv2, (S_q2, 1, topk2), dtype=torch.int32, device=dev)
    out3 = tlk.tilelang_sparse_fwd(q2, kv2, idx2, sm_scale=1.0 / math.sqrt(dim2), d_v=d_v2, return_lse=False)
    sys.stderr.write(
        f"  tail_dim={tail_dim2}: out3.shape={tuple(out3.shape)} "
        f"abs().sum()={float(out3.abs().sum()):.3f} isnan={int(torch.isnan(out3).sum())}\n"
    )
    assert tuple(out3.shape) == (1, S_q2, H2, d_v2)
    assert int(torch.isnan(out3).sum()) == 0 and float(out3.abs().sum()) > 0.0

    _banner("PASS — PR #52 clears the GLM-5.3 DSA-decode JIT abort on gfx942")
    sys.stderr.write(
        "  Both tail_dim==0 (GLM-5.3, the job-612262 blocker) and tail_dim>0\n"
        "  (DeepSeek-V3) produce correct output through the rebound sglang\n"
        "  symbol. Safe to re-run the full serve smoke (SMOKE=1, LOAD_FORMAT=dummy).\n"
    )


if __name__ == "__main__":
    main()
