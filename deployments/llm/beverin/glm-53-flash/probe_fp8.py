#!/usr/bin/env python3
"""Probe the server's actual FP8 GEMM path (aiter w8a8 blockwise) vs dequant reference."""

import sys

OVL = "/capstor/scratch/cscs/xyao/glm-53-flash-beverin/overlay"
for _p in (f"{OVL}/pylib", f"{OVL}/sgl-workspace/sglang/python"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os

os.environ["SGLANG_USE_AITER"] = "1"

import torch

DEV = "cuda"


def err(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    rel = d.mean() / b.abs().mean().clamp_min(1e-9)
    return f"max_abs={d.max().item():.6f} rel_mean={rel.item():.6f}"


def main():
    print(f"torch={torch.__version__} dev={torch.cuda.get_device_name(0)}")
    from sglang.srt.layers.quantization import fp8_utils

    fn = fp8_utils.dispatch_w8a8_block_fp8_linear()
    print("dispatched fp8 backend:", getattr(fn, "__name__", fn))

    N, K, BS = 1024, 4096, 128
    for T in (2, 64, 2505):
        torch.manual_seed(3)
        x = torch.randn(T, K, device=DEV, dtype=torch.bfloat16)
        w = (torch.randn(N, K, device=DEV, dtype=torch.float32) * 0.05).to(torch.float8_e4m3fnuz)
        w_scale = torch.rand(N // BS, K // BS, device=DEV, dtype=torch.float32) * 0.02 + 0.005

        out1 = fn(input=x.clone(), weight=w, block_size=[BS, BS], weight_scale=w_scale)
        out2 = fn(input=x.clone(), weight=w, block_size=[BS, BS], weight_scale=w_scale)
        det = (out1.float() - out2.float()).abs().max().item()

        # reference: dequant weight (exact fp8->f32 x per-block scale), fp32 matmul
        w_deq = w.float() * w_scale.repeat_interleave(BS, 0).repeat_interleave(BS, 1)
        ref = x.float() @ w_deq.T
        print(f"fp8_gemm T={T}: {err(out1, ref)} | run1-vs-run2 det={det:.2e}"
              f"{'  *** NONDET ***' if det > 1e-3 else ''}")



if __name__ == "__main__":
    main()
