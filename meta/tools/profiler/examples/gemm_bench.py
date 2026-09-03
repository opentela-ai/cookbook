"""Example bench: dense bf16 GEMM — compute-bound shape by default.

Run:  python3 microbench.py examples/gemm_bench.py --iters 200
Env:  GEMM_M / GEMM_K / GEMM_N (default 8192: serving GEMM; use small values
      on CPU boxes), GEMM_DTYPE (bf16|fp16|fp32)
"""
import os

import torch


def make_inputs():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        os.environ.get("GEMM_DTYPE", "bf16")]
    M, K, N = (int(os.environ.get(k, d)) for k, d in (("GEMM_M", 8192), ("GEMM_K", 8192), ("GEMM_N", 8192)))
    torch.manual_seed(0)
    return {
        "a": torch.randn(M, K, device=dev, dtype=dt),
        "b": torch.randn(K, N, device=dev, dtype=dt),
        "m": M, "k": K, "n": N, "dt": dt,
    }


def op(inputs):
    return torch.matmul(inputs["a"], inputs["b"])


def flops(inputs):
    return 2 * inputs["m"] * inputs["k"] * inputs["n"]


def bytes(inputs):
    es = inputs["a"].element_size()
    return (inputs["m"] * inputs["k"] + inputs["k"] * inputs["n"] + inputs["m"] * inputs["n"]) * es
