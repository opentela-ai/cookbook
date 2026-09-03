"""Example bench: elementwise triad — memory-bound at large N, launch-bound at small N.

Run:  python3 microbench.py examples/elementwise_bench.py                 # bandwidth-bound demo
      ELEMENTS=4096 python3 microbench.py examples/elementwise_bench.py   # launch-overhead demo
"""
import os

import torch


def make_inputs():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = int(os.environ.get("ELEMENTS", 256 * 1024 * 1024))
    torch.manual_seed(0)
    return {
        "a": torch.randn(n, device=dev),
        "b": torch.randn(n, device=dev),
        "n": n,
    }


def op(inputs):
    inputs["a"].add_(inputs["b"], alpha=2.0)


def flops(inputs):
    return 2 * inputs["n"]


def bytes(inputs):
    return 3 * inputs["n"] * 4  # read a, read b, write a (fp32)
