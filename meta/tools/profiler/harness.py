"""harness — shared bench-file contract and timing primitives for the profiler.

A BENCH FILE is a plain python file exposing:

    def make_inputs():          ...   # build tensors/config once
    def op(inputs):             ...   # the kernel/region to time (may loop internally)
    def flops(inputs) -> float  ...   # optional: nominal math ops per op() call
    def bytes(inputs) -> float  ...   # optional: nominal DRAM traffic per op() call

Everything in this toolkit (microbench.py, forward_breakdown.py) consumes that
contract; examples live in examples/. torch is imported lazily so the
stdlib-only operator tools stay install-free.
"""
from __future__ import annotations

import importlib.util
import statistics
import time


def load_bench(path):
    """Import a bench file by path; fail loudly with the contract."""
    spec = importlib.util.spec_from_file_location("dbg_bench", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load bench file {path!r}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for required in ("make_inputs", "op"):
        if not hasattr(mod, required):
            raise AttributeError(f"bench file {path!r} missing required def {required}(inputs) — "
                                 "see meta/tools/profiler/README.md for the contract")
    return mod


def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timed(fn, warmup=20, iters=100):
    """Time fn() iters times after warmup.

    Returns dict with GPU-event timing stats when CUDA is available (elapsed
    via cuda events, ms), else CPU wall (perf_counter, ms). Always includes
    cpu_wall_ms stats — a large cpu_wall vs gpu time means dispatch overhead.
    """
    import torch

    use_cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
    gpu_ms, cpu_ms = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        if use_cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            gpu_ms.append(s.elapsed_time(e))
        else:
            fn()
        cpu_ms.append((time.perf_counter() - t0) * 1e3)
    return {"gpu_ms": _stats(gpu_ms) if gpu_ms else None, "cpu_wall_ms": _stats(cpu_ms)}


def _stats(xs):
    xs = sorted(xs)
    n = len(xs)
    return {
        "n": n,
        "mean": statistics.fmean(xs),
        "p50": xs[n // 2],
        "min": xs[0],
        "max": xs[-1],
        "std": statistics.pstdev(xs) if n > 1 else 0.0,
    }


def launch_overhead_us(dev):
    """Per-kernel-launch overhead: time a minimal kernel back-to-back.

    A microbench result within ~2-3x of this number means the measurement is
    launch-bound, not kernel-bound (fuse kernels or batch the work).
    """
    import torch

    x = torch.zeros(16, device=dev)
    r = timed(lambda: x.fill_(0.0), warmup=50, iters=500)
    st = r["gpu_ms"] or r["cpu_wall_ms"]
    return st["mean"] * 1e3  # ms -> us
