#!/usr/bin/env python3
"""microbench — time ONE kernel/region in isolation and roofline-classify it.

Phase-3 tool of the profiler method: once forward_breakdown.py has named a
dominant kernel, microbench it ALONE (steady state, GPU-event timed) and get
achieved TFLOP/s / GB/s against the nominal device peak — i.e. how close to
100% the kernel is, and whether it is memory-bound, compute-bound, or
launch-bound. Consumption contract: a bench file (see README.md / examples/).

Usage:
  python3 microbench.py examples/gemm_bench.py --iters 200
  python3 microbench.py my_kernel_bench.py --peak-bw 1228 --peak-flops 590 --json --out res.json

Bound classification (only as good as the bytes()/flops() models):
  launch-bound   mean time within ~3x of a minimal-kernel launch (fuse it)
  memory-bound   achieved GB/s is the limiting utilization (check dtype,
                 layout/coalescing, packaging — e.g. recompute vs re-read)
  compute-bound  achieved TFLOP/s is the limiting utilization (check tile
                 shape, occupancy, dtype path)
Utilization is against NOMINAL peaks (device_info.py) — verify or override.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from device_info import lookup  # noqa: E402


def run(bench_path, warmup, iters, peak_flops, peak_bw, label):
    bench = harness.load_bench(bench_path)
    import torch

    dev = harness.device()
    inputs = bench.make_inputs()
    fn = lambda: bench.op(inputs)  # noqa: E731

    r = harness.timed(fn, warmup=warmup, iters=iters)
    st = r["gpu_ms"] or r["cpu_wall_ms"]
    mean_us = st["mean"] * 1e3

    overhead_us = harness.launch_overhead_us(dev) if dev.type == "cuda" else None

    flops = getattr(bench, "flops", None)
    nbytes = getattr(bench, "bytes", None)
    fl = float(flops(inputs)) if flops else None
    by = float(nbytes(inputs)) if nbytes else None

    have_gpu = r["gpu_ms"] is not None  # roofline math is only meaningful for
    # GPU-event timing: CPU wall against device peaks measures the wrong thing
    if have_gpu and dev.type == "cuda":
        key, (nom_bw, nom_fl) = lookup(torch.cuda.get_device_name(0))
    else:
        key, (nom_bw, nom_fl) = None, (None, None)
    bw = peak_bw or nom_bw if have_gpu else None
    flpk = peak_flops or nom_fl if have_gpu else None

    tflops = fl / st["mean"] / 1e9 if fl and have_gpu else None   # flops/ms -> TFLOP/s
    gbps = by / st["mean"] / 1e6 if by and have_gpu else None     # bytes/ms -> GB/s
    util = {
        "compute_pct": round(100 * tflops / flpk, 1) if tflops and flpk else None,
        "mem_pct": round(100 * gbps / bw, 1) if gbps and bw else None,
    }
    if not have_gpu:
        bound = "cpu-timing (roofline needs a CUDA device; times are CPU wall)"
    elif overhead_us and mean_us <= 3 * overhead_us:
        bound = "launch-bound"
    elif util["mem_pct"] and util["compute_pct"]:
        bound = "memory-bound" if util["mem_pct"] >= util["compute_pct"] else "compute-bound"
    elif util["mem_pct"]:
        bound = "memory-bound"
    elif util["compute_pct"]:
        bound = "compute-bound"
    else:
        bound = "unknown (no peaks; pass --peak-bw/--peak-flops or add bytes()/flops())"

    out = {
        "tool": "microbench",
        "label": label or os.path.basename(bench_path),
        "bench": os.path.abspath(bench_path),
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "timed_by": "cuda_events" if r["gpu_ms"] else "cpu_wall",
        "iters": iters,
        "warmup": warmup,
        "time_ms": st,
        "cpu_wall_ms": r["cpu_wall_ms"],
        "launch_overhead_us": round(overhead_us, 2) if overhead_us else None,
        "flops_per_iter": fl,
        "bytes_per_iter": by,
        "tflops": round(tflops, 2) if tflops else None,
        "gbps": round(gbps, 1) if gbps else None,
        "peak_tflops": flpk,
        "peak_gbps": bw,
        "peak_row": key,
        "util_pct": util,
        "bound": bound,
    }
    return out


def print_table(o):
    print(f"== {o['label']}  [{o['device']}, timed by {o['timed_by']}, n={o['iters']}]")
    t = o["time_ms"]
    print(f"  time ms     mean={t['mean']:.4g}  p50={t['p50']:.4g}  min={t['min']:.4g}  max={t['max']:.4g}  std={t['std']:.4g}")
    cw = o["cpu_wall_ms"]["mean"]
    print(f"  cpu wall ms mean={cw:.4g}" + ("  (>> gpu time -> python/dispatch overhead)" if o["timed_by"] == "cuda_events" and cw > 2 * t["mean"] else ""))
    if o["launch_overhead_us"]:
        print(f"  launch overhead ~{o['launch_overhead_us']} us")
    if o.get("tflops") is not None:
        print(f"  {o['tflops']:g} TFLOP/s  ({o['util_pct']['compute_pct']}% of {o['peak_tflops']} nominal)")
    if o.get("gbps") is not None:
        print(f"  {o['gbps']:g} GB/s  ({o['util_pct']['mem_pct']}% of {o['peak_gbps']} nominal)")
    print(f"  BOUND: {o['bound']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", help="bench file path (contract in README.md)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--peak-flops", type=float, help="nominal peak TFLOP/s (overrides device table)")
    ap.add_argument("--peak-bw", type=float, help="nominal peak GB/s (overrides device table)")
    ap.add_argument("--label", help="label for the result (default: bench filename)")
    ap.add_argument("--json", action="store_true", help="print pure JSON (agent-parseable)")
    ap.add_argument("--out", help="also write the JSON result to this path (for compare.py)")
    args = ap.parse_args()

    result = run(args.bench, args.warmup, args.iters, args.peak_flops, args.peak_bw, args.label)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"result -> {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_table(result)


if __name__ == "__main__":
    main()
