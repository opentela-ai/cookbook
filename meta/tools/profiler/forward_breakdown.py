#!/usr/bin/env python3
"""forward_breakdown — which kernel dominates a forward, and is the GPU even busy?

Phase-1/2 tool of the profiler method: run a serving-shaped forward (same
batch/seq as production — the C=1 trap applies here too, see meta/bench) under
torch.profiler, export the kineto chrome trace, and parse it into:

  * ranked RAW KERNELS by total device time (names, not aten ops),
  * a FAMILY rollup (comm / attention / moe / gemm / norm-elementwise /
    memcpy / scan-conv / other),
  * GPU busy% against the event-timed wall (low busy% = CPU dispatch / launch
    overhead / sync waits — not a kernel problem),
  * GAP analysis on the union kernel timeline (biggest holes and what runs
    before/after them — classic signatures: collective waits, host syncs).

Uses the same bench-file contract as microbench.py; run() should be the full
forward (or any region). Keep CUDA graphs OFF for this (--disable-cuda-graph):
graph replay hides per-kernel timing on several stacks.

Usage:
  python3 forward_breakdown.py my_forward_bench.py --iters 3 --top 25
  python3 forward_breakdown.py ... --json --out breakdown.json --trace keep.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

KERNEL_CATS = ("kernel", "gpu_memcpy", "gpu_memset")

# first matching family wins; order matters (comm before gemm: nccl gemm-ish names)
FAMILIES = [
    ("comm", r"nccl|rccl|allreduce|all_reduce|allgather|all_gather|reducescatter|reduce_scatter|broadcast|sendrecv|msccl"),
    ("attention", r"flash|fmha|sdpa|paged.*attn|attn|attention|mha|sparse_fwd|dsa|kpool"),
    ("moe", r"fused_moe|moe|topk|router|grouped.*gemm|expert|kpool_topk"),
    ("scan_conv", r"scan|mamba|ssm|causal_conv|conv1d"),
    ("gemm", r"gemm|matmul|cutlass|hipblaslt|cublas|wgmma|mma|sgemm|igemm|gemv"),
    ("norm_elementwise", r"norm|rms|elementwise|vectorized|relu|gelu|silu|rope|rotary|cast|copy|fill|triu|index|sort"),
    ("memcpy", r"memcpy|memset"),
]


def family_of(kernel_name):
    n = kernel_name.lower()
    for fam, pat in FAMILIES:
        import re

        if re.search(pat, n):
            return fam
    return "other"


def summarize_trace(trace_path, wall_ms_per_iter, iters):
    """Parse a kineto chrome trace into the breakdown dict. Importable (unit-tested)."""
    with open(trace_path) as f:
        events = json.load(f).get("traceEvents", [])
    kernels = [e for e in events
               if e.get("cat") in KERNEL_CATS and isinstance(e.get("dur"), (int, float))]
    if not kernels:
        return {"kernels": {"count": 0},
                "note": "no GPU kernel events in trace — CUDA unavailable, graphs hiding kernels, "
                        "or the profiled region never ran on device"}

    iv = sorted((float(e["ts"]), float(e["ts"]) + float(e["dur"]), e["name"]) for e in kernels)

    # union timeline (multi-stream safe) for busy% and gaps; we keep the
    # first-starting and last-finishing kernel names of each merged block so
    # every gap can name what runs before/after it
    merged = []  # [start_us, end_us, first_start_name, last_end_name]
    for s, e2, name in iv:
        if merged and s <= merged[-1][1]:
            if e2 > merged[-1][1]:
                merged[-1][1] = e2
                merged[-1][3] = name
        else:
            merged.append([s, e2, name, name])
    span_us = merged[-1][1] - merged[0][0]
    busy_us = sum(e2 - s for s, e2, _, _ in merged)
    gaps = [(merged[i + 1][0] - merged[i][1], merged[i][3], merged[i + 1][2])
            for i in range(len(merged) - 1)]
    gaps.sort(reverse=True)

    # per-kernel rollup
    roll = {}
    for s, e2, name in iv:
        d = roll.setdefault(name, {"count": 0, "total_us": 0.0})
        d["count"] += 1
        d["total_us"] += e2 - s
    total_us = sum(d["total_us"] for d in roll.values())

    wall_us = wall_ms_per_iter * 1e3 * iters
    fam_roll = {}
    for name, d in roll.items():
        f = fam_roll.setdefault(family_of(name), {"count": 0, "total_us": 0.0})
        f["count"] += d["count"]
        f["total_us"] += d["total_us"]

    return {
        "kernels": {
            "count": len(iv),
            "total_ms": round(total_us / 1e3, 3),
            "union_busy_ms": round(busy_us / 1e3, 3),
            "trace_span_ms": round(span_us / 1e3, 3),
            "busy_pct_of_measured_wall": round(100 * busy_us / wall_us, 1) if wall_us else None,
        },
        "families": [
            {"family": f, "total_ms": round(d["total_us"] / 1e3, 3),
             "pct_of_kernel_time": round(100 * d["total_us"] / total_us, 1), "count": d["count"]}
            for f, d in sorted(fam_roll.items(), key=lambda kv: -kv[1]["total_us"])
        ],
        "top": [
            {"name": n[:120], "family": family_of(n), "count": d["count"],
             "total_ms": round(d["total_us"] / 1e3, 3),
             "pct_of_kernel_time": round(100 * d["total_us"] / total_us, 1),
             "mean_us": round(d["total_us"] / d["count"], 1)}
            for n, d in sorted(roll.items(), key=lambda kv: -kv[1]["total_us"])
        ],
        "gaps": {
            "total_ms": round(sum(g[0] for g in gaps) / 1e3, 3),
            "count_gt_100us": sum(1 for g in gaps if g[0] > 100),
            "top": [{"dur_ms": round(g[0] / 1e3, 3), "after": g[1][:80], "before": g[2][:80]}
                    for g in gaps[:8]],
        },
    }


def run(bench_path, warmup, iters, top, trace_keep):
    bench = harness.load_bench(bench_path)
    import torch
    from torch.profiler import ProfilerActivity, profile

    dev = harness.device()
    inputs = bench.make_inputs()
    fn = lambda: bench.op(inputs)  # noqa: E731

    wall = harness.timed(fn, warmup=warmup, iters=max(5, iters))
    wall_st = wall["gpu_ms"] or wall["cpu_wall_ms"]

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    with profile(activities=acts) as prof:
        for _ in range(iters):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
    tmp = trace_keep or tempfile.mktemp(suffix=".json")
    prof.export_chrome_trace(tmp)
    out = summarize_trace(tmp, wall_st["p50"], iters)
    out.update({
        "tool": "forward_breakdown",
        "bench": os.path.abspath(bench_path),
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "iters": iters,
        "wall_ms_per_iter": {k: round(v, 4) for k, v in wall_st.items() if k != "n"},
    })
    if not trace_keep:
        os.unlink(tmp)
    else:
        out["trace"] = trace_keep
    return out


def print_table(o):
    print(f"== forward breakdown  [{o['device']}, {o['iters']} iters, wall p50 = "
          f"{o['wall_ms_per_iter']['p50']} ms/iter]")
    k = o["kernels"]
    if k.get("count", 0) == 0:
        print(f"  {k and o.get('note')}")
        return
    print(f"  kernels: {k['count']} calls, {k['total_ms']} ms total, "
          f"union busy {k['union_busy_ms']} ms, busy = {k['busy_pct_of_measured_wall']}% of measured wall")
    print("  families:")
    for f in o["families"]:
        print(f"    {f['family']:<16} {f['total_ms']:>9} ms  {f['pct_of_kernel_time']:>5}%  ({f['count']} calls)")
    print(f"  top {min(len(o['top']), 25)} kernels:")
    for t in o["top"][:25]:
        print(f"    {t['total_ms']:>9} ms {t['pct_of_kernel_time']:>5}%  n={t['count']:<5} mean={t['mean_us']:>9.1f}us  [{t['family']}] {t['name']}")
    g = o["gaps"]
    print(f"  gaps (union timeline): total {g['total_ms']} ms, {g['count_gt_100us']} gaps > 100us")
    for gap in g["top"][:5]:
        print(f"    {gap['dur_ms']:>8} ms  after: {gap['after']}  |  before: {gap['before']}")
    if k["busy_pct_of_measured_wall"] is not None and k["busy_pct_of_measured_wall"] < 60:
        print("  >> busy% LOW: the bottleneck is between kernels — CPU dispatch, launch "
              "overhead, or sync waits. Kernel microbenchmarks won't help; look at the gaps.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", help="bench file whose op() is the forward/region to break down")
    ap.add_argument("--iters", type=int, default=3, help="profiled iterations (aggregated)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--top", type=int, default=25, help="how many kernels to print")
    ap.add_argument("--trace", help="keep the chrome trace at this path (for nsys/ncu-style deep dives)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", help="also write the JSON breakdown to this path")
    args = ap.parse_args()

    out = run(args.bench, args.warmup, args.iters, args.top, args.trace)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"breakdown -> {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print_table(out)


if __name__ == "__main__":
    main()
