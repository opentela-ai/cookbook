#!/usr/bin/env python3
"""compare — side-by-side two profiler result JSONs (microbench or breakdown).

The profiler's version of diff.py: before/after a kernel patch, torch vs
vendor kernel, or site vs site. Inputs are `--out` files from microbench.py /
forward_breakdown.py; tags are the discipline (see README.md).

Usage:
  python3 compare.py torch_ref.json vkernels.json
Exit 0 always (comparison, not a test); the printed verdict is the result.
"""
from __future__ import annotations

import argparse
import json
import sys

# metric key -> (label, higher_is_better)
METRICS = {
    ("time_ms", "mean"): ("mean time ms", False),
    ("time_ms", "p50"): ("p50 time ms", False),
    ("tflops",): ("TFLOP/s", True),
    ("gbps",): ("GB/s", True),
    ("util_pct", "compute_pct"): ("compute util %", True),
    ("util_pct", "mem_pct"): ("mem util %", True),
    ("launch_overhead_us",): ("launch overhead us", False),
}
BREAKDOWN = {
    ("kernels", "total_ms"): ("kernel time total ms", False),
    ("kernels", "busy_pct_of_measured_wall"): ("GPU busy %", True),
    ("gaps", "total_ms"): ("gap time total ms", False),
}


def _get(d, path):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or cur.get(p) is None:
            return None
        cur = cur[p]
    return cur


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", help="baseline JSON (--out from microbench/forward_breakdown)")
    ap.add_argument("b", help="candidate JSON")
    args = ap.parse_args()
    with open(args.a) as f:
        a = json.load(f)
    with open(args.b) as f:
        b = json.load(f)

    print(f"== A: {a.get('label') or a.get('bench')}  [{a.get('device')}]")
    print(f"== B: {b.get('label') or b.get('bench')}  [{b.get('device')}]")
    if a.get("tool") != b.get("tool"):
        print(f"warning: different tools ({a.get('tool')} vs {b.get('tool')}); comparing what matches", file=sys.stderr)
    if a.get("device") != b.get("device") and a.get("device") and b.get("device"):
        print("note: different devices — speedup is cross-machine, utilization % is the fairer column", file=sys.stderr)

    metrics = METRICS if a.get("tool") == "microbench" else dict(METRICS)
    if a.get("tool") == "forward_breakdown":
        metrics = BREAKDOWN
    any_row = False
    for path, (label, higher_better) in metrics.items():
        va, vb = _get(a, path), _get(b, path)
        if va is None or vb is None:
            continue
        any_row = True
        delta = vb - va
        speed = (vb / va) if va else float("inf")
        better = (delta > 0) if higher_better else (delta < 0)
        mark = "B better" if better else ("worse" if delta else "=")
        if not higher_better and speed != float("inf") and va:
            speed = va / vb  # for times: B/A speedup > 1 means B faster
        print(f"  {label:<22} A={va:<12.4g} B={vb:<12.4g}  x{speed:<6.3g}  {mark}")
    if not any_row:
        print("no comparable metrics found — are both files --out results from the same tool?")


if __name__ == "__main__":
    main()
