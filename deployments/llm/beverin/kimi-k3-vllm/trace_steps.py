#!/usr/bin/env python3
"""Derive layers/step + per-step MoE split on the PP0 gate, from the trace.

For each job: on PP0's dominant compute thread, sort moe:vkernel_apply by ts.
Within-step gaps are small (between consecutive MoE layers); step-boundary
gaps are large (rest of forward: attention+MLP+sampling, captured or eager).
Split at 5x the within-step median gap.
"""
import json, os, sys, statistics
from collections import Counter


def per_step(job, base, agg):
    ev = json.load(open(os.path.join(
        base, "run-%s" % job, "step_profiles",
        "step_profile_rank0.json")))["traceEvents"]
    xs = [e for e in ev if e.get("name") == "moe:vkernel_apply" and "dur" in e]
    tid = Counter(e.get("tid") for e in xs).most_common(1)[0][0]
    xs = sorted([e for e in xs if e.get("tid") == tid], key=lambda e: e["ts"])
    gaps = [xs[i + 1]["ts"] - (xs[i]["ts"] + xs[i]["dur"])
            for i in range(len(xs) - 1)]
    g = sorted(x for x in gaps if x > 0)
    m = statistics.median(g)
    thr = m * 5
    steps = sum(1 for x in g if x > thr) + 1
    layers = len(xs) / steps
    vm = sum(e["dur"] for e in xs) / len(xs)
    # wall per decode step from BENCHMARK throughput (C tokens/step, agg tok/s)
    D = (8.0 / agg) * 1000.0  # ms
    moe = vm * layers / 1000.0  # ms
    non = D - moe
    print(f"  job {job}  PP0: within_gap_med={m:5.0f}us  steps~{steps:4.0f}"
          f"  layers/step~{layers:4.1f}")
    print(f"    vk_apply_mean={vm:6.1f}us  moe/step={moe:6.1f}ms"
          f"  ({moe/D*100:4.0f}% of D={D:.0f}ms)  non-moe/step={non:6.1f}ms")


def main():
    base = (sys.argv[1] if len(sys.argv) > 1 else
            "/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin")
    print("EAGER  (agg 25.3 tok/s @ C=8):")
    per_step("603394", base, 25.3)
    print("BREAKABLE  (agg 35.2 tok/s @ C=8):")
    per_step("603395", base, 35.2)


if __name__ == "__main__":
    sys.exit(main() or 0)
