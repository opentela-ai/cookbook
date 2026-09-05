"""Correct GPU-idle attribution on the MAIN compute stream only.

Builds the op sequence on the single busiest device stream, then for each
idle gap between consecutive ops, classifies by (prev_cat, next_cat) so we
can see what the GPU is waiting for. Usage: python3 gpu_idle_attr.py <trace.json>
"""
import json
import sys
from collections import defaultdict


def hum(us):
    if us >= 1e6:
        return f"{us/1e6:.2f}s"
    if us >= 1e3:
        return f"{us/1e3:.1f}ms"
    return f"{us:.1f}us"


def cat(name):
    n = name.lower()
    if "nccldev" in n or "rccl" in n or "all_reduce" in n or "all_to_all" in n \
            or "nccl" in n:
        return "rccl_all_to_all"
    if "memcpy" in n or "memset" in n or "copy" in n:
        return "memcpy"
    if "moe" in n or "fused" in n or "gemm" in n or "grouped" in n \
            or "gemm" in n or "cublas" in n or "wmma" in n:
        return "moe_gemm"
    if "attn" in n or "flash" in n or "mla" in n or "delta" in n:
        return "attn"
    return "other"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: gpu_idle_attr.py <trace.json>")
    with open(path) as f:
        data = json.load(f)
    ev = data.get("traceEvents", [])

    # device ops only (numeric pid = a device/stream)
    dev = [e for e in ev if isinstance(e.get("pid"), (int, float))
           and "dur" in e and "ts" in e
           and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_user")]
    if not dev:
        print("no device ops"); sys.exit(1)

    # pick the busiest stream by sum-duration
    by_stream = defaultdict(list)
    for e in dev:
        by_stream[(e["pid"], e.get("tid"))].append(e)
    stream = max(by_stream.values(), key=lambda s: sum(e["dur"] for e in s))
    sp = (stream[0]["pid"], stream[0].get("tid"))
    stream.sort(key=lambda e: e["ts"])

    wall_lo = stream[0]["ts"]
    wall_hi = max(e["ts"] + e["dur"] for e in stream)
    wall = wall_hi - wall_lo
    active = sum(e["dur"] for e in stream)
    idle = wall - active
    print(f"=== main stream {sp}: {len(stream)} ops ===")
    print(f"  wall       : {hum(wall)}")
    print(f"  active     : {hum(active)}  ({100*active/wall:.1f}%)")
    print(f"  idle       : {hum(idle)}  ({100*idle/wall:.1f}%)")

    # category totals on this stream
    cattot = defaultdict(float)
    for e in stream:
        cattot[cat(e["name"])] += e["dur"]
    print(f"  --- active by category ---")
    for c, t in sorted(cattot.items(), key=lambda x: -x[1]):
        print(f"  {c:18s}: {hum(t):>9s}  ({100*t/active:.1f}% busy)")

    # idle gaps classified by (prev_cat -> next_cat)
    pair = defaultdict(list)
    for a, b in zip(stream, stream[1:]):
        g_lo, g_hi = a["ts"] + a["dur"], b["ts"]
        gap = g_hi - g_lo
        if gap <= 0:
            continue
        pair[(cat(a["name"]), cat(b["name"]))].append(gap)
    print(f"\n=== idle {hum(idle)} by neighbor transition ===")
    for (p, n), gs in sorted(pair.items(), key=lambda x: -sum(x[1])):
        tot = sum(gs)
        if tot < 1000:  # <1ms total, skip noise
            continue
        print(f"  {p:14s} -> {n:14s}: n={len(gs):4d} total={hum(tot):>9s} "
              f"({100*tot/idle:5.1f}% idle)  max={hum(max(gs))}")

    # top gaps with absolute time
    print(f"\n=== top 12 idle gaps (with neighbor cats) ===")
    allg = []
    for a, b in zip(stream, stream[1:]):
        g_lo, g_hi = a["ts"] + a["dur"], b["ts"]
        gap = g_hi - g_lo
        if gap > 0:
            allg.append((gap, g_lo - wall_lo, cat(a["name"]), cat(b["name"])))
    allg.sort(reverse=True)
    for gap, t, p, n in allg[:12]:
        print(f"  gap={hum(gap):>9s} at t={hum(t):>8s}  {p} -> {n}")


if __name__ == "__main__":
    main()
