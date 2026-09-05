"""Top kernel names (non-nccl) on the main compute stream, with the idle
that PRECEDES each — to see if a fusible/avoidable CPU-launched category
is behind the steady-state gaps. Usage: python3 topk_other.py <trace.json>
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


def main():
    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)
    ev = data.get("traceEvents", [])
    dev = [e for e in ev if isinstance(e.get("pid"), (int, float))
           and "dur" in e and "ts" in e
           and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_user")]
    by_stream = defaultdict(list)
    for e in dev:
        by_stream[(e["pid"], e.get("tid"))].append(e)
    stream = max(by_stream.values(), key=lambda s: sum(e["dur"] for e in s))
    stream.sort(key=lambda e: e["ts"])

    # idle preceding each op
    name_tot = defaultdict(float)
    name_n = defaultdict(int)
    name_idle = defaultdict(float)
    for a, b in zip(stream, stream[1:]):
        gap = b["ts"] - (a["ts"] + a["dur"])
        na = a["name"]
        if gap > 0:
            name_idle[na] += gap
        name_tot[na] += a["dur"]
        name_n[na] += 1
    # also count nccl separately
    nccl = defaultdict(float)
    for e in stream:
        if "nccl" in e["name"].lower() or "rccl" in e["name"].lower():
            nccl[e["name"]] += e["dur"]

    print("=== top 20 non-nccl kernels by total time ===")
    items = [(n, t) for n, t in name_tot.items()
             if "nccl" not in n.lower() and "rccl" not in n.lower()]
    items.sort(key=lambda x: -x[1])
    for n, t in items[:20]:
        print(f"  {hum(t):>9s} n={name_n[n]:6d} idle_before={hum(name_idle[n]):>9s}  {n[:90]}")

    print("\n=== top 20 non-nccl kernels by PRECEDING idle (CPU launch bubble) ===")
    items = [(n, name_idle[n]) for n in name_idle
             if "nccl" not in n.lower() and "rccl" not in n.lower()]
    items.sort(key=lambda x: -x[1])
    for n, i in items[:20]:
        print(f"  idle={hum(i):>9s} n={name_n[n]:6d} kern={hum(name_tot[n]):>9s}  {n[:90]}")


if __name__ == "__main__":
    main()
