#!/usr/bin/env python3
"""Focused follow-up: for each moe:vkernel_apply on the main compute stream,
show the GPU kernels + syncs NESTED inside it (ts within [apply.ts, apply.ts+dur])
and the kernel immediately before/after. Also an NCCL duration histogram.

Answers: is the RCCL all-to-all INSIDE VkernelFusedExperts.apply? Is the
hipDeviceSynchronize INSIDE apply (== the drain that serialises the pipeline)?
"""
import json
import sys
from collections import defaultdict


def humanize(us):
    if us >= 1_000_000:
        return f"{us/1_000_000:.2f}s"
    if us >= 1_000:
        return f"{us/1_000:.1f}ms"
    return f"{us:.0f}us"


def main(path):
    print(f"\n{'='*72}\nLoading {path} ...", flush=True)
    with open(path) as fh:
        data = json.load(fh)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data

    gpu = [e for e in evs if e.get("cat") == "kernel" and "dur" in e]
    st = defaultdict(float)
    for e in gpu:
        st[(e.get("pid"), e.get("tid"))] += e["dur"]
    main, _ = max(st.items(), key=lambda x: x[1]) if st else ((None, None), 0)
    main_gpu = sorted((e for e in gpu if (e.get("pid"), e.get("tid")) == main),
                      key=lambda e: e["ts"])

    annot = sorted((e for e in evs if e.get("cat") == "user_annotation"
                    and "moe:vkernel_apply" in e.get("name", "") and "dur" in e),
                   key=lambda e: e["ts"])
    print(f"  {len(evs):,} events, {len(main_gpu):,} main-stream GPU kernels, "
          f"{len(annot)} moe:vkernel_apply", flush=True)

    # --- NCCL duration histogram (main stream) ---
    nccl = [e for e in main_gpu if "nccl" in e.get("name", "").lower()]
    if nccl:
        durs = sorted(e["dur"] for e in nccl)
        print(f"\n--- NCCL (ncclDevKernel_Generic_*) on main stream: {len(nccl)} kernels, "
              f"total {humanize(sum(durs))} ---")
        buckets = [("<0.5ms", 0), ("0.5-2ms", 0), ("2-10ms", 0),
                   ("10-100ms", 0), (">100ms", 0)]
        for d in durs:
            if d < 500: buckets[0] = (buckets[0][0], buckets[0][1]+1)
            elif d < 2000: buckets[1] = (buckets[1][0], buckets[1]+1) if False else (buckets[1][0], buckets[1][1]+1)
            elif d < 10000: buckets[2] = (buckets[2][0], buckets[2][1]+1)
            elif d < 100000: buckets[3] = (buckets[3][0], buckets[3][1]+1)
            else: buckets[4] = (buckets[4][0], buckets[4][1]+1)
        for label, c in buckets:
            print(f"    {label:<10} {c}")
        print(f"    median {humanize(durs[len(durs)//2])}, p90 {humanize(durs[int(len(durs)*0.9)])}, "
              f"max {humanize(durs[-1])}")

    # --- sync events on the CPU thread that issued the main stream ---
    syncs = sorted((e for e in evs if "synchronize" in e.get("name", "").lower()
                    and "dur" in e and e.get("cat") in ("cuda_runtime", "cpu_op", "operator")),
                   key=lambda e: e["ts"])

    # --- for the first 6 applies, show nested GPU kernels + nested syncs ---
    print("\n--- First 6 moe:vkernel_apply: nested structure ---")
    import bisect
    main_ts = [e["ts"] for e in main_gpu]
    for a in annot[:6]:
        lo, hi = a["ts"], a["ts"] + a["dur"]
        i0 = bisect.bisect_left(main_ts, lo)
        i1 = bisect.bisect_left(main_ts, hi)
        nested = main_gpu[i0:i1]
        # kernel immediately before
        before = main_gpu[i0-1] if i0 > 0 else None
        after = main_gpu[i1] if i1 < len(main_gpu) else None
        print(f"\n  apply dur={humanize(a['dur'])}  ({len(nested)} GPU kern nested)")
        if before:
            print(f"    [before] {humanize(before['dur']):>8}  {before.get('name','?')[:70]}")
        nccl_n = sum(1 for e in nested if "nccl" in e.get("name", "").lower())
        sync_n = 0
        for s in syncs:
            if lo <= s["ts"] < hi:
                sync_n += 1
            elif s["ts"] >= hi:
                break
        print(f"    [inside] nccl={nccl_n}  sync={sync_n}")
        # show the dominant nested kernels (top 4 by dur)
        for e in sorted(nested, key=lambda x: -x["dur"])[:4]:
            tag = "nccl" if "nccl" in e.get("name", "").lower() else "   "
            print(f"    [inside {tag}] {humanize(e['dur']):>8}  {e.get('name','?')[:70]}")
        if after:
            print(f"    [after ] {humanize(after['dur']):>8}  {after.get('name','?')[:70]}")

    # --- aggregate: across ALL applies, how many have nccl inside / sync inside ---
    nccl_in = 0
    sync_in = 0
    for a in annot:
        lo, hi = a["ts"], a["ts"] + a["dur"]
        i0 = bisect.bisect_left(main_ts, lo)
        i1 = bisect.bisect_left(main_ts, hi)
        if any("nccl" in e.get("name", "").lower() for e in main_gpu[i0:i1]):
            nccl_in += 1
        # binary search syncs
        import bisect as b2
        sync_ts = [s["ts"] for s in syncs]
        j0 = b2.bisect_left(sync_ts, lo)
        j1 = b2.bisect_left(sync_ts, hi)
        if j1 > j0:
            sync_in += 1
    print(f"\n--- Across all {len(annot)} applies ---")
    print(f"  applies with an NCCL kernel nested inside : {nccl_in} ({100*nccl_in/len(annot):.1f}%)")
    print(f"  applies with a sync nested inside         : {sync_in} ({100*sync_in/len(annot):.1f}%)")
    print(f"  total syncs on issuing thread             : {len(syncs)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
