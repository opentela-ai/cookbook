#!/usr/bin/env python3
"""Analyze a torch.profiler Chrome trace from the Kimi-K3 step profiler.

Produces a per-step efficiency breakdown so we can answer "where is the
bottleneck": MLA attention vs MoE expert dispatch (VkernelFusedExperts.apply,
which we suspect calls torch.cuda.synchronize()) vs RCCL all-to-all vs the
PP=3 bubble vs CPU launch overhead / idle.

Usage:
  python3 trace_analyze.py <trace.json> [<trace.json> ...]

Each trace is ~230 MB (15 s capture, 3 PP stages * TP=8 GPUs, with stacks +
shapes). We load with json.load (~2 GB RAM on a 128 GB node) and aggregate.
"""
import json
import sys
import re
from collections import defaultdict


# --- categorisation by kernel name substring --------------------------------
def categorize(name: str) -> str:
    n = name.lower()
    if "synchronize" in n or n in ("cudastreamsynchronize", "cudadevicesynchronize",
                                    "aten::synchronize", "cuda_event_record",
                                    "cuda_event_synchronize"):
        return "sync"
    if "nccl" in n or "rccl" in n or "all_reduce" in n or "all_to_all" in n \
       or "allgather" in n or "reducescatter" in n or "all-to-all" in n:
        return "rccl_all_to_all"
    if "gloo" in n or "broadcast" in n or "send" in n and "recv" in n:
        return "pp_send_recv"
    if "recv" in n or "_recv" in n or "send" in n:
        return "pp_send_recv"
    if "mla" in n or "forward_mqa" in n or "vk_hip_mla" in n or "triton_mla" in n \
       or "fwd_mla" in n or "_mla_" in n or "mla_" in n or "_mla" in n:
        return "mla_attention"
    if "expert" in n or "fused_expert" in n or "vk_hip_moe" in n or "_moe_" in n \
       or "moe_" in n or "_moe" in n or "topk" in n or "gate" in n \
       or "grouped" in n or "softmax_topk" in n:
        return "moe_expert"
    if "rmsnorm" in n or "layernorm" in n or "rms_norm" in n:
        return "rmsnorm"
    if "rotary" in n or "rope" in n:
        return "rotary"
    if "memcpy" in n or "copy" in n and "kernel" not in n:
        return "memcpy"
    if "elementwise" in n or "binary" in n or "unary" in n or "activation" in n \
       or "silu" in n or "gelu" in n or "relu" in n:
        return "elementwise"
    if "gemm" in n or "matmul" in n or "mm" in n and "softmax" not in n:
        return "gemm"
    if "triton" in n and "mla" not in n and "moe" not in n:
        return "triton_other"
    return "other"


def humanize(us: float) -> str:
    if us >= 1_000_000:
        return f"{us/1_000_000:.2f}s"
    if us >= 1_000:
        return f"{us/1_000:.1f}ms"
    return f"{us:.0f}us"


def analyze(path: str) -> dict:
    print(f"\n{'='*72}\nLoading {path} ...", flush=True)
    with open(path) as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    print(f"  {len(events):,} events total", flush=True)

    # Separate by category (the profiler 'cat' field, not our categorize()).
    by_cat = defaultdict(int)
    for e in events:
        by_cat[e.get("cat", "?")] += 1
    print("  by profiler cat: " + ", ".join(f"{c}={n}" for c, n in
          sorted(by_cat.items(), key=lambda x: -x[1])[:12]), flush=True)

    # --- GPU kernels (cat == "kernel") -------------------------------------
    gpu = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
    gpu_total = sum(e["dur"] for e in gpu)
    # Busiest (pid,tid) = the main compute stream (most kernel time).
    stream_time = defaultdict(float)
    for e in gpu:
        stream_time[(e.get("pid"), e.get("tid"))] += e["dur"]
    main_stream, main_busy = max(stream_time.items(), key=lambda x: x[1]) \
        if stream_time else ((None, None), 0)
    main_gpu = [e for e in gpu if (e.get("pid"), e.get("tid")) == main_stream]
    main_busy_sum = sum(e["dur"] for e in main_gpu)

    # Time range of the main stream.
    if main_gpu:
        t_lo = min(e["ts"] for e in main_gpu)
        t_hi = max(e["ts"] + e.get("dur", 0) for e in main_gpu)
        wall = t_hi - t_lo
    else:
        t_lo = t_hi = wall = 0

    # Categorise GPU time on the MAIN stream (avoids double-counting memcpy
    # streams etc.) by kernel name.
    cat_time = defaultdict(float)
    cat_count = defaultdict(int)
    name_time = defaultdict(float)
    for e in main_gpu:
        c = categorize(e.get("name", ""))
        cat_time[c] += e["dur"]
        cat_count[c] += 1
        name_time[e.get("name", "?")] += e["dur"]

    print(f"\n--- MAIN COMPUTE STREAM (pid={main_stream[0]} tid={main_stream[1]}) ---")
    print(f"  wall span       : {humanize(wall)}")
    print(f"  GPU busy (sum)  : {humanize(main_busy_sum)}  ({100*main_busy_sum/wall:.1f}% of wall)"
          if wall else "  GPU busy: 0")
    idle = wall - main_busy_sum
    print(f"  GPU idle/bubble : {humanize(idle)}  ({100*idle/wall:.1f}% of wall)"
          if wall else "")
    print(f"  kernel count    : {len(main_gpu):,}")

    print("\n  GPU time by category (main stream):")
    for c, t in sorted(cat_time.items(), key=lambda x: -x[1]):
        pct = 100 * t / main_busy_sum if main_busy_sum else 0
        wpct = 100 * t / wall if wall else 0
        print(f"    {c:<16} {humanize(t):>9}  {pct:5.1f}% of busy | {wpct:5.1f}% of wall"
              f"  ({cat_count[c]} kern)")

    print("\n  Top 20 individual GPU kernels (main stream):")
    for nm, t in sorted(name_time.items(), key=lambda x: -x[1])[:20]:
        print(f"    {humanize(t):>9}  {nm[:90]}")

    # --- the moe:vkernel_apply annotation (user_annotation) -----------------
    annot = [e for e in events if e.get("cat") == "user_annotation"
             and "moe:vkernel_apply" in e.get("name", "") and "dur" in e]
    if annot:
        at = [a["ts"] for a in annot]
        ad = [a["dur"] for a in annot]
        annot_total = sum(ad)
        annot_span = max(at) - min(at) if len(at) > 1 else 0
        print("\n--- moe:vkernel_apply (the wrapped VkernelFusedExperts.apply) ---")
        print(f"  count           : {len(annot)}  (~{len(annot)} MoE apply calls)")
        print(f"  total wall      : {humanize(annot_total)}")
        print(f"  avg per call    : {humanize(annot_total/len(annot))}")
        print(f"  median          : {humanize(sorted(ad)[len(ad)//2])}")
        print(f"  max             : {humanize(max(ad))}")
        print(f"  span            : {humanize(annot_span)}  ({100*annot_total/annot_span:.1f}% of span)"
              if annot_span else "")
        # gaps between consecutive apply calls (= bubble between MoE + next op)
        if len(at) > 2:
            gaps = [at[i+1] - (at[i] + ad[i]) for i in range(len(at)-1)
                    if at[i+1] - (at[i] + ad[i]) > 0]
            if gaps:
                gs = sorted(gaps)
                print(f"  inter-call gap  : n={len(gaps)} total={humanize(sum(gaps))} "
                      f"median={humanize(gs[len(gs)//2])} max={humanize(max(gaps))}")

    # --- explicit synchronize events ---------------------------------------
    syncs = [e for e in events if "synchronize" in e.get("name", "").lower()
             and "dur" in e]
    # also gpu_sync cat
    gs_cat = [e for e in events if e.get("cat") == "gpu_sync" and "dur" in e]
    if syncs or gs_cat:
        print("\n--- synchronize events ---")
        if gs_cat:
            print(f"  gpu_sync cat: {len(gs_cat)} events, total dur {humanize(sum(e['dur'] for e in gs_cat))}")
        if syncs:
            byn = defaultdict(lambda: [0, 0.0])
            for e in syncs:
                byn[e.get("name", "?")][0] += 1
                byn[e.get("name", "?")][1] += e["dur"]
            for nm, (c, t) in sorted(byn.items(), key=lambda x: -x[1][1]):
                print(f"  {nm:<40} n={c:<5} total={humanize(t)}")

    # --- per-step grouping (by moe:vkernel_apply) --------------------------
    if annot and len(annot) > 3:
        annot.sort(key=lambda a: a["ts"])
        # per step: GPU kernel time on main stream falling inside [ts, ts+dur]
        ai = [(a["ts"], a["ts"] + a["dur"]) for a in annot]
        step_gpu = [0.0] * len(ai)
        step_moe = [0.0] * len(ai)
        step_rccl = [0.0] * len(ai)
        for e in main_gpu:
            ets = e["ts"]
            for i, (lo, hi) in enumerate(ai):
                if lo <= ets < hi:
                    step_gpu[i] += e["dur"]
                    c = categorize(e.get("name", ""))
                    if c == "moe_expert":
                        step_moe[i] += e["dur"]
                    elif c == "rccl_all_to_all":
                        step_rccl[i] += e["dur"]
                    break
        print(f"\n--- per-step (grouped by moe:vkernel_apply, {len(ai)} steps) ---")
        print(f"  step | apply_wall | gpu_in_apply | moe_in_apply | rccl_in_apply")
        for i in range(min(8, len(ai))):
            print(f"  {i:>4} | {humanize(ai[i][1]-ai[i][0]):>10} | "
                  f"{humanize(step_gpu[i]):>12} | {humanize(step_moe[i]):>12} | "
                  f"{humanize(step_rccl[i]):>12}")
        if len(ai) > 8:
            avg_gpu = sum(step_gpu)/len(step_gpu)
            avg_moe = sum(step_moe)/len(step_moe)
            avg_rccl = sum(step_rccl)/len(step_rccl)
            print(f"  avg  | {humanize(sum(a[1]-a[0] for a in ai)/len(ai)):>10} | "
                  f"{humanize(avg_gpu):>12} | {humanize(avg_moe):>12} | "
                  f"{humanize(avg_rccl):>12}")
    return {"wall": wall, "main_busy": main_busy_sum, "cat_time": dict(cat_time),
            "annot_count": len(annot)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    for p in sys.argv[1:]:
        analyze(p)
