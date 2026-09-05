"""Analyze the per-phase MoE CPU cost from a torch.profiler Chrome trace.

Usage: python3 trace_apply_phases.py <step_profile_rankN.json>

Extracts the four moe:apply.* user_annotation regions added to
VkernelFusedExperts.apply() and attributes:
  moe:apply.cpu_copy   GPU->CPU sync of topk_ids (waits for the dispatch
                       all-to-all) + host copy
  moe:apply.cpu_align  pure-Python _moe_align_block_size_cpu routing
  moe:apply.gpu_copy   sids/eids host->device
  moe:apply.launch     the ctypes C call (issues kernels on stream)

For cpu_copy (the suspected bottleneck: it synchronises on the all-to-all
that produced topk_ids), also reports the GPU-side cudaMemcpy /
synchronize ops that fall inside its window, so we can tell WAIT from
HOST-WORK.
"""
import json
import statistics
import sys


def humanize(us):
    if us >= 1e6:
        return f"{us/1e6:.2f}s"
    if us >= 1e3:
        return f"{us/1e3:.1f}ms"
    return f"{us:.1f}us"


def stats(name, ds):
    if not ds:
        print(f"  {name:18s}: (none)")
        return 0.0
    ds = sorted(ds)
    n = len(ds)
    tot = sum(ds)
    print(f"  {name:18s}: n={n} total={humanize(tot)} avg={humanize(tot/n)} "
          f"med={humanize(ds[n//2])} max={humanize(ds[-1])}")
    return tot


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: trace_apply_phases.py <trace.json>")
    with open(path) as f:
        data = json.load(f)
    events = data.get("traceEvents", [])

    phases = ["moe:apply.cpu_copy", "moe:apply.cpu_align",
              "moe:apply.gpu_copy", "moe:apply.launch"]
    by_phase = {p: [] for p in phases}
    # (ts, dur) per phase for overlap analysis
    windows = {p: [] for p in phases}
    for e in events:
        n = e.get("name", "")
        if e.get("cat") == "user_annotation" and "dur" in e and n in by_phase:
            by_phase[n].append(e["dur"])
            windows[n].append((e["ts"], e["ts"] + e["dur"]))

    # total moe:vkernel_apply for the % breakdown
    apply = [e for e in events if e.get("cat") == "user_annotation"
             and "moe:vkernel_apply" in e.get("name", "") and "dur" in e]
    apply_total = sum(e["dur"] for e in apply) if apply else 0.0

    print(f"=== per-phase MoE CPU cost (apply_total={humanize(apply_total)}, "
          f"n_apply={len(apply)}) ===")
    grand = 0.0
    for p in phases:
        t = stats(p, by_phase[p])
        grand += t
        if apply_total:
            print(f"  {p:18s}   {100*t/apply_total:5.1f}% of apply_total")
    print(f"  {'SUM(phases)':18s}: {humanize(grand)}  "
          f"({100*grand/apply_total:.1f}% of apply_total)" if apply_total else "")
    # uncovered = apply_total - sum(phases): launch is async (returns fast),
    # so the gap is mostly the time between the last phase end and apply end
    # = waiting for the GPU? No -- apply ends right after output.copy_.
    uncovered = apply_total - grand
    if uncovered > 0:
        print(f"  {'uncovered':18s}: {humanize(uncovered)}  "
              f"(output.copy_ fp32->bf16 + marshalling)")

    # --- cpu_copy attribution: GPU ops overlapping its window ----------------
    # Distinguish WAIT (cpu blocked in sync while GPU runs the dispatch
    # all-to-all) from HOST-WORK (the actual copy / numpy conversion).
    cw = windows["moe:apply.cpu_copy"]
    if cw:
        gpu_ops = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy")
                   and "dur" in e]
        overlap_by_cat = {}
        sync_in_window = 0
        for (lo, hi) in cw:
            for g in gpu_ops:
                g_lo, g_hi = g["ts"], g["ts"] + g["dur"]
                if g_lo < hi and g_hi > lo:  # overlaps
                    c = "rccl_all_to_all" if "ncclDev" in g.get("name", "") \
                        else ("memcpy" if "memcpy" in g.get("name", "").lower()
                              else "other")
                    overlap_by_cat[c] = overlap_by_cat.get(c, 0) + min(g_hi, hi) - max(g_lo, lo)
            # count host sync ops inside the window (cudaStreamSynchronize etc.)
            for s in events:
                if s.get("cat") == "cpu_op" and "synchronize" in s.get("name", "").lower() \
                        and "dur" in s and lo <= s["ts"] < hi:
                    sync_in_window += s["dur"]
        print(f"\n--- cpu_copy attribution (GPU busy overlapping the {len(cw)} windows) ---")
        for c, t in sorted(overlap_by_cat.items(), key=lambda x: -x[1]):
            print(f"  GPU {c:16s} overlap: {humanize(t)}  "
                  f"({100*t/sum(w[1]-w[0] for w in cw):.1f}% of cpu_copy wall)")
        print(f"  host sync ops (cpu) inside cpu_copy: {humanize(sync_in_window)}")


if __name__ == "__main__":
    main()
