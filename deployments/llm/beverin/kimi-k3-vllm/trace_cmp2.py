#!/usr/bin/env python3
"""Per-thread, per-step MoE comparison: eager vs breakable (issue #46).

torch.profiler captures ALL threads, so summing `record_function` X-events
across a rank overcounts (overlapping threads). We isolate the DOMINANT
compute thread per rank (most `moe:vkernel_apply` events) and report only
its non-overlapping wall -- that is the real per-step MoE cost.

  * moe:vkernel_apply  (OUTER record_function, sitecustomize wrap)
  * moe:apply.{cpu_copy,cpu_align,gpu_copy,launch} (INNER, vkernels_experts)

Usage: trace_cmp2.py EAGER_DIR BREAKABLE_DIR [OUT_TOK] [LAYERS_PER_PP]
"""
import json, os, sys
from collections import defaultdict, Counter

SUBS = ["moe:apply.cpu_copy", "moe:apply.cpu_align",
        "moe:apply.gpu_copy", "moe:apply.launch"]


def load(path):
    with open(path) as f:
        return json.load(f)["traceEvents"]


def dominant_tid(events):
    cnt = Counter(e.get("tid") for e in events
                  if e.get("name") == "moe:vkernel_apply" and "dur" in e)
    return cnt.most_common(1)[0] if cnt else (None, 0)


def on_tid(events, tid):
    return [e for e in events if e.get("tid") == tid]


def stats(events, name):
    durs = [e["dur"] for e in events
            if e.get("name") == name and "dur" in e]
    return (sum(durs), len(durs),
            (sum(durs) / len(durs)) if durs else 0.0)


def span_us(events):
    ts = [e["ts"] for e in events if isinstance(e.get("ts"), (int, float))]
    return (max(ts) - min(ts)) if ts else 0.0


def analyze(rank_dir, out_tok=None, layers_per_pp=None):
    rows = {}
    for r in (0, 8, 16):
        p = os.path.join(rank_dir, f"step_profile_rank{r}.json")
        if not os.path.isfile(p):
            continue
        ev = load(p)
        tid, tcnt = dominant_tid(ev)
        tev = on_tid(ev, tid)
        sp = span_us(tev)
        vsum, vcnt, vmean = stats(tev, "moe:vkernel_apply")
        sub = {s: stats(tev, s) for s in SUBS}
        # decode steps inferred from moe:vkernel_apply count / layers_per_PP
        steps = (vcnt / layers_per_pp) if layers_per_pp else None
        per_step_moe = vmean * layers_per_pp if layers_per_pp else None
        rows[r] = dict(tid=tid, span=sp, vsum=vsum, vcnt=vcnt, vmean=vmean,
                       frac=(vsum / sp) if sp else 0.0, sub=sub,
                       steps=steps, per_step_moe=per_step_moe,
                       totn_threads=tcnt)
    return rows


def fmt(rows, label):
    print(f"\n================ {label} ================")
    for r, d in sorted(rows.items()):
        print(f"--- rank {r}  tid={d['tid']}  span={d['span']/1e6:5.2f}s"
              f"  moe_calls={d['vcnt']}  (~{d['steps']:.0f} steps) ---")
        print(f"  moe:vkernel_apply  sum={d['vsum']/1e3:8.1f}ms "
              f"mean={d['vmean']:7.2f}us  frac={d['frac']*100:5.1f}%"
              f"  per_step={d['per_step_moe']/1e3:5.1f}ms")
        for s in SUBS:
            ss, sc, sm = d["sub"][s]
            fm = (ss / d["vsum"] * 100) if d["vsum"] else 0.0
            print(f"    {s:24s} sum={ss/1e3:7.1f}ms n={sc:5d} "
                  f"mean={sm:6.2f}us  {fm:5.1f}% of moe")


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 2
    eager_dir, break_dir = sys.argv[1], sys.argv[2]
    out_tok = int(sys.argv[3]) if len(sys.argv) > 3 else None
    lpp = int(sys.argv[4]) if len(sys.argv) > 4 else None
    e = analyze(eager_dir, out_tok, lpp)
    b = analyze(break_dir, out_tok, lpp)
    fmt(e, f"EAGER  {os.path.basename(os.path.dirname(eager_dir))}")
    fmt(b, f"BREAKABLE  {os.path.basename(os.path.dirname(break_dir))}")
    if 0 in e and 0 in b:
        print("\n================ HEAD-TO-HEAD (rank 0, dominant compute thread) ================")
        eo, bo = e[0], b[0]
        print(f"  decode steps        ~{eo['steps']:.0f}  -> ~{bo['steps']:.0f}")
        print(f"  step wall (est ms)  {1000/( (eo['span']/1e6)/max(eo['steps'],1)) if eo['steps'] else 0:6.1f}"
              f"  -> {1000/( (bo['span']/1e6)/max(bo['steps'],1)) if bo['steps'] else 0:6.1f}"
              f"  (from total_tokens/bench throughput)")
        print(f"  vkernel_apply mean  {eo['vmean']:7.2f}us -> {bo['vmean']:7.2f}us"
              f"  ({bo['vmean']/eo['vmean']:.2f}x)")
        print(f"  per-step MoE        {eo['per_step_moe']/1e3:5.1f}ms"
              f" -> {bo['per_step_moe']/1e3:5.1f}ms"
              f"  ({bo['per_step_moe']/eo['per_step_moe']:.2f}x)")
        print(f"  moe frac of thread  {eo['frac']*100:5.1f}% -> {bo['frac']*100:5.1f}%")
        print(f"  non-moe per step    {(1-eo['frac'])*(eo['span']/1e6)/max(eo['steps'],1)*1e3:6.1f}ms"
              f" -> {(1-bo['frac'])*(bo['span']/1e6)/max(bo['steps'],1)*1e3:6.1f}ms")
        for s in SUBS:
            es, bs = e[0]["sub"][s][2], b[0]["sub"][s][2]  # means
            print(f"  {s:22s} mean {es:7.2f}us -> {bs:7.2f}us  ({bs/es:.2f}x)")


if __name__ == "__main__":
    sys.exit(main() or 0)
