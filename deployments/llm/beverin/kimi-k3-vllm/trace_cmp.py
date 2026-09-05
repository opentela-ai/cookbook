#!/usr/bin/env python3
"""Compare eager vs breakable per-step MoE profiles (issue #46 follow-up).

Reads torch.profiler chrome traces (step_profile_rank{0,8,16}.json) from two
runs and prints, per rank:
  * the OUTER moe:vkernel_apply region (full apply) -- sum, count, mean, frac
  * the INNER moe:apply.{cpu_copy,cpu_align,gpu_copy,launch} sub-regions
  * cudaGraphLaunch count + the captured-majority estimate
  * per-step MoE (mean_apply * layers_per_PP, using the vkernel_apply count /
    decode_steps heuristic)

Usage: trace_cmp.py EAGER_DIR BREAKABLE_DIR [OUT_TOK] [LAYERS_PER_PP]
E.g.    trace_cmp.py run-603394/step_profiles run-603395/step_profiles 256 20
"""
import json, os, sys
from collections import defaultdict

SUBS = ["moe:apply.cpu_copy", "moe:apply.cpu_align",
        "moe:apply.gpu_copy", "moe:apply.launch"]


def load(path):
    with open(path) as f:
        return json.load(f)["traceEvents"]


def stats(events, name):
    durs = [e["dur"] for e in events
            if e.get("name") == name and "dur" in e]
    return (sum(durs), len(durs),
            (sum(durs) / len(durs)) if durs else 0.0)


def graph_launches(events):
    cnt = 0
    tot = 0.0
    for e in events:
        if e.get("name") == "cudaGraphLaunch" and "dur" in e:
            cnt += 1
            tot += e["dur"]
    return cnt, tot


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
        sp = span_us(ev)
        vsum, vcnt, vmean = stats(ev, "moe:vkernel_apply")
        sub = {s: stats(ev, s) for s in SUBS}
        gl_cnt, gl_tot = graph_launches(ev)
        steps = None
        if out_tok and layers_per_pp:
            steps = out_tok  # one MoE apply per layer per decoded token
        per_step_moe = vmean * layers_per_pp if layers_per_pp else None
        rows[r] = dict(span=sp, vsum=vsum, vcnt=vcnt, vmean=vmean,
                       frac=vsum / sp if sp else 0.0, sub=sub,
                       gl_cnt=gl_cnt, gl_tot=gl_tot,
                       per_step_moe=per_step_moe, steps=steps)
    return rows


def fmt(rows, label):
    print(f"\n================ {label} ================")
    for r, d in sorted(rows.items()):
        sp = d["span"] / 1e6
        print(f"--- rank {r}  (span {sp:.2f}s, {d['vcnt']} moe:vkernel_apply) ---")
        print(f"  moe:vkernel_apply  sum={d['vsum']/1e3:8.1f}ms  mean={d['vmean']:7.2f}us"
              f"  frac={d['frac']*100:5.1f}%")
        if d["per_step_moe"] is not None:
            lpp = d["vmean"] * 0 + 0  # placeholder
            print(f"  per-step MoE (mean*{d.get('_lpp','?')}): "
                  f"{d['per_step_moe']/1e3:6.1f}ms")
        for s in SUBS:
            ss, sc, sm = d["sub"][s]
            frac_of_moe = (ss / d["vsum"] * 100) if d["vsum"] else 0.0
            print(f"    {s:24s} sum={ss/1e3:7.1f}ms n={sc:5d} "
                  f"mean={sm:6.2f}us  {frac_of_moe:5.1f}% of moe")
        print(f"  cudaGraphLaunch  n={d['gl_cnt']}  cpu_sum={d['gl_tot']/1e3:.1f}ms")


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 2
    eager_dir, break_dir = sys.argv[1], sys.argv[2]
    out_tok = int(sys.argv[3]) if len(sys.argv) > 3 else None
    lpp = int(sys.argv[4]) if len(sys.argv) > 4 else None
    # inject lpp for display
    e = analyze(eager_dir, out_tok, lpp)
    b = analyze(break_dir, out_tok, lpp)
    for d in list(e.values()) + list(b.values()):
        d["_lpp"] = lpp
    fmt(e, f"EAGER  {os.path.basename(os.path.dirname(eager_dir))}")
    fmt(b, f"BREAKABLE  {os.path.basename(os.path.dirname(break_dir))}")
    # head-to-head on rank 0
    if 0 in e and 0 in b:
        print("\n================ HEAD-TO-HEAD (rank 0) ================")
        eo, bo = e[0], b[0]
        print(f"  span               {eo['span']/1e6:6.2f}s -> {bo['span']/1e6:6.2f}s")
        print(f"  vkernel_apply mean {eo['vmean']:6.2f}us -> {bo['vmean']:6.2f}us"
              f"  ({bo['vmean']/eo['vmean']:.2f}x)")
        print(f"  moe frac of window {eo['frac']*100:5.1f}% -> {bo['frac']*100:5.1f}%")
        for s in SUBS:
            es, bs = e[0]["sub"][s][0], b[0]["sub"][s][0]
            print(f"  {s:22s} {es/1e3:7.1f}ms -> {bs/1e3:7.1f}ms  ({bs/es:.2f}x)")


if __name__ == "__main__":
    sys.exit(main() or 0)
