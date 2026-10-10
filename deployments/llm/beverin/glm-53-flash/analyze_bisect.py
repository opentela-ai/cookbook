#!/usr/bin/env python
"""Summarize a comp_capture MODE=layers manifest (45-layer residual bisect).

Usage: python analyze_bisect.py [manifest.json]
Prints per-layer in/out stats, out/in ratio, nan/inf flags, and flags the first
layer whose ratio or cleanliness goes anomalous. Also prints non-layer tensors
(embed_out etc.) and verdict hints for the GLM-5.3-Flash garbage-output bug.
"""
import json
import re
import sys

DEFAULT = "/capstor/scratch/cscs/xyao/glm-53-flash-beverin/comp_capture/bisect_layers_v1/manifest.json"
path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
m = json.load(open(path))

layers = sorted(int(mm.group(1)) for k in m for mm in [re.match(r"layer(\d+)_out$", k)] if mm)
print(f"manifest: {path}")
print(f"layers captured: {len(layers)} (indices {layers[0] if layers else '-'}..{layers[-1] if layers else '-'})\n")
print(f"{'layer':>5} {'in_absmean':>11} {'out_absmean':>11} {'out/in':>8} {'in_rms':>10} {'out_rms':>10} {'out_max':>10}  flags")

first_bad = None
bad_reason = ""
rows = []
for i in layers:
    o = m.get(f"layer{i:02d}_out")
    inp = m.get(f"layer{i:02d}_in")
    if not o or not inp:
        rows.append(f"{i:>5}  MISSING ({'no out' if not o else ''}{'no in' if not inp else ''})")
        continue
    ratio = o["abs_mean"] / inp["abs_mean"] if inp["abs_mean"] else float("nan")
    flags = ("N" if o["has_nan"] else "") + ("I" if o["has_inf"] else "") + ("n" if inp["has_nan"] else "") + ("i" if inp["has_inf"] else "")
    rows.append(
        f"{i:>5} {inp['abs_mean']:>11.6f} {o['abs_mean']:>11.6f} {ratio:>8.3f} "
        f"{inp['rms']:>10.5f} {o['rms']:>10.5f} {o['abs_max']:>10.3f}  {flags}"
    )
    if first_bad is None and (o["has_nan"] or o["has_inf"] or inp["has_nan"] or inp["has_inf"]):
        first_bad, bad_reason = i, "nan/inf"
    elif first_bad is None and (ratio != ratio or ratio > 5.0 or ratio < 0.05):
        first_bad, bad_reason = i, f"out/in ratio {ratio:.3f} out of sane band"
print("\n".join(rows))

print("\n### non-layer tensors ###")
for k in sorted(m):
    if re.match(r"layer\d+_(in|out)$", k):
        continue
    v = m[k]
    if "abs_mean" in v:
        print(f"{k:>32} abs_mean={v['abs_mean']:.6f} rms={v['rms']:.5f} max={v['abs_max']:.3f} shape={v['shape']}")

# smoothness hint: biggest single-layer jump in out/in ratio
ratios = []
for i in layers:
    o, inp = m.get(f"layer{i:02d}_out"), m.get(f"layer{i:02d}_in")
    if o and inp and inp["abs_mean"]:
        ratios.append((i, o["abs_mean"] / inp["abs_mean"]))
if len(ratios) >= 2:
    jumps = [(abs(ratios[j][1] - ratios[j - 1][1]), ratios[j - 1][0], ratios[j][0]) for j in range(1, len(ratios))]
    jumps.sort(reverse=True)
    print("\n### largest out/in ratio jumps (rank, layer_prev -> layer) ###")
    for d, a, b in jumps[:5]:
        print(f"  {d:8.3f}  layer{a:02d} -> layer{b:02d}")

print(f"\nFIRST ANOMALOUS LAYER: {first_bad if first_bad is not None else 'none (all ratios in sane band, no nan/inf)'}"
      + (f"  reason: {bad_reason}" if first_bad is not None else ""))
