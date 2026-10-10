#!/usr/bin/env python3
"""diff_l0_ops.py -- pairwise diff of two GLM53 components-mode capture dirs.

Usage: python3 diff_l0_ops.py <dirA> <dirB> [--tol 1e-3]
Compares every matching *.pt role (embed_out, comp_layer0_*_in/_out) and
prints a ranked divergence table. bf16 pairs are compared in fp32.
"""

import glob
import json
import os
import sys

import torch

DIRS = sys.argv[1:3]
TOL = 1e-3


def load_pt(p):
    try:
        t = torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        t = torch.load(p, map_location="cpu")
    if isinstance(t, dict) and "tensor" in t:
        t = t["tensor"]
    return t


def compare(name, pa, pb):
    a, b = load_pt(pa), load_pt(pb)
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return None
    if a.shape != b.shape:
        return (name, "SHAPE-MISMATCH", list(a.shape), list(b.shape))
    a, b = a.float().flatten(), b.float().flatten()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    d = (a - b).abs()
    denom = b.abs().mean().clamp_min(1e-9)
    rel = (d.mean() / denom).item()
    mx = d.max().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return (name, f"rel={rel:.5f} max={mx:.5f} cos={cos:.6f}",
            "OK" if rel < TOL else "*** DIVERGENT ***")


def main():
    roles = {}
    for tag, d in zip(("A", "B"), DIRS):
        for p in sorted(glob.glob(os.path.join(d, "*.pt"))):
            roles.setdefault(os.path.basename(p), {})[tag] = p
    print(f"A={DIRS[0]}\nB={DIRS[1]}\n")
    rows = []
    for name, sides in sorted(roles.items()):
        if "A" not in sides or "B" not in sides:
            print(f"{name:44s}  missing {'A' if 'A' not in sides else 'B'}")
            continue
        r = compare(name, sides["A"], sides["B"])
        if r is None:
            print(f"{name:44s}  (not tensors)")
        else:
            rows.append(r)
    for name, stat, verdict in rows:
        print(f"{name:44s}  {stat:44s} {verdict}")
    for d in DIRS:
        m = os.path.join(d, "manifest.json")
        if os.path.exists(m):
            j = json.load(open(m))
            meta = j.get("_meta") or {}
            print(f"\nmanifest {os.path.basename(os.path.dirname(m))}: "
                  f"keys={list(j.keys())[:6]}... meta={meta}")


if __name__ == "__main__":
    main()
