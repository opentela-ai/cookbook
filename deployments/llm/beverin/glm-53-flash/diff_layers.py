#!/usr/bin/env python3
"""Cross-machine per-layer hidden-state diff for glm-5.3-flash comp_capture dumps.

Usage: python3 diff_layers.py <capture_dir_A (suspect)> <capture_dir_B (reference)>
Compares layerNN_in/layerNN_out/embed_out tensors: cosine + rel-diff + abs-diff.
Works with PARTIAL captures (no manifest.json -> enumerate .pt files directly).
"""

import json
import math
import os
import sys

import torch  # noqa: E402

MAX_ROWS = 10


def _load_manifest(d):
    try:
        return json.load(open(os.path.join(d, "manifest.json")))
    except Exception:  # noqa: BLE001
        return {}


def layer_nums(m, d=None):
    out = set()
    if m:
        for k in m:
            if k.startswith("layer") and k.endswith("_out") and k[5:-4].isdigit():
                out.add(int(k[5:-4]))
    if d is not None:
        try:
            for f in os.listdir(d):
                if f.startswith("layer") and f.endswith("_out.pt") and f[5:-7].isdigit():
                    out.add(int(f[5:-7]))
        except OSError:
            pass
    return sorted(out)


def load(d, name):
    p = os.path.join(d, name + ".pt")
    if not os.path.exists(p):
        return None
    try:
        return torch.load(p, map_location="cpu")
    except Exception as e:  # noqa: BLE001
        print(f"  [load {name} FAILED: {e}]")
        return None


def stats(a, b):
    if a.shape != b.shape:
        return f"SHAPE MISMATCH {list(a.shape)} vs {list(b.shape)}"
    af, bf = a.float().flatten(), b.float().flatten()
    dot = torch.dot(af, bf)
    na, nb = af.norm(), bf.norm()
    if na == 0 or nb == 0:
        cos = 0.0 if na != nb else 1.0
    else:
        cos = (dot / (na * nb)).item()
    diff = (af - bf)
    denom = bf.abs().mean().clamp_min(1e-9)
    rel = (diff.abs().mean() / denom).item()
    absmax = diff.abs().max().item()
    return f"cos={cos:+.6f} rel_mean={rel:.6f} abs_max={absmax:.6f}"


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    ma, mb = _load_manifest(a_dir), _load_manifest(b_dir)
    print("A meta:", ma.get("_meta", {}))
    print("B meta:", mb.get("_meta", {}))

    for name in ("input_ids", "positions"):
        ta, tb = load(a_dir, name), load(b_dir, name)
        if ta is None or tb is None:
            print(f"{name}: MISSING (a={ta is not None}, b={tb is not None})")
            continue
        eq = torch.equal(ta.flatten(), tb.flatten())
        print(f"{name}: shapes {list(ta.shape)} vs {list(tb.shape)} identical={eq}")

    la, lb = layer_nums(ma, a_dir), layer_nums(mb, b_dir)
    common = [n for n in la if n in set(lb)]
    print(f"\nlayers A={len(la)} B={len(lb)} common={len(common)} (A up to {max(la) if la else '-'})\n")

    print(f"{'tensor':<16}{'stats':<70}")
    print("-" * 86)
    ea, eb = load(a_dir, "embed_out"), load(b_dir, "embed_out")
    if ea is not None and eb is not None:
        print(f"{'embed_out':<16}{stats(ea, eb):<70}")
    for n in common:
        for suffix in ("_in", "_out"):
            ta, tb = load(a_dir, f"layer{n:02d}{suffix}"), load(b_dir, f"layer{n:02d}{suffix}")
            label = f"layer{n:02d}{suffix}"
            if ta is None or tb is None:
                print(f"{label:<16}MISSING a={ta is not None} b={tb is not None}")
                continue
            print(f"{label:<16}{stats(ta, tb):<70}")

    print("\nNOTE: layers beyond the last common pair exist only on one side"
          " (suspect capture may be partial).")


if __name__ == "__main__":
    main()
