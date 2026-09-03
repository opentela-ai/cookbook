#!/usr/bin/env python3
"""diff — THE bisect diff tool for capture.py output (Phase 3).

Generalizes meta/diag/glm53/comp_diff.py. Compares two first-forward capture
directories (see capture.py) produced from the SAME prompt — enforced by the
identity gate: both manifests must carry the same top_input_ids_sha256, or the
diff REFUSES (exit 3) unless --force. A diff against a different prompt is
meaningless; this gate exists because that mistake is easy and silent.

Subcommands:
  layers A B        layer bisect: test dir A vs reference dir B. The first
                    layer whose OUT diverges (given its IN matched) names the
                    broken kernel family — labelled from the manifests'
                    layer_types when available.
  components A --layer N --ref R
                    drill into layer N's captured components of A against a
                    reference capture dir (pure-torch ref or cross-machine).
  summary D         manifest-only per-tensor health table. Torch-free: runs
                    anywhere, including on a laptop after rsyncing just the
                    manifests.

Verdicts: match | MISMATCH | nan | shape-mismatch | missing.
Exit codes: 0 all match; 1 at least one mismatch; 2 usage/IO error;
3 identity-gate refusal (rerun with --force only if you know why inputs differ).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _load_manifest(d):
    p = os.path.join(d, "manifest.json")
    if not os.path.exists(p):
        print(f"error: no manifest.json in {d}", file=sys.stderr)
        sys.exit(2)
    with open(p) as f:
        return json.load(f)


def _identity_check(dir_a, dir_b, force):
    ma, mb = _load_manifest(dir_a), _load_manifest(dir_b)
    ia = ma.get("top_input_ids_sha256")
    ib = mb.get("top_input_ids_sha256")
    if ia is None or ib is None:
        print(f"warning: identity digest missing (A={ia}, B={ib}); cannot verify same prompt", file=sys.stderr)
        if not force:
            print("refusing to diff without an identity anchor; --force to override", file=sys.stderr)
            sys.exit(3)
        return ma, mb
    if ia != ib:
        if not force:
            print(f"IDENTITY GATE: prompt digests differ (A={ia} B={ib}). A diff against a different "
                  f"prompt is meaningless. Re-capture with the same probe, or --force.", file=sys.stderr)
            sys.exit(3)
        print(f"note: --force over identity mismatch (A={ia} B={ib})", file=sys.stderr)
    else:
        print(f"identity ok (digest {ia})")
    return ma, mb


def _metrics(a, b):
    """cosine + relative error between two tensors (torch; lazy import)."""
    import torch

    fa, fb = a.detach().float().flatten(), b.detach().float().flatten()
    if fa.shape != fb.shape:
        return {"shape": [list(fa.shape), list(fb.shape)]}
    denom = fa.norm() * fb.norm()
    cos = float((fa @ fb) / denom) if denom > 0 else 1.0 if fa.norm() == fb.norm() == 0 else 0.0
    denom_rel = fb.abs().max()
    rel = float((fa - fb).abs().max() / denom_rel) if denom_rel > 0 else float((fa - fb).abs().max())
    return {"cos": cos, "rel": rel, "absdiff_max": float((fa - fb).abs().max())}


def _verdict(m, cos_thresh, rel_thresh):
    if "shape" in m:
        return "shape-mismatch"
    if m["cos"] != m["cos"] or m["cos"] < cos_thresh:
        return "nan" if m["cos"] != m["cos"] else "MISMATCH"
    if m["rel"] > rel_thresh:
        return "MISMATCH"
    return "match"


def _families(man):
    lt = man.get("layer_types") or []
    return {i: t for i, t in enumerate(lt) if t}


def _print_row(name, m, thresh, indent="    "):
    v = _verdict(m, thresh[0], thresh[1])
    if "shape" in m:
        print(f"{indent}{name:<28} {v:<14} shapes={m['shape']}")
    else:
        print(f"{indent}{name:<28} {v:<14} cos={m['cos']:.6f} rel={m['rel']:.3g} absmax={m['absdiff_max']:.3g}")
    return v


def _load_pt(d, name):
    import torch

    p = os.path.join(d, f"{name}.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu", weights_only=True)


# -- subcommands ---------------------------------------------------------------


def _layer_names(man):
    """Distinct layer base names (layer00, layer01, ...) present in a manifest."""
    names = set()
    for k in man.get("tensors", {}):
        if k.startswith("layer") and k[5:7].isdigit():
            names.add(k.rsplit("_", 1)[0])
    return sorted(names)


def _require_torch():
    try:
        import torch  # noqa: F401

        return torch
    except ImportError:
        print("error: torch is required for `diff layers`/`diff components` (it loads .pt tensors). "
              "Use `diff summary` for the manifest-only health table — it needs no torch.", file=sys.stderr)
        sys.exit(2)


def cmd_layers(args):
    _require_torch()
    ma, mb = _identity_check(args.dir_a, args.dir_b, args.force)
    fa, fb = _families(ma), _families(mb)
    common = [x for x in _layer_names(ma) if x in set(_layer_names(mb))
              and (args.layer_max == 0 or int(x[5:7]) <= args.layer_max)]
    if not common:
        print("error: no common layer files between the two captures", file=sys.stderr)
        sys.exit(2)
    first_bad = None
    for lname in common:
        idx = int(lname[5:7])
        fam = fa.get(idx, fb.get(idx, ""))
        print(f"-- {lname}{' (' + fam + ')' if fam else ''}")
        for phase in ("in", "out"):
            a, b = _load_pt(args.dir_a, f"{lname}_{phase}"), _load_pt(args.dir_b, f"{lname}_{phase}")
            if a is None or b is None:
                print(f"    {phase:<28} {'missing':<14}")
                continue
            v = _print_row(phase, _metrics(a, b), (args.cos, args.rel))
            if phase == "out" and v in ("MISMATCH", "nan", "shape-mismatch") and first_bad is None:
                first_bad = (idx, fam)
    print()
    if first_bad:
        idx, fam = first_bad
        print(f"VERDICT: first divergent layer = {idx} (family: {fam or 'unknown — record DBG_LAYER_TYPES in captures'}). "
              f"Next: diff.py components {args.dir_a} --layer {idx} --ref <ref_dir>")
        sys.exit(1)
    print("VERDICT: all captured layers match within thresholds.")
    sys.exit(0)


def cmd_components(args):
    _require_torch()
    ma, mref = _identity_check(args.dir_a, args.dir_ref, args.force)
    fa = _families(ma)
    idx = args.layer
    fam = fa.get(idx, "")
    print(f"== component drill-down: layer {idx}{' (' + fam + ')' if fam else ''}  A={args.dir_a}  ref={args.dir_ref}")
    comps = sorted({k[len(f"comp_layer{idx:02d}_"):].rsplit("_", 1)[0]
                    for k in ma.get("tensors", {}) if k.startswith(f"comp_layer{idx:02d}_")})
    if not comps:
        print(f"error: dir A has no comp_layer{idx:02d}_* captures (was it captured with "
              f"DBG_CAPTURE_MODE=components DBG_COMPONENT_LAYER={idx}?)", file=sys.stderr)
        sys.exit(2)
    first_bad = None
    for c in comps:
        print(f"-- {c}")
        for phase in ("in", "out"):
            name = f"comp_layer{idx:02d}_{c}_{phase}"
            a, b = _load_pt(args.dir_a, name), _load_pt(args.dir_ref, name)
            if a is None or b is None:
                print(f"    {phase:<28} {'missing':<14}")
                continue
            v = _print_row(phase, _metrics(a, b), (args.cos, args.rel))
            if phase == "out" and v in ("MISMATCH", "nan", "shape-mismatch") and first_bad is None:
                first_bad = c
    print()
    if first_bad:
        print(f"VERDICT: broken component = {first_bad!r} in layer {idx}. Isolate that kernel in a "
              f"standalone primitive test (Phase 4), then route it to a torch drop-in.")
        sys.exit(1)
    print(f"VERDICT: all captured components of layer {idx} match within thresholds.")
    sys.exit(0)


def cmd_summary(args):
    man = _load_manifest(args.capture_dir)   # torch-free
    tensors = man.get("tensors", {})
    if not tensors:
        print("manifest has no tensor entries")
        sys.exit(2)
    fam = _families(man)
    print(f"== summary: {args.capture_dir}  tag={man.get('tag')} mode={man.get('mode')} "
          f"prompt_digest={man.get('top_input_ids_sha256')}")
    bad = []
    for name in sorted(tensors):
        t = tensors[name]
        v = "ok"
        if t.get("nan") or t.get("inf"):
            v = "nan"
        elif t.get("abs_mean") is not None and t["abs_mean"] > 1e3:
            v = "explode"
        elif t.get("abs_mean") is not None and t["abs_mean"] < 1e-6:
            v = "collapse"
        if v != "ok":
            bad.append(name)
        idx = int(name[5:7]) if name.startswith("layer") and name[5:7].isdigit() else None
        label = f" ({fam[idx]})" if idx is not None and idx in fam else ""
        print(f"  {name:<28} {v:<10} dtype={t.get('dtype')} shape={t.get('shape')} "
              f"abs_mean={t.get('abs_mean'):.4g} nan={t.get('nan')}{label}")
    print()
    if bad:
        print(f"HEALTH: {len(bad)} suspicious tensor(s): {', '.join(bad)}")
        sys.exit(1)
    print("HEALTH: all captured tensors within nominal bounds.")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("layers", help="layer bisect: dir A (test) vs dir B (reference)")
    pl.add_argument("dir_a")
    pl.add_argument("dir_b")
    pl.add_argument("--layer-max", type=int, default=0, help="cap #layers (0=all)")
    pl.add_argument("--cos", type=float, default=0.999)
    pl.add_argument("--rel", type=float, default=1e-3)
    pl.add_argument("--force", action="store_true", help="override the identity gate")
    pl.set_defaults(fn=cmd_layers)

    pc = sub.add_parser("components", help="component drill-down of layer N vs a reference capture dir")
    pc.add_argument("dir_a")
    pc.add_argument("--layer", type=int, default=0)
    pc.add_argument("--ref", required=True, dest="dir_ref",
                    help="reference capture dir (pure-torch ref or cross-machine)")
    pc.add_argument("--cos", type=float, default=0.999)
    pc.add_argument("--rel", type=float, default=1e-3)
    pc.add_argument("--force", action="store_true")
    pc.set_defaults(fn=cmd_components)

    ps = sub.add_parser("summary", help="manifest-only per-tensor health table (no torch)")
    ps.add_argument("capture_dir")
    ps.set_defaults(fn=cmd_summary)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
