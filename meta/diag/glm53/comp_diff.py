#!/usr/bin/env python3
"""comp_diff.py -- bisect GLM-5.3-Flash beverin-vs-reference divergence.

The ONE analysis tool for the dumps produced by comp_capture.py (one .pt per
tensor + manifest.json).  Lives in meta/diag/glm53 so every site uses the
same copy.

  LAYERS -- which layer first goes wrong? (cross-machine)
    python3 comp_diff.py layers <dirA> <dirB> [--cos T] [--rel T] [--force]
    Verifies BOTH captures saw the SAME request (saved input_ids) — a
    mismatch ABORTS the comparison unless --force — then compares
    layer{i:02d}_in/out between the dirs and reports embed_out (the layer-0
    input sanity check).  The FIRST layer whose OUT diverges (given IN
    matched) names the broken kernel family, labelled from the capture
    manifest's _meta.layer_types (real model class names — no index guessing).

  COMPONENTS -- which component of that layer is wrong?
    python3 comp_diff.py components <dirA> [--layer N]
            [--ref <pure_torch_dir>] [--clariden <dirB>]
    Compares comp_layer{N}_{attn_pre,attn,ffn_pre,mlp,post}_in/_out against
    --ref (a pure-torch reference built from the same weights+input) and,
    where components align, against --clariden.  The first component (in
    execution order: attn_pre -> attn -> ffn_pre -> mlp -> post) whose OUT
    diverges given its IN matched is the broken kernel.

  SUMMARY -- manifest-only health check (no .pt loading)
    python3 comp_diff.py summary <dir>
    Per-layer in/out abs_mean, out/in ratio and nan/inf flags straight from
    manifest.json; flags the first layer whose ratio or cleanliness goes
    anomalous.  Use this first — it is instant and catches gross breakage.

All comparisons report max_abs, mean_abs, rel_max (vs reference magnitude),
cosine similarity, and allclose.  Divergence verdict: cosine < COS_THRESH OR
rel_max > REL_THRESH (defaults 0.999, 1e-3; tune via --cos --rel).
"""
import argparse
import json
import os
import re
import sys

try:
    import torch
except ImportError:  # --help / bad-invocation diagnostics must work anywhere
    torch = None

_COMP_ORDER = ["attn_pre", "attn", "ffn_pre", "mlp", "post"]


def _load(path):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        print(f"  ! load failed {path}: {exc!r}", file=sys.stderr)
        return None


def _load_manifest(d):
    return _load(os.path.join(d, "manifest.json")) or {}


def _layer_types(manifest):
    """{layer_index: "Class(mlp_class)"} from the capture manifest; {} when
    the capture predates layer_types recording."""
    out = {}
    for i, info in enumerate(manifest.get("_meta", {}).get("layer_types", [])):
        out[i] = f"{info.get('class', '?')}(mlp={info.get('mlp', '?')})"
    return out


def _identity_check(dir_a, dir_b, force):
    """Verify both captures fired on the SAME request (saved input_ids).

    The single validity guard for a cross-machine diff: comp_capture saves
    input_ids exactly so this check is possible.  Returns False only when the
    captures provably differ and --force was not given."""
    ia = _load(os.path.join(dir_a, "input_ids.pt"))
    ib = _load(os.path.join(dir_b, "input_ids.pt"))
    if ia is None or ib is None:
        print(
            f"  ! input_ids missing (A={ia is not None}, B={ib is not None}) — "
            "cannot verify the two captures saw the SAME request"
        )
        return force
    same = ia.shape == ib.shape and bool((ia.to(torch.long) == ib.to(torch.long)).all())
    print(f"  input_ids: A shape={list(ia.shape)} B shape={list(ib.shape)} identical={same}")
    if same:
        return True
    print(f"    A first ids: {ia.flatten()[:24].tolist()}")
    print(f"    B first ids: {ib.flatten()[:24].tolist()}")
    print("  !! CAPTURES ARE FROM DIFFERENT REQUESTS — tensor diff is INVALID")
    return force


def _metrics(a, b):
    a = _flat(a)
    b = _flat(b)
    if a is None or b is None:
        return {"status": "missing", "a": a is not None, "b": b is not None}
    if a.shape != b.shape:
        return {
            "status": "shape_mismatch",
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
        }
    # compare in float32 regardless of source dtype (bf16 vs fp32 ref is
    # expected; the *values* must still agree up to dtype rounding).
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    diff = (a - b).abs()
    ref_mag = b.abs().max().clamp(min=1e-12).item()
    cos = torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0
    ).item()
    return {
        "status": "ok",
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rel_max": float(diff.max().item() / ref_mag),
        "cosine": float(cos),
        "allclose": bool(torch.allclose(a, b, rtol=1e-3, atol=1e-4)),
        "shape": list(a.shape),
        "a_abs_mean": float(a.abs().mean().item()),
        "b_abs_mean": float(b.abs().mean().item()),
    }


def _flat(t):
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        xs = [x for x in t if isinstance(x, torch.Tensor)]
        return xs[0] if xs else None
    return t


def _verdict(m, cos_thresh, rel_thresh):
    if m["status"] != "ok":
        return m["status"]
    if m["cosine"] < cos_thresh or m["rel_max"] > rel_thresh:
        return "DIVERGE"
    return "match"


def _print_row(name, m, cos_thresh, rel_thresh, indent="    "):
    v = _verdict(m, cos_thresh, rel_thresh)
    if m["status"] != "ok":
        print(f"{indent}{name:<22} {v:>14}  {m}")
        return
    flag = "  <<<" if v == "DIVERGE" else ""
    print(
        f"{indent}{name:<22} {v:>14}  "
        f"cos={m['cosine']:.5f} rel={m['rel_max']:.2e} "
        f"max={m['max_abs']:.2e} mean={m['mean_abs']:.2e} "
        f"shape={m.get('shape')} {flag}"
    )


def _families(dir_a, dir_b):
    """Family label per layer index, from the manifests' recorded model
    classes (falls back to 'unknown' — never guessed from the index)."""
    types = _layer_types(_load_manifest(dir_a)) or _layer_types(_load_manifest(dir_b))
    if not types:
        return {}
    return {i: name for i, name in types.items()}


def cmd_layers(args):
    a_dir, b_dir = args.dir_a, args.dir_b
    print(f"=== LAYER BISECT: A(test)={a_dir}  B(reference)={b_dir} ===")
    print(f"    cos_thresh={args.cos}  rel_thresh={args.rel}")
    if not _identity_check(a_dir, b_dir, args.force):
        print("  refusing to compare (same-request identity not established); "
              "re-run with --force to compare anyway.")
        return 1

    # embedding (layer-0 input) sanity: distinguishes a deep pre-layer bug
    # from a layer-compute bug before the per-layer table.
    ea, eb = _load(os.path.join(a_dir, "embed_out.pt")), _load(os.path.join(b_dir, "embed_out.pt"))
    if ea is not None and eb is not None and ea.shape == eb.shape:
        cos = torch.nn.functional.cosine_similarity(
            ea.flatten().float(), eb.flatten().float(), dim=0).item()
        print(f"  embed_out: A abs_mean={ea.abs().float().mean():.6f} "
              f"B abs_mean={eb.abs().float().mean():.6f} cos={cos:.6f}")
    elif ea is not None or eb is not None:
        print(f"  embed_out: SHAPE MISMATCH A={list(ea.shape) if ea is not None else '-'} "
              f"B={list(eb.shape) if eb is not None else '-'}")
    print()

    n = 0
    while os.path.exists(os.path.join(a_dir, f"layer{n:02d}_out.pt")):
        n += 1
    if n == 0:
        print("  ! no layerNN_out.pt found in dir A; nothing to compare.")
        return 1
    n = min(n, args.layer_max) if args.layer_max > 0 else n
    fams = _families(a_dir, b_dir)
    print(f"    comparing {n} layers (in & out each)\n")
    first_diverge = None
    for i in range(n):
        fam = fams.get(i, "class unknown (capture predates layer_types)")
        m_in = _metrics(
            _load(os.path.join(a_dir, f"layer{i:02d}_in.pt")),
            _load(os.path.join(b_dir, f"layer{i:02d}_in.pt")),
        )
        m_out = _metrics(
            _load(os.path.join(a_dir, f"layer{i:02d}_out.pt")),
            _load(os.path.join(b_dir, f"layer{i:02d}_out.pt")),
        )
        v_in = _verdict(m_in, args.cos, args.rel)
        v_out = _verdict(m_out, args.cos, args.rel)
        print(f"  layer {i:02d}  [{fam}]")
        _print_row("IN ", m_in, args.cos, args.rel)
        _print_row("OUT", m_out, args.cos, args.rel)
        if v_out == "DIVERGE" and v_in != "DIVERGE" and first_diverge is None:
            first_diverge = (i, fam)
    print("\n=== VERDICT ===")
    if first_diverge is None:
        print("  No layer diverges (within thresholds). Either A is "
              "correct, or divergence is below threshold -- lower --cos/--rel.")
    else:
        i, fam = first_diverge
        print(
            f"  FIRST divergent layer: {i}  [{fam}]\n"
            f"    its IN matched the reference but its OUT diverged -> the "
            f"forward of layer {i} corrupts the residual.\n"
            f"  NEXT: comp_capture components GLM53_COMP_LAYER={i}, then "
            f"comp_diff components --layer {i} --ref <pure_torch_dir>."
        )
    return 0


def cmd_components(args):
    a_dir = args.dir_a
    L = args.layer
    print(f"=== COMPONENT BISECT: A={a_dir}  layer={L} ===")
    refdir = args.ref
    cdir = args.clariden
    if refdir is None and cdir is None:
        print("  ! no --ref or --clariden given; printing A-only stats "
              "(confirms capture, cannot judge divergence).\n")
    else:
        # same-request guard for whichever reference we compare against
        if not _identity_check(a_dir, refdir or cdir, args.force):
            print("  refusing to compare (same-request identity not "
                  "established); re-run with --force to compare anyway.")
            return 1
    # also compare the layer's residual IN/OUT if present
    print(f"  (residual IN/OUT of layer {L})")
    for tag in ("in", "out"):
        b = _load(os.path.join(a_dir, f"layer{L:02d}_{tag}.pt"))
        if b is None:
            print(f"    layer{L:02d}_{tag}: <missing>")
            continue
        b = _flat(b).to(torch.float32) if isinstance(b, torch.Tensor) else b
        info = {
            "status": "A-only",
            "shape": list(b.shape) if isinstance(b, torch.Tensor) else "?",
            "abs_mean": float(b.abs().mean()) if isinstance(b, torch.Tensor) else "?",
            "abs_max": float(b.abs().max()) if isinstance(b, torch.Tensor) else "?",
            "has_nan": bool(torch.isnan(b).any()) if isinstance(b, torch.Tensor) else "?",
        }
        print(f"    layer{L:02d}_{tag}: {info}")
    print()
    first_diverge = None
    for role in _COMP_ORDER:
        b_in = _load(os.path.join(a_dir, f"comp_layer{L}_{role}_in.pt"))
        b_out = _load(os.path.join(a_dir, f"comp_layer{L}_{role}_out.pt"))
        print(f"  component: {role}")
        if refdir:
            r_in = _load(os.path.join(refdir, f"comp_layer{L}_{role}_in.pt"))
            r_out = _load(os.path.join(refdir, f"comp_layer{L}_{role}_out.pt"))
            m_in = _metrics(b_in, r_in)
            m_out = _metrics(b_out, r_out)
            _print_row("IN (ref) ", m_in, args.cos, args.rel)
            _print_row("OUT(ref) ", m_out, args.cos, args.rel)
            if _verdict(m_out, args.cos, args.rel) == "DIVERGE" and _verdict(
                m_in, args.cos, args.rel
            ) != "DIVERGE" and first_diverge is None:
                first_diverge = role
        elif cdir:
            r_out = _load(os.path.join(cdir, f"comp_layer{L}_{role}_out.pt"))
            m_out = _metrics(b_out, r_out)
            _print_row("OUT(ref) ", m_out, args.cos, args.rel)
            if _verdict(m_out, args.cos, args.rel) == "DIVERGE":
                first_diverge = first_diverge or role
        else:
            for tag, t in (("IN", b_in), ("OUT", b_out)):
                t = _flat(t)
                if isinstance(t, torch.Tensor):
                    t = t.to(torch.float32)
                    print(
                        f"      {tag}: shape={list(t.shape)} "
                        f"abs_mean={float(t.abs().mean()):.4e} "
                        f"abs_max={float(t.abs().max()):.4e} "
                        f"nan={bool(torch.isnan(t).any())}"
                    )
                else:
                    print(f"      {tag}: <missing or non-tensor>")
    print("\n=== VERDICT ===")
    if refdir is None and cdir is None:
        print("  (A-only; capture confirmed. Re-run with --ref or "
              "--clariden to judge divergence.)")
    elif first_diverge is None:
        print("  No component diverges (within thresholds).")
    else:
        print(
            f"  FIRST divergent component: {first_diverge}\n"
            f"    its OUT diverged from the reference given its IN matched -> "
            f"{first_diverge} is the broken kernel.\n"
            f"  (attn_pre/ffn_pre = MHC tilelang; attn = KDA mamba or DSA; "
            f"mlp = MoE FP8 dequant or dense GEMM; post = mHC post)"
        )
    return 0


def cmd_summary(args):
    """Manifest-only health table (the old analyze_bisect.py, no .pt loads)."""
    path = os.path.join(args.capture_dir, "manifest.json")
    try:
        with open(path) as fh:
            m = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! cannot read {path}: {exc!r}")
        return 1
    meta = m.get("_meta", {})
    print(f"manifest: {path}  (mode={meta.get('mode')} layer={meta.get('layer')} "
          f"tag={meta.get('tag')})")

    layers = sorted(
        int(mm.group(1)) for k in m for mm in [re.match(r"layer(\d+)_out$", k)] if mm
    )
    print(f"layers captured: {len(layers)}"
          + (f" (indices {layers[0]}..{layers[-1]})" if layers else "") + "\n")
    print(f"{'layer':>5} {'in_absmean':>11} {'out_absmean':>11} {'out/in':>8} "
          f"{'in_rms':>10} {'out_rms':>10} {'out_max':>10}  flags")

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
        flags = (("N" if o["has_nan"] else "") + ("I" if o["has_inf"] else "")
                 + ("n" if inp["has_nan"] else "") + ("i" if inp["has_inf"] else ""))
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
        if re.match(r"layer\d+_(in|out)$", k) or k == "_meta":
            continue
        v = m[k]
        if "abs_mean" in v:
            print(f"{k:>32} abs_mean={v['abs_mean']:.6f} rms={v['rms']:.5f} "
                  f"max={v['abs_max']:.3f} shape={v['shape']}")

    # smoothness hint: biggest single-layer jump in out/in ratio
    ratios = []
    for i in layers:
        o, inp = m.get(f"layer{i:02d}_out"), m.get(f"layer{i:02d}_in")
        if o and inp and inp["abs_mean"]:
            ratios.append((i, o["abs_mean"] / inp["abs_mean"]))
    if len(ratios) >= 2:
        jumps = [(abs(ratios[j][1] - ratios[j - 1][1]), ratios[j - 1][0], ratios[j][0])
                 for j in range(1, len(ratios))]
        jumps.sort(reverse=True)
        print("\n### largest out/in ratio jumps (delta, layer_prev -> layer) ###")
        for d, a, b in jumps[:5]:
            print(f"  {d:8.3f}  layer{a:02d} -> layer{b:02d}")

    print(f"\nFIRST ANOMALOUS LAYER: "
          f"{first_bad if first_bad is not None else 'none (all ratios in sane band, no nan/inf)'}"
          + (f"  reason: {bad_reason}" if first_bad is not None else ""))
    return 0


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("layers", help="layer bisect: dir A (test) vs dir B (reference)")
    pl.add_argument("dir_a")
    pl.add_argument("dir_b")
    pl.add_argument("--layer-max", type=int, default=0, help="cap #layers (0=all)")
    pl.add_argument("--cos", type=float, default=0.999)
    pl.add_argument("--rel", type=float, default=1e-3)
    pl.add_argument("--force", action="store_true",
                    help="compare even when the input_ids identity check fails")
    pl.set_defaults(func=cmd_layers)

    pc = sub.add_parser("components", help="component bisect of layer N vs reference")
    pc.add_argument("dir_a")
    pc.add_argument("--layer", type=int, default=0)
    pc.add_argument("--ref", help="pure-torch reference dump dir")
    pc.add_argument("--clariden", help="cross-machine dump dir (if aligned)")
    pc.add_argument("--cos", type=float, default=0.999)
    pc.add_argument("--rel", type=float, default=1e-3)
    pc.add_argument("--force", action="store_true",
                    help="compare even when the input_ids identity check fails")
    pc.set_defaults(func=cmd_components)

    ps = sub.add_parser("summary", help="manifest-only per-layer health table")
    ps.add_argument("capture_dir")
    ps.set_defaults(func=cmd_summary)

    args = p.parse_args()
    if torch is None:
        print("comp_diff.py requires PyTorch (pip install torch)", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
