#!/usr/bin/env python3
"""comp_diff.py -- bisect GLM-5.3-Flash beverin-vs-reference divergence.

Works on the dumps produced by comp_capture.py (one .pt per tensor +
manifest.json).  Two stages:

  STAGE 1 -- layer bisect (which layer first goes wrong?)
    python3 comp_diff.py layers <beverin_dir> <clariden_dir> [--layer-max N]
    For every layer i, compares layer{i:02d}_in and layer{i:02d}_out between
    the two dirs.  The FIRST layer where OUT diverges (given IN matched) names
    the broken kernel family:
      linear_attention layer (0,1,2,4,5,6,...) -> MHC pre-norm tilelang / Mamba
      deepseek_sparse_attention layer (3,7,...) -> DSA indexer/forward + MoE FP8
    If layer0_in already diverges, the embedding itself is bad (deep, pre-layer).

  STAGE 2 -- component bisect (which component of that layer is wrong?)
    python3 comp_diff.py components <beverin_dir> [--layer N]
            [--ref <pure_torch_dir>] [--clariden <clariden_dir>]
    Compares comp_layer{N}_{attn_pre,attn,ffn_pre,mlp,post}_in/_out of beverin
    against --ref (a pure-torch reference built from the same weights+input) and,
    where components align, against --clariden.  The first component (in
    execution order: attn_pre -> attn -> ffn_pre -> mlp -> post) whose OUT
    diverges given its IN matched is the broken kernel.

All comparisons report max_abs, mean_abs, rel_max (vs reference magnitude),
cosine similarity, and allclose.  Divergence verdict: cosine < COS_THRESH OR
rel_max > REL_THRESH (defaults 0.999, 1e-3; tune via --cos --rel).
"""
import argparse
import json
import os
import sys

import torch

_COMP_ORDER = ["attn_pre", "attn", "ffn_pre", "mlp", "post"]


def _load(path):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        print(f"  ! load failed {path}: {exc!r}", file=sys.stderr)
        return None


def _flat(t):
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        xs = [x for x in t if isinstance(x, torch.Tensor)]
        return xs[0] if xs else None
    return t


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
    if a.dtype != b.dtype:
        # compare in float32 regardless of source dtype (bf16 vs fp32 ref is
        # expected; the *values* must still agree up to dtype rounding).
        a = a.to(torch.float32)
        b = b.to(torch.float32)
    else:
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


def cmd_layers(args):
    bdir, cdir = args.beverin_dir, args.clariden_dir
    print(f"=== LAYER BISECT: beverin={bdir}  clariden={cdir} ===")
    print(f"    cos_thresh={args.cos}  rel_thresh={args.rel}")
    # discover layer count from beverin
    n = 0
    while os.path.exists(os.path.join(bdir, f"layer{n:02d}_out.pt")):
        n += 1
    if n == 0:
        print("  ! no layerNN_out.pt found in beverin dir; nothing to compare.")
        return 1
    n = min(n, args.layer_max) if args.layer_max > 0 else n
    print(f"    comparing {n} layers (in & out each)\n")
    first_diverge = None
    in_matched_prev = True
    for i in range(n):
        m_in = _metrics(
            _load(os.path.join(bdir, f"layer{i:02d}_in.pt")),
            _load(os.path.join(cdir, f"layer{i:02d}_in.pt")),
        )
        m_out = _metrics(
            _load(os.path.join(bdir, f"layer{i:02d}_out.pt")),
            _load(os.path.join(cdir, f"layer{i:02d}_out.pt")),
        )
        v_in = _verdict(m_in, args.cos, args.rel)
        v_out = _verdict(m_out, args.cos, args.rel)
        # Execution-order name of the layer's kernel family. Layer types come
        # from the config (linear_attention vs deepseek_sparse_attention); we
        # don't have it here, so just label the divergence and let the operator
        # map layer_id -> type from the known layout (0,1,2,4,5,6..=linear_attn,
        # 3,7,11,..=DSA; >=3 and %moe_layer_freq==0 -> MoE).
        fam = "linear_attn/MHC+Mamba" if (i % 4 != 3 and i < 12) else "DSA+MoE"
        print(f"  layer {i:02d}  [{fam}]")
        _print_row("IN ", m_in, args.cos, args.rel)
        _print_row("OUT", m_out, args.cos, args.rel)
        if v_out == "DIVERGE" and v_in != "DIVERGE" and first_diverge is None:
            first_diverge = (i, fam)
        in_matched_prev = v_in == "match"
    print("\n=== VERDICT ===")
    if first_diverge is None:
        print("  No layer diverges (within thresholds). Either beverin is "
              "correct, or divergence is below threshold -- lower --cos/--rel.")
    else:
        i, fam = first_diverge
        print(
            f"  FIRST divergent layer: {i}  [{fam}]\n"
            f"    its IN matched clariden but its OUT diverged -> the forward "
            f"of layer {i} corrupts the residual.\n"
            f"  NEXT: comp_capture components GLM53_COMP_LAYER={i}, then "
            f"comp_diff components --layer {i} --ref <pure_torch_dir>."
        )
    return 0


def cmd_components(args):
    bdir = args.beverin_dir
    L = args.layer
    print(f"=== COMPONENT BISECT: beverin={bdir}  layer={L} ===")
    refdir = args.ref
    cdir = args.clariden
    if refdir is None and cdir is None:
        print("  ! no --ref or --clariden given; printing beverin-only stats "
              "(confirms capture, cannot judge divergence).\n")
    # also compare the layer's residual IN/OUT if present
    print(f"  (residual IN/OUT of layer {L})")
    for tag in ("in", "out"):
        b = _load(os.path.join(bdir, f"layer{L:02d}_{tag}.pt"))
        if b is None:
            print(f"    layer{L:02d}_{tag}: <missing>")
            continue
        b = _flat(b).to(torch.float32) if isinstance(b, torch.Tensor) else b
        info = {
            "status": "beverin-only",
            "shape": list(b.shape) if isinstance(b, torch.Tensor) else "?",
            "abs_mean": float(b.abs().mean()) if isinstance(b, torch.Tensor) else "?",
            "abs_max": float(b.abs().max()) if isinstance(b, torch.Tensor) else "?",
            "has_nan": bool(torch.isnan(b).any()) if isinstance(b, torch.Tensor) else "?",
        }
        print(f"    layer{L:02d}_{tag}: {info}")
    print()
    first_diverge = None
    for role in _COMP_ORDER:
        b_in = _load(os.path.join(bdir, f"comp_layer{L}_{role}_in.pt"))
        b_out = _load(os.path.join(bdir, f"comp_layer{L}_{role}_out.pt"))
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
            _print_row("OUT(clar)", m_out, args.cos, args.rel)
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
        print("  (beverin-only; capture confirmed. Re-run with --ref or "
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("layers", help="stage 1: layer bisect beverin vs clariden")
    pl.add_argument("beverin_dir")
    pl.add_argument("clariden_dir")
    pl.add_argument("--layer-max", type=int, default=0, help="cap #layers (0=all)")
    pl.add_argument("--cos", type=float, default=0.999)
    pl.add_argument("--rel", type=float, default=1e-3)
    pl.set_defaults(func=cmd_layers)

    pc = sub.add_parser("components", help="stage 2: component bisect vs ref")
    pc.add_argument("beverin_dir")
    pc.add_argument("--layer", type=int, default=0)
    pc.add_argument("--ref", help="pure-torch reference dump dir")
    pc.add_argument("--clariden", help="clariden component dump dir (if aligned)")
    pc.add_argument("--cos", type=float, default=0.999)
    pc.add_argument("--rel", type=float, default=1e-3)
    pc.set_defaults(func=cmd_components)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
