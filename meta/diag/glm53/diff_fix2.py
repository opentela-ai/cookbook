#!/usr/bin/env python3
"""Diff beverin FIX2 vs clariden reference vs original (bf16) vs earlier FIX.

Validates the router-GEMM fp32 + correction_bias fp32 fix (tag
beverin_l3_moeinner_FIX2) by comparing the MoE layer-3 activations against the
clariden reference (correct, fp32 router) and the original beverin capture
(bf16 router, the baseline the fix should improve).

Usage: python3 diff_fix2.py [BASE_DIR] [tagA] [tagB] ...
Default: BASE=comp_capture dir; tags = clariden_l3_moeinner beverin_l3_moeinner
         beverin_l3_moeinner_FIX beverin_l3_moeinner_FIX2
"""
import os
import sys
import torch

B = sys.argv[1] if len(sys.argv) > 1 else "/capstor/scratch/cscs/xyao/glm-53-flash-beverin/comp_capture"
TAGS = sys.argv[2:] or [
    "clariden_l3_moeinner",      # reference (correct, fp32 router)
    "beverin_l3_moeinner",        # original (bf16 router) -- baseline
    "beverin_l3_moeinner_FIX",    # earlier fix attempt
    "beverin_l3_moeinner_FIX2",   # current fix (no _MOE_INNER -> no topk)
]
REF = "clariden_l3_moeinner"


def load(tag, name):
    p = os.path.join(B, tag, name + ".pt")
    if not os.path.exists(p):
        return None
    try:
        return torch.load(p, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[load] {tag}/{name}: {exc!r}\n")
        return None


def fstat(t):
    if t is None:
        return None
    f = t.float() if hasattr(t, "float") else t
    return dict(
        shape=tuple(t.shape) if hasattr(t, "shape") else None,
        dtype=str(t.dtype) if hasattr(t, "dtype") else None,
        numel=t.numel() if hasattr(t, "numel") else None,
        abs_max=f.abs().max().item() if hasattr(f, "abs") else None,
        abs_mean=f.abs().mean().item() if hasattr(f, "abs") else None,
    )


def diff(a, b):
    """Pairwise stats of two tensors (float, same shape)."""
    if a is None or b is None:
        return None
    fa = a.float() if hasattr(a, "float") else a
    fb = b.float() if hasattr(b, "float") else b
    if fa.shape != fb.shape:
        return dict(shape_a=tuple(fa.shape), shape_b=tuple(fb.shape), MISMATCH=True)
    d = (fa - fb).abs()
    denom = fa.abs().max().item() or 1.0
    return dict(
        max_abs=d.max().item(),
        mean_abs=d.mean().item(),
        rel_max=d.max().item() / denom,
        rms=((d ** 2).mean().sqrt()).item(),
        exact_eq=(fa == fb).all().item(),
    )


def topk_match(a, b):
    """For int64 topk_ids [N, K]: % of expert slots that match, and % of rows
    (tokens) with ALL top-K matching."""
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return dict(shape_a=tuple(a.shape), shape_b=tuple(b.shape), MISMATCH=True)
    eq = (a == b)
    return dict(
        slot_match_pct=eq.float().mean().item() * 100,
        rows_all_match_pct=eq.all(dim=1).float().mean().item() * 100,
        rows_any_mismatch_pct=(~eq.all(dim=1)).float().mean().item() * 100,
        n_tokens=a.shape[0],
        k=a.shape[1] if a.dim() > 1 else 1,
    )


def main():
    print(f"=== input_ids verification (same prompt across all dirs?) ===")
    ref_ids = load(REF, "input_ids")
    ref_stat = fstat(ref_ids)
    print(f"  {REF}: {ref_stat}")
    id_match = {}
    for tag in TAGS:
        if tag == REF:
            continue
        ids = load(tag, "input_ids")
        st = fstat(ids)
        eq = (ids == ref_ids).all().item() if (ids is not None and ref_ids is not None and ids.shape == ref_ids.shape) else None
        id_match[tag] = eq
        print(f"  {tag}: {st}  vs REF exact_eq={eq}")
    print()

    print(f"=== comp_layer3_mlp_in diff vs {REF} (should be ~0: same prompt, same pre-layer-3) ===")
    ref_in = load(REF, "comp_layer3_mlp_in")
    print(f"  {REF} mlp_in: {fstat(ref_in)}")
    for tag in TAGS:
        if tag == REF:
            continue
        t_in = load(tag, "comp_layer3_mlp_in")
        print(f"  {tag} vs {REF}: {diff(t_in, ref_in)}")
    print()

    print(f"=== comp_layer3_mlp_out diff vs {REF} (FIX2 should be CLOSER than original) ===")
    ref_out = load(REF, "comp_layer3_mlp_out")
    print(f"  {REF} mlp_out: {fstat(ref_out)}")
    results = {}
    for tag in TAGS:
        if tag == REF:
            continue
        t_out = load(tag, "comp_layer3_mlp_out")
        dd = diff(t_out, ref_out)
        results[tag] = dd
        print(f"  {tag} vs {REF}: {dd}")
    print()

    # Improvement: original vs FIX2 (relative to clariden ref)
    orig_max = results.get("beverin_l3_moeinner", {}).get("max_abs") if results.get("beverin_l3_moeinner") else None
    fix2_max = results.get("beverin_l3_moeinner_FIX2", {}).get("max_abs") if results.get("beverin_l3_moeinner_FIX2") else None
    if orig_max and fix2_max is not None:
        print(f"=== mlp_out improvement: original max_abs={orig_max:.6g} -> FIX2 max_abs={fix2_max:.6g} ({(1-fix2_max/orig_max)*100:+.1f}%) ===")
        print()

    print(f"=== moe_layer3_topk_ids match vs {REF} (the KEY metric: % expert selections agreeing) ===")
    ref_topk = load(REF, "moe_layer3_topk_ids")
    print(f"  {REF} topk_ids: {fstat(ref_topk)}")
    for tag in TAGS:
        if tag == REF:
            continue
        tk = load(tag, "moe_layer3_topk_ids")
        if tk is None:
            print(f"  {tag}: MISSING (no _MOE_INNER for this capture)")
        else:
            print(f"  {tag} vs {REF}: {topk_match(tk, ref_topk)}")
    print()

    print(f"=== moe_layer3_router_logits diff vs {REF} (the thing the fp32 fix targets) ===")
    ref_rl = load(REF, "moe_layer3_router_logits")
    print(f"  {REF} router_logits: {fstat(ref_rl)}")
    for tag in TAGS:
        if tag == REF:
            continue
        rl = load(tag, "moe_layer3_router_logits")
        if rl is None:
            print(f"  {tag}: MISSING (no _MOE_INNER for this capture)")
        else:
            print(f"  {tag} vs {REF}: {diff(rl, ref_rl)}")


if __name__ == "__main__":
    main()
