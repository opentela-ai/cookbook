#!/usr/bin/env python3
"""Localize the beverin-vs-clariden divergence AT LAYER 0 (the first decoder
layer, a DENSE DeepseekV2MLP — no MoE).  Layers 0-2 are dense per
clariden_l0_2k manifest, so any divergence here is purely attention/GEMM
numerics (no MoE winner-take-all amplification).

Reference: clariden_l0_2k  (layer 0, 2048 tok, 1200 min)
Beverin:    beverin_l0_fixed (layer 0, 2048 tok, 1200 min)  [fp32 router fix]

Chain at layer 0:  embed_out -> layer00_in ->
    comp_layer0_attn_in -> comp_layer0_attn_out ->
    comp_layer0_ffn_pre_in -> comp_layer0_mlp_in -> comp_layer0_mlp_out ->
    comp_layer0_ffn_pre_out -> comp_layer0_post_out -> layer00_out
"""
import os
import sys
import torch

B = sys.argv[1] if len(sys.argv) > 1 else \
    "/capstor/scratch/cscs/xyao/glm-53-flash-beverin/comp_capture"
REF = sys.argv[2] if len(sys.argv) > 2 else "clariden_l0_2k"
TAGS = [t for t in sys.argv[3:]] or ["beverin_l0_fixed", "beverin_l0"]


def load(tag, name):
    p = os.path.join(B, tag, name + ".pt")
    if not os.path.exists(p):
        return None
    try:
        return torch.load(p, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[load] {tag}/{name}: {exc!r}\n")
        return None


def diff(a, b):
    if a is None or b is None:
        return "MISSING"
    fa = a.float()
    fb = b.float()
    if fa.shape != fb.shape:
        return f"SHAPE {tuple(fa.shape)} vs {tuple(fb.shape)}"
    d = (fa - fb).abs()
    denom = fa.abs().max().item() or 1.0
    cos = torch.nn.functional.cosine_similarity(
        fa.flatten().unsqueeze(0), fb.flatten().unsqueeze(0)).item()
    return (f"max_abs={d.max().item():.5g} mean_abs={d.mean().item():.5g} "
            f"rel_max={d.max().item()/denom:.4f} cos={cos:.6f} "
            f"exact_eq={(fa == fb).all().item()}")


def main():
    print(f"# layer-0 localization  REF={REF}  TAGS={TAGS}")
    print(f"#   layers 0-2 = dense DeepseekV2MLP (no MoE amplification here)")
    print()
    # input_ids MUST match (same prompt)
    rid = load(REF, "input_ids")
    for t in TAGS:
        ids = load(t, "input_ids")
        same = (ids is not None and rid is not None and
                ids.shape == rid.shape and (ids == rid).all().item())
        print(f"  input_ids {t} vs {REF}: "
              f"{'EXACT' if same else diff(ids, rid)}")
    print()

    chain = [
        "embed_out", "layer00_in",
        "comp_layer0_attn_in", "comp_layer0_attn_pre_in",
        "comp_layer0_attn_pre_out", "comp_layer0_attn_out",
        "comp_layer0_ffn_pre_in", "comp_layer0_mlp_in", "comp_layer0_mlp_out",
        "comp_layer0_ffn_pre_out", "comp_layer0_post_in",
        "comp_layer0_post_out", "layer00_out",
    ]
    print("# divergence chain (find where cos first drops from ~1.0):")
    for n in chain:
        r = load(REF, n)
        rs = (f"shape={tuple(r.shape)} dtype={r.dtype} "
              f"abs_max={r.float().abs().max().item():.4g}") if r is not None else "MISSING"
        print(f"  [{n:26s}] REF {rs}")
        for t in TAGS:
            b = load(t, n)
            print(f"    {t:26s}: {diff(b, r)}")
    print()
    print("# If embed_out is EXACT but layer00_in (residual add) or attn_* "
          "diverges, the bug is in the layer-0 attention/linear-attention "
          "kernel (DSA-SDPA path) or the input-norm/projection GEMM — NOT the "
          "router (layer 0 is dense, no router).")


if __name__ == "__main__":
    main()
