import os
import sys
# Run under `python3 -S` to skip the editable aiter .pth (which JIT-imports
# aiter on every cold start -> ~10 min stall on a fresh node). Add site-packages
# manually so plain torch (no aiter) is importable.
if "/opt/venv/lib/python3.10/site-packages" not in sys.path:
    sys.path.insert(0, "/opt/venv/lib/python3.10/site-packages")
import torch

base = "/capstor/scratch/cscs/xyao/glm-53-flash-beverin/comp_capture"
A = os.path.join(base, sys.argv[1])  # suspect (beverin)
B = os.path.join(base, sys.argv[2])  # reference (clariden)

def load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    t = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(t, dict):
        t = t.get("tensor", t.get("data", t))
    return t.float()

def cmp(name):
    a, b = load(A, name), load(B, name)
    if a is None or b is None:
        print(f"{name:28s} MISSING (A={a is not None}, B={b is not None})")
        return
    if a.shape != b.shape:
        print(f"{name:28s} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).norm() / (b.norm() + 1e-12)).item()
    print(f"{name:28s} cos={cos:+.6f} rel={rel:.4f} rmsA={a.std():.6f} rmsB={b.std():.6f} ratioA/B={a.std()/(b.std()+1e-12):.4f}")

print(f"A={sys.argv[1]}  B={sys.argv[2]}")
for name in [
    "embed_out", "layer03_in", "layer03_out",
    "comp_layer3_attn_pre_in", "comp_layer3_attn_pre_out",
    "comp_layer3_attn_in", "comp_layer3_attn_out",
    "comp_layer3_ffn_pre_in", "comp_layer3_ffn_pre_out",
    "comp_layer3_mlp_out",
    "comp_layer3_post_in", "comp_layer3_post_out",
]:
    cmp(name)
