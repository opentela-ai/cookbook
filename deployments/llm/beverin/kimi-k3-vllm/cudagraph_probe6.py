#!/usr/bin/env python3
"""Probe v6: K3 layer dispatch (Mega vs FusedMoE) + KimiK3MegaMoEExperts.forward."""
import os, subprocess
OUT = os.environ.get("OUT", "/tmp/k3_disp6.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)
F = "/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/model.py"
src = open(F).read().splitlines()
n = len(src)

w("=== layer dispatch (515-560) ===")
for i in range(515, min(561, n) + 1):
    w(f"{i:4}: {src[i-1].rstrip()[:118]}")
w("\n=== mlp dispatch (770-795) ===")
for i in range(770, min(796, n) + 1):
    w(f"{i:4}: {src[i-1].rstrip()[:118]}")
w("\n=== KimiK3MegaMoEExperts (194-300): def/class/apply/return ===")
for i in range(194, min(301, n) + 1):
    line = src[i-1]
    if any(k in line for k in ("def forward", "self.apply", "method.apply",
            "class ", "VkernelFusedExperts", "return", "self.experts")):
        w(f"{i:4}: {line.rstrip()[:118]}")
w("\n=== grep model.py for mega / use_mega / MEGA / KimiK3MegaMoEExperts ===")
for i, line in enumerate(src, 1):
    if any(k in line for k in ("KimiK3MegaMoEExperts", "use_mega", "MEGA",
            "mega_moe", "sparse_moe")):
        w(f"{i:4}: {line.rstrip()[:118]}")
w("\nDONE")
f.close(); print("WROTE", OUT, flush=True)
