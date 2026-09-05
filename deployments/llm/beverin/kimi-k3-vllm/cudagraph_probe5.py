#!/usr/bin/env python3
"""Probe v5: K3 model structure + MoE call chain (for break-point placement)."""
import os, glob, subprocess
OUT = os.environ.get("OUT", "/tmp/k3_model5.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)
V = "/usr/local/lib/python3.12/dist-packages/vllm"

w("=== kimi_k3 model dir ===")
w("\n".join(sorted(glob.glob(V + "/models/kimi_k3/*.py"))))
w("\n=== kimi_k3/nvidia dir ===")
w("\n".join(sorted(glob.glob(V + "/models/kimi_k3/nvidia/*.py"))))

w("\n=== model class + layer/forward refs ===")
for pat in (V + "/models/kimi_k3/*.py", V + "/models/kimi_k3/nvidia/*.py"):
    for p in sorted(glob.glob(pat)):
        src = open(p).read().splitlines()
        hits = []
        for i, line in enumerate(src, 1):
            if any(k in line for k in ("class KimiK3", "class .*DecoderLayer",
                    "self.mlp =", "self.experts =", "self.attn =",
                    "self.feed_forward =", "def forward")):
                hits.append(f"{i:4}: {line.rstrip()[:110]}")
        if hits:
            w(f"--- {os.path.basename(p)} ---")
            w("\n".join(hits[:25]))

w("\n=== UnfusedOAITritonExperts (parent of VkernelFusedExperts) forward/apply ===")
F = V + "/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py"
src = open(F).read().splitlines()
for i, line in enumerate(src, 1):
    if "class UnfusedOAITritonExperts" in line or "def forward" in line \
            or "self.apply(" in line or "return self.apply" in line:
        w(f"{i:4}: {line.rstrip()[:110]}")
w("\n--- forward body (first 40 lines after 'def forward') ---")
started = False
cnt = 0
for i, line in enumerate(src, 1):
    if "def forward" in line and not started:
        started = True
    if started:
        w(f"{i:4}: {line.rstrip()[:110]}")
        cnt += 1
        if cnt >= 40: break

w("\nDONE")
f.close(); print("WROTE", OUT, flush=True)
