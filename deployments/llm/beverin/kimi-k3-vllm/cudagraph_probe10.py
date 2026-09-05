#!/usr/bin/env python3
"""Probe v10: the exact downgrade gates + breakable runtime + break primitive + vkernels loc. NO vllm import."""
import os, subprocess
OUT = os.environ.get("OUT", "/tmp/k3_cm10.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)
def show(path, a, b):
    if not os.path.exists(path): w(f"  [missing] {path}"); return
    s = open(path).read().splitlines()
    w(f"=== {path} {a}-{b} ===")
    for i in range(a, min(b + 1, len(s) + 1)):
        w(f"{i:4}: {s[i-1].rstrip()[:130]}")
    w("")

# 1. config/compilation.py: the downgrade gate around 1155-1210 + splitting_ops definition
show("/usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py", 1145, 1210)

# 2. is_breakable_cudagraph_enabled: find it everywhere
w("=== locate is_breakable_cudagraph_enabled ===")
r = subprocess.run(["grep", "-rlnE", "def is_breakable_cudagraph_enabled|is_breakable_cudagraph_enabled",
                    "/usr/local/lib/python3.12/dist-packages/vllm/"], capture_output=True, text=True)
for cf in r.stdout.split():
    w(f"  {cf}")
w("")

# 3. BreakableCUDAGraphWrapper class source
w("=== locate BreakableCUDAGraphWrapper ===")
r = subprocess.run(["grep", "-rln", "class BreakableCUDAGraphWrapper", "/usr/local/lib/python3.12/dist-packages/vllm/"],
                   capture_output=True, text=True)
for cf in r.stdout.split():
    s = open(cf).read().splitlines()
    for i, l in enumerate(s, 1):
        if "class BreakableCUDAGraphWrapper" in l:
            w(f"--- {cf} (class @ {i}) ---")
            for j in range(i, min(i + 90, len(s) + 1)):
                w(f"{j:4}: {s[j-1].rstrip()[:130]}")
            break
    # also its apply/replay/capture methods
    for i, l in enumerate(s, 1):
        if l.strip().startswith("def ") and i < 400 and "class BreakableCUDAGraphWrapper" in open(cf).read():
            pass
w("")

# 4. The break primitive name: search splitting_ops + capture + replay + break
w("=== break primitive search ===")
for pat in [r"splitting_ops", r"def capture\b", r"def replay\b", r"\bbreak\b", r"piecewise", r"cudaGraph.*reak", r"graph_break"]:
    r = subprocess.run(["grep", "-rlnE", pat, "/usr/local/lib/python3.12/dist-packages/vllm/worker/",
                        "/usr/local/lib/python3.12/dist-packages/vllm/compilation/"],
                       capture_output=True, text=True)
    files = r.stdout.split()
    w(f"--- '{pat}' -> {len(files)} files ---")
    for cf in files[:6]:
        w(f"    {cf}")
w("")

# 5. where is vkernels actually installed?
w("=== vkernels location ===")
for cmd in [
    "pip show vkernels 2>/dev/null | head -12",
    "python3 -c 'import vkernels; print(vkernels.__file__)' 2>&1 | head -3",
    "find / -maxdepth 8 -name 'experts.py' -path '*vkernels*' 2>/dev/null | head",
    "find / -maxdepth 8 -type d -name 'vkernels' 2>/dev/null | head",
    "env | grep -iE 'MXFP4|VKERNELS|PYTHONPATH|VLLM_ROCM_USE_AITER_MOE' | head",
]:
    w(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    w(r.stdout.strip()[:600] or "(empty)")
w("")

# 6. K3 MoE: does FusedMoE / VkernelFusedExperts.register show splitting op registration?
w("=== grep splitting op registration in vllm + sitecustomize ===")
for pat in [r"register.*split", r"splitting_ops.*append", r"SPLITTING_OPS", r"add.*splitting"]:
    r = subprocess.run(["grep", "-rnE", pat,
                        "/usr/local/lib/python3.12/dist-packages/vllm/compilation/",
                        "/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/kimi-k3-vllm/sitecustomize.py"],
                       capture_output=True, text=True)
    w(f"--- '{pat}' ---")
    for line in r.stdout.splitlines()[:12]:
        w(f"  {line[:140]}")
w("")

w("DONE")
f.close(); print("WROTE", OUT, flush=True)
