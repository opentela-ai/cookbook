#!/usr/bin/env python3
"""Probe v9: NO vllm import. Direct source reads.
- default cudagraph_mode + how VLLM_CUDAGRAPH_MODE maps
- is_breakable_cudagraph_enabled source
- BreakableCUDAGraphWrapper __init__/apply
- gpu_model_runner breakable-path (5430-5470)
- Does VkernelFusedExperts / K3 MoE carry the break marker?
- What is the 'break' primitive (mark_step_boundary / break_graph?)
"""
import os, subprocess
OUT = os.environ.get("OUT", "/tmp/k3_cm9.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)
def grep_file(path, pat, before=0, after=0, head=None):
    if not os.path.exists(path):
        w(f"  [missing] {path}"); return []
    r = subprocess.run(["grep", "-nE", pat, path], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        ln = int(line.split(":", 1)[0])
        out.append(ln)
    return out

# 1. CUDAGraphMode default + VLLM_CUDAGRAPH_MODE env mapping (config/compilation.py)
C = "/usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py"
if os.path.exists(C):
    cs = open(C).read().splitlines()
    w("=== config/compilation.py: CUDAGraphMode enum + default + VLLM_CUDAGRAPH_MODE ===")
    hits = grep_file(C, r"CUDAGraphMode|VLLM_CUDAGRAPH_MODE|cudagraph_mode.*=|default_cudagraph_mode")
    seen = set()
    for ln in hits:
        for j in range(max(1, ln - 1), min(len(cs) + 1, ln + 2)):
            if j not in seen:
                w(f"{j:4}: {cs[j-1].rstrip()[:120]}"); seen.add(j)
    w("")

# 2. is_breakable_cudagraph_enabled (cudagraph_utils.py)
CU = "/usr/local/lib/python3.12/dist-packages/vllm/worker/cudagraph_utils.py"
if os.path.exists(CU):
    cus = open(CU).read().splitlines()
    w("=== cudagraph_utils.py: is_breakable_cudagraph_enabled + BreakableCUDAGraphWrapper class ===")
    for i, l in enumerate(cus, 1):
        if "def is_breakable_cudagraph_enabled" in l or "class BreakableCUDAGraphWrapper" in l:
            for j in range(i, min(i + 45, len(cus) + 1)):
                w(f"{j:4}: {cus[j-1].rstrip()[:120]}")
            w("  ...")
    w("")

# 3. gpu_model_runner breakable path around 5430-5475
G = "/usr/local/lib/python3.12/dist-packages/vllm/worker/gpu_model_runner.py"
if os.path.exists(G):
    gs = open(G).read().splitlines()
    w("=== gpu_model_runner.py 5430-5475 (breakable wrapper selection) ===")
    for j in range(5430, min(5476, len(gs) + 1)):
        w(f"{j:4}: {gs[j-1].rstrip()[:120]}")
    w("")

# 4. The 'break' primitive: what marks an op as breakable?
#    Search vllm.worker + vllm.compilation for mark_step_boundary / break_graph / capture_break
w("=== grep for break primitives across vllm ===")
for pat in [r"mark_step_boundary", r"def capture_break", r"break_graph", r"is_breakable\b", r"_break_op", r"register_break"]:
    r = subprocess.run(["grep", "-rlnE", pat, "/usr/local/lib/python3.12/dist-packages/vllm/"],
                       capture_output=True, text=True)
    w(f"--- '{pat}' -> {len(r.stdout.split())} files ---")
    for cf in r.stdout.split()[:8]:
        w(f"    {cf}")
w("")

# 5. Does vkernels VkernelFusedExperts.apply carry ANY break marker?
#    (sitecustomize.py is what we control; vkernels is the installed pkg)
VK = None
for cand in [
    "/usr/local/lib/python3.12/dist-packages/vkernels/experts.py",
    "/usr/local/lib/python3.12/dist-packages/vkernels/moe/fused_experts.py",
]:
    if os.path.exists(cand):
        VK = cand; break
if not VK:
    # locate
    r = subprocess.run(["find", "/usr/local/lib/python3.12/dist-packages/vkernels", "-name", "*.py"],
                       capture_output=True, text=True)
    w(f"=== vkernels files ===\n{r.stdout[:1500]}")
if VK:
    w(f"\n=== {VK}: 'def apply' + any break/marker markers ===")
    vks = open(VK).read().splitlines()
    for i, l in enumerate(vks, 1):
        if "def apply" in l or "mark_step" in l or "break" in l.lower() or "cudagraph" in l.lower() or "record_function" in l:
            w(f"{i:4}: {l.rstrip()[:120]}")
w("")

# 6. sitecustomize.py: show the _profiled_apply wrapper context
SC = "/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/kimi-k3-vllm/sitecustomize.py"
if os.path.exists(SC):
    scs = open(SC).read().splitlines()
    w("=== sitecustomize.py: VkernelFusedExperts.apply / _profiled_apply / break mentions ===")
    for i, l in enumerate(scs, 1):
        if "apply" in l and ("def " in l or "profiled" in l.lower() or "Vkernel" in l) or "break" in l.lower() or "cudagraph" in l.lower() or "mark_step" in l:
            w(f"{i:4}: {l.rstrip()[:120]}")

w("\nDONE")
f.close(); print("WROTE", OUT, flush=True)
