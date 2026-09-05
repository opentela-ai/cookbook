#!/usr/bin/env python3
"""Probe v12: the 3 make-or-break correctness checks. NO vllm import."""
import os, subprocess
OUT = os.environ.get("OUT", "/tmp/k3_cm12.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)
VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
VK = os.environ.get("VKERNELS_DIR", "/capstor/scratch/cscs/xyao/vkernels")

# A. eager_break_during_capture FULL body (does it skip in FULL mode?)
BC = f"{VLLM}/compilation/breakable_cudagraph.py"
bcs = open(BC).read().splitlines()
w("=== A. eager_break_during_capture FULL body (lines 59-235) ===")
for i in range(59, min(236, len(bcs) + 1)):
    w(f"{i:4}: {bcs[i-1].rstrip()[:150]}")
w("")

# C. gpu_model_runner: BreakableCUDAGraphWrapper install + cudagraph_runtime_mode set
G = f"{VLLM}/v1/worker/gpu_model_runner.py"
gs = open(G).read().splitlines()
w("=== C. gpu_model_runner: BreakableCUDAGraphWrapper + cudagraph_runtime_mode ===")
for i, l in enumerate(gs, 1):
    if ("BreakableCUDAGraphWrapper" in l or "is_breakable_cudagraph_enabled" in l
        or "cudagraph_runtime_mode" in l or "cudagraph_mode" in l and "runtime" in l.lower()):
        lo = max(1, i - 2); hi = min(len(gs) + 1, i + 3)
        for j in range(lo, hi):
            w(f"{j:4}: {gs[j-1].rstrip()[:150]}")
        w("  --")
        if i > 30: break
w("")

# Also: where is cudagraph_runtime_mode SET (not read) in the forward context?
w("=== cudagraph_runtime_mode SET (set_*runtime_mode) across vllm ===")
r = subprocess.run(["grep", "-rnE", "cudagraph_runtime_mode\s*=|set.*cudagraph_runtime_mode",
                    f"{VLLM}/v1/", f"{VLLM}/compilation/"], capture_output=True, text=True)
for line in r.stdout.splitlines()[:25]:
    w(f"  {line[:160]}")
w("")

# B. VkernelFusedExperts.apply signature + in-place output?
w("=== B. locate vkernels_experts.py (search VKERNELS_DIR + site-packages + SCRIPT_DIR) ===")
for d in [VK, f"{VLLM}", "/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/kimi-k3-vllm",
          "/usr/local/lib/python3.12/dist-packages"]:
    r = subprocess.run(["find", d, "-name", "vkernels_experts.py", "-not", "-path", "*/__pycache__/*"],
                       capture_output=True, text=True, timeout=60)
    for p in r.stdout.split():
        w(f"--- {p} ---")
        vs = open(p).read().splitlines()
        # class def + __init__ + apply + forward + return statements + op output buffer
        for i, l in enumerate(vs, 1):
            if ("class VkernelFusedExperts" in l or "def apply" in l or "def forward" in l
                or "def __init__" in l or "out" in l and ("=" in l and ("Tensor" in l or "tensor" in l.lower()))
                or "return" in l and i < 400):
                w(f"{i:4}: {l.rstrip()[:150]}")
        # show the apply method fully
        for i, l in enumerate(vs, 1):
            if "def apply" in l:
                w(f"\n>>> {p} apply (full) {i}-{min(i+55,len(vs))}:")
                for j in range(i, min(i + 56, len(vs) + 1)):
                    w(f"{j:4}: {vs[j-1].rstrip()[:150]}")
                break
w("")

# sitecustomize.py: VkernelFusedExperts import + _profiled_apply (lines 270-335)
SC = "/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/kimi-k3-vllm/sitecustomize.py"
if os.path.exists(SC):
    scs = open(SC).read().splitlines()
    w("=== sitecustomize.py 270-340 (VkernelFusedExperts registration + _profiled_apply) ===")
    for i in range(270, min(341, len(scs) + 1)):
        w(f"{i:4}: {scs[i-1].rstrip()[:150]}")
w("")

w("DONE")
f.close(); print("WROTE", OUT, flush=True)
