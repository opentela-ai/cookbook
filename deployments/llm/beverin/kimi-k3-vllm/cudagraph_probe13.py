#!/usr/bin/env python3
"""Probe v13: FINAL risk checks. NO vllm import."""
import os, subprocess
OUT = os.environ.get("OUT", "/tmp/k3_cm13.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)
VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
K3 = os.environ.get("K3", "/capstor/scratch/cscs/xyao/k3-vllm")

# 1. END of vkernels_experts.py apply() — the torch.cuda.synchronize()
#    Find the file via $K3/home/pylib and the SCRIPT_DIR fallback
cands = []
for base in [f"{K3}/home/pylib", "/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/kimi-k3-vllm",
             "/usr/local/lib/python3.12/dist-packages"]:
    if os.path.isdir(base):
        cands += [p for p in subprocess.run(["find", base, "-name", "vkernels_experts.py", "-not", "-path", "*/__pycache__/*"],
                     capture_output=True, text=True, timeout=60).stdout.split() if p not in cands]
w(f"=== vkernels_experts.py candidates: {cands} ===")
for vf in cands:
    vs = open(vf).read().splitlines()
    # find def apply and show to end (look for synchronize)
    for i, l in enumerate(vs, 1):
        if "def apply(" in l:
            w(f"\n>>> {vf} apply {i}-{min(i+90,len(vs))} (looking for sync at end):")
            for j in range(i, min(i + 91, len(vs) + 1)):
                w(f"{j:4}: {vs[j-1].rstrip()[:150]}")
            break
    # all sync mentions
    w(f"\n>>> {vf}: ALL sync / synchronize / wait mentions:")
    for i, l in enumerate(vs, 1):
        if "sync" in l.lower() or "synchronize" in l.lower() or "wait" in l.lower() and "stream" in l.lower():
            w(f"{i:4}: {l.rstrip()[:150]}")
w("")

# 2. gpu_model_runner install site for BreakableCUDAGraphWrapper
G = f"{VLLM}/v1/worker/gpu_model_runner.py"
gs = open(G).read().splitlines()
w("=== gpu_model_runner.py 5415-5475 (BreakableCUDAGraphWrapper install site) ===")
for i in range(5415, min(5476, len(gs) + 1)):
    w(f"{i:4}: {gs[i-1].rstrip()[:150]}")
w("")

# Also: where is BreakableCUDAGraphWrapper CONSTRUCTED (not just imported/isinstance)?
w("=== BreakableCUDAGraphWrapper( construction sites across vllm ===")
r = subprocess.run(["grep", "-rnE", "BreakableCUDAGraphWrapper\(|BreakableCUDAGraphWrapper\b",
                    f"{VLLM}/v1/", f"{VLLM}/worker/"], capture_output=True, text=True)
for line in r.stdout.splitlines()[:30]:
    w(f"  {line[:170]}")
w("")

# 3. caller of set_splitting_ops_for_v1 (does it run with mode=NONE?)
w("=== caller of set_splitting_ops_for_v1 ===")
r = subprocess.run(["grep", "-rn", "set_splitting_ops_for_v1", f"{VLLM}/"], capture_output=True, text=True)
for line in r.stdout.splitlines()[:20]:
    w(f"  {line[:170]}")
w("")

# 4. ENFORCE_EAGER handling in gpu_model_runner (does --enforce-eager force mode=NONE which we already have?)
w("=== enforce_eager / enforce eager in gpu_model_runner ===")
r = subprocess.run(["grep", "-nE", "enforce_eager|enforce.eager", f"{VLLM}/v1/worker/gpu_model_runner.py"],
                   capture_output=True, text=True)
for line in r.stdout.splitlines()[:15]:
    w(f"  {line[:170]}")

w("\nDONE")
f.close(); print("WROTE", OUT, flush=True)
