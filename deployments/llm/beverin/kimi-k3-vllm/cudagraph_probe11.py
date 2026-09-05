#!/usr/bin/env python3
"""Probe v11: FINAL — breakable capture/split mechanism + MoE op registration. NO vllm import."""
import os, subprocess
OUT = os.environ.get("OUT", "/tmp/k3_cm11.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)
def show(path, a, b, label=None):
    if not os.path.exists(path): w(f"  [missing] {path}"); return
    s = open(path).read().splitlines()
    w(f"=== {label or path} {a}-{b} ===")
    for i in range(a, min(b + 1, len(s) + 1)):
        w(f"{i:4}: {s[i-1].rstrip()[:140]}")
    w("")

VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"

# 1. config/compilation.py: the IF that adds unified_kv splitting_ops + surrounding
show(f"{VLLM}/config/compilation.py", 1100, 1180, "compilation.py (unified_kv splitting_ops IF)")

# 2. breakable_cudagraph.py: is_breakable_cudagraph_enabled + _capture + _replay + split mechanism
BC = f"{VLLM}/compilation/breakable_cudagraph.py"
if os.path.exists(BC):
    bcs = open(BC).read().splitlines()
    w("=== breakable_cudagraph.py: is_breakable_cudagraph_enabled + key fns ===")
    for i, l in enumerate(bcs, 1):
        if ("def is_breakable_cudagraph_enabled" in l or "def _capture" in l
            or "def _replay" in l or "splitting_ops" in l or "record_function" in l
            or "BreakableCUDAGraphCapture" in l or "def capture" in l or "split" in l.lower() and "def " in l):
            w(f"{i:4}: {l.rstrip()[:140]}")
    w("")
    # show is_breakable_cudagraph_enabled body
    for i, l in enumerate(bcs, 1):
        if "def is_breakable_cudagraph_enabled" in l:
            for j in range(i, min(i + 22, len(bcs) + 1)):
                w(f"{j:4}: {bcs[j-1].rstrip()[:140]}")
            break
    w("")
    # show _capture body
    for i, l in enumerate(bcs, 1):
        if "def _capture(" in l:
            for j in range(i, min(i + 60, len(bcs) + 1)):
                w(f"{j:4}: {bcs[j-1].rstrip()[:140]}")
            break
w("")

# 3. How splitting_ops is matched against op names during capture (the break trigger)
w("=== grep: splitting op registration + matching across vllm/compilation ===")
for pat in [r"register_splitting_op", r"SPLITTING_OPS", r"splitting_op_name", r"torch\.library", r"custom_op", r"def.*split.*op"]:
    r = subprocess.run(["grep", "-rnE", pat, f"{VLLM}/compilation/"],
                       capture_output=True, text=True)
    w(f"--- '{pat}' ({len(r.stdout.splitlines())} hits) ---")
    for line in r.stdout.splitlines()[:10]:
        w(f"  {line[:150]}")
w("")

# 4. Is VkernelFusedExperts a torch custom op? Check vkernels_experts.py
w("=== locate vkernels_experts.py + VkernelFusedExperts torch-op-ness ===")
r = subprocess.run(["find", "/capstor/scratch/cscs/xyao/vkernels", "-name", "vkernels_experts.py"],
                   capture_output=True, text=True)
for vf in r.stdout.split():
    w(f"--- {vf} ---")
    vs = open(vf).read().splitlines()
    for i, l in enumerate(vs, 1):
        if ("class VkernelFusedExperts" in l or "def apply" in l or "torch.library" in l
            or "custom_op" in l or "register" in l.lower() or "op_name" in l
            or "namespace" in l or "@support" in l or "def __call__" in l):
            w(f"{i:4}: {l.rstrip()[:140]}")
    # show class def + apply
    for i, l in enumerate(vs, 1):
        if "class VkernelFusedExperts" in l:
            for j in range(i, min(i + 50, len(vs) + 1)):
                w(f"{j:4}: {vs[j-1].rstrip()[:140]}")
            break
w("")

# 5. The model's MoE: is FusedMoE a registered torch op? (vllm/models/kimi_k3)
r = subprocess.run(["find", f"{VLLM}/models/kimi_k3", "-name", "*.py"], capture_output=True, text=True)
w(f"=== kimi_k3 model files ===\n{r.stdout.strip()}")
KM = None
for cand in r.stdout.split():
    if cand.endswith("model.py") or "moe" in cand.lower():
        KM = cand
if KM:
    w(f"\n=== {KM}: MoE op registration / torch op / splitting ===")
    kms = open(KM).read().splitlines()
    for i, l in enumerate(kms, 1):
        if ("torch.library" in l or "custom_op" in l or "splitting" in l.lower()
            or "FusedMoE" in l or "def apply" in l or "op_name" in l
            or "@support_torch_compile" in l or "support_torch_compile" in l):
            w(f"{i:4}: {l.rstrip()[:140]}")
w("")

# 6. support_torch_compile: which models have it (to confirm K3 lacks it)
r = subprocess.run(["grep", "-rln", "support_torch_compile", f"{VLLM}/models/"],
                   capture_output=True, text=True)
w(f"=== models with support_torch_compile ({len(r.stdout.split())} files) ===")
for cf in r.stdout.split()[:30]:
    w(f"  {cf.replace(VLLM+'/models/','')}")
# is kimi_k3 among them?
kk = [c for c in r.stdout.split() if "kimi_k3" in c]
w(f"\n=== kimi_k3 files with support_torch_compile: {kk or 'NONE'} ===")

w("\nDONE")
f.close(); print("WROTE", OUT, flush=True)
