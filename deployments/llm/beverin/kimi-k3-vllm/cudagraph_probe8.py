#!/usr/bin/env python3
"""Probe v8: auto-enable condition + runtime wrapper + requires_piecewise semantics."""
import os
OUT = os.environ.get("OUT", "/tmp/k3_cm8.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)

# 1. config/vllm.py auto-enable block
F = "/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py"
src = open(F).read().splitlines()
n = len(src)
w("=== config/vllm.py 1200-1245 (VLLM_USE_BREAKABLE auto-enable) ===")
for i in range(1200, min(1246, n) + 1):
    w(f"{i:4}: {src[i-1].rstrip()[:120]}")

# 2. CUDAGraphMode enum + requires_piecewise_compilation
import importlib
try:
    from vllm.config.compilation import CUDAGraphMode
    w("\n=== CUDAGraphMode members ===")
    for m in CUDAGraphMode:
        w(f"  {m.name} = {m.value}")
    w("\n=== requires_piecewise_compilation / has_full / has_piecewise ===")
    import inspect
    for nm in ("requires_piecewise_compilation", "has_full_cudagraphs", "has_piecewise_cudagraphs"):
        try:
            w(f"--- {nm} ---")
            w(inspect.getsource(getattr(CUDAGraphMode, nm)))
        except Exception as e:
            w(f"  ERR {e}")
except Exception as e:
    w(f"CUDAGraphMode import ERR: {e}")

# 3. runtime wrapper: gpu_model_runner breakable path + BreakableCUDAGraphWrapper
G = "/usr/local/lib/python3.12/dist-packages/vllm/worker/gpu_model_runner.py"
g = open(G).read().splitlines()
w("\n=== gpu_model_runner.py: is_breakable_cudagraph_enabled + BreakableCUDAGraphWrapper mentions ===")
hits = [(i, l) for i, l in enumerate(g, 1) if "is_breakable" in l or "BreakableCUDAGraphWrapper" in l or "CUDAGraphWrapper" in l]
for i, l in hits[:40]:
    w(f"{i:4}: {l.rstrip()[:120]}")
if len(hits) > 40:
    w(f"  ... +{len(hits)-40} more")

# 4. BreakableCUDAGraphWrapper source location
import subprocess
r = subprocess.run(["grep", "-rln", "class BreakableCUDAGraphWrapper", "/usr/local/lib/python3.12/dist-packages/vllm/"],
                   capture_output=True, text=True)
w(f"\n=== BreakableCUDAGraphWrapper class file ===\n{r.stdout.strip()}")
for cf in r.stdout.split():
    cs = open(cf).read().splitlines()
    # find class def + its __init__ and apply/execute
    for i, l in enumerate(cs, 1):
        if "class BreakableCUDAGraphWrapper" in l or "def __init__" in l and i < 200:
            pass
    # print class def line and next 60
    for i, l in enumerate(cs, 1):
        if "class BreakableCUDAGraphWrapper" in l:
            w(f"\n--- {cf} (from class line {i}) ---")
            for j in range(i, min(i + 70, len(cs) + 1)):
                w(f"{j:4}: {cs[j-1].rstrip()[:120]}")
            break

# 5. vllm.worker.cudagraph_utils is_breakable_cudagraph_enabled
CU = "/usr/local/lib/python3.12/dist-packages/vllm/worker/cudagraph_utils.py"
import os as _os
if _os.path.exists(CU):
    cu = open(CU).read().splitlines()
    w("\n=== cudagraph_utils.py: is_breakable_cudagraph_enabled ===")
    for i, l in enumerate(cu, 1):
        if "is_breakable_cudagraph_enabled" in l and "def " in l:
            for j in range(i, min(i + 25, len(cu) + 1)):
                w(f"{j:4}: {cu[j-1].rstrip()[:120]}")
            break

w("\nDONE")
f.close(); print("WROTE", OUT, flush=True)
