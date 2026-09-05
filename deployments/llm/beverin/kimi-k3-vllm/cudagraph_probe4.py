#!/usr/bin/env python3
"""Probe v4: @eager_break_during_capture placement + PP recv in model + version."""
import os, subprocess
OUT = os.environ.get("OUT", "/tmp/k3_ebreak.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)
V = "/usr/local/lib/python3.12/dist-packages/vllm"

w("=== vLLM version ===")
try:
    import vllm
    w("vllm.__version__:", getattr(vllm, "__version__", "?"))
    import importlib.metadata as m
    w("metadata version:", m.version("vllm"))
except Exception as e:
    w("ver err:", repr(e))

w("\n=== @eager_break_during_capture usages (break-point ops) ===")
r = subprocess.run(["grep", "-rn", "eager_break_during_capture", V],
                   capture_output=True, text=True)
for line in r.stdout.splitlines():
    if "def eager_break" in line or "__pycache__" in line \
            or "import " in line or "breakable_cudagraph.py" in line:
        continue
    w("  " + line[:150])

w("\n=== is unified_mla / unified / deepseek_v4 attention decorated? ===")
r = subprocess.run(["grep", "-rn", "-B4",
                    "def unified_mla_attention_with_output\|def unified_attention_with_output\|def deepseek_v4_attention",
                    V + "/model_executor"], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if "eager_break" in line or "def unified" in line or "def deepseek" in line:
        w("  " + line[:150])

w("\n=== K3 model file: attention + MoE layer names (for break-point coverage) ===")
import glob
for p in glob.glob(V + "/model_executor/models/kimi*.py") \
        + glob.glob(V + "/model_executor/models/deepseek_v4*.py"):
    w(f"--- {os.path.basename(p)} ---")
    src = open(p).read().splitlines()
    for i, line in enumerate(src, 1):
        if any(k in line for k in ("self.attn", "self.mlp", "self.moe",
                                    "self.feed_forward", "MoE(", "MLP(",
                                    "unified_mla", "unified_attention",
                                    "deepseek_v4_attention", "self.experts")):
            w(f"  {i:4}: {line.rstrip()[:130]}")

w("\n=== PP recv INSIDE any model forward? (would break cudagraph) ===")
r = subprocess.run(["grep", "-rln", "get_pp_group\|irecv\|recv_object\|recv_tensor",
                    V + "/model_executor/models"], capture_output=True, text=True)
w("models referencing pp recv: " + (r.stdout.strip()[:500] or "(none)"))
# the v1 worker handles recv outside forward -- confirm
r2 = subprocess.run(["grep", "-rn", "def execute_model",
                     V + "/v1/worker/gpu_worker.py"], capture_output=True, text=True)
w("gpu_worker execute_model: " + r2.stdout.strip()[:200])

w("\n=== BreakableCUDAGraphWrapper: does it gate on runtime_mode for prefill? ===")
r = subprocess.run(["grep", "-n", "runtime_mode\|prefill\|decode\|BatchDescriptor",
                    V + "/compilation/breakable_cudagraph.py"], capture_output=True, text=True)
for line in r.stdout.splitlines()[:20]:
    w("  " + line[:130])

w("\nDONE")
f.close(); print("WROTE", OUT, flush=True)
