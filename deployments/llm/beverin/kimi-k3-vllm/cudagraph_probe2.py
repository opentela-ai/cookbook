#!/usr/bin/env python3
"""Probe v2: dump the exact gate code for the K3 cudagraph patch."""
import os, inspect, importlib

OUT = os.environ.get("OUT", "/tmp/k3_cgprobe2.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)
def sec(t): w("\n" + "=" * 78); w(t); w("=" * 78)

# 1. Find the REAL KimiK3 module + supports_torch_compile decorator
sec("1. KimiK3 model registry + supports_torch_compile")
try:
    import vllm.model_executor.models as models
    # find the module containing KimiK3ForConditionalGeneration
    import importlib.util, glob
    base = os.path.dirname(models.__file__)
    found = []
    for p in glob.glob(os.path.join(base, "**", "*.py"), recursive=True):
        try:
            txt = open(p).read()
        except Exception:
            continue
        if "class KimiK3ForConditionalGeneration" in txt:
            found.append(p)
    w("KimiK3 files:", found)
    for pf in found:
        w("\n--- imports + class def (first 80 lines) of %s ---" % pf)
        for i, line in enumerate(open(pf).read().splitlines()[:80], 1):
            w(f"{i:3}: {line.rstrip()[:118]}")
        # decorators on the class
        src = open(pf).read()
        idx = src.find("class KimiK3ForConditionalGeneration")
        if idx >= 0:
            # show 25 lines before the class (decorators) + 30 after
            pre = src.rsplit("\n", 0)[0]
            start = max(0, idx - 800)
            w("\n--- around class def ---")
            w(src[start:idx + 1200])
    # supports_torch_compile: find it
    w("\n--- locate supports_torch_compile ---")
    import subprocess
    r = subprocess.run(["grep", "-rn", "def supports_torch_compile",
                        "/usr/local/lib/python3.12/dist-packages/vllm"],
                       capture_output=True, text=True)
    w(r.stdout or "(none)")
    r2 = subprocess.run(["grep", "-rln", "compilation_counter", "/usr/local/lib/python3.12/dist-packages/vllm/compilation"],
                        capture_output=True, text=True)
    w("\nfiles with compilation_counter:")
    w(r2.stdout or "(none)")
except Exception as e:
    import traceback; w("FAIL:"); traceback.print_exc(file=f)

# 2. config/vllm.py: the K3 auto-breakable block + the compile gate
sec("2. vllm/config/vllm.py lines 1180-1245 (K3 block) + 2380-2420 (gate)")
try:
    import vllm.config.vllm as cfg
    src = open(cfg.__file__).read().splitlines()
    for lo, hi in [(1180, 1245), (2375, 2420), (1360, 1440), (850, 870)]:
        w(f"\n--- lines {lo}-{hi} ---")
        for i in range(lo, min(hi, len(src)) + 1):
            w(f"{i:4}: {src[i-1].rstrip()[:118]}")
except Exception as e:
    import traceback; w("FAIL:"); traceback.print_exc(file=f)

# 3. compilation: _attention_ops + supports_torch_compile + CudagraphMode
sec("3. compilation internals")
try:
    import vllm.compilation.cuda_graph as cg
    w("cuda_graph FILE:", cg.__file__)
    # BreakableCUDAGraphWrapper
    try:
        from vllm.compilation.cuda_graph import BreakableCUDAGraphWrapper as BC
        w("\n--- BreakableCUDAGraphWrapper (first 90 lines) ---")
        src = inspect.getsource(BC).splitlines()
        for i, line in enumerate(src[:90], 1):
            w(f"{i:3}: {line.rstrip()[:118]}")
    except Exception as e:
        w("BC err:", repr(e))
    # splitting / _attention_ops
    import subprocess
    for kw in ["_attention_ops", "splitting_ops", "def supports_torch_compile",
               "class CudagraphMode", "def has_full_cudagraphs",
               "def has_piecewise_cudagraphs", "def requires_piecewise"]:
        w(f"\n--- grep '{kw}' ---")
        r = subprocess.run(["grep", "-rn", kw, "/usr/local/lib/python3.12/dist-packages/vllm"],
                           capture_output=True, text=True)
        for line in (r.stdout or "").splitlines()[:12]:
            w("  " + line[:116])
except Exception as e:
    import traceback; w("FAIL:"); traceback.print_exc(file=f)

# 4. gpu_model_runner install switch (5430-5475)
sec("4. gpu_model_runner install switch (5430-5475)")
try:
    import vllm.v1.worker.gpu_model_runner as gmr
    src = open(gmr.__file__).read().splitlines()
    for i in range(5430, min(5475, len(src)) + 1):
        w(f"{i:4}: {src[i-1].rstrip()[:118]}")
except Exception as e:
    import traceback; w("FAIL:"); traceback.print_exc(file=f)

# 5. PP recv/send in execute_model (gpu_worker.py 1040-1095)
sec("5. PP recv/send in gpu_worker (1040-1095)")
try:
    import vllm.v1.worker.gpu_worker as gw
    src = open(gw.__file__).read().splitlines()
    for i in range(1040, min(1095, len(src)) + 1):
        w(f"{i:4}: {src[i-1].rstrip()[:118]}")
except Exception as e:
    import traceback; w("FAIL:"); traceback.print_exc(file=f)

w("\nDONE")
f.close()
print("WROTE", OUT, flush=True)
