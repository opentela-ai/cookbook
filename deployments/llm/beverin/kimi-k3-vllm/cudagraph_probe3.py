#!/usr/bin/env python3
"""Probe v3: BreakableCUDAGraphWrapper + is_breakable_cudagraph_enabled source."""
import os, inspect, importlib, subprocess
OUT = os.environ.get("OUT", "/tmp/k3_cgprobe3.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)
def sec(t): w("\n" + "=" * 78); w(t); w("=" * 78)

# locate BreakableCUDAGraphWrapper + is_breakable_cudagraph_enabled
sec("0. locate breakable symbols")
r = subprocess.run(["grep", "-rln", "class BreakableCUDAGraphWrapper\|def is_breakable_cudagraph_enabled",
                    "/usr/local/lib/python3.12/dist-packages/vllm"], capture_output=True, text=True)
w(r.stdout or "(none)")
r = subprocess.run(["grep", "-rn", "def is_breakable_cudagraph_enabled",
                    "/usr/local/lib/python3.12/dist-packages/vllm"], capture_output=True, text=True)
w("is_breakable def:", r.stdout.strip()[:200] or "(none)")

# 1. is_breakable_cudagraph_enabled
sec("1. is_breakable_cudagraph_enabled")
try:
    from vllm.compilation.cuda_graph import is_breakable_cudagraph_enabled
    w(inspect.getsource(is_breakable_cudagraph_enabled))
except Exception as e:
    w("import1 err:", repr(e))
    try:
        from vllm.compilation.breakable_cuda_graph import is_breakable_cudagraph_enabled
        w(inspect.getsource(is_breakable_cudagraph_enabled))
    except Exception as e2:
        w("import2 err:", repr(e2))

# 2. BreakableCUDAGraphWrapper full source
sec("2. BreakableCUDAGraphWrapper")
try:
    mod = None
    for cand in ("vllm.compilation.cuda_graph", "vllm.compilation.breakable_cuda_graph"):
        try:
            mod = importlib.import_module(cand)
            if hasattr(mod, "BreakableCUDAGraphWrapper"):
                w("FOUND in", cand); break
        except Exception:
            pass
    BC = mod.BreakableCUDAGraphWrapper
    src = inspect.getsource(BC)
    w(f"({len(src.splitlines())} lines)")
    # print the class — focus on __init__, forward, capture, replay, split
    w(src[:9000])
    if len(src) > 9000:
        w("\n... (truncated; printing key methods) ...")
        for name in ("capture", "replay", "forward", "split", "_maybe",
                     "run_capture", "run_replay", "capture_begin", "capture_end"):
            try:
                m = getattr(BC, name)
                w(f"\n--- method {name} ---")
                w(inspect.getsource(m)[:2500])
            except Exception:
                pass
except Exception as e:
    import traceback; w("FAIL:"); traceback.print_exc(file=f)

# 3. The model_runner install context (5430-5475 already have; get is_breakable import)
sec("3. gpu_model_runner is_breakable import + cudagraph default")
try:
    import vllm.v1.worker.gpu_model_runner as gmr
    src = open(gmr.__file__).read().splitlines()
    # the import lines (20-35)
    w("--- imports 20-35 ---")
    for i in range(20, 36): w(f"{i:4}: {src[i-1].rstrip()[:118]}")
    # where cudagraph_mode default comes from + _set_cudagraph_sizes
    for kw in ("cudagraph_capture_sizes", "def _set_cudagraph_sizes",
               "default_cudagraph", "cudagraph_mode ="):
        w(f"\n--- {kw!r} (first 5) ---")
        c=0
        for i,line in enumerate(src,1):
            if kw in line and c<5: w(f"{i:4}: {line.rstrip()[:118]}"); c+=1
except Exception as e:
    import traceback; w("FAIL:"); traceback.print_exc(file=f)

# 4. config/compilation.py: _attention_ops + cudagraph_mode enum + splitting
sec("4. config/compilation.py (attention_ops, CudagraphMode, splitting)")
try:
    import vllm.config.compilation as cc
    src = open(cc.__file__).read().splitlines()
    for lo,hi in [(60,110),(740,800),(1120,1170)]:
        w(f"\n--- lines {lo}-{hi} ---")
        for i in range(lo, min(hi, len(src))+1):
            w(f"{i:4}: {src[i-1].rstrip()[:118]}")
except Exception as e:
    import traceback; w("FAIL:"); traceback.print_exc(file=f)

w("\nDONE")
f.close(); print("WROTE", OUT, flush=True)
