#!/usr/bin/env python3
"""Probe vLLM source (inside kimi-k3-vllm container) for the cudagraph gate.

Dumps to $OUT everything needed to design a local @support_torch_compile +
PIECEWISE cudagraph patch for Kimi-K3:
  1. KimiK3ForConditionalGeneration: location, decorators, base, forward,
     attention + MoE module/layer names (splitting_ops candidates)
  2. vllm/config/vllm.py: the K3 auto-breakable + enforce-eager + the
     support_torch_compile WARN (~lines 1180-1215, 2400-2415)
  3. vllm/compilation.py: _attention_ops, splitting ops, support_torch_compile
     registration, has_full_cudagraphs, the cudagraph_mode enum
  4. vllm/v1/worker/gpu_model_runner.py: the cudagraph install + capture path
  5. The PP recv_object location (vllm/v1/ or vllm/distributed/)
"""
import os
import inspect
import importlib

OUT = os.environ.get("OUT", "/tmp/k3_cudagraph_probe.txt")
f = open(OUT, "w")


def w(*a):
    print(*a, file=f, flush=True)


def section(t):
    w("\n" + "=" * 78)
    w(t)
    w("=" * 78)


# 1. K3 model ----------------------------------------------------------------------
section("1. KimiK3 model")
try:
    import vllm.model_executor.models.kimi_k3 as m
    from vllm.model_executor.models.kimi_k3 import KimiK3ForConditionalGeneration as K3

    w("FILE:", m.__file__)
    w("CLASS:", K3)
    w("MRO:", [c.__name__ for c in K3.__mro__])
    w("DECORATORS / __dict__ keys w/ support_torch_compile:")
    for k in ("support_torch_compile", "_supports_torch_compile",
              "torch_compile", "is_fully_aligned"):
        w(f"  attr {k!r}:", getattr(K3, k, "<<missing>>"))
    # the decorator registry
    try:
        from vllm.compilation.wrapper import supports_torch_compile as stc
        w("supports_torch_compile src:")
        w(inspect.getsource(stc))
    except Exception as e:
        w("supports_torch_compile import failed:", repr(e))
    # all nn.Module subclasses defined in this file (attention + moe)
    w("\n-- modules defined in kimi_k3.py --")
    for name, obj in inspect.getmembers(m, inspect.isclass):
        try:
            if obj.__module__ == m.__name__ and issubclass(obj, __import__("torch").nn.Module):
                w(f"  {name}: {obj.__mro__[1].__name__ if len(obj.__mro__)>1 else '?'}")
        except Exception:
            pass
    # forward signature
    w("\n-- KimiK3 forward signature --")
    try:
        w(str(inspect.signature(K3.forward)))
    except Exception as e:
        w("sig fail:", repr(e))
except Exception as e:
    import traceback
    w("K3 probe failed:")
    traceback.print_exc(file=f)

# 2. config/vllm.py: K3 breakable + enforce-eager + support_torch_compile WARN
section("2. vllm/config/vllm.py (K3 cudagraph gate)")
try:
    import vllm.config.vllm as cfg
    src = inspect.getsource(cfg)
    w("FILE:", cfg.__file__)
    # find K3 / breakable / enforce_eager / support_torch_compile mentions
    import re
    for kw in ("VLLM_USE_BREAKABLE_CUDAGRAPH", "KimiK3", "enforce_eager",
               "does not support it", "num_models_seen",
               "compilation_counter", "cudagraph_mode"):
        w(f"\n--- mentions of {kw!r} ---")
        for i, line in enumerate(src.splitlines(), 1):
            if kw in line:
                w(f"  {i}: {line.rstrip()[:120]}")
except Exception as e:
    import traceback
    w("config probe failed:"); traceback.print_exc(file=f)

# 3. compilation: _attention_ops, splitting, support_torch_compile, has_full
section("3. vllm/compilation (attention_ops / splitting / has_full_cudagraphs)")
try:
    import vllm.compilation as comp
    w("FILE:", comp.__file__)
    w("\n-- _attention_ops --")
    try:
        w(repr(comp._attention_ops))
    except Exception as e:
        w("_attention_ops err:", repr(e))
    w("\n-- top-level names with 'split','compile','cudagraph','support' --")
    for n in dir(comp):
        if any(s in n.lower() for s in ("split", "compile", "cudagraph", "support", "attention")):
            w(f"  {n}")
    # the support_torch_compile decorator + compilation_counter
    try:
        from vllm.compilation.wrapper import supports_torch_compile
        w("\n-- supports_torch_compile source --")
        w(inspect.getsource(supports_torch_compile))
    except Exception as e:
        w("stc source err:", repr(e))
    try:
        from vllm.compilation.base import CompilationCounter
        w("\n-- CompilationCounter --")
        w(inspect.getsource(CompilationCounter))
    except Exception as e:
        w("CompilationCounter err:", repr(e))
    # has_full_cudagraphs
    try:
        from vllm.config import CudagraphMode
        w("\n-- CudagraphMode enum --")
        w([x for x in dir(CudagraphMode) if not x.startswith("_")])
        w("has_full_cudagraphs?", hasattr(CudagraphMode, "has_full_cudagraphs"))
    except Exception as e:
        w("CudagraphMode err:", repr(e))
except Exception as e:
    import traceback
    w("comp probe failed:"); traceback.print_exc(file=f)

# 4. gpu_model_runner: cudagraph install + capture path
section("4. gpu_model_runner (cudagraph install/capture)")
try:
    import vllm.v1.worker.gpu_model_runner as gmr
    src = inspect.getsource(gmr)
    w("FILE:", gmr.__file__)
    for kw in ("is_breakable_cudagraph_enabled", "CUDAGraphWrapper",
               "BreakableCUDAGraphWrapper", "capture_begin", "capture_end",
               "defer_decode", "has_full_cudagraphs", "_capture",
               "compilation_config.mode", "VLLM_COMPILE"):
        w(f"\n--- mentions of {kw!r} (first 8) ---")
        c = 0
        for i, line in enumerate(src.splitlines(), 1):
            if kw in line and c < 8:
                w(f"  {i}: {line.rstrip()[:120]}"); c += 1
except Exception as e:
    import traceback
    w("gmr probe failed:"); traceback.print_exc(file=f)

# 5. PP recv_object location
section("5. PP recv_object (gloo, the deadlock source)")
try:
    import vllm.v1.worker.gpu_worker as gw
    w("FILE:", gw.__file__)
    src = inspect.getsource(gw)
    for kw in ("recv_object", "send_object", "get_pp_group",
               "broadcast", "determine_num_available_blocks"):
        w(f"\n--- {kw!r} (first 6) ---")
        c = 0
        for i, line in enumerate(src.splitlines(), 1):
            if kw in line and c < 6:
                w(f"  {i}: {line.rstrip()[:120]}"); c += 1
except Exception as e:
    import traceback
    w("pp probe failed:"); traceback.print_exc(file=f)

w("\nDONE")
f.close()
print("WROTE", OUT, flush=True)
