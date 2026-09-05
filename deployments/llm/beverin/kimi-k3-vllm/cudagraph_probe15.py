#!/usr/bin/env python3
"""Probe v15: read the ACTUAL vLLM breakable auto-default + install logic
so the sbatch --compilation-config is provably correct (no guessing)."""
import os, sys, inspect
OUT = os.environ.get("OUT", "/tmp/k3_cm15.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)

try:
    import vllm
    from vllm.config import vllm as _vmod
    src = inspect.getsource(_vmod)
    lines = src.splitlines()
    w("=== vllm/config/vllm.py: lines containing BREAKABLE/None/auto (context) ===")
    for i, ln in enumerate(lines, 1):
        if any(k in ln for k in ("BREAKABLE","KimiK3","breakable","set_mode","auto",)):
            w(f"{i:5d}: {ln}")
    w("\n=== region 1195-1250 (the auto-default block) ===")
    for i in range(1195, min(1250, len(lines))+1):
        w(f"{i:5d}: {lines[i-1]}")
except Exception as e:
    import traceback; w(f"vllm.config FAILED: {e}"); traceback.print_exc(file=f)

try:
    from vllm.compilation import breakable_cudagraph as _bc
    w("\n=== breakable_cudagraph.is_breakable_cudagraph_enabled (source) ===")
    w(inspect.getsource(_bc.is_breakable_cudagraph_enabled))
    w("=== eager_break_during_capture (head, lines 85-105) ===")
    esrc = inspect.getsource(_bc.eager_break_during_capture).splitlines()
    for i, ln in enumerate(esrc[:25], 1):
        w(f"  {i:2d}: {ln}")
except Exception as e:
    import traceback; w(f"breakable FAILED: {e}"); traceback.print_exc(file=f)

try:
    from vllm.v1.worker import gpu_model_runner as _g
    gsrc = inspect.getsource(_g).splitlines()
    w("\n=== gpu_model_runner: BreakableCUDAGraphWrapper install site (grep) ===")
    for i, ln in enumerate(gsrc, 1):
        if "BreakableCUDAGraphWrapper" in ln or "is_breakable_cudagraph_enabled" in ln:
            w(f"{i:5d}: {ln}")
    w("=== region around the first install (context) ===")
    # find first BreakableCUDAGraphWrapper
    for i, ln in enumerate(gsrc, 1):
        if "BreakableCUDAGraphWrapper" in ln:
            for j in range(max(1,i-6), min(len(gsrc),i+8)+1):
                w(f"{j:5d}: {gsrc[j-1]}")
            break
except Exception as e:
    import traceback; w(f"gmr FAILED: {e}"); traceback.print_exc(file=f)

w("\nDONE")
f.close(); print("WROTE", OUT, flush=True)
