#!/usr/bin/env python3
"""Probe v14: validate K3_BREAKABLE_PIECEWISE decoration (NO full serve).
Imports vllm + sitecustomize (which decorates VkernelFusedExperts.apply when
K3_BREAKABLE_PIECEWISE=1) and checks the breakable machinery is wired.
"""
import os, sys
OUT = os.environ.get("OUT", "/tmp/k3_cm14.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)

# Mirror sitecustomize's sys.path setup so vkernels_experts imports
_k3 = os.environ.get("K3", "")
_k3_pylib = os.path.join(_k3, "home/pylib") if _k3 else ""
if _k3_pylib and _k3_pylib not in sys.path:
    sys.path.insert(0, _k3_pylib)
# Also add the cookbook dir (sitecustomize.py's SCRIPT_DIR) so it imports
_cook = "/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/kimi-k3-vllm"
if _cook not in sys.path:
    sys.path.insert(0, _cook)

w(f"K3={_k3!r}  K3_pylib={_k3_pylib!r}")
w(f"K3_BREAKABLE_PIECEWISE={os.environ.get('K3_BREAKABLE_PIECEWISE')!r}")
w(f"VLLM_USE_BREAKABLE_CUDAGRAPH={os.environ.get('VLLM_USE_BREAKABLE_CUDAGRAPH')!r}")
w(f"VLLM_STEP_PROFILE_DIR={os.environ.get('VLLM_STEP_PROFILE_DIR')!r}")

# 1. import the breakable primitives
try:
    from vllm.compilation.breakable_cudagraph import (
        eager_break_during_capture, is_breakable_cudagraph_enabled,
        BreakableCUDAGraphCapture, BreakableCUDAGraphWrapper,
    )
    w(f"\n[1] import eager_break_during_capture: OK")
    w(f"    is_breakable_cudagraph_enabled()={is_breakable_cudagraph_enabled()}")
except Exception as e:
    import traceback; w(f"[1] FAILED: {e}"); traceback.print_exc(file=f)

# 2. import vkernels_experts directly (the class we decorate)
try:
    from vkernels_experts import VkernelFusedExperts, _find_libvkernels_hip
    _lib = _find_libvkernels_hip()
    w(f"\n[2] VkernelFusedExperts imported; libvkernels_hip={_lib}")
    w(f"    apply before sitecustomize: {VkernelFusedExperts.apply!r}")
    w(f"    apply __module__={getattr(VkernelFusedExperts.apply, '__module__', '?')}")
    w(f"    apply __name__={getattr(VkernelFusedExperts.apply, '__name__', '?')}")
    _pre = VkernelFusedExperts.apply
except Exception as e:
    import traceback; w(f"[2] FAILED: {e}"); traceback.print_exc(file=f)
    _pre = None

# 3. run sitecustomize (the real file) — this registers the backend AND
#    decorates apply when K3_BREAKABLE_PIECEWISE=1 (our env). It will also
#    print its own [sitecustomize] ... lines to stdout (captured here too).
w("\n[3] running sitecustomize (real file) ...")
try:
    import sitecustomize as _sc
    w(f"[3] sitecustomize imported from {_sc.__file__}")
except SystemExit as e:
    w(f"[3] sitecustomize SystemExit: {e}")
except Exception as e:
    import traceback; w(f"[3] sitecustomize FAILED: {e}"); traceback.print_exc(file=f)

# 4. inspect VkernelFusedExperts.apply AFTER sitecustomize
try:
    _post = VkernelFusedExperts.apply
    w(f"\n[4] apply AFTER sitecustomize: {_post!r}")
    w(f"    apply __name__={getattr(_post, '__name__', '?')}")
    w(f"    changed: {_pre is not _post}")
    # The eager_break wrapper sets __wrapped__ via functools.wraps(fn) at line 93.
    _wrapped = getattr(_post, "__wrapped__", None)
    w(f"    __wrapped__ present: {_wrapped is not None}")
    if _wrapped is not None:
        w(f"    __wrapped__ = {_wrapped!r}")
    # Look for breakable markers in the closure
    try:
        cl = _post.__closure__ or []
        w(f"    closure cells: {len(cl)}")
    except Exception as e:
        w(f"    closure err: {e}")
except Exception as e:
    import traceback; w(f"[4] FAILED: {e}"); traceback.print_exc(file=f)

# 5. functional sanity: call the decorated apply OUTSIDE any capture context.
#    eager_break_during_capture (line 95-97): capture is None -> return fn(*args).
#    We can't call the real apply (needs tensors+lib), but we can verify the
#    "no capture -> passthrough" branch by calling with a sentinel instance
#    and catching the early failure (apply will try _get_lib() etc.).
w("\n[5] passthrough sanity (no capture context) ...")
if _post is not None:
    try:
        _post(object())  # should reach _get_lib() then fail -> passthrough confirmed
        w("[5] returned without error (unexpected for real apply)")
    except Exception as e:
        msg = str(e)
        w(f"[5] reached real apply and failed as expected: {type(e).__name__}: {msg[:120]}")
        if "capture" in msg.lower() or "BreakableCUDAGraphCapture" in msg:
            w("[5] !! failure was capture-related, not passthrough")
else:
    w("[5] skipped (apply is None)")

w("\nDONE")
f.close(); print("WROTE", OUT, flush=True)
