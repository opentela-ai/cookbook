#!/usr/bin/env python3
"""Preflight: verify the container + overlay can import GLM-5.3 model code."""
import importlib
import sys
import time

_t0 = time.time()
ok, fail = [], []


def chk(name):
    _s = time.time()
    try:
        importlib.import_module(name)
        ok.append(name)
        print(
            f"  [preflight t={time.time() - _t0:.1f}s] +{name} ({time.time() - _s:.1f}s)",
            flush=True,
        )
    except Exception as e:
        fail.append(f"{name}: {type(e).__name__}: {e}")
        print(
            f"  [preflight t={time.time() - _t0:.1f}s] FAIL {name} ({time.time() - _s:.1f}s): {e}",
            flush=True,
        )


chk("sglang")
chk("sglang.srt.configs.glm5_next")
chk("sglang.srt.models.glm5_next")
chk("sglang.srt.models.glm5_next_nextn")
chk("transformers.models.glm5_next")

# vkernels #60 bridge: the patched kpool module must be the one on sys.path
# (patches_full first) and the vkernels package must provide the dsa_kpool
# kernels — without them the SM80 gate falls back to the fp8e4nv Triton path
# that dies at JIT (job 82822).
try:
    import vkernels.kernels as _vkk

    _has_asm = hasattr(_vkk, "dsa_kpool_assemble")
    _has_dec = hasattr(_vkk, "dsa_kpool_decode_update")
    _core = "compiled" if getattr(_vkk, "_COMPILED", False) else "fallback"
    if _has_asm and _has_dec:
        ok.append(f"vkernels.kernels({_core})")
        print(
            f"  [preflight t={time.time() - _t0:.1f}s] +vkernels.kernels "
            f"backend={_core} file={getattr(_vkk, '__file__', '?')}",
            flush=True,
        )
    else:
        fail.append(f"vkernels.kernels({_core}): missing dsa_kpool entry points")
        print(
            f"  [preflight t={time.time() - _t0:.1f}s] FAIL vkernels.kernels({_core})",
            flush=True,
        )
except Exception as e:
    fail.append(f"vkernels.kernels: {type(e).__name__}: {e}")
    print(f"  [preflight t={time.time() - _t0:.1f}s] FAIL vkernels.kernels: {e}", flush=True)

try:
    import sglang.srt.layers.attention.dsa.kpool_fp8_index as _kp

    _src = getattr(_kp, "__file__", "")
    if "patches_full" in _src:
        _gate = "n/a (no cuda in preflight)"
        ok.append(f"patched_kpool({_src})")
        print(
            f"  [preflight t={time.time() - _t0:.1f}s] +patched kpool_fp8_index: {_src}",
            flush=True,
        )
        print(
            f"  [preflight t={time.time() - _t0:.1f}s]  vk entry points: "
            f"assemble={'set' if getattr(_kp, '_VK_DSA_KPOOL_ASSEMBLE', None) else 'MISSING'} "
            f"decode={'set' if getattr(_kp, '_VK_DSA_KPOOL_DECODE', None) else 'MISSING'}",
            flush=True,
        )
    else:
        fail.append(
            f"kpool_fp8_index resolved OUTSIDE patches_full: {_src} — the SM80 "
            "bridge would never load"
        )
        print(
            f"  [preflight t={time.time() - _t0:.1f}s] FAIL kpool not patched: {_src}",
            flush=True,
        )
except Exception as e:
    fail.append(f"kpool_fp8_index: {type(e).__name__}: {e}")
    print(f"  [preflight t={time.time() - _t0:.1f}s] FAIL kpool_fp8_index: {e}", flush=True)

if fail:
    print("PREFLIGHT FAIL:", *fail, sep="\n  ", file=sys.stderr)
    sys.exit(1)

print("PREFLIGHT OK", flush=True)
