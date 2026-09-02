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

if fail:
    print("PREFLIGHT FAIL:", *fail, sep="\n  ", file=sys.stderr)
    sys.exit(1)

print("PREFLIGHT OK", flush=True)
