#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Issue #45 diagnostic: name the first submodule that emits NaN/Inf.

Guarded by ``K3_NAN_CHECK=1`` and installed from ``sitecustomize.py`` at
interpreter start (before the model is built). We monkeypatch
``torch.nn.Module._call_impl`` so *every* module output is inspected. The
first time a given module CLASS produces a non-finite tensor we print its
class name, output shape and the NaN/Inf counts. The first class to trip is
the culprit: its inputs were finite (a previously-finite module would have
tripped first), so the NaN originates inside it.

Deduplicated per class name, so the log stays small. Once ``K3_NAN_MAX``
(default 16) distinct classes have tripped we unhook ourselves.
"""
import os

_NAN_MAX = int(os.environ.get("K3_NAN_MAX", "16"))
_installed = False
_tripped = set()


def _rank_tag():
    for key in ("SLURM_PROCID", "RANK", "LOCAL_RANK", "VLLM_DP_RANK"):
        v = os.environ.get(key)
        if v is not None:
            return f"{key}={v}"
    return "rank=?"


def install():
    """Install the module-output NaN detector. Idempotent."""
    global _installed
    if _installed:
        return
    import torch

    _installed = True
    orig = torch.nn.Module._call_impl
    tag = _rank_tag()
    state = {"n": 0}

    def patched(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if state["n"] >= _NAN_MAX:
            return out
        try:
            t = out
            if isinstance(t, (tuple, list)):
                t = next((x for x in t if torch.is_tensor(x)), None)
            if t is not None and torch.is_tensor(t) and t.is_floating_point():
                name = type(self).__name__
                if name not in _tripped:
                    bad = int(torch.isnan(t).sum())
                    inf = int(torch.isinf(t).sum())
                    if bad or inf:
                        _tripped.add(name)
                        state["n"] += 1
                        print(
                            f"[NAN] {tag} module={name} "
                            f"out={tuple(t.shape)} dtype={t.dtype} "
                            f"nan={bad} inf={inf}",
                            flush=True,
                        )
        except Exception:
            pass
        return out

    torch.nn.Module._call_impl = patched
    print(f"[sitecustomize] K3_NAN_CHECK: NaN/Inf module hook installed "
          f"({tag}, max {_NAN_MAX} classes)", flush=True)
