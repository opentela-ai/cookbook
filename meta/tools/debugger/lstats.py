"""lstats — no-reference first-forward per-layer residual bisect (Phase 2).

Self-installing, env-gated in-engine hook: prints ``[DBGSTAT]`` lines with
per-layer residual IN/OUT stats for the FIRST real forward, flags the first
layer that explodes / collapses / goes NaN, and writes a small manifest.json.
NO reference forward needed: the first bad layer's block names the broken
kernel family (whatever runs inside that layer). Generalizes the GLM-5.3
campaign's [LSTAT] inline patch (job 616115) which named the culprit family
without any cross-machine diff.

Env (all DBG_-prefixed; module self-installs on import when DBG_LSTAT=1):
  DBG_LSTAT=1              enable
  DBG_TARGET=a.b.Cls       dotted CLASS path whose __init__ is wrapped (required)
  DBG_LAYERS_ATTR=language_model.layers   path from the model to the layer list
  DBG_TOP_ATTR=            optional: dotted CLASS path owning the top forward
                           (default: DBG_TARGET itself)
  DBG_DIR=/scratch/dbg     output dir for the manifest (default: /tmp/dbg_lstats)
  DBG_TAG=beverin_v1       tag in log lines + manifest filename
  DBG_RANKS=0              comma list of ranks that instrument (default 0)
  DBG_MIN_TOKENS=1         skip forwards with fewer tokens (warmup gate)
  DBG_MAX_TOKENS=4096      skip forwards with more tokens
  DBG_LAYER_TYPES=la,la,dsa,...   optional comma list; recorded in the manifest
                           so first-bad-layer maps to a kernel family directly

Key gotcha (learned in the campaign): if the job dies BEFORE the layer loop —
e.g. a page_size/kpool assert at init_forward_metadata — no [DBGSTAT] lines
appear and the crash is a CONFIG bug, not your numerics bug. Check the failure
mode before interpreting silence.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from hook import resolve_attr, resolve_class, run_after_import

ON = os.environ.get("DBG_LSTAT", "0") == "1"
TARGET = os.environ.get("DBG_TARGET", "")
TOP_TARGET = os.environ.get("DBG_TOP_ATTR", "") or TARGET
LAYERS_ATTR = os.environ.get("DBG_LAYERS_ATTR", "model.layers")
DIR = os.environ.get("DBG_DIR", "/tmp/dbg_lstats")
TAG = os.environ.get("DBG_TAG", "site")
MIN_T = int(os.environ.get("DBG_MIN_TOKENS", "1"))
MAX_T = int(os.environ.get("DBG_MAX_TOKENS", "4096"))
LAYER_TYPES = [x for x in os.environ.get("DBG_LAYER_TYPES", "").split(",") if x]
RANKS = {int(x) for x in os.environ.get("DBG_RANKS", "0").split(",") if x != ""}


def _log(msg):
    sys.stderr.write(f"[lstats {TAG}] {msg}\n")
    sys.stderr.flush()


def _rank():
    try:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        return 0


def _tensor_stats(t):
    """abs_mean / abs_max / norm / nan / inf for one tensor-like; None if unusable."""
    try:
        import torch

        if not isinstance(t, torch.Tensor) or t.numel() == 0 or not t.is_floating_point():
            return None
        f = t.detach().float()
        nan = int(torch.isnan(f).sum())
        inf = int(torch.isinf(f).sum())
        am = float(f.abs().mean())
        if nan or inf:
            return {"abs_mean": am, "abs_max": None, "norm": None, "nan": nan, "inf": inf}
        return {"abs_mean": am, "abs_max": float(f.abs().max()), "norm": float(f.norm()), "nan": 0, "inf": 0}
    except Exception:
        return None


def _verdict(s):
    if s is None:
        return "skip"
    if s["nan"] or s["inf"]:
        return "nan"
    if s["abs_mean"] > 1e3:
        return "explode"
    if s["abs_mean"] < 1e-6:
        return "collapse"
    return "ok"


def _hidden_state(args, kwargs):
    """First floating tensor among args / common kwargs."""
    import torch

    hs = kwargs.get("hidden_states")
    if isinstance(hs, torch.Tensor):
        return hs
    for a in args:
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            return a
    return None


def _token_count(args, kwargs):
    import torch

    for cand in list(args) + [kwargs.get("input_ids")]:
        if isinstance(cand, torch.Tensor) and not cand.is_floating_point() and cand.ndim >= 2:
            return int(cand.shape[0] * cand.shape[-1])
    return 0


def _ids_digest(args, kwargs):
    """sha256 over the prompt token ids (identity anchor for later diffs)."""
    import torch

    for cand in list(args) + [kwargs.get("input_ids")]:
        if isinstance(cand, torch.Tensor) and not cand.is_floating_point() and cand.ndim >= 1:
            flat = cand.detach().cpu().contiguous().flatten()
            try:  # numpy may be absent in minimal containers
                raw = flat.numpy().tobytes()
            except Exception:
                raw = ",".join(map(str, flat.tolist())).encode()
            return hashlib.sha256(raw).hexdigest()[:16]
    return None


class _Installer:
    """Installs first-forward stat hooks once the model exists."""

    def __init__(self):
        self.handles = []
        self.latched = False
        self.gate_open = False   # True only during a gate-passing forward
        self.stats = {}          # "layer{N:02d}|in" -> stats dict
        self.layer_types = []
        self.top_in_ids = None

    def _want(self):
        return _rank() in RANKS and not self.latched and self.gate_open


    def _pre(self, idx):
        def hook(module, args, kwargs):
            try:
                if not self._want():
                    return
                s = _tensor_stats(_hidden_state(args, kwargs))
                if s is not None:
                    self.stats[f"layer{idx:02d}|in"] = s
                    _log(f"[DBGSTAT] layer={idx:02d} phase=in abs_mean={s['abs_mean']:.6g} "
                         f"nan={s['nan']} inf={s['inf']} verdict={_verdict(s)}")
            except Exception as exc:  # noqa: BLE001
                _log(f"layer{idx:02d} pre-hook error: {exc!r}")

        return hook

    def _post(self, idx):
        def hook(module, args, kwargs, output):
            try:
                import torch

                if not self._want():
                    return
                out = output[0] if isinstance(output, tuple) else output
                # first floating tensor of a tuple/list output (e.g. (hidden, cache))
                if isinstance(out, (tuple, list)):
                    for item in out:
                        if isinstance(item, torch.Tensor) and item.is_floating_point():
                            out = item
                            break
                s = _tensor_stats(out)
                if s is not None:
                    self.stats[f"layer{idx:02d}|out"] = s
                    _log(f"[DBGSTAT] layer={idx:02d} phase=out abs_mean={s['abs_mean']:.6g} "
                         f"nan={s['nan']} inf={s['inf']} verdict={_verdict(s)}")
            except Exception as exc:  # noqa: BLE001
                _log(f"layer{idx:02d} post-hook error: {exc!r}")

        return hook

    def install(self, model):
        layers = resolve_attr(model, LAYERS_ATTR)
        if layers is None or not hasattr(layers, "__len__"):
            _log(f"WARNING: DBG_LAYERS_ATTR={LAYERS_ATTR!r} not found on {type(model).__name__}; no layer hooks")
            return
        for i, layer in enumerate(layers):
            lt = LAYER_TYPES[i] if i < len(LAYER_TYPES) else ""
            self.layer_types.append(lt)
            try:
                self.handles.append(layer.register_forward_pre_hook(self._pre(i), with_kwargs=True))
                self.handles.append(layer.register_forward_hook(self._post(i), with_kwargs=True))
                layer._dbg_idx = i
                layer._dbg_type = lt
            except Exception as exc:  # noqa: BLE001
                _log(f"layer {i} hook registration failed: {exc!r}")
        _log(f"instrumented {len(layers)} layers of {type(model).__name__} (attr {LAYERS_ATTR!r})")

    def wrap_top_forward(self, model):
        orig = model.forward

        def fwd(*a, **kw):
            if not self.latched:
                # token gate HERE, where input_ids are visible; warmup forwards
                # (outside [MIN_T, MAX_T]) do NOT consume the latch
                n = _token_count(a, kw)
                self.gate_open = MIN_T <= n <= MAX_T
                if self.gate_open:
                    self.top_in_ids = _ids_digest(a, kw)
            else:
                self.gate_open = False
            try:
                return orig(*a, **kw)
            finally:
                if self.gate_open:
                    self.close()

        model.forward = fwd

        model.forward = fwd

    def close(self):
        if self.latched:
            return
        self.latched = True
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.write_manifest()

    def write_manifest(self):
        if _rank() not in RANKS:
            return
        try:
            os.makedirs(DIR, exist_ok=True)
            first_bad = None
            for key, s in self.stats.items():
                idx = int(key.split("|")[0].replace("layer", ""))
                phase = key.split("|")[1]
                if phase == "out" and _verdict(s) != "ok" and (first_bad is None or idx < first_bad):
                    first_bad = idx
            family = self.layer_types[first_bad] if first_bad is not None and first_bad < len(self.layer_types) else "?"
            man = {
                "tool": "lstats",
                "tag": TAG,
                "target": TARGET,
                "layers_attr": LAYERS_ATTR,
                "layer_types": self.layer_types,
                "top_input_ids_sha256": self.top_in_ids,
                "first_bad_layer": first_bad,
                "first_bad_family": family,
                "stats": self.stats,
            }
            path = os.path.join(DIR, f"lstats_{TAG}_r{_rank()}.json")
            with open(path, "w") as f:
                json.dump(man, f, indent=2)
            _log(f"manifest -> {path}; FIRST BAD LAYER = {first_bad} (family: {family})")
        except Exception as exc:  # noqa: BLE001
            _log(f"manifest write failed: {exc!r}")


_INSTALLER = _Installer()


def _arm(cls, on_init):
    orig_init = cls.__init__

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        try:
            on_init(self)
        except Exception as exc:  # noqa: BLE001
            _log(f"install failed on {type(self).__name__}: {exc!r}")

    cls.__init__ = init
    _log(f"armed on {cls.__module__}.{cls.__qualname__}")


def _bootstrap():
    if not ON:
        return
    if not TARGET:
        _log("DBG_LSTAT=1 but DBG_TARGET unset; doing nothing")
        return
    _log(f"armed; target={TARGET} layers={LAYERS_ATTR!r} dir={DIR}")
    mod_path = TARGET.rsplit(".", 1)[0]
    # idempotent hook even if the module is already in sys.modules
    run_after_import(mod_path, lambda _mod: None)
    _mod, cls = resolve_class(TARGET)
    _arm(cls, lambda model: _INSTALLER.install(model) or _INSTALLER.wrap_top_forward(model))
    if TOP_TARGET != TARGET:
        _top_mod, top_cls = resolve_class(TOP_TARGET)
        if top_cls is not cls:
            _arm(top_cls, lambda model: _INSTALLER.wrap_top_forward(model))


_bootstrap()
