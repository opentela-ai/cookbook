"""capture — generic first-forward per-layer / per-component I/O dump (Phase 3).

Self-installing, env-gated in-engine hook. Wraps DBG_TARGET's __init__ so that
once the model exists it tags every layer with an index, registers
``with_kwargs=True`` forward pre/post hooks, and removes every handle after the
FIRST gate-passing forward completes (zero cost afterwards). Each saved tensor
gets a stats entry in manifest.json, and the top-forward prompt token ids are
digested into the manifest — that digest is the IDENTITY GATE diff.py checks
before comparing anything (a diff against a different prompt is meaningless).

Generalizes meta/diag/glm53/comp_capture.py (which localized the GLM-5.3 MI300A
kernel bug via a beverin-vs-clariden layer bisect, then a component drill-down
against a pure-torch reference).

Env (module self-installs on import when DBG_CAPTURE=1):
  DBG_CAPTURE=1            enable
  DBG_TARGET=a.b.Cls       dotted CLASS path whose __init__ is wrapped (required)
  DBG_LAYERS_ATTR=language_model.layers   path from the model to the layer list
  DBG_CAPTURE_MODE=layers  layers | components
  DBG_COMPONENT_LAYER=0    for components mode: which layer to drill into
  DBG_COMPONENTS=attn,mlp  comma list of child-name substrings, or regex:PAT
  DBG_EMBED_ATTR=          optional path from model to the embedding module;
                           its output is captured too (deterministic anchor)
  DBG_DIR=/scratch/dbg     output ROOT dir (required to save)
  DBG_TAG=beverin_v1       files land in $DBG_DIR/$DBG_TAG/
  DBG_RANKS=0              ranks that save tensors (default 0)
  DBG_MIN_TOKENS=1 / DBG_MAX_TOKENS=2048   token gate (skip warmup forwards)
  DBG_CAPTURE_PROBE=1      not handled here: fire your deterministic probe
                           request after /health; the latch lands on it

Output layout ($DBG_DIR/$DBG_TAG/):
  layers mode      layer{N:02d}_in.pt / layer{N:02d}_out.pt
  components mode  comp_layer{N}_{name}_in.pt / _out.pt  + layer{N:02d}_in/out.pt
  manifest.json    per-tensor dtype/shape/stats + top_input_ids_sha256 +
                   layer_types (so diffs can label the broken kernel family)
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from hook import resolve_attr, resolve_class, run_after_import

ON = os.environ.get("DBG_CAPTURE", "0") == "1"
TARGET = os.environ.get("DBG_TARGET", "")
TOP_TARGET = os.environ.get("DBG_TOP_ATTR", "") or TARGET
LAYERS_ATTR = os.environ.get("DBG_LAYERS_ATTR", "model.layers")
MODE = os.environ.get("DBG_CAPTURE_MODE", "layers")   # layers | components
COMP_LAYER = int(os.environ.get("DBG_COMPONENT_LAYER", "0"))
COMPONENTS = [x for x in os.environ.get("DBG_COMPONENTS", "").split(",") if x]
EMBED_ATTR = os.environ.get("DBG_EMBED_ATTR", "")
DIR = os.environ.get("DBG_DIR", "")
TAG = os.environ.get("DBG_TAG", "site")
MIN_T = int(os.environ.get("DBG_MIN_TOKENS", "1"))
MAX_T = int(os.environ.get("DBG_MAX_TOKENS", "2048"))
LAYER_TYPES = [x for x in os.environ.get("DBG_LAYER_TYPES", "").split(",") if x]
RANKS = {int(x) for x in os.environ.get("DBG_RANKS", "0").split(",") if x != ""}


def _log(msg):
    sys.stderr.write(f"[capture {TAG}] {msg}\n")
    sys.stderr.flush()


def _rank():
    try:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        return 0


def _token_count(args, kwargs):
    import torch

    for cand in list(args) + [kwargs.get("input_ids")]:
        if isinstance(cand, torch.Tensor) and not cand.is_floating_point() and cand.ndim >= 2:
            return int(cand.shape[0] * cand.shape[-1])
    return 0


def _ids_digest(args, kwargs):
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


def _first_float(args, kwargs):
    import torch

    hs = kwargs.get("hidden_states")
    if isinstance(hs, torch.Tensor) and hs.is_floating_point():
        return hs
    for a in args:
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            return a
    return None


def _flat_output(output):
    import torch

    out = output[0] if isinstance(output, tuple) else output
    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, torch.Tensor) and item.is_floating_point():
                return item
    return out if isinstance(out, torch.Tensor) else None


def _summarise(t):
    import torch

    f = t.detach().float()
    return {
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "abs_mean": float(f.abs().mean()),
        "abs_max": float(f.abs().max()),
        "norm": float(f.norm()),
        "nan": int(torch.isnan(f).sum()),
        "inf": int(torch.isinf(f).sum()),
    }


class _Capture:
    def __init__(self):
        self.handles = []
        self.latched = False
        self.gate_open = False   # True only during a gate-passing forward
        self.manifest = {
            "tool": "capture",
            "tag": TAG,
            "mode": MODE,
            "target": TARGET,
            "layers_attr": LAYERS_ATTR,
            "layer_types": list(LAYER_TYPES),
            "tensors": {},           # name -> stats
            "top_input_ids_sha256": None,
        }
        self._dirty = False

    # -- gating -------------------------------------------------------------
    def _want_save(self):
        """Hooks save only inside a gate-passing, unlatched forward on a saving rank."""
        return bool(DIR) and _rank() in RANKS and not self.latched and self.gate_open

    def _save(self, name, t):
        if not self._want_save():
            return
        import torch

        try:
            outdir = os.path.join(DIR, TAG)
            os.makedirs(outdir, exist_ok=True)
            torch.save(t.detach().cpu(), os.path.join(outdir, f"{name}.pt"))
            self.manifest["tensors"][name] = _summarise(t)
            self._dirty = True
        except Exception as exc:  # noqa: BLE001
            _log(f"save {name} failed: {exc!r}")

    # -- hook bodies ---------------------------------------------------------
    def _named_pre(self, name):
        def hook(module, args, kwargs):
            try:
                if not self._want_save():
                    return
                t = _first_float(args, kwargs)
                if t is not None:
                    self._save(name, t)
            except Exception as exc:  # noqa: BLE001
                _log(f"{name} pre-hook error: {exc!r}")

        return hook

    def _named_post(self, name):
        def hook(module, args, kwargs, output):
            try:
                if not self._want_save():
                    return
                t = _flat_output(output)
                if t is not None:
                    self._save(name, t)
            except Exception as exc:  # noqa: BLE001
                _log(f"{name} post-hook error: {exc!r}")

        return hook

    def _match_child(self, name):
        for pat in COMPONENTS:
            if pat.startswith("regex:"):
                import re

                if re.search(pat[6:], name):
                    return True
            elif fnmatch.fnmatch(name, f"*{pat}*"):
                return True
        return False

    # -- wiring ----------------------------------------------------------------
    def install(self, model):
        layers = resolve_attr(model, LAYERS_ATTR)
        if layers is None or not hasattr(layers, "__len__"):
            _log(f"WARNING: DBG_LAYERS_ATTR={LAYERS_ATTR!r} not found on {type(model).__name__}; nothing captured")
            return
        n = len(layers)
        _log(f"mode={MODE}; instrumenting {n} layers (attr {LAYERS_ATTR!r})")
        for i, layer in enumerate(layers):
            lt = LAYER_TYPES[i] if i < len(LAYER_TYPES) else ""
            try:
                layer._dbg_idx = i
                layer._dbg_type = lt
            except Exception:
                pass
            if MODE == "layers" or (MODE == "components" and i == COMP_LAYER):
                self.handles.append(layer.register_forward_pre_hook(self._named_pre(f"layer{i:02d}_in"), with_kwargs=True))
                self.handles.append(layer.register_forward_hook(self._named_post(f"layer{i:02d}_out"), with_kwargs=True))
            if MODE == "components" and i == COMP_LAYER:
                for cname, child in layer.named_children():
                    if COMPONENTS and not self._match_child(cname):
                        continue
                    base = f"comp_layer{i:02d}_{cname}"
                    self.handles.append(child.register_forward_pre_hook(self._named_pre(f"{base}_in"), with_kwargs=True))
                    self.handles.append(child.register_forward_hook(self._named_post(f"{base}_out"), with_kwargs=True))
                    _log(f"  layer {i} component hook: {cname}")
        if EMBED_ATTR:
            emb = resolve_attr(model, EMBED_ATTR)
            if emb is not None:
                self.handles.append(emb.register_forward_hook(self._named_post("embed_out"), with_kwargs=True))
        _log(f"armed: {len(self.handles)} hooks")

    def wrap_top_forward(self, model):
        orig = model.forward

        def fwd(*a, **kw):
            if not self.latched:
                # token gate HERE, where input_ids are visible; warmup forwards
                # (outside [MIN_T, MAX_T]) do NOT consume the latch
                n = _token_count(a, kw)
                self.gate_open = MIN_T <= n <= MAX_T
                if self.gate_open:
                    self.manifest["top_input_ids_sha256"] = _ids_digest(a, kw)
            else:
                self.gate_open = False
            try:
                return orig(*a, **kw)
            finally:
                if self.gate_open:
                    self.close()  # first gate-passing forward completes -> latch

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
        if not self._dirty:
            _log("no tensors captured (token gate never passed?) — check DBG_MIN_TOKENS vs warmup length and your probe prompt")
        self.write_manifest()

    def write_manifest(self):
        if not DIR or _rank() not in RANKS:
            return
        try:
            outdir = os.path.join(DIR, TAG)
            os.makedirs(outdir, exist_ok=True)
            path = os.path.join(outdir, "manifest.json")
            with open(path, "w") as f:
                json.dump(self.manifest, f, indent=2)
            _log(f"manifest -> {path} ({len(self.manifest['tensors'])} tensors)")
        except Exception as exc:  # noqa: BLE001
            _log(f"manifest write failed: {exc!r}")


_INSTALLER = _Capture()


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
        _log("DBG_CAPTURE=1 but DBG_TARGET unset; doing nothing")
        return
    _log(f"armed; target={TARGET} mode={MODE} dir={DIR or '(unsaved: DBG_DIR unset)'}")
    mod_path = TARGET.rsplit(".", 1)[0]
    run_after_import(mod_path, lambda _mod: None)
    _mod, cls = resolve_class(TARGET)
    _arm(cls, lambda model: _INSTALLER.install(model) or _INSTALLER.wrap_top_forward(model))
    if TOP_TARGET != TARGET:
        _top_mod, top_cls = resolve_class(TOP_TARGET)
        if top_cls is not cls:
            _arm(top_cls, lambda model: _INSTALLER.wrap_top_forward(model))


_bootstrap()
