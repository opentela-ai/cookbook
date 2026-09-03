#!/usr/bin/env python3
# comp_capture.py -- per-layer + per-component I/O capture for GLM-5.3-Flash
#
# Activated by GLM53_COMP_CAPTURE=1 (the site sitecustomize / clariden heredoc
# imports this module when set).  Wraps Glm5NextModel/Glm5NextForConditional-
# Generation __init__ so that once the layers exist it tags every
# Glm5NextDecoderLayer with an index and registers ``with_kwargs=True``
# forward pre/post hooks, then removes every handle after the FIRST forward
# completes (zero-cost for all subsequent forwards).
#
# The goal: beverin (MI300A, gfx942) serves GLM-5.3-Flash but produces
# *garbage* tokens (real, weakly-peaked, context-present-but-wrong), while
# clariden (GH200, native CUDA kernels) serves the SAME model and produces
# "Paris".  The forward is deterministic (same weights + same prompt), so a
# per-layer then per-component I/O diff localises the broken MI300A kernel
# family WITHOUT guessing kernel-by-kernel and WITHOUT a known-good full
# beverin forward.
#
# Two modes (GLM53_COMP_MODE):
#   layers     (default): dump residual IN/OUT of EVERY layer.  For the
#               beverin-vs-clariden layer bisect (first layer where the
#               residual diverges, given the IN matched, names the family:
#               linear_attention layer -> MHC pre-norm tilelang / Mamba scan;
#               deepseek_sparse_attention layer -> DSA indexer/forward).
#   components (GLM53_COMP_LAYER=N, default 0): dump (input, output) of each
#               component of layer N -- hc_attn_pre, self_attn, hc_ffn_pre,
#               mlp, hc_post -- PLUS that layer's residual IN/OUT.  Layer 0's
#               input is the embedding (deterministic, identical on beverin
#               and clariden for the same prompt), so its component diff is
#               immediately meaningful with no dependency on prior layers.
#
# All hooks are: first-forward-only (removed on top-model forward complete),
# rank-gated (GLM53_COMP_RANKS, default rank 0), and try/except-wrapped so
# they can NEVER break the forward.
#
# Output:  $GLM53_COMP_DIR/$GLM53_COMP_TAG/
#             layers   -> layer{N:02d}_in.pt, layer{N:02d}_out.pt
#             comps    -> comp_layer{N}_{attn_pre,attn,ffn_pre,mlp,post}_in.pt
#                         comp_layer{N}_{attn_pre,attn,ffn_pre,mlp,post}_out.pt
#                         layer{N:02d}_in.pt, layer{N:02d}_out.pt (also)
# A small manifest.json records dtype/shape/norm per tensor (plus each
# layer's model class under _meta.layer_types) so the diff can run even when
# the raw tensors are large / across machines.
#
# Wiring (both sites; the sbatch exports GLM53_DIAG_DIR =
# <cookbook>/meta/diag/glm53 so beverin and clariden run the IDENTICAL
# copy from the cookbook checkout):
#     try:
#         if os.environ.get("GLM53_COMP_CAPTURE", "0") == "1":
#             sys.path.insert(0, os.environ["GLM53_DIAG_DIR"])
#             import comp_capture  # noqa: F401  (self-installing)
#     except Exception as _exc:
#         sys.stderr.write(f"[sitecustomize] comp_capture import failed: {_exc!r}\n")
import json
import os
import sys
import threading

from import_hook import run_after_import

_COMP_ON = os.environ.get("GLM53_COMP_CAPTURE", "0") == "1"
_COMP_MODE = os.environ.get("GLM53_COMP_MODE", "layers")  # layers | components
_COMP_LAYER = int(os.environ.get("GLM53_COMP_LAYER", "0"))
_COMP_DIR = os.environ.get(
    "GLM53_COMP_DIR",
    "/capstor/scratch/cscs/xyao/glm-53-flash-beverin/comp_capture",
)
_COMP_TAG = os.environ.get("GLM53_COMP_TAG", "beverin")
# Capture gating (bisect_layers_v1 lesson): the init-time memory-profile dummy
# forward (1 token) consumed the one-shot latch, so the capture landed on
# DUMMY data instead of the first real request.  With MIN_TOKENS/MAX_TOKENS
# set, a forward only ARMS the capture when its token count is in range; the
# latch then closes on the first ARMED forward.  Both machines run the same
# gen_correctness probe, so the first armed forward is the SAME prompt on
# beverin and clariden (verified post-hoc via the saved input_ids).
_COMP_MIN_TOKENS = int(os.environ.get("GLM53_COMP_MIN_TOKENS", "0"))
_COMP_MAX_TOKENS = int(os.environ.get("GLM53_COMP_MAX_TOKENS", "1024"))
# Ranks allowed to save (default "0": TP-replicated activations make rank0
# sufficient).  For PP>1 the pipeline stages hold DIFFERENT layers — pass
# e.g. GLM53_COMP_RANKS=0,2; files/manifests then carry an r{rank}_ prefix.
_COMP_RANKS = [
    int(x) for x in os.environ.get("GLM53_COMP_RANKS", "0").split(",") if x != ""
]
_armed = False  # True while the CURRENT forward passes the token-count gate
_COMP_TARGET = "sglang.srt.models.glm5_next"

_started = False  # flipped True once the model is built + hooks are live
_completed = False  # flipped True once the FIRST forward completes
_handles = []
_lock = threading.Lock()
_manifest = {}
_layer_info = []  # per-layer {"class", "mlp"} names -> manifest _meta


def _rank():
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:  # noqa: BLE001
        pass
    return 0


def _outdir():
    d = os.path.join(_COMP_DIR, _COMP_TAG)
    os.makedirs(d, exist_ok=True)
    return d


def _should_capture():
    """True on ranks that save tensors (GLM53_COMP_RANKS, default [0])."""
    return _rank() in _COMP_RANKS


def _save_name(name):
    """Rank-prefix file/manifest names when capturing on multiple ranks."""
    if _COMP_RANKS == [0]:
        return name
    return f"r{_rank()}_{name}"


def _summarise(name, t):
    """Record dtype/shape/norm so a cross-machine diff is possible even when
    raw tensors aren't shipped (and so a NaN/inf is visible at a glance)."""
    try:
        import torch

        if not isinstance(t, torch.Tensor) or t.numel() == 0:
            return
        with torch.inference_mode():
            info = {
                "name": name,
                "dtype": str(t.dtype),
                "shape": list(t.shape),
                "device": str(t.device),
                "numel": int(t.numel()),
                "abs_mean": float(t.abs().float().mean().item()),
                "abs_max": float(t.abs().float().max().item()),
                "rms": float((t.float() ** 2).mean().sqrt().item()),
                "has_nan": bool(torch.isnan(t).any().item()),
                "has_inf": bool(torch.isinf(t).any().item()),
            }
        with _lock:
            _manifest[name] = info
        return info
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[comp_capture] _summarise({name}) failed: {exc!r}\n")
        return None


def _save(name, t):
    """torch.save a detached CPU clone + append a manifest entry.  Safe to
    call on GPU tensors; never raises into the forward (try/except)."""
    if not _should_capture() or t is None:
        return
    try:
        import torch

        nm = _save_name(name)

        # GLM53 fix (2026-09-01): without an explicit cuda.synchronize(), the
        # hooks read STALE ZEROED GPU memory before the producing kernel (which
        # sglang runs on a non-default stream) completes -> every captured
        # tensor was abs_mean=0.0 (job 617522).  Sync ALL streams once per save
        # so _summarise and the .to("cpu") clone see the real data.
        if (
            os.environ.get("GLM53_COMP_SYNC", "1") == "1"
            and isinstance(t, torch.Tensor)
            and t.is_cuda
        ):
            torch.cuda.synchronize()

        t = _flat_output(t)
        if not isinstance(t, torch.Tensor):
            # Some component outputs are tuples/dicts with no tensor inside;
            # record the type so the diff knows it was composite.
            with _lock:
                key = "composite" if isinstance(t, (tuple, list)) else "type"
                _manifest[nm] = {"name": nm, key: str(type(t))}
            return
        _summarise(nm, t)
        path = os.path.join(_outdir(), f"{nm}.pt")
        torch.save(t.detach().to("cpu", copy=True), path)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[comp_capture] save({name}) failed: {exc!r}\n")


def _token_count(t):
    try:
        # input_ids are [num_tokens] (sglang flattens the batch) or [bs, seq];
        # numel == total tokens in both cases.  Duck-typed: no torch import
        # needed (keeps the gate working in torch-less unit tests too).
        numel = getattr(t, "numel", None)
        return int(numel()) if callable(numel) else -1
    except Exception:  # noqa: BLE001
        return -1


def _gate_tokens(t):
    n = _token_count(t)
    if n < 0:
        return _COMP_MIN_TOKENS <= 0
    if _COMP_MIN_TOKENS > 0 and n < _COMP_MIN_TOKENS:
        return False
    if _COMP_MAX_TOKENS > 0 and n > _COMP_MAX_TOKENS:
        return False
    return True


# -- tensor extraction from forward args/kwargs -------------------------------
# Glm5NextDecoderLayer.forward(self, positions, hidden_states, forward_batch,
#   residual=None, zero_allocator=None, gemm_output_zero_allocator=None,
#   prev_topk_indices=None, next_full_attention_layer_id=None)
def _arg_hidden_states(args, kwargs):
    """hidden_states is arg[1] in sglang's submodule signatures
    ((positions, hidden_states, forward_batch, ...)); fall back to the lone
    positional for single-argument modules so the helper isn't
    signature-fragile."""
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    if len(args) >= 2:
        return args[1]
    if len(args) == 1:
        return args[0]
    return None


def _arg_residual(args, kwargs):
    if "residual" in kwargs:
        return kwargs["residual"]
    if len(args) >= 4:
        return args[3]
    return None


def _flat_output(output):
    """The layer may return (hidden_states, topk_indices); the residual is
    the first element.  self_attn/mlp return a single tensor (or tuple)."""
    try:
        import torch

        if isinstance(output, (tuple, list)):
            tensors = [x for x in output if isinstance(x, torch.Tensor)]
            return tensors[0] if tensors else None
        return output
    except Exception:  # noqa: BLE001
        return output


# -- layer-level hooks (GLM53_COMP_MODE=layers) -------------------------------
def _layer_pre(module, args, kwargs):
    if not _started or _completed or not _armed:
        return None
    try:
        idx = getattr(module, "_comp_idx", None)
        if idx is None:
            return None
        hs = _arg_hidden_states(args, kwargs)
        _save(f"layer{idx:02d}_in", hs)
    except Exception:  # noqa: BLE001
        pass
    return None


def _layer_post(module, args, kwargs, output):
    if not _started or _completed or not _armed:
        return None
    try:
        idx = getattr(module, "_comp_idx", None)
        if idx is None:
            return None
        _save(f"layer{idx:02d}_out", _flat_output(output))
    except Exception:  # noqa: BLE001
        pass
    return None


# -- component-level hooks (GLM53_COMP_MODE=components, layer N) --------------
# MHC pre/post (hc_attn_pre / hc_ffn_pre / hc_post) are NOT nn.Modules — they
# are methods on Glm5NextDecoderLayer that the MHCLayerCommunicator stores as
# callables on its MHCState dataclass at __init__ time (capturing the bound
# methods BEFORE our hooks install).  So wrapping the LAYER's method is too
# late; the communicator calls the original.  We patch the MHCState's
# callables directly (_install_on_model accesses layer_communicator.mhc).
# self_attn / mlp ARE nn.Modules -> forward pre/post hooks.
def _patch_mhc_state(mhc):
    """Wrap hc_attn_pre / hc_ffn_pre / hc_post directly on the MHCState
    dataclass instance so our saving wrappers fire on every call."""
    for role, attr in (
        ("attn_pre", "hc_attn_pre"),
        ("ffn_pre", "hc_ffn_pre"),
        ("post", "hc_post"),
    ):
        orig = getattr(mhc, attr, None)
        if orig is None or not callable(orig):
            sys.stderr.write(f"[comp_capture] mhc.{attr} not callable, skip\n")
            continue

        def _make_wrapper(role, orig):
            def wrapper(*a, **kw):
                out = orig(*a, **kw)
                if _started and not _completed and _armed and _should_capture():
                    try:
                        # hc_*_pre(hidden_states, out_norm_weight, out_norm_eps)
                        # hc_post(hidden_states, residual, h_res, h_post)
                        # First positional arg is always the input tensor.
                        in_t = a[0] if a else kw.get("hidden_states") or kw.get("x")
                        _save(f"comp_layer{_COMP_LAYER}_{role}_in", in_t)
                        _save(f"comp_layer{_COMP_LAYER}_{role}_out", _flat_output(out))
                    except Exception as exc:  # noqa: BLE001
                        sys.stderr.write(f"[comp_capture] mhc {role} save failed: {exc!r}\n")
                return out
            return wrapper

        setattr(mhc, attr, _make_wrapper(role, orig))


def _comp_sub_pre(role):
    def hook(module, args, kwargs):
        if not _started or _completed or not _armed or not _should_capture():
            return None
        try:
            hs = _arg_hidden_states(args, kwargs)
            _save(f"comp_layer{_COMP_LAYER}_{role}_in", hs)
        except Exception:  # noqa: BLE001
            pass
        return None

    return hook


def _comp_sub_post(role):
    def hook(module, args, kwargs, output):
        if not _started or _completed or not _armed or not _should_capture():
            return None
        try:
            _save(f"comp_layer{_COMP_LAYER}_{role}_out", _flat_output(output))
        except Exception:  # noqa: BLE001
            pass
        return None

    return hook


def _register(mod, pre, post):
    try:
        _handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        _handles.append(mod.register_forward_hook(post, with_kwargs=True))
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[comp_capture] register failed: {exc!r}\n")


def _install_on_model(self):
    """Called from the wrapped Glm5NextModel.__init__ once self.layers exists."""
    global _started, _layer_info
    layers = getattr(self, "layers", None)
    if not layers:
        return
    _started = True
    n = len(layers)
    # Class names per layer -> comp_diff labels the kernel family from the
    # MANIFEST instead of guessing from the layer index.
    _layer_info = [
        {"class": type(layer).__name__,
         "mlp": type(getattr(layer, "mlp", None)).__name__}
        for layer in layers
    ]
    sys.stderr.write(
        f"[sitecustomize] comp_capture: MODE={_COMP_MODE} LAYER={_COMP_LAYER} "
        f"tag={_COMP_TAG} -> installing hooks on {n} layers (rank{_rank()})\n"
    )
    # Directly capture the embedding output (embed_tokens is the model's first
    # op; its output == layer 0 input).  Distinguishes "embedding genuinely
    # zero" (deep pre-layer bug) from "only the layer-0 input read is stale"
    # (async-stream artifact).  Installed in ALL modes; cheap + first-fwd-only.
    try:
        emb = getattr(self, "embed_tokens", None)
        if emb is not None and type(emb).__name__ != "PPMissingLayer":
            def _emb_post(module, args, kwargs, output):
                if not _started or _completed or not _armed or not _should_capture():
                    return None
                _save("embed_out", _flat_output(output))
                return None

            _handles.append(emb.register_forward_hook(_emb_post, with_kwargs=True))
            sys.stderr.write("[comp_capture] embed_tokens hook installed\n")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[comp_capture] embed_tokens hook failed: {exc!r}\n")
    if _COMP_MODE == "layers":
        for i, layer in enumerate(layers):
            layer._comp_idx = i
            _register(layer, _layer_pre, _layer_post)
    elif _COMP_MODE == "components":
        target = layers[_COMP_LAYER] if _COMP_LAYER < n else layers[0]
        target._comp_idx = _COMP_LAYER
        # residual IN/OUT of the target layer (same as layers mode for N)
        _register(target, _layer_pre, _layer_post)
        # self_attn + mlp (nn.Modules -> forward hooks)
        for role, sub in (
            ("attn", getattr(target, "self_attn", None)),
            ("mlp", getattr(target, "mlp", None)),
        ):
            if sub is None:
                continue
            _register(sub, _comp_sub_pre(role), _comp_sub_post(role))
        # hc_attn_pre / hc_ffn_pre / hc_post: patch the communicator's
        # MHCState callables directly (NOT the layer's methods — the
        # communicator captured the bound methods at __init__ before our
        # hooks installed, so a layer-level wrapper would never fire).
        comm = getattr(target, "layer_communicator", None)
        mhc = getattr(comm, "mhc", None) if comm else None
        if mhc is not None:
            _patch_mhc_state(mhc)
        else:
            sys.stderr.write(
                f"[comp_capture] layer {_COMP_LAYER}: no MHCState on "
                f"communicator (config.mhc={getattr(self.config, 'mhc', '?')}); "
                f"attn_pre/ffn_pre/post will NOT be captured\n"
            )
    else:
        sys.stderr.write(f"[sitecustomize] comp_capture: unknown MODE {_COMP_MODE!r}\n")


def _top_pre(module, args, kwargs):
    if _completed:
        return None
    try:
        if _started and _should_capture():
            sys.stderr.write(
                "[sitecustomize] comp_capture: FIRST-FORWARD-START "
                f"(mode={_COMP_MODE}, layer={_COMP_LAYER}, tag={_COMP_TAG})\n"
            )
    except Exception:  # noqa: BLE001
        pass
    return None


def _top_post(module, args, kwargs, output):
    """First forward complete: dump the manifest, then remove EVERY handle so
    the capture is truly zero-cost for all subsequent forwards."""
    global _completed
    if _completed:
        return None
    _completed = True
    try:
        if _should_capture():
            mname = (
                "manifest.json"
                if _COMP_RANKS == [0]
                else f"manifest_r{_rank()}.json"
            )
            out = os.path.join(_outdir(), mname)
            with _lock:
                _manifest["_meta"] = {
                    "mode": _COMP_MODE,
                    "layer": _COMP_LAYER,
                    "tag": _COMP_TAG,
                    "rank": _rank(),
                    "ranks": _COMP_RANKS,
                    "min_tokens": _COMP_MIN_TOKENS,
                    "max_tokens": _COMP_MAX_TOKENS,
                    "layer_types": _layer_info,
                }
                with open(out, "w") as fh:
                    json.dump(_manifest, fh, indent=2, sort_keys=True)
            sys.stderr.write(
                f"[sitecustomize] comp_capture: FIRST-FORWARD-COMPLETE; "
                f"manifest -> {out} ({len(_manifest)} tensors); "
                f"removing {len(_handles)} hooks\n"
            )
        for h in _handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        _handles.clear()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[comp_capture] top_post failed: {exc!r}\n")
    return None


def _wrap(module):
    """Fires once, after sglang.srt.models.glm5_next is first imported: wrap
    Glm5NextModel.__init__ (install the per-layer hooks) and the top-level
    forward (arm the token gate + close the latch)."""
    try:
        Model = module.Glm5NextModel
        TopModel = module.Glm5NextForConditionalGeneration
    except AttributeError as exc:
        sys.stderr.write(
            f"[comp_capture] expected classes not found in "
            f"{module.__name__} ({exc!r}); NOT installing\n"
        )
        return

    orig_model_init = Model.__init__

    def _probed_model_init(self, *a, **kw):
        orig_model_init(self, *a, **kw)
        _install_on_model(self)

    Model.__init__ = _probed_model_init

    orig_top_init = TopModel.__init__

    def _probed_top_init(self, *a, **kw):
        orig_top_init(self, *a, **kw)
        try:
            # sglang calls model.forward() directly (bypassing
            # Module.__call__), so instance forward hooks NEVER fire.
            # Wrap self.forward on the INSTANCE (not the class) so the
            # first-forward-complete logic runs.  If FwdProbe already
            # wrapped TopModel.forward with _argmax_top_fwd, _orig_fwd
            # captures that bound method (argmax still logs).
            _orig_fwd = self.forward

            def _comp_top_fwd(*a, **kw):
                global _armed
                if not _completed and _started:
                    ii = a[0] if a else kw.get("input_ids")
                    _armed = _gate_tokens(ii)
                    if _should_capture():
                        try:
                            sys.stderr.write(
                                "[sitecustomize] comp_capture: forward "
                                f"tokens={_token_count(ii)} armed={_armed} "
                                f"(mode={_COMP_MODE}, layer={_COMP_LAYER}, "
                                f"tag={_COMP_TAG})\n"
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    if _armed:
                        # Save the request identity so the cross-machine
                        # diff can VERIFY both captures used the same
                        # tokens before comparing activations.
                        _save("input_ids", ii)
                        _save(
                            "positions",
                            a[1] if len(a) > 1 else kw.get("positions"),
                        )
                out = _orig_fwd(*a, **kw)
                if _armed and not _completed:
                    _top_post(self, (), {}, out)
                return out

            self.forward = _comp_top_fwd
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[comp_capture] top fwd wrap failed: {exc!r}\n")

    TopModel.__init__ = _probed_top_init
    sys.stderr.write(
        f"[sitecustomize] comp_capture: wrapped Glm5NextModel + "
        f"{TopModel.__name__} __init__ (mode={_COMP_MODE}, "
        f"layer={_COMP_LAYER}, tag={_COMP_TAG})\n"
    )


if _COMP_ON:
    run_after_import(_COMP_TARGET, _wrap)
    sys.stderr.write(
        f"[sitecustomize] comp_capture: import hook installed "
        f"(mode={_COMP_MODE}, layer={_COMP_LAYER}, tag={_COMP_TAG})\n"
    )
