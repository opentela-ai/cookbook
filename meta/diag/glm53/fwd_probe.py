"""Forward-probe (GLM53_FWD_PROBE=1): first-forward-only per-op logger for
sglang's GLM-5.3-Flash, plus the prefill argmax check.

A lazy hook (``import_hook.run_after_import``) wraps
``sglang.srt.models.glm5_next`` on first import and registers, on every
``Glm5NextDecoderLayer`` instance, forward pre/post hooks on the layer
itself plus its ``self_attn`` and ``mlp``. Output is one line per op per
forward, tagged with the layer index, role and class name, e.g.::

    [fwd-probe] rank0 >>>  Glm5NextForConditionalGeneration (FIRST-FORWARD-START) <<< t=0.000s
    [fwd-probe] rank0 ENTER layer[0] Glm5NextDecoderLayer t=0.012s
    [fwd-probe] rank0 ENTER layer[0].self_attn Glm5NextLinearAttention t=0.013s
    [fwd-probe] rank0 EXIT  layer[0].self_attn Glm5NextLinearAttention t=0.031s
    [fwd-probe] rank0 ENTER layer[0].mlp Glm5NextMoE t=0.032s
    ...
    [fwd-probe] rank0 ENTER layer[12].self_attn DeepseekV2AttentionMLA t=2.117s
    <silence — this op hangs>

The LAST line before silence names the exact hanging op:
  * ``...self_attn Glm5NextLinearAttention`` no EXIT -> the mamba/SSM scan or
    the TritonKDA (linear-attention) kernel on gfx942.
  * ``...self_attn DeepseekV2AttentionMLA`` no EXIT -> DSA decode
    (vk_hip_dsa_sparse_fwd) on the M:1 decode shape.
  * ``...mlp Glm5NextMoE`` no EXIT -> the Triton fp8 MoE kernel.
  * ``EXIT layer[k].self_attn`` then no ``ENTER layer[k].mlp`` -> the NCCL
    ``maybe_prefetch`` / ``prepare_mlp`` collective (multimem all-gather
    disabled -> set_signal_pad_size missing on this torch).

After the first full forward completes, a top-model post-hook logs
``FIRST-FORWARD-COMPLETE`` and REMOVES every handle, so subsequent forwards
are zero-cost. If the first forward hangs, the handles stay attached (there
is no second forward) and a call budget caps log volume. Hooks never return
non-None (cannot modify inputs/outputs) and are wrapped in try/except, so the
probe cannot itself introduce a failure mode.
"""
import os
import sys

from import_hook import run_after_import

_FWD_PROBE_ON = os.environ.get("GLM53_FWD_PROBE", "0") == "1"
_FWD_PROBE_TARGET = "sglang.srt.models.glm5_next"
_fwd_probe_active = False  # flipped True once layers are tagged
_fwd_started = False
_fwd_completed = False
_fwd_t0 = None
_fwd_calls = 0
_FWD_CALL_BUDGET = 20000
_fwd_handles = []


def _fwd_rank():
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:  # noqa: BLE001
        pass
    return 0


def _fwd_log(enter, name):
    import time as _time

    global _fwd_t0
    if _fwd_t0 is None:
        _fwd_t0 = _time.monotonic()
    tag = "ENTER" if enter else "EXIT "
    print(
        f"[fwd-probe] rank{_fwd_rank()} {tag} {name} "
        f"t={_time.monotonic() - _fwd_t0:.3f}s",
        flush=True,
    )


def _fwd_note(name):
    """Marker line (START/COMPLETE/PARGMAX) — not an ENTER/EXIT of an op."""
    import time as _time

    global _fwd_t0
    if _fwd_t0 is None:
        _fwd_t0 = _time.monotonic()
    print(
        f"[fwd-probe] rank{_fwd_rank()} ----  {name} "
        f"t={_time.monotonic() - _fwd_t0:.3f}s",
        flush=True,
    )


def _fwd_name(module):
    idx = getattr(module, "_fwd_probe_idx", None)
    if idx is not None:
        return f"layer[{idx}] {type(module).__name__}"
    parent = getattr(module, "_fwd_probe_parent", None)
    role = getattr(module, "_fwd_probe_role", "")
    if parent is not None:
        return f"layer[{parent}].{role} {type(module).__name__}"
    return type(module).__name__


def _fwd_pre(module, args, kwargs):
    global _fwd_calls
    if not _fwd_probe_active or _fwd_calls > _FWD_CALL_BUDGET:
        return None
    try:
        _fwd_calls += 1
        _fwd_log(True, _fwd_name(module))
    except Exception:  # noqa: BLE001
        pass
    return None


def _fwd_post(module, args, kwargs, output):
    global _fwd_calls
    if not _fwd_probe_active or _fwd_calls > _FWD_CALL_BUDGET:
        return None
    try:
        _fwd_calls += 1
        _fwd_log(False, _fwd_name(module))
    except Exception:  # noqa: BLE001
        pass
    return None


def _top_pre(module, args, kwargs):
    global _fwd_started
    if _fwd_started:
        return None
    _fwd_started = True
    try:
        _fwd_note(f">>>  {type(module).__name__} (FIRST-FORWARD-START) <<<")
    except Exception:  # noqa: BLE001
        pass
    return None


def _top_post(module, args, kwargs, output):
    global _fwd_probe_active, _fwd_completed
    if _fwd_completed:
        return None
    _fwd_completed = True
    _fwd_probe_active = False
    try:
        _fwd_note(f"<<<  {type(module).__name__} (FIRST-FORWARD-COMPLETE) >>>")
    except Exception:  # noqa: BLE001
        pass
    # Remove every handle -> truly zero-cost for all subsequent forwards.
    for h in _fwd_handles:
        try:
            h.remove()
        except Exception:  # noqa: BLE001
            pass
    _fwd_handles.clear()
    return None


def _register(mod, pre=_fwd_pre, post=_fwd_post):
    _fwd_handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
    _fwd_handles.append(mod.register_forward_hook(post, with_kwargs=True))


def _install_fwd_probe(module):
    try:
        DecoderLayer = module.Glm5NextDecoderLayer
        Model = module.Glm5NextModel
        TopModel = module.Glm5NextForConditionalGeneration
    except AttributeError as exc:
        sys.stderr.write(
            f"[sitecustomize] fwd-probe: expected classes not found in "
            f"{module.__name__} ({exc!r}); NOT installing\n"
        )
        return

    orig_model_init = Model.__init__

    def _probed_model_init(self, *a, **kw):
        global _fwd_probe_active
        orig_model_init(self, *a, **kw)
        layers = getattr(self, "layers", None)
        if layers is None:
            return
        _fwd_probe_active = True
        sys.stderr.write(
            f"[sitecustomize] fwd-probe: tagging {len(layers)} decoder layers "
            f"+ installing ENTER/EXIT hooks on layer/self_attn/mlp\n"
        )
        for i, layer in enumerate(layers):
            layer._fwd_probe_idx = i
            _register(layer)
            for role, sub in (
                ("self_attn", getattr(layer, "self_attn", None)),
                ("mlp", getattr(layer, "mlp", None)),
            ):
                if sub is None:
                    continue
                sub._fwd_probe_parent = i
                sub._fwd_probe_role = role
                _register(sub)

    Model.__init__ = _probed_model_init

    # Wrap the TOP-level model __init__: after the inner model exists,
    # register FIRST-FORWARD-START / FIRST-FORWARD-COMPLETE as INSTANCE
    # hooks on `self`. (Class-level register_forward_pre_hook is an instance
    # method in this torch build -> calling it on the class raised
    # "missing 1 required positional argument: 'hook'"; instance hooks avoid
    # that and are exactly as effective for the single serving model.)
    orig_top_init = TopModel.__init__

    def _probed_top_init(self, *a, **kw):
        orig_top_init(self, *a, **kw)
        _fwd_handles.append(
            self.register_forward_pre_hook(_top_pre, with_kwargs=True)
        )
        _fwd_handles.append(
            self.register_forward_hook(_top_post, with_kwargs=True)
        )

    TopModel.__init__ = _probed_top_init

    # GLM53 prefill argmax (the REAL hook). sglang calls model.forward()
    # directly (bypassing Module.__call__), so the instance forward hooks
    # above (_top_pre/_top_post) never fire. Wrap TopModel.forward itself;
    # on the FIRST call (prefill) capture argmax of the final-token logits
    # -> confirms the DSA/SDPA route is correct, BEFORE the gfx942 decode
    # (libamdhip64 / MLA-absorb-on-pure-MHA) can crash. Also logs the actual
    # LD_LIBRARY_PATH in this worker (to pin down the libamdhip64 mystery).
    if os.environ.get("GLM53_FWD_PROBE") and not getattr(
        TopModel, "_argmax_wrapped", False
    ):
        _orig_top_fwd = TopModel.forward
        _argmax_done = [False]

        def _argmax_top_fwd(self, *a, **kw):
            out = _orig_top_fwd(self, *a, **kw)
            if not _argmax_done[0]:
                _argmax_done[0] = True
                try:
                    import torch as _t
                    _logits = out if isinstance(out, _t.Tensor) else getattr(out, "next_token_logits", None)
                    if isinstance(_logits, _t.Tensor) and _logits.dim() >= 2:
                        _lo = _logits.float()
                        _last = _lo[-1] if _lo.dim() == 2 else _lo[:, -1, :]
                        _tid = int(_last.argmax(dim=-1).item())
                        _top5 = _last.topk(min(5, _last.shape[-1])).indices.tolist()
                        _fwd_note(f"[PARGMAX] prefill predicted token id={_tid} "
                                  f"top5={_top5} logits_max={float(_last.max().item()):.3f} "
                                  f"shape={tuple(_logits.shape)}")
                    elif isinstance(_logits, _t.Tensor):
                        _fwd_note(f"[PARGMAX] logits dim={_logits.dim()} shape={tuple(_logits.shape)}")
                    else:
                        _fwd_note(f"[PARGMAX] out type={type(out).__name__} "
                                  f"next_token_logits={type(getattr(out, 'next_token_logits', None)).__name__}")
                except Exception as _e:  # noqa: BLE001
                    try:
                        _fwd_note(f"[PARGMAX-ERR] {_e!r}")
                    except Exception:  # noqa: BLE001
                        pass
            return out

        TopModel.forward = _argmax_top_fwd
        TopModel._argmax_wrapped = True
        _fwd_note(f"[PARGMAX] TopModel.forward wrapped (GLM53_FWD_PROBE); "
                  f"LD_LIBRARY_PATH={(os.environ.get('LD_LIBRARY_PATH') or '<UNSET>')[:240]}")
    sys.stderr.write(
        "[sitecustomize] fwd-probe INSTALLED: logs each decoder layer + "
        "self_attn + mlp ENTER/EXIT on the FIRST forward; the last line "
        "before a stall names the hanging op.\n"
    )


if _FWD_PROBE_ON:
    run_after_import(_FWD_PROBE_TARGET, _install_fwd_probe)
