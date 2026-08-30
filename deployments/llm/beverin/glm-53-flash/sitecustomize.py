"""sitecustomize: LAZY rebind of sglang's DSA ``tilelang_sparse_fwd`` to the
vkernels HIP kernel on MI300A (gfx942), bypassing the tilelang JIT abort
(issue #51). Loaded automatically by CPython at startup (installed by
build_overlay.sh into $OVL/pylib/sitecustomize.py, which the sglang-rocm EDF
prepends FIRST on PYTHONPATH).

WHY LAZY (sys.meta_path) — beverin job 612821
----------------------------------------------
An EARLIER version of this module imported
``sglang.kernels.ops.attention.dsa.tilelang_kernel`` at startup. That import
is the ONLY thing in the recipe that pulls in the full ``sglang.kernels``
package + ``tilelang``/``aiter`` (``import sglang`` alone does NOT — those
submodules load lazily, at the first DSA forward). On a cold node that eager
import took ~4-5 min and blew the sbatch's preflight gate. The rebind itself
is pointless until sglang actually runs a DSA forward, so we now DEFER it: a
``sys.meta_path`` finder fires ONLY when sglang itself imports
``tilelang_kernel`` (the engine's first DSA forward, after model load +
warmup — well outside the preflight gate). Zero startup cost; patches exactly
when needed, exactly once (Python's import lock serialises the first import).

THE BLOCKER (vkernels #51, beverin job 612262)
----------------------------------------------
GLM-5.3-Flash's DeepseekSparseAttn has qk_rope_head_dim = 0 -> tail_dim = 0,
so sglang's tilelang ``sparse_mla_fwd_decode_partial`` (the bf16 path selected
on gfx942 by ``tilelang_sparse_fwd``) allocates zero-extent Q_tail_buf /
K_tail_shared and emits a zero-K GEMM. TVM's VectorizePlanner hits
``Check failed: pb->value != 0 (0 vs. 0) : Divide by zero`` at JIT time.
``tilelang`` is the only kpool>1-legal DSA decode backend on MI300A
(fa3/trtllm NVIDIA-only; flashmla_*/aiter fail _check_kpool_tail_backend),
so there is no flag-switch escape.

THE FIX (vkernels PR #52, commit d76517c)
-----------------------------------------
PR #52 re-implements the forward as a plain gfx942 HIP kernel
(``vk_hip_dsa_sparse_fwd`` in ``libvkernels_hip.so``) that takes tail_dim == 0
as a runtime branch (the rope-tail dot loop runs zero iterations — no
zero-size GEMM). Validated on beverin by ``probe_dsa_vkernels.py``: GLM-5.3
(tail_dim==0, the job-612262 blocker) and DeepSeek-V3 (tail_dim>0) both
return finite bf16 ``(1, S_q, H, d_v)`` through the rebound sglang symbol
with no JIT abort. sglang's ``dsa_backend._forward_tilelang`` does a local
``from ...tilelang_kernel import tilelang_sparse_fwd`` at call time (both
forward_extend at L3223 and forward_decode at L3525), so once this finder
patches the module on first import every DSA forward picks up the rebound
symbol. The MHC tilelang path is left untouched (still patched by
tilelang-mhc-reduce-hidden_block-for-mi300a-64KB-LDS.patch via
build_overlay.sh — that's the SECOND blocker, already handled).

==============================================================================
Forward-probe (GLM53_FWD_PROBE=1): first-forward-only per-op logger
==============================================================================
A SECOND lazy meta_path finder (installed only when GLM53_FWD_PROBE=1, default
0) wraps ``sglang.srt.models.glm5_next`` on first import and registers, on
every ``Glm5NextDecoderLayer`` instance, forward pre/post hooks on the layer
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

import importlib.abc
import sys

# =============================================================================
# DSA-vkernels lazy rebind (PR #52, issue #51)
# =============================================================================
_TARGET = "sglang.kernels.ops.attention.dsa.tilelang_kernel"
_patched = False


def _supports_current_device():
    """True only on gfx942 (MI300A). Guards the rebind so a CPU-only preflight
    or a non-MI300A node keeps sglang's native tilelang path (harmless there).
    Probed lazily at the first DSA forward, where torch + a GPU are certain."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, ""
        props = torch.cuda.get_device_properties(0)
        gcn = getattr(props, "gcnArchName", "") or ""
        return ("gfx942" in gcn, gcn)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[sitecustomize] DSA-vkernels patch: device probe failed ({exc!r}); "
            "NOT patching (sglang will use its native tilelang path)\n"
        )
        return False, ""


def _apply(target_module, gcn):
    """Rebind sglang's DSA ``tilelang_sparse_fwd`` (PR #52 forward) AND
    ``tilelang_fp8_paged_mqa_logits`` (issue #51, the kpool>1 indexer's
    per-paged-KV-tile gated logit) to the vkernels HIP ctypes adapters."""
    try:
        import vkernels_dsa
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[sitecustomize] DSA-vkernels patch: could not import vkernels_dsa "
            f"({exc!r}); NOT patching (sglang will use its native tilelang path)\n"
        )
        return
    patched = []
    # PR #52: the sparse-MLA forward (vk_hip_dsa_sparse_fwd).
    target_module.tilelang_sparse_fwd = vkernels_dsa.tilelang_sparse_fwd
    patched.append("tilelang_sparse_fwd->vk_hip_dsa_sparse_fwd (PR #52)")
    # Issue #51: the kpool>1 indexer's gated top-k logits (vk_hip_dsa_topk_logits).
    topk = getattr(vkernels_dsa, "tilelang_fp8_paged_mqa_logits", None)
    if topk is not None:
        target_module.tilelang_fp8_paged_mqa_logits = topk
        patched.append("tilelang_fp8_paged_mqa_logits->vk_hip_dsa_topk_logits (#51)")
    else:
        sys.stderr.write(
            "[sitecustomize] DSA-vkernels patch: tilelang_fp8_paged_mqa_logits "
            "not in vkernels_dsa (only the forward was rebound); the indexer "
            "will fall back to its native tilelang path (the #51 hang).\n"
        )
    sys.stderr.write(
        "[sitecustomize] DSA-vkernels patch APPLIED on " + gcn + ": "
        + "; ".join(patched) + ".\n"
    )


class _WrapLoader(importlib.abc.Loader):
    """Runs the real loader (defines ``@tilelang.jit tilelang_sparse_fwd``),
    then — once, on gfx942 — rebinds it to the vkernels ctypes adapter."""

    def __init__(self, real, name):
        self._real = real
        self._name = name

    def create_module(self, spec):
        # Defer to the real loader's create_module (SourceFileLoader has none
        # in CPython -> returns None -> Python creates a fresh module).
        if hasattr(self, "_real") and hasattr(self._real, "create_module"):
            return self._real.create_module(spec)
        return None

    def exec_module(self, module):
        self._real.exec_module(module)  # runs the real @tilelang.jit defs
        global _patched
        if _patched:
            return
        _patched = True
        ok, gcn = _supports_current_device()
        if ok:
            _apply(module, gcn)


class _Patcher(importlib.abc.MetaPathFinder):
    """Finds sglang's ``tilelang_kernel`` and wraps its loader so the rebind
    runs exactly once, right after the module loads — NOT at interpreter
    startup. Returns None for every other module (zero overhead)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET or _patched:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is None:
                continue
            spec.loader = _WrapLoader(spec.loader, fullname)
            return spec
        return None


# Install at startup. Zero work: no sglang/tilelang/torch import. The finder
# acts only when sglang imports tilelang_kernel (the engine's first DSA
# forward); a CPU-only preflight never imports it, so the gate stays fast.
sys.meta_path.insert(0, _Patcher())


# =============================================================================
# kpool top-k TRANSFORM torch-fallback bridge (vkernels #56, GLM53_TOPK_TRANSFORM_BACKEND=torch)
# =============================================================================
# sglang's decode DSA path calls ``fast_kpool_topk_transform_fused`` (the radix
# top-k *transform*; ``kernels/ops/moe/kpool_topk_transform.py``), which routes
# through the TVM JIT ``sgl_kernel_jit_kpool_topk_transform_{group_topk}``. That
# JIT compiles on the ROCm container only after the ``cuda_fp16.h``->
# ``hip_runtime.h`` guard (cookbook 9b6ab33), and on the GLM-5.3 decode shape the
# launched kernel does not return (vkernels #56: every scheduler_TP{0..3}_EP{0..3}
# stuck in ``Dl`` state, ENTER/EXIT 584/0, no crash). vkernels ships the two
# bookend kernels (``vk_hip_dsa_topk_logits`` #51, ``vk_hip_dsa_sparse_fwd`` #52)
# but NOT this middle transform.
#
# This finder lazily rebinds ``fast_kpool_topk_transform_fused`` to a
# pure-PyTorch implementation (``vkernels_dsa_topk``) that is numerically
# equivalent to ``kernels/jit/csrc/dsa/kpool_topk_transform.cuh:249-310``
# (validated 7/7 by ``test_vkernels_dsa_topk.py`` on the container). Goals:
#   1. UNBLOCK: if the hang is the JIT, decode serves immediately.
#   2. LOCALISE: if the hang persists, it is definitively in a bookend kernel.
#   3. REFERENCE: an oracle for the future HIP kernel (#56).
# Gate: GLM53_TOPK_TRANSFORM_BACKEND=torch (default off -> the JIT is the
# control, so the two paths are A/B-testable) AND gfx942 (the JIT is fine on
# CUDA; the torch fallback is slower and only needed on MI300A).
_TOPK_TARGET = "sglang.kernels.ops.moe.kpool_topk_transform"
_topk_patched = False


def _apply_topk(target_module, gcn):
    """Rebind sglang's ``fast_kpool_topk_transform_fused`` to the torch
    fallback (``vkernels_dsa_topk``)."""
    try:
        import vkernels_dsa_topk
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            "[sitecustomize] topk-torch patch: could not import vkernels_dsa_topk "
            f"({exc!r}); NOT patching (sglang will use the TVM JIT path).\n"
        )
        return
    target_module.fast_kpool_topk_transform_fused = (
        vkernels_dsa_topk.fast_kpool_topk_transform_fused
    )
    sys.stderr.write(
        "[sitecustomize] topk-torch patch APPLIED on " + gcn + ": "
        "fast_kpool_topk_transform_fused->torch (vkernels #56 bridge). "
        "NOT graph-safe (per-batch .item() syncs); fine for the smoke probe.\n"
    )


class _TopkWrapLoader(importlib.abc.Loader):
    """Runs the real loader (defines the JIT-backed fast_kpool_topk_transform_fused),
    then — once, on gfx942 with GLM53_TOPK_TRANSFORM_BACKEND=torch — rebinds it."""

    def __init__(self, real, name):
        self._real = real
        self._name = name

    def create_module(self, spec):
        if hasattr(self._real, "create_module"):
            return self._real.create_module(spec)
        return None

    def exec_module(self, module):
        self._real.exec_module(module)
        global _topk_patched
        if _topk_patched:
            return
        _topk_patched = True
        if _os.environ.get("GLM53_TOPK_TRANSFORM_BACKEND", "") != "torch":
            return
        ok, gcn = _supports_current_device()
        if ok:
            _apply_topk(module, gcn)


class _TopkPatcher(importlib.abc.MetaPathFinder):
    """Finds sglang's ``kernels.ops.moe.kpool_topk_transform`` and wraps its
    loader so the torch rebind runs exactly once, right after the module loads
    (the engine's first decode top-k). Returns None for every other module."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _TOPK_TARGET or _topk_patched:
            return None
        # Cheap env gate: if the torch bridge is off, never wrap (zero overhead;
        # sglang keeps its native JIT path). The gfx942 check happens in
        # exec_module (where torch is certain to be importable).
        if _os.environ.get("GLM53_TOPK_TRANSFORM_BACKEND", "") != "torch":
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is None:
                continue
            spec.loader = _TopkWrapLoader(spec.loader, fullname)
            return spec
        return None


# Install at startup. The env gate in find_spec makes this zero-work when the
# torch bridge is off; when on, it acts only on sglang's first kpool_topk import.
sys.meta_path.insert(0, _TopkPatcher())


# =============================================================================
# Forward-probe (GLM53_FWD_PROBE=1): first-forward-only per-op logger
# =============================================================================
import os as _os

_FWD_PROBE_ON = _os.environ.get("GLM53_FWD_PROBE", "0") == "1"
_FWD_PROBE_TARGET = "sglang.srt.models.glm5_next"
_fwd_probe_installed = False
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
        _fwd_log(False, f">>>  {type(module).__name__} (FIRST-FORWARD-START) <<<")
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
        _fwd_log(False, f"<<<  {type(module).__name__} (FIRST-FORWARD-COMPLETE) >>>")
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
    global _fwd_probe_installed
    if _fwd_probe_installed:
        return
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
    _fwd_probe_installed = True
    sys.stderr.write(
        "[sitecustomize] fwd-probe INSTALLED: logs each decoder layer + "
        "self_attn + mlp ENTER/EXIT on the FIRST forward; the last line "
        "before a stall names the hanging op.\n"
    )


class _FwdProbeLoader(importlib.abc.Loader):
    def __init__(self, real, name):
        self._real = real
        self._name = name

    def create_module(self, spec):
        if hasattr(self._real, "create_module"):
            return self._real.create_module(spec)
        return None

    def exec_module(self, module):
        self._real.exec_module(module)
        _install_fwd_probe(module)


class _FwdProbeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _FWD_PROBE_TARGET or _fwd_probe_installed:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is None:
                continue
            spec.loader = _FwdProbeLoader(spec.loader, fullname)
            return spec
        return None


if _FWD_PROBE_ON:
    sys.meta_path.insert(0, _FwdProbeFinder())
