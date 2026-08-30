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
"""

import importlib.abc
import sys

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
    """Rebind sglang's ``tilelang_sparse_fwd`` to the vkernels HIP ctypes adapter."""
    try:
        import vkernels_dsa
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[sitecustomize] DSA-vkernels patch: could not import vkernels_dsa "
            f"({exc!r}); NOT patching (sglang will use its native tilelang path)\n"
        )
        return
    target_module.tilelang_sparse_fwd = vkernels_dsa.tilelang_sparse_fwd
    sys.stderr.write(
        "[sitecustomize] DSA-vkernels patch APPLIED: sglang DSA sparse-MLA "
        "forward now routes through vk_hip_dsa_sparse_fwd (PR #52) on "
        f"{gcn}, bypassing the tilelang tail_dim==0 JIT abort (#51).\n"
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
        if hasattr(self._real, "create_module"):
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
