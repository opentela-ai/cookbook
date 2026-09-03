"""Lazy rebind of sglang's DSA ``tilelang_sparse_fwd`` (and the kpool top-k
logits) to the vkernels HIP kernels on MI300A (gfx942) — bypasses the tilelang
JIT abort (vkernels #51) with PR #52's ``vk_hip_dsa_sparse_fwd``. Ungated by
env: the gfx942 device gate (``gfx942.supports_current_device``) is the only
condition, probed at fire time.

WHY LAZY (sys.meta_path) — beverin job 612821
---------------------------------------------
An EARLIER version imported
``sglang.kernels.ops.attention.dsa.tilelang_kernel`` at startup. That import
is the ONLY thing in the recipe that pulls in the full ``sglang.kernels``
package + ``tilelang``/``aiter`` (``import sglang`` alone does NOT — those
submodules load lazily, at the first DSA forward). On a cold node that eager
import took ~4-5 min and blew the sbatch's preflight gate. The rebind itself
is pointless until sglang actually runs a DSA forward, so we DEFER it:
``import_hook.run_after_import`` fires ONLY when sglang itself imports
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
forward_extend at L3223 and forward_decode at L3525), so once this hook
patches the module on first import every DSA forward picks up the rebound
symbol. The MHC tilelang path is left untouched (still patched by
tilelang-mhc-reduce-hidden_block-for-mi300a-64KB-LDS.patch via
build_overlay.sh — that's the SECOND blocker, already handled).
"""
import sys

from gfx942 import supports_current_device
from import_hook import run_after_import

_TARGET = "sglang.kernels.ops.attention.dsa.tilelang_kernel"


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


def _install(target_module):
    ok, gcn = supports_current_device()
    if ok:
        _apply(target_module, gcn)


# Install at startup. Zero work: no sglang/tilelang/torch import. The hook
# acts only when sglang imports tilelang_kernel (the engine's first DSA
# forward); a CPU-only preflight never imports it, so the gate stays fast.
run_after_import(_TARGET, _install)
