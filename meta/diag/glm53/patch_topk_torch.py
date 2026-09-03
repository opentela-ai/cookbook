"""kpool top-k TRANSFORM torch-fallback bridge (vkernels #56,
GLM53_TOPK_TRANSFORM_BACKEND=torch).

sglang's decode DSA path calls ``fast_kpool_topk_transform_fused`` (the radix
top-k *transform*; ``kernels/ops/moe/kpool_topk_transform.py``), which routes
through the TVM JIT ``sgl_kernel_jit_kpool_topk_transform_{group_topk}``. That
JIT compiles on the ROCm container only after the ``cuda_fp16.h``->
``hip_runtime.h`` guard (cookbook 9b6ab33), and on the GLM-5.3 decode shape the
launched kernel does not return (vkernels #56: every scheduler_TP{0..3}_EP{0..3}
stuck in ``Dl`` state, ENTER/EXIT 584/0, no crash). vkernels ships the two
bookend kernels (``vk_hip_dsa_topk_logits`` #51, ``vk_hip_dsa_sparse_fwd`` #52)
but NOT this middle transform.

This hook lazily rebinds ``fast_kpool_topk_transform_fused`` to a
pure-PyTorch implementation (``vkernels_dsa_topk``) that is numerically
equivalent to ``kernels/jit/csrc/dsa/kpool_topk_transform.cuh:249-310``
(validated 7/7 by ``test_vkernels_dsa_topk.py`` on the container). Goals:
  1. UNBLOCK: if the hang is the JIT, decode serves immediately.
  2. LOCALISE: if the hang persists, it is definitively in a bookend kernel.
  3. REFERENCE: an oracle for the future HIP kernel (#56).

Gate: GLM53_TOPK_TRANSFORM_BACKEND=torch (default off -> the JIT is the
control, so the two paths are A/B-testable) AND gfx942 (the JIT is fine on
CUDA; the torch fallback is slower and only needed on MI300A). The env gate
is checked at install time AND at fire time (the gfx942 probe needs torch,
which is only certain to import at fire time).
"""
import os
import sys

from gfx942 import supports_current_device
from import_hook import run_after_import

_TARGET = "sglang.kernels.ops.moe.kpool_topk_transform"


def _apply(target_module, gcn):
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


def _install(target_module):
    # Re-check the env gate at fire time (mirrors the old find_spec +
    # exec_module double check; env is normally fixed by then).
    if os.environ.get("GLM53_TOPK_TRANSFORM_BACKEND", "") != "torch":
        return
    ok, gcn = supports_current_device()
    if ok:
        _apply(target_module, gcn)


if os.environ.get("GLM53_TOPK_TRANSFORM_BACKEND", "") == "torch":
    run_after_import(_TARGET, _install)
