"""sitecustomize — GLM-5.3-Flash engine shim DISPATCHER (beverin / MI300A).

Installed by build_overlay.sh into $OVL/pylib (FIRST on PYTHONPATH inside the
sglang-rocm EDF), so CPython auto-imports it at startup, before sglang runs.

This file is ONLY a dispatcher. Every patch/diagnostic is an individual
module in $GLM53_DIAG_DIR (exported by the sbatch; default
<cookbook>/meta/diag/glm53) and self-installs on import, gated by its own
env var / device check:

    patch_dsa_vk.py      tilelang DSA forward + topk logits -> vkernels HIP
                         (always on gfx942; the #51/#52 unblock)
    patch_topk_torch.py  kpool top-k transform -> torch bridge
                         (GLM53_TOPK_TRANSFORM_BACKEND=torch, #56)
    fwd_probe.py         first-forward per-op ENTER/EXIT logger + prefill
                         argmax (GLM53_FWD_PROBE=1)
    patch_dsa_sdpa.py    DSA prefill -> PyTorch SDPA on gfx942 (always on
                         gfx942)
    comp_capture.py      first-forward component/layer I/O dump for the
                         beverin-vs-clariden bisect (GLM53_COMP_CAPTURE=1)

Lazy by construction: nothing here imports sglang/tilelang/torch at startup.
Every patcher installs a sys.meta_path hook via meta/diag/import_hook
(run_after_import) that fires on the target module's first real import —
the eager-import-at-startup version cost ~4-5 min on a cold node and blew
the preflight gate (beverin job 612821). Failures are per-module and logged,
never fatal: a broken diagnostic must not take the engine down.
"""
import os
import sys

_DIAG = os.environ.get("GLM53_DIAG_DIR", "")
if not _DIAG:
    sys.stderr.write(
        "[sitecustomize] GLM53_DIAG_DIR is NOT set; engine shims DISABLED. "
        "The sbatch normally exports it (<cookbook>/meta/diag/glm53).\n"
    )
else:
    if _DIAG not in sys.path:
        sys.path.insert(0, _DIAG)
    for _mod in ("patch_dsa_vk", "patch_topk_torch", "fwd_probe", "patch_dsa_sdpa"):
        try:
            __import__(_mod)
        except Exception as _exc:  # noqa: BLE001
            sys.stderr.write(f"[sitecustomize] {_mod} import failed: {_exc!r}\n")
    if os.environ.get("GLM53_COMP_CAPTURE", "0") == "1":
        try:
            import comp_capture  # noqa: F401  (self-installing)
        except Exception as _exc:  # noqa: BLE001
            sys.stderr.write(
                f"[sitecustomize] comp_capture import failed: {_exc!r}\n")
