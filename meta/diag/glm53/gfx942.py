"""gfx942 — the MI300A device gate shared by the GLM-5.3 engine patches.

Probing torch/CUDA at sitecustomize time is unreliable (torch may not be
importable yet, or no GPU visible in a CPU-only preflight — the flake in jobs
616583/616594), so every caller probes LAZILY: at the first real forward /
module-import hook fire, where torch and a GPU are certain.
"""
import sys


def supports_current_device():
    """Return (is_gfx942, gcn_name) for CUDA device 0.

    True only on gfx942 (MI300A). Guards the engine patches so a CPU-only
    preflight or a non-MI300A node keeps sglang's native paths (harmless
    there). Returns (False, "") with a logged reason on any probe failure.
    """
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
