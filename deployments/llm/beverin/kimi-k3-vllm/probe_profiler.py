#!/usr/bin/env python3
"""One-shot container probe: vLLM version + VLLM_TORCH_PROFILER_DIR support
+ the Kimi-K3 model's MLA / MoE layer names (for per-step profiling)."""
import glob, sys, importlib.util as u

print("PY", sys.version.split()[0])
try:
    import vllm
    print("VLLMVER", vllm.__version__)
except Exception as e:
    print("VLLMVER_ERR", str(e)[:120])

# Built-in API-triggered torch.profiler (the cleanest path).
hits = [p.split("site-packages/")[-1] for p in
        glob.glob("/opt/venv/lib/python*/site-packages/vllm/**/*.py", recursive=True)
        if "VLLM_TORCH_PROFILER_DIR" in open(p, errors="ignore").read()]
print("VLLM_TORCH_PROFILER_DIR", hits[:8])

try:
    from torch.profiler import profile, ProfilerActivity
    print("TORCH_PROFILER_OK", [a for a in dir(ProfilerActivity) if not a.startswith("_")])
except Exception as e:
    print("TORCH_PROFILER_ERR", str(e)[:120])

# Model layer structure: where do MLA + MoE live?
for mod in ["vllm.model_executor.models.kimi_k3",
            "vllm.model_executor.models.moonshot",
            "vllm.model_executor.layers.fused_moe.fused_moe",
            "vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe"]:
    print("MOD", mod, bool(u.find_spec(mod)))

# NCCL / all-to-all primitives the MoE path uses.
for mod in ["vllm.distributed.parallel_state",
            "vllm.distributed.device_communicators.pynccl_wrapper"]:
    print("DIST", mod, bool(u.find_spec(mod)))
print("PROBE_DONE")
