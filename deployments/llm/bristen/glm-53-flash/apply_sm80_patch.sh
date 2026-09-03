#!/bin/bash
# Apply SM80 (A100) FP8 compute patches to a copy of the Beverin GLM-5.3
# SGLang overlay.  The patched tree is placed under $DEPLOY_DIR/patches_full
# and should be prepended to PYTHONPATH.
set -euo pipefail

: "${DEPLOY_DIR:=/capstor/scratch/cscs/xyao/glm-53-flash-bristen}"
# OVL is the top-level GLM-5.3 overlay directory (same as the sbatch).
: "${OVL:=/capstor/scratch/cscs/xyao/glm-53-flash-beverin/overlay}"
PATCH_DIR="$DEPLOY_DIR/patches_full"
SRC_DIR="$OVL/sgl-workspace/sglang/python/sglang"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$PATCH_DIR"

if [ ! -f "$SRC_DIR/__init__.py" ]; then
    echo "FATAL: overlay source not found: $SRC_DIR" >&2
    exit 1
fi

echo "[$(date -Is)] copying overlay sglang tree to $PATCH_DIR ..."
rm -rf "$PATCH_DIR/sglang"
cp -a "$SRC_DIR" "$PATCH_DIR/sglang"

echo "[$(date -Is)] applying SM80 FP8 compute patched sources ..."
cp "$SCRIPT_DIR/patched_sources/sglang/kernels/ops/quantization/fp8_kernel.py" \
   "$PATCH_DIR/sglang/kernels/ops/quantization/fp8_kernel.py"
cp "$SCRIPT_DIR/patched_sources/sglang/kernels/ops/moe/fused_moe_triton_kernels.py" \
   "$PATCH_DIR/sglang/kernels/ops/moe/fused_moe_triton_kernels.py"

# vkernels #60: kpool-cache path for the DSA indexer (SM80 has no fp8e4nv
# Triton). Dispatches on the cache dtype: uint8 -> legacy fp8+scale LAYOUT
# but vkernels dsa_kpool compute + torch requant store (the bristen serving
# shape -- allocator/readers/offload untouched); bf16 -> native vkernels
# bf16-cache layout. Requires the vkernels python package importable
# (pure-Python fallback works, compiled/CUDA backend for serving).
mkdir -p "$PATCH_DIR/sglang/srt/layers/attention/dsa"
cp "$SCRIPT_DIR/patched_sources/sglang/srt/layers/attention/dsa/kpool_fp8_index.py" \
   "$PATCH_DIR/sglang/srt/layers/attention/dsa/kpool_fp8_index.py"

# SM80 act_quant for the DSA indexer decode path: _act_quant_kernel stores
# through *fp8e4nv pointers (Hopper+ only in Triton) and killed jobs 82822/
# 83091 at forward_absorb_prepare -> act_quant on the first decode. The
# patched act_quant dispatches SM80 to a torch equivalent (software fp8
# casts, same fp8+scale bytes/shapes).
cp "$SCRIPT_DIR/patched_sources/sglang/kernels/ops/attention/dsa/triton_kernel.py" \
   "$PATCH_DIR/sglang/kernels/ops/attention/dsa/triton_kernel.py"

# sitecustomize: on SM80, rebind deep_gemm's fp8_paged_mqa_logits /
# fp8_mqa_logits / get_paged_mqa_logits_metadata (SM90+-only JIT) to pure-
# torch fallbacks for the DSA-indexer prefill/decode logits. No-op elsewhere
# (non-SM80 capability check inside). patches_full is first on PYTHONPATH, so
# CPython imports this automatically at interpreter startup.
cp "$SCRIPT_DIR/patched_sources/sitecustomize.py" "$PATCH_DIR/sitecustomize.py"

# Hardened post-warmup GC freeze: the single POST to /freeze_gc raced
# uvicorn's accept loop on job 83153 (connection refused), the unfrozen
# Tokenizer Manager then wedged its event loop under gen2 GC storms and the
# API never came up. Retry + direct in-process fallback.
cp "$SCRIPT_DIR/patched_sources/sglang/srt/entrypoints/http_server.py" \
   "$PATCH_DIR/sglang/srt/entrypoints/http_server.py"

# CUDA-graph capture fix for GLM-5.3 + PP2: the decode capture dummy buffers
# (DecodeInputBuffers.create + the legacy _allocate_decode_buffers helper)
# skipped provisioning pp_proxy_tensors["residual"] when is_mhc
# (hc_hidden_size set), but GLM-5.3 -- though hybrid-channel -- still passes
# a SEPARATE residual between pipeline stages (its forward reads
# pp_proxy_tensors["residual"] unconditionally on non-first ranks). The
# KeyError killed graph capture (job 83167). Always provision residual,
# matching the eager scheduler path (scheduler_pp_mixin.py).
cp "$SCRIPT_DIR/patched_sources/sglang/srt/model_executor/runner_utils/buffers.py" \
   "$PATCH_DIR/sglang/srt/model_executor/runner_utils/buffers.py"
cp "$SCRIPT_DIR/patched_sources/sglang/srt/model_executor/runner/base_runner.py" \
   "$PATCH_DIR/sglang/srt/model_executor/runner/base_runner.py"

echo "[$(date -Is)] SM80 patch tree ready at $PATCH_DIR/sglang"
