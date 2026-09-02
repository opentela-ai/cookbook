#!/bin/bash
set -uo pipefail
RANK="${SLURM_NODEID:-0}"
NODE="bristen-ln001"
echo "[rank $RANK] $(hostname) node-rank=$RANK dist-init=$HEAD:$MASTER_PORT serve=$SERVE_PORT ctx=$CTX_LEN tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE"

# Cache / tmp are already exported by the parent shell.
mkdir -p "$TRITON_CACHE_DIR" "$HF_HOME" "$TMPDIR" 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

SGLANG_ARGS=(
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --host 0.0.0.0 --port "$SERVE_PORT"
  --tp-size "$TP_SIZE"
  --pp-size "$PP_SIZE"
  --ep-size "$EP_SIZE"
  --context-length "$CTX_LEN"
  --mem-fraction-static "$GPU_MEM_UTIL"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --watchdog-timeout "$WATCHDOG_TIMEOUT"
  --dist-timeout "$DIST_TIMEOUT"
  --enable-metrics
  --quantization fp8
)

# GLM-5.3 uses index_kpool=4 with tail selection. On A100 (SM80) the default
# flashmla_sparse DSA prefill backend does not support kpool>1 tails; the
# dsa_backend overlay patches _resolve_kpool_tail_backend to route SM80 to
# the TileLang sparse prefill backend instead. Override with
# DSA_PREFILL_BACKEND=fa3 to experiment, but note FA3 on SM80 rejects the
# different QK/V head dims used by GLM-5.3.
export DSA_PREFILL_BACKEND="${DSA_PREFILL_BACKEND:-}"
[ -n "${DSA_PREFILL_BACKEND:-}" ] && \
  SGLANG_ARGS+=(--dsa-prefill-backend "$DSA_PREFILL_BACKEND")

if [ "${NNODES:-1}" -gt 1 ]; then
  SGLANG_ARGS+=(
    --nnodes "$NNODES" --node-rank "$RANK"
    --dist-init-addr "${HEAD_IP:-127.0.0.1}:$MASTER_PORT"
  )
fi

# GLM-5.3 ships chat_template.jinja separately; pass it explicitly.
if [ -f "$MODEL_PATH/chat_template.jinja" ]; then
  SGLANG_ARGS+=( --chat-template "$MODEL_PATH/chat_template.jinja" )
fi

[ "${DISABLE_CUDA_GRAPH:-1}" = "1" ] && SGLANG_ARGS+=(--disable-cuda-graph)
[ "${SKIP_SERVER_WARMUP:-1}" = "1" ] && SGLANG_ARGS+=(--skip-server-warmup)
[ -n "${LOAD_FORMAT:-}" ] && [ "$LOAD_FORMAT" != "auto" ] && \
  SGLANG_ARGS+=(--load-format "$LOAD_FORMAT")

# GLM-5.3 uses hybrid linear-attention layers on A100; the auto mamba cache
# budget overshoots the tiny remaining memory after FP8 weights. Cap it.
[ -n "${MAX_MAMBA_CACHE_SIZE:-}" ] && \
  SGLANG_ARGS+=(--max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE")

# A100 (SM80) does not support the cooperative-groups cluster features used by
# the DeepSeek-V4 topk_v2 JIT kernel and the DeepGEMM HC prenorm kernel.
export SGLANG_OPT_USE_TOPK_V2="${SGLANG_OPT_USE_TOPK_V2:-0}"
export SGLANG_OPT_DEEPGEMM_HC_PRENORM="${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-0}"

# GLM-5.3-Flash weights are FP8 (e4m3). On A100 (SM80) none of the optimized
# FP8 GEMM backends are available:
#   - DeepGEMM / FlashInfer TRT-LLM / CUTLASS require SM90+ or SM100+.
#   - torch._scaled_mm requires SM89+ or ROCm MI300+.
#   - Triton maps torch.float8_e4m3fn to tl.float8e4nv, which is Hopper-only.
# The practical default on SM80 is therefore the Triton backend, but it will
# fail to compile until the kernels are patched to upcast FP8 to bf16 before
# the tl.dot (FP8 storage, bf16 compute). See README.md for details.
export MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-triton}"
export FP8_GEMM_RUNNER_BACKEND="${FP8_GEMM_RUNNER_BACKEND:-triton}"
[ -n "${MOE_RUNNER_BACKEND:-}" ] && \
  SGLANG_ARGS+=(--moe-runner-backend "$MOE_RUNNER_BACKEND")
[ -n "${FP8_GEMM_RUNNER_BACKEND:-}" ] && \
  SGLANG_ARGS+=(--fp8-gemm-backend "$FP8_GEMM_RUNNER_BACKEND")

echo "[rank $RANK] launching SGLang args=${SGLANG_ARGS[*]}"
exec python3 -m sglang.launch_server "${SGLANG_ARGS[@]}"
