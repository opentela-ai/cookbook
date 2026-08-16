#!/usr/bin/env bash
# Serve Qwen/Qwen3.8-27B-FP8 on a DGX Spark (single NVIDIA GB10, sm_121,
# aarch64, 122 GB unified memory) REUSING the s3er-qwen36-dgx-spark image, and
# wait for it to be ready. The OpenTela sidecar (a separate host process) is
# started by register_qwen38_otela.sh — this script does NOT register
# anything; run register after this one prints READY.
#
# Why no build_image.sh here: Qwen3.8-27B-FP8 is the same hybrid-GDN
# (Gated DeltaNet + full attention + MTP) qwen3_5 family as Qwen3.6-35B-A3B-FP8,
# and the existing s3er-qwen36-dgx-spark image already contains
# sglang.srt.models.qwen3_5 (verified by importing it in the image). So the
# model runs on the same image — only the weight path and served-model name
# differ. If a future sglang bump is needed for Qwen3.8, rebuild via the
# qwen36-35b-a3b recipe (same base + vendored sglang).
#
# Differences from qwen36-35b-a3b (the sibling recipe this is derived from):
#   - dense 27B (not MoE). VERIFIED on ds6 (2026-08-14): Load weight end
#     elapsed=182.55 s, mem usage=29.12 GB (vs the qwen36 MoE's 34.7 GB); Mamba
#     ssm_state=27.98 GB + conv_state=0.55 GB (=28.5 GB, ~same as qwen36). Cold
#     start end-to-end ~7 min (prefill CUDA graph capture alone is 228.6 s).
#   - vision-language model (Qwen3_5ForConditionalGeneration, vision_config +
#     text_config). The engine serves the text LM; --reasoning-parser qwen3 and
#     --tool-call-parser qwen3_coder are kept from qwen36 (same qwen3_5 family)
#     and VERIFIED for 3.8 on ds6 (see the flag comments + README Verify).
#
# Prerequisites:
#   - s3er-qwen36-dgx-spark image present (built once via the qwen36 recipe)
#   - Model downloaded to $MODEL (FP8 shard tree: layers-N.safetensors)
#
# Usage:
#   bash serve_qwen38_dgx_spark.sh           # start (waits for readiness)
#   bash serve_qwen38_dgx_spark.sh stop      # stop the container
#
# Topology (no scheduler; single box):
#
#   docker container (qwen38-dgx-spark, --network host)  :$SERVE_PORT
#        ^   /health = 200 once weights are loaded
#        |   127.0.0.1:$SERVE_PORT   (host netns == container netns)
#        |
#   otela sidecar (register_qwen38_otela.sh, host proc)  :43905 libp2p
#        --> bootstrap peer (remote head)
#
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------- deployment ----
# Defaults keep the recipe self-contained: all runtime state (logs,
# last_service.env, otela config) lives under $DEPLOY_DIR, which defaults to a
# `run/` dir next to the recipe. Override DEPLOY_DIR to put it on a larger FS.
DEPLOY_DIR="${DEPLOY_DIR:-${RECIPE_DIR}/run}"
MODEL="${MODEL:-${HOME}/models/Qwen3.8-27B-FP8}"
# REUSE the qwen36 image: same qwen3_5 hybrid-GDN family, and that image already
# ships sglang.srt.models.qwen3_5 (verified). No rebuild needed for Qwen3.8.
IMAGE="${IMAGE:-s3er-qwen36-dgx-spark}"
SERVE_PORT="${SERVE_PORT:-30000}"
CONTAINER="${CONTAINER:-qwen38-dgx-spark}"
TP_SIZE="${TP_SIZE:-1}"
LAST_SERVICE_ENV="${LAST_SERVICE_ENV:-${DEPLOY_DIR}/last_service.env}"

# ----------------------------------------------------------- sglang flags ----
# --sglang-* flags are forwarded by the s3er entrypoint to sglang.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.8-27B-FP8}"   # org/model-name form; see ../../../../../conventions/
# attention-backend triton: hybrid GDN (Gated DeltaNet) on Blackwell rejects
#   torch_native; triton is the only backend that accepts the hybrid
#   linear+full attention schedule this model uses. Carried over from qwen36
#   (same qwen3_5 family).
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
# mem-fraction-static 0.80: VERIFIED on ds6 (2026-08-14). Cold load measured:
#   Load weight end elapsed=182.55 s, mem usage=29.12 GB (FP8 e4m3); Mamba Cache
#   ssm_state=27.98 GB + conv_state=0.55 GB (=28.5 GB, ~same as the qwen36 35B
#   MoE — same hybrid-GDN family); prefill CUDA graph 228.6 s / 3.31 GB; decode
#   graph 4.7 s / 0.47 GB; final available_gpu_mem=17.60 GB. 0.80 is safe and
#   leaves headroom for the multimodal (vision) path; raise toward 0.83-0.85 to
#   grow the KV cache if more concurrency is needed.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
# reasoning-parser qwen3: the model emits a thinking trace which the parser
#   strips into `reasoning_content`. VERIFIED on ds6 (2026-08-14): a "think
#   briefly" prompt returned populated reasoning_content with
#   reasoning_tokens=213; a routed request likewise returned reasoning_tokens.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
# tool-call-parser qwen3_coder: VERIFIED on ds6 (2026-08-14). Without it the
#   qwen36 engine registered no tool-call parser and the gateway could not serve
#   function calls. For Qwen3.8 a request with tools=[get_weather] returned a
#   proper tool_calls=[{function:{name:"get_weather",arguments:'{"city":"Paris"}'}}]
#   response. (Same qwen3_5 family as qwen36, so qwen3_coder is the right
#   parser here too.)
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
# How long to poll /v1/models before giving up (seconds). VERIFIED: cold FP8
# load + CUDA graph capture on a single GB10 reaches /v1/models in ~7 min, so
# 600s is a comfortable bound. Raise for very cold loads (first-ever JIT).
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

# ----------------------------------------------------------- subcommand ----
case "${1:-start}" in
  stop)
    docker rm -f "${CONTAINER}" 2>/dev/null && echo "Stopped ${CONTAINER}" \
      || echo "${CONTAINER} not running"
    exit 0
    ;;
  start|"")
    ;;
  *)
    echo "usage: $0 {start|stop}" >&2
    exit 2
    ;;
esac

# ----------------------------------------------------------- preflight ----
docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
  echo "FATAL: image ${IMAGE} not found." >&2
  echo "       This recipe reuses the qwen36-35b-a3b image. Build it once" >&2
  echo "       with the sibling recipe's build_image.sh:" >&2
  echo "         bash ${RECIPE_DIR}/../qwen36-35b-a3b/build_image.sh" >&2
  echo "       (or set IMAGE to an existing s3er image that ships" >&2
  echo "        sglang.srt.models.qwen3_5)." >&2
  exit 1; }
[ -d "${MODEL}" ] || { echo "FATAL: weights not found: ${MODEL}" >&2
  echo "       Download the Qwen3.8-27B-FP8 shard tree (layers-N.safetensors," >&2
  echo "       config.json, chat_template.jinja, ...) to ${MODEL}, or set MODEL." >&2
  echo "       On this host the qwen36 recipe downloaded its weights with:" >&2
  echo "         docker run --rm --network host -v \"\$HOME/models\":/models \\" >&2
  echo "           -e HF_HUB_OFFLINE=0 --entrypoint /bin/bash \\" >&2
  echo "           sglang-golden-gb10 -c 'hf download Qwen/Qwen3.8-27B-FP8 --local-dir /models/Qwen3.8-27B-FP8'" >&2
  exit 1; }
mkdir -p "${DEPLOY_DIR}"

# -------------------------------------------------------------- launch ----
docker rm -f "${CONTAINER}" 2>/dev/null || true
# --network host: the otela sidecar (separate host process) reaches the engine
#   at 127.0.0.1:${SERVE_PORT}; a docker bridge would NAT the port away from it.
# --privileged: the golden image's CRIU checkpoint tooling needs it.
# --ulimit stack=67108864: 64 MB stack; sglang's hybrid-GDN call stack overflows
#   the default 8 MB. Carried over from qwen36 (same hybrid-GDN family).
# -v ${MODEL}:${MODEL}: mount the model tree (rw, same absolute path) so sglang/
#   transformers can write tokenizer/JIT caches into the model tree (same reason
#   as qwen36). (sglang Triton JIT caches stay in the container and are cold
#   each start — add a cache mount if you want warm.)
docker run -d --privileged --gpus all --ipc=host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${MODEL}":"${MODEL}" \
    -e HF_HUB_OFFLINE=1 \
    -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    --name "${CONTAINER}" \
    --shm-size 16g \
    "${IMAGE}" \
    serve "${MODEL}" \
    --port "${SERVE_PORT}" --tp "${TP_SIZE}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --sglang-attention-backend="${ATTENTION_BACKEND}" \
    --sglang-mem-fraction-static="${MEM_FRACTION_STATIC}" \
    --sglang-reasoning-parser="${REASONING_PARSER}" \
    ${TOOL_CALL_PARSER:+--sglang-tool-call-parser="${TOOL_CALL_PARSER}"}

echo ""
echo "Container ${CONTAINER} started. Monitoring startup..."
echo "  endpoint: http://$(hostname -I | awk '{print $1}'):${SERVE_PORT}"
echo "  logs:     docker logs -f ${CONTAINER}"
echo "  state:    ${LAST_SERVICE_ENV}"
echo ""
echo "Waiting for server readiness (timeout ${HEALTH_TIMEOUT}s)..."
for _ in $(seq 1 $((HEALTH_TIMEOUT / 5))); do
    if curl -sf "http://localhost:${SERVE_PORT}/v1/models" >/dev/null 2>&1; then
        echo "READY! endpoint=http://localhost:${SERVE_PORT}"
        {
            echo "ENDPOINT=http://localhost:${SERVE_PORT}"
            echo "SERVED_MODEL_ID=${SERVED_MODEL_NAME}"
            echo "CONTAINER=${CONTAINER}"
            echo "READY_AT=$(date --iso-8601=seconds)"
        } > "${LAST_SERVICE_ENV}"
        echo ""
        echo "Next: register on OpenTela with bash ${RECIPE_DIR}/register_qwen38_otela.sh daemon"
        exit 0
    fi
    sleep 5
done
echo "TIMEOUT: server not ready after ${HEALTH_TIMEOUT}s. Check: docker logs ${CONTAINER}" >&2
exit 1
