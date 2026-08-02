#!/usr/bin/env bash
# Serve Qwen/Qwen3.6-35B-A3B-FP8 on a DGX Spark (single NVIDIA GB10, sm_121,
# aarch64, 122 GB unified memory) inside the s3er-qwen36-dgx-spark container,
# and wait for it to be ready. The OpenTela sidecar (a separate host process)
# is started by register_qwen36_otela.sh — this script does NOT register
# anything; run register after this one prints READY.
#
# Prerequisites:
#   - s3er-qwen36-dgx-spark image built (bash build_image.sh)
#   - Model downloaded to $MODEL (FP8 shard tree: layers-N.safetensors)
#
# Usage:
#   bash serve_qwen36_dgx_spark.sh           # start (waits for readiness)
#   bash serve_qwen36_dgx_spark.sh stop      # stop the container
#
# Topology (no scheduler; single box):
#
#   docker container (qwen36-dgx-spark, --network host)  :$SERVE_PORT
#        ^   /health = 200 once weights are loaded
#        |   127.0.0.1:$SERVE_PORT   (host netns == container netns)
#        |
#   otela sidecar (register_qwen36_otela.sh, host proc)  :43905 libp2p
#        --> bootstrap peer (remote head)
#
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------- deployment ----
# Defaults keep the recipe self-contained: all runtime state (logs,
# last_service.env, otela config) lives under $DEPLOY_DIR, which defaults to a
# `run/` dir next to the recipe. Override DEPLOY_DIR to put it on a larger FS.
DEPLOY_DIR="${DEPLOY_DIR:-${RECIPE_DIR}/run}"
MODEL="${MODEL:-${HOME}/models/Qwen3.6-35B-A3B-FP8}"
IMAGE="${IMAGE:-s3er-qwen36-dgx-spark}"
SERVE_PORT="${SERVE_PORT:-30000}"
CONTAINER="${CONTAINER:-qwen36-dgx-spark}"
TP_SIZE="${TP_SIZE:-1}"
LAST_SERVICE_ENV="${LAST_SERVICE_ENV:-${DEPLOY_DIR}/last_service.env}"

# ----------------------------------------------------------- sglang flags ----
# --sglang-* flags are forwarded by the s3er entrypoint to sglang.
# attention-backend triton: hybrid GDN (Gated DeltaNet) on Blackwell rejects
#   torch_native; triton is the only backend that accepts the hybrid
#   linear+full attention schedule this model uses.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.6-35B-A3B-FP8}"   # org/model-name form; see ../../../../../conventions/
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
# mem-fraction-static 0.85: leaves headroom for the ~29 GB Mamba/SSM + conv
#   state that sglang does NOT count in its KV budget (documented in the
#   original bring-up), plus CUDA graph capture, within the 122 GB unified mem.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
# reasoning-parser qwen3: model emits <think>...</think>; the parser strips it
#   into `reasoning_content`.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
# How long to poll /v1/models before giving up (seconds). FP8 weights + SSM
# state load on a single GB10 takes several minutes; 600s matched the original
# bring-up. Raise for cold loads.
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
  echo "FATAL: image ${IMAGE} not found. Build it first with:" >&2
  echo "       bash ${RECIPE_DIR}/build_image.sh" >&2
  exit 1
}
[ -d "${MODEL}" ] || { echo "FATAL: weights not found: ${MODEL}" >&2
  echo "       Download the Qwen3.6-35B-A3B-FP8 shard tree (layers-N.safetensors," >&2
  echo "       config.json, chat_template.jinja, ...) to ${MODEL}, or set MODEL." >&2
  exit 1; }
mkdir -p "${DEPLOY_DIR}"

# -------------------------------------------------------------- launch ----
docker rm -f "${CONTAINER}" 2>/dev/null || true
# --network host: the otela sidecar (separate host process) reaches the engine
#   at 127.0.0.1:${SERVE_PORT}; a docker bridge would NAT the port away from it.
# --privileged: the golden image's CRIU checkpoint tooling needs it.
# --ulimit stack=67108864: 64 MB stack; sglang's hybrid-GDN call stack overflows
#   the default 8 MB.
# -v ${MODEL}:${MODEL}: the original bring-up mounted all of $HOME (rw) so the
#   model path was valid inside and sglang/transformers could write tokenizer/
#   JIT caches into the model tree. Only the model tree is actually needed, so
#   mount just that (rw, same reason) at the same absolute path. (sglang Triton
#   JIT caches stay in the container and are cold each start — add a cache mount
#   if you want warm.)
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
        echo "Next: register on OpenTela with bash ${RECIPE_DIR}/register_qwen36_otela.sh daemon"
        exit 0
    fi
    sleep 5
done
echo "TIMEOUT: server not ready after ${HEALTH_TIMEOUT}s. Check: docker logs ${CONTAINER}" >&2
exit 1
