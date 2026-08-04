#!/usr/bin/env bash
# Serve ollama/qwen3:1.7b on a DGX Spark (single NVIDIA GB10, sm_121, aarch64,
# 122 GB unified memory) with Ollama, and wait for it to be ready. The OpenTela
# sidecar (a separate host process) is started by register_qwen3_otela.sh —
# this script does NOT register anything; run register after this one prints
# READY.
#
# Why Ollama (not sglang): Ollama ships a single static binary with **bundled
# CUDA v13 libs that include sm_121** (compute capability 12.1), so it runs on
# the GB10 out of the box — no golden image, no sgl-kernel build, no vendored
# sglang. A 1.7 B model loads in seconds and fits trivially in 122 GB.
#
# Prerequisites:
#   - Ollama arm64 binary at $OLLAMA_BIN (default $RECIPE_DIR/run/ollama/bin/ollama).
#     download_ollama.sh fetches it; or set OLLAMA_BIN to an existing install.
#
# Usage:
#   bash serve_qwen3_ollama.sh           # start (pulls model, waits for readiness)
#   bash serve_qwen3_ollama.sh stop      # stop ollama serve
#
# Topology (no scheduler; single box):
#
#   ollama serve (host process, 127.0.0.1:$SERVE_PORT)   / = 200, /v1/models ready
#        ^
#        | 127.0.0.1:$SERVE_PORT
#        |
#   otela sidecar (register_qwen3_otela.sh, host proc)   :43905 libp2p
#        --> bootstrap peer (remote head, ocf-1)
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------- deployment ----
# Defaults keep the recipe self-contained: all runtime state (logs, the pulled
# model blobs, otela config) lives under $DEPLOY_DIR, which defaults to a
# `run/` dir next to the recipe. Override DEPLOY_DIR to put it on a larger FS.
DEPLOY_DIR="${DEPLOY_DIR:-${RECIPE_DIR}/run}"
OLLAMA_BIN="${OLLAMA_BIN:-${DEPLOY_DIR}/ollama/bin/ollama}"
# OLLAMA_MODELS must be an absolute path; default to $DEPLOY_DIR/models so the
# blobs live with the recipe instead of polluting ~/.ollama.
OLLAMA_MODELS="${OLLAMA_MODELS:-${DEPLOY_DIR}/models}"
SERVE_PORT="${SERVE_PORT:-11434}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:${SERVE_PORT}}"
LOGFILE="${LOGFILE:-${DEPLOY_DIR}/ollama.log}"
PIDFILE="${PIDFILE:-${DEPLOY_DIR}/ollama.pid}"
LAST_SERVICE_ENV="${LAST_SERVICE_ENV:-${DEPLOY_DIR}/last_service.env}"

# ----------------------------------------------------------- model ----
# BASE_TAG is the upstream Ollama tag pulled from ollama.com. SERVED_MODEL_NAME
# is the org/model-name identity published on OpenTela (see ../../../../../conventions/).
# The alias below makes Ollama's /v1/models report SERVED_MODEL_NAME instead of
# BASE_TAG, so all three naming points (engine /v1/models, otela identity_group,
# client `model` field) agree on a single string.
BASE_TAG="${BASE_TAG:-qwen3:1.7b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ollama/qwen3:1.7b}"

# ----------------------------------------------------------- subcommand ----
case "${1:-start}" in
  stop)
    if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      kill "$(cat "${PIDFILE}")" && echo "Stopped ollama (pid $(cat "${PIDFILE}"))"
      rm -f "${PIDFILE}"
    else
      echo "ollama not running"
    fi
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
[ -x "${OLLAMA_BIN}" ] || { echo "FATAL: Ollama binary not found/executable: ${OLLAMA_BIN}" >&2
  echo "       Run bash ${RECIPE_DIR}/download_ollama.sh to fetch it," >&2
  echo "       or set OLLAMA_BIN to an existing ollama (e.g. /usr/local/bin/ollama)." >&2
  exit 1; }
mkdir -p "${OLLAMA_MODELS}" "${DEPLOY_DIR}"

# ----------------------------------------------------------- start serve ----
# Idempotent: if ollama is already serving on $SERVE_PORT, reuse it.
if curl -sf "http://${OLLAMA_HOST}/" >/dev/null 2>&1; then
  echo "ollama already serving on http://${OLLAMA_HOST}"
else
  echo "Starting ollama serve (logs -> ${LOGFILE})..."
  # OLLAMA_HOST sets the listen address/port; OLLAMA_MODELS sets the blob dir.
  # No CUDA env needed — Ollama auto-detects the GB10 via its bundled CUDA 13
  # libs (verified: "inference compute id=0 library=CUDA compute=12.1 name=CUDA0
  # description=NVIDIA GB10 total=119.7 GiB"). The bundled CUDA 12 libs are
  # auto-skipped ("compute capability not in compiled architectures", cc=1210),
  # which is expected and harmless.
  OLLAMA_HOST="${OLLAMA_HOST}" OLLAMA_MODELS="${OLLAMA_MODELS}" \
    setsid "${OLLAMA_BIN}" serve </dev/null >"${LOGFILE}" 2>&1 &
  echo $! > "${PIDFILE}"
  sleep 3
  if ! kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
    echo "FATAL: ollama failed to start — check ${LOGFILE}" >&2
    exit 1
  fi
fi

# ----------------------------------------------------------- pull + alias ----
# Pull the base tag, then create an org/model-name alias from it via a
# Modelfile. The alias shares the base blobs (no duplicate download) but gets
# its own manifest, so /v1/models reports ollama/qwen3:1.7b with
# owned_by=ollama (instead of qwen3:1.7b with owned_by=library).
export OLLAMA_HOST="${OLLAMA_HOST}" OLLAMA_MODELS="${OLLAMA_MODELS}"

echo "Pulling ${BASE_TAG} (if not already present)..."
"${OLLAMA_BIN}" pull "${BASE_TAG}" >/dev/null

# Create the alias only if absent (idempotent). ollama create with a real file
# (not stdin) is required: `ollama create X -f -` fails with "no Modelfile or
# safetensors files found" on stdin.
if ! "${OLLAMA_BIN}" list 2>/dev/null | grep -q "^${SERVED_MODEL_NAME}\b"; then
  echo "Creating alias ${SERVED_MODEL_NAME} -> ${BASE_TAG}..."
  TMPF="$(mktemp)"
  printf 'FROM %s\n' "${BASE_TAG}" > "${TMPF}"
  "${OLLAMA_BIN}" create "${SERVED_MODEL_NAME}" -f "${TMPF}" >/dev/null
  rm -f "${TMPF}"
fi

# Remove the bare base tag so /v1/models lists ONLY the org/model-name alias.
# This is what otela auto-registers as the identity_group; leaving both would
# publish two identities (qwen3:1.7b AND ollama/qwen3:1.7b), which violates the
# single-org/model-name convention. The alias still works after the bare tag is
# removed because they share blobs. To keep both, comment this out.
if "${OLLAMA_BIN}" list 2>/dev/null | grep -q "^${BASE_TAG}\b"; then
  echo "Removing bare ${BASE_TAG} (alias ${SERVED_MODEL_NAME} keeps the blobs)..."
  "${OLLAMA_BIN}" rm "${BASE_TAG}" >/dev/null
fi

# ----------------------------------------------------------- readiness ----
echo "Waiting for /v1/models to report ${SERVED_MODEL_NAME} ..."
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-120}"
for _ in $(seq 1 $((HEALTH_TIMEOUT / 2))); do
  if curl -sf "http://${OLLAMA_HOST}/v1/models" 2>/dev/null \
      | grep -q "\"id\":\"${SERVED_MODEL_NAME}\""; then
    echo "READY! endpoint=http://${OLLAMA_HOST}"
    {
      echo "ENDPOINT=http://${OLLAMA_HOST}"
      echo "SERVED_MODEL_ID=${SERVED_MODEL_NAME}"
      echo "OLLAMA_PID=$(cat "${PIDFILE}" 2>/dev/null || echo unknown)"
      echo "READY_AT=$(date --iso-8601=seconds)"
    } > "${LAST_SERVICE_ENV}"
    echo ""
    echo "Next: register on OpenTela with bash ${RECIPE_DIR}/register_qwen3_otela.sh daemon"
    exit 0
  fi
  sleep 2
done
echo "TIMEOUT: ${SERVED_MODEL_NAME} not ready after ${HEALTH_TIMEOUT}s. Check: tail -80 ${LOGFILE}" >&2
exit 1
