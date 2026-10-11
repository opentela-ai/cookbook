#!/usr/bin/env bash
# One-shot bring-up: fetch the ollama/otela binaries if missing → start ollama
# (pulls the model) → register the otela sidecar. Knobs: README.md
set -euo pipefail
cd "$(dirname "$0")"

DEPLOY_DIR="${DEPLOY_DIR:-$(pwd)/run}"
[ -x "${OLLAMA_BIN:-${DEPLOY_DIR}/ollama/bin/ollama}" ] || bash download_ollama.sh
# Stage the latest official OpenTela (arm64 — DGX Spark) under $DEPLOY_DIR/otela;
# rerun-safe: skips if the staged release is already current, FORCE=1 re-downloads.
OTELA_DIR="${OTELA_DIR:-${DEPLOY_DIR}/otela}" bash ../../../../../meta/common/download_otela.sh

bash serve_qwen3_ollama.sh            # pulls the model, waits for readiness
bash register_qwen3_otela.sh daemon
