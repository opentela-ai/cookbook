#!/usr/bin/env bash
# One-shot bring-up: fetch the ollama/otela binaries if missing → start ollama
# (pulls the model) → register the otela sidecar. Knobs: README.md
set -euo pipefail
cd "$(dirname "$0")"

DEPLOY_DIR="${DEPLOY_DIR:-$(pwd)/run}"
[ -x "${OLLAMA_BIN:-${DEPLOY_DIR}/ollama/bin/ollama}" ] || bash download_ollama.sh
[ -x "${OTELA_BIN:-${DEPLOY_DIR}/otela/otela}" ] || bash download_otela.sh

bash serve_qwen3_ollama.sh            # pulls the model, waits for readiness
bash register_qwen3_otela.sh daemon
