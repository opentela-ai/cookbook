#!/usr/bin/env bash
# Download the Ollama arm64 binary (with bundled CUDA v12 + v13 libs) into
# $DEPLOY_DIR/ollama/ so serve_qwen3_ollama.sh can find it at the default
# $OLLAMA_BIN. Run once; no GPU needed.
#
#   bash download_ollama.sh
#
# The v0.32.5 arm64 release is a 1.5 GB .tar.zst (zstd-compressed) that
# extracts to bin/ollama + lib/ollama/. The lib/ tree bundles CUDA v13 libs
# that include sm_121 (compute 12.1), which is what makes Ollama run on the
# GB10 without a custom CUDA install. (The bundled CUDA v12 libs lack sm_121
# and are auto-skipped at startup — expected and harmless.)
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="${DEPLOY_DIR:-${RECIPE_DIR}/run}"
OLLAMA_VERSION="${OLLAMA_VERSION:-v0.32.5}"
DEST="${DEPLOY_DIR}/ollama"

mkdir -p "${DEST}"
if [ -x "${DEST}/bin/ollama" ]; then
  echo "Ollama already present at ${DEST}/bin/ollama"
  "${DEST}/bin/ollama" --version 2>&1 | head -1 || true
  exit 0
fi

URL="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-arm64.tar.zst"
echo "Downloading ${URL} ..."
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
curl -fL -o "${TMP}/ollama.tar.zst" "${URL}"

# tar --zstd is available on Ubuntu 24.04+ (ds5). On older systems install zstd
# or use `zstd -d | tar x`.
tar --zstd -xf "${TMP}/ollama.tar.zst" -C "${DEST}"
chmod +x "${DEST}/bin/ollama"

echo "Done. ${DEST}/bin/ollama"
"${DEST}/bin/ollama" --version 2>&1 | head -1
