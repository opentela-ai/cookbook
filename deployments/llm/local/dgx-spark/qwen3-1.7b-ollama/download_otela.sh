#!/usr/bin/env bash
# Download the OpenTela v0.2.3 arm64 binary (released as opentela-arm64) into
# $DEPLOY_DIR/otela/otela so register_qwen3_otela.sh can find it at the default
# $OTELA_BIN. Run once; no GPU needed.
#
#   bash download_otela.sh
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="${DEPLOY_DIR:-${RECIPE_DIR}/run}"
OTELA_VERSION="${OTELA_VERSION:-v0.2.3}"
DEST_DIR="${DEPLOY_DIR}/otela"
DEST="${DEST_DIR}/otela"

mkdir -p "${DEST_DIR}"
if [ -x "${DEST}" ]; then
  echo "otela already present at ${DEST}"
  "${DEST}" version 2>&1 | head -1 || true
  exit 0
fi

URL="https://github.com/eth-easl/OpenTela/releases/download/${OTELA_VERSION}/opentela-arm64"
echo "Downloading ${URL} ..."
curl -fL -o "${DEST}" "${URL}"
chmod +x "${DEST}"

echo "Done. ${DEST}"
"${DEST}" version 2>&1 | head -1
