#!/usr/bin/env bash
# Build the qwen36-dgx-spark serving image: a thin overlay on the golden GB10
# SGLang image (sglang-golden-gb10). Host-side step — no GPU needed. Run once
# after the golden GB10 image is present on this host, then run
# serve_qwen36_dgx_spark.sh.
#
#   bash build_image.sh
#
# What it does, in order:
#   1. preflights the golden base image + the two vendored source trees
#   2. stages sglang-src.tar.gz + entrypoint-src.tar.gz into a temp context
#   3. docker build -t $IMAGE -f Dockerfile.overlay <temp-context>
#   4. removes the temp context (the tarballs, like the original bring-up)
#
# Prerequisites (all relative to this recipe; override via env):
#   - sglang-golden-gb10 image present (built on a sibling DGX Spark, ds5)
#   - vendored sglang source at $SGLANG_SRC (default ./sglang-src; needs
#     python/pyproject.toml) — release/v0.5.16 + latest cherry-picks. This is
#     a LOCAL-ONLY checkout on dgx-spark (no git remote), so it must be staged
#     manually; the preflight below prints exactly what to do if it is missing.
#   - s3er entrypoint source at $ENTRYPOINT_SRC (default ./entrypoint-src;
#     needs pyproject.toml + s3er/).
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------- deployment ----
# Defaults keep the recipe self-contained: the vendored source checkouts sit
# next to the recipe and the build context is a temp dir. Override the paths if
# your vendored trees live elsewhere (e.g. an existing serving-stack checkout).
SGLANG_SRC="${SGLANG_SRC:-${RECIPE_DIR}/sglang-src}"
ENTRYPOINT_SRC="${ENTRYPOINT_SRC:-${RECIPE_DIR}/entrypoint-src}"
BASE_IMAGE="${BASE_IMAGE:-sglang-golden-gb10:latest}"
IMAGE="${IMAGE:-s3er-qwen36-dgx-spark}"

# ----------------------------------------------------------- preflight ----
# The golden GB10 image is built on a sibling DGX Spark (ds5) and is NOT in any
# registry. Without it the FROM in Dockerfile.overlay fails with
# "manifest unknown". Print the exact transfer from a host that already has it.
if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  echo "FATAL: base image ${BASE_IMAGE} not found on this host." >&2
  echo "       It is built on a sibling DGX Spark (ds5) and not in any registry." >&2
  echo "       On ds5, transfer it to this host ($(hostname)) with:" >&2
  echo "         docker save ${BASE_IMAGE} | gzip | ssh $(hostname) 'gunzip | docker load'" >&2
  exit 1
fi

[ -d "${SGLANG_SRC}" ] || { echo "FATAL: vendored sglang source not found: ${SGLANG_SRC}" >&2
  echo "       Place a checkout of sglang (release/v0.5.16 + latest cherry-picks)" >&2
  echo "       at ${SGLANG_SRC} (needs python/pyproject.toml), or set SGLANG_SRC." >&2
  echo "       On dgx-spark this is a local-only checkout (no git remote)." >&2
  exit 1; }
[ -f "${SGLANG_SRC}/python/pyproject.toml" ] || {
  echo "FATAL: ${SGLANG_SRC}/python/pyproject.toml missing" >&2
  echo "       (need the sglang \`python/\` package the Dockerfile reinstalls)" >&2
  exit 1
}
[ -d "${ENTRYPOINT_SRC}" ] || { echo "FATAL: vendored s3er entrypoint source not found: ${ENTRYPOINT_SRC}" >&2
  echo "       Place a checkout of the s3er entrypoint at ${ENTRYPOINT_SRC}" >&2
  echo "       (needs pyproject.toml + s3er/), or set ENTRYPOINT_SRC." >&2
  exit 1; }
[ -f "${ENTRYPOINT_SRC}/pyproject.toml" ] || {
  echo "FATAL: ${ENTRYPOINT_SRC}/pyproject.toml missing (need the s3er package)" >&2
  exit 1
}

# ------------------------------------------------------------- stage -----
# Temp build context: only the Dockerfile + the two tarballs. Keeps the
# (large, changing) vendored checkouts out of the context Docker sends to the
# daemon. The tarballs archive the CONTENTS of each source dir (not the dir
# itself): the Dockerfile does `tar -xzf sglang-src.tar.gz -C sglang-src` then
# `pip install ./python`, so the extract dir must contain `python/` directly
# (and `s3er/` + `pyproject.toml` for the entrypoint).
BUILD_CTX="$(mktemp -d)"
trap 'rm -rf "${BUILD_CTX}"' EXIT
cp "${RECIPE_DIR}/Dockerfile.overlay" "${BUILD_CTX}/Dockerfile"

echo "Staging sglang-src.tar.gz from ${SGLANG_SRC} ..."
tar -czf "${BUILD_CTX}/sglang-src.tar.gz" -C "${SGLANG_SRC}" .
echo "Staging entrypoint-src.tar.gz from ${ENTRYPOINT_SRC} ..."
tar -czf "${BUILD_CTX}/entrypoint-src.tar.gz" -C "${ENTRYPOINT_SRC}" .

# -------------------------------------------------------------- build ----
echo "Building ${IMAGE} (overlay on ${BASE_IMAGE}; ~30s, pure-Python) ..."
docker build -t "${IMAGE}" -f "${BUILD_CTX}/Dockerfile" "${BUILD_CTX}"

echo ""
echo "Done. Image: ${IMAGE}"
echo "Next: bash ${RECIPE_DIR}/serve_qwen36_dgx_spark.sh"
