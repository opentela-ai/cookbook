#!/usr/bin/env bash
# =============================================================================
# download_otela.sh — stage the official OpenTela arm64 release on clariden
# -----------------------------------------------------------------------------
# Downloads the official release binary (github.com/eth-easl/OpenTela releases,
# asset `opentela-arm64`) to a VERSIONED path on /iopsstor so
# serve_glm_5_3_flash_otela_clariden.sbatch can pin it via OTELA_BIN.
#
# Run ONCE on a clariden login node (egress + /iopsstor), or from the repo
# root via the renew script: scripts/renew_glm_5_3_flash.sh --update-otela.
#
#   bash deployments/llm/clariden/glm-53-flash/download_otela.sh           # pinned default
#   OTELA_VERSION=latest bash deployments/llm/clariden/glm-53-flash/download_otela.sh
#   OTELA_VERSION=v0.2.4 bash deployments/llm/clariden/glm-53-flash/download_otela.sh
#
# Versioned filename ($OTELA_DIR/opentela-arm64-<tag>) keeps every prior
# release on disk — so rollback is simply the PREVIOUS official release
# (OTELA_VERSION=v0.2.5 re-stages it; submit with
# OTELA_BIN=$OTELA_DIR/opentela-arm64-v0.2.5). The legacy sai-v0.0.6 fork
# is retired and unreferenced.
#
# NOTE the asset name: releases >= v0.2.0 ship `opentela-arm64`;
# `otela-arm64` was the name last used by v0.1.12 (stale in some docs).
#
# After downloading, bump OTELA_BIN's default in the sbatch (or export
# OTELA_BIN) and re-run the renew script; the recipe preflight prints the
# exact binary path it resolved.
# =============================================================================
set -euo pipefail

OTELA_DIR="${OTELA_DIR:-/iopsstor/scratch/cscs/xyao/opentela}"
OTELA_VERSION="${OTELA_VERSION:-v0.2.6}"   # pinned default; "latest" resolves via the GitHub API
GITHUB_API="https://api.github.com/repos/eth-easl/OpenTela/releases"
FORCE="${FORCE:-0}"                        # FORCE=1 re-download over an existing file

tag="$OTELA_VERSION"
if [ "$tag" = "latest" ]; then
  echo "Resolving latest OpenTela release via the GitHub API ..."
  tag="$(curl -sf --max-time 30 "$GITHUB_API/latest" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')" \
    || { echo "FATAL: could not resolve the latest release tag" >&2; exit 1; }
  echo "Latest release: $tag"
fi

dest="$OTELA_DIR/opentela-arm64-$tag"
url="https://github.com/eth-easl/OpenTela/releases/download/$tag/opentela-arm64"

mkdir -p "$OTELA_DIR"

if [ -x "$dest" ] && [ "$FORCE" != "1" ]; then
  echo "Already present: $dest"
  "$dest" version 2>&1 | head -2 || true
  echo "Set FORCE=1 to re-download."
  exit 0
fi

echo "Downloading $url -> $dest ..."
curl -fL --retry 3 -o "$dest.part" "$url"
chmod +x "$dest.part"

# Sanity checks before renaming into place: must be an aarch64 executable
# that actually runs on this host (clariden login nodes are aarch64).
if command -v file >/dev/null 2>&1; then
  file "$dest.part" | grep -qE 'ELF 64-bit.*aarch64|ARM aarch64' \
    || { echo "FATAL: $dest.part is not an aarch64 ELF binary:" >&2; file "$dest.part" >&2; rm -f "$dest.part"; exit 1; }
fi
version_out="$("$dest.part" version 2>&1 | head -2)" \
  || { echo "FATAL: '$dest.part version' failed to run:" >&2; echo "$version_out" >&2; rm -f "$dest.part"; exit 1; }

mv "$dest.part" "$dest"
echo "Done: $dest"
echo "$version_out"
echo
echo "Next: ensure OTELA_BIN in serve_glm_5_3_flash_otela_clariden.sbatch points at"
echo "  $dest"
echo "(the default pins $OTELA_VERSION; override per-submit with OTELA_BIN=<path>)"
