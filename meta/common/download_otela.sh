#!/usr/bin/env bash
# =============================================================================
# download_otela.sh — stage the official OpenTela release (eth-easl/OpenTela)
# -----------------------------------------------------------------------------
# Downloads the official release asset for the target architecture and repoints
# the `opentela-<arch>-latest` symlink at it, so every recipe that defaults
# OTELA_BIN to "$OTELA_DIR/opentela-<arch>-latest" always tracks the newest
# release. Run it wherever egress to github.com exists (login node, laptop):
# up.sh calls it before every submit.
#
#   bash meta/common/download_otela.sh                        # latest release
#   OTELA_VERSION=v0.2.6 bash meta/common/download_otela.sh   # pin a tag
#   FORCE=1 bash meta/common/download_otela.sh                # re-download
#   OTELA_DIR=/capstor/.../opentela OTELA_ARCH=arm64 bash ... # explicit target
#   DRY_RUN=1 bash meta/common/download_otela.sh              # resolve only
#
# Versioned files ($OTELA_DIR/opentela-<arch>-<tag>) are kept, so rollback is
# simply re-staging the previous tag (OTELA_VERSION=v0.2.5 ...) or pointing
# OTELA_BIN at the old versioned file directly.
#
# Asset notes: releases >= v0.2.0 ship `opentela-arm64` and `opentela-amd64`
# (`otela-arm64` was the v0.1.12-era name — retired, along with the sai fork).
# OTELA_ARCH overrides `uname -m` for cross-arch staging (e.g. staging an
# arm64 binary from an x86_64 login node); the exec smoke test is skipped for
# cross-arch staging since the binary cannot run on the staging host.
# =============================================================================
set -euo pipefail

OTELA_DIR="${OTELA_DIR:-$HOME/opentela-bin}"
OTELA_VERSION="${OTELA_VERSION:-latest}"   # "latest" resolves via the GitHub API
GITHUB_API="https://api.github.com/repos/eth-easl/OpenTela/releases"
FORCE="${FORCE:-0}"                        # FORCE=1 re-download over an existing file
DRY_RUN="${DRY_RUN:-0}"

_req_arch="${OTELA_ARCH:-$(uname -m)}"
case "$_req_arch" in
  x86_64)        ARCH=amd64 ;;
  aarch64|arm64) ARCH=arm64 ;;
  *) echo "FATAL: unsupported arch '$_req_arch' (set OTELA_ARCH=amd64|arm64)" >&2; exit 1 ;;
esac
_host_arch="$(uname -m)"
case "$_host_arch" in x86_64) HOST_ARCH=amd64 ;; aarch64|arm64) HOST_ARCH=arm64 ;; *) HOST_ARCH="$_host_arch" ;; esac

tag="$OTELA_VERSION"
if [ "$tag" = "latest" ]; then
  echo "Resolving latest OpenTela release via the GitHub API ..."
  tag="$(curl -sf --max-time 30 "$GITHUB_API/latest" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')" \
    || { echo "FATAL: could not resolve the latest release tag" >&2; exit 1; }
  echo "Latest release: $tag"
fi

dest="$OTELA_DIR/opentela-$ARCH-$tag"
link="$OTELA_DIR/opentela-$ARCH-latest"
url="https://github.com/eth-easl/OpenTela/releases/download/$tag/opentela-$ARCH"

mkdir -p "$OTELA_DIR"

if [ -x "$dest" ] && [ "$FORCE" != "1" ]; then
  echo "Already present: $dest"
else
  [ "$DRY_RUN" = "1" ] && { echo "DRY_RUN: would fetch $url -> $dest"; exit 0; }
  echo "Downloading $url -> $dest ..."
  curl -fL --retry 3 -o "$dest.part" "$url"
  chmod +x "$dest.part"

  # Sanity checks before renaming into place: correct-arch ELF that runs here.
  if command -v file >/dev/null 2>&1; then
    case "$ARCH" in
      arm64) file_pat='ELF 64-bit.*aarch64|ARM aarch64' ;;
      amd64) file_pat='ELF 64-bit.*(x86-64|AMD x86-64)' ;;
    esac
    file "$dest.part" | grep -qE "$file_pat" \
      || { echo "FATAL: $dest.part is not a $ARCH ELF binary:" >&2; file "$dest.part" >&2; rm -f "$dest.part"; exit 1; }
  fi
  if [ "$ARCH" = "$HOST_ARCH" ]; then
    version_out="$("$dest.part" version 2>&1 | head -2)" \
      || { echo "FATAL: '$dest.part version' failed to run:" >&2; echo "$version_out" >&2; rm -f "$dest.part"; exit 1; }
    echo "$version_out"
  else
    echo "NOTE: staged for $ARCH on a $HOST_ARCH host — exec smoke test skipped (OTELA_ARCH override)."
  fi

  mv "$dest.part" "$dest"
fi

# Repoint <arch>-latest (atomic: temp symlink + rename) so recipes pinning the
# symlink path pick up the new release on their next submit. Relative target
# keeps the symlink valid across differing mount paths of the same volume.
ln -sfn "opentela-$ARCH-$tag" "$OTELA_DIR/.opentela-$ARCH-latest.tmp.$$"
mv -Tf "$OTELA_DIR/.opentela-$ARCH-latest.tmp.$$" "$link"

echo "Staged: $link -> $dest"
echo "Recipes defaulting OTELA_BIN=\$OTELA_DIR/opentela-$ARCH-latest pick this up on the next submit."
echo "Rollback: OTELA_VERSION=<older-tag> bash meta/common/download_otela.sh  (or pin OTELA_BIN to the versioned file)"
