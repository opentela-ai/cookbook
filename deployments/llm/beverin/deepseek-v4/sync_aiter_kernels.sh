#!/bin/bash
# Share the aiter JIT kernel cache through a Hugging Face bucket.
#
# WHY THIS EXISTS
#   DeepSeek-V4-Flash's routed experts are mxfp4, so on gfx942 sglang runs them
#   through aiter's native MXFP4 MoE. That kernel
#   (module_moe_ck2stages_b16_fp4x2_preshuffle_off_b16_silu_per_1x32_mulWeightStage2)
#   is Composable-Kernel C++ that aiter hipcc-compiles on the FIRST request —
#   ~20 hipcc + ~20 clang processes, with the other TP ranks blocked on aiter's
#   baton lock, and every CK template header read through the container's
#   squashfs FUSE mount. Measured on beverin: still compiling after 35 min.
#
#   serve_deepseek_v4_flash.sbatch already keeps that cache on /capstor
#   (AITER_JIT_DIR) so a given deploy dir pays it once. This script lifts it one
#   level further: push the compiled kernels to a shared bucket so a *fresh*
#   deploy dir, a different user, or a different cluster starts warm.
#
# USAGE
#   ./sync_aiter_kernels.sh upload   [opts]   # push locally-built kernels
#   ./sync_aiter_kernels.sh download [opts]   # restore into a JIT dir
#   ./sync_aiter_kernels.sh list     [opts]   # what is in the bucket
#
#   Run it on a LOGIN node: compute nodes have the JIT dir but the login node is
#   where `hf` and your HF credentials live. /capstor is visible from both.
#
# WHAT GETS UPLOADED
#   Only the DELTA — the kernels this site actually compiled — never the ~4.6 GB
#   the sbatch seeded out of the container image. Uploading the seeded tree would
#   redistribute the image's prebuilt binaries and waste the bucket.
set -euo pipefail

BUCKET="${BUCKET:-researchcomputer/kernels}"
JIT_DIR="${AITER_JIT_DIR:-/capstor/scratch/cscs/$USER/deepseek-v4-flash-sglang/cache/aiter-jit}"
ARCH="${ARCH:-}"
AITER_COMMIT="${AITER_COMMIT:-}"
IMAGE_DIGEST="${IMAGE_DIGEST:-}"
KEY_OVERRIDE=""
DRY_RUN=0
ASSUME_YES=0
SO_ONLY=0
UPLOAD_ALL=0
STAGE_DIR="${STAGE_DIR:-}"

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "==> $*"; }

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options (all subcommands):
  --bucket <ns/name>     bucket id            (default: researchcomputer/kernels)
  --jit-dir <path>       aiter JIT dir        (default: $AITER_JIT_DIR, else the
                                               beverin DSv4 deploy dir)
  --arch <gfx…>          GPU arch key         (default: from provenance file)
  --aiter-commit <sha>   aiter commit key     (default: from provenance file)
  --image-digest <sha>   image digest key     (default: from provenance file)
  --key <string>         use this exact remote key, skipping all of the above
  --stage-dir <path>     where to build the upload staging tree (default: the
                         JIT dir's parent — NEVER $TMPDIR, which is quota-limited
                         home on CSCS)
  --dry-run              print the plan, change nothing

upload only:
  --yes                  skip the confirmation prompt
  --so-only              upload just the built .so files, not their build trees
  --all                  upload the WHOLE jit dir, including the image-seeded
                         part (you almost never want this)
EOF
}

[ $# -ge 1 ] || { usage; exit 2; }
CMD="$1"; shift

while [ $# -gt 0 ]; do
  case "$1" in
    --bucket)        BUCKET="$2"; shift 2 ;;
    --jit-dir)       JIT_DIR="$2"; shift 2 ;;
    --arch)          ARCH="$2"; shift 2 ;;
    --aiter-commit)  AITER_COMMIT="$2"; shift 2 ;;
    --image-digest)  IMAGE_DIGEST="$2"; shift 2 ;;
    --key)           KEY_OVERRIDE="$2"; shift 2 ;;
    --stage-dir)     STAGE_DIR="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --yes|-y)        ASSUME_YES=1; shift ;;
    --so-only)       SO_ONLY=1; shift ;;
    --all)           UPLOAD_ALL=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *)               die "unknown option: $1 (try --help)" ;;
  esac
done

# --------------------------------------------------------------- preflight ---
command -v hf >/dev/null 2>&1 || die "the 'hf' CLI is not on PATH.
  Install it with:  pip install --upgrade 'huggingface_hub>=1.18'
  On beverin it lives in ~/.local/bin — add it:  export PATH=\$HOME/.local/bin:\$PATH"

# Buckets are a recent addition; an older hub has `hf` but no `hf buckets`.
hf buckets --help >/dev/null 2>&1 || die "this 'hf' has no 'buckets' command.
  Buckets need huggingface_hub >= 1.18. Upgrade with:  hf update"

if [ -z "${HF_TOKEN:-}" ] && ! hf auth whoami >/dev/null 2>&1; then
  die "not authenticated to Hugging Face.
  Either:  export HF_TOKEN=hf_...
  Or:      hf auth login
  The token needs WRITE access to the '${BUCKET%%/*}' namespace."
fi

# ------------------------------------------------------------- provenance ---
# Kernels are binaries built for one GPU arch by one aiter revision inside one
# image. Restoring a gfx942 build onto gfx950 would silently hand the engine
# wrong code, so the remote key pins all three. serve_deepseek_v4_flash.sbatch
# writes aiter-provenance.env next to the JIT dir; flags override it.
PROV_FILE="${JIT_DIR%/}/../aiter-provenance.env"
if [ -r "$PROV_FILE" ]; then
  note "reading provenance from $PROV_FILE"
  # shellcheck disable=SC1090  # path is computed at runtime
  . "$PROV_FILE"
  ARCH="${ARCH:-${PROV_ARCH:-}}"
  AITER_COMMIT="${AITER_COMMIT:-${PROV_AITER_COMMIT:-}}"
  IMAGE_DIGEST="${IMAGE_DIGEST:-${PROV_IMAGE_DIGEST:-}}"
  ROCM_VERSION="${PROV_ROCM_VERSION:-unknown}"
  SGLANG_VERSION="${PROV_SGLANG_VERSION:-unknown}"
  TORCH_VERSION="${PROV_TORCH_VERSION:-unknown}"
else
  ROCM_VERSION="unknown"; SGLANG_VERSION="unknown"; TORCH_VERSION="unknown"
fi

short() { printf '%s' "${1:0:12}"; }

if [ -n "$KEY_OVERRIDE" ]; then
  REMOTE_KEY="$KEY_OVERRIDE"
else
  [ -n "$ARCH" ] || die "GPU arch unknown — pass --arch (e.g. gfx942), or run the
  serve sbatch once so it writes $PROV_FILE."
  [ -n "$AITER_COMMIT" ] || die "aiter commit unknown — pass --aiter-commit, or run
  the serve sbatch once so it writes $PROV_FILE."
  [ -n "$IMAGE_DIGEST" ] || die "image digest unknown — pass --image-digest, or run
  the serve sbatch once so it writes $PROV_FILE."
  # Strip the ":sramecc+:xnack-" feature suffix torch reports; the target-feature
  # bits do not change kernel-object compatibility for our purposes, and leaving
  # them in would make the key unreadable in a URL.
  ARCH="${ARCH%%:*}"
  REMOTE_KEY="aiter/${ARCH}/$(short "${AITER_COMMIT#sha256:}")/$(short "${IMAGE_DIGEST#sha256:}")"
fi

DEST="hf://buckets/${BUCKET}/${REMOTE_KEY}"

# ------------------------------------------------------------------ list -----
if [ "$CMD" = "list" ]; then
  note "bucket ${BUCKET}"
  hf buckets info "$BUCKET" || die "cannot read bucket '$BUCKET' (does it exist? do you have access?)"
  echo
  note "contents"
  hf buckets ls "$BUCKET" --recursive 2>/dev/null || hf buckets ls "$BUCKET"
  exit 0
fi

# -------------------------------------------------------------- download -----
if [ "$CMD" = "download" ]; then
  note "restoring $DEST -> $JIT_DIR"
  mkdir -p "$JIT_DIR"
  # NOTE: no --delete. The local jit dir also holds the image-seeded modules,
  # and --delete would wipe every one of them that this key does not carry.
  if [ "$DRY_RUN" = 1 ]; then
    hf buckets sync "${DEST}/files" "$JIT_DIR" --dry-run
  else
    hf buckets sync "${DEST}/files" "$JIT_DIR"
    note "done — point the engine at it with AITER_JIT_DIR=$JIT_DIR"
  fi
  exit 0
fi

[ "$CMD" = "upload" ] || die "unknown command '$CMD' (expected upload|download|list)"

# ---------------------------------------------------------------- upload -----
[ -d "$JIT_DIR" ] || die "JIT dir not found: $JIT_DIR"

# Refuse while aiter is mid-build. aiter drops a lock_module_<name> next to the
# module it is compiling and removes it on success; uploading now would publish
# a half-written module and poison the cache for everyone who restores it.
if compgen -G "${JIT_DIR}/build/lock_module_*" >/dev/null 2>&1; then
  echo "ERROR: aiter is still building — refusing to upload a partial cache." >&2
  echo "In-progress module(s):" >&2
  for l in "${JIT_DIR}"/build/lock_module_*; do echo "  ${l##*/}" >&2; done
  echo "Wait for the engine to serve its first request, then re-run." >&2
  exit 1
fi

# Stage NEXT TO the JIT dir, not in $TMPDIR. On CSCS boxes TMPDIR points at
# /users/<u>/.tmp — the quota-limited home this whole recipe steers around — and
# a multi-GB kernel delta staged there either blows the quota or dies with
# "cp: preserving permissions: Operation not supported" on its ACLs. Staging on
# the same filesystem as the source also lets us hardlink instead of copy.
STAGE_ROOT="${STAGE_DIR:-$(dirname "${JIT_DIR%/}")}"
[ -d "$STAGE_ROOT" ] || die "stage root does not exist: $STAGE_ROOT (pass --stage-dir)"
STAGE="$(mktemp -d "${STAGE_ROOT}/.aiter-upload-stage.XXXXXX")" \
  || die "cannot create a staging dir under $STAGE_ROOT (pass --stage-dir)"
# shellcheck disable=SC2064  # expand STAGE now, not at trap time
trap "rm -rf -- '$STAGE'" EXIT

note "computing the delta under $JIT_DIR"
if [ "$UPLOAD_ALL" = 1 ]; then
  note "--all: taking the entire JIT dir (including the image-seeded part)"
  ( cd "$JIT_DIR" && find . -type f ! -name '.seeded' ! -path './__pycache__/*' -print0 ) > "$STAGE/.list0"
elif [ -r "${JIT_DIR}/.seeded.manifest" ]; then
  # Preferred: the sbatch recorded exactly what it copied out of the image, so
  # the delta is "everything not in that list" regardless of timestamps.
  note "using .seeded.manifest to subtract the image-seeded files"
  ( cd "$JIT_DIR" && find . -type f ! -name '.seeded' ! -name '.seeded.manifest' ! -path './__pycache__/*' -print ) \
    | LC_ALL=C sort > "$STAGE/.have"
  LC_ALL=C sort "${JIT_DIR}/.seeded.manifest" > "$STAGE/.seeded"
  LC_ALL=C comm -23 "$STAGE/.have" "$STAGE/.seeded" | tr '\n' '\0' > "$STAGE/.list0"
elif [ -e "${JIT_DIR}/.seeded" ]; then
  # Fallback for caches seeded before the manifest existed: the sbatch touches
  # .seeded *after* the copy, so anything newer was built here.
  note "no .seeded.manifest — falling back to files newer than .seeded"
  ( cd "$JIT_DIR" && find . -type f -newer .seeded ! -path './__pycache__/*' -print0 ) > "$STAGE/.list0"
else
  die "$JIT_DIR has neither .seeded.manifest nor .seeded — cannot tell locally
  built kernels from image-shipped ones. Re-run the serve sbatch (it seeds and
  marks the dir), or force a full upload with --all."
fi

# Drop build scratch. hipcc writes <obj>.o.tmp as it goes and renames on
# success, so a *.o.tmp is by definition either in-flight or abandoned by a
# killed build — never something a consumer should restore. Same for aiter's
# baton locks. (A cache seeded before .seeded.manifest existed falls back to
# mtime comparison, which happily picks up this debris.)
PRUNED=0
if [ "$UPLOAD_ALL" != 1 ]; then
  tr '\0' '\n' < "$STAGE/.list0" \
    | grep -vE '(\.tmp|\.lock)$|/lock_module_|/\.seeded' > "$STAGE/.list1" || true
  BEFORE=$(tr '\0' '\n' < "$STAGE/.list0" | grep -c . || true)
  AFTER=$(grep -c . < "$STAGE/.list1" || true)
  PRUNED=$((BEFORE - AFTER))
  [ "$PRUNED" -gt 0 ] && note "pruned ${PRUNED} transient build artefact(s) (*.tmp / lock_module_*)"
  tr '\n' '\0' < "$STAGE/.list1" > "$STAGE/.list0"
fi

if [ "$SO_ONLY" = 1 ]; then
  note "--so-only: keeping just the built .so files"
  tr '\0' '\n' < "$STAGE/.list0" | grep '\.so$' | tr '\n' '\0' > "$STAGE/.list0.tmp" || true
  mv "$STAGE/.list0.tmp" "$STAGE/.list0"
fi

FILE_COUNT=0
mkdir -p "$STAGE/files"
while IFS= read -r -d '' rel; do
  rel="${rel#./}"
  mkdir -p "$STAGE/files/$(dirname "$rel")"
  # Hardlink first: stage and source are on the same filesystem, so this is
  # instant and costs no extra space (the delta can be gigabytes). Fall back to
  # a plain copy across filesystems. NOT cp -a — preserving ownership/ACLs
  # fails outright on some CSCS mounts.
  ln "$JIT_DIR/$rel" "$STAGE/files/$rel" 2>/dev/null \
    || cp "$JIT_DIR/$rel" "$STAGE/files/$rel" \
    || die "could not stage $rel"
  FILE_COUNT=$((FILE_COUNT + 1))
done < "$STAGE/.list0"

[ "$FILE_COUNT" -gt 0 ] || die "delta is empty — nothing was compiled on top of the
  seeded cache, so there is nothing to share. (Has the engine served a request yet?)"

STAGE_BYTES=$(du -sb "$STAGE/files" | cut -f1)
STAGE_HUMAN=$(du -sh "$STAGE/files" | cut -f1)

# A manifest so a consumer can tell what these binaries are and where they came
# from without trusting the path alone.
cat > "$STAGE/manifest.json" <<EOF
{
  "kind": "aiter-jit-cache",
  "produced_by": "cookbook/deployments/llm/beverin-deepseek-v4/sync_aiter_kernels.sh",
  "remote_key": "${REMOTE_KEY}",
  "gpu_arch": "${ARCH:-unknown}",
  "aiter_commit": "${AITER_COMMIT:-unknown}",
  "image_digest": "${IMAGE_DIGEST:-unknown}",
  "rocm_version": "${ROCM_VERSION}",
  "sglang_version": "${SGLANG_VERSION}",
  "torch_version": "${TORCH_VERSION}",
  "file_count": ${FILE_COUNT},
  "total_bytes": ${STAGE_BYTES},
  "so_only": $( [ "$SO_ONLY" = 1 ] && echo true || echo false ),
  "full_tree": $( [ "$UPLOAD_ALL" = 1 ] && echo true || echo false ),
  "uploaded_by": "${USER:-unknown}@$(hostname -s 2>/dev/null || echo unknown)",
  "uploaded_at": "$(date --iso-8601=seconds)",
  "source_jit_dir": "${JIT_DIR}",
  "license_note": "Build artifacts of AMD aiter (MIT) compiled inside lmsysorg/sglang ROCm image; see upstream licenses."
}
EOF

echo
note "plan"
echo "  from    : $JIT_DIR"
echo "  to      : $DEST/files"
echo "  files   : $FILE_COUNT"
echo "  size    : $STAGE_HUMAN"
echo "  arch    : ${ARCH:-unknown}"
echo "  aiter   : ${AITER_COMMIT:-unknown}"
echo "  image   : ${IMAGE_DIGEST:-unknown}"
echo
echo "  sample:"
( cd "$STAGE/files" && find . -type f | head -8 | sed 's|^\./|    |' )
[ "$FILE_COUNT" -gt 8 ] && echo "    … and $((FILE_COUNT - 8)) more"
echo

if [ "$DRY_RUN" = 1 ]; then
  note "--dry-run: showing what hf would transfer, then stopping"
  hf buckets sync "$STAGE/files" "${DEST}/files" --dry-run || true
  echo
  note "manifest.json that would be written:"
  cat "$STAGE/manifest.json"
  exit 0
fi

# Uploading publishes these binaries to a bucket other people read. Make that an
# explicit choice rather than a side effect of running the script.
if [ "$ASSUME_YES" != 1 ]; then
  if [ ! -t 0 ]; then
    die "refusing to upload non-interactively without --yes"
  fi
  printf 'Upload %s (%s) to %s ? [y/N] ' "$FILE_COUNT files" "$STAGE_HUMAN" "$DEST"
  read -r reply
  case "$reply" in [yY]|[yY][eE][sS]) ;; *) note "aborted"; exit 1 ;; esac
fi

note "ensuring bucket ${BUCKET} exists"
hf buckets create "$BUCKET" --exist-ok

note "uploading kernels"
hf buckets sync "$STAGE/files" "${DEST}/files"

note "uploading manifest"
hf buckets cp "$STAGE/manifest.json" "${DEST}/manifest.json"

echo
note "done — https://huggingface.co/buckets/${BUCKET}"
note "restore elsewhere with:"
echo "    ./sync_aiter_kernels.sh download --key '${REMOTE_KEY}' --jit-dir <dir>"
