#!/bin/bash
#SBATCH --job-name=k3-image
#SBATCH --account=infra02
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k3/logs/image-%j.out
#SBATCH --error=/capstor/scratch/cscs/xyao/kimi-k3/logs/image-%j.out
#
# Import docker://lmsysorg/sglang:kimi-k3 (CUDA 13, aarch64, sm_90 cubins
# present) into a local .sqsh ONCE, so an 8-node serving job does not re-pull
# per node. Output path matches serve_kimi_k3_otela_clariden.sbatch's $IMAGE.
#
#   sbatch build_kimi_k3_image.sbatch
#
# Env overrides:
#   KIMI_K3_DIR=/capstor/scratch/cscs/xyao/kimi-k3   deploy root (= sbatch $DEPLOY_DIR)
#   IMAGE_SOURCE=docker://lmsysorg/sglang:kimi-k3    source image
#   IMAGE=$KIMI_K3_DIR/images/sglang-kimi-k3.aarch64.sqsh   output path
#
# Diskless GH200 compute node: /dev/shm is the only node-local store, so enroot
# scratch (data/cache/runtime/temp) and XDG_RUNTIME_DIR MUST be redirected there
# (clariden operator guidance), otherwise import dies on the read-only rootfs.
set -uo pipefail

PROJECT="${PROJECT:-infra02}"
export KIMI_K3_DIR="${KIMI_K3_DIR:-/capstor/scratch/cscs/$USER/kimi-k3}"
export IMAGE="${IMAGE:-$KIMI_K3_DIR/images/sglang-kimi-k3.aarch64.sqsh}"
SRC="${IMAGE_SOURCE:-docker://lmsysorg/sglang:kimi-k3}"
IMG_DIR="$(dirname "$IMAGE")"

SHM_ROOT=/dev/shm/$USER
export XDG_RUNTIME_DIR="$SHM_ROOT/run"
export ENROOT_DATA_PATH="$SHM_ROOT/enroot/data"
export ENROOT_CACHE_PATH="$SHM_ROOT/enroot/cache"
export ENROOT_RUNTIME_PATH="$SHM_ROOT/enroot/runtime"
export ENROOT_TEMP_PATH="$SHM_ROOT/enroot/temp"
# A stale user-bus socket on a diskless node breaks the container launch.
unset DBUS_SESSION_BUS_ADDRESS || true
mkdir -p "$XDG_RUNTIME_DIR" "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" \
         "$ENROOT_RUNTIME_PATH" "$ENROOT_TEMP_PATH" "$IMG_DIR" "$KIMI_K3_DIR/logs"

command -v enroot >/dev/null || { echo "FATAL: enroot not found (run on a clariden login/compute node)" >&2; exit 1; }

echo "[build $(date -Is)] src=$SRC"
echo "[build ] dst=$IMAGE"
echo "[build ] arch=$(uname -m)  /dev/shm capacity: $(df -h /dev/shm | tail -1 | awk '{print $4}')"

# Back up any prior import so a re-run never clobbers a known-good image.
[ -f "$IMAGE" ] && { echo "[build ] backing up existing $(basename "$IMAGE") -> .prev"; mv -f "$IMAGE" "$IMAGE.prev"; }

# Import straight from the Docker registry to a .sqsh. The image is multi-arch;
# on aarch64 this pulls the arm64 manifest.
enroot import -o "$IMAGE" "$SRC"
rc=$?

# enroot import can exit non-zero AFTER writing a usable .sqsh — verify by size.
if [ -f "$IMAGE" ]; then
  SZ=$(stat -c %s "$IMAGE")
  echo "[build ] wrote $IMAGE : $(numfmt --to=iec "$SZ")"
  if [ "$SZ" -lt $((5 * 1024 * 1024 * 1024)) ]; then
    echo "FAIL: .sqsh under 5 GiB — import truncated (rc=$rc)" >&2; exit 1
  fi
  echo "OK: image looks complete (rc=$rc)"
else
  echo "FAIL: no .sqsh written (rc=$rc)" >&2; exit 1
fi

# Drop the node-local enroot scratch (it dies with the node anyway).
rm -rf "$SHM_ROOT/enroot" || true

echo "[build $(date -Is)] done. Verify with:"
echo "  cd \"${SLURM_SUBMIT_DIR:-.}\" && EDF_PATH=\"$(pwd):${EDF_PATH:-$HOME/.edf}\" \\"
echo "    srun -p debug -A $PROJECT -n1 --gres=none -t0:05:00 --environment=kimi-k3-clariden sglang --version"
