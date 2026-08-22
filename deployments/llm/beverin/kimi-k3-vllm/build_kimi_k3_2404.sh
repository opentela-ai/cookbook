#!/bin/bash
# Build a glibc-2.39 rebase of vllm+vllm-openai-rocm+kimi-k3 for CXI on Beverin
# (issue #19). The existing image is Ubuntu 22.04 / glibc 2.35; the host Cray
# libfabric 1.29.1 + libcxi import GLIBC_2.38 *bound to libc.so.6* (readelf -V),
# so they never loaded in-container and RCCL silently fell back to Socket. A
# runtime LD_PRELOAD shim is provably dead for that (commit e8ec777).
#
# This rebase layers the ENTIRE bespoke 22.04 stack (custom ROCm 7.2.3 +
# vLLM fork + torch 2.11 + triton + kimi_k3 + deps) onto an ubuntu:24.04 base
# (glibc 2.39) WITHOUT recompiling anything: glibc is forward-compatible
# (2.35-built .so load on 2.39), and the kimi_k3 model + reasoning/tool parser
# + amd/ops are all pure Python (vllm/models/kimi_k3/*.py), so no HIP rebuild.
#
# NO DOCKER BUILD: podman-export ubuntu:24.04 to /capstor, cp the bespoke 22.04
# paths onto it, install the few system libs the minimal base lacks via an
# UNPRIVILEGED chroot (unshare -U -r — the login node blocks plain chroot but
# allows user namespaces), then mksquashfs into .edf_imagestore. All heavy work
# on /capstor (persistent, fast) — the 22.04 stack (27G) is extracted once.
#
# Run on a Beverin LOGIN node (has podman + internet + /opt/cray):
#   rcc -p beverin run -s 'bash deployments/llm/beverin/kimi-k3-vllm/build_kimi_k3_2404.sh'
set -uo pipefail

RECIPE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OLD_SQSH=/capstor/scratch/cscs/xyao/.edf_imagestore/vllm+vllm-openai-rocm+kimi-k3.x86_64.sqsh
BUILD=/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/2404-rebuild
SRC_ROOT="$BUILD/src-rootfs"          # the 22.04 bespoke stack (extracted once)
NEW="$BUILD/new-rootfs"               # the 2404 image under construction
NEW_SQSH=/capstor/scratch/cscs/xyao/.edf_imagestore/vllm+vllm-openai-rocm+kimi-k3-2404.x86_64.sqsh
mkdir -p "$BUILD" "$(dirname "$NEW_SQSH")" "$BUILD/logs"
LOG="$BUILD/logs/build-2404.out"
exec > >(tee -a "$LOG") 2>&1
# The login node forbids plain chroot (no CAP_SYS_CHROOT) but allows unprivileged
# user namespaces, so an `unshare -U -r chroot $NEW ...` is a real rootless chroot
# (proper absolute-path resolution, apt/dpkg/postinst all run in the image).
UCH="unshare -U -r chroot $NEW"
echo "================ build_kimi_k3_2404 $(date -Is) pid=$$ ================"
echo "RECIPE_DIR=$RECIPE_DIR  OLD_SQSH=$OLD_SQSH"
echo "SRC_ROOT=$SRC_ROOT  NEW=$NEW  NEW_SQSH=$NEW_SQSH"
command -v podman >/dev/null || { echo "FATAL: podman not on this node (run on a Beverin login node)"; exit 1; }
unshare -U -r sh -c true 2>/dev/null || { echo "FATAL: unprivileged userns blocked (unshare -U -r)"; exit 1; }

echo "[$(date -Is)] STEP 1: ensure 22.04 bespoke stack extracted to $SRC_ROOT"
if [ ! -d "$SRC_ROOT/usr/local/lib/python3.12/dist-packages/vllm" ]; then
  echo "  extracting $OLD_SQSH (~27G) ..."
  rm -rf "$SRC_ROOT"; mkdir -p "$SRC_ROOT"
  unsquashfs -f -d "$SRC_ROOT" "$OLD_SQSH" >/tmp/uq22 2>&1 || { tail -8 /tmp/uq22; exit 1; }
  echo "  extracted: $(du -sh "$SRC_ROOT/opt/rocm-7.2.3" "$SRC_ROOT/usr/local/lib/python3.12" 2>/dev/null)"
else
  echo "  already extracted, reusing"
fi

echo "[$(date -Is)] STEP 2: ensure ubuntu:24.04 base rootfs at $NEW"
if [ ! -d "$NEW/usr/bin" ]; then
  echo "  pulling + exporting ubuntu:24.04 ..."
  podman pull docker://docker.io/ubuntu:24.04 >/dev/null 2>&1 || true
  podman rm -f ub2404-build >/dev/null 2>&1 || true
  podman create --name ub2404-build docker.io/ubuntu:24.04 true >/dev/null
  podman export ub2404-build -o "$BUILD/ub2404.tar"
  podman rm ub2404-build >/dev/null
  mkdir -p "$NEW"; tar -C "$NEW" -xf "$BUILD/ub2404.tar"; rm -f "$BUILD/ub2404.tar"
else
  echo "  already present, reusing (rm -rf $NEW to rebuild base)"
fi
echo "  base glibc: $(strings "$NEW/lib/x86_64-linux-gnu/libc.so.6" 2>/dev/null | grep -m1 'GNU C Library')"

echo "[$(date -Is)] STEP 3: install system libs the minimal base lacks (unpriv chroot)"
# Bespoke ROCm/torch NEEDEDs: libnuma.so.1, libdrm.so.2, libdrm_amdgpu.so.1,
# librdmacm.so.1, libtbb.so.12, libunwind.so.1, libsqlite3.so.0, libzstd.so.1,
# libffi.so.8, libgomp.so.1, libatomic.so.1, libssl.so.3, libcrypto.so.3,
# libstdc++.so.6, libz.so.1. ubuntu:24.04 minimal ships libc6/libstdc++6/
# zlib1g/libgcc-s1; the 22.04 SRC_ROOT is MISSING libtbb.so.12 + libunwind.so.1,
# so apt (auto + transitive) is more reliable than hand-copying. 24.04's newer
# libs are forward-compatible with the 22.04-built .so (same SONAMEs).
cp /etc/resolv.conf "$NEW/etc/resolv.conf" 2>/dev/null || true
$UCH env TMPDIR=/tmp HOME=/root DEBIAN_FRONTEND=noninteractive apt-get -o APT::Sandbox::User=root update -qq 2>&1 | tail -3
$UCH env TMPDIR=/tmp HOME=/root DEBIAN_FRONTEND=noninteractive apt-get -o APT::Sandbox::User=root install -y --no-install-recommends \
    libnuma1 libdrm2 libdrm-amdgpu1 librdmacm1 libtbb12 libunwind8 \
    libsqlite3-0 libzstd1 libffi8 libgomp1 libatomic1 libssl3t64 \
    ca-certificates >/tmp/apt24.log 2>&1 || { echo "APT FAILED:"; tail -20 /tmp/apt24.log; rm -f "$NEW/etc/resolv.conf"; exit 1; }
rm -f "$NEW/etc/resolv.conf"; rm -rf "$NEW/var/lib/apt/lists/"*
echo "  system libs installed OK"

echo "[$(date -Is)] STEP 4: layer bespoke 22.04 stack onto the 24.04 base"
# ROCm 7.2.3 (custom RCCL 1.0.70203, MIOpen, hipblaslt, aiter) - ~20G.
# Replace any base /opt/rocm with a symlink to the real /opt/rocm-7.2.3 so
# /etc/environment's LD_LIBRARY_PATH=/opt/rocm/lib resolves to the custom libs.
rm -rf "$NEW/opt/rocm" "$NEW/opt/rocm-7.2.3"
cp -a "$SRC_ROOT/opt/rocm-7.2.3" "$NEW/opt/"
ln -s /opt/rocm-7.2.3 "$NEW/opt/rocm"
# Python 3.12 interpreter + stdlib (EXACT ABI for the bespoke .so extensions;
# the base has none, and 24.04's python3.12 is a different patch build).
mkdir -p "$NEW/usr/bin"
cp -a "$SRC_ROOT/usr/bin/python3.12" "$NEW/usr/bin/python3.12"
ln -sf python3.12 "$NEW/usr/bin/python3"      # vllm shebang is #!/usr/bin/python3
rm -rf "$NEW/usr/lib/python3.12"; mkdir -p "$NEW/usr/lib/python3.12"
cp -a "$SRC_ROOT/usr/lib/python3.12/." "$NEW/usr/lib/python3.12/"
# Bespoke site-packages: vLLM fork + torch 2.11 + triton + kimi_k3 + deps (~7G).
rm -rf "$NEW/usr/local/lib/python3.12"
mkdir -p "$NEW/usr/local/lib/python3.12"
cp -a "$SRC_ROOT/usr/local/lib/python3.12/." "$NEW/usr/local/lib/python3.12/"
# Bespoke CLIs (vllm, etc.) - merge over the base's (minimal) /usr/local/bin.
mkdir -p "$NEW/usr/local/bin"
cp -a "$SRC_ROOT/usr/local/bin/." "$NEW/usr/local/bin/" 2>/dev/null || true
# K3 runtime pins baked into /etc/environment by the original build.
cp -a "$SRC_ROOT/etc/environment" "$NEW/etc/environment"
echo "  layered: /opt/rocm->rocm-7.2.3, python3.12+stdlib, dist-packages($(du -sh "$NEW/usr/local/lib/python3.12" 2>/dev/null|cut -f1)), /etc/environment"

echo "[$(date -Is)] STEP 5: verify glibc >= 2.38 + python imports the bespoke stack"
echo "  glibc: $(strings "$NEW/lib/x86_64-linux-gnu/libc.so.6" 2>/dev/null | grep -m1 'GNU C Library')"
# /etc/environment is read by PAM, not the unpriv chroot; set LD_LIBRARY_PATH.
$UCH env LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib: \
    /usr/bin/python3.12 - <<'PYC' 2>&1 | tail -6 || echo "  (import check non-fatal)"
import torch, vllm, vllm.models.kimi_k3
print("IMPORT_OK torch", torch.__version__,
      "| vllm", getattr(__import__("vllm"), "__version__", "?"),
      "| kimi_k3 OK")
PYC
# The decisive check: does the host Cray libfabric now dlopen (glibc wall gone)?
# The EDF mounts /opt/cray/libfabric at RUNTIME; in this build chroot it is NOT
# mounted, so copy the host Cray libfabric + its full /usr/lib64 closure into the
# chroot temporarily and dlopen RTLD_NOW. G23_REBASE_DLOPEN_OK means the entire
# 1.29.1 stack loads on glibc 2.39 (the wall is gone); any other error names the
# next gap cheaply, here, before the mksquashfs + smoke.
if [ -f /opt/cray/libfabric/2.3.1/lib64/libfabric.so.1 ]; then
  echo "  Cray libfabric dlopen (RTLD_NOW, full /usr/lib64 closure) inside 2404 image:"
  mkdir -p "$NEW/tmp/ldtest"
  cp -aL /opt/cray/libfabric/2.3.1/lib64/libfabric.so.1 "$NEW/tmp/ldtest/"
  cp -aL /usr/lib64/lib*.so* "$NEW/tmp/ldtest/" 2>/dev/null || true
  $UCH env LD_LIBRARY_PATH=/tmp/ldtest \
      /usr/bin/python3.12 - <<'PYC2' 2>&1 | tail -5 || true
import ctypes
RTLD_NOW = 1
try:
    h = ctypes.CDLL("/tmp/ldtest/libfabric.so.1", mode=RTLD_NOW)
    print("G23_REBASE_DLOPEN_OK: Cray libfabric 1.29.1 loaded on glibc >= 2.39")
except Exception as e:
    msg = str(e)[:180]
    tag = "GWALL" if "GLIBC_2.38" in msg or "GLIBC_2.3" in msg else "GAP"
    print("G23_REBASE_DLOPEN_" + tag + ": " + msg)
PYC2
  rm -rf "$NEW/tmp/ldtest"
fi

echo "[$(date -Is)] STEP 6: mksquashfs the new image (zstd) -> $NEW_SQSH"
rm -f "$NEW_SQSH"
mksquashfs "$NEW" "$NEW_SQSH" -comp zstd -noappend >/tmp/sqfs.log 2>&1 || { echo "MKSQUASHFS FAILED:"; tail -12 /tmp/sqfs.log; exit 1; }
ls -lh "$NEW_SQSH"
echo "[$(date -Is)] DONE. New image: $NEW_SQSH"
echo "  Next: re-point kimi-k3-vllm.toml + ofi-rccl-smoke.toml 'image =' to this,"
echo "        push the vkernels rccl-net-ofi plugin + smoke, and re-run for CXI."
