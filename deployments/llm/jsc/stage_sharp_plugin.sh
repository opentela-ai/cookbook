#!/bin/bash
# Stage a self-contained JSC SHARP (NCCL switch-offload) plugin dir on
# /e/scratch, so it can be injected into the CUDA-12 sglang container on
# COMPUTE nodes. Run ONCE on a LOGIN node (needs /opt/mellanox + /lib64, both
# absent on compute nodes).
#
# WHY this exists: JSC's NCCL SHARP plugin (nccl_rdma_sharp) is what makes
# cross-node ALLGATHER stable on the Booster fabric — without it,
# ALLGATHER_BASE hangs after ~800-2000 ops (see the sbatch "WHY TP4/PP8"
# block). The plugin cannot be loaded naively; four gaps must be closed:
#   1. ABI: it links libcudart.so.12 — only the CUDA-12 sglang image provides
#      it (the CUDA-13 image does not). So build sglang-kimi-k3-cu12.sif first
#      (build_kimi_k3_image.sh with IMAGE_SOURCE=...:kimi-k3-cu12).
#   2. /opt/mellanox (SHARP runtime libs) is mounted on LOGIN nodes only, NOT
#      on compute nodes — so the whole dep chain is copied to /e/scratch here.
#   3. Two SONAME gaps: only libsmx-3.13.so / libsharprdmacm-3.13.so ship, but
#      the plugin asks for the -3.10 SONAMEs -> symlinks.
#   4. libmlx5 symbol gap: the cu12 container ships MLX5_1.24, but
#      libsharp_coll needs MLX5_1.25 -> inject the host /lib64 rdma-core stack
#      (an internally consistent version set).
# VERIFIED 2-node/8-GPU: plugin loads, NCCL logs "Loaded collnet plugin SHARP",
# "8 collnet channels" allocated, 3000x all_reduce + all_gather complete clean.
# TODO(unverified): full 8-node TP32/EP32 sglang run.
#
# Usage (login node):
#   bash deployments/llm/jsc/stage_sharp_plugin.sh
# Env overrides:
#   PROJECT=reformo                                 Slurm account / scratch project
#   DEPLOY_DIR=/e/scratch/$PROJECT/$USER/kimi-k3    deploy root (matches the sbatch)
#   SHARP_OUT=$DEPLOY_DIR/sharp-plugin-only         output dir (sbatch default)
set -euo pipefail

PROJECT="${PROJECT:-reformo}"
DEPLOY_DIR="${DEPLOY_DIR:-/e/scratch/$PROJECT/$USER/kimi-k3}"
OUT="${SHARP_OUT:-$DEPLOY_DIR/sharp-plugin-only}"

# JSC HPC-X CUDA-12 comm stack: NCCL 2.26 + the SHARP network plugin.
B=/e/software/fs/jupiter/stages/2025/software/NVHPC/25.5-CUDA-12/Linux_aarch64/25.5
PLUGIN=$B/comm_libs/12.9/hpcx/hpcx-2.22.1/nccl_rdma_sharp_plugin/lib

[ -f "$PLUGIN/libnccl-net.so" ] || { echo "FATAL: SHARP plugin not at $PLUGIN (NVHPC stage moved?)" >&2; exit 1; }
[ -d /opt/mellanox/sharp/lib ]  || { echo "FATAL: /opt/mellanox/sharp/lib missing (run on a LOGIN node)" >&2; exit 1; }

echo "[sharp] staging -> $OUT"
rm -rf "$OUT"; mkdir -p "$OUT"

# (1) the NCCL network plugin itself — dlopen'd by NCCL at communicator init.
cp -aL "$PLUGIN"/libnccl-net.so* "$OUT"/
# (2) SHARP runtime + helpers (login-only under /opt/mellanox).
cp -aL /opt/mellanox/sharp/lib/libsharp*.so* /opt/mellanox/sharp/lib/libalog.so* \
       /opt/mellanox/sharp/lib/libsmx*.so* /opt/mellanox/sharp/lib/libsharprdmacm*.so* "$OUT"/
# (3) SONAME gap: only the -3.13 SONAMEs ship; the plugin wants -3.10.
ln -sf libsmx-3.13.so         "$OUT"/libsmx-3.10.so
ln -sf libsharprdmacm-3.13.so "$OUT"/libsharprdmacm-3.10.so
# (4) libmlx5 symbol gap: inject host rdma-core so libsharp_coll finds MLX5_1.25.
for l in libmlx5.so.1 libibverbs.so.1 libibumad.so.3 libnl-3.so.200 libnl-route-3.so.200 librdmacm.so.1; do
  cp -aL "/lib64/$l" "$OUT"/
done

echo "[sharp] ldd check (expect NO 'not found'; libcudart resolves from the cu12 container at runtime):"
LD_LIBRARY_PATH="$OUT" ldd "$OUT"/libnccl-net.so | grep -iE 'not found|sharp|mlx5|cudart' || true
NF=$(LD_LIBRARY_PATH="$OUT" ldd "$OUT"/libnccl-net.so 2>/dev/null | grep -c 'not found')
if [ "$NF" -ne 0 ]; then
  echo "FAIL: $NF unresolved deps — re-check the NVHPC stage path and /opt/mellanox" >&2
  exit 1
fi
echo "[sharp] OK -> $OUT"
echo "[sharp] to enable TP32/EP32, submit with:"
echo "  sbatch --nodes=8 \\"
echo "    --export=ALL,IMAGE=$DEPLOY_DIR/images/sglang-kimi-k3-cu12.sif,TP_SIZE=32,EP_SIZE=32,PP_SIZE=1,SHARP_PLUGIN_DIR=$OUT \\"
echo "    serve_llm_otela_jsc.sbatch"
