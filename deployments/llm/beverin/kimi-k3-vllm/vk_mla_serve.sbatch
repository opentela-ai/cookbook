#!/bin/bash
# Standalone TP1 --load-format dummy serve on one gfx942 node, with the new
# VkernelMLA backend ACTIVE (VKERNELS_MLA=1 + VKERNELS_MLA_FORCE=1) and the
# CPU-oracle cross-check OFF (no libvkernels_c.so on the cluster).  Validates
# forward_mqa -> vk_hip_mla_fwd end-to-end during a real generation request.
#
# The live serving deployment ($K3=$DEPLOY_DIR, jobs 600279/600469) is NOT
# touched: k3_patch.py runs with K3=$ISOLATED, deploying the aiter overlay +
# the latest cookbook sitecustomize/vkernels_attn/vkernels_experts/.so into
# $ISOLATED/home/pylib (fresh mkdir).  engine.sh (unquoted heredoc -> all
# paths resolved at write time) puts $ISOLATED/home/pylib first on PYTHONPATH,
# so the container imports my new code regardless of the EDF env replacement.
#SBATCH --job-name=vk-mla-serve
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:25:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/capstor/scratch/cscs/xyao/vk-mla-serve/logs/%x-%j.out
set -uo pipefail

export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# kimi-k3-vllm.toml (EDF) lives in the cookbook recipe dir:
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
OTELA_EDF_NAME="${OTELA_EDF_NAME:-kimi-k3-vllm}"

ISOLATED="/capstor/scratch/cscs/xyao/vk-mla-serve"
VK="/capstor/scratch/cscs/xyao/vkernels"
MODEL_PATH="/capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K3"
SERVED_MODEL_NAME="moonshotai/Kimi-K3"
SERVE_PORT="${SERVE_PORT:-18080}"
MASTER_PORT="${MASTER_PORT:-29588}"
RUNDIR="$ISOLATED/run-$SLURM_JOB_ID"

echo "[$(date -Is)] job=$SLURM_JOB_ID isolated=$ISOLATED vk=$VK image=..."
mkdir -p "$ISOLATED/home/pylib" "$ISOLATED/home/.tmp" "$ISOLATED/logs" "$RUNDIR"

# --- rank 0: deploy aiter overlay + my latest sitecustomize/vkernels/.so
#     into the ISOLATED pylib (k3_patch.py reads $K3; live deploy untouched) ---
export K3="$ISOLATED"
export VKERNELS_DIR="$VK"
export K3_NO_ASYNC=1
export K3_SKIP_FLYDSL_PATCHES=1
rm -f "$ISOLATED/home/pylib/.k3_patch_done"
echo "[$(date -Is)] applying k3_patch.py INSIDE container (K3=$ISOLATED, fresh aiter overlay; live deploy untouched) ..."
# Run inside the container: the host (batch-shell) python3 is 3.6 and has
# no /usr/local/lib/python3.12/dist-packages/aiter (an in-container path).
# k3_patch.py's first step copytree's that SRC into $K3/home/pylib/aiter;
# on a FRESH isolated dir that needs the container's py3.12 + aiter tree.
# Text patches + copyfile of sitecustomize/vkernels/.so also run here, so my
# latest cookbook files land in $ISOLATED/home/pylib in one shot.
srun --ntasks=1 --environment="$OTELA_EDF_NAME" \
  bash -c 'export K3="'"$ISOLATED"'"; export VKERNELS_DIR="'"$VK"'"; export K3_NO_ASYNC=1; export K3_SKIP_FLYDSL_PATCHES=1; python3 "'"$SCRIPT_DIR/k3_patch.py"'" && touch "'"$ISOLATED/home/pylib/.k3_patch_done"'"'
rc=$?
echo "[$(date -Is)] k3_patch rc=$rc marker=$([ -f "$ISOLATED/home/pylib/.k3_patch_done" ] && echo yes || echo no)"
[ $rc -eq 0 ] && [ -f "$ISOLATED/home/pylib/.k3_patch_done" ] || {
  echo "FATAL: k3_patch.py failed; see above. Aborting (live deploy untouched)." >&2
  exit 1
}
ls -la "$ISOLATED/home/pylib" | head

# --- engine.sh: launched INSIDE the container via srun --environment.
#     Unquoted heredoc -> EVERY ${VAR} below is resolved by THIS batch shell
#     (where they are all known) and written as literals.  This defeats the
#     EDF's env-replacement: the values reach the container as fixed strings. ---
cat > "$RUNDIR/engine.sh" <<ENGINE
#!/bin/bash
set -uo pipefail
export PYTHONPATH="${ISOLATED}/home/pylib:\${PYTHONPATH:-}"
export TMPDIR="${ISOLATED}/home/.tmp"
export TMP="\$TMPDIR"
mkdir -p "\$TMPDIR" 2>/dev/null || true

# gfx942 env (mirrors serve_kimi_k3_otela_beverin.sbatch; the proven path)
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_MLA=0
export VLLM_ROCM_USE_SKINNY_GEMM=0
export K3_NO_ASYNC=1
export K3_SKIP_FLYDSL_PATCHES=1
export VKERNELS_DIR="${VK}"
export VKERNELS_LIB="${VK}/build/hip/src/c/libvkernels_hip.so"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/home/.cache/triton"
mkdir -p "\$TRITON_CACHE_DIR" 2>/dev/null || true
export OMP_NUM_THREADS=1
export MALLOC_TRIM_THRESHOLD_=65536
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_NTHREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# MLA integration (issue #42).  FORCE=1 actually runs forward_mqa
# (vk_hip_mla_fwd) instead of raising->falling-back; VALIDATE=0 skips the
# missing CPU-oracle cross-check; KDA stays OFF (issue #45).
export VKERNELS_MLA=1
export VKERNELS_MLA_FORCE=1
export VKERNELS_MLA_VALIDATE=0
export VKERNELS_KDA=0

echo "[engine] PYTHONPATH=\$PYTHONPATH"
echo "[engine] VKERNELS_MLA=\$VKERNELS_MLA FORCE=\$VKERNELS_MLA_FORCE VALIDATE=\$VKERNELS_MLA_VALIDATE KDA=\$VKERNELS_KDA"
echo "[engine] TRITON_CACHE_DIR=\$TRITON_CACHE_DIR TMPDIR=\$TMPDIR"

VLLM_ARGS=(
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --tensor-parallel-size 1
  --pipeline-parallel-size 1
  --nnodes 1 --node-rank 0
  --master-addr 127.0.0.1 --master-port ${MASTER_PORT}
  --max-model-len 4096
  --max-num-seqs 1
  --max-num-batched-tokens 4096
  --load-format dummy
  --enforce-eager
  --host 0.0.0.0 --port ${SERVE_PORT}
)

echo "[engine] launching vLLM (TP1 dummy) on port ${SERVE_PORT} ..."
python3 -m vllm.entrypoints.openai.api_server "\${VLLM_ARGS[@]}" > "$RUNDIR/engine.log" 2>&1 &
VLLM_PID=\$!
echo "[engine] vLLM pid=\$VLLM_PID; waiting for /health (up to 900s for dummy load + JIT) ..."

healthy=0
for i in \$(seq 1 180); do
  if ! kill -0 \$VLLM_PID 2>/dev/null; then
    echo "[engine] FATAL: vLLM exited early (after \$((i*5))s)"; echo "--- engine.log tail ---"; tail -60 "$RUNDIR/engine.log"; exit 1
  fi
  if python3 - <<'HEALTH' 2>/dev/null
import urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:${SERVE_PORT}/health", timeout=5); raise SystemExit(0)
except Exception:
    raise SystemExit(1)
HEALTH
  then
    healthy=1; echo "[engine] healthy after \$((i*5))s"; break
  fi
  sleep 5
done
if [ "\$healthy" != "1" ]; then
  echo "[engine] FATAL: /health timeout (900s); engine.log tail:"; echo "---"; tail -80 "$RUNDIR/engine.log"; exit 1
fi

echo "[engine] running GEN_CORRECTNESS_SMOKE probe (dummy weights -> tokens are garbage, pipeline must not fault) ..."
GEN_CORRECTNESS_SMOKE=1 GEN_CORRECTNESS_TIMEOUT=300 GEN_CORRECTNESS_PER_REQ_TIMEOUT=200 \
  python3 "${SCRIPT_DIR}/gen_correctness.py" \
    "http://127.0.0.1:${SERVE_PORT}" "${SERVED_MODEL_NAME}" "$RUNDIR/probe.json" 16 1
PRC=\$?
echo "[engine] probe rc=\$PRC (report in $RUNDIR/probe.json)"
echo "================ VkernelMLA / fault diagnostics from engine.log ================"
grep -aE "VkernelMLA|VKERNELS_MLA|sitecustomize.*MLA|vk_hip_mla_fwd|NotImplementedError|RuntimeError|Traceback|forward_mqa|nan=|NaN|fault|Illegal|SIGSEGV|SIGNAL|abort" "$RUNDIR/engine.log" | tail -50
echo "==============================================================================="
cat "$RUNDIR/probe.json" 2>/dev/null | head -60

kill -TERM \$VLLM_PID 2>/dev/null
wait \$VLLM_PID 2>/dev/null
echo "[engine] done, probe rc=\$PRC"
exit \$PRC
ENGINE
chmod +x "$RUNDIR/engine.sh"

echo "[$(date -Is)] launching engine inside container (EDF=$OTELA_EDF_NAME) ..."
srun --ntasks=1 --environment="$OTELA_EDF_NAME" bash "$RUNDIR/engine.sh"
SRC=$?
echo "[$(date -Is)] srun_exited code=$SRC"
exit "$SRC"
