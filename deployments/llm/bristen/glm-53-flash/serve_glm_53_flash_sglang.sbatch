#!/bin/bash
#SBATCH --job-name=glm53-bristen
#SBATCH --account=infra02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-53-flash-bristen/logs/%x-%j.out
#
# =============================================================================
# Serve zai-org/GLM-5.3-Flash (FP8 e4m3 ~306 GB) on CSCS Bristen (4x A100 /
# node, x86_64, NVIDIA) using the upstream CUDA SGLang image from Docker Hub.
#
# The Beverin overlay (built from the Clariden aarch64/CUDA image) supplies the
# GLM-5.3 Python source code (sglang.srt.models.glm5_next +
# transformers.models.glm5_next).  Its pure-Python files are prepended to
# PYTHONPATH over the dev-cu13 image.  The ROCm-specific pieces
# (overlay/pylib/sitecustomize.py, rocm_libs, vkernels) are intentionally NOT
# added, because they target MI300A gfx942.
#
# Memory note: GLM-5.3-Flash FP8 is ~306 GB.  On 4x A100 80 GB that is ~77 GB
# weights / GPU, leaving very little headroom.  Keep --context-length small for
# smoke tests (the recipe defaults to 2048) and --mem-fraction-static high.
#
# Usage:
#   sbatch serve_glm_53_flash_sglang.sbatch
#   LOAD_FORMAT=auto SMOKE=1 sbatch serve_glm_53_flash_sglang.sbatch   # real weights
#   SMOKE=0 sbatch serve_glm_53_flash_sglang.sbatch                    # + otela
# =============================================================================
set -uo pipefail

# ----------------------------------------------------- script + deploy dirs --
# Slurm may stage the batch script under /var/spool/slurmd/jobXXXX, so do not
# derive helper-script paths from BASH_SOURCE.  The deploy dir is the fixed
# location where preflight.py, engine.sh and gen_correctness.py live.
export DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-53-flash-bristen}"
export SCRIPT_DIR="$DEPLOY_DIR"

# GLM-5.3 Python overlay (reused from Beverin; pure Python only)
export OVL="${OVL:-/capstor/scratch/cscs/xyao/glm-53-flash-beverin/overlay}"

# Container image.  Default is the locally-cached enroot squashfs built once
# with "enroot import -o $ENROOT_CACHE_PATH/sglang-dev-cu13.sqsh dockerd://...".
# Override IMAGE=dockerhub... to pull fresh (slower; first pull can time out
# the preflight step).
export IMAGE="${IMAGE:-/capstor/scratch/cscs/xyao/glm-53-flash-bristen/cache/enroot/sglang-dev-cu13.sqsh}"

# GLM-5.3-Flash weights on shared /capstor
export MODEL_PATH="${MODEL_PATH:-/capstor/scratch/cscs/xyao/models/zai-org/GLM-5.3-Flash}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-5.3-Flash}"

# ------------------------------------------------------------------- shape ----
export NNODES="${NNODES:-$SLURM_JOB_NUM_NODES}"
export NNODES="${NNODES:-1}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-$TP_SIZE}"

# A100 80 GB is tight: high static fraction, small context for smoke.
export CTX_LEN="${CTX_LEN:-2048}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"

# Request-length bound + KV pool sizing.
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
export SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-1}"

# --- networking + ports -----------------------------------------------------
export SERVE_PORT="${SERVE_PORT:-30000}"
export MASTER_PORT="${MASTER_PORT:-30011}"

# --- watchdogs --------------------------------------------------------------
export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-7200}"
export DIST_TIMEOUT="${DIST_TIMEOUT:-1800}"
# Real FP8 weights (~306 GB from /capstor) can take 60-90 min to load on 4x A100.
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
export GEN_PROBE="${GEN_PROBE:-1}"
export GEN_PROBE_TIMEOUT="${GEN_PROBE_TIMEOUT:-600}"
export GEN_CORRECTNESS_PER_REQ_TIMEOUT="${GEN_CORRECTNESS_PER_REQ_TIMEOUT:-180}"
export GEN_CORRECTNESS_MAX_TOKENS="${GEN_CORRECTNESS_MAX_TOKENS:-64}"
export GEN_CORRECTNESS_MIN_PASS="${GEN_CORRECTNESS_MIN_PASS:-1}"
export GEN_CORRECTNESS_SMOKE="${GEN_CORRECTNESS_SMOKE:-1}"
export SMOKE="${SMOKE:-1}"

# --- OpenTela ---------------------------------------------------------------
export OTELA_DIR="${OTELA_DIR:-/capstor/scratch/cscs/xyao/opentela}"
export OTELA_BIN="${OTELA_BIN:-$OTELA_DIR/otela}"   # x86_64 build
export OTELA_RELAY_ADDR="${OTELA_RELAY_ADDR:-/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7}"
export OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
export OTELA_TCP_PORT="${OTELA_TCP_PORT:-43905}"
export OTELA_UDP_PORT="${OTELA_UDP_PORT:-59820}"
export OTELA_SEED="${OTELA_SEED:-$RANDOM$RANDOM}"
export OTELA_API_PORT="${OTELA_API_PORT:-18094}"

# --- load format ------------------------------------------------------------
# LOAD_FORMAT=dummy: random weights.  For GLM-5.3 dummy weights are allocated
# in bfloat16 (~612 GB) and OOM on 4x A100 80 GB, so real weights are the
# practical default for this model.
# LOAD_FORMAT=auto: real FP8 weights (~306 GB, 60-90 min load from /capstor).
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# ------------------------------------------------------------------- dirs ----
mkdir -p "$DEPLOY_DIR" "$DEPLOY_DIR/logs" "$DEPLOY_DIR/cache/triton" \
         "$DEPLOY_DIR/cache/hf" "$DEPLOY_DIR/cache/tmp" \
         "$DEPLOY_DIR/cache/enroot" "$DEPLOY_DIR/home" 2>/dev/null || true
export RUNDIR="$DEPLOY_DIR/run/${SLURM_JOB_ID:-$$}"
mkdir -p "$RUNDIR"
export HEAD="${SLURM_JOB_NODELIST:-$(hostname)}"
export HEAD="$(echo "$HEAD" | sed 's/\[.*//')"   # strip any node-array suffix -> first node
export HEAD_IP="$(getent hosts "$HEAD" 2>/dev/null | awk '{print $1; exit}')"
export HEAD_IP="${HEAD_IP:-127.0.0.1}"

# Cache / tmp inside the container (visible because /capstor is mounted)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DEPLOY_DIR/cache/triton}"
export HF_HOME="${HF_HOME:-$DEPLOY_DIR/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TMPDIR="${TMPDIR:-$DEPLOY_DIR/cache/tmp}"
export TMP="$TMPDIR"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$DEPLOY_DIR/cache/enroot}"
mkdir -p "$TRITON_CACHE_DIR" "$HF_HOME" "$TMPDIR" "$ENROOT_CACHE_PATH"

# PYTHONPATH: overlay source first, then a cp312-compatible tokenizers 0.23.1
# sidecar (extracted from the overlay's cp310 pkgs310 because tokenizers is
# abi3).  Do NOT put pkgs310 itself on PYTHONPATH -- it contains cp310-only
# compiled extensions (e.g. regex) that break on cp312.
export PYTHONPATH="${OVL}/sgl-workspace/sglang/python:${OVL}/sgl-workspace/transformers/src:${OVL}/site-extra-cp312:${PYTHONPATH:-}"

# Propagate PYTHONPATH into the container unchanged.  Note: pyxis's
# --container-env=PYTHONPATH="value" form mangles colon-separated paths, so we
# export the value on the host and pass the key without a value.
CONTAINER_OPTS=(
  --container-image="$IMAGE"
  --container-name=sglang-cu13
  --container-mounts=/capstor:/capstor
  --container-env=PYTHONPATH
)

echo "[$(date -Is)] started job=$SLURM_JOB_ID node=$HEAD image=$IMAGE load_format=$LOAD_FORMAT smoke=$SMOKE"
echo "[$(date -Is)] tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE ctx=$CTX_LEN mem=$GPU_MEM_UTIL port=$SERVE_PORT"
echo "[$(date -Is)] PYTHONPATH=$PYTHONPATH"

# ----------------------------------------------------------------- preflight --
# Quick in-container import test: the overlay must resolve glm5_next.
if [ "$LOAD_FORMAT" != "dummy" ]; then
  [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ] || {
    echo "FATAL: weights/config not found: $MODEL_PATH" >&2; exit 1; }
fi
[ -f "$OVL/sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py" ] || {
  echo "FATAL: GLM-5.3 sglang overlay not found under $OVL" >&2; exit 1; }
[ -f "$OVL/sgl-workspace/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py" ] || {
  echo "FATAL: GLM-5.3 transformers overlay not found under $OVL" >&2; exit 1; }
[ -f "$OVL/site-extra-cp312/tokenizers/__init__.py" ] || {
  echo "FATAL: cp312 tokenizers sidecar not found under $OVL/site-extra-cp312" >&2; exit 1; }

echo "[$(date -Is)] container preflight: import sglang + glm5_next ..."
srun -N1 -n1 --cpus-per-task=2 --gpus-per-task=0 -t0:10:00 \
     --export=ALL "${CONTAINER_OPTS[@]}" \
  python3 -u "$SCRIPT_DIR/preflight.py" || { echo "[$(date -Is)] PREFLIGHT FAILED" >&2; exit 1; }
echo "[$(date -Is)] preflight OK"

# ================================================================ engine ======
cp "$SCRIPT_DIR/engine.sh" "$RUNDIR/engine.sh"
chmod +x "$RUNDIR/engine.sh"

# --------------------------------------------------------- probe ------------
(
  deadline=$(( SECONDS + HEALTH_TIMEOUT ))
  until curl -sf --max-time 5 "http://127.0.0.1:$SERVE_PORT/health" >/dev/null 2>&1; do
    [ $SECONDS -ge $deadline ] && { echo "[$(date -Is)] health timeout after ${HEALTH_TIMEOUT}s" >&2; exit 1; }
    sleep 20
  done
  echo "[$(date -Is)] engine healthy after ~$((HEALTH_TIMEOUT - (deadline - SECONDS)))s"

  if [ "$GEN_PROBE" = "1" ]; then
    echo "[$(date -Is)] generation probe (model=$SERVED_MODEL_NAME, max_tokens=$GEN_CORRECTNESS_MAX_TOKENS, min_pass=$GEN_CORRECTNESS_MIN_PASS) ..."
    PROBE_OUT="$RUNDIR/gen_correctness_${SLURM_JOB_ID:-$$}.json"
    GEN_CORRECTNESS_TIMEOUT="$GEN_PROBE_TIMEOUT" \
    GEN_CORRECTNESS_PER_REQ_TIMEOUT="$GEN_CORRECTNESS_PER_REQ_TIMEOUT" \
    GEN_CORRECTNESS_SMOKE="$GEN_CORRECTNESS_SMOKE" \
      timeout "$((GEN_PROBE_TIMEOUT + 300))" \
      srun --jobid="${SLURM_JOB_ID:-0}" --overlap --gres=none --nodes=1 --ntasks=1 \
           -w "$HEAD" --export=ALL "${CONTAINER_OPTS[@]}" \
           python3 "$SCRIPT_DIR/gen_correctness.py" \
               "http://127.0.0.1:$SERVE_PORT" "$SERVED_MODEL_NAME" \
               "$PROBE_OUT" "$GEN_CORRECTNESS_MAX_TOKENS" "$GEN_CORRECTNESS_MIN_PASS"
    rc=$?
    if [ "$rc" = 0 ]; then
      echo "[$(date -Is)] GEN_PROBE_OK rc=$rc (report in $PROBE_OUT)"
    else
      echo "[$(date -Is)] GEN_PROBE_FAILED rc=$rc (report in $PROBE_OUT)" >&2
      exit 1
    fi
  fi

  if [ "$SMOKE" = "1" ]; then
    echo "[$(date -Is)] SMOKE=1: engine held for manual inspection. curl http://127.0.0.1:$SERVE_PORT/v1/models ; scancel $SLURM_JOB_ID when done."
  else
    # OpenTela worker runs on the head inside the same container / host netns.
    # It proxies the local SGLang HTTP endpoint onto the mesh.
    [ -x "$OTELA_BIN" ] || { echo "[$(date -Is)] FATAL: otela binary missing: $OTELA_BIN" >&2; exit 1; }
    echo "[$(date -Is)] starting otela worker on $HEAD for $SERVED_MODEL_NAME bootstrap=$OTELA_RELAY_ADDR"
    srun --jobid="${SLURM_JOB_ID:-0}" --overlap --gres=none --nodes=1 --ntasks=1 \
         -w "$HEAD" --export=ALL "${CONTAINER_OPTS[@]}" \
      "$OTELA_BIN" start --mode node --subprocess \
        --service.type llm --service.name "$OTELA_SERVICE_NAME" \
        --service.llm.endpoint "http://127.0.0.1:$SERVE_PORT" \
        --label "model=$SERVED_MODEL_NAME" \
        --bootstrap.static "$OTELA_RELAY_ADDR" \
        --tcp-port "$OTELA_TCP_PORT" --udp-port "$OTELA_UDP_PORT" \
        --api-port "$OTELA_API_PORT" --seed "$OTELA_SEED" \
        >"$RUNDIR/otela_${SLURM_JOB_ID:-$$}.log" 2>&1 &
    OTELA_PID=$!
    echo "[$(date -Is)] otela pid=$OTELA_PID log=$RUNDIR/otela_${SLURM_JOB_ID:-$$}.log"
  fi
) &

# ------------------------------------------------------------- run engine ----
NODE_CPUS="${SLURM_CPUS_ON_NODE:-$(scontrol show node "$HEAD" 2>/dev/null | awk -F= '/^CPUTot=/{print $2; exit}')}"
NODE_CPUS="${NODE_CPUS:-128}"
srun --nodes="$NNODES" --ntasks-per-node=1 \
     --cpus-per-task="$NODE_CPUS" --gpus-per-task=4 \
     --export=ALL "${CONTAINER_OPTS[@]}" bash "$RUNDIR/engine.sh"
SRUN_EXIT=$?
echo "[$(date -Is)] srun_exited code=${SRUN_EXIT}"
exit "$SRUN_EXIT"
