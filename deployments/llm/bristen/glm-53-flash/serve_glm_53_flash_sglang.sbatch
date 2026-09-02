#!/bin/bash
#SBATCH --job-name=glm53-bristen
#SBATCH --account=infra02
#SBATCH --partition=normal
# Default 2 nodes x TP4 x PP2 (8 GPUs) for a 12-hour-capable serve: FP8
# weights ~306 GB / 8 GPUs = ~38 GiB/GPU, leaving ~41 GiB headroom (vs the
# razor-edge ~2.4 GiB on 1 node). PP2 puts one pipeline stage per node, so
# only PP send/recv crosses the Slingshot fabric (validated, job 82749).
# Override with: sbatch --nodes=1 --export=ALL,PP_SIZE=1 ... (1-node smoke)
# or  NNODES=2 PP_SIZE=2 sbatch serve_glm_53_flash_sglang.sbatch
#SBATCH --nodes=2
# 2026-09-02 snapshot of bad-boot nodes (scontrol shows IDLE/gpu:4 but slurmd
# won't launch: "Something is wrong with the boot of the nodes"). Refresh with
# find_bad_nodes.sh and override via: sbatch --exclude=... serve_glm_53_flash_sglang.sbatch
#SBATCH --exclude=nid002284,nid002293,nid002297,nid002312,nid002316,nid002321,nid002324
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=12:00:00
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
# One pipeline stage per node (Clariden pattern). PP2 on 2 nodes halves the
# per-GPU weight to ~38 GiB (the path to a 12-hour serve on A100-80GB).
export PP_SIZE="${PP_SIZE:-$NNODES}"
export EP_SIZE="${EP_SIZE:-$TP_SIZE}"
# Slingshot high-speed NIC (one per GPU NUMA domain). hsn0 carries the
# routable 172.28/16 IPv4 used for the NCCL/GLOO TCPStore bootstrap; the
# aws_ofi_nccl plugin (engine.sh) carries the data path over CXI.
export GLM53_NIC="${GLM53_NIC:-hsn0}"

# A100 80GB is extremely tight at 1 node x TP4: FP8 weights ~306 GB / 4
# GPUs = ~77 GiB/GPU, leaving ~2.4 GiB free. mem_fraction_static=0.98 makes
# the activation reserve (slack = total * (1-mem)) ~1.6 GiB, below the ~2.25
# GiB FP8->bf16 MoE upcast transient (job 82473 OOM). Lowering mem to fit the
# transient is impossible at 1 node (KV+mamba = total*mem - weights - mamba
# goes negative: 80*0.95 - 77 - 0.4 < 0).
#
# DEFAULT 2 nodes x TP4 x PP2 (8 GPUs) FIXES THIS: weights ~306 GB / 8 =
# ~38 GiB/GPU, leaving ~41 GiB free. mem_fraction_static=0.93 gives slack
# ~5.5 GiB (> the 2.25 GiB upcast transient, ~2.4x margin) while KV+mamba
# stays ~35 GiB (vs going negative on 1 node). This is the path to a 12-hour
# serve on A100-80GB. The mamba state cache is no longer binding (no
# --max-mamba-cache-size needed); max_running_requests stays small as a
# conservative first-boot choice.
export CTX_LEN="${CTX_LEN:-2048}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.93}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"

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
# The first request on SM80 triggers a long TileLang compile; raise the
# detokenizer heartbeat timeout so the health check does not kill the server.
export SGLANG_HEALTH_CHECK_TIMEOUT="${SGLANG_HEALTH_CHECK_TIMEOUT:-300}"
# TileLang's data-race checker prints huge warnings on SM80; disable it.
export TL_DISABLE_DATA_RACE_CHECK="${TL_DISABLE_DATA_RACE_CHECK:-1}"
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

# SM80 FP8 compute patch: FP8 storage with bf16/fp16 compute inside Triton,
# because A100 has no native FP8 tensor cores and Triton's fp8e4nv dtype is
# Hopper-only.
export SGLANG_PATCH_DIR="${SGLANG_PATCH_DIR:-$DEPLOY_DIR/patches_full}"
# Always re-apply so edits to patched_sources/ are picked up on resubmits.
echo "[$(date -Is)] applying SM80 FP8 compute patches ..."
bash "$SCRIPT_DIR/apply_sm80_patch.sh" || { echo "[$(date -Is)] FATAL: SM80 patch failed" >&2; exit 1; }

export RUNDIR="$DEPLOY_DIR/run/${SLURM_JOB_ID:-$$}"
mkdir -p "$RUNDIR"
export HEAD="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" 2>/dev/null | head -1)"
export HEAD="${HEAD:-$(hostname)}"
export HEAD_IP="$(getent hosts "$HEAD" 2>/dev/null | awk '{print $1; exit}')"
export HEAD_IP="${HEAD_IP:-127.0.0.1}"
# HEAD/HEAD_IP must reach engine.sh INSIDE the container, where the
# --dist-init-addr "$HEAD_IP:$MASTER_PORT" (PP2 TCPStore bootstrap) is built.
# Without them engine.sh falls back to 127.0.0.1 and node 1 can never reach
# node 0's store (job 82802 hung for dist_timeout). MASTER_PORT is set above
# (line ~102); all three are listed explicitly in --container-env below for
# defense in depth.

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
export PYTHONPATH="${SGLANG_PATCH_DIR}:${OVL}/sgl-workspace/sglang/python:${OVL}/sgl-workspace/transformers/src:${OVL}/site-extra-cp312:${PYTHONPATH:-}"

# Propagate PYTHONPATH into the container unchanged.  Note: pyxis's
# --container-env=PYTHONPATH="value" form mangles colon-separated paths, so we
# export the value on the host and pass the key without a value.
CONTAINER_OPTS=(
  --container-image="$IMAGE"
  --container-name=sglang-cu13
  --container-mounts=/capstor:/capstor
  # NCCL vars are authored in engine.sh (:= defaults) and overridable from the
  # host via this list (pyxis inherits them into the container unmodified).
  # HEAD/HEAD_IP/MASTER_PORT are the PP2 dist-init-addr (engine.sh); without
  # them node 1's TCPStore bootstrap points at 127.0.0.1 and hangs.
  --container-env=PYTHONPATH,HEAD,HEAD_IP,MASTER_PORT,NCCL_NET_PLUGIN,NCCL_NET,NCCL_CROSS_NIC,FI_CXI_DISABLE_HOST_REGISTER,FI_CXI_DEFAULT_CQ_SIZE,FI_CXI_RDZV_THRESHOLD,FI_CXI_RDZV_GET_MIN,FI_MR_CACHE_MONITOR,NCCL_SOCKET_IFNAME,GLOO_SOCKET_IFNAME,NCCL_SOCKET_FAMILY,GLOO_SOCKET_FAMILY,NCCL_DEBUG,GLM53_NIC
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
