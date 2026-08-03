#!/bin/bash
#SBATCH --job-name=qwen36-otela
#SBATCH --account=es_icp
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/cluster/scratch/%u/otela-qwen36/logs/qwen36-%j.out
#SBATCH --error=/cluster/scratch/%u/otela-qwen36/logs/qwen36-%j.out
#
# =============================================================================
# Serve Qwen/Qwen3.6-35B-A3B-FP8 (hybrid GDN MoE, FP8, Blackwell) on one
# Euler RTX PRO 6000 and register it on OpenTela via a login-node relay.
# Self-contained sbatch, no sibling scripts (the relay is started separately
# on a login node — see the README and start_relay_euler.sh).
#
# Submit:
#   sbatch serve_qwen36_otela_euler.sbatch
#   sbatch --export=ALL,SERVED_MODEL_NAME=Qwen/Qwen3.6-35B-A3B-FP8 \
#          serve_qwen36_otela_euler.sbatch
#
# What it does, in order:
#   1. Resolves the shared OpenTela relay on the login node (started by
#      start_relay_euler.sh; prefilled default below).
#   2. Starts sglang (apptainer --nv, single GPU) serving the FP8 model.
#   3. Waits for /health, then starts an otela WORKER on the same compute
#      node that registers the `llm` service + model id via the relay.
#   4. On job end / scancel, stops the worker so it announces LEFT cleanly.
#
# -----------------------------------------------------------------------------
# WHY A LOGIN-NODE RELAY (same topology as JSC)
# -----------------------------------------------------------------------------
# Euler compute nodes have NO outbound raw-TCP internet (api.opentela.ai:443,
# p2p.opentela.ai:443, github.com:443 all fail with "Cannot assign requested
# address"). The `eth_proxy` module gives HTTP(S) egress only — NOT the raw
# TCP that libp2p needs. The login node DOES have raw-TCP egress to the public
# internet (verified: bootstraps.opentela.ai:443, 140.238.223.116:8092 both
# OPEN). So the compute-side otela worker cannot reach the public head
# directly; it hops through a relay running on a login node:
#
#   client ─https─▶ api.opentela.ai ──P2P──▶ ocf head
#                                              ▲ WSS out via p2p.opentela.ai:443
#   ┌── Euler login node (e.g. eu-login-02, 129.132.93.77) ──────────────┐
#   │  RELAY  otela --role relay   (tmux; shared by all jobs/users)       │
#   │  listens on <login-ip>:18905 peer QmPAfjhRSrnR…M8n (seed 99)      │
#   └─────────────────────────────────────────────────────────────────────┘
#                          ▲ TCP, Euler-internal (compute CAN reach login)
#   ┌── compute node (this job, e.g. eu-g7-003) ─────────────────────────┐
#   │  WORKER  otela --role worker --service.port $SERVE_PORT             │
#   │  sglang serve (apptainer --nv, host netns) on 127.0.0.1:$SERVE_PORT │
#   └─────────────────────────────────────────────────────────────────────┘
#
# CRITICAL — otela BadgerDB (.ocfcore) must NOT go on /cluster/home (over
# quota, 46/48 GB at time of writing) NOR on Lustre scratch (stale file
# handles after days — documented in the JSC recipe). The worker's HOME is
# overridden to $TMPDIR/otela-home (node-local tmpfs/SSD, auto-cleaned by
# Slurm). With cleanslate: true the worker starts fresh each time, so the
# loss of BadgerDB on requeue is harmless.
#
# NO socat IS NEEDED (same as JSC): otela health-checks a hardcoded
# localhost:<port>. Because the worker runs ON the serving node (and
# apptainer shares the host network namespace), sglang genuinely is on
# 127.0.0.1:$SERVE_PORT.
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------- deployment --
# All runtime state lives under /cluster/scratch (home is over quota).
# /cluster/scratch/$USER/otela-qwen36 is pre-created by the operator before
# submitting (model download + image pull take a while; see the README).
export DEPLOY_DIR="${DEPLOY_DIR:-/cluster/scratch/$USER/otela-qwen36}"
export IMAGE="${IMAGE:-$DEPLOY_DIR/images/sglang_v0.5.16_cu129.sif}"
export MODEL="${MODEL:-$DEPLOY_DIR/model/Qwen3.6-35B-A3B-FP8}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.6-35B-A3B-FP8}"

# ------------------------------------------------------------------- otela ----
# The otela binary is OpenTela v0.2.3 (commit e9d1696, built 2026-07-29),
# pre-downloaded to scratch from
#   https://github.com/eth-easl/OpenTela/releases/download/v0.2.3/opentela-amd64
# The old ~/opentela/entry (sai-v0.0.6) is NOT used — v0.2.3 is required.
# To update: curl -L -o $DEPLOY_DIR/bin/otela <github-release-url>; chmod +x
OTELA_BIN="${OTELA_BIN:-$DEPLOY_DIR/bin/otela-v0.2.3}"
# The shared login-node relay (started by start_relay_euler.sh). An explicit
# $OTELA_RELAY_ADDR or a non-empty $RELAY_ADDR_FILE overrides this default.
# Verified: /ip4/129.132.93.77/tcp/18905/p2p/QmPAfjhRSrnRJn526V4QEHmoVASYMCAxCuVxgrQtmEM8n
# (eu-login-02, OpenTela v0.2.3, seed 99, 2026-08-03).
OTELA_RELAY_DEFAULT="/ip4/129.132.93.77/tcp/18905/p2p/QmPAfjhRSrnRJn526V4QEHmoVASYMCAxCuVxgrQtmEM8n"
RELAY_ADDR_FILE="${RELAY_ADDR_FILE:-$DEPLOY_DIR/relay.multiaddr}"
OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
# Labels attached to the service announcement (visible to the gateway as
# service.identity_group). The Euler otela binary has no --label flag, so
# labels are written to a cfg.yaml in the --config-dir (see below).
# Override with a space-separated list of "key=value" pairs.
OTELA_IDENTITY_GROUP="${OTELA_IDENTITY_GROUP:-model=$SERVED_MODEL_NAME cluster=euler gpu=rtx_pro_6000}"
# Fresh peer ID per invocation (avoids the LEFT-tombstone stall documented in
# the dgx-spark recipe: a graceful stop announces LEFT; restarting with the
# same seed reproduces the peer ID but a fresh CRDT cannot override the
# tombstone, so the peer never re-appears on the gateway for many minutes).
# Set OTELA_SEED to a constant ONLY if you want a stable peer ID.
OPENTELA_SEED="${OPENTELA_SEED:-$$}"
OTELA_TCP_PORT="${OTELA_TCP_PORT:-43905}"
OTELA_UDP_PORT="${OTELA_UDP_PORT:-59820}"

# ------------------------------------------------------------------- shape ----
# EXPORTED (not just assigned): engine.sh is written via a QUOTED heredoc
# (<<'ENGINE'), so $VAR references stay literal in the file and are
# resolved at engine.sh runtime from the inherited environment. Without
# `export`, engine.sh's `set -uo pipefail` fails with "$VAR: unbound".
export SERVE_PORT="${SERVE_PORT:-30000}"
# Hybrid GDN (Gated DeltaNet) on Blackwell rejects torch_native; triton is the
# only backend that accepts the hybrid linear+full attention schedule this
# model uses. (From the dgx-spark bring-up of the same model.)
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
# mem-fraction-static 0.85: leaves headroom for the ~29 GB Mamba/SSM + conv
# state that sglang does NOT count in its KV budget, plus CUDA graph capture,
# within the 96 GB RTX PRO 6000.
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
# Model emits <dynamic_thinking> reasoning tags; the qwen3 parser strips
# them into reasoning_content.
export REASONING_PARSER="${REASONING_PARSER:-qwen3}"
# How long to poll /health before giving up (seconds). FP8 weight load + SSM
# state init + CUDA graph capture on a single Blackwell GPU takes several
# minutes; 900 s is generous. Raise for cold loads.
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
# Passthroughs (default empty so set -u in heredoc children is happy).
export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
export OPENTELA_EXTRA_ARGS="${OPENTELA_EXTRA_ARGS:-}"

RUNDIR="$DEPLOY_DIR/run-$SLURM_JOB_ID"
mkdir -p "$RUNDIR" "$DEPLOY_DIR/logs"

echo "[$(date -Is)] job=$SLURM_JOB_ID node=$(hostname) gpus=$SLURM_JOB_GPUS"
echo "[$(date -Is)] model=$MODEL  served_as=$SERVED_MODEL_NAME"

# ----------------------------------------------------------- preflight --------
[ -f "$IMAGE" ] || { echo "FATAL: image not found: $IMAGE" >&2
  echo "       Build the SIF once via a compute job (login nodes OOM during" >&2
  echo "       mksquashfs — needs ~20 GB RAM). Compute nodes reach Docker Hub" >&2
  echo "       via eth_proxy. See README.md > Prerequisites for the exact sbatch." >&2
  exit 1; }
[ -d "$MODEL" ] || { echo "FATAL: weights not found: $MODEL" >&2
  echo "       Download the Qwen3.6-35B-A3B-FP8 shard tree (layers-N.safetensors," >&2
  echo "       config.json, chat_template.jinja, …) to $MODEL, or set MODEL." >&2
  exit 1; }
[ -x "$OTELA_BIN" ] || { echo "FATAL: otela binary not found/executable: $OTELA_BIN" >&2
  echo "       Obtain the OpenTela 'otela' binary and place it at $OTELA_BIN," >&2
  echo "       or set OTELA_BIN to an existing path." >&2
  exit 1; }

# ----------------------------------------------------------- resolve relay ----
RELAY_ADDR="${OTELA_RELAY_ADDR:-}"
[ -z "$RELAY_ADDR" ] && [ -s "$RELAY_ADDR_FILE" ] && RELAY_ADDR="$(cat "$RELAY_ADDR_FILE")"
[ -z "$RELAY_ADDR" ] && RELAY_ADDR="$OTELA_RELAY_DEFAULT"
echo "[$(date -Is)] relay=$RELAY_ADDR"

# ----------------------------------------------------------- last_service.env -
cat > "$DEPLOY_DIR/last_service.env" <<ENV
SERVICE_JOB_ID=$SLURM_JOB_ID
SERVICE_NODE=$(hostname)
SERVICE_PORT=$SERVE_PORT
SERVICE_URL=http://127.0.0.1:$SERVE_PORT
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
RELAY_ADDR=$RELAY_ADDR
ENV

# ============================================================== engine ========
cat > "$RUNDIR/engine.sh" <<'ENGINE'
#!/bin/bash
set -uo pipefail

# Caches off the quota-limited /cluster/home. Triton + tvm-ffi JIT-compile
# FP8 and attention kernels on first run; keep them on scratch for reuse.
export HF_HOME="${HF_HOME:-$DEPLOY_DIR/cache/hf}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DEPLOY_DIR/cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DEPLOY_DIR/cache/xdg}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ulimit stack 64 MB: sglang's hybrid-GDN call stack overflows the default
# 8 MB (documented in the dgx-spark bring-up). Apptainer inherits the host
# ulimit, so set it in the shell before `apptainer exec`.
ulimit -s 67108864

echo "[$(date -Is)] engine starting on $(hostname)"
exec apptainer exec --nv \
  --bind "$DEPLOY_DIR:$DEPLOY_DIR" \
  --env HF_HOME="$HF_HOME" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  --env HF_HUB_OFFLINE=1 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$SERVE_PORT" \
    --host 127.0.0.1 \
    --trust-remote-code \
    --attention-backend "$ATTENTION_BACKEND" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --reasoning-parser "$REASONING_PARSER" \
    $SGLANG_EXTRA_ARGS
ENGINE
chmod +x "$RUNDIR/engine.sh"

# ============================================================ otela worker ====
# All otela settings go into a cfg.yaml in the --config-dir (the Euler
# binary has no --label flag, so service.identity_group labels MUST go in
# the config file). BadgerDB goes to $HOME/.ocfcore — HOME is overridden to
# node-local $TMPDIR/otela-home to avoid both home quota and Lustre stale
# file handles. With --cleanslate true the worker starts fresh each time.
OTELA_HOME="$TMPDIR/otela-home"
mkdir -p "$OTELA_HOME" "$RUNDIR/otela"

cat > "$RUNDIR/otela/cfg.yaml" <<YAML
name: worker-$SLURM_JOB_ID
seed: "$OPENTELA_SEED"
tcpport: "$OTELA_TCP_PORT"
udpport: "$OTELA_UDP_PORT"
mode: node
role: worker
cleanslate: true
service:
  name: $OTELA_SERVICE_NAME
  port: "$SERVE_PORT"
  identity_group:
$(for _lbl in $OTELA_IDENTITY_GROUP; do echo "    - \"$_lbl\""; done)
solana:
  skip_verification: true
YAML

cat > "$RUNDIR/otela.sh" <<OTELA
#!/bin/bash
set -uo pipefail
# HOME override relocates BadgerDB to node-local storage (NOT home/Lustre).
# --config-dir picks up cfg.yaml (seed, identity_group, service name/port).
export HOME="$OTELA_HOME"
exec "$OTELA_BIN" start \\
  --config-dir "$RUNDIR/otela" \\
  --bootstrap.static "$RELAY_ADDR" \\
  $OPENTELA_EXTRA_ARGS
OTELA
chmod +x "$RUNDIR/otela.sh"

# ------------------------------------------------------------- lifecycle -----
# otela only leaves the registry cleanly on SIGINT/SIGTERM (AnnounceLeave). A
# SIGKILLed peer stays `connected: true` in the registry and the head
# round-robins traffic into a dead endpoint. Signal the child and wait for it
# to actually exit.
_OTELA_PID=""
_OTELA_STOPPED=0
stop_otela() {
  [ "$_OTELA_STOPPED" = 1 ] && return 0
  _OTELA_STOPPED=1
  if [ -n "$_OTELA_PID" ] && kill -0 "$_OTELA_PID" 2>/dev/null; then
    echo "[$(date -Is)] stopping otela worker (pid $_OTELA_PID)"
    kill -TERM "$_OTELA_PID" 2>/dev/null
    # Wait up to 20 s for graceful leave (AnnounceLeave -> CRDT drain ~10 s).
    # --signal=B:TERM@120 covers wall-time expiry; on scancel the only grace
    # is KillWait (default 30 s), so 20 s leaves margin before SIGKILL.
    for _ in $(seq 1 20); do
      kill -0 "$_OTELA_PID" 2>/dev/null || break
      sleep 1
    done
    grep -aqE "Announcing myself as LEFT|Leaving network" "$RUNDIR/otela.log" 2>/dev/null \
      && echo "[$(date -Is)] otela: graceful leave confirmed" \
      || echo "[$(date -Is)] WARN: otela did not announce LEFT" >&2
  fi
}
trap 'stop_otela' EXIT TERM INT

# Start the engine in the foreground (job lifetime == engine lifetime).
echo "[$(date -Is)] starting sglang engine"
bash "$RUNDIR/engine.sh" &
ENGINE_PID=$!

# Wait for /health, then start the otela worker (background, same node).
(
  deadline=$(( SECONDS + HEALTH_TIMEOUT ))
  until curl -sf --max-time 5 "http://127.0.0.1:$SERVE_PORT/health" >/dev/null 2>&1; do
    [ $SECONDS -ge $deadline ] && { echo "[$(date -Is)] health timeout; not registering" >&2; exit 1; }
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
      echo "[$(date -Is)] engine died before /health" >&2; exit 1
    fi
    sleep 10
  done
  echo "[$(date -Is)] engine healthy — starting otela worker"
  bash "$RUNDIR/otela.sh" >"$RUNDIR/otela.log" 2>&1 &
  _OTELA_PID=$!
  echo "[$(date -Is)] otela worker started (pid $_OTELA_PID)"
  # Wait for the worker; if it exits, the trap will handle cleanup.
  wait "$_OTELA_PID"
) &
REGISTER_PID=$!

# Wait for the engine. When it exits, the trap stops otela.
wait "$ENGINE_PID"
EXIT_CODE=$?

# Give otela a moment to shut down, then clean up the subshell.
stop_otela
kill "$REGISTER_PID" 2>/dev/null || true
wait "$REGISTER_PID" 2>/dev/null || true

echo "[$(date -Is)] job exiting with code $EXIT_CODE"
exit $EXIT_CODE
