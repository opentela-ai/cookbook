#!/bin/bash
#SBATCH --job-name=llm-otela-jsc
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --output=/e/scratch/reformo/%u/otela-llm/logs/llm-otela-%j.out
#SBATCH --error=/e/scratch/reformo/%u/otela-llm/logs/llm-otela-%j.out
#
# =============================================================================
# Serve an LLM on JSC Jupiter Booster (GH200 / aarch64) and register it on
# OpenTela — one self-contained sbatch, no sibling scripts.
#
# Submit:
#   sbatch serve_llm_otela_jsc.sbatch                        # defaults: Kimi-K3, 8 nodes, TP4xPP8
#   sbatch --nodes=1 --export=ALL,NNODES=1,PP_SIZE=1,MODEL_PATH=/path/to/GLM,SERVED_MODEL_NAME=zai-org/GLM-4.7-Flash \
#          serve_llm_otela_jsc.sbatch
#
# What it does, in order:
#   1. resolves the shared OpenTela relay on the login node (us-managed; prefilled, see below)
#   2. srun's one engine process per node -> apptainer exec --nv sglang serve
#   3. waits for the head node's /health, then starts an otela WORKER on the
#      head node (srun --overlap) which registers the `llm` service
#   4. on job end / scancel, stops the worker so it announces LEFT cleanly
#
# -----------------------------------------------------------------------------
# THE SHARED LOGIN-NODE RELAY (us-managed; this script only dials it)
# -----------------------------------------------------------------------------
# JSC compute nodes have NO outbound internet — not even Cloudflare:443. The
# login node does. So a compute-side otela cannot reach the public head; it must
# hop through a relay running on the login node:
#
#   client ─https─▶ api.opentela.ai ──P2P──▶ ocf head
#                                              ▲ WSS out via p2p.opentela.ai:443
#   ┌── JSC login node (jpbl-s01-01, ib0 10.128.1.1) ───────────────────────┐
#   │  RELAY opentela --role relay   (tmux; shared by all jobs/users)       │
#   │  listens on 10.128.1.1:43917   peer QmXbXAnu9…DaZ1An                  │
#   └────────────────────────────────────────────────────────────────────────┘
#                          ▲ TCP, JSC-internal 10.x
#   ┌── compute head node (this job) ────────────────────────────────┐
#   │  WORKER opentela --role worker --service.port $SERVE_PORT      │
#   │  sglang serve (apptainer --nv, host netns) on localhost        │
#   └────────────────────────────────────────────────────────────────┘
#
# The relay is a long-lived, shared resource OPERATED BY US — started once in a
# tmux session on the login node and outliving any single job. An sbatch body
# runs on a COMPUTE node, so it could not start it anyway. This script therefore
# just dials the prefilled $OTELA_RELAY_ADDR (overridable via env or
# $RELAY_ADDR_FILE); it never starts a relay.
#
# The relay runs an OFFICIAL release (v0.2.3+). Its second hop depends on the
# P2PForwardHandler fix ("dial the target peer, not the incoming Host header";
# without it every hop self-dials — "failed to dial: dial to self attempted" —
# and the job registers but never serves a request). That fix landed in v0.2.3,
# so the stock release is now correct for the relay — no custom build needed.
#
# -----------------------------------------------------------------------------
# NO socat IS NEEDED
# -----------------------------------------------------------------------------
# otela's registrar health-checks a hardcoded localhost:<port>. Because the
# worker runs ON the serving node (and apptainer shares the host network
# namespace), sglang genuinely is on localhost:$SERVE_PORT. Forwarders are only
# needed when otela and the engine sit on different hosts.
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------- deployment --
PROJECT="${PROJECT:-reformo}"
export DEPLOY_DIR="${DEPLOY_DIR:-/e/scratch/$PROJECT/$USER/otela-llm}"
export IMAGE="${IMAGE:-/e/scratch/$PROJECT/$USER/kimi-k3/images/sglang-kimi-k3.sif}"
export MODEL_PATH="${MODEL_PATH:-/e/data1/datasets/products/mmlaion/shared/models/moonshotai/Kimi-K3}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3}"

# ------------------------------------------------------------------- shape ----
# TP stays intra-node (4x GH200 on NVLink); the pipeline spans nodes so captured
# decode collectives never cross the fabric. PP_SIZE MUST equal the node count.
export NNODES="${NNODES:-$SLURM_JOB_NUM_NODES}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-$NNODES}"
export EP_SIZE="${EP_SIZE:-$TP_SIZE}"      # ep_size * moe_dp_size == tp_size
export CG_DECODE="${CG_DECODE:-full}"
# Keep `marlin` for mxfp4 models: it is the only SM90 backend that leaves the
# weights PACKED. `auto` picks TRITON_KERNELS, whose upcast_from_mxfp()
# dequantizes to bf16 and OOMs a 4-bit model on 95 GiB HBM regardless of
# --mem-fraction-static.
export MOE_BACKEND="${MOE_BACKEND:-marlin}"
export CTX_LEN="${CTX_LEN:-65536}"
export MEM_FRAC="${MEM_FRAC:-0.90}"
export DIST_PORT="${DIST_PORT:-20000}"
export SERVE_PORT="${SERVE_PORT:-30000}"
export K3_NIC="${K3_NIC:-ib0}"             # confirmed present on booster nodes

# ------------------------------------------------------------------- otela ----
# Stock release is fine for the WORKER (it only receives hops). v0.2.3 also
# carries the relay's P2PForwardHandler fix, so the SAME release runs the
# us-managed login-node relay (see header) — no special build anymore.
OTELA_VERSION="${OTELA_VERSION:-v0.2.3}"
OTELA_BIN="${OTELA_BIN:-$DEPLOY_DIR/opentela-arm64}"
OTELA_URL="${OTELA_URL:-https://github.com/eth-easl/OpenTela/releases/download/$OTELA_VERSION/opentela-arm64}"
OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
OTELA_SEED="${OTELA_SEED:-21}"
OTELA_TCP_PORT="${OTELA_TCP_PORT:-43920}"
OTELA_UDP_PORT="${OTELA_UDP_PORT:-59840}"
OTELA_API_PORT="${OTELA_API_PORT:-18094}"
OTELA_STEP_NAME="${OTELA_STEP_NAME:-otela-worker-$SLURM_JOB_ID}"
# Peers DISCARD every CRDT record published by a binary without a build
# attestation (self-built binaries report version "dev"). The failure is silent
# and one-directional: the node connects, relays HTTP and serves traffic, but
# never registers and never announces LEFT. Set to true only if every node in
# the path runs an official signed release.
OTELA_REQUIRE_SIGNED="${OTELA_REQUIRE_SIGNED:-false}"
# The shared login-node relay (us-managed; see header). An explicit
# $OTELA_RELAY_ADDR or a non-empty $RELAY_ADDR_FILE overrides this default.
OTELA_RELAY_DEFAULT="/ip4/10.128.1.1/tcp/43917/p2p/QmXbXAnu9XVvfyXPaS57rzvKHbTjC4xMpxSdTpFQDaZ1An"
# Where the login-node relay may publish its dial address (optional override).
RELAY_ADDR_FILE="${RELAY_ADDR_FILE:-/e/scratch/$PROJECT/$USER/otela-relay/relay.multiaddr}"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-5400}"   # weights can be ~1.5 TB; be patient
RUNDIR="$DEPLOY_DIR/run-$SLURM_JOB_ID"
mkdir -p "$DEPLOY_DIR/logs" "$RUNDIR" "$DEPLOY_DIR/cache" "$DEPLOY_DIR/home"

HEAD="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export HEAD_IP="$(getent hosts "$HEAD" | awk '{print $1}' | head -1)"

echo "[$(date -Is)] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST"
echo "[$(date -Is)] head=$HEAD ($HEAD_IP)  tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE nnodes=$NNODES"
echo "[$(date -Is)] model=$MODEL_PATH  served_as=$SERVED_MODEL_NAME"

[ -f "$IMAGE" ] || { echo "FATAL: image not found: $IMAGE" >&2; echo "       build it on a login node with build_kimi_k3_image.sh (next to this sbatch)" >&2; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "FATAL: weights not found: $MODEL_PATH" >&2; exit 1; }
[ "$PP_SIZE" = "$NNODES" ] || echo "WARN: PP_SIZE=$PP_SIZE != NNODES=$NNODES (intended?)" >&2

# ------------------------------------------------------ resolve the relay -----
# Precedence: explicit $OTELA_RELAY_ADDR > non-empty $RELAY_ADDR_FILE > the
# built-in us-managed login-node relay ($OTELA_RELAY_DEFAULT). This script
# never starts the relay — see the header.
RELAY_ADDR="${OTELA_RELAY_ADDR:-}"
[ -z "$RELAY_ADDR" ] && [ -s "$RELAY_ADDR_FILE" ] && RELAY_ADDR="$(cat "$RELAY_ADDR_FILE")"
[ -z "$RELAY_ADDR" ] && RELAY_ADDR="$OTELA_RELAY_DEFAULT"
echo "[$(date -Is)] relay=$RELAY_ADDR"

if [ ! -x "$OTELA_BIN" ]; then
  echo "[$(date -Is)] fetching otela $OTELA_VERSION -> $OTELA_BIN"
  # This sbatch body runs on a COMPUTE node, which on JSC has NO outbound
  # internet — so the curl below is usually doomed here. Try it anyway with a
  # short timeout (clusters whose compute nodes do have egress self-serve), and
  # on failure print the EXACT login-node command: stage the binary once on a
  # login node and every job after reads it.
  if curl -fL --max-time 20 "$OTELA_URL" -o "$OTELA_BIN" && chmod +x "$OTELA_BIN"; then
    echo "[$(date -Is)] otela fetched"
  else
    rm -f "$OTELA_BIN"
    cat >&2 <<EOM
FATAL: otela binary missing at $OTELA_BIN and unreachable from here.
  JSC compute nodes have no outbound internet. Stage it ONCE on a LOGIN node:

    mkdir -p "$(dirname "$OTELA_BIN")"
    curl -fL "$OTELA_URL" -o "$OTELA_BIN" && chmod +x "$OTELA_BIN"

  then re-submit. (Override path/version with \$OTELA_BIN / \$OTELA_VERSION.)
EOM
    exit 1
  fi
fi

cat > "$DEPLOY_DIR/last_service.env" <<ENV
SERVICE_JOB_ID=$SLURM_JOB_ID
SERVICE_HEAD_NODE=$HEAD
SERVICE_HEAD_IP=$HEAD_IP
SERVICE_PORT=$SERVE_PORT
SERVICE_URL=http://$HEAD_IP:$SERVE_PORT
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
ENV

# ================================================================ engine ======
cat > "$RUNDIR/engine.sh" <<'ENGINE'
#!/bin/bash
set -uo pipefail
RANK="${SLURM_PROCID:-0}"
IFACE="${K3_NIC:-ib0}"
export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE"
SGLANG_HOST_IP="$(ip -o -4 addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
[ -z "$SGLANG_HOST_IP" ] && SGLANG_HOST_IP="$(hostname -I | awk '{print $1}')"
export SGLANG_HOST_IP
export NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}" NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Keep caches off the quota-limited /e/home; the deploy dir is bind-mounted.
export HOME="$DEPLOY_DIR/home" HF_HOME="$DEPLOY_DIR/cache/hf"
export XDG_CACHE_HOME="$DEPLOY_DIR/cache/xdg" TRITON_CACHE_DIR="$DEPLOY_DIR/cache/triton"
mkdir -p "$HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="" PYTHONNOUSERSITE=1

echo "[rank $RANK] $(hostname) ip=$SGLANG_HOST_IP iface=$IFACE head=$HEAD_IP:$DIST_PORT"

exec apptainer exec --nv \
  --bind /e/scratch:/e/scratch --bind /e/home:/e/home --bind /e/data1:/e/data1 \
  --pwd "$DEPLOY_DIR" "$IMAGE" \
  sglang serve \
    --trust-remote-code \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" --ep-size "$EP_SIZE" \
    --nnodes "$NNODES" --node-rank "$RANK" \
    --dist-init-addr "$HEAD_IP:$DIST_PORT" \
    --moe-runner-backend "$MOE_BACKEND" \
    --prefill-attention-backend flashmla --decode-attention-backend flashmla \
    --cuda-graph-backend-decode "$CG_DECODE" \
    --context-length "$CTX_LEN" --mem-fraction-static "$MEM_FRAC" \
    --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
    --enable-metrics --uvicorn-access-log-exclude-prefixes /metrics /health \
    --enable-hierarchical-cache --hicache-ratio 2.0 \
    --hicache-write-policy write_through_selective --hicache-io-backend kernel \
    --host 0.0.0.0 --port "$SERVE_PORT"
ENGINE
chmod +x "$RUNDIR/engine.sh"

# ============================================================ otela worker ====
mkdir -p "$RUNDIR/otela"
cat > "$RUNDIR/otela/cfg.yaml" <<CFG
name: jsc-worker-$SLURM_JOB_ID
seed: "$OTELA_SEED"
port: "$OTELA_API_PORT"
tcpport: "$OTELA_TCP_PORT"
udpport: "$OTELA_UDP_PORT"
mode: node
role: worker
loglevel: debug
cleanslate: false
# The worker genuinely has no inbound reachability, so AutoRelay stays on and
# maintains the circuit reservation on the relay. That reservation is what makes
# the head able to route to us at all — do NOT set this to public.
reachability: private
security:
  require_signed_binary: $OTELA_REQUIRE_SIGNED
solana:
  skip_verification: true
CFG

cat > "$RUNDIR/otela.sh" <<OTELA
#!/bin/bash
set -uo pipefail
exec "$OTELA_BIN" start --config-dir "$RUNDIR/otela" \\
  --bootstrap.static "$RELAY_ADDR" --solana.skip_verification \\
  --service.name "$OTELA_SERVICE_NAME" --service.port "$SERVE_PORT"
OTELA
chmod +x "$RUNDIR/otela.sh"

# ------------------------------------------------------------- lifecycle -----
# otela only leaves the registry cleanly on SIGINT/SIGTERM (AnnounceLeave). If it
# is SIGKILLed the head keeps a `connected: true` row forever, because the
# staleness sweep skips peers that registered a service — and then round-robins
# real traffic into a dead endpoint. Signal the STEP (srun does not reliably
# relay signals to the task) and wait for it to actually exit: "Leaving network"
# is logged at the START of shutdown, and the CRDT delta drains after that, so
# killing on the log line cuts the announcement off.
stop_otela() {
  local step
  step="$(squeue -s -j "$SLURM_JOB_ID" -h -o '%i %j' 2>/dev/null \
          | awk -v n="$OTELA_STEP_NAME" '$2==n{print $1}' | head -1)"
  [ -z "$step" ] && return 0
  echo "[$(date -Is)] stopping otela step $step"
  scancel -s TERM "$step" 2>/dev/null
  for _ in $(seq 1 45); do
    [ -z "$(squeue -s -j "$SLURM_JOB_ID" -h -o '%i %j' 2>/dev/null \
            | awk -v n="$OTELA_STEP_NAME" '$2==n{print $1}')" ] && break
    sleep 1
  done
  grep -aqE "Announcing myself as LEFT|Leaving network" "$RUNDIR/otela.log" 2>/dev/null \
    && echo "[$(date -Is)] otela: graceful leave confirmed" \
    || echo "[$(date -Is)] WARN: otela did not announce LEFT" >&2
}
trap 'stop_otela' EXIT TERM INT

# Register once the engine answers. Runs on the HEAD node, alongside sglang.
(
  deadline=$(( SECONDS + HEALTH_TIMEOUT ))
  until curl -sf --max-time 5 "http://$HEAD_IP:$SERVE_PORT/health" >/dev/null 2>&1; do
    [ $SECONDS -ge $deadline ] && { echo "[$(date -Is)] health timeout; not registering" >&2; exit 1; }
    sleep 20
  done
  echo "[$(date -Is)] engine healthy — starting otela worker on $HEAD"
  srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
       -J "$OTELA_STEP_NAME" bash "$RUNDIR/otela.sh" >"$RUNDIR/otela.log" 2>&1
) &

# ------------------------------------------------------------- run engine ----
# Pre-warm the tvm-ffi JIT cache single-process first. sglang JIT-builds kernels
# into ~/.cache/tvm-ffi (the REAL home — tvm_ffi ignores a redirected HOME). On a
# cold cache every rank races to build into that one shared directory and some
# load a half-written .so, surfacing as
#   AttributeError: Module has no function 'gptq_marlin_repack'
# during model-runner init. Building once here removes the race; no-op when warm.
echo "[$(date -Is)] pre-warming JIT cache"
apptainer exec --nv --bind /e/scratch:/e/scratch --bind /e/home:/e/home "$IMAGE" \
  python3 -c "
from sglang.jit_kernel.gptq_marlin_repack import _jit_gptq_marlin_repack_module as m
assert hasattr(m(), 'gptq_marlin_repack')
print('jit warm ok')
" 2>&1 | tail -2

srun --ntasks-per-node=1 bash "$RUNDIR/engine.sh"
