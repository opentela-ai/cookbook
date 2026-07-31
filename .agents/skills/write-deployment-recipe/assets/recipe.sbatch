#!/bin/bash
#SBATCH --job-name=<service>-otela
#SBATCH --account=<site-account>          # derive on-site: sshare -u $USER
#SBATCH --partition=<site-partition>      # derive on-site: sinfo; say WHY in README
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=<n>
#SBATCH --time=12:00:00
#SBATCH --output=<site-scratch>/%u/<service>/logs/<service>-%j.out
#
# =============================================================================
# Serve <service> on <site> (<arch>) and register it on OpenTela — one
# self-contained sbatch, no sibling scripts.
#
# Submit:
#   sbatch serve_<thing>.sbatch
#
# What it does, in order:
#   1. preflights prerequisites (image, weights, otela binary)
#   2. srun's the engine via apptainer --nv
#   3. waits for the head node's /health, then starts the otela WORKER
#   4. on job end / scancel, signals TERM to the worker so it announces LEFT
#
# -----------------------------------------------------------------------------
# SITE NETWORK TOPOLOGY (fill in — it decides whether you need a relay):
#   Can compute nodes reach p2p.opentela.ai:443 / the bootstrap peer directly?
#   Derive with:  curl -m 5 -sI https://api.opentela.ai  from a COMPUTE node.
#   - YES (e.g. Beverin)  -> worker dials the bootstrap directly; no relay.
#   - NO  (e.g. JSC)      -> us-managed relay on the login node; document its
#                            multiaddr, where its data dir lives (NOT on NFS —
#                            a stale handle freezes the CRDT -> 503 from the
#                            public API), and the >=v0.2.3 P2PForwardHandler
#                            requirement. See deployments/llm/jsc header.
# =============================================================================

set -uo pipefail

# ------------------------------------------------------------- deployment ----
# Every var: default = the documented site's value; overridable via env.
# Put caches and writable dirs on site SCRATCH, never on quota-limited home —
# quotas corrupt shared JIT caches and stale NFS handles freeze state stores.
DEPLOY_DIR="${DEPLOY_DIR:-<site-scratch>/$USER/<service>}"
IMAGE="${IMAGE:-$DEPLOY_DIR/images/<service>.sif}"   # pin by digest where possible
SERVE_PORT="${SERVE_PORT:-30000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-5400}"             # large weights take long; say why

# ----------------------------------------------------------------- otela -----
# Model the block below on the closest existing recipe:
#   direct egress site  -> deployments/llm/beverin-glm47-flash/serve_glm_47_flash_sglang.sbatch
#   relay site          -> deployments/llm/jsc/serve_llm_otela_jsc.sbatch
OTELA_BIN="${OTELA_BIN:-$DEPLOY_DIR/opentela-<arch>}"
OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
RELAY_ADDR="${OTELA_RELAY_ADDR:-<us-managed relay multiaddr, or bootstrap>}"

RUNDIR="$DEPLOY_DIR/run-$SLURM_JOB_ID"
mkdir -p "$DEPLOY_DIR/logs" "$RUNDIR"

HEAD="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
HEAD_IP="$(getent hosts "$HEAD" | awk '{print $1}' | head -1)"

# -------------------------------------------------------------- preflight ----
# Pattern: check, and on failure print the EXACT staging command for a node
# that HAS the missing capability (usually a login node with egress), then
# exit non-zero. Compute nodes here (may|may not) have outbound internet —
# verify and say which in the header.
[ -f "$IMAGE" ] || { echo "FATAL: image not found: $IMAGE" >&2
  echo "       build/pull it on a LOGIN node with: <exact command>" >&2; exit 1; }
[ -x "$OTELA_BIN" ] || { echo "FATAL: otela missing: $OTELA_BIN" >&2
  echo "       stage it on a LOGIN node with:" >&2
  echo "       curl -fL https://github.com/eth-easl/OpenTela/releases/download/<ver>/opentela-<arch> -o $OTELA_BIN && chmod +x $OTELA_BIN" >&2
  exit 1; }

cat > "$DEPLOY_DIR/last_service.env" <<ENV
SERVICE_JOB_ID=$SLURM_JOB_ID
SERVICE_HEAD_NODE=$HEAD
SERVICE_HEAD_IP=$HEAD_IP
SERVICE_PORT=$SERVE_PORT
ENV

# ----------------------------------------------------------------- engine ----
# Per-rank script heredoc'd into $RUNDIR so the sbatch stays self-contained.
# Re-derive per site: NCCL/GLOO socket interface (ip -o -4 addr), IPv4 forcing
# if AAAA records break bootstrap, IB retry/timeout if the fabric hiccups,
# and any accelerator-specific memory/affinity fixes (see beverin README).
cat > "$RUNDIR/engine.sh" <<'ENGINE'
#!/bin/bash
set -uo pipefail
echo "[rank ${SLURM_PROCID:-0}] $(hostname)"
exec apptainer exec --nv "$IMAGE" <serve-command> --host 0.0.0.0 --port "$SERVE_PORT"
ENGINE
chmod +x "$RUNDIR/engine.sh"

# ------------------------------------------------------- register & guard ----
# TERM (never KILL) the otela step on exit so it announces LEFT; otherwise the
# registry keeps the peer `connected: true` forever and round-robins real
# traffic into a dead endpoint.
trap '<stop-otela-function — see jsc recipe stop_otela()>' EXIT TERM INT

(
  deadline=$(( SECONDS + HEALTH_TIMEOUT ))
  until curl -sf --max-time 5 "http://$HEAD_IP:$SERVE_PORT/health" >/dev/null 2>&1; do
    [ $SECONDS -ge $deadline ] && { echo "health timeout; not registering" >&2; exit 1; }
    sleep 20
  done
  srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
       "$OTELA_BIN" start --bootstrap.static "$RELAY_ADDR" \
       --service.name "$OTELA_SERVICE_NAME" --service.port "$SERVE_PORT" \
       >"$RUNDIR/otela.log" 2>&1
) &

srun --ntasks-per-node=1 bash "$RUNDIR/engine.sh"
