#!/bin/bash
#SBATCH --job-name=llm-otela-jsc
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/e/scratch/reformo/%u/otela-llm/logs/llm-otela-%j.out
#SBATCH --error=/e/scratch/reformo/%u/otela-llm/logs/llm-otela-%j.out
#
# Kimi-K3 on JSC Jupiter Booster -> OpenTela. One self-contained sbatch:
# sglang engine (apptainer --nv, wrapped in servekit profile), a
# pre-registration verification bench, then an otela worker on the head node
# that registers the `llm` service and announces LEFT cleanly on job end.
# Defaults: Kimi-K3, 8 nodes, TP4xPP8 (~542 tok/s @ C=32).
#
#   sbatch serve_llm_otela_jsc.sbatch                 # production default
#   sbatch --nodes=1 --export=ALL,NNODES=1,PP_SIZE=1,MODEL_PATH=/path/to/GLM,SERVED_MODEL_NAME=zai-org/GLM-4.7-Flash \
#          serve_llm_otela_jsc.sbatch                 # small experiment
#   bash up.sh                                        # build-if-missing + relay + submit
#
# Every non-obvious setting below carries a one-line what-breaks-without-it and
# a pointer into README.md: §1 shape, §2 SHARP, §3 EP Blackwell-gated, §4
# JIT-cache races, §6 relay, §7 teardown, §8 engine-env guards.

set -uo pipefail

# ---------------------------------------------------------------- deployment --
PROJECT="${PROJECT:-reformo}"
export DEPLOY_DIR="${DEPLOY_DIR:-/e/scratch/$PROJECT/$USER/otela-llm}"
export IMAGE="${IMAGE:-/e/scratch/$PROJECT/$USER/kimi-k3/images/sglang-kimi-k3.sif}"
export MODEL_PATH="${MODEL_PATH:-/e/data1/datasets/products/mmlaion/shared/models/moonshotai/Kimi-K3}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3}"

# --------------------------------------------------------------------- shape --
# TP4 keeps collectives inside one node's NVLink domain; PP across the fabric is
# point-to-point. TP32/EP32 boots (SHARP, §2) but is 38x slower on Hopper (§3).
export NNODES="${NNODES:-$SLURM_JOB_NUM_NODES}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-$NNODES}"
export EP_SIZE="${EP_SIZE:-$TP_SIZE}"                       # ep_size * moe_dp_size == tp_size
export CG_DECODE="${CG_DECODE:-full}"
export MOE_BACKEND="${MOE_BACKEND:-marlin}"                 # only SM90 runner that keeps mxfp4 packed; 'auto' OOMs (§1)
export MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-}"               # empty=naive cross-node allgather; deepep crashes on sm90 (§3)
export CTX_LEN="${CTX_LEN:-1048576}"                        # bounds request length, NOT the KV pool
export MEM_FRAC="${MEM_FRAC:-0.90}"
export DIST_TIMEOUT="${DIST_TIMEOUT:-60}"                   # short on purpose: skips the doomed cross-node multimem rendezvous fast (§8)
export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"           # passthrough, e.g. --mamba-radix-cache-strategy extra_buffer_lazy
export SHARP_PLUGIN_DIR="${SHARP_PLUGIN_DIR:-}"             # empty=off; set + cu12 image to enable cross-node SHARP (§2)
# FlashKDA auto-on when the prefix exists: +14%/+19% agg @ C=8/16, flat at C=32 (§1).
FLASHKADA_PREFIX="${FLASHKADA_PREFIX:-$DEPLOY_DIR/flashkda_prefix}"
if [ -f "$FLASHKADA_PREFIX/flash_kda/__init__.py" ]; then
  [ -z "${SGLANG_PYTHONPATH_EXTRA:-}" ] && export SGLANG_PYTHONPATH_EXTRA="$FLASHKADA_PREFIX"
  case "$SGLANG_EXTRA_ARGS" in *linear-attn-prefill-backend*) ;; *)
    SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:+$SGLANG_EXTRA_ARGS }--linear-attn-prefill-backend flashkda"
    export SGLANG_EXTRA_ARGS ;;
  esac
fi
export DIST_PORT="${DIST_PORT:-20000}"
export SERVE_PORT="${SERVE_PORT:-30000}"
export K3_NIC="${K3_NIC:-ib0}"                              # confirmed present on booster nodes
# Metrics stay OFF: the NCCL watchdog re-registers Prometheus metrics on a
# collective timeout -> DuplicateTimeseries -> server crash (§8).
export ENABLE_METRICS="${ENABLE_METRICS:-}"

# --------------------------------------------------------------------- lib ---
# Shared helpers from the repo's meta/common/ — lib.sh (stop_otela),
# download_otela.sh; per-recipe copies were removed. Walk up from the submit
# dir (BASH_SOURCE is the spool copy — last resort only).
COMMON_DIR=""
for _root in "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
  [ -n "$_root" ] || continue
  _d="$_root"
  for _ in 1 2 3 4 5 6; do
    [ -f "$_d/meta/common/lib.sh" ] && { COMMON_DIR="$_d/meta/common"; break 2; }
    _d="$(dirname "$_d")"
  done
done
[ -n "$COMMON_DIR" ] || { echo "FATAL: meta/common/lib.sh not found above ${SLURM_SUBMIT_DIR:-<unset>} — submit from within the repo checkout" >&2; exit 1; }
source "$COMMON_DIR/lib.sh"   # stop_otela + teardown contract
unset _root _d
# --------------------------------------------------------------------- otela ---
# Official OpenTela release (arm64), staged on the login node by up.sh via
# meta/common/download_otela.sh (latest release; rollback = OTELA_VERSION pin).
# OTELA_VERSION/OTELA_URL below only back the emergency in-sbatch curl
# fallback (compute nodes have no egress — usually prints staging commands).
OTELA_VERSION="${OTELA_VERSION:-v0.2.4}"
OTELA_BIN="${OTELA_BIN:-$DEPLOY_DIR/opentela-arm64-latest}"
OTELA_URL="${OTELA_URL:-https://github.com/eth-easl/OpenTela/releases/download/$OTELA_VERSION/opentela-arm64}"
OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
OTELA_SEED="${OTELA_SEED:-21}"
OTELA_TCP_PORT="${OTELA_TCP_PORT:-43920}"
OTELA_UDP_PORT="${OTELA_UDP_PORT:-59840}"
OTELA_API_PORT="${OTELA_API_PORT:-18094}"
OTELA_STEP_NAME="${OTELA_STEP_NAME:-otela-worker-$SLURM_JOB_ID}"
OTELA_REQUIRE_SIGNED="${OTELA_REQUIRE_SIGNED:-false}"       # unsigned builds' CRDT records are silently discarded (§6)
# Relay precedence: $OTELA_RELAY_ADDR > non-empty $RELAY_ADDR_FILE > the
# built-in us-managed login-node relay. This script never starts the relay (§6).
OTELA_RELAY_DEFAULT="/ip4/10.128.1.1/tcp/43917/p2p/QmXbXAnu9XVvfyXPaS57rzvKHbTjC4xMpxSdTpFQDaZ1An"
RELAY_ADDR_FILE="${RELAY_ADDR_FILE:-/e/scratch/$PROJECT/$USER/otela-relay/relay.multiaddr}"

export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-5400}"             # weights can be ~1.5 TB; be patient
export RUNDIR="$DEPLOY_DIR/run-$SLURM_JOB_ID"

# ------------------------------------------------------------------ servekit --
# Cold-start profile + pre-registration bench. stdlib-only, runs in-container
# from a checkout on /e/scratch; missing checkout -> WARN, never a failed job.
export SERVEKIT_DIR="${SERVEKIT_DIR:-$DEPLOY_DIR/servekit}"
export SERVEKIT_BENCH="${SERVEKIT_BENCH:-1}"                # 0 skips the auto-bench
export SERVEKIT_BENCH_REQUESTS="${SERVEKIT_BENCH_REQUESTS:-64}"
export SERVEKIT_BENCH_CONCURRENCY="${SERVEKIT_BENCH_CONCURRENCY:-16}"
export SERVEKIT_BENCH_IN="${SERVEKIT_BENCH_IN:-768}"        # words (~1024 tokens)
export SERVEKIT_BENCH_OUT="${SERVEKIT_BENCH_OUT:-256}"      # tokens, ignore_eos
export SERVEKIT_BENCH_CORRECTNESS="${SERVEKIT_BENCH_CORRECTNESS:-1}"

mkdir -p "$DEPLOY_DIR/logs" "$RUNDIR" "$DEPLOY_DIR/cache" "$DEPLOY_DIR/home"

HEAD="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export HEAD_IP="$(getent hosts "$HEAD" | awk '{print $1}' | head -1)"

echo "[$(date -Is)] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST"
echo "[$(date -Is)] head=$HEAD ($HEAD_IP)  tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE nnodes=$NNODES"
echo "[$(date -Is)] model=$MODEL_PATH  served_as=$SERVED_MODEL_NAME"

[ -f "$IMAGE" ] || { echo "FATAL: image not found: $IMAGE" >&2; echo "       build it on a login node with build_kimi_k3_image.sh (next to this sbatch)" >&2; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "FATAL: weights not found: $MODEL_PATH" >&2; exit 1; }
[ "$PP_SIZE" = "$NNODES" ] || echo "WARN: PP_SIZE=$PP_SIZE != NNODES=$NNODES (intended?)" >&2
[ -d "$SERVEKIT_DIR/src/servekit" ] || {
  echo "[$(date -Is)] WARN: servekit checkout not found: $SERVEKIT_DIR — no cold-start profile, no auto-bench." >&2
  echo "[$(date -Is)]       Stage ONCE on a login node: git clone --depth=1 https://github.com/eth-easl/servekit $SERVEKIT_DIR" >&2
}

RELAY_ADDR="${OTELA_RELAY_ADDR:-}"
[ -z "$RELAY_ADDR" ] && [ -s "$RELAY_ADDR_FILE" ] && RELAY_ADDR="$(cat "$RELAY_ADDR_FILE")"
[ -z "$RELAY_ADDR" ] && RELAY_ADDR="$OTELA_RELAY_DEFAULT"
echo "[$(date -Is)] relay=$RELAY_ADDR"

# Compute nodes have no egress, so this fetch usually fails here; on failure
# print the exact login-node staging commands instead of failing obscurely.
if [ ! -x "$OTELA_BIN" ]; then
  echo "[$(date -Is)] fetching otela $OTELA_VERSION -> $OTELA_BIN"
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
# Force IPv4 for NCCL/Gloo: some racks resolve peers to IPv6, NCCL has no
# fallback -> communicator init deadlocks (5/32 comms without this; §8).
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-IPv4}" GLOO_SOCKET_FAMILY="${GLOO_SOCKET_FAMILY:-IPv4}"
# Ride out transient IB switch hiccups (defaults abort the RC QP too early; §8).
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}" NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"

# Caches on /e/scratch (quota-limited /e/home), JIT dirs RANK-LOCAL: a cold
# 32-rank build race -> "file too short" mid graph capture (§4).
export HOME="$DEPLOY_DIR/home" HF_HOME="$DEPLOY_DIR/cache/hf"
export XDG_CACHE_HOME="$DEPLOY_DIR/cache/xdg" TRITON_CACHE_DIR="$DEPLOY_DIR/cache/triton"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$DEPLOY_DIR/cache/tvm-ffi}/rank-${SLURM_PROCID:-0}"
export EP_JIT_CACHE_DIR="${EP_JIT_CACHE_DIR:-$DEPLOY_DIR/cache/deepep}/rank-${SLURM_PROCID:-0}"
mkdir -p "$HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TVM_FFI_CACHE_DIR" "$EP_JIT_CACHE_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SGLANG_PYTHONPATH_EXTRA:-}" PYTHONNOUSERSITE=1

# SHARP injection: --env, NOT srun --export, which apptainer silently discards (§2).
SHARP_ARGS=()
if [ -n "${SHARP_PLUGIN_DIR:-}" ] && [ -d "${SHARP_PLUGIN_DIR}" ]; then
  export LD_LIBRARY_PATH="$SHARP_PLUGIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  SHARP_ARGS=(--env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
  echo "[rank $RANK] SHARP plugin injected: $SHARP_PLUGIN_DIR"
fi

echo "[rank $RANK] $(hostname) ip=$SGLANG_HOST_IP iface=$IFACE head=$HEAD_IP:$DIST_PORT"

# servekit profile wraps the engine: log passes through unmodified, per-phase
# report at readiness. --timeout KILLS the engine on expiry: keep == HEALTH_TIMEOUT.
PROFILE=()
if [ -d "${SERVEKIT_DIR:-}/src/servekit" ]; then
  export PYTHONPATH="${SERVEKIT_DIR}/src:${PYTHONPATH:-}"
  PROFILE=(python3 -m servekit.cli profile --out "${RUNDIR}/coldstart.json"
           --timeout "${HEALTH_TIMEOUT:-5400}" --)
else
  echo "[rank $RANK] WARN: servekit not at ${SERVEKIT_DIR:-<unset>} — cold start unprofiled" >&2
fi

exec apptainer exec --nv "${SHARP_ARGS[@]}" \
  --bind /e/scratch:/e/scratch --bind /e/home:/e/home --bind /e/data1:/e/data1 \
  --pwd "$DEPLOY_DIR" "$IMAGE" \
  ${PROFILE[@]+"${PROFILE[@]}"} sglang serve \
    --trust-remote-code \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" --ep-size "$EP_SIZE" \
    --nnodes "$NNODES" --node-rank "$RANK" \
    --dist-init-addr "$HEAD_IP:$DIST_PORT" \
    --moe-runner-backend "$MOE_BACKEND" ${MOE_A2A_BACKEND:+--moe-a2a-backend "$MOE_A2A_BACKEND"} \
    --prefill-attention-backend flashmla --decode-attention-backend flashmla \
    --cuda-graph-backend-decode "$CG_DECODE" \
    --context-length "$CTX_LEN" --mem-fraction-static "$MEM_FRAC" \
    --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
    ${ENABLE_METRICS:+--enable-metrics} --uvicorn-access-log-exclude-prefixes /metrics /health \
    --enable-hierarchical-cache --hicache-ratio 2.0 \
    --hicache-write-policy write_through_selective --hicache-io-backend kernel \
    --host 0.0.0.0 --port "$SERVE_PORT" --dist-timeout "$DIST_TIMEOUT" $SGLANG_EXTRA_ARGS
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
# AutoRelay stays on: the circuit reservation on the relay is what makes the
# head able to route to us at all — do NOT set reachability: public here.
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

# ------------------------------------------------------------- teardown -------
# otela announces LEFT only on SIGTERM/INT; SIGKILL leaves a stale
# connected:true row that round-robins traffic into a dead endpoint (§7).
# The worker is an srun STEP because srun does not reliably relay signals to
# tasks — scancel the step instead. stop_otela comes from meta/common/lib.sh.
trap 'stop_otela' EXIT TERM INT

# ------------------------------------------------------------- registration ---
# Health-gated. The bench runs BEFORE registration so otela traffic cannot
# pollute the numbers; it merges into the cold-start report when present.
(
  deadline=$(( SECONDS + HEALTH_TIMEOUT ))
  until curl -sf --max-time 5 "http://$HEAD_IP:$SERVE_PORT/health" >/dev/null 2>&1; do
    [ $SECONDS -ge $deadline ] && { echo "[$(date -Is)] health timeout; not registering" >&2; exit 1; }
    sleep 20
  done
  if [ "${SERVEKIT_BENCH:-1}" = "1" ] && [ -d "$SERVEKIT_DIR/src/servekit" ]; then
    echo "[$(date -Is)] engine healthy — servekit bench C=$SERVEKIT_BENCH_CONCURRENCY n=$SERVEKIT_BENCH_REQUESTS (SERVEKIT_BENCH=0 to skip)"
    BENCH_ARGS=(--url "http://127.0.0.1:$SERVE_PORT"
                --requests "$SERVEKIT_BENCH_REQUESTS" --concurrency "$SERVEKIT_BENCH_CONCURRENCY"
                --input-len "$SERVEKIT_BENCH_IN" --output-len "$SERVEKIT_BENCH_OUT")
    [ "${SERVEKIT_BENCH_CORRECTNESS:-1}" = "1" ] || BENCH_ARGS+=(--no-correctness)
    if [ -f "$RUNDIR/coldstart.node0.json" ]; then
      BENCH_ARGS+=(--into "$RUNDIR/coldstart.node0.json")
    else
      BENCH_ARGS+=(--out "$RUNDIR/bench.json")
    fi
    srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
         apptainer exec --bind /e/scratch:/e/scratch --bind /e/data1:/e/data1 "$IMAGE" \
         env "PYTHONPATH=$SERVEKIT_DIR/src" python3 -m servekit.cli bench "${BENCH_ARGS[@]}" \
      || echo "[$(date -Is)] WARN: servekit bench failed; registering anyway" >&2
  fi
  echo "[$(date -Is)] engine healthy — starting otela worker on $HEAD"
  srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
       -J "$OTELA_STEP_NAME" bash "$RUNDIR/otela.sh" >"$RUNDIR/otela.log" 2>&1
  _OTELA_LAUNCHED=1   # stop_otela (lib.sh) is active only after a real launch
) &

# ------------------------------------------------------------- JIT pre-warm ---
# Must match engine.sh's rank-0 dir exactly, or the warm-up builds into a
# different dir than the ranks read and is a silent no-op (§4).
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$DEPLOY_DIR/cache/tvm-ffi}/rank-0"
mkdir -p "$TVM_FFI_CACHE_DIR"
# Single-process warm of every kernel 32 ranks would otherwise race to build
# (marlin repack, tiny_gemm TP widths, KDA decode). Warm cache = instant no-op.
echo "[$(date -Is)] pre-warming JIT cache"
apptainer exec --nv --bind /e/scratch:/e/scratch --bind /e/home:/e/home "$IMAGE" \
  python3 -c "
import torch
from sglang.jit_kernel.gptq_marlin_repack import _jit_gptq_marlin_repack_module as m
assert hasattr(m(), 'gptq_marlin_repack')
from sglang.kernels.ops.kimi_k3 import kimi_k3_tiny_gemm
for n, k in [(144, 7168), (896, 7168), (1536, 128)]:
    x = torch.zeros(min(n, 8), k, dtype=torch.bfloat16, device='cuda')
    w = torch.zeros(n, k, dtype=torch.bfloat16, device='cuda')
    kimi_k3_tiny_gemm(x, w)
    torch.cuda.synchronize()
# Kimi-K3's linear-attention decode kernel; JIT-built during graph capture.
try:
    from sglang.kernels.ops.attention.kda_packed_decode import _jit_kda_packed_decode_module
    _jit_kda_packed_decode_module()
except Exception as e:
    print('kda warm skipped:', repr(e)[:120])
print('jit warm ok')
" 2>&1 | tail -3

srun --ntasks-per-node=1 bash "$RUNDIR/engine.sh"
