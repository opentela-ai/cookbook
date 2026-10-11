#!/bin/bash
#SBATCH --job-name=k3-otela-clariden
#SBATCH --account=infra02
#SBATCH --partition=normal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
# Whole node (288 logical CPUs): the image pins TP ranks to GPU NUMA domains
# via host CPU ids; a shrunken cpuset raises "CPU number N is not eligible".
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
# No --mem/--exclusive: normal partition is OverSubscribe=EXCLUSIVE (whole
# nodes via 288 cpus + 4 gpus); /dev/shm is a fixed 334 GiB tmpfs — enough
# for servekit's ~190 GiB/rank presharded staging. (Requesting 850 GiB was
# too close to 870 GiB RealMemory for reliable placement.)
# B:TERM@120 signals the BATCH SHELL so the stop_otela trap can TERM the otela
# step before KillWait — without B: the trap never runs, no LEFT is published,
# and the head keeps round-robining into a dead endpoint.
#SBATCH --signal=B:TERM@120
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k3/logs/k3-clariden-%j.out
#SBATCH --error=/capstor/scratch/cscs/xyao/kimi-k3/logs/k3-clariden-%j.out
#
# =============================================================================
# Serve Kimi-K3 (2.8 T-param hybrid MoE, mxfp4) on CSCS Clariden (4x GH200 /
# node, aarch64, Slingshot fabric, NO InfiniBand) and register it on OpenTela —
# one self-contained sbatch, no sibling scripts.
#
# Submit:
#   sbatch serve_kimi_k3_otela_clariden.sbatch                      # defaults: 8 nodes, TP4 x PP8
#   sbatch --nodes=4 --export=ALL,NNODES=4,PP_SIZE=4 \
#          serve_kimi_k3_otela_clariden.sbatch                       # 4 nodes, TP4 x PP4
#
# Flow: preflight (image/weights/otela, exact staging commands on failure) →
# one sglang rank/node in the kimi-k3-clariden container (TP4 x PP<NNODES>,
# only the head serves HTTP) → servekit verification bench PRE-registration
# (mesh traffic must not pollute; SERVEKIT_BENCH=0 skips) → ONE otela worker
# on the head → SIGTERM on end (LEFT).
#
# TOPOLOGY: ~1.5 TB mxfp4 does not fit one node — the JSC pattern (ONE
# distributed engine, ONE otela worker on the head), not Beverin's per-node
# replicas. TP4 keeps allreduce on NVLink; PP<NNODES> means only pipeline
# send/recv crosses Slingshot (flat TP32: ~1-5 tok/s, no CUDA-graph capture —
# README fix 2). Verified job 3000965: C=1 34.7 / C=16 374.9 / C=32 561.4 /
# C=64 559.3 tok/s; cold start ~105 min — README "Verified benchmarks".
# NO RELAY: compute reaches the public bootstrap directly (unlike JSC).
# CONTAINER + NCCL: EDF hooks aws_ofi_nccl (netstack source=artifact bind-
# mounts the +dl plugin + CXI provider; no IB verbs on clariden); engine.sh's
# FI_CXI_*/NCCL_* tunables are := fallbacks that yield to the hook.
# =============================================================================

set -uo pipefail

# ----------------------------------------------------- script + deploy dirs --
# Locate the EDF for both submit styles (manual from the recipe dir; rcc from
# the repo root — the two jobs-3032795/3034577 failures are why this search
# exists; README "Submit-style note"). BASH_SOURCE is the spool copy, not the
# EDF dir — last resort only. ~/.edf kept for the legacy kimi-k3.toml.
OTELA_EDF_NAME="${OTELA_EDF_NAME:-kimi-k3-clariden}"
_edf_dir=""
for _cand in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/deployments/llm/clariden/kimi-k3" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
  [ -n "$_cand" ] && [ -f "$_cand/${OTELA_EDF_NAME}.toml" ] && { _edf_dir="$_cand"; break; }
done
if [ -z "$_edf_dir" ]; then
  echo "FATAL: cannot locate ${OTELA_EDF_NAME}.toml (searched SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>} and the BASH_SOURCE dir). Submit from the deploy dir, or set EDF_PATH explicitly." >&2
  exit 1
fi
export EDF_PATH="${_edf_dir}:${EDF_PATH:-${HOME}/.edf}"

# --------------------------------------------------------------------- lib ---
# Shared helpers from the repo's meta/common/ — lib.sh (stop_otela),
# download_otela.sh; per-recipe copies were removed. Walk up from the EDF dir
# (and, for repo-root submits, SLURM_SUBMIT_DIR) to find it.
COMMON_DIR=""
for _root in "$_edf_dir" "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
  [ -n "$_root" ] || continue
  _d="$_root"
  for _ in 1 2 3 4 5 6; do
    [ -f "$_d/meta/common/lib.sh" ] && { COMMON_DIR="$_d/meta/common"; break 2; }
    _d="$(dirname "$_d")"
  done
done
[ -n "$COMMON_DIR" ] || { echo "FATAL: meta/common/lib.sh not found above $_edf_dir — submit from within the repo checkout" >&2; exit 1; }
source "$COMMON_DIR/lib.sh"   # stop_otela + teardown contract
unset _edf_dir _cand _root _d

# ---------------------------------------------------------------- deployment --
# Reuses the existing /capstor/scratch/cscs/xyao/kimi-k3 tree: the imported
# image, the EDF cache dirs, and a warm Triton cache already live here, so a
# fresh submit does not re-pull or re-JIT from cold.
export DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/kimi-k3}"
export IMAGE="${IMAGE:-$DEPLOY_DIR/images/sglang-kimi-k3.aarch64.sqsh}"
export MODEL_PATH="${MODEL_PATH:-/capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K3}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3}"

# ------------------------------------------------------------------- shape ----
# TP4 x PP<NNODES>: allreduce stays on NVLink, only pipeline send/recv crosses
# Slingshot (TP32 flat: ~1-5 tok/s, no graph capture — README §2).
# PP_SIZE=NNODES (one stage/node); EP_SIZE=TP_SIZE (ep*moe_dp == tp).
export NNODES="${NNODES:-$SLURM_JOB_NUM_NODES}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-$NNODES}"
export EP_SIZE="${EP_SIZE:-$TP_SIZE}"
# Pre-sharded checkpoint (mxfp4->bf16), cut offline once with `servekit prepare
# --format presharded` and keyed by the FULL tp/pp/ep+quant config: a serve job
# whose parallel sizes differ from the dump's checksum.json takes a cache MISS
# (a full source load + re-dump into /dev/shm). Each PP stage reads a DIFFERENT
# file set out of the dump's own checksum.json (not a glob). Path tracks TP/PP:
# /capstor/.../kimi-k3-presharded-tp4pp8/TP-4-sig-<hash>/.
export PRESHARDED_ROOT_BASE="${PRESHARDED_ROOT_BASE:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/kimi-k3-presharded}"
export PRESHARDED_ROOT="${PRESHARDED_ROOT:-${PRESHARDED_ROOT_BASE}-tp${TP_SIZE}pp${PP_SIZE}}"
# PRESHARDED_ENABLE=0 (default): direct Lustre load — the presharded dump is
# unreadable by xyao (ACL mask::---, owner yboughizane); README "Pre-sharded
# checkpoint" has the setfacl fix. ~57 min Lustre load is the working path.
export PRESHARDED_ENABLE="${PRESHARDED_ENABLE:-0}"
# CUDA graph for DECODE only. TP4 x PP8 keeps every captured collective
# intra-node on NVLink, so `full` works. If you switch to TP32 flat you MUST
# set CG_DECODE=disabled (cross-node allreduce cannot be captured on CXI).
export CG_DECODE="${CG_DECODE:-full}"
# Keep `marlin`: the only SM90 MoE backend that keeps K3's mxfp4 weights packed
# (W4A16). `auto` would pick Triton-Kernels, whose upcast_from_mxfp4() dequantizes
# to bf16 and OOMs a 4-bit model on 96 GiB. (Same finding as the JSC recipe.)
export MOE_BACKEND="${MOE_BACKEND:-marlin}"
# Bounds request admission, not the KV pool (sized by MEM_FRAC: 3,732,608
# tokens at 0.90 = 8.01 GB/GPU, job 3000965). Cannot change on a running server.
export CTX_LEN="${CTX_LEN:-1048576}"
export MEM_FRAC="${MEM_FRAC:-0.90}"
# Slingshot high-speed NIC. Every clariden GH200 node exposes hsn0..hsn3 (one
# per GPU NUMA domain); hsn0 is used for the NCCL/GLOO socket bootstrap and is
# where SGLANG_HOST_IP is read from. The ofi plugin handles the data path over
# CXI independently of this setting.
export K3_NIC="${K3_NIC:-hsn0}"
# NCCL operation-watchdog floor (via init_process_group timeout): the 600 s
# default killed job 3018155 ~5.5 h before the scheduler watchdog could ride
# out a transient PP SEND stall (README fix 12). 3600 matches the scheduler
# watchdog; SGLANG_EXTRA_ARGS="--dist-timeout N" overrides.
export DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"

# --- Scheduler watchdog: survive transient PP-broadcast stalls ------------
# The HARD scheduler watchdog (default 300 s) tore down job 3002366 after 8 h
# of healthy serving when one PP metadata broadcast stalled >300 s — a >5-min
# hard kill on a healthy engine is the anti-pattern the load-barrier patch
# already rejects (README fix 11 + op-notes). 3600 s hard / 600 s soft:
# soft dumps a stack (diagnosed 3002366) without crashing; keep soft < hard.
export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-3600}"
export SOFT_WATCHDOG_TIMEOUT="${SOFT_WATCHDOG_TIMEOUT:-600}"

# --- NCCL process-group watchdog (fix 12: clariden job 3018155) ------------
# The NCCL watchdog (init_process_group timeout == --dist-timeout) is a
# SECOND killer on the same transient PP stall; it fired ~5.5 h before the
# scheduler watchdog (README fix 12 has the full symptom chain). Passing
# --dist-timeout aligns the floors; the soft dump is the first signal.

# --- Weight-loading barrier tolerance (fixes clariden job 2999818) --------
# sglang hardcodes a 480 s post-load monitored_barrier (rank 0 only); Lustre
# load spread up to 1555 s was observed (2920471 survived only because rank 0
# was slowest). sitecustomize.py monkey-patches it to 3600 s (README).
export UNBALANCED_MODEL_LOADING_TIMEOUT_S="${UNBALANCED_MODEL_LOADING_TIMEOUT_S:-3600}"
# Distributed torchrun store (rank discovery) and the sglang HTTP port. Both
# are EXPORTED here so srun carries them into the container, where engine.sh
# (a quoted heredoc) resolves them at runtime.
export DIST_PORT="${DIST_PORT:-20000}"
export SERVE_PORT="${SERVE_PORT:-30000}"
# Passthrough for experimental sglang flags without forking this script.
export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

# --- HiCache: offload evicted radix prefixes (MLA KV pages + KDA state) -----
# from HBM to the node's 480 GB Grace LPDDR (~900 GB/s C2C). Grows prefix-cache
# capacity, not max_total_num_tokens. write_through_selective only backs up
# prefixes that get reused; the `kernel` io backend's sgl_kernel.kvcacheio ops
# are confirmed present in this aarch64 image.
#
# HICACHE: off by default to match the proven run (2920471, HICACHE=0).
#
# HiCache is an OPT-IN on this image (verified 2026-08-05, jobs
# 3008263/3009024/3010201). It boots cleanly, adds no measured throughput cost,
# and uses ~59 GB/node LPDDR at ratio 1.0 with the default 3.73M-token HBM
# pool. The host tier is an LRU extension of the radix cache: post-eviction
# hits are dramatic with a capped pool (0.27 s vs 1.90 s), but at full scale a
# large unique flood can LRU-flush even hot prefixes. Leave HICACHE_ENABLE=0
# unless you need the extra cache headroom.
#   - HICACHE_WRITE_POLICY=write_back is the ONLY working policy on this
#     image: write_through_selective never backs up (hit-count trigger is
#     dead in the UnifiedRadixCache hybrid-SSM path) -- do not use it.
# Defaults below are ratio 1.0, policy write_back; enable with HICACHE_ENABLE=1.
#
# NOTE: weight loading from /capstor (Lustre) has WIDE within-node variance
# (up to 1555 s observed in the proven run 2920471). If the slowest rank in a
# TP group is non-zero, the default 480 s monitored_barrier on rank 0 kills
# the job — see the UNBALANCED_MODEL_LOADING_TIMEOUT_S patch below. A job
# that appears frozen at a shard is almost certainly still loading.
export HICACHE_ENABLE="${HICACHE_ENABLE:-0}"
export HICACHE_RATIO="${HICACHE_RATIO:-1.0}"
export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_back}"
export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-kernel}"

# ------------------------------------------------------------------- otela ----
# Official OpenTela release (arm64), staged on a login node by up.sh via
# meta/common/download_otela.sh (latest release; OTELA_VERSION pins a tag).
# Runs INSIDE the kimi-k3 container on the head (mounted via /capstor) so its
# HOME (set by the EDF to /capstor, not quota /users) is correct for its
# libp2p identity + badger DB. The sai-v0.0.6 fork (and the ocf-share copy of
# it) is retired.
export OTELA_DIR="${OTELA_DIR:-/capstor/scratch/cscs/xyao/opentela}"
export OTELA_BIN="${OTELA_BIN:-$OTELA_DIR/opentela-arm64-latest}"
# Direct-mesh bootstrap (an Alps OpenTela peer). No relay needed.
export OTELA_RELAY_ADDR="${OTELA_RELAY_ADDR:-/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7}"
export OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
export OTELA_TCP_PORT="${OTELA_TCP_PORT:-43905}"
export OTELA_UDP_PORT="${OTELA_UDP_PORT:-59820}"
OTELA_STEP_NAME="${OTELA_STEP_NAME:-otela-worker-$SLURM_JOB_ID}"
OTELA_SEED="${OTELA_SEED:-21}"
OTELA_API_PORT="${OTELA_API_PORT:-18094}"
# Cold start ~105 min on a bad Lustre day (~80-132 s/shard x 96); 9000 s
# gives 45 min margin beyond the barrier patch.
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-9000}"

# ------------------------------------------------------------- servekit -----
# servekit (eth-easl/servekit, branch multinode-pp): presharded launch
# (per-rank /dev/shm staging + cold-start profile + run.node<R>.json ready
# rows) and the pre-registration verification bench. Pure stdlib, module-style
# from the checkout. Missing checkout: unwrapped run, slower cold start.
export SERVEKIT_DIR="${SERVEKIT_DIR:-$DEPLOY_DIR/servekit}"
export SERVEKIT_BENCH="${SERVEKIT_BENCH:-1}"              # 0 skips the auto-bench
export SERVEKIT_BENCH_REQUESTS="${SERVEKIT_BENCH_REQUESTS:-64}"
export SERVEKIT_BENCH_CONCURRENCY="${SERVEKIT_BENCH_CONCURRENCY:-16}"
export SERVEKIT_BENCH_IN="${SERVEKIT_BENCH_IN:-768}"      # words (~1024 tokens)
export SERVEKIT_BENCH_OUT="${SERVEKIT_BENCH_OUT:-256}"    # tokens, ignore_eos
export SERVEKIT_BENCH_CORRECTNESS="${SERVEKIT_BENCH_CORRECTNESS:-1}"

export RUNDIR="$DEPLOY_DIR/run-$SLURM_JOB_ID"
PATCH_DIR="$DEPLOY_DIR/patches"
mkdir -p "$DEPLOY_DIR/logs" "$RUNDIR" "$DEPLOY_DIR/cache" "$DEPLOY_DIR/home" \
         "$PATCH_DIR"

# sitecustomize.py: auto-imported at interpreter startup (before sglang code)
# — patches UNBALANCED_MODEL_LOADING_TIMEOUT_S in the serve process. $PATCH_DIR
# is on /capstor; engine.sh re-adds it to PYTHONPATH (the EDF clears it).
cat > "$PATCH_DIR/sitecustomize.py" <<'PYEOF'
import os
try:
    import sglang.srt.model_executor.model_runner_components.load_model_utils as _lm
    _lm.UNBALANCED_MODEL_LOADING_TIMEOUT_S = int(
        os.environ.get("UNBALANCED_MODEL_LOADING_TIMEOUT_S", "3600"))
except Exception:
    pass  # sglang not installed yet (e.g. preflight) — skip
PYEOF
export PATCH_DIR

HEAD="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export HEAD_IP="$(getent hosts "$HEAD" | awk '{print $1}' | head -1)"

echo "[$(date -Is)] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST"
echo "[$(date -Is)] head=$HEAD ($HEAD_IP)  tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE nnodes=$NNODES"
echo "[$(date -Is)] model=$MODEL_PATH  served_as=$SERVED_MODEL_NAME"
echo "[$(date -Is)] edf=$OTELA_EDF_NAME  image=$IMAGE"
echo "[$(date -Is)] otela=$OTELA_BIN  bootstrap=$OTELA_RELAY_ADDR"
echo "[$(date -Is)] hicache: enable=$HICACHE_ENABLE ratio=$HICACHE_RATIO policy=$HICACHE_WRITE_POLICY io=$HICACHE_IO_BACKEND"

# ----------------------------------------------------------------- preflight --
# Pattern: check, and on failure print the EXACT staging command for a node
# that HAS the capability (login node with egress), then exit non-zero.
[ -f "$IMAGE" ] || {
  echo "FATAL: image not found: $IMAGE" >&2
  echo "       Import it ONCE on a login node with build_kimi_k3_image.sbatch" >&2
  echo "       (next to this sbatch), which runs:" >&2
  echo "         enroot import -o $IMAGE docker://lmsysorg/sglang:kimi-k3" >&2
  exit 1
}
[ -d "$MODEL_PATH" ] || { echo "FATAL: weights not found: $MODEL_PATH" >&2; exit 1; }
[ "$PP_SIZE" = "$NNODES" ] || echo "WARN: PP_SIZE=$PP_SIZE != NNODES=$NNODES (intended?)" >&2
[ -x "$OTELA_BIN" ] || {
  echo "FATAL: otela binary not found/executable: $OTELA_BIN" >&2
  echo "       bash up.sh stages the latest official OpenTela (arm64) automatically" >&2
  echo "         OTELA_DIR=$OTELA_DIR OTELA_ARCH=arm64 bash ../../../../meta/common/download_otela.sh" >&2
  echo "       (Override with OTELA_BIN=<path>; rollback: OTELA_VERSION=<tag> download_otela.sh)" >&2
  exit 1
}
# servekit is OPTIONAL (presharded staging + cold-start profile +
# pre-registration bench). Missing checkout -> warn loudly and run unwrapped
# from $MODEL_PATH (slower cold load, no profile); never fail the job.
[ -d "$SERVEKIT_DIR/src/servekit" ] || {
  echo "WARN: servekit checkout not found: $SERVEKIT_DIR — no preshard," >&2
  echo "      no cold-start profile, no auto-bench. Stage it once on a login node:" >&2
  echo "        git clone --depth=1 -b multinode-pp https://github.com/eth-easl/servekit $SERVEKIT_DIR" >&2
}

# --- container preflight (1 node, no GPUs, ~20 s) -----------------------
# Validates the EDF + image modules BEFORE launching 8 nodes (flash_mla
# missing -> FA3 ImportError, job 2914289; kvcacheio missing -> HiCache
# crash at first eviction). --overlap --gres=none: no GPU consumption.
echo "[$(date -Is)] container preflight (EDF=$OTELA_EDF_NAME) ..."
if ! srun --jobid="$SLURM_JOB_ID" --overlap --gres=none --nodes=1 --ntasks=1 \
         -w "$HEAD" --environment="$OTELA_EDF_NAME" python3 -c '
import sglang, sgl_kernel.flash_mla
try:
    import sgl_kernel.kvcacheio   # HiCache kernel io backend
except Exception as e:
    print("kvcacheio:", repr(e)[:80])
print("preflight: sglang", getattr(sglang,"__version__","?"), "flash_mla OK")
         ' 2>&1 | tail -3; then
  echo "FATAL: container preflight failed (EDF load or critical import)." >&2
  echo "       Check the EDF ($OTELA_EDF_NAME) is on EDF_PATH=$EDF_PATH and" >&2
  echo "       the image $IMAGE is intact (sglang-kimi-k3, aarch64, CUDA 13)." >&2
  exit 1
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
# Per-rank wrapper, heredoc'd into $RUNDIR so the sbatch stays self-contained.
# It runs INSIDE the kimi-k3 container (via srun --environment=$OTELA_EDF_NAME).
cat > "$RUNDIR/engine.sh" <<'ENGINE'
#!/bin/bash
set -uo pipefail
RANK="${SLURM_PROCID:-0}"
IFACE="${K3_NIC:-hsn0}"
# Pin NCCL/GLOO socket bootstrap to the high-speed NIC. The ofi plugin (loaded
# by the EDF's com.hooks.aws_ofi_nccl hook) carries the data path over CXI
# independently of this; this is only for the TCP bootstrap path.
export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE"
SGLANG_HOST_IP="$(ip -o -4 addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
[ -z "$SGLANG_HOST_IP" ] && SGLANG_HOST_IP="$(hostname -I | awk '{print $1}')"
export SGLANG_HOST_IP
# --- Slingshot NCCL tunables (reconciled with CSCS netstack hook) -----------
# The EDF's aws_ofi_nccl hook sets curated CXI/NCCL env at container start;
# := lets the hook win, ours only fill gaps (K3-tuned, 2026-08).
# FI_MR_CACHE_MONITOR=userfaultfd avoids mishandled CXI-registered memory.
: "${NCCL_NET:=AWS Libfabric}" "${NCCL_CROSS_NIC:=1}" \
  "${FI_CXI_DISABLE_HOST_REGISTER:=1}" "${FI_CXI_DEFAULT_CQ_SIZE:=131072}" \
  "${FI_CXI_RDZV_THRESHOLD:=0}" "${FI_CXI_RDZV_GET_MIN:=0}" \
  "${FI_MR_CACHE_MONITOR:=userfaultfd}"
export NCCL_NET NCCL_CROSS_NIC FI_CXI_DISABLE_HOST_REGISTER \
  FI_CXI_DEFAULT_CQ_SIZE FI_CXI_RDZV_THRESHOLD FI_CXI_RDZV_GET_MIN \
  FI_MR_CACHE_MONITOR
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
echo "[rank $RANK] netstack: NCCL_NET=$NCCL_NET NCCL_CROSS_NIC=$NCCL_CROSS_NIC FI_CXI_DCQ=$FI_CXI_DEFAULT_CQ_SIZE FI_MR_CACHE_MONITOR=$FI_MR_CACHE_MONITOR"
# Force IPv4 for bootstrap sockets (inherited from the JSC recipe; clariden hsn
# interfaces are IPv4 so this is a safe no-op there).
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-IPv4}" GLOO_SOCKET_FAMILY="${GLOO_SOCKET_FAMILY:-IPv4}"

echo "[rank $RANK] $(hostname) ip=$SGLANG_HOST_IP iface=$IFACE head=$HEAD_IP:$DIST_PORT"
echo "[rank $RANK] shape: tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE cuda_graph_decode=$CG_DECODE"

# sitecustomize.py ($PATCH_DIR on /capstor) patches the load-barrier constant
# at Python startup; PYTHONPATH is re-added here (the EDF clears it).
export PYTHONPATH="${PATCH_DIR:-/capstor/scratch/cscs/xyao/kimi-k3/patches}:${PYTHONPATH:-}"

# Verify the monkey-patch will take effect in this process tree
python3 -c "import sglang.srt.model_executor.model_runner_components.load_model_utils as m; print('[rank '$RANK'] UNBALANCED_MODEL_LOADING_TIMEOUT_S =', m.UNBALANCED_MODEL_LOADING_TIMEOUT_S)" 2>/dev/null || true

# servekit launch: stages this rank's slice (from the dump's own checksum.json
# — PP stages hold DIFFERENT files, a glob can't express that) to /dev/shm,
# rewrites --model-loader-extra-config, profiles cold start, writes
# run.node<RANK>.json at ready (workers keyed on sglang's "Dummy health check
# server started" line — recheck run.node<R>.json after image bumps).
# --overlap is SAFE for presharded: the READY sentinel is written last on a
# clean stage (early readers take a cache MISS, not zeros; check overlap_gate).
# servekit forwards SIGTERM and exits on child exit — the trap still fires.
# --timeout KILLS the engine on expiry: must track $HEALTH_TIMEOUT.
# ulimit -c 0 + unset BASH_ENV: BASH_ENV would nvidia-smi-init every stager
# subshell (16.89 -> 0.41 GB/s); a 1.5 TB core dump is never useful.
ulimit -c 0
unset BASH_ENV

HICACHE_ARGS=()
if [ "${HICACHE_ENABLE:-0}" = "1" ]; then
  HICACHE_ARGS+=(--enable-hierarchical-cache
                 --hicache-ratio "${HICACHE_RATIO:-2.0}"
                 --hicache-write-policy "${HICACHE_WRITE_POLICY:-write_through_selective}"
                 --hicache-io-backend "${HICACHE_IO_BACKEND:-kernel}")
fi

# Prefill AND decode must BOTH be flashmla: this aarch64 sgl_kernel ships no
# FA3, and setting only one knob cancels auto-resolution for the others ->
# silent FA3 fallback -> ImportError (job 2914289). No --mamba-ssm-dtype
# bfloat16: SM90 KDA state is float32 (GDN bf16 is SM100+ only).
# `python -m sglang.launch_server`: the entry point servekit validates.
ENGINE_CMD=(python -m sglang.launch_server
  --trust-remote-code
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" --ep-size "$EP_SIZE"
  --nnodes "$NNODES" --node-rank "$RANK"
  --dist-init-addr "$HEAD_IP:$DIST_PORT"
  --dist-timeout "$DIST_TIMEOUT"
  --moe-runner-backend "$MOE_BACKEND"
  --prefill-attention-backend flashmla --decode-attention-backend flashmla
  --cuda-graph-backend-decode "$CG_DECODE"
  --context-length "$CTX_LEN" --mem-fraction-static "$MEM_FRAC"
  --watchdog-timeout "$WATCHDOG_TIMEOUT" --soft-watchdog-timeout "$SOFT_WATCHDOG_TIMEOUT"
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3
  --enable-metrics --uvicorn-access-log-exclude-prefixes /metrics /health
  --host 0.0.0.0 --port "$SERVE_PORT")
ENGINE_CMD+=(${HICACHE_ARGS[@]+"${HICACHE_ARGS[@]}"})
ENGINE_CMD+=($SGLANG_EXTRA_ARGS)

# PRESHARDED_ENABLE=0 (default): skip servekit launch and exec the engine
# directly — the proven path (jobs 3000965/3002366/3018155/3035026 each served
# ~970 requests before the PP-pipeline stall; dump ACLs broken, README).
# srun forwards SIGTERM to the engine child — the trap still fires.
if [ -d "${SERVEKIT_DIR:-}/src/servekit" ] && [ "${PRESHARDED_ENABLE:-0}" = "1" ]; then
  export PYTHONPATH="${SERVEKIT_DIR}/src:${PYTHONPATH:-}"
  ENGINE_CMD+=(--load-format presharded
               --model-loader-extra-config "{\"presharded_path\": \"$PRESHARDED_ROOT\"}")
  exec python3 -m servekit.cli launch \
    --out "$RUNDIR/run.json" \
    --timeout "${HEALTH_TIMEOUT:-9000}" \
    --overlap \
    -- "${ENGINE_CMD[@]}"
else
  if [ "${PRESHARDED_ENABLE:-0}" != "1" ]; then
    echo "[rank $RANK] PRESHARDED_ENABLE=0: cold load from $MODEL_PATH on Lustre (no servekit launch, no preshard, no profile). Set PRESHARDED_ENABLE=1 once the dump ACLs are fixed." >&2
  else
    echo "[rank $RANK] WARN: servekit not at ${SERVEKIT_DIR:-<unset>} — cold load from $MODEL_PATH (no preshard, no profile)" >&2
  fi
  exec "${ENGINE_CMD[@]}"
fi
ENGINE
chmod +x "$RUNDIR/engine.sh"

# ============================================================ otela worker ====
# Runs INSIDE the container on the HEAD (netns is shared, so it could run on
# the host — job 2999317 — but the EDF's HOME is on /capstor; a host otela
# inherits the quota-limited /users/xyao NFS home and freezes on stale
# handles, per the JSC recipe). One worker only — one endpoint to register.
OTELA_CFG_DIR="$DEPLOY_DIR/otela"
OTELA_CFG="$OTELA_CFG_DIR/cfg.yaml"
mkdir -p "$OTELA_CFG_DIR"
# otela is sai-v0.0.6 (arm64); CLI differs from Beverin's x86 v0.2.2: the
# multiaddr goes to --bootstrap.static (--bootstrap.addr expects an HTTP
# bootstraps-list URL), no --subprocess (JSC separate-worker topology).
# The cfg.yaml (loglevel debug / reachability private / role worker / seed)
# is CRITICAL — without it otela silently does nothing during CRDT sync
# (verified job 3000965: with it, registered + 2000 peers in 45 s).
cat > "$OTELA_CFG" <<CFG
name: clariden-worker-$SLURM_JOB_ID
seed: "$OTELA_SEED"
port: "$OTELA_API_PORT"
tcpport: "$OTELA_TCP_PORT"
udpport: "$OTELA_UDP_PORT"
mode: node
role: worker
loglevel: debug
reachability: private
cleanslate: false
security:
  require_signed_binary: false
solana:
  skip_verification: true
CFG
cat > "$RUNDIR/otela.sh" <<OTELA
#!/bin/bash
set -uo pipefail
# Config file is written by the batch shell (above). It carries loglevel,
# reachability, role, seed, ports, solana.skip_verification — everything that
# must be set before CRDT sync begins. CLI flags below are per-job overrides.
echo "[otela] start head=$HEAD_IP service=$OTELA_SERVICE_NAME:$SERVE_PORT bootstrap=$OTELA_RELAY_ADDR"
exec "$OTELA_BIN" start \\
  --config "$OTELA_CFG" \\
  --bootstrap.static "$OTELA_RELAY_ADDR" \\
  --service.name "$OTELA_SERVICE_NAME" \\
  --service.port "$SERVE_PORT" \\
  --label "model=$SERVED_MODEL_NAME" \\
  --label "cluster=clariden" \\
  --label "launched_by=$USER"
OTELA
chmod +x "$RUNDIR/otela.sh"

# ------------------------------------------------------------- lifecycle -----
# otela publishes LEFT only on SIGTERM/INT; SIGKILL leaves a stale
# connected:true row in the head's round-robin. Signal the STEP and wait.
# stop_otela comes from meta/common/lib.sh (sourced above); the comment trail:
# 20 s fuse < KillWait (30 s); otela needs ~10 s (AnnounceLeave -> 5 s CRDT
# drain -> srv.Shutdown). A vanished step (SIGKILL) keeps a stale
# connected=true row — harmless, the next run mints a fresh peer id.
trap 'stop_otela' EXIT TERM INT

# Register once the head's engine answers. The wait runs in the BATCH shell
# (host netns == container netns, so $HEAD_IP:$SERVE_PORT is reachable here),
# then the otela worker is launched as its own step inside the container.
(
  deadline=$(( SECONDS + HEALTH_TIMEOUT ))
  until curl -sf --max-time 5 "http://$HEAD_IP:$SERVE_PORT/health" >/dev/null 2>&1; do
    [ $SECONDS -ge $deadline ] && { echo "[$(date -Is)] health timeout; not registering" >&2; exit 1; }
    sleep 20
  done
  # Verification bench BEFORE registration: otela traffic must not pollute
  # the numbers. Stdlib-only servekit runs inside the container, module-style
  # from the checkout (the EDF clears PYTHONPATH, hence the explicit env).
  # When the head's servekit-launch report has landed (same ready event the
  # curl loop above waited on), the bench is merged into it (--into) as one
  # verified artifact: cold-start profile + throughput. A small race is
  # harmless: servekit writes run.node0.json at ready, and if it has not
  # landed yet the bench writes a standalone bench.json instead.
  if [ "${SERVEKIT_BENCH:-1}" = "1" ] && [ -d "$SERVEKIT_DIR/src/servekit" ]; then
    echo "[$(date -Is)] engine healthy — servekit bench C=$SERVEKIT_BENCH_CONCURRENCY n=$SERVEKIT_BENCH_REQUESTS (SERVEKIT_BENCH=0 to skip)"
    BENCH_ARGS=(--url "http://127.0.0.1:$SERVE_PORT"
                --requests "$SERVEKIT_BENCH_REQUESTS" --concurrency "$SERVEKIT_BENCH_CONCURRENCY"
                --input-len "$SERVEKIT_BENCH_IN" --output-len "$SERVEKIT_BENCH_OUT")
    [ "${SERVEKIT_BENCH_CORRECTNESS:-1}" = "1" ] || BENCH_ARGS+=(--no-correctness)
    if [ -f "$RUNDIR/run.node0.json" ]; then
      BENCH_ARGS+=(--into "$RUNDIR/run.node0.json")
    else
      BENCH_ARGS+=(--out "$RUNDIR/bench.json")
    fi
    srun --jobid="$SLURM_JOB_ID" --overlap --gres=none --nodes=1 --ntasks=1 \
         -w "$HEAD" --environment="$OTELA_EDF_NAME" \
         env "PYTHONPATH=$SERVEKIT_DIR/src" python3 -m servekit.cli bench "${BENCH_ARGS[@]}" \
      || echo "[$(date -Is)] WARN: servekit bench failed; registering anyway" >&2
  fi
  echo "[$(date -Is)] engine healthy — starting otela worker on $HEAD"
  # Redirect INSIDE the container (not via srun's pipe) to avoid srun
  # output buffering that hides otela log lines until the buffer fills.
  # The log file is on /capstor (mounted in both host and container).
  srun --jobid="$SLURM_JOB_ID" --overlap --gres=none --nodes=1 --ntasks=1 \
       -w "$HEAD" --environment="$OTELA_EDF_NAME" -J "$OTELA_STEP_NAME" \
       bash -c '"$0" >"$1" 2>&1' "$RUNDIR/otela.sh" "$RUNDIR/otela.log"
  _OTELA_LAUNCHED=1   # stop_otela (lib.sh) is active only after a real launch
) &

# ------------------------------------------------------------- run engine ----
# srun inherits the batch shell's exported vars (HEAD_IP, DIST_PORT, SERVE_PORT,
# TP_SIZE, HICACHE_*, MODEL_PATH, …) — standard Slurm behavior — and
# --environment="$OTELA_EDF_NAME" overlays the EDF's [env] (HOME on /capstor,
# HF_HOME, the com.hooks.aws_ofi_nccl annotation, …) on top. The quoted
# engine.sh heredoc then resolves these at runtime inside the container. If a
# var is mysteriously empty on a rank, check: `srun -n1 --environment=$OTELA_EDF_NAME env | grep NAME`.
# One engine rank per node, inside the container, in the foreground so the job
# lifetime == engine lifetime.
srun --ntasks-per-node=1 --environment="$OTELA_EDF_NAME" bash "$RUNDIR/engine.sh"
