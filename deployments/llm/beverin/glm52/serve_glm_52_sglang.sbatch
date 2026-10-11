#!/bin/bash
#SBATCH --job-name=glm52-sglang-beverin
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
# No --cpus-per-task here: shrinking the cpuset breaks SGLANG_SET_CPU_AFFINITY=1
# NUMA pinning (needs the full 0..191 cpuset, one MI300A die per TP rank).
# The srun step passes --cpus-per-task=NODE_CPUS. glm47 lesson.
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
# B:TERM@120 signals the BATCH SHELL so the stop_otela trap can TERM the otela
# step before KillWait — without B: the trap never runs and no LEFT is
# published (head keeps round-robining into a dead endpoint).
#SBATCH --signal=B:TERM@120
#SBATCH --output=/capstor/scratch/cscs/xyao/glm52-sglang-beverin/logs/%x-%j.out
#
# =============================================================================
# Serve zai-org/GLM-5.2 (BF16 ~1.5 TB, DeepSeek Sparse Attention + MoE + MTP)
# on CSCS Beverin (4x MI300A / node, x86_64, gfx942, Slingshot fabric) with the
# plain upstream SGLang ROCm image, and register it on OpenTela — one
# self-contained sbatch + a sglang-rocm.toml EDF + sglang_launcher_mi300a.py +
# gen_correctness.py siblings in this directory.
#
# Submit (defaults: 6 nodes, TP8 x PP3 = 24 GPUs, real weights, served as
# zai-org/GLM-5.2, max-model-len 8192):
#   rcc --profile beverin run --cwd "$PWD" -- sbatch serve_glm_52_sglang.sbatch
#
# FAST FIRST-BRING-UP (recommended — isolates the DSA forward path in ~10 min,
# not the ~4 h real-weight cold start; mirrors how the Kimi-K3 gfx942 forward
# hang was root-caused via a dummy-weight smoke run):
#   rcc --profile beverin run --cwd "$PWD" -- sbatch \
#       --export=ALL,LOAD_FORMAT=dummy,GEN_CORRECTNESS_SMOKE=1 \
#       serve_glm_52_sglang.sbatch
#
# Override context length / shape (must satisfy TP*PP == nodes*GPUS_PER_NODE):
#   rcc --profile beverin run --cwd "$PWD" -- sbatch \
#       --export=ALL,CTX_LEN=131072 serve_glm_52_sglang.sbatch
#
# NOTE on `rcc run --cwd`: `rcc job submit` runs sbatch from the remote_dir
# root (NOT this directory), so SLURM_SUBMIT_DIR points there and the sibling
# files (gen_correctness.py, sglang-rocm.toml) are not found. `rcc run --cwd
# <ABSOLUTE recipe dir> -- sbatch <script>` makes BASH_SOURCE + the resolved
# SCRIPT_DIR land here. Always use the absolute recipe dir for --cwd.
#
# Flow: preflight → one SGLang rank/node in the sglang-rocm container
# (ONE distributed TPxPP engine; only the head binds HTTP) → MANDATORY
# generation probe (meta/common/gen_correctness.py) → only then ONE otela
# worker on the head → on end, SIGTERM the otela step (LEFT).
#
# TOPOLOGY: GLM-5.2 BF16 ~1.5 TB does NOT fit one node — the JSC/kimi-k3
# pattern (ONE distributed engine, ONE otela worker on the head), NOT the
# glm47 per-node-replicas pattern. Default TP8xPP3 over RCCL Socket transport.
# WHY LOWER RISK THAN K3 + container/RCCL details: README sections
# "Why SGLang, and why GLM-5.2 is lower risk" and "NCCL / RCCL on Slingshot".
# =============================================================================

set -uo pipefail

# ----------------------------------------------------- script + deploy dirs --
# `rcc run --cwd <ABSOLUTE recipe dir>` makes BASH_SOURCE resolve here; plain
# `rcc job submit` sets SLURM_SUBMIT_DIR to the remote_dir root and siblings
# are not found (kimi-k3 job 590745).
export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
# Shared scripts (gen_correctness.py, sglang_launcher_mi300a.py) live in the
# repo's meta/common/ — walk up from the recipe dir to find it. Per-recipe
# copies were removed; meta/common is the single canonical home.
COMMON_DIR=""
_cwd="$SCRIPT_DIR"
for _ in 1 2 3 4 5 6; do
  [ -d "$_cwd/meta/common" ] && { COMMON_DIR="$_cwd/meta/common"; break; }
  _cwd="$(dirname "$_cwd")"
done
[ -n "$COMMON_DIR" ] || { echo "FATAL: meta/common/ not found above $SCRIPT_DIR — submit from within the repo checkout" >&2; exit 1; }
export COMMON_DIR   # reaches the container via srun-inherited env (like SCRIPT_DIR)
unset _cwd
export OTELA_EDF_NAME="${OTELA_EDF_NAME:-sglang-rocm}"

# ---------------------------------------------------------------- deployment --
export DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm52-sglang-beverin}"
# GLM-5.2 BF16 on /capstor/store (SwissAI HF cache): 282 shards, ~1.41 TB.
# No remote modeling .py — GlmMoeDsaForCausalLM resolves from the image's
# transformers 5.12.1.
export MODEL_PATH="${MODEL_PATH:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.2}"
# Served under the canonical HF id so the peer is the GLM-5.2 endpoint on the
# mesh (coexists with the glm47-flash registration). Override to a short alias
# if the mesh's served-model routing prefers one.
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-5.2}"
export IMAGE="${IMAGE:-/capstor/scratch/cscs/xyao/.edf_imagestore/sglang+sglang+v0.5.16-rocm720-mi30x.x86_64.sqsh}"

# ------------------------------------------------------------------- shape ----
# TP8xPP3: world=24 must equal nodes*GPUS_PER_NODE. 64 heads -> 8/GPU (valid
# MLA); 78 layers -> 26/stage; BF16 ~59 GiB/GPU of 137 GiB HBM. Full math +
# the TP4-PP6 alternative: README "Multi-node topology".
export NNODES="${NNODES:-$SLURM_JOB_NUM_NODES}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-3}"
if [ "$(( TP_SIZE * PP_SIZE ))" -ne "$(( NNODES * GPUS_PER_NODE ))" ]; then
  echo "FATAL: TP*PP=$((TP_SIZE*PP_SIZE)) != nodes*GPUS_PER_NODE=$((NNODES*GPUS_PER_NODE)). Examples:" >&2
  echo "       6 nodes x 4 GPUs -> TP=8 PP=3 (default, ~59 GiB/GPU BF16)" >&2
  echo "       6 nodes x 4 GPUs -> TP=4 PP=6 (16 heads/GPU, valid but unverified)" >&2
  exit 1
fi
# 8192 = fastest cold start + probe (KV pool is sized by mem-fraction, not
# this); raise toward the 1M max once short-context forward is confirmed.
export CTX_LEN="${CTX_LEN:-8192}"
# 131072 per the AMD cookbook; moot at short context (one chunk).
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-131072}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
# 0.85: ~59 GiB weights + ~57 GiB KV per GPU (glm47 default); lower to give
# cold-start JIT more room if an OOM recurs.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# 1200 s (AMD cookbook) for the first forward's tilelang JIT. Note: the first
# OOM-looking failure was PyTorch's NCCL watchdog, not this one — see DIST_TIMEOUT.
export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-1200}"
# PyTorch NCCL watchdog for ALL collectives; the 600 s default killed job
# 594176 (PP RECV timed out while the previous stage JIT-compiled DSA kernels).
# 7200 s = 2 h; the shared TILELANG_CACHE_DIR makes later forwards fast.
export DIST_TIMEOUT="${DIST_TIMEOUT:-7200}"

# ----------------------------------------------------- MI300A workarounds -----
# Six MI300A (gfx942 APU) workarounds; full text + job numbers: README
# "MI300A (gfx942) workarounds" 1-6 (same numbering). One-liners:
#   1. SGLANG_USE_AITER: BOTH values broken on MI300A today (0: decode hang,
#      1: first forward never starts) — README #1 + DSA matrix.
#   2. DISABLE_CUDA_GRAPH=1: capture SIGKILLs with the MTP (nextn) layer.
#   3. SKIP_SERVER_WARMUP=1: the MTP layer also crashes the warmup pass.
#   4. HSA_NO_SCRATCH_RECLAIM=1: ROCm 7.2 hard-requirement on MI300A.
#   5. is_integrated launcher: per-GPU KV accounting (else cgroup OOM ~35 s
#      after load) — meta/common/sglang_launcher_mi300a.py.
#   6. _patched_get_device_properties(device=None): the aiter DSA decode path
#      calls it with NO positional argument; the patch must replicate that.
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-0}"
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
export SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-${SGLANG_SKIP_SERVER_WARMUP:-1}}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# Cookbook parsers glm45/glm47; GLM52_ENABLE_PARSERS=0 if the image lacks
# them. The /v1/completions probe applies no chat template, so the mandatory
# gate is unaffected.
export SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-glm45}"
export SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-glm47}"
export GLM52_ENABLE_PARSERS="${GLM52_ENABLE_PARSERS:-1}"

# MTP/EAGLE spec-decode DISABLED per the cookbook (draft kernel not validated
# on gfx942) — do not add --speculative-* until it is.
export GLM52_SPECULATIVE="${GLM52_SPECULATIVE:-0}"

# ------------------------------------------------------------------- otela ----
# sai-v0.0.6 build (same as glm47/kimi-k3). multiaddr → --bootstrap.static
# (NOT --bootstrap.addr, which takes a URL). cfg.yaml with loglevel/reachability/
# role/seed is required (Clariden job 3000965): without it otela goes silent
# during CRDT sync and only LOOKS hung. Runs inside the container on the head
# so its HOME (EDF → /capstor) is writable for identity + badger DB.
export OTELA_DIR="${OTELA_DIR:-/capstor/scratch/cscs/xyao/opentela}"
export OTELA_BIN="${OTELA_BIN:-$OTELA_DIR/otela}"
# Direct-mesh bootstrap (an Alps OpenTela peer — the same one the glm47-flash
# recipe uses; the older /ip4/148.187.108.178/... from kimi-k3 is an overrideable
# fallback). No relay needed (Beverin compute has full outbound internet).
export OTELA_RELAY_ADDR="${OTELA_RELAY_ADDR:-/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7}"
export OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
export OTELA_TCP_PORT="${OTELA_TCP_PORT:-43905}"
export OTELA_UDP_PORT="${OTELA_UDP_PORT:-59820}"
export OTELA_API_PORT="${OTELA_API_PORT:-18094}"
OTELA_STEP_NAME="${OTELA_STEP_NAME:-otela-worker-$SLURM_JOB_ID}"
OTELA_SEED="${OTELA_SEED:-21}"

# Cold start: ~1.41 TB over 282 shards; ~3-4 h worst case unstriped — the
# one-time lfs setstripe in the README (Submission → Prerequisites) cuts it
# to minutes. Until then this gates the pre-registration wait only.
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-14400}"
# Mandatory pre-registration generation probe (deepseek-v4 / K3 lesson).
# Set GEN_PROBE=0 to skip (NOT recommended — registers on /health alone).
export GEN_PROBE="${GEN_PROBE:-1}"
# First request compiles DSA tilelang kernels → 1200 s (kimi-k3 uses 600 s).
export GEN_PROBE_TIMEOUT="${GEN_PROBE_TIMEOUT:-1200}"
# Correctness probe knobs (see gen_correctness.py). MIN_PASS=5 tolerates one
# soft-prompt fluke while requiring ALL crisp prompts (Paris, primes, 40 km/h).
export GEN_CORRECTNESS_MAX_TOKENS="${GEN_CORRECTNESS_MAX_TOKENS:-64}"
export GEN_CORRECTNESS_MIN_PASS="${GEN_CORRECTNESS_MIN_PASS:-5}"
export GEN_CORRECTNESS_PER_REQ_TIMEOUT="${GEN_CORRECTNESS_PER_REQ_TIMEOUT:-300}"
# GEN_CORRECTNESS_SMOKE=1 (+ LOAD_FORMAT=dummy): probe checks non-empty
# continuation instead of factual answers — the fastest forward-pass smoke.
export GEN_CORRECTNESS_SMOKE="${GEN_CORRECTNESS_SMOKE:-0}"

# Image bundles transformers 5.12.1 (ships GlmMoeDsaForCausalLM); the engine
# guard is a no-op unless overridden for an older image.
export GLM52_TRANSFORMERS_VERSION="${GLM52_TRANSFORMERS_VERSION:-5.12.1}"

export RUNDIR="$DEPLOY_DIR/run-$SLURM_JOB_ID"
export TRANSFORMERS_MARKER="$RUNDIR/.transformers_ok"
mkdir -p "$DEPLOY_DIR/logs" "$RUNDIR" \
         "$DEPLOY_DIR/home/pylib" \
         "$DEPLOY_DIR/cache/huggingface" \
         "$DEPLOY_DIR/cache/triton" \
         "$DEPLOY_DIR/cache/tilelang" \
         "$DEPLOY_DIR/cache/tmp"

# Per-job TMPDIR under $RUNDIR (Triton/LLVM JIT scratch), cleaned at job end.
export TMPDIR="${TMPDIR:-$RUNDIR/tmp}"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR" 2>/dev/null || true

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export HEAD="${NODES[0]}"
export HEAD_IP="$(getent hosts "$HEAD" | awk '{print $1}' | head -1)"
# Distributed-init port (TCPStore master on the head). Separate from SERVE_PORT
# (the HTTP port the head binds) — same separation as kimi-k3's MASTER_PORT/
# SERVE_PORT.
export MASTER_PORT="${MASTER_PORT:-29500}"
export SERVE_PORT="${SERVE_PORT:-8080}"

echo "[$(date -Is)] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST nnodes=$NNODES"
echo "[$(date -Is)] head=$HEAD ($HEAD_IP)  tp=$TP_SIZE pp=$PP_SIZE  (world=$((TP_SIZE*PP_SIZE)) gpus=$((NNODES*GPUS_PER_NODE)) gpus_per_node=$GPUS_PER_NODE)"
echo "[$(date -Is)] model=$MODEL_PATH  served_as=$SERVED_MODEL_NAME  ctx=$CTX_LEN  max_running=$MAX_RUNNING_REQUESTS  gpu_mem=$GPU_MEM_UTIL  chunked_prefill=$CHUNKED_PREFILL_SIZE"
echo "[$(date -Is)] dsa_prefill=$GLM52_DSA_PREFILL dsa_decode=$GLM52_DSA_DECODE dsa_topk=sgl-kernel(default) speculative=$GLM52_SPECULATIVE watchdog=$WATCHDOG_TIMEOUT dist=$DIST_TIMEOUT"
echo "[$(date -Is)] disable_cuda_graph=$DISABLE_CUDA_GRAPH skip_server_warmup=$SKIP_SERVER_WARMUP sglang_use_aiter=$SGLANG_USE_AITER load_format=$LOAD_FORMAT"
echo "[$(date -Is)] parsers=glm45/$SGLANG_REASONING_PARSER glm47/$SGLANG_TOOL_CALL_PARSER enable_parsers=$GLM52_ENABLE_PARSERS"
echo "[$(date -Is)] edf=$OTELA_EDF_NAME  image=$IMAGE"
echo "[$(date -Is)] otela=$OTELA_BIN  bootstrap=$OTELA_RELAY_ADDR  serve_port=$SERVE_PORT  master_port=$MASTER_PORT"
echo "[$(date -Is)] gen_probe=$GEN_PROBE  smoke=$GEN_CORRECTNESS_SMOKE  min_pass=$GEN_CORRECTNESS_MIN_PASS  max_tokens=$GEN_CORRECTNESS_MAX_TOKENS  health_timeout=$HEALTH_TIMEOUT"
echo "[$(date -Is)] started=$(date --iso-8601=seconds)"

# ----------------------------------------------------------------- preflight --
[ -f "$IMAGE" ] || {
  echo "FATAL: SGLang image not found: $IMAGE" >&2
  echo "       One-time import onto a login node with egress, e.g.:" >&2
  echo "         enroot import -o $IMAGE docker://lmsysorg/sglang:v0.5.16-rocm720-mi30x" >&2
  exit 1
}
# With LOAD_FORMAT=dummy the weights are not read, so skip the model-dir check
# to allow a fast forward-path smoke run before the ~4 h real-weight cold start.
if [ "$LOAD_FORMAT" != "dummy" ]; then
  [ -d "$MODEL_PATH" ] || { echo "FATAL: weights not found: $MODEL_PATH" >&2; exit 1; }
fi
[ -x "$OTELA_BIN" ] || {
  echo "FATAL: otela binary not found/executable: $OTELA_BIN" >&2
  echo "       Stage the verified sai-dialect x86_64 build, e.g.:" >&2
  echo "         mkdir -p $OTELA_DIR && cp <x86_64 otela> $OTELA_BIN && chmod +x $OTELA_BIN" >&2
  exit 1
}
_sib() { # name, why — checks the recipe dir, then meta/common (shared)
  { [ -f "$SCRIPT_DIR/$1" ] || [ -f "$COMMON_DIR/$1" ]; } || {
    echo "FATAL: $1 not found in $SCRIPT_DIR or $COMMON_DIR" >&2
    echo "       $2" >&2
    echo "       Submit with: rcc --profile beverin run --cwd \"$(cd "$SCRIPT_DIR" && pwd)\" -- sbatch $0" >&2
    exit 1
  }
}
_sib gen_correctness.py "Shared probe, canonical copy in meta/common/."
_sib sglang-rocm.toml "The EDF (image + env). Found via EDF_PATH=$SCRIPT_DIR."
_sib sglang_launcher_mi300a.py "The is_integrated MI300A memory-accounting fix launcher, canonical in meta/common/."
COMMON() { [ -f "$SCRIPT_DIR/$1" ] && echo "$SCRIPT_DIR/$1" || echo "$COMMON_DIR/$1"; }
# A stale marker from a prior run would make siblings skip the transformers
# check while rank 0 re-installs. Clean it (the check is idempotent).
rm -f "$TRANSFORMERS_MARKER" 2>/dev/null || true

# ================================================================ engine ======
# Per-rank wrapper, heredoc'd into $RUNDIR so the sbatch stays self-contained
# apart from the EDF + gen_correctness.py + sglang_launcher_mi300a.py siblings.
# Runs INSIDE the sglang-rocm container (via srun --environment=$OTELA_EDF_NAME).
# Quoted heredoc: vars resolve at runtime inside the container from the
# srun-inherited + EDF-overlaid env.
cat > "$RUNDIR/engine.sh" <<'ENGINE'
#!/bin/bash
set -uo pipefail
RANK="${SLURM_NODEID:-0}"
NODE="$(hostname)"
echo "[rank $RANK] $(hostname) node-rank=$RANK dist-init=$HEAD:$MASTER_PORT serve=$SERVE_PORT ctx=$CTX_LEN tp=$TP_SIZE pp=$PP_SIZE"

# --- transformers: ensure GlmMoeDsaForCausalLM is importable ----------------
# Image bundles 5.12.1; rank 0 checks + guarded install, siblings wait on a
# marker (no concurrent pip).
if [ "$RANK" = "0" ]; then
  if python3 -c "from transformers import GlmMoeDsaForCausalLM" 2>/dev/null; then
    echo "[rank 0] transformers has GlmMoeDsaForCausalLM (no install needed)"
  else
    echo "[rank 0] transformers lacks GlmMoeDsaForCausalLM; installing $GLM52_TRANSFORMERS_VERSION ..."
    python3 -m pip install --no-input --break-system-packages \
      "transformers==$GLM52_TRANSFORMERS_VERSION" || {
      echo "[rank 0] FATAL: pip install transformers==$GLM52_TRANSFORMERS_VERSION failed." >&2
      echo "[rank 0]        The image must bundle a transformers with glm_moe_dsa" >&2
      echo "[rank 0]        (the EDF pins SGLANG_USE_AITER=0 + the image's 5.12.1 should have it)." >&2
      exit 1
    }
  fi
  touch "$TRANSFORMERS_MARKER"
  echo "[rank 0] TRANSFORMERS_OK marker=$TRANSFORMERS_MARKER"
else
  echo "[rank $RANK] waiting for transformers marker $TRANSFORMERS_MARKER ..."
  for i in $(seq 1 900); do
    [ -f "$TRANSFORMERS_MARKER" ] && break
    sleep 2
  done
fi
[ -f "$TRANSFORMERS_MARKER" ] || { echo "[rank $RANK] FATAL: transformers marker missing after 30 min" >&2; exit 1; }
sleep 5   # full barrier past the marker write

# --- MI300A + RCCL environment ----------------------------------------------
# HSA_NO_SCRATCH_RECLAIM=1 is also in the EDF [env] (mandatory on MI300A +
# ROCm 7.2); re-export in case a parent shell clobbered it.
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
# SGLANG_USE_AITER=0 (EDF [env]) avoids the aiter get_rope crash on GLM rope
# configs. The DSA path (dsa_backend.py) does NOT read SGLANG_USE_AITER, so
# DSA's aiter paged-MQA-logits usage is unaffected. Re-export for safety.
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-0}"

# Shared Triton + tilelang caches on /capstor: only the first run pays JIT.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DEPLOY_DIR/cache/triton}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$DEPLOY_DIR/cache/tilelang}"
mkdir -p "$TRITON_CACHE_DIR" "$TILELANG_CACHE_DIR" 2>/dev/null || true
# TMPDIR on scratch (enroot overlay tmpfs competes with model memory + hits
# "Disk quota exceeded" during JIT).
export TMPDIR="${TMPDIR:-$DEPLOY_DIR/cache/tmp}"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR" 2>/dev/null || true
echo "[rank $RANK] dirs: TRITON_CACHE_DIR=$TRITON_CACHE_DIR TILELANG_CACHE_DIR=$TILELANG_CACHE_DIR TMPDIR=$TMPDIR HOME=$HOME"
# Single-threaded LLVM/Triton/tilelang JIT keeps per-worker memory low during
# the simultaneous cold-start compile on 4 workers/node.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# Cause glibc malloc to return freed heap to the OS more aggressively (default
# 128 KiB). With 4 workers/node each freeing Python objects after load_model,
# this prevents the heap from retaining several GiB of fragmented space.
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-65536}"

# --- RCCL networking: drop the image's CUDA OFI plugin, use Socket ----------
# Same as kimi-k3: CUDA-built aws_ofi_nccl cannot init on ROCm; fall back to
# RCCL's built-in Socket transport over hsn0 (slow; TODO HIP-aware OFI/CXI).
unset NCCL_NET_PLUGIN NCCL_NET
export NCCL_SOCKET_IFNAME="${GLM52_NIC:-hsn0}"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-1}"
# gfx942 RCCL has no LL cutoff points: tiny collectives pick LL and loop
# forever (job 594097, C-level hang below every watchdog). Force SIMPLE +
# one channel; re-enable LL via --export=ALL,NCCL_PROTO=LL for perf only.
# Details: README "NCCL_PROTO=SIMPLE — gfx942 LL protocol hang".
export NCCL_PROTO="${NCCL_PROTO:-SIMPLE}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# --- build the SGLang serve args (ALL ranks; node_rank>=1 auto-skips HTTP) ---
SGLANG_ARGS=(
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --host 0.0.0.0 --port "$SERVE_PORT"
  --tensor-parallel-size "$TP_SIZE"
  --pipeline-parallel-size "$PP_SIZE"
  --nnodes "$NNODES" --node-rank "$RANK" --dist-init-addr "$HEAD:$MASTER_PORT"
  --context-length "$CTX_LEN"
  --mem-fraction-static "$GPU_MEM_UTIL"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --watchdog-timeout "$WATCHDOG_TIMEOUT"
  --dist-timeout "$DIST_TIMEOUT"
  --enable-metrics
)
# DSA backends: cookbook tilelang is validated on MI300X but crashes on MI300A
# (jobs 594304/594340); aiter works for PREFILL but its decode hangs (README
# "DSA backend test matrix" has the full job-by-job table). aiter is the
# default pending the flashmla_sparse test (job 594548). Override via
# --export=ALL,GLM52_DSA_PREFILL=...,GLM52_DSA_DECODE=... (fa3/trtllm NVIDIA-only).
GLM52_DSA_PREFILL="${GLM52_DSA_PREFILL:-aiter}"
GLM52_DSA_DECODE="${GLM52_DSA_DECODE:-aiter}"
SGLANG_ARGS+=(
  --dsa-prefill-backend "$GLM52_DSA_PREFILL"
  --dsa-decode-backend "$GLM52_DSA_DECODE"
)
# GLM-5.2 reasoning + tool-call parsers (cookbook: glm45 / glm47).
if [ "${GLM52_ENABLE_PARSERS:-1}" = "1" ]; then
  SGLANG_ARGS+=(
    --reasoning-parser "$SGLANG_REASONING_PARSER"
    --tool-call-parser "$SGLANG_TOOL_CALL_PARSER"
  )
fi
# MI300A workarounds #2/#3: capture + warmup both crash the MTP layer.
if [ "${DISABLE_CUDA_GRAPH:-1}" = "1" ]; then
  SGLANG_ARGS+=(--disable-cuda-graph)
fi
if [ "${SKIP_SERVER_WARMUP:-1}" = "1" ]; then
  SGLANG_ARGS+=(--skip-server-warmup)
fi
# --load-format: override weight loading (e.g. LOAD_FORMAT=dummy for a fast
# forward-path smoke run without the ~4 h real-weight cold start).
if [ -n "${LOAD_FORMAT:-}" ] && [ "${LOAD_FORMAT:-}" != "auto" ]; then
  SGLANG_ARGS+=(--load-format "$LOAD_FORMAT")
fi

echo "[rank $RANK] launching SGLang (head=$([ "$RANK" = 0 ] && echo "serve(HTTP)+probe+otela" || echo "node_rank>=1(no HTTP)")) args=${SGLANG_ARGS[*]}"

# is_integrated fix launcher (README workaround #5); every node runs its own
# copy via srun.
exec python3 "$COMMON_DIR/sglang_launcher_mi300a.py" "${SGLANG_ARGS[@]}"
ENGINE
chmod +x "$RUNDIR/engine.sh"

# ============================================================ otela worker ====
# Runs INSIDE the sglang-rocm container on the HEAD node only (one peer: the
# single distributed engine has one head HTTP endpoint, and launching on every
# node would over-register it). The container shares the host netns, so otela
# on the head reaches SGLang on localhost:$SERVE_PORT.
OTELA_CFG_DIR="$DEPLOY_DIR/otela"
OTELA_CFG="$OTELA_CFG_DIR/cfg.yaml"
mkdir -p "$OTELA_CFG_DIR"
cat > "$OTELA_CFG" <<CFG
name: beverin-worker-$SLURM_JOB_ID
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
echo "[otela] start head=$HEAD service=$OTELA_SERVICE_NAME:$SERVE_PORT bootstrap=$OTELA_RELAY_ADDR"
exec "$OTELA_BIN" start \\
  --config "$OTELA_CFG" \\
  --bootstrap.static "$OTELA_RELAY_ADDR" \\
  --service.name "$OTELA_SERVICE_NAME" \\
  --service.port "$SERVE_PORT" \\
  --label "model=$SERVED_MODEL_NAME" \\
  --label "cluster=beverin" \\
  --label "framework=sglang" \\
  --label "launched_by=\${USER:-unknown}"
OTELA
chmod +x "$RUNDIR/otela.sh"

# ------------------------------------------------------------- lifecycle -----
# otela only publishes LEFT on SIGTERM (AnnounceLeave); SIGKILL leaves a stale
# connected:true row. --signal=B:TERM@120 + this batch-shell trap TERM the step.
_OTELA_STOPPED=0
stop_otela() {
  [ "$_OTELA_STOPPED" = 1 ] && return 0   # trap fires for both TERM and EXIT
  _OTELA_STOPPED=1
  local step
  step="$(squeue -s -j "$SLURM_JOB_ID" -h -o '%i %j' 2>/dev/null \
          | awk -v n="$OTELA_STEP_NAME" '$2==n{print $1}' | head -1)"
  if [ -z "$step" ]; then
    echo "[$(date -Is)] WARN: otela step $OTELA_STEP_NAME not found; cannot signal it." >&2
    echo "[$(date -Is)]       If it was SIGKILLed, no LEFT was published and the head" >&2
    echo "[$(date -Is)]       may keep a stale connected=true row for this worker." >&2
    return 0
  fi
  echo "[$(date -Is)] stopping otela step $step"
  scancel -s TERM "$step" 2>/dev/null
  for _ in $(seq 1 20); do
    [ -z "$(squeue -s -j "$SLURM_JOB_ID" -h -o '%i %j' 2>/dev/null \
            | awk -v n="$OTELA_STEP_NAME" '$2==n{print $1}')" ] && break
    sleep 1
  done
  grep -aqE "Announcing myself as LEFT|Leaving network" "$RUNDIR/otela.log" 2>/dev/null \
    && echo "[$(date -Is)] otela: graceful leave confirmed" \
    || echo "[$(date -Is)] WARN: otela did not announce LEFT" >&2
}
trap 'stop_otela' EXIT TERM INT

# Wait + probe run in the BATCH shell (host netns == container netns); the
# otela worker launches as its own step inside the container. The probe is
# MANDATORY: /health 200 does not prove a forward works (deepseek-v4 502'd
# every request after registering on /health; K3 hung with /health up).
(
  deadline=$(( SECONDS + HEALTH_TIMEOUT ))
  until curl -sf --max-time 5 "http://$HEAD:$SERVE_PORT/health" >/dev/null 2>&1; do
    [ $SECONDS -ge $deadline ] && { echo "[$(date -Is)] health timeout after ${HEALTH_TIMEOUT}s; not registering" >&2; exit 1; }
    sleep 20
  done
  echo "[$(date -Is)] engine healthy (head=$HEAD:$SERVE_PORT) after ~$((HEALTH_TIMEOUT - (deadline - SECONDS)))s"

  if [ "$GEN_PROBE" = "1" ]; then
    echo "[$(date -Is)] generation ${GEN_CORRECTNESS_SMOKE:+smoke }probe (model=$SERVED_MODEL_NAME, max_tokens=$GEN_CORRECTNESS_MAX_TOKENS, min_pass=$GEN_CORRECTNESS_MIN_PASS, budget=${GEN_PROBE_TIMEOUT}s) ..."
    PROBE_OUT="$RUNDIR/gen_correctness_${SLURM_JOB_ID}.json"
    # Run INSIDE the container on the head (host python may be too old; the
    # container shares the host netns so 127.0.0.1 reaches the engine).
    GEN_CORRECTNESS_TIMEOUT="$GEN_PROBE_TIMEOUT" \
    GEN_CORRECTNESS_PER_REQ_TIMEOUT="$GEN_CORRECTNESS_PER_REQ_TIMEOUT" \
    GEN_CORRECTNESS_SMOKE="$GEN_CORRECTNESS_SMOKE" \
      timeout "$((GEN_PROBE_TIMEOUT + 300))" \
      srun --jobid="$SLURM_JOB_ID" --overlap --gres=none --nodes=1 --ntasks=1 \
           -w "$HEAD" --environment="$OTELA_EDF_NAME" \
           python3 "$(COMMON gen_correctness.py)" \
               "http://127.0.0.1:$SERVE_PORT" "$SERVED_MODEL_NAME" \
               "$PROBE_OUT" "$GEN_CORRECTNESS_MAX_TOKENS" "$GEN_CORRECTNESS_MIN_PASS"
    rc=$?
    if [ "$rc" = 0 ]; then
      echo "[$(date -Is)] GEN_PROBE_OK rc=$rc (report in $PROBE_OUT)"
    else
      echo "[$(date -Is)] GEN_PROBE_FAILED rc=$rc; NOT registering (report in $PROBE_OUT)" >&2
      echo "[$(date -Is)]       Inspect $PROBE_OUT + the engine log, then scancel if needed." >&2
      echo "[$(date -Is)]       Most likely the DSA (tilelang/aiter) forward path" >&2
      echo "[$(date -Is)]       faults on MI300A gfx942 — the one new piece vs glm47." >&2
      exit 1
    fi
  fi

  echo "[$(date -Is)] starting otela worker on $HEAD (inside container, EDF=$OTELA_EDF_NAME)"
  # Redirect INSIDE the container (not via srun's pipe) to avoid srun output
  # buffering that hides otela log lines until the buffer fills.
  srun --jobid="$SLURM_JOB_ID" --overlap --gres=none --nodes=1 --ntasks=1 \
       -w "$HEAD" --environment="$OTELA_EDF_NAME" -J "$OTELA_STEP_NAME" \
       bash -c '"$0" >"$1" 2>&1' "$RUNDIR/otela.sh" "$RUNDIR/otela.log"
) &

# Record endpoints for all nodes before srun starts so the metadata is
# available while the services are still running.
{
  echo "SERVICE_JOB_ID=$SLURM_JOB_ID"
  echo "SERVICE_HEAD_NODE=$HEAD"
  echo "SERVICE_HEAD_IP=$HEAD_IP"
  echo "SERVICE_PORT=$SERVE_PORT"
  echo "SERVICE_URL=http://$HEAD_IP:$SERVE_PORT"
  echo "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
  echo "FRAMEWORK=sglang"
  echo "TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE CTX_LEN=$CTX_LEN"
  echo "DSA_PREFILL=$GLM52_DSA_PREFILL DSA_DECODE=$GLM52_DSA_DECODE DSA_TOPK=sgl-kernel"
  echo "BOOTSTRAP=$OTELA_RELAY_ADDR"
  echo "SUBMITTED_AT=$(date --iso-8601=seconds)"
  for i in "${!NODES[@]}"; do
    echo "ENDPOINT_${i}=${NODES[$i]}:$SERVE_PORT"
  done
} > "$DEPLOY_DIR/last_service.env"

# ------------------------------------------------------------- run engine ----
# One SGLang rank per node, foreground (job lifetime == engine lifetime).
# srun inherits the exported vars; --environment overlays the EDF [env];
# --cpus-per-task=NODE_CPUS keeps the 0..191 cpuset for NUMA pinning.
NODE_CPUS="${SLURM_CPUS_ON_NODE:-$(scontrol show node "${HEAD}" 2>/dev/null | awk -F= '/^CPUTot=/{print $2; exit}')}"
NODE_CPUS="${NODE_CPUS:-192}"
srun --nodes="$NNODES" --ntasks-per-node=1 \
     --cpus-per-task="$NODE_CPUS" --gpus-per-task=4 \
     --environment="$OTELA_EDF_NAME" bash "$RUNDIR/engine.sh"
SRUN_EXIT=$?
echo "[$(date -Is)] srun_exited code=${SRUN_EXIT}"

exit "$SRUN_EXIT"
