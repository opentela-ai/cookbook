#!/bin/bash
#SBATCH --job-name=k3-vllm-beverin
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
# Serve Kimi-K3 (2.8 T hybrid MoE, mxfp4) on CSCS Beverin (4x MI300A/node,
# gfx942, Slingshot) with vLLM, registered on OpenTela — one self-contained
# sbatch + siblings (kimi-k3-vllm.toml EDF, k3_patch.py, sitecustomize.py,
# gen_correctness.py via meta/common walk-up).
#
# Submit (defaults: 6 nodes, TP8 x PP3 = 24 GPUs, moonshotai/Kimi-K3, 128K ctx):
#   sbatch serve_kimi_k3_otela_beverin.sbatch
#   sbatch --export=ALL,CTX_LEN=1048576 serve_kimi_k3_otela_beverin.sbatch
#   sbatch --nodes=2 --export=ALL,TP_SIZE=4,PP_SIZE=2 serve_kimi_k3_otela_beverin.sbatch
#
# Pipeline: preflight (EDF/image/weights/otela/patch siblings) → one vLLM
# engine rank per node (rank 0 head serves HTTP; ranks 1..N --headless P2P) →
# rank 0 applies k3_patch.py before launch → MANDATORY correctness probe
# (six greedy /v1/completions prompts, MIN_PASS=5/6) BEFORE otela registration
# (/health alone is not enough; K3 never generated a token on MI300A) → one
# otela worker on the head registers the `llm` service → on job end/scancel
# SIGTERM the otela step so it announces LEFT cleanly.
#
# TOPOLOGY — JSC pattern (NOT per-node replicas like GLM-4.7): ONE distributed
# vLLM engine across all nodes (K3 ~1.5 TB mxfp4 doesn't fit a node), ONE
# otela worker on the head. Default TP8 x PP3 spans 2 nodes per TP group, so
# MoE allreduce crosses Slingshot over RCCL Socket transport (hsn*) — the
# verified bring-up path (k3-eng8/11 reached MoE backend selection:
# ROCM_AITER_MLA + AITER_MXFP4_BF16). The EDF's OFI/CXI plugin initialises
# but the first cross-node collective HANGS (rccl-xport 601320/601321/…) —
# engine.sh neutralises it; see README. Pivot all-to-all + channel tuning
# default ON since run-601453 (a2a GPU time −31%, agg tok/s 20.6→23.8).
# TP4 x PP4 keeps TP allreduce intra-node; unverified, see README.
#
# NO RELAY: beverin compute nodes reach the Alps OpenTela bootstrap directly
# (verified) — the worker dials /ip4/148.187.108.178/... without a relay.
#
# AITER MLA head constraint (job 585845): num_attention_heads=96 → heads/GPU
# must be in [1,15] or mult of 16. TP8→12 OK; TP4→24 INVALID.
# APU memory: 4 workers/node x ~99.5 GiB (71 GiB dummy weights + overhead)
# = 398 < 478 GiB node RAM; each GPU holds ~71 GiB weights, ~57 GiB HBM free
# (24 GPUs, 6 nodes). gpu-memory-utilization 0.8 (0.9 OOMs the OS).
#
# Shape rationale + experiments (K3_PIECEWISE / K3_BREAKABLE_PIECEWISE /
# cudagraph paths) live in README.md — the knobs remain functional here.

set -uo pipefail

# ----------------------------------------------------- script + deploy dirs --
# SCRIPT_DIR: this sbatch + siblings (EDF, k3_patch.py, sitecustomize.py).
# Slurm runs the sbatch from a spool copy, so ${BASH_SOURCE[0]} is NOT the
# submit dir — SLURM_SUBMIT_DIR is.
export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"

# TMPDIR/TRITON_CACHE_DIR must live on /capstor/scratch (unlimited quota):
# $HOME/.tmp and /capstor/store both hit disk-quota failures mid-JIT (jobs
# 583899, 585851) — see TRITON_CACHE_DIR in engine.sh.
OTELA_EDF_NAME="${OTELA_EDF_NAME:-kimi-k3-vllm}"

# ---------------------------------------------------------------- deployment --
export DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin}"
# k3_patch.py reads $K3 and writes its aiter overlay to $K3/home/pylib; the EDF
# sets HOME=$DEPLOY_DIR/home, so $K3/home == container HOME. Keep them in lockstep.
export K3="${K3:-$DEPLOY_DIR}"
export IMAGE="${IMAGE:-/capstor/scratch/cscs/xyao/.edf_imagestore/vllm+vllm-openai-rocm+kimi-k3.x86_64.sqsh}"
export MODEL_PATH="${MODEL_PATH:-/capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K3}"
# Canonical HF id: coexists with the Clariden sglang K3 registration (the
# mesh round-robins them as additive capacity).
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-SwissAI-Research/moonshot/kimi-k3-rocm}"

# ------------------------------------------------------------------- shape ----
# TP8 x PP3 (default): world 24 = nodes*4. TP=8 spans 2 nodes -> MoE
# allreduce crosses Slingshot (verified bring-up path, k3-eng8/11). PP=3:
# 31 layers each, 112 experts/GPU. AITER MLA: heads/GPU must be in [1,15]
# or mult of 16 (96 heads): TP8->12 OK, TP4->24 INVALID (job 585845). 24
# GPUs hold ~71 GiB weights/GPU, ~57 GiB HBM free (TP8xPP2/16 GPUs
# HSA-OOM'd, job 585844). Details: README "Why the mi300 partition".
export NNODES="${NNODES:-$SLURM_JOB_NUM_NODES}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-3}"
if [ "$(( TP_SIZE * PP_SIZE ))" -ne "$(( NNODES * GPUS_PER_NODE ))" ]; then
  echo "FATAL: TP*PP=$((TP_SIZE*PP_SIZE)) != nodes*GPUS_PER_NODE=$((NNODES*GPUS_PER_NODE)). Examples:" >&2
  echo "       6 nodes x 4 GPUs -> TP=8 PP=3 (default)" >&2
  echo "       8 nodes x 2 GPUs -> TP=8 PP=2 (GPU HBM OOM, see header notes)" >&2
  echo "       6 nodes x 4 GPUs -> TP=4 PP=6 (AITER MLA: 24 heads/GPU INVALID)" >&2
  exit 1
fi
if [ "${K3_BREAKABLE_PIECEWISE:-0}" = "1" ] && [ "${PP_SIZE:-3}" != "1" ]; then
  # K3_BREAKABLE_PIECEWISE with PP>1: relies on breakable PIECEWISE (breaks
  # fire at each @eager_break_during_capture op) to avoid the PP recv_object
  # deadlock (jobs 588153/588156); a deadlock now fails in
  # DISTRIBUTED_TIMEOUT_SECONDS, not 1 h. See README "Cudagraph experiments".
  echo "[rank ${RANK:-0}] K3_BREAKABLE_PIECEWISE=1 with PP_SIZE=${PP_SIZE:-3}: PP recv present;" >&2
  echo "[rank ${RANK:-0}]    relying on breakable PIECEWISE (breaks fire) to avoid the PP recv deadlock." >&2
  echo "[rank ${RANK:-0}]    Deadlock timeout=${DISTRIBUTED_TIMEOUT_SECONDS:-3600}s." >&2
fi
# CTX_LEN: K3 supports max_position_embeddings=1048576 (1M). The KV pool is
# sized by --gpu-memory-utilization, not this value (MLA+KDA ~107 KB/token).
export CTX_LEN="${CTX_LEN:-131072}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
# --max-num-batched-tokens: vLLM default 2048 starves high-concurrency
# sweeps (job 600469); 8192 matches the BENCHMARK.md baseline (job 598005).
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# MI300A APU: HBM shared CPU/GPU. 0.9 OOMs the OS; 0.8 leaves ~91 GiB.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
# vLLM distributed init store (rank discovery) + head HTTP port.
export MASTER_PORT="${MASTER_PORT:-6379}"
export SERVE_PORT="${SERVE_PORT:-8080}"
# Passthrough for experimental vLLM flags (e.g. ENFORCE_EAGER=1 adds
# --enforce-eager) without forking this script.
# K3_NIC=hsn: fan Socket over ALL Slingshot NICs (single hsn0 = pre-601453
# path); gloo is pinned to hsn0 separately below.
export K3_NIC="${K3_NIC:-hsn}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
if [ "${ENFORCE_EAGER:-0}" = "1" ]; then
  export VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS --enforce-eager"
fi

# Chat template emits thought blocks; tool calls need --enable-auto-tool-choice +
# --tool-call-parser kimi_k3 + --reasoning-parser (ships in the image). If an
# image bump drops the parser, set K3_ENABLE_PARSERS=0 to recover. The
# /v1/completions probe is unaffected (no chat template applied).
export K3_ENABLE_PARSERS="${K3_ENABLE_PARSERS:-1}"
# K3_PIECEWISE=1: CONFIRMED NOT VIABLE (job 589322) — K3 carries no
# @support_torch_compile, so mode=VLLM_COMPILE only warns and the model runs
# EAGER. Kept for the record; see README "Cudagraph experiments".
export K3_PIECEWISE="${K3_PIECEWISE:-0}"

# K3_BREAKABLE_PIECEWISE=1: the vLLM-intended piecewise path for models
# WITHOUT @support_torch_compile. config/vllm.py auto-enables
# VLLM_USE_BREAKABLE_CUDAGRAPH for KimiK3 BUT forces mode=NONE; the break
# primitive (@eager_break_during_capture on VkernelFusedExperts.apply, in
# sitecustomize.py) checks is_breakable_enabled() AT DECORATION TIME (Python
# startup, BEFORE vLLM's auto-default) — so the export must precede
# `exec python3` or the decoration is a silent no-op. cudagraph_mode=PIECEWISE
# (simple enum -> decode_mode=PIECEWISE -> breaks fire); wrapper install is
# independent of mode (gpu_model_runner.py:5444). MUST run TP8 PP1 (no PP
# boundary -> no recv_object to deadlock); the PP guard above warns. See
# README "Cudagraph experiments".
export K3_BREAKABLE_PIECEWISE="${K3_BREAKABLE_PIECEWISE:-0}"
if [ "${K3_BREAKABLE_PIECEWISE:-0}" = "1" ] && [ "${K3_PIECEWISE:-0}" = "1" ]; then
  echo "[rank ${RANK:-0}] WARN: K3_PIECEWISE=1 and K3_BREAKABLE_PIECEWISE=1 are" >&2
  echo "[rank ${RANK:-0}]       mutually exclusive (K3_PIECEWISE sets" >&2
  echo "[rank ${RANK:-0}]       VLLM_USE_BREAKABLE=0, which defeats breakable)." >&2
  echo "[rank ${RANK:-0}]       Resetting K3_PIECEWISE=0 for this run." >&2
  export K3_PIECEWISE=0
fi

# ------------------------------------------------------------------- otela ----
# x86_64 otela (sai dialect, same as the Clariden arm64 build): a multiaddr
# goes to --bootstrap.static (NOT --bootstrap.addr) and requires a cfg.yaml
# with loglevel/reachability/role/seed (verified on Clariden).
# appears to "hang" after the tracing-disabled line — it is actually working,
# just at a log level that goes silent during CRDT sync. Started INSIDE the
# container on the head so its HOME (EDF -> /capstor, not quota /users) is
# correct for its libp2p identity + badger DB.
export OTELA_DIR="${OTELA_DIR:-/capstor/scratch/cscs/xyao/opentela}"
export OTELA_BIN="${OTELA_BIN:-$OTELA_DIR/otela}"
# Direct-mesh bootstrap (an Alps OpenTela peer). No relay needed.
export OTELA_RELAY_ADDR="${OTELA_RELAY_ADDR:-/ip4/148.187.108.178/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9}"
export OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
export OTELA_TCP_PORT="${OTELA_TCP_PORT:-43905}"
export OTELA_UDP_PORT="${OTELA_UDP_PORT:-59820}"
export OTELA_API_PORT="${OTELA_API_PORT:-18094}"
OTELA_STEP_NAME="${OTELA_STEP_NAME:-otela-worker-$SLURM_JOB_ID}"
OTELA_SEED="${OTELA_SEED:-21}"
# Cold start from single-OST Lustre (stripe_count:1, ~147 s/shard x 96 shards)
# projects ~3h20m + init. 14400 s (4 h) gives margin under the 1-day wall.
# The REAL fix for the load bottleneck is a one-time Lustre restripe
# (lfs setstripe -s 4m -c -1) — see README. HEALTH_TIMEOUT only gates the
# pre-registration health/generation wait, not the load barrier itself.
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-14400}"
# Mandatory pre-registration correctness probe (deepseek-v4 lesson). Set
# GEN_PROBE=0 to skip (NOT recommended — registers on /health alone).
export GEN_PROBE="${GEN_PROBE:-1}"
# GEN_PROBE_TIMEOUT is the overall wall budget for the 6-prompt probe
# (the first request may trigger CUDA-graph capture on a freshly cold-
# started engine, so 600 s is safer than the old single-curl 300 s).
export GEN_PROBE_TIMEOUT="${GEN_PROBE_TIMEOUT:-600}"
# Correctness probe knobs (see gen_correctness.py). MIN_PASS=5 tolerates one
# soft-prompt fluke while requiring ALL crisp prompts (Paris, primes, 40 km/h)
# to pass — a broken AITER MXFP4 MoE path fails the crisp prompts at once.
export GEN_CORRECTNESS_MAX_TOKENS="${GEN_CORRECTNESS_MAX_TOKENS:-64}"
export GEN_CORRECTNESS_MIN_PASS="${GEN_CORRECTNESS_MIN_PASS:-5}"
export GEN_CORRECTNESS_PER_REQ_TIMEOUT="${GEN_CORRECTNESS_PER_REQ_TIMEOUT:-180}"
# GEN_CORRECTNESS_SMOKE=1: with --load-format dummy (random weights), the
# probe checks each prompt yields a NON-EMPTY continuation instead of factual
# correctness. Verifies the full pipeline (load -> JIT -> forward -> HTTP)
# without requiring real weights. Exit 0 iff all 6 prompts return tokens.
export GEN_CORRECTNESS_SMOKE="${GEN_CORRECTNESS_SMOKE:-0}"
# BENCH=1: after the correctness probe passes, run benchmark.py — a
# CONCURRENCY SWEEP of /v1/completions (ignore_eos, temperature 0) that
# reports aggregate + per-request decode throughput + latency per level.
# A warmup pass (C=4 N=8, discarded) runs first so the first measured level
# sees warm JIT kernels. BENCH_SPEC is a space-separated list of CONC:NUMREQ
# pairs (default 1:16 8:32 32:64 64:64). BENCH_OUT_TOK / BENCH_IN_TOK set the
# output and prompt lengths in tokens (defaults 256 / 32 — short prompt
# isolates decode throughput, matching BENCHMARK.md's table methodology).
# BENCH_TIMEOUT is the overall wall budget (default 1800 s). Results are
# written to $RUNDIR/benchmark_${SLURM_JOB_ID}.json. Runs BEFORE registration,
# so pair with OTELA_BIN=/bin/true for a pure benchmark (no mesh join).
export BENCH="${BENCH:-0}"
export BENCH_SPEC="${BENCH_SPEC:-1:16 8:32 32:64 64:64}"
export BENCH_OUT_TOK="${BENCH_OUT_TOK:-256}"
export BENCH_IN_TOK="${BENCH_IN_TOK:-32}"
export BENCH_TIMEOUT="${BENCH_TIMEOUT:-1800}"

export RUNDIR="$DEPLOY_DIR/run-$SLURM_JOB_ID"
export PATCH_MARKER="$K3/home/pylib/.k3_patch_done"
mkdir -p "$DEPLOY_DIR/logs" "$RUNDIR" "$DEPLOY_DIR/home/pylib" \
         "$DEPLOY_DIR/home/.cache/huggingface" \
         "$DEPLOY_DIR/home/.cache/triton"

# Per-job TMPDIR on scratch (see comment above).  Triton/LLVM writes
# intermediate .o/.ll during JIT; enroot writes UUID dirs here.  Being
# under $RUNDIR, it is cleaned at job end -- no accumulation across runs.
export TMPDIR="${TMPDIR:-$RUNDIR/tmp}"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR" 2>/dev/null || true

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export HEAD="${NODES[0]}"
export HEAD_IP="$(getent hosts "$HEAD" | awk '{print $1}' | head -1)"

echo "[$(date -Is)] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST nnodes=$NNODES"
echo "[$(date -Is)] head=$HEAD ($HEAD_IP)  tp=$TP_SIZE pp=$PP_SIZE  (world=$((TP_SIZE*PP_SIZE)) gpus=$((NNODES*GPUS_PER_NODE)) gpus_per_node=$GPUS_PER_NODE)"
echo "[$(date -Is)] model=$MODEL_PATH  served_as=$SERVED_MODEL_NAME  ctx=$CTX_LEN  max_seqs=$MAX_NUM_SEQS  gpu_mem=$GPU_MEM_UTIL"
echo "[$(date -Is)] edf=$OTELA_EDF_NAME  image=$IMAGE"
echo "[$(date -Is)] otela=$OTELA_BIN  bootstrap=$OTELA_RELAY_ADDR  serve_port=$SERVE_PORT  master_port=$MASTER_PORT"
echo "[$(date -Is)] gen_probe=$GEN_PROBE  smoke=$GEN_CORRECTNESS_SMOKE  min_pass=$GEN_CORRECTNESS_MIN_PASS  max_tokens=$GEN_CORRECTNESS_MAX_TOKENS  health_timeout=$HEALTH_TIMEOUT"
echo "[$(date -Is)] started=$(date --iso-8601=seconds)"

# ----------------------------------------------------------------- preflight --
[ -f "$IMAGE" ] || {
  echo "FATAL: vLLM image not found: $IMAGE" >&2
  echo "       It is a one-time import from docker://vllm/vllm-openai-rocm:kimi-k3" >&2
  echo "       onto a login node with egress, e.g.:" >&2
  echo "         enroot import -o $IMAGE docker://vllm/vllm-openai-rocm:kimi-k3" >&2
  exit 1
}
[ -d "$MODEL_PATH" ] || { echo "FATAL: weights not found: $MODEL_PATH" >&2; exit 1; }
[ -x "$OTELA_BIN" ] || {
  echo "FATAL: otela binary not found/executable: $OTELA_BIN" >&2
  echo "       Stage the verified sai-dialect x86_64 build, e.g.:" >&2
  echo "         mkdir -p $OTELA_DIR && cp <x86_64 otela> $OTELA_BIN && chmod +x $OTELA_BIN" >&2
  exit 1
}
[ -f "$SCRIPT_DIR/k3_patch.py" ] || {
  echo "FATAL: k3_patch.py not found next to this sbatch: $SCRIPT_DIR/k3_patch.py" >&2
  echo "       The recipe bundle (sbatch + kimi-k3-vllm.toml + k3_patch.py + sitecustomize.py + gen_correctness.py)." >&2
  exit 1
}
[ -f "$SCRIPT_DIR/sitecustomize.py" ] || {
  echo "FATAL: sitecustomize.py not found next to this sbatch: $SCRIPT_DIR/sitecustomize.py" >&2
  echo "       k3_patch.py installs it into the overlay so CPython auto-imports it" >&2
  echo "       at startup and forces vLLM's K3 SiTU MoE backend onto AITER on gfx942." >&2
  echo "       The recipe bundle (sbatch + kimi-k3-vllm.toml + k3_patch.py + sitecustomize.py + gen_correctness.py)." >&2
  exit 1
}
# gen_correctness.py resolves via directory walk-up to the shared canonical
# copy in meta/common/ (like the other recipes) — no per-recipe copy.
GEN_CORRECTNESS=""
_d="$SCRIPT_DIR"
while [ "$_d" != "/" ]; do
  if [ -f "$_d/gen_correctness.py" ]; then GEN_CORRECTNESS="$_d/gen_correctness.py"; break; fi
  _d="$(dirname "$_d")"
done
[ -n "$GEN_CORRECTNESS" ] || {
  echo "FATAL: gen_correctness.py not found walking up from $SCRIPT_DIR." >&2
  echo "       The canonical copy lives in meta/common/ in the repo root." >&2
  exit 1
}
[ -f "$SCRIPT_DIR/kimi-k3-vllm.toml" ] || {
  echo "FATAL: EDF not found next to this sbatch: $SCRIPT_DIR/kimi-k3-vllm.toml" >&2
  exit 1;
}
# A stale marker from a prior run would make siblings skip the wait while rank 0
# re-patches (k3_patch.py restores pristine + re-applies, so it is safe to always
# run on rank 0 — but the marker must not pre-exist or siblings proceed before
# the new patch lands).
rm -f "$PATCH_MARKER" 2>/dev/null || true

# ================================================================ engine ======
# Per-rank wrapper, heredoc'd into $RUNDIR so the sbatch stays self-contained
# apart from k3_patch.py + the EDF. It runs INSIDE the kimi-k3-vllm container
# (via srun --environment=$OTELA_EDF_NAME). Quoted heredoc: vars resolve at
# runtime inside the container from the srun-inherited + EDF-overlaid env.
cat > "$RUNDIR/engine.sh" <<'ENGINE'
#!/bin/bash
set -uo pipefail
RANK="${SLURM_NODEID:-0}"
NODE="$(hostname)"
echo "[rank $RANK] $(hostname) node-rank=$RANK head=$HEAD:$MASTER_PORT serve=$SERVE_PORT ctx=$CTX_LEN tp=$TP_SIZE pp=$PP_SIZE"

# --- gfx942 a16w4 flydsl patch (serialized across ranks) -------------------
# k3_patch.py copies the image's aiter tree to $K3/home/pylib, installs
# sitecustomize.py (forces vLLM's K3 MoE selector onto AITER on gfx942) and
# applies four gfx942 kernel patches. Rank 0 patches + touches a marker;
# siblings poll so they never import a half-patched aiter.
export K3
if [ "$RANK" = "0" ]; then
  echo "[rank 0] applying k3_patch.py (K3=$K3) ..."
  python3 "$SCRIPT_DIR/k3_patch.py" && touch "$PATCH_MARKER"
  echo "[rank 0] PATCH_DONE marker=$PATCH_MARKER rc=$?"
else
  echo "[rank $RANK] waiting for patch marker $PATCH_MARKER ..."
  for i in $(seq 1 1800); do
    [ -f "$PATCH_MARKER" ] && break
    sleep 2
  done
fi
[ -f "$PATCH_MARKER" ] || { echo "[rank $RANK] FATAL: patch marker missing after 1 h" >&2; exit 1; }
sleep 5   # full barrier past the marker write

# Patched aiter overlay must shadow the image's aiter; sitecustomize.py at
# its root is auto-imported by CPython at startup.
export PYTHONPATH="${K3}/home/pylib:${PYTHONPATH:-}"

# --- vLLM + AITER environment ----------------------------------------------
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
# gfx942: AITER MLA decode needs mla_gluon (gfx950-only); no valid TP gives
# a head multiple of 16 for persistent decode -> all TPs fail (job 586002).
# VLLM_ROCM_USE_AITER_MLA=0 falls back to TRITON_MLA; AITER MoE stays on for
# MXFP4 expert GEMMs.
export VLLM_ROCM_USE_AITER_MLA=0
# gfx942: custom skinny GEMM (wvSplitKrc_) device-asserts on M=10-class
# shapes and poisons the queue (/health 200 but forwards hang; jobs
# 586723/588129) -> torch/hipBLAS fallback.
export VLLM_ROCM_USE_SKINNY_GEMM=0
# K3_SKIP_FLYDSL_PATCHES=1: the TRITON/VKERNELS backends (sitecustomize.py)
# don't use the AITER FlyDSL kernels -> skip the slow MLIR patches (steps
# 3-6). K3_NO_ASYNC=1 disables hardcoded use_async_copy in fused_moe.py.
export K3_NO_ASYNC=1
export K3_SKIP_FLYDSL_PATCHES=1
# VKERNELS_DIR: vkernels build tree (libvkernels_hip.so), built by
# deployments/llm/beverin/vkernels/build_test_vkernels.sbatch. If absent,
# sitecustomize.py falls back to Triton experts (also works on gfx942).
export VKERNELS_DIR="${VKERNELS_DIR:-/capstor/scratch/cscs/xyao/vkernels}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Shared Triton cache on /capstor/scratch (unlimited; was /capstor/store
# where job 585851 hit the project quota mid-JIT and every worker segfaulted).
# All workers share compiled kernels; only the first run pays LLVM cost.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DEPLOY_DIR/triton_cache}"
mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true
# TMPDIR on scratch: the outer-shell export is LOST under srun
# --environment=EDF (EDF replaces, not merges, env). Without this,
# Triton/LLVM defaults to the container overlay /tmp -> "Disk quota
# exceeded" (jobs 585851, 585992). K3 == DEPLOY_DIR (/capstor/scratch).
export TMPDIR="${TMPDIR:-$K3/home/.tmp}"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR" 2>/dev/null || true
# Debug: log actual cache/scratch dirs at startup to diagnose JIT failures.
echo "[rank $RANK] dirs: TRITON_CACHE_DIR=$TRITON_CACHE_DIR TMPDIR=$TMPDIR HOME=$HOME"
# OMP_NUM_THREADS=1: single-threaded LLVM keeps per-worker JIT memory at
# ~5-8 GB (vs ~15 GB) — 4 workers/node must each stay under ~17 GB.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# K3_NO_CACHING=0 (default): keep the caching allocator — torch.cuda.
# memory_allocated() must work for vLLM's internal memory management (reads
# 0 with caching off). Per-worker overhead ~28.5 GiB is CUDA context +
# Mamba state + attention setup, NOT the allocator (job 585840). 4x99.5
# GiB = 398 < 478 GiB node RAM.
if [ "${K3_NO_CACHING:-0}" = "1" ]; then
  export PYTORCH_NO_CUDA_MEMORY_CACHING=1
  echo "[rank $RANK] PYTORCH_NO_CUDA_MEMORY_CACHING=1 — caching allocator disabled (saves ~10-15 GiB/worker on UMA)"
fi
# MALLOC_TRIM_THRESHOLD_: return freed heap to the OS aggressively — with
# 4 workers/node this prevents multi-GiB fragmented-but-unreserved heap.
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-65536}"
# K3 JIT stagger: one worker per node at a time (delay = local_rank * 60 s)
# so 4x simultaneous weight allocation doesn't OOM the node (jobs 584773+).
export K3_JIT_STAGGER="${K3_JIT_STAGGER:-1}"
export K3_JIT_STAGGER_DELAY="${K3_JIT_STAGGER_DELAY:-60}"
# Memory monitor (5 s free-RAM log) for OOM diagnosis.
( while true; do
    echo "$(date +%H:%M:%S) free=$(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null)kB comm=$(awk '/Committed_AS/{print $2}' /proc/meminfo 2>/dev/null)kB"
    sleep 5
  done ) &
_MEM_MON_PID=$!
trap 'kill $_MEM_MON_PID 2>/dev/null' EXIT

# --- RCCL networking ---------------------------------------------------------
# The EDF ships a fully-tuned RCCL-over-OFI/Slingshot config. The RCCL-native
# plugin (librccl-net-ofi.so, aws-ofi-nccl 1.20.0) INITIALISES cleanly
# (rccl-xport 601320/601321/601339) but the FIRST cross-node collective HANGS
# (>=2:47, with or without GDR) on this allocation — the EDF's com.hooks
# aws_ofi_nccl hook is CUDA-built (unusable on ROCm) and RCCL's native OFI
# data path does not complete on CSCS Alps CXI (a netstack gap, not an env
# tweak). Socket is therefore the DEFAULT; set K3_RCCL_SOCKET=0 to test a
# future OFI fix.
if [ "${K3_RCCL_SOCKET:-1}" = "0" ]; then
  echo "[rank $RANK] RCCL: EDF OFI/CXI over Slingshot (NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-?})"
else
  echo "[rank $RANK] RCCL: Socket fallback (K3_RCCL_SOCKET=${K3_RCCL_SOCKET:-1})"
  unset NCCL_NET_PLUGIN NCCL_NET
  # Fan Socket over ALL Slingshot NICs (gloo pinned to hsn0 below — no
  # prefix matching there).
  export NCCL_SOCKET_IFNAME="${K3_NIC:-hsn0}"
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-1}"
  # All-to-all tuning DEFAULT ON since run-601453: RCCL_ALL_TO_ALL_PIVOT_
  # ENABLE cut MoE a2a GPU time 15.50s -> 10.71s (-31%) + apply tail
  # 4.65s -> 1.40s; RCCL_CHANNEL_TUNING_ENABLE auto-tunes channels.
  # Disable either per-run via --export=...,RCCL_ALL_TO_ALL_PIVOT_ENABLE=0
  # (or ='' to drop it entirely).
  export RCCL_ALL_TO_ALL_PIVOT_ENABLE="${RCCL_ALL_TO_ALL_PIVOT_ENABLE:-1}"
  export RCCL_CHANNEL_TUNING_ENABLE="${RCCL_CHANNEL_TUNING_ENABLE:-1}"
  # Empty value = drop the knob (don't export an empty env var).
  [ -z "$RCCL_ALL_TO_ALL_PIVOT_ENABLE" ] && unset RCCL_ALL_TO_ALL_PIVOT_ENABLE
  [ -z "$RCCL_CHANNEL_TUNING_ENABLE" ] && unset RCCL_CHANNEL_TUNING_ENABLE
  echo "[rank $RANK] RCCL tuning: pivot=${RCCL_ALL_TO_ALL_PIVOT_ENABLE:-0} ch_tune=${RCCL_CHANNEL_TUNING_ENABLE:-0} nic=${NCCL_SOCKET_IFNAME}"
fi
# gloo is a SEPARATE CPU transport (MQ-broadcaster check + init barrier).
# Without GLOO_SOCKET_IFNAME it binds a non-routable iface -> cross-node
# recv timeout after 1800 s (job 600189, 1h52m lost). Pinned to hsn0 EVEN
# when NCCL fans over hsn*: gloo needs an EXACT iface name (job 601406:
# "Unable to find address for: hsn"). Override with K3_GLOO_NIC.
export GLOO_SOCKET_IFNAME="${K3_GLOO_NIC:-hsn0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

echo "[rank $RANK] launching vLLM (head=$([ "$RANK" = 0 ] && echo api_server || echo headless))"

VLLM_ARGS=(
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --tensor-parallel-size "$TP_SIZE"
  --pipeline-parallel-size "$PP_SIZE"
  --nnodes "$NNODES" --node-rank "$RANK"
  --master-addr "$HEAD" --master-port "$MASTER_PORT"
  --max-model-len "$CTX_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
)
# --enable-prefix-caching: gated by K3_PREFIX_CACHE (default 0 = OFF).
# Prefix caching is UNCONDITIONALLY BROKEN on this image for Kimi-K3:
# with --enable-prefix-caching, the KV cache manager selects
# HybridKVCacheCoordinator (kv_cache_coordinator.py:886, whenever
# len(kv_cache_groups) > 1). Kimi-K3 has >1 kv_cache_groups (Mamba +
# attention) but they all share ONE KV cache spec (Mamba recurrent state
# is managed separately, not via kv_cache_groups), so
# verify_and_split_kv_cache_groups collapses every group into a single
# SpecGroup and asserts 'HybridKVCacheCoordinator requires at least two
# attention groups' (kv_cache_coordinator.py:627). This fires for ANY
# config with prefix caching ON — eager (589044), PIECEWISE (589322), and
# the FULL-decode-cudagraph path alike — at scheduler init, BEFORE any
# cudagraph work. The commit that added --enable-prefix-caching (9f3177e)
# thus REGRESSED the only proven-working path (ENFORCE_EAGER=1, no prefix
# caching; 10+ jobs 588302-588856, up to 415 tok/s). K3_PREFIX_CACHE=0
# (-> KVCacheCoordinatorNoPrefixCache, no assertion) restores that
# known-good config. Set K3_PREFIX_CACHE=1 ONLY after the upstream
# HybridKVCacheCoordinator selection bug is fixed (it should key on
# distinct-spec count, not raw group count, and fall back to
# UnitaryKVCacheCoordinator when all groups share one spec).
export K3_PREFIX_CACHE="${K3_PREFIX_CACHE:-0}"
if [ "$K3_PREFIX_CACHE" = "1" ]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi
# --load-format: override weight loading (e.g. LOAD_FORMAT=dummy for fast
# iteration without real weights — a 5-min cold start instead of 3 h).
if [ -n "${LOAD_FORMAT:-}" ]; then
  VLLM_ARGS+=(--load-format "$LOAD_FORMAT")
fi
# --distributed-timeout-seconds: override the gloo/NCCL init+recv timeout
# (vLLM default 3600 s). Set low (e.g. 300) with K3_PIECEWISE=1 so a decode
# deadlock fails in 5 min, not 1 h.
if [ -n "${DISTRIBUTED_TIMEOUT_SECONDS:-}" ]; then
  VLLM_ARGS+=(--distributed-timeout-seconds "$DISTRIBUTED_TIMEOUT_SECONDS")
fi
# Reasoning + tool-call parsing (mirrors Clariden/JSC sglang --reasoning-parser
# kimi_k3 --tool-call-parser kimi_k3). This kimi-k3 vLLM fork uses the same
# --tool-call-parser spelling. Gated: set K3_ENABLE_PARSERS=0 if the image
# lacks the kimi_k3 parser (startup errors "invalid tool call parser" first).
if [ "${K3_ENABLE_PARSERS:-1}" = "1" ]; then
  VLLM_ARGS+=(
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
  )
fi
# K3_PIECEWISE=1: the "textbook fix" for the gfx942 PP=3 decode deadlock.
# vLLM auto-enables VLLM_USE_BREAKABLE_CUDAGRAPH=1 for
# KimiK3ForConditionalGeneration (config/vllm.py:1209), which forces
# compilation_config.mode=NONE. With mode=NONE and empty splitting_ops, the
# default FULL_AND_PIECEWISE cudagraph_mode degrades to FULL for decode — a
# whole-decode forward graph is captured that contains the gloo recv_object
# on the PP boundary, which cannot be replayed, deadlocking at the first real
# decode (jobs 588153/588156, BENCHMARK.md). K3_PIECEWISE=1 opts out of
# breakable (VLLM_USE_BREAKABLE_CUDAGRAPH=0) and sets mode=VLLM_COMPILE +
# cudagraph_mode=PIECEWISE so torch.compile splits the forward at attention
# ops, keeping the PP recv/send (which run eagerly in the worker driver,
# outside the model forward) out of the captured graph pieces. splitting_ops
# omitted from the JSON so vLLM auto-populates from _attention_ops
# (compilation.py:1144). CONFIRMED NOT VIABLE on this image (job 589322):
# vLLM accepts mode=VLLM_COMPILE + cudagraph_mode=PIECEWISE and
# auto-populates 15 splitting_ops, but KimiK3ForConditionalGeneration
# carries NO @support_torch_compile, so compilation_counter.num_models_seen
# is never incremented — vllm.py:2410 only WARNS ("torch.compile is turned
# on, but the model ... does not support it") and never wraps the model, so
# NO FX graph is produced to split. At init (gpu_model_runner.py:5442),
# is_breakable_cudagraph_enabled()=False (we set VLLM_USE_BREAKABLE=0)
# skips BreakableCUDAGraphWrapper, and PIECEWISE.has_full_cudagraphs()=False
# (PIECEWISE is a simple enum, not a FULL/PIECEWISE tuple, compilation.py:83)
# (continued) Net: no wrapper installed -> runs EAGER, identical to
# ENFORCE_EAGER=1. The REAL correctness gate is ENFORCE_EAGER=1 +
# K3_PREFIX_CACHE=0 (no wrapper -> no FULL decode graph -> no recv deadlock);
# add LOAD_FORMAT=dummy + DISTRIBUTED_TIMEOUT_SECONDS=300 for iteration.
if [ "${K3_PIECEWISE:-0}" = "1" ]; then
  # --enforce-eager would force mode=NONE, silently overriding the
  # experiment: warn + strip it so the test is valid.
  if [ "${ENFORCE_EAGER:-0}" = "1" ]; then
    echo "[rank $RANK] WARN: K3_PIECEWISE=1 and ENFORCE_EAGER=1 are mutually" >&2
    echo "[rank $RANK]       exclusive (--enforce-eager forces mode=NONE)." >&2
    echo "[rank $RANK]       Stripping --enforce-eager for this experiment." >&2
    VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS// --enforce-eager/}"
  fi
  export VLLM_USE_BREAKABLE_CUDAGRAPH=0
  VLLM_ARGS+=(
    --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"PIECEWISE"}'
  )
fi
if [ "${K3_BREAKABLE_PIECEWISE:-0}" = "1" ]; then
  # --enforce-eager would set cudagraph_mode=NONE -> no BreakableCUDAGraph
  # wrapper (gpu_model_runner.py:5444 requires mode != NONE). Strip it.
  if [ "${ENFORCE_EAGER:-0}" = "1" ]; then
    echo "[rank $RANK] WARN: K3_BREAKABLE_PIECEWISE=1 and ENFORCE_EAGER=1 are" >&2
    echo "[rank $RANK]       mutually exclusive. Stripping --enforce-eager." >&2
    VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS// --enforce-eager/}"
  fi
  # MUST be exported before `exec python3`: eager_break_during_capture
  # checks is_breakable_enabled() AT DECORATION TIME (sitecustomize import),
  # before vLLM's K3 auto-default — otherwise the decoration is a silent
  # no-op (equivalent to the config auto-default, but in time).
  export VLLM_USE_BREAKABLE_CUDAGRAPH=1
  # cudagraph_mode=PIECEWISE (simple enum -> decode_mode=PIECEWISE -> breaks
  # fire); non-empty splitting_ops guards the compilation.py downgrade
  # (PIECEWISE->NONE when splitting_ops empty).
  VLLM_ARGS+=(
    --compilation-config '{"cudagraph_mode":"PIECEWISE","splitting_ops":["vllm::unified_kv_cache_update"]}'
  )
fi
if [ "$RANK" = "0" ]; then
  VLLM_ARGS+=(--host 0.0.0.0 --port "$SERVE_PORT")
else
  VLLM_ARGS+=(--headless)
fi
ENGINE

# Issue #42 (KDA/MLA routing): VKERNELS_KDA=1 replaces the GPU-faulting
# AITER/Triton KDA kernels with the validated vk_hip_kda_delta_rule_fwd and
# re-enables the delta-rule layer that K3_DISABLE_KDA=1 drops (better output
# quality). VKERNELS_MLA=1 routes MLA to VkernelMLABackend (vk_hip_mla_fwd);
# VKERNELS_MLA_FORCE=1 is required for real traffic (vLLM calls forward_mqa
# bare, mla_attention.py:895); VKERNELS_MLA_VALIDATE=0 skips the CPU-oracle
# cross-check (libvkernels_c.so not built yet). Both OPT-IN until on-cluster
# validation (6/6 probes + KDA recall probe + rocprof evidence) — the
# default keeps the pre-#42 safe state (K3_DISABLE_KDA=1, all-MLA fallback).
# After validation: flip VKERNELS_KDA=1 and K3_DISABLE_KDA=0 (sitecustomize
# gives VKERNELS_KDA precedence). EDF replaces the inherited env, so the
# outer-shell values are embedded via the double-quoted heredoc below.
# STEP_PROFILE knobs (override via --export): SPEC=CONC:NUMREQ for the
# profiled load (8:16 = warmup + ~15 s steady decode, ~64 tok/req); BUDGET =
# wall budget for the profile srun (must exceed the eager warmup or the
# measured level is skipped — it reaches benchmark.py as BENCH_TIMEOUT, so
# the driver wraps it in timeout +300); PER_REQ_TIMEOUT (benchmark.py prefix
# — does NOT reach the remote task under --environment=EDF, harmless for
# C=8:16); VLLM_STEP_PROFILE_DUR (per-rank capture window, s);
# VLLM_STEP_PROFILE_RANKS (default 0,8,16 = one per PP stage at TP=8).
export STEP_PROFILE_SPEC="${STEP_PROFILE_SPEC:-8:16}"
export STEP_PROFILE_BUDGET="${STEP_PROFILE_BUDGET:-600}"
export STEP_PROFILE_PER_REQ_TIMEOUT="${STEP_PROFILE_PER_REQ_TIMEOUT:-300}"
export VLLM_STEP_PROFILE_DUR="${VLLM_STEP_PROFILE_DUR:-15}"
export VLLM_STEP_PROFILE_RANKS="${VLLM_STEP_PROFILE_RANKS:-0,8,16}"
if [ "${STEP_PROFILE:-0}" = "1" ]; then
  export VLLM_STEP_PROFILE_DIR="${VLLM_STEP_PROFILE_DIR:-$RUNDIR/step_profiles}"
  echo "[$(date -Is)] STEP_PROFILE on: dir=$VLLM_STEP_PROFILE_DIR ranks=${VLLM_STEP_PROFILE_RANKS} dur=${VLLM_STEP_PROFILE_DUR}s"
fi
cat >> "${RUNDIR}/engine.sh" <<EOFA
export VKERNELS_MLA="${VKERNELS_MLA:-0}"
export VKERNELS_MLA_FORCE="${VKERNELS_MLA_FORCE:-0}"
export VKERNELS_MLA_VALIDATE="${VKERNELS_MLA_VALIDATE:-0}"
export VKERNELS_KDA="${VKERNELS_KDA:-0}"
export K3_DISABLE_KDA="${K3_DISABLE_KDA:-1}"
export VLLM_STEP_PROFILE_DIR="${VLLM_STEP_PROFILE_DIR:-}"
export VLLM_STEP_PROFILE_DUR="${VLLM_STEP_PROFILE_DUR:-15}"
export VLLM_STEP_PROFILE_RANKS="${VLLM_STEP_PROFILE_RANKS:-0,8,16}"
EOFA

# Launch guard: the KDA exports MUST precede `exec` (earlier they were
# appended AFTER exec — dead code, so sitecustomize saw neither var -> the
# Triton KDA path faulted -> NaN garbage). Assert a VALID state before exec
# so a future silent drop fails at launch.
cat >> "${RUNDIR}/engine.sh" <<'ENGINE2'
if [ "${VKERNELS_KDA:-0}" != "1" ] && [ "${K3_DISABLE_KDA:-1}" != "1" ]; then
  echo "FATAL: KDA mis-configured (VKERNELS_KDA='${VKERNELS_KDA:-0}' K3_DISABLE_KDA='${K3_DISABLE_KDA:-1}');" \
       "refusing to launch -- set VKERNELS_KDA=1 (validated HIP) or K3_DISABLE_KDA=1 (MLA-only). See issue #42." >&2
  exit 1
fi
exec python3 -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" $VLLM_EXTRA_ARGS
ENGINE2
chmod +x "$RUNDIR/engine.sh"

# ============================================================ otela worker ====
# Runs INSIDE the container on the HEAD node only (one peer: the DNT lists
# one row per peer; per-node workers would over-register the single head
# endpoint). The container shares the host netns -> otela on the head
# reaches vLLM on localhost:$SERVE_PORT.
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
  --label "framework=vllm" \\
  --label "launched_by=\${USER:-unknown}"
OTELA
chmod +x "$RUNDIR/otela.sh"

# ------------------------------------------------------------- lifecycle -----
# otela leaves the registry cleanly only on SIGTERM (AnnounceLeave); SIGKILL
# leaves a stale connected:true row round-robining traffic into a dead
# endpoint. --signal=B:TERM@120 lets this trap scancel the otela STEP with
# TERM and wait for it to exit.
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
  # 20 s: --signal=B:TERM@120 covers wall-time expiry, but on the `scancel`
  # path the only grace is KillWait (default 30 s). otela needs ~10 s to shut
  # down (AnnounceLeave -> 5 s CRDT drain -> srv.Shutdown), so 20 s leaves
  # margin and still finishes before SIGKILL.
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

# Registration gate: wait for /health AND a real token. The wait + probe run
# in the BATCH shell (host netns == container netns, so $HEAD:$SERVE_PORT is
# reachable), then the otela worker is launched as its own step inside the
# container. The probe is MANDATORY
# generated a token on gfx942, and /health returning 200 does NOT prove the
# engine can run a forward pass (deepseek-v4 registered on /health and 502'd
# every request).
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
    # Run INSIDE the kimi-k3-vllm container on the head (same
    # --overlap --environment=$OTELA_EDF_NAME launch as the otela worker): the
    # host's python3 is 3.6 but the container has 3.12 + urllib, and the
    # container shares the host netns so 127.0.0.1:$SERVE_PORT reaches vLLM
    # (bound on 0.0.0.0). gen_correctness.py mirrors the Clariden servekit
    # bench (same six /v1/completions greedy prompts, temperature 0,
    # max_tokens 64) but ADDS automated factual-substring checks (servekit only
    # captures outputs for human comparison). Exit 0 == all CRISP prompts
    # correct AND >= MIN_PASS of 6 correct; full per-prompt report in
    # $PROBE_OUT. The `timeout` backstop guards a hung request past per-req.
    GEN_CORRECTNESS_TIMEOUT="$GEN_PROBE_TIMEOUT" \
    GEN_CORRECTNESS_PER_REQ_TIMEOUT="$GEN_CORRECTNESS_PER_REQ_TIMEOUT" \
    GEN_CORRECTNESS_SMOKE="$GEN_CORRECTNESS_SMOKE" \
      timeout "$((GEN_PROBE_TIMEOUT + 300))" \
      srun --jobid="$SLURM_JOB_ID" --overlap --gres=none --nodes=1 --ntasks=1 \
           -w "$HEAD" --environment="$OTELA_EDF_NAME" \
           python3 "$GEN_CORRECTNESS" \
               "http://127.0.0.1:$SERVE_PORT" "$SERVED_MODEL_NAME" \
               "$PROBE_OUT" "$GEN_CORRECTNESS_MAX_TOKENS" "$GEN_CORRECTNESS_MIN_PASS"
    rc=$?
    if [ "$rc" = 0 ]; then
      echo "[$(date -Is)] GEN_PROBE_OK rc=$rc (report in $PROBE_OUT)"
    else
      echo "[$(date -Is)] GEN_PROBE_FAILED rc=$rc; NOT registering (report in $PROBE_OUT)" >&2
      echo "[$(date -Is)]       Inspect $PROBE_OUT + the engine log, then scancel if needed." >&2
      echo "[$(date -Is)]       A non-0 rc with pass>=5 usually means a CRISP prompt failed" >&2
      echo "[$(date -Is)]       (Paris / primes / 40 km/h) — treat as a real MoE-corruption signal." >&2
      exit 1
    fi
  fi

  # Per-step torch.profiler pass (STEP_PROFILE=1). Drives a SHORT C=8 load
  # via benchmark.py, which writes $VLLM_STEP_PROFILE_DIR/.go right before
  # its first MEASURED level (warmup excluded). The in-worker profiler
  # threads (sitecustomize.py, ranks $VLLM_STEP_PROFILE_RANKS) then capture
  # a $VLLM_STEP_PROFILE_DUR-second window of steady-state decode and export
  # per-rank chrome traces. Generous per-req timeout avoids the C>1 client
  # pool artifact (this measures the ENGINE, not the harness limits).
  if [ "${STEP_PROFILE:-0}" = "1" ]; then
    mkdir -p "$VLLM_STEP_PROFILE_DIR"
    echo "[$(date -Is)] step profile: driving spec=$STEP_PROFILE_SPEC (per_req=${STEP_PROFILE_PER_REQ_TIMEOUT}s, budget=${STEP_PROFILE_BUDGET}s) -> traces in $VLLM_STEP_PROFILE_DIR"
    BENCH_PER_REQ_TIMEOUT="$STEP_PROFILE_PER_REQ_TIMEOUT" \
    BENCH_TIMEOUT="$STEP_PROFILE_BUDGET" \
      timeout "$((STEP_PROFILE_BUDGET + 300))" \
      srun --jobid="$SLURM_JOB_ID" --overlap --gres=none --nodes=1 \
           --ntasks=1 -w "$HEAD" --environment="$OTELA_EDF_NAME" \
      python3 "$SCRIPT_DIR/benchmark.py" \
          "http://127.0.0.1:$SERVE_PORT" "$SERVED_MODEL_NAME" \
          "$VLLM_STEP_PROFILE_DIR/bench_profile.json" \
          "$STEP_PROFILE_SPEC" "$BENCH_OUT_TOK" "$BENCH_IN_TOK"
    echo "[$(date -Is)] step profile: waiting ${VLLM_STEP_PROFILE_DUR}s for trace export, then clearing .go"
    sleep "$((VLLM_STEP_PROFILE_DUR + 10))"
    rm -f "$VLLM_STEP_PROFILE_DIR/.go"
    echo "[$(date -Is)] STEP_PROFILE done (see $VLLM_STEP_PROFILE_DIR/step_profile_rank*.json)"
  fi

  # Concurrent throughput sweep (BENCH=1). Runs in the same container as the
  # probe (aiohttp is bundled by vLLM); host netns is shared so 127.0.0.1
  # reaches the engine. Warmup first, then one JSON line per concurrency
  # level to stdout; full summary in $RUNDIR/benchmark_${SLURM_JOB_ID}.json.
  # Runs BEFORE registration so a pure benchmark uses OTELA_BIN=/bin/true.
  if [ "$BENCH" = "1" ]; then
    BENCH_OUT="$RUNDIR/benchmark_${SLURM_JOB_ID}.json"
    echo "[$(date -Is)] benchmark sweep (model=$SERVED_MODEL_NAME, spec=$BENCH_SPEC, out_tok=$BENCH_OUT_TOK, in_tok=$BENCH_IN_TOK, budget=${BENCH_TIMEOUT}s) ..."
    BENCH_TIMEOUT="$BENCH_TIMEOUT" \
    BENCH_PER_REQ_TIMEOUT="${GEN_CORRECTNESS_PER_REQ_TIMEOUT:-180}" \
      timeout "$((BENCH_TIMEOUT + 300))" \
      srun --jobid="$SLURM_JOB_ID" --overlap --gres=none --nodes=1 --ntasks=1 \
           -w "$HEAD" --environment="$OTELA_EDF_NAME" \
           python3 "$SCRIPT_DIR/benchmark.py" \
               "http://127.0.0.1:$SERVE_PORT" "$SERVED_MODEL_NAME" \
               "$BENCH_OUT" "$BENCH_SPEC" "$BENCH_OUT_TOK" "$BENCH_IN_TOK"
    brc=$?
    echo "[$(date -Is)] BENCH rc=$brc (report in $BENCH_OUT)"
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
  echo "FRAMEWORK=vllm"
  echo "TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE CTX_LEN=$CTX_LEN"
  echo "BOOTSTRAP=$OTELA_RELAY_ADDR"
  echo "SUBMITTED_AT=$(date --iso-8601=seconds)"
  for i in "${!NODES[@]}"; do
    echo "ENDPOINT_${i}=${NODES[$i]}:$SERVE_PORT"
  done
} > "$DEPLOY_DIR/last_service.env"

# ------------------------------------------------------------- run engine ----
# One engine rank per node, inside the container, in the foreground so the job
# lifetime == engine lifetime. srun inherits the exported vars above (HEAD,
# MASTER_PORT, SERVE_PORT, TP_SIZE, …) and --environment overlays the EDF's
# [env] (HOME on /capstor, HF_HOME, TRITON_CACHE_DIR, the aws_ofi_nccl hook …).
# If a var is mysteriously empty on a rank, check:
#   srun -n1 --environment=$OTELA_EDF_NAME env | grep NAME
srun --nodes="$NNODES" --ntasks-per-node=1 --environment="$OTELA_EDF_NAME" bash "$RUNDIR/engine.sh"
SRUN_EXIT=$?
echo "[$(date -Is)] srun_exited code=${SRUN_EXIT}"

exit "$SRUN_EXIT"
