#!/bin/bash
#SBATCH --job-name=k3-vllm-beverin
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
# MI300A APU (UMA): 4 GPUs/node, ~501 GiB system RAM.  6 nodes x 4 GPUs
# = 24 GPUs (TP8 PP3).  Each GPU holds ~71 GiB dummy weights (896/8=112
# experts), leaving ~57 GiB HBM free for activations + JIT.  Per-worker
# system RAM ~99.5 GiB (71 weights + ~28.5 overhead); 4 workers/node =>
# 398 < 478 GiB available (80 GiB margin).  K3_NO_CACHING=0 (default):
# keep PyTorch caching allocator so torch.cuda.memory_allocated() works
# correctly for vLLM's internal memory management.
# AITER MLA constraint (job 585845): num_attention_heads=96, heads/GPU
# must be in [1,15] or mult of 16.  TP=8 -> 12 heads OK; TP=4 -> 24 INVALID.
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
# B:TERM@120 signals the BATCH SHELL (B:) 120 s before wall-time, giving the
# stop_otela trap below a window to scancel the otela step with TERM. Without
# B: the signal goes to the job steps, the batch-shell trap never runs, and
# the only grace is KillWait (30 s) — less than otela's ~10 s graceful leave
# (AnnounceLeave -> 5 s CRDT drain -> srv.Shutdown), so no LEFT is published
# and the head keeps a `connected: true` row round-robining traffic into a
# dead endpoint.
#SBATCH --signal=B:TERM@120
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/logs/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/logs/%x-%j.out
#
# =============================================================================
# Serve Kimi-K3 (2.8 T-param hybrid MoE, mxfp4) on CSCS Beverin (4x MI300A /
# node, x86_64, gfx942, Slingshot fabric) with vLLM, and register it on
# OpenTela — one self-contained sbatch + a kimi-k3-vllm.toml EDF + k3_patch.py
# + sitecustomize.py + gen_correctness.py siblings in this directory.
#
# Submit (defaults: 4 nodes, TP8 x PP2 = 16 GPUs, served as moonshotai/Kimi-K3,
#                          max-model-len 131072 / 128K):
#   sbatch serve_kimi_k3_otela_beverin.sbatch
#
# Override context length / shape (must satisfy TP*PP == nodes*GPUS_PER_NODE):
#   sbatch --export=ALL,CTX_LEN=1048576 serve_kimi_k3_otela_beverin.sbatch
#   sbatch --nodes=2 --export=ALL,TP_SIZE=4,PP_SIZE=2 serve_kimi_k3_otela_beverin.sbatch
#
# What it does, in order:
#   1. preflights the EDF/image, weights, otela binary, and k3_patch.py sibling
#      (prints exact staging commands on failure)
#   2. srun's one vLLM engine rank per node INSIDE the kimi-k3-vllm container
#      (enroot/EDF), forming a single distributed TP x PP engine. Rank 0 (head)
#      serves HTTP on 0.0.0.0:$SERVE_PORT; ranks 1..N run --headless (P2P only).
#      Rank 0 applies k3_patch.py (gfx942 a16w4 flydsl + installs
#      sitecustomize.py, which forces vLLM's MoE backend selector onto AITER
#      on gfx942) before launch; siblings wait on a marker.
#   3. once the head's /health answers, runs a MANDATORY correctness probe
#      BEFORE registration — /health alone is not enough (deepseek-v4 lesson),
#      and K3 has never generated a token on this GPU family. gen_correctness.py
#      sends the same six greedy /v1/completions prompts the Clariden servekit
#      bench uses (temperature 0, max_tokens 64) and checks each answer for
#      its expected factual substring (Paris; the prime sequence; 60 km/1.5 h
#      = 40 km/h; Rayleigh; a fibonacci function body; entropy). Only if all
#      CRISP prompts pass AND >= MIN_PASS (default 5) of 6 are correct does it
#      start ONE otela WORKER — also inside the container, on the head via
#      srun --overlap --gres=none — which registers the `llm` service.
#   4. on job end / scancel, SIGTERM (never KILL) the otela step so it
#      announces LEFT cleanly.
#
# -----------------------------------------------------------------------------
# THE TOPOLOGY (JSC/engine shape, Beverin network)
# -----------------------------------------------------------------------------
# Kimi-K3 is ~1.5 TB mxfp4 — it does NOT fit on one 4x137 GB MI300A node, so
# this is NOT the Beverin per-node-independent-replicas pattern (GLM-4.7-Flash).
# It is the JSC pattern: ONE distributed vLLM engine across all nodes, ONE
# otela worker on the head registering the single head HTTP endpoint.
#
#   beverin compute (host netns; enroot container shares it)
#     rank 0 (HEAD)  -> vllm api_server (HTTP on 0.0.0.0:$SERVE_PORT) + otela worker
#     rank 1..N      -> vllm --headless (no HTTP; pipeline P2P to neighbours)
#
# Default TP8 x PP2: each tensor-parallel group of 8 spans 2 nodes, so MoE
# allreduce DOES cross Slingshot — but over RCCL Socket transport (hsn0), which
# is the verified bring-up path (k3-eng8/k3-eng11 reached MoE backend
# selection — ROCM_AITER_MLA + AITER_MXFP4_BF16 via patched flydsl +
# sitecustomize.py, PP2xTP8=16 workers; k3-eng8 loaded shards to 16% @ ~147
# s/shard pre-restripe before being cancelled). A token has not
# yet been confirmed on this GPU family; the mandatory generation probe gates
# registration on it. TP4 x PP4 (4 nodes) keeps TP all-reduce intra-node (4
# GPUs) but stages two pipeline layers per node — unverified, see README.
#
# -----------------------------------------------------------------------------
# NO RELAY (direct mesh, like Clariden)
# -----------------------------------------------------------------------------
# Beverin compute nodes have full outbound internet and reach the Alps OpenTela
# bootstrap /ip4/148.187.108.178/... directly (verified). The worker dials the
# bootstrap directly — no us-managed login-node relay (unlike JSC).
#
# -----------------------------------------------------------------------------
# CONTAINER + RCCL ON SLINGSHOT (the Beverin-specific part)
# -----------------------------------------------------------------------------
# Image: vllm/vllm-openai-rocm:kimi-k3 (vLLM 0.1.dev19253+g5f76ae224.rocm723),
# imported to a local .sqsh. The EDF (kimi-k3-vllm.toml) carries
# com.hooks.aws_ofi_nccl.enabled=true, which LD_PRELOADs a CUDA-built
# libnccl-net.so. That plugin is unusable on ROCm (no libcudart), so engine.sh
# neutralises it (unset NCCL_NET_PLUGIN/NCCL_NET, NCCL_SOCKET_IFNAME=hsn0,
# NCCL_IB_DISABLE=1) and lets RCCL use its built-in Socket transport over
# Slingshot. TODO: a HIP-aware OFI plugin for real CXI throughput.
# =============================================================================

set -uo pipefail

# ----------------------------------------------------- script + deploy dirs --
# SCRIPT_DIR is where this sbatch + kimi-k3-vllm.toml + k3_patch.py live, so
# the EDF and patch are found regardless of where the checkout is copied. Slurm
# runs the sbatch from a spool copy (/var/spool/...), so ${BASH_SOURCE[0]} is
# NOT the submit dir — SLURM_SUBMIT_DIR is.
export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"

# Job temp is set to $RUNDIR/tmp (per-job, on scratch, auto-cleaned) AFTER
# RUNDIR is defined below. Job 583899 died when TMPDIR defaulted to
# $HOME/.tmp (22 GiB stale enroot UUID dirs); job 585851 hit
# "Disk quota exceeded" (LLVM ERROR on Triton JIT) when TMPDIR was on
# /capstor/store (project 7659 ~90% full). Scratch (project 2161) is
# unlimited with 37 PB free -- see TRITON_CACHE_DIR below for same fix.
OTELA_EDF_NAME="${OTELA_EDF_NAME:-kimi-k3-vllm}"

# ---------------------------------------------------------------- deployment --
export DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin}"
# k3_patch.py reads $K3 and writes its aiter overlay to $K3/home/pylib; the EDF
# sets HOME=$DEPLOY_DIR/home, so $K3/home == container HOME. Keep them in lockstep.
export K3="${K3:-$DEPLOY_DIR}"
export IMAGE="${IMAGE:-/capstor/scratch/cscs/xyao/.edf_imagestore/vllm+vllm-openai-rocm+kimi-k3.x86_64.sqsh}"
export MODEL_PATH="${MODEL_PATH:-/capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K3}"
# Served under the canonical HF id so the peer coexists with the Clariden
# sglang K3 registration (both advertise model=moonshotai/Kimi-K3; the mesh
# round-robins them as additive capacity). Override to `kimi-k3` to match the
# original bring-up probe script.
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-SwissAI-Research/moonshot/kimi-k3-rocm}"

# ------------------------------------------------------------------- shape ----
# TP8 x PP3 (default): world = TP*PP = 24 must equal nodes*GPUS_PER_NODE.
# MI300A APU: 4 workers/node * ~99.5 GiB/worker (dummy weights) = 398 < 478
# GiB node RAM available.  TP=8 spans 2 nodes (4 GPUs/node * 2 = 8) for
# allreduce over Slingshot Socket — the verified bring-up path (k3-eng8/11).
# PP=3 stages 93/3 = 31 layers each.  896/8 = 112 experts/GPU.
#
# AITER MLA constraint (job 585845): num_attention_heads=96, so heads/GPU
# must be in [1,15] or a multiple of 16.  TP=4 gives 96/4=24 heads/GPU —
# INVALID (neither range).  TP=8 gives 96/8=12 heads/GPU — OK.
#
# GPU HBM: previous TP8 x PP2 on 16 GPUs (8 nodes x 2 GPUs) held ~106.5
# GiB weights/GPU, leaving only ~21 GiB -> HSA OOM during _dummy_run
# (job 585844).  With 24 GPUs (6 nodes x 4), each GPU holds ~71 GiB,
# leaving ~57 GiB free for activations + JIT.
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
# Request-length bound. Kimi-K3 supports up to max_position_embeddings=1048576
# (1M). The KV pool is sized by --gpu-memory-utilization, NOT this value (MLA +
# KDA hybrid attention: ~107 KB/token), so even large --max-model-len admits a
# healthy pool. Default 131072 (128K) per the bring-up target; raise toward
# 1048576 once a cold start + generation are confirmed fast.
export CTX_LEN="${CTX_LEN:-131072}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
# MI300A is an APU: HBM is shared between CPU and 4 GPU complexes.
# 0.9 leaves only ~40 GiB for OS → OOM kill. 0.8 leaves ~91 GiB.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
# vLLM distributed init store (rank discovery) + head HTTP port.
export MASTER_PORT="${MASTER_PORT:-6379}"
export SERVE_PORT="${SERVE_PORT:-8080}"
# Passthrough for experimental vLLM flags (e.g. ENFORCE_EAGER=1 adds
# --enforce-eager) without forking this script.
export K3_NIC="${K3_NIC:-hsn0}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
if [ "${ENFORCE_EAGER:-0}" = "1" ]; then
  export VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS --enforce-eager"
fi

# Kimi-K3 is a hybrid reasoning model: its chat template emits a <think>...</think>
# (reasoning) block and supports OpenAI-style tool calls. vLLM exposes
# reasoning_content + tool_calls on /v1/chat/completions ONLY with
# --enable-auto-tool-choice + --tool-call-parser + --reasoning-parser. This
# kimi-k3 vLLM fork spells the tool flag --tool-call-parser, same as Clariden/
# JSC (sglang). The `kimi_k3` parser ships in the custom vllm-openai-rocm:kimi-k3
# image; if an image bump drops it, startup errors "invalid tool call parser:
# kimi_k3" first -- set K3_ENABLE_PARSERS=0 to recover without editing this
# file. The /v1/completions probe below is unaffected (no chat template / tools
# applied), so parsing stays out of the mandatory correctness gate.
export K3_ENABLE_PARSERS="${K3_ENABLE_PARSERS:-1}"
# K3_PIECEWISE=1: was the "textbook fix" hypothesis for the gfx942 PP=3
# decode deadlock (VLLM_USE_BREAKABLE_CUDAGRAPH=0 + --compilation-config
# mode=VLLM_COMPILE/cudagraph_mode=PIECEWISE, so torch.compile would split
# the forward at attention ops and keep the gloo recv_object out of the
# captured graph pieces). CONFIRMED NOT VIABLE (job 589322): Kimi-K3
# carries NO @support_torch_compile, so mode=VLLM_COMPILE only warns
# (vllm.py:2410) and at init (gpu_model_runner.py:5442)
# is_breakable=False + PIECEWISE.has_full_cudagraphs()=False installs NO
# wrapper — the model runs EAGER, identical to ENFORCE_EAGER=1. It does
# NOT validate the PIECEWISE cudagraph hypothesis. Real PIECEWISE needs
# upstream @support_torch_compile. Default 0 = unchanged (ENFORCE_EAGER=1
# is the proven path per BENCHMARK.md).
export K3_PIECEWISE="${K3_PIECEWISE:-0}"

# ------------------------------------------------------------------- otela ----
# x86_64 build, sai-v0.0.6 (same dialect as the Clariden arm64 build). A
# multiaddr goes to --bootstrap.static (NOT --bootstrap.addr, which expects an
# HTTP bootstraps-list URL). Requires a proper cfg.yaml with loglevel/reach
# ability/role/seed (verified on Clariden job 3000965); without it otela
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
  echo "       Stage an x86_64 (sai-v0.0.6) build, e.g.:" >&2
  echo "         mkdir -p $OTELA_DIR && cp <x86_64 otela> $OTELA_BIN && chmod +x $OTELA_BIN" >&2
  exit 1
}
[ -f "$SCRIPT_DIR/k3_patch.py" ] || {
  echo "FATAL: k3_patch.py not found next to this sbatch: $SCRIPT_DIR/k3_patch.py" >&2
  echo "       The recipe is a 5-file bundle (sbatch + kimi-k3-vllm.toml + k3_patch.py + sitecustomize.py + gen_correctness.py)." >&2
  exit 1
}
[ -f "$SCRIPT_DIR/sitecustomize.py" ] || {
  echo "FATAL: sitecustomize.py not found next to this sbatch: $SCRIPT_DIR/sitecustomize.py" >&2
  echo "       k3_patch.py installs it into the overlay so CPython auto-imports it" >&2
  echo "       at startup and forces vLLM's K3 SiTU MoE backend onto AITER on gfx942." >&2
  echo "       The recipe is a 5-file bundle (sbatch + kimi-k3-vllm.toml + k3_patch.py + sitecustomize.py + gen_correctness.py)." >&2
  exit 1
}
[ -f "$SCRIPT_DIR/gen_correctness.py" ] || {
  echo "FATAL: gen_correctness.py not found next to this sbatch: $SCRIPT_DIR/gen_correctness.py" >&2
  echo "       The recipe is a 5-file bundle (sbatch + kimi-k3-vllm.toml + k3_patch.py + sitecustomize.py + gen_correctness.py)." >&2
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
# sitecustomize.py at the overlay root (forces vLLM's K3 SiTU MoE backend
# selector onto AITER on gfx942 — without it on_gfx950() is False on real
# MI300A and the engine hits NotImplementedError before any shard loads), and
# applies four gfx942 kernel patches (LDS fill, K16 MFMA split, software
# fp4->bf16 dequant, async off). Idempotent + strictly asserts expected source
# text. Rank 0 patches and touches a marker; siblings poll so they do not
# import a half-patched aiter.
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

# The patched aiter overlay must take precedence over the image's aiter, and
# sitecustomize.py at its root is auto-imported by CPython at startup (it
# forces vLLM's K3 SiTU MoE backend selector onto AITER on gfx942).
export PYTHONPATH="${K3}/home/pylib:${PYTHONPATH:-}"

# --- vLLM + AITER environment ----------------------------------------------
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
# K3 on MI300A (gfx942): AITER MLA decode uses mla_gluon, a gfx950
# (CDNA4) hardware feature.  No valid TP for 24 GPUs produces a head
# count that is a multiple of 16 (persistent decode), so ALL valid
# TPs select the Gluon path and fail with: mla_gluon requires gfx950
# (job 586002).  Disable AITER MLA and fall back to TRITON_MLA (pure
# Triton, no arch dependency).  AITER MoE (VLLM_ROCM_USE_AITER_MOE=1
# above) remains enabled for MXFP4 expert GEMMs.
export VLLM_ROCM_USE_AITER_MLA=0
# gfx942: vLLM custom ROCm skinny GEMM (wvSplitKrc_ in csrc/rocm/skinny_gemms.hip)
# fires Device-side assertion on M=10,7168x1536-class shapes and poisons the
# device queue (async server: /health stays 200 but all forwards hang - jobs
# 586723, 588129).  Fall back to torch/hipBLAS for those GEMMs.
export VLLM_ROCM_USE_SKINNY_GEMM=0
# K3_SKIP_FLYDSL_PATCHES=1: the TRITON_UNFUSED MoE backend (sitecustomize.py)
# does not use the AITER FlyDSL kernels, so skip the time-consuming
# MLIR/kernel patches in k3_patch.py (steps 3-6). K3_NO_ASYNC=1 disables
# hardcoded use_async_copy in fused_moe.py on gfx942.
export K3_NO_ASYNC=1
export K3_SKIP_FLYDSL_PATCHES=1
# VKERNELS_DIR: path to the vkernels build tree (for libvkernels_hip.so).
# The VkernelFusedExperts backend (sitecustomize.py) loads the C ABI
# shared library from $VKERNELS_DIR/build/hip/src/c/libvkernels_hip.so.
# Built by deployments/llm/beverin/vkernels/build_test_vkernels.sbatch.
# If not found, sitecustomize.py falls back to Triton
# (UnfusedOAITritonExperts) which also works on gfx942.
export VKERNELS_DIR="${VKERNELS_DIR:-/capstor/scratch/cscs/xyao/vkernels}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Shared Triton cache on /capstor/scratch (project 2161, unlimited quota,
# 37 PB free).  Was on /capstor/store (project 7659 ~90% full) -- job
# 585851 hit "LLVM ERROR: IO failure on output stream: Disk quota
# exceeded" when 24 workers JIT-compiled batched_gemm_a16wfp4_ and the
# clang++ output hit the project block/inode quota, crashing every
# worker (segfault).  All workers load compiled kernels from the same
# dir here so only the first run pays the LLVM cost.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DEPLOY_DIR/triton_cache}"
mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true
# TMPDIR on scratch (project 2161, unlimited).  AITER_JIT_DIR intentionally NOTs
# outer-shell `export TMPDIR` is lost when srun uses
# --environment=$OTELA_EDF_NAME (the EDF replaces, not merges, the
# inherited env).  Without this, Triton/LLVM defaults to /tmp on the
# enroot overlay (tmpfs, competes with model memory) causing
# "LLVM ERROR: IO failure on output stream: Disk quota exceeded"
# (jobs 585851, 585992).  K3 == DEPLOY_DIR (on /capstor/scratch).
export TMPDIR="${TMPDIR:-$K3/home/.tmp}"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR" 2>/dev/null || true
# Debug: log actual cache/scratch dirs at startup to diagnose JIT failures.
echo "[rank $RANK] dirs: TRITON_CACHE_DIR=$TRITON_CACHE_DIR TMPDIR=$TMPDIR HOME=$HOME"
# Limit OMP threads per worker to reduce CPU memory during JIT compilation.
# With 4 workers/node and ~69 GB system RAM free after load_model, each
# worker's LLVM/Triton compilation must stay under ~17 GB.  OMP_NUM_THREADS=1
# (single-threaded LLVM) keeps per-worker JIT memory at ~5-8 GB instead of
# ~15 GB, giving a comfortable margin for 4× simultaneous compilation during
# profile_run().
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# K3_NO_CACHING=0 (default): keep the PyTorch CUDA caching allocator so
# torch.cuda.memory_allocated() works correctly on ROCm (with caching
# disabled it returns 0, causing vLLM to report "Model loading took 0.0
# GiB" and potentially malfunctioning in internal memory management).
# The ~28.5 GiB/worker overhead is from CUDA context, Mamba state, and
# attention setup — NOT from the caching allocator (confirmed in job
# 585840: per-worker identical with/without caching).  With TP4 PP6,
# per-worker is ~99.5 GiB (71 weights + 28.5 overhead); 4x99.5 = 398 < 478.
# Set K3_NO_CACHING=1 only if system RAM OOM recurs and you accept the
# broken memory_allocated() tradeoff.
if [ "${K3_NO_CACHING:-0}" = "1" ]; then
  export PYTORCH_NO_CUDA_MEMORY_CACHING=1
  echo "[rank $RANK] PYTORCH_NO_CUDA_MEMORY_CACHING=1 — caching allocator disabled (saves ~10-15 GiB/worker on UMA)"
fi
# MALLOC_TRIM_THRESHOLD_: cause glibc malloc to return freed heap to the OS
# more aggressively (default 128 KiB).  With 4 workers/node each freeing
# Python objects after load_model, this prevents the heap from retaining
# several GiB of fragmented but unreserved address space.
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-65536}"
# K3 JIT stagger: only 1 worker per node starts at a time (delay =
# local_rank * K3_JIT_STAGGER_DELAY) so weight allocation (~108 GB/worker,
# ~26s) happens one at a time instead of 4× simultaneously OOM'ing the
# node (jobs 584773/584789/584984).  60s is enough for each worker's
# load_model to finish before the next starts.
export K3_JIT_STAGGER="${K3_JIT_STAGGER:-1}"
export K3_JIT_STAGGER_DELAY="${K3_JIT_STAGGER_DELAY:-60}"
# Memory monitor: log free -m every 5s to /tmp/k3_mem_monitor.log on
# each node, so we can diagnose the OOM root cause.
( while true; do
    echo "$(date +%H:%M:%S) free=$(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null)kB comm=$(awk '/Committed_AS/{print $2}' /proc/meminfo 2>/dev/null)kB"
    sleep 5
  done ) &
_MEM_MON_PID=$!
trap 'kill $_MEM_MON_PID 2>/dev/null' EXIT

# --- RCCL networking: drop the EDF's CUDA-built OFI plugin, use Socket ------
# The EDF com.hooks.aws_ofi_nccl hook LD_PRELOADs a CUDA libnccl-net.so that
# cannot initialise on ROCm (no libcudart.so). Unset the plugin selectors and
# let RCCL fall back to its built-in Socket transport over hsn0 (Slingshot IP).
# This is the verified bring-up path (k3-eng11). Cross-node TP allreduce is
# slow over Socket; TODO a HIP-aware OFI/CXI plugin for real throughput.
unset NCCL_NET_PLUGIN NCCL_NET
export NCCL_SOCKET_IFNAME="${K3_NIC:-hsn0}"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Prevent RCCL socket bootstrap race when peers start with slight delay
# (job 584844: nid002668 workers 18s late -> "Message truncated" cascade).
# NCCL_SOCKET_NTHREADS=1 serialises socket I/O to avoid interleaved messages.
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-1}"
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
# skips CUDAGraphWrapper(FULL). Net: NO wrapper is installed -> the model
# runs EAGER, identical runtime to ENFORCE_EAGER=1. So K3_PIECEWISE=1 does
# NOT validate the PIECEWISE cudagraph hypothesis — it is blocked by the
# missing upstream @support_torch_compile on Kimi-K3 (requires vLLM/model
# work). The REAL correctness gate is ENFORCE_EAGER=1 K3_PREFIX_CACHE=0
# (cudagraph_mode=NONE -> no wrapper -> no FULL decode graph -> no
# recv_object deadlock). Use with LOAD_FORMAT=dummy and
# DISTRIBUTED_TIMEOUT_SECONDS=300 for iteration.
if [ "${K3_PIECEWISE:-0}" = "1" ]; then
  # ENFORCE_EAGER=1 forces mode=NONE + cudagraph_mode=NONE at config time
  # (vllm.py:~1190), silently overriding the PIECEWISE experiment. Warn and
  # strip --enforce-eager from VLLM_EXTRA_ARGS so the test is valid.
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
if [ "$RANK" = "0" ]; then
  VLLM_ARGS+=(--host 0.0.0.0 --port "$SERVE_PORT")
else
  VLLM_ARGS+=(--headless)
fi

exec python3 -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" $VLLM_EXTRA_ARGS
ENGINE

# K3_DISABLE_KDA: EDF replaces inherited env, so embed outer-shell value
# here (double-quoted heredoc IS expanded by the sbatch outer shell).
cat >> "${RUNDIR}/engine.sh" <<EOFA
export K3_DISABLE_KDA="${K3_DISABLE_KDA:-0}"
EOFA
chmod +x "$RUNDIR/engine.sh"

# ============================================================ otela worker ====
# Runs INSIDE the kimi-k3-vllm container on the HEAD node only (one peer: the
# DNT lists one row per peer, and launching on every node would over-register
# the single head endpoint). The container shares the host netns, so otela on
# the head reaches vLLM on localhost:$SERVE_PORT.
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
# otela only leaves the registry cleanly on SIGTERM (AnnounceLeave). If it is
# SIGKILLed the head keeps a `connected: true` row and round-robins real
# traffic into a dead endpoint. --signal=B:TERM@120 lets this trap (in the
# BATCH shell) scancel the otela STEP with TERM and wait for it to exit.
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

# Register once the head answers /health AND a real token comes back. The wait
# + probe run in the BATCH shell (host netns == container netns, so
# $HEAD:$SERVE_PORT is reachable here), then the otela worker is launched as
# its own step inside the container. The probe is MANDATORY: K3 has never
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
           python3 "$SCRIPT_DIR/gen_correctness.py" \
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
