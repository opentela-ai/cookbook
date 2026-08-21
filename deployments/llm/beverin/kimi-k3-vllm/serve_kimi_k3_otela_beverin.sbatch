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
# allreduce DOES cross Slingshot — over RCCL Socket transport (hsn*), which
# is the verified bring-up path (k3-eng8/k3-eng11 reached MoE backend
# selection — ROCM_AITER_MLA + AITER_MXFP4_BF16 via patched flydsl +
# sitecustomize.py, PP2xTP8=16 workers; k3-eng8 loaded shards to 16% @ ~147
# s/shard pre-restripe before being cancelled). Socket is the DEFAULT: the
# EDF's OFI/CXI plugin initialises but the first cross-node collective HANGS
# on this allocation (rccl-xport 601320/601321/601339/601341); see the RCCL
# block below. Two Socket-side all-to-all knobs are now DEFAULT ON since
# run-601453: RCCL_ALL_TO_ALL_PIVOT_ENABLE + RCCL_CHANNEL_TUNING_ENABLE cut
# the MoE all-to-all GPU time -31% and lifted agg tok/s 20.6 -> 23.8.
# A token has not yet been confirmed on this GPU family; the mandatory
# generation probe gates registration on it. TP4 x PP4 (4 nodes) keeps TP
# all-reduce intra-node (4 GPUs) but stages two pipeline layers per node —
# unverified, see README.
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
if [ "${K3_BREAKABLE_PIECEWISE:-0}" = "1" ] && [ "${PP_SIZE:-3}" != "1" ]; then
  # PP_SIZE>1 (e.g. the proven TP8 PP3) has a recv_object at each PP
  # boundary in the forward. The breakable PIECEWISE path is what makes
  # this SAFE: each @eager_break_during_capture op (VkernelFusedExperts.apply)
  # ends the current cudagraph segment and runs eagerly on the capture
  # stream, so the forward is captured as MANY small segments (split at MoE)
  # rather than one monolithic graph. A monolithic graph + PP recv was the
  # prior deadlock (jobs 588153/588156, cudagraph_mode=FULL_AND_PIECEWISE ->
  # decode_mode=FULL -> no break). This path (cudagraph_mode=PIECEWISE ->
  # decode_mode=PIEWISE -> breaks fire) is the vLLM-INTENDED fix
  # (config/vllm.py:1208-1230). A deadlock now fails in
  # DISTRIBUTED_TIMEOUT_SECONDS=${DISTRIBUTED_TIMEOUT_SECONDS:-3600}s, not 1 h.
  echo "[rank ${RANK:-0}] K3_BREAKABLE_PIECEWISE=1 with PP_SIZE=${PP_SIZE:-3}: PP recv present;" >&2
  echo "[rank ${RANK:-0}]    relying on breakable PIECEWISE (breaks fire) to avoid the PP recv deadlock." >&2
  echo "[rank ${RANK:-0}]    Deadlock timeout=${DISTRIBUTED_TIMEOUT_SECONDS:-3600}s." >&2
fi
# Request-length bound. Kimi-K3 supports up to max_position_embeddings=1048576
# (1M). The KV pool is sized by --gpu-memory-utilization, NOT this value (MLA +
# KDA hybrid attention: ~107 KB/token), so even large --max-model-len admits a
# healthy pool. Default 131072 (128K) per the bring-up target; raise toward
# 1048576 once a cold start + generation are confirmed fast.
export CTX_LEN="${CTX_LEN:-131072}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
# --max-num-batched-tokens caps tokens/step (prefill + decode). vLLM's
# default 2048 is too small for high-concurrency sweeps: at C=256 with
# in_tok~=50, prefill demand is ~12800 tokens and the chunked schedule
# exceeds benchmark.py's HTTP timeout -> TimeoutError + JSONDecodeError
# (job 600469). BENCHMARK.md's baseline (job 598005) used 8192, which lets
# the C=256 sweep complete (~172 tok/s). Default 8192 matches that baseline.
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# MI300A is an APU: HBM is shared between CPU and 4 GPU complexes.
# 0.9 leaves only ~40 GiB for OS → OOM kill. 0.8 leaves ~91 GiB.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
# vLLM distributed init store (rank discovery) + head HTTP port.
export MASTER_PORT="${MASTER_PORT:-6379}"
export SERVE_PORT="${SERVE_PORT:-8080}"
# Passthrough for experimental vLLM flags (e.g. ENFORCE_EAGER=1 adds
# --enforce-eager) without forking this script.
# All Slingshot NICs (hsn*) for RCCL Socket data path — fans the
# cross-node MoE all-to-all over the node's aggregate Slingshot BW (vs
# single hsn0). Default ON since run-601453. Override to hsn0 to reproduce
# the pre-601453 single-NIC path. gloo is pinned to hsn0 separately below.
export K3_NIC="${K3_NIC:-hsn}"
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

# K3_BREAKABLE_PIECEWISE=1: the vLLM-INTENDED piecewise-cudagraph path for
# models WITHOUT @support_torch_compile (which Kimi-K3 lacks: no kimi_k3 model
# file carries the decorator). config/vllm.py:1211-1241 already auto-enables
# VLLM_USE_BREAKABLE_CUDAGRAPH=1 for KimiK3ForConditionalGeneration (line 1230)
# AND, via a SEPARATE check at line 1236, forces compilation_config.mode=NONE
# (line 1241). The two prior attempts ran eager or deadlocked:
#   (1) AUTO default left cudagraph_mode=FULL_AND_PIECEWISE (the default),
#       whose decode_mode()=FULL (compilation.py: FULL_AND_PIECEWISE=(2,1),
#       decode_mode()=value[0]=FULL) means eager_break_during_capture sees
#       FULL and does NOT break (breakable_cudagraph.py:100-103) -> one
#       monolithic decode graph -> stale MoE routing + PP3 recv deadlock
#       (jobs 588153/588156).
#   (2) K3_PIECEWISE=1 set VLLM_USE_BREAKABLE=0 + mode=VLLM_COMPILE, but
#       without @support_torch_compile torch.compile never wraps the model
#       (vllm.py:~2410 only WARNS) -> no FX graph -> is_breakable=False +
#       PIECEWISE.has_full_cudagraphs()=False installs NO wrapper -> EAGER.
# This THIRD path sets VLLM_USE_BREAKABLE_CUDAGRAPH=1 BEFORE `exec python3`
# (see the block near line 700) AND sets cudagraph_mode=PIECEWISE (simple
# enum -> decode_mode=PIECEWISE -> breaks fire). sitecustomize.py decorates
# VkernelFusedExperts.apply with @eager_break_during_capture (the break
# primitive, breakable_cudagraph.py:59-117; apply writes its result into the
# caller-provided `output` tensor per the decorator's in-place requirement).
# Unlike the FX-splitting path, breakable does NOT need torch.compile: it uses
# its own BreakableCUDAGraphCapture + the @eager_break_during_capture op.
# CRITICAL: VLLM_USE_BREAKABLE_CUDAGRAPH=1 is exported in engine.sh (line
# ~700, before `exec python3` at ~827) because eager_break_during_capture
# checks is_breakable_enabled() AT DECORATION TIME (breakable_cudagraph.py:
# 90-91), i.e. when sitecustomize runs at Python startup, BEFORE vLLM's K3
# auto-default (config/vllm.py:1230) runs. Relying on the auto-default would
# let the decoration silently return fn undecorated -> the experiment is a
# no-op (runs eager). Pre-setting it is equivalent to the auto-default for
# the config path (line 1236 still forces mode=NONE) AND in time for the
# decoration.
# gpu_model_runner.py:5444-5448 installs BreakableCUDAGraphWrapper when
# is_breakable_cudagraph_enabled() AND cudagraph_mode != NONE AND not
# ubatching -- INDEPENDENT of `mode`, so mode=NONE + cudagraph_mode=PIECEWISE
# installs the wrapper.
# MUST run TP8 PP1 (2 nodes): no PP boundary -> no recv_object in the forward
# -> cannot deadlock even if a break is missed. The PP_SIZE=1 guard above
# enforces this. Default 0 = unchanged (ENFORCE_EAGER=1 is the proven path).
#   sbatch --export=ALL,K3_BREAKABLE_PIECEWISE=1,TP_SIZE=8,PP_SIZE=1,NNODES=2 \
#     serve_kimi_k3_otela_beverin.sbatch
export K3_BREAKABLE_PIECEWISE="${K3_BREAKABLE_PIECEWISE:-0}"
if [ "${K3_BREAKABLE_PIECEWISE:-0}" = "1" ] && [ "${K3_PIECEWISE:-0}" = "1" ]; then
  echo "[rank ${RANK:-0}] WARN: K3_PIECEWISE=1 and K3_BREAKABLE_PIECEWISE=1 are" >&2
  echo "[rank ${RANK:-0}]       mutually exclusive (K3_PIECEWISE sets" >&2
  echo "[rank ${RANK:-0}]       VLLM_USE_BREAKABLE=0, which defeats breakable)." >&2
  echo "[rank ${RANK:-0}]       Resetting K3_PIECEWISE=0 for this run." >&2
  export K3_PIECEWISE=0
fi

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

# --- RCCL networking ---------------------------------------------------------
# The EDF (kimi-k3-vllm.toml) ships a fully-tuned RCCL-over-OFI/Slingshot
# config: NCCL_NET_PLUGIN=ofi, NCCL_NET="AWS Libfabric", NCCL_NET_GDR_READ=1,
# NCCL_NET_GDR_C2C=1, NCCL_NCHANNELS_PER_NET_PEER=4, NCCL_P2P_LEVEL=NVL,
# NCCL_PROTO=^LL128, CXI provider via /opt/cscs/netstack. The OFI plugin
# (librccl-net-ofi.so) is RCCL-native (ROCm 7.2.3, "aws-ofi-nccl 1.20.0"),
# initialises cleanly across nodes (verified rccl-xport jobs 601320/601321/
# 601339). USE IT: the Socket path caps the cross-node TP all-to-all at
# ~78% of GPU wall (run-601239: rccl_all_to_all 15.5s/19.7s).
#
# The prior Socket override (k3-eng11) worked around a then-CUDA-built
# aws_ofi_nccl hook that could not init on ROCm (no libcudart.so). That
# hook is no longer in the lib path; the RCCL-native OFI plugin works, so
# the override is now a pessimization. Reproduce the legacy Socket path
# (debug only) with K3_RCCL_SOCKET=1.
# Verified rccl-xport 601320/601321/601339/601341: the EDF's OFI plugin
# INITIALIZES cleanly (aws-ofi-nccl 1.20.0, Libfabric 2.6, HIP 7.2.5) but
# the FIRST cross-node collective HANGS (>=2:47, no warmup) on this
# allocation, with or without GDR (K3_GDR=0). The EDF's com.hooks.
# aws_ofi_nccl hook is CUDA-built and unusable on ROCm, and RCCL's native
# built-in OFI data path does not (yet) complete on CSCS Alps CXI — a gap
# that needs CSCS netstack work, not an env tweak. So Socket stays the
# DEFAULT; set K3_RCCL_SOCKET=0 only once a working OFI path is found.
if [ "${K3_RCCL_SOCKET:-1}" = "0" ]; then
  echo "[rank $RANK] RCCL: EDF OFI/CXI over Slingshot (NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-?})"
else
  echo "[rank $RANK] RCCL: Socket fallback (K3_RCCL_SOCKET=${K3_RCCL_SOCKET:-1})"
  unset NCCL_NET_PLUGIN NCCL_NET
  # Fan Socket over ALL Slingshot NICs (vs single hsn0). gloo is pinned to
  # hsn0 below (no prefix matching); NCCL fans freely.
  export NCCL_SOCKET_IFNAME="${K3_NIC:-hsn0}"
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-1}"
  # All-to-all algo tuning (Socket path), DEFAULT ON since run-601453:
  #   RCCL_ALL_TO_ALL_PIVOT_ENABLE : pivot-based all-to-all (fewer round-
  #     trips). Cut the K3 MoE all-to-all GPU time 15.50s -> 10.71s (-31%)
  #     and the apply tail 4.65s -> 1.40s (-70%), agg tok/s 21.7 -> 23.8.
  #   RCCL_CHANNEL_TUNING_ENABLE   : RCCL channel auto-tuning.
  # Disable either per-run via --export=...,RCCL_ALL_TO_ALL_PIVOT_ENABLE=0
  # (or ='' to drop it entirely).
  export RCCL_ALL_TO_ALL_PIVOT_ENABLE="${RCCL_ALL_TO_ALL_PIVOT_ENABLE:-1}"
  export RCCL_CHANNEL_TUNING_ENABLE="${RCCL_CHANNEL_TUNING_ENABLE:-1}"
  # Empty value = drop the knob (don't export an empty env var).
  [ -z "$RCCL_ALL_TO_ALL_PIVOT_ENABLE" ] && unset RCCL_ALL_TO_ALL_PIVOT_ENABLE
  [ -z "$RCCL_CHANNEL_TUNING_ENABLE" ] && unset RCCL_CHANNEL_TUNING_ENABLE
  echo "[rank $RANK] RCCL tuning: pivot=${RCCL_ALL_TO_ALL_PIVOT_ENABLE:-0} ch_tune=${RCCL_CHANNEL_TUNING_ENABLE:-0} nic=${NCCL_SOCKET_IFNAME}"
fi
# gloo is a SEPARATE CPU transport from RCCL — vLLM V1 uses it for the
# MQ-broadcaster `in_the_same_node_as` check + torch.distributed.barrier
# during WorkerProc init. Without GLOO_SOCKET_IFNAME gloo binds a non-
# routable iface and cross-node recv times out 1800 s -> "WorkerProc
# initialization failed" (job 600189, real weights, 1h52m). Pinned to hsn0
# EVEN when NCCL fans out over hsn* (K3_NIC=hsn): gloo needs an EXACT
# iface name (no prefix matching) and a single routable NIC suffices for
# its tiny control-plane traffic (job 601406 crashed: "Unable to find
# address for: hsn"). Override with K3_GLOO_NIC if ever needed.
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
if [ "${K3_BREAKABLE_PIECEWISE:-0}" = "1" ]; then
  # --enforce-eager forces mode=NONE + cudagraph_mode=NONE at config time
  # (vllm.py:~1198-1199), which would defeat the experiment: with
  # cudagraph_mode=NONE, BreakableCUDAGraphWrapper is NOT installed
  # (gpu_model_runner.py:5444-5448 requires cudagraph_mode != NONE). Strip it.
  if [ "${ENFORCE_EAGER:-0}" = "1" ]; then
    echo "[rank $RANK] WARN: K3_BREAKABLE_PIECEWISE=1 and ENFORCE_EAGER=1 are" >&2
    echo "[rank $RANK]       mutually exclusive. Stripping --enforce-eager." >&2
    VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS// --enforce-eager/}"
  fi
  # MUST be set before `exec python3` (this heredoc is appended to engine.sh
  # and exec'd at ~line 827). eager_break_during_capture in sitecustomize.py
  # checks is_breakable_cudagraph_enabled() AT DECORATION TIME (Python
  # startup), before vLLM's K3 auto-default (config/vllm.py:1230). Without
  # this export the decoration is a silent no-op. Pre-setting is equivalent
  # to the auto-default for the config path (line 1236 still forces
  # mode=NONE) AND in time for sitecustomize.
  export VLLM_USE_BREAKABLE_CUDAGRAPH=1
  # cudagraph_mode=PIECEWISE (simple enum, NOT the (2,1) FULL_AND_PIECEWISE
  # tuple) -> decode_mode()=PIECEWISE -> eager_break_during_capture fires
  # (breakable_cudagraph.py:100-103, skips ONLY when mode==FULL).
  # splitting_ops non-empty is a belt-and-suspenders guard against the
  # compilation.py:1175 downgrade (PIECEWISE->NONE when splitting_ops empty).
  # With mode=NONE set_splitting_ops_for_v1 returns early (compilation.py:
  # 1127-1130) so the downgrade is not strictly reached, but a non-empty list
  # is harmless and matches the intended default content.
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

# KDA routing on gfx942 (issue #42): the faulting AITER / Triton
# chunked KDA kernels are replaced by the validated vk_hip_kda_delta_rule_
# fwd (routed in vkernels_attn.py via sitecustomize.py) when VKERNELS_KDA=1.
# This re-enables the delta-rule attention layer that K3_DISABLE_KDA=1
# used to silently drop, improving output quality.  EDF replaces the
# inherited env, so embed the outer-shell values here (the double-quoted
# heredoc IS expanded by the sbatch outer shell).
#
# Default VKERNELS_KDA=0 keeps the integration OPT-IN until the device
# kernel is validated on the cluster against the CPU oracle (max_rel<0.01,
# see VKERNELS_KDA_VALIDATE in vkernels_attn.py) and the 6/6 factual-
# correctness probes pass with KDA active (issue #42 acceptance).
#
# Until then the DEFAULT is the pre-#42 safe state: K3_DISABLE_KDA=1
# (all-MLA fallback, delta-rule layer dropped -- no regression vs. today).
# sitecustomize.py gives VKERNELS_KDA=1 PRECEDENCE over K3_DISABLE_KDA=1,
# so after validation the operator flips a SINGLE variable
# (VKERNELS_KDA=1) to serve the validated HIP kernel; then set
# K3_DISABLE_KDA=0 (or unset it) to complete AC4 (K3_DISABLE_KDA removed).
# MLA routing on gfx942 (issue #42): VKERNELS_MLA=1 patches
# vllm.v1.attention.selector.get_attn_backend to return VkernelMLABackend
# (vk_hip_mla_fwd via ctypes) ahead of TRITON_MLA for every use_mla layer.
# VKERNELS_MLA_FORCE=1 actually runs forward_mqa -> vk_hip_mla_fwd (vLLM
# calls forward_mqa bare at mla_attention.py:895 with NO try/except, so
# FORCE=1 is the only way the kernel runs end-to-end).  VKERNELS_MLA_VALIDATE=0
# skips the CPU-oracle cross-check (libvkernels_c.so not built yet).
# Default VKERNELS_MLA=0 / VKERNELS_MLA_FORCE=0 keep the integration OPT-IN
# until validated on the cluster (mirrors VKERNELS_KDA).
# STEP_PROFILE knobs (defaults; override via --export at sbatch submit). The
# gating block below arms a per-step torch.profiler on the selected ranks
# (sitecustomize.py); benchmark.py triggers capture by writing
# $VLLM_STEP_PROFILE_DIR/.go right before the first measured CONC:NUMREQ level.
#   STEP_PROFILE_SPEC         CONC:NUMREQ for the profiled load (default 8:16 —
#                             warmup + ~15s of steady-state decode, ~64 tok/req).
#   STEP_PROFILE_BUDGET       overall wall budget for the profile srun, s
#                             (default 600 — eager warmup _run(4,8) ~154s
#                             + measured C=8:16 ~150s; under job --export=ALL
#                             this reaches benchmark.py as BENCH_TIMEOUT, so it
#                             must exceed the warmup or the measured level is
#                             skipped and no execute_model fires to STOP the
#                             profiler. The driver wraps it in timeout +300).
#   STEP_PROFILE_PER_REQ_TIMEOUT  per-request timeout handed to benchmark.py as
#                             a BENCH_PER_REQ_TIMEOUT prefix (default 300; NOTE:
#                             under --environment=EDF the prefix does NOT reach
#                             the remote task, so benchmark.py uses its own 600s
#                             default — harmless for C=8:16 ~64 tok/req).
#   VLLM_STEP_PROFILE_DUR     per-rank capture window, s (default 15).
#   VLLM_STEP_PROFILE_RANKS   comma list of ranks to trace (default 0,8,16 =
#                             one per PP stage at TP=8).
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

# Issue #42 launch guard: the KDA exports above MUST precede `exec`.
# Earlier these exports were appended AFTER `exec python3 ...` and were
# therefore DEAD CODE (exec replaces the shell), so sitecustomize.py saw
# neither var set -> the default Triton KDA path faulted on gfx942 -> NaN
# ("!!!..." / repetitive garbage) on every prompt. Assert one of the two
# VALID states is in effect BEFORE exec'ing vLLM, so a future silent drop
# is caught at launch instead of producing junk.
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
