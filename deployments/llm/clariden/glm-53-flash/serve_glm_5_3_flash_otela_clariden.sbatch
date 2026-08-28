#!/bin/bash
#SBATCH --job-name=glm53-otela-clariden
#SBATCH --account=infra02
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
# Full node (288 logical CPUs): the sglang glm-5.3 image pins TP ranks to GPU
# NUMA domains using host CPU ids; a shrunken cpuset raises "CPU number N is
# not eligible" and kills ranks. 1 task/node x 288 cpus = whole node.
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
# No --mem/--exclusive: clariden normal has OverSubscribe=EXCLUSIVE
# (jobs get dedicated whole nodes via 288 cpus + 4 gpus/node), and /dev/shm
# is a fixed 334 GiB tmpfs per node — enough for servekit's per-rank
# presharded staging with headroom.
# B:TERM@120 signals the BATCH SHELL (B:) 120 s before wall-time, giving the
# stop_otela trap below a window to scancel the otela step with TERM. Without
# B: the signal goes to the job steps, the batch-shell trap never runs, and
# the only grace is KillWait (30 s) — less than otela's ~10 s graceful leave
# (AnnounceLeave -> 5 s CRDT drain -> srv.Shutdown), so no LEFT is published
# and the head keeps a `connected: true` row round-robining traffic into a
# dead endpoint.
#SBATCH --signal=B:TERM@120
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-53-flash/logs/serve-%j.out
#SBATCH --error=/capstor/scratch/cscs/xyao/glm-53-flash/logs/serve-%j.out
#
# =============================================================================
# Serve zai-org/GLM-5.3-Flash (FP8 e4m3 dynamic, ~306 GiB, multimodal
# vision+text, DSA + linear + KDA hybrid attention, 45 layers / 288 routed
# experts top-8, MTP) on CSCS Clariden (4x GH200 / node, aarch64, Slingshot
# fabric, NO InfiniBand) and register it on OpenTela — one self-contained
# sbatch, no sibling scripts.
#
# Submit:
#   sbatch serve_glm_5_3_flash_otela_clariden.sbatch        # defaults: 2 nodes, TP4 x PP2
#   sbatch --nodes=4 --export=ALL,NNODES=4,PP_SIZE=4 \
#          serve_glm_5_3_flash_otela_clariden.sbatch        # 4 nodes, TP4 x PP4
#
# FAST FIRST-BRING-UP (recommended — isolates the GLM-5.3 forward path in
# ~10-15 min, not the multi-hour real-weight cold start; mirrors how the
# Kimi-K3 / GLM-5.2 forward hangs were root-caused via dummy-weight smoke):
#   sbatch --export=ALL,LOAD_FORMAT=dummy,GEN_CORRECTNESS_SMOKE=1 \
#          serve_glm_5_3_flash_otela_clariden.sbatch
#
# What it does, in order:
#   1. preflights the image, weights (or LOAD_FORMAT=dummy), and otela binary
#      (prints exact staging commands on failure)
#   2. srun's one sglang engine rank per node INSIDE the glm-5.3-flash-clariden
#      container (enroot/EDF), forming a single distributed TP4 x PP<NNODES>
#      engine. Only rank 0 (head) serves HTTP.
#   3. once the head's /health answers, runs a MANDATORY generation probe
#      BEFORE registration — /health alone is not enough (deepseek-v4 /
#      kimi-k3 / glm-5.2 lesson: an engine can answer /health while a broken
#      kernel path serves garbage or hangs). gen_correctness.py (reused from
#      the kimi-k3/glm52 recipes, model-agnostic) sends six greedy
#      /v1/completions prompts and checks each answer for its expected
#      factual substring.
#   4. ONLY if the probe passes does it start ONE otela WORKER — also inside
#      the container, on the head node via srun --overlap --gres=none — which
#      registers the `llm` service with the OpenTela mesh
#   5. on job end / scancel, SIGTERM (never KILL) the otela step so it
#      announces LEFT cleanly
#
# -----------------------------------------------------------------------------
# THE TOPOLOGY: ONE distributed engine across all nodes (like kimi-k3)
# -----------------------------------------------------------------------------
# GLM-5.3-Flash is ~306 GiB FP8 e4m3 — it does NOT fit on one 4x96 GB GH200
# node, so this is NOT the Beverin per-node-independent-replicas pattern. It
# is the JSC / kimi-k3 pattern: ONE distributed sglang engine across all
# nodes, ONE otela worker on the head registering the single head HTTP
# endpoint.
#
#   clariden compute (host netns; enroot container shares it)
#     rank 0 (HEAD)  -> sglang launch_server (HTTP on 0.0.0.0:$SERVE_PORT) + otela
#     rank 1..N      -> sglang workers (no HTTP; pipeline P2P to neighbours)
#
# TP4 keeps tensor-parallel all-reduce INSIDE one node's 4-GH200 NVLink domain.
# PP<NNODES> puts one pipeline stage per node, so the only traffic that
# crosses the Slingshot fabric is pipeline send/recv (point-to-point) —
# bandwidth-tolerant and CUDA-graph-safe. Cross-node allreduce (TP8 flat on
# 2 nodes) boots but is ~1-5 tok/s and cannot be captured in a CUDA graph on
# Slingshot/CXI (cudaErrorStreamCaptureInvalidated; OFI logs "GIN only
# supports RDMA transport protocol") — same finding as kimi-k3 on clariden.
#
# Memory (2 nodes TP4 x PP2, 8 GPUs total):
#   306 GiB FP8 weights / 8 GPUs = ~38 GiB weights/GPU of 96 GiB HBM
#   -> ~58 GiB free for KV pool + activations + JIT. Comfortable (unlike the
#      1.5 TB kimi-k3 on 24 GPUs). PP2 = 23 layers/stage (45 / 2, odd split —
#      stage 0 takes 23, stage 1 takes 22; sglang's load balancing handles
#      uneven layer counts).
#
# -----------------------------------------------------------------------------
# CONTAINER + NCCL ON SLINGSHOT (the clariden-specific part)
# -----------------------------------------------------------------------------
# Image is docker://lmsysorg/sglang:glm-5.3-flash (a GLM-5.3-specific SGLang
# build, multi-arch amd64 + arm64 — on aarch64 enroot import pulls the arm64
# manifest with sm_90 cubins for GH200) imported to a local .sqsh ONCE
# (build_glm_5_3_flash_image.sbatch). The EDF (glm-53-flash-clariden.toml,
# found via EDF_PATH below) carries com.hooks.aws_ofi_nccl.enabled=true.
# Clariden defaults to netstack source=artifact, so the hook bind-mounts the
# dynamically-linked (+dl) aws-ofi-ccl plugin + libfabric CXI provider at
# container start. That gives NCCL the native Slingshot transport — clariden
# has NO InfiniBand verbs, so NCCL's built-in IB transport is unused. The
# per-rank engine.sh below adds FI_CXI_* / NCCL_* tunables as := fallbacks
# that yield to whatever the hook sets (reconciled with CSCS resource-hook
# docs, carried from the verified kimi-k3 clariden recipe).
# =============================================================================

set -uo pipefail

# ----------------------------------------------------- script + deploy dirs --
# Locate the EDF ($OTELA_EDF_NAME.toml) regardless of the submit working
# directory, so both submit styles work:
#   - manual:  `cd <deploy-dir> && sbatch serve_*.sbatch`  (SLURM_SUBMIT_DIR
#     is the deploy dir, so the EDF is a sibling there)
#   - rcc:     `rcc job submit deployments/.../serve_*.sbatch` runs sbatch from
#     the repo root (remote_dir), so SLURM_SUBMIT_DIR is the ROOT and the EDF
#     lives in the deployments/llm/clariden/glm-53-flash subdirectory.
# Without this search pyxis fails at the preflight srun with "failed to find a
# readable environment file" (observed on kimi-k3 jobs 3032795/3034577,
# submitted from the root; the prior successful runs were submitted manually
# from the subdir). Slurm copies the sbatch to /var/spool/slurmd/job<id>/ and
# runs it from there, so ${BASH_SOURCE[0]} is the SPOOL copy and is NOT the
# EDF dir — BASH_SOURCE is only a last-resort fallback for non-Slurm runs.
# Keep ~/.edf on EDF_PATH so a pre-existing ~/.edf/<name>.toml remains usable.
OTELA_EDF_NAME="${OTELA_EDF_NAME:-glm-53-flash-clariden}"
_edf_dir=""
for _cand in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/deployments/llm/clariden/glm-53-flash" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
  [ -n "$_cand" ] && [ -f "$_cand/${OTELA_EDF_NAME}.toml" ] && { _edf_dir="$_cand"; break; }
done
if [ -z "$_edf_dir" ]; then
  echo "FATAL: cannot locate ${OTELA_EDF_NAME}.toml (searched SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>} and the BASH_SOURCE dir). Submit from the deploy dir, or set EDF_PATH explicitly." >&2
  exit 1
fi
export EDF_PATH="${_edf_dir}:${EDF_PATH:-${HOME}/.edf}"
export SCRIPT_DIR="$_edf_dir"
unset _edf_dir _cand

# ---------------------------------------------------------------- deployment --
# Reuses the existing /capstor/scratch/cscs/xyao/glm-53-flash tree: the
# imported image, EDF cache dirs, and a warm Triton cache live here, so a
# fresh submit does not re-pull or re-JIT from cold.
export DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-53-flash}"
export IMAGE="${IMAGE:-$DEPLOY_DIR/images/sglang-glm-5.3-flash.aarch64.sqsh}"
# Weights on /capstor/scratch (verified fast metadata; the /capstor/store MDT
# was overloaded during download, so the canonical HF cache move was deferred
# — see download_glm_5_3_flash.sbatch. Serving from /capstor/scratch matches
# the existing HF_HOME convention and is the better serving location anyway.)
export MODEL_PATH="${MODEL_PATH:-/capstor/scratch/cscs/xyao/models/zai-org/GLM-5.3-Flash}"
# Served under the canonical HF id so the peer is the GLM-5.3-Flash endpoint
# on the mesh (coexists with the kimi-k3 / glm47-flash registrations). Override
# to a short alias if the mesh's served-model routing prefers one.
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-5.3-Flash}"

# ------------------------------------------------------------------- shape ----
# WHY TP4 x PP<NNODES> is the default (verified on clariden for kimi-k3, same
# site/fabric/runtime):
# TP4 keeps collectives on the intra-node 4-GH200 NVLink domain. PP<NNODES>
# stages one pipeline stage per node, so only point-to-point send/recv
# crosses Slingshot — stable and CUDA-graph-safe. The flat TP8 alternative
# (2 nodes x 4 GPUs = 8) boots but every MoE allreduce goes cross-node over
# CXI without graph capture (~1-5 tok/s measured on kimi-k3; CG_DECODE must
# be `disabled`). Default PP_SIZE = NNODES (one stage/node); for 2 nodes
# that gives 45/2 = 23/22 layers per stage (uneven, handled by sglang).
export NNODES="${NNODES:-$SLURM_JOB_NUM_NODES}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-$NNODES}"
# EP_SIZE for GLM-5.3-Flash: 288 routed experts, top-8. With TP4, EP_SIZE=TP
# gives 288/4 = 72 experts/GPU (manageable). GLM-5.3's config does not pin a
# strict ep_size * moe_dp_size == tp_size rule (unlike kimi-k3), but EP=TP is
# the safe default that keeps expert dispatch intra-node. TODO(unverified):
# confirm on first run that the glm-5.3 image accepts --ep-size; if it errors
# or requires a different value, override EP_SIZE.
export EP_SIZE="${EP_SIZE:-$TP_SIZE}"

# MoE runner backend. GLM-5.3-Flash is FP8 e4m3 dynamic (quant_method=fp8 in
# config.json), NOT mxfp4 like kimi-k3. SGLang's fp8 MoE path is standard on
# sm_90 (GH200) — `auto` lets the image pick the right backend. Unlike kimi-k3
# (which NEEDED marlin to keep mxfp4 packed), fp8 dynamic quant has a mature
# native path. TODO(unverified): if `auto` picks a backend that dequantizes
# to bf16 and OOMs (the kimi-k3/Triton-Kernels anti-pattern), pin
# MOE_BACKEND=marlin or the image's validated fp8 runner. Check the first
# run's "Using MoE runner backend" log line.
export MOE_BACKEND="${MOE_BACKEND:-auto}"

# Attention backends. GLM-5.3-Flash has a HYBRID attention scheme:
#   - 34/45 layers are LINEAR attention (linear_attn_config: 64 heads,
#     head_dim 128, short_conv_kernel_size 4, with KDA on 34 of them)
#   - 11/45 layers are DeepSeek Sparse Attention (DSA) — every 4th layer
#     (indices 3,7,11,...,43)
# The glm-5.3-flash image auto-resolves the attention backends from the
# model's layer_types + index_types config. Verified on the first smoke
# (job 3208159, dummy weights): it set attention_backend=dsa,
# dsa_prefill_backend=flashmla_sparse, mamba_backend=triton (all fine —
# sgl_kernel.flash_mla is present on this aarch64 image) BUT it chose
# dsa_decode_backend=fa3, and FA3 is ABSENT on aarch64 sgl_kernel (only
# flash_mla, no flash_ops) — the SAME root cause as kimi-k3 clariden fix 3,
# manifesting on the DSA DECODE backend instead of the general decode one.
# The engine crashed during CUDA-graph capture: "cannot import name
# 'flash_ops' from 'sgl_kernel' ... Can not import FA3".
#
# FIX (GLM-5.3 analog of kimi-k3 fix 3): pin BOTH DSA backends to
# flashmla_sparse (which uses sgl_kernel.flash_mla, confirmed present).
# Set BOTH --dsa-prefill-backend AND --dsa-decode-backend together, mirroring
# the kimi-k3 "set both prefill+decode or auto-resolution cancels" lesson —
# setting only one DSA knob risks the other reverting to fa3. Do NOT pass
# the kimi-k3 GENERAL knobs (--prefill/decode-attention-backend): those
# target non-DSA models and would break GLM-5.3's linear-attention layers.
# The linear (mamba/KDA) layers are unaffected — mamba_backend=triton needs
# no flash dependency.
# Valid DSA backend choices (from the glm52 beverin recipe):
# flashmla_sparse, flashmla_sparse_q8, flashmla_kv, flashmla_auto, fa3,
# tilelang, aiter, trtllm (fa3/trtllm are NVIDIA-only -> crash on aarch64).
export GLM53_DSA_PREFILL="${GLM53_DSA_PREFILL:-flashmla_sparse}"
export GLM53_DSA_DECODE="${GLM53_DSA_DECODE:-flashmla_sparse}"
export CG_DECODE="${CG_DECODE:-full}"

# Request-length bound. GLM-5.3-Flash supports max_position_embeddings=
# 1048576 (1M). The KV pool is sized by MEM_FRAC, not this value. Default
# 32768 for a fast cold start + probe (shorter than the kimi-k3 1M default
# because GLM-5.3 is brand-new on clariden — confirm a short-context forward
# first, then raise toward 1M).
export CTX_LEN="${CTX_LEN:-32768}"
export MEM_FRAC="${MEM_FRAC:-0.88}"

# Slingshot high-speed NIC. Every clariden GH200 node exposes hsn0..hsn3 (one
# per GPU NUMA domain); hsn0 is used for the NCCL/GLOO socket bootstrap and is
# where SGLANG_HOST_IP is read from. The ofi plugin handles the data path over
# CXI independently of this setting.
export GLM53_NIC="${GLM53_NIC:-hsn0}"

# --- HiCache: offload evicted radix prefixes to Grace LPDDR ----------------
# On GH200 the Grace CPU's ~870 GiB LPDDR (RealMemory, verified in the kimi-k3
# clariden README) is connected to the 4 GPUs over NVLink-C2C at ~900 GB/s —
# NOT Slingshot (~50 GB/s) and NOT PCIe (~64 GB/s). SGLang's HiCache offloads
# EVICTED radix-cache prefixes (MLA KV pages + KDA state for the linear-SSM
# layers) from HBM to that LPDDR, growing the prefix cache, NOT
# max_total_num_tokens (the live batch KV stays in HBM, sized by --mem-
# fraction-static). Post-eviction hits on a reused prefix are pulled from
# LPDDR instead of recomputed — dramatic under concurrent shared-prefix
# traffic. This is the mechanism that makes a 1-node (TP4 PP1, ~19 GiB HBM
# headroom after 306 GiB weights) serve viable: the small HBM pool holds the
# live working set, HiCache extends the cold-prefix tail onto 870 GiB LPDDR.
#
# HICACHE_ENABLE=0 by default to match the proven kimi-k3 run (2920471). Enable
# with HICACHE_ENABLE=1 for the real 1-node serve (and for any 2-node run that
# wants the extra prefix-cache headroom). Verified on clariden 2026-08-05
# (kimi-k3 jobs 3008263/3009024/3010201): boots cleanly, no measured throughput
# cost, ~59 GB/node LPDDR at ratio 1.0.
#   - HICACHE_WRITE_POLICY=write_back is the ONLY working policy on the
#     kimi-k3 image: write_through_selective never backs up (hit-count trigger
#     is dead in the UnifiedRadixCache hybrid-SSM path) — do not use it.
#     TODO(unverified): confirm the same on the glm-5.3-flash image; if it
#     differs, override HICACHE_WRITE_POLICY.
#   - HICACHE_IO_BACKEND=kernel uses sgl_kernel.kvcacheio ops (verified present
#     in the aarch64 kimi-k3 image; the preflight below checks them on this image).
# Defaults: ratio 1.0, policy write_back, io kernel.
export HICACHE_ENABLE="${HICACHE_ENABLE:-0}"
export HICACHE_RATIO="${HICACHE_RATIO:-1.0}"
export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_back}"
export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-kernel}"

# --- Scheduler + NCCL watchdogs (fixes 11/12/13 from kimi-k3 clariden) ------
# sglang's HARD scheduler watchdog (ServerArgs.watchdog_timeout, default 300 s,
# soft=False) raises on any forward batch longer than the limit, and that
# raise KILLS the whole distributed engine. kimi-k3 job 3002366 served healthy
# 8 h then a PP request-metadata broadcast (broadcast_pyobj) hung >300 s and
# every stage tripped simultaneously — a >5-min hard kill on a healthy engine.
# kimi-k3 job 3151383 pinned the same stall to a single 289K-token request
# (TODO(unverified) — a per-request input cap is a hypothesis, not a fix).
# FIX (flags verified via `sglang serve --help` on this aarch64 image):
#   --watchdog-timeout      HARD: raised to 3600 s so a transient distributed
#                           stall rides out (1 h matches the load-barrier
#                           precedent); a true permanent hang still dies in
#                           finite time.
#   --soft-watchdog-timeout SOFT: dumps a stack trace at 600 s WITHOUT
#                           crashing, so the next stall is diagnosable even
#                           when the engine survives it. Keep soft < hard.
#   --dist-timeout          PyTorch ProcessGroupNCCL watchdog
#                           (init_process_group timeout). kimi-k3 job 3018155
#                           proved this is a SEPARATE, lower floor (default
#                           600 s) that fired ~5.5 h before the scheduler
#                           watchdog on the same PP SEND stall. Default 3600
#                           so the NCCL floor matches the scheduler watchdog;
#                           the SOFT_WATCHDOG_TIMEOUT dump is the first
#                           diagnostic, not a kill.
export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-3600}"
export SOFT_WATCHDOG_TIMEOUT="${SOFT_WATCHDOG_TIMEOUT:-600}"
export DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"

# Distributed torchrun store (rank discovery) and the sglang HTTP port. Both
# are EXPORTED here so srun carries them into the container, where engine.sh
# (a quoted heredoc) resolves them at runtime.
export DIST_PORT="${DIST_PORT:-20000}"
export SERVE_PORT="${SERVE_PORT:-30000}"
# Passthrough for experimental sglang flags without forking this script.
export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

# --- MTP / nextn prediction ------------------------------------------------
# GLM-5.3-Flash ships num_nextn_predict_layers=1 (one MTP/draft layer).
# SGLang only ACTIVATES it with --speculative-algorithm EAGLE. We do NOT pass
# that by default: the GLM-5.2 recipe documented "the spec-decode draft kernel
# is not yet validated on gfx950/gfx942; omit the --speculative-* flags and
# serve without MTP" (AMD finding), and GLM-5.3 has NEVER run on clariden
# (GH200/sm_90), so a plain (non-speculative) serve is the correct first
# bring-up. The nextn layer is simply unused without --speculative-*.
# Set GLM53_SPECULATIVE=1 (TODO(unverified): requires an EAGLE draft config
# validated for glm5_next on sm_90) to test speculative decoding later.
export GLM53_SPECULATIVE="${GLM53_SPECULATIVE:-0}"

# --- Load format -----------------------------------------------------------
# LOAD_FORMAT=auto (default): real weights from $MODEL_PATH on Lustre.
# LOAD_FORMAT=dummy: serve random weights for a fast forward-path smoke run
# (pair with GEN_CORRECTNESS_SMOKE=1) — proves the full GLM-5.3 init -> JIT
# -> forward -> HTTP pipeline without the multi-hour real-weight cold start.
# This is exactly how the kimi-k3 and GLM-5.2 forward hangs were isolated.
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# ------------------------------------------------------------------- otela ----
# arm64 build (sai-v0.0.6), statically linked -> runs on the host and inside
# the container (mounted via /capstor). It is started INSIDE the glm-5.3-flash
# container on the head so its HOME (set by the EDF to /capstor, not quota
# /users) is correct for its libp2p identity + badger DB. The x86 binary at
# the same dir is NOT usable on clariden (Exec format error).
export OTELA_DIR="${OTELA_DIR:-/capstor/scratch/cscs/xyao/opentela}"
export OTELA_BIN="${OTELA_BIN:-$OTELA_DIR/otela-arm64}"
# Shared alternative: /capstor/store/cscs/swissai/infra01/ocf-share/otela-arm64
# Direct-mesh bootstrap (an Alps OpenTela peer; verified reachable from a
# clariden compute node via `cat < /dev/null > /dev/tcp/140.238.223.116/43905`
# before this recipe was written). No relay needed — clariden compute has
# outbound internet (verified: the 306 GiB GLM-5.3-Flash HF download
# completed on a debug node via the same path).
export OTELA_RELAY_ADDR="${OTELA_RELAY_ADDR:-/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7}"
export OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
export OTELA_TCP_PORT="${OTELA_TCP_PORT:-43905}"
export OTELA_UDP_PORT="${OTELA_UDP_PORT:-59820}"
OTELA_STEP_NAME="${OTELA_STEP_NAME:-otela-worker-$SLURM_JOB_ID}"
OTELA_SEED="${OTELA_SEED:-21}"
OTELA_API_PORT="${OTELA_API_PORT:-18094}"

# Cold start: GLM-5.3-Flash is 62 shards x ~5 GiB = ~306 GiB FP8 on /capstor
# (single-OST Lustre stripe, like the verified kimi-k3 download pattern). At
# the kimi-k3 rate (~147 s/shard worst case on a bad Lustre day), cold start
# is up to ~2.5 h; typically much faster (FP8 weights are 4x smaller per
# param than kimi-k3's mxfp4, and only 62 shards vs 96). HEALTH_TIMEOUT gates
# the pre-registration wait only; the UNBALANCED_MODEL_LOADING_TIMEOUT_S
# patch (below) is what prevents the load barrier from killing slow PP groups.
# For a dummy-weight smoke run, set LOAD_FORMAT=dummy and HEALTH_TIMEOUT=1200.
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-10800}"

# ----------------------------------------------------- mandatory gen probe --
# GLM-5.3-Flash has NEVER generated a token on clariden (GH200/sm_90) — this
# is a first-ever forward on this architecture/site combination. /health
# returning 200 does NOT prove the DSA + linear + KDA hybrid path can run a
# forward (deepseek-v4 registered on /health and 502'd every request; kimi-k3
# hung at the forward with /health still up; GLM-5.2 on beverin served /health
# then the DSA decode hung at 100% GPU). The probe is MANDATORY.
#
# gen_correctness.py (sibling, reused from kimi-k3/glm52 — model-agnostic)
# sends six greedy /v1/completions prompts (temperature 0, max_tokens 64) and
# checks each answer for its expected factual substring (Paris; the prime
# sequence; 60 km/1.5 h = 40 km/h; Rayleigh; a fibonacci function body;
# entropy). With GEN_CORRECTNESS_SMOKE=1 (pair with LOAD_FORMAT=dummy), the
# gate is non-empty continuation instead — proving the full GLM-5.3
# (Glm5NextForConditionalGeneration) init -> JIT -> forward -> HTTP pipeline
# without real weights.
#
# Set GEN_PROBE=0 to skip (NOT recommended — registers on /health alone).
export GEN_PROBE="${GEN_PROBE:-1}"
export GEN_PROBE_TIMEOUT="${GEN_PROBE_TIMEOUT:-1200}"
export GEN_CORRECTNESS_MAX_TOKENS="${GEN_CORRECTNESS_MAX_TOKENS:-64}"
export GEN_CORRECTNESS_MIN_PASS="${GEN_CORRECTNESS_MIN_PASS:-5}"
export GEN_CORRECTNESS_PER_REQ_TIMEOUT="${GEN_CORRECTNESS_PER_REQ_TIMEOUT:-300}"
export GEN_CORRECTNESS_SMOKE="${GEN_CORRECTNESS_SMOKE:-0}"

export RUNDIR="$DEPLOY_DIR/run-$SLURM_JOB_ID"
PATCH_DIR="$DEPLOY_DIR/patches"
mkdir -p "$DEPLOY_DIR/logs" "$RUNDIR" "$DEPLOY_DIR/cache" "$DEPLOY_DIR/home" \
         "$PATCH_DIR"

# sitecustomize.py is auto-imported by the Python interpreter at startup
# (after PYTHONPATH is on sys.path) — BEFORE sglang's own code runs. This
# lets us patch the module-level UNBALANCED_MODEL_LOADING_TIMEOUT_S in the
# SAME process as `sglang serve`. $PATCH_DIR is on /capstor (mounted into
# the container by the EDF). engine.sh prepends it to PYTHONPATH before
# exec'ing sglang, because the EDF sets PYTHONPATH="" (clearing anything
# the batch shell exported).
#
# kimi-k3 fix 10: sglang's dist_barrier_after_load calls dist.monitored_barrier(
# group=tp_group, timeout=UNBALANCED_MODEL_LOADING_TIMEOUT_S, wait_all_ranks=
# True). The constant is HARDCODED at 480 s (8 min) in load_model_utils.py.
# The barrier is per-TP-group (GLOO, 4 ranks/node); only rank 0 has the
# timeout — non-zero ranks block indefinitely waiting for rank 0. If the
# slowest rank in a node happens to be non-zero, rank 0 waits 480 s then
# raises ValueError, killing the whole job. We monkey-patch the constant to
# 3600 s (1 h) so even a non-zero slow rank gets up to 1 h to finish loading
# from Lustre. Verified on kimi-k3 (job 3000965, all 32 ranks reached "Load
# weight end" with a 29-min spread, zero exceptions).
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

export UNBALANCED_MODEL_LOADING_TIMEOUT_S="${UNBALANCED_MODEL_LOADING_TIMEOUT_S:-3600}"

HEAD="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export HEAD_IP="$(getent hosts "$HEAD" | awk '{print $1}' | head -1)"

echo "[$(date -Is)] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST"
echo "[$(date -Is)] head=$HEAD ($HEAD_IP)  tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE nnodes=$NNODES"
echo "[$(date -Is)] model=$MODEL_PATH  served_as=$SERVED_MODEL_NAME  ctx=$CTX_LEN  mem=$MEM_FRAC  moe=$MOE_BACKEND  dsa=$GLM53_DSA_PREFILL/$GLM53_DSA_DECODE  cg_decode=$CG_DECODE"
echo "[$(date -Is)] edf=$OTELA_EDF_NAME  image=$IMAGE"
echo "[$(date -Is)] otela=$OTELA_BIN  bootstrap=$OTELA_RELAY_ADDR"
echo "[$(date -Is)] watchdog=$WATCHDOG_TIMEOUT soft=$SOFT_WATCHDOG_TIMEOUT dist=$DIST_TIMEOUT  health_timeout=$HEALTH_TIMEOUT"
echo "[$(date -Is)] gen_probe=$GEN_PROBE  smoke=$GEN_CORRECTNESS_SMOKE  min_pass=$GEN_CORRECTNESS_MIN_PASS  max_tokens=$GEN_CORRECTNESS_MAX_TOKENS  load_format=$LOAD_FORMAT  speculative=$GLM53_SPECULATIVE  hicache=$HICACHE_ENABLE/$HICACHE_RATIO/$HICACHE_WRITE_POLICY"
echo "[$(date -Is)] started=$(date --iso-8601=seconds)"

# ----------------------------------------------------------------- preflight --
# Pattern: check, and on failure print the EXACT staging command for a node
# that HAS the capability (login node with egress), then exit non-zero.
[ -f "$IMAGE" ] || {
  echo "FATAL: image not found: $IMAGE" >&2
  echo "       Import it ONCE on a login node with build_glm_5_3_flash_image.sbatch" >&2
  echo "       (next to this sbatch), which runs:" >&2
  echo "         enroot import -o $IMAGE docker://lmsysorg/sglang:glm-5.3-flash" >&2
  exit 1
}
if [ "$LOAD_FORMAT" != "dummy" ]; then
  [ -d "$MODEL_PATH" ] || { echo "FATAL: weights not found: $MODEL_PATH" >&2; exit 1; }
fi
[ "$PP_SIZE" = "$NNODES" ] || echo "WARN: PP_SIZE=$PP_SIZE != NNODES=$NNODES (intended?)" >&2
[ -x "$OTELA_BIN" ] || {
  echo "FATAL: otela binary not found/executable: $OTELA_BIN" >&2
  echo "       Stage an arm64 build on a login node, e.g.:" >&2
  echo "         mkdir -p $OTELA_DIR && cp /capstor/store/cscs/swissai/infra01/ocf-share/otela-arm64 $OTELA_BIN && chmod +x $OTELA_BIN" >&2
  exit 1
}
# gen_correctness.py sibling (mandatory probe — see above).
[ -f "$SCRIPT_DIR/gen_correctness.py" ] || {
  echo "FATAL: gen_correctness.py not found next to this sbatch: $SCRIPT_DIR/gen_correctness.py" >&2
  echo "       The recipe is a bundle (sbatch + glm-53-flash-clariden.toml +" >&2
  echo "       build_glm_5_3_flash_image.sbatch + gen_correctness.py)." >&2
  echo "       Submit with: cd \"$(cd "$SCRIPT_DIR" && pwd)\" && sbatch $0" >&2
  exit 1
}

# --- container preflight (1 node, no GPUs, ~20 s) -----------------------
# Validates the EDF actually loads + the aarch64 image has the modules this
# recipe depends on, BEFORE launching N nodes. The glm-5.3-flash image is
# purpose-built, so we check for the GLM-5.3 model class resolution (the
# image's transformers must ship Glm5NextForConditionalGeneration /
# glm5_next_text) and the sgl_kernel native ops the hybrid attention path
# needs. Runs via --overlap --gres=none so it doesn't consume the job's GPU
# allocation. TODO(unverified): the exact import names below are inferred
# from config.json (model_type=glm5_next, architectures=[Glm5NextForCausal
# LM...]); the purpose-built image should resolve them — if not, this
# preflight is the place to catch it before N nodes launch.
echo "[$(date -Is)] container preflight (EDF=$OTELA_EDF_NAME) ..."
if ! srun --jobid="$SLURM_JOB_ID" --overlap --gres=none --nodes=1 --ntasks=1 \
         -w "$HEAD" --environment="$OTELA_EDF_NAME" python3 -c '
import sglang
# GLM-5.3 model class. The image is docker://lmsysorg/sglang:glm-5.3-flash,
# so its transformers MUST ship the glm5_next architecture. We check the
# text model class (Glm5NextForCausalLM) — the multimodal wrapper
# (Glm5NextForConditionalGeneration) wraps it.
try:
    from transformers import Glm5NextForCausalLM
    print("preflight: glm5_next model class OK")
except Exception as e:
    print("glm5_next:", repr(e)[:120])
# sgl_kernel native ops used by the hybrid attention path. The exact names
# depend on the image build; presence (or absence) here guides which
# backend the image auto-resolves to. Missing ops are NOT fatal — the
# image may provide them under different names or via Triton.
for mod in ["sgl_kernel.flash_mla", "sgl_kernel.kvcacheio"]:
    try:
        __import__(mod)
        print(f"preflight: {mod} OK")
    except Exception as e:
        print(f"{mod}:", repr(e)[:80])
print("preflight: sglang", getattr(sglang,"__version__","?"))
         ' 2>&1 | tail -8; then
  echo "FATAL: container preflight failed (EDF load or critical import)." >&2
  echo "       Check the EDF ($OTELA_EDF_NAME) is on EDF_PATH=$EDF_PATH and" >&2
  echo "       the image $IMAGE is intact (sglang-glm-5.3-flash, aarch64)." >&2
  exit 1
fi

{
  echo "SERVICE_JOB_ID=$SLURM_JOB_ID"
  echo "SERVICE_HEAD_NODE=$HEAD"
  echo "SERVICE_HEAD_IP=$HEAD_IP"
  echo "SERVICE_PORT=$SERVE_PORT"
  echo "SERVICE_URL=http://$HEAD_IP:$SERVE_PORT"
  echo "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
  echo "FRAMEWORK=sglang"
  echo "TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE EP_SIZE=$EP_SIZE CTX_LEN=$CTX_LEN"
  echo "MOE_BACKEND=$MOE_BACKEND CG_DECODE=$CG_DECODE"
  echo "BOOTSTRAP=$OTELA_RELAY_ADDR"
  echo "SUBMITTED_AT=$(date --iso-8601=seconds)"
} > "$DEPLOY_DIR/last_service.env"

# ================================================================ engine ======
# Per-rank wrapper, heredoc'd into $RUNDIR so the sbatch stays self-contained
# (apart from the EDF + gen_correctness.py siblings). Runs INSIDE the
# glm-5.3-flash container (via srun --environment=$OTELA_EDF_NAME).
cat > "$RUNDIR/engine.sh" <<'ENGINE'
#!/bin/bash
set -uo pipefail
RANK="${SLURM_PROCID:-0}"
IFACE="${GLM53_NIC:-hsn0}"
# Pin NCCL/GLOO socket bootstrap to the high-speed NIC. The ofi plugin (loaded
# by the EDF's com.hooks.aws_ofi_nccl hook) carries the data path over CXI
# independently of this; this is only for the TCP bootstrap path.
export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE"
SGLANG_HOST_IP="$(ip -o -4 addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
[ -z "$SGLANG_HOST_IP" ] && SGLANG_HOST_IP="$(hostname -I | awk '{print $1}')"
export SGLANG_HOST_IP

# --- Slingshot NCCL tunables (reconciled with CSCS netstack hook) -----------
# The EDF's com.hooks.aws_ofi_nccl hook (Clariden netstack source=artifact) sets
# CXI/NCCL env at container start "to help prevent application stalls" (per
# resource-hook docs). engine.sh runs INSIDE the container after the hook, so an
# unconditional export would clobber those curated values. Use := so the hook's
# values win when present; our values (tuned on clariden for kimi-k3 multi-node,
# 2026-08) are only the fallback when the hook sets nothing. FI_MR_CACHE_MONITOR=
# userfaultfd avoids the default monitor that mishandles CXI-registered memory.
: "${NCCL_NET:=AWS Libfabric}" "${NCCL_CROSS_NIC:=1}" \
  "${FI_CXI_DISABLE_HOST_REGISTER:=1}" "${FI_CXI_DEFAULT_CQ_SIZE:=131072}" \
  "${FI_CXI_RDZV_THRESHOLD:=0}" "${FI_CXI_RDZV_GET_MIN:=0}" \
  "${FI_MR_CACHE_MONITOR:=userfaultfd}"
export NCCL_NET NCCL_CROSS_NIC FI_CXI_DISABLE_HOST_REGISTER \
  FI_CXI_DEFAULT_CQ_SIZE FI_CXI_RDZV_THRESHOLD FI_CXI_RDZV_GET_MIN \
  FI_MR_CACHE_MONITOR
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
echo "[rank $RANK] netstack: NCCL_NET=$NCCL_NET NCCL_CROSS_NIC=$NCCL_CROSS_NIC FI_CXI_DCQ=$FI_CXI_DEFAULT_CQ_SIZE FI_MR_CACHE_MONITOR=$FI_MR_CACHE_MONITOR"
# Force IPv4 for bootstrap sockets (carried from kimi-k3; clariden hsn
# interfaces are IPv4 so this is a safe no-op there).
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-IPv4}" GLOO_SOCKET_FAMILY="${GLOO_SOCKET_FAMILY:-IPv4}"

echo "[rank $RANK] $(hostname) ip=$SGLANG_HOST_IP iface=$IFACE head=$HEAD_IP:$DIST_PORT"
echo "[rank $RANK] shape: tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE moe=$MOE_BACKEND cg_decode=$CG_DECODE ctx=$CTX_LEN load_format=$LOAD_FORMAT"

# sitecustomize.py (in $PATCH_DIR, on /capstor) patches
# UNBALANCED_MODEL_LOADING_TIMEOUT_S at Python startup, BEFORE sglang runs.
# The EDF sets PYTHONPATH="" so we must re-add it here, before exec sglang.
export PYTHONPATH="${PATCH_DIR:-/capstor/scratch/cscs/xyao/glm-53-flash/patches}:${PYTHONPATH:-}"

# Verify the monkey-patch will take effect in this process tree.
python3 -c "import sglang.srt.model_executor.model_runner_components.load_model_utils as m; print('[rank '$RANK'] UNBALANCED_MODEL_LOADING_TIMEOUT_S =', m.UNBALANCED_MODEL_LOADING_TIMEOUT_S)" 2>/dev/null || true

# --- build the SGLang serve args -------------------------------------------
# `python -m sglang.launch_server` (not `sglang serve`): the entry point the
# kimi-k3 clariden recipe is validated against, and the one the multinode
# reference runs. The checkpoint's own config.json (still read from
# --model-path) supplies the architecture (Glm5NextForConditionalGeneration,
# model_type=glm5_next, 45 layers, 288 experts top-8, DSA+linear+KDA hybrid,
# MTP, FP8 e4m3 dynamic quant — read directly from disk, NOT hardcoded here).
#
# CRITICAL DIFFERENCE vs kimi-k3 clariden: we do NOT pass the GENERAL
# knobs (--prefill-attention-backend / --decode-attention-backend /
# --mamba-ssm-dtype). Those target non-DSA models and would break GLM-5.3's
# 34 linear-attention layers (mamba_backend=triton, no flash dependency).
# We DO pin the DSA-SPECIFIC backends (--dsa-prefill-backend /
# --dsa-decode-backend) to flashmla_sparse, because the aarch64 sgl_kernel
# has no FA3 (smoke job 3208159: auto-resolution chose dsa_decode_backend=
# fa3 and crashed at CUDA-graph capture with "cannot import flash_ops from
# sgl_kernel" — the kimi-k3 fix-3 root cause, one layer up). Set BOTH DSA
# knobs together, mirroring the kimi-k3 "set both or auto-resolution
# cancels" lesson — setting only one risks the other reverting to fa3.
#
# CRITICAL DIFFERENCE vs GLM-5.2 on beverin: we do NOT pass --disable-cuda-graph
# or --skip-server-warmup. Those were MI300A (gfx942) workarounds for the
# GLM-5.2 MTP/nextn layer crashing capture/warmup. GLM-5.3-Flash on GH200
# (sm_90) should support CUDA graphs (kimi-k3 on clariden runs CG_DECODE=full).
# If the first cold start crashes during capture/warmup, set
# DISABLE_CUDA_GRAPH=1 + SKIP_SERVER_WARMUP=1 (the GLM-5.2 workaround) and
# re-test — that would be a TODO(unverified) finding to log.
ENGINE_CMD=(python -m sglang.launch_server
  --trust-remote-code
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" --ep-size "$EP_SIZE"
  --nnodes "$NNODES" --node-rank "$RANK"
  --dist-init-addr "$HEAD_IP:$DIST_PORT"
  --dist-timeout "$DIST_TIMEOUT"
  --moe-runner-backend "$MOE_BACKEND"
  --dsa-prefill-backend "$GLM53_DSA_PREFILL"
  --dsa-decode-backend "$GLM53_DSA_DECODE"
  --cuda-graph-backend-decode "$CG_DECODE"
  --context-length "$CTX_LEN" --mem-fraction-static "$MEM_FRAC"
  --watchdog-timeout "$WATCHDOG_TIMEOUT" --soft-watchdog-timeout "$SOFT_WATCHDOG_TIMEOUT"
  --enable-metrics --uvicorn-access-log-exclude-prefixes /metrics /health
  --host 0.0.0.0 --port "$SERVE_PORT")

# LOAD_FORMAT override (e.g. dummy for a fast forward-path smoke run).
if [ -n "${LOAD_FORMAT:-}" ] && [ "${LOAD_FORMAT:-}" != "auto" ]; then
  ENGINE_CMD+=(--load-format "$LOAD_FORMAT")
fi

# MTP / EAGLE speculative decoding: DISABLED by default (GLM-5.3-Flash has
# never run on clariden; the GLM-5.2 recipe documented the spec-decode draft
# kernel is not validated on this family). The model ships one nextn layer
# (num_nextn_predict_layers=1); SGLang only activates it with
# --speculative-algorithm EAGLE, which we do NOT pass. TODO(unverified): set
# GLM53_SPECULATIVE=1 to test EAGLE with the image's validated draft config
# once a plain forward is confirmed.
if [ "${GLM53_SPECULATIVE:-0}" = "1" ]; then
  ENGINE_CMD+=(--speculative-algorithm EAGLE
               --speculative-num-steps 2
               --speculative-eagle-topk 4
               --speculative-num-draft-tokens 8)
  echo "[rank $RANK] WARN: speculative decoding ENABLED — GLM-5.3 EAGLE draft path is TODO(unverified) on sm_90." >&2
fi

# --- HiCache args (Grace LPDDR prefix-cache extension; see top-level) ------
# Gated on HICACHE_ENABLE=1. The := fallbacks MATCH the proven defaults
# (write_back, 1.0, kernel) — NOT write_through_selective, which the kimi-k3
# image docs say never backs up in the hybrid-SSM path.
HICACHE_ARGS=()
if [ "${HICACHE_ENABLE:-0}" = "1" ]; then
  HICACHE_ARGS+=(--enable-hierarchical-cache
                 --hicache-ratio "${HICACHE_RATIO:-1.0}"
                 --hicache-write-policy "${HICACHE_WRITE_POLICY:-write_back}"
                 --hicache-io-backend "${HICACHE_IO_BACKEND:-kernel}")
  echo "[rank $RANK] hicache: ENABLED ratio=${HICACHE_RATIO:-1.0} policy=${HICACHE_WRITE_POLICY:-write_back} io=${HICACHE_IO_BACKEND:-kernel}" >&2
fi
ENGINE_CMD+=(${HICACHE_ARGS[@]+"${HICACHE_ARGS[@]}"})

ENGINE_CMD+=(${SGLANG_EXTRA_ARGS:+ $SGLANG_EXTRA_ARGS})

# ulimit -c 0 + unset BASH_ENV: a multi-hundred-GB core dump is never useful,
# and BASH_ENV would run an init in every subshell the loader forks.
ulimit -c 0
unset BASH_ENV

echo "[rank $RANK] launching sglang (head=$([ "$RANK" = 0 ] && echo "serve(HTTP)+probe+otela" || echo "node_rank>=1(no HTTP)")) args=${ENGINE_CMD[*]}"
exec "${ENGINE_CMD[@]}"
ENGINE
chmod +x "$RUNDIR/engine.sh"

# ============================================================ otela worker ====
# Runs INSIDE the glm-5.3-flash container on the HEAD node only (one peer:
# the single distributed engine has one head HTTP endpoint, and launching on
# every node would over-register it). The container shares the host netns
# (verified on clariden for kimi-k3: a 2nd container on the head reaches the
# engine on localhost:$SERVE_PORT, and the host's `ss` sees the container's
# listening socket), so otela on the head reaches sglang on
# localhost:$SERVE_PORT. HOME is the EDF's /capstor/.../home (not quota
# /users), so its libp2p identity + badger DB persist cleanly across jobs.
OTELA_CFG_DIR="$DEPLOY_DIR/otela"
OTELA_CFG="$OTELA_CFG_DIR/cfg.yaml"
mkdir -p "$OTELA_CFG_DIR"
# otela is sai-v0.0.6 (arm64). Its CLI differs from the x86 opentela-v0.2.2
# the Beverin scripts use:
#   - a multiaddr goes to --bootstrap.static (NOT --bootstrap.addr, which
#     expects an HTTP bootstraps-list URL — the default is an http:// URL)
#   - --label is supported
#   - there is NO --subprocess flag (we use the JSC/kimi-k3 separate-worker
#     topology: one engine srun in the foreground + one otela srun on the head).
#
# CRITICAL: otela requires a proper cfg.yaml with `loglevel: debug`,
# `reachability: private`, `role: worker`, and `seed`. Without these it
# appears to "hang" after "AXIOM_DATASET not set, tracing disabled" — but it
# is actually working, just at INFO level which stops logging during CRDT
# sync. The JSC recipe (v0.2.3, x86) writes the same config file; verified on
# clariden job 3000965 (kimi-k3): with the config file, otela registered the
# LLM service and synced 2000+ peers within 45 s. Without it, 90 s of silence.
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
# otela only leaves the registry cleanly on SIGINT/SIGTERM (AnnounceLeave). If
# it is SIGKILLed the head keeps a `connected: true` row (the staleness sweep
# skips peers that registered a service) and round-robins real traffic into a
# dead endpoint. Signal the STEP (srun does not reliably relay signals to the
# task) and wait for it to actually exit.
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
    echo "[$(date -Is)]       Restart with the SAME seed (default $OTELA_SEED) so the" >&2
    echo "[$(date -Is)]       new worker re-adopts the peer ID and overwrites the" >&2
    echo "[$(date -Is)]       stale row in the CRDT." >&2
    return 0
  fi
  echo "[$(date -Is)] stopping otela step $step"
  scancel -s TERM "$step" 2>/dev/null
  # 20 s: --signal=B:TERM@120 covers wall-time expiry, but on the `scancel` path
  # the only grace is KillWait (default 30 s). otela needs ~10 s to shut down,
  # so 20 s leaves margin and still finishes before SIGKILL.
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

# Register once the head's engine answers /health AND the probe passes. The
# wait + probe run in the BATCH shell (host netns == container netns, so
# $HEAD_IP:$SERVE_PORT is reachable here), then the otela worker is launched as
# its own step inside the container.
(
  deadline=$(( SECONDS + HEALTH_TIMEOUT ))
  until curl -sf --max-time 5 "http://$HEAD_IP:$SERVE_PORT/health" >/dev/null 2>&1; do
    [ $SECONDS -ge $deadline ] && { echo "[$(date -Is)] health timeout after ${HEALTH_TIMEOUT}s; not registering" >&2; exit 1; }
    sleep 20
  done
  echo "[$(date -Is)] engine healthy (head=$HEAD_IP:$SERVE_PORT) after ~$((HEALTH_TIMEOUT - (deadline - SECONDS)))s"

  if [ "$GEN_PROBE" = "1" ]; then
    echo "[$(date -Is)] generation ${GEN_CORRECTNESS_SMOKE:+smoke }probe (model=$SERVED_MODEL_NAME, max_tokens=$GEN_CORRECTNESS_MAX_TOKENS, min_pass=$GEN_CORRECTNESS_MIN_PASS, budget=${GEN_PROBE_TIMEOUT}s) ..."
    PROBE_OUT="$RUNDIR/gen_correctness_${SLURM_JOB_ID}.json"
    # Run INSIDE the glm-5.3-flash container on the head (same
    # --overlap --environment=$OTELA_EDF_NAME launch as the otela worker):
    # the host's python3 may be too old and the container shares the host
    # netns so 127.0.0.1:$SERVE_PORT reaches sglang (bound on 0.0.0.0).
    # gen_correctness.py is reused from the kimi-k3/glm52 recipes
    # (model-agnostic: six greedy /v1/completions prompts, temperature 0,
    # factual-substring checks; with GEN_CORRECTNESS_SMOKE=1 the gate is
    # non-empty continuation instead of factual correctness).
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
      echo "[$(date -Is)]       For GLM-5.3-Flash a failure here most likely means the" >&2
      echo "[$(date -Is)]       DSA + linear + KDA hybrid forward path faults on" >&2
      echo "[$(date -Is)]       clariden (GH200/sm_90) — the one new piece on this site." >&2
      exit 1
    fi
  fi

  echo "[$(date -Is)] starting otela worker on $HEAD (inside container, EDF=$OTELA_EDF_NAME)"
  # Redirect INSIDE the container (not via srun's pipe) to avoid srun
  # output buffering that hides otela log lines until the buffer fills.
  # The log file is on /capstor (mounted in both host and container).
  srun --jobid="$SLURM_JOB_ID" --overlap --gres=none --nodes=1 --ntasks=1 \
       -w "$HEAD" --environment="$OTELA_EDF_NAME" -J "$OTELA_STEP_NAME" \
       bash -c '"$0" >"$1" 2>&1' "$RUNDIR/otela.sh" "$RUNDIR/otela.log"
) &

# ------------------------------------------------------------- run engine ----
# srun inherits the batch shell's exported vars (HEAD_IP, DIST_PORT, SERVE_PORT,
# TP_SIZE, MODEL_PATH, …) — standard Slurm behavior — and
# --environment="$OTELA_EDF_NAME" overlays the EDF's [env] (HOME on /capstor,
# HF_HOME, the com.hooks.aws_ofi_nccl annotation, …) on top. The quoted
# engine.sh heredoc then resolves these at runtime inside the container. If a
# var is mysteriously empty on a rank, check:
#   srun -n1 --environment=$OTELA_EDF_NAME env | grep NAME
# One engine rank per node, inside the container, in the foreground so the job
# lifetime == engine lifetime.
srun --ntasks-per-node=1 --environment="$OTELA_EDF_NAME" bash "$RUNDIR/engine.sh"
SRUN_EXIT=$?
echo "[$(date -Is)] srun_exited code=${SRUN_EXIT}"

exit "$SRUN_EXIT"
