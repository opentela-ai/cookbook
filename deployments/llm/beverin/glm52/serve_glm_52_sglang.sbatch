#!/bin/bash
#SBATCH --job-name=glm52-sglang-beverin
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
# Do NOT set --cpus-per-task here: shrinking the cpuset breaks the image's
# SGLANG_SET_CPU_AFFINITY=1 (set in /etc/environment), whose NUMA pinning
# assumes the full 0..191 logical-CPU cpuset (one MI300A die per TP rank:
#   gpu0->[0-23,96-119] gpu1->[24-47,120-143] gpu2->[48-71,144-167] gpu3->[72-95,168-191]).
# --exclusive gives the whole node; the srun step requests the full
# SLURM_CPUS_ON_NODE per task so the container sees 0-191 (glm47 lesson).
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
# B:TERM@120 signals the BATCH SHELL (B:) 120 s before wall-time, giving the
# stop_otela trap below a window to scancel the otela step with TERM. Without
# B: the signal goes to the job steps, the batch-shell trap never runs, and
# the only grace is KillWait (30 s) — less than otela's ~10 s graceful leave
# (AnnounceLeave -> 5 s CRDT drain -> srv.Shutdown), so no LEFT is published
# and the head keeps a `connected: true` row round-robining into a dead endpoint.
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
# What it does, in order:
#   1. preflights the EDF/image, weights (or LOAD_FORMAT=dummy), otela binary,
#      and the gen_correctness.py + sglang-rocm.toml + sglang_launcher_mi300a.py
#      siblings.
#   2. srun's one SGLang engine rank per node INSIDE the sglang-rocm container
#      (enroot/EDF), forming a single distributed TP x PP engine. SGLang auto-
#      skips HTTP on node_rank>=1 (engine.py L1138); only node_rank 0 (head)
#      binds 0.0.0.0:$SERVE_PORT. All ranks share --nnodes/--node-rank/
#      --dist-init-addr (the head is the TCPStore master on $MASTER_PORT).
#   3. once the head's /health answers, runs a MANDATORY generation probe
#      BEFORE registration — /health alone is not enough (deepseek-v4 / K3
#      lesson: an engine can answer /health while a broken kernel path serves
#      garbage or hangs). gen_correctness.py (reused from the kimi-k3 recipe,
#      model-agnostic) sends six greedy /v1/completions prompts (temperature 0,
#      max_tokens 64) and checks each answer for its expected factual substring
#      (Paris; the prime sequence; 60 km/1.5 h = 40 km/h; Rayleigh; a fibonacci
#      function body; entropy). With GEN_CORRECTNESS_SMOKE=1 (pair with
#      LOAD_FORMAT=dummy), the gate is non-empty continuation instead — proving
#      the full DSA forward -> HTTP pipeline without real weights.
#   4. ONLY if the probe passes does it start ONE otela WORKER — inside the
#      container, on the head via srun --overlap --gres=none — which registers
#      the `llm` service (one peer for the single head endpoint).
#   5. on job end / scancel, SIGTERM (never KILL) the otela step so it
#      announces LEFT cleanly.
#
# -----------------------------------------------------------------------------
# THE TOPOLOGY (ONE distributed engine across all nodes, like kimi-k3)
# -----------------------------------------------------------------------------
# GLM-5.2 BF16 is ~1.5 TB — it does NOT fit one 4x137 GB MI300A node, so this
# is NOT the per-node-independent-replicas pattern (GLM-4.7-Flash). It is the
# JSC/kimi-k3 pattern: ONE distributed SGLang engine across all nodes, ONE
# otela worker on the head registering the single head HTTP endpoint.
#
#   beverin compute (host netns; enroot container shares it)
#     rank 0 (HEAD)  -> sglang serve (HTTP on 0.0.0.0:$SERVE_PORT) + otela worker
#     rank 1..N      -> sglang serve, node_rank>=1 (no HTTP; pipeline P2P)
#
# Default TP8 x PP3: each TP group of 8 spans 2 nodes, so MoE allreduce DOES
# cross Slingshot — but over RCCL Socket transport (hsn0), the verified bring-
# up path (kimi-k3 job 591401 isolated-RCCL test passed). GLM-5.2 has 64
# attention heads; TP8 -> 8 heads/GPU (in [1,15], valid for MLA; unlike K3's
# 96 heads which forbade TP4). PP3 stages 78/3 = 26 layers each.
#
# -----------------------------------------------------------------------------
# WHY THIS IS LOWER RISK THAN KIMI-K3 ON BEVERIN
# -----------------------------------------------------------------------------
#   * BF16 weights (NOT MXFP4) -> no AITER FlyDSL / marlin MXFP4 MoE path (the
#     K3 gfx942 hang locus). Standard Triton bf16 MoE, already proven for
#     GLM-4.7-Flash on this image.
#   * hidden_act=silu (NOT SiTU) -> no K3 SiTU-and_mul custom kernel.
#   * MLA already proven on MI300A (glm47-flash, TRITON_MLA via SGLANG_USE_AITER=0).
#   * DSA is SGLang-native on gfx942: SGLang v0.5.16 ships dsa_backend.py +
#     model_config.py L114 (GlmMoeDsaForCausalLM) + overrides.py L329/L575
#     (auto attention_backend=dsa for "DeepSeek 3.2/GLM 5"), and on hip auto-
#     defaults dsa_{prefill,decode}_backend=tilelang. transformers 5.12.1
#     (bundled) ships GlmMoeDsaForCausalLM + registers glm_moe_dsa in auto.py.
#   * The ONE new/risky piece vs glm47-flash is the DSA indexer + sparse
#     attention via tilelang/aiter on the MI300A APU specifically. The probe
#     is designed to surface exactly that.
#
# -----------------------------------------------------------------------------
# CONTAINER + RCCL ON SLINGSHOT (same as kimi-k3)
# -----------------------------------------------------------------------------
# Image: lmsysorg/sglang@sha256:80d04638... (v0.5.16-rocm720-mi30x, 66 GiB,
# already imported to .edf_imagestore; same image as glm47-flash). The EDF
# (sglang-rocm.toml) does NOT enable the com.hooks.aws_ofi_nccl hook (that
# CUDA-built libnccl-net.so cannot init on ROCm); engine.sh additionally
# unsets NCCL_NET_PLUGIN/NCCL_NET, pins NCCL_SOCKET_IFNAME=hsn0 +
# NCCL_IB_DISABLE=1, and lets RCCL use its built-in Socket transport over
# Slingshot. TODO: a HIP-aware OFI/CXI plugin for real throughput.
# =============================================================================

set -uo pipefail

# ----------------------------------------------------- script + deploy dirs --
# SCRIPT_DIR is where this sbatch + the EDF + gen_correctness.py +
# sglang_launcher_mi300a.py live. `rcc run --cwd <ABSOLUTE recipe dir>` makes
# BASH_SOURCE resolve here; without it `rcc job submit` sets SLURM_SUBMIT_DIR
# to the remote_dir root and the siblings are not found (kimi-k3 job 590745).
export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
export OTELA_EDF_NAME="${OTELA_EDF_NAME:-sglang-rocm}"

# ---------------------------------------------------------------- deployment --
export DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm52-sglang-beverin}"
# GLM-5.2 BF16 on /capstor/store (the CSCS SwissAI HF cache, project 7659).
# 282 safetensors shards, ~1.41 TB. No remote modeling .py -> the class
# (GlmMoeDsaForCausalLM) is resolved from the image's transformers (5.12.1,
# which ships it + registers glm_moe_dsa in auto.py L749).
export MODEL_PATH="${MODEL_PATH:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.2}"
# Served under the canonical HF id so the peer is the GLM-5.2 endpoint on the
# mesh (coexists with the glm47-flash registration). Override to a short alias
# if the mesh's served-model routing prefers one.
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-5.2}"
export IMAGE="${IMAGE:-/capstor/scratch/cscs/xyao/.edf_imagestore/sglang+sglang+v0.5.16-rocm720-mi30x.x86_64.sqsh}"

# ------------------------------------------------------------------- shape ----
# TP8 x PP3 (default): world = TP*PP = 24 must equal nodes*GPUS_PER_NODE.
# GLM-5.2: 64 attention heads -> TP8 = 8 heads/GPU (valid for MLA, in [1,15]).
# 78 layers / PP3 = 26 layers/stage. 256 routed experts, top-8 -> 896/8 = 112
# experts/GPU. BF16 ~1.41 TB / 24 GPUs = ~59 GiB weights/GPU of 137 GiB HBM,
# leaving ~78 GiB free for KV + activations + JIT. Compare kimi-k3 (~1.44 TB,
# same 24-GPU layout, ~71 GiB/GPU mxfp4). TP4 is also valid for GLM-5.2
# (64/4=16, a multiple of 16) but needs 6 nodes x PP6 and is unverified here.
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
# Request-length bound. GLM-5.2 supports max_position_embeddings=1048576 (1M).
# The KV pool is sized by --mem-fraction-static, NOT this value. Default 8192
# for the fastest possible cold start + probe; raise toward 1048576 once a
# short-context forward is confirmed (then exercise DSA at >2048 tokens).
export CTX_LEN="${CTX_LEN:-8192}"
# chunked-prefill-size: the GLM-5.2 AMD cookbook recipe pins 131072. At short
# context (<=8192) the whole prompt is one chunk, so this is moot for the probe
# but matches the validated MI300X config for later long-context runs.
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-131072}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
# mem-fraction-static: glm47-flash default 0.85 holds ~59 GiB BF16 weights +
# ~57 GiB KV per GPU. Lower (e.g. 0.80) leaves more room for DSA tilelang JIT
# if a cold-start OOM recurs.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# watchdog-timeout: the first forward (no warmup, SKIP_SERVER_WARMUP=1) compiles
# the tilelang DSA kernels. 1200 s (20 min) matches the GLM-5.2 AMD cookbook.
# Note (job 594176): SGLang's own watchdog did NOT fire during the first
# forward's JIT; it was PyTorch's ProcessGroupNCCL watchdog (separate, 600 s
# default) that fired. See DIST_TIMEOUT below.
export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-1200}"
# dist-timeout: PyTorch torch.distributed init_process_group(timeout=), which
# sets the ProcessGroupNCCL watchdog for ALL collectives (TP allreduce AND
# the PP pipeline point-to-point RECV/SEND of sampled token ids). Defaults to
# 600 s, which is far too short for the first forward's tilelang DSA JIT on
# 24 ranks: job 594176 hit "Watchdog caught collective operation timeout:
# WorkNCCL(SeqNum=7, OpType=RECV, NumelIn=1, NumelOut=1) ran for 600022 ms"
# — a PP stage's RECV timed out while the previous stage was still
# JIT-compiling its DSA kernels. 7200 s (2 h) gives each collective ample
# room to wait for cross-stage JIT (the TILELANG_CACHE_DIR on /capstor is
# shared, so the first forward populates the cache and subsequent forwards
# are fast). Override with --export=ALL,DIST_TIMEOUT=14400 for more room.
export DIST_TIMEOUT="${DIST_TIMEOUT:-7200}"

# ----------------------------------------------------- MI300A workarounds -----
# Six workarounds for MI300A (gfx942 APU).  Items 2-5 are carried from the
# glm47-flash recipe (validated SGLang-on-MI300A); items 1 and 6 are new for
# the GLM-5.2 DSA path and were discovered during the bring-up below.
#
#   1. SGLANG_USE_AITER (EDF default 1): gates the DSA indexer's preshuffle
#      paged-MQA path (aiter_can_use_preshuffle_paged_mqa, dsa/utils.py L49,
#      checks SGLANG_USE_AITER FIRST).  On MI300A BOTH values are broken:
#        =0 -> page_size=1 fallback; prefill WORKS (14 s, /health 200 OK)
#              but decode HANGS at 100% GPU, 0 Decode batches (jobs 594462,
#              594527; AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1 did NOT help
#              — SGLANG_USE_AITER=0 returns False before that flag).
#        =1 -> page_size=64 (preshuffle ENABLED); NO get_rope crash (GLM-5.2
#              has NO rope_scaling; confirmed job 594528), but the FIRST
#              FORWARD NEVER STARTS — 0% GPU, detokenizer idle (job 594528).
#      The glm47 recipe set this to 0 to avoid aiter's get_rope crash
#      (it indexes rope_scaling["original_max_position_embeddings"]).  That
#      risk does NOT apply to GLM-5.2 (no rope_scaling, and the DSA backend
#      uses its own fused_qk_rmsnorm, not get_rope).  EDF default is 1 (the
#      intended preshuffle config for future SGLang versions that fix the
#      gfx942 deadlock); override with --export=ALL,SGLANG_USE_AITER=0 to
#      test the page_size=1 fallback (currently the closest to working).
#   2. DISABLE_CUDA_GRAPH=1 : cuda-graph capture SIGKILLs on MI300A with the
#      MTP (nextn) layer ~30 s after weight load, during capture/warmup.
#   3. SKIP_SERVER_WARMUP=1 : the MTP layer also crashes the warmup pass.
#   4. HSA_NO_SCRATCH_RECLAIM=1 : ROCm 7.2 hard-requirement on MI300A.
#   5. is_integrated launcher (sglang_launcher_mi300a.py): PyTorch reports
#      is_integrated=True for every MI300A GPU, but each has its OWN dedicated
#      137 GiB HBM3; without the fix sglang uses whole-node ~428 GiB/TP rank
#      for the KV budget -> cgroup OOM ~35 s after load.
#   6. _patched_get_device_properties(device=None) in the launcher: the
#      aiter DSA DECODE path (dsa_indexer.py:1969, forward_cuda) calls
#      get_device_properties() with NO positional argument.  The original
#      torch.cuda.get_device_properties(device=None) defaults device to the
#      current device; the monkey-patch MUST replicate this default or the
#      scheduler crashes with "TypeError: missing 1 required positional
#      argument: 'device'" (job 594459, before the fix).
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-0}"
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
export SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-${SGLANG_SKIP_SERVER_WARMUP:-1}}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# GLM-5.2 reasoning + tool-call parsers (matches the GLM-5.2 cookbook
# `--reasoning-parser glm45 --tool-call-parser glm47`). Gated: set
# GLM52_ENABLE_PARSERS=0 if the image lacks them (startup errors "invalid
# parser" first). The /v1/completions probe below is unaffected (no chat
# template / tools applied), so parsing stays out of the mandatory gate.
export SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-glm45}"
export SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-glm47}"
export GLM52_ENABLE_PARSERS="${GLM52_ENABLE_PARSERS:-1}"

# MTP / EAGLE speculative decoding: DISABLED on AMD per the GLM-5.2 cookbook
# ("the spec-decode draft kernel is not yet validated on gfx950/gfx942; omit
# the --speculative-* flags and serve without MTP"). The model ships one nextn
# layer (num_nextn_predict_layers=1); SGLang only activates it with
# --speculative-algorithm EAGLE, which we do NOT pass. Do not add
# --speculative-* until MTP is validated on gfx942.
export GLM52_SPECULATIVE="${GLM52_SPECULATIVE:-0}"

# ------------------------------------------------------------------- otela ----
# x86_64 build, sai-v0.0.6 (same as the glm47-flash + kimi-k3 recipes). A
# multiaddr goes to --bootstrap.static (NOT --bootstrap.addr, which expects an
# HTTP bootstraps-list URL). Requires a proper cfg.yaml with loglevel/
# reachability/role/seed (verified on Clariden job 3000965); without it otela
# appears to "hang" after the tracing-disabled line — it is actually working,
# just at a log level that goes silent during CRDT sync. Started INSIDE the
# container on the head so its HOME (EDF -> /capstor, not quota /users) is
# correct for its libp2p identity + badger DB.
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

# Cold start from /capstor/store (single-OST Lustre stripe, ~147 s/shard at the
# kimi-k3 rate). GLM-5.2 is 282 shards x ~5 GiB = ~1.41 TB — comparable to
# kimi-k3's 96 shards x ~15 GiB = ~1.44 TB, so ~3-4 h worst case if unstriped.
# A one-time `lfs setstripe -s 4m -c -1` on the model dir (see README) cuts this
# to minutes; until then HEALTH_TIMEOUT gates the pre-registration wait only.
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-14400}"
# Mandatory pre-registration generation probe (deepseek-v4 / K3 lesson).
# Set GEN_PROBE=0 to skip (NOT recommended — registers on /health alone).
export GEN_PROBE="${GEN_PROBE:-1}"
# GEN_PROBE_TIMEOUT: overall wall budget for the 6-prompt probe. The first
# request (no warmup) compiles the DSA tilelang kernels, so 1200 s (matching
# --watchdog-timeout) is safer than the kimi-k3 600 s.
export GEN_PROBE_TIMEOUT="${GEN_PROBE_TIMEOUT:-1200}"
# Correctness probe knobs (see gen_correctness.py). MIN_PASS=5 tolerates one
# soft-prompt fluke while requiring ALL crisp prompts (Paris, primes, 40 km/h).
export GEN_CORRECTNESS_MAX_TOKENS="${GEN_CORRECTNESS_MAX_TOKENS:-64}"
export GEN_CORRECTNESS_MIN_PASS="${GEN_CORRECTNESS_MIN_PASS:-5}"
export GEN_CORRECTNESS_PER_REQ_TIMEOUT="${GEN_CORRECTNESS_PER_REQ_TIMEOUT:-300}"
# GEN_CORRECTNESS_SMOKE=1: with --load-format dummy (random weights), the probe
# checks each prompt yields a NON-EMPTY continuation instead of factual
# correctness. Verifies the full DSA pipeline (init -> JIT -> forward -> HTTP)
# without real weights — the fastest way to know if GLM-5.2 can do a forward
# pass on MI300A at all (mirrors the kimi-k3 dummy-weight hang isolation).
export GEN_CORRECTNESS_SMOKE="${GEN_CORRECTNESS_SMOKE:-0}"

# transformers: the image bundles 5.12.1, which already ships
# GlmMoeDsaForCausalLM + registers glm_moe_dsa in auto.py. The engine.sh
# guard below is a no-op unless GLM52_TRANSFORMERS_VERSION is overridden to a
# different version (or the image is older than expected).
export GLM52_TRANSFORMERS_VERSION="${GLM52_TRANSFORMERS_VERSION:-5.12.1}"

export RUNDIR="$DEPLOY_DIR/run-$SLURM_JOB_ID"
export TRANSFORMERS_MARKER="$RUNDIR/.transformers_ok"
mkdir -p "$DEPLOY_DIR/logs" "$RUNDIR" \
         "$DEPLOY_DIR/home/pylib" \
         "$DEPLOY_DIR/cache/huggingface" \
         "$DEPLOY_DIR/cache/triton" \
         "$DEPLOY_DIR/cache/tilelang" \
         "$DEPLOY_DIR/cache/tmp"

# Per-job TMPDIR on scratch (enroot writes UUID dirs here; Triton/LLVM write
# intermediate .o/.ll during JIT). Under $RUNDIR, so it is cleaned at job end.
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
  echo "       Stage an x86_64 (sai-v0.0.6) build, e.g.:" >&2
  echo "         mkdir -p $OTELA_DIR && cp <x86_64 otela> $OTELA_BIN && chmod +x $OTELA_BIN" >&2
  exit 1
}
_sib() { # name, why
  [ -f "$SCRIPT_DIR/$1" ] || {
    echo "FATAL: $1 not found next to this sbatch: $SCRIPT_DIR/$1" >&2
    echo "       $2" >&2
    echo "       Submit with: rcc --profile beverin run --cwd \"$(cd "$SCRIPT_DIR" && pwd)\" -- sbatch $0" >&2
    exit 1
  }
}
_sib gen_correctness.py "The recipe is a 4-file bundle (sbatch + sglang-rocm.toml + sglang_launcher_mi300a.py + gen_correctness.py)."
_sib sglang-rocm.toml "The EDF (image + env). Found via EDF_PATH=$SCRIPT_DIR."
_sib sglang_launcher_mi300a.py "The is_integrated MI300A memory-accounting fix launcher."
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
# The image bundles 5.12.1 (which ships it). Rank 0 does the check + a no-op
# install guard; siblings wait on a marker so concurrent pip never collides.
# If the image somehow lacks glm_moe_dsa and the pinned version is not on the
# public PyPI index, rank 0 exits with a clear message.
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

# Shared Triton + tilelang caches on /capstor/scratch (project 2161, unlimited
# quota). All workers load compiled kernels from the same dirs so only the
# first run pays the LLVM/tilelang JIT cost.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DEPLOY_DIR/cache/triton}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$DEPLOY_DIR/cache/tilelang}"
mkdir -p "$TRITON_CACHE_DIR" "$TILELANG_CACHE_DIR" 2>/dev/null || true
# TMPDIR on scratch (NOT the enroot overlay tmpfs, which competes with model
# memory and hits "Disk quota exceeded" during JIT).
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
# (Same as kimi-k3: the image may carry a CUDA-built aws_ofi_nccl plugin that
#  cannot init on ROCm. Unset the plugin selectors and let RCCL fall back to
#  its built-in Socket transport over hsn0 on Slingshot. Cross-node TP
#  allreduce is slow over Socket; TODO a HIP-aware OFI/CXI plugin.)
unset NCCL_NET_PLUGIN NCCL_NET
export NCCL_SOCKET_IFNAME="${GLM52_NIC:-hsn0}"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-1}"
# MI300A (gfx942) RCCL has no LL (low-latency) cutoff points. For small TP
# collectives (e.g. the 12288-byte input-embedding broadcast on TP8) RCCL
# picks the LL protocol, cannot tune the LL channels, and loops forever in
# pre-adjustment/post-adjustment/minNChannels. SGLang's Python watchdog
# never fires (the hang is C-level, below the heartbeat) — observed in job
# 594097 at ~17:36, stuck 12+ min past the watchdog_timeout deadline.
# glm47-flash (TP4, 4096-byte broadcast, same image) did NOT hit this.
# Force the SIMPLE protocol (ring/tree) for ALL collectives and keep a
# single channel: correctness is unaffected, latency rises only for tiny
# collectives (acceptable for a probe). Override with
# --export=ALL,NCCL_PROTO=LL to re-enable LL for perf experiments.
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
# DSA backends.  The SGLang AMD cookbook uses tilelang (validated on MI300X,
# 192 GB discrete).  On MI300A (128 GB APU, same gfx942 ISA) tilelang crashes
# in forward_c4_indexer during the first forward (job 594304 silent hang at
# 0% GPU for 2.5 h; job 594340 fast SIGQUIT in forward_c4_indexer).  The
# aiter DSA backend is the working PREFILL path on MI300A (jobs 594459,
# 594462, 594527: PP0->PP1->PP2 in 14 s, /health 200 OK, tokens generated)
# but its DECODE path hangs at 100% GPU (page_size=1, SGLANG_USE_AITER=0)
# or deadlocks before the first forward (page_size=64, SGLANG_USE_AITER=1).
# The flashmla_sparse backend (job 594548, PENDING) dispatches to aiter's
# STANDARD mla_prefill_fwd/mla_decode_fwd on AMD — a different path from
# the DSA preshuffle paged-MQA kernel, and the last untested option.
# Override with --export=ALL,GLM52_DSA_PREFILL=<backend>,GLM52_DSA_DECODE=<backend>.
# Valid CLI choices: flashmla_sparse, flashmla_sparse_q8, flashmla_kv,
# flashmla_auto, fa3, tilelang, aiter, trtllm (fa3/trtllm are NVIDIA-only).
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
# MI300A workarounds: disable cuda-graph capture + skip warmup (both crash the
# MTP/nextn layer on MI300A; glm47 lesson).
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

# The is_integrated MI300A memory-accounting fix launcher. It patches
# torch.cuda.get_device_properties to report is_integrated=False (each MI300A
# GPU has its OWN 137 GiB HBM3, not shared sysmem), restoring correct per-GPU
# KV accounting. Rank-agnostic: every node runs its own copy via srun.
exec python3 "$SCRIPT_DIR/sglang_launcher_mi300a.py" "${SGLANG_ARGS[@]}"
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

# Register once the head answers /health AND the probe passes. The wait +
# probe run in the BATCH shell (host netns == container netns, so
# $HEAD:$SERVE_PORT is reachable here), then the otela worker is launched as
# its own step inside the container. The probe is MANDATORY: GLM-5.2 has never
# run a forward on MI300A, and /health returning 200 does NOT prove the DSA
# path can run a forward (deepseek-v4 registered on /health and 502'd every
# request; K3 hung at the forward with /health still up).
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
    # Run INSIDE the sglang-rocm container on the head (same
    # --overlap --environment=$OTELA_EDF_NAME launch as the otela worker): the
    # host's python3 may be too old and the container shares the host netns so
    # 127.0.0.1:$SERVE_PORT reaches SGLang (bound on 0.0.0.0). gen_correctness.py
    # is reused from the kimi-k3 recipe (model-agnostic: six greedy
    # /v1/completions prompts, temperature 0, factual-substring checks).
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
      echo "[$(date -Is)]       For GLM-5.2 a failure here most likely means the DSA" >&2
      echo "[$(date -Is)]       (tilelang/aiter) forward path faults on MI300A gfx942 —" >&2
      echo "[$(date -Is)]       the one new piece vs the proven glm47-flash path." >&2
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
# One SGLang rank per node, inside the container, in the foreground so the job
# lifetime == engine lifetime. srun inherits the exported vars above (HEAD,
# MASTER_PORT, SERVE_PORT, TP_SIZE, SCRIPT_DIR, …) and --environment overlays
# the EDF's [env] (HOME on /capstor, HF_HOME, TRITON_CACHE_DIR, TILELANG_CACHE_DIR,
# HSA_NO_SCRATCH_RECLAIM, SGLANG_USE_AITER, …). --cpus-per-task=NODE_CPUS gives
# each (1 per node) task the whole node (0..191) so the image's
# SGLANG_SET_CPU_AFFINITY NUMA pinning is valid.
NODE_CPUS="${SLURM_CPUS_ON_NODE:-$(scontrol show node "${HEAD}" 2>/dev/null | awk -F= '/^CPUTot=/{print $2; exit}')}"
NODE_CPUS="${NODE_CPUS:-192}"
srun --nodes="$NNODES" --ntasks-per-node=1 \
     --cpus-per-task="$NODE_CPUS" --gpus-per-task=4 \
     --environment="$OTELA_EDF_NAME" bash "$RUNDIR/engine.sh"
SRUN_EXIT=$?
echo "[$(date -Is)] srun_exited code=${SRUN_EXIT}"

exit "$SRUN_EXIT"
