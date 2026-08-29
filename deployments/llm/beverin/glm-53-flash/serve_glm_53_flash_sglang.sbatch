#!/bin/bash
#SBATCH --job-name=glm53-sglang-beverin
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Do NOT set --cpus-per-task here: shrinking the cpuset breaks the image's
# SGLANG_SET_CPU_AFFINITY=1 (set in /etc/environment), whose NUMA pinning
# assumes the full 0..191 logical-CPU cpuset (one MI300A die per TP rank:
#   gpu0->[0-23,96-119] gpu1->[24-47,120-143] gpu2->[48-71,144-167] gpu3->[72-95,168-191]).
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
# B:TERM@120 signals the BATCH shell (B:) 120 s before wall-time, giving the
# stop_otela trap (SMOKE=0 only) a window to scancel the otela step with TERM.
#SBATCH --signal=B:TERM@120
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-53-flash-beverin/logs/%x-%j.out
#
# =============================================================================
# Serve zai-org/GLM-5.3-Flash (FP8 e4m3 ~306 GB, DeepSeek Sparse Attention +
# linear/KDA attention + MTP, MULTIMODAL) on CSCS Beverin (4x MI300A / node,
# x86_64, gfx942) via a PYTHON-SOURCE OVERLAY of the only GLM-5.3 image (the
# Clariden aarch64/CUDA build) onto the upstream ROCm v0.5.18 image — and
# register it on OpenTela.
#
# The overlay (build_overlay.sh, sibling): extracts the Clariden sglang +
# transformers source (pure Python, cp312 — imports on cp310; no ABI) to
# $OVL and prepends them on PYTHONPATH over the v0.5.18 image. The compiled
# torch/sgl_kernel/aiter/tilelang come from the ROCm image; a few skewed
# pure/compiled deps (tokenizers, huggingface_hub, safetensors, regex,
# accelerate) are bumped to clariden transformers 5.16's pins as cp310 wheels
# in $OVL/pkgs310. Import gate VERIFIED (all of sglang, transformers, sgl_kernel,
# torch, sglang.srt.configs.glm5_next, models.glm5_next, models.glm5_next_nextn,
# function_call.glm47_moe_detector, transformers.models.glm5_next, and
# AutoConfig.from_pretrained(GLM-5.3-Flash) load OK on this ROCm container).
#
# Submit (SMOKE default — fast forward-path gate with dummy weights, NO otela):
#   rcc --profile beverin run --cwd "$PWD" -- sbatch serve_glm_53_flash_sglang.sbatch
#
# REAL-WEIGHT SERVE + otela registration (after the smoke passes):
#   rcc --profile beverin run --cwd "$PWD" -- sbatch \
#       --export=ALL,LOAD_FORMAT=auto,SMOKE=0 serve_glm_53_flash_sglang.sbatch
#
# Override the DSA backend (the load-bearing unknown — see below):
#   rcc --profile beverin run --cwd "$PWD" -- sbatch \
#       --export=ALL,GLM53_DSA_PREFILL=aiter,GLM53_DSA_DECODE=aiter \
#       serve_glm_53_flash_sglang.sbatch
#
# NOTE on `rcc run --cwd`: `rcc job submit` runs sbatch from the remote_dir
# root (NOT this directory), so SLURM_SUBMIT_DIR points there and the sibling
# files are not found. `rcc run --cwd <ABSOLUTE recipe dir> -- sbatch <script>`
# makes BASH_SOURCE + SCRIPT_DIR land here. Always use the absolute recipe dir.
#
# What it does, in order:
#   1. preflights the EDF/image, overlay, weights (or LOAD_FORMAT=dummy), otela
#      binary (SMOKE=0 only), and the gen_correctness.py + sglang-rocm.toml +
#      sglang_launcher_mi300a.py siblings.
#   2. srun's ONE SGLang engine rank (1 node, TP4) INSIDE the sglang-rocm
#      container (enroot/EDF). Rank 0 binds 0.0.0.0:$SERVE_PORT (HTTP head).
#   3. once the head's /health answers, runs a MANDATORY generation probe
#      BEFORE any registration — /health alone is not enough (deepseek-v4 / K3 /
#      glm52 lesson: an engine can answer /health while a broken kernel path
#      serves garbage or hangs). gen_correctness.py sends greedy /v1/completions
#      prompts and checks each for its expected substring (smoke: non-empty
#      continuation; real: factual substring with min_pass).
#   4. (SMOKE=0 only) ONLY if the probe passes, starts ONE otela WORKER —
#      inside the container, on the head via srun --overlap --gres=none — which
#      registers the `llm` service. On job end, SIGTERM (never KILL) the otela
#      step so it announces LEFT cleanly.
#
# -----------------------------------------------------------------------------
# THE LOAD-BEARING UNKNOWN: DSA kpool>1 on MI300A (gfx942)
# -----------------------------------------------------------------------------
# GLM-5.3 is MULTIMODAL with dsa_index_kpool > 1. sglang's DeepseekSparseAttnBackend
# has a HARD correctness gate (_check_kpool_tail_backend, raised on EVERY
# forward when kpool>1) that ONLY permits {fa3, tilelang, trtllm}:
#   "index_kpool > 1 appends tail tokens to topk_indices and is currently only
#    supported by the FA3/TileLang/TRTLLM DSA backend."
# On Beverin (ROCm, no flash-attention-3, no TRTLLM), tilelang is the ONLY
# kpool>1-legal backend PRESENT. So GLM53_DSA_PREFILL=GLM53_DSA_DECODE=tilelang
# is forced — it is the ONLY choice that can pass both guards.
#
# THE RISK (carried over from the GLM-5.2 MI300A bring-up, v0.5.16):
#   * tilelang CRASHED in forward_c4_indexer during the first forward on
#     MI300A (job 594304 silent hang at 0% GPU for 2.5 h; job 594340 fast
#     SIGQUIT in forward_c4_indexer). GLM-5.2 worked around this with the
#     aiter backend for PREFILL — but aiter is NOT kpool>1-legal, so that
#     workaround is UNAVAILABLE here (GLM-5.2 is not multimodal kpool>1).
#   * This smoke is the empirical gate: does v0.5.18's tilelang forward_c4_indexer
#     work on MI300A for GLM-5.3's kpool>1 layers? If it faults, GLM-5.3
#     CANNOT run on Beverin with the current upstream stack (no other legal
#     backend exists). Capture the full engine log on the first fault.
# Valid CLI DSA backends: flashmla_sparse, flashmla_sparse_q8, flashmla_kv,
#   flashmla_auto, fa3, tilelang, aiter, trtllm (fa3/trtllm NVIDIA-only;
#   flashmla_*/aiter NOT kpool>1-legal -> will fail _check_kpool_tail_backend).
#
# -----------------------------------------------------------------------------
# GLM-5.3 vs GLM-5.2 (why the args differ)
# -----------------------------------------------------------------------------
#   * MULTIMODAL (Glm5NextForConditionalGeneration) -> needs --mm-attention-backend
#     triton_attn (the vision tower's own dispatch; sm_100/gfx942 auto-defaults
#     to fa3, which is MISSING on ROCm -> ImportError). triton_attn uses Triton
#     kernels, no flash_ops dep.
#   * kpool>1 (see above) -> DSA pinned to tilelang (GLM-5.2's aiter workaround
#     is illegal here).
#   * FP8 e4m3 DYNAMIC (quant_method=fp8 in config.json) -> NOT mxfp4 (no AITER
#     FlyDSL / marlin MXFP4 MoE path). MOE_BACKEND=auto lets the image pick.
#   * MTP nextn=1 -> --disable-cuda-graph + --skip-server-warmup (the MI300A
#     MTP/nextn crash lesson from glm47/glm52; capture crashes cuda-graph).
#   * Parsers: --reasoning-parser glm45 --tool-call-parser glm47 (verified
#     present in the clariden overlay; gated by GLM53_ENABLE_PARSERS).
#
# =============================================================================
set -uo pipefail

# ----------------------------------------------------- script + deploy dirs --
export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
export OTELA_EDF_NAME="${OTELA_EDF_NAME:-sglang-rocm}"

# ---------------------------------------------------------------- deployment --
export DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-53-flash-beverin}"
# The overlay (built once by build_overlay.sh, cached on /capstor). The EDF's
# PYTHONPATH prepends these over the v0.5.18 image defaults.
export OVL="${OVL:-$DEPLOY_DIR/overlay}"
# Clariden aarch64/CUDA build = the only image shipping GLM-5.3 Python. Its
# Python source is extracted into $OVL by build_overlay.sh (it is NOT the
# runtime image — the ROCm v0.5.18 image in the EDF is the runtime).
export OVERLAY_SRC="${OVERLAY_SRC:-/capstor/scratch/cscs/xyao/glm-53-flash/images/sglang-glm-5.3-flash.aarch64.sqsh}"
# GLM-5.3-Flash on /capstor (HF cache). ~306 GB FP8 e4m3. config.json has
# max_position_embeddings=1048576; glm5_next model class resolved from the
# overlay's transformers (5.16.0.dev0), NOT the image's 5.12.
export MODEL_PATH="${MODEL_PATH:-/capstor/scratch/cscs/xyao/models/zai-org/GLM-5.3-Flash}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-5.3-Flash}"

# ------------------------------------------------------------------- shape ----
# 1 node x 4 GPU = TP4 (PP1). GLM-5.3-Flash ~306 GB FP8 / 4 MI300A (137 GiB
# HBM3 each) = ~77 GiB weights/GPU, leaving ~60 GiB for KV + activations +
# JIT. EP_SIZE=TP keeps expert dispatch intra-node (288 routed experts,
# top-8 -> 72 experts/GPU, manageable). For >1 node, set NNODES and PP_SIZE
# (the kimi-k3 distributed pattern); DSA allreduce then crosses Slingshot.
export NNODES="${NNODES:-$SLURM_JOB_NUM_NODES}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-$TP_SIZE}"
# MoE runner backend. GLM-5.3-Flash is FP8 e4m3 DYNAMIC (not mxfp4) -> SGLang's
# native fp8 MoE path is standard; `auto` lets the image pick. If `auto`
# dequantizes to bf16 and OOMs (the kimi-k3 anti-pattern), pin MOE_BACKEND.
export MOE_BACKEND="${MOE_BACKEND:-auto}"

# --- DSA backends (THE kpool>1 gate — see the header block) -----------------
export GLM53_DSA_PREFILL="${GLM53_DSA_PREFILL:-tilelang}"
export GLM53_DSA_DECODE="${GLM53_DSA_DECODE:-tilelang}"
# Multimodal vision-tower attention. GLM-5.3 auto-defaults to fa3 on gfx942,
# which is MISSING on ROCm -> ImportError. Pin triton_attn (Triton kernels,
# no flash_ops dep; VisionTritonAttention in STRIDED_QKV_BACKENDS).
export GLM53_MM_ATTENTION="${GLM53_MM_ATTENTION:-triton_attn}"

# Reasoning + tool-call parsers (verified present in the clariden overlay).
export GLM53_REASONING_PARSER="${GLM53_REASONING_PARSER:-glm45}"
export GLM53_TOOL_CALL_PARSER="${GLM53_TOOL_CALL_PARSER:-glm47}"
export GLM53_ENABLE_PARSERS="${GLM53_ENABLE_PARSERS:-1}"

# Request-length bound + KV pool sizing. GLM-5.3 supports 1M positions; the
# KV pool is sized by MEM_FRAC (not this). 32768 for a fast cold start + probe
# (shorter than kimi-k3's 1M; raise toward 1M once the forward is proven).
export CTX_LEN="${CTX_LEN:-32768}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"

# MI300A workarounds: disable cuda-graph capture + skip warmup (both crash the
# MTP/nextn layer on MI300A; glm47/glm52 lesson).
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
export SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-1}"

# --- networking + ports -----------------------------------------------------
# Slingshot NIC (h<n>; engine.sh pins NCCL_SOCKET_IFNAME=$GLM53_NIC). Single
# node -> no cross-node RCCL; the value only matters for >1 node.
export GLM53_NIC="${GLM53_NIC:-hsn0}"
# Stable ports across the single head (override at submit time if occupied).
export SERVE_PORT="${SERVE_PORT:-30000}"
export MASTER_PORT="${MASTER_PORT:-30011}"

# --- watchdogs --------------------------------------------------------------
export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-3600}"
export DIST_TIMEOUT="${DIST_TIMEOUT:-1800}"
# Health gate: how long to wait for /health before declaring failure. Real
# weights ~13 min cold start (FP8 306 GB on 4 GPU); dummy ~3-5 min (JIT).
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
export GEN_PROBE="${GEN_PROBE:-1}"
export GEN_PROBE_TIMEOUT="${GEN_PROBE_TIMEOUT:-600}"
export GEN_CORRECTNESS_PER_REQ_TIMEOUT="${GEN_CORRECTNESS_PER_REQ_TIMEOUT:-120}"
export GEN_CORRECTNESS_MAX_TOKENS="${GEN_CORRECTNESS_MAX_TOKENS:-64}"
export GEN_CORRECTNESS_MIN_PASS="${GEN_CORRECTNESS_MIN_PASS:-1}"
# GEN_CORRECTNESS_SMOKE=1 (default): the probe accepts any NON-EMPTY
# continuation — proving the full DSA -> HTTP forward pipeline without real
# weights. Set 0 (with LOAD_FORMAT=auto) for the factual-substring gate.
export GEN_CORRECTNESS_SMOKE="${GEN_CORRECTNESS_SMOKE:-1}"
# SMOKE=1 (default): run the forward + probe, do NOT register on otela
# (leaves the mesh clean while you iterate on the DSA path). Set SMOKE=0
# (with LOAD_FORMAT=auto) for the full serve + otela registration.
export SMOKE="${SMOKE:-1}"

# --- load format ------------------------------------------------------------
# LOAD_FORMAT=dummy (default): serve random weights for a fast forward-path
# smoke (no ~13 min real-weight cold start). Set LOAD_FORMAT=auto for real.
export LOAD_FORMAT="${LOAD_FORMAT:-dummy}"

# ------------------------------------------------------------------- otela ----
export OTELA_DIR="${OTELA_DIR:-/capstor/scratch/cscs/xyao/opentela}"
export OTELA_BIN="${OTELA_BIN:-$OTELA_DIR/otela}"   # x86_64 build (not -arm64)
export OTELA_RELAY_ADDR="${OTELA_RELAY_ADDR:-/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7}"
export OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
export OTELA_TCP_PORT="${OTELA_TCP_PORT:-43905}"
export OTELA_UDP_PORT="${OTELA_UDP_PORT:-59820}"
export OTELA_SEED="${OTELA_SEED:-$RANDOM$RANDOM}"
export OTELA_STEP_NAME="${OTELA_STEP_NAME:-otela}"

# ---------------------------------------------------------------- dirs/HEAD --
mkdir -p "$DEPLOY_DIR" "$DEPLOY_DIR/logs" "$DEPLOY_DIR/cache/triton" \
         "$DEPLOY_DIR/cache/tilelang" "$DEPLOY_DIR/cache/tmp" \
         "$DEPLOY_DIR/home" "$DEPLOY_DIR/cache/hf" 2>/dev/null || true
export RUNDIR="$DEPLOY_DIR/run/${SLURM_JOB_ID:-$$}"
mkdir -p "$RUNDIR"
export HEAD="${SLURM_JOB_NODELIST:-$(hostname)}"
export HEAD="${HEAD%\[*}"   # strip any node-array suffix -> first node
export HEAD_IP="$(getent hosts "$HEAD" 2>/dev/null | awk '{print $1; exit}')"
export HEAD_IP="${HEAD_IP:-127.0.0.1}"

echo "[$(date -Is)] started=$(date --iso-8601=seconds) smoke=$SMOKE load_format=$LOAD_FORMAT"
echo "[$(date -Is)] parsers=$GLM53_REASONING_PARSER/$GLM53_TOOL_CALL_PARSER/$GLM53_ENABLE_PARSERS gen_probe=$GEN_PROBE smoke_probe=$GEN_CORRECTNESS_SMOKE min_pass=$GEN_CORRECTNESS_MIN_PASS"
echo "[$(date -Is)] DSA=$GLM53_DSA_PREFILL/$GLM53_DSA_DECODE mm=$GLM53_MM_ATTENTION tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE ctx=$CTX_LEN mem=$GPU_MEM_UTIL"

# ----------------------------------------------------------------- preflight --
# SMOKE=0 requires otela; SMOKE=1 (default) skips the otela binary check so a
# smoke can run before otela is even staged.
if [ "$SMOKE" = "0" ]; then
  [ -x "$OTELA_BIN" ] || {
    echo "FATAL: otela binary not found/executable: $OTELA_BIN" >&2
    echo "       Stage an x86_64 build (not -arm64), e.g.:" >&2
    echo "         cp <x86_64 otela> $OTELA_BIN && chmod +x $OTELA_BIN" >&2
    exit 1
  }
fi
# LOAD_FORMAT=dummy skips the weight dir (fast forward-path smoke before the
# ~13 min real-weight cold start). auto requires the model dir + config.json.
if [ "$LOAD_FORMAT" != "dummy" ]; then
  [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ] || {
    echo "FATAL: weights/config not found: $MODEL_PATH" >&2; exit 1; }
fi
# The overlay (GLM-5.3 Python). build_overlay.sh (sibling) builds it once; the
# EDF PYTHONPATH will fail to import sglang/transformers without it.
if [ ! -f "$OVL/sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py" ] || \
   [ ! -f "$OVL/sgl-workspace/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py" ]; then
  echo "FATAL: GLM-5.3 overlay not found under $OVL" >&2
  echo "       Build it once from the login node: bash $SCRIPT_DIR/build_overlay.sh" >&2
  exit 1
fi
[ -f "$OVL/pkgs310/tokenizers/__init__.py" ] || {
  echo "FATAL: pkgs310 deps missing under $OVL/pkgs310 (tokenizers)." >&2
  echo "       Re-run: bash $SCRIPT_DIR/build_overlay.sh" >&2; exit 1; }
_sib() { # name, why
  [ -f "$SCRIPT_DIR/$1" ] || {
    echo "FATAL: $1 not found next to this sbatch: $SCRIPT_DIR/$1" >&2
    echo "       $2" >&2; exit 1; }
}
_sib gen_correctness.py "The recipe is a 4-file bundle (sbatch + sglang-rocm.toml + sglang_launcher_mi300a.py + gen_correctness.py) + build_overlay.sh."
_sib sglang-rocm.toml "The EDF (v0.5.18 image + overlay PYTHONPATH + MI300A env). Found via EDF_PATH=$SCRIPT_DIR."
_sib sglang_launcher_mi300a.py "The is_integrated MI300A memory-accounting fix launcher."

# Quick in-container preflight: the image + overlay actually import the GLM-5.3
# model (catches a stale/wrong overlay BEFORE the ~5 min srun cold start). ~20 s.
echo "[$(date -Is)] container preflight: import sglang + glm5_next + AutoConfig ..."
srun --nodes="$NNODES" --ntasks-per-node=1 --cpus-per-task=2 --gpus-per-task=0 \
     --environment="$OTELA_EDF_NAME" -N1 -n1 -t0:05:00 \
  python3 - <<'PREFLIGHT' || { echo "[$(date -Is)] PREFLIGHT FAILED — see above" >&2; exit 1; }
import importlib, sys
ok, fail = [], []
def chk(name, imp="import "):
    try:
        m = importlib.import_module(name)
        ok.append(name)
        return m
    except Exception as e:
        fail.append(f"{name}: {type(e).__name__}: {e}")
chk("sglang"); chk("transformers")
chk("sglang.srt.configs.glm5_next")
chk("sglang.srt.models.glm5_next")
chk("sglang.srt.models.glm5_next_nextn")
chk("sglang.srt.function_call.glm47_moe_detector")
chk("transformers.models.glm5_next")
from transformers import AutoConfig
import os
cfg = AutoConfig.from_pretrained(os.environ["MODEL_PATH"], trust_remote_code=True)
ok.append(f"AutoConfig({os.environ['MODEL_PATH']!r}) -> {type(cfg).__name__}")
print("PREFLIGHT OK:", *ok, sep="\n  ")
if fail:
    print("PREFLIGHT FAIL:", *fail, sep="\n  ", file=sys.stderr); sys.exit(1)
PREFLIGHT
echo "[$(date -Is)] preflight OK"

# ================================================================ engine ======
cat > "$RUNDIR/engine.sh" <<'ENGINE'
#!/bin/bash
set -uo pipefail
RANK="${SLURM_NODEID:-0}"
NODE="$(hostname)"
echo "[rank $RANK] $(hostname) node-rank=$RANK dist-init=$HEAD:$MASTER_PORT serve=$SERVE_PORT ctx=$CTX_LEN tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE"

# --- MI300A + RCCL environment ----------------------------------------------
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"   # mandatory MI300A
# Shared Triton + tilelang caches on /capstor (all workers load the same
# compiled kernels; only the first run pays the LLVM/tilelang JIT cost).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DEPLOY_DIR/cache/triton}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$DEPLOY_DIR/cache/tilelang}"
mkdir -p "$TRITON_CACHE_DIR" "$TILELANG_CACHE_DIR" 2>/dev/null || true
export TMPDIR="${TMPDIR:-$DEPLOY_DIR/cache/tmp}"; export TMP="$TMPDIR"
mkdir -p "$TMPDIR" 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"   # single-thread JIT keeps mem low
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-65536}"
# Drop the image's CUDA-built aws_ofi_nccl plugin (cannot init on ROCm); let
# RCCL fall back to its built-in Socket transport over hsn0 on Slingshot.
# (Single-node TP4 runs no cross-node RCCL, but keep this consistent with the
#  >1-node path for when PP is raised.)
unset NCCL_NET_PLUGIN NCCL_NET
export NCCL_SOCKET_IFNAME="${GLM53_NIC:-hsn0}"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_PROTO="${NCCL_PROTO:-SIMPLE}"       # MI300A gfx942: avoid the LL hang
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# --- build the SGLang serve args (rank 0 = HTTP head) -----------------------
SGLANG_ARGS=(
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --host 0.0.0.0 --port "$SERVE_PORT"
  --tp-size "$TP_SIZE"
  --pp-size "$PP_SIZE"
  --ep-size "$EP_SIZE"
  --context-length "$CTX_LEN"
  --mem-fraction-static "$GPU_MEM_UTIL"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --watchdog-timeout "$WATCHDOG_TIMEOUT"
  --dist-timeout "$DIST_TIMEOUT"
  --enable-metrics
)
# Distributed (NNODES>1): one engine rank per node via srun --ntasks-per-node=1,
# so add the torchrun store coords. Single-node (NNODES=1) omits these — sglang
# spawns the 4 local TP workers itself with a 127.0.0.1 store (default).
if [ "${NNODES:-1}" -gt 1 ]; then
  SGLANG_ARGS+=(
    --nnodes "$NNODES" --node-rank "$RANK"
    --dist-init-addr "${HEAD_IP:-127.0.0.1}:$MASTER_PORT"
  )
fi
# --- DSA backends (THE kpool>1 gate) ----------------------------------------
# tilelang is the ONLY kpool>1-legal backend present on ROCm (fa3/trtllm
# NVIDIA-only; flashmla_*/aiter fail _check_kpool_tail_backend). Override with
# --export=ALL,GLM53_DSA_PREFILL=<b>,GLM53_DSA_DECODE=<b> to experiment (any
# non-tilelang value will fail the kpool>1 hard gate on the first forward).
SGLANG_ARGS+=(
  --dsa-prefill-backend "$GLM53_DSA_PREFILL"
  --dsa-decode-backend "$GLM53_DSA_DECODE"
)
# --- multimodal vision-tower attention (fa3 MISSING on ROCm) ----------------
SGLANG_ARGS+=( --mm-attention-backend "$GLM53_MM_ATTENTION" )
# --- reasoning + tool-call parsers (verified in the clariden overlay) -------
if [ "${GLM53_ENABLE_PARSERS:-1}" = "1" ]; then
  SGLANG_ARGS+=(
    --reasoning-parser "$GLM53_REASONING_PARSER"
    --tool-call-parser "$GLM53_TOOL_CALL_PARSER"
  )
fi
# --- MI300A workarounds: MTP/nextn crashes cuda-graph capture ---------------
[ "${DISABLE_CUDA_GRAPH:-1}" = "1" ] && SGLANG_ARGS+=(--disable-cuda-graph)
[ "${SKIP_SERVER_WARMUP:-1}" = "1" ] && SGLANG_ARGS+=(--skip-server-warmup)
# --- load format (dummy = fast smoke; auto = real weights) ------------------
[ -n "${LOAD_FORMAT:-}" ] && [ "${LOAD_FORMAT:-}" != "auto" ] && \
  SGLANG_ARGS+=(--load-format "$LOAD_FORMAT")

echo "[rank $RANK] launching SGLang (head=HTTP+probe$([ "$SMOKE" = 0 ] && echo "+otela")) args=${SGLANG_ARGS[*]}"

# The is_integrated MI300A memory-accounting fix launcher (patches
# torch.cuda.get_device_properties.is_integrated=False so each MI300A GPU's
# 137 GiB HBM3 is accounted per-GPU, not as shared sysmem).
exec python3 "$SCRIPT_DIR/sglang_launcher_mi300a.py" "${SGLANG_ARGS[@]}"
ENGINE
chmod +x "$RUNDIR/engine.sh"

# ============================================================ otela worker ===
# (SMOKE=0 only — the smoke does NOT register on otela, leaving the mesh clean
#  while you iterate on the DSA forward path.) Runs INSIDE the container on the
#  HEAD only (one peer for the single distributed head endpoint); the container
#  shares the host netns so otela on the head reaches SGLang on localhost.
_OTELA_CFG_DIR="$RUNDIR/otela"
if [ "$SMOKE" = "0" ]; then
  mkdir -p "$_OTELA_CFG_DIR"
  cat > "$_OTELA_CFG_DIR/cfg.yaml" <<CFG
name: beverin-glm53-$SLURM_JOB_ID
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
  --config "$_OTELA_CFG_DIR/cfg.yaml" \\
  --bootstrap.static "$OTELA_RELAY_ADDR" \\
  --service.name "$OTELA_SERVICE_NAME" \\
  --service.port "$SERVE_PORT" \\
  --label "model=$SERVED_MODEL_NAME" \\
  --label "cluster=beverin" \\
  --label "framework=sglang" \\
  --label "launched_by=\${USER:-unknown}"
OTELA
  chmod +x "$RUNDIR/otela.sh"
fi

# --- lifecycle: clean otela leave on SIGTERM (SMOKE=0 only) -----------------
_OTELA_STOPPED=0
stop_otela() {
  [ "$SMOKE" = "1" ] && return 0
  [ "$_OTELA_STOPPED" = 1 ] && return 0
  _OTELA_STOPPED=1
  local step
  step="$(squeue -s -j "$SLURM_JOB_ID" -h -o '%i %j' 2>/dev/null \
          | awk -v n="$OTELA_STEP_NAME" '$2==n{print $1}' | head -1)"
  if [ -z "$step" ]; then echo "[$(date -Is)] WARN: otela step not found; no LEFT published." >&2; return 0; fi
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

# --------------------------------------------------------- probe + register --
# Wait for /health, run the MANDATORY generation probe (the gate that /health
# alone cannot prove — a broken DSA kernel can answer /health and 502/hang
# every request), then start the otela worker (SMOKE=0 only). The wait + probe
# run in the BATCH shell (host netns == container netns -> 127.0.0.1:$SERVE_PORT
# reaches SGLang); the probe runs INSIDE the container via srun --overlap so
# the container's python3 (cp310, not the login's 3.6) loads gen_correctness.py.
(
  deadline=$(( SECONDS + HEALTH_TIMEOUT ))
  until curl -sf --max-time 5 "http://127.0.0.1:$SERVE_PORT/health" >/dev/null 2>&1; do
    [ $SECONDS -ge $deadline ] && { echo "[$(date -Is)] health timeout after ${HEALTH_TIMEOUT}s; not registering" >&2; exit 1; }
    sleep 20
  done
  echo "[$(date -Is)] engine healthy after ~$((HEALTH_TIMEOUT - (deadline - SECONDS)))s"

  if [ "$GEN_PROBE" = "1" ]; then
    echo "[$(date -Is)] generation ${GEN_CORRECTNESS_SMOKE:+smoke }probe (model=$SERVED_MODEL_NAME, max_tokens=$GEN_CORRECTNESS_MAX_TOKENS, min_pass=$GEN_CORRECTNESS_MIN_PASS, budget=${GEN_PROBE_TIMEOUT}s) ..."
    PROBE_OUT="$RUNDIR/gen_correctness_${SLURM_JOB_ID:-$$}.json"
    GEN_CORRECTNESS_TIMEOUT="$GEN_PROBE_TIMEOUT" \
    GEN_CORRECTNESS_PER_REQ_TIMEOUT="$GEN_CORRECTNESS_PER_REQ_TIMEOUT" \
    GEN_CORRECTNESS_SMOKE="$GEN_CORRECTNESS_SMOKE" \
      timeout "$((GEN_PROBE_TIMEOUT + 300))" \
      srun --jobid="${SLURM_JOB_ID:-0}" --overlap --gres=none --nodes=1 --ntasks=1 \
           -w "$HEAD" --environment="$OTELA_EDF_NAME" \
           python3 "$SCRIPT_DIR/gen_correctness.py" \
               "http://127.0.0.1:$SERVE_PORT" "$SERVED_MODEL_NAME" \
               "$PROBE_OUT" "$GEN_CORRECTNESS_MAX_TOKENS" "$GEN_CORRECTNESS_MIN_PASS"
    rc=$?
    if [ "$rc" = 0 ]; then
      echo "[$(date -Is)] GEN_PROBE_OK rc=$rc (report in $PROBE_OUT)"
    else
      echo "[$(date -Is)] GEN_PROBE_FAILED rc=$rc; NOT registering (report in $PROBE_OUT)" >&2
      echo "[$(date -Is)]       For GLM-5.3 a failure here most likely means the" >&2
      echo "[$(date -Is)]       tilelang DSA forward (kpool>1) faults on MI300A gfx942 —" >&2
      echo "[$(date -Is)]       the ONE new piece. Inspect $PROBE_OUT + the engine log." >&2
      exit 1
    fi
  fi

  if [ "$SMOKE" = "0" ]; then
    echo "[$(date -Is)] starting otela worker on $HEAD (inside container, EDF=$OTELA_EDF_NAME)"
    srun --jobid="${SLURM_JOB_ID:-0}" --overlap --gres=none --nodes=1 --ntasks=1 \
         -w "$HEAD" --environment="$OTELA_EDF_NAME" -J "$OTELA_STEP_NAME" \
         bash -c '"$0" >"$1" 2>&1' "$RUNDIR/otela.sh" "$RUNDIR/otela.log"
  else
    echo "[$(date -Is)] SMOKE=1: probe passed; engine held by the foreground srun below for manual inspection (no otela registration)."
    echo "[$(date -Is)]       curl http://127.0.0.1:$SERVE_PORT/v1/models ; scancel $SLURM_JOB_ID when done."
    # The foreground srun (below) owns the engine lifetime; this subshell's
    # job was only the gate, so it can exit cleanly here.
  fi
) &

# ------------------------------------------------------------- run engine ----
# One SGLang rank on the single node, inside the container, in the foreground
# so the job lifetime == engine lifetime. srun inherits the exported vars
# (HEAD, MASTER_PORT, SERVE_PORT, TP_SIZE, …) and --environment overlays the
# EDF's [env] (overlay PYTHONPATH, HOME on /capstor, TRITON_CACHE_DIR,
# TILELANG_CACHE_DIR, HSA_NO_SCRATCH_RECLAIM, SGLANG_USE_AITER, …).
NODE_CPUS="${SLURM_CPUS_ON_NODE:-$(scontrol show node "$HEAD" 2>/dev/null | awk -F= '/^CPUTot=/{print $2; exit}')}"
NODE_CPUS="${NODE_CPUS:-192}"
srun --nodes="$NNODES" --ntasks-per-node=1 \
     --cpus-per-task="$NODE_CPUS" --gpus-per-task=4 \
     --environment="$OTELA_EDF_NAME" bash "$RUNDIR/engine.sh"
SRUN_EXIT=$?
echo "[$(date -Is)] srun_exited code=${SRUN_EXIT}"

exit "$SRUN_EXIT"
