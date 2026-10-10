#!/bin/bash
#SBATCH --job-name=glm53-sglang-beverin
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# No --cpus-per-task here: shrinking the cpuset breaks SGLANG_SET_CPU_AFFINITY=1
# NUMA pinning (needs the full 0..191 cpuset, one MI300A die per TP rank).
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
# nid002922 excluded: repeatedly failed mid-run with an unrecoverable RCCL/
# HSA fault during the glm53-otela jobs (see comp-capture campaign logs on
# /capstor); re-try after the next maintenance window.
#SBATCH --exclude=nid002922
# B:TERM@120 signals the BATCH shell (B:) 120 s before wall-time, giving the
# stop_otela trap (SMOKE=0 only) a window to scancel the otela step with TERM.
#SBATCH --signal=B:TERM@120
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-53-flash-beverin/logs/%x-%j.out
#
# =============================================================================
# Serve zai-org/GLM-5.3-Flash (FP8 e4m3 ~306 GB, DSA + linear/KDA + MTP,
# MULTIMODAL) on CSCS Beverin (4x MI300A / node, gfx942) via a PYTHON-SOURCE
# OVERLAY of the only GLM-5.3 image (Clariden aarch64/CUDA build) onto the
# upstream ROCm v0.5.18 image — and register it on OpenTela.
#
# The overlay (build_overlay.sh, sibling): pure-Python sglang+transformers
# sources from the Clariden image prepended on PYTHONPATH over the ROCm image;
# skewed pure/compiled deps shipped as cp310 wheels in $OVL/pkgs310. Import
# gate verified (see the in-container preflight below). Full story: README.
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
# NOTE: `rcc job submit` submits from the remote_dir root and siblings are not
# found — use `rcc run --cwd <ABSOLUTE recipe dir>` (up.sh does).
#
# Flow: preflight (EDF/overlay/weights/otela + siblings) → ONE TP4 engine rank
# in the sglang-rocm container → MANDATORY gen probe BEFORE registration
# (deepseek-v4/K3/glm52 lesson: /health can pass while the kernel path serves
# garbage or hangs) → (SMOKE=0 only) ONE otela worker on the head → SIGTERM
# on job end (LEFT).
#
# -----------------------------------------------------------------------------
# THE LOAD-BEARING UNKNOWN: DSA kpool>1 on MI300A (gfx942)
# -----------------------------------------------------------------------------
# GLM-5.3 is MULTIMODAL with dsa_index_kpool > 1: sglang's hard gate
# (_check_kpool_tail_backend, every forward) only permits {fa3, tilelang,
# trtllm}; on ROCm tilelang is the ONLY one present -> DSA is FORCED to
# tilelang. GLM-5.2's aiter-prefill workaround is NOT kpool>1-legal here.
# Boot-blocker chain (612093 deep_gemm NameError -> 612201 MHC smem abort ->
# 612262 TVM FloorMod -> vkernels PR #52 HIP kernel) + the kpool gate:
# README "Boot-blocker history" + "The kpool>1 gate".
#
# GLM-5.3 vs GLM-5.2 arg differences (mm-attention, FP8-dynamic MoE, chat
# template, bf16-gemm backend): README "GLM-5.3 vs GLM-5.2".
# =============================================================================
set -uo pipefail

# ----------------------------------------------------- script + deploy dirs --
export SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
export OTELA_EDF_NAME="${OTELA_EDF_NAME:-sglang-rocm}"

# ---------------------------------------------------------------- deployment --
export DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-53-flash-beverin}"
# Overlay built once by build_overlay.sh; EDF PYTHONPATH prepends it over the
# v0.5.18 image defaults.
export OVL="${OVL:-$DEPLOY_DIR/overlay}"
# Clariden aarch64/CUDA build = the only image shipping GLM-5.3 Python; its
# source is extracted into $OVL (NOT the runtime image — the EDF's ROCm image
# is). Build it once: bash build_overlay.sh (stages from the aarch64 .sqsh).
# ~306 GB FP8 e4m3; glm5_next resolves from the OVERLAY's transformers
# (5.16.0.dev0), not the image's 5.12.
export MODEL_PATH="${MODEL_PATH:-/capstor/scratch/cscs/xyao/models/zai-org/GLM-5.3-Flash}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-5.3-Flash}"

# ------------------------------------------------------------------- shape ----
# TP4 (PP1): ~306 GB FP8 / 4 MI300A = ~77 GiB weights/GPU. EP_SIZE=TP keeps
# expert dispatch intra-node. >1 node: set NNODES + PP_SIZE (kimi-k3 pattern).
export NNODES="${NNODES:-$SLURM_JOB_NUM_NODES}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-$TP_SIZE}"
# FP8 e4m3 DYNAMIC (not mxfp4) -> native fp8 MoE path; GLM53_MOE_RUNNER=auto
# lets the image pick (pin triton/marlin if auto dequantizes to bf16 and OOMs).

# --- DSA backends (THE kpool>1 gate — see the header block) -----------------
export GLM53_DSA_PREFILL="${GLM53_DSA_PREFILL:-tilelang}"
export GLM53_DSA_DECODE="${GLM53_DSA_DECODE:-tilelang}"
# DSA top-k TRANSFORM (third DSA path, after #51 scorer + PR #52 fwd):
# sgl-kernel's default JIT #includes <cuda_fp16.h> (absent on ROCm) and
# SIGQUITs the engine; 'torch' = pure-torch fallback (needs
# SGLANG_DSA_FUSE_TOPK=0, set below — sglang only auto-forces that under
# deterministic inference).
export GLM53_DSA_TOPK_BACKEND="${GLM53_DSA_TOPK_BACKEND:-torch}"
# BUT the active GLM-5.3 kpool path calls the TVM JIT unconditionally — the
# ACTUAL torch bridge (vkernels #56) is the sitecustomize rebind, gated here:
export GLM53_TOPK_TRANSFORM_BACKEND="${GLM53_TOPK_TRANSFORM_BACKEND:-torch}"
# Pure torch, 7/7 parity (test_vkernels_dsa_topk.py), not graph-safe; =sgl
# keeps the JIT as the control.
export SGLANG_DSA_FUSE_TOPK="${SGLANG_DSA_FUSE_TOPK:-0}"
# auto pick (aiter) JIT-hangs ~52 min on gfx942 for the fp8-e4m3 ck2stages
# kernel; triton bypasses it until aiter is fixed.
export GLM53_MOE_RUNNER="${GLM53_MOE_RUNNER:-triton}"
# Mamba SSM backend: triton (default; stochastic-rounding sub-path needs
# SM100, base scan does not) | flashinfer (isolates the SSM scan from KDA for
# the job-612826 stall hunt; KDA is triton-only on gfx942).
export GLM53_MAMBA_BACKEND="${GLM53_MAMBA_BACKEND:-triton}"
# Forward-probe: logs layer/op ENTER/EXIT on the first forward — the last
# line before a silent stall names the hanging op. Default ON (campaign);
# GLM53_FWD_PROBE=0 for production.
export GLM53_FWD_PROBE="${GLM53_FWD_PROBE:-1}"
# Shared diag harness — identical copies on beverin + clariden (one source of
# truth). SLURM_SUBMIT_DIR-first: spooled batch scripts may not resolve
# BASH_SOURCE (job 625057 resolved to /meta/diag/glm53).
export GLM53_DIAG_DIR="${GLM53_DIAG_DIR:-$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/../../../.." && pwd)/meta/diag/glm53}"
# Component/layer I/O capture: first-forward-only, layer 0 starts from the
# deterministic embedding so the bisect vs pure-torch is meaningful.
# DIAGNOSTIC DEFAULTS: remove the =1 once the bisect closes.
export GLM53_COMP_CAPTURE="${GLM53_COMP_CAPTURE:-1}"
export GLM53_COMP_MODE="${GLM53_COMP_MODE:-components}"
export GLM53_COMP_LAYER="${GLM53_COMP_LAYER:-0}"
export GLM53_COMP_TAG="${GLM53_COMP_TAG:-beverin_l0}"
export GLM53_COMP_DIR="${GLM53_COMP_DIR:-/capstor/scratch/cscs/xyao/glm-53-flash-beverin/comp_capture}"
# tilelang DSA-topk JIT probe + DEBUG log: localizes the fp8_paged_mqa_logits
# JIT hang on gfx942. Campaign default; remove after diagnosis.
export GLM53_TILELANG_PROBE="${GLM53_TILELANG_PROBE:-1}"
export GLM53_TL_LOG="${GLM53_TL_LOG:-DEBUG}"
# vkernels HIP lib (PR #52). MUST be the container-ROCm build (hip-roc72,
# ROCm 7.2): the login-node build/hip links libamdhip64.so.6 (absent in the
# container) and fails dlopen. sitecustomize.py ($OVL/pylib) rebinds sglang's
# tilelang_sparse_fwd -> vk_hip_dsa_sparse_fwd on gfx942 (README PR #52).
export VKERNELS_LIB="${VKERNELS_LIB:-/capstor/scratch/cscs/xyao/vkernels/build/hip-roc72/src/c/libvkernels_hip.so}"
# Vision tower auto-defaults to fa3 (missing on ROCm) -> ImportError; pin
# triton_attn (no flash_ops dep).
export GLM53_MM_ATTENTION="${GLM53_MM_ATTENTION:-triton_attn}"

# bf16 GEMMs (MLA projections) routed to rocBLAS: with aiter on, sglang sent
# every bf16 GEMM through the tuned-GEMM dispatcher, which has no config for
# M:1 decode shapes and SIGKILLs (job 613339) — README "GLM-5.3 vs GLM-5.2".
export GLM53_BF16_GEMM_BACKEND="${GLM53_BF16_GEMM_BACKEND:-torch}"

# Reasoning + tool-call parsers (verified present in the clariden overlay).
export GLM53_REASONING_PARSER="${GLM53_REASONING_PARSER:-glm45}"
export GLM53_TOOL_CALL_PARSER="${GLM53_TOOL_CALL_PARSER:-glm47}"
export GLM53_ENABLE_PARSERS="${GLM53_ENABLE_PARSERS:-1}"

# 32768 for a fast cold start + probe (1M supported; raise once proven).
export CTX_LEN="${CTX_LEN:-32768}"
# APU memory math (unified ~513 GB pool; 0.88 OOMs the JIT round, 0.80 OOMs
# the HOST side on cold starts — job 621559): README "APU memory math".
# 0.70 restores headroom; override per-deploy (0.75 for prod).
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"

# skip-warmup stays on (MTP warmup crash). CUDA graphs ON: capture-time fa3
# ImportError is routed to tilelang (PR #52 shim) by a patched resolver —
# capture is the real per-step perf win.
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
export SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-1}"

# --- networking + ports -----------------------------------------------------
# Slingshot NIC; only matters >1 node (no cross-node RCCL at TP4/PP1).
export GLM53_NIC="${GLM53_NIC:-hsn0}"
# Stable ports across the single head (override at submit time if occupied).
export SERVE_PORT="${SERVE_PORT:-30000}"
export MASTER_PORT="${MASTER_PORT:-30011}"

# --- watchdogs --------------------------------------------------------------
export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-3600}"
export DIST_TIMEOUT="${DIST_TIMEOUT:-1800}"
# Real weights ~13 min cold start (FP8 306 GB); dummy ~3-5 min (JIT).
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
# SMOKE=1 (default): forward + probe, NO otela registration (mesh stays clean
# while iterating). SMOKE=0 (+ LOAD_FORMAT=auto): full serve + registration.
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
echo "[$(date -Is)] DSA=$GLM53_DSA_PREFILL/$GLM53_DSA_DECODE topk=$GLM53_DSA_TOPK_BACKEND fuse_topk=$SGLANG_DSA_FUSE_TOPK mm=$GLM53_MM_ATTENTION mamba=$GLM53_MAMBA_BACKEND moe=$GLM53_MOE_RUNNER probe=$GLM53_FWD_PROBE tlp=$GLM53_TILELANG_PROBE tl=$GLM53_TL_LOG tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE ctx=$CTX_LEN mem=$GPU_MEM_UTIL"

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
# Shared scripts (gen_correctness.py, sglang_launcher_mi300a.py) are
# canonical in the repo's meta/common/ — walk up from the recipe dir.
COMMON_DIR=""
_cwd="$SCRIPT_DIR"
for _ in 1 2 3 4 5 6; do
  [ -d "$_cwd/meta/common" ] && { COMMON_DIR="$_cwd/meta/common"; break; }
  _cwd="$(dirname "$_cwd")"
done
[ -n "$COMMON_DIR" ] || { echo "FATAL: meta/common/ not found above $SCRIPT_DIR — submit from within the repo checkout" >&2; exit 1; }
export COMMON_DIR   # reaches the container via srun-inherited env
unset _cwd
[ -f "$COMMON_DIR/gen_correctness.py" ] || { echo "FATAL: $COMMON_DIR/gen_correctness.py missing" >&2; exit 1; }
[ -f "$COMMON_DIR/sglang_launcher_mi300a.py" ] || { echo "FATAL: $COMMON_DIR/sglang_launcher_mi300a.py missing" >&2; exit 1; }
_sib sglang-rocm.toml "The EDF (v0.5.18 image + overlay PYTHONPATH + MI300A env). Found via EDF_PATH=$SCRIPT_DIR."
[ -f "$GLM53_DIAG_DIR/capture_probe.py" ] || {
  echo "FATAL: capture_probe.py not found in $GLM53_DIAG_DIR" >&2
  echo "       (the shared diag harness lives in <cookbook>/meta/diag/glm53; submit from a full cookbook checkout)" >&2; exit 1; }

# Quick in-container preflight: the image + overlay actually import the GLM-5.3
# model (catches a stale/wrong overlay BEFORE the ~5 min srun cold start). ~20 s
# on a warm TILELANG/AITER cache; the first run on a cold node can take
# several minutes (tilelang/TVM JIT) so the gate is 15 min (one-time on /capstor).
echo "[$(date -Is)] container preflight: import sglang + glm5_next + AutoConfig ..."
srun --nodes="$NNODES" --ntasks-per-node=1 --cpus-per-task=2 --gpus-per-task=0 \
     --environment="$OTELA_EDF_NAME" -N1 -n1 -t0:15:00 \
  python3 -u - <<'PREFLIGHT' || { echo "[$(date -Is)] PREFLIGHT FAILED — see above" >&2; exit 1; }
import time, importlib, sys
_t0=time.time()
ok, fail = [], []
def chk(name, imp="import "):
    _s=time.time()
    try:
        m = importlib.import_module(name)
        ok.append(name); print(f"  [preflight t={time.time()-_t0:.1f}s] +{name} ({time.time()-_s:.1f}s)", flush=True)
        return m
    except Exception as e:
        fail.append(f"{name}: {type(e).__name__}: {e}"); print(f"  [preflight t={time.time()-_t0:.1f}s] FAIL {name} ({time.time()-_s:.1f}s): {e}", flush=True)
chk("sglang"); chk("transformers")
chk("sglang.srt.configs.glm5_next")
chk("sglang.srt.models.glm5_next")
chk("sglang.srt.models.glm5_next_nextn")
chk("sglang.srt.function_call.glm47_moe_detector")
chk("transformers.models.glm5_next")
from transformers import AutoConfig
import os
_s=time.time(); cfg = AutoConfig.from_pretrained(os.environ["MODEL_PATH"], trust_remote_code=True); print(f"  [preflight t={time.time()-_t0:.1f}s] AutoConfig ({time.time()-_s:.1f}s) -> {type(cfg).__name__}", flush=True)
ok.append(f"AutoConfig({os.environ['MODEL_PATH']!r}) -> {type(cfg).__name__}")
print("PREFLIGHT OK:", *ok, sep="\n  ", flush=True)
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
# GLM-5.3's MHC hybrid-attention pre-norm has two deep_gemm paths (mhc.py
# 850/1450), gated by this flag. deep_gemm is CUDA-only and NOT imported on
# ROCm (the name is unbound), so the default EnvBool(True) -> NameError on the
# first forward (job 612093). =0 takes the Triton/tilelang fallbacks (no
# deep_gemm). See sglang-rocm.toml for the full rationale.
export SGLANG_OPT_DEEPGEMM_HC_PRENORM="${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-0}"
# That =0 selects tilelang's MHC pre-norm kernels (mhc.py: mhc_pre_gemm_sqrsum_
# {splitk_kernel,tilelang}). They request 98304 B dynamic shared at the default
# hidden_block=256 + num_stages=2 (x_smem 16 KB bf16 + fn_smem 32 KB fp32,
# doubled). MI300A (gfx942) has only 64 KB *non-optin* shared/block, and
# tilelang's HIP codegen (jit/adapter/wrapper.py PREDEF_ATTRIBUTE_SET_DYNAMIC_
# MEMORY_HIP) only QUERIES hipDeviceAttributeMaxSharedMemoryPerBlock and aborts
# if the request exceeds it — it never calls hipFuncSetAttribute to opt in to
# the larger (~228 KB) dynamic cap (a known tilelang HIP gap; the CUDA template
# PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY does opt in). So the JIT init() aborted:
# "Requested dynamic shared memory 98304 exceeds device limit 65536" (job 612201).
# Fix (applied by build_overlay.sh via tilelang-mhc-reduce-hidden_block-for-
# mi300a-64KB-LDS.patch to the overlay's mhc.py): hidden_block 256->128 in BOTH
# the num_tokens<=2048 splitk path and the num_tokens>2048 tilelang path ->
# per-stage 24.5 KB -> 49152 B < 65536 (largest power-of-2 divisor of
# hc_hidden_size=16384 and split_size=512 that fits). All six MHC-forward
# tilelang kernels then fit MI300A's non-optin LDS. NOTE: hidden_block=128 is
# ~2x more GEMM iterations than 256 -> lower arithmetic intensity; the proper
# fix is the tilelang HIP dynamic-smem opt-in (emit hipFuncSetAttribute + check
# against hipDeviceAttributeMaxSharedMemoryPerBlockOptin), which would restore
# hidden_block=256 at full speed. Patches are version-controlled next to this
# script; build_overlay.sh applies them to the extracted overlay source.
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
  --dsa-topk-backend "$GLM53_DSA_TOPK_BACKEND"
  --moe-runner-backend "$GLM53_MOE_RUNNER"
  --mamba-backend "$GLM53_MAMBA_BACKEND"
)
# --- multimodal vision-tower attention (fa3 MISSING on ROCm) ----------------
SGLANG_ARGS+=( --mm-attention-backend "$GLM53_MM_ATTENTION" )
# Route unquantized bf16 GEMMs to rocBLAS (see GLM53_BF16_GEMM_BACKEND above):
SGLANG_ARGS+=( --bf16-gemm-backend "$GLM53_BF16_GEMM_BACKEND" )
# --- reasoning + tool-call parsers (verified in the clariden overlay) -------
if [ "${GLM53_ENABLE_PARSERS:-1}" = "1" ]; then
  SGLANG_ARGS+=(
    --reasoning-parser "$GLM53_REASONING_PARSER"
    --tool-call-parser "$GLM53_TOOL_CALL_PARSER"
  )
fi
# --- chat template (GLM-5.3 ships chat_template.jinja as a SEPARATE file, not
#     in tokenizer_config.json — sglang's auto-load misses it, causing raw-text
#     input and garbage output. Pass --chat-template explicitly. Set
#     GLM53_CHAT_TEMPLATE=0 to disable (e.g. for /v1/completions raw-prompt tests).
if [ "${GLM53_CHAT_TEMPLATE:-1}" = "1" ] && [ -f "$MODEL_PATH/chat_template.jinja" ]; then
  SGLANG_ARGS+=( --chat-template "$MODEL_PATH/chat_template.jinja" )
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
exec python3 "$COMMON_DIR/sglang_launcher_mi300a.py" "${SGLANG_ARGS[@]}"
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
  # Health gate, two modes:
  #   default: poll /health (fast engines; the detok heartbeat stays fresh).
  #   SLOW_ENGINE=1 (eager decode / streaming dequant): poll /generate with
  #   max_tokens=1 instead.  /health livelocks on slow engines: every aborted
  #   curl leaves a ~24-s generate queued, the detokenizer never idles, its
  #   heartbeat goes stale and EVERY later /health fails until the deadline
  #   (625724 lesson).  A 1-token /generate is the stronger end-to-end proof
  #   anyway and it drains the queue instead of growing it.
  if [ "${SLOW_ENGINE:-0}" = "1" ]; then
    until curl -sf --max-time 300 -X POST "http://127.0.0.1:$SERVE_PORT/generate" \
            -H 'Content-Type: application/json' \
            -d '{"text": "ping", "sampling_params": {"max_new_tokens": 1, "temperature": 0}}' \
            >/dev/null 2>&1; do
      [ $SECONDS -ge $deadline ] && { echo "[$(date -Is)] health timeout after ${HEALTH_TIMEOUT}s; not registering" >&2; exit 1; }
      sleep 30
    done
  else
    until curl -sf --max-time 5 "http://127.0.0.1:$SERVE_PORT/health" >/dev/null 2>&1; do
      [ $SECONDS -ge $deadline ] && { echo "[$(date -Is)] health timeout after ${HEALTH_TIMEOUT}s; not registering" >&2; exit 1; }
      sleep 20
    done
  fi
  echo "[$(date -Is)] engine healthy after ~$((HEALTH_TIMEOUT - (deadline - SECONDS)))s"

  # --- capture probe (GLM53_CAPTURE_PROBE=1): fire ONE deterministic ~2k-token
  # request BEFORE the gen probe so comp_capture (GLM53_COMP_MIN_TOKENS=1200)
  # lands on THIS forward identically on beverin and clariden (init profile
  # forwards are machine-dependent: 1-token dummy vs bs=9).
  if [ "${GLM53_CAPTURE_PROBE:-0}" = "1" ]; then
    echo "[$(date -Is)] capture probe (deterministic long-prompt prefill) ..."
    timeout 900 \
      srun --jobid="${SLURM_JOB_ID:-0}" --overlap --gres=none --nodes=1 --ntasks=1 \
           -w "$HEAD" --environment="$OTELA_EDF_NAME" \
           python3 "$GLM53_DIAG_DIR/capture_probe.py" \
               "http://127.0.0.1:$SERVE_PORT" "$SERVED_MODEL_NAME" \
      || echo "[$(date -Is)] capture probe FAILED (rc=$?); continuing" >&2
  fi

  if [ "$GEN_PROBE" = "1" ]; then
    echo "[$(date -Is)] generation ${GEN_CORRECTNESS_SMOKE:+smoke }probe (model=$SERVED_MODEL_NAME, max_tokens=$GEN_CORRECTNESS_MAX_TOKENS, min_pass=$GEN_CORRECTNESS_MIN_PASS, budget=${GEN_PROBE_TIMEOUT}s) ..."
    PROBE_OUT="$RUNDIR/gen_correctness_${SLURM_JOB_ID:-$$}.json"
    GEN_CORRECTNESS_TIMEOUT="$GEN_PROBE_TIMEOUT" \
    GEN_CORRECTNESS_PER_REQ_TIMEOUT="$GEN_CORRECTNESS_PER_REQ_TIMEOUT" \
    GEN_CORRECTNESS_SMOKE="$GEN_CORRECTNESS_SMOKE" \
      timeout "$((GEN_PROBE_TIMEOUT + 300))" \
      srun --jobid="${SLURM_JOB_ID:-0}" --overlap --gres=none --nodes=1 --ntasks=1 \
           -w "$HEAD" --environment="$OTELA_EDF_NAME" \
           python3 "$GEN_PROBE_SCRIPT" \
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
