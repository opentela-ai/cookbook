#!/bin/bash
#SBATCH --job-name=glm53-otela-jsc
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
# Full node (288 logical CPUs): the sglang glm-5.3 image pins TP ranks to GPU
# NUMA domains using host CPU ids; a shrunken cpuset raises "CPU number N is
# not eligible" and kills ranks (clariden finding, same GH200 image).
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
# B:TERM@120 signals the BATCH SHELL (B:) 120 s before wall-time, giving the
# stop_otela trap below a window to scancel the otela step with TERM. Without
# B: the signal goes to the job steps, the batch-shell trap never runs, and
# the only grace is KillWait (default 30 s) — less than otela's ~10 s graceful
# leave (AnnounceLeave -> 5 s CRDT drain -> srv.Shutdown), so no LEFT is
# published and the head keeps a `connected: true` row round-robining traffic
# into a dead endpoint.
#SBATCH --signal=B:TERM@120
#SBATCH --output=/e/scratch/reformo/yao4/glm-53-flash/logs/serve-%j.out
#SBATCH --error=/e/scratch/reformo/yao4/glm-53-flash/logs/serve-%j.out
#
# =============================================================================
# Serve zai-org/GLM-5.3-Flash (FP8 e4m3 dynamic, ~306 GiB, multimodal
# vision+text, DSA + linear + KDA hybrid attention, 45 layers / 288 routed
# experts top-8, MTP) on JSC Jupiter Booster (4x GH200 / node, aarch64,
# InfiniBand) and register it on OpenTela — one self-contained sbatch.
#
# Ported from the VALIDATED clariden recipe (same GH200/sm_90/aarch64 GPU and
# the same lmsysorg/sglang:glm-5.3-flash image): model+image facts (DSA
# tilelang pins, sitecustomize FA3 patches, parsers, CTX_LEN/MEM_FRAC,
# watchdogs) carry over verbatim; site facts (apptainer instead of enroot/EDF,
# NCCL over InfiniBand instead of Slingshot/CXI, login-node relay instead of
# direct mesh, reformo/booster, /e/scratch paths) are re-derived from the
# kimi-k3 JSC recipe + live probes (sinfo/sacctmgr 2026-10-07).
#
# Submit (login node):
#   sbatch serve_glm_53_flash_otela_jsc.sbatch          # default: 1 node, TP4 (clariden-validated shape)
#   sbatch --nodes=2 --export=ALL,NNODES=2,PP_SIZE=2 \
#          serve_glm_53_flash_otela_jsc.sbatch          # 2 nodes, TP4 x PP2 (more KV headroom; unvalidated)
#
# FAST FIRST-BRING-UP (recommended): dummy weights, generation smoke only, NO
# registration — isolates the full init -> JIT -> forward -> HTTP pipeline in
# ~10-15 min instead of the multi-hour real-weight cold start:
#   sbatch --export=ALL,LOAD_FORMAT=dummy,GEN_CORRECTNESS_SMOKE=1,HEALTH_TIMEOUT=1800 \
#          serve_glm_53_flash_otela_jsc.sbatch
#
# What it does, in order:
#   1. preflights the image, weights (or LOAD_FORMAT=dummy), the otela binary
#      and gen_correctness.py (prints exact staging commands on failure), then
#      a ~30 s in-container import preflight on the head (glm5_next class +
#      sgl_kernel op availability — prints whether flash_ops/FA3 is present,
#      which is what the sitecustomize patches key off)
#   2. srun's one sglang engine rank per node inside the apptainer image
#      (servekit-profiled when the checkout exists), forming a single
#      distributed TP4 x PP<NNODES> engine. Only rank 0 (head) serves HTTP.
#   3. once the head's /health answers, runs a MANDATORY generation probe
#      BEFORE registration — /health alone is not enough (deepseek-v4 /
#      kimi-k3 / glm-5.2 lesson: an engine can answer /health while a broken
#      kernel path serves garbage or hangs)
#   4. runs the servekit verification bench (C=16 n=64, non-fatal,
#      SERVEKIT_BENCH=0 skips) so mesh traffic never pollutes the numbers
#   5. ONLY if the probe passes, starts ONE otela WORKER on the head (native
#      host binary via srun --overlap) which registers the `llm` service with
#      the OpenTela mesh through the login-node relay
#   6. on job end / scancel, SIGTERM (never KILL) the otela step so it
#      announces LEFT cleanly
#
# -----------------------------------------------------------------------------
# THE LOGIN-NODE RELAY (us-managed; this script only dials it)
# -----------------------------------------------------------------------------
# JSC compute nodes have NO outbound internet — not even Cloudflare:443. The
# login node does. So a compute-side otela cannot reach the public head; it
# must hop through a relay running on the login node:
#
#   client ─https─▶ api.opentela.ai ──P2P──▶ ocf head
#                                              ▲ WSS out via p2p.opentela.ai:443
#   ┌── JSC login node (jpbl-s01-01, ib0 10.128.1.1) ──────────────────────┐
#   │  RELAY opentela --role relay   (tmux; shared by all jobs)            │
#   │  listens on 10.128.1.1:45195   peer QmakkeU7ZYN…CPVpEvSqzgd          │
#   └───────────────────────────────────────────────────────────────────────┘
#                          ▲ TCP, JSC-internal 10.x
#   ┌── compute head node (this job) ────────────────────────────────┐
#   │  WORKER opentela --role worker --service.port $SERVE_PORT      │
#   │  sglang serve (apptainer --nv, host netns) on 0.0.0.0          │
#   └──────────────────────────────────────────────────────────────────┘
#
# The relay is a long-lived, shared resource OPERATED BY US — started with
# start_relay_jsc.sh (deployments/llm/jsc/kimi-k3/) in a tmux session on the
# login node, outliving any single job. This script dials the address in
# $RELAY_ADDR_FILE (written by that script); $OTELA_RELAY_ADDR overrides.
# Peer identity: v0.2.4 resolves the BadgerDB home via /etc/passwd (NOT $HOME
# env — verified live), landing at $PASSWD_HOME/.ocfcore on NFS-backed
# /e/home, which goes ESTALE after a few days and freezes the CRDT. The
# symlink $PASSWD_HOME/.ocfcore -> /e/scratch/... is set up by the relay
# script; the WORKER's short-lived store tolerates it for a 12 h job.
#
# NO socat IS NEEDED: the otela worker runs ON the serving node and apptainer
# shares the host network namespace, so sglang genuinely is on
# localhost:$SERVE_PORT for the registrar's health check.
#
# -----------------------------------------------------------------------------
# MEMORY (1 node, TP4 — the clariden-validated default)
# -----------------------------------------------------------------------------
# 306 GiB FP8 weights / 4 GPUs = ~76.5 GiB weights/GPU of ~96 GiB usable
# GH200 HBM -> ~20 GiB free for KV pool + activations + JIT — tight but
# sufficient (clariden jobs 3213420/3219525 served 12 h on exactly this
# shape; max_total_num_tokens=287104 at MEM_FRAC=0.88, max_running_requests
# =17, the Mamba state-pool cap). 2 nodes TP4xPP2 widen KV headroom to ~58
# GiB/GPU but were never validated end-to-end on clariden either — hence
# NNODES=1 default and the PP override above.
# =============================================================================

set -uo pipefail

# ----------------------------------------------------- script dir resolution --
# gen_correctness.py is a SIBLING of this sbatch (the recipe is a bundle).
# Slurm copies the sbatch to /var/spool/slurmd/job<id>/ and runs it from
# there, so ${BASH_SOURCE[0]} is the spool copy — only a fallback for
# non-Slurm runs. The real search: the submit dir, then the repo-relative
# path (rcc submits from the repo root; manual submits from the bundle dir).
SCRIPT_DIR=""
for _cand in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/deployments/llm/jsc/glm-53-flash" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
  [ -n "$_cand" ] && [ -f "$_cand/gen_correctness.py" ] && { SCRIPT_DIR="$_cand"; break; }
done
if [ -z "$SCRIPT_DIR" ]; then
  echo "FATAL: cannot locate gen_correctness.py (searched SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>} and the BASH_SOURCE dir)." >&2
  echo "       The recipe is a bundle: submit from the bundle dir or the repo root," >&2
  echo "       or cp gen_correctness.py next to the spool copy." >&2
  exit 1
fi
unset _cand

# ---------------------------------------------------------------- deployment --
# Reuses the /e/scratch/reformo/yao4/glm-53-flash tree: the apptainer image,
# weights, patch dir and caches live here, so a fresh submit does not re-pull
# or re-JIT from cold. /e/scratch (EXA) IS mounted on compute nodes — /p/scratch
# (JUSTSCRATCH) is NOT (glm-47-flash JSC finding: jobs with output on /p/scratch
# die with ExitCode=0:53 in ~8 s) — every artefact this job reads must stay
# under /e/scratch (or /e/data1, also mounted).
export DEPLOY_DIR="${DEPLOY_DIR:-/e/scratch/reformo/yao4/glm-53-flash}"
export IMAGE="${IMAGE:-$DEPLOY_DIR/images/sglang-glm-5.3-flash.sif}"
export MODEL_PATH="${MODEL_PATH:-$DEPLOY_DIR/models/GLM-5.3-Flash}"
# Served under the canonical HF id so the peer is THE GLM-5.3-Flash endpoint
# on the mesh (coexists with kimi-k3 / GLM-5.3 registrations).
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-5.3-Flash}"

# ------------------------------------------------------------------- shape ----
# WHY TP4 x PP<NNODES>: TP4 keeps collectives on the intra-node 4-GH200 NVLink
# domain; PP puts one pipeline stage per node so only point-to-point send/recv
# crosses the IB fabric — stable and CUDA-graph-safe. Flat cross-node TP boots
# but crawls (~1-5 tok/s; same finding on clariden/Slingshot and kimi-k3/JSC
# without SHARP). 1 node = pure TP4 (the clariden-validated shape).
export NNODES="${NNODES:-$SLURM_JOB_NUM_NODES}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-$NNODES}"
# 288 routed experts, top-8. EP=TP keeps expert dispatch intra-node
# (72 experts/GPU at TP4). Clariden default, carried verbatim.
export EP_SIZE="${EP_SIZE:-$TP_SIZE}"
# MoE runner: FP8 e4m3 dynamic (NOT mxfp4 like kimi-k3) — the native fp8 path
# is mature on sm_90, `auto` lets the image pick (clariden-validated). If `auto`
# ever dequantizes to bf16 and OOMs (the kimi-k3/Triton-Kernels anti-pattern),
# override MOE_BACKEND and check the "Using MoE runner backend" log line.
export MOE_BACKEND="${MOE_BACKEND:-auto}"
export CG_DECODE="${CG_DECODE:-full}"

# --- Attention backends (model facts, carried verbatim from clariden) --------
# GLM-5.3-Flash is a HYBRID: 34/45 linear/KDA layers (mamba_backend=triton,
# auto-resolved, no flash dep) + 11/45 DSA layers. THE kpool>1 HARD GATE:
# dsa_index_kpool > 1 (multimodal) is only legal with fa3/tilelang/trtllm DSA
# backends (sglang _check_kpool_tail_backend raises on EVERY forward
# otherwise). On this aarch64 image fa3 (sgl_kernel.flash_ops) and trtllm are
# MISSING; tilelang is present and handles the kpool tail natively => pin BOTH
# --dsa-prefill-backend and --dsa-decode-backend to tilelang. The DSA layer's
# full-MHA prefill ALSO hard-codes FA3 on SM90 (flash_attn_varlen_func ver=3)
# — the sitecustomize.py patch below swaps it for a PyTorch SDPA ragged loop
# (correct, slower; correctness first for a first JSC serve).
export GLM53_DSA_PREFILL="${GLM53_DSA_PREFILL:-tilelang}"
export GLM53_DSA_DECODE="${GLM53_DSA_DECODE:-tilelang}"
# Multimodal (vision-tower) attention: platform default on sm_90 is fa3 ->
# ImportError (no sgl_kernel.flash_ops). triton_attn is the documented
# non-FA3 CUDA default (clariden job 3209753).
export GLM53_MM_ATTENTION="${GLM53_MM_ATTENTION:-triton_attn}"
# Reasoning + tool-call parsers (clariden-verified on this image).
export GLM53_REASONING_PARSER="${GLM53_REASONING_PARSER:-glm45}"
export GLM53_TOOL_CALL_PARSER="${GLM53_TOOL_CALL_PARSER:-glm47}"
export GLM53_ENABLE_PARSERS="${GLM53_ENABLE_PARSERS:-1}"

# Request-length bound: 262144 (256K). pi's catalog advertises max-out 131072
# and sends max_completion_tokens=131072; the gateway 400s any request above
# the advertised max_model_len, so 32768 broke every default pi request
# (clariden finding). A single 256K request fits the ~287K live-KV pool at
# 91%; max_running_requests stays 17 (Mamba ssm_state cap) either way. Raise
# toward 1M only with HICACHE_ENABLE=1.
export CTX_LEN="${CTX_LEN:-262144}"
export MEM_FRAC="${MEM_FRAC:-0.88}"

# --- HiCache: offload evicted radix prefixes to Grace LPDDR (off by default) -
# HICACHE_ENABLE=0 matches the proven clariden run; enable for extra
# prefix-cache headroom (write_back is the only working policy on the hybrid
# SSM path; io=kernel uses sgl_kernel.kvcacheio).
export HICACHE_ENABLE="${HICACHE_ENABLE:-0}"
export HICACHE_RATIO="${HICACHE_RATIO:-1.0}"
export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_back}"
export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-kernel}"

# --- Scheduler + NCCL watchdogs (clariden fixes 11/12/13) --------------------
# --watchdog-timeout 3600 (hard kill), --soft-watchdog-timeout 600 (stack dump
# without crash), --dist-timeout 3600 (ProcessGroupNCCL floor — a SEPARATE,
# lower default of 600 s fired 5.5 h before the scheduler watchdog on the same
# PP SEND stall on clariden kimi-k3).
export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-3600}"
export SOFT_WATCHDOG_TIMEOUT="${SOFT_WATCHDOG_TIMEOUT:-600}"
export DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"

export DIST_PORT="${DIST_PORT:-20000}"
export SERVE_PORT="${SERVE_PORT:-30000}"
export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

# --- /health must NOT do a generation round-trip ----------------------------
# THIS sglang fork defaults SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION to TRUE:
# GET /health fires a real generate gated on the detokenizer heartbeat; a
# heartbeat stall flips the server to UnHealthy (503) BEFORE the scheduler
# and otela's 3 failed probes WITHDRAW the advertisement => gateway 503 for a
# healthy engine (clariden job 3419248). =0 returns 200 immediately.
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION:-0}"
export SGLANG_HEALTH_CHECK_TIMEOUT="${SGLANG_HEALTH_CHECK_TIMEOUT:-300}"

# --- MTP / nextn prediction: OFF (clariden default; plain serve first) -------
export GLM53_SPECULATIVE="${GLM53_SPECULATIVE:-0}"

# --- escape hatches (GLM-5.2-on-MI300A workarounds; NOT needed on GH200 -----
# per clariden, but kept as knobs): if the first cold start crashes during
# CUDA-graph capture or warmup, submit with DISABLE_CUDA_GRAPH=1
# SKIP_SERVER_WARMUP=1 and re-test — that would be a new JSC finding to log.
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
export SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-0}"

# The sitecustomize.py patch below overrides sglang's 480 s post-load barrier
# with THIS env value (default 3600) — 306 GiB / 62 shards off /e/scratch can
# exceed 8 min on the slowest rank (kimi-k3 fix 10, clariden-carried).
export UNBALANCED_MODEL_LOADING_TIMEOUT_S="${UNBALANCED_MODEL_LOADING_TIMEOUT_S:-3600}"

# --- Load format ------------------------------------------------------------
# LOAD_FORMAT=dummy: random weights, fast forward-path smoke (pair with
# GEN_CORRECTNESS_SMOKE=1 + a lower HEALTH_TIMEOUT).
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# --- metrics: OFF by default (JSC kimi-k3 convention) ------------------------
# PyTorch's NCCL watchdog, on a collective timeout, re-registers Prometheus
# metrics sglang already registered -> DuplicateTimeseries -> server crash
# (kimi-k3 JSC finding). Nothing in this pipeline reads /metrics; opt in with
# ENABLE_METRICS=1 only when you can tolerate that risk.
export ENABLE_METRICS="${ENABLE_METRICS:-}"

# ------------------------------------------------------------------- otela ----
# Official OpenTela v0.2.4 arm64 release, staged on /e/scratch by the kimi-k3
# recipe (same binary the JSC relay runs; verified end-to-end on this site).
# JSC compute has no egress, so the preflight prints the exact login-node
# staging command if the binary is missing. Rollback = re-stage another
# official release and point OTELA_BIN at it.
OTELA_VERSION="${OTELA_VERSION:-v0.2.4}"
OTELA_BIN="${OTELA_BIN:-/e/scratch/reformo/yao4/otela-llm/opentela-arm64}"
OTELA_URL="${OTELA_URL:-https://github.com/eth-easl/OpenTela/releases/download/$OTELA_VERSION/opentela-arm64}"
export OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
OTELA_TCP_PORT="${OTELA_TCP_PORT:-43920}"
OTELA_UDP_PORT="${OTELA_UDP_PORT:-59840}"
OTELA_API_PORT="${OTELA_API_PORT:-18094}"
OTELA_STEP_NAME="${OTELA_STEP_NAME:-otela-worker-$SLURM_JOB_ID}"
# Peers DISCARD every CRDT record published by a binary without a build
# attestation (self-built = "dev"); the mesh path here includes official
# releases only, but keep false so older unsigned peers are still accepted
# (kimi-k3 JSC setting, verified with this relay).
OTELA_REQUIRE_SIGNED="${OTELA_REQUIRE_SIGNED:-false}"
# FRESH PEER ID PER JOB (clariden v0.2.6 learning, applies to v0.2.4 + per-job
# config-dir too): a CONSTANT seed reproduces the same peer id, which cannot
# re-register after a graceful leave until the LEFT tombstone expires in the
# head's CRDT (the euler/dgx-spark stall). $$ = this sbatch's PID => random
# per job. Set a constant ONLY to pin the peer id (and expect the stall).
OTELA_SEED="${OTELA_SEED:-$$}"
# NEVER auto-register a dummy-weight engine (random weights would serve
# garbage to mesh traffic). auto (default): register only when LOAD_FORMAT !=
# dummy. "1": always register (e.g. an end-to-end relay-hop test on dummy
# weights — expect garbage generations). "0": never register.
OTELA_REGISTER="${OTELA_REGISTER:-auto}"
# The shared login-node relay (us-managed; see header). Precedence: explicit
# $OTELA_RELAY_ADDR > non-empty $RELAY_ADDR_FILE > the baked-in default.
RELAY_ADDR_FILE="${RELAY_ADDR_FILE:-/e/scratch/reformo/yao4/otela-relay/relay.multiaddr}"
OTELA_RELAY_DEFAULT="/ip4/10.128.1.1/tcp/45195/p2p/QmakkeU7ZYNRdA2XhYThT3r8QHd2qxbekTNCPVpEvSqzgd"

# Cold start: ~306 GiB FP8 / 62 shards off /e/scratch (EXA). Clariden budgeted
# up to ~2.5 h worst case off Lustre; keep 3 h. Gates the pre-registration
# wait only (the UNBALANCED_MODEL_LOADING_TIMEOUT_S patch below is what
# actually prevents the load barrier from killing slow ranks).
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-10800}"

# ----------------------------------------------------- mandatory gen probe ---
# GLM-5.3-Flash has never generated a token on JSC — /health 200 does NOT
# prove the DSA + linear + KDA hybrid forward works (deepseek-v4 answered
# /health and 502'd every request; glm-5.2 served /health then the DSA decode
# hung at 100% GPU). gen_correctness.py (sibling, stdlib-only, reused from
# kimi-k3/glm52/clariden-glm53) sends six greedy /v1/completions prompts and
# checks factual substrings. GEN_CORRECTNESS_SMOKE=1 (with LOAD_FORMAT=dummy)
# gates on non-empty continuation instead. GEN_PROBE=0 skips (NOT recommended).
export GEN_PROBE="${GEN_PROBE:-1}"
export GEN_PROBE_TIMEOUT="${GEN_PROBE_TIMEOUT:-1200}"
export GEN_CORRECTNESS_MAX_TOKENS="${GEN_CORRECTNESS_MAX_TOKENS:-64}"
export GEN_CORRECTNESS_MIN_PASS="${GEN_CORRECTNESS_MIN_PASS:-5}"
export GEN_CORRECTNESS_PER_REQ_TIMEOUT="${GEN_CORRECTNESS_PER_REQ_TIMEOUT:-300}"
export GEN_CORRECTNESS_SMOKE="${GEN_CORRECTNESS_SMOKE:-0}"

# ---- servekit (cold-start profile + pre-registration verification bench) ----
# engine.sh wraps the engine with `servekit profile` (github.com/eth-easl/
# servekit, pure stdlib — runs module-style from a checkout on /e/scratch,
# bind-mounted into the container; no install, no egress needed on compute).
# After /health + the gen probe and BEFORE otela registration, a fixed servekit
# bench runs inside the container and merges into the node-0 report. Missing
# checkout: unwrapped engine, bench skipped with a WARN.
export SERVEKIT_DIR="${SERVEKIT_DIR:-/e/scratch/reformo/yao4/otela-llm/servekit}"
export SERVEKIT_BENCH="${SERVEKIT_BENCH:-1}"              # 0 skips the auto-bench
export SERVEKIT_BENCH_REQUESTS="${SERVEKIT_BENCH_REQUESTS:-64}"
export SERVEKIT_BENCH_CONCURRENCY="${SERVEKIT_BENCH_CONCURRENCY:-16}"
export SERVEKIT_BENCH_IN="${SERVEKIT_BENCH_IN:-768}"      # words (~1024 tokens)
export SERVEKIT_BENCH_OUT="${SERVEKIT_BENCH_OUT:-256}"    # tokens, ignore_eos
export SERVEKIT_BENCH_CORRECTNESS="${SERVEKIT_BENCH_CORRECTNESS:-1}"

export RUNDIR="$DEPLOY_DIR/run-$SLURM_JOB_ID"
export PATCH_DIR="$DEPLOY_DIR/patches"
mkdir -p "$DEPLOY_DIR/logs" "$RUNDIR" "$DEPLOY_DIR/cache" "$DEPLOY_DIR/home" \
         "$PATCH_DIR"

# ============================================================================ --
# sitecustomize.py — auto-imported by Python at startup (PATCH_DIR is prepended
# to PYTHONPATH by engine.sh), BEFORE sglang runs. Three patches, all carried
# VERBATIM from the validated clariden recipe (same image, same aarch64/
# sm_90 GH200):
#   1. UNBALANCED_MODEL_LOADING_TIMEOUT_S 480 -> 3600: sglang's post-load
#      monitored_barrier kills the job if the slowest TP rank exceeds 8 min
#      streaming weights; 306 GiB off /e/scratch can exceed that.
#   2. kpool-tail FA3-safe: sglang's _resolve_kpool_tail_backend hard-overrides
#      dsa_impl flashmla_sparse -> fa3 on sm_90 assuming FA3 is installed; the
#      aarch64 image has NO sgl_kernel.flash_ops -> ImportError at graph
#      capture (clariden job 3208291). Patch keeps flashmla_sparse when fa3 is
#      absent. INERT while the tilelang pins above are in effect (defensive).
#   3. _forward_standard_mha SDPA fallback: the DSA full-MHA prefill calls
#      flash_attn_varlen_func ver=3 (FA3) unconditionally on SM90 (clariden
#      job 3209880); on this image BOTH built-in paths are dead (FA3 absent,
#      trtllm Blackwell-only — job 3210211 "Unsupported architecture"), so the
#      patch swaps in a PyTorch SDPA ragged loop. Fires ONLY when fa3 is
#      absent on sm_90; behavior-preserving elsewhere.
cat > "$PATCH_DIR/sitecustomize.py" <<'PYEOF'
import os

# ---- kimi-k3 fix 10: dist barrier timeout after model load ------------
try:
    import sglang.srt.model_executor.model_runner_components.load_model_utils as _lm
    _lm.UNBALANCED_MODEL_LOADING_TIMEOUT_S = int(
        os.environ.get("UNBALANCED_MODEL_LOADING_TIMEOUT_S", "3600"))
except Exception:
    pass  # sglang not installed yet (e.g. preflight) — skip

# ---- GLM-5.3 clariden: kpool-tail backend must respect FA3 availability
try:
    import importlib as _il

    def _importable(name):
        try:
            _il.import_module(name)
            return True
        except Exception:
            return False

    # sgl_kernel.flash_ops is the symbol FA3 actually needs. On aarch64 it
    # is MISSING (only flash_mla is present); on x86 it is present. Probe
    # once at patch time — availability is fixed for the process lifetime.
    _fa3_ok = _importable("sgl_kernel.flash_ops")
    from sglang.srt.layers.attention import dsa_backend as _db

    def _resolve_kpool_tail_backend_fa3safe(self, topk_indices, dsa_impl):
        # Replicate the original guard verbatim (all pass-through cases).
        if (topk_indices is None
                or self.dsa_index_kpool <= 1
                or dsa_impl != "flashmla_sparse"):
            return dsa_impl
        target = ("trtllm" if self.device_sm_major >= 10 else
                  "fa3" if self.device_sm_major == 9 else dsa_impl)
        if target == dsa_impl:
            return dsa_impl
        # Apply the perf override ONLY when the target is importable on this
        # arch. On aarch64 (no flash_ops/FA3) keep flashmla_sparse — the
        # image's own auto-selection, which routes to
        # sgl_kernel.flash_mla.flash_mla_sparse_fwd (no FA3 dep) — instead of
        # crashing at CUDA-graph capture (job 3208291).
        if target == "fa3" and not _fa3_ok:
            return dsa_impl
        if target == "trtllm" and not _importable("tensorrt_llm"):
            return dsa_impl
        return target

    _db.DeepseekSparseAttnBackend._resolve_kpool_tail_backend = _resolve_kpool_tail_backend_fa3safe
except Exception as _e:  # sglang/dsa_backend not importable yet (preflight)
    import sys
    sys.stderr.write(
        f"[sitecustomize] kpool-tail FA3-safe patch skipped: {_e!r}\n")

# ---- GLM-5.3 clariden: _forward_standard_mha hard-codes FA3 on SM90 ----
# The DSA layer's FULL-MHA prefill (forward_extend -> _forward_standard_mha,
# dsa_backend.py:4131) calls flash_attn_varlen_func with the module-top
# default ver=3 -> flash_attention_v3 -> _load_fa3_kernels -> ImportError
# (job 3209880). NO CLI override exists: the method has an SM100+
# (Blackwell) flashinfer branch (L4105-4128 -> flashinfer.prefill.
# trtllm_ragged_attention_deepseek), but that is a Blackwell-only TRT-LLM
# kernel -- job 3210211 crashed with tvm.error "TllmGenFmhaRunner ...
# Unsupported architecture" on sm_90 (the earlier "device_sm_major=10"
# trick was wrong: SM90 is NOT functionally Blackwell for this op). So on
# this aarch64 Hopper image BOTH built-in paths are dead (FA3 absent,
# trtllm Blackwell-only); sgl_kernel FA2, flash_attn_interface (FA2 pkg),
# and flashinfer BatchPrefill/MLA are all ABSENT too (verified). The ONLY
# universal, sm_90-runnable full-MHA kernel on this image is PyTorch SDPA
# (torch.nn.functional.scaled_dot_product_attention). GLM-5.3's DSA layers
# are pure MHA here (config: num_attention_heads==num_key_value_heads==64,
# qk_nope_head_dim==v_head_dim==256, qk_rope_head_dim==0 -> head_dim
# ==v_head_dim, no GQA), so a per-request SDPA ragged loop over
# metadata.cu_seqlens_{q,k} with is_causal=True is a drop-in, correct
# replacement (slower than a fused kernel -- correctness first for a
# first-ever aarch64 serve). Behavior-preserving when _fa3_ok (x86 Hopper
# runs the original FA3 path) and on SM100+ (x86 Blackwell runs the
# original trtllm path): the replacement fires ONLY on not-_fa3_ok SM90.
try:
    if not _fa3_ok:  # set by the kpool-tail block above (same process)
        import torch as _torch
        _F = _torch.nn.functional
        _fsm = _db.DeepseekSparseAttnBackend._forward_standard_mha

        def _forward_standard_mha_fa3safe(self, q, k, v, layer,
                                          forward_batch, metadata):
            if self.device_sm_major != 9:
                return _fsm(self, q, k, v, layer, forward_batch, metadata)
            # aarch64 SM90 without FA3: PyTorch SDPA ragged loop.
            q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
            v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
            cu_q, cu_k = metadata.cu_seqlens_q, metadata.cu_seqlens_k
            causal = True
            scale = layer.scaling
            gqa = (q.shape[-2] != k.shape[-2])
            out = _torch.empty_like(q)
            for _i in range(len(cu_q) - 1):
                _qs, _qe = int(cu_q[_i]), int(cu_q[_i + 1])
                if _qe <= _qs:
                    continue
                _ks, _ke = int(cu_k[_i]), int(cu_k[_i + 1])
                # FA stores (total, H, D); SDPA wants (batch, H, seq, D).
                _qi = q[_qs:_qe].transpose(0, 1)[None]  # (1, H,  sl_q, D)
                _ki = k[_ks:_ke].transpose(0, 1)[None]  # (1, Hk, sl_k, D)
                _vi = v[_ks:_ke].transpose(0, 1)[None]  # (1, Hk, sl_k, Dv)
                _sl_q, _sl_k = _qe - _qs, _ke - _ks
                if causal and _sl_q == _sl_k:
                    _oi = _F.scaled_dot_product_attention(
                        _qi, _ki, _vi, is_causal=True, scale=scale,
                        enable_gqa=gqa)
                elif causal:  # sl_q < sl_k: bottom-right additive mask
                    _m = _torch.full((_sl_q, _sl_k), float("-inf"),
                                     device=q.device, dtype=q.dtype).tril(
                        diagonal=_sl_k - _sl_q)[None, None]
                    _oi = _F.scaled_dot_product_attention(
                        _qi, _ki, _vi, attn_mask=_m, scale=scale,
                        enable_gqa=gqa)
                else:
                    _oi = _F.scaled_dot_product_attention(
                        _qi, _ki, _vi, scale=scale, enable_gqa=gqa)
                out[_qs:_qe] = _oi[0].transpose(0, 1)  # back to (sl_q, H, D)
            return out

        _db.DeepseekSparseAttnBackend._forward_standard_mha = _forward_standard_mha_fa3safe
except Exception as _e2:
    import sys
    sys.stderr.write(
        f"[sitecustomize] _forward_standard_mha FA3-safe patch skipped: {_e2!r}\n")
PYEOF
echo "sitecustomize: written to $PATCH_DIR/sitecustomize.py"

# --------------------------------------------------------------- preflight ---
echo "=============================== preflight ==============================="
echo "job       : $SLURM_JOB_ID  nodes=$NNODES (TP$TP_SIZE x PP$PP_SIZE, EP$EP_SIZE)"
echo "model     : $SERVED_MODEL_NAME @ $MODEL_PATH"
echo "image     : $IMAGE"
echo "script dir: $SCRIPT_DIR"

ok=1
if ! srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 test -r "$IMAGE"; then
  ok=0; echo "FATAL: apptainer image missing: $IMAGE" >&2
  echo "  On a JSC LOGIN node run (once, ~10-25 min; or build_glm_5_3_flash_image.sh):" >&2
  echo "    export APPTAINER_CACHEDIR=$DEPLOY_DIR/cache/apptainer APPTAINER_TMPDIR=$DEPLOY_DIR/tmp" >&2
  echo "    mkdir -p \$APPTAINER_CACHEDIR \$APPTAINER_TMPDIR" >&2
  echo "    apptainer build --force '$IMAGE' docker://lmsysorg/sglang:glm-5.3-flash" >&2
fi
if [ "$LOAD_FORMAT" != "dummy" ] && [ ! -f "$MODEL_PATH/config.json" ]; then
  ok=0; echo "FATAL: model weights missing: $MODEL_PATH/config.json" >&2
  echo "  On a JSC LOGIN node run (once, resumable; or download_glm_5_3_flash.sh):" >&2
  echo "    deployments/llm/jsc/glm-53-flash/download_glm_5_3_flash.sh" >&2
fi
if ! srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 test -x "$OTELA_BIN"; then
  ok=0; echo "FATAL: otela binary missing: $OTELA_BIN" >&2
  echo "  On a JSC LOGIN node run (compute has no egress):" >&2
  echo "    mkdir -p '$(dirname "$OTELA_BIN")'" >&2
  echo "    curl -L --fail -o '$OTELA_BIN' '$OTELA_URL'" >&2
  echo "    chmod +x '$OTELA_BIN'" >&2
fi
if [ "$GEN_PROBE" = "1" ] && [ ! -f "$SCRIPT_DIR/gen_correctness.py" ]; then
  ok=0; echo "FATAL: gen_correctness.py not found next to this sbatch ($SCRIPT_DIR)" >&2
fi
if [ "$ok" != "1" ]; then echo "preflight FAILED"; exit 1; fi
echo "preflight: image / weights / otela / gen_correctness.py OK"

# In-container preflight (~30 s, --nv = same env the engine will see): the
# model class import exercises torch + the checkpoint's architecture code;
# the flash_ops probe prints the FA3 availability the sitecustomize patches
# and tilelang pins key off (documented absent on aarch64).
echo "--- container preflight (head) ---"
srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 \
     apptainer exec --nv --bind /e/scratch:/e/scratch --bind /e/data1:/e/data1 --pwd "$DEPLOY_DIR" "$IMAGE" \
     python3 -c '
import importlib, traceback
try:
    m = importlib.import_module("sglang.srt.models.glm5_next")
    print("[preflight] glm5_next EntryClass:", getattr(m, "EntryClass", None))
except Exception:
    traceback.print_exc()
try:
    import sgl_kernel.flash_ops  # noqa: F401
    print("[preflight] sgl_kernel.flash_ops: PRESENT (FA3 available)")
except Exception as e:
    print("[preflight] sgl_kernel.flash_ops: ABSENT (%s) -> tilelang DSA pins + sitecustomize patches are load-bearing" % e.__class__.__name__)
' || { echo "FATAL: container preflight failed"; exit 1; }

# ----------------------------------------------------------------- topology --
HEAD="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
HEAD_IP="$(scontrol show node "$HEAD" | awk -F'= *' '/NodeAddr=/{print $2; exit}')"
export HEAD HEAD_IP
echo "head: $HEAD ($HEAD_IP)  serve port: $SERVE_PORT"

RELAY_ADDR="${OTELA_RELAY_ADDR:-}"
if [ -z "$RELAY_ADDR" ] && [ -f "$RELAY_ADDR_FILE" ]; then
  RELAY_ADDR="$(head -n1 "$RELAY_ADDR_FILE")"
fi
if [ -z "$RELAY_ADDR" ]; then RELAY_ADDR="$OTELA_RELAY_DEFAULT"; fi
echo "relay: $RELAY_ADDR"

# last_service.env (consumed by verification / renewal tooling; kimi-k3 fields)
cat > "$DEPLOY_DIR/last_service.env" <<ENV
SERVICE_JOB_ID=$SLURM_JOB_ID
SERVICE_HEAD_NODE=$HEAD
SERVICE_HEAD_IP=$HEAD_IP
SERVICE_PORT=$SERVE_PORT
SERVICE_URL=http://$HEAD_IP:$SERVE_PORT
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
OTELA_RELAY_ADDR=$RELAY_ADDR
ENV

# self-contained run dir: the probe runs from $RUNDIR (always /e/scratch) even
# if this sbatch was submitted from outside the shared tree
cp "$SCRIPT_DIR/gen_correctness.py" "$RUNDIR/"

# ============================================================ engine.sh ======
# One engine rank per node: TP4 intra-node (NVLink) + PP across nodes. The
# servekit profile wrapper (when the checkout exists) is INSIDE the container,
# wrapping the engine command — it detects per-rank readiness (head: "The
# server is fired up and ready to roll!"; workers: the multi-node "Dummy
# health check server started" line) and reports the cold-start phases
# independently of the health-wait below. Same command, same env, same
# signals, terminal attached; the report lands in $RUNDIR/coldstart*.
#
# NCCL over the JSC InfiniBand fabric (kimi-k3 JSC, site-verified): NCCL's
# default GDR read path stalls on this fabric -> force WRITE (RETRY_CNT 7,
# TIMEOUT 22 = 22*2^25 us ~= 12 min; kernel default 18 = 30 s) + pinned IPv4
# ib0 (GDR write + IPv4 auto-detection needs NCCL_SOCKET_FAMILY=IPv4; the ib0
# NIC peer isn't in DNS, which otherwise delays topology discovery by up to
# 2 min). No SHARP: cross-node traffic is PP point-to-point only, and the
# glm-5.3-flash image is CUDA13-based while its NCCL links libsharp against
# libcudart.so.12 (kimi-k3 JSC finding).
#
# The engine binds 0.0.0.0:$SERVE_PORT in the HOST network namespace
# (apptainer --nv shares host netns) — that is what makes the otela
# registrar's localhost:$SERVE_PORT health check genuinely local.
cat > "$RUNDIR/engine.sh" <<'ENGINE'
#!/bin/bash
set -uo pipefail
RANK="${SLURM_PROCID:-0}"
echo "[rank $RANK] engine on $(hostname) (SLURM_NODEID=${SLURM_NODEID:-0})"
echo "[rank $RANK] shape: tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE moe=$MOE_BACKEND cg_decode=$CG_DECODE ctx=$CTX_LEN load_format=${LOAD_FORMAT:-auto}"

# --- network (InfiniBand via ib0; JSC booster) -------------------------------
NCCL_SOCKET_IFNAME=ib0
GLOO_SOCKET_IFNAME=ib0
export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
NCCL_SOCKET_FAMILY=IPv4
GLOO_SOCKET_FAMILY=IPv4
export NCCL_SOCKET_FAMILY GLOO_SOCKET_FAMILY
export NCCL_IB_DISABLE=0
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=22
export NCCL_DEBUG=WARN
SGLANG_HOST_IP="$(ip -4 addr show "$NCCL_SOCKET_IFNAME" 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -n1)"
export SGLANG_HOST_IP

# --- caches + HOME inside the container (kimi-k3 JSC / clariden pattern) -----
# HOME on /e/scratch (NOT $PASSWD_HOME on NFS-backed /e/home: quota + ESTALE);
# TRITON_CACHE_DIR shared across ranks & re-submits so nothing re-JITs;
# HF_*_OFFLINE because compute nodes have no egress (any accidental hub read
# would hang to the request timeout).
export HOME="${HOME:-/e/scratch/reformo/yao4/glm-53-flash/home}"
export HF_HOME="${HF_HOME:-/e/scratch/reformo/yao4/glm-53-flash/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/e/scratch/reformo/yao4/glm-53-flash/cache/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/e/scratch/reformo/yao4/glm-53-flash/cache/triton}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
mkdir -p "$HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

# sitecustomize.py (weight-load barrier + kpool-tail + SDPA patches) is
# auto-imported by Python at startup with PATCH_DIR first on PYTHONPATH;
# servekit's cli module (profile wrapper below) needs its checkout too.
export PYTHONPATH="${PATCH_DIR:-/e/scratch/reformo/yao4/glm-53-flash/patches}${SERVEKIT_DIR:+:$SERVEKIT_DIR/src}${PYTHONPATH:+:$PYTHONPATH}"

# Verify the load-barrier monkey-patch took effect IN THE CONTAINER python
# (sitecustomize auto-imports at startup; this only prints the patched value).
apptainer exec --nv --bind /e/scratch:/e/scratch --bind /e/data1:/e/data1 "$IMAGE" \
  env "PYTHONPATH=$PYTHONPATH" python3 -c "
import sitecustomize  # noqa: F401
import sglang.srt.model_executor.model_runner_components.load_model_utils as m
print('[rank $RANK] UNBALANCED_MODEL_LOADING_TIMEOUT_S =', m.UNBALANCED_MODEL_LOADING_TIMEOUT_S)
" 2>/dev/null || echo "[rank $RANK] WARN: patch verify failed (check PYTHONPATH)" >&2

# --- build the SGLang serve args (clariden-validated for THIS image) ---------
# `python -m sglang.launch_server`: the entry point validated on this image.
# The checkpoint's config.json supplies the architecture (Glm5Next..., 45
# layers, 288 experts top-8, DSA+linear+KDA hybrid, MTP, FP8 e4m3 dynamic).
# GENERAL attention knobs (--prefill/--decode-attention-backend,
# --mamba-ssm-dtype) are deliberately NOT passed (they target non-DSA models
# and would break the 34 linear-attention layers); only the DSA-SPECIFIC
# backends are pinned (tilelang — the only kpool>1-legal backend present on
# aarch64) plus the multimodal backend (triton_attn; fa3 is absent).
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
  --mm-attention-backend "$GLM53_MM_ATTENTION"
  --cuda-graph-backend-decode "$CG_DECODE"
  --context-length "$CTX_LEN" --mem-fraction-static "$MEM_FRAC"
  --watchdog-timeout "$WATCHDOG_TIMEOUT" --soft-watchdog-timeout "$SOFT_WATCHDOG_TIMEOUT"
  --uvicorn-access-log-exclude-prefixes /metrics /health
  --host 0.0.0.0 --port "$SERVE_PORT")

# metrics are OFF by default (kimi-k3 JSC: PyTorch's NCCL watchdog
# re-registers Prometheus metrics on a collective timeout ->
# DuplicateTimeseries crash). Opt in with ENABLE_METRICS=1.
if [ -n "${ENABLE_METRICS:-}" ]; then
  ENGINE_CMD+=(--enable-metrics)
fi

# GLM-5.3 reasoning + tool-call parsers (glm45 / glm47; cookbook). NOTE: this
# build has NO --enable-reasoning/--enable-tools/--enable-auto-tool-choice —
# just the two parser flags (clariden, same image).
if [ "${GLM53_ENABLE_PARSERS:-1}" = "1" ]; then
  ENGINE_CMD+=(
    --reasoning-parser "$GLM53_REASONING_PARSER"
    --tool-call-parser "$GLM53_TOOL_CALL_PARSER"
  )
fi

# LOAD_FORMAT override (dummy = random weights, fast forward-path smoke).
if [ -n "${LOAD_FORMAT:-}" ] && [ "${LOAD_FORMAT:-}" != "auto" ]; then
  ENGINE_CMD+=(--load-format "$LOAD_FORMAT")
fi

# CUDA-graph / warmup escape hatches (see the batch-script knob comments).
if [ "${DISABLE_CUDA_GRAPH:-0}" = "1" ]; then
  ENGINE_CMD+=(--disable-cuda-graph)
fi
if [ "${SKIP_SERVER_WARMUP:-0}" = "1" ]; then
  ENGINE_CMD+=(--skip-server-warmup)
fi

# MTP / EAGLE speculative decoding: OFF by default (unvalidated draft path on
# this family per the clariden recipe; the model ships num_nextn_predict_
# layers=1 but SGLang only activates it with --speculative-algorithm EAGLE).
if [ "${GLM53_SPECULATIVE:-0}" = "1" ]; then
  ENGINE_CMD+=(--speculative-algorithm EAGLE
               --speculative-num-steps 2
               --speculative-eagle-topk 4
               --speculative-num-draft-tokens 8)
  echo "[rank $RANK] WARN: speculative decoding ENABLED — EAGLE draft path unvalidated on sm_90." >&2
fi

# HiCache (Grace LPDDR prefix-cache extension), gated on HICACHE_ENABLE=1.
if [ "${HICACHE_ENABLE:-0}" = "1" ]; then
  ENGINE_CMD+=(--enable-hierarchical-cache
               --hicache-ratio "${HICACHE_RATIO:-1.0}"
               --hicache-write-policy "${HICACHE_WRITE_POLICY:-write_back}"
               --hicache-io-backend "${HICACHE_IO_BACKEND:-kernel}")
fi

ENGINE_CMD+=(${SGLANG_EXTRA_ARGS:+ $SGLANG_EXTRA_ARGS})

# ulimit -c 0 + unset BASH_ENV: a multi-hundred-GB core dump is never useful,
# and BASH_ENV would run an init in every subshell the loader forks.
ulimit -c 0
unset BASH_ENV

# --- servekit profile wrapper (optional; no behavior change) -----------------
PROFILE=()
if [ -n "${SERVEKIT_DIR:-}" ] && [ -f "$SERVEKIT_DIR/src/servekit/cli.py" ]; then
  PROFILE=(python3 -m servekit.cli profile
           --out "${RUNDIR}/coldstart.json"
           --timeout "${HEALTH_TIMEOUT:-10800}"
           --)
fi

echo "[rank $RANK] launching sglang (head=$([ "$RANK" = 0 ] && echo "serve(HTTP)+probe+otela" || echo "node_rank>=1(no HTTP)")) args=${ENGINE_CMD[*]}"
exec apptainer exec --nv \
     --bind /e/scratch:/e/scratch --bind /e/data1:/e/data1 --bind /e/home:/e/home \
     --pwd "$DEPLOY_DIR" \
     "$IMAGE" ${PROFILE[@]+"${PROFILE[@]}"} "${ENGINE_CMD[@]}"
ENGINE

# ============================================================ otela.cfg =====
# config-dir is PER-JOB ($RUNDIR/otela): the worker writes its cfg + libp2p
# identity there (v0.2.4 --config-dir; verified flag on the staged binary).
# v0.2.4 BadgerDB home resolution ignores $HOME env (reads /etc/passwd ->
# $PASSWD_HOME/.ocfcore; the pre-existing symlink sends it to /e/scratch —
# see header). loglevel MUST stay debug: the registration markers this
# recipe greps for fire at DEBUG/INFO and otela otherwise looks hung during
# CRDT sync (kimi-k3 job finding, verified on this site).
mkdir -p "$RUNDIR/otela"
cat > "$RUNDIR/otela/cfg.yaml" <<OTELA_CFG
name: jsc-glm53-worker-$SLURM_JOB_ID
seed: "$OTELA_SEED"
port: "$OTELA_API_PORT"
tcpport: "$OTELA_TCP_PORT"
udpport: "$OTELA_UDP_PORT"
mode: node
role: worker
loglevel: debug
cleanslate: false
reachability: private
security:
  require_signed_binary: $OTELA_REQUIRE_SIGNED
solana:
  skip_verification: true
OTELA_CFG

cat > "$RUNDIR/otela.sh" <<OTELA_SH
#!/bin/bash
set -uo pipefail
# --label (repeatable k=v peer metadata; supported since v0.2.4 — verified on
# the staged binary) makes the peer identifiable on the mesh:
#   otela peers show -> labels model=..., cluster=jsc
exec "$OTELA_BIN" start \\
  --config-dir "$RUNDIR/otela" \\
  --bootstrap.static "$RELAY_ADDR" \\
  --solana.skip_verification \\
  --label "model=$SERVED_MODEL_NAME" \\
  --label "cluster=jsc" \\
  --label "launched_by=$USER" \\
  --service.name "$OTELA_SERVICE_NAME" \\
  --service.port "$SERVE_PORT"
OTELA_SH
echo "otela worker: $OTELA_BIN ($OTELA_VERSION) via relay $RELAY_ADDR"

# ================================================= registration + lifecycle ==
_OTELA_STOPPED=0
stop_otela() {
  # The trap below fires on both TERM and EXIT, and bash runs the EXIT trap
  # after the TERM handler returns — so without this guard the teardown runs
  # twice and the second pass finds no step and reports a spurious warning.
  [ "$_OTELA_STOPPED" = 1 ] && return 0
  _OTELA_STOPPED=1

  local step
  step="$(squeue -s -j "$SLURM_JOB_ID" -h -o '%i %j' 2>/dev/null \
          | awk -v n="$OTELA_STEP_NAME" '$2==n{print $1}' | head -1)"
  if [ -z "$step" ]; then
    # NOT a clean state — the step should still exist while the job winds
    # down. Say so loudly instead of returning silently.
    echo "[$(date -Is)] WARN: otela step $OTELA_STEP_NAME not found; cannot signal it." >&2
    echo "[$(date -Is)]       If it was SIGKILLed, no LEFT was published and the head" >&2
    echo "[$(date -Is)]       may keep a stale connected=true row for this worker." >&2
    return 0
  fi
  echo "[$(date -Is)] stopping otela step $step"
  scancel -s TERM "$step" 2>/dev/null
  # 20s, not 45s: --signal=B:TERM@120 only covers wall-time expiry; on the
  # scancel path the only grace is KillWait (default 30s). otela needs ~10s
  # to shut down; 20s leaves margin and finishes before SIGKILL.
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

# Register once the engine answers. Runs on the HEAD node alongside sglang.
# Order (header items 3-5): /health -> MANDATORY generation probe ->
# servekit bench (informational) -> otela worker.
(
  deadline=$(( SECONDS + HEALTH_TIMEOUT ))
  until curl -sf --max-time 5 "http://$HEAD_IP:$SERVE_PORT/health" >/dev/null 2>&1; do
    [ $SECONDS -ge $deadline ] && { echo "[$(date -Is)] health timeout; not registering" >&2; exit 1; }
    sleep 20
  done
  echo "[$(date -Is)] engine healthy after ${SECONDS}s"

  if [ "$GEN_PROBE" = "1" ]; then
    echo "[$(date -Is)] generation ${GEN_CORRECTNESS_SMOKE:+smoke }probe (model=$SERVED_MODEL_NAME, max_tokens=$GEN_CORRECTNESS_MAX_TOKENS, min_pass=$GEN_CORRECTNESS_MIN_PASS, budget=${GEN_PROBE_TIMEOUT}s) ..."
    PROBE_OUT="$RUNDIR/gen_correctness_${SLURM_JOB_ID}.json"
    # Run INSIDE the glm-5.3-flash container on the head (host python3 is
    # 3.9; the container shares the host netns so 127.0.0.1:$SERVE_PORT
    # reaches sglang). Smoke mode comes from GEN_CORRECTNESS_SMOKE env; the
    # posargs are <base_url> <model> [out_json] [max_tokens] [min_pass].
    GEN_CORRECTNESS_TIMEOUT="$GEN_PROBE_TIMEOUT" \
    GEN_CORRECTNESS_PER_REQ_TIMEOUT="$GEN_CORRECTNESS_PER_REQ_TIMEOUT" \
    GEN_CORRECTNESS_SMOKE="$GEN_CORRECTNESS_SMOKE" \
      timeout "$((GEN_PROBE_TIMEOUT + 300))" \
      srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
           apptainer exec --bind /e/scratch:/e/scratch "$IMAGE" \
           python3 "$RUNDIR/gen_correctness.py" \
               "http://127.0.0.1:$SERVE_PORT" "$SERVED_MODEL_NAME" \
               "$PROBE_OUT" "$GEN_CORRECTNESS_MAX_TOKENS" "$GEN_CORRECTNESS_MIN_PASS"
    rc=$?
    if [ "$rc" = 0 ]; then
      echo "[$(date -Is)] GEN_PROBE_OK rc=$rc (report in $PROBE_OUT)"
    else
      echo "[$(date -Is)] GEN_PROBE_FAILED rc=$rc; NOT registering (report in $PROBE_OUT)" >&2
      echo "[$(date -Is)]       Inspect $PROBE_OUT + the engine log, then scancel if needed" >&2
      echo "[$(date -Is)]       (engine left running for debugging; it burns GPU-hours)." >&2
      echo "[$(date -Is)]       For GLM-5.3-Flash a failure here most likely means the" >&2
      echo "[$(date -Is)]       DSA + linear + KDA hybrid forward faults on JSC GH200." >&2
      exit 1
    fi
  fi

  # Verification bench BEFORE registration: otela traffic must not pollute
  # the numbers (kimi-k3 JSC pattern). Stdlib-only servekit runs inside the
  # container, module-style from the checkout on /e/scratch; non-fatal.
  if [ "${SERVEKIT_BENCH:-1}" = "1" ] && [ -d "$SERVEKIT_DIR/src/servekit" ]; then
    echo "[$(date -Is)] engine healthy — servekit bench C=$SERVEKIT_BENCH_CONCURRENCY n=$SERVEKIT_BENCH_REQUESTS (SERVEKIT_BENCH=0 to skip)"
    BENCH_ARGS=(--url "http://127.0.0.1:$SERVE_PORT"
                --requests "$SERVEKIT_BENCH_REQUESTS"
                --concurrency "$SERVEKIT_BENCH_CONCURRENCY"
                --input-len "$SERVEKIT_BENCH_IN" --output-len "$SERVEKIT_BENCH_OUT")
    [ "$SERVEKIT_BENCH_CORRECTNESS" = "1" ] || BENCH_ARGS+=(--no-correctness)
    if [ -f "$RUNDIR/coldstart.node0.json" ]; then
      BENCH_ARGS+=(--into "$RUNDIR/coldstart.node0.json")
    else
      BENCH_ARGS+=(--out "$RUNDIR/bench.json")
    fi
    if srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
         apptainer exec --bind /e/scratch:/e/scratch "$IMAGE" \
         env "PYTHONPATH=$SERVEKIT_DIR/src" \
         python3 -m servekit.cli bench "${BENCH_ARGS[@]}"; then
      echo "[$(date -Is)] servekit bench done (report in $RUNDIR)"
    else
      echo "[$(date -Is)] WARN: servekit bench failed; registering anyway (gen probe already gated correctness)" >&2
    fi
  fi

  # --- registration gate (dummy-weight engines never hit the mesh) -----
  _register=0
  if [ "$OTELA_REGISTER" = "1" ]; then
    _register=1
  elif [ "$OTELA_REGISTER" = "auto" ] && [ "$LOAD_FORMAT" != "dummy" ]; then
    _register=1
  fi

  if [ "$_register" != "1" ]; then
    echo "[$(date -Is)] NOT registering (OTELA_REGISTER=$OTELA_REGISTER, LOAD_FORMAT=$LOAD_FORMAT)"
    if [ "$LOAD_FORMAT" = "dummy" ]; then
      echo "[$(date -Is)] smoke complete: ending the job cleanly (no worker was started)"
      # The engine srun gets TERM, the trap's stop_otela finds no step (WARN is
      # expected), Slurm tears the allocation down. Nothing was advertised.
      scancel "$SLURM_JOB_ID"
    else
      echo "[$(date -Is)] engine stays local until walltime/scancel"
    fi
    exit 0
  fi

  echo "[$(date -Is)] starting otela worker (service $OTELA_SERVICE_NAME :$SERVE_PORT, step $OTELA_STEP_NAME)"
  # FOREGROUND srun inside this (background) subshell: keeps the step visible
  # for stop_otela's scancel; output redirected INSIDE (srun-side redirects
  # buffer) to $RUNDIR/otela.log.
  srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
       -J "$OTELA_STEP_NAME" \
       bash -c '"$0" >"$1" 2>&1' "$RUNDIR/otela.sh" "$RUNDIR/otela.log"
  echo "[$(date -Is)] otela worker exited rc=$?"
) &
REG_PID=$!
echo "registration subshell pid $REG_PID; run dir $RUNDIR"

# ======================================================== the engine itself ==
# FOREGROUND srun (inherits the job's gres; kimi-k3 JSC pattern): when the
# engine dies, the batch script proceeds to the trap (stop_otela -> clean
# LEFT) and exits with the engine's exit code.
srun --jobid="$SLURM_JOB_ID" --nodes="$NNODES" --ntasks-per-node=1 \
     bash "$RUNDIR/engine.sh"
ENGINE_RC=$?

kill "$REG_PID" 2>/dev/null || true
wait "$REG_PID" 2>/dev/null || true
stop_otela

if [ "$ENGINE_RC" = 0 ]; then
  echo "engine exited 0 (natural end of the time window); last service:"
  cat "$DEPLOY_DIR/last_service.env"
fi
echo "done (engine rc=$ENGINE_RC)"
exit "$ENGINE_RC"
