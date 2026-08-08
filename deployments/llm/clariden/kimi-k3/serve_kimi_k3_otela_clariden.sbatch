#!/bin/bash
#SBATCH --job-name=k3-otela-clariden
#SBATCH --account=infra02
#SBATCH --partition=normal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
# Full node (288 logical CPUs): the sglang kimi-k3 image pins TP ranks to GPU
# NUMA domains using host CPU ids; a shrunken cpuset raises "CPU number N is
# not eligible" and kills ranks. 1 task/node x 288 cpus = whole node.
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
# Request the whole node's memory, not DefMemPerNode (~450 GiB on clariden).
# servekit launch stages this rank's pre-sharded weights to /dev/shm (tmpfs),
# and tmpfs pages are charged to the step's cgroup. At TP4xPP8 kimi-k3 stages
# ~190 GiB/node on top of the ranks (~100 GiB); DefMemPerNode fits but leaves
# little headroom. 850 GB < the node's RealMemory (870 GB), so the job still
# schedules, and --exclusive keeps /dev/shm free of other tenants.
#SBATCH --mem=850000
#SBATCH --exclusive
# B:TERM@120 signals the BATCH SHELL (B:) 120 s before wall-time, giving the
# stop_otela trap below a window to scancel the otela step with TERM. Without
# B: the signal goes to the job steps, the batch-shell trap never runs, and the
# only grace is KillWait (30 s) — less than otela's ~10 s graceful leave
# (AnnounceLeave -> 5 s CRDT drain -> srv.Shutdown), so no LEFT is published and
# the head keeps a `connected: true` row round-robining traffic into a dead
# endpoint.
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
# What it does, in order:
#   1. preflights the image, weights, and otela binary (prints exact staging
#      commands on failure)
#   2. srun's one sglang engine rank per node INSIDE the kimi-k3-clariden
#      container (enroot/EDF), forming a single distributed TP4 x PP<NNODES>
#      engine. Only rank 0 (head) serves HTTP.
#   3. once the head's /health answers, runs the servekit verification bench
#      (pre-registration, so mesh traffic can't pollute it; SERVEKIT_BENCH=0
#      skips), then starts ONE otela WORKER — also inside the container, on
#      the head node via srun --overlap --gres=none — which registers the
#      `llm` service with the OpenTela mesh
#   4. on job end / scancel, SIGTERM (never KILL) the otela step so it
#      announces LEFT cleanly
#
# -----------------------------------------------------------------------------
# THE TOPOLOGY: JSC engine shape, Beverin network, Clariden fabric
# -----------------------------------------------------------------------------
# Kimi-K3 is ~1.5 TB mxfp4 — it does NOT fit on one 4x96 GB GH200 node, so this
# is NOT the Beverin per-node-independent-replicas pattern. It is the JSC
# pattern: ONE distributed sglang engine across all nodes, ONE otela worker on
# the head registering the single head HTTP endpoint.
#
#   clariden compute (host netns; enroot container shares it)
#     rank 0 (HEAD)  -> sglang launch_server (HTTP on 0.0.0.0:$SERVE_PORT) + otela
#     rank 1..N      -> sglang workers (no HTTP; pipeline P2P to neighbours)
#
# TP4 keeps tensor-parallel all-reduce INSIDE one node's 4-GH200 NVLink domain.
# PP<NNODES> puts one pipeline stage per node, so the only traffic that crosses
# the Slingshot fabric is pipeline send/recv (point-to-point) — bandwidth-
# tolerant and CUDA-graph-safe. Cross-node allreduce (TP32 flat) boots but is
# ~1-5 tok/s and cannot be captured in a CUDA graph on Slingshot/CXI
# (cudaErrorStreamCaptureInvalidated; OFI logs "GIN only supports RDMA
# transport protocol"). See the README's fix 2 for the details.
#
# Verified performance (job 3000965, 2026-08-04, 8 nodes TP4xPP8; measured
# with the legacy meta/bench/bench.py — the cookbook harness is now
# meta/bench/cbench.sh (servekit bench), same protocol: 1024-in / 256-out,
# warmup discarded, HiCache OFF):
#   C=1  -> 34.7 tok/s   (single-request decode)
#   C=16 -> 374.9 tok/s  aggregate
#   C=32 -> 561.4 tok/s  aggregate  <- operating point; curve flattens here
#   C=64 -> 559.3 tok/s  aggregate  (per-req drops 17.6 -> 11.7 tok/s)
# Cold start ~105 min (weight load from /capstor dominates). Raw results:
# bench_3000965.jsonl. Full protocol + table: README "Verified benchmarks".
#
# -----------------------------------------------------------------------------
# NO RELAY (direct mesh, like Beverin)
# -----------------------------------------------------------------------------
# Clariden compute nodes reach the public OpenTela bootstrap
# /ip4/140.238.223.116/... directly. (The previous Alps peer
# /ip4/148.187.108.178/... was verified reachable from a compute node via
# `cat < /dev/null > /dev/tcp/148.187.108.178/43905`; re-verify egress to
# 140.238.223.116:43905 the same way before the first run.) So the worker
# dials the bootstrap directly — no us-managed login-node relay (unlike JSC,
# whose compute nodes have no outbound internet).
#
# -----------------------------------------------------------------------------
# CONTAINER + NCCL ON SLINGSHOT (the clariden-specific part)
# -----------------------------------------------------------------------------
# Image is docker://lmsysorg/sglang:kimi-k3 (CUDA 13) imported to a local
# .sqsh (build_kimi_k3_image.sbatch). The EDF (kimi-k3-clariden.toml, found via
# EDF_PATH below) carries com.hooks.aws_ofi_nccl.enabled=true. Clariden defaults
# to netstack source=artifact, so the hook bind-mounts the dynamically-linked
# (+dl) aws-ofi-ccl plugin + libfabric CXI provider at container start (the
# com.hooks.aws_ofi_nccl.variant annotation is ignored on artifact source).
# That gives NCCL the native Slingshot transport — clariden has NO InfiniBand
# verbs, so NCCL's built-in IB transport is unused. The per-rank engine.sh
# below adds FI_CXI_* / NCCL_* tunables as := fallbacks that yield to whatever
# the hook sets (reconciled with CSCS resource-hook docs, 2026-08).
# =============================================================================

set -uo pipefail

# ----------------------------------------------------- script + deploy dirs --
# Locate the EDF ($OTELA_EDF_NAME.toml) regardless of the submit working
# directory, so both submit styles work:
#   - manual:  `cd <deploy-dir> && sbatch serve_*.sbatch`  (SLURM_SUBMIT_DIR
#     is the deploy dir, so the EDF is a sibling there)
#   - rcc:     `rcc job submit deployments/.../serve_*.sbatch` runs sbatch from
#     the repo root (remote_dir), so SLURM_SUBMIT_DIR is the ROOT and the EDF
#     lives in the deployments/llm/clariden/kimi-k3 subdirectory.
# Without this search pyxis fails at the preflight srun with "failed to find a
# readable environment file" (observed 2026-08-08 on jobs 3032795/3034577,
# submitted from the root; the prior successful runs were submitted manually
# from the subdir). Slurm copies the sbatch to /var/spool/slurmd/job<id>/ and
# runs it from there, so ${BASH_SOURCE[0]} is the SPOOL copy and is NOT the
# EDF dir — BASH_SOURCE is only a last-resort fallback for non-Slurm runs.
# Keep ~/.edf on EDF_PATH so the pre-existing ~/.edf/kimi-k3.toml remains
# usable (override with OTELA_EDF_NAME=kimi-k3).
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
unset _edf_dir _cand

# ---------------------------------------------------------------- deployment --
# Reuses the existing /capstor/scratch/cscs/xyao/kimi-k3 tree: the imported
# image, the EDF cache dirs, and a warm Triton cache already live here, so a
# fresh submit does not re-pull or re-JIT from cold.
export DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/kimi-k3}"
export IMAGE="${IMAGE:-$DEPLOY_DIR/images/sglang-kimi-k3.aarch64.sqsh}"
export MODEL_PATH="${MODEL_PATH:-/capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K3}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3}"

# ------------------------------------------------------------------- shape ----
# WHY TP4 x PP<NNODES> is the default (verified on clariden):
# TP4 keeps collectives on the intra-node 4-GH200 NVLink domain. PP<NNODES>
# stages one pipeline layer per node, so only point-to-point send/recv crosses
# Slingshot — stable and CUDA-graph-safe. The flat TP32 alternative (8 nodes x
# 4 GPUs = 32) boots but every MoE allreduce goes cross-node over CXI without
# graph capture (~1-5 tok/s measured; CG_DECODE must be `disabled`). See README
# §2. Default PP_SIZE = NNODES (one stage/node); EP_SIZE = TP_SIZE (K3 requires
# ep_size * moe_dp_size == tp_size).
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
# CUDA graph for DECODE only. TP4 x PP8 keeps every captured collective
# intra-node on NVLink, so `full` works. If you switch to TP32 flat you MUST
# set CG_DECODE=disabled (cross-node allreduce cannot be captured on CXI).
export CG_DECODE="${CG_DECODE:-full}"
# Keep `marlin`: the only SM90 MoE backend that keeps K3's mxfp4 weights packed
# (W4A16). `auto` would pick Triton-Kernels, whose upcast_from_mxfp4() dequantizes
# to bf16 and OOMs a 4-bit model on 96 GiB. (Same finding as the JSC recipe.)
export MOE_BACKEND="${MOE_BACKEND:-marlin}"
# Start modest; raise once the boot log reports the KV pool size. Bounds request
# length, not the KV pool (KV pages are allocated on demand).
# Request-length bound. Kimi-K3 supports up to max_position_embeddings=
# 1048576 (1M). The KV pool is sized by MEM_FRAC, not this value: at 0.90 it
# holds 3,732,608 tokens (8.01 GB/GPU, job 3000965) — hybrid model (mostly
# linear/KDA attention, MLA-compressed full-attn layers), so even ~3
# concurrent 1M-token requests fit. --context-length cannot be changed on a
# running server; it only gates request admission.
export CTX_LEN="${CTX_LEN:-1048576}"
export MEM_FRAC="${MEM_FRAC:-0.90}"
# Slingshot high-speed NIC. Every clariden GH200 node exposes hsn0..hsn3 (one
# per GPU NUMA domain); hsn0 is used for the NCCL/GLOO socket bootstrap and is
# where SGLANG_HOST_IP is read from. The ofi plugin handles the data path over
# CXI independently of this setting.
export K3_NIC="${K3_NIC:-hsn0}"
# PyTorch distributed init + Store.wait() timeout (seconds), AND — via
# sglang's --dist-timeout -> init_process_group(timeout=...) -> the default
# for every ProcessGroupNCCL collective — the NCCL operation-watchdog floor.
# That floor is what killed clariden job 3018155 (fix 12): a PP pipeline
# SEND (SeqNum=3678376, NumelIn=3584) stalled at 16:08:43 after 5h20m of
# healthy serving, and the ProcessGroupNCCL watchdog tripped at 600 s
# (sglang's DEFAULT dist-timeout, which this recipe used to NOT pass) and
# aborted every rank ~5.5 h before the sglang scheduler watchdog (fix 11,
# 3600 s) could ride it out. Default now 3600 so the NCCL floor matches the
# scheduler watchdog; the SOFT_WATCHDOG_TIMEOUT dump (600 s, no crash) is
# the first diagnostic. SGLANG_EXTRA_ARGS="--dist-timeout N" (which comes
# later on the cmdline) overrides this for a shorter collective failure.
export DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"

# --- Scheduler watchdog: survive transient PP-broadcast stalls ------------
# sglang's HARD scheduler watchdog (ServerArgs.watchdog_timeout, default 300 s,
# soft=False) raises on any forward batch that runs longer than the limit, and
# that raise KILLS the whole distributed engine. Verified the hard way on
# clariden job 3002366 (1M, TP4xPP8): the engine served healthy 04:20->12:24
# (steady decode ~100 tok/s, cuda graph True, POST /v1 200 OK through 12:23:53),
# then at 12:29:35 EVERY PP stage tripped simultaneously:
#     [2026-08-05 12:29:35 PP2 TP3 EP3] Scheduler watchdog timeout
#        (self.watchdog_timeout=300, self.soft=False)
#     ... (PP2 TP1, PP7 TP2, PP7 TP3, ...) -> 16 tracebacks total.
# The blocked scheduler thread was in torch.distributed.broadcast_pyobj ->
# _broadcast_reqs_across_ranks (scheduler_components/request_receiver.py) ->
# recv_requests -> event_loop_pp, i.e. the per-iteration PP request-metadata
# broadcast hung >300 s. The raise tore the engine down; otela could not
# announce LEFT ("WARN: otela did not announce LEFT") -> stale registry row.
# Root cause of THAT particular stall is TODO(unverified) (Slingshot/CXI
# hiccup after 8 h? a specific request?), but a >5-min hard kill on a
# healthy 8 h-serving engine is the same anti-pattern this recipe already
# rejects for weight loading (see the UNBALANCED_MODEL_LOADING_TIMEOUT_S
# 480->3600 patch + README op-notes: "transient pauses are normal — check
# before killing").
#
# FIX (flags verified via `sglang serve --help` in this aarch64 image):
#   --watchdog-timeout      HARD: "the server will crash to prevent hanging".
#                           Raised to 3600 s so a transient distributed
#                           stall rides out (1 h matches the load-barrier
#                           precedent); a true permanent hang still dies in
#                           finite time instead of leaving a zombie serving.
#   --soft-watchdog-timeout SOFT: "the server will dump information for
#                           debugging". At 600 s it dumps a stack trace (like
#                           the one that diagnosed 3002366) WITHOUT crashing,
#                           so the next stall is diagnosable even when the
#                           engine survives it. Keep soft < hard.
export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-3600}"
export SOFT_WATCHDOG_TIMEOUT="${SOFT_WATCHDOG_TIMEOUT:-600}"

# --- NCCL process-group watchdog (fix 12: clariden job 3018155) ------------
# Fix 11 raised only sglang's SCHEDULER watchdog. 3018155 proved that is NOT
# the only watchdog that can tear the engine down on the same transient PP
# stall: PyTorch's ProcessGroupNCCL::Watchdog has its own timeout, taken
# from init_process_group(timeout=...) == sglang --dist-timeout (default
# 600 s). With the recipe not passing --dist-timeout, the NCCL floor sat at
# 600 s and fired ~5.5 h before the scheduler watchdog (3600 s) would have.
# Symptom: "Watchdog caught collective operation timeout: WorkNCALL(...,
# OpType=SEND, NumelIn=3584, Timeout(ms)=600000) ran for 600003 milliseconds"
# -> "Fatal Python error: Aborted" on every rank -> "Connection closed by
# peer [172.28.46.x]" cascade -> "WARN: otela did not announce LEFT". The
# DuplicateTimeseries/OSError lines that follow are red herrings (a sglang
# HTTP restart dying on the already-registered prometheus registry + the
# quota-/users tmpdir cleanup of fix 8). Fix: pass --dist-timeout
# $DIST_TIMEOUT (default 3600 above) so the NCCL floor matches the
# scheduler watchdog; a transient send/recv stall now rides out at 10 min
# and the SOFT_WATCHDOG_TIMEOUT dump is the first signal, not a kill.

# --- Weight-loading barrier tolerance (fixes clariden job 2999818) --------
# sglang's dist_barrier_after_load calls dist.monitored_barrier(group=
# tp_group, timeout=UNBALANCED_MODEL_LOADING_TIMEOUT_S, wait_all_ranks=True).
# The constant is HARDCODED at 480 s (8 min) in load_model_utils.py. The
# barrier is per-TP-group (GLOO, 4 ranks/node); only rank 0 has the timeout
# — non-zero ranks block indefinitely waiting for rank 0. If the slowest
# rank in a node happens to be non-zero, rank 0 waits 480 s then raises
# ValueError, killing the whole job. The proven run (2920471) survived a
# 1555 s within-PP1 spread ONLY because the slowest rank was rank 0 (no
# wait). We monkey-patch the constant to 3600 s (1 h) via sitecustomize.py
# so even a non-zero slow rank gets up to 1 h to finish loading from Lustre.
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
# arm64 build (sai-v0.0.6), statically linked -> runs on the host and inside the
# container (mounted via /capstor). It is started INSIDE the kimi-k3 container
# on the head so its HOME (set by the EDF to /capstor, not quota /users) is
# correct for its libp2p identity + badger DB. The x86 binary at the same dir
# is NOT usable on clariden (Exec format error).
export OTELA_DIR="${OTELA_DIR:-/capstor/scratch/cscs/xyao/opentela}"
export OTELA_BIN="${OTELA_BIN:-$OTELA_DIR/otela-arm64}"
# Shared alternative: /capstor/store/cscs/swissai/infra01/ocf-share/otela-arm64
# Direct-mesh bootstrap (an Alps OpenTela peer). No relay needed.
export OTELA_RELAY_ADDR="${OTELA_RELAY_ADDR:-/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7}"
export OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
export OTELA_TCP_PORT="${OTELA_TCP_PORT:-43905}"
export OTELA_UDP_PORT="${OTELA_UDP_PORT:-59820}"
OTELA_STEP_NAME="${OTELA_STEP_NAME:-otela-worker-$SLURM_JOB_ID}"
OTELA_SEED="${OTELA_SEED:-21}"
OTELA_API_PORT="${OTELA_API_PORT:-18094}"
# Cold start is ~105 min on a bad Lustre day (job 3000965: 17:29→19:14).
# Weight loading from /capstor dominates (~80–132 s/shard × 96 shards).
# 9000 s (150 min) gives 45 min margin; the distributed-load-barrier
# monkey-patch (UNBALANCED_MODEL_LOADING_TIMEOUT_S=3600) is what actually
# prevents the load barrier from killing slow PP groups, but the health
# check must also wait long enough for the server to answer.
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-9000}"

# ------------------------------------------------------------- servekit -----
# servekit (github.com/eth-easl/servekit, BRANCH multinode-pp) provides:
#   1. presharded launch: engine.sh wraps `python -m sglang.launch_server`
#      with `servekit launch --overlap`, which stages this rank's pre-cut
#      shard set (read out of the dump's own checksum.json, not a glob — PP
#      stages hold DIFFERENT files) from $PRESHARDED_ROOT (Lustre) to
#      /dev/shm, rewrites --model-loader-extra-config to the staged copy,
#      profiles each cold-start phase, and writes $RUNDIR/run.node<RANK>.json
#      when the node reports ready (the head's row lands at full readiness).
#   2. verification bench: after /health answers and BEFORE otela
#      registration (so mesh traffic can't pollute the numbers), a fixed
#      servekit bench runs inside the container on the head and is merged
#      into the head's report. Protocol: meta/bench/README.md.
# Pure stdlib -> run module-style from this checkout (PYTHONPATH); no pip
# install inside the image. Missing checkout: run unwrapped from $MODEL_PATH
# (no preshard, no profile, slower cold start).
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

# sitecustomize.py is auto-imported by the Python interpreter at startup
# (after PYTHONPATH is on sys.path) — BEFORE sglang's own code runs. This
# lets us patch the module-level UNBALANCED_MODEL_LOADING_TIMEOUT_S in the
# SAME process as `sglang serve`. $PATCH_DIR is on /capstor (mounted into
# the container by the EDF). engine.sh prepends it to PYTHONPATH before
# exec'ing sglang, because the EDF sets PYTHONPATH="" (clearing anything
# the batch shell exported).
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
  echo "       Stage an arm64 build on a login node, e.g.:" >&2
  echo "         mkdir -p $OTELA_DIR && cp /capstor/store/cscs/swissai/infra01/ocf-share/otela-arm64 $OTELA_BIN && chmod +x $OTELA_BIN" >&2
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
# Validates the EDF actually loads + the aarch64 image has the two modules
# this recipe depends on, BEFORE launching 8 nodes. Both failures are real:
#   - flash_mla missing -> prefill/decode fall back to FA3 -> ImportError on
#     this image (job 2914289)
#   - sgl_kernel.kvcacheio missing -> HiCache with --hicache-io-backend kernel
#     would crash at first eviction
# Runs via --overlap --gres=none so it doesn't consume the job's GPU allocation.
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
# The EDF's com.hooks.aws_ofi_nccl hook (Clariden netstack source=artifact) sets
# CXI/NCCL env at container start "to help prevent application stalls" (per
# resource-hook docs). engine.sh runs INSIDE the container after the hook, so an
# unconditional export would clobber those curated values. Use := so the hook's
# values win when present; our values (tuned on clariden for K3 multi-node,
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
# Force IPv4 for bootstrap sockets (inherited from the JSC recipe; clariden hsn
# interfaces are IPv4 so this is a safe no-op there).
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-IPv4}" GLOO_SOCKET_FAMILY="${GLOO_SOCKET_FAMILY:-IPv4}"

echo "[rank $RANK] $(hostname) ip=$SGLANG_HOST_IP iface=$IFACE head=$HEAD_IP:$DIST_PORT"
echo "[rank $RANK] shape: tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE cuda_graph_decode=$CG_DECODE"

# sitecustomize.py (in $PATCH_DIR, on /capstor) patches
# UNBALANCED_MODEL_LOADING_TIMEOUT_S at Python startup, BEFORE sglang runs.
# The EDF sets PYTHONPATH="" so we must re-add it here, before exec sglang.
export PYTHONPATH="${PATCH_DIR:-/capstor/scratch/cscs/xyao/kimi-k3/patches}:${PYTHONPATH:-}"

# Verify the monkey-patch will take effect in this process tree
python3 -c "import sglang.srt.model_executor.model_runner_components.load_model_utils as m; print('[rank '$RANK'] UNBALANCED_MODEL_LOADING_TIMEOUT_S =', m.UNBALANCED_MODEL_LOADING_TIMEOUT_S)" 2>/dev/null || true

# servekit launch wraps the engine: it stages this rank's pre-sharded slice
# (read out of the dump's own checksum.json — a glob can't express that PP
# stages hold DIFFERENT files) from $PRESHARDED_ROOT (Lustre) to /dev/shm,
# rewrites --model-loader-extra-config to the staged copy, profiles each
# cold-start phase, and writes $RUNDIR/run.node<RANK>.json when this node
# reports ready (head: "The server is fired up..."; workers: sglang's
# multi-node "Dummy health check server started" line — if a future image
# stops printing it, workers stay unprofiled and the profiler would
# terminate them at --timeout; check run.node<R>.json after image bumps).
# --overlap is SAFE for presharded (unlike sharded_state): the loader will
# not touch a dump without its READY sentinel, written last and only on a
# clean stage — an engine that looks too early takes a cache MISS (a full
# source load + re-dump into /dev/shm), not zeros; check overlap_gate in
# run.node<R>.json before trusting a timing. servekit forwards SIGTERM to
# the engine child and exits on child exit, so the batch shell's
# --signal=B:TERM@120 + stop_otela trap still fire unchanged. --timeout
# KILLS the engine on expiry, so it must track $HEALTH_TIMEOUT — never less.
# ulimit -c 0 + unset BASH_ENV: the stager forks subshells; BASH_ENV would
# run an nvidia-smi init in every one (16.89 -> 0.41 GB/s) and a 1.5 TB core
# dump is never useful.
ulimit -c 0
unset BASH_ENV

HICACHE_ARGS=()
if [ "${HICACHE_ENABLE:-0}" = "1" ]; then
  HICACHE_ARGS+=(--enable-hierarchical-cache
                 --hicache-ratio "${HICACHE_RATIO:-2.0}"
                 --hicache-write-policy "${HICACHE_WRITE_POLICY:-write_through_selective}"
                 --hicache-io-backend "${HICACHE_IO_BACKEND:-kernel}")
fi

# NOTE: prefill AND decode must BOTH be flashmla. This aarch64 sgl_kernel ships
# no FA3 (`flash_ops` absent, only `flash_mla`), and setting only one of the
# three attention knobs cancels auto-resolution for the others -> prefill
# silently falls back to FA3 and dies at ImportError (clariden job 2914289).
# FlashMLABackend imports only sgl_kernel.flash_mla and inherits prefill from
# FlashInferMLAAttnBackend. Do NOT apply --mamba-ssm-dtype bfloat16: SM90 uses
# float32 KDA state (FlashInfer GDN bf16 is SM100+ only).
# `python -m sglang.launch_server` (not `sglang serve`): the entry point
# servekit's detect_framework + the presharded loader are validated against,
# and the one the multinode-pp reference runs. The checkpoint's own
# config.json (still read from --model-path) supplies the architecture.
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

if [ -d "${SERVEKIT_DIR:-}/src/servekit" ]; then
  export PYTHONPATH="${SERVEKIT_DIR}/src:${PYTHONPATH:-}"
  # --load-format presharded + presharded_path (the ROOT holding the
  # TP-<tp>-sig-<hash>/ subdir): servekit stages to /dev/shm and rewrites
  # this to the staged copy. Left off the fallback below — without staging
  # the engine reads straight from $MODEL_PATH on Lustre (slow, unverified).
  ENGINE_CMD+=(--load-format presharded
               --model-loader-extra-config "{\"presharded_path\": \"$PRESHARDED_ROOT\"}")
  exec python3 -m servekit.cli launch \
    --out "$RUNDIR/run.json" \
    --timeout "${HEALTH_TIMEOUT:-9000}" \
    --overlap \
    -- "${ENGINE_CMD[@]}"
else
  echo "[rank $RANK] WARN: servekit not at ${SERVEKIT_DIR:-<unset>} — cold load from $MODEL_PATH (no preshard, no profile)" >&2
  exec "${ENGINE_CMD[@]}"
fi
ENGINE
chmod +x "$RUNDIR/engine.sh"

# ============================================================ otela worker ====
# Runs INSIDE the kimi-k3 container on the HEAD node (via srun --overlap
# --environment). Why inside the container, not the host: the enroot
# container SHARES the host netns (verified: a 2nd container on the head
# reaches the engine on localhost:$SERVE_PORT, and the host's `ss` sees the
# container's listening socket — job 2999317), so otela COULD reach sglang
# from the host too. But the EDF sets HOME=/capstor/.../home (on /capstor,
# not the quota-limited /users/xyao NFS home), and otela (sai-v0.0.6)
# resolves its libp2p identity + badger DB via os/user.Current() HOME. A
# host otela would inherit /users/xyao and, per the JSC recipe, freeze on
# stale NFS handles. One peer only: the DNT lists one row per peer, and
# launching on every node would over-register the same single endpoint.
OTELA_CFG_DIR="$DEPLOY_DIR/otela"
OTELA_CFG="$OTELA_CFG_DIR/cfg.yaml"
mkdir -p "$OTELA_CFG_DIR"
# otela is sai-v0.0.6 (arm64). Its CLI differs from the x86 opentela-v0.2.2
# the Beverin scripts use:
#   - a multiaddr goes to --bootstrap.static (NOT --bootstrap.addr, which
#     expects an HTTP bootstraps-list URL — the default is an http:// URL)
#   - --label is supported
#   - there is NO --subprocess flag (we use the JSC separate-worker topology:
#     one engine srun in the foreground + one otela srun on the head).
#
# CRITICAL: otela requires a proper cfg.yaml with `loglevel: debug`,
# `reachability: private`, `role: worker`, and `seed`. Without these it
# appears to "hang" after "AXIOM_DATASET not set, tracing disabled" — but it
# is actually working, just at INFO level which stops logging during CRDT
# sync. The JSC recipe (v0.2.3, x86) writes the same config file; verified on
# clariden job 3000965: with the config file, otela registered the LLM service
# and synced 2000+ peers within 45 s. Without it, 90 s of silence.
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
