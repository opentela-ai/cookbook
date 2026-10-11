#!/bin/bash
#SBATCH --job-name=glm53-otela-clariden
#SBATCH --account=infra02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
# Whole node (288 logical CPUs): the image pins TP ranks to GPU NUMA domains
# via host CPU ids; a shrunken cpuset kills ranks.
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
# No --mem/--exclusive: normal is OverSubscribe=EXCLUSIVE (whole nodes);
# /dev/shm is a fixed 334 GiB tmpfs per node.
# B:TERM@120 gives the stop_otela trap a TERM window before KillWait —
# without B: no LEFT is published and the head round-robins into a dead
# endpoint.
#SBATCH --signal=B:TERM@120
#SBATCH --output=/iopsstor/scratch/cscs/xyao/glm-53-flash/logs/serve-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyao/glm-53-flash/logs/serve-%j.out
#
# =============================================================================
# Serve zai-org/GLM-5.3-Flash (~306 GiB FP8 e4m3, DSA + linear + KDA hybrid)
# on CSCS Clariden (4x GH200/node, aarch64, Slingshot) + OpenTela — one
# self-contained sbatch.
# Submit variants (incl. the dummy-weight fast smoke), the flow, registration
# details and the full "why" essays: README.md.
# Flow: preflight -> one sglang rank/node in the EDF container (TP4 x
# PP<NNODES>, head serves HTTP) -> /health -> MANDATORY gen probe -> ONE
# otela worker on the head -> SIGTERM/LEFT on end.
# Topology: one distributed engine (JSC/kimi-k3 pattern). TP4 keeps allreduce
# on NVLink; PP<NNODES> = only point-to-point send/recv crosses Slingshot
# (flat TP8: ~1-5 tok/s, no CUDA-graph capture — kimi-k3 finding). 2-node
# TP4xPP2: ~38 GiB weights/GPU, ~58 GiB free; 45/2 = 23+22 layers/stage.
# Container + NCCL: EDF hooks aws_ofi_nccl (netstack source=artifact
# bind-mounts the +dl plugin + CXI provider; no IB verbs on clariden);
# engine.sh's FI_CXI_*/NCCL_* tunables are := fallbacks yielding to the hook.
# =============================================================================

set -uo pipefail

# ----------------------------------------------------- script + deploy dirs --
# Locate the EDF for both submit styles (manual from the recipe dir; rcc from
# the repo root — kimi-k3 jobs 3032795/3034577 are why this search exists).
# BASH_SOURCE is the spool copy — last resort only. ~/.edf kept for legacy.
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

# --------------------------------------------------------------------- lib ---
# Shared helpers from the repo's meta/common/ — lib.sh (stop_otela),
# gen_correctness.py, download_otela.sh; per-recipe copies were removed. Walk
# up from the EDF dir (and, for repo-root submits, SLURM_SUBMIT_DIR) to find it.
COMMON_DIR=""
for _root in "$_edf_dir" "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
  [ -n "$_root" ] || continue
  _d="$_root"
  for _ in 1 2 3 4 5 6; do
    [ -f "$_d/meta/common/lib.sh" ] && { COMMON_DIR="$_d/meta/common"; break 2; }
    _d="$(dirname "$_d")"
  done
done
[ -n "$COMMON_DIR" ] || { echo "FATAL: meta/common/lib.sh not found above $SCRIPT_DIR — submit from within the repo checkout" >&2; exit 1; }
source "$COMMON_DIR/lib.sh"   # stop_otela + teardown contract
unset _edf_dir _cand _root _d

# ---------------------------------------------------------------- deployment --
# Reuses the /iopsstor tree (image, caches, warm Triton — no re-pull/re-JIT).
export DEPLOY_DIR="${DEPLOY_DIR:-/iopsstor/scratch/cscs/xyao/glm-53-flash}"
# Shared GLM-5.3 diagnostic harness — identical copies on beverin + clariden
# (one source of truth).
export GLM53_DIAG_DIR="${GLM53_DIAG_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)/meta/diag/glm53}"
export IMAGE="${IMAGE:-$DEPLOY_DIR/images/sglang-glm-5.3-flash.aarch64.sqsh}"
# Weights on /iopsstor/scratch (the /capstor/store MDT was overloaded during
# download — see download_glm_5_3_flash.sbatch).
export MODEL_PATH="${MODEL_PATH:-/iopsstor/scratch/cscs/xyao/glm-53-flash/models/GLM-5.3-Flash}"
# Served under the canonical HF id (coexists with kimi-k3 / glm47 peers).
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-5.3-Flash}"

# ------------------------------------------------------------------- shape ----
# TP4 x PP<NNODES>: allreduce stays on NVLink, only pipeline send/recv
# crosses Slingshot (TP8 flat: ~1-5 tok/s, no graph capture). PP=NNODES.
export NNODES="${NNODES:-$SLURM_JOB_NUM_NODES}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-$NNODES}"
# EP=TP: 288 routed experts / 4 = 72 experts/GPU, dispatch stays intra-node.
export EP_SIZE="${EP_SIZE:-$TP_SIZE}"

# FP8 e4m3 dynamic (NOT mxfp4) — mature native path on sm_90; `auto` picks.
export MOE_BACKEND="${MOE_BACKEND:-auto}"

# Attention backends. HYBRID: 34/45 linear/KDA (mamba_backend=triton, no
# flash dep) + 11/45 DSA (every 4th layer). THE kpool>1 HARD GATE (jobs
# 3208159/3208291/3209156; full trail in README "Site fixes"): GLM-5.3 is
# multimodal with dsa_index_kpool > 1, guarded twice — a perf override
# (_resolve_kpool_tail_backend) that sends flashmla_sparse -> fa3 on sm_90,
# and a hard raise (_check_kpool_tail_backend) on EVERY forward unless the
# backend is fa3/tilelang/trtllm. On this aarch64 image fa3 (flash_ops) and
# trtllm are MISSING; tilelang handles the kpool tail natively => pin BOTH
# DSA knobs to tilelang (forward_extend AND forward_decode hit the gate).
# Forcing flashmla_sparse (the kimi-k3 fix) trips the hard gate (3209156).
# Do NOT pass the kimi-k3 GENERAL attention knobs — they'd break the 34
# linear layers. PERF TODO(unverified): tilelang on sm_90 may be slower than
# fa3 on x86; revisit if a future image ships flash_ops.
export GLM53_DSA_PREFILL="${GLM53_DSA_PREFILL:-tilelang}"
export GLM53_DSA_DECODE="${GLM53_DSA_DECODE:-tilelang}"
export CG_DECODE="${CG_DECODE:-full}"

# Multimodal (vision-tower) attention: platform default on sm_90 is fa3 ->
# ImportError (job 3209753) — the THIRD FA3 path, independent of the DSA
# backends. Pin triton_attn (Triton kernels, zero flash_ops call sites).
export GLM53_MM_ATTENTION="${GLM53_MM_ATTENTION:-triton_attn}"

# Reasoning + tool-call parsers (glm45 / glm47, verified in this image;
# cookbook convention). Gated: GLM53_ENABLE_PARSERS=0 if an image bump drops
# them ('invalid ... parser' startup errors first). /v1/completions (the
# mandatory probe) is unaffected; these flags only affect chat completions.
export GLM53_REASONING_PARSER="${GLM53_REASONING_PARSER:-glm45}"
export GLM53_TOOL_CALL_PARSER="${GLM53_TOOL_CALL_PARSER:-glm47}"
export GLM53_ENABLE_PARSERS="${GLM53_ENABLE_PARSERS:-1}"

# 262144 (256K): pi's catalog advertises max-out 131072 and the gateway
# 400s anything above the advertised max_model_len, so 32768 broke every
# default pi request. Live-KV 287104 at MEM_FRAC=0.88 (job 3269479) — a
# 256K request fits at 91%; max_running_requests stays 17 (the Mamba
# ssm_state cap, not KV). 1M needs HICACHE_ENABLE=1 first.
export CTX_LEN="${CTX_LEN:-262144}"
export MEM_FRAC="${MEM_FRAC:-0.88}"

# Slingshot NIC: hsn0 for the NCCL/GLOO bootstrap; the ofi plugin owns the
# CXI data path independently.
export GLM53_NIC="${GLM53_NIC:-hsn0}"

# --- HiCache: offload evicted radix prefixes to Grace LPDDR ----------------
# Grace LPDDR (~870 GiB) hangs off NVLink-C2C at ~900 GB/s; HiCache offloads
# EVICTED prefixes there — grows the prefix cache, not max_total_num_tokens.
# This is what makes a 1-node TP4 serve viable (~19 GiB HBM headroom after
# 306 GiB weights). Verified on kimi-k3 (jobs 3008263/3009024/3010201): no
# throughput cost, ~59 GB/node at ratio 1.0. write_back is the only working
# policy in the hybrid-SSM path (write_through_selective never backs up);
# io=kernel uses sgl_kernel.kvcacheio (preflight checks it). README "Site fixes".
export HICACHE_ENABLE="${HICACHE_ENABLE:-0}"
export HICACHE_RATIO="${HICACHE_RATIO:-1.0}"
export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_back}"
export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-kernel}"

# --- Scheduler + NCCL watchdogs (fixes 11/12/13 from kimi-k3 clariden) ------
# Three watchdogs can kill a healthy engine on a transient stall (README
# "Site fixes" has the job trail): HARD scheduler watchdog (300 s default,
# job 3002366) -> 3600; SOFT watchdog (600 s stack dump, no crash — the
# diagnostic; keep soft < hard); ProcessGroupNCCL watchdog (--dist-timeout,
# default 600 s, job 3018155 fired 5.5 h before the scheduler watchdog)
# -> 3600.
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

# --- /health must NOT do a generation round-trip ---------------------------
# This fork defaults health-generation TRUE: GET /health fires a real
# generate gated on the detokenizer heartbeat; a heartbeat stall 503s a
# healthy engine AND otela withdraws the ad after 3 failed probes -> gateway
# 503 "No provider found" (clariden job 3419248). =0 returns 200
# immediately. Same conclusion as beverin's CORRECTNESS_BUG_INVESTIGATION.md.
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION:-0}"
# Headroom for the pre-registration /health gate: the engine gates on /health
# BEFORE GEN_PROBE, and 20 s is too short while 306 GiB of weights stream in.
export SGLANG_HEALTH_CHECK_TIMEOUT="${SGLANG_HEALTH_CHECK_TIMEOUT:-300}"

# --- MTP / nextn: OFF (draft path unvalidated on this family; the model's
# one nextn layer is unused without --speculative-algorithm EAGLE).
export GLM53_SPECULATIVE="${GLM53_SPECULATIVE:-0}"

# --- Load format: auto = real weights; dummy = fast forward-path smoke
# (pair with GEN_CORRECTNESS_SMOKE=1) — how the kimi-k3/glm-5.2 hangs were
# isolated.
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# ------------------------------------------------------------------- otela ----
# Official OpenTela v0.2.6 arm64 (staged by download_otela.sh), run INSIDE
# the container on the head. Source-audited CLI/behavior + the v0.2.6
# differences that matter: README "OpenTela registration" (config-dir, HOME
# override, withdraw-after-3, account linking, require_signed_binary).
# sai-v0.0.6 is retired; rollback = OTELA_VERSION=v0.2.5 re-stage.
# The x86 binary in the same dir is NOT usable on clariden (Exec format error).
export OTELA_DIR="${OTELA_DIR:-/iopsstor/scratch/cscs/xyao/opentela}"
export OTELA_BIN="${OTELA_BIN:-$OTELA_DIR/opentela-arm64-latest}"   # official release; up.sh stages via meta/common/download_otela.sh
# Direct-mesh bootstrap (Alps peer, verified from a compute node). No relay
# — clariden compute has outbound internet (the 306 GiB HF download proved it).
export OTELA_RELAY_ADDR="${OTELA_RELAY_ADDR:-/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7}"
export OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
export OTELA_TCP_PORT="${OTELA_TCP_PORT:-43905}"
export OTELA_UDP_PORT="${OTELA_UDP_PORT:-59820}"
OTELA_STEP_NAME="${OTELA_STEP_NAME:-otela-worker-$SLURM_JOB_ID}"
# Fresh peer id per job (per-job config-dir + $$ seed): a constant seed
# cannot re-register until its LEFT tombstone expires (euler/dgx-spark
# stall). Pin only knowingly.
OTELA_SEED="${OTELA_SEED:-$$}"
OTELA_API_PORT="${OTELA_API_PORT:-18094}"
# --- OpenTela Cloud account linking (v0.2.6 deploy keys; best-effort) -----
# Key resolution: OTELA_DEPLOY_KEY env > OTD_API_KEY parsed out of ~/.zshrc
# (bash never sources it). Lands in the per-job cfg.yaml (chmod 600);
# linking NEVER gates registration/serving — failures WARN and continue.
# Cost: fresh peer id per job = one deploy-key use per renewal (~2/day) —
# README "OpenTela registration".
OTELA_DEPLOY_KEY="${OTELA_DEPLOY_KEY:-}"
if [ -z "$OTELA_DEPLOY_KEY" ] && [ -f "$HOME/.zshrc" ]; then
  OTELA_DEPLOY_KEY="$(grep -E '^[[:space:]]*(export[[:space:]]+)?OTD_API_KEY=' "$HOME/.zshrc" 2>/dev/null | tail -1 \
                    | grep -oE 'otd-[A-Za-z0-9_-]+' | head -1)"
fi
[ "$OTELA_DEPLOY_KEY" = "none" ] && OTELA_DEPLOY_KEY=""
case "$OTELA_DEPLOY_KEY" in
  otd-?*) ;;   # well-formed deploy key
  "")        ;;   # not configured: linking simply disabled
  *) echo "[$(date -Is)] WARN: OTELA_DEPLOY_KEY/OTD_API_KEY is not a deploy key (expected otd-...); skipping account linking" >&2
     OTELA_DEPLOY_KEY="" ;;
esac
OTELA_LINK_ARMED=0; [ -n "$OTELA_DEPLOY_KEY" ] && OTELA_LINK_ARMED=1

# Cold start: 62 shards x ~5 GiB = ~306 GiB; up to ~2.5 h on a bad Lustre
# day. HEALTH_TIMEOUT gates the pre-registration wait only (the barrier
# patch prevents slow-rank kills). Dummy smoke: HEALTH_TIMEOUT=1200.
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-10800}"

# ----------------------------------------------------- mandatory gen probe --
# /health 200 does NOT prove the DSA + linear + KDA hybrid forward works
# (deepseek-v4 502'd; kimi-k3 hung; glm-5.2 hung at decode). gen_correctness
# (meta/common) sends six greedy prompts + checks factual substrings;
# SMOKE=1 gates on non-empty continuation. GEN_PROBE=0: skip (NOT recommended).
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

# sitecustomize.py: auto-imported at interpreter startup (PATCH_DIR first on
# PYTHONPATH via engine.sh) — patches sglang in the serve process itself.
#   1. kimi-k3 fix 10: UNBALANCED_MODEL_LOADING_TIMEOUT_S 480 -> 3600 (the
#      post-load monitored_barrier hardcodes 480 s on rank 0; a 29-min rank
#      spread was observed on kimi-k3 job 3000965).
#   2. kpool-tail FA3-safe (job 3208291): _resolve_kpool_tail_backend
#      hard-overrides flashmla_sparse -> fa3 on sm_90 assuming FA3 exists;
#      this patch keeps flashmla_sparse when fa3 is absent so the engine
#      dies with the CLEAR _check_kpool_tail_backend NotImplementedError
#      instead of the cryptic ImportError. INERT while the tilelang pins are
#      in effect (defensive); behavior-preserving when fa3/trtllm exist.
#   3. FOURTH FA3 path (jobs 3209880/3210211; the LAST one): the DSA
#      FULL-MHA prefill (_forward_standard_mha, dsa_backend.py:4131) calls
#      flash_attn_varlen_func ver=3 unconditionally on SM90 — no arch
#      guard, no CLI override. The SM100+ flashinfer branch is
#      Blackwell-only ("Unsupported architecture", job 3210211 — the
#      device_sm_major=10 trick was wrong). FIX below: replace the method
#      with a PyTorch SDPA ragged loop (correct, slower; fires ONLY on
#      not-_fa3_ok SM90).
# Patch details + histories: README "Site fixes" + clariden/glm53 diag docs.
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

# --- comp_capture (GLM53_COMP_CAPTURE=1): beverin-vs-clariden bisect --------
# Shared harness in $GLM53_DIAG_DIR (<cookbook>/meta/diag/glm53, exported by
# this sbatch): the SAME copy of comp_capture.py runs on beverin (via its
# sitecustomize dispatcher) and here — the identical-capture guarantee.
try:
    import os as _os_cc
    if _os_cc.environ.get("GLM53_COMP_CAPTURE", "0") == "1":
        import sys as _sys_cc
        _cc = _os_cc.environ.get("GLM53_DIAG_DIR", "")
        if not _cc:
            raise RuntimeError("GLM53_DIAG_DIR not set (this sbatch exports it)")
        if _cc not in _sys_cc.path:
            _sys_cc.path.insert(0, _cc)
        import comp_capture  # noqa: F401  (self-installing)
except Exception as _exc_cc:
    import sys as _sys_cc2
    _sys_cc2.stderr.write(f"[sitecustomize] comp_capture import failed: {_exc_cc!r}\n")
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
echo "[$(date -Is)] parsers=$GLM53_REASONING_PARSER/$GLM53_TOOL_CALL_PARSER/$GLM53_ENABLE_PARSERS  gen_probe=$GEN_PROBE  smoke=$GEN_CORRECTNESS_SMOKE  min_pass=$GEN_CORRECTNESS_MIN_PASS  max_tokens=$GEN_CORRECTNESS_MAX_TOKENS  load_format=$LOAD_FORMAT  speculative=$GLM53_SPECULATIVE  hicache=$HICACHE_ENABLE/$HICACHE_RATIO/$HICACHE_WRITE_POLICY"
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
  echo "       bash up.sh stages the latest official OpenTela (arm64) automatically" >&2
  echo "         OTELA_DIR=$OTELA_DIR OTELA_ARCH=arm64 bash ../../../../meta/common/download_otela.sh" >&2
  echo "       (or from the repo root: scripts/renew_glm_5_3_flash.sh --update-otela)" >&2
  echo "       Rollback: OTELA_VERSION=v0.2.5 bash meta/common/download_otela.sh" >&2
  exit 1
}
# gen_correctness.py (meta/common — mandatory probe, see above).
[ -f "$COMMON_DIR/gen_correctness.py" ] || {
  echo "FATAL: gen_correctness.py not found at $COMMON_DIR/gen_correctness.py" >&2
  echo "       meta/common is part of the repo checkout — submit from within it." >&2
  echo "       Submit with: cd \"$(cd "$SCRIPT_DIR" && pwd)\" && sbatch $0" >&2
  exit 1
}

# --- deep_gemm JIT cache self-heal (jobs 3632183/3632230, 2026-10-10) ---
# deep_gemm hard-fails at the FIRST cached kernel missing kernel.cu or
# kernel.cubin ("Corrupted JIT cache directory"), lazily — one bad entry =
# one dead job. The shared EDF-HOME cache is validated up front: any
# incomplete entry quarantines the whole cache (~minutes re-JIT vs a dead
# 12 h job). README "Site fixes".
_DG_CACHE="$DEPLOY_DIR/home/.cache/sglang/deep_gemm/cache"
if [ -d "$_DG_CACHE" ]; then
  _nbad=0; _ntot=0
  for _d in "$_DG_CACHE"/kernel.*; do
    [ -d "$_d" ] || continue
    _ntot=$((_ntot+1))
    { [ -f "$_d/kernel.cu" ] && [ -f "$_d/kernel.cubin" ]; } || _nbad=$((_nbad+1))
  done
  if [ "$_nbad" -gt 0 ]; then
    _q="${_DG_CACHE}.corrupt-$(date +%Y%m%dT%H%M%S)"
    echo "[$(date -Is)] WARN: deep_gemm JIT cache has ${_nbad}/${_ntot} incomplete entries (missing kernel.cu/kernel.cubin) — quarantining to $_q; engine will re-JIT" >&2
    mv "$_DG_CACHE" "$_q" \
      || echo "[$(date -Is)] WARN: quarantine mv failed; deep_gemm may crash at capture" >&2
  fi
fi
unset _d _nbad _ntot _q _DG_CACHE

# --- container preflight (1 node, no GPUs, ~20 s) -----------------------
# Validates the EDF + the GLM-5.3 model class + sgl_kernel ops BEFORE
# launching N nodes (--overlap --gres=none: no GPU consumption).
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
# sgl_kernel.flash_ops is the symbol FA3 needs. On aarch64 it is MISSING
# (only flash_mla is present); the sitecustomize kpool-tail FA3-safe patch
# is ACTIVE precisely when flash_ops is absent on sm_90. Printing the
# import result here makes this fix self-documenting at runtime
# (job 3208291 crashed because the override assumed fa3 present).
for mod in ["sgl_kernel.flash_mla", "sgl_kernel.kvcacheio", "sgl_kernel.flash_ops"]:
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
# Pin NCCL/GLOO socket bootstrap to the high-speed NIC (ofi plugin owns the
# CXI data path independently).
export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE"
SGLANG_HOST_IP="$(ip -o -4 addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
[ -z "$SGLANG_HOST_IP" ] && SGLANG_HOST_IP="$(hostname -I | awk '{print $1}')"
export SGLANG_HOST_IP

# --- Slingshot NCCL tunables (reconciled with CSCS netstack hook) -----------
# The EDF's aws_ofi_nccl hook curates CXI/NCCL env at container start, so
# these are := fallbacks (hook values win). userfaultfd avoids the CXI
# registration-monitor bug. Tuned on clariden for kimi-k3 multi-node, 2026-08.
: "${NCCL_NET:=AWS Libfabric}" "${NCCL_CROSS_NIC:=1}" \
  "${FI_CXI_DISABLE_HOST_REGISTER:=1}" "${FI_CXI_DEFAULT_CQ_SIZE:=131072}" \
  "${FI_CXI_RDZV_THRESHOLD:=0}" "${FI_CXI_RDZV_GET_MIN:=0}" \
  "${FI_MR_CACHE_MONITOR:=userfaultfd}"
export NCCL_NET NCCL_CROSS_NIC FI_CXI_DISABLE_HOST_REGISTER \
  FI_CXI_DEFAULT_CQ_SIZE FI_CXI_RDZV_THRESHOLD FI_CXI_RDZV_GET_MIN \
  FI_MR_CACHE_MONITOR
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
echo "[rank $RANK] netstack: NCCL_NET=$NCCL_NET NCCL_CROSS_NIC=$NCCL_CROSS_NIC FI_CXI_DCQ=$FI_CXI_DEFAULT_CQ_SIZE FI_MR_CACHE_MONITOR=$FI_MR_CACHE_MONITOR"
# Force IPv4 bootstrap sockets (carried from kimi-k3; no-op on hsn).
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-IPv4}" GLOO_SOCKET_FAMILY="${GLOO_SOCKET_FAMILY:-IPv4}"

echo "[rank $RANK] $(hostname) ip=$SGLANG_HOST_IP iface=$IFACE head=$HEAD_IP:$DIST_PORT"
echo "[rank $RANK] shape: tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE moe=$MOE_BACKEND cg_decode=$CG_DECODE ctx=$CTX_LEN load_format=$LOAD_FORMAT"

# sitecustomize.py (in $PATCH_DIR, on /capstor) patches
# UNBALANCED_MODEL_LOADING_TIMEOUT_S at Python startup, BEFORE sglang runs.
# The EDF sets PYTHONPATH="" so we must re-add it here, before exec sglang.
export PYTHONPATH="${PATCH_DIR:-/iopsstor/scratch/cscs/xyao/glm-53-flash/patches}:${PYTHONPATH:-}"

# Verify the monkey-patch will take effect in this process tree.
python3 -c "import sglang.srt.model_executor.model_runner_components.load_model_utils as m; print('[rank '$RANK'] UNBALANCED_MODEL_LOADING_TIMEOUT_S =', m.UNBALANCED_MODEL_LOADING_TIMEOUT_S)" 2>/dev/null || true

# --- build the SGLang serve args -------------------------------------------
# python -m sglang.launch_server: the kimi-k3-validated entry point; the
# architecture comes from the checkpoint's own config.json (not hardcoded).
#
# CRITICAL DIFFERENCE vs kimi-k3: NO general attention knobs (they'd break
# the 34 linear layers); DSA-specific backends pinned to tilelang (BOTH:
# forward_extend AND forward_decode hit the hard gate).
#
# CRITICAL DIFFERENCE vs GLM-5.2 beverin: NO --disable-cuda-graph /
# --skip-server-warmup (MI300A workarounds; GH200 runs CG_DECODE=full). If
# capture/warmup crashes: set both =1 and re-test (TODO(unverified) finding).
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
  --enable-metrics --uvicorn-access-log-exclude-prefixes /metrics /health
  --host 0.0.0.0 --port "$SERVE_PORT")

# --- capture/diagnostic knobs (capture runs ONLY; see header TODO note) -----
# DISABLE_CUDA_GRAPH=1: hooks must see REAL kernel output — captured graphs
#   emit placeholders (beverin bisect lesson; jobs 617522/619846).
# SKIP_SERVER_WARMUP=1: sglang's warmup would be the first >=MIN_TOKENS
#   forward on clariden while beverin never sees it — the cross-machine
#   gen_correctness request on BOTH machines.
if [ "${DISABLE_CUDA_GRAPH:-0}" = "1" ]; then
  ENGINE_CMD+=(--disable-cuda-graph)
fi
if [ "${SKIP_SERVER_WARMUP:-0}" = "1" ]; then
  ENGINE_CMD+=(--skip-server-warmup)
fi

# GLM-5.3 reasoning + tool-call parsers (cookbook: glm45 / glm47).
# NOTE: this sglang build has NO --enable-auto-tool-choice (that's a vLLM
# flag); tool_choice is handled per-request. Just the two parser flags.
if [ "${GLM53_ENABLE_PARSERS:-1}" = "1" ]; then
  ENGINE_CMD+=(
    --reasoning-parser "$GLM53_REASONING_PARSER"
    --tool-call-parser "$GLM53_TOOL_CALL_PARSER"
  )
fi

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
# INSIDE the container on the HEAD only (one peer per engine; the container
# shares the host netns, so otela reaches sglang on localhost:$SERVE_PORT).
# Identity + BadgerDB per-job by design (fresh peer id; sai-v0.0.6 minted
# fresh ids too — runs 3219525 + 4 prior). "Stale row" note by stop_otela.
OTELA_CFG_DIR="$RUNDIR/otela"
OTELA_DATA_HOME="$RUNDIR/otela-home"
mkdir -p "$OTELA_CFG_DIR" "$OTELA_DATA_HOME"
# CRITICAL: cfg.yaml must carry loglevel: debug / reachability: private /
# role: worker / seed — without them otela looks "hung" after the AXIOM
# line (INFO stops logging during CRDT sync; kimi-k3 job 3000965: with the
# config, LLM registered + 2000+ peers synced in 45 s; without, 90 s of
# silence). CLI facts + registration markers: README.
OTELA_INSTANCE_CFG=""
if [ "$OTELA_LINK_ARMED" = 1 ]; then
  # v0.2.6: instance.deploy_key feeds BOTH `otela instance link` and the
  # boot-time re-confirm; instance.label is read by `otela instance link`
  # (boot's re-confirm passes an empty label and preserves the stored one).
  OTELA_INSTANCE_CFG="instance:
  deploy_key: \"$OTELA_DEPLOY_KEY\"
  label: clariden-glm53-$SLURM_JOB_ID"
fi
cat > "$OTELA_CFG_DIR/cfg.yaml" <<CFG
name: clariden-worker-$SLURM_JOB_ID
seed: "$OTELA_SEED"
port: "$OTELA_API_PORT"
tcpport: "$OTELA_TCP_PORT"
udpport: "$OTELA_UDP_PORT"
mode: node
role: worker
loglevel: debug
reachability: private
cleanslate: true
security:
  require_signed_binary: false
solana:
  skip_verification: true
$OTELA_INSTANCE_CFG
CFG
# cfg.yaml carries the deploy key when linking is armed -> keep it private
# (0600; the container reads it as the same user). The key never appears in
# otela.sh, the sbatch log, or any echo.
chmod 600 "$OTELA_CFG_DIR/cfg.yaml"
cat > "$RUNDIR/otela.sh" <<OTELA
#!/bin/bash
set -uo pipefail
# Config file is written by the batch shell (above). It carries loglevel,
# reachability, role, seed, ports, solana.skip_verification — everything that
# must be set before CRDT sync begins. CLI flags below are per-job overrides.
# HOME override: v0.2.6 puts the BadgerDB/CRDT store at \$HOME/.ocfcore
# (go-homedir honors \$HOME first), so this confines the store to the per-job
# run dir — no ocfcore.Qm*.db pile-up on the persistent tree.
export HOME="$OTELA_DATA_HOME"
echo "[otela] start head=$HEAD_IP service=$OTELA_SERVICE_NAME:$SERVE_PORT bootstrap=$OTELA_RELAY_ADDR seed=$OTELA_SEED home=$OTELA_DATA_HOME"
if [ "$OTELA_LINK_ARMED" = 1 ]; then
  # Link BEFORE start, with a label (boot's auto-link passes no label):
  # challenge-response signed with this config-dir's identity key, so the
  # linked peer id == the serving peer id. Best-effort -- on failure start
  # still boots and retries the link itself (also best-effort).
  echo "[otela] linking instance to OpenTela Cloud (label clariden-glm53-$SLURM_JOB_ID)"
  "$OTELA_BIN" instance link --config-dir "$OTELA_CFG_DIR" 2>&1 || \
    echo "[otela] WARN: instance link failed (key revoked/exhausted? control plane reachable?); continuing -- start retries best-effort"
fi
exec "$OTELA_BIN" start \\
  --config-dir "$OTELA_CFG_DIR" \\
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
# stop_otela comes from meta/common/lib.sh (sourced above); the comment trail:
# 20 s fuse < KillWait (30 s); otela needs ~10 s (AnnounceLeave -> 5 s CRDT
# drain -> srv.Shutdown). A vanished step (SIGKILL) keeps a stale
# connected=true row — harmless, the next run mints a fresh peer id.
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

  # --- capture probe (GLM53_CAPTURE_PROBE=1): fire ONE deterministic ~2k-token
  # request BEFORE the gen probe so comp_capture (GLM53_COMP_MIN_TOKENS=1200)
  # lands on THIS forward identically on beverin and clariden (init profile
  # forwards are machine-dependent: 1-token dummy vs bs=9).
  if [ "${GLM53_CAPTURE_PROBE:-0}" = "1" ]; then
    echo "[$(date -Is)] capture probe (deterministic long-prompt prefill) ..."
    timeout 900 \
      srun --jobid="$SLURM_JOB_ID" --overlap --gres=none --nodes=1 --ntasks=1 \
           -w "$HEAD" --environment="$OTELA_EDF_NAME" \
           python3 "$GLM53_DIAG_DIR/capture_probe.py" \
               "http://127.0.0.1:$SERVE_PORT" "$SERVED_MODEL_NAME" \
      || echo "[$(date -Is)] capture probe FAILED (rc=$?); continuing" >&2
  fi

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
           python3 "$COMMON_DIR/gen_correctness.py" \
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

  # --- stale CRDT hygiene (historical; largely obsolete in v0.2.6) -------
  # The v0.2.6 worker overrides HOME to the per-job $RUNDIR/otela-home, so
  # its BadgerDB dies with the job — no more ocfcore.Qm*.db pile-up in a
  # persistent HOME (the sai-v0.0.6 era accumulated 5 across Aug 28 -> Sep 1
  # in $DEPLOY_DIR/home/.ocfcore). One cleanup remains useful after the
  # version switch, to clear the OLD pile (run inside the container,
  # single-quoted so $HOME resolves there):
  #   srun --jobid=$SLURM_JOB_ID --overlap --gres=none -w "$HEAD" \
  #        --environment="$OTELA_EDF_NAME" \
  #        bash -c 'rm -rf "$HOME/.ocfcore/"ocfcore.Qm*.db 2>/dev/null || true'
  # NB: this clears only THIS worker's local store; it does NOT stop the
  # dial-mismatch loop, whose stale rows live in the bootstrap head's CRDT
  # ($OTELA_RELAY_ADDR, which this recipe does not run).
  echo "[$(date -Is)] starting otela worker on $HEAD (inside container, EDF=$OTELA_EDF_NAME)"
  if [ "$OTELA_LINK_ARMED" = 1 ]; then
    echo "[$(date -Is)] otela account-link: armed (deploy key resolved; label clariden-glm53-$SLURM_JOB_ID)"
  else
    echo "[$(date -Is)] otela account-link: disabled (no OTELA_DEPLOY_KEY / OTD_API_KEY in ~/.zshrc)"
  fi
  # Redirect INSIDE the container (not via srun's pipe) to avoid srun
  # output buffering that hides otela log lines until the buffer fills.
  # The log file is on /capstor (mounted in both host and container).
  srun --jobid="$SLURM_JOB_ID" --overlap --gres=none --nodes=1 --ntasks=1 \
       -w "$HEAD" --environment="$OTELA_EDF_NAME" -J "$OTELA_STEP_NAME" \
       bash -c '"$0" >"$1" 2>&1' "$RUNDIR/otela.sh" "$RUNDIR/otela.log"
  _OTELA_LAUNCHED=1   # stop_otela (lib.sh) is active only after a real launch
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
