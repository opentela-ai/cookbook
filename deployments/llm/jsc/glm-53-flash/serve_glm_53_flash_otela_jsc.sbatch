#!/bin/bash
#SBATCH --job-name=glm53-otela-jsc
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
# Whole node (288 logical CPUs): the image pins TP ranks to GPU NUMA domains
# via host CPU ids; a shrunken cpuset kills ranks (clariden finding).
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
# B:TERM@120 gives the stop_otela trap a TERM window before KillWait —
# without B: no LEFT is published and the head round-robins into a dead
# endpoint (kimi-k3 JSC).
#SBATCH --signal=B:TERM@120
#SBATCH --output=/e/scratch/reformo/yao4/glm-53-flash/logs/serve-%j.out
#SBATCH --error=/e/scratch/reformo/yao4/glm-53-flash/logs/serve-%j.out
#
# =============================================================================
# Serve zai-org/GLM-5.3-Flash (~306 GiB FP8 e4m3, DSA + linear + KDA hybrid)
# on JSC Jupiter Booster (4x GH200, aarch64, InfiniBand) + OpenTela.
# Ported from the validated clariden recipe (same GPU/image): model facts
# (tilelang DSA pins, sitecustomize FA3 patches, watchdogs) carry over;
# site facts (apptainer, NCCL/IB, login-node relay, /e/scratch) re-derived
# from the kimi-k3 JSC recipe. Submit variants, relay topology, memory math
# and every knob essay: README.md.
# Flow: preflight (+30 s container import preflight) -> one sglang rank/node
# in apptainer (TP4 x PP<NNODES>, head serves HTTP) -> /health -> MANDATORY
# generation probe -> servekit bench -> ONE otela worker via the relay ->
# SIGTERM/LEFT on end.
# =============================================================================

set -uo pipefail

# ----------------------------------------------------- script dir resolution --
# gen_correctness.py lives in the repo's meta/common/ (shared; per-recipe
# copies removed). BASH_SOURCE is the spool copy — fallback only; the real
# search is submit dir -> its meta/common -> recipe dir -> BASH_SOURCE dir.
SCRIPT_DIR=""
for _cand in \
    "${SLURM_SUBMIT_DIR:-}" \
    "${SLURM_SUBMIT_DIR:-}/meta/common" \
    "${SLURM_SUBMIT_DIR:-}/deployments/llm/jsc/glm-53-flash" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
  [ -n "$_cand" ] && [ -f "$_cand/gen_correctness.py" ] && { SCRIPT_DIR="$_cand"; break; }
done
if [ -z "$SCRIPT_DIR" ]; then
  echo "FATAL: cannot locate gen_correctness.py (searched SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}, its meta/common/, and the BASH_SOURCE dir)." >&2
  echo "       Submit from the repo root or the recipe dir," >&2
  echo "       or cp meta/common/gen_correctness.py next to the spool copy." >&2
  exit 1
fi
unset _cand

# --------------------------------------------------------------------- lib ---
# Shared helpers from the repo's meta/common/ — lib.sh (stop_otela) and
# gen_correctness.py live there. SCRIPT_DIR may already BE the meta/common
# dir (submit-dir candidate); otherwise walk up from it.
COMMON_DIR=""
[ -f "$SCRIPT_DIR/lib.sh" ] && [ -f "$SCRIPT_DIR/gen_correctness.py" ] && COMMON_DIR="$SCRIPT_DIR"
if [ -z "$COMMON_DIR" ]; then
  _d="$SCRIPT_DIR"
  for _ in 1 2 3 4 5 6; do
    [ -f "$_d/meta/common/lib.sh" ] && { COMMON_DIR="$_d/meta/common"; break; }
    _d="$(dirname "$_d")"
  done
fi
[ -n "$COMMON_DIR" ] || { echo "FATAL: meta/common/lib.sh not found from $SCRIPT_DIR — submit from within the repo checkout" >&2; exit 1; }
source "$COMMON_DIR/lib.sh"   # stop_otela + teardown contract
unset _d

# ---------------------------------------------------------------- deployment --
# Reuses the /e/scratch tree (image, weights, caches — no re-pull/re-JIT).
# /e/scratch and /e/data1 are the compute-mounted filesystems; /p/scratch is
# NOT (README "One-time staging").
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
# Hybrid: 34/45 linear/KDA + 11/45 DSA. kpool>1 (multimodal) is only legal
# with fa3/tilelang/trtllm; this aarch64 image lacks fa3+trtllm -> tilelang
# pins. The DSA full-MHA prefill hard-codes FA3 on SM90 — sitecustomize.py
# swaps in a PyTorch SDPA ragged loop (README "Carried findings").
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
# 3600 hard / 600 soft / 3600 NCCL floor (the 600 s NCCL default fired 5.5 h
# before the scheduler watchdog on a transient PP SEND stall).
export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-3600}"
export SOFT_WATCHDOG_TIMEOUT="${SOFT_WATCHDOG_TIMEOUT:-600}"
export DIST_TIMEOUT="${DIST_TIMEOUT:-3600}"

export DIST_PORT="${DIST_PORT:-20000}"
export SERVE_PORT="${SERVE_PORT:-30000}"
export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

# --- /health must NOT do a generation round-trip ----------------------------
# This fork defaults health-generation TRUE: a detokenizer-heartbeat stall
# 503s a healthy engine (clariden job 3419248). =0 returns 200 immediately.
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION:-0}"
export SGLANG_HEALTH_CHECK_TIMEOUT="${SGLANG_HEALTH_CHECK_TIMEOUT:-300}"

# --- MTP / nextn prediction: OFF (clariden default; plain serve first) -------
export GLM53_SPECULATIVE="${GLM53_SPECULATIVE:-0}"

# --- escape hatches (not needed on GH200, kept as knobs): crash at capture
# or warmup -> DISABLE_CUDA_GRAPH=1 SKIP_SERVER_WARMUP=1 and re-test.
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
export SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-0}"

# sitecustomize.py overrides sglang's 480 s post-load barrier with this
# (306 GiB off /e/scratch can exceed 8 min on the slowest rank — fix 10).
export UNBALANCED_MODEL_LOADING_TIMEOUT_S="${UNBALANCED_MODEL_LOADING_TIMEOUT_S:-3600}"

# --- Load format ------------------------------------------------------------
# LOAD_FORMAT=dummy: random weights, fast forward-path smoke (pair with
# GEN_CORRECTNESS_SMOKE=1 + a lower HEALTH_TIMEOUT).
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# --- metrics: OFF by default (NCCL watchdog DuplicateTimeseries crash,
# kimi-k3 JSC; nothing here reads /metrics). Opt in only knowingly.
export ENABLE_METRICS="${ENABLE_METRICS:-}"

# ------------------------------------------------------------------- otela ----
# Official OpenTela release (arm64), staged on the login node by up.sh via
# meta/common/download_otela.sh (latest release; OTELA_VERSION pins a tag).
# Compute has no egress — preflight prints the staging command.
OTELA_DIR="${OTELA_DIR:-/e/scratch/${PROJECT:-reformo}/${USER}/otela-llm}"
OTELA_VERSION="${OTELA_VERSION:-v0.2.4}"
OTELA_BIN="${OTELA_BIN:-$OTELA_DIR/opentela-arm64-latest}"
OTELA_URL="${OTELA_URL:-https://github.com/eth-easl/OpenTela/releases/download/$OTELA_VERSION/opentela-arm64}"
export OTELA_SERVICE_NAME="${OTELA_SERVICE_NAME:-llm}"
OTELA_TCP_PORT="${OTELA_TCP_PORT:-43920}"
OTELA_UDP_PORT="${OTELA_UDP_PORT:-59840}"
OTELA_API_PORT="${OTELA_API_PORT:-18094}"
OTELA_STEP_NAME="${OTELA_STEP_NAME:-otela-worker-$SLURM_JOB_ID}"
# Peers discard CRDT records from unsigned binaries; keep false so older
# unsigned peers are still accepted (kimi-k3 JSC, relay-verified).
OTELA_REQUIRE_SIGNED="${OTELA_REQUIRE_SIGNED:-false}"
# Fresh peer id per job ($$ seed): a constant seed cannot re-register until
# its LEFT tombstone expires (euler/dgx-spark stall). Pin only knowingly.
OTELA_SEED="${OTELA_SEED:-$$}"
# NEVER auto-register dummy weights (garbage to the mesh). auto/1/0.
OTELA_REGISTER="${OTELA_REGISTER:-auto}"
# The shared login-node relay (us-managed; see header). Precedence: explicit
# $OTELA_RELAY_ADDR > non-empty $RELAY_ADDR_FILE > the baked-in default.
RELAY_ADDR_FILE="${RELAY_ADDR_FILE:-/e/scratch/reformo/yao4/otela-relay/relay.multiaddr}"
OTELA_RELAY_DEFAULT="/ip4/10.128.1.1/tcp/45195/p2p/QmakkeU7ZYNRdA2XhYThT3r8QHd2qxbekTNCPVpEvSqzgd"

# 3 h budget for the 306 GiB load (gates the pre-registration wait only; the
# barrier patch is what prevents slow-rank kills).
export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-10800}"

# ----------------------------------------------------- mandatory gen probe ---
# /health 200 does NOT prove generation (deepseek-v4 502'd, glm-5.2 hung).
# gen_correctness.py (meta/common) sends six greedy prompts + checks factual
# substrings; SMOKE=1 gates on non-empty continuation. GEN_PROBE=0: skip.
export GEN_PROBE="${GEN_PROBE:-1}"
export GEN_PROBE_TIMEOUT="${GEN_PROBE_TIMEOUT:-1200}"
export GEN_CORRECTNESS_MAX_TOKENS="${GEN_CORRECTNESS_MAX_TOKENS:-64}"
export GEN_CORRECTNESS_MIN_PASS="${GEN_CORRECTNESS_MIN_PASS:-5}"
export GEN_CORRECTNESS_PER_REQ_TIMEOUT="${GEN_CORRECTNESS_PER_REQ_TIMEOUT:-300}"
export GEN_CORRECTNESS_SMOKE="${GEN_CORRECTNESS_SMOKE:-0}"

# ---- servekit (cold-start profile + pre-registration bench) ----
# Pure stdlib, module-style from a checkout on /e/scratch (no install, no
# egress). Bench merges into the node-0 report; missing checkout = WARN.
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
# sitecustomize.py — auto-imported at Python startup (PATCH_DIR first on
# PYTHONPATH via engine.sh), BEFORE sglang runs. Three patches carried
# verbatim from the validated clariden recipe (same image / sm_90 GH200);
# full histories: README "Carried findings" + the clariden recipe.
#   1. UNBALANCED_MODEL_LOADING_TIMEOUT_S 480 -> 3600 (slow-rank load barrier).
#   2. kpool-tail FA3-safe (keep flashmla_sparse when fa3 absent; clariden
#      job 3208291). INERT while the tilelang pins are in effect (defensive).
#   3. _forward_standard_mha SDPA fallback (FA3 hard-coded on SM90, job
#      3209880; both built-in paths dead on this image — FA3 absent,
#      trtllm Blackwell-only, job 3210211). Fires only when fa3 is absent.
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

    # sgl_kernel.flash_ops is what FA3 needs; MISSING on aarch64. Probe
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
        # Apply the override ONLY when the target is importable on this
        # arch; on aarch64 keep flashmla_sparse (no FA3 dep) instead of
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
# No CLI override: the SM100+ branch is a Blackwell-only TRT-LLM kernel
# (job 3210211 "Unsupported architecture" on sm_90), and FA2/FA3/flashinfer
# are ALL absent on this aarch64 image — PyTorch SDPA is the only sm_90-
# runnable full-MHA kernel. DSA layers here are pure MHA (64/64 heads, no
# GQA), so a per-request SDPA ragged loop is a correct drop-in (slower —
# correctness first). Fires ONLY on not-_fa3_ok SM90; behavior-preserving
# on x86 Hopper (FA3) and SM100+ (trtllm).
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
  echo "  On a JSC LOGIN node run (compute has no egress) — bash up.sh stages the latest" >&2
  echo "  official OpenTela automatically via meta/common/download_otela.sh, or manually:" >&2
  echo "    OTELA_DIR='$OTELA_DIR' OTELA_ARCH=arm64 bash ../../../../meta/common/download_otela.sh" >&2
  echo "    curl -L --fail -o '$OTELA_BIN' '$OTELA_URL' && chmod +x '$OTELA_BIN'   # pinned fallback" >&2
fi
if [ "$GEN_PROBE" = "1" ] && [ ! -f "$SCRIPT_DIR/gen_correctness.py" ]; then
  ok=0; echo "FATAL: gen_correctness.py not found next to this sbatch ($SCRIPT_DIR)" >&2
fi
if [ "$ok" != "1" ]; then echo "preflight FAILED"; exit 1; fi
echo "preflight: image / weights / otela / gen_correctness.py OK"

# In-container preflight (~30 s, --nv): glm5_next class import + the
# flash_ops probe the patches/pins key off (documented absent on aarch64).
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
# DNS-resolve the head hostname (job 2215062: scontrol's NodeAddr line
# splits into "IP NodeHostName" and c10d hangs on "gai error" retries).
HEAD_IP="$(getent hosts "$HEAD" | awk '{print $1}' | head -1)"
[ -z "$HEAD_IP" ] && HEAD_IP="$(scontrol show node "$HEAD" | grep -oE 'NodeAddr=[^ ]+' | cut -d= -f2 | head -1)"
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
# One engine rank per node: TP4 intra-node (NVLink) + PP across nodes; the
# servekit profile wrapper (when present) sits INSIDE the container and
# reports cold-start phases to $RUNDIR/coldstart*.
# NCCL over JSC InfiniBand (site-verified): force GDR WRITE (RETRY_CNT 7,
# TIMEOUT 22 ~= 12 min) + pinned IPv4 ib0 (ib0 peers not in DNS — up to
# 2 min discovery delay otherwise). No SHARP: PP point-to-point only, and
# the cu13 image links libsharp against libcudart.so.12 (kimi-k3 JSC).
# Engine binds 0.0.0.0 in the HOST netns (apptainer shares it) — what makes
# the otela registrar's localhost health check genuinely local.
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

# --- caches + HOME inside the container (kimi-k3 JSC pattern) ----------------
# HOME on /e/scratch (NFS /e/home: quota + ESTALE); caches shared across
# re-submits; HF offline (no egress — a hub read hangs to the timeout).
export HOME="${HOME:-/e/scratch/reformo/yao4/glm-53-flash/home}"
export HF_HOME="${HF_HOME:-/e/scratch/reformo/yao4/glm-53-flash/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/e/scratch/reformo/yao4/glm-53-flash/cache/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/e/scratch/reformo/yao4/glm-53-flash/cache/triton}"
# DeepGEMM JIT cache (~10-20 min first time): apptainer remaps HOME to the
# passwd home regardless of the exported HOME (job 2215846) — pin it.
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-/e/scratch/reformo/yao4/glm-53-flash/cache/sglang}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
mkdir -p "$HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$SGLANG_CACHE_DIR"

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
# `python -m sglang.launch_server`: the validated entry point; config.json
# supplies the architecture. General attention knobs are deliberately NOT
# passed (they'd break the 34 linear layers); only DSA-SPECIFIC backends +
# the multimodal backend are pinned.
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

# metrics OFF by default (NCCL watchdog DuplicateTimeseries crash).
if [ -n "${ENABLE_METRICS:-}" ]; then
  ENGINE_CMD+=(--enable-metrics)
fi

# GLM-5.3 parsers (glm45/glm47; this build takes just the two parser flags).
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

# MTP/EAGLE speculative decoding: OFF (draft path unvalidated on sm_90).
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

# ulimit -c 0 + unset BASH_ENV: no giant cores; no per-subshell init.
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
# PER-JOB config-dir (cfg + libp2p identity; v0.2.4 --config-dir). BadgerDB
# home resolves via /etc/passwd, not $HOME (README "Why a relay"). loglevel
# MUST stay debug: the registration markers grep'd in Verify fire at
# DEBUG/INFO — otherwise otela looks hung during CRDT sync.
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
# --label (v0.2.4, verified) makes the peer identifiable: otela peers show
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
# +x: the worker srun exec's the path directly — exit 126 without it.
chmod +x "$RUNDIR/otela.sh"
echo "otela worker: $OTELA_BIN ($OTELA_VERSION) via relay $RELAY_ADDR"

# ================================================= registration + lifecycle ==
# stop_otela comes from meta/common/lib.sh (sourced above); _OTELA_LAUNCHED is
# set below right after the worker step starts.
trap 'stop_otela' EXIT TERM INT

# Registration gate: /health -> MANDATORY gen probe -> servekit bench ->
# otela worker. Runs on the head alongside sglang.
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
    # Inside the container on the head (host python3 is 3.9; netns shared).
    # Posargs: <base_url> <model> [out_json] [max_tokens] [min_pass].
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
      echo "[$(date -Is)]       Inspect $PROBE_OUT + the engine log (engine left running — burns GPU-hours)." >&2
      echo "[$(date -Is)]       Likely the DSA + linear + KDA hybrid forward faults on JSC GH200." >&2
      exit 1
    fi
  fi

  # Bench BEFORE registration: mesh traffic must not pollute the numbers.
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
      scancel "$SLURM_JOB_ID"
    else
      echo "[$(date -Is)] engine stays local until walltime/scancel"
    fi
    exit 0
  fi

  echo "[$(date -Is)] starting otela worker (service $OTELA_SERVICE_NAME :$SERVE_PORT, step $OTELA_STEP_NAME)"
  # Foreground srun inside this subshell: keeps the step visible for
  # stop_otela's scancel; redirect INSIDE (srun-side redirects buffer).
  srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
       -J "$OTELA_STEP_NAME" \
       bash -c '"$0" >"$1" 2>&1' "$RUNDIR/otela.sh" "$RUNDIR/otela.log"
  rc=$?
  _OTELA_LAUNCHED=1   # stop_otela (lib.sh) is active only after a real launch
  echo "[$(date -Is)] otela worker exited rc=$rc"
) &
REG_PID=$!
echo "registration subshell pid $REG_PID; run dir $RUNDIR"

# ======================================================== the engine itself ==
# Foreground srun: when the engine dies the trap publishes a clean LEFT and
# the script exits with the engine's rc.
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
