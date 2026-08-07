#!/bin/bash
#SBATCH --job-name=dsv4-flash-sglang
#SBATCH --partition=mi300
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Do NOT set --cpus-per-task here: shrinking the cpuset breaks the image's
# SGLANG_SET_CPU_AFFINITY=1 (baked into /etc/environment), whose NUMA pinning
# assumes the full 0..191 logical-CPU cpuset (one MI300A die per TP rank:
#   gpu0->[0-23,96-119] gpu1->[24-47,120-143] gpu2->[48-71,144-167] gpu3->[72-95,168-191]).
# With --exclusive the whole node is allocated; the srun step below requests
# the full SLURM_CPUS_ON_NODE per task so the container sees 0-191.
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=12:00:00
# NOTE: no `B:` prefix here, unlike the JSC recipe. `B:` signals only the batch
# shell; our cleanup() trap lives inside the OTELA_LAUNCHER heredoc and therefore
# runs as a JOB STEP under srun, so the signal must go to the steps. Without
# this, wall-time expiry gives only KillWait (default 30s) between SIGTERM and
# SIGKILL — cleanup() cannot finish, no LEFT is published, and the head keeps a
# `connected: true` row round-robining traffic into a dead endpoint.
#SBATCH --signal=TERM@120
#SBATCH --output=/capstor/scratch/cscs/xyao/deepseek-v4-flash-sglang/logs/%x-%j.out
#
# Serve deepseek-ai/DeepSeek-V4-Flash on beverin (AMD MI300A / gfx942) with the
# plain upstream SGLang ROCm image, and register it on OpenTela. One
# self-contained sbatch — the per-rank SGLang wrapper, the MI300A launcher and
# the otela launcher are all generated here.
#
# Submit (defaults: 1 node, TP=4, served as deepseek-ai/DeepSeek-V4-Flash):
#   sbatch serve_deepseek_v4_flash.sbatch
#
# Scale out (one OpenTela peer per node, each an independent TP=4 replica):
#   sbatch --nodes=4 serve_deepseek_v4_flash.sbatch
#
# Override knobs via env:
#   sbatch --export=ALL,MAX_MODEL_LEN=131072,MEM_FRACTION_STATIC=0.90 \
#          serve_deepseek_v4_flash.sbatch
#
# HOW IT WORKS (why no relay, unlike the JSC recipe):
#   beverin compute nodes have full outbound internet and reach the public
#   OpenTela bootstrap /ip4/140.238.223.116/... directly (the old Alps peer
#   /ip4/148.187.108.178/... is kept as an overrideable fallback). So each rank
#   runs one `otela start --mode node --subprocess <sglang-wrapper>` that
#     1. starts SGLang on 0.0.0.0:${SGLANG_PORT},
#     2. bootstraps to the given peer, and
#     3. registers the `llm` service with label
#        `model=deepseek-ai/DeepSeek-V4-Flash`.
#   The OpenTela head then routes /v1/service/llm/... requests to these peers.
#
# CONTAINER ENGINE: the image comes from ./deepseek-v4-rocm.toml (EDF), found
#   via EDF_PATH below. `srun --environment=deepseek-v4-rocm` runs everything
#   inside it.
set -euo pipefail

# ---------------------------------------------------------------- deployment --
DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/deepseek-v4-flash-sglang}"
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V4-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Flash}"
SGLANG_PORT="${SGLANG_PORT:-8080}"
TP_SIZE="${TP_SIZE:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
# The config advertises max_position_embeddings=1048576 (1 M, YaRN factor 16
# over a 65536 base). Serving the full 1 M by default would size the KV pool
# and the per-request metadata for a context nobody asks for; 65536 is the
# un-scaled native window. Raise it explicitly when you want long context.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"
SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-deepseek-v4}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-deepseekv4}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# --- routed-expert numerics on gfx942 (the DSV4-specific decision) ------------
# DeepSeek-V4-Flash ships its 256 routed experts mxfp4-packed: I8 weight
# tensors (2 fp4 values per byte) + F8_E8M0 block scales, 132 GiB of the
# 149 GiB checkpoint. gfx942 has NO native MX compute — sglang's
# mxfp_supported() is gfx95-only (MI350), so the generic Mxfp4Config path
# would raise "Current platform gfx942 not support mxfp4 computation".
#
# sglang instead routes DSV4 through Fp8Config with is_fp4_experts=True
# (auto-detected from the safetensors header) and offers two ways out:
#
#   SGLANG_USE_AITER=1 (default here) -> aiter's native MXFP4 MoE kernels.
#       Weights stay fp4 (~132 GiB experts, ~37 GiB/GPU at TP=4), leaving the
#       most room for KV cache. NOTE this is the OPPOSITE of the GLM-4.7-Flash
#       beverin recipe, which must set SGLANG_USE_AITER=0 because aiter's
#       get_rope KeyErrors on a config with no original_max_position_embeddings.
#       DeepSeek-V4-Flash's rope_scaling IS a full YaRN block with that key, so
#       the crash does not apply and aiter can be left on.
#
#   DSV4_FP4_DEQUANT=1 -> cast_e2m1fn_to_e4m3fn every expert at load time.
#       Weights DOUBLE to fp8 (~264 GiB experts, ~70 GiB/GPU at TP=4) and the
#       ordinary blockwise-fp8 MoE runs them. Slower to load and far less KV
#       headroom, but it is the fallback if an aiter kernel misbehaves. It
#       asserts moe_runner_backend=auto, so do not combine it with an explicit
#       --moe-runner-backend.
SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
DSV4_FP4_DEQUANT="${DSV4_FP4_DEQUANT:-0}"

# DSA sparse-attention kernel. On HIP the backend is ALWAYS read from this env
# var — hip_flash_mla.flash_mla_with_kvcache_entrypoint() overrides whatever
# --dsa-prefill-backend/--dsa-decode-backend say — and the upstream default
# "tilelang" does NOT compile on gfx942: the first forward dies in TVM's
# InjectSoftwarePipeline pass with
#   ValueError: Check failed: src_info.order < dst_info.order (6 vs. 4) :
#   two statements with buffer access dependency in the same stage of the
#   software pipeline cannot be reordered
# which SIGQUITs every scheduler rank. "triton" is the ROCm alternative
# sglang's own hisparse hook points at; "unified_kv_triton" is the other HIP
# path (incompatible with --enable-hisparse).
FLASHMLA_BACKEND="${FLASHMLA_BACKEND:-triton}"

# Persistent aiter JIT cache. The native MXFP4 MoE kernel
# (module_moe_ck2stages_b16_fp4x2_preshuffle_off_b16_silu_per_1x32_mulWeightStage2)
# is Composable-Kernel C++ that aiter hipcc-compiles on the FIRST request, not
# at load — ~20 hipcc + ~20 clang processes, and the other 3 TP ranks block on
# aiter's baton lock meanwhile. aiter's get_user_jit_dir() defaults to the
# in-image /sgl-workspace/aiter/aiter/jit (writable, so it wins over the $HOME
# fallback), which is a squashfs overlay that dies with the container — so
# every job re-pays the compile.
#
# Pointing AITER_JIT_DIR at /capstor keeps it. It must be SEEDED from the
# image's jit dir, not left empty: the image ships prebuilt .so modules
# (module_aiter_core.so and friends) next to the build tree, and an empty
# AITER_JIT_DIR would force aiter to rebuild those from scratch too. The srun
# step below seeds it on first use.
AITER_JIT_DIR="${AITER_JIT_DIR:-${DEPLOY_DIR}/cache/aiter-jit}"
# Digest of the image in deepseek-v4-rocm.toml. Not derivable from inside the
# container, so it is threaded through for the provenance file that keys the
# shared kernel bucket (see sync_aiter_kernels.sh). Keep in sync with the EDF.
IMAGE_DIGEST="${IMAGE_DIGEST:-sha256:80d04638deb64fac000fa565cb46e5d2f692173dc125a32a956014a6383ecaee}"

# CUDA-graph capture is left ON (unlike the GLM-4.7-Flash beverin recipe, whose
# MTP/nextn layers SIGKILL during capture). Set DISABLE_CUDA_GRAPH=1 to fall
# back to eager decode if capture regresses on a future image.
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-0}"

# ------------------------------------------------------------------- otela ----
# x86-64 build. The worker (not a relay) uses the release directly.
OPENTELA_DIR="${OPENTELA_DIR:-/capstor/scratch/cscs/xyao/opentela}"
OPENTELA_BIN="${OPENTELA_BIN:-${OPENTELA_DIR}/otela}"
OPENTELA_BOOTSTRAP="${OPENTELA_BOOTSTRAP:-/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7}"
OPENTELA_TCP_PORT="${OPENTELA_TCP_PORT:-43905}"
OPENTELA_UDP_PORT="${OPENTELA_UDP_PORT:-59820}"
OPENTELA_SERVICE_NAME="${OPENTELA_SERVICE_NAME:-llm}"

# Per-rank metrics scraping (optional; only starts if the binary exists).
METRICS_REMOTE_WRITE_URL="${METRICS_REMOTE_WRITE_URL:-https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write}"
METRICS_AGENT_BIN="${METRICS_AGENT_BIN:-/capstor/store/cscs/swissai/infra01/ocf-share/vmagent-amd64}"

export EDF_PATH="${DEPLOY_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}"/{logs,home,run} \
  "${DEPLOY_DIR}"/cache/{hf,xdg,config,triton,pip,tmp}

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD_NODE="${NODES[0]}"
NUM_NODES="${#NODES[@]}"
# Full logical-CPU count of one MI300A node (192). The image bakes
# SGLANG_SET_CPU_AFFINITY=1 and pins each TP rank to its GPU's NUMA die using
# host CPU ids 0..191; the srun step MUST hand each task the whole node or the
# pin raises "CPU number N is not eligible" and kills ranks 2/3.
NODE_CPUS="${SLURM_CPUS_ON_NODE:-$(scontrol show node "${HEAD_NODE}" 2>/dev/null | awk -F= '/^CPUTot=/{print $2; exit}')}"
NODE_CPUS="${NODE_CPUS:-192}"

echo "job_id=${SLURM_JOB_ID}"
echo "num_nodes=${NUM_NODES} nodes=${NODES[*]} head_node=${HEAD_NODE}"
echo "node_cpus=${NODE_CPUS}"
echo "model=${MODEL}"
echo "served_model_name=${SERVED_MODEL_NAME}"
echo "tp=${TP_SIZE} max_model_len=${MAX_MODEL_LEN} mem_fraction_static=${MEM_FRACTION_STATIC}"
echo "use_aiter=${SGLANG_USE_AITER} dsv4_fp4_dequant=${DSV4_FP4_DEQUANT}"
echo "disable_cuda_graph=${DISABLE_CUDA_GRAPH} skip_server_warmup=${SKIP_SERVER_WARMUP}"
echo "reasoning_parser=${SGLANG_REASONING_PARSER} tool_call_parser=${SGLANG_TOOL_CALL_PARSER:-<none>}"
echo "port=${SGLANG_PORT}"
echo "opentela_bin=${OPENTELA_BIN} bootstrap=${OPENTELA_BOOTSTRAP}"
echo "started=$(date --iso-8601=seconds)"

[ -d "${MODEL}" ] || {
  echo "FATAL: weights not found: ${MODEL}" >&2
  echo "Stage them with:" >&2
  echo "  huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir ${MODEL}" >&2
  exit 1
}
[ -x "${OPENTELA_BIN}" ] || {
  echo "FATAL: otela binary not found/executable: ${OPENTELA_BIN}" >&2
  echo "Stage it with (login node, needs egress):" >&2
  echo "  mkdir -p ${OPENTELA_DIR} && cd ${OPENTELA_DIR} && \\" >&2
  echo "    curl -fsSL -o otela <otela-x86_64-release-url> && chmod +x otela" >&2
  exit 1
}

# --- sweep stale aiter baton locks -------------------------------------------
# aiter guards each module build with a baton that only the BUILDING process
# removes, and it uses TWO of them at different depths:
#   <jit>/build/lock_module_<name>            (outer, one per module)
#   <jit>/build/module_<name>/build/lock      (inner, the ninja build)
# scancel a job mid-compile and both survive on /capstor. The next job then
# acquires the outer lock, starts the ninja build, and blocks forever on the
# DEAD job's inner lock — 0% CPU, no compiler running, no error, just a hang.
# (Cleaning only the outer one is not enough; that is exactly how job 574156
# stalled.) Anything predating this job's start cannot belong to it, so sweep
# it here — in the sbatch body, before srun, so it runs exactly once and cannot
# race a rank that is legitimately building.
if [ -d "${AITER_JIT_DIR}/build" ]; then
  STALE=0
  while IFS= read -r stale_lock; do
    echo "sweeping stale aiter lock: ${stale_lock}"
    rm -rf -- "${stale_lock}"
    STALE=$((STALE + 1))
  done < <(find "${AITER_JIT_DIR}/build" -maxdepth 3 \
             \( -name 'lock_module_*' -o -path '*/build/lock' \) \
             ! -newermt '-60 seconds' -print 2>/dev/null)
  # 0-byte *.o.tmp are hipcc scratch abandoned by the killed build; ninja will
  # regenerate them, but leaving them confuses the mtime-based upload delta.
  while IFS= read -r stale_tmp; do
    rm -f -- "${stale_tmp}"
    STALE=$((STALE + 1))
  done < <(find "${AITER_JIT_DIR}/build" -name '*.o.tmp' -size 0 \
             ! -newermt '-60 seconds' -print 2>/dev/null)
  [ "${STALE}" -gt 0 ] && echo "aiter_stale_locks_swept=${STALE}"
fi

# ================================================================ engine ======
# MI300A APU memory-accounting fix. PyTorch reports is_integrated=True for
# every MI300A GPU, so sglang's get_available_gpu_memory() substitutes
# psutil.virtual_memory().available (whole-node, 477 GiB measured) for the
# per-GPU mem_get_info() value (128 GiB) on EVERY rank. Each rank then sizes
# its KV pool against memory that does not exist and a rank gets SIGKILLed by
# the cgroup OOM killer (exit -9) shortly after "Load weight end", with no
# Python traceback. The launcher proxies get_device_properties to report
# is_integrated=False. Verified: ranks report "avail mem=119.15 GB" with it.
SGLANG_LAUNCHER="${DEPLOY_DIR}/run/sglang_launcher_mi300a.py"
cat > "${SGLANG_LAUNCHER}" <<'SGLANG_LAUNCHER_PY'
#!/usr/bin/env python3
"""sglang launcher with the MI300A integrated-memory accounting fix."""
import os
import sys

import torch


class _DevicePropsProxy:
    """Transparent proxy reporting is_integrated=False; forwards the rest."""

    __slots__ = ("_orig",)

    def __init__(self, original):
        object.__setattr__(self, "_orig", original)

    def __getattr__(self, name):
        if name == "is_integrated":
            return False
        return getattr(object.__getattribute__(self, "_orig"), name)


_orig_get_device_properties = torch.cuda.get_device_properties


def _patched_get_device_properties(device):
    return _DevicePropsProxy(_orig_get_device_properties(device))


torch.cuda.get_device_properties = _patched_get_device_properties


def main():
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


# The __main__ guard is LOAD-BEARING: sglang starts its per-TP-rank scheduler
# processes with multiprocessing "spawn", which re-imports this file in every
# child. Without it, run_server() re-executes at import time and every child
# dies with "An attempt has been made to start a new process before the current
# process has finished its bootstrapping phase"; the parent only reports
# "Rank N scheduler died during initialization (exit code: 1)".
if __name__ == "__main__":
    main()
SGLANG_LAUNCHER_PY

# Shared SGLang subprocess wrapper used by every rank. Each rank runs on its own
# node, so one wrapper is safe to share. Single-node TP=4: no cross-node init,
# no torchrun, no --nnodes/--dist-init-addr.
SGLANG_SUBPROCESS="${DEPLOY_DIR}/run/sglang_subprocess-${SLURM_JOB_ID}.sh"
cat > "${SGLANG_SUBPROCESS}" <<SGLANG_WRAPPER
#!/bin/bash
set -euo pipefail

echo "sglang_subprocess_started rank=\${SLURM_PROCID} host=\$(hostname) \$(date --iso-8601=seconds)"

# tempfile on /capstor, NOT the Slurm default /users/<u>/.tmp (home is
# quota-limited; sglang's tempfile cleanup on a quota dir raises
# OSError [Errno 39] Directory not empty). Belt-and-suspenders alongside the
# EDF [env] TMPDIR so this holds even if a parent shell re-exported it.
export TMPDIR="${DEPLOY_DIR}/cache/tmp"
mkdir -p "\${TMPDIR}"

# Routed-expert numerics + DSA kernel — see the blocks at the top of this sbatch.
export SGLANG_USE_AITER="${SGLANG_USE_AITER}"
export SGLANG_DSV4_FP4_DEQUANT="${DSV4_FP4_DEQUANT}"
export SGLANG_HACK_FLASHMLA_BACKEND="${FLASHMLA_BACKEND}"
export PROV_IMAGE_DIGEST="${IMAGE_DIGEST}"

# Seed the persistent aiter JIT dir from the image on first use (see the
# AITER_JIT_DIR block at the top), then point aiter at it so the CK MXFP4 MoE
# kernel is compiled once per image, not once per job.
AITER_IMAGE_JIT_DIR=/sgl-workspace/aiter/aiter/jit
if [[ -d "\${AITER_IMAGE_JIT_DIR}" ]]; then
  mkdir -p "\$(dirname "${AITER_JIT_DIR}")"
  if [[ ! -e "${AITER_JIT_DIR}/.seeded" ]]; then
    # mkdir is atomic on Lustre, so exactly one rank across all nodes seeds and
    # the rest wait. Without this the ranks cp -r into the same tree
    # concurrently and a loser can import a half-copied .so.
    if mkdir "${AITER_JIT_DIR}.lock" 2>/dev/null; then
      echo "aiter_jit_seed_start dst=${AITER_JIT_DIR} rank=\${SLURM_PROCID} \$(date --iso-8601=seconds)"
      mkdir -p "${AITER_JIT_DIR}"
      # -n: never clobber a newer cached build with the image's stale copy.
      cp -rn "\${AITER_IMAGE_JIT_DIR}/." "${AITER_JIT_DIR}/" 2>/dev/null || true
      # Record exactly what came from the image. sync_aiter_kernels.sh subtracts
      # this list to find the locally-compiled delta, which is far more robust
      # than comparing mtimes.
      ( cd "${AITER_JIT_DIR}" && find . -type f ! -name '.seeded*' -print ) \\
        > "${AITER_JIT_DIR}/.seeded.manifest" 2>/dev/null || true
      touch "${AITER_JIT_DIR}/.seeded"
      rmdir "${AITER_JIT_DIR}.lock" 2>/dev/null || true
      echo "aiter_jit_seed_done \$(date --iso-8601=seconds)"
    else
      echo "aiter_jit_seed_wait rank=\${SLURM_PROCID} \$(date --iso-8601=seconds)"
      for _ in \$(seq 1 300); do
        [[ -e "${AITER_JIT_DIR}/.seeded" ]] && break
        sleep 2
      done
    fi
  fi
  if [[ -e "${AITER_JIT_DIR}/.seeded" ]]; then
    export AITER_JIT_DIR="${AITER_JIT_DIR}"
    echo "aiter_jit_dir=\${AITER_JIT_DIR}"
  else
    echo "WARN aiter jit seed never completed; falling back to the in-image (ephemeral) jit dir" >&2
  fi

  # Record what this cache was built against, so sync_aiter_kernels.sh can key
  # the shared bucket on (arch, aiter commit, image) without re-entering the
  # container. Kernels are arch- and revision-specific binaries; restoring a
  # gfx942 build onto gfx950 would hand the engine wrong code.
  python3 - >"${AITER_JIT_DIR}/../aiter-provenance.env" 2>/dev/null <<'PROV_PY' || true
import os, torch
def q(v):
    return str(v).replace("'", "")
try:
    arch = torch.cuda.get_device_properties(0).gcnArchName
except Exception:
    arch = os.environ.get("PROV_ARCH", "")
try:
    import sglang; sgl = sglang.__version__
except Exception:
    sgl = "unknown"
print(f"PROV_ARCH='{q(arch)}'")
print(f"PROV_AITER_COMMIT='{q(os.environ.get('AITER_COMMIT', ''))}'")
print(f"PROV_IMAGE_DIGEST='{q(os.environ.get('PROV_IMAGE_DIGEST', ''))}'")
print(f"PROV_ROCM_VERSION='{q(torch.version.hip)}'")
print(f"PROV_TORCH_VERSION='{q(torch.__version__)}'")
print(f"PROV_SGLANG_VERSION='{q(sgl)}'")
PROV_PY
fi

SGLANG_ARGS=(
  --model-path "${MODEL}"
  --host 0.0.0.0
  --port "${SGLANG_PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP_SIZE}"
  --context-length "${MAX_MODEL_LEN}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --enable-metrics
)

if [[ -n "${SGLANG_TOOL_CALL_PARSER}" ]]; then
  SGLANG_ARGS+=(--tool-call-parser "${SGLANG_TOOL_CALL_PARSER}")
fi
if [[ -n "${SGLANG_REASONING_PARSER}" ]]; then
  SGLANG_ARGS+=(--reasoning-parser "${SGLANG_REASONING_PARSER}")
fi
if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
  SGLANG_ARGS+=(--disable-cuda-graph)
fi
if [[ "${SKIP_SERVER_WARMUP}" == "1" ]]; then
  SGLANG_ARGS+=(--skip-server-warmup)
fi
if [[ "${LOAD_FORMAT}" != "auto" ]]; then
  SGLANG_ARGS+=(--load-format "${LOAD_FORMAT}")
fi

# NOTE: no transformers upgrade step here (the GLM-4.7-Flash beverin recipe
# needs one). The v0.5.16 ROCm image already ships transformers 5.12.1, and
# sglang carries its own DeepSeekV4Config, so AutoConfig resolves
# model_type=deepseek_v4 out of the box — verified by probe_sglang.sbatch.

exec python "${SGLANG_LAUNCHER}" "\${SGLANG_ARGS[@]}"
SGLANG_WRAPPER
chmod +x "${SGLANG_SUBPROCESS}"

# ============================================================ otela worker ====
SCRAPE_CONFIG_TEMPLATE="${DEPLOY_DIR}/run/vmagent-scrape-${SLURM_JOB_ID}.yaml.tpl"
cat > "${SCRAPE_CONFIG_TEMPLATE}" <<SCRAPE
global:
  scrape_interval: 30s
scrape_configs:
  - job_name: slurm-llm
    metrics_path: /metrics
    static_configs:
      - targets: ['LOCALHOST_IP:${SGLANG_PORT}']
SCRAPE

OTELA_LAUNCHER="${DEPLOY_DIR}/run/opentela_launcher-${SLURM_JOB_ID}.sh"
cat > "${OTELA_LAUNCHER}" <<OTELA_LAUNCHER
#!/bin/bash
set -euo pipefail

RANK="\${SLURM_PROCID}"
NODE="\$(hostname)"
LOCALHOST_IP="\$(hostname -i | awk '{print \$1}')"
# Per-rank libp2p ports. Every peer used to bind the same fixed tcp/udp port,
# so multiple peers behind one NAT egress collided on a single external port
# mapping and only ~half registered. Offsetting by SLURM_PROCID gives each peer
# a distinct port.
TCP_PORT=\$(( ${OPENTELA_TCP_PORT} + \${RANK} ))
UDP_PORT=\$(( ${OPENTELA_UDP_PORT} + \${RANK} ))
OPENTELA_HOME="${DEPLOY_DIR}/home/opentela-${SLURM_JOB_ID}-\${RANK}"
OPENTELA_CONFIG="\${OPENTELA_HOME}/.config/otela/cfg.yaml"
OPENTELA_LOG="${DEPLOY_DIR}/logs/opentela-${SLURM_JOB_ID}-\${RANK}.log"
VMAGENT_LOG="${DEPLOY_DIR}/logs/vmagent-${SLURM_JOB_ID}-\${RANK}.log"
SCRAPE_CONFIG="${DEPLOY_DIR}/run/vmagent-scrape-${SLURM_JOB_ID}-\${RANK}.yaml"
mkdir -p "\${OPENTELA_HOME}/.config/otela"

export HOME="\${OPENTELA_HOME}"

sed "s/LOCALHOST_IP/\${LOCALHOST_IP}/g" "${SCRAPE_CONFIG_TEMPLATE}" > "\${SCRAPE_CONFIG}"

"${OPENTELA_BIN}" init --config "\${OPENTELA_CONFIG}" >/dev/null 2>&1 || true

echo "opentela_launcher_started rank=\${RANK} node=\${NODE} ip=\${LOCALHOST_IP} tcp=\${TCP_PORT} udp=\${UDP_PORT} \$(date --iso-8601=seconds)"

# Stagger peer joins so all N ranks do not hit the single bootstrap at once
# (the thundering herd was correlated with failed registrations).
sleep \$(( \${RANK} * 10 ))

# otela is backgrounded with stdin from /dev/zero because it exits when run in
# the foreground with EOF on stdin. It starts SGLang as its subprocess.
nohup "${OPENTELA_BIN}" start \\
  --config "\${OPENTELA_CONFIG}" \\
  --bootstrap.addr "${OPENTELA_BOOTSTRAP}" \\
  --tcpport "\${TCP_PORT}" \\
  --udpport "\${UDP_PORT}" \\
  --mode node \\
  --solana.skip_verification \\
  --service.name "${OPENTELA_SERVICE_NAME}" \\
  --service.port "${SGLANG_PORT}" \\
  --subprocess "${SGLANG_SUBPROCESS}" \\
  --label "model=${SERVED_MODEL_NAME}" \\
  --label "framework=sglang" \\
  --label "slurm_job_id=${SLURM_JOB_ID}" \\
  --label "node=\${NODE}" \\
  < /dev/zero \\
  > "\${OPENTELA_LOG}" 2>&1 &
OPENTELA_PID=\$!
echo "opentela_started rank=\${RANK} pid=\${OPENTELA_PID} log=\${OPENTELA_LOG}"

# Stop the peer with SIGTERM (never SIGKILL) so it leaves the mesh gracefully:
# a killed peer stays connected:true in the registry and the head keeps
# round-robining traffic into a dead endpoint.
cleanup() {
  echo "otela_stopping rank=\${RANK} \$(date --iso-8601=seconds)"
  kill -TERM "\${OPENTELA_PID}" 2>/dev/null || true
  # 20s, not 30s. --signal=TERM@120 covers wall-time expiry, but on the \`scancel\`
  # path the only grace is KillWait (default 30s). otela needs ~10s to shut down
  # (AnnounceLeave -> 5s CRDT drain -> srv.Shutdown), so 20s leaves margin and
  # still completes before SIGKILL. Waiting the full 30s just guarantees we are
  # still waiting when the process is destroyed.
  for _ in \$(seq 1 20); do
    kill -0 "\${OPENTELA_PID}" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "\${OPENTELA_PID}" 2>/dev/null; then
    echo "WARN otela did not exit within 20s of SIGTERM rank=\${RANK}; no LEFT published" >&2
  fi
  [[ -n "\${VMAGENT_PID:-}" ]] && kill "\${VMAGENT_PID}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

# Wait for SGLang before starting vmagent: vmagent's Go HTTP client can wedge
# if it scrapes before the target listens. /get_model_info returns 200 only
# once weights are loaded.
echo "waiting_for_sglang rank=\${RANK} url=http://\${LOCALHOST_IP}:${SGLANG_PORT}/get_model_info \$(date --iso-8601=seconds)"
while true; do
  if ! kill -0 "\${OPENTELA_PID}" 2>/dev/null; then
    echo "otela_exited_before_ready rank=\${RANK}" >&2
    break
  fi
  if curl -s "http://\${LOCALHOST_IP}:${SGLANG_PORT}/get_model_info" >/dev/null 2>&1; then
    echo "sglang_ready rank=\${RANK} \$(date --iso-8601=seconds)"
    break
  fi
  sleep 2
done

VMAGENT_PID=""
if [[ -x "${METRICS_AGENT_BIN}" ]]; then
  nohup "${METRICS_AGENT_BIN}" \\
    -promscrape.config="\${SCRAPE_CONFIG}" \\
    -remoteWrite.url="${METRICS_REMOTE_WRITE_URL}" \\
    -remoteWrite.label="slurm_job_id=${SLURM_JOB_ID}" \\
    -remoteWrite.label="model=${SERVED_MODEL_NAME}" \\
    -remoteWrite.label="framework=sglang" \\
    -remoteWrite.label="user=\${SLURM_JOB_USER}" \\
    -remoteWrite.label="node=\${NODE}" \\
    "-remoteWrite.tmpDataPath=${DEPLOY_DIR}/cache/vmagent-data-${SLURM_JOB_ID}-\${RANK}" \\
    > "\${VMAGENT_LOG}" 2>&1 &
  VMAGENT_PID=\$!
  echo "vmagent_started rank=\${RANK} pid=\${VMAGENT_PID} log=\${VMAGENT_LOG}"
else
  echo "vmagent_binary_not_found at ${METRICS_AGENT_BIN}"
fi

wait "\${OPENTELA_PID}"
OPENTELA_CODE=\$?
echo "opentela_exited rank=\${RANK} code=\${OPENTELA_CODE} \$(date --iso-8601=seconds)"
exit "\${OPENTELA_CODE}"
OTELA_LAUNCHER
chmod +x "${OTELA_LAUNCHER}"

# Record endpoints before srun starts so the metadata is readable while the
# services are still coming up.
{
  echo "JOB_ID=${SLURM_JOB_ID}"
  echo "JOBID=${SLURM_JOB_ID}"
  echo "NUM_NODES=${NUM_NODES}"
  echo "PORT=${SGLANG_PORT}"
  echo "MODEL=${SERVED_MODEL_NAME}"
  echo "FRAMEWORK=sglang"
  echo "HEAD_NODE=${HEAD_NODE}"
  echo "BOOTSTRAP=${OPENTELA_BOOTSTRAP}"
  echo "SUBMITTED_AT=$(date --iso-8601=seconds)"
  echo "LOG=${DEPLOY_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
  for i in "${!NODES[@]}"; do
    echo "ENDPOINT_${i}=${NODES[$i]}:${SGLANG_PORT}"
  done
} > "${DEPLOY_DIR}/last_service.env"

# One OpenTela+SGLang service per node inside the SGLang ROCm container.
# --cpus-per-task=NODE_CPUS gives each task the whole node (0..191) so the
# image's SGLANG_SET_CPU_AFFINITY NUMA pinning stays valid.
srun \
  --nodes="${NUM_NODES}" \
  --ntasks="${NUM_NODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${NODE_CPUS}" \
  --gpus-per-task=4 \
  --environment=deepseek-v4-rocm \
  bash -lc "${OTELA_LAUNCHER}"
SRUN_EXIT=$?
echo "srun_exited code=${SRUN_EXIT} $(date --iso-8601=seconds)"

exit "${SRUN_EXIT}"
