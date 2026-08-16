#!/bin/bash
#SBATCH --job-name=glm-47-flash-sglang
#SBATCH --partition=mi300
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Do NOT set --cpus-per-task here: shrinking the cpuset breaks the image's
# SGLANG_SET_CPU_AFFINITY=1 (set in /etc/environment), whose NUMA pinning
# assumes the full 0..191 logical-CPU cpuset (one MI300A die per TP rank:
#   gpu0->[0-23,96-119] gpu1->[24-47,120-143] gpu2->[48-71,144-167] gpu3->[72-95,168-191]).
# With --exclusive the whole node is allocated; the srun step below requests
# the full SLURM_CPUS_ON_NODE per task so the container sees 0-191.
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=12:00:00
# NOTE: no `B:` prefix. `B:` signals only the batch shell; our cleanup() trap
# lives inside the OTELA_LAUNCHER heredoc and therefore runs as a JOB STEP under
# srun, so the signal must reach the steps. Without this, wall-time expiry gives
# only KillWait (default 30s) between SIGTERM and SIGKILL — otela cannot finish
# its ~10s graceful leave, no LEFT is published, and the head keeps a
# `connected: true` row round-robining traffic into a dead endpoint.
#SBATCH --signal=TERM@120
#SBATCH --output=/capstor/scratch/cscs/xyao/glm47-flash-sglang-beverin/logs/%x-%j.out
#
# Serve zai-org/GLM-4.7-Flash on Beverin (AMD MI300A / gfx942) with the plain
# upstream SGLang ROCm image, and register it on OpenTela. One self-contained
# sbatch — no sibling scripts; the per-rank wrapper + otela launcher are
# generated here.
#
# Submit (defaults: 1 node, TP=4, served as zai-org/GLM-4.7-Flash):
#   sbatch serve_glm_47_flash_sglang.sbatch
#
# Scale out (one OpenTela peer per node, each a TP=4 replica):
#   sbatch --nodes=4 serve_glm_47_flash_sglang.sbatch
#
# Override knobs via env:
#   SBATCH --export=ALL,MAX_MODEL_LEN=65536,MEM_FRACTION_STATIC=0.92 \
#          serve_glm_47_flash_sglang.sbatch
#
# HOW IT WORKS (why no relay, unlike the JSC recipe):
#   Beverin compute nodes have full outbound internet and can reach the
#   public OpenTela bootstrap /ip4/140.238.223.116/... directly (verified
#   2026-08-07; the previous Alps peer /ip4/148.187.108.178/... is kept as
#   an overrideable fallback). So each rank runs
#   one `otela start --mode node --subprocess <sglang-wrapper>` that
#     1. starts SGLang on 0.0.0.0:${SGLANG_PORT},
#     2. bootstraps to the given peer, and
#     3. registers the `llm` service with label `model=zai-org/GLM-4.7-Flash`.
#   The OpenTela head then routes /v1/service/llm/... requests to these peers.
#
# CONTAINER ENGINE: the image is supplied by ./sglang-rocm.toml (EDF), found via
#   EDF_PATH below. `srun --environment=sglang-rocm` runs everything inside it.
set -euo pipefail

# ---------------------------------------------------------------- deployment --
DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm47-flash-sglang-beverin}"
SCRIPT_DIR="${DEPLOY_DIR}/deploy"
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-4.7-Flash}"
SGLANG_PORT="${SGLANG_PORT:-8080}"
TP_SIZE="${TP_SIZE:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"
SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-glm45}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-glm47}"
# CUDA graph capture SIGKILLs on ROCm MI300A with GLM-4.7-Flash's MTP
# (nextn_predict) layers ~30 s after weight load ends, during the
# capture/warmup phase (confirmed vs the known-good beverin
# sglang_subprocess-73057.sh, which forces both flags off). Override to
# 0 only if you have validated capture on a specific image/model combo.
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-${SGLANG_SKIP_SERVER_WARMUP:-1}}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# GLM-4.7-Flash (model_type glm4_moe_lite, with nextn_predict_layers) is newer
# than the transformers pinned in some SGLang images. The v0.5.16 ROCm image
# already ships 5.12.1, so this guard is normally a no-op; it only installs if
# the image is older. system mode keeps the upgraded package on the container
# filesystem (target mode would isolate it on /capstor).
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"
TRANSFORMERS_INSTALL_MODE="${TRANSFORMERS_INSTALL_MODE:-system}"
TRANSFORMERS_TARGET_DIR="${TRANSFORMERS_TARGET_DIR:-${DEPLOY_DIR}/cache/pydeps/transformers-${TRANSFORMERS_VERSION}}"

# ------------------------------------------------------------------- otela ----
# x86-64 build, version sai-v0.0.6. The worker (not a relay) can use this
# release directly; --require_signed is left false so a dev build is accepted.
OPENTELA_DIR="${OPENTELA_DIR:-/capstor/scratch/cscs/xyao/opentela}"
OPENTELA_BIN="${OPENTELA_BIN:-${OPENTELA_DIR}/otela}"
OPENTELA_BOOTSTRAP="${OPENTELA_BOOTSTRAP:-/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7}"
OPENTELA_TCP_PORT="${OPENTELA_TCP_PORT:-43905}"
OPENTELA_UDP_PORT="${OPENTELA_UDP_PORT:-59820}"
OPENTELA_SERVICE_NAME="${OPENTELA_SERVICE_NAME:-llm}"

# Per-rank metrics scraping (optional; only starts if the binary exists).
METRICS_REMOTE_WRITE_URL="${METRICS_REMOTE_WRITE_URL:-https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write}"
METRICS_AGENT_BIN="${METRICS_AGENT_BIN:-/capstor/store/cscs/swissai/infra01/ocf-share/vmagent-amd64}"

export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" \
  "${DEPLOY_DIR}/home" \
  "${DEPLOY_DIR}/cache/hf" \
  "${DEPLOY_DIR}/cache/xdg" \
  "${DEPLOY_DIR}/cache/config" \
  "${DEPLOY_DIR}/cache/triton" \
  "${DEPLOY_DIR}/cache/pip" \
  "${DEPLOY_DIR}/cache/tmp"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD_NODE="${NODES[0]}"
NUM_NODES="${#NODES[@]}"
# Full logical-CPU count of one MI300A node (192). The upstream sglang ROCm
# image bakes SGLANG_SET_CPU_AFFINITY=1 into /etc/environment and pins each TP
# rank to its GPU's NUMA die using host CPU ids 0..191; the srun step MUST give
# each task the whole node (SLURM_CPUS_ON_NODE) or the pin will raise
# "CPU number N is not eligible" and kill ranks 2/3. Slurm sets
# SLURM_CPUS_ON_NODE for the job; fall back to scontrol, then to 192.
NODE_CPUS="${SLURM_CPUS_ON_NODE:-$(scontrol show node "${HEAD_NODE}" 2>/dev/null | awk -F= '/^CPUTot=/{print $2; exit}')}"
NODE_CPUS="${NODE_CPUS:-192}"

echo "job_id=${SLURM_JOB_ID}"
echo "num_nodes=${NUM_NODES}"
echo "nodes=${NODES[*]}"
echo "head_node=${HEAD_NODE}"
echo "node_cpus=${NODE_CPUS}"
echo "model=${MODEL}"
echo "served_model_name=${SERVED_MODEL_NAME}"
echo "tp=${TP_SIZE}"
echo "max_model_len=${MAX_MODEL_LEN} max_running_requests=${MAX_RUNNING_REQUESTS} mem_fraction_static=${MEM_FRACTION_STATIC}"
echo "reasoning_parser=${SGLANG_REASONING_PARSER} tool_call_parser=${SGLANG_TOOL_CALL_PARSER:-<none>} disable_cuda_graph=${DISABLE_CUDA_GRAPH} skip_server_warmup=${SKIP_SERVER_WARMUP}"
echo "port=${SGLANG_PORT}"
echo "transformers_pin=${TRANSFORMERS_VERSION} install_mode=${TRANSFORMERS_INSTALL_MODE}"
echo "opentela_bin=${OPENTELA_BIN}"
echo "opentela_bootstrap=${OPENTELA_BOOTSTRAP}"
echo "metrics_agent_bin=${METRICS_AGENT_BIN}"
echo "started=$(date --iso-8601=seconds)"

[ -d "${MODEL}" ] || { echo "FATAL: weights not found: ${MODEL}" >&2; exit 1; }
[ -x "${OPENTELA_BIN}" ] || { echo "FATAL: otela binary not found/executable: ${OPENTELA_BIN}" >&2; exit 1; }

# ================================================================ engine ======
# Shared SGLang subprocess wrapper used by every rank. Each rank runs on its
# own node, so one wrapper is safe to share. Single-node TP=4: no cross-node
# init, no torchrun, no --nnodes/--dist-init-addr.
SGLANG_SUBPROCESS="${DEPLOY_DIR}/sglang_subprocess-${SLURM_JOB_ID}.sh"

# MI300A APU memory-accounting fix. PyTorch reports is_integrated=True for
# every MI300A GPU, but each GPU has its OWN dedicated 137 GB HBM3 (not
# shared sysmem like a laptop iGPU). sglang's get_available_gpu_memory()
# then uses psutil.virtual_memory().available (whole-node ~428 GB) per TP
# rank instead of per-GPU mem_get_info (137 GB), so with mem_fraction_static
# =0.85 each of 4 ranks computes a ~301 GB KV budget and 4*301 >> 365 GB
# physically free -> cgroup OOM (SIGKILL, exit -9) ~35 s after Load weight
# end, with no Python traceback. This launcher wraps
# torch.cuda.get_device_properties to report is_integrated=False, restoring
# correct per-GPU accounting (same as a discrete GPU).
SGLANG_LAUNCHER="${DEPLOY_DIR}/sglang_launcher_mi300a.py"
cat > "${SGLANG_LAUNCHER}" <<'SGLANG_LAUNCHER_PY'
#!/usr/bin/env python3
"""sglang launcher with MI300A integrated-memory accounting fix."""
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


def _patched_get_device_properties(device=None):
    return _DevicePropsProxy(_orig_get_device_properties(device))


torch.cuda.get_device_properties = _patched_get_device_properties


def main():
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.launch_server import run_server

    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
SGLANG_LAUNCHER_PY

cat > "${SGLANG_SUBPROCESS}" <<SGLANG_WRAPPER
#!/bin/bash
set -euo pipefail

echo "sglang_subprocess_started rank=\${SLURM_PROCID} host=\$(hostname) \$(date --iso-8601=seconds)"

# tempfile on /capstor, NOT the Slurm default /users/<u>/.tmp (home is
# quota-limited; sglang's tempfile cleanup on a quota dir raises
# OSError [Errno 39] Directory not empty). Belt-and-suspenders alongside the
# EDF [env] TMPDIR= so this holds even if a parent login shell re-exported it.
export TMPDIR="${DEPLOY_DIR}/cache/tmp"
mkdir -p "${TMPDIR}"
# Override the image's /etc/environment SGLANG_USE_AITER=1: aiter's RoPE
# crashes on GLM-4.7-Flash (no original_max_position_embeddings in
# rope_scaling). sglang's own get_rope handles it. See sglang-rocm.toml.
export SGLANG_USE_AITER=0

SGLANG_ARGS=(
  --model-path "${MODEL}"
  --host 0.0.0.0
  --port "${SGLANG_PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP_SIZE}"
  --trust-remote-code
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

# GLM-4.7-Flash (model_type glm4_moe_lite) is newer than the transformers
# pinned in the official SGLang ROCm image. Upgrade at startup (compute nodes
# have PyPI access) with a persistent pip cache. No-op when already correct.
export PIP_CACHE_DIR="${DEPLOY_DIR}/cache/pip"
if [[ "${TRANSFORMERS_INSTALL_MODE}" == "system" ]]; then
  CURRENT_TRANSFORMERS_VERSION="\$(python - <<'PY' 2>/dev/null || true
import importlib.metadata
try:
    print(importlib.metadata.version("transformers"), end="")
except importlib.metadata.PackageNotFoundError:
    pass
PY
)"
  if [[ "\${CURRENT_TRANSFORMERS_VERSION}" != "${TRANSFORMERS_VERSION}" ]]; then
    echo "transformers_install_start mode=system current=\${CURRENT_TRANSFORMERS_VERSION:-<none>} desired=${TRANSFORMERS_VERSION} rank=\${SLURM_PROCID} \$(date --iso-8601=seconds)"
    python -m pip install --no-input --break-system-packages --upgrade \
      "transformers==${TRANSFORMERS_VERSION}" >&2
    CURRENT_TRANSFORMERS_VERSION="\$(python - <<'PY' 2>/dev/null || true
import importlib.metadata
try:
    print(importlib.metadata.version("transformers"), end="")
except importlib.metadata.PackageNotFoundError:
    pass
PY
)"
  else
    echo "transformers_install_skipped mode=system current=\${CURRENT_TRANSFORMERS_VERSION} rank=\${SLURM_PROCID} \$(date --iso-8601=seconds)"
  fi
else
  mkdir -p "${TRANSFORMERS_TARGET_DIR}"
  export PYTHONPATH="${TRANSFORMERS_TARGET_DIR}:\${PYTHONPATH:-}"
  TRANSFORMERS_MARKER="${TRANSFORMERS_TARGET_DIR}/.transformers-version"
  CURRENT_TRANSFORMERS_VERSION=""
  if [[ -r "\${TRANSFORMERS_MARKER}" && -r "${TRANSFORMERS_TARGET_DIR}/transformers/__init__.py" ]]; then
    CURRENT_TRANSFORMERS_VERSION="\$(tr -d '[:space:]' < "\${TRANSFORMERS_MARKER}" || true)"
  fi
  if [[ "\${CURRENT_TRANSFORMERS_VERSION}" != "${TRANSFORMERS_VERSION}" ]]; then
    CURRENT_TRANSFORMERS_VERSION="\$(python - <<'PY' 2>/dev/null || true
import importlib.metadata
try:
    print(importlib.metadata.version("transformers"), end="")
except importlib.metadata.PackageNotFoundError:
    pass
PY
)"
  fi
  if [[ "\${CURRENT_TRANSFORMERS_VERSION}" != "${TRANSFORMERS_VERSION}" ]]; then
    echo "transformers_install_start mode=target target=${TRANSFORMERS_TARGET_DIR} current=\${CURRENT_TRANSFORMERS_VERSION:-<none>} desired=${TRANSFORMERS_VERSION} rank=\${SLURM_PROCID} \$(date --iso-8601=seconds)"
    python -m pip install --no-input --target "${TRANSFORMERS_TARGET_DIR}" --upgrade \
      "transformers==${TRANSFORMERS_VERSION}" >&2
    CURRENT_TRANSFORMERS_VERSION="\$(python - <<'PY' 2>/dev/null || true
import importlib.metadata
try:
    print(importlib.metadata.version("transformers"), end="")
except importlib.metadata.PackageNotFoundError:
    pass
PY
)"
  else
    echo "transformers_install_skipped mode=target target=${TRANSFORMERS_TARGET_DIR} current=\${CURRENT_TRANSFORMERS_VERSION} rank=\${SLURM_PROCID} \$(date --iso-8601=seconds)"
  fi
  if [[ "\${CURRENT_TRANSFORMERS_VERSION}" == "${TRANSFORMERS_VERSION}" ]]; then
    printf "%s\n" "\${CURRENT_TRANSFORMERS_VERSION}" > "\${TRANSFORMERS_MARKER}"
  fi
fi
if [[ "\${CURRENT_TRANSFORMERS_VERSION}" != "${TRANSFORMERS_VERSION}" ]]; then
  echo "transformers_version_mismatch current=\${CURRENT_TRANSFORMERS_VERSION:-<none>} desired=${TRANSFORMERS_VERSION}" >&2
  exit 2
fi
echo "transformers_version=\${CURRENT_TRANSFORMERS_VERSION} rank=\${SLURM_PROCID} \$(date --iso-8601=seconds)"

exec python "${SGLANG_LAUNCHER}" "\${SGLANG_ARGS[@]}"
SGLANG_WRAPPER
chmod +x "${SGLANG_SUBPROCESS}"

# ============================================================ otela worker ====
# Per-rank vmagent scrape config template; the actual container IP is filled in
# by the launcher (vmagent's Go resolver has trouble reaching SGLang via
# localhost in this container setup).
SCRAPE_CONFIG_TEMPLATE="${DEPLOY_DIR}/vmagent-scrape-${SLURM_JOB_ID}.yaml.tpl"
cat > "${SCRAPE_CONFIG_TEMPLATE}" <<SCRAPE
global:
  scrape_interval: 30s
scrape_configs:
  - job_name: slurm-llm
    metrics_path: /metrics
    static_configs:
      - targets: ['LOCALHOST_IP:${SGLANG_PORT}']
SCRAPE

# OpenTela launcher runs inside the SGLang container on every rank. It uses
# SLURM_PROCID to create a per-rank config/datastore directory on the shared
# filesystem so concurrent OpenTela instances do not collide.
OTELA_LAUNCHER="${DEPLOY_DIR}/opentela_launcher-${SLURM_JOB_ID}.sh"
cat > "${OTELA_LAUNCHER}" <<OTELA_LAUNCHER
#!/bin/bash
set -euo pipefail

RANK="\${SLURM_PROCID}"
NODE="\$(hostname)"
LOCALHOST_IP="\$(hostname -i | awk '{print \$1}')"
# Per-rank libp2p ports. Every peer used to bind the same fixed tcp/udp port,
# so multiple peers sharing one NAT egress collided on a single external port
# mapping to the bootstrap and only ~half registered. Offsetting by
# SLURM_PROCID gives each peer a distinct port.
TCP_PORT=\$(( ${OPENTELA_TCP_PORT} + \${RANK} ))
UDP_PORT=\$(( ${OPENTELA_UDP_PORT} + \${RANK} ))
OPENTELA_HOME="${DEPLOY_DIR}/home/opentela-${SLURM_JOB_ID}-\${RANK}"
OPENTELA_CONFIG="\${OPENTELA_HOME}/.config/otela/cfg.yaml"
OPENTELA_LOG="${DEPLOY_DIR}/logs/opentela-${SLURM_JOB_ID}-\${RANK}.log"
VMAGENT_LOG="${DEPLOY_DIR}/logs/vmagent-${SLURM_JOB_ID}-\${RANK}.log"
SCRAPE_CONFIG="${DEPLOY_DIR}/vmagent-scrape-${SLURM_JOB_ID}-\${RANK}.yaml"
mkdir -p "\${OPENTELA_HOME}/.config/otela"

export HOME="\${OPENTELA_HOME}"

sed "s/LOCALHOST_IP/\${LOCALHOST_IP}/g" "${SCRAPE_CONFIG_TEMPLATE}" > "\${SCRAPE_CONFIG}"

"${OPENTELA_BIN}" init --config "\${OPENTELA_CONFIG}" >/dev/null 2>&1 || true

echo "opentela_launcher_started rank=\${RANK} node=\${NODE} ip=\${LOCALHOST_IP} tcp=\${TCP_PORT} udp=\${UDP_PORT} home=\${OPENTELA_HOME} \$(date --iso-8601=seconds)"

# Stagger peer joins so all N ranks do not hit the single bootstrap in the same
# instant (the thundering herd was correlated with failed registrations).
sleep \$(( \${RANK} * 10 ))

# OpenTela is backgrounded with stdin from /dev/zero because it exits when run
# as a foreground process with EOF on stdin. It will start SGLang as its
# subprocess.
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
# round-robining traffic into a dead endpoint. Mirrors the deepseek-v4 recipe.
# Previously this recipe had no teardown at all — otela was left to be reaped by
# SLURM, so it depended entirely on winning an unguaranteed race against KillWait.
cleanup() {
  # The trap fires for both TERM and EXIT, and bash runs the EXIT trap after the
  # TERM handler returns, so guard against running the teardown twice.
  [ "\${_OTELA_CLEANED:-0}" = 1 ] && return 0
  _OTELA_CLEANED=1
  echo "otela_stopping rank=\${RANK} \$(date --iso-8601=seconds)"
  kill -TERM "\${OPENTELA_PID}" 2>/dev/null || true
  # 20s: --signal=TERM@120 covers wall-time expiry, but on the \`scancel\` path the
  # only grace is KillWait (default 30s) and otela needs ~10s to shut down
  # (AnnounceLeave -> 5s CRDT drain -> srv.Shutdown). 20s leaves margin.
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

# Wait for SGLang to be ready before starting vmagent. vmagent's Go HTTP client
# can get stuck if it starts scraping before the target is listening. SGLang's
# /get_model_info returns 200 only once the model is loaded (no inference).
echo "waiting_for_sglang rank=\${RANK} url=http://\${LOCALHOST_IP}:${SGLANG_PORT}/get_model_info \$(date --iso-8601=seconds)"
while true; do
  if curl -s "http://\${LOCALHOST_IP}:${SGLANG_PORT}/get_model_info" >/dev/null 2>&1; then
    echo "sglang_ready rank=\${RANK} \$(date --iso-8601=seconds)"
    break
  fi
  sleep 2
done

# Start vmagent to scrape SGLang /metrics and push to the remote write endpoint.
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

if [[ -n "\${VMAGENT_PID}" ]]; then
  kill "\${VMAGENT_PID}" 2>/dev/null || true
  echo "vmagent_stopped rank=\${RANK} \$(date --iso-8601=seconds)"
fi

exit "\${OPENTELA_CODE}"
OTELA_LAUNCHER
chmod +x "${OTELA_LAUNCHER}"

# Record endpoints for all nodes before srun starts so the metadata is
# available while the services are still running.
{
  echo "JOB_ID=${SLURM_JOB_ID}"
  echo "JOBID=${SLURM_JOB_ID}"
  echo "NUM_NODES=${NUM_NODES}"
  echo "PORT=${SGLANG_PORT}"
  echo "MODEL=${SERVED_MODEL_NAME}"
  echo "FRAMEWORK=sglang"
  echo "BOOTSTRAP=${OPENTELA_BOOTSTRAP}"
  echo "SUBMITTED_AT=$(date --iso-8601=seconds)"
  echo "LOG=${DEPLOY_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
  for i in "${!NODES[@]}"; do
    echo "ENDPOINT_${i}=${NODES[$i]}:${SGLANG_PORT}"
  done
} > "${DEPLOY_DIR}/last_service.env"

# Launch one OpenTela+SGLang service per node inside the SGLang ROCm container.
# --cpus-per-task=NODE_CPUS gives each (1 per node) task the whole node
# (0..191) so the image's SGLANG_SET_CPU_AFFINITY NUMA pinning is valid.
srun \
  --nodes="${NUM_NODES}" \
  --ntasks="${NUM_NODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${NODE_CPUS}" \
  --gpus-per-task=4 \
  --environment=sglang-rocm \
  bash -lc "${OTELA_LAUNCHER}"
SRUN_EXIT=$?
echo "srun_exited code=${SRUN_EXIT} $(date --iso-8601=seconds)"

exit "${SRUN_EXIT}"
