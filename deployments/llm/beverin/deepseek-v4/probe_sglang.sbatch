#!/bin/bash
#SBATCH --job-name=dsv4-flash-probe
#SBATCH --partition=mi300
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Do NOT set --cpus-per-task here: shrinking the cpuset breaks the image's
# SGLANG_SET_CPU_AFFINITY=1 (set in /etc/environment), whose NUMA pinning
# assumes the full 0..191 logical-CPU cpuset (one MI300A die per TP rank).
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/xyao/deepseek-v4-flash-sglang/logs/%x-%j.out
#
# 1-node bring-up probe for deepseek-ai/DeepSeek-V4-Flash on beverin (MI300A).
# Starts SGLang ALONE (no OpenTela), waits for /get_model_info, runs a real
# /generate + /v1/chat/completions, dumps HBM accounting, then exits.
#
# This is the cheap gate before the full serve: it catches the fp4-expert,
# memory-accounting, and architecture problems on one node instead of N.
#
#   sbatch probe_sglang.sbatch
#   # try the fp4->fp8 dequant fallback instead of aiter's native fp4 MoE:
#   sbatch --export=ALL,DSV4_FP4_DEQUANT=1,SGLANG_USE_AITER=0 probe_sglang.sbatch
set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/deepseek-v4-flash-sglang}"
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V4-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Flash}"
SGLANG_PORT="${SGLANG_PORT:-8080}"
TP_SIZE="${TP_SIZE:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-1}"
# Routed experts ship mxfp4-packed (I8 + F8_E8M0 scales). Two ways to run them
# on gfx942, which has no native MX compute (mxfp_supported() is gfx95-only):
#   SGLANG_USE_AITER=1 (default) -> aiter's native MXFP4 MoE, weights stay fp4
#                                   (~132 GiB experts)
#   DSV4_FP4_DEQUANT=1           -> cast_e2m1fn_to_e4m3fn at load, weights
#                                   double to fp8 (~264 GiB experts)
SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
DSV4_FP4_DEQUANT="${DSV4_FP4_DEQUANT:-0}"
# DSA sparse-attention kernel. On HIP the backend is ALWAYS taken from this env
# var (hip_flash_mla.py overrides whatever --dsa-*-backend says), and its
# upstream default "tilelang" does not compile on gfx942 — the TVM software
# pipeline pass aborts with "two statements with buffer access dependency in
# the same stage of the software pipeline cannot be reordered". "triton" is the
# documented ROCm alternative; "unified_kv_triton" is the other HIP path.
FLASHMLA_BACKEND="${FLASHMLA_BACKEND:-triton}"
# Persistent aiter JIT cache — the CK MXFP4 MoE kernel is hipcc-compiled on the
# FIRST request and aiter's default cache dir is inside the ephemeral
# container. Seeded from the image below (an empty dir would make aiter rebuild
# the prebuilt modules too).
AITER_JIT_DIR="${AITER_JIT_DIR:-${DEPLOY_DIR}/cache/aiter-jit}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"

export EDF_PATH="${DEPLOY_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}"/{logs,home,run} \
  "${DEPLOY_DIR}"/cache/{hf,xdg,config,triton,pip,tmp}

NODE_CPUS="${SLURM_CPUS_ON_NODE:-192}"

echo "job_id=${SLURM_JOB_ID} node=$(hostname) node_cpus=${NODE_CPUS}"
echo "model=${MODEL}"
echo "tp=${TP_SIZE} max_model_len=${MAX_MODEL_LEN} mem_fraction_static=${MEM_FRACTION_STATIC}"
echo "use_aiter=${SGLANG_USE_AITER} dsv4_fp4_dequant=${DSV4_FP4_DEQUANT} flashmla_backend=${FLASHMLA_BACKEND}"
echo "disable_cuda_graph=${DISABLE_CUDA_GRAPH} skip_warmup=${SKIP_SERVER_WARMUP}"
echo "started=$(date --iso-8601=seconds)"

[ -d "${MODEL}" ] || { echo "FATAL: weights not found: ${MODEL}" >&2; exit 1; }

RUNNER="${DEPLOY_DIR}/run/probe_run-${SLURM_JOB_ID}.sh"
cat > "${RUNNER}" <<RUNNER_EOF
#!/bin/bash
set -uo pipefail
export TMPDIR="${DEPLOY_DIR}/cache/tmp"
mkdir -p "\${TMPDIR}"
export SGLANG_USE_AITER="${SGLANG_USE_AITER}"
export SGLANG_DSV4_FP4_DEQUANT="${DSV4_FP4_DEQUANT}"
export SGLANG_HACK_FLASHMLA_BACKEND="${FLASHMLA_BACKEND}"
AITER_IMAGE_JIT_DIR=/sgl-workspace/aiter/aiter/jit
if [[ -d "\${AITER_IMAGE_JIT_DIR}" ]]; then
  if [[ ! -e "${AITER_JIT_DIR}/.seeded" ]]; then
    echo "aiter_jit_seed_start dst=${AITER_JIT_DIR} \$(date --iso-8601=seconds)"
    mkdir -p "${AITER_JIT_DIR}"
    cp -rn "\${AITER_IMAGE_JIT_DIR}/." "${AITER_JIT_DIR}/" 2>/dev/null || true
    ( cd "${AITER_JIT_DIR}" && find . -type f ! -name '.seeded*' -print ) \\
      > "${AITER_JIT_DIR}/.seeded.manifest" 2>/dev/null || true
    touch "${AITER_JIT_DIR}/.seeded"
    echo "aiter_jit_seed_done \$(date --iso-8601=seconds)"
  fi
  export AITER_JIT_DIR="${AITER_JIT_DIR}"
  echo "aiter_jit_dir=\${AITER_JIT_DIR}"
  export PROV_IMAGE_DIGEST="sha256:80d04638deb64fac000fa565cb46e5d2f692173dc125a32a956014a6383ecaee"
  # Provenance keys the shared kernel bucket (see sync_aiter_kernels.sh).
  python3 - >"${AITER_JIT_DIR}/../aiter-provenance.env" 2>/dev/null <<'PROV_PY' || true
import os, torch
def q(v): return str(v).replace("'", "")
try: arch = torch.cuda.get_device_properties(0).gcnArchName
except Exception: arch = ""
try:
    import sglang; sgl = sglang.__version__
except Exception: sgl = "unknown"
print(f"PROV_ARCH='{q(arch)}'")
print(f"PROV_AITER_COMMIT='{q(os.environ.get('AITER_COMMIT',''))}'")
print(f"PROV_IMAGE_DIGEST='{q(os.environ.get('PROV_IMAGE_DIGEST',''))}'")
print(f"PROV_ROCM_VERSION='{q(torch.version.hip)}'")
print(f"PROV_TORCH_VERSION='{q(torch.__version__)}'")
print(f"PROV_SGLANG_VERSION='{q(sgl)}'")
PROV_PY
fi
IP="\$(hostname -i | awk '{print \$1}')"

# MI300A APU memory-accounting fix (see README fix #1).
LAUNCHER="${DEPLOY_DIR}/run/sglang_launcher_mi300a.py"
cat > "\${LAUNCHER}" <<'PY'
import os, sys, torch


class _P:
    __slots__ = ("_orig",)

    def __init__(self, o):
        object.__setattr__(self, "_orig", o)

    def __getattr__(self, n):
        if n == "is_integrated":
            return False
        return getattr(object.__getattribute__(self, "_orig"), n)


_o = torch.cuda.get_device_properties
torch.cuda.get_device_properties = lambda d: _P(_o(d))


def main():
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.launch_server import run_server

    load_plugins()
    sa = prepare_server_args(sys.argv[1:])
    try:
        run_server(sa)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


# The __main__ guard is LOAD-BEARING: sglang starts its per-TP-rank scheduler
# processes with multiprocessing "spawn", which re-imports this file in every
# child. Without the guard, run_server() re-executes at import time and the
# child dies with "An attempt has been made to start a new process before the
# current process has finished its bootstrapping phase" -> the parent then
# reports "Rank N scheduler died during initialization (exit code: 1)".
if __name__ == "__main__":
    main()
PY

ARGS=(
  --model-path "${MODEL}"
  --host 0.0.0.0
  --port "${SGLANG_PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP_SIZE}"
  --context-length "${MAX_MODEL_LEN}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --reasoning-parser deepseek-v4
  --enable-metrics
)
[[ "${DISABLE_CUDA_GRAPH}" == "1" ]] && ARGS+=(--disable-cuda-graph)
[[ "${SKIP_SERVER_WARMUP}" == "1" ]] && ARGS+=(--skip-server-warmup)

echo "engine_launch \$(date --iso-8601=seconds)"
python "\${LAUNCHER}" "\${ARGS[@]}" > "${DEPLOY_DIR}/logs/engine-${SLURM_JOB_ID}.log" 2>&1 &
ENGINE_PID=\$!
echo "engine_pid=\${ENGINE_PID} ip=\${IP}"

# Wait for readiness, failing fast if the engine dies.
DEADLINE=\$(( \$(date +%s) + ${READY_TIMEOUT} ))
READY=0
while [ "\$(date +%s)" -lt "\${DEADLINE}" ]; do
  if ! kill -0 "\${ENGINE_PID}" 2>/dev/null; then
    echo "ENGINE_DIED before ready \$(date --iso-8601=seconds)"
    break
  fi
  if curl -sf "http://\${IP}:${SGLANG_PORT}/get_model_info" >/dev/null 2>&1; then
    READY=1
    echo "sglang_ready \$(date --iso-8601=seconds)"
    break
  fi
  sleep 5
done

if [ "\${READY}" != "1" ]; then
  echo "===== PROBE FAILED: engine not ready ====="
  echo "--- last 120 lines of engine log ---"
  tail -120 "${DEPLOY_DIR}/logs/engine-${SLURM_JOB_ID}.log"
  kill "\${ENGINE_PID}" 2>/dev/null || true
  exit 1
fi

echo "===== get_model_info ====="
curl -s "http://\${IP}:${SGLANG_PORT}/get_model_info"; echo
echo "===== /v1/models ====="
curl -s "http://\${IP}:${SGLANG_PORT}/v1/models"; echo
echo "===== /generate ====="
curl -s "http://\${IP}:${SGLANG_PORT}/generate" -H 'Content-Type: application/json' \
  -d '{"text":"The capital of Switzerland is","sampling_params":{"max_new_tokens":24,"temperature":0}}'; echo
echo "===== /v1/chat/completions ====="
curl -s "http://\${IP}:${SGLANG_PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"${SERVED_MODEL_NAME}","messages":[{"role":"user","content":"In one sentence, what is tensor parallelism?"}],"max_tokens":128,"temperature":0}'; echo
echo "===== HBM per GPU ====="
rocm-smi --showmeminfo vram 2>/dev/null | grep -Ei "GPU\[|Total|Used" | head -20 || true
echo "===== KV pool / mem lines from engine log ====="
grep -Ei "KV cache|max_total_num_tokens|Load weight end|available_gpu_mem|Dequantized" \
  "${DEPLOY_DIR}/logs/engine-${SLURM_JOB_ID}.log" | tail -20 || true

kill "\${ENGINE_PID}" 2>/dev/null || true
echo "PROBE_OK \$(date --iso-8601=seconds)"
RUNNER_EOF
chmod +x "${RUNNER}"

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${NODE_CPUS}" --gpus-per-task=4 \
  --environment=deepseek-v4-rocm bash -lc "${RUNNER}"
echo "srun_exited code=$? $(date --iso-8601=seconds)"
