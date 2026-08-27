#!/bin/bash
#SBATCH --job-name=glm53-dl
#SBATCH --account=infra02
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/capstor/scratch/cscs/xyao/opentela-cookbook/logs/glm53-download-%j.out
#SBATCH --error=/capstor/scratch/cscs/xyao/opentela-cookbook/logs/glm53-download-%j.out
#
# Download zai-org/GLM-5.3-Flash (328 GB BF16, 62 safetensors shards +
# config.json/chat_template.jinja/LICENSE) to /capstor/scratch (confirmed fast
# metadata), then move to the canonical /capstor/store model cache.
#
# WHY scratch-first: /capstor/store metadata (MDT) was overloaded on
# 2026-08-27 (many running serving jobs) — mkdir/stat on the model dir hung
# and `hf download` raised BrokenPipeError [Errno 108] on .cache creation
# (jobs 3201560/3201604/3201724, all stuck at 0/72 for 4+ min). /capstor/scratch
# (33 PB free) has no such issue. The move to /capstor/store is non-fatal: if
# it times out, the model is ready to serve from /capstor/scratch (override
# MODEL_PATH in the serve recipe).
#
# Run on a Clariden debug node (like build_kimi_k3_image.sbatch): the job
# survives SSH disconnect and clariden compute has outbound egress to HF.
# Debug caps at 90 min; the 328 GB download targets ~40-55 min.
#
#   sbatch deployments/llm/clariden/glm-53-flash/download_glm_5_3_flash.sbatch
#
# Monitor:
#   scontrol show job <JOBID>
#   tail -f /capstor/scratch/cscs/xyao/opentela-cookbook/logs/glm53-download-<JOBID>.out
#
# Env overrides:
#   MODEL_TARGET  default /capstor/scratch/cscs/xyao/models/zai-org/GLM-5.3-Flash
#   STORE_TARGET  default /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.3-Flash
#                 (set empty to skip the post-download move)
#   HF_HOME       default /capstor/scratch/cscs/xyao/models  (hub cache on /capstor, NOT home quota)
#   CLEAN_CACHE   default 1 — remove a stale/broken .cache before starting.
#                 Set 0 to resume a good partial download (keeps hf metadata).
#   HF_REPO       default zai-org/GLM-5.3-Flash
set -uo pipefail

MODEL_TARGET="${MODEL_TARGET:-/capstor/scratch/cscs/xyao/models/zai-org/GLM-5.3-Flash}"
STORE_TARGET="${STORE_TARGET:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.3-Flash}"
HF_REPO="${HF_REPO:-zai-org/GLM-5.3-Flash}"
export HF_HOME="${HF_HOME:-/capstor/scratch/cscs/xyao/models}"

LOG_DIR=/capstor/scratch/cscs/xyao/opentela-cookbook/logs
mkdir -p "$MODEL_TARGET" "$LOG_DIR" "$HF_HOME"

# hf lives in the user's ~/.local/bin (mounted on compute via the home FS).
HF_BIN="$(command -v hf || true)"
[ -n "$HF_BIN" ] || HF_BIN="$HOME/.local/bin/hf"
[ -x "$HF_BIN" ] || { echo "FATAL: hf CLI not found (looked in PATH and $HOME/.local/bin). Install with: pip install --user huggingface_hub[hf_xet]" >&2; exit 1; }

# GLM-5.3-Flash shards are served from HF Xet storage, which requires the
# compiled hf_xet extension. /users is shared CSCS NFS across x86_64
# (beverin) and aarch64 (clariden) hosts, so the hf_xet in the default
# ~/.local/lib/python3.11/site-packages is an x86_64 build that the
# clariden (aarch64) loader cannot dlopen (ImportError: ... hf_xet.abi3.so:
# cannot open shared object file). A per-arch aarch64 build was installed
# aside via `python3.11 -m pip install --target /users/xyao/.local-aarch64-py311 hf_xet`
# and is prepended here so the aarch64 build wins without disturbing the
# x86_64 one used on beverin. (observed 2026-08-27, job 3201943)
export PYTHONPATH="/users/xyao/.local-aarch64-py311${PYTHONPATH:+:$PYTHONPATH}"
python3.11 -c "import hf_xet" 2>/dev/null \
  || { echo "FATAL: hf_xet import failed on $(uname -m). Install with: python3.11 -m pip install --target /users/xyao/.local-aarch64-py311 hf_xet" >&2; exit 1; }

echo "[glm53-dl $(date -Is)] host=$(hostname -s) repo=$HF_REPO"
echo "[glm53-dl ] target=$MODEL_TARGET"
echo "[glm53-dl ] hf=$($HF_BIN --version 2>&1 | head -1)  HF_HOME=$HF_HOME"

# A prior download killed mid-mkdir leaves a .cache/ that raises
# BrokenPipeError on the next os.stat (observed 2026-08-27 when
# MODEL_TARGET was on /capstor/store). Remove it before starting so the
# dir is clean. On /capstor/scratch this is cheap; on /capstor/store it
# may hang — set CLEAN_CACHE=0 to resume an existing good download.
if [ "${CLEAN_CACHE:-1}" = "1" ] && [ -d "$MODEL_TARGET/.cache" ]; then
  echo "[glm53-dl ] removing stale .cache"
  rm -rf "$MODEL_TARGET/.cache" || true
fi

# Default Lustre stripe (like GLM-4.7-Flash). A prior attempt used
# `lfs setstripe -S 4m -c -1` (over-stripe across all OSTs) but the MDT
# rejected subsequent mkdir/stat on that directory with BrokenPipeError
# [Errno 108] (observed 2026-08-27, jobs 3201560/3201604). Default stripe
# avoids the issue. To speed up cold-start reads AFTER the download,
# run from the login node:
#   lfs migrate -S 4m -c -1 $STORE_TARGET/*.safetensors

echo "[glm53-dl ] starting hf download ..."
# Resumable: hf skips files already present + complete in $MODEL_TARGET and
# tracked in .cache/.metadata. Writes the shard list to stdout for monitoring.
"$HF_BIN" download "$HF_REPO" --local-dir "$MODEL_TARGET"
rc=$?
echo "[glm53-dl ] hf download rc=$rc ($(date -Is))"

# Verify: 62 safetensors shards + the two model-def files the engine needs.
SHARDS=$(ls -1 "$MODEL_TARGET"/*.safetensors 2>/dev/null | wc -l)
INDEX=$(ls -1 "$MODEL_TARGET"/model.safetensors.index.json 2>/dev/null | wc -l)
CFG=$([ -f "$MODEL_TARGET/config.json" ] && echo 1 || echo 0)
TPL=$([ -f "$MODEL_TARGET/chat_template.jinja" ] && echo 1 || echo 0)
echo "[glm53-dl ] shards=$SHARDS/62  index=$INDEX  config.json=$CFG  chat_template.jinja=$TPL"
echo "[glm53-dl ] du=$(du -sh "$MODEL_TARGET" 2>/dev/null | awk '{print $1}')"

if [ "$rc" -ne 0 ]; then
  echo "FAIL: hf download exited non-zero (rc=$rc). Re-run with CLEAN_CACHE=0 to resume, or CLEAN_CACHE=1 to restart clean." >&2
  exit "$rc"
fi
if [ "$SHARDS" -ne 62 ] || [ "$CFG" != "1" ]; then
  echo "FAIL: expected 62 shards + config.json, got $SHARDS shards (config=$CFG). Re-run with CLEAN_CACHE=0 to resume." >&2
  exit 1
fi
echo "OK: GLM-5.3-Flash complete at $MODEL_TARGET (62 shards, $(du -sh "$MODEL_TARGET" | awk '{print $1}'))."

# Move to the canonical /capstor/store model cache (non-fatal: if the
# store MDT is overloaded the mv times out and the model is ready to
# serve from /capstor/scratch — override MODEL_PATH in the serve recipe).
if [ -n "$STORE_TARGET" ] && [ "$MODEL_TARGET" != "$STORE_TARGET" ]; then
  echo "[glm53-dl ] moving $MODEL_TARGET -> $STORE_TARGET (non-fatal on timeout)"
  mkdir -p "$(dirname "$STORE_TARGET")" 2>/dev/null || true
  if mv "$MODEL_TARGET" "$STORE_TARGET" 2>&1; then
    echo "OK: GLM-5.3-Flash at $STORE_TARGET (62 shards). Ready to serve."
  else
    echo "WARN: move to $STORE_TARGET failed (store MDT likely overloaded)." >&2
    echo "      Model is ready at $MODEL_TARGET — serve with MODEL_PATH=$MODEL_TARGET" >&2
    echo "      Retry the move later: mv $MODEL_TARGET $STORE_TARGET" >&2
  fi
fi
