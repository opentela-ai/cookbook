#!/usr/bin/env bash
# One-shot bring-up: probe gate → submit the serve job.
# NOTE the README banner: DeepSeek-V4-Flash does not generate tokens on this
# image yet (aiter bug) — this runs the bring-up log, not a working deployment.
set -euo pipefail
cd "$(dirname "$0")"

sbatch probe_sglang.sbatch
sbatch "$@" serve_deepseek_v4_flash.sbatch
