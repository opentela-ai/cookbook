#!/usr/bin/env bash
# One-shot bring-up: submit the serving job (the image comes from the EDF and
# is pulled on first run — pre-warm command in README.md → Submit).
set -euo pipefail
cd "$(dirname "$0")"
sbatch "$@" serve_glm_47_flash_sglang.sbatch
