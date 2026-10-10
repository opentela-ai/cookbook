#!/usr/bin/env bash
# One-shot bring-up: ensure the login-node relay is running, then submit the
# serving job. One-time prerequisites (model, SIF image, otela binary) are
# documented in README.md → Prerequisites; knobs: README.md → Knobs.
set -euo pipefail
cd "$(dirname "$0")"

bash start_relay_euler.sh status >/dev/null 2>&1 || bash start_relay_euler.sh
sbatch serve_qwen36_otela_euler.sbatch
