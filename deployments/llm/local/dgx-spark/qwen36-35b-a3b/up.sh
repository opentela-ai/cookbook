#!/usr/bin/env bash
# One-shot bring-up: build the overlay image if missing → start the engine →
# register the otela sidecar. Skips steps whose output already exists.
# Knobs, prerequisites and troubleshooting: README.md
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-s3er-qwen36-dgx-spark}"
docker image inspect "$IMAGE" >/dev/null 2>&1 || bash build_image.sh

# Stage the latest official OpenTela (arm64 — DGX Spark) under run/otela;
# rerun-safe: skips if the staged release is already current, FORCE=1 re-downloads.
OTELA_DIR="${OTELA_DIR:-$PWD/run/otela}" bash ../../../../../meta/common/download_otela.sh
bash serve_qwen36_dgx_spark.sh        # waits for READY, writes run/last_service.env
bash register_qwen36_otela.sh daemon
