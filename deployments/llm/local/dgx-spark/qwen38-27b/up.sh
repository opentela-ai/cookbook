#!/usr/bin/env bash
# One-shot bring-up: start the engine on the shared qwen36 overlay image (this
# recipe has no build step) → register the otela sidecar. Knobs: README.md
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-s3er-qwen36-dgx-spark}"
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "FATAL: image $IMAGE not found — build it once via ../qwen36-35b-a3b/build_image.sh" >&2
  exit 1
}

bash serve_qwen38_dgx_spark.sh        # waits for READY, writes run/last_service.env
bash register_qwen38_otela.sh daemon
