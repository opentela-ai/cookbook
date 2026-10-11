#!/usr/bin/env bash
# One-shot bring-up through JSC (mirrors README "Submit").
#   bash up.sh                                                       # real run, TP4 on 1 node
#   bash up.sh --export=ALL,LOAD_FORMAT=dummy,GEN_CORRECTNESS_SMOKE=1,HEALTH_TIMEOUT=1800   # fast smoke
#   bash up.sh --nodes=2                                             # TP4 x PP2 (TODO(unverified))
# Prerequisites (image build + login-node relay, both idempotent): README "Start here".
set -euo pipefail
cd "$(dirname "$0")"
export PROJECT="${PROJECT:-reformo}"
IMAGE_DEFAULT="/e/scratch/$PROJECT/$USER/glm-53-flash/images/glm-5.3-flash.sif"
[ -f "${IMAGE:-$IMAGE_DEFAULT}" ] || bash build_glm_5_3_flash_image.sh
# Login-node relay is shared by all JSC recipes (deployed copy of
# deployments/llm/jsc/kimi-k3/start_relay_jsc.sh); idempotent.
RELAY="${RELAY:-/e/scratch/$PROJECT/$USER/otela-relay/start_relay_jsc.sh}"
if [ -f "$RELAY" ]; then bash "$RELAY"; else echo "NOTE: relay script not found at $RELAY (already running?); continuing" >&2; fi
# Stage the latest official OpenTela (arm64) on /e/scratch — best effort: a
# stale staged binary still serves; a missing one FATALs in the sbatch preflight.
OTELA_DIR="${OTELA_DIR:-/e/scratch/$PROJECT/$USER/otela-llm}" OTELA_ARCH=arm64 \
  bash ../../../../meta/common/download_otela.sh \
  || echo "[up.sh] WARN: OpenTela latest-check failed (no egress?); submitting with the staged binary"
sbatch "$@" serve_glm_53_flash_otela_jsc.sbatch
