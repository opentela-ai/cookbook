#!/bin/bash
# Download zai-org/GLM-5.3-Flash weights (~306 GiB, FP8 e4m3, 62 safetensors
# shards + tokenizer/config) to $DEPLOY_DIR/models/GLM-5.3-Flash on a JSC
# LOGIN node. Resumable: huggingface_hub skips complete files, so re-run
# after any interruption. JSC's HF bandwidth measured ~2.4 GB/s (full repo
# in ~2 min on 2026-10-07).
#
# Prerequisite (login node, once):
#   pip3 install --user "huggingface_hub[cli]" hf_xet
# The repo is NOT gated — no HF token needed (one is set => higher rate
# limits).
#
# Usage (login node):  bash download_glm_5_3_flash.sh
set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/e/scratch/reformo/yao4/glm-53-flash}"
TARGET="$DEPLOY_DIR/models/GLM-5.3-Flash"

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "FATAL: huggingface_hub missing. Run: pip3 install --user 'huggingface_hub[cli]' hf_xet" >&2
  exit 1
fi

mkdir -p "$TARGET" "$DEPLOY_DIR/cache/hf" "$DEPLOY_DIR/logs"

echo "[$(date -Is)] snapshot_download -> $TARGET"
HF_HOME="$DEPLOY_DIR/cache/hf" python3 - "$TARGET" <<'EOF'
import sys, time
from huggingface_hub import snapshot_download
target = sys.argv[1]
t0 = time.time()
p = snapshot_download(repo_id="zai-org/GLM-5.3-Flash",
                      local_dir=target, max_workers=8)
print(f"DONE {p} in {time.time()-t0:.0f}s")
EOF

echo "--- verify ---"
n=$(ls "$TARGET"/*.safetensors 2>/dev/null | wc -l)
echo "safetensors shards: $n (expect 62)"
[ -f "$TARGET/config.json" ] && echo "config.json OK" || { echo "FATAL: config.json missing" >&2; exit 1; }
du -sh "$TARGET"
