#!/usr/bin/env bash
# Probe sglang parser modules from INSIDE the running job's container.
set -e
# find python with sglang
for p in python python3 /usr/local/bin/python3 /opt/conda/bin/python3; do
  if $p -c "import sglang" 2>/dev/null; then PY=$p; break; fi
done
echo "PY=$PY"
$PY - <<'PYEOF'
import os, glob
base = os.path.dirname(os.path.dirname(__import__('sglang').__file__))
sg = os.path.join(base, "sglang", "srt")
print("BASE:", base)
print("=== reasoning_parser modules ===")
for f in sorted(glob.glob(os.path.join(sg, "reasoning_parser", "*.py"))):
    n = os.path.basename(f)[:-3]
    if not n.startswith("__"): print(" ", n)
print("=== managers/tool_call_parser modules ===")
for f in sorted(glob.glob(os.path.join(sg, "managers", "tool_call_parser", "*.py"))):
    n = os.path.basename(f)[:-3]
    if not n.startswith("__"): print(" ", n)
PYEOF
