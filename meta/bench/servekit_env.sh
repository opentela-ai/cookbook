# shellcheck shell=bash
# servekit_env.sh — resolve how to invoke servekit on THIS host.
#
# Sourced by cbench.sh and by recipe bench helpers. Sets the array
# SERVEKIT_RUN (argv prefix, e.g. "${SERVEKIT_RUN[@]}" bench --url ...) or
# exits 1 after printing the exact staging command.
#
# Resolution order:
#   1. $SERVEKIT_DIR — a git checkout of github.com/eth-easl/servekit with
#      src/servekit inside. servekit bench/profile are PURE STDLIB (urllib,
#      re, subprocess), so running module-style needs no pip install and works
#      in offline containers (JSC compute nodes, enroot images): just prepend
#      $SERVEKIT_DIR/src to PYTHONPATH. This is the default everywhere.
#   2. a `servekit` executable on PATH (a real pip/uv install).
#
# Knobs: SERVEKIT_DIR (no default — set it or fall through), PYTHON
# (interpreter for module-style; default python3).
if [ -n "${SERVEKIT_DIR:-}" ] && [ -d "$SERVEKIT_DIR/src/servekit" ]; then
  SERVEKIT_RUN=(env "PYTHONPATH=$SERVEKIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
                "${PYTHON:-python3}" -m servekit.cli)
elif command -v servekit >/dev/null 2>&1; then
  SERVEKIT_RUN=(servekit)
else
  cat >&2 <<EOM
FATAL: servekit not found.
  Stage a checkout once on a node WITH egress (no install needed — stdlib only):
    git clone --depth=1 https://github.com/eth-easl/servekit ${SERVEKIT_DIR:-<shared-fs-path>/servekit}
  then point SERVEKIT_DIR at it, or \`pip install .\` it for a \`servekit\` on PATH.
EOM
  exit 1
fi
