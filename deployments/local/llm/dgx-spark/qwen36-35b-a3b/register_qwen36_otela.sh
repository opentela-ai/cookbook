#!/usr/bin/env bash
# Register the running Qwen3.6-35B-A3B-FP8 engine as the `llm` service on the
# OpenTela network (swiss-ai/OpenTela sai-v0.0.6 / arm64). Sidecar topology:
# the engine (sglang in the qwen36-dgx-spark container, --network host) is
# already serving on :${SERVE_PORT}. This script launches a standalone otela
# peer that connects to the given bootstrap and publishes the engine's `llm`
# service + model id into the shared CRDT. It does NOT supervise the engine —
# if the engine restarts on a different port, re-run this script with the new
# port (the registrar re-polls /health and drops unhealthy peers).
#
#   engine container (qwen36-dgx-spark, --network host)  :${SERVE_PORT}  <-- /health = 200
#        ^
#        | 127.0.0.1:${SERVE_PORT}  (host netns == container netns)
#        |
#   otela sidecar (this script)   :43905 libp2p  -->  bootstrap peer
#
# Usage:
#   bash register_qwen36_otela.sh          # start (foreground)
#   bash register_qwen36_otela.sh daemon   # start (background, logs to file)
#   bash register_qwen36_otela.sh stop     # stop the daemon
#   bash register_qwen36_otela.sh status   # check the daemon
set -uo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="${DEPLOY_DIR:-${RECIPE_DIR}/run}"

# Defaults keep the recipe self-contained: the otela binary, config, pid, and
# log all live under $OTELA_DIR (default $DEPLOY_DIR/otela, i.e. a `run/otela`
# dir next to the recipe). Drop the OpenTela `otela` binary at $OTELA_BIN.
OTELA_DIR="${OTELA_DIR:-${DEPLOY_DIR}/otela}"
OTELA_BIN="${OTELA_BIN:-${OTELA_DIR}/otela}"
CFG_DIR="${OPENTELA_CFG_DIR:-${OTELA_DIR}}"
PIDFILE="${PIDFILE:-${OTELA_DIR}/otela.pid}"
LOGFILE="${LOGFILE:-${OTELA_DIR}/otela.log}"

# User-supplied bootstrap (public relay / head peer).
BOOTSTRAP="${OPENTELA_BOOTSTRAP:-/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7}"
SERVICE_PORT="${SERVE_PORT:-30000}"
SERVED_MODEL_ID="${SERVED_MODEL_ID:-Qwen3.6-35B-A3B-FP8}"
# WHY: otela derives its libp2p peer ID from --seed (sai-v0.0.6 has no stored
# identity key). A graceful `stop` announces LEFT, and if you restart with the
# SAME seed (the binary's default "0") the gateway keeps a LEFT tombstone that
# a fresh-CRDT restart (DAG head 0) cannot override — re-registration stalls
# for many minutes and the gateway returns `{"error":"No provider found for the
# requested service."}` (HTTP 503) to /v1/service/llm/... in the meantime.
# Defaulting to $$ gives a fresh peer ID per invocation, so stop/start cycles
# register cleanly in ~60 s. Set OPENTELA_SEED to a constant ONLY if you want a
# stable peer ID and accept that you must not stop-and-immediately-restart with
# that seed (or wait out the tombstone).
OPENTELA_SEED="${OPENTELA_SEED:-$$}"

[ -x "${OTELA_BIN}" ] || { echo "FATAL: otela binary not found/executable: ${OTELA_BIN}" >&2
  echo "       Obtain the OpenTela 'otela' binary (sai-v0.0.6, arm64) and place it" >&2
  echo "       at ${OTELA_BIN}, or set OTELA_BIN to an existing path." >&2
  exit 1; }
mkdir -p "${CFG_DIR}"
# Idempotent: create the OpenTela config + identity if absent (matches the
# Slurm recipe pattern). In sai-v0.0.6 this writes nothing and `otela start`
# generates an ephemeral identity from --seed (default 0, deterministic); kept
# for forward-compat and so a stale/missing cfg.yaml never fails start.
"${OTELA_BIN}" init --config "${CFG_DIR}/cfg.yaml" >/dev/null 2>&1 || true

CMD=(
  "${OTELA_BIN}" start
  --config "${CFG_DIR}/cfg.yaml"
  --mode node
  --service.name "${OPENTELA_SERVICE_NAME:-llm}"
  --service.port "${SERVICE_PORT}"
  --bootstrap.static "${BOOTSTRAP}"
  --solana.skip_verification
  --label "model=${SERVED_MODEL_ID}"
  --label "cluster=dgx-spark"
  --label "host=$(hostname)"
  --label "launched_by=${USER}"
  --seed "${OPENTELA_SEED}"
  --tcpport "${OPENTELA_TCP_PORT:-43905}"
  --udpport "${OPENTELA_UDP_PORT:-59820}"
)

case "${1:-start}" in
  stop)
    if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      # SIGTERM, never SIGKILL: a killed peer stays connected:true in the
      # registry and the head keeps round-robining traffic into a dead endpoint.
      kill "$(cat "${PIDFILE}")" && echo "Stopped otela (pid $(cat "${PIDFILE}"))"
      rm -f "${PIDFILE}"
    else
      echo "otela not running"
    fi
    exit 0
    ;;
  status)
    if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      echo "otela RUNNING (pid $(cat "${PIDFILE}"))"
      echo "  tcp listen :$(grep -oP '(?<=--tcpport )\S+' "${LOGFILE}" 2>/dev/null | tail -1 || echo 43905)"
      echo "  logs:      ${LOGFILE}"
      echo "  config:    ${CFG_DIR}/cfg.yaml"
    else
      echo "otela NOT RUNNING"
      exit 1
    fi
    exit 0
    ;;
  daemon)
    if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      echo "otela already running (pid $(cat "${PIDFILE}"))"; exit 0
    fi
    # otela exits on stdin EOF — redirect from /dev/null (foreground `start`
    # below relies on the same thing).
    echo "Starting otela sidecar (logs -> ${LOGFILE})..."
    setsid "${CMD[@]}" </dev/null >"${LOGFILE}" 2>&1 &
    echo $! > "${PIDFILE}"
    sleep 3
    if kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      echo "otela started (pid $(cat "${PIDFILE}"))"
      echo "  bootstrap: ${BOOTSTRAP}"
      echo "  service:   llm @ :${SERVICE_PORT} (model=${SERVED_MODEL_ID})"
    else
      echo "otela FAILED to start — check ${LOGFILE}"; exit 1
    fi
    ;;
  start|"")
    # foreground — useful for debugging.
    exec "${CMD[@]}" </dev/null
    ;;
  *)
    echo "usage: $0 {start|daemon|stop|status}" >&2; exit 2
    ;;
esac
