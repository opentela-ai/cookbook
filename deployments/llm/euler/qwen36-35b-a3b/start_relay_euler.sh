#!/bin/bash
# Start the OpenTela relay node on an Euler login node.
#
# Euler compute nodes have NO outbound raw-TCP internet (eth_proxy gives
# HTTP(S) only). The login node DOES. So a compute-side otela worker cannot
# reach the public head directly; it hops through this relay. See the
# sbatch header for the full topology diagram.
#
# Run this ONCE on a login node (any of them). It starts the relay in a tmux
# session so it survives SSH disconnects, writes the relay's verified
# multiaddr to $DEPLOY_DIR/relay.multiaddr (the sbatch reads this), and
# keeps going indefinitely.
#
# Usage:
#   bash start_relay_euler.sh              # start (or restart) the relay
#   bash start_relay_euler.sh status       # check if relay is running
#   bash start_relay_euler.sh stop         # stop the relay

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/cluster/scratch/$USER/otela-qwen36}"
BINARY="${OTELA_BIN:-$DEPLOY_DIR/bin/otela-v0.2.3}"
CFG="${RELAY_CFG:-$HOME/opentela/relay.cfg.yaml}"
SESSION="opentela-relay"
LOGFILE="$DEPLOY_DIR/logs/relay-$(date +%s).log"
LATEST_LOG="$DEPLOY_DIR/logs/relay-latest.log"

mkdir -p "$DEPLOY_DIR/logs"
[ -x "$BINARY" ] || { echo "FATAL: otela binary not found: $BINARY" >&2; exit 1; }
[ -f "$CFG" ]   || { echo "FATAL: relay config not found: $CFG" >&2
  echo "       A relay cfg.yaml (role: relay, tcpport, udpport, bootstrap)" >&2
  echo "       must exist. See the README for the expected shape." >&2
  exit 1; }

# Detect the primary IP of this login node (first non-loopback IPv4).
# The round-robin DNS (euler.ethz.ch) may not resolve to this node, so we
# must advertise the actual IP for compute-node connectivity.
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Login node: $(hostname)  IP: $NODE_IP"

# Patch public-addr in config to this node's actual IP (the IP changes
# depending on which login node SSH landed on).
sed -i "s/^public-addr:.*/public-addr: \"${NODE_IP}\"/" "$CFG"
echo "Patched public-addr -> $NODE_IP"

RELAY_HOME="/tmp/opentela-relay"
# BadgerDB goes to $HOME/.ocfcore/ (Euler binary resolves the $HOME env var).
# Use /tmp (node-local, not Lustre) to avoid both home quota and stale file
# handles after days of uptime.

case "${1:-start}" in
  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "relay RUNNING in tmux session '$SESSION' on $(hostname) ($NODE_IP)"
      echo "  config:  $CFG"
      echo "  log:     $LATEST_LOG"
      echo "  multiaddr (saved): $(cat "$DEPLOY_DIR/relay.multiaddr" 2>/dev/null || echo '(none)')"
    else
      echo "relay NOT RUNNING"
      exit 1
    fi
    ;;

  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      # Send SIGTERM via tmux so otela announces LEFT cleanly.
      tmux send-keys -t "$SESSION" C-c 2>/dev/null || true
      sleep 3
      tmux kill-session -t "$SESSION" 2>/dev/null || true
      echo "relay stopped"
    else
      echo "relay not running"
    fi
    ;;

  start|"")
    # Kill any stale relay (by config path) before starting fresh.
    pkill -f "otela.*start.*relay.cfg.yaml" 2>/dev/null || true
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    sleep 1

    rm -rf "$RELAY_HOME"
    mkdir -p "$RELAY_HOME"

    echo "Starting relay in tmux session '$SESSION' ..."
    tmux new-session -d -s "$SESSION" \
      "env HOME='$RELAY_HOME' '$BINARY' start --config '$CFG' 2>&1 | tee '$LOGFILE'"

    # Wait for the relay to come up and extract its peer ID + multiaddr.
    echo "Waiting for relay to come up ..."
    for i in $(seq 1 30); do
      sleep 2
      if grep -qE "Connected to peer:|merged delta|Updating peer" "$LOGFILE" 2>/dev/null; then
        # Extract peer ID from the key generation or CRDT path.
        PEER_ID=$(grep -oE 'Qm[1-9A-HJ-NP-Za-km-z]{44}' "$LOGFILE" 2>/dev/null | head -1)
        if [ -n "$PEER_ID" ]; then
          MULTIADDR="/ip4/${NODE_IP}/tcp/$(grep -E '^tcpport:' "$CFG" | awk '{print $2}' | tr -d '\"')/p2p/${PEER_ID}"
          echo "$MULTIADDR" > "$DEPLOY_DIR/relay.multiaddr"
          ln -sf "$LOGFILE" "$LATEST_LOG"
          echo ""
          echo "=== RELAY UP ==="
          echo "  node:      $(hostname) ($NODE_IP)"
          echo "  peer ID:   $PEER_ID"
          echo "  multiaddr: $MULTIADDR"
          echo "  config:    $CFG"
          echo "  log:       $LATEST_LOG"
          echo ""
          echo "The sbatch reads $DEPLOY_DIR/relay.multiaddr automatically."
          echo "Submit the serving job with:"
          echo "  sbatch serve_qwen36_otela_euler.sbatch"
          exit 0
        fi
      fi
    done

    echo "WARN: relay did not produce a peer ID within 60 s." >&2
    echo "     Check the log: $LOGFILE" >&2
    ln -sf "$LOGFILE" "$LATEST_LOG"
    tail -20 "$LOGFILE" 2>/dev/null || true
    exit 1
    ;;

  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
