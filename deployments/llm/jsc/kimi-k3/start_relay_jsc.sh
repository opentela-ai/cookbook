#!/bin/bash
# Start the OpenTela relay on a JSC Jupiter login node.
#
# JSC compute nodes have NO outbound internet (not even Cloudflare:443). The
# login node DOES, and it is reachable from compute over the internal 10.x
# fabric. So a compute-side otela worker — which sits next to sglang on the
# serving node and genuinely has no inbound reachability (reachability:
# private) — cannot dial the public head directly; it hops through this relay.
#
# Unlike the Euler recipe, this script starts the relay OUT of the apptainer
# image. Rationale: the promoted SGLang SIF is an aarch64 sglang-only image;
# the relay is a long-lived, cluster-wide, us-managed process that outlives
# every job and must not carry the engine's multi-GB working set. Use a native
# arm64 otela binary, staged once on /e/scratch.
#
# Run this ONCE on a login node, from your workstation clone via `rcc`, or
# directly over SSH. It starts the relay in a tmux session (survives SSH
# disconnects), writes the verified multiaddr to $DEPLOY_DIR/relay.multiaddr
# (the sbatch reads this automatically), and keeps going indefinitely.
#
# CRITICAL (JSC-specific, see serve_llm_otela_jsc.sbatch header) — unlike
# Euler, setting `HOME=...` on the otela command line does NOTHING here:
# otela resolves the home directory via os/user.Current(), which reads
# /etc/passwd (your real /e/home/<user>, NFS-backed) and ignores the $HOME
# env var. On JSC the /etc/passwd home already ends in /jupiter (e.g.
# /e/home/jusers/<user>/jupiter), so otela's go-ds-crdt/badger store lands at
# $PASSWD_HOME/.ocfcore — which goes ESTALE after a few days on NFS, the
# CRDT DAG freezes, and `api.opentela.ai` starts returning 503 "No provider
# found". So this script SYMLINKS $PASSWD_HOME/.ocfcore -> a dir on
# /e/scratch BEFORE starting otela. The relay's peer identity (PeerID) comes
# from --config-dir, already on /e/scratch, so the symlink does not affect it.
#
# Usage:
#   bash start_relay_jsc.sh                 # start (idempotent: reuses if up; per-user ports by default)
#   bash start_relay_jsc.sh attach          # print the running relay's multiaddr WITHOUT starting one
#   bash start_relay_jsc.sh status          # is the relay running?
#   bash start_relay_jsc.sh stop            # stop the relay (refuses if another uid owns the port)
#   bash start_relay_jsc.sh multiaddr       # print the saved relay multiaddr
set -euo pipefail

# ---------------------------------------------------------------- deployment --
# Match the sbatch defaults so a fresh checkout works without exports. Override
# either to relocate the relay's data (e.g. when a second operator wants an
# independent relay with its own peer ID).
PROJECT="${PROJECT:-reformo}"
DEPLOY_DIR="${DEPLOY_DIR:-/e/scratch/$PROJECT/$USER/otela-relay}"
BINARY="${OTELA_BIN:-$DEPLOY_DIR/bin/opentela}"
# v0.2.4 (latest release, 2026-08-25) contains the libp2p self-dial fix
# (commit 7f421838c5, merged before v0.2.3), so the official arm64 binary
# works as a relay out of the box — no locally-patched build needed (the
# older two-hop recipe here predated the fix and pinned v0.2.2 + relayfix).
# NB the v0.2.x asset is named `opentela-arm64`, NOT `otela-arm64` (which
# 404s) or `ocf-arm64` (<= v0.1.11); `releases/latest` drifts, so this pins
# an explicit tag. Stage once on /e/scratch (login nodes have egress;
# compute nodes do not).
OTELA_URL="${OTELA_URL:-https://github.com/eth-easl/OpenTela/releases/download/v0.2.4/opentela-arm64}"
# The tmux session is per-UID by name too. tmux sockets are already
# per-uid (one user cannot see or kill another's session), but a distinct
# name keeps `tmux ls` and the stop logic unambiguous when several relays
# run on one login node.
SESSION="${RELAY_SESSION:-jsc-otela-relay-$(id -un)}"

# Relay identity knobs. SEED gives a DETERMINISTIC peer ID (see the
# use-opentela skill: `--seed 0` -> fixed peer, `--seed` unset -> new peer
# every start). Keep SEED stable across restarts so workers'
# relay.multiaddr does not need re-writing.
SEED="${RELAY_SEED:-0}"
# The relay bootstraps onto the OpenTela DHT through this peer so it can
# register as a relay candidate. The public head (p2p.opentela.ai /
# QmTtnXKHvovC...) is the proven default; override RELAY_BOOTSTRAP with any
# multiaddr / HTTP dnt source `otela start --bootstrap.static` accepts.
BOOTSTRAP="${RELAY_BOOTSTRAP:-/dns4/p2p.opentela.ai/tcp/443/wss/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7}"
#
# PORTS ARE PER-USER BY DEFAULT. Two operators on the same login node would
# otherwise both try to bind the fixed libp2p tcp/udp ports and the HTTP API
# port; the second `otela start` dies with EADDRINUSE, never publishes a peer
# ID, and its sbatch silently latches onto the first user's relay (or the
# baked-in OTELA_RELAY_DEFAULT) -- a correctness hazard, since
# api.opentela.ai then has two providers for one model under two operator
# peer IDs. The offsets (45000/59000 + `id -u` % 1000) match the GLM-5.3
# campaign guide; `id -u` is the right per-user key on JSC (a site-local
# integer), NOT `$USER` which can repeat across sites. Set the ports
# explicitly only to run a deliberate second relay on the same node.
RELAY_TCP_PORT="${RELAY_TCP_PORT:-$((45000 + $(id -u) % 1000))}"
RELAY_UDP_PORT="${RELAY_UDP_PORT:-$((59000 + $(id -u) % 1000))}"
RELAY_API_PORT="${RELAY_API_PORT:-$((18000 + $(id -u) % 1000))}"

mkdir -p "$DEPLOY_DIR" "$DEPLOY_DIR/logs" "$DEPLOY_DIR/cfg"

LOGFILE="$DEPLOY_DIR/logs/relay-$(date +%s).log"
LATEST_LOG="$DEPLOY_DIR/logs/relay-latest.log"
CFG_DIR="$DEPLOY_DIR/cfg"
CFG="$CFG_DIR/cfg.yaml"
MULTIADDR_FILE="$DEPLOY_DIR/relay.multiaddr"

# ----------------------------------------------------------------- helpers ----
echo_ts() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

resolve_binary() {
  if [ -x "$BINARY" ]; then return 0; fi
  echo_ts "otela binary missing at $BINARY; trying to fetch (login node has egress)"
  if curl -fL --max-time 60 "$OTELA_URL" -o "$BINARY" && chmod +x "$BINARY"; then
    echo_ts "otela fetched -> $BINARY"
  else
    rm -f "$BINARY"
    cat >&2 <<EOM
FATAL: otela binary not found at $BINARY and fetch failed.
       Stage it once on this login node:

         mkdir -p "$(dirname "$BINARY")"
         curl -fL "$OTELA_URL" -o "$BINARY" && chmod +x "$BINARY"

       Override the URL/path with \$OTELA_URL / \$OTELA_BIN.
EOM
    exit 1
  fi
}

# Detect this login node's primary IP on ib0 (the fabric compute nodes reach).
# The sbatch's default relay is 10.128.1.1; a relay started here will be on
# whichever login node SSH/rcc landed on, so advertise THAT address.
NODE_IP="${RELAY_PUBLIC_ADDR:-}"
if [ -z "$NODE_IP" ]; then
  NODE_IP=$(ip -o -4 addr show ib0 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
  [ -z "$NODE_IP" ] && NODE_IP=$(ip -o -4 addr show | awk '$2!="lo"{print $4}' | cut -d/ -f1 | head -1)
fi
[ -n "$NODE_IP" ] || { echo "FATAL: could not detect a non-loopback IPv4 on ib0" >&2; exit 1; }
echo_ts "login node: $(hostname)  ib0: $NODE_IP"

write_cfg() {
  # cfg.yaml lives on /e/scratch (NOT /e/home), so the peer identity (from
  # --config-dir) is stable and never hits an NFS stale handle. otela start
  # --config-dir loads the file NAMED cfg.yaml (verified live); older
  # versions of this script wrote relay.cfg.yaml, which otela ignored unless
  # cfg.yaml was ALSO present (fragile merge). Remove the legacy name so
  # exactly one config -- ours -- is authoritative.
  rm -f "$CFG_DIR/relay.cfg.yaml"
  cat > "$CFG" <<YAML
name: jsc-relay
seed: "$SEED"
port: "$RELAY_API_PORT"
tcpport: "$RELAY_TCP_PORT"
udpport: "$RELAY_UDP_PORT"
# 'mode: node' + 'role: relay' + 'reachability: public' is the proven relay
# config (validated live on JSC with v0.2.4); 'mode: full' was wrong here.
# The relay does NOT advertise public-addr (the "won't be discoverable as a
# bootstrap" log line is benign -- workers dial the explicit multiaddr this
# script writes to relay.multiaddr via --bootstrap.static), and the
# bootstrap source is passed on the CLI, not as a cfg field.
mode: node
role: relay
reachability: public
loglevel: debug
cleanslate: false
security:
  require_signed_binary: false
solana:
  skip_verification: true
YAML
}

# Move otela's BadgerDB (resolved via /etc/passwd -> $PASSWD_HOME/.ocfcore,
# NFS-backed) onto /e/scratch via symlink. os/user.Current() ignores $HOME,
# so an env override does NOT work on JSC (verified live: `otela start
# --config-dir <scratch>` still writes BadgerDB at $PASSWD_HOME/.ocfcore,
# not under the config dir). On JSC the passwd home already ends in /jupiter,
# so the store is $PASSWD_HOME/.ocfcore (single) — NOT $PASSWD_HOME/jupiter/.
relocate_ocfcore() {
  local passwd_home ocfcore
  # otela writes to "$PASSWD_HOME/.ocfcore" where $PASSWD_HOME is the
  # /etc/passwd home, not the $HOME env var. Resolve it the same way.
  passwd_home=$(getent passwd "$USER" | cut -d: -f6)
  [ -n "$passwd_home" ] || { echo "FATAL: no /etc/passwd home for $USER" >&2; exit 1; }
  ocfcore="$passwd_home/.ocfcore"
  local scratch_store="$DEPLOY_DIR/ocfcore"
  mkdir -p "$scratch_store" "$(dirname "$ocfcore")"
  if [ -e "$ocfcore" ] && [ ! -L "$ocfcore" ]; then
    # A real directory already exists (e.g. from a relay started without this
    # script). Move it onto /e/scratch and symlink, preserving any data.
    echo_ts "moving existing $ocfcore -> $scratch_store and symlinking"
    if [ -d "$scratch_store" ] && [ "$(ls -A "$scratch_store" 2>/dev/null)" ]; then
      echo_ts "WARN: $scratch_store already populated; leaving $ocfcore in place." >&2
      echo_ts "      If that is an NFS-backed real dir, stop the relay and either" >&2
      echo_ts "      \`rm -rf $ocfcore\` (data is ephemeral CRDT state) or move it" >&2
      echo_ts "      aside before rerunning." >&2
      return 0
    fi
    mv "$ocfcore" "$scratch_store"
  fi
  if [ ! -e "$ocfcore" ]; then
    echo_ts "symlinking $ocfcore -> $scratch_store (BadgerDB off NFS)"
    ln -s "$scratch_store" "$ocfcore"
  elif [ -L "$ocfcore" ]; then
    echo_ts "$ocfcore already a symlink -> $(readlink -f "$ocfcore")"
  fi
}

# Wait for the relay to publish its peer ID, then write the multiaddr the
# sbatch will dial. Matches the form in start_relay_euler.sh.
capture_multiaddr() {
  local log="$1"
  echo_ts "waiting for relay to publish its peer ID ..."
  local peer
  for _ in $(seq 1 30); do
    sleep 2
    # The peer ID is a base-58 libp2p key (Qm... 46 chars).
    peer=$(grep -oE 'Qm[1-9A-HJ-NP-Za-km-z]{44}' "$log" 2>/dev/null | head -1)
    if [ -n "$peer" ]; then
      local ma="/ip4/${NODE_IP}/tcp/${RELAY_TCP_PORT}/p2p/${peer}"
      echo "$ma" > "$MULTIADDR_FILE"
      ln -sf "$log" "$LATEST_LOG"
      echo_ts "=== RELAY UP ==="
      echo_ts "  node:      $(hostname) ($NODE_IP)"
      echo_ts "  peer ID:   $peer"
      echo_ts "  multiaddr: $ma   (saved to $MULTIADDR_FILE)"
      echo_ts "  config:    $CFG"
      echo_ts "  log:       $LATEST_LOG"
      echo ""
      echo "The serving sbatch reads $MULTIADDR_FILE automatically when"
      echo "OTELA_RELAY_ADDR is unset. Submit with:"
      echo "  sbatch deployments/llm/jsc/kimi-k3/serve_llm_otela_jsc.sbatch"
      return 0
    fi
  done
  echo_ts "WARN: relay did not publish a peer ID within 60 s." >&2
  ln -sf "$log" "$LATEST_LOG"
  tail -20 "$log" 2>/dev/null || true
  return 1
}

# Who is listening on $1 (tcp)? Prints the owning uid, or empty. We use this
# for two guards: (a) refuse `start` when the port is already ours from a
# stray process or another user's relay, and (b) refuse `stop` when another
# user owns the libp2p port -- that relay is not ours to stop, and SIGKILLing
# it would strand a stale `connected: true` row for every worker dialing it.
#
# `ss -p` prints `users:(("otela",pid=N,fd=M))` WITHOUT a uid field, and
# WITHOUT ROOT it shows the LISTEN line but HIDES the process info for other
# users' sockets. So: if we get a pid it's provably ours (resolve pid->uid
# via `ps`, which works for any pid); if the port is listening but no pid is
# visible, that's another user's relay and we return "" -- the guards then
# refuse conservatively rather than race an otela onto a busy port.
port_listening() {
  ss -ltnH "sport = :$1" 2>/dev/null | grep -q .
}

port_owner_uid() {
  local port="$1" pid
  pid=$(ss -ltnpH "sport = :$port" 2>/dev/null \
        | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  [ -n "$pid" ] || return 0
  ps -o uid= -p "$pid" 2>/dev/null | tr -d ' '
}

my_uid="$(id -u)"

# ------------------------------------------------------------------ main ------
case "${1:-start}" in
  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo_ts "relay RUNNING in tmux session '$SESSION' on $(hostname) ($NODE_IP)"
      echo_ts "  config:    $CFG"
      echo_ts "  log:       $LATEST_LOG"
      echo_ts "  multiaddr: $(cat "$MULTIADDR_FILE" 2>/dev/null || echo '(none)')"
    else
      echo_ts "relay NOT RUNNING"
      exit 1
    fi
    ;;

  stop)
    # Guard: never stop another user's relay. tmux sockets are per-uid so
    # `kill-session` would no-op anyway, but if the libp2p port is owned by
    # a different uid we say so LOUDLY instead of exiting 0 -- a silent
    # "stopped" here would make the next `start` look safe while the shared
    # relay is still serving other workers.
    if port_listening "$RELAY_TCP_PORT"; then
      owner="$(port_owner_uid "$RELAY_TCP_PORT" || true)"
      if [ -n "$owner" ] && [ "$owner" != "$my_uid" ]; then
        echo_ts "REFUSING: libp2p port $RELAY_TCP_PORT is owned by uid $owner, not you ($my_uid)." >&2
        echo_ts "That relay may be serving other workers; it is not yours to stop." >&2
        echo_ts "If you started it, set RELAY_TCP_PORT to match and retry; otherwise" >&2
        echo_ts "coordinate with that operator before stopping the shared relay." >&2
        exit 1
      fi
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      # C-c over tmux -> SIGINT -> otela announces LEFT cleanly (AnnounceLeave
      # -> 5 s CRDT drain -> srv.Shutdown). Killing the session directly would
      # SIGKILL and the head keeps a stale `connected: true` row.
      tmux send-keys -t "$SESSION" C-c 2>/dev/null || true
      sleep 5
      tmux kill-session -t "$SESSION" 2>/dev/null || true
      echo_ts "relay stopped (clean AnnounceLeave sent)"
    else
      echo_ts "relay not running (no tmux session '$SESSION')"
    fi
    ;;

  attach)
    # Print the running relay's multiaddr WITHOUT starting a new one. This is
    # the path a second operator on the same login node should take: reuse
    # the already-running relay (copy its multiaddr to YOUR
    # \$DEPLOY_DIR/relay.multiaddr, or set OTELA_RELAY_ADDR) instead of
    # `start`-ing a second one that would lose the port race or fragment the
    # model's providers under two peer IDs.
    if tmux has-session -t "$SESSION" 2>/dev/null && [ -s "$MULTIADDR_FILE" ]; then
      cat "$MULTIADDR_FILE"
      echo_ts "(running as session '$SESSION'; copy to your RELAY_ADDR_FILE or set OTELA_RELAY_ADDR)" >&2
    elif port_listening "$RELAY_TCP_PORT"; then
      owner="$(port_owner_uid "$RELAY_TCP_PORT" || true)"
      echo_ts "port $RELAY_TCP_PORT is held by ${owner:-another user (uid hidden without root)};" >&2
      echo_ts "  that relay's relay.multiaddr is not visible to you. Ask the operator" >&2
      echo_ts "  for the multiaddr and set OTELA_RELAY_ADDR, or run your own on a" >&2
      echo_ts "  free port: RELAY_TCP_PORT=... RELAY_UDP_PORT=... bash $0 start" >&2
      exit 1
    else
      echo_ts "no running relay for this user; run 'start' first" >&2
      exit 1
    fi
    ;;

  multiaddr)
    [ -s "$MULTIADDR_FILE" ] || { echo_ts "no multiaddr yet (run start first)" >&2; exit 1; }
    cat "$MULTIADDR_FILE"
    ;;

  start|"")
    # REUSE before START. If this user already has a relay up and has
    # published a multiaddr, just re-print it -- a second `otela start` on
    # the same ports would fail EADDRINUSE and (worse) clobber
    # relay.multiaddr with a never-published file. Makes `start` idempotent
    # for the common "I forgot whether I started it" case.
    if tmux has-session -t "$SESSION" 2>/dev/null && [ -s "$MULTIADDR_FILE" ]; then
      echo_ts "relay already running as session '$SESSION'; reusing:"
      echo_ts "  multiaddr: $(cat "$MULTIADDR_FILE")"
      exit 0
    fi

    resolve_binary
    relocate_ocfcore
    # otela init seeds the peer keypair in $CFG_DIR/keys/ (on /e/scratch, so
    # the peer ID is stable across restarts) plus a Solana wallet in
    # $HOME/.config/opentela (tiny; NFS is fine), and idempotently writes a
    # default cfg.yaml if none exists. Verified live: `otela start
    # --config-dir` loads the file NAMED `cfg.yaml` (not relay.cfg.yaml), so
    # init MUST run BEFORE write_cfg and write_cfg MUST target cfg.yaml --
    # then exactly one authoritative config (ours) is present, with no
    # dependence on a fragile multi-file merge.
    if [ ! -s "$CFG_DIR/keys/id" ]; then
      echo_ts "creating relay peer keypair via 'otela init --config-dir $CFG_DIR'"
      "$BINARY" init --config-dir "$CFG_DIR" >/dev/null 2>&1 \
        || { echo_ts "FATAL: otela init failed (see $CFG_DIR)" >&2; exit 1; }
    fi
    write_cfg

    # Port-busy guard. If something is already listening on RELAY_TCP_PORT
    # and it is NOT our tmux session (caught above), refuse instead of
    # launching an otela that dies mid-start and leaves a confusing 60 s
    # capture timeout. Two cases: (a) another user's relay -- they own the
    # port, `attach` instead; (b) your own stray otela from a crashed tmux
    # -- the pkill below reaps it.
    if port_listening "$RELAY_TCP_PORT"; then
      owner="$(port_owner_uid "$RELAY_TCP_PORT" || true)"
      if [ -z "$owner" ]; then
        # Without root, `ss -p` hides other users' pids/uids but still shows
        # the LISTEN line. A busy port we can't attribute to ourselves is
        # conservatively another user's relay -- refuse rather than race an
        # otela that will die on EADDRINUSE.
        echo_ts "REFUSING: libp2p port $RELAY_TCP_PORT is busy (owner hidden without root)." >&2
        echo_ts "That is likely another operator's relay on this login node. Either:" >&2
        echo_ts "  - reuse it: get their multiaddr and set OTELA_RELAY_ADDR (no start), or" >&2
        echo_ts "  - run your own on a free port: RELAY_TCP_PORT=... RELAY_UDP_PORT=... bash $0 start" >&2
        exit 1
      elif [ "$owner" != "$my_uid" ]; then
        echo_ts "REFUSING: libp2p port $RELAY_TCP_PORT already held by uid $owner." >&2
        echo_ts "That is likely another operator's relay on this login node. Either:" >&2
        echo_ts "  - reuse it: get their multiaddr and set OTELA_RELAY_ADDR (no start), or" >&2
        echo_ts "  - run your own on a free port: RELAY_TCP_PORT=... RELAY_UDP_PORT=... bash $0 start" >&2
        exit 1
      fi
      echo_ts "WARN: port $RELAY_TCP_PORT held by your own stray process; killing it first"
    fi

    # Kill any stale relay by config-dir, then a fresh tmux session.
    pkill -f "otela start --config-dir.*$CFG_DIR" 2>/dev/null || true
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    sleep 1

    echo_ts "starting relay in tmux session '$SESSION' (config-dir $CFG_DIR)"
    echo_ts "  libp2p tcp=$RELAY_TCP_PORT udp=$RELAY_UDP_PORT  api=$RELAY_API_PORT on $NODE_IP"
    echo_ts "  bootstrap: $BOOTSTRAP"
    # NOTE: we do NOT set HOME= here. On JSC otela resolves the home directory
    # via /etc/passwd (os/user.Current), ignoring $HOME. The BadgerDB path
    # is relocated to /e/scratch by relocate_ocfcore instead. --config-dir
    # puts the peer keypair + cfg on /e/scratch; the bootstrap source and
    # solana skip are passed on the CLI (matches the proven two-hop recipe).
    tmux new-session -d -s "$SESSION" \
      "'$BINARY' start --config-dir '$CFG_DIR' --bootstrap.static '$BOOTSTRAP' --solana.skip_verification 2>&1 | tee '$LOGFILE'"

    capture_multiaddr "$LOGFILE" || exit 1
    ;;

  *)
    echo "usage: $0 {start|attach|stop|status|multiaddr}" >&2
    echo "  start     start the relay (idempotent: reuses if already up)" >&2
    echo "  attach    print the running relay's multiaddr without starting one" >&2
    echo "  stop      stop the relay (refuses if another uid owns the port)" >&2
    echo "  status    is the relay running?" >&2
    echo "  multiaddr print the saved multiaddr" >&2
    exit 2
    ;;
esac
