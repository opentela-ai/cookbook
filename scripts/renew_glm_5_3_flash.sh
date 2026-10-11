#!/usr/bin/env bash
# =============================================================================
# renew_glm_5_3_flash.sh — keep GLM-5.3-Flash served on CSCS Clariden
# -----------------------------------------------------------------------------
# Repeats the rolling-submit / handover process used to keep the
# zai-org/GLM-5.3-Flash service continuously up across 12h Slurm windows on
# clariden, registered on the OpenTela mesh.
#
# What one invocation does (default):
#   1. prechecks your CSCS SSH cert (offers to renew it via `cscs-key sign
#      --headless` if expired, printing the device code)
#   2. looks at your current glm53-otela-clariden jobs (running? pending? how
#      much walltime is left?)
#   3. submits a fresh 12h job ONLY when a handover is warranted — never a
#      wasteful duplicate (override with --force)
#   4. prints status: current jobs, cluster pressure, and (if a job is
#      RUNNING) engine /health + live metrics + OpenTela registration
#
# Modes:
#   (default)        decide + submit-if-needed + status
#   --status         just print status (no submit)
#   --watch          submit-if-needed, then block until the new job is
#                    RUNNING + healthy + registered (bounded by --max-wait)
#   --handover       like --watch, but once the new job is confirmed live it
#                    scancels the expiring running job for a zero-gap swap
#   --force          submit even when a job already covers the window
#   --update-otela   push meta/common/download_otela.sh to clariden and run it
#                    there to stage the LATEST official OpenTela release
#                    (OTELA_VERSION=<tag> pins one), then exit — does NOT
#                    submit. The recipe pins
#                    $OTELA_DIR/opentela-arm64-latest (repointed by the
#                    stager); rollback = the previous official release
#                    (OTELA_VERSION=v0.2.5, or pin OTELA_BIN to the old
#                    versioned file). The legacy sai-v0.0.6 fork is retired
#                    and unreferenced.
#
# Submit overrides (use the supported `cd remote_dir && sbatch` style, which
# is equivalent to `rcc job submit` for this self-contained recipe):
#   --partition NAME   e.g. preemptable (default: normal, from the recipe)
#   --time DUR         e.g. 12:00:00 (default: recipe's 12:00:00)
#   --account NAME     (default: infra02, from the recipe)
#
# Usage:
#   scripts/renew_glm_5_3_flash.sh                 # the usual "submit the next 12h job"
#   scripts/renew_glm_5_3_flash.sh --status
#   scripts/renew_glm_5_3_flash.sh --handover      # submit + swap when new is live
#   scripts/renew_glm_5_3_flash.sh --partition preemptable --watch
#
# Runs from anywhere (derives the repo root from its own location). Needs `rcc`
# on PATH for the idempotent push and `ssh clariden` (IdentityFile ~/.ssh/cscs-key,
# renewed daily via cscs-key).
# =============================================================================
set -uo pipefail

# ------------------------------------------------------------- constants -----
RECIPE_REL="deployments/llm/clariden/glm-53-flash/serve_glm_5_3_flash_otela_clariden.sbatch"
JOB_NAME="glm53-otela-clariden"                 # #SBATCH --job-name in the recipe
DEPLOY_DIR="${DEPLOY_DIR:-/iopsstor/scratch/cscs/xyao/glm-53-flash}"  # recipe's DEPLOY_DIR
SERVE_PORT="${SERVE_PORT:-30000}"
SERVED_MODEL="${SERVED_MODEL:-zai-org/GLM-5.3-Flash}"
# Bootstrap relay peer (OTELA_RELAY_ADDR in the recipe). Documented for
# reference — the worker reserves on VARIABLE relay peers (job 3462573 used
# BOTH QmTtn... and QmbUKJkCfotD; job 3474381 used only QmbUKJkCfotD), so
# registration_of() matches ANY recent relay-reservation marker, never this
# specific id.
RELAY_PEER="QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7"
OTELA_BIN_NAME="opentela-arm64"                              # for worker-alive pgrep (matches the versioned opentela-arm64-v0.2.6 too)
OTELA_DL_REL="meta/common/download_otela.sh"  # --update-otela payload (shared stager)
GATEWAY_MODELS="https://api.opentela.ai/v1/service/llm/v1/models"

PROFILE="clariden"
HOST="clariden"
NEAR_EXPIRY=3600          # submit a handover when the running job has <= 1h left
WATCH="0"
HANDOVER="0"
FORCE="0"
STATUS_ONLY="0"
UPDATE_OTELA="0"
DRY_RUN="0"
PARTITION=""              # empty = use recipe's --partition=normal
TIME_LIMIT=""
ACCOUNT=""
POLL=60                   # --watch/--handover poll interval, seconds
MAX_WAIT=7200             # --watch/--handover give-up, seconds (2h)
SSH_TIMEOUT=90

usage() {
  awk 'NR==1{next}
       /^[[:space:]]*#[[:space:]]*={4,}[[:space:]]*$/ {next}
       /^[[:space:]]*#/ {sub(/^[[:space:]]*#[[:space:]]?/,""); print; next}
       /^[[:space:]]*$/ {next}
       {exit}' "$0"
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --status)      STATUS_ONLY=1 ;;
    --watch)       WATCH=1 ;;
    --handover)    HANDOVER=1; WATCH=1 ;;
    --force)       FORCE=1 ;;
    --update-otela) UPDATE_OTELA=1 ;;
    --dry-run|-n)  DRY_RUN=1 ;;
    --partition)   PARTITION="${2:?--partition needs a value}"; shift ;;
    --time)        TIME_LIMIT="${2:?--time needs a value}"; shift ;;
    --account)     ACCOUNT="${2:?--account needs a value}"; shift ;;
    --profile)     PROFILE="${2:?--profile needs a value}"; HOST="$2"; shift ;;
    --host)        HOST="${2:?--host needs a value}"; shift ;;
    --near-expiry) NEAR_EXPIRY="${2:?--near-expiry needs a value}"; shift ;;
    --poll)        POLL="${2:?--poll needs a value}"; shift ;;
    --max-wait)    MAX_WAIT="${2:?--max-wait needs a value}"; shift ;;
    -h|--help)     usage ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
  shift
done

# Repo root (one level up from scripts/, so rcc finds .rcc/config.toml
# regardless of cwd).
SRC="${BASH_SOURCE[0]}"
REPO="$(cd "$(dirname "$SRC")/.." 2>/dev/null && pwd)" || REPO="$(pwd)"
[ -f "$REPO/$RECIPE_REL" ] || { echo "FATAL: recipe not found: $REPO/$RECIPE_REL" >&2; exit 1; }
[ -f "$REPO/.rcc/config.toml" ] || { echo "FATAL: no .rcc/config.toml at $REPO" >&2; exit 1; }
command -v rcc >/dev/null || { echo "FATAL: rcc not on PATH" >&2; exit 1; }

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; C=$'\033[36m'; B=$'\033[1m'; N=$'\033[0m'
else G=;Y=;R=;C=;B=;N=; fi

log()  { printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
note() { printf '%s\n' "$*"; }
die()  { printf '%sFATAL:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }
ok()   { printf '%s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '%s!%s %s\n' "$Y" "$N" "$*"; }

# ----------------------------------------------------------------- cert/ssh --
# BatchMode=yes so a missing/expired cert fails fast instead of prompting.
# Optional 2nd arg overrides the timeout (default $SSH_TIMEOUT) for
# long-running remote commands (e.g. --update-otela's binary download).
ssh_run() { local t="${2:-$SSH_TIMEOUT}"; timeout "$t" ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$1"; }

cert_check() {
  if ssh_run 'true' >/dev/null 2>&1; then ok "SSH to $HOST works"; return 0; fi
  local cert="$HOME/.ssh/cscs-key-cert.pub"
  [ -f "$cert" ] || die "no CSCS cert at $cert — run: cscs-key sign --headless"
  local to now exp
  to="$(ssh-keygen -L -f "$cert" 2>/dev/null | awk '/Valid:/{print $NF}')"
  now="$(date +%s)"
  exp="$(date -d "$to" +%s 2>/dev/null || echo 0)"
  if [ "$exp" -gt 0 ] && [ "$exp" -lt "$now" ]; then
    warn "CSCS SSH cert EXPIRED at $to. Renewal required for the next 24h."
    if [ "${NO_RENEW:-0}" = 1 ]; then die "cert expired (NO_RENEW set)."; fi
    local lf="/tmp/cscs-renew-$$.log"
    rm -f "$lf"
    log "launching: cscs-key sign --headless (valid 1 day) — pid backgrounded"
    nohup cscs-key sign --headless >"$lf" 2>&1 &
    sleep 12
    grep -iE "enter the code|open this URL|user_code=" "$lf" 2>/dev/null | head -4
    die "open https://auth.cscs.ch/auth/realms/cscs/device, enter the code above, \
then re-run this script."
  fi
  die "SSH to $HOST failed (cert reportedly valid until ${to:-unknown}). Check network/keys."
}

remote_user() { ssh_run 'whoami' 2>/dev/null | tr -d '\r\n'; }

# --------------------------------------------------------------- job state ---
# Globals set by gather_jobs:
#   RUN_JOB RUN_NODE RUN_LEFT RUN_LEFT_S   (RUN_LEFT/RUN_LEFT_S are the REAL
#   remaining walltime, read from squeue's %L = TIME_LEFT — NOT %l, which is
#   the fixed TIME LIMIT and would always read 12:00:00 for these jobs.)
#   PEND_JOB PEND_REASON
gather_jobs() {
  RUN_JOB=""; RUN_NODE=""; RUN_LEFT=""; RUN_LEFT_S=0
  PEND_JOB=""; PEND_REASON=""
  local ruser out
  ruser="$(remote_user)" || ruser="$USER"
  out="$(ssh_run "squeue -u '$ruser' -n '$JOB_NAME' -h -o '%i|%T|%M|%L|%N|%j|%r'" 2>/dev/null)" || return 0
  [ -z "$out" ] && return 0
  local id state used left nodes name reason
  while IFS='|' read -r id state used left nodes name reason; do
    [ -z "$id" ] && continue
    if [ "$state" = "RUNNING" ]; then
      RUN_JOB="$id"; RUN_LEFT="$left"
      RUN_LEFT_S="$(left_to_secs "$left")"
      RUN_NODE="$(printf '%s' "$nodes" | awk '{print $1}' | tr -d '[]')"
    elif [ "$state" = "PENDING" ]; then
      PEND_JOB="$id"; PEND_REASON="$reason"
    fi
  done <<< "$out"
}

# Parse Slurm TIME_LEFT → seconds. Slurm formats it as MM:SS (<1h),
# H:MM:SS (>=1h), or D-H:MM:SS (days) — the colon count disambiguates.
# UNLIMITED/INVALID → -1.
left_to_secs() {
  local s="$1" days=0 h m sec ncolons
  case "$s" in
    UNLIMITED|INVALID|"") echo -1; return ;;
  esac
  [[ "$s" == *-* ]] && { days="${s%%-*}"; s="${s#*-}"; }
  ncolons="${s//[^:]/}"
  case ${#ncolons} in
    1) h=0;            IFS=: read -r m sec <<<"$s" ;;  # MM:SS
    2)                 IFS=: read -r h m sec <<<"$s" ;;  # H:MM:SS
    *) echo -1; return ;;                                   # unexpected
  esac
  h="${h:-0}"; m="${m:-0}"; sec="${sec:-0}"
  echo $(( 10#$days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$sec ))
}
secs_to_hms() {
  local t=$1 h m s
  h=$(( t/3600 )); m=$(( (t%3600)/60 )); s=$(( t%60 ))
  printf '%d:%02d:%02d' "$h" "$m" "$s"
}

cluster_pressure() {
  ssh_run 'idle=$(sinfo -h -p normal -t idle -N -o "%N" 2>/dev/null | wc -l)
           alloc=$(sinfo -h -p normal -t alloc,mix -N -o "%N" 2>/dev/null | wc -l)
           pend=$(squeue -h -p normal -t PENDING -o "%i" 2>/dev/null | wc -l)
           qgl=$(squeue -h -p normal -t PENDING -o "%r" 2>/dev/null | grep -c QOSGrpNodeLimit)
           printf "normal: idle=%s alloc=%s pending=%s QOSGrpNodeLimit=%s\n" "$idle" "$alloc" "$pend" "$qgl"' \
    2>/dev/null || echo "normal: (cluster query failed)"
}

# Engine /health + live metrics, run on the job's head node via srun --overlap.
engine_health() {
  local jobid="$1"
  [ -z "$jobid" ] && { echo "no running job"; return; }
  local node
  node="$(ssh_run "squeue -j '$jobid' -h -o '%N'" 2>/dev/null | awk '{print $1}' | tr -d '[]')"
  [ -z "$node" ] && { echo "engine: job $jobid has no node yet"; return; }
  local probe
  probe="$(ssh_run "srun --jobid='$jobid' --overlap --gres=none -N1 -n1 -w '$node' --quiet \
      curl -s -m 10 -o /dev/null -w '%{http_code}' http://localhost:$SERVE_PORT/health" 2>/dev/null)" \
    || probe="err"
  if [ "$probe" != "200" ]; then echo "engine /health http=$probe (not ready yet)"; return; fi
  local m
  m="$(ssh_run "srun --jobid='$jobid' --overlap --gres=none -N1 -n1 -w '$node' --quiet \
      curl -s -m 10 http://localhost:$SERVE_PORT/metrics" 2>/dev/null)" || m=""
  local run=0 q=0 gen=0.0 kv=0.0
  run="$(printf '%s\n' "$m" | awk -F'[ ]' '/^sglang:num_running_reqs\{/{print $NF; exit}')"
  q="$(printf '%s\n' "$m"   | awk -F'[ ]' '/^sglang:num_queue_reqs\{/{print $NF; exit}')"
  gen="$(printf '%s\n' "$m" | awk -F'[ ]' '/^sglang:gen_throughput\{/{print $NF; exit}')"
  kv="$(printf '%s\n' "$m"  | awk -F'[ ]' '/^sglang:token_usage\{/{print $NF; exit}')"
  run="${run:-0}"; q="${q:-0}"; gen="${gen:-0}"; kv="${kv:-0}"
  printf 'engine /health=200  running=%s queued=%s gen=%s tok/s  KV=%.0f%%\n' \
    "$run" "$q" "$gen" "$(awk -v x="$kv" 'BEGIN{print x*100}')"
}

# OpenTela registration proof for THIS job:
#   1. otela worker process alive on the head node (srun pgrep)
#   2. a relay-reservation / registration marker in THIS JOB's otela.log
#      (the log is per-job, run-$jobid/otela.log, so any hit is from this
#      job's worker — never a stale one from a previous run).
#      v0.2.6 fires "Relay reservation on <peer> succeeded" + "Registered
#      relay peer <peer> in CRDT" (INFO) ONCE, ~10 s after CRDT init, and
#      "Registering LLM service" (DEBUG) once after the engine health check
#      — unlike sai-v0.0.6 they are NOT re-published every ~60 s, so we grep
#      the WHOLE log (a tail -500 window could have scrolled past the marker
#      under debug-level log noise); liveness comes from the pgrep above.
#      The <peer> is NOT stable (job 3462573 reserved on two different relay
#      peers), so we accept ANY relay-reservation / registration marker.
# Advisory: the OpenTela Cloud account link (v0.2.6 deploy keys; see the
#      recipe's OTELA_DEPLOY_KEY block) — informational only, never a gate.
# Advisory (if $OTELA_API is set): the model id present in the gateway /models.
#      HEAD-SIDE REJECTION (2026-10-10, jobs 3632183/3632230/3632369 era): the
#      markers above prove only WORKER-side intent. The head (ocf-1 at
#      140.238.223.116) can hold a WEDGED CRDT DAG: it re-broadcasts an
#      orphan delta whose child blocks live only on dead workers, stops
#      merging (zero "merged delta" journal lines), and new registrations
#      get a minimal connected-only row in /v1/dnt/table (no hostname /
#      services) so the gateway answers "No provider found" even though
#      registration_of() above says yes. NOT a signing issue (head runs
#      require_signed_binary=false; official GitHub builds are signed).
#      Fix (2026-10-10, worked): on ocf-1, systemctl stop otela && mv
#      ~/.ocfcore ~/.ocfcore.wedged-<ts> && systemctl start otela (seed "1"
#      preserves the peer id; live peers re-feed the DAG in ~2 min).
#      Ground truth for "serving on the mesh" = gateway /models (or the
#      head table row carrying services).
registration_of() {
  local jobid="$1" verdict="no"
  [ -z "$jobid" ] && { echo "registration: n/a (no running job)"; return 0; }
  local node
  node="$(ssh_run "squeue -j '$jobid' -h -o '%N'" 2>/dev/null | awk '{print $1}' | tr -d '[]')"
  local otelalog="$DEPLOY_DIR/run-$jobid/otela.log"
  local pid_alive=0 marker=""
  ssh_run "srun --jobid='$jobid' --overlap --gres=none -N1 -n1 -w '$node' --quiet \
      pgrep -af '$OTELA_BIN_NAME' >/dev/null 2>&1" >/dev/null 2>&1 && pid_alive=1
  marker="$(ssh_run "grep -aE 'Relay reservation on .* succeeded|Registered relay peer.* in CRDT|Registering LLM service' '$otelalog' 2>/dev/null | tail -1")" || marker=""
  if [ "$pid_alive" = 1 ] && [ -n "$marker" ]; then verdict="yes"; fi
  local acctlink=""
  acctlink="$(ssh_run "grep -aE 'Node linked to your OpenTela Cloud account|Instance .+ (linked|re-confirmed) with your OpenTela Cloud account|Deploy-key link check failed|instance link failed' '$otelalog' 2>/dev/null | tail -1")" || acctlink=""
  printf 'registration: %s  (worker %s, relay-marker %s)\n' \
    "$verdict" "$([ $pid_alive = 1 ] && echo alive || echo dead)" \
    "$([ -n "$marker" ] && echo 'this-job' || echo none)"
  printf '  account-link: %s\n' "$([ -n "$acctlink" ] && echo "$acctlink" || echo 'not linked (no deploy key, or pre-v0.2.6 binary)')"
  if [ -n "${OTELA_API:-}" ]; then
    local g
    g="$(curl -s -m 20 "$GATEWAY_MODELS" -H "Authorization: Bearer $OTELA_API" 2>/dev/null \
        | grep -o "\"id\":\"$SERVED_MODEL\"" | head -1)" || g=""
    printf '  gateway /models has %s: %s\n' "$SERVED_MODEL" "$([ -n "$g" ] && echo yes || echo no)"
  fi
}

print_status() {
  gather_jobs
  note "${B}== GLM-5.3-Flash on clariden ($HOST) ==${N}"
  if [ -n "$RUN_JOB" ]; then
    printf '  RUNNING  job=%s  node=%s  time_left=%s (%ss)\n' \
      "$RUN_JOB" "$RUN_NODE" "$RUN_LEFT" "$RUN_LEFT_S"
    engine_health "$RUN_JOB" | sed 's/^/  /'
    registration_of "$RUN_JOB" | sed 's/^/  /'
  else
    warn '  no RUNNING glm53-otela-clariden job'
  fi
  if [ -n "$PEND_JOB" ]; then
    printf '  PENDING  job=%s  reason=%s\n' "$PEND_JOB" "$PEND_REASON"
  fi
  note "  $(cluster_pressure)"
}

# ----------------------------------------------------------------- submit ----
do_submit() {
  # Idempotent sync of the recipe to clariden (respects keep_remote).
  # All progress goes to stderr; only the "Submitted batch job N" line is
  # echoed to stdout so the caller can capture the jobid cleanly.
  log "pushing recipe to clariden (idempotent)" >&2
  rcc --profile "$PROFILE" push "$RECIPE_REL" >/dev/null 2>&1 \
    || warn "rcc push reported a problem (continuing — remote is usually in sync)"

  if [ -z "$PARTITION" ] && [ -z "$TIME_LIMIT" ] && [ -z "$ACCOUNT" ]; then
    # Proven path — identical to `rcc --profile clariden job submit <rel>`.
    log "submitting via rcc job submit (normal, 12:00:00 from recipe)" >&2
    if [ "$DRY_RUN" = 1 ]; then
      note "[dry-run] would run: rcc --profile $PROFILE job submit $RECIPE_REL" >&2
      echo "DRY-RUN-NO-JOB"; return 0
    fi
    local out
    out="$(rcc --profile "$PROFILE" job submit "$RECIPE_REL" 2>&1)" || die "rcc job submit failed: $out"
    grep -oE 'Submitted batch job [0-9]+' <<<"$out" | tail -1
  else
    # Override path: `cd remote_dir && sbatch --partition=... --time=...`.
    # Equivalent to rcc for this self-contained recipe (its EDF is found via
    # SLURM_SUBMIT_DIR = remote_dir root; the recipe does not depend on
    # submit-time env beyond what it exports itself).
    local remote_dir opts out
    remote_dir="$(rcc --profile "$PROFILE" config 2>/dev/null | sed -n "s/^remote_dir = '\(.*\)'/\1/p" | head -1)"
    [ -n "$remote_dir" ] || remote_dir="/iopsstor/scratch/cscs/xyao/opentela-cookbook"
    opts=""
    [ -n "$PARTITION" ]  && opts="$opts --partition=$PARTITION"
    [ -n "$TIME_LIMIT" ] && opts="$opts --time=$TIME_LIMIT"
    [ -n "$ACCOUNT" ]    && opts="$opts --account=$ACCOUNT"
    log "submitting via ssh+cd (override:$opts) from $remote_dir" >&2
    if [ "$DRY_RUN" = 1 ]; then
      note "[dry-run] would run: ssh $HOST \"cd $remote_dir && sbatch$opts $RECIPE_REL\"" >&2
      echo "DRY-RUN-NO-JOB"; return 0
    fi
    out="$(ssh_run "cd '$remote_dir' && sbatch $opts '$RECIPE_REL'" 2>&1)" \
      || die "sbatch (override) failed: $out"
    grep -oE 'Submitted batch job [0-9]+' <<<"$out" | tail -1
  fi
}

# ------------------------------------------------------------ otela binary --
# Run the shared stager (meta/common/download_otela.sh) on clariden (needs a
# login node with GitHub egress + /iopsstor; ~66 MB download). Default stages
# the LATEST official release and repoints the opentela-arm64-latest symlink,
# which is what the recipe's OTELA_BIN default pins — no recipe bump needed.
# Versioned files ($OTELA_DIR/opentela-arm64-<tag>) stay on disk, so rollback
# is OTELA_VERSION=v0.2.5 (or pinning OTELA_BIN to the old versioned file).
update_otela() {
  [ -f "$REPO/$OTELA_DL_REL" ] || die "download script not found: $REPO/$OTELA_DL_REL"
  log "pushing $OTELA_DL_REL to clariden (idempotent)"
  rcc --profile "$PROFILE" push "$OTELA_DL_REL" >/dev/null 2>&1 \
    || die "rcc push of $OTELA_DL_REL failed"
  local remote_dir
  remote_dir="$(rcc --profile "$PROFILE" config 2>/dev/null | sed -n "s/^remote_dir = '\(.*\)'/\1/p" | head -1)"
  [ -n "$remote_dir" ] || remote_dir="/iopsstor/scratch/cscs/xyao/opentela-cookbook"
  log "running download_otela.sh on $HOST (OTELA_VERSION=${OTELA_VERSION:-latest}; ~66 MB)"
  ssh_run "OTELA_DIR=/iopsstor/scratch/cscs/xyao/opentela OTELA_ARCH=arm64 OTELA_VERSION='${OTELA_VERSION:-latest}' bash '$remote_dir/$OTELA_DL_REL'" 900 \
    || die "download_otela.sh failed on $HOST (egress to github.com OK? /iopsstor writable?)"
  note "next: nothing — the recipe pins \$OTELA_DIR/opentela-arm64-latest (repointed by the stager)"
  note "v0.2.6 adds OpenTela Cloud account linking: set OTD_API_KEY (otd-...) in"
  note "~/.zshrc on the cluster, or OTELA_DEPLOY_KEY at submit, and each job's peer"
  note "id is bound to your account (best-effort; one deploy-key use per job)."
}

# -------------------------------------------------------------- decision -----
decide_and_submit() {
  gather_jobs
  local action="submit"
  if [ -z "$RUN_JOB" ] && [ -z "$PEND_JOB" ]; then
    note "no glm53 job in the queue — submitting to bring the service up"
  elif [ -n "$PEND_JOB" ]; then
    action="skip"
    warn "pending job $PEND_JOB already queued to take over ($PEND_REASON); not duplicating"
  elif [ -n "$RUN_JOB" ]; then
    # running, no pending: hand over only near expiry
    if [ "$RUN_LEFT_S" -ge 0 ] && [ "$RUN_LEFT_S" -le "$NEAR_EXPIRY" ]; then
      note "running job $RUN_JOB has $(secs_to_hms $RUN_LEFT_S) left (<= --near-expiry $(secs_to_hms $NEAR_EXPIRY)) — handover warranted"
    elif [ "$RUN_LEFT_S" -ge 0 ]; then
      action="skip"
      warn "running job $RUN_JOB has $(secs_to_hms $RUN_LEFT_S) left (> --near-expiry $(secs_to_hms $NEAR_EXPIRY)); no handover needed"
    else
      action="skip"
      warn "running job $RUN_JOB time-left is '$RUN_LEFT' (unknown) — assuming covered, not submitting"
    fi
  fi
  [ "$FORCE" = 1 ] && { note "--force: submitting anyway"; action="submit"; }

  if [ "$action" = "submit" ]; then
    local submitted
    submitted="$(do_submit)" || exit $?
    if [ "$submitted" = "DRY-RUN-NO-JOB" ]; then
      note "[dry-run] no job submitted (would submit to $HOST)"
    else
      NEW_JOB="$(printf '%s' "$submitted" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | tail -1)"
      [ -n "$NEW_JOB" ] && ok "submitted job $NEW_JOB" || die "could not parse new jobid from: $submitted"
    fi
  fi
}

# --------------------------------------------------------------- watch ------
watch_until_ready() {
  # $NEW_JOB already set if we submitted; otherwise use any pending glm job.
  [ -n "${NEW_JOB:-}" ] || { gather_jobs; NEW_JOB="$PEND_JOB"; }
  [ -n "${NEW_JOB:-}" ] || { warn "nothing to watch (no new/pending glm job)"; return 0; }

  local target="$NEW_JOB" start elapsed
  start="$(date +%s)"
  note "${B}== watching job $target until RUNNING + healthy + registered ==${N}"
  note "  (poll every ${POLL}s, give up after $(secs_to_hms $MAX_WAIT))"

  local ready=0
  while :; do
    elapsed=$(( $(date +%s) - start ))
    [ "$elapsed" -gt "$MAX_WAIT" ] && { warn "gave up after $(secs_to_hms $elapsed); job not ready yet"; break; }

    # Resolve target state fresh.
    local state node
    state="$(ssh_run "squeue -j '$target' -h -o '%T:%N:%r'" 2>/dev/null)"
    local st="${state%%:*}"
    case "$st" in
      RUNNING)
        # Fresh state for engine/registration probes (both derive the node
        # from the jobid themselves, so no global here).
        local h reg
        h="$(engine_health "$target")"
        reg="$(registration_of "$target")"
        log "job $target: $h | $reg"
        if printf '%s' "$h" | grep -q 'health=200' && printf '%s' "$reg" | grep -q 'registration: yes'; then
          ok "job $target is serving and registered"
          ready=1; break
        fi ;;
      PENDING)
        log "job $target: PENDING ($st reason=$(printf '%s' "$state" | cut -d: -f3-)) — waited $(secs_to_hms $elapsed)"; ;;
      "")
        # Gone? Check sacct.
        local s2
        s2="$(ssh_run "sacct -j '$target' --format=State -n -P 2>/dev/null | head -1")"
        case "$s2" in
          TIMEOUT|COMPLETED|CANCELLED*|FAILED) warn "job $target ended ($s2) before becoming ready"; break 2 ;;
          *) log "job $target: not found in squeue (sacct=$s2) — waited $(secs_to_hms $elapsed)";;
        esac ;;
      *)
        log "job $target: state=$st — waited $(secs_to_hms $elapsed)";;
    esac
    sleep "$POLL"
  done

  if [ "$ready" != 1 ]; then
    warn "job $target did not reach ready within the watch window"
    return 1
  fi

  # Handover: cancel the OTHER (expiring) running glm job, if any, now that
  # the new one is confirmed live — zero serving gap by construction.
  if [ "$HANDOVER" = 1 ]; then
    gather_jobs
    local other=""
    [ -n "$RUN_JOB" ] && [ "$RUN_JOB" != "$target" ] && other="$RUN_JOB"
    if [ -n "$other" ]; then
      ok "handover: cancelling expiring job $other (new job $target is live)"
      [ "$DRY_RUN" = 1 ] && { note "[dry-run] would run: scancel $other"; return 0; }
      ssh_run "scancel '$other'" >/dev/null 2>&1 \
        && ok "scancelled $other" || warn "scancel $other failed"
    else
      note "handover: no other running glm job to cancel (nothing to free)"
    fi
  fi
  return 0
}

# ----------------------------------------------------------------- main ------
cert_check
cd "$REPO"

if [ "$UPDATE_OTELA" = 1 ]; then
  update_otela
  exit 0
fi

if [ "$STATUS_ONLY" = 1 ]; then
  print_status
  exit 0
fi

decide_and_submit

if [ "$WATCH" = 1 ] && [ "$DRY_RUN" != 1 ]; then
  # Handover only makes sense if we actually have a new job to watch.
  if [ -n "${NEW_JOB:-}" ] || { gather_jobs; [ -n "$PEND_JOB" ] || [ -n "$RUN_JOB" ]; }; then
    watch_until_ready || exit 1
  else
    warn "--watch requested but no glm job exists to watch"
    exit 1
  fi
fi

echo
print_status
