# =============================================================================
# meta/common/lib.sh — shared shell helpers for the serving sbatches
# -----------------------------------------------------------------------------
# Sourced (NOT executed) by every serve*.sbatch. Each recipe resolves meta/common
# by walking up from its submit dir (SLURM_SUBMIT_DIR / SCRIPT_DIR / the EDF
# dir), then:
#
#   [ -n "$COMMON_DIR" ] || { echo "FATAL: meta/common/lib.sh not found ..." >&2; exit 1; }
#   source "$COMMON_DIR/lib.sh"
#
# (BASH_SOURCE inside a batch job is the spool copy — do NOT walk up from it
# alone; it only serves as the last-resort candidate.)
#
# Provides:
#   COMMON_DIR   — absolute path of meta/common/ (gen_correctness.py lives here;
#                  export it so srun-inherited env reaches container processes)
#   stop_otela   — graceful otela teardown (see below)
#
# Teardown contract (shared by all otela recipes): otela publishes its registry
# LEFT only on SIGTERM/INT (AnnounceLeave -> 5 s CRDT drain -> srv.Shutdown);
# SIGKILL leaves a stale connected:true row that round-robins traffic into a
# dead endpoint. srun does not reliably relay signals to tasks, so the worker
# is torn down via `scancel -s TERM <step>` (or kill for a background PID).
# =============================================================================
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_OTELA_STOPPED=0

# stop_otela — stop the otela worker and confirm the LEFT announcement.
# Launch modes (auto-detected):
#   srun step (default):  finds the step by $OTELA_STEP_NAME and scancel -s TERM it
#   background PID:       if $_OTELA_PID is set, kill -TERM it directly
# Recipes must set _OTELA_LAUNCHED=1 immediately after launching the worker:
# stop_otela is a silent no-op unless the worker actually started (failed
# preflight, SMOKE runs), and a LOUD WARN if the step vanished (SIGKILL path).
stop_otela() {
  [ "$_OTELA_STOPPED" = 1 ] && return 0          # TERM and EXIT both fire the trap
  _OTELA_STOPPED=1
  [ "${_OTELA_LAUNCHED:-0}" = 1 ] || return 0    # worker never launched this job
  if [ -n "${_OTELA_PID:-}" ]; then
    if kill -0 "$_OTELA_PID" 2>/dev/null; then
      echo "[$(date -Is)] stopping otela worker (pid $_OTELA_PID)"
      kill -TERM "$_OTELA_PID" 2>/dev/null
      for _ in $(seq 1 20); do
        kill -0 "$_OTELA_PID" 2>/dev/null || break
        sleep 1
      done
    fi
  else
    local step
    step="$(squeue -s -j "${SLURM_JOB_ID:-}" -h -o '%i %j' 2>/dev/null \
            | awk -v n="${OTELA_STEP_NAME:-}" '$2==n{print $1}' | head -1)"
    if [ -z "$step" ]; then
      echo "[$(date -Is)] WARN: otela step ${OTELA_STEP_NAME:-} not found; cannot signal it." >&2
      echo "[$(date -Is)]       If it was SIGKILLed, no LEFT was published and the head" >&2
      echo "[$(date -Is)]       may keep a stale connected=true row for this worker." >&2
      echo "[$(date -Is)]       Harmless: the next run mints a fresh peer id (per-job" >&2
      echo "[$(date -Is)]       config-dir/seed), so rows never overwrite — it only shows" >&2
      echo "[$(date -Is)]       as ~5-min dial-then-mismatch DEBUG spam on the head's CRDT." >&2
      return 0
    fi
    echo "[$(date -Is)] stopping otela step $step"
    scancel -s TERM "$step" 2>/dev/null
    # 20 s fuse < KillWait (30 s): otela needs ~10 s to shut down, so 20 s
    # leaves margin and still finishes before SIGKILL.
    for _ in $(seq 1 20); do
      [ -z "$(squeue -s -j "${SLURM_JOB_ID:-}" -h -o '%i %j' 2>/dev/null \
              | awk -v n="${OTELA_STEP_NAME:-}" '$2==n{print $1}')" ] && break
      sleep 1
    done
  fi
  grep -aqE "Announcing myself as LEFT|Leaving network" "${RUNDIR:-/tmp}/otela.log" 2>/dev/null \
    && echo "[$(date -Is)] otela: graceful leave confirmed" \
    || echo "[$(date -Is)] WARN: otela did not announce LEFT" >&2
}
