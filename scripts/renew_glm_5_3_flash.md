# renew_glm_5_3_flash.sh — rolling 12h handover for GLM-5.3-Flash on Clariden

Automates the repeatable process of keeping `zai-org/GLM-5.3-Flash` (served by
[`serve_glm_5_3_flash_otela_clariden.sbatch`](../deployments/llm/clariden/glm-53-flash/serve_glm_5_3_flash_otela_clariden.sbatch))
continuously up across 12h Slurm windows on CSCS Clariden, registered on the
OpenTela mesh. One invocation checks state, **submits the next 12h job only
when a handover is warranted** (never a wasteful duplicate), and prints live
status — engine `/health`, throughput, and OpenTela registration proof.

> The serving recipe and its many knobs (TP/PP shape, DSA/attention backends,
> HiCache, watchdogs, the gen probe) are documented in the
> [deployment README](../deployments/llm/clariden/glm-53-flash/README.md).
> This doc is only about driving the rolling handover.

## Requirements

- `rcc` on `PATH` (profiles `clariden`/`beverin` share `/capstor`).
- A valid CSCS SSH cert at `~/.ssh/cscs-key-cert.pub` (renewed daily via
  `cscs-key sign --headless`). The script prechecks it and, if expired,
  launches `cscs-key sign --headless` in the background and prints the device
  code + URL — re-run the script once you've authorised.
- Optionally `OTELA_API` in the environment for an advisory gateway
  `/models` cross-check (the per-job registration proof does **not** require
  it).

## Quick start

```bash
# The usual case: decide + submit the next 12h job if needed + status.
scripts/renew_glm_5_3_flash.sh

# Just look — submit nothing.
scripts/renew_glm_5_3_flash.sh --status
```

A bare run is safe to repeat any time: if the running job still has plenty of
walltime and a pending successor is already queued, it prints status and exits
without submitting.

## Modes

| Mode | What it does |
|------|--------------|
| *(default)* | Decide whether a handover is warranted, submit if so, then print status. |
| `--status` | Print status only. Never submits. |
| `--watch` | Submit-if-needed, then block until the new job is `RUNNING` + `/health=200` + registered (bounded by `--max-wait`). |
| `--handover` | Like `--watch`, but once the new job is confirmed live it `scancel`s the expiring running job for a zero-gap swap. |
| `--force` | Submit even when the window is already covered (running job far from expiry, or a pending successor exists). |
| `--update-otela` | Push the shared stager ([`meta/common/download_otela.sh`](../meta/common/download_otela.sh)) to clariden and run it there to stage the **latest** official OpenTela release (`OTELA_VERSION=<tag>` pins one), repointing the `opentela-arm64-latest` symlink the recipe's `OTELA_BIN` default follows — no recipe bump needed. Then exits — never submits. Rollback = `OTELA_VERSION=v0.2.5` (stages the old versioned file alongside) and submit with `OTELA_BIN=$OTELA_DIR/opentela-arm64-v0.2.5`; the legacy sai-v0.0.6 fork is retired. |
| `--dry-run` (`-n`) | Resolve and print the exact submit command, but submit nothing. Skips `--watch`/`--handover`. |

## Submit overrides

Without overrides the script uses `rcc --profile clariden job submit <recipe>`
— identical to the proven manual path. Passing **any** of these switches it
instead uses `ssh clariden "cd <remote_dir> && sbatch <opts> <recipe>"`
(equivalent for this self-contained recipe, whose EDF is found via
`SLURM_SUBMIT_DIR`):

| Flag | Default | Notes |
|------|---------|-------|
| `--partition NAME` | `normal` (from the recipe) | e.g. `preemptable` (instant start, REQUEUE/preemption risk). |
| `--time DUR` | `12:00:00` (from the recipe) | e.g. `06:00:00` for a shorter window. |
| `--account NAME` | `infra02` (from the recipe) | charge account. |

## Tuning (rarely needed)

| Flag | Default | Meaning |
|------|---------|---------|
| `--near-expiry SECS` | `3600` (1h) | A running job with ≤ this much walltime triggers a submit. |
| `--poll SECS` | `60` | `--watch`/`--handover` poll interval. |
| `--max-wait SECS` | `7200` (2h) | `--watch`/`--handover` give up if the new job isn't ready in time. |
| `--profile NAME` | `clariden` | Also sets the SSH host. |
| `--host NAME` | `clariden` | Override the SSH host independently. |

Run `scripts/renew_glm_5_3_flash.sh --help` for the same reference inline.

## How it decides (one of three states)

| State | Decision |
|-------|----------|
| No running, no pending job | **Submit** (service down or first bring-up). |
| A job is already `PENDING` | **Skip** — it's queued to take over; not duplicated (unless `--force`). |
| A job is `RUNNING` with **≤ `--near-expiry`** walltime left and nothing pending | **Submit** the handover. |
| A job is `RUNNING` with more than that and nothing pending | **Skip** — window covered (unless `--force`). |

The walltime check uses Slurm's real `TIME_LEFT` (see the gotchas below);
with the default `--near-expiry 3600`, a bare run at the ~1h-left mark submits
the successor in time to start **behind** the QOS wall before the old job
expires — avoiding the unavoidable ~8 min shard-load gap you get when the old
job times out first.

## What "ready" means (for `--watch` / `--handover`)

A new job is considered ready when **both** hold, on its head node:

1. **Engine `/health` = 200** — probed via `srun --jobid --overlap --gres=none
   curl http://localhost:30000/health`. (The recipe sets
   `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`, so this is a cheap liveness
   check, not a generation round-trip.)
2. **OpenTela registration** — the otela worker process is alive **and** a
   relay-reservation / registration marker appears in this job's
   `run-<jobid>/otela.log` (per-job log, so any hit is from this job's
   worker; v0.2.6 fires `Relay reservation on <peer> succeeded` /
   `Registered relay peer … in CRDT` once, ~10 s after CRDT init — liveness
   comes from the process check, not marker recency).

When `OTELA_API` is set, an advisory `gateway /v1/service/llm/v1/models` check
is also printed — but it is **not** a gate (mesh propagation can lag the local
worker by a few seconds). The status output additionally reports the OpenTela
Cloud **account link** (`account-link:` line, from the job's `otela.log`) —
also advisory: linking is best-effort in the recipe, and a missing link never
blocks the watch/handover gates.

## Examples

```bash
# The normal cadence: run it once an hour (or just before the old job's last
# hour) and it submits exactly when needed.
scripts/renew_glm_5_3_flash.sh

# Submit + block until the new job is live, then scancel the old one for a
# zero-gap swap. Use this when you want the swap to happen on your schedule.
scripts/renew_glm_5_3_flash.sh --handover

# Preemptable partition for an instant start (carries REQUEUE/preemption risk).
scripts/renew_glm_5_3_flash.sh --partition preemptable --watch

# Shorter window, e.g. a 6h slot.
scripts/renew_glm_5_3_flash.sh --time 06:00:00

# See exactly what it would do, submit nothing.
scripts/renew_glm_5_3_flash.sh --dry-run --force
```

## Gotchas (for maintainers)

Three non-obvious things this script gets right that a naive rewrite would
break:

1. **`%l` is the TIME LIMIT, not time left.** Slurm's `squeue` format codes:
   `%M` = elapsed, `%l` = TIME LIMIT (fixed, always `12:00:00` for these
   jobs), `%L` = TIME LEFT. The handover decision **must** use `%L`; using
   `%l` makes the running job always appear to have 43200s left, so it never
   triggers a handover. (Verified via `scontrol show job`: `RunTime=11:57
   TimeLimit=12:00:00 TimeLeft=2:45`.)

2. **Slurm writes `MM:SS`, not `H:MM:SS`, when time left is under an hour.**
   With ~3 min left, `%L` returns `2:45` (= 2 min 45 s = 165s), which a naive
   `H:MM:SS` parser reads as 2h45m = 9900s — 60× too high, again suppressing
   the handover. `left_to_secs` disambiguates on the colon count
   (`MM:SS` / `H:MM:SS` / `D-H:MM:SS`).

3. **The otela worker reserves on variable relay peer ids.** The bootstrap
   peer (`/ip4/.../p2p/QmTtn…`) is for joining the mesh; the relay(s) the
   worker then *reserves* on are not stable — job 3462573 reserved on both
   `QmTtn…` and `QmbUKJkCfotD`, job 3474381 only on `QmbUKJkCfotD`. The
   registration check therefore matches **any** recent
   `Relay reservation on <peer> succeeded` / `Registered relay peer … in CRDT`
   marker, never a pinned peer.

## See also

- [Deployment README](../deployments/llm/clariden/glm-53-flash/README.md) — the
  serving recipe, knobs, measured performance, and the graceful-stop procedure.
- [`serve_glm_5_3_flash_otela_clariden.sbatch`](../deployments/llm/clariden/glm-53-flash/serve_glm_5_3_flash_otela_clariden.sbatch) — the recipe this script submits.
