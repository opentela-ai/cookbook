---
name: manage-opentela-fleet
description: Launch, inspect, and reconcile OpenTela serving workloads across SLURM-backed clusters with otela-fleet (direct SSH) or remote-cluster-controller/rcc as the local transport. Use when starting LLM serving on SLURM from a fleet YAML, scaling deployments, or wiring rcc job submit into an OpenTela workflow.
---

# Manage OpenTela with `otela-fleet`

`otela-fleet` is a Python CLI that wraps OpenTela cluster configuration, SLURM
job submission, and multi-cluster reconciliation. It is the higher-level
alternative to authoring a self-contained sbatch recipe by hand (the
`write-deployment-recipe` skill) — `otela-fleet` writes and submits the
sbatch for you from a cluster config + a single `--cmd`.

Canonical source: <https://opentela.ai/docs/extensions/fleet-manager>.

## Prereqs

```bash
command -v otela-fleet >/dev/null 2>&1 || {
  pip install otela-fleet        # or: cd contrib/fleet_manager && pip install -e .
  exit 1
}
# otela-fleet runs locally and SSHes to each cluster; SLURM itself must be
# installed on the cluster host, not on this machine. binary.local_path must
# exist locally (the fleet manager syncs it to each cluster).
```

## Cluster configs

Searched in this order (override with `--cluster-dir`):

1. `./clusters/` in the cwd
2. `~/.config/opentela/fleet/clusters/`

Each cluster is one YAML file; the **filename without `.yaml`** is the cluster
name you use in commands. Scaffold the user dir, then list/inspect:

```bash
mkdir -p ~/.config/opentela/fleet/clusters
otela-fleet clusters                     # all configured clusters + their presets
otela-fleet presets jsc                  # preset details (partition/account/gpus/time)
```

Required fields per cluster, the two container runtimes, and a full annotated
example: [references/cluster-config.md](references/cluster-config.md).

## Start, inspect, stop

`--cmd` runs **inside** the container; `--backend` selects the serving
engine, `--preset` the SLURM shape, `--replicas` the count (default 1). Two
env vars are available inside `--cmd`:

| Var | From |
|---|---|
| `$SERVICE_PORT` | `worker.service_port` |
| `$HF_HOME` | `container.hf_cache` |

```bash
otela-fleet start jsc \
  --backend sglang \
  --cmd "python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port \$SERVICE_PORT --host 127.0.0.1" \
  --preset A100_4_dev --replicas 1

otela-fleet status jsc                 # running jobs on the cluster
otela-fleet logs jsc 12345             # one job
otela-fleet stop jsc 12345             # stop one job
otela-fleet stop jsc                   # stop ALL OpenTela jobs on the cluster
```

The fleet manager syncs the `otela` binary to the cluster, ensures a relay if
the cluster needs one, and submits a SLURM job that runs your `--cmd` in the
configured container.

## Multi-node presets

A preset with `nodes > 1` makes the fleet manager: discover the master from
`$SLURM_NODELIST`, set NCCL env from `container.env`, wrap `--cmd` in
`srun --ntasks-per-node=1` with a per-node launcher, and run health on the
master. Your `--cmd` still carries the distributed args the backend needs
(`--nnodes`, `--node-rank`, `--tp`, …).

## Declarative reconciliation — `otela-fleet apply`

Define desired state in a fleet file and reconcile:

```yaml
# fleet.yaml
deployments:
  - cluster: jsc
    backend: sglang
    cmd: "python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port $SERVICE_PORT --host 127.0.0.1 --tp-size 4"
    preset: A100_4
    replicas: 2
  - cluster: euler
    backend: sglang
    cmd: "python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port $SERVICE_PORT --host 127.0.0.1"
    preset: RTX3090_1
    replicas: 1
```

```bash
otela-fleet apply fleet.yaml --dry-run    # show planned actions, change nothing
otela-fleet apply fleet.yaml              # submit/cancel to reach desired replicas
```

**Job identity is a hash of `backend + cmd + preset`.** So:

- change `cmd` → redeploy
- change `preset` → redeploy
- change **only** `replicas` → scale (no redeploy)

Reconciliation model: too few replicas → submit more; too many → cancel the
**newest** excess first; correct → no-op. To remove a deployment, set
`replicas: 0` (or drop the entry) and re-apply.

## `remote-cluster-controller` (`rcc`) transport

`otela-fleet` normally SSHes directly to each cluster login node. You can also
use [ResearchComputer/remote-cluster-controller](https://github.com/ResearchComputer/remote-cluster-controller)
(`rcc`) as the local-to-cluster transport: keep per-cluster config in
`.rcc/config.toml`, sync the project with `rcc push`, and submit/manage SLURM
jobs with `rcc job submit/status/cancel` while keeping the control loop on your
laptop.

Two integration patterns:

1. **`rcc` + `otela-fleet` together (works today)** — configure `rcc` for the
   cluster, push the `otela-fleet` cluster config to the remote, then run
   `otela-fleet` with `ssh.host` matching the `rcc` profile host. `rcc` owns the
   sync/project directory; `otela-fleet` owns the OpenTela/SLURM orchestration.
2. **`rcc` native job backend (future)** — replace the direct-SSH calls inside
   `otela-fleet` with `rcc job submit/list/status/cancel`. This needs a code
   change in the `otela-fleet` package; the cluster-config mapping is documented
   in [references/rcc-integration.md](references/rcc-integration.md).

### Common `rcc` commands

```bash
# sync local project to the remote scratch directory
rcc --profile <cluster> push

# submit a self-contained recipe directly (no otela-fleet)
rcc --profile <cluster> job submit deployments/llm/<site>/<model>/serve_*.sbatch

# list, inspect, stream logs, cancel
rcc --profile <cluster> job list
rcc --profile <cluster> job status <JOBID>
rcc --profile <cluster> job tail <JOBID> -f
rcc --profile <cluster> job cancel <JOBID>

# run an arbitrary command on the login node
rcc --profile <cluster> run -- sinfo -p <partition>
```

See [references/rcc-integration.md](references/rcc-integration.md) for example
`.rcc/config.toml` files per site and the exact command mapping.

## When to use this vs the recipe skill

- **`manage-opentela-fleet`** (this) — operate a known set of SLURM clusters
  quickly via a Python CLI and YAML (direct SSH or via `rcc`). Best when the
  tooling is already configured for the cluster and you want repeatable,
  reconciled deploys.
- **`write-deployment-recipe`** — author a single self-contained sbatch under
  `deployments/<kind>/<site>/<model>/` in this repo, carrying the
  site-specific details (firewall, runtime quirks, why-this-setting) that are
  hard to rediscover. Best for a brand-new site whose facts must be recorded,
  or where `otela-fleet`/`rcc` isn't set up.

Both produce LLM serving on OpenTela; `otela-fleet`/`rcc` are the higher-level
levers, the recipe is the durable artifact.
