# `remote-cluster-controller` integration for OpenTela

[`rcc`](https://github.com/ResearchComputer/remote-cluster-controller) is a
local CLI that generalizes `ssh host "cd dir && cmd"` + `rsync` into a
per-project config. It keeps the control loop on your laptop while SLURM runs
on the cluster login node. This doc shows how to use `rcc` with OpenTela
deployments and how it maps to `otela-fleet` concepts.

## Install

```bash
command -v rcc >/dev/null 2>&1 || {
  uv tool install remote-cluster-controller
  # or: pipx install remote-cluster-controller
  exit 1
}
```

## Concept map: `rcc` ↔ `otela-fleet`

| `otela-fleet` | `rcc` equivalent | Notes |
|---|---|---|
| `clusters/<cluster>.yaml` filename | `.rcc/config.toml` `[profiles.<name>]` | one profile per cluster |
| `ssh.host` | `profiles.<name>.host` | `otela-fleet` host should match the rcc profile host |
| `binary.local_path` | synced by `rcc push` | push the binary into the project directory on the remote |
| `binary.remote_path` | path under `remote_dir` | e.g. `remote_dir + "/bin/otela"` |
| `container.*` | `profiles.<name>.env` + `docker.*` or container setup inside the sbatch | rcc forwards env; container runtime stays inside the SLURM job |
| `otela-fleet start/status/stop` | `rcc job submit/list/status/cancel` when using native rcc backend | today, call `otela-fleet` with `ssh.host` matching the rcc host |

## Per-project setup

Inside the directory that holds your `clusters/` and fleet YAML:

```bash
rcc init
```

Edit `.rcc/config.toml` (gitignored by default). The OpenTela cookbook ships a
project-local `.rcc/config.toml` in its root with concrete profiles for
Beverin, Clariden, Euler, and JSC, ready to use after you verify the SSH host
aliases in `~/.ssh/config`. Ready-to-copy starter files also live at
[`examples/rcc/`](../examples/rcc/) (one `.toml` per site). Replace commented
placeholders with real hostnames and paths after probing the site (`sinfo`,
`df -h`, `ip -o -4 addr`, cluster docs).

### Generic template

```toml
default = "jsc"   # change to your most-used cluster

[profiles.jsc]
host = "jsc-login"              # set to the actual login node / bastion target
remote_dir = "/p/project/.../opentela-fleet"
# WHY proxy_jump: some sites require a bastion; rcc encapsulates it instead of
# forcing everyone to maintain matching ~/.ssh/config entries.
# proxy_jump = "bastion.example.com"
# identity_file = "~/.ssh/id_ed25519"

# Env defaults forwarded to every rcc run / rcc job submit on this cluster.
[profiles.jsc.env]
HF_HOME = "/p/scratch/.../models"
TRITON_CACHE_DIR = "/p/scratch/.../cache/triton"

# Paths owned by the running job that must survive syncs. rccignore-with-teeth:
# these are protected from --delete and --mirror even when they vanish locally.
keep_remote = [
  "logs/",
  "*.safetensors",
  "last_service.env",
  ".rcc-runs/",
]

# Optional: if you use rcc run --docker on the login node (not inside SLURM).
# Most OpenTela recipes run the container inside the sbatch instead.
# [profiles.jsc.docker]
# image = "..."
# workdir = "/workspace"
# gpus = "all"
```

### Euler (ETH Zürich) — login-node relay for egress

```toml
[profiles.euler]
host = "euler-login"            # set to the actual ETH login host
remote_dir = "/cluster/scratch/.../opentela-fleet"

[profiles.euler.env]
HF_HOME = "/cluster/scratch/.../models"
# WHY: compute nodes reach the outside world only via eth_proxy; the relay
# runs on the login node, so env needed there differs from compute env.
http_proxy = "http://proxy.ethz.ch:3128"
https_proxy = "http://proxy.ethz.ch:3128"

keep_remote = [
  "logs/",
  "*.safetensors",
  "last_service.env",
  ".rcc-runs/",
]
```

### Clariden / Beverin (CSCS Alps) — direct mesh, enroot/EDF

```toml
[profiles.clariden]
host = "clariden-login"         # set to the actual CSCS login host
remote_dir = "/capstor/scratch/.../opentela-fleet"

[profiles.clariden.env]
HF_HOME = "/capstor/scratch/.../models"
TRITON_CACHE_DIR = "/capstor/scratch/.../cache/triton"
# WHY NCCL_SOCKET_IFNAME: Alps uses Slingshot-11 / CXI, not InfiniBand; the
# recipe sets the interface inside the sbatch, but staging tools may need it.
# NCCL_SOCKET_IFNAME = "hsn0"

keep_remote = [
  "logs/",
  "*.safetensors",
  "last_service.env",
  ".rcc-runs/",
]

# Beverin is the same runtime on a different vCluster / partition.
[profiles.beverin]
host = "beverin-login"
remote_dir = "/capstor/scratch/.../opentela-fleet"

[profiles.beverin.env]
HF_HOME = "/capstor/scratch/.../models"
TRITON_CACHE_DIR = "/capstor/scratch/.../cache/triton"

keep_remote = [
  "logs/",
  "*.safetensors",
  "last_service.env",
  ".rcc-runs/",
]
```

## Workflow: `rcc` + `otela-fleet` today

1. Sync the project (cluster configs, binaries, recipes) to the remote:

   ```bash
   rcc push
   ```

2. Ensure `clusters/<cluster>.yaml` has `ssh.host` equal to the rcc profile
   `host`. `otela-fleet` will connect over the same SSH target `rcc` uses.

3. Run `otela-fleet` normally:

   ```bash
   otela-fleet start jsc --backend sglang --cmd "..." --preset A100_4
   otela-fleet status jsc
   otela-fleet stop jsc
   ```

4. Pull back logs/results when you are done:

   ```bash
   rcc pull logs/
   ```

## Workflow: `rcc job` directly with a recipe

For sites where you already have a self-contained sbatch recipe (see the
`write-deployment-recipe` skill), `rcc job` can submit and manage it directly
without `otela-fleet`:

```bash
# from the cookbook repository root
rcc --profile <site> push
rcc --profile <site> job submit deployments/llm/<site>/<model>/serve_*.sbatch
# prints JOBID
rcc --profile <site> job status <JOBID>
rcc --profile <site> job tail <JOBID> -f
rcc --profile <site> job cancel <JOBID>
```

Each recipe's README contains a "From your local machine via `rcc`" section
with the exact profile name and command for that model/site. Use
`-p <profile>` (or `--profile`) to select a cluster other than `default`.

## Native `rcc` backend for `otela-fleet` (future)

To make `otela-fleet` speak `rcc` natively, the fleet manager would replace its
direct-SSH calls with these mappings:

| `otela-fleet` action | `rcc` command |
|---|---|
| submit a SLURM job | `rcc -p <cluster> job submit <generated.sbatch>` |
| list running jobs | `rcc -p <cluster> job list --json` |
| get job status | `rcc -p <cluster> job status <JOBID> --json` |
| stream logs | `rcc -p <cluster> job tail <JOBID> -f` |
| cancel a job | `rcc -p <cluster> job cancel <JOBID>` |
| sync project before submit | `rcc -p <cluster> push` |
| pull back artifacts | `rcc -p <cluster> pull logs/` |

Required additions to `clusters/<cluster>.yaml`:

```yaml
name: jsc

# New: select rcc as the transport backend.
remote_cluster_controller:
  profile: jsc          # rcc profile name; falls back to cluster name if absent
  auto_push: true       # run `rcc push` before start/apply

# Existing ssh block stays valid and is used when backend is omitted.
ssh:
  host: jsc-login
  host_any: jsc-login

# ... rest of cluster config unchanged
```

`rcc job list --json` emits one record per active job; `rcc job status <JOBID>
--json` emits one record per sacct row. A native backend should match jobs by
the same `backend + cmd + preset` hash `otela-fleet` already uses for fleet
reconciliation.

## Safety notes

- Always set `keep_remote` for `logs/`, `*.safetensors`, `last_service.env`,
  and `.rcc-runs/`. These are owned by the remote job; a stray `rcc push
  --mirror` must not delete them.
- `rcc push --delete` is bounded: it deletes only files inside the non-ignored
  transfer scope. Use `--mirror` only when you genuinely want a full mirror,
  and still keep the paths above.
- `rcc run --docker` runs Docker on the SSH remote, so image names and mount
  sources are resolved on the cluster, not on your laptop. Most OpenTela
  recipes use the container runtime inside the sbatch instead (Apptainer or
  enroot), so Docker blocks are optional.
