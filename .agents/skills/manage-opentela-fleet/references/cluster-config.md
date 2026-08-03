# Cluster config for `otela-fleet`

One YAML file per cluster, named `<cluster>.yaml`. Filename (minus `.yaml`)
is the cluster name used in every command. Canonical source:
<https://opentela.ai/docs/extensions/fleet-manager>.

## Required fields

| Field | Purpose |
|---|---|
| `name` | cluster identifier |
| `ssh.host` | SSH hostname for relay operations |
| `arch` | `amd64` or `arm64` (which otela binary to deploy) |
| `binary.local_path` | local path to the otela binary |
| `binary.remote_path` | remote path to deploy the binary |
| `relay.*` | relay node config: `seed`, `peer_id`, `host_ip`, `port`, `tcp_port`, `udp_port`, `home_override`, `bootstrap[]`, and `skip` when the cluster reaches heads directly (e.g. JSC WSS) |
| `worker.*` | worker config: `seed`, `port`, `service_port` |
| `container.runtime` | `apptainer` or `enroot` |
| `container.image` | container image URI |
| `presets` | ≥1 hardware preset |

## Container runtimes

**Apptainer** — needs `container.sif_path`. Fleet runs:

```
apptainer exec [flags] --bind [mounts] [sif_path] [your_cmd]
```

**Enroot** — needs `container.edf_template` and `container.edf_remote_path`.
Fleet runs:

```
srun --environment=[edf_path] [your_cmd]
```

`container.env` (NCCL iface, cache dirs…) and `container.env_from_host`
(names of host env to forward) apply to both. `container.apptainer_flags`
(e.g. `--containall`, `--writable-tmpfs`, `--nv`) extend the apptainer line.

## Presets

Each preset is the SLURM shape for one deploy:

| Field | Required | Default | Notes |
|---|---|---|---|
| `partition` | yes | | SLURM partition |
| `account` | yes | | SLURM account |
| `time` | yes | | `HH:MM:SS` |
| `gpus` | yes | | count (`4`) or typed (`"rtx_3090:1"`) |
| `nodes` | no | `1` | `> 1` triggers the multi-node template |
| `cpus_per_task` | no | none | |
| `extra_sbatch` | no | `[]` | extra `#SBATCH` lines, e.g. `--mem-per-cpu=8G`, `--exclusive` |

## Full annotated example (Apptainer, amd64)

```yaml
name: jsc

ssh:
  host: jsc-login
  host_any: jsc-login

arch: amd64

binary:
  local_path: ./binaries/otela-amd64
  remote_path: ~/opentela/entry

relay:
  seed: "299"
  peer_id: QmPneGvHmWMngc8BboFasEJQ7D2aN9C65iMDwgCRGaTazs
  host_ip: "127.0.0.1"
  port: "18092"
  tcp_port: "43900"
  udp_port: "18820"
  home_override: /tmp/opentela-relay
  skip: true                       # JSC reaches heads directly via WSS; no relay needed
  bootstrap:
    - "https://bootstraps.opentela.ai/v1/dnt/bootstraps"

worker:
  seed: "300"
  port: "8092"
  service_port: "30000"

container:
  runtime: apptainer
  image: "lmsysorg/sglang:dev"
  sif_path: "/p/scratch/.../sglang-dev.sif"
  pull_if_missing: true
  hf_cache: "/p/scratch/.../models"
  mounts:
    - "/p/scratch/...:/p/scratch/..."
    - "/p/home/.../juwels:/p/home/.../juwels"
  env:
    FLASHINFER_WORKSPACE_DIR: "/p/scratch/.../sglang_cache/flashinfer"
    TRITON_CACHE_DIR: "/p/scratch/.../sglang_cache/triton"
  apptainer_flags:
    - "--containall"
    - "--writable-tmpfs"
    - "--nv"

security:
  require_signed_binary: false
solana:
  skip_verification: true

presets:
  A100_4:
    partition: booster
    account: laionize
    time: "04:00:00"
    gpus: 4
    nodes: 1
    extra_sbatch:
      - "#SBATCH --gpus-per-node=4"
  A100_8_multinode:                 # nodes>1 → multi-node template
    partition: booster
    account: laionize
    time: "08:00:00"
    gpus: 4
    nodes: 2
    extra_sbatch:
      - "#SBATCH --gpus-per-node=4"
```

## Variations worth knowing

- **Enroot on arm64** (e.g. CSCS Clariden): `arch: arm64`,
  `container.runtime: enroot`, `container.edf_template: <name>.toml.j2`,
  `container.edf_remote_path: ~/.edf/<name>.toml`, `relay.skip: true` when a
  long-running relay already exists (workers bootstrap straight from it).
  Forward a gated HF token with `container.env_from_host: [HF_TOKEN]`.
- **Euler (login-node relay for egress)**: keep `relay.skip: false` and set
  `container.hf_cache` to a scratch path; add `stack/2025-06`-style entries to
  `modules:` for the scheduler environment.
- **`modules:`** lists environment modules to load on the cluster (e.g.
  `GCC`, `CUDA/12`); optional and cluster-specific.

When you add a brand-new cluster, derive every field on-site (partition,
account, scratch paths, NIC names for `NCCL_SOCKET_IFNAME`, whether the relay
is needed) — record how each was learned in the cluster file's comments, the
same way the recipe skill records site facts.
