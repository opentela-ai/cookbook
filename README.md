# OpenTela Cookbook

Working, end-to-end deployment recipes for running services on OpenTela.

Each recipe is meant to be **copied and run**, not read as prose. They carry the
site-specific details that are hard to rediscover — firewall shape, container
runtime, scheduler quirks, and the OpenTela settings those force.

## Layout

```
deployments/<service-kind>/<site>/<model>/          # one self-contained recipe per directory
deployments/<service-kind>/local/<site>/<model>/    # no scheduler: single-box, hand-run scripts
conventions/                                       # cross-recipe rules (LLM served-model naming, …)
meta/bench/                                        # benchmark harness shared by all recipes
meta/tools/debugger/                               # agent toolkit for correctness bugs (probes, capture/diff bisect, journal)
meta/tools/profiler/                               # agent toolkit for kernel perf (forward breakdown, roofline microbench, compare)
.rcc/config.toml                                   # project-local rcc profiles for remote submit
```

## Submitting recipes from your local machine with `rcc`

Most recipes can be submitted without SSH-ing into the login node by using
[`rcc`](https://github.com/eth-easl/remote-cluster-controller) (install:
`uv tool install remote-cluster-controller`). The repository root contains a
project-local `.rcc/config.toml` with one profile per site:

```toml
[profiles.beverin]
host = "beverin"                                # SSH alias from ~/.ssh/config
remote_dir = "/capstor/scratch/cscs/xyao/opentela-cookbook"
```

Supported profiles: `beverin`, `clariden`, `euler`, `jsc`.

```bash
# sync local changes to the remote scratch directory
rcc --profile <site> push

# submit a recipe
rcc --profile <site> job submit deployments/llm/<site>/<model>/serve_*.sbatch

# monitor
rcc --profile <site> job status <JOBID>
rcc --profile <site> job tail <JOBID> -f

# run an arbitrary command on the login node (e.g. tail logs)
rcc --profile <site> run -- tail -f /path/to/log
```

Each recipe's README has a "From your local machine via `rcc`" subsection
with the exact commands for that site. Profiles for CSCS Alps sites
(Beverin/Clariden) share `/capstor`, so a `rcc --profile beverin push` also
updates the files visible from Clariden.

## Compute substrates

| Substrate | Hardware | Fabric / connectivity | Scheduler + runtime | OpenTela path | Recipes |
|---|---|---|---|---|---|
| **CSCS Alps — Clariden** | 4× NVIDIA GH200 120 GB per node (aarch64, 288 cores) | HPE Slingshot 11 via CXI libfabric (`aws-ofi-ccl-plugin`); **no InfiniBand** | Slurm + Pyxis + enroot (EDF) | direct p2p mesh to the Alps bootstrap, no relay | [`clariden/kimi-k3`](deployments/llm/clariden/kimi-k3/) |
| **CSCS Alps — Beverin** | 4× AMD MI300A APU per node (gfx942, unified memory) | same Alps fabric | Slurm + Pyxis + enroot (EDF) | direct — compute nodes have full outbound | [`beverin/glm47-flash`](deployments/llm/beverin/glm47-flash/), [`beverin/deepseek-v4`](deployments/llm/beverin/deepseek-v4/) |
| **JSC — Jupiter Booster** | 4× NVIDIA GH200 per node (aarch64) | InfiniBand (SHARP) | Slurm + Apptainer (`.sqsh`) | direct | [`jsc/kimi-k3`](deployments/llm/jsc/kimi-k3/) |
| **ETH Zürich — Euler** | NVIDIA RTX PRO 6000 Blackwell (one GPU per service) | compute nodes: outbound HTTP(S) only, via the `eth_proxy` module | Slurm + Apptainer | **login-node relay required** | [`euler/qwen36-35b-a3b`](deployments/llm/euler/qwen36-35b-a3b/) |
| **Local — NVIDIA DGX Spark** | 1× NVIDIA GB10 (aarch64, sm_121, 122 GB unified memory) | single box | no scheduler — Docker overlay or bare-metal Ollama | direct | [`dgx-spark/qwen36-35b-a3b`](deployments/llm/local/dgx-spark/qwen36-35b-a3b/), [`dgx-spark/qwen3-1.7b-ollama`](deployments/llm/local/dgx-spark/qwen3-1.7b-ollama/) |

**Alps** is CSCS's umbrella infrastructure; Clariden and Beverin are two
vClusters on it (same Slingshot-11 fabric, `/capstor` scratch, and the
container-engine / EDF runtime) — a recipe ported between them changes
site paths and partition names, not the runtime model. The big split is
**fabric**: on InfiniBand (Jupiter) cross-node TP is viable, on Slingshot
(Alps) cross-node collectives cannot be captured by CUDA graphs, so large
recipes go **TP-within-node × PP-across-nodes** instead. On single-box
substrates (DGX Spark) everything runs without a scheduler.

| Path | What it covers |
|------|----------------|
| `deployments/llm/jsc/kimi-k3/` | `moonshotai/Kimi-K3` serving on JSC Jupiter Booster (GH200, Slurm, Apptainer). Start with its [`README.md`](deployments/llm/jsc/kimi-k3/README.md) for the verified findings (why TP4×PP8, the SHARP story, why TP32/EP32 is Blackwell-gated). |
| `deployments/llm/clariden/kimi-k3/` | `moonshotai/Kimi-K3` serving on CSCS Clariden (GH200, aarch64, Slurm + enroot/EDF, Slingshot fabric — no InfiniBand). TP4×PP8, verified 561 tok/s aggregate @ C=32 (1024-in/256-out), 1M context window. Start with its [`README.md`](deployments/llm/clariden/kimi-k3/README.md) for the 10 site-specific fixes (Slingshot NCCL env, the 480 s loading-barrier monkey-patch, the otela cfg.yaml requirements). |
| `deployments/llm/beverin/glm47-flash/` | `zai-org/GLM-4.7-Flash` serving on Beverin (AMD MI300A, ROCm, EDF). |
| `deployments/llm/beverin/deepseek-v4/` | `deepseek-ai/DeepSeek-V4-Flash` serving on Beverin (AMD MI300A, ROCm, EDF). |
| `deployments/llm/euler/qwen36-35b-a3b/` | `Qwen/Qwen3.6-35B-A3B-FP8` serving on ETH Zürich Euler (RTX PRO 6000, Blackwell, Slurm + Apptainer). Login-node relay for egress (compute has HTTP(S)-only via `eth_proxy`). Start with its [`README.md`](deployments/llm/euler/qwen36-35b-a3b/README.md) for the cli_filter/GRES routing story and why a login-node relay is required. |
| `deployments/llm/local/dgx-spark/qwen36-35b-a3b/` | `Qwen/Qwen3.6-35B-A3B-FP8` serving on a DGX Spark (NVIDIA GB10, sm_121, aarch64). No scheduler: a Docker overlay on a golden GB10 image plus a standalone `otela` sidecar. |
| `deployments/llm/local/dgx-spark/qwen3-1.7b-ollama/` | `ollama/qwen3:1.7b` serving on a DGX Spark (NVIDIA GB10, sm_121, aarch64) via **Ollama** (bundled CUDA v13 libs, no golden image, no container). Start with its [`README.md`](deployments/llm/local/dgx-spark/qwen3-1.7b-ollama/README.md) for the Ollama-has-no-`/health` and Modelfile-alias-for-`org/model-name` stories. |
| [`conventions/`](conventions/) | Cross-recipe rules, starting with LLM served-model names use the `org/model-name` form. |
| [`meta/bench/`](meta/bench/) | How we benchmark an LLM service: strategy, the C=1 trap, the shared harness (**servekit bench** via `cbench.sh` / `cbench_report.py`, stdlib-only so it runs zero-install on no-egress compute nodes; `servekit profile` for cold-start timelines), and the reporting checklist every throughput claim must carry. |
| [`meta/tools/debugger/`](meta/tools/debugger/) | How we debug **correctness** bugs (garbage output, numerical drift): the five-phase method distilled from the GLM-5.3-Flash MI300A bisect — live-server failure-signature probes, no-reference per-layer residual bisect, identity-gated capture→diff, isolated kernel primitive tests, and the investigation journal. Operator tools are stdlib-only; in-engine hook modules are env-gated and self-installing. |
| [`meta/tools/profiler/`](meta/tools/profiler/) | How we find **where the milliseconds go** (utilization, slow kernels, bottlenecks): torch.profiler-based forward breakdown ranked into kernel families with GPU busy% and gap analysis, an event-timed microbench that roofline-classifies kernels (memory-/compute-/launch-bound vs nominal device peaks), and a before/after compare — portable across CUDA/ROCm, bench-file contract + JSON outputs. Comm bottlenecks → `meta/bench/nccl_sharp_probe.py`. |

## Conventions

- LLM served-model names use the `org/model-name` form and must agree across
  `--served-model-name`, the otela `model=` label, and the client `model` field.
  See [`conventions/`](conventions/).
- One recipe per directory, self-contained where the environment allows it.
  Where it does not, the script says so explicitly and prints the exact commands
  for the missing prerequisite rather than failing obscurely.
- Every non-obvious setting carries a comment explaining **what breaks without
  it**, not just what it does. Most of these were found by hitting the failure.
- Defaults target the documented site. Everything else is an environment
  variable override.
- Local recipes (under `deployments/<service-kind>/local/`) have no scheduler: they are
  plain bash scripts run by hand on the box — a `build_image.sh` for the one-time
  image, a `serve_*.sh` for the engine, and a `register_*.sh` (or the engine's own
  `--enable-opentela`) to join OpenTela. Defaults still target the documented site.
