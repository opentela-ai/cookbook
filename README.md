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
```

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
| [`meta/bench/`](meta/bench/) | How we benchmark an LLM service: strategy, the C=1 trap, the shared benchmark harness, and the reporting checklist every throughput claim must carry. |

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
