# OpenTela Cookbook

Working, end-to-end deployment recipes for running services on OpenTela.

Each recipe is meant to be **copied and run**, not read as prose. They carry the
site-specific details that are hard to rediscover — firewall shape, container
runtime, scheduler quirks, and the OpenTela settings those force.

## Layout

```
deployments/<service-kind>/<site>/<model>/    # one self-contained recipe per directory
meta/bench/                                   # benchmark harness shared by all recipes
```

| Path | What it covers |
|------|----------------|
| `deployments/llm/jsc/kimi-k3/` | Kimi-K3 serving on JSC Jupiter Booster (GH200, Slurm, Apptainer). Start with its [`README.md`](deployments/llm/jsc/kimi-k3/README.md) for the verified findings (why TP4×PP8, the SHARP story, why TP32/EP32 is Blackwell-gated). |
| `deployments/llm/beverin/glm47-flash/` | GLM-4.7-Flash serving on Beverin (AMD MI300A, ROCm, EDF). |
| `deployments/llm/beverin/deepseek-v4/` | DeepSeek-V4-Flash serving on Beverin (AMD MI300A, ROCm, EDF). |
| [`meta/bench/`](meta/bench/) | How we benchmark an LLM service: strategy, the C=1 trap, the shared benchmark harness, and the reporting checklist every throughput claim must carry. |

## Conventions

- One recipe per directory, self-contained where the environment allows it.
  Where it does not, the script says so explicitly and prints the exact commands
  for the missing prerequisite rather than failing obscurely.
- Every non-obvious setting carries a comment explaining **what breaks without
  it**, not just what it does. Most of these were found by hitting the failure.
- Defaults target the documented site. Everything else is an environment
  variable override.
