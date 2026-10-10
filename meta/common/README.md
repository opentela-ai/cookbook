# meta/common — scripts shared by all recipes

One canonical copy of the scripts that several recipes had drifted copies of.
Recipes reference these in place (the repo is synced whole by `rcc`); do not
re-copy them into recipe directories.

| File | Used by | What it is |
|---|---|---|
| `sglang_launcher_mi300a.py` | all Beverin (MI300A) recipes | Drop-in `python -m sglang.launch_server` replacement that forces `is_integrated=False` (per-GPU memory accounting on the APU). Carries the `device=None` default fix — older per-recipe copies without it raise `TypeError` on the aiter DSA decode path. Most sbatches generate this inline; this file is the canonical standalone copy. |
| `gen_correctness.py` | glm-5.3-flash recipes (jsc/clariden/beverin/bristen), glm52 | The mandatory pre-registration generation probe (stdlib-only). Beverin `kimi-k3-vllm` carries a campaign fork with extra KDA probes — deliberately not merged. |

Not here (deliberately per-recipe): `sglang-rocm.toml` (differs per model),
`start_relay_*.sh` (per-site: binary pins, port schemes, store relocation).
