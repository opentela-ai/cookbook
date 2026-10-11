# meta/common — scripts shared by all recipes

One canonical copy of the scripts that several recipes had drifted copies of.
Recipes reference these in place (the repo is synced whole by `rcc`); do not
re-copy them into recipe directories.

| File | Used by | What it is |
|---|---|---|
| `sglang_launcher_mi300a.py` | all Beverin (MI300A) recipes | Drop-in `python -m sglang.launch_server` replacement that forces `is_integrated=False` (per-GPU memory accounting on the APU). Carries the `device=None` default fix — older per-recipe copies without it raise `TypeError` on the aiter DSA decode path. Most sbatches generate this inline; this file is the canonical standalone copy. |
| `gen_correctness.py` | glm-5.3-flash recipes (jsc/clariden/beverin/bristen), glm52 | The mandatory pre-registration generation probe (stdlib-only). Beverin `kimi-k3-vllm` carries a campaign fork with extra KDA probes — deliberately not merged. |
| `lib.sh` | every otela serving sbatch (beverin ×3, bristen*, jsc ×2, clariden ×2, euler) | Shared teardown contract: `stop_otela` (srun-step mode via `$OTELA_STEP_NAME`/`scancel -s TERM`, or PID mode via `$_OTELA_PID`), 20 s grace fuse < KillWait, LEFT-confirmation check. Recipes source it via a `meta/common` walk-up and set `_OTELA_LAUNCHED=1` after launching the worker (silent no-op for SMOKE runs / failed preflights). *bristen runs its worker without this lib — staged-copy deploy dir. |
| `download_otela.sh` | every recipe's `up.sh` | Stages the official OpenTela release from `eth-easl/OpenTela` GitHub releases. Default `OTELA_VERSION=latest` (GitHub API resolve); arch-detects `amd64`/`arm64` (`OTELA_ARCH` override); installs a versioned file + repoints `opentela-<arch>-latest` (what every recipe's `OTELA_BIN` default pins); `FORCE=1` re-downloads, `DRY_RUN=1` resolves only, rollback = pin a tag. The sai-v0.0.6 fork and the per-recipe stager copies are retired. |

Not here (deliberately per-recipe): `sglang-rocm.toml` (differs per model),
`start_relay_*.sh` (per-site: binary pins, port schemes, store relocation).
