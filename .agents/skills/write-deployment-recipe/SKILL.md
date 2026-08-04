---
name: write-deployment-recipe
description: Add or modify an OpenTela deployment recipe under deployments/<service-kind>/<site>/<model>/ — a self-contained Slurm sbatch that serves a service and registers it on OpenTela. Use when creating a new recipe, porting an existing recipe to a new site (different cluster, GPU, partition, or container runtime), or serving a different model through an existing recipe.
---

# Write a deployment recipe

A recipe is a **copy-and-run sbatch**, not prose. It carries the site-specific
details that are hard to rediscover: firewall shape, container runtime,
scheduler quirks, and the OpenTela settings those force.

Canonical references to imitate:
- `deployments/llm/jsc/kimi-k3/serve_llm_otela_jsc.sbatch` (Slurm + Apptainer, relay)
- `deployments/llm/beverin/glm47-flash/` (Slurm + Pyxis/EDF, direct mesh, per-site README)

A starting skeleton lives at [assets/recipe.sbatch](assets/recipe.sbatch)
(inline comments mark every block you must re-derive per site).

## Layout

```
deployments/<service-kind>/<site>/<model>/
├── serve_<thing>.sbatch          # the recipe — one self-contained sbatch
├── build_<thing>.sh              # OPTIONAL: login-node-only step (image build, binary staging)
├── <site>.toml                   # OPTIONAL: container-runtime config (EDF/enroot)
└── README.md                     # why-this-site + workarounds + submit/verify/knobs
```

Sibling scripts are allowed only when the step genuinely cannot run in the
job body (e.g. building an image needs a login node with egress). Otherwise
heredoc helpers into `$RUNDIR` from inside the sbatch.

## Conventions (all load-bearing)

1. **Self-contained.** One recipe per directory. Where the environment forbids
   self-containment (e.g. JSC compute nodes have no outbound internet, so the
   otela binary must be staged on a login node), the recipe detects the
   missing prerequisite and **prints the exact staging command**, then exits
   non-zero. Never fail obscurely.
2. **Comments say what breaks, not what a setting does.** Style:
   `# WHY X: without it, Y fails with <observed error>`. These values were
   found by hitting the failure; new non-obvious settings must carry the same.
3. **Only verified claims.** Throughput numbers, timeout values, and "only
   stable topology" claims must be measured on-site. Mark anything unverified
   `TODO(unverified)` rather than guessing. Ask the user for measured data
   instead of inventing plausible numbers.
4. **Defaults target the documented site; everything else is an env override.**
   Pattern: `VAR="${VAR:-site-default}"`. Every knob is listed in the README's
   **Knobs** section.
5. **Site facts are re-derived, never copied.** Partition, account, scratch
   paths, NIC names (`NCCL_SOCKET_IFNAME`), container runtime, and relay
   topology differ per site. When porting, ask the user or run read-only
   probes (`sinfo`, `ip -o -4 addr`, `df -h`, checking `$SCRATCH`/quota) and
   record HOW each fact was learned in a comment.
6. **Measurement is built in, not bolted on.** The measurement engine is
   `servekit` (`github.com/eth-easl/servekit`, stdlib-only — runs via
   `PYTHONPATH=<checkout>/src python3 -m servekit.cli`, so no-egress compute
   nodes need only a checkout on the shared FS). Each recipe:
   - exports `SERVEKIT_DIR` (default: a servekit checkout under $DEPLOY_DIR),
     `SERVEKIT_BENCH=1`, `SERVEKIT_BENCH_REQUESTS=64`,
     `SERVEKIT_BENCH_CONCURRENCY=16` knobs and documents them in Knobs;
   - wraps the engine exec in `python3 -m servekit.cli profile --out
     "$RUNDIR/coldstart.json" --timeout "$HEALTH_TIMEOUT" --` when the
     checkout exists (per-node JSON, timeline of the cold start); missing
     checkout → WARN with the exact `git clone --depth=1` staging command,
     engine run unprofiled. Profile `--timeout` MUST equal the health
     timeout: on timeout the profiler KILLS the engine;
   - runs one verification bench (C=16, n=64) **after `/health` and before
     otela registration** so mesh traffic never pollutes it; write the JSON
     into `$RUNDIR` (`--into coldstart.node0.json` when profiling was active
     so startup timeline and bench live in one artifact), non-fatal (WARN +
     register anyway). `SERVEKIT_BENCH=0` must skip it entirely.
   Canonical implementation: the kimi-k3 recipes
   (`deployments/llm/clariden/kimi-k3/serve_kimi_k3_otela_clariden.sbatch`
   enroot/EDF; `deployments/llm/jsc/kimi-k3/serve_llm_otela_jsc.sbatch`
   Apptainer — note the explicit `env PYTHONPATH=...` inside the container
   exec). Full protocol: `meta/bench/README.md` (C=1 trap, words-vs-tokens
   input-len, reporting checklist).

## Recipe anatomy (in sbatch order)

1. `#SBATCH` header + big comment block: submit examples, what it does in
   order, and the site's network/relay topology drawn as ASCII if non-obvious.
2. Deployment vars (paths, image, model) → shape vars (TP/PP/EP with WHY) →
   otela vars (ports, seed, service name) → site quirks (timeouts, caches off
   quota-limited home).
3. Preflight: check image/weights/binaries exist; on failure print the exact
   command to stage them. Account for "compute nodes may have no egress".
4. Write `$DEPLOY_DIR/last_service.env` (job id, head node, head IP, port)
   so humans and later tooling can find the live service.
5. Heredoc `$RUNDIR/engine.sh`: per-rank env (NCCL/GLOO iface, socket family,
   IB retry knobs, cache dirs), then `exec <container> serve ...`.
6. Heredoc otela config + worker script; start the worker only **after**
   `/health` answers, in the background, with a generous timeout — and run
   the servekit verification bench between the health barrier and the worker
   start (convention 6).
7. `trap` on EXIT/TERM/INT to stop the otela worker with a **signal (TERM),
   never SIGKILL** — a killed peer stays `connected: true` in the registry and
   the head round-robins traffic into a dead endpoint. Wait for the graceful
-leave log line; warn if it never comes.
8. Run the engine via `srun` in the foreground so the job lifetime == engine
   lifetime.

Step ordering is deliberate; reorder only with a comment saying why.

## Site README format

Match `deployments/llm/beverin/glm47-flash/README.md`:

1. One-paragraph summary: what model/service, what site (GPU/arch/partition),
   what serving stack, and how it reaches OpenTela (direct vs relay — and WHY,
   i.e. the egress/firewall fact that decides it).
2. **Why the `<partition>`** — the constraint that forced the partition/image
   choice.
3. **Site-specific fixes (all baked into the sbatch)** — numbered list; each
   entry: symptom with the verbatim error, root cause, fix. This is the most
   valuable section; write it as you hit failures, not after.
4. **Files** table. 5. **Submit** (copy-paste commands). 6. **Verify**
   (commands that were actually executed, including log lines to look for and
   what "requests are being routed in" looks like). 7. **Knobs** — every env
   override with its default.

## Done checklist

- [ ] `bash -n` passes; recipe ran end-to-end on the target site at defaults
- [ ] servekit verification bench printed its throughput line before the
      otela worker registered, and `$RUNDIR/coldstart.node*.json` exists
      (or the missing-checkout WARN path was exercised instead)
- [ ] every `#SBATCH` value and env default derived on-site, with HOW noted
- [ ] each site workaround has a verbatim-error comment; cross-referenced from
      the README's numbered fixes
- [ ] failure paths print exact fix commands
- [ ] README Verify commands were executed and output sanity-checked
