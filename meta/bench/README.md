# Benchmarking an LLM service

This directory holds the shared tooling behind every throughput, latency, or
startup number quoted in this repo (recipe READMEs, site notes). If a claim
isn't reproducible with what's here, it doesn't go in the cookbook.

**The one number we quote: `agg_out_tok_s = total_completion_tokens / wall_clock`
at a stated concurrency C.** (servekit reports it as
`throughput.output_tok_per_s = output_tokens / wall_s`; legacy `bench.py`
called it `agg_tok_s`.) Cross-site like-for-like: canonical prompt
~1024 tokens in / 256 out, seeds fixed.

## Harness: servekit bench

Since 2026-Q1 the measurement engine is
[`servekit bench`](https://github.com/eth-easl/servekit) (pure stdlib —
urllib/re/json — so it runs **in any compute-allocation container with no pip
install**, which is what made the old `--nowarm` aiohttp variant necessary on
no-egress JSC nodes). `cbench.sh` wraps it into the cookbook protocol below.

Get the runner (once, on a node with egress; no installation needed):

```bash
git clone --depth=1 https://github.com/eth-easl/servekit <shared-fs>/servekit
export SERVEKIT_DIR=<shared-fs>/servekit     # cbench runs it module-style from here
```

What `servekit bench` gives us for free: a pre-benchmark **correctness probe**
(qualitative — catches a broken template/parser before GPU-hours are spent),
readiness polling, exact output lengths via `ignore_eos`, and a raw JSON
report per level (`--out`) that can be **merged into a `servekit profile`
cold-start report** via `--into`. What it can't do: send auth/custom headers
(so key-gated endpoints like `https://api.opentela.ai/...` can't be benched
directly — target the engine or a local head instead), and one concurrency
per invocation (the wrapper handles that).

## Protocol

0. **Smoke** — see the *test-opentela-llm* skill (or any OpenAI client):
   one question, verify a coherent complete answer. This catches a broken
   chat template before burning GPU hours.
1. **Warmup, discarded** — `cbench.sh` runs C=4/n=8 first and does not
   report it. The first measured level pays for KV-cache allocator growth,
   lazy JIT, and cuDNN autotune at that concurrency; never quote it.
2. **Sweep** — see *Levels* below. `cbench.sh` runs the correctness probe at
   the first measured level, saves one raw report per level
   (`cbench_<label>_<ts>_c<C>.json`), then prints the summary table and
   writes `cbench_<label>_<ts>.summary.jsonl` (the full reports — throughput
   stats, `latency_s{mean,p50,p99,max}`, errors, correctness samples —
   archival material for quoted tables).
3. **NN > 1: probe inter-node collectives** — sweep throughput can look
   right while grads/activations crawl on a degraded path (e.g. SHARP
   half-engaged). Run `nccl_sharp_probe.sh` (or the apptainer variant) once
   per job before believing multi-node numbers. It uses a stable slice
   (first 3 nodes × 2 GPUs) so first-call effects don't pollute it.
4. **Report** (see below).

### Levels

```
canonical:  1:8  8:32  16:48  32:64  <knee>:<knee>
```

- n must be ≥ 2×C and a multiple of C (the wrapper rounds up if needed).
- Keep extending levels until agg throughput flattens (<5% gain level over
  level); that's the saturation knee. Quote at minimum: **C=1**, **the
  first level past the knee**, **the knee**, and where possible a level
  sized to the engine's `max_running_requests`.
- **A number at C=1 is never the headline figure for TP>1 or EP>1**
  (aggregated throughput under load is the honest shape). Example
  (Kimi-K3, TP4/PP8, JSC job 2980476): with `moe_a2a_backend=none` the
  engine reports **0.76 tok/s at C=1** and the smoke test still returns
  perfect answers — correct but useless in the single-stream regime; the
  same runtime reaches ~425 tok/s aggregate at C≈64 by level, vs the
  TP4/PP8 ceiling of 542 tok/s at C_sub=52 (`max_running_requests=52`,
  jsc job 3018918). With deepep (the default): 38.8 tok/s at C=1, 486 at
  C=32, 456 at C=64. Deepep only pays off above C≈32; a two-point C=1/C=64
  report would have missed all of this.
- Sweep shape tells you where you are: **linear to the knee** → compute-bound
  (healthy); **knee ≪ max_concurrency** → KV/attention-bound or memory-cap;
  **flat from C=1** → transport- or sync-bound — re-probe SHARP, check
  `moe_a2a_backend` / EP overlap, re-check NVLink.

Run it:

```bash
bash meta/bench/cbench.sh http://<head-ip>:<port> \
  "1:8 8:32 16:48 32:64 52:52" 768 256 --label <model>-<site> --out-dir .
```

(jobs: run from a login/head shell against `http://127.0.0.1:$SERVE_PORT`,
or let the recipe do it — see *Auto-bench in recipes* below.)

### Reasoning models

`servekit bench` drives `/v1/completions` (no chat template applied), so it
measures **raw decode throughput without reasoning tokens** — good for
comparability. TTFT is measured time-to-first-token *visible to a streaming
client*, which for reasoning models includes the reasoning preamble. A
`/v1/chat/completions`-path measurement (e.g. sglang's own `bench_serving`)
will differ; never mix the two within one table, and state which path was
used. Kimi-K3 is cap at `reasoning_effort="high"` (no `"none"`); see the JSC
K3 README for how effort moves the Pareto curve.

## Cold-start profiling: servekit profile

The flagship K3 recipes (clariden, jsc) wrap `sglang serve` with
`servekit profile --out $RUNDIR/coldstart.json --timeout $HEALTH_TIMEOUT --`,
which parses the engine's own log line-by-line into a per-phase startup
breakdown (phases are normalized across sglang/vllm so reports are diffable)
and writes `$RUNDIR/coldstart.node<RANK>.json` when each node reports ready.
The engine log passes through unmodified into the job log.

Two safety properties to remember before wrapping other engines:

- **Timeout kills.** On `--timeout` expiry the profiler *terminates the
  engine*. The recipes pass `--timeout $HEALTH_TIMEOUT` (a value that
  already bounds a successful start with margin) — never shorten it.
- **Multi-node readiness depends on sglang's worker log line.** Non-head
  ranks resolve on `Dummy health check server started`. If a future image
  stops printing it, workers will run unblocked until the timeout and then
  be killed by the profiler — check one workernode's
  `coldstart.node<R>.json` on the first run after an image bump.

Opt-in for other recipes (euler, beverin, dgx): stage the checkout, then
wrap the engine command the same way (`profile` detects sglang/vllm from the
command tokens; an exotic launcher it can't classify gains nothing).

## Auto-bench in recipes

Both K3 recipes run a fixed verification bench after health and **before**
otela registration (`SERVEKIT_BENCH=1`, C=16 n=64 by default), inside the
engine container via `srun --overlap`, merged into the profile report
(`--into $RUNDIR/coldstart.node0.json`, falling back to
`$RUNDIR/bench.json`). Registration is deliberately last so otela traffic
can't pollute the measurement. Results land in the job log + the JSON
artifacts. `SERVEKIT_BENCH=0 sbatch ...` skips it.

## Site notes (in-container)

Compute nodes often have no Internet and minimal images. Because the bench
is stdlib-only now, the in-container incantation is just module-style with a
PYTHONPATH — no aiohttp venv needed:

```bash
apptainer exec --bind /e/scratch:/e/scratch --bind /e/data1:/e/data1 "$IMAGE" \
  env PYTHONPATH=$SERVEKIT_DIR/src python3 -m servekit.cli bench \
  --url http://127.0.0.1:30000 --requests 32 --concurrency 8 \
  --input-len 768 --output-len 256 --out /e/scratch/.../bench_c8.json
```

(For clariden/Pyxis use `srun --environment=<edf> env PYTHONPATH=... python3 -m servekit.cli bench ...`.)

| Site | bench endpoint |
|---|---|
| jsc | `http://127.0.0.1:30000` from the head node (job env) |
| beverin | `http://<head-node>:8080` |
| alps via api.opentela.ai | `v1/service/llm/v1` is **bearer-token-gated → servekit bench can't reach it**; bench the engine or a local head |

Ollama (dgx-spark): `ignore_eos` is ignored, so per-request lengths vary —
numbers still useful for smoke/regression, but never compare Ollama
`served_tok_s` against a vllm/sglang run from the same table.

## Files

| File | Role |
|---|---|
| `cbench.sh` | **Default harness** — servekit-driven sweep enforcing the protocol above |
| `cbench_report.py` | aggregates per-level JSONs → summary table + `.summary.jsonl` |
| `servekit_env.sh` | runner resolution: `$SERVEKIT_DIR` checkout (module-style) or PATH `servekit` |
| `oneshot.py` | single-/low-request completion; returns `completion_tokens / wall_s` |
| `nccl_sharp_probe.sh` | NN>1 sanity: all_reduce on a stable 3×2 slice catches SHARP off-path |
| `nccl_sharp_probe_apptainer.sh` | same, for apptainer sites (Downloads NCCL tests when the image lacks them) |
| `bench.py` / `bench_nowarm.py` | LEGACY aiohttp sweep superseded by `cbench.sh`; kept for historical `bench_*.jsonl` comparisons. Its `--input-len` is tokens; servekit's is words (~1.4 tok/word on these prompts — hence `768` ≈ 1024-token canonical). |

## Reporting checklist

In every PR / README / Slurm-output clipboard that quotes numbers:
- the raw `cbench_*.json` / `.summary.jsonl` is committed as `bench_<jobid>.jsonl`
  next to the recipe (or the servekit `<ts>.summary.jsonl`, renamed);
- quote agg_out_tok_s with **job ID, config (`CG_DECODE`, backends, EP/PP/TP,
  `MEM_FRAC`, context, quantization), concurrency, `in_words→realized tokens`/
  output lens, harness ("servekit bench via cbench.sh"), and date;
- commit the exact env at job time (`env | grep -E 'SGLANG|NCCL|HICACHE|MOE|SERVEKIT' > $RUNDIR/env.txt`);
- when comparing against a number produced by `bench.py`: same site, same
  engine+version, same C levels, prompt token length matched to ±5%.
