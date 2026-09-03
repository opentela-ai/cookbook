# meta/tools/debugger — agent toolkit for correctness bugs

Reusable, model-agnostic tooling for the failure class this cookbook hits most
often in a serving campaign: **the service boots, `/health` is green, the
engine log is clean — and the output is wrong** (garbage tokens, weakly-peaked
logprobs, context-present-but-wrong answers, or subtly degraded quality).

It distills the methodology that cracked the GLM-5.3-Flash MI300A garbage-output
bug (beverin bisect, Aug–Sep 2026; see
`deployments/llm/beverin/glm-53-flash/CORRECTNESS_BUG_INVESTIGATION.md` and the
model-specific harness in [`meta/diag/glm53/`](../../diag/glm53/)) into
primitives any recipe/model can reuse. `meta/diag/glm53` remains the
GLM-5.3-specific instance; this directory is the general toolkit.

Agents: the skill `.agents/skills/debug-correctness-bug/SKILL.md` is the
step-by-step orchestration of this toolkit; this README is the canonical
deep-dive (env var tables, wiring, site gotchas).

Everything an operator or agent runs directly is **stdlib-only** (runs
zero-install on no-egress compute nodes, over a bare SSH session, or from a
laptop). Only the in-engine hook modules (`capture.py`, `lstats.py`) need
torch, because they run inside the engine process on a compute node.

## The method — run the phases in order, journal as you go

Do not guess kernels. Each phase is cheap and either names the next phase or
produces a recorded verdict. Record **every** verdict with evidence in the
investigation journal (`journal.py new` + `ruled-out`) — the GLM-5.3
investigation was tractable precisely because every dead end was written down
with the test that killed it.

### Phase 0 — Characterize the failure (probe.py)

Run `probe.py` against the live server **before** touching anything. The
failure *signature* selects the suspect class:

| Signature (from probes) | Points at |
|---|---|
| confident wrong answers, coherent | prompt/template handling, sampler, quantization drift |
| real tokens, weakly-peaked logprobs (~ −3.5…−4.5), context-present-but-wrong | forward-pass numerical corruption (a novel kernel), NOT load/template |
| uniform-ish distribution (logprobs ≈ log V) | broken weights / lm_head / residual destroyed early |
| copies a recent input token to the wrong target | attention/indexer path (attendee set wrong), e.g. sparse-attention kpool gather |
| deterministic garbage | numerical kernel; nondeterministic garbage | race/allocator/communication |

Two decisive cheap probes from the GLM-5.3 campaign, both built into `probe.py`:
**context sensitivity** (same last token, different prefixes → outputs MUST
differ; else context never reaches the output) and **determinism** (same
prompt twice → identical; else it's not a numerics bug at all).

### Phase 1 — Rule out the broad classes with primitive tests

Each class is killed or kept with one small in-container script and a recorded
verdict — not with reasoning. The campaign's ruled-out list (with the exact
test that closed each):

| Class | Primitive test | Verdict format |
|---|---|---|
| weight loading | scan load section: missing keys / shape mismatch / OOM | `ruled-out "weight loading" --evidence "no missing keys..."` |
| quantize/dequant at load | load one real tensor, run the server's exact dequant path vs a bit-exact reference in the same container | constant-ratio check proved e4m3fnuz normalize correct on MI300A |
| prompt/template | same request raw `/v1/completions` AND template-applied `/v1/chat/completions` | both garbage → template exonerated |
| context plumbing | probe.py context-sensitivity result | context present but wrong → forward pass |

After Phase 1 you should be able to write one sentence: *"a forward-pass
numerical error in kernel family X, not a load/format/template issue."*

### Phase 2 — No-reference per-layer bisect (lstats.py)

Before hunting for a reference forward, run `lstats.py` in the engine: it
prints per-layer residual IN/OUT stats for the first real forward and flags
the first layer whose stats explode / collapse / go NaN. **That layer index
names the broken kernel family** (whatever runs in that layer's block) with no
reference needed. Gotcha learned the hard way: an assert *before* the layer
loop (e.g. a page_size/kpool config assert) kills the job with no forward at
all — a crash-before-forward is a config bug, not your numerics bug; fix the
crash, then re-run.

### Phase 3 — Reference bisect (capture.py + diff.py)

Capture the first-forward I/O of every layer (then every component of the
first divergent layer) and diff against a reference:

- **cross-machine**: the same model serving correctly elsewhere (e.g. native
  kernels on another GPU) is the best reference — the identical deterministic
  probe prompt must fire on both machines;
- **pure-torch**: route the suspect op to a plain torch reference and capture
  both.

`diff.py` refuses to compare captures whose prompt token ids differ (the
identity gate) — a diff against a different prompt is meaningless. Drill in:
first divergent *layer* → first divergent *component* → that is the kernel.

### Phase 4 — Isolated kernel primitive test, then the fix shape

Extract the suspect kernel and test it standalone against a reference in the
same container (one expert's FP8 GEMM vs per-block dequant in bf16; one
pre-norm vs torch RMSNorm). Large error → you have the bug in one file.

Proven fix shape (clariden commit `b8d5296`): when a vendor/accelerator kernel
is numerically wrong, **route that path to a plain torch implementation**
(slower, no sparse savings) and gate it behind a device check — correctness
first, performance later.

## Tools

| File | Role | Runs where | Needs |
|---|---|---|---|
| `probe.py` | live-server sanity probes (coherence, determinism, context-sensitivity, repetition, concurrency, prefix) | anywhere, via HTTP | stdlib |
| `lstats.py` | no-reference first-forward per-layer residual stats + first-bad-layer verdict (`[DBGSTAT]` log lines + manifest) | in-engine, self-installing | torch, `DBG_LSTAT=1` |
| `capture.py` | first-forward per-layer / per-component I/O dump + manifest with input-ids identity digest | in-engine, self-installing | torch, `DBG_CAPTURE=1` |
| `diff.py` | THE diff tool: `layers` (identity-gated cross-machine bisect), `components` (drill into a layer vs a reference), `summary` (manifest-only health, torch-free) | anywhere | torch for `layers`/`components` |
| `journal.py` | investigation journal: `new` (proven template) + `ruled-out` / `suspect` / `verdict` / `next` appenders | anywhere | stdlib |
| `hook.py` | `run_after_import(target, on_loaded)` — the one import-hook primitive the in-engine modules share | in-engine | stdlib |

All in-engine gates use the `DBG_` env prefix and are **self-installing**:
importing the module is the only wiring step. All hooks are first-forward-only
(handles removed on latch close), rank-gated (`DBG_RANKS`, default `0`), and
try/except-wrapped so they can never break the forward itself.

## Wiring an engine (sitecustomize / heredoc dispatcher)

```python
# inside sitecustomize.py (or the sbatch heredoc equivalent)
import os, sys
_d = os.environ.get("DBG_TOOL_DIR")           # exported by the sbatch:
if _d and _d not in sys.path:                 #   DBG_TOOL_DIR=<cookbook>/meta/tools/debugger
    sys.path.insert(0, _d)
if os.environ.get("DBG_LSTAT", "0") == "1":
    import lstats                             # noqa: F401  self-installing
if os.environ.get("DBG_CAPTURE", "0") == "1":
    import capture                            # noqa: F401  self-installing
```

```bash
# engine-side env (in the sbatch, BEFORE the srun line)
DBG_TOOL_DIR=$ROOT/meta/tools/debugger
DBG_TARGET=sglang.srt.models.glm5_next.Glm5NextModel   # dotted class to instrument
DBG_LAYERS_ATTR=language_model.layers                  # path from model to the layer list
DBG_CAPTURE=1 DBG_CAPTURE_MODE=layers DBG_DIR=/scratch/dbg DBG_TAG=beverin_v1
DBG_MIN_TOKENS=1500                                    # above warmup length
DBG_CAPTURE_PROBE=1                                    # fire one deterministic prefill after ready
```

Then, from anywhere:

```bash
python3 meta/tools/debugger/probe.py --url http://NODE:30001 --model org/model-name
python3 meta/tools/debugger/diff.py layers  /scratch/dbg/test_v1 /scratch/dbg/ref_v1
python3 meta/tools/debugger/diff.py components /scratch/dbg/test_v1 --layer 3 --ref /scratch/dbg/torchref_v1
python3 meta/tools/debugger/diff.py summary /scratch/dbg/test_v1        # no torch needed
python3 meta/tools/debugger/journal.py ruled-out INV.md "FP8 load normalize" \
    --evidence "12-value bit-exact ratio test in-container: constant 2.0 bias-only"
```

## Operating rules for the agent

1. **Budget by cold starts, not by probes.** A cold start can cost ~20 min;
   batch every probe of a phase into ONE job submission (capture + probe +
   the primitive test together), never one experiment per job.
2. **One variable at a time**, and the same deterministic probe prompt on both
   sides of every comparison — the capture latch only lands on the same
   forward if the prompt matches.
3. **Identical code on both machines.** Diagnostic code lives here (one copy),
   rsynced/pushed, never copy-pasted into a recipe — the cross-machine diff is
   only meaningful if capture and diff behave identically.
4. **Tag discipline**: every capture dir gets `DBG_TAG=<site>_<what>_v<N>`;
   never overwrite a capture — disk is cheaper than a 20-minute cold start.
5. **A crash is not the bug you are hunting.** If the job dies before the
   first forward, you have a config/env bug (see gotchas) — record it, fix it,
   continue. Don't reinterpret it as progress on the numerics bug.
6. **Write the journal entry when you get the verdict**, not later. The
   ruled-out list is what keeps a multi-day bisect sane.

## Site gotchas (learned in the field)

- **EDF/container env overrides sbatch exports.** Values baked into the
  registered environment (e.g. `SGLANG_USE_AITER=0`) silently clobber your
  `export`. Vars that survive are only those the EDF does not list. Fix: pass
  overrides inline on the srun line —
  `srun --environment=sglang-rocm env PYTHONPATH="$PP" MYVAR=1 bash engine.sh`.
  The same trick is required for `PYTHONPATH` itself: `--environment` REPLACES
  it with the container path, so an overlay must be forced inline (verified by
  printing `sglang.__file__` and grepping the patched file in the job log).
- **In-container python**: enter the running allocation with
  `srun --jobid=JOB --overlap --gres=none -w NODE --environment=<env> -n1 python3 …`
  — without `--environment`, `python3` may resolve to a host python with no
  torch. `nsenter` into the live server PID is usually blocked (no
  CAP_SYS_ADMIN): probe the server over HTTP, don't plan on attaching to it.
- **Stale EDF workdirs**: a registered environment may pin a deleted workdir;
  `ln -sfn <real> <stale>` unblocks, or re-register.
- **Warmup forwards**: engines run dummy forwards first; set `DBG_MIN_TOKENS`
  above the warmup length and use `DBG_CAPTURE_PROBE` to fire one deterministic
  prefill so the latch lands on the same forward everywhere.

## Journal template

`journal.py new` writes the structure the GLM-5.3 investigation converged on:
symptom characterization → RULED OUT (with evidence) → narrowed bug statement →
ranked suspects → next steps → operational notes. Keep it updated; it doubles
as the PR/commit-message source when the fix lands.
