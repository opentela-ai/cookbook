---
name: debug-correctness-bug
description: Debug a served LLM whose outputs are WRONG while the service is up — garbage tokens, weakly-peaked logprobs, context-blind answers, numerical drift, or a quality regression on any site/engine. Use when the user says the model "serves but produces garbage/wrong output", reports a correctness or numerics bug, or asks to bisect which layer/kernel corrupts activations. Not for crash/OOM-at-boot (fix that config bug first) or throughput questions (use meta/bench).
---

# Debug a correctness bug on a served LLM

Toolkit: `meta/tools/debugger/` (all paths below are relative to the cookbook
root). Operator tools are stdlib-only — they run on no-egress compute nodes,
over SSH, or locally. Canonical deep detail: the toolkit
[README](../../../meta/tools/debugger/README.md); worked real-world example:
[glm-53-flash investigation](../../../deployments/llm/beverin/glm-53-flash/CORRECTNESS_BUG_INVESTIGATION.md).

## Step 0 — classify the failure, start the journal

Service up + output wrong → this skill. Dead/OOM/assert at startup → that is a
config bug; a crash BEFORE the first forward produces no activations to
bisect, so fix it first and do not log it as progress on the numerics bug.

Start the journal before any experiment — every verdict is recorded the moment
you get it, with the test that produced it. The ruled-out-with-evidence list
is what keeps a multi-day bisect sane and becomes the fix's commit message:

```bash
python3 meta/tools/debugger/journal.py new CORRECTNESS_BUG_INVESTIGATION.md \
  --title "<model> on <site>: <one-line symptom>" \
  --symptom "<exact signature once probe.py has run>"
```

## Phase 0 — characterize with probe.py (never skip)

```bash
python3 meta/tools/debugger/probe.py --url http://<NODE>:<PORT> --model <org/model-name> --json
```

Exit 0 = sane; exit 1 = read the failing probes as a SIGNATURE (with `--json`,
stdout is pure JSON):

| Signature | Suspect class |
|---|---|
| confident wrong answers, coherent | template/sampler/quantization drift |
| real tokens, weakly-peaked logprobs (≈ −3.5…−4.5), deterministic, context-sensitive | forward-pass numerics (novel kernel) — the classic fingerprint |
| context-blind (same last token ⇒ same output regardless of prefix) | context never reaches the output — plumbing/indexer |
| nondeterministic across identical requests | NOT pure numerics — race/allocator/comms |
| near-uniform distribution | weights/lm_head broken, residual destroyed early |

## Phase 1 — rule out the broad classes with primitive tests

One small in-container script per class, verdict journaled each time
(`journal.py ruled-out INV.md "<class>" --evidence "<exact test + result>"`):
weight loading (scan load section), quantize/dequant at load (run the server's
exact path vs a bit-exact reference on one real tensor), prompt template (raw
`/v1/completions` AND chat), context plumbing (probe C). After this phase you
should write ONE sentence: "a forward-pass numerical error in kernel family X,
not a load/format/template issue."

## Phase 2 — no-reference per-layer bisect (lstats)

Names the broken kernel family WITHOUT a reference forward. Gate is
`DBG_LSTAT=1`; the module self-installs on import (wire it from the site
`sitecustomize.py` / sbatch heredoc):

```python
import os, sys
_d = os.environ.get("DBG_TOOL_DIR")
if _d and _d not in sys.path: sys.path.insert(0, _d)
if os.environ.get("DBG_LSTAT", "0") == "1": import lstats      # noqa: F401
if os.environ.get("DBG_CAPTURE", "0") == "1": import capture   # noqa: F401
```

Engine env (export in the sbatch; see gotcha below):

```bash
DBG_TOOL_DIR=$ROOT/meta/tools/debugger
DBG_TARGET=sglang.srt.models.glm5_next.Glm5NextModel  # dotted CLASS to instrument
DBG_LAYERS_ATTR=language_model.layers                 # path from model to layer list
DBG_LSTAT=1 DBG_DIR=/scratch/dbg DBG_TAG=<site>_v1 DBG_LAYER_TYPES=la,la,dsa,moe,...
DBG_MIN_TOKENS=1500                                   # ABOVE warmup length
```

Read `[DBGSTAT]` lines: the first layer whose OUT explodes / collapses / NaNs
names the family (from `DBG_LAYER_TYPES`). Warmup forwards outside
`[DBG_MIN_TOKENS, DBG_MAX_TOKENS]` do NOT consume the latch, so the stats land
on the first real forward. All env vars: toolkit README.

## Phase 3 — reference bisect: capture → diff

Reference = the same model serving correctly elsewhere (native kernels), or a
pure-torch route of the suspect op. Run `DBG_CAPTURE=1` (same wiring, add
`DBG_CAPTURE_MODE=layers`, then `components` for the drill-down) on BOTH sides
with the SAME deterministic probe prompt, then:

```bash
python3 meta/tools/debugger/diff.py layers <test_dir> <ref_dir>          # first divergent layer
python3 meta/tools/debugger/diff.py components <test_dir> --layer N --ref <ref_dir>  # broken component
python3 meta/tools/debugger/diff.py summary <dir>                        # manifest-only, no torch
```

`diff` REFUSES captures whose prompt digests differ (exit 3) — a diff against
a different prompt is meaningless. The verdict line prescribes the next
command. Follow it: layer → component → kernel.

## Phase 4 — isolate the kernel, fix by torch drop-in

Standalone primitive test of the suspect kernel vs a reference in the same
container (one expert's FP8 GEMM vs per-block dequant; one pre-norm vs torch).
Proven fix shape: route the broken vendor/accelerator path to a plain torch
implementation behind a device gate — correct first, fast later (precedent:
clariden commit `b8d5296`, GLM-5.3 DSA → SDPA).

## Hard rules (and what breaks without them)

- **Budget by cold starts, not probes** (~20 min each): batch every probe of a
  phase into ONE job submission. One experiment per job wastes hours.
- **One variable at a time, same prompt both sides**: the capture latch lands
  on the same forward only if the prompt matches (that is what the identity
  gate checks).
- **Identical diagnostic code on both machines**: run it from one copy of
  `meta/tools/debugger/` (push/rsync), never copy-pasted into a recipe —
  otherwise the cross-machine diff compares two different tools.
- **Never overwrite a capture**: new `DBG_TAG=_v<N>` each run; disk is cheaper
  than a cold start.
- **EDF/container env clobbers sbatch exports**: values baked into the
  registered environment silently override your `export`s (and `--environment`
  replaces PYTHONPATH entirely). Pass overrides inline:
  `srun --environment=<env> env PYTHONPATH="$PP" MYVAR=1 bash engine.sh`, and
  verify the patch landed (print `module.__file__`, grep the patched file).
- **Probe the server over HTTP**: `nsenter` into the live PID is usually
  blocked (no CAP_SYS_ADMIN). In-container python:
  `srun --jobid=<JOB> --overlap --gres=none -w <NODE> --environment=<env> -n1 python3 …`.
