---
name: profile-kernel-bottleneck
description: Profile a served LLM or a single kernel that is CORRECT but SLOW — low tokens/s, high per-request latency, low GPU utilization, a kernel patch needing before/after numbers, or "which kernel dominates the forward". Use when the user says the model "serves but is slow", asks where the milliseconds go, to rank kernels or roofline-classify (memory- vs compute- vs launch-bound), or to compare torch vs vendor kernel performance. Not for wrong/garbage output (use the debug-correctness-bug skill), end-to-end throughput/cold-start benchmarking (use meta/bench), or crash/OOM at boot (config bug).
---

# Profile kernel performance and find the bottleneck

Toolkit: `meta/tools/profiler/` (paths below are relative to the cookbook
root). Method in one line: **rank kernels in the real forward → microbench the
suspect in isolation → compare before/after**. Canonical deep detail: the
toolkit [README](../../../meta/tools/profiler/README.md). Correctness
counterpart: [debug-correctness-bug](../debug-correctness-bug/SKILL.md) — a
perf fix that changes outputs is a correctness bug, not a perf problem.

## Step 0 — classify the question

Correct output but slow / want utilization → this skill. Wrong output → the
debug skill (their Phase 0 probe is also your post-fix regression check).
End-to-end tok/s, cold start, scheduler or serving-config questions →
`meta/bench/` (`cbench.sh`, `servekit`); cross-node comm specifically →
`meta/bench/nccl_sharp_probe.py`. One kernel or one forward's internals →
proceed.

## Phase 0 — know the substrate (device_info)

```bash
python3 meta/tools/profiler/device_info.py --json
```

Prints the device and the NOMINAL peak row (GB/s, dense-bf16 TFLOP/s) the
roofline math will use. If your dtype is fp8/fp32 or your clocks are capped,
pass `--peak-flops`/`--peak-bw` to microbench later. Never quote a utilization
without stating which peak it is against.

## Phase 1 — rank the forward (forward_breakdown)

Write a bench file whose `op()` is the REAL serving-shaped forward — same
batch × seq × experts as production. The C=1 trap from `meta/bench` applies:
the wrong shape is a different kernel story, not a smaller one. Contract
(`examples/gemm_bench.py` and `examples/elementwise_bench.py` are templates):

```python
def make_inputs(): ...            # built once
def op(inputs): ...               # the forward/region to profile
def flops(inputs) -> float: ...   # optional
def bytes(inputs) -> float: ...   # optional
```

```bash
python3 meta/tools/profiler/forward_breakdown.py my_forward_bench.py \
  --iters 3 --top 25 --json --out breakdown_<tag>.json
```

Reading, in order:

1. **busy% of measured wall < ~60** → the bottleneck is BETWEEN kernels (CPU
   dispatch, launch overhead, sync/collective waits). Kernel microbenches will
   NOT help. Read the gap list — it names the kernel before and after each
   hole; a gap that starts after an NCCL/RCCL kernel is a comm wait →
   `meta/bench/nccl_sharp_probe.py`. Fix dispatch/sync, not kernels.
2. **One family ≫ others** → that is your suspect; microbench its top kernel
   (Phase 2).
3. Kernel detail missing or thin → CUDA graphs are hiding per-kernel timing;
   profile with graphs disabled.

## Phase 2 — isolate the suspect (microbench)

```bash
python3 meta/tools/profiler/microbench.py my_kernel_bench.py \
  --iters 200 --warmup 50 --json --out res_<tag>.json
```

`flops()`/`bytes()` must model the IDEAL traffic of the computation (GEMM:
`2*M*K*N` flops; one read of each operand + output), not what the bad
implementation happens to move — utilization then measures distance to that
ideal. The BOUND verdict prescribes the next move:

| Bound | Next move |
|---|---|
| memory-bound (mem util is the max) | dtype, layout/coalescing, packaging; recompute vs re-read tradeoffs |
| compute-bound but LOW util | tile shape, occupancy, dtype path, tail effect |
| launch-bound (time ≈ launch overhead) | fuse kernels, batch the work, or CUDA graphs (never across cross-node collectives on Slingshot) |
| unknown | add `bytes()`/`flops()` to the bench file, or pass `--peak-*` |

## Phase 3 — compare (before/after, torch vs vendor, site vs site)

```bash
python3 meta/tools/profiler/compare.py res_torch_ref.json res_vkernels_v3.json
```

Only as honest as the pairing: same shapes, same inputs, same tool, tagged
files. Cross-device comparisons → the utilization % column is the fair one;
speedup across different GPUs is marketing.

## Hard rules (and what breaks without them)

- **Steady state only**: every tool warms up first; quote p50/mean, never the
  first iterations.
- **Profiler overhead is real**: `forward_breakdown` ranks and finds gaps;
  `microbench` (event-timed) gives absolute kernel numbers. Never quote
  in-profiler wall time as serving tok/s — that is what `meta/bench` measures.
- **One change at a time, tagged artifacts**: `res_<tag>.json` / `breakdown_<tag>.json`
  like the debugger's capture tags. After each fix, RE-RANK the forward —
  family shares reshuffle and the new top is rarely the one you just fixed.
- **mean ≫ p50 or high std** → interference (other ranks, NCCL threads) or
  clock throttling; rerun on a quieter allocation before believing it.
- **Deep-dive counters only after the portable answer**: nsys/ncu (NVIDIA),
  rocprof/omniperf (AMD) when "which kernel + which class" is known but
  per-SM counters are needed. The toolkit answers the first two questions on
  any site without extra deps.
- **After any kernel swap, re-run the debug skill's probe** (exit 0 = sane):
  perf changes to a kernel can silently break numerics, and "faster but
  wrong" is a correctness bug again.
