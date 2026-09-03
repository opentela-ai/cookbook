# meta/tools/profiler — kernel-level performance toolkit

> Agent skill: **`.agents/skills/profile-kernel-bottleneck/SKILL.md`** — the
> orchestration layer over this toolkit (when to profile, what each verdict
> prescribes). This README stays the canonical deep-dive.

The perf-side sibling of [`meta/tools/debugger/`](../debugger/): the debugger
answers *why is the output wrong*, this toolkit answers *where do the
milliseconds go* — utilization, per-kernel cost, and the bottleneck class of
serving forwards and individual kernels, on any site (CUDA/ROCm via torch /
kineto; portable fallbacks everywhere).

Method (each phase hands its output to the next):

1. **Identify the substrate** — `device_info.py` prints the device and the
   NOMINAL peak the roofline math will use (override with `--peak-*`; never
   quote a utilization without stating the peak).
2. **Break down the forward** — `forward_breakdown.py` on a bench file whose
   `op()` is a serving-shaped forward (SAME batch/seq as production — the
   C=1 trap from meta/bench applies here too) ranks raw kernels, rolls them
   into families, and reports GPU busy% + gap analysis.
3. **Microbench the suspect kernel** — `microbench.py` times it in isolation
   (GPU events, warmup + steady state) and roofline-classifies it:
   memory-bound / compute-bound / launch-bound, with achieved % of nominal
   peak.
4. **Compare** — `compare.py` puts two result JSONs side by side (torch vs
   vendor kernel, before/after a patch, site vs site — same discipline as the
   debugger's cross-machine diff: same shapes, same inputs, tagged files).
5. **Deep dive only if needed** — the portable numbers above answer "which
   kernel and what class"; for per-SM counters reach for the platform tools
   (NVIDIA: `nsys`, `ncu`; AMD: `rocprof`, `omniprof`-style omniperf).

## Bench file contract

A plain python file (examples in `examples/`):

```python
def make_inputs(): ...            # built once
def op(inputs): ...               # the kernel/region to time
def flops(inputs) -> float: ...   # optional, nominal math ops per op()
def bytes(inputs) -> float: ...   # optional, nominal DRAM traffic per op()
```

`flops`/`bytes` power the roofline math — model the IDEAL traffic of the
computation (e.g. GEMM: `2*M*K*N` flops, one read of each operand plus the
output), not what a bad implementation happens to move; the utilization then
measures how close the kernel is to that ideal.

## Reading the results

| Signal | Meaning / next step |
|---|---|
| busy% < ~60 (breakdown) | bottleneck is BETWEEN kernels: CPU dispatch, launch overhead, sync waits. Kernel microbenches won't help; read the gap list (what runs before/after each hole). Classic: collective waits → see `meta/bench/nccl_sharp_probe.py`. |
| one family ≫ others | drill: microbench that family's dominant kernel |
| memory-bound kernel (high GB/s util) | dtype/layout/coalescing/packaging; recompute vs re-read tradeoffs |
| compute-bound but LOW util | tile shape, occupancy, dtype path, tail effect |
| launch-bound (time ≈ launch overhead) | fuse kernels, batch the work, or CUDA graphs (note: cross-node collectives on Slingshot cannot be graph-captured — see the root README) |
| mean ≫ p50 / high std | interference (other ranks, NCCL threads) or clock throttling — rerun quieter |

## Operational rules for the agent

- **Steady state only**: every tool warms up first; quote p50/mean of the
  steady iterations, never the first runs.
- **Serving-shaped inputs**: a GEMM at the wrong M is a different kernel
  story. Match batch × seq × experts to what production actually issues.
- **One change at a time, tagged artifacts**: `--out res_<tag>.json` like the
  debugger's capture tags; `compare.py` is only as honest as the pairing.
- **Profiler overhead is real**: use `forward_breakdown` for RANKING and the
  gap structure; use event-timed `microbench` for absolute kernel numbers.
- **CUDA graphs hide per-kernel timing** on several stacks: profile with
  graphs disabled if the breakdown comes back empty/thin.

## Tools

| File | Role | Needs |
|---|---|---|
| `device_info.py` | device identity + nominal peak row (`--json`) | torch |
| `forward_breakdown.py` | profiled forward → ranked kernels, family rollup, busy%, gap analysis (`--trace` keeps the chrome trace) | torch (CUDA for kernel events) |
| `microbench.py` | isolated kernel timing + roofline utilization + bound classification | torch |
| `compare.py` | side-by-side two result JSONs, speedup + utilization delta | stdlib |
| `harness.py` | shared bench-file loader + event timing + launch-overhead probe | torch (lazy) |
| `examples/` | `gemm_bench.py` (compute-bound), `elementwise_bench.py` (memory-/launch-bound demo) | torch |

Results are JSON (`--json` / `--out`): `time_ms` stats, `tflops`/`gbps`,
`util_pct`, `bound`, and for breakdowns `families`/`top`/`gaps` — pipe-safe
for agent parsing. Exit 0 on success; failures exit non-zero with the fix in
the message (missing bench def, no torch, profiler unsupported).
