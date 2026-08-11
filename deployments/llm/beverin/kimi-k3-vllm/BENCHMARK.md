# Kimi-K3 ROCm Benchmark — Beverin (MI300A / gfx942)

**Date:** 2026-08-11
**Model:** `SwissAI-Research/moonshot/kimi-k3-rocm` (dummy weights)
**Engine:** vLLM 0.1.dev19253+g5f76ae224.rocm723
**Topology:** TP=8 PP=3 across 6 nodes (24 MI300A GPUs, 128 GiB HBM each)
**Key fixes:**
- `VLLM_ROCM_USE_SKINNY_GEMM=0` — bypasses vLLM's `wvSplitKrc_` kernel
  (`csrc/rocm/skinny_gemms.hip:1769`) which asserts `false` on gfx942
- `VLLM_ROCM_USE_AITER_MLA=0` — selects TRITON_MLA (Gluon requires gfx950)
- `--load-format dummy` — eliminates page-cache thrash during startup
- `--skip-mm-profiling` — prevents vision-tower OOM during `profile_run()`

## CUDA Graphs: NOT viable on gfx942 with PP=3

Tested across jobs 588153, 588154, 588156 with:
- `ENFORCE_EAGER=0` (CUDA graphs ON)
- `K3_DISABLE_KDA=1` (KDA disabled, only MLA layers)
- `VLLM_ROCM_USE_SKINNY_GEMM=0`
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600` (1-h RPC timeout)
- `--distributed-timeout-seconds 3600` (1-h NCCL timeout)

**Result:** CUDA graph capture succeeds during startup (13–39 s, 0.36 GiB per
worker). However, the **first real decode step after prefill hangs
indefinitely** — the engine reports `Avg generation throughput: 0.0 tokens/s,
Running: 1 reqs` for 30 minutes until the gloo CPU inter-PP communication
times out (`Application timeout caused pair closure`, 1800 s default), then
the EngineCore crashes with `KeyError: 'cmpl-...'`.

**Root cause:** vLLM ROCm + pipeline-parallel + CUDA-graph replay interaction.
The pre-captured decode graph (from a dummy batch during `_dummy_run()`) does
not correctly replay for the first real decode after prefill. The hang is on
the inter-PP `torch.distributed.recv` (gloo), not on JIT (enforce_eager has
142 JIT during inference and completes the first request in 0.8 s).

**Precise mechanism** (source `config/vllm.py:1209`, `config/compilation.py`,
hung-run logs 588153/588156): `KimiK3ForConditionalGeneration` is in the
architecture list that auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, which
forces `compilation_config.mode = NONE`. With `mode=NONE` and empty
`splitting_ops`, the default `FULL_AND_PIECEWISE` cudagraph_mode degrades to
`FULL` for decode — a whole-decode forward graph is captured that contains
the gloo `recv_object` on the PP boundary (`parallel_state.py:838`), which
cannot be replayed, deadlocking at the first real decode.

**Workaround:** Use `ENFORCE_EAGER=1` (no CUDA graphs). PROVEN: 10+ jobs
(588302–588856) served real tokens at up to 415 tok/s aggregate; the
confirming run 589456 (`ENFORCE_EAGER=1 K3_PREFIX_CACHE=0`, dummy weights,
smoke probe) passed 6/6 prompts at ~6 tok/s single-request. Single-request
decode is 7.3 tok/s; aggregate throughput scales via batching (see below).

**`K3_PIECEWISE=1` is CONFIRMED NOT VIABLE** (job 589322). The recipe knob
opts out of breakable (`VLLM_USE_BREAKABLE_CUDAGRAPH=0`) and sets
`mode=VLLM_COMPILE`/`cudagraph_mode=PIECEWISE`. vLLM accepts the config and
auto-populates 15 `splitting_ops`, but `KimiK3ForConditionalGeneration`
carries NO `@support_torch_compile` decorator, so
`compilation_counter.num_models_seen` is never incremented and `vllm.py:2410`
only WARNS ("torch.compile is turned on, but the model ... does not support
it") — torch.compile never wraps the model, no FX graph is produced to
split. At init (`gpu_model_runner.py:5442`) `is_breakable_cudagraph_enabled()`
is False (we set `VLLM_USE_BREAKABLE=0`) → skips `BreakableCUDAGraphWrapper`,
and `PIECEWISE.has_full_cudagraphs()` is False (`PIECEWISE` is a simple enum,
not a `FULL/PIECEWISE` tuple, `compilation.py:83`) → skips
`CUDAGraphWrapper(FULL)`. Net: **NO wrapper is installed → the model runs
EAGER, identical runtime to `ENFORCE_EAGER=1`.** It does NOT validate the
PIECEWISE cudagraph hypothesis. Real PIECEWISE cudagraph requires upstream
`@support_torch_compile` on Kimi-K3 (vLLM/model work).

## Prefix caching: REGRESSION on this image (K3_PREFIX_CACHE=0 default)

Commit 9f3177e added `--enable-prefix-caching` unconditionally. On this
image it is **unconditionally broken** for Kimi-K3: with prefix caching ON,
the KV cache manager selects `HybridKVCacheCoordinator`
(`kv_cache_coordinator.py:886`, whenever `len(kv_cache_groups) > 1`). K3 has
>1 `kv_cache_groups` (Mamba + attention) but they all share ONE KV cache
spec (Mamba recurrent state is managed separately, not via
`kv_cache_groups`), so `verify_and_split_kv_cache_groups` collapses every
group into a single `SpecGroup` and asserts **"HybridKVCacheCoordinator
requires at least two attention groups"** (`kv_cache_coordinator.py:627`).
This fires for ANY config with prefix caching ON — eager (589044), PIECEWISE
(589322), and the FULL-decode-cudagraph path alike — at scheduler init,
BEFORE any cudagraph work. The fix is `K3_PREFIX_CACHE=0` (recipe default;
selects `KVCacheCoordinatorNoPrefixCache`, no assertion), which restores the
proven-working path. Set `K3_PREFIX_CACHE=1` only after the upstream
`HybridKVCacheCoordinator` selection bug is fixed (it should key on
distinct-spec count, not raw group count, and fall back to
`UnitaryKVCacheCoordinator` when all groups share one spec).

## Batching benchmark (ENFORCE_EAGER=1)

All runs use `max_num_batched_tokens=2048–4096`, `--kv-cache-memory-bytes`
matching `max_num_seqs` (1–16 GiB). Prompt length ≈ 30 tokens (short,
isolating decode throughput). No errors, no OOM, no device assertions in any
run.

| max_num_seqs | concurrent | max_tokens | aggregate (tok/s) | per-req decode (tok/s) | per-req latency (s) | speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | 128 | 7.32 | 7.32 | ~5 | 1.0× |
| 8 | 4 | 32 | 17.27 | 2.27 | 14.1 | 2.4× |
| 8 | 8 | 32 | 26.88 | 3.46 | 9.3 | 3.7× |
| 8 | 8 | 128 | 43.00 | 5.47 | 23.4 | 5.9× |
| 8 | 8 | 256 | 48.91 | 6.13 | 41.7 | 6.7× |
| 8 | 8 | 512 | 50.91 | 6.38 | 80.3 | 7.0× |
| 16 | 16 | 128 | 77.46 | 4.87 | 26.3 | 10.6× |
| 16 | 16 | 256 | 81.18 | 5.10 | 50.2 | 11.1× |
| 16 | 16 | 512 | 86.75 | 5.43 | 94.2 | 11.9× |
| 32 | 32 | 128 | 119.69 | 3.75 | 34.2 | 16.4× |
| 32 | 32 | 256 | 134.85 | 4.22 | 60.7 | 18.4× |
| 32 | 32 | 512 | 122.43 | 3.85 | 133.1 | 16.7× |
| 64 | 64 | 256 | 178.09 | 2.79 | 91.7 | 24.3× |
| 64 | 64 | 512 | 206.71 | 3.23 | 158.5 | 28.2× |
| 128 | 128 | 256 | 249.88 | 1.96 | 130.8 | 34.1× |
| 128 | 128 | 512 | 285.35 | 2.23 | 229.5 | 39.0× |
| 256 | 256 | 512 | **414.94** | 1.63 | 314.7 | **56.7×** |

### Key observations

1. **Aggregate throughput scales with `max_num_seqs`**: 7 → 51 → 87 → 135 →
   207 → 285 → 415 tok/s — a **56.7× improvement** over single-request at
   `max_num_seqs=256`.

2. **Per-request decode rate decreases** with more batching (7.32 → 1.63),
   but aggregate throughput keeps climbing. At `max_num_seqs=8`, per-request
   decode is 6.38 (87 % of single-request) — batching is nearly free.

3. **Longer sequences benefit more from batching**: 8×512 = 50.9 tok/s vs
   8×32 = 26.9 tok/s. Prefill overhead is amortized across more decode steps.

4. **Diminishing returns** at high `max_num_seqs`: 8→16 (1.7×), 16→32
   (1.5×), 32→64 (1.5×), 64→128 (1.4×), 128→256 (1.5×).

5. **Engine peak throughput** during 256-concurrent run: 435.2 tok/s
   (momentary, reported by vLLM engine stats).

6. **No crashes or errors** at any `max_num_seqs` — the engine is stable at
   all tested concurrency levels.

### Recommended configurations

| Use case | max_num_seqs | max_tokens | aggregate | per-req latency |
|---|:---:|:---:|:---:|:---:|
| Low latency (interactive) | 1 | 128 | 7 tok/s | ~5 s |
| Balanced | 16 | 512 | 87 tok/s | ~94 s |
| High throughput | 64 | 512 | 207 tok/s | ~159 s |
| Maximum throughput | 256 | 512 | 415 tok/s | ~315 s |

## Caveats

- **Dummy weights** → output is degenerate. Throughput/latency are real
  (same compute path); quality is not measurable.
- **`enforce_eager=True`** → no CUDA graph capture. Per-step kernel launch
  overhead is present. If CUDA graphs are fixed for gfx942 + PP, expect a
  further 1.5–2× improvement in per-request decode rate.
- **Single-node head** (TP=8 PP=3, 6 nodes) — inter-node NCCL all-reduce is
  a bottleneck. A topology with more nodes (TP=8 PP=3 → more PP stages)
  would reduce per-GPU memory but increase inter-PP communication.
- **`max_num_batched_tokens=2048–4096`** limits prefill throughput for long
  prompts. Increasing to 8192+ would speed up long-prompt TTFT.
- **KV cache**: 1–16 GiB (depending on `max_num_seqs`). Real serving with
  long context (up to 1 M tokens) would require much more KV cache memory.
