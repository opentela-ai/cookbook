# Kimi-K3 on Beverin (vLLM, ROCm) → OpenTela

Serve **Kimi-K3** (Moonshot, 2.8 T-param hybrid MoE, mxfp4, 1 M context) on
**Beverin** (AMD MI300A / gfx942, `mi300` partition, x86_64) with the vLLM
ROCm K3 image through the CSCS Slurm Container Engine (EDF + enroot + Pyxis),
and register it on OpenTela.

sglang's published ROCm image has **no Kimi-K3 support at all** (no
`kimi_k3.py`; its mxfp4 MoE path is CUDA-only / uncompilable on gfx942). The
vLLM image (`vllm/vllm-openai-rocm:kimi-k3`) is the only engine with a
first-class, ROCm-dispatched K3 package — `vllm/models/kimi_k3/{amd,nvidia}/`
with `current_platform.is_rocm()` dispatch, `KimiK3ForConditionalGeneration`
matching the weights' `config.json`, MLA attention + AITER MXFP4 MoE.

## The topology (one distributed engine, not per-node replicas)

Kimi-K3 is ~1.5 TB mxfp4 — it does **not** fit on one 4×137 GB MI300A node,
so this is **not** the Beverin per-node-independent-replicas pattern used by
GLM-4.7-Flash. It is the JSC/engine shape: **one** distributed vLLM engine
across all allocated nodes (TP × PP = nodes × 4 GPUs), with **one** otela
worker on the head registering the single head HTTP endpoint.

```
beverin compute (host netns; enroot container shares it)
  rank 0 (HEAD)  -> vllm api_server (HTTP on 0.0.0.0:$SERVE_PORT) + otela worker
  rank 1..N      -> vllm --headless (no HTTP; pipeline P2P to neighbours)
```

Default **TP8 × PP2** across 4 nodes (= 16 GPUs): each tensor-parallel group
of 8 spans 2 nodes, so MoE all-reduce crosses the Slingshot fabric — over
RCCL Socket transport (see fix 4). This is the shape that reached MoE backend
selection on `k3-eng8`/`k3-eng11` (`ROCM_AITER_MLA` + `AITER_MXFP4_BF16` via
patched flydsl, PP2×TP8=16 workers); `k3-eng8` loaded shards to 16% (15/96 @
~147 s/shard, pre-restripe) before being cancelled — **full init / a token
have not yet been confirmed on gfx942**. **TP4 × PP4** (also 4 nodes) keeps
TP all-reduce intra-node (4 GPUs) but stages two pipeline layers per node —
unverified; a faster-network alternative once a token is confirmed.

## Why the `mi300` partition

vLLM's K3 image is built for **MI300A (gfx942)** / MI350 (gfx950). There is no
MI250X/gfx90a K3 build, so the job **must** run on `mi300` (4 GPUs/node,
192 CPUs, 1-day wall limit). `--account=a-infra02`.

## MI300A / gfx942-specific fixes

Four kernel-level gaps in the aiter flydsl MXFP4 MoE path, **one Python
MoE-backend-selection gap** (the job 580844 lesson), plus an RCCL network
issue, are worked around. All are handled automatically by the sbatch + EDF
+ `k3_patch.py` + `sitecustomize.py`; nothing needs to be done manually.

1. **`k3_patch.py` — gfx942 a16w4 flydsl (the kernel side).** The image's
   aiter targets gfx950 (CDNA4) and crashes on gfx942 in four ways. `k3_patch.py`
   (rank 0, serialized via a `.k3_patch_done` marker) copies the image's
   `aiter` tree to `$K3/home/pylib` and applies four arch-conditional patches:

   | patch | gfx942 gap | fix |
   |-------|-----------|-----|
   | `GFX942_SW_LDS_FILL` | no direct-to-LDS buffer DMA (`rocdl.raw_ptr_buffer_load_lds`, CDNA4-only) | SW fill: `buffer_load v4i32` from global + `llvm.store` into LDS at the same lane-major addresses |
   | `GFX942_K16_SPLIT` | no `llvm.amdgcn.mfma.f32.16x16x32.bf16` (K32 MFMA, CDNA4-only) | two K16 `mfma_f32_16x16x16bf16_1k` ops on `<4 × bf16>` halves |
   | `GFX942_SW_CVT` | hardware fp4→bf16 dequant path dead on gfx942 | force `use_hw_cvt=False` (software dequant) |
   | `GFX942_ASYNC_OFF` | hardcoded `use_async_copy=True` | `(os.environ.get("K3_NO_ASYNC") != "1")` |

   `$K3/home/pylib` is prepended to `PYTHONPATH` so the patched aiter
   overrides the image's. The patch is idempotent and strictly asserts
   expected source text (file is restored pristine, then re-applied).

2. **`sitecustomize.py` — vLLM MoE backend selection (the job 580844 lesson,
   the load-bearing one).** The kernel patches make the flydsl a16w4 MoE
   *run* on gfx942, but vLLM's *Python* MoE-backend selector
   (`quantization/mxfp4.py:_use_k3_situ_aiter`) ALSO gates the direct AITER
   path on `on_gfx950()`, which queries amdsmi and returns **False** on real
   MI300A. With it False the selector falls through to
   `oracle/mxfp4.py:select_deepseek_v4_mxfp4_moe_backend`, finds no backend for
   Kimi-K3's SiTU activation, and raises `NotImplementedError: No MXFP4 MoE
   backend supports the deployment configuration.` ~4 min in, before any shard
   loads (job 580844). `sitecustomize.py` (a sibling of `k3_patch.py`, installed
   at the overlay root by `k3_patch.py`) is auto-imported by CPython at startup
   and (1) monkey-patches `rocm.on_gfx950 = lambda: True` so `_use_k3_situ_aiter`
   takes the direct `AITER_MXFP4_BF16` branch (the one `k3-eng8`/`k3-eng11`
   reached), and (2) patches `RocmAiterMxfp4MoeBase._supports_activation` to
   accept `MoEActivation.SITU` for any future oracle path. Without this file
   the recipe is dead on arrival on real MI300A — **do not remove it** even
   though it is a small monkey-patch.

3. **MI300A integrated-memory accounting — NOT a bug for vLLM.** Unlike
   sglang (which uses `psutil.virtual_memory()` and over-counts on the APU),
   vLLM's `vllm/platforms/rocm.py` `get_device_total_memory()` queries
   per-GPU HBM3 via `amdsmi`. So vLLM gets the correct 137 GB/GPU without a
   launcher shim. **Do not** copy the sglang `is_integrated=False` launcher
   here — it is unnecessary and would mask nothing.

4. **RCCL networking — drop the CUDA OFI plugin, use Socket.** The EDF
   carries `com.hooks.aws_ofi_nccl.enabled=true`, which LD_PRELOADs a
   **CUDA-built** `libnccl-net.so` that cannot initialise on ROCm (no
   `libcudart.so`). `engine.sh` neutralises it at runtime
   (`unset NCCL_NET_PLUGIN NCCL_NET`; `NCCL_SOCKET_IFNAME=hsn0`;
   `NCCL_IB_DISABLE=1`) so RCCL uses its built-in Socket transport over
   Slingshot IP. This is the verified bring-up path; cross-node TP all-reduce
   is slow over Socket. **TODO:** a HIP-aware OFI/CXI plugin for real
   throughput (filed separately).

5. **Weight-loading barrier tolerance.** vLLM's distributed init
   (`--master-addr`/`--master-port` store) and per-shard load from Lustre have
   wide within-node variance. The checkpoint is now restriped across 8 OSTs
   (16 MB stripe, see below), so a cold start is projected at ~25–50 min
   rather than the old ~3 h 20 m. `--time=1-00:00:00` (1 day) and
   `HEALTH_TIMEOUT=14400` (4 h) give comfortable margin; the generation probe
   + otela registration run only after the head is both healthy *and*
   has produced a real token.

## The weight-loading fix (one-time Lustre restripe — DONE)

The Kimi-K3 checkpoint on `/capstor/store/cscs/swissai/infra01/hf_models/
models/moonshotai/Kimi-K3` (~1.4 TiB, 96 shards) was originally on a Lustre
mount with `stripe_count:1` (single OST, 1 MB stripe), giving ~147 s/shard →
~3 h 20 m for a full cold load (with RAM tight, 68–90 GB/worker). Prior runs
(`k3-eng8`, `k3-eng11`) were **cancelled mid-load**, and **no token had been
confirmed on gfx942**.

**As of 2026-08-06 the checkpoint has been restriped in place** (`lfs migrate`):
every shard now has `stripe_count: 8, stripe_size: 16 MB, pattern: raid0`
across 8 OSTs (verified: 0/96 shards remain on a single OST; total size and
per-shard byte counts unchanged; `config.json` + `model.safetensors.index.json`
intact). **Measured (job 580876, post-restripe): ~125 s/shard** over shards
2–5 — only ~15% faster than `k3-eng8`'s 147 s/shard, **not** the 4–8× the
striping alone might suggest. The likely reason: vLLM's distributed loader has
**all 16 ranks open all 96 shards** (each rank extracts its own TP slice from
every file), so the bottleneck is Lustre **metadata** (open/stat) contention +
CPU safetensors deserialization + GPU HBM staging, not per-file read bandwidth
that 8-way striping would relieve. So a cold start is still **~3–3½ h**; raise
`HEALTH_TIMEOUT` toward 18000 (5 h) if margin feels tight (the default 14400 /
4 h covers ~3.6 h load with ~24 min to spare). The restripe still helps the
Clariden sglang path (different loader, fewer concurrent opens). No
`MODEL_PATH` override is needed; the restripe is on the original directory.

The restripe also benefits the Clariden sglang K3 path (same weights). If you
ever want to make a *separate* restriped copy (e.g. to experiment without
 touching the canonical dir):

```bash
mkdir -p /capstor/scratch/cscs/xyao/kimi-k3-restriped
lfs setstripe -s 4m -c -1 /capstor/scratch/cscs/xyao/kimi-k3-restriped
cp -n /capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K3/* \
      /capstor/scratch/cscs/xyao/kimi-k3-restriped/
# then: sbatch --export=ALL,MODEL_PATH=/capstor/scratch/cscs/xyao/kimi-k3-restriped \
#              serve_kimi_k3_otela_beverin.sbatch
```

## The mandatory correctness probe

`/health` returning 200 does **not** prove the engine can run a forward
pass (the deepseek-v4 recipe registered on `/health` and 502'd every
request), and a non-empty chat response does **not** prove the MoE compute
path is correct. K3 has generated tokens on this GPU family **only** with
`ENFORCE_EAGER=1` (no cudagraph); a non-cudagraph launch has served real
tokens at up to 415 tok/s aggregate (10+ jobs, `BENCHMARK.md`). A
cudagraph launch deadlocks on PP3 decode (full-decode graph captures the
gloo `recv_object`), so this recipe runs a **mandatory** factual-correctness
probe (`GEN_PROBE=1`, default) before registration.

`gen_correctness.py` sends the **same six greedy `/v1/completions` prompts
the Clariden sglang servekit bench uses** (temperature 0, `max_tokens` 64 —
identical requests, so the Beverin output is directly comparable to the
Clariden `coldstart.node0.json` baseline), and checks each answer for its
expected factual substring:

| # | prompt (completion) | expects | type |
|---|---------------------|---------|------|
| 1 | `The capital of France is` | `Paris` | crisp |
| 2 | `Explain in one sentence why the sky is blue.` | `Rayleigh` | soft |
| 3 | `List the first 10 prime numbers.` | `2, 3, 5, 7, 11` | crisp |
| 4 | `Q: If a train travels 60 km in 1.5 hours, what is its average speed? A:` | `40 km/h` | crisp |
| 5 | `def fibonacci(n):` | a `return` (function body) | soft |
| 6 | `The three laws of thermodynamics are:` | `entropy` | soft |

The three **crisp** prompts are the strongest corruption detectors (Paris,
the prime sequence, and `60/1.5 = 40 km/h` are produced with essentially no
variance on a correct model) and must **all** pass. The verdict is
`PASS` only when all crisp prompts pass **and** ≥ `GEN_CORRECTNESS_MIN_PASS`
(default 5) of 6 are correct — so one soft-prompt fluke is tolerated but a
broken AITER MXFP4 MoE path (which yields garbage, not "Paris") fails the
crisp prompts at once and is caught. MXFP4 matmul accumulation order differs
between AITER (MI300A) and marlin (GH200), so byte-identical output across
the two stacks is **not** expected; substring matching is the right level.

On probe failure it logs `GEN_PROBE_FAILED`, writes a per-prompt report to
`$RUNDIR/gen_correctness_<JOB>.json`, and does **not** register (the engine
is left running for inspection — `scancel` when ready). Set `GEN_PROBE=0` to
skip (not recommended).

## Graceful leave

`--signal=B:TERM@120` signals the **batch shell** 120 s before wall-time,
giving the `stop_otela` trap a window to `scancel -s TERM` the otela step
(otela needs ~10 s: `AnnounceLeave` → 5 s CRDT drain → `srv.Shutdown`).
Without `B:`, only KillWait (30 s) applies and the head keeps a
`connected: true` row round-robining traffic into a dead endpoint.

## Verified state (2026-08-06)

| aspect | status |
|--------|--------|
| MoE backend selection on gfx942 | ✅ `k3-eng8`/`k3-eng11`: `PATCH_OK`, `KimiK3ForConditionalGeneration` resolved, `ROCM_AITER_MLA` + `AITER_MXFP4_BF16` (patched flydsl + `sitecustomize.py`), PP2×TP8=16 workers |
| vLLM K3 engine *full init* on gfx942 | ❌ **BLOCKED (job 580876, post-fix):** `installed sitecustomize.py` → `PATCH_DONE rc=0` → `Using AITER_MXFP4_BF16` at `mxfp4.py:520` (the fix in action — 580844 died here) → **all 96 shards loaded (97.33 GiB/GPU, 2h13m)** → then `_initialize_kv_caches` crashed on the 3 follower nodes with `AssertionError: collective_rpc should not be called on follower node` (`multiproc_executor.py:367`). `EngineCore.__init__` calls `collective_rpc("get_kv_cache_spec")` on EVERY node, but SHM `rpc_broadcast_mq` is only set on `node_rank_within_dp==0`. This is a **vLLM V1 multi-node PP regression**, not a recipe bug (k3-eng8 used the same TP8×PP2 launch and would have hit it — it was cancelled at 16% before KV cache init). Kimi-K3 needs ≥4 nodes (1557 GiB total), so multi-node + follower nodes are unavoidable; the standard multi-node TP/PP path cannot reach inference on this image without a code patch or a fixed build. |
| Token generation on gfx942 | ⏳ **NOT YET CONFIRMED** — mandatory probe gates registration on it |
| OpenTela registration on Beverin | ⏳ pending (first successful generation) |
| Cold-start time | ⚠️ checkpoint restriped (8 OSTs, 16 MB) but **only ~15% faster measured** (job 580876: ~125 s/shard vs `k3-eng8` 147 pre-restripe) — distributed loader is MDS-open/CPU-deserialize bound, not bandwidth bound; **~3–3½ h** cold start (raise `HEALTH_TIMEOUT` toward 18000 if tight) |
| Wall budget | 1 day (`mi300` max) — ~20 h serving after cold start |

## Files

| File | Purpose |
|------|---------|
| `kimi-k3-vllm.toml` | EDF: image (`.sqsh`), mounts (`/capstor`, `/users`, `/iopsstor`), HOME on `/capstor`, HF offline, `aws_ofi_nccl` annotation |
| `serve_kimi_k3_otela_beverin.sbatch` | One self-contained sbatch: per-rank engine wrapper (`k3_patch.py` serialization + vLLM launch) + health/correctness-probe-gated otela worker on the head |
| `k3_patch.py` | The four gfx942 a16w4 flydsl kernel patches + installs `sitecustomize.py` into the overlay (vendored verbatim from the working `k3-eng8`/`k3-eng11` bring-up) |
| `sitecustomize.py` | Python runtime overrides (auto-imported at CPython startup, every process): (1)+(2) force `on_gfx950()→True` + SiTU activation support so the K3 MoE takes the AITER path on gfx942 (job 580844 lesson); (3) multi-node follower fix — skips the leader-only `_initialize_kv_caches` collective and stubs `get_supported_tasks()` on `node_rank_within_dp != 0` (job 580876 lesson) |
| `gen_correctness.py` | Six-prompt factual correctness probe (mirrors the Clariden servekit bench); gates otela registration on real answers, not just a non-empty body |

## Submit

```bash
# default: 4 nodes, TP8 x PP2 (16 GPUs), served as moonshotai/Kimi-K3,
#                                  max-model-len 131072 (128K)
sbatch serve_kimi_k3_otela_beverin.sbatch

# from your laptop via remote-cluster-controller (rcc):
rcc -p beverin push
rcc -p beverin job submit serve_kimi_k3_otela_beverin.sbatch
rcc -p beverin job status <JOBID>
rcc -p beverin job tail <JOBID> -f
rcc -p beverin job cancel <JOBID>
rcc -p beverin pull logs/

# 128K context explicitly (same as default)
sbatch --export=ALL,CTX_LEN=131072 serve_kimi_k3_otela_beverin.sbatch

# toward 1M context (raise once a 128K cold start + generation are fast)
sbatch --export=ALL,CTX_LEN=1048576 serve_kimi_k3_otela_beverin.sbatch

# alternative shape: TP4 x PP4 (4 nodes) — keeps TP intra-node, unverified
sbatch --export=ALL,TP_SIZE=4,PP_SIZE=4 serve_kimi_k3_otela_beverin.sbatch

# with a restriped checkpoint (skip the ~3 h single-OST load)
sbatch --export=ALL,MODEL_PATH=/capstor/scratch/cscs/xyao/kimi-k3-restriped \
       serve_kimi_k3_otela_beverin.sbatch
```

The image is a one-time import (shared `$SCRATCH/.edf_imagestore`, cached for
every later job). To pre-warm from a login node:

```bash
enroot import -o /capstor/scratch/cscs/xyao/.edf_imagestore/vllm+vllm-openai-rocm+kimi-k3.x86_64.sqsh \
  docker://vllm/vllm-openai-rocm:kimi-k3
```

## Reasoning and tool calls

Kimi-K3 is a hybrid reasoning model: its chat template emits a `<think>`
(reasoning) block before the answer and supports OpenAI-style tool calls.
vLLM splits these into the `reasoning_content` and `tool_calls` response
fields **only** when launched with `--enable-auto-tool-choice` + a
`--reasoning-parser` + `--tool-call-parser`. This recipe adds all three
(parser `kimi_k3`, shipped by the `vllm-openai-rocm:kimi-k3` image) plus
`--enable-prefix-caching`, mirroring the Clariden/JSC sglang launch
(`--reasoning-parser kimi_k3 --tool-call-parser kimi_k3`); this kimi-k3 vLLM
fork spells the tool flag `--tool-call-parser`, the same as sglang.

The whole group is gated by `K3_ENABLE_PARSERS` (default `1`). If an image
bump drops the `kimi_k3` parser, vLLM errors `invalid tool call parser:
kimi_k3 (chose from ...)` at startup; set `K3_ENABLE_PARSERS=0` to launch
without parsing while the parser is restored. The mandatory correctness probe
sends plain `/v1/completions` (no chat template, no tools), so parsing does
not gate registration.

`--enable-prefix-caching` is gated **separately** by `K3_PREFIX_CACHE`
(default `0` = OFF) — it is **unconditionally broken** on this image for
Kimi-K3 (the KV cache manager selects `HybridKVCacheCoordinator` whenever
`len(kv_cache_groups) > 1`, but K3's Mamba+attention groups all share one
KV cache spec, so it asserts *"requires at least two attention groups"*
(`kv_cache_coordinator.py:627`) at scheduler init, for ANY config). See
`BENCHMARK.md` for the full root-cause and the regression history.

```bash
# /v1/chat/completions surfaces reasoning_content + tool_calls:
curl -s http://<HEAD>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"SwissAI-Research/moonshot/kimi-k3-rocm",'
      '"messages":[{"role":"user","content":"Use the get_weather tool for Zurich."}],'
      '"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],'
      '"max_tokens":256}'
```

## Verify

This is a **local deployment on the Alps mesh** — the bootstrap
`/ip4/148.187.108.178/...` is a peer on Alps itself, not the public
`api.opentela.ai`. Compute nodes' `:8080` is not routable from the login
node, so checks run from inside the allocation (the sbatch already does the
health + generation probe before registering).

```bash
# job log: look for PATCH_OK, "Resolved architecture: KimiK3...",
#   ROCM_AITER_MLA / AITER_MXFP4_BF16, "The server is fired up",
#   GEN_PROBE_OK, then otela's Peer ID + "synced" peers.
tail -f /capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/logs/k3-vllm-beverin-<JOB>.out

# otela worker log (registration, peers, LEFT on shutdown)
tail -f /capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/run-<JOB>/otela.log

# direct vLLM health + a generation from inside the allocation
srun -p mi300 -A a-infra02 -N1 -n1 --time=00:05:00 --overlap \
  curl -s http://<HEAD>:8080/v1/models | python3 -m json.tool

srun -p mi300 -A a-infra02 -N1 -n1 --time=00:05:00 --overlap \
  curl -s http://<HEAD>:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"moonshotai/Kimi-K3","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'

# once a peer is registered, route to it through any reachable OpenTela head:
curl -s http://<alps-head>/v1/service/llm/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Otela-Model: moonshotai/Kimi-K3" \
  -d '{"model":"moonshotai/Kimi-K3","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

## Knobs (env, all overridable)

| Knob | Default | Notes |
|------|---------|-------|
| `DEPLOY_DIR` | `/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin` | Work/logs/otela home. If overridden, update the EDF `workdir`/`HOME`/`HF_HOME` too. |
| `MODEL_PATH` | `/capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K3` | Weights. Point at a restriped copy to fix the load bottleneck. |
| `SERVED_MODEL_NAME` | `moonshotai/Kimi-K3` | Mesh-consistent (coexists with the Clariden sglang K3 as additive capacity). Override `kimi-k3` to match the original probe. |
| `TP_SIZE` | `8` | Must satisfy `TP × PP == nodes × 4`. |
| `PP_SIZE` | `2` | As above. |
| `CTX_LEN` | `131072` | `--max-model-len`. K3 supports up to `1048576`; KV pool is sized by `GPU_MEM_UTIL` (MLA+KDA, ~107 KB/token), not this. |
| `MAX_NUM_SEQS` | `8` | `--max-num-seqs` (upper bound; vLLM admits fewer if KV is tight). |
| `GPU_MEM_UTIL` | `0.9` | `--gpu-memory-utilization`. |
| `MASTER_PORT` | `6379` | vLLM distributed-init store port. |
| `SERVE_PORT` | `8080` | Head HTTP port (registered as the `llm` service). |
| `K3_NIC` | `hsn0` | NCCL/GLOO socket bootstrap NIC. |
| `HEALTH_TIMEOUT` | `14400` | Max wait for `/health` + generation before giving up on registration. |
| `GEN_PROBE` | `1` | Mandatory factual-correctness probe before registration (0 = skip, not recommended). |
| `GEN_PROBE_TIMEOUT` | `600` | Overall wall budget for the 6-prompt probe (s). |
| `GEN_CORRECTNESS_MAX_TOKENS` | `64` | `max_tokens` per prompt (matches the Clariden servekit baseline). |
| `GEN_CORRECTNESS_MIN_PASS` | `5` | Min correct of 6; all 3 crisp must also pass. Set `6` for a strict baseline match. |
| `GEN_CORRECTNESS_PER_REQ_TIMEOUT` | `180` | Per-request urllib timeout (s); the first request may trigger CUDA-graph capture. |
| `ENFORCE_EAGER` | `0` | Set `1` to add `--enforce-eager` (skip CUDA-graph capture; the proven path on gfx942+PP3 per `BENCHMARK.md`). |
| `K3_PREFIX_CACHE` | `0` | Set `1` to add `--enable-prefix-caching`. **Default `0` because prefix caching is unconditionally broken on this image** (HybridKVCacheCoordinator assertion at `kv_cache_coordinator.py:627`; K3's Mamba+attention groups share one KV cache spec). `0` selects `KVCacheCoordinatorNoPrefixCache` and restores the proven-working `ENFORCE_EAGER=1` path. Set `1` only after the upstream selection bug is fixed. |
| `K3_ENABLE_PARSERS` | `1` | Adds `--enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3` so `/v1/chat/completions` returns `reasoning_content` + `tool_calls`. Set `0` if an image bump drops the `kimi_k3` parser (startup errors `invalid tool call parser` first). |
| `K3_PIECEWISE` | `0` | **Confirmed NOT viable** (job 589322). Sets `VLLM_USE_BREAKABLE_CUDAGRAPH=0` + `--compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"PIECEWISE"}'`, but `KimiK3ForConditionalGeneration` carries NO `@support_torch_compile`, so `mode=VLLM_COMPILE` only warns (`vllm.py:2410`) and at init (`gpu_model_runner.py:5442`) `is_breakable=False` + `PIECEWISE.has_full_cudagraphs()=False` installs NO wrapper — the model runs EAGER, identical to `ENFORCE_EAGER=1`. Real PIECEWISE cudagraph requires upstream `@support_torch_compile` on Kimi-K3. |
| `LOAD_FORMAT` | _unset_ | Override weight loading (e.g. `dummy` for a 5-min cold start without real weights — used with `K3_PIECEWISE=1`). |
| `DISTRIBUTED_TIMEOUT_SECONDS` | _unset_ | Override the gloo/NCCL init+recv timeout (vLLM default 3600 s). Set low (e.g. `300`) with `K3_PIECEWISE=1` so a decode deadlock fails in 5 min, not 1 h. |
| `VLLM_EXTRA_ARGS` | _empty_ | Passthrough for additional vLLM flags. |
| `OTELA_BIN` | `/capstor/scratch/cscs/xyao/opentela/otela` | x86_64, sai-v0.0.6 (`--bootstrap.static`, `--label`). |
| `OTELA_RELAY_ADDR` | `/ip4/148.187.108.178/tcp/43905/p2p/QmbUKJk…` | Alps OpenTela bootstrap peer (direct, no relay). |
| `OTELA_SERVICE_NAME` | `llm` | Service registered on the mesh. |
| `OTELA_SEED` | `21` | Deterministic libp2p identity. |
| `OTELA_EDF_NAME` | `kimi-k3-vllm` | EDF name (found via `EDF_PATH=$SCRIPT_DIR:$HOME/.edf`). |

## Troubleshooting

- **`AssertionError: collective_rpc should not be called on follower node`** (job 580876, crash ~2h15m in, right after `Loading safetensors checkpoint shards: 100%`): a vLLM V1 multi-node PP regression. `EngineCore.__init__` (`vllm/v1/engine/core.py`, `_initialize_kv_caches`) calls `model_executor.get_kv_cache_specs()` → `collective_rpc("get_kv_cache_spec")` on **every** node, but `MultiprocExecutor` only sets `rpc_broadcast_mq` when `parallel_config.node_rank_within_dp == 0` (SHM message queues are intra-node). With 4 nodes and DP=1, nodes 1–3 are followers (`rpc_broadcast_mq is None`) and assert at `vllm/v1/executor/multiproc_executor.py:367`. **Not a recipe bug** — `k3-eng8` (job 579566) used the same `TP8×PP2×4nodes` launch and was cancelled at 16%, before ever reaching KV cache init. No multi-node config avoids it (Kimi-K3 needs ≥4 nodes at ~1557 GiB total; DP>1 independent replicas don't fit at 128 GiB/GPU). **Fix applied in this recipe — `sitecustomize.py` section (3):** on follower nodes (`node_rank_within_dp != 0`) it (a) replaces `_initialize_kv_caches` with a return of `KVCacheConfig(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[])` (Scheduler asserts `cache_config.num_gpu_blocks > 0`; empty groups short-circuit `resolve_kv_cache_block_sizes`, so `cache_config.block_size` is forced to 16 — all local-only) and (b) stubs `get_supported_tasks()` → `("generate",)` (the API server's startup RPC hits the same leader-only collective via the executor's `supported_tasks` cached_property). This is safe because the fork is strictly leader-driven: follower *workers* join the leader node's world-wide broadcast MQ (`v1/executor/multiproc_executor.py:589` `create_mq_broadcaster(external_writer_handle=...)`) and stream outputs back to the driver via `create_single_reader_mq_broadcasters(reader_rank_in_group=0)` — every scheduling/execute/profile/init collective (including `initialize_from_config`, which ships the real `KVCacheConfig` to all 16 workers) originates from the leader. Follower EngineCores are pure scaffolding. Kill switch: `K3_DISABLE_FOLLOWER_KV_SKIP=1`. Validation: job **581184**. (Requesting a fixed image from the builder remains worthwhile so this workaround can be retired.)
- **`NotImplementedError: No MXFP4 MoE backend supports the deployment configuration.`** (crash ~4 min in, before shard load): `sitecustomize.py` is missing from the overlay or did not auto-import, so `on_gfx950()` returned False on real MI300A and the MoE backend selector fell through to the unsupported oracle. Check the engine log for `installed sitecustomize.py -> ...` (k3_patch.py prints it on rank 0); confirm `sitecustomize.py` is present next to the sbatch and that `PYTHONPATH` puts `$K3/home/pylib` first (engine.sh does this). Inside the container, `python3 -c "from vllm.platforms.rocm import on_gfx950; print(on_gfx950())"` should print `True` once `sitecustomize` has imported.
- **`GEN_PROBE_FAILED`**: the engine is up by `/health` but the
  correctness probe did not pass. Inspect `$RUNDIR/gen_correctness_<JOB>.json`
  (per-prompt `text`/`ok`/`error`) and the job log (`[CORRECTNESS]` lines);
  do **not** expect otela to have registered. `scancel` when done. A non-0
  rc with `pass>=5` usually means a **crisp** prompt failed (Paris /
  primes / 40 km/h) — treat that as a real AITER MXFP4 MoE-corruption signal
  and inspect the engine log for NaN/illegal-value kernel errors. If the
  outputs are correct but phrased so a substring misses (e.g. a long
  reasoning trace before the answer), raise `GEN_CORRECTNESS_MAX_TOKENS`
  and re-run the probe by hand against the still-live head.
- **`health timeout after 14400s`**: cold start exceeded 4 h. Almost always
  the single-OST Lustre load (see "weight-loading bottleneck"). Restripe the
  checkpoint, or raise `HEALTH_TIMEOUT` (within the 1-day wall).
- **`FATAL: patch marker missing after 1 h`**: rank 0's `k3_patch.py` failed
  or was killed during the aiter copytree. Check the job log for the assert
  messages (expected-source-text mismatch after an image bump).
- **otela did not announce LEFT**: the step was SIGKILLed before finishing
  its graceful leave. `--signal=B:TERM@120` covers wall-time expiry; on a
  manual `scancel` the only grace is KillWait (30 s) which is enough. A
  stale `connected: true` row clears on the head's staleness sweep.
