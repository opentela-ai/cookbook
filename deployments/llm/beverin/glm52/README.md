# zai-org/GLM-5.2 (BF16) on Beverin (MI300A / gfx942) — SGLang + OpenTela

Serve **GLM-5.2** (zai-org/GLM-5.2, BF16 ~1.4 TB, DeepSeek Sparse Attention +
MoE + MTP) on CSCS Beverin compute nodes (4× MI300A / node, x86_64, gfx942,
Slingshot fabric) with the plain upstream SGLang ROCm image, and register it
on the OpenTela mesh.

> **STATUS (2026-08-15): BLOCKED — DSA decode on MI300A.** Every DSA backend
> tested (tilelang, aiter) either crashes, hangs at 100% GPU, or deadlocks
> before the first forward. A `flashmla_sparse` backend test (job 594548) is
> PENDING. See the [DSA backend test matrix](#dsa-backend-test-matrix) and
> [Job outcome history](#job-outcome-history) below. GLM-4.7-Flash (non-DSA,
> same image) remains the working GLM on Beverin.

---

## Table of contents

1. [GLM-5.2 architecture](#glm-52-architecture)
2. [Why SGLang, and why GLM-5.2 is lower risk than Kimi-K3](#why-sglang-and-why-glm-52-is-lower-risk-than-kimi-k3)
3. [Files in this recipe](#files-in-this-recipe)
4. [Multi-node topology](#multi-node-topology)
5. [MI300A (gfx942) workarounds](#mi300a-gfx942-workarounds)
6. [NCCL / RCCL on Slingshot](#nccl--rccl-on-slingshot)
7. [DSA backend test matrix](#dsa-backend-test-matrix)
8. [Root-cause notes](#root-cause-notes)
9. [Submission](#submission)
10. [Two-phase probe methodology](#two-phase-probe-methodology-smoke--real)
11. [Job outcome history](#job-outcome-history)
12. [Conclusion and next steps](#conclusion-and-next-steps)

---

## GLM-5.2 architecture

From `config.json` (on Beverin at
`/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.2`,
282 safetensors shards, ~1.41 TB, **no remote modeling `.py` files**):

| Field | Value | Note |
|---|---|---|
| `architectures` | `["GlmMoeDsaForCausalLM"]` | DSA = DeepSeek Sparse Attention |
| `model_type` | `glm_moe_dsa` | registered by bundled transformers 5.12.1 |
| `hidden_size` | 6144 | |
| `num_hidden_layers` | 78 | |
| `num_attention_heads` | 64 | head_dim 192 (qk_rope 64 + kv_lora 128) |
| `q_lora_rank` | 2048 | |
| `kv_lora_rank` | 512 | |
| `num_experts` | 256 | routed, top-8 |
| `moe_intermediate_size` | 2048 | per-expert |
| `intermediate_size` | 12240 | dense layers (first 3) |
| `first_k_dense_replace` | 3 | first 3 layers are dense |
| `hidden_act` | **silu** | NOT SiTU (unlike Kimi-K3) |
| `scoring_func` | sigmoid | |
| `topk_method` | noaux_tc | |
| `n_group` | 1 | |
| `routed_scaling_factor` | 2.5 | |
| `num_nextn_predict_layers` | 1 | MTP layer (speculative, disabled here) |
| `rope_interleave` | true | |
| `rope_theta` | None | |
| `rope_scaling` | **NOT PRESENT** | no rope_scaling — relevant to the SGLANG_USE_AITER question |
| `vocab_size` | 154880 | |
| `max_position_embeddings` | 1 048 576 (1 M) | |

**DSA-specific fields**: `index_head_dim` 128, `index_n_heads` 32,
`index_topk` 2048, `index_topk_freq` 4, `index_skip_topk_offset` 3.

The model ships **no remote modeling code**. The `GlmMoeDsaForCausalLM`
class is resolved from the SGLang image's bundled `transformers` (v5.12.1,
which ships `src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py`
and registers `glm_moe_dsa → GlmMoeDsaForCausalLM` in `auto.py` L749).
`--trust-remote-code` is kept as a safety no-op.

---

## Why SGLang, and why GLM-5.2 is lower risk than Kimi-K3

**SGLang (not vLLM)** is used because the v0.5.16 ROCm image already ships the
DSA backend (`layers/attention/dsa_backend.py`, `model_config.py` L114
registering `GlmMoeDsaForCausalLM`, `overrides.py` L329/L575 auto-setting
`attention_backend=dsa` for "DeepSeek 3.2/GLM 5"). On hip it auto-defaults
`dsa_{prefill,decode}_backend=tilelang`.

**GLM-5.2 is fundamentally lower risk than Kimi-K3 on Beverin**:

| Risk factor | Kimi-K3 | GLM-5.2 |
|---|---|---|
| Weights | MXFP4 (AITER FlyDSL / marlin MoE — gfx942 hang locus) | **BF16** (standard Triton bf16 MoE, proven) |
| Activation | SiTU (custom `and_mul` kernel — gfx942 unimplemented) | **silu** (standard, no custom kernel) |
| Attention | MLA + KDA (gfx942 KDA Triton kernels documented gfx950-only) | **MLA + DSA** (SGLang-native, already on glm47-flash) |
| Num heads / TP8 | 96 → 12/GPU (valid for KDA but unverified) | 64 → **8/GPU** (in [1,15], valid for MLA) |
| DSA indexer | — | new vs glm47-flash, but SGLang-native |

The **one new/risky piece** vs the proven glm47-flash path is the DSA
indexer + sparse attention via tilelang/aiter on the MI300A APU specifically.
The probe is designed to surface exactly that.

---

## Files in this recipe

| File | Purpose |
|---|---|
| `serve_glm_52_sglang.sbatch` | Self-contained sbatch: preflights EDF/image/weights/otela, sruns one SGLang engine rank per node inside the sglang-rocm container (enroot/EDF), runs a mandatory generation probe before registration, and starts one otela worker on the head. |
| `sglang-rocm.toml` | EDF (Enroot Definition File) for the `sglang-rocm` environment. Pins the image by digest, sets `/capstor` cache dirs (Triton, tilelang, HF, pip, TMPDIR), and the MI300A env vars (`HSA_NO_SCRATCH_RECLAIM=1`, `SGLANG_USE_AITER`, `LD_LIBRARY_PATH`, `NCCL_SOCKET_IFNAME=hsn`). |
| `sglang_launcher_mi300a.py` | Python launcher that monkey-patches `torch.cuda.get_device_properties` to report `is_integrated=False` (each MI300A GPU has its own 137 GiB HBM3, not shared sysmem) and `_patched_get_device_properties(device=None)` (aiter DSA decode calls it with no args). |
| `gen_correctness.py` | Model-agnostic generation probe (reused from the kimi-k3 recipe). Sends six greedy `/v1/completions` prompts (temperature 0, max_tokens 64) and checks each answer for its expected factual substring (Paris; primes; 40 km/h; Rayleigh; fibonacci; entropy). With `GEN_CORRECTNESS_SMOKE=1`, checks for non-empty continuation instead. |

---

## Multi-node topology

GLM-5.2 BF16 is ~1.4 TB — it does **not** fit one 4×137 GB MI300A node, so
this is **not** the per-node-independent-replicas pattern (GLM-4.7-Flash).
It is the **distributed engine** pattern (like kimi-k3): one SGLang engine
across all nodes, one otela worker on the head registering the single head
HTTP endpoint.

```
beverin compute (host netns; enroot container shares it)
  rank 0 (HEAD)  -> sglang serve (HTTP on 0.0.0.0:$SERVE_PORT) + otela worker
  rank 1..N      -> sglang serve, node_rank>=1 (no HTTP; pipeline P2P)
```

Default **TP8 × PP3** (24 GPUs = 6 nodes × 4 GPUs/node):
- Each TP group of 8 spans 2 nodes, so MoE allreduce crosses Slingshot (over
  RCCL Socket transport, the verified bring-up path).
- 78 layers / PP3 = 26 layers/stage.
- 256 routed experts top-8 → 112 experts/GPU.
- ~59 GiB BF16 weights / 137 GiB HBM, leaving ~78 GiB for KV + JIT.

TP4 × PP6 is also valid (64/4=16, a multiple of 16) but needs 6 nodes ×
PP6 and is unverified here.

---

## MI300A (gfx942) workarounds

Six workarounds. Items 2–5 are carried from the glm47-flash recipe
(validated SGLang-on-MI300A); items 1 and 6 are new for the GLM-5.2 DSA path.

### 1. `SGLANG_USE_AITER` (EDF default `1`) — gates the DSA preshuffle paged-MQA path

`aiter_can_use_preshuffle_paged_mqa()` in `dsa/utils.py` L49 checks
`get_bool_env_var("SGLANG_USE_AITER")` **first**, before
`AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS` or the Triton ≥ 3.5.0 check:

```python
def aiter_can_use_preshuffle_paged_mqa() -> bool:
    if not is_hip(): return False
    if not get_bool_env_var("SGLANG_USE_AITER"): return False  # L49 — FIRST GATE
    if get_bool_env_var("AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS"): return True
    try:
        from packaging.version import Version
        return Version(Version(triton.__version__).base_version) >= Version("3.5.0")
    except Exception: return False
```

On MI300A, **both values are broken**:

| `SGLANG_USE_AITER` | page_size | Prefill | Decode | Jobs |
|---|---|---|---|---|
| `0` | 1 (fallback) | ✅ works (14 s, /health 200 OK) | ❌ hangs at **100% GPU**, 0 Decode batches | 594462, 594527 |
| `1` | 64 (preshuffle) | ❌ first forward **never starts** (0% GPU, detokenizer idle) | — | 594528 |

The glm47 recipe set `SGLANG_USE_AITER=0` to avoid aiter's `get_rope` crash
(it indexes `rope_scaling["original_max_position_embeddings"]`). That risk
does **not** apply to GLM-5.2 — confirmed in job 594528: no `get_rope`
KeyError, because (a) GLM-5.2 has **no** `rope_scaling`, and (b) the DSA
backend uses its **own** rope computation via `fused_qk_rmsnorm`
(`forward_mla.py` L117, gated by `_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip`), which takes pre-computed rope, **not** `rope_scaling` config.

The EDF default is `1` (the intended preshuffle config for future SGLang
versions that fix the gfx942 deadlock). Override at submit time with
`--export=ALL,SGLANG_USE_AITER=0` to test the page_size=1 fallback (currently
the closest to working — prefill succeeds, only decode hangs).

### 2. `DISABLE_CUDA_GRAPH=1` — MTP layer crashes cuda-graph capture

Cuda-graph capture SIGKILLs on MI300A with the MTP (nextn) layer ~30 s
after weight load, during capture/warmup. Mandatory on MI300A.

### 3. `SKIP_SERVER_WARMUP=1` — MTP layer crashes the warmup pass

The MTP layer also crashes SGLang's server warmup. Skipped to avoid the
crash; the probe (step 3 below) serves as the real readiness gate.

### 4. `HSA_NO_SCRATCH_RECLAIM=1` — ROCm 7.2 hard-requirement

MI300A + ROCm 7.2 hard-requires this or HIP/RCCL aborts at kernel launch.
Set in the EDF (`sglang-rocm.toml`). See `~/.edf/minisgl-rocm.toml` and
the `tokenspeed-rocm*` EDFs.

### 5. `is_integrated=False` launcher — per-GPU KV accounting

PyTorch reports `is_integrated=True` for every MI300A GPU (it's an APU),
but each GPU has its **own** dedicated 137 GiB HBM3 (not shared sysmem).
Without the fix, SGLang uses the whole-node ~428 GiB / TP rank for the KV
budget → cgroup OOM ~35 s after weight load. The launcher
(`sglang_launcher_mi300a.py`) patches `torch.cuda.get_device_properties`
to report `is_integrated=False`, restoring correct per-GPU KV accounting.

### 6. `_patched_get_device_properties(device=None)` — aiter DSA decode crash

The aiter DSA **decode** path (`dsa_indexer.py:1969`, `forward_cuda`)
calls `torch.cuda.get_device_properties()` with **no positional argument**.
The original `torch.cuda.get_device_properties(device=None)` defaults
`device` to the current device. The monkey-patch in the launcher **must**
replicate this default:

```python
def _patched_get_device_properties(device=None):  # NOT: def ...(device):
    ...
```

Without `device=None`, the scheduler crashes with
`TypeError: _patched_get_device_properties() missing 1 required positional argument: 'device'`
(job 594459, before the fix). Fixed in all three copies:
- `glm52/sglang_launcher_mi300a.py` L69
- `glm47-flash/sglang_launcher_mi300a.py` L64
- `glm47-flash/serve_glm_47_flash_sglang.sbatch` L182 (inline heredoc)

---

## NCCL / RCCL on Slingshot

Three fixes for distributed collectives on Beverin (gfx942 + Slingshot):

### `NCCL_PROTO=SIMPLE` — gfx942 LL protocol hang

RCCL in the v0.5.16 image has **no LL cutoff points for gfx942**
(`NCCL WARN LL cutoff points not detected for a supported arch gfx942`).
TP8's 12 288-byte input broadcast triggers the LL protocol, which hangs
indefinitely. `NCCL_PROTO=SIMPLE` forces ring/tree, bypassing the broken
LL path. (Job 594097 — root-caused and fixed.)

### `--dist-timeout 7200` — PyTorch NCCL watchdog too short for tilelang JIT

PyTorch's `ProcessGroupNCCL` watchdog (controlled by
`--dist-timeout`, default 600 s) monitors **individual** collective
operations. The first forward's tilelang DSA JIT takes ~2 min/layer on 24
ranks, so a PP stage's RECV times out while the previous stage is still
compiling. `--dist-timeout 7200` (2 h) gives each collective ample room.
(Job 594176 — root-caused and fixed.)

> **Note:** SGLang's `--watchdog-timeout` and PyTorch's `ProcessGroupNCCL`
> watchdog are **separate**. SGLang's monitors the Python-level forward
> loop (does NOT fire during C-level NCCL or tilelang compile hangs);
> PyTorch's monitors individual collective operations.

### Socket transport (no `aws_ofi_nccl`)

The EDF does **not** enable the `com.hooks.aws_ofi_nccl` hook (that
CUDA-built `libnccl-net.so` cannot init on ROCm). The engine.sh wrapper
unsets `NCCL_NET_PLUGIN`/`NCCL_NET`, pins `NCCL_SOCKET_IFNAME=hsn0` and
`NCCL_IB_DISABLE=1`, and lets RCCL use its built-in Socket transport over
Slingshot. (TODO: a HIP-aware OFI/CXI plugin for real throughput.)

---

## DSA backend test matrix

The DSA backend is selected by `--dsa-prefill-backend` /
`--dsa-decode-backend`. Valid CLI choices (from the v0.5.16 argparse):
`flashmla_sparse`, `flashmla_sparse_q8`, `flashmla_kv`, `flashmla_auto`,
`fa3`, `tilelang`, `aiter`, `trtllm`.

| Backend | `SGLANG_USE_AITER` | page_size | Prefill | Decode | Outcome |
|---|---|---|---|---|---|
| `tilelang` (default) | 0 | 1 | ❌ hang / crash | — | 594304 (0% GPU 2.5 h), 594340 (SIGQUIT in `forward_c4_indexer`) |
| `aiter` | 0 | 1 | ✅ 14 s, /health 200 OK | ❌ hang 100% GPU | 594462, 594527 |
| `aiter` | 1 | 64 | ❌ first forward never starts | — | 594528 (0% GPU, detokenizer idle) |
| `flashinfer_sparse_mla` | — | — | ❌ invalid CLI choice | — | 594547 (exited in 1:14) |
| `flashmla_sparse` | 1 | 64 | ⏳ PENDING (6-node) | ⏳ PENDING | 594548 (dispatches to aiter standard MLA on AMD). 594552 (2-node PP1): OOM during model load, NOT a kernel issue — flashmla_sparse init succeeded, no deadlock. |
| `fa3` | — | — | NVIDIA-only | — | not tested |
| `trtllm` | — | — | NVIDIA-only | — | not tested |

**Key insight**: on AMD (hip), the DSA backend module
(`dsa_backend.py` L110) imports aiter's `mla_prefill_fwd` / `mla_decode_fwd`
as the standard MLA kernels. The `flashmla_sparse` choice uses these
**standard** kernels — a **different** path from the DSA preshuffle
paged-MQA kernel (which hangs at 100% GPU with page_size=1, or deadlocks
with page_size=64). If the standard MLA decode kernel works on gfx942,
`flashmla_sparse` might succeed where `aiter` DSA failed.

---

## Root-cause notes

### The DSA indexer dispatch chain

```
forward_mla.py:409  forward_absorb_prepare
  → self.indexer (MultiPlatform.forward:83 → forward_hip:95 → forward_cuda)
  → C4Indexer.forward:868
  → attn_backend.forward_c4_indexer:878
```

`--dsa-prefill-backend` / `--dsa-decode-backend` change the
`forward_c4_indexer` implementation. `SGLANG_OPT_USE_AITER_INDEXER`
(`environ.py` L1009, `EnvBool(False)`) is a **red herring** — it defaults to
False, is never `.set(True)`, and does not control the DSA indexer dispatch.

### Why the page_size=1 decode hangs at 100% GPU (jobs 594462, 594527)

With `SGLANG_USE_AITER=0`, `aiter_can_use_preshuffle_paged_mqa()` returns
False at L49 (before checking `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS`), so
the DSA decode falls back to `page_size=1` / `KVBlockSize=1`. The aiter
preshuffle paged-MQA decode kernel then runs at 100% GPU but produces **zero
Decode batches** after 10+ minutes. No error, no traceback, no watchdog fires
(SGLang's watchdog monitors the Python loop; the GPU kernel is stuck in C).
This is the **same GPU-kernel-stuck pattern** as K3 and tilelang, but at 100%
(not 0%) GPU.

### Why the page_size=64 first forward never starts (job 594528)

With `SGLANG_USE_AITER=1` and Triton 3.6.0 (≥ 3.5.0),
`aiter_can_use_preshuffle_paged_mqa()` returns True → `page_size=64` →
"Setting page size to 64 for DeepSeek DSA." (×3, all PP stages). The server
fires up in ~7 min, but the **first forward never starts**: GPU at 0%
activity (82–83% VRAM static), detokenizer process alive (pid found via
`pgrep`) but idle (no tokens to detokenize). The health-check loop fails
every 25 s ("couldn't get a response from detokenizer for last 20 seconds").
This is a **different** hang mode from page_size=1 (0% vs 100% GPU) — a
deadlock before the first GPU kernel runs, likely in the preshuffle path's
collective or buffer setup on gfx942.

### The `_patched_get_device_properties` bug (job 594459)

Before the fix, the aiter DSA decode path crashed with
`torch._inductor.exc.InductorError: LoweringException: TypeError:
_patched_get_device_properties() missing 1 required positional argument:
'device'` at `dsa_indexer.py:1969, forward_cuda`. The fix (adding
`device=None` default to the monkey-patch) resolved this and allowed the
decode path to run (and then hit the 100% GPU hang above). This confirms
the aiter DSA **prefill** works on gfx942 (the first DSA computation to
succeed on MI300A), and the **decode** path gets further than before but
still hangs.

---

## Submission

### Prerequisites

- The SGLang ROCm image must be imported to the EDF imagestore (one-time):
  ```bash
  enroot import docker://lmsysorg/sglang:v0.5.16-rocm720-mi30x
  ```
  (Already on Beverin at
  `/capstor/scratch/cscs/xyao/.edf_imagestore/sglang+sglang+v0.5.16-rocm720-mi30x.x86_64.sqsh`,
  62 GB, digest `sha256:80d04638deb64fac000fa565cb46e5d2f692173dc125a32a956014a6383ecaee`.)

- The GLM-5.2 weights must be on `/capstor/store/.../zai-org/GLM-5.2` (282
  shards, ~1.41 TB). For faster cold starts, set a wide Lustre stripe:
  ```bash
  lfs setstripe -s 4m -c -1 /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.2
  ```

- The otela binary at `/capstor/scratch/cscs/xyao/opentela/otela`.

### Fast smoke test (recommended first)

Isolates the DSA forward path in ~10 min (not the ~4 h real-weight cold
start). Uses dummy weights and checks for non-empty continuation:

```bash
rcc --profile beverin run --cwd "$PWD" -- sbatch \
    --export=ALL,LOAD_FORMAT=dummy,GEN_CORRECTNESS_SMOKE=1 \
    serve_glm_52_sglang.sbatch
```

### Real weights (after smoke passes)

```bash
rcc --profile beverin run --cwd "$PWD" -- sbatch \
    --export=ALL,GEN_CORRECTNESS_SMOKE=0 \
    serve_glm_52_sglang.sbatch
```

### Override DSA backend or SGLANG_USE_AITER

```bash
# Try the page_size=1 fallback (SGLANG_USE_AITER=0, aiter DSA — closest to working)
rcc --profile beverin run --cwd "$PWD" -- sbatch \
    --export=ALL,LOAD_FORMAT=dummy,GEN_CORRECTNESS_SMOKE=1,SGLANG_USE_AITER=0 \
    serve_glm_52_sglang.sbatch

# Try the flashmla_sparse backend (standard aiter MLA, not DSA preshuffle)
rcc --profile beverin run --cwd "$PWD" -- sbatch \
    --export=ALL,LOAD_FORMAT=dummy,GEN_CORRECTNESS_SMOKE=1,GLM52_DSA_PREFILL=flashmla_sparse,GLM52_DSA_DECODE=flashmla_sparse \
    serve_glm_52_sglang.sbatch
```

> **`--cwd` is mandatory.** `rcc job submit` runs sbatch from the
> `remote_dir` root (not this directory), so sibling files
> (`gen_correctness.py`, `sglang-rocm.toml`) are not found. Use
> `rcc run --cwd <ABSOLUTE recipe dir> -- sbatch <script>`.

---

## Two-phase probe methodology (SMOKE → REAL)

The recipe runs a **mandatory generation probe before registration** —
`/health` alone is not enough (deepseek-v4 registered on /health and 502'd
every request; K3 hung at the forward with /health still up).

1. **Wait for `/health`** on the head (up to `HEALTH_TIMEOUT`, default 4 h
   for real weights, ~10 min for dummy).

2. **Generation probe** (`gen_correctness.py`, run inside the container on
   the head via `srun --overlap`): sends six greedy `/v1/completions` prompts
   (temperature 0, max_tokens 64).
   - **SMOKE** (`GEN_CORRECTNESS_SMOKE=1`, pair with `LOAD_FORMAT=dummy`):
     checks each prompt yields a **non-empty continuation** — proves the
     full DSA → HTTP pipeline without real weights.
   - **REAL** (`GEN_CORRECTNESS_SMOKE=0`, default): checks each answer for
     its **expected factual substring** (Paris; prime sequence;
     60 km/1.5 h = 40 km/h; Rayleigh scattering; a fibonacci function body;
     entropy). Requires `GEN_CORRECTNESS_MIN_PASS` (default 5/6) to pass.

3. **Register on OpenTela** — only if the probe passes, starts one otela
   worker on the head (inside the container, via `srun --overlap --gres=none`)
   which registers the `llm` service.

4. **Clean shutdown** — on job end / `scancel`, SIGTERM (never KILL) the
   otela step so it announces `LEFT` cleanly (`--signal=B:TERM@120` gives
   the batch-shell trap a 120 s window).

---

## Job outcome history

Chronological. All jobs: 6 nodes, TP8 × PP3, `mi300` partition,
`LOAD_FORMAT=dummy` unless noted, `NCCL_PROTO=SIMPLE`, `--dist-timeout 7200`.

| Job | DSA prefill | DSA decode | `SGLANG_USE_AITER` | page_size | Result |
|---|---|---|---|---|---|
| 594097 | tilelang | tilelang | 0 | 1 | ❌ NCCL LL protocol hang (12 288-byte broadcast). **Fixed: `NCCL_PROTO=SIMPLE`.** |
| 594113 | tilelang | tilelang | 0 | 1 | ⚠️ NCCL_PROTO=SIMPLE worked. tilelang JIT compiling (~23 min) when **NODE FAILURE** on nid002698 cancelled job. |
| 594176 | tilelang | tilelang | 0 | 1 | ❌ PyTorch NCCL watchdog 600 s timeout during PP RECV. **Fixed: `--dist-timeout 7200`.** |
| 594304 | tilelang | tilelang | 0 | 1 | ⏳ Silent PP2 deadlock. GPU 0%, 83% VRAM, no JIT progress in 2.5 h. Neither watchdog fired. Cancelled. |
| 594340 | tilelang | tilelang | 0 | 1 | ❌ Fast crash at 8:55. Traceback: `forward_mla.py:409 → self.indexer → forward_hip → forward_cuda` (forward_c4_indexer). SIGQUIT→SIGTERM. |
| 594459 | aiter | aiter | 0 | 1 | ⚠️ Aiter **PREFILL WORKED** (14 s, /health 200 OK, probe started). Aiter **DECODE CRASHED**: `_patched_get_device_properties() missing arg 'device'` at `dsa_indexer.py:1969`. 0/6 probe. Before `_patched` fix. |
| 594462 | aiter | aiter | 0 | 1 | ⏳ Aiter DSA with `_patched` fix (no crash). /health 200 OK. But 0 Decode batches, GPU 100%, silent hang after 22:20:55. "page size to 1" warning. Cancelled. |
| 594527 | aiter | aiter | 0 | 1 | ⏳ Same as 594462 despite `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1` (env confirmed in container). The function checks `SGLANG_USE_AITER` **first** (L49), returns False before the AOT flag. Cancelled. |
| 594528 | aiter | aiter | 1 | 64 | ⏳ **No get_rope crash!** "Setting page size to 64" (preshuffle ENABLED, ×3 PP stages). Server fired up in ~7 min. But **first forward never starts** — 0% GPU, detokenizer idle, health check failing. Cancelled. |
| 594547 | flashinfer_sparse_mla | flashinfer_sparse_mla | 0 | — | ❌ Invalid CLI choice (exited in 1:14). Valid: `flashmla_sparse`, `flashmla_sparse_q8`, `flashmla_kv`, `flashmla_auto`, `fa3`, `tilelang`, `aiter`, `trtllm`. |
| 594548 | flashmla_sparse | flashmla_sparse | 1 | 64 | ⏳ **PENDING** (6-node TP8×PP3, starts tomorrow) — dispatches to aiter standard `mla_prefill_fwd`/`mla_decode_fwd` on AMD (different from DSA preshuffle paged-MQA). Last untested option. |
| 594552 | flashmla_sparse | flashmla_sparse | 1 | 64 | ❌ OOM at 3:54 during model load (2-node TP8×PP1 — all 78 layers on 8 GPUs, ~177 GiB exceeds 137 GiB HBM). **Not a flashmla_sparse kernel issue** — PP1 is not viable for GLM-5.2 (minimum is PP3 = 26 layers/stage ~59 GiB/GPU). Importantly, flashmla_sparse + page_size=64 did NOT deadlock during init (unlike aiter DSA in 594528); it got to model load before the OOM. |

---

## Conclusion and next steps

### Current state

GLM-5.2 DSA on MI300A (gfx942) is **not servable** with SGLang v0.5.16. The
DSA kernels (tilelang, aiter) have fundamental issues on the MI300A APU
variant (128 GB, `is_integrated=True`), despite the same gfx942 ISA as the
MI300X (192 GB, discrete) where the SGLang cookbook validates `tilelang`.

The **best result** so far is `aiter` DSA with `SGLANG_USE_AITER=0`
(page_size=1): the **prefill works** (14 s, /health 200 OK, tokens generated)
— the first DSA computation to succeed on MI300A. But the **decode hangs**
at 100% GPU with zero Decode batches (a GPU-kernel-stuck pattern, no error,
no watchdog fires).

### The remaining option

`flashmla_sparse` (job 594548, PENDING) dispatches to aiter's **standard**
`mla_prefill_fwd`/`mla_decode_fwd` on AMD — a **different** path from the
DSA preshuffle paged-MQA kernel that hangs. If the standard MLA decode
kernel works on gfx942, `flashmla_sparse` might succeed where the DSA
backends failed.

### If `flashmla_sparse` also fails

The conclusion is that GLM-5.2 DSA on MI300A requires kernel-level fixes
(not configuration tweaks). Options:
1. **Wait for a newer SGLang image** with DSA kernels fixed for gfx942
   (the preshuffle page_size=64 path is the intended config — it currently
   deadlocks before the first forward, but this is likely a bug that will be
   fixed).
2. **File an issue on SGLang** — the DSA preshuffle paged-MQA decode kernel
   hangs at 100% GPU on MI300A (gfx942) with page_size=1 (jobs 594462,
   594527) and deadlocks before the first forward with page_size=64 (job
   594528). The tilelang `forward_c4_indexer` also crashes on MI300A (jobs
   594304, 594340). MI300X (same gfx942 ISA) works with tilelang.
3. **Continue serving GLM-4.7-Flash** (non-DSA, same image, already working
   on MI300A) until DSA is fixed.

### What works on MI300A (confirmed)

- NCCL collectives over Slingshot: `NCCL_PROTO=SIMPLE` + Socket transport.
- DSA **prefill** via aiter backend: 14 s, /health 200 OK, tokens generated
  (the first DSA forward to succeed on MI300A — jobs 594459, 594462, 594527).
- The `_patched_get_device_properties(device=None)` fix (aiter DSA decode
  no longer crashes — it gets further and then hangs).
- `SGLANG_USE_AITER=1` does NOT crash GLM-5.2 (no `get_rope` KeyError, no
  `rope_scaling` — the DSA backend has its own rope via `fused_qk_rmsnorm`).

### What does not work on MI300A (confirmed)

- DSA **decode** via aiter (page_size=1): hangs at 100% GPU, 0 Decode batches.
- DSA **first forward** via aiter (page_size=64, preshuffle): deadlocks at
  0% GPU, detokenizer idle, health check failing.
- tilelang DSA (page_size=1): crashes/hangs in `forward_c4_indexer`.
- `flashinfer_sparse_mla`: not a valid CLI choice.
