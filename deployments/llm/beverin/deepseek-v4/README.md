# DeepSeek-V4-Flash on Beverin (SGLang, ROCm) → OpenTela

Serve `deepseek-ai/DeepSeek-V4-Flash` on **Beverin** (AMD MI300A / gfx942,
`mi300` partition) with the plain upstream SGLang ROCm image through the CSCS
Slurm Container Engine (EDF + enroot + Pyxis), and register it on OpenTela.

> ## ⛔ STATUS: does not serve on this image
>
> Everything here is validated up to and including weight load — the container,
> EDF, MI300A memory accounting, DSA attention backend, OpenTela wiring and the
> kernel cache all work, and the model loads in ~10 min at 43.76 GB/rank. But
> **no token has ever been generated on MI300A.** The MXFP4 expert kernel is
> disabled by an upstream aiter macro-name bug (fix #2 → *Root cause*), which no
> setting in this recipe can work around; it needs a patched aiter.
>
> Fixes #1 and #3–#8 are independently verified and worth keeping — they are the
> non-obvious parts of getting *any* DeepSeek-V4 to this point on ROCm. Treat
> the recipe as a bring-up log, not a working deployment, until the aiter patch
> is built and this banner comes off.

Like the sibling [`glm47-flash/`](../glm47-flash/) GLM-4.7-Flash recipe and unlike the
JSC one, **no relay is needed**: Beverin compute nodes have full outbound
internet and reach the bootstrap `/ip4/148.187.108.178/...` directly, so each
rank runs `otela start --mode node --subprocess <sglang-wrapper>` on the same
node as SGLang.

> **Read the GLM recipe first if you are new to Beverin.**
> [`../glm47-flash/README.md`](../glm47-flash/README.md) documents the MI300A basics
> (NUMA affinity, TMPDIR, integrated-memory accounting) that both recipes
> share. This page covers what is *different* for DeepSeek-V4 — and two of the
> shared knobs are set to the **opposite** value here, so do not copy settings
> between the two recipes without reading fix #2 and #3 below.

## What makes this model different

DeepSeek-V4-Flash is a ~283 B-parameter sparse MoE (43 layers, 256 routed
experts + 1 shared, 6 experts/token) with DeepSeek Sparse Attention (a C4
indexer, `index_topk=512`) and per-layer KV compression. Two properties drive
everything below:

| Property | Value | Consequence on gfx942 |
|---|---|---|
| Routed-expert layout | **mxfp4**: `I8` weights (2 fp4/byte) + `F8_E8M0` block scales, 132 GiB of the 149 GiB checkpoint | gfx942 has no native MX compute → needs aiter's CK MXFP4 kernels (fix #2) |
| Attention | DSA / sparse MLA, `kv_cache_dtype=fp8_e4m3` | default TileLang kernel does not compile (fix #3) |

SGLang detects the expert layout itself by reading the safetensors header
(`try_detect_fp4_experts` → `I8` → `is_fp4_experts=True`) and routes the model
through `Fp8Config`, **not** through the generic `Mxfp4Config` — which is
important, because `Mxfp4Config` would hard-fail here:
`mxfp_supported()` is gfx95-only, so it raises
`ValueError: Current platform gfx942 not support mxfp4 computation`.

## Why the `mi300` partition

sglang only publishes ROCm builds for **MI300A (gfx942, `*-mi30x`)** and the
MI350 series (gfx950, `*-mi35x`). There is no MI250X/gfx90a image, so even
though the login node is MI250X the job **must** run on `mi300`. The image
(`lmsysorg/sglang@sha256:80d046…` = `v0.5.16-rocm720-mi30x`) is pinned by
digest so the Triton/aiter JIT caches on shared `/capstor` stay warm across
launches. v0.5.16 is also the first tag carrying
`sglang/srt/models/deepseek_v4.py`.

## MI300A + DeepSeek-V4 fixes (all baked into the sbatch)

### 1. MI300A integrated-memory accounting

MI300A is an APU, so PyTorch reports `is_integrated=True` for every GPU.
sglang's `get_available_gpu_memory()` then substitutes
`psutil.virtual_memory().available` — the **whole-node** figure, measured at
**477 GiB** on Beverin — for the per-GPU `mem_get_info()` value (**128 GiB**),
on *every* TP rank. Since the distributed path takes `all_reduce(MIN)` and all
ranks see the same psutil number, the minimum stays 477 GiB, each rank sizes
its KV pool against memory that does not exist, and a rank gets SIGKILLed by
the cgroup OOM killer (exit `-9`) shortly after `Load weight end` with no
Python traceback.

**Fix:** `sglang_launcher_mi300a.py` proxies
`torch.cuda.get_device_properties()` to report `is_integrated=False`.
Verified — with it, every rank logs `Load weight begin. avail mem=119.15 GB`
instead of ~477 GB.

> This is the same fix as the GLM recipe, and its `__main__` guard is
> **load-bearing**: sglang spawns its per-TP-rank schedulers with
> multiprocessing `spawn`, which re-imports the launcher in every child.
> Without the guard the children re-enter `run_server()` at import time and die
> with *"An attempt has been made to start a new process before the current
> process has finished its bootstrapping phase"*, which the parent reports only
> as the much less helpful `RuntimeError: Rank 1 scheduler died during
> initialization (exit code: 1)`.

### 2. `SGLANG_USE_AITER=1` — the opposite of the GLM recipe

The GLM-4.7-Flash recipe **must** set `SGLANG_USE_AITER=0`, because aiter's
`get_rope` unconditionally indexes `rope_scaling["original_max_position_
embeddings"]`, which GLM's config omits. DeepSeek-V4-Flash's `rope_scaling`
*is* a full YaRN block (`factor: 16`, `original_max_position_embeddings:
65536`), so that crash does not apply — and aiter must be **on**, because it
selects the native MXFP4 MoE path in
`Fp8MoEMethod.process_weights_after_loading_block_quant()`:

```python
# AMD FP4 experts: use aiter's native MXFP4 MoE path
if _use_aiter and self.is_fp4_expert:
    fp4_weight_dtype = _require_fp4_dtype()   # torch.float4_e2m1fn_x2
```

With aiter on, expert weights **stay fp4**: measured **43.76 GB resident per
rank** (175 GB across TP=4), leaving `avail mem=75.3 GB` per GPU for the KV
pool. The alternative — `SGLANG_DSV4_FP4_DEQUANT=1`, which runs
`cast_e2m1fn_to_e4m3fn` over every expert at load — **doubles** the expert
footprint to ~264 GiB (~70 GB/rank) and leaves less KV headroom. (It asserts
`moe_runner_backend=auto`, so it cannot be combined with an explicit
`--moe-runner-backend`.)

> **⚠ On this image, the memory-efficient path does not actually run.** The CK
> MXFP4 kernel compiles (fix #4) and then fails at dispatch on the first token:
>
> ```
> File "aiter/fused_moe.py", line 2448, in ck_moe_stage1
>     aiter.ck_moe_stage1_fwd(...)
> RuntimeError: Unsupported kernel config for moe heuristic dispatch
> ```
>
> aiter 9127c94a ships no CK instance for this model's MoE shape on gfx942
> (`hidden_size=4096`, `moe_intermediate_size=2048`, so
> `intermediate_per_partition=512` at TP=4 — already 256-aligned, so it is not
> the documented padding constraint). Observed on job 574177, *after* the
> memory fix in #6, so it is a genuine kernel-coverage gap and not another
> resource problem.
>
> **And `DSV4_FP4_DEQUANT` cannot rescue it — the dequant is unreachable on
> gfx942.** `Fp8MoEMethod.process_weights_after_loading_block_quant()` is one
> if/elif chain, and the dequant sits in its final `else`:
>
> ```python
> 1319:  if _use_aiter and self.is_fp4_expert:      # aiter native MXFP4
> ...
> 1474:  if _is_fp8_fnuz:                           # gfx94* is ALWAYS fnuz
> 1476:      normalize_e4m3fn_to_e4m3fnuz(...)      #   -> assert on I8 weights
> 1504:  elif _use_aiter: ...
> 1512:  elif _is_cpu: ...
> 1517:  else:                                      # CUDA only
> 1525:      if self.is_fp4_expert and self.dequant_fp4_to_fp8:
> ```
>
> `is_fp8_fnuz()` returns `"gfx94" in gcnArchName`, so on MI300A the `_is_fp8_fnuz`
> arm always wins and the `else` never runs. Both settings were tried:
>
> | Config | Result |
> |---|---|
> | `USE_AITER=1` (default) | line 1319 → CK dispatch gap, dies on **first token** |
> | `USE_AITER=1, FP4_DEQUANT=1` | silent no-op — job 574280 still logged `mem usage=43.76 GB`, the *fp4* footprint, so the cast never ran |
> | `USE_AITER=0, FP4_DEQUANT=1` | line 1474 → `assert weight.dtype == torch.float8_e4m3fn` **AssertionError at load** (job 574330) |
>
> So on this image the mxfp4 checkpoint has exactly **one** live code path on
> gfx942, and it has no kernel for this shape. v0.5.16-rocm720-mi30x is already
> the newest `mi30x` tag (2026-07-24), so there is nothing to upgrade to.
>
> `--moe-runner-backend humming` selects `Mxfp4HummingMoEMethod`, which sglang
> documents as *"used for DeepSeek-V4 FP8 checkpoints"* and which brings its own
> `process_weights_after_loading`, bypassing the fnuz chain. It gets furthest —
> `quant_method=Mxfp4HummingMoEMethod` is selected and weights load — then dies
> with `ModuleNotFoundError: No module named 'humming'` (job 574342). The
> `humming` package is neither in the image nor on PyPI.

#### Root cause: an aiter macro-name bug, not a shape or config problem

Do not waste node time sweeping TP sizes or memory settings — the CK MXFP4
kernel is dead on arrival for a reason that has nothing to do with either.
The generated stage-1/stage-2 dispatch headers wrap their *entire* FP4 body in

```c
#if defined(__Float4_e2m1fn_x2)
    ... every supported (block_m, inter_dim) instance ...
#endif
    TORCH_CHECK(false, "Unsupported kernel config for moe heuristic dispatch");
```

and **nothing ever defines `__Float4_e2m1fn_x2`**. The macro aiter's build
actually passes is the *differently spelled* `-DTORCH_Float4_e2m1fn_x2`:

| Location | Macro |
|---|---|
| `csrc/include/py_itfs_common.h:54` | `#ifdef TORCH_Float4_e2m1fn_x2` ✓ matches the build |
| `csrc/ck_gemm_moe_2stages_codegen/gen_instances.py:267,301,620,677` | emits `#if defined(__Float4_e2m1fn_x2)` ✗ never defined |
| `build.ninja` (generated) | `-DTORCH_Float4_e2m1fn_x2` |

So the whole FP4 dispatch table is preprocessed away and every call falls to
the `TORCH_CHECK`. This is **shape- and arch-independent**: no TP size, no
`mem_fraction_static`, no runner backend changes it. Confirmed by reading the
generated headers straight out of the JIT cache (fix #4 makes them persist,
which is how this was found without another job).

Related, and consistent: the build also logs
`Current hipcc not support: -mllvm -amdgpu-coerce-illegal-types=1, skip it.` —
the flag that enables FP4 illegal-type codegen on gfx942 is dropped too.

**The fix requires patching aiter**, which is tractable because
`AITER_META_DIR` is overridable (`aiter/jit/core.py:402`) and selects
`AITER_CSRC_DIR` — i.e. the codegen tree:

1. copy `/sgl-workspace/aiter` to `/capstor`,
2. in `csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`, emit
   `TORCH_Float4_e2m1fn_x2` (or add `-D__Float4_e2m1fn_x2` to the build flags),
3. `export AITER_META_DIR=<patched copy>`,
4. delete the cached `module_moe_ck2stages_*fp4x2*` from `AITER_JIT_DIR` so it
   regenerates, and re-pay the ~35 min compile (once — fix #4 keeps it).

`TODO(unverified)` — the patch is derived from reading the source, not yet
built and run. **As of this image, DeepSeek-V4-Flash does not serve on
MI300A.**

### 3. `SGLANG_HACK_FLASHMLA_BACKEND=triton` — the default does not compile

On HIP, `hip_flash_mla.flash_mla_with_kvcache_entrypoint()` **ignores**
`--dsa-prefill-backend` / `--dsa-decode-backend` and reads the backend straight
from this env var:

```python
def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if is_hip():
        backend = envs.SGLANG_HACK_FLASHMLA_BACKEND.get()   # default "tilelang"
```

The default `tilelang` does not compile on gfx942. The server comes up and
answers `/get_model_info` and `/v1/models` fine, then the **first `/generate`**
kills every scheduler rank:

```
File "sglang/kernels/ops/attention/dsa/tilelang_kernel.py", line 2513, in dpsk_v4_fp8_attention_fwd
File "/opt/tilelang/tilelang/engine/phase.py", line 227, in OptimizeForTarget
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
ValueError: Check failed: src_info.order < dst_info.order (6 vs. 4) :
  two statements with buffer access dependency in the same stage of the
  software pipeline cannot be reordered
[2026-07-31 12:10:16] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
```

Note the failure mode: **a healthy-looking server that dies on the first real
request.** A readiness probe that only checks `/get_model_info` will not catch
it — which is why `probe_sglang.sbatch` issues an actual `/generate`.

**Fix:** `SGLANG_HACK_FLASHMLA_BACKEND=triton`, the ROCm alternative sglang's
own hisparse hook points at (*"Either set `SGLANG_HACK_FLASHMLA_BACKEND=triton`,
or run without `--enable-hisparse`"*). `unified_kv_triton` is the other HIP
path; it is incompatible with `--enable-hisparse`.

### 4. Persistent aiter JIT cache (`AITER_JIT_DIR`)

The native MXFP4 MoE kernel
(`module_moe_ck2stages_b16_fp4x2_preshuffle_off_b16_silu_per_1x32_mulWeightStage2`)
is Composable-Kernel C++ that aiter hipcc-compiles **on the first request, not
at load** — ~20 `hipcc` + ~20 `clang` processes, with the other 3 TP ranks
blocked on aiter's baton lock (`waiting for baton release at …/lock_module_moe_
ck2stages_…`). It is slow partly because every CK template header is read
through the squashfs FUSE mount (`squashfuse_ll` sits at ~78 % CPU during it).

aiter's `get_user_jit_dir()` prefers the in-image
`/sgl-workspace/aiter/aiter/jit` whenever it is writable — which it is, and
which dies with the container — so **every job re-pays the compile**.

**Fix:** the sbatch seeds `$DEPLOY_DIR/cache/aiter-jit` from the image's jit
dir once (under an atomic `mkdir` lock, so exactly one rank across all nodes
copies and the others wait) and exports `AITER_JIT_DIR` at it. Seeding rather
than pointing at an empty dir matters: the image ships **prebuilt** `.so`
modules (`module_aiter_core.so` and friends) in that tree, and an empty
`AITER_JIT_DIR` would make aiter rebuild those from scratch too.

### 5. Weight load is ~10 min and is bounded by a full re-read

`Load weight end. elapsed=608 s` — but only ~2 min of that is the 46-shard
read. The rest is `_dequant_fp8_wo_a()`, which sglang runs because
`server_args.py` does this on ROCm:

```python
elif is_hip():
    ...
    envs.SGLANG_OPT_FP8_WO_A_GEMM.set(False)   # DeepGEMM is CUDA-only
```

With the flag off, `deepseek_v4.py` takes the branch that does
`weights = list(weights)` and then `dict(weights)` over all ~69 k tensors,
mmap-faulting the whole 149 GiB checkpoint back in from Lustre at roughly
100 MB/s/rank. This is **expected upstream behaviour on ROCm, not a
misconfiguration** — do not "fix" it by forcing `SGLANG_OPT_FP8_WO_A_GEMM=1`,
which would send it down a DeepGEMM path that does not exist here.

Host RAM is not at risk (the checkpoint is mmap'd; measured 261 GiB used /
239 GiB available at the peak), but it does mean **budget ~12 min from job
start to first token**, plus the one-time aiter compile in fix #4.

### 6. The first (kernel-compiling) run needs a LOW `mem_fraction_static`

This is the one that costs you a node for three hours if you get it wrong, and
it is a direct consequence of fix #1's observation that **on MI300A the GPU
memory *is* the host memory** — one 501 GiB pool, not 501 GiB of RAM plus
512 GiB of HBM.

At the normal `mem_fraction_static=0.85`, the engine reserves
0.85 × 128 GiB × 4 ≈ **435 GiB** of that single pool. Which is fine — until
aiter starts the fix-#4 CK compile *on the same node*, forking ~20 hipcc +
~20 clang. C++ template compiles of this size want multiple GB each, and only
~66 GiB is left. Observed on job 573882:

```
              total  used  free  shared  buff/cache  available
Mem:            501   501     1       4          13           0     <-- available: 0
```

with `kswapd0`/`kswapd2` pinned, 16 clang processes parked in `D` state, 16
more in `T`, and **no file written to the build dir for three hours**. It is a
deadlock, not slow progress: the stalled compiles never release aiter's baton
lock, so all four schedulers wait on it forever. There is no OOM kill and no
traceback — the job just sits there looking busy.

**Fix:** do the first run — the one that populates the kernel cache — at a low
static fraction, then serve normally once the cache is warm:

```bash
# cold cache: leave the compiler ~270 GiB to work in
sbatch --export=ALL,MEM_FRACTION_STATIC=0.45 probe_sglang.sbatch

# warm cache: no compile, so the engine can have the memory back
sbatch serve_deepseek_v4_flash.sbatch
```

Weights are only 43.76 GB/rank, so 0.45 (57.6 GiB/rank) holds them comfortably
with a small KV pool — enough to drive the generation that triggers the build.
Better still, skip the compile entirely by restoring a prebuilt cache:
`./sync_aiter_kernels.sh download` (see below).

### 7. A killed job leaves aiter's baton lock behind

aiter guards each module build with a baton that only the *building* process
removes — and it uses **two of them, at different depths**:

```
<jit>/build/lock_module_<name>          # outer, one per module
<jit>/build/module_<name>/build/lock    # inner, the ninja build
```

`scancel` a job mid-compile and both survive on `/capstor`. The next job then
acquires the *outer* lock, starts its ninja build, and blocks forever on the
**dead job's inner lock**: 0 % CPU, no compiler process, no error, no log line
after `start build` — just a hang that looks exactly like a slow compile.

Cleaning only the outer lock is not enough, and that trap is not hypothetical:
it is precisely how job 574156 stalled after 574156's predecessor was
cancelled. The give-away is a `waiting for baton release` line whose path ends
in `/build/lock` rather than `/lock_module_…`, emitted by the *same* pid that
just logged `start build`:

```
[aiter] [pid=58819] start build [module_moe_ck2stages_b16_fp4x2_…]
[aiter] [pid=58819] waiting for baton release at …/module_moe_ck2stages_…/build/lock
```

**Fix:** both sbatches sweep every baton older than the job's own start, once,
in the sbatch body before `srun` (so it cannot race a rank that is legitimately
building). It also drops the 0-byte `*.o.tmp` hipcc scratch the dead build left
behind. Verified on job 574177: `aiter_stale_locks_swept=11`.

To do it by hand after a non-clean shutdown:

```bash
find <deploy>/cache/aiter-jit/build -maxdepth 3 \
     \( -name 'lock_module_*' -o -path '*/build/lock' \) -delete
```

`sync_aiter_kernels.sh` treats the outer lock as a refuse-to-upload signal, so
a stale one will also block uploads until it is cleared.

### 8. NUMA CPU affinity and TMPDIR (inherited)

Both as in the GLM recipe, and both still required:

- No `--cpus-per-task` in the `#SBATCH` header; the `srun` step passes
  `--cpus-per-task="${SLURM_CPUS_ON_NODE}"` (=192) so the image's
  `SGLANG_SET_CPU_AFFINITY=1` NUMA pinning can resolve host CPU ids 0–191.
  Otherwise ranks 2–3 die with `ValueError: CPU number 168 is not eligible`.
- `TMPDIR` on `/capstor`, not the Slurm default `/users/<u>/.tmp` — home is
  quota-limited and sglang's tempfile cleanup there raises
  `OSError [Errno 39] Directory not empty`.

## Files

| File | Purpose |
|---|---|
| `deepseek-v4-rocm.toml` | EDF: image (by digest), mounts, ROCm env |
| `probe_sglang.sbatch` | 1-node bring-up gate: starts SGLang alone, waits for readiness, then issues a **real `/generate`** (the only way to catch fix #3) |
| `serve_deepseek_v4_flash.sbatch` | The recipe: per-rank SGLang wrapper + otela worker + optional vmagent |
| `sglang_launcher_mi300a.py` | Standalone copy of the MI300A `is_integrated=False` launcher (also generated inline by the sbatch) |
| `sync_aiter_kernels.sh` | Login-node helper: share the compiled aiter kernels through a Hugging Face bucket (see below) |

## Sharing the compiled kernels (`sync_aiter_kernels.sh`)

Fix #4 keeps the CK compile on `/capstor` so one deploy dir pays it once.
`sync_aiter_kernels.sh` lifts that one level further — push the compiled
kernels to [`researchcomputer/kernels`](https://huggingface.co/buckets/researchcomputer/kernels)
so a *fresh* deploy dir, another user, or another cluster starts warm.

Run it on a **login node** (that is where `hf` and your credentials live;
`/capstor` is visible from both).

```bash
export PATH=$HOME/.local/bin:$PATH        # hf lives here on beverin
export HF_TOKEN=hf_...                    # needs WRITE on the researchcomputer namespace

# always look first — it prints the exact file list and the remote key
./sync_aiter_kernels.sh upload --dry-run

./sync_aiter_kernels.sh upload            # prompts before publishing

# on a fresh machine, before the first job:
./sync_aiter_kernels.sh download --jit-dir <deploy>/cache/aiter-jit
```

Three things it is deliberate about:

1. **It uploads only the delta.** The JIT dir is ~4.6 GB, almost all of it
   copied out of the container image at seed time. The sbatch records a
   `.seeded.manifest`, and the script subtracts it, so only kernels this site
   actually compiled go up — it does not redistribute the image's prebuilt
   binaries. (`--all` overrides; you almost never want it.)
2. **It refuses to upload mid-build.** aiter leaves a `lock_module_<name>`
   beside whatever it is compiling; publishing then would ship a half-written
   module and poison the cache for everyone who restores it. Verified against a
   live in-flight build — the script names the module and exits non-zero.
3. **The remote key pins `(gpu_arch, aiter_commit, image_digest)`** —
   `aiter/gfx942/<aiter12>/<image12>/`. These are architecture-specific
   binaries; silently restoring a gfx942 build onto gfx950 would hand the
   engine wrong code. The sbatch writes `aiter-provenance.env` next to the JIT
   dir so the key is derivable without re-entering the container, and every
   upload carries a `manifest.json` recording arch, ROCm/torch/sglang versions,
   who uploaded it and when.

> **Unverified:** which subset of the delta is *sufficient* for a cache hit.
> The script uploads the full delta (a safe superset); `--so-only` uploads just
> the `.so` files, which is likely enough but has not been confirmed by a
> restore-and-skip-compile test. Confirm before relying on `--so-only`.

## Submit

```bash
# ALWAYS probe first on a new image — it is a 1-node job and it catches the
# expensive-to-debug kernel problems before you burn N nodes.
sbatch probe_sglang.sbatch

# default: 1 node, TP=4, served as deepseek-ai/DeepSeek-V4-Flash
sbatch serve_deepseek_v4_flash.sbatch

# scale out: one OpenTela peer per node, each an independent TP=4 replica
sbatch --nodes=4 serve_deepseek_v4_flash.sbatch
```

The image is already in `$SCRATCH/.edf_imagestore` (66 GiB). To warm it on a
fresh account, from a login node:

```bash
enroot import docker://lmsysorg/sglang:v0.5.16-rocm720-mi30x
```

Weights are pre-staged at
`/capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V4-Flash`
(149 GiB, 46 shards).

## Verify

This is a **local deployment on the Alps mesh** — the bootstrap
`/ip4/148.187.108.178/...` is a peer running on Alps itself, not the public
`api.opentela.ai`. Compute-node `:8080` is not routable from the login node, so
health checks run from inside the allocation.

```bash
# job + per-rank OpenTela logs (look for: opentela_started, aiter_jit_dir=…,
# sglang_ready, a Peer ID: line)
tail -f /capstor/scratch/cscs/xyao/deepseek-v4-flash-sglang/logs/opentela-<JOB>-*.log

# engine-side milestones
grep -E "Load weight end|Memory pool end|ready to roll" \
  /capstor/scratch/cscs/xyao/deepseek-v4-flash-sglang/logs/*-<JOB>.out

# direct SGLang health from inside the allocation (login node can't reach it)
srun --jobid=<JOB> --overlap -N1 -n1 \
  bash -lc 'curl -s http://$(hostname -i | awk "{print \$1}"):8080/get_model_info | python3 -m json.tool'
```

`/get_model_info` returns, on a healthy server:

```json
{"model_path": ".../DeepSeek-V4-Flash", "is_generation": true,
 "model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"]}
```

**`/get_model_info` answering is not sufficient** — see fix #3. Always follow
it with a real generation:

```bash
srun --jobid=<JOB> --overlap -N1 -n1 bash -lc \
  'curl -s http://$(hostname -i | awk "{print \$1}"):8080/generate \
     -H "Content-Type: application/json" \
     -d "{\"text\":\"The capital of Switzerland is\",\"sampling_params\":{\"max_new_tokens\":24,\"temperature\":0}}"'
```

## Knobs (env, all overridable)

| Knob | Default | Notes |
|---|---|---|
| `MODEL` | staged capstor path | |
| `SERVED_MODEL_NAME` | `deepseek-ai/DeepSeek-V4-Flash` | |
| `SGLANG_PORT` | `8080` | |
| `TP_SIZE` | `4` | one MI300A node |
| `MEM_FRACTION_STATIC` | `0.85` | |
| `MAX_MODEL_LEN` | `65536` | config advertises 1 048 576 (YaRN ×16); 65536 is the un-scaled window |
| `MAX_RUNNING_REQUESTS` | `256` | |
| `SGLANG_USE_AITER` | `1` | **fix #2** — opposite of the GLM recipe. Do not set `0`: that path asserts at load on gfx942 |
| `DSV4_FP4_DEQUANT` | `0` | **has no effect on gfx942** — the dequant branch is unreachable behind `_is_fp8_fnuz` (see fix #2) |
| `FLASHMLA_BACKEND` | `triton` | **fix #3**; `tilelang` (upstream default) does not compile on gfx942 |
| `MOE_RUNNER_BACKEND` | `auto` | `humming` selects `Mxfp4HummingMoEMethod`, the remaining candidate for the fix-#2 kernel gap |
| `AITER_JIT_DIR` | `$DEPLOY_DIR/cache/aiter-jit` | fix #4 |
| `SGLANG_REASONING_PARSER` | `deepseek-v4` | |
| `SGLANG_TOOL_CALL_PARSER` | `deepseekv4` | |
| `DISABLE_CUDA_GRAPH` | `0` | GLM must force `1`; not required here |
| `SKIP_SERVER_WARMUP` | `0` | |
| `LOAD_FORMAT` | `auto` | |
