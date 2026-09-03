# GLM-5.3-Flash on Bristen (SGLang, A100) → OpenTela

Serve `zai-org/GLM-5.3-Flash` on **Bristen** (CSCS, 4× A100 80 GB / node,
x86_64, NVIDIA) with the upstream CUDA SGLang image and the Beverin
GLM-5.3 Python overlay, then register it on OpenTela.

> **Current status — experimental SM80 fallback, A100 FP8 floor (not a serving target):** the
> model ships in FP8 (`e4m3`, ~306 GB). The recipe applies a local patch
> that upcasts the FP8 MoE expert weights/activations to the compute dtype
> *before* the Triton kernel launch, because on SM80 Triton cannot even
> declare a `*fp8e4nv` pointer in the kernel signature (`type fp8e4nv not
> supported ... supported fp8 dtypes are ('fp8e4b15','fp8e5')`). This is
> **not** native FP8 tensor-core execution; it is slow and memory-heavy.
>
> As of job 82473 (1-node TP4) the engine **boots on Bristen** (the MoE FP8
> compile error is gone via the patch below, the hybrid mamba state cache
> and KV pool allocate, and the server reports "fired up and ready to roll"),
> but the **first MoE forward OOMs**: the bf16 expert-weight upcast transient
> (~2.25 GiB/layer) cannot coexist with the KV cache + mamba state in the
> ~2.4 GiB/GPU left after the 75.3 GB FP8 weights — a genuine
> **A100-80GB HBM floor**, not a fixable bug (see [A100 HBM floor](#a100-hbm-floor)).
>
> The 2-node PP2 target (job **82822**) hits a **more fundamental** blocker
> *before* any MoE forward: the model's DSA attention cache is also FP8
> (hardcoded `torch.float8_e4m3fn` → `tl.float8e4nv` in `kpool_fp8_index.py`,
> `dsa_indexer_kpool.py`, `dsa_indexer.py`, `deepseek_v2.py`) and is **not**
> covered by the MoE/GEMM patch. The first forward fails at Triton JIT compile,
> `_kpool_assemble_softmax_rotate_write_cache_kernel`:
> `ValueError("type fp8e4nv not supported in this architecture")`. There is no
> config-only escape (`_compress_write` runs on every forward); patching it is
> a layout-sensitive re-engineering of the attention cache (allocate/store/
> read across 4 files, plus the decode `kpool_decode_update_index_cache` path)
> — a larger effort than the MoE patch, with more untested FP8 paths behind it.
> **Tracked as [vkernels#60](https://github.com/opentela-ai/vkernels/issues/60)**
> (native SM80 CUDA kernels, bf16 storage + fp32 accum, mirroring PR #52);
> the recipe-side re-validation is [cookbook#1](https://github.com/opentela-ai/cookbook/issues/1).
>
> **Update — the 82822 wall has a validated recipe-side workaround** (see
> [SM80 DSA kpool wiring](#sm80-dsa-kpool-wiring-vkernels-60--prototype-status)):
> vkernels `1de4eec` landed the SM80-proven `dsa_kpool` kernels, and the
> prototype here bridges them into the **legacy fp8+scale cache layout**
> (allocator/readers untouched, bit-exact store bytes, T0–T8 green locally)
> plus a `sitecustomize` shim replacing the SM90-only `deep_gemm` DSA-indexer
> logits on SM80. What remains is on-node proof: the serving-grade device
> path, a first-forward smoke, and 2-node PP2 correctness. The 1-node HBM
> floor below stands regardless.
> (The "detokenizer health check failed / last_heartbeat" messages around boot
> are a red herring: the TokenizerManager's health monitor simply stalls during
> the slow weight load; no detokenizer crash occurs.)
>
> **Bristen is below the FP8-compile floor for GLM-5.3-Flash, not merely the
> HBM floor, and stays a compile/boot smoke only.** Serve the model on
> **Clariden (GH200, native FP8, OpenTela-connected — see
> [`../clariden/glm-53-flash/`](../clariden/glm-53-flash/))** or Beverin
> (MI300A, 128 GB, SM90+), where native FP8 makes the entire upcast patch
> a no-op.

## SM80 DSA kpool wiring (vkernels #60) — prototype status

The job-82822 blocker (DSA kpool cache hardcoded `fp8e4nv` → SM80 Triton JIT
`ValueError`) has a kernel-side fix landed in vkernels `1de4eec`
(`dsa_kpool_assemble` / `dsa_kpool_decode_update`, bf16 storage + fp32
accum, verified 14/14 vs fp32 oracles on **both** MI300A and A100-SXM4-80GB).
The recipe side is prototyped here:

- `patched_sources/sglang/srt/layers/attention/dsa/kpool_fp8_index.py` — the
  upstream sglang file (submodule pin `third_party/sglang` @ `f5bed255`) with
  a dtype-dispatched vkernels path in the two host wrappers:
  cache `uint8` → **legacy-layout bridge** (default bristen path): vkernels
  `dsa_kpool_*` compute + a torch requant store that reproduces the Triton
  kernels' fp8e4m3 + per-vector-scale bytes EXACTLY (same store offsets, same
  `absmax/448` scale, optional pow2 `round_scale`) — the cache buffer, its
  allocator, the deep_gemm/tilelang readers, cpu offload and `move()` all
  stay untouched; cache `bf16` → native vkernels bf16 layout
  (`[num_pages, ssp, 128]`, upstream #60 follow-up shape). Gated to SM80
  (`VKERNELS_DSA_KPOOL_FORCE=1` forces, `VKERNELS_DSA_KPOOL_DISABLE=1`
  vetoes). Call sites are unchanged — **no cache-allocation flip needed** for
  the bridge path.
- `patched_sources/sitecustomize.py` — installed at `patches_full/` root
  (first on PYTHONPATH, auto-imported at interpreter startup): on SM80 it
  rebinds `deep_gemm.fp8_paged_mqa_logits` / `fp8_mqa_logits` /
  `get_paged_mqa_logits_metadata` (SM90+-only JIT, the other half of the
  job-82822 wall — the DSA indexer's prefill/decode logits) to pure-torch
  fallbacks replicating the tilelang/DeepGEMM semantics
  (`relu(q·k)·w` head-sum × `k_scale`, paged `col = table_idx*64 + j`).
  No-op on non-SM80; `SGLANG_SM80_DG_SHIM_DISABLE=1` vetoes.
- `patched_sources/sglang/kernels/ops/attention/dsa/triton_kernel.py` —
  **job-83091 wall**: the 2-node PP2 server booted, loaded FP8 weights and
  served `/health` + prefill, but the first decode died in
  `forward_absorb_prepare → indexer → act_quant` — `_act_quant_kernel`
  stores through `*fp8e4nv` pointers, which Triton cannot JIT on SM80
  (`type fp8e4nv not supported in this architecture`; same crash had killed
  82822 after its prefill). The patched `act_quant` dispatches SM80 to a
  pure-torch equivalent (`_act_quant_sm80`) with the kernel's exact semantics
  (fp32 math, `amax` clamp `1e-4`, `scale = amax/448` or pow2-ceiling
  `round_scale`, RTNE e4m3 store — torch's software fp8 casts have no arch
  requirement, probed on-node) and the same shapes/dtypes; consumers
  dequantize via `y*s`, so it is drop-in. `SGLANG_SM80_ACT_QUANT_DISABLE=1`
  vetoes.
- Local validation (`tests_local/test_dsa_kpool_wiring.py`, CPU-only,
  torch-cpu + triton, via the vkernels pure-Python fallback) — T0–T8 all
  green: the native path matches an **independent** torch re-implementation
  of the Triton kernels' math (assemble incl. write_mask/tail-chunk mix;
  decode incl. pool-complete gating, current-token substitution,
  unconditional tail update, block_tables clamp) at fp32 tight tolerance,
  preserves untouched cache slots, and — with every Triton kernel replaced by
  a raising sentinel — still completes (Triton path unreachable). Negative
  control: the unpatched wrapper enters the Triton launch path.
  Storage-envelope check: native bf16 max|err| 0.006 vs the fp32 reference;
  the legacy fp8+scale storage is 0.079 — the native path is numerically
  **tighter** than what it replaces. Bridge checks (T6/T7): the legacy-layout
  store is **bit-exact** vs a quantize emulation of the vkernels fp32 output
  (K bytes + scale bytes, both `round_scale` variants), dequantized values
  land within one e4m3 ulp + input-diff of the reference fp8, decode tails
  are identical to the native path, and untouched slots stay zero.
  Shim check (T8): both torch logits fallbacks match a per-element loop
  reference of the DeepGEMM/tilelang semantics, including the page-table
  column mapping, the −1e30 fill for unwritten paged columns, and ragged
  `clean_logits` masking. act_quant check (T9): the SM80 torch fallback is
  **byte-equal** (fp8 payload + scale) to an op-for-op emulation of the
  Triton `_act_quant_kernel` for 3 shapes × both `scale_fmt` variants, with
  the zero-guard/clamp/contiguity contract verified.

Remaining before a bristen go/no-go (mirrors the vkernels#60 acceptance list):

1. Device path: this dispatch currently uses the vkernels public Python API
   (numpy fp32, host round trip) — fine for validation, not for serving.
   Plug the C-ABI device adapters (`vk_dsa_kpool_assemble` /
   `vk_dsa_kpool_decode_update`) into the two `_vk_dsa_kpool_*_native`
   functions, mirroring `meta/diag/glm53/patch_dsa_vk.py` from PR #52, or
   build the compiled backend (`VKERNELS_BUILD_PYTHON=ON`) in the container.
   The bridge's requant store is plain torch (device-side, no JIT) and can
   stay as-is.
2. ~~First-forward smoke on bristen `flashmla_sparse`~~ **DONE** (see
   [Serving status](#serving-status)); 2-node PP2 `gen_correctness.py` →
   `PASS pass=5/6 crisp=3/3` (Clariden parity) still pending at full probe
   budget — at the current ~0.4 tok/s decode the in-job 600 s budget passes
   2/6 (both crisp), and a warm-server manual probe answers **all six
   prompts correctly** (32-token generations, ~85 s each).

Known not-bridged (would still JIT-fail if reached, neither runs in the
2-node PP2 smoke config): `kpool_write_tail_and_maybe_compress`
(target-verify / spec-decode) and the `return_compressed` path of
`kpool_softmax_rotate_write_cache` (CP / layer-shard).

The 1-node TP4 A100 HBM floor (first MoE forward OOM) is a memory limit, not
a kernel limit — it is unaffected by this wiring; 2-node PP2 remains the
serving shape for bristen.

## Serving status

The model **serves end-to-end on bristen** (2-node PP2, real FP8 weights):

- **Job 83091** — boot + FP8 load + `/health` + prefill PASS; first decode
  died at the indexer `act_quant` (`fp8e4nv` Triton JIT). Fixed by the SM80
  `act_quant` torch fallback.
- **Job 83115** — preflight PASS (incl. the new act_quant gate), prefill +
  indexer decode chain PASS; first decode died in the FA3 core (`Only
  Hopper supports different V headdim`). Fixed by `DSA_DECODE_BACKEND=
tilelang`.
- **Job 83120** — `prefill=flashmla_sparse, decode=tilelang`: prefill,
  indexer, kpool topk and TileLang sparse decode all run; in-job probe
  `pass=2/6` (both crisp: primes, train) with 4 timeouts from the ~0.4 tok/s
  JIT-cold decode vs the probe's 180 s/600 s budgets — no crashes. A
  warm-server manual probe (`run/manual_probe_83120.txt`) answers **6/6
  prompts correctly** (Paris; air-molecule blue-scatter; "2, 3, 5"; "40
  km/h"; a real fibonacci body; entropy/conservation laws) at ~85 s per
  32-token greedy request.

Throughput work (compiled vkernels backend, CUDA graphs, decode-kernel
tuning) is the next lever; correctness of the serving path is proven.

## Quick start

```bash
# from the Bristen login node
sbatch deployments/llm/bristen/glm-53-flash/serve_glm_53_flash_sglang.sbatch

# real weights + OpenTela registration
SMOKE=0 LOAD_FORMAT=auto sbatch deployments/llm/bristen/glm-53-flash/serve_glm_53_flash_sglang.sbatch
```

## Files

| File | Purpose |
|------|---------|
| `serve_glm_53_flash_sglang.sbatch` | Slurm batch: container setup, preflight, SGLang engine, generation probe |
| `engine.sh` | Per-rank SGLang launcher (args, MoE/FP8 backend selection, DSA override) |
| `apply_sm80_patch.sh` | Copy the Beverin overlay and apply the SM80 FP8→bf16 compute patches |
| `patched_sources/sglang/...` | SM80-patched copies of `fp8_kernel.py`, `fused_moe_triton_kernels.py`, `srt/layers/attention/dsa/kpool_fp8_index.py` (vkernels #60 kpool bridge: legacy fp8+scale store, bf16 native path), and `kernels/ops/attention/dsa/triton_kernel.py` (SM80 `act_quant` torch fallback — job-83091 decode wall) |
| `patched_sources/sitecustomize.py` | SM80 `deep_gemm` logits shim (paged + ragged torch fallbacks), auto-imported from `patches_full` |
| `tests_local/test_dsa_kpool_wiring.py` | CPU-only validation of the vkernels #60 kpool wiring + deep_gemm shim (see below) |
| `preflight.py` | In-container import test for the GLM-5.3 overlay |
| `gen_correctness.py` | Greedy correctness/smoke probe against `/v1/completions` |
| `README.md` | This file |

## Hardware / image

- **Node:** Bristen `normal` partition, 4× NVIDIA A100-SXM4-80GB (SM80),
  x86_64.
- **Container:** locally-cached enroot squashfs
  `/capstor/scratch/cscs/xyao/glm-53-flash-bristen/cache/enroot/sglang-dev-cu13.sqsh`
  (built once from `lmsysorg/sglang:dev-cu13`).
- **GLM-5.3 overlay:** reused from Beverin (`/capstor/scratch/cscs/xyao/glm-53-flash-beverin/overlay`).
  Only the pure-Python parts are used (`sgl-workspace/sglang/python`,
  `sgl-workspace/transformers/src`, `site-extra-cp312`). The ROCm-specific
  pieces (`pkgs310`, `rocm_libs`, `vkernels`, `sitecustomize.py`) are **not**
  added.
- **Weights:** `/capstor/scratch/cscs/xyao/models/zai-org/GLM-5.3-Flash`
  (~306 GB in FP8, 62 shards).

## Submit

### From the login node

```bash
# fast smoke with dummy weights (loads the overlay, fails later on FP8 if dummy)
sbatch deployments/llm/bristen/glm-53-flash/serve_glm_53_flash_sglang.sbatch

# real weights, hold the job for manual inspection
LOAD_FORMAT=auto SMOKE=1 sbatch deployments/llm/bristen/glm-53-flash/serve_glm_53_flash_sglang.sbatch

# real weights + OpenTela registration
LOAD_FORMAT=auto SMOKE=0 sbatch deployments/llm/bristen/glm-53-flash/serve_glm_53_flash_sglang.sbatch
```

### From your local machine via `rcc`

The repository ships a project-local `.rcc/config.toml` with a `bristen`
profile.

```bash
rcc --profile bristen push
rcc --profile bristen job submit deployments/llm/bristen/glm-53-flash/serve_glm_53_flash_sglang.sbatch
rcc --profile bristen job tail <JOBID> -f
```

## OpenTela registration

Once the engine is healthy, register it on the public OpenTela bootstrap.
The sbatch already defaults to `SMOKE=1` (no otela); set `SMOKE=0` to register,
or run the otela command manually from inside the allocation on the head node:

```bash
export OTELA_BIN=/capstor/scratch/cscs/xyao/opentela/otela
export OTELA_RELAY_ADDR=/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7
export OTELA_SERVICE_NAME=llm
export OTELA_TCP_PORT=43905
export OTELA_UDP_PORT=59820
export OTELA_SEED=42        # stable peer identity across restarts
export OTELA_API_PORT=18094

srun --jobid=<JOBID> --overlap --gres=none --nodes=1 -n1 -w <HEAD> \
  --container-image=/capstor/scratch/cscs/xyao/glm-53-flash-bristen/cache/enroot/sglang-dev-cu13.sqsh \
  --container-name=sglang-cu13 --container-env=PYTHONPATH \
  "$OTELA_BIN" start --mode node --subprocess \
    --service.type llm --service.name "$OTELA_SERVICE_NAME" \
    --service.llm.endpoint "http://127.0.0.1:30000" \
    --label model=zai-org/GLM-5.3-Flash \
    --bootstrap.static "$OTELA_RELAY_ADDR" \
    --tcp-port "$OTELA_TCP_PORT" --udp-port "$OTELA_UDP_PORT" \
    --api-port "$OTELA_API_PORT" --seed "$OTELA_SEED"
```

The same `--served-model-name zai-org/GLM-5.3-Flash` and `--label model=...`
keep direct calls and routed calls consistent (see `conventions/README.md`).

## SM80 FP8 compute patch

The sbatch automatically builds a patched copy of the SGLang Python tree in
`$DEPLOY_DIR/patches_full` and prepends it to `PYTHONPATH`:

- `sglang/kernels/ops/quantization/fp8_kernel.py`: the `_w8a8_block_fp8_matmul`
  kernel loads FP8 tensors and upcasts them to the compute dtype before
  `tl.dot`; the launcher upcasts the A/B tensors before launching the kernel
  so the Triton signature never sees an FP8 pointer.
- `sglang/kernels/ops/moe/fused_moe_triton_kernels.py`: TMA descriptors are
  disabled on SM80, and both pointer and descriptor loads are upcast to
  `compute_type` before `tl.dot`; the launcher upcasts A/B to the compute
  dtype before launching the kernel.

The patch is applied by `apply_sm80_patch.sh`, which copies the whole Beverin
overlay `sglang` tree and then overwrites the two files above with the versions
in `patched_sources/`.

### Caveats

- **Slow**: A100 has no native FP8 tensor cores; the matmuls run in bf16/fp16.
- **A100 HBM floor** (job 82473, 1-node TP4, measured): after the FP8 weight load each
  GPU has `avail mem=2.43 GB, mem usage=75.33 GB`. With `--mem-fraction-static
  0.98` the non-static activation slack is `~1.55 GB`, the hybrid mamba state
  cache allocates (`max_mamba_cache_size=8`, conv 0.01 GB + ssm 0.30 GB, which
  caps `max_running_requests` to 1 at 5 state-slots/req), the KV pool reaches
  `max_total_num_tokens=33856`, and `Memory pool end. avail mem ≈ 1.65 GB`.
  The **first MoE forward then OOMs** at `B = B.to(compute_dtype)` trying to
  allocate **2.25 GiB** with only ~1.6 GiB free. The upcast transient (~2.25
  GiB per MoE-layer GEMM, the `(72, 4096, 4096)` experts per GPU ×2 for bf16)
  cannot coexist with the KV cache + mamba state: the `--mem-fraction-static`
  window where the KV budget stays positive (`slack < A − mm − mamba ≈
  2.12 GB`) does **not** overlap the window where the upcast fits (`slack ≥
  2.15 GB`). CPU offload (`--cpu-offload-gb`) is not implemented for the
  default EP loader (`startup_weight_load.py` rejects `ep_size != 1` and
  `cpu_offload_gb > 0`). **4×A100-80GB is below the viable HBM floor for
  GLM-5.3-Flash in FP8 with the required bf16 upcast**; a node needs ≥ ~3
  GiB free/GPU after weights, i.e. Beverin/Clariden.
- **Memory-heavy**: weights are still stored as FP8, but each GEMM/MoE call
  upcasts the activation and (for MoE) weight slices to bf16 at runtime.
  This can push the 80 GB budget hard; keep `--context-length` and the
  running-batch size small.
- **Smoke-test / fallback only**: Beverin (MI300A) and Clariden (GH200)
  remain the production-grade targets for this model.

## Knobs (env, all overridable)

| Knob | Default | Notes |
|------|---------|-------|
| `DEPLOY_DIR` | `/capstor/scratch/cscs/xyao/glm-53-flash-bristen` | scratch dir for logs, caches, run state |
| `IMAGE` | `$DEPLOY_DIR/cache/enroot/sglang-dev-cu13.sqsh` | enroot squashfs |
| `OVL` | `/capstor/scratch/cscs/xyao/glm-53-flash-beverin/overlay` | GLM-5.3 pure-Python overlay |
| `SGLANG_PATCH_DIR` | `$DEPLOY_DIR/patches_full` | patched SGLang tree for SM80 |
| `MODEL_PATH` | `/capstor/scratch/cscs/xyao/models/zai-org/GLM-5.3-Flash` | weights |
| `SERVED_MODEL_NAME` | `zai-org/GLM-5.3-Flash` | sglang + OpenTela model id |
| `TP_SIZE` / `PP_SIZE` / `EP_SIZE` | `4` / `1` / `4` | one node, TP=4, EP=4 |
| `CTX_LEN` | `2048` | small context; FP8 weights leave only ~2.4 GB/GPU HBM |
| `GPU_MEM_UTIL` | `0.98` | raised from 0.94 so the hybrid mamba state cache fits the ~2.4 GB free after weights |
| `MAX_RUNNING_REQUESTS` | `8` | auto-capped to 1 by the mamba state cache (8 slots ÷ 5 state-slots/req); kept small for the tight HBM |
| `LOAD_FORMAT` | `auto` | `dummy` for fast overlay smoke, `auto` for real FP8 weights |
| `CHUNKED_PREFILL_SIZE` | `2048` | matches `CTX_LEN`; keeps prefill activations within the tight slack |
| `SMOKE` | `1` | `1` = hold job after health, `0` = OpenTela registration step |
| `DISABLE_CUDA_GRAPH` / `SKIP_SERVER_WARMUP` | `1` | defaults match Beverin/Clariden stability knobs |
| `DSA_PREFILL_BACKEND` | *(unset → `flashmla_sparse` on 2-node PP2)* | `flashmla_sparse`'s job-82822 blockers (fp8 kpool JIT, deep_gemm SM90 logits) are addressed by the [SM80 DSA kpool wiring](#sm80-dsa-kpool-wiring-vkernels-60--prototype-status) (vkernels bridge + `sitecustomize` shim) — still to be proven on-node; `tilelang` (overlay SM80 default) avoids them but then hits the [A100 HBM floor](#a100-hbm-floor) at MoE; `fa3` rejects the model's QK/V head dims on SM80. |
| `MOE_RUNNER_BACKEND` | `triton` | only backend that compiles on SM80; with the upcast patch it runs to the first MoE forward (then hits the A100 HBM floor, not a compile error) |
| `FP8_GEMM_RUNNER_BACKEND` | `triton` | same as above |
| `OTELA_BIN` | `/capstor/scratch/cscs/xyao/opentela/otela` | x86_64 otela binary |
| `OTELA_RELAY_ADDR` | public bootstrap | override if using a private bootstrap |
| `OTELA_SEED` | random | stable seed keeps the same libp2p peer id across restarts |

## Verify

Inside the allocation on the head node:

```bash
# direct sglang health
srun --jobid=<JOBID> --overlap --gres=none --nodes=1 -n1 -w <HEAD> \
  --container-image=$IMAGE --container-name=sglang-cu13 \
  bash -lc 'curl -s http://127.0.0.1:30000/health; echo'

# list models
curl -s http://127.0.0.1:30000/v1/models | python3 -m json.tool

# routed through OpenTela (after registration; replace <head> with any peer that can reach the mesh)
curl -s http://<alps-or-public-head>/v1/service/llm/v1/models \
  -H "X-Otela-Model: zai-org/GLM-5.3-Flash"
```
