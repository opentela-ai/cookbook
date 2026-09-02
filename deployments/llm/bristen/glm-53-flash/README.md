# GLM-5.3-Flash on Bristen (SGLang, A100) → OpenTela

Serve `zai-org/GLM-5.3-Flash` on **Bristen** (CSCS, 4× A100 80 GB / node,
x86_64, NVIDIA) with the upstream CUDA SGLang image and the Beverin
GLM-5.3 Python overlay, then register it on OpenTela.

> **Current status — experimental SM80 fallback, A100 HBM floor:** the
> model ships in FP8 (`e4m3`, ~306 GB). The recipe applies a local patch
> that upcasts the FP8 MoE expert weights/activations to the compute dtype
> *before* the Triton kernel launch, because on SM80 Triton cannot even
> declare a `*fp8e4nv` pointer in the kernel signature (`type fp8e4nv not
> supported ... supported fp8 dtypes are ('fp8e4b15','fp8e5')`). This is
> **not** native FP8 tensor-core execution; it is slow and memory-heavy.
>
> As of job 82473 the engine **boots on Bristen** (the FP8 compile error is
> gone, the hybrid mamba state cache and KV pool allocate, and the server
> reports "fired up and ready to roll"), but the **first MoE forward OOMs**:
> the bf16 expert-weight upcast transient (~2.25 GiB/layer) cannot coexist
> with the KV cache + mamba state in the ~2.4 GiB/GPU left after the 75.3 GB
> FP8 weights. This is a genuine **A100-80GB HBM floor**, not a fixable bug
> — see [A100 HBM floor](#a100-hbm-floor). Bristen stays a compile/boot
> smoke only; run the generation smoke on **Beverin (MI300A, 128 GB,
> SM90+)** or **Clariden (GH200, 96 GB, SM90+)**, where native FP8 GEMM makes
> the upcast patch a no-op.

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
| `patched_sources/sglang/...` | SM80-patched copies of `fp8_kernel.py` and `fused_moe_triton_kernels.py` |
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
- **A100 HBM floor** (job 82473, measured): after the FP8 weight load each
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
| `DSA_PREFILL_BACKEND` | *(unset)* | set to `fa3` to experiment; overlay default routes SM80 to TileLang |
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
