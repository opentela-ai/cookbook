# GLM-5.3-Flash on Bristen (SGLang, A100) → OpenTela

Serve `zai-org/GLM-5.3-Flash` on **Bristen** (CSCS, 4× A100 80 GB / node,
x86_64, NVIDIA) with the upstream CUDA SGLang image and the Beverin
GLM-5.3 Python overlay, then register it on OpenTela.

> **Current status — experimental SM80 fallback:** the model ships in FP8
> (`e4m3`, ~306 GB). The recipe now applies a local patch that converts
> FP8 storage to bf16/fp16 compute inside the Triton kernels, so the engine
> compiles on A100 (SM80). This is **not** native FP8 tensor-core execution;
> it is slow and memory-heavy, but it can be used to smoke-test GLM-5.3-Flash
> on Bristen.

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
| `MAX_RUNNING_REQUESTS` | `8` | small so the tiny mamba cache (~9 slots) is not over-subscribed |
| `LOAD_FORMAT` | `auto` | `dummy` for fast overlay smoke, `auto` for real FP8 weights |
| `CHUNKED_PREFILL_SIZE` | `2048` | matches `CTX_LEN`; keeps prefill activations within the tight slack |
| `SMOKE` | `1` | `1` = hold job after health, `0` = OpenTela registration step |
| `DISABLE_CUDA_GRAPH` / `SKIP_SERVER_WARMUP` | `1` | defaults match Beverin/Clariden stability knobs |
| `DSA_PREFILL_BACKEND` | *(unset)* | set to `fa3` to experiment; overlay default routes SM80 to TileLang |
| `MOE_RUNNER_BACKEND` | `triton` | only backend that even compiles on SM80 (then hits the FP8 dtype error) |
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
