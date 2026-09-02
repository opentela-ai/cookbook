# GLM-5.3-Flash on Bristen (SGLang, A100) → OpenTela

Serve `zai-org/GLM-5.3-Flash` on **Bristen** (CSCS, 4× A100 80 GB / node,
x86_64, NVIDIA) with the upstream CUDA SGLang image and the Beverin
GLM-5.3 Python overlay, then register it on OpenTela.

> **Current status — known blocker on A100 (SM80):** the model ships in FP8
> (`e4m3`, ~306 GB). No bundled FP8 GEMM backend supports SM80 out of the box:
> Triton maps `torch.float8_e4m3fn` to the Hopper-only `fp8e4nv` dtype,
> `torch._scaled_mm` requires SM89+/MI300+, and the FlashInfer/DeepGEMM/CUTLASS
> backends require SM90+/SM100+. The recipe therefore boots and loads weights,
> but the first forward pass fails to compile. See *SM80 FP8 compute blocker*
> below for the exact errors and the remaining path to a working serve.

## Quick start (for debugging the blocker)

```bash
# from the Bristen login node
sbatch deployments/llm/bristen/glm-53-flash/serve_glm_53_flash_sglang.sbatch

# with real weights and OpenTela registration (will still hit the SM80 FP8 compile)
SMOKE=0 LOAD_FORMAT=auto sbatch deployments/llm/bristen/glm-53-flash/serve_glm_53_flash_sglang.sbatch
```

## Files

| File | Purpose |
|------|---------|
| `serve_glm_53_flash_sglang.sbatch` | Slurm batch: container setup, preflight, SGLang engine, generation probe |
| `engine.sh` | Per-rank SGLang launcher (args, MoE/FP8 backend selection, DSA override) |
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

# real weights + OpenTela registration (currently still blocked at first forward)
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

## Status: SM80 FP8 compute blocker

### What fails

With the default Triton backend on A100, the first FP8 GEMM compile dies:

```text
triton.compiler.errors.CompilationError: at 1:0:
def _w8a8_block_fp8_matmul(
^
ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
```

The same error occurs in the fused MoE Triton kernel. Forcing the
FlashInfer/CUTLASS/TRT-LLM backends fails earlier at model initialization
because they gate on SM90+ or SM100+.

### Why

- `torch.float8_e4m3fn` is Triton's `fp8e4nv`, which is Hopper-only (SM90).
- A100 (SM80) supports only the storage FP8 types `fp8e4b15` / `fp8e5` in
  Triton, and has **no native FP8 tensor-core instructions**.
- The practical path on SM80 is **FP8 storage + bf16 compute**: load the FP8
  tensor, upcast to `bf16`/`fp16`, then call `tl.dot`. None of the bundled
  SGLang kernels currently do this.

### What has been tried

| Attempt | Result |
|---------|--------|
| Default Triton backend (`--fp8-gemm-backend triton`, `--moe-runner-backend triton`) | Fails at first forward with `fp8e4nv not supported in this architecture`. |
| FlashInfer CUTLASS (`flashinfer_cutlass`) | Runtime error at init: requires Blackwell (SM100+). |
| FlashInfer TRT-LLM (`flashinfer_trtllm`) | Runtime error at init: requires Blackwell (SM100+) and weight shuffling. |
| `torch._scaled_mm` direct test | `only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+`. |
| `--dsa-prefill-backend fa3` | Not viable: FA3 on SM80 rejects GLM-5.3's different QK/V head dims. The overlay already routes SM80 to the TileLang sparse DSA backend for `index_kpool > 1` tails. |

### Remaining path to a working serve

The kernels in the Beverin overlay would need to be patched to upcast FP8
inputs to bf16 before the matmul on SM80:

- `sglang/kernels/ops/quantization/fp8_kernel.py`: in `_w8a8_block_fp8_matmul`,
  convert `a` and `b` to the compute dtype after `tl.load` and before
  `tl.dot`.
- `sglang/kernels/ops/moe/fused_moe_triton_kernels.py`: in
  `fused_moe_kernel`, disable TMA descriptors on SM80 and upcast the
  pointer-loaded FP8 activations/weights to `compute_type` before `tl.dot`.

This is expected to be **functionally correct but very slow** on A100; it
would at best be a smoke-test / fallback path, not a production serving
configuration. The Beverin (MI300A) and Clariden (GH200) recipes are the
production-grade targets for this model.

## Knobs (env, all overridable)

| Knob | Default | Notes |
|------|---------|-------|
| `DEPLOY_DIR` | `/capstor/scratch/cscs/xyao/glm-53-flash-bristen` | scratch dir for logs, caches, run state |
| `IMAGE` | `$DEPLOY_DIR/cache/enroot/sglang-dev-cu13.sqsh` | enroot squashfs |
| `OVL` | `/capstor/scratch/cscs/xyao/glm-53-flash-beverin/overlay` | GLM-5.3 pure-Python overlay |
| `MODEL_PATH` | `/capstor/scratch/cscs/xyao/models/zai-org/GLM-5.3-Flash` | weights |
| `SERVED_MODEL_NAME` | `zai-org/GLM-5.3-Flash` | sglang + OpenTela model id |
| `TP_SIZE` / `PP_SIZE` / `EP_SIZE` | `4` / `1` / `4` | one node, TP=4, EP=4 |
| `CTX_LEN` | `2048` | small context because 77 GB weights/GPU leaves little HBM |
| `GPU_MEM_UTIL` | `0.94` | high static fraction for the tight 80 GB budget |
| `LOAD_FORMAT` | `auto` | `dummy` for fast overlay smoke, `auto` for real FP8 weights |
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
