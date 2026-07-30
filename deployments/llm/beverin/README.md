# GLM-4.7-Flash on Beverin (SGLang, ROCm) → OpenTela

Serve `zai-org/GLM-4.7-Flash` on **Beverin** (AMD MI300A / gfx942, `mi300`
partition) with the plain upstream SGLang ROCm image through the CSCS Slurm
Container Engine (EDF + enroot + Pyxis), and register it on OpenTela.

Unlike the JSC recipe, **no relay is needed**: Beverin compute nodes have full
outbound internet and reach the bootstrap `/ip4/148.187.108.178/...` directly, so
each rank runs `otela start --mode node --subprocess <sglang-wrapper>` on the
same node as SGLang.

## Why the `mi300` partition

sglang only publishes ROCm builds for **MI300A (gfx942, `*-mi30x`)** and the
MI350 series (gfx950, `*-mi35x`). There is no MI250X/gfx90a image, so even
though the login node is MI250X the job **must** run on `mi300`. The image
(`lmsysorg/sglang@sha256:80d046…` = `v0.5.16-rocm720-mi30x`) is pinned by digest
so the Triton JIT caches on shared `/capstor` stay warm across launches.

## MI300A-specific fixes (all baked into the sbatch)

Five ROCm/MI300A issues had to be worked around vs the CUDA (GH200) recipe.
All five are handled automatically by the sbatch + EDF; nothing needs to be
done manually unless tuning.

1. **NUMA CPU affinity crash** — `SGLANG_SET_CPU_AFFINITY=1` (from the image's
   `/etc/environment`) pins each TP rank to its GPU's NUMA die using *host* CPU
   IDs (0–191 on a 192-CPU node). With `--cpus-per-task=64` the container
   cpuset shrank to a 32-CPU subset, so ranks 2–3 crashed with
   `ValueError: CPU number 168 is not eligible; choose between [0, …, 31]`.
   **Fix:** removed `--cpus-per-task` from the header; the srun step passes
   `--cpus-per-task="${SLURM_CPUS_ON_NODE}"` (=192) so the container sees all
   CPUs and the affinity pins resolve.

2. **TMPDIR on home quota** — Slurm defaults `TMPDIR=/users/<u>/.tmp` (home is
   quota-limited). sglang's tempfile cleanup on a quota dir raises
   `OSError [Errno 39] Directory not empty`. **Fix:** `TMPDIR` is set to
   `${DEPLOY_DIR}/cache/tmp` (on `/capstor`) in both the EDF `[env]` and the
   SGLang wrapper.

3. **aiter RoPE KeyError** — `SGLANG_USE_AITER=1` (image `/etc/environment`)
   makes aiter's `get_rope` index `rope_scaling["original_max_position_
   embeddings"]`, which GLM-4.7-Flash's config omits. **Fix:**
   `export SGLANG_USE_AITER=0` in the wrapper; sglang's own `get_rope` handles
   missing keys gracefully.

4. **CUDA-graph / warmup instability on ROCm** — cuda-graph capture and server
   warmup SIGKILL the scheduler ~35 s after weight load with GLM-4.7-Flash's
   MTP (nextn) layers on MI300A (confirmed vs the known-good bristen
   `sglang_subprocess-73057.sh`, which also forces both off). **Fix:**
   `DISABLE_CUDA_GRAPH=1` and `SKIP_SERVER_WARMUP=1` are the defaults.

5. **MI300A integrated-memory accounting (the subtle one)** — MI300A is an APU;
   PyTorch reports `is_integrated=True` for every GPU. sglang's
   `get_available_gpu_memory()` then uses `psutil.virtual_memory().available`
   (the **whole-node** ~428 GB) instead of per-GPU `mem_get_info()` (137 GB).
   With `mem_fraction_static=0.85` each of 4 TP ranks computes a ~301 GB KV
   budget; 4 × 301 ≫ 365 GB physically free → cgroup OOM (SIGKILL, exit -9)
   ~35 s after `Load weight end`, with no Python traceback. **Fix:** the
   sbatch writes `sglang_launcher_mi300a.py` — a thin launcher that wraps
   `torch.cuda.get_device_properties()` in a proxy reporting
   `is_integrated=False`, so sglang uses the correct 137 GB-per-GPU value.
   Result: each rank gets ~87 GB KV (1.7 M-token total budget) with 17.8 GB
   free, exactly like a discrete GPU.

## Files

| File | Purpose |
|------|---------|
| `sglang-rocm.toml` | EDF: image (by digest), mounts, ROCm env |
| `serve_glm_47_flash_sglang.sbatch` | One self-contained sbatch: per-rank SGLang wrapper (with MI300A memory fix + transformers upgrade) + otela worker + (optional) vmagent |
| `sglang_launcher_mi300a.py` | Standalone copy of the MI300A `is_integrated=False` launcher (also generated inline by the sbatch) |

## Submit

```bash
# default: 1 node, TP=4, served as zai-org/GLM-4.7-Flash
sbatch serve_glm_47_flash_sglang.sbatch

# scale out: one OpenTela peer per node, each a TP=4 replica
sbatch --nodes=4 serve_glm_47_flash_sglang.sbatch
```

First run pulls ~26 GiB into `$SCRATCH/.edf_imagestore` (shared, cached for
every later job). To pre-warm from the login node:

```bash
enroot import docker://lmsysorg/sglang:v0.5.16-rocm720-mi30x
```

## Verify

This is a **local deployment on the Alps mesh** — the bootstrap
`/ip4/148.187.108.178/...` is a peer running on Alps itself, not the public
`api.opentela.ai`. The compute nodes' `:8080` is not routable from the login
node, so health checks run from inside the allocation.

```bash
# job + per-rank OpenTela logs (look for: opentela_started, sglang_ready,
# a Peer ID: line, and 200 OK on POST /v1/chat/completions = requests routed
# in from other peers on the mesh)
tail -f /capstor/scratch/cscs/xyao/glm47-flash-sglang-beverin/logs/opentela-<JOB>-*.log

# direct SGLang health from inside the allocation (login node can't reach it):
srun -p mi300 -A a-infra02 -N1 -n1 --time=00:02:00 --overlap \
  bash -lc 'curl -s http://nid00XXXXX:8080/get_model_info | python3 -m json.tool'

# once a peer is registered, other peers on the mesh route to it; test from
# any peer that can reach the local OpenTela head:
curl -s http://<alps-head>/v1/service/llm/v1/models -H "X-Otela-Model: zai-org/GLM-4.7-Flash"
```

## Knobs (env, all overridable)

`MODEL`, `SERVED_MODEL_NAME`, `SGLANG_PORT`, `TP_SIZE`, `MEM_FRACTION_STATIC`,
`MAX_MODEL_LEN`, `MAX_RUNNING_REQUESTS`, `SGLANG_REASONING_PARSER` (default
`glm45`), `SGLANG_TOOL_CALL_PARSER` (default `glm47`), `DISABLE_CUDA_GRAPH`,
`SKIP_SERVER_WARMUP`, `LOAD_FORMAT`, `TRANSFORMERS_VERSION` (default `5.12.1`,
no-op when the image already ships it), `TRANSFORMERS_INSTALL_MODE`.
