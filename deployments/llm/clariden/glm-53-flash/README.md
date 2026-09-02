# GLM-5.3-Flash on Clariden (SGLang, GH200) → OpenTela

Serves `zai-org/GLM-5.3-Flash` — FP8 e4m3 dynamic, ~306 GiB, multimodal
(vision + text), DSA + linear + KDA hybrid attention, 45 layers / 288 routed
experts top-8, MTP — on **CSCS Clariden** (4× NVIDIA GH200 120 GB
(≈96 GiB usable) / node, aarch64, Slingshot 11 via CXI libfabric,
**no InfiniBand**) with the upstream multi-arch `lmsysorg/sglang:glm-5.3-flash`
image, and registers it on the OpenTela mesh — one self-contained sbatch,
no sibling scripts.

## Why this site (and not A100)

GLM-5.3-Flash is **FP8-native end-to-end** — weights, MoE, *and* its DSA
attention kpool cache (`torch.float8_e4m3fn`). GH200 (SM90) has native FP8, so
**no upcast patch is needed** here; the DSA prefill runs `tilelang` natively.
This is the opposite of Bristen (A100, SM80): A100 has no native FP8, the
MoE/GEMM upcast patch only gets it to boot, and the unpatched DSA kpool
`fp8e4nv` kernel **fails at Triton JIT compile on the first forward** — below
the FP8-compile floor (and the A100-80GB HBM floor). See
[the Bristen README](../../bristen/glm-53-flash/README.md) for that diagnosis.

**One node (TP4, the default) is the validated config** (jobs 3213420 /
3219525, gen-probe `PASS 5/6`, served 12 h until the wall limit): 306 GiB
FP8 / 4 GPUs = **~76.5 GiB weights/GPU of ≈96 GiB usable** → ~20 GiB free
for KV pool + activations + JIT — tight but sufficient for single-stream /
low concurrency. **2 nodes (TP4 × PP2) widens KV headroom to ~58 GiB/GPU
but is not yet validated end-to-end**; add a node only if you need more
concurrency. TP4 keeps tensor-parallel all-reduce **inside one node's 4-GH200
NVLink domain**; the only cross-Slingshot traffic is pipeline send/recv
(point-to-point) — bandwidth-tolerant and CUDA-graph-safe. (Flat TP8 across
2 nodes boots but is ~1–5 tok/s and cannot be captured in a CUDA graph on
Slingshot/CXI — same finding as kimi-k3.)

## Quick start

```bash
# from the clariden login node — default: 1 node, TP4 (validated; one engine, no PP)
sbatch deployments/llm/clariden/glm-53-flash/serve_glm_5_3_flash_otela_clariden.sbatch

# 2 nodes, TP4 x PP2 (more KV headroom; NOT yet validated end-to-end)
sbatch --nodes=2 --export=ALL,NNODES=2,PP_SIZE=2 \
       deployments/llm/clariden/glm-53-flash/serve_glm_5_3_flash_otela_clariden.sbatch

# 4 nodes, TP4 x PP4
sbatch --nodes=4 --export=ALL,NNODES=4,PP_SIZE=4 \
       deployments/llm/clariden/glm-53-flash/serve_glm_5_3_flash_otela_clariden.sbatch

# FAST first-bring-up (recommended): dummy weights, generation smoke only, NO
# registration — isolates the full init -> JIT -> forward -> HTTP pipeline in
# ~10-15 min instead of the multi-hour real-weight cold start.
sbatch --export=ALL,LOAD_FORMAT=dummy,GEN_CORRECTNESS_SMOKE=1 \
       deployments/llm/clariden/glm-53-flash/serve_glm_5_3_flash_otela_clariden.sbatch
```

From your local machine via `rcc` (profiles `beverin`/`clariden` share
`/capstor`, so `rcc --profile beverin push` also updates these files):

```bash
rcc --profile clariden push
rcc --profile clariden job submit deployments/llm/clariden/glm-53-flash/serve_glm_5_3_flash_otela_clariden.sbatch
rcc --profile clariden job tail <JOBID> -f
```

## What it does, in order

1. **preflight** the image, weights (or `LOAD_FORMAT=dummy`), and the otela
   binary (prints the exact staging commands on failure).
2. **srun one sglang engine rank per node** inside the `glm-53-flash-clariden`
   container (enroot/EDF), forming a single distributed `TP4 × PP<NNODES>`
   engine. Only rank 0 (head) serves HTTP.
3. once the head's `/health` answers, run a **mandatory generation probe**
   (`gen_correctness.py`, reused from the kimi-k3/glm-5.2 recipes) *before*
   registration — `/health` alone is not enough (deepseek-v4 / kimi-k3 /
   glm-5.2 lesson: an engine can answer `/health` while a broken kernel path
   serves garbage or hangs).
4. **only if the probe passes**, start ONE **otela worker** — also inside the
   container, on the head via `srun --overlap --gres=none` — which registers
   the `llm` service with the OpenTela mesh.
5. on job end / `scancel`, **SIGTERM (never KILL)** the otela step
   (`#SBATCH --signal=B:TERM@120`) so it announces `LEFT` cleanly and the
   mesh stops routing traffic to a dead endpoint.

## OpenTela registration

The sbatch starts the otela worker automatically in step 4. The worker
(`otela-arm64`, sai-v0.0.6) reads a generated `cfg.yaml` (`mode: node`,
`role: worker`, `reachability: private`, `seed`, `loglevel: debug`) and joins
the direct-mesh bootstrap — Clariden compute has outbound internet, so
**no relay is needed**.

| Where | Value |
|-------|-------|
| `--service.name` | `llm` (`OTELA_SERVICE_NAME`) |
| `--service.port` | `$SERVE_PORT` (the sglang head HTTP port) |
| `--label model=` | `zai-org/GLM-5.3-Flash` (`SERVED_MODEL_NAME`) |
| `--label cluster=` / `launched_by=` | `clariden` / `$USER` |
| `--bootstrap.static` | `$OTELA_RELAY_ADDR` (an Alps OpenTela peer, default `/ip4/140.238.223.116/tcp/43905/p2p/Qm…`) |
| `seed` | `OTELA_SEED` (default `21`) — stable libp2p identity across restarts |

> **Identity quirk (observed, runs 3219525 + 4 prior):** the peer id is **not**
> pinned by the shared `HOME` — `$HOME/.config/opentela/id` and a fresh
> `$HOME/.ocfcore/ocfcore.Qm*.db` are regenerated on every start, so even with
> `OTELA_SEED` unchanged the last runs minted different peer ids
> (QmcrhK7z, QmPnaDQD, QmRAdK3p, QmQLa2Bu, QmdV1…). The `stop_otela` SIGTERM
> trap + a left-row cleanup handle the resulting stale rows.

The three places the model is named are kept equal — sglang
`--served-model-name`, otela `--label model=`, and the client request
`"model"` — per [`conventions/README.md`](../../../../conventions/README.md).

## Verify

Inside the allocation on the head node (direct, before registration):

```bash
srun --jobid=<JOBID> --overlap --gres=none --nodes=1 -n1 -w <HEAD> \
  --container-image=$IMAGE --container-name=glm-5.3-flash-clariden \
  bash -lc 'curl -s http://127.0.0.1:30000/health; echo; \
            curl -s http://127.0.0.1:30000/v1/models | python3 -m json.tool'
```

Routed through OpenTela (after registration; from any peer that can reach the
Alps mesh):

```bash
curl -s http://<alps-or-public-head>/v1/service/llm/v1/models \
  -H "X-Otela-Model: zai-org/GLM-5.3-Flash"
```

Routed serving **registered successfully** in job 3219525 (`model=zai-org/GLM-5.3-Flash`, served ~11h40m), but the only 3 routed requests observed
(~10h in) returned **HTTP 400 in ~22ms** — fast rejections, almost
certainly malformed external probes (missing `X-Otela-Model` or wrong model
name), not a serve failure. Direct localhost serving is gen-probe-validated;
a clean routed `200` is the one open confirmation to close on the next run.

## Files

| File | Purpose |
|------|---------|
| `serve_glm_5_3_flash_otela_clariden.sbatch` | Self-contained Slurm batch: container/EDF setup, preflight, one distributed sglang engine, mandatory gen probe, otela registration, SIGTERM leave trap |
| `glm-53-flash-clariden.toml` | enroot EDF — mounts + env + `com.hooks.aws_ofi_nccl.enabled=true` (bind-mounts the aws-ofi-ccl plugin + CXI provider for native Slingshot) |
| `build_glm_5_3_flash_image.sbatch` | One-time enroot import of `docker://lmsysorg/sglang:glm-5.3-flash` (arm64, sm_90 cubins for GH200) to a local `.sqsh` |
| `download_glm_5_3_flash.sbatch` | One-time HF weights download to `/capstor` (~306 GiB, 62 shards) |
| `gen_correctness.py` | Mandatory pre-registration greedy generation probe (reused from kimi-k3 / glm-5.2) |
| `verify_parsers.py` / `inspect_fa3_paths.py` / `verify_kpool_patch.py` | reasoning/tool-call parser + DSA/FA3/kpool path verification helpers |

## Knobs (env, all overridable)

| Knob | Default | Notes |
|------|---------|-------|
| `NNODES` / `TP_SIZE` / `PP_SIZE` | 1 / 4 / `NNODES` | one distributed engine; PP = nodes, TP4 keeps allreduce in-node (2-node PP2 = more KV headroom, unvalidated) |
| `MEM_FRAC` | 0.88 | 306 GiB/8 = ~38 GiB weights/GPU; ~58 GiB free |
| `CTX_LEN` | 32768 | |
| `GLM53_DSA_PREFILL` / `GLM53_DSA_DECODE` | `tilelang` / `tilelang` | native FP8; the `flashmla_sparse` path that blocks Bristen is avoided here |
| `GLM53_MM_ATTENTION` | `triton_attn` | multimodal attention backend |
| `GLM53_REASONING_PARSER` / `GLM53_TOOL_CALL_PARSER` | `glm45` / `glm47` | gated by `GLM53_ENABLE_PARSERS=1` |
| `LOAD_FORMAT` | `auto` | `dummy` for a fast forward-path smoke (pair with `GEN_CORRECTNESS_SMOKE=1`) |
| `GEN_CORRECTNESS_SMOKE` | 0 | 1 = probe only, skip otela registration |
| `OTELA_BIN` | `$OTELA_DIR/otela-arm64` | sai-v0.0.6; the x86 `otela` is **not** usable on clariden (Exec format error) |
| `OTELA_RELAY_ADDR` | `/ip4/140.238.223.116/tcp/43905/p2p/Qm…` | direct-mesh bootstrap (clariden has outbound; no relay) |
| `OTELA_SEED` | 21 | stable libp2p identity (best-effort — see the identity quirk above) |

## See also

- [Bristen GLM-5.3-Flash README](../../bristen/glm-53-flash/README.md) — why the same model **cannot** be served on A100 (SM80): below the FP8-compile floor (unpatched DSA kpool `fp8e4nv` JIT) and the A100-80GB HBM floor.
- [`../kimi-k3/`](../kimi-k3/) — the kimi-k3-on-clariden recipe this topology (one distributed engine + one head otela worker) was carried from.
