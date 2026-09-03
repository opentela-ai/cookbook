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
but is **not yet validated end-to-end** (tracked #2); add a node only if you need more
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

Routed serving **registered successfully** in job 3219525 (`model=zai-org/GLM-5.3-Flash`, served ~11h40m). The routed `200` confirmation has since closed: in job 3270164 the gateway routed end-to-end repeatedly with `max_completion_tokens: 131072` + `reasoning_effort: high` (8/8 HTTP 200s via `api.ada.ai`) and the full `pi` agent path (`--provider ada --model zai-org/GLM-5.3-Flash`) returned clean `stopReason:"stop"` turns.

## Measured serving performance (job 3270164, nid006990, 2026-09-03)

Measured with [`bench_serving.py`](bench_serving.py) from the Clariden login node against the engine head directly (`http://172.28.37.184:30000`), streaming `/v1/chat/completions`, `temperature=0`, `ignore_eos` (fixed output length), `stream_options.include_usage` for exact token counts. Each level is a closed batch (all requests launched at once) after a warmup request; engine-side numbers are parsed from the sglang serve log's `Decode batch ... gen throughput` lines inside each level's wall-clock window. Raw results: [`bench_sweep_1k.json`](bench_sweep_1k.json), [`bench_longinput_8k.json`](bench_longinput_8k.json).

**Decode sweep** — ~1K-token input (actual mean 985), 256 output tokens, 224/224 requests OK:

| c | TTFT p50/p95 (s) | e2e p50 (s) | TPOT p50 (ms) | agg out tok/s | agg total tok/s | engine decode mean/max (tok/s) | running max |
|---|------------------|-------------|---------------|---------------|-----------------|-------------------------------|-------------|
| 1 | 0.20 / 0.81 | 2.76 | 9.6 | 67 | 324 | 180 / 209 | 1 |
| 2 | 0.37 / 0.55 | 3.27 | 11.3 | 158 | 765 | 232 / 284 | 2 |
| 4 | 0.46 / 0.63 | 3.18 | 10.6 | 319 | 1,548 | 384 / 443 | 4 |
| 8 | 0.39 / 0.57 | 2.90 | 9.8 | 697 | 3,378 | 744 / 818 | 8 |
| 16 | 0.63 / 0.68 | 6.61 | 17.7 | 617 | 2,991 | 934 / 1,552 | 16 |
| 32 | 3.56 / 6.70 | 6.82 | 12.7 | 909 | 4,406 | 1,051 / 1,469 | 17 |
| 64 | 3.85 / 11.45 | 7.34 | 13.6 | 1,095 | 5,307 | 1,146 / 1,596 | 17 |

**Long-input spot check** — 7.6K-token input (actual mean 7,564), 128 output tokens:

| c | TTFT p50/p95 (s) | e2e p50 (s) | TPOT p50 (ms) | agg out tok/s | agg total tok/s |
|---|------------------|-------------|---------------|---------------|-----------------|
| 1 | 0.22 / 0.22 | 1.35 | 8.9 | 95 | 5,677 |
| 8 | 0.59 / 0.59 | 2.09 | 12.1 | 490 | 29,434 |

Interpretation:

- **Scaling is linear up to c=8**: 67 → 697 agg out tok/s (~87 tok/s per stream at c=8), TPOT stays ~10 ms/token. This is the regime pi/agentic traffic lives in — sub-second TTFT even with 7.5K-token prompts (DSA/tilelang prefill ≈ 30K tok/s).
- **Decode step time grows ~1.8× from bs=1 to bs=16** (9.6 → 17.7 ms/token), so per-stream speed halves but instantaneous aggregate still rises (~905 tok/s at bs=16).
- **Hard cap: `max_running_requests=17`** (Mamba state cache: `max_mamba_cache_size=88`, 5 state slots/request). Above 17, requests queue instead of batching: agg throughput plateaus at **~1.0–1.1K out tok/s (~5.3K total tok/s)** — engine steady decode ≈ 1.15K tok/s, transient max ≈ 1.6K tok/s — while TTFT p95 grows with queue depth (6.7 s @ c=32, 11.5 s @ c=64).
- **Guidance**: for interactive/agentic use (c ≤ 4) expect ~0.2–0.5 s TTFT and ~10 ms/token. For batch workloads, cap client concurrency at 17 to avoid queue TTFT, or raise the cap on the next deployment via `--max-mamba-cache-size` / `--mamba-full-memory-ratio`, or halve state size with `--mamba-ssm-dtype bfloat16` (unvalidated here).

Reproduce (login node, engine head IP from `last_service.env`):

```bash
python3 bench_serving.py \
  --url http://<HEAD_IP>:30000/v1/chat/completions \
  --model zai-org/GLM-5.3-Flash \
  --levels 1,2,4,8,16,32,64 --out-tokens 256 --input-tokens 1450 \
  --serve-log /capstor/scratch/cscs/$USER/glm-53-flash/logs/serve-<JOBID>.out \
  --save bench_sweep_1k.json
```

## Stopping a job cleanly (verified 2026-09-03)

The trap path (`B:TERM@120` at wall-time end) fires `stop_otela` correctly, but a
**manual step-level signal may not**: `scancel -s TERM <jobid>.<otela-step>` on
job 3273975 killed the worker with **no LEFT published** (no drain logged,
`otela.log` ends abruptly). The head de-facto stopped routing to that peer
immediately (zero routed traffic after worker death), but the row-level risk is
real. The reliably graceful manual procedure is to TERM the otela **process**
inside the running container, confirm LEFT, then cancel the job:

```bash
IMG=/capstor/scratch/cscs/$USER/glm-53-flash/images/sglang-glm-5.3-flash.aarch64.sqsh
# find the worker pid
srun --jobid=<JOBID> --overlap --gres=none --nodes=1 -w <HEAD_NODE> \
  --container-image=$IMG --container-name=glm-5.3-flash-clariden \
  bash -c 'ps -eo pid,args | grep -a otela-arm64 | grep -v grep'
# graceful leave (~10 s: AnnounceLeave -> 5 s CRDT drain -> shutdown)
srun --jobid=<JOBID> --overlap --gres=none --nodes=1 -w <HEAD_NODE> \
  --container-image=$IMG --container-name=glm-5.3-flash-clariden \
  bash -c 'kill -TERM <OTELA_PID>'
sleep 15
grep -acE "Announcing myself as LEFT|Leaving network" \
  /capstor/scratch/cscs/$USER/glm-53-flash/run-<JOBID>/otela.log   # must be >= 1
scancel <JOBID>
```

With the worker dead but the engine still up, the head stops routing to that
peer at once (observed on 3273975) — so even the ungraceful case is mitigated
by keeping the job alive a few minutes before `scancel`.

## Files

| File | Purpose |
|------|---------|
| `serve_glm_5_3_flash_otela_clariden.sbatch` | Self-contained Slurm batch: container/EDF setup, preflight, one distributed sglang engine, mandatory gen probe, otela registration, SIGTERM leave trap |
| `glm-53-flash-clariden.toml` | enroot EDF — mounts + env + `com.hooks.aws_ofi_nccl.enabled=true` (bind-mounts the aws-ofi-ccl plugin + CXI provider for native Slingshot) |
| `build_glm_5_3_flash_image.sbatch` | One-time enroot import of `docker://lmsysorg/sglang:glm-5.3-flash` (arm64, sm_90 cubins for GH200) to a local `.sqsh` |
| `download_glm_5_3_flash.sbatch` | One-time HF weights download to `/capstor` (~306 GiB, 62 shards) |
| `gen_correctness.py` | Mandatory pre-registration greedy generation probe (reused from kimi-k3 / glm-5.2) |
| `bench_serving.py` | stdlib-only concurrency-sweep benchmark: streaming TTFT/TPOT/e2e + aggregate throughput per level, cross-checked against the engine's own decode-log throughput |
| `bench_sweep_1k.json` / `bench_longinput_8k.json` | Raw results of the 2026-09-03 measured-performance run above (job 3270164) |
| `verify_parsers.py` / `inspect_fa3_paths.py` / `verify_kpool_patch.py` | reasoning/tool-call parser + DSA/FA3/kpool path verification helpers |

## Knobs (env, all overridable)

| Knob | Default | Notes |
|------|---------|-------|
| `NNODES` / `TP_SIZE` / `PP_SIZE` | 1 / 4 / `NNODES` | one distributed engine; PP = nodes, TP4 keeps allreduce in-node (2-node PP2 = more KV headroom, unvalidated) |
| `MEM_FRAC` | 0.88 | 306 GiB/8 = ~38 GiB weights/GPU; ~58 GiB free |
| `CTX_LEN` | 262144 | catalog advertises max-out 131072 (pi sends `max_completion_tokens=131072`); gateway rejects any > advertised `max_model_len`. Live-KV budget 287104 at MEM_FRAC=0.88 (probe job 3269479), so a single 256K request fits at 91%. Raise toward 1M only with `HICACHE_ENABLE=1` |
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
