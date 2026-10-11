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

The handover script automates the rolling 12h cadence (submit the next
window only when needed, verify health + registration): see
[`scripts/renew_glm_5_3_flash.sh`](../../../../scripts/renew_glm_5_3_flash.sh) / its
[usage doc](../../../../scripts/renew_glm_5_3_flash.md).

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
   (`meta/common/gen_correctness.py`, shared by the glm-5.2/5.3 recipes) *before*
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

The sbatch starts the otela worker automatically in step 4. The worker is
the **latest official OpenTela arm64 release** (staged on `/iopsstor` by the
shared [`meta/common/download_otela.sh`](../../../../meta/common/download_otela.sh),
run automatically by `up.sh`, or via the renew script's `--update-otela`). It reads a generated per-job `cfg.yaml`
(`mode: node`, `role: worker`, `reachability: private`, `loglevel: debug`,
`cleanslate: true`) and joins the direct-mesh bootstrap — Clariden compute
has outbound internet, so **no relay is needed**. Replaces the hand-staged
`sai-v0.0.6` fork, now **retired** — nothing references it anymore (the
old binary at `$OTELA_DIR/opentela-arm64` can be deleted). Rollback is a
pinned previous *official* release: `OTELA_VERSION=v0.2.5` re-stages it
(keeps `opentela-arm64-latest` untouched), then submit with
`OTELA_BIN=$OTELA_DIR/opentela-arm64-v0.2.5`.

| Where | Value |
|-------|-------|
| `--config-dir` | per-job `$RUNDIR/otela` (cfg.yaml + libp2p key `keys/id`) |
| `--service.name` | `llm` (`OTELA_SERVICE_NAME`) |
| `--service.port` | `$SERVE_PORT` (the sglang head HTTP port) |
| `--label model=` | `zai-org/GLM-5.3-Flash` (`SERVED_MODEL_NAME`) |
| `--label cluster=` / `launched_by=` | `clariden` / `$USER` |
| `--bootstrap.static` | `$OTELA_RELAY_ADDR` (an Alps OpenTela peer, default `/ip4/140.238.223.116/tcp/43905/p2p/Qm…`) |
| `seed` | `OTELA_SEED` (default: randomized per job) — fresh libp2p identity per job |

v0.2.6 differences that matter here (source-audited): the worker
health-checks the engine and fetches `/v1/models` to derive `model=<id>`
identity groups automatically (re-checked every 30 s, advertisement
withdrawn after 3 failures); the BadgerDB lives at `$HOME/.ocfcore`, which
`otela.sh` overrides to a per-job dir so the store dies with the job;
`security.require_signed_binary` defaults to `true` and is explicitly
disabled in `cfg.yaml` for compatibility with older unsigned mesh peers.

**OpenTela Cloud account linking** (new in v0.2.6, best-effort): each
job's peer id can be bound to your OpenTela Cloud account with a scoped
deploy key (`otd-…`, minted in the console under Account → Instances →
Deploy keys). The recipe resolves it from `OTELA_DEPLOY_KEY` (env
override; `none` disables) or the `OTD_API_KEY` export in `~/.zshrc`
(parsed directly — the batch shell is bash and never sources it), writes
it into the per-job `cfg.yaml` (`instance.deploy_key`, `chmod 600`), and
runs `otela instance link` (label `clariden-glm53-<jobid>`) before
`start`; boot then re-confirms the link idempotently. Linking never
gates anything: a revoked/exhausted/invalid key logs a warning and the
worker continues standalone. **Cost:** each job mints a fresh peer id, so
every renewal claims a new console instance and spends one deploy-key
use (~2/day at the 12h cadence — size the key's use budget accordingly;
same-peer re-links would be free, but a pinned peer id re-hits the
LEFT-tombstone stall on every handover). The renew script's status output
reports the link state on the `account-link:` line.

> **Identity is per-job by design (v0.2.6):** the per-job `--config-dir` +
> randomized seed mint a FRESH peer id on every job. A constant seed would
> reproduce the same peer id — which then cannot re-register after a
> graceful leave until the LEFT tombstone expires in the head's CRDT (the
> stall the euler/dgx-spark recipes hit). sai-v0.0.6 showed the same
> de-facto behavior (5 runs, 5 different peer ids). The `stop_otela`
> SIGTERM trap + a left-row cleanup handle the resulting stale rows.

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
  bash -c 'ps -eo pid,args | grep -a opentela-arm64 | grep -v grep'
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
| [`meta/common/gen_correctness.py`](../../../../meta/common/gen_correctness.py) | Mandatory pre-registration greedy generation probe (shared canonical copy) |
| `up.sh` | One-shot submit through `rcc --profile clariden` (real run by default; smoke via `--sbatch-args`) |
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
| `OTELA_BIN` | `$OTELA_DIR/opentela-arm64-latest` | latest official OpenTela arm64 release — staged by `up.sh` via `meta/common/download_otela.sh` (or the renew script's `--update-otela`); rollback = previous official release (`OTELA_VERSION=v0.2.5`, then `OTELA_BIN=$OTELA_DIR/opentela-arm64-v0.2.5`). The x86 `otela` is **not** usable on clariden (Exec format error) |
| `OTELA_DEPLOY_KEY` | from `OTD_API_KEY` in `~/.zshrc` | OpenTela Cloud deploy key (`otd-…`) for account linking; `none` disables — see [Account linking](#opentela-registration) above |
| `OTELA_RELAY_ADDR` | `/ip4/140.238.223.116/tcp/43905/p2p/Qm…` | direct-mesh bootstrap (clariden has outbound; no relay) |
| `OTELA_SEED` | randomized (`$$`) | fresh libp2p identity per job — avoids the LEFT-tombstone re-registration stall; set a constant only to pin the peer id |
| `HEALTH_TIMEOUT` | 10800 | 3 h pre-registration budget (62 shards x ~5 GiB off /capstor; up to ~2.5 h on a bad Lustre day) |
| `EP_SIZE` / `MOE_BACKEND` / `CG_DECODE` | `TP` / `auto` / `full` | EP=TP keeps expert dispatch intra-node; fp8-dynamic has a mature native path (unlike kimi-k3's mxfp4/marlin) |
| `WATCHDOG_TIMEOUT` / `SOFT_WATCHDOG_TIMEOUT` / `DIST_TIMEOUT` | 3600 / 600 / 3600 | see *Site fixes* below — the three watchdogs that can kill a healthy engine |
| `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` | 0 | this fork defaults it TRUE; a detokenizer-heartbeat stall would 503 a healthy engine and otela would withdraw the ad after 3 failed probes (job 3419248) |
| `HICACHE_ENABLE` / `RATIO` / `WRITE_POLICY` / `IO_BACKEND` | 0 / 1.0 / `write_back` / `kernel` | Grace-LPDDR prefix offload (see *Site fixes*); `write_through_selective` never backs up in the hybrid-SSM path |
| `GLM53_SPECULATIVE` | 0 | EAGLE/nextn draft path unvalidated on sm_90; the model's 1 nextn layer stays unused |
| `DISABLE_CUDA_GRAPH` / `SKIP_SERVER_WARMUP` | 0 / 0 | capture-crash escape hatches (GLM-5.2 MI300A workarounds); with capture probes, both must be 1 (see *Site fixes*) |
| `GLM53_NIC` | `hsn0` | Slingshot NIC for the NCCL/GLOO bootstrap socket |
| `UNBALANCED_MODEL_LOADING_TIMEOUT_S` | 3600 | sitecustomize patch of sglang's hardcoded 480 s post-load barrier |
| `GLM53_DIAG_DIR` | `<cookbook>/meta/diag/glm53` | shared diagnostic harness (comp_capture/capture_probe) — identical copies on beverin + clariden |
| `GLM53_CAPTURE_PROBE` / `GLM53_COMP_CAPTURE` | 0 / 0 | beverin-vs-clariden bisect: deterministic ~2k-token prefire before the gen probe / layer-I/O capture in the engine process |

## Site fixes (why the defaults are what they are)

- **THE kpool>1 HARD GATE (jobs 3208159 / 3208291 / 3209156).** GLM-5.3 is
  multimodal with `dsa_index_kpool > 1`; sglang guards it twice:
  `_resolve_kpool_tail_backend` (perf override `flashmla_sparse -> fa3` on
  sm_90 — crashes at capture when FA3 is absent) and
  `_check_kpool_tail_backend` (hard raise on EVERY forward unless the DSA
  backend is fa3/tilelang/trtllm — `flashmla_sparse` does not implement the
  tail-token append). On this aarch64 image fa3 (`sgl_kernel.flash_ops`) and
  trtllm are MISSING; tilelang is present and handles the kpool tail
  natively ⇒ **tilelang is the only legal + present backend; both DSA knobs
  pinned** (forward_extend AND forward_decode each hit the gate). Forcing
  `flashmla_sparse` (the kimi-k3 fix) is WRONG here — job 3209156 proved it
  trips the hard gate.
- **FOUR independent FA3 paths on this image** (aarch64 ships no
  `sgl_kernel.flash_ops`): (1) kpool-tail perf override (sitecustomize
  availability-guard, INERT under the tilelang pins — defensive); (2) the
  DSA full-MHA prefill hard-codes FA3 on SM90 (job 3209880; no CLI
  override — the SM100+ branch is a Blackwell-only TRT-LLM kernel, job
  3210211 "Unsupported architecture"; sitecustomize swaps a PyTorch SDPA
  ragged loop — fires only when fa3 is absent); (3) the multimodal vision
  tower defaults to fa3 on sm_90 (job 3209753 — pinned `triton_attn`);
  (4) the DSA sparse path itself, already bypassed by the tilelang pins.
- **Three watchdogs kill a healthy engine on transient stalls** (from
  kimi-k3 clariden): the HARD scheduler watchdog (300 s default; job
  3002366 died after 8 h healthy on one PP metadata broadcast stall) →
  3600; the SOFT watchdog (600 s stack dump, no crash — the diagnostic);
  and PyTorch's ProcessGroupNCCL watchdog (`--dist-timeout`, default 600 s;
  job 3018155 fired ~5.5 h before the scheduler watchdog on the same PP
  SEND stall) → 3600.
- **deep_gemm JIT cache self-heal (jobs 3632183/3632230, 2026-10-10).**
  deep_gemm hard-fails at the first cached kernel missing `kernel.cu` or
  `kernel.cubin` ("Corrupted JIT cache directory"), lazily — one bad entry
  = one dead job. The shared EDF HOME cache is swept by the sbatch before
  launch: any incomplete entry quarantines the whole cache (~minutes of
  re-JIT, proven cheaper than a dead 12 h job).
- **HiCache math.** GH200's Grace LPDDR (~870 GiB) hangs off NVLink-C2C at
  ~900 GB/s (not Slingshot). HiCache offloads EVICTED prefixes there —
  grows the prefix cache, not `max_total_num_tokens`. This is what makes a
  1-node TP4 serve viable (~19 GiB HBM headroom after 306 GiB weights).
  Verified on kimi-k3 clariden (jobs 3008263/3009024/3010201): no measured
  throughput cost, ~59 GB/node LPDDR at ratio 1.0.

## See also

- [Bristen GLM-5.3-Flash README](../../bristen/glm-53-flash/README.md) — why the same model **cannot** be served on A100 (SM80): below the FP8-compile floor (unpatched DSA kpool `fp8e4nv` JIT) and the A100-80GB HBM floor.
- [`../kimi-k3/`](../kimi-k3/) — the kimi-k3-on-clariden recipe this topology (one distributed engine + one head otela worker) was carried from.
