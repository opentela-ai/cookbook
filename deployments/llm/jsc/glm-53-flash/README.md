# GLM-5.3-Flash on JSC (Jupiter Booster, GH200) via OpenTela

Serve **zai-org/GLM-5.3-Flash** (~306 GiB FP8 e4m3 dynamic, multimodal
vision+text, DSA + linear + KDA hybrid attention, 45 layers / 288 routed
experts top-8, MTP) on **JSC Jupiter Booster** (4x GH200 / node, aarch64
sm_90, InfiniBand) and register it on the OpenTela mesh — one
self-contained sbatch.

Ported from the **validated clariden recipe** (`deployments/llm/clariden/
glm-53-flash/` — same GH200 GPU, same `lmsysorg/sglang:glm-5.3-flash`
image, 12 h serving runs); site mechanics re-derived from the
**kimi-k3 JSC recipe** (`deployments/llm/jsc/kimi-k3/` — same cluster,
account, relay, apptainer pattern). Differences vs clariden: apptainer
instead of enroot/EDF, NCCL over InfiniBand (ib0, GDR-write, no SHARP)
instead of Slingshot/CXI, login-node relay for mesh egress, `/e/scratch`
paths. Differences vs kimi-k3: the model, the image, the tilelang DSA
pins + `sitecustomize.py` FA3 patches (aarch64 image has no
`sgl_kernel.flash_ops`), `UNBALANCED_MODEL_LOADING_TIMEOUT_S`, and the
mandatory pre-registration generation probe.

## Files

| file | purpose |
|---|---|
| `serve_glm_53_flash_otela_jsc.sbatch` | the deployment: preflight → engine → gen probe → bench → otela worker |
| `build_glm_5_3_flash_image.sh` | login-node: apptainer build of the .sif (once) |
| `download_glm_5_3_flash.sh` | login-node: HF snapshot download, resumable (once) |
| `gen_correctness.py` | the mandatory pre-registration generation probe (stdlib-only) |

## One-time staging (JSC login node)

```bash
# 1. image (~10-25 min)
bash deployments/llm/jsc/glm-53-flash/build_glm_5_3_flash_image.sh
# 2. weights (~2 min at JSC's HF bandwidth; 306 GiB under /e/scratch)
bash deployments/llm/jsc/glm-53-flash/download_glm_5_3_flash.sh
# 3. servekit checkout (cold-start profile + verification bench)
[ -d /e/scratch/reformo/yao4/otela-llm/servekit/src/servekit ] || \
  git clone --depth=1 https://github.com/eth-easl/servekit \
    /e/scratch/reformo/yao4/otela-llm/servekit
# 4. otela binary (v0.2.4 arm64; shared with kimi-k3)
/e/scratch/reformo/yao4/otela-llm/opentela-arm64 --help >/dev/null
# 5. the shared login-node relay (us-managed, outlives jobs)
bash /e/scratch/reformo/yao4/otela-relay/start_relay_jsc.sh status
```

The sbatch's preflight prints the exact missing-command for any of these.

**Why a relay:** JSC compute nodes have no outbound internet (not even
Cloudflare:443); the login node does. The relay (tmux on the login node,
peer `QmakkeU7ZYN…CPVpEvSqzgd` on `10.128.1.1:45195`) is started by
`deployments/llm/jsc/kimi-k3/start_relay_jsc.sh` and shared by all jobs.
The serving sbatch reads `/e/scratch/reformo/yao4/otela-relay/
relay.multiaddr` (or `$OTELA_RELAY_ADDR`).

## Submit

**Fast first bring-up (dummy weights, ~10-15 min, no registration):**
isolates the full init → JIT → forward → HTTP pipeline from the multi-hour
real-weight cold start:

```bash
sbatch --export=ALL,LOAD_FORMAT=dummy,GEN_CORRECTNESS_SMOKE=1,HEALTH_TIMEOUT=1800 \
       serve_glm_53_flash_otela_jsc.sbatch
```

**Real run (TP4 on 1 node — the clariden-validated shape):**

```bash
sbatch serve_glm_53_flash_otela_jsc.sbatch
```

**2 nodes (TP4 x PP2, more KV headroom — TODO(unverified) end-to-end):**

```bash
sbatch --nodes=2 serve_glm_53_flash_otela_jsc.sbatch
```

Logs: `/e/scratch/reformo/yao4/glm-53-flash/logs/serve-<jobid>.out`;
run dir: `/e/scratch/reformo/yao4/glm-53-flash/run-<jobid>/`
(`otela.log`, `gen_correctness_<jobid>.json`, `coldstart*.json`).

## Verify

```bash
# on JSC: registration markers (worker registered + relay reservation)
grep -E "Registering LLM service|Relay reservation|registered relay" \
  /e/scratch/reformo/yao4/glm-53-flash/run-<jobid>/otela.log
# from any meshed machine (e.g. via the gateway or an otela head)
curl -s https://api.opentela.ai/v1/service/llm/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/GLM-5.3-Flash","messages":[{"role":"user","content":"ping"}],"max_tokens":32}'
```

The engine serves HTTP on the head node at `$SERVICE_URL` (see
`last_service.env`) for the full 12 h window; renewal = submit a fresh job
before walltime (the `B:TERM@120` signal + `stop_otela` trap publish a
clean LEFT).

## Key knobs

| knob | default | notes |
|---|---|---|
| `TP_SIZE`/`PP_SIZE`/`EP_SIZE` | 4 / NNODES / 4 | TP intra-node NVLink; PP per node |
| `GLM53_DSA_PREFILL`/`GLM53_DSA_DECODE` | tilelang | only kpool>1-legal DSA backend present on aarch64 |
| `GLM53_MM_ATTENTION` | triton_attn | fa3 absent in this image |
| `MOE_BACKEND` | auto | `--moe-runner-backend` |
| `CTX_LEN`/`MEM_FRAC` | 262144 / 0.88 | pi's catalog needs > 131072 |
| `LOAD_FORMAT` | auto | `dummy` = random weights |
| `GEN_PROBE`/`GEN_CORRECTNESS_SMOKE` | 1 / 0 | mandatory gate; `SMOKE=1` pairs with dummy |
| `HEALTH_TIMEOUT` | 10800 | 3 h budget for 306 GiB load |
| `ENABLE_METRICS` | off | see DuplicateTimeseries note in the sbatch |
| `HICACHE_ENABLE` | 0 | Grace LPDDR prefix offload |
| `GLM53_SPECULATIVE` | 0 | EAGLE draft path unvalidated on sm_90 |
| `DISABLE_CUDA_GRAPH`/`SKIP_SERVER_WARMUP` | 0 | capture-crash escape hatches |
| `SERVEKIT_BENCH` | 1 | pre-registration bench C=16 n=64 |
| `OTELA_REGISTER` | auto | auto/1/0: dummy-weight engines never hit the mesh; smoke run ends the job cleanly |

## First bring-up (2026-10-07) — verified end-to-end

- Weights: 306 GiB in **126 s** from HF (login node); engine-side shard
  load **163 s** (4 ranks, /e/scratch EXA). Image build ~45 min (11 GB SIF).
- Dummy smoke (job 2215846): pipeline healthy in **382 s**, SMOKE 6/6,
  bench OK, registration correctly skipped, clean self-cancel.
- Real run (job 2217607, TP4 on jpbo-078-36): healthy in **362 s** (JIT
  cached), gen probe **5/6 crisp 3/3 PASS**, bench **333 output tok/s**
  (C=16, n=64, in 1024 tok / out 256 tok), `max_total_num_tokens=286592`,
  `max_running_requests=17` (the clariden shape). Worker registered via
  the relay (`Registering LLM service` + `Relay reservation … succeeded`)
  and **api.opentela.ai served correct completions from JSC** within
  ~2 min of registration.
- Findings fixed along the way (all committed):
  1. `HEAD_IP` must come from `getent hosts` — scontrol's `NodeAddr=` line
     splits into `"IP NodeHostName"`, and c10d hangs resolving it (2215062).
  2. `otela.sh` needs `chmod +x` — the worker srun exec's the path and dies
     with exit 126 otherwise (2217107).
  3. apptainer remaps `HOME` to the passwd home (`/e/home`, NFS quota)
     regardless of the exported HOME — pin `SGLANG_CACHE_DIR` to /e/scratch
     so the DeepGEMM JIT session (10-20 min first run) is paid once.
- Scheduling: a 12 h `--time` backfills poorly on busy days (planned-node
  reservations eat idle capacity); the 4 h submit started in ~2 min.
  Use `--time` to trade window length for queue latency; renewal is the
  documented pattern anyway.

## Carried findings (why the defaults are what they are)

- **tilelang DSA pins + sitecustomize patches** — the aarch64 image has no
  `sgl_kernel.flash_ops` (FA3) and no tensorrt_llm; `dsa_index_kpool > 1`
  (multimodal) hard-errors on every forward without a legal backend, and
  the DSA full-MHA prefill calls FA3 unconditionally on sm_90. See the
  sbatch header + `sitecustomize.py` comments (clariden jobs
  3208159/3208291/3209156/3209753/3209880/3210211).
- **`UNBALANCED_MODEL_LOADING_TIMEOUT_S=3600`** — sglang's post-load
  barrier kills ranks exceeding 480 s; 306 GiB off /e/scratch can exceed
  that on the slowest rank (kimi-k3 fix 10).
- **NCCL GDR-write + `NCCL_SOCKET_FAMILY=IPv4` + ib0** — NCCL's default
  read path stalls on the JSC IB fabric; ib0 peers aren't in DNS (kimi-k3
  JSC). No SHARP: cu13 image, cu12-linked libsharp, and PP needs no
  aggregated multicast.
- **mandatory gen probe before registration** — /health 200 does not prove
  generation (deepseek-v4 answered /health and 502'd; glm-5.2's DSA decode
  hung at 100 % GPU after /health).
- **`OTELA_SEED=$$`** — a constant seed reproduces the same peer id, which
  cannot re-register until its LEFT tombstone expires in the head's CRDT
  (clariden renewal stall). Per-job config dir (`$RUNDIR/otela`) + random
  seed = fresh identity per submit.
- **full-node cpuset (288 CPUs)** — the image pins TP ranks to GPU NUMA
  domains by host CPU id; a shrunken cpuset raises "CPU number N is not
  eligible" and kills ranks (clariden).
- **`B:TERM@120` + `stop_otela`** — without the trap the otela worker is
  SIGKILLed at walltime, publishes no LEFT, and the head keeps a
  `connected: true` row round-robining traffic into a dead endpoint
  (kimi-k3 JSC).
- **metrics off by default** — PyTorch's NCCL watchdog re-registers
  Prometheus metrics on a collective timeout → `DuplicateTimeseries` crash
  (kimi-k3 JSC). Nothing in this pipeline reads /metrics.
