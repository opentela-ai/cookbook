# Kimi-K3 on Clariden (SGLang, GH200) → OpenTela

Serve `moonshotai/Kimi-K3` (2.8 T-parameter hybrid MoE, mxfp4) on **CSCS
Clariden** (4× GH200 120 GB / node, aarch64, **Slingshot fabric — no
InfiniBand**, Slurm + enroot/EDF) and register it on OpenTela. The serving
stack is the JSC Kimi-K3 shape (one distributed SGLang engine across all
nodes + a single otela worker on the head) carried over to CSCS's container
engine and Slingshot NCCL transport. Unlike JSC, **no relay is needed**:
Clariden compute nodes reach the Alps OpenTela bootstrap directly, so the
worker dials it straight (like the Beverin recipe).

This README records **what was verified on this cluster** (image, EDF,
topology, NCCL env, otela registration, and measured throughput) and the exact
failures behind each non-obvious choice. The companion
`serve_kimi_k3_otela_clariden.sbatch` is the runnable artifact.

## TL;DR — the verified operating point

| | Value | Notes |
|---|---|---|
| **Production topology** | **TP4 × PP8** (8 nodes) | Keeps collectives intra-node on NVLink; only pipeline P2P crosses Slingshot. |
| **Throughput @ C=32** | **561 tok/s** aggregate | Measured, 1024-in / 256-out (JSC protocol). Curve flattens at C=32. |
| **Alternative** | TP32 flat (32 GPUs) | Boots, ~1–5 tok/s, `CG_DECODE=disabled` (fix 2). |
| **Free HBM per GPU** | ~30 GiB (of 96) | At `mem_fraction_static=0.90`. |
| **Fabric** | Slingshot (CXI) via aws-ofi-ccl-plugin | No IB verbs on the node. |
| **Container** | enroot/EDF, `lmsysorg/sglang:kimi-k3` (CUDA 13, arm64 .sqsh) | Imported once by `build_kimi_k3_image.sbatch`. |
| **OpenTela** | direct-mesh worker (arm64 `otela-arm64`), on head, inside the container | No relay (compute can reach the Alps bootstrap). |
| **Cold start** | **~105 min** (job 3000965) | Weight loading from /capstor dominates (~80–132 s/shard × 96 shards). |
| **Loading barrier** | **3600 s** (monkey-patched from 480 s) | `UNBALANCED_MODEL_LOADING_TIMEOUT_S` via `sitecustomize.py` (fix 10). |

### Verified benchmarks (TP4/PP8, 8 nodes, 1024-in / 1716-tok-actual prompt / 256-out)

The numbers below were taken with the legacy vendored `bench.py`: warmup
`4:8` discarded, then the sweep, 256 max output tokens, `ignore_eos`,
deterministic. Run inside the container on the head (nid007464) against
`127.0.0.1:30000` of job 3000965 (2026-08-04); HiCache OFF. Raw output:
`bench_3000965.jsonl` (in this directory).

**The harness is now `meta/bench/cbench.sh` (servekit bench)** — see
*Benchmark* below and `meta/bench/README.md`. The sbatch also runs a single
verification level (C=16, n=64) automatically after health, before OpenTela
registration, and writes a per-node cold-start profile
(`$RUNDIR/run.node<RANK>.json`). Reproduce the table with the standard
sweep `1:8 4:16 8:32 16:48 32:64 64:96`. Note: servekit measures prompt
length in **words** (~1.4 tokens/word), so its 768-word default ≈ the
1024-token prompt protocol used here — not identical, quote which you ran.

| Concurrency | n | Aggregate out tok/s | per-req tok/s | p50 lat (s) |
|---|---|---|---|---|
| 1  | 8  | 34.7  | 34.7 | 7.4  |
| 4  | 16 | 103.8 | 25.9 | 9.9  |
| 8  | 32 | 198.1 | 24.8 | 10.4 |
| 16 | 48 | 374.9 | 23.4 | 10.9 |
| 32 | 64 | **561.4** | 17.6 | 14.6 |
| 64 | 96 | 559.3 | 11.7 | 29.2 |

Linear-ish to C=16, knee at C=32, flat at C=64 (aggregate unchanged while
per-req drops 34 %) — **C≈32 is the operating point**. Essentially the same
curve the JSC recipe measures on the same model/hardware shape (542 tok/s @
C=32), i.e. Slingshot-PP costs nothing measurable next to IB-PP at this
scale. Single-request decode is *faster* than JSC's (34.7 vs 28.8 tok/s).

### HiCache (Grace-LPDDR host tier) — opt-in, not default

Tested end-to-end on K3/Clariden 2026-08-05 (jobs 3008263/3009024/3010201,
this image, `HICACHE_ENABLE=1`, `kernel` io). **It is safe to turn on, but
remains OFF by default** because the proven production run (2920471) used
`HICACHE_ENABLE=0` and the full-pool benefit is workload-dependent, not a
guaranteed free lunch.

- **Boots clean and adds no measurable throughput cost** at the verification
  level: 294.5–304.5 tok/s @ C=16; manual C=32/64 sweep = 489/491 tok/s
  (HiCache OFF reference: 374.9/561.4/559.3 on a different day; same-day
  HiCache-OFF prod: 219.2 @ C=16 — variance swamps any HiCache effect).
- **Host memory use is modest.** At ratio 1.0 and the default 3.73 M-token
  HBM pool, measured allocation is 8.6–17.2 GB KV + 1.8–2.0 GB mamba per
  rank (varies by PP stage) ≈ **59 GB/node** — comfortably inside the
  ~350 GB of free LPDDR. Ratio 2.0 would be ~118 GB/node.
- **Working policy: `write_back`.** `write_through_selective` is broken on
  this image: the host tier never leaves one page because the hit-count
  trigger never reaches `write_backup` in the UnifiedRadixCache hybrid-SSM
  path. `write_back` is now the default when HiCache is enabled.
- **Host-tier hits are real, but only when the prefix survives the LRU.**
  With a *capped* 262 k-token pool (`--max-total-tokens 262144`) the
  benchmark shows a clean **1.90 s → 0.28 s → 0.27 s** triplet (cold →
  HBM hit → post-eviction host hit, `cache_hit_rate` 0.99). With the full
  3.73 M-token pool, a 4.8 M-token unique flood fills the 3.73 M-token
  host tier to 99% and LRU-flushes even a hot probe, so the post-eviction
  probe returns at cold-compute speed. The host tier behaves as an
  opportunistic extension of the radix cache, not an archive of every old
  prefix.
- **Benchmark prefix caching via `/v1/completions`, not chat** — the K3
  chat template injects a variable system header, so identical
  `/v1/chat_completions` texts never share a token prefix and *never hit*
  (verified on both engines); raw completions of the same text hit
  immediately. Real chat traffic therefore only benefits from prefix
  caching up to the template-injected variable.

To enable: submit with
`--export=ALL,HICACHE_ENABLE=1,HICACHE_RATIO=1.0,OTELA_SERVICE_NAME=llm-hicache-test,OTELA_SEED=<new>`.
Test harness: `hicache_bench.py` (stdlib, runs from the login node).

## Why the `normal` partition

Clariden's GPU partitions (`normal`/`debug`/`low`) all expose the same
**4× GH200 120 GB / node** nodes (sinfo `gpu:4`, 288 logical CPUs, ~870 GB
RAM). `normal` (12 h limit) is the default for serving; `debug` (1:30, faster
queue) is used for the one-time image import. `low` (1-00:00:00, lowest
priority) is an option for long-running instances. Verified: `nvidia-smi -L`
on a `debug` node reports 4× `NVIDIA GH200 120GB`, 97871 MiB each.

## Site-specific fixes (all baked into the sbatch + EDF)

1. **No InfiniBand → NCCL over Slingshot.** Clariden has no `/dev/infiniband`
   and no IB verbs (`ibv_devinfo` empty); the high-speed fabric is Slingshot
   (NICs `hsn0`..`hsn3`, `/opt/cray/libfabric` CXI provider). NCCL's built-in
   IB transport is therefore unused. **Fix:** the EDF (`kimi-k3-clariden.toml`)
   carries `com.hooks.aws_ofi_nccl.enabled=true`. Per the CSCS resource-hook
   docs (software/container-engine/resource-hook/#hpe-slingshot-interconnect),
   the Clariden vCluster defaults to `com.hooks.netstack.source="artifact"`, so
   at container start the hook bind-mounts a standalone, **dynamically-linked
   (`+dl`) aws-ofi-ccl plugin + libfabric CXI provider** (default aarch64
   artifact: `gpu:cuda13,cxi:12.0.1,ofi:2.5.1,aws:1.18.0+dl`) and sets the
   CXI/NCCL env that helps prevent application stalls. On artifact source the
   `com.hooks.aws_ofi_nccl.variant` annotation is **ignored** (all artifacts are
   `+dl`); it would only apply on `source=host`, where the docs' recommended
   variant is `cuda-dl`, not the statically-linked `cuda13` the EDF previously
   set. The engine.sh now adds those `NCCL_*`/`FI_CXI_*` values as `:=`
   fallbacks that **yield to whatever the hook sets** (reconciled 2026-08);
   `NCCL_SOCKET_IFNAME=hsn0` is for the TCP bootstrap path only (the ofi plugin
   carries data). **Scope caveat — important:** this hook configures the NCCL
   **data plane only**. It does NOT touch Gloo, which carries the PP
   **control plane** over TCP (`hsn0`). The fix-11/12 PP-broadcast stalls
   (`Connection closed by peer` in `gloo/transport/tcp/pair.cc`, observed on
   3002366/3018155/3029640) are Gloo peer-disconnects the aws-ofi-nccl hook
   cannot prevent; that needs a separate Gloo-level lever (timeout/retry or a
   non-TCP PP control transport).
2. **Cross-node collectives cannot be captured on CXI (TP32 flat).** Under
   `TP_SIZE=32 PP_SIZE=1`, the cross-node MoE all-reduces fail CUDA-graph
   capture with `cudaErrorStreamCaptureInvalidated` (job 2914910), and the OFI
   plugin logs `"NET/OFI GIN only supports RDMA transport protocol"`. **Fix:**
   the default TP4 × PP8 keeps every captured collective intra-node on NVLink,
   so `CG_DECODE=full` works. If you run the flat TP32 shape, set
   `CG_DECODE=disabled` (it boots but is ~1–5 tok/s).
3. **FlashMLA for prefill AND decode (aarch64 ships no FA3).** This aarch64
   `sgl_kernel` has no `flash_ops` (only `flash_mla`). Setting only one of the
   three attention knobs cancels auto-resolution for the others → prefill
   silently falls back to FA3 and dies at `ImportError` (job 2914289).
   **Fix:** BOTH `--prefill-attention-backend flashmla` AND
   `--decode-attention-backend flashmla` (FlashMLABackend inherits prefill from
   FlashInferMLAAttnBackend). Do NOT add `--mamba-ssm-dtype bfloat16`: SM90
   uses float32 KDA state (FlashInfer GDN bf16 is SM100+ only).
4. **`marlin` MoE runner (mxfp4-packed).** The only SM90 backend that keeps
   K3's mxfp4 weights packed (W4A16). `auto` picks Triton-Kernels, whose
   `upcast_from_mxfp4()` dequantizes to bf16 and OOMs a 4-bit model on 96 GiB.
   (Same finding as the JSC recipe.)
5. **otela runs INSIDE the container on the head (host netns).** The
   distributed engine exposes HTTP only on rank 0 (head). CSCS enroot shares
   the host network namespace (verified: `hostname -I` inside the container
   returns the same `172.28.*`/`10.100.*` IPs as the host), so a separate
   `srun --overlap --environment --nodes=1 -w $HEAD` step reaches
   `localhost:$SERVE_PORT`. otela's HOME is the EDF's `/capstor/.../home` (not
   quota-limited `/users`), so its libp2p identity + badger DB persist cleanly
   across jobs.
6. **arm64 otela binary.** The x86 binary at the same deploy dir gives
   `cannot execute binary file: Exec format error` on aarch64. **Fix:** use
   `otela-arm64` (default `OTELA_BIN=/capstor/scratch/cscs/xyao/opentela/otela-arm64`;
   shared alternative `/capstor/store/cscs/swissai/infra01/ocf-share/otela-arm64`).
7. **Diskless node enroot scratch (image import).** Clariden GH200 compute
   nodes are diskless; the only node-local store is `/dev/shm`. `enroot import`
   on the read-only rootfs fails unless `ENROOT_DATA_PATH`/`CACHE_PATH`/
   `RUNTIME_PATH`/`TEMP_PATH` + `XDG_RUNTIME_DIR` are redirected to `/dev/shm/$USER`.
   **Fix:** `build_kimi_k3_image.sbatch` does this and unsets the stale
   `DBUS_SESSION_BUS_ADDRESS`.
8. **Writable caches off quota `/users`.** sglang tempfile cleanup on a
   quota-limited `/users` HOME raises `OSError [Errno 39] Directory not
   empty`, and an otela data dir on NFS can return ESTALE and freeze the CRDT
   (documented in the JSC recipe). **Fix:** the EDF points `HOME`,
   `HF_HOME`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR` at `/capstor/.../cache`.
9. **otela requires a JSC-style `cfg.yaml` (sai-v0.0.6 arm64).** Without a
   config file carrying `loglevel: debug`, `reachability: private`,
   `role: worker`, `seed`, `cleanslate: false`, and `solana.skip_verification:
   true`, otela appears to "hang" after the single INFO line `AXIOM_DATASET
   not set, tracing disabled` — but it is actually working (peer discovery,
   CRDT DNT sync, LLM service registration all proceed at DEBUG level, which
   is silent by default). 90 s of silence misled us into killing the process.
   **Fix:** the batch shell writes `$OTELA_CFG_DIR/cfg.yaml` directly (the
   `otela init` subcommand creates nothing on this build). Verified on job
   3000965: with the config file, otela synced 2000+ peers and registered the
   LLM service (`protocol/registrar.go:145 Registering LLM service: … connected
   localhost 30000 …`) within 45 s.
10. **`UNBALANCED_MODEL_LOADING_TIMEOUT_S` monkey-patch (sglang hardcoded 480 s).**
    sglang's `dist_barrier_after_load` (in
    `model_executor/model_runner_components/load_model_utils.py`) calls
    `dist.monitored_barrier(group=tp_group, timeout=UNBALANCED_MODEL_LOADING_TIMEOUT_S,
    wait_all_ranks=True)` where the constant is **hardcoded at 480 s** (8 min).
    The barrier is per-TP-group (GLOO, 4 ranks/node); only rank 0 has the
    timeout — non-zero ranks block indefinitely waiting for rank 0. If the
    slowest rank in a node happens to be non-zero (normal Lustre I/O
    variation — up to 29 min within-PP spread observed), rank 0 times out at
    480 s and raises `ValueError: TP rank 1 could finish the model loading,
    but there are other ranks that didn't finish loading.`, killing the whole
    job. This is NOT `--dist-timeout` (which controls NCCL init, not the
    weight-loading barrier). **Fix:** a `sitecustomize.py` in `$PATCH_DIR`
    (on /capstor, mounted into the container) patches the constant to 3600 s
    (1 h) at Python startup. `engine.sh` prepends `$PATCH_DIR` to `PYTHONPATH`
    (the EDF sets `PYTHONPATH=""`, clearing any batch-shell export, so this
    must be done inside `engine.sh` before `exec sglang`). Verified: the
    constant is 3600 in the running container (job 3000965, all 32 ranks
    reached `Load weight end` with a 29-min spread, zero exceptions).

11. **Hard scheduler watchdog killed a healthy 8 h-serving engine (job 3002366).**
    sglang's HARD scheduler watchdog (`ServerArgs.watchdog_timeout`, default
    300 s, `soft=False`) raises on any forward batch longer than the limit, and
    that raise kills the whole distributed engine. On job 3002366 (1M, TP4×PP8)
    the engine served healthy 04:20→12:24 (steady decode ~100 tok/s, CUDA graph
    on, `POST /v1 … 200 OK` through 12:23:53), then at 12:29:35 **every PP
    stage tripped simultaneously**:
        [2026-08-05 12:29:35 PP2 TP3 EP3] Scheduler watchdog timeout
           (self.watchdog_timeout=300, self.soft=False)
        … (PP2 TP1, PP7 TP2, PP7 TP3; 16 tracebacks total)
    The blocked scheduler thread was in `torch.distributed.broadcast_pyobj →
    _broadcast_reqs_across_ranks → recv_requests → event_loop_pp` — the
    per-iteration PP **request-metadata broadcast** (not a tensor collective;
    PP point-to-point was "always fine" in fix 2) hung >300 s. The raise tore
    the engine down; otela could not announce LEFT → stale `connected: true`
    registry row. Root cause of *that* stall is `TODO(unverified)` (Slingshot/
    CXI hiccup after 8 h? a specific request?); a >5-min hard kill on a healthy
    engine is the same anti-pattern fix 10 already rejects for weight loading.
    **Fix:** pass `--watchdog-timeout 3600` (raised hard limit; rides out
    transient stalls, a true permanent hang still dies in 1 h) **and**
    `--soft-watchdog-timeout 600` (dumps a stack trace at 10 min **without
    crashing**, so the next stall is diagnosable even when the engine survives
    it). Both are env knobs (`WATCHDOG_TIMEOUT` / `SOFT_WATCHDOG_TIMEOUT`) and
    were verified via `sglang serve --help`; `python -m sglang.launch_server`
    (now used here, wrapped by `servekit launch` when `PRESHARDED_ENABLE=1`)
    shares the same `ServerArgs`.

12. **NCCL process-group watchdog killed a healthy 5 h20 m-serving engine
    (job 3018155) — the very stall fix 11 tried to survive.** Fix 11 raised
    only sglang's **scheduler** watchdog (`--watchdog-timeout 3600`). The
    same transient PP pipeline stall has a second, lower floor: PyTorch's
    `ProcessGroupNCCL::Watchdog`, whose timeout comes from
    `init_process_group(timeout=…)` == sglang's `--dist-timeout` (default
    600 s). The recipe deliberately did NOT pass `--dist-timeout` (comment:
    “NOT passed to sglang by default”), so the NCCL watchdog sat at 600 s.
    Timeline: 3018155 served 10:47→16:08 (last `200 OK` 16:08:43), a PP
    `SEND` (`SeqNum=3678376`, `NumelIn=3584`) stalled and never completed,
    and at **16:18:52** — 600 s later, ~5.5 h before the scheduler watchdog
    would have tripped — every rank aborted:
        [rank1]:[E806 16:18:52.413 … ProcessGroupNCCL.cpp:689] [Rank 1]
          Watchdog caught collective operation timeout: WorkNCALL(SeqNum=
          3678376, OpType=SEND, NumelIn=3584, NumelOut=3584, Timeout(ms)=
          600000) ran for 600003 milliseconds before timing out.
        Fatal Python error: Aborted  (every PP stage)
        … “Connection closed by peer [172.28.46.x]” cascade …
        [2026-08-06T16:27:07+02:00] WARN: otela did not announce LEFT
    The `DuplicateTimeseries`/`OSError Directory not empty` lines that
    follow are red herrings (a sglang HTTP restart dying on the
    already-registered prometheus registry + the quota-`/users` tmpdir
    cleanup of fix 8). **Fix:** pass `--dist-timeout $DIST_TIMEOUT`
    (default 3600, see knob) so the NCCL floor matches the scheduler
    watchdog; a transient stall now rides out at 10 min and the
    `SOFT_WATCHDOG_TIMEOUT` dump is the first diagnostic, not a kill at
    10 min.

## Files

| File | Purpose |
|------|---------|
| `serve_kimi_k3_otela_clariden.sbatch` | One self-contained sbatch: distributed SGLang engine (enroot, TP4×PP8; `servekit launch --overlap` wrapping is opt-in via `PRESHARDED_ENABLE=1`, default 0 = direct Lustre load) + pre-registration servekit verification bench + one otela worker on the head + lifecycle trap. Defaults: 8 nodes, TP4×PP8. |
| `kimi-k3-clariden.toml` | EDF: image (local .sqsh), mounts, caches, `com.hooks.aws_ofi_nccl` (cuda13) for Slingshot NCCL. |
| `build_kimi_k3_image.sbatch` | One-time: import `docker://lmsysorg/sglang:kimi-k3` to the local arm64 .sqsh (on a `debug` node, with /dev/shm enroot scratch). |
| `bench.py` | Legacy vendored throughput harness (`--input-len` counts tokens here, unlike servekit's words). New default: `meta/bench/cbench.sh` + `cbench_report.py` (servekit bench); the 3000965 numbers still come from this script. |
| `bench_3000965.jsonl` | Raw verified benchmark output (job 3000965, 2026-08-04) backing the table above. |
| `hicache_bench.py` | HiCache functional + latency test (probe → HBM hit → evict → host-tier hit; prints T1/T2/T3 + `/metrics` evidence). Stdlib, runs from the login node. Uses `/v1/completions` — the chat template breaks prefix equality (see *HiCache*). |
| `README.md` | This file. |

## Submit

### From the login node (SSH)

```bash
# 0. one-time prep (debug partition, ~15-25 min for the ~21 GB image):
sbatch build_kimi_k3_image.sbatch
# wait for the .sqsh at /capstor/scratch/cscs/xyao/kimi-k3/images/sglang-kimi-k3.aarch64.sqsh

# 1. production default: Kimi-K3, 8 nodes, TP4×PP8
sbatch serve_kimi_k3_otela_clariden.sbatch

# 2. fewer nodes (e.g. 4), TP4×PP4:
sbatch --nodes=4 --export=ALL,NNODES=4,PP_SIZE=4 \
       serve_kimi_k3_otela_clariden.sbatch
```

### From your local machine via `rcc`

The repository ships a project-local `.rcc/config.toml` with a `clariden`
profile that syncs to `/capstor/scratch/cscs/xyao/opentela-cookbook` and
submits through the `clariden` SSH alias (configured in `~/.ssh/config`).

```bash
# one-time: sync local code and this recipe to Clariden
rcc --profile clariden push

# build the arm64 image (debug partition, ~15-25 min)
rcc --profile clariden job submit deployments/llm/clariden/kimi-k3/build_kimi_k3_image.sbatch

# serve Kimi-K3 (default 8 nodes, TP4×PP8)
rcc --profile clariden job submit deployments/llm/clariden/kimi-k3/serve_kimi_k3_otela_clariden.sbatch

# fewer nodes, e.g. 4 nodes TP4×PP4
rcc --profile clariden job submit --sbatch-args='--nodes=4 --export=ALL,NNODES=4,PP_SIZE=4' \
  deployments/llm/clariden/kimi-k3/serve_kimi_k3_otela_clariden.sbatch

# monitor
rcc --profile clariden job status <JOBID>
rcc --profile clariden job tail <JOBID> -f

# inspect logs from your local machine
rcc --profile clariden run -- tail -f /capstor/scratch/cscs/xyao/kimi-k3/logs/k3-clariden-<JOBID>.out
```

## Verify

The compute nodes' `:30000` is reachable on the head node's host netns. Health
checks can run from the allocation via `srun --overlap`:

```bash
JOB=<jobid>
source /capstor/scratch/cscs/xyao/kimi-k3/last_service.env   # sets SERVICE_HEAD_IP, SERVICE_PORT
# engine + otela log:
tail -f /capstor/scratch/cscs/xyao/kimi-k3/logs/k3-clariden-$JOB.out
# look for: "[rank 0] ... shape: tp=4 pp=8 ep=4", "Load weight end" (32 of them,
#   up to 29 min apart — the monkey-patch tolerates this), then the otela line
#   "engine healthy — starting otela worker on <head>" and
#   "protocol/registrar.go:145 Registering LLM service: … connected localhost
#   30000 …" (DEBUG — only visible because the cfg.yaml sets loglevel: debug).

# via rcc from your local machine:
rcc --profile clariden run -- bash -lc 'tail -f /capstor/scratch/cscs/xyao/kimi-k3/logs/k3-clariden-'"$JOB"'.out'

# health + model from the head, inside the container:
srun --jobid=$JOB --overlap --gres=none --nodes=1 -n1 -w "$SERVICE_HEAD_NODE" \
     --environment=kimi-k3-clariden \
     bash -lc 'curl -s http://localhost:'"$SERVICE_PORT"'/health; echo; curl -s http://localhost:'"$SERVICE_PORT"'/v1/models | python3 -m json.tool'

# via rcc from your local machine (run on the head inside the job):
rcc --profile clariden run -- bash -lc \
  'source /capstor/scratch/cscs/xyao/kimi-k3/last_service.env && \
   srun --jobid='"$JOB"' --overlap --gres=none --nodes=1 -n1 -w "$SERVICE_HEAD_NODE" \
        --environment=kimi-k3-clariden \
        bash -lc "curl -s http://localhost:$SERVICE_PORT/health; echo; curl -s http://localhost:$SERVICE_PORT/v1/models | python3 -m json.tool"'

# once the worker is registered, route a request through the OpenTela head:
curl -s http://<alps-head>/v1/service/llm/v1/chat/completions \
  -H "Content-Type: application/json" -H "X-Otela-Model: moonshotai/Kimi-K3" \
  -d '{"model":"moonshotai/Kimi-K3","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'

# decisive proof the mesh routes to us, without any API token:
grep -ac 'POST /v1/chat/completions.*200 OK' $DEPLOY_DIR/logs/k3-clariden-$JOB.out

# exactly one registration should exist on the API
# (https://api.swissai.svc.cscs.ch/v1/models lists all providers, no auth).
# If you see several entries for the same model, duplicate otela workers are
# health-checking the same engine — typically after a manual srun relaunch
# during debugging. Both will say `ready` because they proxy the same
# localhost backend. Fix: SIGTERM the extra `otela-arm64` process on the head
# (clean LEFT), do not SIGKILL, then re-check the listing.
```

### Benchmark

Standard after any change (protocol and rationale: `meta/bench/README.md`).
The default sweep from any login node, via the checkout on the shared FS:

```bash
SERVEKIT_DIR=$DEPLOY_DIR/servekit bash meta/bench/cbench.sh \
    "http://$SERVICE_HEAD_IP:$SERVICE_PORT" "1:8 4:16 8:32 16:48 32:64 64:96" \
    768 256 --label k3-clariden-pp8
```

(768 words ≈ the 1024-token prompt above.) From *inside* the allocation
instead (pure stdlib, no install — no-egress safe):

```bash
srun --jobid=$JOB --overlap --gres=none --nodes=1 -n1 -w "$SERVICE_HEAD_NODE" \
     --environment=kimi-k3-clariden \
     env PYTHONPATH="$DEPLOY_DIR/servekit/src" \
     python3 -m servekit.cli bench --url "http://127.0.0.1:$SERVICE_PORT" \
       --requests 64 --concurrency 16 --input-len 768 --output-len 256 \
       --out "$DEPLOY_DIR/bench_c16.json"
```

Already done for you on every launch: the sbatch benches C=16 n=64 **before**
registering on OpenTela (merged into `$RUNDIR/run.node0.json` when
`PRESHARDED_ENABLE=1`, or standalone `$RUNDIR/bench.json` with the default
`PRESHARDED_ENABLE=0`). The engine runs under `servekit launch` only when
`PRESHARDED_ENABLE=1`, so `$RUNDIR/run.node<RANK>.json` captures every node's
startup timeline (weight-load/compile/CG phases, `ready_wait_s`); with the
default 0 the startup timeline is in the job log instead.
Set `SERVEKIT_BENCH=0` to skip the automatic bench.

Historical: the legacy `bench.py` invocation that produced the verified table
is preserved in the git history of this file; its protocol is 1024 token-in,
not words.

Alternatively, sglang's own `bench_serving` (random-ids dataset, local
tokenizer path since `HF_HUB_OFFLINE=1`):

```bash
srun --jobid=$JOB --overlap --gres=none --nodes=1 -n1 -w "$SERVICE_HEAD_NODE" \
     --environment=kimi-k3-clariden \
     python3 -m sglang.bench_serving --backend sglang \
       --host "$SERVICE_HEAD_IP" --port "$SERVICE_PORT" \
       --dataset-name random-ids \
       --model /capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K3 \
       --tokenizer /capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K3 \
       --random-input-len 1024 --random-output-len 128 --random-range-ratio 1.0 \
       --num-prompts 64 --max-concurrency 64
```

### Operational notes from real runs

**Changing CTX_LEN requires a restart — use an overlapping swap for zero
downtime.** `context-length` is read once at engine start (sglang has no
runtime hot-change), and this recipe's batch shell runs the engine in the
foreground, so killing the engine step ends the whole job. The zero-downtime
pattern (used 2026-08-04 for the 64K → 1M switch): submit the new job while
the old one keeps serving — the API will intentionally show TWO providers
during the new job's cold start (that overlap *is* the handover mechanism) —
wait for the new engine's `/health`, then `scancel` the old job so its trap
gives otela a clean LEFT. `swap_to_1m.sh` (alongside the recipe on Clariden)
automates this. Change of record: job 3000965 (64K) → job 3002366 (1M) — **3002366
ran 8 h then died on a hard scheduler watchdog (fix 11); the next attempt
(3018155) carried `WATCHDOG_TIMEOUT=3600 SOFT_WATCHDOG_TIMEOUT=600` and still
died at 5 h20 m on the NCCL process-group watchdog at 600 s (fix 12), which
the scheduler raise never touched. The next attempt adds `--dist-timeout 3600`
so both watchdogs share the same floor.**

**Transient Lustre pauses during weight loading are normal.** The proven
run (2920471) paused 4 min 11 s at shard 2 and 2-2.5 min at shards 52-54; a
later run (2999818) paused 3 min at shard 2 and ~2 min at shard 10. These are
/capstor I/O stalls, not recipe bugs. A job that appears frozen at a shard
for <20 min is almost certainly still loading — check page-cache trend and
`/proc/<pid>/io` before killing. (Job 2999398 was cancelled after 37 min
under the false assumption it had hung; HiCache ON vs OFF was not the
cause.) HiCache is verified safe as an opt-in — see *HiCache* above.

**The 1M run (3002366) served 8 h, then a transient PP-broadcast stall tripped
the hard watchdog.** At 12:29:35 every PP stage's scheduler watchdog (default
300 s, `soft=False`) fired on a >300 s hang in the per-iteration PP
request-metadata broadcast (`broadcast_pyobj → _broadcast_reqs_across_ranks`),
crashing the healthy engine and leaving otela unable to announce LEFT. The
stall was transient-shaped (steady `200 OK` through 12:23:53, then a sudden
multi-minute broadcast hang) — not an OOM or KV exhaustion, and **1M context
length is not the cause** (1M booted and served fine for 8 h). The real
trigger is `TODO(unverified)`; the recipe now raises the hard watchdog to
3600 s and adds a 600 s soft watchdog that dumps a stack trace without killing
(see fix 11). If a future run stalls again, the soft-watchdog dump (the job
log / `$RUNDIR`) is the diagnostic — do **not** `scancel` a merely-stalled
engine before the hard limit unless that dump shows a true deadlock; a
premature scancel is exactly how 2999398 was lost.

**3018155 (the fix-11 attempt) still died at 5 h20 m — on the NCCL
process-group watchdog, not the sglang scheduler one.** It carried
`WATCHDOG_TIMEOUT=3600 SOFT_WATCHDOG_TIMEOUT=600` and served cleanly until a
PP `SEND` (`SeqNum=3678376`) stalled at 16:08:43; the sglang watchdog never
tripped (3600 s ≫ the stall), but PyTorch's `ProcessGroupNCCL::Watchdog`
fired at its 600 s default (`--dist-timeout`, which the recipe did not pass)
and aborted every rank — `WARN: otela did not announce LEFT` again. Fix 12
passes `--dist-timeout 3600` so both watchdogs share the same floor; the
soft-watchdog dump at 600 s is the first diagnostic on the next stall.

**Superseded numbers.** An earlier measurement (2026-07-28, run 2920471,
`sglang.bench_serving`, 1024-in / 128-out, cold CUDA graphs) reported
~12.6 tok/s at C=1 saturating at ~205–215 tok/s around C=64–256. Those
figures came from a different harness on a first-boot run and are superseded
by the **Verified benchmarks** table above (job 3000965), which matches the
JSC recipe's curve on identical hardware and protocol.

**Crash cores accumulate in `DEPLOY_DIR` — clean them after a successful
bring-up.** Linux's `core_pattern` writes each crashing process's dump to its
cwd (the engine runs in `DEPLOY_DIR`) as `core_<host>_<pid>`; the recipe
deliberately does **not** set `ulimit -c 0`, so every hard-crash drops 1-8
cores per node (one ~18-20 GB apparent per crashed rank, though they are
largely *sparse* — real block usage is ~28%). Over the Jul 28 → Aug 7 bring-up
this piled up to **71 stale cores, ~199 GB actual / 704 GiB apparent**, all
removed 2026-08-08 (verified none newer than the serving job, all PIDs gone):
the TP32-shape CUDA-graph crashes (job 2914910 family), the 1M-context runs,
and the PP-stall fixes 11/12 (3002366 hard watchdog @300 s, 3018155 NCCL
watchdog @600 s, and 3029640 whose Gloo `Connection closed by peer` first
surfaced the real control-plane cause — that one's four newest cores at
Aug 7 20:51 were the last removed). **The cores were diagnostically
redundant:** every PP-stall root-cause came from the *live* log tracebacks +
the fix-11 soft-watchdog dump, never the binary cores. They're kept enabled
only so a *novel* future failure (not the known PP-stall family) can still
yield one; if scratch pressure recurs, add `ulimit -c 0` to the engine srun
step (or point `/proc/sys/kernel/core_pattern` at `|/bin/false` site-wide).

## Knobs (env, all overridable)

| Knob | Default | Notes |
|---|---|---|
| `DEPLOY_DIR` | `/capstor/scratch/cscs/xyao/kimi-k3` | reuses the imported image + warm caches |
| `IMAGE` | `$DEPLOY_DIR/images/sglang-kimi-k3.aarch64.sqsh` | built by `build_kimi_k3_image.sbatch` |
| `MODEL_PATH` | `/capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K3` | |
| `SERVED_MODEL_NAME` | `moonshotai/Kimi-K3` | |
| `OTELA_EDF_NAME` | `kimi-k3-clariden` | EDF in this dir; set `kimi-k3` to use `~/.edf/kimi-k3.toml` |
| `TP_SIZE` / `PP_SIZE` / `EP_SIZE` | `4` / `$NNODES` / `4` | EP×moe_dp == TP. PP=node. |
| `CG_DECODE` | `full` | set `disabled` for the flat TP32 shape |
| `MOE_BACKEND` | `marlin` | only SM90 mxfp4-packed runner |
| `CTX_LEN` | `1048576` | bounds request length, not the KV pool. 1M = model's `max_position_embeddings`; pool holds 3.73M tokens @ 0.90 (hybrid attn: tiny KV/token), so 1M needs no extra memory. Startup-only — cannot be hot-changed; plan a restart (see Operational notes) |
| `MEM_FRAC` | `0.90` | |
| `K3_NIC` | `hsn0` | Slingshot HSN NIC (bootstrap path) |
| `DIST_PORT` | `20000` | torchrun distributed store (rank discovery) |
| `SERVE_PORT` | `30000` | sglang HTTP port (bound on 0.0.0.0 on rank 0) |
| `DIST_TIMEOUT` | `3600` | PyTorch distributed init + NCCL collective timeout (s), passed as `--dist-timeout`. Was 600 and NOT passed; the NCCL `ProcessGroupNCCL` watchdog at its 600 s default killed job 3018155 on a transient PP `SEND` stall after 5 h20 m of healthy serving (fix 12). Raised to 3600 to match `WATCHDOG_TIMEOUT` so the `SOFT_WATCHDOG_TIMEOUT` dump at 600 s is the first signal. `SGLANG_EXTRA_ARGS="--dist-timeout N"` overrides for a shorter collective failure. |
| `HICACHE_ENABLE` | `0` | OFF by default. Verified safe opt-in 2026-08-05 — see *HiCache* above. Offloads evicted prefixes to Grace LPDDR. |
| `HICACHE_RATIO` / `HICACHE_WRITE_POLICY` / `HICACHE_IO_BACKEND` | `1.0` / `write_back` / `kernel` | only used when `HICACHE_ENABLE=1`. Ratio 1.0 is the safe setting (~59 GB/node); ratio 2.0 ~doubles the pool but needs ~118 GB/node. `write_back` is the ONLY working policy (`write_through_selective` never backs up). | |
| `OTELA_BIN` | `/capstor/scratch/cscs/xyao/opentela/otela-arm64` | arm64; x86 gives Exec format error |
| `OTELA_RELAY_ADDR` | `/ip4/140.238.223.116/tcp/43905/p2p/Qm…` | Public OpenTela bootstrap (direct, no relay) |
| `OTELA_SERVICE_NAME` / `OTELA_TCP_PORT` / `OTELA_UDP_PORT` | `llm` / `43905` / `59820` | one otela per node at a time |
| `OTELA_SEED` / `OTELA_API_PORT` | `21` / `18094` | libp2p identity seed; otela HTTP API port |
| `HEALTH_TIMEOUT` | `9000` | 150 min — cold start is ~105 min (job 3000965); ~1.5 TB weights, be patient |
| `UNBALANCED_MODEL_LOADING_TIMEOUT_S` | `3600` | sglang weight-load barrier (fix 10); patched via `$DEPLOY_DIR/patches/sitecustomize.py` |
| `WATCHDOG_TIMEOUT` | `3600` | sglang HARD scheduler watchdog (s) — raises & kills the engine on a forward batch longer than this. Default 300 killed job 3002366 after 8 h of healthy serving on a >300 s PP request-broadcast stall; raised to 3600 (fix 11) so transient distributed stalls ride out, a true permanent hang still dies in 1 h |
| `SOFT_WATCHDOG_TIMEOUT` | `600` | sglang SOFT watchdog (s) — dumps a stack trace at 600 s WITHOUT crashing (fix 11). Keep < `WATCHDOG_TIMEOUT` |
| `SGLANG_EXTRA_ARGS` | *(empty)* | appended to the `python -m sglang.launch_server` command (array; word-split) |
| `SERVEKIT_DIR` | `$DEPLOY_DIR/servekit` | servekit checkout (**BRANCH multinode-pp**, stdlib-only, runs via `PYTHONPATH` — no install). Stage once with egress: `git clone --depth=1 -b multinode-pp https://github.com/eth-easl/servekit $DEPLOY_DIR/servekit`. Only used when `PRESHARDED_ENABLE=1` (wraps the engine in `servekit launch --overlap` for presharded staging + per-node cold-start JSON); default 0 runs the engine directly from `$MODEL_PATH` with no servekit wrapping. Missing checkout + `PRESHARDED_ENABLE=1` → WARN, cold load, auto-bench skipped. |
| `PRESHARDED_ROOT_BASE` | `/capstor/store/cscs/swissai/infra01/cold-start-experiments/kimi-k3-presharded` | root of the offline pre-sharded dump; `-tp${TP}pp${PP}` is appended (→ `kimi-k3-presharded-tp4pp8/TP-4-sig-<hash>/`). Only used when `PRESHARDED_ENABLE=1`: each PP stage stages its OWN file set (read from the dump's `checksum.json`) to `/dev/shm` via `servekit launch --overlap`. A config mismatch is a cache MISS (full re-load + re-dump), not a silent wrong-shape serve. |
| `PRESHARDED_ENABLE` | `0` | **0 (default): the engine runs directly from `$MODEL_PATH` on Lustre — the proven path (jobs 3000965/3002366/3018155/3035026 each served ~970 requests before the recurring PP-pipeline stall). No servekit wrapping, no preshard, no per-node cold-start profile.** `1`: servekit `launch --overlap` stages each rank's pre-sharded slice from `$PRESHARDED_ROOT` to `/dev/shm` and wraps the engine for lifecycle + profiling. **Currently blocked:** the dump's `.safetensors` carry ACL `mask::---` (owner `yboughizane` only; group infra01/csstaff/infra01adm all effective `---`), so servekit's `dd` staging fails with `Permission denied` and the engine's cache-MISS re-dump crashes (`FileNotFoundError` in `_tmp_presharding`). Fix: `yboughizane` runs `setfacl -R -m m::r $PRESHARDED_ROOT` (or `setfacl -R -m u:xyao:r …`). |
| `SERVEKIT_BENCH` | `1` | `0` skips the post-health, pre-registration verification bench |
| `SERVEKIT_BENCH_REQUESTS` / `SERVEKIT_BENCH_CONCURRENCY` | `64` / `16` | single-level verification bench size (pp=8 ↔ max_running_requests=32; keep C≤32) |
| `SERVEKIT_BENCH_CORRECTNESS` | `1` | attach the greedy correctness probe (non-gating) to the verification bench |

## Cluster facts (the things you can't rediscover from a manual)

- **aarch64** (Grace); 4× GH200 120 GB / node (97871 MiB HBM each), 288 logical
  CPUs, ~870 GB RAM (`sinfo`/`nvidia-smi` verified).
- **Slingshot fabric, NO InfiniBand**: NICs `hsn0`..`hsn3` (172.28.*/16),
  `/opt/cray/libfabric` (CXI provider), `/opt/slingshot/firmware/cassini`.
  NCCL uses the `aws-ofi-ccl-plugin` (CSCS), injected by the EDF's
  `com.hooks.aws_ofi_nccl` hook; variants under
  `/opt/cscs/aws-ofi-ccl-plugin/{cuda12,cuda13}/libnccl-net.so` MUST match the
  image's CUDA major (this image is CUDA 13).
- **enroot + Pyxis/EDF** (`srun --environment=<name>`), NOT Apptainer (unlike
  JSC). Containers share the host network namespace (verified), so
  `localhost:$SERVE_PORT` from a sibling `srun --overlap` step reaches sglang.
- **Direct OpenTela egress**: compute reaches the public bootstrap
  `/ip4/140.238.223.116/tcp/43905/p2p/Qm…` directly — no login-node relay.
- **Diskless compute nodes**: only `/dev/shm` is node-local; enroot scratch
  must be redirected there for image import (see `build_kimi_k3_image.sbatch`).
- **Account `infra02`**, partition `normal` (12 h) for serving, `debug`
  (1:30) for the image import. Caches and writable state on `/capstor`
  (NOT quota-limited `/users`).
- **EP for K3 is Blackwell-gated on the a2a side** (same as JSC): the
  `MOE_A2A_BACKEND` knob is intentionally not wired here; on Hopper K3-mxfp4
  can only run on Marlin, which has no optimized cross-node a2a path. TP4×PP8
  is the Hopper throughput ceiling on this fabric.
