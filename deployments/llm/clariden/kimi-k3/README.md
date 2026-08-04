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

Taken with `bench.py` (vendored in this directory — the same script and
protocol as `meta/bench/` and the JSC recipe): warmup `4:8` discarded, then
the sweep below, 256 max output tokens, `ignore_eos`, deterministic. Run
inside the container on the head (nid007464) against `127.0.0.1:30000` of job
3000965 (2026-08-04); HiCache OFF. Raw output: `bench_3000965.jsonl` (in this
directory).

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
   IB transport is therefore unused. **Fix:** the EDF
   (`kimi-k3-clariden.toml`) carries the annotation
   `com.hooks.aws_ofi_nccl.enabled=true` / `variant=cuda13` — the CSCS
   Slurm/enroot hook that, at container start, `LD_PRELOAD`s
   `/opt/cscs/aws-ofi-ccl-plugin/cuda13/libnccl-net.so` and wires libfabric's
   CXI provider. The engine.sh adds the verified `NCCL_NET="AWS Libfabric"`,
   `NCCL_CROSS_NIC=1`, `FI_CXI_DISABLE_HOST_REGISTER=1`,
   `FI_CXI_DEFAULT_CQ_SIZE=131072`, `FI_CXI_RDZV_THRESHOLD=0`,
   `FI_CXI_RDZV_GET_MIN=0`, `FI_MR_CACHE_MONITOR=userfaultfd`. `NCCL_SOCKET_IFNAME=hsn0`
   is for the TCP bootstrap path only (the ofi plugin carries data).
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

## Files

| File | Purpose |
|------|---------|
| `serve_kimi_k3_otela_clariden.sbatch` | One self-contained sbatch: distributed SGLang engine (enroot, TP4×PP8) + one otela worker on the head + lifecycle trap. Defaults: 8 nodes, TP4×PP8. |
| `kimi-k3-clariden.toml` | EDF: image (local .sqsh), mounts, caches, `com.hooks.aws_ofi_nccl` (cuda13) for Slingshot NCCL. |
| `build_kimi_k3_image.sbatch` | One-time: import `docker://lmsysorg/sglang:kimi-k3` to the local arm64 .sqsh (on a `debug` node, with /dev/shm enroot scratch). |
| `bench.py` | Vendored copy of the shared `meta/bench/` throughput-vs-concurrency harness (aiohttp only, no tokenizer — works with `HF_HUB_OFFLINE=1`). |
| `bench_3000965.jsonl` | Raw verified benchmark output (job 3000965, 2026-08-04) backing the table above. |
| `README.md` | This file. |

## Submit

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

# health + model from the head, inside the container:
srun --jobid=$JOB --overlap --gres=none --nodes=1 -n1 -w "$SERVICE_HEAD_NODE" \
     --environment=kimi-k3-clariden \
     bash -lc 'curl -s http://localhost:'"$SERVICE_PORT"'/health; echo; curl -s http://localhost:'"$SERVICE_PORT"'/v1/models | python3 -m json.tool'

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

Benchmark from inside the allocation (`bench.py` ships in this recipe
directory — copy it alongside the sbatch to `$DEPLOY_DIR/recipe/` on Clariden;
it needs no tokenizer/downloads, only `aiohttp`, which the image has):

```bash
srun --jobid=$JOB --overlap --gres=none --nodes=1 -n1 -w "$SERVICE_HEAD_NODE" \
     --environment=kimi-k3-clariden \
     python3 /capstor/scratch/cscs/xyao/kimi-k3/recipe/bench.py \
       "1:8 4:16 8:32 16:48 32:64 64:96" 127.0.0.1 "$SERVICE_PORT" 1024 256
```

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
automates this. Change of record: job 3000965 (64K) → job 3002366 (1M).

**Transient Lustre pauses during weight loading are normal.** The proven
run (2920471) paused 4 min 11 s at shard 2 and 2-2.5 min at shards 52-54; a
later run (2999818) paused 3 min at shard 2 and ~2 min at shard 10. These are
/capstor I/O stalls, not recipe bugs. A job that appears frozen at a shard
for <20 min is almost certainly still loading — check page-cache trend and
`/proc/<pid>/io` before killing. (Job 2999398 was cancelled after 37 min
under the false assumption it had hung; HiCache ON vs OFF was not the cause.)
HiCache is OFF by default (matches the proven run); re-enabling is untested.

**Superseded numbers.** An earlier measurement (2026-07-28, run 2920471,
`sglang.bench_serving`, 1024-in / 128-out, cold CUDA graphs) reported
~12.6 tok/s at C=1 saturating at ~205–215 tok/s around C=64–256. Those
figures came from a different harness on a first-boot run and are superseded
by the **Verified benchmarks** table above (job 3000965), which matches the
JSC recipe's curve on identical hardware and protocol.

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
| `DIST_TIMEOUT` | `600` | PyTorch distributed init timeout (s) |
| `HICACHE_ENABLE` | `0` | OFF by default (matches proven run 2920471). ON is untested; see *Operational notes* on transient Lustre pauses. Offload to Grace LPDDR when enabled. |
| `HICACHE_RATIO` / `HICACHE_WRITE_POLICY` / `HICACHE_IO_BACKEND` | `2.0` / `write_through_selective` / `kernel` | only used when `HICACHE_ENABLE=1` | |
| `OTELA_BIN` | `/capstor/scratch/cscs/xyao/opentela/otela-arm64` | arm64; x86 gives Exec format error |
| `OTELA_RELAY_ADDR` | `/ip4/148.187.108.178/tcp/43905/p2p/Qm…` | Alps OpenTela bootstrap (direct, no relay) |
| `OTELA_SERVICE_NAME` / `OTELA_TCP_PORT` / `OTELA_UDP_PORT` | `llm` / `43905` / `59820` | one otela per node at a time |
| `OTELA_SEED` / `OTELA_API_PORT` | `21` / `18094` | libp2p identity seed; otela HTTP API port |
| `HEALTH_TIMEOUT` | `9000` | 150 min — cold start is ~105 min (job 3000965); ~1.5 TB weights, be patient |
| `UNBALANCED_MODEL_LOADING_TIMEOUT_S` | `3600` | sglang weight-load barrier (fix 10); patched via `$DEPLOY_DIR/patches/sitecustomize.py` |
| `SGLANG_EXTRA_ARGS` | *(empty)* | appended to the `sglang serve` line |

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
- **Direct OpenTela egress**: compute reaches the Alps bootstrap
  `/ip4/148.187.108.178/tcp/43905/p2p/Qm…` directly — no login-node relay.
- **Diskless compute nodes**: only `/dev/shm` is node-local; enroot scratch
  must be redirected there for image import (see `build_kimi_k3_image.sbatch`).
- **Account `infra02`**, partition `normal` (12 h) for serving, `debug`
  (1:30) for the image import. Caches and writable state on `/capstor`
  (NOT quota-limited `/users`).
- **EP for K3 is Blackwell-gated on the a2a side** (same as JSC): the
  `MOE_A2A_BACKEND` knob is intentionally not wired here; on Hopper K3-mxfp4
  can only run on Marlin, which has no optimized cross-node a2a path. TP4×PP8
  is the Hopper throughput ceiling on this fabric.
