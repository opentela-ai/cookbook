# Kimi-K3 on JSC Jupiter (SGLang, GH200) → OpenTela

Serve `moonshotai/Kimi-K3` (2.8 T-parameter hybrid MoE, mxfp4) on **JSC Jupiter
Booster** (4× GH200 / node, aarch64, InfiniBand + SHARP, Slurm, Apptainer) and
register it on OpenTela.

This README records **what we verified on this cluster** — the topology that
wins, the ones that don't and why, and the exact failures behind each choice.
The companion `serve_llm_otela_jsc.sbatch` is the runnable artifact; the
findings here are the "why" behind its non-obvious settings.

## Start here

On a login node (`cd deployments/llm/jsc/kimi-k3` first; from a workstation,
prefix each command with `rcc --profile jsc run --`):

```bash
bash build_kimi_k3_image.sh          # once: .sif on /e/scratch (10–25 min)
bash start_relay_jsc.sh              # once: relay in tmux, writes relay.multiaddr
sbatch serve_llm_otela_jsc.sbatch    # per job: sglang on 8 nodes + otela worker on the head
# or all three in one shot (skips what already exists):  bash up.sh
```

```bash
# verify (login node CANNOT reach compute — run inside the allocation)
JOB=<jobid>; D=/e/scratch/reformo/$USER/otela-llm
HEAD=$(grep SERVICE_HEAD_NODE $D/last_service.env | cut -d= -f2)
srun --jobid=$JOB --overlap --nodes=1 -w "$HEAD" curl -s http://127.0.0.1:30000/health
grep -iE "regist|LEFT|error" $D/run-$JOB/otela.log   # registration + clean leave
tmux attach -t jsc-otela-relay-$(id -un)             # relay log

# end-to-end through the public API:
curl -s https://api.opentela.ai/v1/service/llm/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"moonshotai/Kimi-K3","messages":[{"role":"user","content":"ping"}]}'
```

Everything below is the verified findings behind those settings — read it when
a run misbehaves or before changing a default, not before the first submit.

---

## TL;DR — the verified operating point

| | Value | Notes |
|---|---|---|
| **Production topology** | **TP4 × PP8** (8 nodes) | The Hopper ceiling. Stable, fastest. |
| **Throughput @ C=32** | **542 tok/s** aggregate | Measured (1024-in / 256-out). |
| **Scaling** | near-linear C1→C32 (28.8 → 542) | **Pipeline-bound**, not capacity-bound. |
| **Free HBM per GPU** | ~30 GiB (of 96) | At `mem_fraction_static=0.90`. |
| **Concurrency ceiling** | `max_running_requests=136` | The KDA state pool; not the bottleneck at C=32. |

**TP4/PP8 is as fast as Kimi-K3 gets on GH200.** Cross-node TP32/EP32 boots and
is stable (the SHARP hang is solved), but it is **38× slower per request** and
**cannot be sped up** on Hopper — see §2 and §3.

---

## §1. What the production run uses (and why)

A boot log of the TP4/PP8 default (`serve_llm_otela_jsc.sbatch`, 8 nodes) shows:
`max_total_num_tokens ≈ 3.72 M`, `max_mamba_cache_size = 136`, Marlin mxfp4
runner, FlashMLA attention, FlashKDA linear-attention decode, HiCache on, CUDA
graph capture `full`. ~30 GiB HBM idle per GPU.

Why each of these is pinned the way it is (each was a discovered failure mode):

- **`--moe-runner-backend marlin`** — Marlin is the *only* SM90 MoE backend that
  keeps K3's mxfp4 weights **packed** (W4A16). The `auto` default would pick
  Triton-Kernels, whose `upcast_from_mxfp4()` dequantizes to bf16 and OOMs a
  4-bit model on 96 GiB.
- **TP4** — keeps tensor-parallel collectives inside one node's **4-GH200 NVLink
  domain**. No cross-node all-reduce. Cross-node traffic is pipeline send/recv
  only (point-to-point, always fine).
- **PP8** — one pipeline stage per node. 8 serial stages is the throughput
  ceiling (we are pipeline-bound), but it is *stable* and avoids the cross-node
  collective problems that TP32 hits (§2).
- **`--mem-fraction-static 0.90`** — packs the KV + KDA pools; leaves ~30 GiB
  headroom for Marlin workspace + CUDA-graph capture.
- **FlashKDA / FlashMLA / HiCache / Marlin** — the SGLang Kimi-K3 cookbook's
  Hopper capacity levers. All already applied; most other capacity knobs would
  raise the 136-slot KDA ceiling, which is *not* the binding constraint at C=32.
  FlashKDA is **auto-enabled** by the sbatch when a built prefix exists at
  `$DEPLOY_DIR/flashkda_prefix` (build it with `build_flashkda_prefix.sh`):
  measured **+14% / +19% aggregate at C=8 / C=16** (prefill-bound), flat at
  C=32 (pipeline-bound). No-op if absent.

### Verified benchmarks (TP4/PP8, 8 nodes, 1024-in / 256-out)

Numbers below predate the harness switch: they were taken with the legacy
token-counting `bench.py`. **The harness is now `meta/bench/cbench.sh`
(servekit bench)** — the sbatch runs a C=16 n=64 verification level
automatically after health, before OpenTela registration, and the engine
runs under `servekit profile` (per-node `$RUNDIR/coldstart.node<RANK>.json`).
See [`meta/bench/README.md`](../../../meta/bench/README.md) for the full protocol (warmup, the
C=1 trap, how to read the curve shape); servekit `--input-len` counts words
(~1.4 tok/word), so 768 words ≈ the 1024-token protocol below.

| Concurrency | Aggregate out tok/s | per-req tok/s |
|---|---|---|
| 1  | 28.8  | 28.8 |
| 4  | 93.9  | 23.5 |
| 8  | 182.6 | 22.8 |
| 16 | 333.0 | 20.8 |
| 32 | 542.9 | 17.0 |

Linear-ish to C=32; the *per-request* rate only slowly degrades, so the limit is
how fast the 8-stage pipeline drains, not GPU memory. **The untested lever is
the high-C regime (C=48/64/96/128)** to find where the curve flattens — needs a
fresh allocation.

### The single biggest real-world lever: `reasoning_effort`

K3 always thinks; depth is `reasoning_effort` (low / high / **max** default).
`max` emits long chains-of-thought that dominate decode time. For interactive
serving, **`reasoning_effort=low`/`high` is a far bigger throughput win than any
topology change** — it cuts output tokens at the source. Recommended as an
app-level default.

---

## §2. Cross-node TP32/EP32: the SHARP story (stability — SOLVED)

The SGLang Kimi-K3 cookbook publishes a 4×8 / TP32 / EP32 / PP1 "High-Throughput"
shape. On JSC's Booster fabric, **cross-node NCCL ALLGATHER_BASE hangs after
~800–2000 ops without the SHARP plugin** (reproduced at 4–8 nodes). Getting
SHARP loaded inside the container was a 5-part puzzle — **all solved and
verified**:

1. **ABI** — the JSC SHARP plugin (`nccl_rdma_sharp`) links `libcudart.so.12`,
   absent in the default CUDA-13 sglang image. **Fix:** build the **cu12**
   image (`sglang-kimi-k3-cu12.sif`) — see `build_kimi_k3_image.sh`.
   *Note: `module load CUDA/13` on the host does NOT help — host modules are
   invisible inside apptainer.*
2. **`/opt/mellanox` absent on compute nodes** — SHARP runtime libs live only on
   login nodes. **Fix:** `stage_sharp_plugin.sh` copies the whole dependency
   chain to `/e/scratch` (self-contained).
3. **Two SONAME gaps** — only `libsmx-3.13.so` / `libsharprdmacm-3.13.so` ship;
   the plugin asks for the `-3.10` SONAMEs → symlinks.
4. **`libmlx5` symbol gap** — the cu12 container ships `MLX5_1.24`; `libsharp_coll`
   needs `MLX5_1.25`. **Fix:** inject the host `/lib64` rdma-core stack
   (internally consistent version set).
5. **Apptainer env propagation** — `srun --export=LD_LIBRARY_PATH=...` is
   **silently discarded** by apptainer; the plugin then fails to load and NCCL
   silently falls back to built-in IB. **Fix:** inject via
   `apptainer --env LD_LIBRARY_PATH=$SHARP_PLUGIN_DIR`.

**Verified end-to-end at 8 nodes / 32 GH200** (cu12 image + staged plugin):
- `Loaded collnet plugin SHARP`, **32 `/SHARP` device refs** (4 HCAs × 8 ranks),
  `8 collnet channels`/rank.
- NCCL TP32/EP32 init across 32 ranks completes in **~7–9 s** (no hang).
- Transport: **GPUDirect RDMA over IB** (`NET/NCCL RDMA Plugin/.../GDRDMA`),
  intra-node `P2P/IPC` (NVLink). **Zero** Socket/TCP fallback, **zero** NCCL WARN.
- **The documented hang is GONE** — 0 watchdog timeouts under load.

This is all **opt-in** in the recipe via `SHARP_PLUGIN_DIR=...`. The default
(TP4/PP8) leaves SHARP off; cross-node collectives never happen, so it's
unneeded. The whole SHARP injection path is reproduced by
`stage_sharp_plugin.sh` (run once on a login node).

---

## §3. Cross-node TP32/EP32: the throughput story (Blackwell-gated)

With SHARP the network is solved, so TP32/EP32/PP1 **boots and is stable**. But
its **throughput is catastrophic** unless an optimized MoE all-to-all backend is
engaged — and on Hopper that backend crashes. This is the single most important
finding from this bring-up.

### Measured: TP32/EP32 is ~38× slower per request

Running the official community TP32/EP32 config (TP32/EP32/PP1, Marlin,
`moe_a2a_backend='none'`, no DSPARK) at C=1:

| | TP4/PP8 (C=1) | TP32/EP32 (C=1) |
|---|---|---|
| throughput | 28.8 tok/s | **0.76 tok/s** |
| per-token latency | ~35 ms | **~1,300 ms** |
| ratio | — | **38× slower** |

**Root cause:** with `--moe-a2a-backend none` (the default), every MoE layer
(896 experts, top-16) does a **naive NCCL allgather across 32 GPUs on 8 nodes
per decode step**. The 1.3-second-per-token latency is almost entirely that
cross-node allgather. Even with perfect scaling to the `max_running_requests=52`
ceiling (set by the official `--mamba-full-memory-ratio 0.21`), aggregate
throughput would max at **~40 tok/s** — vs TP4/PP8's **542 tok/s**.

> **The official config implies an a2a backend the pasted command omits.** The
> cookbook's published TP32/EP32 command pairs it with
> `--moe-a2a-backend megamoe --moe-runner-backend deep_gemm` — the optimized MoE
> all-to-all that makes EP viable. Community copies often strip those flags.
> **Without an optimized a2a backend, EP32 is a latency trap.**

### The a2a backends all crash on K3-mxfp4+SiTU at sm90

DeepEP, MegaMoE, and Mooncake are all already **baked into the cu12 image**
(`deep_ep` 1.1.0 pip-installed with native sm_90 cubins + K3 patches; `deep_gemm`
0.1.4.post1). Nothing to build. They **engage cleanly** — and then crash:

| Attempt (`--moe-a2a-backend` / `--moe-runner-backend`) | Result |
|---|---|
| `none` / `marlin` (community default) | boots, **0.76 tok/s** (the §3 number) |
| `deepep` / `marlin` | `NotImplementedError: MARLIN requires a fused func for deepep, but none is registered` |
| `deepep` / `deep_gemm` | `RuntimeError: layout.hpp:60: Unknown SF transformation` on first MoE forward |

**Why — traced to source.** Kimi-K3's mxfp4 experts use a **custom
scale-factor format ("SiTU")**. The *only* DeepGEMM kernel that understands it
is `sm100_fp8_fp4_mega_moe.cuh` (patched by sglang's `apply_deepgemm_situ_patch.py`,
sentinel `activation_clamp == 0.03125` → SiTU with β=4.0, linear_β=25.0). That
kernel is **both Blackwell-only (`sm100`) and `mega_moe`-only**. DeepGEMM ships
**no `sm90_fp4_*_mega_moe.cuh`** at all — only `sm90_fp8_mega_moe.cuh` (fp8,
wrong dtype). So on Hopper K3-mxfp4 can *only* run on Marlin, and Marlin has no
DeepEP path. Every a2a-capable runner either needs Blackwell or can't read SiTU.

**Net: the network blocker (hang) is solved by SHARP, but the throughput
blocker (EP a2a) is a quantization-kernel gap that needs Blackwell.** EP for
Kimi-K3 cannot be enabled on GH200 regardless of image or network work. Filed
upstream as [ResearchComputer/xkernels#105](https://github.com/ResearchComputer/xkernels/issues/105).

### Where TP32/EP *would* work

On **B200 / GB200 / GB300** (sm100), `--moe-a2a-backend deepep` (or `megamoe`)
+ the SiTU kernel should Just Work. The recipe keeps the `MOE_A2A_BACKEND`
and rank-local `EP_JIT_CACHE_DIR` knobs for exactly that future run.

---

## §4. The cold-JIT-cache race (affects both cu12 and any EP run)

Two unrelated-but-similar bugs found en route, both fixed in the recipe:

1. **`tvm-ffi` JIT cache race** — sglang JIT-compiles several kernels (Marlin,
   tiny_gemm, KDA decode, …) into `TVM_FFI_CACHE_DIR`. On a cold cache, **32
   ranks race** into one shared dir → one rank reads a `.so` another is still
   writing → `"file too short"` / `"Module has no function 'run'"` mid
   CUDA-graph capture. The cu13 production run only survived because its cache
   was already warm. **Fix:** `TVM_FFI_CACHE_DIR` is **rank-local**
   (`.../rank-${SLURM_PROCID}`), with rank-0 pre-warmed single-process.
2. **cu12/cu13 cache ABI collision** — the JIT cache key omits the CUDA major
   version, so cu13-built `.so`s fail under cu12 (`libcudart.so.13 not found`).
   **Fix:** per-image cache dirs (`cache-cu12/tvm-ffi` etc.).

DeepEP V2 JITs its dispatch/combine kernels the same way (`EP_JIT_CACHE_DIR`,
default `~/.deep_ep`) — the recipe makes that rank-local too for the same reason.

---

## §5. Files

| File | Purpose |
|------|---------|
| `up.sh` | One-shot: build image if missing → start the relay (idempotent) → `sbatch`. The three Start-here steps as a single command; extra args are passed to `sbatch`. |
| `serve_llm_otela_jsc.sbatch` | One self-contained sbatch: sglang engine (apptainer `--nv`) + servekit cold-start profile + pre-registration bench + otela worker. Defaults: Kimi-K3, 8 nodes, TP4×PP8. |
| `start_relay_jsc.sh` | Login-node relay (tmux, writes `relay.multiaddr`). Run once before submitting; outlives every job. Per-user ports by default + `attach`/`stop` guards, so multiple operators coexist on one login node. See §6. |
| `build_kimi_k3_image.sh` | Build the sglang `.sif` on a login node. CUDA-13 (default) and CUDA-12 (`-cu12`, required for SHARP) variants. |
| `stage_sharp_plugin.sh` | One-command: stage a self-contained SHARP plugin dir on `/e/scratch` (closes all 5 gaps in §2). Run once on a login node. |
| `build_flashkda_prefix.sh` | Optional: build the FlashKDA Python prefix to bind-mount into the container. |

## §6. The login-node relay (start it too)

Compute nodes have no outbound internet, so the serving sbatch's `otela`
worker (which sits next to sglang on the compute head, `reachability:
private`) cannot dial the public head directly — it hops through a relay on a
login node. The sbatch has always **dialed** that relay via
`$OTELA_RELAY_ADDR` / `$RELAY_ADDR_FILE`, falling back to a baked-in
`OTELA_RELAY_DEFAULT`. Until now there was **no in-repo script to start it**;
`start_relay_jsc.sh` adds that.

```bash
bash deployments/llm/jsc/kimi-k3/start_relay_jsc.sh           # start (idempotent: reuses if up; per-user ports by default)
bash deployments/llm/jsc/kimi-k3/start_relay_jsc.sh attach    # print a RUNNING relay's multiaddr (no new start)
bash deployments/llm/jsc/kimi-k3/start_relay_jsc.sh status    # is it running?
bash deployments/llm/jsc/kimi-k3/start_relay_jsc.sh multiaddr # print the saved relay.multiaddr
bash deployments/llm/jsc/kimi-k3/start_relay_jsc.sh stop      # clean AnnounceLeave (refuses if another uid owns the port)
```

The relay is **us-managed and long-lived** (tmux session `jsc-otela-relay-<user>`):
it outlives every Slurm job and is shared by all of them. Start it **once**
before submitting; the sbatch reads `$DEPLOY_DIR/relay.multiaddr`
(`/e/scratch/$PROJECT/$USER/otela-relay/relay.multiaddr` by default)
automatically, so no `OTELA_RELAY_ADDR` export is needed for the run that
follows.

Two JSC-specific points the script handles (both are documented failures
in the serving sbatch header, ported here so the relay doesn't trip them):

- **BadgerDB off NFS.** otela resolves the home directory via `os/user.Current()`
  (`/etc/passwd`), **not** the `$HOME` env var — so unlike the Euler relay,
  `env HOME=...` on the command line does nothing here. On JSC the `/etc/passwd`
  home already ends in `/jupiter` (e.g. `/e/home/jusers/<user>/jupiter`), so
  the store lands at `$PASSWD_HOME/.ocfcore` on NFS-backed `/e/home`, goes
  `ESTALE` after a few days, the CRDT DAG freezes, and `api.opentela.ai`
  returns `503 No provider found`. Verified live with v0.2.4: `--config-dir`
  does NOT relocate the BadgerDB either (it still lands at
  `$PASSWD_HOME/.ocfcore`), so the symlink is still required.
  `start_relay_jsc.sh` **symlinks** `$PASSWD_HOME/.ocfcore` →
  `$DEPLOY_DIR/ocfcore` on `/e/scratch` before starting otela (and leaves an
  existing off-NFS symlink alone). The relay's peer identity comes from
  `--config-dir` (also on `/e/scratch`), so the symlink does not affect it.
- **Native binary, not the SIF.** The promoted image is an aarch64
  **sglang-only** image; the relay is a long-lived cluster-wide process and
  must not carry the engine's multi-GB working set. Stage the official
  `opentela-arm64` **v0.2.4** release asset once (the script fetches it to
  `$DEPLOY_DIR/bin/opentela` if missing — login nodes have egress). v0.2.4 is
  pinned because it is the first **official** build with both AutoRelay and
  the libp2p self-dial fix (`7f421838c5`, "dial the target peer, not the
  incoming Host header" — merged before v0.2.3); the older two-hop recipe
  here pinned v0.2.2 + a local `relayfix` patch because the fix wasn't
  released yet. NB the v0.2.x asset is named `opentela-arm64`, not
  `otela-arm64` (≤ v0.1.12) or `ocf-arm64` (≤ v0.1.11).

- **Signed binaries only, end to end.** Peers silently DISCARD every CRDT
  record published by a binary without a build attestation (self-built
  binaries report version `dev`): the node connects, relays HTTP and serves
  traffic — but never registers and never announces LEFT. The failure is
  silent and one-directional. `OTELA_REQUIRE_SIGNED=true` on the worker is
  only safe when every node in the path (relay included) runs an official
  signed release; the recipe defaults it to `false` to stay operable with
  self-built worker binaries.

### Concurrency on one login node

Two operators on the **same** login node used to both try to bind the fixed
relay ports; the second `otela start` died with `EADDRINUSE`, never
published a peer ID, and its sbatch silently latched onto the first user's
relay — a correctness hazard, since `api.opentela.ai` would then see two
providers for one model under two operator peer IDs. `start_relay_jsc.sh`
now handles that:

- **Ports are per-user by default** — `RELAY_TCP_PORT=45000+$(id -u)%1000`,
  `RELAY_UDP_PORT=59000+$(id -u)%1000` (matching the GLM-5.3 campaign guide;
  `id -u`, not `$USER`, which can repeat across sites). Two users get two
  relays on two ports, each with its own peer — no collision.
- **`start` refuses a busy port** it can't attribute to itself (without
  root, `ss -p` hides other users' pids but still shows the LISTEN line), so
  you get a clear message instead of a 60 s capture timeout.
- **`start` is idempotent** — if your tmux session is already up and
  `relay.multiaddr` exists, it re-prints and exits 0 instead of clobbering
  the file.
- **`stop` refuses** to kill a relay whose libp2p port is owned by another
  uid (that relay may be serving other workers).
- **`attach`** is the sharing path: if a relay is already running (yours or
  another's), print its multiaddr without starting a new one, then point your
  sbatch at it with `RELAY_ADDR_FILE` / `OTELA_RELAY_ADDR`.

For a fully independent relay (own peer ID, own store), set `DEPLOY_DIR` and
`RELAY_SEED`/`RELAY_TCP_PORT`/`RELAY_UDP_PORT` before running
`start_relay_jsc.sh`, then point the sbatch at the new `relay.multiaddr`
with `RELAY_ADDR_FILE`.

## §7. Teardown: how the job leaves OpenTela cleanly

The registry failure mode this choreography exists for: an otela worker that
is SIGKILLed never publishes its LEFT record, and the head then keeps a
`connected: true` row pointing at a dead endpoint — the staleness sweep skips
peers that registered a service, so the head keeps round-robining real traffic
into a worker that no longer exists. Three coordinated mechanisms prevent it:

1. **`#SBATCH --signal=B:TERM@120`** — signals the *batch shell* (that is what
   `B:` is for) 120 s before wall-time expiry, which is the window `stop_otela`
   needs. Without it the only grace is `KillWait` (default 30 s) between
   SIGTERM and SIGKILL — less than stop_otela's poll plus otela's ~10 s
   shutdown (AnnounceLeave → 5 s CRDT drain → `srv.Shutdown`) — so teardown
   loses the race at every wall-time expiry and the LEFT record is never
   published.
2. **Signal the step, not the task.** The worker runs as an `srun --overlap`
   step (`-J otela-worker-$JOBID`) because srun does not reliably relay signals
   to the task; `stop_otela` finds the step via `squeue` and `scancel -s TERM`s
   it, then polls up to **20 s**. On the `scancel` path the only grace is
   KillWait's 30 s, so a longer poll just guarantees we are still waiting when
   the SIGKILL lands.
3. **Verify, never assume.** "Step already gone" and "left cleanly" used to
   look identical in job logs, which made a lost LEFT invisible after the
   fact. The teardown now WARNs loudly when the step is missing, and greps
   `Announcing myself as LEFT|Leaving network` in `otela.log` to confirm the
   leave. ("Leaving network" is logged at the *start* of shutdown and the CRDT
   delta drains after it — don't cut the wait short on that line.)

`stop_otela` also guards against double execution: bash runs the EXIT trap
after the TERM handler returns, so without the guard the teardown runs twice
and the second pass reports a spurious warning.

## §8. Engine-env guards (the lines that look paranoid)

Each of these is in the sbatch's `engine.sh` because its absence failed live:

- **`NCCL_SOCKET_FAMILY=IPv4` + `GLOO_SOCKET_FAMILY=IPv4`** — some JSC racks
  resolve peer hostnames to IPv6 (`fdff::`); NCCL's bootstrap tries the AAAA
  record, fails with errno 97 (address family not supported) and has **no
  fallback** → PP/EP communicator init deadlocks. Verified: TP8/EP8/PP4 init
  completes all 32 comms with this set, vs **5/32 without**.
- **`NCCL_IB_RETRY_CNT=7` + `NCCL_IB_TIMEOUT=22`** — the defaults (0 retries,
  ~36 s timeout) let one transient switch hiccup abort the RC QP; raised
  values let the transport ride out the hiccup.
- **`DIST_TIMEOUT=60`** — racks whose GH200s advertise CUDA multimem make
  sglang attempt a `CUDASymmetricMemory` rendezvous on the default (32-rank,
  cross-node) process group; IPC handles cannot cross nodes, and every rank
  burns the full NCCL default (600 s) before falling back. 60 s is ample for
  the real rendezvous (sub-second) while keeping the fallback fast.
- **Metrics OFF by default (`ENABLE_METRICS`)** — PyTorch's NCCL watchdog, on
  a collective timeout, re-registers Prometheus metrics that sglang already
  registered → `prometheus_client` `DuplicateTimeseries` → the *server
  crashes*. One transient cross-node hang is enough to bring down the whole
  deployment. Opt in only for short, supervised benchmarking runs.
- **Caches on `/e/scratch`, never `/e/home`** — `/e/home` is quota-limited
  NFS; `$HOME`, HF/XDG/Triton caches are redirected, and the JIT cache dirs
  are rank-local (§4).
- **No socat/forwarder is needed** — otela's registrar health-checks the
  hardcoded `localhost:$SERVE_PORT`; the worker runs *on* the serving node and
  apptainer shares the host network namespace, so the engine genuinely is on
  localhost. Forwarders are only for engine/worker on different hosts.

## Submit

(The step-by-step commands are in [Start here](#start-here) at the top; this
section adds the rcc flow and the TP32/EP32 experiment form.)

### From the login node (SSH)

```bash
# 0. one-time prep (login node): build image + (optional) stage SHARP
bash deployments/llm/jsc/kimi-k3/build_kimi_k3_image.sh
bash deployments/llm/jsc/kimi-k3/stage_sharp_plugin.sh

# 1. start the login-node relay (tmux; writes relay.multiaddr; see §6)
bash deployments/llm/jsc/kimi-k3/start_relay_jsc.sh

# 2. production default: Kimi-K3, 8 nodes, TP4×PP8 (~542 tok/s @ C=32)
sbatch deployments/llm/jsc/kimi-k3/serve_llm_otela_jsc.sbatch

# 3. experiment: TP32/EP32/PP1 with SHARP (boots stable; see §3 for the
#    throughput caveat — a2a=none is ~38× slower, deepep/megamoe crash on sm90)
sbatch --export=ALL,\
IMAGE=/e/scratch/reformo/$USER/kimi-k3/images/sglang-kimi-k3-cu12.sif,\
TP_SIZE=32,EP_SIZE=32,PP_SIZE=1,NNODES=8,\
SHARP_PLUGIN_DIR=/e/scratch/reformo/$USER/otela-llm/sharp-plugin-only \
  deployments/llm/jsc/kimi-k3/serve_llm_otela_jsc.sbatch
```

### From your local machine via `rcc`

The repository ships a project-local `.rcc/config.toml` with a `jsc`
profile that syncs to `/e/scratch/reformo/yao4/opentela-cookbook` and submits
through the `jsc` SSH alias (`login01.jupiter.fz-juelich.de`, configured in
`~/.ssh/config`).

```bash
# one-time: sync local code and this recipe to JSC
rcc --profile jsc push

# build image + (optional) stage SHARP
rcc --profile jsc run -- bash -lc \
  'bash /e/scratch/reformo/yao4/opentela-cookbook/deployments/llm/jsc/kimi-k3/build_kimi_k3_image.sh && \
   bash /e/scratch/reformo/yao4/opentela-cookbook/deployments/llm/jsc/kimi-k3/stage_sharp_plugin.sh'

# start the login-node relay (tmux; writes relay.multiaddr; see §6)
rcc --profile jsc run -- bash /e/scratch/reformo/yao4/opentela-cookbook/deployments/llm/jsc/kimi-k3/start_relay_jsc.sh

# production default: Kimi-K3, 8 nodes, TP4×PP8
rcc --profile jsc job submit deployments/llm/jsc/kimi-k3/serve_llm_otela_jsc.sbatch

# experiment: TP32/EP32/PP1 with SHARP
rcc --profile jsc job submit --sbatch-args='--export=ALL,IMAGE=/e/scratch/reformo/yao4/kimi-k3/images/sglang-kimi-k3-cu12.sif,TP_SIZE=32,EP_SIZE=32,PP_SIZE=1,NNODES=8,SHARP_PLUGIN_DIR=/e/scratch/reformo/yao4/otela-llm/sharp-plugin-only' \
  deployments/llm/jsc/kimi-k3/serve_llm_otela_jsc.sbatch

# monitor
rcc --profile jsc job status <JOBID>
rcc --profile jsc job tail <JOBID> -f
```

## Verify

The login node **cannot reach** the compute endpoint (`curl` → connection
refused; JSC firewalls login↔compute). Health checks must run from inside the
allocation. With `rcc` you can run those commands from your local machine by
prefixing them with `rcc --profile jsc run -- bash -lc '...'`:

```bash
rcc --profile jsc run -- bash -lc 'JOB=<JOBID>; HEAD=$(grep SERVICE_HEAD_NODE /e/scratch/reformo/yao4/kimi-k3/last_service.env | cut -d= -f2); srun --jobid=$JOB --overlap --nodes=1 -w "$HEAD" curl -s http://127.0.0.1:30000/health ; echo'
```

```bash
JOB=<jobid>; HEAD=$(grep SERVICE_HEAD_NODE .../last_service.env | cut -d= -f2)
# health + model, from the head node:
srun --jobid=$JOB --overlap --nodes=1 -w "$HEAD" \
  curl -s http://127.0.0.1:30000/health ; echo
srun --jobid=$JOB --overlap --nodes=1 -w "$HEAD" \
  curl -s http://127.0.0.1:30000/v1/models | python3 -m json.tool
```

Benchmark from inside the allocation. servekit is pure stdlib, so the
no-egress compute nodes need zero installation beyond a checkout on the
shared FS (the sbatch already warns if `$SERVEKIT_DIR` is missing — default
`$DEPLOY_DIR/servekit`):

```bash
# one-time on a login node:
git clone --depth=1 https://github.com/eth-easl/servekit "$DEPLOY_DIR/servekit"

# standard sweep (login node CANNOT reach the engine — run inside the allocation):
srun --jobid=$JOB --overlap --nodes=1 -w "$HEAD" \
  bash meta/bench/cbench.sh http://127.0.0.1:30000 "1:16 8:32 32:64" 768 256 \
       --label k3-jsc-pp8
# or directly, container-side:
srun --jobid=$JOB --overlap --nodes=1 -w "$HEAD" \
  apptainer exec --bind /e/scratch:/e/scratch "$IMAGE" \
  env PYTHONPATH="$DEPLOY_DIR/servekit/src" \
  python3 -m servekit.cli bench --url http://127.0.0.1:30000 \
    --requests 64 --concurrency 32 --input-len 768 --output-len 256 \
    --out "$RUNDIR/bench_c32.json"
```

(The legacy `bench.py` form is retired here; its `--input-len` counted
tokens, servekit's counts words — do not mix the two records.)

Every launch also produces, without any of the above: **before** the otela
worker registers, a verification bench C=16 n=64 written into
`$RUNDIR/coldstart.node0.json` (merged with the cold-start timeline), and a
per-node cold-start profile `$RUNDIR/coldstart.node<RANK>.json` — the
engine's phase timestamps from spawn to serving. The bench runs **before**
registration so otela traffic cannot pollute the numbers; `SERVEKIT_BENCH=0`
disables the bench; a missing servekit checkout just skips both, with a WARN.

Two servekit profile semantics worth knowing: `servekit profile` **kills** the
engine on `--timeout` expiry, so its timeout tracks `HEALTH_TIMEOUT` — never
set it shorter. Workers are profiled on sglang's multi-node "Dummy health
check server started" log line; if a future image stops printing it, workers
stay unprofiled (check `coldstart.node<R>.json` after image bumps).

Confirm SHARP engagement in the sglang log (`NCCL_DEBUG=INFO`):
```
NET/Plugin: Loaded collnet plugin SHARP (v9)
8 coll channels, 8 collnet channels, ...
NET/IB : Using [0]mlx5_0:1/IB/SHARP ...   (/SHARP suffix = engaged)
```

## Knobs (env, all overridable)

| Knob | Default | Notes |
|---|---|---|
| `TP_SIZE` | 4 | intra-node NVLink domain. |
| `PP_SIZE` | `$NNODES` | one stage/node. |
| `EP_SIZE` | `$TP_SIZE` | only meaningful with `MOE_A2A_BACKEND` set. |
| `MOE_BACKEND` | `marlin` | only SM90 mxfp4-packed runner. |
| `CG_DECODE` | `full` | decode CUDA-graph backend. |
| `MOE_A2A_BACKEND` | *(empty=none)* | `deepep`/`megamoe` crash on K3-sm90 (§3); kept for Blackwell. |
| `SHARP_PLUGIN_DIR` | *(empty=off)* | set to the staged dir + a cu12 image to enable cross-node SHARP (§2). |
| `MEM_FRAC` | 0.90 | |
| `CTX_LEN` | 1048576 | Kimi-K3's native 1 Mi-token window; bounds request length, does not size the KV pool. |
| `IMAGE` | `.../sglang-kimi-k3.sif` | use `-cu12.sif` for SHARP. |
| `MODEL_PATH` | `/e/data1/.../Kimi-K3` | shared model cache (no internet on compute). |
| `DIST_TIMEOUT` | 60 | short on purpose — skips the doomed cross-node multimem rendezvous fast (§8). |
| `HEALTH_TIMEOUT` | 5400 | health wait before giving up registration; the servekit profile timeout tracks it (never set shorter, §"Verify"). |
| `K3_NIC` / `DIST_PORT` / `SERVE_PORT` | `ib0` / 20000 / 30000 | fabric iface, torch-dist rendezvous port, engine port. |
| `FLASHKADA_PREFIX` | `$DEPLOY_DIR/flashkda_prefix` | auto-enables FlashKDA when built (§1); injects `SGLANG_PYTHONPATH_EXTRA` unless that is already set. |
| `ENABLE_METRICS` | *(empty=off)* | `1` opts into Prometheus metrics — crash risk, see §8. |
| `OTELA_REQUIRE_SIGNED` | `false` | set `true` only when every node in the path is an official signed release (§6). |
| `TVM_FFI_CACHE_DIR` / `EP_JIT_CACHE_DIR` | rank-local under `$DEPLOY_DIR/cache` | race-proof (§4). |
| `SGLANG_EXTRA_ARGS` | *(empty)* | appended to the `sglang serve` line. |
| `SERVEKIT_DIR` | `$DEPLOY_DIR/servekit` | servekit checkout on shared FS (stdlib-only, no install; stage once from a login node). Missing → WARN, engine runs unprofiled, auto-bench skipped. |
| `SERVEKIT_BENCH` | `1` | `0` skips the post-health, pre-registration verification bench (C=16 n=64 by default) in `$RUNDIR`. |
| `SERVEKIT_BENCH_REQUESTS` / `SERVEKIT_BENCH_CONCURRENCY` | `64` / `16` | single-level verification size; keep C ≤ 136 (`max_running_requests`, KDA pool). |

## Cluster facts (the things you can't rediscover from a manual)

- **Compute nodes have NO outbound internet** (not even Cloudflare:443) →
  OpenTela registers via a **login-node relay**; model + image + all caches must
  be pre-staged on `/e/data1` or `/e/scratch`.
- **`/opt/mellanox` and host `/lib64` are login-node-only** (not mounted on
  compute) → SHARP runtime must be staged to `/e/scratch`.
- **`/e/software` (NVHPC bundles) IS mounted on compute** → usable for staging
  sources, but the *binaries* are arch/CUDA-pinned.
- **JSC default stage is 2026** as of this writing.
- **Apptainer-only** (no Enroot/Pyxis/EDF) → `apptainer exec --nv <sif>`.
- **GH200 = sm90, 96 GiB HBM**, 4×/node, NVLink intra-node, IB+SHARP inter-node.
