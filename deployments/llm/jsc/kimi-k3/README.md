# Kimi-K3 on JSC Jupiter (SGLang, GH200) → OpenTela

Serve `moonshotai/Kimi-K3` (2.8 T-parameter hybrid MoE, mxfp4) on **JSC Jupiter
Booster** (4× GH200 / node, aarch64, InfiniBand + SHARP, Slurm, Apptainer) and
register it on OpenTela.

This README records **what we verified on this cluster** — the topology that
wins, the ones that don't and why, and the exact failures behind each choice.
The companion `serve_llm_otela_jsc.sbatch` is the runnable artifact; the
findings here are the "why" behind its non-obvious settings.

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
  4-bit model on 96 GiB. (Recipe comment line ~162.)
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

### Verified benchmarks (TP4/PP8, 8 nodes, 1024-in / 256-out)

Taken with the shared `meta/bench/` harness — see
[`meta/bench/README.md`](../../../meta/bench/README.md) for the full protocol (warmup, the
C=1 trap, how to read the curve shape).

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
| `serve_llm_otela_jsc.sbatch` | One self-contained sbatch: sglang engine (apptainer `--nv`) + otela worker + optional vmagent. Defaults: Kimi-K3, 8 nodes, TP4×PP8. |
| `build_kimi_k3_image.sh` | Build the sglang `.sif` on a login node. CUDA-13 (default) and CUDA-12 (`-cu12`, required for SHARP) variants. |
| `stage_sharp_plugin.sh` | One-command: stage a self-contained SHARP plugin dir on `/e/scratch` (closes all 5 gaps in §2). Run once on a login node. |
| `build_flashkda_prefix.sh` | Optional: build the FlashKDA Python prefix to bind-mount into the container. |

## Submit

```bash
# 0. one-time prep (login node): build image + (optional) stage SHARP
bash deployments/llm/jsc/kimi-k3/build_kimi_k3_image.sh
bash deployments/llm/jsc/kimi-k3/stage_sharp_plugin.sh

# 1. production default: Kimi-K3, 8 nodes, TP4×PP8 (~542 tok/s @ C=32)
sbatch deployments/llm/jsc/kimi-k3/serve_llm_otela_jsc.sbatch

# 2. experiment: TP32/EP32/PP1 with SHARP (boots stable; see §3 for the
#    throughput caveat — a2a=none is ~38× slower, deepep/megamoe crash on sm90)
sbatch --export=ALL,\
IMAGE=/e/scratch/reformo/$USER/kimi-k3/images/sglang-kimi-k3-cu12.sif,\
TP_SIZE=32,EP_SIZE=32,PP_SIZE=1,NNODES=8,\
SHARP_PLUGIN_DIR=/e/scratch/reformo/$USER/otela-llm/sharp-plugin-only \
  deployments/llm/jsc/kimi-k3/serve_llm_otela_jsc.sbatch
```

## Verify

The login node **cannot reach** the compute endpoint (`curl` → connection
refused; JSC firewalls login↔compute). Health checks must run from inside the
allocation:

```bash
JOB=<jobid>; HEAD=$(grep SERVICE_HEAD_NODE .../last_service.env | cut -d= -f2)
# health + model, from the head node:
srun --jobid=$JOB --overlap --nodes=1 -w "$HEAD" \
  curl -s http://127.0.0.1:30000/health ; echo
srun --jobid=$JOB --overlap --nodes=1 -w "$HEAD" \
  curl -s http://127.0.0.1:30000/v1/models | python3 -m json.tool
```

Benchmark from inside the allocation (compute nodes have `aiohttp` only inside
the container):

```bash
srun --jobid=$JOB --overlap --nodes=1 -w "$HEAD" \
  apptainer exec --bind /e/scratch:/e/scratch "$IMAGE" \
  python3 bench.py "1:16 8:32 32:64" 127.0.0.1 30000 1024 256
```

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
| `MOE_A2A_BACKEND` | *(empty=none)* | `deepep`/`megamoe` crash on K3-sm90 (§3); kept for Blackwell. |
| `SHARP_PLUGIN_DIR` | *(empty=off)* | set to the staged dir + a cu12 image to enable cross-node SHARP (§2). |
| `MEM_FRAC` | 0.90 | |
| `CTX_LEN` | 1048576 | Kimi-K3's native 1 Mi-token window; bounds request length, does not size the KV pool. |
| `IMAGE` | `.../sglang-kimi-k3.sif` | use `-cu12.sif` for SHARP. |
| `MODEL_PATH` | `/e/data1/.../Kimi-K3` | shared model cache (no internet on compute). |
| `DIST_TIMEOUT` | 60 | raise for large TP/EP init. |
| `TVM_FFI_CACHE_DIR` / `EP_JIT_CACHE_DIR` | rank-local under `$DEPLOY_DIR/cache` | race-proof (§4). |
| `SGLANG_EXTRA_ARGS` | *(empty)* | appended to the `sglang serve` line. |

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
