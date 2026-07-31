# Benchmarking an LLM service

How we measure LLM serving throughput in this cookbook — the strategy, the
exact protocol, and the traps we hit (so you don't repeat them). This governs
every `deployments/llm/<site>/` recipe. The harness lives in `bench/` and is
shared across sites; only the run mechanics (login-vs-compute reachability,
container) differ.

## The one number we care about

**Aggregate output tokens/sec at a controlled concurrency**, measured after a
warmup, from *inside the allocation*, with sequence length and model config
stated alongside it.

```
agg_out_tok_s = total_completion_tokens_across_all_requests / wall_clock
```

Everything else (latency p50/max, per-req tok/s) is a diagnostic that explains
*why* the aggregate is what it is, not a headline. A model is "fast" on a
cluster when `agg_out_tok_s` scales up with concurrency until it hits a
ceiling — and we report the ceiling, not the peak of a broken run.

---

## The C=1 rule (the trap we hit)

> **A single-request measurement is not a serving number for a distributed
> topology.** State concurrency with every throughput claim, and never quote
> C=1 as the headline for TP>1 or EP>1.

We learned this the hard way on JSC. The SGLang Kimi-K3 cookbook recommends
TP32/EP32; we got it booting and measured **0.76 tok/s at C=1** — and almost
wrote that down as "the config is 38× slower." It is, but for a subtle reason:
with `moe_a2a_backend=none`, every MoE layer does an *unoptimized cross-node
allgather per decode step*, so C=1 is the **worst case** (latency-bound, no
bandwidth amortization). At C=1 you measure the per-step collective overhead,
not serving capacity. Even at the `max_running_requests=52` ceiling the same
config would max at ~40 tok/s aggregate — still bad, still 13× worse than
TP4/PP8's 542 — but you only know that by *sweeping*, not by one shot.

**Rules that follow:**
- Always sweep `C = 1, 8, 16, 32, …` up to *and past* `max_running_requests`.
- Report the full curve. The shape (linear → knee → flat) tells you whether
  you're pipeline-bound, capacity-bound, or collective-bound (see §Diagnostics).
- A C=1 number is a **latency floor / health check**, labeled as such — never
  the headline.

---

## Protocol

### 0. Confirm the server is up (one shot, labeled)
```bash
# from INSIDE the allocation (JSC: login can't reach compute)
python3 bench/oneshot.py 127.0.0.1 30000 64 16
# → model=... wall=1.3s in=64 out=16 decode_tok/s=12.3
```
This is the latency floor and a routing check. **Do not report it as
throughput.**

### 1. Warm up, then sweep (the real number)
```bash
# spec is CONC:NUMREQ, space-separated. Warmup is automatic (C=4 n=8, discarded).
# The FIRST measured level must be past warmup or it's meaningless.
python3 bench/bench.py "1:8 8:32 16:48 32:64 52:52" 127.0.0.1 30000 1024 256 \
  | tee bench-$MODEL-$JOB.txt
```
Each level prints one JSON line as it completes, so a timeout or kill still
leaves you the finished levels. The warmup pass builds CUDA graphs, JIT
kernels and warms the KV/KDA pools before measurement begins.

### 2. Find the knee (push past the ceiling)
`max_running_requests` is a hard ceiling — concurrency above it just queues.
Sweep past it deliberately to see the knee:
```bash
python3 bench/bench.py "32:64 48:48 64:64 96:96 128:64" 127.0.0.1 30000 1024 256
```
- `agg_out_tok_s` flat from C=48→128 while `lat_max_s` climbs → **capacity-
  bound** at `max_running_requests`; raise it (more HBM) if you have headroom.
- `agg_out_tok_s` climbs with C but `per_req_out_tok_s` barely moves →
  **pipeline-bound** (the good case on JSC TP4/PP8: 28.8→542, per-req only
  28.8→17.0).
- `agg_out_tok_s` is flat-and-low from C=1, `lat` huge and flat →
  **collective-bound** (the TP32/EP32 a2a=none case: fix the *backend*, not
  the bench).

### 3. (Cross-node topologies only) transport check
Before trusting a TP>1node / EP>1 benchmark, confirm the collectives actually
use the fast path — otherwise you benchmark a silent TCP fallback and blame
the model. This is a *transport* probe, never an LLM number:
```bash
NCCL_DEBUG=INFO srun -n$WORLD python3 bench/nccl_sharp_probe.py
# log: /SHARP suffix = switch offload engaged; Socket/IB = fallback
```
On JSC this is what caught that the SHARP plugin wasn't loading (apptainer
discards `srun --export`) — the symptom would otherwise have looked like a
slow model instead of a slow fabric.

### 4. (Reasoning models only) pin `reasoning_effort`
Kimi-K3 always thinks. Depth is `reasoning_effort` (low/high/**max** default).
`max` emits long chains-of-thought that dominate decode time and are the
**single biggest real-world throughput lever** — far bigger than topology.
Pin it explicitly per run and report it with every number:
```bash
# bench sends reasoning_effort in the payload — set it via the server's
# served config or pass it through your client. A run at effort=max and a
# run at effort=low are NOT comparable; never average across them.
```

---

## How to run from inside the allocation (site notes)

| | JSC Jupiter | Beverin |
|---|---|---|
| Can login reach compute? | **No** (firewall) — `srun --overlap` into the job | **Yes** |
| Internet on compute? | **No** — no HF, no tokenizer download | Yes (full) |
| Bench needs | `apptainer exec $IMAGE python3 bench.py …` (aiohttp only in container) | `python3 bench.py …` directly |
| Hostname to dial | head node of the job (`127.0.0.1` from that node) | per-rank `nid00XXXX:8080` |

Because token counts come from the response `usage` field (not a local
tokenizer), the harness needs **zero** internet and **zero** model files —
only the server and `aiohttp` (which the sglang container ships). This is why
the bench works unchanged on a no-egress cluster.

JSC example (run on the job's head node, not the login node):
```bash
JOB=1138651; HEAD=$(awk -F= '/SERVICE_HEAD_NODE/{print $2}' $DEPLOY_DIR/last_service.env)
srun --jobid=$JOB --overlap --nodes=1 -w "$HEAD" \
  apptainer exec --bind /e/scratch:/e/scratch "$IMAGE" \
  python3 $DEPLOY_DIR/../../cookbook/deployments/llm/bench/bench.py \
    "1:8 8:32 16:48 32:64 52:52" 127.0.0.1 30000 1024 256
```

---

## Reporting checklist (every published number carries)

- [ ] **Topology**: TP×PP×EP, nodes/GPUs, runner backend, a2a backend.
- [ ] **Sequence length**: in_tokens / out_tokens (a 1024/256 number and a
      512/128 number are NOT comparable).
- [ ] **Concurrency curve**: not one point. At minimum the knee and one
      point past `max_running_requests`.
- [ ] **`max_running_requests`** of the server config (the ceiling).
- [ ] **`reasoning_effort`** for reasoning models (low/high/max).
- [ ] **Warm state**: "post-warmup" vs "cold" — label it.
- [ ] **Status**: `VERIFIED` (with the job id and scale) or `TODO(unverified)`.

Example of a fully-specified claim:
> Kimi-K3, TP4×PP8, 8 nodes / 32 GH200, Marlin, a2a=none. 1024-in / 256-out,
> `max_running_requests=136`, `reasoning_effort` default. VERIFIED (job
> 1117371): 28.8 / 93.9 / 182.6 / 333.0 / 542.9 agg out tok/s at C = 1/4/8/
> 16/32, near-linear, pipeline-bound.

---

## Diagnostics: what the curve shape means

| Shape | `lat_max_s` | `per_req_out_tok_s` | Diagnosis | Fix |
|---|---|---|---|---|
| `agg` climbs, `per_req` flat-ish | moderate | holds | **Pipeline-bound** (good) | add stages / raise concurrency |
| `agg` flat past ceiling | climbs sharply | falls | **Capacity-bound** at `max_running_requests` | raise it (HBM permitting) |
| `agg` flat-and-low from C=1 | huge, flat | ~agg/C | **Collective-bound** (a2a=none on cross-node) | fix the MoE a2a backend |
| `agg` high at low C, collapses at high C | wild | collapses | **Unstable** (NCCL hang / OOM under load) | transport check (§3), then capacity |

The JSC results are the canonical example of reading this table:
- **TP4/PP8**: row 1 (pipeline-bound, 542 tok/s ceiling, the production point).
- **TP32/EP32 a2a=none**: row 3 (collective-bound, ~40 tok/s max, Blackwell-gated fix — see `jsc/README.md` §3).

---

## Files

| File | Purpose |
|------|---------|
| `bench/bench.py` | Warmup + concurrency sweep, JSON per level. The default harness. |
| `bench/bench_nowarm.py` | Same sweep, no warmup. Use for cold-start or when wall-clock is tight and partial results matter. |
| `bench/oneshot.py` | One request, stdlib only. Latency floor + health check — **not** throughput. |
| `bench/nccl_sharp_probe.py` | NCCL transport probe (SHARP/RDMA vs fallback). Network-layer check, **not** an LLM number. |

`MODEL` is an env var on all three LLM scripts, so the same harness serves
Kimi-K3 (JSC), GLM-4.7-Flash (Beverin), and anything else without edits.
