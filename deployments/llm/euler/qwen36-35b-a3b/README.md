# Qwen3.6-35B-A3B-FP8 on Euler (ETH Zürich, RTX PRO 6000) → OpenTela

Serves `Qwen/Qwen3.6-35B-A3B-FP8` (hybrid GDN MoE, FP8, 256 experts / 8 active,
mixed linear + full attention, MTP head, VLM) on a single **NVIDIA RTX PRO 6000
96 GB** (Blackwell, sm_120) inside the stock `lmsysorg/sglang:v0.5.16-cu129-runtime`
Apptainer image, and registers it on OpenTela via a **login-node relay**.
Euler compute nodes have no outbound raw-TCP internet (the `eth_proxy` module
gives HTTP(S) only), so the otela worker on the compute node cannot dial the
public head directly — it hops through a relay running on a login node, exactly
like the JSC recipe. The RTX PRO 6000 is the only FP8-capable GPU type
accessible on Euler; the RTX 3090 (sm_86), Quadro RTX 6000 (sm_75), and A100
(sm_80) have no FP8 hardware.

## Why `--gpus=nvidia_rtx_pro_6000:1` (no `--partition`)

The FP8 model needs FP8-capable hardware for native FP8 blockwise GEMM. On
Euler that means **Blackwell RTX PRO 6000** (eu-g7-001–010, sm_120, 96 GB,
partition `cuda13pr`). But Euler's `cli_filter/lua` plugin **strips the GRES
type** from `--gpus=nvidia_rtx_pro_6000:1`, leaving a generic `gres/gpu=1`
request, and **reassigns partitions** — an explicit `--partition=cuda13pr.4h`
is silently routed to `gpu.4h` (non-Blackwell, e.g. RTX 2080 Ti 11 GB — not
enough for a 35 GB model). The only incantation that avoids `gpu.4h` is a
typed GRES with **no** `--partition`: the cli_filter routes it to
`cuda13pr.4h,gpupr.4h` (Blackwell 96 GB **or** A100 80 GB — whichever is free
first). (Verified: `sbatch --test --gpus=nvidia_rtx_pro_6000:1` →
"nodes eu-g7-002 in partition cuda13pr.4h"; a plain `--gpus=1
--partition=cuda13pr.4h` → "Partition=gpu.4h … eu-lo-g2-031", an RTX 2080 Ti.)

Because the GRES type is stripped, the job CAN land on an A100 80 GB
(gpupr.4h) when Blackwell is fully allocated (observed: job 9466280 ran on
`eu-a65-05`). The A100 has no FP8 hardware, so sglang dequantizes FP8→BF16
per kernel (slower, ~same memory since weights stay in FP8). The default
`MEM_FRACTION_STATIC=0.85` is tuned for Blackwell 96 GB; on A100 80 GB it
may need lowering (see Knobs). This is acceptable — the goal is a serving
endpoint, not peak throughput.

The `--mem-per-cpu=8G` (not `--mem`) is also forced by the cli_filter — plain
`--mem` is rejected with "sbatch: error: Batch job submission failed: Requested
node configuration is not available".

## Site-specific fixes (all baked into the sbatch)

1. **Login-node relay for egress.** Symptom: compute node
   `bash -c "exec 3<>/dev/tcp/api.opentela.ai/443"` → "Cannot assign
   requested address"; same for p2p.opentela.ai, github.com, huggingface.co.
   Root cause: Euler compute nodes have no outbound raw TCP; `eth_proxy` only
   forwards HTTP(S) via `proxy.service.consul:3128`. The login node DOES have
   raw TCP to all of the above (verified OPEN). Fix: run
   `start_relay_euler.sh` on a login node; the sbatch's worker dials it via
   Euler-internal TCP (compute CAN reach login:22 and login:18905, both
   verified OPEN).

2. **Typed GRES, no `--partition`.** See "Why" above. The cli_filter strips
   the GRES type and reassigns `--partition`; only `--gpus=nvidia_rtx_pro_6000:1`
   (no partition) lands on `cuda13pr.4h,gpupr.4h` (Blackwell or A100 80 GB),
   never `gpu.4h` (RTX 2080 Ti 11 GB). On A100 the FP8 weights are dequantized
   per kernel (no native FP8 GEMM); lower `MEM_FRACTION_STATIC` if OOM.

3. **`--mem-per-cpu`, not `--mem`.** Euler's cli_filter rejects `--mem`.
   Use `--mem-per-cpu=8G`.

4. **All state off `/cluster/home` (over quota).** Home was at 46.4/47.7 GB
   at time of writing. The model (35 GB), SGLang image, caches, otela
   config, and run dirs all live on `/cluster/scratch/$USER/otela-qwen36`
   (482 TB, 2.6 TB quota). The otela *binary* (`$DEPLOY/bin/otela-v0.2.3`,
   72 MB) is read-only on scratch — that's fine. The otela *BadgerDB* (`.ocfcore`) is
   relocated to `$TMPDIR/otela-home` (node-local SSD, auto-cleaned by Slurm)
   via a `HOME` env override, avoiding both home quota and Lustre stale file
   handles (documented in the JSC recipe).

5. **`ulimit -s 67108864` (64 MB stack).** sglang's hybrid-GDN call stack
   overflows the default 8 MB (documented in the dgx-spark bring-up of the
   same model). Apptainer inherits the host ulimit, so the engine script
   sets it before `apptainer exec`.

6. **`--attention-backend triton`.** The hybrid GDN (Gated DeltaNet) linear +
   full attention schedule is rejected by `torch_native` on Blackwell; triton
   is the only backend that accepts it (from the dgx-spark bring-up).

7. **Fresh peer ID per invocation (`OPENTELA_SEED=$$`).** A graceful `stop`
   announces LEFT to the mesh (a CRDT tombstone). Restarting with the same
   seed reproduces the peer ID but a fresh CRDT cannot override the
   tombstone, so the peer never re-appears for many minutes and the gateway
   returns 503 "No provider found". Defaulting to `$$` (PID) gives a fresh
   peer ID every start. (From the dgx-spark recipe; the old sai-v0.0.6
   binary had the same issue.)

## Architecture

```
client ─https─▶ api.opentela.ai ──P2P──▶ ocf head
                                         ▲ WSS out via p2p.opentela.ai:443
┌── Euler login node (e.g. eu-login-02, 129.132.93.77) ─────────────┐
│  RELAY   start_relay_euler.sh  (tmux session `opentela-relay`)      │
│          listens on <login-ip>:18905  peer QmPAfjhRSrnR…M8n      │
│          BadgerDB in /tmp/opentela-relay/ (node-local)             │
└────────────────────────────────────────────────────────────────────┘
                    ▲ TCP, Euler-internal (compute CAN reach login)
┌── compute node (this job, e.g. eu-g7-003) ─────────────────────────┐
│  sglang serve  (apptainer --nv, host netns) on 127.0.0.1:30000      │
│  otela worker  (host proc) --role worker --service.port 30000       │
│                BadgerDB in $TMPDIR/otela-home/ (node-local)        │
└────────────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `serve_qwen36_otela_euler.sbatch` | The recipe — serves sglang + registers otela worker, one self-contained sbatch |
| `start_relay_euler.sh` | Login-node relay (tmux, writes `relay.multiaddr`); run once before submitting |
| `README.md` | This file |

## Prerequisites (one-time, on a login node)

```bash
DEPLOY=/cluster/scratch/$USER/otela-qwen36
mkdir -p "$DEPLOY"/{model,images,logs,cache/hf,cache/triton,cache/xdg,cache/apptainer,home,bin}

# 1. Model (35 GB FP8 shard tree, 56 files)
export HF_HOME="$DEPLOY/cache/hf"
huggingface-cli download Qwen/Qwen3.6-35B-A3B-FP8 \
  --local-dir "$DEPLOY/model/Qwen3.6-35B-A3B-FP8"

# 2. SGLang image (v0.5.16, cu129 — the oldest release that supports the
#    hybrid GDN schedule). NOTE: mksquashfs needs ~20 GB RAM to build the
#    SIF; login nodes are often OOM-killed. Submit a CPU job instead:
#      sbatch --account=es_icp --partition=hpc.4h --cpus-per-task=4 \
#        --mem-per-cpu=16G --time=01:00:00 \
#        --wrap='module load eth_proxy; export APPTAINER_CACHEDIR=$TMPDIR/ac \
#          APPTAINER_TMPDIR=$TMPDIR/ab; mkdir -p $TMPDIR/ac $TMPDIR/ab \
#          $DEPLOY/images; apptainer pull $DEPLOY/images/sglang_v0.5.16_cu129.sif \
#          docker://lmsysorg/sglang:v0.5.16-cu129-runtime'
#    (eth_proxy gives compute nodes HTTPS egress to Docker Hub.)

# 3. OpenTela binary (v0.2.3, amd64 — from GitHub releases)
curl -L -o "$DEPLOY/bin/otela-v0.2.3" \
  "https://github.com/eth-easl/OpenTela/releases/download/v0.2.3/opentela-amd64"
chmod +x "$DEPLOY/bin/otela-v0.2.3"

# 4. Relay config (if not already at ~/opentela/relay.cfg.yaml)
cat > ~/opentela/relay.cfg.yaml <<'YAML'
name: euler-relay
seed: "99"
port: "18092"
tcpport: "18905"
udpport: "18820"
mode: full
loglevel: debug
cleanslate: true
role: relay
public-addr: "127.0.0.1"
security:
  require_signed_binary: false
solana:
  skip_verification: true
bootstrap:
  sources:
    - "https://bootstraps.opentela.ai/v1/dnt/bootstraps"
YAML
```

## Run

### From the login node (SSH)

```bash
# 1. Start the relay on a login node (writes relay.multiaddr automatically)
bash start_relay_euler.sh

# 2. Submit the serving job (reads relay.multiaddr; lands on Blackwell)
sbatch serve_qwen36_otela_euler.sbatch

# 3. Watch it come up
squeue -u $USER
tail -f /cluster/scratch/$USER/otela-qwen36/logs/qwen36-*.out
```

### From your local machine via `rcc`

The repository ships a project-local `.rcc/config.toml` with an `euler`
profile that syncs to `/cluster/scratch/xiayao/opentela-cookbook` and submits
through the `euler` SSH alias (configured in `~/.ssh/config`). The profile
also forwards `http_proxy`/`https_proxy` so the login-node relay can reach
OpenTela bootstraps through `eth_proxy`.

```bash
# one-time: sync local code and this recipe to Euler
rcc --profile euler push

# start the login-node relay (kept running in a tmux session by the script)
rcc --profile euler run -- bash -lc \
  'bash /cluster/scratch/xiayao/opentela-cookbook/deployments/llm/euler/qwen36-35b-a3b/start_relay_euler.sh'

# submit the serving job
rcc --profile euler job submit deployments/llm/euler/qwen36-35b-a3b/serve_qwen36_otela_euler.sbatch

# monitor
rcc --profile euler job status <JOBID>
rcc --profile euler job tail <JOBID> -f

# inspect logs from your local machine
rcc --profile euler run -- tail -f /cluster/scratch/xiayao/otela-qwen36/logs/qwen36-<JOBID>.out
```

## Verify

```bash
# Engine is serving (from the compute node, or via the gateway after registration)
curl -s http://127.0.0.1:30000/v1/models | python3 -m json.tool

# Direct inference (from the compute node)
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.6-35B-A3B-FP8","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'

# otela worker joined the mesh (look for "Updating peer" and no LEFT line)
tail -f /cluster/scratch/$USER/otela-qwen36/run-*/otela.log

# via rcc from your local machine:
rcc --profile euler run -- tail -f /cluster/scratch/xiayao/otela-qwen36/run-<JOBID>/otela.log

# Routed request through the gateway (from anywhere with internet)
curl -s http://140.238.223.116:8092/v1/service/llm/v1/chat/completions \
  -H 'Authorization: Bearer test-token' -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.6-35B-A3B-FP8","messages":[{"role":"user","content":"hi"}]}'
```

Log lines to look for in `qwen36-<jobid>.out`:
- `[<ts>] engine healthy — starting otela worker` (sglang is up)
- `[<ts>] otela worker started (pid …)` (registration in progress)

In `run-<jobid>/otela.log`:
- `Connected to peer: QmPAfjhRSrnR…` (relay reached)
- `Updating peer: [<peer-id>] triggered by msg received` (on the mesh)

## Knobs

### `serve_qwen36_otela_euler.sbatch`
| Env | Default | Why |
|-----|---------|-----|
| `DEPLOY_DIR` | `/cluster/scratch/$USER/otela-qwen36` | Home is over quota |
| `IMAGE` | `$DEPLOY_DIR/images/sglang_v0.5.16_cu129.sif` | v0.5.16 = oldest hybrid-GDN release |
| `MODEL` | `$DEPLOY_DIR/model/Qwen3.6-35B-A3B-FP8` | FP8 shard tree |
| `SERVED_MODEL_NAME` | `Qwen/Qwen3.6-35B-A3B-FP8` | OpenTela identity (see conventions) |
| `SERVE_PORT` | `30000` | sglang listen port |
| `ATTENTION_BACKEND` | `triton` | torch_native rejected on Blackwell for hybrid GDN |
| `MEM_FRACTION_STATIC` | `0.85` | Room for SSM state + CUDA graphs within 96 GB |
| `REASONING_PARSER` | `qwen3` | Model emits `<dynamic_thinking>` tags |
| `HEALTH_TIMEOUT` | `900` | FP8 load + graph capture takes several min |
| `OTELA_BIN` | `$DEPLOY_DIR/bin/otela-v0.2.3` | otela binary (OpenTela v0.2.3, x86_64) |
| `OTELA_RELAY_ADDR` | *(see relay.multiaddr)* | Override the relay multiaddr |
| `RELAY_ADDR_FILE` | `$DEPLOY_DIR/relay.multiaddr` | Written by start_relay_euler.sh |
| `OPENTELA_SEED` | `$$` | Fresh peer ID per start (avoids LEFT tombstone) |
| `OTELA_TCP_PORT` | `43905` | Worker libp2p TCP |
| `OTELA_UDP_PORT` | `59820` | Worker libp2p UDP |
| `OTENTELA_SERVICE_NAME` | `llm` | OpenTela service name |
| `OTELA_IDENTITY_GROUP` | `model=… cluster=euler gpu=rtx_pro_6000` | Labels attached to service (via cfg.yaml — no --label flag) |
| `SGLANG_EXTRA_ARGS` | *(empty)* | Passthrough for experimental sglang flags |

### `start_relay_euler.sh`
| Env | Default | Why |
|-----|---------|-----|
| `DEPLOY_DIR` | `/cluster/scratch/$USER/otela-qwen36` | Logs + relay.multiaddr |
| `OTELA_BIN` | `$DEPLOY_DIR/bin/otela-v0.2.3` | otela binary (v0.2.3) |
| `RELAY_CFG` | `$HOME/opentela/relay.cfg.yaml` | Relay config (role: relay) |
