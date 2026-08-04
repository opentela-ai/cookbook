# `ollama/qwen3:1.7b` on a DGX Spark (NVIDIA GB10) via Ollama → OpenTela

Serves `ollama/qwen3:1.7b` (Qwen3 1.7 B, reasoning model) on a **DGX Spark** —
a single-node workstation with one NVIDIA GB10 GPU (sm_121, aarch64, 122 GB
unified memory) — using **Ollama** (a single static binary with bundled CUDA
v13 libs that include sm_121), and registers it on OpenTela as a standalone
`otela` sidecar. There is no scheduler and no container: the binary and the
pulled model blobs live under `./run/` next to the scripts.

## Why this site / this stack

A single GB10 in a workstation is the whole point — no cluster, no Slurm, no
relay. The trade-off in the sibling `qwen36-35b-a3b` recipe is that GB10/aarch64
has no CUDA torch wheel, no flashinfer wheel, and no prebuilt sgl-kernel, so
everything heavy comes from a golden image built on another DGX Spark.

**Ollama sidesteps all of that.** Its arm64 release bundles CUDA v13 libraries
that include sm_121 (compute capability 12.1), so it runs on the GB10 straight
out of the tarball — no golden image, no sgl-kernel build, no vendored sglang,
no Docker overlay. A 1.7 B model loads in seconds and occupies ~1.4 GB of the
122 GB unified memory. The cost is throughput and control: Ollama does not
expose sglang's TP/PP/mem-fraction knobs, and for a model this small on a GPU
this large that is the right trade-off.

## What makes this non-trivial

1. **Ollama has no `/health` endpoint.** Its root path `/` returns `200
   "Ollama is running"`, but `/health` returns `404`. OpenTela's default
   health-check path is `/health`, so registration silently hangs forever
   unless `--service.health_path /` is set. See [Site-specific fixes](#site-specific-fixes).
2. **Ollama's `/v1/models` reports bare tags (`qwen3:1.7b`), not `org/model-name`.**
   The cookbook convention requires a single `org/model-name` identity agreed
   across the engine, the otela `identity_group`, and the client `model` field
   (see [`conventions/`](../../../../../conventions/)). An Ollama **Modelfile
   alias** (`FROM qwen3:1.7b`) publishes `ollama/qwen3:1.7b` instead, and the
   bare tag is then removed so only the alias is served.
3. **`ollama create` rejects stdin Modelfiles.** `ollama create X -f -` fails
   with `no Modelfile or safetensors files found`; a real file on disk is
   required. The serve script writes a temp Modelfile for this.

## Prerequisites

All relative to the recipe directory; every one is overridable via env.

| What | Default | How it is obtained |
|------|---------|--------------------|
| Ollama binary | `./run/ollama/bin/ollama` | `bash download_ollama.sh` fetches the v0.32.5 arm64 release (1.5 GB `.tar.zst`, bundles CUDA v12 + v13 libs). Or set `OLLAMA_BIN` to an existing install. |
| `otela` binary | `./run/otela/otela` | OpenTela `opentela-arm64` v0.2.3 (commit e9d1696, official release). `bash download_otela.sh` fetches it; or set `OTELA_BIN`. |
| Model | pulled automatically | `qwen3:1.7b` (1.4 GB) is pulled from ollama.com by `serve_qwen3_ollama.sh` on first run. |

## Architecture

```
ollama serve (host process, 127.0.0.1:11434)   / = 200, /v1/models ready
     ^   model id reported: ollama/qwen3:1.7b (Modelfile alias)
     |
     | 127.0.0.1:11434
     |
otela sidecar (host process)   :43905 libp2p  ──►  bootstrap peer (ocf-1 head)
```

The sidecar topology means the engine is **not supervised** by `otela` — if
Ollama restarts on a different port, re-run `register_qwen3_otela.sh daemon`
with the new port.

## Memory budget

The 1.4 GB model occupies a negligible fraction of the 122 GB unified memory;
no tuning is required. Ollama's vram-based default context is 262,144 tokens
(logged at startup), far more than this model needs.

## Files

| File | Purpose |
|------|---------|
| `download_ollama.sh` | Fetch + extract the Ollama arm64 binary (with bundled CUDA libs) into `./run/ollama/` |
| `download_otela.sh` | Fetch the OpenTela `opentela-arm64` v0.2.3 binary into `./run/otela/otela` |
| `serve_qwen3_ollama.sh` | Start/stop Ollama, pull `qwen3:1.7b`, create the `ollama/qwen3:1.7b` alias, wait for `/v1/models`, write `last_service.env` |
| `register_qwen3_otela.sh` | Start/stop/status the standalone `otela` sidecar that publishes the `llm` service |

## Run

```bash
# 1. one-time: fetch the two binaries
bash download_ollama.sh
bash download_otela.sh

# 2. start Ollama + pull model + create alias (waits for readiness)
bash serve_qwen3_ollama.sh

# 3. register on OpenTela as a background sidecar
bash register_qwen3_otela.sh daemon

# direct inference (Ollama is on 127.0.0.1:11434):
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ollama/qwen3:1.7b","messages":[{"role":"user","content":"hi"}],"max_tokens":128}'

# tear down
bash register_qwen3_otela.sh stop
bash serve_qwen3_ollama.sh stop
```

## Verify

Verified on ds5 (dgx-spark-05, aarch64, GB10) on 2026-08-04: Ollama v0.32.5
served on `127.0.0.1:11434`, the otela sidecar (peer
`QmdtGkzQn5cCZ6CEW33mw6wto4pKfzTwbwD9BkPQG1qzeA`) registered against the ocf-1
head, and a routed request returned `200` with `model: ollama/qwen3:1.7b` and
`X-Computing-Node: QmdtGkzQn5cCZ6CEW33mw6wto4pKfzTwbwD9BkPQG1qzeA`.

Re-check on a fresh start:

```bash
# Ollama is serving (note: /health is 404; use / or /v1/models)
curl -s http://127.0.0.1:11434/v1/models | python3 -m json.tool
#  -> one entry: {"id": "ollama/qwen3:1.7b", "owned_by": "ollama"}

# otela sidecar joined the mesh (look for "Server started" + "Health check passed")
tail -f ./run/otela/otela.log
bash register_qwen3_otela.sh status

# our peer in the head's gateway table:
curl -s http://140.238.223.116:8092/v1/dnt/table | python3 -m json.tool
#  -> our peer shows: service=[{"name":"llm","status":"connected",
#     "identity_group":["model=ollama/qwen3:1.7b"],"port":"11434"}]

# routed request through the ocf-1 head gateway:
curl -s http://140.238.223.116:8092/v1/service/llm/v1/chat/completions \
  -H 'Authorization: Bearer test-token' -H 'Content-Type: application/json' \
  -d '{"model":"ollama/qwen3:1.7b","messages":[{"role":"user","content":"Reply with one word: ping"}],"max_tokens":256}' \
  | python3 -m json.tool
#  -> 200, choices[0].message.content = "ping" (qwen3:1.7b is a reasoning model,
#     so most of max_tokens is spent in message.reasoning first)
```

## Site-specific fixes

1. **Registration hangs forever with default `/health` health path.** Symptom:
   `otela start` logs `Health check passed after 1/6000 attempts` only if
   `/health` answers; on Ollama it never does (`404`). Root cause: Ollama
   exposes no `/health` route — only `/` (`200 "Ollama is running"`) and
   `/api/version` (`200 {"version":"..."}`). Fix: pass
   `--service.health_path /` to `otela start` (set via `HEALTH_PATH` env in
   `register_qwen3_otela.sh`). Verified: with `/`, the health check passes on
   the first attempt and the service registers within seconds.
2. **`/v1/models` reported `qwen3:1.7b`, not `org/model-name`.** Symptom: the
   peer registers with `identity_group=["model=qwen3:1.7b"]`, which violates
   the cookbook's single `org/model-name` convention and is inconsistent with
   the client `model` field. Root cause: Ollama tags are bare (`name:tag`),
   and `/v1/models` echoes the tag with `owned_by=library`. Fix: create an
   alias `ollama/qwen3:1.7b` from `qwen3:1.7b` via a Modelfile (`FROM
   qwen3:1.7b`), which makes `/v1/models` report `ollama/qwen3:1.7b`
   (`owned_by=ollama`); then `ollama rm qwen3:1.7b` so only the alias is
   listed (the alias keeps the shared blobs). This is baked into
   `serve_qwen3_ollama.sh`.
3. **`ollama create X -f -` (stdin) fails.** Symptom: `Error: no Modelfile or
   safetensors files found`. Root cause: the `-f -` form does not read a
   Modelfile from stdin in v0.32.5 (despite the name); a real file path is
   required. Fix: `serve_qwen3_ollama.sh` writes a one-line `FROM qwen3:1.7b`
   Modelfile to a temp file and passes that path to `ollama create`.

## Knobs (env, all overridable)

### `serve_qwen3_ollama.sh`
`DEPLOY_DIR` (default `./run`), `OLLAMA_BIN` (default
`$DEPLOY_DIR/ollama/bin/ollama`), `OLLAMA_MODELS` (default
`$DEPLOY_DIR/models`), `SERVE_PORT` (11434), `OLLAMA_HOST`
(`127.0.0.1:$SERVE_PORT`), `LOGFILE`, `PIDFILE`, `LAST_SERVICE_ENV`,
`BASE_TAG` (`qwen3:1.7b`), `SERVED_MODEL_NAME` (`ollama/qwen3:1.7b`),
`HEALTH_TIMEOUT` (120).

### `register_qwen3_otela.sh`
`DEPLOY_DIR` (default `./run`), `OTELA_DIR` (default `$DEPLOY_DIR/otela`),
`OTELA_BIN` (default `$OTELA_DIR/otela`), `OPENTELA_CFG_DIR`, `PIDFILE`,
`LOGFILE`, `OPENTELA_BOOTSTRAP` (the ocf-1 head multiaddr),
`SERVE_PORT` (11434), `SERVED_MODEL_ID` (`ollama/qwen3:1.7b`),
`OPENTELA_SERVICE_NAME` (`llm`), `OPENTELA_SEED` (default `0`; vestigial in
v0.2.3 — peer ID comes from `$CFG_DIR/keys/id`, not `--seed`; see the
sibling `qwen36-35b-a3b` README), `HEALTH_PATH` (`/`), `OPENTELA_TCP_PORT`
(43905), `OPENTELA_UDP_PORT` (59820).

| Flag | Why |
|------|-----|
| `--service.health_path /` | Ollama has no `/health` (404); `/` returns 200. Without this the health check never passes and registration hangs. |
| `--role worker` | Publishes the `llm` service (a head would only relay). |
| `--solana.skip_verification` | Testing without on-chain token verification. |
| `--bootstrap.static` | The ocf-1 head peer; without it the sidecar has no one to gossip to. |

## OpenTela connection

The service is registered on the OpenTela network as a sidecar peer (Ollama
runs unsupervised as a host process; a standalone `otela` process advertises
its `llm` service).

- **Binary:** `./run/otela/otela` — OpenTela v0.2.3 (arm64, released as
  `opentela-arm64`)
- **Local peer:** stable across restarts (peer ID from `run/otela/keys/id`;
  see the sibling `qwen36-35b-a3b` README for the v0.2.3 peer-ID-from-keys
  behavior). Observed during validation:
  `QmdtGkzQn5cCZ6CEW33mw6wto4pKfzTwbwD9BkPQG1qzeA`, libp2p :43905.
- **Bootstrap:** `/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7`
- **Gateway:** `http://140.238.223.116:8092`

> **Sidecar vs supervised:** this topology does NOT supervise the engine. If
> Ollama restarts on a different port, re-run `register_qwen3_otela.sh daemon`
> with the new port (or set `SERVE_PORT=...`).
