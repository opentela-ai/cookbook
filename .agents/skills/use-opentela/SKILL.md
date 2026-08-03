---
name: use-opentela
description: Run an OpenTela cluster end to end — install the otela binary, init a wallet, start a head node and an LLM worker, register a non-LLM HTTP service, and send routed requests through /v1/service/.... Use when spinning OpenTela up for the first time, adding a node to an existing mesh, sending requests to a head node you didn't author, or choosing the right routing endpoint and X-Otela-Fallback header.
---

# Use OpenTela

OpenTela is a decentralized fabric (libp2p + CRDT) that pools GPU nodes and
routes OpenAI-compatible requests to them. This skill is the operator's
runbook for the happy path. It is **not** for authoring the Slurm sbatch
recipes in this repo (see the `write-deployment-recipe` skill) or for
smoke-testing/debugging a served LLM (see the `test-opentela-llm` skill).

Source of truth for everything below: <https://opentela.ai/docs>. Verify any
flag you're unsure about against the docs or `otela <cmd> --help` before
trusting it.

## Prereqs

```bash
otela version >/dev/null 2>&1 || {                      # or a local path: ./otela version
  echo "install otela first (step 1)"; exit 1;
}
# OTELA_API is the bearer token for the PUBLIC gateway (api.opentela.ai).
# You do NOT need it to run your own head/worker — only to call the gateway.
: "${OTELA_API:-<unset> OTELA_API not set; fine if using your own head}"
```

## 1. Install the binary

Pre-built Linux binaries (one per arch). Download the one that matches the host:

```bash
# x86_64
wget https://github.com/eth-easl/OpenTela/releases/latest/download/otela-amd64 -O otela && chmod +x otela
# arm64
wget https://github.com/eth-easl/OpenTela/releases/latest/download/otela-arm64 -O otela && chmod +x otela
otela --help          # lists: init | start | update | version | wallet
```

Build from source (needs Go): `git clone git@github.com:eth-easl/OpenTela.git`,
`cd src && make build-release` → `src/build/release/otela-{amd64,arm64}`.

## 2. Wallet & ownership

The wallet is a Solana Ed25519 keypair; a short **Provider ID** (`otela-<pubkey>`)
is derived from it and written into the `owner` field of every service the node
registers, so participants can see who runs what.

```bash
./otela init                                   # creates ~/.config/opentela/ + cfg.yaml + first wallet
./otela wallet info                            # show default wallet pubkey + Provider ID
./otela wallet list                            # * marks the default (active) wallet
```

The wallet is **loaded automatically** at `otela start` — no flag needed. Skip
it for local testing with `otela start --wallet.account ""` (logs
`Wallet account set to 'none'`). For create/export/import/transfer/airdrop,
see [references/wallet.md](references/wallet.md).

## 3. Start the head node

The head node is the public entry point. It needs **no GPU** — just a reachable
address. `--seed 0` gives a **deterministic peer ID** so workers can hard-code
it; without `--seed` the peer ID changes every start.

```bash
./otela start --mode standalone --public-addr "$HEAD_IP" --seed 0
```

From the logs, copy the `Peer ID: <Qm...>` line (base-58). The status page is
`http://$HEAD_IP:8092/v1/dnt/table` (default API port **8092**); the default
libp2p listen address is `/ip4/$HEAD_IP/tcp/43905/p2p/<PEER_ID>`.

## 4. Start a worker node (LLM)

A worker boots an LLM via `--subprocess` and advertises a `llm` service. For
`llm` services OpenTela queries the engine's `/v1/models` and registers a
`model=<id>` identity group **automatically** — keep `--service.name llm`.

```bash
# bare vLLM
./otela start \
  --bootstrap.addr /ip4/$HEAD_IP/tcp/43905/p2p/$HEAD_PEER_ID \
  --subprocess "vllm serve Qwen/Qwen3-8B --max_model_len 16384 --port 8080" \
  --service.name llm --service.port 8080 --seed 1

# Docker (no host CUDA install; needs NVIDIA Container Toolkit + --network host)
./otela start \
  --bootstrap.addr /ip4/$HEAD_IP/tcp/43905/p2p/$HEAD_PEER_ID \
  --subprocess "docker run --rm --gpus all --network host -v $HOME/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF_TOKEN lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 --host 0.0.0.0" \
  --service.name llm --service.port 30000 --seed 1
```

Two sharp edges (both bite silently):

- **`--subprocess` is split on whitespace, not a shell.** No quoting, `&&`,
  pipes, or `$VAR` expansion inside the string. For anything complex, write a
  wrapper script and pass its path (`--subprocess ./start-sglang.sh`); the
  script must be executable and on a path without spaces.
- **Multiple otela processes on one host** (one GPU each): give each its own
  libp2p ports and API port — `--tcpport 43906 --udpport 59821 --port 8093`
  for worker 1 — plus a distinct `--service.port` and `--seed`.

For registering an **already-running, non-LLM** HTTP service (no
`--subprocess`), see [references/register-service.md](references/register-service.md).

## 5. Verify the mesh

```bash
curl -s "http://$HEAD_IP:8092/v1/dnt/table" | python3 -m json.tool
```

A healthy worker entry shows `"service": [{"name":"llm","status":"connected",
"identity_group":["model=Qwen/Qwen3-8B"], "port":"8080"}]` and the head's
peer alongside it. **No provider for a model = 503 `No provider found for the
requested service.`** — the worker never registered; check its health check
(default `GET /health`, override `--service.health_path`) and that
`--service.port` matches the engine port.

## 6. Send a request

Point any OpenAI-compatible client at the head. The `model` field in the body
is what the router matches against identity groups.

```python
from openai import OpenAI
client = OpenAI(base_url=f"http://{HEAD_IP}:8092/v1/service/llm/v1", api_key="test-token")
print(client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role":"user","content":"ping"}]).choices[0].message.content)
```

- The response carries an `X-Computing-Node` header = the serving peer's ID.
- **Fallback tiers** are opt-in via the `X-Otela-Fallback` header
  (`0`=exact only/default, `1`=+wildcard, `2`=+catch-all). Without it, a
  missing exact match returns **503**, not a fallback. Full table in
  [references/routing.md](references/routing.md).
- To call the public gateway instead of your own head, base URL is
  `https://api.opentela.ai/v1/service/llm/v1` with `Authorization: Bearer
  $OTELA_API` — see the `test-opentela-llm` skill for the smoke-test matrix.

## 7. Routing at a glance

| Prefix | Routes by | When to use |
|---|---|---|
| `/v1/service/:service/*path` | service name + identity group | the normal path for end users |
| `/v1/p2p/:peerId/*path` | specific peer ID | debug, or deterministic routing |
| `/v1/p2p-service/:peerId/:service/*path` | specific peer + exact service | prefer over legacy `/v1/p2p` |
| `/v1/regions/:region/service/:service/*path` | trusted partition | when provider identity must hold |
| `/v1/_service/:service/*path` | local process on this node | internal only |

Identity-group match tiers (highest first): `key=value` (exact) → `key=*`
(wildcard) → `all` (catch-all). A worker with **no** identity group entries is
visible in the table but **never** routed to. If a worker sits behind a
firewall (e.g. HPC compute nodes), the head auto-routes through that worker's
advertised relay (~30 ms overhead) — no client change. Deep version with the
priority/fallback decision table and relay registration:
[references/routing.md](references/routing.md).

## Where to go next

- **Author a site-specific Slurm sbatch recipe** (this repo's
  `deployments/<kind>/<site>/<model>/`) → `write-deployment-recipe` skill.
- **Manage many SLURM clusters with `otela-fleet`** (pip CLI, YAML fleet file)
  → `manage-opentela-fleet` skill.
- **Smoke-test or debug a served LLM** (gateway, streaming, `reasoning_content`,
  symptom→cause) → `test-opentela-llm` skill.
- Docs: <https://opentela.ai/docs> (Tutorial, Advanced, Extensions).
