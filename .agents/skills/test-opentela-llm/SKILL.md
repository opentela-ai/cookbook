---
name: test-opentela-llm
description: Smoke-test or query an LLM served through OpenTela — the public api.opentela.ai/v1/service/llm/v1 gateway or a local mesh head (e.g. Alps/Beverin). Use when verifying a deployment actually serves requests, debugging registration/routing, writing client code against the OpenAI-compatible endpoint (streaming, reasoning_content), or mapping a serving failure to its documented cause.
---

# Test an OpenTela LLM endpoint

## Public gateway (api.opentela.ai)

OpenAI-compatible API at `https://api.opentela.ai/v1/service/llm/v1`, auth via
the `OTELA_API` env var as a bearer token.

```bash
# list models the public head currently routes to
curl -s https://api.opentela.ai/v1/service/llm/v1/models \
  -H "Authorization: Bearer $OTELA_API" | python3 -m json.tool

# minimal chat request (non-streaming proves routing end-to-end)
curl -s https://api.opentela.ai/v1/service/llm/v1/chat/completions \
  -H "Authorization: Bearer $OTELA_API" -H "Content-Type: application/json" \
  -d '{"model": "<served-model-name>", "messages": [{"role": "user", "content": "ping"}]}' \
  | python3 -m json.tool
```

A local (gitignored, untracked) reference client `test_llm.py` shows the streaming pattern (prints
`reasoning_content` separately from `content` — reasoning models emit both,
and naive clients silently drop or jumble them).

## Local mesh head (e.g. Alpine mesh behind Beverin)

No API key; route to a specific served model with the `X-Otela-Model` header
from any peer that can reach the local head:

```bash
curl -s http://<mesh-head>/v1/service/llm/v1/models -H "X-Otela-Model: <served-model-name>"
```

## Engine health from inside the allocation

Compute-node ports are often unreachable from login nodes, so health checks
run **inside** the Slurm allocation (pattern from the Beverin README):

```bash
srun -p <partition> -A <account> -N1 -n1 --time=00:02:00 --overlap \
  bash -lc 'curl -s http://<head-node>:<port>/get_model_info | python3 -m json.tool'
```

Job env facts (head node/IP/port) are written by the recipes to
`$DEPLOY_DIR/last_service.env`.

## What "healthy" looks like in the logs

A working deployment shows, in order: otela announcing its Peer ID, the
engine ready (`sglang_ready` / `/health` answering), the worker registering
the `llm` service, and `200 OK` on `POST /v1/chat/completions` in the serving
log (= real requests being routed in from the mesh — the decisive proof).

## Symptom -> documented cause

| Symptom | Cause | Source of truth |
|---|---|---|
| Public API returns 503 "No provider found" | relay data dir on NFS went stale (ESTALE) -> badger value log unwritable -> CRDT frozen; fix: symlink relay's `~/.ocfcore` to scratch | `deployments/llm/jsc/serve_llm_otela_jsc.sbatch` header |
| Job registers but never serves a request | relay predates the P2PForwardHandler fix (needs official >= v0.2.3) | jsc sbatch header |
| Self-built otela relays/serves but never registers | peers discard records from unattested ("dev") binaries; use a signed release | jsc sbatch, `OTELA_REQUIRE_SIGNED` |
| Peer stuck `connected: true` after job ended | otela was SIGKILLed; always TERM the srun **step** and wait for the leave announcement | jsc sbatch, `stop_otela()` |
| Registry never propagates the service at all | same relay ESTALE/CRDT freeze as the 503 above | jsc sbatch header |
| OOM SIGKILL ~35 s after "Load weight end", no traceback (MI300A) | APU `is_integrated=True` makes sglang budget KV against whole-node RAM; launcher forces `is_integrated=False` | `deployments/llm/beverin-glm47-flash/README.md` fix #5 |
| `ValueError: CPU number N is not eligible` (MI300A) | container cpuset smaller than host CPU ids used for NUMA pinning; pass full node CPUs to the srun step | beverin README fix #1 |
| `KeyError` in aiter `get_rope` (GLM on ROCm) | model config lacks `rope_scaling` key; `SGLANG_USE_AITER=0` | beverin README fix #3 |

When a new serving failure is root-caused, add its row here only after the
fix is baked into the recipe and the verbatim error is recorded — the recipes
remain the source of truth; this table is a signpost, not a replacement.
