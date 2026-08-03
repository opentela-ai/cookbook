# Register a non-LLM HTTP service

OpenTela can advertise a service it did **not** start. The `--subprocess` flag
only launches and supervises a child process — it is **not** required to
register a service. Use this when the process is already managed by systemd,
Docker, Kubernetes, Slurm, or another terminal.

Canonical source: <https://opentela.ai/docs/tutorial/register-service>.

## Requirements

The service must:

1. run on the **same machine** as the OpenTela worker;
2. listen on a **known local port**; and
3. answer **2xx** on a health path (default `GET /health`; override with
   `--service.health_path`).

Use a service name **other than `llm`.** The name `llm` has special
registration behavior (auto `model=<id>` identity groups from `/v1/models`);
every other name makes you configure identity groups yourself.

## Configure

```yaml
# ~/.config/opentela/cfg.yaml
bootstrap:
  addr: "/ip4/<HEAD_IP>/tcp/43905/p2p/<HEAD_PEER_ID>"

service:
  name: "image-resizer"
  port: "9000"
  health_path: "/health"
  identity_group:
    - "format=webp"
```

Start OpenTela **without** `--subprocess`. CLI flags override `service.name`/
`service.port` while keeping `identity_group` from the file:

```bash
./otela start --config ~/.config/opentela/cfg.yaml \
  --service.name image-resizer --service.port 9000 --seed 1
```

## Verify and call

```bash
curl http://<HEAD_IP>:8092/v1/dnt/table     # worker shows name="image-resizer", identity_group=["format=webp"]
```

If the service is missing, check that `service.health_path` returns 2xx —
OpenTela waits and retries until it does, up to the startup retry limit.

```bash
curl --request POST --header 'Content-Type: application/json' \
  --data '{"format":"webp","source":"https://example.com/image.png"}' \
  "http://<HEAD_IP>:8092/v1/service/image-resizer/resize"
```

`format=webp` matches the top-level `"format": "webp"`. The head selects a
worker with an exact service + identity-group match and proxies `/resize` to
`http://localhost:9000/resize` on that worker.

## Identity-group values

| Config | Matches | Needs `X-Otela-Fallback`? |
|---|---|---|
| `format=webp` | body has `"format": "webp"` | no (default exact) |
| `format=*` | body has any `format` value | `1` or `2` |
| `all` | any request | `2` |

`all` suits a service whose bodies don't share a routing field (mixed `GET`/
`DELETE`/stateful follow-ups) — callers then send `X-Otela-Fallback: 2`.

## Gotcha: one provider per stateful service

Registering two **unrelated** coordinators under the same service name can
route a follow-up request to a coordinator that doesn't own that stateful
resource (e.g. a sandbox created on cluster A being addressed on cluster B).
Give each independently-stateful cluster its **own** service name, or keep
one OpenTela provider per cluster. The OpenTela startup health check does not
send an upstream API key, so the configured health path must stay reachable
locally to the worker.
