# OpenTela routing

How the head node turns a request into a forwarding decision. Canonical source:
<https://opentela.ai/docs/tutorial/routing>.

## URL prefixes

| Prefix | Handler | Purpose |
|---|---|---|
| `/v1/service/:service/*path` | Global Service Forward | by **service name + identity group** — the usual end-user path |
| `/v1/p2p/:peerId/*path` | P2P Forward | to a **specific peer**, bypassing identity groups |
| `/v1/p2p-service/:peerId/:service/*path` | service-aware P2P | to a **specific peer + exact service name** (prefer over `/v1/p2p`) |
| `/v1/_service/:service/*path` | Local Service Forward | to a **locally running** process — internal, used by the 2-hop chain |
| `/v1/regions/:region/service/:service/*path` | Trusted Region Forward | through the **trusted partition** |
| `/v1/regions/:region/p2p-service/:peerId/:service/*path` | trusted direct P2P | one **trusted peer + exact service** |
| `/v1/_regions/:region/service/:service/*path` | trusted worker destination | internal worker route for trusted traffic |

All accept `GET, POST, PATCH, DELETE`. The response carries
`X-Computing-Node: <serving peer ID>`.

## Identity groups

A `key=value` label on a worker's service. The router reads a top-level JSON
field named `key` from the request body and compares its value.

- For `llm` services, OpenTela creates one `model=<id>` entry per model from
  `/v1/models` **automatically**.
- For any other service name, set `service.identity_group` yourself in
  `cfg.yaml` (see register-service.md).
- A service with an **empty** identity group is visible in the node table but
  **never** selected for `/v1/service/...` — give every worker ≥1 entry.

## Match tiers and `X-Otela-Fallback`

| Tier | Match | Example |
|---|---|---|
| 1 (highest) | exact `key=value` | `model=Qwen/Qwen3-8B` |
| 2 | wildcard `key=*` | `model=*` |
| 3 (lowest) | catch-all `all` | `all` |

| `X-Otela-Fallback` | Tiers considered | If nothing matches at the top tier |
|---|---|---|
| not set / `0` | exact only | **503** — no fallback |
| `1` | exact → wildcard | falls to `key=*` providers |
| `2` | exact → wildcard → catch-all | falls through all three |

A provider with several entries lands in its **best** tier
(`["model=Qwen/Qwen3-8B","all"]` is exact, not catch-all). Within a tier,
candidates are picked uniformly at random. `400` = no providers for the
service name at all; `503` = providers exist but none matched.

## Forwarding topology

**Direct** (worker reachable from head): two hops —
`User → Head (/v1/service/…) → Worker (/v1/_service/…) → local engine:port`.
The head selects a candidate by identity group, then opens a libp2p stream to
the worker, which proxies to `localhost:<service.port>`.

**Relay-hop** (worker behind a firewall): the worker reserves a relay v2 slot
at startup and stores the relay's peer ID in its `relay_peer` CRDT field. When
the head has no direct libp2p connection to the worker, it routes through that
relay automatically (~30 ms). Workers only need:

```yaml
bootstrap:
  sources:
    - "https://bootstraps.opentela.ai/v1/dnt/bootstraps"
```

Relays register with `role: relay` + `public-addr` and appear in the public
bootstrap list. The whole relay hop is transparent — same URL, same body.

## Trusted vs permissionless

Two logical partitions over the same mesh:

- `permissionless` — the public net. Any peer may join and advertise a service
  name; service names and identity groups here are **routing hints, not proof
  of provider identity.**
- `trusted_region` — a named, control-plane-managed set of claimed peer IDs.
  Membership requires an allowed `head|worker|combined` role, fresh ownership,
  and exact service binding; revalidated again at the worker. Trusted traffic
  is direct-only (no application relay) and never falls back to permissionless.

A caller pins a minimum trust level with `X-Otela-Trust: <n>` (0=any, default;
1=self-attested; 2=user-trusted). No qualifying peer → `503`, never a silent
drop to untrusted.
