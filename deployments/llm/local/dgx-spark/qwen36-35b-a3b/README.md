# Qwen3.6-35B-A3B-FP8 on a DGX Spark (NVIDIA GB10) → OpenTela

Serves `Qwen/Qwen3.6-35B-A3B-FP8` (hybrid GDN MoE, FP8, 256 experts / 8 active,
40 layers of mixed linear + full attention, MTP head, VLM) on a **DGX Spark** —
a single-node workstation with one NVIDIA GB10 GPU (sm_121, aarch64, 122 GB
unified memory) — inside a thin Docker overlay on the golden GB10 SGLang image,
and registers it on OpenTela via a standalone `otela` sidecar. There is no
scheduler: the three scripts are run by hand on the host.

The recipe is self-contained: all runtime state (logs, `last_service.env`,
`otela` config/pid) lives under `./run/` next to the scripts, and the build
sources sit as sibling checkouts (`./sglang-src`, `./entrypoint-src`). Nothing
is pinned to a specific home directory or host layout — every path is an
overridable env var with a portable default.

## Why this site

A single GB10 in a workstation form factor is the whole point — no cluster, no
Slurm, no relay. The trade-off is that **GB10/aarch64 has no CUDA torch wheel
on PyPI, no flashinfer wheel, and no prebuilt sgl-kernel**: everything heavy is
inherited from a golden image built on a sibling DGX Spark (ds5), and this
recipe is only the pure-Python overlay + run scripts on top.

## What makes this non-trivial

1. **GB10/aarch64** has no CUDA torch wheel on PyPI, no flashinfer wheel, and no
   prebuilt sgl-kernel — everything builds on the NGC base.
2. **Qwen3.6** is a bleeding-edge hybrid GDN (Gated DeltaNet) model requiring
   SGLang ≥0.5.10, the `triton` attention backend (torch_native is rejected on
   Blackwell for hybrid GDN), and FP8 blockwise GEMM JIT kernels.
3. The **golden GB10 image** (built on ds5) ships sgl-kernel 0.4.4; the
   vendored sglang pins 0.4.5 — bypassed with
   `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`.

## Prerequisites

All relative to the recipe directory; every one is overridable via env.

| What | Default | How it is obtained |
|------|---------|--------------------|
| Golden GB10 SGLang image | `sglang-golden-gb10:latest` | Built on a sibling DGX Spark (ds5), not in any registry. `build_image.sh` prints the exact `docker save \| ssh …` transfer if it is missing. |
| Vendored sglang source | `./sglang-src` | A local-only checkout (release/v0.5.16 + latest cherry-picks, **no git remote** on the bring-up host). Must contain `python/pyproject.toml`. Point `SGLANG_SRC` at an existing checkout to reuse it. |
| s3er entrypoint source | `./entrypoint-src` | A checkout with `pyproject.toml` + `s3er/`. Point `ENTRYPOINT_SRC` at an existing checkout to reuse it. |
| Model weights | `$HOME/models/Qwen3.6-35B-A3B-FP8` | FP8 shard tree (`layers-N.safetensors`, `config.json`, `chat_template.jinja`, …). Override `MODEL`. |
| `otela` binary | `./run/otela/otela` | OpenTela `otela`, **v0.2.3** (commit e9d1696, official `opentela-arm64` release), **arm64**. `register_qwen36_otela.sh` prints the path if missing; set `OTELA_BIN` to use one elsewhere. |

## Architecture

```
sglang-golden-gb10  (from ds5: NGC torch 2.10 sm_121 + sgl-kernel + snapshot)
  └─ s3er-qwen36-dgx-spark  (overlay: updated vendored sglang + s3er CLI + CUTLASS fix)
        │
        │  --network host, listens on :30000
        ▼
   otela sidecar (host process)  :43905 libp2p  ──►  bootstrap peer (remote head)
```

The overlay reinstalls sglang from the current vendored source (documented as
release/v0.5.16 + latest cherry-picks; imports in-container as `0.0.0.dev0`),
adds the `s3er` entrypoint CLI, and symlinks NGC-bundled CUTLASS headers into
the flashinfer stub's data directory so JIT-compiled FP8 GEMM kernels can find
`cutlass/util/packed_stride.hpp`. The sidecar topology means the engine is
**not supervised** by `otela` — see [OpenTela connection](#opentela-connection).

## Memory budget (122 GB unified)

Documented during the original bring-up (not re-measured for this port):

| Component          | Size    |
|--------------------|---------|
| FP8 weights        | 34.7 GB |
| Mamba/SSM state    | 28.0 GB |
| Conv state         | 0.7 GB  |
| KV cache (bf16)    | 31.9 GB |
| **Total**          | ~95 GB  |

KV cache supports 1,670,348 tokens. `max_model_len` = 262,144.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.overlay` | Overlay on `sglang-golden-gb10`: reinstalls vendored sglang + `s3er` CLI, fixes CUTLASS headers |
| `build_image.sh` | Host-side: preflight golden image, stage the two source tarballs in a temp context, `docker build` the overlay |
| `serve_qwen36_dgx_spark.sh` | Start/stop the engine container; waits on `/v1/models`, writes `last_service.env` |
| `register_qwen36_otela.sh` | Start/stop/status the standalone `otela` sidecar that publishes the `llm` service |

## Build

```bash
# Stage the two vendored source checkouts next to the recipe (one-time):
#   git clone <your sglang (release/v0.5.16 + cherry-picks)>  ./sglang-src
#   git clone <your s3er entrypoint>                         ./entrypoint-src
# (On the bring-up host these were local-only, with no git remote.)

# After the golden GB10 image has been loaded from ds5:
bash build_image.sh
```

`build_image.sh` archives `./sglang-src` and `./entrypoint-src` into a temp
build context (only the Dockerfile + the two tarballs, so the large checkouts
are never sent to the Docker daemon), runs
`docker build -t s3er-qwen36-dgx-spark -f Dockerfile.overlay <ctx>`, then
removes the context — exactly as the original bring-up did. The build is
pure-Python (~30 s) since all GPU artifacts come from the base image.

## Run

```bash
# 1. start the engine (waits for readiness, ~several min for a cold FP8 load)
bash serve_qwen36_dgx_spark.sh

# 2. register on OpenTela as a background sidecar
bash register_qwen36_otela.sh daemon

# direct inference (engine is on host :30000):
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.6-35B-A3B-FP8","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'

# tear down
bash register_qwen36_otela.sh stop
bash serve_qwen36_dgx_spark.sh stop
```

## Verify

Verified on a DGX Spark during the original bring-up (the `qwen36-dgx-spark`
container stayed up for days; a routed request landed at 2026-08-02 18:56).
Re-check on a fresh start:

```bash
# engine is serving
curl -s http://localhost:30000/v1/models | python3 -m json.tool

# otela sidecar joined the mesh (look for "connected: true" and no LEFT line)
tail -f ./run/otela/otela.log
bash register_qwen36_otela.sh status
```

The peer appears on the gateway status page at
`http://140.238.223.116:8092/v1/dnt/table` as `connected: true` with
`service: llm` and `identity_group: ["model=Qwen/Qwen3.6-35B-A3B-FP8"]`. A routed
request through the gateway returns 200:

```bash
curl -s http://140.238.223.116:8092/v1/service/llm/v1/chat/completions \
  -H 'Authorization: Bearer test-token' -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.6-35B-A3B-FP8","messages":[{"role":"user","content":"hi"}]}'
```

## Knobs (env, all overridable)

### `build_image.sh`
`SGLANG_SRC` (default `./sglang-src`), `ENTRYPOINT_SRC` (default
`./entrypoint-src`), `BASE_IMAGE` (default `sglang-golden-gb10:latest`),
`IMAGE` (default `s3er-qwen36-dgx-spark`).

### `serve_qwen36_dgx_spark.sh`
`DEPLOY_DIR` (default `./run`), `MODEL` (default
`$HOME/models/Qwen3.6-35B-A3B-FP8`), `IMAGE`, `SERVE_PORT` (30000), `CONTAINER`
(`qwen36-dgx-spark`), `TP_SIZE` (1), `ATTENTION_BACKEND` (`triton`),
`MEM_FRACTION_STATIC` (0.85), `REASONING_PARSER` (`qwen3`), `TOOL_CALL_PARSER`
(`qwen3_coder`), `SERVED_MODEL_NAME` (`Qwen/Qwen3.6-35B-A3B-FP8`),
`HEALTH_TIMEOUT` (600), `LAST_SERVICE_ENV`.

| Flag | Why |
|------|-----|
| `--attention-backend triton` | Hybrid GDN on Blackwell rejects torch_native |
| `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` | Golden image has sgl-kernel 0.4.4; vendored sglang wants ≥0.4.5 |
| `--mem-fraction-static 0.85` | Leaves room for SSM state + CUDA graph capture |
| `--reasoning-parser qwen3` | Model is a reasoning model (`<think>` tags) |
| `--tool-call-parser qwen3_coder` | Without it the engine registers no tool-call parser, so the gateway can't serve function calls for this model |
| `--network host` | The otela sidecar (separate host proc) reaches the engine at 127.0.0.1:30000 |
| `--privileged` | Golden image CRIU checkpoint tooling |
| `--ulimit stack=67108864` | sglang hybrid-GDN call stack overflows the default 8 MB |
| `--shm-size 16g` | Single-node tensor/NCCL shared memory |
| `HF_HUB_OFFLINE=1` | Weights are local; never call HuggingFace hub |

### `register_qwen36_otela.sh`
`DEPLOY_DIR` (default `./run`), `OTELA_DIR` (default `$DEPLOY_DIR/otela`),
`OTELA_BIN` (default `$OTELA_DIR/otela`), `OPENTELA_CFG_DIR`, `PIDFILE`,
`LOGFILE`, `OPENTELA_BOOTSTRAP`, `SERVE_PORT` (30000), `SERVED_MODEL_ID`
  (`Qwen/Qwen3.6-35B-A3B-FP8`), `OPENTELA_SERVICE_NAME` (`llm`), `OPENTELA_SEED`
  (default `0`; vestigial in v0.2.3 — peer ID comes from `$CFG_DIR/keys/id`,
  not `--seed`; see [Site-specific fixes](#site-specific-fixes)),
`OPENTELA_TCP_PORT` (43905), `OPENTELA_UDP_PORT` (59820).

## Site-specific fixes

1. **Re-registration after a stop/start cycle.** A graceful
   `register_qwen36_otela.sh stop` sends SIGTERM, and otela announces LEFT to
   the mesh (a CRDT tombstone). In v0.2.3 the libp2p peer ID comes from
   `$CFG_DIR/keys/id` (created by `otela init --config-dir` if absent), NOT from
   `--seed` (verified on ds6: `--seed 0`, `1`, and `2` all yield the same peer
   ID with the same keys; deleting `$CFG_DIR/keys` and re-initing yields a
   different one). The Solana provider key at `~/.config/opentela/` is separate
   and not affected by `--config-dir`. Restarting with the same keys reproduces
   the peer ID and re-registers in ~36 s (verified:
   `QmfXLJZRsMJq1BT9ZeHq5S2Ea8gbhH7NT9SEDeCxLcwcof` returned 200 through the
   gateway ~36 s after restart) — a brief 503 window is normal while the LEFT
   tombstone is overridden. The sai-v0.0.6 binary this replaces could not
   override the tombstone and stalled at 503 for many minutes; that is no
   longer the case. To force a fresh peer ID (e.g. if the current peer is
   permanently tombstoned): `rm -rf run/otela/keys` and re-run
   `register_qwen36_otela.sh daemon` — the `init` step creates new keys and a
   brand-new peer registers in ~21 s (verified)

## OpenTela connection

The service is registered on the OpenTela network as a sidecar peer (the
engine itself runs unsupervised in the container; a standalone `otela` process
on the DGX Spark host advertises its `llm` service).

- **Binary:** `./run/otela/otela` — OpenTela `otela`, v0.2.3 (arm64)
- **Local peer:** stable across restarts (peer ID from `run/otela/keys/id`;
  see [Site-specific fixes](#site-specific-fixes)). Observed during validation:
  `QmfXLJZRsMJq1BT9ZeHq5S2Ea8gbhH7NT9SEDeCxLcwcof`, libp2p :43905.
- **Bootstrap:** `/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7`
- **Gateway:** `http://140.238.223.116:8092`

> **Sidecar vs supervised:** this topology does NOT supervise the engine. If
> the engine restarts on a different port, re-run `register_qwen36_otela.sh
> daemon` with the new port (or set `SERVE_PORT=...`). For lifecycle-managed
> supervision, rebuild the overlay with `otela` and use
> `s3er serve --enable-opentela`.
