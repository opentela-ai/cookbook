# Qwen3.8-27B-FP8 on a DGX Spark (NVIDIA GB10) → OpenTela

Serves `Qwen/Qwen3.8-27B-FP8` — a **dense 27B** hybrid GDN (Gated DeltaNet +
full attention + MTP) vision-language model (`qwen3_5` / `Qwen3_5ForConditionalGeneration`,
hidden 5120, 64 layers, 24 attn & 4 KV heads, vocab 248320, max 262144, FP8
e4m3/dynamic) — on the **same DGX Spark** that ran `Qwen/Qwen3.6-35B-A3B-FP8`
(single NVIDIA GB10, sm_121, aarch64, 122 GB unified memory), **reusing the
`s3er-qwen36-dgx-spark` image with no rebuild**, and registers it on OpenTela
via a standalone `otela` sidecar. There is no scheduler: the two scripts are
run by hand on the host.

## Why this site (and no `build_image.sh`)

A single GB10 in a workstation form factor is the whole point — no cluster, no
Slurm, no relay. The trade-off (from the qwen36 recipe) is that GB10/aarch64 has
no CUDA torch wheel on PyPI, no flashinfer wheel, and no prebuilt sgl-kernel;
everything heavy lives in a golden image built on a sibling DGX Spark (ds5).

**Qwen3.8-27B-FP8 is the same `qwen3_5` hybrid-GDN family as Qwen3.6-35B-A3B-FP8**,
and the existing `s3er-qwen36-dgx-spark` image already contains
`sglang.srt.models.qwen3_5` (verified by importing it in the image: both
`s3er-qwen36-dgx-spark` and `sglang-golden-gb10` report `sglang.__version__
0.0.0.dev0` / vendored release/v0.5.16 and `has model module:
sglang.srt.models.qwen3_5`). So this recipe is **serve + register scripts only**
— no `Dockerfile.overlay`, no `build_image.sh`. If a future sglang bump is
needed for Qwen3.8, rebuild via the sibling `qwen36-35b-a3b` recipe (same base
+ vendored sglang) and point `IMAGE` here at the new tag.

## What changed from the qwen36 recipe

| Aspect | qwen36 (`Qwen3.6-35B-A3B-FP8`) | qwen38 (this recipe) |
|---|---|---|
| Model | 35B MoE (256 experts / 8 active) | **dense 27B** (no experts) |
| Arch family | `qwen3_5` hybrid GDN | `qwen3_5` hybrid GDN (same) |
| Modality | text | **vision-language** (vision_config + text_config) |
| `--mem-fraction-static` | 0.85 (tuned for ~29 GB MoE SSM/conv state) | **0.80 (verified on ds6)** — 29.1 GB weights + 28.5 GB SSM/conv, 17.6 GB headroom remaining |
| `--attention-backend triton` | required (hybrid GDN on Blackwell) | **kept, verified** (same family) |
| `--reasoning-parser qwen3` | required (model emits a thinking trace) | **kept, verified on ds6** (reasoning_content populated, reasoning_tokens=213) |
| `--tool-call-parser qwen3_coder` | required (else no tool parser registered) | **kept, verified on ds6** (tool_calls `get_weather({"city":"Paris"})` returned) |
| `--ulimit stack=67108864` | required (hybrid-GDN call stack) | **kept** (same family) |
| Image | `s3er-qwen36-dgx-spark` (built here) | **reuses** `s3er-qwen36-dgx-spark` (no build) |
| Cold start | (several min) | **~7 min verified** (weights 182.6 s + prefill CUDA graph 228.6 s) |

## Weights

The FP8 shard tree (`layers-N.safetensors`, `config.json`,
`chat_template.jinja`, `generation_config.json`, `tokenizer*.json`,
`model.safetensors.index.json`) is downloaded **out of band** to
`$HOME/models/Qwen3.8-27B-FP8` (default; override `MODEL`). The ds6 host has no
`huggingface-cli`/`git-lfs` and no `huggingface_hub` module, but the
`sglang-golden-gb10` image ships the `hf` CLI (huggingface_hub 1.22.0); that
image bakes in `HF_HUB_OFFLINE=1`, so the download must unset it:

```bash
# from a login/interactive shell on ds6 (host has egress to HF, 3.4 TB free)
docker run --rm --network host -v "$HOME/models":/models \
    -e HF_HUB_OFFLINE=0 --entrypoint /bin/bash \
    sglang-golden-gb10 -c 'hf download Qwen/Qwen3.8-27B-FP8 --local-dir /models/Qwen3.8-27B-FP8'
```

Verified on ds6 during this bring-up: `hf download exit=0`, 81 files,
66 `*.safetensors` shards (== `model.safetensors.index.json` weight_map),
30.87 GB of safetensors, 0 missing / 0 extra. The model is **non-gated**, so no
`HF_TOKEN` is required.

## OpenTela connection (sidecar topology)

The service is registered on the OpenTela network as a sidecar peer (the
engine itself runs unsupervised in the container; a standalone `otela` process
on the DGX Spark host advertises its `llm` service). Same topology and same
otela binary/version as the qwen36 recipe:

- **Binary:** `./run/otela/otela` — OpenTela `otela`, v0.2.3 (arm64). On this
  host, copy from the sibling recipe:
  `cp ../qwen36-35b-a3b/run/otela/otela run/otela/otela`
  (or `OTELA_BIN=../qwen36-35b-a3b/run/otela/otela`).
- **Bootstrap:** `/ip4/140.238.223.116/tcp/43905/p2p/QmTtnXKHvovCwkBZRR4NcxeHfnt5EJQgN4wo9KV8U8nYP7`
- **Gateway:** `http://140.238.223.116:8092`
- **Peer (this bring-up):** `QmRxxqovNmQiHxaB6fM1DA57J1fk3LumeAxBsGaPuwkjkU`,
  libp2p :43905, `bootstrap_connected=true` ~14 s after `otela start`.

> **Sidecar vs supervised:** this topology does NOT supervise the engine. If
> the engine restarts on a different port, re-run `register_qwen38_otela.sh
> daemon` with the new port (or set `SERVE_PORT=...`). For lifecycle-managed
> supervision, rebuild the overlay with `otela` and use `s3er serve
> --enable-opentela`.

## Files

| File | Purpose |
|------|---------|
| `serve_qwen38_dgx_spark.sh` | Start/stop the engine container (reuses `s3er-qwen36-dgx-spark`); waits on `/v1/models`, writes `last_service.env` |
| `register_qwen38_otela.sh` | Start/stop/status the standalone `otela` sidecar that publishes the `llm` service |
| `.gitignore` | Excludes `run/` (runtime state) and the out-of-band weights |

## Run

```bash
# 0. (once) stage the otela binary next to this recipe
mkdir -p run/otela && cp ../qwen36-35b-a3b/run/otela/otela run/otela/otela

# 1. start the engine (waits for readiness, ~7 min for a cold FP8 load)
bash serve_qwen38_dgx_spark.sh

# 2. register on OpenTela as a background sidecar
bash register_qwen38_otela.sh daemon

# direct inference (engine is on host :30000):
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.8-27B-FP8","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'

# tear down
bash register_qwen38_otela.sh stop
bash serve_qwen38_dgx_spark.sh stop
```

## Verify

Verified on ds6 on 2026-08-14 (the commands below were executed end to end).
Look for the `READY!` line from the serve script, `Server started ...
bootstrap_connected=true` (and no `LEFT`/`leaving` line) in the otela log, and
a 200 with the correct answer from a routed request through the gateway.

```bash
# engine is serving the new model id (max_model_len 262144)
curl -s http://localhost:30000/v1/models | python3 -m json.tool

# otela sidecar joined the mesh (look for bootstrap_connected=true and no LEFT line)
tail -f ./run/otela/otela.log
bash register_qwen38_otela.sh status

# reasoning_content is populated (verified: reasoning_tokens=213 on a "think briefly" prompt)
curl -s http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.8-27B-FP8","messages":[{"role":"user","content":"Think briefly, then reply with exactly one word: the capital of France is"}],"max_tokens":256}' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);c=d["choices"][0]["message"];print("content:",repr(c.get("content"))[:200]);print("reasoning_content:",repr(c.get("reasoning_content"))[:200])'

# tool-call parser (verified: returned tool_calls get_weather({"city":"Paris"}))
curl -s http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.8-27B-FP8","messages":[{"role":"user","content":"What is the weather in Paris? Call the get_weather function."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":512}' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);c=d["choices"][0]["message"];print("tool_calls:",c.get("tool_calls"))'

# routed through the gateway (the peer's identity_group is model=Qwen/Qwen3.8-27B-FP8)
curl -s http://140.238.223.116:8092/v1/service/llm/v1/chat/completions \
  -H 'Authorization: Bearer test-token' -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.8-27B-FP8","messages":[{"role":"user","content":"Reply with exactly one word: the capital of Japan is"}],"max_tokens":128}'
```

Results obtained during this bring-up: `/v1/models` returned
`id=Qwen/Qwen3.8-27B-FP8` with `max_model_len=262144`; the direct reasoning
prompt returned `content="\n\nParis"` with `reasoning_tokens=213`; the tool-call
prompt returned `tool_calls=[{function:{name:"get_weather",
arguments:'{"city":"Paris"}'}}]`; the routed gateway request returned
`content="\n\nTokyo"` with `model=Qwen/Qwen3.8-27B-FP8`. The gateway table at
`http://140.238.223.116:8092/v1/dnt/table` shows peer
`QmRxxqovNmQiHxaB6fM1DA57J1fk3LumeAxBsGaPuwkjkU` with
`service:[{name:"llm",status:"connected",port:"30000",identity_group:["model=Qwen/Qwen3.8-27B-FP8"]}]`
and `gpus:[{name:"NVIDIA GB10"}]`; no `Qwen3.6` peer remains and there are 0
`LEFT` lines.

## Site-specific fixes

All fixes from the sibling `qwen36-35b-a3b` README carry over unchanged (same
host, same otela, same topology); they are not duplicated here:

1. **Re-registration after a stop/start cycle** — graceful `stop` sends SIGTERM,
   otela announces LEFT (a CRDT tombstone), and restarting with the same
   `$CFG_DIR/keys/id` reproduces the peer ID and re-registers in ~36 s
   (verified on ds6 for qwen36). v0.2.3 derives the peer ID from
   `$CFG_DIR/keys/id`, NOT `--seed`. To force a fresh peer ID: `rm -rf
   run/otela/keys` and re-run `register_qwen38_otela.sh daemon`.

### New for this recipe (recorded as hit during bring-up)

- **`HF_HUB_OFFLINE=1` is baked into the serving images.** `hf download` run in
  `sglang-golden-gb10` immediately fails with
  `Local entry not found. Cannot reach ... : offline mode is enabled. To
  disable it, please unset the HF_HUB_OFFLINE environment variable.` Fix: pass
  `-e HF_HUB_OFFLINE=0` (and `TRANSFORMERS_OFFLINE=0`) to the download
  container; the serve recipe still sets `HF_HUB_OFFLINE=1` (weights are local).
  (This host has no `huggingface-cli`/`git-lfs`/`huggingface_hub`, so the
  container's `hf` CLI is the download path.)
- **`--mem-fraction-static 0.80` is verified, not a guess.** Cold load on ds6
  (2026-08-14): `Load weight end elapsed=182.55 s, mem usage=29.12 GB` (FP8
  e4m3); `Mamba Cache ssm_state=27.98 GB + conv_state=0.55 GB` (=28.5 GB, ~same
  as the qwen36 35B MoE — same hybrid-GDN family); prefill CUDA graph 228.6 s /
  3.31 GB; decode graph 4.7 s / 0.47 GB; final `available_gpu_mem=17.60 GB`.
  Less total than the qwen36 MoE (34.7 GB weights), so 0.80 is safe and leaves
  headroom for the multimodal (vision) path; raise toward 0.83–0.85 to grow the
  KV cache if more concurrency is needed.
- **`--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder` are
  verified for Qwen3.8** (same `qwen3_5` family as qwen36). A "think briefly"
  prompt returned populated `reasoning_content` with `reasoning_tokens=213`; a
  request with `tools=[get_weather]` returned a proper
  `tool_calls=[{function:{name:"get_weather",arguments:'{"city":"Paris"}'}}]`
  response. (qwen36 needed `qwen3_coder` else the engine registered no tool-call
  parser; that carries over to 3.8.)

## Knobs (env, all overridable)

### `serve_qwen38_dgx_spark.sh`
`DEPLOY_DIR` (default `./run`), `MODEL` (default
`$HOME/models/Qwen3.8-27B-FP8`), `IMAGE` (default `s3er-qwen36-dgx-spark`,
reused — no build), `SERVE_PORT` (30000), `CONTAINER` (`qwen38-dgx-spark`),
`TP_SIZE` (1), `ATTENTION_BACKEND` (`triton`), `MEM_FRACTION_STATIC` (`0.80`,
verified), `REASONING_PARSER` (`qwen3`), `TOOL_CALL_PARSER`
(`qwen3_coder`), `SERVED_MODEL_NAME` (`Qwen/Qwen3.8-27B-FP8`),
`HEALTH_TIMEOUT` (600), `LAST_SERVICE_ENV`.

| Flag | Why |
|------|-----|
| `--attention-backend triton` | Hybrid GDN on Blackwell rejects torch_native (same family as qwen36) |
| `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` | Reused image has sgl-kernel 0.4.4; vendored sglang wants ≥0.4.5 |
| `--mem-fraction-static 0.80` | Verified: 29.1 GB weights + 28.5 GB SSM/conv, 17.6 GB headroom remaining |
| `--reasoning-parser qwen3` | Model is a reasoning model (emits a thinking trace); verified reasoning_content + reasoning_tokens=213 |
| `--tool-call-parser qwen3_coder` | Else the engine registers no tool-call parser; verified tool_calls `get_weather({"city":"Paris"})` |
| `--network host` | The otela sidecar (separate host proc) reaches the engine at 127.0.0.1:30000 |
| `--privileged` | Golden image CRIU checkpoint tooling |
| `--ulimit stack=67108864` | sglang hybrid-GDN call stack overflows the default 8 MB |
| `--shm-size 16g` | Single-node tensor/NCCL shared memory |
| `HF_HUB_OFFLINE=1` | Weights are local; never call HuggingFace hub |

### `register_qwen38_otela.sh`
`DEPLOY_DIR` (default `./run`), `OTELA_DIR` (default `$DEPLOY_DIR/otela`),
`OTELA_BIN` (default `$OTELA_DIR/otela`), `OPENTELA_CFG_DIR`, `PIDFILE`,
`LOGFILE`, `OPENTELA_BOOTSTRAP`, `SERVE_PORT` (30000), `SERVED_MODEL_ID`
(`Qwen/Qwen3.8-27B-FP8`), `OPENTELA_SERVICE_NAME` (`llm`), `OPENTELA_SEED`
(`0`; vestigial in v0.2.3 — peer ID comes from `$CFG_DIR/keys/id`, not
`--seed`; see Site-specific fixes), `OPENTELA_TCP_PORT` (43905),
`OPENTELA_UDP_PORT` (59820).
