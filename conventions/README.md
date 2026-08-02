# Conventions

Cross-recipe rules that every deployment in this cookbook follows. Recipes are
self-contained, but these conventions are shared so a model served on one site
is addressable on OpenTela the same way it would be on any other.

## LLM served-model names

Every LLM recipe publishes its model on OpenTela under a single
**`org/model-name`** string — a HuggingFace-style org slug, a literal `/`, then
the model name. This is the model's identity on the network, **not** the path
to its weights on disk.

The same `org/model-name` string is used in all three places a recipe names the
model, and they are kept equal:

| Where | What it sets |
|-------|--------------|
| sglang `--served-model-name` | The model id reported on the engine's `/v1/models`, and the value the engine accepts in the request `model` field (for direct `localhost:<port>` calls). |
| otela `--label model=<org/model-name>` | Becomes the peer's `identity_group` on the gateway; the gateway routes a client's `model` field to a peer whose `identity_group` contains it. |
| Client request body `"model": "<org/model-name>"` | What callers send to `/v1/service/llm/...` through the gateway. |

A request for a model that no registered peer advertises cannot be routed — the
gateway returns HTTP 503 `{"error":"No provider found for the requested
service."}`, the failure observed on the dgx-spark recipe when a peer failed to
register. Keeping the engine's `--served-model-name` on the same string keeps
direct calls and the engine's `/v1/models` listing consistent with routed calls.

The **local filesystem path** to the weights (`MODEL` / `MODEL_PATH`) is a
separate, site-specific value and is **not** required to use the `org/model-name`
form — it is wherever the shard tree happens to live on that host (some
checkouts mirror the HuggingFace `org/model-name` layout, some do not).

The recipe **directory** name (`<model>` in the layout) is the bare model name,
not the `org/model-name` served identity — e.g. `deployments/llm/jsc/kimi-k3/`
serves `moonshotai/Kimi-K3`.

### Examples in this cookbook

| Recipe | Served model name |
|--------|-------------------|
| `deployments/llm/jsc/kimi-k3/` | `moonshotai/Kimi-K3` |
| `deployments/llm/beverin/glm47-flash/` | `zai-org/GLM-4.7-Flash` |
| `deployments/llm/beverin/deepseek-v4/` | `deepseek-ai/DeepSeek-V4-Flash` |
| `deployments/local/llm/dgx-spark/qwen36-35b-a3b/` | `Qwen/Qwen3.6-35B-A3B-FP8` |
