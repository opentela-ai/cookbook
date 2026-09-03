# GLM-5.3 diagnostic harness (`meta/diag/glm53`)

Shared, cross-site diagnostic tooling for the GLM-5.3-Flash serving campaign
(beverin/MI300A garbage-output bisect). One copy lives here so every site
runs the IDENTICAL code — the cross-machine bisect is only meaningful if
capture and diff behave identically on both machines. Recipes point at this
directory via `GLM53_DIAG_DIR` (exported by their sbatches).

## Components

| File | Role | Gate |
|---|---|---|
| `import_hook.py` | generic `run_after_import(target, on_loaded)` — the one import-hook primitive all engine shims share (replaces the five copy-pasted MetaPathFinder/Loader scaffolds) | — |
| `gfx942.py` | `supports_current_device()` — the MI300A device gate, probed lazily | — |
| `patch_dsa_vk.py` | rebinds sglang's DSA `tilelang_sparse_fwd` + `tilelang_fp8_paged_mqa_logits` to the vkernels HIP kernels (vkernels #51/#52) | always on gfx942 |
| `patch_topk_torch.py` | rebinds `fast_kpool_topk_transform_fused` to the torch bridge (vkernels #56) | `GLM53_TOPK_TRANSFORM_BACKEND=torch` + gfx942 |
| `fwd_probe.py` | first-forward per-op ENTER/EXIT logger + prefill argmax | `GLM53_FWD_PROBE=1` |
| `patch_dsa_sdpa.py` | DSA prefill -> PyTorch SDPA ragged loop on gfx942 | always on gfx942 |
| `comp_capture.py` | first-forward per-layer/per-component I/O dump (+ manifest) | `GLM53_COMP_CAPTURE=1` |
| `capture_probe.py` | fires ONE deterministic ~2k-token prefill so the capture latch lands on the same forward on both machines | run by the sbatch (`GLM53_CAPTURE_PROBE=1`) |
| `comp_diff.py` | THE diff tool: `layers` (cross-machine, input_ids-identity-checked), `components` (vs pure-torch ref), `summary` (manifest-only health) | operator tool |
| `live_probe.py` | sanity probes C–I against a live server (echo logprobs, determinism, retrieval, batch/length sensitivity) | operator tool |

## Wiring

- **beverin**: `build_overlay.sh` installs the dispatcher (`sitecustomize.py`)
  into `$OVL/pylib`; the sbatch exports `GLM53_DIAG_DIR`; the dispatcher
  imports the patch modules from here. The recipe dir keeps only the
  dispatcher + the engine drop-ins (`vkernels_dsa*.py`).
- **clariden**: the sbatch's heredoc `sitecustomize.py` keeps its own
  GH200 patches and imports `comp_capture` from `$GLM53_DIAG_DIR`.

## Capture -> diff workflow

```bash
# 1. submit both sites with matching comp env (same TAG family, same probe)
sbatch --export=ALL,GLM53_COMP_CAPTURE=1,GLM53_COMP_MODE=layers,\
GLM53_COMP_TAG=bisect_vN,GLM53_CAPTURE_PROBE=1 serve_*.sbatch

# 2. diff (identity check on saved input_ids runs first, automatically)
python3 meta/diag/glm53/comp_diff.py layers \
  /capstor/.../comp_capture/bisect_vN \
  /capstor/.../comp_capture/clariden_vN

# 3. drill into the first divergent layer
python3 meta/diag/glm53/comp_diff.py components <beverin_dir> --layer 3 --ref <pure_torch_dir>

# quick manifest-only health check (no .pt loads):
python3 meta/diag/glm53/comp_diff.py summary <capture_dir>
```
