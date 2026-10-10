# GLM-5.3-Flash on Beverin (SGLang, ROCm overlay) → OpenTela

Serve `zai-org/GLM-5.3-Flash` (FP8 e4m3 ~306 GB, DSA + linear/KDA + MTP,
**multimodal**) on CSCS Beverin (4× MI300A / node, gfx942) via a
**Python-source overlay** of the only GLM-5.3 image (the Clariden
aarch64/CUDA build) onto the upstream ROCm v0.5.18 image, and register it on
OpenTela.

> **Status: boots and serves HTTP end-to-end; output correctness under
> investigation.** The boot blockers (deep_gemm NameError → MHC shared-memory
> abort → TVM vectorizer FloorMod) are all fixed — see the bring-up history
> below and [vkernels PR #52](../../../../../../meta/diag/glm53/). The current
> front is **garbage output** (job 614562 and on):
> [CORRECTNESS_BUG_INVESTIGATION.md](CORRECTNESS_BUG_INVESTIGATION.md) is the
> living campaign log (root cause found in the fused MHC RMSNorm epilogue;
> fix + kernel validation in job 621461).

## Start here

From the login node (or local machine — `up.sh` wraps the required
`rcc run --cwd` so sibling files resolve):

```bash
bash up.sh                                              # fast smoke: dummy weights, forward gate, no otela
bash up.sh --export=ALL,LOAD_FORMAT=auto,SMOKE=0        # real weights + OpenTela registration

tail -f /capstor/scratch/cscs/xyao/glm-53-flash-beverin/logs/<job>.out
# after health + GEN_PROBE_OK: curl http://<HEAD>:30000/v1/models
```

One-time prerequisites: `bash build_overlay.sh` (extracts the Clariden GLM-5.3
Python sources into `$OVL`; the sbatch fails preflight without it), the otela
x86_64 binary (SMOKE=0 only), and the ROCm image in the EDF.

## Boot-blocker history (jobs 612093 / 612201 / 612262 → PR #52)

The chain that got a first forward through on MI300A, in order:

1. **`SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`** — GLM-5.3's MHC hybrid-attention
   pre-norm has two `deep_gemm` paths (mhc.py:850/1450); `deep_gemm` is
   CUDA-only and never imported on ROCm → `NameError` on the first forward
   (job 612093). =0 takes the Triton/tilelang fallbacks.
2. **MHC shared-memory patch** — tilelang's HIP codegen only *queries*
   `hipDeviceAttributeMaxSharedMemoryPerBlock` (the 64 KB non-opt-in cap) and
   never calls `hipFuncSetAttribute` to opt into the ~228 KB dynamic cap, so
   the default `hidden_block=256` kernels abort: "Requested dynamic shared
   memory 98304 exceeds device limit 65536" (job 612201).
   `tilelang-mhc-reduce-hidden_block-for-mi300a-64KB-LDS.patch` (applied by
   `build_overlay.sh`) reduces `hidden_block` 256→128 in both MHC pre-norm
   kernels → per-stage 49152 B < 65536. Note this halves arithmetic intensity;
   the proper fix is a tilelang HIP dynamic-smem opt-in (would restore 256 at
   full speed).
3. **vkernels PR #52** — with 1+2 the engine reached the first forward's DSA
   decode, where `sparse_mla_fwd_decode_partial` hit a tilelang/TVM
   *vectorizer* bug: a symbolic FloorMod divisor folds to 0 for GLM-5.3's
   kpool>1 decode shape (`Check failed: pb->value != 0 (0 vs. 0): Divide by
   zero`, job 612262). No flag-switch escape exists: `tilelang` is the ONLY
   kpool>1-legal DSA backend present on ROCm (fa3/trtllm are NVIDIA-only;
   `flashmla_*`/`aiter` fail the `_check_kpool_tail_backend` hard gate).
   PR #52 (commit d76517c) re-implements the DSA forward as a plain gfx942
   HIP kernel (`vk_hip_dsa_sparse_fwd` in `libvkernels_hip.so`) taking
   `tail_dim==0` as a runtime branch — validated 11/11 on beverin (both
   tail_dim==0 GLM-5.3 and tail_dim>0 DeepSeek-V3 shapes). Wired in by
   `sitecustomize.py` + `vkernels_dsa.py` on `$OVL/pylib` (FIRST on
   PYTHONPATH), which rebind sglang's `tilelang_sparse_fwd` on gfx942 before
   the first DSA forward. Requires `VKERNELS_LIB` to point at the
   container-ROCm build (`build/hip-roc72/...libvkernels_hip.so`, ROCm 7.2) —
   the login-node `build/hip` links libamdhip64.so.6 and fails dlopen.

## The kpool>1 gate (why every DSA knob is pinned)

GLM-5.3 is multimodal with `dsa_index_kpool > 1`. sglang's
`DeepseekSparseAttnBackend` has a hard correctness gate
(`_check_kpool_tail_backend`, raised on every forward when kpool>1) that only
permits `{fa3, tilelang, trtllm}` — on ROCm that leaves tilelang alone, so
`GLM53_DSA_PREFILL=GLM53_DSA_DECODE=tilelang` is forced. The GLM-5.2
workaround (aiter for prefill) is **illegal here** because GLM-5.3's kpool>1
fails that gate. Any non-tilelang override fails on the first forward.

## GLM-5.3 vs GLM-5.2 (why the args differ)

- **Multimodal** (`Glm5NextForConditionalGeneration`) → needs
  `--mm-attention-backend triton_attn`: the vision tower auto-defaults to fa3
  on gfx942, which is missing on ROCm → ImportError. triton_attn uses Triton
  kernels, no flash_ops dependency.
- **kpool>1** (above) → DSA pinned to tilelang.
- **FP8 e4m3 dynamic** (not mxfp4) → no AITER FlyDSL/marlin MXFP4 MoE path;
  `--moe-runner-backend triton` because the auto pick (aiter) JIT-hangs ~52
  min on gfx942 for GLM-5.3's fp8-e4m3 ck2stages kernel
  ("Current hipcc not support: -amdgpu-coerce-illegal-types=1").
- **MTP nextn=1** → `--skip-server-warmup` stays on (MTP warmup crash);
  CUDA graphs are ON (the capture-time fa3 ImportError is routed around by a
  patched `_resolve_kpool_tail_backend` → tilelang → the PR #52 shim), which
  is the real per-step perf win.
- **Parsers** glm45/glm47 — verified present in the Clariden overlay.
- **Chat template**: GLM-5.3 ships `chat_template.jinja` as a separate file;
  sglang's auto-load misses it → raw-text input and garbage output. Passed
  explicitly (`--chat-template`), gated by `GLM53_CHAT_TEMPLATE`.
- **bf16 GEMM backend**: GLM-5.3 keeps MLA projections in bf16 while experts
  are FP8; with `SGLANG_USE_AITER=1` sglang routed every bf16 GEMM through
  aiter's tuned-GEMM dispatcher, which has no tuned config for the M:1 decode
  shapes and SIGKILLs on gfx942 (job 613339, ~336 s stall then exit -9 in
  `DeepseekV2AttentionMLA`). `--bf16-gemm-backend torch` (rocBLAS) fixes it;
  the underlying sglang short-circuit bug is fixed by
  `sglang-unquant-bf16-gemm-respect-backend-flag.patch`.

## APU memory math (why `GPU_MEM_UTIL=0.70`)

MI300A shares one ~513 GB HBM pool across CPU + 4 dies. At 0.88 each die
reserves 104.7 GB (weights 75.9 + KV/Mamba ~29), leaving ~14 GB/die for
serving-time inductor/Triton JIT — a second JIT round (decode shapes) OOMs
with fragmentation. 0.80 frees ~24 GB/die GPU headroom but — unified memory —
leaves only ~20% of the pool for the four ranks' HOST-side structures
(schedulers, mamba state cache, detokenizer, KV bookkeeping): on slow cold
starts (≥24 min dist-init + ≥36 min weights at ~22 s/shard) the scheduler OOMs
at init (exit -9, job 621559). 0.70 restores headroom; the mamba-cache cap
already bounds `max_running_requests` to ~54, so the extra GPU reservation
buys nothing for a correctness re-serve. Override per-deploy (e.g. 0.75).

## Diagnostics shipped in this sbatch

- **Forward probe** (`GLM53_FWD_PROBE=1`, default while diagnosing): logs each
  decoder layer + self_attn + mlp ENTER/EXIT on the first forward, so the last
  line before a silent stall names the hanging op. Zero-cost when off.
- **Slow-engine health mode** (`SLOW_ENGINE=1`): polls `/generate`
  (max_tokens=1) instead of `/health` — on eager/slow engines every aborted
  curl leaves a ~24 s generate queued, the detokenizer never idles, its
  heartbeat goes stale and every later `/health` fails (job 625724 lesson).
  A 1-token generate drains the queue instead of growing it.
- **Shared diag harness**: `GLM53_DIAG_DIR` → `meta/diag/glm53` (comp_capture,
  capture_probe, comp_diff, import hook) — identical copies run on beverin and
  clariden. `GLM53_CAPTURE_PROBE=1` fires one deterministic ~2k-token request
  before the gen probe so component captures align across sites.
- **tilelang JIT probe** (`GLM53_TILELANG_PROBE=1` + `GLM53_TL_LOG=DEBUG`):
  localizes the fp8_paged_mqa_logits JIT hang on gfx942.

## Files

| File | Purpose |
|---|---|
| `serve_glm_53_flash_sglang.sbatch` | The recipe: preflight (overlay + imports), 1× TP4 engine in the sglang-rocm container, mandatory gen probe, otela worker (SMOKE=0), clean LEFT on SIGTERM |
| `up.sh` | One-shot submit via `rcc run --cwd` (forwards extra sbatch args) |
| `build_overlay.sh` | One-time: extracts the Clariden GLM-5.3 Python sources + cp310 dep wheels into `$OVL`, applies the patches below |
| `sglang-rocm.toml` | EDF: v0.5.18 ROCm image by digest, overlay PYTHONPATH, MI300A env |
| `sglang_launcher_mi300a.py` | `is_integrated=False` memory-accounting launcher (also canonical in [`meta/common`](../../../../meta/common/)) |
| `sitecustomize.py` + `vkernels_dsa.py` / `vkernels_dsa_topk.py` | gfx942 rebinds of sglang's DSA tilelang paths onto the vkernels HIP kernels (PR #52, #51, #56) |
| `*.patch` | sglang/tilelang/aiter fixes applied by `build_overlay.sh` (MHC LDS, FNUZ dtype, kpool backend routing, bf16-gemm flag, …) |
| `probe_*.py` / `diff_*.py` / `analyze_bisect.py` / `pylib_comp_capture.py` | Kernel-level probes + bisect tooling for the correctness campaign |
| `core_*` | Crash-dump artifacts from the campaign (not consumed by the recipe) |
| `CORRECTNESS_BUG_INVESTIGATION.md` | The garbage-output campaign log (root cause + fix, job 621461) |

Diagnostic defaults (`GLM53_FWD_PROBE=1`, `GLM53_COMP_CAPTURE=1`,
`GLM53_TILELANG_PROBE=1`) come from the campaign; remove them once the
beverin-vs-clariden bisect closes.
