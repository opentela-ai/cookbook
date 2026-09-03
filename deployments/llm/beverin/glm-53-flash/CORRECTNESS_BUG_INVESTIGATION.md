# GLM-5.3-Flash on MI300A (beverin) — correctness bug investigation

Job 614562 (real weights, chat-template fix applied) serves HTTP but produces
**garbage output**: real tokens, mostly common (" ", " the"), occasional
input-driven flips ("The capital of France is the city of" -> " France",
which copies a recent token but to the wrong target). Top-token logprobs are
~ -3.5 to -4.5 (weakly peaked, not uniform[-11.9] but far below a confident
model's ~ -0.3 to -1.5).

## What is RULED OUT (with evidence)

1. **Weight loading** — no missing keys, no shape mismatches, no OOM in the
   weight-load section of the engine log (scan excluded the giant server_args
   line). Weights load fully.
2. **FP8 load-time normalize** — PROVEN CORRECT in-container on MI300 hardware
   (`fp8test.py`): for 12 values spanning mantissas/zero/-0/100, the ratio
   `ground_truth(e4m3fn->f32) / fnuz_direct(same bits)` is a CONSTANT 2.0, and
   `sglang.normalize_e4m3fn_to_e4m3fnuz(...) * 2 == ground_truth` for all 12.
   => PyTorch's `float8_e4m3fnuz` on MI300 behaves like bias-8 *with* implicit
   leading 1 (only the bias differs from e4m3fn), so the `*2` scale in
   normalize is correct. **The earlier "AMD e4m3fnuz has no implicit-1"
   assumption was WRONG.** So the MoE weight dequant *at load* is fine.
3. **Chat template** — added `--chat-template "$MODEL_PATH/chat_template.jinja"`
   (verified `chat_template: '.../chat_template.jinja'` in the log). Both raw
   `/v1/completions` AND template-applied `/v1/chat/completions` give garbage,
   so the template is not the issue. (Note: chat returns empty `content` at
   max_tokens<=3 because GLM-5.3 is a reasoning model that emits a `<>...`
   think block first, parsed into `reasoning_content` by the GLM53 parsers.)
4. **Pure last-token bigram** — `bigram.py`: prompts with the SAME last token
   (" is") but DIFFERENT context give DIFFERENT outputs (" "," a"," and",
   " _","ning","i"). => context DOES reach the output; it is not a pure
   frequency/bigram prior. The output is "context present but WRONG."

## What the bug is (narrowed)

A **forward-pass numerical error in a novel MI300A kernel** that corrupts the
context/residual stream across all prompts. Not a load/format/template issue.

## Prime suspects (forward-pass kernels on MI300A, in execution order)

- **MHC pre-norm (tilelang), in MOST layers (linear_attention 0,1,2,...).**
  `mhc.py` (kernels/ops/layernorm) is tilelang-only on MI300A. The overlay
  patch `tilelang-mhc-reduce-hidden_block-for-mi300a-64KB-LDS.patch` reduces
  `hidden_block 256->128` (splitk) *only to fit the 64 KB non-optin LDS cap*
  (job 612201 crashed at `mhc_pre_gemm_sqrsum_splitk_stage_0_kernel` 98304 >
  65536). **Numerical correctness of the reduced-splitk MHC pre-norm was
  NEVER validated** — it was only gated for "doesn't crash". A wrong pre-norm
  GEMM (sqrsum) corrupts the residual of every linear-attention layer => global
  garbage. STRONGEST candidate for global corruption. No flag bypass
  (tilelang-only on ROCm, like DSA).
- **MoE in-kernel FP8 dequant (triton fused_moe).** Load-time normalize is
  correct, but the triton fused_moe kernel does its OWN per-block FP8 dequant
  during the GEMM. Untested for correctness. FP8 layout: per-expert
  `gate_proj`/`up_proj` [2048,4096] e4m3fn with `weight_scale_inv [16,32]`;
  `down_proj` [4096,2048] e4m3fn with `weight_scale_inv [32,16]`; block
  [128,128]; activation_scheme=dynamic. If the triton dequant uses a wrong
  bias/scale-index/granularity, every MoE layer is wrong.
- **DSA forward (`vk_hip_dsa_sparse_fwd`, PR #52 this session).** Validated
  11/11 on beverin (both tail_dim==0 GLM-5.3 and tail_dim>0 DeepSeek-V3).
  DEPRIORITIZED — but the **DSA indexer** (which tokens to attend to:
  `get_dsa_index_kpool`, kpool_plan, metadata fusion) was NOT validated for
  correctness. A wrong indexer makes DSA attend to wrong tokens => corrupted
  context (DSA layers are a minority per `layer_types`).
- **Mamba scan (triton)** — in linear-attention layers. Untested.

## Config / layout facts (from the live model)

- 62 safetensors. Tensors are `model.language_model.layers.N...` (NOT
  `model.layers...`).
- `text_config.layer_types` starts `['linear_attention','linear_attention',
  'linear_attention','deepseek_sparse_attention',...]`.
- `linear_attn_config = {num_heads:64, gate_lower_bound:-5.0, head_dim:128,
  short_conv_kernel_size:4, kda_layers:[0,1,...]}`.
- `index_kpool>1` (multimodal). `quantization_config = {quant_method:fp8,
  fmt:e4m3, activation_scheme:dynamic, weight_block_size:[128,128]}`.
- `lm_head.weight (154880,4096)` bf16, `embed_tokens` bf16 (not quantized).
- Per-expert MoE: gate_proj/up_proj [2048,4096] e4m3fn, down_proj [4096,2048]
  e4m3fn, weight_scale_inv [16,32]/[16,32]/[32,16] f32. (intermediate per
  expert = 2048; config `intermediate_size=12288`.)

## Infrastructure notes

- In-container python3.10 (has torch+sglang) is entered via
  `srun --jobid=JOB --overlap --gres=none -w NODE --environment=sglang-rocm -n1 python3 ...`
  (the `--environment=sglang-rocm` Enroot/EDF flag is ESSENTIAL — without it
  `python3` resolves to the host 3.6 which has no torch).
- The EDF `sglang-rocm` has a **stale workdir** (`glm47-flash-sglang-beverin`,
  deleted). Workaround applied: `ln -sfn .../glm-53-flash-beverin
  .../glm47-flash-sglang-beverin`. (Cleaner fix: re-register the EDF with the
  correct workdir, but that's shared across jobs.)
- `nsenter` into the live server PID is BLOCKED (no CAP_SYS_ADMIN).
- The model is APU (MI300A, gfx942, CPU+GPU shared memory).

## NEXT STEP (when warm server is available)

Job **614856** (4h, real weights, GEN_PROBE=0 -> server stays up past the gate)
is cold-starting (~21 min). When up, on that job's container:

1. **Single-expert FP8 GEMM primitive test**: load one expert's
   gate_proj+weight_scale_inv from safetensors; apply sglang's
   normalize_e4m3fn_to_e4m3fnuz (+scale*2); run sglang's triton fused_moe
   per-expert FP8 path (the EXACT server kernel) on a small fp8-exact input;
   compare to a BF16 per-block-dequant reference. Large error => bug in the
   triton FP8 dequant (fix it). Small error => MoE is correct.
2. **Single-layer MHC pre-norm test**: extract one linear-attention layer's
   input to the MHC pre-norm; run the tilelang `mhc_pre_gemm_sqrsum` (the
   server kernel) vs a pure-torch RMSNorm+sqrsum reference. Large error => bug
   in the tilelang reduced-splitk MHC pre-norm (fix it / route to torch).
3. If both pass, test the **DSA indexer** (forward with the real indexer vs a
   dense full-attention reference, per clariden commit b8d5296 which proved
   GLM-5.3 DSA layers are pure MHA and SDPA is a correct drop-in).
4. Once the broken kernel is pinpointed, correct it (rebuild the
   overlay/vkernels/triton kernel) and re-test coherence on a new job
   (~21-min cold start each).

## Clariden precedent (strong lead for the FIX shape)

Commit `b8d5296 fix(glm53-clariden): route DSA full-MHA prefill to PyTorch SDPA
(4th/last FA3)`: clariden is aarch64 GH200 (NVIDIA). It established GLM-5.3
DSA layers are **pure MHA** (num_heads==kv_heads==64, qk_nope_head_dim==
v_head_dim==256, qk_rope_head_dim==0 -> no GQA), so a per-request PyTorch SDPA
ragged loop is a **correct drop-in** for the full-MHA path. On beverin, if DSA
or MHC kernels are the bug, a torch/SDPA reference gives the correct answer
(slower, no sparse savings) — the same fix shape as clariden.

## In-progress: per-layer residual bisect (LSTAT) — Aug 31 ~18:00

Goal: find the FIRST layer whose residual goes bad (abs_mean explodes/collapses/
NaN) on a real beverin forward — names the broken kernel family (KDA-layer-0 vs
DSA-layer-3 vs MoE-layer-4) with NO reference needed.

- PATCH: `/tmp/glm53_layer_stats_patch.py` added `[LSTAT]` IN/OUT prints (rank 0,
  first forward only, try/except-wrapped so it can never break the forward) to
  `Glm5NextModel.forward` layer loop (glm5_next.py:1135). Applied to beverin at
  `.../overlay/sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py`
  (bak: `.bak_lstats`).
- KEY GOTCHA: the engine is launched `srun --environment=sglang-rocm bash
  engine.sh`. `--environment=sglang-rocm` REPLACES PYTHONPATH with the
  CONTAINER's `/sgl-workspace/sglang/python` (sglang 0.5.16, NO Glm5Next class,
  rejects `--bf16-gemm-backend torch`). The OVERLAY (`0.0.0.dev1`, HAS
  Glm5NextForConditionalGeneration -> glm5_next.py + accepts `torch`) must be
  forced INLINE: `srun --environment=sglang-rocm env PYTHONPATH=<overlay> bash
  engine.sh` (an `export PYTHONPATH` in the sbatch does NOT survive). Verified
  by precheck: `sglang.__file__` = overlay, `glm5_next.py` LSTAT_count=2.
- JOB 616115 on beverin (nid002964), 1h, TP4/EP4, identical backends to the
  broken engine (dsa-prefill/decode=tilelang, dsa-topk=torch, moe=triton,
  mamba=triton, bf16-gemm=torch, kv=bf16, cuda-graph OFF, skip-warmup) +
  GLM53_LAYER_STATS=1, port 30001 (no otela head). Sbatch auto-probes
  "The capital of France is" max_tokens=8 on /v1/models ready, greps [LSTAT].
- Next: read `[LSTAT]` lines from `lstat_engine.log` / `lstat_job_*.out`;
  the first layer where abs_mean/NaN jumps is the culprit family. Then a
  targeted isolated/SDPA test confirms the exact kernel.

### KEY GATE: SGLANG_USE_AITER -> page_size -> reproduces ' 1 ' (Aug 31 ~18:47)

LSTAT job 616115 CRASHED before the layer loop — NOT the garbage bug. Root cause:
- `dsa_backend.py:~1344` asserts `use_kpool = get_dsa_index_kpool(cfg) > 1` requires
  `real_page_size == 64`. GLM-5.3-Flash HF config has `index_kpool=4` (no env
  override; `get_dsa_index_kpool = getattr(config,'index_kpool',1)`).
- `aiter_can_use_preshuffle_paged_mqa()` (dsa/utils.py) sets page_size: True -> 64
  (preshuffle), False -> 1 (legacy). Gated by `SGLANG_USE_AITER` FIRST, then
  `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1` OR Triton>=3.5 (beverin has 3.6.0).
- The REGISTERED `sglang-rocm` EDF hardcodes `SGLANG_USE_AITER=0` (live check
  confirmed). EDF values OVERRIDE sbatch `export` (GLM53_* survive because the
  EDF doesn't list them; SGLANG_USE_AITER=1 from `export` was clobbered to 0).
  => page_size=1 => kpool>1 + page_size=1 => `AssertionError: kpool path
  requires page_size == 64` at `init_forward_metadata`, BEFORE the layer loop
  (no forward, no [LSTAT], empty probe body).
- FIX (inline, post-EDF, same pattern as PYTHONPATH):
  `srun --environment=sglang-rocm env PYTHONPATH="$PP" SGLANG_USE_AITER=1
  AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1 bash engine.sh` -> log shows
  `Setting page size to 64 for DeepSeek DSA.` (job 616424). This is the SAME
  config the broken ' 1 ' engine used, so the forward will now RUN and [LSTAT]
  will fire per-layer. (AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1 is belt-and-
  suspenders; Triton 3.6 alone would suffice once SGLANG_USE_AITER=1.)
- NOTE glm52 contrast: GLM-5.2 page_size=64 DEADLOCKS (100% GPU); GLM-5.3
  page_size=64 RUNS but gives garbage ' 1 ' -> points at the aiter preshuffle
  paged-MQA / kpool gather (or tilelang DSA prefill) producing WRONG output on
  gfx942, not a hang.

## Harness reorg (Sep 1) — bisect tooling moved to meta/diag/glm53

The LSTAT inline patch above was superseded by the first-forward component
capture, and the whole bisect harness now lives ONE level up in the cookbook:
`<cookbook>/meta/diag/glm53/` (see its README.md). One copy serves both
beverin and clariden; recipes point at it via `GLM53_DIAG_DIR`.

- `sitecustomize.py` here is now a thin DISPATCHER (~40 lines): it imports the
  individual patch modules (`patch_dsa_vk`, `patch_topk_torch`, `fwd_probe`,
  `patch_dsa_sdpa`) and, on `GLM53_COMP_CAPTURE=1`, `comp_capture` — all from
  `$GLM53_DIAG_DIR`. Engine drop-ins stay here: `vkernels_dsa.py`,
  `vkernels_dsa_topk.py` (installed into $OVL/pylib by build_overlay.sh).
- `comp_capture.py` / `capture_probe.py` / `comp_diff.py` (canonical, with the
  input_ids identity check) are in `meta/diag/glm53/`; the ad-hoc
  `analyze_bisect.py` / `diff_layers.py` / `probe_live*.py` / `_run_probe.sh`
  were folded into `comp_diff.py summary` and `live_probe.py` respectively
  (pre-deletion copies: ~/glm53-cleanup-backup-20260901).
- Clariden's sbatch heredoc now imports `comp_capture` from `$GLM53_DIAG_DIR`
  (no more hardcoded beverin path, no more GLM53_COMP_PYLIB).
