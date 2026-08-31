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
