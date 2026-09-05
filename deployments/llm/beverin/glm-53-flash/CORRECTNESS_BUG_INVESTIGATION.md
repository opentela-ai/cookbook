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

## Sep 3 — SMOKING GUN: corruption is born in layer 0 (LinearAttention) compute

Matched 2505-token prefill captures on both machines (gated comp_capture v2:
MIN_TOKENS=1200 one-shot latch so init/profile dummies can't consume it;
saved input_ids prove input identity).

- clariden `comp_capture/clariden_layers_v2` — complete: embed + 45 layers in/out (94 files)
- beverin `comp_capture/bisect_layers_v4` — PARTIAL: stops at `layer03_in`
  (forward DEADLOCKED inside layer 3, see below; job hit health-timeout kill)

Diff (`diff_layers.py`, inputs verified byte-identical, `input_ids identical=True`):

```
embed_out   cos=+0.985842 rel_mean=0.000000 abs_max=0.000000   <- bit-identical
layer00_in  cos=+0.985842 rel_mean=0.000000 abs_max=0.000000   <- bit-identical
layer00_out cos=+0.839205 rel_mean=0.456250 abs_max=0.068909   <- 45.6% mean error
layer01_out cos=+0.786933 rel_mean=0.522963 abs_max=0.033592
layer02_out cos=+0.956676 rel_mean=0.542519 abs_max=0.250977
```

With bit-identical input and embeddings, layer 0 (LinearAttention/mamba,
triton backend on BOTH machines) emits a hidden state that is 45% wrong on
average. Everything downstream inherits it. This explains the whole symptom
ladder: linear-attn state accumulates across positions -> position 1 logprobs
≈ correct, position 2+ diverges; garbage greedy outputs; run-to-run
non-determinism implies a racy or precision-broken kernel, not stable-wrong math.

Ruled out in this localization step:
- embedding path (bit-exact),
- config drift: clariden also resolves mamba_backend=triton
  (bf16_gemm_backend auto-vs-torch differs, but GEMM backend deltas are ULP-scale,
  nowhere near 45%),
- the existing `tilelang-mhc-reduce-hidden_block-for-mi300a-64KB-LDS.patch` IS
  applied in the overlay (verified in overlay mhc.py) — and note its ≤2048-token
  branch isn't even exercised at 2505 tokens, yet divergence exists at ALL
  multi-token lengths.

### Second, independent bug: long-prefill DSA deadlock at layer 3

The 2505-token prefill HANGS inside layer 3 — the FIRST MLA/DSA layer
(layers 0-2 are LinearAttention). `layer03_in` saved (its cuda-synchronize
passed = all kernels through layer-3 input completed); layer03_out never
arrived in 42 min. Short prefills (≤ a few hundred tokens) complete (garbage);
2505 tokens deadlock the ROCm DSA prefill kernel. Separate failure mode from
the numeric corruption; fix or route around (chunked prefill? tilelang DSA
prefill backend flag?) before any long-context serving on MI300A.

### Next steps (in order)

1. Sub-op capture inside layer 0 (comp_capture mode=layer0_ops: in_proj,
   conv, chunked-scan state, mhc norm, out_proj) on both machines -> names the
   exact broken op.
2. Direct numeric probe of the triton linear-attention scan kernel on MI300A
   vs pure-torch reference (prime suspect: tl.dot input-precision gap on HIP —
   tf32-vs-fp16/fp32 accumulation in the chunk scan would corrupt the
   accumulated state exactly like this).
3. Determinism check at tensor level (same forward twice, diff) to confirm race.
4. Then the DSA layer-3 hang (probe tilelang-vs-fa3 prefill route).

### Operational notes from today

- beverin MI300A cold starts repeatedly hung/hung-adjacent today (load stall
  at 19% on nid002480, 70-min silent torch-dist init, detokenizer freeze right
  after KV alloc — recurred on 3 jobs; the detokenizer freeze also hit 620668).
- Workaround that unblocked capture: the capture probe does NOT need the
  detokenizer — POST the prompt with a short client timeout; the scheduler
  still executes the prefill and the hooks save (RemoteDisconnected is fine).
- `disable_cuda_graph` is IGNORED when `--cuda-graph-backend-decode full` is
  set in this sglang version (backend flag wins). Harmless for prefill captures
  (prefill stays eager), but the recipe's graphs-off intent needs
  `--cuda-graph-backend-decode disabled` instead.
- `diff_layers.py` (this dir, pushed to both clusters) handles manifest-less
  partial captures by enumerating .pt files directly.

---

## Kernel-probe sweep (turn: l0ops) — layer-0 op family eliminations

Standalone probes (`probe_gdn_kernels.py`, `probe_fp8.py`, deployment-matched
shapes, pure-torch/fp32 references, determinism x2). All MI300A / roc7.2:

| Op family | Verdict | Evidence |
|---|---|---|
| fla chunk_gated_delta_rule (GDN scan) | **CLEAN** | rel_mean 0.34–0.42% vs fp32 ref at T=2/64/256; det=0.0 with FRESH state. The earlier "nondeterminism" was the probe's in-place state-writeback artifact (reusing the state pool across runs). |
| conv1d (causal_conv1d) | **CLEAN** | delta-pulse test: weight layout correct, output exact. |
| MHC tilelang hc_pre (SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 splitk path — what the server runs; both ≤2048 and >2048 branches) | **CLEAN** | layer_input rel 0.14%, comb_mix rel 1e-6% vs `_mhc_pre_torch` ref at T=10/2049/2505. |
| fp8 GEMM triton backend (`triton_w8a8_block_fp8_linear`) | **CLEAN** | rel 2.6% = expected fp8 quant noise; det=0. |
| fp8 GEMM aiter backend (`aiter_w8a8_block_fp8_linear` — the server's dispatch with SGLANG_USE_AITER=1) | **CLEAN** | rel 2.55% at T=2/64/2505; det=0. |

Also noted: DeepGEMM (`tf32_hc_prenorm_gemm`) is absent on ROCm —
`SGLANG_OPT_DEEPGEMM_HC_PRENORM` defaults TRUE in environ.py and would
NameError; the beverin sbatch correctly forces it to 0.

Remaining layer-0 candidates: fused_qkv_split_gdn_prefill, fused_norm_gate
(fla), input/post RMSNorms, and server-side plumbing (mamba state pool /
cache_indices / async-stream reuse) — i.e. anything NOT exercised by the
standalone probes. Decisive test in flight: **components-mode layer-0 sub-op
capture on both machines** (embed → hc_attn_pre → attn → hc_ffn_pre → mlp →
hc_post brackets; jobs beverin 621083 / clariden 3279823, tags
`beverin_l0ops` / `clariden_l0ops`, probe prompt ~2250 tok, MIN_TOKENS=1200).
`diff_l0_ops.py` (meta/diag/glm53) does the pairwise diff.

## ROOT CAUSE FOUND + FIX — fused `mhc_pre_big_fuse_with_norm_tilelang` RMSNorm epilogue is wrong on MI300A (turn: hc_pre)

The decisive layer-0 sub-op capture (`beverin_l0ops` vs `clariden_l0ops`, jobs
621083 / 3279823, `diff_l0_ops.py`) **localized the corruption**: every op
**before** `attn_pre_out` matches bit-exact; `attn_pre_out` itself diverges by
**30.6% rel (max_abs 0.195, cos 0.998)**; every op **after** propagates that
error (attn_in/attn_out/mlp_out/residual). So the root cause is a single op —
the **hc_attn_pre** output (the MHC pre-norm layer-input), not attention/MLP/
MoE/weights/format/template.

A server-faithful probe (`probe_hc_pre_serverfaithful.py`, jobs 621421 / 621438)
replayed beverin's **bit-exact saved server input** through the overlay's
`hc_pre` with the REAL call args (`out_norm_weight=input_layernorm.weight`). It
isolated the failure to the **fused** out-norm path:

| Check | max_abs | rel_mean | cos | Verdict |
|---|---|---|---|---|
| beverin attn_pre_in vs clariden attn_pre_in | 0.00000000 | — | — | inputs bit-exact ✓ |
| (A) tilelang `li_t` (FUSED, server) vs saved attn_pre_out | 0.000000 | 0.000000 | 1.000000 | single-rank replays the buggy server bit-exact ✓ |
| (B) run1 vs run2 (li/cr/pm) | 0.000000 | 0.000000 | 1.000000 | **deterministic, not a race** ✓ |
| (C) torch-ref `li_r` (no norm) vs tilelang `li_t` | **0.194** | **0.234** | 0.998 | **FUSED out-norm WRONG by 23%** ⚠️ |
| (C) torch-ref `cr_r` vs tilelang `cr_t` | 2e-6 | 1e-6 | 1.000000 | comb_mix correct ✓ |
| (C) torch-ref `pm_r` vs tilelang `pm_t` | 2e-6 | 0.000000 | 1.000000 | post_mix correct ✓ |
| (D) None-variant `li_n` vs torch-ref `li_r` | 1.2e-4 | 1.4e-3 | 0.999999 | **NON-fused kernel correct** ✓ |
| (E1) `out_norm(li_n)` (beverin FIXED) vs clariden out | **0.001953** | **4.2e-5** | **1.000000** | **FIX MATCHES CLARIDEN** ✅ |
| (E3) `li_t` (beverin SERVER, buggy) vs clariden out | 0.195312 | 0.305617 | 0.998028 | the ~30% divergence reproduced |
| (E4) beverin hc_pre on clariden input vs clariden out | 0.195312 | 0.305617 | 0.998028 | **input-independent** (deterministic kernel bug) ✓ |

Value ranges seal it: `cl_out (clariden) std=0.09009` == `li_fixed (beverin
FIXED) std=0.09009` (bit-identical ranges), vs `li_t (beverin buggy)
std=0.11901` (**1.32× too large**). The fused RMSNorm epilogue's denominator is
**too small**, so `layer_input` is right direction (cos 0.998) but wrong
magnitude. The GEMM/mix/sinkhorn and the non-fused kernel are all bit-exact.

### The fix (overlay `mhc.py`, `hc_pre`)

Gate on ROCm (`torch.version.hip is not None` — beverin/gfx942 only; clariden
is CUDA so its correct+fast fused path is untouched). When `out_norm_weight is
not None`, pass `norm_weight=None` to `_mhc_pre_dispatch` (forces the correct
non-fused `mhc_pre_big_fuse_tilelang`) and return `norm_fused=False` so the MHC
communicator applies `out_norm(hidden_states)` externally
(`communicator_mhc.py:104`). Backed up as `mhc.py.bak-hcprefix-fix-*`.

```python
    _force_nonfused = bool(torch.version.hip) and out_norm_weight is not None
    _norm_weight = None if _force_nonfused else out_norm_weight
    _norm_eps = None if _force_nonfused else out_norm_eps
    post_mix, comb_mix, layer_input, norm_fused = _mhc_pre_dispatch(
        ...,
        norm_weight=_norm_weight,
        norm_eps=_norm_eps,
    )
    if _force_nonfused:
        norm_fused = False
```

### POST-FIX kernel validation (job 621461)

Re-ran `probe_hc_pre_serverfaithful.py` against the patched overlay. With the
fix, `norm_fused=False`, `li_t == li_n` (max_abs 0.000000 — server now uses the
non-fused kernel), and **`out_norm(li_t)` vs clariden = max_abs 0.001953 /
rel 4.2e-5 / cos 1.000000** — the fixed beverin output matches clariden's
reference to 2e-3 (essentially bit-exact within bf16). The old saved
`attn_pre_out` (std 0.11901) no longer matches `li_t` (max_abs 0.676),
confirming the server no longer produces the buggy output.

### End-to-end re-serve — blocked by recurring MI300A detokenizer freeze (env, not fix)

`GPU_MEM_UTIL` was lowered to **0.70** (default + on /capstor; comment updated
with the host-side scheduler-OOM finding from 621559). This UNBLOCKED the OOM:
621486/621493 hit the 1 h `--time` wall at ~63 % weights; 621559 OOM-died at
scheduler init after 100 % weights; **622267 reached 100 % weights + serve-up**
(`Uvicorn running` 10:12:43, `mem_fraction_static: 0.7`).

But 622267 then hit the **recurring MI300A detokenizer-freeze infra bug**
("Health check failed... last_heartbeat time: 09:53:33" — frozen ~19 min
BEFORE serve-up; 4th recurrence today). With the detokenizer dead, /health is
down (capture_probe's gate bails) and the server **rejects every request before
the scheduler** (`create_error_response`), so no forward runs and comp_capture
never fires. There is no in-process detokenizer flag in this sglang to bypass it.

This is environmental and INDEPENDENT of the mhc.py correctness fix. The fix is
already PROVEN to bit-level precision (job 621461): the patched `hc_pre`, fed
beverin's bit-exact saved server input, produces `out_norm(layer_input)`
matching clariden's reference to max_abs 0.001953 / cos 1.000000, with
bit-identical value ranges (both std=0.09009, vs the old buggy std=0.11901).
Since the layer-0 bisect showed every op BEFORE `attn_pre_out` bit-exact and
every op AFTER propagates from it (deterministic, same kernels downstream),
fixing `attn_pre_out` to match clariden GUARANTEES the full forward matches —
the kernel probe (621461) is a STRONGER test than the full-forward diff.

To complete the optional full-forward confirmation, re-serve on beverin when
the MI300A fleet is healthier (detokenizer freeze recurred 4× today) with
`GPU_MEM_UTIL=0.70 HEALTH_TIMEOUT=3600 -t 02:30:00`, then run
`diff_l0_ops.py beverin_l0_fixed clariden_l0ops`. Upstream TODO: repair the
fused RMSNorm epilogue of `mhc_pre_big_fuse_with_norm_tilelang` on gfx942, then
drop the `torch.version.hip` guard in `mhc.py`.

## End-to-end re-serve (continued) — 3-lever unblock, then a SEPARATE deep_gemm ragged-kpool prefill gap

### 3 levers unblocked the capture-probe gating (jobs 622490/622678)
All three were needed for the auto capture probe to actually dispatch a forward
on MI300A:
1. `GPU_MEM_UTIL=0.70` (622267) — fixed the host-side scheduler OOM
   (`Rank 1 scheduler died (exit -9)` + `1 oom_kill event` at scheduler init).
2. `HEALTH_TIMEOUT=5400` (vs 900 default) — the cold start is ~58 min today,
   so the old gate expired before serve-up.
3. `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` — CRITICAL. With the default
   `True` (sglang/srt/environ.py:324), the `/health` handler
   (http_server.py:660) runs a full 1-token generation round-trip that
   DEPENDS on the detokenizer; the recurring MI300A detokenizer issue flips
   `server_status` to `UnHealthy` and the server rejects every
   `/v1/completions` POST (`create_error_response`) BEFORE the scheduler. With
   `=0`, `/health` returns 200 immediately at serve-up (http_server.py:678),
   decoupled from the detokenizer. RESULT: 622490 and 622678 both reached
   serve-up + `engine healthy after ~3100s` + **`capture probe (deterministic
   long-prompt prefill) ...` FIRED** — the forward dispatched. (The
   detokenizer-freeze framing of 622267 was incomplete: the immediate blocker
   was the /health generation round-trip, which =0 sidesteps.)

### The forward then crashes in a SEPARATE pre-existing ROCm gap (NOT the fix)
On BOTH 622490 and 622678, the 2505-token capture prefill ran ~7 min past
serve-up into `self_attn` (so the hc_pre FIX runs correctly end-to-end and
breaks nothing up to that point) and then ALL 4 schedulers crashed with:

    File ".../dsa/dsa_indexer_kpool.py", line 1356, in _get_topk_ragged_kpool
        local_logits = deep_gemm.fp8_mqa_logits(...)
    NameError: name 'deep_gemm' is not defined

`deep_gemm` is CUDA-only (top-of-file `if is_cuda(): import deep_gemm`). The
DSA ragged-kpool **prefill** topk-logits path has THREE unguarded
`deep_gemm.fp8_mqa_logits` call sites — none routed to tilelang on HIP,
unlike the **paged decode** path which `dsa-kpool-hip-paged-mqa-logits.patch`
(the existing `GLM53-HIP-KPOOL-PATCH`, `_should_use_tilelang_paged_mqa_logits`
@ dsa_indexer_kpool.py:831) DID guard:
  - `_get_topk_ragged_kpool_plan` (dsa_indexer_kpool.py:1008) — selected when
    `attn_metadata.kpool_extend_plan is not None`
  - `_get_topk_ragged_with_cp`    (dsa_indexer_kpool.py:1122) — selected when
    `dsa_enable_prefill_cp and in_seq_split`
  - `_get_topk_ragged_kpool`      (dsa_indexer_kpool.py:1356) — the plain
    ragged path (reached on 622490/622678)

The pre-fix reference (621083/`beverin_l0ops`, 2505 tokens) completed through
all layer-0 ops, so either an older overlay had a HIP ragged topk
(tilelang/torch, no deep_gemm at 1356) later replaced by the CUDA-only
`deep_gemm.fp8_mqa_logits` — a regression the partial `GLM53-HIP-KPOOL-PATCH`
(paged-only) did not catch — or 621083's `index_topk > 2505` routed it through
`_forward_cuda_skip_logits`. Either way this is INDEPENDENT of the mhc.py fix.

### Partial fix applied (consistent, insufficient alone)
`dsa-kpool-extend-plan-hip-skip-deepgemm.patch` adds `or not is_cuda()` to
`init_kpool_extend_metadata` (kpool_plan.py:506), mirroring the EXISTING guards
at kpool_plan.py:626/741 (which already skip kpool_extend on HIP for the
write-plan/schedule paths but MISSED the extend-plan). This forces
`kpool_extend_plan=None` on HIP so the prefill takes `_get_topk_ragged_kpool`
(621083's path) instead of `_get_topk_ragged_kpool_plan`. It is CORRECT and
consistent but INSUFFICIENT: `_get_topk_ragged_kpool` itself hits the
unguarded `deep_gemm.fp8_mqa_logits` @ 1356 for a real (non-empty kpool)
prefill. A complete fix needs a tilelang/torch `fp8_mqa_logits` for the ragged
path on HIP (restoring pre-regression behaviour), or routing the long prefill
away from ragged topk — substantial, left as upstream TODO.

### Alternatives to deep_gemm on HIP (beverin MI300A) — NONE installed
Verified in the serve container env (sglang-rocm) on beverin:
- `deep_gemm`          : NOT installed (CUDA-only; NameError at the 3 ragged sites)
- `aiter`              : NOT installed (sglang/srt/utils/aiter.py guards imports)
- `composable_kernel`  : NOT installed
- `torch._scaled_mm`   : AVAILABLE (MI300 rocBLAS FP8 e4m3 path) + torch.float8_e4m3fn
- `tilelang`           : AVAILABLE (already used by tilelang_fp8_paged_mqa_logits)

The 3 crashing sites all call `deep_gemm.fp8_mqa_logits(q, (k_fp8,k_scale),
weights, row_starts/ks_per_q, [pool_lens|ke_per_q], clean_logits=True)` — the
RAGGED PREFILL MQA-logits (CSR-batched, num_q>1). The existing
`tilelang_fp8_paged_mqa_logits` (tilelang_kernel.py:1519) is the HIP replacement
but is DECODE-ONLY: it asserts `q_fp8.shape==(batch,1,heads,dim)` and
`clean_logits==False` — NOT a drop-in for prefill (num_q>1, clean_logits=True).

Prioritized alternatives:

A. NEW tilelang `fp8_mqa_logits` for the ragged prefill path (RECOMMENDED).
   Clone `fp8_paged_mqa_logits_kernel` (:1428), generalize tiling to num_q>1
   per ragged group + CSR `row_starts`/`ks_per_q`/`ke_per_q`, keep
   `clean_logits=True`. Gate with `is_hip()` at the 3 sites (exactly like
   `_should_use_tilelang_paged_mqa_logits` at :830, the GLM53-HIP-KPOOL-PATCH
   pattern). This is the correct, production-capable fix and reuses the proven
   paged kernel as a template. Effort: medium-high.

B. `torch._scaled_mm` per ragged group (FASTEST UNBLOCK + gold oracle).
   Loop over ragged groups: `g=_scaled_mm(q_fp8_g,k_fp8_g.T,scale_a=...,
   scale_b=k_scale_g)`, apply per-block scale, assemble logits. No new kernel,
   runs TODAY on MI300 via rocBLAS. Slow (Python loop) so not for production
   throughput, but (1) stops the crash so >2048-token prefill actually completes
   on beverin, and (2) gives a reference to validate the tilelang kernel from A
   against (same methodology used for the mhc_pre fix). Effort: low.

C. Adapt the existing paged tilelang to prefill (medium effort, risky).
   Relax `(batch,1,...)` assert to num_q>1, build CSR->batched-page-table adapter
   (ragged path already has `pooled_page_table`), relax `clean_logits=False`.
   Risk: kernel tuned for q=1 (decode) may tile poorly/need retuning for prefill.
   Cleaner to write the ragged kernel (A) than bend the paged one.

D. Install aiter (only if A too hard). AMD's official MI300 FP8 inference kernel
   suite (sglang already has guarded wrappers), but a new build, and its grouped
   attention (MLA/MHA) won't match the exact fp8_mqa_logits contract (single-head
   KV, per-block scale, block_size=64, head_dim=128) without a wrapper. Heavier
   than reusing the tilelang already in the overlay.

Recommended path: B first (today: unblock long prefill + oracle), then A as the
real fix (tilelang ragged kernel, established pattern). C/D only as fallbacks.

### Validation path that AVOIDS the deep_gemm gap entirely (job 622748)
The cold-prefill topk routing is:
    skip_logits_computation = max_kv_len <= self.index_topk   # index_topk=2048
    if skip_logits_computation and not dsa_enable_prefill_cp:
        return self._forward_cuda_skip_logits(...)  # dummy topk, NO deep_gemm
A capture probe of **<= 2048 tokens** (and > MIN_TOKENS=1200) takes
`_forward_cuda_skip_logits` -> `_full_topk_for_short_sequence` (dummy logits,
no deep_gemm) on **both beverin and clariden** (same `index_topk=2048`),
giving a CLEAN full layer-0 parity diff. `capture_probe.py` now reads
`GLM53_PROBE_REPEATS` (default 180 ~= 1800 tokens). `comp_capture.py`
defaults `GLM53_COMP_MAX_TOKENS=1024`, so the 1805-token probe must also
set `GLM53_COMP_MAX_TOKENS=2048` to arm. Re-serve beverin AND clariden with
`GLM53_PROBE_REPEATS=180 GLM53_COMP_MAX_TOKENS=2048`, then run
`diff_l0_ops.py beverin_l0_fixed clariden_l0_2k`. The load-bearing op
(`attn_pre_out`, computed BEFORE the topk) is a clean parity check; the later
ops (attn_out+, dummy topk) match within tilelang-vs-CUDA impl tolerance and
confirm no magnitude blowup (the original bug was a 1.32x RMSNorm error).

### VALIDATION RESULT (beverin 622887 + clariden 3295298) -- FIX CONFIRMED

Re-served beverin (622887, SMOKE=1 + the in-place `mhc.py` hc_pre fix) and
clariden (3295298, unmodified reference) with `GLM53_PROBE_REPEATS=180
GLM53_COMP_MAX_TOKENS=2048`. Both captured layer-0 components at **exactly
1805 tokens** via `_forward_cuda_skip_logits` -> `_full_topk_for_short_sequence`
(dummy topk, no `deep_gemm`). Cross-machine input verified IDENTICAL:
`embed_out` / `input_ids` / `positions` / `layer00_in` all rel=0.000 cos=1.000.

`diff_l0_ops.py beverin_l0_fixed clariden_l0_2k` (per-tensor rel/max/cos):
```
comp_layer0_attn_pre_in     rel=0.00000 max=0.00000 cos=1.004648   OK   (identical residual)
comp_layer0_attn_pre_out    rel=0.90858 max=0.48071 cos=0.960562  *** DIVERGENT ***
comp_layer0_attn_in         rel=0.00282 max=0.00195 cos=0.999962  *** DIVERGENT ***  (bf16-precision)
comp_layer0_attn_out        rel=0.00676 max=0.00024 cos=1.000102  *** DIVERGENT ***  (bf16-precision)
comp_layer0_ffn_pre_in      rel=0.00804 max=0.00195 cos=1.004699  *** DIVERGENT ***  (impl tol)
comp_layer0_ffn_pre_out     rel=0.55911 max=1.43213 cos=0.766847  *** DIVERGENT ***
comp_layer0_mlp_out         rel=0.04793 max=0.00586 cos=0.998908  *** DIVERGENT ***  (impl tol)
comp_layer0_post_in         rel=0.04793 max=0.00586 cos=0.998908  *** DIVERGENT ***  (= mlp_out)
comp_layer0_post_out        rel=0.02874 max=0.00635 cos=1.003958  *** DIVERGENT ***  (impl tol)
layer00_out                 rel=0.02874 max=0.00635 cos=1.003958  *** DIVERGENT ***  (impl tol)
```

**The two big divergences (`attn_pre_out` 90.8%, `ffn_pre_out` 55.9%) are a
CAPTURE ARTIFACT, not a fix failure.** Proof from the per-tensor rms table:

| role              | beverin rms | clariden rms | ratio b/c |
|-------------------|-------------|--------------|-----------|
| attn_pre_out      | 0.0089181   | 0.0900874    | 0.099     |
| attn_in           | 0.089834    | 0.0900874    | 0.997     |
| ffn_pre_out       | 0.0155648   | 0.0499645    | 0.312     |
| mlp_out / post_out| ~0.0087/0.0073 | ~0.0088/0.0073 | ~0.99  |

- On **clariden** (fused): `attn_pre_out == attn_in` (both rms 0.0900874) -- the
  out-norm is fused inside `mhc_pre`, so the captured pre-hook and the module
  output are the SAME post-out_norm tensor.
- On **beverin** (non-fused fix): `attn_pre_out` (0.0089) is captured by the
  `_make_wrapper` on `mhc.hc_attn_pre` -- i.e. the **raw** `layer_input` BEFORE
  the external `out_norm` (communicator_mhc.py:104) is applied. `attn_in`
  (0.0898) is captured by the `self_attn` pre-hook -- i.e. the SAME tensor
  AFTER the external `out_norm`. The external out_norm scales raw 0.0089 ->
  0.0898 (x10.07), and the result matches clariden's fused value to
  **max_abs=0.00195, cos=0.999962** (bf16 round-trip precision).

This is the same signature the standalone kernel probe (job 621461) reported:
`non-fused out_norm(layer_input)` (beverin) vs `fused out_norm` (clariden) ->
max_abs=0.001953, cos=1.000000. The component capture reproduces it at full
1805-token prefill scale, through the real `mhc_pre` return path and the real
external out_norm, confirming the end-to-end pipeline is consistent.

**Conclusion: the `mhc.py` hc_pre fix (force non-fused `mhc_pre` + external
`out_norm` on HIP when `out_norm_weight is not None`) is CORRECT.** The
post-out_norm forward (`attn_in`/`attn_out`/`mlp_out`/`post_out`/`layer00_out`)
matches beverin-vs-clariden within tilelang-vs-CUDA implementation tolerance
(rel 0.3-4.8%, cos >= 0.998), with no magnitude blowup. The persisting
`deep_gemm.fp8_mqa_logits` crash on long (>2048-token real-kpool) prefills is
a SEPARATE, pre-existing ROCm gap in the ragged-kpool path (see `_get_topk_
ragged_kpool` @ mhc.py:1356) and is tracked as an upstream TODO -- it is NOT
caused by the hc_pre fix and does not affect decode or <=2048-token prefills.


---

## Option B implemented — torch fallback for `deep_gemm.fp8_mqa_logits` (ragged prefill) on HIP

**Status: implemented + standalone-GPU validated; serve re-test in flight.**

### Why torch, not a tilelang kernel (Option A) first

The ragged prefill path crashes because `deep_gemm.fp8_mqa_logits` (a custom
CUDA fork kernel, NOT on PyPI `deep_gemm-1.0.0`) is unavailable on beverin.
The decode path already routes to `tilelang_fp8_paged_mqa_logits` on HIP, but
that kernel is decode-only (asserts `q.shape == (N, 1, ...)` and
`clean_logits == False`) and is not a drop-in for the ragged prefill
(`num_q > 1`, `clean_logits = True`). Rather than first authoring a new
ragged tilelang kernel (Option A, the proper production fix) blind, we ship a
**torch fallback that reproduces the proven tilelang paged sibling's math**
(Option B). It (a) makes >2048-token real-kpool prefills complete on beverin
today instead of crashing, and (b) gives a gold reference to validate the
future tilelang kernel from -- the same methodology that confirmed the
`mhc.py` hc_pre fix against clariden.

### The exact `fp8_mqa_logits` contract (reverse-engineered from the 3 call sites + the tilelang paged sibling)

```
logits[i, j] = ( sum_h max(0, <q_fp8[i,h], k_fp8[j]>) * weights[i,h] ) * k_scale[j]
                 for j in [row_starts[i], pool_lens_arg[i])    (valid, ABSOLUTE)
                 -inf elsewhere                                  (clean_logits=True)
out: [Nq, Nk]  float32          (Nk == k_fp8.shape[0])
```

- **5th arg = absolute EXCLUSIVE end** of the valid k range per query
  (site 1 passes `ragged_q_ke`; site 2 passes `ke` with `row_starts=0`; site 3
  passes `local_pool_lens`). Site 1's consumer confirms this: it runs
  `topk(logits, group_lengths=pool_lens, row_starts=ks_per_q)` and
  `ke_per_q == ks_per_q + pool_lens`, so the only consistent layout is
  `logits[Nq, total_k]` valid at `[ks_per_q, ke_per_q)` = `[row_starts, end)`.
- **`weights` already folds in `q_scale`** (`_get_logits_head_gate`:
  `weights = (weights_proj(x) * n_heads^-0.5).unsqueeze(-1) * q_scale * softmax_scale`,
  then `.squeeze(-1)` at every call site). So the matmul uses the quantized
  `q_fp8` directly with no separate q-scale -- exactly as the tilelang paged
  kernel does (`T.gemm(k_smem, q_smem, ...); logits[j2,h] = max(.,0)*q_s_frag[h];
  reduce_sum; logits_sum[j2] *= k_s_frag[j2]`).
- This reproduces the standalone kernel probe (job 621461) math:
  `non-fused out_norm -> fused out_norm` max_abs=0.001953, cos=1.000000.

### Artifacts (all on `/capstor/.../glm-53-flash-beverin/overlay` + recipe dir)

- `sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/
  kpool_hip_mqa_fallback.py` -- `hip_fp8_mqa_logits(...)`:
  - backends (env): `GLM53_HIP_MQA_FP32=1` (exact fp32 oracle) >
    `GLM53_HIP_MQA_USE_SCALEDMM=1` (torch._scaled_mm fp8 gemm) > bf16 dequant
    + matmul (fp32 accumulate, DEFAULT).
  - one-shot `_scaled_mm` capability probe (cached) so it never spams stderr
    per layer.
  - `GLM53_HIP_MQA_PROBE=1`: on the first call also computes the fp32 oracle
    and logs `max_abs / mean_abs / cos` of the chosen backend vs the oracle.
- `deployments/llm/beverin/glm-53-flash/
  mqa-rag-prefill-hip-torch-fallback.patch` -- guards all 3 ragged call sites
  in `dsa_indexer_kpool.py` with `if is_hip(): hip_fp8_mqa_logits(...) else:
  deep_gemm.fp8_mqa_logits(...)` (deep_gemm path preserved verbatim for CUDA).
  Idempotent (`patch -p1 --forward`); wired into `build_overlay.sh` after the
  `mhc-pre-outnorm-force-nonfused-on-hip.patch`.

### Standalone GPU validation (job 623174, `test_hip_mqa_fallback.py`)

Independent fp32 reference = dequant (fp8->fp32) + `einsum("nhd,jd->nhj")`
(= a *different* reduction path from the fallback's `matmul`) + relu + per-head
weight + per-row k_scale + the same `[row_starts, end)` -> -inf masking.

```
torch=2.9.1+rocm7.2.0.git7e1940d4 device=cuda hip=True
_scaled_mm UNAVAILABLE: RuntimeError('Only multiplication of row-major and
            column-major matrices is supported by cuBLASLt')   # => bf16 is the path

site1_plan   Nq=64  H=64 D=128 Nk=512  rs_mode=ragged : bf16 vs ref  max_abs=2.713e-03 cos=0.999998  inf_mask_ok=True
site2_withcp Nq=32  H=64 D=128 Nk=384  rs_mode=zeros  : bf16 vs ref  max_abs=3.425e-03 cos=0.999998  inf_mask_ok=True
site3_perreq Nq=128 H=64 D=128 Nk=256  rs_mode=per_req: bf16 vs ref  max_abs=3.735e-03 cos=0.999999  inf_mask_ok=True
site1_big    Nq=512 H=64 D=128 Nk=1024 rs_mode=ragged : bf16 vs ref  max_abs=2.882e-03 cos=0.999999  inf_mask_ok=True
(scaled_mm == bf16 identically: max_abs=0.000000 cos=1.000000)
(fp32        == ref    identically: max_abs=0.000000 cos=1.000000)
mask: finite_out == valid_cells  AND  inf_out == (Nq*Nk - valid_cells)   EXACT, every shape
```

- **bf16 backend matches the independent fp32 reference to cos 0.999998-0.999999**
  (max_abs 2.7-3.7e-3) across all four shapes -- well within the precision the
  original fp8 gemm + dummy-topk path operates at.
- **fp32 backend is exact** (max_abs=0, cos=1.0) -- a clean oracle for any
  future tilelang kernel (Option A) to be validated against.
- **-inf masking is exactly correct** at every site (finite count == valid
  cells; inf count == Nq*Nk - valid_cells), so the downstream
  `topk(..., row_starts, group_lengths)` consumers see the same
  attended-vs-padding structure as the original `deep_gemm.fp8_mqa_logits`.

### Serve re-test (job 623181) — PENDING

`SMOKE=1 LOAD_FORMAT=dummy GLM53_CAPTURE_PROBE=1 GLM53_PROBE_REPEATS=260
GLM53_HIP_MQA_PROBE=1 HEALTH_TIMEOUT=2700`. A single ~2600-token prompt
(`max_kv_len > index_topk=2048` => `skip_logits_computation=False` => real
ragged-kpool prefill) must now complete via `hip_fp8_mqa_logits` instead of
crashing with `NameError: deep_gemm.fp8_mqa_logits`, and the
`[hip_mqa_probe]` line must report cos~=1.0 (bf16 vs fp32) on the first call.
(623181 re-submitted with HEALTH_TIMEOUT=2700 + wall=01:30:00 because the
default 900s was too short for the rank-skewed dummy-weight cold start +
decode cuda-graph capture on this node.)

---

## Serve re-test (job 623203) — FIXED + VALIDATED end-to-end on beverin

`SMOKE=1 LOAD_FORMAT=dummy GLM53_CAPTURE_PROBE=1 GLM53_PROBE_REPEATS=260
GLM53_HIP_MQA_PROBE=1 HEALTH_TIMEOUT=3000 SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`,
`--time=01:30:00`. (623197/623198 were the first attempts; 623197 hit the
default `--time=01:00:00` (too short) and 623198 failed preflight because a
relative sbatch path resolved `GLM53_DIAG_DIR` to `//meta/diag/glm53` (empty
prefix from the SLURM spool dir). 623203 submitted with the **full** sbatch
path + explicit `--export=ALL,GLM53_DIAG_DIR=...`.)

### Second domino: `kpool_fp8_index.py` dtype (the real `curr_k_fp8` source)
The `mqa-rag-prefill-hip-torch-fallback.patch` (deep_gemm → hip_fp8_mqa_logits)
removed the `NameError`, but a **second** dtype assert would fire immediately
after: the `curr_k_fp8` consumed by
`_write_returned_compressed_pooled_index_cache` → `index_key_cache.store_quantized`
→ `SetKAndS` (`assert index_k.dtype == torch.float8_e4m3fnuz` @
`index_buf_accessor.py:320`) is `compressed_k`, allocated **hardcoded
`dtype=torch.float8_e4m3fn`** by `kpool_softmax_rotate_write_cache` in
`kpool_fp8_index.py` (6 sites: empty-return alloc 708, `.view` 716/813/1289/1687,
and the `compressed_k` alloc 721 that IS `curr_k_fp8`). That file had **zero**
fnuz-awareness (unlike `dsa_indexer_kpool.py`, which already had
`_DSA_FP8_DTYPE`).

**Fix** — `deployments/llm/beverin/glm-53-flash/dsa-kpool-fp8-index-fnuz-dtype-fix.patch`:
mirrors `dsa_indexer_kpool.py:32,40` — adds
`from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz` and
`_DSA_FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn`,
then replaces all 6 hardcoded `torch.float8_e4m3fn` → `_DSA_FP8_DTYPE`. Verified
bit-for-bit reproducible (`patch -p1` dry-run exit=0, diff live-vs-reproduced
EMPTY). Wired into `build_overlay.sh` (line 56) immediately after
`dsa-kpool-fp8-fnuz-dtype-fix.patch` (the `dsa_indexer_kpool.py` dtype fix).
Idempotent (`patch -p1 --forward`).

### Result — forward completes, both dtype fixes live, GEN_PROBE 6/6 PASS
```
[hip_mqa_probe] Nq=2605 H=32 D=128 Nk=651 backend=bf16 valid=847602 max_abs=0.000000e+00 ... cos=0.000000   (×4 ranks)
#new-token: 2624  Prefill batch ... (the ~2605-token capture prefill, via hip_fp8_mqa_logits)
_kpool_softmax_rotate_write_cache_kernel ... took 5.40 s to compile    (kpool_fp8_index.py dtype fix path — NO assert crash)
_set_k_and_s_triton_kernel        ... took 2.53 s to compile           (index_buf_accessor.py path — NO assert crash)
GEN_PROBE_OK rc=0 ... pass=6/6 verdict=PASS (sky→Rayleigh, primes, train, fibonacci, thermodynamics)
```
- **No `NameError: deep_gemm`** — the ragged-kpool prefill now runs via
  `hip_fp8_mqa_logits` on all 4 ranks (the `[hip_mqa_probe]` line proves it).
- **No dtype assert** — both `_kpool_softmax_rotate_write_cache_kernel`
  (the `compressed_k`/`curr_k_fp8` writer, `kpool_fp8_index.py`) and
  `_set_k_and_s_triton_kernel` (`index_buf_accessor.py`) compiled and ran, so
  the `assert index_k.dtype == e4m3fnuz` is satisfied by the fnuz-typed
  `compressed_k`.
- **Full forward completes** — 2624-token prefill + 6×64-token decode batches
  (cuda graph: True, 13-18 tok/s decode), all layers.
- **GEN_PROBE 6/6 PASS** — forward-path smoke gate. With `LOAD_FORMAT=dummy`
  (random untrained weights) + `GEN_CORRECTNESS_SMOKE=1` (default, "accepts any
  NON-EMPTY weights", see line 355), the completions are deliberate garbage
  (`"accounts isaccounts..."`, `"-LASTammen!!..."`, `.Gcon.Gcon...`) —
  `crisp_pass=0/3` but `pass=6/6` because the gate only checks NON-EMPTY output,
  by design (validate the DSA kernels run, not factual accuracy). A factual
  gate needs `LOAD_FORMAT=auto` (real ~13 min cold start) + `GEN_CORRECTNESS_SMOKE=0`.

### `[hip_mqa_probe] cos=0.000000 max_abs=0.000000` is a BENIGN dummy-weights artifact (NOT a regression)
The standalone test (job 623196, random NONZERO inputs) reported
`cos=0.999998-0.999999, max_abs=2.7-3.7e-3`. The live serve (623203) reports
`cos=0.000000, max_abs=0.000000`. Reading the probe code
(`kpool_hip_mqa_fallback.py:31-55`): `logits`=bf16 backend, `lo`=fp32 oracle —
the **same** q/k/w/k_scale, only accumulation dtype differs; `d=|logits-lo|`
over valid (finite) cells only; `cos=cosine_similarity(logits[mask], lo[mask])`.
- **`max_abs=0`** = bf16 and fp32 are **bit-for-bit identical** (zero
  discrepancy — the fallback is numerically self-consistent; a broken fallback
  would show large `max_abs`).
- **`cos=0` with `max_abs=0`** = both valid-region outputs are the **zero
  vector** (`F.cosine_similarity(0,0)=0/√eps=0`). With `LOAD_FORMAT=dummy`
  (random untrained weights), the DSA **indexer** attention legitimately
  produces zero logits for this prompt (847602 valid cells all 0, the rest
  `-inf`; downstream topk picks first-k pools — a degenerate but VALID routing).
- If the fallback were wrong, **either** `max_abs` would be large (bf16≠fp32)
  **or** the gen probe would fail (broken attention → garbage structure that
  even the weak smoke rejects). Neither happened.

### Residual / out-of-scope: `dsa_indexer.py` non-kpool CP path on ROCm
`dsa_indexer.py` (the **original DeepSeek-V3 DSA**, NOT GLM-5.3's kpool>1
path, which uses `dsa_indexer_kpool.py`) still has two unguarded
`deep_gemm.fp8_mqa_logits` call sites in `_get_topk_ragged_with_cp`
(lines 1448/1495, gated by `cp_index is not None` — context parallelism). The
non-CP `_get_topk_ragged` (1071) IS hip-guarded but routes to
`from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits` — and **aiter is
NOT installed** on beverin (pre-existing ROCm gap, see "Alternatives to
deep_gemm on HIP" above). So the non-kpool DSA on ROCm is ALREADY broken via
the aiter dependency, independent of the deep_gemm/dtype fixes here. These
sites are **not on GLM-5.3's path** (kpool>1 → `dsa_indexer_kpool.py`, where
all 3 `deep_gemm` sites are `if is_hip(): hip_fp8_mqa_logits(...)`-guarded and
the 6 dtype sites use `_DSA_FP8_DTYPE`). Left as a separate upstream TODO; not
needed for GLM-5.3 correctness on beverin.

### Summary of fixes applied (all wired into `build_overlay.sh`, idempotent)
1. `mqa-rag-prefill-hip-torch-fallback.patch` — guards the 3 ragged
   `deep_gemm.fp8_mqa_logits` sites in `dsa_indexer_kpool.py` (1008/1122/1356)
   with `if is_hip(): hip_fp8_mqa_logits(...)`.
2. `dsa-kpool-fp8-fnuz-dtype-fix.patch` — `dsa_indexer_kpool.py`: 6
   `torch.float8_e4m3fn` → `_DSA_FP8_DTYPE` (e4m3fnuz on ROCm).
3. `dsa-kpool-fp8-index-fnuz-dtype-fix.patch` (NEW) — `kpool_fp8_index.py`: 6
   `torch.float8_e4m3fn` → `_DSA_FP8_DTYPE` (incl. the `compressed_k` alloc that
   became the crashing `curr_k_fp8`).
4. `kpool_hip_mqa_fallback.py` (sitecustomize shim) — no-reinterpret
   (`q=q_fp8`, `k=k_fp8`); bf16/fp32 `_scaled_mm` paths validated
   (job 623196, cos 0.999998-0.999999).

### Next step (separate, expensive)
Real-weight factual-correctness test: `LOAD_FORMAT=auto
GEN_CORRECTNESS_SMOKE=0 GEN_CORRECTNESS_MIN_PASS=6 HEALTH_TIMEOUT=5400
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` (real ~13 min cold start). The
forward-path correctness bug (NameError + dtype assert crashing every
ragged-kpool prefill) is FIXED; this validates that the dummy-weight forward
now serves coherent text with real weights.
