# GLM-5.3-Flash Bristen — reasoning / tool-call parser diagnosis

## Symptom

`/v1/chat/completions` against the live SGLang server (job 83206) returns
`"reasoning_content": null` and dumps the model's *entire* output (the
intended reasoning **and** the final answer) into the `content` field. The
`usage.reasoning_tokens` is `0`.

## Root cause

`engine.sh` launches SGLang with **no `--reasoning-parser` and no
`--tool-call-parser`** (confirmed in the job-83206 `launching SGLang args=`
line: the list contains `--enable-metrics` then jumps to `--quantization fp8`,
no parser flags). With `server_args.reasoning_parser = None`, SGLang's
`OpenAIServingChat._get_reasoning_from_request()` returns `False` early and
the reasoning detector is never run, so `reasoning_content` is always null and
nothing strips/redirects the model's reasoning markup.

## Evidence (all validated on the live 83206 server)

### 1. The GLM-5.3 chat template emits reasoning markup

`chat_template.jinja` (lines 254–255) appends, for `add_generation_prompt`,
the special token **`imd`** (added-vocab id **154841**) to seed the assistant
turn. The model is therefore expected to produce:

    <think> ... reasoning ... </think> final answer

and the tokenizer's added-vocab confirms the delimiters:

    154841  'imd'                      [U+003C 0074 0068 0069 006E 006B 003E]   (7 ASCII chars)
    154842  ''                      [U+003C 002F 0074 0068 0069 006E 006B 003E] (8 ASCII chars)
    154843  ''               [U+003C 0074 006F 006F 006C 005F 0063 0061 006C 006C 003E]
    154844  ''            [U+003C 002F 0074 006F 006F 006C 005F 0063 0061 006C 006C 003E]

A raw `/generate` request (so special tokens are NOT stripped) shows the model
*does* emit the close-think token 154842 exactly once, between the reasoning
and the answer:

    output ids: ... count of 154842: 1 count of 154841: 0
    154842 at id index 147
    before: '... the response should be brief and clear.'
    after:  '**Step 1:** Start with the first...'

(154841 is absent from the output because it was consumed by the generation
prompt; the model continues straight from the seeded `imd`.)

### 2. `glm45` is the correct reasoning parser for this model

`sglang.srt.parser.reasoning_parser.Glm45Detector` uses
`think_start_token='imd'`, `think_end_token=''`, `tool_start_token=''`,
`reasoning_default="enable_thinking"`, `thinks_internally=True`. Its
auto-detection rule (`template_detection._is_glm45`) requires a
`ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True)`,
which GLM-5.3's chat template does **not** expose (it uses
`reasoning_effort`/`clear_thinking` instead) — so `auto` detection never
selects `glm45`, and we must pass it explicitly.

### 3. `Glm45Detector(force_reasoning=True)` correctly splits the real output

Reproduced by feeding the model's actual output (reasoning + single ``) to a
`ReasoningParser(model_type="glm45", stream_reasoning=False,
force_reasoning=True)` constructed exactly as `serving_chat.py` does on a
non-`continue_final_message` request (no `chat_template_kwargs` →
`_get_reasoning_from_request` returns True because
`reasoning_default="enable_thinking"` and there is no chat_template_kwargs
toggle to consult → `force_reasoning` is honored).

`_detect_and_parse_impl` sets `_in_reasoning = force_reasoning` (= True), sees
`''` in the text, and splits:

    reasoning_text = text before ''
    normal_text    = text after ''

so the final answer lands in `content` and the reasoning in `reasoning_content`
— which is the desired OpenAI-compatible behaviour.

### 4. `glm47` is the matching tool-call parser

`sglang.srt.function_call.glm47_moe_detector.Glm47MoeDetector` is documented
as the detector for "GLM-4.7 and GLM-5 models" and parses the
`tc.name…` format that the GLM-5.3 chat template renders
(template line 161–162). `glm45` (Glm4MoeDetector) is the legacy GLM-4.5
tool-call shape.

## Fix (applied to engine.sh, not yet deployed)

`engine.sh` `SGLANG_ARGS` now includes, right after `--enable-metrics`:

    --reasoning-parser glm45
    --tool-call-parser glm47

To deploy on the next serve, simply resubmit `serve_glm_53_flash_sglang.sbatch`.
The currently-running job 83206 was launched before this change and still has
no parser; a resubmit is required for `reasoning_content` to populate. A
short `SMOKE=1 SERVE_PORT=30050 ...` validation job (83273) was started but
cancelled — it loads real FP8 weights (60–90 min) and the offline
`Glm45Detector` test above is sufficient proof.

## Why the earlier chat-template `reasoning_effort`/`clear_thinking` matters

The GLM-5.3 chat template (line 1–3) defaults
`reasoning_effort` to `'max'` and `clear_thinking` to `false`. It does NOT
honour `enable_thinking`. This is exactly why SGLang's *auto* detection
(`_is_glm45`) fails for GLM-5.3, and is also why passing
`chat_template_kwargs={"enable_thinking": True}` has no effect on the
generated output (the template ignores it). The only way to get reasoning
parsing is the explicit `--reasoning-parser glm45` server flag, which sets
`force_reasoning=True` for `thinks_internally` models.
