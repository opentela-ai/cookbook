#!/usr/bin/env python3
"""gen_correctness.py -- factual-correctness gate for Kimi-K3 on Beverin (vLLM).

The Clariden sglang servekit bench (servekit/src/servekit/bench.py,
run_correctness) sends these same six greedy completions to
``/v1/completions`` and CAPTURES the outputs for human comparison -- it does
not judge them. This probe runs the IDENTICAL requests (same prompts, same
endpoint, same ``temperature``/``max_tokens``) AND checks each answer for its
expected factual substring, so the recipe can gate otela registration on a real
"is the AITER MXFP4 MoE path producing correct tokens" verdict instead of on a
non-empty ``/v1/chat/completions`` body.

Why this matters: K3 has never generated a token on gfx942 (MI300A), and an
engine can answer ``/health`` while a broken MoE kernel serves garbage (the
deepseek-v4 recipe answered /health and 502'd every request). Greedy decoding
is deterministic up to floating-point; MXFP4 matmul accumulation order differs
between AITER (MI300A) and marlin (GH200), so byte-identical output across the
two stacks is NOT expected. Substring matching is the right level -- robust to
tiny logit differences, sensitive to actual corruption (which yields garbage,
not "Paris").

The six prompts split into:
  CRISP  -- the strongest corruption detectors; ALL must pass. A correct model
            produces these with essentially no variance (Paris for the capital,
            the prime sequence, 60/1.5 = 40 km/h).
  SOFT   -- reliable on a correct model, but more variance in phrasing; only
            enough of them are needed (MIN_PASS - #crisp) to reach MIN_PASS.
Default MIN_PASS=5 (all 3 crisp + >=2 of 3 soft): the one-prompt tolerance
covers the known Clariden anomaly (prompt 2 leads with the correct answer
"Rayleigh scattering" then rolls into a diff-format continuation), while a
broken MoE path fails the crisp prompts immediately.

Usage (run inside the kimi-k3-vllm container on the head; host netns shared,
so 127.0.0.1:$SERVE_PORT reaches vLLM on 0.0.0.0):

    python3 gen_correctness.py <base_url> <model> \\
        [out_json] [max_tokens=64] [min_pass=5]

Env (defaults, overridden by argv):
  GEN_CORRECTNESS_TIMEOUT         overall wall budget, s (default 600)
  GEN_CORRECTNESS_PER_REQ_TIMEOUT per-request urllib timeout, s (default 180;
                                  the first request may trigger CUDA-graph
                                  capture on a freshly cold-started engine)
Exit 0 == all CRISP prompts correct AND >= MIN_PASS of 6 correct, else 1.
A full per-prompt JSON report is written to ``out_json`` for inspection.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

# Identical to servekit.bench.CORRECTNESS_PROMPTS -- keep in sync if servekit's
# set changes (the Clariden coldstart.node0.json "correctness" block is the
# baseline these are compared against).
CORRECTNESS_PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue.",
    "List the first 10 prime numbers.",
    "Q: If a train travels 60 km in 1.5 hours, what is its average speed? A:",
    "def fibonacci(n):",
    "The three laws of thermodynamics are:",
]


def _has(needle):
    """Case-insensitive literal substring (needle must already be lower)."""
    return lambda text: needle in text.lower()


def _re(pattern, flags=0):
    compiled = re.compile(pattern, flags)
    return lambda text: bool(compiled.search(text))


def _train(text):
    # "average speed = 60 km / 1.5 h = 40 km/h". The number 40 (the answer) and
    # a km unit both appear near the start of a correct continuation; a broken
    # MoE path produces neither. First 60 chars avoids matching a later Q&A
    # turn the model might emit (Clariden continued with 150 km/2 h = 75 etc.).
    head = text[:60].lower()
    return ("40" in head) and ("km" in head or "kilometer" in head)


def _nonempty(text):
    """Smoke-test matcher: ANY non-empty continuation passes.

    Used with --load-format dummy (GEN_CORRECTNESS_SMOKE=1): the model serves
    random weights, so tokens are garbage but the full pipeline (weight load ->
    JIT compile -> forward pass -> /v1/completions -> HTTP response) is
    exercised end-to-end. A non-empty response proves the AITER/Triton MoE path
    does not crash, NaN, or 502 — the real question on first gfx942 bring-up.
    """
    return len(text.strip()) > 0


# (prompt_index, label, matcher, crisp?)
#  matchers receive the raw continuation (choices[0].text).
CASES = [
    (0, "capital -> Paris",                     _has("paris"),                              True),
    (1, "sky -> Rayleigh (scattering)",         _has("rayleigh"),                           False),
    (2, "primes -> 2, 3, 5, 7, 11",             _re(r"2\s*,\s*3\s*,\s*5\s*,\s*7\s*,\s*11"), True),
    (3, "train -> 40 km/h",                     _train,                                     True),
    (4, "fibonacci -> function body (return)",  _has("return"),                             False),
    (5, "thermodynamics -> entropy",            _has("entropy"),                            False),
]
assert [c[0] for c in CASES] == list(range(len(CORRECTNESS_PROMPTS))), "CASES must cover every prompt"
CRISP_TOTAL = sum(1 for c in CASES if c[3])


def _completions(base_url, model, prompt, max_tokens, timeout):
    """Mirror servekit.bench._completions exactly (POST /v1/completions)."""
    body = json.dumps(
        {"model": model, "prompt": prompt, "temperature": 0.0, "max_tokens": max_tokens}
    ).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0


def _text(resp):
    """Mirror servekit.bench._text: robust to missing choices."""
    choices = resp.get("choices") or [{}]
    return choices[0].get("text", "") if choices else ""


def _snip(text, width=72):
    """One-line, length-limited preview for the job log."""
    s = text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", " ")
    return s if len(s) <= width else s[:width] + "..."


def main():
    if len(sys.argv) < 3:
        print(
            "usage: gen_correctness.py <base_url> <model> "
            "[out_json] [max_tokens] [min_pass]",
            file=sys.stderr,
        )
        return 2
    base_url = sys.argv[1]
    model = sys.argv[2]
    out_json = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("GEN_CORRECTNESS_OUT", "")
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("GEN_CORRECTNESS_MAX_TOKENS", "64"))
    min_pass = int(sys.argv[5]) if len(sys.argv) > 5 else int(os.environ.get("GEN_CORRECTNESS_MIN_PASS", "5"))
    per_req = float(os.environ.get("GEN_CORRECTNESS_PER_REQ_TIMEOUT", "180"))
    overall = float(os.environ.get("GEN_CORRECTNESS_TIMEOUT", "600"))
    smoke = int(os.environ.get("GEN_CORRECTNESS_SMOKE", "0"))
    if smoke:
        # --load-format dummy: verify the PIPELINE (load + JIT + forward +
        # HTTP) produces non-empty tokens, NOT factual correctness. All
        # prompts are equal (no crisp gate); every one must be non-empty.
        min_pass = len(CORRECTNESS_PROMPTS)

    by_idx = {c[0]: c for c in CASES}
    t_start = time.time()
    results = []
    ok = 0
    crisp_ok = 0

    print(f"[{'SMOKE' if smoke else 'CORRECTNESS'}] probing {base_url.rstrip('/')}/v1/completions "
          f"model={model} max_tokens={max_tokens} min_pass={min_pass} "
          f"crisp={CRISP_TOTAL} budget={overall:g}s")

    for idx, prompt in enumerate(CORRECTNESS_PROMPTS):
        _, label, matcher, crisp = by_idx[idx]
        entry = {
            "prompt": prompt,
            "label": label,
            "crisp": crisp,
            "text": "",
            "ok": False,
            "error": "",
            "elapsed_s": 0.0,
        }
        try:
            if time.time() - t_start > overall:
                raise TimeoutError(f"overall budget {overall:g}s exceeded before prompt {idx + 1}")
            resp, dt = _completions(base_url, model, prompt, max_tokens, per_req)
            entry["elapsed_s"] = round(dt, 3)
            txt = _text(resp)
            entry["text"] = txt
            if smoke:
                entry["ok"] = _nonempty(txt)
            else:
                entry["ok"] = bool(matcher(txt))
            if entry["ok"]:
                ok += 1
                if crisp and not smoke:
                    crisp_ok += 1
        except Exception as exc:  # noqa: BLE001 -- record + continue so the
            # operator sees EVERY prompt's status, not just the first failure.
            entry["error"] = f"{type(exc).__name__}: {exc}"

        results.append(entry)
        flag = "PASS" if entry["ok"] else "FAIL"
        mark = "*" if crisp else " "
        preview = "<" + entry["error"][:72] + ">" if entry["error"] else _snip(entry["text"])
        print(f"[CORRECTNESS] {idx + 1}/{len(CORRECTNESS_PROMPTS)} {flag}{mark} "
              f"{label:<34} | {preview}")

    if smoke:
        verdict = ok == len(CORRECTNESS_PROMPTS)
    else:
        verdict = (crisp_ok == CRISP_TOTAL) and (ok >= min_pass)
    print(f"[{'SMOKE' if smoke else 'CORRECTNESS'}] pass={ok}/{len(CORRECTNESS_PROMPTS)} "
          f"crisp={crisp_ok}/{CRISP_TOTAL} "
          f"verdict={'PASS' if verdict else 'FAIL'} "
          f"(model={model}, max_tokens={max_tokens}, min_pass={min_pass}, "
          f"elapsed={time.time() - t_start:.1f}s)")

    report = {
        "base_url": base_url.rstrip("/"),
        "model": model,
        "max_tokens": max_tokens,
        "min_pass": min_pass,
        "crisp_total": CRISP_TOTAL,
        "pass": ok,
        "crisp_pass": crisp_ok,
        "verdict": "PASS" if verdict else "FAIL",
        "elapsed_s": round(time.time() - t_start, 3),
        "results": results,
    }
    if out_json:
        try:
            with open(out_json, "w") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
            print(f"[CORRECTNESS] report -> {out_json}")
        except Exception as exc:  # noqa: BLE001
            print(f"[CORRECTNESS] WARN: could not write {out_json}: {exc}",
                  file=sys.stderr)

    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
