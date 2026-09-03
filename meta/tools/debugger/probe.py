#!/usr/bin/env python3
"""probe — live-server sanity probes for correctness bugs (Phase 0).

Run this BEFORE touching anything else. The failure signature selects the
suspect class; see README.md ("The method"). Stdlib-only: runs on no-egress
compute nodes, over SSH, or from a laptop.

Probes (generalized from the GLM-5.3-Flash MI300A campaign's probes C–I):
  A coherence         known-answer prompts -> expected substring present?
  B determinism       same prompt twice -> identical output? (nondeterminism
                      means it is NOT a pure numerics bug)
  C context-sensitivity  same last token, different prefixes -> outputs MUST
                      differ ("bigram test": caught 'context present but wrong')
  D repetition        neutral prompt, long max_tokens -> degenerate loop?
  E concurrency       4 identical requests -> consistent outputs?
  F prefix-following  long prefix then a question -> answer follows the prefix?
  G counting          simple token-counting -> positional/tracking sanity

Usage:
  python3 probe.py --url http://NODE:30001 --model org/model-name
  python3 probe.py --url ... --model ... --json          # machine-readable
  python3 probe.py --url ... --model ... --only A,C      # subset
  python3 probe.py --url ... --model ... --completions   # /v1/completions (raw)

Exit codes: 0 all probes pass; 1 at least one fails (detail printed/json).
The failure SIGNATURE matters more than pass/fail: real tokens with weakly-
peaked logprobs (-3.5..-4.5) that stay deterministic and context-sensitive is
the classic forward-pass-numerics fingerprint.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TIMEOUT = 240

DEFAULT_KNOWN_ANSWERS = [
    {"prompt": "The capital of France is the city of", "expect": "Paris"},
    {"prompt": "The capital of Japan is the city of", "expect": "Tokyo"},
    {"prompt": "2 + 2 =", "expect": "4"},
]
# Same last token (" is"), different context: outputs MUST differ. If they do
# not, context never reaches the output (pure prior/bigram behavior). If they
# differ but are all wrong, context IS reaching the output and is corrupted —
# the forward-pass fingerprint.
CONTEXT_PAIRS = [
    ("The name of the French capital is", "The boiling point of water in Celsius is"),
]
REPETITION_PROMPT = "Write a short story about a robot."
PREFIX_CASE = (
    "Answer with exactly one word.\n\n"
    "The sky is blue. Grass is green. Fire is hot. Snow is white.\n"
    "Question: what color is snow?\nAnswer:"
)
COUNT_PROMPT = "Count: 1, 2, 3,"

# ---------------------------------------------------------------------------
# HTTP (stdlib only)
# ---------------------------------------------------------------------------


def _post(url, payload, timeout=TIMEOUT):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post_completions(url, payload, timeout=TIMEOUT):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _gen(url, model, prompt, max_tokens=16, temperature=0.0, raw=False, logprobs=True):
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": logprobs and not raw,
        "top_logprobs": 3 if (logprobs and not raw) else None,
    }
    if raw:
        payload["prompt"] = prompt
        return _post_completions(url, payload)
    payload["messages"] = [{"role": "user", "content": prompt}]
    return _post(url, payload)


def _text(resp, raw=False):
    try:
        ch = resp["choices"][0]
        return ch["message"]["content"] if not raw else ch["text"]
    except Exception:
        return ""


def _top_lp(resp, raw=False):
    """First generated token's top logprob, or None. Weak peak detection."""
    try:
        ch = resp["choices"][0]
        lp = ch.get("logprobs") or {}
        content = lp.get("content") or lp.get("top_logprobs") or []
        if content and isinstance(content[0], dict):
            tops = content[0].get("top_logprobs") or [{ "logprob": content[0].get("logprob") }]
            vals = [d.get("logprob") for d in tops if isinstance(d, dict) and d.get("logprob") is not None]
            return max(vals) if vals else None
        return None
    except Exception:
        return None


def _run(label, fn, results, echo=print):
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"probe raised: {exc!r}"
    results.append({"probe": label, "ok": ok, "detail": detail})
    mark = "PASS" if ok else "FAIL"
    echo(f"[{mark}] {label}: {detail}")
    return ok


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def make_probes(url, model, raw, only):
    P = {}

    def wanted(key):
        return only is None or key in only

    def p_coherence():
        bad = []
        for case in DEFAULT_KNOWN_ANSWERS:
            r = _gen(url, model, case["prompt"], max_tokens=8, raw=raw)
            out = _text(r, raw)
            good = case["expect"].lower() in out.lower()
            if not good:
                bad.append(f"{case['prompt']!r}->{out[:40]!r}")
        lp = _top_lp(_gen(url, model, DEFAULT_KNOWN_ANSWERS[0]["prompt"], max_tokens=1, raw=raw), raw)
        peak = f" top1_logprob={lp:.2f}" if isinstance(lp, float) else ""
        if bad:
            return False, f"{len(bad)}/{len(DEFAULT_KNOWN_ANSWERS)} known-answer prompts wrong{peak}: " + "; ".join(bad)
        return True, f"known answers correct{peak}"

    def p_determinism():
        outs = [_text(_gen(url, model, DEFAULT_KNOWN_ANSWERS[0]["prompt"], max_tokens=12, raw=raw), raw) for _ in range(2)]
        return outs[0] == outs[1], f"two runs identical={outs[0] == outs[1]} out={outs[0][:40]!r}"

    def p_context():
        outs = []
        for prefix in CONTEXT_PAIRS[0]:
            r = _gen(url, model, prefix + " is", max_tokens=6, raw=raw)
            outs.append(_text(r, raw))
        differ = outs[0] != outs[1]
        return differ, f"same last token, different prefixes -> differ={differ} outs={outs!r}"

    def p_repetition():
        out = _text(_gen(url, model, REPETITION_PROMPT, max_tokens=48, raw=raw), raw)
        words = out.split()
        uniq = len(set(words)) / max(1, len(words))
        return uniq > 0.2, f"unique-word ratio={uniq:.2f} (<=0.2 smells like a degenerate loop) out={out[:60]!r}"

    def p_concurrency():
        with ThreadPoolExecutor(max_workers=4) as ex:
            outs = list(ex.map(lambda _: _text(_gen(url, model, DEFAULT_KNOWN_ANSWERS[0]["prompt"], max_tokens=8, raw=raw), raw), range(4)))
        consistent = len(set(outs)) == 1
        return consistent, f"4 concurrent requests -> {len(set(outs))} distinct outputs {outs[0][:40]!r}"

    def p_prefix():
        out = _text(_gen(url, model, PREFIX_CASE, max_tokens=6, raw=raw), raw)
        return "white" in out.lower(), f"prefix question answered: {out[:40]!r} (want 'white')"

    def p_counting():
        out = _text(_gen(url, model, COUNT_PROMPT, max_tokens=4, raw=raw), raw)
        return "4" in out, f"continuation {out[:20]!r} (want '4')"

    if wanted("A"):
        P["A"] = ("A coherence (known answers + logprob peak)", p_coherence)
    if wanted("B"):
        P["B"] = ("B determinism (same prompt twice)", p_determinism)
    if wanted("C"):
        P["C"] = ("C context-sensitivity (bigram test)", p_context)
    if wanted("D"):
        P["D"] = ("D repetition (degenerate loop)", p_repetition)
    if wanted("E"):
        P["E"] = ("E concurrency (batch-shape sensitivity)", p_concurrency)
    if wanted("F"):
        P["F"] = ("F prefix-following", p_prefix)
    if wanted("G"):
        P["G"] = ("G counting", p_counting)
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="e.g. http://nid002964:30001")
    ap.add_argument("--model", required=True, help="served model name (org/model-name)")
    ap.add_argument("--completions", action="store_true", help="use raw /v1/completions instead of chat")
    ap.add_argument("--only", help="comma list of probe letters, e.g. A,C")
    ap.add_argument("--json", action="store_true", help="print JSON results")
    args = ap.parse_args()

    only = {x.strip().upper() for x in args.only.split(",")} if args.only else None
    results = []
    # in --json mode stdout carries ONLY the JSON object; progress -> stderr
    echo = (lambda s: sys.stderr.write(s + "\n")) if args.json else print
    for key, (label, fn) in make_probes(args.url, args.model, args.completions, only).items():
        echo(f"--- {label}")
        _run(label, fn, results, echo)

    failed = [r for r in results if not r["ok"]]
    if args.json:
        # stdout is PURE JSON (pipe to jq); human summary goes to stderr
        print(json.dumps({"url": args.url, "model": args.model, "results": results,
                          "failed": len(failed), "total": len(results)}, indent=2))
        sys.stderr.write(f"probe: {len(results) - len(failed)}/{len(results)} probes passed\n")
    else:
        print(f"\n{len(results) - len(failed)}/{len(results)} probes passed.")
        if failed:
            print("failure SIGNATURE (see meta/tools/debugger/README.md): "
                  "deterministic + context-sensitive + weak peak => forward-pass numerics; "
                  "nondeterministic => race/allocator/comms; context-blind => pure prior/bigram behavior")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
