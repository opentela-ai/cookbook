#!/usr/bin/env python
"""Live-server diagnostic probes against a running SGLang on 127.0.0.1:30000.

C: echo prompt-logprobs  -> is the prefill forward sane on its own input text?
D: same prompt x3 greedy -> is decode deterministic at temp=0 (state/race)?
E: retrieval             -> can attention copy an answer from context?
"""
import json
import urllib.request

URL = "http://127.0.0.1:30000/v1/completions"
MODEL = "zai-org/GLM-5.3-Flash"


def post(payload):
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def show_prompt_logprobs(resp, max_show=40):
    c = resp["choices"][0]
    lp = c.get("logprobs") or {}
    toks = lp.get("tokens", [])
    tops = lp.get("top_logprobs", [])
    n = min(len(toks), max_show)
    for i in range(n):
        info = tops[i] if i < len(tops) else {}
        v = list(info.values())[0] if info else float("nan")
        print(f"  {i:>3}: {toks[i]!r:>16} logprob={v:8.3f}")
    print("  gen:", repr(c.get("text", "")))


print("=== C: echo prompt-logprobs (prefill sanity on own input) ===")
try:
    r = post({
        "model": MODEL,
        "prompt": "Paris is the capital of France. The capital of France is",
        "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 1,
    })
    show_prompt_logprobs(r)
except Exception as exc:
    print("  FAILED:", repr(exc))

print("\n=== D: same prompt x3 greedy (determinism / state corruption) ===")
for i in range(3):
    try:
        r = post({"model": MODEL, "prompt": "The capital of France is",
                  "max_tokens": 6, "temperature": 0})
        print(f"  run{i+1}: {r['choices'][0]['text']!r}")
    except Exception as exc:
        print(f"  run{i+1} FAILED:", repr(exc))

print("\n=== E: in-context retrieval (copy 'Paris' from prompt) ===")
try:
    r = post({
        "model": MODEL,
        "prompt": "Q: What is the capital of France? A: Paris.\nQ: What is the capital of Germany? A:",
        "max_tokens": 4, "temperature": 0, "logprobs": 3,
    })
    c = r["choices"][0]
    lp = c.get("logprobs") or {}
    for i, (t, info) in enumerate(zip(lp.get("tokens", []), lp.get("top_logprobs", []))):
        tops = " | ".join(f"{k!r}:{v:.2f}" for k, v in list(info.items())[:3])
        print(f"  step{i}: chose={t!r}  top: {tops}")
    print("  text:", repr(c["text"]))
except Exception as exc:
    print("  FAILED:", repr(exc))

print("\n=== F: repetition (trivial continuation) ===")
try:
    r = post({"model": MODEL, "prompt": "a a a a a a a a a a",
              "max_tokens": 4, "temperature": 0, "logprobs": 3})
    c = r["choices"][0]
    lp = c.get("logprobs") or {}
    for i, (t, info) in enumerate(zip(lp.get("tokens", []), lp.get("top_logprobs", []))):
        tops = " | ".join(f"{k!r}:{v:.2f}" for k, v in list(info.items())[:3])
        print(f"  step{i}: chose={t!r}  top: {tops}")
except Exception as exc:
    print("  FAILED:", repr(exc))
