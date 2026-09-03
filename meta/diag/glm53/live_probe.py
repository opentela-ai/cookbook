#!/usr/bin/env python3
"""live_probe.py -- sanity probes against a running SGLang server.

Fire at a live endpoint (no engine restart needed) to characterise HOW a
model is broken before bisecting WHERE:

  C: echo prompt-logprobs  -> is the prefill forward sane on its own input text?
  D: same prompt x3 greedy -> is decode deterministic at temp=0 (state/race)?
  E: retrieval             -> can attention copy an answer from context?
  F: repetition            -> trivial continuation sanity
  --batch adds the batch/length sensitivity round:
  G: 4 identical requests concurrently -> batch-shape sensitivity
  H: growing irrelevant prefix        -> length sensitivity
  I: tiny-context counting            -> tokenizer/context sanity

Usage: live_probe.py [base_url] [--model ID] [--batch]
"""
import argparse
import concurrent.futures as cf
import json
import sys
import urllib.request


def _post(url, model, payload):
    req = urllib.request.Request(
        url + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.load(r)


def _show_toplogprobs(resp, n_steps=None):
    c = resp["choices"][0]
    lp = c.get("logprobs") or {}
    toks, tops = lp.get("tokens", []), lp.get("top_logprobs", [])
    steps = range(min(len(toks), n_steps if n_steps is not None else len(toks)))
    for i in steps:
        info = tops[i] if i < len(tops) else {}
        tops_s = " | ".join(f"{k!r}:{v:.2f}" for k, v in list(info.items())[:3])
        print(f"  step{i}: chose={toks[i]!r}  top: {tops_s}")
    print("  text:", repr(c.get("text", "")))


def _gen(url, model, prompt, max_tokens=6):
    r = _post(url, model, {"model": model, "prompt": prompt,
                           "max_tokens": max_tokens, "temperature": 0})
    return r["choices"][0]["text"]


def _run(label, fn):
    print(f"=== {label} ===")
    try:
        fn()
    except Exception as exc:
        print("  FAILED:", repr(exc))
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base_url", nargs="?", default="http://127.0.0.1:30000")
    p.add_argument("--model", default="zai-org/GLM-5.3-Flash")
    p.add_argument("--batch", action="store_true",
                   help="also run the batch/length sensitivity round (G/H/I)")
    args = p.parse_args()
    url, model = args.base_url.rstrip("/"), args.model

    def c_echo():
        r = _post(url, model, {
            "model": model,
            "prompt": "Paris is the capital of France. The capital of France is",
            "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 1,
        })
        c = r["choices"][0]
        lp = c.get("logprobs") or {}
        toks, tops = lp.get("tokens", []), lp.get("top_logprobs", [])
        for i in range(min(len(toks), 40)):
            info = tops[i] if i < len(tops) else {}
            v = list(info.values())[0] if info else float("nan")
            print(f"  {i:>3}: {toks[i]!r:>16} logprob={v:8.3f}")
        print("  gen:", repr(c.get("text", "")))

    def d_determinism():
        for i in range(3):
            print(f"  run{i+1}: {_gen(url, model, 'The capital of France is')!r}")

    def e_retrieval():
        r = _post(url, model, {
            "model": model,
            "prompt": "Q: What is the capital of France? A: Paris.\n"
                      "Q: What is the capital of Germany? A:",
            "max_tokens": 4, "temperature": 0, "logprobs": 3,
        })
        _show_toplogprobs(r)

    def f_repetition():
        r = _post(url, model, {"model": model, "prompt": "a a a a a a a a a a",
                               "max_tokens": 4, "temperature": 0, "logprobs": 3})
        _show_toplogprobs(r, n_steps=4)

    _run("C: echo prompt-logprobs (prefill sanity on own input)", c_echo)
    _run("D: same prompt x3 greedy (determinism / state corruption)", d_determinism)
    _run("E: in-context retrieval (copy 'Paris' from prompt)", e_retrieval)
    _run("F: repetition (trivial continuation)", f_repetition)

    if not args.batch:
        return 0

    def g_concurrent():
        with cf.ThreadPoolExecutor(4) as ex:
            outs = list(ex.map(lambda i: _gen(url, model, "The capital of France is"), range(4)))
        for i, o in enumerate(outs):
            print(f"  concurrent run{i+1}: {o!r}")

    def h_prefix():
        base = "Answer with one word. The capital of France is"
        for pad in ["", "x ", ("lorem ipsum " * 8), ("lorem ipsum " * 40)]:
            pr = pad + base
            try:
                o = _gen(url, model, pr, max_tokens=4)
                print(f"  prefix_chars={len(pad):>4} prompt_toks~{len(pr)//4:>3}: {o!r}")
            except Exception as exc:
                print(f"  prefix_chars={len(pad):>4} FAILED: {exc!r}")

    def i_counting():
        for pr in ["1 2 3", "one two three, four,", "A B C D"]:
            print(f"  {pr!r} -> {_gen(url, model, pr, max_tokens=4)!r}")

    _run("G: 4 identical requests CONCURRENTLY (batch-shape sensitivity)", g_concurrent)
    _run("H: same core prompt, increasing irrelevant prefix", h_prefix)
    _run("I: '1 2 3' counting (tiny-context sanity)", i_counting)
    return 0


if __name__ == "__main__":
    sys.exit(main())
