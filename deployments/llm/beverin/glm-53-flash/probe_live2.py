#!/usr/bin/env python
"""Round 2: batch/concurrency sensitivity + length sensitivity."""
import concurrent.futures as cf
import json
import urllib.request

URL = "http://127.0.0.1:30000/v1/completions"
MODEL = "zai-org/GLM-5.3-Flash"


def post(payload):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.load(r)


def gen(prompt, max_tokens=6):
    r = post({"model": MODEL, "prompt": prompt,
              "max_tokens": max_tokens, "temperature": 0})
    return r["choices"][0]["text"]


print("=== G: 4 identical requests CONCURRENTLY (batch-shape sensitivity) ===")
with cf.ThreadPoolExecutor(4) as ex:
    outs = list(ex.map(lambda i: gen("The capital of France is"), range(4)))
for i, o in enumerate(outs):
    print(f"  concurrent run{i+1}: {o!r}")

print("\n=== H: same core prompt, increasing irrelevant prefix ===")
base = "Answer with one word. The capital of France is"
for pad in ["", "x ", ("lorem ipsum " * 8), ("lorem ipsum " * 40)]:
    p = pad + base
    try:
        o = gen(p, max_tokens=4)
        print(f"  prefix_chars={len(pad):>4} prompt_toks~{len(p)//4:>3}: {o!r}")
    except Exception as exc:
        print(f"  prefix_chars={len(pad):>4} FAILED: {exc!r}")

print("\n=== I: '1 2 3' counting (tiny-context sanity) ===")
for p in ["1 2 3", "one two three, four,", "A B C D"]:
    try:
        o = gen(p, max_tokens=4)
        print(f"  {p!r} -> {o!r}")
    except Exception as exc:
        print(f"  {p!r} FAILED: {exc!r}")
