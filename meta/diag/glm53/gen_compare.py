#!/usr/bin/env python3
"""Deterministic generation probe for GLM-5.3 cross-arch output comparison.

Sends greedy (temperature=0) completions to a server and dumps the generated
text + token ids, so beverin vs clariden can be compared token-for-token.

Usage: gen_compare.py <URL> <MODEL_NAME> [OUT_JSON]

Uses /v1/completions (GLM-5.3 Mamba/linear-attn hybrid — the running server
already accepted POST /v1/completions at 00:55:38). Greedy decoding makes
output deterministic for a fixed prompt, so any difference between arches is
a real correctness signal (not sampling noise).
"""
import json
import sys
import time
import urllib.request
import urllib.error


PROMPTS = [
    "The capital of France is",
    "The color of the sky on a clear day is",
    "Two plus two equals",
    "The largest planet in our solar system is",
    "The chemical symbol for water is",
]


def gen(url, model, prompt, max_tokens=16):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "top_p": 1.0,
        "ignore_eos": False,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read())
    dt = time.time() - t0
    text = out.get("choices", [{}])[0].get("text", "")
    tok_ids = out.get("choices", [{}])[0].get("token_ids") or \
              out.get("output_token_ids", [[]])[0] if out.get("output_token_ids") else None
    finish = out.get("choices", [{}])[0].get("finish_reason")
    usage = out.get("usage", {})
    return {
        "prompt": prompt,
        "text": text,
        "token_ids": tok_ids,
        "finish_reason": finish,
        "usage": usage,
        "wall_s": round(dt, 2),
    }


def main():
    url = sys.argv[1]
    model = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"# gen_compare: url={url} model={model} greedy temp=0")
    results = []
    for p in PROMPTS:
        try:
            r = gen(url, model, p)
            print(f"  [{p!r:45s}] -> {r['text']!r}  (finish={r['finish_reason']}, "
                  f"toks={r['usage'].get('completion_tokens')}, {r['wall_s']}s)")
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{p!r:45s}] -> ERROR {exc!r}")
            results.append({"prompt": p, "error": repr(exc)})

    payload = {"url": url, "model": model, "results": results}
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"# wrote {out_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
