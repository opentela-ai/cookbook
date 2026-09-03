#!/usr/bin/env python3
"""capture_probe.py -- fire ONE deterministic long-prompt request so
comp_capture (GLM53_COMP_MIN_TOKENS gate) lands on THIS forward on BOTH
machines.

Why: sglang's init-time profile/warmup forwards have machine-dependent batch
sizes (beverin dummy=1 token, clariden dummy=9 tokens at bs=9) and can
therefore consume the one-shot capture latch BEFORE the first real request.
A ~2k-token prompt is far above any init forward and far below
chunked_prefill_size, so with MIN_TOKENS=1200 this request is the ONLY
forward that can arm the capture -- identically on beverin and clariden.

Usage: capture_probe.py <base_url> <model>
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "zai-org/GLM-5.3-Flash"

# Deterministic prompt: same tokenizer on both machines -> identical ids.
# ~2250 tokens (> MIN_TOKENS=1200, << chunked_prefill 8192).
SENT = "The quick brown fox jumps over the lazy dog. "
PROMPT = SENT * 250 + "The capital of France is"


def _wait_health(timeout=7200):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(10)
    return False


def main():
    if not _wait_health():
        print("capture_probe: health never came up", file=sys.stderr)
        return 1
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": PROMPT,
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    txt = out["choices"][0]["text"]
    n = out.get("usage", {}).get("prompt_tokens", -1)
    print(f"capture_probe: fired {n}-token prefill in {time.time()-t0:.1f}s; "
          f"continuation={txt!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
