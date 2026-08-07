#!/usr/bin/env python3
"""HiCache (host-memory / Grace-LPDDR tier) functional test for Kimi-K3 on Clariden.

Protocol (assumes the engine was started with HICACHE_ENABLE=1 and a
deliberately small HBM radix pool, e.g. the recipe's test submit:

  sbatch --export=ALL,HICACHE_ENABLE=1,OTELA_SERVICE_NAME=llm-hicache-test,\
OTELA_SEED=4213,CTX_LEN=131072,SGLANG_EXTRA_ARGS="--max-total-tokens 262144" \
  serve_kimi_k3_otela_clariden.sbatch

):
  P1  MISS       send probe prefix P (cold)                -> TTFT ~ full prefill
  P2  HBM HIT    send P again                              -> TTFT << P1 (radix hit),
                   also marks P "reused" so write_through_selective backs it up
  FILL           evict P from HBM with >pool unique tokens (--max-total-tokens)
  P3  HOST HIT?  send P again                              -> TTFT << P1 means the
                   prefix came back from the LPDDR host tier (C2C, not recompute)

VERDICT: HOST-TIER HIT iff T3 < 60% of T1. Also prints sglang /metrics cache
counters before/after P3 when present.

Usage: python3 hicache_bench.py [http://HEAD_IP:PORT]  (runs from login node;
stdlib only so it also runs inside the container).
"""
import concurrent.futures as cf
import json
import os
import random
import sys
import time
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000").rstrip("/")
MODEL = "moonshotai/Kimi-K3"
# Env-tunable. Filler must EXCEED the HBM pool (mem-fraction sized
# max_total_tokens) to force eviction: 480k tokens vs the 262k test pool,
# ~4.5M vs the default 3.73M prod pool.
PROBE_TOKENS = int(os.environ.get("HICB_PROBE_TOKENS", "6000"))
FILLER_TOKENS = int(os.environ.get("HICB_FILLER_TOKENS", str(40 * 12000)))
FILLER_CONC = int(os.environ.get("HICB_FILLER_CONCURRENCY", "8"))

# --- synthetic prose: numbered sentences => unique, tokenizer-friendly ------
def prose(n_words, tag):
    words = []
    i = 0
    while len(words) < n_words:
        i += 1
        words += (f"{tag} paragraph {i} of the synthetic hicache benchmark corpus "
                  f"contains deliberately ordinary english words so the causal "
                  f"language model sees realistic text rather than repeated "
                  f"tokens that would compress or merge across bpe boundaries . ").split()
    return " ".join(words[:n_words])

# ~0.75 words/token for tiktoken-class tokenizers; server returns exact counts.
# Salt every run: a rerun with identical texts is 100% served from the HiCache
# host tier (observed: 442k "prefill" tokens in 2.3 s) — useless as a bench.
SALT = "%x" % random.getrandbits(48)
PROBE = prose(int(PROBE_TOKENS * 0.75), "PROBE-" + SALT)

def chat(prompt, max_tokens=1):
    # /v1/completions, NOT chat: the K3 chat template injects a variable
    # system header (date/time), which breaks token-prefix equality between
    # requests and defeats radix matching (verified on clariden 2026-08-05:
    # identical chat prompts never hit; raw completions hit on both engines).
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
            "prompt": prompt}
    req = urllib.request.Request(URL + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return time.monotonic() - t0, d["usage"]["prompt_tokens"]

def cache_metrics():
    try:
        with urllib.request.urlopen(URL + "/metrics", timeout=10) as r:
            txt = r.read().decode()
    except Exception as e:
        return f"(metrics unavailable: {e})"
    out = []
    for line in txt.splitlines():
        l = line.lower()
        if ("cache" in l or "hicache" in l) and not line.startswith("#"):
            out.append(line)
    return "\n".join(out) if out else "(no cache metrics lines)"

def main():
    print(f"== HiCache bench against {URL}  model={MODEL}")
    print(f"== probe ~{PROBE_TOKENS} tok; filler {FILLER_TOKENS} tok @ C={FILLER_CONC}")

    t1, ptok = chat(PROBE + "\nQuestion: summarize in one word.")
    print(f"P1 MISS       : {t1:7.3f}s  prompt_tokens={ptok}")

    t2, _ = chat(PROBE + "\nQuestion: summarize in one sentence.")
    print(f"P2 HBM HIT    : {t2:7.3f}s  (expect << P1; marks probe as reused)")

    print(f"FILL          : evicting HBM pool with {FILLER_TOKENS} unique tokens ...",
          flush=True)
    f0 = time.monotonic()
    with cf.ThreadPoolExecutor(FILLER_CONC) as ex:
        futs = [ex.submit(chat, prose(9000, f"FILLER-{SALT}-{k}"), 1)
                for k in range(FILLER_TOKENS // 12000)]
        done = sum(f.result()[1] for f in futs)
    print(f"FILL done     : {done} unique tokens in {time.monotonic()-f0:.1f}s")
    time.sleep(5)  # allow async write-back to the host tier to drain

    m_before = cache_metrics()
    t3, _ = chat(PROBE + "\nQuestion: summarize in one phrase.")
    print(f"P3 POST-EVICT : {t3:7.3f}s")
    m_after = cache_metrics()

    print("\n--- sglang cache metrics (before P3) ---\n" + m_before)
    print("\n--- sglang cache metrics (after  P3) ---\n" + m_after)

    ratio = t3 / t1 if t1 > 0 else 1.0
    print(f"\nT1(miss)={t1:.3f}s  T2(hbm)={t2:.3f}s  T3(post-evict)={t3:.3f}s"
          f"  T3/T1={ratio:.2f}")
    if t2 < 0.6 * t1 and ratio < 0.6:
        print("VERDICT: HOST-TIER HIT — prefix survived HBM eviction (HiCache works)")
    elif t2 < 0.6 * t1 and ratio >= 0.6:
        print("VERDICT: HBM hit OK but NO host-tier hit — prefix was recomputed "
              "after eviction (HiCache not storing/restoring)")
    else:
        print("VERDICT: inconclusive — even P2 was not a radix hit; "
              "is the radix cache disabled?")

if __name__ == "__main__":
    main()
