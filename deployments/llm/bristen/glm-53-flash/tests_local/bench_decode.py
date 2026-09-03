#!/usr/bin/env python3
"""bench_decode.py -- TPOT/TTFT/throughput probe for the bristen GLM-5.3 serve.

Usage: python3 bench_decode.py <base_url> [out_json]
Mirrors the job-83152 serving numbers (bs=1 TPOT/TTFT, bs=8 aggregate).
"""
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:30000"
MODEL = "zai-org/GLM-5.3-Flash"
PROMPT = "The history of computing is a long and rich story. It begins with"
MAX_TOKENS = 64


def stream_one(results, idx, barrier=None):
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": PROMPT,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip().decode()
                if payload == "[DONE]":
                    break
                d = json.loads(payload)
                if d.get("choices") and d["choices"][0].get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    ntok += 1
    except Exception as e:  # noqa: BLE001
        results[idx] = {"error": str(e)}
        return
    total = time.perf_counter() - t0
    results[idx] = {
        "ttft": ttft,
        "total": total,
        "ntok": ntok,
        "tpot": (total - ttft) / max(ntok - 1, 1),
    }


def bench_bs1(runs=3):
    out = []
    for _ in range(runs):
        res = [None]
        stream_one(res, 0)
        if res[0] is None or "error" in res[0]:
            out.append(res[0])
        else:
            out.append(res[0])
        time.sleep(1)
    return out


def bench_bs8(runs=2):
    out = []
    for _ in range(runs):
        results = [None] * 8
        threads = [threading.Thread(target=stream_one, args=(results, i)) for i in range(8)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        ok = [r for r in results if r and "error" not in r]
        agg_toks = sum(r["ntok"] for r in ok)
        out.append(
            {
                "wall": wall,
                "aggregate_tok_s": agg_toks / wall if wall else 0,
                "mean_tpot": (sum(r["tpot"] for r in ok) / len(ok)) if ok else None,
                "n_ok": len(ok),
            }
        )
        time.sleep(2)
    return out


if __name__ == "__main__":
    report = {"base": BASE, "prompt_tokens_est": 16, "max_tokens": MAX_TOKENS}
    b1 = bench_bs1()
    good1 = [r for r in b1 if r and "error" not in r]
    report["bs1"] = b1
    if good1:
        report["bs1_median"] = {
            k: sorted(r[k] for r in good1)[len(good1) // 2] for k in ("ttft", "tpot", "total")
        }
    b8 = bench_bs8()
    report["bs8"] = b8
    print(json.dumps(report, indent=2))
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            json.dump(report, f, indent=2)
