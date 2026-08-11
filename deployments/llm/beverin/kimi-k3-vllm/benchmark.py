#!/usr/bin/env python3
"""benchmark.py -- concurrent decode-throughput bench for Kimi-K3 on Beverin.

Companion to gen_correctness.py (the correctness gate). gen_correctness.py
sends six greedy /v1/completions SEQUENTIALLY and checks factual substrings;
THIS script sends a CONCURRENCY SWEEP of /v1/completions and reports
aggregate + per-request throughput + latency, so the recipe can quantify the
working (ENFORCE_EAGER=1, no-prefix-cache) configuration.

Design (mirrors deployments/llm/clariden/kimi-k3/bench.py):
  - /v1/completions (not /chat): lower overhead, no chat template or parser
    dependency, same endpoint the correctness probe uses.
  - Token counts come from the response `usage` field (prompt_tokens +
    completion_tokens), so no tokenizer / HF download is needed on the
    compute node.
  - ignore_eos: True + max_tokens=OUT_TOK so every request emits exactly
    OUT_TOK tokens (controlled output length for clean throughput math).
  - temperature: 0 (deterministic; throughput is independent of sampling).
  - A warmup pass (C=4 N=8, discarded) runs first so the first measured
    level sees warm JIT kernels and KV cache (enforce_eager re-JITs on the
    first real batch; with --enforce-eager there is no CUDA graph to warm).
  - One JSON line is printed per level as it completes, so partial results
    survive a timeout or kill.

Each spec in SPEC is CONC:NUMREQ -- NUMREQ requests held at CONCURRENCY via a
semaphore. Aggregate throughput = total_completion_tokens / wall. Per-request
decode rate = completion_tokens / per-request-wall (median across the level).

Usage (run inside the kimi-k3-vllm container on the head; host netns is shared
so 127.0.0.1:$SERVE_PORT reaches vLLM on 0.0.0.0):

    python3 benchmark.py <base_url> <model> <out_json> <spec> [out_tok] [in_tok]

    python3 benchmark.py http://127.0.0.1:8080 SwissAI-Research/moonshot/kimi-k3-rocm \\
        run-589999/benchmark_589999.json "1:16 8:32 32:64 64:64" 256 32

Env (defaults, overridden by argv where applicable):
  BENCH_TIMEOUT          overall wall budget, s (default 1800)
  BENCH_PER_REQ_TIMEOUT  per-request aiohttp timeout, s (default 600)
"""
import asyncio
import json
import os
import sys
import time

try:
    import aiohttp
except ImportError:  # pragma: no cover -- container always has aiohttp
    print("FATAL: aiohttp not found (vLLM image bundles it). Run inside the container.", file=sys.stderr)
    raise SystemExit(2)

# ~1.3 words/token of English; pad generously, real count read from usage.
def _prompt(in_tok):
    base = "The quick brown fox jumps over the lazy dog. "
    return (base * (in_tok * 4))[: in_tok * 7]


async def _one(session, url, model, prompt, out_tok, per_req):
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": out_tok,
        "ignore_eos": True,   # full-length outputs so out_tokens is controlled
        "stream": False,
    }
    t0 = time.time()
    async with session.post(url, json=payload) as r:
        body = await r.json()
    lat = time.time() - t0
    u = body.get("usage") or {}
    return int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), lat, body


async def _run(c, n, session, url, model, prompt, out_tok, per_req, tag):
    sem = asyncio.Semaphore(c)
    async def bound():
        async with sem:
            return await _one(session, url, model, prompt, out_tok, per_req)
    t0 = time.time()
    rs = await asyncio.gather(*[bound() for _ in range(n)])
    wall = time.time() - t0
    ok = [x for x in rs if x[1] > 0]
    if not ok:
        res = {"tag": tag, "concurrency": c, "n": n, "wall_s": round(wall, 1),
               "ok": 0, "error": "; ".join(str(x[3].get("error", x[3]))[:120] for x in rs[:3])}
        print(json.dumps(res), flush=True)
        return res
    tot_out = sum(x[1] for x in ok)
    tot_in = sum(x[0] for x in ok)
    lats = sorted(x[2] for x in ok)
    per_req = [x[1] / x[2] for x in ok if x[2] > 0]
    res = {
        "tag": tag, "model": model,
        "concurrency": c, "n": n, "ok": len(ok), "wall_s": round(wall, 1),
        "in_tok_avg": (tot_in // len(ok)) if ok else 0,
        "out_tok_total": tot_out,
        "out_tok_per_req": (tot_out // len(ok)) if ok else 0,
        "agg_out_tok_s": round(tot_out / wall, 1),
        "per_req_out_tok_s_med": round(sorted(per_req)[len(per_req) // 2], 1) if per_req else 0,
        "lat_p50_s": round(lats[len(lats) // 2], 1),
        "lat_max_s": round(lats[-1], 1),
    }
    print(json.dumps(res), flush=True)
    return res


async def main():
    if len(sys.argv) < 5:
        print("usage: benchmark.py <base_url> <model> <out_json> <spec> [out_tok] [in_tok]",
              file=sys.stderr)
        return 2
    base_url = sys.argv[1].rstrip("/")
    model = sys.argv[2]
    out_json = sys.argv[3]
    spec = sys.argv[4]
    out_tok = int(sys.argv[5]) if len(sys.argv) > 5 else int(os.environ.get("BENCH_OUT_TOK", "256"))
    in_tok = int(sys.argv[6]) if len(sys.argv) > 6 else int(os.environ.get("BENCH_IN_TOK", "32"))
    overall = float(os.environ.get("BENCH_TIMEOUT", "1800"))
    per_req = float(os.environ.get("BENCH_PER_REQ_TIMEOUT", "600"))

    url = f"{base_url}/v1/completions"
    prompt = _prompt(in_tok)
    levels = []
    for s in spec.split():
        c, n = (int(x) for x in s.split(":"))
        levels.append((c, n))

    print(f"[BENCH] {url} model={model} out_tok={out_tok} in_tok={in_tok} "
          f"levels={[f'{c}:{n}' for c, n in levels]} budget={overall:g}s", flush=True)

    t_start = time.time()
    results = []
    timeout = aiohttp.ClientTimeout(total=per_req)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Warmup (discarded) -- first batch triggers any remaining JIT.
        try:
            await _run(4, 8, session, url, model, prompt, out_tok, per_req, "warmup")
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"tag": "warmup", "error": f"{type(exc).__name__}: {exc}"}), flush=True)

        for c, n in levels:
            if time.time() - t_start > overall:
                print(json.dumps({"tag": "skipped", "concurrency": c, "n": n,
                                   "error": f"overall budget {overall:g}s exceeded"}), flush=True)
                results.append({"tag": "skipped", "concurrency": c, "n": n})
                continue
            try:
                results.append(await _run(c, n, session, url, model, prompt, out_tok, per_req, "measured"))
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"tag": "measured", "concurrency": c, "n": n,
                                   "error": f"{type(exc).__name__}: {exc}"}), flush=True)
                results.append({"tag": "measured", "concurrency": c, "n": n,
                                 "error": f"{type(exc).__name__}: {exc}"})

    measured = [r for r in results if r.get("tag") == "measured" and r.get("ok", 0) > 0]
    peak = max((r.get("agg_out_tok_s", 0) for r in measured), default=0)
    summary = {
        "base_url": base_url,
        "model": model,
        "out_tok": out_tok,
        "in_tok": in_tok,
        "spec": spec,
        "levels_measured": len(measured),
        "peak_agg_out_tok_s": peak,
        "elapsed_s": round(time.time() - t_start, 1),
        "results": results,
    }
    if out_json:
        try:
            with open(out_json, "w") as fh:
                json.dump(summary, fh, indent=2)
            print(f"[BENCH] summary -> {out_json}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[BENCH] WARN: could not write {out_json}: {exc}", file=sys.stderr)
    print(f"[BENCH] peak_agg={peak:g} tok/s over {len(measured)} measured levels "
          f"(elapsed {summary['elapsed_s']}s)", flush=True)
    return 0 if measured else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
