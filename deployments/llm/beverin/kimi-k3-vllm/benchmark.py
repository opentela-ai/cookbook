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


async def _sweep(base_url, model, levels, out_tok, in_tok, overall, per_req,
                  session, tag):
    """Run warmup + the concurrency spec against one server; return a dict.

    Refactor of the original inline loop so main() can run the SAME spec
    against a second KDA-disabled baseline server (issue #42) without
    duplicating the warmup/budget/skip logic.
    """
    url = f"{base_url}/v1/completions"
    prompt = _prompt(in_tok)
    t_start = time.time()
    results = []
    try:
        await _run(4, 8, session, url, model, prompt, out_tok, per_req, "warmup")
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"tag": "warmup", "server": tag,
                           "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    for c, n in levels:
        if time.time() - t_start > overall:
            print(json.dumps({"tag": "skipped", "server": tag, "concurrency": c,
                               "n": n, "error": f"overall budget {overall:g}s exceeded"}),
                  flush=True)
            results.append({"tag": "skipped", "server": tag, "concurrency": c, "n": n})
            continue
        try:
            results.append(await _run(c, n, session, url, model, prompt, out_tok, per_req, "measured"))
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"tag": "measured", "server": tag, "concurrency": c,
                               "n": n, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
            results.append({"tag": "measured", "server": tag, "concurrency": c,
                             "error": f"{type(exc).__name__}: {exc}"})
    return {"tag": tag, "t_start": t_start, "results": results}


async def main():
    if len(sys.argv) < 5:
        print("usage: benchmark.py <base_url> <model> <out_json> <spec> "
              "[out_tok] [in_tok] [base_url_kda_off]", file=sys.stderr)
        return 2
    base_url = sys.argv[1].rstrip("/")
    model = sys.argv[2]
    out_json = sys.argv[3]
    spec = sys.argv[4]
    out_tok = int(sys.argv[5]) if len(sys.argv) > 5 else int(os.environ.get("BENCH_OUT_TOK", "256"))
    in_tok = int(sys.argv[6]) if len(sys.argv) > 6 else int(os.environ.get("BENCH_IN_TOK", "32"))
    # KDA compare (issue #42): a SECOND base_url for the KDA-disabled
    # baseline (TRITON_MLA + K3_DISABLE_KDA=1). When set, the SAME spec is
    # run against both servers and a per-level throughput + per-request
    # latency delta is reported, so the operator can verify the acceptance
    # "no regression in per-request latency vs the KDA-disabled baseline
    # (target <=1.05x)" and the quality delta (gen_correctness.py with
    # KDA_RECALL_PROBE=1). Defaults to the env so existing one-server
    # launches are unchanged.
    base_off = (sys.argv[7] if len(sys.argv) > 7 else "").rstrip("/") or \
        os.environ.get("BENCH_BASE_KDA_OFF", "").rstrip("/")
    overall = float(os.environ.get("BENCH_TIMEOUT", "1800"))
    per_req = float(os.environ.get("BENCH_PER_REQ_TIMEOUT", "600"))

    levels = []
    for s in spec.split():
        c, n = (int(x) for x in s.split(":"))
        levels.append((c, n))

    print(f"[BENCH] {base_url}/v1/completions model={model} out_tok={out_tok} "
          f"in_tok={in_tok} levels={[f'{c}:{n}' for c, n in levels]} "
          f"budget={overall:g}s" + (f"\n[BENCH] KDA compare vs baseline "
          f"{base_off}/v1/completions (KDA-disabled)" if base_off else ""),
          flush=True)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=per_req)) as session:
        sweep_on = await _sweep(base_url, model, levels, out_tok, in_tok,
                                overall, per_req, session, "kda_on")
        sweep_off = None
        if base_off:
            sweep_off = await _sweep(base_off, model, levels, out_tok, in_tok,
                                     overall, per_req, session, "kda_off")

    measured_on = [r for r in sweep_on["results"] if r.get("tag") == "measured" and r.get("ok", 0) > 0]
    peak_on = max((r.get("agg_out_tok_s", 0) for r in measured_on), default=0)
    summary = {
        "base_url": base_url,
        "base_url_kda_off": base_off or None,
        "model": model,
        "out_tok": out_tok,
        "in_tok": in_tok,
        "spec": spec,
        "levels_measured": len(measured_on),
        "peak_agg_out_tok_s": peak_on,
        "elapsed_s": round(time.time() - sweep_on["t_start"], 1),
        "results": sweep_on["results"],
    }
    if sweep_off is not None:
        measured_off = [r for r in sweep_off["results"] if r.get("tag") == "measured" and r.get("ok", 0) > 0]
        peak_off = max((r.get("agg_out_tok_s", 0) for r in measured_off), default=0)
        summary["kda_off"] = {
            "peak_agg_out_tok_s": peak_off,
            "levels_measured": len(measured_off),
            "results": sweep_off["results"],
        }
        # Per-level throughput + per-request latency delta vs the KDA-disabled
        # baseline (issue #42 acceptance: <=1.05x per-request latency).
        by_conc_on = {r["concurrency"]: r for r in measured_on}
        by_conc_off = {r["concurrency"]: r for r in measured_off}
        delta = []
        for c, _ in levels:
            ro, rf = by_conc_on.get(c), by_conc_off.get(c)
            if not (ro and rf):
                continue
            lat_on = ro.get("lat_p50_s") or 0
            lat_off = rf.get("lat_p50_s") or 0
            agg_on = ro.get("agg_out_tok_s") or 0
            agg_off = rf.get("agg_out_tok_s") or 0
            delta.append({
                "concurrency": c,
                "lat_p50_on_s": lat_on, "lat_p50_off_s": lat_off,
                "lat_ratio_on_over_off": round(lat_on / lat_off, 3) if lat_off else None,
                "agg_on_tok_s": agg_on, "agg_off_tok_s": agg_off,
                "agg_ratio_on_over_off": round(agg_on / agg_off, 3) if agg_off else None,
            })
        summary["kda_delta"] = delta
        print("[BENCH] KDA vs KDA-disabled baseline (lat_ratio = KDA-on / KDA-off; "
              "acceptance <=1.05x):", flush=True)
        for d in delta:
            lr = d["lat_ratio_on_over_off"]
            ar = d["agg_ratio_on_over_off"]
            print(f"  C={d['concurrency']:<3} lat_p50 on={d['lat_p50_on_s']:>5}s "
                  f"off={d['lat_p50_off_s']:>5}s ratio={lr} | agg on={d['agg_on_tok_s']:>6} "
                  f"off={d['agg_off_tok_s']:>6} ratio={ar}", flush=True)

    if out_json:
        try:
            with open(out_json, "w") as fh:
                json.dump(summary, fh, indent=2)
            print(f"[BENCH] summary -> {out_json}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[BENCH] WARN: could not write {out_json}: {exc}", file=sys.stderr)
    print(f"[BENCH] peak_agg={peak_on:g} tok/s over {len(measured_on)} measured levels "
          f"(elapsed {summary['elapsed_s']}s)" +
          (f"; kda_off peak={summary['kda_off']['peak_agg_out_tok_s']:g} tok/s"
           if sweep_off is not None else ""), flush=True)
    return 0 if measured_on else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
