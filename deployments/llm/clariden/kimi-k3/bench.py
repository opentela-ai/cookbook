#!/usr/bin/env python3
"""Throughput vs concurrency bench against a local sglang OpenAI server.

No tokenizer, no HF downloads: token counts come from the response `usage`
field. This matters on clusters where compute nodes have no outbound internet
(e.g. JSC) — the bench needs nothing but the server and aiohttp.

Usage:
  bench.py "1:8 4:16 8:32 16:48 32:64" [host] [port] [in_tokens] [out_tokens]
  MODEL=zai-org/GLM-4.7-Flash bench.py "1:8 4:16" 127.0.0.1 30000 512 128

Each spec is CONC:NUMREQ. A warmup pass (C=4 n=8, discarded) runs first so
the first measured level sees warm CUDA graphs, JIT kernels and KV cache.
One JSON line is printed per level as it completes, so partial results
survive a timeout or kill. See meta/bench/README.md.
"""
import asyncio, aiohttp, sys, time, json, os

HOST = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
IN_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
OUT_TOK = int(sys.argv[5]) if len(sys.argv) > 5 else 256
MODEL = os.environ.get("MODEL", "moonshotai/Kimi-K3")
URL = f"http://{HOST}:{PORT}/v1/chat/completions"
# ~1.3 words/token of English; pad generously, actual count read from usage.
PROMPT = ("The quick brown fox jumps over the lazy dog. " * (IN_TOK * 2))[: IN_TOK * 7]

async def one(session, sem):
    async with sem:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": OUT_TOK,
            "ignore_eos": True,   # full-length outputs so out_tokens is controlled
            "temperature": 0.0,   # deterministic
            "stream": False,
        }
        t0 = time.time()
        async with session.post(URL, json=payload) as r:
            body = await r.json()
        lat = time.time() - t0
        u = body["usage"]
        return u["prompt_tokens"], u["completion_tokens"], lat

async def run(c, n, session, tag):
    sem = asyncio.Semaphore(c)
    t0 = time.time()
    rs = await asyncio.gather(*[one(session, sem) for _ in range(n)])
    wall = time.time() - t0
    tot_out = sum(x[1] for x in rs)
    tot_in = sum(x[0] for x in rs)
    lats = sorted(x[2] for x in rs)
    per_req = [x[1] / x[2] for x in rs]
    res = {
        "tag": tag, "model": MODEL,
        "concurrency": c, "n": n, "wall_s": round(wall, 1),
        "in_tok_avg": tot_in // n, "out_tok_total": tot_out,
        "agg_out_tok_s": round(tot_out / wall, 1),
        "per_req_out_tok_s": round(sum(per_req) / len(per_req), 1),
        "lat_p50_s": round(lats[len(lats) // 2], 1),
        "lat_max_s": round(lats[-1], 1),
    }
    print(json.dumps(res), flush=True)

async def main():
    specs = sys.argv[1].split()
    to = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=to) as s:
        print(json.dumps({"tag": "warmup", "concurrency": 4, "n": 8, "note": "discarded"}), flush=True)
        await run(4, 8, s, "warmup")
        for spec in specs:
            c, n = (int(x) for x in spec.split(":"))
            await run(c, n, s, f"measured")

asyncio.run(main())
