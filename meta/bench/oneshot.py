#!/usr/bin/env python3
"""One-shot latency probe (stdlib only — no aiohttp, no internet).

The smallest possible request: a fixed prompt, `max_tokens` short, temperature
0. Prints wall time and decode tok/s from the `usage` field. Use it to (a)
confirm the server is up and routing, (b) get a single-request latency floor
BEFORE the full sweep. A single-shot number is NOT a serving number for
distributed topologies (see meta/bench/README.md §"The C=1 rule") — it is a health
+ latency-floor check, nothing more.

Usage:
  oneshot.py [host] [port] [in_tokens] [out_tokens]
  MODEL=zai-org/GLM-4.7-Flash oneshot.py 127.0.0.1 30000 64 16
"""
import sys, os, time, urllib.request, json
HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
IN_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 64
OUT_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 16
MODEL = os.environ.get("MODEL", "moonshotai/Kimi-K3")
url = f"http://{HOST}:{PORT}/v1/chat/completions"
payload = {"model": MODEL,
           "messages": [{"role": "user", "content": "x " * IN_TOK}],
           "max_tokens": OUT_TOK, "temperature": 0, "ignore_eos": True}
data = json.dumps(payload).encode()
t0 = time.time()
r = urllib.request.urlopen(urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}), timeout=110)
body = json.loads(r.read())
dt = time.time() - t0
u = body["usage"]
print(f"model={MODEL} wall={dt:.2f}s in={u['prompt_tokens']} "
      f"out={u['completion_tokens']} decode_tok/s={u['completion_tokens']/dt:.2f}")
