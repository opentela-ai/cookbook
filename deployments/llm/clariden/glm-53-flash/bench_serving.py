#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concurrency sweep benchmark for an OpenAI-compatible sglang server.

Self-contained (python3 + requests only). Streams /v1/chat/completions with
stream_options.include_usage, measures per-request TTFT / TPOT / e2e latency
and aggregate output+total throughput per concurrency level, and cross-checks
against the ENGINE's own "Decode batch ... gen throughput (token/s)" log lines
(parsed from the sglang serve log between the wall-clock window of each level).

Typical use (from a host that can reach the engine, e.g. the cluster login
node when the engine runs on a compute node):

  python3 bench_serving.py \
    --url http://172.28.37.184:30000/v1/chat/completions \
    --model zai-org/GLM-5.3-Flash \
    --levels 1,2,4,8,16,32,64 \
    --serve-log /capstor/scratch/cscs/xyao/glm-53-flash/logs/serve-3270164.out \
    --save bench_results.json

Python 3.6 compatible (no walrus, no statistics.quantiles).
"""
import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# ---------------------------------------------------------------- prompt ----
PROMPT_BASE = (
    "Distributed systems rely on consistency models to describe the guarantees "
    "a datastore makes to its clients. Linearizability is the strongest single "
    "object model: every operation appears to take effect atomically at some "
    "moment between its invocation and its response, and every process observes "
    "the same interleaving. Sequential consistency relaxes this by dropping the "
    "real-time ordering requirement, while causal consistency keeps only the "
    "happens-before relation that clients can themselves observe. Weaker models "
    "trade immediacy for availability and partition tolerance, which is why "
    "production databases expose tunable quorum settings and read-repair paths "
    "so operators can pick a point on that trade-off curve per workload. "
    "Understanding these models matters when reasoning about replication "
    "protocols, leader election, and conflict-free replicated data types. "
)

REQ_PLAN = {1: 12, 2: 16, 4: 24, 8: 32, 16: 32, 32: 48, 64: 64}


def build_prompt(target_tokens):
    """Approximate a prompt of ~target_tokens (4 chars/token heuristic)."""
    target_chars = int(target_tokens * 4)
    buf = []
    n = 0
    while n < target_chars:
        buf.append(PROMPT_BASE)
        n += len(PROMPT_BASE)
    return "".join(buf)[:target_chars]


def pct(vals, p):
    """Percentile on a small sample (linear interpolation)."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * p
    f = int(math.floor(k))
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


class LogTail(object):
    """Incremental reader of the sglang serve log (shared scratch FS)."""

    DECODE_RE = re.compile(r"Decode batch")
    RUN_RE = re.compile(r"#running-req:\s*(\d+)")
    TPS_RE = re.compile(r"gen throughput \(token/s\):\s*([0-9.]+)")

    def __init__(self, path):
        self.path = path
        try:
            self.pos = os.path.getsize(path)
        except OSError:
            self.pos = 0

    def new_lines(self):
        try:
            size = os.path.getsize(self.path)
            if size < self.pos:
                self.pos = 0  # log rotated/truncated
            if size == self.pos:
                return []
            with open(self.path, "r", errors="replace") as fh:
                fh.seek(self.pos)
                data = fh.read()
                self.pos = fh.tell()
        except OSError:
            return []
        return data.splitlines()

    def decode_stats(self):
        """(tps_list, max_running) over Decode-batch lines appended since last call."""
        tps, running = [], 0
        for line in self.new_lines():
            if not self.DECODE_RE.search(line):
                continue
            m_t = self.TPS_RE.search(line)
            m_r = self.RUN_RE.search(line)
            if m_t:
                tps.append(float(m_t.group(1)))
            if m_r:
                running = max(running, int(m_r.group(1)))
        return tps, running


def one_request(session, args, prompt, out_tokens):
    """One streaming chat completion. Returns metrics dict."""
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Be detailed."},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": out_tokens,
        "temperature": 0,
        "ignore_eos": True,  # ask sglang for a fixed output length
    }
    t0 = time.time()
    t_first = None
    t_end = None
    usage = None
    err = None
    try:
        r = session.post(
            args.url, json=body, stream=True, timeout=(10, args.timeout),
            headers={"Accept-Encoding": "identity"},
        )
        r.raise_for_status()
        buf = b""
        for chunk in r.iter_content(chunk_size=None):
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                if t_first is None:
                    for ch in obj.get("choices") or []:
                        d = ch.get("delta") or {}
                        if d.get("content") or d.get("reasoning_content"):
                            t_first = time.time()
                            break
                if obj.get("usage"):
                    usage = obj["usage"]
                    t_end = time.time()
        t_end = t_end or time.time()
    except Exception as exc:  # noqa: BLE001 - report and continue
        err = "%s: %s" % (type(exc).__name__, exc)
        t_end = time.time()

    out_tok = (usage or {}).get("completion_tokens", 0)
    in_tok = (usage or {}).get("prompt_tokens", 0)
    ttft = (t_first - t0) if t_first else None
    lat = t_end - t0
    tpot = None
    if ttft is not None and out_tok > 1:
        tpot = (lat - ttft) / (out_tok - 1) * 1000.0
    return {
        "ok": usage is not None,
        "err": err,
        "ttft": ttft,
        "lat": lat,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "tpot": tpot,
    }


def run_level(session, args, conc, num_req, prompt, out_tokens, logtail):
    for _ in range(args.warmup):  # warm radix/paging, not counted
        one_request(session, args, prompt, min(out_tokens, 32))
    tps_before, _ = logtail.decode_stats()  # drain lines from warmup

    t_wall0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(one_request, session, args, prompt, out_tokens)
                for _ in range(num_req)]
        res = [f.result() for f in futs]
    wall = time.time() - t_wall0

    tps_lines, running_max = logtail.decode_stats()
    oks = [r for r in res if r["ok"]]
    ttfts = [r["ttft"] for r in oks if r["ttft"] is not None]
    tpots = [r["tpot"] for r in oks if r["tpot"] is not None]
    lats = [r["lat"] for r in oks]
    out_total = sum(r["out_tok"] for r in oks)
    in_total = sum(r["in_tok"] for r in oks)
    return {
        "concurrency": conc,
        "requests": num_req,
        "ok": len(oks),
        "fail": num_req - len(oks),
        "in_tok_mean": round(sum(r["in_tok"] for r in oks) / max(1, len(oks)), 1),
        "out_tok_mean": round(out_total / max(1, len(oks)), 1),
        "ttft_p50_s": round(pct(ttfts, 0.50), 3),
        "ttft_p95_s": round(pct(ttfts, 0.95), 3),
        "lat_p50_s": round(pct(lats, 0.50), 3),
        "lat_p95_s": round(pct(lats, 0.95), 3),
        "tpot_p50_ms": round(pct(tpots, 0.50), 1),
        "agg_out_tok_s": round(out_total / wall, 1),
        "agg_total_tok_s": round((out_total + in_total) / wall, 1),
        "eng_decode_tps_mean": round(sum(tps_lines) / len(tps_lines) if tps_lines else 0, 1),
        "eng_decode_tps_max": round(max(tps_lines) if tps_lines else 0, 1),
        "eng_running_max": running_max,
        "wall_s": round(wall, 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", default="1,2,4,8,16,32,64")
    ap.add_argument("--num", type=int, default=None,
                    help="override requests-per-level for ALL levels")
    ap.add_argument("--num-map", default=None,
                    help="per-level override, e.g. '1:6,8:16'")
    ap.add_argument("--out-tokens", type=int, default=256)
    ap.add_argument("--input-tokens", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600, help="read timeout s")
    ap.add_argument("--serve-log", default=None,
                    help="sglang serve log to parse engine decode throughput")
    ap.add_argument("--save", default=None, help="write full JSON results here")
    args = ap.parse_args()

    plan = dict(REQ_PLAN)
    if args.num_map:
        for kv in args.num_map.split(","):
            k, v = kv.split(":")
            plan[int(k)] = int(v)
    if args.num:
        plan = dict((lv, args.num) for lv in plan)

    levels = [int(x) for x in args.levels.split(",") if x]
    prompt = build_prompt(args.input_tokens)
    logtail = LogTail(args.serve_log) if args.serve_log else None

    session = requests.Session()
    session.trust_env = False  # never route via login-node proxies

    print("# bench_serving: model=%s url=%s input~%d tok out=%d tok "
          "levels=%s" % (args.model, args.url, args.input_tokens,
                         args.out_tokens, levels), flush=True)
    results = []
    for conc in levels:
        num = plan.get(conc, max(12, min(64, 2 * conc)))
        row = run_level(session, args, conc, num, prompt, args.out_tokens, logtail)
        results.append(row)
        print(json.dumps(row), flush=True)

    if args.save:
        doc = {
            "meta": {
                "model": args.model, "url": args.url,
                "input_tokens_target": args.input_tokens,
                "output_tokens": args.out_tokens,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "host_note": "client on cluster login node; engine on compute node",
            },
            "levels": results,
        }
        with open(args.save, "w") as fh:
            json.dump(doc, fh, indent=1)
        print("# saved -> %s" % args.save, flush=True)


if __name__ == "__main__":
    sys.exit(main())
