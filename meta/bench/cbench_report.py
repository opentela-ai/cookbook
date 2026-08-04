#!/usr/bin/env python3
"""Aggregate cbench level reports (servekit bench JSON) into the sweep table.

Called by cbench.sh at the end of a sweep; also usable standalone:

    python3 cbench_report.py bench_dir/cbench_<label>_<ts>_c*.json

Prints the cookbook summary table (one row per concurrency level — a number
without its concurrency is meaningless, per meta/bench/README.md) and writes
<common-prefix>.summary.jsonl with the full per-level reports for
archival/quoting. Tolerates both the current servekit schema
(`throughput{wall_s,output_tok_per_s,...}`) and, for headers, older keys.
"""

import json
import re
import sys


def _metrics(r):
    """(concurrency, requests, wall_s, tok_s, lat{mean,p50,p99,max}, errors)."""
    t = r.get("throughput") or r.get("metrics") or r
    lat = t.get("latency_s") or {}
    return (
        t["concurrency"],
        t.get("requests", t.get("completed")),
        t["wall_s"],
        t.get("output_tok_per_s", t.get("agg_out_tok_s")),
        lat.get("mean", t.get("lat_mean_s")),
        lat.get("p50", t.get("lat_p50_s")),
        lat.get("p99", t.get("lat_p95_s")),  # p95 fallback: older docs named it so
        lat.get("max", t.get("lat_max_s")),
        t.get("errors", 0),
        t.get("input_len"),
        t.get("output_len"),
    )


def main(paths):
    reports = []
    for p in paths:
        try:
            with open(p) as f:
                r = json.load(f)
            _metrics(r)  # validate shape early
        except Exception as e:  # noqa: BLE001 - report and skip bad files
            print(f"[cbench] WARN: skipping {p}: {e}", file=sys.stderr)
            continue
        r["_file"] = p
        reports.append(r)
    if not reports:
        print("[cbench] no reports found", file=sys.stderr)
        return 1

    reports.sort(key=lambda r: _metrics(r)[0])
    r0 = reports[0]
    m0 = _metrics(r0)
    url = r0.get("base_url", r0.get("url"))
    print(f"\nmodel={r0.get('model')}  url={url}")
    print(
        f"protocol: servekit-bench, in_len={m0[9]} words, "
        f"out_len={m0[10]} tokens (ignore_eos); warmup discarded by cbench.sh"
    )
    c = r0.get("correctness")
    if c and "results" in c:
        print(f"correctness probe (first level): {len(c['results'])} unique greedy outputs — inspect/cmp in the raw JSON")
    hdr = (
        f"{'C':>4} {'n':>4} {'wall_s':>8} {'out_tok_s':>10} "
        f"{'lat_mean':>8} {'lat_p50':>8} {'lat_p99':>8} {'lat_max':>8} {'err':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in reports:
        C, N, W, T, ME, P50, P99, MX, E, _IL, _OL = _metrics(r)
        print(
            f"{C:>4} {N:>4} {W:>8.2f} {T:>10.1f} "
            f"{ME:>8.2f} {P50:>8.2f} {P99:>8.2f} {MX:>8.2f} {E:>4}"
        )

    suffix = re.search(r"_c\d+\.json$", paths[0])
    base = paths[0][: -len(suffix.group(0))] if suffix else paths[0]
    out = base + ".summary.jsonl"
    with open(out, "w") as f:
        for r in reports:
            r.pop("_file", None)
            f.write(json.dumps(r) + "\n")
    print(f"[cbench] combined reports -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
