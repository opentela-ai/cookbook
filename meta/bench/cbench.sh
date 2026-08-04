#!/bin/bash
# cbench.sh — the cookbook LLM benchmark: a concurrency sweep driven by
# `servekit bench` (github.com/eth-easl/servekit), which is pure-stdlib
# (urllib) and therefore runs in any compute-allocation container without
# pip installs (JSC has no egress; older sweep tooling needed aiohttp).
#
# A raw `servekit bench` invocation measures ONE concurrency. This wrapper
# enforces the cookbook protocol (meta/bench/README.md):
#   1. explicit warmup pass (C=4 n=8), DISCARDED — never quote it;
#   2. a sweep of CONCURRENCY:REQUESTS levels, one report per level
#      (+ always quote the concurrency next to every number);
#   3. the qualitative correctness probe runs at the FIRST measured level
#      (catches a broken template/parser before you spend GPU-hours on the
#      sweep; later levels skip it to keep timings clean);
#   4. a summary table + combined JSONL at the end.
#
# Usage:
#   bash cbench.sh URL "C:N [C:N ...]" [IN_WORDS] [OUT_TOKENS] [flags]
# Example (canonical cookbook sweep):
#   bash cbench.sh http://127.0.0.1:30000 "1:8 8:32 16:48 32:64 52:52" 768 256 --label kimi-k3
#
#   URL        engine endpoint (http://IP:port of /v1/completions server)
#   C:N        level = concurrency C with N requests; N is rounded UP to a
#              multiple of C (servekit splits requests evenly across workers)
#   IN_WORDS   servekit --input-len: prompt length in WORDS, not tokens
#              (default 768 ~= 1024 tokens on these prompts; the realized
#              token counts are what matter and live in the raw reports)
#   OUT_TOKENS --output-len, exact via ignore_eos (default 256)
# Flags:
#   --label NAME         report file prefix (default "bench")
#   --out-dir DIR        where cbench_<label>_<ts>_c<C>.json land (default .)
#   --wait-ready S       per-level readiness poll budget (default 1800)
#   --no-warmup          skip the warmup pass (step 1) — only for re-runs
#   --correctness-all    run the probe at every level (default: first only)
#   --no-correctness     never run the probe
#
# Runner: resolved by servekit_env.sh ($SERVEKIT_DIR checkout, else PATH).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() { sed -n '2,43p' "${BASH_SOURCE[0]}"; exit "${1:-2}"; }
[ $# -ge 2 ] || usage 2
URL="$1"; LEVELS="$2"; IN_WORDS="${3:-768}"; OUT_TOKENS="${4:-256}"
shift $(( $# >= 4 ? 4 : $# ))

LABEL=bench; OUT_DIR=.; WAIT=1800; WARMUP=1; CORRECTNESS=first
while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --wait-ready) WAIT="$2"; shift 2;;
    --no-warmup) WARMUP=0; shift;;
    --correctness-all) CORRECTNESS=all; shift;;
    --no-correctness) CORRECTNESS=none; shift;;
    *) echo "FATAL: unknown flag: $1" >&2; usage 2;;
  esac
done
case "$URL" in http://*|https://*) ;; *) echo "FATAL: URL must start with http(s)://" >&2; exit 2;; esac

# shellcheck source=servekit_env.sh
source "$HERE/servekit_env.sh"

mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
PFX="$OUT_DIR/cbench_${LABEL}_${TS}"

echo "[cbench] target=$URL sweep=[$LEVELS] in_words=$IN_WORDS out_tokens=$OUT_TOKENS"
echo "[cbench] runner: ${SERVEKIT_RUN[*]}"

if [ "$WARMUP" = 1 ]; then
  echo "[cbench] warmup (C=4 n=8) — DISCARDED, never quote it"
  "${SERVEKIT_RUN[@]}" bench --url "$URL" --wait-ready "$WAIT" \
    --requests 8 --concurrency 4 --input-len "$IN_WORDS" --output-len "$OUT_TOKENS" \
    --no-correctness --out "${PFX}_warmup_DISCARDED.json" >/dev/null \
    || { echo "[cbench] FATAL: warmup failed — engine unreachable or crashing" >&2; exit 1; }
  rm -f "${PFX}_warmup_DISCARDED.json"
fi

RC=0; first=1
for lvl in $LEVELS; do
  C="${lvl%%:*}"; N="${lvl##*:}"
  case "$C:$N" in *[!0-9:]*) echo "[cbench] FATAL: bad level '$lvl' (want C:N integers)" >&2; exit 2;; esac
  if [ $(( N % C )) -ne 0 ]; then
    NEWN=$(( (N / C + 1) * C ))
    echo "[cbench] note: level $lvl -> n=$NEWN (rounded up to a multiple of C)"
    N=$NEWN
  fi
  args=(bench --url "$URL" --wait-ready "$WAIT" --requests "$N" --concurrency "$C"
        --input-len "$IN_WORDS" --output-len "$OUT_TOKENS" --out "${PFX}_c${C}.json")
  if [ "$CORRECTNESS" = none ] || { [ "$CORRECTNESS" = first ] && [ "$first" = 0 ]; }; then
    args+=(--no-correctness)
  fi
  echo "[cbench] level C=$C n=$N -> ${PFX}_c${C}.json"
  "${SERVEKIT_RUN[@]}" "${args[@]}" || { echo "[cbench] WARN: level C=$C failed — stopping sweep" >&2; RC=1; break; }
  first=0
done

"${PYTHON:-python3}" "$HERE/cbench_report.py" "${PFX}"_c*.json \
  || echo "[cbench] WARN: report aggregation failed; per-level JSONs are complete" >&2
exit "$RC"
