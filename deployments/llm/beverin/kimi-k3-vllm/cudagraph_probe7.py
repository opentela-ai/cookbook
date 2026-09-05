#!/usr/bin/env python3
"""Probe v7: config/vllm.py piecewise-requires-compile override (the gate)."""
import os
OUT = os.environ.get("OUT", "/tmp/k3_cm7.txt")
f = open(OUT, "w")
def w(*a): print(*a, file=f, flush=True)
F = "/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py"
src = open(F).read().splitlines()
n = len(src)
w("=== 1300-1345 (piecewise-requires-compile override?) ===")
for i in range(1300, min(1346, n) + 1):
    w(f"{i:4}: {src[i-1].rstrip()[:118]}")
w("\n=== 1535-1560 (cudagraph_mode final gates) ===")
for i in range(1535, min(1561, n) + 1):
    w(f"{i:4}: {src[i-1].rstrip()[:118]}")
w("\n=== is_breakable / VLLM_USE_BREAKABLE mentions in config/vllm.py ===")
for i, line in enumerate(src, 1):
    if "is_breakable" in line or "VLLM_USE_BREAKABLE" in line or "Breakable" in line:
        w(f"{i:4}: {line.rstrip()[:118]}")
w("\nDONE")
f.close(); print("WROTE", OUT, flush=True)
