#!/usr/bin/env python3
"""device_info — what GPU am I on, and what does "100%" mean there.

Prints the torch-visible device identity and the NOMINAL peak table entry the
profiler will use for roofline utilization. Peaks are datasheet-class numbers
for dense bf16 matrix math and HBM bandwidth — clocks vary, so always state
the peak when quoting a utilization, and override with --peak-flops/--peak-bw
(in TFLOP/s / GB/s) when you know better.
"""
from __future__ import annotations

import argparse
import json
import sys

# (HBM/unified BW GB/s, dense bf16 matrix peak TFLOP/s) — NOMINAL, verify.
# Entries are substring-matched against torch.cuda.get_device_name().lower().
PEAKS = {
    "a100": (2039, 312),
    "h100": (3350, 989),
    "h200": (4800, 989),
    "gh200": (4000, 989),      # GH200 120GB HBM3 variant; 480GB HBM3e is 4800
    "mi300a": (1228, 590),
    "mi300x": (5300, 1307),
}
MISSING = (None, None)


def lookup(device_name):
    n = (device_name or "").lower()
    for key, peaks in PEAKS.items():
        if key in n:
            return key, peaks
    return None, MISSING


def describe():
    import torch

    name = "cpu"
    props = {}
    plat = "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        name = props.name
        plat = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    key, (bw, flops) = lookup(name)
    return {
        "torch": torch.__version__,
        "platform": plat,
        "device_name": name,
        "capability": f"{props.major}.{props.minor}" if props else None,
        "total_mem_gb": round(props.total_memory / 2**30, 1) if props else None,
        "peak_row": key,
        "peak_bw_gbps": bw,
        "peak_bf16_tflops": flops,
        "peaks_are": "nominal datasheet (dense bf16 matrix, HBM BW); override with --peak-*",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    d = describe()
    if args.json:
        print(json.dumps(d, indent=2))
        return
    print(f"torch {d['torch']}  platform={d['platform']}")
    print(f"device: {d['device_name']}" +
          (f"  (sm_{d['capability'].replace('.', '')}, {d['total_mem_gb']} GB)" if d["capability"] else ""))
    if d["peak_row"]:
        print(f"nominal peaks [{d['peak_row']}]: {d['peak_bw_gbps']} GB/s, {d['peak_bf16_tflops']} TFLOP/s (bf16 dense)")
        print(d["peaks_are"])
    else:
        print("no nominal peak row for this device — pass --peak-flops/--peak-bw to microbench")
        print(f"known rows: {', '.join(sorted(PEAKS))}")
    sys.exit(0)


if __name__ == "__main__":
    main()
