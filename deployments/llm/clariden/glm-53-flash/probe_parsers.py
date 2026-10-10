#!/usr/bin/env python3
import os, glob
import sglang
base = os.path.dirname(os.path.dirname(sglang.__file__))  # .../site-packages
sg = os.path.join(base, "sglang", "srt")
print("BASE:", base)
print("=== reasoning_parser/*.py ===")
for f in sorted(glob.glob(os.path.join(sg, "reasoning_parser", "*.py"))):
    if not os.path.basename(f).startswith("__"):
        print(" ", os.path.basename(f)[:-3])
print("=== managers/tool_call_parser/*.py ===")
for f in sorted(glob.glob(os.path.join(sg, "managers", "tool_call_parser", "*.py"))):
    if not os.path.basename(f).startswith("__"):
        print(" ", os.path.basename(f)[:-3])
