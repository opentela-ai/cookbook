#!/usr/bin/env python3
"""Inspect GLM-5.3 chat_template.jinja delimiters + sglang parser options."""
import json, os, re, dataclasses

M = "/capstor/scratch/cscs/xyao/models/zai-org/GLM-5.3-Flash"
ct = open(os.path.join(M, "chat_template.jinja")).read()

# 1) special tokens (repr so terminal doesn't eat them)
toks = sorted(set(re.findall(r"<\|[^|>]+\|>", ct)))
print("=== SPECIAL TOKENS (repr) ===")
for t in toks:
    print(f"  {t!r:42} x{ct.count(t)}")

# 2) the assistant-generation block: find reasoning + tool-call emit
print("\n=== ASSISTANT BLOCK (around role assistant content emit) ===")
# find the section that emits <think>...</think> and tool calls
i = ct.find("split('')")
if i < 0:
    i = ct.find('split(\\"\\"')
print(ct[max(0, i-300):i+600])

# 3) sglang ServerArgs parser fields
try:
    from sglang.srt.server_args import ServerArgs
    for name in ("tool_call_parser", "reasoning_parser"):
        if name in {f.name for f in dataclasses.fields(ServerArgs)}:
            f = next(f for f in dataclasses.fields(ServerArgs) if f.name == name)
            print(f"\nServerArgs.{name} default={f.default!r}")
            # try to find the choices in metadata
            if f.metadata:
                print(f"  metadata={f.metadata}")
except Exception as e:
    print(f"ServerArgs err: {type(e).__name__}: {e}")

# 4) list reasoning parser implementations
try:
    import sglang.srt.reasoning_parser as rp
    p = os.path.dirname(rp.__file__)
    print(f"\n=== reasoning_parser dir: {p} ===")
    import pkgutil
    for m in pkgutil.iter_modules([p]):
        print(f"  {m.name}")
except Exception as e:
    print(f"reasoning_parser dir err: {type(e).__name__}: {e}")

# 5) list tool call parser implementations
for modpath in ["sglang.srt.managers.tool_call_parser", "sglang.srt.tool_call_parser"]:
    try:
        tcp = __import__(modpath, fromlist=["x"])
        p = os.path.dirname(tcp.__file__)
        print(f"\n=== {modpath} dir: {p} ===")
        import pkgutil
        for m in pkgutil.iter_modules([p]):
            print(f"  {m.name}")
        break
    except Exception as e:
        print(f"{modpath} err: {type(e).__name__}: {e}")

# 6) tool_call_parser class registry / choices
try:
    from sglang.srt.managers.tool_call_parser import ToolCallParser
    import inspect
    src = inspect.getsource(ToolCallParser)
    # find class names mentioned
    names = sorted(set(re.findall(r"class\s+(\w+)\(", src)) | set(re.findall(r"'([\w_-]+)':\s*\w+Parser", src)))
    print(f"\n=== ToolCallParser classes/keys: {names} ===")
except Exception as e:
    print(f"ToolCallParser err: {type(e).__name__}: {e}")
