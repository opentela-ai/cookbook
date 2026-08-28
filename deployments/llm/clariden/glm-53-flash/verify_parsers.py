#!/usr/bin/env python3
"""Verify GLM-5.3 reasoning + tool-call parsers via /v1/chat/completions.
Checks: (1) a plain chat request returns content (and reasoning_content
parsed, not leaked into content), (2) a tool-call request returns tool_calls."""
import json, urllib.request, sys, time

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "zai-org/GLM-5.3-Flash"

def chat(messages, tools=None, tool_choice=None, max_tokens=128):
    body = {"model": MODEL, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read()), time.time() - t0

# 1) Plain chat: reasoning should be separated into reasoning_content, content clean
r, dt = chat([{"role": "user", "content": "What is the capital of France? Answer in one word."}])
msg = r["choices"][0]["message"]
print(f"[1] plain chat ({dt:.1f}s)")
print(f"    content={msg.get('content')!r}")
print(f"    reasoning_content={'present' if msg.get('reasoning_content') else 'absent'} (len={len(msg.get('reasoning_content') or '')})")

# 2) Tool call
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]
r2, dt2 = chat([{"role": "user", "content": "What's the weather in Paris?"}], tools=tools, tool_choice="auto")
msg2 = r2["choices"][0]["message"]
print(f"[2] tool call ({dt2:.1f}s)")
print(f"    tool_calls={json.dumps(msg2.get('tool_calls'), ensure_ascii=False)[:300]}")
print(f"    content={msg2.get('content')!r}")
