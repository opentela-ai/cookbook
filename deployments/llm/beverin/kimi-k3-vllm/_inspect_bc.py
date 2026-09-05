import re
F = "/usr/local/lib/python3.12/dist-packages/vllm/compilation/breakable_cudagraph.py"
src = open(F).read().splitlines()
print("=== KEY DEFS / LINES ===")
for i, l in enumerate(src, 1):
    if re.search(r'def eager_break_during_capture|def __call__|def capture\b|def _capture|def replay\b|def _replay|def is_capturing|def is_breakable_cudagraph_enabled|decode_mode|current_stream|wait_event|side_stream|extra_stream|begin_capture|end_capture|_run_eager|run_eagerly|self\.extra|new_stream', l):
        print(f"{i}: {l.rstrip()}")
print("\n=== eager_break_during_capture body ===")
for i, l in enumerate(src, 1):
    if 'def eager_break_during_capture' in l:
        for j in range(i, min(i + 55, len(src) + 1)):
            print(f"{j}: {src[j-1].rstrip()}")
        break
