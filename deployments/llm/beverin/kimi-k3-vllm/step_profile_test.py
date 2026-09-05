#!/usr/bin/env python3
"""De-risk v2: start torch.profiler on the SAME thread as the GPU work
(mirrors wrapping Worker.execute_model on the worker's main thread), NOT a
daemon thread. Fixes ROCm's 'External init callback must run in same thread
as registerClient' error. A .go signal arms the profiler; it then captures
a DUR-second window of repeated GPU steps and exports a chrome trace that
must contain the moe:vkernel_apply record_function region."""
import os, time, json, torch
from torch.profiler import profile, ProfilerActivity, record_function

DUR = float(os.environ.get("VLLM_STEP_PROFILE_DUR", "5"))
DIR = os.environ.get("VLLM_STEP_PROFILE_DIR", "/tmp/step_profile_test")
os.makedirs(DIR, exist_ok=True)
go = os.path.join(DIR, ".go")
out = os.path.join(DIR, "step_profile_rank0.json")
state = {"prof": None, "stop_at": None, "done": False}

def execute_step():
    # Called repeatedly on the MAIN thread (mimics Worker.execute_model).
    if not state["done"] and os.path.exists(go) and state["prof"] is None:
        state["prof"] = profile(activities=[ProfilerActivity.CPU,
                                            ProfilerActivity.CUDA],
                                with_stack=False, record_shapes=False)
        state["prof"].start()
        state["stop_at"] = time.time() + DUR
        print(f"[T] START profile dur={DUR}s -> {out}", flush=True)
    # The "step" (GPU work on this same thread)
    with record_function("moe:vkernel_apply"):
        a = torch.randn(512, 512, device="cuda")
        b = a @ a
        torch.cuda.synchronize()
        b.sum().item()
    if (not state["done"] and state["prof"] is not None
            and time.time() >= state["stop_at"]):
        state["prof"].stop()
        state["prof"].export_chrome_trace(out)
        state["done"] = True
        print(f"[T] STOP  saved {out}", flush=True)

# Main-thread loop: spin steps until profiling done (writing .go externally).
print("[T] spinning steps (main thread); write .go to arm profiler", flush=True)
deadline = time.time() + 60
while not state["done"] and time.time() < deadline:
    execute_step()
    time.sleep(0.05)
assert os.path.exists(out), "no trace written!"
ev = json.load(open(out))
evs = ev if isinstance(ev, list) else ev.get("traceEvents", [])
has_moe = any(e.get("name") == "moe:vkernel_apply" for e in evs)
print(f"[T] events={len(evs)} moe_region={has_moe}")
assert has_moe, "no moe:vkernel_apply region in trace"
print("[T] OVERALL_PASS")
