#!/usr/bin/env python3
"""Per-rank per-call MoE means (reliable: per-event, no thread overlap).

cpu_copy% = moe:apply.cpu_copy mean / moe:vkernel_apply mean (the GPU->CPU
sync of topk_ids dominates the MoE region). PP0 (rank0) computes routing on
GPU -> pays the full sync; PP1/PP2 (rank8/16) receive it -> far cheaper.
"""
import json, os, sys

SUBS = ["moe:apply.cpu_copy", "moe:apply.cpu_align",
        "moe:apply.gpu_copy", "moe:apply.launch"]


def mean_dur(ev, name):
    d = [e["dur"] for e in ev if e.get("name") == name and "dur" in e]
    return (sum(d) / len(d), len(d)) if d else (0.0, 0)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else \
        "/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin"
    jobs = [("603394", "EAGER "),
            ("603395", "BREAKABLE")]
    for job, lbl in jobs:
        print("\n===== %s  (per-call mean us, ALL threads on traced rank) =====" % lbl)
        print("%4s %9s %9s %7s %8s %7s %9s %7s" %
              ("rank", "vk_apply", "cpu_copy", "cpu_al", "gpu_cp", "launch",
               "cp_frac", "n_apply"))
        for r in (0, 8, 16):
            p = os.path.join(base, "run-%s" % job, "step_profiles",
                             "step_profile_rank%d.json" % r)
            if not os.path.isfile(p):
                continue
            ev = json.load(open(p))["traceEvents"]
            vm, vc = mean_dur(ev, "moe:vkernel_apply")
            cm = mean_dur(ev, "moe:apply.cpu_copy")[0]
            am = mean_dur(ev, "moe:apply.cpu_align")[0]
            gm = mean_dur(ev, "moe:apply.gpu_copy")[0]
            lm = mean_dur(ev, "moe:apply.launch")[0]
            pct = (cm / vm * 100.0) if vm else 0.0
            print("%4d %9.1f %9.1f %7.1f %8.1f %7.1f %8.0f%% %7d" %
                  (r, vm, cm, am, gm, lm, pct, vc))
        # PP0 vs PP1/PP2 summary
        p0 = os.path.join(base, "run-%s" % job, "step_profiles",
                          "step_profile_rank0.json")
        p1 = os.path.join(base, "run-%s" % job, "step_profiles",
                          "step_profile_rank8.json")
        if os.path.isfile(p0) and os.path.isfile(p1):
            v0 = mean_dur(json.load(open(p0))["traceEvents"], "moe:vkernel_apply")[0]
            v1 = mean_dur(json.load(open(p1))["traceEvents"], "moe:vkernel_apply")[0]
            print("  PP0/PP1 vk_apply ratio: %.1fx  (PP0 is the pipeline gate)" %
                  (v0 / v1 if v1 else 0))


if __name__ == "__main__":
    sys.exit(main() or 0)
