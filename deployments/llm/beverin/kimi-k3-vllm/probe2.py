import glob,sys
# VLLM_TRACE_FUNCTION support (V0/V1 step tracer)
hits=[p.split("site-packages/")[-1] for p in glob.glob("/opt/venv/lib/python*/site-packages/vllm/**/*.py",recursive=True) if "VLLM_TRACE_FUNCTION" in open(p,errors="ignore").read()]
print("VLLM_TRACE_FUNCTION",hits[:6])
# Where does the MoE dispatch (all_to_all) live in vLLM?
import subprocess
for kw in ["def fused_experts","all_to_all_single","pynccl_all_to_all","class FusedMoE","def forward"]:
    r=subprocess.run(["grep","-rln",kw,"/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/"],capture_output=True,text=True)
    print(kw, [x.split("fused_moe/")[-1] for x in r.stdout.split()][:3])
# Model path: where is the Kimi-K3 trust-remote-code?
import os
for p in ["/capstor/scratch/cscs/xyao/models","/capstor/scratch/cscs/xyao/kimi-k3"]:
    if os.path.isdir(p):
        for f in os.listdir(p):
            if "kimi" in f.lower() or "moon" in f.lower(): print("MODELDIR",p+"/"+f)
