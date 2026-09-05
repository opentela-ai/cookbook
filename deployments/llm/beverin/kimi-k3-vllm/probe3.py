import glob,subprocess,os
# locate vllm site-packages + fused_moe dir
sp=[p for p in glob.glob("/opt/venv/lib/python*/site-packages/vllm") if os.path.isdir(p)]
print("SP",sp[:2])
fmd=[p for p in glob.glob("/opt/venv/lib/python*/site-packages/vllm/model_executor/layers/fused_moe") if os.path.isdir(p)]
print("FUSED_MOE_DIR",fmd[:1])
if fmd:
    print("FILES",[os.path.basename(x) for x in glob.glob(fmd[0]+"/*.py")][:25])
    # all_to_all / dispatch sites in this dir
    for kw in ["all_to_all","alltoall","permute","combine","dispatch"]:
        r=subprocess.run(["grep","-rln",kw,fmd[0]],capture_output=True,text=True)
        print(kw,[os.path.basename(x) for x in r.stdout.split()][:4])
# MoE all-to-all primitive in distributed/
r=subprocess.run(["grep","-rln","all_to_all_single","/opt/venv/lib/python3.12/site-packages/vllm/"],capture_output=True,text=True)
print("A2A_SINGLE",[x.split("vllm/")[-1] for x in r.stdout.split()][:5])
# trust-remote-code model: find kimi/moonshot
for base in ["/capstor/scratch/cscs/xyao","/capstor/scratch/cscs/xyao/models"]:
    if os.path.isdir(base):
        for root,dirs,files in os.walk(base):
            if "modeling" in root or "configuration_kimi" in files or "modeling_kimi" in files:
                print("TRC",root[:90]); break
            if root.count("/")>7: dirs[:] = [d for d in dirs if "kimi" in d.lower() or "moon" in d.lower()]
