import os
print("ENVTEST EXPORTED=" + os.environ.get("EXP_VAR", "MISSING"))
print("ENVTEST PREFIX=" + os.environ.get("PRE_VAR", "MISSING"))
print("ENVTEST STEP_PROFILE_DIR=" + os.environ.get("VLLM_STEP_PROFILE_DIR", "MISSING"))
