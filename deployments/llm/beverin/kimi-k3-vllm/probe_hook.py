import inspect
from vllm.v1.worker.gpu_worker import Worker
ms=[m for m in dir(Worker) if not m.startswith("__") and callable(getattr(Worker,m,None))]
print("WORKER_METHODS", ms)
for m in ("execute_model","load_model","start_worker","init_worker","_execute_model"):
    f=getattr(Worker,m,None)
    if f: print(m, "sig:", str(inspect.signature(f))[:120])
# Does the worker call execute_model repeatedly (the per-step hook)?
print("EXE_DOC", (inspect.getdoc(getattr(Worker,"execute_model",None)) or "")[:160])
