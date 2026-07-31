#!/usr/bin/env python3
"""NCCL transport probe — NOT an LLM throughput test.

Checks the collective fabric a serving topology rests on: does cross-node
all_reduce / all_gather actually use the offload path (SHARP, GPUDirect RDMA)
or silently fall back to TCP? Runs ~3000 of each at 16 MB/rank and reports
per-op latency. Use it once, early, to confirm the network layer the bench
sits on — never report these numbers as "model throughput."

Run one process per GPU (e.g. srun -n$WORLD python3 nccl_sharp_probe.py) with
NCCL_DEBUG=INFO and grep the log for `/SHARP` (engaged) vs `Socket/IB`
(fallback). See meta/bench/README.md §"Transport check".
"""
import os, time, torch, torch.distributed as dist
rank  = int(os.environ["SLURM_PROCID"]); world = int(os.environ["SLURM_NTASKS"]); local = int(os.environ["SLURM_LOCALID"])
os.environ["RANK"]=str(rank); os.environ["WORLD_SIZE"]=str(world); os.environ["LOCAL_RANK"]=str(local)
torch.cuda.set_device(local)
print(f"[rank {rank}/{world}] dev={torch.cuda.get_device_name(0)} torch_nccl={torch.cuda.nccl.version()}", flush=True)
dist.init_process_group("nccl"); g=torch.cuda.current_device()
t = torch.randn(16*1024*1024//4, device=g, dtype=torch.float32)            # 16 MB/rank
N = 3000
def loop(fn, tag):
    for _ in range(20): fn()
    dist.barrier()
    if rank==0: print(f"begin {tag} x{N}", flush=True)
    t0=time.time()
    for _ in range(N): fn()
    torch.cuda.synchronize(); dist.barrier(); dt=time.time()-t0
    if rank==0: print(f"DONE_{tag} dt={dt:.2f}s per_op={dt/N*1000:.2f} ms", flush=True)
loop(lambda: dist.all_reduce(t), "AR")
parts=[torch.empty_like(t) for _ in range(world)]
loop(lambda: dist.all_gather(parts, t), "AG")
dist.barrier(); dist.destroy_process_group()
