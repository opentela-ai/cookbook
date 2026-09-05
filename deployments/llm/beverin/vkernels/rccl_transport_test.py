#!/usr/bin/env python3
"""rccl_transport_test.py — cross-node RCCL all-to-all latency, Socket vs OFI.

Runs on 2 ranks (1 GPU each) across 2 nodes so the collective is forced over
the network transport (not intra-node shm). Prints the transport RCCL chose
(from NCCL_DEBUG=INFO, captured on stderr) and per-op latency for the actual
hot op in K3 serving: all_to_all_single. A short allreduce validates basic
cross-node connectivity first.

Env: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT (torch env:// init).
"""
import os, sys, time, statistics, traceback

# torch is heavy; import lazily so a fast init failure still prints.
try:
    import torch
    import torch.distributed as dist
except Exception as e:
    print(f"[rank {os.environ.get('RANK','?')}] IMPORT_FAIL: {e}", flush=True)
    sys.exit(2)

# Self-contained env:// init: srun sets SLURM_NODEID/SLURM_NNODES per task and
# the host exports HEAD_IP/MASTER_PORT (both survive the --environment=EDF
# launch, like the serving job's HEAD). Fill torch's expected RANK/
# WORLD_SIZE/MASTER_ADDR from those so we don't depend on an MPI launcher.
rank = int(os.environ.get("RANK", os.environ.get("SLURM_NODEID", "0")))
world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS",
             os.environ.get("SLURM_NNODES", "1"))))
if not os.environ.get("MASTER_ADDR"):
    os.environ["MASTER_ADDR"] = os.environ.get("HEAD_IP",
                                 os.environ.get("HEAD", "127.0.0.1"))
if not os.environ.get("MASTER_PORT"):
    os.environ["MASTER_PORT"] = "6381"
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world)
dev = torch.device("cuda:0")

print(f"[rank {rank}] init world={world} backend=nccl dev={dev} "
      f"master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}", flush=True)

print(f"[rank {rank}] init world={world} backend=rccl dev={dev}", flush=True)
try:
    dist.init_process_group(backend="nccl", init_method="env://")
    # ^ vLLM container names the RCCL backend "nccl" (NCCL env-compat shim).
except Exception as e:
    print(f"[rank {rank}] INIT_FAIL: {e}", flush=True)
    traceback.print_exc()
    sys.exit(3)

# warmup allreduce (validates cross-node connectivity)
wa = torch.ones(1024, dtype=torch.float32, device=dev)
for _ in range(5):
    dist.all_reduce(wa)
if rank == 0:
    print(f"[rank 0] warmup allreduce ok sum={wa[0].item():.1f}", flush=True)

# ---- the hot op: all_to_all_single (MoE dispatch/combine shape) ----------
# K3 TP=8: each step an all_to_all of ~hidden tokens. Use a [world, chunk]
# tensor split across world ranks (same logical pattern). chunk ~ a few KiB
# to ~1 MiB to cover small-token vs batched regimes.
def bench_all_to_all(shape_bytes_label, chunk):
    send = torch.randn(world, chunk, dtype=torch.float32, device=dev)
    recv = torch.empty_like(send)
    try:
        for _ in range(5):                      # warmup
            dist.all_to_all_single(recv, send)
        torch.cuda.synchronize()
        t = []
        for _ in range(20):
            s = time.perf_counter()
            dist.all_to_all_single(recv, send)
            torch.cuda.synchronize()            # host-measured latency
            t.append((time.perf_counter() - s) * 1e3)  # ms
        t.sort()
        print(f"[rank {rank}] [a2a {shape_bytes_label}] chunk={chunk}f32({chunk*4}B) "
              f"n={len(t)} min={t[0]:.3f}ms med={statistics.median(t):.3f}ms "
              f"p95={t[int(0.95*len(t))-1]:.3f}ms", flush=True)
    except Exception as e:
        print(f"[rank {rank}] [a2a {shape_bytes_label}] FAIL: {e}", flush=True)
        traceback.print_exc()
        return False
    return True

ok = True
for label, chunk in [("4KiB",1024),("64KiB",16384),("512KiB",131072)]:
    ok = bench_all_to_all(label, chunk) and ok

dist.barrier()
if rank == 0:
    print(f"[rank 0] ALL_OK ok={ok}", flush=True)
dist.destroy_process_group()
