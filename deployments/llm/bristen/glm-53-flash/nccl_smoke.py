#!/usr/bin/env python3
"""Tiny 2-node NCCL allreduce probe. Each rank (1 GPU) reduces [rank]; the
result must equal sum(range(world)). Validates the aws_ofi_nccl Slingshot/CXI
transport end-to-end without loading model weights."""
import os
import torch
import torch.distributed as dist

rank = int(os.environ.get("SLURM_NODEID", os.environ.get("RANK", "0")))
world = int(os.environ.get("SLURM_JOB_NUM_NODES", os.environ.get("WORLD_SIZE", "1")))
local = int(os.environ.get("SLURM_LOCALID", "0"))

torch.cuda.set_device(local if torch.cuda.device_count() > local else 0)
dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://{os.environ['HEAD_IP']}:{os.environ['MASTER_PORT']}",
    rank=rank,
    world_size=world,
)
t = torch.tensor([float(rank)], device="cuda")
dist.all_reduce(t)
expected = float(sum(range(world)))
ok = abs(t.item() - expected) < 1e-6
print(
    f"[rank {rank}/{world}] allreduce={t.item():.1f} expected={expected:.1f} "
    f"ok={ok} dev={torch.cuda.get_device_name(0)} peer={os.environ.get('HEAD_IP')}",
    flush=True,
)
dist.barrier()
if rank == 0:
    print(f"RESULT: {'PASS' if ok else 'FAIL'}", flush=True)
dist.destroy_process_group()
