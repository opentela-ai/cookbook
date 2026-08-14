#!/usr/bin/env python3
"""Minimal cross-node RCCL all-gather test.

Reproduces the exact collective that hangs in the Kimi-K3 serving recipe:
an all-gather across TP8 (2 nodes x 4 GPUs) over RCCL Socket transport
(hsn0, IB disabled), with message sizes matching the two collectives that
timed out in jobs 590747 / 590922:

  _ALLGATHER_BASE NumelIn=896   NumelOut=7168   (MLA K/V all-gather)
  _ALLGATHER_BASE NumelIn=20480 NumelOut=163840

Runs under srun --environment=kimi-k3-vllm (the same EDF container as the
real recipe), so RCCL version + aws_ofi_nccl LD_PRELOAD hook are identical.
"""
import os

# Neutralise the aws_ofi_nccl hook + force RCCL built-in Socket transport,
# exactly as engine.sh does. Must happen before torch / dist are imported.
os.environ.pop("NCCL_NET_PLUGIN", None)
os.environ.pop("NCCL_NET", None)
os.environ["NCCL_SOCKET_IFNAME"] = "hsn0"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SOCKET_NTHREADS"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,ENV"
# Fail fast: 120 s collective watchdog instead of torch's default 600 s.
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"

import sys
import time
import socket
import torch
import torch.distributed as dist


def main() -> int:
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    host = socket.gethostname()
    master_addr = os.environ.get("MASTER_ADDR", "")
    master_port = os.environ.get("MASTER_PORT", "29500")

    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    print(f"[{time.strftime('%H:%M:%S')}] rank={rank}/{world} ({host}) "
          f"local_rank={local_rank} dev={torch.cuda.get_device_name(local_rank)} "
          f"master={master_addr}:{master_port}", flush=True)

    dist.init_process_group(
        init_method="env://",
        backend="nccl",  # RCCL on ROCm
        world_size=world,
        rank=rank,
        timeout=torch.distributed.timedelta(seconds=120),
    )
    print(f"[{time.strftime('%H:%M:%S')}] rank={rank} init_process_group OK", flush=True)
    dist.barrier()
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] all {world} ranks reached barrier", flush=True)

    # The two message sizes from the failing runs.
    for size in (896, 20480, 1_048_576):
        t = torch.randn(size, device=dev, dtype=torch.float32)
        out = [torch.empty(size, device=dev, dtype=torch.float32) for _ in range(world)]
        try:
            dist.all_gather(out, t)
            torch.cuda.synchronize()
            n = 25
            t0 = time.perf_counter()
            for _ in range(n):
                dist.all_gather(out, t)
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) / n * 1000.0
            print(f"[{time.strftime('%H:%M:%S')}] rank={rank} "
                  f"all_gather({size}->{size*world}) x{n} avg={dt_ms:.3f} ms", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{time.strftime('%H:%M:%S')}] rank={rank} "
                  f"all_gather({size}) FAILED: {type(e).__name__}: {e}", flush=True)
            return 1

    dist.barrier()
    if rank == 0:
        print("RCCL_ALLGATHER_OK", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
