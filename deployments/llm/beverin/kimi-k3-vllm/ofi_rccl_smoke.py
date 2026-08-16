#!/usr/bin/env python3
# ofi_rccl_smoke.py — RCCL net-plugin smoke for vkernels issue #19.
#
# A deliberately tiny torch.distributed RCCL allreduce across N ranks. With
# NCCL_NET=<librccl-net-ofi.so> + FI_PROVIDER=cxi set by the caller, RCCL
# must (a) dlopen the vkernels plugin, (b) accept its net-plugin ABI, and
# (c) move the cross-node allreduce over the Cray CXI (Slingshot) fabric
# instead of the built-in Socket fallback. If any of those fail, this script
# exits non-zero and the smoke is a FAIL.
#
# The plugin's correctness as a *faster* transport than Socket is NOT proven
# here (that needs a latency sweep); this only proves it LOADS and COMPLETES,
# which is the gate before touching the 3h Kimi-K3 serving recipe.

import os
import sys
import time

import torch
import torch.distributed as dist


def log(msg: str) -> None:
    r = dist.get_rank() if dist.is_initialized() else -1
    print(f"[r{r}] {msg}", flush=True)


def main() -> int:
    backend = "nccl"
    dist.init_process_group(backend=backend, init_method="env://")
    r = dist.get_rank()
    w = dist.get_world_size()

    # One GPU per rank (CUDA_VISIBLE_DEVICES is pinned by the launcher).
    dev = torch.device("cuda", 0)
    expect = w * (w + 1) // 2  # sum of 1..w
    t = torch.tensor([float(r + 1)], device=dev)

    dist.barrier()
    t0 = time.time()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dt_us = (time.time() - t0) * 1e6

    ok = bool(abs(t.item() - expect) < 1e-3)
    log(
        f"OFI_SMOKE backend={backend} world={w} got={t.item():.1f} "
        f"expect={expect} ok={ok} allreduce_us={dt_us:.1f} "
        f"NCCL_NET={os.environ.get('NCCL_NET', '<unset>')} "
        f"FI_PROVIDER={os.environ.get('FI_PROVIDER', '<unset>')}"
    )
    if r == 0:
        print("OFI_SMOKE_PASS" if ok else "OFI_SMOKE_FAIL", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
