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
#
# Debug mode (set by the caller via the run.sh heredoc):
#   SMOKE_PG_TIMEOUT   seconds applied to init_process_group and therefore
#                      to every blocking collective (all_reduce, barrier).
#                      Default 1800 = the original no-override wall. When the
#                      RCCL net plugin hangs, a short timeout turns a silent
#                      SIGTERM-at-the-wall into a torch.distributed
#                      TimeoutError with a stack trace. Pair with the run.sh
#                      NCCL_DEBUG=TRACE / FI_LOG_LEVEL=DEBUG overrides.

import os
import sys
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist


def log(msg: str) -> None:
    r = dist.get_rank() if dist.is_initialized() else -1
    print(f"[r{r}] {msg}", flush=True)


def _pg_timeout() -> timedelta:
    try:
        s = float(os.environ.get("SMOKE_PG_TIMEOUT", "0") or 0)
    except ValueError:
        s = 0.0
    return timedelta(seconds=s if s > 0 else 1800.0)


def main() -> int:
    backend = "nccl"
    to = _pg_timeout()
    log(
        f"OFI_SMOKE phase=init backend={backend} "
        f"pg_timeout={to.total_seconds():.0f}s "
        f"NCCL_NET={os.environ.get('NCCL_NET', '<unset>')} "
        f"NCCL_DEBUG={os.environ.get('NCCL_DEBUG', '<unset>')} "
        f"NCCL_DEBUG_SUBSYS={os.environ.get('NCCL_DEBUG_SUBSYS', '<all>')} "
        f"FI_PROVIDER={os.environ.get('FI_PROVIDER', '<unset>')} "
        f"FI_LOG_LEVEL={os.environ.get('FI_LOG_LEVEL', '<unset>')}"
    )
    try:
        dist.init_process_group(backend=backend, init_method="env://",
                                timeout=to)
        r = dist.get_rank()
        w = dist.get_world_size()

        # One GPU per rank (CUDA_VISIBLE_DEVICES is pinned by the launcher).
        dev = torch.device("cuda", 0)
        expect = w * (w + 1) // 2  # sum of 1..w
        t = torch.tensor([float(r + 1)], device=dev)

        log(f"OFI_SMOKE phase=barrier world={w} rank={r}")
        dist.barrier()
        log("OFI_SMOKE phase=all_reduce starting (first cross-node collective)")
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

        log("OFI_SMOKE phase=final_barrier")
        dist.barrier()
        dist.destroy_process_group()
        return 0 if ok else 1
    except Exception as e:  # noqa: BLE001 - report any failure clearly
        # A TimeoutError here is the signal that the net plugin hung at the
        # phase named in the last OFI_SMOKE phase=... line above.
        log(f"OFI_SMOKE_FAIL {type(e).__name__}: {e}")
        traceback.print_exc()
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
