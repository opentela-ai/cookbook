#!/usr/bin/env python3
"""sglang launcher with the MI300A integrated-memory accounting fix.

The MI300A is an APU (CPU + GPU on one package sharing HBM3 over a coherent
fabric), so PyTorch reports ``is_integrated = True`` for every device. sglang's
``get_available_gpu_memory()`` reads that flag the way it would for a laptop
iGPU and, instead of calling ``torch.cuda.mem_get_info()``, falls back to
``psutil.virtual_memory().available`` — i.e. the *whole-node* figure (477 GiB
measured on beverin) is handed to **every** TP rank instead of the correct
per-GPU 128 GiB.

Because the distributed path takes ``all_reduce(MIN)`` and every rank observes
the same psutil number, the minimum stays ~477 GiB. Each of the 4 ranks then
sizes its KV pool against a pool that does not exist, and the cgroup OOM killer
SIGKILLs a rank (exit -9) shortly after ``Load weight end`` with no Python
traceback.

Forcing ``is_integrated = False`` makes sglang use per-GPU ``mem_get_info()``
(128 GiB), restoring discrete-GPU accounting. The attribute on the C++
``_CudaDeviceProperties`` object is read-only, so the return value of
``torch.cuda.get_device_properties`` is wrapped in a thin proxy that overrides
that one flag and forwards everything else.

Usage:
    python3 sglang_launcher_mi300a.py <sglang-args...>

  (drop-in replacement for ``python -m sglang.launch_server``)
"""

import os
import sys

import torch


class _DevicePropsProxy:
    """Transparent proxy that reports ``is_integrated = False``.

    Only ``is_integrated`` is overridden; every other attribute (``name``,
    ``total_memory``, ``gcnArchName``, ``multi_processor_count``, …) is
    forwarded to the original ``_CudaDeviceProperties`` C++ object.
    """

    __slots__ = ("_orig",)

    def __init__(self, original):
        object.__setattr__(self, "_orig", original)

    def __getattr__(self, name):
        if name == "is_integrated":
            return False
        return getattr(object.__getattribute__(self, "_orig"), name)


_orig_get_device_properties = torch.cuda.get_device_properties


def _patched_get_device_properties(device):
    return _DevicePropsProxy(_orig_get_device_properties(device))


torch.cuda.get_device_properties = _patched_get_device_properties


def main():
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


# The __main__ guard is LOAD-BEARING: sglang starts its per-TP-rank scheduler
# processes with multiprocessing "spawn", which re-imports this file in every
# child. Without the guard, run_server() re-executes at import time and each
# child dies with "An attempt has been made to start a new process before the
# current process has finished its bootstrapping phase"; the parent surfaces it
# only as "Rank N scheduler died during initialization (exit code: 1)".
if __name__ == "__main__":
    main()
