#!/usr/bin/env python3
"""sglang launcher with MI300A integrated-memory accounting fix.

The MI300A is an APU (CPU + GPU on the same package, sharing HBM3 via a
coherent fabric).  PyTorch therefore reports ``is_integrated = True`` for
every MI300A GPU.  sglang's ``get_available_gpu_memory()`` interprets that
flag the same way it does for a laptop iGPU and, instead of calling
``torch.cuda.mem_get_info()``, falls back to ``psutil.virtual_memory().
available`` — i.e. the *whole-node* memory budget (~428 GB on a 4 × 128 GB
node) is returned to **every** TP rank.

Because the distributed path takes the ``all_reduce(MIN)`` of that value and
all ranks observe the same psutil number, the minimum is still ~428 GB.
Each of the 4 ranks then computes a KV-cache budget of roughly

    rest = 365 − 428 × (1 − 0.85) ≈ 301 GB

and tries to allocate 301 GB from a physical pool that has only ~365 GB free
in total.  Rank 0 grabs ~301 GB, leaving ~64 GB; rank 1 (or whichever ranks
allocate later) is SIGKILLed by the cgroup OOM killer — reproducibly, ~35 s
after ``Load weight end``, with no Python traceback (the kernel delivers the
signal directly).

The fix is to force ``is_integrated = False`` so sglang uses the correct
per-GPU ``torch.cuda.mem_get_info()`` value (137 GB per GPU).  The attribute
on the C++ ``_CudaDeviceProperties`` object is read-only, so we wrap the
return value of ``torch.cuda.get_device_properties`` in a thin proxy that
overrides the single flag and transparently forwards every other attribute
(``name``, ``total_memory``, ``major``, ``minor``,
``multi_processor_count``, …) to the original object.

Usage:
    python3 sglang_launcher_mi300a.py <sglang-args...>

  (drop-in replacement for ``python -m sglang.launch_server``)
"""

import os
import sys
import torch


class _DevicePropsProxy:
    """Transparent proxy that reports ``is_integrated = False``.

    Only ``is_integrated`` is overridden; everything else is forwarded to the
    original ``_CudaDeviceProperties`` C++ object via ``__getattr__``.
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
    import warnings

    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.launch_server import run_server

    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
