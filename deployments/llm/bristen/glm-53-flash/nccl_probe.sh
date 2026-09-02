#!/bin/bash
echo "== ldd cuda13 plugin (default LD_LIBRARY_PATH, system libfabric) =="
ldd /opt/cscs/aws-ofi-ccl-plugin/cuda13/libnccl-net.so 2>&1 | head -25
echo
echo "== system libfabric providers dir =="
ls -1 /usr/lib/x86_64-linux-gnu/libfabric/providers/ 2>/dev/null || echo no-providers-dir
echo "== fi_info --list (system libfabric) cxi/sock/tcp =="
fi_info --list 2>&1 | grep -iE 'cxi|sock|tcp|verbs' | head || echo no-fiinfo-or-no-match
echo "== libcxi / libfabric on system path (ldconfig) =="
ldconfig -p 2>/dev/null | grep -iE 'cxi|fabric' | head
echo "== NCCL net search: libnccl-net* / libnccl_net* anywhere =="
find /usr /opt -maxdepth 4 \( -name 'libnccl-net*.so*' -o -name 'libnccl_net*.so*' \) 2>/dev/null | head
echo "== nccl soname =="
ldconfig -p 2>/dev/null | grep -i 'libnccl' | head
