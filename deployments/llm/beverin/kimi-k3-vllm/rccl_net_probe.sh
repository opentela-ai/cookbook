#!/bin/bash
# rccl_net_probe.sh — focus on NET transport options + libfabric/CXI.
set -uo pipefail
RCCL_SO=$(ldconfig -p 2>/dev/null | awk '/librccl\.so\./{print $NF; exit}')
echo "rccl_so=$RCCL_SO"
echo
echo "=== libfabric installed? (CXI provider = Slingshot RDMA) ==="
which fi_info 2>/dev/null && { echo "--- providers ---"; fi_info -l 2>/dev/null | head -25; } || echo "no fi_info"
echo "--- libfabric in ldconfig ---"
ldconfig -p 2>/dev/null | grep -iE "libfabric" | head
echo "--- libfabric .so on FS ---"
find /usr /opt -xdev \( -name "libfabric.so*" -o -name "*ofi*.so*" -o -name "libcxi*.so*" \) 2>/dev/null | head -15
echo
echo "=== RCCL-shipped NET plugins (besides Socket) ==="
PLUGDIR=$(dirname "$RCCL_SO")/rccl-libs
echo "plugin_dir=$PLUGDIR"
ls -la "$PLUGDIR" 2>/dev/null | head
echo "--- any nccl-net / ofi-net .so anywhere ---"
find /usr /opt -xdev \( -name "*nccl-net*.so*" -o -name "*ofi*net*.so*" -o -name "librccl*net*.so*" \) 2>/dev/null | head -15
echo
echo "=== Does librccl itself contain OFI/Smaug/CXI transport symbols? ==="
strings "$RCCL_SO" 2>/dev/null | grep -iE "ofi|libfabric|cxi|smaug|cassini|roce|infiniband|verbs" | sort -u | head -25
echo
echo "=== RCCL transport env vars (the actual selectors) ==="
strings "$RCCL_SO" 2>/dev/null | grep -iE "^RCCL_(NET|TRANSPORT|SOCKET|OFI|INFINIBAND|IB|GDRDMA)|^NCCL_(NET|TRANSPORT|SOCKET_IFNAME|IB_)" | sort -u | head -25
echo "--- done ---"
