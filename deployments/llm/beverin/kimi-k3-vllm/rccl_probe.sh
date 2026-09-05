#!/bin/bash
# rccl_probe.sh — run INSIDE a mi300 allocation. Report RCCL version,
# available NET plugins (Socket vs OFI/CXI), and all-to-all tuning knobs.
set -uo pipefail
echo "=== librccl.so resolution ==="
RCCL_SO=$(ldconfig -p 2>/dev/null | awk '/librccl\.so\./{print $NF; exit}')
echo "rccl_so=$RCCL_SO"
ls -la "$RCCL_SO" 2>/dev/null
echo
echo "=== RCCL version string ==="
strings "$RCCL_SO" 2>/dev/null | grep -iE "^rccl[ -][0-9]|RCCL version" | head -3
strings "$RCCL_SO" 2>/dev/null | grep -iE "[0-9]+\.[0-9]+\.[0-9]+-[a-f0-9]" | head -3
echo
echo "=== NET plugins available (Socket, OFI, CXI, IB verbs) ==="
# RCCL plugins live in librccl/plugins or alongside librccl
PLUGDIR=$(dirname "$RCCL_SO")/rccl-libs
echo "plugin_dir=$PLUGDIR"
ls -la "$PLUGDIR" 2>/dev/null | head
echo "--- search for any ofi/cxi/net plugin .so in LD path + container ---"
find / -xdev \( -name "*nccl-net*.so*" -o -name "*rccl*net*.so*" -o -name "*ofinet*.so*" -o -name "libfabric*.so*" \) 2>/dev/null | head -20
echo "--- libfabric installed? (CXI provider = Slingshot RDMA) ---"
which fi_info 2>/dev/null && fi_info -l 2>/dev/null | head -20
ldconfig -p 2>/dev/null | grep -iE "libfabric" | head
echo
echo "=== all-to-all / channel / algo tuning knobs present in librccl ==="
strings "$RCCL_SO" 2>/dev/null | grep -iE "^NCCL_(ALGO|PROTO|MIN_NCHANNELS|MAX_NCHANNELS|NET_GDR|BUFFSIZE|NET|TUNING|P2P_LEVEL|SOCKET_IFNAME|NETS|PROXY)|^RCCL_" | sort -u | head -50
echo
echo "=== NCCL_ALGO values referenced (Ring, Simple, Tree, Collnet, NVLS, ...) ==="
strings "$RCCL_SO" 2>/dev/null | grep -iE "^(Ring|Simple|Tree|CollnetDirect|CollnetChain|NVLSTree|NVLS|Patented)$" | sort -u
echo "--- done ---"
