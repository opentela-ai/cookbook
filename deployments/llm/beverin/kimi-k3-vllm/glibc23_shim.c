/* glibc23_shim.c - provides __isoc23_* symbols under node GLIBC_2.38 so that
 * binaries built against glibc >= 2.38 load on older glibc (here 2.35,
 * Ubuntu 22.04, the vllm+vllm-openai-rocm+kimi-k3 image).
 *
 * CONTEXT: the CSCS Cray libfabric 1.29.1 + libcxi 1.5.0 stack on the Alps
 * compute host is the ONLY path to the CXI (Slingshot) transport. Every
 * object in it imports exactly these glibc-2.38 symbols and nothing else
 * 2.38-specific (verified: `objdump -T libfabric.so.1 | grep GLIBC_2.38`):
 *
 *     (GLIBC_2.38) __isoc23_fscanf  __isoc23_sscanf
 *                   __isoc23_strtol  __isoc23_strtoul
 *
 * These are glibc's C23 standard-library redirections: when you compile with
 * a C23-capable GCC (the host's SLES15-SP6 toolchain) the headers redirect
 * plain fscanf/sscanf/strtol/strtoul to the __isoc23_* variants so that
 * -std=c2x code gets the C23 declarations. The redirection is a header/ABI
 * artifact, NOT a runtime feature: the implementations are the same number/
 * string parsers the transport lib has always used. So forwarding them to the
 * plain glibc functions (which 2.35 provides) is functionally identical here.
 *
 * The shim MUST be compiled in-container against glibc 2.35: on 2.35 the
 * headers do NOT redirect, so the shim's own fscanf/strtol calls resolve to
 * the plain 2.35 symbols (no recursion, no new __isoc23_* dependency).
 * Compiling it on the head host (glibc 2.38) would reintroduce the very
 * redirection we are shimming.
 *
 * Build (in-container):
 *   gcc -std=c11 -D_DEFAULT_SOURCE -shared -fPIC -O2 \
 *       -Wl,--version-script=glibc23_shim.map \
 *       glibc23_shim.c -o libglibc23_shim.so
 *
 * Load (the EDF replaces the inherited env at srun --environment, so bake
 * this into run.sh, not the outer env):
 *   export LD_PRELOAD="$RUNDIR/libglibc23_shim.so"
 * LD_PRELOAD places the shim first in the global symbol scope, so the
 * __isoc23_* references of libfabric.so.1, libcxi.so.1, libcurl.so.4 and
 * libsasl2.so.3 (the plugin's full transitive closure) all resolve to it.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

int __isoc23_fscanf(FILE *stream, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int r = vfscanf(stream, fmt, ap);
    va_end(ap);
    return r;
}

int __isoc23_sscanf(const char *str, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int r = vsscanf(str, fmt, ap);
    va_end(ap);
    return r;
}

long __isoc23_strtol(const char *nptr, char **endptr, int base)
{
    return strtol(nptr, endptr, base);
}

unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base)
{
    return strtoul(nptr, endptr, base);
}
