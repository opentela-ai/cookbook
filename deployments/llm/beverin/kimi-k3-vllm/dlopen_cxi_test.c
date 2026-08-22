/* dlopen_cxi_test.c - prove the glibc23 shim lets the CSCS Cray libfabric
 * 1.29.1 (with the cxi provider builtin) LOAD on the glibc-2.35 image.
 *
 * This test answers ONE narrow question: does the shim bridge the
 * __isoc23_*@GLIBC_2.38 gap so libfabric.so.1.29.1 (and, transitively,
 * libcxi.so.1 + libcurl.so.4 + libsasl2.so.3) dlopen inside the image?
 * The full OFI/RCCL smoke (ofi_rccl_smoke.sbatch) does the real CXI
 * transport enumeration; here we only need a load/no-load signal.
 *
 * Deliberately links ONLY -ldl (no libfabric): without the shim, the
 * program still STARTS, so dlopen() runs and prints a clean dlerror()
 * instead of the dynamic loader aborting before main() with
 *   .../libc.so.6: version `GLIBC_2.38' not found (required by libfabric.so.1)
 * With the shim (LD_PRELOAD=libglibc23_shim.so), dlopen(RTLD_NOW) resolves
 * every undefined symbol up front and runs libfabric's constructors; success
 * + a present fi_getinfo means the cxi provider library is fully loaded.
 *
 * Built and run IN-CONTAINER (glibc 2.35) by glibc23_shim_test.sbatch.
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const char *lf = getenv("TARGET_LIBFABRIC");
    if (!lf) lf = "/opt/cray/libfabric/2.3.1/lib64/libfabric.so.1";

    fprintf(stderr, "[cxi] LD_PRELOAD=%s\n",
            getenv("LD_PRELOAD") ? getenv("LD_PRELOAD") : "<unset>");
    fprintf(stderr, "[cxi] dlopen(%s, RTLD_NOW|RTLD_LOCAL) ...\n", lf);

    /* RTLD_NOW resolves every undefined symbol up front (so a missing
     * __isoc23_*@GLIBC_2.38 fails HERE with dlerror(), not later inside
     * RCCL). RTLD_LOCAL keeps libfabric's symbols out of the global
     * namespace so the test links against the shim/host libc. */
    void *h = dlopen(lf, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "G23SHIM_DLOPEN_FAIL: %s\n", dlerror());
        return 2;
    }
    fprintf(stderr, "[cxi] dlopen OK (all symbols resolved, ctors ran)\n");

    void *fi_getinfo = dlsym(h, "fi_getinfo");
    void *fi_freeinfo = dlsym(h, "fi_freeinfo");
    void *fi_strerror = dlsym(h, "fi_strerror");
    if (!fi_getinfo || !fi_freeinfo || !fi_strerror) {
        fprintf(stderr, "G23SHIM_FAIL: not libfabric (missing fi_getinfo/"
                        "fi_freeinfo/fi_strerror): %s\n", dlerror());
        dlclose(h);
        return 3;
    }
    fprintf(stderr, "G23SHIM_PASS: libfabric 1.29.1 loaded under glibc %s; "
                    "cxi provider library present (fi_getinfo @ %p)\n",
            getenv("GLIBC_TEST_LABEL") ? getenv("GLIBC_TEST_LABEL") : "2.35",
            fi_getinfo);
    dlclose(h);
    return 0;
}
