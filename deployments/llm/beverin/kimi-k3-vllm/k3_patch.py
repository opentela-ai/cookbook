"""Set up the gfx942 a16w4 flydsl MoE on vLLM (kimi-k3 image) in two parts:

KERNEL SIDE (this file, runs once on rank 0 inside the container):
  1. Copy /usr/local/lib/python3.12/dist-packages/aiter -> $K3/home/pylib/aiter
     (full tree so package imports resolve from the overlay), plus the csrc
     and aiter_meta siblings (aiter_types.py needs aiter_enum.h).
  2. Install sitecustomize.py (sibling of this file) -> $K3/home/pylib/
     so CPython auto-imports it at startup and forces the K3 SiTU MoE backend
     selector onto AITER on gfx942 (see sitecustomize.py for the rationale).
  3. In aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py, replace the two
     hardware direct-to-LDS DMA calls (rocdl.raw_ptr_buffer_load_lds) used by the
     A16W4 stage1/stage2 X-tile prefetch with an arch-conditional:
     gfx950 -> original hardware DMA; otherwise -> software fill
     (buffer_load v4i32 from global + llvm.store into LDS at the same
     lane-major addresses the hardware would have written).
  4. Split the K32 bf16 MFMA into two K16 bf16_1k MFMAs on gfx942 (CDNA4-only
     instruction on gfx950).
  5. Force software fp4->bf16 dequant (use_hw_cvt default False) on gfx942.
  6. Disable hardcoded use_async_copy in fused_moe.py on gfx942 (K3_NO_ASYNC=1).

Idempotent (markers GFX942_SW_LDS_FILL / GFX942_K16_SPLIT / GFX942_SW_CVT /
GFX942_ASYNC_OFF). Strictly asserts expected source text.

Vendored verbatim from the working bring-up at
/capstor/scratch/cscs/xyao/kimi-k3/k3_patch.py (verified: BOOT / init under
vLLM 0.1.dev19253+g5f76ae224.rocm723, job k3-eng11). sitecustomize.py is the
Python counterpart (vendored from k3-eng11's overlay) — without it, job 580844
hit `NotImplementedError: No MXFP4 MoE backend supports the deployment
configuration.` at ~4 min because on_gfx950() is False on real MI300A. See
README.md for the kernel rationale (xkernels issues pending).
"""
import os
import shutil
import sys

K3 = os.environ["K3"]
SRC = "/usr/local/lib/python3.12/dist-packages/aiter"
DST = os.path.join(K3, "home/pylib/aiter")
TARGET_REL = "ops/flydsl/kernels/mixed_moe_gemm_2stage.py"

if not os.path.isdir(DST):
    print(f"copying {SRC} -> {DST} ...", flush=True)
    shutil.copytree(SRC, DST, symlinks=False)
    print("copy done", flush=True)
else:
    print("overlay already present", flush=True)

# aiter_types.py expects aiter_enum.h at <pylib>/csrc/include/ — copy sibling dirs
for sib in ("csrc", "aiter_meta"):
    s = os.path.join("/usr/local/lib/python3.12/dist-packages", sib)
    d = os.path.join(K3, "home/pylib", sib)
    if os.path.isdir(s) and not os.path.isdir(d):
        print(f"copying {s} -> {d} ...", flush=True)
        shutil.copytree(s, d, symlinks=False)

# --- install sitecustomize.py into the overlay (gfx942 MoE-backend fix) ---
# Auto-imported by CPython at startup (it sits at the overlay root, which
# engine.sh puts first on PYTHONPATH) and forces vLLM's K3 SiTU MoE selector
# onto the AITER backend on gfx942. Without it, on_gfx950() is False on real
# MI300A -> oracle -> NotImplementedError before any shard loads. Always copy
# fresh so the overlay is deterministic; siblings wait on PATCH_MARKER (touched
# by engine.sh only after this script returns) and never import a half-set-up
# overlay. See sitecustomize.py for the full rationale.
_sc_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sitecustomize.py")
_sc_dst = os.path.join(K3, "home/pylib", "sitecustomize.py")
if not os.path.isfile(_sc_src):
    print(f"FATAL: sitecustomize.py not found next to k3_patch.py: {_sc_src}", flush=True)
    sys.exit(1)
shutil.copyfile(_sc_src, _sc_dst)
print(f"installed sitecustomize.py -> {_sc_dst}", flush=True)

# --- install vkernels_experts.py + libvkernels_hip.so into the overlay ---
# VkernelFusedExperts (imported by sitecustomize.py) calls the vkernels HIP
# C ABI via ctypes. The .so must be on the container's filesystem; we place
# it in $K3/home/pylib/ alongside sitecustomize.py. VKERNELS_DIR points at
# the vkernels build tree (set by serve_kimi_k3_otela_beverin.sbatch).
_ve_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "vkernels_experts.py")
if os.path.isfile(_ve_src):
    _ve_dst = os.path.join(K3, "home/pylib", "vkernels_experts.py")
    shutil.copyfile(_ve_src, _ve_dst)
    print(f"installed vkernels_experts.py -> {_ve_dst}", flush=True)
else:
    print("WARN: vkernels_experts.py not found next to k3_patch.py "
          f"({_ve_src}), vkernels backend disabled", flush=True)

# --- install vkernels_attn.py into the overlay (issue #42 MLA+KDA) ---
# sitecustomize.py imports register_vkernels_attn from this module (alongside
# vkernels_experts.py, which it also imports). Same overlay location so both
# the import and the ctypes C ABI load resolve on the worker.
_va_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "vkernels_attn.py")
if os.path.isfile(_va_src):
    _va_dst = os.path.join(K3, "home/pylib", "vkernels_attn.py")
    shutil.copyfile(_va_src, _va_dst)
    print(f"installed vkernels_attn.py -> {_va_dst} (issue #42 MLA+KDA)",
          flush=True)
else:
    print("WARN: vkernels_attn.py not found next to k3_patch.py "
          f"({_va_src}), VkernelMLA/VkernelKDA disabled", flush=True)

import glob as _glob_mod
_vdir = os.environ.get("VKERNELS_DIR", "/capstor/scratch/cscs/xyao/vkernels")
# WHY prefer build/hip/ over build/cabi/: the build/cabi/ .so (PR #44) is
# linked against libamdhip64.so.6 (old ROCm 6), but the
# vllm-openai-rocm:kimi-k3 container is ROCm 7.2.3 and only ships
# libamdhip64.so.7. Loading the build/cabi/ .so fails at the first MoE
# forward with
#   OSError: libamdhip64.so.6: cannot open shared object file
# (job 598876, ~1h32m in, during determine_available_memory -> profile_run
# -> VkernelFusedExperts.apply -> ctypes.CDLL). build/hip/ links against
# libamdhip64.so.7 and is the build the verified run (597880) used.
# vkernels_experts._resolve_moe_fn already falls back to the legacy
# vk_fused_moe_mxfp4 symbol that build/hip/ exports. Re-enable build/cabi/
# only after it is rebuilt against ROCm 7 (set VKERNELS_BUILD=cabi).
_build_pref = os.environ.get("VKERNELS_BUILD", "hip")
_build_order = ("cabi", "hip") if _build_pref == "cabi" else ("hip", "cabi")
_so_cands = []
for _b in _build_order:
    _so_cands += _glob_mod.glob(
        os.path.join(_vdir, "build", _b, "**", "libvkernels_hip.so"),
        recursive=True)
if _so_cands:
    _so_dst = os.path.join(K3, "home/pylib", "libvkernels_hip.so")
    shutil.copyfile(_so_cands[0], _so_dst)
    print(f"installed libvkernels_hip.so -> {_so_dst} "
          f"(from {_so_cands[0]})", flush=True)
else:
    print(f"WARN: libvkernels_hip.so not found under {_vdir}, "
          "vkernels backend will fall back to Triton", flush=True)

t = os.path.join(DST, TARGET_REL)
# always start from the pristine image copy -> deterministic, no drift
import shutil as _sh
_sh.copy(os.path.join(SRC, TARGET_REL), t)
print("restored pristine kernel source", flush=True)
src = open(t).read()

OLD_G1 = """                        lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                        lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                        rocdl.raw_ptr_buffer_load_lds(
                            x_rsrc,
                            lds_ptr,
                            arith.constant(_dma_bytes, type=i32),
                            global_offset,
                            arith.constant(0, type=i32),
                            arith.constant(0, type=i32),
                            arith.constant(0, type=i32),
                        )
"""

NEW_G1 = """                        lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                        lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                        # GFX942_SW_LDS_FILL: gfx942 lacks direct-to-LDS buffer
                        # DMA (CDNA4-only); replicate lane-major fill in VGPRs.
                        if gpu_arch == "gfx950":
                            rocdl.raw_ptr_buffer_load_lds(
                                x_rsrc,
                                lds_ptr,
                                arith.constant(_dma_bytes, type=i32),
                                global_offset,
                                arith.constant(0, type=i32),
                                arith.constant(0, type=i32),
                                arith.constant(0, type=i32),
                            )
                        else:
                            _dw = arith.shrui(
                                global_offset, arith.constant(2, type=i32)
                            )
                            _vec = buffer_ops.buffer_load(
                                x_rsrc, _dw, vec_width=4, dtype=i32
                            )
                            _lds_thr = lds_ptr_i64 + arith.index_cast(
                                i64, lane_id * arith.constant(16, index=True)
                            )
                            _lds_thr_ptr = llvm.inttoptr(
                                lds_ptr_type, _lds_thr
                            )
                            llvm.store(_vec, _lds_thr_ptr, alignment=16)
"""

# gemm2 uses 28-space indentation for the same block
OLD_G2 = OLD_G1.replace("                        ", "                            ")
NEW_G2 = NEW_G1.replace("                        ", "                            ")

if "GFX942_SW_LDS_FILL" in src:
    print("LDS-fill patch already applied", flush=True)
else:
    n1 = src.count(OLD_G1)
    n2 = src.count(OLD_G2)
    print(f"found {n1} gemm1 block(s), {n2} gemm2 block(s)", flush=True)
    assert n1 == 1, f"expected exactly 1 gemm1 DMA block, found {n1}"
    assert n2 == 1, f"expected exactly 1 gemm2 DMA block, found {n2}"
    src = src.replace(OLD_G1, NEW_G1).replace(OLD_G2, NEW_G2)

# --- patch 2: K32 bf16 MFMA -> two K16 bf16_1k MFMAs on gfx942 ---
# gfx942 has no llvm.amdgcn.mfma.f32.16x16x32.bf16 (CDNA4-only -> 'Cannot select'.
# gfx942 does have 16x16x16.bf16.1k. Split the <8 x bf16> operands and chain 2 ops.
OLD_M = """        def mfma_f32_bf16_k32(result_type, operands, *, loc=None, ip=None):
            a, b, c, cbsz, abid, blgp = _split_mfma(operands)
            return _mfma_k32_raw(result_type, a, b, c, cbsz, abid, blgp, loc=loc, ip=ip)"""

NEW_M = """        def mfma_f32_bf16_k32(result_type, operands, *, loc=None, ip=None):
            a, b, c, cbsz, abid, blgp = _split_mfma(operands)
            if gpu_arch == "gfx950":
                return _mfma_k32_raw(result_type, a, b, c, cbsz, abid, blgp, loc=loc, ip=ip)
            # GFX942_K16_SPLIT: two K16 bf16_1k MFMAs on <4 x bf16> halves
            # (dialect requires vector<4xi16> operands -> bitcast).
            _vec4_bf16 = T.vec(4, T.bf16)
            _vec4_i16 = T.vec(4, T.i16)

            def _halves(v):
                e = [
                    vector.extract(v, static_position=[i], dynamic_position=[])
                    for i in range(8)
                ]
                return (
                    vector.bitcast(_vec4_i16, vector.from_elements(_vec4_bf16, e[0:4])),
                    vector.bitcast(_vec4_i16, vector.from_elements(_vec4_bf16, e[4:8])),
                )

            def _k16(va, vb, vc):
                return rocdl.mfma_f32_16x16x16bf16_1k(
                    result_type, [va, vb, vc, cbsz, abid, blgp]
                )

            a_lo, a_hi = _halves(a)
            b_lo, b_hi = _halves(b)
            acc = _k16(a_lo, b_lo, c)
            return _k16(a_hi, b_hi, acc)"""

if OLD_M in src:
    nm = src.count(OLD_M)
    print(f"found {nm} mfma_k32 helper(s), patching", flush=True)
    assert nm == 2, f"expected 2 mfma_k32 helpers, found {nm}"
    src = src.replace(OLD_M, NEW_M)
    print("MFMA-split patch applied", flush=True)
elif NEW_M in src:
    print("MFMA-split patch already applied", flush=True)
else:
    # tolerate one previous buggy variant: force-replace the old patched helper
    import re
    pat = re.compile(
        r"        def mfma_f32_bf16_k32\(result_type, operands, \*, loc=None, ip=None\):"
        r".*?(?=\n        \S)",
        re.S,
    )
    hits = pat.findall(src)
    assert len(hits) > 0 and all("GFX942_K16_SPLIT" in h for h in hits), (
        f"unexpected mfma helper variants: {len(hits)}"
    )
    src = pat.sub(NEW_M, src)
    print(f"force-replaced {len(hits)} previous MFMA-split variant(s)", flush=True)

open(t, "w").write(src)

# sanity: file still parses
import ast

ast.parse(src)

# --- patch 3: force software fp4->bf16 dequant (use_hw_cvt default False) ---
PIPE_REL = "ops/flydsl/kernels/mfma_preshuffle_pipeline.py"
pt = os.path.join(DST, PIPE_REL)
_sh.copy(os.path.join(SRC, PIPE_REL), pt)
psrc = open(pt).read()
OLD_D = "def unpack_b_mxfp4_bf16(packed32, arith, vector, scale_f32=None, use_hw_cvt=True):"
NEW_D = "def unpack_b_mxfp4_bf16(packed32, arith, vector, scale_f32=None, use_hw_cvt=False):  # GFX942_SW_CVT"
nd = psrc.count(OLD_D)
assert nd == 1, f"expected 1 unpack default, found {nd}"
open(pt, "w").write(psrc.replace(OLD_D, NEW_D))
ast.parse(open(pt).read())
print("sw-cvt patch applied", flush=True)

# --- patch 4: disable hardcoded use_async_copy in fused_moe.py ---
FMOE_REL = "fused_moe.py"
ft = os.path.join(DST, FMOE_REL)
_sh.copy(os.path.join(SRC, FMOE_REL), ft)
fsrc = open(ft).read()
OLD_A = "use_async_copy=True,"
NEW_A = "use_async_copy=(os.environ.get(\"K3_NO_ASYNC\") != \"1\"),  # GFX942_ASYNC_OFF"
na = fsrc.count(OLD_A)
assert na >= 1, f"expected >=1 use_async_copy=True, found {na}"
fsrc = fsrc.replace(OLD_A, NEW_A)
open(ft, "w").write(fsrc)
ast.parse(open(ft).read())
print(f"async-off patch applied ({na} site(s))", flush=True)

# --- patch 5: add Kimi-K3 shape to mxfp4_moe_aux codegen (job 581700 lesson) ---
# module_moe_mxfp4_aux is a JIT-compiled HIP module (NOT cached for gfx942)
# that provides mxfp4_moe_sort / mxfp4_moe_quant / mxfp4_moe_sort_scales /
# mxfp4_moe_scatter_reduce / mxfp4_moe_scatter_reduce_q.  These are the
# ORCHESTRATION ops (token sorting, per-block MXFP4 quant, output
# scatter-reduce) that bracket the FlyDSL gemm1/gemm2 (which k3_patch.py
# already patches for gfx942).
#
# The module uses a codegen'd C++ lookup table keyed on (NE, TOPK, MB,
# D_HIDDEN).  gen_instances.py enumerates a hardcoded SHAPES list of
# (NE, D_HIDDEN, D_INTER, TOPK) tuples for known models (Kimi-K2/K2.5,
# DSR, minimax, qwen, dsv4) but does NOT include Kimi-K3
# (NE=112, D_HIDDEN=7168, D_INTER=3072, TOPK=16 with TP=8).  At runtime
# the C++ aux_find() does TORCH_CHECK(key in table, ...) -> crash:
#   "no codegen'd instance for shape key 'aux_sort3s_NE112_TOPK16_MB32'"
#   (See moe_aux/codegen/gen_instances.py (enumerate_instances).)
#
# The .cu sources are self-contained (no CK / rocprim / hip_reduce.h
# header conflict that crashed module_moe_asm), so the rebuild for
# gfx942 will succeed once the shape is in the codegen.
#
# All six functions and their MB ranges (sort_quant MB=32, sort3stage
# MB in {32,64,128}, sort_only_zi MB=16, sort_only MB=16, quant MB in
# {32,64,128}, sort_scales BM in {32,64,128}, scatter keyed by (H,
# TOPK, NT)) are generated for every shape in SHAPES, so adding the
# K3 tuple covers every dispatch path the profiling forward pass may
# take regardless of the tuned/untuned config's block_m.
META_SRC = os.path.join("/usr/local/lib/python3.12/dist-packages", "aiter_meta")
META_DST = os.path.join(K3, "home/pylib/aiter_meta")
GEN_REL = "csrc/kernels/mxfp4_moe/moe_aux/codegen/gen_instances.py"
gen_src = os.path.join(META_SRC, GEN_REL)
gen_dst = os.path.join(META_DST, GEN_REL)
os.makedirs(os.path.dirname(gen_dst), exist_ok=True)
_sh.copy(gen_src, gen_dst)  # fresh copy from source -> deterministic
gs = open(gen_dst).read()
OLD_SHAPES_END = """    (385, 7168, 512, 7),  # dsv4 NE=385 TOPK=7 (tp6/tp8)
]"""
NEW_SHAPES_END = """    (385, 7168, 512, 7),  # dsv4 NE=385 TOPK=7 (tp6/tp8)
    (112, 7168, 3072, 16),  # Kimi-K3 TP=8 (NE=896/8, D_INTER=3072, TOPK=16)
]"""
if "(112, 7168, 3072, 16)" in gs:
    print("mxfp4_moe_aux codegen: K3 shape already present", flush=True)
else:
    assert OLD_SHAPES_END in gs, "expected SHAPES list end marker in gen_instances.py"
    gs = gs.replace(OLD_SHAPES_END, NEW_SHAPES_END)
    open(gen_dst, "w").write(gs)
    ast.parse(gs)
    print("mxfp4_moe_aux codegen: K3 shape added (K3_MOE_AUX_CODEGEN)", flush=True)

# --- patch 6: fix module_quant compile on gfx942 (job 581812 lesson) ---
# module_quant (quant_kernels.cu + quant_mxfp4.cu + quant_pybind.cu) is the
# last at-risk JIT module on the K3 MXFP4 MoE path.  It fails to JIT-build on
# gfx942 with two distinct errors:
#
#  (a) rocprim/iterator/texture_cache_iterator.hpp:178 calls memset() in HOST
#      code, but the only visible memset is the __device__ overload from
#      hip/amd_detail/amd_device_functions.h (pulled in via
#      aiter_hip_common.h -> hip/hip_runtime.h).  rmsnorm_quant_kernels.cu
#      avoids this because it includes py_itfs_common.h -> <torch/all.h> ->
#      <cstring> (host ::memset) BEFORE aiter_opus_plus.h -> rocprim.
#      Fix: add `#include <cstring>` at the very top of both quant .cu files,
#      BEFORE aiter_hip_common.h -> hip/hip_runtime.h.
#
#  (b) quant_mxfp4.cu:121 `static_cast<float>(__half)` fails because the build
#      passes -D__HIP_NO_HALF_CONVERSIONS__=1, which removes the explicit
#      conversion operator.  The gfx950 #if-branch uses the same cast but is
#      dead code on gfx942; the #else branch (lines 119-123) is what we
#      compile.  Fix: `#undef __HIP_NO_HALF_CONVERSIONS__` at the top of
#      quant_mxfp4.cu, restoring the explicit __half->__float (and
#      __bf16->__float) conversion operator for the rest of the TU.
#      Safer than hand-rewriting casts because `float_type` is a template
#      parameter that may be __half or __bf16, and undef handles both.
_SPDX = "// SPDX-License-Identifier: MIT\n"
_QUANT_SENTINEL = "/* K3-gfx942-quant-fix */"
for _rel, _adds in [
    (
        "csrc/kernels/quant_kernels.cu",
        ["#include <cstring>  // K3-gfx942-quant-fix: host ::memset for rocprim texture_cache_iterator"],
    ),
    (
        "csrc/kernels/quant_mxfp4.cu",
        [
            "#include <cstring>  // K3-gfx942-quant-fix: host ::memset for rocprim texture_cache_iterator",
            "#ifdef __HIP_NO_HALF_CONVERSIONS__",
            "#undef __HIP_NO_HALF_CONVERSIONS__  // K3-gfx942-quant-fix: restore explicit __half->__float cast",
            "#endif",
        ],
    ),
]:
    _qp = os.path.join(META_DST, _rel)
    _sh.copy(os.path.join(META_SRC, _rel), _qp)  # fresh copy -> deterministic
    _qs = open(_qp).read()
    if _QUANT_SENTINEL in _qs:
        print(f"quant-fix already applied: {_rel}", flush=True)
        continue
    assert _qs.startswith(_SPDX), f"expected SPDX header in {_rel}"
    _block = _QUANT_SENTINEL + "\n" + "".join(l + "\n" for l in _adds)
    open(_qp, "w").write(_qs[: len(_SPDX)] + "\n" + _block + _qs[len(_SPDX) :])
    print(f"quant-fix applied: {_rel}", flush=True)



# --- patch 8: hip_flag_checker must strip PYTHONPATH (job 582964 lesson) ---
# hip_flag_checker (jit/core.py:492) runs:
#   subprocess.check_output([hipcc, flag, "-x", "hip", "-E", "-P",
#                             "/dev/null", "-o", "/dev/null"], stderr=DEVNULL)
# It INHERITS the full environment, including PYTHONPATH=$K3/home/pylib.
# hipcc is an ELF binary but its device-libs linking step spawns a Python
# subprocess that picks up PYTHONPATH, imports `aiter`, and triggers JIT
# module import -- which writes "[aiter] import [module_aiter_core] ..." to
# stderr and returns non-zero.  hip_flag_checker sees CalledProcessError and
# returns False -> EVERY flag is "not supported" and filtered out, INCLUDING
# -D__Float4_e2m1fn_x2 (the fp4x2 quant enable flag we need on gfx942).
#
# Confirmed: with PYTHONPATH absent, hip_flag_checker("-D__Float4_e2m1fn_x2")
# returns True.  With PYTHONPATH present (any value), it returns False.
#
# FIX: pass a minimal env (PATH + HOME only, i.e. strip PYTHONPATH) to the
# hipcc subprocess.  This mirrors what a clean shell would do and is the
# minimal change that makes hip_flag_checker work.
_HFC_OLD = '''@functools.lru_cache()
def hip_flag_checker(flag_hip: str) -> bool:
    import subprocess

    cmd = (
        [executable_path("hipcc")]
        + flag_hip.split()
        + ["-x", "hip", "-E", "-P", "/dev/null", "-o", "/dev/null"]
    )
    try:
        subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        logger.warning(f"Current hipcc not support: {flag_hip}, skip it.")
        return False
    return True'''
_HFC_NEW = '''@functools.lru_cache()
def hip_flag_checker(flag_hip: str) -> bool:
    import subprocess

    cmd = (
        [executable_path("hipcc")]
        + flag_hip.split()
        + ["-x", "hip", "-E", "-P", "/dev/null", "-o", "/dev/null"]
    )
    # K3_hip_flag_clean_env: strip PYTHONPATH so hipcc's device-libs linker
    # subprocess does not import `aiter` (which triggers JIT and returns
    # non-zero, causing EVERY flag to be "not supported").  PATH + HOME are
    # sufficient for the preprocessor-only probe.
    _env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        subprocess.check_output(cmd, stderr=subprocess.DEVNULL, env=_env)
    except subprocess.CalledProcessError:
        logger.warning(f"Current hipcc not support: {flag_hip}, skip it.")
        return False
    return True'''
_jt = os.path.join(DST, "jit/core.py")
_js = open(_jt).read()
if "K3_hip_flag_clean_env" in _js:
    print("hip_flag_checker patch already applied", flush=True)
else:
    assert _HFC_OLD in _js, "expected hip_flag_checker body in jit/core.py"
    open(_jt, "w").write(_js.replace(_HFC_OLD, _HFC_NEW, 1))
    ast.parse(open(_jt).read())
    print("hip_flag_checker patch applied (K3_hip_flag_clean_env)", flush=True)

# --- patch 9: enable -D__Float4_e2m1fn_x2 on gfx942 (job 581813 lesson) ---
# The AITER JIT framework (jit/core.py:881) deliberately EXCLUDES gfx942
# from -D__Float4_e2m1fn_x2:
#
#     if get_gfx() != "gfx942" and int(os.getenv("AITER_FP4x2", "1")) > 0:
#         flags_hip += ["-D__Float4_e2m1fn_x2"]
#
# This compiles out the fp4x2 output path in quant_kernels.cu (lines 788,
# 823, 938, 1106, 1453, 1671, 1979), causing a runtime crash:
#   "operator() not support output type: fp4x2"
# when per_1x32_mx_quant_hip (K3 MXFP4 MoE activation quant) is called.
#
# The fix: remove the `get_gfx() != "gfx942"` condition so the flag is
# added on ALL archs (including gfx942).  This is SAFE because:
#   - The fp4x2 quant kernel (opus::fp4_t) does bit manipulation, not
#     hardware MXFP4 instructions
#   - --offload-arch=native still compiles for real gfx942 hardware
#   - get_gfx_runtime() (used for runtime dispatch) is unaffected
#   - Only affects JIT BUILD flags, not any runtime get_gfx() checks
_FP4_OLD = '        if get_gfx() != "gfx942" and int(os.getenv("AITER_FP4x2", "1")) > 0:'
_FP4_NEW = '        if md_name == "module_quant" and int(os.getenv("AITER_FP4x2", "1")) > 0:  # K3_gfx942_fp4x2_quant_only: enable fp4x2 quant on all archs'
_js = open(_jt).read()
if "K3_gfx942_fp4x2" in _js:
    print("jit/core.py fp4x2 patch already applied", flush=True)
else:
    assert _FP4_OLD in _js, "expected get_gfx() != gfx942 condition in jit/core.py"
    open(_jt, "w").write(_js.replace(_FP4_OLD, _FP4_NEW, 1))
    ast.parse(open(_jt).read())
    print("jit/core.py fp4x2 patch applied (K3_gfx942_fp4x2)", flush=True)

# --- patch 10: FlyDSL MoE LDS limit fix for gfx942 (jobs 583297 + 583591) ---
# The heuristic FlyDSL fallback in fused_moe.py (lines 2244-2251) picks
# stage-1 tile_m from a token tier. EMPIRICAL: the LDS of moe_gemm1_0 is set
# by the TILE GEOMETRY encoded in the kernel name (t{m}x128x256), NOT by the
# _w/_bnt suffix:
#   job 583297: kn1=...t128x128x256_w2_bnt0 -> LDS 131072 > 65536 (crash)
#   job 583591: kn1=...t128x128x256 (suffix stripped) -> LDS 131272... same
#   131072 > 65536 (crash)  => tile_m=128 itself does not fit gfx942.
# gfx942 (MI300A) has 64KB LDS per workgroup.  v3 measurement (thread of
# jobs): t32x128x256 alone -> LDS 82944 = A-tile bf16 dbuf (32*256*2*2
# 32768) + W-tile fp4 packed dbuf (256*128/2*2 32768) + scales + misc.
# So clamping tile_m is NOT enough; the gfx942 name must also shrink
# tile_n and tile_k.  m=32, n=64, k=128 -> ~27KB base + overhead: fits the
# 65536 B limit with wide margin (~2x headroom under estimation error).
# The error signature:
#   "local memory (82944) exceeds limit (65536) in function 'moe_gemm1_0'"
# Fix: on gfx942, clamp stage-1 tile_m to 32, rename stage-1 geometry to
# t32x64x128, and drop the stage-1 suffix (waves_per_eu=1).  Stage 2 is
# unchanged: its t{m}x128x{tk} tile fits (jobs only ever errored on
# moe_gemm1_0).
_flydsl_t = os.path.join(DST, "fused_moe.py")
_flydsl_src = open(_flydsl_t).read()
if "K3_gfx942_lds_clamp" in _flydsl_src:
    print("fused_moe.py LDS tile-clamp patch already applied", flush=True)
else:
    _LDS_OLD = '''        if token < 2048:
            _tile_m, _s1_sfx, _s2_sfx = 32, "_w2", "_bnt2"
        elif token < 4096:
            _tile_m, _s1_sfx, _s2_sfx = 64, "_w3_bnt0", ""
        elif token < 16384:
            _tile_m, _s1_sfx, _s2_sfx = 128, "_w2_bnt0", ""
        else:
            _tile_m, _s1_sfx, _s2_sfx = 64, "_w4_bnt0", ""'''
    _LDS_NEW = '''        if token < 2048:
            _tile_m, _s1_sfx, _s2_sfx = 32, "_w2", "_bnt2"
        elif token < 4096:
            _tile_m, _s1_sfx, _s2_sfx = 64, "_w3_bnt0", ""
        elif token < 16384:
            _tile_m, _s1_sfx, _s2_sfx = 128, "_w2_bnt0", ""
        else:
            _tile_m, _s1_sfx, _s2_sfx = 64, "_w4_bnt0", ""
        # K3_gfx942_lds_clamp: gfx942 (MI300A) has 64KB LDS per CU.  Jobs
        # 583845/583929/583949 ALL fell back to base ...t32x{128,64}x256
        # (wpe=1, bnt=2, kw=1 from dict) and ALL report LDS 82944.
        # Explicit allocations are only 32768 (2 x tile_m*lds_stride*2,
        # tile_n-insensitive); 50176 is implicit pipeline LDS.  The base
        # already has waves_per_eu=1 (NO _w1 suffix -- wpe=1 is default,
        # confirmed by probe_knames.py).  Remaining levers: b_nt 2->0 and
        # k_wave 1->4 (K=256 sliced 4x64, smaller resident state/wave).
        # _bnt0_kw4 is a direct key in _KERNEL_PARAMS (probe-verified).
        if get_gfx() == "gfx942":
            _tile_m = min(_tile_m, 32)
            _s1_sfx = "_bnt0_kw4"
            _s2_sfx = ""'''
    assert _LDS_OLD in _flydsl_src, "expected token-tier block in fused_moe.py"
    _flydsl_src = _flydsl_src.replace(_LDS_OLD, _LDS_NEW, 1)
    open(_flydsl_t, "w").write(_flydsl_src)
    ast.parse(open(_flydsl_t).read())
    print("fused_moe.py LDS tile-clamp patch applied (K3_gfx942_lds_clamp)", flush=True)

# --- patch 10b: gfx942 stage-1 kernel geometry t{m}x64x256 (fits 64KB) ---
# v2 (tile_m clamp alone, job 583845) still produced LDS 82944 > 65536 in
# moe_gemm1_0; the W-tile (k*n packed-fp4, double-buffered) dominates at
# n=128, k=256.  moe_kernels.py grammar for fp4-weight moe1 at tm=32:
#   tile_ns = [32, 64, 128], tile_ks = [256]  (!! k is fixed at 256)
# Job 583860 tried t32x64x128 -> ValueError (k must be 256).
# Job 583915 tried t32x32x256 -> LDS OK but hot_loop_scheduler ZeroDiv:
#   num_acc_n = (tile_n // num_waves) // 16 = 0 for tile_n=32; VALID minimum
#   tile_n is 64 (num_acc_n = 16//16 = 1).
# Exact-fit LDS model from two measured anchors (err <= 0.8 KB):
#   LDS ~= A(m*k*2) + W_dbuf(k*n) + scales(k/32*n*4) + ~28.7K
#   t128x128x256: 65536+32768+4096+28672  = 131072 (= measured, EXACT)
#   t32x128x256 : 16384+32768+4096+28928  =  83176 (~ measured 82944)
#   t32x64x256  : 16384+16384+2048+28928  =  63744  <= 65536  -> FITS
_G1_OLD = '''        _base_kn1 = flydsl_kernel_name(
            1, _a_type, _w_type, _out_type, _tile_m, 128, 256
        )'''
_G1_NEW = '''        _base_kn1 = flydsl_kernel_name(
            1, _a_type, _w_type, _out_type, _tile_m,
            (64 if get_gfx() == "gfx942" else 128),   # K3-G1-N tile_n
            256,
        )'''
_flydsl_src = open(_flydsl_t).read()
if "(64 if get_gfx()" in _flydsl_src:
    print("fused_moe.py gemm1-geometry patch already applied", flush=True)
else:
    assert _G1_OLD in _flydsl_src, "expected _base_kn1 call in fused_moe.py"
    _flydsl_src = _flydsl_src.replace(_G1_OLD, _G1_NEW, 1)
    open(_flydsl_t, "w").write(_flydsl_src)
    ast.parse(open(_flydsl_t).read())
    print("fused_moe.py gemm1-geometry patch applied (t32x64x256 on gfx942)", flush=True)


# --- patch 8: enable Triton SW MXFP4 GEMM (batched_gemm_a16wfp4_) on gfx942 ---
# arch_info.is_fp4_avail() returns True only for gfx950/gfx1250, but the A16WFP4
# GEMM kernels are PURE TRITON (tl.load/tl.dot/_mxfp4_quant_op software dequant).
# The guard is overly conservative.  On Kimi-K3 the ATTENTION weights are
# A16WFP4, so the forward pass hits this assert in every self_attn (linear.py:564
# -> batched_gemm_a16wfp4.py:104).  Adding gfx942 lets the Triton kernel compile
# and run.  is_gluon_avail stays False (real hardware MLA, not Triton).  See job
# 585846 AssertionError: "MXFP4 is not available on your device".
_ARCH_REL = "ops/triton/utils/_triton/arch_info.py"
_at = os.path.join(DST, _ARCH_REL)
_sh.copy(os.path.join(SRC, _ARCH_REL), _at)
_asrc = open(_at).read()
_OLD_FP4 = 'def is_fp4_avail():\n    return get_arch() in ("gfx950", "gfx1250")'
_NEW_FP4 = ('def is_fp4_avail():\n'
    '    return get_arch() in ("gfx950", "gfx1250", "gfx942")  '
    '# K3_gfx942_fp4_avail: Triton SW MXFP4 GEMM works on gfx942')
if "K3_gfx942_fp4_avail" in _asrc:
    print("arch_info.py is_fp4_avail patch already applied", flush=True)
else:
    assert _asrc.count(_OLD_FP4) == 1, ("expected 1 is_fp4_avail def",)
    open(_at, "w").write(_asrc.replace(_OLD_FP4, _NEW_FP4, 1))
    ast.parse(open(_at).read())
    print("arch_info.py is_fp4_avail patch applied (K3_gfx942_fp4_avail)", flush=True)

# --- patch 9: create gfx942 A16WFP4 GEMM config files (job 585991) ---
# Patch 8 enabled is_fp4_avail() for gfx942, so the A16WFP4 Triton GEMM
# kernels are now selected.  But AITER's get_gemm_config() requires a per-
# arch default tuning JSON (fpath_should_exist=True) and gfx942 has none:
#   AssertionError: Required config file doesn't exist:
#     .../gfx942-BATCHED_GEMM-A16WFP4.json
# (linear.py:580 forward_attn_residual -> batched_gemm_a16wfp4.py:112 ->
# get_gemm_config("BATCHED_GEMM-A16WFP4", ...)).  We clone from gfx950 (MI350,
# closest CDNA arch with a tuned A16WFP4 config) as a stopgap.  No
# PRESHUFFLED variant needed (is_mx_scale_preshuffling_avail() is False on
# gfx942, not patched).
import json as _json9
_CFG_DIR = os.path.join(DST, "ops/triton/configs/gemm")
_CFG_NOTE = (
    "gfx942 stopgap: cloned from gfx950 (MI350, closest CDNA arch with a "
    "tuned A16WFP4 config). Created by k3_patch.py patch 9 to satisfy AITER "
    "get_gemm_config() which requires a per-arch default tuning file "
    "(fpath_should_exist=True). NOT tuned for gfx942 (MI300A) -> potential "
    "perf and precision-validation risk; revisit if accuracy checks fail. "
    "K3_gfx942_a16wfp4_cfg"
)
for _src9, _dst9 in [
    ("gfx950-BATCHED_GEMM-A16WFP4.json", "gfx942-BATCHED_GEMM-A16WFP4.json"),
    ("gfx950-GEMM-A16WFP4.json", "gfx942-GEMM-A16WFP4.json"),
]:
    _dpath = os.path.join(_CFG_DIR, _dst9)
    _spath = os.path.join(_CFG_DIR, _src9)
    if os.path.exists(_spath):
        _cfg9 = _json9.load(open(_spath))
        if os.path.exists(_dpath) and "K3_gfx942_a16wfp4_cfg" in open(_dpath).read():
            print(f"{_dst9} already present (K3_gfx942_a16wfp4_cfg)", flush=True)
        else:
            _cfg9["_note"] = _CFG_NOTE
            _json9.dump(_cfg9, open(_dpath, "w"), indent=4)
            print(f"created {_dst9} (cloned from {_src9})", flush=True)
    else:
        print(f"WARN: {_src9} not found in {_CFG_DIR}", flush=True)

print("PATCH_OK", flush=True)
