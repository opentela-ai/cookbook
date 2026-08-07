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
        "kernels/quant_kernels.cu",
        ["#include <cstring>  // K3-gfx942-quant-fix: host ::memset for rocprim texture_cache_iterator"],
    ),
    (
        "kernels/quant_mxfp4.cu",
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

print("PATCH_OK", flush=True)
