# vkernels build + test on Beverin (MI300A / gfx942)

**Result: ALL GPU tests PASS. C ABI (`libvkernels_hip.so`) is the integration
surface for the vLLM MoE backend shim (Path A, chosen).**

## Integration approach — C ABI shared library

Instead of pybind11 (which has an LTO pruning issue with PyInit__core on
ROCm clang++ 19), the integration uses a plain C ABI shared library:

- `src/c/vkernels/hip_api.cpp` — `extern "C"` wrappers around
  `vkernels::kernels::hip::*`
- Built as `libvkernels_hip.so` (no pybind11, no LTO, no fvisibility=hidden)
- Loaded from Python with `ctypes.CDLL`, called with `tensor.data_ptr()`
- Validated by `test_capi_moe` (C++ calling C ABI) + `test_hip_bindings.py`
  (Python ctypes calling C ABI — exact values match C++ test)

### C ABI exports (3 kernels)

| Symbol | Wraps | Note |
|---|---|---|
| `vk_fused_moe_mxfp4` | `hip::fused_moe_mxfp4` | MXFP4 MoE grouped GEMM (SiTU via `activation=1`) |
| `vk_mla_fwd` | `hip::mla_fwd` | MLA forward (absorbed form) |
| `vk_kda_delta_rule_fwd` | `hip::kda_delta_rule_fwd` | KDA delta-rule forward |

## Files

| File | Purpose |
|---|---|
| `vkernels-rocm.toml` | EDF — reuses the SGLang ROCm image |
| `build_test_vkernels.sbatch` | Build (cmake --preset hip) + Phase 1/2 |
| `test_hip_bindings.py` | Python ctypes smoke test (C ABI validation) |

## Jobs

| Job | Result | Notes |
|---|---|---|
| **596027** | 8/8 C++ GPU tests PASS | First full validation on MI300A |
| **596222** | 8/8 + C ABI C++ PASS | Added `test_capi_moe` + `libvkernels_hip.so` |
| **596227** | 8/8 + C ABI + Python PASS | Python ctypes validated, exact match with C++ |

### GPU correctness tests: 8 passed, 0 failed (all jobs)

| Test | Validates | Result |
|---|---|---|
| `test_kda_correct` | KDA delta-rule fwd vs CPU | PASS (max_rel ≤ 8.5e-5) |
| `test_mla_correct` | MLA fwd vs CPU | PASS (max_rel ≤ 3e-6) |
| `test_gemm_bf16_correct` | bf16 MFMA GEMM vs CPU | PASS (bit-exact) |
| `test_moe_aux_correct` | MXFP4 sort/quant/reduce | PASS (bit-exact, 16/16) |
| `test_moe_fused_correct` (SwiGLU) | Fused MoE, SwiGLU | PASS (max_rel=4e-6) |
| `test_moe_fused_correct` (SiTU) | Fused MoE, **K3 SiTU** | PASS (max_rel=3e-6) |
| `test_moe_fused_prefill_correct` | Fused MoE prefill | PASS (max_rel=0.011) |
| `test_moe_fused_dist_correct` | Distributed MoE (TP=2) | PASS (max_rel=1e-6) |

### C ABI validation (jobs 596222+596227)

| Test | Result | Notes |
|---|---|---|
| `test_capi_moe` (C++ → C ABI) | **PASS** | with/without bias, nnz=2048, nan=0 |
| Python ctypes → C ABI (swiglu) | **PASS** | max_abs=231.97, exact match with C++ |
| Python ctypes → C ABI (situ) | **PASS** | max_abs=200.24, activation=1 works |
| Python ctypes → C ABI (mla) | **PASS** | max_abs=2.63 |
| Python ctypes → C ABI (kda) | **PASS** | max_abs=0.20 |

### What this means for Kimi-K3

Every kernel K3 needs now has a **verified working gfx942 implementation**
callable from Python via ctypes:

| K3 component | vLLM path (broken) | vkernels C ABI | Status |
|---|---|---|---|
| MXFP4 MoE | AITER (hang) | `vk_fused_moe_mxfp4` | ✅ validated |
| MLA attention | AITER (no tuned config) | `vk_mla_fwd` | ✅ validated |
| Kimi Delta Attn | KDA Triton (gfx950-only) | `vk_kda_delta_rule_fwd` | ✅ validated |
| **SiTU activation** | Hard-coded SwiGLU | `activation=1` | ✅ validated |

### Next steps (vLLM backend shim)

1. **Register `VKERNELS_MXFP4_BF16` backend** in `sitecustomize.py` (add to
   priority list alongside `TRITON_UNFUSED`)
2. **Write `VkernelFusedExperts`** (vLLM `Experts` subclass) that:
   - Takes vLLM's weight buffers (w13, w2, scales) and activation
   - Calls `vk_fused_moe_mxfp4` via ctypes
3. **Test K3 serving** with vkernels backend on 6-node TP8×PP3

## Build fix

- `meta/benchmarks/bench_moe.hip` missing `#include <chrono>` → vkernels #38
- `src/python/_core.cpp` HIP namespace fix: `hip::` → `kernels::hip::`
  (The pybind11 HIP bindings are not needed — C ABI is used instead.)

## Submission

```bash
rcc --profile beverin run --cwd \
  /capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/vkernels \
  -- sbatch build_test_vkernels.sbatch
```

## Source (on Beverin)

```
/capstor/scratch/cscs/xyao/vkernels/
  src/c/vkernels/hip_api.cpp           ← C ABI wrappers (extern "C")
  src/c/CMakeLists.txt                 ← adds vkernels_hip shared library target
  meta/benchmarks/test_capi_moe.hip   ← C++ C ABI validation test
  test_hip_bindings.py               ← Python ctypes smoke test
  build/hip/src/c/libvkernels_hip.so  ← the shared library
```

### C ABI loading (quick reference)

```python
import ctypes, torch

lib = ctypes.CDLL("/path/to/libvkernels_hip.so")

# MXFP4 fused MoE (activation: 0=SwiGLU, 1=SiTU/K3)
lib.vk_fused_moe_mxfp4(
    ctypes.c_void_p(dA.data_ptr()),   # bf16 activations [M, hidden]
    ctypes.c_void_p(dw13.data_ptr()), # packed MXFP4 gate_up [E, 2*ispp, hidden/2]
    ctypes.c_void_p(dw13s.data_ptr()),# ue8m0 gate_up scales [E, 2*ispp, hidden/32]
    ctypes.c_void_p(dw2.data_ptr()),  # packed MXFP4 down [E, hidden, ispp/2]
    ctypes.c_void_p(dw2s.data_ptr()), # ue8m0 down scales [E, hidden, ispp/32]
    ctypes.c_void_p(dTids.data_ptr()),# raw topk_ids [M*top_k]
    ctypes.c_void_p(dTw.data_ptr()),  # raw topk_w [M*top_k]
    ctypes.c_void_p(dact.data_ptr()), # act_scratch [EM, ispp] bf16
    ctypes.c_void_p(dout.data_ptr()), # output [M*hidden] fp32
    ctypes.c_int(M), ctypes.c_int(hidden), ctypes.c_int(ispp), ctypes.c_int(top_k),
    ctypes.c_void_p(dSids.data_ptr()),# sorted_ids (from moe_align_block_size)
    ctypes.c_void_p(dEids.data_ptr()),# expert_ids (from moe_align_block_size)
    ctypes.c_int(EM),                 # padded EM
    ctypes.c_float(4.0),              # swiglu_limit
    ctypes.c_int(activation),         # 0=SwiGLU, 1=SiTU
    ctypes.c_float(4.0),              # beta (SiTU)
    ctypes.c_float(25.0),             # linear_beta (SiTU)
    ctypes.c_void_p(db13.data_ptr()),# b13 bias [E*2*ispp]
    ctypes.c_void_p(db2.data_ptr()), # b2 bias [E*hidden]
    ctypes.c_int(16))                 # block_size (16=decode, 64=prefill)
torch.cuda.synchronize()
```
