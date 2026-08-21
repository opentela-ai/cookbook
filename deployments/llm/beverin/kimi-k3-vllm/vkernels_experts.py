"""vLLM MoE backend shim: VkernelFusedExperts.

Calls vkernels' HIP C ABI (libvkernels_hip.so) for MXFP4 fused MoE on
gfx942 (MI300A). Replaces the broken AITER/Triton MoE path for Kimi-K3.

The C ABI library is loaded via ctypes. All weight/scale/bias tensors are
passed as raw device pointers. The kernel does the full computation:
gate-up GEMM -> activation (SwiGLU/SiTU) -> down GEMM -> routing weight
application -> top-k summation. Output is fp32, converted to bf16.

moe_align_block_size is done on CPU (small cost for routing metadata,
O(M*top_k) tokens sorted by expert). This can be replaced with a GPU
implementation later for performance.

Validated on MI300A (gfx942) — all 8 C++ GPU tests + 4 Python ctypes
tests pass (job 596227). See deployments/llm/beverin/vkernels/README.md.
"""

import ctypes
import glob
import os

import numpy as np
import torch
from torch.profiler import record_function

from vllm.model_executor.layers.fused_moe.config import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
    UnfusedOAITritonExperts,
)
from vllm.platforms import current_platform


# ---------------------------------------------------------------------------
# Persistent scratch buffers (capture-safe)
# ---------------------------------------------------------------------------
# The vkernels HIP C ABI expects caller-provided, PERSISTENT scratch
# ("the launcher performs no device allocation of its own"): act_scratch is
# [EM, ispp] bf16 and out is [M, hidden] fp32, both supplied by the caller and
# reused across calls.  The original wrapper violated this with per-call
# `torch.empty()` for `out_fp32` (EVERY call) and an `act_scratch` fallback.
#
# Why that faults under CUDA-graph capture (the K3 breakable path):
#   `@eager_break_during_capture` runs `apply` via `BreakableCUDAGraphCapture
#   .add_eager`, which does `_end_segment()` (stream leaves capture) ->
#   `fn()` (apply, on a momentarily NON-capturing stream) -> `_begin_segment()`
#   (re-enters capture).  A `torch.empty()` here grabs fresh caching-allocator
#   storage whose address is NOT stable across replay, so when the captured
#   graph is replayed the C kernel is launched against memory the allocator has
#   recycled -> "Memory access fault by GPU node-X on address 0x...".  The same
#   happens during replay itself (apply re-runs between graph segments).
#
# Fix: allocate each (dev, ispp) act_scratch and (dev, hidden) out_fp32 ONCE,
# at the absolute maximum size (sized from MAX_NUM_BATCHED_TOKENS, which reaches
# the container via the `srun --environment=EDF` env merge), during the eager
# profile/warmup run (before any capture begins).  Every subsequent call —
# capture-time eager break AND replay — reuses the same storage (a sliced view)
# and never allocates.  We refuse to grow while a capture session is active
# (growing would free storage a captured graph still references -> replay
# fault); growth only happens on eager (non-capture-active) calls.
try:
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
    )
except Exception:  # pragma: no cover - older vLLM / non-cudagraph path
    BreakableCUDAGraphCapture = None

# (dev.index, ispp)   -> bf16  [max_EM*ispp]      (act_scratch storage)
_persistent_act: dict = {}
# (dev.index, hidden)  -> fp32  [max_M*hidden]     (out_fp32 storage)
_persistent_out: dict = {}
# (dev.index,)         -> int32 [max_EM]           (routing sids storage)
_persistent_sids: dict = {}
# (dev.index,)         -> int32 [max_EM]           (routing eids storage;
#                                  eids has EM//block_size <= EM els)
_persistent_eids: dict = {}


def _max_num_tokens_hint() -> int:
    """Best-effort global token budget for sizing persistent buffers.

    MAX_NUM_BATCHED_TOKENS reaches the container via the
    `srun --environment=$OTELA_EDF_NAME` env merge (the EDF does not list it;
    the outer sbatch `export` is merged in).  Falls back to a conservative
    constant so the first eager warmup call still over-allocates and we
# effectively never allocate again.
    """
    for k in ("MAX_NUM_BATCHED_TOKENS", "VLLM_MAX_NUM_BATCHED_TOKENS"):
        v = os.environ.get(k)
        if v:
            try:
                n = int(v)
                if n > 0:
                    return n
            except ValueError:
                pass
    return 8192


def _capture_active() -> bool:
    """True iff a (breakable) CUDA-graph capture session is active.

    Covers the eager-break window where `torch.cuda.is_current_stream_capturing()`
    is False but a `BreakableCUDAGraphCapture` context is still on the call
    stack (between `_end_segment()` and `_begin_segment()`); allocations here
    are unsafe because their addresses are not replay-stable.  Falls back to
    the raw PyTorch capture flag for the non-breakable capture path.
    """
    if BreakableCUDAGraphCapture is not None:
        try:
            if BreakableCUDAGraphCapture.is_active():
                return True
        except Exception:
            pass
    try:
        if torch.cuda.is_current_stream_capturing():
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_lib_cache = {}


def _find_libvkernels_hip():
    """Find libvkernels_hip.so, checking env var and common paths."""
    env_path = os.environ.get("VKERNELS_LIB")
    if env_path and os.path.exists(env_path):
        return env_path

    k3 = os.environ.get("K3", "")
    if k3:
        k3_path = os.path.join(k3, "home/pylib/libvkernels_hip.so")
        if os.path.exists(k3_path):
            return k3_path

    vdir = os.environ.get("VKERNELS_DIR", "/capstor/scratch/cscs/xyao/vkernels")
    cands = sorted(
        glob.glob(
            os.path.join(vdir, "build", "hip", "**", "libvkernels_hip.so"),
            recursive=True,
        )
    )
    if cands:
        return cands[0]

    return None


def _get_lib():
    """Load and cache libvkernels_hip.so."""
    if "lib" not in _lib_cache:
        path = _find_libvkernels_hip()
        if path is None:
            raise RuntimeError(
                "libvkernels_hip.so not found. Set VKERNELS_LIB or "
                "VKERNELS_DIR, or place it in $K3/home/pylib/"
            )
        _lib_cache["lib"] = ctypes.CDLL(path)
    return _lib_cache["lib"]


def _resolve_moe_fn(lib):
    """Return the fused-MoE C ABI function.

    Upstream PR #44 names it ``vk_hip_fused_moe_mxfp4`` (namespaced away
    from the CPU reference ``vk_fused_moe_mxfp4`` in capi.hpp).  Older
    local builds used the bare ``vk_fused_moe_mxfp4`` name from the
    since-removed ``hip_api.cpp``.  Try the new name first, fall back.
    """
    fn = getattr(lib, "vk_hip_fused_moe_mxfp4", None)
    if fn is None:
        fn = getattr(lib, "vk_fused_moe_mxfp4", None)
    if fn is None:
        raise RuntimeError(
            "neither vk_hip_fused_moe_mxfp4 nor vk_fused_moe_mxfp4 "
            "found in libvkernels_hip.so — rebuild with PR #44"
        )
    return fn


# ---------------------------------------------------------------------------
# moe_align_block_size (CPU reference, matching vkernels' expected format)
# ---------------------------------------------------------------------------


def _moe_align_block_size_cpu(
    topk_ids_flat, num_experts, block_size, expert_map=None
):
    """Pure-CPU moe_align_block_size, matching vkernels' expected format.

    Args:
        topk_ids_flat: [M*top_k] numpy int32 (global expert IDs)
        num_experts: global number of experts
        block_size: GEMM block size (16 decode, 64 prefill)
        expert_map: [num_experts] numpy int32 (global->local, -1=skip)

    Returns:
        sorted_ids: [EM] int32 (token indices, padded with N=M*top_k)
        expert_ids: [EM//block_size] int32 (local expert per block, -1=pad)
        EM: int (total padded token count)
    """
    N = len(topk_ids_flat)
    per_expert = {}
    for i in range(N):
        global_e = int(topk_ids_flat[i])
        if expert_map is not None:
            local_e = int(expert_map[global_e])
            if local_e == -1:
                continue
        else:
            local_e = global_e
        per_expert.setdefault(local_e, []).append(i)

    local_experts = sorted(per_expert.keys())
    EM = sum(
        ((len(per_expert[e]) + block_size - 1) // block_size) * block_size
        for e in local_experts
    )
    # Ensure EM is at least block_size (avoids zero-length arrays)
    if EM == 0:
        EM = block_size
    sids = np.full(EM, N, dtype=np.int32)
    eids = np.full(EM // block_size, -1, dtype=np.int32)

    idx = 0
    blk_idx = 0
    for e in local_experts:
        tokens = per_expert[e]
        for i in tokens:
            sids[idx] = i
            idx += 1
        padded_nt = ((len(tokens) + block_size - 1) // block_size) * block_size
        num_blocks = padded_nt // block_size
        for b in range(num_blocks):
            if b * block_size < len(tokens):
                eids[blk_idx] = e
            blk_idx += 1
        for _ in range(len(tokens), padded_nt):
            sids[idx] = N
            idx += 1

    return sids, eids, EM


# ---------------------------------------------------------------------------
# VkernelFusedExperts
# ---------------------------------------------------------------------------


class VkernelFusedExperts(UnfusedOAITritonExperts):
    """vkernels HIP C ABI backend for MXFP4 MoE on gfx942 (MI300A).

    Calls vk_hip_fused_moe_mxfp4 (PR #44) via ctypes.  The kernel does the
    computation: gate-up GEMM -> activation -> down GEMM -> routing weight
    application -> top-k summation. Output is fp32, converted to bf16.

    moe_align_block_size is done on CPU (small cost for routing metadata).
    Weight format: [E, 2*ispp, hidden/2] uint8 (w13), [E, hidden, ispp/2]
    uint8 (w2) — matches vLLM's Mxfp4MoEMethod.create_weights() exactly.
    """

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_rocm():
            return False
        # The library was compiled for gfx942 (MI300A).  Check the
        # actual device architecture to avoid a segfault on gfx90a.
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                gcn = getattr(props, "gcnArchName", "")
                return "gfx942" in gcn
        except Exception:
            pass
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SITU,  # Kimi-K3 SiTU
        ]

    # --- persistent scratch (see module docstring) -----------------------
    def _get_persistent_act(
        self, dev, ispp, EM, top_k, global_num_experts, local_num_experts
    ):
        """Return a persistent bf16 [EM, ispp] act_scratch view.

        Storage is keyed by (dev.index, ispp) and sized for the maximum
        token budget (MAX_NUM_BATCHED_TOKENS) on first use (the eager
        profile/warmup run).  Subsequent calls — including the capture-time
        eager break and replay — reuse the same storage (a sliced view) and
        never allocate.  If the storage is too small while a capture session
        is active we raise instead of allocating (which would corrupt the
        captured graph on replay).
        """
        key = (dev.index, ispp)
        st = _persistent_act.get(key)
        need = EM * ispp
        if st is not None and st.numel() >= need:
            return st[:need].view(EM, ispp)
        if _capture_active():
            cap = 0 if st is None else st.numel()
            raise RuntimeError(
                f"[VkernelFusedExperts] persistent act_scratch "
                f"(key={key}, cap={cap}) is too small for EM*ispp={need} "
                f"AND a CUDA-graph capture session is active — refusing to "
                f"allocate (would corrupt replay). Increase MAX_NUM_BATCHED_TOKENS "
                f"so the eager warmup pins a larger high-water mark, or disable "
                f"cudagraph capture for this layer."
            )
        # Size for the absolute max token budget (over-allocates exactly
        # once, on the eager profile/warmup run).  Use block_size=64 (the
        # large-M / prefill layout) for the high-water mark.
        max_M = _max_num_tokens_hint()
        local_n = (
            local_num_experts
            if local_num_experts and local_num_experts > 0
            else global_num_experts
        )
        max_EM = ((max_M * top_k + local_n + 63) // 64) * 64
        cap = max(max_EM * ispp, need)
        # Round up to a multiple of ispp for clean [., ispp] views.
        cap = ((cap + ispp - 1) // ispp) * ispp
        st = torch.empty(cap, dtype=torch.bfloat16, device=dev)
        _persistent_act[key] = st
        print(
            f"[VkernelFusedExperts] persistent act_scratch alloc "
            f"(key={key}, cap={cap} bf16, max_M={max_M}, top_k={top_k}, "
            f"local_n={local_n} -> max_EM={max_EM})",
            flush=True,
        )
        return st[:need].view(EM, ispp)

    def _get_persistent_out(self, dev, hidden, M):
        """Return a persistent fp32 [M, hidden] out_fp32 view.

        Storage is keyed by (dev.index, hidden) and sized for the maximum
        token budget on first use.  See `_get_persistent_act` for the
        capture-safety rationale.
        """
        key = (dev.index, hidden)
        st = _persistent_out.get(key)
        need = M * hidden
        if st is not None and st.numel() >= need:
            return st[:need].view(M, hidden)
        if _capture_active():
            cap = 0 if st is None else st.numel()
            raise RuntimeError(
                f"[VkernelFusedExperts] persistent out_fp32 "
                f"(key={key}, cap={cap}) is too small for M*hidden={need} "
                f"AND a CUDA-graph capture session is active — refusing to "
                f"allocate (would corrupt replay). Increase MAX_NUM_BATCHED_TOKENS "
                f"so the eager warmup pins a larger high-water mark, or disable "
                f"cudagraph capture for this layer."
            )
        max_M = _max_num_tokens_hint()
        cap = max(max_M * hidden, need)
        # Round up to a multiple of hidden for clean [., hidden] views.
        cap = ((cap + hidden - 1) // hidden) * hidden
        st = torch.empty(cap, dtype=torch.float32, device=dev)
        _persistent_out[key] = st
        print(
            f"[VkernelFusedExperts] persistent out_fp32 alloc "
            f"(key={key}, cap={cap} fp32, max_M={max_M})",
            flush=True,
        )
        return st[:need].view(M, hidden)

    @staticmethod
    def _max_em_count(max_M, top_k, local_n):
        """High-water mark for EM = padded(M*top_k + local_experts, 64)."""
        return ((max_M * top_k + local_n + 63) // 64) * 64

    @staticmethod
    def _local_n(local_num_experts, global_num_experts):
        return (
            local_num_experts
            if local_num_experts and local_num_experts > 0
            else global_num_experts
        )

    def _get_persistent_sids(
        self, dev, EM, top_k, global_num_experts, local_num_experts
    ):
        """Persistent int32 [EM] routing-sids view (copied into, never
        re-allocated).  Same capture-safety rationale as `_get_persistent_out`.
        """
        key = (dev.index,)
        st = _persistent_sids.get(key)
        if st is not None and st.numel() >= EM:
            return st[:EM]
        if _capture_active():
            cap = 0 if st is None else st.numel()
            raise RuntimeError(
                f"[VkernelFusedExperts] persistent sids (key={key}, "
                f"cap={cap}) is too small for EM={EM} AND a CUDA-graph "
                f"capture session is active — refusing to allocate "
                f"(would corrupt replay). Increase MAX_NUM_BATCHED_TOKENS "
                f"so the eager warmup pins a larger high-water mark, or "
                f"disable cudagraph capture for this layer."
            )
        max_M = _max_num_tokens_hint()
        local_n = self._local_n(local_num_experts, global_num_experts)
        cap = max(self._max_em_count(max_M, top_k, local_n), EM)
        st = torch.empty(cap, dtype=torch.int32, device=dev)
        _persistent_sids[key] = st
        print(
            f"[VkernelFusedExperts] persistent sids alloc "
            f"(key={key}, cap={cap} int32, max_M={max_M}, top_k={top_k}, "
            f"local_n={local_n})",
            flush=True,
        )
        return st[:EM]

    def _get_persistent_eids(
        self, dev, EM, top_k, global_num_experts, local_num_experts
    ):
        """Persistent int32 [EM] routing-eids storage (used as [:EM//bs]).

        The valid element count is EM//block_size (<= EM), so sizing the
        storage at max_EM (>= EM for any block_size >= 1) is always safe.
        """
        key = (dev.index,)
        st = _persistent_eids.get(key)
        if st is not None and st.numel() >= EM:
            return st
        if _capture_active():
            cap = 0 if st is None else st.numel()
            raise RuntimeError(
                f"[VkernelFusedExperts] persistent eids (key={key}, "
                f"cap={cap}) is too small for EM={EM} AND a CUDA-graph "
                f"capture session is active — refusing to allocate "
                f"(would corrupt replay). Increase MAX_NUM_BATCHED_TOKENS "
                f"so the eager warmup pins a larger high-water mark, or "
                f"disable cudagraph capture for this layer."
            )
        max_M = _max_num_tokens_hint()
        local_n = self._local_n(local_num_experts, global_num_experts)
        cap = max(self._max_em_count(max_M, top_k, local_n), EM)
        st = torch.empty(cap, dtype=torch.int32, device=dev)
        _persistent_eids[key] = st
        print(
            f"[VkernelFusedExperts] persistent eids alloc "
            f"(key={key}, cap={cap} int32, max_M={max_M}, top_k={top_k}, "
            f"local_n={local_n})",
            flush=True,
        )
        return st

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        """Size workspace13 for act_scratch [EM_max, ispp] bf16.

        The framework allocates workspace tensors with the dtype from
        workspace_dtype() (bf16 by default).  We use workspace13 as the
        preferred act_scratch when it is already large enough (it lives in
        the cudagraph pool, so its address is replay-stable); out_fp32 and
        any too-small-workspace13 fallback are backed by the capture-safe
        persistent buffers (see module docstring), never by per-call
        torch.empty().
        """
        ispp = N // 2
        block_size = 16 if M <= 32 else 64
        # EM_max: padded(M*topk + local_num_experts) — matches the C ABI's
        # moe_align_block_size output size.
        EM_max = ((M * topk + local_num_experts + block_size - 1)
                  // block_size) * block_size
        # workspace13: act_scratch [EM_max, ispp] bf16
        ws13 = (EM_max, ispp)
        # workspace2: not used (we allocate out_fp32 separately)
        ws2 = (0,)
        output = (M, K)
        return (ws13, ws2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta,
        apply_router_weight_on_input: bool,
    ) -> None:
        lib = _get_lib()
        moe_fn = _resolve_moe_fn(lib)
        dev = hidden_states.device

        M, K = hidden_states.shape
        E, N, _ = w1.shape  # E=local experts, N=2*ispp
        ispp = N // 2
        top_k = topk_ids.size(1)

        if global_num_experts == -1:
            global_num_experts = E

        # Block size: 16 for decode (small M), 64 for prefill (large M)
        block_size = 16 if M <= 32 else 64

        # moe_align_block_size (CPU, then copy to GPU). Phase profiling:
        #   moe:apply.cpu_copy  — GPU->CPU sync (waits for the dispatch
        #     all-to-all that produced topk_ids) + the host copy
        #   moe:apply.cpu_align — the pure-Python _moe_align_block_size_cpu
        #   moe:apply.gpu_copy  — sids/eids host->device
        #   moe:apply.launch    — the ctypes C call (issues kernels on stream)
        # record_function is a no-op when no profiler is attached.
        with record_function("moe:apply.cpu_copy"):
            topk_ids_flat = (
                topk_ids.contiguous().view(-1).cpu().numpy().astype(np.int32)
            )
            expert_map_np = (
                expert_map.cpu().numpy().astype(np.int32)
                if expert_map is not None
                else None
            )
        with record_function("moe:apply.cpu_align"):
            sids_np, eids_np, EM = _moe_align_block_size_cpu(
                topk_ids_flat, global_num_experts, block_size, expert_map_np
            )
        with record_function("moe:apply.gpu_copy"):
            # Copy routing metadata INTO persistent buffers (no per-call GPU
            # allocation).  The C kernel reads sids/eids via raw ctypes
            # pointers that PyTorch's allocator cannot track, so re-
            # allocating here would let the allocator recycle the storage
            # mid-capture/replay -> "Memory access fault by GPU node-X".
            d_sids = self._get_persistent_sids(
                dev, EM, top_k, global_num_experts, E
            )
            d_eids = self._get_persistent_eids(
                dev, EM, top_k, global_num_experts, E
            )
            d_sids.copy_(torch.from_numpy(sids_np))
            d_eids[: EM // block_size].copy_(torch.from_numpy(eids_np))

        # Routing weights: if already applied on input, use ones
        if apply_router_weight_on_input:
            topk_w = torch.ones_like(topk_weights).contiguous().view(-1)
        else:
            topk_w = topk_weights.contiguous().view(-1)

        topk_ids_dev = topk_ids.contiguous().view(-1)

        # Activation parameters
        if activation == MoEActivation.SITU:
            act_code = 1  # SiTU
            beta_raw = getattr(self.moe_config, "activation_situ_beta", None)
            beta = float(beta_raw) if beta_raw is not None else 1.0
            linear_beta_raw = getattr(
                self.moe_config, "activation_situ_linear_beta", None
            )
            linear_beta = float(linear_beta_raw) if linear_beta_raw is not None else 25.0
        else:
            act_code = 0  # SwiGLU
            beta = 0.0
            linear_beta = 0.0

        swiglu_limit_raw = getattr(self, "gemm1_clamp_limit", None)
        swiglu_limit = float(swiglu_limit_raw) if swiglu_limit_raw is not None else 4.0

        # Scales and biases (from quant_config properties)
        w13_scale = self.w1_scale
        w2_scale = self.w2_scale
        b13 = self.w1_bias
        b2 = self.w2_bias

        # act_scratch [EM, ispp] bf16.  Prefer the framework-provided
        # workspace13 (it lives in the cudagraph pool, so its address is
        # replay-stable) when it is already large enough; otherwise fall
        # back to our persistent scratch — sized for the max token budget
        # on the eager warmup run and reused forever (NEVER torch.empty()
        # under capture / during replay, which would corrupt the graph).
        _ws13_flat = workspace13.view(-1)
        need_act = EM * ispp
        if _ws13_flat.numel() >= need_act:
            act_scratch = _ws13_flat[:need_act].view(EM, ispp)
        else:
            act_scratch = self._get_persistent_act(
                dev, ispp, EM, top_k, global_num_experts, E
            )
            print(
                f"[VkernelFusedExperts] workspace13 too small "
                f"({_ws13_flat.numel()} < {need_act}); using persistent "
                f"act_scratch (EM={EM}, ispp={ispp}, "
                f"ws13.shape={workspace13.shape})",
                flush=True,
            )

        # out_fp32 [M, K] fp32 — persistent (the original per-call
        # torch.empty() here is the primary capture/replay fault source).
        out_fp32 = self._get_persistent_out(dev, K, M)

        # Launch the MoE kernels on PyTorch's *current* compute stream so
        # their writes to out_fp32 (and act_scratch) are ordered with the
        # fp32->bf16 copy below on that same stream.  This removes the
        # device-wide torch.cuda.synchronize() the vkernels wrapper used
        # as a cross-stream correctness guard: that drain serialised every
        # TP all-to-all and made each MoE layer a per-step barrier (the
        # dominant K3-on-beverin bottleneck, ~78% of wall on PP0).
        stream = torch.cuda.current_stream().cuda_stream

        with record_function("moe:apply.launch"):
            moe_fn(
                ctypes.c_void_p(hidden_states.data_ptr()),
                ctypes.c_void_p(w1.data_ptr()),
                ctypes.c_void_p(w13_scale.data_ptr()),
                ctypes.c_void_p(w2.data_ptr()),
                ctypes.c_void_p(w2_scale.data_ptr()),
                ctypes.c_void_p(topk_ids_dev.data_ptr()),
                ctypes.c_void_p(topk_w.data_ptr()),
                ctypes.c_void_p(act_scratch.data_ptr()),
                ctypes.c_void_p(out_fp32.data_ptr()),
                ctypes.c_int(M),
                ctypes.c_int(K),
                ctypes.c_int(ispp),
                ctypes.c_int(top_k),
                ctypes.c_void_p(d_sids.data_ptr()),
                ctypes.c_void_p(d_eids.data_ptr()),
                ctypes.c_int(EM),
                ctypes.c_float(swiglu_limit),
                ctypes.c_int(act_code),
                ctypes.c_float(beta),
                ctypes.c_float(linear_beta),
                ctypes.c_void_p(b13.data_ptr()) if b13 is not None else None,
                ctypes.c_void_p(b2.data_ptr()) if b2 is not None else None,
                ctypes.c_int(block_size),
                ctypes.c_void_p(stream),
            )

        # Convert fp32 output to bf16 and write to vLLM's output tensor
        output.copy_(out_fp32.view(M, K).to(torch.bfloat16))
