"""sitecustomize: route K3 SiTU MXFP4 MoE to Triton-unfused on gfx942.

Loaded automatically by the CPython interpreter at startup (this file lives at
$K3/home/pylib/sitecustomize.py, which k3_patch.py installs and engine.sh puts
first on PYTHONPATH), so it runs BEFORE vllm.model_executor builds any layer.

Background (k3-eng11 -> job 580844 lesson)
------------------------------------------
The AITER FlyDSL a16w4 MoE kernel (used by the AITER_MXFP4_BF16 backend)
requires 82944 bytes of LDS on gfx942, but the hardware limit is 65536
(64 KB).  The kernel was designed for gfx950 (128 KB LDS) and cannot fit
on MI300A regardless of tile size, suffix, or wave-per-EU parameters.
This was exhaustively verified across jobs 583297-583962.

Solution (job 584xxx)
---------------------
Instead of trying to make FlyDSL fit on gfx942, we route the MoE backend
selection to ``TRITON_UNFUSED`` — the ``UnfusedOAITritonExperts`` class
from ``gpt_oss_triton_kernels_moe.py``.  This class uses ``matmul_ogs``
(plain Triton kernels, NOT FlyDSL) for the GEMMs and applies the activation
separately (unfused), so there is no 82 KB LDS requirement.

The class already supports ``(kMxfp4Static, None)`` quantization and all
routing methods on gfx942; the only blocker was ``_supports_activation``
not including ``MoEActivation.SITU``.  We patch that, and the unfused
``activation()`` method falls through to ``super().activation()`` which
calls ``apply_moe_activation(SITU, ...)`` which uses the compiled
``torch.ops._C.situ_and_mul(output, input, beta, linear_beta)`` kernel.

Patches applied here:
  (1) Keep ``on_gfx950 = True`` for MLA and other AITER ops that need it.
  (2) Patch ``_use_k3_situ_aiter`` to return False so the MoE selector goes
      through ``select_deepseek_v4_mxfp4_moe_backend`` instead of the
      direct AITER branch.
  (3) Patch ``_get_priority_backends`` on ROCm to include TRITON_UNFUSED
      after AITER_MXFP4_BF16 (Kimi-K3 uses ``DeepSeekV3`` routing, which
      does NOT get the special ``[AITER_MXFP4_BF16, TRITON_UNFUSED]`` list
      that ``DeepseekV4`` gets).
  (4) Patch ``UnfusedOAITritonExperts._supports_activation`` to include SITU.

The AITER MoE backend's ``_supports_activation`` deliberately does NOT
include SITU (see ``rocm_aiter_moe.py:484``), so the oracle naturally
falls through from AITER to TRITON_UNFUSED.
"""
import os

# Belt-and-suspenders: engine.sh sets these too, but setdefault keeps the
# AITER MoE backend enabled even if sitecustomize is imported before engine.sh
# exports them (e.g. a helper subprocess).
os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_MOE", "1")

# (1) on_gfx950() -> True.  We STILL need this for the MLA backend
# (`_fp8_mla_prefill_supported()` in `rocm_aiter_mla.py:71`) and for AITER
# linear / FP4 BMM ops.  The MoE backend selection is handled separately
# by patch (2) below so that `on_gfx950=True` no longer forces FlyDSL.
try:
    from vllm.platforms import rocm as _rocm

    _rocm.on_gfx950 = lambda: True
except Exception:
    pass

# (2) Patch _use_k3_situ_aiter to return False.
# Without this, `Mxfp4MoEMethod.__init__` (mxfp4.py:516) would see
# `on_gfx950()=True` + `is_fused_moe_enabled()=True` + SITU activation and
# directly select the AITER FlyDSL kernel — which exceeds the 64 KB LDS
# limit on gfx942 and crashes at JIT compile time.
# With this patch, the selector calls `select_deepseek_v4_mxfp4_moe_backend`
# which tries backends in priority order and falls through to TRITON_UNFUSED.
try:
    from vllm.model_executor.layers.quantization import mxfp4 as _mxfp4_mod

    _mxfp4_mod._use_k3_situ_aiter = lambda moe: False
except Exception:
    pass

# (3) Patch _get_priority_backends on ROCm to include VKERNELS_MXFP4_BF16.
# Kimi-K3 uses `RoutingMethodType.DeepSeekV3` (not DeepseekV4), so the
# special `if current_platform.is_rocm() and config.routing_method ==
# DeepseekV4` branch in `select_deepseek_v4_mxfp4_moe_backend` (which
# already includes TRITON_UNFUSED) is NOT taken.  Instead the else branch
# calls `_get_priority_backends()` which on ROCm returns only
# `[AITER_MXFP4_BF16]`.  We add VKERNELS_MXFP4_BF16 (a new backend we
# add to the Mxfp4MoeBackend enum below) so the oracle can fall through
# from AITER (which fails `_supports_activation(SITU)`) to the vkernels
# C ABI kernel.  VKERNELS_MXFP4_BF16 passes weights through WITHOUT
# swizzling (unlike TRITON_UNFUSED which is in TRITON_BACKENDS and
# calls _swizzle_mxfp4 + wraps scales in PrecisionConfig).
try:
    import torch
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as _oracle_mod
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.platforms import current_platform as _cp

    # Add VKERNELS_MXFP4_BF16 to the Mxfp4MoeBackend enum as a
    # proper member (not a plain string).  The oracle calls
    # backend.value / backend.name in _make_log_backend and
    # _make_log_unsupported, so a bare string would crash with
    # AttributeError: 'str' object has no attribute 'value'.
    # Python's Enum functional API (Enum(name, value)) creates a
    # new enum *class*, not a new member, so we construct the
    # member manually via object.__new__ and register it in the
    # internal _member_map_ / _value2member_map_.
    if not hasattr(_oracle_mod.Mxfp4MoeBackend, "VKERNELS_MXFP4_BF16"):
        _vk = object.__new__(_oracle_mod.Mxfp4MoeBackend)
        _vk._name_ = "VKERNELS_MXFP4_BF16"
        _vk._value_ = "VKERNELS_MXFP4_BF16"
        # EnumType.__setattr__ blocks reassignment of existing members
        # but allows setting NEW attributes (falls through to
        # type.__setattr__).  This adds _vk to __dict__ so that
        # Mxfp4MoeBackend.VKERNELS_MXFP4_BF16 attribute access works.
        _oracle_mod.Mxfp4MoeBackend.VKERNELS_MXFP4_BF16 = _vk
        _oracle_mod.Mxfp4MoeBackend._member_map_[
            "VKERNELS_MXFP4_BF16"
        ] = _vk
        _oracle_mod.Mxfp4MoeBackend._value2member_map_[
            "VKERNELS_MXFP4_BF16"
        ] = _vk

    # Patch convert_gpt_oss_weight_to_mxfp4_moe_kernel_format: for
    # VKERNELS_MXFP4_BF16, pass weights and scales through unchanged
    # (only convert biases to float32).  This avoids the _swizzle_mxfp4
    # + PrecisionConfig wrapping that TRITON_BACKENDS does.
    _orig_convert = _oracle_mod.convert_gpt_oss_weight_to_mxfp4_moe_kernel_format

    def _patched_convert(
        mxfp4_backend,
        layer,
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w13_bias=None,
        w2_bias=None,
        _cache_permute_indices=None,
    ):
        if mxfp4_backend == _oracle_mod.Mxfp4MoeBackend.VKERNELS_MXFP4_BF16:
            if w13_bias is not None:
                w13_bias = w13_bias.data.to(torch.float32)
            if w2_bias is not None:
                w2_bias = w2_bias.data.to(torch.float32)
            return (
                w13_weight.data,
                w2_weight.data,
                w13_weight_scale.data,
                w2_weight_scale.data,
                w13_bias,
                w2_bias,
            )
        return _orig_convert(
            mxfp4_backend,
            layer,
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
            _cache_permute_indices=_cache_permute_indices,
        )

    _oracle_mod.convert_gpt_oss_weight_to_mxfp4_moe_kernel_format = (
        _patched_convert
    )

    # Also patch convert_weight_to_mxfp4_moe_kernel_format (the non-GPT-OSS
    # path used by K3 / DeepSeekV3-routing models).  Same pass-through for
    # VKERNELS_MXFP4_BF16: return weights and scales unchanged, convert
    # biases to float32.
    _orig_convert2 = (
        _oracle_mod.convert_weight_to_mxfp4_moe_kernel_format
    )

    def _patched_convert2(
        mxfp4_backend,
        layer,
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w13_bias=None,
        w2_bias=None,
        _cache_permute_indices=None,
    ):
        if mxfp4_backend == _oracle_mod.Mxfp4MoeBackend.VKERNELS_MXFP4_BF16:
            if w13_bias is not None:
                w13_bias = w13_bias.data.to(torch.float32)
            if w2_bias is not None:
                w2_bias = w2_bias.data.to(torch.float32)
            return (
                w13_weight.data,
                w2_weight.data,
                w13_weight_scale.data,
                w2_weight_scale.data,
                w13_bias,
                w2_bias,
            )
        return _orig_convert2(
            mxfp4_backend,
            layer,
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
            _cache_permute_indices=_cache_permute_indices,
        )

    _oracle_mod.convert_weight_to_mxfp4_moe_kernel_format = (
        _patched_convert2
    )

    # CRITICAL: quantization/mxfp4.py imports these functions at module
    # load time (from ... import convert_weight_to_mxfp4_moe_kernel_format).
    # Patching the oracle module alone is NOT enough — the quantization
    # module has its own reference to the ORIGINAL function.  We must
    # patch both modules.
    from vllm.model_executor.layers.quantization import mxfp4 as _qm_mod
    _qm_mod.convert_gpt_oss_weight_to_mxfp4_moe_kernel_format = (
        _patched_convert
    )
    _qm_mod.convert_weight_to_mxfp4_moe_kernel_format = (
        _patched_convert2
    )

    if _cp.is_rocm():
        _orig_gpb = _oracle_mod._get_priority_backends

        def _patched_gpb():
            return [
                _oracle_mod.Mxfp4MoeBackend.AITER_MXFP4_BF16,
                _oracle_mod.Mxfp4MoeBackend.VKERNELS_MXFP4_BF16,
                _oracle_mod.Mxfp4MoeBackend.TRITON_UNFUSED,
            ]

        _oracle_mod._get_priority_backends = _patched_gpb
except Exception as _e:
    import traceback
    print(f"[sitecustomize] WARNING: VKERNELS_MXFP4_BF16 patch failed: {_e}",
          flush=True)
    traceback.print_exc()

# (4) Patch UnfusedOAITritonExperts._supports_activation to include SITU.
# The class already supports `(kMxfp4Static, None)` quantization, all
# routing methods, and `_supports_current_device()` on gfx942 (via
# `on_gfx9()`).  The `activation()` method falls through to
# `super().activation()` for unknown activations, which calls
# `apply_moe_activation(SITU, ...)` which uses the compiled
# `torch.ops._C.situ_and_mul(output, input, beta, linear_beta)` kernel.
try:
    from vllm.model_executor.layers.fused_moe.config import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
        UnfusedOAITritonExperts,
    )

    _orig_unfused_act = UnfusedOAITritonExperts._supports_activation

    @staticmethod
    def _patched_unfused_act(activation):
        if activation == MoEActivation.SITU:
            return True
        return _orig_unfused_act(activation)

    UnfusedOAITritonExperts._supports_activation = _patched_unfused_act
except Exception:
    pass

# (5) Register VkernelFusedExperts — vkernels HIP C ABI MoE backend.
# Instead of using Triton kernels for the MoE GEMMs (which work but are
# slow on gfx942), we route the new VKERNELS_MXFP4_BF16 backend to
# VkernelFusedExperts which calls vk_hip_fused_moe_mxfp4 via ctypes. The
# vkernels kernel does the full MoE computation (gate-up GEMM ->
# activation -> down GEMM -> routing weight -> top-k sum) on gfx942 HIP,
# validated in job 596227.
#
# VKERNELS_MXFP4_BF16 is NOT in TRITON_BACKENDS, so weights are passed
# through WITHOUT swizzling (raw MXFP4 uint8 + ue8m0 scales), matching
# vLLM's Mxfp4MoEMethod.create_weights() output exactly:
#   w13_weight:      [E, 2*ispp, hidden/2]  uint8 (packed MXFP4)
#   w13_weight_scale: [E, 2*ispp, hidden/32] uint8 (ue8m0)
#   w2_weight:       [E, hidden, ispp/2]    uint8 (packed MXFP4)
#   w2_weight_scale: [E, hidden, ispp/32]   uint8 (ue8m0)
#   w13_bias/w2_bias: [E, ...]              float32 (if has_bias)
#
# If libvkernels_hip.so is not found, falls back to the original
# UnfusedOAITritonExperts (Triton kernels) silently.
try:
    import sys as _sys_mod
    _k3_pylib = os.path.join(os.environ.get("K3", ""), "home/pylib")
    if _k3_pylib not in _sys_mod.path:
        _sys_mod.path.insert(0, _k3_pylib)
    from vkernels_experts import VkernelFusedExperts, _find_libvkernels_hip

    if _find_libvkernels_hip() is not None:
        _orig_btoc = _oracle_mod.backend_to_kernel_cls

        def _patched_btoc(backend):
            if backend == _oracle_mod.Mxfp4MoeBackend.VKERNELS_MXFP4_BF16:
                return [VkernelFusedExperts]
            return _orig_btoc(backend)

        _oracle_mod.backend_to_kernel_cls = _patched_btoc
        print("[sitecustomize] VkernelFusedExperts registered "
              "(VKERNELS_MXFP4_BF16)", flush=True)
    else:
        print("[sitecustomize] libvkernels_hip.so not found, "
              "using Triton fallback", flush=True)
except Exception as _vk_err:
    print(f"[sitecustomize] VkernelFusedExperts registration skipped: "
          f"{_vk_err}", flush=True)

# (6) Register VkernelMLA + VkernelKDA -- the two remaining critical-path
# attention layers of K3 wired to their validated vkernels HIP kernels on
# gfx942 (issue #42).  Like VkernelFusedExperts above, both load
# libvkernels_hip.so via ctypes.  They are OPT-IN (VKERNELS_MLA=1 /
# VKERNELS_KDA=1) and default OFF: importing this file and calling
# register_vkernels_attn() does not change serving until the flags are set
# and the device kernels have been validated against the CPU oracle on the
# cluster (see vkernels_attn.py -- VKERNELS_MLA_FORCE / VKERNELS_KDA + the
# VKERNELS_*_VALIDATE cross-checks gate < 1e-2).  When enabled, VkernelMLA
# is selected ahead of TRITON_MLA and the KDA delta-rule goes through
# vk_hip_kda_delta_rule_fwd, so K3_DISABLE_KDA (see below) is no longer
# needed and the delta-rule layer is no longer silently dropped.
try:
    import sys as _sys_attn
    _k3_pylib_attn = os.path.join(os.environ.get("K3", ""), "home/pylib")
    if _k3_pylib_attn not in _sys_attn.path:
        _sys_attn.path.insert(0, _k3_pylib_attn)
    from vkernels_attn import register_vkernels_attn  # noqa: E402
    register_vkernels_attn()
except Exception as _vk_attn_err:
    print(f"[sitecustomize] VkernelMLA/VkernelKDA registration skipped: "
          f"{_vk_attn_err}", flush=True)

# --- K3 JIT stagger: prevent OOM from simultaneous worker startup --
# On gfx942 with 4 workers/node and ~501 GB system RAM, all 4 workers
# simultaneously constructing the Kimi-K3 model + JIT-compiling Triton
# matmul_ogs exceeds the node's system RAM (jobs 584773/584789 OOM).
# Stagger worker startup by local_rank: rank 0 starts immediately and
# populates the shared TRITON_CACHE_DIR; ranks 1-3 wait in sequence so
# they load compiled kernels from cache instead of compiling them.
# After EACH worker's load_model, gc.collect() + cuda.empty_cache()
# release temporary CPU buffers (safetensors deserialization, weight
# remap) and GPU caching-allocator memory back to the OS.  On UMA this
# is critical: GPU memory IS system RAM, so the ~20 GB/worker temp
# buffer during the 4th worker's load was the tipping point in job
# 585639 (4×138 GB > 501 GB node RAM).  Freeing it immediately after
# each worker finishes load_model keeps the peak under the node limit.
try:
    import gc as _gc_mod
    import time as _time_mod
    import torch as _torch_mod
    from vllm.v1.worker.gpu_worker import Worker as _K3Worker
    _orig_load_model = _K3Worker.load_model

    def _k3_staggered_load_model(self, *, load_dummy_weights=False):
        _delay = self.local_rank * int(
            os.environ.get("K3_JIT_STAGGER_DELAY", "60")
        )
        if _delay > 0 and os.environ.get("K3_JIT_STAGGER", "1") != "0":
            print(
                f"[K3_STAGGER] local_rank={self.local_rank} rank={self.rank} "
                f"sleeping {_delay}s before load_model",
                flush=True,
            )
            _time_mod.sleep(_delay)
        _orig_load_model(self, load_dummy_weights=load_dummy_weights)
        # Release temporary CPU buffers (safetensors deserialization,
        # weight remap) and GPU caching-allocator memory back to the OS.
        # On UMA this returns memory to the OS immediately, reducing the
        # peak for the next worker's load (job 585639: ~20 GB/worker temp).
        try:
            _gc_mod.collect()
            if hasattr(self, "device"):
                _torch_mod.cuda.synchronize(self.device)
            _torch_mod.cuda.empty_cache()
            _gc_mod.collect()
        except Exception as _e:
            print(f"[K3_STAGGER] gc/empty_cache warning: {_e}", flush=True)
        print(
            f"[K3_STAGGER] local_rank={self.local_rank} rank={self.rank} "
            f"load_model done (gc+empty_cache)",
            flush=True,
        )
    _K3Worker.load_model = _k3_staggered_load_model
    print("[K3] JIT stagger patch applied (Worker.load_model)", flush=True)
except Exception as e:
    print(f"[K3] JIT stagger patch FAILED: {e}", flush=True)

# ---------------------------------------------------------------------
# (3) Multi-node PP follower fix (job 580876 lesson)
# ---------------------------------------------------------------------
# This image's multi-node MP design is leader-driven: the leader node's
# MultiprocExecutor (node_rank_within_dp == 0) owns the world-wide broadcast
# MessageQueue (world_size=16); workers on follower nodes join it via
# get_inner_dp_world_group().create_mq_broadcaster(external_writer_handle=...),
# and model outputs flow back to the driver via
# create_single_reader_mq_broadcasters(reader_rank_in_group=0). ALL
# scheduler/execute/profile fan-out originates from the leader; follower
# EngineCores are pure scaffolding. Correspondingly, MultiprocExecutor on a
# follower deliberately has rpc_broadcast_mq=None and its collective_rpc()
# asserts: "collective_rpc should not be called on follower node".
#
# The bug: EngineCore.__init__ unconditionally calls _initialize_kv_caches()
# -> executor.get_kv_cache_specs() -> collective_rpc(...) -> AssertionError on
# every follower node, ~2h13m in, right after the model finishes loading.
# The follower's workers do NOT need that collective: they receive the real
# KVCacheConfig from the leader's initialize_from_config() fan-out. The
# follower only needs a structurally valid KVCacheConfig to finish building
# its (never-scheduling) Scheduler.
#
# Three follower-killing collectives patched here (job 581184 found #3):
#   a) _initialize_kv_caches  -> return KVCacheConfig(num_blocks=1, [], [])
#      (num_blocks must be > 0: Scheduler.__init__ asserts it; empty groups
#      short-circuit resolve_kv_cache_block_sizes to cache_config.block_size,
#      so we also make sure block_size is an int). Workers get the real config
#      from the leader's initialize_from_config() fan-out.
#   b) get_supported_tasks    -> ("generate",)
#      (the API server asks the engine at startup; the underlying
#      Executor.supported_tasks cached_property is another collective_rpc.)
#   c) reset_mm_cache         -> the API server also calls reset_mm_cache()
#      at startup (api_server.py:188); EngineCore.reset_mm_cache() ->
#      model_executor.reset_mm_cache() -> collective_rpc("reset_mm_cache")
#      -> assert on followers. Rather than play whack-a-mole with each
#      startup utility, (c) patches the ROOT CAUSE: MultiprocExecutor.
#      collective_rpc() short-circuits to a no-op when rpc_broadcast_mq is
#      None -- which is true ONLY on follower nodes, by construction
#      (multiproc_executor.py:135 creates it for node_rank_within_dp == 0).
#      This covers reset_mm_cache and any future startup collective a
#      follower EngineCore is asked to run (sleep/wake_up/profile/etc are
#      only ever client-driven and no client talks to a follower API server).
# Guarded on nnodes_within_dp > 1 and node_rank_within_dp != 0, so single-node
# runs and the leader are untouched. Set K3_DISABLE_FOLLOWER_KV_SKIP=1 to
# disable (e.g. when testing a fixed image).
def _k3_is_follower(parallel_config) -> bool:
    try:
        nnodes = getattr(parallel_config, "nnodes_within_dp", 1) or 1
        return nnodes > 1 and parallel_config.node_rank_within_dp != 0
    except Exception:
        return False


if os.environ.get("K3_DISABLE_FOLLOWER_KV_SKIP", "0") != "1":
    try:
        from vllm.logger import init_logger as _init_logger
        from vllm.v1.engine import core as _v1core
        from vllm.v1.kv_cache_interface import KVCacheConfig as _KVCacheConfig

        _log = _init_logger("k3.sitecustomize")

        # (a) skip _initialize_kv_caches on followers ---------------------
        _orig_init_kv_caches = _v1core.EngineCore._initialize_kv_caches

        def _patched_init_kv_caches(self, vllm_config):
            if _k3_is_follower(vllm_config.parallel_config):
                rank = vllm_config.parallel_config.node_rank_within_dp
                _log.info(
                    "K3 multi-node: node_rank_within_dp=%d is a follower; "
                    "skipping leader-only KV-cache init (collective_rpc would "
                    "assert). Local workers receive the real KVCacheConfig "
                    "from the leader's broadcast.",
                    rank,
                )
                cc = vllm_config.cache_config
                if getattr(cc, "block_size", None) is None:
                    cc.block_size = 16
                if getattr(cc, "num_gpu_blocks", None) is None:
                    cc.num_gpu_blocks = 1
                return _KVCacheConfig(
                    num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[]
                )
            return _orig_init_kv_caches(self, vllm_config)

        _v1core.EngineCore._initialize_kv_caches = _patched_init_kv_caches

        # (b) stub get_supported_tasks on followers -----------------------
        _orig_get_supported_tasks = _v1core.EngineCore.get_supported_tasks

        def _patched_get_supported_tasks(self):
            if _k3_is_follower(self.vllm_config.parallel_config):
                return ("generate",)
            return _orig_get_supported_tasks(self)

        _v1core.EngineCore.get_supported_tasks = _patched_get_supported_tasks

        # (c) root-cause: MultiprocExecutor.collective_rpc no-op when the
        #     follower has no broadcast MQ (covers reset_mm_cache + any other
        #     startup utility the API server issues). Returns per-method
        #     sensible defaults so the awaiting client on a follower is happy.
        from vllm.v1.executor.multiproc_executor import (
            MultiprocExecutor as _MPX,
        )

        _orig_collective_rpc = _MPX.collective_rpc

        def _patched_collective_rpc(
            self, method, timeout=None, args=(), kwargs=None,
            non_block=False, unique_reply_rank=None,
            kv_output_aggregator=None,
        ):
            if getattr(self, "rpc_broadcast_mq", None) is None:
                name = method if isinstance(method, str) else getattr(
                    method, "__name__", "<callable>")
                _log.info(
                    "K3 multi-node: follower no-op collective_rpc(%r)",
                    name,
                )
                if name == "get_supported_tasks":
                    return [("generate",)]
                if name in ("add_lora", "remove_lora", "pin_lora"):
                    return [True]
                if name == "list_loras":
                    return [[]]
                # reset_mm_cache, reset_encoder_cache, profile, sleep,
                # wake_up, update_max_model_len, and unknowns all return None.
                return None
            return _orig_collective_rpc(
                self, method, timeout, args, kwargs, non_block,
                unique_reply_rank, kv_output_aggregator,
            )

        _MPX.collective_rpc = _patched_collective_rpc

        print(
            "K3_SITECUSTOMIZE: multi-node follower KV-init/supported-tasks/"
            "collective_rpc no-op installed (K3_DISABLE_FOLLOWER_KV_SKIP=1 "
            "to disable)",
            flush=True,
        )
    except Exception:
        # vllm.v1.engine.core may be unavailable in helper interpreters; the
        # patch only matters in engine-core processes, which import the full
        # stack anyway and will succeed.
        pass

# ---------------------------------------------------------------------
# (4) AITER module_moe_asm fallback for gfx942 (job 581700 lesson)
# ---------------------------------------------------------------------
# The pre-compiled module_moe_asm.so in this image targets gfx950 ONLY
# (amdgcn-amd-amdhsa--gfx950, no gfx942 code). When the AITER JIT
# framework tries to import it (for topk_softmax, moe_sum,
# moe_align_block_size) or JIT-compile from source (--offload-arch=native),
# BOTH paths fail on MI300A:
#   ModuleNotFoundError: No module named 'aiter.jit.module_moe_asm'
# This crashes the profiling forward pass (determine_available_memory)
# inside _initialize_kv_caches.
#
# These functions are MoE ORCHESTRATION ops (expert routing, token-block
# alignment, output reduction) -- NOT the MXFP4 GEMM itself, which lives in
# module_gemm_a16w16_asm.so (HAS gfx942 code via gfx9f). We replace them
# with vLLM's own native ops (torch.ops._moe_C.* / Triton kernels) that
# work on gfx942. The AITER MXFP4 GEMM (the correctness-critical part we
# want to test) stays on its own kernel.
#
# Additionally, aiter/ops/topk.py has biased_grouped_topk_hip, grouped_topk,
# and moe_fused_gate (also module_moe_asm). These already have *_torch
# fallbacks in the same file. Set K3_DISABLE_MOE_ASM_FALLBACK=1 to disable.
if os.environ.get("K3_DISABLE_MOE_ASM_FALLBACK", "0") != "1":
    try:
        import torch
        import vllm._custom_ops as _vllm_ops
        from aiter.ops import moe_op as _moe_op
        from aiter.ops import topk as _aiter_topk
        import aiter as _aiter_mod

        # (a) topk_softmax -------------------------------------------------
        # AITER: (topk_weights, topk_indices, token_expert_indices,
        #        gating_output, need_renorm, num_shared_experts=0,
        #        shared_expert_scoring_func="")  -> writes, returns None
        # vLLM:  (topk_weights, topk_ids, token_expert_indices,
        #        gating_output, renormalize=False,
        #        e_score_correction_bias=None, is_padding=None) -> writes
        def _py_topk_softmax(
            topk_weights, topk_indices, token_expert_indices,
            gating_output, need_renorm, num_shared_experts=0,
            shared_expert_scoring_func="",
        ):
            g = gating_output
            if num_shared_experts > 0:
                g = gating_output[..., :-num_shared_experts]
            _vllm_ops.topk_softmax(
                topk_weights, topk_indices, token_expert_indices,
                g, need_renorm, e_score_correction_bias=None,
                is_padding=None,
            )

        _moe_op.topk_softmax = _py_topk_softmax
        if hasattr(_aiter_mod, "topk_softmax"):
            _aiter_mod.topk_softmax = _py_topk_softmax

        # (a2) topk_softmax_asm (different JIT module, same fallback) -----
        # AITER: (topk_weights, topk_indices, token_expert_indices,
        #        gating_output, need_renorm)  -> writes, returns None
        # NOTE: in fused_moe.py:3516 the ASM path allocates topk_weights /
        # topk_ids padded to (M+3)//4*4; slice to :M before passing to the
        # vLLM kernel (which uses gating_output.shape[0] = M as token count).
        def _py_topk_softmax_asm(
            topk_weights, topk_indices, token_expert_indices,
            gating_output, need_renorm,
        ):
            M = gating_output.shape[0]
            _vllm_ops.topk_softmax(
                topk_weights[:M], topk_indices[:M], token_expert_indices,
                gating_output, need_renorm,
            )

        _moe_op.topk_softmax_asm = _py_topk_softmax_asm
        if hasattr(_aiter_mod, "topk_softmax_asm"):
            _aiter_mod.topk_softmax_asm = _py_topk_softmax_asm

        # (b) moe_sum ------------------------------------------------------
        # AITER: (input, output)  | vLLM: (input, output, topk_ids, expert_map)
        def _py_moe_sum(input, output, topk_ids=None, expert_map=None):
            _vllm_ops.moe_sum(input, output, topk_ids, expert_map)

        _moe_op.moe_sum = _py_moe_sum
        if hasattr(_aiter_mod, "moe_sum"):
            _aiter_mod.moe_sum = _py_moe_sum

        # (c) moe_align_block_size ----------------------------------------
        # AITER: (topk_ids, num_experts, block_size, sorted_token_ids,
        #        experts_ids, token_nums, num_tokens_post_pad)
        # vLLM:  (topk_ids, num_experts, block_size, sorted_token_ids,
        #        experts_ids, num_tokens_post_pad, expert_map=None)
        # token_nums (per-expert count) is extra in AITER; compute it.
        def _py_moe_align_block_size(
            topk_ids, num_experts, block_size, sorted_token_ids,
            experts_ids, token_nums, num_tokens_post_pad,
        ):
            _vllm_ops.moe_align_block_size(
                topk_ids, num_experts, block_size, sorted_token_ids,
                experts_ids, num_tokens_post_pad, expert_map=None,
            )
            if token_nums is not None:
                token_nums.copy_(
                    torch.bincount(
                        topk_ids.view(-1).to(torch.int64),
                        minlength=num_experts,
                    ).to(token_nums.dtype)
                )

        _moe_op.moe_align_block_size = _py_moe_align_block_size
        if hasattr(_aiter_mod, "moe_align_block_size"):
            _aiter_mod.moe_align_block_size = _py_moe_align_block_size

        # (d) topk.py: wrappers around existing *_torch fallbacks -----
        # AITER signatures differ from *_torch: AITER takes pre-allocated
        # (topk_weights, topk_ids) tensors and writes IN-PLACE returning
        # None; *_torch takes int topk and returns (tw, ti) new tensors.
        # We write in-place to satisfy callers that ignore the return.
        #
        # biased_grouped_topk_hip:
        #   AITER: (gating_output, correction_bias, topk_weights, topk_ids,
        #          num_expert_group, topk_grp, need_renorm,
        #          routed_scaling_factor=1.0) -> None  (in-place)
        #   torch: (gating_output, correction_bias, topk, renormalize,
        #          num_expert_group=0, topk_group=0, return_score=False)
        #          -> (topk_weights, topk_ids)
        def _py_biased_grouped_topk_hip(
            gating_output, correction_bias, topk_weights, topk_ids,
            num_expert_group, topk_grp, need_renorm,
            routed_scaling_factor=1.0,
        ):
            topk = topk_ids.shape[1]
            tw, ti = _aiter_topk.biased_grouped_topk_torch(
                gating_output, correction_bias, topk,
                renormalize=need_renorm,
                num_expert_group=num_expert_group,
                topk_group=topk_grp,
                return_score=False,
            )
            topk_weights.copy_(tw * routed_scaling_factor)
            topk_ids.copy_(ti)

        _aiter_topk.biased_grouped_topk_hip = _py_biased_grouped_topk_hip
        if hasattr(_aiter_mod, "biased_grouped_topk_hip"):
            _aiter_mod.biased_grouped_topk_hip = _py_biased_grouped_topk_hip

        # grouped_topk:
        #   AITER: (gating_output, topk_weights, topk_ids, num_expert_group,
        #          topk_group, need_renorm, is_softmax=True,
        #          routed_scaling_factor=1.0) -> None  (in-place)
        #   torch: (gating_output, topk, renormalize, num_expert_group=0,
        #          topk_group=0, scoring_func="softmax")
        #          -> (topk_weights, topk_ids)
        def _py_grouped_topk(
            gating_output, topk_weights, topk_ids,
            num_expert_group, topk_group, need_renorm,
            is_softmax=True, routed_scaling_factor=1.0,
        ):
            topk = topk_ids.shape[1]
            scoring_func = "softmax" if is_softmax else "sigmoid"
            tw, ti = _aiter_topk.grouped_topk_torch(
                gating_output, topk,
                renormalize=need_renorm,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
            )
            topk_weights.copy_(tw * routed_scaling_factor)
            topk_ids.copy_(ti)

        _aiter_topk.grouped_topk = _py_grouped_topk
        if hasattr(_aiter_mod, "grouped_topk"):
            _aiter_mod.grouped_topk = _py_grouped_topk

        # (e) moe_fused_gate: PyTorch fallback (no existing *_torch) -------
        # AITER: (input, bias, topk_weights, topk_ids, num_expert_group,
        #        topk_group, topk, n_share_experts_fusion,
        #        routed_scaling_factor=1.0) -> (topk_weights, topk_ids)
        # Writes in-place AND returns, since the biased_grouped_topk
        # dispatcher (topk.py:160) does `return moe_fused_gate(...)` while
        # the outer caller (_rocm_aiter_biased_grouped_topk_impl) ignores
        # the return value — both behaviours must be satisfied.
        def _py_moe_fused_gate(
            input, bias, topk_weights, topk_ids, num_expert_group,
            topk_group, topk, n_share_experts_fusion,
            routed_scaling_factor=1.0,
        ):
            tw, ti = _aiter_topk.biased_grouped_topk_torch(
                input, bias, topk, renormalize=True,
                num_expert_group=num_expert_group,
                topk_group=topk_group, return_score=False,
            )
            tw_scaled = tw * routed_scaling_factor
            topk_weights.copy_(tw_scaled)
            topk_ids.copy_(ti)
            return tw_scaled, ti

        _aiter_topk.moe_fused_gate = _py_moe_fused_gate
        if hasattr(_aiter_mod, "moe_fused_gate"):
            _aiter_mod.moe_fused_gate = _py_moe_fused_gate

        print(
            "K3_SITECUSTOMIZE: AITER module_moe_asm PyTorch fallback "
            "installed (topk_softmax, topk_softmax_asm, moe_sum, "
            "moe_align_block_size, grouped_topk, biased_grouped_topk, "
            "moe_fused_gate) (K3_DISABLE_MOE_ASM_FALLBACK=1 to disable)",
            flush=True,
        )
    except Exception:
        # aiter may be unavailable in helper interpreters; the fallback
        # only matters in worker processes, which import the full stack.
        pass

# ── KDA on gfx942: vkernels delta-rule replaces the Triton path ──────
# With issue #42 the faulting chunked KDA kernels are replaced on gfx942 by
# the validated vk_hip_kda_delta_rule_fwd (VKERNELS_KDA=1, routed in
# vkernels_attn.py above).  KDA then stays ACTIVE and K3_DISABLE_KDA is no
# longer needed in the serving recipe (removed by default; see
# serve_kimi_k3_otela_beverin.sbatch).  K3_DISABLE_KDA is kept below as an
# explicit override (default off) for the --load-format dummy all-MLA probe
# and as an emergency kill-switch.
# The KDA (Kimi Delta Attention) Triton kernels are validated on gfx950 only.
# On gfx942 (MI300A), they cause GPU memory access faults during execution
# (job 586165: 8 GPU faults across 4 GPUs on PP0, right after JIT compilation
# of kda_gate_chunk_cumsum_vector_kernel, chunk_kda_fwd_kernel_intra_sub_chunk,
# chunk_kda_fwd_kernel_inter_solve_fused, chunk_gated_delta_rule_fwd_kernel,
# chunk_gla_fwd_kernel_o, layer_norm_gated_fwd_kernel, pack_bitmatrix).
# Setting K3_DISABLE_KDA=1 patches KimiLinearConfig.is_kda_layer() to return
# False, so ALL layers use KimiMLAAttention (TRITON_MLA backend, verified
# working on gfx942 in job 586165).  This is ONLY valid with --load-format
# dummy (the model architecture is incorrect — ~2/3 of layers should be KDA,
# not MLA).  State management (MambaStateDtypeCalculator etc.) is still set
# up but harmless when no KDA layers exist.
# Precedence (issue #42): VKERNELS_KDA=1 WINS over K3_DISABLE_KDA=1 so the
# operator opts into the validated HIP path with a single variable -- they
# do NOT also have to unset K3_DISABLE_KDA.  K3_DISABLE_KDA=1 (the pre-#42
# all-MLA workaround) is the safe default in serve_kimi_k3_otela_beverin.
# sbatch until VKERNELS_KDA=1 has passed the on-cluster validation (device-vs-
# CPU max_rel<0.01 + the 6/6 factual probes + rocprof + latency, see
# BENCHMARK.md).  Once validated, set K3_DISABLE_KDA=0 (or unset it) to
# complete AC4: the delta-rule layer is then served by vk_hip_kda_delta_
# rule_fwd and no longer silently dropped.
if os.environ.get("VKERNELS_KDA", "0") == "1":
    print(
        "[K3] KDA active via vk_hip_kda_delta_rule_fwd "
        "(VKERNELS_KDA=1, issue #42); supersedes any K3_DISABLE_KDA=1 "
        "fallback -- the validated HIP kernel replaces the faulting "
        "Triton path, so the delta-rule layer is served, not dropped.",
        flush=True,
    )
elif os.environ.get("K3_DISABLE_KDA", "0") == "1":
    try:
        from vllm.transformers_utils.configs.kimi_linear import (
            KimiLinearConfig as _KLC,
        )

        def _patched_is_kda(self, layer_idx: int):
            return False

        _KLC.is_kda_layer = _patched_is_kda
        print(
            "[K3] KDA disabled (is_kda_layer -> False); all layers use MLA "
            "(K3_DISABLE_KDA=1) -- safe fallback until VKERNELS_KDA=1 is "
            "validated, or for the --load-format dummy gen-probe.",
            flush=True,
        )
    except Exception as e:
        print(f"[K3] KDA disable patch FAILED: {e}", flush=True)
else:
    print(
        "[K3] KDA routing default: neither VKERNELS_KDA=1 nor "
        "K3_DISABLE_KDA=1 -- the Triton KDA path will fault on gfx942. "
        "Set VKERNELS_KDA=1 (issue #42, validated HIP kernel) or "
        "K3_DISABLE_KDA=1 (all-MLA fallback).",
        flush=True,
    )
