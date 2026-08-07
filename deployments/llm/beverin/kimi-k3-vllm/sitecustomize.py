"""sitecustomize: force vLLM's K3 SiTU MoE onto the AITER backend on gfx942.

Loaded automatically by the CPython interpreter at startup (this file lives at
$K3/home/pylib/sitecustomize.py, which k3_patch.py installs and engine.sh puts
first on PYTHONPATH), so it runs BEFORE vllm.model_executor builds any layer.

Why this is needed (k3-eng11 -> job 580844 lesson)
--------------------------------------------------
k3_patch.py makes the gfx950-targeted flydsl a16w4 MoE kernels RUN on gfx942
(LDS software fill, K16 MFMA split, software fp4->bf16 dequant, async off).
But vLLM's Python MoE-backend selector (`quantization/mxfp4.py:
_use_k3_situ_aiter`) ALSO gates the direct AITER path on `on_gfx950()`, which
queries amdsmi and returns False on real MI300A hardware. With it False, the
selector falls through to `oracle/mxfp4.py:select_deepseek_v4_mxfp4_moe_backend`
which finds no supported backend for Kimi-K3's SiTU activation and raises

    NotImplementedError: No MXFP4 MoE backend supports the deployment configuration.

- crash at ~4 min, before any shard loads. This sitecustomize.py is the Python
counterpart to k3_patch.py's C++ patches: it lies to the selector so the
engine takes the direct `AITER_MXFP4_BF16` branch (the path that k3-eng11
reached full init on, same image v0.1.dev19253+g5f76ae224, same TP8 x PP2).

Vendored verbatim from the working bring-up at
/capstor/scratch/cscs/xyao/kimi-k3/home/pylib/sitecustomize.py (job k3-eng11).
"""
import os

# Belt-and-suspenders: engine.sh sets these too, but setdefault keeps the
# AITER MoE backend enabled even if sitecustomize is imported before engine.sh
# exports them (e.g. a helper subprocess).
os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_MOE", "1")

# (1) on_gfx950() -> True. The flydsl kernels are PATCHED (k3_patch.py), not
# stock gfx950, so the amdsmi-reported arch (gfx942) must be hidden from the
# selector. Monkey-patching the module attribute is picked up by the late
# `from vllm.platforms.rocm import on_gfx950` inside _use_k3_situ_aiter.
try:
    from vllm.platforms import rocm as _rocm

    _rocm.on_gfx950 = lambda: True
except Exception:
    pass

# (2) Tell the AITER MXFP4 backend's is_supported_config that SiTU is OK.
# The direct `is_k3_situ_aiter=True` branch does not consult this, but the
# oracle (taken on other shapes / future images) does, so keep it in step.
try:
    from vllm.model_executor.layers.fused_moe.modular_kernel import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        RocmAiterMxfp4MoeBase,
    )

    _orig = RocmAiterMxfp4MoeBase._supports_activation

    @staticmethod
    def _patched(activation):
        if activation == MoEActivation.SITU:
            return True
        return _orig(activation)

    RocmAiterMxfp4MoeBase._supports_activation = _patched
except Exception:
    pass

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
