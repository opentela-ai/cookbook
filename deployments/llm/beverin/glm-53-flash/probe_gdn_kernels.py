#!/usr/bin/env python3
"""Probe v4: (1) clean GDN chunk determinism, (2) tilelang mhc hc_pre vs _mhc_pre_torch."""

import sys

OVL = "/capstor/scratch/cscs/xyao/glm-53-flash-beverin/overlay"
for _p in (f"{OVL}/pylib", f"{OVL}/sgl-workspace/sglang/python"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

DEV = "cuda"


def l2norm(x):
    return x / (x.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)


def ref_chunk_gdn(q, k, v, g, beta, scale):
    T, H, K = q.shape
    V = v.shape[-1]
    qn, kn = l2norm(q), l2norm(k)
    S = torch.zeros(H, K, V, dtype=torch.float32, device=q.device)
    outs = []
    for t in range(T):
        S = S * torch.exp(g[t]).float()[:, None, None]
        kv = torch.einsum("hkv,hk->hv", S, kn[t])
        delta = v[t].float() - kv
        S = S + beta[t].float()[:, None, None] * kn[t][:, :, None] * delta[:, None, :]
        outs.append(torch.einsum("hkv,hk->hv", S, (scale * qn[t]).float()))
    return torch.stack(outs), S


def err(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    rel = d.mean() / b.abs().mean().clamp_min(1e-9)
    return f"max_abs={d.max().item():.6f} rel_mean={rel.item():.6f}"


def main():
    print(f"torch={torch.__version__} dev={torch.cuda.get_device_name(0)}")
    scale = 128 ** -0.5

    # ---------- 1. chunk_gdn with FRESH state per run ----------
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule

    for T in (2, 64, 256):
        H, K, V = 16, 128, 128
        torch.manual_seed(2)
        q = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16)
        k = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(1, T, H, V, device=DEV, dtype=torch.bfloat16) * 0.5
        g = -(torch.rand(1, T, H, device=DEV, dtype=torch.float32) * 0.1 + 0.01)
        beta = torch.sigmoid(torch.randn(1, T, H, device=DEV, dtype=torch.float32) * 0.5)
        outs, sts = [], []
        for run in range(2):
            ssm = torch.zeros(2, H, V, K, device=DEV, dtype=torch.float32)  # FRESH
            idx = torch.tensor([0], dtype=torch.int32, device=DEV)
            csl = torch.tensor([0, T], dtype=torch.int64, device=DEV)
            out = chunk_gated_delta_rule(
                q=q.clone(), k=k.clone(), v=v.clone(), g=g, beta=beta,
                initial_state=ssm, initial_state_indices=idx,
                cu_seqlens=csl, head_first=False, use_qk_l2norm_in_kernel=True,
            )
            outs.append(out[0])
            sts.append(ssm[0].clone())  # state written back in-place
        ro, _ = ref_chunk_gdn(q[0].float(), k[0].float(), v[0].float(), g[0], beta[0], scale)
        det_o = (outs[0].float() - outs[1].float()).abs().max().item()
        det_s = (sts[0].float() - sts[1].float()).abs().max().item()
        print(f"chunk_gdn T={T}: vs-ref {err(outs[0].reshape(T,H,V), ro)} | "
              f"det_o={det_o:.2e} det_S={det_s:.2e}"
              f"{'  *** NONDET ***' if max(det_o, det_s) > 1e-4 else ''}")

    # ---------- 2. mhc hc_pre: tilelang vs torch reference ----------
    import os
    os.environ["SGLANG_OPT_DEEPGEMM_HC_PRENORM"] = "0"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    try:
        import sglang.kernels.ops.layernorm.mhc as M
        from sglang.srt.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method="env://"
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        print("dist: 1-rank TP group initialized")
    except Exception as e:  # noqa: BLE001
        print(f"dist init failed ({e}); monkeypatching symmetric memory")
        import contextlib

        M = __import__("sglang.kernels.ops.layernorm.mhc", fromlist=["mhc"])
        M.get_tp_group = lambda *a, **k: None
        M.is_allocation_symmetric = lambda *a, **k: False
        M.use_symmetric_memory = lambda *a, **k: contextlib.nullcontext()
    from sglang.kernels.ops.layernorm.mhc import hc_pre, hc_post, _mhc_pre_torch, _mhc_post_torch

    N, HID, EPS = 4, 4096, 1e-6
    mix_hc = (2 + N) * N
    hc_dim = N * HID
    for T in (2049, 2505):
        torch.manual_seed(7)
        x = (torch.randn(T, N * HID, device=DEV, dtype=torch.bfloat16) * 0.5)
        hc_fn = torch.randn(mix_hc, hc_dim, device=DEV, dtype=torch.float32) * 0.02
        hc_scale = torch.rand(3, device=DEV, dtype=torch.float32) * 0.5 + 0.5
        hc_base = torch.randn(mix_hc, device=DEV, dtype=torch.float32) * 0.1

        args = dict(
            hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base, hc_mult=N,
            rms_eps=1e-5, hc_eps=EPS, sinkhorn_iters=20, post_mult_value=2.0,
            hc_norm_weight=None, out_norm_weight=None, out_norm_eps=None,
        )
        try:
            li_t, cr_t, pm_t, nf_t = hc_pre(x.clone(), **args)  # tilelang (env default 1)
            li_t2, cr_t2, pm_t2, _ = hc_pre(x.clone(), **args)
            # torch reference
            residual = x.view(T, N, HID).float()
            fn = hc_fn
            pm_r, cr_r, li_r = _mhc_pre_torch(
                residual=residual, fn=fn, hc_scale=hc_scale, hc_base=hc_base,
                rms_eps=1e-5, hc_pre_eps=EPS, hc_sinkhorn_eps=EPS,
                hc_post_mult_value=2.0, sinkhorn_repeat=20,
            )
            det = max(
                (li_t.float() - li_t2.float()).abs().max().item(),
                (cr_t.float() - cr_t2.float()).abs().max().item(),
                (pm_t.float() - pm_t2.float()).abs().max().item(),
            )
            print(f"\nhc_pre T={T}: shapes li={list(li_t.shape)} cr={list(cr_t.shape)} pm={list(pm_t.shape)}")
            print(f"    tilelang-vs-torchREF layer_input: {err(li_t, li_r)}")
            print(f"    tilelang-vs-torchREF comb_mix:    {err(cr_t, cr_r.reshape(T, N * N))}")
            print(f"    tilelang-vs-torchREF post_mix:    {err(pm_t, pm_r)}")
            print(f"    run1-vs-run2 det: {det:.2e}{'  *** NONDET ***' if det > 1e-3 else ''}")

            # hc_post: tilelang dispatch vs torch ref
            x_sub = torch.randn(T, HID, device=DEV, dtype=torch.bfloat16) * 0.3
            out_t = hc_post(x_sub.clone(), x.clone(), pm_t.clone(), cr_t.clone(), N)
            out_t2 = hc_post(x_sub.clone(), x.clone(), pm_t.clone(), cr_t.clone(), N)
            out_r = _mhc_post_torch(
                x_sub.float(), residual, pm_t.reshape(T, N, 1).float(), cr_t.reshape(T, N, N).float()
            )
            det_p = (out_t.float() - out_t2.float()).abs().max().item()
            print(f"    hc_post tilelang-vs-torchREF:    {err(out_t, out_r.reshape(T, -1))}")
            print(f"    hc_post run1-vs-run2 det: {det_p:.2e}{'  *** NONDET ***' if det_p > 1e-3 else ''}")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
