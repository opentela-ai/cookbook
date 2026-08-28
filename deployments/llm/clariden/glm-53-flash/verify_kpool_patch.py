#!/usr/bin/env python3
"""Round 8h: verify the SDPA ragged re-impl of _forward_standard_mha runs
on sm_90 with the EXACT GLM-5.3 DSA dims (64 heads, head_dim=v_head_dim=256,
no GQA), single + multi-request, sl_q==sl_k (one-shot prefill)."""
import torch, types
from sglang.srt.layers.attention import dsa_backend as _db
import importlib as _il
def _importable(n):
    try:
        _il.import_module(n); return True
    except Exception:
        return False
_fa3_ok = _importable("sgl_kernel.flash_ops")
print(f"_fa3_ok = {_fa3_ok}  (expect False on aarch64)")
print(f"cuda: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'}  "
      f"sm={torch.cuda.get_device_capability() if torch.cuda.is_available() else '-'}")

# --- replicate the patched method (same code as the sbatch heredoc) ---
_F = torch.nn.functional
def _forward_standard_mha_fa3safe(self, q, k, v, layer, forward_batch, metadata):
    if self.device_sm_major != 9:
        raise RuntimeError("non-SM90 branch should not run in this verify")
    q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
    k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
    v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
    cu_q, cu_k = metadata.cu_seqlens_q, metadata.cu_seqlens_k
    causal = True
    scale = layer.scaling
    gqa = (q.shape[-2] != k.shape[-2])
    out = torch.empty_like(q)
    for _i in range(len(cu_q) - 1):
        _qs, _qe = int(cu_q[_i]), int(cu_q[_i + 1])
        if _qe <= _qs:
            continue
        _ks, _ke = int(cu_k[_i]), int(cu_k[_i + 1])
        _qi = q[_qs:_qe][None]; _ki = k[_ks:_ke][None]; _vi = v[_ks:_ke][None]
        _sl_q, _sl_k = _qe - _qs, _ke - _ks
        if causal and _sl_q == _sl_k:
            _oi = _F.scaled_dot_product_attention(_qi, _ki, _vi, is_causal=True, scale=scale, enable_gqa=gqa)
        elif causal:
            _m = torch.ones(_sl_q, _sl_k, device=q.device, dtype=torch.bool).tril(diagonal=_sl_k - _sl_q)[None, None]
            _oi = _F.scaled_dot_product_attention(_qi, _ki, _vi, attn_mask=_m, scale=scale, enable_gqa=gqa)
        else:
            _oi = _F.scaled_dot_product_attention(_qi, _ki, _vi, scale=scale, enable_gqa=gqa)
        out[_qs:_qe] = _oi[0]
    return out

# REAL GLM-5.3 DSA dims (config.json text_config)
NUM_Q = NUM_KV = 64           # num_attention_heads == num_key_value_heads
HEAD_DIM = V_HEAD_DIM = 256   # qk_nope_head_dim == v_head_dim, qk_rope=0
dev = "cuda"; dt = torch.bfloat16
self_ = types.SimpleNamespace(device_sm_major=9, device=dev)
layer = types.SimpleNamespace(tp_q_head_num=NUM_Q, tp_k_head_num=NUM_KV,
                              tp_v_head_num=NUM_KV, head_dim=HEAD_DIM,
                              v_head_dim=V_HEAD_DIM, scaling=1.0/HEAD_DIM**0.5)

# --- single request, sl_q == sl_k (one-shot prefill, no prefix) ---
sl = 128
q = torch.randn(sl, NUM_Q, HEAD_DIM, device=dev, dtype=dt)
k = torch.randn(sl, NUM_KV, HEAD_DIM, device=dev, dtype=dt)
v = torch.randn(sl, NUM_KV, V_HEAD_DIM, device=dev, dtype=dt)
md = types.SimpleNamespace(
    cu_seqlens_q=torch.tensor([0, sl], device=dev, dtype=torch.int32),
    cu_seqlens_k=torch.tensor([0, sl], device=dev, dtype=torch.int32))
o = _forward_standard_mha_fa3safe(self_, q, k, v, layer, None, md)
print(f"\n[single req, sl={sl}] out {tuple(o.shape)}  expect ({sl},{NUM_Q},{HEAD_DIM})  "
      f"{'OK' if tuple(o.shape)==(sl,NUM_Q,HEAD_DIM) else 'FAIL'}  "
      f"finite={torch.isfinite(o).all().item()}")

# --- 3 requests, ragged, sl_q == sl_k each ---
s_q = [0, 32, 96, 64+0]; s_k = [0, 32, 96, 64+0]
# build distinct seq lens: 32, 64, 128
cu_q = torch.tensor([0, 32, 96, 224], device=dev, dtype=torch.int32)
cu_k = torch.tensor([0, 32, 96, 224], device=dev, dtype=torch.int32)
tot = int(cu_q[-1])
q3 = torch.randn(tot, NUM_Q, HEAD_DIM, device=dev, dtype=dt)
k3 = torch.randn(tot, NUM_KV, HEAD_DIM, device=dev, dtype=dt)
v3 = torch.randn(tot, NUM_KV, V_HEAD_DIM, device=dev, dtype=dt)
md3 = types.SimpleNamespace(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k)
o3 = _forward_standard_mha_fa3safe(self_, q3, k3, v3, layer, None, md3)
print(f"[3 ragged reqs 32/64/128] out {tuple(o3.shape)}  expect ({tot},{NUM_Q},{HEAD_DIM})  "
      f"{'OK' if tuple(o3.shape)==(tot,NUM_Q,HEAD_DIM) else 'FAIL'}  "
      f"finite={torch.isfinite(o3).all().item()}")

# --- sl_q < sl_k (prefix+current) via the explicit-mask branch ---
sl_q2, sl_k2 = 16, 64
q2 = torch.randn(sl_q2, NUM_Q, HEAD_DIM, device=dev, dtype=dt)
k2 = torch.randn(sl_k2, NUM_KV, HEAD_DIM, device=dev, dtype=dt)
v2 = torch.randn(sl_k2, NUM_KV, V_HEAD_DIM, device=dev, dtype=dt)
md2 = types.SimpleNamespace(
    cu_seqlens_q=torch.tensor([0, sl_q2], device=dev, dtype=torch.int32),
    cu_seqlens_k=torch.tensor([0, sl_k2], device=dev, dtype=torch.int32))
try:
    o2 = _forward_standard_mha_fa3safe(self_, q2, k2, v2, layer, None, md2)
    print(f"[prefix+current sl_q=16<sl_k=64] out {tuple(o2.shape)}  expect (16,{NUM_Q},{HEAD_DIM})  "
          f"{'OK' if tuple(o2.shape)==(16,NUM_Q,HEAD_DIM) else 'FAIL'}  "
          f"finite={torch.isfinite(o2).all().item()}")
except Exception as e:
    print(f"[prefix+current sl_q=16<sl_k=64] FAIL: {type(e).__name__}: {str(e)[:160]}")
