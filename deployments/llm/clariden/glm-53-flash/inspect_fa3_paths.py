#!/usr/bin/env python3
"""Round 8g (final gate): exact GLM-5.3 DSA dims + SDPA viability test
with those EXACT dims (q/k head_dim, v v_head_dim, GQA enable)."""
import torch, json
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(
    "/capstor/scratch/cscs/xyao/models/zai-org/GLM-5.3-Flash",
    trust_remote_code=True)
print("=== GLM-5.3 DSA-relevant config ===")
for k_ in ("head_dim","v_head_dim","kv_lora_rank","num_attention_heads",
           "num_key_value_heads","qk_nope_head_dim","qk_rope_head_dim",
           "hidden_size","num_hidden_layers","linear_attn_config",
           "layer_types","num_experts","moe_intermediate_size"):
    v = getattr(cfg, k_, None)
    if v is not None:
        s = json.dumps(v, default=str)[:160] if not isinstance(v,(int,float,str,bool)) else v
        print(f"  {k_} = {s}")

# Build the EXACT _forward_standard_mha shapes for one synthetic request.
# q: (seq_q, num_q_heads=tp_q, head_dim); k: (seq_k, num_kv=tp_kv, head_dim);
# v: (seq_k, num_kv, v_head_dim)
num_q = getattr(cfg, "num_attention_heads", 8)
num_kv = getattr(cfg, "num_key_value_heads", num_q)
hd = getattr(cfg, "head_dim", None) or (getattr(cfg,"hidden_size",128)//num_q)
vhd = getattr(cfg, "v_head_dim", None) or hd
qkn = getattr(cfg, "qk_nope_head_dim", None)
qkr = getattr(cfg, "qk_rope_head_dim", None)
print(f"\n=== _forward_standard_mha EXACT dims (1 TP rank, no TP) ===")
print(f"  num_q={num_q} num_kv={num_kv} head_dim={hd} v_head_dim={vhd} qk_nope={qkn} qk_rope={qkr}")
print(f"  GQA? num_q != num_kv -> {num_q != num_kv}; v_head_dim != head_dim -> {vhd != hd}")

F = torch.nn.functional
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16
sl_q, sl_k = 64, 64  # one-shot prefill, q==k
q = torch.randn(1, sl_q, num_q, hd, device=dev, dtype=dt)
k = torch.randn(1, sl_k, num_kv, hd, device=dev, dtype=dt)
v = torch.randn(1, sl_k, num_kv, vhd, device=dev, dtype=dt)
print(f"\n=== SDPA test (EXACT dims, q==sl_k, enable_gqa=True, is_causal=True) ===")
try:
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                       scale=1.0/hd**0.5, enable_gqa=(num_q!=num_kv))
    print(f"  OK: out {tuple(o.shape)}  (expect (1,{sl_q},{num_q},{vhd}))")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:160]}")

# also test sl_q != sl_k (prefix+current) with explicit bottom-right mask
sl_q2 = 16
q2 = torch.randn(1, sl_q2, num_q, hd, device=dev, dtype=dt)
k2 = torch.randn(1, sl_k, num_kv, hd, device=dev, dtype=dt)
v2 = torch.randn(1, sl_k, num_kv, vhd, device=dev, dtype=dt)
mask = torch.ones(sl_q2, sl_k, device=dev, dtype=torch.bool).tril(diagonal=sl_k - sl_q2)
print("=== SDPA test (sl_q<sl_k, explicit bottom-right mask, enable_gqa) ===")
try:
    o2 = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=mask,
                                        scale=1.0/hd**0.5, enable_gqa=(num_q!=num_kv))
    print(f"  OK: out {tuple(o2.shape)}  (expect (1,{sl_q2},{num_q},{vhd}))")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:160]}")
