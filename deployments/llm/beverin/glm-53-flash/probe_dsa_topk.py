"""Probe vk_hip_dsa_topk_logits (vkernels issue #51, the kpool>1 indexer's
per-paged-KV-tile gated logit) on MI300A — a NUMERICALLY EXACT smoke that
needs no sglang import (no JIT, no model load; ~10 s once a GPU is on).

WHY NO SGLANG IMPORT
--------------------
Importing sglang.kernels.ops.attention.dsa.tilelang_kernel pulls in the full
tilelang/aiter graph (~4-5 min on a cold node) and the @tilelang.jit defs only
run on first call. This probe instead imports the ctypes adapter
(vkernels_dsa.tilelang_fp8_paged_mqa_logits, a drop-in for sglang's symbol)
and calls it directly on synthetic inputs with COMPUTABLE expected values.
The serve smoke (serve_glm_53_flash_sglang.sbatch) exercises the real rebind
end-to-end; this just proves the HIP kernel is numerically correct on the
shapes the indexer will feed it.

KV-CACHE LAYOUT (IndexKeyCache._buffer_shape, index_key_cache.py:32-37 +
SetKAndS._set_k_and_s_triton, index_buf_accessor.py:346)
------------------------------------------------------------------------
The index-K-with-scale buffer is allocated 2-D uint8:
    (num_pages, page_size * (index_head_dim + index_head_dim // quant_block_size * 4))
For GLM-5.3 (page_size=64, index_head_dim=128, quant_block_size=128):
    (num_pages, 64 * (128 + 4)) = (num_pages, 8448)
and the WRITE path stores all 64 keys' bytes FIRST then all 64 fp32 scales:
    [page, 0      : 8192] = 64 tokens x 128 fp8 e4m3fnuz keys   (S_OFFSET=8192)
    [page, 8192   : 8448] = 64 fp32 per-token scales (packed, 256 bytes)
This is LAYOUT A (contiguous keys then packed scales), which is exactly what
the tilelang kernel (tilelang_kernel.py:1483-1491) AND my C kernel
(dsa.hip:271-328) read via the wrapper's `view(-1, block*(hd+4))` ->
`kvcache_u8[page, 0:8192]` keys + `kvcache_u8[page, 8192:8448]` scales.
(The decorative `.view((64,1,132))` in _get_topk_paged:889 is never read
that way — the wrapper re-views as (-1, 8448) before the kernel.)

KERNEL CONTRACT (dsa.hip:271-328), per output token t = i*B + lane:
  page = page_table[b*max_table_len + i]
  K_dequant[lane,d] = fp8e4m3fnuz_to_f32(kvcache_u8[page*B*(D+4) + lane*D + d])
  K_scale[lane]     = reinterpret_cast<float>(kvcache_u8[page*B*(D+4) + B*D + lane*4])
  Q_dequant[h,d]    = fp8e4m3fnuz_to_f32(q_fp8[b*H*D + h*D + d])
  dot_h             = sum_d (Q_dequant[h,d] * K_dequant[lane,d])      # no 1/sqrt(D)
  acc               = sum_h (max(dot_h, 0.0) * gate[b,h])            # ReLU then gate
  out[b,t]          = K_scale[lane] * acc                             # only if t < seq_len[b]
(else out[b,t] stays 0 — the adapter zero-inits; strictly safer than the
tilelang wrapper's new_empty, and correct: tokens t >= seq_lens[b] are
masked out of the top-k by topk_from_pooled_history_logits via
group_lengths/topk_offsets before selection.)

The three cases below each have a closed-form expected value:
  * case 1: bs=1 nh=1 — Q=1,K=1,scale=1,gate=1 -> every valid token == 128.0
  * case 2: bs=1 nh=2 — Q[h0]=1,Q[h1]=2 -> 384.0; max_seq_len=128 but
            seq_len=64 -> tokens [64,128) are ZERO (page 1 never iterated)
  * case 3: bs=2 nh=1 — Q[1]=-1 -> ReLU zeros it -> 0.0; seq_lens=[64,32]
            -> batch 1 tokens [32,64) are ZERO (not written: t < seq_len)
"""

import sys

import torch
import vkernels_dsa


def _fp8u(x, *, shape, device):
    """Tensor of `x` (python float) in torch.float8_e4m3fnuz, the kernel dtype."""
    return torch.full(shape, float(x), dtype=torch.float8_e4m3fnuz, device=device)


def _kv_layout_a(num_pages, k_val, scale_val, *, block, head_dim, device):
    """Build the index-K-with-scale buffer in LAYOUT A (matching
    IndexKeyCache._buffer_shape + SetKAndS._set_k_and_s_triton,
    S_OFFSET = page_size*head_dim), then return the DECORATIVE 4-D
    (num_pages, block, 1, head_dim+4) view the adapter asserts
    (vkernels_dsa.py L359-361 mirror the tilelang wrapper's asserts). The
    underlying bytes are [page, 0:block*D]=B keys | [page, block*D:B*(D+4)]=B
    packed fp32 scales; the 4-D view is shape-only — the adapter immediately
    re-views as (-1, block*(D+4)) so the kernel reads layout A correctly."""
    B, D = block, head_dim
    kv = torch.zeros((num_pages, B * (D + 4)), dtype=torch.uint8, device=device)
    k_byte = torch.tensor([float(k_val)], dtype=torch.float8_e4m3fnuz,
                          device=device).view(torch.uint8).item()
    kv[:, : B * D] = k_byte                                    # B*D key bytes
    scale_bytes = torch.tensor([float(scale_val)], dtype=torch.float32,
                               device=device).view(torch.uint8).repeat(B)  # B*4 bytes
    kv[:, B * D : B * (D + 4)] = scale_bytes
    return kv.view(num_pages, B, 1, D + 4)                     # decorative 4-D


def _case1(device):
    # bs=1 nh=1 hd=128 block=64 max_table_len=1 max_seq_len=64 seq_len=64
    # Q=1,K=1,scale=1,gate=1 -> dot=128,acc=128,out=128.0 (every token)
    q = _fp8u(1.0, shape=(1, 1, 1, 128), device=device)
    kv = _kv_layout_a(1, 1.0, 1.0, block=64, head_dim=128, device=device)
    weight = torch.full((1, 1), 1.0, dtype=torch.float32, device=device)
    seq_lens = torch.tensor([64], dtype=torch.int32, device=device)
    page_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    out = vkernels_dsa.tilelang_fp8_paged_mqa_logits(
        q, kv, weight, seq_lens, page_table, None, max_seq_len=64, clean_logits=False
    )
    exp = torch.full((1, 64), 128.0, dtype=torch.float32, device=device)
    return _check("case1 (nh=1 all-ones, 1 full page)", out, exp, (1, 64))


def _case2(device):
    # bs=1 nh=2 hd=128 block=64 max_table_len=2 max_seq_len=128 seq_len=64
    # Q[h0]=1,Q[h1]=2; K=1; scale=1; gate=[1,1]
    # dot_0=128, dot_1=256, acc=128+256=384; only page 0 iterated
    # (np_total=(64+63)/64=1) -> tokens [64,128) ZERO
    q = torch.cat([_fp8u(1.0, shape=(1, 1, 1, 128), device=device),
                   _fp8u(2.0, shape=(1, 1, 1, 128), device=device)], dim=2)  # (1,1,2,128)
    kv = _kv_layout_a(2, 1.0, 1.0, block=64, head_dim=128, device=device)
    weight = torch.full((1, 2), 1.0, dtype=torch.float32, device=device)
    seq_lens = torch.tensor([64], dtype=torch.int32, device=device)
    page_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    out = vkernels_dsa.tilelang_fp8_paged_mqa_logits(
        q, kv, weight, seq_lens, page_table, None, max_seq_len=128, clean_logits=False
    )
    exp = torch.zeros((1, 128), dtype=torch.float32, device=device)
    exp[0, :64] = 384.0
    return _check("case2 (nh=2, partial page table -> tail zero)", out, exp, (1, 128))


def _case3(device):
    # bs=2 nh=1 hd=128 block=64 max_table_len=1 max_seq_len=64 seq_lens=[64,32]
    # Q[0]=1 -> dot=128 -> acc=128 -> out=128 (tokens [0,64))
    # Q[1]=-1 -> dot=-128 -> max(-128,0)=0 -> acc=0 -> out=0 (tokens [0,32))
    # batch 1 tokens [32,64) NOT written (t < seq_len=32) -> 0 (zero-init)
    q = torch.cat([_fp8u(1.0, shape=(1, 1, 1, 128), device=device),
                   _fp8u(-1.0, shape=(1, 1, 1, 128), device=device)], dim=0)  # (2,1,1,128)
    kv = _kv_layout_a(2, 1.0, 1.0, block=64, head_dim=128, device=device)
    weight = torch.full((2, 1), 1.0, dtype=torch.float32, device=device)
    seq_lens = torch.tensor([64, 32], dtype=torch.int32, device=device)
    page_table = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
    out = vkernels_dsa.tilelang_fp8_paged_mqa_logits(
        q, kv, weight, seq_lens, page_table, None, max_seq_len=64, clean_logits=False
    )
    exp = torch.zeros((2, 64), dtype=torch.float32, device=device)
    exp[0, :64] = 128.0   # batch 0: all 64 valid
    exp[1, :32] = 0.0     # batch 1: ReLU zeroes it, 32 valid
    return _check("case3 (bs=2, Q=-1 ReLU + per-batch seq_len mask)", out, exp, (2, 64))


def _check(name, out, exp, want_shape):
    ok = True
    if tuple(out.shape) != want_shape:
        sys.stderr.write(f"  {name}: SHAPE {tuple(out.shape)} != {want_shape}\n")
        ok = False
    else:
        diff = (out - exp).abs()
        max_d = float(diff.max().item())
        nbad = int((diff > 1e-4).sum().item())
        if nbad:
            sys.stderr.write(
                f"  {name}: MISMATCH max|err|={max_d:.6g} ({nbad}/{exp.numel()} > 1e-4)\n"
            )
            sys.stderr.write(f"    out[:16]  = {out.flatten()[:16].tolist()}\n")
            sys.stderr.write(f"    exp[:16]  = {exp.flatten()[:16].tolist()}\n")
            sys.stderr.write(f"    out[-16:] = {out.flatten()[-16:].tolist()}\n")
            ok = False
        else:
            sys.stderr.write(
                f"  {name}: OK (max|err|={max_d:.3g}, "
                f"out[0,0]={out.flatten()[0].item():.6g})\n"
            )
    sys.stderr.flush()
    return ok


def main():
    if not torch.cuda.is_available():
        sys.stderr.write("NO CUDA DEVICE — cannot run the HIP kernel smoke\n")
        sys.exit(2)
    dev = torch.device("cuda:0")
    gcn = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    sys.stderr.write(
        f"[probe_dsa_topk] device={dev} gcn={gcn} "
        f"torch={torch.__version__} cuda={torch.version.cuda}\n"
    )
    sys.stderr.write("[probe_dsa_topk] symbols: vk_hip_dsa_topk_logits=")
    lib = vkernels_dsa._get_lib()
    fn = getattr(lib, "vk_hip_dsa_topk_logits", None)
    sys.stderr.write(f"{fn is not None}\n")
    if fn is None:
        sys.stderr.write("  vk_hip_dsa_topk_logits NOT in libvkernels_hip.so\n")
        sys.exit(3)
    # bind (sets argtypes; idempotent)
    vkernels_dsa._bind_dsa_topk_fn(lib)

    results = [_case1(dev), _case2(dev), _case3(dev)]
    if all(results):
        sys.stderr.write(
            "[probe_dsa_topk] ALL 3 CASES PASSED — vk_hip_dsa_topk_logits is "
            "numerically correct on gfx942 (layout A: keys then packed scales); "
            "the kpool>1 indexer path is ready for the serve smoke.\n"
        )
        sys.exit(0)
    sys.stderr.write(
        f"[probe_dsa_topk] FAILED ({sum(results)}/{len(results)} cases)\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
