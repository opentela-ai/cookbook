"""Standalone validation of hip_fp8_mqa_logits (no serving needed).

Run on a beverin GPU node:
    SGLANG_USE_HIP=1 python3 test_hip_mqa_fallback.py

Checks:
 1. torch._scaled_mm availability + correctness on MI300A (fp8 e4m3, per-tensor scale).
 2. hip_fp8_mqa_logits vs an INDEPENDENT fp32 reference, for each of the 3
    call-site shapes (plan / with-cp / per-request loop).
 3. -inf masking correctness (valid positions filled, others -inf).
 4. backend agreement: _scaled_mm vs bf16 vs fp32.
"""

import os
import sys

import torch

sys.path.insert(
    0,
    "/capstor/scratch/cscs/xyao/glm-53-flash-beverin/overlay/sgl-workspace/sglang/python",
)
from sglang.srt.layers.attention.dsa.kpool_hip_mqa_fallback import (  # noqa: E402
    hip_fp8_mqa_logits,
)

torch.manual_seed(0)
dev = torch.device("cuda")
H, D = 64, 128
INF = float("inf")
# Validate BOTH fp8 layouts: e4m3fn (CUDA) and e4m3fnuz (ROCm/MI300 unsigned-zero).
# The live beverin path quantizes to e4m3fnuz, so this is the case that matters.
FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e4m3fnuz]


def fp8(x: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    return x.to(dt)


def ref_logits(q_fp8, k_fp8, k_scale, w, rs, pe, clean):
    """Independent fp32 reference (dequant via fp32, einsum, fp32 throughout).

    Uses q_fp8 / k_fp8 with their NATIVE dtype (no .view reinterprets) -- this
    is the same contract as the fixed hip_fp8_mqa_logits, so it is correct for
    both e4m3fn and e4m3fnuz inputs.
    """
    q = q_fp8
    if q.dim() == 4:
        q = q.squeeze(1)
    k = k_fp8
    ks = k_scale.view(torch.float32).squeeze(-1)
    wf = w.to(torch.float32)
    # einsum is an independent path vs matmul
    g = torch.einsum("nhd,jd->nhj", q.to(torch.float32), k.to(torch.float32))
    g = torch.relu(g) * wf.unsqueeze(2)
    logits = g.sum(dim=1) * ks.unsqueeze(0)
    if clean:
        Nk = k.shape[0]
        idx = torch.arange(Nk, device=logits.device)
        valid = (idx.unsqueeze(0) >= rs.to(torch.int64).unsqueeze(1)) & (
            idx.unsqueeze(0) < pe.to(torch.int64).unsqueeze(1)
        )
        logits = torch.where(valid, logits, torch.full_like(logits, INF))
    return logits


def cmp(name, a, b):
    m = torch.isfinite(a) & torch.isfinite(b)
    if int(m.sum()) == 0:
        print(f"  {name:24s} NO finite overlap")
        return
    d = (a[m] - b[m]).abs()
    cos = torch.nn.functional.cosine_similarity(
        a[m].unsqueeze(0), b[m].unsqueeze(0)
    ).item()
    # also check -inf agreement
    infa = torch.isinf(a)
    infb = torch.isinf(b)
    inf_ok = int((infa == infb).sum()) == a.numel()
    print(
        f"  {name:24s} valid={int(m.sum()):6d} max_abs={float(d.max()):.6e} "
        f"mean_abs={float(d.mean()):.6e} cos={cos:.6f} inf_mask_ok={inf_ok}"
    )


def run_shape(tag, Nq, Nk, rs_mode):
    for dt in FP8_DTYPES:
        _run_shape(tag, Nq, Nk, rs_mode, dt)


def _run_shape(tag, Nq, Nk, rs_mode, dt):
    # random fp8-representable data (pre-quantized inputs)
    qf = (torch.randn(Nq, 1, H, D, device=dev) * 0.3)
    kf = (torch.randn(Nk, D, device=dev) * 0.3)
    q_fp8 = fp8(qf, dt)
    k_fp8 = fp8(kf, dt)
    k_scale = torch.rand(Nk, device=dev, dtype=torch.float32) * 0.5 + 0.5
    w = torch.randn(Nq, H, device=dev, dtype=torch.float32) * 0.1
    if rs_mode == "zeros":
        rs = torch.zeros(Nq, dtype=torch.int32, device=dev)
        pe = torch.randint(1, max(2, Nk // 2 + 1), (Nq,), dtype=torch.int32, device=dev)
    elif rs_mode == "ragged":
        # CSR: increasing starts, end = start + count (like site 1 ke=ks+pool_lens)
        counts = torch.randint(1, max(2, Nk // 4 + 1), (Nq,), dtype=torch.int32, device=dev)
        rs = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.int32, device=dev), counts[:-1]]), 0
        )
        pe = rs + counts
        # clip
        pe = torch.clamp(pe, max=Nk)
        rs = torch.clamp(rs, max=Nk)
    else:  # per-request (site 3): rs often 0
        rs = torch.zeros(Nq, dtype=torch.int32, device=dev)
        pe = torch.randint(1, Nk + 1, (Nq,), dtype=torch.int32, device=dev)

    print(f"\n=== {tag}: Nq={Nq} H={H} D={D} Nk={Nk} rs_mode={rs_mode} dt={dt} ===")

    ref = ref_logits(q_fp8, k_fp8, k_scale, w, rs, pe, clean=True)

    # bf16 backend
    os.environ["GLM53_HIP_MQA_FP32"] = "0"
    os.environ["GLM53_HIP_MQA_USE_SCALEDMM"] = "0"
    import importlib
    import sglang.srt.layers.attention.dsa.kpool_hip_mqa_fallback as fb
    importlib.reload(fb)
    out_bf16 = fb.hip_fp8_mqa_logits(q_fp8, (k_fp8, k_scale), w, rs, pe, clean_logits=True)
    cmp("bf16 vs ref", out_bf16, ref)

    # _scaled_mm backend
    os.environ["GLM53_HIP_MQA_USE_SCALEDMM"] = "1"
    importlib.reload(fb)
    try:
        out_smm = fb.hip_fp8_mqa_logits(q_fp8, (k_fp8, k_scale), w, rs, pe, clean_logits=True)
        cmp("scaled_mm vs ref", out_smm, ref)
        cmp("scaled_mm vs bf16", out_smm, out_bf16)
    except Exception as exc:  # noqa: BLE001
        print(f"  scaled_mm: UNAVAILABLE ({exc!r})")

    # fp32 backend (exact oracle)
    os.environ["GLM53_HIP_MQA_FP32"] = "1"
    os.environ["GLM53_HIP_MQA_USE_SCALEDMM"] = "0"
    importlib.reload(fb)
    out_fp32 = fb.hip_fp8_mqa_logits(q_fp8, (k_fp8, k_scale), w, rs, pe, clean_logits=True)
    cmp("fp32 vs ref", out_fp32, ref)

    # masking sanity: count of -inf == count of non-valid
    valid_cnt = int(((torch.arange(Nk, device=dev).unsqueeze(0) >= rs.to(torch.int64).unsqueeze(1)) &
                     (torch.arange(Nk, device=dev).unsqueeze(0) < pe.to(torch.int64).unsqueeze(1))).sum())
    inf_cnt = int(torch.isinf(out_fp32).sum())
    print(f"  mask: valid_cells={valid_cnt} finite_out={int((~torch.isinf(out_fp32)).sum())} "
          f"inf_out={inf_cnt} (expected_inf={Nq*Nk - valid_cnt})")


def test_scaled_mm():
    print("=== torch._scaled_mm availability (fp8 e4m3 + e4m3fnuz, per-tensor scale, MI300A) ===")
    M, K, Nn = 128, 128, 128
    ok_any = False
    for dt in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        a = (torch.randn(M, K, device=dev) * 0.3).to(dt)
        b = (torch.randn(K, Nn, device=dev) * 0.3).to(dt)
        one = torch.ones((1,), dtype=torch.float32, device=dev)
        try:
            g = torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32)
            ref = (a.to(torch.float32) @ b.to(torch.float32))
            d = (g - ref).abs()
            print(f"  _scaled_mm {dt}: OK out={tuple(g.shape)} dtype={g.dtype} "
                  f"max_abs_vs_fp32={float(d.max()):.6e}")
            ok_any = True
        except Exception as exc:  # noqa: BLE001
            print(f"  _scaled_mm {dt}: UNAVAILABLE ({exc!r})")
    return ok_any


if __name__ == "__main__":
    print(f"torch={torch.__version__} device={dev} "
          f"hip={getattr(torch.version, 'hip', None) is not None}")
    smm_ok = test_scaled_mm()

    # site 1 (plan): Nq up to ~2k, full ragged concatenation, CSR starts
    run_shape("site1_plan", Nq=64, Nk=512, rs_mode="ragged")
    # site 2 (with_cp): Nq=actual_seq_q (small), rs=0
    run_shape("site2_withcp", Nq=32, Nk=384, rs_mode="zeros")
    # site 3 (per-request loop): Nq=q_len (prefill tokens), rs=0, Nk=pool_seq_len
    run_shape("site3_perreq", Nq=128, Nk=256, rs_mode="per_request")
    # a bigger one to stress
    run_shape("site1_big", Nq=512, Nk=1024, rs_mode="ragged")

    print("\nDONE")
