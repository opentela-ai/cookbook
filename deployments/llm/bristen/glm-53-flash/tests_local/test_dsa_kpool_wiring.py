#!/usr/bin/env python3
"""Local CPU validation: vkernels#60 dsa_kpool wiring vs sglang's fp8e4nv
Triton kpool-cache kernels (bristen blocker, job 82822).

Run:  ~/.venvs/glm53-wiring/bin/python tests_local/test_dsa_kpool_wiring.py

What is tested (no GPU needed):
  T0  patched module imports; SM80 gate logic (env FORCE/DISABLE, cuda cap).
  T1  assemble: vkernels dsa_kpool_assemble (public API) == independent torch
      reference of the Triton kernel math (tight fp32 tol); patched wrapper
      writes the same values into a flat bf16 cache and preserves untouched
      slots (sentinel check).
  T2  decode: same, incl. pool-complete gating, current-token substitution,
      unconditional tail update, invalid-row no-ops and the block_tables
      row clamp.
  T3  dispatch proof: with every Triton kernel in the patched module replaced
      by a launch sentinel that RAISES, both wrappers complete through the
      native path -- the fp8e4nv JIT (job 82822's failure) is unreachable.
  T4  negative control: the ORIGINAL (unpatched) wrapper enters the Triton
      launch path (raises on this CPU box; on SM80 the real error is the
      fp8e4nv ValueError).
  T5  legacy-fp8 relationship: an emulation of the original kernel's
      fp8-quantize (scale = absmax/448) of the same reference x dequantizes
      back to x within fp8 tolerance -- i.e. both storages approximate the
      same math; the native one just drops the scale.

The reference implementations below are written from
third_party/sglang .../dsa/kpool_fp8_index.py::_kpool_assemble_softmax_rotate_write_cache_kernel
and ::_kpool_decode_update_and_maybe_write_cache_kernel (submodule pin
f5bed255), NOT from the vkernels code, so agreement is meaningful.

  T6  assemble BRIDGE (the bristen serving path): uint8 cache (legacy fp8+
      scale layout) + vkernels compute + requant -- dequantized cache values
      == ref_assemble's fp8 emulation within one fp8 ulp; pow2 round_scale
      variant; untouched slots keep zero bytes. Completion on CPU is itself
      the dispatch proof (the legacy Triton launch would raise here).
  T7  decode BRIDGE vs the T2-validated native (bf16) path: the bridge must
      equal quantize(native values) in the legacy layout, tails identical.
  T8  sitecustomize deep_gemm shim: paged + ragged torch logits fallbacks vs
      an independent per-element loop reference of the tilelang/DeepGEMM
      semantics (relu(q.k)*w summed over heads, x k_scale), including the
      page-table column mapping (col = table_idx*64 + j), the -1e30 fill for
      unwritten paged columns, and the ragged clean_logits masking.
  T9  act_quant SM80 fallback (patched triton_kernel.py): torch fp8+scale
      quantize byte-equal to an independent emulation of _act_quant_kernel
      (amax clamp 1e-4, scale = absmax/448 or pow2-ceiling round_scale, RTNE
      e4m3 store), exact shape/dtype contract, zero-guard and clamp behavior,
      2-D/3-D/multi-block shapes -- the decode-path wall of jobs 82822/83091.
"""

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

RECIPE = Path(__file__).resolve().parents[1]
COOKBOOK = RECIPE.parents[3]
sys.path.insert(0, str(COOKBOOK / "third_party" / "vkernels" / "src" / "python"))

PATCHED = RECIPE / "patched_sources" / "sglang" / "srt" / "layers" / "attention" / "dsa" / "kpool_fp8_index.py"
ORIGINAL = COOKBOOK / "third_party" / "sglang" / "python" / "sglang" / "srt" / "layers" / "attention" / "dsa" / "kpool_fp8_index.py"
TRITON_KERNEL_PATCHED = RECIPE / "patched_sources" / "sglang" / "kernels" / "ops" / "attention" / "dsa" / "triton_kernel.py"

HEAD = 128
SENTINEL = 999.0


def _quant_bytes(x, round_scale=False):
    """Raw fp8 bytes + scale of the kernel's quantize (scale = absmax/448 or
    enclosing pow2); x [..., 128] fp32 -> (fp8e4m3 tensor, scale [...])."""
    absmax = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1e-4)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    else:
        scale = absmax / 448.0
    q = torch.clamp(x / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale.squeeze(-1)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Pool:
    def __init__(self, index_kpool=4, tail_extra_slots=2, slots_per_page=8):
        self.index_kpool = index_kpool
        self.tail_extra_slots = tail_extra_slots
        self.page_size = 64
        self.slots_per_page = slots_per_page
        self.index_head_dim = HEAD


def hadamard128():
    i = torch.arange(HEAD).unsqueeze(1)
    j = torch.arange(HEAD).unsqueeze(0)
    ij = i & j  # int64
    par = torch.zeros_like(ij)
    for bit in range(7):  # popcount parity over 7 bits (0..127)
        par += (ij >> bit) & 1
    signs = 1.0 - 2.0 * (par % 2).float()
    return signs / math.sqrt(HEAD)


H128 = hadamard128()


def had(x):
    """x: [..., 128] fp32 -> Hadamard-rotated (torch, fp32)."""
    return x @ H128.T


# --------------------------------------------------------------------------
# References (independent re-implementation of the sglang Triton kernels)
# --------------------------------------------------------------------------

def _fp8_quantize_emulation(x, round_scale=False):
    """The original kernel's tail: fp8e4m3 quantize with per-vector scale.

    x is the (already double-bf16-rounded) Hadamard output, fp32.
    Returns (dequantized fp32, scale) -- what the fp8 cache preserves.
    """
    absmax = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1e-4)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    else:
        scale = absmax / 448.0
    q = torch.clamp(x / scale, -448.0, 448.0)
    # e4m3: 3 mantissa bits -> half-ulp rel err 2^-4 for normals;
    # dequantize through the real fp8 dtype to be exact.
    q_fp8 = q.to(torch.float8_e4m3fn).float()
    return q_fp8 * scale, scale.squeeze(-1)


def ref_assemble(chunk_k, chunk_score, tail_k, tail_score, ape, rpi, nft,
                 css, tlb, loc, wm, pool_size, tail_size, ssp, num_pages):
    """Returns fp32 [num_pages, ssp, 128]: the Hadamard-rotated gated mean
    (pre-fp8-quantize) per written row, zeros elsewhere -- plus the legacy
    fp8 dequant emulation of the same rows."""
    x_ref = torch.zeros(num_pages, ssp, HEAD)
    x_fp8 = torch.zeros(num_pages, ssp, HEAD)
    n_pools = rpi.shape[0]
    for r in range(n_pools):
        if wm is not None and wm[r] == 0:
            continue
        m = torch.full((HEAD,), -float("inf"))
        acc = torch.zeros(HEAD)
        denom = torch.zeros(HEAD)
        for s in range(pool_size):
            if s < nft[r]:
                phys = (tlb[r] + s) % tail_size
                sc = tail_score[rpi[r], phys].float()
                k = tail_k[rpi[r], phys].float()
            else:
                ci = css[r] + (s - nft[r])
                sc = chunk_score[ci].float()
                k = chunk_k[ci].float()
            score = sc + ape[s]
            new_m = torch.maximum(m, score)
            rescale = torch.exp(m - new_m)
            prob = torch.exp(score - new_m)
            denom = denom * rescale + prob
            acc = acc * rescale + k * prob
            m = new_m
        x = had(acc / denom)  # fp32 truth (kernel rounds bf16 twice before fp8)
        page, sip = loc[r] // ssp, loc[r] % ssp
        x_ref[page, sip] = x
        xb = (acc / denom).to(torch.bfloat16).float()
        x_fp8[page, sip] = _fp8_quantize_emulation(had(xb).to(torch.bfloat16).float())[0]
    return x_ref, x_fp8


def ref_decode(key, slot_score, tail_k, tail_score, ape, block_tables,
               rpi, pos, seq_lens, ocl, pool_size, tail_size, ssp,
               num_pages):
    """Returns (cache fp32 [num_pages, ssp, 128], tail_k', tail_score')."""
    batch = key.shape[0]
    n_reqs = tail_k.shape[0]
    btc = block_tables.shape[1]
    cache = torch.zeros(num_pages, ssp, HEAD)
    tk = tail_k.clone()
    ts = tail_score.clone()
    for r in range(batch):
        req_raw = int(rpi[r])
        req_valid = 0 <= req_raw < n_reqs
        req = min(max(req_raw, 0), n_reqs - 1)
        p = int(pos[r])
        pos_valid = req_valid and int(ocl[r]) != 0 and 0 <= p < int(seq_lens[r])
        slot = max(p, 0) % pool_size
        phys = max(p, 0) % tail_size
        if pos_valid:
            tk[req, phys] = key[r].float()
            ts[req, phys] = slot_score[r].float()
        if pos_valid and slot == pool_size - 1:
            pool_start = max(p, 0) - slot
            m = torch.full((HEAD,), -float("inf"))
            acc = torch.zeros(HEAD)
            denom = torch.zeros(HEAD)
            for s in range(pool_size):  # two-pass (max, then sum)
                phys_s = (pool_start + s) % tail_size
                sc = slot_score[r].float() if s == slot else ts[req, phys_s].float()
                k = key[r].float() if s == slot else tk[req, phys_s].float()
                score = sc + ape[s]
                m = torch.maximum(m, score)
            for s in range(pool_size):
                phys_s = (pool_start + s) % tail_size
                sc = slot_score[r].float() if s == slot else ts[req, phys_s].float()
                k = key[r].float() if s == slot else tk[req, phys_s].float()
                score = sc + ape[s]
                prob = torch.exp(score - m)
                denom = denom + prob
                acc = acc + k * prob
            x = had(acc / denom)
            pool_id = max(p, 0) // pool_size
            bt_row = min(max(pool_id // ssp * pool_size, 0), btc - 1)
            page = int(block_tables[r, bt_row])
            cache[page, pool_id % ssp] = x
    return cache, tk, ts


# --------------------------------------------------------------------------
# Case builders
# --------------------------------------------------------------------------

def make_assemble_case(seed):
    g = torch.Generator().manual_seed(seed)
    pool_size, tail_size, ssp, num_pages, num_chunks, n_pools, n_reqs = 4, 6, 8, 3, 12, 5, 3
    chunk_k = torch.randn(num_chunks, HEAD, generator=g).to(torch.bfloat16)
    chunk_score = torch.randn(num_chunks, HEAD, generator=g).to(torch.bfloat16)
    tail_k = torch.randn(n_reqs, tail_size, HEAD, generator=g).to(torch.bfloat16)
    tail_score = torch.randn(n_reqs, tail_size, HEAD, generator=g).to(torch.bfloat16)
    ape = torch.randn(pool_size, HEAD, generator=g)
    rpi = torch.tensor([0, 1, 2, 0, 1])
    nft = torch.tensor([0, 2, 4, 4, 1])
    css = torch.tensor([0, 2, 4, 8, 5])  # css + (pool - nft) <= 12
    tlb = torch.tensor([3, 5, 1, 0, 4])
    loc = torch.tensor([0, 7, 8, 17, 23])  # across pages 0..2
    return dict(pool_size=pool_size, tail_size=tail_size, ssp=ssp,
                num_pages=num_pages, chunk_k=chunk_k, chunk_score=chunk_score,
                tail_k=tail_k, tail_score=tail_score, ape=ape, rpi=rpi,
                nft=nft, css=css, tlb=tlb, loc=loc, wm=None, g=g)


def make_decode_case(seed):
    g = torch.Generator().manual_seed(seed)
    pool_size, tail_size, ssp, num_pages, batch, n_reqs, btc = 4, 6, 2, 4, 6, 3, 3
    key = torch.randn(batch, HEAD, generator=g).to(torch.bfloat16)
    slot_score = torch.randn(batch, HEAD, generator=g).to(torch.bfloat16)
    tail_k = torch.randn(n_reqs, tail_size, HEAD, generator=g).to(torch.bfloat16)
    tail_score = torch.randn(n_reqs, tail_size, HEAD, generator=g).to(torch.bfloat16)
    ape = torch.randn(pool_size, HEAD, generator=g)
    block_tables = torch.tensor([[1, 0, 2], [0, 2, 1], [2, 1, 0],
                                 [1, 1, 2], [0, 0, 0], [2, 2, 0]], dtype=torch.int32)
    # r0: valid, complete, no clamp | r1: valid, incomplete | r2: bad req
    # r3: ocl=0 | r4: pos >= seq_len | r5: valid, complete, bt_row CLAMPED
    rpi = torch.tensor([0, 1, -1, 2, 1, 2])
    pos = torch.tensor([3, 5, 3, 7, 99, 15])
    seq_lens = torch.tensor([10, 10, 10, 10, 10, 20])
    ocl = torch.tensor([5, 6, 7, 0, 9, 10])
    return dict(pool_size=pool_size, tail_size=tail_size, ssp=ssp,
                num_pages=num_pages, key=key, slot_score=slot_score,
                tail_k=tail_k, tail_score=tail_score, ape=ape,
                block_tables=block_tables, rpi=rpi, pos=pos,
                seq_lens=seq_lens, ocl=ocl, g=g)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def t0_gate(mod):
    os.environ.pop("VKERNELS_DSA_KPOOL_FORCE", None)
    assert mod._vk_dsa_kpool_active(torch.device("cpu")) is False, \
        "native gate must be OFF on CPU without FORCE"
    os.environ["VKERNELS_DSA_KPOOL_FORCE"] = "1"
    assert mod._vk_dsa_kpool_active(torch.device("cpu")) is True
    os.environ["VKERNELS_DSA_KPOOL_DISABLE"] = "1"
    assert mod._vk_dsa_kpool_active(torch.device("cpu")) is False
    os.environ.pop("VKERNELS_DSA_KPOOL_DISABLE", None)
    assert mod._VK_DSA_KPOOL_ASSEMBLE is not None, "vkernels import failed"
    print("  T0 gate logic + vkernels import: OK")


def t1_assemble(mod, c):
    pool = Pool(slots_per_page=c["ssp"])
    # -- tight: vkernels public API vs independent reference (fp32) --
    n = lambda t, dt=np.float32: np.ascontiguousarray(t.to(torch.float32).numpy(), dtype=dt)
    i32 = lambda t: np.ascontiguousarray(t.numpy(), dtype=np.int32)
    out = mod._VK_DSA_KPOOL_ASSEMBLE(
        n(c["chunk_k"]), n(c["chunk_score"]), n(c["tail_k"]), n(c["tail_score"]),
        n(c["ape"]), i32(c["rpi"]), i32(c["nft"]), i32(c["css"]), i32(c["tlb"]),
        i32(c["loc"]), slots_per_page=c["ssp"], num_pages=c["num_pages"],
        write_mask=None, out=None)
    x_ref, x_fp8 = ref_assemble(c["chunk_k"], c["chunk_score"], c["tail_k"],
                                c["tail_score"], c["ape"], c["rpi"], c["nft"],
                                c["css"], c["tlb"], c["loc"], None,
                                c["pool_size"], c["tail_size"], c["ssp"], c["num_pages"])
    torch.testing.assert_close(torch.from_numpy(out), x_ref, rtol=1e-3, atol=1e-4)
    # -- wrapper level: bf16 cache, sentinel preservation --
    buf = torch.full((c["num_pages"], c["ssp"] * HEAD), SENTINEL).to(torch.bfloat16)
    before = buf.clone()
    mod.kpool_assemble_softmax_rotate_write_cache(
        pool, buf, c["chunk_k"], c["chunk_score"], c["tail_k"], c["tail_score"],
        c["rpi"], c["nft"], c["css"], c["tlb"], c["ape"], c["loc"],
        write_mask=None, round_scale=False)
    buf3 = buf.float().view(c["num_pages"], c["ssp"], HEAD)
    before3 = before.float().view(c["num_pages"], c["ssp"], HEAD)
    written = torch.zeros(c["num_pages"], c["ssp"], dtype=torch.bool)
    for l in c["loc"].tolist():
        written[l // c["ssp"], l % c["ssp"]] = True
    torch.testing.assert_close(buf3[written], x_ref[written], rtol=1e-2, atol=2e-2)
    assert torch.equal(buf3[~written], before3[~written]), "untouched cache slots changed"
    # -- with write_mask: masked rows must not be touched --
    wm = torch.tensor([True, False, True, True, True])
    buf2 = torch.full((c["num_pages"], c["ssp"] * HEAD), SENTINEL).to(torch.bfloat16)
    before2 = buf2.clone()
    mod.kpool_assemble_softmax_rotate_write_cache(
        pool, buf2, c["chunk_k"], c["chunk_score"], c["tail_k"], c["tail_score"],
        c["rpi"], c["nft"], c["css"], c["tlb"], c["ape"], c["loc"],
        write_mask=wm, round_scale=False)
    buf2f = buf2.float().view(c["num_pages"], c["ssp"], HEAD)
    loc1 = c["loc"][1].item()
    assert torch.equal(buf2f[loc1 // c["ssp"], loc1 % c["ssp"]],
                       before2.float().view(c["num_pages"], c["ssp"], HEAD)[loc1 // c["ssp"], loc1 % c["ssp"]]), \
        "write_mask=0 row was written"
    print("  T1 assemble (public==ref tight; wrapper bf16 cache; mask; sentinel): OK")
    return x_ref, x_fp8, buf3


def t2_decode(mod, c):
    pool = Pool(slots_per_page=c["ssp"])
    n = lambda t, dt=np.float32: np.ascontiguousarray(t.to(torch.float32).numpy(), dtype=dt)
    i32 = lambda t: np.ascontiguousarray(t.numpy(), dtype=np.int32)
    tk32, ts32 = n(c["tail_k"]).copy(), n(c["tail_score"]).copy()
    out = mod._VK_DSA_KPOOL_DECODE(
        n(c["key"]), n(c["slot_score"]), tk32, ts32, n(c["ape"]),
        i32(c["block_tables"]), i32(c["rpi"]), i32(c["pos"]), i32(c["seq_lens"]),
        i32(c["ocl"]), tail_size=c["tail_size"], slots_per_page=c["ssp"],
        num_pages=c["num_pages"], out=None)
    c_ref, tk_ref, ts_ref = ref_decode(c["key"], c["slot_score"], c["tail_k"].float(),
                                       c["tail_score"].float(), c["ape"],
                                       c["block_tables"], c["rpi"], c["pos"],
                                       c["seq_lens"], c["ocl"], c["pool_size"],
                                       c["tail_size"], c["ssp"], c["num_pages"])
    torch.testing.assert_close(torch.from_numpy(out), c_ref, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(torch.from_numpy(tk32), tk_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.from_numpy(ts32), ts_ref, rtol=1e-5, atol=1e-6)
    # -- wrapper level: bf16 cache + in-place bf16 tail update --
    buf = torch.full((c["num_pages"], c["ssp"] * HEAD), SENTINEL).to(torch.bfloat16)
    before = buf.clone()
    tk = c["tail_k"].clone()
    ts = c["tail_score"].clone()
    mod.kpool_decode_update_and_maybe_write_cache(
        pool, buf, tk, ts, c["key"], c["slot_score"], c["ape"],
        c["block_tables"], c["rpi"], c["pos"], c["seq_lens"], c["ocl"],
        round_scale=False)
    buf3 = buf.float().view(c["num_pages"], c["ssp"], HEAD)
    before3 = before.float().view(c["num_pages"], c["ssp"], HEAD)
    wr = torch.zeros(c["num_pages"], c["ssp"], dtype=torch.bool)
    for r in range(c["key"].shape[0]):
        if int(c["ocl"][r]) == 0 or int(c["rpi"][r]) < 0:
            continue
        p = int(c["pos"][r])
        if not (0 <= p < int(c["seq_lens"][r])):
            continue
        if p % c["pool_size"] == c["pool_size"] - 1:
            pool_id = p // c["pool_size"]
            bt_row = min(max(pool_id // c["ssp"] * c["pool_size"], 0),
                         c["block_tables"].shape[1] - 1)
            wr[int(c["block_tables"][r, bt_row]), pool_id % c["ssp"]] = True
    torch.testing.assert_close(buf3[wr], c_ref[wr], rtol=1e-2, atol=2e-2)
    assert torch.equal(buf3[~wr], before3[~wr]), "untouched decode cache slots changed"
    torch.testing.assert_close(tk.float(), tk_ref, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(ts.float(), ts_ref, rtol=1e-2, atol=2e-2)
    # invalid rows: tail must be a no-op (sentinel in a probe row)
    print("  T2 decode (gating, substitution, tail update, clamp, sentinel): OK")
    return c_ref


def t3_dispatch_proof(mod, c, cd):
    """With all Triton kernels replaced by raising sentinels, the patched
    wrappers must still complete -- the fp8e4nv JIT path is unreachable."""
    class Boom:
        def __getitem__(self, grid):
            raise RuntimeError("TRITON LAUNCH REACHED (fp8e4nv JIT path)")

    saved = {}
    for k in ("_kpool_assemble_softmax_rotate_write_cache_kernel",
              "_kpool_decode_update_and_maybe_write_cache_kernel",
              "_kpool_softmax_rotate_write_cache_kernel",
              "_scatter_kpool_tail_updates_kernel"):
        saved[k] = getattr(mod, k)
        setattr(mod, k, Boom())
    try:
        pool = Pool(slots_per_page=c["ssp"])
        buf = torch.zeros(c["num_pages"], c["ssp"] * HEAD).to(torch.bfloat16)
        mod.kpool_assemble_softmax_rotate_write_cache(
            pool, buf, c["chunk_k"], c["chunk_score"], c["tail_k"], c["tail_score"],
            c["rpi"], c["nft"], c["css"], c["tlb"], c["ape"], c["loc"], None, False)
        pool_d = Pool(slots_per_page=cd["ssp"])
        buf_d = torch.zeros(cd["num_pages"], cd["ssp"] * HEAD).to(torch.bfloat16)
        tk, ts = cd["tail_k"].clone(), cd["tail_score"].clone()
        mod.kpool_decode_update_and_maybe_write_cache(
            pool_d, buf_d, tk, ts, cd["key"], cd["slot_score"], cd["ape"],
            cd["block_tables"], cd["rpi"], cd["pos"], cd["seq_lens"], cd["ocl"], False)
    finally:
        for k, v in saved.items():
            setattr(mod, k, v)
    print("  T3 dispatch proof (Triton sentinels raise; native path completes): OK")


def t4_negative_control(orig, c):
    """The ORIGINAL wrapper goes down the Triton launch path (raises here on
    CPU; on SM80 the actual error is the fp8e4nv ValueError of job 82822)."""
    pool = Pool(slots_per_page=c["ssp"])
    buf = torch.zeros(c["num_pages"], c["ssp"] * (HEAD + 4), dtype=torch.uint8)
    raised = None
    try:
        orig.kpool_assemble_softmax_rotate_write_cache(
            pool, buf, c["chunk_k"], c["chunk_score"], c["tail_k"], c["tail_score"],
            c["rpi"], c["nft"], c["css"], c["tlb"], c["ape"], c["loc"], None, False)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    assert raised is not None, \
        "original wrapper did NOT enter the Triton path (unexpected on CPU)"
    print(f"  T4 negative control (original enters Triton path -> {type(raised).__name__}): OK")


def t5_fp8_relationship(x_ref, x_fp8, buf3):
    """The legacy fp8 storage and the native bf16 storage approximate the SAME
    fp32 math (the Hadamard-rotated gated mean) within their respective
    quantization envelopes:
      native : single bf16 store of x_ref            -> rel err <= 2^-8
      legacy : bf16(mean) -> bf16(H(.)) -> fp8 + scale
               the two intermediate bf16 roundings propagate through the
               Hadamard mix by up to ~sqrt(128)*2^-9 ~ 2.2% of the vector
               scale, then fp8 adds <= 6.25% relative."""
    wr = x_ref.abs().sum(-1) > 0
    vec_scale = x_ref.abs().amax(dim=-1, keepdim=True).expand_as(x_ref)
    d_native = (buf3 - x_ref)[wr]
    d_legacy = (x_fp8 - x_ref)[wr]
    assert torch.all(d_native.abs() <= 2e-2 + 1e-2 * x_ref[wr].abs()), \
        "native bf16 store exceeds single-bf16-rounding envelope"
    tol_legacy = 0.08 * x_ref[wr].abs() + 0.04 * vec_scale[wr] + 1e-4
    assert torch.all(d_legacy.abs() <= tol_legacy), \
        f"fp8 dequant emulation diverges: max excess {(d_legacy - tol_legacy).max()}"
    print(f"  T5 storage equivalence (native bf16 max|err|={d_native.abs().max():.4f}, "
          f"legacy fp8 max|err|={d_legacy.abs().max():.4f}, same fp32 math): OK")


def _dequant_legacy_buf(buf, ssp):
    """Dequantize the legacy fp8+scale cache (BLOCK layout, matching the
    Triton store offsets): per page, first ssp*128 bytes = fp8 K slots,
    then ssp*4 bytes = fp32 scales. Returns ([pages, ssp, 128] fp32, scales)."""
    num_pages = buf.shape[0]
    k = (buf[:, : ssp * HEAD].contiguous().view(torch.float8_e4m3fn)
         .view(num_pages, ssp, HEAD).float())
    s = buf[:, ssp * HEAD:].contiguous().view(torch.float32).reshape(num_pages, ssp)
    return k * s.unsqueeze(-1), s


def t6_assemble_bridge(mod, c):
    pool = Pool(slots_per_page=c["ssp"])
    x_ref, x_fp8 = ref_assemble(c["chunk_k"], c["chunk_score"], c["tail_k"],
                                c["tail_score"], c["ape"], c["rpi"], c["nft"],
                                c["css"], c["tlb"], c["loc"], None,
                                c["pool_size"], c["tail_size"], c["ssp"],
                                c["num_pages"])
    written = torch.zeros(c["num_pages"], c["ssp"], dtype=torch.bool)
    for l in c["loc"].tolist():
        written[l // c["ssp"], l % c["ssp"]] = True
    # exact store-path check: quantize the vkernels fp32 output rows (public
    # API, same scratch the bridge reads) and expect IDENTICAL bytes -- the
    # bridge must be a bit-exact requant+store of its own compute input.
    n = lambda t, dt=np.float32: np.ascontiguousarray(t.to(torch.float32).numpy(), dtype=dt)
    i32 = lambda t: np.ascontiguousarray(t.numpy(), dtype=np.int32)
    for round_scale in (False, True):
        scratch = np.zeros(c["num_pages"] * c["ssp"] * HEAD, dtype=np.float32)
        mod._VK_DSA_KPOOL_ASSEMBLE(
            n(c["chunk_k"]), n(c["chunk_score"]), n(c["tail_k"]), n(c["tail_score"]),
            n(c["ape"]), i32(c["rpi"]), i32(c["nft"]), i32(c["css"]), i32(c["tlb"]),
            i32(c["loc"]), slots_per_page=c["ssp"], num_pages=c["num_pages"],
            write_mask=None, out=scratch)
        loc_l = c["loc"].to(torch.int64).numpy()
        pages_l, sips_l = loc_l // c["ssp"], loc_l % c["ssp"]
        x_vk = scratch.reshape(c["num_pages"], c["ssp"], HEAD)[pages_l, sips_l]
        q_exp, s_exp = _quant_bytes(torch.from_numpy(x_vk), round_scale)
        buf = torch.zeros(c["num_pages"], c["ssp"] * (HEAD + 4), dtype=torch.uint8)
        mod.kpool_assemble_softmax_rotate_write_cache(
            pool, buf, c["chunk_k"], c["chunk_score"], c["tail_k"], c["tail_score"],
            c["rpi"], c["nft"], c["css"], c["tlb"], c["ape"], c["loc"],
            write_mask=None, round_scale=round_scale)
        deq, scales = _dequant_legacy_buf(buf, c["ssp"])
        # (a) bit-exact store: K bytes and scale bytes == the emulation
        k_got = (buf[:, : c["ssp"] * HEAD].contiguous().view(torch.float8_e4m3fn)
                 .view(c["num_pages"], c["ssp"], HEAD)
                 [torch.from_numpy(pages_l), torch.from_numpy(sips_l)])
        assert torch.equal(k_got.view(torch.uint8), q_exp.view(torch.uint8)), \
            "bridge K bytes != emulation of the vkernels fp32 rows"
        s_exp_b = s_exp.numpy().view(np.uint8).reshape(-1, 4)
        s_got = scales[pages_l, sips_l].numpy().view(np.uint8).reshape(-1, 4)
        assert (s_got == s_exp_b).all(), "bridge scale bytes != emulation"
        # (b) semantic anchor: dequantized bridge ~ reference fp8. NOT
        # bit-equal: independent fp8 rounding chains over slightly different
        # inputs (vkernels fp32 vs the ref's double-bf16, T1 tol). One e4m3
        # step is 6.25% rel of the value's binade -- up to ~12.5% rel just
        # below a power-of-2 edge -- plus the ~0.1% scale-chain offset.
        err = (deq[written] - x_fp8[written]).abs()
        bound = 0.14 * x_fp8[written].abs() + 4e-3
        assert (err <= bound).all(), \
            f"bridge fp8 dequant exceeds fp8-ulp+input-diff bound " \
            f"(round_scale={round_scale}, max excess {(err - bound).max()})"
        assert (deq[~written] == 0).all(), "untouched legacy slots changed"
        # (c) scale sanity: enclosing pow2 or absmax/448 of the dequant row
        absmax = deq[written].abs().amax(dim=-1).clamp_min(1e-9)
        if round_scale:
            assert (scales[written] >= absmax / 448.0 * 0.999).all() and \
                (scales[written] < absmax / 448.0 * 2.01).all(), "pow2 scale not enclosing"
        else:
            torch.testing.assert_close(
                scales[written], absmax / 448.0, rtol=0.07, atol=1e-9)
    print("  T6 assemble bridge (bit-exact fp8 store of vk output; dequant "
          "within fp8 ulp of ref; both round_scale): OK")


def t7_decode_bridge(mod, cd):
    pool = Pool(slots_per_page=cd["ssp"])
    # native (bf16 cache) run = T2-validated ground truth for the vkernels math
    buf_bf16 = torch.zeros(cd["num_pages"], cd["ssp"] * HEAD).to(torch.bfloat16)
    tk = cd["tail_k"].clone()
    ts = cd["tail_score"].clone()
    mod.kpool_decode_update_and_maybe_write_cache(
        pool, buf_bf16, tk, ts, cd["key"], cd["slot_score"], cd["ape"],
        cd["block_tables"], cd["rpi"], cd["pos"], cd["seq_lens"], cd["ocl"],
        round_scale=False)
    # bridge (uint8 legacy cache) run
    buf_u8 = torch.zeros(cd["num_pages"], cd["ssp"] * (HEAD + 4), dtype=torch.uint8)
    tk2 = cd["tail_k"].clone()
    ts2 = cd["tail_score"].clone()
    mod.kpool_decode_update_and_maybe_write_cache(
        pool, buf_u8, tk2, ts2, cd["key"], cd["slot_score"], cd["ape"],
        cd["block_tables"], cd["rpi"], cd["pos"], cd["seq_lens"], cd["ocl"],
        round_scale=False)
    torch.testing.assert_close(tk2, tk, rtol=1e-6, atol=1e-6, msg="tails diverged")
    torch.testing.assert_close(ts2, ts, rtol=1e-6, atol=1e-6, msg="tail scores diverged")
    deq, scales = _dequant_legacy_buf(buf_u8, cd["ssp"])
    native = buf_bf16.float().view(cd["num_pages"], cd["ssp"], HEAD)
    wr = buf_bf16.abs().view(cd["num_pages"], cd["ssp"], -1).sum(-1) > 0
    # the bridge quantizes the fp32 vkernels output; the native path stores it
    # as bf16 first -- so expect fp8-ulp agreement with the bf16 values, not
    # bit equality (bf16 rounding is up to 2^-9 rel; fp8 adds 2^-4 rel).
    err = (deq[wr] - native[wr]).abs()
    bound = 0.07 * native[wr].abs() + 1.5e-3
    assert (err <= bound).all(), \
        f"bridge dequant != native bf16 values beyond fp8+bf16 rounding " \
        f"(max excess {(err - bound).max()})"
    # scale self-consistency: absmax(dequant row) / 448
    absmax = deq[wr].abs().amax(dim=-1).clamp_min(1e-9)
    torch.testing.assert_close(scales[wr], absmax / 448.0, rtol=0.07, atol=1e-9)
    print("  T7 decode bridge (fp8-ulp agreement with T2-validated native; "
          "tails identical): OK")


def t8_shim_logits():
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "sm80_sitecustomize", RECIPE / "patched_sources" / "sitecustomize.py")
    sc = _ilu.module_from_spec(spec)
    spec.loader.exec_module(sc)  # _install() no-ops without CUDA
    g = torch.Generator().manual_seed(7)

    # ---- paged ----
    B, D, P, R, H, L = 64, 128, 3, 4, 8, 2
    S = L * B
    q = torch.randn(R, 1, H, D, generator=g).to(torch.float8_e4m3fn)
    w = torch.randn(R, H, generator=g).abs() + 0.5
    k = torch.randn(P, B, D, generator=g).clamp(-3, 3).to(torch.float8_e4m3fn)
    ks = torch.rand(P, B, generator=g) + 0.01
    cache = torch.zeros(P, B * (D + 4), dtype=torch.uint8)
    cache[:, : B * D] = k.view(torch.uint8).reshape(P, B * D)
    cache[:, B * D:] = torch.from_numpy(
        np.ascontiguousarray(ks.numpy()).view(np.uint8).reshape(P, B * 4))
    table = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 0]], dtype=torch.int32)
    out = sc._sm80_fp8_paged_mqa_logits(
        q, cache, w, None, table, None, S, clean_logits=False)
    assert out.shape == (R, S)
    qf = q.reshape(R, H, D).float()
    for r in range(R):
        for i in range(L):
            pg = int(table[r, i])
            for j in range(B):
                want = 0.0
                for h in range(H):
                    want += max(float(qf[r, h] @ k[pg, j].float()), 0.0) * float(w[r, h])
                want *= float(ks[pg, j])
                got = float(out[r, i * B + j])
                assert abs(got - want) < 2e-2 * max(abs(want), 1e-3) + 1e-4, \
                    f"paged ({r},{i},{j}): {got} != {want}"
    # unwritten cols (S covers L*B exactly here -> use a bigger S)
    out2 = sc._sm80_fp8_paged_mqa_logits(
        q, cache, w, None, table, None, S + 64, clean_logits=False)
    assert (out2[:, S:] == -1e30).all(), "unwritten paged cols must be -1e30"

    # ---- ragged ----
    N = 100
    k2 = torch.randn(N, D, generator=g).clamp(-3, 3).to(torch.float8_e4m3fn)
    ks2 = torch.rand(N, generator=g) + 0.01
    q2 = torch.randn(R, H, D, generator=g).to(torch.float8_e4m3fn)
    w2 = torch.randn(R, H, generator=g).abs() + 0.5
    ks_per = torch.tensor([0, 10, 20, 30], dtype=torch.int32)
    ke_per = torch.tensor([100, 60, 70, 35], dtype=torch.int32)
    out3 = sc._sm80_fp8_mqa_logits(
        q2, (k2, ks2), w2, ks_per, ke_per, clean_logits=True)
    q2f = q2.float()
    k2f = k2.float()
    for r in range(R):
        for n in range(N):
            want = 0.0
            for h in range(H):
                want += max(float(q2f[r, h] @ k2f[n]), 0.0) * float(w2[r, h])
            want *= float(ks2[n])
            if not (ks_per[r] <= n < ke_per[r]):
                want = 0.0
            got = float(out3[r, n])
            assert abs(got - want) < 2e-2 * max(abs(want), 1e-3) + 1e-4, \
                f"ragged ({r},{n}): {got} != {want}"
    print("  T8 deep_gemm shim (paged + ragged torch fallbacks vs loop ref): OK")


def t9_act_quant_sm80():
    """The patched triton_kernel.act_quant SM80 fallback vs an independent
    emulation of _act_quant_kernel (the decode wall of jobs 82822/83091:
    forward_absorb_prepare -> act_quant -> fp8e4nv ValueError). Byte-equal
    fp8 payloads and scales for both round_scale variants, exact shape/dtype
    contract, zero-guard, clamp and contiguity behavior."""

    def _act_quant_ref(x_r, round_scale=False):
        # op-for-op from the Triton kernel source: fp8_max_inv = 1/448, the
        # kernel MULTIPLIES by the reciprocal (never divides), amax clamp
        # 1e-4, round_scale = pow2 ceiling of (amax * fp8_max_inv).
        fp8_max_inv = 1.0 / 448.0
        amax = torch.clamp(x_r.abs().amax(dim=-1, keepdim=True), min=1e-4)
        if round_scale:
            scale = torch.exp2(torch.ceil(torch.log2(amax * fp8_max_inv)))
        else:
            scale = amax * fp8_max_inv
        q = torch.clamp(x_r / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
        return q, scale

    mod = load_module("triton_kernel_sm80_patched", TRITON_KERNEL_PATCHED)
    src = TRITON_KERNEL_PATCHED.read_text()
    assert "SGLANG_SM80_ACT_QUANT_DISABLE" in src, "SM80 veto env missing"
    assert "get_device_capability" in src, "arch gate missing"
    assert "_act_quant_sm80(x, block_size, scale_fmt)" in src, "dispatch missing"

    g = torch.Generator().manual_seed(11)
    for shape in [(5, 3, 128), (7, 128), (1, 4, 256)]:  # decode 3-D, 2-D, 2 blocks
        x = (torch.randn(*shape, generator=g) * 3.0).contiguous()
        x_r = x.view(*shape[:-1], -1, 128)
        for fmt in (None, "ue8m0"):
            y, s = mod._act_quant_sm80(x, 128, fmt)
            assert y.dtype == torch.float8_e4m3fn and y.shape == x.shape, (shape, fmt)
            assert s.dtype == torch.float32, (shape, fmt)
            assert s.shape == (*shape[:-1], shape[-1] // 128), (shape, fmt)
            q_ref, s_ref = _act_quant_ref(x_r.float(), round_scale=fmt is not None)
            assert torch.equal(y.view_as(x_r).view(torch.uint8),
                               q_ref.view(torch.uint8)), (shape, fmt)
            assert torch.equal(s, s_ref.reshape(s.shape)), (shape, fmt)
            deq = (y.view_as(x_r).float() * s.unsqueeze(-1)).view(x.shape)
            rel = ((deq - x).abs() / x.abs().clamp(min=1e-3)).max().item()
            assert rel < 0.08, (shape, fmt, rel)  # half-ulp e4m3 ~ 2^-4 for normals

    # zero guard: all-zero rows keep a positive scale (kernel clamps 1e-4)
    x0 = torch.zeros(2, 128)
    x0[0, 0] = 1.0
    y0, s0 = mod._act_quant_sm80(x0, 128, None)
    assert (s0 > 0).all() and float(y0[1].float().abs().max()) == 0.0

    # clamp: |y| <= 448 by construction (amax/scale == 448 for scale_fmt=None)
    big = torch.randn(3, 2, 128, generator=g) * 100.0
    yb, _ = mod._act_quant_sm80(big, 128, None)
    assert yb.float().abs().max().item() <= 448.0

    # contiguity contract matches the original kernel
    try:
        mod._act_quant_sm80(torch.randn(4, 256, generator=g).t(), 128, None)
        raise SystemExit("non-contiguous input must raise")
    except AssertionError:
        pass

    print("  T9 act_quant SM80 (torch fallback == Triton-kernel emulation, "
          "bytes+scale, 3 shapes x 2 scale_fmts, guards): OK")


def main():
    os.environ["VKERNELS_DSA_KPOOL_FORCE"] = "1"
    print(f"cookbook root: {COOKBOOK}")
    patched = load_module("kpool_fp8_index_vk60_patched", PATCHED)
    original = load_module("kpool_fp8_index_original", ORIGINAL)
    assert PATCHED.read_text() != ORIGINAL.read_text()

    # H sanity: involutory, normalized (matches sglang's butterfly + 1/sqrt(128))
    torch.testing.assert_close(H128 @ H128.T, torch.eye(HEAD), atol=1e-5, rtol=1e-5)
    print(f"H128 check: involutory, 1/sqrt(128)={1 / math.sqrt(128):.17f} "
          f"(sglang constant 0.08838834764831845)")

    c = make_assemble_case(0)
    cd = make_decode_case(1)
    t0_gate(patched)
    x_ref, x_fp8, buf3 = t1_assemble(patched, c)
    t2_decode(patched, cd)
    t3_dispatch_proof(patched, c, cd)
    t4_negative_control(original, c)
    t5_fp8_relationship(x_ref, x_fp8, buf3)
    t6_assemble_bridge(patched, c)
    t7_decode_bridge(patched, cd)
    t8_shim_logits()
    t9_act_quant_sm80()
    print("\nALL TESTS PASSED -- vkernels#60 dsa_kpool wiring (native + legacy-"
          "layout bridge), the SM80 deep_gemm shim and the SM80 act_quant "
          "fallback are semantically sound.")


if __name__ == "__main__":
    main()
