#!/usr/bin/env python3
"""On-GPU validation + benchmark: vkernels#60 dsa_kpool DEVICE path (SM80).

Runs on an A100 (compute capability 8.x) with torch+cuda -- on bristen
inside the sglang container, or any CUDA torch env. Complements the
CPU-only tests_local/test_dsa_kpool_wiring.py (T0-T10): here the patched
wrappers dispatch to the NATIVE device kernels (vk_hip_dsa_kpool_* via the
vkernels.dsa_kpool_device ctypes loader, VKERNELS_LIB env) and are checked
against the torch bridge (the job-83152 serving path) and the fp32
Hadamard reference, then both paths are timed at serving shapes.

Checks
  D1  assemble: device-fp8 cache dequantizes onto the torch-bridge cache
      within one fp8 ulp (+ input diff), scales agree (rtol 2e-2), untouched
      slots are byte-identical, and both sit on the fp32 reference within
      the native-storage envelope.
  D2  decode: same cache contract; the IN-PLACE live tails are EXACTLY the
      torch bridge's tails (bf16 copies of the same inputs).
  D3  dispatch: VKERNELS_DSA_KPOOL_DEVICE=1 really takes the one-launch
      path (stderr note + the Triton sentinel stays unreachable), =0 pins
      the torch bridge.

Bench (per-call wall time, cuda-event timed, median of repeats)
  assemble n_pools in {1, 16, 64, 256} (ps=4, tail=64, ssp=8)
  decode   batch   in {1, 32, 128, 256}
  device kernel vs torch bridge (the 83152 serving path); GB/s and the
  step-level delta against the job-83152 profile (~157 eager launches).

Run:
  VKPYTHON=/capstor/.../vkernels-python VKERNELS_LIB=.../libvkernels_c.so \
      python3 tests_local/test_dsa_kpool_device_path.py
"""

import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import torch

RECIPE = Path(__file__).resolve().parents[1]
PATCHED = (
    RECIPE / "patched_sources" / "sglang" / "srt" / "layers" / "attention"
    / "dsa" / "kpool_fp8_index.py"
)

# vkernels python package: $VKPYTHON first (the deploy snapshot), else the
# cookbook third_party submodule, else whatever is already importable.
_VK = os.environ.get("VKPYTHON")
if _VK:
    sys.path.insert(0, str(Path(_VK)))
else:
    _sub = RECIPE.parents[3] / "third_party" / "vkernels" / "src" / "python"
    if _sub.is_dir():
        sys.path.insert(0, str(_sub))

HEAD = 128
DEV = "cuda"


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


_H128 = None


def h128():
    global _H128
    if _H128 is None:
        i = torch.arange(HEAD, device=DEV).unsqueeze(1)
        j = torch.arange(HEAD, device=DEV).unsqueeze(0)
        ij = i & j
        pc = torch.zeros_like(ij)
        for bit in range(7):
            pc += (ij >> bit) & 1
        signs = 1.0 - 2.0 * (pc % 2).float()
        _H128 = signs / math.sqrt(HEAD)
    return _H128


def make_assemble_case(seed, n_pools, pool_size=4, tail_size=64, ssp=8,
                       num_pages=512, n_chunks=2048, n_reqs=8):
    g = torch.Generator(device=DEV).manual_seed(seed)
    chunk_k = (torch.randn(n_chunks, HEAD, generator=g, device=DEV) * 0.5).to(
        torch.bfloat16
    )
    chunk_score = (torch.randn(n_chunks, HEAD, generator=g, device=DEV)).to(
        torch.bfloat16
    )
    tail_k = (torch.randn(n_reqs, tail_size, HEAD, generator=g, device=DEV)
              * 0.5).to(torch.bfloat16)
    tail_score = torch.randn(n_reqs, tail_size, HEAD, generator=g,
                             device=DEV).to(torch.bfloat16)
    ape = torch.randn(pool_size, HEAD, generator=g, device=DEV,
                      dtype=torch.float32) * 0.1
    rpi = torch.randint(0, n_reqs, (n_pools,), generator=g, device=DEV,
                        dtype=torch.int64)
    nft = torch.randint(0, pool_size + 1, (n_pools,), generator=g,
                        device=DEV, dtype=torch.int64)
    css = torch.randint(0, max(n_chunks - pool_size, 1), (n_pools,),
                        generator=g, device=DEV, dtype=torch.int64)
    tlb = torch.randint(0, tail_size, (n_pools,), generator=g, device=DEV,
                        dtype=torch.int64)
    loc = torch.randperm(num_pages * ssp, generator=g, device=DEV)[
        :n_pools
    ].to(torch.int64)
    mask = torch.ones(n_pools, dtype=torch.bool, device=DEV)
    mask[::7] = False  # exercise the write_mask rows too
    return dict(chunk_k=chunk_k, chunk_score=chunk_score, tail_k=tail_k,
                tail_score=tail_score, ape=ape, rpi=rpi, nft=nft, css=css,
                tlb=tlb, loc=loc, mask=mask, pool_size=pool_size,
                tail_size=tail_size, ssp=ssp, num_pages=num_pages,
                n_chunks=n_chunks, n_reqs=n_reqs)


def make_decode_case(seed, batch, pool_size=4, tail_size=64, ssp=8,
                     num_pages=512, btc=64, n_reqs=8):
    g = torch.Generator(device=DEV).manual_seed(seed)
    key = (torch.randn(batch, HEAD, generator=g, device=DEV) * 0.5).to(
        torch.bfloat16
    )
    slot_score = torch.randn(batch, HEAD, generator=g, device=DEV).to(
        torch.bfloat16
    )
    tail_k = (torch.randn(n_reqs, tail_size, HEAD, generator=g, device=DEV)
              * 0.5).to(torch.bfloat16)
    tail_score = torch.randn(n_reqs, tail_size, HEAD, generator=g,
                             device=DEV).to(torch.bfloat16)
    ape = torch.randn(pool_size, HEAD, generator=g, device=DEV,
                      dtype=torch.float32) * 0.1
    bt = torch.randint(0, num_pages, (batch, btc), generator=g, device=DEV,
                       dtype=torch.int64)
    rpi = torch.randint(0, n_reqs, (batch,), generator=g, device=DEV,
                        dtype=torch.int64)
    # pool-complete positions for ~1/4 of the rows, valid everywhere
    pos = torch.randint(0, 4 * pool_size, (batch,), generator=g, device=DEV,
                        dtype=torch.int64) * pool_size + (pool_size - 1)
    pos[::3] -= 1  # some rows not pool-complete
    pos[::5] = -1  # some invalid rows
    seqlen = (pos + 1).clamp(min=1)
    seqlen[::7] += 8  # some rows past their seq_len (invalid)
    ocl = torch.randint(1, num_pages * ssp, (batch,), generator=g, device=DEV,
                        dtype=torch.int64)
    ocl[::11] = 0  # some no-write rows
    return dict(key=key, slot_score=slot_score, tail_k=tail_k,
                tail_score=tail_score, ape=ape, bt=bt, rpi=rpi, pos=pos,
                seqlen=seqlen, ocl=ocl, pool_size=pool_size,
                tail_size=tail_size, ssp=ssp, num_pages=num_pages, btc=btc,
                n_reqs=n_reqs)


def ref_rotate(src_k, src_s, ape_row):
    scores = src_s + ape_row
    mx = scores.amax(dim=1, keepdim=True)
    p = torch.exp(scores - mx)
    mean = (p * src_k).sum(dim=1) / p.sum(dim=1)
    return mean @ h128().T


def dequant(buf):
    """[num_pages, ssp*(132)] uint8 -> (fp32 K [.., 128], scale [..]) rows."""
    n = buf.shape[0]
    ssp = buf.shape[1] // (HEAD + 4)
    k = buf.view(torch.float8_e4m3fn)[
        :, : ssp * HEAD
    ].reshape(n * ssp, HEAD).float()
    s = buf.view(torch.float32)[:, ssp * 32:].reshape(n * ssp)
    return k, s


def run_assemble(mod, c, cache):
    mod.kpool_assemble_softmax_rotate_write_cache(
        pool=Pool(slots_per_page=c["ssp"]),
        buf=cache,
        chunk_k=c["chunk_k"],
        chunk_score=c["chunk_score"],
        tail_k=c["tail_k"],
        tail_score=c["tail_score"],
        req_pool_idx=c["rpi"],
        n_from_tail=c["nft"],
        chunk_src_start=c["css"],
        tail_logical_base=c["tlb"],
        ape=c["ape"],
        loc=c["loc"],
        write_mask=c["mask"],
        round_scale=False,
    )


def run_decode(mod, c, cache, tail_k, tail_score):
    mod.kpool_decode_update_and_maybe_write_cache(
        pool=Pool(slots_per_page=c["ssp"]),
        buf=cache,
        tail_k=tail_k,
        tail_score=tail_score,
        key=c["key"],
        slot_score=c["slot_score"],
        ape=c["ape"],
        block_tables=c["bt"],
        req_pool_indices=c["rpi"],
        positions=c["pos"],
        seq_lens=c["seqlen"],
        out_cache_loc=c["ocl"],
        round_scale=False,
    )


def fresh_cache(num_pages, ssp):
    return torch.zeros(num_pages, ssp * (HEAD + 4), dtype=torch.uint8,
                       device=DEV)


def d1_assemble(mod):
    c = make_assemble_case(1234, n_pools=97)
    ssp, np_ = c["ssp"], c["num_pages"]
    # torch bridge
    os.environ["VKERNELS_DSA_KPOOL_DEVICE"] = "0"
    cache_t = fresh_cache(np_, ssp)
    run_assemble(mod, c, cache_t)
    torch.cuda.synchronize()
    # device path
    os.environ["VKERNELS_DSA_KPOOL_DEVICE"] = "1"
    cache_d = fresh_cache(np_, ssp)
    run_assemble(mod, c, cache_d)
    torch.cuda.synchronize()
    del os.environ["VKERNELS_DSA_KPOOL_DEVICE"]
    touched = torch.zeros(np_ * ssp, dtype=torch.bool, device=DEV)
    touched[c["loc"][c["mask"]].long()] = True
    # untouched slots byte-identical
    assert torch.equal(cache_t[~touched], cache_d[~touched]), "untouched delta"
    kt, st = dequant(cache_t)
    kd, sd = dequant(cache_d)
    # scale agreement (both derive from near-identical fp32 absmax)
    serr = ((st - sd).abs() / st.clamp(min=1e-12))[touched]
    # dequantized K within one fp8 ulp (2^-9 relative ~ 2e-3) + input diff
    kerr = (kt - kd).abs()[touched]
    bound = 0.14 * kt[touched].abs() + 4e-3
    assert (serr.max().item() < 2e-2) and (kerr <= bound).all(), (
        f"assemble device-vs-torch: scale {serr.max().item():.2e} "
        f"K {kerr.max().item():.3e} vs bound {bound.max().item():.3e}"
    )
    # both vs the fp32 reference (row 0 = first written row)
    r = 0
    loc_r = c["loc"][c["mask"]].long()[r]
    nft0 = int(c["nft"][c["mask"]][r])
    ps = c["pool_size"]
    tail0 = int(c["tlb"][c["mask"]][r])
    req0 = int(c["rpi"][c["mask"]][r])
    css0 = int(c["css"][c["mask"]][r])
    idx = torch.arange(ps, device=DEV)
    src_k = torch.where(
        (idx < nft0).unsqueeze(1),
        c["tail_k"][req0, (tail0 + idx) % c["tail_size"]].float(),
        c["chunk_k"][(css0 + (idx - nft0).clamp(min=0)).clamp(
            max=c["n_chunks"] - 1)].float(),
    )
    src_s = torch.where(
        (idx < nft0).unsqueeze(1),
        c["tail_score"][req0, (tail0 + idx) % c["tail_size"]].float(),
        c["chunk_score"][(css0 + (idx - nft0).clamp(min=0)).clamp(
            max=c["n_chunks"] - 1)].float(),
    )
    x_ref = ref_rotate(src_k, src_s, c["ape"].float()[0])
    row_t = kt[loc_r]
    row_d = kd[loc_r]
    rt = (row_t - x_ref).abs().max().item()
    rd = (row_d - x_ref).abs().max().item()
    print(f"  D1 assemble: untouched exact; scale rel {serr.max().item():.2e}; "
          f"K dev-vs-torch {kerr.max().item():.3e}; vs-fp32-ref torch {rt:.4f} "
          f"device {rd:.4f} (envelope ~0.08)")
    assert rd < 0.10, f"device storage off the fp32 reference: {rd}"


def d2_decode(mod):
    c = make_decode_case(4321, batch=64)
    ssp, np_ = c["ssp"], c["num_pages"]
    os.environ["VKERNELS_DSA_KPOOL_DEVICE"] = "0"
    cache_t = fresh_cache(np_, ssp)
    tk_t, ts_t = c["tail_k"].clone(), c["tail_score"].clone()
    run_decode(mod, c, cache_t, tk_t, ts_t)
    torch.cuda.synchronize()
    os.environ["VKERNELS_DSA_KPOOL_DEVICE"] = "1"
    cache_d = fresh_cache(np_, ssp)
    tk_d, ts_d = c["tail_k"].clone(), c["tail_score"].clone()
    run_decode(mod, c, cache_d, tk_d, ts_d)
    torch.cuda.synchronize()
    del os.environ["VKERNELS_DSA_KPOOL_DEVICE"]
    # live tails EXACT (bf16 copies of the same inputs)
    assert torch.equal(tk_t, tk_d) and torch.equal(ts_t, ts_d), "tail delta"
    valid = (
        (c["rpi"] >= 0)
        & (c["rpi"] < c["n_reqs"])
        & (c["ocl"] != 0)
        & (c["pos"] >= 0)
        & (c["pos"] < c["seqlen"])
    )
    touched = torch.zeros(np_ * ssp, dtype=torch.bool, device=DEV)
    ps = c["pool_size"]
    written = valid & ((c["pos"] % ps) == ps - 1)
    if written.any():
        pid = c["pos"].clamp(min=0)[written] // ps
        tpr = ((pid // ssp) * ps).clamp(0, c["btc"] - 1)
        pages = c["bt"][written.nonzero(as_tuple=True)[0], tpr].long()
        slots = pages * ssp + (pid % ssp)
        touched[slots] = True
    assert torch.equal(cache_t[~touched], cache_d[~touched]), "untouched delta"
    kt, st = dequant(cache_t)
    kd, sd = dequant(cache_d)
    serr = ((st - sd).abs() / st.clamp(min=1e-12))[touched]
    kerr = (kt - kd).abs()[touched]
    bound = 0.14 * kt[touched].abs() + 4e-3
    assert (serr.max().item() < 2e-2) and (kerr <= bound).all(), (
        f"decode device-vs-torch: scale {serr.max().item():.2e} "
        f"K {kerr.max().item():.3e}"
    )
    print(f"  D2 decode: tails exact; untouched exact; scale rel "
          f"{serr.max().item():.2e}; K dev-vs-torch {kerr.max().item():.3e}")


def d3_dispatch(mod):
    import io
    import contextlib
    c = make_assemble_case(77, n_pools=8)
    buf = fresh_cache(c["num_pages"], c["ssp"])
    os.environ["VKERNELS_DSA_KPOOL_DEVICE"] = "1"
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        run_assemble(mod, c, buf)
    torch.cuda.synchronize()
    del os.environ["VKERNELS_DSA_KPOOL_DEVICE"]
    assert "device C ABI active" in err.getvalue(), "device note missing"
    assert "Triton" not in err.getvalue()
    print("  D3 dispatch: DEVICE=1 notes the C ABI and completes "
          "(fp8e4nv Triton unreachable)")


def bench(fn, warmup=20, iters=200, repeats=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = None
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) * 1000.0 / iters
        best = us if best is None else min(best, us)
    return best


def bench_paths(mod):
    print("\n  bench (us/call, min of 5x200, cuda events, bs-1 stream):")
    print("  path      |  assemble n_pools 1/16/64/256   |  decode batch 1/32/128/256")
    rows = {}
    for dev_mode, label in (("1", "device"), ("0", "torch")):
        os.environ["VKERNELS_DSA_KPOOL_DEVICE"] = dev_mode
        a = []
        d = []
        for npools in (1, 16, 64, 256):
            c = make_assemble_case(9, n_pools=npools)
            cache = fresh_cache(c["num_pages"], c["ssp"])
            us = bench(lambda: run_assemble(mod, c, cache))
            a.append(us)
        for b in (1, 32, 128, 256):
            c = make_decode_case(9, batch=b)
            cache = fresh_cache(c["num_pages"], c["ssp"])
            tk, ts = c["tail_k"].clone(), c["tail_score"].clone()
            us = bench(lambda: run_decode(mod, c, cache, tk, ts))
            d.append(us)
        rows[label] = (a, d)
        del os.environ["VKERNELS_DSA_KPOOL_DEVICE"]
    for label, (a, d) in rows.items():
        print(f"  {label:9} | " + " ".join(f"{x:8.1f}" for x in a)
              + "   |  " + " ".join(f"{x:8.1f}" for x in d))
    ga, gd = rows["device"]
    ta, td = rows["torch"]
    print("  speedup   | " + " ".join(f"{t/g:8.1f}" for t, g in zip(ta, ga))
          + "   |  " + " ".join(f"{t/g:8.1f}" for t, g in zip(td, gd)))


def main():
    assert torch.cuda.is_available(), "CUDA required"
    major, _ = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"device: {name} (sm_{major}0)")
    assert major == 8, "this validation targets SM80 (A100)"
    mod = load_module("kpool_fp8_index_device_test", PATCHED)
    from vkernels import dsa_kpool_device as dev

    lib = os.environ.get("VKERNELS_LIB")
    print(f"vkernels device lib: {lib or dev.find_libvkernels()}")
    assert dev.available(), "device library not loadable (VKERNELS_LIB?)"
    print("D1/D2/D3 + bench:")
    d3_dispatch(mod)
    d1_assemble(mod)
    d2_decode(mod)
    bench_paths(mod)
    print("\nALL DEVICE-PATH CHECKS PASSED")


if __name__ == "__main__":
    main()
