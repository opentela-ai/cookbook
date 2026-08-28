#!/usr/bin/env python3
"""Verify the kpool-tail FA3-safe patch (correct class: DeepseekSparseAttnBackend).
Also checks whether the MTP/multistep subclasses inherit _resolve_kpool_tail_backend
from the parent, so a single patch covers GLM-5.3's num_nextn_predict_layers=1 path."""
import importlib, torch

def _importable(name):
    try:
        importlib.import_module(name); return True
    except Exception:
        return False

_fa3_ok = _importable("sgl_kernel.flash_ops")
print(f"_fa3_ok (sgl_kernel.flash_ops importable) = {_fa3_ok}")

from sglang.srt.layers.attention import dsa_backend as _db
CLS = _db.DeepseekSparseAttnBackend
print(f"\nTarget class = {CLS.__name__}  (was wrongly 'DsaBackend' in patch v1)")
print(f"  _resolve_kpool_tail_backend original = {CLS._resolve_kpool_tail_backend!r}")

# Do the MTP / multistep backends inherit it (so one patch covers GLM-5.3 MTP)?
for sub in ("DeepseekSparseAttnMultiStepBackend",
            "DeepseekSparseAttnBackendMTPPrecomputeMixin"):
    s = getattr(_db, sub, None)
    if s is None:
        print(f"  {sub}: NOT in module"); continue
    own = "_resolve_kpool_tail_backend" in s.__dict__
    via = getattr(s, "_resolve_kpool_tail_backend", None)
    via_src = via.__qualname__ if via else "?"
    print(f"  {sub}: own={own}  resolves via {via_src}")

# Apply the EXACT patch from sitecustomize.py (corrected class name).
def _resolve_kpool_tail_backend_fa3safe(self, topk_indices, dsa_impl):
    if (topk_indices is None or self.dsa_index_kpool <= 1
            or dsa_impl != "flashmla_sparse"):
        return dsa_impl
    target = ("trtllm" if self.device_sm_major >= 10 else
              "fa3" if self.device_sm_major == 9 else dsa_impl)
    if target == dsa_impl:
        return dsa_impl
    if target == "fa3" and not _fa3_ok:
        return dsa_impl
    if target == "trtllm" and not _importable("tensorrt_llm"):
        return dsa_impl
    return target

CLS._resolve_kpool_tail_backend = _resolve_kpool_tail_backend_fa3safe
# Verify subclasses now see the patched method (inheritance).
for sub in ("DeepseekSparseAttnBackendMultiStepBackend"
            if False else "DeepseekSparseAttnMultiStepBackend",):
    s = getattr(_db, sub, None)
    if s: print(f"  after patch, {sub}._resolve_kpool_tail_backend = {getattr(s,'_resolve_kpool_tail_backend',None)!r}")

class _Fake:
    def __init__(self, sm, kpool):
        self.device_sm_major, self.dsa_index_kpool = sm, kpool

topk = torch.zeros(1, dtype=torch.int32)
cases = [
    ("sm9 kpool2 flashmla_sparse (THE CRASH)", _Fake(9, 2), "flashmla_sparse", topk),
    ("sm9 kpool2 fa3 (explicit wrong)",        _Fake(9, 2), "fa3",            topk),
    ("sm9 kpool1 flashmla_sparse (guard)",     _Fake(9, 1), "flashmla_sparse", topk),
    ("sm9 kpool2 none-topk (guard)",           _Fake(9, 2), "flashmla_sparse", None),
    ("sm10 kpool2 flashmla_sparse",            _Fake(10,2), "flashmla_sparse", topk),
    ("sm8 kpool2 flashmla_sparse (no ovrd)",   _Fake(8, 2), "flashmla_sparse", topk),
]
print("\n=== patched method behavior ===")
crash_case_ok = None
for label, fake, impl, tk in cases:
    got = _resolve_kpool_tail_backend_fa3safe(fake, tk, impl)
    if "THE CRASH" in label:
        crash_case_ok = (got == "flashmla_sparse")
    print(f"  {label:44s} -> {got}")

print(f"\nVERDICT: {'PASS' if crash_case_ok else 'FAIL'} — crash case "
      f"returns {'flashmla_sparse (no FA3 crash)' if crash_case_ok else 'fa3 (WOULD CRASH)'}")
print(f"FA3 present on this arch: {_fa3_ok} (perf override would be ACTIVE here if True)")
