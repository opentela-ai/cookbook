"""Self-contained correctness test for vkernels_dsa_topk.fast_kpool_topk_transform_fused.

Exercises the documented contract of kpool_topk_transform.cuh:249-310 on CPU
(the torch fallback is device-agnostic). Each case hand-computes the expected
dst_token_indices and asserts equality. Run inside the sglang-rocm container
(``srun --overlap --environment=sglang-rocm python3 test_vkernels_dsa_topk.py``).
"""
import torch
from vkernels_dsa_topk import fast_kpool_topk_transform_fused


def _case_identity():
    # length(=2) <= group_topk(=4): LINEAR. No page_table, no offset -> identity.
    # pool_size=2, topk=8 -> group_topk=4, out_cols=8.
    # groups 0,1; history_len = min(2*2, 8) = 4.
    # cols: g0s0,g0s1,g1s0,g1s1 -> raw 0,1,2,3; cols 4..7 -> -1
    score = torch.zeros(1, 3, dtype=torch.float32)  # S=3 (>= length), unused for linear
    lengths = torch.tensor([2], dtype=torch.int32)
    out = fast_kpool_topk_transform_fused(score, lengths, pool_size=2, topk=8)
    exp = torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32)
    assert torch.equal(out, exp), f"identity\n{out}\n!=\n{exp}"
    print("[ok] case 1 (linear, identity)")


def _case_page_table_gather():
    # LINEAR + page_table -> PURE GATHER (no *pool_size, no +offset).
    # pool_size=2, topk=4 -> group_topk=2, out_cols=4. length=2 -> history_len=4.
    # raw_token: 0,1,2,3 ; page_table = [[10,11,12,13]] -> dst = 10,11,12,13
    score = torch.zeros(1, 4, dtype=torch.float32)
    lengths = torch.tensor([2], dtype=torch.int32)
    page_table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    out = fast_kpool_topk_transform_fused(score, lengths, pool_size=2, topk=4,
                                          page_table=page_table)
    exp = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    assert torch.equal(out, exp), f"page_table gather\n{out}\n!=\n{exp}"
    print("[ok] case 2 (linear, page_table gather)")


def _case_offset():
    # LINEAR + topk_indices_offset -> raw_token + offset.
    # pool_size=2, topk=4 -> group_topk=2, out_cols=4. length=2 -> history_len=4.
    # raw: 0,1,2,3 ; offset=100 -> 100,101,102,103
    score = torch.zeros(1, 4, dtype=torch.float32)
    lengths = torch.tensor([2], dtype=torch.int32)
    topk_offsets = torch.tensor([100], dtype=torch.int32)
    out = fast_kpool_topk_transform_fused(score, lengths, pool_size=2, topk=4,
                                          topk_indices_offset=topk_offsets)
    exp = torch.tensor([[100, 101, 102, 103]], dtype=torch.int32)
    assert torch.equal(out, exp), f"offset\n{out}\n!=\n{exp}"
    print("[ok] case 3 (linear, +offset)")


def _case_tail():
    # LINEAR + seq_lens tail. pool_size=4, topk=8 -> group_topk=2, out_cols=8+(4-1)=11.
    # length=2 -> full_pool=8, history_len=min(8,8)=8. tail=seq%4=2.
    # history cols 0..7: g0s0..g1s3 = 0,1,2,3,4,5,6,7
    # tail cols 8,9: raw = length*pool_size + (col-8) = 8+0, 8+1 = 8,9
    # col 10 -> -1
    score = torch.zeros(1, 5, dtype=torch.float32)
    lengths = torch.tensor([2], dtype=torch.int32)
    seq_lens = torch.tensor([10], dtype=torch.int32)  # 10 % 4 = 2
    out = fast_kpool_topk_transform_fused(score, lengths, pool_size=4, topk=8,
                                          seq_lens=seq_lens)
    exp = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1]], dtype=torch.int32)
    assert torch.equal(out, exp), f"tail\n{out}\n!=\n{exp}"
    print("[ok] case 4 (linear + seq_lens tail)")


def _case_topk_branch():
    # length(=6) > group_topk(=2): RADIX topk branch. pool_size=2, topk=4.
    # score = [10,50,20,90,30,40] ; topk=2 largest -> indices of 90,50.
    # torch.topk(largest=True,sorted=True): values [90,50], indices [3,1].
    # group_ids = [3,1]; history_len=min(6*2,4)=4.
    # cols: g3s0,g3s1,g1s0,g1s1 = 3*2+0,3*2+1,1*2+0,1*2+1 = 6,7,2,3 ; cols 4,5 -> -1
    score = torch.tensor([[10.0, 50.0, 20.0, 90.0, 30.0, 40.0]], dtype=torch.float32)
    lengths = torch.tensor([6], dtype=torch.int32)
    out = fast_kpool_topk_transform_fused(score, lengths, pool_size=2, topk=4)
    exp = torch.tensor([[6, 7, 2, 3]], dtype=torch.int32)  # out_cols=4 (no seq_lens), history_len=4
    assert torch.equal(out, exp), f"topk\n{out}\n!=\n{exp}"
    print("[ok] case 5 (radix topk branch)")


def _case_page_table_row_index():
    # page_table_row_index: 2 batches, batch 1 reads row 0 of page_table.
    # pool_size=1, topk=2 -> group_topk=2, out_cols=2. length=2 -> history_len=2.
    # raw: b0 0,1 ; b1 0,1. page_table=[[100,200],[999,999]], ptr=[0,0].
    # b0: pt[0][0],pt[0][1] = 100,200 ; b1: pt[0][0],pt[0][1] = 100,200 (ptr=0)
    score = torch.zeros(2, 3, dtype=torch.float32)
    lengths = torch.tensor([2, 2], dtype=torch.int32)
    page_table = torch.tensor([[100, 200], [999, 999]], dtype=torch.int32)
    ptr = torch.tensor([0, 0], dtype=torch.int32)
    out = fast_kpool_topk_transform_fused(score, lengths, pool_size=1, topk=2,
                                          page_table=page_table, page_table_row_index=ptr)
    exp = torch.tensor([[100, 200], [100, 200]], dtype=torch.int32)
    assert torch.equal(out, exp), f"ptr\n{out}\n!=\n{exp}"
    print("[ok] case 6 (page_table_row_index)")


def _case_empty_history():
    # length=0: no history, only tail. pool_size=4, topk=8 -> out_cols=11.
    # history_len=0; tail = seq%4 = 2. raw = 0*4+0, 0*4+1 = 0,1 ; rest -1
    score = torch.zeros(1, 2, dtype=torch.float32)
    lengths = torch.tensor([0], dtype=torch.int32)
    seq_lens = torch.tensor([2], dtype=torch.int32)  # 2 % 4 = 2
    out = fast_kpool_topk_transform_fused(score, lengths, pool_size=4, topk=8,
                                          seq_lens=seq_lens)
    exp = torch.tensor([[0, 1] + [-1] * 9], dtype=torch.int32)
    assert torch.equal(out, exp), f"empty\n{out}\n!=\n{exp}"
    print("[ok] case 7 (empty history + tail)")


if __name__ == "__main__":
    _case_identity()
    _case_page_table_gather()
    _case_offset()
    _case_tail()
    _case_topk_branch()
    _case_page_table_row_index()
    _case_empty_history()
    print("\nALL 7 CASES PASSED — vkernels_dsa_topk matches the kpool_topk_transform contract.")
