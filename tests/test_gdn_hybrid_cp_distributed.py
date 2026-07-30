#!/usr/bin/env python3
"""Distributed precision tests for hybrid GDN CP (miles#885 port).

Kept intentionally tiny: first FLA/Triton kernel compile dominates wall time,
so we use small dims, run the non-CP baseline on rank 0 only, and skip the
packed case unless ``GDN_CP_TEST_PACKED=1``.

Run (needs >=2 GPUs)::

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \\
      tests/test_gdn_hybrid_cp_distributed.py

    GDN_CP_TEST_PACKED=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone \\
      --nproc_per_node=2 tests/test_gdn_hybrid_cp_distributed.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Allow `torchrun tests/...` without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist

# Small enough for fast Triton compile; lengths stay multiples of CP and of
# FLA's common chunk size (64).
_HIDDEN = 64
_SEQ_PER_RANK = 64
_TOL = 2e-2


def _log(rank: int, msg: str, t0: float) -> None:
    if rank == 0:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def setup_dist():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def _make_subchunk(sample_id: int, sub_id: int, chunk_len: int, device: torch.device) -> torch.Tensor:
    base = sample_id * 1000 + sub_id * 100
    values = torch.arange(base, base + chunk_len, device=device, dtype=torch.float32)
    return values.view(-1, 1, 1)


def _build_rank_inputs(rank: int, world_size: int, device: torch.device):
    chunk_lens = [3, 5]
    tail_pad_local_len = 3
    zigzag_chunks = []
    full_sequences = []
    cu = [0]

    for sample_id, chunk_len in enumerate(chunk_lens):
        subchunks = [_make_subchunk(sample_id, sub_id, chunk_len, device) for sub_id in range(2 * world_size)]
        zigzag_chunks.extend([subchunks[rank], subchunks[2 * world_size - 1 - rank]])
        full_sequences.append(torch.cat(subchunks, dim=0))
        cu.append(cu[-1] + 2 * world_size * chunk_len)

    tail_pad = (rank * 10000 + torch.arange(tail_pad_local_len, device=device, dtype=torch.float32)).view(-1, 1, 1)
    zigzag_chunks.append(tail_pad)
    full_sequences.append(
        torch.cat(
            [
                (r * 10000 + torch.arange(tail_pad_local_len, device=device, dtype=torch.float32)).view(-1, 1, 1)
                for r in range(world_size)
            ],
            dim=0,
        )
    )
    cu.append(cu[-1] + world_size * tail_pad_local_len)

    zigzag = torch.cat(zigzag_chunks, dim=0).requires_grad_(True)
    packed_full = torch.cat(full_sequences, dim=0)
    local_len = zigzag.size(0)
    packed_shard = packed_full[rank * local_len : (rank + 1) * local_len]
    cu_seqlens = torch.tensor(cu, device=device, dtype=torch.int32)
    return zigzag, packed_shard, cu_seqlens


def test_relayout(rank: int, world_size: int, t0: float) -> bool:
    from slime_plugins.models.gdn_cp import packed_shard_to_zigzag, zigzag_to_packed_shard

    device = torch.device(f"cuda:{rank}")
    cp_group = dist.group.WORLD

    zigzag, expected_packed_shard, cu_seqlens = _build_rank_inputs(rank, world_size, device)

    packed_shard = zigzag_to_packed_shard(zigzag, cu_seqlens, cp_group, rank, world_size)
    roundtrip = packed_shard_to_zigzag(packed_shard, cu_seqlens, cp_group, rank, world_size)

    packed_ok = torch.equal(packed_shard, expected_packed_shard)
    roundtrip_ok = torch.equal(roundtrip, zigzag)

    loss = roundtrip.sum()
    loss.backward()
    grad_ok = torch.equal(zigzag.grad, torch.ones_like(zigzag))

    passed = packed_ok and roundtrip_ok and grad_ok
    _log(rank, f"relayout packed={packed_ok} roundtrip={roundtrip_ok} bwd={grad_ok}", t0)
    return passed


def _build_gdn(device, dtype=torch.bfloat16):
    class FakeConfig:
        hidden_size = _HIDDEN
        linear_num_value_heads = 2
        linear_num_key_heads = 1
        linear_key_head_dim = 32
        linear_value_head_dim = 32
        linear_conv_kernel_dim = 4
        hidden_act = "silu"
        rms_norm_eps = 1e-6

    FakeConfig.dtype = dtype

    from slime_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet

    return Qwen3_5GatedDeltaNet(FakeConfig(), layer_idx=0, args=None).to(device=device, dtype=dtype)


def _broadcast_tensor(t: torch.Tensor | None, rank: int, shape: tuple[int, ...], dtype, device) -> torch.Tensor:
    if rank == 0:
        assert t is not None
        out = t.detach().contiguous()
    else:
        out = torch.empty(shape, device=device, dtype=dtype)
    dist.broadcast(out, src=0)
    return out


def _compare_fwd_bwd(rank, world_size, ref_out, ref_grad, cp_out, local_grad, *, tag: str, t0: float) -> bool:
    gathered_out = [torch.zeros_like(cp_out) for _ in range(world_size)]
    dist.all_gather(gathered_out, cp_out.contiguous())
    full_cp_out = torch.cat(gathered_out, dim=1)

    gathered_grad = [torch.zeros_like(local_grad) for _ in range(world_size)]
    dist.all_gather(gathered_grad, local_grad.contiguous())
    full_cp_grad = torch.cat(gathered_grad, dim=1)

    if rank != 0:
        return True

    out_diff = (ref_out.detach().float() - full_cp_out.detach().float()).abs()
    grad_diff = (ref_grad.float() - full_cp_grad.float()).abs()
    out_max = out_diff.max().item()
    grad_max = grad_diff.max().item()
    out_rel = (out_diff / (ref_out.detach().float().abs() + 1e-5)).max().item()
    grad_rel = (grad_diff / (ref_grad.float().abs() + 1e-5)).max().item()

    fwd_ok = out_max < _TOL
    bwd_ok = grad_max < _TOL
    _log(
        rank,
        f"{tag}: fwd abs={out_max:.3e} rel={out_rel:.3e} PASS={fwd_ok}; "
        f"bwd abs={grad_max:.3e} rel={grad_rel:.3e} PASS={bwd_ok}",
        t0,
    )
    return fwd_ok and bwd_ok


def test_gdn_single_sequence(rank: int, world_size: int, t0: float) -> bool:
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    total_seq_len = _SEQ_PER_RANK * world_size
    local_seq_len = _SEQ_PER_RANK

    # Non-CP baseline on rank 0 only (avoids N× Triton compiles).
    ref_out = None
    ref_grad = None
    state_dict = None
    if rank == 0:
        _log(rank, "GDN single-seq: building baseline (first FLA compile is slow)", t0)
        torch.manual_seed(42)
        model_ref = _build_gdn(device, dtype)
        torch.manual_seed(123)
        full_hidden = torch.randn(1, total_seq_len, _HIDDEN, device=device, dtype=dtype, requires_grad=True)
        full_cu = torch.tensor([0, total_seq_len], dtype=torch.int32, device=device)
        ref_out = model_ref(full_hidden, cu_seqlens=full_cu)
        ref_out.sum().backward()
        ref_grad = full_hidden.grad.detach().contiguous()
        ref_out = ref_out.detach().contiguous()
        state_dict = {k: v.detach().contiguous() for k, v in model_ref.state_dict().items()}
        _log(rank, "GDN single-seq: baseline done", t0)
        del model_ref, full_hidden

    dist.barrier()
    ref_out = _broadcast_tensor(ref_out, rank, (1, total_seq_len, _HIDDEN), dtype, device)
    ref_grad = _broadcast_tensor(ref_grad, rank, (1, total_seq_len, _HIDDEN), dtype, device)

    # Broadcast weights from rank 0.
    torch.manual_seed(42)
    model_cp = _build_gdn(device, dtype)
    if rank == 0:
        assert state_dict is not None
        for name, tensor in state_dict.items():
            dist.broadcast(tensor, src=0)
            state_dict[name] = tensor
        model_cp.load_state_dict(state_dict)
    else:
        buf = {k: torch.empty_like(v) for k, v in model_cp.state_dict().items()}
        for name, tensor in buf.items():
            dist.broadcast(tensor, src=0)
        model_cp.load_state_dict(buf)

    model_cp.cp_group = dist.group.WORLD
    model_cp.cp_rank = rank
    model_cp.cp_world_size = world_size

    start, end = rank * local_seq_len, (rank + 1) * local_seq_len
    torch.manual_seed(123)
    full_hidden_cp = torch.randn(1, total_seq_len, _HIDDEN, device=device, dtype=dtype)
    local_hidden = full_hidden_cp[:, start:end, :].clone().contiguous().requires_grad_(True)
    global_cu = torch.tensor([0, total_seq_len], dtype=torch.int32, device=device)

    _log(rank, "GDN single-seq: CP forward/backward", t0)
    cp_out = model_cp(local_hidden, cu_seqlens=global_cu)
    cp_out.sum().backward()
    _log(rank, "GDN single-seq: CP done", t0)

    return _compare_fwd_bwd(
        rank,
        world_size,
        ref_out,
        ref_grad,
        cp_out,
        local_hidden.grad,
        tag="GDN single-seq",
        t0=t0,
    )


def test_gdn_packed_sequences(rank: int, world_size: int, t0: float) -> bool:
    """Two packed sequences; contiguous packed-shard layout (post-relayout)."""
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    len0 = _SEQ_PER_RANK * world_size
    len1 = (_SEQ_PER_RANK // 2) * world_size
    total = len0 + len1
    global_cu = torch.tensor([0, len0, total], dtype=torch.int32, device=device)
    local_len = total // world_size

    ref_out = None
    ref_grad = None
    state_dict = None
    if rank == 0:
        _log(rank, "GDN packed: baseline", t0)
        torch.manual_seed(7)
        model_ref = _build_gdn(device, dtype)
        torch.manual_seed(99)
        full_hidden = torch.randn(1, total, _HIDDEN, device=device, dtype=dtype, requires_grad=True)
        ref_out = model_ref(full_hidden, cu_seqlens=global_cu)
        ref_out.sum().backward()
        ref_grad = full_hidden.grad.detach().contiguous()
        ref_out = ref_out.detach().contiguous()
        state_dict = {k: v.detach().contiguous() for k, v in model_ref.state_dict().items()}
        del model_ref, full_hidden

    dist.barrier()
    ref_out = _broadcast_tensor(ref_out, rank, (1, total, _HIDDEN), dtype, device)
    ref_grad = _broadcast_tensor(ref_grad, rank, (1, total, _HIDDEN), dtype, device)

    torch.manual_seed(7)
    model_cp = _build_gdn(device, dtype)
    if rank == 0:
        assert state_dict is not None
        for name, tensor in state_dict.items():
            dist.broadcast(tensor, src=0)
            state_dict[name] = tensor
        model_cp.load_state_dict(state_dict)
    else:
        buf = {k: torch.empty_like(v) for k, v in model_cp.state_dict().items()}
        for name, tensor in buf.items():
            dist.broadcast(tensor, src=0)
        model_cp.load_state_dict(buf)

    model_cp.cp_group = dist.group.WORLD
    model_cp.cp_rank = rank
    model_cp.cp_world_size = world_size

    start, end = rank * local_len, (rank + 1) * local_len
    torch.manual_seed(99)
    full_hidden_cp = torch.randn(1, total, _HIDDEN, device=device, dtype=dtype)
    local_hidden = full_hidden_cp[:, start:end, :].clone().contiguous().requires_grad_(True)

    _log(rank, "GDN packed: CP forward/backward", t0)
    cp_out = model_cp(local_hidden, cu_seqlens=global_cu)
    cp_out.sum().backward()

    return _compare_fwd_bwd(
        rank,
        world_size,
        ref_out,
        ref_grad,
        cp_out,
        local_hidden.grad,
        tag="GDN packed multi-seq",
        t0=t0,
    )


def main():
    rank, world_size, _ = setup_dist()
    t0 = time.time()
    results = []
    try:
        _log(rank, f"start CP={world_size} precision suite", t0)
        results.append(("relayout", test_relayout(rank, world_size, t0)))
        dist.barrier()
        results.append(("gdn_single", test_gdn_single_sequence(rank, world_size, t0)))
        if os.environ.get("GDN_CP_TEST_PACKED", "0") == "1":
            dist.barrier()
            results.append(("gdn_packed", test_gdn_packed_sequences(rank, world_size, t0)))
        else:
            _log(rank, "skip gdn_packed (set GDN_CP_TEST_PACKED=1 to enable)", t0)
    finally:
        flags = torch.tensor([1 if ok else 0 for _, ok in results], device=f"cuda:{rank}", dtype=torch.int32)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        if rank == 0:
            print("\n=== Summary ===", flush=True)
            for (name, _), flag in zip(results, flags.tolist(), strict=True):
                print(f"{name}: {'PASS' if flag else 'FAIL'}", flush=True)
            if not all(flags.tolist()):
                dist.destroy_process_group()
                sys.exit(1)
            print(f"All CP={world_size} precision checks PASSED in {time.time() - t0:.1f}s", flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
