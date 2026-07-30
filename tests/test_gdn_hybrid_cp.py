"""Unit tests for GDN hybrid-CP zig-zag ↔ packed shard relayout (miles#885)."""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist
import torch.nn as nn

from slime_plugins.models.gdn_cp import (
    packed_shard_to_zigzag,
    packed_shard_to_zigzag_impl,
    zigzag_to_packed_shard,
    zigzag_to_packed_shard_impl,
)


class _FakeProcessGroup:
    def __init__(self, world_size: int, rank: int):
        self._world_size = world_size
        self._rank = rank

    def size(self):
        return self._world_size

    def rank(self):
        return self._rank


def _patch_all_gather(monkeypatch, shards: list[torch.Tensor]):
    world_size = len(shards)

    def fake_get_world_size(group=None):
        return world_size

    def fake_all_gather(tensor_list, tensor, group=None):
        assert len(tensor_list) == world_size
        for i in range(world_size):
            tensor_list[i].copy_(shards[i])

    monkeypatch.setattr(dist, "get_world_size", fake_get_world_size)
    monkeypatch.setattr(dist, "all_gather", fake_all_gather)


def _zigzag_shards_from_full(full: torch.Tensor, cp_size: int) -> list[torch.Tensor]:
    n = full.size(0)
    assert n % (2 * cp_size) == 0
    chunk = n // (2 * cp_size)
    parts = full.split(chunk, dim=0)
    return [torch.cat([parts[r], parts[2 * cp_size - 1 - r]], dim=0) for r in range(cp_size)]


def test_zigzag_to_packed_and_back_roundtrip(monkeypatch):
    cp_size = 2
    full = torch.arange(8, dtype=torch.float32).unsqueeze(-1)
    shards = _zigzag_shards_from_full(full, cp_size)
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)

    _patch_all_gather(monkeypatch, shards)
    packed = [
        zigzag_to_packed_shard_impl(shards[r], cu_seqlens, _FakeProcessGroup(cp_size, r), r, cp_size)
        for r in range(cp_size)
    ]
    assert torch.equal(packed[0].squeeze(-1), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(packed[1].squeeze(-1), torch.tensor([4.0, 5.0, 6.0, 7.0]))

    _patch_all_gather(monkeypatch, packed)
    for r in range(cp_size):
        zig = packed_shard_to_zigzag_impl(packed[r], cu_seqlens, _FakeProcessGroup(cp_size, r), r, cp_size)
        assert torch.equal(zig, shards[r])


def test_multi_sequence_packing_relayout(monkeypatch):
    """Two packed sequences under CP=2 → contiguous shards of the packed stream."""
    cp_size = 2
    full0 = torch.arange(0, 8, dtype=torch.float32).unsqueeze(-1)
    full1 = torch.arange(100, 108, dtype=torch.float32).unsqueeze(-1)
    zig0 = _zigzag_shards_from_full(full0, cp_size)
    zig1 = _zigzag_shards_from_full(full1, cp_size)
    shards = [torch.cat([zig0[r], zig1[r]], dim=0) for r in range(cp_size)]
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)

    _patch_all_gather(monkeypatch, shards)
    packed = [
        zigzag_to_packed_shard_impl(shards[r], cu_seqlens, _FakeProcessGroup(cp_size, r), r, cp_size)
        for r in range(cp_size)
    ]
    # full_stream = [0..7, 100..107]; each rank takes a contiguous half of the stream
    assert torch.equal(packed[0].squeeze(-1), torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    assert torch.equal(
        packed[1].squeeze(-1),
        torch.tensor([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]),
    )

    _patch_all_gather(monkeypatch, packed)
    for r in range(cp_size):
        zig = packed_shard_to_zigzag_impl(packed[r], cu_seqlens, _FakeProcessGroup(cp_size, r), r, cp_size)
        assert torch.equal(zig, shards[r])


def test_tail_padding_uses_contiguous_layout(monkeypatch):
    cp_size = 2
    full = torch.arange(6, dtype=torch.float32).unsqueeze(-1)
    local_len = 6 // cp_size
    shards = [full[r * local_len : (r + 1) * local_len].clone() for r in range(cp_size)]
    cu_seqlens = torch.tensor([0, 6], dtype=torch.int32)

    _patch_all_gather(monkeypatch, shards)
    packed = [
        zigzag_to_packed_shard_impl(shards[r], cu_seqlens, _FakeProcessGroup(cp_size, r), r, cp_size)
        for r in range(cp_size)
    ]
    assert torch.equal(packed[0].squeeze(-1), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(packed[1].squeeze(-1), torch.tensor([3.0, 4.0, 5.0]))

    _patch_all_gather(monkeypatch, packed)
    for r in range(cp_size):
        zig = packed_shard_to_zigzag_impl(packed[r], cu_seqlens, _FakeProcessGroup(cp_size, r), r, cp_size)
        assert torch.equal(zig, shards[r])


def test_autograd_function_matches_impl(monkeypatch):
    cp_size = 2
    full = torch.arange(8, dtype=torch.float32).view(8, 1, 1)
    shards = _zigzag_shards_from_full(full, cp_size)
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)

    _patch_all_gather(monkeypatch, shards)
    for r in range(cp_size):
        via_fn = zigzag_to_packed_shard(shards[r], cu_seqlens, _FakeProcessGroup(cp_size, r), r, cp_size)
        via_impl = zigzag_to_packed_shard_impl(shards[r], cu_seqlens, _FakeProcessGroup(cp_size, r), r, cp_size)
        assert torch.equal(via_fn, via_impl)

    packed = [
        zigzag_to_packed_shard_impl(shards[r], cu_seqlens, _FakeProcessGroup(cp_size, r), r, cp_size)
        for r in range(cp_size)
    ]
    _patch_all_gather(monkeypatch, packed)
    for r in range(cp_size):
        via_fn = packed_shard_to_zigzag(packed[r], cu_seqlens, _FakeProcessGroup(cp_size, r), r, cp_size)
        via_impl = packed_shard_to_zigzag_impl(packed[r], cu_seqlens, _FakeProcessGroup(cp_size, r), r, cp_size)
        assert torch.equal(via_fn, via_impl)
        assert torch.equal(via_fn, shards[r])


def test_cp4_single_sequence_relayout(monkeypatch):
    cp_size = 4
    full = torch.arange(16, dtype=torch.float32).unsqueeze(-1)
    shards = _zigzag_shards_from_full(full, cp_size)
    cu_seqlens = torch.tensor([0, 16], dtype=torch.int32)

    _patch_all_gather(monkeypatch, shards)
    packed = [
        zigzag_to_packed_shard_impl(shards[r], cu_seqlens, _FakeProcessGroup(cp_size, r), r, cp_size)
        for r in range(cp_size)
    ]
    for r in range(cp_size):
        assert torch.equal(packed[r], full[r * 4 : (r + 1) * 4])

    _patch_all_gather(monkeypatch, packed)
    for r in range(cp_size):
        zig = packed_shard_to_zigzag_impl(packed[r], cu_seqlens, _FakeProcessGroup(cp_size, r), r, cp_size)
        assert torch.equal(zig, shards[r])


def test_detect_and_setup_sets_hybrid_flag(monkeypatch):
    from tests.test_qwen3_linear_attention_cu_seqlens import install_megatron_stubs

    install_megatron_stubs()
    for name in list(sys.modules):
        if name.startswith("slime_plugins.models.hf_attention") or name.startswith("slime_plugins.models.gdn_cp"):
            del sys.modules[name]

    from slime_plugins.models.gdn_cp import detect_and_setup_hybrid_cp
    from slime_plugins.models.hf_attention import HuggingfaceAttention

    class _DummyGDN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_kernel_size = 4

    class _DummyAttn(HuggingfaceAttention):
        def __init__(self):
            nn.Module.__init__(self)
            self.linear_attn = _DummyGDN()
            self.hybrid_cp = False

        def hf_forward(self, hidden_states, packed_seq_params):
            return hidden_states

    root = nn.Module()
    root.attn = _DummyAttn()
    count = detect_and_setup_hybrid_cp(root, _FakeProcessGroup(2, 0), 0, 2)
    assert count == 1
    assert root.attn.hybrid_cp is True
    assert root.attn.linear_attn.cp_group is not None
    assert root.attn.linear_attn.cp_world_size == 2
