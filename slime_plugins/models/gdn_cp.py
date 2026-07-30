"""Native FLA context-parallel helpers for Qwen GDN layers.

Ported from radixark/miles#885 (hybrid CP + zig-zag/packed relayout).

When CP>1, Megatron ring-attention uses a zig-zag token layout while
``flash-linear-attention`` CP expects contiguous packed shards. Relayout
bridges the two so GDN can run with per-rank seq length ``L/CP`` instead of
all-gathering the full sequence.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from fla.ops.cp import build_cp_context as _fla_build_cp_context
except ImportError:
    _fla_build_cp_context = None


def _get_cp_sequence_lengths(cu_seqlens, cp_size, local_total_len=None):
    global_seq_lengths = [(cu_seqlens[i + 1] - cu_seqlens[i]).item() for i in range(len(cu_seqlens) - 1)]
    local_seq_lengths = []
    for global_seq_len in global_seq_lengths:
        if global_seq_len % cp_size != 0:
            raise ValueError(f"Expected sequence length {global_seq_len} to be divisible by cp_size={cp_size}")
        local_seq_lengths.append(global_seq_len // cp_size)

    if local_total_len is not None and sum(local_seq_lengths) != local_total_len:
        raise ValueError(f"Expected local total length {local_total_len}, got {sum(local_seq_lengths)}")

    return global_seq_lengths, local_seq_lengths


def _gather_cp_tensors(x, cp_group):
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size(group=cp_group))]
    dist.all_gather(gathered, x.contiguous(), group=cp_group)
    return gathered


def zigzag_to_packed_shard_impl(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    """Convert zigzag ring-attn layout to the contiguous packed shard expected by fla CP."""
    global_seq_lengths, local_seq_lengths = _get_cp_sequence_lengths(cu_seqlens, cp_size, hidden_states.size(0))
    gathered_by_rank = [
        gathered.split(local_seq_lengths, dim=0) for gathered in _gather_cp_tensors(hidden_states, cp_group)
    ]

    full_sequences = []
    for seq_idx, global_seq_len in enumerate(global_seq_lengths):
        per_rank = [rank_seqs[seq_idx] for rank_seqs in gathered_by_rank]
        if global_seq_len % (2 * cp_size) == 0:
            subchunk_len = global_seq_len // (2 * cp_size)
            full_seq = torch.cat(
                [seq[:subchunk_len] for seq in per_rank] + [seq[subchunk_len:] for seq in per_rank][::-1],
                dim=0,
            )
        else:
            # Final local padding is appended contiguously on each rank, not in zigzag order.
            full_seq = torch.cat(per_rank, dim=0)
        full_sequences.append(full_seq)

    full_stream = torch.cat(full_sequences, dim=0) if full_sequences else hidden_states[:0]
    shard_len = hidden_states.size(0)
    return full_stream[cp_rank * shard_len : (cp_rank + 1) * shard_len]


def packed_shard_to_zigzag_impl(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    """Convert contiguous packed shard layout back to zigzag ring-attn layout."""
    global_seq_lengths, local_seq_lengths = _get_cp_sequence_lengths(cu_seqlens, cp_size, hidden_states.size(0))
    full_stream = torch.cat(_gather_cp_tensors(hidden_states, cp_group), dim=0)
    full_sequences = full_stream.split(global_seq_lengths, dim=0)

    local_sequences = []
    for full_seq, global_seq_len, local_seq_len in zip(
        full_sequences, global_seq_lengths, local_seq_lengths, strict=True
    ):
        if global_seq_len % (2 * cp_size) == 0:
            subchunk_len = global_seq_len // (2 * cp_size)
            parts = full_seq.split(subchunk_len, dim=0)
            local_sequences.append(torch.cat([parts[cp_rank], parts[2 * cp_size - 1 - cp_rank]], dim=0))
        else:
            local_sequences.append(full_seq.split(local_seq_len, dim=0)[cp_rank])

    return torch.cat(local_sequences, dim=0) if local_sequences else hidden_states[:0]


class _ZigzagToPackedShard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.save_for_backward(cu_seqlens)
        return zigzag_to_packed_shard_impl(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)

    @staticmethod
    def backward(ctx, grad_output):
        (cu_seqlens,) = ctx.saved_tensors
        result = packed_shard_to_zigzag_impl(grad_output, cu_seqlens, ctx.cp_group, ctx.cp_rank, ctx.cp_size)
        return result, None, None, None, None


class _PackedShardToZigzag(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.save_for_backward(cu_seqlens)
        return packed_shard_to_zigzag_impl(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)

    @staticmethod
    def backward(ctx, grad_output):
        (cu_seqlens,) = ctx.saved_tensors
        result = zigzag_to_packed_shard_impl(grad_output, cu_seqlens, ctx.cp_group, ctx.cp_rank, ctx.cp_size)
        return result, None, None, None, None


def zigzag_to_packed_shard(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    return _ZigzagToPackedShard.apply(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)


def packed_shard_to_zigzag(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    return _PackedShardToZigzag.apply(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)


def build_gdn_cp_context(module: nn.Module, cu_seqlens: torch.Tensor, device: torch.device):
    """Build fla CP context for a GatedDeltaNet module from packed sequence boundaries.

    Returns ``None`` when CP is not configured on the module (``cp_group`` not set).
    """
    cp_group = getattr(module, "cp_group", None)
    if cp_group is None:
        return None
    if _fla_build_cp_context is None:
        raise RuntimeError(
            "Hybrid GDN CP requires fla.ops.cp (flash-linear-attention >= 0.4.2) " "but it could not be imported."
        )
    if cu_seqlens is None or cu_seqlens.numel() < 2:
        raise ValueError(f"Hybrid CP requires valid cu_seqlens (at least 2 elements) but got {cu_seqlens}")
    return _fla_build_cp_context(
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
        group=cp_group,
        conv1d_kernel_size=module.conv_kernel_size,
    )


def detect_and_setup_hybrid_cp(model: nn.Module, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> int:
    """Scan for GDN Attention modules and enable native fla CP (no full-seq all-gather)."""
    # Lazy import avoids circular deps at module load.
    from slime_plugins.models.hf_attention import HuggingfaceAttention

    if _fla_build_cp_context is None:
        raise RuntimeError(
            "context_parallel_size > 1 with Qwen GDN requires flash-linear-attention >= 0.4.2 "
            "(fla.ops.cp). Upgrade with: pip install 'flash-linear-attention>=0.4.2'"
        )

    count = 0
    for module in model.modules():
        if isinstance(module, HuggingfaceAttention):
            linear_attn = getattr(module, "linear_attn", None)
            if linear_attn is not None:
                linear_attn.cp_group = cp_group
                linear_attn.cp_rank = cp_rank
                linear_attn.cp_world_size = cp_world_size
                module.hybrid_cp = True
                count += 1

    if count > 0:
        logger.info("Configured hybrid CP on %d GDN modules (fla native state passing)", count)
    return count
