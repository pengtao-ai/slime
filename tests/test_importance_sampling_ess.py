"""Tests for standard importance-sampling ESS aggregation."""

from __future__ import annotations

import pytest
import torch

from slime.backends.megatron_utils.cp_utils import (
    ESS_LOG_KEY,
    compute_importance_sampling_ess_parts,
    gather_and_reduce_log_dict,
    reduce_importance_sampling_ess,
)

NUM_GPUS = 0


def test_reduce_importance_sampling_ess():
    # w = [1, 1, 1] -> ESS = 9/3 = 3
    assert reduce_importance_sampling_ess(3.0, 3.0) == pytest.approx(3.0)
    # single token w=2 -> ESS = 4/4 = 1
    assert reduce_importance_sampling_ess(2.0, 4.0) == pytest.approx(1.0)
    assert reduce_importance_sampling_ess(0.0, 0.0) != pytest.approx(0.0)  # nan


def test_compute_importance_sampling_ess_parts_uniform(monkeypatch):
    monkeypatch.setattr(
        "slime.backends.megatron_utils.cp_utils.mpu.get_context_parallel_world_size",
        lambda: 1,
    )
    train = [torch.zeros(3), torch.zeros(2)]
    rollout = [torch.zeros(3), torch.zeros(2)]
    masks = [torch.ones(3), torch.ones(2)]
    sum_w, sum_w2 = compute_importance_sampling_ess_parts(
        train,
        rollout,
        masks,
        total_lengths=[7, 6],
        response_lengths=[3, 2],
    )
    assert sum_w == pytest.approx(5.0)
    assert sum_w2 == pytest.approx(5.0)
    assert reduce_importance_sampling_ess(sum_w, sum_w2) == pytest.approx(5.0)


def test_compute_importance_sampling_ess_parts_respects_mask(monkeypatch):
    monkeypatch.setattr(
        "slime.backends.megatron_utils.cp_utils.mpu.get_context_parallel_world_size",
        lambda: 1,
    )
    train = [torch.tensor([0.0, 1.0, 0.0])]
    rollout = [torch.zeros(3)]
    masks = [torch.tensor([1.0, 1.0, 0.0])]
    sum_w, sum_w2 = compute_importance_sampling_ess_parts(
        train,
        rollout,
        masks,
        total_lengths=[6],
        response_lengths=[3],
    )
    # masked tokens: w=[1, e^1]
    assert sum_w == pytest.approx(1.0 + torch.exp(torch.tensor(1.0)).item())
    assert reduce_importance_sampling_ess(sum_w, sum_w2) == pytest.approx(
        (sum_w * sum_w) / sum_w2
    )


def test_gather_and_reduce_log_dict_ess(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        dist,
        "gather_object",
        lambda obj, gathered, dst, group: gathered.__setitem__(slice(None), [{ESS_LOG_KEY: (1.0, 1.0)}, {ESS_LOG_KEY: (2.0, 4.0)}]),
    )
    reduced = gather_and_reduce_log_dict(
        {ESS_LOG_KEY: (1.0, 1.0)},
        dp_size=2,
        dp_src_rank=0,
        dp_group=object(),
    )
    assert reduced is not None
    # global w = [1, 2] -> ESS = 9/5
    assert reduced[ESS_LOG_KEY] == pytest.approx(9.0 / 5.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
