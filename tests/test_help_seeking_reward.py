"""Unit tests for offload.help_seeking_reward."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples.coding_agent_rl import offload

NUM_GPUS = 0


def _stats(*, oc: int = 0, outside: int = 0, small_o: int = 100, glm_o: int = 0) -> dict:
    return {
        "offload_count": oc,
        "offload_outside_think_count": outside,
        "small_prompt_tokens": 1000,
        "small_output_tokens": small_o,
        "glm_input_tokens": 500 if oc else 0,
        "glm_output_tokens": glm_o if oc else 0,
    }


def _sample(*, reward: float, solved: float, oc: int = 0, outside: int = 0, empty_patch: bool = False):
    return SimpleNamespace(
        reward=reward,
        metadata={
            "solved": solved,
            "grading_solved": solved == 1.0,
            "empty_patch": empty_patch,
            "offload_stats": _stats(oc=oc, outside=outside),
        },
    )


def test_unsolved_no_offload_is_zero():
    assert offload.help_seeking_reward(0.0, _stats(oc=0), alpha=0.1) == 0.0


def test_unsolved_in_think_offload_gets_alpha():
    assert offload.help_seeking_reward(0.0, _stats(oc=2), alpha=0.1) == pytest.approx(0.1)


def test_unsolved_encourage_seek_false_is_zero():
    assert (
        offload.help_seeking_reward(0.0, _stats(oc=2), alpha=0.1, encourage_seek=False) == 0.0
    )


def test_unsolved_empty_patch_scales_alpha():
    r = offload.help_seeking_reward(
        0.0, _stats(oc=1), alpha=0.1, empty_patch=True, empty_scale=0.5
    )
    assert r == pytest.approx(0.05)


def test_unsolved_outside_think_gets_zero():
    assert offload.help_seeking_reward(0.0, _stats(oc=3, outside=1), alpha=0.1) == 0.0


def test_solved_matches_cost_aware_without_unique_bonus():
    st = _stats(oc=1, glm_o=50)
    a = offload.cost_aware_reward(1.0, st, usage=None, lam=0.05)
    b = offload.help_seeking_reward(1.0, st, usage=None, lam=0.05)
    assert b == pytest.approx(a)
    assert 0.0 < b < 1.0


def test_unique_solver_bonus():
    st = _stats(oc=1, glm_o=10)
    base = offload.help_seeking_reward(1.0, st, usage=None, lam=0.05)
    bumped = offload.help_seeking_reward(
        1.0, st, usage=None, lam=0.05, unique_solver=True, unique_bonus=0.15
    )
    assert bumped == pytest.approx(base + 0.15)


def test_reward_mode_help_seeking(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    assert offload.reward_mode() == "help_seeking"
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "cost_aware")
    assert offload.reward_mode() == "cost_aware"


def test_shape_group_all_wrong_grants_alpha(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "0.1")
    a = _sample(reward=0.0, solved=0.0, oc=1)
    b = _sample(reward=0.0, solved=0.0, oc=0)
    offload.shape_group_help_seeking_rewards(None, [[a, b]])
    assert a.reward == pytest.approx(0.1)
    assert b.reward == 0.0


def test_shape_group_any_solved_does_not_encourage(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "0.1")
    failed_offload = _sample(reward=0.0, solved=0.0, oc=2)
    solved = _sample(reward=0.9, solved=1.0, oc=0)
    offload.shape_group_help_seeking_rewards(None, [[failed_offload, solved]])
    assert failed_offload.reward == 0.0
    assert solved.reward == pytest.approx(0.9)


def test_shape_group_noop_without_flag(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.delenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", raising=False)
    a = _sample(reward=0.0, solved=0.0, oc=1)
    offload.shape_group_help_seeking_rewards(None, [[a]])
    assert a.reward == 0.0


def test_shape_group_fanout_segments(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "0.1")
    seg0 = _sample(reward=0.0, solved=0.0, oc=1)
    seg1 = _sample(reward=0.0, solved=0.0, oc=1)
    other = _sample(reward=0.0, solved=0.0, oc=0)
    offload.shape_group_help_seeking_rewards(None, [[[seg0, seg1], other]])
    assert seg0.reward == pytest.approx(0.1)
    assert seg1.reward == pytest.approx(0.1)
    assert other.reward == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
