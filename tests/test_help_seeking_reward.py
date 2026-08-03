"""Unit tests for offload.help_seeking_reward."""

from __future__ import annotations

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


def test_unsolved_no_offload_is_zero():
    assert offload.help_seeking_reward(0.0, _stats(oc=0), alpha=0.1) == 0.0


def test_unsolved_in_think_offload_gets_alpha():
    assert offload.help_seeking_reward(0.0, _stats(oc=2), alpha=0.1) == pytest.approx(0.1)


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
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "group_aware")
    assert offload.reward_mode() == "group_aware"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
