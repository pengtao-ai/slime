"""Unit tests for help_seeking / turn-α offload rewards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples.coding_agent_rl import offload

NUM_GPUS = 0


@pytest.fixture(autouse=True)
def _clear_reward_env(monkeypatch):
    for key in (
        "OFFLOAD_SEEK_BUDGET",
        "OFFLOAD_SEEK_BUDGET_TURN_K",
        "OFFLOAD_SEEK_BUDGET_DECAY",
        "OFFLOAD_SEEK_OVERAGE_PENALTY",
        "OFFLOAD_SOLVED_REWARD_FLOOR",
        "OFFLOAD_SEEK_ALPHA",
        "OFFLOAD_MALFORMED_PENALTY",
    ):
        monkeypatch.delenv(key, raising=False)


def _stats(*, oc: int = 0, outside: int = 0, small_o: int = 100, glm_o: int = 0) -> dict:
    return {
        "offload_count": oc,
        "offload_outside_think_count": outside,
        "small_prompt_tokens": 1000,
        "small_output_tokens": small_o,
        "glm_input_tokens": 500 if oc else 0,
        "glm_output_tokens": glm_o if oc else 0,
    }


def _turn(*, valid=False, outside=False, orphan=0, mal=0, small_o=10, glm_o=0, opens=0, closes=0):
    return {
        "small_prompt_tokens": 100,
        "small_output_tokens": small_o,
        "glm_input_tokens": 50 if valid else 0,
        "glm_output_tokens": glm_o if valid else 0,
        "valid_offload": valid,
        "outside_think": outside,
        "orphan_open_count": orphan,
        "malformed_count": mal,
        "open_count": opens or (1 if valid else orphan),
        "close_count": closes or (1 if valid else 0),
        "special_mark_count": (opens or orphan) + (closes or (1 if valid else 0)),
    }


def _sample(*, reward: float, solved: float, oc: int = 0, outside: int = 0, turns=None):
    turns = turns if turns is not None else []
    return SimpleNamespace(
        reward=reward,
        metadata={
            "solved": solved,
            "grading_solved": solved == 1.0,
            "empty_patch": False,
            "offload_stats": {**_stats(oc=oc, outside=outside), "turn_costs": turns},
            "turn_costs": turns,
            "turn_rewards": [0.0] * len(turns) if turns else [],
        },
    )


def test_unsolved_episode_always_zero():
    assert offload.help_seeking_reward(0.0, _stats(oc=2), alpha=1.0) == 0.0
    assert offload.help_seeking_reward(0.0, _stats(oc=0), alpha=1.0) == 0.0


def test_solved_matches_cost_aware():
    st = _stats(oc=1, glm_o=50)
    a = offload.cost_aware_reward(1.0, st, usage=None, lam=0.05)
    b = offload.help_seeking_reward(1.0, st, usage=None, lam=0.05)
    assert b == pytest.approx(a)
    assert 0.0 < b < 1.0


def test_solved_outside_think_does_not_cut_episode():
    bad = _stats(oc=0, outside=1, small_o=0, glm_o=0)
    bad["small_prompt_tokens"] = 0
    bad["glm_input_tokens"] = 0
    assert offload.cost_aware_reward(1.0, bad, usage=None, lam=0.0) == 1.0


def test_reward_mode_help_seeking(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    assert offload.reward_mode() == "help_seeking"
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "cost_aware")
    assert offload.reward_mode() == "cost_aware"


def test_compute_turn_rewards_failed_episode_zero_with_turn_alpha(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    turns = [_turn(valid=True)] + [_turn(valid=False) for _ in range(3)]
    stats = {"turn_costs": turns, "offload_count": 1, "offload_outside_think_count": 0}
    out = offload.compute_turn_rewards(0.0, stats, alpha=1.0, encourage_seek=True)
    assert out["reward"] == 0.0
    assert out["turn_rewards"][0] == pytest.approx(1.0)
    assert out["turn_rewards"][1:] == [0.0, 0.0, 0.0]


def test_compute_turn_rewards_outside_and_malformed_neg(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    turns = [
        _turn(valid=True),
        _turn(valid=True, outside=True),
        _turn(orphan=2, opens=2),
    ]
    stats = {"turn_costs": turns, "offload_count": 1, "offload_outside_think_count": 1}
    out = offload.compute_turn_rewards(
        0.0, stats, alpha=1.0, encourage_seek=True, malformed_pen=0.25
    )
    assert out["reward"] == 0.0
    assert out["turn_rewards"][0] == pytest.approx(1.0)
    assert out["turn_rewards"][1] == pytest.approx(-0.25)
    assert out["turn_rewards"][2] == pytest.approx(-0.25)


def test_compute_turn_rewards_stacks_illegal_and_legal(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    # Valid in-think offload + orphan on the same turn → α − β.
    stacked = _turn(valid=True, orphan=1, opens=2, closes=1)
    assert offload.turn_offload_tag_violation(stacked)
    turns = [stacked]
    stats = {"turn_costs": turns, "offload_count": 1}
    out = offload.compute_turn_rewards(
        0.0, stats, alpha=1.0, encourage_seek=True, malformed_pen=0.25
    )
    assert out["reward"] == 0.0
    assert out["turn_rewards"][0] == pytest.approx(1.0 - 0.25)


def test_compute_turn_rewards_solved_broadcast_and_turn_neg(monkeypatch):
    monkeypatch.setenv("OFFLOAD_SOLVED_REWARD_FLOOR", "0.0")
    turns = [_turn(valid=False), _turn(valid=True, outside=True)]
    stats = {
        "turn_costs": turns,
        "offload_count": 1,
        "offload_outside_think_count": 1,
        "small_prompt_tokens": 0,
        "small_output_tokens": 0,
        "glm_input_tokens": 0,
        "glm_output_tokens": 0,
    }
    out = offload.compute_turn_rewards(1.0, stats, lam=0.0, malformed_pen=0.25)
    assert out["reward"] == pytest.approx(1.0)
    assert out["turn_rewards"][0] == pytest.approx(1.0)
    assert out["turn_rewards"][1] == pytest.approx(-0.25)


def test_compute_turn_rewards_solved_stacks_legal_and_illegal(monkeypatch):
    monkeypatch.setenv("OFFLOAD_SOLVED_REWARD_FLOOR", "0.0")
    stacked = _turn(valid=True, mal=1, opens=2, closes=1)
    assert offload.turn_offload_tag_violation(stacked)
    stats = {
        "turn_costs": [stacked],
        "offload_count": 1,
        "offload_outside_think_count": 0,
        "small_prompt_tokens": 0,
        "small_output_tokens": 0,
        "glm_input_tokens": 0,
        "glm_output_tokens": 0,
    }
    out = offload.compute_turn_rewards(1.0, stats, lam=0.0, malformed_pen=0.25)
    assert out["reward"] == pytest.approx(1.0)
    assert out["turn_rewards"][0] == pytest.approx(1.0 - 0.25)


def test_compute_turn_rewards_defer_seek(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    turns = [_turn(valid=True)]
    stats = {"turn_costs": turns, "offload_count": 1}
    out = offload.compute_turn_rewards(0.0, stats, alpha=1.0, encourage_seek=False)
    assert out["reward"] == 0.0
    assert out["turn_rewards"] == [0.0]


def test_shape_group_all_wrong_paints_turn_alpha_keeps_episode_zero(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "1.0")
    turns_a = [_turn(valid=True), _turn(valid=False)]
    a = _sample(reward=0.0, solved=0.0, oc=1, turns=turns_a)
    b = _sample(reward=0.0, solved=0.0, oc=0, turns=[_turn(valid=False)])
    offload.shape_group_help_seeking_rewards(None, [[a, b]])
    assert a.reward == 0.0
    assert a.metadata["turn_rewards"][0] == pytest.approx(1.0)
    assert a.metadata["turn_rewards"][1] == 0.0
    assert b.reward == 0.0


def test_shape_group_partial_solved_no_turn_alpha(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "1.0")
    failed = _sample(reward=0.0, solved=0.0, oc=2, turns=[_turn(valid=True)])
    solved = _sample(reward=0.9, solved=1.0, oc=0, turns=[_turn(valid=False)])
    solved.metadata["turn_rewards"] = [0.9]
    offload.shape_group_help_seeking_rewards(None, [[failed, solved]])
    assert failed.reward == 0.0
    assert failed.metadata["turn_rewards"] == [0.0]
    assert solved.reward == pytest.approx(0.9)


def test_shape_group_stacks_malformed_and_valid(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "1.0")
    monkeypatch.setenv("OFFLOAD_MALFORMED_PENALTY", "0.25")
    bad = _turn(valid=True, orphan=1, opens=2, closes=1)
    assert offload.turn_offload_tag_violation(bad)
    s = _sample(reward=0.0, solved=0.0, oc=1, turns=[bad])
    s.metadata["turn_rewards"] = [-0.25]
    offload.shape_group_help_seeking_rewards(None, [[s]])
    assert s.metadata["turn_rewards"][0] == pytest.approx(1.0 - 0.25)
    assert s.reward == 0.0


def test_shape_group_malformed_only_keeps_penalty(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "1.0")
    monkeypatch.setenv("OFFLOAD_MALFORMED_PENALTY", "0.25")
    bad = _turn(orphan=2, opens=2)
    assert offload.turn_offload_tag_violation(bad)
    assert not bad["valid_offload"]
    s = _sample(reward=0.0, solved=0.0, oc=0, turns=[bad])
    s.metadata["turn_rewards"] = [-0.25]
    offload.shape_group_help_seeking_rewards(None, [[s]])
    assert s.metadata["turn_rewards"][0] == pytest.approx(-0.25)
    assert s.reward == 0.0


def test_shape_group_fanout_segments(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "1.0")
    turns = [_turn(valid=True)]
    seg0 = _sample(reward=0.0, solved=0.0, oc=1, turns=turns)
    seg1 = _sample(reward=0.0, solved=0.0, oc=1, turns=turns)
    other = _sample(reward=0.0, solved=0.0, oc=0, turns=[_turn(valid=False)])
    offload.shape_group_help_seeking_rewards(None, [[[seg0, seg1], other]])
    assert seg0.metadata["turn_rewards"][0] == pytest.approx(1.0)
    assert seg1.metadata["turn_rewards"][0] == pytest.approx(1.0)
    assert seg0.reward == 0.0


def test_compact_and_shape_does_not_drop(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "1.0")
    good = _sample(reward=0.0, solved=0.0, oc=1, turns=[_turn(valid=True)])
    spam = _sample(
        reward=0.0,
        solved=0.0,
        oc=0,
        turns=[_turn(orphan=10, opens=10, closes=0)],
    )
    offload.compact_and_shape_group_help_seeking_rewards(None, [[good, spam]])
    assert good.reward == 0.0
    assert good.metadata["turn_rewards"][0] == pytest.approx(1.0)
    assert not getattr(spam, "remove_sample", False)


def test_analyze_offload_tags_valid_and_orphan():
    raw = "think <|llm_offload|>3<|/llm_offload|> ok <|llm_offload|>"
    tags = offload.analyze_offload_tags(raw)
    assert tags["valid_count"] == 1
    assert tags["orphan_open_count"] >= 1


def test_default_malformed_penalty_is_008():
    assert offload.malformed_penalty() == pytest.approx(0.08)


def test_default_seek_alpha_is_one():
    assert offload.seek_alpha() == pytest.approx(1.0)


def test_decide_offload_valid_in_think_earns_alpha():
    raw = f"plan {offload.OFFLOAD_OPEN}7{offload.OFFLOAD_CLOSE}"
    assert offload.decide_offload_directive(raw) == (7, True)


def test_decide_offload_orphan_close_fallback():
    raw = f"stuck {offload.OFFLOAD_CLOSE}"
    assert offload.decide_offload_directive(raw) == (3, False)


def test_decide_offload_bad_payload_fallback():
    raw = f"x {offload.OFFLOAD_OPEN}12{offload.OFFLOAD_CLOSE}"
    assert offload.decide_offload_directive(raw) == (3, False)


def test_decide_offload_orphan_open_no_call():
    raw = f"x {offload.OFFLOAD_OPEN}"
    assert offload.decide_offload_directive(raw) is None


def test_decide_offload_digit_span_outside_think_no_call():
    raw = f"</think>\nvisible {offload.OFFLOAD_OPEN}5{offload.OFFLOAD_CLOSE}"
    assert offload.decide_offload_directive(raw) is None
    assert offload.parse_offload_directive(raw) is not None


def test_decide_offload_orphan_close_outside_think_no_call():
    raw = f"</think>\n{offload.OFFLOAD_CLOSE}"
    assert offload.decide_offload_directive(raw) is None
    assert offload.fallback_offload_n(raw) is None


def test_fallback_offload_no_alpha_but_still_beta():
    """Fallback GLM turns keep valid_offload=False → -β only, no α."""
    tc = _turn(valid=False, mal=1, closes=1, opens=0)
    tc["fallback_offload"] = True
    stats = {"turn_costs": [tc], "offload_count": 1, "offload_outside_think_count": 0}
    out = offload.compute_turn_rewards(
        0.0, stats, alpha=1.0, encourage_seek=True, malformed_pen=0.08
    )
    assert out["reward"] == 0.0
    assert out["turn_rewards"][0] == pytest.approx(-0.08)


def test_turn_advantage_paints_residuals():
    import torch

    from examples.coding_agent_rl.offload_turn_advantage import compute_turn_advantages

    kl = [torch.zeros(6)]
    rollout_data = {
        "kl": kl,
        "rewards": [0.5],
        "metadata": [
            {
                "turn_rewards": [1.0, 0.0],
                "turn_token_spans": [[0, 3], [3, 6]],
            }
        ],
    }
    compute_turn_advantages(None, rollout_data)
    adv = rollout_data["advantages"][0]
    assert adv[0].item() == pytest.approx(1.0)
    assert adv[3].item() == pytest.approx(0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
