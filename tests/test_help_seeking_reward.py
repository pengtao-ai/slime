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
    monkeypatch.setenv("OFFLOAD_NO_SEEK_PENALTY", "0.1")
    a = _sample(reward=0.0, solved=0.0, oc=1)
    b = _sample(reward=0.0, solved=0.0, oc=0)
    offload.shape_group_help_seeking_rewards(None, [[a, b]])
    assert a.reward == pytest.approx(0.1)
    assert b.reward == pytest.approx(-0.1)


def test_shape_group_all_wrong_no_seek_penalty(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_NO_SEEK_PENALTY", "0.15")
    a = _sample(reward=0.0, solved=0.0, oc=0)
    b = _sample(reward=0.0, solved=0.0, oc=0)
    offload.shape_group_help_seeking_rewards(None, [[a, b]])
    assert a.reward == pytest.approx(-0.15)
    assert b.reward == pytest.approx(-0.15)


def test_shape_group_not_all_wrong_skips_no_seek_penalty(monkeypatch):
    """If someone solved via offload, failed non-seekers stay at 0 (not 全做错)."""
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "0.1")
    monkeypatch.setenv("OFFLOAD_NO_SEEK_PENALTY", "0.1")
    failed = _sample(reward=0.0, solved=0.0, oc=0)
    offload_solved = _sample(reward=0.85, solved=1.0, oc=2)
    offload.shape_group_help_seeking_rewards(None, [[failed, offload_solved]])
    assert failed.reward == 0.0
    assert offload_solved.reward == pytest.approx(0.85)


def test_shape_group_solo_solved_does_not_encourage(monkeypatch):
    """Solo solve (no offload) blocks α for failed help-seekers."""
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "0.1")
    failed_offload = _sample(reward=0.0, solved=0.0, oc=2)
    solo_solved = _sample(reward=0.9, solved=1.0, oc=0)
    offload.shape_group_help_seeking_rewards(None, [[failed_offload, solo_solved]])
    assert failed_offload.reward == 0.0
    assert solo_solved.reward == pytest.approx(0.9)


def test_shape_group_offload_solved_still_grants_alpha(monkeypatch):
    """A sibling that solved *with* offload must not block α."""
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "0.1")
    failed_offload = _sample(reward=0.0, solved=0.0, oc=2)
    offload_solved = _sample(reward=0.85, solved=1.0, oc=3)
    offload.shape_group_help_seeking_rewards(None, [[failed_offload, offload_solved]])
    assert failed_offload.reward == pytest.approx(0.1)
    assert offload_solved.reward == pytest.approx(0.85)


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
    monkeypatch.setenv("OFFLOAD_NO_SEEK_PENALTY", "0.1")
    seg0 = _sample(reward=0.0, solved=0.0, oc=1)
    seg1 = _sample(reward=0.0, solved=0.0, oc=1)
    other = _sample(reward=0.0, solved=0.0, oc=0)
    offload.shape_group_help_seeking_rewards(None, [[[seg0, seg1], other]])
    assert seg0.reward == pytest.approx(0.1)
    assert seg1.reward == pytest.approx(0.1)
    assert other.reward == pytest.approx(-0.1)


def test_tmax_empty_patch_does_not_scale_alpha():
    st = _stats(oc=1)
    r = offload.help_seeking_reward(
        0.0, st, alpha=0.1, empty_patch=True, empty_scale=0.5, protocol="tmax"
    )
    assert r == pytest.approx(0.1)


def test_scaleswe_empty_patch_scales_alpha():
    st = _stats(oc=1)
    r = offload.help_seeking_reward(
        0.0, st, alpha=0.1, empty_patch=True, empty_scale=0.5, protocol="scaleswe"
    )
    assert r == pytest.approx(0.05)


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
        "max_open_run": orphan,
        "special_mark_count": (opens or orphan) + (closes or (1 if valid else 0)),
    }


def test_compute_turn_rewards_solved_uses_completion_budget():
    turns = [_turn(valid=True, small_o=100, glm_o=0), _turn(valid=False, small_o=100, glm_o=0)]
    stats = {"turn_costs": turns, "offload_count": 1, "offload_outside_think_count": 0}
    # completion_tokens=200, n=2 → budget 100 completion tokens/turn @ COST_GLM_OUTPUT
    out = offload.compute_turn_rewards(
        1.0,
        stats,
        completion_tokens=200,
        metadata={"completion_tokens": 200},
        lam=0.0,  # isolate cost term off
    )
    assert len(out["turn_rewards"]) == 2
    assert out["reward"] == pytest.approx(1.0)


def test_compute_turn_rewards_malformed_negative_unsolved(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_MALFORMED_PENALTY", "0.25")
    turns = [_turn(valid=True), _turn(orphan=3, opens=3)]
    stats = {"turn_costs": turns, "offload_count": 1, "offload_outside_think_count": 0}
    out = offload.compute_turn_rewards(
        0.0, stats, alpha=0.1, encourage_seek=True, malformed_pen=0.25
    )
    assert out["turn_rewards"][0] == pytest.approx(0.1)
    # Flat −β (not scaled by orphan count).
    assert out["turn_rewards"][1] == pytest.approx(-0.25)


def test_repair_incomplete_offload_second_open_as_close():
    raw = f"{offload.OFFLOAD_OPEN}4{offload.OFFLOAD_OPEN}4{offload.OFFLOAD_OPEN}"
    out = offload.repair_incomplete_offload(raw)
    assert out is not None
    assert out["n"] == 4
    assert out["repaired_raw"].startswith(f"{offload.OFFLOAD_OPEN}4{offload.OFFLOAD_CLOSE}")
    assert offload.parse_offload_directive(out["repaired_raw"]) == (4, "")


def test_repair_incomplete_offload_no_digit_defaults_zero():
    raw = f"{offload.OFFLOAD_OPEN}{offload.OFFLOAD_OPEN}x"
    out = offload.repair_incomplete_offload(raw)
    assert out is not None
    assert out["n"] == 0
    assert out["defaulted_digit"] is True
    assert offload.parse_offload_directive(out["repaired_raw"])[0] == 0


def test_repair_incomplete_offload_rejects_non_digit_payload():
    junk = f"{offload.OFFLOAD_OPEN}abc{offload.OFFLOAD_OPEN}"
    assert offload.repair_incomplete_offload(junk) is None
    multi = f"{offload.OFFLOAD_OPEN}12{offload.OFFLOAD_OPEN}"
    assert offload.repair_incomplete_offload(multi) is None
    orphan_junk = f"{offload.OFFLOAD_OPEN}4x"
    assert offload.repair_incomplete_offload(orphan_junk) is None


def test_repair_incomplete_offload_orphan_to_eos():
    raw = f"prefix {offload.OFFLOAD_OPEN}7"
    out = offload.repair_incomplete_offload(raw)
    assert out is not None
    assert out["n"] == 7
    assert out["repaired_raw"] == f"prefix {offload.OFFLOAD_OPEN}7{offload.OFFLOAD_CLOSE}"


def test_repair_skips_already_valid():
    raw = f"{offload.OFFLOAD_OPEN}3{offload.OFFLOAD_CLOSE} more"
    assert offload.repair_incomplete_offload(raw) is None


def test_compute_turn_rewards_solved_solo_uses_cost_formula():
    turns = [_turn(valid=False, small_o=50), _turn(valid=True, small_o=50, glm_o=20)]
    stats = {"turn_costs": turns}
    lam = 0.5
    out = offload.compute_turn_rewards(
        1.0, stats, completion_tokens=100, metadata={"completion_tokens": 100}, lam=lam
    )
    b_i = offload.per_turn_baseline_cost(
        n_turns=2, completion_tokens=100, metadata={"completion_tokens": 100}
    )
    for tc, r in zip(turns, out["turn_rewards"], strict=True):
        c_i = offload.turn_actual_cost(tc)
        expected = max(0.0, 1.0 - lam * (c_i / b_i))
        assert r == pytest.approx(expected)
    # Offload turn pays GLM cost → lower reward than solo on the same SLM budget.
    assert out["turn_rewards"][1] < out["turn_rewards"][0]


def test_compute_turn_rewards_repaired_no_alpha_unsolved(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    repaired = _turn(valid=True)
    repaired["repaired"] = True
    clean = _turn(valid=True)
    stats = {"turn_costs": [repaired, clean], "offload_count": 2}
    out = offload.compute_turn_rewards(
        0.0, stats, alpha=0.1, encourage_seek=True, malformed_pen=0.25
    )
    assert out["turn_rewards"][0] == pytest.approx(-0.25)
    assert out["turn_rewards"][1] == pytest.approx(0.1)


def test_compute_turn_rewards_repaired_solved_subtracts_beta():
    repaired = _turn(valid=True, small_o=10, glm_o=0)
    repaired["repaired"] = True
    stats = {"turn_costs": [repaired]}
    out = offload.compute_turn_rewards(
        1.0,
        stats,
        completion_tokens=10,
        metadata={"completion_tokens": 10},
        lam=0.0,
        malformed_pen=0.25,
    )
    assert out["turn_rewards"][0] == pytest.approx(0.75)


def test_shape_group_skips_alpha_on_repaired(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "0.1")
    monkeypatch.setenv("OFFLOAD_NO_SEEK_PENALTY", "0.1")
    a = _sample(reward=0.0, solved=0.0, oc=1)
    a.metadata["turn_costs"] = [{"valid_offload": True, "repaired": True, "outside_think": False,
                                  "orphan_open_count": 0, "malformed_count": 0, "max_open_run": 0}]
    a.metadata["turn_rewards"] = [0.0]
    a.metadata["offload_stats"] = {"turn_costs": a.metadata["turn_costs"], "offload_count": 1}
    b = _sample(reward=0.0, solved=0.0, oc=0)
    offload.shape_group_help_seeking_rewards(None, [[a, b]])
    assert a.reward == pytest.approx(-0.25)
    assert b.reward == pytest.approx(-0.1)


def test_truncate_offload_open_spam_consecutive():
    spam = offload.OFFLOAD_OPEN * 5
    cut, did = offload.truncate_offload_open_spam(spam, max_run=2, max_orphan=99)
    assert did
    assert cut == offload.OFFLOAD_OPEN
    assert offload.analyze_offload_tags(cut)["orphan_open_count"] == 1


def test_truncate_offload_open_spam_interleaved_orphans():
    raw = (
        f"{offload.OFFLOAD_OPEN}12{offload.OFFLOAD_OPEN}34"
        f"{offload.OFFLOAD_OPEN}56"
    )
    cut, did = offload.truncate_offload_open_spam(raw, max_run=99, max_orphan=2)
    assert did
    assert cut.count(offload.OFFLOAD_OPEN) == 1
    assert offload.OFFLOAD_OPEN + "12" == cut


def test_truncate_offload_open_spam_ids():
    oid, cid = 248077, 248078
    ids = [1, oid, oid, oid, 2]
    keep, did = offload.truncate_offload_open_spam_ids(
        ids, open_id=oid, close_id=cid, max_run=2, max_orphan=99
    )
    assert did and keep == 2
    ids2 = [oid, 99, oid, 99]  # two orphans
    keep2, did2 = offload.truncate_offload_open_spam_ids(
        ids2, open_id=oid, close_id=cid, max_run=99, max_orphan=2
    )
    assert did2 and keep2 == 2


def test_analyze_offload_tags_valid_and_orphan():
    raw = "think <|llm_offload|>3<|/llm_offload|> ok <|llm_offload|>"
    tags = offload.analyze_offload_tags(raw)
    assert tags["valid_count"] == 1
    assert tags["orphan_open_count"] >= 1
    assert tags["open_count"] == 2
    assert tags["close_count"] == 1


def test_compact_removes_single_orphan():
    from slime.utils.types import Sample

    turns = [_turn(orphan=1, opens=1, closes=0, small_o=5)]
    stats = {
        "turn_costs": turns,
        "offload_count": 0,
        "offload_outside_think_count": 0,
        "small_output_tokens": 5,
    }
    s = Sample(
        index=0,
        prompt="p",
        response="x",
        response_length=5,
        status=Sample.Status.COMPLETED,
        metadata={"offload_stats": stats},
    )
    remove, reason = offload.compact_should_remove_sample(s)
    assert remove
    assert reason == "orphan_open"


def test_compact_removes_orphan_spam():
    from slime.utils.types import Sample

    turns = [_turn(orphan=10, opens=10, closes=0, small_o=20) for _ in range(2)]
    stats = {
        "turn_costs": turns,
        "offload_count": 0,
        "offload_outside_think_count": 0,
        "small_output_tokens": 40,
    }
    s = Sample(
        index=0,
        prompt="p",
        response="x",
        response_length=40,
        status=Sample.Status.COMPLETED,
        metadata={"offload_stats": stats},
    )
    remove, reason = offload.compact_should_remove_sample(s)
    assert remove
    assert reason == "orphan_open"


def test_compact_keeps_paired_offload_despite_high_tag_density():
    """Valid OPEN/CLOSE pairs must not trip special_token_ratio on short SLM turns."""
    from slime.utils.types import Sample

    turns = [_turn(valid=True, small_o=3, glm_o=80, opens=1, closes=1) for _ in range(19)]
    stats = {
        "turn_costs": turns,
        "offload_count": 19,
        "offload_outside_think_count": 0,
        "small_output_tokens": 57,
    }
    s = Sample(
        index=0,
        prompt="p",
        response="x",
        response_length=57,
        status=Sample.Status.COMPLETED,
        metadata={"grading_solved": True, "offload_stats": stats},
    )
    remove, reason = offload.compact_should_remove_sample(s)
    assert not remove
    assert reason is None


def test_compact_special_ratio_uses_unmatched_marks_only():
    from slime.utils.types import Sample

    # Extra CLOSEs (no orphan OPEN): open/close ratio stays below 3.0, unmatched
    # density still trips special_token_ratio.
    extra_close = [_turn(valid=True, small_o=2, glm_o=0, opens=1, closes=1) for _ in range(2)]
    extra_close.append(
        {
            "small_prompt_tokens": 100,
            "small_output_tokens": 6,
            "glm_input_tokens": 0,
            "glm_output_tokens": 0,
            "valid_offload": False,
            "outside_think": False,
            "orphan_open_count": 0,
            "malformed_count": 4,
            "open_count": 0,
            "close_count": 4,
            "max_open_run": 0,
            "special_mark_count": 4,
        }
    )
    stats = {
        "turn_costs": extra_close,
        "offload_count": 2,
        "offload_outside_think_count": 0,
        "small_output_tokens": 10,
    }
    s = Sample(
        index=1,
        prompt="p",
        response="x",
        response_length=10,
        status=Sample.Status.COMPLETED,
        metadata={"offload_stats": stats},
    )
    remove, reason = offload.compact_should_remove_sample(s)
    assert remove
    assert reason == "special_token_ratio"


def test_compact_removes_timeout_and_max_steps():
    from slime.utils.types import Sample

    s1 = Sample(
        index=0,
        prompt="p",
        response="",
        response_length=0,
        status=Sample.Status.ABORTED,
        metadata={"abort_reason": "wall_clock_timeout", "timeout": True},
    )
    s2 = Sample(
        index=1,
        prompt="p",
        response="",
        response_length=0,
        status=Sample.Status.COMPLETED,
        metadata={"max_steps_reached": True, "offload_stats": {}},
    )
    assert offload.compact_should_remove_sample(s1)[0]
    assert offload.compact_should_remove_sample(s2)[0]


def test_compact_and_shape_group(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "0.1")
    from slime.utils.types import Sample

    good = _sample(reward=0.0, solved=0.0, oc=1)
    spam_stats = {
        "turn_costs": [_turn(orphan=10, opens=10, closes=0)],
        "offload_count": 0,
        "offload_outside_think_count": 0,
        "small_output_tokens": 20,
    }
    spam = Sample(
        index=1,
        prompt="p",
        response="x",
        response_length=20,
        reward=0.0,
        status=Sample.Status.COMPLETED,
        metadata={
            "solved": 0.0,
            "grading_solved": False,
            "empty_patch": False,
            "offload_stats": spam_stats,
        },
    )
    offload.compact_and_shape_group_help_seeking_rewards(None, [[good, spam]])
    assert good.reward == pytest.approx(0.1)
    assert spam.remove_sample is True


def test_shape_group_tmax_empty_no_scale(monkeypatch):
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "0.1")
    monkeypatch.setenv("OFFLOAD_SEEK_EMPTY_SCALE", "0.5")
    monkeypatch.setenv("OFFLOAD_NO_SEEK_PENALTY", "0.1")
    a = _sample(reward=0.0, solved=0.0, oc=1, empty_patch=True)
    a.metadata["protocol"] = "tmax"
    b = _sample(reward=0.0, solved=0.0, oc=0)
    offload.shape_group_help_seeking_rewards(None, [[a, b]])
    assert a.reward == pytest.approx(0.1)
    assert b.reward == pytest.approx(-0.1)


def test_resolved_train_config_and_dump(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_OFFLOAD", "1")
    monkeypatch.setenv("OFFLOAD_REWARD_MODE", "help_seeking")
    monkeypatch.setenv("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "1")
    monkeypatch.setenv("OFFLOAD_SEEK_ALPHA", "0.25")
    monkeypatch.setenv("OFFLOAD_EFFICIENCY_LAMBDA", "0.3")
    monkeypatch.setenv("OFFLOAD_SFT_LAMBDA", "1")
    monkeypatch.setenv("RUN_ROOT", str(tmp_path))
    offload._TRAIN_CONFIG_LOGGED = False
    cfg = offload.resolved_train_config()
    assert cfg["OFFLOAD_REWARD_MODE"] == "help_seeking"
    assert cfg["OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG"] is True
    assert cfg["OFFLOAD_SEEK_ALPHA"] == pytest.approx(0.25)
    assert cfg["OFFLOAD_EFFICIENCY_LAMBDA"] == pytest.approx(0.3)
    assert cfg["OFFLOAD_SFT_LAMBDA"] == pytest.approx(1.0)
    logged = offload.log_train_config_once(None)
    assert logged["OFFLOAD_SEEK_ALPHA"] == pytest.approx(0.25)
    path = tmp_path / "offload_config.json"
    assert path.is_file()
    import json

    disk = json.loads(path.read_text())
    assert disk["OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG"] is True
    # second call is a no-op for logging / overwrite is fine; flag stays set
    offload.log_train_config_once(None)
    assert offload._TRAIN_CONFIG_LOGGED is True


def test_turn_advantage_paints_residuals():
    import torch

    from examples.coding_agent_rl.offload_turn_advantage import compute_turn_advantages

    kl = [torch.zeros(6)]
    rollout_data = {
        "kl": kl,
        "rewards": [0.5],  # already group-demeaned A_s
        "metadata": [
            {
                "turn_rewards": [1.0, 0.0],
                "turn_token_spans": [[0, 3], [3, 6]],
            }
        ],
    }
    compute_turn_advantages(None, rollout_data)
    adv = rollout_data["advantages"][0]
    # mean_r = 0.5; residual +0.5 / -0.5
    assert adv[0].item() == pytest.approx(1.0)
    assert adv[3].item() == pytest.approx(0.0)


def test_per_turn_baseline_uses_completion_over_n():
    b = offload.per_turn_baseline_cost(n_turns=10, completion_tokens=4243, metadata={})
    expected = (
        offload._DEFAULT_BASELINE_PROMPT_TOKENS / 10 * offload.COST_GLM_INPUT
        + 424.3 * offload.COST_GLM_OUTPUT
    )
    assert b == pytest.approx(expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
