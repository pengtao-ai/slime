"""Unit tests for slime.utils.gigpo and coding-agent GiGPO helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from examples.coding_agent_rl.gigpo_anchor import (
    branch_key_from_user_task,
    content_fingerprint,
    init_anchor_obs,
    make_anchor_obs,
    normalize_tool_text,
)
from examples.coding_agent_rl.offload import group_aware_rewards
from slime.utils.gigpo import (
    build_step_group,
    compute_gigpo_outcome_advantage,
    compute_step_discounted_returns,
    episode_norm_reward,
)

NUM_GPUS = 0


def test_discounted_returns_sparse_terminal():
    rewards = [0.0, 0.0, 1.0, 0.0, 0.5]
    traj = ["a", "a", "a", "b", "b"]
    g = compute_step_discounted_returns(rewards, traj, gamma=0.5)
    # traj a: G = [0.25, 0.5, 1.0]
    assert g[0].item() == pytest.approx(0.25)
    assert g[1].item() == pytest.approx(0.5)
    assert g[2].item() == pytest.approx(1.0)
    # traj b: G = [0.25, 0.5]
    assert g[3].item() == pytest.approx(0.25)
    assert g[4].item() == pytest.approx(0.5)


def test_discounted_returns_mixed_int_str_traj_uids():
    """Regression: int rollout_id fallback + str traj_uid must not crash np.unique."""
    rewards = [0.0, 1.0, 0.5]
    traj = [12, "12:main", "12:main"]  # mixed types as seen in 20260802_152323 crash
    g = compute_step_discounted_returns(rewards, traj, gamma=0.5)
    # "12" alone; "12:main" gets discounted [1.0, 0.5] → [1.25, 0.5]
    assert g[0].item() == pytest.approx(0.0)
    assert g[1].item() == pytest.approx(1.25)
    assert g[2].item() == pytest.approx(0.5)


def test_build_step_group_mixed_index_types():
    anchors = ["a", "a", "b"]
    index = [0, "0", 1]
    uids = build_step_group(anchors, index)
    # int 0 and str "0" collapse to the same episode group after str coercion
    assert uids[0] == uids[1]
    assert uids[0] != uids[2]


def test_build_step_group_exact_match_across_turn_index():
    # Same anchor on traj A step3 and traj B step5 → same group.
    anchors = ["init", "s1", "s2", "same", "x", "same"]
    index = [0, 0, 0, 0, 0, 0]
    uids = build_step_group(anchors, index)
    assert uids[3] == uids[5]
    assert uids[0] != uids[3]
    assert len(set(uids)) == 5


def test_episode_norm_one_per_traj():
    # Two trajs, each with 2 steps sharing episode reward → mean over trajs not steps.
    ep = torch.tensor([1.0, 1.0, 0.0, 0.0])
    index = [0, 0, 0, 0]
    traj = ["t0", "t0", "t1", "t1"]
    masks = [torch.ones(3) for _ in range(4)]
    adv = episode_norm_reward(ep, masks, index, traj, remove_std=True, compute_mean_std_cross_steps=False)
    # mean of {1.0, 0.0} = 0.5 → adv = +0.5 / -0.5
    assert adv[0][0].item() == pytest.approx(0.5)
    assert adv[2][0].item() == pytest.approx(-0.5)


def test_gigpo_joint_advantage_shapes():
    ep = torch.tensor([1.0, 0.0])
    step = torch.tensor([1.0, 0.0])
    masks = [torch.ones(4), torch.ones(2)]
    anchors = ["init", "init"]
    index = [0, 0]
    traj = ["a", "b"]
    adv, ret = compute_gigpo_outcome_advantage(
        ep, step, masks, anchors, index, traj, step_advantage_w=1.0, mode="mean_norm"
    )
    assert len(adv) == 2
    assert adv[0].shape == (4,)
    assert adv[1].shape == (2,)
    # A_E: +0.5 / -0.5; A_S same; joint ±1.0
    assert adv[0][0].item() == pytest.approx(1.0)
    assert adv[1][0].item() == pytest.approx(-1.0)


def test_anchor_init_and_read_match():
    a = init_anchor_obs("auth0_pr671", "# Title\nPAR fails")
    b = init_anchor_obs("auth0_pr671", "# Title\nPAR fails")
    assert a == b
    r1 = make_anchor_obs(
        instance_id="auth0_pr671",
        branch_key="main",
        tool_name="Read",
        tool_input={"file_path": "/workspace/auth0-python/PROBLEM_STATEMENT.md"},
        tool_result_text="1\t# Title\n2\tPAR fails\n",
    )
    r2 = make_anchor_obs(
        instance_id="auth0_pr671",
        branch_key="main",
        tool_name="Read",
        tool_input={"file_path": "/workspace/auth0-python/PROBLEM_STATEMENT.md"},
        tool_result_text="1\t# Title\n2\tPAR fails\n",
    )
    assert r1 == r2
    # Different file → different anchor
    r3 = make_anchor_obs(
        instance_id="auth0_pr671",
        branch_key="main",
        tool_name="Read",
        tool_input={"file_path": "/workspace/auth0-python/other.py"},
        tool_result_text="1\t# Title\n2\tPAR fails\n",
    )
    assert r1 != r3


def test_normalize_strips_agent_id():
    t = normalize_tool_text("hello agentId: abc123 world duration_ms: 12")
    assert "agentId" not in t
    assert "duration_ms" not in t
    assert "hello" in t and "world" in t


def test_branch_key_main_vs_sub():
    ep = "Read PROBLEM_STATEMENT.md and resolve the issue."
    assert branch_key_from_user_task(ep, episode_user=ep) == "main"
    assert branch_key_from_user_task("Find class PushedAuthorizationRequests", episode_user=ep).startswith("sub:")


def test_group_aware_all_fail_offload():
    group = [
        {"solved": 0.0, "stats": {"offload_count": 0, "offload_outside_think_count": 0}, "empty_patch": True},
        {"solved": 0.0, "stats": {"offload_count": 1, "offload_outside_think_count": 0}, "empty_patch": False},
    ]
    rs = group_aware_rewards(group, alpha=0.1, empty_scale=0.5)
    assert rs[0] == pytest.approx(0.0)
    assert rs[1] == pytest.approx(0.1)


def test_group_aware_unique_glm_solver():
    group = [
        {"solved": 1.0, "stats": {"offload_count": 1, "offload_outside_think_count": 0, "small_prompt_tokens": 10, "small_output_tokens": 10, "glm_input_tokens": 10, "glm_output_tokens": 10}},
        {"solved": 0.0, "stats": {"offload_count": 0, "offload_outside_think_count": 0}},
        {"solved": 0.0, "stats": {"offload_count": 0, "offload_outside_think_count": 0}},
    ]
    report = {}
    rs = group_aware_rewards(group, lam=0.0, unique_bonus=0.15, report=report)
    assert rs[0] > rs[1]
    # unique bonus applied on top of cost_aware base (~1.0 with lam=0)
    assert rs[0] == pytest.approx(1.0 + 0.15, abs=0.05)
    assert report["unique_bonus_applied"] is True
    assert report["n_solved"] == 1
    assert report["n_offload"] == 1


def test_group_aware_no_offload_solver_bonus():
    group = [
        {"solved": 1.0, "stats": {"offload_count": 0, "offload_outside_think_count": 0, "small_prompt_tokens": 10, "small_output_tokens": 10, "glm_input_tokens": 0, "glm_output_tokens": 0}},
        {"solved": 1.0, "stats": {"offload_count": 2, "offload_outside_think_count": 0, "small_prompt_tokens": 10, "small_output_tokens": 10, "glm_input_tokens": 100, "glm_output_tokens": 100}},
    ]
    rs = group_aware_rewards(group, lam=0.0, no_offload_bonus_v=0.15)
    assert rs[0] > rs[1]


def test_content_fingerprint_read_ignores_line_numbers():
    a = content_fingerprint("Read", "1\thello\n2\tworld\n")
    b = content_fingerprint("Read", "10\thello\n11\tworld\n")
    assert a == b


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
