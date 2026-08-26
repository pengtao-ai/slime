"""CPU tests for GiGPO math and ScaleSWE / Tmax grouping."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from examples.coding_agent_rl import gigpo
from slime.utils.gigpo import compute_step_discounted_returns, episode_norm_reward, step_norm_reward

NUM_GPUS = 0


def _turn(*, tools, intent, diff_key="", edit_key="", empty=True):
    return {
        "turn": 0,
        "diff_key": diff_key,
        "empty_diff": empty,
        "diff_has_test": False,
        "edit_key": edit_key,
        "tools": tools,
        "tools_str": ", ".join(tools) or "无 tool",
        "intent": intent,
    }


def _sample(*, protocol, instance_id, group_index, reward, turns, index=0):
    numbered = []
    for i, t in enumerate(turns):
        row = dict(t)
        row["turn"] = i
        numbered.append(row)
    return SimpleNamespace(
        reward=reward,
        index=index,
        group_index=group_index,
        remove_sample=False,
        metadata={
            "protocol": protocol,
            "instance_id": instance_id,
            "episode_reward": reward,
        },
        train_metadata={
            "objective": "grpo",
            "protocol": protocol,
            "instance_id": instance_id,
            "gigpo_turns": numbered,
            "turn_token_spans": [[i * 2, (i + 1) * 2] for i in range(len(numbered))],
        },
    )


def test_discounted_returns_last_reward_only():
    g = compute_step_discounted_returns([0.0, 0.0, 1.0], ["a", "a", "a"], gamma=0.5)
    assert g.tolist() == pytest.approx([0.25, 0.5, 1.0])


def test_episode_norm_separates_groups():
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
    masks = [torch.ones(1) for _ in range(4)]
    index = ["swe::x", "swe::x", "tmax::x", "tmax::x"]
    traj = ["t0", "t1", "t2", "t3"]
    a = episode_norm_reward(rewards, masks, index, traj, remove_std=True, compute_mean_std_cross_steps=False)
    vals = [float(t.reshape(-1)[0]) for t in a]
    assert vals[0] == pytest.approx(0.5)
    assert vals[1] == pytest.approx(-0.5)
    assert vals[2] == pytest.approx(0.0)
    assert vals[3] == pytest.approx(0.0)


def test_step_norm_singleton_is_zero():
    g = torch.tensor([0.9, 0.1])
    masks = [torch.ones(1), torch.ones(1)]
    a = step_norm_reward(g, masks, ["u1", "u2"], remove_std=True)
    assert [float(t.reshape(-1)[0]) for t in a] == pytest.approx([0.0, 0.0])


def test_compact_and_split_scaleswe_by_git_diff():
    records = [
        {"turn_index": 0, "git_diff": "", "tool_calls": [{"name": "Read", "path": "a.py"}]},
        {"turn_index": 1, "git_diff": "", "tool_calls": [{"name": "Read", "path": "b.py"}]},
        {
            "turn_index": 2,
            "git_diff": "diff --git a/a.py b/a.py\n",
            "tool_calls": [{"name": "Edit", "path": "a.py"}],
        },
        {
            "turn_index": 3,
            "git_diff": "diff --git a/a.py b/a.py\n",
            "tool_calls": [{"name": "Bash", "kind": "pytest"}],
        },
    ]
    turns = gigpo.compact_turns("scaleswe", records)
    assert gigpo.segment_ids(turns, "scaleswe") == [0, 0, 1, 1]
    assert turns[0]["intent"] == "探索定位"
    assert turns[2]["intent"] == "实现修复"
    assert turns[3]["intent"] == "验证修复"


def test_compact_and_split_tmax_by_edit():
    records = [
        {"turn_index": 0, "git_diff": "", "tool_calls": [{"name": "Read", "path": "main.c"}]},
        {"turn_index": 1, "git_diff": "", "tool_calls": [{"name": "Bash", "kind": "cat"}]},
        {"turn_index": 2, "git_diff": "", "tool_calls": [{"name": "Edit", "path": "main.c"}]},
        {"turn_index": 3, "git_diff": "", "tool_calls": [{"name": "Bash", "kind": "build"}]},
    ]
    turns = gigpo.compact_turns("tmax", records)
    assert gigpo.segment_ids(turns, "tmax") == [0, 0, 1, 1]
    assert turns[0]["intent"] == "探索定位"
    assert turns[2]["intent"] == "实现修复"
    assert turns[3]["intent"] == "验证运行"


def test_scaleswe_and_tmax_episode_groups_are_separate():
    explore = _turn(tools=["Read"], intent="探索定位")
    swe = _sample(protocol="scaleswe", instance_id="same", group_index=0, reward=1.0, turns=[explore], index=0)
    tmax = _sample(protocol="tmax", instance_id="same", group_index=0, reward=1.0, turns=[explore], index=1)
    gigpo.assign_gigpo_to_samples([swe, tmax], gamma=0.95, step_w=1.0)
    # Each protocol is a singleton group → A_E = R (verl mean=0 when n=1), not mixed to 0.
    assert swe.train_metadata["gigpo_turn_advantages"][0] == pytest.approx(1.0)
    assert tmax.train_metadata["gigpo_turn_advantages"][0] == pytest.approx(1.0)


def test_step_group_by_intent_and_tool():
    t_explore = _turn(tools=["Read"], intent="探索定位", diff_key="")
    t_edit = _turn(tools=["Edit"], intent="实现修复", diff_key="d1", empty=False)
    t_explore_only = _turn(tools=["Read"], intent="探索定位", diff_key="")
    a = _sample(
        protocol="scaleswe",
        instance_id="pr1",
        group_index=1,
        reward=1.0,
        turns=[t_explore, t_edit],
        index=0,
    )
    b = _sample(
        protocol="scaleswe",
        instance_id="pr1",
        group_index=1,
        reward=0.0,
        turns=[t_explore_only],
        index=1,
    )
    gigpo.assign_gigpo_to_samples([a, b], gamma=0.5, step_w=1.0)
    # Explore turns share intent+tool: G is 0.5 vs 0.0 → A_S ±0.25
    a_rows = a.metadata["gigpo_step_rows"]
    b_rows = b.metadata["gigpo_step_rows"]
    assert a_rows[0]["intent"] == b_rows[0]["intent"] == "探索定位"
    assert a_rows[0]["step_uid"] == b_rows[0]["step_uid"]
    assert a_rows[0]["A_S"] == pytest.approx(0.25)
    assert b_rows[0]["A_S"] == pytest.approx(-0.25)
    # Edit turn is a singleton step group → A_S = 0
    assert a_rows[1]["A_S"] == pytest.approx(0.0)


def test_compute_advantages_paints_spans():
    kl = [torch.zeros(4)]
    rollout_data = {
        "kl": kl,
        "rewards": [0.0],
        "metadata": [
            {
                "gigpo_turn_advantages": [0.5, -0.25],
                "turn_token_spans": [[0, 2], [2, 4]],
            }
        ],
    }
    gigpo.compute_advantages(None, rollout_data)
    assert rollout_data["advantages"][0].tolist() == pytest.approx([0.5, 0.5, -0.25, -0.25])


def test_sft_rows_get_zero_advantage():
    kl = [torch.ones(3)]
    rollout_data = {
        "kl": kl,
        "rewards": [1.0],
        "metadata": [{"objective": "sft", "gigpo_turn_advantages": [9.0]}],
    }
    gigpo.compute_advantages(None, rollout_data)
    assert rollout_data["advantages"][0].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_parse_manager_tool_calls():
    msg = {
        "role": "assistant",
        "tool_calls": [
            {"type": "function", "function": {"name": "Bash", "arguments": {"command": "cd /tmp && pytest -q"}}},
            {"type": "function", "function": {"name": "Edit", "arguments": {"file_path": "src/a.py"}}},
        ],
    }
    calls = gigpo.parse_manager_tool_calls(msg)
    assert calls[0]["name"] == "Bash"
    assert calls[0]["kind"] == "pytest"
    assert calls[1] == {"name": "Edit", "path": "src/a.py"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
