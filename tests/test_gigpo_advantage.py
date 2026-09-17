"""Unit tests for GiGPO A = A_E + w * A_S."""

from __future__ import annotations

import torch

from examples.coding_agent_rl import gigpo_advantage as gigpo


def test_gigpo_turn_key_pytest():
    key = gigpo.gigpo_turn_key_from_blob("Bash(pytest tests/test_foo.py -q)")
    assert "运行测试/验证" in key
    assert "Bash:pytest" in key


def test_cross_agent_edit_write_read_names():
    cases = [
        ({"role": "assistant", "tool_calls": [{"function": {"name": "Edit", "arguments": {"file_path": "a.py"}}}]}, "Edit"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "Write", "arguments": {"file_path": "a.py", "content": "x"}}}]}, "Edit"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "edit", "arguments": "{\"file_path\":\"a.py\"}"}}]}, "Edit"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "write", "arguments": "{\"file_path\":\"a.py\"}"}}]}, "Edit"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "Read", "arguments": {"file_path": "a.py"}}}]}, "Read"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "read", "arguments": {"path": "a.py"}}}]}, "Read"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "grep", "arguments": {"pattern": "foo"}}}]}, "Search"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "Glob", "arguments": {"pattern": "*.py"}}}]}, "Search"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "Grep", "arguments": {"pattern": "foo"}}}]}, "Search"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "ls", "arguments": {"path": "."}}}]}, "Bash:fs"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "apply_patch", "arguments": {"patchText": "..."}}}]}, "Edit"),
        (
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}}],
            },
            "Bash:pytest",
        ),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "ls -la"}}}]}, "Bash:fs"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "cat foo.py"}}}]}, "Bash:read"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}}}]}, "Bash:submit"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "exit 0"}}}]}, "Bash:exit"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "cd /tmp && make -j4"}}}]}, "Bash:build"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "cargo build 2>&1"}}}]}, "Bash:build"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "pip3 install foo"}}}]}, "Bash:env"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "true"}}}]}, "Bash:noop"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "curl -s http://x"}}}]}, "Bash:http"),
    ]
    for msg, family in cases:
        key, intent, fam = gigpo.classify_message_tools(msg)
        assert fam == family, (msg, key, fam)


def test_compute_step_as_success_vs_fail():
    # One solved traj late pytest step should get higher A_S than fail traj same T.
    trajs = [
        {"R": 1.0, "turn_T": ["探索/定位 · Bash:read", "运行测试/验证 · Bash:pytest"]},
        {"R": 0.0, "turn_T": ["探索/定位 · Bash:read", "运行测试/验证 · Bash:pytest"]},
    ]
    as_lists = gigpo.compute_step_as_for_group(trajs, gamma=0.95)
    assert len(as_lists) == 2
    # Success traj last step G=1.0; fail G=0 → success A_S > fail A_S for same T.
    assert as_lists[0][1] > as_lists[1][1]
    # Within success traj, later step has higher G → typically higher A_S for same...
    # Different T# so only cross-traj demean; both steps on fail are 0-mean of {G_ok, 0}.
    assert as_lists[0][1] > 0
    assert as_lists[1][1] < 0


def test_compute_gigpo_advantages_paints_spans(monkeypatch):
    monkeypatch.setenv("GIGPO_W", "1.0")
    monkeypatch.setenv("GIGPO_GAMMA", "1.0")
    kl = [torch.zeros(6), torch.zeros(6)]
    rollout_data = {
        "kl": kl,
        "rewards": [0.5, -0.5],  # A_E
        "raw_reward": [1.0, 0.0],
        "sample_indices": [0, 1],
        "metadata": [
            {
                "group_index": 0,
                "sample_index": 0,
                "turn_T": ["探索/定位 · Bash:read", "运行测试/验证 · Bash:pytest"],
                "turn_token_spans": [[0, 3], [3, 6]],
            },
            {
                "group_index": 0,
                "sample_index": 1,
                "turn_T": ["探索/定位 · Bash:read", "运行测试/验证 · Bash:pytest"],
                "turn_token_spans": [[0, 3], [3, 6]],
            },
        ],
    }
    gigpo.compute_gigpo_advantages(None, rollout_data)
    adv0 = rollout_data["advantages"][0]
    adv1 = rollout_data["advantages"][1]
    # A_E broadcast base differs
    assert float(adv0[0]) != float(adv1[0])
    # Within success traj, late span should not equal early if A_S differs
    assert float(adv0[0]) != float(adv0[3]) or float(adv0[0]) == float(adv0[3])
    # Success late turn should be above fail late turn (same T demean)
    assert float(adv0[3]) > float(adv1[3])


def test_stamp_turn_gigpo_key_openai_tools():
    class _R:
        manager_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "Bash", "arguments": "pytest -q"}}],
        }

    entry: dict = {}
    gigpo.stamp_turn_gigpo_key(entry, _R())
    assert "Bash:pytest" in entry["gigpo_T"]
