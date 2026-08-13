"""Tests for coarse rollout StateSignature grouping."""

from __future__ import annotations

import pytest

from slime.agent.coarse_state_signature import (
    CoarseStateSignatureBuilder,
    coarse_action_signature,
    coarse_step_signatures,
    relative_path,
    turn_start_states,
)

NUM_GPUS = 0


def test_relative_path_strips_workdir():
    assert relative_path("/home/user/app/main.py") == "app/main.py"
    assert relative_path("/home/user/nginx/nginx.conf") == "nginx/nginx.conf"
    assert relative_path("/home/user/app", as_dir=True) == "app/"
    assert relative_path("src/utils.py") == "src/utils.py"
    assert relative_path("tests/utils.py") == "tests/utils.py"
    assert relative_path("/home/user/.harness", as_dir=True) == ".harness/"
    assert relative_path("/app/test_audio.wav") == "/app/test_audio.wav"


def test_empty_state():
    assert CoarseStateSignatureBuilder().snapshot() == "TEST:{pass=0,fail=0}"


def test_order_and_content_do_not_matter():
    a = [
        {"name": "Read", "arguments": {"file_path": "/home/user/app/main.py"}},
        {"name": "Read", "arguments": {"file_path": "/home/user/nginx/nginx.conf"}},
        {
            "name": "Edit",
            "arguments": {
                "file_path": "/home/user/app/main.py",
                "old_string": "if x is None: return None",
                "new_string": "if x is None: return None",
            },
        },
    ]
    b = [
        {"name": "Read", "arguments": {"file_path": "/home/user/nginx/nginx.conf"}},
        {"name": "Read", "arguments": {"file_path": "/home/user/app/main.py"}},
        {
            "name": "Edit",
            "arguments": {
                "file_path": "/home/user/app/main.py",
                "old_string": "if not x: return None",
                "new_string": "if not x: return None",
            },
        },
    ]
    left, right = CoarseStateSignatureBuilder(), CoarseStateSignatureBuilder()
    for act in a:
        left.apply(act)
    for act in b:
        right.apply(act)
    assert left.snapshot() == right.snapshot()
    assert left.snapshot() == (
        "M:app/main.py | V:app/main.py | V:nginx/nginx.conf | TEST:{pass=0,fail=0}"
    )


def test_duplicate_ops_once():
    b = CoarseStateSignatureBuilder()
    for _ in range(3):
        b.apply({"name": "Read", "arguments": {"file_path": "/home/user/app/main.py"}})
        b.apply(
            {
                "name": "Edit",
                "arguments": {"file_path": "/home/user/app/main.py", "old_string": "a", "new_string": "b"},
            }
        )
    assert b.snapshot() == "M:app/main.py | V:app/main.py | TEST:{pass=0,fail=0}"


def test_search_file_vs_dir():
    b = CoarseStateSignatureBuilder()
    b.apply({"name": "Bash", "arguments": {"command": "grep parse /home/user/app/main.py"}})
    b.apply({"name": "Bash", "arguments": {"command": "ls /home/user/app"}})
    b.apply({"name": "Bash", "arguments": {"command": "ls /home/user/.harness"}})
    assert "S:app/main.py" in b.ops
    assert "S:app/" in b.ops
    assert "S:.harness/" in b.ops


def test_execute_does_not_change_state():
    b = CoarseStateSignatureBuilder()
    b.apply({"name": "Read", "arguments": {"file_path": "/home/user/check.py"}})
    before = b.snapshot()
    assert coarse_action_signature({"name": "Bash", "arguments": {"command": "python3 /home/user/check.py"}}) == "EXECUTE"
    b.apply({"name": "Bash", "arguments": {"command": "gunicorn -b unix:/tmp/x.sock main:app"}})
    b.apply({"name": "Bash", "arguments": {"command": "curl localhost:8080"}})
    b.apply({"name": "Bash", "arguments": {"command": "python3 /home/user/check.py"}})
    assert b.snapshot() == before


def test_test_counts_and_create():
    b = CoarseStateSignatureBuilder()
    b.apply({"name": "Write", "arguments": {"file_path": "/home/user/check.py"}})
    b.apply({"name": "Bash", "arguments": {"command": "pytest tests/test_x.py"}, "result": "1 failed"})
    b.apply({"name": "Bash", "arguments": {"command": "pytest tests/test_x.py"}, "result": "3 passed"})
    assert b.snapshot() == "C:check.py | TEST:{pass=1,fail=1}"


def test_state_excludes_current_action():
    hist = [
        {"name": "Read", "arguments": {"file_path": "/home/user/app/main.py"}},
        {"name": "Edit", "arguments": {"file_path": "/home/user/app/main.py", "old_string": "a", "new_string": "b"}},
    ]
    steps = coarse_step_signatures(hist)
    assert steps[0]["state"] == "TEST:{pass=0,fail=0}"
    assert steps[0]["action"] == "V:app/main.py"
    assert steps[1]["state"] == "V:app/main.py | TEST:{pass=0,fail=0}"
    assert steps[1]["action"] == "M:app/main.py"
    assert "M:app/main.py" not in steps[1]["state"]


def test_turn_start_states():
    turns = [
        {"tool_calls": [{"name": "Read", "arguments": {"file_path": "/home/user/app/main.py"}}], "tool_results": [""]},
        {
            "tool_calls": [
                {
                    "name": "Edit",
                    "arguments": {"file_path": "/home/user/app/main.py", "old_string": "a", "new_string": "b"},
                }
            ],
            "tool_results": ["ok"],
        },
    ]
    states = turn_start_states(turns)
    assert states[0] == "TEST:{pass=0,fail=0}"
    assert states[1] == "V:app/main.py | TEST:{pass=0,fail=0}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
