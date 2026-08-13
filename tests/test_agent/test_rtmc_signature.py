"""Unit tests for RTMC state / action signatures."""

from __future__ import annotations

import hashlib

import pytest

from slime.agent.rtmc_signature import (
    StateSignatureBuilder,
    action_signature,
    content_hash,
    flatten_turns,
    group_by_state,
    normalize_path,
    step_signatures,
    view_buckets,
)

NUM_GPUS = 0


def _md5_4(old: str, new: str) -> str:
    return hashlib.md5(f"{old}{new}".encode()).hexdigest()[:4]


def test_empty_state_is_flags_only():
    assert StateSignatureBuilder().snapshot() == "(test_fail_count=0,test_pass_count=0,think_count=0)"


def test_view_buckets():
    assert view_buckets(200, 299) == "V[2]"
    assert view_buckets(100, 299) == "V[1-2]"
    assert view_buckets(1, 50) == "V[0]"


def test_content_hash_is_md5_prefix():
    assert content_hash("old", "new") == _md5_4("old", "new")


def test_full_read_then_edit_state():
    old = "x = 1\n"
    new = "x = 2\n"
    digest = _md5_4(old, new)
    hist = [
        {"name": "Read", "arguments": {"file_path": "/repo/core.py"}, "result": "1\tx = 1\n"},
        {
            "name": "Edit",
            "arguments": {"file_path": "/repo/core.py", "old_string": old, "new_string": new},
            "result": "ok",
        },
    ]
    b = StateSignatureBuilder()
    assert b.state_signature(hist, 0) == "(test_fail_count=0,test_pass_count=0,think_count=0)"
    assert b.state_signature(hist, 1) == "/repo/core.py:Vf|(test_fail_count=0,test_pass_count=0,think_count=0)"
    assert (
        b.state_signature(hist, 2)
        == f"/repo/core.py:M:{digest},Vf|(test_fail_count=0,test_pass_count=0,think_count=0)"
    )


def test_order_invariant_and_dedup():
    a = {"name": "Read", "arguments": {"file_path": "/b.py"}, "result": "1\tok\n"}
    c = {"name": "Read", "arguments": {"file_path": "/a.py"}, "result": "1\tok\n"}
    again = {"name": "Read", "arguments": {"file_path": "/b.py"}, "result": "1\tok\n"}
    left = StateSignatureBuilder()
    right = StateSignatureBuilder()
    for act in (a, c, again):
        left.apply(act)
    for act in (c, a):
        right.apply(act)
    assert left.snapshot() == right.snapshot()
    assert left.snapshot().startswith("/a.py:Vf|/b.py:Vf|")


def test_partial_view_and_search():
    b = StateSignatureBuilder()
    b.apply(
        {
            "name": "Read",
            "arguments": {"file_path": "lib.py", "offset": "200", "limit": "50"},
            "result": "200\tfoo\n249\tbar\n",
        }
    )
    b.apply({"name": "Grep", "arguments": {"path": "lib.py", "pattern": "foo"}, "result": "lib.py:200:foo"})
    assert b.snapshot() == "lib.py:S,V[2]|(test_fail_count=0,test_pass_count=0,think_count=0)"


def test_insert_vs_modify_and_create():
    ins = _md5_4("", "print(1)\n")
    b = StateSignatureBuilder()
    b.apply({"name": "Write", "arguments": {"file_path": "new.py", "contents": "print(1)\n"}})
    b.apply(
        {
            "name": "Edit",
            "arguments": {"file_path": "new.py", "old_string": "", "new_string": "print(1)\n"},
        }
    )
    assert b.snapshot() == f"new.py:C,I:{ins}|(test_fail_count=0,test_pass_count=0,think_count=0)"


def test_action_signature_formats():
    assert action_signature({"name": "think", "text": "hmm"}) == "think"
    assert action_signature({"name": "finish"}) == "finish"
    assert action_signature({"name": "Read", "arguments": {"file_path": "core.py"}}) == "view:full@core.py"
    assert (
        action_signature({"name": "Read", "arguments": {"file_path": "core.py", "offset": "100", "limit": "200"}})
        == "view:partial[1-2]@core.py"
    )
    old, new = "a", "b"
    digest = _md5_4(old, new)
    assert (
        action_signature(
            {"name": "Edit", "arguments": {"file_path": "core.py", "old_string": old, "new_string": new}}
        )
        == f"modify:replace:{digest}@core.py"
    )


def test_state_excludes_current_action():
    hist = [
        {"name": "think", "text": "plan"},
        {"name": "Read", "arguments": {"file_path": "a.py"}, "result": "1\nx\n"},
    ]
    steps = step_signatures(hist)
    assert steps[0]["state"].endswith("think_count=0)")
    assert steps[0]["action"] == "think"
    assert steps[1]["state"].endswith("think_count=1)")
    assert "a.py" not in steps[1]["state"]
    assert steps[1]["action"] == "view:full@a.py"


def test_test_flags_and_bash_search():
    b = StateSignatureBuilder()
    b.apply({"name": "Bash", "arguments": {"command": "ls /repo"}, "result": "core.py\n"})
    b.apply({"name": "Bash", "arguments": {"command": "pytest tests/test_x.py"}, "result": "1 failed, 2 passed"})
    assert b.flags["test_fail_count"] == 1
    assert b.flags["test_pass_count"] == 0
    assert b.files["."] == {"S"}
    assert b.snapshot().startswith(".:S|")


def test_flatten_turns_and_group():
    turns = [
        {
            "tool_calls": [{"name": "Read", "arguments": {"file_path": "a.py"}}],
            "tool_results": ["1\nok\n"],
        },
        {"think": "", "text": "done", "tool_calls": [], "tool_results": []},
    ]
    flat = flatten_turns(turns)
    assert [a["name"] for a in flat] == ["Read", "finish"]
    groups = group_by_state({"r0": flat, "r1": flat})
    init = "(test_fail_count=0,test_pass_count=0,think_count=0)"
    assert set(groups[init]) == {("r0", 0), ("r1", 0)}


def test_normalize_path():
    assert normalize_path("  /repo/./a/../b.py\n") == "/repo/b.py"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
