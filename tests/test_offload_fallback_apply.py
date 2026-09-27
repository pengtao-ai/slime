"""apply_offload_if_needed: fallback CLOSE / bad payload vs orphan OPEN."""

from __future__ import annotations

import asyncio

import pytest

from examples.coding_agent_rl import offload
from slime.agent.adapters.common import Reply, Session
from slime.agent.trajectory import TurnRecord

NUM_GPUS = 0


@pytest.fixture(autouse=True)
def _enable_offload(monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_OFFLOAD", "1")
    monkeypatch.setenv("SLIME_OFFLOAD_EMBED_IN_TRAJECTORY", "0")


def _apply(raw: str, monkeypatch, *, glm_calls: list):
    async def _fake_chat(*_a, **kwargs):
        glm_calls.append(kwargs)
        return "glm-out", "glm-think", {"prompt_tokens": 10, "completion_tokens": 5}, []

    monkeypatch.setattr(offload, "call_remote_chat", _fake_chat)

    async def run_case():
        session = Session()
        turn = TurnRecord(prompt_ids=[1, 2], output_ids=[3, 4], finish_reason="stop")
        reply = Reply(
            manager_message={"role": "assistant", "content": raw},
            finish_reason="stop",
            wire=None,
        )
        out = await offload.apply_offload_if_needed(
            reply,
            raw_output=raw,
            translated=[{"role": "user", "content": "task"}],
            turn=turn,
            session=session,
            sid="t-fallback",
            tokenizer=None,
        )
        return out, session

    return asyncio.run(run_case())


def test_orphan_close_calls_glm_n3_no_alpha(monkeypatch):
    calls: list = []
    raw = f"partial {offload.OFFLOAD_CLOSE}"
    reply, session = _apply(raw, monkeypatch, glm_calls=calls)
    assert len(calls) == 1
    assert calls[0].get("reasoning_effort") == "low"  # N=3
    stats = session.offload_stats
    assert stats["offload_count"] == 1
    assert stats["last_offload_n"] == 3
    tc = stats["turn_costs"][-1]
    assert tc["valid_offload"] is False
    assert tc["fallback_offload"] is True
    assert int(tc["malformed_count"]) >= 1
    assert "glm-out" in (reply.manager_message.get("content") or "")


def test_bad_payload_calls_glm_n3(monkeypatch):
    calls: list = []
    raw = f"x {offload.OFFLOAD_OPEN}ab{offload.OFFLOAD_CLOSE}"
    _, session = _apply(raw, monkeypatch, glm_calls=calls)
    assert len(calls) == 1
    assert session.offload_stats["last_offload_n"] == 3
    tc = session.offload_stats["turn_costs"][-1]
    assert tc["valid_offload"] is False
    assert tc["fallback_offload"] is True


def test_orphan_open_skips_glm(monkeypatch):
    calls: list = []
    raw = f"x {offload.OFFLOAD_OPEN}"
    _, session = _apply(raw, monkeypatch, glm_calls=calls)
    assert calls == []
    assert int(session.offload_stats.get("offload_count", 0) or 0) == 0
    tc = session.offload_stats["turn_costs"][-1]
    assert int(tc["orphan_open_count"]) >= 1


def test_outside_think_digit_span_skips_glm(monkeypatch):
    calls: list = []
    raw = f"</think>\n{offload.OFFLOAD_OPEN}4{offload.OFFLOAD_CLOSE}"
    _, session = _apply(raw, monkeypatch, glm_calls=calls)
    assert calls == []
    assert int(session.offload_stats.get("offload_outside_think_count", 0) or 0) == 1
    tc = session.offload_stats["turn_costs"][-1]
    assert tc["outside_think"] is True


def test_valid_in_think_still_earns_alpha_flag(monkeypatch):
    calls: list = []
    raw = f"plan {offload.OFFLOAD_OPEN}7{offload.OFFLOAD_CLOSE}"
    _, session = _apply(raw, monkeypatch, glm_calls=calls)
    assert len(calls) == 1
    assert calls[0].get("reasoning_effort") == "max"
    tc = session.offload_stats["turn_costs"][-1]
    assert tc["valid_offload"] is True
    assert tc.get("fallback_offload") is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
