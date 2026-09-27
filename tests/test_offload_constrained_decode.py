"""Offload Free + ebnf constrained decode (call_sglang_generate two-step)."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from examples.coding_agent_rl import offload
from slime.agent.adapters import common as common_mod
from tests.test_agent._fakes import FakeSGLangServer

NUM_GPUS = 0

OPEN_ID = 248077
CLOSE_ID = 248078
DIGIT_1 = 16


@pytest.fixture(autouse=True)
def _offload_env(monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_OFFLOAD", "1")
    monkeypatch.setenv("OFFLOAD_CONSTRAINED_DECODE", "1")
    monkeypatch.setenv("OFFLOAD_OPEN_TOKEN_ID", str(OPEN_ID))
    monkeypatch.setenv("OFFLOAD_CLOSE_TOKEN_ID", str(CLOSE_ID))
    monkeypatch.setenv("OFFLOAD_CLOSE_LOGIT_BIAS", "-100")


def test_apply_free_phase_bans_close_and_stops_on_open():
    sp = offload.apply_free_phase_constrained_sampling(
        {"stop_token_ids": [248046, 248044, CLOSE_ID], "max_new_tokens": 32}
    )
    assert CLOSE_ID not in sp["stop_token_ids"]
    assert OPEN_ID in sp["stop_token_ids"]
    assert sp["logit_bias"][str(CLOSE_ID)] == pytest.approx(-100.0)
    assert "ebnf" not in sp


def test_apply_ebnf_phase_digit_then_close():
    sp = offload.apply_ebnf_phase_constrained_sampling(
        {"stop_token_ids": [248046, OPEN_ID], "max_new_tokens": 100, "logit_bias": {"1": 1.0}}
    )
    assert OPEN_ID not in sp["stop_token_ids"]
    assert CLOSE_ID in sp["stop_token_ids"]
    assert "logit_bias" not in sp
    assert sp["ebnf"] == offload.offload_digit_close_ebnf()
    assert sp["max_new_tokens"] == 4
    assert "[0-9]" in sp["ebnf"] and offload.OFFLOAD_CLOSE in sp["ebnf"]


def test_constrained_decode_disabled_by_flag(monkeypatch):
    monkeypatch.setenv("OFFLOAD_CONSTRAINED_DECODE", "0")
    assert offload.constrained_decode_enabled() is False


def test_call_sglang_two_step_merges_open_digit_close(monkeypatch):
    async def run_case():
        # Free returns ... OPEN; ebnf returns digit + CLOSE.
        turns = [
            [(-0.1, 101), (-0.2, OPEN_ID)],
            [(-0.3, DIGIT_1), (-0.01, CLOSE_ID)],
        ]
        async with FakeSGLangServer(turns) as sglang:
            adapter = SimpleNamespace(
                logger=logging.getLogger("test_offload_cd"),
                log_prefix="test",
                sglang_url=sglang.url,
                max_token_keys=("max_tokens",),
                stop_keys=("stop",),
            )
            session = common_mod.Session(
                sampling_defaults={
                    "temperature": 0.0,
                    "stop_token_ids": [248046, 248044, CLOSE_ID],
                    "max_new_tokens": 32,
                },
                max_context_tokens=0,
            )
            turn = await common_mod.call_sglang_generate(
                [1, 2, 3],
                session,
                {"max_tokens": 32},
                adapter=adapter,
                session_id="sid-cd",
            )

        assert len(sglang.requests) == 2
        free_sp = sglang.requests[0]["sampling_params"]
        assert free_sp["logit_bias"][str(CLOSE_ID)] == pytest.approx(-100.0)
        assert OPEN_ID in free_sp["stop_token_ids"]
        assert CLOSE_ID not in free_sp["stop_token_ids"]
        assert "ebnf" not in free_sp

        ebnf_sp = sglang.requests[1]["sampling_params"]
        assert "ebnf" in ebnf_sp
        assert CLOSE_ID in ebnf_sp["stop_token_ids"]
        assert sglang.requests[1]["input_ids"] == [1, 2, 3, 101, OPEN_ID]

        assert turn.output_ids == [101, OPEN_ID, DIGIT_1, CLOSE_ID]
        assert turn.output_log_probs == pytest.approx([-0.1, -0.2, -0.3, -0.01])

    asyncio.run(run_case())


def test_call_sglang_no_continue_without_open_tail(monkeypatch):
    async def run_case():
        turns = [[(-0.1, 101), (-0.2, 102)]]
        async with FakeSGLangServer(turns) as sglang:
            adapter = SimpleNamespace(
                logger=logging.getLogger("test_offload_cd"),
                log_prefix="test",
                sglang_url=sglang.url,
                max_token_keys=("max_tokens",),
                stop_keys=("stop",),
            )
            session = common_mod.Session(sampling_defaults={"max_new_tokens": 8}, max_context_tokens=0)
            turn = await common_mod.call_sglang_generate(
                [9], session, {}, adapter=adapter, session_id="sid-x"
            )
        assert len(sglang.requests) == 1
        assert turn.output_ids == [101, 102]

    asyncio.run(run_case())


def test_call_sglang_single_step_when_constrained_off(monkeypatch):
    monkeypatch.setenv("OFFLOAD_CONSTRAINED_DECODE", "0")

    async def run_case():
        # Would be two-step if constrained were on; with flag off only one call.
        turns = [[(-0.1, OPEN_ID)]]
        async with FakeSGLangServer(turns) as sglang:
            adapter = SimpleNamespace(
                logger=logging.getLogger("test_offload_cd"),
                log_prefix="test",
                sglang_url=sglang.url,
                max_token_keys=("max_tokens",),
                stop_keys=("stop",),
            )
            session = common_mod.Session(sampling_defaults={"max_new_tokens": 8}, max_context_tokens=0)
            turn = await common_mod.call_sglang_generate(
                [1], session, {}, adapter=adapter, session_id="sid-off"
            )
        assert len(sglang.requests) == 1
        assert "ebnf" not in sglang.requests[0]["sampling_params"]
        assert "logit_bias" not in sglang.requests[0]["sampling_params"]
        assert turn.output_ids == [OPEN_ID]

    asyncio.run(run_case())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
