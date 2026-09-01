#!/usr/bin/env python3
"""Unit tests for multiturn SFT construction from offload turn fields."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest  # noqa: E402

from examples.coding_agent_rl import offload  # noqa: E402
from examples.coding_agent_rl.offload_sft import (  # noqa: E402
    build_assistant_target,
    build_multiturn_sft_messages,
    build_multiturn_sft_token_sequence,
    build_sft_samples,
    build_sft_token_sequence,
    episode_has_valid_offload,
    post_process_rewards_grpo_only,
    turn_eligible_for_sft,
)
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0


class _FakeTok:
    """Minimal chat-template tokenizer: char-level ids, Qwen-like think format."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, token_ids):
        return "".join(chr(t) for t in token_ids)

    def apply_chat_template(self, messages, tokenize=True, tools=None, add_generation_prompt=False, **kwargs):
        # Match Qwen/PyroDash: reject system-only prefixes.
        if not any(m.get("role") == "user" for m in messages):
            raise ValueError("No user query found in messages.")
        parts = []
        for m in messages:
            role = m["role"]
            reasoning = m.get("reasoning_content") or ""
            content = m.get("content") or ""
            if reasoning:
                parts.append(
                    f"<|im_start|>{role}\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>\n"
                )
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if args is not None and not isinstance(args, dict):
                    raise TypeError("Can only get item pairs from a mapping.")
                parts.append(f"<tool_call>{fn.get('name')}:{args}</tool_call>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        text = "".join(parts)
        if tokenize:
            return self.encode(text)
        return text


def test_strip_offload_not_in_x() -> None:
    raw = f"<think>\nneed help {offload.OFFLOAD_OPEN}3{offload.OFFLOAD_CLOSE}"
    prefix = offload._offload_prefix(raw)
    assert offload.OFFLOAD_OPEN not in prefix
    assert offload.OFFLOAD_CLOSE not in prefix
    assert "need help" in prefix


def test_turn_eligible_requires_history_and_payload() -> None:
    assert not turn_eligible_for_sft({"valid_offload": True})
    assert turn_eligible_for_sft(
        {
            "valid_offload": True,
            "teacher_think": "why",
            "qwen_think_prefix": "local",
            "sft_history_messages": [{"role": "user", "content": "q"}],
        }
    )
    assert turn_eligible_for_sft(
        {
            "slm_raw_output": "solo think",
            "sft_history_messages": [{"role": "user", "content": "q"}],
        }
    )


class _RewriteTok(_FakeTok):
    """Prefix-unstable template: later user turns rewrite earlier user encoding."""

    def apply_chat_template(self, messages, tokenize=True, tools=None, add_generation_prompt=False, **kwargs):
        if not any(m.get("role") == "user" for m in messages):
            raise ValueError("No user query found in messages.")
        users = [m for m in messages if m.get("role") == "user"]
        # Merging consecutive users changes earlier ids once a second user appears.
        merged_users = [{"role": "user", "content": "|".join(str(u.get("content") or "") for u in users)}]
        rest = [m for m in messages if m.get("role") != "user"]
        # Keep system first, then one merged user, then non-user turns in order.
        systems = [m for m in rest if m.get("role") == "system"]
        others = [m for m in rest if m.get("role") != "system"]
        return super().apply_chat_template(
            systems + merged_users + others,
            tokenize=tokenize,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )


def test_tokenize_skips_system_only_prefix() -> None:
    """Regression: must not call chat template on [system] alone."""
    tok = _FakeTok()
    turn_costs = [
        {
            "slm_raw_output": "think",
            "sft_assistant_message": {"role": "assistant", "reasoning_content": "think", "content": "ans"},
            "sft_history_messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ],
        }
    ]
    built = build_multiturn_sft_token_sequence(tok, turn_costs)
    assert built is not None
    tokens, loss_mask, response_length = built
    assert response_length > 0 and 1 in loss_mask
    assert "think" in tok.decode(tokens[-response_length:])


def test_tokenize_realigns_when_template_rewrites_prefix() -> None:
    """Consecutive users that rewrite earlier ids must not abort the whole sample."""
    tok = _RewriteTok()
    turn_costs = [
        {
            "slm_raw_output": "solo",
            "sft_assistant_message": {"role": "assistant", "reasoning_content": "solo", "content": "ok"},
            "sft_history_messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q1"},
                {"role": "user", "content": "q2"},
            ],
        }
    ]
    built = build_multiturn_sft_token_sequence(tok, turn_costs)
    assert built is not None
    tokens, loss_mask, response_length = built
    assert 1 in loss_mask
    supervised = tok.decode([t for t, m in zip(tokens[-response_length:], loss_mask) if m])
    assert "solo" in supervised
    # Rewritten user prefix is context, not a wiped assistant span.
    assert loss_mask.count(1) >= len("solo")


def test_sft_mask_supervises_all_assistant_spans() -> None:
    tok = _FakeTok()
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix bug"},
    ]
    built = build_sft_token_sequence(
        tok,
        history=history,
        tools=None,
        qwen_think="local plan",
        glm_think="remote why",
        glm_content="edit foo.py",
        glm_tool_calls=None,
    )
    assert built is not None
    tokens, loss_mask, response_length = built
    assert len(loss_mask) == response_length
    assert all(m == 1 for m in loss_mask)
    decoded = tok.decode(tokens[-response_length:])
    assert "local plan" in decoded
    assert "remote why" in decoded
    assert "edit foo.py" in decoded


def test_sft_random_offload_tag() -> None:
    tok = _FakeTok()
    history = [{"role": "user", "content": "fix bug"}]
    with_tag = build_sft_token_sequence(
        tok,
        history=history,
        tools=None,
        qwen_think="local plan",
        glm_think="remote why",
        glm_content="edit foo.py",
        glm_tool_calls=None,
        include_offload_tag=True,
        offload_n=3,
    )
    without_tag = build_sft_token_sequence(
        tok,
        history=history,
        tools=None,
        qwen_think="local plan",
        glm_think="remote why",
        glm_content="edit foo.py",
        glm_tool_calls=None,
        include_offload_tag=False,
    )
    assert with_tag is not None and without_tag is not None
    _, _, resp_w = with_tag
    _, _, resp_wo = without_tag
    dec_w = tok.decode(with_tag[0][-resp_w:])
    dec_wo = tok.decode(without_tag[0][-resp_wo:])
    assert offload.OFFLOAD_OPEN in dec_w and offload.OFFLOAD_CLOSE in dec_w
    assert offload.OFFLOAD_OPEN not in dec_wo


def test_tokenize_skips_system_only_prefix() -> None:
    """Qwen templates reject system-only prefixes; multiturn SFT must still emit."""
    tok = _FakeTok()
    turn_costs = [
        {
            "valid_offload": True,
            "n": 2,
            "slm_raw_output": f"{offload.OFFLOAD_OPEN}2{offload.OFFLOAD_CLOSE}",
            "teacher_think": "remote",
            "teacher_content": "do it",
            "teacher_tool_calls": [],
            "sft_history_messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ],
        }
    ]
    built = build_multiturn_sft_token_sequence(tok, turn_costs, rng=__import__("random").Random(0))
    assert built is not None
    tokens, loss_mask, response_length = built
    assert len(loss_mask) == response_length
    assert 1 in loss_mask
    decoded = tok.decode(tokens)
    assert "sys" in decoded and "remote" in decoded and "do it" in decoded


def test_multiturn_two_assistants_both_supervised() -> None:
    tok = _FakeTok()
    turn_costs = [
        {
            "valid_offload": True,
            "n": 4,
            "qwen_think_prefix": "local1",
            "teacher_think": "glm1",
            "teacher_content": "step1",
            "teacher_tool_calls": [],
            "sft_history_messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q1"},
            ],
        },
        {
            "slm_raw_output": "solo2",
            "sft_assistant_message": {
                "role": "assistant",
                "reasoning_content": "solo2",
                "content": "step2",
            },
            "sft_history_messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q1"},
                {
                    "role": "assistant",
                    "reasoning_content": f"{offload.OFFLOAD_OPEN}4{offload.OFFLOAD_CLOSE}",
                    "content": "step1",
                },
                {"role": "user", "content": "q2"},
            ],
        },
    ]
    built = build_multiturn_sft_token_sequence(tok, turn_costs, rng=__import__("random").Random(0))
    assert built is not None
    tokens, loss_mask, response_length = built
    assert len(loss_mask) == response_length
    assert loss_mask.count(1) > 0
    decoded = tok.decode(tokens)
    assert "local1" in decoded and "glm1" in decoded and "solo2" in decoded
    # Two assistant blocks → two supervised regions (mask may include template chars between)
    assert sum(loss_mask) > len("local1glm1step1solo2")


def test_multiturn_history_reset_keeps_longest_segment() -> None:
    tok = _FakeTok()
    turn_costs = [
        {
            "valid_offload": True,
            "n": 1,
            "slm_raw_output": "local0",
            "teacher_think": "glm0",
            "teacher_content": "do0",
            "teacher_tool_calls": [],
            "sft_history_messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q0"},
                {"role": "user", "content": "q0b"},
            ],
        },
        {
            "slm_raw_output": "solo-reset",
            "sft_assistant_message": {
                "role": "assistant",
                "reasoning_content": "solo-reset",
                "content": "after-reset",
            },
            "sft_history_messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q1"},
            ],
        },
        {
            "slm_raw_output": "solo-tail",
            "sft_assistant_message": {
                "role": "assistant",
                "reasoning_content": "solo-tail",
                "content": "tail",
            },
            "sft_history_messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "reasoning_content": "solo-reset", "content": "after-reset"},
                {"role": "user", "content": "q2"},
            ],
        },
    ]
    built = build_multiturn_sft_token_sequence(tok, turn_costs)
    assert built is not None
    decoded = tok.decode(built[0])
    assert "solo-reset" in decoded and "solo-tail" in decoded
    assert "local0" not in decoded


def test_build_assistant_target_merges_slm_and_glm_think() -> None:
    tc = {
        "valid_offload": True,
        "n": 1,
        "slm_raw_output": f"{offload.OFFLOAD_OPEN}1{offload.OFFLOAD_CLOSE}",
        "teacher_think": "remote reasoning",
        "teacher_content": "visible answer",
        "teacher_tool_calls": [],
        "sft_assistant_message": {
            "role": "assistant",
            "reasoning_content": f"{offload.OFFLOAD_OPEN}1{offload.OFFLOAD_CLOSE}remote reasoning",
            "content": "visible answer",
        },
    }
    built = build_assistant_target(tc, include_offload_tag=True)
    assert built is not None
    reasoning = built.get("reasoning_content") or ""
    assert reasoning == f"{offload.OFFLOAD_OPEN}1{offload.OFFLOAD_CLOSE}remote reasoning"
    assert reasoning.count("remote reasoning") == 1


def test_build_assistant_target_offload_tag_toggle() -> None:
    tc = {
        "valid_offload": True,
        "n": 5,
        "slm_raw_output": "local plan",
        "qwen_think_prefix": "local plan",
        "teacher_think": "why",
        "teacher_content": "do",
        "teacher_tool_calls": [],
    }
    with_tag = build_assistant_target(tc, include_offload_tag=True)
    without_tag = build_assistant_target(tc, include_offload_tag=False)
    assert f"{offload.OFFLOAD_OPEN}5{offload.OFFLOAD_CLOSE}" in (with_tag.get("reasoning_content") or "")
    assert "local plan" in (with_tag.get("reasoning_content") or "")
    assert "why" in (with_tag.get("reasoning_content") or "")
    assert offload.OFFLOAD_OPEN not in (without_tag.get("reasoning_content") or "")
    assert "local planwhy" in (without_tag.get("reasoning_content") or "")


def test_forced_offload_sft_keeps_tag_even_if_tag_prob_zero() -> None:
    prev = os.environ.get("OFFLOAD_SFT_TAG_PROB")
    os.environ["OFFLOAD_SFT_TAG_PROB"] = "0"
    try:
        tc = {
            "valid_offload": True,
            "forced_offload": True,
            "n": 5,
            "slm_raw_output": "local plan",
            "qwen_think_prefix": "local plan",
            "teacher_think": "why",
            "teacher_content": "do",
            "teacher_tool_calls": [],
            "sft_history_messages": [{"role": "user", "content": "fix"}],
        }
        out = build_multiturn_sft_messages([tc], rng=random.Random(0))
        assert out is not None
        messages, idxs = out
        asst = messages[max(idxs)]
        assert offload.OFFLOAD_OPEN in (asst.get("reasoning_content") or "")
    finally:
        if prev is None:
            os.environ.pop("OFFLOAD_SFT_TAG_PROB", None)
        else:
            os.environ["OFFLOAD_SFT_TAG_PROB"] = prev


def test_sft_tool_calls_accept_openai_json_arguments() -> None:
    tok = _FakeTok()
    history = [{"role": "user", "content": "fix B.py"}]
    built = build_sft_token_sequence(
        tok,
        history=history,
        tools=None,
        qwen_think="need to read B.py",
        glm_think="stack points at B.py:80",
        glm_content="",
        glm_tool_calls=[
            {
                "id": "chatcmpl-tool-glm-0",
                "type": "function",
                "function": {"name": "Read", "arguments": '{"path": "B.py"}'},
            }
        ],
    )
    assert built is not None
    tokens, loss_mask, response_length = built
    decoded = tok.decode(tokens[-response_length:])
    assert "Read" in decoded
    assert "B.py" in decoded
    assert 1 in loss_mask


def test_build_sft_samples_emits_one_multiturn_row() -> None:
    tok = _FakeTok()
    os.environ["OFFLOAD_SFT_LAMBDA"] = "0"
    os.environ["OFFLOAD_SFT_TAG_PROB"] = "0"
    messages: list[dict[str, str]] = [{"role": "user", "content": "q"}]
    turn0 = {
        "valid_offload": True,
        "n": 2,
        "qwen_think_prefix": "local",
        "teacher_think": "why",
        "teacher_content": "do it",
        "teacher_tool_calls": [],
        "sft_history_messages": list(messages),
    }
    messages = list(messages) + [
        {
            "role": "assistant",
            "reasoning_content": "localwhy",
            "content": "do it",
        },
        {"role": "user", "content": "next"},
    ]
    turn1 = {
        "slm_raw_output": "follow up",
        "sft_history_messages": list(messages),
    }
    grpo = Sample(
        index=7,
        group_index=1,
        tokens=[1, 2, 3],
        response_length=2,
        loss_mask=[1, 1],
        reward=1.0,
        metadata={"grading_solved": True, "turn_costs": [turn0, turn1]},
    )
    assert build_sft_samples(grpo_samples=[grpo], tokenizer=tok, grading_solved=True) == []

    os.environ["OFFLOAD_SFT_LAMBDA"] = "0.1"
    out = build_sft_samples(grpo_samples=[grpo], tokenizer=tok, grading_solved=True)
    assert len(out) == 1
    assert out[0].train_metadata["objective"] == "sft"
    assert out[0].train_metadata["sft_multiturn"] is True
    assert out[0].rollout_id == 7
    assert sum(out[0].loss_mask) > 0

    fail_out = build_sft_samples(grpo_samples=[grpo], tokenizer=tok, grading_solved=False)
    assert fail_out == []
    os.environ.pop("OFFLOAD_SFT_TAG_PROB", None)
    os.environ.pop("OFFLOAD_SFT_LAMBDA", None)


def test_build_sft_samples_includes_solo_pass_skips_fail() -> None:
    tok = _FakeTok()
    os.environ["OFFLOAD_SFT_LAMBDA"] = "0.1"
    solo = {
        "slm_raw_output": "I can do this",
        "sft_history_messages": [{"role": "user", "content": "q"}],
    }
    grpo = Sample(
        index=3,
        tokens=[1, 2, 3],
        response_length=2,
        loss_mask=[1, 1],
        reward=1.0,
        metadata={"grading_solved": True, "turn_costs": [solo]},
    )
    assert not episode_has_valid_offload([solo])
    out = build_sft_samples(grpo_samples=[grpo], tokenizer=tok, grading_solved=True)
    assert len(out) == 1
    unsolved = Sample(
        index=4,
        tokens=[1, 2, 3],
        response_length=2,
        loss_mask=[1, 1],
        reward=0.0,
        metadata={"grading_solved": False, "turn_costs": [solo]},
    )
    assert build_sft_samples(grpo_samples=[unsolved], tokenizer=tok, grading_solved=False) == []
    os.environ.pop("OFFLOAD_SFT_LAMBDA", None)


def test_sft_max_samples_keeps_last_turns() -> None:
    tok = _FakeTok()
    os.environ["OFFLOAD_SFT_LAMBDA"] = "0.1"
    os.environ["OFFLOAD_SFT_MAX_SAMPLES"] = "1"
    os.environ["OFFLOAD_SFT_MAX_SEQ_LEN"] = "0"
    messages: list[dict[str, str]] = [{"role": "user", "content": "q0"}]
    turns = []
    for i in range(2):
        turns.append(
            {
                "valid_offload": True,
                "n": 1,
                "qwen_think_prefix": f"local{i}",
                "teacher_think": f"why{i}",
                "teacher_content": f"do{i}",
                "teacher_tool_calls": [],
                "sft_history_messages": list(messages),
            }
        )
        messages = list(messages) + [
            {
                "role": "assistant",
                "reasoning_content": f"local{i}why{i}",
                "content": f"do{i}",
            },
            {"role": "user", "content": f"q{i + 1}"},
        ]
    grpo = Sample(
        index=3,
        group_index=1,
        tokens=[1, 2, 3],
        response_length=2,
        loss_mask=[1, 1],
        reward=1.0,
        metadata={"grading_solved": True, "turn_costs": turns},
    )
    out = build_sft_samples(grpo_samples=[grpo], tokenizer=tok, grading_solved=True)
    assert len(out) == 1
    decoded = tok.decode(out[0].tokens)
    assert "local1" in decoded
    assert out[0].loss_mask.count(1) > 0
    # Prior assistant stays in context but is not supervised when max_samples=1.
    assert "local0why0" not in tok.decode(
        [t for t, m in zip(out[0].tokens[-out[0].response_length :], out[0].loss_mask) if m == 1]
    )
    os.environ.pop("OFFLOAD_SFT_MAX_SAMPLES", None)
    os.environ.pop("OFFLOAD_SFT_MAX_SEQ_LEN", None)


def test_sft_left_trims_old_history_keeps_tail() -> None:
    tok = _FakeTok()
    os.environ["OFFLOAD_SFT_TAG_PROB"] = "0"
    old = "OLDCTX" * 40
    turn_costs = [
        {
            "valid_offload": True,
            "n": 1,
            "qwen_think_prefix": "local plan",
            "teacher_think": "remote why",
            "teacher_content": "edit foo.py",
            "teacher_tool_calls": [],
            "sft_history_messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": old},
                {"role": "assistant", "content": "old-reply"},
                {"role": "user", "content": "latest-q"},
            ],
        }
    ]
    full = build_multiturn_sft_token_sequence(tok, turn_costs, max_seq=0)
    assert full is not None
    last_only = build_multiturn_sft_token_sequence(
        tok,
        [
            {
                "valid_offload": True,
                "n": 1,
                "qwen_think_prefix": "local plan",
                "teacher_think": "remote why",
                "teacher_content": "edit foo.py",
                "teacher_tool_calls": [],
                "sft_history_messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "latest-q"},
                ],
            }
        ],
        max_seq=0,
    )
    assert last_only is not None
    cap = len(last_only[0]) + 80
    assert len(full[0]) > cap > len(last_only[0])
    trimmed = build_multiturn_sft_token_sequence(tok, turn_costs, max_seq=cap)
    assert trimmed is not None
    tokens, loss_mask, response_length = trimmed
    assert len(tokens) <= cap
    decoded = tok.decode(tokens)
    assert "latest-q" in decoded
    assert "remote why" in decoded
    assert old not in decoded
    assert 1 in loss_mask
    assert build_multiturn_sft_token_sequence(tok, turn_costs, max_seq=8) is None
    os.environ.pop("OFFLOAD_SFT_TAG_PROB", None)


def test_post_process_skips_sft_in_group() -> None:
    class _Args:
        grpo_std_normalization = True
        reward_key = None

    grpo_a = Sample(index=0, group_index=9, reward=1.0, train_metadata={"objective": "grpo"})
    grpo_b = Sample(index=1, group_index=9, reward=0.0, train_metadata={"objective": "grpo"})
    sft = Sample(index=0, group_index=9, reward=0.0, train_metadata={"objective": "sft"})
    raw, norm = post_process_rewards_grpo_only(_Args(), [grpo_a, grpo_b, sft])
    assert raw == [1.0, 0.0, 0.0]
    assert norm[2] == 0.0
    assert abs(norm[0] + norm[1]) < 1e-5
    assert norm[0] > 0 > norm[1]


def test_post_process_excludes_remove_sample_from_baseline() -> None:
    class _Args:
        grpo_std_normalization = False
        reward_key = None

    kept_hi = Sample(index=0, group_index=1, reward=1.0, train_metadata={"objective": "grpo"})
    removed = Sample(
        index=1,
        group_index=1,
        reward=1.0,
        remove_sample=True,
        train_metadata={"objective": "grpo"},
    )
    kept_lo = Sample(index=2, group_index=1, reward=0.0, train_metadata={"objective": "grpo"})
    raw, norm = post_process_rewards_grpo_only(_Args(), [kept_hi, removed, kept_lo])
    assert raw == [1.0, 1.0, 0.0]
    # Baseline uses only kept rows (mean 0.5), not the compact-removed solved row.
    assert norm[0] == pytest.approx(0.5)
    assert norm[1] == 0.0
    assert norm[2] == pytest.approx(-0.5)


if __name__ == "__main__":
    test_strip_offload_not_in_x()
    test_turn_eligible_requires_history_and_payload()
    test_sft_mask_supervises_all_assistant_spans()
    test_sft_random_offload_tag()
    test_tokenize_skips_system_only_prefix()
    test_tokenize_realigns_when_template_rewrites_prefix()
    test_multiturn_two_assistants_both_supervised()
    test_multiturn_history_reset_keeps_longest_segment()
    test_build_assistant_target_merges_slm_and_glm_think()
    test_build_assistant_target_offload_tag_toggle()
    test_forced_offload_sft_keeps_tag_even_if_tag_prob_zero()
    test_sft_tool_calls_accept_openai_json_arguments()
    test_build_sft_samples_emits_one_multiturn_row()
    test_build_sft_samples_includes_solo_pass_skips_fail()
    test_sft_max_samples_keeps_last_turns()
    test_sft_left_trims_old_history_keeps_tail()
    test_post_process_skips_sft_in_group()
    test_post_process_excludes_remove_sample_from_baseline()
    print("ok")
