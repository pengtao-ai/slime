#!/usr/bin/env python3
"""Unit tests for offline SFT (x, y) construction from offload turn fields."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from examples.coding_agent_rl import offload  # noqa: E402
from examples.coding_agent_rl.offload_sft import (  # noqa: E402
    build_sft_samples,
    build_sft_token_sequence,
    post_process_rewards_grpo_only,
    turn_eligible_for_sft,
)
from slime.utils.types import Sample  # noqa: E402


class _FakeTok:
    """Minimal chat-template tokenizer: char-level ids, Qwen-like think format."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, token_ids):
        return "".join(chr(t) for t in token_ids)

    def apply_chat_template(self, messages, tokenize=True, tools=None, add_generation_prompt=False, **kwargs):
        parts = []
        for m in messages:
            role = m["role"]
            content = m.get("content") or ""
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                parts.append(f"<tool_call>{fn.get('name')}:{fn.get('arguments')}</tool_call>")
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


def test_turn_eligible_requires_glm_payload() -> None:
    assert not turn_eligible_for_sft({"valid_offload": True})
    assert turn_eligible_for_sft(
        {"valid_offload": True, "teacher_think": "why", "teacher_content": "", "teacher_tool_calls": []}
    )
    assert not turn_eligible_for_sft({"valid_offload": True, "repaired": True, "teacher_think": "why"})


def test_sft_mask_zeros_x_ones_glm() -> None:
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
    assert 0 in loss_mask and 1 in loss_mask
    first_one = loss_mask.index(1)
    assert all(m == 0 for m in loss_mask[:first_one])
    assert all(m == 1 for m in loss_mask[first_one:])
    decoded = tok.decode(tokens[-response_length:])
    assert "local plan" in decoded
    assert "remote why" in decoded
    assert "edit foo.py" in decoded
    assert offload.OFFLOAD_OPEN not in decoded


def test_build_sft_samples_only_when_lambda_and_pass() -> None:
    tok = _FakeTok()
    os.environ["OFFLOAD_SFT_LAMBDA"] = "0"
    grpo = Sample(
        index=7,
        group_index=1,
        tokens=[1, 2, 3],
        response_length=2,
        loss_mask=[1, 1],
        reward=1.0,
        metadata={
            "grading_solved": True,
            "turn_costs": [
                {
                    "valid_offload": True,
                    "qwen_think_prefix": "local",
                    "teacher_think": "why",
                    "teacher_content": "do it",
                    "teacher_tool_calls": [],
                    "sft_history_messages": [
                        {"role": "user", "content": "q"},
                    ],
                }
            ],
        },
    )
    assert build_sft_samples(grpo_samples=[grpo], tokenizer=tok, grading_solved=True) == []

    os.environ["OFFLOAD_SFT_LAMBDA"] = "0.1"
    out = build_sft_samples(grpo_samples=[grpo], tokenizer=tok, grading_solved=True)
    assert len(out) == 1
    assert out[0].train_metadata["objective"] == "sft"
    assert out[0].rollout_id is not None and out[0].rollout_id < 0
    assert sum(out[0].loss_mask) > 0
    assert out[0].loss_mask.count(0) > 0

    assert build_sft_samples(grpo_samples=[grpo], tokenizer=tok, grading_solved=False) == []


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


if __name__ == "__main__":
    test_strip_offload_not_in_x()
    test_turn_eligible_requires_glm_payload()
    test_sft_mask_zeros_x_ones_glm()
    test_build_sft_samples_only_when_lambda_and_pass()
    test_post_process_skips_sft_in_group()
    print("ok")
