"""Online SFT samples from successful DeepSeek/GLM offload turns.

Target pair (per PASS offload turn)::

    x = Qwen think with ``<|llm_offload|>N<|/llm_offload|>`` stripped
    y = x + GLM think + content + tool_calls   (Qwen chat/tool template)

``loss_mask`` is 0 on ``x`` and 1 on the GLM continuation. Offload tags never
enter the SFT sequence. These samples are emitted alongside the GRPO trajectory
from the same ``generate()`` call but use a distinct ``rollout_id`` and
``train_metadata.objective == "sft"`` so they never enter GRPO grouping.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def sft_lambda() -> float:
    """Weight λ for L_SFT. Default 0 → Exp 1 (GRPO only, no SFT rows)."""
    raw = (os.environ.get("OFFLOAD_SFT_LAMBDA") or "0").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def is_sft_sample(sample: Any) -> bool:
    tmd = getattr(sample, "train_metadata", None) or {}
    return isinstance(tmd, dict) and tmd.get("objective") == "sft"


def _normalize_think_text(text: str) -> str:
    """Strip optional ``<think>`` wrappers; chat templates re-add them."""
    t = (text or "").strip()
    if t.startswith("<think>"):
        t = t[len("<think>") :].lstrip("\n")
    if t.endswith("</think>"):
        t = t[: -len("</think>")].rstrip("\n")
    return t


def _assistant_content_with_think(*, reasoning: str, content: str) -> str:
    """Qwen-style assistant ``content`` embedding think + visible text."""
    reason = _normalize_think_text(reasoning)
    answer = (content or "").strip()
    if reason and answer:
        return f"<think>\n{reason}\n</think>\n\n{answer}"
    if reason:
        return f"<think>\n{reason}\n</think>\n\n"
    return answer


def _normalize_tool_calls(tool_calls: list[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, call in enumerate(tool_calls or []):
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") or call.get("name") or "tool"
        arguments = function.get("arguments")
        if arguments is None:
            arguments = call.get("arguments", {})
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
            except TypeError:
                arguments = json.dumps({"_raw": str(arguments)}, ensure_ascii=False)
        call_id = call.get("id") or f"sft-call-{i}"
        out.append(
            {
                "id": str(call_id),
                "type": "function",
                "function": {"name": str(name), "arguments": arguments},
            }
        )
    return out


def _render_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None,
    add_generation_prompt: bool,
) -> list[int]:
    enc = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(ids)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_sft_token_sequence(
    tokenizer: Any,
    *,
    history: list[dict[str, Any]],
    tools: list[dict] | None,
    qwen_think: str,
    glm_think: str,
    glm_content: str,
    glm_tool_calls: list[Any] | None,
) -> tuple[list[int], list[int], int] | None:
    """Return ``(tokens, loss_mask, response_length)`` or None if unusable.

    ``loss_mask`` covers only the response region (length == response_length):
    0 on Qwen ``x``, 1 on GLM think/content/tool continuation.
    """
    x = _normalize_think_text(qwen_think)
    g_think = _normalize_think_text(glm_think)
    g_content = (glm_content or "").strip()
    g_tools = _normalize_tool_calls(glm_tool_calls)
    if not x and not g_think and not g_content and not g_tools:
        return None
    if not g_think and not g_content and not g_tools:
        return None

    y_reasoning = f"{x}{g_think}" if (x or g_think) else ""
    assistant_y: dict[str, Any] = {
        "role": "assistant",
        "content": _assistant_content_with_think(reasoning=y_reasoning, content=g_content),
    }
    if g_tools:
        assistant_y["tool_calls"] = g_tools

    assistant_x: dict[str, Any] = {
        "role": "assistant",
        "content": _assistant_content_with_think(reasoning=x, content=""),
    }

    try:
        prompt_ids = _render_ids(tokenizer, history, tools=tools, add_generation_prompt=True)
        ids_y = _render_ids(tokenizer, history + [assistant_y], tools=tools, add_generation_prompt=False)
        ids_x = _render_ids(tokenizer, history + [assistant_x], tools=tools, add_generation_prompt=False)
    except Exception:
        logger.exception("[offload_sft] apply_chat_template failed; skip SFT sample")
        return None

    prefix = _common_prefix_len(prompt_ids, ids_y)
    if prefix < len(prompt_ids) // 2:
        # Generation-prompt form may not be an exact prefix of the full assistant
        # render on some templates; fall back to shared prefix only.
        logger.warning(
            "[offload_sft] prompt not a strong prefix of y (prefix=%d prompt=%d y=%d); using shared prefix",
            prefix,
            len(prompt_ids),
            len(ids_y),
        )
    tokens = list(ids_y)
    response = tokens[prefix:]
    if not response:
        return None

    # x-boundary: how much of the response matches the x-only assistant render.
    x_resp = ids_x[prefix:] if prefix <= len(ids_x) else []
    x_overlap = _common_prefix_len(response, x_resp)
    # Prefer encoding plain ``x`` as the supervised-free prefix when it aligns.
    if x:
        try:
            x_plain = list(tokenizer.encode(x, add_special_tokens=False))
        except Exception:
            x_plain = []
        if x_plain and response[: len(x_plain)] == x_plain and len(x_plain) > x_overlap:
            x_overlap = len(x_plain)

    if x_overlap >= len(response):
        # Degenerate: nothing left for GLM CE.
        return None

    loss_mask = [0] * x_overlap + [1] * (len(response) - x_overlap)
    response_length = len(response)
    assert len(loss_mask) == response_length
    return tokens, loss_mask, response_length


def turn_eligible_for_sft(tc: dict[str, Any]) -> bool:
    if not tc.get("valid_offload"):
        return False
    if tc.get("repaired") or tc.get("outside_think"):
        return False
    think = str(tc.get("teacher_think") or "")
    content = str(tc.get("teacher_content") or tc.get("teacher_response") or "")
    tools = tc.get("teacher_tool_calls") or []
    return bool(think.strip() or content.strip() or tools)


def build_sft_samples(
    *,
    grpo_samples: list[Sample],
    tokenizer: Any,
    grading_solved: bool,
) -> list[Sample]:
    """Emit 0+ SFT ``Sample``s from PASS + valid offload turns on this episode."""
    if sft_lambda() <= 0.0:
        return []
    if not grading_solved or not grpo_samples or tokenizer is None:
        return []

    base = grpo_samples[0]
    md = dict(getattr(base, "metadata", None) or {})
    turn_costs = list(md.get("turn_costs") or (md.get("offload_stats") or {}).get("turn_costs") or [])
    if not turn_costs:
        return []

    # Ensure train_metadata is present so slime packs metadata for the whole batch.
    for s in grpo_samples:
        tmd = dict(getattr(s, "train_metadata", None) or {})
        tmd.setdefault("objective", "grpo")
        s.train_metadata = tmd

    out: list[Sample] = []
    base_index = int(getattr(base, "index", 0) or 0)
    group_index = getattr(base, "group_index", None)

    for turn_idx, tc in enumerate(turn_costs):
        if not isinstance(tc, dict) or not turn_eligible_for_sft(tc):
            continue
        history = tc.get("sft_history_messages")
        if not isinstance(history, list) or not history:
            logger.debug("[offload_sft] turn %d missing sft_history_messages; skip", turn_idx)
            continue
        tools = tc.get("sft_tools_schema")
        if tools is not None and not isinstance(tools, list):
            tools = None

        built = build_sft_token_sequence(
            tokenizer,
            history=history,
            tools=tools,
            qwen_think=str(tc.get("qwen_think_prefix") or ""),
            glm_think=str(tc.get("teacher_think") or ""),
            glm_content=str(tc.get("teacher_content") or tc.get("teacher_response") or ""),
            glm_tool_calls=tc.get("teacher_tool_calls"),
        )
        if built is None:
            continue
        tokens, loss_mask, response_length = built
        prompt_len = len(tokens) - response_length
        if prompt_len < 0:
            continue

        sft = Sample(
            index=base_index,
            group_index=group_index,
            # Distinct rollout_id so SFT tokens do not inflate GRPO rollout_mask_sums.
            rollout_id=-((base_index + 1) * 10_000 + turn_idx + 1),
            prompt=copy.deepcopy(getattr(base, "prompt", "")),
            label=getattr(base, "label", None),
            tokens=tokens,
            response_length=response_length,
            loss_mask=loss_mask,
            rollout_log_probs=[0.0] * response_length,
            reward=0.0,
            status=Sample.Status.COMPLETED,
            metadata={
                "objective": "sft",
                "sft_turn_index": turn_idx,
                "instance_id": md.get("instance_id"),
                "grading_solved": True,
            },
            train_metadata={
                "objective": "sft",
                "sft_turn_index": turn_idx,
            },
        )
        out.append(sft)

    if out:
        logger.info(
            "[offload_sft] emitted %d SFT sample(s) from index=%s (λ=%.4f)",
            len(out),
            base_index,
            sft_lambda(),
        )
    return out


def post_process_rewards_grpo_only(args: Any, samples: list[Sample]):
    """GRPO mean/std by ``group_index``; SFT rows stay at reward 0 and are excluded."""
    import torch
    from collections import defaultdict

    raw_rewards = [s.get_reward_value(args) for s in samples]
    groups: dict[Any, list[tuple[int, float]]] = defaultdict(list)
    out = [0.0] * len(samples)
    for i, s in enumerate(samples):
        if is_sft_sample(s):
            continue
        groups[getattr(s, "group_index", i)].append((i, float(raw_rewards[i])))

    use_std = getattr(args, "grpo_std_normalization", True)
    for indexed in groups.values():
        positions = [p for p, _ in indexed]
        rewards = torch.tensor([r for _, r in indexed], dtype=torch.float)
        rewards = rewards - rewards.mean()
        if use_std and len(indexed) > 1:
            rewards = rewards / (rewards.std() + 1e-6)
        for pos, r in zip(positions, rewards.tolist(), strict=True):
            out[pos] = r
    return raw_rewards, out
