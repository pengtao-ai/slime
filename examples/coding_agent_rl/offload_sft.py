"""Online multiturn SFT from solved coding-agent episodes.

One long chat-template sequence per PASS episode: every assistant turn is
supervised (``loss_mask=1`` on assistant spans; user/tool spans in the
response tail stay 0). Offload turns use the GLM teacher continuation; other
turns use the SLM rollout. On offload turns, ``reasoning_content`` is
``slm_raw_output + teacher_think`` (same as ``compose_complete_assistant``);
``<|llm_offload|>N<|/llm_offload|>`` is included with probability
``OFFLOAD_SFT_TAG_PROB`` (default 0.3); otherwise the tag is stripped (GLM
tail still supervised).

Assistant targets use separate ``reasoning_content`` + ``content`` fields (same
as rollout history); do not embed ``<think>`` in ``content`` — the
tokenizer chat template adds think formatting.

Samples share the parent trajectory ``rollout_id`` and
``train_metadata.objective == "sft"`` so they skip GRPO grouping.
"""

from __future__ import annotations

import copy
import logging
import os
import random
from typing import Any

from slime.agent.adapters.openai import _arguments_as_dict
from slime.utils.types import Sample

from examples.coding_agent_rl.offload import (
    OFFLOAD_CLOSE,
    OFFLOAD_OPEN,
    _strip_offload_tag_from_text,
    compose_complete_assistant,
)

logger = logging.getLogger(__name__)


def sft_lambda() -> float:
    """Weight λ for L_SFT. Default 0 → Exp 1 (GRPO only, no SFT rows)."""
    raw = (os.environ.get("OFFLOAD_SFT_LAMBDA") or "0").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def sft_max_samples() -> int:
    """Max assistant turns to include (from the end). Default 0 = all turns."""
    raw = (os.environ.get("OFFLOAD_SFT_MAX_SAMPLES") or "0").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 0
    return max(0, n)


def sft_max_seq_len() -> int:
    """Optional SFT length budget. Default 0 = no cap (same as GRPO trajectories)."""
    raw = (os.environ.get("OFFLOAD_SFT_MAX_SEQ_LEN") or "0").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 0
    return max(0, n)


def sft_tag_prob() -> float:
    """Per offload turn: probability of supervising ``<|llm_offload|>N<|/llm_offload|>``."""
    raw = (os.environ.get("OFFLOAD_SFT_TAG_PROB") or "0.3").strip()
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return 0.3


def is_sft_sample(sample: Any) -> bool:
    tmd = getattr(sample, "train_metadata", None) or {}
    return isinstance(tmd, dict) and tmd.get("objective") == "sft"


def _normalize_think_text(text: str) -> str:
    """Strip optional ``<think>`` wrappers from legacy payloads."""
    t = (text or "").strip()
    if t.startswith("<think>"):
        t = t[len("<think>") :].lstrip("\n")
    if t.endswith("</think>"):
        t = t[: -len("</think>")].rstrip("\n")
    return t


def _make_assistant_message(
    *,
    content: str,
    reasoning: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """OpenAI-shaped assistant message with separate ``reasoning_content``."""
    msg: dict[str, Any] = {"role": "assistant", "content": content if content is not None else ""}
    reason = _normalize_think_text(reasoning)
    if reason:
        msg["reasoning_content"] = reason
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _offload_tag_text(n: Any) -> str:
    if n is None:
        return ""
    try:
        digit = int(n)
    except (TypeError, ValueError):
        return ""
    if not (0 <= digit <= 9):
        return ""
    return f"{OFFLOAD_OPEN}{digit}{OFFLOAD_CLOSE}"


def _reasoning_with_offload_tag(reasoning: str, *, n: Any, include: bool) -> str:
    base = _strip_offload_tag_from_text(_normalize_think_text(reasoning))
    if not include:
        return base
    tag = _offload_tag_text(n)
    if not tag:
        return base
    return f"{base}{tag}" if base else tag


def _slm_reasoning_part(tc: dict[str, Any], *, include_offload_tag: bool) -> str:
    """SLM think prefix before GLM continuation (keeps offload tag in SLM segment)."""
    raw = str(tc.get("slm_raw_output") or "")
    if raw.strip():
        if include_offload_tag:
            base = _strip_offload_tag_from_text(raw)
            tag = _offload_tag_text(tc.get("n"))
            if tag:
                return f"{base}{tag}" if base.strip() else tag
            return raw
        return _strip_offload_tag_from_text(raw)
    prefix = _normalize_think_text(str(tc.get("qwen_think_prefix") or ""))
    return _reasoning_with_offload_tag(prefix, n=tc.get("n"), include=include_offload_tag)


def _compose_offload_reasoning(tc: dict[str, Any], *, include_offload_tag: bool) -> str:
    """Match rollout ``compose_complete_assistant``: ``slm_raw_output + teacher_think``."""
    slm = _slm_reasoning_part(tc, include_offload_tag=include_offload_tag)
    _, think = compose_complete_assistant(
        slm_content=slm,
        glm_content="",
        glm_think=str(tc.get("teacher_think") or ""),
    )
    return think


def _assistant_visible_and_reasoning_from_message(mm: dict[str, Any]) -> tuple[str, str]:
    content = mm.get("content")
    if isinstance(content, list):
        text_parts: list[str] = []
        think_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "thinking":
                think_parts.append(str(block.get("thinking") or ""))
            elif block.get("type") == "text":
                text_parts.append(str(block.get("text") or ""))
        return "\n".join(p for p in text_parts if p), "\n".join(p for p in think_parts if p)
    visible = str(content or "")
    reasoning = str(mm.get("reasoning_content") or "")
    if "<think>" in visible and not reasoning.strip():
        return "", _normalize_think_text(visible)
    return visible, reasoning


def _canonical_chat_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Normalize one chat message for prefix stitching (reasoning stays separate)."""
    role = msg.get("role")
    out: dict[str, Any] = {"role": role}
    if role == "assistant":
        content = msg.get("content")
        reasoning = msg.get("reasoning_content")
        if isinstance(content, list):
            text_parts: list[str] = []
            think_parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "thinking":
                    think_parts.append(str(block.get("thinking") or ""))
                elif block.get("type") == "text":
                    text_parts.append(str(block.get("text") or ""))
            visible = "\n".join(p for p in text_parts if p)
            if not reasoning:
                reasoning = "\n".join(p for p in think_parts if p)
        else:
            visible = str(content or "")
            if isinstance(visible, str) and "<think>" in visible and not reasoning:
                visible = _normalize_think_text(visible)
        out["content"] = visible
        if isinstance(reasoning, str) and reasoning.strip():
            out["reasoning_content"] = _normalize_think_text(reasoning)
        tool_calls = _normalize_tool_calls(msg.get("tool_calls"))
        if tool_calls:
            out["tool_calls"] = tool_calls
        return out

    content = msg.get("content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        out["content"] = "\n".join(p for p in parts if p)
    else:
        out["content"] = str(content or "")
    for key in ("tool_call_id", "name"):
        if msg.get(key):
            out[key] = msg[key]
    return out


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
        entry: dict[str, Any] = {
            "type": "function",
            "function": {"name": str(name), "arguments": _arguments_as_dict(arguments)},
        }
        call_id = call.get("id")
        entry["id"] = str(call_id) if call_id else f"sft-call-{i}"
        out.append(entry)
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


def _split_system_prefix(history: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    i = 0
    while i < len(history) and history[i].get("role") == "system":
        i += 1
    return history[:i], history[i:]


def left_trim_messages_to_fit(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None,
    max_seq: int,
) -> list[dict[str, Any]] | None:
    """Drop oldest non-system messages until the full conversation fits ``max_seq``."""
    if max_seq <= 0:
        return messages
    try:
        if len(_render_ids(tokenizer, messages, tools=tools, add_generation_prompt=False)) <= max_seq:
            return messages
    except Exception:
        return None
    sys_msgs, rest = _split_system_prefix(messages)
    if not rest:
        return sys_msgs if len(_render_ids(tokenizer, sys_msgs, tools=tools, add_generation_prompt=False)) <= max_seq else None
    for start in range(len(rest)):
        trimmed = sys_msgs + rest[start:]
        try:
            if len(_render_ids(tokenizer, trimmed, tools=tools, add_generation_prompt=False)) <= max_seq:
                return trimmed
        except Exception:
            continue
    return None


def _offload_teacher_available(tc: dict[str, Any]) -> bool:
    if not tc.get("valid_offload") or tc.get("repaired") or tc.get("outside_think"):
        return False
    think = str(tc.get("teacher_think") or "")
    content = str(tc.get("teacher_content") or tc.get("teacher_response") or "")
    tools = tc.get("teacher_tool_calls") or []
    return bool(think.strip() or content.strip() or tools)


def build_assistant_target(tc: dict[str, Any], *, include_offload_tag: bool) -> dict[str, Any] | None:
    """Build one OpenAI-shaped assistant message for SFT (``reasoning_content`` separate from ``content``)."""
    if _offload_teacher_available(tc):
        reasoning = _compose_offload_reasoning(tc, include_offload_tag=include_offload_tag)
        content = str(tc.get("teacher_content") or tc.get("teacher_response") or "")
        tools = _normalize_tool_calls(tc.get("teacher_tool_calls"))
        if not reasoning.strip() and not content and not tools:
            return None
        return _make_assistant_message(content=content, reasoning=reasoning, tool_calls=tools or None)

    mm = tc.get("sft_assistant_message")
    if isinstance(mm, dict):
        visible, reasoning = _assistant_visible_and_reasoning_from_message(mm)
        reasoning = _reasoning_with_offload_tag(
            reasoning,
            n=tc.get("n"),
            include=include_offload_tag and bool(tc.get("valid_offload")),
        )
        if not reasoning.strip() and not visible.strip() and not mm.get("tool_calls"):
            return None
        return _make_assistant_message(
            content=visible,
            reasoning=reasoning,
            tool_calls=_normalize_tool_calls(mm.get("tool_calls")),
        )

    raw = str(tc.get("slm_raw_output") or tc.get("qwen_think_prefix") or "").strip()
    if raw:
        reasoning = _reasoning_with_offload_tag(
            raw,
            n=tc.get("n"),
            include=include_offload_tag and bool(tc.get("valid_offload")),
        )
        return _make_assistant_message(content="", reasoning=reasoning)

    return None


def turn_eligible_for_sft(tc: dict[str, Any]) -> bool:
    if not isinstance(tc.get("sft_history_messages"), list) or not tc.get("sft_history_messages"):
        return False
    return build_assistant_target(tc, include_offload_tag=False) is not None


def _tokenize_multiturn(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None,
    supervise_assistant_indices: set[int] | None = None,
) -> tuple[list[int], list[int]] | None:
    """Tokenize a conversation; mask=1 on selected assistant spans only."""
    tokens: list[int] = []
    loss_mask: list[int] = []
    for i in range(len(messages)):
        try:
            ids_prev = (
                _render_ids(tokenizer, messages[:i], tools=tools, add_generation_prompt=False)
                if i > 0
                else []
            )
            ids_curr = _render_ids(tokenizer, messages[: i + 1], tools=tools, add_generation_prompt=False)
        except Exception as exc:
            logger.warning("[offload_sft] apply_chat_template failed at message %d: %s", i, exc)
            return None
        chunk = ids_curr[len(ids_prev) :]
        if not chunk and i < len(messages):
            return None
        tokens.extend(chunk)
        supervise = messages[i].get("role") == "assistant" and (
            supervise_assistant_indices is None or i in supervise_assistant_indices
        )
        loss_mask.extend([1 if supervise else 0] * len(chunk))
    return tokens, loss_mask


def _leading_prompt_len(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None,
) -> int | None:
    try:
        first_asst = next(i for i, m in enumerate(messages) if m.get("role") == "assistant")
        return len(
            _render_ids(tokenizer, messages[:first_asst], tools=tools, add_generation_prompt=True)
        )
    except StopIteration:
        return None
    except Exception:
        return None


def build_multiturn_sft_messages(
    turn_costs: list[dict[str, Any]],
    *,
    rng: random.Random | None = None,
) -> tuple[list[dict[str, Any]], set[int]] | None:
    """Stitch per-turn history + assistant targets; return supervised assistant indices."""
    rnd = rng or random.Random()
    tag_p = sft_tag_prob()
    eligible = [tc for tc in turn_costs if isinstance(tc, dict) and turn_eligible_for_sft(tc)]
    max_n = sft_max_samples()
    if max_n > 0 and len(eligible) > max_n:
        eligible = eligible[-max_n:]
    if not eligible:
        return None

    segments: list[tuple[list[dict[str, Any]], set[int]]] = []
    messages: list[dict[str, Any]] = []
    supervised_indices: set[int] = set()
    for i, tc in enumerate(eligible):
        hist = [_canonical_chat_message(m) for m in tc["sft_history_messages"]]
        include_tag = bool(_offload_teacher_available(tc) and rnd.random() < tag_p)
        assistant = build_assistant_target(tc, include_offload_tag=include_tag)
        if assistant is None:
            return None

        if not messages:
            messages = list(hist)
        else:
            prev_hist = [
                _canonical_chat_message(m) for m in eligible[i - 1]["sft_history_messages"]
            ]
            if len(hist) < len(prev_hist):
                segments.append((messages, supervised_indices))
                logger.info(
                    "[offload_sft] history reset at turn %d (%d -> %d msgs); start new segment",
                    i,
                    len(prev_hist),
                    len(hist),
                )
                messages = list(hist)
                supervised_indices = set()
            else:
                delta = hist[len(prev_hist) :]
                if delta and delta[0].get("role") == "assistant":
                    delta = delta[1:]
                messages.extend(delta)

        messages.append(assistant)
        supervised_indices.add(len(messages) - 1)

    if messages:
        segments.append((messages, supervised_indices))
    if not segments:
        return None
    return max(segments, key=lambda seg: len(seg[1]))


def build_multiturn_sft_token_sequence(
    tokenizer: Any,
    turn_costs: list[dict[str, Any]],
    *,
    tools: list[dict] | None = None,
    max_seq: int | None = None,
    rng: random.Random | None = None,
) -> tuple[list[int], list[int], int] | None:
    """Return ``(tokens, loss_mask, response_length)`` for a full multiturn episode."""
    built_messages = build_multiturn_sft_messages(turn_costs, rng=rng)
    if built_messages is None:
        return None
    messages, supervised_indices = built_messages
    n_supervised = len(supervised_indices)

    if tools is None:
        for tc in reversed(turn_costs):
            if isinstance(tc, dict) and isinstance(tc.get("sft_tools_schema"), list):
                tools = tc.get("sft_tools_schema")
                break

    cap = sft_max_seq_len() if max_seq is None else max_seq
    if cap > 0:
        trimmed = left_trim_messages_to_fit(tokenizer, messages, tools=tools, max_seq=cap)
        if trimmed is None:
            logger.warning(
                "[offload_sft] skip: multiturn conversation exceeds OFFLOAD_SFT_MAX_SEQ_LEN=%d",
                cap,
            )
            return None
        if len(trimmed) < len(messages):
            logger.info(
                "[offload_sft] left-trim multiturn %d -> %d messages to fit max_seq=%d",
                len(messages),
                len(trimmed),
                cap,
            )
        messages = trimmed

    asst_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    supervised_indices = set(asst_indices[-n_supervised:]) if n_supervised else set()

    tokenized = _tokenize_multiturn(
        tokenizer, messages, tools=tools, supervise_assistant_indices=supervised_indices
    )
    if tokenized is None:
        return None
    tokens, full_mask = tokenized
    if not tokens or not any(full_mask):
        return None

    leading = _leading_prompt_len(tokenizer, messages, tools=tools)
    if leading is None or leading >= len(tokens):
        return None

    response_length = len(tokens) - leading
    loss_mask = full_mask[leading:]
    if len(loss_mask) != response_length or not any(loss_mask):
        return None
    return tokens, loss_mask, response_length


def build_sft_token_sequence(
    tokenizer: Any,
    *,
    history: list[dict[str, Any]],
    tools: list[dict] | None,
    qwen_think: str,
    glm_think: str,
    glm_content: str,
    glm_tool_calls: list[Any] | None,
    max_seq: int | None = None,
    include_offload_tag: bool = False,
    offload_n: int | None = None,
) -> tuple[list[int], list[int], int] | None:
    """Single-turn helper (tests / legacy). Prefer :func:`build_multiturn_sft_token_sequence`."""
    tc: dict[str, Any] = {
        "valid_offload": True,
        "qwen_think_prefix": qwen_think,
        "slm_raw_output": qwen_think,
        "teacher_think": glm_think,
        "teacher_content": glm_content,
        "teacher_tool_calls": glm_tool_calls,
        "sft_history_messages": history,
        "sft_tools_schema": tools,
    }
    if include_offload_tag and offload_n is not None:
        tc["n"] = offload_n
    elif include_offload_tag:
        tc["n"] = 3

    class _FixedRng:
        def __init__(self, always_include: bool) -> None:
            self._always_include = always_include

        def random(self) -> float:
            return 0.0 if self._always_include else 1.0

    return build_multiturn_sft_token_sequence(
        tokenizer,
        [{**tc, "sft_history_messages": history}],
        tools=tools,
        max_seq=max_seq,
        rng=_FixedRng(include_offload_tag),
    )


def build_sft_samples(
    *,
    grpo_samples: list[Sample],
    tokenizer: Any,
    grading_solved: bool,
) -> list[Sample]:
    """Emit one multiturn SFT ``Sample`` per solved episode (all assistant turns)."""
    if sft_lambda() <= 0.0:
        return []
    if not grading_solved or not grpo_samples or tokenizer is None:
        return []

    base = grpo_samples[0]
    md = dict(getattr(base, "metadata", None) or {})
    turn_costs = list(md.get("turn_costs") or (md.get("offload_stats") or {}).get("turn_costs") or [])
    if not turn_costs:
        return []

    for s in grpo_samples:
        tmd = dict(getattr(s, "train_metadata", None) or {})
        tmd.setdefault("objective", "grpo")
        s.train_metadata = tmd

    built = build_multiturn_sft_token_sequence(tokenizer, turn_costs)
    if built is None:
        logger.warning(
            "[offload_sft] skip SFT for solved index=%s: multiturn build/tokenize failed (%d turns)",
            int(getattr(base, "index", 0) or 0),
            len(turn_costs),
        )
        return []
    tokens, loss_mask, response_length = built

    base_index = int(getattr(base, "index", 0) or 0)
    group_index = getattr(base, "group_index", None)
    base_rid = getattr(base, "rollout_id", None)
    if base_rid is None:
        base_rid = base_index

    n_assistant = sum(1 for tc in turn_costs if turn_eligible_for_sft(tc))
    sft = Sample(
        index=base_index,
        group_index=group_index,
        rollout_id=base_rid,
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
            "sft_multiturn": True,
            "sft_assistant_turns": n_assistant,
            "instance_id": md.get("instance_id"),
            "grading_solved": True,
        },
        train_metadata={
            "objective": "sft",
            "sft_multiturn": True,
            "pack_singleton": True,
        },
    )
    logger.info(
        "[offload_sft] emitted 1 multiturn SFT sample from index=%s (%d assistant turns, λ=%.4f, tag_p=%.2f)",
        base_index,
        n_assistant,
        sft_lambda(),
        sft_tag_prob(),
    )
    return [sft]


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
