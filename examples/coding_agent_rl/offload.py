"""Mid-turn LLM offload for coding-agent RL.

Per agent round (every Claude Code / Codex request):

1. Adapter always calls the local SLM first.
2. If the SLM emits ``<|llm_offload|>N<|/llm_offload|>`` *inside thinking*
   (before ``</think>``; Qwen often omits the opening ``<think>`` from output_ids),
   call remote LLM with thinking selected by ``N`` (0=off, 1-3=low, 4-6=high, 7-9=max)
   via ``chat_template_kwargs.thinking`` / ``reasoning_effort``.
   Spans after ``</think>`` do not call the remote LLM and incur a think-format reward penalty.
3. Compose SLM prefix + GLM continuation into one complete assistant reply and
   only then flush it to the agent.

System-prompt contract:
  - SLM: Claude Code's full system (incl. ``gitStatus``) + ``OFFLOAD_SYSTEM_PROMPT_APPEND``
    injected by the coding adapter on each request
  - GLM: agent system + ``CODING_HANDOFF_PROMPT`` in OpenAI chat.completions form
    (``OFFLOAD_SYSTEM_PROMPT_APPEND`` is stripped if present; history keeps
    structured ``tool_calls`` / ``role: tool`` + ``tool_call_id``; the agent's
    ``tools`` schema is forwarded as a top-level ``tools`` field, same split as
    Claude Code → slime)

Only local-model ``output_ids`` are trained by default. After a successful GLM
call, the continuation may be tokenized and appended to ``turn.output_ids``
with ``output_loss_mask=0`` so rollout dumps contain the full assistant turn
without contributing to the policy loss (``SLIME_OFFLOAD_EMBED_IN_TRAJECTORY``).

Train shaping (when ``SLIME_AGENT_OFFLOAD=1``):
  - Per-turn costs in ``offload_stats["turn_costs"]``; ``r_i`` via
    :func:`compute_turn_rewards` (baseline completion = ``metadata.completion_tokens / n``).
  - Hard compact filter + group α:
    :func:`compact_and_shape_group_help_seeking_rewards`.
  - Turn-painted advantages:
    ``examples.coding_agent_rl.offload_turn_advantage.compute_turn_advantages``.
  - tmax ``empty_patch`` does **not** multiply ``empty_scale``.

Enable with ``SLIME_AGENT_OFFLOAD=1`` (see ``generate.py`` Offload* adapters).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
from typing import Any

import requests

from slime.agent.adapters.common import Reply, Session, flatten_content, tool_call_dict
from slime.agent.chrome_trace import chrome_span, session_trace_ctx
from slime.agent.trajectory import TurnRecord

logger = logging.getLogger(__name__)

OFFLOAD_OPEN = "<|llm_offload|>"
OFFLOAD_CLOSE = "<|/llm_offload|>"
# Back-compat alias used by docs / older call sites.
OFFLOAD_TAG = OFFLOAD_OPEN
_OFFLOAD_SPAN_RE = re.compile(re.escape(OFFLOAD_OPEN) + r"(\d)" + re.escape(OFFLOAD_CLOSE))

DEFAULT_DASHSCOPE_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "http://208.64.254.187:8001/v1")
DEFAULT_DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_MODEL", "deepseek-v4-flash-0731")
DEFAULT_OFFLOAD_MAX_TOKENS = int(os.environ.get("OFFLOAD_MAX_TOKENS", "8192"))

COST_SMALL_PROMPT = float(os.environ.get("OFFLOAD_COST_SMALL_PROMPT", "0.017"))
COST_SMALL_OUTPUT = float(os.environ.get("OFFLOAD_COST_SMALL_OUTPUT", "0.026"))
COST_GLM_INPUT = float(os.environ.get("OFFLOAD_COST_GLM_INPUT", "0.315"))
COST_GLM_OUTPUT = float(os.environ.get("OFFLOAD_COST_GLM_OUTPUT", "1.0"))

# Fallback baseline when dataset metadata has no ``usage`` (GLM-only tokens).
_DEFAULT_BASELINE_PROMPT_TOKENS = int(os.environ.get("OFFLOAD_BASELINE_PROMPT_TOKENS", "1093525"))
_DEFAULT_BASELINE_COMPLETION_TOKENS = int(os.environ.get("OFFLOAD_BASELINE_COMPLETION_TOKENS", "15207"))

# Appended after the black-box agent's system text when calling remote GLM.
CODING_HANDOFF_PROMPT = (
    "You are a helpful assistant completing a task that was partially solved "
    "by a smaller local model before offload.\n"
    "Collaborative handoff protocol:\n"
    "- The assistant message may contain <part_think>...</part_think> with reasoning "
    "the small model already produced before offload.\n"
    "- Because part of the reasoning is already in <part_think>, continue from "
    "where it stopped: your reasoning channel should pick up at the first "
    "unresolved step and carry forward to the final answer. Do not repeat, "
    "paraphrase, or re-derive anything already present in <part_think>.\n"
    "- If <part_think> already concludes the task, proceed directly to the final answer.\n"
    "- Put the user-facing answer only in normal assistant content. Never quote "
    "or mention <part_think> to the user.\n"
    "- <part_think> is an internal marker, not user input."
)

# Appended to the *SLM* system after Claude Code's full system (incl. gitStatus).
# Injected by the coding adapter on each request.
OFFLOAD_SYSTEM_PROMPT_APPEND = (
    "For very difficult steps, you can output "
    f"{OFFLOAD_OPEN}N{OFFLOAD_CLOSE} where N is 0-9 indicating the thinking "
    "level for a more capable model."
)
# Back-compat alias.
DEFAULT_OFFLOAD_SWE_PROMPT = OFFLOAD_SYSTEM_PROMPT_APPEND

# Train reward: subtract this once if any offload span appeared outside <think>.
DEFAULT_OFFLOAD_THINK_FORMAT_PENALTY = 0.25
# Extra per-turn penalty for malformed offload tags (orphan OPEN, bad N, etc.).
DEFAULT_OFFLOAD_MALFORMED_PENALTY = 0.25
# help_seeking_reward: partial credit when unsolved but the SLM asked for help in-think.
DEFAULT_OFFLOAD_SEEK_ALPHA = 0.1
DEFAULT_OFFLOAD_SEEK_EMPTY_SCALE = 0.5
DEFAULT_OFFLOAD_UNIQUE_SOLVER_BONUS = 0.15
# Compact filter defaults (hard remove_sample).
DEFAULT_COMPACT_ORPHAN_OPEN_K = 8
DEFAULT_COMPACT_OPEN_CLOSE_RATIO = 20.0
DEFAULT_COMPACT_SPECIAL_TOKEN_RATIO = 0.5
DEFAULT_COMPACT_SPECIAL_TOKEN_RUN = 32


def offload_system_append_text() -> str:
    """Effective SLM-only offload instructions (env override or default)."""
    return (os.environ.get("SLIME_AGENT_OFFLOAD_SYSTEM_APPEND") or OFFLOAD_SYSTEM_PROMPT_APPEND).strip()


def inject_offload_into_request_body(body: dict) -> None:
    """Append offload instructions to the request system field (in-place).

    Runs on each adapter turn so the SLM sees: CC system (… gitStatus) + append.
    Idempotent if the append text is already present. No-op when offload is off.
    """
    if not offload_enabled():
        return
    text = offload_system_append_text()
    if not text:
        return

    if "system" in body and body.get("system") is not None:
        body["system"] = _append_to_anthropic_system(body.get("system"), text)
        return

    messages = body.get("messages")
    if isinstance(messages, list):
        _append_to_openai_messages(messages, text)


def _append_to_anthropic_system(system: Any, text: str) -> Any:
    flat = flatten_content(system) if system else ""
    if text in flat:
        return system
    if system is None or system == "":
        return text
    if isinstance(system, str):
        return system.rstrip() + "\n\n" + text
    if isinstance(system, list):
        out = [b for b in system if isinstance(b, dict)]
        out.append({"type": "text", "text": "\n\n" + text})
        return out if out else text
    return flat.rstrip() + "\n\n" + text


def _append_to_openai_messages(messages: list, text: str) -> None:
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        flat = content if isinstance(content, str) else flatten_content(content)
        if text in (flat or ""):
            return
        if isinstance(content, str):
            msg["content"] = content.rstrip() + "\n\n" + text
        else:
            msg["content"] = (flat or "").rstrip() + "\n\n" + text
        return
    messages.insert(0, {"role": "system", "content": text})


def offload_enabled() -> bool:
    return os.environ.get("SLIME_AGENT_OFFLOAD", "").strip().lower() in ("1", "true", "yes", "on")


def efficiency_lambda() -> float:
    return float(os.environ.get("OFFLOAD_EFFICIENCY_LAMBDA", "0.6"))


def think_format_penalty() -> float:
    return float(os.environ.get("OFFLOAD_THINK_FORMAT_PENALTY", str(DEFAULT_OFFLOAD_THINK_FORMAT_PENALTY)))


def malformed_penalty() -> float:
    return float(os.environ.get("OFFLOAD_MALFORMED_PENALTY", str(DEFAULT_OFFLOAD_MALFORMED_PENALTY)))


def reward_mode() -> str:
    """``cost_aware`` (default) or ``help_seeking`` — see :func:`help_seeking_reward`."""
    mode = (os.environ.get("OFFLOAD_REWARD_MODE") or "cost_aware").strip().lower()
    if mode in ("help_seeking", "help-seeking", "seek"):
        return "help_seeking"
    return "cost_aware"


def seek_alpha() -> float:
    return float(os.environ.get("OFFLOAD_SEEK_ALPHA", str(DEFAULT_OFFLOAD_SEEK_ALPHA)))


def seek_empty_scale() -> float:
    return float(os.environ.get("OFFLOAD_SEEK_EMPTY_SCALE", str(DEFAULT_OFFLOAD_SEEK_EMPTY_SCALE)))


def unique_solver_bonus() -> float:
    return float(os.environ.get("OFFLOAD_UNIQUE_SOLVER_BONUS", str(DEFAULT_OFFLOAD_UNIQUE_SOLVER_BONUS)))


def seek_only_when_all_wrong() -> bool:
    """If true, help-seeking α is granted only when every sibling in the group failed.

    Per-sample finish then uses ``encourage_seek=False``; group shaping is applied
    later by :func:`shape_group_help_seeking_rewards` (``--rollout-sample-filter-path``).
    """
    return os.environ.get("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _api_key() -> str:
    return (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()


def _base_url() -> str:
    return (os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_DASHSCOPE_BASE_URL).rstrip("/")


def _model() -> str:
    return os.environ.get("DASHSCOPE_MODEL") or DEFAULT_DASHSCOPE_MODEL


def _max_tokens() -> int:
    return int(os.environ.get("OFFLOAD_MAX_TOKENS", str(DEFAULT_OFFLOAD_MAX_TOKENS)))


def parse_offload_directive(raw: str) -> tuple[int, str] | None:
    """Return ``(N, text_before_span)`` for the first complete offload span, else None.

    Does not require the span to be inside ``<think>``; use
    :func:`offload_span_inside_think` / :func:`parse_valid_offload_directive`
    for the protocol that only fires GLM when the span is in-think.
    """
    match = _OFFLOAD_SPAN_RE.search(raw)
    if match is None:
        return None
    return int(match.group(1)), raw[: match.start()]


def offload_span_inside_think(raw: str) -> bool:
    """True iff the first complete offload span is still in the thinking region.

    Qwen / PyroDash note: the opening ``<think>`` is usually injected by the chat
    template into the *prompt*, so ``output_ids`` often start with think text and
    only emit ``</think>`` (see rollout dumps). Mid-turn offload that stops on the
    close token may have neither tag yet — that still counts as in-think.

    Rules (first complete offload span at ``pos``):
      1. Inside an explicit ``<think>`` … (unclosed) region → in-think
      2. ``pos`` before the first ``</think>`` → in-think
      3. No ``</think>`` in ``raw`` → in-think (stopped during initial think)
      4. Else → outside think (visible / tool body after think ended)
    """
    match = _OFFLOAD_SPAN_RE.search(raw)
    if match is None:
        return False
    pos = match.start()
    before = raw[:pos]

    last_open = before.rfind("<think>")
    if last_open >= 0 and "</think>" not in before[last_open:]:
        return True

    first_close = raw.find("</think>")
    if first_close < 0:
        return True
    return pos < first_close


def parse_valid_offload_directive(raw: str) -> tuple[int, str] | None:
    """Like :func:`parse_offload_directive`, but only if the span is inside think."""
    parsed = parse_offload_directive(raw)
    if parsed is None or not offload_span_inside_think(raw):
        return None
    return parsed


def reasoning_from_n(n: int) -> tuple[bool, str | None]:
    """Map digit N -> ``(enable_thinking, reasoning_effort)``."""
    if n <= 0:
        return False, None
    if n <= 3:
        return True, "low"
    if n <= 6:
        return True, "high"
    return True, "max"


def _ensure_stats(session: Session) -> dict[str, Any]:
    stats = session.offload_stats
    if not stats:
        stats.update(
            {
                "offload_count": 0,
                "offload_outside_think_count": 0,
                "small_prompt_tokens": 0,
                "small_output_tokens": 0,
                "glm_input_tokens": 0,
                "glm_output_tokens": 0,
                "last_offload_n": None,
                "last_reasoning_effort": None,
                "turn_costs": [],
                "max_steps_reached": False,
            }
        )
    stats.setdefault("turn_costs", [])
    stats.setdefault("max_steps_reached", False)
    return stats


def analyze_offload_tags(raw: str) -> dict[str, Any]:
    """Count offload OPEN/CLOSE tags and classify format violations.

    Valid complete span: ``<|llm_offload|>N<|/llm_offload|>`` with single digit N.
    """
    text = raw or ""
    open_count = text.count(OFFLOAD_OPEN)
    close_count = text.count(OFFLOAD_CLOSE)
    valid = list(_OFFLOAD_SPAN_RE.finditer(text))
    valid_count = len(valid)

    # Consume matched valid spans; leftover OPEN/CLOSE are malformed/orphan.
    consumed = [(m.start(), m.end()) for m in valid]
    orphan_open = 0
    malformed = 0
    pos = 0
    while True:
        oi = text.find(OFFLOAD_OPEN, pos)
        if oi < 0:
            break
        if any(s <= oi < e for s, e in consumed):
            pos = oi + len(OFFLOAD_OPEN)
            continue
        # Incomplete or bad payload before CLOSE.
        ci = text.find(OFFLOAD_CLOSE, oi + len(OFFLOAD_OPEN))
        if ci < 0:
            orphan_open += 1
            pos = oi + len(OFFLOAD_OPEN)
            continue
        payload = text[oi + len(OFFLOAD_OPEN) : ci]
        if not (len(payload) == 1 and payload.isdigit()):
            malformed += 1
        else:
            # Digit span that somehow wasn't matched (shouldn't happen).
            malformed += 1
        pos = ci + len(OFFLOAD_CLOSE)

    # CLOSE without a preceding unmatched OPEN in leftover stream.
    leftover_close = max(0, close_count - valid_count - malformed)
    # Prefer counting unmatched closes as malformed.
    if leftover_close > 0:
        malformed += leftover_close

    # Longest consecutive OPEN spam (open markers with no close between).
    max_open_run = 0
    run = 0
    scan = 0
    while scan < len(text):
        if text.startswith(OFFLOAD_OPEN, scan):
            run += 1
            max_open_run = max(max_open_run, run)
            scan += len(OFFLOAD_OPEN)
            continue
        if text.startswith(OFFLOAD_CLOSE, scan):
            run = 0
            scan += len(OFFLOAD_CLOSE)
            continue
        scan += 1

    special_marks = open_count + close_count
    return {
        "open_count": open_count,
        "close_count": close_count,
        "valid_count": valid_count,
        "orphan_open_count": orphan_open,
        "malformed_count": malformed,
        "max_open_run": max_open_run,
        "special_mark_count": special_marks,
    }


def record_local_turn_tokens(session: Session, turn: TurnRecord, *, raw_output: str = "") -> dict[str, Any]:
    """Accumulate per-round SLM tokens and append a ``turn_costs`` ledger entry."""
    stats = _ensure_stats(session)
    prompt_n = len(turn.prompt_ids or [])
    # Called before GLM embed → output_ids are SLM-only.
    out_n = len(turn.output_ids or [])
    stats["small_prompt_tokens"] = int(stats.get("small_prompt_tokens", 0)) + prompt_n
    stats["small_output_tokens"] = int(stats.get("small_output_tokens", 0)) + out_n

    tag = analyze_offload_tags(raw_output)
    outside = False
    valid_offload = parse_valid_offload_directive(raw_output) is not None
    if parse_offload_directive(raw_output) is not None and not valid_offload:
        outside = True
    entry: dict[str, Any] = {
        "small_prompt_tokens": prompt_n,
        "small_output_tokens": out_n,
        "glm_input_tokens": 0,
        "glm_output_tokens": 0,
        "response_token_len": out_n,
        "valid_offload": valid_offload,
        "outside_think": outside,
        "orphan_open_count": int(tag["orphan_open_count"]),
        "malformed_count": int(tag["malformed_count"]),
        "open_count": int(tag["open_count"]),
        "close_count": int(tag["close_count"]),
        "max_open_run": int(tag["max_open_run"]),
        "special_mark_count": int(tag["special_mark_count"]),
    }
    stats["turn_costs"].append(entry)
    return entry


def _estimate_tokens(text: str) -> int:
    """Rough token estimate when the remote API omits ``usage``."""
    return max(0, (len(text) + 3) // 4)


def _record_glm_usage(
    stats: dict[str, Any],
    usage: dict[str, Any] | None,
    *,
    messages: list[dict[str, Any]],
    content: str,
    think: str,
) -> tuple[int, int]:
    if usage:
        inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    else:
        chunks: list[str] = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, str) and c:
                chunks.append(c)
            for tc in m.get("tool_calls") or []:
                chunks.append(json.dumps(tc, ensure_ascii=False))
        inp = _estimate_tokens("\n".join(chunks))
        out = _estimate_tokens(f"{think}{content}")
        logger.warning(
            "[coding_agent_offload] remote usage missing; estimated glm_in=%d glm_out=%d",
            inp,
            out,
        )
    stats["glm_input_tokens"] = int(stats.get("glm_input_tokens", 0)) + inp
    stats["glm_output_tokens"] = int(stats.get("glm_output_tokens", 0)) + out
    return inp, out


def _offload_prefix(raw: str) -> str:
    parsed = parse_offload_directive(raw)
    if parsed is None:
        # Incomplete / legacy open-only tag: drop from first open marker.
        idx = raw.find(OFFLOAD_OPEN)
        prefix = raw[:idx] if idx >= 0 else raw
    else:
        prefix = parsed[1]
    return _strip_offload_tag_from_text(prefix).strip()


def _assistant_content_for_openai(msg: dict[str, Any]) -> str:
    """Visible assistant text for GLM: optional ``<think>`` + content (no tool_calls)."""
    parts: list[str] = []
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        parts.append(f"<think>\n{reasoning.strip()}\n</think>")
    content = flatten_content(msg.get("content"))
    if content:
        parts.append(content)
    return _strip_offload_tag_from_text("\n\n".join(parts)).strip()


def _arguments_as_openai_json(arguments: Any) -> str:
    """OpenAI chat.completions expects ``function.arguments`` as a JSON string."""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
    except TypeError:
        return json.dumps({"_raw": str(arguments)}, ensure_ascii=False)


def _normalize_openai_tool_calls(
    tool_calls: list[Any] | None,
    *,
    id_prefix: str,
) -> list[dict[str, Any]]:
    """Translate adapter ``tool_calls`` into OpenAI wire shape (with ids)."""
    out: list[dict[str, Any]] = []
    for i, call in enumerate(tool_calls or []):
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") or call.get("name") or "tool"
        arguments = function.get("arguments")
        if arguments is None:
            arguments = call.get("arguments", {})
        call_id = call.get("id") or f"{id_prefix}-{i}"
        out.append(
            {
                "id": str(call_id),
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": _arguments_as_openai_json(arguments),
                },
            }
        )
    return out


def _offload_system_append_variants() -> list[str]:
    """Text that belongs on the *SLM* system prompt only (not GLM)."""
    variants = [OFFLOAD_SYSTEM_PROMPT_APPEND, offload_system_append_text()]
    # Preserve order, drop empties/dupes.
    out: list[str] = []
    for v in variants:
        if v and v not in out:
            out.append(v)
    return out


def strip_offload_system_append(text: str) -> str:
    """Delete SLM-only offload instructions from system text before calling GLM.

    Contract:
      SLM / CC request  <- CC system (… gitStatus) + OFFLOAD_SYSTEM_PROMPT_APPEND
      GLM               <- agent system (offload text removed) + CODING_HANDOFF_PROMPT
    """
    out = text
    for variant in _offload_system_append_variants():
        if variant and variant in out:
            out = out.replace(variant, "")
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def build_offload_messages(translated: list[dict], raw_output: str) -> list[dict[str, Any]]:
    """Build GLM chat messages in OpenAI ``chat.completions`` tool protocol.

    Emits ``system`` / ``user`` / ``assistant`` (+ structured ``tool_calls``) /
    ``tool`` (+ ``tool_call_id``), then a final ``assistant`` ``<part_think>``
    handoff turn.

    Removes SLM-only bits that must not reach GLM:
      - ``OFFLOAD_SYSTEM_PROMPT_APPEND`` from system text
      - ``<|llm_offload|>N<|/llm_offload|>`` spans from history / prefix
    Those markers are kept on the CC reply path (see ``compose_complete_assistant``).
    """
    agent_system_parts: list[str] = []
    rest: list[dict[str, Any]] = []
    pending_tool_ids: list[str] = []
    synth_i = 0

    for msg in translated:
        role = str(msg.get("role") or "user")
        if role == "system":
            text = flatten_content(msg.get("content"))
            cleaned_system = strip_offload_system_append(text) if text else ""
            if cleaned_system:
                agent_system_parts.append(cleaned_system)
            continue

        if role == "user":
            text = _strip_offload_tag_from_text(flatten_content(msg.get("content")))
            if text:
                rest.append({"role": "user", "content": text})
            continue

        if role == "assistant":
            content = _assistant_content_for_openai(msg)
            tool_calls = _normalize_openai_tool_calls(
                msg.get("tool_calls"),
                id_prefix=f"chatcmpl-tool-offload{synth_i}",
            )
            synth_i += 1
            if not content and not tool_calls:
                continue
            out_msg: dict[str, Any] = {"role": "assistant", "content": content or None}
            if tool_calls:
                out_msg["tool_calls"] = tool_calls
                pending_tool_ids = [tc["id"] for tc in tool_calls]
            else:
                pending_tool_ids = []
            rest.append(out_msg)
            continue

        if role == "tool":
            text = _strip_offload_tag_from_text(flatten_content(msg.get("content")))
            tool_call_id = msg.get("tool_call_id") or msg.get("tool_use_id")
            if not tool_call_id and pending_tool_ids:
                tool_call_id = pending_tool_ids.pop(0)
            if not tool_call_id:
                tool_call_id = f"chatcmpl-tool-orphan-{synth_i}"
                synth_i += 1
            rest.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_call_id),
                    "content": text if text else "",
                }
            )
            continue

        # Unknown roles -> user text fallback.
        text = _strip_offload_tag_from_text(flatten_content(msg.get("content")))
        if text:
            rest.append({"role": "user", "content": text})

    system = "\n\n".join([*agent_system_parts, CODING_HANDOFF_PROMPT]).strip()
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend(rest)

    # Prefix only (no offload span) -> <part_think>; never send the tag to GLM.
    partial = _offload_prefix(raw_output)
    cleaned = partial.replace("<think>", "").replace("</think>", "").strip()
    if cleaned:
        messages.append({"role": "assistant", "content": f"<part_think>{cleaned}</part_think>"})
    return messages


def _normalize_openai_tools(tools_schema: list[dict] | None) -> list[dict[str, Any]] | None:
    """Pass-through / light-normalize chat-template tools into OpenAI ``tools``."""
    if not tools_schema:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools_schema:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if function is not None:
            name = function.get("name")
            if not name:
                continue
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": function.get("description", ""),
                        "parameters": function.get("parameters")
                        or {"type": "object", "properties": {}},
                    },
                }
            )
            continue
        name = tool.get("name")
        if not name:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema")
                    or tool.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return out or None


def _parse_openai_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    """Normalize ``message.tool_calls`` from a chat.completions response."""
    if not isinstance(raw_tool_calls, list):
        return []
    out: list[dict[str, Any]] = []
    for i, call in enumerate(raw_tool_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") or call.get("name")
        if not name:
            continue
        arguments = function.get("arguments")
        if arguments is None:
            arguments = call.get("arguments", {})
        out.append(
            {
                "id": str(call.get("id") or f"chatcmpl-tool-glm-{i}"),
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": _arguments_as_openai_json(arguments),
                },
            }
        )
    return out


def _openai_tool_calls_to_anthropic_blocks(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for tc in tool_calls:
        function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        args_raw = function.get("arguments")
        if isinstance(args_raw, str):
            try:
                args_obj = json.loads(args_raw)
            except json.JSONDecodeError:
                args_obj = {"_raw": args_raw}
        elif isinstance(args_raw, dict):
            args_obj = args_raw
        else:
            args_obj = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(tc.get("id") or f"toolu_{secrets.token_hex(8)}"),
                "name": str(function.get("name") or "tool"),
                "input": args_obj,
            }
        )
    return blocks


def _call_remote_chat_sync(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    enable_thinking: bool,
    reasoning_effort: str | None,
    tools: list[dict[str, Any]] | None = None,
    timeout: float = 600.0,
) -> tuple[str, str, dict[str, Any] | None, list[dict[str, Any]]]:
    api_key = _api_key()
    if not api_key:
        return "[Error: DASHSCOPE_API_KEY not set]", "", None, []
    if max_tokens <= 0:
        return "[Error: no remaining offload token budget]", "", None, []

    url = f"{_base_url()}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # deepseek-v4-flash: chat_template_kwargs.thinking (+ reasoning_effort).
    # Older GLM gateways used enable_thinking; this endpoint expects thinking.
    chat_kwargs: dict[str, Any] = {"thinking": enable_thinking}
    if enable_thinking and reasoning_effort:
        chat_kwargs["reasoning_effort"] = reasoning_effort
    body: dict[str, Any] = {
        "model": _model(),
        "messages": messages,
        "max_tokens": max_tokens,
        "chat_template_kwargs": chat_kwargs,
    }
    openai_tools = _normalize_openai_tools(tools)
    if openai_tools:
        # Same split as Claude Code → slime: tools schema is request-level, not
        # only baked into system text.
        body["tools"] = openai_tools
    try:
        response = requests.post(url, headers=headers, json=body, timeout=timeout)
        if response.status_code != 200:
            return f"[Error: status {response.status_code}: {response.text[:400]}]", "", None, []
        data = response.json()
        message = data["choices"][0].get("message", {})
        think = str(message.get("reasoning") or message.get("reasoning_content") or "")
        content = str(message.get("content") or "")
        tool_calls = _parse_openai_tool_calls(message.get("tool_calls"))
        usage = data.get("usage")
        if usage is not None and not isinstance(usage, dict):
            usage = None
        return content, think, usage, tool_calls
    except Exception as exc:
        return f"[Error: remote call failed: {exc}]", "", None, []


async def call_remote_chat(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    enable_thinking: bool,
    reasoning_effort: str | None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, str, dict[str, Any] | None, list[dict[str, Any]]]:
    return await asyncio.to_thread(
        _call_remote_chat_sync,
        messages,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        tools=tools,
    )


def _strip_offload_tag_from_text(text: str) -> str:
    text = _OFFLOAD_SPAN_RE.sub("", text)
    # Drop dangling open/close markers if the span was truncated.
    return text.replace(OFFLOAD_OPEN, "").replace(OFFLOAD_CLOSE, "").rstrip()


def _join_nonempty(*parts: str, sep: str = "\n") -> str:
    return sep.join(p for p in parts if p)


def compose_complete_assistant(
    *,
    slm_content: str,
    glm_content: str,
    glm_think: str,
) -> tuple[str, str]:
    """Merge SLM prefix + GLM continuation into one assistant (text, think) pair.

    Keeps ``<|llm_offload|>N<|/llm_offload|>`` in the text returned to CC.
    GLM never sees that span (stripped in ``build_offload_messages``).
    """
    text = _join_nonempty("", glm_content)
    think = _join_nonempty(slm_content, glm_think, sep="")
    return text, think


def embed_offload_in_trajectory_enabled() -> bool:
    """Whether to append GLM tokens into ``turn.output_ids`` with loss_mask=0."""
    return (os.environ.get("SLIME_OFFLOAD_EMBED_IN_TRAJECTORY") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _embed_max_tokens() -> int | None:
    raw = (os.environ.get("SLIME_OFFLOAD_EMBED_MAX_TOKENS") or "").strip()
    if not raw or raw == "0":
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def build_glm_trajectory_suffix(*, raw_output: str, glm_content: str, glm_think: str) -> str:
    """Text to tokenize after SLM ``output_ids`` so the dump mirrors the agent turn.

    SLM generation usually stops at ``<|/llm_offload|>`` mid-think (no ``</think>``).
    Agent-facing compose uses ``think = raw_output + glm_think`` and
    ``content = glm_content``. We append the GLM pieces plus a closing think tag
    when needed so decoded trajectories stay readable.
    """
    parts: list[str] = []
    if glm_think:
        parts.append(glm_think)
    if "</think>" not in raw_output:
        parts.append("\n</think>\n")
    if glm_content:
        parts.append(glm_content if not parts else f"\n{glm_content}")
    return "".join(parts)


def append_glm_tokens_to_turn(
    turn: TurnRecord,
    *,
    tokenizer: Any,
    raw_output: str,
    glm_content: str,
    glm_think: str,
) -> None:
    """Extend ``turn.output_ids`` with tokenized GLM text; mark those tokens mask=0.

    Mutates the turn's list fields in place (``TurnRecord`` is frozen but lists
    are mutable). SLM tokens keep loss_mask=1; GLM suffix is loss_mask=0.
    """
    if not embed_offload_in_trajectory_enabled():
        return
    if tokenizer is None:
        logger.warning("[coding_agent_offload] tokenizer missing; skip GLM trajectory embed")
        return
    suffix = build_glm_trajectory_suffix(
        raw_output=raw_output, glm_content=glm_content, glm_think=glm_think
    )
    if not suffix:
        return
    try:
        glm_ids = list(tokenizer.encode(suffix, add_special_tokens=False))
    except Exception:
        logger.exception("[coding_agent_offload] failed to tokenize GLM suffix; skip embed")
        return
    max_toks = _embed_max_tokens()
    if max_toks is not None and len(glm_ids) > max_toks:
        glm_ids = glm_ids[:max_toks]
    if not glm_ids:
        return

    slm_n = len(turn.output_ids or [])
    mask = turn.output_loss_mask
    if not mask:
        mask.extend([1] * slm_n)
    elif len(mask) != slm_n:
        raise ValueError(
            f"output_loss_mask length {len(mask)} != output_ids length {slm_n} before GLM embed"
        )

    # Keep logprobs aligned when present; pad with 0.0 for the GLM suffix.
    lps = turn.output_log_probs
    if lps:
        if len(lps) < slm_n:
            lps.extend([0.0] * (slm_n - len(lps)))
        elif len(lps) > slm_n:
            del lps[slm_n:]
    else:
        # No SLM logprobs recorded; leave empty unless we already started a mask
        # (then pad zeros for the whole sequence so lengths stay consistent).
        if mask:
            lps.extend([0.0] * slm_n)

    turn.output_ids.extend(glm_ids)
    mask.extend([0] * len(glm_ids))
    if lps:
        lps.extend([0.0] * len(glm_ids))


def amend_reply_with_offload(
    reply: Reply,
    *,
    raw_output: str,
    glm_content: str,
    glm_think: str,
    glm_tool_calls: list[dict[str, Any]] | None = None,
) -> Reply:
    """Replace the SLM-only reply with the composed complete assistant turn for the agent.

    Also see ``append_glm_tokens_to_turn``: GLM text may be embedded into
    ``turn.output_ids`` with ``loss_mask=0`` for dumps; trainable tokens remain SLM.
    """
    mm = dict(reply.manager_message)
    text, think = compose_complete_assistant(
        slm_content=raw_output,
        glm_content=glm_content,
        glm_think=glm_think,
    )
    mm["content"] = text
    if think:
        mm["reasoning_content"] = think
    else:
        mm.pop("reasoning_content", None)

    glm_tool_calls = list(glm_tool_calls or [])
    if glm_tool_calls:
        manager_tcs: list[dict[str, Any]] = []
        for tc in glm_tool_calls:
            function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            args_raw = function.get("arguments")
            if isinstance(args_raw, str):
                try:
                    args_obj = json.loads(args_raw)
                except json.JSONDecodeError:
                    args_obj = {"_raw": args_raw}
            elif isinstance(args_raw, dict):
                args_obj = args_raw
            else:
                args_obj = {}
            manager_tcs.append(tool_call_dict(str(function.get("name") or "tool"), args_obj))
        mm["tool_calls"] = manager_tcs
    else:
        mm.pop("tool_calls", None)

    wire = reply.wire
    if isinstance(wire, tuple) and len(wire) == 2 and isinstance(wire[0], list):
        # Anthropic: thinking + text + tool_use (prefer GLM tool_calls when present).
        slm_tool_blocks = [b for b in wire[0] if b.get("type") == "tool_use"]
        tool_blocks = (
            _openai_tool_calls_to_anthropic_blocks(glm_tool_calls) if glm_tool_calls else slm_tool_blocks
        )
        blocks: list[dict] = []
        if think:
            blocks.append({"type": "thinking", "thinking": think})
        if text or not tool_blocks:
            blocks.append({"type": "text", "text": text})
        blocks.extend(tool_blocks)
        stop_reason = "tool_use" if tool_blocks else "end_turn"
        finish = "tool_calls" if tool_blocks else reply.finish_reason
        return Reply(manager_message=mm, finish_reason=finish, wire=(blocks, stop_reason))

    if isinstance(wire, tuple) and len(wire) == 2 and isinstance(wire[0], dict):
        # OpenAI: one message with merged content / reasoning / tool_calls.
        wm = dict(wire[0])
        if glm_tool_calls:
            wm["tool_calls"] = glm_tool_calls
            if think:
                wm["reasoning_content"] = think
            wm["content"] = text or None
            finish = "tool_calls"
        elif wm.get("tool_calls"):
            if think:
                wm["reasoning_content"] = think
            if text:
                wm["content"] = text
            finish = "tool_calls"
        else:
            wm["content"] = text or None
            if think:
                wm["reasoning_content"] = think
            else:
                wm.pop("reasoning_content", None)
            finish = "stop"
        return Reply(manager_message=mm, finish_reason=finish, wire=(wm, finish))

    return Reply(manager_message=mm, finish_reason=reply.finish_reason, wire=wire)


async def apply_offload_if_needed(
    reply: Reply,
    *,
    raw_output: str,
    translated: list[dict],
    turn: TurnRecord,
    session: Session,
    sid: str,
    tokenizer: Any | None = None,
    tools_schema: list[dict] | None = None,
) -> Reply:
    """Per agent round: account SLM tokens; if in-think offload span, call GLM.

    Protocol: ``<|llm_offload|>N<|/llm_offload|>`` must sit inside ``<think>``.
    A complete span outside think does not call GLM; it increments
    ``offload_outside_think_count`` for the think-format reward penalty.

    On success, optionally appends tokenized GLM text to ``turn.output_ids`` with
    ``output_loss_mask=0`` (see ``SLIME_OFFLOAD_EMBED_IN_TRAJECTORY``).
    """
    if not offload_enabled():
        return reply

    turn_entry = record_local_turn_tokens(session, turn, raw_output=raw_output)
    parsed = parse_valid_offload_directive(raw_output)
    if parsed is None:
        # Complete span exists but not in <think> → format violation, no GLM.
        if parse_offload_directive(raw_output) is not None:
            stats = _ensure_stats(session)
            stats["offload_outside_think_count"] = int(stats.get("offload_outside_think_count", 0)) + 1
            turn_entry["outside_think"] = True
            logger.info(
                "[coding_agent_offload] sid=%s skip GLM: offload span outside <think> "
                "(outside_think#%d)",
                sid,
                stats["offload_outside_think_count"],
            )
        turn_entry["response_token_len"] = len(turn.output_ids or [])
        return reply

    n, _prefix = parsed
    enable_thinking, reasoning_effort = reasoning_from_n(n)

    stats = _ensure_stats(session)
    small_out = len(turn.output_ids or [])
    glm_budget = max(0, _max_tokens() - small_out)
    messages = build_offload_messages(translated, raw_output)
    events, tid, timing = session_trace_ctx(session)
    turn_idx = int((timing or {}).get("current_turn", 0) or 0)
    with chrome_span(
        events,
        "glm_offload",
        cat="llm",
        tid=tid,
        args={
            "turn": turn_idx,
            "n": n,
            "reasoning_effort": reasoning_effort,
            "session_id": sid,
        },
    ):
        content, think, usage, glm_tool_calls = await call_remote_chat(
            messages,
            max_tokens=glm_budget,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            tools=tools_schema,
        )

    stats["offload_count"] = int(stats.get("offload_count", 0)) + 1
    if timing is not None:
        timing["n_offloads"] = int(timing.get("n_offloads", 0) or 0) + 1
    stats["last_offload_n"] = n
    stats["last_reasoning_effort"] = reasoning_effort
    turn_entry["valid_offload"] = True
    gin, gout = _record_glm_usage(stats, usage, messages=messages, content=content, think=think)
    turn_entry["glm_input_tokens"] = gin
    turn_entry["glm_output_tokens"] = gout

    logger.info(
        "[coding_agent_offload] sid=%s offload#%d N=%d thinking=%s effort=%s "
        "glm_budget=%d content_len=%d think_len=%d tool_calls=%d "
        "cum_slm=(%d,%d) cum_glm=(%d,%d)",
        sid,
        stats["offload_count"],
        n,
        enable_thinking,
        reasoning_effort,
        glm_budget,
        len(content),
        len(think),
        len(glm_tool_calls),
        int(stats.get("small_prompt_tokens", 0)),
        int(stats.get("small_output_tokens", 0)),
        int(stats.get("glm_input_tokens", 0)),
        int(stats.get("glm_output_tokens", 0)),
    )
    # N=0: no remote think channel; drop any accidental reasoning payload.
    if not enable_thinking:
        think = ""
    append_glm_tokens_to_turn(
        turn,
        tokenizer=tokenizer,
        raw_output=raw_output,
        glm_content=content,
        glm_think=think,
    )
    turn_entry["response_token_len"] = len(turn.output_ids or [])
    return amend_reply_with_offload(
        reply,
        raw_output=raw_output,
        glm_content=content,
        glm_think=think,
        glm_tool_calls=glm_tool_calls,
    )


def actual_cost(stats: dict[str, Any]) -> float:
    return (
        int(stats.get("small_prompt_tokens", 0)) * COST_SMALL_PROMPT
        + int(stats.get("small_output_tokens", 0)) * COST_SMALL_OUTPUT
        + int(stats.get("glm_input_tokens", 0)) * COST_GLM_INPUT
        + int(stats.get("glm_output_tokens", 0)) * COST_GLM_OUTPUT
    )


def turn_actual_cost(turn: dict[str, Any]) -> float:
    return (
        int(turn.get("small_prompt_tokens", 0)) * COST_SMALL_PROMPT
        + int(turn.get("small_output_tokens", 0)) * COST_SMALL_OUTPUT
        + int(turn.get("glm_input_tokens", 0)) * COST_GLM_INPUT
        + int(turn.get("glm_output_tokens", 0)) * COST_GLM_OUTPUT
    )


def resolve_baseline_completion_tokens(
    usage: dict[str, Any] | None = None,
    *,
    completion_tokens: int | float | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Prefer metadata.completion_tokens, then usage, then default."""
    if completion_tokens is not None:
        return max(0, int(completion_tokens))
    md = metadata or {}
    if md.get("completion_tokens") is not None:
        return max(0, int(md["completion_tokens"]))
    if usage and usage.get("completion_tokens") is not None:
        return max(0, int(usage["completion_tokens"]))
    return _DEFAULT_BASELINE_COMPLETION_TOKENS


def resolve_baseline_prompt_tokens(
    usage: dict[str, Any] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> int:
    md = metadata or {}
    if md.get("prompt_tokens") is not None:
        return max(0, int(md["prompt_tokens"]))
    if usage and usage.get("prompt_tokens") is not None:
        return max(0, int(usage["prompt_tokens"]))
    return _DEFAULT_BASELINE_PROMPT_TOKENS


def baseline_cost(usage: dict[str, Any] | None) -> float:
    if usage:
        prompt_t = int(usage.get("prompt_tokens") or _DEFAULT_BASELINE_PROMPT_TOKENS)
        completion_t = int(usage.get("completion_tokens") or _DEFAULT_BASELINE_COMPLETION_TOKENS)
    else:
        prompt_t = _DEFAULT_BASELINE_PROMPT_TOKENS
        completion_t = _DEFAULT_BASELINE_COMPLETION_TOKENS
    return prompt_t * COST_GLM_INPUT + completion_t * COST_GLM_OUTPUT


def per_turn_baseline_cost(
    *,
    n_turns: int,
    usage: dict[str, Any] | None = None,
    completion_tokens: int | float | None = None,
    metadata: dict[str, Any] | None = None,
) -> float:
    """Per-turn baseline ``b_i``: prompt_default/n + (completion_tokens/n)*GLM_OUT."""
    n = max(int(n_turns), 1)
    prompt_t = resolve_baseline_prompt_tokens(usage, metadata=metadata)
    comp_t = resolve_baseline_completion_tokens(
        usage, completion_tokens=completion_tokens, metadata=metadata
    )
    return (prompt_t / n) * COST_GLM_INPUT + (comp_t / n) * COST_GLM_OUTPUT


def cost_ratio(stats: dict[str, Any], usage: dict[str, Any] | None = None) -> float:
    base = baseline_cost(usage)
    if base <= 0:
        return 0.0
    return actual_cost(stats) / base


def _protocol_is_tmax(protocol: str | None, metadata: dict[str, Any] | None = None) -> bool:
    if protocol is not None:
        return str(protocol).strip().lower() == "tmax"
    md = metadata or {}
    return str(md.get("protocol") or "").strip().lower() == "tmax"


def _apply_empty_scale(
    credit: float,
    *,
    empty_patch: bool,
    empty_scale: float,
    protocol: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> float:
    """Scale α on empty_patch except for tmax (no meaningful git diff)."""
    if not empty_patch or _protocol_is_tmax(protocol, metadata):
        return credit
    return credit * float(empty_scale)


def cost_aware_reward(
    solved: float,
    stats: dict[str, Any] | None,
    *,
    usage: dict[str, Any] | None = None,
    lam: float | None = None,
    format_penalty: float | None = None,
) -> float:
    """Efficiency-shaped train reward with in-think offload format gate.

    - unsolved: ``0``
    - solved: ``1 - λ * cost_ratio``, then subtract think-format penalty once if
      any offload span appeared outside ``<think>`` (clamped at 0).
    """
    if float(solved) <= 0.0:
        return 0.0
    if not stats:
        return float(solved)
    ratio = cost_ratio(stats, usage)
    reward = float(solved) - float(lam if lam is not None else efficiency_lambda()) * ratio
    outside = int(stats.get("offload_outside_think_count", 0) or 0)
    if outside > 0:
        pen = float(format_penalty if format_penalty is not None else think_format_penalty())
        reward -= pen
    return max(0.0, reward)


def help_seeking_reward(
    solved: float,
    stats: dict[str, Any] | None,
    *,
    usage: dict[str, Any] | None = None,
    lam: float | None = None,
    format_penalty: float | None = None,
    alpha: float | None = None,
    empty_patch: bool = False,
    empty_scale: float | None = None,
    unique_solver: bool = False,
    unique_bonus: float | None = None,
    encourage_seek: bool = True,
    protocol: str | None = None,
) -> float:
    """Train reward that keeps a gradient toward offloading when stuck.

    Compared to :func:`cost_aware_reward` (which gives ``0`` on all failures),
    this credits legitimate in-think help-seeking even when grading fails.

    - unsolved, ``encourage_seek=False``: ``0`` (defer α to group shaping)
    - unsolved, no in-think offload: ``0``
    - unsolved, ``offload_count>0`` and no outside-think spans: ``α``
      (scaled by ``empty_scale`` when ``empty_patch``, **except tmax**)
    - unsolved with only outside-think offload: ``0`` (format still discouraged)
    - solved: ``1 - λ * cost_ratio`` (− format penalty if outside-think), then
      optional ``unique_bonus`` when this traj is the sole solver in its group

    Prefer :func:`compute_turn_rewards` for per-turn shaping used in training.
    """
    st = stats or {}
    oc = int(st.get("offload_count", 0) or 0)
    outside = int(st.get("offload_outside_think_count", 0) or 0)
    alpha_v = float(alpha if alpha is not None else seek_alpha())
    emp_scale = float(empty_scale if empty_scale is not None else seek_empty_scale())

    if float(solved) <= 0.0:
        if not encourage_seek or oc < 1 or outside > 0:
            return 0.0
        credit = _apply_empty_scale(
            alpha_v, empty_patch=empty_patch, empty_scale=emp_scale, protocol=protocol
        )
        return max(0.0, credit)

    reward = cost_aware_reward(
        solved,
        st,
        usage=usage,
        lam=lam,
        format_penalty=format_penalty,
    )
    if unique_solver:
        bonus = float(unique_bonus if unique_bonus is not None else unique_solver_bonus())
        reward += bonus
    return max(0.0, reward)


def compute_turn_rewards(
    solved: float,
    stats: dict[str, Any] | None,
    *,
    usage: dict[str, Any] | None = None,
    completion_tokens: int | float | None = None,
    metadata: dict[str, Any] | None = None,
    lam: float | None = None,
    format_penalty: float | None = None,
    malformed_pen: float | None = None,
    alpha: float | None = None,
    empty_patch: bool = False,
    empty_scale: float | None = None,
    encourage_seek: bool = True,
    protocol: str | None = None,
) -> dict[str, Any]:
    """Per-turn rewards ``r_i`` and scalar ``mean(r_i)``.

    Solved: ``r_i = max(0, 1 - λ * c_i/b_i - format - malformed)``.
    Unsolved help_seeking: α' on valid in-think turns (unless deferred),
    ``-β`` on malformed/orphan spam turns, else 0.
    """
    st = dict(stats or {})
    turns: list[dict[str, Any]] = list(st.get("turn_costs") or [])
    n = len(turns)
    lam_v = float(lam if lam is not None else efficiency_lambda())
    fmt_pen = float(format_penalty if format_penalty is not None else think_format_penalty())
    mal_pen = float(malformed_pen if malformed_pen is not None else malformed_penalty())
    alpha_v = float(alpha if alpha is not None else seek_alpha())
    emp_scale = float(empty_scale if empty_scale is not None else seek_empty_scale())
    proto = protocol if protocol is not None else (metadata or {}).get("protocol")

    if n == 0:
        # Fallback to legacy scalar when no turn ledger (compat).
        if reward_mode() == "help_seeking":
            r = help_seeking_reward(
                solved,
                st,
                usage=usage,
                lam=lam_v,
                format_penalty=fmt_pen,
                alpha=alpha_v,
                empty_patch=empty_patch,
                empty_scale=emp_scale,
                encourage_seek=encourage_seek,
                protocol=proto,
            )
        else:
            r = cost_aware_reward(solved, st, usage=usage, lam=lam_v, format_penalty=fmt_pen)
        return {"reward": float(r), "turn_rewards": [], "turn_costs": turns}

    b_i = per_turn_baseline_cost(
        n_turns=n, usage=usage, completion_tokens=completion_tokens, metadata=metadata
    )
    turn_rewards: list[float] = []

    if float(solved) > 0.0:
        for tc in turns:
            c_i = turn_actual_cost(tc)
            ratio = (c_i / b_i) if b_i > 0 else 0.0
            r_i = 1.0 - lam_v * ratio
            if tc.get("outside_think"):
                r_i -= fmt_pen
            mal = int(tc.get("malformed_count", 0) or 0) + int(tc.get("orphan_open_count", 0) or 0)
            if mal > 0:
                r_i -= mal_pen
            turn_rewards.append(max(0.0, float(r_i)))
    elif reward_mode() != "help_seeking":
        # cost_aware: unsolved → all zeros (still record malformed as -β for advantage).
        for tc in turns:
            mal = int(tc.get("malformed_count", 0) or 0) + int(tc.get("orphan_open_count", 0) or 0)
            if mal > 0 or int(tc.get("max_open_run", 0) or 0) >= 8:
                turn_rewards.append(-mal_pen)
            else:
                turn_rewards.append(0.0)
    else:
        credit = _apply_empty_scale(
            alpha_v, empty_patch=empty_patch, empty_scale=emp_scale, protocol=proto, metadata=metadata
        )
        for tc in turns:
            mal = int(tc.get("malformed_count", 0) or 0) + int(tc.get("orphan_open_count", 0) or 0)
            if mal > 0 or int(tc.get("max_open_run", 0) or 0) >= 8:
                turn_rewards.append(-mal_pen)
                continue
            if (
                encourage_seek
                and tc.get("valid_offload")
                and not tc.get("outside_think")
            ):
                turn_rewards.append(max(0.0, float(credit)))
            else:
                turn_rewards.append(0.0)

    reward = float(sum(turn_rewards) / max(len(turn_rewards), 1))
    return {"reward": reward, "turn_rewards": turn_rewards, "turn_costs": turns}


def build_turn_token_spans(
    response_length: int,
    loss_mask: list[int] | None,
    turn_costs: list[dict[str, Any]],
    turn_rewards: list[float],
) -> list[list[int]] | None:
    """Map turns onto response indices via trainable-token proportions.

    Returns ``[[start, end), ...]`` in response coordinates, or None to signal
    broadcast fallback.
    """
    if not turn_costs or not turn_rewards or response_length <= 0:
        return None
    if len(turn_costs) != len(turn_rewards):
        return None
    mask = list(loss_mask) if loss_mask is not None else [1] * response_length
    if len(mask) != response_length:
        return None
    trainable = [i for i, m in enumerate(mask) if int(m)]
    if not trainable:
        return None
    weights = [max(1, int(tc.get("small_output_tokens", 0) or 0)) for tc in turn_costs]
    total_w = sum(weights)
    spans: list[list[int]] = []
    cursor = 0
    n_train = len(trainable)
    for i, w in enumerate(weights):
        remaining_turns = len(weights) - i
        if remaining_turns == 1:
            chunk = trainable[cursor:]
        else:
            n_tok = max(1, int(round(n_train * w / total_w)))
            end = min(cursor + n_tok, n_train - (remaining_turns - 1))
            end = max(cursor + 1, end)
            chunk = trainable[cursor:end]
            cursor = end
        if not chunk:
            spans.append([0, 0])
        else:
            spans.append([int(chunk[0]), int(chunk[-1]) + 1])
    return spans


def attach_turn_advantage_metadata(
    samples: list[Any],
    *,
    turn_rewards: list[float],
    turn_costs: list[dict[str, Any]],
) -> None:
    """Write turn_rewards / spans into sample.metadata and train_metadata."""
    for sample in samples:
        md = dict(getattr(sample, "metadata", None) or {})
        md["turn_rewards"] = list(turn_rewards)
        md["turn_costs"] = list(turn_costs)
        spans = build_turn_token_spans(
            int(getattr(sample, "response_length", 0) or 0),
            getattr(sample, "loss_mask", None),
            turn_costs,
            turn_rewards,
        )
        if spans is not None:
            md["turn_token_spans"] = spans
        else:
            md.pop("turn_token_spans", None)
        sample.metadata = md
        train_md = dict(getattr(sample, "train_metadata", None) or {})
        train_md["turn_rewards"] = list(turn_rewards)
        if spans is not None:
            train_md["turn_token_spans"] = spans
        else:
            train_md.pop("turn_token_spans", None)
        sample.train_metadata = train_md


def _aggregate_tag_counts(stats: dict[str, Any]) -> dict[str, int]:
    turns = list(stats.get("turn_costs") or [])
    if turns:
        return {
            "open_count": sum(int(t.get("open_count", 0) or 0) for t in turns),
            "close_count": sum(int(t.get("close_count", 0) or 0) for t in turns),
            "orphan_open_count": sum(int(t.get("orphan_open_count", 0) or 0) for t in turns),
            "malformed_count": sum(int(t.get("malformed_count", 0) or 0) for t in turns),
            "max_open_run": max((int(t.get("max_open_run", 0) or 0) for t in turns), default=0),
            "special_mark_count": sum(int(t.get("special_mark_count", 0) or 0) for t in turns),
            "small_output_tokens": sum(int(t.get("small_output_tokens", 0) or 0) for t in turns),
        }
    return {
        "open_count": 0,
        "close_count": 0,
        "orphan_open_count": 0,
        "malformed_count": 0,
        "max_open_run": 0,
        "special_mark_count": 0,
        "small_output_tokens": int(stats.get("small_output_tokens", 0) or 0),
    }


def compact_should_remove_sample(sample: Any) -> tuple[bool, str | None]:
    """Return (remove, reason) for hard compact/overlong filters."""
    from slime.utils.types import Sample as _Sample

    md = getattr(sample, "metadata", None) or {}
    status = getattr(sample, "status", None)
    abort_reason = str(md.get("abort_reason") or "")

    if status == getattr(_Sample.Status, "TRUNCATED", None) or str(status).endswith("TRUNCATED"):
        return True, "truncated"
    if md.get("timeout") or "timeout" in abort_reason.lower():
        return True, "timeout"
    if status == getattr(_Sample.Status, "ABORTED", None) and "timeout" in abort_reason.lower():
        return True, "timeout"
    if md.get("max_steps_reached") or stats_flag_max_steps(md) or abort_reason == "max_turns":
        return True, "max_steps"

    stats = md.get("offload_stats") or {}
    tags = _aggregate_tag_counts(stats)
    k = int(os.environ.get("OFFLOAD_COMPACT_ORPHAN_OPEN_K", str(DEFAULT_COMPACT_ORPHAN_OPEN_K)))
    ratio_r = float(os.environ.get("OFFLOAD_COMPACT_OPEN_CLOSE_RATIO", str(DEFAULT_COMPACT_OPEN_CLOSE_RATIO)))
    special_ratio = float(
        os.environ.get("OFFLOAD_COMPACT_SPECIAL_TOKEN_RATIO", str(DEFAULT_COMPACT_SPECIAL_TOKEN_RATIO))
    )
    special_run = int(os.environ.get("OFFLOAD_COMPACT_SPECIAL_TOKEN_RUN", str(DEFAULT_COMPACT_SPECIAL_TOKEN_RUN)))

    open_c = tags["open_count"]
    close_c = tags["close_count"]
    orphan = tags["orphan_open_count"]
    if orphan >= k or (open_c >= k and close_c == 0):
        return True, "orphan_open"
    if open_c / max(close_c, 1) >= ratio_r and open_c >= k:
        return True, "open_close_ratio"
    if tags["max_open_run"] >= special_run:
        return True, "special_token_run"
    out_tok = max(1, tags["small_output_tokens"], int(getattr(sample, "response_length", 0) or 0))
    if tags["special_mark_count"] / out_tok >= special_ratio and tags["special_mark_count"] >= k:
        return True, "special_token_ratio"
    return False, None


def stats_flag_max_steps(md: dict[str, Any]) -> bool:
    stats = md.get("offload_stats") or {}
    return bool(stats.get("max_steps_reached"))


def compact_filter_offload_samples(args: Any, groups: list) -> None:
    """Hard-remove truncated / timeout / max-steps / tag-spam trajectories."""
    del args
    for group in groups:
        for item in group:
            for sample in _session_segments(item):
                remove, reason = compact_should_remove_sample(sample)
                if remove:
                    sample.remove_sample = True
                    md = dict(getattr(sample, "metadata", None) or {})
                    md["compact_remove_reason"] = reason
                    sample.metadata = md


def _session_solved(metadata: dict[str, Any] | None) -> bool:
    md = metadata or {}
    if md.get("grading_solved") is True:
        return True
    return float(md.get("solved", 0) or 0) > 0.0


def _session_segments(group_item: Any) -> list[Any]:
    """One GRPO sibling may be a Sample or a fan-out ``list[Sample]``."""
    if isinstance(group_item, list):
        return [s for s in group_item if s is not None]
    return [group_item] if group_item is not None else []


def shape_group_help_seeking_rewards(args: Any, groups: list) -> None:
    """Grant help-seeking α on valid offload turns when the whole group failed.

    Mutates ``turn_rewards`` / ``sample.reward`` in place. tmax does not apply
    ``empty_scale`` on empty_patch.
    """
    del args
    if reward_mode() != "help_seeking" or not seek_only_when_all_wrong():
        return

    alpha_v = seek_alpha()
    emp_scale = seek_empty_scale()
    mal_pen = malformed_penalty()

    for group in groups:
        sessions = [_session_segments(item) for item in group]
        sessions = [segs for segs in sessions if segs]
        if not sessions:
            continue
        if any(_session_solved(getattr(segs[0], "metadata", None)) for segs in sessions):
            continue

        for segs in sessions:
            md = dict(getattr(segs[0], "metadata", None) or {})
            stats = md.get("offload_stats") or {}
            turn_rewards = list(md.get("turn_rewards") or [])
            turn_costs = list(md.get("turn_costs") or stats.get("turn_costs") or [])
            credit = _apply_empty_scale(
                alpha_v,
                empty_patch=bool(md.get("empty_patch")),
                empty_scale=emp_scale,
                protocol=md.get("protocol"),
                metadata=md,
            )
            credit = max(0.0, float(credit))

            if turn_costs and (not turn_rewards or len(turn_rewards) != len(turn_costs)):
                turn_rewards = [0.0] * len(turn_costs)

            if turn_costs and turn_rewards:
                for i, tc in enumerate(turn_costs):
                    mal = int(tc.get("malformed_count", 0) or 0) + int(tc.get("orphan_open_count", 0) or 0)
                    if mal > 0:
                        if turn_rewards[i] > -mal_pen:
                            turn_rewards[i] = -mal_pen
                        continue
                    if tc.get("valid_offload") and not tc.get("outside_think"):
                        if float(turn_rewards[i]) <= 0.0:
                            turn_rewards[i] = credit
                if turn_rewards:
                    mean_r = float(sum(turn_rewards) / max(len(turn_rewards), 1))
                    for sample in segs:
                        smd = dict(getattr(sample, "metadata", None) or {})
                        smd["turn_rewards"] = list(turn_rewards)
                        smd["turn_costs"] = list(turn_costs)
                        sample.metadata = smd
                        tmd = dict(getattr(sample, "train_metadata", None) or {})
                        tmd["turn_rewards"] = list(turn_rewards)
                        if "turn_token_spans" in smd:
                            tmd["turn_token_spans"] = smd["turn_token_spans"]
                        sample.train_metadata = tmd
                        if float(getattr(sample, "reward", 0.0) or 0.0) <= 0.0:
                            sample.reward = mean_r
                continue

            # Legacy scalar path (no turn ledger).
            oc = int(stats.get("offload_count", 0) or 0)
            outside = int(stats.get("offload_outside_think_count", 0) or 0)
            if oc < 1 or outside > 0:
                continue
            for sample in segs:
                if float(getattr(sample, "reward", 0.0) or 0.0) <= 0.0:
                    sample.reward = credit


def compact_and_shape_group_help_seeking_rewards(args: Any, groups: list) -> None:
    """Compact hard-filter then help-seeking group α shaping."""
    compact_filter_offload_samples(args, groups)
    shape_group_help_seeking_rewards(args, groups)
