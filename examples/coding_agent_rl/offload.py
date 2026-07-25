"""Mid-turn LLM offload for coding-agent RL.

Per agent round (every Claude Code / Codex request):

1. Adapter always calls the local SLM first.
2. If the SLM emits ``<|llm_offload|>N<|/llm_offload|>``, call remote GLM with
   thinking selected by ``N`` (0=off, 1-5=high, 6-9=max).
3. Compose SLM prefix + GLM continuation into one complete assistant reply and
   only then flush it to the agent.

System-prompt contract:
  - SLM / Claude Code: agent system + ``OFFLOAD_SYSTEM_PROMPT_APPEND``
    (via ``--append-system-prompt``)
  - GLM: agent system + ``CODING_HANDOFF_PROMPT``
    (``OFFLOAD_SYSTEM_PROMPT_APPEND`` is stripped if the agent echoed it)

Only local-model ``output_ids`` are trained; GLM tokens are never loss-masked in.

Enable with ``SLIME_AGENT_OFFLOAD=1`` (see ``generate.py`` Offload* adapters).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

import requests

from slime.agent.adapters.common import Reply, Session, flatten_content
from slime.agent.trajectory import TurnRecord

logger = logging.getLogger(__name__)

OFFLOAD_OPEN = "<|llm_offload|>"
OFFLOAD_CLOSE = "<|/llm_offload|>"
# Back-compat alias used by docs / older call sites.
OFFLOAD_TAG = OFFLOAD_OPEN
_OFFLOAD_SPAN_RE = re.compile(re.escape(OFFLOAD_OPEN) + r"(\d)" + re.escape(OFFLOAD_CLOSE))

DEFAULT_DASHSCOPE_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_MODEL", "glm-5.2-fp8")
DEFAULT_OFFLOAD_MAX_TOKENS = int(os.environ.get("OFFLOAD_MAX_TOKENS", "8192"))

COST_SMALL_PROMPT = float(os.environ.get("OFFLOAD_COST_SMALL_PROMPT", "0.017"))
COST_SMALL_OUTPUT = float(os.environ.get("OFFLOAD_COST_SMALL_OUTPUT", "0.026"))
COST_GLM_INPUT = float(os.environ.get("OFFLOAD_COST_GLM_INPUT", "0.315"))
COST_GLM_OUTPUT = float(os.environ.get("OFFLOAD_COST_GLM_OUTPUT", "1.0"))

# Fallback baseline when dataset metadata has no ``usage`` (GLM-only tokens).
_DEFAULT_BASELINE_PROMPT_TOKENS = int(os.environ.get("OFFLOAD_BASELINE_PROMPT_TOKENS", "2000"))
_DEFAULT_BASELINE_COMPLETION_TOKENS = int(os.environ.get("OFFLOAD_BASELINE_COMPLETION_TOKENS", "8000"))

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

# Appended to the black-box agent's *system* prompt (Claude Code:
# ``--append-system-prompt``), not the task user prompt (``SWE_PROMPT``).
OFFLOAD_SYSTEM_PROMPT_APPEND = (
    f"For very difficult steps, you can output {OFFLOAD_OPEN}N{OFFLOAD_CLOSE} to request help "
    "from a more capable model. N is a single digit: 0 = no remote thinking, "
    "1-5 = high reasoning effort, 6-9 = max reasoning effort."
)
# Back-compat alias.
DEFAULT_OFFLOAD_SWE_PROMPT = OFFLOAD_SYSTEM_PROMPT_APPEND


def ensure_offload_system_prompt_append() -> None:
    """Wire offload instructions into Claude Code's system prompt when enabled.

    Sets ``SLIME_AGENT_CC_APPEND_SYSTEM_PROMPT`` (consumed by
    ``ClaudeCodeHarness``) unless the caller already provided one. Does **not**
    modify ``SWE_PROMPT`` / the ``claude -p`` task text.
    """
    if not offload_enabled():
        return
    key = "SLIME_AGENT_CC_APPEND_SYSTEM_PROMPT"
    if not (os.environ.get(key) or "").strip():
        os.environ[key] = os.environ.get("SLIME_AGENT_OFFLOAD_SYSTEM_APPEND") or OFFLOAD_SYSTEM_PROMPT_APPEND


def offload_enabled() -> bool:
    return os.environ.get("SLIME_AGENT_OFFLOAD", "").strip().lower() in ("1", "true", "yes", "on")


def efficiency_lambda() -> float:
    return float(os.environ.get("OFFLOAD_EFFICIENCY_LAMBDA", "0.6"))


def _api_key() -> str:
    return (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()


def _base_url() -> str:
    return (os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_DASHSCOPE_BASE_URL).rstrip("/")


def _model() -> str:
    return os.environ.get("DASHSCOPE_MODEL") or DEFAULT_DASHSCOPE_MODEL


def _max_tokens() -> int:
    return int(os.environ.get("OFFLOAD_MAX_TOKENS", str(DEFAULT_OFFLOAD_MAX_TOKENS)))


def parse_offload_directive(raw: str) -> tuple[int, str] | None:
    """Return ``(N, text_before_span)`` for the first complete offload span, else None."""
    match = _OFFLOAD_SPAN_RE.search(raw)
    if match is None:
        return None
    return int(match.group(1)), raw[: match.start()]


def reasoning_from_n(n: int) -> tuple[bool, str | None]:
    """Map digit N -> ``(enable_thinking, reasoning_effort)``."""
    if n <= 0:
        return False, None
    if n <= 5:
        return True, "high"
    return True, "max"


def _ensure_stats(session: Session) -> dict[str, Any]:
    stats = session.offload_stats
    if not stats:
        stats.update(
            {
                "offload_count": 0,
                "small_prompt_tokens": 0,
                "small_output_tokens": 0,
                "glm_input_tokens": 0,
                "glm_output_tokens": 0,
                "last_offload_n": None,
                "last_reasoning_effort": None,
            }
        )
    return stats


def record_local_turn_tokens(session: Session, turn: TurnRecord) -> None:
    """Accumulate per-round SLM prompt/output token counts for session cost."""
    stats = _ensure_stats(session)
    stats["small_prompt_tokens"] = int(stats.get("small_prompt_tokens", 0)) + len(turn.prompt_ids or [])
    stats["small_output_tokens"] = int(stats.get("small_output_tokens", 0)) + len(turn.output_ids or [])


def _estimate_tokens(text: str) -> int:
    """Rough token estimate when the remote API omits ``usage``."""
    return max(0, (len(text) + 3) // 4)


def _record_glm_usage(
    stats: dict[str, Any],
    usage: dict[str, Any] | None,
    *,
    messages: list[dict[str, str]],
    content: str,
    think: str,
) -> None:
    if usage:
        inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    else:
        inp = _estimate_tokens("\n".join(m.get("content", "") for m in messages))
        out = _estimate_tokens(f"{think}{content}")
        logger.warning(
            "[coding_agent_offload] remote usage missing; estimated glm_in=%d glm_out=%d",
            inp,
            out,
        )
    stats["glm_input_tokens"] = int(stats.get("glm_input_tokens", 0)) + inp
    stats["glm_output_tokens"] = int(stats.get("glm_output_tokens", 0)) + out


def _offload_prefix(raw: str) -> str:
    parsed = parse_offload_directive(raw)
    if parsed is None:
        # Incomplete / legacy open-only tag: drop from first open marker.
        idx = raw.find(OFFLOAD_OPEN)
        prefix = raw[:idx] if idx >= 0 else raw
    else:
        prefix = parsed[1]
    return _strip_offload_tag_from_text(prefix).strip()


def _message_text(msg: dict[str, Any]) -> str:
    parts: list[str] = []
    reasoning = msg.get("reasoning_content")
    if reasoning:
        parts.append(f"<think>{reasoning}</think>")
    content = flatten_content(msg.get("content"))
    if content:
        parts.append(content)
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        parts.append(f"[tool_calls] {json.dumps(tool_calls, ensure_ascii=False)}")
    name = msg.get("name")
    if name and msg.get("role") == "tool":
        parts.insert(0, f"[tool:{name}]")
    return "\n".join(parts).strip()


def _offload_system_append_variants() -> list[str]:
    """Text that belongs on the *SLM/agent* system prompt only (not GLM)."""
    variants = [
        OFFLOAD_SYSTEM_PROMPT_APPEND,
        (os.environ.get("SLIME_AGENT_OFFLOAD_SYSTEM_APPEND") or "").strip(),
        (os.environ.get("SLIME_AGENT_CC_APPEND_SYSTEM_PROMPT") or "").strip(),
    ]
    return [v for v in variants if v]


def strip_offload_system_append(text: str) -> str:
    """Remove SLM-only offload instructions from agent system text before GLM.

    Contract:
      SLM  <- agent system + OFFLOAD_SYSTEM_PROMPT_APPEND
      GLM  <- agent system + CODING_HANDOFF_PROMPT  (no OFFLOAD_SYSTEM_PROMPT_APPEND)
    """
    out = text
    for variant in _offload_system_append_variants():
        if variant in out:
            out = out.replace(variant, "")
    # Collapse whitespace left by removals.
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def build_offload_messages(translated: list[dict], raw_output: str) -> list[dict[str, str]]:
    """Build GLM chat messages: agent system + handoff append, then history, then part_think.

    Strips ``OFFLOAD_SYSTEM_PROMPT_APPEND`` (SLM-only) from any agent system text
    that Claude Code / Codex echoed into the request.
    """
    agent_system_parts: list[str] = []
    rest: list[dict[str, str]] = []
    for msg in translated:
        role = str(msg.get("role") or "user")
        text = _message_text(msg)
        if not text:
            continue
        if role == "system":
            cleaned_system = strip_offload_system_append(text)
            if cleaned_system:
                agent_system_parts.append(cleaned_system)
            continue
        if role == "tool":
            rest.append({"role": "user", "content": text})
        elif role in ("user", "assistant"):
            rest.append({"role": role, "content": text})
        else:
            rest.append({"role": "user", "content": text})

    system = "\n\n".join([*agent_system_parts, CODING_HANDOFF_PROMPT]).strip()
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(rest)

    partial = _offload_prefix(raw_output)
    cleaned = partial.replace("<think>", "").replace("</think>", "").strip()
    if cleaned:
        messages.append({"role": "assistant", "content": f"<part_think>{cleaned}</part_think>"})
    return messages


def _call_remote_chat_sync(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    enable_thinking: bool,
    reasoning_effort: str | None,
    timeout: float = 600.0,
) -> tuple[str, str, dict[str, Any] | None]:
    api_key = _api_key()
    if not api_key:
        return "[Error: DASHSCOPE_API_KEY not set]", "", None
    if max_tokens <= 0:
        return "[Error: no remaining offload token budget]", "", None

    url = f"{_base_url()}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    chat_kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
    if enable_thinking and reasoning_effort:
        chat_kwargs["reasoning_effort"] = reasoning_effort
    body: dict[str, Any] = {
        "model": _model(),
        "messages": messages,
        "max_tokens": max_tokens,
        "chat_template_kwargs": chat_kwargs,
    }
    if enable_thinking and reasoning_effort:
        # Some OpenAI-compatible gateways read this at the top level.
        body["reasoning_effort"] = reasoning_effort
    try:
        response = requests.post(url, headers=headers, json=body, timeout=timeout)
        if response.status_code != 200:
            return f"[Error: status {response.status_code}: {response.text[:400]}]", "", None
        data = response.json()
        message = data["choices"][0].get("message", {})
        think = str(message.get("reasoning") or message.get("reasoning_content") or "")
        content = str(message.get("content") or "")
        usage = data.get("usage")
        if usage is not None and not isinstance(usage, dict):
            usage = None
        return content, think, usage
    except Exception as exc:
        return f"[Error: remote call failed: {exc}]", "", None


async def call_remote_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    enable_thinking: bool,
    reasoning_effort: str | None,
) -> tuple[str, str, dict[str, Any] | None]:
    return await asyncio.to_thread(
        _call_remote_chat_sync,
        messages,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )


def _strip_offload_tag_from_text(text: str) -> str:
    text = _OFFLOAD_SPAN_RE.sub("", text)
    # Drop dangling open/close markers if the span was truncated.
    return text.replace(OFFLOAD_OPEN, "").replace(OFFLOAD_CLOSE, "").rstrip()


def _join_nonempty(*parts: str, sep: str = "\n") -> str:
    return sep.join(p for p in parts if p)


def compose_complete_assistant(
    *,
    slm_text: str,
    slm_think: str,
    glm_content: str,
    glm_think: str,
) -> tuple[str, str]:
    """Merge SLM prefix + GLM continuation into one assistant (text, think) pair."""
    text = _join_nonempty(_strip_offload_tag_from_text(slm_text), glm_content)
    think = _join_nonempty(slm_think, glm_think, sep="")
    return text, think


def amend_reply_with_offload(reply: Reply, *, glm_content: str, glm_think: str) -> Reply:
    """Replace the SLM-only reply with the composed complete assistant turn for the agent.

    Training still recorded ``turn.output_ids`` from the SLM only; this amends what
    the coding agent receives / echoes on the next round.
    """
    mm = dict(reply.manager_message)
    text, think = compose_complete_assistant(
        slm_text=str(mm.get("content") or ""),
        slm_think=str(mm.get("reasoning_content") or ""),
        glm_content=glm_content,
        glm_think=glm_think,
    )
    mm["content"] = text
    if think:
        mm["reasoning_content"] = think
    else:
        mm.pop("reasoning_content", None)

    wire = reply.wire
    if isinstance(wire, tuple) and len(wire) == 2 and isinstance(wire[0], list):
        # Anthropic: rebuild as one thinking + one text (+ any tool_use from SLM).
        tool_blocks = [b for b in wire[0] if b.get("type") == "tool_use"]
        blocks: list[dict] = []
        if think:
            blocks.append({"type": "thinking", "thinking": think})
        if text or not tool_blocks:
            blocks.append({"type": "text", "text": text})
        blocks.extend(tool_blocks)
        stop_reason = "tool_use" if tool_blocks else "end_turn"
        wire = (blocks, stop_reason)
    elif isinstance(wire, tuple) and len(wire) == 2 and isinstance(wire[0], dict):
        # OpenAI: one message with merged content / reasoning.
        wm = dict(wire[0])
        if wm.get("tool_calls"):
            # Keep SLM tool calls; still attach merged reasoning when present.
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
        wire = (wm, finish)

    return Reply(manager_message=mm, finish_reason=reply.finish_reason, wire=wire)


async def apply_offload_if_needed(
    reply: Reply,
    *,
    raw_output: str,
    translated: list[dict],
    turn: TurnRecord,
    session: Session,
    sid: str,
) -> Reply:
    """Per agent round: account SLM tokens; if offload span, call GLM and compose full reply.

    No offload span -> return SLM reply unchanged (still the complete output for
    this round). With a span -> wait for GLM, then return SLM+GLM as one reply
    *before* the adapter responds to the agent.
    """
    if not offload_enabled():
        return reply

    record_local_turn_tokens(session, turn)
    parsed = parse_offload_directive(raw_output)
    if parsed is None:
        return reply

    n, _prefix = parsed
    enable_thinking, reasoning_effort = reasoning_from_n(n)

    stats = _ensure_stats(session)
    small_out = len(turn.output_ids or [])
    glm_budget = max(0, _max_tokens() - small_out)
    messages = build_offload_messages(translated, raw_output)
    content, think, usage = await call_remote_chat(
        messages,
        max_tokens=glm_budget,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )

    stats["offload_count"] = int(stats.get("offload_count", 0)) + 1
    stats["last_offload_n"] = n
    stats["last_reasoning_effort"] = reasoning_effort
    _record_glm_usage(stats, usage, messages=messages, content=content, think=think)

    logger.info(
        "[coding_agent_offload] sid=%s offload#%d N=%d thinking=%s effort=%s "
        "glm_budget=%d content_len=%d think_len=%d "
        "cum_slm=(%d,%d) cum_glm=(%d,%d)",
        sid,
        stats["offload_count"],
        n,
        enable_thinking,
        reasoning_effort,
        glm_budget,
        len(content),
        len(think),
        int(stats.get("small_prompt_tokens", 0)),
        int(stats.get("small_output_tokens", 0)),
        int(stats.get("glm_input_tokens", 0)),
        int(stats.get("glm_output_tokens", 0)),
    )
    # N=0: no remote think channel; drop any accidental reasoning payload.
    if not enable_thinking:
        think = ""
    return amend_reply_with_offload(reply, glm_content=content, glm_think=think)


def actual_cost(stats: dict[str, Any]) -> float:
    return (
        int(stats.get("small_prompt_tokens", 0)) * COST_SMALL_PROMPT
        + int(stats.get("small_output_tokens", 0)) * COST_SMALL_OUTPUT
        + int(stats.get("glm_input_tokens", 0)) * COST_GLM_INPUT
        + int(stats.get("glm_output_tokens", 0)) * COST_GLM_OUTPUT
    )


def baseline_cost(usage: dict[str, Any] | None) -> float:
    if usage:
        prompt_t = int(usage.get("prompt_tokens") or _DEFAULT_BASELINE_PROMPT_TOKENS)
        completion_t = int(usage.get("completion_tokens") or _DEFAULT_BASELINE_COMPLETION_TOKENS)
    else:
        prompt_t = _DEFAULT_BASELINE_PROMPT_TOKENS
        completion_t = _DEFAULT_BASELINE_COMPLETION_TOKENS
    return prompt_t * COST_GLM_INPUT + completion_t * COST_GLM_OUTPUT


def cost_ratio(stats: dict[str, Any], usage: dict[str, Any] | None = None) -> float:
    base = baseline_cost(usage)
    if base <= 0:
        return 0.0
    return actual_cost(stats) / base


def cost_aware_reward(
    solved: float,
    stats: dict[str, Any] | None,
    *,
    usage: dict[str, Any] | None = None,
    lam: float | None = None,
) -> float:
    """``reward = solved - λ * cost_ratio`` (no math format bonuses)."""
    if not stats:
        return float(solved)
    ratio = cost_ratio(stats, usage)
    return float(solved) - float(lam if lam is not None else efficiency_lambda()) * ratio
