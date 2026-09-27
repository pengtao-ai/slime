"""Mid-turn LLM offload for coding-agent RL.

Per agent round (every Claude Code / Codex request):

1. Adapter always calls the local SLM first.
2. If the SLM emits ``<|llm_offload|>N<|/llm_offload|>`` *inside thinking*
   (before ``</think>``; Qwen often omits the opening ``<think>`` from output_ids),
   call remote LLM with thinking selected by ``N`` (0=off, 1-3=low, 4-6=high, 7-9=max)
   via ``chat_template_kwargs.thinking`` / ``reasoning_effort``.
   In-think **fallback** also calls GLM with ``N=3`` (no turn α) for a bad
   payload span or an orphan ``<|/llm_offload|>``; orphan OPEN alone does not.
   Complete digit spans after ``</think>`` do not call GLM (outside-think -β).
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
with ``output_loss_mask=0`` so dumps contain the full assistant turn without
training on the remote suffix (``SLIME_OFFLOAD_EMBED_IN_TRAJECTORY``). The
SLM prefix (and later local turns) keep ``loss_mask=1``.

Train shaping (when ``SLIME_AGENT_OFFLOAD=1``):
  - Solved episode: ``R = max(floor, 1 - λ * cost_ratio)`` (no turn-count /
    outside-think episode penalty).
  - Failed episode: ``R = 0`` always.
  - Turn α (default 1.0) only when the whole GRPO group failed: valid in-think
    digit-span offload turns get α; malformed / outside-think turns get -β
    (fallback GLM calls do not earn α); when both apply on one turn, rewards
    are summed (α − β or R − β).
  - Group shaping: :func:`compact_and_shape_group_help_seeking_rewards`
    (no compact drop, no unique-solver bonus).
  - GiGPO: ``examples.coding_agent_rl.gigpo_advantage.compute_gigpo_advantages``.

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

# Local import kept lazy-free: gigpo_advantage does not import offload.
from examples.coding_agent_rl.gigpo_advantage import stamp_turn_gigpo_key

logger = logging.getLogger(__name__)

OFFLOAD_OPEN = "<|llm_offload|>"
OFFLOAD_CLOSE = "<|/llm_offload|>"
# Back-compat alias used by docs / older call sites.
OFFLOAD_TAG = OFFLOAD_OPEN
_OFFLOAD_SPAN_RE = re.compile(re.escape(OFFLOAD_OPEN) + r"(\d)" + re.escape(OFFLOAD_CLOSE))

DEFAULT_DASHSCOPE_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "http://208.64.254.187:8001/v1")
DEFAULT_DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_MODEL", "deepseek-v4-flash-0731")
DEFAULT_OFFLOAD_MAX_TOKENS = int(os.environ.get("OFFLOAD_MAX_TOKENS", "8192"))

# USD / MTok relative weights (scale cancels in cost_ratio):
# Qwen3.5-4B 0.016/0.23 + DeepSeek-V4-Flash 0.44/1.32
# Defaults only — prefer cost_*() getters so Ray workers pick up runtime_env
# overrides after import (module-level float() would freeze driver env at import).
_DEFAULT_COST_SMALL_PROMPT = 0.016
_DEFAULT_COST_SMALL_OUTPUT = 0.23
_DEFAULT_COST_GLM_INPUT = 0.44
_DEFAULT_COST_GLM_OUTPUT = 1.32

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

# Per-turn penalty for malformed tags or outside-think offload.
DEFAULT_OFFLOAD_MALFORMED_PENALTY = 0.08
# Fallback GLM thinking level for in-think bad payload / orphan CLOSE (no α).
DEFAULT_FALLBACK_OFFLOAD_N = 3
# Constrained decode: Free ban CLOSE + stop@OPEN, then ebnf digit+CLOSE.
DEFAULT_OFFLOAD_OPEN_TOKEN_ID = 248077
DEFAULT_OFFLOAD_CLOSE_TOKEN_ID = 248078
DEFAULT_OFFLOAD_CLOSE_LOGIT_BIAS = -100.0
DEFAULT_OFFLOAD_EBNF_MAX_NEW_TOKENS = 4
# All-wrong group: valid in-think offload turn credit (episode R stays 0).
DEFAULT_OFFLOAD_SEEK_ALPHA = 1.0
# Soft seek budget: 0 disables.
DEFAULT_OFFLOAD_SEEK_BUDGET = 0
DEFAULT_OFFLOAD_SEEK_BUDGET_TURN_K = 0
DEFAULT_OFFLOAD_SEEK_BUDGET_DECAY = 0.5
DEFAULT_OFFLOAD_SEEK_OVERAGE_PENALTY = 0.05
DEFAULT_OFFLOAD_SOLVED_REWARD_FLOOR = 0.0


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


def constrained_decode_enabled() -> bool:
    """Free (ban CLOSE) + ebnf digit+CLOSE continue when offload is on.

    Default on when ``SLIME_AGENT_OFFLOAD=1``; set ``OFFLOAD_CONSTRAINED_DECODE=0``
    to disable.
    """
    if not offload_enabled():
        return False
    raw = (os.environ.get("OFFLOAD_CONSTRAINED_DECODE") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def offload_open_token_id() -> int:
    return int(os.environ.get("OFFLOAD_OPEN_TOKEN_ID", str(DEFAULT_OFFLOAD_OPEN_TOKEN_ID)))


def offload_close_token_id() -> int:
    return int(os.environ.get("OFFLOAD_CLOSE_TOKEN_ID", str(DEFAULT_OFFLOAD_CLOSE_TOKEN_ID)))


def offload_close_logit_bias() -> float:
    return float(os.environ.get("OFFLOAD_CLOSE_LOGIT_BIAS", str(DEFAULT_OFFLOAD_CLOSE_LOGIT_BIAS)))


def offload_ebnf_max_new_tokens() -> int:
    return int(os.environ.get("OFFLOAD_EBNF_MAX_NEW_TOKENS", str(DEFAULT_OFFLOAD_EBNF_MAX_NEW_TOKENS)))


def offload_digit_close_ebnf() -> str:
    """GBNF: single digit then the CLOSE special-token string."""
    return f'root ::= [0-9] "{OFFLOAD_CLOSE}"\n'


def apply_free_phase_constrained_sampling(sp: dict[str, Any]) -> dict[str, Any]:
    """Mutate a copy of sampling_params for the Free phase (ban CLOSE, stop on OPEN)."""
    out = dict(sp)
    open_id = offload_open_token_id()
    close_id = offload_close_token_id()
    stops = [int(x) for x in (out.get("stop_token_ids") or [])]
    stops = [t for t in stops if t != close_id]
    if open_id not in stops:
        stops.append(open_id)
    out["stop_token_ids"] = stops
    bias = dict(out.get("logit_bias") or {})
    bias[str(close_id)] = float(offload_close_logit_bias())
    out["logit_bias"] = bias
    out["no_stop_trim"] = True
    out["skip_special_tokens"] = False
    out["spaces_between_special_tokens"] = False
    return out


def apply_ebnf_phase_constrained_sampling(sp: dict[str, Any]) -> dict[str, Any]:
    """Mutate a copy for the after-OPEN ebnf continuation (digit then CLOSE)."""
    out = dict(sp)
    open_id = offload_open_token_id()
    close_id = offload_close_token_id()
    stops = [int(x) for x in (out.get("stop_token_ids") or [])]
    stops = [t for t in stops if t != open_id]
    if close_id not in stops:
        stops.append(close_id)
    out["stop_token_ids"] = stops
    out.pop("logit_bias", None)
    out["ebnf"] = offload_digit_close_ebnf()
    out["max_new_tokens"] = min(int(out.get("max_new_tokens") or 4), offload_ebnf_max_new_tokens())
    out["no_stop_trim"] = True
    out["skip_special_tokens"] = False
    out["spaces_between_special_tokens"] = False
    return out


def efficiency_lambda() -> float:
    return float(os.environ.get("OFFLOAD_EFFICIENCY_LAMBDA", "0.6"))


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


def solved_reward_floor() -> float:
    return float(os.environ.get("OFFLOAD_SOLVED_REWARD_FLOOR", str(DEFAULT_OFFLOAD_SOLVED_REWARD_FLOOR)))


def cost_small_prompt() -> float:
    return float(os.environ.get("OFFLOAD_COST_SMALL_PROMPT", str(_DEFAULT_COST_SMALL_PROMPT)))


def cost_small_output() -> float:
    return float(os.environ.get("OFFLOAD_COST_SMALL_OUTPUT", str(_DEFAULT_COST_SMALL_OUTPUT)))


def cost_glm_input() -> float:
    return float(os.environ.get("OFFLOAD_COST_GLM_INPUT", str(_DEFAULT_COST_GLM_INPUT)))


def cost_glm_output() -> float:
    return float(os.environ.get("OFFLOAD_COST_GLM_OUTPUT", str(_DEFAULT_COST_GLM_OUTPUT)))


# Back-compat for tests / notebooks that still read module attributes.
COST_SMALL_PROMPT = _DEFAULT_COST_SMALL_PROMPT
COST_SMALL_OUTPUT = _DEFAULT_COST_SMALL_OUTPUT
COST_GLM_INPUT = _DEFAULT_COST_GLM_INPUT
COST_GLM_OUTPUT = _DEFAULT_COST_GLM_OUTPUT


def seek_only_when_all_wrong() -> bool:
    """If true, defer turn-α to group shaping (only when every sibling failed)."""
    return os.environ.get("OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def seek_budget_fixed() -> int:
    """Fixed max in-think offloads before soft overage (0 = unused)."""
    try:
        return max(0, int(os.environ.get("OFFLOAD_SEEK_BUDGET", str(DEFAULT_OFFLOAD_SEEK_BUDGET))))
    except ValueError:
        return DEFAULT_OFFLOAD_SEEK_BUDGET


def seek_budget_turn_k() -> int:
    """If >0, budget includes ``max(1, n_turns // K)``."""
    try:
        return max(0, int(os.environ.get("OFFLOAD_SEEK_BUDGET_TURN_K", str(DEFAULT_OFFLOAD_SEEK_BUDGET_TURN_K))))
    except ValueError:
        return DEFAULT_OFFLOAD_SEEK_BUDGET_TURN_K


def seek_budget_decay() -> float:
    """Per over-budget seek: α *= decay**excess."""
    try:
        return float(os.environ.get("OFFLOAD_SEEK_BUDGET_DECAY", str(DEFAULT_OFFLOAD_SEEK_BUDGET_DECAY)))
    except ValueError:
        return DEFAULT_OFFLOAD_SEEK_BUDGET_DECAY


def seek_overage_penalty() -> float:
    """Solved: subtract ``penalty * excess`` per over-budget seek turn."""
    try:
        return float(
            os.environ.get("OFFLOAD_SEEK_OVERAGE_PENALTY", str(DEFAULT_OFFLOAD_SEEK_OVERAGE_PENALTY))
        )
    except ValueError:
        return DEFAULT_OFFLOAD_SEEK_OVERAGE_PENALTY


def resolve_seek_budget(n_turns: int) -> int | None:
    """Soft in-think offload budget, or ``None`` when unlimited.

    - ``OFFLOAD_SEEK_BUDGET_TURN_K=K>0`` → ``max(1, n_turns // K)``
    - ``OFFLOAD_SEEK_BUDGET=M>0`` → fixed ``M`` (or ``min`` with turn budget when both set)
    - both 0 → ``None`` (no soft limit)
    """
    fixed = seek_budget_fixed()
    k = seek_budget_turn_k()
    if fixed <= 0 and k <= 0:
        return None
    budget: int | None = None
    if k > 0:
        budget = max(1, int(n_turns) // k)
    if fixed > 0:
        budget = fixed if budget is None else min(budget, fixed)
    return budget


def seek_budget_alpha_scale(
    offload_ordinal: int,
    budget: int | None,
    *,
    decay: float | None = None,
) -> float:
    """1-based ordinal of valid in-think offloads → α multiplier (soft decay past budget)."""
    if budget is None or int(offload_ordinal) <= int(budget):
        return 1.0
    excess = int(offload_ordinal) - int(budget)
    d = float(decay if decay is not None else seek_budget_decay())
    if d <= 0.0:
        return 0.0
    return float(d**excess)


def seek_budget_overage_penalty_value(
    offload_ordinal: int,
    budget: int | None,
    *,
    pen: float | None = None,
) -> float:
    """Extra solved-path penalty for the ``offload_ordinal``-th seek (0 within budget)."""
    if budget is None or int(offload_ordinal) <= int(budget):
        return 0.0
    p = float(pen if pen is not None else seek_overage_penalty())
    return p * (int(offload_ordinal) - int(budget))


def _is_valid_in_think_offload(tc: dict[str, Any]) -> bool:
    return bool(tc.get("valid_offload")) and not bool(tc.get("outside_think"))


def _turn_offload_negative(tc: dict[str, Any]) -> bool:
    """True when this turn should receive -β (malformed tag or outside-think)."""
    return bool(tc.get("outside_think")) or turn_offload_tag_violation(tc)


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


def _position_inside_think(raw: str, pos: int) -> bool:
    """True iff ``pos`` is still in the thinking region.

    Qwen / PyroDash note: the opening ``<think>`` is usually injected by the chat
    template into the *prompt*, so ``output_ids`` often start with think text and
    only emit ``</think>``. Mid-turn offload that stops on the close token may
    have neither tag yet — that still counts as in-think.

    Rules:
      1. Inside an explicit ``<think>`` … (unclosed) region → in-think
      2. ``pos`` before the first ``</think>`` → in-think
      3. No ``</think>`` in ``raw`` → in-think (stopped during initial think)
      4. Else → outside think (visible / tool body after think ended)
    """
    before = raw[:pos]
    last_open = before.rfind("<think>")
    if last_open >= 0 and "</think>" not in before[last_open:]:
        return True

    first_close = raw.find("</think>")
    if first_close < 0:
        return True
    return pos < first_close


def offload_span_inside_think(raw: str) -> bool:
    """True iff the first complete offload span is still in the thinking region."""
    match = _OFFLOAD_SPAN_RE.search(raw)
    if match is None:
        return False
    return _position_inside_think(raw, match.start())


def parse_valid_offload_directive(raw: str) -> tuple[int, str] | None:
    """Like :func:`parse_offload_directive`, but only if the span is inside think."""
    parsed = parse_offload_directive(raw)
    if parsed is None or not offload_span_inside_think(raw):
        return None
    return parsed


def fallback_offload_n(raw: str) -> int | None:
    """Default N for in-think recoverable malformation, else None.

    Triggers (must be in-think):
      - bad payload: ``OPEN`` + non-single-digit + ``CLOSE``
      - orphan ``CLOSE`` (not paired with an OPEN…CLOSE span)

    Does **not** trigger on orphan OPEN alone. Complete digit spans are handled
    by :func:`parse_valid_offload_directive` / outside-think skip instead.
    """
    text = raw or ""
    consumed: list[tuple[int, int]] = []
    triggers: list[int] = []
    pos = 0
    while True:
        oi = text.find(OFFLOAD_OPEN, pos)
        if oi < 0:
            break
        ci = text.find(OFFLOAD_CLOSE, oi + len(OFFLOAD_OPEN))
        if ci < 0:
            # Orphan OPEN — never fallback-offload.
            pos = oi + len(OFFLOAD_OPEN)
            continue
        end = ci + len(OFFLOAD_CLOSE)
        consumed.append((oi, end))
        payload = text[oi + len(OFFLOAD_OPEN) : ci]
        if not (len(payload) == 1 and payload.isdigit()) and _position_inside_think(text, oi):
            triggers.append(oi)
        pos = end

    search = 0
    while True:
        ci = text.find(OFFLOAD_CLOSE, search)
        if ci < 0:
            break
        if any(s <= ci < e for s, e in consumed):
            search = ci + len(OFFLOAD_CLOSE)
            continue
        if _position_inside_think(text, ci):
            triggers.append(ci)
        search = ci + len(OFFLOAD_CLOSE)

    if not triggers:
        return None
    return int(DEFAULT_FALLBACK_OFFLOAD_N)


def decide_offload_directive(raw: str) -> tuple[int, bool] | None:
    """Return ``(N, earns_alpha)`` when GLM should run, else None.

    - Valid in-think ``OPEN+digit+CLOSE`` → ``(N, True)``
    - In-think bad payload / orphan CLOSE → ``(DEFAULT_FALLBACK_OFFLOAD_N, False)``
    - Complete digit span outside think, orphan OPEN, or no tag → ``None``
      (outside-think digit spans are detected separately via
      :func:`parse_offload_directive` + :func:`offload_span_inside_think`)
    """
    parsed = parse_valid_offload_directive(raw)
    if parsed is not None:
        return int(parsed[0]), True
    # Digit span exists but outside think → do not fallback-offload.
    if parse_offload_directive(raw) is not None:
        return None
    n = fallback_offload_n(raw)
    if n is None:
        return None
    return int(n), False


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
    Anything else (orphan OPEN, bad payload, stray CLOSE) counts as a violation.
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

    special_marks = open_count + close_count
    return {
        "open_count": open_count,
        "close_count": close_count,
        "valid_count": valid_count,
        "orphan_open_count": orphan_open,
        "malformed_count": malformed,
        "special_mark_count": special_marks,
    }


def turn_offload_tag_violation(tc: dict[str, Any]) -> bool:
    """True if this turn has any non-conforming ``<|llm_offload|>N<|/llm_offload|>`` usage.

    Rule: every offload mark must be part of a valid single-digit span. Orphan OPEN,
    bad payload, or stray CLOSE → violation (penalize with ``-β``).
    """
    return int(tc.get("malformed_count", 0) or 0) > 0 or int(tc.get("orphan_open_count", 0) or 0) > 0


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
    """Whether to append GLM tokens into ``turn.output_ids`` (mask=0, dump only)."""
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
    are mutable). SLM tokens keep loss_mask=1; the GLM suffix is loss_mask=0 so
    remote tokens are not trained, without clearing the SLM mask that precedes
    them. Later local turns still append with loss_mask=1 as usual.
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
    ``turn.output_ids`` with ``loss_mask=0`` (dump only; SLM prefix stays trainable).
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
    """Per agent round: account SLM tokens; if in-think offload, call GLM.

    Protocol:
      - Valid ``OPEN+N+CLOSE`` inside think → GLM with that N (earns turn α).
      - In-think bad payload or orphan CLOSE → GLM with
        :data:`DEFAULT_FALLBACK_OFFLOAD_N` (no α; still -β via malformed).
      - Orphan OPEN alone → no GLM.
      - Complete digit span outside think → no GLM; increments
        ``offload_outside_think_count``.

    On success, optionally appends tokenized GLM text to ``turn.output_ids`` with
    ``output_loss_mask=0`` (see ``SLIME_OFFLOAD_EMBED_IN_TRAJECTORY``).
    """
    if not offload_enabled():
        return reply

    turn_entry = record_local_turn_tokens(session, turn, raw_output=raw_output)
    decision = decide_offload_directive(raw_output)
    if decision is None:
        # Complete digit span exists but not in <think> → format violation, no GLM.
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
        stamp_turn_gigpo_key(turn_entry, reply)
        return reply

    n, earns_alpha = decision
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
            "fallback": not earns_alpha,
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
    # Only the well-formed digit span earns turn α; fallback keeps valid_offload=False.
    turn_entry["valid_offload"] = bool(earns_alpha)
    turn_entry["fallback_offload"] = not bool(earns_alpha)
    gin, gout = _record_glm_usage(stats, usage, messages=messages, content=content, think=think)
    turn_entry["glm_input_tokens"] = gin
    turn_entry["glm_output_tokens"] = gout

    logger.info(
        "[coding_agent_offload] sid=%s offload#%d N=%d thinking=%s effort=%s "
        "fallback=%s glm_budget=%d content_len=%d think_len=%d tool_calls=%d "
        "cum_slm=(%d,%d) cum_glm=(%d,%d)",
        sid,
        stats["offload_count"],
        n,
        enable_thinking,
        reasoning_effort,
        not earns_alpha,
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
    reply = amend_reply_with_offload(
        reply,
        raw_output=raw_output,
        glm_content=content,
        glm_think=think,
        glm_tool_calls=glm_tool_calls,
    )
    stamp_turn_gigpo_key(turn_entry, reply)
    return reply


def actual_cost(stats: dict[str, Any]) -> float:
    return (
        int(stats.get("small_prompt_tokens", 0)) * cost_small_prompt()
        + int(stats.get("small_output_tokens", 0)) * cost_small_output()
        + int(stats.get("glm_input_tokens", 0)) * cost_glm_input()
        + int(stats.get("glm_output_tokens", 0)) * cost_glm_output()
    )


def turn_actual_cost(turn: dict[str, Any]) -> float:
    return (
        int(turn.get("small_prompt_tokens", 0)) * cost_small_prompt()
        + int(turn.get("small_output_tokens", 0)) * cost_small_output()
        + int(turn.get("glm_input_tokens", 0)) * cost_glm_input()
        + int(turn.get("glm_output_tokens", 0)) * cost_glm_output()
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
    return prompt_t * cost_glm_input() + completion_t * cost_glm_output()


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
    return (prompt_t / n) * cost_glm_input() + (comp_t / n) * cost_glm_output()


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
    n_turns: int | None = None,
) -> float:
    """Efficiency-shaped episode reward.

    - unsolved: ``0``
    - solved: ``max(floor, 1 - λ * cost_ratio)``
    """
    del n_turns  # call-site compatibility
    if float(solved) <= 0.0:
        return 0.0
    if not stats:
        return max(solved_reward_floor(), float(solved))
    ratio = cost_ratio(stats, usage)
    reward = float(solved) - float(lam if lam is not None else efficiency_lambda()) * ratio
    return max(solved_reward_floor(), float(reward))


def help_seeking_reward(
    solved: float,
    stats: dict[str, Any] | None,
    *,
    usage: dict[str, Any] | None = None,
    lam: float | None = None,
    **_kwargs: Any,
) -> float:
    """Episode return only: solved cost-aware, failed always 0.

    Turn-level seek α is applied in :func:`compute_turn_rewards` /
    :func:`shape_group_help_seeking_rewards`, not here.
    """
    return cost_aware_reward(solved, stats, usage=usage, lam=lam)


def compute_turn_rewards(
    solved: float,
    stats: dict[str, Any] | None,
    *,
    usage: dict[str, Any] | None = None,
    completion_tokens: int | float | None = None,
    metadata: dict[str, Any] | None = None,
    lam: float | None = None,
    malformed_pen: float | None = None,
    alpha: float | None = None,
    encourage_seek: bool = True,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Per-turn ledger + episode return.

    Episode:
      - solved: ``max(floor, 1 - λ * cost_ratio)``
      - failed: ``0`` (seek credit is turn-only)

    Turns (components sum when both apply on one turn):
      - malformed / outside-think → ``-β``
      - failed + ``encourage_seek`` + valid in-think offload → ``α``
      - solved + (clean turn or valid in-think offload) → episode ``R``
      - else → ``0``
    """
    st = dict(stats or {})
    turns: list[dict[str, Any]] = list(st.get("turn_costs") or [])
    lam_v = float(lam if lam is not None else efficiency_lambda())
    mal_pen = float(malformed_pen if malformed_pen is not None else malformed_penalty())
    alpha_v = float(alpha if alpha is not None else seek_alpha())
    eff_usage = {
        "prompt_tokens": resolve_baseline_prompt_tokens(usage, metadata=metadata),
        "completion_tokens": resolve_baseline_completion_tokens(
            usage, completion_tokens=completion_tokens, metadata=metadata
        ),
    }

    if float(solved) > 0.0:
        r_base = cost_aware_reward(solved, st, usage=eff_usage, lam=lam_v)
    else:
        r_base = 0.0

    if not turns:
        return {"reward": float(r_base), "turn_rewards": [], "turn_costs": turns}

    turn_rewards: list[float] = []
    for tc in turns:
        r = 0.0
        if _turn_offload_negative(tc):
            r -= mal_pen
        if float(solved) > 0.0:
            if _is_valid_in_think_offload(tc) or not _turn_offload_negative(tc):
                r += float(r_base)
        elif reward_mode() == "help_seeking" and encourage_seek and _is_valid_in_think_offload(tc):
            r += alpha_v
        turn_rewards.append(r)

    return {
        "reward": float(r_base),
        "turn_rewards": turn_rewards,
        "turn_costs": turns,
    }

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
    """Write turn_rewards / spans / GiGPO T# keys into sample.metadata and train_metadata."""
    turn_T = [str(tc.get("gigpo_T") or "其他 · 无 tool") for tc in turn_costs]
    for sample in samples:
        md = dict(getattr(sample, "metadata", None) or {})
        md["turn_rewards"] = list(turn_rewards)
        md["turn_costs"] = list(turn_costs)
        md["turn_T"] = list(turn_T)
        if getattr(sample, "group_index", None) is not None:
            md["group_index"] = sample.group_index
        if getattr(sample, "index", None) is not None:
            md["sample_index"] = sample.index
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
        train_md["turn_T"] = list(turn_T)
        if getattr(sample, "group_index", None) is not None:
            train_md["group_index"] = sample.group_index
        if getattr(sample, "index", None) is not None:
            train_md["sample_index"] = sample.index
        if spans is not None:
            train_md["turn_token_spans"] = spans
        else:
            train_md.pop("turn_token_spans", None)
        sample.train_metadata = train_md


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
    """Paint turn-α only when every sibling failed; never raise episode R.

    Valid in-think offload turns get α; malformed / outside-think get -β;
    both on one turn sum to α − β. If any sibling solved, leave failed
    seekers with no turn α.
    """
    del args
    if reward_mode() != "help_seeking" or not seek_only_when_all_wrong():
        return

    alpha_v = seek_alpha()
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
            turn_costs = list(md.get("turn_costs") or stats.get("turn_costs") or [])
            turn_rewards = list(md.get("turn_rewards") or [])
            if not turn_costs:
                continue
            if not turn_rewards or len(turn_rewards) != len(turn_costs):
                turn_rewards = [0.0] * len(turn_costs)

            for i, tc in enumerate(turn_costs):
                r = 0.0
                if _turn_offload_negative(tc):
                    r -= mal_pen
                if _is_valid_in_think_offload(tc):
                    r += alpha_v
                turn_rewards[i] = r

            for sample in segs:
                smd = dict(getattr(sample, "metadata", None) or {})
                smd["turn_rewards"] = list(turn_rewards)
                smd["turn_costs"] = list(turn_costs)
                sample.metadata = smd
                tmd = dict(getattr(sample, "train_metadata", None) or {})
                tmd["turn_rewards"] = list(turn_rewards)
                if md.get("turn_T") is not None:
                    tmd["turn_T"] = list(md["turn_T"])
                if getattr(sample, "group_index", None) is not None:
                    tmd["group_index"] = sample.group_index
                if getattr(sample, "index", None) is not None:
                    tmd["sample_index"] = sample.index
                if "turn_token_spans" in smd:
                    tmd["turn_token_spans"] = smd["turn_token_spans"]
                sample.train_metadata = tmd
                if float(getattr(sample, "reward", 0.0) or 0.0) != 0.0:
                    # Failures must stay at episode 0 (generate may have left zeros).
                    if not _session_solved(smd):
                        sample.reward = 0.0


def compact_and_shape_group_help_seeking_rewards(args: Any, groups: list) -> None:
    """Group turn-α shaping for all-wrong help_seeking (no compact drop)."""
    shape_group_help_seeking_rewards(args, groups)
