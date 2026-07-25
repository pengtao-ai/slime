#!/usr/bin/env python3
"""Force-trigger mid-turn offload and call the real GLM endpoint.

Contract under test:
  SLM <- agent system + OFFLOAD_SYSTEM_PROMPT_APPEND
  GLM <- agent system + CODING_HANDOFF_PROMPT  (append stripped)

Prints:
  1) system prompt sent to GLM
  2) full messages payload
  3) GLM content / reasoning / usage
  4) composed SLM+GLM assistant reply the agent would see

Does not need GPU / Docker / Ray. Requires DASHSCOPE_* env (or launcher defaults).

  export DASHSCOPE_API_KEY=...
  export DASHSCOPE_BASE_URL=http://host:8000/v1
  export DASHSCOPE_MODEL=glm-5.2-fp8
  python examples/coding_agent_rl/smoke_offload_real_glm.py
  # optional: OFFLOAD_N=3 OFFLOAD_MAX_TOKENS=512
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

os.environ.setdefault("SLIME_AGENT_OFFLOAD", "1")
os.environ.setdefault("DASHSCOPE_MODEL", "glm-5.2-fp8")
os.environ.setdefault("OFFLOAD_MAX_TOKENS", "512")

from examples.coding_agent_rl import offload  # noqa: E402
from slime.agent.adapters.common import Reply  # noqa: E402


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


async def main() -> None:
    if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("ERROR: set DASHSCOPE_API_KEY (or OPENAI_API_KEY)", file=sys.stderr)
        sys.exit(1)
    if not (os.environ.get("DASHSCOPE_BASE_URL") or "").strip():
        print("ERROR: set DASHSCOPE_BASE_URL to an OpenAI-compatible GLM .../v1 endpoint", file=sys.stderr)
        sys.exit(1)

    n = int(os.environ.get("OFFLOAD_N", "3"))
    enable_thinking, effort = offload.reasoning_from_n(n)

    # Simulated Claude-Code-like Anthropic messages (translated form).
    agent_system = (
        "You are Claude Code, an agentic coding assistant.\n"
        "Working directory: /workspace/preliz\n"
        f"{offload.OFFLOAD_SYSTEM_PROMPT_APPEND}"
    )
    translated = [
        {"role": "system", "content": agent_system},
        {
            "role": "user",
            "content": (
                "Bug: `from_preliz` fails when `dist` is a truncated distribution.\n"
                "Reproduce and fix with a minimal patch."
            ),
        },
        {
            "role": "assistant",
            "content": "I'll inspect the package layout and locate `from_preliz`.",
            "reasoning_content": "Need to find the failing helper first.",
        },
        {
            "role": "user",
            "content": (
                "[tool:Bash] ls preliz && rg -n \"def from_preliz\" -g '*.py'\n"
                "preliz/\npreliz/distributions.py:120:def from_preliz(dist):"
            ),
        },
    ]

    slm_prefix = (
        "I found `from_preliz` but the truncated-dist edge case is ambiguous; "
        "handing off for a careful fix."
    )
    raw_output = f"{slm_prefix} {offload.OFFLOAD_OPEN}{n}{offload.OFFLOAD_CLOSE}"
    # Fake SLM token ids — only used for max_tokens budget / stats.
    slm_output_ids = [101, 102, 103, 104, 105, 248077, n + 48, 248078]

    messages = offload.build_offload_messages(translated, raw_output)
    glm_budget = max(0, offload._max_tokens() - len(slm_output_ids))

    _banner("1) GLM system prompt")
    print(messages[0]["content"])
    assert offload.CODING_HANDOFF_PROMPT in messages[0]["content"]
    assert offload.OFFLOAD_SYSTEM_PROMPT_APPEND not in messages[0]["content"], (
        "OFFLOAD_SYSTEM_PROMPT_APPEND must be stripped before GLM"
    )
    assert "You are Claude Code" in messages[0]["content"]

    _banner("2) Full messages to GLM")
    print(json.dumps(messages, ensure_ascii=False, indent=2))

    _banner("3) Call params")
    print(
        json.dumps(
            {
                "base_url": os.environ["DASHSCOPE_BASE_URL"],
                "model": os.environ["DASHSCOPE_MODEL"],
                "N": n,
                "enable_thinking": enable_thinking,
                "reasoning_effort": effort,
                "max_tokens": glm_budget,
                "OFFLOAD_MAX_TOKENS": offload._max_tokens(),
                "slm_output_ids": len(slm_output_ids),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    content, think, usage = await offload.call_remote_chat(
        messages,
        max_tokens=glm_budget,
        enable_thinking=enable_thinking,
        reasoning_effort=effort,
    )

    _banner("4) GLM raw result")
    print(
        json.dumps(
            {
                "usage": usage,
                "content": content,
                "reasoning_content": think,
                "content_len": len(content or ""),
                "think_len": len(think or ""),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # Compose with the same GLM payload (no second remote call).
    reply = Reply(
        manager_message={
            "role": "assistant",
            "content": raw_output,
            "reasoning_content": "local partial plan",
        },
        finish_reason="stop",
        wire=(
            [
                {"type": "thinking", "thinking": "local partial plan"},
                {"type": "text", "text": raw_output},
            ],
            "end_turn",
        ),
    )
    glm_think = "" if not enable_thinking else think
    composed = offload.amend_reply_with_offload(
        reply, glm_content=content, glm_think=glm_think
    )
    stats = {
        "offload_count": 1,
        "last_offload_n": n,
        "last_reasoning_effort": effort,
        "small_prompt_tokens": 3,
        "small_output_tokens": len(slm_output_ids),
        "glm_input_tokens": int((usage or {}).get("prompt_tokens") or 0),
        "glm_output_tokens": int((usage or {}).get("completion_tokens") or 0),
    }

    _banner("5) Composed assistant reply (what agent receives)")
    print(
        json.dumps(
            {
                "offload_stats": stats,
                "manager_message": composed.manager_message,
                "wire": composed.wire,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    ok = (
        not str(content).startswith("[Error:")
        and offload.OFFLOAD_OPEN not in str(composed.manager_message.get("content") or "")
        and "part_think" not in str(composed.manager_message.get("content") or "")
    )
    _banner("6) Verdict")
    print(json.dumps({"ok": ok, "offload_count": 1}, indent=2))
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
