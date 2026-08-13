"""Per-sample coding-agent registry for coding_agent_rl.

Canonical names match ``BaseHarness.name``. Aliases: ``cc`` -> ``claude_code``,
``mini-swe-agent`` / ``mini_swe`` -> ``miniswe``.

Adapter protocol is ``anthropic`` for every agent except ``codex`` (``openai``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from slime.agent.harness import (
    ClaudeCodeHarness,
    CodexHarness,
    MiniSweHarness,
    OpenCodeHarness,
    PiHarness,
)
from slime.agent.harness.common import BaseHarness

AdapterProtocol = Literal["anthropic", "openai"]

_ALIASES: dict[str, str] = {
    "cc": "claude_code",
    "mini-swe-agent": "miniswe",
    "mini_swe": "miniswe",
}

CANONICAL_AGENTS: tuple[str, ...] = (
    "claude_code",
    "codex",
    "pi",
    "opencode",
    "miniswe",
)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    harness_cls: type[BaseHarness]
    adapter_protocol: AdapterProtocol


_SPECS: dict[str, AgentSpec] = {
    "claude_code": AgentSpec("claude_code", ClaudeCodeHarness, "anthropic"),
    "codex": AgentSpec("codex", CodexHarness, "openai"),
    "pi": AgentSpec("pi", PiHarness, "anthropic"),
    "opencode": AgentSpec("opencode", OpenCodeHarness, "anthropic"),
    "miniswe": AgentSpec("miniswe", MiniSweHarness, "anthropic"),
}


def default_agent_name() -> str:
    return os.environ.get("SWE_AGENT", "claude_code")


def normalize_agent_name(name: str | None) -> str:
    """Map aliases / env default to a canonical agent name (does not validate)."""
    raw = (name if name is not None else default_agent_name()).strip()
    if not raw:
        raw = default_agent_name()
    return _ALIASES.get(raw, raw)


def resolve_agent(name: str | None = None) -> AgentSpec:
    canonical = normalize_agent_name(name)
    spec = _SPECS.get(canonical)
    if spec is None:
        raise ValueError(
            f"unknown agent {name!r} (normalized={canonical!r}); "
            f"expected one of {sorted(_SPECS)} or aliases {sorted(_ALIASES)}"
        )
    return spec
