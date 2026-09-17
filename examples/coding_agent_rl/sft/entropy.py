"""Per-token top-k renormalized entropy, then mean over a generation turn.

Vendored from offload_entropybased/adapter/entropy.py for slime SFT tooling.

For each generated token i with top-k logprobs {ℓ_j}:

    p_j = exp(ℓ_j) / Σ_k exp(ℓ_k)
    H_i = -Σ_j p_j * ln(p_j)     # nats, range [0, ln(k)]

avg_H = mean_i H_i over tokens that have at least one top logprob.
If a position returns fewer than the requested k, renormalize over the
returned set.

scope="thinking" (default for routing): average only over reasoning /
reasoning_content tokens (the model thinking block), ignoring answer content.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_THINKING_SOURCES = frozenset({"reasoning_content", "reasoning"})
THINKING_SOURCES = _THINKING_SOURCES


def align_logprob_source(entry_source: str, delta_source: str) -> str:
    """Map streamed logprob bucket to the real token phase.

    vLLM often emits thinking as ``delta.reasoning`` / ``delta.reasoning_content``
    while still putting per-token logprobs under ``logprobs.content``. In that
    case prefer the delta phase so thinking entropy is not treated as answer
    content.
    """
    entry = str(entry_source or "content")
    delta = str(delta_source or "content")
    if entry == "content" and delta in _THINKING_SOURCES:
        return delta
    return entry


def _as_logprob(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        lp = float(value)
        if math.isfinite(lp):
            return lp
        return None
    if isinstance(value, dict):
        if "logprob" in value:
            return _as_logprob(value.get("logprob"))
        if "logprobs" in value:
            return _as_logprob(value.get("logprobs"))
    return None


def top_logprob_values(entry: Any) -> list[float]:
    """Extract the top-k logprobs for one generated-token position."""
    if not isinstance(entry, dict):
        lp = _as_logprob(entry)
        return [lp] if lp is not None else []

    top = entry.get("top_logprobs")
    out: list[float] = []
    if isinstance(top, list):
        for item in top:
            lp = _as_logprob(item)
            if lp is not None:
                out.append(lp)
    elif isinstance(top, dict):
        for value in top.values():
            lp = _as_logprob(value)
            if lp is not None:
                out.append(lp)

    if out:
        return out
    chosen = _as_logprob(entry.get("logprob"))
    return [chosen] if chosen is not None else []


def token_entropy_nats(logprobs: list[float]) -> float | None:
    """Entropy of a distribution after softmax over the given logprobs."""
    if not logprobs:
        return None
    finite = [lp for lp in logprobs if math.isfinite(lp)]
    if not finite:
        return None
    maximum = max(finite)
    weights = [math.exp(lp - maximum) for lp in finite]
    total = sum(weights)
    if total <= 0.0:
        return None
    entropy = 0.0
    for weight in weights:
        p = weight / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return entropy


def _token_text(entry: Any) -> str:
    if isinstance(entry, dict) and entry.get("token") is not None:
        return str(entry.get("token"))
    return ""


def iter_token_entries_with_source(logprobs: Any) -> list[tuple[str, Any]]:
    """(source, entry) pairs. source is content / reasoning_content / reasoning."""
    if logprobs is None:
        return []
    if isinstance(logprobs, list):
        return [("content", item) for item in logprobs if item is not None]
    if not isinstance(logprobs, dict):
        return []
    out: list[tuple[str, Any]] = []
    for key in ("reasoning_content", "reasoning", "content"):
        items = logprobs.get(key)
        if isinstance(items, list):
            out.extend((key, item) for item in items if item is not None)
    if out:
        return out
    top = logprobs.get("top_logprobs")
    tokens = logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")
    if isinstance(top, list) and top and isinstance(top[0], dict):
        if tokens is None or (isinstance(tokens, list) and len(tokens) == len(top)):
            synthesized: list[tuple[str, Any]] = []
            for i, item in enumerate(top):
                entry: dict[str, Any] = {"top_logprobs": item}
                if isinstance(tokens, list) and i < len(tokens):
                    entry["token"] = tokens[i]
                if isinstance(token_logprobs, list) and i < len(token_logprobs):
                    entry["logprob"] = token_logprobs[i]
                synthesized.append(("content", entry))
            return synthesized
    return out


def iter_token_entries(logprobs: Any) -> list[Any]:
    return [entry for _, entry in iter_token_entries_with_source(logprobs)]


@dataclass
class TokenEntropy:
    token: str
    entropy: float | None
    source: str = "content"

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "entropy": None if self.entropy is None else round(float(self.entropy), 6),
            "source": self.source,
        }


@dataclass
class EntropyResult:
    avg_entropy: float | None
    n_tokens: int = 0
    n_used: int = 0
    per_token: list[TokenEntropy] = field(default_factory=list)
    unavailable: bool = True
    scope: str = "thinking"

    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_entropy": None if self.avg_entropy is None else round(float(self.avg_entropy), 6),
            "n_tokens": self.n_tokens,
            "n_used": self.n_used,
            "unavailable": self.unavailable,
            "scope": self.scope,
            "tokens": [t.to_dict() for t in self.per_token],
        }


def _normalize_scope(scope: str) -> str:
    text = str(scope or "thinking").strip().lower()
    if text in ("all", "full", "turn"):
        return "all"
    if text in ("thinking", "reason", "reasoning", "think"):
        return "thinking"
    raise ValueError(f"unsupported entropy scope: {scope!r}")


def _source_in_scope(source: str, scope: str) -> bool:
    if scope == "all":
        return True
    return source in _THINKING_SOURCES


def average_turn_entropy(logprobs: Any, *, scope: str = "thinking") -> EntropyResult:
    """Mean top-k-renormalized token entropy over one SLM generation.

    scope:
      thinking — only reasoning_content / reasoning tokens (default)
      all — every returned logprob token (content + thinking)
    """
    scope_norm = _normalize_scope(scope)
    pairs = iter_token_entries_with_source(logprobs)
    per_token: list[TokenEntropy] = []
    used: list[float] = []
    for source, entry in pairs:
        entropy = token_entropy_nats(top_logprob_values(entry))
        per_token.append(TokenEntropy(token=_token_text(entry), entropy=entropy, source=source))
        if entropy is not None and _source_in_scope(source, scope_norm):
            used.append(entropy)
    n_in_scope = sum(1 for source, _ in pairs if _source_in_scope(source, scope_norm))
    n_tokens = len(per_token) if scope_norm == "all" else n_in_scope
    if not used:
        return EntropyResult(
            avg_entropy=None,
            n_tokens=n_tokens,
            n_used=0,
            per_token=per_token,
            unavailable=True,
            scope=scope_norm,
        )
    avg = sum(used) / len(used)
    return EntropyResult(
        avg_entropy=avg,
        n_tokens=n_tokens,
        n_used=len(used),
        per_token=per_token,
        unavailable=False,
        scope=scope_norm,
    )


def entropy_from_entry(entry: Any, *, source: str = "content") -> TokenEntropy:
    return TokenEntropy(
        token=_token_text(entry),
        entropy=token_entropy_nats(top_logprob_values(entry)),
        source=source,
    )


def running_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def extract_choice_logprobs(response: Any) -> Any:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    return choice.get("logprobs")
