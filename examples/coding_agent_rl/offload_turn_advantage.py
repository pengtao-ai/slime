"""Per-turn advantage painting for coding-agent offload RL.

Wire with::

    --custom-advantage-function-path \\
      examples.coding_agent_rl.offload_turn_advantage.compute_turn_advantages

``rollout_data["rewards"]`` is already GRPO group-demeaned (see
``RolloutManager._post_process_rewards``). This function broadcasts each sample's
``A_s`` then adds the within-trajectory turn residual
``r_i - mean(r)`` onto that turn's response span.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _paint_sample_advantage(
    base: torch.Tensor,
    *,
    a_s: float,
    turn_rewards: list[float],
    turn_token_spans: list[list[int]] | None,
) -> torch.Tensor:
    adv = torch.ones_like(base, dtype=torch.float32) * float(a_s)
    if not turn_rewards:
        return adv
    mean_r = _mean([float(x) for x in turn_rewards])
    if not turn_token_spans or len(turn_token_spans) != len(turn_rewards):
        return adv
    n = int(adv.numel())
    for span, r_i in zip(turn_token_spans, turn_rewards, strict=False):
        if not span or len(span) < 2:
            continue
        start, end = int(span[0]), int(span[1])
        if end <= start:
            continue
        start = max(0, min(start, n))
        end = max(start, min(end, n))
        adv[start:end] = float(a_s) + (float(r_i) - mean_r)
    return adv


def compute_turn_advantages(args: Namespace, rollout_data: dict[str, Any]) -> None:
    """Populate ``advantages`` / ``returns`` with turn-painted GRPO advantages."""
    del args  # rewards already group-normalized upstream
    kl: list[torch.Tensor] = rollout_data["kl"]
    rewards: list[float] = list(rollout_data["rewards"])
    metadata_list = rollout_data.get("metadata") or [None] * len(kl)

    advantages: list[torch.Tensor] = []
    for i, (k, a_s) in enumerate(zip(kl, rewards, strict=False)):
        md = metadata_list[i] if i < len(metadata_list) else None
        md = md if isinstance(md, dict) else {}
        turn_rewards = list(md.get("turn_rewards") or [])
        spans = md.get("turn_token_spans")
        if spans is not None and not isinstance(spans, list):
            spans = None
        if turn_rewards and spans is None:
            logger.debug(
                "compute_turn_advantages: sample %d missing turn_token_spans; broadcast A_s",
                i,
            )
        advantages.append(
            _paint_sample_advantage(
                k,
                a_s=float(a_s),
                turn_rewards=turn_rewards,
                turn_token_spans=spans,
            )
        )

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [a.clone() for a in advantages]
