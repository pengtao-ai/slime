"""Mixed GRPO + SFT loss for coding-agent offload training.

Wire with::

    --loss-type custom_loss
    --custom-loss-function-path examples.coding_agent_rl.grpo_sft_loss.grpo_sft_loss_function

Computes ``L = L_GRPO + λ L_SFT`` where SFT rows are identified by
``batch["metadata"][i]["objective"] == "sft"``. λ comes from ``OFFLOAD_SFT_LAMBDA``
(default 0 → pure GRPO).
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch

from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, policy_loss_function
from slime.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


def _sft_lambda() -> float:
    raw = (os.environ.get("OFFLOAD_SFT_LAMBDA") or "0").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _is_sft_meta(meta: Any) -> bool:
    return isinstance(meta, dict) and meta.get("objective") == "sft"


def grpo_sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Policy loss on GRPO rows + λ * CE on SFT rows (mask-aware)."""
    pg_loss, reported = policy_loss_function(args, batch, logits, sum_of_sample_mean)

    lam = _sft_lambda()
    metadata = batch.get("metadata") or []
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    loss_masks = batch["loss_masks"]

    sft_indices = {i for i, meta in enumerate(metadata) if _is_sft_meta(meta)}
    if lam <= 0.0 or not sft_indices:
        # Keep graph alive when no SFT contribution this step.
        sft_loss = logits.sum() * 0.0
        reported = dict(reported)
        reported["sft_loss"] = sft_loss.detach()
        reported["loss"] = pg_loss.clone().detach()
        return pg_loss + sft_loss, reported

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
    )
    log_probs_list = log_probs_and_entropy["log_probs"]

    # Zero loss_masks on non-SFT samples so the reducer only averages SFT CE.
    sft_masks: list[torch.Tensor] = []
    for i, mask in enumerate(loss_masks):
        if i in sft_indices:
            sft_masks.append(mask)
        else:
            sft_masks.append(torch.zeros_like(mask))

    sft_reducer = get_sum_of_sample_mean(
        total_lengths,
        response_lengths,
        sft_masks,
        sample_denoms=None,
        calculate_per_token_loss=getattr(args, "calculate_per_token_loss", False),
    )
    log_probs = torch.cat(log_probs_list, dim=0)
    sft_nll = -sft_reducer(log_probs)
    if log_probs.numel() == 0:
        sft_nll = sft_nll + 0 * logits.sum()

    loss = pg_loss + lam * sft_nll
    reported = dict(reported)
    reported["sft_loss"] = sft_nll.detach()
    reported["pg_loss"] = reported.get("pg_loss", pg_loss.detach())
    reported["loss"] = loss.clone().detach()
    return loss, reported
