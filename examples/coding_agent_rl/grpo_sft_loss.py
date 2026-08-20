"""Mixed GRPO + SFT loss for coding-agent offload training.

Wire with::

    --loss-type custom_loss
    --custom-loss-function-path examples.coding_agent_rl.grpo_sft_loss.grpo_sft_loss_function

Computes ``L = L_GRPO + λ L_SFT`` where SFT rows are identified by
``batch["metadata"][i]["objective"] == "sft"``. λ comes from ``OFFLOAD_SFT_LAMBDA``
(default 0 → pure GRPO).

SFT CE reuses ``batch["_cat_log_probs"]`` from ``policy_loss_function`` so we do
not run a second log-softmax over the vocab (that second pass OOMs on long mbs).
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch

from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from slime.backends.megatron_utils.loss import policy_loss_function
from slime.utils.types import RolloutBatch


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
    # Train actor process: same resolved knobs as rollout (once per process).
    from examples.coding_agent_rl import offload as _offload

    _offload.log_train_config_once(args)

    pg_loss, reported = policy_loss_function(args, batch, logits, sum_of_sample_mean)
    log_probs = batch.pop("_cat_log_probs", None)

    lam = _sft_lambda()
    metadata = batch.get("metadata") or []
    sft_indices = {i for i, meta in enumerate(metadata) if _is_sft_meta(meta)}
    reported = dict(reported)
    if lam <= 0.0 or not sft_indices or log_probs is None:
        # pg_loss already depends on logits; do not add logits.sum() (that
        # materializes a second [T, V] backward and OOMs on packed mbs).
        reported["sft_loss"] = pg_loss.detach() * 0.0
        reported["loss"] = pg_loss.clone().detach()
        return pg_loss, reported

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    loss_masks = batch["loss_masks"]
    sft_masks = [
        mask if i in sft_indices else torch.zeros_like(mask) for i, mask in enumerate(loss_masks)
    ]
    sft_reducer = get_sum_of_sample_mean(
        total_lengths,
        response_lengths,
        sft_masks,
        sample_denoms=None,
        calculate_per_token_loss=getattr(args, "calculate_per_token_loss", False),
    )
    sft_nll = -sft_reducer(log_probs)

    loss = pg_loss + lam * sft_nll
    reported["sft_loss"] = sft_nll.detach()
    reported["pg_loss"] = reported.get("pg_loss", pg_loss.detach())
    reported["loss"] = loss.clone().detach()
    return loss, reported
