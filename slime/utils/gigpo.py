"""GiGPO (Group-in-Group Policy Optimization) core — adapted from verl-agent.

Reference: https://github.com/langfengQ/verl-agent/blob/master/gigpo/core_gigpo.py
Paper: https://arxiv.org/abs/2505.10978

No DataProto dependency; operates on plain tensors / numpy arrays.
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def to_hashable(x: Any) -> Any:
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return tuple(x.flatten().tolist())
    if isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    if isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    raise TypeError(f"Unsupported type: {type(x)}")


def _uid_array(uids: np.ndarray | list[Any]) -> np.ndarray:
    """Object array of string UIDs.

    Batches may mix int ``rollout_id`` / ``index`` fallbacks with string
    ``traj_uid`` (e.g. ``\"12:main\"``). ``np.unique`` on a heterogeneous
    object array tries to sort and raises ``TypeError: '<' not supported
    between instances of 'int' and 'str'``.
    """
    arr = np.asarray(uids, dtype=object).reshape(-1)
    return np.asarray([str(u) for u in arr.tolist()], dtype=object)


def summarize_group_size(group_size: list[int]) -> dict[str, Any]:
    """Build / log a histogram of step-group sizes. Returns a compact summary dict."""
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts) if counts else 0
    hist = {size: counts[size] for size in sorted(counts)}
    n_matched = sum(c for sz, c in counts.items() if sz >= 2)
    summary = {
        "n_step_groups": total,
        "avg_size": float(np.mean(group_size)) if group_size else 0.0,
        "max_size": max_size,
        "n_singleton": counts.get(1, 0),
        "n_matched_groups": n_matched,  # groups with size >= 2
        "hist": hist,
    }
    hist_str = " ".join(f"{sz}:{cnt}" for sz, cnt in hist.items())
    logger.info(
        "GiGPO step-groups: n=%d avg=%.2f max=%d singleton=%d matched=%d | hist[size:count] %s",
        summary["n_step_groups"],
        summary["avg_size"],
        summary["max_size"],
        summary["n_singleton"],
        summary["n_matched_groups"],
        hist_str or "-",
    )
    return summary


def compute_step_discounted_returns(
    rewards: np.ndarray | list[float],
    traj_uids: np.ndarray | list[Any],
    gamma: float,
) -> torch.Tensor:
    """Discounted returns along each trajectory (Eq. 5)."""
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    traj_arr = _uid_array(traj_uids)
    returns_by_traj: dict[str, np.ndarray] = {}
    for uid in np.unique(traj_arr):
        traj_indices = np.where(traj_arr == uid)[0]
        traj_rewards = rewards_arr[traj_indices]
        traj_returns = np.zeros_like(traj_rewards)
        running = 0.0
        for t in reversed(range(len(traj_rewards))):
            running = float(traj_rewards[t]) + gamma * running
            traj_returns[t] = running
        returns_by_traj[str(uid)] = traj_returns

    all_returns = np.zeros_like(rewards_arr)
    for i, uid in enumerate(traj_arr):
        traj_indices = np.where(traj_arr == uid)[0]
        idx_in_traj = int(np.where(traj_indices == i)[0][0])
        all_returns[i] = returns_by_traj[str(uid)][idx_in_traj]
    return torch.tensor(all_returns, dtype=torch.float32)


def build_step_group(
    anchor_obs: np.ndarray | list[Any],
    index: np.ndarray | list[Any],
    *,
    summarize: bool = False,
    sizes_out: list[int] | None = None,
) -> np.ndarray:
    """Cluster identical observations within each episode group → step_group_uid."""
    anchor_arr = np.asarray(anchor_obs, dtype=object)
    index_arr = _uid_array(index)
    step_group_uids = np.empty(len(anchor_arr), dtype=object)
    group_size: list[int] = []

    for idx in np.unique(index_arr):
        locs = np.where(index_arr == idx)[0]
        clusters: dict[Any, list[int]] = defaultdict(list)
        for loc in locs:
            clusters[to_hashable(anchor_arr[loc])].append(int(loc))
        for _obs, original_indices in clusters.items():
            uid = str(uuid.uuid4())
            group_size.append(len(original_indices))
            for original_idx in original_indices:
                step_group_uids[original_idx] = uid

    if None in step_group_uids or np.any(step_group_uids == None):  # noqa: E711
        missing = np.where(step_group_uids == None)[0]  # noqa: E711
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing}")

    if sizes_out is not None:
        sizes_out.extend(group_size)
    if summarize and group_size:
        summarize_group_size(group_size)
    return step_group_uids


def episode_norm_reward(
    episode_rewards: torch.Tensor,
    response_mask: list[torch.Tensor] | None,
    index: np.ndarray | list[Any],
    traj_index: np.ndarray | list[Any],
    *,
    epsilon: float = 1e-6,
    remove_std: bool = True,
    compute_mean_std_cross_steps: bool = False,
) -> list[torch.Tensor]:
    """Episode-level advantage; one scalar per step row, broadcast to tokens.

    When ``compute_mean_std_cross_steps=False`` (default here for SWE), each
    ``(index, traj_index)`` pair contributes once to the group mean/std.
    """
    scores = episode_rewards.clone().float()
    index_arr = _uid_array(index)
    traj_arr = _uid_array(traj_index)

    id2score: dict[Any, list[torch.Tensor]] = defaultdict(list)
    seen_pairs: set[tuple[Any, Any]] = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            key = (index_arr[i], traj_arr[i])
            if key in seen_pairs:
                continue
            id2score[index_arr[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add(key)

        id2mean: dict[Any, torch.Tensor] = {}
        id2std: dict[Any, torch.Tensor] = {}
        for idx, vals in id2score.items():
            if len(vals) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                stacked = torch.stack([v.float() for v in vals])
                id2mean[idx] = stacked.mean()
                id2std[idx] = stacked.std()

        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index_arr[i]]
            else:
                scores[i] = (scores[i] - id2mean[index_arr[i]]) / (id2std[index_arr[i]] + epsilon)

    return _broadcast_to_tokens(scores, response_mask)


def step_norm_reward(
    step_rewards: torch.Tensor,
    response_mask: list[torch.Tensor] | None,
    step_group_uids: np.ndarray | list[Any],
    *,
    epsilon: float = 1e-6,
    remove_std: bool = True,
) -> list[torch.Tensor]:
    """Step-level advantage within each anchor group."""
    scores = step_rewards.clone().float()
    index_arr = np.asarray(step_group_uids, dtype=object)

    id2score: dict[Any, list[torch.Tensor]] = defaultdict(list)
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index_arr[i]].append(scores[i])

        id2mean: dict[Any, torch.Tensor] = {}
        id2std: dict[Any, torch.Tensor] = {}
        for idx, vals in id2score.items():
            stacked = torch.stack([v.float() for v in vals])
            id2mean[idx] = stacked.mean()
            if len(vals) == 1:
                id2std[idx] = torch.tensor(1.0)
            else:
                id2std[idx] = stacked.std()

        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index_arr[i]]
            else:
                scores[i] = (scores[i] - id2mean[index_arr[i]]) / (id2std[index_arr[i]] + epsilon)

    return _broadcast_to_tokens(scores, response_mask)


def compute_gigpo_outcome_advantage(
    episode_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: list[torch.Tensor],
    anchor_obs: np.ndarray | list[Any],
    index: np.ndarray | list[Any],
    traj_index: np.ndarray | list[Any],
    *,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    mode: str = "mean_norm",
    summarize_step_groups: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Joint GiGPO advantages: A = A_E + w * A_S."""
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    episode_advantages = episode_norm_reward(
        episode_rewards,
        response_mask,
        index,
        traj_index,
        epsilon=epsilon,
        remove_std=remove_std,
        compute_mean_std_cross_steps=False,
    )
    step_group_uids = build_step_group(anchor_obs, index, summarize=summarize_step_groups)
    step_advantages = step_norm_reward(
        step_rewards,
        response_mask,
        step_group_uids,
        epsilon=epsilon,
        remove_std=remove_std,
    )

    advantages = [
        e + step_advantage_w * s for e, s in zip(episode_advantages, step_advantages, strict=True)
    ]
    return advantages, advantages


def _broadcast_to_tokens(
    scores: torch.Tensor,
    response_mask: list[torch.Tensor] | None,
) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for i in range(scores.shape[0]):
        if response_mask is None:
            out.append(scores[i].reshape(1))
            continue
        mask = response_mask[i]
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.float32, device=scores.device)
        else:
            mask = mask.to(device=scores.device, dtype=torch.float32)
        out.append(scores[i] * torch.ones_like(mask) * mask)
    return out
