"""GiGPO core (https://arxiv.org/abs/2505.10978).

Ported from verl-agent ``gigpo/core_gigpo.py`` without ``DataProto``. Trainer
code and coding-agent grouping both call these primitives.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import torch

_Index = Any
_MaskList = list[torch.Tensor] | torch.Tensor


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


def summarize_group_size(group_size: list[int]) -> dict[int, tuple[int, float]]:
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts) if counts else 0
    summary: dict[int, tuple[int, float]] = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0.0
        summary[size] = (cnt, prop)
    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
    return summary


def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO.")
    return SequenceMatcher(None, a, b).ratio() >= threshold


def compute_step_discounted_returns(
    rewards: list[float] | np.ndarray | torch.Tensor,
    traj_uids: list[Any] | np.ndarray,
    gamma: float,
) -> torch.Tensor:
    """Discounted returns per step (Eq. 5). Caller must order steps within each traj."""
    rewards_np = np.asarray(
        rewards.detach().cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards,
        dtype=np.float32,
    )
    uids = np.asarray(traj_uids, dtype=object)
    if rewards_np.shape[0] != uids.shape[0]:
        raise ValueError(f"rewards ({rewards_np.shape[0]}) and traj_uids ({uids.shape[0]}) length mismatch")
    out = np.zeros_like(rewards_np)
    for uid in dict.fromkeys(uids.tolist()):
        idx = np.where(uids == uid)[0]
        running = 0.0
        for t in reversed(range(len(idx))):
            running = float(rewards_np[idx[t]]) + float(gamma) * running
            out[idx[t]] = running
    return torch.tensor(out, dtype=torch.float32)


def _episode_scores(token_level_rewards: torch.Tensor | list[float]) -> torch.Tensor:
    if not isinstance(token_level_rewards, torch.Tensor):
        return torch.tensor(list(token_level_rewards), dtype=torch.float32)
    if token_level_rewards.ndim == 1:
        return token_level_rewards.float().clone()
    return token_level_rewards.sum(dim=-1).float()


def _paint_scores(scores: torch.Tensor, response_mask: _MaskList) -> list[torch.Tensor] | torch.Tensor:
    if isinstance(response_mask, torch.Tensor) and response_mask.ndim == 2:
        return scores.unsqueeze(-1).expand(-1, response_mask.shape[-1]) * response_mask.float()
    painted: list[torch.Tensor] = []
    for i, mask in enumerate(response_mask):
        m = mask if isinstance(mask, torch.Tensor) else torch.tensor(mask)
        painted.append(torch.ones_like(m, dtype=torch.float32) * scores[i] * m.float())
    return painted


def episode_norm_reward(
    token_level_rewards: torch.Tensor | list[float],
    response_mask: _MaskList,
    index: list[Any] | np.ndarray,
    traj_index: list[Any] | np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True,
    compute_mean_std_cross_steps: bool = True,
) -> list[torch.Tensor] | torch.Tensor:
    """Episode-level advantage (Eq. 3). One scalar reward per episode.

    ``compute_mean_std_cross_steps=False`` counts each ``(index, traj_index)``
    once so repeated steps from the same trajectory do not inflate the mean.
    """
    scores = _episode_scores(token_level_rewards)
    index_arr = np.asarray(index, dtype=object)
    traj_arr = np.asarray(traj_index, dtype=object)
    id2score: dict[Any, list[torch.Tensor]] = defaultdict(list)
    seen_pairs: set[tuple[Any, Any]] = set()
    with torch.no_grad():
        for i in range(int(scores.shape[0])):
            pair = (index_arr[i], traj_arr[i])
            if pair in seen_pairs:
                continue
            id2score[index_arr[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add(pair)

        id2mean: dict[Any, torch.Tensor] = {}
        id2std: dict[Any, torch.Tensor] = {}
        for idx, vals in id2score.items():
            if len(vals) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(vals) > 1:
                stacked = torch.stack([v.detach().float().reshape(()) for v in vals])
                id2mean[idx] = stacked.mean()
                id2std[idx] = stacked.std()
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        out = scores.clone()
        for i in range(int(out.shape[0])):
            mean = id2mean[index_arr[i]]
            if remove_std:
                out[i] = out[i] - mean
            else:
                out[i] = (out[i] - mean) / (id2std[index_arr[i]] + epsilon)
    return _paint_scores(out, response_mask)


def build_step_group(
    anchor_obs: np.ndarray | list[Any],
    index: np.ndarray | list[Any],
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    summarize: bool = False,
) -> np.ndarray:
    """Cluster identical (or similar) anchors within each ``index`` group (Eq. 6)."""
    if enable_similarity and not (0.0 < similarity_thresh < 1.0):
        raise ValueError("When enabling similarity-based grouping, similarity_thresh should be in (0, 1)")

    anchors = np.asarray(anchor_obs, dtype=object)
    index_arr = np.asarray(index, dtype=object)
    step_group_uids = np.empty(len(anchors), dtype=object)
    group_size: list[int] = []

    for idx in dict.fromkeys(index_arr.tolist()):
        locs = np.where(index_arr == idx)[0]
        obs_group = anchors[locs]
        if not enable_similarity:
            clusters: dict[Any, list[int]] = defaultdict(list)
            for loc, obs in zip(locs.tolist(), obs_group.tolist(), strict=True):
                clusters[to_hashable(obs)].append(loc)
            for original_indices in clusters.values():
                uid = str(uuid.uuid4())
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
            continue

        clusters_sim: list[dict[str, Any]] = []
        for obs, loc in zip(obs_group.tolist(), locs.tolist(), strict=True):
            placed = False
            for cluster in clusters_sim:
                if are_similar(str(obs), str(cluster["rep"]), similarity_thresh):
                    cluster["locs"].append(loc)
                    placed = True
                    break
            if not placed:
                clusters_sim.append({"rep": obs, "locs": [loc]})
        for cluster in clusters_sim:
            uid = str(uuid.uuid4())
            group_size.append(len(cluster["locs"]))
            for loc in cluster["locs"]:
                step_group_uids[loc] = uid

    if None in step_group_uids.tolist():
        missing = np.where(step_group_uids == None)[0]  # noqa: E711
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing}")

    if summarize:
        summarize_group_size(group_size)
        print(f"Avg size of step-level group: {np.mean(group_size) if group_size else 0.0}")
    return step_group_uids


def step_norm_reward(
    step_rewards: torch.Tensor | list[float],
    response_mask: _MaskList,
    index: np.ndarray | list[Any],
    epsilon: float = 1e-6,
    remove_std: bool = True,
) -> list[torch.Tensor] | torch.Tensor:
    """Step-level advantage (Eq. 7). Singleton groups get A_S = 0."""
    scores = _episode_scores(step_rewards)
    index_arr = np.asarray(index, dtype=object)
    id2score: dict[Any, list[torch.Tensor]] = defaultdict(list)
    with torch.no_grad():
        for i in range(int(scores.shape[0])):
            id2score[index_arr[i]].append(scores[i])
        id2mean: dict[Any, torch.Tensor] = {}
        id2std: dict[Any, torch.Tensor] = {}
        for idx, vals in id2score.items():
            stacked = torch.stack([v.detach().float().reshape(()) for v in vals])
            id2mean[idx] = stacked.mean()
            id2std[idx] = torch.tensor(1.0) if len(vals) == 1 else stacked.std()
        out = scores.clone()
        for i in range(int(out.shape[0])):
            mean = id2mean[index_arr[i]]
            if remove_std:
                out[i] = out[i] - mean
            else:
                out[i] = (out[i] - mean) / (id2std[index_arr[i]] + epsilon)
    return _paint_scores(out, response_mask)
