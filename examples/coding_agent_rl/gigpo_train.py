"""GiGPO train hooks for coding-agent RL.

Wire via launcher env / args:
  --custom-reward-post-process-path examples.coding_agent_rl.gigpo_train.post_process_rewards
  --custom-convert-samples-to-train-data-path examples.coding_agent_rl.gigpo_train.convert_samples_to_train_data
  --custom-advantage-function-path examples.coding_agent_rl.gigpo_train.compute_advantages
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from examples.coding_agent_rl import offload
from slime.utils.gigpo import compute_gigpo_outcome_advantage, compute_step_discounted_returns
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return (os.environ.get(name) or default).strip()


def _env_bool(name: str, default: bool = True) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _session_key(sample: Sample) -> tuple[Any, Any]:
    """Group siblings that share one SWE session / episode."""
    md = sample.metadata or {}
    # Prefer explicit session id; fall back to (group_index, index).
    sid = md.get("session_id") or md.get("adapter_session_id")
    if sid is not None:
        return (sample.group_index, sid)
    return (sample.group_index, sample.index)


def _traj_key(sample: Sample) -> str:
    """Stable string traj id for GiGPO discounting / episode grouping.

    Must always be ``str``: batches mix per-turn ``traj_uid`` (e.g. ``\"12:main\"``)
    with int ``rollout_id`` / ``index`` fallbacks; heterogeneous object arrays
    break ``np.unique`` inside ``compute_step_discounted_returns``.
    """
    md = sample.metadata or {}
    if md.get("traj_uid") is not None:
        return str(md["traj_uid"])
    if sample.rollout_id is not None:
        return str(sample.rollout_id)
    return str(sample.index)


def post_process_rewards(args, samples: list[Sample]):
    """Group-aware offload shaping + pass-through rewards (no reshape GRPO).

    Returns ``(raw_rewards, processed_rewards)``. When
    ``OFFLOAD_REWARD_MODE=group_aware``, reshapes episode rewards using sibling
    sessions in the same ``group_index``. GiGPO advantage normalization happens
    later in :func:`compute_advantages`; here we only shape scalar ``R``.
    """
    raw = [float(s.get_reward_value(args)) for s in samples]
    mode = offload.reward_mode()

    if mode != "group_aware":
        # Still write episode_reward for GiGPO; leave rewards unchanged.
        for s, r in zip(samples, raw, strict=True):
            md = s.metadata if s.metadata is not None else {}
            s.metadata = md
            md["episode_reward"] = float(md.get("episode_reward", r) or r)
            if md.get("gigpo") and md.get("is_terminal_step") and md.get("branch_key") == "main":
                md["step_immediate_reward"] = float(md["episode_reward"])
        return raw, raw

    # Aggregate one representative row per session for shaping.
    by_group: dict[Any, list[Sample]] = defaultdict(list)
    for s in samples:
        by_group[s.group_index].append(s)

    shaped_by_session: dict[tuple[Any, Any], float] = {}
    log_groups = _env_bool("GIGPO_LOG_GROUPS", True)
    n_unique_bonus = 0
    n_no_offload_bonus = 0
    n_all_fail = 0
    for g, group_samples in by_group.items():
        # Dedup sessions.
        reps: dict[tuple[Any, Any], Sample] = {}
        for s in group_samples:
            reps.setdefault(_session_key(s), s)
        items = []
        keys = []
        for key, s in reps.items():
            md = s.metadata or {}
            items.append(
                {
                    "solved": float(md.get("solved", 1.0 if md.get("grading_solved") else 0.0) or 0.0),
                    "stats": md.get("offload_stats"),
                    "usage": md.get("usage"),
                    "empty_patch": bool(md.get("empty_patch", False)),
                }
            )
            keys.append(key)
        report: dict[str, Any] = {}
        rewards = offload.group_aware_rewards(items, report=report)
        for key, r in zip(keys, rewards, strict=True):
            shaped_by_session[key] = float(r)
        if report.get("unique_bonus_applied"):
            n_unique_bonus += 1
        n_no_offload_bonus += int(report.get("no_offload_bonus_count") or 0)
        if report.get("all_fail"):
            n_all_fail += 1
        if log_groups and report:
            instance_id = None
            for s in reps.values():
                md = s.metadata or {}
                instance_id = md.get("instance_id") or md.get("swe_instance_id")
                if instance_id:
                    break
            logger.info(
                "group_aware group=%s instance=%s n=%d solved=%d offload=%d "
                "all_fail=%s unique_bonus=%s no_offload_bonus=%d rewards=%s",
                g,
                instance_id,
                report.get("n"),
                report.get("n_solved"),
                report.get("n_offload"),
                report.get("all_fail"),
                report.get("unique_bonus_applied"),
                report.get("no_offload_bonus_count"),
                [round(float(x), 4) for x in (report.get("rewards") or [])],
            )

    if log_groups:
        logger.info(
            "group_aware batch: n_episode_groups=%d unique_bonus_triggers=%d "
            "no_offload_bonus_sessions=%d all_fail_groups=%d",
            len(by_group),
            n_unique_bonus,
            n_no_offload_bonus,
            n_all_fail,
        )

    processed: list[float] = []
    for s in samples:
        key = _session_key(s)
        r = shaped_by_session.get(key, float(s.get_reward_value(args)))
        processed.append(r)
        md = s.metadata if s.metadata is not None else {}
        s.metadata = md
        md["episode_reward"] = r
        s.reward = r
        if md.get("gigpo"):
            if md.get("branch_key") == "main" and md.get("is_terminal_step"):
                md["step_immediate_reward"] = r
            else:
                md["step_immediate_reward"] = float(md.get("step_immediate_reward", 0.0) or 0.0)

    return raw, processed


def convert_samples_to_train_data(args, samples: list[Sample] | list[list[Sample]]):
    """Like RolloutManager._convert_samples_to_train_data, plus GiGPO fields."""
    import itertools

    from slime.utils.types import Sample as SampleType

    flat: list[SampleType] = list(samples)  # type: ignore[arg-type]
    while flat and isinstance(flat[0], list):
        flat = list(itertools.chain.from_iterable(flat))  # type: ignore[arg-type]

    raw_rewards, rewards = post_process_rewards(args, flat)

    rollout_ids = [sample.rollout_id for sample in flat]
    existed = {rid for rid in rollout_ids if rid is not None}
    tmp_id = 0
    for i, rid in enumerate(rollout_ids):
        if rid is None:
            while tmp_id in existed:
                tmp_id += 1
            rollout_ids[i] = tmp_id
            existed.add(tmp_id)

    loss_masks = []
    for sample in flat:
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        assert len(sample.loss_mask) == sample.response_length
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)

    mask_sums = [sum(m) for m in loss_masks]
    rollout_total: dict[Any, int] = {}
    for rid, ms in zip(rollout_ids, mask_sums, strict=True):
        rollout_total[rid] = rollout_total.get(rid, 0) + ms

    train_data: dict[str, Any] = {
        "tokens": [s.tokens for s in flat],
        "response_lengths": [s.response_length for s in flat],
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if s.status == Sample.Status.TRUNCATED else 0 for s in flat],
        "sample_indices": [s.index for s in flat],
        "rollout_ids": rollout_ids,
        "loss_masks": loss_masks,
        "rollout_mask_sums": [rollout_total[rid] for rid in rollout_ids],
        # GiGPO extras
        "group_indices": [s.group_index for s in flat],
        "traj_uids": [_traj_key(s) for s in flat],
        "anchor_obs": [(s.metadata or {}).get("anchor_obs", "") for s in flat],
        "step_immediate_rewards": [
            float((s.metadata or {}).get("step_immediate_reward", 0.0) or 0.0) for s in flat
        ],
        "episode_rewards": [float((s.metadata or {}).get("episode_reward", r) or r) for s, r in zip(flat, rewards)],
        "turn_indices": [int((s.metadata or {}).get("turn_index") or 0) for s in flat],
    }

    if flat and flat[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [s.rollout_log_probs for s in flat]

    return train_data


def compute_advantages(args, rollout_data: dict[str, Any]) -> None:
    """Custom advantage hook: A = A_E + w * A_S (GiGPO).

    Broadcasts scalar advantages onto each sample's CP-local ``kl`` layout —
    same contract as ``get_grpo_returns`` — so CP>1 logging/training don't
    see full-response tensors.
    """
    kl: list[torch.Tensor] = rollout_data["kl"]
    device = kl[0].device if kl else torch.device("cpu")
    # Match GRPO: ones_like(kl[i]) * scalar. Do NOT use full loss_masks —
    # with context parallel those are longer than the local log-prob chunks.
    broadcast_masks = [torch.ones_like(k, dtype=torch.float32) for k in kl]

    episode_rewards = torch.tensor(
        rollout_data.get("episode_rewards") or rollout_data["rewards"],
        dtype=torch.float32,
        device=device,
    )
    step_imm = rollout_data.get("step_immediate_rewards")
    if step_imm is None:
        # Fallback: put episode reward on every row (GRPO-like) if no GiGPO metadata.
        step_imm = [float(r) for r in episode_rewards.tolist()]

    traj_uids = [str(u) for u in (rollout_data.get("traj_uids") or rollout_data.get("rollout_ids") or [])]
    group_indices = rollout_data.get("group_indices")
    if group_indices is None:
        group_indices = rollout_data.get("sample_indices")
    group_indices = [str(g) for g in (group_indices or [])]
    anchor_obs = rollout_data.get("anchor_obs")
    if anchor_obs is None:
        anchor_obs = [""] * len(episode_rewards)

    gamma = _env_float("GIGPO_GAMMA", float(getattr(args, "gamma", 0.95) or 0.95))
    step_w = _env_float("GIGPO_STEP_ADVANTAGE_W", 1.0)
    mode = _env_str("GIGPO_MODE", "mean_norm")

    # Reorder within each traj by turn_index so discounted returns are causal.
    n = len(step_imm)
    if len(traj_uids) != n:
        raise ValueError(f"traj_uids length {len(traj_uids)} != step_imm length {n}")
    if len(group_indices) != n:
        raise ValueError(f"group_indices length {len(group_indices)} != step_imm length {n}")
    turn_indices = rollout_data.get("turn_indices") or list(range(n))
    order = sorted(range(n), key=lambda i: (traj_uids[i], int(turn_indices[i])))
    step_imm_sorted = [step_imm[i] for i in order]
    traj_sorted = [traj_uids[i] for i in order]
    step_rewards_sorted = compute_step_discounted_returns(step_imm_sorted, traj_sorted, gamma).to(device)
    step_rewards = torch.zeros(n, dtype=torch.float32, device=device)
    for new_pos, old_i in enumerate(order):
        step_rewards[old_i] = step_rewards_sorted[new_pos]

    log_groups = _env_bool("GIGPO_LOG_GROUPS", True)
    advantages, returns = compute_gigpo_outcome_advantage(
        episode_rewards=episode_rewards,
        step_rewards=step_rewards,
        response_mask=broadcast_masks,
        anchor_obs=np.asarray(anchor_obs, dtype=object),
        index=group_indices,
        traj_index=traj_uids,
        step_advantage_w=step_w,
        mode=mode,
        summarize_step_groups=log_groups,
    )
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
    # Per-sample scalars for debugging; keep off the token-metric logging path.
    rollout_data["step_rewards"] = [float(x) for x in step_rewards.detach().cpu().tolist()]
