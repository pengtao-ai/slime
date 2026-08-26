"""GiGPO for coding-agent RL (ScaleSWE + Tmax).

Paper: https://arxiv.org/abs/2505.10978 (verl-agent ``gigpo/core_gigpo.py``).

  r_imm[last segment] = episode reward, else 0
  G_t                 = discounted return along intra-traj segments (γ)
  A_E                 = episode mean-norm within (protocol, instance)
  A_S                 = step mean-norm of G within (protocol, instance, intent, tools)
  A                   = A_E + w · A_S

Intra-traj segments:
  * scaleswe — split when cumulative git diff changes
  * tmax     — split when an Edit/Write happens (no meaningful git)

Inter-traj groups never mix ScaleSWE with Tmax.

Wire with::

    --custom-reward-post-process-path examples.coding_agent_rl.gigpo.post_process_rewards
    --custom-advantage-function-path  examples.coding_agent_rl.gigpo.compute_advantages
    --gigpo-gamma 0.95
    --gigpo-step-advantage-w 1.0

``post_process_rewards`` runs on the full rollout (before DP split) so sibling
trajectories can form step groups. ``compute_advantages`` only paints the
precomputed per-turn advantages onto response tokens.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from argparse import Namespace
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from slime.utils.gigpo import (
    build_step_group,
    compute_step_discounted_returns,
    episode_norm_reward,
    step_norm_reward,
)

logger = logging.getLogger(__name__)

DEFAULT_GAMMA = 0.95
DEFAULT_STEP_W = 1.0

PROTOCOL_SCALESWE = "scaleswe"
PROTOCOL_TMAX = "tmax"
PROTOCOL_SWEBENCH = "swebench"

_EDIT_TOOLS = {"Edit", "Write", "NotebookEdit"}
_BASH_READ = {"cat", "head", "tail", "less", "more", "nl", "bat"}
_BASH_GREP = {"grep", "egrep", "fgrep", "rg", "ag"}
_BASH_SED = {"sed", "awk"}
_BASH_PYTHON = {"python", "python3", "pypy", "pypy3"}
_BASH_PYTEST = {"pytest", "py.test"}
_BASH_LS = {"ls", "tree", "pwd", "find", "stat", "realpath"}
_BASH_BUILD = {"gcc", "g++", "cc", "make", "cmake", "cargo", "go", "javac", "mvn", "gradle"}
_ENV = {"pip", "pip3", "pipx", "poetry", "uv", "conda", "apt", "apt-get", "npm", "pnpm", "yarn"}
_TEST_RUN_HINT = re.compile(
    r"\b(pytest|py\.test|unittest|nose2?|tox|nox|hatch\s+test|npm\s+test|cargo\s+test|go\s+test|make\s+test)\b",
    re.I,
)
_TEST_PATH = re.compile(
    r"(^|/)(tests?|testing|testdata|specs?|conftest\.py|test_[^/]+\.py|[^/]+_test\.py)(/|$)",
    re.I,
)
DIFF_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)


def _is_sft(sample: Any) -> bool:
    tmd = getattr(sample, "train_metadata", None) or {}
    return isinstance(tmd, dict) and tmd.get("objective") == "sft"


def protocol_of(sample_or_md: Any) -> str:
    md = sample_or_md if isinstance(sample_or_md, dict) else (getattr(sample_or_md, "metadata", None) or {})
    tmd = {} if isinstance(sample_or_md, dict) else (getattr(sample_or_md, "train_metadata", None) or {})
    raw = str((tmd or {}).get("protocol") or md.get("protocol") or PROTOCOL_SCALESWE).strip().lower()
    if raw == PROTOCOL_TMAX:
        return PROTOCOL_TMAX
    if raw == PROTOCOL_SWEBENCH:
        return PROTOCOL_SWEBENCH
    return PROTOCOL_SCALESWE


def is_tmax(protocol: str | None) -> bool:
    return str(protocol or "").strip().lower() == PROTOCOL_TMAX


def _bash_binaries(command: str) -> list[str]:
    parts = re.split(r"\s*(?:&&|\|\||;|\|)\s*", command or "")
    binaries: list[str] = []
    for part in parts:
        toks = part.split()
        while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):
            toks = toks[1:]
        if toks and toks[0] in {"sudo", "time", "env", "command"}:
            toks = toks[1:]
            while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):
                toks = toks[1:]
        if not toks:
            continue
        binaries.append(toks[0].rsplit("/", 1)[-1])
    return binaries


def bash_kind(command: str) -> str:
    bins = _bash_binaries(command)
    text = (command or "").lower()
    if any(b in _BASH_PYTEST for b in bins) or "-m pytest" in text or re.search(r"\bpytest\b", text):
        return "pytest"
    if _TEST_RUN_HINT.search(command or ""):
        return "testrun"
    if any(b in _BASH_BUILD for b in bins):
        return "build"
    if any(b in _ENV for b in bins):
        return "env"
    if any(b in _BASH_PYTHON for b in bins):
        return "python"
    if any(b in _BASH_GREP for b in bins):
        return "grep"
    if any(b in _BASH_READ for b in bins):
        return "cat"
    if any(b in _BASH_SED for b in bins):
        return "sed"
    if any(b == "git" for b in bins):
        return "git"
    if any(b in _BASH_LS for b in bins):
        return "ls"
    return "other"


def parse_manager_tool_calls(message: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Compact tool records from an adapter ``manager_message``."""
    if not message:
        return []
    out: list[dict[str, Any]] = []
    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = str(fn.get("name") or tc.get("name") or "")
        args = fn.get("arguments") if "arguments" in fn else tc.get("input") or tc.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        rec: dict[str, Any] = {"name": name}
        if name == "Bash":
            rec["kind"] = bash_kind(str(args.get("command") or ""))
        path = args.get("file_path") or args.get("path") or args.get("target_file")
        if path:
            rec["path"] = str(path)
        out.append(rec)
    return out


def tool_labels_from_calls(calls: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for rec in calls:
        name = str(rec.get("name") or "")
        if name == "Bash":
            labels.append(f"Bash:{rec.get('kind') or bash_kind(str(rec.get('command') or ''))}")
        elif name:
            labels.append(name)
    return list(dict.fromkeys(labels))


def _is_test_path(path: str) -> bool:
    return bool(_TEST_PATH.search((path or "").replace("\\", "/")))


def _diff_paths(diff: str) -> list[str]:
    paths: list[str] = []
    for a, b in DIFF_RE.findall(diff or ""):
        paths.append(b if b != "/dev/null" else a)
    return paths


def _diff_hash(diff: str) -> str:
    text = (diff or "").strip()
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _edit_signature(calls: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for rec in calls:
        name = str(rec.get("name") or "")
        if name in _EDIT_TOOLS:
            parts.append(f"{name}:{rec.get('path') or ''}")
    return "|".join(parts)


def scaleswe_intent(labels: list[str], *, empty_diff: bool, diff_has_test: bool, edit_paths: list[str]) -> str:
    names = set(labels)
    bash = {lab.split(":", 1)[1] for lab in labels if lab.startswith("Bash:")}
    edits = [p for p in edit_paths if p] or [lab for lab in labels if lab in _EDIT_TOOLS]
    if any(n in names for n in _EDIT_TOOLS) or edit_paths:
        paths = edit_paths if edit_paths else (["test"] if diff_has_test else [])
        if paths and all(_is_test_path(p) or p == "test" for p in paths):
            return "写/改测试"
        return "实现修复"
    test_run = bool(bash & {"pytest", "testrun"})
    if test_run:
        return "复现失败" if empty_diff else "验证修复"
    if not empty_diff:
        return "改后阅读"
    return "探索定位"


def tmax_intent(labels: list[str], *, dirty: bool, edit_paths: list[str]) -> str:
    names = set(labels)
    bash = {lab.split(":", 1)[1] for lab in labels if lab.startswith("Bash:")}
    if any(n in names for n in _EDIT_TOOLS) or edit_paths:
        return "实现修复"
    run_like = bool(bash & {"pytest", "testrun", "build", "python", "other"})
    if run_like:
        return "复现试跑" if not dirty else "验证运行"
    if dirty:
        return "改后阅读"
    return "探索定位"


def intent_for_turn(protocol: str, labels: list[str], *, empty_diff: bool, diff_has_test: bool, edit_paths: list[str], dirty: bool) -> str:
    if is_tmax(protocol):
        return tmax_intent(labels, dirty=dirty, edit_paths=edit_paths)
    return scaleswe_intent(labels, empty_diff=empty_diff, diff_has_test=diff_has_test, edit_paths=edit_paths)


def compact_turns(protocol: str | None, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build compact per-turn records for train_metadata (no full git diffs)."""
    proto = PROTOCOL_TMAX if is_tmax(protocol) else PROTOCOL_SCALESWE
    dirty = False
    seen_edits: list[str] = []
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(records or []):
        if not isinstance(rec, dict):
            continue
        diff = str(rec.get("git_diff") or "")
        calls = list(rec.get("tool_calls") or [])
        labels = tool_labels_from_calls(calls)
        edit_paths = [str(c.get("path") or "") for c in calls if str(c.get("name") or "") in _EDIT_TOOLS]
        edit_sig = _edit_signature(calls)
        if edit_sig:
            seen_edits.append(edit_sig)
        empty_diff = not diff.strip()
        if is_tmax(proto):
            dirty = bool(seen_edits)
        else:
            dirty = (not empty_diff) or bool(seen_edits)
        paths = _diff_paths(diff)
        diff_has_test = any(_is_test_path(p) for p in paths) or "test" in ",".join(paths).lower()
        turn = {
            "turn": int(rec.get("turn_index") if rec.get("turn_index") is not None else i),
            "diff_key": _diff_hash(diff),
            "empty_diff": empty_diff,
            "diff_has_test": bool(diff_has_test),
            "edit_key": "\n".join(seen_edits),
            "tools": labels,
            "tools_str": ", ".join(labels) or "无 tool",
        }
        turn["intent"] = intent_for_turn(
            proto,
            labels,
            empty_diff=empty_diff,
            diff_has_test=diff_has_test,
            edit_paths=edit_paths,
            dirty=dirty,
        )
        out.append(turn)
    return out


def split_key(turn: dict[str, Any], protocol: str) -> str:
    if is_tmax(protocol):
        return str(turn.get("edit_key") or "")
    return str(turn.get("diff_key") or "")


def segment_ids(turns: list[dict[str, Any]], protocol: str) -> list[int]:
    """Assign a segment id to each turn; a new segment starts when the split key changes."""
    ids: list[int] = []
    last: str | None = None
    seg = -1
    for turn in turns:
        key = split_key(turn, protocol)
        if last is None or key != last:
            seg += 1
            last = key
        ids.append(seg)
    return ids if ids else [0]


def episode_group_key(protocol: str, instance_id: str, group_index: Any) -> str:
    inst = instance_id or ("" if group_index is None else str(group_index))
    return f"{protocol}::{inst}"


def _sample_fields(sample: Any) -> tuple[str, str, Any, list[dict[str, Any]], float]:
    md = getattr(sample, "metadata", None) or {}
    tmd = getattr(sample, "train_metadata", None) or {}
    proto = protocol_of(sample)
    instance_id = str(tmd.get("instance_id") or md.get("instance_id") or getattr(sample, "label", None) or "")
    group_index = getattr(sample, "group_index", None)
    turns = list(tmd.get("gigpo_turns") or md.get("gigpo_turns") or [])
    if not turns:
        records = list(md.get("turn_git_diffs") or [])
        if records:
            turns = compact_turns(proto, records)
    if not turns:
        turns = [
            {
                "turn": 0,
                "diff_key": "",
                "empty_diff": True,
                "diff_has_test": False,
                "edit_key": "",
                "tools": [],
                "tools_str": "无 tool",
                "intent": "探索定位",
            }
        ]
    reward = float(md.get("episode_reward", getattr(sample, "reward", 0.0) or 0.0) or 0.0)
    return proto, instance_id, group_index, turns, reward


def _scalar_list(painted: list[torch.Tensor] | torch.Tensor) -> list[float]:
    if isinstance(painted, torch.Tensor):
        return [float(x) for x in painted.detach().reshape(-1).tolist()]
    out: list[float] = []
    for t in painted:
        out.append(float(t.detach().reshape(-1)[0]))
    return out


def assign_gigpo_to_samples(
    samples: list[Any],
    *,
    gamma: float = DEFAULT_GAMMA,
    step_w: float = DEFAULT_STEP_W,
) -> None:
    """Write per-turn A_E / A_S / A onto each sample's train_metadata.

    Must run on the full rollout (all sibling trajectories visible).
    """
    rows: list[dict[str, Any]] = []
    sample_turns: dict[int, list[dict[str, Any]]] = {}

    for si, sample in enumerate(samples):
        if _is_sft(sample) or getattr(sample, "remove_sample", False):
            continue
        proto, instance_id, group_index, turns, episode_r = _sample_fields(sample)
        segs = segment_ids(turns, proto)
        sample_turns[si] = turns
        traj_uid = f"{si}:{getattr(sample, 'index', si)}"
        ekey = episode_group_key(proto, instance_id, group_index)
        for turn, seg in zip(turns, segs, strict=True):
            rows.append(
                {
                    "sample_i": si,
                    "turn_i": int(turn.get("turn") or 0),
                    "seg": int(seg),
                    "traj_uid": traj_uid,
                    "episode_key": ekey,
                    "episode_reward": episode_r,
                    "protocol": proto,
                    "intent": str(turn.get("intent") or "探索定位"),
                    "tools_str": str(turn.get("tools_str") or "无 tool"),
                }
            )

    if not rows:
        return

    last_seg: dict[str, int] = {}
    first_of_seg: dict[tuple[str, int], int] = {}
    for i, row in enumerate(rows):
        last_seg[row["traj_uid"]] = max(last_seg.get(row["traj_uid"], -1), int(row["seg"]))
        first_of_seg.setdefault((row["traj_uid"], int(row["seg"])), i)

    # Discount along unique (traj, segment) then broadcast G to member turns.
    # Immediate reward is the episode return on the last segment only (Eq. 5).
    seg_order = sorted(first_of_seg.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    imm: list[float] = []
    trajs: list[str] = []
    for (traj_uid, seg), i in seg_order:
        trajs.append(traj_uid)
        imm.append(float(rows[i]["episode_reward"]) if seg == last_seg[traj_uid] else 0.0)
    g_seg = compute_step_discounted_returns(imm, trajs, gamma)
    g_by_seg: dict[tuple[str, int], float] = {}
    for (key, _), value in zip(seg_order, g_seg.tolist(), strict=True):
        g_by_seg[key] = float(value)
    g = torch.tensor([g_by_seg[(r["traj_uid"], int(r["seg"]))] for r in rows], dtype=torch.float32)

    episode = torch.tensor([r["episode_reward"] for r in rows], dtype=torch.float32)
    masks = [torch.ones(1, dtype=torch.float32) for _ in rows]
    episode_keys = [r["episode_key"] for r in rows]
    traj_uids = [r["traj_uid"] for r in rows]
    a_e = _scalar_list(
        episode_norm_reward(
            episode,
            masks,
            episode_keys,
            traj_uids,
            remove_std=True,
            compute_mean_std_cross_steps=False,
        )
    )
    anchors = [f"{r['intent']}||{r['tools_str']}" for r in rows]
    step_uids = build_step_group(np.asarray(anchors, dtype=object), episode_keys, summarize=False)
    a_s = _scalar_list(step_norm_reward(g, masks, step_uids, remove_std=True))

    by_sample: dict[int, list[float]] = defaultdict(list)
    by_sample_meta: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row, ae, as_, g_i, uid in zip(rows, a_e, a_s, g.tolist(), step_uids.tolist(), strict=True):
        adv = float(ae) + float(step_w) * float(as_)
        by_sample[row["sample_i"]].append(adv)
        by_sample_meta[row["sample_i"]].append(
            {
                "turn": row["turn_i"],
                "seg": row["seg"],
                "intent": row["intent"],
                "tools_str": row["tools_str"],
                "G": float(g_i),
                "A_E": float(ae),
                "A_S": float(as_),
                "A": adv,
                "step_uid": str(uid),
            }
        )

    for si, sample in enumerate(samples):
        tmd = dict(getattr(sample, "train_metadata", None) or {})
        md = dict(getattr(sample, "metadata", None) or {})
        if si not in by_sample:
            tmd.pop("gigpo_turn_advantages", None)
            sample.train_metadata = tmd
            continue
        tmd["gigpo_turn_advantages"] = by_sample[si]
        tmd["gigpo_turns"] = sample_turns.get(si) or tmd.get("gigpo_turns") or []
        tmd["protocol"] = protocol_of(sample)
        tmd["instance_id"] = str(tmd.get("instance_id") or md.get("instance_id") or "")
        tmd["gigpo_gamma"] = float(gamma)
        tmd["gigpo_step_w"] = float(step_w)
        sample.train_metadata = tmd
        md["gigpo_turn_advantages"] = by_sample[si]
        md["gigpo_step_rows"] = by_sample_meta[si]
        sample.metadata = md


def _gigpo_hparams(args: Namespace | None) -> tuple[float, float]:
    gamma = DEFAULT_GAMMA
    step_w = DEFAULT_STEP_W
    if args is not None:
        gamma = float(getattr(args, "gigpo_gamma", gamma) or gamma)
        step_w = float(getattr(args, "gigpo_step_advantage_w", step_w) or 0.0)
    return gamma, step_w


def post_process_rewards(args: Namespace, samples: list[Any]):
    """Assign GiGPO on the full rollout, then GRPO-normalize scalar rewards for logs."""
    from examples.coding_agent_rl.offload_sft import post_process_rewards_grpo_only

    gamma, step_w = _gigpo_hparams(args)
    assign_gigpo_to_samples(samples, gamma=gamma, step_w=step_w)
    return post_process_rewards_grpo_only(args, samples)


def _paint_turn_advantages(
    base: torch.Tensor,
    *,
    advantages: list[float],
    turn_token_spans: list[list[int]] | None,
    fallback: float,
) -> torch.Tensor:
    adv = torch.ones_like(base, dtype=torch.float32) * float(fallback)
    if not advantages:
        return adv
    n = int(adv.numel())
    if not turn_token_spans or len(turn_token_spans) != len(advantages):
        # No reliable spans: broadcast the trajectory-mean GiGPO advantage.
        return torch.ones_like(base, dtype=torch.float32) * float(sum(advantages) / max(len(advantages), 1))
    for span, value in zip(turn_token_spans, advantages, strict=False):
        if not span or len(span) < 2:
            continue
        start, end = int(span[0]), int(span[1])
        if end <= start:
            continue
        start = max(0, min(start, n))
        end = max(start, min(end, n))
        adv[start:end] = float(value)
    return adv


def compute_advantages(args: Namespace, rollout_data: dict[str, Any]) -> None:
    """Paint precomputed GiGPO turn advantages onto each sample's response tokens."""
    del args
    kl: list[torch.Tensor] = rollout_data["kl"]
    rewards: list[float] = list(rollout_data.get("rewards") or [0.0] * len(kl))
    metadata_list = rollout_data.get("metadata") or [None] * len(kl)

    advantages: list[torch.Tensor] = []
    for i, k in enumerate(kl):
        md = metadata_list[i] if i < len(metadata_list) else None
        md = md if isinstance(md, dict) else {}
        if md.get("objective") == "sft":
            advantages.append(torch.zeros_like(k, dtype=torch.float32))
            continue
        turn_adv = [float(x) for x in (md.get("gigpo_turn_advantages") or [])]
        spans = md.get("turn_token_spans")
        if spans is not None and not isinstance(spans, list):
            spans = None
        fallback = float(rewards[i]) if i < len(rewards) else 0.0
        advantages.append(
            _paint_turn_advantages(
                k,
                advantages=turn_adv,
                turn_token_spans=spans,
                fallback=fallback,
            )
        )

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [a.clone() for a in advantages]
