#!/usr/bin/env python3
"""Offline GiGPO step-group analysis from coding-agent rollout dumps.

The Jul-31 dumps are episode-packed Samples (not per-turn), but ``tokens``
contain the full chat with ``<tool_call>`` / ``<tool_response>``. This script
decodes them, rebuilds env-obs anchors via ``gigpo_anchor``, then clusters
identical anchors within each ``group_index`` (same instance siblings).

Example:
  python examples/coding_agent_rl/analyze_gigpo_groups_from_rollout.py \\
    --run-dir runs/agent_offload_pyrodash4b_docker_async_20260731_112705 \\
    --max-rollouts 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from examples.coding_agent_rl.gigpo_anchor import (
    branch_key_from_user_task,
    init_anchor_obs,
    make_anchor_obs,
)
from examples.coding_agent_rl.offload import group_aware_rewards
from slime.utils.gigpo import build_step_group

_IM_START = re.compile(r"<\|im_start\|>(\w+)\n")
_TOOL_CALL = re.compile(r"<tool_call>\n?(.*?)</tool_call>", re.DOTALL)
_TOOL_RESP = re.compile(r"<tool_response>\n?(.*?)</tool_response>", re.DOTALL)
_FUNCTION = re.compile(r"<function=([^\s>]+)>\s*(.*?)</function>", re.DOTALL)
_PARAM = re.compile(r"<parameter=([^\s>]+)>\n?(.*?)</parameter>", re.DOTALL)


def _parse_tool_calls(assistant_text: str) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for block in _TOOL_CALL.findall(assistant_text or ""):
        fm = _FUNCTION.search(block)
        if not fm:
            continue
        name = fm.group(1).strip()
        params = {pm.group(1).strip(): pm.group(2).strip() for pm in _PARAM.finditer(fm.group(2))}
        out.append((name, params))
    return out


def _split_messages(text: str) -> list[tuple[str, str]]:
    """Return [(role, content), ...] from a decoded chat transcript."""
    parts = _IM_START.split(text)
    # parts: [preamble, role1, body1, role2, body2, ...]
    msgs: list[tuple[str, str]] = []
    i = 1
    while i + 1 < len(parts):
        role = parts[i]
        body = parts[i + 1]
        if body.endswith("<|im_end|>\n"):
            body = body[: -len("<|im_end|>\n")]
        elif body.endswith("<|im_end|>"):
            body = body[: -len("<|im_end|>")]
        msgs.append((role, body))
        i += 2
    return msgs


def extract_turns_from_decoded(
    text: str,
    *,
    instance_id: str,
    problem_statement: str,
    workdir: str = "",
) -> list[dict[str, Any]]:
    """Rebuild per-assistant-turn rows with GiGPO anchors (mirrors trajectory.py)."""
    msgs = _split_messages(text)
    episode_user = problem_statement
    for role, body in msgs:
        if role != "user":
            continue
        if _TOOL_RESP.search(body):
            continue
        episode_user = body.strip() or episode_user
        break

    first_user = ""
    for role, body in msgs:
        if role == "user" and body.strip():
            first_user = body.strip()
            break
    bkey = branch_key_from_user_task(first_user, episode_user=episode_user or problem_statement)

    turns: list[dict[str, Any]] = []
    for idx, (role, body) in enumerate(msgs):
        if role != "assistant":
            continue

        prev_tool_name: str | None = None
        prev_tool_input: dict[str, Any] | None = None
        prev_tool_result: str | None = None
        for j in range(idx - 1, -1, -1):
            r, b = msgs[j]
            if r not in ("user", "tool"):
                continue
            resps = _TOOL_RESP.findall(b)
            if not resps:
                # Skip system-reminder / plain user pings; keep looking for tool obs.
                continue
            prev_tool_result = resps[-1]
            for k in range(j - 1, -1, -1):
                if msgs[k][0] == "assistant":
                    calls = _parse_tool_calls(msgs[k][1])
                    if calls:
                        # Prefer last tool_call when multiple were issued in one turn.
                        prev_tool_name, prev_tool_input = calls[-1]
                    break
            break

        # Init = no tool_response has appeared before this assistant turn.
        seen_tool_resp = any(
            _TOOL_RESP.search(msgs[t][1]) for t in range(idx) if msgs[t][0] in ("user", "tool")
        )
        is_init = not seen_tool_resp
        if is_init:
            anchor = init_anchor_obs(instance_id, episode_user or problem_statement, branch_key=bkey)
        elif prev_tool_name:
            anchor = make_anchor_obs(
                instance_id=instance_id,
                branch_key=bkey,
                tool_name=prev_tool_name,
                tool_input=prev_tool_input,
                tool_result_text=prev_tool_result,
                workdir=workdir,
                is_init=False,
                problem_statement=episode_user or problem_statement,
            )
        else:
            # Avoid make_anchor_obs(tool_name=None) → collapses into __init__.
            anchor = make_anchor_obs(
                instance_id=instance_id,
                branch_key=bkey,
                tool_name="UnknownObs",
                tool_input={},
                tool_result_text=prev_tool_result or "",
                workdir=workdir,
                is_init=False,
                problem_statement=episode_user or problem_statement,
            )

        calls = _parse_tool_calls(body)
        turns.append(
            {
                "turn_index": len(turns),
                "branch_key": bkey,
                "anchor_obs": anchor,
                "is_init": is_init,
                "n_tool_calls": len(calls),
                "tool_names": [c[0] for c in calls],
                "has_offload": ("llm_offload" in body) or ("<|llm_offload|>" in body),
                "assistant_chars": len(body),
            }
        )
    return turns


def _guess_workdir(instance_id: str, text: str) -> str:
    # Paths look like /workspace/<repo>/...
    m = re.search(r"(/workspace/[A-Za-z0-9_.-]+)/", text)
    if m:
        return m.group(1)
    # Fallback from instance_id owner_repo_prN
    parts = (instance_id or "").split("_")
    if len(parts) >= 2:
        return f"/workspace/{parts[1]}"
    return ""


def load_samples(path: Path) -> list[dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return list(data.get("samples") or [])


def analyze_run(
    run_dir: Path,
    *,
    hf_checkpoint: str,
    max_rollouts: int | None,
    rollout_ids: list[int] | None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True, local_files_only=True)
    dump_dir = run_dir / "rollout_dumps"
    paths = sorted(dump_dir.glob("rollout_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if rollout_ids is not None:
        want = set(rollout_ids)
        paths = [p for p in paths if int(p.stem.split("_")[1]) in want]
    if max_rollouts is not None:
        paths = paths[:max_rollouts]

    # Flattened step rows for GiGPO clustering: key = (rollout_id, group_index)
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    decode_errors = 0

    for path in paths:
        rid = int(path.stem.split("_")[1])
        samples = load_samples(path)
        for s in samples:
            md = s.get("metadata") or {}
            instance_id = str(md.get("instance_id") or s.get("label") or "")
            group_index = s.get("group_index")
            prompt = s.get("prompt") or []
            problem = ""
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
                problem = str(prompt[0].get("content") or "")
            elif isinstance(prompt, str):
                problem = prompt

            stats = md.get("offload_stats") or {}
            episode_rows.append(
                {
                    "rollout_id": rid,
                    "group_index": group_index,
                    "instance_id": instance_id,
                    "index": s.get("index"),
                    "reward": float(s.get("reward") or 0.0),
                    "solved": float(md.get("solved", 1.0 if md.get("grading_solved") else 0.0) or 0.0),
                    "empty_patch": bool(md.get("empty_patch", False)),
                    "offload_count": int(stats.get("offload_count") or 0),
                    "offload_outside_think_count": int(stats.get("offload_outside_think_count") or 0),
                    "response_length": int(s.get("response_length") or 0),
                    "n_tokens": len(s.get("tokens") or []),
                }
            )

            try:
                text = tok.decode(s.get("tokens") or [], skip_special_tokens=False)
            except Exception:
                decode_errors += 1
                continue
            workdir = _guess_workdir(instance_id, text)
            turns = extract_turns_from_decoded(
                text,
                instance_id=instance_id,
                problem_statement=problem,
                workdir=workdir,
            )
            for t in turns:
                step_rows.append(
                    {
                        "rollout_id": rid,
                        "group_index": group_index,
                        "instance_id": instance_id,
                        "sample_index": s.get("index"),
                        **t,
                    }
                )

    # ---- episode group_aware ----
    group_aware_reports: list[dict[str, Any]] = []
    unique_bonus_triggers = 0
    no_offload_bonus_sessions = 0
    all_fail_groups = 0
    by_ep: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for ep in episode_rows:
        by_ep[(ep["rollout_id"], ep["group_index"])].append(ep)
    for (rid, gidx), eps in sorted(by_ep.items()):
        items = [
            {
                "solved": e["solved"],
                "stats": {
                    "offload_count": e["offload_count"],
                    "offload_outside_think_count": e["offload_outside_think_count"],
                },
                "empty_patch": e["empty_patch"],
            }
            for e in eps
        ]
        report: dict[str, Any] = {}
        rewards = group_aware_rewards(items, report=report)
        if report.get("unique_bonus_applied"):
            unique_bonus_triggers += 1
        no_offload_bonus_sessions += int(report.get("no_offload_bonus_count") or 0)
        if report.get("all_fail"):
            all_fail_groups += 1
        group_aware_reports.append(
            {
                "rollout_id": rid,
                "group_index": gidx,
                "instance_id": eps[0]["instance_id"],
                "n": len(eps),
                "rewards_orig": [e["reward"] for e in eps],
                "rewards_shaped": rewards,
                **{k: report.get(k) for k in ("n_solved", "n_offload", "unique_bonus_applied", "no_offload_bonus_count", "all_fail")},
            }
        )

    # ---- step groups within (rollout_id, group_index) ----
    # Use composite index so different rollouts don't collide.
    anchors = [r["anchor_obs"] for r in step_rows]
    composite_index = [f"{r['rollout_id']}:{r['group_index']}" for r in step_rows]
    sizes_out: list[int] = []
    if step_rows:
        build_step_group(anchors, composite_index, sizes_out=sizes_out, summarize=False)

    size_hist = Counter(sizes_out)
    n_singleton = size_hist.get(1, 0)
    n_matched_groups = sum(c for sz, c in size_hist.items() if sz >= 2)
    n_matched_steps = sum(sz * c for sz, c in size_hist.items() if sz >= 2)

    # Tool-name breakdown among matched groups
    # Rebuild clusters for examples.
    clusters: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    for r in step_rows:
        clusters[(f"{r['rollout_id']}:{r['group_index']}", r["anchor_obs"])].append(r)

    matched_examples: list[dict[str, Any]] = []
    tool_of_matched: Counter[str] = Counter()
    init_matched = 0
    for (comp, anchor), rows in clusters.items():
        if len(rows) < 2:
            continue
        try:
            obs = json.loads(anchor)
            tool = (obs.get("obs") or ["?"])[0]
        except json.JSONDecodeError:
            tool = "?"
        tool_of_matched[str(tool)] += 1
        if rows[0].get("is_init"):
            init_matched += 1
        if len(matched_examples) < 15:
            matched_examples.append(
                {
                    "composite_group": comp,
                    "instance_id": rows[0]["instance_id"],
                    "size": len(rows),
                    "tool": tool,
                    "is_init": bool(rows[0].get("is_init")),
                    "sample_indices": sorted({r["sample_index"] for r in rows}),
                    "turn_indices": sorted({r["turn_index"] for r in rows}),
                    "branch_keys": sorted({r["branch_key"] for r in rows}),
                    "anchor_preview": anchor[:180],
                }
            )

    # Per episode-group: how many step-groups / match rate
    per_group_stats: list[dict[str, Any]] = []
    by_comp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in step_rows:
        by_comp[f"{r['rollout_id']}:{r['group_index']}"].append(r)
    for comp, rows in sorted(by_comp.items()):
        local_sizes: list[int] = []
        build_step_group(
            [r["anchor_obs"] for r in rows],
            [comp] * len(rows),
            sizes_out=local_sizes,
            summarize=False,
        )
        hist = Counter(local_sizes)
        n_steps = len(rows)
        n_sess = len({r["sample_index"] for r in rows})
        matched = sum(c for sz, c in hist.items() if sz >= 2)
        per_group_stats.append(
            {
                "composite": comp,
                "instance_id": rows[0]["instance_id"],
                "n_sessions": n_sess,
                "n_steps": n_steps,
                "avg_turns": n_steps / max(n_sess, 1),
                "n_step_groups": len(local_sizes),
                "n_matched_groups": matched,
                "match_step_frac": (sum(sz * c for sz, c in hist.items() if sz >= 2) / n_steps) if n_steps else 0.0,
                "hist": dict(sorted(hist.items())),
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "n_rollouts": len(paths),
        "n_episode_samples": len(episode_rows),
        "n_step_rows": len(step_rows),
        "decode_errors": decode_errors,
        "episode_group_aware": {
            "n_episode_groups": len(by_ep),
            "unique_bonus_triggers": unique_bonus_triggers,
            "no_offload_bonus_sessions": no_offload_bonus_sessions,
            "all_fail_groups": all_fail_groups,
            "solved_rate": (
                sum(1 for e in episode_rows if e["solved"] > 0) / len(episode_rows) if episode_rows else 0.0
            ),
            "offload_rate": (
                sum(1 for e in episode_rows if e["offload_count"] > 0) / len(episode_rows) if episode_rows else 0.0
            ),
        },
        "step_groups": {
            "n_step_groups": len(sizes_out),
            "avg_size": (sum(sizes_out) / len(sizes_out)) if sizes_out else 0.0,
            "max_size": max(sizes_out) if sizes_out else 0,
            "n_singleton": n_singleton,
            "n_matched_groups": n_matched_groups,
            "n_matched_steps": n_matched_steps,
            "match_step_frac": (n_matched_steps / len(step_rows)) if step_rows else 0.0,
            "hist": {str(k): v for k, v in sorted(size_hist.items())},
            "matched_tool_hist": dict(tool_of_matched.most_common()),
            "init_matched_groups": init_matched,
        },
        "matched_examples": matched_examples,
        "per_group_stats_top": sorted(per_group_stats, key=lambda x: -x["match_step_frac"])[:20],
        "per_group_stats_bottom": sorted(per_group_stats, key=lambda x: x["match_step_frac"])[:10],
        "group_aware_examples": [
            g
            for g in group_aware_reports
            if g.get("unique_bonus_applied") or g.get("no_offload_bonus_count") or (g.get("n_solved") or 0) > 0
        ][:20],
    }
    return summary


def _print_report(summary: dict[str, Any]) -> None:
    sg = summary["step_groups"]
    eg = summary["episode_group_aware"]
    print("=" * 70)
    print(f"run: {summary['run_dir']}")
    print(
        f"rollouts={summary['n_rollouts']} episodes={summary['n_episode_samples']} "
        f"step_rows={summary['n_step_rows']} decode_errors={summary['decode_errors']}"
    )
    print("-" * 70)
    print("Episode group_aware:")
    print(
        f"  groups={eg['n_episode_groups']} solved_rate={eg['solved_rate']:.2%} "
        f"offload_rate={eg['offload_rate']:.2%}"
    )
    print(
        f"  unique_bonus_triggers={eg['unique_bonus_triggers']} "
        f"no_offload_bonus_sessions={eg['no_offload_bonus_sessions']} "
        f"all_fail_groups={eg['all_fail_groups']}"
    )
    print("-" * 70)
    print("GiGPO step-groups (within rollout×group_index):")
    hist_str = " ".join(f"{k}:{v}" for k, v in sg["hist"].items())
    print(
        f"  n={sg['n_step_groups']} avg={sg['avg_size']:.2f} max={sg['max_size']} "
        f"singleton={sg['n_singleton']} matched_groups={sg['n_matched_groups']} "
        f"match_step_frac={sg['match_step_frac']:.2%}"
    )
    print(f"  hist[size:count] {hist_str or '-'}")
    print(f"  matched_tool_hist={sg['matched_tool_hist']}")
    print(f"  init_matched_groups={sg['init_matched_groups']}")
    print("-" * 70)
    print("Top matched examples:")
    for ex in summary["matched_examples"][:8]:
        print(
            f"  size={ex['size']} tool={ex['tool']} init={ex['is_init']} "
            f"inst={ex['instance_id']} samples={ex['sample_indices']} turns={ex['turn_indices']}"
        )
    print("-" * 70)
    print("Best episode-groups by match_step_frac:")
    for g in summary["per_group_stats_top"][:5]:
        print(
            f"  {g['composite']} {g['instance_id']} sess={g['n_sessions']} steps={g['n_steps']} "
            f"match={g['match_step_frac']:.0%} hist={g['hist']}"
        )
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/agent_offload_pyrodash4b_docker_async_20260731_112705"),
    )
    p.add_argument(
        "--hf-checkpoint",
        default="/workspace/models/pyromind/PyroDash-4B-SFT-07313",
    )
    p.add_argument("--max-rollouts", type=int, default=None, help="Only first N rollout_*.pt")
    p.add_argument("--rollout-ids", type=int, nargs="*", default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON report (default: <run-dir>/gigpo_group_analysis.json)",
    )
    args = p.parse_args(argv)

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        # Resolve relative to repo root (two levels up from this file).
        repo = Path(__file__).resolve().parents[2]
        cand = repo / run_dir
        run_dir = cand if cand.exists() else Path.cwd() / run_dir
    if not (run_dir / "rollout_dumps").exists():
        print(f"ERROR: missing rollout_dumps under {run_dir}", file=sys.stderr)
        return 1

    summary = analyze_run(
        run_dir,
        hf_checkpoint=args.hf_checkpoint,
        max_rollouts=args.max_rollouts,
        rollout_ids=args.rollout_ids,
    )
    _print_report(summary)
    # Drop bulky helpers before write? keep matched examples.
    out = args.out or (run_dir / "gigpo_group_analysis.json")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
