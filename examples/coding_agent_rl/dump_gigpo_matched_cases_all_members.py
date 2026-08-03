#!/usr/bin/env python3
"""Dump GiGPO matched step-groups with ALL members, for inspection.

Reads a rollout dump, decodes tokens, rebuilds per-turn anchors via
``gigpo_anchor`` (mirroring training path), clusters identical anchors
within each ``group_index``, and writes JSON cases with every member.

Usage:
  PYTHONPATH=. python examples/coding_agent_rl/dump_gigpo_matched_cases_all_members.py \\
    --rollout runs/.../rollout_dumps/rollout_0.pt \\
    --out runs/.../gigpo_matched_cases_all.json \\
    --max-cases 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from examples.coding_agent_rl.analyze_gigpo_groups_from_rollout import (
    _guess_workdir,
    _parse_tool_calls,
    _split_messages,
    _TOOL_RESP,
    extract_turns_from_decoded,
)


def _load_samples(path: Path) -> list[dict[str, Any]]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return list(d.get("samples") or [])


def build_rows(samples: list[dict[str, Any]], *, tok) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in samples:
        md = s.get("metadata") or {}
        instance_id = str(md.get("instance_id") or s.get("label") or "")
        prompt = s.get("prompt") or []
        problem = ""
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            problem = str(prompt[0].get("content") or "")
        elif isinstance(prompt, str):
            problem = prompt
        text = tok.decode(s.get("tokens") or [], skip_special_tokens=False)
        workdir = _guess_workdir(instance_id, text)
        turns = extract_turns_from_decoded(
            text, instance_id=instance_id, problem_statement=problem, workdir=workdir
        )
        msgs = _split_messages(text)
        asst_idxs = [i for i, (r, _) in enumerate(msgs) if r == "assistant"]
        stats = md.get("offload_stats") or {}
        for t in turns:
            ai = asst_idxs[t["turn_index"]]
            prev_tool_name = None
            prev_tool_input = None
            prev_tool_result = None
            for j in range(ai - 1, -1, -1):
                r, b = msgs[j]
                if r not in ("user", "tool"):
                    continue
                resps = _TOOL_RESP.findall(b)
                if not resps:
                    continue
                prev_tool_result = resps[-1]
                for k in range(j - 1, -1, -1):
                    if msgs[k][0] == "assistant":
                        calls = _parse_tool_calls(msgs[k][1])
                        if calls:
                            prev_tool_name, prev_tool_input = calls[-1]
                        break
                break
            this_calls = _parse_tool_calls(msgs[ai][1])
            rows.append(
                {
                    "sample_index": s.get("index"),
                    "turn_index": t["turn_index"],
                    "anchor_obs": t["anchor_obs"],
                    "is_init": t["is_init"],
                    "branch_key": t["branch_key"],
                    "prev_tool_call": (
                        {"name": prev_tool_name, "input": prev_tool_input}
                        if prev_tool_name
                        else None
                    ),
                    "prev_tool_response_preview": (prev_tool_result or "")[:400],
                    "this_turn_tool_calls": [
                        {"name": n, "input": inp} for n, inp in this_calls[:4]
                    ],
                    "assistant_preview": msgs[ai][1][:280].replace("\n", " "),
                    "reward": float(s.get("reward") or 0.0),
                    "solved": float(md.get("solved", 1.0 if md.get("grading_solved") else 0.0) or 0.0),
                    "offload_count": int(stats.get("offload_count") or 0),
                    "response_length": int(s.get("response_length") or 0),
                    "group_index": s.get("group_index"),
                    "instance_id": instance_id,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout", type=Path, required=True, help="Path to rollout_N.pt")
    p.add_argument("--out", type=Path, required=True, help="Output JSON path")
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument(
        "--min-size", type=int, default=2, help="Minimum group size to include (default 2)"
    )
    p.add_argument(
        "--min-distinct-samples",
        type=int,
        default=2,
        help="Require members from at least N distinct samples (default 2)",
    )
    p.add_argument(
        "--hf-checkpoint",
        default="/workspace/models/pyromind/PyroDash-4B-SFT-07313",
    )
    p.add_argument(
        "--exclude-init",
        action="store_true",
        help="Skip __init__ groups (usually trivial 1-per-sample)",
    )
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.hf_checkpoint, trust_remote_code=True, local_files_only=True
    )

    rollout_path = args.rollout
    data = torch.load(rollout_path, map_location="cpu", weights_only=False)
    rollout_id = int(data.get("rollout_id") or rollout_path.stem.split("_")[1])
    samples = list(data.get("samples") or [])
    rows = build_rows(samples, tok=tok)

    # Cluster by (group_index, anchor_obs).
    by_cluster: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cluster[(r["group_index"], r["anchor_obs"])].append(r)

    cases: list[dict[str, Any]] = []
    # Sort by size desc so the biggest matched groups come first.
    for (gidx, anchor), members in sorted(by_cluster.items(), key=lambda kv: -len(kv[1])):
        size = len(members)
        if size < args.min_size:
            continue
        distinct_samples = {m["sample_index"] for m in members}
        if len(distinct_samples) < args.min_distinct_samples:
            continue
        try:
            obs = json.loads(anchor)
            tool = (obs.get("obs") or ["?"])[0]
        except json.JSONDecodeError:
            tool = "?"
        if args.exclude_init and tool == "__init__":
            continue
        instance_id = members[0].get("instance_id") or ""
        # Sort members by (sample_index, turn_index) for readability.
        members_sorted = sorted(members, key=lambda m: (m["sample_index"], m["turn_index"]))
        cases.append(
            {
                "case_id": len(cases) + 1,
                "rollout_id": rollout_id,
                "group_index": gidx,
                "instance_id": instance_id,
                "tool": tool,
                "group_size": size,
                "n_distinct_samples": len(distinct_samples),
                "anchor_obs": anchor,
                "why_grouped": "Same group_index + identical anchor_obs (exact match).",
                "members": members_sorted,
            }
        )
        if len(cases) >= args.max_cases:
            break

    out_obj = {
        "source_rollout": str(rollout_path),
        "rollout_id": rollout_id,
        "n_samples": len(samples),
        "n_step_rows": len(rows),
        "n_cases": len(cases),
        "note": (
            "ALL members included. Anchors mirror the offline analyzer "
            "(skips system-reminder; for multi-tool turns takes the last tool_call "
            "of the preceding assistant and the last tool_response). Training path "
            "takes the first tool_call only."
        ),
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n")
    print(
        f"wrote {args.out}: {len(cases)} cases "
        f"(rollout {rollout_id}, {len(rows)} step rows, {len(samples)} samples)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
