#!/usr/bin/env python3
"""Compute per-trajectory prompt/output token lengths from infer run dirs.

For each ``*/trajectory.json``, sum ``openai_response.usage.prompt_tokens`` and
``completion_tokens`` across turns (API-billed totals). Also report max prompt
tokens (peak context) and turn count.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _usage_of(turn: dict) -> dict:
    resp = turn.get("openai_response") or {}
    usage = resp.get("usage") if isinstance(resp, dict) else None
    return usage if isinstance(usage, dict) else {}


def summarize_trajectory(path: Path) -> dict | None:
    with path.open() as f:
        data = json.load(f)

    turns = data.get("turns") or []
    if not turns:
        return None

    prompt_sum = 0
    output_sum = 0
    reasoning_sum = 0
    max_prompt = 0
    turns_with_usage = 0

    for turn in turns:
        usage = _usage_of(turn)
        if not usage:
            continue
        turns_with_usage += 1
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        reasoning = int(usage.get("reasoning_tokens") or 0)
        prompt_sum += prompt
        output_sum += output
        reasoning_sum += reasoning
        if prompt > max_prompt:
            max_prompt = prompt

    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    return {
        "name": path.parent.name,
        "instance_id": summary.get("instance_id") or path.parent.name,
        "turns": len(turns),
        "turns_with_usage": turns_with_usage,
        "prompt_tokens_sum": prompt_sum,
        "output_tokens_sum": output_sum,
        "reasoning_tokens_sum": reasoning_sum,
        "max_prompt_tokens": max_prompt,
        "total_tokens_sum": prompt_sum + output_sum,
    }


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Infer run directory containing i*/trajectory.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write per-trajectory CSV",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary averages (no per-traj table)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    missing_traj = 0
    empty_traj = 0

    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        traj = child / "trajectory.json"
        if not traj.exists():
            missing_traj += 1
            continue
        row = summarize_trajectory(traj)
        if row is None:
            empty_traj += 1
            continue
        rows.append(row)

    if not rows:
        print("no trajectories with turns found", file=sys.stderr)
        return 1

    if not args.quiet:
        header = (
            f"{'name':<55} {'turns':>5} {'prompt_sum':>12} {'output_sum':>12} "
            f"{'max_prompt':>12} {'total':>12}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['name']:<55} {r['turns']:>5} {r['prompt_tokens_sum']:>12} "
                f"{r['output_tokens_sum']:>12} {r['max_prompt_tokens']:>12} "
                f"{r['total_tokens_sum']:>12}"
            )
        print()

    n = len(rows)
    avg_prompt = mean([r["prompt_tokens_sum"] for r in rows])
    avg_output = mean([r["output_tokens_sum"] for r in rows])
    avg_max_prompt = mean([r["max_prompt_tokens"] for r in rows])
    avg_total = mean([r["total_tokens_sum"] for r in rows])
    avg_turns = mean([r["turns"] for r in rows])
    avg_reasoning = mean([r["reasoning_tokens_sum"] for r in rows])

    print(f"run_dir:              {run_dir}")
    print(f"trajectories:         {n}")
    print(f"dirs_missing_traj:    {missing_traj}")
    print(f"empty_trajectories:   {empty_traj}")
    print(f"avg_turns:            {avg_turns:.2f}")
    print(f"avg_prompt_tokens:    {avg_prompt:.1f}   (sum over turns)")
    print(f"avg_output_tokens:    {avg_output:.1f}   (sum over turns)")
    print(f"avg_reasoning_tokens: {avg_reasoning:.1f}   (sum over turns)")
    print(f"avg_max_prompt:       {avg_max_prompt:.1f}   (peak context)")
    print(f"avg_total_tokens:     {avg_total:.1f}   (prompt+output sum)")

    if args.csv:
        fieldnames = [
            "name",
            "instance_id",
            "turns",
            "turns_with_usage",
            "prompt_tokens_sum",
            "output_tokens_sum",
            "reasoning_tokens_sum",
            "max_prompt_tokens",
            "total_tokens_sum",
        ]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote csv:            {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
