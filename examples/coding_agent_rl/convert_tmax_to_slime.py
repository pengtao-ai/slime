#!/usr/bin/env python3
"""Convert allenai/tmax-15k-open-instruct into slime coding-agent prompt data.

Produces rows with metadata.protocol=tmax for mixed ScaleSWE+Tmax training:

  prompt / label / metadata.{protocol,agent,image,workdir,problem_statement,test_sh}

Example:
  python examples/coding_agent_rl/convert_tmax_to_slime.py \\
    --dst examples/coding_agent_rl/data/tmax_train_smoke.jsonl \\
    --limit 50

  # reuse an already-extracted task-data tree:
  python examples/coding_agent_rl/convert_tmax_to_slime.py \\
    --task-data-dir /path/to/task-data.tar.gz.extracted \\
    --dst /tmp/tmax.jsonl --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from contextlib import suppress
from pathlib import Path

from examples.coding_agent_rl.agents_registry import resolve_agent

# Strip vanillux-only harness rules that confuse Claude Code.
_VANILLUX_TAIL_MARKERS = (
    "\n## Recommended Workflow\n",
    "\n## Important Rules\n",
    "\nYou can execute bash commands and edit files",
)


def _strip_vanillux_harness(user_text: str) -> str:
    text = user_text.strip()
    if text.startswith("Please solve this task:\n\n"):
        text = text[len("Please solve this task:\n\n") :]
    cut = len(text)
    for marker in _VANILLUX_TAIL_MARKERS:
        i = text.find(marker)
        if i >= 0:
            cut = min(cut, i)
    return text[:cut].strip()


def _user_problem(messages) -> str:
    if not messages:
        return ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return _strip_vanillux_harness(str(m.get("content") or ""))
    return ""


def resolve_task_data_dir(task_data_dir: str | None, hf_repo: str) -> Path:
    if task_data_dir:
        p = Path(task_data_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"--task-data-dir not a directory: {p}")
        return p

    from huggingface_hub import snapshot_download

    repo_dir = Path(snapshot_download(hf_repo, repo_type="dataset"))
    tarball = repo_dir / "task-data.tar.gz"
    if not tarball.is_file():
        raise FileNotFoundError(f"task-data.tar.gz missing under {repo_dir}")

    extract_dir = Path(str(tarball) + ".extracted")
    complete = extract_dir / ".extraction_complete"
    lock_dir = Path(str(extract_dir) + ".lock")
    while not complete.is_file():
        try:
            os.mkdir(lock_dir)
        except FileExistsError:
            time.sleep(1)
            continue
        try:
            if complete.is_file():
                break
            print(f"Extracting {tarball} -> {extract_dir} ...", flush=True)
            if extract_dir.is_dir():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["tar", "-xzf", str(tarball), "-C", str(extract_dir)], check=True)
            complete.write_text("ok\n", encoding="utf-8")
        finally:
            with suppress(FileNotFoundError, OSError):
                lock_dir.rmdir()
    return extract_dir


def _read_test_sh(task_data: Path, task_id: str) -> str | None:
    path = task_data / task_id / "tests" / "test.sh"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    # Some extractions nest one extra directory level.
    matches = list(task_data.glob(f"**/task_*/tests/test.sh"))
    for m in matches:
        if m.parent.parent.name == task_id:
            return m.read_text(encoding="utf-8")
    return None


def convert_row(row: dict, *, task_data: Path, default_agent: str = "claude_code") -> dict | None:
    env = row.get("env_config") or {}
    task_id = (
        (env.get("task_id") if isinstance(env, dict) else None)
        or row.get("ground_truth")
        or row.get("task_id")
        or ""
    )
    task_id = str(task_id).strip()
    image = (env.get("image") if isinstance(env, dict) else None) or row.get("image")
    if not task_id or not image:
        return None
    test_sh = _read_test_sh(task_data, task_id)
    if not test_sh:
        return None
    problem = _user_problem(row.get("messages"))
    if not problem:
        # Fallback: instruction.md from task-data
        instr = task_data / task_id / "instruction.md"
        if instr.is_file():
            problem = instr.read_text(encoding="utf-8").strip()
    if not problem:
        return None
    agent = resolve_agent(row.get("agent") or default_agent).name
    return {
        "prompt": [{"role": "user", "content": problem}],
        "label": task_id,
        "metadata": {
            "protocol": "tmax",
            "agent": agent,
            "instance_id": task_id,
            "image": image,
            "workdir": "/home/user",
            "problem_statement": problem,
            "test_sh": test_sh,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dst", type=Path, required=True, help="Output slime jsonl")
    p.add_argument("--hf-repo", default="allenai/tmax-15k-open-instruct")
    p.add_argument("--task-data-dir", default=None, help="Pre-extracted task-data directory")
    p.add_argument("--limit", type=int, default=0, help="Keep at most N rows (0 = all)")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--default-agent",
        default="claude_code",
        help="metadata.agent when source row has no agent (default: claude_code)",
    )
    args = p.parse_args()

    from datasets import load_dataset

    task_data = resolve_task_data_dir(args.task_data_dir, args.hf_repo)
    ds = load_dataset(args.hf_repo, split=args.split)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with args.dst.open("w", encoding="utf-8") as fout:
        for row in ds:
            if args.limit and kept >= args.limit:
                break
            out = convert_row(dict(row), task_data=task_data, default_agent=args.default_agent)
            if out is None:
                skipped += 1
                continue
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    print(f"wrote {kept} rows -> {args.dst} (skipped {skipped}; task_data={task_data})")


if __name__ == "__main__":
    main()
