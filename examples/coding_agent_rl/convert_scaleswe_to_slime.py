#!/usr/bin/env python3
"""Convert AweAI-Team/Scale-SWE jsonl into slime coding-agent prompt data.

Scale-SWE fields -> slime Sample row (metadata.protocol=scaleswe):

  prompt / label / metadata.{protocol,agent,image,workdir,problem_statement,pre_commands}
  metadata.remote_env_info.f2p_script  (required by swe._metadata_scaleswe)

Example:
  python examples/coding_agent_rl/convert_scaleswe_to_slime.py \\
    --src /path/to/processed_to_upload.jsonl \\
    --dst /path/to/swe_train.jsonl \\
    --limit 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.coding_agent_rl.agents_registry import resolve_agent


def convert_row(row: dict, *, default_agent: str = "claude_code") -> dict | None:
    image = row.get("image_url") or row.get("image")
    workdir = row.get("workdir")
    f2p_script = row.get("f2p_script")
    problem = row.get("problem_statement") or ""
    instance_id = row.get("instance_id") or "unknown"
    if not image or not workdir or not f2p_script:
        return None
    pre_commands = row.get("pre_commands")
    agent = resolve_agent(row.get("agent") or default_agent).name
    # Qwen3.5 HF checkpoints ship a VLM processor; Dataset requires conversation
    # prompts (list[dict]) whenever a processor loads successfully.
    return {
        "prompt": [{"role": "user", "content": problem}],
        "label": instance_id,
        "metadata": {
            "protocol": "scaleswe",
            "agent": agent,
            "instance_id": instance_id,
            "image": image,
            "workdir": workdir,
            "problem_statement": problem,
            "pre_commands": pre_commands,
            "remote_env_info": {
                "instance_id": instance_id,
                "image_url": image,
                "workdir": workdir,
                "pre_commands": pre_commands,
                "f2p_script": f2p_script,
            },
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="Scale-SWE processed_to_upload.jsonl")
    p.add_argument("--dst", type=Path, required=True, help="Output slime jsonl")
    p.add_argument("--limit", type=int, default=0, help="Keep at most N rows (0 = all)")
    p.add_argument(
        "--default-agent",
        default="claude_code",
        help="metadata.agent when source row has no agent (default: claude_code)",
    )
    args = p.parse_args()

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with args.src.open() as fin, args.dst.open("w") as fout:
        for line in fin:
            if args.limit and kept >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            out = convert_row(json.loads(line), default_agent=args.default_agent)
            if out is None:
                skipped += 1
                continue
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    print(f"wrote {kept} rows -> {args.dst} (skipped {skipped})")


if __name__ == "__main__":
    main()
