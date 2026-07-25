#!/usr/bin/env python3
"""Convert phase2 offload GRPO jsonl (problem/answer/usage/think) into slime prompt data.

Input row (pyroDash phase2)::

    {"problem", "answer", "system"?, "usage"?, "think"?, ...}

Output row (slime)::

    {
      "prompt": [{"role":"system",...},{"role":"user",...}],
      "label": "<answer>",
      "metadata": {"usage": ..., "think": ..., "messages": <same as prompt>}
    }

Example::

  python examples/llm_offload/convert_offload_dataset.py \\
    --src /workspace/work/spt/pyroDash-training/data/phase2/glm52_hint_8b_answers.jsonl \\
    --dst examples/llm_offload/data/offload_grpo_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SYSTEM = (
    "You are a helpful assistant. Solve the given problem step by step. "
    "For very difficult steps, you can output <|llm_offload|> to request help "
    "from a more capable model."
)


def convert_row(row: dict) -> dict | None:
    user = row.get("problem") or row.get("question")
    answer = row.get("answer")
    if user is None or answer is None:
        return None
    messages = [
        {"role": "system", "content": row.get("system") or DEFAULT_SYSTEM},
        {"role": "user", "content": str(user)},
    ]
    metadata: dict = {"messages": messages}
    if row.get("usage") is not None:
        metadata["usage"] = row["usage"]
    if row.get("think") is not None:
        metadata["think"] = row["think"]
    if row.get("id") is not None:
        metadata["id"] = row["id"]
    return {
        "prompt": messages,
        "label": str(answer),
        "metadata": metadata,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = p.parse_args()

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with args.src.open() as fin, args.dst.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            out = convert_row(row)
            if out is None:
                continue
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
            if args.limit and n_out >= args.limit:
                break
    print(f"converted {n_out}/{n_in} -> {args.dst}")


if __name__ == "__main__":
    main()
