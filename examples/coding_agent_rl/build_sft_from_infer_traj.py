#!/usr/bin/env python3
"""Build SFT jsonl from infer_cc_offload_traj run dirs.

For each sample that has ``summary.json``, take the last ``requests/req_*.json``,
append the model ``response`` as a final assistant message, and emit one jsonl
row keeping only ``turn_index``, ``sid``, ``messages``, ``tools``.

Example::

    python examples/coding_agent_rl/build_sft_from_infer_traj.py \\
      --run-dir runs/infer_cc_glm_20260728_141118
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_REQ_RE = re.compile(r"^req_(\d+)\.json$")


def _response_to_assistant(response: dict[str, Any] | None) -> dict[str, Any]:
    resp = response or {}
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": resp.get("content") or "",
    }
    reasoning = resp.get("reasoning_content")
    if reasoning:
        msg["reasoning_content"] = reasoning
    tool_calls = resp.get("tool_calls") or []
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _last_request_path(sample_dir: Path) -> Path | None:
    req_dir = sample_dir / "requests"
    if not req_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for path in req_dir.iterdir():
        m = _REQ_RE.match(path.name)
        if not m or not path.is_file():
            continue
        idx = int(m.group(1))
        if best is None or idx > best[0]:
            best = (idx, path)
    return best[1] if best else None


def _sample_to_row(sample_dir: Path) -> dict[str, Any] | None:
    if not (sample_dir / "summary.json").is_file():
        return None
    req_path = _last_request_path(sample_dir)
    if req_path is None:
        return None
    try:
        payload = json.loads(req_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[skip] {sample_dir.name}: bad request json ({exc})", file=sys.stderr)
        return None
    messages = list(payload.get("messages") or [])
    messages.append(_response_to_assistant(payload.get("response")))
    return {
        "turn_index": payload.get("turn_index"),
        "sid": payload.get("sid"),
        "messages": messages,
        "tools": payload.get("tools") or [],
    }


def build(run_dir: Path, out_path: Path) -> tuple[int, int]:
    sample_dirs = sorted(
        p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("i")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for sample_dir in sample_dirs:
            row = _sample_to_row(sample_dir)
            if row is None:
                n_skip += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1
    return n_ok, n_skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/infer_cc_glm_20260728_141118"),
        help="Infer run directory containing iXXX_* sample dirs",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output jsonl path (default: <run-dir>/sft_last_turn.jsonl)",
    )
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir not found: {run_dir}")
    out_path = (args.out or (run_dir / "sft_last_turn.jsonl")).resolve()
    n_ok, n_skip = build(run_dir, out_path)
    print(f"[sft] wrote {n_ok} rows -> {out_path} (skipped={n_skip})", flush=True)
    if n_ok == 0:
        raise SystemExit("FAIL: no samples with summary.json + last request")


if __name__ == "__main__":
    main()
