#!/usr/bin/env python3
"""Build multi-agent smoke jsonl from existing scaleswe / tmax rows.

Writes 5 agents x 2 rows each (= 10) per protocol:

  examples/coding_agent_rl/data/scaleswe_agents_smoke.jsonl
  examples/coding_agent_rl/data/tmax_agents_smoke.jsonl

Each output row keeps the source grading fields and sets metadata.agent.
If the source has fewer than 10 distinct rows, rows are reused with a new agent.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from examples.coding_agent_rl.agents_registry import CANONICAL_AGENTS

_DATA = Path(__file__).resolve().parent / "data"
_DEFAULT_SCALESWE_SRC = _DATA / "swe_train_scaleswe_200_baked.jsonl"
_DEFAULT_TMAX_SRC = _DATA / "tmax_train_200.jsonl"
_PER_AGENT = 2


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"empty source: {path}")
    return rows


def _with_agent(row: dict, agent: str, *, protocol: str | None = None) -> dict:
    out = copy.deepcopy(row)
    md = out.setdefault("metadata", {})
    if not isinstance(md, dict):
        raise TypeError(f"metadata must be dict, got {type(md)}")
    md["agent"] = agent
    if protocol is not None:
        md["protocol"] = protocol
    return out


def build_agents_smoke(
    src_rows: list[dict],
    *,
    per_agent: int = _PER_AGENT,
    protocol: str | None = None,
) -> list[dict]:
    out: list[dict] = []
    i = 0
    n = len(src_rows)
    for agent in CANONICAL_AGENTS:
        for _ in range(per_agent):
            out.append(_with_agent(src_rows[i % n], agent, protocol=protocol))
            i += 1
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate(rows: list[dict], *, protocol: str | None) -> None:
    assert len(rows) == len(CANONICAL_AGENTS) * _PER_AGENT
    counts = Counter((r.get("metadata") or {}).get("agent") for r in rows)
    assert dict(counts) == {a: _PER_AGENT for a in CANONICAL_AGENTS}, counts
    if protocol is not None:
        for r in rows:
            assert (r.get("metadata") or {}).get("protocol") == protocol


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scaleswe-src", type=Path, default=_DEFAULT_SCALESWE_SRC)
    p.add_argument("--tmax-src", type=Path, default=_DEFAULT_TMAX_SRC)
    p.add_argument("--scaleswe-dst", type=Path, default=_DATA / "scaleswe_agents_smoke.jsonl")
    p.add_argument("--tmax-dst", type=Path, default=_DATA / "tmax_agents_smoke.jsonl")
    args = p.parse_args()

    scaleswe = build_agents_smoke(_load_jsonl(args.scaleswe_src), protocol="scaleswe")
    _validate(scaleswe, protocol="scaleswe")
    _write_jsonl(args.scaleswe_dst, scaleswe)

    tmax_src = args.tmax_src
    if not tmax_src.is_file():
        tmax_src = _DATA / "tmax_smoke_3.jsonl"
    tmax = build_agents_smoke(_load_jsonl(tmax_src), protocol="tmax")
    _validate(tmax, protocol="tmax")
    _write_jsonl(args.tmax_dst, tmax)

    print(f"wrote {len(scaleswe)} -> {args.scaleswe_dst}")
    print(f"wrote {len(tmax)} -> {args.tmax_dst}")


if __name__ == "__main__":
    main()
