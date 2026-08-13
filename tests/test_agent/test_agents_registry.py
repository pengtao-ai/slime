"""Unit tests for coding_agent_rl agents_registry + metadata.agent routing."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl.agents_registry import (  # noqa: E402
    CANONICAL_AGENTS,
    normalize_agent_name,
    resolve_agent,
)
from examples.coding_agent_rl import swe  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0

_DATA = REPO_ROOT / "examples" / "coding_agent_rl" / "data"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("claude_code", "claude_code"),
        ("cc", "claude_code"),
        ("codex", "codex"),
        ("pi", "pi"),
        ("opencode", "opencode"),
        ("miniswe", "miniswe"),
        ("mini-swe-agent", "miniswe"),
        ("mini_swe", "miniswe"),
    ],
)
def test_resolve_agent_aliases(raw, expected):
    spec = resolve_agent(raw)
    assert spec.name == expected
    assert spec.harness_cls().name == expected
    if expected == "codex":
        assert spec.adapter_protocol == "openai"
    else:
        assert spec.adapter_protocol == "anthropic"


def test_resolve_agent_default_from_env(monkeypatch):
    monkeypatch.delenv("SWE_AGENT", raising=False)
    assert resolve_agent(None).name == "claude_code"
    monkeypatch.setenv("SWE_AGENT", "codex")
    assert resolve_agent(None).name == "codex"
    assert normalize_agent_name(None) == "codex"


def test_resolve_agent_unknown():
    with pytest.raises(ValueError, match="unknown agent"):
        resolve_agent("not-an-agent")


def test_get_metadata_passes_agent():
    sample = Sample(
        prompt=[{"role": "user", "content": "fix it"}],
        label="inst1",
        metadata={
            "protocol": "scaleswe",
            "agent": "cc",
            "instance_id": "inst1",
            "image": "img:tag",
            "workdir": "/workspace/repo",
            "remote_env_info": {"f2p_script": "print(1)"},
        },
    )
    md = swe.get_metadata(sample, swe.PROTOCOL_SCALESWE)
    assert md["agent"] == "claude_code"


@pytest.mark.parametrize(
    "filename,protocol",
    [
        ("scaleswe_agents_smoke.jsonl", "scaleswe"),
        ("tmax_agents_smoke.jsonl", "tmax"),
    ],
)
def test_agents_smoke_jsonl(filename, protocol):
    path = _DATA / filename
    if not path.is_file():
        pytest.skip(f"missing {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == len(CANONICAL_AGENTS) * 2
    counts = Counter((r.get("metadata") or {}).get("agent") for r in rows)
    assert dict(counts) == {a: 2 for a in CANONICAL_AGENTS}
    for r in rows:
        assert (r.get("metadata") or {}).get("protocol") == protocol


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
