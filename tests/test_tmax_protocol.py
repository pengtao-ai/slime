"""Unit tests for tmax protocol metadata / evaluability / inplace grading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import examples.coding_agent_rl.swe as swe  # noqa: E402
from slime.utils.types import Sample  # noqa: E402


def _tmax_sample(**overrides) -> Sample:
    md = {
        "protocol": "tmax",
        "instance_id": "task_000000_c19dda5b",
        "image": "hamishi740/swerl-tmax-v3:deadbeef",
        "workdir": "/home/user",
        "problem_statement": "Fix the parser.",
        "test_sh": "#!/bin/bash\nexit 0\n",
    }
    md.update(overrides)
    return Sample(
        prompt=[{"role": "user", "content": md["problem_statement"]}],
        label=md["instance_id"],
        metadata=md,
    )


def test_metadata_tmax_fields():
    md = swe.get_metadata(_tmax_sample(), swe.PROTOCOL_TMAX)
    assert md["protocol"] == swe.PROTOCOL_TMAX
    assert md["instance_id"] == "task_000000_c19dda5b"
    assert md["image"].startswith("hamishi740/")
    assert md["workdir"] == "/home/user"
    assert "Fix the parser" in md["problem_statement"]
    assert "exit 0" in md["grading"]["test_sh"]


def test_metadata_tmax_defaults_workdir_from_env_config():
    s = Sample(
        prompt=[{"role": "user", "content": "do it"}],
        label="task_x",
        metadata={
            "protocol": "tmax",
            "env_config": {"image": "img:tag", "task_id": "task_x"},
            "test_sh": "exit 0\n",
        },
    )
    md = swe.get_metadata(s, swe.PROTOCOL_TMAX)
    assert md["image"] == "img:tag"
    assert md["workdir"] == "/home/user"
    assert md["instance_id"] == "task_x"


def test_evaluability_tmax_requires_test_sh():
    md = swe.get_metadata(_tmax_sample(test_sh=""), swe.PROTOCOL_TMAX)
    assert swe.evaluability_check(md) == "missing_test_sh"
    md_ok = swe.get_metadata(_tmax_sample(), swe.PROTOCOL_TMAX)
    assert swe.evaluability_check(md_ok) is None


def test_scaleswe_without_protocol_field_still_scaleswe():
    """Legacy jsonl rows omit metadata.protocol; env fallback uses scaleswe."""
    s = Sample(
        prompt=[{"role": "user", "content": "issue"}],
        label="legacy-1",
        metadata={
            "instance_id": "legacy-1",
            "image": "scaleswe:img",
            "workdir": "/workspace/repo",
            "problem_statement": "issue",
            "remote_env_info": {"f2p_script": "import sys\nsys.exit(0)\n"},
        },
    )
    md = swe.get_metadata(s, swe.PROTOCOL_SCALESWE)
    assert md["protocol"] == swe.PROTOCOL_SCALESWE
    assert swe.evaluability_check(md) is None


def test_grade_tmax_inplace_exit_codes():
    import asyncio

    class FakeSB:
        def __init__(self, exit_code: int, reward_txt: str | None = None):
            self.exit_code = exit_code
            self.reward_txt = reward_txt
            self.written = {}

        async def exec(self, cmd, **kwargs):
            if cmd.startswith("bash "):
                return self.exit_code, "ok", ""
            return 0, "", ""

        async def write_file(self, path, content, **kwargs):
            self.written[path] = content

        async def read_file(self, path, **kwargs):
            if path.endswith("reward.txt") and self.reward_txt is not None:
                return self.reward_txt
            raise FileNotFoundError(path)

    async def run_case():
        md = swe.get_metadata(_tmax_sample(), swe.PROTOCOL_TMAX)
        ok = await swe.grade_tmax_inplace(FakeSB(0), md, timeout_sec=30)
        assert ok.reward == 1.0
        bad = await swe.grade_tmax_inplace(FakeSB(1), md, timeout_sec=30)
        assert bad.reward == 0.0
        from_file = await swe.grade_tmax_inplace(FakeSB(1, reward_txt="1.0\n"), md, timeout_sec=30)
        assert from_file.reward == 1.0

    asyncio.run(run_case())


def test_run_evaluation_rejects_tmax():
    import asyncio

    async def run_case():
        md = swe.get_metadata(_tmax_sample(), swe.PROTOCOL_TMAX)
        with pytest.raises(ValueError, match="grade_tmax_inplace"):
            await swe.run_evaluation(md, diff_text="x", timeout_sec=10)

    asyncio.run(run_case())


def test_convert_tmax_strip_vanillux():
    from examples.coding_agent_rl.convert_tmax_to_slime import _strip_vanillux_harness

    raw = (
        "Please solve this task:\n\n"
        "Fix the bug in /home/user/app.\n\n"
        "## Recommended Workflow\n\n"
        "1. Analyze\n"
        "## Important Rules\n\n"
        "1. Every response must contain exactly one tool call\n"
    )
    out = _strip_vanillux_harness(raw)
    assert "Fix the bug" in out
    assert "Recommended Workflow" not in out
    assert "tool call" not in out
