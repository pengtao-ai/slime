#!/usr/bin/env python3
"""mini-swe-agent inference via AnthropicAdapter -> SGLang, with OpenCode-style traj dumps."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_EX = Path(__file__).resolve().parent
sys.path[:0] = [str(_REPO), str(_EX)]

from infer_anthropic_harness import add_common_infer_args, main_async  # noqa: E402
from slime.agent.harness import MiniSweHarness  # noqa: E402


def _setup_env(args: argparse.Namespace) -> None:
    os.environ["SLIME_AGENT_MINISWE_WHEEL"] = str(args.miniswe_wheel)


def main() -> None:
    ex = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    add_common_infer_args(p, default_bind_port=18093)
    p.add_argument(
        "--miniswe-wheel",
        type=Path,
        default=ex / "tarballs/miniswe-wheels",
        help="Directory of wheels (from build_miniswe_wheels.py) or a single .whl",
    )
    args = p.parse_args()
    asyncio.run(
        main_async(
            args,
            harness=MiniSweHarness(),
            agent_name="miniswe",
            mode="miniswe_sglang",
            sid_prefix="infer-ms",
            thread_name="infer-ms-adapter",
            setup_env=_setup_env,
        )
    )


if __name__ == "__main__":
    main()
