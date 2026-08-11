#!/usr/bin/env python3
"""Codex inference via OpenAIAdapter -> SGLang, with OpenCode-style traj dumps."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_EX = Path(__file__).resolve().parent
sys.path[:0] = [str(_REPO), str(_EX)]

from infer_openai_harness import add_common_infer_args, main_async  # noqa: E402
from slime.agent.harness import CodexHarness  # noqa: E402


def _setup_env(args: argparse.Namespace) -> None:
    os.environ["SLIME_AGENT_NODE_TARBALL"] = str(args.node_tarball)
    os.environ["SLIME_AGENT_CODEX_TARBALL"] = str(args.codex_tarball)


def main() -> None:
    ex = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    add_common_infer_args(p, default_bind_port=18094)
    p.add_argument("--node-tarball", type=Path, default=ex / "tarballs/node-v22.20.0-linux-x64.tar.xz")
    p.add_argument("--codex-tarball", type=Path, default=ex / "tarballs/openai-codex-local.tgz")
    args = p.parse_args()
    asyncio.run(
        main_async(
            args,
            harness=CodexHarness(),
            agent_name="codex",
            mode="codex_sglang",
            sid_prefix="infer-cdx",
            thread_name="infer-cdx-adapter",
            setup_env=_setup_env,
        )
    )


if __name__ == "__main__":
    main()
