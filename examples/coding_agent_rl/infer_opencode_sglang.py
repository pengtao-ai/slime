#!/usr/bin/env python3
"""OpenCode inference via AnthropicAdapter -> SGLang /generate, with traj dumps.

Per-sample outputs match ``opencode_smith_infer5_v3`` layout (see traj_dump.py).
"""
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
from slime.agent.harness import OpenCodeHarness  # noqa: E402


def _setup_env(args: argparse.Namespace) -> None:
    os.environ["SLIME_AGENT_NODE_TARBALL"] = str(args.node_tarball)
    os.environ["SLIME_AGENT_OPENCODE_TARBALL"] = str(args.opencode_tarball)


def main() -> None:
    ex = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    add_common_infer_args(p, default_bind_port=18001)
    p.add_argument("--node-tarball", type=Path, default=ex / "tarballs/node-v22.20.0-linux-x64.tar.xz")
    p.add_argument("--opencode-tarball", type=Path, default=ex / "tarballs/opencode-ai-local-linux-x64.tgz")
    args = p.parse_args()
    asyncio.run(
        main_async(
            args,
            harness=OpenCodeHarness(),
            agent_name="opencode",
            mode="opencode_sglang",
            sid_prefix="infer-oc",
            thread_name="infer-oc-adapter",
            setup_env=_setup_env,
        )
    )


if __name__ == "__main__":
    main()
