#!/usr/bin/env python3
"""Smoke-test public e2b.dev with one ScaleSWE Docker image.

Requires ``E2B_API_KEY`` in the environment (real key from https://e2b.dev).

Steps:
  1. ``Template.from_image(docker_image)`` → ``Template.build(name=alias)``
  2. Boot ``AsyncSandbox.create(template=alias)`` via ``E2BSandbox``
  3. Assert ``workdir`` exists (ScaleSWE ships the repo under /workspace/...)

Example::

    export E2B_API_KEY=e2b_...
    python examples/coding_agent_rl/smoke_public_e2b.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# slime repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_DOCKER_IMAGE = "aweaiteam/scaleswe:arviz-devs_preliz_pr249"
DEFAULT_WORKDIR = "/workspace/preliz"
DEFAULT_TEMPLATE = "scaleswe-preliz-pr249"


def _require_api_key() -> None:
    if not os.environ.get("E2B_API_KEY"):
        raise SystemExit(
            "E2B_API_KEY is not set. Export a real key from https://e2b.dev "
            "before running this smoke test."
        )


def build_template(docker_image: str, template_name: str, *, memory_mb: int, cpu_count: int) -> None:
    from e2b import Template

    print(f"[smoke] building template {template_name!r} from {docker_image!r} ...", flush=True)

    def on_log(entry) -> None:
        # Keep logs short; never dump secrets.
        msg = getattr(entry, "message", None) or str(entry)
        print(f"[build] {msg}", flush=True)

    tpl = Template().from_image(docker_image)
    info = Template.build(
        tpl,
        name=template_name,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        on_build_logs=on_log,
    )
    print(f"[smoke] build done: {info}", flush=True)


async def probe(template_name: str, workdir: str) -> None:
    os.environ["SLIME_AGENT_E2B_USE_TEMPLATE"] = "1"
    from slime.agent.sandbox import E2BSandbox

    print(f"[smoke] creating sandbox template={template_name!r} ...", flush=True)
    async with E2BSandbox(template_name, timeout=600) as sb:
        print(f"[smoke] sandbox_id={sb.sandbox_id}", flush=True)
        code, out, err = await sb.exec(f"ls -la /workspace 2>&1 | head -40", timeout=60)
        print(f"[smoke] ls /workspace (exit={code}):\n{out}{err}", flush=True)
        code, out, err = await sb.exec(
            f"test -d {workdir} && echo WORKDIR_OK || echo WORKDIR_MISSING",
            timeout=30,
        )
        status = (out or err or "").strip()
        print(f"[smoke] workdir check: {status}", flush=True)
        if "WORKDIR_OK" not in status:
            raise SystemExit(
                f"workdir {workdir!r} missing after template boot — "
                "image may be non-Debian or build did not include the repo."
            )
    print("[smoke] PASS", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    p.add_argument("--template", default=DEFAULT_TEMPLATE, help="E2B template name/alias")
    p.add_argument("--skip-build", action="store_true", help="Reuse an already-built template")
    p.add_argument("--memory-mb", type=int, default=8192)
    p.add_argument("--cpu-count", type=int, default=4)
    args = p.parse_args()

    _require_api_key()
    if not args.skip_build:
        build_template(args.docker_image, args.template, memory_mb=args.memory_mb, cpu_count=args.cpu_count)
    asyncio.run(probe(args.template, args.workdir))


if __name__ == "__main__":
    main()
