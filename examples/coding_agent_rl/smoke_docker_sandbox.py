#!/usr/bin/env python3
"""Smoke-test the local Docker :class:`~slime.agent.sandbox.DockerSandbox`.

Requires a working ``docker`` CLI. Does **not** need E2B.

Default image is ``alpine:3.19`` (pull if missing). For ScaleSWE-style checks::

    python examples/coding_agent_rl/smoke_docker_sandbox.py \\
        --image aweaiteam/scaleswe:arviz-devs_preliz_pr249 \\
        --workdir /workspace/preliz

Env (optional)::

    SLIME_AGENT_DOCKER_NETWORK=bridge|host
    SLIME_AGENT_DOCKER_PULL=1
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slime.agent.sandbox import DockerSandbox, ensure_agent_user


async def smoke(image: str, workdir: str | None, *, pull: bool) -> None:
    os.environ["SLIME_AGENT_SANDBOX_BACKEND"] = "docker"
    print(f"[smoke] image={image!r} pull={pull}", flush=True)
    async with DockerSandbox(image, pull=pull) as sb:
        print(f"[smoke] sandbox_id={sb.sandbox_id}", flush=True)

        code, out, err = await sb.exec("uname -a && id", timeout=30)
        print(f"[smoke] uname/id exit={code}\n{out}{err}", flush=True)
        if code != 0:
            raise SystemExit(f"basic exec failed: {err or out}")

        await sb.write_file("/tmp/slime_docker_smoke.txt", "hello-from-host\n")
        text = await sb.read_file("/tmp/slime_docker_smoke.txt")
        if text.strip() != "hello-from-host":
            raise SystemExit(f"read_file mismatch: {text!r}")
        print("[smoke] write_file/read_file OK", flush=True)

        if workdir:
            code, out, err = await sb.exec(
                f"test -d {workdir} && echo WORKDIR_OK || echo WORKDIR_MISSING; "
                f"ls -la $(dirname {workdir}) 2>&1 | head -20",
                timeout=30,
            )
            print(f"[smoke] workdir check:\n{out}{err}", flush=True)
            if "WORKDIR_OK" not in (out or ""):
                raise SystemExit(f"workdir {workdir!r} missing in image")
            await ensure_agent_user(sb, workdir)
            code, out, _ = await sb.exec("id agent", timeout=30)
            print(f"[smoke] ensure_agent_user OK: {(out or '').strip()}", flush=True)

    print("[smoke] PASS", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", default="alpine:3.19")
    p.add_argument("--workdir", default=None, help="Optional path that must exist in the image")
    p.add_argument("--pull", action="store_true", help="docker pull before run")
    args = p.parse_args()
    asyncio.run(smoke(args.image, args.workdir, pull=args.pull))


if __name__ == "__main__":
    main()
