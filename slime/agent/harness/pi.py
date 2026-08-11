"""Pi coding-agent harness (badlogic / earendil pi CLI).

Non-interactive entrypoint is ``pi -p``. Provider traffic is pointed at the host
Anthropic adapter via ``~/.pi/agent/models.json`` (api: anthropic-messages) and
``ANTHROPIC_*`` env vars. Custom model ids must be registered under the anthropic
provider or pi rejects them.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from slime.agent.sandbox import Sandbox

from .common import BaseHarness, HarnessContext, install_npm_cli, run_agent


class PiHarness(BaseHarness):
    name = "pi"

    node_tarball_env = "SLIME_AGENT_NODE_TARBALL"
    cli_tarball_env = "SLIME_AGENT_PI_TARBALL"
    extra_args_env = "SLIME_AGENT_PI_EXTRA_ARGS"
    extra_envs_env = "SLIME_AGENT_PI_EXTRA_ENVS"

    # print mode + json event stream for harness_trajectory capture
    launch_flags = "-p --mode json --thinking off"

    static_env = {
        "PI_SKIP_VERSION_CHECK": "1",
    }

    async def install_cli(self, sb: Sandbox) -> None:
        await install_npm_cli(
            sb,
            node_runtime=Path(os.environ[self.node_tarball_env]),
            npm_package=Path(os.environ[self.cli_tarball_env]),
            check_cmd="ls -la /usr/local/bin/pi && /usr/local/bin/pi --version",
        )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        """Point anthropic provider at slime adapter; register custom model."""
        # pi anthropic-messages appends /v1/messages itself; slime serves
        # /v1/messages on the adapter root. Passing .../v1 yields /v1/v1/messages 404
        # (unlike OpenCode, which only appends /messages).
        models = {
            "providers": {
                "anthropic": {
                    "baseUrl": ctx.adapter_url,
                    "api": "anthropic-messages",
                    "apiKey": ctx.session_id,
                    "models": [
                        {
                            "id": ctx.model_label,
                            "name": ctx.model_label,
                            "reasoning": False,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 128000,
                            "maxTokens": 32768,
                        }
                    ],
                }
            }
        }
        payload = json.dumps(models, separators=(",", ":"))
        await sb.exec(
            "mkdir -p /home/agent/.pi/agent && "
            f"echo {shlex.quote(payload)} > /home/agent/.pi/agent/models.json && "
            "chown -R agent:agent /home/agent/.pi",
            user="root",
            check=True,
            timeout=60,
        )

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        cmd = (
            f"/usr/local/bin/pi {self.launch_flags} "
            f"--provider anthropic --model {shlex.quote(ctx.model_label)} "
            f"--api-key {shlex.quote(ctx.session_id)} "
            f"{shlex.quote(prompt)}"
        )
        extra = os.environ.get(self.extra_args_env, "").strip()
        if extra:
            cmd = f"{cmd} {extra}"
        env = {
            "ANTHROPIC_API_KEY": ctx.session_id,
            "ANTHROPIC_AUTH_TOKEN": ctx.session_id,
            "ANTHROPIC_BASE_URL": ctx.adapter_url,
            "PI_CODING_AGENT_DIR": "/home/agent/.pi/agent",
            **self.static_env,
        }
        extra_envs = os.environ.get(self.extra_envs_env, "").strip()
        if extra_envs:
            env.update(json.loads(extra_envs))
        return await run_agent(sb, workdir=ctx.workdir, start_cmd=cmd, env=env, time_budget_sec=time_budget_sec)
