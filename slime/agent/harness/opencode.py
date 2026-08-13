"""OpenCode harness.

Non-interactive entrypoint is ``opencode run``. Provider traffic is pointed at
the host Anthropic adapter (same protocol as Claude Code). OpenCode's Anthropic
SDK path expects ``baseURL`` to *include* ``/v1`` (unlike Claude Code's
``ANTHROPIC_BASE_URL``), so write_config appends it.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from slime.agent.sandbox import Sandbox

from .common import BaseHarness, HarnessContext, install_npm_cli, run_agent


class OpenCodeHarness(BaseHarness):
    name = "opencode"

    # host paths + CLI knobs, all under the agent-layer SLIME_AGENT_* prefix
    node_tarball_env = "SLIME_AGENT_NODE_TARBALL"
    cli_tarball_env = "SLIME_AGENT_OPENCODE_TARBALL"
    extra_args_env = "SLIME_AGENT_OPENCODE_EXTRA_ARGS"
    extra_envs_env = "SLIME_AGENT_OPENCODE_EXTRA_ENVS"

    # ``--auto`` auto-approves permissions that are not explicitly denied
    launch_flags = "--auto --format json"

    static_env = {
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
        # Headless: allow all tools (mirrors Claude Code bypassPermissions).
        "OPENCODE_PERMISSION": json.dumps({"*": "allow"}),
    }

    async def install_cli(self, sb: Sandbox) -> None:
        # Pre-baked images already ship opencode under /usr/local; skip upload.
        ec, _, _ = await sb.exec(
            "test -x /usr/local/bin/opencode && /usr/local/bin/opencode --version",
            user="root",
            check=False,
            timeout=60,
        )
        if ec == 0:
            return
        await install_npm_cli(
            sb,
            node_runtime=Path(os.environ[self.node_tarball_env]),
            npm_package=Path(os.environ[self.cli_tarball_env]),
            check_cmd="ls -la /usr/local/bin/opencode && /usr/local/bin/opencode --version",
        )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        """Point the Anthropic provider at the slime adapter; allow all tools."""
        # OpenCode appends /messages to baseURL; slime serves /v1/messages, so
        # baseURL must end with /v1 (Claude Code's env var does not need this).
        #
        # OpenCode validates provider/model against a local catalog. Custom labels
        # like slime-actor are rejected with ProviderModelNotFoundError unless
        # registered under provider.anthropic.models. small_model keeps title
        # generation on the same local model (otherwise it defaults to a Claude
        # haiku id that is also absent from the catalog).
        model_id = f"anthropic/{ctx.model_label}"
        config = {
            "$schema": "https://opencode.ai/config.json",
            "model": model_id,
            "small_model": model_id,
            "provider": {
                "anthropic": {
                    "options": {
                        "baseURL": f"{ctx.adapter_url}/v1",
                        "apiKey": ctx.session_id,
                    },
                    "models": {
                        ctx.model_label: {
                            "name": ctx.model_label,
                            "limit": {"context": 128000, "output": 32768},
                        }
                    },
                }
            },
            "permission": {"*": "allow"},
        }
        payload = json.dumps(config, separators=(",", ":"))
        await sb.exec(
            "mkdir -p /home/agent/.config/opencode && "
            f"echo {shlex.quote(payload)} > /home/agent/.config/opencode/opencode.json && "
            "chown -R agent:agent /home/agent/.config",
            user="root",
            check=True,
            timeout=60,
        )

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        model = f"anthropic/{ctx.model_label}"
        cmd = (
            f"/usr/local/bin/opencode run {self.launch_flags} "
            f"-m {shlex.quote(model)} {shlex.quote(prompt)}"
        )
        extra = os.environ.get(self.extra_args_env, "").strip()
        if extra:
            cmd = f"{cmd} {extra}"
        env = {
            # Belt-and-suspenders with opencode.json apiKey: Anthropic clients
            # send this as Authorization / X-Api-Key; adapter resolves sid from it.
            "ANTHROPIC_API_KEY": ctx.session_id,
            "ANTHROPIC_AUTH_TOKEN": ctx.session_id,
            "ANTHROPIC_BASE_URL": f"{ctx.adapter_url}/v1",
            **self.static_env,
        }
        extra_envs = os.environ.get(self.extra_envs_env, "").strip()
        if extra_envs:
            env.update(json.loads(extra_envs))
        return await run_agent(sb, workdir=ctx.workdir, start_cmd=cmd, env=env, time_budget_sec=time_budget_sec)
