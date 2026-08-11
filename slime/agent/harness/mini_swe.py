"""mini-swe-agent harness.

Runs ``mini`` inside the existing DockerSandbox (LocalEnvironment), pointing
litellm's Anthropic provider at the host Anthropic adapter. Trajectories are
recorded by the adapter (same dumps as OpenCode / pi), not by mini's native
SWE-bench host-side runner.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from slime.agent.sandbox import Sandbox, exec_and_wait

from .common import BaseHarness, HarnessContext, run_agent

PIP_INSTALL_RETRIES = 3
PIP_INSTALL_BACKOFF_SEC = 2.0


class MiniSweHarness(BaseHarness):
    name = "miniswe"

    wheel_env = "SLIME_AGENT_MINISWE_WHEEL"
    extra_args_env = "SLIME_AGENT_MINISWE_EXTRA_ARGS"
    extra_envs_env = "SLIME_AGENT_MINISWE_EXTRA_ENVS"

    config_path = "/home/agent/.config/mini-swe-agent/slime.yaml"

    async def install_cli(self, sb: Sandbox) -> None:
        import asyncio
        import shutil
        import tarfile
        import tempfile

        wheel = Path(os.environ[self.wheel_env])
        if not wheel.exists():
            raise FileNotFoundError(f"{self.wheel_env} missing: {wheel}")

        # Upload a directory of wheels (deps + mini-swe-agent) or a single wheel.
        # Prefer one tarball copy — per-wheel docker cp is prohibitively slow.
        if wheel.is_dir():
            whls = sorted(wheel.glob("*.whl"))
            if not whls:
                raise FileNotFoundError(f"no *.whl under {wheel}")
            cached_tar = wheel.parent / "miniswe-wheels.tar"
            newest = max(w.stat().st_mtime for w in whls)
            if not (cached_tar.exists() and cached_tar.stat().st_mtime >= newest):
                with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    with tarfile.open(tmp_path, "w") as tar:
                        for whl in whls:
                            tar.add(whl, arcname=whl.name)
                    shutil.move(str(tmp_path), cached_tar)
                finally:
                    tmp_path.unlink(missing_ok=True)
            await sb.write_file("/tmp/miniswe-wheels.tar", cached_tar)
            # Prefer the SWE-Smith testbed interpreter when present so install and
            # agent-user launch share the same prefix; always symlink into
            # /usr/local/bin for a stable PATH entry.
            install_cmd = (
                "bash -lc '"
                "set -euo pipefail; "
                "if [ -x /opt/miniconda3/envs/testbed/bin/python3 ]; then "
                "  PY=/opt/miniconda3/envs/testbed/bin/python3; "
                "else "
                "  PY=python3; "
                "fi; "
                "rm -rf /tmp/miniswe-wheels && mkdir -p /tmp/miniswe-wheels; "
                "tar xf /tmp/miniswe-wheels.tar -C /tmp/miniswe-wheels; "
                "echo miniswe_wheel_count=$(ls -1 /tmp/miniswe-wheels | wc -l); "
                "ls /tmp/miniswe-wheels | grep -i pyyaml; "
                "\"$PY\" -m pip install --no-cache-dir --no-index "
                "--find-links=file:///tmp/miniswe-wheels mini-swe-agent; "
                "MINI=$(\"$PY\" -c \"import shutil; print(shutil.which(\\\"mini\\\"))\"); "
                "test -n \"$MINI\" && test -x \"$MINI\"; "
                "ln -sfn \"$MINI\" /usr/local/bin/mini; "
                "/usr/local/bin/mini --help >/dev/null"
                "'"
            )
        else:
            await sb.write_file("/tmp/miniswe.whl", wheel)
            install_cmd = (
                "python3 -m pip install --no-deps /tmp/miniswe.whl 2>/dev/null || "
                "python3 -m pip install /tmp/miniswe.whl; "
                "mini --help >/dev/null"
            )

        last_log = ""
        exit_code = 1
        for attempt in range(PIP_INSTALL_RETRIES):
            exit_code, last_log = await exec_and_wait(
                sb,
                cmd=install_cmd,
                user="root",
                time_budget_sec=600,
                tag="miniswe-pip-install",
                want_output=True,
            )
            if exit_code == 0:
                return
            if attempt + 1 < PIP_INSTALL_RETRIES:
                await asyncio.sleep(PIP_INSTALL_BACKOFF_SEC * (attempt + 1))
        raise RuntimeError(
            f"pip install mini-swe-agent failed after {PIP_INSTALL_RETRIES} attempts "
            f"(exit={exit_code}):\n{last_log[-4000:]}"
        )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        """Write mini YAML: Anthropic via litellm → slime adapter, yolo local env."""
        # litellm anthropic provider requests {api_base}/v1/messages, matching
        # Claude Code (api_base = adapter root, no trailing /v1).
        config = {
            "agent": {
                "system_template": (
                    "You are a helpful assistant that can interact with a computer "
                    "to solve software engineering tasks using bash."
                ),
                "instance_template": (
                    "Please solve this issue: {{task}}\n\n"
                    "You can execute bash commands and edit files. "
                    "Submit by running: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                ),
                "mode": "yolo",
                "confirm_exit": False,
                "step_limit": 0,
                "cost_limit": 0,
            },
            "environment": {
                "environment_class": "local",
                "cwd": ctx.workdir,
                "env": {
                    "PAGER": "cat",
                    "MANPAGER": "cat",
                    "LESS": "-R",
                    "PIP_PROGRESS_BAR": "off",
                    "TQDM_DISABLE": "1",
                },
            },
            "model": {
                "model_name": f"anthropic/{ctx.model_label}",
                "cost_tracking": "ignore_errors",
                "model_kwargs": {
                    "api_base": ctx.adapter_url,
                    "api_key": ctx.session_id,
                    "drop_params": True,
                },
            },
            "run": {
                "confirm_exit": False,
            },
        }
        # Prefer YAML via a tiny emitter (no PyYAML dependency in the harness).
        yaml_text = _dict_to_simple_yaml(config)
        await sb.exec(
            "mkdir -p /home/agent/.config/mini-swe-agent && "
            "chown -R agent:agent /home/agent/.config",
            user="root",
            check=True,
            timeout=60,
        )
        await sb.write_file(self.config_path, yaml_text, user="agent")

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        cmd = (
            f"/usr/local/bin/mini -y --exit-immediately "
            f"-c {shlex.quote(self.config_path)} "
            f"-t {shlex.quote(prompt)}"
        )
        extra = os.environ.get(self.extra_args_env, "").strip()
        if extra:
            cmd = f"{cmd} {extra}"
        env = {
            "ANTHROPIC_API_KEY": ctx.session_id,
            "ANTHROPIC_AUTH_TOKEN": ctx.session_id,
            "ANTHROPIC_BASE_URL": ctx.adapter_url,
            "MSWEA_COST_TRACKING": "ignore_errors",
            "MSWEA_CONFIGURED": "true",
        }
        extra_envs = os.environ.get(self.extra_envs_env, "").strip()
        if extra_envs:
            env.update(json.loads(extra_envs))
        return await run_agent(sb, workdir=ctx.workdir, start_cmd=cmd, env=env, time_budget_sec=time_budget_sec)


def _dict_to_simple_yaml(obj, indent: int = 0) -> str:
    """Minimal YAML emitter for nested dict/list/scalars (no anchors)."""
    sp = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return sp + "{}\n"
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{k}:\n{_dict_to_simple_yaml(v, indent + 1)}")
            else:
                lines.append(f"{sp}{k}: {_yaml_scalar(v)}\n")
        return "".join(lines)
    if isinstance(obj, list):
        if not obj:
            return sp + "[]\n"
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{sp}-\n{_dict_to_simple_yaml(item, indent + 1)}")
            else:
                lines.append(f"{sp}- {_yaml_scalar(item)}\n")
        return "".join(lines)
    return f"{sp}{_yaml_scalar(obj)}\n"


def _yaml_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    # Quote anything YAML could reinterpret (bool/null/numbers) or that needs escaping.
    lowered = s.lower()
    if (
        s == ""
        or lowered in {"true", "false", "yes", "no", "on", "off", "null", "~"}
        or any(c in s for c in ":#{}[]&*!|>%@`\"'\n")
        or s.strip() != s
        or s[0] in "-?,"
        or s.replace(".", "", 1).isdigit()
    ):
        return json.dumps(s)
    return s
