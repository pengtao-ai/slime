#!/usr/bin/env python3
"""Lightweight Pi ↔ AnthropicAdapter ↔ local SGLang smoke (no Ray / no train.sh).

Prereq — start SGLang, then build the offline pi tarball once::

    python examples/coding_agent_rl/build_pi_local_tarball.py
    python examples/coding_agent_rl/smoke_pi_docker.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EXAMPLE_DIR))

import smoke_claude_code_docker as smoke  # noqa: E402
import swe  # noqa: E402
from slime.agent.adapters import AnthropicAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.harness import PiHarness
from slime.agent.sandbox import DockerSandbox, ensure_agent_user
from slime.utils.processing_utils import load_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smoke_pi")

DEFAULT_JSONL = smoke.DEFAULT_JSONL
DEFAULT_NODE = smoke.DEFAULT_NODE
DEFAULT_PI = _EXAMPLE_DIR / "tarballs" / "pi-coding-agent-local.tgz"
DEFAULT_HF = smoke.DEFAULT_HF
DEFAULT_SGLANG = smoke.DEFAULT_SGLANG
DEFAULT_PROMPT = (
    "Read PROBLEM_STATEMENT.md. Reply with a one-line investigation plan only, "
    "then stop. Do not edit files in this smoke run."
)


def _setup_docker_env(*, network: str) -> None:
    os.environ["SLIME_AGENT_SANDBOX_BACKEND"] = "docker"
    os.environ["SLIME_AGENT_DOCKER_NETWORK"] = network
    if network == "host":
        os.environ.pop("SLIME_AGENT_DOCKER_ADD_HOST", None)
    else:
        os.environ["SLIME_AGENT_DOCKER_ADD_HOST"] = os.environ.get(
            "SLIME_AGENT_DOCKER_ADD_HOST", "host.docker.internal:host-gateway"
        )


async def run_smoke(args: argparse.Namespace) -> None:
    _setup_docker_env(network=args.network)
    os.environ["SLIME_AGENT_NODE_TARBALL"] = str(args.node_tarball)
    os.environ["SLIME_AGENT_PI_TARBALL"] = str(args.pi_tarball)
    os.environ["SWE_AGENT"] = "pi"

    smoke._check_sglang(args.sglang_url)

    print(f"[smoke] loading tokenizer {args.hf_checkpoint}", flush=True)
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    adapter = AnthropicAdapter(
        tokenizer=tokenizer,
        sglang_url=args.sglang_url.rstrip("/"),
        tool_parser="qwen3_coder",
        reasoning_parser="qwen3",
    )
    handle = run_app_in_thread(
        adapter.app,
        host=args.bind_host,
        port=args.bind_port,
        thread_name="smoke-pi-adapter",
        runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
    )
    print(
        f"[smoke] anthropic adapter listening {args.bind_host}:{handle.port}  sglang={args.sglang_url}",
        flush=True,
    )

    sample = smoke._load_sample(args.jsonl)
    md = swe.get_metadata(sample, swe.PROTOCOL_SCALESWE)
    image = args.image or md["image"]
    workdir = md["workdir"]
    instance_id = md["instance_id"]
    session_id = f"smoke-pi-{secrets.token_hex(8)}"

    adapter.open_session(
        session_id,
        sampling_defaults={"temperature": 0.7, "max_new_tokens": args.max_new_tokens},
        max_context_tokens=args.max_context_len,
    )

    agent_exit_code = -999
    segments: list = []
    diag = ""
    try:
        print(f"[smoke] docker image={image} network={args.network}", flush=True)
        if args.pull:
            os.environ["SLIME_AGENT_DOCKER_PULL"] = "1"
        async with DockerSandbox(image) as sb:
            print(f"[smoke] sandbox_id={sb.sandbox_id}", flush=True)
            await ensure_agent_user(sb, workdir)

            try:
                adapter_host = await smoke._pick_adapter_host(
                    sb, port=handle.port, preferred=(args.public_host or None)
                )
            except RuntimeError as exc:
                raise SystemExit(f"FAIL: {exc}") from exc
            adapter_url = f"http://{adapter_host}:{handle.port}"
            print(f"[smoke] adapter_url will be {adapter_url} (pi baseUrl=adapter root)", flush=True)

            harness = PiHarness()
            print("[smoke] install_cli (node + pi)...", flush=True)
            await harness.install_cli(sb)

            print("[smoke] prepare_workspace...", flush=True)
            await swe.prepare_workspace(sb, workdir, md)
            await sb.write_file(
                f"{workdir}/PROBLEM_STATEMENT.md",
                "# Smoke\n\nReply with a one-line plan and stop. Do not edit files.\n",
                user="agent",
            )

            print(
                f"[smoke] pi.run budget={args.time_budget}s prompt={args.prompt[:80]!r}...",
                flush=True,
            )
            agent_exit_code = await harness.run(
                sb,
                workdir=workdir,
                session_id=session_id,
                adapter_url=adapter_url,
                time_budget_sec=args.time_budget,
                prompt=args.prompt,
            )
            print(f"[smoke] agent_exit_code={agent_exit_code}", flush=True)

            _, diag, _ = await sb.exec(
                "echo '=== launcher head ==='; head -n 25 /tmp/.run.sh 2>/dev/null; "
                "echo '=== pi models.json ==='; "
                "cat /home/agent/.pi/agent/models.json 2>/dev/null || true; "
                f"echo '=== trajectory tail ==='; "
                f"tail -c 1500 {workdir}/.harness/trajectory.jsonl 2>/dev/null || true",
                timeout=30,
            )
            print(f"[smoke] diag:\n{(diag or '')[:4000]}", flush=True)

            reward = 0.0

        segments = await adapter.finish_session(
            session_id,
            base_sample=sample,
            reward=float(reward),
            extra_metadata={"instance_id": instance_id, "smoke": True, "agent": "pi"},
        )
    finally:
        await adapter.drop_session(session_id, wait_timeout=10)
        handle.stop()

    print(
        f"[smoke] segments={len(segments)} agent_exit_code={agent_exit_code} "
        f"response_lens={[getattr(s, 'response_length', None) for s in segments]}",
        flush=True,
    )
    if not segments:
        raise SystemExit(
            "FAIL: adapter_session_empty — pi never completed a turn "
            "through AnthropicAdapter→SGLang. Check diag above."
        )
    if args.require_exit_zero and agent_exit_code != 0:
        raise SystemExit(f"FAIL: agent_exit_code={agent_exit_code}")
    print("[smoke] PASS", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    p.add_argument("--image", default=None)
    p.add_argument("--pull", action="store_true")
    p.add_argument("--node-tarball", type=Path, default=Path(os.environ.get("SLIME_AGENT_NODE_TARBALL", DEFAULT_NODE)))
    p.add_argument(
        "--pi-tarball",
        type=Path,
        default=Path(os.environ.get("SLIME_AGENT_PI_TARBALL", DEFAULT_PI)),
    )
    p.add_argument("--hf-checkpoint", default=os.environ.get("HF_CHECKPOINT", DEFAULT_HF))
    p.add_argument("--sglang-url", default=os.environ.get("SMOKE_SGLANG_URL", DEFAULT_SGLANG))
    p.add_argument("--bind-host", default=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"))
    p.add_argument("--bind-port", type=int, default=int(os.environ.get("ADAPTER_PORT", "18092")))
    p.add_argument("--public-host", default=os.environ.get("ADAPTER_PUBLIC_HOST", ""))
    p.add_argument("--network", default=os.environ.get("SLIME_AGENT_DOCKER_NETWORK", "bridge"))
    p.add_argument("--time-budget", type=int, default=int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "180")))
    p.add_argument("--max-context-len", type=int, default=int(os.environ.get("SMOKE_MAX_CONTEXT_LEN", "96000")))
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "1024")))
    p.add_argument("--prompt", default=os.environ.get("SWE_PI_PROMPT", DEFAULT_PROMPT))
    p.add_argument("--require-exit-zero", action="store_true")
    args = p.parse_args()

    smoke._require_file(args.node_tarball, "SLIME_AGENT_NODE_TARBALL")
    smoke._require_file(args.pi_tarball, "SLIME_AGENT_PI_TARBALL")
    smoke._require_file(args.jsonl, "jsonl")
    if not Path(args.hf_checkpoint).exists():
        raise SystemExit(f"missing hf checkpoint: {args.hf_checkpoint}")

    asyncio.run(run_smoke(args))


if __name__ == "__main__":
    main()
