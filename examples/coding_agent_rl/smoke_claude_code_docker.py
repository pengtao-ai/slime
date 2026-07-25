#!/usr/bin/env python3
"""Lightweight Claude Code ↔ AnthropicAdapter ↔ local SGLang smoke (no Ray / no train.sh).

Mirrors the training path in ``examples.coding_agent_rl.generate``:

    DockerSandbox
    -> ClaudeCodeHarness.install_cli
    -> swe.prepare_workspace
    -> AnthropicAdapter on the host (sandbox auto-probes a reachable address)
    -> ClaudeCodeHarness.run
    -> adapter.finish_session  (must yield >=1 segment)

On docker-rt in k8s, ``--network host`` often joins the *node* netns, so
sandbox ``127.0.0.1`` is not the pod adapter (ConnectionRefused). Default is
``bridge`` + probe ``host.docker.internal`` / pod IP.

Prereq — start SGLang yourself (one GPU is enough)::

    CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \\
      --model-path /workspace/models/Qwen/Qwen3.5-4B \\
      --served-model-name qwen \\
      --host 127.0.0.1 --port 30000

Then::

    python examples/coding_agent_rl/smoke_claude_code_docker.py

Optional knobs (env or flags)::

    --sglang-url http://127.0.0.1:30000
    --hf-checkpoint /workspace/models/Qwen/Qwen3.5-4B
    --time-budget 180
    --eval          # also run scaleswe grading (slower)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EXAMPLE_DIR))

import swe  # noqa: E402
from slime.agent.adapters import AnthropicAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.harness import ClaudeCodeHarness
from slime.agent.sandbox import DockerSandbox
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smoke_claude_code")

DEFAULT_JSONL = _EXAMPLE_DIR / "data" / "swe_smoke_preliz_docker.jsonl"
DEFAULT_NODE = _EXAMPLE_DIR / "tarballs" / "node-v22.20.0-linux-x64.tar.xz"
DEFAULT_CC = _EXAMPLE_DIR / "tarballs" / "anthropic-ai-claude-code-local-linux-x64.tgz"
DEFAULT_HF = "/workspace/models/Qwen/Qwen3.5-4B"
DEFAULT_SGLANG = "http://127.0.0.1:30000"
DEFAULT_PROMPT = (
    "Read PROBLEM_STATEMENT.md. Reply with a one-line investigation plan only, "
    "then stop. Do not edit files in this smoke run."
)


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    return path


def _check_sglang(url: str, timeout: float = 5.0) -> None:
    base = url.rstrip("/")
    # SGLang serves /health or /get_model_info depending on version; try a few.
    last_err: Exception | None = None
    for path in ("/health", "/get_model_info", "/v1/models"):
        try:
            with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as resp:
                if resp.status < 500:
                    print(f"[smoke] sglang OK {base}{path} -> HTTP {resp.status}", flush=True)
                    return
        except Exception as e:
            last_err = e
    raise SystemExit(
        f"SGLang not reachable at {base} (tried /health,/get_model_info,/v1/models). "
        f"Start it first, e.g.\n"
        f"  CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \\\n"
        f"    --model-path {DEFAULT_HF} --host 127.0.0.1 --port 30000\n"
        f"last_error={last_err}"
    )


def _load_sample(jsonl: Path) -> Sample:
    row = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    return Sample(prompt=row.get("prompt") or "", label=row.get("label"), metadata=row.get("metadata") or {})


def _local_ips() -> list[str]:
    """IPs on this process's network namespace (the pod), that sandboxes may reach."""
    import socket
    import subprocess

    ips: list[str] = []
    seen: set[str] = set()

    def _add(ip: str) -> None:
        ip = (ip or "").strip()
        if not ip or ip.startswith("127.") or ip in seen:
            return
        seen.add(ip)
        ips.append(ip)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            _add(s.getsockname()[0])
    except OSError:
        pass
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True, timeout=5)
        for tok in out.split():
            _add(tok)
    except Exception:
        pass
    return ips


async def _pick_adapter_host(sb, *, port: int, preferred: str | None) -> str:
    """From inside the sandbox, find a host that accepts TCP to the adapter port.

    docker-rt ``--network host`` often attaches to the *node* netns, so
    127.0.0.1 inside the sandbox is NOT the pod where the adapter listens.
    Bridge + host.docker.internal (host-gateway) or the pod IP usually works.
    """
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    # Prefer this process's pod IPs — docker-rt often ignores --add-host so
    # host.docker.internal may not resolve.
    candidates.extend(_local_ips())
    candidates.extend(
        [
            "host.docker.internal",
            "172.17.0.1",
            "10.88.0.1",
            "127.0.0.1",
        ]
    )
    # de-dupe preserving order
    seen: set[str] = set()
    ordered = [h for h in candidates if not (h in seen or seen.add(h))]  # type: ignore[func-returns-value]

    probe = f"""
import socket, sys
hosts = {ordered!r}
port = {port}
for h in hosts:
    try:
        s = socket.create_connection((h, port), timeout=2)
        s.close()
        print(h)
        sys.exit(0)
    except Exception as e:
        print(f"FAIL {{h}}: {{e}}", file=sys.stderr)
sys.exit(2)
"""
    await sb.write_file("/tmp/smoke_adapter_probe.py", probe)
    code, out, err = await sb.exec("python3 /tmp/smoke_adapter_probe.py", timeout=60)
    print(f"[smoke] adapter reachability probe exit={code}\n{(err or '').strip()[:800]}", flush=True)
    host = (out or "").strip().splitlines()[-1] if (out or "").strip() else ""
    if code != 0 or not host or host.startswith("FAIL"):
        raise SystemExit(
            f"FAIL: sandbox cannot TCP-connect to adapter :{port} via any of {ordered}. "
            f"Adapter must bind 0.0.0.0 (not only 127.0.0.1). "
            f"Try --network bridge (default) and ensure SLIME_AGENT_DOCKER_ADD_HOST="
            f"host.docker.internal:host-gateway."
        )
    print(f"[smoke] sandbox reaches adapter via host={host!r}", flush=True)
    return host


def _setup_docker_env(*, network: str) -> None:
    os.environ["SLIME_AGENT_SANDBOX_BACKEND"] = "docker"
    os.environ["SLIME_AGENT_DOCKER_NETWORK"] = network
    if network == "host":
        # Keep empty; host netns may be the node, not the pod.
        os.environ.pop("SLIME_AGENT_DOCKER_ADD_HOST", None)
    else:
        os.environ["SLIME_AGENT_DOCKER_ADD_HOST"] = os.environ.get(
            "SLIME_AGENT_DOCKER_ADD_HOST", "host.docker.internal:host-gateway"
        )
    if "SLIME_AGENT_CC_EXTRA_ARGS" not in os.environ:
        settings = json.dumps({"permissions": {"defaultMode": "bypassPermissions"}})
        os.environ["SLIME_AGENT_CC_EXTRA_ARGS"] = (
            f"--settings '{settings}' --disallowedTools WebFetch WebSearch"
        )
    # Claude Code defaults max output to 32k and errors if the wire claims more;
    # keep smoke replies small. Also avoid the adapter returning empty "length"
    # turns just because --max-context-len was tiny.
    if "SLIME_AGENT_CC_EXTRA_ENVS" not in os.environ:
        os.environ["SLIME_AGENT_CC_EXTRA_ENVS"] = json.dumps(
            {
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096",
                "MAX_THINKING_TOKENS": "0",
            }
        )


async def run_smoke(args: argparse.Namespace) -> None:
    _setup_docker_env(network=args.network)
    os.environ["SLIME_AGENT_NODE_TARBALL"] = str(args.node_tarball)
    os.environ["SLIME_AGENT_CC_TARBALL"] = str(args.cc_tarball)
    os.environ["SWE_AGENT"] = "claude_code"
    os.environ["SWE_CC_PROMPT"] = args.prompt

    _check_sglang(args.sglang_url)

    print(f"[smoke] loading tokenizer {args.hf_checkpoint}", flush=True)
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    adapter = AnthropicAdapter(
        tokenizer=tokenizer,
        sglang_url=args.sglang_url.rstrip("/"),
        tool_parser="qwen3_coder",
        reasoning_parser="qwen3",
    )
    # Must listen on all interfaces so bridge sandboxes can dial the pod IP /
    # docker host-gateway — binding only 127.0.0.1 makes ConnectionRefused from
    # sibling containers.
    handle = run_app_in_thread(
        adapter.app,
        host=args.bind_host,
        port=args.bind_port,
        thread_name="smoke-anthropic-adapter",
        runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
    )
    print(
        f"[smoke] anthropic adapter listening {args.bind_host}:{handle.port}  sglang={args.sglang_url}",
        flush=True,
    )

    sample = _load_sample(args.jsonl)
    md = swe.get_metadata(sample, swe.PROTOCOL_SCALESWE)
    image = args.image or md["image"]
    workdir = md["workdir"]
    instance_id = md["instance_id"]
    session_id = f"smoke-{secrets.token_hex(8)}"

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
        async with DockerSandbox(image, pull=args.pull) as sb:
            print(f"[smoke] sandbox_id={sb.sandbox_id}", flush=True)

            from slime.agent.sandbox import ensure_agent_user

            print("[smoke] ensure_agent_user...", flush=True)
            await ensure_agent_user(sb, workdir)

            code, out, err = await sb.exec("id -u; id -un", user="agent", timeout=30)
            print(f"[smoke] agent id exit={code} out={(out or '').strip()!r}", flush=True)
            uid = (out or "").strip().splitlines()[0] if (out or "").strip() else ""
            if code != 0 or uid == "0" or "root" in (out or "").split():
                raise SystemExit(
                    f"FAIL: sandbox still running as root under user=agent "
                    f"(docker-rt -u / runuser bug?). out={out!r} err={err!r}"
                )

            adapter_host = await _pick_adapter_host(
                sb, port=handle.port, preferred=(args.public_host or None)
            )
            adapter_url = f"http://{adapter_host}:{handle.port}"
            print(f"[smoke] ANTHROPIC_BASE_URL will be {adapter_url}", flush=True)

            harness = ClaudeCodeHarness()
            print("[smoke] install_cli (node + claude)...", flush=True)
            await harness.install_cli(sb)

            print("[smoke] prepare_workspace...", flush=True)
            await swe.prepare_workspace(sb, workdir, md)
            # Shrink the on-disk problem statement for smoke so Claude Code's
            # first turn (system+tools+prompt) fits; training uses the full text.
            await sb.write_file(
                f"{workdir}/PROBLEM_STATEMENT.md",
                "# Smoke\n\nReply with a one-line plan and stop. Do not edit files.\n",
                user="agent",
            )

            print(f"[smoke] claude_code.run budget={args.time_budget}s prompt={args.prompt[:80]!r}...", flush=True)
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
                f"echo '=== trajectory tail ==='; "
                f"tail -c 1500 {workdir}/.harness/trajectory.jsonl 2>/dev/null || true",
                timeout=30,
            )
            print(f"[smoke] diag:\n{(diag or '')[:2000]}", flush=True)

            if args.eval:
                diff_text = await swe.git_diff(sb, workdir)
                reward, applied = await swe.run_evaluation(
                    md, diff_text=diff_text, timeout_sec=args.eval_timeout
                )
                print(f"[smoke] eval reward={reward} applied={applied}", flush=True)
            else:
                reward = 0.0

        segments = await adapter.finish_session(
            session_id,
            base_sample=sample,
            reward=float(reward),
            extra_metadata={"instance_id": instance_id, "smoke": True},
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
            "FAIL: adapter_session_empty — Claude Code never completed a turn "
            "through AnthropicAdapter→SGLang. Check diag above "
            "(root/sudo? adapter unreachable? SGLang /generate errors?)."
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
    p.add_argument("--cc-tarball", type=Path, default=Path(os.environ.get("SLIME_AGENT_CC_TARBALL", DEFAULT_CC)))
    p.add_argument("--hf-checkpoint", default=os.environ.get("HF_CHECKPOINT", DEFAULT_HF))
    p.add_argument("--sglang-url", default=os.environ.get("SMOKE_SGLANG_URL", DEFAULT_SGLANG))
    p.add_argument("--bind-host", default=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"))
    p.add_argument("--bind-port", type=int, default=int(os.environ.get("ADAPTER_PORT", "18001")))
    p.add_argument(
        "--public-host",
        default=os.environ.get("ADAPTER_PUBLIC_HOST", ""),
        help="Preferred adapter host for sandboxes; empty = auto-probe "
        "(host.docker.internal / pod IP / …). Do NOT use 127.0.0.1 with docker-rt.",
    )
    p.add_argument(
        "--network",
        default=os.environ.get("SLIME_AGENT_DOCKER_NETWORK", "bridge"),
        help="Docker network (default bridge). docker-rt --network host usually is "
        "the node netns, so sandbox 127.0.0.1 != pod adapter.",
    )
    p.add_argument("--time-budget", type=int, default=int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "180")))
    p.add_argument(
        "--max-context-len",
        type=int,
        default=int(os.environ.get("SMOKE_MAX_CONTEXT_LEN", "96000")),
        help="Adapter session max_context_tokens (must fit Claude Code system+tools+prompt; "
        "8192 is too small and yields empty length turns → max_output_tokens noise)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "1024")),
        help="Default sampling max_new_tokens for the adapter session",
    )
    p.add_argument("--prompt", default=os.environ.get("SWE_CC_PROMPT", DEFAULT_PROMPT))
    p.add_argument("--eval", action="store_true", help="Also run scaleswe grading (slow)")
    p.add_argument("--eval-timeout", type=int, default=600)
    p.add_argument("--require-exit-zero", action="store_true")
    args = p.parse_args()

    _require_file(args.node_tarball, "SLIME_AGENT_NODE_TARBALL")
    _require_file(args.cc_tarball, "SLIME_AGENT_CC_TARBALL")
    _require_file(args.jsonl, "jsonl")
    if not Path(args.hf_checkpoint).exists():
        raise SystemExit(f"missing hf checkpoint: {args.hf_checkpoint}")

    asyncio.run(run_smoke(args))


if __name__ == "__main__":
    main()
