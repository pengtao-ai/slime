"""Shared OpenAI-harness infer loop (Codex).

Same Docker + traj-dump shape as ``infer_anthropic_harness``, but speaks
``/v1/chat/completions`` via ``TrajRecordingOpenAIAdapter``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.harness.common import BaseHarness
from slime.agent.sandbox import DockerSandbox, ensure_agent_user
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

import swe
from traj_dump import TrajRecordingOpenAIAdapter, is_tool_turn, load_rows, save_outputs

logger = logging.getLogger("infer_openai_harness")


async def run_one(
    args: argparse.Namespace,
    adapter: TrajRecordingOpenAIAdapter,
    handle: Any,
    row: dict[str, Any],
    idx: int,
    out_root: Path,
    *,
    harness: BaseHarness,
    agent_name: str,
    mode: str,
    sid_prefix: str,
) -> dict[str, Any]:
    sample = Sample(prompt=row.get("prompt") or "", label=row.get("label"), metadata=row.get("metadata") or {})
    md = swe.get_metadata(sample, swe.PROTOCOL_SCALESWE)
    image, workdir, iid = md["image"], md["workdir"], md["instance_id"]
    sid = f"{sid_prefix}-{idx}-{secrets.token_hex(4)}"
    sample_dir = out_root / f"i{idx:03d}_{iid.replace('/', '_')}"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    requests_dir = sample_dir / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    adapter.bind_session_outdir(sid, requests_dir)

    adapter.open_session(
        sid,
        sampling_defaults={"temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        max_context_tokens=args.max_context_len,
    )
    prompt = args.prompt or swe.SWE_PROMPT
    meta = {"instance_id": iid, "agent": agent_name}

    agent_exit_code = -999
    harness_traj = ""
    patch_diff = ""
    error: str | None = None
    t0 = time.monotonic()
    try:
        print(f"[infer:{idx}] start {iid} image={image} agent={agent_name}", flush=True)
        async with DockerSandbox(image) as sb:
            await ensure_agent_user(sb, workdir)
            from smoke_claude_code_docker import _pick_adapter_host

            host = await _pick_adapter_host(sb, port=handle.port, preferred=args.public_host or None)
            adapter_url = f"http://{host}:{handle.port}"
            await harness.install_cli(sb)
            await swe.prepare_workspace(sb, workdir, md)
            agent_exit_code = await harness.run(
                sb,
                workdir=workdir,
                session_id=sid,
                adapter_url=adapter_url,
                time_budget_sec=args.time_budget,
                prompt=prompt,
            )
            _, traj, _ = await sb.exec(
                f"cat {workdir}/.harness/trajectory.jsonl 2>/dev/null || true",
                timeout=60,
            )
            harness_traj = traj or ""
            patch_diff = await swe.git_diff(sb, workdir)
            print(
                f"[infer:{idx}] agent_exit={agent_exit_code} patch_bytes={len(patch_diff or '')}",
                flush=True,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("[infer:%s] failed instance=%s", idx, iid)
    finally:
        traj_turns = adapter.pop_session_traj(sid)
        try:
            segs = await adapter.finish_session(sid, base_sample=sample, reward=0.0, extra_metadata=meta)
        except Exception:
            segs = []
        await adapter.drop_session(sid, wait_timeout=10)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ok": bool(traj_turns) and error is None,
        "mode": mode,
        "index": idx,
        "instance_id": iid,
        "session_id": sid,
        "agent_exit_code": agent_exit_code,
        "turns": len(traj_turns),
        "tool_use_turns": sum(1 for t in traj_turns if is_tool_turn(t)),
        "reward": 0.0,
        "eval_applied": False,
        "patch_chars": len(patch_diff or ""),
        "elapsed_sec": round(time.monotonic() - t0, 3),
        "error": error,
        "sglang_url": args.sglang_url,
        "hf_checkpoint": args.hf_checkpoint,
        "model": adapter.model_name,
        "prompt": prompt,
        "time_budget_sec": args.time_budget,
        "segments": len(segs),
    }
    save_outputs(
        out_dir=sample_dir,
        traj_turns=traj_turns,
        harness_traj=harness_traj,
        patch_diff=patch_diff or "",
        summary=summary,
    )
    print(
        f"[infer:{idx}] done ok={summary['ok']} turns={summary['turns']} "
        f"exit={agent_exit_code} err={error}",
        flush=True,
    )
    return summary


async def main_async(
    args: argparse.Namespace,
    *,
    harness: BaseHarness,
    agent_name: str,
    mode: str,
    sid_prefix: str,
    thread_name: str,
    setup_env: Callable[[argparse.Namespace], None],
) -> None:
    os.environ["SLIME_AGENT_SANDBOX_BACKEND"] = "docker"
    os.environ["SLIME_AGENT_DOCKER_NETWORK"] = args.network
    os.environ["SLIME_AGENT_DOCKER_ADD_HOST"] = "host.docker.internal:host-gateway"
    os.environ["SWE_AGENT"] = agent_name
    setup_env(args)
    if args.pull:
        os.environ["SLIME_AGENT_DOCKER_PULL"] = "1"

    rows = load_rows(args.jsonl, args.limit)
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    model_name = Path(args.hf_checkpoint).name or "slime-actor"
    adapter = TrajRecordingOpenAIAdapter(
        tokenizer=tok,
        sglang_url=args.sglang_url.rstrip("/"),
        tool_parser="qwen3_coder",
        reasoning_parser="qwen3",
        model_name=model_name,
    )
    handle = run_app_in_thread(
        adapter.app,
        host=args.bind_host,
        port=args.bind_port,
        thread_name=thread_name,
        runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
    )
    print(
        f"[infer] openai adapter {args.bind_host}:{handle.port} sglang={args.sglang_url} "
        f"agent={agent_name} n={len(rows)}",
        flush=True,
    )
    results = []
    try:
        for i, row in enumerate(rows):
            results.append(
                await run_one(
                    args,
                    adapter,
                    handle,
                    row,
                    i,
                    out_root,
                    harness=harness,
                    agent_name=agent_name,
                    mode=mode,
                    sid_prefix=sid_prefix,
                )
            )
    finally:
        handle.stop()
    (out_root / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[infer] ALL", json.dumps(results, indent=2), flush=True)


def add_common_infer_args(p: argparse.ArgumentParser, *, default_bind_port: int) -> None:
    from infer_anthropic_harness import add_common_infer_args as _add

    _add(p, default_bind_port=default_bind_port)
