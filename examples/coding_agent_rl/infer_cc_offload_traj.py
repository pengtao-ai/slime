#!/usr/bin/env python3
"""Inference-only: Claude Code → Anthropic adapter → GLM (no SLM / no offload relay).

No Ray, no training, no local SGLang. Each CC ``/v1/messages`` turn is translated
to OpenAI chat.completions and sent to ``DASHSCOPE_BASE_URL``.

Batch mode (default): first ``--limit`` jsonl rows with ``--concurrency`` parallel
docker+CC sessions sharing one adapter. Per-sample outputs live under::

    OUT_DIR/summary.json                 # batch aggregate
    OUT_DIR/i000_<instance_id>/
      requests/req_XX.json               # one file per GLM call
      trajectory.json / turns.jsonl / summary.json / patch.diff

Prereqs::

    export DASHSCOPE_API_KEY=...
    export DASHSCOPE_BASE_URL=https://.../v1
    export DASHSCOPE_MODEL=deepseek-v4-flash-0731   # optional

Then::

    python examples/coding_agent_rl/infer_cc_offload_traj.py \\
      --out-dir /tmp/cc_glm_traj --time-budget 600

Or::

    bash examples/coding_agent_rl/run_infer_cc_offload_traj.sh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiohttp import web

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EXAMPLE_DIR))

import smoke_claude_code_docker as smoke  # noqa: E402
import swe  # noqa: E402
from examples.coding_agent_rl import offload  # noqa: E402
from slime.agent.adapters.anthropic import AnthropicAdapter  # noqa: E402
from slime.agent.adapters.common import (  # noqa: E402
    Reply,
    Session,
    tool_call_dict,
)
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.harness import ClaudeCodeHarness
from slime.agent.sandbox import DockerSandbox, ensure_agent_user
from slime.utils.types import Sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_cc_glm")

DEFAULT_PROMPT = "Read PROBLEM_STATEMENT.md and fix the issue. Use tools as needed."


class _StubTokenizer:
    """Placeholder so AnthropicAdapter.__init__ is happy; GLM path never renders templates."""

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [0] * max(1, len(text) // 4)

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return ""


def _glm_endpoint() -> tuple[str, str]:
    base = (os.environ.get("DASHSCOPE_BASE_URL") or offload.DEFAULT_DASHSCOPE_BASE_URL).rstrip("/")
    model = os.environ.get("DASHSCOPE_MODEL") or offload.DEFAULT_DASHSCOPE_MODEL
    return base, model


def _require_glm_env() -> None:
    api_key = (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit(
            "DASHSCOPE_API_KEY (or OPENAI_API_KEY) is required. "
            "Also set DASHSCOPE_BASE_URL to an OpenAI-compatible /v1 endpoint."
        )
    base, model = _glm_endpoint()
    print(f"[infer] GLM-only model={model} base={base}", flush=True)


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def _translated_to_openai_messages(translated: list[dict]) -> list[dict[str, Any]]:
    """Chat-template / manager messages → OpenAI chat.completions messages."""
    out: list[dict[str, Any]] = []
    for m in translated:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("reasoning_content"):
                msg["reasoning_content"] = m["reasoning_content"]
            tcs = m.get("tool_calls")
            if isinstance(tcs, list) and tcs:
                openai_tcs = []
                for i, tc in enumerate(tcs):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    name = fn.get("name") or tc.get("name")
                    if not name:
                        continue
                    args = fn.get("arguments") if fn else tc.get("arguments")
                    openai_tcs.append(
                        {
                            "id": str(tc.get("id") or f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": str(name),
                                "arguments": offload._arguments_as_openai_json(args),
                            },
                        }
                    )
                if openai_tcs:
                    msg["tool_calls"] = openai_tcs
            out.append(msg)
        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": str(m.get("tool_call_id") or "unknown"),
                    "content": m.get("content") or "",
                }
            )
        elif role in ("system", "user"):
            out.append({"role": role, "content": m.get("content") or ""})
    return out


def _glm_reply_to_anthropic(
    *,
    content: str,
    think: str,
    tool_calls: list[dict[str, Any]],
) -> tuple[list[dict], str, dict[str, Any]]:
    blocks: list[dict] = []
    if think:
        blocks.append({"type": "thinking", "thinking": think})
    if content:
        blocks.append({"type": "text", "text": content})
    anth_tools = offload._openai_tool_calls_to_anthropic_blocks(tool_calls)
    blocks.extend(anth_tools)
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    stop_reason = "tool_use" if anth_tools else "end_turn"
    manager_message: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if think:
        manager_message["reasoning_content"] = think
    if anth_tools:
        manager_message["tool_calls"] = [
            tool_call_dict(b["name"], b.get("input")) | ({"id": b["id"]} if b.get("id") else {})
            for b in anth_tools
        ]
    return blocks, stop_reason, manager_message


def _split_system_and_history(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Pull leading system text out; keep full history (including system) separately."""
    system_parts: list[str] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            system_parts.append(str(m.get("content") or ""))
    return "\n\n".join(p for p in system_parts if p), list(messages)


def _write_request_file(requests_dir: Path, record: dict[str, Any]) -> Path:
    """One file per GLM request: full system + history messages (+ response)."""
    requests_dir.mkdir(parents=True, exist_ok=True)
    idx = int(record["turn_index"])
    req = record["openai_request"]
    messages = list(req.get("messages") or [])
    system, history = _split_system_and_history(messages)
    payload = {
        "turn_index": idx,
        "sid": record.get("sid"),
        "elapsed_sec": record.get("elapsed_sec"),
        "model": req.get("model"),
        "max_tokens": req.get("max_tokens"),
        "system": system,
        "messages": history,
        "tools": req.get("tools"),
        "response": record.get("openai_response"),
        "anthropic_response": record.get("anthropic_response"),
    }
    path = requests_dir / f"req_{idx:02d}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


class GlmOnlyAnthropicAdapter(AnthropicAdapter):
    """Anthropic /v1/messages → GLM chat.completions; no SGLang / offload."""

    log_prefix = "glm_only_adapter"

    def __init__(
        self,
        *,
        enable_thinking: bool = True,
        reasoning_effort: str | None = None,
        debug_callback=None,
    ) -> None:
        super().__init__(
            tokenizer=_StubTokenizer(),
            sglang_url="http://127.0.0.1:0",  # unused
            debug_callback=debug_callback,
        )
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.traj_by_sid: dict[str, list[dict[str, Any]]] = {}
        self.sid_requests_dir: dict[str, Path] = {}

    def bind_session_outdir(self, sid: str, requests_dir: Path) -> None:
        self.sid_requests_dir[sid] = Path(requests_dir)
        self.traj_by_sid.setdefault(sid, [])

    def pop_session_traj(self, sid: str) -> list[dict[str, Any]]:
        self.sid_requests_dir.pop(sid, None)
        return list(self.traj_by_sid.pop(sid, []))

    async def _run_turn(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self._preprocess_body(body)
        sid = self._session_id(request, body)
        if sid in self.closed:
            return web.Response(status=503, text="session closed")
        capped = self._check_turn_cap(sid)
        if capped is not None:
            return capped

        s = self.store.setdefault(sid, Session())
        task = asyncio.current_task()
        self.inflight.setdefault(sid, set()).add(task)
        t0 = time.monotonic()
        try:
            translated, tools_schema = self._translate(body)
            openai_messages = _translated_to_openai_messages(translated)
            max_tokens = int(
                body.get("max_tokens")
                or (s.sampling_defaults or {}).get("max_new_tokens")
                or offload.DEFAULT_OFFLOAD_MAX_TOKENS
            )

            content, think, usage, tool_calls = await offload.call_remote_chat(
                openai_messages,
                max_tokens=max_tokens,
                enable_thinking=self.enable_thinking,
                reasoning_effort=self.reasoning_effort,
                tools=tools_schema,
            )
            if content.startswith("[Error:"):
                logger.error("[%s] sid=%s GLM error: %s", self.log_prefix, sid, content[:500])

            blocks, stop_reason, manager_message = _glm_reply_to_anthropic(
                content=content,
                think=think,
                tool_calls=tool_calls,
            )
            reply = Reply(
                manager_message=manager_message,
                finish_reason="tool_calls" if tool_calls else "stop",
                wire=(blocks, stop_reason),
            )

            in_tok = int((usage or {}).get("prompt_tokens") or 0)
            out_tok = int((usage or {}).get("completion_tokens") or 0)
            stream = body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")

            try:
                response = await self._respond(request, body, reply, in_tok, out_tok, stream)
            except (ConnectionResetError, asyncio.CancelledError) as e:
                logger.warning(
                    "[%s] sid=%s client disconnected: %s after %.1fs",
                    self.log_prefix,
                    sid,
                    type(e).__name__,
                    time.monotonic() - t0,
                )
                if isinstance(e, asyncio.CancelledError):
                    raise
                return web.Response(status=499, text="client disconnected")

            bucket = self.traj_by_sid.setdefault(sid, [])
            record = {
                "sid": sid,
                "turn_index": len(bucket),
                "elapsed_sec": round(time.monotonic() - t0, 3),
                "openai_request": {
                    "messages": _jsonable(openai_messages),
                    "tools": _jsonable(offload._normalize_openai_tools(tools_schema)),
                    "max_tokens": max_tokens,
                    "model": _glm_endpoint()[1],
                },
                "openai_response": {
                    "content": content,
                    "reasoning_content": think,
                    "tool_calls": _jsonable(tool_calls),
                    "usage": _jsonable(usage),
                },
                "anthropic_response": {
                    "content": blocks,
                    "stop_reason": stop_reason,
                },
                "manager_message": _jsonable(manager_message),
            }
            bucket.append(record)
            req_dir = self.sid_requests_dir.get(sid)
            if req_dir is not None:
                path = _write_request_file(req_dir, record)
                logger.info("[%s] wrote %s (messages=%d)", self.log_prefix, path, len(openai_messages))
            self._run_debug_callback(sid, translated, tools_schema, manager_message, None)
            return response
        finally:
            self.inflight.get(sid, set()).discard(task)


def _save_outputs(
    *,
    out_dir: Path,
    traj_turns: list[dict],
    harness_traj: str,
    patch_diff: str,
    summary: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    requests_dir = out_dir / "requests"
    for row in traj_turns:
        _write_request_file(requests_dir, row)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "created_at": summary.get("created_at"),
                "mode": "glm_only",
                "summary": summary,
                "turns": traj_turns,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (out_dir / "turns.jsonl").open("w", encoding="utf-8") as f:
        for row in traj_turns:
            compact = {
                "turn_index": row["turn_index"],
                "elapsed_sec": row["elapsed_sec"],
                "stop_reason": row["anthropic_response"]["stop_reason"],
                "content": row["openai_response"]["content"],
                "reasoning_content": row["openai_response"]["reasoning_content"],
                "tool_calls": row["openai_response"]["tool_calls"],
                "usage": row["openai_response"]["usage"],
                "n_messages": len(row["openai_request"]["messages"]),
                "n_tools": len(row["openai_request"]["tools"] or []),
                "request_file": f"requests/req_{int(row['turn_index']):02d}.json",
            }
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")
    if harness_traj:
        (out_dir / "harness_trajectory.jsonl").write_text(harness_traj, encoding="utf-8")
    if patch_diff:
        (out_dir / "patch.diff").write_text(patch_diff, encoding="utf-8")
    print(f"[infer] wrote {out_dir} ({len(traj_turns)} request files under requests/)", flush=True)


def _load_samples(jsonl: Path, *, limit: int, offset: int = 0) -> list[tuple[int, Sample]]:
    out: list[tuple[int, Sample]] = []
    data_i = -1
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data_i += 1
            if data_i < offset:
                continue
            if len(out) >= limit:
                break
            row = json.loads(line)
            sample = Sample(
                index=data_i,
                prompt=row.get("prompt") or "",
                label=row.get("label"),
                metadata=row.get("metadata") or {},
            )
            out.append((data_i, sample))
    if not out:
        raise SystemExit(f"no samples loaded from {jsonl} (offset={offset} limit={limit})")
    return out


def _sample_out_dir(root: Path, index: int, instance_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in instance_id)[:80]
    return root / f"i{index:03d}_{safe}"


def _load_done_summary(sample_dir: Path) -> dict[str, Any] | None:
    path = sample_dir / "summary.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


async def _run_one_sample(
    *,
    args: argparse.Namespace,
    adapter: GlmOnlyAnthropicAdapter,
    adapter_port: int,
    index: int,
    sample: Sample,
    sample_dir: Path,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        md = swe.get_metadata(sample, swe.PROTOCOL_SCALESWE)
        image = args.image or md["image"]
        workdir = md["workdir"]
        instance_id = md["instance_id"]
        session_id = f"infer-glm-{index}-{secrets.token_hex(6)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        requests_dir = sample_dir / "requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        adapter.bind_session_outdir(session_id, requests_dir)
        adapter.open_session(
            session_id,
            sampling_defaults={"temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
            max_context_tokens=args.max_context_len,
        )

        agent_exit_code = -999
        harness_traj = ""
        patch_diff = ""
        reward = 0.0
        applied = False
        error: str | None = None
        t0 = time.monotonic()
        try:
            print(f"[infer:{index}] start instance={instance_id} image={image}", flush=True)
            async with DockerSandbox(image) as sb:
                await ensure_agent_user(sb, workdir)
                code, out, err = await sb.exec("id -u; id -un", user="agent", timeout=30)
                uid = (out or "").strip().splitlines()[0] if (out or "").strip() else ""
                if code != 0 or uid == "0" or "root" in (out or "").split():
                    raise RuntimeError(f"sandbox still root under user=agent out={out!r} err={err!r}")

                adapter_host = await smoke._pick_adapter_host(
                    sb, port=adapter_port, preferred=(args.public_host or None)
                )
                adapter_url = f"http://{adapter_host}:{adapter_port}"
                harness = ClaudeCodeHarness()
                await harness.install_cli(sb)
                await swe.prepare_workspace(sb, workdir, md)
                if args.shrink_problem:
                    await sb.write_file(
                        f"{workdir}/PROBLEM_STATEMENT.md",
                        "# Infer smoke\n\nInvestigate briefly with tools, then stop.\n",
                        user="agent",
                    )
                agent_exit_code = await harness.run(
                    sb,
                    workdir=workdir,
                    session_id=session_id,
                    adapter_url=adapter_url,
                    time_budget_sec=args.time_budget,
                    prompt=args.prompt,
                )
                _, traj_out, _ = await sb.exec(
                    f"cat {workdir}/.harness/trajectory.jsonl 2>/dev/null || true",
                    timeout=60,
                )
                harness_traj = traj_out or ""
                patch_diff = await swe.git_diff(sb, workdir)
                if args.eval:
                    reward, applied = await swe.run_evaluation(
                        md, diff_text=patch_diff, timeout_sec=args.eval_timeout
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("[infer:%s] failed instance=%s", index, instance_id)
        finally:
            traj_turns = adapter.pop_session_traj(session_id)
            await adapter.drop_session(session_id, wait_timeout=10)

        base, model = _glm_endpoint()
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ok": bool(traj_turns) and error is None,
            "mode": "glm_only",
            "index": index,
            "instance_id": instance_id,
            "session_id": session_id,
            "agent_exit_code": agent_exit_code,
            "turns": len(traj_turns),
            "tool_use_turns": sum(
                1 for t in traj_turns if t["anthropic_response"]["stop_reason"] == "tool_use"
            ),
            "reward": reward,
            "eval_applied": applied,
            "patch_chars": len(patch_diff or ""),
            "elapsed_sec": round(time.monotonic() - t0, 3),
            "error": error,
            "dashscope_base_url": base,
            "dashscope_model": model,
            "prompt": args.prompt,
            "time_budget_sec": args.time_budget,
        }
        _save_outputs(
            out_dir=sample_dir,
            traj_turns=traj_turns,
            harness_traj=harness_traj,
            patch_diff=patch_diff or "",
            summary=summary,
        )
        print(
            f"[infer:{index}] done ok={summary['ok']} turns={summary['turns']} "
            f"reward={reward} exit={agent_exit_code} err={error}",
            flush=True,
        )
        return summary


async def run_infer(args: argparse.Namespace) -> None:
    # Explicitly disable SLM↔GLM offload relay.
    os.environ["SLIME_AGENT_OFFLOAD"] = "0"
    _require_glm_env()
    smoke._setup_docker_env(network=args.network)
    os.environ["SLIME_AGENT_NODE_TARBALL"] = str(args.node_tarball)
    os.environ["SLIME_AGENT_CC_TARBALL"] = str(args.cc_tarball)
    os.environ["SWE_AGENT"] = "claude_code"
    os.environ["SWE_CC_PROMPT"] = args.prompt
    if args.pull:
        os.environ["SLIME_AGENT_DOCKER_PULL"] = "1"

    samples = _load_samples(args.jsonl, limit=args.limit, offset=args.offset)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force and not args.resume:
        raise SystemExit(
            f"--out-dir already exists and is non-empty: {out_dir} (pass --force or --resume)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[int, Sample, Path]] = []
    done_results: list[dict[str, Any]] = []
    for index, sample in samples:
        instance_id = str(
            (sample.metadata or {}).get("instance_id") or sample.label or f"row{index}"
        )
        sample_dir = _sample_out_dir(out_dir, index, instance_id)
        existing = _load_done_summary(sample_dir) if args.resume else None
        if existing is not None:
            done_results.append(existing)
            print(
                f"[infer:{index}] skip (resume) ok={existing.get('ok')} "
                f"turns={existing.get('turns')} instance={instance_id}",
                flush=True,
            )
            continue
        if args.resume and sample_dir.exists():
            shutil.rmtree(sample_dir)
            print(f"[infer:{index}] cleared incomplete dir {sample_dir.name}", flush=True)
        pending.append((index, sample, sample_dir))

    print(
        f"[infer] samples={len(samples)} resume_skip={len(done_results)} "
        f"pending={len(pending)} concurrency={args.concurrency}",
        flush=True,
    )

    base, model = _glm_endpoint()
    new_results: list[dict[str, Any]] = []
    if pending:
        adapter = GlmOnlyAnthropicAdapter(
            enable_thinking=not args.no_thinking,
            reasoning_effort=args.reasoning_effort or None,
        )
        handle = run_app_in_thread(
            adapter.app,
            host=args.bind_host,
            port=args.bind_port,
            thread_name="infer-glm-adapter",
            runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
        )
        print(
            f"[infer] adapter {args.bind_host}:{handle.port}  glm={model} @ {base}",
            flush=True,
        )
        sem = asyncio.Semaphore(max(1, int(args.concurrency)))
        try:
            tasks = [
                _run_one_sample(
                    args=args,
                    adapter=adapter,
                    adapter_port=handle.port,
                    index=index,
                    sample=sample,
                    sample_dir=sample_dir,
                    sem=sem,
                )
                for index, sample, sample_dir in pending
            ]
            new_results = list(await asyncio.gather(*tasks))
        finally:
            handle.stop()

    results = sorted(done_results + new_results, key=lambda r: int(r.get("index", -1)))

    root_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "glm_only_batch",
        "jsonl": str(args.jsonl),
        "offset": args.offset,
        "limit": args.limit,
        "concurrency": args.concurrency,
        "resume": bool(args.resume),
        "resume_skipped": len(done_results),
        "n": len(results),
        "ok": sum(1 for r in results if r.get("ok")),
        "failed": sum(1 for r in results if not r.get("ok")),
        "mean_reward": (
            sum(float(r.get("reward") or 0.0) for r in results) / len(results) if results else 0.0
        ),
        "dashscope_base_url": base,
        "dashscope_model": model,
        "time_budget_sec": args.time_budget,
        "results": results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(root_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[infer] batch done ok={root_summary['ok']}/{root_summary['n']} "
        f"failed={root_summary['failed']} mean_reward={root_summary['mean_reward']:.3f} "
        f"out={out_dir}",
        flush=True,
    )
    if root_summary["ok"] == 0:
        raise SystemExit("FAIL: all samples failed / empty")
    if args.require_exit_zero and any(int(r.get("agent_exit_code", -1)) != 0 for r in results):
        raise SystemExit("FAIL: some agent_exit_code != 0")
    print("[infer] PASS", flush=True)


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = _REPO_ROOT / "runs" / f"infer_cc_glm_{stamp}"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", type=Path, default=smoke.DEFAULT_JSONL)
    p.add_argument("--image", default=None)
    p.add_argument("--pull", action="store_true")
    p.add_argument("--node-tarball", type=Path, default=Path(os.environ.get("SLIME_AGENT_NODE_TARBALL", smoke.DEFAULT_NODE)))
    p.add_argument("--cc-tarball", type=Path, default=Path(os.environ.get("SLIME_AGENT_CC_TARBALL", smoke.DEFAULT_CC)))
    p.add_argument("--bind-host", default=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"))
    p.add_argument("--bind-port", type=int, default=int(os.environ.get("ADAPTER_PORT", "18011")))
    p.add_argument("--public-host", default=os.environ.get("ADAPTER_PUBLIC_HOST", ""))
    p.add_argument("--network", default=os.environ.get("SLIME_AGENT_DOCKER_NETWORK", "bridge"))
    p.add_argument("--time-budget", type=int, default=int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "600")))
    p.add_argument("--max-context-len", type=int, default=int(os.environ.get("SMOKE_MAX_CONTEXT_LEN", "96000")))
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "8192")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("SMOKE_TEMPERATURE", "1.0")))
    p.add_argument("--prompt", default=os.environ.get("SWE_CC_PROMPT", DEFAULT_PROMPT))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("INFER_OUT_DIR", str(default_out))),
    )
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse --out-dir: skip samples that already have summary.json; re-run incomplete dirs",
    )
    p.add_argument("--shrink-problem", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--eval-timeout", type=int, default=600)
    p.add_argument("--require-exit-zero", action="store_true")
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable chat_template_kwargs.thinking",
    )
    p.add_argument(
        "--reasoning-effort",
        default=os.environ.get("INFER_REASONING_EFFORT", ""),
        help="Optional chat_template_kwargs.reasoning_effort (e.g. max)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("INFER_LIMIT", "16")),
        help="Number of jsonl rows to run (from --offset)",
    )
    p.add_argument(
        "--offset",
        type=int,
        default=int(os.environ.get("INFER_OFFSET", "0")),
        help="Skip this many jsonl rows before --limit",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("INFER_CONCURRENCY", "16")),
        help="Max concurrent docker+CC sessions",
    )
    args = p.parse_args()

    smoke._require_file(args.node_tarball, "SLIME_AGENT_NODE_TARBALL")
    smoke._require_file(args.cc_tarball, "SLIME_AGENT_CC_TARBALL")
    smoke._require_file(args.jsonl, "jsonl")
    if args.limit <= 0:
        raise SystemExit("--limit must be > 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")

    asyncio.run(run_infer(args))


if __name__ == "__main__":
    main()
