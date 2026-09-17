#!/usr/bin/env python3
"""SLM-only agent trajectories (no offload / no GLM relay) + think entropy.

Uses local vLLM PyroDash-4B-SFT-0902 as the sole LLM behind the agent adapter.
No ``<|llm_offload|>`` path, no remote teacher model.

Pipeline
--------
1. Start vLLM for 0902 (see ``launch_vllm_pyrodash4b_sft0902.sh``).
2. Point ``DASHSCOPE_*`` at that OpenAI-compatible ``/v1`` endpoint.
3. Run on ``mixed_reward1_agents_baked.jsonl`` (tmax + scaleswe).
4. Each turn hits vLLM with ``logprobs`` and records thinking entropy on
   ``openai_response.entropy`` / ``req_*.json``.

Example::

    bash examples/coding_agent_rl/sft/launch_vllm_pyrodash4b_sft0902.sh
    # other terminal:
    bash examples/coding_agent_rl/sft/run_infer_sft_traj.sh --eval --limit 2
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

import requests
from aiohttp import web

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EXAMPLE_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

import infer.infer_cc_offload_traj as base  # noqa: E402  (reuse GLM-only adapter scaffolding only)
import smoke.smoke_claude_code_docker as smoke  # noqa: E402
import swe  # noqa: E402
from slime.agent.adapters.common import Reply, Session  # noqa: E402
from slime.agent.sandbox import DockerSandbox, ensure_agent_user  # noqa: E402
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

from agents_registry import resolve_agent  # noqa: E402

# Reuse entropy helpers from annotate_reward1_entropy (same remap + scope logic).
import annotate_reward1_entropy as ent  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_sft")

DEFAULT_JSONL = (
    _EXAMPLE_DIR / "data" / "release" / "mixed_reward1_agents_baked.jsonl"
)
DEFAULT_MODEL = "/workspace/models/pyromind/PyroDash-4B-SFT-0902"
DEFAULT_VLLM_URL = "http://127.0.0.1:8066/v1"
DEFAULT_PROMPT = (
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the task. "
    "Use tools as needed. When finished, print a one-line summary and exit."
)


def _resolve_protocol(sample: Sample) -> str:
    raw = (sample.metadata or {}).get("protocol")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return swe.PROTOCOL_TMAX


def _entropy_from_vllm_data(data: dict[str, Any], *, scope: str) -> dict[str, Any]:
    """Thinking entropy payload (no per-token dump — keeps req_*.json small)."""
    payload = ent._compute_entropy_from_response(data, scope=scope)
    # Drop bulky per-token list if annotate helpers ever add it via to_dict.
    payload.pop("tokens", None)
    return payload


def _call_vllm_chat_sync(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    enable_thinking: bool,
    reasoning_effort: str | None,
    tools: list[dict[str, Any]] | None,
    top_logprobs: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout: float,
) -> tuple[str, str, dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    """OpenAI chat.completions against local vLLM with logprobs for entropy."""
    api_key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    ).strip()
    base_url, model = base._glm_endpoint()
    if max_tokens <= 0:
        return "[Error: no remaining token budget]", "", None, [], None

    chat_kwargs: dict[str, Any] = {"enable_thinking": bool(enable_thinking)}
    # Some gateways also honor ``thinking``; keep both when effort is set.
    if enable_thinking and reasoning_effort:
        chat_kwargs["thinking"] = True
        chat_kwargs["reasoning_effort"] = reasoning_effort

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "logprobs": True,
        "top_logprobs": int(top_logprobs),
        "chat_template_kwargs": chat_kwargs,
    }
    openai_tools = base.offload._normalize_openai_tools(tools)
    if openai_tools:
        body["tools"] = openai_tools

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        if resp.status_code != 200:
            return (
                f"[Error: status {resp.status_code}: {resp.text[:400]}]",
                "",
                None,
                [],
                None,
            )
        data = resp.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        think = str(message.get("reasoning") or message.get("reasoning_content") or "")
        content = str(message.get("content") or "")
        tool_calls = base.offload._parse_openai_tool_calls(message.get("tool_calls"))
        usage = data.get("usage")
        if usage is not None and not isinstance(usage, dict):
            usage = None
        return content, think, usage, tool_calls, data
    except Exception as exc:
        return f"[Error: remote call failed: {exc}]", "", None, [], None


async def _slm_chat(
    adapter: Any,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    tools: list[dict[str, Any]] | None,
) -> tuple[str, str, dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    """Call local SLM (vLLM); never the offload/GLM relay."""
    return await asyncio.to_thread(
        _call_vllm_chat_sync,
        messages,
        max_tokens=max_tokens,
        enable_thinking=adapter.enable_thinking,
        reasoning_effort=adapter.reasoning_effort,
        tools=tools,
        top_logprobs=adapter.top_logprobs,
        temperature=adapter.temperature,
        top_p=adapter.top_p,
        top_k=adapter.top_k,
        timeout=adapter.timeout,
    )


class SlmEntropyAnthropicAdapter(base.GlmOnlyAnthropicAdapter):
    """Anthropic adapter → local SLM only; records think entropy per turn."""

    log_prefix = "slm_entropy_adapter"

    def __init__(
        self,
        *,
        enable_thinking: bool = True,
        reasoning_effort: str | None = None,
        top_logprobs: int = 20,
        entropy_scope: str = "thinking",
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        timeout: float = 600.0,
        debug_callback=None,
    ) -> None:
        super().__init__(
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            debug_callback=debug_callback,
        )
        self.top_logprobs = top_logprobs
        self.entropy_scope = entropy_scope
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout

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
            openai_messages = base._translated_to_openai_messages(translated)
            max_tokens = int(
                body.get("max_tokens")
                or (s.sampling_defaults or {}).get("max_new_tokens")
                or 8192
            )
            content, think, usage, tool_calls, data = await _slm_chat(
                self, openai_messages, max_tokens=max_tokens, tools=tools_schema
            )
            if content.startswith("[Error:"):
                logger.error("[%s] sid=%s SLM error: %s", self.log_prefix, sid, content[:500])

            blocks, stop_reason, manager_message = base._glm_reply_to_anthropic(
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

            openai_response: dict[str, Any] = {
                "content": content,
                "reasoning_content": think,
                "tool_calls": base._jsonable(tool_calls),
                "usage": base._jsonable(usage),
            }
            if data is not None:
                openai_response["entropy"] = _entropy_from_vllm_data(
                    data, scope=self.entropy_scope
                )

            bucket = self.traj_by_sid.setdefault(sid, [])
            record = {
                "sid": sid,
                "turn_index": len(bucket),
                "elapsed_sec": round(time.monotonic() - t0, 3),
                "openai_request": {
                    "messages": base._jsonable(openai_messages),
                    "tools": base._jsonable(base.offload._normalize_openai_tools(tools_schema)),
                    "max_tokens": max_tokens,
                    "model": base._glm_endpoint()[1],
                },
                "openai_response": openai_response,
                "anthropic_response": {
                    "content": blocks,
                    "stop_reason": stop_reason,
                },
                "manager_message": base._jsonable(manager_message),
            }
            bucket.append(record)
            req_dir = self.sid_requests_dir.get(sid)
            if req_dir is not None:
                path = base._write_request_file(req_dir, record)
                logger.info("[%s] wrote %s (messages=%d)", self.log_prefix, path, len(openai_messages))
            self._run_debug_callback(sid, translated, tools_schema, manager_message, None)
            return response
        finally:
            self.inflight.get(sid, set()).discard(task)


class SlmEntropyOpenAIAdapter(base.GlmOnlyOpenAIAdapter):
    """OpenAI adapter → local SLM only; records think entropy per turn."""

    log_prefix = "slm_entropy_openai_adapter"

    def __init__(
        self,
        *,
        enable_thinking: bool = True,
        reasoning_effort: str | None = None,
        top_logprobs: int = 20,
        entropy_scope: str = "thinking",
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        timeout: float = 600.0,
        debug_callback=None,
    ) -> None:
        super().__init__(
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            debug_callback=debug_callback,
        )
        self.top_logprobs = top_logprobs
        self.entropy_scope = entropy_scope
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout

    async def _run_turn(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
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
            openai_messages = base._translated_to_openai_messages(translated)
            max_tokens = int(
                body.get("max_completion_tokens")
                or body.get("max_tokens")
                or (s.sampling_defaults or {}).get("max_new_tokens")
                or 8192
            )
            content, think, usage, tool_calls, data = await _slm_chat(
                self, openai_messages, max_tokens=max_tokens, tools=tools_schema
            )
            if content.startswith("[Error:"):
                logger.error("[%s] sid=%s SLM error: %s", self.log_prefix, sid, content[:500])

            wire_message, manager_message, wire_finish, stop_reason = base._openai_wire_from_glm(
                content=content,
                think=think,
                tool_calls=tool_calls,
            )
            reply = Reply(
                manager_message=manager_message,
                finish_reason="tool_calls" if tool_calls else "stop",
                wire=(wire_message, wire_finish),
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

            openai_response: dict[str, Any] = {
                "content": content,
                "reasoning_content": think,
                "tool_calls": base._jsonable(tool_calls),
                "usage": base._jsonable(usage),
            }
            if data is not None:
                openai_response["entropy"] = _entropy_from_vllm_data(
                    data, scope=self.entropy_scope
                )

            bucket = self.traj_by_sid.setdefault(sid, [])
            record = {
                "sid": sid,
                "turn_index": len(bucket),
                "elapsed_sec": round(time.monotonic() - t0, 3),
                "openai_request": {
                    "messages": base._jsonable(openai_messages),
                    "tools": base._jsonable(base.offload._normalize_openai_tools(tools_schema)),
                    "max_tokens": max_tokens,
                    "model": base._glm_endpoint()[1],
                },
                "openai_response": openai_response,
                "anthropic_response": {
                    "content": [{"type": "text", "text": content}] if content else [],
                    "stop_reason": stop_reason,
                },
                "manager_message": base._jsonable(manager_message),
            }
            bucket.append(record)
            req_dir = self.sid_requests_dir.get(sid)
            if req_dir is not None:
                path = base._write_request_file(req_dir, record)
                logger.info("[%s] wrote %s (messages=%d)", self.log_prefix, path, len(openai_messages))
            self._run_debug_callback(sid, translated, tools_schema, manager_message, None)
            return response
        finally:
            self.inflight.get(sid, set()).discard(task)


class SlmEntropyDualAdapters(base.DualInferAdapters):
    """SLM-only dual adapters (no offload) with per-turn think entropy."""

    def __init__(
        self,
        *,
        enable_thinking: bool = True,
        reasoning_effort: str | None = None,
        top_logprobs: int = 20,
        entropy_scope: str = "thinking",
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        timeout: float = 600.0,
    ) -> None:
        self.anthropic = SlmEntropyAnthropicAdapter(
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            top_logprobs=top_logprobs,
            entropy_scope=entropy_scope,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            timeout=timeout,
        )
        self.openai = SlmEntropyOpenAIAdapter(
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            top_logprobs=top_logprobs,
            entropy_scope=entropy_scope,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            timeout=timeout,
        )
        self.openai._register_routes(self.anthropic.app)
        self._sid_protocol: dict[str, str] = {}
        self.app = self.anthropic.app


def _turn_entropies(traj_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in traj_turns:
        ent_obj = (t.get("openai_response") or {}).get("entropy")
        rows.append(
            {
                "turn_index": t.get("turn_index"),
                "avg_entropy": (ent_obj or {}).get("avg_entropy") if isinstance(ent_obj, dict) else None,
                "n_used": (ent_obj or {}).get("n_used") if isinstance(ent_obj, dict) else None,
                "unavailable": (ent_obj or {}).get("unavailable") if isinstance(ent_obj, dict) else None,
            }
        )
    return rows


def _mean_think_entropy(traj_turns: list[dict[str, Any]]) -> float | None:
    vals = [
        float(r["avg_entropy"])
        for r in _turn_entropies(traj_turns)
        if isinstance(r.get("avg_entropy"), (int, float))
    ]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 6)


async def _run_one_sample(
    *,
    args: argparse.Namespace,
    adapters: SlmEntropyDualAdapters,
    adapter_port: int,
    index: int,
    sample: Sample,
    sample_dir: Path,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        protocol = _resolve_protocol(sample)
        md = swe.get_metadata(sample, protocol)
        agent_spec = resolve_agent(md.get("agent"))
        harness = agent_spec.harness_cls()
        image = args.image or md["image"]
        workdir = md["workdir"]
        instance_id = md["instance_id"]
        if not image or not workdir:
            raise SystemExit(f"[infer:{index}] missing image/workdir for {instance_id}")
        uneval = swe.evaluability_check(md)
        if args.eval and uneval:
            raise SystemExit(f"[infer:{index}] unevaluable ({uneval}): {instance_id}")

        session_id = f"infer-sft-{agent_spec.name}-{index}-{secrets.token_hex(6)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        requests_dir = sample_dir / "requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        adapters.open_session(
            session_id,
            protocol=agent_spec.adapter_protocol,
            sampling_defaults={"temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
            max_context_tokens=args.max_context_len,
        )
        adapters.bind_session_outdir(session_id, requests_dir)

        agent_exit_code = -999
        harness_traj = ""
        patch_diff = ""
        reward = 0.0
        applied = False
        error: str | None = None
        t0 = time.monotonic()
        try:
            print(
                f"[infer:{index}] start agent={agent_spec.name} instance={instance_id} "
                f"protocol={md.get('protocol')} image={image}",
                flush=True,
            )
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

                is_tmax = md.get("protocol") == swe.PROTOCOL_TMAX
                if not is_tmax:
                    patch_diff = await swe.git_diff(sb, workdir)

                if args.eval:
                    if is_tmax:
                        reward, applied = await swe.grade_tmax_inplace(
                            sb, md, timeout_sec=args.eval_timeout
                        )
                    else:
                        reward, applied = await swe.run_evaluation(
                            md, diff_text=patch_diff, timeout_sec=args.eval_timeout
                        )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "[infer:%s] failed instance=%s agent=%s", index, instance_id, agent_spec.name
            )
        finally:
            traj_turns = adapters.pop_session_traj(session_id)
            await adapters.drop_session(session_id, wait_timeout=10)

        base_url, model = base._glm_endpoint()
        turn_ents = _turn_entropies(traj_turns)
        mean_ent = _mean_think_entropy(traj_turns)
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ok": bool(traj_turns) and error is None,
            "mode": "slm_only_0902_entropy",
            "protocol": md.get("protocol"),
            "agent": agent_spec.name,
            "adapter_protocol": agent_spec.adapter_protocol,
            "index": index,
            "instance_id": instance_id,
            "session_id": session_id,
            "agent_exit_code": agent_exit_code,
            "turns": len(traj_turns),
            "tool_use_turns": base._count_tool_use_turns(traj_turns),
            "reward": reward,
            "eval_applied": applied,
            "eval": bool(args.eval),
            "patch_chars": len(patch_diff or ""),
            "elapsed_sec": round(time.monotonic() - t0, 3),
            "error": error,
            "dashscope_base_url": base_url,
            "dashscope_model": model,
            "prompt": args.prompt,
            "time_budget_sec": args.time_budget,
            "entropy_scope": args.entropy_scope,
            "mean_think_entropy": mean_ent,
            "turn_entropies": turn_ents,
        }
        base._save_outputs(
            out_dir=sample_dir,
            traj_turns=traj_turns,
            harness_traj=harness_traj,
            patch_diff=patch_diff or "",
            summary=summary,
        )
        (sample_dir / "turn_entropies.json").write_text(
            json.dumps(
                {
                    "mean_think_entropy": mean_ent,
                    "entropy_scope": args.entropy_scope,
                    "turns": turn_ents,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[infer:{index}] done agent={agent_spec.name} ok={summary['ok']} turns={summary['turns']} "
            f"reward={reward} mean_H={mean_ent} exit={agent_exit_code} err={error}",
            flush=True,
        )
        return summary


async def run_infer(args: argparse.Namespace) -> None:
    # Hard-disable offload; this path is SLM-only (local vLLM).
    os.environ["SLIME_AGENT_OFFLOAD"] = "0"
    os.environ.pop("SLIME_OFFLOAD_EMBED_IN_TRAJECTORY", None)
    # Defaults for local vLLM 0902 if caller did not set them.
    os.environ.setdefault("DASHSCOPE_BASE_URL", DEFAULT_VLLM_URL)
    os.environ.setdefault("DASHSCOPE_MODEL", DEFAULT_MODEL)
    os.environ.setdefault("DASHSCOPE_API_KEY", "EMPTY")
    base._require_glm_env()
    print(
        f"[infer] SLM-only (no offload) model={os.environ['DASHSCOPE_MODEL']} "
        f"base={os.environ['DASHSCOPE_BASE_URL']}",
        flush=True,
    )
    smoke._setup_docker_env(network=args.network)
    base.setup_agent_tarball_envs(node_tarball=args.node_tarball, cc_tarball=args.cc_tarball)
    os.environ["SWE_CC_PROMPT"] = args.prompt
    if args.pull:
        os.environ["SLIME_AGENT_DOCKER_PULL"] = "1"

    samples = base._load_samples(args.jsonl, limit=args.limit, offset=args.offset)
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
        sample_dir = base._sample_out_dir(out_dir, index, instance_id)
        existing = base._load_done_summary(sample_dir) if args.resume else None
        if existing is not None:
            done_results.append(existing)
            print(
                f"[infer:{index}] skip (resume) ok={existing.get('ok')} "
                f"turns={existing.get('turns')} reward={existing.get('reward')} "
                f"instance={instance_id}",
                flush=True,
            )
            continue
        if args.resume and sample_dir.exists():
            shutil.rmtree(sample_dir)
            print(f"[infer:{index}] cleared incomplete dir {sample_dir.name}", flush=True)
        pending.append((index, sample, sample_dir))

    print(
        f"[infer] samples={len(samples)} resume_skip={len(done_results)} "
        f"pending={len(pending)} concurrency={args.concurrency} eval={args.eval}",
        flush=True,
    )

    base_url, model = base._glm_endpoint()
    try:
        health = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {os.environ.get('DASHSCOPE_API_KEY', 'EMPTY')}"},
            timeout=10,
        )
        print(f"[vllm] GET /models -> HTTP {health.status_code} @ {base_url}", flush=True)
        if health.status_code != 200:
            print(health.text[:300], file=sys.stderr)
    except Exception as exc:
        print(f"[warn] vLLM health check failed: {exc}", file=sys.stderr)

    new_results: list[dict[str, Any]] = []
    if pending:
        adapters = SlmEntropyDualAdapters(
            enable_thinking=not args.no_thinking,
            reasoning_effort=args.reasoning_effort or None,
            top_logprobs=args.top_logprobs,
            entropy_scope=args.entropy_scope,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout=args.llm_timeout,
        )
        handle = run_app_in_thread(
            adapters.app,
            host=args.bind_host,
            port=args.bind_port,
            thread_name="infer-sft-adapter",
            runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
        )
        print(
            f"[infer] adapter {args.bind_host}:{handle.port}  model={model} @ {base_url} "
            f"entropy_scope={args.entropy_scope} top_logprobs={args.top_logprobs}",
            flush=True,
        )
        sem = asyncio.Semaphore(max(1, int(args.concurrency)))
        try:
            tasks = [
                _run_one_sample(
                    args=args,
                    adapters=adapters,
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
    mean_ents = [
        float(r["mean_think_entropy"])
        for r in results
        if isinstance(r.get("mean_think_entropy"), (int, float))
    ]

    root_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "slm_only_0902_entropy_batch",
        "jsonl": str(args.jsonl),
        "offset": args.offset,
        "limit": args.limit,
        "concurrency": args.concurrency,
        "eval": bool(args.eval),
        "resume": bool(args.resume),
        "resume_skipped": len(done_results),
        "n": len(results),
        "ok": sum(1 for r in results if r.get("ok")),
        "failed": sum(1 for r in results if not r.get("ok")),
        "solved": sum(1 for r in results if float(r.get("reward") or 0.0) >= 1.0),
        "mean_reward": (
            sum(float(r.get("reward") or 0.0) for r in results) / len(results) if results else 0.0
        ),
        "entropy_scope": args.entropy_scope,
        "mean_think_entropy": (
            round(sum(mean_ents) / len(mean_ents), 6) if mean_ents else None
        ),
        "dashscope_base_url": base_url,
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
        f"failed={root_summary['failed']} solved={root_summary['solved']} "
        f"mean_reward={root_summary['mean_reward']:.3f} "
        f"mean_H={root_summary['mean_think_entropy']} out={out_dir}",
        flush=True,
    )
    if root_summary["ok"] == 0:
        raise SystemExit("FAIL: all samples failed / empty")
    if args.require_exit_zero and any(int(r.get("agent_exit_code", -1)) != 0 for r in results):
        raise SystemExit("FAIL: some agent_exit_code != 0")
    print("[infer] PASS", flush=True)


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = _REPO_ROOT / "runs" / f"infer_sft_0902_{stamp}"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    p.add_argument("--image", default=None)
    p.add_argument("--pull", action="store_true")
    p.add_argument("--node-tarball", type=Path, default=Path(os.environ.get("SLIME_AGENT_NODE_TARBALL", smoke.DEFAULT_NODE)))
    p.add_argument("--cc-tarball", type=Path, default=Path(os.environ.get("SLIME_AGENT_CC_TARBALL", smoke.DEFAULT_CC)))
    p.add_argument("--bind-host", default=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"))
    p.add_argument("--bind-port", type=int, default=int(os.environ.get("ADAPTER_PORT", "18041")))
    p.add_argument("--public-host", default=os.environ.get("ADAPTER_PUBLIC_HOST", ""))
    p.add_argument("--network", default=os.environ.get("SLIME_AGENT_DOCKER_NETWORK", "bridge"))
    p.add_argument("--time-budget", type=int, default=int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "900")))
    p.add_argument("--max-context-len", type=int, default=int(os.environ.get("SMOKE_MAX_CONTEXT_LEN", "96000")))
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "8192")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.6")))
    p.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "0.95")))
    p.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "20")))
    p.add_argument("--top-logprobs", type=int, default=int(os.environ.get("TOP_LOGPROBS", "20")))
    p.add_argument(
        "--entropy-scope",
        default=os.environ.get("ENTROPY_SCOPE", "thinking"),
        choices=("thinking", "all"),
    )
    p.add_argument("--llm-timeout", type=float, default=float(os.environ.get("LLM_TIMEOUT", "600")))
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
    p.add_argument(
        "--eval",
        action="store_true",
        help="Run grader after agent exit (tmax: grade_tmax_inplace; scaleswe: run_evaluation)",
    )
    p.add_argument("--eval-timeout", type=int, default=600)
    p.add_argument("--require-exit-zero", action="store_true")
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable chat_template_kwargs.enable_thinking",
    )
    p.add_argument(
        "--reasoning-effort",
        default=os.environ.get("INFER_REASONING_EFFORT", ""),
        help="Optional chat_template_kwargs.reasoning_effort",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("INFER_LIMIT", "3")),
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
        default=int(os.environ.get("INFER_CONCURRENCY", "2")),
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
