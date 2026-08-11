"""Shared traj recording + dump helpers for harness infer scripts.

Produces the same per-sample layout as ``infer_opencode_sglang.py`` /
``opencode_smith_infer5_v3``::

    OUT_DIR/i000_<instance_id>/
      requests/req_XX.json
      trajectory.json / turns.jsonl / summary.json / patch.diff
      harness_trajectory.jsonl   # optional

Anthropic harnesses record ``anthropic_response.stop_reason``; OpenAI/Codex
also fills a mapped ``anthropic_response`` so dump consumers stay uniform, plus
``openai_response.finish_reason`` (``tool_calls`` / ``stop`` / ``length``).
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

from aiohttp import web

import offload
from slime.agent.adapters import AnthropicAdapter, OpenAIAdapter
from slime.agent.adapters.common import (
    Session,
    call_sglang_generate,
    _render_token_ids,
)
from slime.agent.parsing import parse_model_output

logger = logging.getLogger("traj_dump")


def turn_stop_reason(row: dict[str, Any]) -> str:
    """Prefer Anthropic stop_reason; fall back to OpenAI finish_reason."""
    ar = row.get("anthropic_response") or {}
    if isinstance(ar, dict) and ar.get("stop_reason"):
        return str(ar["stop_reason"])
    oai = row.get("openai_response") or {}
    if isinstance(oai, dict) and oai.get("finish_reason"):
        return str(oai["finish_reason"])
    return ""


def is_tool_turn(row: dict[str, Any]) -> bool:
    return turn_stop_reason(row) in ("tool_use", "tool_calls")


def _openai_finish_to_anthropic_stop(wire_finish: str) -> str:
    if wire_finish == "tool_calls":
        return "tool_use"
    if wire_finish == "length":
        return "max_tokens"
    return "end_turn"


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if len(rows) >= limit:
            break
    return rows


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


def _openai_tool_calls_from_anthropic_blocks(blocks: list[dict] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, b in enumerate(blocks or []):
        if not isinstance(b, dict) or b.get("type") != "tool_use":
            continue
        out.append(
            {
                "id": str(b.get("id") or f"call_{i}"),
                "type": "function",
                "function": {
                    "name": str(b.get("name") or "tool"),
                    "arguments": offload._arguments_as_openai_json(b.get("input")),
                },
            }
        )
    return out


def _split_system_and_history(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            system_parts.append(str(m.get("content") or ""))
    return "\n\n".join(p for p in system_parts if p), list(messages)


def write_request_file(requests_dir: Path, record: dict[str, Any]) -> Path:
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


def save_outputs(
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
        write_request_file(requests_dir, row)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "created_at": summary.get("created_at"),
                "mode": summary.get("mode", "anthropic_harness_sglang"),
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
            oai = row.get("openai_response") or {}
            req = row.get("openai_request") or {}
            compact = {
                "turn_index": row["turn_index"],
                "elapsed_sec": row["elapsed_sec"],
                "stop_reason": turn_stop_reason(row),
                "content": oai.get("content"),
                "reasoning_content": oai.get("reasoning_content"),
                "tool_calls": oai.get("tool_calls"),
                "usage": oai.get("usage"),
                "n_messages": len(req.get("messages") or []),
                "n_tools": len(req.get("tools") or []),
                "request_file": f"requests/req_{int(row['turn_index']):02d}.json",
            }
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")
    if harness_traj:
        (out_dir / "harness_trajectory.jsonl").write_text(harness_traj, encoding="utf-8")
    (out_dir / "patch.diff").write_text(patch_diff or "", encoding="utf-8")
    print(f"[infer] wrote {out_dir} ({len(traj_turns)} turns under requests/)", flush=True)


class TrajRecordingAnthropicAdapter(AnthropicAdapter):
    """AnthropicAdapter that records each /v1/messages turn for traj dumps."""

    log_prefix = "traj_adapter"

    def __init__(self, *args, model_name: str = "slime-actor", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name
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
            self.logger.debug("[%s] sid=%s request after session closed", self.log_prefix, sid)
            return web.Response(status=503, text="session closed")
        capped = self._check_turn_cap(sid)
        if capped is not None:
            return capped

        tok = self.tokenizer
        s = self.store.setdefault(sid, Session())
        task = asyncio.current_task()
        self.inflight.setdefault(sid, set()).add(task)
        t0 = time.monotonic()
        try:
            translated, tools_schema = self._translate(body)
            prompt_ids = _render_token_ids(translated, tok, tools=tools_schema, add_generation_prompt=True)

            turn = await call_sglang_generate(prompt_ids, s, body, adapter=self, session_id=sid)

            raw_output = tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
            parsed = parse_model_output(
                raw_output,
                tools_schema=tools_schema,
                tool_parser_name=self.tool_parser,
                reasoning_parser_name=self.reasoning_parser,
            )
            reply = self._build_reply(parsed, turn.finish_reason, translated, tools_schema)
            turn = dataclasses.replace(turn, ill_formed=parsed.ill_formed)
            reply = await self._postprocess_reply(
                reply,
                raw_output=raw_output,
                translated=translated,
                tools_schema=tools_schema,
                turn=turn,
                session=s,
                sid=sid,
            )

            in_tok, out_tok = len(prompt_ids), len(turn.output_ids)
            stream = body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")

            try:
                response = await self._respond(request, body, reply, in_tok, out_tok, stream)
            except (ConnectionResetError, asyncio.CancelledError) as e:
                self.logger.warning(
                    "[%s] sid=%s client disconnected before response flush: %s after %.1fs",
                    self.log_prefix,
                    sid,
                    type(e).__name__,
                    time.monotonic() - t0,
                )
                if isinstance(e, asyncio.CancelledError):
                    raise
                return web.Response(status=499, text="client disconnected")

            blocks, stop_reason = reply.wire
            manager_message = reply.manager_message
            openai_messages = _translated_to_openai_messages(translated)
            max_tokens = int(
                body.get("max_tokens")
                or (s.sampling_defaults or {}).get("max_new_tokens")
                or 4096
            )
            tool_calls = _openai_tool_calls_from_anthropic_blocks(blocks)
            usage = {
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
                "prompt_tokens_details": None,
                "reasoning_tokens": None,
            }

            bucket = self.traj_by_sid.setdefault(sid, [])
            record = {
                "sid": sid,
                "turn_index": len(bucket),
                "elapsed_sec": round(time.monotonic() - t0, 3),
                "openai_request": {
                    "messages": _jsonable(openai_messages),
                    "tools": _jsonable(offload._normalize_openai_tools(tools_schema)),
                    "max_tokens": max_tokens,
                    "model": body.get("model") or self.model_name,
                },
                "openai_response": {
                    "content": manager_message.get("content") or "",
                    "reasoning_content": manager_message.get("reasoning_content") or "",
                    "tool_calls": _jsonable(tool_calls),
                    "usage": usage,
                },
                "anthropic_response": {
                    "content": _jsonable(blocks),
                    "stop_reason": stop_reason,
                },
                "manager_message": _jsonable(manager_message),
            }
            bucket.append(record)
            req_dir = self.sid_requests_dir.get(sid)
            if req_dir is not None:
                path = write_request_file(req_dir, record)
                logger.info("[%s] wrote %s (messages=%d)", self.log_prefix, path, len(openai_messages))

            self._run_debug_callback(
                sid,
                translated,
                tools_schema,
                reply.manager_message,
                turn,
            )
            self.manager.record_turn(
                sid,
                turn=turn,
                prompt_messages=translated,
                response_message=reply.manager_message,
                metadata={"sid": sid},
            )
            return response
        finally:
            self.inflight.get(sid, set()).discard(task)


class TrajRecordingOpenAIAdapter(OpenAIAdapter):
    """OpenAIAdapter that records each /v1/chat/completions turn for traj dumps."""

    log_prefix = "traj_openai_adapter"

    def __init__(self, *args, model_name: str = "slime-actor", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name
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
            self.logger.debug("[%s] sid=%s request after session closed", self.log_prefix, sid)
            return web.Response(status=503, text="session closed")
        capped = self._check_turn_cap(sid)
        if capped is not None:
            return capped

        tok = self.tokenizer
        s = self.store.setdefault(sid, Session())
        task = asyncio.current_task()
        self.inflight.setdefault(sid, set()).add(task)
        t0 = time.monotonic()
        try:
            translated, tools_schema = self._translate(body)
            prompt_ids = _render_token_ids(translated, tok, tools=tools_schema, add_generation_prompt=True)

            turn = await call_sglang_generate(prompt_ids, s, body, adapter=self, session_id=sid)

            raw_output = tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
            parsed = parse_model_output(
                raw_output,
                tools_schema=tools_schema,
                tool_parser_name=self.tool_parser,
                reasoning_parser_name=self.reasoning_parser,
            )
            reply = self._build_reply(parsed, turn.finish_reason, translated, tools_schema)
            turn = dataclasses.replace(turn, ill_formed=parsed.ill_formed)
            reply = await self._postprocess_reply(
                reply,
                raw_output=raw_output,
                translated=translated,
                tools_schema=tools_schema,
                turn=turn,
                session=s,
                sid=sid,
            )

            in_tok, out_tok = len(prompt_ids), len(turn.output_ids)
            stream = body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")

            try:
                response = await self._respond(request, body, reply, in_tok, out_tok, stream)
            except (ConnectionResetError, asyncio.CancelledError) as e:
                self.logger.warning(
                    "[%s] sid=%s client disconnected before response flush: %s after %.1fs",
                    self.log_prefix,
                    sid,
                    type(e).__name__,
                    time.monotonic() - t0,
                )
                if isinstance(e, asyncio.CancelledError):
                    raise
                return web.Response(status=499, text="client disconnected")

            wire_message, wire_finish = reply.wire
            manager_message = reply.manager_message
            max_tokens = int(
                body.get("max_completion_tokens")
                or body.get("max_tokens")
                or body.get("max_output_tokens")
                or (s.sampling_defaults or {}).get("max_new_tokens")
                or 4096
            )
            usage = {
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
                "prompt_tokens_details": None,
                "reasoning_tokens": None,
            }
            wire_tcs = wire_message.get("tool_calls") if isinstance(wire_message, dict) else None
            mgr_tcs = manager_message.get("tool_calls") if isinstance(manager_message, dict) else None
            tool_calls = wire_tcs if wire_tcs else mgr_tcs

            bucket = self.traj_by_sid.setdefault(sid, [])
            record = {
                "sid": sid,
                "turn_index": len(bucket),
                "elapsed_sec": round(time.monotonic() - t0, 3),
                "openai_request": {
                    "messages": _jsonable(body.get("messages") or _translated_to_openai_messages(translated)),
                    "tools": _jsonable(body.get("tools") or offload._normalize_openai_tools(tools_schema)),
                    "max_tokens": max_tokens,
                    "model": body.get("model") or self.model_name,
                },
                "openai_response": {
                    "content": (manager_message.get("content") if isinstance(manager_message, dict) else "") or "",
                    "reasoning_content": (
                        (wire_message.get("reasoning_content") if isinstance(wire_message, dict) else None)
                        or (manager_message.get("reasoning_content") if isinstance(manager_message, dict) else None)
                        or ""
                    ),
                    "tool_calls": _jsonable(tool_calls or []),
                    "finish_reason": wire_finish,
                    "usage": usage,
                },
                # Mapped Anthropic-shaped stop_reason so dump consumers stay uniform.
                "anthropic_response": {
                    "content": [],
                    "stop_reason": _openai_finish_to_anthropic_stop(str(wire_finish)),
                },
                "manager_message": _jsonable(manager_message),
            }
            bucket.append(record)
            req_dir = self.sid_requests_dir.get(sid)
            if req_dir is not None:
                path = write_request_file(req_dir, record)
                n_msg = len(record["openai_request"]["messages"] or [])
                logger.info("[%s] wrote %s (messages=%d)", self.log_prefix, path, n_msg)

            self._run_debug_callback(
                sid,
                translated,
                tools_schema,
                reply.manager_message,
                turn,
            )
            self.manager.record_turn(
                sid,
                turn=turn,
                prompt_messages=translated,
                response_message=reply.manager_message,
                metadata={"sid": sid},
            )
            return response
        finally:
            self.inflight.get(sid, set()).discard(task)
