"""Shared adapter primitives for token-capturing agent rollouts.

A protocol adapter (Anthropic / OpenAI) subclasses BaseAdapter and fills in the
wire-specific hooks (_register_routes, _session_id, _translate, _build_reply,
_respond, and optionally _preprocess_body) plus a few class attributes (logger,
log_prefix, max_token_keys, stop_keys). The session lifecycle, per-sid turn cap,
inflight-task bookkeeping and the one-turn _run_turn pipeline are inherited.

flatten_content, tool_call_dict and manager_finish_reason cover the parts both
protocols handle identically.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from slime.agent.chrome_trace import chrome_span, session_trace_ctx
from slime.agent.parsing import parse_model_output
from slime.agent.trajectory import TrajectoryManager, TurnRecord


__all__ = ["TurnRecord"]


@dataclasses.dataclass
class Session:
    """Per-sid adapter state: sampling defaults and context budget.

    Trajectory state lives in the shared TrajectoryManager (BaseAdapter.manager),
    not here. ``offload_stats`` is an optional side channel for mid-turn LLM
    offload accounting (filled by coding-agent adapters when enabled).
    ``timing`` holds optional Chrome Trace event state (``trace_events``, ``tid``,
    ``n_turns``, ``n_offloads``) when the caller enables rollout timelines.

    When ``preserve_reasoning_history`` is on, ``assistant_history`` /
    ``last_turn_ids`` keep the server's canonical structured replies and exact
    ``prompt_ids + output_ids`` so later turns append instead of re-tokenizing.
    """

    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    offload_stats: dict = dataclasses.field(default_factory=dict)
    timing: dict = dataclasses.field(default_factory=dict)
    assistant_history: list[dict] = dataclasses.field(default_factory=list)
    last_turn_ids: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Reply:
    """Output of an adapter's _build_reply, consumed by _run_turn.

    manager_message and finish_reason feed record_turn and the debug callback;
    wire is opaque to the pipeline and only the adapter's own _respond reads it.
    """

    manager_message: dict
    finish_reason: str
    wire: Any


def _render_token_ids(
    messages: list[dict],
    tokenizer,
    *,
    tools: list[dict] | None,
    add_generation_prompt: bool = True,
    chat_template: str | None = None,
) -> list[int]:
    """Render a chat-message list to token ids with the served chat template."""
    kwargs: dict[str, Any] = {
        "tools": tools,
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    enc = tokenizer.apply_chat_template(messages, **kwargs)
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(ids)


_QWEN_REASONING_HISTORY_BRANCH = "{%- if loop.index0 > ns.last_query_index %}"


def _load_tokenizer_chat_template(tokenizer) -> str | None:
    """Return the tokenizer's chat template string, including jinja-on-disk."""
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and template:
        return template
    name = getattr(tokenizer, "name_or_path", None)
    if not name:
        return None
    jinja = Path(name) / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    return None


def _reasoning_preserving_chat_template(tokenizer) -> str | None:
    """Return the Qwen chat template variant needed for append-only RL turns.

    Qwen3/3.5 normally drops reasoning from assistant messages before the most
    recent user message. That inference-time compaction rewrites already sampled
    tokens, which breaks a multi-turn RL trajectory. In this explicit mode all
    available ``reasoning_content`` is rendered while the normal open generation
    prompt remains unchanged.

    Returns ``None`` when the served template has no Qwen reasoning branch; the
    caller can still run append-only token continuation with the default template.
    """
    template = _load_tokenizer_chat_template(tokenizer)
    if not isinstance(template, str) or _QWEN_REASONING_HISTORY_BRANCH not in template:
        return None
    return template.replace(_QWEN_REASONING_HISTORY_BRANCH, "{%- if true %}", 1)


def _assistant_visible_signature(message: dict) -> tuple[Any, Any]:
    """Content + tool_calls, ignoring wire-only correlation ids."""
    normalized = {
        k: v
        for k, v in message.items()
        if k in {"content", "tool_calls"}
    }
    tcs = normalized.get("tool_calls")
    if isinstance(tcs, list):
        normalized["tool_calls"] = [
            ({k: v for k, v in tc.items() if k != "id"} if isinstance(tc, dict) else tc) for tc in tcs
        ]
    return normalized.get("content"), normalized.get("tool_calls")


def _restore_canonical_assistant_history(messages: list[dict], assistant_history: list[dict]) -> int:
    """Replace client-echoed assistant turns with the server's canonical copy.

    Clients may omit reasoning and normalize text or tool-call arguments when
    replaying a response. The adapter generated those turns and owns their exact
    structured form, so align assistant messages from the newest turn backwards
    and restore them wholesale. User and tool messages remain client-owned.
    """
    positions = [i for i, message in enumerate(messages) if message.get("role") == "assistant"]
    restored = 0
    for position, canonical in zip(reversed(positions), reversed(assistant_history), strict=False):
        if messages[position] != canonical:
            messages[position] = copy.deepcopy(canonical)
            restored += 1
    return restored


def _assert_append_only_prompt(previous_turn_ids: list[int], prompt_ids: list[int]) -> None:
    """Require a new prompt to preserve every token sampled in the prior turn."""
    if not previous_turn_ids:
        return
    if prompt_ids[: len(previous_turn_ids)] == previous_turn_ids:
        return
    common = 0
    limit = min(len(previous_turn_ids), len(prompt_ids))
    while common < limit and previous_turn_ids[common] == prompt_ids[common]:
        common += 1
    raise RuntimeError(
        "non-append-only agent prompt: "
        f"previous_tokens={len(previous_turn_ids)} prompt_tokens={len(prompt_ids)} common_prefix={common}"
    )


def _continue_from_canonical_turn(
    messages: list[dict],
    tokenizer,
    *,
    tools: list[dict] | None,
    chat_template: str | None,
    rendered_prompt_ids: list[int],
    previous_turn_ids: list[int],
    previous_response_message: dict,
) -> list[int]:
    """Append only the client messages added after the last sampled response.

    A parsed assistant response is semantically equivalent to the sampled text,
    but tool parsers and chat templates can normalize whitespace or argument
    formatting. The sampled token ids are therefore the canonical history. We
    use message identity only to locate the assistant boundary in the freshly
    rendered prompt, then append the new suffix to those canonical ids.
    """
    assistant_positions = [i for i, message in enumerate(messages) if message.get("role") == "assistant"]
    if not assistant_positions:
        raise RuntimeError("append-only agent prompt has no replayed assistant message")
    boundary = assistant_positions[-1]
    if _assistant_visible_signature(messages[boundary]) != _assistant_visible_signature(previous_response_message):
        raise RuntimeError("latest replayed assistant does not match the previous sampled response")

    rendered_through_assistant = _render_token_ids(
        messages[: boundary + 1],
        tokenizer,
        tools=tools,
        add_generation_prompt=False,
        chat_template=chat_template,
    )
    if rendered_prompt_ids[: len(rendered_through_assistant)] != rendered_through_assistant:
        raise RuntimeError("chat template rewrote messages before the latest assistant boundary")

    prompt_ids = previous_turn_ids + rendered_prompt_ids[len(rendered_through_assistant) :]
    _assert_append_only_prompt(previous_turn_ids, prompt_ids)
    return prompt_ids


def flatten_content(c: Any) -> str:
    """Flatten a wire content value into a chat-template string.

    Handles both Anthropic and OpenAI block shapes. A non-list value (str /
    dict / other) is returned via str() unchanged.
    """
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return str(c)
    parts: list[str] = []
    for b in c:
        if isinstance(b, str):
            parts.append(b)
            continue
        if not isinstance(b, dict):
            parts.append(str(b))
            continue
        t = b.get("type")
        if t in {"text", "input_text", "output_text"}:
            parts.append(b.get("text", ""))
        elif t == "tool_result":
            parts.append(flatten_content(b.get("content")))
        elif t in {"image", "image_url", "input_image"}:
            parts.append("[image omitted]")
        elif "content" in b:
            parts.append(flatten_content(b.get("content")))
        elif "text" in b:
            parts.append(str(b.get("text") or ""))
    return "\n".join(p for p in parts if p)


def tool_call_dict(name: str, arguments: dict | None) -> dict:
    """Canonical OpenAI-shape tool call stored on manager_message.

    arguments stays a dict (not a JSON string): the chat template needs a
    mapping, and the trajectory manager matches history by dict equality, so a
    sampled leaf and its replayed echo compare equal regardless of key order.
    The wire-only tool-call id is dropped for the same reason.
    """
    return {"type": "function", "function": {"name": name, "arguments": arguments or {}}}


def manager_finish_reason(tool_uses: list[dict], raw_finish: str) -> str:
    """Finish reason stored on the manager turn: tool_calls if the turn called a
    tool, else the raw sglang finish."""
    return "tool_calls" if tool_uses else (raw_finish or "stop")


class BaseAdapter:
    """Base HTTP adapter: session lifecycle plus the shared one-turn pipeline.

    See the module docstring for the class attributes and hooks a subclass must
    supply; everything else is inherited.
    """

    logger: logging.Logger = logging.getLogger(__name__)
    log_prefix: str = "adapter"
    # body keys that cap max_new_tokens and carry stop sequences, in priority order
    max_token_keys: tuple[str, ...] = ()
    stop_keys: tuple[str, ...] = ()
    manager: Any

    def __init__(
        self,
        *,
        tokenizer,
        sglang_url,
        tool_parser=None,
        reasoning_parser=None,
        preserve_reasoning_history: bool = False,
        max_turns_per_sid: int | None = None,
        fork_threshold_tokens: int | None = None,
        debug_callback: Callable[..., None] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.sglang_url = sglang_url.rstrip("/") if isinstance(sglang_url, str) else sglang_url
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser
        self.preserve_reasoning_history = bool(preserve_reasoning_history)
        self.chat_template = (
            _reasoning_preserving_chat_template(tokenizer) if self.preserve_reasoning_history else None
        )
        if self.preserve_reasoning_history and self.chat_template is None:
            self.logger.warning(
                "[%s] preserve_reasoning_history=1 but no Qwen reasoning-history branch in "
                "chat template; append-only token continuation still enabled",
                self.log_prefix,
            )
        self.store: dict[str, Any] = {}
        self.inflight: dict[str, set[asyncio.Task]] = {}
        self.closed: set[str] = set()
        self.app = web.Application(client_max_size=64 * 1024 * 1024)

        # one manager shared across all sids; per-sid trees live inside it.
        # fork_threshold_tokens left None means the manager uses its own default.
        mgr_kwargs: dict[str, int] = {}
        if fork_threshold_tokens is not None:
            mgr_kwargs["fork_threshold_tokens"] = fork_threshold_tokens
        self.manager = TrajectoryManager(**mgr_kwargs)

        self.debug_callback: Callable[..., None] | None = debug_callback
        # per-sid turn cap: return 429 to kill the run once exceeded
        self.max_turns_per_sid: int | None = max_turns_per_sid
        self._sid_turn_count: dict[str, int] = {}

        self.app.router.add_get("/healthz", _health)
        self.app.router.add_get("/v1/models", _health)
        self._register_routes(self.app)

    # -- wire hooks (subclass overrides) -------------------------------------

    def _register_routes(self, app: web.Application) -> None:
        """Register the protocol's POST route(s) and bind self._run_turn."""
        raise NotImplementedError

    def _session_id(self, request: web.Request, body: dict) -> str:
        raise NotImplementedError

    def _preprocess_body(self, body: dict) -> None:
        """Mutate the parsed body in place before sid resolution (default no-op)."""

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        """Return (chat_messages, tools_schema) from the wire body."""
        raise NotImplementedError

    def _build_reply(self, parsed, raw_finish: str, translated: list[dict], tools_schema: list[dict] | None) -> Reply:
        """Pack parsed model output into a Reply."""
        raise NotImplementedError

    async def _postprocess_reply(
        self,
        reply: Reply,
        *,
        raw_output: str,
        translated: list[dict],
        tools_schema: list[dict] | None,
        turn: TurnRecord,
        session: Session,
        sid: str,
    ) -> Reply:
        """Optional hook after ``_build_reply`` (e.g. mid-turn LLM offload).

        Default is a no-op. Subclasses may mutate ``session.offload_stats``,
        amend ``reply``, and/or extend ``turn.output_ids`` in place (e.g. GLM
        offload text with ``output_loss_mask=0``) before the client flush.
        """
        return reply

    async def _respond(
        self,
        request: web.Request,
        body: dict,
        reply: Reply,
        in_tok: int,
        out_tok: int,
        stream: bool,
    ) -> web.StreamResponse:
        raise NotImplementedError

    # -- session lifecycle ---------------------------------------------------

    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
    ) -> None:
        """Register a fresh per-sid Session; sids must be unique."""
        if sid in self.store:
            raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
        self.store[sid] = Session(
            sampling_defaults=dict(sampling_defaults or {}),
            max_context_tokens=int(max_context_tokens or 0),
        )

    async def shutdown_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        """Mark a sid closed and drain its in-flight turn tasks."""
        self.closed.add(sid)
        tasks = [t for t in self.inflight.pop(sid, ()) if not t.done()]
        if not tasks:
            return

        async def _drain() -> None:
            _, pending = await asyncio.wait(tasks, timeout=wait_timeout)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        loop = tasks[0].get_loop()
        try:
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_drain(), loop))
        except Exception:
            self.logger.exception("[%s] sid=%s shutdown drain failed", self.log_prefix, sid)

    async def finish_session(
        self,
        sid: str,
        *,
        base_sample,
        reward: float = 0.0,
        extra_metadata: dict | None = None,
        wait_timeout: float = 5.0,
    ) -> list:
        """Drain a session's trajectory into fully-formed Sample objects.

        Waits out in-flight requests for the sid, linearises the per-sid tree,
        then decodes each sample's trained tail into .response (the manager is
        tokenizer-free, so the adapter that owns the tokenizer fills this in).
        Idempotent: a second call for an already-popped sid returns [].
        """
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        session = self.store.pop(sid, None)
        max_sample_tokens = int(getattr(session, "max_context_tokens", 0) or 0) if session is not None else 0
        samples = self.manager.get_trajectory(
            sid,
            base_sample=base_sample,
            reward=reward,
            extra_metadata=extra_metadata,
            max_sample_tokens=max_sample_tokens,
        )
        for s in samples:
            rlen = int(s.response_length or 0)
            s.response = (
                self.tokenizer.decode(s.tokens[-rlen:], skip_special_tokens=False) if rlen and s.tokens else ""
            )
        return samples

    async def drop_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        self.store.pop(sid, None)
        self.manager.drop_session(sid)

    # -- shared request pipeline ---------------------------------------------

    def _check_turn_cap(self, sid: str) -> web.Response | None:
        """Enforce max_turns_per_sid, returning a 429 response once exceeded.

        Increments the per-sid counter as a side effect when under the cap.
        """
        cap = self.max_turns_per_sid
        if cap is None:
            return None
        prior = self._sid_turn_count.get(sid, 0)
        if prior >= cap:
            self.logger.warning("[%s] sid=%s exceeded max_turns_per_sid=%d; killing run", self.log_prefix, sid, cap)
            session = self.store.get(sid)
            if session is not None:
                stats = session.offload_stats if isinstance(session.offload_stats, dict) else None
                if stats is not None:
                    stats["max_steps_reached"] = True
            return web.json_response(
                {
                    "error": {
                        "type": "rate_limit_error",
                        "message": (f"adapter: sid {sid!r} exceeded max_turns_per_sid={cap}; killing run"),
                    }
                },
                status=429,
            )
        self._sid_turn_count[sid] = prior + 1
        return None

    def _run_debug_callback(self, sid, translated, tools_schema, manager_message, turn) -> None:
        """Run the optional debug-only data dump callback; unset in production."""
        callback = self.debug_callback
        if callback is None:
            return
        try:
            callback(sid, translated, tools_schema, manager_message, turn)
        except Exception:
            self.logger.exception("debug_callback failed (sid=%s)", sid)

    async def _run_turn(self, request: web.Request) -> web.StreamResponse:
        """One agent round: translate -> SLM /generate -> parse -> postprocess -> respond.

        ``_postprocess_reply`` runs before the client flush so coding-agent offload
        can call a remote GLM and compose a *complete* assistant reply for this
        round. Each subsequent agent tool/chat round repeats the same pipeline.
        """
        body = await request.json()
        self._preprocess_body(body)
        sid = self._session_id(request, body)
        if sid in self.closed:  # session drained; refuse stragglers
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
        events, tid, timing = session_trace_ctx(s)
        turn_idx = int((timing or {}).get("n_turns", 0) or 0)
        if timing is not None:
            timing["n_turns"] = turn_idx + 1
            timing["current_turn"] = turn_idx
        try:
            with chrome_span(
                events,
                "adapter_turn",
                cat="adapter",
                tid=tid,
                args={"turn": turn_idx, "session_id": sid},
            ):
                translated, tools_schema = self._translate(body)
                # A harness may echo a prior assistant turn re-rendered rather than
                # byte-identical; render from what we generated so the turn stays CLEAN.
                translated = self.manager.restore_generated_messages(sid, translated, tools_schema)
                if self.preserve_reasoning_history:
                    restored = _restore_canonical_assistant_history(translated, s.assistant_history)
                    if restored:
                        self.logger.debug(
                            "[%s] sid=%s restored canonical history for %d assistant turn(s)",
                            self.log_prefix,
                            sid,
                            restored,
                        )
                rendered_prompt_ids = _render_token_ids(
                    translated,
                    tok,
                    tools=tools_schema,
                    add_generation_prompt=True,
                    chat_template=self.chat_template,
                )
                if self.preserve_reasoning_history and s.last_turn_ids:
                    prompt_ids = _continue_from_canonical_turn(
                        translated,
                        tok,
                        tools=tools_schema,
                        chat_template=self.chat_template,
                        rendered_prompt_ids=rendered_prompt_ids,
                        previous_turn_ids=s.last_turn_ids,
                        previous_response_message=s.assistant_history[-1],
                    )
                else:
                    prompt_ids = rendered_prompt_ids

                # 1) Always SLM first for this agent round.
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
                # 2) Optional GLM offload + 3) compose complete output for the agent.
                # Must finish before _respond so the agent never sees a partial SLM-only turn.
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

                # Flush the response before recording the trajectory: a client that
                # disconnected during generation makes _respond raise here, and we
                # must not record a turn the client never received.
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

                self._run_debug_callback(
                    sid,
                    translated,
                    tools_schema,
                    reply.manager_message,
                    turn,
                )

                superseded = self.manager.record_turn(
                    sid,
                    turn=turn,
                    prompt_messages=translated,
                    response_message=reply.manager_message,
                    metadata={"sid": sid},
                )
                if self.preserve_reasoning_history:
                    # Tip regeneration drops the abandoned leaf in the manager;
                    # mirror that in canonical history so restore stays aligned
                    # with what the client actually kept.
                    if superseded and s.assistant_history:
                        s.assistant_history.pop()
                    s.assistant_history.append(copy.deepcopy(reply.manager_message))
                    s.last_turn_ids = list(prompt_ids) + list(turn.output_ids)
                return response
        finally:
            self.inflight.get(sid, set()).discard(task)


def sid_from_bearer(request: web.Request) -> str | None:
    """sid from the Authorization: Bearer <sid> header, or None if absent."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def sid_from_body(body: dict | None) -> str | None:
    """sid from the OpenAI-shape body (metadata.session_id / user), or None."""
    if not body:
        return None
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and metadata.get("session_id"):
        return str(metadata["session_id"])
    if body.get("user"):
        return str(body["user"])
    return None


def _sampling_params(session: Any, body: dict, *, max_token_keys: tuple[str, ...], stop_keys: tuple[str, ...]) -> dict:
    sp: dict[str, Any] = {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "no_stop_trim": True,
        "max_new_tokens": 4096,
        **(session.sampling_defaults or {}),
    }

    for key in max_token_keys:
        if body.get(key) is not None:
            sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", body[key])), int(body[key]))
            break

    for src_k, dst_k in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if src_k in body:
            sp[dst_k] = body[src_k]

    for key in stop_keys:
        if body.get(key):
            sp["stop"] = body[key]
            break

    return sp


def _offload_constrained_decode_api() -> Any | None:
    """Lazy import so non-offload adapters do not depend on examples.*."""
    if os.environ.get("SLIME_AGENT_OFFLOAD", "").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    try:
        from examples.coding_agent_rl import offload as offload_mod
    except ImportError:
        return None
    if not offload_mod.constrained_decode_enabled():
        return None
    return offload_mod


async def _sglang_generate_once(
    *,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    sglang_url: str,
    headers: dict[str, str] | None,
    logger: logging.Logger,
    log_prefix: str,
    session_id: str | None,
) -> tuple[list[int], list[float], str]:
    """POST ``/generate`` once; return ``(output_ids, output_log_probs, finish_reason)``."""
    rid = uuid.uuid4().hex
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as sess, sess.post(
            f"{sglang_url}/generate",
            json={
                "rid": rid,
                "input_ids": prompt_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            },
            headers=headers,
        ) as r:
            if r.status >= 400:
                text = await r.text()
                logger.warning(
                    "[%s] sid=%s rid=%s sglang upstream %d: %.200s",
                    log_prefix,
                    session_id,
                    rid,
                    r.status,
                    text,
                )
                raise RuntimeError(f"sglang upstream {r.status}: {text[:400]}")
            data = await r.json(content_type=None)
        meta = data.get("meta_info") or {}
        output_token_logprobs = meta.get("output_token_logprobs") or []
        output_ids = [x[1] for x in output_token_logprobs]
        output_log_probs = [float(x[0]) for x in output_token_logprobs]
        finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
        return output_ids, output_log_probs, finish
    except (asyncio.CancelledError, aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.debug("[%s] sid=%s rid=%s turn aborted: %s", log_prefix, session_id, rid, type(e).__name__)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s2:
                await s2.post(f"{sglang_url}/abort_request", json={"rid": rid})
        except Exception:
            pass
        raise


async def call_sglang_generate(
    prompt_ids: list[int],
    session: Any,
    body: dict,
    *,
    adapter: BaseAdapter,
    session_id: str | None = None,
) -> TurnRecord:
    """POST one turn to sglang /generate and pack the reply into a TurnRecord.

    When coding-agent offload constrained decode is enabled, runs Free phase
    (ban CLOSE, stop on OPEN) then an ebnf continuation for digit+CLOSE, merging
    tokens/logprobs into one :class:`TurnRecord`.

    Module-level (not a method) so tests can monkeypatch it.
    """
    logger = adapter.logger
    sp = _sampling_params(session, body, max_token_keys=adapter.max_token_keys, stop_keys=adapter.stop_keys)

    if session.max_context_tokens > 0:
        remaining_context = session.max_context_tokens - len(prompt_ids)
        if remaining_context <= 0:
            logger.warning(
                "[%s] sid=%s prompt exceeds max_context_tokens (%d >= %d)",
                adapter.log_prefix,
                session_id,
                len(prompt_ids),
                session.max_context_tokens,
            )
            return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[], finish_reason="length")
        sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", remaining_context)), remaining_context)

    offload_mod = _offload_constrained_decode_api()
    if offload_mod is not None:
        sp = offload_mod.apply_free_phase_constrained_sampling(sp)

    sglang_url = adapter.sglang_url
    headers = {"X-SMG-Routing-Key": session_id} if session_id and session_id != "default" else None
    events, tid, timing = session_trace_ctx(session)
    turn_idx = int((timing or {}).get("current_turn", 0) or 0)
    post_offload = int((timing or {}).get("n_offloads", 0) or 0) > 0
    with chrome_span(
        events,
        "sglang_generate",
        cat="llm",
        tid=tid,
        args={"turn": turn_idx, "post_offload": post_offload, "session_id": session_id},
    ):
        output_ids, output_log_probs, finish = await _sglang_generate_once(
            prompt_ids=prompt_ids,
            sampling_params=sp,
            sglang_url=sglang_url,
            headers=headers,
            logger=logger,
            log_prefix=adapter.log_prefix,
            session_id=session_id,
        )

        if offload_mod is not None and output_ids and output_ids[-1] == offload_mod.offload_open_token_id():
            ebnf_sp = offload_mod.apply_ebnf_phase_constrained_sampling(sp)
            # Remaining context after free tokens (0 = unlimited).
            if session.max_context_tokens > 0:
                used = len(prompt_ids) + len(output_ids)
                rem = session.max_context_tokens - used
                if rem <= 0:
                    logger.warning(
                        "[%s] sid=%s no context left for offload ebnf continue",
                        adapter.log_prefix,
                        session_id,
                    )
                    rem = 0
                else:
                    ebnf_sp["max_new_tokens"] = min(int(ebnf_sp.get("max_new_tokens", rem)), rem)
            else:
                rem = 1  # proceed
            if rem > 0:
                cont_ids, cont_lps, finish = await _sglang_generate_once(
                    prompt_ids=list(prompt_ids) + list(output_ids),
                    sampling_params=ebnf_sp,
                    sglang_url=sglang_url,
                    headers=headers,
                    logger=logger,
                    log_prefix=adapter.log_prefix,
                    session_id=session_id,
                )
                output_ids = list(output_ids) + list(cont_ids)
                output_log_probs = list(output_log_probs) + list(cont_lps)

    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=output_ids,
        finish_reason=finish,
        output_log_probs=output_log_probs,
    )


async def _health(request: web.Request) -> web.Response:
    """Handler for /healthz and /v1/models readiness probes."""
    return web.json_response({"ok": True})
