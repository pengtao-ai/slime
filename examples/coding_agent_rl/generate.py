"""Coding-Agent RL: per-sample generate() function for slime.

    --custom-generate-function-path examples.coding_agent_rl.generate.generate

generate() is a four-stage orchestrator: swe.prepare_workspace + harness.run
-> swe.git_diff -> swe.run_evaluation -> adapter.finish_session. The (harness,
adapter) pair is chosen by the SWE_AGENT env var (claude_code | codex); see
_AGENTS below.
Sandbox-side work is split across three layers: the provider-agnostic sandbox
contract (slime.agent.sandbox), the swappable harness lifecycle
(slime.agent.harness), and the SWE task layer (examples.coding_agent_rl.swe --
dataset parsing, workspace prep, diff, eval). LLM plumbing (Anthropic / OpenAI
<-> SGLang /generate, token capture, segment split) is the matching
slime.agent.adapters adapter. swe.get_metadata documents the dataset row schema
and produces the md dict consumed below.

Chrome Trace timestamps (ph=B/E) are recorded on each sample under
``metadata["timeline"]`` and exported per rollout by
``examples.coding_agent_rl.log_rollout_timeline``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import secrets
import time
import traceback
import zlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from slime.agent.adapters import AnthropicAdapter, OpenAIAdapter
from slime.agent.adapters.common import Reply, Session
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.chrome_trace import chrome_span, ensure_session_timing
from slime.agent.harness import ClaudeCodeHarness, CodexHarness
from slime.agent.sandbox import make_sandbox
from slime.agent.trajectory import TurnRecord
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from . import offload, swe

logger = logging.getLogger(__name__)
logging.getLogger("e2b").setLevel(logging.WARNING)


class _OffloadMixin:
    """Per-turn pipeline: agent -> SLM -> (optional GLM) -> complete reply -> agent.

    Every Claude Code / Codex ``/v1/messages`` (or chat) round hits this hook
    *before* the adapter flushes the response. Training still uses only SLM
    ``output_ids``; the agent always sees the composed full assistant turn.

    Offload instructions are appended to the request ``system`` after Claude
    Code's full system (incl. ``gitStatus``).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sid_repo_state: dict[str, dict[str, Any]] = {}
        self._sid_turn_git_diffs: dict[str, list[dict[str, Any]]] = {}

    def bind_repo_state(self, sid: str, *, sb: Any, workdir: str) -> None:
        self._sid_repo_state[sid] = {"sb": sb, "workdir": workdir}

    def unbind_repo_state(self, sid: str) -> None:
        self._sid_repo_state.pop(sid, None)

    def pop_turn_git_diffs(self, sid: str) -> list[dict[str, Any]]:
        return self._sid_turn_git_diffs.pop(sid, [])

    async def _capture_turn_git_diff(self, sid: str, session: Session) -> None:
        state = self._sid_repo_state.get(sid)
        if not state:
            return
        turn_index = int(((session.timing or {}).get("current_turn", 0) or 0))
        record = {"turn_index": turn_index, "git_diff": ""}
        try:
            record["git_diff"] = await swe.capture_turn_git_diff(state["sb"], state["workdir"])
        except Exception as exc:
            record["capture_error"] = f"{type(exc).__name__}: {exc}"
        self._sid_turn_git_diffs.setdefault(sid, []).append(record)

    def _preprocess_body(self, body: dict) -> None:
        super()._preprocess_body(body)
        offload.inject_offload_into_request_body(body)

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
        reply = await offload.apply_offload_if_needed(
            reply,
            raw_output=raw_output,
            translated=translated,
            turn=turn,
            session=session,
            sid=sid,
            tokenizer=self.tokenizer,
            tools_schema=tools_schema,
        )
        await self._capture_turn_git_diff(sid, session)
        return reply


class CodingAnthropicAdapter(_OffloadMixin, AnthropicAdapter):
    pass


class CodingOpenAIAdapter(_OffloadMixin, OpenAIAdapter):
    pass


_AGENTS = {
    "claude_code": (ClaudeCodeHarness, CodingAnthropicAdapter),
    "codex": (CodexHarness, CodingOpenAIAdapter),
}
AGENT_NAME = os.environ.get("SWE_AGENT", "claude_code")
if AGENT_NAME not in _AGENTS:
    raise ValueError(f"SWE_AGENT={AGENT_NAME!r} not in {sorted(_AGENTS)}")
HARNESS_CLS, ADAPTER_CLS = _AGENTS[AGENT_NAME]

@dataclass(frozen=True)
class SweConfig:
    eval_protocol: str  # eval-path schema/grader (SWE_EVAL_PROTOCOL)
    train_protocol: str  # train-path schema/grader (SWE_TRAIN_PROTOCOL)
    adapter_public_host: str | None
    adapter_public_url: str | None
    adapter_bind_host: str
    adapter_port: int
    fork_merge_threshold: int | None
    agent_time_budget_sec: int
    eval_timeout_sec: int
    rollout_guard_sec: int
    boot_concurrency: int
    boot_retries: int

    @classmethod
    def from_env(cls) -> SweConfig:
        agent_time_budget = int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "1800"))
        eval_timeout = int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "600"))
        guard = int(os.environ.get("SWE_ROLLOUT_GUARD_SEC", "0") or 0) or (agent_time_budget + eval_timeout + 180)
        fork = int(v) if (v := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        return cls(
            eval_protocol=os.environ.get("SWE_EVAL_PROTOCOL", swe.PROTOCOL_SCALESWE),
            train_protocol=os.environ.get("SWE_TRAIN_PROTOCOL", swe.PROTOCOL_SCALESWE),
            adapter_public_host=os.environ.get("ADAPTER_PUBLIC_HOST"),
            adapter_public_url=(os.environ.get("ADAPTER_PUBLIC_URL") or "").rstrip("/") or None,
            adapter_bind_host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "18001")),
            fork_merge_threshold=fork,
            agent_time_budget_sec=agent_time_budget,
            eval_timeout_sec=eval_timeout,
            rollout_guard_sec=guard,
            boot_concurrency=int(os.environ.get("SWE_BOOT_CONCURRENCY", "16")),
            boot_retries=int(os.environ.get("SWE_BOOT_RETRIES", "2")),
        )


CONFIG = SweConfig.from_env()

_BOOT_SEM = asyncio.Semaphore(CONFIG.boot_concurrency)


def _sample_tid(sample: Sample, session_id: str) -> int:
    if sample.index is not None:
        return int(sample.index) + 1
    return (zlib.crc32(session_id.encode("utf-8")) & 0x7FFFFFFF) or 1


def _timeline_payload(
    *,
    tid: int,
    session_id: str,
    instance_id: str,
    events: list[dict[str, Any]],
    thread_name: str | None = None,
) -> dict[str, Any]:
    return {
        "tid": tid,
        "session_id": session_id,
        "instance_id": instance_id,
        "thread_name": thread_name or f"{instance_id}",
        "trace_events": events,
    }


def _attach_timeline(samples: list[Sample], timeline: dict[str, Any]) -> None:
    for s in samples:
        s.metadata = {**(s.metadata or {}), "timeline": timeline}


@asynccontextmanager
async def boot_agent_sandbox(
    image: str,
    instance_id: str,
    *,
    events: list[dict[str, Any]],
    tid: int,
) -> AsyncIterator:
    """Boot a fresh sandbox and install the selected harness toolchain.

    Create the sandbox from the dataset image, install Node 22 + the harness CLI
    from host tarballs, retry transient boot/install failures, and close the
    sandbox when the caller leaves the context.

    Records ``boot_wait`` (semaphore queue) and ``boot_sandbox`` (create+install)
    as separate Chrome Trace spans.
    """
    sb = None
    last_err: Exception | None = None
    for attempt in range(CONFIG.boot_retries):
        cand = make_sandbox(image)
        try:
            with chrome_span(events, "boot_wait", cat="outer", tid=tid, args={"instance_id": instance_id}):
                await _BOOT_SEM.acquire()
            try:
                with chrome_span(
                    events,
                    "boot_sandbox",
                    cat="outer",
                    tid=tid,
                    args={"instance_id": instance_id, "attempt": attempt + 1},
                ):
                    await cand.__aenter__()
                    logger.info(
                        "[coding_agent_rl] %s: sandbox_id=%s image=%s",
                        instance_id,
                        cand.sandbox_id,
                        image,
                    )
                    try:
                        await HARNESS_CLS().install_cli(cand)
                    except BaseException:
                        await cand.__aexit__(None, None, None)
                        raise
            finally:
                _BOOT_SEM.release()
            sb = cand
            break
        except Exception as e:
            last_err = e
            logger.warning(
                "[coding_agent_rl] %s: provision attempt %d/%d failed: %s: %s",
                instance_id,
                attempt + 1,
                CONFIG.boot_retries,
                type(e).__name__,
                str(e)[:200],
            )
            await asyncio.sleep(1 + attempt + random.random())
    if sb is None:
        assert last_err is not None
        raise last_err
    try:
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


class _AdapterService(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        if not CONFIG.adapter_public_url and not CONFIG.adapter_public_host:
            raise RuntimeError(
                "Set ADAPTER_PUBLIC_URL (full URL, e.g. a Cloudflare tunnel) or "
                "ADAPTER_PUBLIC_HOST (host sandboxes can reach) so the coding "
                "agent can dial back to the Anthropic adapter."
            )
        self.adapter = ADAPTER_CLS(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            fork_threshold_tokens=CONFIG.fork_merge_threshold,
        )
        # handler_cancellation=True so a client disconnect cancels the handler
        # coroutine, arming the fire-and-forget /abort_request in the adapter.
        # Otherwise a cancelled client leaves an inflight sglang /generate that
        # races the next release_memory_occupation and trips its idle assertion.
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=CONFIG.adapter_bind_host,
            port=CONFIG.adapter_port,
            thread_name="anthropic-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        # Prefer a full public URL (HTTPS reverse proxy / Cloudflare tunnel).
        # Otherwise fall back to http://HOST:PORT for in-cluster / LAN gateways.
        if CONFIG.adapter_public_url:
            self.adapter_url = CONFIG.adapter_public_url
        else:
            self.adapter_url = f"http://{CONFIG.adapter_public_host}:{self.app_handle.port}"
        logger.info(
            "[coding_agent_rl] tokenizer=%s adapter=%s max_context_len=%s tool_parser=%s reasoning_parser=%s",
            args.hf_checkpoint,
            self.adapter_url,
            self.max_context_len,
            self.tool_parser,
            self.reasoning_parser,
        )


async def generate(args, base_sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Per-sample agent function with wall-clock guard (see rollout_guard_sec)."""
    state = _AdapterService(args)
    raw_protocol = (base_sample.metadata or {}).get("protocol")
    protocol = raw_protocol or (CONFIG.eval_protocol if evaluation else CONFIG.train_protocol)
    md = swe.get_metadata(base_sample, protocol)
    instance_id = md["instance_id"]
    if not md["image"] or not md["workdir"]:
        return _abort_result(base_sample, "missing_image_or_workdir", instance_id)
    reason = swe.evaluability_check(md)
    if reason:
        return _abort_result(base_sample, f"unevaluatable:{reason}", instance_id)

    session_id = base_sample.session_id = _session_id(base_sample, instance_id)
    tid = _sample_tid(base_sample, session_id)
    group_index = base_sample.group_index
    thread_name = f"{instance_id}" + (f"#{group_index}" if group_index is not None else "")
    trace_events: list[dict[str, Any]] = []
    timeline = _timeline_payload(
        tid=tid,
        session_id=session_id,
        instance_id=instance_id,
        events=trace_events,
        thread_name=thread_name,
    )

    state.adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        max_context_tokens=state.max_context_len,
    )
    session_obj = state.adapter.store[session_id]
    ensure_session_timing(session_obj, tid=tid, events=trace_events)

    t0 = time.time()
    result_samples: list[Sample] | None = None
    try:
        async with asyncio.timeout(CONFIG.rollout_guard_sec):
            async with boot_agent_sandbox(md["image"], instance_id, events=trace_events, tid=tid) as sb:
                is_tmax = md.get("protocol") == swe.PROTOCOL_TMAX
                with chrome_span(
                    trace_events,
                    "prepare_workspace",
                    cat="outer",
                    tid=tid,
                    args={"instance_id": instance_id},
                ):
                    await swe.prepare_workspace(sb, md["workdir"], md)
                state.adapter.bind_repo_state(session_id, sb=sb, workdir=md["workdir"])
                with chrome_span(
                    trace_events,
                    "agent_run",
                    cat="outer",
                    tid=tid,
                    args={"instance_id": instance_id, "session_id": session_id},
                ):
                    agent_exit_code = await HARNESS_CLS().run(
                        sb,
                        workdir=md["workdir"],
                        session_id=session_id,
                        adapter_url=state.adapter_url,
                        time_budget_sec=CONFIG.agent_time_budget_sec,
                        prompt=swe.SWE_PROMPT,
                    )
                diff_text = ""
                reward = 0.0
                applied_cleanly = True
                if is_tmax:
                    with chrome_span(
                        trace_events,
                        "eval",
                        cat="outer",
                        tid=tid,
                        args={"instance_id": instance_id, "protocol": "tmax"},
                    ):
                        reward, applied_cleanly = await swe.grade_tmax_inplace(
                            sb,
                            md,
                            timeout_sec=CONFIG.eval_timeout_sec,
                        )
                else:
                    with chrome_span(
                        trace_events,
                        "git_diff",
                        cat="outer",
                        tid=tid,
                        args={"instance_id": instance_id},
                    ):
                        diff_text = await swe.git_diff(sb, md["workdir"])
                # Best-effort diagnostics before the sandbox is torn down. Used when
                # finish_session later returns no turns (adapter_session_empty).
                diag = ""
                try:
                    _, diag, _ = await sb.exec(
                        "echo '=== /tmp/.run.sh (env+cmd) ==='; "
                        "head -n 30 /tmp/.run.sh 2>/dev/null; "
                        f"echo '=== trajectory tail ==='; "
                        f"tail -c 1200 {md['workdir']}/.harness/trajectory.jsonl 2>/dev/null || true",
                        timeout=30,
                    )
                except Exception:
                    diag = ""

            if not is_tmax:
                with chrome_span(
                    trace_events,
                    "eval",
                    cat="outer",
                    tid=tid,
                    args={"instance_id": instance_id},
                ):
                    reward, applied_cleanly = await swe.run_evaluation(
                        md,
                        diff_text=diff_text,
                        timeout_sec=CONFIG.eval_timeout_sec,
                    )
            solved = float(reward)
            empty_patch = not (diff_text or "").strip()
            # Belt-and-suspenders: never train on "solved" with no repo change
            # (scaleswe/swebench only — tmax scores final environment state).
            if (not is_tmax) and empty_patch and solved == 1.0:
                logger.warning(
                    "[coding_agent_rl] %s: grader returned solved with empty patch; forcing solved=0",
                    instance_id,
                )
                solved = 0.0
            offload_stats = {}
            session_obj = state.adapter.store.get(session_id)
            if session_obj is not None and getattr(session_obj, "offload_stats", None):
                offload_stats = dict(session_obj.offload_stats)
            train_reward = solved
            if offload.offload_enabled():
                usage = md.get("usage") if isinstance(md.get("usage"), dict) else None
                if offload.reward_mode() == "help_seeking":
                    # When OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG=1, defer α to
                    # shape_group_help_seeking_rewards (group all-failed only).
                    train_reward = offload.help_seeking_reward(
                        solved,
                        offload_stats,
                        usage=usage,
                        empty_patch=empty_patch,
                        encourage_seek=not offload.seek_only_when_all_wrong(),
                    )
                else:
                    train_reward = offload.cost_aware_reward(solved, offload_stats, usage=usage)
                logger.info(
                    "[coding_agent_rl] %s: solved=%.1f train_reward=%.4f empty_patch=%s "
                    "reward_mode=%s offload=%s",
                    instance_id,
                    solved,
                    train_reward,
                    empty_patch,
                    offload.reward_mode(),
                    offload_stats,
                )
            if evaluation:
                logger.info(
                    "[coding_agent_rl] %s: reward=%.2f applied=%s agent_exit_code=%d elapsed=%.1fs "
                    "timeline_events=%d (eval-only)",
                    instance_id,
                    solved,
                    bool(applied_cleanly),
                    agent_exit_code,
                    time.time() - t0,
                    len(trace_events),
                )
                result_samples = _eval_result(
                    base_sample,
                    reward=solved,
                    applied_cleanly=bool(applied_cleanly),
                    agent_exit_code=agent_exit_code,
                    instance_id=instance_id,
                )
                _attach_timeline(result_samples, timeline)
                return result_samples

            with chrome_span(
                trace_events,
                "finish_session",
                cat="outer",
                tid=tid,
                args={"instance_id": instance_id, "session_id": session_id},
            ):
                samples = await state.adapter.finish_session(
                    session_id,
                    base_sample=base_sample,
                    reward=float(train_reward),
                    extra_metadata={
                        "grading_solved": solved == 1.0,
                        "instance_id": instance_id,
                        "solved": solved,
                        "empty_patch": empty_patch,
                        "protocol": md.get("protocol"),
                        "offload_stats": offload_stats,
                        "timeline": timeline,
                        "turn_git_diffs": state.adapter.pop_turn_git_diffs(session_id),
                    },
                )
            if not samples:
                logger.warning(
                    "[coding_agent_rl] %s: adapter_session_empty "
                    "(agent_exit_code=%s adapter=%s). sandbox diag:\n%s",
                    instance_id,
                    agent_exit_code,
                    state.adapter_url,
                    (diag or "<empty>")[:1500],
                )
                result_samples = _abort_result(base_sample, "adapter_session_empty", instance_id)
                _attach_timeline(result_samples, timeline)
                return result_samples

            for s in samples:
                s.metadata = {**(s.metadata or {}), "agent_exit_code": agent_exit_code, "timeline": timeline}
            if agent_exit_code != 0:
                reason = "time budget exceeded" if agent_exit_code < 0 else f"CLI error (exit {agent_exit_code})"
                logger.warning(
                    "[coding_agent_rl] %s: agent_exit_code=%d (%s)",
                    instance_id,
                    agent_exit_code,
                    reason,
                )
            logger.info(
                "[coding_agent_rl] %s: reward=%.2f applied=%s agent_exit_code=%d elapsed=%.1fs "
                "segments=%d timeline_events=%d",
                instance_id,
                float(train_reward),
                bool(applied_cleanly),
                agent_exit_code,
                time.time() - t0,
                len(samples),
                len(trace_events),
            )
            result_samples = samples
            return result_samples

    except asyncio.TimeoutError:
        _log_timeout_diagnostic(t0, instance_id)
        result_samples = _abort_result(base_sample, "wall_clock_timeout", instance_id)
        _attach_timeline(result_samples, timeline)
        return result_samples
    except Exception as e:
        logger.warning(
            "[coding_agent_rl] %s: rollout failed: %s\n%s",
            instance_id,
            e,
            traceback.format_exc(),
        )
        result_samples = _abort_result(base_sample, f"exception:{type(e).__name__}", instance_id)
        _attach_timeline(result_samples, timeline)
        return result_samples
    finally:
        state.adapter.unbind_repo_state(session_id)
        state.adapter.pop_turn_git_diffs(session_id)
        await state.adapter.drop_session(session_id, wait_timeout=30)  # cleanup only, idempotent
        with chrome_span(
            trace_events,
            "cleanup_sleep",
            cat="outer",
            tid=tid,
            args={"instance_id": instance_id},
        ):
            await asyncio.sleep(10)
        if result_samples is not None:
            _attach_timeline(result_samples, timeline)


def _log_timeout_diagnostic(t0: float, instance_id: str) -> None:
    # Dump pending-task names when the wall-clock guard fires. Must not crash.
    try:
        elapsed = time.time() - t0
        pending = [t for t in asyncio.all_tasks() if not t.done()]
        stuck = []
        for t in pending[:5]:  # cap to avoid log spam
            coro = getattr(t, "_coro", None)
            stuck.append(getattr(coro, "__qualname__", repr(coro)))
        logger.warning(
            "[coding_agent_rl] %s: wall_clock_timeout after %.1fs "
            "(guard=%ds); %d tasks pending; sample of stuck: %s",
            instance_id,
            elapsed,
            CONFIG.rollout_guard_sec,
            len(pending),
            stuck,
        )
    except Exception:  # pragma: no cover - diag must never crash
        pass


def _session_id(sample: Sample, instance_id: str) -> str:
    if sample.session_id:
        return sample.session_id
    if sample.index is not None and sample.group_index is not None:
        return f"cagent-{instance_id}-{sample.index}-{sample.group_index}"
    return f"cagent-{instance_id}-{secrets.token_hex(8)}"


def _abort_result(sample: Sample, reason: str, instance_id: str) -> list[Sample]:
    """Mark ``sample`` aborted in place and return it in the list shape this
    fan-out generate function always yields."""
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED
    sample.metadata = {
        **(sample.metadata or {}),
        "abort_reason": reason,
        "instance_id": instance_id,
    }
    logger.warning("[coding_agent_rl] %s aborted: %s", instance_id, reason)
    return [sample]


def _eval_result(
    sample: Sample,
    *,
    reward: float,
    applied_cleanly: bool,
    agent_exit_code: int | None,
    instance_id: str,
) -> list[Sample]:
    """Eval-path placeholder: only ``reward`` matters for ``eval/sweb``."""

    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = float(reward)
    sample.remove_sample = True
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        **(sample.metadata or {}),
        "instance_id": instance_id,
        "grading_solved": float(reward) == 1.0,
        "applied_cleanly": applied_cleanly,
        "agent_exit_code": agent_exit_code,
    }
    return [sample]
