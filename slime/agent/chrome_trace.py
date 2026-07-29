"""Chrome Trace Event Format helpers (chrome://tracing / Perfetto).

Records begin/end pairs (ph=B/E) with wall-clock timestamps in microseconds.
Durations are not stored; consumers derive them in the UI or offline.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def now_us() -> float:
    """Wall-clock timestamp in microseconds (Chrome Trace ``ts`` unit)."""
    return time.time() * 1e6


def span_begin(
    events: list[dict[str, Any]],
    name: str,
    *,
    cat: str,
    tid: int,
    pid: int = 1,
    args: dict[str, Any] | None = None,
) -> None:
    ev: dict[str, Any] = {
        "name": name,
        "cat": cat,
        "ph": "B",
        "pid": pid,
        "tid": int(tid),
        "ts": now_us(),
    }
    if args:
        ev["args"] = dict(args)
    events.append(ev)


def span_end(
    events: list[dict[str, Any]],
    name: str,
    *,
    cat: str,
    tid: int,
    pid: int = 1,
    args: dict[str, Any] | None = None,
) -> None:
    ev: dict[str, Any] = {
        "name": name,
        "cat": cat,
        "ph": "E",
        "pid": pid,
        "tid": int(tid),
        "ts": now_us(),
    }
    if args:
        ev["args"] = dict(args)
    events.append(ev)


@contextmanager
def chrome_span(
    events: list[dict[str, Any]] | None,
    name: str,
    *,
    cat: str,
    tid: int | None,
    pid: int = 1,
    args: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Record a B/E pair around a block. No-op if ``events`` or ``tid`` is missing."""
    if events is None or tid is None:
        yield
        return
    span_begin(events, name, cat=cat, tid=tid, pid=pid, args=args)
    try:
        yield
    except BaseException:
        span_end(events, name, cat=cat, tid=tid, pid=pid, args={"status": "aborted"})
        raise
    else:
        span_end(events, name, cat=cat, tid=tid, pid=pid)


def ensure_session_timing(session: Any, *, tid: int, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach a shared timing dict to ``session`` (same ``events`` list as the caller)."""
    timing = getattr(session, "timing", None)
    if not isinstance(timing, dict):
        timing = {}
        session.timing = timing
    timing.setdefault("n_offloads", 0)
    timing.setdefault("n_turns", 0)
    timing["tid"] = int(tid)
    timing["trace_events"] = events
    return timing


def session_trace_ctx(session: Any) -> tuple[list[dict[str, Any]] | None, int | None, dict[str, Any] | None]:
    """Return ``(events, tid, timing)`` if this session is tracing; else nulls."""
    timing = getattr(session, "timing", None)
    if not isinstance(timing, dict):
        return None, None, None
    events = timing.get("trace_events")
    tid = timing.get("tid")
    if not isinstance(events, list) or tid is None:
        return None, None, None
    return events, int(tid), timing
