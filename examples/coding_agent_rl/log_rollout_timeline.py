"""Export coding-agent rollout timelines as Chrome Trace Event Format JSON.

Wire via::

    --custom-rollout-log-function-path examples.coding_agent_rl.log_rollout_timeline.log_rollout_timeline

Writes ``${RUN_ROOT}/timelines/rollout_{rollout_id}.json`` (sibling of
``rollout_dumps/``). Returns ``False`` so default slime perf logging still runs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PID = 1


def _timeline_dir(args: Any) -> Path:
    env = (os.environ.get("SLIME_TIMELINE_DIR") or "").strip()
    if env:
        return Path(env)
    dump = getattr(args, "save_debug_rollout_data", None)
    if isinstance(dump, str) and dump:
        # e.g. .../rollout_dumps/rollout_{rollout_id}.pt -> RUN_ROOT/timelines
        dump_path = Path(dump.replace("{rollout_id}", "0"))
        # parent = rollout_dumps, parent.parent = RUN_ROOT
        if dump_path.parent.name == "rollout_dumps":
            return dump_path.parent.parent / "timelines"
        return dump_path.parent / "timelines"
    return Path("timelines")


def _iter_samples(samples: Any) -> list[Any]:
    """Flatten slime rollout sample groups into a single list."""
    if samples is None:
        return []
    flat: list[Any] = []
    if not isinstance(samples, list):
        return flat
    for item in samples:
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, list):
                    flat.extend(sub)
                else:
                    flat.append(sub)
        else:
            flat.append(item)
    return flat


def _sample_timeline(sample: Any) -> dict[str, Any] | None:
    md = getattr(sample, "metadata", None)
    if not isinstance(md, dict):
        return None
    timeline = md.get("timeline")
    return timeline if isinstance(timeline, dict) else None


def build_chrome_trace(samples: list[Any], *, rollout_id: int) -> dict[str, Any]:
    """Merge per-sample timeline events into one Chrome Trace document.

    Fan-out segments from one agent run share the same ``tid`` / event list;
    only the first sample per ``tid`` is exported to avoid duplicate slices.
    """
    events: list[dict[str, Any]] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": _PID,
            "args": {"name": f"coding_agent_rollout_{rollout_id}"},
        }
    ]
    seen_tids: set[int] = set()
    n_samples_with_timeline = 0

    for sample in samples:
        timeline = _sample_timeline(sample)
        if not timeline:
            continue
        raw_events = timeline.get("trace_events")
        if not isinstance(raw_events, list) or not raw_events:
            continue
        tid = int(timeline.get("tid") or getattr(sample, "index", None) or 0) or 1
        if tid in seen_tids:
            continue
        seen_tids.add(tid)
        n_samples_with_timeline += 1
        thread_name = (
            timeline.get("thread_name")
            or timeline.get("instance_id")
            or f"sample-{tid}"
        )
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": _PID,
                "tid": tid,
                "args": {"name": str(thread_name)},
            }
        )
        for ev in raw_events:
            if not isinstance(ev, dict):
                continue
            # Defensive copy; force pid for a single-process view.
            out = dict(ev)
            out["pid"] = _PID
            out.setdefault("tid", tid)
            events.append(out)

    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "meta_rollout_id": rollout_id,
        "meta_n_samples_with_timeline": n_samples_with_timeline,
    }


def log_rollout_timeline(
    rollout_id: int,
    args: Any,
    samples: Any,
    extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Custom rollout log hook: dump Chrome Trace JSON, then defer to defaults."""
    flat = _iter_samples(samples)
    doc = build_chrome_trace(flat, rollout_id=rollout_id)
    out_dir = _timeline_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rollout_{rollout_id}.json"
    out_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    n_events = len(doc["traceEvents"])
    ts_values = [float(e["ts"]) for e in doc["traceEvents"] if e.get("ph") in {"B", "E"} and "ts" in e]
    wall_range = ""
    if ts_values:
        wall_range = f" ts_us=[{min(ts_values):.0f},{max(ts_values):.0f}]"
    logger.info(
        "[coding_agent_timeline] rollout=%s path=%s n_samples=%d n_events=%d "
        "rollout_time=%.1fs%s",
        rollout_id,
        out_path,
        doc.get("meta_n_samples_with_timeline", 0),
        n_events,
        float(rollout_time or 0.0),
        wall_range,
    )
    # False => keep default slime perf / rollout metric logging.
    return False
