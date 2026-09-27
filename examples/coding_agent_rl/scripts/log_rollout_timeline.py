"""Export coding-agent rollout timelines as Chrome Trace Event Format JSON.

Wire via::

    --custom-rollout-log-function-path examples.coding_agent_rl.scripts.log_rollout_timeline.log_rollout_timeline

Writes ``${RUN_ROOT}/timelines/rollout_{rollout_id}.json`` (sibling of
``rollout_dumps/``). Also injects offload / solved W&B metrics into
``extra_metrics`` (``rollout/offload_*``, ``rollout/solved_*``). Returns
``False`` so default slime perf logging still runs.
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


def _sample_metadata(sample: Any) -> dict[str, Any]:
    md = getattr(sample, "metadata", None)
    return md if isinstance(md, dict) else {}


def _traj_key(sample: Any) -> tuple[Any, ...]:
    """Identity for one agent trajectory (fan-out segments share this key)."""
    md = _sample_metadata(sample)
    gid = md.get("group_index", getattr(sample, "group_index", None))
    idx = md.get("sample_index", getattr(sample, "index", None))
    instance_id = md.get("instance_id")
    return (gid, idx, instance_id)


def _offload_stats(sample: Any) -> dict[str, Any]:
    stats = _sample_metadata(sample).get("offload_stats")
    return stats if isinstance(stats, dict) else {}


def per_traj_offload_counts(samples: list[Any]) -> list[int]:
    """One ``offload_count`` per agent traj (dedupe fan-out segments)."""
    seen: set[tuple[Any, ...]] = set()
    counts: list[int] = []
    for sample in samples:
        key = _traj_key(sample)
        if key in seen:
            continue
        seen.add(key)
        stats = _offload_stats(sample)
        counts.append(int(stats.get("offload_count", 0) or 0))
    return counts


def _sample_solved(sample: Any) -> bool:
    md = _sample_metadata(sample)
    if md.get("grading_solved") is True:
        return True
    try:
        return float(md.get("solved", 0) or 0) > 0.0
    except (TypeError, ValueError):
        return False


def _prompt_key(sample: Any) -> Any:
    """Group identity for one prompt in the rollout batch (``rollout_batch_size``)."""
    md = _sample_metadata(sample)
    gid = md.get("group_index", getattr(sample, "group_index", None))
    if gid is not None:
        return gid
    return md.get("instance_id") or md.get("label") or getattr(sample, "index", id(sample))


def per_traj_solved_flags(samples: list[Any]) -> list[tuple[Any, bool]]:
    """``(prompt_key, solved)`` per agent traj (dedupe fan-out segments)."""
    seen: set[tuple[Any, ...]] = set()
    out: list[tuple[Any, bool]] = []
    for sample in samples:
        key = _traj_key(sample)
        if key in seen:
            continue
        seen.add(key)
        out.append((_prompt_key(sample), _sample_solved(sample)))
    return out


def compute_offload_rollout_metrics(samples: list[Any]) -> dict[str, float]:
    """W&B keys under ``rollout/offload_*`` / ``rollout/solved_*`` for one batch."""
    counts = per_traj_offload_counts(samples)
    solved_rows = per_traj_solved_flags(samples)
    metrics: dict[str, float] = {}
    if counts:
        n = len(counts)
        total = float(sum(counts))
        metrics.update(
            {
                "rollout/offload_count_mean": total / n,
                "rollout/offload_count_max": float(max(counts)),
                "rollout/offload_count_sum": total,
                "rollout/offload_frac": sum(1 for c in counts if c > 0) / n,
                "rollout/offload_n_trajs": float(n),
            }
        )
    if solved_rows:
        n_traj = len(solved_rows)
        traj_solved = sum(1 for _, sol in solved_rows if sol)
        by_prompt: dict[Any, list[bool]] = {}
        for prompt_key, sol in solved_rows:
            by_prompt.setdefault(prompt_key, []).append(sol)
        n_prompts = len(by_prompt)
        prompt_solved = sum(1 for flags in by_prompt.values() if any(flags))
        metrics.update(
            {
                # Trajectory-level mean (e.g. 41/128).
                "rollout/solved_mean": traj_solved / n_traj,
                "rollout/solved_traj_count": float(traj_solved),
                # Prompt-level (rollout_batch_size): how many of 16 are solvable.
                "rollout/solved_prompt_frac": prompt_solved / n_prompts if n_prompts else 0.0,
                "rollout/solved_prompt_count": float(prompt_solved),
                "rollout/n_prompts": float(n_prompts),
            }
        )
    return metrics


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

    offload_metrics = compute_offload_rollout_metrics(flat)
    if offload_metrics and isinstance(extra_metrics, dict):
        extra_metrics.update(offload_metrics)
    elif offload_metrics:
        # Debug-load path may pass metrics=None; still emit W&B/TB points.
        from slime.utils import logging_utils
        from slime.utils.metric_utils import compute_rollout_step

        step = compute_rollout_step(args, rollout_id)
        logging_utils.log(args, {**offload_metrics, "rollout/step": step}, step_key="rollout/step")

    n_events = len(doc["traceEvents"])
    ts_values = [float(e["ts"]) for e in doc["traceEvents"] if e.get("ph") in {"B", "E"} and "ts" in e]
    wall_range = ""
    if ts_values:
        wall_range = f" ts_us=[{min(ts_values):.0f},{max(ts_values):.0f}]"
    logger.info(
        "[coding_agent_timeline] rollout=%s path=%s n_samples=%d n_events=%d "
        "rollout_time=%.1fs offload_mean=%.2f offload_frac=%.2f "
        "solved_mean=%.3f solved_prompt=%s/%s%s",
        rollout_id,
        out_path,
        doc.get("meta_n_samples_with_timeline", 0),
        n_events,
        float(rollout_time or 0.0),
        float(offload_metrics.get("rollout/offload_count_mean", 0.0)),
        float(offload_metrics.get("rollout/offload_frac", 0.0)),
        float(offload_metrics.get("rollout/solved_mean", 0.0)),
        int(offload_metrics.get("rollout/solved_prompt_count", 0.0)),
        int(offload_metrics.get("rollout/n_prompts", 0.0)),
        wall_range,
    )
    # False => keep default slime perf / rollout metric logging.
    return False
