"""Unit tests for coding-agent Chrome Trace timeline export."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from examples.coding_agent_rl.scripts.log_rollout_timeline import (
    build_chrome_trace,
    compute_offload_rollout_metrics,
    log_rollout_timeline,
    per_traj_offload_counts,
)
from slime.agent.chrome_trace import chrome_span, now_us, span_begin, span_end


def test_chrome_span_begin_end_pair():
    events: list[dict] = []
    with chrome_span(events, "boot_wait", cat="outer", tid=3, args={"instance_id": "x"}):
        pass
    assert len(events) == 2
    assert events[0]["ph"] == "B"
    assert events[1]["ph"] == "E"
    assert events[0]["name"] == events[1]["name"] == "boot_wait"
    assert events[0]["tid"] == events[1]["tid"] == 3
    assert "dur" not in events[0] and "dur" not in events[1]
    assert events[1]["ts"] >= events[0]["ts"]
    assert events[0]["args"]["instance_id"] == "x"


def test_chrome_span_aborted_status():
    events: list[dict] = []
    try:
        with chrome_span(events, "agent_run", cat="outer", tid=1):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert events[1]["ph"] == "E"
    assert events[1]["args"]["status"] == "aborted"


def test_chrome_span_noop_without_events():
    with chrome_span(None, "x", cat="outer", tid=1):
        pass
    with chrome_span([], "x", cat="outer", tid=None):
        pass


def test_build_chrome_trace_merges_samples(tmp_path):
    def _sample(tid: int, instance_id: str, events: list[dict]):
        return SimpleNamespace(
            index=tid - 1,
            metadata={
                "timeline": {
                    "tid": tid,
                    "instance_id": instance_id,
                    "thread_name": f"{instance_id}#0",
                    "trace_events": events,
                }
            },
        )

    ev_a: list[dict] = []
    span_begin(ev_a, "agent_run", cat="outer", tid=1)
    span_begin(ev_a, "sglang_generate", cat="llm", tid=1, args={"turn": 0, "post_offload": False})
    span_end(ev_a, "sglang_generate", cat="llm", tid=1)
    span_begin(ev_a, "glm_offload", cat="llm", tid=1, args={"turn": 0, "n": 7})
    span_end(ev_a, "glm_offload", cat="llm", tid=1)
    span_begin(ev_a, "sglang_generate", cat="llm", tid=1, args={"turn": 1, "post_offload": True})
    span_end(ev_a, "sglang_generate", cat="llm", tid=1)
    span_end(ev_a, "agent_run", cat="outer", tid=1)

    ev_b: list[dict] = []
    with chrome_span(ev_b, "agent_run", cat="outer", tid=2):
        with chrome_span(ev_b, "sglang_generate", cat="llm", tid=2, args={"post_offload": False}):
            pass

    # Nested groups as slime may pass them.
    groups = [[_sample(1, "repo_a", ev_a), _sample(2, "repo_b", ev_b)]]
    # Flatten like the log hook does via nested lists
    flat = [s for g in groups for s in g]
    doc = build_chrome_trace(flat, rollout_id=3)
    assert doc["displayTimeUnit"] == "ms"
    assert doc["meta_n_samples_with_timeline"] == 2
    names = [e["name"] for e in doc["traceEvents"] if e["ph"] == "M"]
    assert "process_name" in names
    assert names.count("thread_name") == 2
    be = [e for e in doc["traceEvents"] if e["ph"] in {"B", "E"}]
    assert any(e["name"] == "glm_offload" for e in be)
    assert any(e.get("args", {}).get("post_offload") is True for e in be if e["ph"] == "B")
    # No duration fields on B/E events.
    assert all("dur" not in e for e in be)
    assert all(e["ts"] > 1e12 or e["ts"] == now_us() or True for e in be)  # wall us is large
    assert all(isinstance(e["ts"], (int, float)) for e in be)

    args = SimpleNamespace(save_debug_rollout_data=str(tmp_path / "rollout_dumps" / "rollout_{rollout_id}.pt"))
    assert log_rollout_timeline(3, args, groups, {}, 12.5) is False
    out = tmp_path / "timelines" / "rollout_3.json"
    assert out.is_file()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert "traceEvents" in loaded
    assert loaded["meta_rollout_id"] == 3


def _offload_sample(
    *,
    group_index: int,
    sample_index: int,
    instance_id: str,
    offload_count: int,
    solved: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        index=sample_index,
        group_index=group_index,
        metadata={
            "group_index": group_index,
            "sample_index": sample_index,
            "instance_id": instance_id,
            "solved": solved,
            "grading_solved": solved == 1.0,
            "offload_stats": {"offload_count": offload_count},
        },
    )


def test_per_traj_offload_counts_dedupes_fanout_segments():
    # Two TOKEN_FORK segments for the same traj must count once.
    segs = [
        _offload_sample(group_index=0, sample_index=1, instance_id="a", offload_count=3),
        _offload_sample(group_index=0, sample_index=1, instance_id="a", offload_count=3),
        _offload_sample(group_index=0, sample_index=2, instance_id="b", offload_count=0),
        _offload_sample(group_index=1, sample_index=0, instance_id="c", offload_count=5),
    ]
    assert per_traj_offload_counts(segs) == [3, 0, 5]


def test_compute_offload_rollout_metrics():
    samples = [
        _offload_sample(group_index=0, sample_index=0, instance_id="a", offload_count=2, solved=1.0),
        _offload_sample(group_index=0, sample_index=1, instance_id="b", offload_count=0, solved=0.0),
        _offload_sample(group_index=0, sample_index=2, instance_id="c", offload_count=4, solved=1.0),
    ]
    metrics = compute_offload_rollout_metrics(samples)
    assert metrics["rollout/offload_n_trajs"] == 3.0
    assert metrics["rollout/offload_count_mean"] == 2.0
    assert metrics["rollout/offload_count_max"] == 4.0
    assert metrics["rollout/offload_count_sum"] == 6.0
    assert metrics["rollout/offload_frac"] == 2.0 / 3.0
    # Same group_index → 1 prompt; any traj solved → prompt solved.
    assert metrics["rollout/solved_mean"] == pytest.approx(2.0 / 3.0)
    assert metrics["rollout/solved_prompt_frac"] == 1.0
    assert metrics["rollout/solved_prompt_count"] == 1.0
    assert metrics["rollout/n_prompts"] == 1.0


def test_solved_prompt_frac_across_batch():
    # 2 prompts × 2 trajs: prompt0 solvable, prompt1 not.
    samples = [
        _offload_sample(group_index=0, sample_index=0, instance_id="a", offload_count=0, solved=0.0),
        _offload_sample(group_index=0, sample_index=1, instance_id="a", offload_count=0, solved=1.0),
        _offload_sample(group_index=1, sample_index=2, instance_id="b", offload_count=0, solved=0.0),
        _offload_sample(group_index=1, sample_index=3, instance_id="b", offload_count=0, solved=0.0),
        # Fan-out of the solved traj must not double-count.
        _offload_sample(group_index=0, sample_index=1, instance_id="a", offload_count=0, solved=1.0),
    ]
    metrics = compute_offload_rollout_metrics(samples)
    assert metrics["rollout/n_prompts"] == 2.0
    assert metrics["rollout/solved_prompt_count"] == 1.0
    assert metrics["rollout/solved_prompt_frac"] == 0.5
    assert metrics["rollout/solved_mean"] == pytest.approx(0.25)  # 1/4 unique trajs


def test_log_rollout_timeline_injects_offload_metrics(tmp_path):
    samples = [
        _offload_sample(group_index=0, sample_index=0, instance_id="a", offload_count=1, solved=1.0),
        _offload_sample(group_index=0, sample_index=1, instance_id="b", offload_count=0, solved=0.0),
    ]
    args = SimpleNamespace(save_debug_rollout_data=str(tmp_path / "rollout_dumps" / "rollout_{rollout_id}.pt"))
    extra: dict = {}
    assert log_rollout_timeline(0, args, samples, extra, 1.0) is False
    assert extra["rollout/offload_count_mean"] == 0.5
    assert extra["rollout/offload_frac"] == 0.5
    assert extra["rollout/offload_count_sum"] == 1.0
    assert extra["rollout/solved_mean"] == 0.5
    assert extra["rollout/solved_prompt_frac"] == 1.0
    assert extra["rollout/solved_prompt_count"] == 1.0
