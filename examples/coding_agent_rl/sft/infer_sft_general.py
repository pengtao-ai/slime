#!/usr/bin/env python3
"""Think-first-sentence entropy on phase3 filtered / offload jsonl.

For every assistant turn, POST the conversation prefix to local vLLM and
record the average entropy of the **first sentence** of thinking (through
the first ``.`` or ``。``). A single-turn sample yields one number; a
multi-turn sample yields one number per turn.

If a sample is multi-turn **and** contains offload (``<|llm_offload|>`` or
the offload system-prompt append), those markers are stripped before the
call. Single-turn offload samples are scored with the offload prompt left
in place.

Example::

    bash examples/coding_agent_rl/sft/launch_vllm_pyrodash4b_sft0902.sh
    python examples/coding_agent_rl/sft/infer_sft_general.py --limit 2
    python examples/coding_agent_rl/sft/infer_sft_general.py --resume --concurrency 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import annotate_reward1_entropy as ent  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_sft_general")

DEFAULT_DATA_DIRS = [
    Path("/workspace/datasets/phase3/filtered"),
    Path("/workspace/datasets/phase3/offload"),
]
DEFAULT_MODEL = "/workspace/models/pyromind/PyroDash-4B-SFT-0918"
DEFAULT_VLLM_URL = "http://127.0.0.1:9016/v1"
DEFAULT_OUT = _REPO_ROOT / "runs" / "phase3_think_entropy"

OFFLOAD_OPEN = "<|llm_offload|>"
OFFLOAD_CLOSE = "<|/llm_offload|>"
# Tags are inserted on a period: ``server.<|llm_offload|>N<|/llm_offload|>.py``.
_OFFLOAD_SPAN_RE = re.compile(
    r"\." + re.escape(OFFLOAD_OPEN) + r"\d" + re.escape(OFFLOAD_CLOSE) + r"\."
)
_OFFLOAD_SPAN_BARE_RE = re.compile(
    re.escape(OFFLOAD_OPEN) + r"\d" + re.escape(OFFLOAD_CLOSE)
)
# Keep in sync with offload.OFFLOAD_SYSTEM_PROMPT_APPEND.
_OFFLOAD_SYSTEM_APPEND = (
    "For very difficult steps, you can output "
    f"{OFFLOAD_OPEN}N{OFFLOAD_CLOSE} where N is 0-9 indicating the thinking "
    "level for a more capable model."
)

_TLS = threading.local()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return round(sum(vals) / len(vals), 6)


def _session() -> requests.Session:
    sess = getattr(_TLS, "session", None)
    if sess is None:
        sess = requests.Session()
        _TLS.session = sess
    return sess


def _call_vllm(
    *,
    url: str,
    api_key: str,
    body: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any] | None, str | None, float]:
    t0 = time.monotonic()
    try:
        resp = _session().post(
            f"{url.rstrip('/')}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=body,
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text[:500]}", time.monotonic() - t0
        return resp.json(), None, time.monotonic() - t0
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", time.monotonic() - t0


def _text(content: Any) -> str:
    return content if isinstance(content, str) else ""


def _has_offload(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        if OFFLOAD_OPEN in _text(msg.get("content")) or OFFLOAD_CLOSE in _text(msg.get("content")):
            return True
    return False


def _n_assistants(messages: list[dict[str, Any]]) -> int:
    return sum(1 for msg in messages if msg.get("role") == "assistant")


def _strip_offload_text(text: str, *, system: bool) -> str:
    if system and _OFFLOAD_SYSTEM_APPEND in text:
        text = text.replace(_OFFLOAD_SYSTEM_APPEND, "")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = _OFFLOAD_SPAN_RE.sub(".", text)
    text = _OFFLOAD_SPAN_BARE_RE.sub("", text)
    return text.replace(OFFLOAD_OPEN, "").replace(OFFLOAD_CLOSE, "")


def _strip_offload_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        copied = dict(msg)
        if isinstance(copied.get("content"), str):
            copied["content"] = _strip_offload_text(
                copied["content"], system=copied.get("role") == "system"
            )
        out.append(copied)
    return out


def _arguments_json(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
    except TypeError:
        return json.dumps({"_raw": str(arguments)}, ensure_ascii=False)


def _normalize_tool_calls(tool_calls: Any, *, synth: int) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    if not isinstance(tool_calls, list):
        return out, synth
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") or call.get("name") or "tool"
        arguments = function.get("arguments")
        if arguments is None:
            arguments = call.get("arguments", {})
        call_id = str(call.get("id") or f"call_{synth}")
        if not call.get("id"):
            synth += 1
        out.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": str(name), "arguments": _arguments_json(arguments)},
            }
        )
    return out, synth


def _tools_for_vllm(tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tools, list):
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if function is None:
            name = tool.get("name")
            if not name:
                continue
            function = {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema")
                or tool.get("parameters")
                or {"type": "object", "properties": {}},
            }
        name = function.get("name")
        if not name:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": function.get("description") or "",
                    "parameters": function.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return out or None


def _to_vllm_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI chat messages: fill missing tool_call ids, drop dataset extras."""
    out: list[dict[str, Any]] = []
    pending: list[str] = []
    synth = 0
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)

        if role == "assistant":
            calls, synth = _normalize_tool_calls(msg.get("tool_calls"), synth=synth)
            pending = [call["id"] for call in calls]
            item: dict[str, Any] = {"role": "assistant", "content": content}
            if calls:
                item["tool_calls"] = calls
            out.append(item)
            continue

        if role == "tool":
            tid = msg.get("tool_call_id") or msg.get("tool_use_id")
            if tid:
                tid = str(tid)
                if tid in pending:
                    pending.remove(tid)
            elif pending:
                tid = pending.pop(0)
            else:
                tid = f"call_orphan_{synth}"
                synth += 1
            item = {"role": "tool", "tool_call_id": tid, "content": content}
            name = msg.get("name")
            if isinstance(name, str) and name:
                item["name"] = name
            out.append(item)
            continue

        out.append({"role": role, "content": content})
    return out


def _prepare_sample(obj: dict[str, Any]) -> dict[str, Any]:
    raw_messages = [m for m in (obj.get("messages") or []) if isinstance(m, dict)]
    n_turns = _n_assistants(raw_messages)
    multi = n_turns > 1
    had_offload = _has_offload(raw_messages)
    stripped = multi and had_offload
    messages = _strip_offload_messages(raw_messages) if stripped else raw_messages
    if stripped and _has_offload(messages):
        logger.warning("offload marker still present after strip")
    vllm_messages = _to_vllm_messages(messages)
    asst_idxs = [i for i, m in enumerate(vllm_messages) if m.get("role") == "assistant"]
    # Source assistants, before offload stripping. Scoring still uses the prefix
    # only; these are written into the result unchanged.
    raw_assistants = [dict(m) for m in raw_messages if m.get("role") == "assistant"]
    return {
        "n_turns": n_turns,
        "multi_turn": multi,
        "had_offload": had_offload,
        "stripped_offload": stripped,
        "messages": vllm_messages,
        "assistant_indices": asst_idxs,
        "raw_assistants": raw_assistants,
        "tools": _tools_for_vllm(obj.get("tools")),
    }


def _sample_id(obj: dict[str, Any], index: int) -> str:
    for key in ("instance_id", "uuid", "source_id", "question_id", "prompt_id"):
        val = obj.get(key)
        if val:
            return str(val)
    return str(index)


def _entropy_fields(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "avg_entropy": None,
            "n_used": None,
            "n_tokens": None,
            "unavailable": None,
            "first_sentence": None,
        }
    sentence = payload.get("first_sentence")
    if isinstance(sentence, str) and len(sentence) > 500:
        sentence = sentence[:500] + "…"
    return {
        "avg_entropy": payload.get("avg_entropy"),
        "n_used": payload.get("n_used"),
        "n_tokens": payload.get("n_tokens"),
        "unavailable": payload.get("unavailable"),
        "first_sentence": sentence,
    }


def _score_turn(
    *,
    turn_index: int,
    msg_index: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    top_logprobs: int,
    enable_thinking: bool,
    stop: list[str] | None,
    timeout: float,
) -> dict[str, Any]:
    body = ent._build_vllm_body(
        model=model,
        messages=messages[:msg_index],
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        top_logprobs=top_logprobs,
        include_tools=bool(tools),
        enable_thinking=enable_thinking,
        stop=stop,
    )
    data, err, elapsed = _call_vllm(url=url, api_key=api_key, body=body, timeout=timeout)
    row: dict[str, Any] = {
        "turn_index": turn_index,
        "msg_index": msg_index,
        "elapsed_sec": round(elapsed, 3),
        "ok": err is None,
        "error": err,
    }
    if data is None:
        row.update(_entropy_fields(None))
        return row
    try:
        finish = (data.get("choices") or [{}])[0].get("finish_reason")
    except Exception:
        finish = None
    row["finish_reason"] = finish
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    row["prompt_tokens"] = usage.get("prompt_tokens")
    row["completion_tokens"] = usage.get("completion_tokens")
    row.update(_entropy_fields(ent._compute_entropy_from_response(data, scope="thinking")))
    if not isinstance(row.get("avg_entropy"), (int, float)):
        row["ok"] = False
        row["error"] = row.get("error") or row.get("unavailable") or "no think-sentence entropy"
    return row


def _score_sample(
    *,
    index: int,
    obj: dict[str, Any],
    source: str,
    url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    top_logprobs: int,
    enable_thinking: bool,
    stop: list[str] | None,
    timeout: float,
) -> dict[str, Any]:
    prepared = _prepare_sample(obj)
    base = {
        "source": source,
        "index": index,
        "sample_id": _sample_id(obj, index),
        "n_turns": prepared["n_turns"],
        "multi_turn": prepared["multi_turn"],
        "had_offload": prepared["had_offload"],
        "stripped_offload": prepared["stripped_offload"],
    }
    if not prepared["assistant_indices"]:
        return {
            **base,
            "ok": False,
            "error": "no assistant turns",
            "mean_think_entropy": None,
            "turn_entropies": [],
        }

    raw_assistants: list[dict[str, Any]] = prepared["raw_assistants"]
    turns: list[dict[str, Any]] = []
    for turn_index, msg_index in enumerate(prepared["assistant_indices"]):
        assistant = raw_assistants[turn_index] if turn_index < len(raw_assistants) else None
        try:
            row = _score_turn(
                turn_index=turn_index,
                msg_index=msg_index,
                messages=prepared["messages"],
                tools=prepared["tools"],
                url=url,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                top_logprobs=top_logprobs,
                enable_thinking=enable_thinking,
                stop=stop,
                timeout=timeout,
            )
        except Exception as exc:
            row = {
                "turn_index": turn_index,
                "msg_index": msg_index,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                **_entropy_fields(None),
            }
        row["assistant"] = assistant
        turns.append(row)

    vals = [
        float(t["avg_entropy"])
        for t in turns
        if isinstance(t.get("avg_entropy"), (int, float))
    ]
    return {
        **base,
        "ok": bool(turns) and all(t.get("ok") for t in turns),
        "error": next((t.get("error") for t in turns if t.get("error")), None),
        "mean_think_entropy": _mean(vals),
        "turn_entropies": turns,
    }


def _collect_inputs(data_dirs: list[Path], include: str) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    for raw in data_dirs:
        path = raw.expanduser().resolve()
        if path.is_file():
            jobs.append((path.parent.name, path))
            continue
        if not path.is_dir():
            raise SystemExit(f"missing data path: {path}")
        matched = sorted(p for p in path.glob(include) if p.is_file() and p.suffix == ".jsonl")
        if not matched:
            logger.warning("no jsonl under %s (include=%s)", path, include)
        for item in matched:
            if item.stat().st_size <= 0:
                logger.info("skip empty %s", item)
                continue
            jobs.append((path.name, item))
    return jobs


def _output_jsonl(out_dir: Path, split: str, src: Path) -> Path:
    return out_dir / split / f"{src.stem}.entropy.jsonl"


def _load_done(path: Path, *, retry_failed: bool) -> set[int]:
    done: set[int] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "index" not in row:
                continue
            if retry_failed and not row.get("ok"):
                continue
            done.add(int(row["index"]))
    return done


def _summarize_jsonl(path: Path, *, split: str, src: Path) -> dict[str, Any]:
    """Last row per index wins (so --retry-failed does not double-count).

    Keeps only the numeric fields. Assistant text stays in the jsonl and is
    not held in memory for the aggregate.
    """
    latest: dict[int, dict[str, Any]] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "index" not in row:
                    continue
                turn_avgs = [
                    (int(turn.get("turn_index") or 0), turn.get("avg_entropy"))
                    for turn in (row.get("turn_entropies") or [])
                    if isinstance(turn, dict)
                ]
                latest[int(row["index"])] = {
                    "ok": row.get("ok"),
                    "multi_turn": row.get("multi_turn"),
                    "had_offload": row.get("had_offload"),
                    "stripped_offload": row.get("stripped_offload"),
                    "mean_think_entropy": row.get("mean_think_entropy"),
                    "turn_avgs": turn_avgs,
                }

    sample_means: list[float] = []
    turn_vals: list[float] = []
    by_turn: dict[int, list[float]] = defaultdict(list)
    n_ok = 0
    n_multi = 0
    n_stripped = 0
    n_offload = 0
    for row in latest.values():
        if row.get("ok"):
            n_ok += 1
        if row.get("multi_turn"):
            n_multi += 1
        if row.get("had_offload"):
            n_offload += 1
        if row.get("stripped_offload"):
            n_stripped += 1
        if isinstance(row.get("mean_think_entropy"), (int, float)):
            sample_means.append(float(row["mean_think_entropy"]))
        for turn_index, avg in row.get("turn_avgs") or []:
            if isinstance(avg, (int, float)):
                val = float(avg)
                turn_vals.append(val)
                by_turn[int(turn_index)].append(val)

    return {
        "created_at": _utc_now(),
        "split": split,
        "file": src.name,
        "jsonl": str(src),
        "out": str(path),
        "n": len(latest),
        "ok": n_ok,
        "failed": len(latest) - n_ok,
        "n_multi_turn": n_multi,
        "n_had_offload": n_offload,
        "n_stripped_offload": n_stripped,
        "n_turn_scores": len(turn_vals),
        "mean_think_entropy": _mean(sample_means),
        "mean_turn_entropy": _mean(turn_vals),
        "mean_by_turn_index": {
            str(idx): _mean(vals) for idx, vals in sorted(by_turn.items())
        },
    }


def _iter_samples(path: Path, *, offset: int, limit: int | None):
    seen = 0
    yielded = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if seen < offset:
                seen += 1
                continue
            if limit is not None and yielded >= limit:
                break
            index = seen
            seen += 1
            yielded += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                yield index, None, f"JSONDecodeError: {exc}"
                continue
            if not isinstance(obj, dict):
                yield index, None, "row is not an object"
                continue
            yield index, obj, None


def _process_file(
    *,
    split: str,
    src: Path,
    out_dir: Path,
    args: argparse.Namespace,
    url: str,
    api_key: str,
    model: str,
    stop: list[str] | None,
) -> dict[str, Any]:
    out_path = _output_jsonl(out_dir, split, src)
    source = f"{split}/{src.name}"
    if args.dry_run:
        n = 0
        n_multi = 0
        n_stripped = 0
        n_turns = 0
        n_bad = 0
        for _index, obj, err in _iter_samples(src, offset=args.offset, limit=args.limit):
            n += 1
            if err or obj is None:
                n_bad += 1
                continue
            prepared = _prepare_sample(obj)
            n_turns += int(prepared["n_turns"])
            n_multi += int(bool(prepared["multi_turn"]))
            n_stripped += int(bool(prepared["stripped_offload"]))
        summary = {
            "split": split,
            "file": src.name,
            "dry_run": True,
            "n": n,
            "n_bad": n_bad,
            "n_multi_turn": n_multi,
            "n_stripped_offload": n_stripped,
            "n_turns": n_turns,
        }
        print(
            f"[dry] {source} samples={n} multi={n_multi} stripped={n_stripped} "
            f"turns={n_turns} bad={n_bad}",
            flush=True,
        )
        return summary

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path, retry_failed=args.retry_failed) if args.resume or args.retry_failed else set()
    pending: list[tuple[int, dict[str, Any] | None, str | None]] = []
    n_skip = 0
    for index, obj, err in _iter_samples(src, offset=args.offset, limit=args.limit):
        if index in done:
            n_skip += 1
            continue
        pending.append((index, obj, err))

    print(
        f"[file] {source} pending={len(pending)} resume_skip={n_skip} -> {out_path}",
        flush=True,
    )
    n_done = 0
    n_fail_logged = 0
    means: list[float] = []
    write_lock = threading.Lock()

    def _one(item: tuple[int, dict[str, Any] | None, str | None]) -> dict[str, Any]:
        index, obj, err = item
        if err or obj is None:
            return {
                "source": source,
                "index": index,
                "ok": False,
                "error": err or "bad row",
                "n_turns": 0,
                "multi_turn": False,
                "had_offload": False,
                "stripped_offload": False,
                "mean_think_entropy": None,
                "turn_entropies": [],
            }
        return _score_sample(
            index=index,
            obj=obj,
            source=source,
            url=url,
            api_key=api_key,
            model=model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            top_logprobs=args.top_logprobs,
            enable_thinking=not args.no_thinking,
            stop=stop,
            timeout=args.timeout,
        )

    def _write(row: dict[str, Any]) -> None:
        nonlocal n_done, n_fail_logged
        with write_lock:
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done += 1
            if isinstance(row.get("mean_think_entropy"), (int, float)):
                means.append(float(row["mean_think_entropy"]))
            if not row.get("ok") and n_fail_logged < 3:
                n_fail_logged += 1
                logger.error(
                    "[%s] idx=%s failed: %s", source, row.get("index"), row.get("error")
                )
            if n_done == 1 or n_done % 20 == 0:
                print(
                    f"[file] {source} scored={n_done}/{len(pending)} "
                    f"running_mean_H={_mean(means)}",
                    flush=True,
                )

    if pending:
        workers = max(1, int(args.concurrency))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one, item) for item in pending]
            for fut in as_completed(futures):
                _write(fut.result())

    summary = _summarize_jsonl(out_path, split=split, src=src)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[file] {source} done n={summary['n']} ok={summary['ok']} "
        f"mean_H={summary['mean_think_entropy']} "
        f"turns={summary['mean_by_turn_index']}",
        flush=True,
    )
    return summary


def run(args: argparse.Namespace) -> None:
    url = args.url.rstrip("/")
    model = args.model
    api_key = args.api_key
    stop = None if args.no_stop else (list(args.stop) if args.stop else [".", "。"])

    jobs = _collect_inputs(args.data_dir, args.include)
    if not jobs:
        raise SystemExit("no jsonl inputs")

    out_dir = Path(args.out_dir)
    existing = [p for p in (out_dir.rglob("*.entropy.jsonl") if out_dir.exists() else [])]
    if existing and not args.force and not args.resume and not args.retry_failed and not args.dry_run:
        raise SystemExit(
            f"--out-dir already has entropy jsonl ({len(existing)} files): {out_dir} "
            "(pass --resume or --force)"
        )
    if args.force and not args.dry_run:
        for split, src in jobs:
            out_path = _output_jsonl(out_dir, split, src)
            summary_path = out_path.with_suffix(".summary.json")
            out_path.unlink(missing_ok=True)
            summary_path.unlink(missing_ok=True)

    print(
        f"[infer] files={len(jobs)} model={model} url={url} "
        f"concurrency={args.concurrency} limit={args.limit} stop={stop} out={out_dir}",
        flush=True,
    )
    for split, src in jobs:
        print(f"[infer] input {split}/{src.name}", flush=True)

    if not args.dry_run:
        try:
            health = requests.get(
                f"{url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            print(f"[vllm] GET /models -> HTTP {health.status_code}", flush=True)
            if health.status_code != 200:
                print(health.text[:300], file=sys.stderr)
        except Exception as exc:
            print(f"[warn] vLLM health check failed: {exc}", file=sys.stderr)

    file_summaries: list[dict[str, Any]] = []
    for split, src in jobs:
        file_summaries.append(
            _process_file(
                split=split,
                src=src,
                out_dir=out_dir,
                args=args,
                url=url,
                api_key=api_key,
                model=model,
                stop=stop,
            )
        )

    if args.dry_run:
        print("[infer] dry-run done", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    root = {
        "created_at": _utc_now(),
        "mode": "phase3_think_first_sentence_entropy",
        "url": url,
        "model": model,
        "entropy_scope": "thinking_first_sentence",
        "stop": stop,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "top_logprobs": args.top_logprobs,
        "files": file_summaries,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(root, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("[infer] per-file mean think-first-sentence entropy:", flush=True)
    for row in file_summaries:
        print(
            f"  {row.get('split')}/{row.get('file')}: "
            f"mean_H={row.get('mean_think_entropy')} n={row.get('n')} "
            f"multi={row.get('n_multi_turn')} stripped={row.get('n_stripped_offload')}",
            flush=True,
        )
    print(f"[infer] wrote {out_dir / 'summary.json'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        default=None,
        help="Jsonl file or directory. Repeatable. Default: phase3 filtered + offload.",
    )
    p.add_argument("--include", default="*.jsonl", help="Glob under each --data-dir")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("INFER_OUT_DIR", str(DEFAULT_OUT))),
    )
    p.add_argument(
        "--url",
        default=os.environ.get("VLLM_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or DEFAULT_VLLM_URL,
    )
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "EMPTY")
    p.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL") or os.environ.get("DASHSCOPE_MODEL") or DEFAULT_MODEL,
    )
    p.add_argument("--top-logprobs", type=int, default=int(os.environ.get("TOP_LOGPROBS", "20")))
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "1024")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.6")))
    p.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "0.95")))
    p.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "20")))
    p.add_argument("--timeout", type=float, default=float(os.environ.get("LLM_TIMEOUT", "600")))
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("INFER_CONCURRENCY", "8")))
    p.add_argument("--limit", type=int, default=None, help="Max samples per file (default: all)")
    p.add_argument("--offset", type=int, default=0, help="Skip this many samples per file")
    p.add_argument("--force", action="store_true", help="Delete existing per-file outputs first")
    p.add_argument("--resume", action="store_true", help="Skip sample indices already in the output jsonl")
    p.add_argument("--retry-failed", action="store_true", help="With existing outputs, re-score rows whose ok is false")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument(
        "--stop",
        action="append",
        default=None,
        help="Generation stop string. Default: '.' and '。' (first think sentence). Repeat to add more.",
    )
    p.add_argument("--no-stop", action="store_true", help="Do not stop at the first sentence marker")
    p.add_argument("--dry-run", action="store_true", help="Count turns / offload strips; do not call vLLM")
    args = p.parse_args()
    if args.data_dir is None:
        args.data_dir = list(DEFAULT_DATA_DIRS)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be > 0")
    if args.offset < 0:
        raise SystemExit("--offset must be >= 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")
    run(args)


if __name__ == "__main__":
    main()
