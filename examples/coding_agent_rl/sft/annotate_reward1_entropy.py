#!/usr/bin/env python3
"""Annotate reward=1 trajectories: last req JSON, entropy on every assistant.

Pipeline
--------
1. Scan ``infer_cc_*`` run dirs; keep samples with ``summary.reward >= 1``.
2. For each sample, load **only the last** ``requests/req_*.json``.
3. Build the full chat: ``messages`` + final ``response`` as assistant.
4. For **each** assistant message, POST the prefix (messages before it) to
   vLLM with ``logprobs`` / ``top_logprobs=20`` (same scoring as
   ``offload_entropybased``), then write ``entropy`` onto that assistant.
5. Write a **new** annotated JSON under ``--out-dir`` (never modifies the
   original ``runs/.../requests/req_*.json``). Same schema; each assistant /
   final ``response`` gains ``entropy`` + ``vllm_response``.

Example::

    python annotate_reward1_entropy.py \\
      --run-dir /workspace/work/spt/slime/runs/infer_cc_dsv4flash_20260806_141118 \\
      --url http://127.0.0.1:8066/v1 \\
      --model /workspace/models/pyromind/PyroDash-4B-SFT-0902 \\
      --limit-samples 2 --concurrency 2 --assistant-concurrency 4
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
_ENTROPY_ROOT = Path(
    os.environ.get("OFFLOAD_ENTROPY_ROOT", "/workspace/work/mjy/offload_entropybased")
)
if _ENTROPY_ROOT.is_dir():
    sys.path.insert(0, str(_ENTROPY_ROOT))

from adapter.entropy import (  # noqa: E402
    average_turn_entropy,
    extract_choice_logprobs,
)

_REQ_RE = re.compile(r"^req_(\d+)\.json$")
_ANNOTATION_KEYS = frozenset({"entropy", "vllm_response"})
_DEFAULT_RUNS = [
    Path("/workspace/work/spt/slime/runs/infer_cc_glm_20260728_141118"),
    Path("/workspace/work/spt/slime/runs/infer_cc_dsv4flash_20260806_141118"),
    Path("/workspace/work/spt/slime/runs/infer_cc_tmax_20260807_162106"),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _reward_of(sample_dir: Path) -> float | None:
    summary = sample_dir / "summary.json"
    if not summary.is_file():
        return None
    try:
        raw = _load_json(summary).get("reward")
        return float(raw if raw is not None else 0.0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _list_reward1_samples(run_dir: Path) -> list[Path]:
    out: list[Path] = []
    for sample_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("i")):
        reward = _reward_of(sample_dir)
        if reward is not None and reward >= 1.0:
            out.append(sample_dir)
    return out


def _last_request(sample_dir: Path) -> tuple[int, Path, dict[str, Any]] | None:
    req_dir = sample_dir / "requests"
    if not req_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for path in req_dir.iterdir():
        m = _REQ_RE.match(path.name)
        if not m or not path.is_file():
            continue
        idx = int(m.group(1))
        if best is None or idx > best[0]:
            best = (idx, path)
    if best is None:
        return None
    try:
        payload = _load_json(best[1])
    except (OSError, json.JSONDecodeError):
        return None
    return best[0], best[1], payload


def _response_to_assistant(response: dict[str, Any] | None) -> dict[str, Any]:
    resp = response or {}
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": resp.get("content") if resp.get("content") is not None else "",
    }
    reasoning = resp.get("reasoning_content") or resp.get("reasoning")
    if reasoning:
        msg["reasoning_content"] = reasoning
    tool_calls = resp.get("tool_calls") or []
    if tool_calls:
        msg["tool_calls"] = tool_calls
    for key in _ANNOTATION_KEYS:
        if key in resp:
            msg[key] = resp[key]
    return msg


def _full_messages_from_last_req(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """History messages + final response as the last assistant."""
    messages = [dict(m) for m in (payload.get("messages") or [])]
    messages.append(_response_to_assistant(payload.get("response")))
    return messages


def _assistant_indices(messages: list[dict[str, Any]]) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def _entropy_payload(result: Any) -> dict[str, Any]:
    d = result.to_dict()
    return {
        "avg_entropy": d.get("avg_entropy"),
        "n_tokens": d.get("n_tokens"),
        "n_used": d.get("n_used"),
        "unavailable": d.get("unavailable"),
        "scope": d.get("scope"),
    }


def _choice_message(data: dict[str, Any]) -> dict[str, Any]:
    try:
        return (data.get("choices") or [{}])[0].get("message") or {}
    except Exception:
        return {}


def _message_reasoning_text(data: dict[str, Any]) -> str:
    msg = _choice_message(data)
    return str(msg.get("reasoning_content") or msg.get("reasoning") or "")


def _split_entries_by_prefix(
    entries: list[Any], prefix: str
) -> tuple[list[Any], list[Any]]:
    """Split token entries so join(tokens[:k]) covers ``prefix`` when possible."""
    if not prefix:
        return [], list(entries)
    built = ""
    for i, entry in enumerate(entries):
        tok = str((entry or {}).get("token") or "") if isinstance(entry, dict) else ""
        built += tok
        if built.startswith(prefix) and len(built) >= len(prefix):
            return list(entries[: i + 1]), list(entries[i + 1 :])
    # Truncated generation: all tokens are still inside reasoning.
    if prefix.startswith(built) and built:
        return list(entries), []
    return [], list(entries)


def _split_entries_at_stop(entries: list[Any], stop: str) -> tuple[list[Any], list[Any]]:
    if not stop:
        return [], list(entries)
    built = ""
    for i, entry in enumerate(entries):
        tok = str((entry or {}).get("token") or "") if isinstance(entry, dict) else ""
        built += tok
        if stop in built:
            return list(entries[: i + 1]), list(entries[i + 1 :])
    return [], list(entries)


def _remap_logprobs_for_thinking(logprobs: Any, *, reasoning: str) -> Any:
    """Relabel content-bucket logprobs that belong to the thinking segment.

    With ``--reasoning-parser``, vLLM puts think text on ``message.reasoning``
    but still returns all token logprobs under ``logprobs.content``. Remap the
    reasoning-text prefix (or tokens up to ``</think>``) to source=reasoning.
    """
    if not isinstance(logprobs, dict):
        return logprobs
    if logprobs.get("reasoning_content") or logprobs.get("reasoning"):
        return logprobs
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return logprobs

    if reasoning:
        think_entries, rest = _split_entries_by_prefix(content, reasoning)
    else:
        think_entries, rest = _split_entries_at_stop(content, "</think>")

    if not think_entries:
        return logprobs

    remapped = dict(logprobs)
    remapped["reasoning"] = think_entries
    remapped["content"] = rest
    return remapped


def _compute_entropy_from_response(data: dict[str, Any], *, scope: str) -> dict[str, Any]:
    logprobs = extract_choice_logprobs(data)
    if scope == "thinking":
        logprobs = _remap_logprobs_for_thinking(
            logprobs, reasoning=_message_reasoning_text(data)
        )
    primary = average_turn_entropy(logprobs, scope=scope)
    if primary.unavailable and scope == "thinking":
        fallback = average_turn_entropy(logprobs, scope="all")
        payload = _entropy_payload(fallback)
        payload["scope_requested"] = "thinking"
        payload["scope_fallback"] = "all"
        return payload
    payload = _entropy_payload(primary)
    payload["scope_requested"] = scope
    return payload


def _strip_annotations(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in m.items() if k not in _ANNOTATION_KEYS} for m in messages]


def _vllm_response_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Save the full Qwen/vLLM assistant message (no logprobs).

    Keeps content / reasoning / tool_calls even when empty, plus finish_reason
    and usage. Requires generating the full turn (no think-stop truncation).
    """
    try:
        choice = (data.get("choices") or [{}])[0] or {}
    except Exception:
        choice = {}
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning")
    if reasoning is None:
        reasoning = msg.get("reasoning_content")
    out: dict[str, Any] = {
        "role": msg.get("role") or "assistant",
        "content": msg.get("content"),
        "reasoning": reasoning,
        "tool_calls": msg.get("tool_calls"),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
    }
    # Drop only unused null optionals; keep content/reasoning keys always.
    return {
        k: v
        for k, v in out.items()
        if v is not None or k in ("content", "reasoning")
    }


def _build_vllm_body(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[Any] | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    top_logprobs: int,
    include_tools: bool,
    enable_thinking: bool,
    stop: list[str] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": _strip_annotations(messages),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "logprobs": True,
        "top_logprobs": int(top_logprobs),
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    if include_tools and tools:
        body["tools"] = tools
    # Optional only; with --reasoning-parser qwen3 do not default-stop on </think>.
    if stop:
        body["stop"] = list(stop)
        body["include_stop_str_in_output"] = True
    return body


def _call_vllm(
    *,
    url: str,
    api_key: str,
    body: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any] | None, str | None, float]:
    t0 = time.monotonic()
    try:
        resp = requests.post(
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


def _sample_out_dir(out_root: Path, run_dir: Path, sample_dir: Path) -> Path:
    return out_root / run_dir.name / sample_dir.name


def _score_one_assistant(
    *,
    asst_ord: int,
    msg_index: int,
    prefix: list[dict[str, Any]],
    tools: list[Any] | None,
    url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    top_logprobs: int,
    include_tools: bool,
    enable_thinking: bool,
    entropy_scope: str,
    stop: list[str] | None,
    timeout: float,
) -> dict[str, Any]:
    body = _build_vllm_body(
        model=model,
        messages=prefix,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        top_logprobs=top_logprobs,
        include_tools=include_tools,
        enable_thinking=enable_thinking,
        stop=stop,
    )
    data, err, elapsed = _call_vllm(url=url, api_key=api_key, body=body, timeout=timeout)
    row: dict[str, Any] = {
        "assistant_ord": asst_ord,
        "msg_index": msg_index,
        "prefix_len": len(prefix),
        "elapsed_sec": round(elapsed, 3),
        "ok": err is None,
        "error": err,
    }
    if data is not None:
        # Entropy on reasoning tokens only (see _remap_logprobs_for_thinking).
        # Full Qwen message (content / reasoning / tool_calls) is saved as-is.
        row["entropy"] = _compute_entropy_from_response(data, scope=entropy_scope)
        row["vllm_response"] = _vllm_response_payload(data)
        try:
            finish = (data.get("choices") or [{}])[0].get("finish_reason")
        except Exception:
            finish = None
        row["finish_reason"] = finish
    return row


def _annotated_req_name(req_path: Path) -> str:
    """e.g. req_12.json -> req_12_entropy.json"""
    return f"{req_path.stem}_entropy.json"


def _process_sample(
    *,
    run_dir: Path,
    sample_dir: Path,
    out_root: Path,
    url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    top_logprobs: int,
    include_tools: bool,
    enable_thinking: bool,
    entropy_scope: str,
    stop: list[str] | None,
    timeout: float,
    resume: bool,
    assistant_concurrency: int,
) -> dict[str, Any]:
    dest = _sample_out_dir(out_root, run_dir, sample_dir)
    last = _last_request(sample_dir)
    if last is None:
        return {
            "sample": sample_dir.name,
            "run": run_dir.name,
            "ok": False,
            "error": "no last request",
            "out": str(dest),
        }

    turn_index, req_path, payload = last
    out_path = dest / _annotated_req_name(req_path)
    if resume and out_path.is_file():
        return {
            "sample": sample_dir.name,
            "run": run_dir.name,
            "ok": True,
            "skipped": True,
            "last_req": req_path.name,
            "out": str(out_path),
        }

    summary = _load_json(sample_dir / "summary.json")
    # Deep copy so the original on-disk req is never mutated.
    payload = copy.deepcopy(payload)
    out_messages = list(payload.get("messages") or [])
    full_messages = out_messages + [_response_to_assistant(payload.get("response"))]
    asst_idxs = _assistant_indices(full_messages)
    tools = list(payload.get("tools") or [])
    req_max = min(int(payload.get("max_tokens") or max_tokens), max_tokens)

    if not asst_idxs:
        return {
            "sample": sample_dir.name,
            "run": run_dir.name,
            "ok": False,
            "error": "no assistant messages in last request",
            "out": str(dest),
        }

    call_rows: list[dict[str, Any]] = []

    def _one(asst_ord: int, msg_index: int) -> dict[str, Any]:
        return _score_one_assistant(
            asst_ord=asst_ord,
            msg_index=msg_index,
            prefix=full_messages[:msg_index],
            tools=tools,
            url=url,
            api_key=api_key,
            model=model,
            max_tokens=req_max,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            top_logprobs=top_logprobs,
            include_tools=include_tools,
            enable_thinking=enable_thinking,
            entropy_scope=entropy_scope,
            stop=stop,
            timeout=timeout,
        )

    if assistant_concurrency <= 1:
        for ord_i, msg_i in enumerate(asst_idxs):
            call_rows.append(_one(ord_i, msg_i))
    else:
        with ThreadPoolExecutor(max_workers=assistant_concurrency) as pool:
            futs = {
                pool.submit(_one, ord_i, msg_i): ord_i
                for ord_i, msg_i in enumerate(asst_idxs)
            }
            by_ord: dict[int, dict[str, Any]] = {}
            for fut in as_completed(futs):
                row = fut.result()
                by_ord[int(row["assistant_ord"])] = row
            call_rows = [by_ord[i] for i in range(len(asst_idxs))]

    n_fail = sum(1 for r in call_rows if not r.get("ok"))
    n_hist = len(out_messages)
    out_response = payload.get("response")
    if not isinstance(out_response, dict):
        out_response = {}
        payload["response"] = out_response
    for row in call_rows:
        idx = int(row["msg_index"])
        target = out_messages[idx] if idx < n_hist else out_response
        if row.get("entropy") is not None:
            target["entropy"] = row["entropy"]
        if row.get("vllm_response") is not None:
            target["vllm_response"] = row["vllm_response"]

    avg_vals = [
        float(r["entropy"]["avg_entropy"])
        for r in call_rows
        if isinstance((r.get("entropy") or {}).get("avg_entropy"), (int, float))
    ]
    mean_avg = round(sum(avg_vals) / len(avg_vals), 6) if avg_vals else None

    # New file under --out-dir; never overwrite runs/.../requests/req_*.json.
    payload["messages"] = out_messages
    dest.mkdir(parents=True, exist_ok=True)
    _dump_json(out_path, payload)

    sft_row = {
        "sid": summary.get("session_id") or summary.get("instance_id") or sample_dir.name,
        "instance_id": summary.get("instance_id"),
        "reward": summary.get("reward"),
        "source_run": run_dir.name,
        "source_sample": sample_dir.name,
        "source_req": str(req_path),
        "last_req": req_path.name,
        "annotated_req": out_path.name,
        "n_assistants": len(asst_idxs),
        "messages": _full_messages_from_last_req(payload),
        "tools": tools,
    }
    (dest / "sft_messages.json").write_text(
        json.dumps(sft_row, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return {
        "sample": sample_dir.name,
        "run": run_dir.name,
        "ok": n_fail == 0,
        "skipped": False,
        "last_req": req_path.name,
        "annotated_req": out_path.name,
        "last_turn_index": turn_index,
        "n_assistants": len(asst_idxs),
        "n_fail": n_fail,
        "out": str(out_path),
        "mean_avg_entropy": mean_avg,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", action="append", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=_SCRIPT_DIR / "trajectories_entropy")
    p.add_argument("--url", default=os.environ.get("VLLM_URL", "http://127.0.0.1:8066/v1"))
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    p.add_argument(
        "--model",
        default=os.environ.get(
            "VLLM_MODEL", "/workspace/models/pyromind/PyroDash-4B-SFT-0902"
        ),
    )
    p.add_argument("--top-logprobs", type=int, default=int(os.environ.get("TOP_LOGPROBS", "20")))
    p.add_argument(
        "--entropy-scope",
        default=os.environ.get("ENTROPY_SCOPE", "thinking"),
        choices=("thinking", "all"),
    )
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "8192")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.6")))
    p.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "0.95")))
    p.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "20")))
    p.add_argument("--timeout", type=float, default=float(os.environ.get("TIMEOUT", "600")))
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("SAMPLE_CONCURRENCY", "2")),
        help="Parallel samples",
    )
    p.add_argument(
        "--assistant-concurrency",
        type=int,
        default=int(os.environ.get("ASSISTANT_CONCURRENCY", "4")),
        help="Parallel vLLM calls for assistants within one sample",
    )
    p.add_argument("--limit-samples", type=int, default=None)
    p.add_argument("--offset-samples", type=int, default=0)
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument(
        "--stop",
        action="append",
        default=None,
        help="Optional vLLM stop string(s). Default: none (full response; "
        "use with --reasoning-parser qwen3). Pass multiple times for several.",
    )
    p.add_argument(
        "--no-stop",
        action="store_true",
        help="Explicitly disable stop strings (default behavior).",
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Default: no stop — save full Qwen response; entropy uses reasoning remap.
    if args.no_stop:
        stop_strs: list[str] | None = None
    elif args.stop:
        stop_strs = list(args.stop)
    else:
        env_stop = os.environ.get("STOP_STR", "").strip()
        stop_strs = [env_stop] if env_stop else None

    run_dirs = [Path(x).resolve() for x in (args.run_dir or _DEFAULT_RUNS)]
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, Path]] = []
    inventory: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"[warn] missing run dir: {run_dir}", file=sys.stderr)
            continue
        print(f"[scan] {run_dir.name} ...", flush=True)
        samples = _list_reward1_samples(run_dir)
        samples = samples[args.offset_samples :]
        if args.limit_samples is not None:
            samples = samples[: args.limit_samples]
        n_asst_est = 0
        for s in samples:
            try:
                # summary.turns ≈ number of assistants / last-req response turns
                n_asst_est += int((_load_json(s / "summary.json").get("turns") or 0))
            except Exception:
                pass
            jobs.append((run_dir, s))
        inventory.append(
            {
                "run_dir": str(run_dir),
                "reward1": len(samples),
                "n_vllm_calls_est": n_asst_est,
            }
        )
        print(
            f"[scan] {run_dir.name}: reward1={len(samples)} "
            f"vllm_calls≈{n_asst_est} (1 per assistant in last req)",
            flush=True,
        )

    _dump_json(
        out_root / "inventory.json",
        {
            "created_at": _utc_now(),
            "mode": "last_req_all_assistants",
            "runs": inventory,
            "n_jobs": len(jobs),
        },
    )
    if args.dry_run:
        print(f"[dry-run] {len(jobs)} samples -> {out_root}", flush=True)
        return

    if not jobs:
        raise SystemExit("no reward=1 samples found")

    try:
        health = requests.get(
            f"{args.url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {args.api_key}"},
            timeout=10,
        )
        print(f"[vllm] GET /models -> HTTP {health.status_code}", flush=True)
        if health.status_code != 200:
            print(health.text[:300], file=sys.stderr)
    except Exception as exc:
        print(f"[warn] vLLM health check failed: {exc}", file=sys.stderr)

    results: list[dict[str, Any]] = []
    t0 = time.monotonic()
    resume = not args.no_resume

    def _run_job(pair: tuple[Path, Path]) -> dict[str, Any]:
        run_dir, sample_dir = pair
        try:
            return _process_sample(
                run_dir=run_dir,
                sample_dir=sample_dir,
                out_root=out_root,
                url=args.url,
                api_key=args.api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                top_logprobs=args.top_logprobs,
                include_tools=not args.no_tools,
                enable_thinking=not args.no_thinking,
                entropy_scope=args.entropy_scope,
                stop=stop_strs,
                timeout=args.timeout,
                resume=resume,
                assistant_concurrency=max(1, args.assistant_concurrency),
            )
        except Exception as exc:
            return {
                "sample": sample_dir.name,
                "run": run_dir.name,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-800:],
            }

    print(
        f"[run] jobs={len(jobs)} sample_concurrency={args.concurrency} "
        f"assistant_concurrency={args.assistant_concurrency} out={out_root}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futs = [pool.submit(_run_job, job) for job in jobs]
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            results.append(row)
            done += 1
            if done % 10 == 0 or done == len(jobs):
                ok = sum(1 for r in results if r.get("ok"))
                skip = sum(1 for r in results if r.get("skipped"))
                print(
                    f"[progress] {done}/{len(jobs)} ok={ok} skipped={skip} "
                    f"elapsed={time.monotonic()-t0:.1f}s last={row.get('run')}/{row.get('sample')}",
                    flush=True,
                )

    summary = {
        "created_at": _utc_now(),
        "mode": "last_req_all_assistants",
        "out_dir": str(out_root),
        "url": args.url,
        "model": args.model,
        "n_jobs": len(jobs),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_fail": sum(1 for r in results if not r.get("ok")),
        "n_skipped": sum(1 for r in results if r.get("skipped")),
        "elapsed_sec": round(time.monotonic() - t0, 3),
        "inventory": inventory,
        "results": results,
    }
    _dump_json(out_root / "summary.json", summary)

    jsonl_path = out_root / "sft_messages.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for r in results:
            out = r.get("out")
            if not out:
                continue
            out_p = Path(out)
            sft_path = (out_p.parent if out_p.suffix == ".json" else out_p) / "sft_messages.json"
            if sft_path.is_file():
                fout.write(sft_path.read_text(encoding="utf-8").rstrip("\n") + "\n")

    print(
        f"[done] ok={summary['n_ok']} fail={summary['n_fail']} skipped={summary['n_skipped']} "
        f"-> {out_root} ({summary['elapsed_sec']}s)",
        flush=True,
    )
    if summary["n_ok"] == 0 and summary["n_skipped"] == 0:
        raise SystemExit("FAIL: no successful samples")


if __name__ == "__main__":
    main()
