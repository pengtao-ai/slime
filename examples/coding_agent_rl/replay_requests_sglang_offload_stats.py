#!/usr/bin/env python3
"""Replay saved CC→GLM request dumps against a local SGLang chat.completions server.

Reads ``runs/.../i*/requests/req_*.json``, POSTs messages(+tools) to SGLang, then
counts ``<|llm_offload|>N<|/llm_offload|>`` spans in responses.

By default appends ``offload.OFFLOAD_SYSTEM_PROMPT_APPEND`` to each request's
system message (same text used in training):

    For very difficult steps, you can output <|llm_offload|>N<|/llm_offload|>
    where N is 0-9 indicating the thinking level for a more capable model.

Launch server::

    bash examples/coding_agent_rl/launch_sglang_pyrodash4b.sh

Run eval::

    bash examples/coding_agent_rl/run_eval_request_offload_stats.sh

Or directly::

    python examples/coding_agent_rl/replay_requests_sglang_offload_stats.py \\
      --run-dir runs/infer_cc_glm_20260728_094722 \\
      --url http://127.0.0.1:30000/v1 \\
      --model PyroDash-4B-SFT-0728 \\
      --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from examples.coding_agent_rl import offload  # noqa: E402

_OFFLOAD_SPAN_RE = re.compile(
    re.escape(offload.OFFLOAD_OPEN) + r"(\d)" + re.escape(offload.OFFLOAD_CLOSE)
)
_OFFLOAD_ANY_RE = re.compile(re.escape(offload.OFFLOAD_OPEN) + r"|" + re.escape(offload.OFFLOAD_CLOSE))


def _inject_offload_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append SLM offload instructions to the first system message (idempotent)."""
    append = offload.offload_system_append_text()
    if not append:
        return messages
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            content = str(m.get("content") or "")
            if append not in content:
                m["content"] = content.rstrip() + "\n\n" + append
            break
    else:
        out.insert(0, {"role": "system", "content": append})
    return out


def _collect_requests(run_dir: Path) -> list[Path]:
    paths = sorted(run_dir.glob("i*/requests/req_*.json"))
    if not paths:
        raise SystemExit(f"no req_*.json under {run_dir}/i*/requests/")
    return paths


def _response_text(data: dict[str, Any]) -> str:
    try:
        msg = data["choices"][0].get("message") or {}
    except (KeyError, IndexError, TypeError):
        return ""
    parts = [
        str(msg.get("reasoning_content") or msg.get("reasoning") or ""),
        str(msg.get("content") or ""),
    ]
    return "\n".join(p for p in parts if p)


def _call_one(
    *,
    url: str,
    model: str,
    path: Path,
    stop_token_ids: list[int],
    max_tokens: int,
    timeout: float,
    inject_offload: bool,
    temperature: float,
    no_stop_trim: bool,
    skip_special_tokens: bool,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    messages = list(payload.get("messages") or [])
    if inject_offload:
        messages = _inject_offload_system(messages)
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(payload.get("max_tokens") or max_tokens),
        "temperature": temperature,
        "stop_token_ids": stop_token_ids,
        # Keep <|/llm_offload|> in the decoded text (adapter /generate uses the same).
        "no_stop_trim": no_stop_trim,
        "skip_special_tokens": skip_special_tokens,
    }
    tools = payload.get("tools")
    if tools:
        body["tools"] = tools
    t0 = time.monotonic()
    err = None
    data: dict[str, Any] | None = None
    try:
        resp = requests.post(
            f"{url.rstrip('/')}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
        if resp.status_code != 200:
            err = f"HTTP {resp.status_code}: {resp.text[:400]}"
        else:
            data = resp.json()
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    text = _response_text(data) if data else ""
    spans = _OFFLOAD_SPAN_RE.findall(text)
    open_n = text.count(offload.OFFLOAD_OPEN)
    close_n = text.count(offload.OFFLOAD_CLOSE)
    msg = {}
    if data:
        try:
            msg = (data.get("choices") or [{}])[0].get("message") or {}
        except Exception:
            msg = {}
    return {
        "path": str(path),
        "sample": path.parents[1].name,
        "turn_index": payload.get("turn_index"),
        "elapsed_sec": round(time.monotonic() - t0, 3),
        "ok": err is None,
        "error": err,
        "offload_span_count": len(spans),
        "offload_ns": [int(n) for n in spans],
        "offload_open_count": open_n,
        "offload_close_count": close_n,
        "has_offload_marker": bool(_OFFLOAD_ANY_RE.search(text)),
        "response_chars": len(text),
        "response_content": str(msg.get("content") or ""),
        "response_reasoning": str(msg.get("reasoning_content") or msg.get("reasoning") or ""),
        "response_tool_calls": msg.get("tool_calls"),
        "usage": (data or {}).get("usage") if data else None,
        "finish_reason": (
            ((data or {}).get("choices") or [{}])[0].get("finish_reason") if data else None
        ),
        # Full wire payload POSTed to SGLang / returned by chat.completions.
        # Also mirrored under pairs/; decoded text under response_texts/.
        "request": body,
        "response": data,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=_REPO / "runs" / "infer_cc_glm_20260728_094722",
    )
    p.add_argument("--url", default=os.environ.get("SGLANG_URL", "http://127.0.0.1:30000/v1"))
    p.add_argument("--model", default=os.environ.get("SGLANG_MODEL", "PyroDash-4B-SFT-0728"))
    p.add_argument(
        "--stop-token-ids",
        default=os.environ.get("STOP_TOKEN_IDS", "248046,248044,248078"),
        help="Comma-separated stop_token_ids",
    )
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("REPLAY_CONCURRENCY", "8")))
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument(
        "--no-inject-offload",
        action="store_true",
        help="Do not append OFFLOAD_SYSTEM_PROMPT_APPEND to system (replay messages as-is)",
    )
    p.add_argument(
        "--trim-stop",
        action="store_true",
        help="Allow SGLang to strip stop tokens from output (default keeps "
        "<|/llm_offload|> via no_stop_trim=true)",
    )
    p.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Decode with skip_special_tokens=true (default false so offload tags stay)",
    )
    p.add_argument("--limit", type=int, default=0, help="Optional cap on number of requests (0=all)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write replay results (default: <run-dir>/sglang_replay_<stamp>)",
    )
    args = p.parse_args()

    stop_ids = [int(x.strip()) for x in str(args.stop_token_ids).split(",") if x.strip()]
    paths = _collect_requests(args.run_dir)
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (args.run_dir / f"sglang_replay_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    resp_dir = out_dir / "responses"
    text_dir = out_dir / "response_texts"
    pair_dir = out_dir / "pairs"
    resp_dir.mkdir(exist_ok=True)
    text_dir.mkdir(exist_ok=True)
    pair_dir.mkdir(exist_ok=True)

    # Health check
    try:
        r = requests.get(f"{args.url.rstrip('/')}/models", timeout=5)
        print(f"[replay] sglang OK {args.url} /models -> {r.status_code}", flush=True)
    except Exception as exc:
        raise SystemExit(
            f"SGLang not reachable at {args.url}: {exc}\n"
            f"Start e.g.\n"
            f"  bash examples/coding_agent_rl/launch_sglang_pyrodash4b.sh"
        )

    inject = not args.no_inject_offload
    no_stop_trim = not args.trim_stop
    skip_special = bool(args.skip_special_tokens)
    print(
        f"[replay] n={len(paths)} concurrency={args.concurrency} model={args.model} "
        f"stop={stop_ids} inject_offload={inject} no_stop_trim={no_stop_trim} "
        f"skip_special_tokens={skip_special}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = {
            ex.submit(
                _call_one,
                url=args.url,
                model=args.model,
                path=path,
                stop_token_ids=stop_ids,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                inject_offload=inject,
                temperature=args.temperature,
                no_stop_trim=no_stop_trim,
                skip_special_tokens=skip_special,
            ): path
            for path in paths
        }
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            results.append(row)
            done += 1
            # Persist full output next to mirrored request path.
            rel = Path(row["path"]).relative_to(args.run_dir)
            out_path = resp_dir / rel.with_suffix(".json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            txt_path = text_dir / rel.with_suffix(".txt")
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(_response_text(row.get("response") or {}), encoding="utf-8")
            # Clean request/response pair (exact wire payloads).
            pair_path = pair_dir / rel.with_suffix(".json")
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            pair_path.write_text(
                json.dumps(
                    {
                        "source_request_path": row["path"],
                        "sample": row["sample"],
                        "turn_index": row["turn_index"],
                        "elapsed_sec": row["elapsed_sec"],
                        "ok": row["ok"],
                        "error": row["error"],
                        "finish_reason": row["finish_reason"],
                        "offload_span_count": row["offload_span_count"],
                        "offload_open_count": row["offload_open_count"],
                        "offload_close_count": row["offload_close_count"],
                        "offload_ns": row["offload_ns"],
                        "request": row.get("request"),
                        "response": row.get("response"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if done % 10 == 0 or done == len(paths):
                off_so_far = sum(int(r["offload_span_count"]) for r in results)
                close_so_far = sum(int(r["offload_close_count"]) for r in results)
                ok = sum(1 for r in results if r["ok"])
                print(
                    f"[replay] {done}/{len(paths)} ok={ok} "
                    f"offload_spans={off_so_far} close_tags={close_so_far}",
                    flush=True,
                )

    results.sort(key=lambda r: r["path"])
    total_spans = sum(int(r["offload_span_count"]) for r in results)
    total_open = sum(int(r["offload_open_count"]) for r in results)
    total_close = sum(int(r["offload_close_count"]) for r in results)
    reqs_with_offload = sum(1 for r in results if r["offload_span_count"] > 0)
    reqs_with_close = sum(1 for r in results if r["offload_close_count"] > 0)
    samples_with_offload = sorted(
        {r["sample"] for r in results if r["offload_span_count"] > 0 or r["offload_close_count"] > 0}
    )
    n_hist: dict[str, int] = {}
    for r in results:
        for n in r["offload_ns"]:
            n_hist[str(n)] = n_hist.get(str(n), 0) + 1

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir),
        "url": args.url,
        "model": args.model,
        "stop_token_ids": stop_ids,
        "no_stop_trim": no_stop_trim,
        "skip_special_tokens": skip_special,
        "inject_offload_system": inject,
        "n_requests": len(results),
        "n_ok": sum(1 for r in results if r["ok"]),
        "n_failed": sum(1 for r in results if not r["ok"]),
        "offload_span_total": total_spans,
        "offload_open_total": total_open,
        "offload_close_total": total_close,
        "requests_with_offload_span": reqs_with_offload,
        "requests_with_offload_close": reqs_with_close,
        "samples_with_offload": samples_with_offload,
        "offload_n_histogram": n_hist,
        "mean_elapsed_sec": (
            sum(float(r["elapsed_sec"]) for r in results) / len(results) if results else 0.0
        ),
        "artifacts": {
            "summary": "summary.json",
            "results_jsonl": "results.jsonl",
            "pairs": "pairs/**/req_XX.json  # {request, response} full wire payloads",
            "responses_json": "responses/**/req_XX.json",
            "response_texts": "response_texts/**/req_XX.txt",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("======== offload stats ========", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[replay] wrote {out_dir}", flush=True)
    print(
        f"[replay] outputs: {out_dir}/pairs/  {out_dir}/responses/  "
        f"{out_dir}/response_texts/  {out_dir}/results.jsonl",
        flush=True,
    )


if __name__ == "__main__":
    main()
