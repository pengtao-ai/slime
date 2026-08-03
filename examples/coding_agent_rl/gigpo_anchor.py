"""Structured-first env-obs anchors for coding-agent GiGPO.

``anchor_obs`` is a stable string key; equality ⇒ same step group.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

_VOLATILE_RE = re.compile(
    r"(agentId\s*[:=]\s*\S+|"
    r"duration_ms\s*[:=]\s*\d+|"
    r"\(internal ID[^)]*\)|"
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\s]*|"
    r"toolu_[A-Za-z0-9]+|"
    r"call_[A-Za-z0-9]+)",
    re.IGNORECASE,
)
_PERSIST_RE = re.compile(
    r"Output too large[^\n]*\n.*?saved to:\s*(\S+).*?Preview[^\n]*\n(.*)",
    re.IGNORECASE | re.DOTALL,
)
_LINE_NUM_RE = re.compile(r"(?m)^\s*\d+\t")


def gigpo_enabled() -> bool:
    return os.environ.get("SLIME_GIGPO", "").strip().lower() in ("1", "true", "yes", "on")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


def normalize_tool_text(text: str, *, k: int = 4096) -> str:
    t = (text or "").replace("\r\n", "\n")
    t = _VOLATILE_RE.sub("", t)
    m = _PERSIST_RE.search(t)
    if m:
        path, preview = m.group(1), m.group(2)[:2048]
        t = f"PERSISTED|{path}|PREVIEW|{preview}"
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) > 2 * k:
        t = t[:k] + "\n…\n" + t[-k:]
    return t


def strip_read_line_numbers(text: str) -> str:
    return _LINE_NUM_RE.sub("", text or "")


def canonical_tool_args(tool_name: str, tool_input: dict[str, Any] | None, *, workdir: str = "") -> str:
    inp = dict(tool_input or {})
    name = (tool_name or "").strip()

    def _rel(path: str) -> str:
        p = str(path or "").strip()
        if workdir and p.startswith(workdir.rstrip("/") + "/"):
            return p[len(workdir.rstrip("/")) + 1 :]
        return p

    if name in ("Read", "Edit", "Write", "NotebookEdit"):
        key = {
            "path": _rel(str(inp.get("file_path") or inp.get("path") or "")),
            "offset": inp.get("offset"),
            "limit": inp.get("limit"),
        }
        return _sha(json.dumps(key, sort_keys=True, default=str))
    if name == "Bash":
        cmd = re.sub(r"\s+", " ", str(inp.get("command") or "").strip())
        return _sha(cmd)
    if name in ("Grep", "Glob", "SearchFiles"):
        key = {
            "pattern": inp.get("pattern") or inp.get("query"),
            "path": _rel(str(inp.get("path") or inp.get("glob") or "")),
            "flags": inp.get("flags") or inp.get("case_insensitive"),
        }
        return _sha(json.dumps(key, sort_keys=True, default=str))
    if name == "Agent":
        blob = f"{inp.get('description', '')}|{inp.get('prompt', '')}"
        return _sha(blob)
    return _sha(json.dumps(inp, sort_keys=True, default=str))


def content_fingerprint(tool_name: str, tool_result_text: str, *, tool_input: dict[str, Any] | None = None) -> str:
    name = (tool_name or "").strip()
    text = tool_result_text or ""
    if name == "Read":
        body = strip_read_line_numbers(text)
        return _sha(normalize_tool_text(body))
    if "Output too large" in text or "persisted-output" in text.lower():
        norm = normalize_tool_text(text)
        return _sha(norm)
    if len(text) <= 4096:
        return _sha(normalize_tool_text(text))
    return _sha(normalize_tool_text(text, k=2048))


def init_anchor_obs(instance_id: str, problem_statement: str, *, branch_key: str = "main") -> str:
    return json.dumps(
        {
            "instance_id": instance_id,
            "branch_key": branch_key,
            "obs": ["__init__", _sha(problem_statement or "")],
        },
        sort_keys=True,
    )


def make_anchor_obs(
    *,
    instance_id: str,
    branch_key: str,
    tool_name: str | None,
    tool_input: dict[str, Any] | None,
    tool_result_text: str | None,
    workdir: str = "",
    is_init: bool = False,
    problem_statement: str = "",
) -> str:
    if is_init or not tool_name:
        return init_anchor_obs(instance_id, problem_statement, branch_key=branch_key)
    args_key = canonical_tool_args(tool_name, tool_input, workdir=workdir)
    body_key = content_fingerprint(tool_name, tool_result_text or "", tool_input=tool_input)
    return json.dumps(
        {
            "instance_id": instance_id,
            "branch_key": branch_key,
            "obs": [tool_name, args_key, body_key],
        },
        sort_keys=True,
    )


def branch_key_from_user_task(user_task: str, *, episode_user: str) -> str:
    u = (user_task or "").strip()
    e = (episode_user or "").strip()
    if not u:
        return "main"
    if e and (u == e or e[:120] in u or u[:120] in e):
        return "main"
    # Common coding-agent episode prompt prefix.
    if "PROBLEM_STATEMENT.md" in u and ("resolve the issue" in u.lower() or "fix the issue" in u.lower()):
        return "main"
    return f"sub:{_sha(u)[:16]}"


def extract_tool_name_and_input(message: dict[str, Any] | None) -> tuple[str | None, dict[str, Any] | None]:
    if not message:
        return None, None
    # Anthropic-style content blocks
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("tool_use", "function"):
                name = block.get("name") or (block.get("function") or {}).get("name")
                inp = block.get("input") or block.get("arguments")
                if isinstance(inp, str):
                    try:
                        inp = json.loads(inp)
                    except json.JSONDecodeError:
                        inp = {"raw": inp}
                return (str(name) if name else None), (inp if isinstance(inp, dict) else None)
    # OpenAI-style tool_calls
    tcs = message.get("tool_calls")
    if isinstance(tcs, list) and tcs:
        tc0 = tcs[0] if isinstance(tcs[0], dict) else {}
        fn = tc0.get("function") if isinstance(tc0.get("function"), dict) else {}
        name = fn.get("name") or tc0.get("name")
        args = fn.get("arguments") or tc0.get("arguments") or tc0.get("input")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        return (str(name) if name else None), (args if isinstance(args, dict) else None)
    return None, None


def extract_text_content(message: dict[str, Any] | None) -> str:
    if not message:
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in ("text", "tool_result", "output_text"):
                    parts.append(str(block.get("text") or block.get("content") or ""))
                elif "content" in block:
                    c = block["content"]
                    parts.append(c if isinstance(c, str) else json.dumps(c, default=str))
        return "\n".join(p for p in parts if p)
    return ""
