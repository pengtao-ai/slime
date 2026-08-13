"""RTMC-style state / action signatures for coding-agent tool histories.

StateSignature(t) is the deterministic encoding of all actions *before* step t.
ActionSignature(action_t) encodes only the current tool interaction.

See ``RTMC: Step-Level Credit Assignment via Rollout Trees`` (arXiv:2604.11037).
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

BUCKET = 100
CATEGORIES = (
    "view",
    "search",
    "modify",
    "create",
    "execute",
    "test",
    "install",
    "fileop",
    "think",
    "finish",
)
FLAG_KEYS = ("test_fail_count", "test_pass_count", "think_count")

_LINE_PREFIX_RE = re.compile(r"^(\d+)\t", re.MULTILINE)
_PATH_TOKEN_RE = re.compile(r"(?:^|[\s=`\"'(])(/?(?:[\w.-]+/)+[\w.-]+)")


def content_hash(old_text: str, new_text: str) -> str:
    """First 4 hex chars of MD5(old_text + new_text)."""
    payload = f"{old_text}{new_text}".encode("utf-8", errors="replace")
    return hashlib.md5(payload).hexdigest()[:4]


def normalize_path(path: str | None) -> str | None:
    if path is None:
        return None
    text = str(path).strip().strip("'\"")
    if not text:
        return None
    text = os.path.normpath(text).replace("\\", "/")
    if text in {".", ""}:
        return "."
    return text


def view_buckets(start_line: int, end_line: int) -> str:
    """Encode a 1-indexed inclusive line range as Vf / V[b] / V[start-end].

    Bucket index is ``line // 100`` (lines 200–299 → ``V[2]``).
    """
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    lo = start_line // BUCKET
    hi = end_line // BUCKET
    if lo == hi:
        return f"V[{lo}]"
    return f"V[{lo}-{hi}]"


def _as_dict(action: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(action, Mapping):
        return dict(action)
    return {
        "name": getattr(action, "name", None),
        "arguments": getattr(action, "arguments", None) or getattr(action, "input", None),
        "result": getattr(action, "result", None),
    }


def _args(action: Mapping[str, Any]) -> dict[str, Any]:
    raw = action.get("arguments") or action.get("input") or action.get("args") or {}
    return raw if isinstance(raw, dict) else {}


def _arg(args: Mapping[str, Any], *keys: str, strip_newlines: bool = True) -> str:
    for key in keys:
        if key in args and args[key] is not None:
            text = str(args[key])
            return text.strip("\n") if strip_newlines else text
    return ""


def _result_text(action: Mapping[str, Any]) -> str:
    for key in ("result", "tool_result", "content", "output"):
        val = action.get(key)
        if isinstance(val, str):
            return val
    return ""


def _infer_line_range(args: Mapping[str, Any], result: str) -> tuple[int, int] | None:
    offset_s = _arg(args, "offset", "start_line", "startLine")
    limit_s = _arg(args, "limit", "count")
    hits = [int(m.group(1)) for m in _LINE_PREFIX_RE.finditer(result)]
    if offset_s.isdigit():
        start = int(offset_s)
        if limit_s.isdigit():
            return start, start + int(limit_s) - 1
        if hits:
            return start, hits[-1]
        return start, start
    if hits:
        return hits[0], hits[-1]
    return None


def _looks_full_view(args: Mapping[str, Any], result: str) -> bool:
    if _arg(args, "offset", "start_line", "startLine", "limit"):
        return False
    rng = _infer_line_range(args, result)
    if rng is None:
        return True
    return rng[0] <= 1


def _command_ok(result: str) -> bool:
    text = result.lower()
    if re.search(r"\b[1-9]\d* failed\b", text):
        return False
    if re.search(r"\b(traceback|assertionerror|command not found)\b", text):
        return False
    if re.search(r"exit_code['\"]?\s*[:=]\s*[1-9]", text):
        return False
    return True


def _first_path_in_text(text: str) -> str | None:
    match = _PATH_TOKEN_RE.search(text or "")
    return normalize_path(match.group(1)) if match else None


def _bash_kind(command: str) -> str:
    head = command.strip()
    # strip common wrappers
    head = re.sub(r"^(?:cd\s+\S+\s*(?:&&|;)\s*)+", "", head).strip()
    low = head.lower()
    if re.match(r"(pip3?|uv pip|conda|apt(?:-get)?|yum|npm|pnpm|yarn)\s+install\b", low):
        return "install"
    if re.search(r"\b(pytest|unittest|cargo test|npm test|nox|tox)\b", low):
        return "test"
    if re.match(r"(mkdir|rm|rmdir|cp|mv|chmod|chown|touch|ln)\b", low):
        return "fileop"
    if re.match(r"(ls|find|grep|rg|ag|fd|glob)\b", low):
        return "search"
    if re.match(r"(cat|head|tail|less|more|nl|sed\s+-n)\b", low):
        return "view"
    if re.match(r"(python3?|pytest|node|bash|sh)\b", low):
        return "execute"
    return "execute"


@dataclass
class NormalizedAction:
    category: str
    path: str | None
    file_ops: tuple[str, ...] = ()
    scope: str = ""
    result_tag: str | None = None
    flag_deltas: dict[str, int] = field(default_factory=dict)

    def action_signature(self) -> str:
        cat = self.category
        if cat in {"think", "finish"} and not self.path:
            return cat
        scope = self.scope
        target = self.path or "_"
        if scope:
            body = f"{cat}:{scope}@{target}"
        else:
            body = f"{cat}@{target}"
        if self.result_tag:
            return f"{body}:{self.result_tag}"
        return body


def normalize_action(action: Mapping[str, Any] | Any) -> NormalizedAction:
    raw = _as_dict(action)
    name = str(raw.get("name") or raw.get("tool") or raw.get("category") or "").strip()
    args = _args(raw)
    result = _result_text(raw)
    return _normalize(name, args, result, raw)


def _normalize(name: str, args: dict[str, Any], result: str, raw: dict[str, Any]) -> NormalizedAction:
    lower = name.lower()
    explicit = str(raw.get("category") or "").lower()
    if explicit in CATEGORIES:
        return _from_explicit(explicit, args, result)

    if lower in {"think", "reasoning"} or (not name and (raw.get("think") or raw.get("text")) and not args):
        return NormalizedAction("think", None, flag_deltas={"think_count": 1})
    if lower in {"finish", "submit", "exit", "exitworktree"}:
        return NormalizedAction("finish", None)

    if lower in {"read", "view", "cat", "head", "tail"}:
        return _as_view(args, result, path=_arg(args, "file_path", "path", "target_file"))
    if lower in {"grep", "glob", "search", "find"}:
        path = normalize_path(_arg(args, "path", "file_path", "target_file")) or "."
        return NormalizedAction("search", path, file_ops=("S",))
    if lower in {"edit", "strreplace", "str_replace", "replace"}:
        return _as_edit(args)
    if lower in {"write", "create", "create_file"}:
        path = normalize_path(_arg(args, "file_path", "path"))
        return NormalizedAction("create", path, file_ops=("C",) if path else (), scope="")
    if lower == "bash":
        return _as_bash(_arg(args, "command", "cmd"), result)
    if lower in {"notebookedit", "notebook_edit"}:
        return _as_edit(args)

    if not name and raw.get("text"):
        return NormalizedAction("think", None, flag_deltas={"think_count": 1})
    return NormalizedAction("execute", _first_path_in_text(_arg(args, "command") or result), result_tag=_ok_tag(result))


def _from_explicit(category: str, args: dict[str, Any], result: str) -> NormalizedAction:
    path = normalize_path(_arg(args, "file_path", "path", "target_file"))
    if category == "view":
        return _as_view(args, result, path=path or "")
    if category == "search":
        return NormalizedAction("search", path or ".", file_ops=("S",))
    if category == "modify":
        return _as_edit(args)
    if category == "create":
        return NormalizedAction("create", path, file_ops=("C",) if path else ())
    if category == "test":
        ok = _command_ok(result)
        return NormalizedAction(
            "test",
            path,
            result_tag="ok" if ok else "error",
            flag_deltas={"test_pass_count": 1} if ok else {"test_fail_count": 1},
        )
    if category == "think":
        return NormalizedAction("think", None, flag_deltas={"think_count": 1})
    if category == "finish":
        return NormalizedAction("finish", None)
    return NormalizedAction(category, path, result_tag=_ok_tag(result) if category in {"execute", "install"} else None)


def _as_view(args: dict[str, Any], result: str, *, path: str) -> NormalizedAction:
    npath = normalize_path(path)
    if _looks_full_view(args, result):
        ops = ("Vf",) if npath else ()
        return NormalizedAction("view", npath, file_ops=ops, scope="full")
    rng = _infer_line_range(args, result) or (1, 1)
    op = view_buckets(*rng)
    inner = op[1:]  # "[2]" or "[1-2]"
    return NormalizedAction("view", npath, file_ops=(op,) if npath else (), scope=f"partial{inner}")


def _as_edit(args: dict[str, Any]) -> NormalizedAction:
    path = normalize_path(_arg(args, "file_path", "path"))
    old = _arg(args, "old_string", "old_str", "old", "old_text", strip_newlines=False)
    new = _arg(args, "new_string", "new_str", "new", "new_text", strip_newlines=False)
    digest = content_hash(old, new)
    if not old.strip():
        return NormalizedAction("modify", path, file_ops=(f"I:{digest}",) if path else (), scope=f"insert:{digest}")
    return NormalizedAction("modify", path, file_ops=(f"M:{digest}",) if path else (), scope=f"replace:{digest}")


def _ok_tag(result: str) -> str:
    return "ok" if _command_ok(result) else "error"


def _as_bash(command: str, result: str) -> NormalizedAction:
    kind = _bash_kind(command)
    path = _first_path_in_text(command)
    ok = _command_ok(result)
    if kind == "view":
        return _as_view({"file_path": path or ""}, result, path=path or "")
    if kind == "search":
        return NormalizedAction("search", path or ".", file_ops=("S",))
    if kind == "test":
        return NormalizedAction(
            "test",
            path,
            result_tag="ok" if ok else "error",
            flag_deltas={"test_pass_count": 1} if ok else {"test_fail_count": 1},
        )
    if kind == "install":
        return NormalizedAction("install", path, result_tag=_ok_tag(result))
    if kind == "fileop":
        return NormalizedAction("fileop", path, scope=command.strip().split()[0] if command.strip() else "")
    return NormalizedAction("execute", path, result_tag=_ok_tag(result))


class StateSignatureBuilder:
    """Accumulate file-op sets + FLAGS; snapshot is order-invariant."""

    def __init__(self) -> None:
        self.files: dict[str, set[str]] = {}
        self.flags: dict[str, int] = {k: 0 for k in FLAG_KEYS}

    def snapshot(self) -> str:
        parts: list[str] = []
        for path in sorted(self.files):
            ops = self.files[path]
            if not ops:
                continue
            joined = ",".join(sorted(ops))
            parts.append(f"{path}:{joined}")
        flags = ",".join(f"{k}={self.flags[k]}" for k in FLAG_KEYS)
        parts.append(f"({flags})")
        return "|".join(parts)

    def apply(self, action: Mapping[str, Any] | Any | NormalizedAction) -> NormalizedAction:
        norm = action if isinstance(action, NormalizedAction) else normalize_action(action)
        if norm.path and norm.file_ops:
            bucket = self.files.setdefault(norm.path, set())
            bucket.update(norm.file_ops)
        for key, delta in norm.flag_deltas.items():
            self.flags[key] = self.flags.get(key, 0) + int(delta)
        return norm

    def state_signature(self, history: Iterable[Mapping[str, Any] | Any], t: int) -> str:
        """Signature of actions ``history[0:t]`` (excludes action t)."""
        clone = StateSignatureBuilder()
        for action in list(history)[:t]:
            clone.apply(action)
        return clone.snapshot()


def action_signature(action: Mapping[str, Any] | Any) -> str:
    return normalize_action(action).action_signature()


def step_signatures(history: Iterable[Mapping[str, Any] | Any]) -> list[dict[str, str]]:
    """For each step t: StateSignature(t) from prior actions + ActionSignature(t)."""
    builder = StateSignatureBuilder()
    out: list[dict[str, str]] = []
    for t, action in enumerate(history):
        state = builder.snapshot()
        norm = builder.apply(action)
        out.append({"t": t, "state": state, "action": norm.action_signature()})
    return out


def flatten_turns(turns: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand per-turn parsed rollouts into a time-ordered tool-call list."""
    actions: list[dict[str, Any]] = []
    rows = list(turns)
    for i, turn in enumerate(rows):
        calls = list(turn.get("tool_calls") or [])
        results = list(turn.get("tool_results") or [])
        if calls:
            for j, call in enumerate(calls):
                item = dict(call)
                if j < len(results) and "result" not in item:
                    item["result"] = results[j]
                actions.append(item)
            continue
        if (turn.get("think") or "").strip() or (turn.get("text") or "").strip():
            last = i == len(rows) - 1
            actions.append(
                {
                    "name": "finish" if last else "think",
                    "text": turn.get("text") or "",
                    "think": turn.get("think") or "",
                }
            )
    return actions


def group_by_state(trajectories: Mapping[str, Iterable[Mapping[str, Any]]]) -> dict[str, list[tuple[str, int]]]:
    """Map StateSignature → [(traj_id, step_t), ...] across rollouts."""
    groups: dict[str, list[tuple[str, int]]] = {}
    for traj_id, history in trajectories.items():
        for row in step_signatures(history):
            groups.setdefault(row["state"], []).append((str(traj_id), int(row["t"])))
    return groups
