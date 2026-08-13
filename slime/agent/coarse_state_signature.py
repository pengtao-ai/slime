"""Coarse StateSignature for grouping coding-agent rollouts.

Records only (op_type, relative_path) sets plus test pass/fail counts.
No file contents, diffs, hashes, line ranges, or bash command strings.
StateSignature(t) uses history before action t only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from slime.agent.rtmc_signature import _arg, _args, _as_dict, _command_ok, _result_text, flatten_turns

WORKDIR_PREFIXES = ("/home/user/", "/home/user")
_FILE_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")
_PATH_RE = re.compile(r"(?:^|[\s=`\"'(])(/?(?:[\w.-]+/)*[\w.-]+/?)")


def relative_path(path: str | None, *, as_dir: bool = False) -> str | None:
    if path is None:
        return None
    text = str(path).strip().strip("'\"")
    if not text:
        return None
    text = os.path.normpath(text).replace("\\", "/")
    under_workdir = False
    for prefix in WORKDIR_PREFIXES:
        if text == prefix.rstrip("/"):
            text = "."
            under_workdir = True
            break
        if text.startswith(prefix):
            text = text[len(prefix) :]
            under_workdir = True
            break
    while text.startswith("./"):
        text = text[2:]
    if under_workdir:
        text = text.lstrip("/")
    if not text or text == ".":
        text = "."
    if as_dir:
        if text != "." and not text.endswith("/"):
            text += "/"
        elif text == ".":
            text = "./"
    return text


def _looks_dir(path: str) -> bool:
    if path.endswith("/") or path in {".", ".."}:
        return True
    base = path.rstrip("/").rsplit("/", 1)[-1]
    if base.startswith("."):
        return "." not in base[1:]
    return not _FILE_EXT_RE.search(base)


def _paths_in_text(text: str) -> list[str]:
    out: list[str] = []
    for match in _PATH_RE.finditer(text or ""):
        token = match.group(1)
        if token in {".", ".."} or token.isdigit():
            continue
        if "/" not in token and not _FILE_EXT_RE.search(token):
            continue
        out.append(token)
    return out


def _bash_kind(command: str) -> str:
    head = re.sub(r"^(?:cd\s+\S+\s*(?:&&|;)\s*)+", "", command.strip()).strip()
    low = head.lower()
    if re.search(r"\b(pytest|unittest|cargo test|npm test|nox|tox)\b", low):
        return "test"
    if re.match(r"(ls|find|grep|rg|ag|fd|glob)\b", low):
        return "search"
    if re.match(r"(cat|head|tail|less|more|nl)\b", low):
        return "view"
    if re.match(r"(mkdir|rm|rmdir|cp|mv|chmod|chown|touch|ln)\b", low):
        return "ignore"
    return "execute"


@dataclass
class CoarseAction:
    kind: str
    tokens: tuple[str, ...] = ()
    test_pass_delta: int = 0
    test_fail_delta: int = 0
    action_signature: str = ""


def normalize_coarse_action(action: Mapping[str, Any] | Any) -> CoarseAction:
    raw = _as_dict(action)
    name = str(raw.get("name") or raw.get("tool") or "").strip().lower()
    args = _args(raw)
    result = _result_text(raw)

    if name in {"read", "view", "cat", "head", "tail"}:
        path = relative_path(_arg(args, "file_path", "path", "target_file"))
        if not path:
            return CoarseAction("ignore", action_signature="")
        token = f"V:{path}"
        return CoarseAction("V", (token,), action_signature=token)

    if name in {"grep", "glob", "search", "find"}:
        raw_path = _arg(args, "path", "file_path", "target_file")
        as_dir = name in {"glob", "find"} or _looks_dir(raw_path or ".")
        path = relative_path(raw_path or ".", as_dir=as_dir)
        token = f"S:{path}"
        return CoarseAction("S", (token,), action_signature=token)

    if name in {"edit", "strreplace", "str_replace", "replace", "notebookedit", "notebook_edit"}:
        path = relative_path(_arg(args, "file_path", "path"))
        if not path:
            return CoarseAction("ignore", action_signature="")
        token = f"M:{path}"
        return CoarseAction("M", (token,), action_signature=token)

    if name in {"write", "create", "create_file"}:
        path = relative_path(_arg(args, "file_path", "path"))
        if not path:
            return CoarseAction("ignore", action_signature="")
        token = f"C:{path}"
        return CoarseAction("C", (token,), action_signature=token)

    if name == "bash":
        return _coarse_bash(_arg(args, "command", "cmd"), result)

    if name in {"think", "finish", "submit", "exit", "exitworktree", "reasoning"}:
        return CoarseAction("ignore", action_signature=name or "ignore")
    return CoarseAction("EXECUTE", action_signature="EXECUTE")


def _coarse_bash(command: str, result: str) -> CoarseAction:
    kind = _bash_kind(command)
    if kind == "test":
        ok = _command_ok(result)
        if ok:
            return CoarseAction("T_PASS", test_pass_delta=1, action_signature="T_PASS")
        return CoarseAction("T_FAIL", test_fail_delta=1, action_signature="T_FAIL")
    if kind == "search":
        tokens = []
        for raw_path in _paths_in_text(command) or ["."]:
            tokens.append(f"S:{relative_path(raw_path, as_dir=_looks_dir(raw_path))}")
        uniq = tuple(sorted(set(tokens)))
        return CoarseAction("S", uniq, action_signature=uniq[0] if uniq else "S")
    if kind == "view":
        paths = _paths_in_text(command)
        if not paths:
            return CoarseAction("ignore", action_signature="")
        tokens = tuple(f"V:{relative_path(p)}" for p in paths if relative_path(p))
        return CoarseAction("V", tokens, action_signature=tokens[0] if tokens else "V")
    if kind == "ignore":
        return CoarseAction("ignore", action_signature="")
    return CoarseAction("EXECUTE", action_signature="EXECUTE")


class CoarseStateSignatureBuilder:
    """Set of `OP:relpath` plus TEST pass/fail counts."""

    def __init__(self) -> None:
        self.ops: set[str] = set()
        self.test_pass_count = 0
        self.test_fail_count = 0

    def snapshot(self) -> str:
        parts = sorted(self.ops)
        parts.append(f"TEST:{{pass={self.test_pass_count},fail={self.test_fail_count}}}")
        return " | ".join(parts)

    def apply(self, action: Mapping[str, Any] | Any | CoarseAction) -> CoarseAction:
        norm = action if isinstance(action, CoarseAction) else normalize_coarse_action(action)
        if norm.kind == "EXECUTE" or norm.kind == "ignore":
            return norm
        self.ops.update(norm.tokens)
        self.test_pass_count += norm.test_pass_delta
        self.test_fail_count += norm.test_fail_delta
        return norm

    def state_signature(self, history: Iterable[Mapping[str, Any] | Any], t: int) -> str:
        clone = CoarseStateSignatureBuilder()
        for action in list(history)[:t]:
            clone.apply(action)
        return clone.snapshot()


def coarse_action_signature(action: Mapping[str, Any] | Any) -> str:
    return normalize_coarse_action(action).action_signature


def coarse_step_signatures(history: Iterable[Mapping[str, Any] | Any]) -> list[dict[str, str]]:
    builder = CoarseStateSignatureBuilder()
    out: list[dict[str, str]] = []
    for t, action in enumerate(history):
        state = builder.snapshot()
        norm = builder.apply(action)
        out.append({"t": t, "state": state, "action": norm.action_signature})
    return out


def coarse_group_by_state(
    trajectories: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, list[tuple[str, int]]]:
    groups: dict[str, list[tuple[str, int]]] = {}
    for traj_id, history in trajectories.items():
        for row in coarse_step_signatures(history):
            groups.setdefault(row["state"], []).append((str(traj_id), int(row["t"])))
    return groups


def turn_start_states(turns: Iterable[Mapping[str, Any]]) -> list[str]:
    """StateSignature at the start of each agent turn (history of prior turns only)."""
    builder = CoarseStateSignatureBuilder()
    states: list[str] = []
    for turn in turns:
        states.append(builder.snapshot())
        for action in flatten_turns([turn]):
            builder.apply(action)
    return states
