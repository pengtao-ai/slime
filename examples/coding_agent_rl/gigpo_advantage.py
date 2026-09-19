"""GiGPO advantages for coding-agent RL: ``A = A_E + w * A_S`` (no ``A_I``).

Wire with::

    --custom-advantage-function-path \\
      examples.coding_agent_rl.gigpo_advantage.compute_gigpo_advantages

``rollout_data["rewards"]`` is already GRPO group-demeaned (= ``A_E``).
``raw_reward`` is the episode return ``R`` used for discounted ``G_t``.

Per turn anchor ``T#`` = ``intent · tool`` (see :func:`gigpo_turn_key_from_message`),
stored on ``turn_costs[].gigpo_T`` at rollout and copied into ``train_metadata.turn_T``.

  G_t = R * γ^{n-1-t}
  A_S = G_t - mean(G | same T# across sibling trajs)
  A_t = A_E + w * A_S

Env:
  ``GIGPO_W`` (default 1.0), ``GIGPO_GAMMA`` (default 0.95).
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
from argparse import Namespace
from collections import defaultdict
from typing import Any

import torch

logger = logging.getLogger(__name__)

DEFAULT_GIGPO_W = 1.0
DEFAULT_GIGPO_GAMMA = 0.95


def gigpo_w() -> float:
    return float(os.environ.get("GIGPO_W", str(DEFAULT_GIGPO_W)))


def gigpo_gamma() -> float:
    return float(os.environ.get("GIGPO_GAMMA", str(DEFAULT_GIGPO_GAMMA)))


# ---------------------------------------------------------------------------
# Cross-agent tool-name → canonical family (CC / opencode / pi / miniswe)
# ---------------------------------------------------------------------------
# Claude Code: Bash, Read, Write, Edit, Explore, Find, ...
# opencode:    bash, read, write, edit, glob, grep, ...
# pi:          bash, read, edit, write
# miniswe:     bash only (classify via command args)

_EDIT_NAMES = frozenset(
    {
        "edit",
        "write",
        "notebookedit",
        "delete",
        "str_replace",
        "search_replace",
        "apply_patch",
        "edit_file",
        "write_file",
    }
)
_READ_NAMES = frozenset({"read"})
_SEARCH_NAMES = frozenset({"grep", "glob", "explore", "find"})  # dedicated tools, not bash find
_SHELL_NAMES = frozenset({"bash", "shell", "run_terminal_cmd", "execute", "run"})
_FS_NAMES = frozenset({"ls"})  # pi dedicated ls tool
_META_PREFIXES = ("task", "cron", "mcp__")
_META_NAMES = frozenset(
    {
        "exit",
        "exitworktree",
        "enterworktree",
        "exitplanmode",
        "enterplanmode",
        "askuserquestion",
        "slashcommand",
        "agent",
        "skill",
        "todowrite",
        "todoread",
        "webfetch",
        "websearch",
        "question",
        "lsp",
        "print",
        "print_summary",
        "sendmessage",
        "reportfindings",
        "schedulewakeup",
        "workflow",
        "task",
        "notebookread",
        "toolsearch",
    }
)


def _args_to_str(args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    if isinstance(args, dict):
        # Prefer common shell/command fields, else full dict.
        for k in ("command", "cmd", "script", "code"):
            if args.get(k) is not None:
                return str(args[k])
        return json.dumps(args, ensure_ascii=False)
    return str(args)


def iter_tool_calls(msg: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Return ``[(tool_name, args_str), ...]`` from OpenAI or Anthropic assistant msgs."""
    if not isinstance(msg, dict):
        return []
    out: list[tuple[str, str]] = []
    calls = msg.get("tool_calls") or []
    content = msg.get("content")
    if not calls and isinstance(content, list):
        for b in content:
            if not isinstance(b, dict) or b.get("type") != "tool_use":
                continue
            out.append((str(b.get("name") or "?"), _args_to_str(b.get("input"))))
        return out
    for c in calls:
        if not isinstance(c, dict):
            continue
        fn = c.get("function") if isinstance(c.get("function"), dict) else {}
        name = str((fn or {}).get("name") or c.get("name") or "?")
        args = (fn or {}).get("arguments")
        if args is None:
            args = c.get("input")
        # OpenAI may JSON-encode arguments.
        if isinstance(args, str):
            s = args.strip()
            if s.startswith("{") or s.startswith("["):
                try:
                    args = json.loads(s)
                except json.JSONDecodeError:
                    pass
        out.append((name, _args_to_str(args)))
    return out


def _strip_shell_wrappers(cmd: str) -> str:
    """Drop leading ``cd`` / ``sudo`` / ``env assignments`` so the real verb is visible."""
    s = (cmd or "").strip()
    if not s:
        return s
    # Join continued lines lightly.
    s = re.sub(r"\\\n", " ", s)
    changed = True
    while changed and s:
        changed = False
        m = re.match(r"^(sudo|time|nice|nohup)\s+", s, flags=re.I)
        if m:
            s = s[m.end() :].lstrip()
            changed = True
            continue
        m = re.match(r"^(?:[A-Za-z_][\w]*=\S+\s+)+", s)
        if m:
            s = s[m.end() :].lstrip()
            changed = True
            continue
        # ``cd PATH &&`` / ``cd PATH;``
        m = re.match(r"^cd\s+(?:'[^\']*'|\"[^\"]*\"|\S+)\s*(?:&&|;)\s*", s, flags=re.I)
        if m:
            s = s[m.end() :].lstrip()
            changed = True
            continue
    return s


def _first_shell_token(cmd: str) -> str:
    s = (cmd or "").strip()
    if not s:
        return ""
    # ``( subshell`` / ``{``
    s = re.sub(r"^[\(\{]\s*", "", s)
    tok = re.split(r"[\s|;<&>]", s, maxsplit=1)[0]
    tok = tok.strip("'\"")
    if "/" in tok:
        tok = tok.rsplit("/", 1)[-1]
    return tok.lower()


def _shell_family_from_args(args: str) -> str:
    """Refine a shell/bash invocation from its command string."""
    raw = (args or "").strip()
    low = raw.lower()
    if not low or low in ("{}", "{ }"):
        return "Bash:empty"

    # Drop leading full-line comments then retry once.
    if raw.lstrip().startswith("#"):
        rest = re.sub(r"^\s*#[^\n]*\n?", "", raw, count=1).strip()
        if rest and rest != raw:
            return _shell_family_from_args(rest)

    # High-signal phrases (anywhere) before token tables.
    if "complete_task_and_submit_final_output" in low:
        return "Bash:submit"
    if re.search(r"\b(pytest|unittest|tox|runtests)\b", low):
        return "Bash:pytest"
    if re.search(r"\b(grep|rg)\b", low) or "xargs grep" in low:
        return "Bash:grep"
    if re.search(r"str_replace|search_replace|apply_patch|edit_file|write_file", low):
        return "Edit"
    if re.search(r"cat\s*>|tee\s+|>>\s*|<<", low):
        return "Edit"
    if re.search(r"(^|[\s;|&])sed\s+-i\b", low):
        return "Edit"

    core = _strip_shell_wrappers(raw)
    core_low = core.lower()
    tok = _first_shell_token(core)

    if tok in ("for", "while", "until", "do"):
        return "Bash:loop"

    # Pure navigation / control.
    if re.fullmatch(r"cd(\s+\S+)?", core_low):
        return "Bash:cd"
    if tok in ("exit", "logout"):
        return "Bash:exit"
    if tok in ("true", "false", ":") or core_low in ("true", ":", "false"):
        return "Bash:noop"
    if tok in ("echo", "printf"):
        return "Bash:echo"

    # Build / compile / package managers that install.
    if tok in (
        "make",
        "cmake",
        "ninja",
        "cargo",
        "gcc",
        "g++",
        "c++",
        "cc",
        "clang",
        "clang++",
        "go",
        "mvn",
        "gradle",
        "bazel",
        "meson",
        "dotnet",
        "rustc",
    ):
        return "Bash:build"
    if tok.endswith((".sh", ".bash")):
        return "Bash:script"
    if tok in ("bash", "sh", "zsh"):
        rest = re.sub(rf"^{re.escape(tok)}\s+", "", core, count=1, flags=re.I).strip()
        if rest:
            # ``bash foo.sh`` / ``bash -lc '...'`` — classify the remainder once.
            nested = _shell_family_from_args(rest)
            if nested != "Bash:other":
                return nested
            if re.search(r"\.(sh|bash)\b", rest.lower()):
                return "Bash:script"
        return "Bash:script"

    if tok in (
        "pip",
        "pip3",
        "pipx",
        "apt",
        "apt-get",
        "yum",
        "dnf",
        "conda",
        "npm",
        "yarn",
        "pnpm",
        "uv",
        "poetry",
        "bundle",
        "gem",
    ):
        return "Bash:env"

    if re.search(r"(^|[\s;|&])find\s", core_low) or re.search(r"\bfind\s+/", core_low):
        return "Bash:find"
    if tok == "git" or re.search(r"(^|[\s;|&])git\s", core_low):
        return "Bash:git"

    if tok in ("curl", "wget", "http", "httpie"):
        return "Bash:http"
    if tok in ("sqlite3", "psql", "mysql", "mongosh", "mongod", "redis-cli"):
        return "Bash:db"
    if tok in ("tar", "unzip", "zip", "gzip", "gunzip", "zcat", "xz", "unxz", "bzip2", "bunzip2"):
        return "Bash:archive"
    if tok in ("xxd", "od", "hexdump", "hd", "nm", "objdump", "readelf"):
        return "Bash:hex"
    if tok in ("ffmpeg", "ffprobe", "tesseract", "whisper", "pocketsphinx", "sox", "magick", "convert"):
        return "Bash:media"
    if tok in ("diff", "cmp"):
        return "Bash:diff"
    if tok in ("awk", "wc", "cut", "sort", "uniq", "tr", "column", "paste", "comm", "jq", "yq", "seq", "iconv", "base64", "md5sum", "sha256sum"):
        return "Bash:text"
    if tok in ("which", "whoami", "id", "file", "stat", "hostname", "uname", "type", "command", "whereis", "env", "printenv", "locale"):
        return "Bash:probe"
    if tok in ("pkill", "kill", "killall", "pgrep", "fuser", "sleep", "timeout", "wait", "jobs", "bg", "fg", "nohup"):
        return "Bash:proc"
    if tok in ("crontab", "systemctl", "service", "journalctl", "passwd", "useradd", "usermod", "groupadd"):
        return "Bash:sys"
    if tok in ("docker", "podman", "kubectl", "nerdctl"):
        return "Bash:container"
    if tok in ("black", "ruff", "prettier", "eslint", "flake8", "isort", "autopep8", "clang-format", "gofmt"):
        return "Bash:format"

    if re.search(r"(^|[\s;|&])(sed\s+-n|\bcat\b|\bnl\b|\bhead\b|\btail\b|\bless\b|\bmore\b)\b", core_low):
        return "Bash:read"
    if re.search(r"\bpython(?:3)?\b", core_low) or tok in ("python", "python3", "ipython", "py"):
        return "Bash:python"
    if tok in ("node", "deno", "bun", "ruby", "perl", "php", "lua", "Rscript"):
        return "Bash:runtime"
    if re.search(r"\b(ls|pwd|mkdir|chmod|chown|touch|rm|rmdir|cp|mv|ln|du|df|tree)\b", core_low):
        return "Bash:fs"

    # Custom binary / path-only runner under home/workspace.
    if tok and (core.startswith("./") or core.startswith("/") or tok.endswith(".py") or tok.endswith(".bin")):
        return "Bash:run"

    return "Bash:other"


def classify_tool_call(name: str, args: str = "") -> str:
    """Map one tool call to a canonical family shared across agents."""
    n = (name or "").strip().lower()
    args_s = args or ""
    if not n and not args_s.strip():
        return "无 tool"
    if n in _EDIT_NAMES:
        return "Edit"
    if n in _READ_NAMES:
        return "Read"
    if n in _SEARCH_NAMES:
        return "Search"
    if n in _FS_NAMES:
        return "Bash:fs"
    if n in _SHELL_NAMES:
        return _shell_family_from_args(args_s)
    if n in _META_NAMES or n.startswith(_META_PREFIXES):
        return "Meta"
    # Nameless / "?": treat args as a shell blob (legacy).
    if not n or n == "?":
        return _shell_family_from_args(args_s)
    # Unknown named tool: only run shell heuristics when args look like a real command.
    # Empty / "{}" must NOT become Bash:empty (that bucket is for empty Bash calls).
    args_stripped = args_s.strip()
    if args_stripped and args_stripped not in ("{}", "{ }", "null", "None"):
        fam = _shell_family_from_args(args_s)
        if fam != "Bash:other":
            return fam
    return "Meta"


def _intent_from_family(family: str, args: str = "") -> str:
    low = (args or "").lower()
    if family == "Bash:pytest" or re.search(r"\b(pytest|unittest)\b", low):
        return "运行测试/验证"
    if family == "Edit":
        if re.search(r"test_|tests/|/tmp/.*test", low):
            return "编写/调整测试"
        return "实现/修改"
    if family in ("Bash:env", "Bash:build", "Bash:cd", "Bash:proc", "Bash:sys", "Bash:container", "Bash:archive", "Bash:format"):
        return "环境准备"
    if family in (
        "Search",
        "Bash:grep",
        "Bash:find",
        "Bash:git",
        "Bash:fs",
        "Bash:hex",
        "Bash:probe",
        "Bash:diff",
        "Bash:text",
        "Bash:db",
        "Bash:media",
        "Bash:http",
        "Bash:run",
        "Bash:script",
        "Bash:runtime",
        "Bash:loop",
    ):
        return "探索/定位"
    if family in ("Read", "Bash:read"):
        return "阅读代码"
    if family == "Bash:python":
        if re.search(r"pytest|unittest", low):
            return "运行测试/验证"
        return "探索/定位"
    if family in ("Bash:submit", "Bash:exit", "Bash:echo", "Bash:noop", "Bash:empty", "Bash:other", "Meta", "无 tool"):
        return "其他"
    return "其他"


def classify_message_tools(msg: dict[str, Any] | None) -> tuple[str, str, str]:
    """Return ``(T#, intent, family)`` for the primary (first) tool call."""
    calls = iter_tool_calls(msg)
    if not calls:
        return "其他 · 无 tool", "其他", "无 tool"
    name, args = calls[0]
    family = classify_tool_call(name, args)
    intent = _intent_from_family(family, args)
    return f"{intent} · {family}", intent, family


def _tool_brief_from_message(msg: dict[str, Any] | None) -> str:
    """Flatten assistant tool calls for debugging / legacy blob APIs."""
    parts = [f"{n}({a[:100]})" for n, a in iter_tool_calls(msg)[:3]]
    if parts:
        return "; ".join(parts)
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content[:120]
    if isinstance(content, list):
        texts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                texts.append(str(b.get("text") or ""))
            elif isinstance(b, str):
                texts.append(b)
        return "".join(texts)[:120]
    return str(content or "")[:120]


def _tool_family(blob: str) -> str:
    """Legacy blob classifier (bash-arg heuristics). Prefer :func:`classify_tool_call`."""
    low = (blob or "").lower().strip()
    if not low:
        return "无 tool"
    # If blob looks like ``Name(...)``, honor the name first.
    m = re.match(r"([A-Za-z_][\w\-]*)\(", blob.strip())
    if m:
        name = m.group(1)
        args = blob.strip()[len(name) + 1 :]
        if args.endswith(")"):
            args = args[:-1]
        return classify_tool_call(name, args)
    return _shell_family_from_args(blob)


def _intent_from_blob(blob: str, tool: str) -> str:
    return _intent_from_family(tool, blob)


def gigpo_turn_key_from_blob(blob: str) -> str:
    tool = _tool_family(blob)
    intent = _intent_from_blob(blob, tool)
    return f"{intent} · {tool}"


def gigpo_turn_key_from_message(msg: dict[str, Any] | None) -> str:
    return classify_message_tools(msg)[0]


def stamp_turn_gigpo_key(turn_entry: dict[str, Any], reply: Any) -> None:
    """Write ``gigpo_T`` onto a turn_costs ledger entry from the final Reply."""
    msg = getattr(reply, "manager_message", None)
    if not isinstance(msg, dict):
        msg = turn_entry.get("sft_assistant_message")
    turn_entry["gigpo_T"] = gigpo_turn_key_from_message(msg if isinstance(msg, dict) else None)


def _paint_ae_as(
    base: torch.Tensor,
    *,
    a_e: float,
    a_s_list: list[float],
    turn_token_spans: list[list[int]] | None,
    w: float,
) -> torch.Tensor:
    adv = torch.ones_like(base, dtype=torch.float32) * float(a_e)
    if not a_s_list or not turn_token_spans or len(turn_token_spans) != len(a_s_list):
        return adv
    n = int(adv.numel())
    for span, a_s in zip(turn_token_spans, a_s_list, strict=False):
        if not span or len(span) < 2:
            continue
        start, end = int(span[0]), int(span[1])
        if end <= start:
            continue
        start = max(0, min(start, n))
        end = max(start, min(end, n))
        adv[start:end] = float(a_e) + float(w) * float(a_s)
    return adv


def _traj_key(md: dict[str, Any], sample_index: Any, group_index: Any) -> tuple[Any, Any]:
    gid = md.get("group_index", group_index)
    idx = md.get("sample_index", sample_index)
    return gid, idx


def compute_step_as_for_group(
    trajs: list[dict[str, Any]],
    *,
    gamma: float,
) -> list[list[float]]:
    """Compute per-traj ``A_S`` lists for one sibling group.

    Each traj dict needs ``R`` (float) and ``turn_T`` (list[str]).
    """
    by_T: dict[str, list[float]] = defaultdict(list)
    packed: list[list[tuple[str, float]]] = []
    for traj in trajs:
        R = float(traj.get("R") or 0.0)
        keys = list(traj.get("turn_T") or [])
        n = len(keys)
        pairs: list[tuple[str, float]] = []
        for t, T in enumerate(keys):
            G = R * (gamma ** max(0, n - 1 - t))
            by_T[str(T)].append(G)
            pairs.append((str(T), G))
        packed.append(pairs)
    T_bar = {k: statistics.mean(v) for k, v in by_T.items()}
    out: list[list[float]] = []
    for pairs in packed:
        out.append([G - T_bar[T] for T, G in pairs])
    return out


def compute_gigpo_advantages(args: Namespace, rollout_data: dict[str, Any]) -> None:
    """Populate ``advantages`` / ``returns`` with ``A_E + w * A_S``."""
    del args
    kl: list[torch.Tensor] = rollout_data["kl"]
    a_e_list: list[float] = list(rollout_data["rewards"])
    raw_rewards = list(rollout_data.get("raw_reward") or a_e_list)
    metadata_list = rollout_data.get("metadata") or [None] * len(kl)
    sample_indices = list(rollout_data.get("sample_indices") or list(range(len(kl))))
    w = gigpo_w()
    gamma = gigpo_gamma()

    # One representative sample per (group_index, sample_index) for T# pooling.
    reps: dict[tuple[Any, Any], int] = {}
    for i in range(len(kl)):
        md = metadata_list[i] if i < len(metadata_list) else None
        md = md if isinstance(md, dict) else {}
        key = _traj_key(
            md,
            sample_indices[i] if i < len(sample_indices) else i,
            md.get("group_index"),
        )
        if key not in reps:
            reps[key] = i

    by_gid: dict[Any, list[int]] = defaultdict(list)
    for (gid, _idx), i in reps.items():
        by_gid[gid].append(i)

    as_by_rep: dict[int, list[float]] = {}
    for _gid, indices in by_gid.items():
        trajs = []
        for i in indices:
            md = metadata_list[i] if isinstance(metadata_list[i], dict) else {}
            turn_T = list(md.get("turn_T") or [])
            if not turn_T:
                # Legacy dumps: derive from turn_costs if present.
                turn_T = [
                    str(tc.get("gigpo_T") or "其他 · 无 tool")
                    for tc in (md.get("turn_costs") or [])
                ]
            trajs.append({"R": float(raw_rewards[i]), "turn_T": turn_T})
        as_lists = compute_step_as_for_group(trajs, gamma=gamma)
        for i, a_s_list in zip(indices, as_lists, strict=True):
            as_by_rep[i] = a_s_list

    advantages: list[torch.Tensor] = []
    for i, (k, a_e) in enumerate(zip(kl, a_e_list, strict=False)):
        md = metadata_list[i] if i < len(metadata_list) else None
        md = md if isinstance(md, dict) else {}
        key = _traj_key(
            md,
            sample_indices[i] if i < len(sample_indices) else i,
            md.get("group_index"),
        )
        rep_i = reps.get(key, i)
        a_s_list = list(as_by_rep.get(rep_i) or [])
        spans = md.get("turn_token_spans")
        if spans is not None and not isinstance(spans, list):
            spans = None
        if a_s_list and spans is None:
            logger.debug(
                "compute_gigpo_advantages: sample %d missing turn_token_spans; broadcast A_E",
                i,
            )
        advantages.append(
            _paint_ae_as(
                k,
                a_e=float(a_e),
                a_s_list=a_s_list,
                turn_token_spans=spans,
                w=w,
            )
        )

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = advantages
