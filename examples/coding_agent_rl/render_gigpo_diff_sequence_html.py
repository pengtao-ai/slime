#!/usr/bin/env python3
"""Render distinct cumulative git diffs for eight sibling trajectories."""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

DIFF_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)
_IM_START = re.compile(r"<\|im_start\|>(\w+)\n")
_TOOL_CALL = re.compile(r"<tool_call>\n?(.*?)</tool_call>", re.DOTALL)
_FUNCTION = re.compile(r"<function=([^\s>]+)>\s*(.*?)</function>", re.DOTALL)
_PARAM = re.compile(r"<parameter=([^\s>]+)>\n?(.*?)</parameter>", re.DOTALL)
CSS = """
* { box-sizing: border-box; }
body { margin: 0; background: #0d1117; color: #c9d1d9; font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }
header { position: sticky; top: 0; z-index: 2; padding: 12px 16px; background: #161b22; border-bottom: 1px solid #30363d; font-family: sans-serif; }
.grid { display: grid; grid-template-columns: repeat(8, minmax(380px, 1fr)); gap: 8px; min-width: 3040px; padding: 10px; align-items: start; }
.traj { background: #161b22; border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }
.traj-title { padding: 8px; background: #21262d; font: 600 12px sans-serif; color: #f0f6fc; }
.snapshot { border-top: 1px solid #30363d; }
.snapshot-title { padding: 6px 8px; background: #1c2128; color: #79c0ff; font: 600 11px sans-serif; }
.intent { padding: 7px 8px; background: #18232f; color: #f0c674; font: 12px sans-serif; border-top: 1px solid #30363d; border-bottom: 1px solid #30363d; }
pre { margin: 0; padding: 8px; overflow: auto; max-height: 560px; white-space: pre; }
.empty { padding: 14px 8px; color: #8b949e; font-style: italic; }
"""


def _parse_tool_calls(assistant_text: str) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    for block in _TOOL_CALL.findall(assistant_text or ""):
        match = _FUNCTION.search(block)
        if match:
            params = {m.group(1).strip(): m.group(2).strip() for m in _PARAM.finditer(match.group(2))}
            calls.append((match.group(1).strip(), params))
    return calls


def _split_messages(text: str) -> list[tuple[str, str]]:
    parts = _IM_START.split(text)
    messages: list[tuple[str, str]] = []
    for index in range(1, len(parts) - 1, 2):
        body = parts[index + 1]
        body = body.removesuffix("<|im_end|>\n").removesuffix("<|im_end|>")
        messages.append((parts[index], body))
    return messages


def _parse_tool_calls(assistant_text: str) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    for block in _TOOL_CALL.findall(assistant_text or ""):
        match = _FUNCTION.search(block)
        if match:
            params = {m.group(1).strip(): m.group(2).strip() for m in _PARAM.finditer(match.group(2))}
            calls.append((match.group(1).strip(), params))
    return calls


def _split_messages(text: str) -> list[tuple[str, str]]:
    parts = _IM_START.split(text)
    messages: list[tuple[str, str]] = []
    for index in range(1, len(parts) - 1, 2):
        body = parts[index + 1]
        body = body.removesuffix("<|im_end|>\n").removesuffix("<|im_end|>")
        messages.append((parts[index], body))
    return messages


def files(diff: str) -> str:
    paths = []
    for a, b in DIFF_RE.findall(diff):
        paths.append(b if b != "/dev/null" else a)
    return ", ".join(sorted(set(paths))) or "<empty diff>"


def distinct_diffs(sample: dict) -> list[tuple[int, int, str]]:
    entries = [x for x in (sample.get("metadata") or {}).get("turn_git_diffs", []) if isinstance(x, dict)]
    entries.sort(key=lambda x: int(x.get("turn_index", 0)))
    result: list[list[object]] = []
    for entry in entries:
        turn = int(entry.get("turn_index", 0))
        diff = str(entry.get("git_diff") or "")
        if result and diff == result[-1][2]:
            result[-1][1] = turn
        else:
            result.append([turn, turn, diff])
    return [(int(a), int(b), str(diff)) for a, b, diff in result]


def infer_intent(diff: str, problem_statement: str) -> str:
    if not diff.strip():
        return "探索/未修改"
    paths = files(diff)
    added = "\n".join(line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = "\n".join(line[1:] for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    text = f"{problem_statement}\n{added}\n{removed}".lower()
    if "test_repro" in paths or "test_direction" in paths or "test.md" in paths:
        if "pr296" in problem_statement.lower() or "consecutive start" in problem_statement.lower():
            return "复现/回归测试：验证连续 start directive 是否被拒绝。"
        return "复现/回归测试：确认自定义 skip directive 的错误消息。"
    if "patch.py" in paths:
        return "临时验证/替代方案：运行时修改 AbstractSkipParser 以验证问题。"
    if "pr296" in problem_statement.lower() or "consecutive start" in problem_statement.lower():
        if "grouped_code_block" in paths or "consecutive" in text or "start" in text and "end" in text:
            return "连续 start 约束修复"
    if "pr304" in problem_statement.lower() or "customdirectiveskipparser" in problem_statement.lower():
        if "custom_directive_skip.py" in paths:
            if "self._abstract_skip_parser.directive" in added.lower() or "skipper.directive" in added.lower():
                return "自定义 directive 传播修复"
            if "directive=directive" in added.lower():
                return "自定义 directive 参数接入"
            return "skip parser 实现迭代"
        if "abstract/skip.py" in paths or "sybil/abstract" in paths:
            return "AbstractSkipParser 核心修复"
    if "test" in paths:
        return "测试补充"
    if added and not removed:
        return "新增实现"
    if removed and not added:
        return "删除/清理实现"
    return "实现迭代"


def tool_intent(tool_names: list[str], diff: str, problem_statement: str) -> str:
    names = set(tool_names)
    paths = files(diff)
    if not diff.strip():
        return "探索/定位问题"
    if "Edit" in names or "Write" in names:
        if "test" in paths:
            return "编写/调整回归测试"
        return "实现代码修复"
    if "Bash" in names:
        return "运行测试并验证修复"
    if "Read" in names or "Grep" in names:
        return "阅读代码并定位修改点"
    return "继续分析当前修改"


def snapshot_tools(sample: dict, first: int, last: int, tokenizer) -> list[str]:
    text = tokenizer.decode(sample.get("tokens") or [], skip_special_tokens=False)
    messages = _split_messages(text)
    names: list[str] = []
    assistants = [body for role, body in messages if role == "assistant"]
    for turn in range(first, last + 1):
        if turn < len(assistants):
            names.extend(name for name, _ in _parse_tool_calls(assistants[turn]))
    return names


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--instance", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "/workspace/models/pyromind/PyroDash-4B-SFT-07313",
        trust_remote_code=True,
        local_files_only=True,
    )

    data = torch.load(args.rollout, map_location="cpu", weights_only=False)
    samples = [s for s in data.get("samples", []) if str((s.get("metadata") or {}).get("instance_id") or s.get("label") or "") == args.instance]
    samples.sort(key=lambda s: int(s.get("index", 0)))
    if len(samples) != 8:
        raise SystemExit(f"expected 8 trajectories, found {len(samples)}")

    task_context = {}
    with Path("examples/coding_agent_rl/data/swe_train_scaleswe_200_baked.jsonl").open(encoding="utf-8") as task_file:
        for line in task_file:
            record = json.loads(line)
            metadata = record.get("metadata") or {}
            task_id = str(metadata.get("instance_id") or record.get("label") or "")
            if task_id == args.instance:
                task_context = metadata
                break
    problem_statement = str(task_context.get("problem_statement") or "")

    columns = []
    total = 0
    for sample in samples:
        snapshots = distinct_diffs(sample)
        total += len(snapshots)
        cards = []
        for number, (first, last, diff) in enumerate(snapshots, 1):
            turn = f"t{first}" if first == last else f"t{first}–t{last}"
            title = f"#{number} · {turn} · {files(diff)}"
            tool_names = snapshot_tools(sample, first, last, tokenizer)
            unique_tools = list(dict.fromkeys(tool_names))
            tool_text = ", ".join(unique_tools) or "无 tool"
            intent = tool_intent(unique_tools, diff, problem_statement)
            body = f"<div class='intent'>工具：{html.escape(tool_text)}<br>意图：{html.escape(intent)}</div>"
            body += f"<pre>{html.escape(diff)}</pre>" if diff else "<div class='empty'>&lt;empty diff&gt;</div>"
            cards.append(f"<section class='snapshot'><div class='snapshot-title'>{html.escape(title)}</div>{body}</section>")
        columns.append(f"<article class='traj'><div class='traj-title'>trajectory {sample.get('index')}</div>{''.join(cards)}</article>")

    rollout_id = data.get("rollout_id", "?")
    out = "<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
    out += f"<title>{html.escape(args.instance)} diff sequence · rollout {rollout_id}</title><style>{CSS}</style></head><body>"
    out += f"<header>{html.escape(args.instance)} · 8 条轨迹并排 · 参考 problem_statement 的意图分类 · 每条轨迹按 turn 顺序展示不同累计 diff（共 {total} 个快照）</header>"
    out += f"<main class='grid'>{''.join(columns)}</main></body></html>"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out, encoding="utf-8")
    print(f"wrote {args.out}: 8 trajectories, {total} distinct diffs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
