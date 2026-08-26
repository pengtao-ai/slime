#!/usr/bin/env python3
"""Assign per-turn GiGPO rewards, then render HTML.

Matches ``gigpo_train.compute_advantages``:
  r_imm[T] = episode reward on the last turn, else 0
  G_t      = discounted return (γ=0.95)
  A_E      = episode mean-norm across sibling trajectories
  A_S      = step mean-norm of G within step groups
  A        = A_E + w · A_S

``--group-by git-diff``: exact cumulative git diff (original GiGPO-style).
``--group-by tool-intent``: same tools + same intent across sibling trajs.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.coding_agent_rl.render_gigpo_diff_sequence_html import (
    _parse_tool_calls,
    _split_messages,
)
from slime.utils.gigpo import (
    build_step_group,
    compute_step_discounted_returns,
    episode_norm_reward,
    step_norm_reward,
)

_BASH_READ = {"cat", "head", "tail", "less", "more", "nl", "bat"}
_BASH_GREP = {"grep", "egrep", "fgrep", "rg", "ag"}
_BASH_SED = {"sed", "awk"}
_BASH_PYTHON = {"python", "python3", "pypy", "pypy3"}
_BASH_PYTEST = {"pytest", "py.test"}
_BASH_LS = {"ls", "tree", "pwd", "find", "stat", "realpath"}


def _bash_binaries(command: str) -> list[str]:
    parts = re.split(r"\s*(?:&&|\|\||;|\|)\s*", command or "")
    binaries: list[str] = []
    for part in parts:
        toks = part.split()
        while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):
            toks = toks[1:]
        if toks and toks[0] in {"sudo", "time", "env", "command"}:
            toks = toks[1:]
            while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):
                toks = toks[1:]
        if not toks:
            continue
        binaries.append(toks[0].rsplit("/", 1)[-1])
    return binaries


def bash_kind(command: str) -> str:
    """Map a Bash `command` string to a coarse action: cat/grep/sed/python/pytest/..."""
    bins = _bash_binaries(command)
    text = (command or "").lower()
    if any(b in _BASH_PYTEST for b in bins) or "-m pytest" in text or re.search(r"\bpytest\b", text):
        return "pytest"
    if any(b in _BASH_PYTHON for b in bins):
        return "python"
    if any(b in _BASH_GREP for b in bins):
        return "grep"
    if any(b in _BASH_READ for b in bins):
        return "cat"
    if any(b in _BASH_SED for b in bins):
        return "sed"
    if any(b == "git" for b in bins):
        return "git"
    if any(b in _BASH_LS for b in bins):
        return "ls"
    return "other"


def tool_labels(calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    labels: list[str] = []
    for name, params in calls:
        if name == "Bash":
            labels.append(f"Bash:{bash_kind(str((params or {}).get('command') or ''))}")
        else:
            labels.append(name)
    return list(dict.fromkeys(labels))


def turn_intent(labels: list[str], diff: str, problem_statement: str) -> str:
    names = set(labels)
    bash = {lab.split(":", 1)[1] for lab in labels if lab.startswith("Bash:")}
    paths = _files(diff)
    empty = not (diff or "").strip()
    if empty:
        if bash & {"pytest", "python"}:
            return "探索/跑测试复现"
        return "探索/定位问题"
    if "Edit" in names or "Write" in names:
        if "test" in paths:
            return "编写/调整回归测试"
        return "实现代码修复"
    if bash & {"pytest", "python"}:
        return "运行测试并验证修复"
    if bash & {"cat", "grep", "sed", "git", "ls"} or "Read" in names or "Grep" in names:
        return "阅读代码并定位修改点"
    if any(lab.startswith("Bash:") for lab in labels):
        return "运行测试并验证修复"
    return "继续分析当前修改"


DIFF_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)
GROUP_COLORS = [
    "#1f6feb",
    "#2ea043",
    "#d2a8ff",
    "#e3b341",
    "#f85149",
    "#79c0ff",
    "#56d364",
    "#db61dd",
    "#ffa657",
    "#a371f7",
    "#39c5cf",
    "#ff7eb6",
]
DEFAULT_GAMMA = 0.95
DEFAULT_STEP_W = 1.0


def _files(diff: str) -> str:
    paths = []
    for a, b in DIFF_RE.findall(diff):
        paths.append(b if b != "/dev/null" else a)
    return ", ".join(sorted(set(paths))) or "<empty diff>"


def _summarize_diff(diff: str) -> str:
    if not (diff or "").strip():
        return "empty diff"
    first = diff.splitlines()[0].strip()
    if first.startswith("diff --git "):
        return first.removeprefix("diff --git ")[:120]
    return first[:120]


def _dedup_episodes(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[Any, Any], dict[str, Any]] = {}
    for s in samples:
        key = (s.get("group_index"), s.get("index"))
        cur = best.get(key)
        if cur is None or len(s.get("tokens") or []) > len(cur.get("tokens") or []):
            best[key] = s
    return sorted(best.values(), key=lambda x: (int(x.get("group_index") or 0), int(x.get("index") or 0)))


def _expand_turns(sample: dict[str, Any]) -> list[dict[str, Any]] | None:
    md = sample.get("metadata") or {}
    entries = [x for x in (md.get("turn_git_diffs") or []) if isinstance(x, dict)]
    if not entries:
        return None
    entries = sorted(entries, key=lambda x: int(x.get("turn_index") or 0))
    n = len(entries)
    episode_r = float(md.get("episode_reward", sample.get("reward") or 0.0) or 0.0)
    solved = float(md.get("solved") or 0.0)
    stats = md.get("offload_stats") or {}
    instance_id = str(md.get("instance_id") or sample.get("label") or "")
    traj_uid = str(md.get("traj_uid") or f"{sample.get('group_index')}:{sample.get('index')}")
    rows = []
    for i, ent in enumerate(entries):
        diff = str(ent.get("git_diff") or "")
        rows.append(
            {
                "turn": int(ent.get("turn_index") or i),
                "git_diff": diff,
                "diff_summary": _summarize_diff(diff),
                "files": _files(diff),
                "r_imm": episode_r if i == n - 1 else 0.0,
                "episode_reward": episode_r,
                "solved": solved,
                "empty_patch": bool(md.get("empty_patch")),
                "offload_count": int(stats.get("offload_count") or 0),
                "instance_id": instance_id,
                "group_index": sample.get("group_index"),
                "index": sample.get("index"),
                "traj_uid": traj_uid,
                "n_turns": n,
                "tools": [],
                "tools_str": "无 tool",
                "intent": "探索/定位问题",
                "group_key": "",
            }
        )
    return rows


def _problem_statement(sample: dict[str, Any]) -> str:
    md = sample.get("metadata") or {}
    if md.get("problem_statement"):
        return str(md["problem_statement"])
    prompt = sample.get("prompt") or []
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        return str(prompt[0].get("content") or "")
    if isinstance(prompt, str):
        return prompt
    return ""


def attach_tools_intent(sample: dict[str, Any], rows: list[dict[str, Any]], tokenizer) -> None:
    text = tokenizer.decode(sample.get("tokens") or [], skip_special_tokens=False)
    assistants = [body for role, body in _split_messages(text) if role == "assistant"]
    problem = _problem_statement(sample)
    for row in rows:
        calls: list[tuple[str, dict[str, Any]]] = []
        turn = int(row["turn"])
        if 0 <= turn < len(assistants):
            calls = _parse_tool_calls(assistants[turn])
        labels = tool_labels(calls)
        row["tools"] = labels
        row["tools_str"] = ", ".join(labels) or "无 tool"
        row["intent"] = turn_intent(labels, row.get("git_diff") or "", problem)
        row["group_key"] = f"{row['intent']}||{row['tools_str']}"


def _scalar_list(tensors: list[torch.Tensor]) -> list[float]:
    out = []
    for t in tensors:
        v = t.detach().reshape(-1)[0]
        out.append(float(v))
    return out


def assign_gigpo(rows: list[dict[str, Any]], *, gamma: float, step_w: float, group_by: str = "git-diff") -> None:
    if not rows:
        return
    n = len(rows)
    episode = torch.tensor([r["episode_reward"] for r in rows], dtype=torch.float32)
    r_imm = [r["r_imm"] for r in rows]
    traj_uids = [str(r["traj_uid"]) for r in rows]
    group_ids = [str(r["group_index"]) for r in rows]
    if group_by == "tool-intent":
        anchors = [r.get("group_key") or f"{r.get('intent')}||{r.get('tools_str')}" for r in rows]
        prefix = "T"
    else:
        anchors = [r["git_diff"] for r in rows]
        prefix = "D"
    order = sorted(range(n), key=lambda i: (traj_uids[i], int(rows[i]["turn"])))
    imm_sorted = [r_imm[i] for i in order]
    traj_sorted = [traj_uids[i] for i in order]
    g_sorted = compute_step_discounted_returns(imm_sorted, traj_sorted, gamma)
    g = torch.zeros(n, dtype=torch.float32)
    for new_pos, old_i in enumerate(order):
        g[old_i] = g_sorted[new_pos]

    masks = [torch.ones(1, dtype=torch.float32) for _ in range(n)]
    a_e = _scalar_list(
        episode_norm_reward(
            episode,
            masks,
            group_ids,
            traj_uids,
            remove_std=True,
            compute_mean_std_cross_steps=False,
        )
    )
    step_uids = build_step_group(np.asarray(anchors, dtype=object), group_ids, summarize=False)
    a_s = _scalar_list(step_norm_reward(g, masks, step_uids, remove_std=True))

    uid_to_members: dict[str, list[int]] = defaultdict(list)
    for i, uid in enumerate(step_uids.tolist()):
        uid_to_members[str(uid)].append(i)
    gid_by_uid: dict[str, int] = {}
    next_gid = 0
    # Stable labels: larger groups first, empty-diff last among ties.
    ranked = sorted(
        uid_to_members.items(),
        key=lambda kv: (-len(kv[1]), -len({rows[j]["index"] for j in kv[1]}), kv[0]),
    )
    for uid, members in ranked:
        if uid not in gid_by_uid:
            gid_by_uid[uid] = next_gid
            next_gid += 1

    for i, row in enumerate(rows):
        uid = str(step_uids[i])
        gid = gid_by_uid[uid]
        members = uid_to_members[uid]
        size = len(members)
        n_traj = len({rows[j]["index"] for j in members})
        row["G"] = float(g[i])
        row["A_E"] = a_e[i]
        row["A_S"] = a_s[i]
        row["A"] = a_e[i] + step_w * a_s[i]
        row["group_uid"] = uid
        row["group_size"] = size
        row["group_n_traj"] = n_traj
        row["group_kind"] = "cross-traj" if n_traj >= 2 else ("matched" if size >= 2 else "singleton")
        row["group_label"] = f"{prefix}{gid}" if size >= 2 else "·"
        row["group_color"] = GROUP_COLORS[gid % len(GROUP_COLORS)] if size >= 2 else "#3a3f47"
        row["group_mean_G"] = float(sum(float(g[j]) for j in members) / size)
        row["group_caption"] = f"{row.get('intent') or ''} · {row.get('tools_str') or row.get('diff_summary') or ''}"


def _collapse_snapshots(turns: list[dict[str, Any]], *, by: str = "git_diff") -> list[dict[str, Any]]:
    snaps: list[dict[str, Any]] = []
    for row in turns:
        same = snaps and (
            snaps[-1]["end"]["group_key"] == row["group_key"]
            if by == "group_key"
            else snaps[-1]["git_diff"] == row["git_diff"]
        )
        if same:
            snaps[-1]["last"] = row["turn"]
            snaps[-1]["end"] = row
            snaps[-1]["n"] += 1
        else:
            snaps.append(
                {
                    "first": row["turn"],
                    "last": row["turn"],
                    "n": 1,
                    "git_diff": row["git_diff"],
                    "start": row,
                    "end": row,
                }
            )
    return snaps


def render_sequence_html(
    instance_id: str,
    trajs: list[dict[str, Any]],
    *,
    rollout_id: int,
    gamma: float,
    group_by: str,
) -> str:
    n_traj = len(trajs)
    collapse_by = "group_key" if group_by == "tool-intent" else "git_diff"
    legend_seen: dict[str, dict[str, Any]] = {}
    columns = []
    for traj in trajs:
        snaps = _collapse_snapshots(traj["turns"], by=collapse_by)
        cards = []
        for number, snap in enumerate(snaps, 1):
            row = snap["end"]
            first, last = snap["first"], snap["last"]
            turn = f"t{first}" if first == last else f"t{first}–t{last}"
            files = row["files"]
            title = f"#{number} · {turn} · {files}"
            label = row["group_label"]
            color = row["group_color"]
            if label != "·" and label not in legend_seen:
                legend_seen[label] = row
            intent_bar = (
                f"<div class='intent'>工具：{html_lib.escape(row.get('tools_str') or '无 tool')}"
                f"<br>意图：{html_lib.escape(row.get('intent') or '—')}"
                f"<br>跨轨迹：{int(row.get('group_n_traj') or 1)}/{n_traj} 条</div>"
            )
            reward_bar = (
                f"<div class='reward' style='border-left:4px solid {color}'>"
                f"<span class='gid' style='background:{color}'>{html_lib.escape(label)}</span>"
                f"<span class='chip'>size={row['group_size']}</span>"
                f"<span class='chip'>trajs={int(row.get('group_n_traj') or 1)}</span>"
                f"<span class='chip'>r_imm={row['r_imm']:.4f}</span>"
                f"<span class='chip'>G={row['G']:.4f}</span>"
                f"<span class='chip'>Ā_G={row['group_mean_G']:.4f}</span>"
                f"<span class='chip'>A_S={row['A_S']:+.4f}</span>"
                f"<span class='chip'>A_E={row['A_E']:+.4f}</span>"
                f"<span class='chip A'>A={row['A']:+.4f}</span>"
                f"</div>"
            )
            body = f"<pre>{html_lib.escape(snap['end']['git_diff'])}</pre>" if snap["end"]["git_diff"].strip() else "<div class='empty'>&lt;empty diff&gt;</div>"
            cards.append(
                f"<section class='snapshot'><div class='snapshot-title'>{html_lib.escape(title)}</div>{intent_bar}{reward_bar}{body}</section>"
            )
        solved = traj["turns"][-1]["solved"] > 0 if traj["turns"] else False
        ep = traj["turns"][-1]["episode_reward"] if traj["turns"] else 0.0
        mark = "✓" if solved else "✗"
        title = (
            f"trajectory {traj['index']} · {mark} · R={ep:.4f} · {len(traj['turns'])} turns · "
            f"{len(snaps)} snapshots"
        )
        columns.append(f"<article class='traj'><div class='traj-title'>{html_lib.escape(title)}</div>{''.join(cards)}</article>")

    legend_parts = []
    for label, row in sorted(legend_seen.items(), key=lambda kv: int(kv[0][1:]) if kv[0][1:].isdigit() else 999):
        caption = row.get("group_caption") or row.get("diff_summary") or ""
        ntr = int(row.get("group_n_traj") or 1)
        legend_parts.append(
            f"<span class='chip' style='background:{row['group_color']}' title='{html_lib.escape(caption)}'>"
            f"{html_lib.escape(label)} ×{row['group_size']} · {ntr}traj · {html_lib.escape(caption[:80])}</span>"
        )
    legend = " ".join(legend_parts) or "<span class='muted'>无成组</span>"
    width = max(380, int(3040 * n_traj / 8)) if n_traj else 380
    group_desc = (
        "同色 T# = 相同「意图 + 工具集合」（跨轨迹聚合 A_S）"
        if group_by == "tool-intent"
        else "同色 D# = 相同 git diff（跨轨迹成组做 A_S）"
    )
    return f"""<!doctype html>
<html lang='zh'><head><meta charset='utf-8'>
<title>{html_lib.escape(instance_id)} {html_lib.escape(group_by)} GiGPO rewards · rollout {rollout_id}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #0d1117; color: #c9d1d9; font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }}
header {{ position: sticky; top: 0; z-index: 2; padding: 12px 16px; background: #161b22; border-bottom: 1px solid #30363d; font-family: sans-serif; }}
header h1 {{ margin: 0 0 4px; font: 600 14px sans-serif; color: #f0f6fc; }}
header .sub {{ color: #8b949e; font: 12px sans-serif; }}
.legend {{ margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px; font-family: sans-serif; }}
.legend .chip {{ font-size: 10px; padding: 2px 7px; border-radius: 8px; color: #fff; }}
.grid {{ display: grid; grid-template-columns: repeat({n_traj}, minmax(380px, 1fr)); gap: 8px; min-width: {width}px; padding: 10px; align-items: start; }}
.traj {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }}
.traj-title {{ padding: 8px; background: #21262d; font: 600 12px sans-serif; color: #f0f6fc; }}
.snapshot {{ border-top: 1px solid #30363d; }}
.snapshot-title {{ padding: 6px 8px; background: #1c2128; color: #79c0ff; font: 600 11px sans-serif; }}
.intent {{ padding: 7px 8px; background: #18232f; color: #f0c674; font: 12px sans-serif; border-top: 1px solid #30363d; border-bottom: 1px solid #30363d; }}
.reward {{ padding: 6px 8px; background: #18232f; font: 11px sans-serif; display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }}
.reward .gid {{ display: inline-block; min-width: 22px; padding: 0 5px; border-radius: 6px; color: #fff; font-weight: 700; }}
.reward .chip {{ background: #21262d; color: #c9d1d9; padding: 1px 6px; border-radius: 8px; }}
.reward .chip.A {{ background: #1f6feb; color: #fff; font-weight: 700; }}
pre {{ margin: 0; padding: 8px; overflow: auto; max-height: 420px; white-space: pre; }}
.empty {{ padding: 14px 8px; color: #8b949e; font-style: italic; }}
.muted {{ color: #8b949e; font: 11px sans-serif; }}
</style></head><body>
<header>
  <h1>{html_lib.escape(instance_id)} · {html_lib.escape(group_by)} 分组 · 轨迹间 GiGPO 每轮 reward</h1>
  <div class='sub'>rollout {rollout_id} · {n_traj} 条 sibling traj · γ={gamma} · A = A_E + A_S · {group_desc} · 快照上的 G/A 取该段最后一轮</div>
  <div class='legend'>{legend}</div>
</header>
<main class='grid'>{''.join(columns)}</main>
</body></html>"""


def render_interactive_html(
    instances: list[dict[str, Any]],
    *,
    run_dir: Path,
    rollout_id: int,
    gamma: float,
    group_by: str,
) -> str:
    payload = []
    for inst in instances:
        trajs_out = []
        for traj in inst["trajs"]:
            turns_out = []
            for t in traj["turns"]:
                turns_out.append(
                    {
                        "turn": t["turn"],
                        "files": t["files"],
                        "diff_summary": t["diff_summary"],
                        "r_imm": round(t["r_imm"], 6),
                        "G": round(t["G"], 6),
                        "A_E": round(t["A_E"], 6),
                        "A_S": round(t["A_S"], 6),
                        "A": round(t["A"], 6),
                        "group_label": t["group_label"],
                        "group_size": t["group_size"],
                        "group_color": t["group_color"],
                        "group_mean_G": round(t["group_mean_G"], 6),
                        "group_kind": t["group_kind"],
                        "tools_str": t.get("tools_str") or "",
                        "intent": t.get("intent") or "",
                        "group_n_traj": int(t.get("group_n_traj") or 1),
                        "group_caption": t.get("group_caption") or "",
                    }
                )
            last = traj["turns"][-1]
            trajs_out.append(
                {
                    "index": last["index"],
                    "traj_uid": last["traj_uid"],
                    "solved": last["solved"],
                    "episode_reward": last["episode_reward"],
                    "offload_count": last["offload_count"],
                    "n_turns": last["n_turns"],
                    "turns": turns_out,
                }
            )
        payload.append({"instance_id": inst["instance_id"], "group_index": inst["group_index"], "trajs": trajs_out})
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"/>
<title>GiGPO {html_lib.escape(group_by)} turn rewards · rollout {rollout_id}</title>
<style>
:root {{ --bg:#0d1117; --panel:#161b22; --line:#30363d; --ink:#c9d1d9; --muted:#8b949e; --blue:#79c0ff; --green:#56d364; --red:#f85149; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:13px/1.45 -apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif; }}
header {{ position:sticky; top:0; z-index:5; background:var(--panel); border-bottom:1px solid var(--line); padding:12px 16px; }}
header h1 {{ margin:0 0 4px; font-size:16px; }}
header .sub {{ color:var(--muted); font-size:12px; }}
.controls {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; align-items:center; }}
.controls select, .controls input {{ background:#0d1117; color:var(--ink); border:1px solid var(--line); border-radius:6px; padding:4px 8px; }}
.layout {{ display:grid; grid-template-columns:280px 1fr; min-height:calc(100vh - 120px); }}
.sidebar {{ border-right:1px solid var(--line); overflow:auto; max-height:calc(100vh - 120px); padding:8px; }}
.main {{ overflow:auto; max-height:calc(100vh - 120px); padding:12px 16px 40px; }}
.item {{ display:block; width:100%; text-align:left; background:transparent; border:1px solid transparent; border-radius:8px; padding:8px; color:var(--ink); cursor:pointer; margin:0 0 4px; }}
.item:hover {{ background:#21262d; }}
.item.active {{ border-color:#1f6feb; background:#1f2937; }}
.badge {{ display:inline-block; font-size:10px; padding:1px 6px; border-radius:999px; background:#21262d; margin-right:4px; }}
.badge.yes {{ background:#1b3a23; color:var(--green); }}
.badge.no {{ background:#3a1b1b; color:var(--red); }}
.legend {{ display:flex; flex-wrap:wrap; gap:4px; margin:8px 0 12px; }}
.legend .chip {{ font-size:10px; padding:2px 7px; border-radius:8px; color:#fff; }}
table.grid {{ border-collapse:separate; border-spacing:3px; width:100%; table-layout:fixed; }}
th.col-head {{ padding:6px 8px; background:#21262d; border-radius:6px; font-size:11px; text-align:left; position:sticky; top:0; }}
td.tlabel {{ width:44px; color:var(--muted); font-size:10px; text-align:right; }}
td.step {{ background:var(--panel); border-radius:4px; padding:5px 6px 5px 10px; position:relative; font-size:11px; vertical-align:top; }}
td.step.empty {{ background:transparent; }}
td.step .bar {{ position:absolute; left:0; top:2px; bottom:2px; width:4px; border-radius:4px; }}
.gid {{ display:inline-block; min-width:20px; padding:0 4px; border-radius:6px; color:#fff; font-size:9px; }}
.num {{ font-variant-numeric:tabular-nums; }}
.pos {{ color:var(--green); }}
.neg {{ color:var(--red); }}
.zero {{ color:var(--muted); }}
.note {{ color:var(--muted); font-size:11px; margin:0 0 10px; }}
</style></head>
<body>
<header>
  <h1>GiGPO {html_lib.escape(group_by)} 分组 · 每轮 reward</h1>
  <div class="sub">{html_lib.escape(str(run_dir))} · rollout {rollout_id} · γ={gamma} · A = A_E + A_S（轨迹间按工具+意图聚合 A_S）</div>
  <div class="controls">
    <label>instance <select id="inst"></select></label>
  </div>
</header>
<div class="layout">
  <aside class="sidebar" id="list"></aside>
  <main class="main" id="main"></main>
</div>
<script>
const DATA = {data_json};
const instSel = document.getElementById('inst');
const listEl = document.getElementById('list');
const mainEl = document.getElementById('main');
DATA.forEach((inst, i) => {{
  const o = document.createElement('option');
  o.value = String(i);
  o.textContent = inst.instance_id + ' (' + inst.trajs.length + ')';
  instSel.appendChild(o);
}});
function cls(x) {{
  const v = Number(x);
  if (v > 1e-9) return 'num pos';
  if (v < -1e-9) return 'num neg';
  return 'num zero';
}}
function fmt(x) {{ return Number(x).toFixed(4); }}
function render() {{
  const inst = DATA[Number(instSel.value) || 0];
  listEl.innerHTML = '';
  inst.trajs.forEach((t, i) => {{
    const b = document.createElement('button');
    b.className = 'item' + (i===0 ? ' active' : '');
    b.innerHTML = `<div><b>sample ${{t.index}}</b></div>
      <div><span class="badge ${{t.solved>0?'yes':'no'}}">${{t.solved>0?'solved':'fail'}}</span>
      <span class="badge">R=${{fmt(t.episode_reward)}}</span>
      <span class="badge">${{t.n_turns}} turns</span></div>`;
    b.onclick = () => {{
      [...listEl.children].forEach(el => el.classList.remove('active'));
      b.classList.add('active');
      /* keep grid of all trajs; just scroll not needed */
    }};
    listEl.appendChild(b);
  }});
  const groups = new Map();
  inst.trajs.forEach(t => t.turns.forEach(row => {{
    if (row.group_label !== '·' && !groups.has(row.group_label)) groups.set(row.group_label, row);
  }}));
  const legend = [...groups.entries()].sort((a,b)=>Number(a[0].slice(1))-Number(b[0].slice(1)))
    .map(([lab,row]) => `<span class="chip" style="background:${{row.group_color}}">${{lab}} ×${{row.group_size}} · ${{row.group_n_traj}}traj · ${{row.intent}} · ${{row.tools_str}}</span>`).join(' ');
  const maxTurn = Math.max(0, ...inst.trajs.map(t => t.turns.length ? t.turns[t.turns.length-1].turn : 0));
  const byTurn = inst.trajs.map(t => {{
    const m = new Map(t.turns.map(row => [row.turn, row]));
    return m;
  }});
  const heads = inst.trajs.map(t => `<th class="col-head">s${{t.index}} <span class="badge ${{t.solved>0?'yes':'no'}}">${{t.solved>0?'✓':'✗'}}</span><span class="badge">R=${{fmt(t.episode_reward)}}</span></th>`).join('');
  let body = '';
  for (let turn=0; turn<=maxTurn; turn++) {{
    const cells = byTurn.map(m => {{
      const row = m.get(turn);
      if (!row) return "<td class='step empty'></td>";
      return `<td class="step">
        <div class="bar" style="background:${{row.group_color}}"></div>
        <span class="gid" style="background:${{row.group_color}}">${{row.group_label}}</span>
        <div>${{row.intent}}</div>
        <div>${{row.tools_str}}</div>
        <div>G <span class="${{cls(row.G)}}">${{fmt(row.G)}}</span>
            A_S <span class="${{cls(row.A_S)}}">${{fmt(row.A_S)}}</span>
            A <span class="${{cls(row.A)}}">${{fmt(row.A)}}</span></div>
      </td>`;
    }});
    if (cells.every(c => c.includes('empty'))) continue;
    body += `<tr><td class="tlabel">t${{turn}}</td>${{cells.join('')}}</tr>`;
  }}
  mainEl.innerHTML = `
    <h2>${{inst.instance_id}}</h2>
    <p class="note">每列一条 sibling 轨迹；同行是同一 turn 索引。同色 T# = 相同意图+工具集合，A_S = G − 组内 mean(G)。A_E 按整条轨迹 episode reward 相对 sibling 均值。</p>
    <div class="legend">${{legend || '<span class="badge">无跨轨迹成组</span>'}}</div>
    <table class="grid"><tr><th class="col-head" style="width:44px">turn</th>${{heads}}</tr>${{body}}</table>`;
}}
instSel.addEventListener('change', render);
render();
</script>
</body></html>"""


def load_rollout(path: Path) -> tuple[int, list[dict[str, Any]]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    rollout_id = int(data.get("rollout_id") or path.stem.split("_")[1])
    samples = _dedup_episodes(list(data.get("samples") or []))
    return rollout_id, samples


def build_instances(
    samples: list[dict[str, Any]],
    *,
    gamma: float,
    step_w: float,
    group_by: str,
    tokenizer=None,
) -> list[dict[str, Any]]:
    by_inst: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        md = s.get("metadata") or {}
        inst = str(md.get("instance_id") or s.get("label") or "")
        by_inst[(s.get("group_index"), inst)].append(s)

    instances = []
    for (gidx, inst), group_samples in sorted(by_inst.items(), key=lambda kv: (int(kv[0][0] or 0), kv[0][1])):
        all_rows: list[dict[str, Any]] = []
        traj_rows: list[tuple[int, list[dict[str, Any]]]] = []
        for s in group_samples:
            rows = _expand_turns(s)
            if not rows:
                continue
            if group_by == "tool-intent":
                if tokenizer is None:
                    raise ValueError("tokenizer is required for tool-intent grouping")
                attach_tools_intent(s, rows, tokenizer)
            else:
                for row in rows:
                    row["group_key"] = row["git_diff"]
            traj_rows.append((int(s.get("index") or 0), rows))
            all_rows.extend(rows)
        if not all_rows:
            continue
        assign_gigpo(all_rows, gamma=gamma, step_w=step_w, group_by=group_by)
        instances.append(
            {
                "instance_id": inst,
                "group_index": gidx,
                "trajs": [{"index": idx, "turns": rows} for idx, rows in traj_rows],
            }
        )
    return instances


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/workspace/work/spt/slime/runs/agent_offload_pyrodash4b_docker_async_20260804_035850"),
    )
    p.add_argument("--rollout", type=int, default=0)
    p.add_argument("--instance", default="adamtheturtle_sybil-extras_pr296")
    p.add_argument("--group-by", choices=["git-diff", "tool-intent"], default="tool-intent")
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--step-w", type=float, default=DEFAULT_STEP_W)
    p.add_argument("--hf-checkpoint", default="/workspace/models/pyromind/PyroDash-4B-SFT-07313")
    args = p.parse_args()

    dump = args.run_dir / "rollout_dumps" / f"rollout_{args.rollout}.pt"
    if not dump.exists():
        print(f"missing {dump}", file=sys.stderr)
        return 1
    print(f"load {dump}", flush=True)
    rollout_id, samples = load_rollout(dump)
    print(f"episodes after dedup: {len(samples)}", flush=True)
    tokenizer = None
    if args.group_by == "tool-intent":
        from transformers import AutoTokenizer

        print(f"load tokenizer {args.hf_checkpoint}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True, local_files_only=True)
    instances = build_instances(
        samples, gamma=args.gamma, step_w=args.step_w, group_by=args.group_by, tokenizer=tokenizer
    )
    print(f"instances: {[i['instance_id'] for i in instances]}", flush=True)

    tag = "tool_intent" if args.group_by == "tool-intent" else "gitdiff"
    all_path = args.run_dir / f"gigpo_{tag}_turn_rewards_rollout{rollout_id}.html"
    all_path.write_text(
        render_interactive_html(
            instances, run_dir=args.run_dir, rollout_id=rollout_id, gamma=args.gamma, group_by=args.group_by
        ),
        encoding="utf-8",
    )
    print(f"wrote {all_path}", flush=True)

    target = next((i for i in instances if i["instance_id"] == args.instance), None)
    if target is None:
        print(f"instance {args.instance} not in rollout {rollout_id}", file=sys.stderr)
        return 0
    seq_path = args.run_dir / f"gigpo_{tag}_turn_rewards_{args.instance.split('_')[-1]}_diff_sequence.html"
    seq_path.write_text(
        render_sequence_html(
            target["instance_id"],
            target["trajs"],
            rollout_id=rollout_id,
            gamma=args.gamma,
            group_by=args.group_by,
        ),
        encoding="utf-8",
    )
    print(f"wrote {seq_path}", flush=True)
    n_cross = sum(1 for t in target["trajs"] for row in t["turns"] if row["group_kind"] == "cross-traj")
    n_total = sum(len(t["turns"]) for t in target["trajs"])
    n_groups = len({row["group_label"] for t in target["trajs"] for row in t["turns"] if row["group_label"] != "·"})
    print(
        f"{args.instance}: {len(target['trajs'])} trajs, {n_cross}/{n_total} steps in cross-traj groups, "
        f"{n_groups} labeled groups",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
