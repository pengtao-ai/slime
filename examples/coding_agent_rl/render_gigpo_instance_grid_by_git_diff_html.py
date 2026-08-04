#!/usr/bin/env python3
"""Render one instance's sibling trajectories as a side-by-side HTML table.

Each column = one sibling trajectory. Each row = turn index. Each cell = one
SLM step, colored by git-diff groups.

Grouping strategy:
  - exact group: identical diff text
  - similar group: high file-set overlap + token overlap
  - singleton: shown in grey

Usage:
  PYTHONPATH=. python examples/coding_agent_rl/render_gigpo_instance_grid_by_git_diff_html.py \
    --rollout runs/.../rollout_dumps/rollout_1.pt \
    --instance adamtheturtle_sybil-extras_pr304 \
    --out runs/.../gigpo_instance_grid_by_git_diff_sybil_pr304_fuzzy.html
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from examples.coding_agent_rl.analyse_py.analyze_gigpo_groups_from_rollout import (
    _TOOL_RESP,
    _guess_workdir,
    _parse_tool_calls,
    _split_messages,
    extract_turns_from_decoded,
)

GROUP_COLORS = [
    "#1f6feb", "#2ea043", "#d2a8ff", "#e3b341", "#f85149", "#79c0ff",
    "#56d364", "#db61dd", "#ffa657", "#a371f7", "#39c5cf", "#ff7eb6",
]
FILE_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")
SIMILARITY_THRESHOLD = 0.65
FILE_JACCARD_THRESHOLD = 0.80
ACTION_SIGNATURE_THRESHOLD = 0.60

CSS = """
body { font: 11px/1.4 -apple-system, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
       margin: 0; background: #0d1117; color: #c9d1d9; }
header { padding: 12px 16px; background: #161b22; border-bottom: 1px solid #30363d; position: sticky; top: 0; z-index: 2; }
header h1 { margin: 0 0 3px 0; font-size: 14px; }
header .sub { color: #8b949e; font-size: 11px; }
.legend { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; }
.legend .chip { font-size: 10px; padding: 1px 6px; border-radius: 8px; color: #fff; cursor: default; }
main { padding: 10px 12px 40px; }
table.grid { border-collapse: separate; border-spacing: 3px; table-layout: fixed; width: 100%; }
table.grid th, table.grid td { vertical-align: top; }
th.col-head { padding: 6px 8px; background: #21262d; border-radius: 6px; font-weight: 600; font-size: 11px;
  text-align: left; position: sticky; top: 64px; z-index: 1; }
th.tcol { width: 44px; background: #161b22; color: #8b949e; font-weight: 600; font-size: 10px; }
td.tlabel { padding: 4px 6px; color: #8b949e; font-size: 10px; text-align: right; background: #161b22; width: 44px; }
td.step { padding: 4px 6px 4px 10px; background: #161b22; border-radius: 4px; position: relative; min-height: 22px; }
td.step.empty { background: transparent; }
td.step .bar { position: absolute; left: 0; top: 2px; bottom: 2px; width: 4px; border-radius: 4px; }
td.step .tn { color: #79c0ff; font-weight: 600; }
td.step .tv { color: #d2a8ff; }
td.step .muted { color: #6e7681; }
td.step .gid { display: inline-block; min-width: 24px; padding: 0 4px; border-radius: 6px; font-size: 9px; color: #fff; margin-right: 3px; }
.badge { font-size: 9px; padding: 0 4px; border-radius: 6px; background: #21262d; color: #c9d1d9; }
.badge.yes { background: #1b3a23; color: #56d364; }
.badge.no { background: #3a1b1b; color: #f85149; }
.badge.off { background: #3b2e10; color: #e3b341; }
.badge.len { background: #21262d; color: #8b949e; }
.badge.kind { background: #253041; color: #9cc2ff; }
.tooltip { position: fixed; z-index: 50; background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  padding: 10px 12px; width: 520px; max-width: 90vw; max-height: 70vh; overflow: auto; font-size: 12px; line-height: 1.5;
  color: #c9d1d9; white-space: pre-wrap; word-break: break-word; pointer-events: none; display: none;
  box-shadow: 0 6px 20px rgba(0,0,0,0.7); }
"""


def _tool_of_anchor(anchor: str) -> str:
    try:
        obs = (json.loads(anchor).get("obs") or ["?"])
        return str(obs[0]) if obs else "?"
    except (json.JSONDecodeError, TypeError, AttributeError, IndexError):
        return "?"


def _summarize_diff(diff_text: str) -> str:
    if not diff_text.strip():
        return "empty diff"
    first = diff_text.splitlines()[0].strip()
    if first.startswith("diff --git "):
        return first.removeprefix("diff --git ")[:120]
    return first[:120]


def _extract_files(diff_text: str) -> tuple[str, ...]:
    if not diff_text.strip():
        return ()
    files = []
    for a_path, b_path in FILE_RE.findall(diff_text):
        path = b_path if b_path != "/dev/null" else a_path
        files.append(path)
    return tuple(sorted(set(files)))


def _token_counter(diff_text: str) -> Counter[str]:
    return Counter(tok.lower() for tok in TOKEN_RE.findall(diff_text))


def _jaccard_files(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _counter_jaccard(a: Counter[str], b: Counter[str]) -> float:
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / max(union, 1)


def _shape_similarity(a: str, b: str) -> float:
    la = len(a.splitlines())
    lb = len(b.splitlines())
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb, 1)


def _similarity(ma: dict[str, Any], mb: dict[str, Any], *, mode: str = "weighted") -> float:
    file_sim = _jaccard_files(ma["diff_files"], mb["diff_files"])
    if mode == "files-only":
        return file_sim
    token_sim = _counter_jaccard(ma["diff_tokens"], mb["diff_tokens"])
    shape_sim = _shape_similarity(ma["diff_key"], mb["diff_key"])
    return 0.5 * file_sim + 0.35 * token_sim + 0.15 * shape_sim


def _normalize_tool_input(inp: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    items: list[tuple[str, str]] = []
    for k in sorted(inp):
        v = inp[k]
        if k == "file_path":
            items.append((k, str(v).split("/")[-1]))
        elif k == "command":
            items.append((k, str(v).split()[0] if str(v).split() else ""))
        elif isinstance(v, (str, int, float, bool)):
            items.append((k, str(v)[:40]))
        else:
            items.append((k, type(v).__name__))
    return tuple(items)


def _action_signature(row: dict[str, Any]) -> tuple[str, str, tuple[str, ...], tuple[tuple[str, str], ...]]:
    actions = row.get("this_turn_tool_calls") or []
    first = actions[0] if actions else {}
    tool_name = str(first.get("name") or "—")
    tool_input = _normalize_tool_input((first.get("input") if first else {}) or {})
    anchor_tool = _tool_of_anchor(row.get("anchor_obs") or "")
    return (anchor_tool, tool_name, row.get("diff_files") or (), tool_input)


def _action_signature_summary(sig: tuple[str, str, tuple[str, ...], tuple[tuple[str, str], ...]]) -> str:
    anchor_tool, tool_name, diff_files, tool_input = sig
    files = ",".join(diff_files[:2]) if diff_files else "∅"
    arg_keys = ",".join(k for k, _ in tool_input[:2]) if tool_input else "∅"
    return f"{anchor_tool} → {tool_name} · files={files} · args={arg_keys}"


def _action_signature_similarity(ma: dict[str, Any], mb: dict[str, Any]) -> float:
    sig_a = ma["action_signature"]
    sig_b = mb["action_signature"]
    anchor_sim = 1.0 if sig_a[0] == sig_b[0] else 0.0
    tool_sim = 1.0 if sig_a[1] == sig_b[1] else 0.0
    file_sim = _jaccard_files(ma["diff_files"], mb["diff_files"])
    input_a = Counter(f"{k}={v}" for k, v in sig_a[3])
    input_b = Counter(f"{k}={v}" for k, v in sig_b[3])
    input_sim = _counter_jaccard(input_a, input_b)
    return 0.30 * anchor_sim + 0.30 * tool_sim + 0.25 * file_sim + 0.15 * input_sim


def build_rows(samples: list[dict[str, Any]], *, tok) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in samples:
        md = s.get("metadata") or {}
        instance_id = str(md.get("instance_id") or s.get("label") or "")
        prompt = s.get("prompt") or []
        problem = ""
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            problem = str(prompt[0].get("content") or "")
        elif isinstance(prompt, str):
            problem = prompt
        text = tok.decode(s.get("tokens") or [], skip_special_tokens=False)
        workdir = _guess_workdir(instance_id, text)
        turns = extract_turns_from_decoded(
            text, instance_id=instance_id, problem_statement=problem, workdir=workdir
        )
        msgs = _split_messages(text)
        asst_idxs = [i for i, (r, _) in enumerate(msgs) if r == "assistant"]
        stats = md.get("offload_stats") or {}
        turn_git_diffs = md.get("turn_git_diffs") or []
        diff_by_turn = {
            int(item.get("turn_index")): str(item.get("git_diff") or "")
            for item in turn_git_diffs
            if isinstance(item, dict) and item.get("turn_index") is not None
        }
        for t in turns:
            ai = asst_idxs[t["turn_index"]]
            prev_tool_name = None
            prev_tool_input = None
            prev_tool_result = None
            for j in range(ai - 1, -1, -1):
                r, b = msgs[j]
                if r not in ("user", "tool"):
                    continue
                resps = _TOOL_RESP.findall(b)
                if not resps:
                    continue
                prev_tool_result = resps[-1]
                for k in range(j - 1, -1, -1):
                    if msgs[k][0] == "assistant":
                        calls = _parse_tool_calls(msgs[k][1])
                        if calls:
                            prev_tool_name, prev_tool_input = calls[-1]
                        break
                break
            this_calls = _parse_tool_calls(msgs[ai][1])
            git_diff = diff_by_turn.get(t["turn_index"], "")
            rows.append(
                {
                    "sample_index": s.get("index"),
                    "turn_index": t["turn_index"],
                    "row_id": f"{s.get('index')}:{t['turn_index']}",
                    "anchor_obs": t["anchor_obs"],
                    "diff_key": git_diff,
                    "diff_summary": _summarize_diff(git_diff),
                    "diff_files": _extract_files(git_diff),
                    "diff_tokens": _token_counter(git_diff),
                    "group_kind": "singleton",
                    "group_score": 0.0,
                    "group_label": "·",
                    "is_init": t["is_init"],
                    "prev_tool_call": (
                        {"name": prev_tool_name, "input": prev_tool_input}
                        if prev_tool_name
                        else None
                    ),
                    "prev_tool_response_preview": (prev_tool_result or "")[:800],
                    "this_turn_tool_calls": [
                        {"name": n, "input": inp} for n, inp in this_calls[:4]
                    ],
                    "reward": float(s.get("reward") or 0.0),
                    "solved": float(md.get("solved", 1.0 if md.get("grading_solved") else 0.0) or 0.0),
                    "offload_count": int(stats.get("offload_count") or 0),
                    "response_length": int(s.get("response_length") or 0),
                    "instance_id": instance_id,
                }
            )
            rows[-1]["action_signature"] = _action_signature(rows[-1])
            rows[-1]["action_summary"] = _action_signature_summary(rows[-1]["action_signature"])
    return rows


def assign_groups(
    rows: list[dict[str, Any]],
    *,
    grouping_mode: str = "git-diff",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    file_jaccard_threshold: float = FILE_JACCARD_THRESHOLD,
    similarity_mode: str = "weighted",
    action_signature_threshold: float = ACTION_SIGNATURE_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    if grouping_mode == "action-signature":
        return assign_action_signature_groups(rows, action_signature_threshold=action_signature_threshold)

    by_diff: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_diff[r["diff_key"]].append(r)

    diff_meta: dict[str, dict[str, Any]] = {}
    gid = 0

    exact_candidates = [
        (diff_text, members)
        for diff_text, members in by_diff.items()
        if len(members) >= 2
    ]
    for diff_text, members in sorted(exact_candidates, key=lambda kv: (-len(kv[1]), kv[0])):
        label = f"D{gid}"
        gid += 1
        diff_meta[diff_text] = {
            "gid": gid - 1,
            "label": label,
            "kind": "exact",
            "score": 1.0,
            "color": GROUP_COLORS[(gid - 1) % len(GROUP_COLORS)],
            "size": len(members),
            "summary": _summarize_diff(diff_text),
        }
        for m in members:
            diff_meta[m["row_id"]] = {
                "gid": gid - 1,
                "label": label,
                "kind": "exact",
                "score": 1.0,
                "color": GROUP_COLORS[(gid - 1) % len(GROUP_COLORS)],
                "size": len(members),
                "summary": _summarize_diff(diff_text),
            }
            m["group_kind"] = "exact"
            m["group_score"] = 1.0
            m["group_label"] = label

    singleton_keys = [k for k in by_diff if all(m["group_label"] == "·" for m in by_diff[k])]
    used: set[str] = set()
    for key in singleton_keys:
        if key in used:
            continue
        base = by_diff[key][0]
        cluster = [key]
        used.add(key)
        for other in singleton_keys:
            if other in used:
                continue
            cand = by_diff[other][0]
            file_sim = _jaccard_files(base["diff_files"], cand["diff_files"])
            if file_sim < file_jaccard_threshold:
                continue
            sim = _similarity(base, cand, mode=similarity_mode)
            if sim >= similarity_threshold:
                cluster.append(other)
                used.add(other)
        if len(cluster) < 2:
            continue
        label = f"S{gid}"
        gid += 1
        color = GROUP_COLORS[(gid - 1) % len(GROUP_COLORS)]
        total_members = sum(len(by_diff[k]) for k in cluster)
        best_summary = _summarize_diff(cluster[0])
        avg_score = 0.0
        count = 0
        for diff_text in cluster:
            members = by_diff[diff_text]
            sim = 1.0 if diff_text == cluster[0] else _similarity(base, members[0], mode=similarity_mode)
            avg_score += sim * len(members)
            count += len(members)
            for m in members:
                diff_meta[m["row_id"]] = {
                    "gid": gid - 1,
                    "label": label,
                    "kind": "similar",
                    "score": sim,
                    "color": color,
                    "size": total_members,
                    "summary": best_summary,
                }
                m["group_kind"] = "similar"
                m["group_score"] = sim
                m["group_label"] = label
        _ = avg_score / max(count, 1)

    return diff_meta


def assign_action_signature_groups(
    rows: list[dict[str, Any]],
    *,
    action_signature_threshold: float = ACTION_SIGNATURE_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    by_sig: dict[tuple[str, str, tuple[str, ...], tuple[tuple[str, str], ...]], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sig[r["action_signature"]].append(r)

    diff_meta: dict[str, dict[str, Any]] = {}
    gid = 0
    exact_candidates = [
        (sig, members)
        for sig, members in by_sig.items()
        if len(members) >= 2
    ]
    for sig, members in sorted(exact_candidates, key=lambda kv: (-len(kv[1]), str(kv[0]))):
        label = f"A{gid}"
        gid += 1
        summary = _action_signature_summary(sig)
        color = GROUP_COLORS[(gid - 1) % len(GROUP_COLORS)]
        for m in members:
            diff_meta[m["row_id"]] = {
                "gid": gid - 1,
                "label": label,
                "kind": "action-exact",
                "score": 1.0,
                "color": color,
                "size": len(members),
                "summary": summary,
            }
            m["group_kind"] = "action-exact"
            m["group_score"] = 1.0
            m["group_label"] = label

    singleton_rows = [r for r in rows if r["group_label"] == "·"]
    used: set[int] = set()
    for idx, base in enumerate(singleton_rows):
        if idx in used:
            continue
        cluster = [idx]
        used.add(idx)
        for j, cand in enumerate(singleton_rows):
            if j in used:
                continue
            sim = _action_signature_similarity(base, cand)
            if sim >= action_signature_threshold:
                cluster.append(j)
                used.add(j)
        if len(cluster) < 2:
            continue
        label = f"AS{gid}"
        gid += 1
        color = GROUP_COLORS[(gid - 1) % len(GROUP_COLORS)]
        members = [singleton_rows[k] for k in cluster]
        summary = members[0]["action_summary"]
        for m in members:
            sim = 1.0 if m is members[0] else _action_signature_similarity(members[0], m)
            diff_meta[m["row_id"]] = {
                "gid": gid - 1,
                "label": label,
                "kind": "action-similar",
                "score": sim,
                "color": color,
                "size": len(members),
                "summary": summary,
            }
            m["group_kind"] = "action-similar"
            m["group_score"] = sim
            m["group_label"] = label
    return diff_meta


def _tooltip(m: dict[str, Any]) -> str:
    prev = m.get("prev_tool_call") or {}
    prev_name = prev.get("name") if prev else "—"
    prev_inp = prev.get("input") if prev else {}
    prev_summary = json.dumps(prev_inp, ensure_ascii=False)[:200] if prev_inp else ""
    actions = m.get("this_turn_tool_calls") or []
    acts = ", ".join(
        f"{c.get('name')}({json.dumps(c.get('input') or {}, ensure_ascii=False)[:120]})" for c in actions
    )
    diff_text = m.get("diff_key") or ""
    diff_preview = diff_text[:3000] if diff_text.strip() else "<empty diff>"
    files = ", ".join(m.get("diff_files") or ()) or "<none>"
    text = (
        f"sample {m['sample_index']} · turn {m['turn_index']}\n"
        f"group: {m.get('group_label', '·')} ({m.get('group_kind', 'singleton')}, score={m.get('group_score', 0.0):.2f})\n"
        f"files: {files}\n"
        f"action signature: {m.get('action_summary', '—')}\n"
        f"prev obs: {prev_name} {prev_summary}\n"
        f"prev resp:\n{(m.get('prev_tool_response_preview') or '')[:800]}\n"
        f"this action: {acts}\n"
        f"anchor tool: {_tool_of_anchor(m['anchor_obs'])}\n"
        f"git diff:\n{diff_preview}"
    )
    return html.escape(text)


def _step_cell(m: dict[str, Any] | None, diff_meta: dict[str, dict[str, Any]]) -> str:
    if m is None:
        return "<td class='step empty'></td>"
    meta = diff_meta.get(m["row_id"])
    if meta is None:
        color = "#3a3f47"
        gid_label = "·"
        kind_label = "single"
    else:
        color = meta["color"]
        gid_label = meta["label"]
        kind_label = meta["kind"]
    acts = m.get("this_turn_tool_calls") or []
    act_name = acts[0].get("name") if acts else "—"
    inp = (acts[0].get("input") if acts else {}) or {}
    if "file_path" in inp:
        arg = str(inp["file_path"]).split("/")[-1][:22]
    elif "command" in inp:
        arg = str(inp["command"])[:22]
    else:
        arg = m.get("diff_summary") or ""
    tip = _tooltip(m)
    return (
        "<td class='step' onmouseenter='showTip(this)' onmouseleave='hideTip()'>"
        f"<div class='bar' style='background:{color}'></div>"
        f"<span class='gid' style='background:{color}'>{html.escape(gid_label)}</span>"
        f"<span class='badge kind'>{html.escape(kind_label)}</span> "
        f"<span class='tn'>{html.escape(act_name)}</span> "
        f"<span class='tv'>{html.escape(arg)}</span> "
        f"<span class='badge len'>{m['response_length']}</span>"
        f"<div class='tipdata' style='display:none'>{tip}</div>"
        "</td>"
    )


def _col_header(sample_index: int, m0: dict[str, Any]) -> str:
    solved_cls = "yes" if m0["solved"] > 0 else "no"
    solved_txt = "✓" if m0["solved"] > 0 else "✗"
    off = f"<span class='badge off'>off×{m0['offload_count']}</span>" if m0["offload_count"] else ""
    return (
        f"<th class='col-head'>sample {sample_index} "
        f"<span class='badge {solved_cls}'>{solved_txt}</span>"
        f"<span class='badge'>R={m0['reward']:.2f}</span>{off}</th>"
    )


def render_html(
    *,
    instance_id: str,
    rollout_id: int,
    rollout_path: str,
    trajs: list[tuple[int, list[dict[str, Any]]]],
    diff_meta: dict[str, dict[str, Any]],
    n_matched_steps: int,
    n_total_steps: int,
    grouping_mode: str,
    similarity_threshold: float,
    file_jaccard_threshold: float,
    similarity_mode: str,
    action_signature_threshold: float,
) -> str:
    group_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, steps in trajs:
        for s in steps:
            if s["row_id"] in diff_meta:
                group_buckets[s["group_label"]].append(s)

    legend_parts = []
    ordered = sorted(
        group_buckets.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    for label, members in ordered:
        meta = diff_meta[members[0]["row_id"]]
        color = meta["color"]
        summary = meta["summary"]
        kind = meta["kind"]
        legend_parts.append(
            f"<span class='chip' style='background:{color}' title='{html.escape(summary)}'>"
            f"{html.escape(label)} {html.escape(kind)} ×{len(members)}</span>"
        )
    legend = " ".join(legend_parts) or "<span class='muted'>无成组 git diff</span>"

    by_turn: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    max_turn = 0
    for sample_idx, steps in trajs:
        for m in steps:
            by_turn[m["turn_index"]][sample_idx] = m
            max_turn = max(max_turn, m["turn_index"])
    sample_indices = [idx for idx, _ in trajs]

    head_cells = []
    for idx in sample_indices:
        m0 = by_turn.get(0, {}).get(idx)
        if m0 is not None:
            head_cells.append(_col_header(idx, m0))
        else:
            head_cells.append(f"<th class='col-head'>sample {idx}</th>")
    head_row = f"<tr><th class='col-head tcol'>turn</th>{''.join(head_cells)}</tr>"

    body_rows = []
    for t in range(max_turn + 1):
        cells = []
        any_cell = False
        for idx in sample_indices:
            m = by_turn[t].get(idx)
            if m is not None:
                any_cell = True
            cells.append(_step_cell(m, diff_meta))
        if not any_cell:
            continue
        body_rows.append(f"<tr><td class='tlabel'>t{t}</td>{''.join(cells)}</tr>")
    table = f"<table class='grid'>{head_row}{''.join(body_rows)}</table>"

    match_pct = (n_matched_steps / n_total_steps * 100) if n_total_steps else 0
    return (
        "<!doctype html>\n"
        f"<html lang='zh'><head><meta charset='utf-8'>"
        f"<title>GiGPO git diff fuzzy {html.escape(instance_id)} · rollout {rollout_id}</title>"
        f"<style>{CSS}</style></head><body>\n"
        "<header>\n"
        f"  <h1>GiGPO git diff 模糊入组 — {html.escape(instance_id)}（{len(trajs)} 条 traj 并排）</h1>\n"
        f"  <div class='sub'>rollout {rollout_id} · {rollout_path} · {len(trajs)} 条 traj · "
        f"成组 step {n_matched_steps}/{n_total_steps} ({match_pct:.0f}%) · "
        f"group={html.escape(grouping_mode)} · mode={html.escape(similarity_mode)} · sim≥{similarity_threshold:.2f} · file≥{file_jaccard_threshold:.2f} · action≥{action_signature_threshold:.2f}</div>\n"
        f"  <div class='legend'>{legend}</div>\n"
        "</header>\n"
        "<main>\n"
        "  <p style='color:#8b949e;font-size:10px;margin:4px 0 8px'>"
        "每行 = 同一 turn 索引；每列 = 一条 sibling traj。git-diff 模式下按 diff 完全相同或相似分组；action-signature 模式下按 anchor/tool/files/参数签名分组。鼠标悬停看详情。</p>\n"
        f"{table}\n"
        "</main>\n"
        "<div id='tooltip' class='tooltip'></div>\n"
        "<script>\n"
        "function showTip(el) {\n"
        "  const data = el.querySelector('.tipdata');\n"
        "  if (!data) return;\n"
        "  const tip = document.getElementById('tooltip');\n"
        "  tip.innerHTML = data.innerHTML;\n"
        "  tip.style.display = 'block';\n"
        "  const r = el.getBoundingClientRect();\n"
        "  let left = r.right + 8;\n"
        "  if (left + 520 > window.innerWidth) left = Math.max(8, r.left - 528);\n"
        "  let top = r.top + 4;\n"
        "  if (top + 520 > window.innerHeight) top = Math.max(8, window.innerHeight - 530);\n"
        "  tip.style.left = left + 'px';\n"
        "  tip.style.top = top + 'px';\n"
        "}\n"
        "function hideTip() { document.getElementById('tooltip').style.display = 'none'; }\n"
        "</script>\n"
        "</body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--instance", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--grouping-mode", choices=["git-diff", "action-signature"], default="git-diff")
    p.add_argument("--similarity-mode", choices=["weighted", "files-only"], default="weighted")
    p.add_argument("--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD)
    p.add_argument("--file-jaccard-threshold", type=float, default=FILE_JACCARD_THRESHOLD)
    p.add_argument("--action-signature-threshold", type=float, default=ACTION_SIGNATURE_THRESHOLD)
    p.add_argument("--hf-checkpoint", default="/workspace/models/pyromind/PyroDash-4B-SFT-07313")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True, local_files_only=True)
    data = torch.load(args.rollout, map_location="cpu", weights_only=False)
    rollout_id = int(data.get("rollout_id") or args.rollout.stem.split("_")[1])
    samples = list(data.get("samples") or [])
    target = [
        s
        for s in samples
        if str((s.get("metadata") or {}).get("instance_id") or s.get("label") or "") == args.instance
    ]
    if not target:
        print(f"ERROR: no samples with instance_id={args.instance}", file=sys.stderr)
        return 1
    rows = build_rows(target, tok=tok)
    diff_meta = assign_groups(
        rows,
        grouping_mode=args.grouping_mode,
        similarity_threshold=args.similarity_threshold,
        file_jaccard_threshold=args.file_jaccard_threshold,
        similarity_mode=args.similarity_mode,
        action_signature_threshold=args.action_signature_threshold,
    )
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sample[r["sample_index"]].append(r)
    for k in by_sample:
        by_sample[k].sort(key=lambda m: m["turn_index"])
    trajs = sorted(by_sample.items())
    n_matched = sum(1 for r in rows if r["row_id"] in diff_meta)
    html_str = render_html(
        instance_id=args.instance,
        rollout_id=rollout_id,
        rollout_path=str(args.rollout),
        trajs=trajs,
        diff_meta=diff_meta,
        n_matched_steps=n_matched,
        n_total_steps=len(rows),
        grouping_mode=args.grouping_mode,
        similarity_threshold=args.similarity_threshold,
        file_jaccard_threshold=args.file_jaccard_threshold,
        similarity_mode=args.similarity_mode,
        action_signature_threshold=args.action_signature_threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_str, encoding="utf-8")
    digest = hashlib.md5(args.out.read_bytes()).hexdigest()[:8]
    group_labels = {(m["label"], m["kind"]) for m in diff_meta.values()}
    exact = sum(1 for _, kind in group_labels if kind in {"exact", "action-exact"})
    similar = sum(1 for _, kind in group_labels if kind in {"similar", "action-similar"})
    print(
        f"wrote {args.out}: {len(trajs)} trajs, {len(rows)} steps, "
        f"{n_matched} grouped ({n_matched / max(len(rows), 1) * 100:.0f}%), exact_keys={exact}, similar_keys={similar}, md5={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
