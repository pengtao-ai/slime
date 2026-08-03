#!/usr/bin/env python3
"""Render all trajectories of one instance as a side-by-side HTML table.

Each column = one sibling trajectory. Each row = turn index (aligned across
trajs). Each cell = one SLM step, colored by its step-group. Same color =
same anchor_obs = matched (will share step-level advantage).

Usage:
  PYTHONPATH=. python examples/coding_agent_rl/render_gigpo_instance_grid_html.py \\
    --rollout runs/.../rollout_dumps/rollout_0.pt \\
    --instance adamtheturtle_sybil-extras_pr296 \\
    --out    runs/.../gigpo_instance_grid_sybil_pr296.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from examples.coding_agent_rl.analyze_gigpo_groups_from_rollout import (
    _guess_workdir,
    _parse_tool_calls,
    _split_messages,
    _TOOL_RESP,
    extract_turns_from_decoded,
)

GROUP_COLORS = [
    "#1f6feb", "#2ea043", "#d2a8ff", "#e3b341", "#f85149", "#79c0ff",
    "#56d364", "#db61dd", "#ffa657", "#a371f7", "#39c5cf", "#ff7eb6",
]


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
            rows.append(
                {
                    "sample_index": s.get("index"),
                    "turn_index": t["turn_index"],
                    "anchor_obs": t["anchor_obs"],
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
    return rows


def _tool_of(anchor: str) -> str:
    try:
        return (json.loads(anchor).get("obs") or ["?"])[0]
    except (json.JSONDecodeError, TypeError):
        return "?"


def assign_group_ids(rows: list[dict[str, Any]]) -> dict[str, int]:
    by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_anchor[r["anchor_obs"]].append(r)
    anchor_to_gid: dict[str, int] = {}
    gid = 0
    for anchor, members in sorted(by_anchor.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(members) >= 2:
            anchor_to_gid[anchor] = gid
            gid += 1
    return anchor_to_gid


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
th.col-head .b { font-size: 9px; margin-left: 4px; }
th.tcol { width: 44px; background: #161b22; color: #8b949e; font-weight: 600; font-size: 10px; }
td.tlabel { padding: 4px 6px; color: #8b949e; font-size: 10px; text-align: right; background: #161b22; width: 44px; }
td.step { padding: 4px 6px 4px 10px; background: #161b22; border-radius: 4px; position: relative; min-height: 22px; }
td.step.empty { background: transparent; }
td.step .bar { position: absolute; left: 0; top: 2px; bottom: 2px; width: 4px; border-radius: 4px; }
td.step .tn { color: #79c0ff; font-weight: 600; }
td.step .tv { color: #d2a8ff; }
td.step .muted { color: #6e7681; }
td.step .gid { display: inline-block; min-width: 18px; padding: 0 4px; border-radius: 6px; font-size: 9px; color: #fff; margin-right: 3px; }
.badge { font-size: 9px; padding: 0 4px; border-radius: 6px; background: #21262d; color: #c9d1d9; }
.badge.yes { background: #1b3a23; color: #56d364; }
.badge.no { background: #3a1b1b; color: #f85149; }
.badge.off { background: #3b2e10; color: #e3b341; }
.badge.len { background: #21262d; color: #8b949e; }
.tooltip { position: fixed; z-index: 50; background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  padding: 10px 12px; width: 520px; max-width: 90vw; max-height: 70vh; overflow: auto; font-size: 12px; line-height: 1.5;
  color: #c9d1d9; white-space: pre-wrap; word-break: break-word; pointer-events: none; display: none;
  box-shadow: 0 6px 20px rgba(0,0,0,0.7); }
"""


def _tooltip(m: dict[str, Any]) -> str:
    prev = m.get("prev_tool_call") or {}
    prev_name = prev.get("name") if prev else "—"
    prev_inp = prev.get("input") if prev else {}
    prev_summary = json.dumps(prev_inp, ensure_ascii=False)[:200] if prev_inp else ""
    actions = m.get("this_turn_tool_calls") or []
    acts = ", ".join(
        f"{c.get('name')}({json.dumps(c.get('input') or {}, ensure_ascii=False)[:120]})" for c in actions
    )
    # Plain text with newlines; the tooltip container uses white-space: pre-wrap.
    text = (
        f"sample {m['sample_index']} · turn {m['turn_index']}\n"
        f"prev obs: {prev_name} {prev_summary}\n"
        f"prev resp:\n{(m.get('prev_tool_response_preview') or '')[:800]}\n"
        f"this action: {acts}"
    )
    # Escape fully so no nested tags can break the tipdata container.
    return html.escape(text)


def _step_cell(m: dict[str, Any] | None, anchor_to_gid: dict[str, int]) -> str:
    if m is None:
        return "<td class='step empty'></td>"
    gid = anchor_to_gid.get(m["anchor_obs"], -1)
    color = GROUP_COLORS[gid % len(GROUP_COLORS)] if gid >= 0 else "#3a3f47"
    gid_label = f"G{gid}" if gid >= 0 else "·"
    acts = m.get("this_turn_tool_calls") or []
    act_name = acts[0].get("name") if acts else "—"
    inp = (acts[0].get("input") if acts else {}) or {}
    if "file_path" in inp:
        arg = str(inp["file_path"]).split("/")[-1][:22]
    elif "command" in inp:
        arg = str(inp["command"])[:22]
    else:
        arg = ""
    tip = _tooltip(m)
    return (
        "<td class='step' onmouseenter='showTip(this)' onmouseleave='hideTip()'>"
        f"<div class='bar' style='background:{color}'></div>"
        f"<span class='gid' style='background:{color}'>{html.escape(gid_label)}</span>"
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
    anchor_to_gid: dict[str, int],
    n_matched_steps: int,
    n_total_steps: int,
) -> str:
    by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, steps in trajs:
        for s in steps:
            by_anchor[s["anchor_obs"]].append(s)
    legend_parts = []
    for anchor, members in sorted(by_anchor.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            continue
        gid = anchor_to_gid[anchor]
        tool = _tool_of(anchor)
        color = GROUP_COLORS[gid % len(GROUP_COLORS)]
        legend_parts.append(
            f"<span class='chip' style='background:{color}' title='{html.escape(anchor[:120])}'>"
            f"G{gid} {html.escape(tool)} ×{len(members)}</span>"
        )
    legend = " ".join(legend_parts) or "<span class='muted'>无成组</span>"

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
            cells.append(_step_cell(m, anchor_to_gid))
        if not any_cell:
            continue
        body_rows.append(f"<tr><td class='tlabel'>t{t}</td>{''.join(cells)}</tr>")
    table = f"<table class='grid'>{head_row}{''.join(body_rows)}</table>"

    match_pct = (n_matched_steps / n_total_steps * 100) if n_total_steps else 0
    return (
        "<!doctype html>\n"
        f"<html lang='zh'><head><meta charset='utf-8'>"
        f"<title>GiGPO {html.escape(instance_id)} · rollout {rollout_id}</title>"
        f"<style>{CSS}</style></head><body>\n"
        "<header>\n"
        f"  <h1>GiGPO 入组情况 — {html.escape(instance_id)}（{len(trajs)} 条 traj 并排）</h1>\n"
        f"  <div class='sub'>rollout {rollout_id} · {rollout_path} · {len(trajs)} 条 traj · "
        f"成组 step {n_matched_steps}/{n_total_steps} ({match_pct:.0f}%) · 同色 = 同 anchor = 互相算 A_S</div>\n"
        f"  <div class='legend'>{legend}</div>\n"
        "</header>\n"
        "<main>\n"
        "  <p style='color:#8b949e;font-size:10px;margin:4px 0 8px'>"
        "每行 = 同一 turn 索引；每列 = 一条 sibling traj。同色块 = 同 anchor_obs（会互相算 step advantage）；灰 = singleton。鼠标悬停看详情。</p>\n"
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
    anchor_to_gid = assign_group_ids(rows)
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sample[r["sample_index"]].append(r)
    for k in by_sample:
        by_sample[k].sort(key=lambda m: m["turn_index"])
    trajs = sorted(by_sample.items())
    n_matched = sum(1 for r in rows if r["anchor_obs"] in anchor_to_gid)
    html_str = render_html(
        instance_id=args.instance,
        rollout_id=rollout_id,
        rollout_path=str(args.rollout),
        trajs=trajs,
        anchor_to_gid=anchor_to_gid,
        n_matched_steps=n_matched,
        n_total_steps=len(rows),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_str, encoding="utf-8")
    print(
        f"wrote {args.out}: {len(trajs)} trajs, {len(rows)} steps, "
        f"{n_matched} matched ({n_matched / max(len(rows), 1) * 100:.0f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
