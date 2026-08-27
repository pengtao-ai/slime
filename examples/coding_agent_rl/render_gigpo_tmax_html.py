#!/usr/bin/env python3
"""Render GiGPO HTML for a Tmax (or ScaleSWE) instance from a rollout dump.

Intra-traj segments: git-diff (scaleswe) or Edit/Write (tmax).
Inter-traj groups: intent + tools. Per-turn: dump r_i and GiGPO G / G_t / A_E / A_S / A_I / A.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from examples.coding_agent_rl import gigpo

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
SEG_COLORS = ["#3fb950", "#1f6feb", "#d29922", "#f85149", "#a371f7", "#39c5cf", "#ff7eb6", "#8b949e"]


def _is_sft(sample: dict[str, Any]) -> bool:
    tmd = sample.get("train_metadata") or {}
    return isinstance(tmd, dict) and tmd.get("objective") == "sft"


def _dedup(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[Any, Any], dict[str, Any]] = {}
    for s in samples:
        if _is_sft(s) or s.get("remove_sample"):
            continue
        key = (s.get("group_index"), s.get("index"))
        cur = best.get(key)
        if cur is None or len(s.get("tokens") or []) > len(cur.get("tokens") or []):
            best[key] = s
    return sorted(best.values(), key=lambda x: (int(x.get("group_index") or 0), int(x.get("index") or 0)))


def _tool_calls_from_turn_cost(tc: dict[str, Any]) -> list[dict[str, Any]]:
    raw = tc.get("teacher_tool_calls") or []
    if not raw:
        msg = tc.get("sft_assistant_message") or {}
        raw = msg.get("tool_calls") or []
    if not raw:
        return []
    return gigpo.parse_manager_tool_calls({"tool_calls": raw})


def _records_from_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    md = sample.get("metadata") or {}
    costs = [x for x in (md.get("turn_costs") or []) if isinstance(x, dict)]
    diffs = [x for x in (md.get("turn_git_diffs") or []) if isinstance(x, dict)]
    n = max(len(costs), len(diffs), len(md.get("turn_rewards") or []), 1)
    out: list[dict[str, Any]] = []
    for i in range(n):
        git = diffs[i] if i < len(diffs) else {}
        tc = costs[i] if i < len(costs) else {}
        calls = git.get("tool_calls") if git.get("tool_calls") else _tool_calls_from_turn_cost(tc)
        rec = {
            "turn_index": int(git.get("turn_index") if git.get("turn_index") is not None else i),
            "git_diff": str(git.get("git_diff") or ""),
            "tool_calls": calls,
        }
        out.append(rec)
    return out


def _as_ns(sample: dict[str, Any], proto: str, instance_id: str) -> SimpleNamespace:
    md = dict(sample.get("metadata") or {})
    tmd = dict(sample.get("train_metadata") or {})
    turns = gigpo.compact_turns(proto, _records_from_sample(sample))
    tmd["objective"] = "grpo"
    tmd["protocol"] = proto
    tmd["instance_id"] = instance_id
    tmd["gigpo_turns"] = turns
    md["protocol"] = proto
    md["instance_id"] = instance_id
    md["gigpo_turns"] = turns
    return SimpleNamespace(
        reward=float(sample.get("reward") or 0.0),
        index=sample.get("index"),
        group_index=sample.get("group_index"),
        remove_sample=False,
        metadata=md,
        train_metadata=tmd,
    )


def _label_groups(trajs: list[dict[str, Any]]) -> None:
    uid_members: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for ti, traj in enumerate(trajs):
        for ri, row in enumerate(traj["turns"]):
            uid_members[str(row["step_uid"])].append((ti, ri))
    ranked = sorted(uid_members.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    uid_to_t: dict[str, tuple[str, str, int]] = {}
    gid = 0
    for uid, members in ranked:
        n_traj = len({ti for ti, _ in members})
        size = len(members)
        if size < 2:
            uid_to_t[uid] = ("·", "#3a3f47", n_traj)
            continue
        uid_to_t[uid] = (f"T{gid}", GROUP_COLORS[gid % len(GROUP_COLORS)], n_traj)
        gid += 1
    for traj in trajs:
        for row in traj["turns"]:
            label, color, n_traj = uid_to_t[str(row["step_uid"])]
            row["t_label"] = label
            row["t_color"] = color
            row["group_n_traj"] = n_traj
            row["group_size"] = len(uid_members[str(row["step_uid"])])
            row["s_label"] = f"S{int(row['seg'])}"
            row["s_color"] = SEG_COLORS[int(row["seg"]) % len(SEG_COLORS)]


def _signed(v: float) -> str:
    cls = "pos" if v > 1e-9 else ("neg" if v < -1e-9 else "zero")
    return f"<span class='{cls}'>{v:+.4f}</span>"


def _gid(kind: str, label: str, color: str, *, sk: str = "", t: str = "") -> str:
    esc = html_lib.escape(label)
    if kind == "t":
        attrs = f" data-kind='t' data-t='{html_lib.escape(t or label, quote=True)}'"
    else:
        attrs = f" data-kind='s' data-sk='{html_lib.escape(sk, quote=True)}'"
    return f"<span class='gid'{attrs} style='background:{color}'>{esc}</span>"


def render_html(
    instance_id: str,
    trajs: list[dict[str, Any]],
    *,
    rollout_id: int,
    gamma: float,
    step_w: float,
    proto: str = "tmax",
) -> str:
    n_traj = len(trajs)
    legend_t: dict[str, dict[str, Any]] = {}
    columns = []
    for traj in trajs:
        cards = []
        last_seg = None
        traj_i = int(traj["index"])
        for row in traj["turns"]:
            if row["t_label"] != "·" and row["t_label"] not in legend_t:
                legend_t[row["t_label"]] = row
            first_seg = last_seg is None
            seg_break = last_seg is not None and int(row["seg"]) != last_seg
            last_seg = int(row["seg"])
            brk = " seg-break" if seg_break else ""
            r_turn = row.get("r_turn")
            r_turn_s = f"{float(r_turn):.4f}" if r_turn is not None else "—"
            sk = f"{traj_i}:{row['s_label']}"
            tlab = row["t_label"]
            files = str(row.get("files") or "")
            git_html = ""
            if gigpo.is_tmax(proto):
                git_html = (
                    f"<div class='git'>修改：{html_lib.escape(files) if files else '尚未修改'}</div>"
                )
            else:
                files = files or ("<empty diff>" if row.get("empty_diff") else "")
                git_html = f"<div class='git'>git：{html_lib.escape(files)}</div>"
                if seg_break or first_seg:
                    body = str(row.get("git_diff") or "")
                    git_html += (
                        f"<pre class='diff'>{html_lib.escape(body)}</pre>"
                        if body.strip()
                        else "<div class='empty-diff'>&lt;empty diff&gt;</div>"
                    )
            cards.append(
                f"<section class='turn{brk}' data-t='{html_lib.escape(tlab, quote=True)}' data-sk='{html_lib.escape(sk, quote=True)}'>"
                f"<div class='turn-title'>t{row['turn']}"
                f"{_gid('s', row['s_label'], row['s_color'], sk=sk)}"
                f"{_gid('t', tlab, row['t_color'], t=tlab)}"
                f"</div>"
                f"<div class='intent'>意图：{html_lib.escape(row['intent'])}"
                f"<br>工具：{html_lib.escape(row['tools_str'])}"
                f"<br>轨迹间：{row['group_n_traj']}/{n_traj} 条 · size={row['group_size']}</div>"
                f"{git_html}"
                f"<div class='reward' style='border-left:4px solid {row['t_color']}'>"
                f"<span class='chip'>r_i={r_turn_s}</span>"
                f"<span class='chip'>r_imm={row['r_imm']:.4f}</span>"
                f"<span class='chip'>G={row['G']:.4f}</span>"
                f"<span class='chip'>G_t={float(row.get('G_turn', row['G'])):.4f}</span>"
                f"<span class='chip'>A_E={row['A_E']:+.4f}</span>"
                f"<span class='chip'>A_S={row['A_S']:+.4f}</span>"
                f"<span class='chip'>A_I={float(row.get('A_I', 0.0)):+.4f}</span>"
                f"<span class='chip A'>A={row['A']:+.4f}</span>"
                f"</div></section>"
            )
        mark = "✓" if traj["solved"] else "✗"
        n_seg = len({int(r["seg"]) for r in traj["turns"]})
        title = (
            f"traj {traj['index']} · {mark} · R={traj['episode_reward']:.4f} · "
            f"{len(traj['turns'])} turns · {n_seg} 段(S#)"
        )
        columns.append(f"<article class='traj'><div class='traj-title'>{html_lib.escape(title)}</div>{''.join(cards)}</article>")

    t_parts = []
    for label, row in sorted(legend_t.items(), key=lambda kv: int(kv[0][1:]) if kv[0][1:].isdigit() else 999):
        cap = f"{row['intent']} · {row['tools_str']}"
        t_parts.append(
            f"<span class='chip' data-t='{html_lib.escape(label, quote=True)}' style='background:{row['t_color']}'>"
            f"{html_lib.escape(label)} ×{row['group_size']} · {row['group_n_traj']}traj · {html_lib.escape(cap[:90])}</span>"
        )
    legend = " ".join(t_parts) or "<span class='muted'>无跨轨迹 T 组</span>"
    width = max(360, int(2800 * n_traj / 8)) if n_traj else 360

    # compact table of all turns
    th = "".join(f"<th>traj {t['index']}<br>R={t['episode_reward']:.3f} {'✓' if t['solved'] else '✗'}</th>" for t in trajs)
    max_t = max((len(t["turns"]) for t in trajs), default=0)
    body_rows = []
    for ti in range(max_t):
        cells = [f"<td class='tlabel'>t{ti}</td>"]
        for traj in trajs:
            row = next((r for r in traj["turns"] if int(r["turn"]) == ti), None)
            if row is None:
                cells.append("<td class='empty'></td>")
                continue
            sk = f"{int(traj['index'])}:{row['s_label']}"
            tlab = row["t_label"]
            files = str(row.get("files") or "")
            file_line = (
                f"<div class='muted'>修改：{html_lib.escape(files) if files else '尚未修改'}</div>"
                if gigpo.is_tmax(proto)
                else f"<div class='muted'>git：{html_lib.escape(files)}</div>"
            )
            cells.append(
                f"<td class='cell' data-t='{html_lib.escape(tlab, quote=True)}' data-sk='{html_lib.escape(sk, quote=True)}'>"
                f"<div class='bar' style='background:{row['t_color']}'></div>"
                f"<div>{_gid('s', row['s_label'], row['s_color'], sk=sk)} "
                f"{_gid('t', tlab, row['t_color'], t=tlab)}</div>"
                f"<div class='muted'>{html_lib.escape(row['intent'])}</div>"
                f"<div class='muted'>{html_lib.escape(row['tools_str'])}</div>"
                f"{file_line}"
                f"<div>r_i={float(row['r_turn']):.3f}</div>"
                f"<div>G_t={float(row.get('G_turn', row['G'])):.3f} A_I={_signed(float(row.get('A_I', 0.0)))}</div>"
                f"<div>G={row['G']:.3f} A_E={_signed(row['A_E'])} A_S={_signed(row['A_S'])}</div>"
                f"<div><b>A={_signed(row['A'])}</b></div>"
                "</td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    proto_name = "Tmax" if gigpo.is_tmax(proto) else "ScaleSWE"
    seg_rule = (
        "Edit/Write/Bash改文件 执行前的累计修改一切开（改文件是旧组的最后一轮）"
        if gigpo.is_tmax(proto)
        else "工具执行前的累计 git diff 一切开（Edit 是旧 diff 组的最后一轮；绿横线处展开 diff）"
    )
    intent_rule = (
        "意图：实现修复 / 复现试跑 / 验证运行 / 改后阅读 / 探索定位"
        if gigpo.is_tmax(proto)
        else "意图：探索定位 · 写/改测试 · 实现修复 · 复现失败 · 验证修复 · 改后阅读"
    )
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>{html_lib.escape(instance_id)} {proto_name} GiGPO</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #0d1117; color: #c9d1d9; font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }}
header {{ position: sticky; top: 0; z-index: 2; padding: 12px 16px; background: #161b22; border-bottom: 1px solid #30363d; font-family: sans-serif; }}
header h1 {{ margin: 0 0 4px; font: 600 15px sans-serif; color: #f0f6fc; }}
header .sub {{ color: #8b949e; font: 12px sans-serif; }}
.legend {{ margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px; font-family: sans-serif; }}
.legend .chip, .chip {{ font-size: 10px; padding: 2px 7px; border-radius: 8px; color: #fff; background: #21262d; }}
.legend .chip, .gid {{ cursor: pointer; user-select: none; }}
.legend .chip:hover, .gid:hover {{ filter: brightness(1.18); }}
.turn, td.cell {{ cursor: pointer; transition: opacity .12s, box-shadow .12s; }}
body.picking .turn, body.picking td.cell {{ opacity: .22; }}
body.picking .turn.on, body.picking td.cell.on {{
  opacity: 1;
  box-shadow: inset 0 0 0 2px #f0c674;
}}
body.picking .legend .chip {{ opacity: .35; }}
body.picking .legend .chip.on {{ opacity: 1; outline: 2px solid #f0c674; }}
.grid {{ display: grid; grid-template-columns: repeat({n_traj}, minmax(300px, 1fr)); gap: 8px; min-width: {width}px; padding: 10px; align-items: start; }}
.traj {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }}
.traj-title {{ padding: 8px; background: #21262d; font: 600 12px sans-serif; color: #f0f6fc; }}
.turn {{ border-top: 1px solid #30363d; }}
.turn.seg-break {{ border-top: 3px solid #3fb950; }}
.turn-title {{ padding: 6px 8px; background: #1c2128; color: #79c0ff; font: 600 11px sans-serif; display: flex; gap: 6px; align-items: center; }}
.intent {{ padding: 7px 8px; background: #18232f; color: #f0c674; font: 12px sans-serif; }}
.reward {{ padding: 6px 8px; background: #18232f; font: 11px sans-serif; display: flex; flex-wrap: wrap; gap: 4px; }}
.reward .chip.A {{ background: #1f6feb; color: #fff; font-weight: 700; }}
.gid {{ display: inline-block; min-width: 22px; padding: 0 5px; border-radius: 6px; color: #fff; font-weight: 700; }}
.git {{ padding: 4px 8px; color: #79c0ff; font: 11px sans-serif; border-top: 1px solid #30363d; }}
pre.diff {{ margin: 0; padding: 8px; max-height: 260px; overflow: auto; background: #0d1117; white-space: pre; }}
.empty-diff {{ padding: 8px; color: #8b949e; font-style: italic; }}
.pos {{ color: #56d364; }}
.neg {{ color: #f85149; }}
.zero {{ color: #8b949e; }}
.muted {{ color: #8b949e; font-size: 11px; }}
h2 {{ font: 600 13px sans-serif; margin: 16px 12px 8px; color: #f0f6fc; }}
table.board {{ border-collapse: separate; border-spacing: 3px; margin: 0 10px 24px; }}
th {{ padding: 6px 8px; background: #21262d; border-radius: 6px; font-size: 11px; text-align: left; }}
td.tlabel {{ width: 36px; color: #8b949e; text-align: right; }}
td.cell {{ background: #161b22; border-radius: 4px; padding: 5px 6px 5px 10px; position: relative; font-size: 11px; vertical-align: top; min-width: 160px; }}
td.cell .bar {{ position: absolute; left: 0; top: 2px; bottom: 2px; width: 4px; border-radius: 4px; }}
td.empty {{ background: transparent; }}
</style></head><body>
<header>
  <h1>{html_lib.escape(instance_id)} · {proto_name} GiGPO</h1>
  <div class="sub">rollout {rollout_id} · {n_traj} sibling · γ={gamma} · w={step_w} ·
    轨迹内 <b>S#</b> = {seg_rule} ·
    段内 G_t = G_seg·γ^{{距段末}}，A_I = G_t − mean(G_t | 同 S#)，越靠近段末 A_I 越大 ·
    轨迹间 <b>T#</b> = 相同「意图 + 工具」· {intent_rule} ·
    A = A_E + w·(A_S + A_I) · r_i 是 dump 里的 turn_rewards，r_imm 只在最后一段 = episode R<br>
    点击 <b>T#</b> 高亮跨轨迹同组，点击 <b>S#</b> 高亮本轨迹同段；再点一次取消</div>
  <div class="legend"><span class="muted">轨迹间 T 组：</span>{legend}</div>
</header>
<h2>每轮卡片（绿横线 = 新的轨迹内 segment）</h2>
<main class="grid">{''.join(columns)}</main>
<h2>对齐表</h2>
<table class="board">
<tr><th></th>{th}</tr>
{''.join(body_rows)}
</table>
<script>
(function () {{
  let cur = null;
  function clear() {{
    document.body.classList.remove('picking');
    document.querySelectorAll('.on').forEach(el => el.classList.remove('on'));
    cur = null;
  }}
  function pick(kind, key) {{
    if (!key || key === '·') return;
    if (cur && cur.kind === kind && cur.key === key) {{ clear(); return; }}
    clear();
    cur = {{kind, key}};
    document.body.classList.add('picking');
    document.querySelectorAll(kind === 't' ? '[data-t]' : '[data-sk]').forEach(el => {{
      const val = kind === 't' ? el.dataset.t : el.dataset.sk;
      if (val === key) el.classList.add('on');
    }});
  }}
  document.addEventListener('click', (e) => {{
    const gid = e.target.closest('.gid[data-kind]');
    if (gid) {{
      const kind = gid.dataset.kind;
      pick(kind, kind === 't' ? gid.dataset.t : gid.dataset.sk);
      e.stopPropagation();
      return;
    }}
    const chip = e.target.closest('.legend .chip[data-t]');
    if (chip) {{ pick('t', chip.dataset.t); return; }}
    const unit = e.target.closest('.turn[data-t], td.cell[data-t]');
    if (unit) {{
      if (unit.dataset.t && unit.dataset.t !== '·') pick('t', unit.dataset.t);
      else pick('s', unit.dataset.sk);
      return;
    }}
    clear();
  }});
}})();
</script>
</body></html>"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/workspace/work/spt/slime/runs/agent_offload_pyrodash4b_phase2_sft03_prob_20260824_113854"),
    )
    p.add_argument("--rollout", type=int, default=0)
    p.add_argument("--instance", default="task_001793_2abe082a")
    p.add_argument("--gamma", type=float, default=gigpo.DEFAULT_GAMMA)
    p.add_argument("--step-w", type=float, default=gigpo.DEFAULT_STEP_W)
    args = p.parse_args()

    dump = args.run_dir / "rollout_dumps" / f"rollout_{args.rollout}.pt"
    data = torch.load(dump, map_location="cpu", weights_only=False)
    raw = [
        s
        for s in _dedup(data["samples"])
        if str((s.get("metadata") or {}).get("instance_id") or s.get("label") or "") == args.instance
    ]
    if not raw:
        print(f"no GRPO samples for {args.instance}")
        return 1
    proto = str((raw[0].get("metadata") or {}).get("protocol") or "tmax")
    wrapped = [_as_ns(s, proto, args.instance) for s in raw]
    gigpo.assign_gigpo_to_samples(wrapped, gamma=args.gamma, step_w=args.step_w)

    trajs = []
    for s in wrapped:
        md = s.metadata
        rows = list(md.get("gigpo_step_rows") or [])
        last_seg = max((int(r["seg"]) for r in rows), default=0)
        turn_rewards = list((md.get("turn_rewards") or []))
        tmd = s.train_metadata or {}
        turns_meta = list(tmd.get("gigpo_turns") or [])
        orig = next((x for x in raw if x.get("index") == s.index), None)
        recs = _records_from_sample(orig) if orig else []
        for i, r in enumerate(rows):
            r["r_imm"] = float(s.reward) if int(r["seg"]) == last_seg else 0.0
            r["r_turn"] = float(turn_rewards[i]) if i < len(turn_rewards) else None
            src = turns_meta[i] if i < len(turns_meta) else {}
            r["files"] = src.get("files") or ""
            if not gigpo.is_tmax(proto):
                r["empty_diff"] = bool(src.get("empty_diff"))
                r["git_diff"] = str((recs[i].get("git_diff") if i < len(recs) else "") or "")
        trajs.append(
            {
                "index": int(s.index or 0),
                "solved": float(md.get("solved") or 0.0) > 0,
                "episode_reward": float(s.reward),
                "turns": rows,
            }
        )
    _label_groups(trajs)
    html = render_html(
        args.instance,
        trajs,
        rollout_id=int(data.get("rollout_id") or args.rollout),
        gamma=args.gamma,
        step_w=args.step_w,
        proto=proto,
    )
    tag = "tmax" if gigpo.is_tmax(proto) else "scaleswe"
    out = args.run_dir / f"gigpo_{tag}_{args.instance}_turn_rewards.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    n_cross = sum(1 for t in trajs for r in t["turns"] if r["t_label"] != "·")
    n_total = sum(len(t["turns"]) for t in trajs)
    n_t = len({r["t_label"] for t in trajs for r in t["turns"] if r["t_label"] != "·"})
    print(f"{args.instance}: {len(trajs)} trajs, {n_cross}/{n_total} turns in T-groups, {n_t} T labels")
    print("episode R:", [(t["index"], round(t["episode_reward"], 4), "ok" if t["solved"] else "fail") for t in trajs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
