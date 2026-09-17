#!/usr/bin/env python3
"""GiGPO advantages for traj_compare 8 cases (DeepSeek / SFT / Qwen siblings).

  A = A_E + A_S + A_I
  A_E = R − mean(R | 3 models)                 # one value per traj
  G_t = R · γ^{n−1−t}                          # discounted return
  T#  = 意图 · 工具
  A_S = G_t − mean(G | same T# across siblings)  # per-step, not broadcast
  S#  = edit-segment proxy (no git dump: empty until Edit/Write, then +1)
  A_I = G_t − mean(G | same S# on this traj)
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


GAMMA = 0.95
MODEL_ORDER = ["DeepSeek", "SFT", "Qwen"]

INTENT_FROM_TAG = {
    "edit": "实现/修改",
    "test": "运行测试/验证",
    "read": "阅读代码",
    "search": "探索/定位",
    "probe": "探索/定位",
    "git": "探索/定位",
    "env": "环境准备",
    "other": "其他",
}


def _primary_tag(tags: list[str] | None) -> str:
    for t in tags or []:
        if t in INTENT_FROM_TAG:
            return t
    return "other"


def _tool_family(turn: dict[str, Any]) -> str:
    cmds = [str(c) for c in (turn.get("cmds") or [])]
    blob = " ; ".join(cmds) if cmds else str(turn.get("tool_calls") or "")
    low = blob.lower()
    if not blob.strip():
        return "无 tool"
    if re.search(r"\b(pytest|unittest|tox|runtests\.py)\b", low):
        return "Bash:pytest"
    if re.search(r"\b(grep|rg|ag)\b", low) or "xargs grep" in low:
        return "Bash:grep"
    if re.search(r"\bfind\b", low):
        return "Bash:find"
    if re.search(r"\bgit\b", low):
        return "Bash:git"
    if re.search(r"\b(pip|apt|conda)\b", low):
        return "Bash:env"
    if re.search(r"\b(sed\s+-n|cat\s+-n|\bcat\b|\bnl\b|\bhead\b|\btail\b)\b", low):
        # heredoc write/edit often uses cat >
        if re.search(r"cat\s*>|cat\s*<<|tee\s+", low) or "<<" in blob:
            if re.search(r"test_|tests/|/tmp/.*\.py", low):
                return "Write:test"
            return "Write/Edit"
        return "Bash:read"
    if re.search(r"str_replace|search_replace|apply_patch|edit_file", low):
        return "Edit"
    if re.search(r"\bpython(?:3)?\b", low):
        return "Bash:python"
    if re.search(r"\b(ls|pwd|mkdir|chmod|cp|mv|rm)\b", low):
        return "Bash:fs"
    tags = turn.get("tags") or []
    if "edit" in tags:
        return "Edit"
    if "test" in tags:
        return "Bash:pytest"
    if "read" in tags:
        return "Bash:read"
    if "search" in tags or "probe" in tags:
        return "Bash:search"
    return "Bash:other"


def group_key(turn: dict[str, Any]) -> tuple[str, str, str]:
    tag = _primary_tag(turn.get("tags"))
    intent = INTENT_FROM_TAG.get(tag, "其他")
    tool = _tool_family(turn)
    if tool.startswith("Write") and intent == "其他":
        intent = "实现/修改"
    if tool == "Bash:pytest":
        intent = "运行测试/验证"
    key = f"{intent} · {tool}"
    return key, intent, tool


def _is_edit_turn(turn: dict[str, Any], tool: str) -> bool:
    tags = turn.get("tags") or []
    if "edit" in tags:
        return True
    return tool in ("Edit", "Write/Edit", "Write:test")


def compute_case_group_advantages(case: dict[str, Any], *, gamma: float = GAMMA) -> dict[str, Any]:
    models: dict[str, Any] = {}
    rewards: dict[str, float] = {}
    for name in MODEL_ORDER:
        m = case["models"].get(name)
        if not m:
            continue
        R = 1.0 if m.get("solved") else 0.0
        rewards[name] = R
        turns_out = []
        n = int(m.get("n_turns") or len(m.get("llm_turn_rewards") or []))
        seg = 0  # S# before this turn; Edit closes the old segment
        for tr in m.get("llm_turn_rewards") or []:
            gkey, intent, tool = group_key(tr)
            i = int(tr.get("turn", 0))
            G = R * (gamma ** max(0, n - 1 - i))
            s_label = "<empty>" if seg == 0 else f"S{seg}"
            turns_out.append(
                {
                    "turn": i,
                    "group": gkey,
                    "T": gkey,
                    "S": s_label,
                    "intent": intent,
                    "tool": tool,
                    "G": G,
                    "cmds": tr.get("cmds") or [],
                    "tags": tr.get("tags") or [],
                    "offloaded": tr.get("offloaded"),
                    "spin": tr.get("spin") or [],
                    "llm_score": tr.get("score"),
                    "llm_residual": tr.get("residual"),
                    "llm_advantage": tr.get("advantage"),
                    "reason": tr.get("reason"),
                }
            )
            if _is_edit_turn(tr, tool):
                seg += 1
        models[name] = {
            "solved": bool(m.get("solved")),
            "R": R,
            "n_turns": n,
            "exit_status": m.get("exit_status"),
            "mean_r": m.get("mean_r"),
            "turns": turns_out,
        }

    if not models:
        return {"inst": case["inst"], "note": case.get("note"), "models": {}, "legend": []}

    mean_R = statistics.mean(rewards.values())
    for name, md in models.items():
        md["A_E"] = md["R"] - mean_R

    by_T: dict[str, list[float]] = defaultdict(list)
    for md in models.values():
        for tr in md["turns"]:
            by_T[tr["group"]].append(float(tr["G"]))
    t_bar = {k: statistics.mean(v) for k, v in by_T.items()}

    for name, md in models.items():
        by_S: dict[str, list[float]] = defaultdict(list)
        for tr in md["turns"]:
            by_S[tr["S"]].append(float(tr["G"]))
        s_bar = {k: statistics.mean(v) for k, v in by_S.items()}
        ae = md["A_E"]
        for tr in md["turns"]:
            a_s = float(tr["G"]) - t_bar[tr["group"]]
            a_i = float(tr["G"]) - s_bar[tr["S"]]
            tr["A_E"] = ae
            tr["A_S"] = a_s
            tr["A_I"] = a_i
            tr["A"] = ae + a_s + a_i

    counts: dict[str, int] = defaultdict(int)
    trajs_per: dict[str, set[str]] = defaultdict(set)
    for name, md in models.items():
        for tr in md["turns"]:
            counts[tr["group"]] += 1
            trajs_per[tr["group"]].add(name)
    legend = [
        {
            "group": g,
            "count": counts[g],
            "n_traj": len(trajs_per[g]),
            "mean_G": t_bar.get(g, 0.0),
        }
        for g, _ in sorted(counts.items(), key=lambda kv: -kv[1])
    ]

    for name, md in models.items():
        seen: dict[str, dict[str, Any]] = {}
        for tr in md["turns"]:
            g = tr["group"]
            if g not in seen:
                seen[g] = {"group": g, "A_E": tr["A_E"], "n": 0, "A_S_sum": 0.0, "A_sum": 0.0}
            seen[g]["n"] += 1
            seen[g]["A_S_sum"] += float(tr["A_S"])
            seen[g]["A_sum"] += float(tr["A"])
        md["group_summary"] = [
            {
                "group": v["group"],
                "n": v["n"],
                "A_E": v["A_E"],
                "A_S": v["A_S_sum"] / v["n"],
                "A": v["A_sum"] / v["n"],
            }
            for v in seen.values()
        ]
        md["n_groups"] = len(seen)

    return {
        "inst": case["inst"],
        "note": case.get("note"),
        "mean_R": mean_R,
        "gamma": gamma,
        "models": models,
        "legend": legend,
    }


def build_report(src: dict[str, Any], *, gamma: float = GAMMA) -> dict[str, Any]:
    cases = [compute_case_group_advantages(c, gamma=gamma) for c in src.get("cases") or []]
    return {
        "title": "Trajectory Compare · GiGPO A=A_E+A_S+A_I",
        "source": "compare_judge.json / traj_compare_dp_sft_qwen",
        "formula": (
            "A=A_E+A_S+A_I；A_E=R−mean(R|3模型)；G_t=R·γ^{n−1−t}；"
            "A_S=G_t−mean(G|同T#跨模型逐步，非广播)；"
            "A_I=G_t−mean(G|本轨同S#)；S#=edit 切段近似（无 git dump，Edit/Write 后进入下一段）"
        ),
        "gamma": gamma,
        "models": MODEL_ORDER,
        "cases": cases,
    }


def render_html(report: dict[str, Any]) -> str:
    payload = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trajectory Compare · 组级优势</title>
<style>
:root {{
  --bg:#0f1115; --panel:#171a21; --border:#2a3140; --text:#e7eaf0; --muted:#9aa3b2;
  --accent:#6ea8fe; --good:#3dd68c; --bad:#ff7b72; --chip:#1f6feb;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--text); }}
header {{ position:sticky; top:0; z-index:20; background:#0c0e12f2; border-bottom:1px solid var(--border); padding:12px 18px; }}
header h1 {{ margin:0 0 6px; font-size:18px; }}
header p {{ margin:0; color:var(--muted); font-size:12px; line-height:1.45; max-width:1100px; }}
.tabs, .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
.tab {{ border:1px solid var(--border); background:#12151c; color:var(--text); padding:6px 10px; border-radius:8px; cursor:pointer; font-size:12px; }}
.tab.active {{ border-color:var(--accent); background:#182033; }}
.toolbar {{ color:var(--muted); font-size:12px; align-items:center; }}
.toolbar label {{ display:flex; gap:6px; align-items:center; cursor:pointer; }}
.case {{ display:none; padding:14px 18px 48px; }}
.case.active {{ display:block; }}
.note {{ margin:0 0 10px; padding:10px 12px; background:#1a2030; border-left:3px solid var(--accent); border-radius:0 8px 8px 0; font-size:13px; color:#c9d4e8; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px; margin:0 0 12px; }}
.legend .chip {{ font-size:11px; padding:3px 8px; border-radius:999px; color:#fff; cursor:pointer; border:1px solid transparent; }}
.legend .chip.on {{ outline:2px solid #fff; }}
.meta {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-bottom:12px; }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:10px 12px; }}
.card h3 {{ margin:0 0 6px; font-size:14px; }}
.stat {{ color:var(--muted); font-size:12px; line-height:1.45; }}
.stat b {{ color:var(--text); font-family: ui-monospace, Menlo, Consolas, monospace; }}
.pos {{ color:var(--good); }} .neg {{ color:var(--bad); }}
.grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; align-items:start; }}
.col {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; max-height:calc(100vh - 280px); overflow:auto; }}
.col h2 {{ position:sticky; top:0; margin:0; padding:10px 12px; background:#141821; border-bottom:1px solid var(--border); font-size:13px; z-index:5; }}
.gsum {{ padding:8px 10px; border-bottom:1px solid var(--border); font-size:11px; color:var(--muted); }}
.gsum div {{ display:flex; justify-content:space-between; gap:8px; padding:2px 0; }}
.turn {{ border-bottom:1px solid #222836; padding:8px 10px; font-size:12px; }}
.turn.dim {{ opacity:0.22; }}
.turn.hi {{ background:#1a2740; box-shadow: inset 3px 0 0 var(--accent); }}
.turn-h {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:4px; }}
.badge {{ font-size:10px; padding:1px 6px; border-radius:999px; border:1px solid var(--border); color:#dbe7ff; }}
.gid {{ font-weight:700; color:#fff; border:none; }}
.score {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-weight:700; min-width:4.5rem; }}
.raw {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size:10px; color:var(--muted); }}
.bar-wrap {{ flex:1; min-width:5rem; height:6px; background:#2a3140; position:relative; border-radius:99px; overflow:hidden; }}
.bar {{ position:absolute; top:0; bottom:0; }}
.bar.p {{ left:50%; background:var(--good); }}
.bar.n {{ right:50%; background:var(--bad); }}
.cmd {{ font-family: ui-monospace, Menlo, Consolas, monospace; white-space:pre-wrap; word-break:break-word; background:#0e1218; border:1px solid #243044; border-radius:6px; padding:6px 8px; margin:4px 0; color:#d6deea; line-height:1.35; max-height:90px; overflow:auto; }}
@media (max-width: 1100px) {{ .grid,.meta {{ grid-template-columns:1fr; }} .col {{ max-height:none; }} }}
</style>
</head>
<body>
<header>
  <h1>Trajectory Compare · GiGPO（A = A_E + A_S + A_I）</h1>
  <p id="formula"></p>
  <div class="tabs" id="tabs"></div>
  <div class="toolbar">
    <label><input type="checkbox" id="onlyNegA"/> 只看 A&lt;0</label>
    <label><input type="checkbox" id="collapseSame"/> 同 T# 只显示首条（查看用，不是广播）</label>
    <label><input type="checkbox" id="syncScroll" checked/> 同步滚动</label>
    <span id="activeGroup" class="stat"></span>
  </div>
</header>
<div id="cases"></div>
<script id="report-data" type="application/json">{payload}</script>
<script>
const report = JSON.parse(document.getElementById('report-data').textContent);
const names = report.models || ["DeepSeek","SFT","Qwen"];
const COLORS = ["#1f6feb","#2ea043","#d2a8ff","#e3b341","#f85149","#79c0ff","#56d364","#db61dd","#ffa657","#a371f7","#39c5cf","#ff7b72"];
let activeCase = 0;
let activeGroup = null;

document.getElementById('formula').textContent = report.formula || '';

function esc(s) {{
  return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function fmt(n, d=3) {{
  if (n==null || Number.isNaN(n)) return '—';
  const x = Number(n);
  return (x>=0?'+':'') + x.toFixed(d);
}}
function clsNum(n) {{ return Number(n) >= 0 ? 'pos' : 'neg'; }}
function barHtml(val, scale) {{
  const s = Math.max(-scale, Math.min(scale, Number(val)||0));
  const pct = Math.abs(s) / Math.max(scale, 1e-6) * 50;
  if (s>=0) return `<div class="bar-wrap"><div class="bar p" style="width:${{pct}}%"></div></div>`;
  return `<div class="bar-wrap"><div class="bar n" style="width:${{pct}}%"></div></div>`;
}}
function colorFor(g, legend) {{
  const i = legend.findIndex(x => x.group === g);
  return COLORS[(i>=0?i:0) % COLORS.length];
}}

function buildTabs() {{
  const tabs = document.getElementById('tabs');
  tabs.innerHTML = report.cases.map((c,i) => {{
    const short = c.inst.replace(/__/g,'/').split('/').slice(-1)[0];
    return `<button class="tab ${{i===activeCase?'active':''}}" data-i="${{i}}">${{esc(short)}}</button>`;
  }}).join('');
  tabs.querySelectorAll('.tab').forEach(b => b.onclick = () => {{
    activeCase = +b.dataset.i; activeGroup=null; render();
  }});
}}

function turnRows(md, legend, collapse) {{
  const scale = Math.max(0.2, ...md.turns.map(t => Math.abs(t.A||0)));
  let turns = md.turns;
  if (collapse) {{
    const seen = new Set();
    turns = md.turns.filter(t => {{
      if (seen.has(t.group)) return false;
      seen.add(t.group);
      return true;
    }});
  }}
  return turns.map(tr => {{
    const dim = activeGroup && tr.group !== activeGroup;
    const hi = activeGroup && tr.group === activeGroup;
    const col = colorFor(tr.group, legend);
    const cmd = (tr.cmds && tr.cmds[0]) ? tr.cmds[0] : '';
    const nSame = md.turns.filter(x => x.group===tr.group).length;
    return `<div class="turn ${{dim?'dim':''}} ${{hi?'hi':''}}" data-group="${{esc(tr.group)}}">
      <div class="turn-h">
        <span class="badge">t${{tr.turn}}${{collapse?` · ×${{nSame}}`:''}}</span>
        <span class="badge gid" style="background:${{col}}" title="${{esc(tr.group)}}">${{esc(tr.group)}}</span>
        <span class="score ${{clsNum(tr.A)}}">A ${{fmt(tr.A)}}</span>
        ${{barHtml(tr.A, scale)}}
        <span class="raw">A_E=${{fmt(tr.A_E)}} · A_S=${{fmt(tr.A_S)}} · A_I=${{fmt(tr.A_I)}} · G=${{fmt(tr.G,4)}} · ${{esc(tr.S||'')}}</span>
        ${{tr.llm_advantage!=null?`<span class="raw">LLM A_t=${{fmt(tr.llm_advantage)}}</span>`:''}}
      </div>
      ${{cmd?`<div class="cmd">${{esc(cmd.slice(0,240))}}</div>`:''}}
    </div>`;
  }}).join('');
}}

function render() {{
  const c = report.cases[activeCase];
  const collapse = document.getElementById('collapseSame').checked;
  const onlyNeg = document.getElementById('onlyNegA').checked;
  const host = document.getElementById('cases');
  const legend = c.legend || [];
  document.getElementById('activeGroup').textContent = activeGroup ? ('高亮组: '+activeGroup) : '';

  const legendHtml = `<div class="legend">${{legend.slice(0,18).map(g => {{
    const col = colorFor(g.group, legend);
    const on = activeGroup===g.group ? 'on':'';
    return `<span class="chip ${{on}}" style="background:${{col}}" data-g="${{esc(g.group)}}" title="跨模型 A_S">${{esc(g.group)}} · ×${{g.count}} · ${{g.n_traj}}traj</span>`;
  }}).join('')}}</div>`;

  const meta = `<div class="meta">${{names.map(n => {{
    const m = c.models[n];
    if (!m) return `<div class="card"><h3>${{n}}</h3><div class="stat">missing</div></div>`;
    return `<div class="card"><h3>${{n}} ${{m.solved?'✓':'✗'}}</h3>
      <div class="stat">R=<b>${{m.R.toFixed(1)}}</b> · A_E=<b class="${{clsNum(m.A_E)}}">${{fmt(m.A_E)}}</b> · turns=<b>${{m.n_turns}}</b> · groups=<b>${{m.n_groups}}</b></div>
      <div class="stat">exit=${{esc(m.exit_status||'')}} · LLM mean_r=${{m.mean_r!=null?Number(m.mean_r).toFixed(3):'—'}}</div>
    </div>`;
  }}).join('')}}</div>`;

  const cols = `<div class="grid">${{names.map(n => {{
    const m = c.models[n];
    if (!m) return `<div class="col"><h2>${{n}}</h2></div>`;
    let htmlTurns = turnRows(m, legend, collapse);
    if (onlyNeg) {{
      // filter after build is hard; rebuild
      const scale = Math.max(0.2, ...m.turns.map(t => Math.abs(t.A||0)));
      let turns = m.turns;
      if (collapse) {{
        const seen = new Set();
        turns = m.turns.filter(t => {{ if (seen.has(t.group)) return false; seen.add(t.group); return true; }});
      }}
      turns = turns.filter(t => Number(t.A) < 0);
      htmlTurns = turns.map(tr => {{
        const dim = activeGroup && tr.group !== activeGroup;
        const hi = activeGroup && tr.group === activeGroup;
        const col = colorFor(tr.group, legend);
        const cmd = (tr.cmds && tr.cmds[0]) ? tr.cmds[0] : '';
        const nSame = m.turns.filter(x => x.group===tr.group).length;
        return `<div class="turn ${{dim?'dim':''}} ${{hi?'hi':''}}" data-group="${{esc(tr.group)}}">
          <div class="turn-h">
            <span class="badge">t${{tr.turn}}${{collapse?` · ×${{nSame}}`:''}}</span>
            <span class="badge gid" style="background:${{col}}">${{esc(tr.group)}}</span>
            <span class="score neg">A ${{fmt(tr.A)}}</span>
            ${{barHtml(tr.A, scale)}}
            <span class="raw">A_E=${{fmt(tr.A_E)}} · A_S=${{fmt(tr.A_S)}} · A_I=${{fmt(tr.A_I)}} · ${{esc(tr.S||'')}}</span>
          </div>
          ${{cmd?`<div class="cmd">${{esc(cmd.slice(0,240))}}</div>`:''}}
        </div>`;
      }}).join('');
    }}
    const gsum = (m.group_summary||[]).map(g =>
      `<div><span>${{esc(g.group)}} ×${{g.n}}</span><span class="${{clsNum(g.A_S)}}">mean A_S ${{fmt(g.A_S)}}</span></div>`
    ).join('');
    return `<div class="col" data-col="${{n}}">
      <h2>${{n}} · A_E ${{fmt(m.A_E)}} · R=${{m.R}}</h2>
      <div class="gsum">${{gsum || '<i>no groups</i>'}}</div>
      ${{htmlTurns}}
    </div>`;
  }}).join('')}}</div>`;

  host.innerHTML = `<div class="case active">
    <div class="note"><b>${{esc(c.inst)}}</b> · ${{esc(c.note||'')}} · mean_R=${{Number(c.mean_R).toFixed(3)}} · γ=${{c.gamma}}</div>
    ${{legendHtml}}${{meta}}${{cols}}
  </div>`;

  host.querySelectorAll('.legend .chip').forEach(ch => ch.onclick = () => {{
    const g = ch.getAttribute('data-g');
    activeGroup = (activeGroup === g) ? null : g;
    render();
  }});
  host.querySelectorAll('.turn').forEach(el => el.onclick = () => {{
    const g = el.getAttribute('data-group');
    activeGroup = (activeGroup === g) ? null : g;
    render();
  }});

  // sync scroll
  if (document.getElementById('syncScroll').checked) {{
    const cols = [...host.querySelectorAll('.col')];
    cols.forEach(col => {{
      col.onscroll = () => {{
        cols.forEach(o => {{ if (o!==col) o.scrollTop = col.scrollTop; }});
      }};
    }});
  }}
}}

document.getElementById('onlyNegA').onchange = render;
document.getElementById('collapseSame').onchange = render;
document.getElementById('syncScroll').onchange = render;
buildTabs();
render();
</script>
</body>
</html>
"""


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        type=Path,
        default=Path("/workspace/work/mjy/traj_compare_judge/compare_judge.json"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/work/mjy/traj_compare_group_advantage.html"),
    )
    p.add_argument("--gamma", type=float, default=GAMMA)
    args = p.parse_args()

    src = json.loads(args.src.read_text())
    report = build_report(src, gamma=args.gamma)
    html = render_html(report)
    args.out.write_text(html)
    # also copy next to judge dir
    alt = args.out.with_name(args.out.name)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    print(f"cases={len(report['cases'])} gamma={args.gamma}")
    for c in report["cases"]:
        parts = []
        for n in MODEL_ORDER:
            m = c["models"].get(n)
            if not m:
                continue
            parts.append(f"{n}:{'✓' if m['solved'] else '✗'} A_E={m['A_E']:+.3f} g={m['n_groups']}")
        print(f"  {c['inst']}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
