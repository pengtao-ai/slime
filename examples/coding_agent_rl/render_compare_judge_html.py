#!/usr/bin/env python3
"""Render simplified score + A_t + spin HTML from compare_judge.json."""

from __future__ import annotations

import json
from pathlib import Path


def render_html(report: dict) -> str:
    payload = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trajectory Compare · score + A_t</title>
<style>
:root {{
  --bg:#0f1115; --panel:#171a21; --border:#2a3140; --text:#e7eaf0; --muted:#9aa3b2;
  --accent:#6ea8fe; --llm:#3a2f14; --llm-b:#ffb020; --spin:#5a2a2a; --spin-b:#e35d6a;
  --good:#3dd68c; --bad:#ff7b72;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--text); }}
header {{ position:sticky; top:0; z-index:20; background:#0c0e12ee; border-bottom:1px solid var(--border); padding:12px 18px; }}
header h1 {{ margin:0 0 6px; font-size:18px; }}
header p {{ margin:0; color:var(--muted); font-size:13px; }}
.tabs, .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
.tab {{ border:1px solid var(--border); background:#12151c; color:var(--text); padding:6px 10px; border-radius:8px; cursor:pointer; font-size:12px; }}
.tab.active {{ border-color:var(--accent); background:#182033; }}
.toolbar {{ color:var(--muted); font-size:12px; align-items:center; }}
.toolbar label {{ display:flex; gap:6px; align-items:center; cursor:pointer; }}
.case {{ display:none; padding:14px 18px 48px; }}
.case.active {{ display:block; }}
.note {{ margin:0 0 12px; padding:10px 12px; background:#1a2030; border-left:3px solid var(--accent); border-radius:0 8px 8px 0; font-size:13px; color:#c9d4e8; }}
.meta {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-bottom:12px; }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:10px 12px; }}
.card h3 {{ margin:0 0 6px; font-size:14px; }}
.stat {{ color:var(--muted); font-size:12px; line-height:1.45; }}
.stat b {{ color:var(--text); font-family: ui-monospace, Menlo, Consolas, monospace; }}
.pos {{ color:var(--good); }} .neg {{ color:var(--bad); }}
.grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; align-items:start; }}
.col {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; max-height:calc(100vh - 240px); overflow:auto; }}
.col h2 {{ position:sticky; top:0; margin:0; padding:10px 12px; background:#141821; border-bottom:1px solid var(--border); font-size:13px; z-index:5; }}
.turn {{ border-bottom:1px solid #222836; padding:8px 10px; font-size:12px; }}
.turn.llm {{ background:var(--llm); box-shadow: inset 3px 0 0 var(--llm-b); }}
.turn.spin {{ background:var(--spin); box-shadow: inset 3px 0 0 var(--spin-b); }}
.turn.spin.llm {{ background:#4a3020; box-shadow: inset 3px 0 0 var(--spin-b), inset 6px 0 0 var(--llm-b); }}
.turn.low:not(.spin) {{ background:#2a1616; }}
.turn-h {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:4px; }}
.badge {{ font-size:10px; padding:1px 6px; border-radius:999px; border:1px solid var(--border); color:var(--muted); }}
.badge.spinb {{ background:#6b2030; color:#ffc0c8; border-color:#e35d6a; }}
.badge.llmb {{ background:#5a4510; color:#ffd27a; border-color:#ffb020; }}
.badge.pp {{ background:#203040; color:#9ecbff; border-color:#6ea8fe; }}
.score, .adv {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-weight:600; min-width:4.2rem; }}
.adv {{ font-size:11px; }}
.raw {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size:10px; color:var(--muted); }}
.bar-wrap {{ flex:1; min-width:5rem; height:6px; background:#2a3140; position:relative; border-radius:99px; overflow:hidden; }}
.bar {{ position:absolute; top:0; bottom:0; }}
.bar.p {{ left:50%; background:var(--good); }}
.bar.n {{ right:50%; background:var(--bad); }}
.cmd {{ font-family: ui-monospace, Menlo, Consolas, monospace; white-space:pre-wrap; word-break:break-word; background:#0e1218; border:1px solid #243044; border-radius:6px; padding:6px 8px; margin:4px 0; color:#d6deea; line-height:1.35; }}
.reason {{ color:#c9d4e8; margin:4px 0; }}
.spark {{ display:flex; align-items:flex-end; gap:1px; height:36px; overflow:auto; margin-top:6px; }}
.spark i {{ display:block; width:3px; min-width:2px; background:var(--good); }}
.spark i.neg {{ background:var(--bad); }}
@media (max-width: 1100px) {{ .grid,.meta {{ grid-template-columns:1fr; }} .col {{ max-height:none; }} }}
</style>
</head>
<body>
<header>
  <h1>Trajectory Compare · judge score + A_t</h1>
  <p>A_t = r − mean(r)。分数=LLM 按打分标准直接给出（无规则夹紧）。harness resolved 对齐对错；红底=spin 仅展示。</p>
  <div class="tabs" id="tabs"></div>
  <div class="toolbar">
    <label><input type="checkbox" id="onlySpin"/> 只看 spin</label>
    <label><input type="checkbox" id="onlyNegA"/> 只看 A_t&lt;0</label>
    <label><input type="checkbox" id="onlyPp"/> 只看后处理下调</label>
    <label><input type="checkbox" id="syncScroll" checked/> 同步滚动</label>
  </div>
</header>
<div id="cases"></div>
<script id="report-data" type="application/json">{payload}</script>
<script>
const report = JSON.parse(document.getElementById('report-data').textContent);
const names = ["DeepSeek","SFT","Qwen"];
function esc(s) {{
  return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function fmt(n, d=3) {{
  if (n==null || Number.isNaN(n)) return '—';
  return Number(n).toFixed(d);
}}
function clsNum(n) {{ return Number(n) >= 0 ? 'pos' : 'neg'; }}
function barHtml(val, scale) {{
  const s = Math.max(-scale, Math.min(scale, Number(val)||0));
  const pct = Math.abs(s) / scale * 50;
  if (s>=0) return `<div class="bar-wrap"><div class="bar p" style="width:${{pct}}%"></div></div>`;
  return `<div class="bar-wrap"><div class="bar n" style="width:${{pct}}%"></div></div>`;
}}
function spark(rewards, key) {{
  const maxAbs = Math.max(...rewards.map(r => Math.abs(Number(r[key])||0)), 1e-6);
  return '<div class="spark">' + rewards.map(r => {{
    const s = Number(r[key])||0;
    const h = Math.max(2, Math.round(Math.abs(s)/maxAbs*34));
    return `<i class="${{s<0?'neg':''}}" style="height:${{h}}px" title="T${{r.turn}} ${{key}}=${{s.toFixed(3)}}"></i>`;
  }}).join('') + '</div>';
}}
function turnHtml(tr, advScale) {{
  const tags = (tr.tags||[]).map(t => `<span class="badge">${{esc(t)}}</span>`).join('');
  const spin = (tr.spin||[]).map(s => `<span class="badge spinb">${{esc(s)}}</span>`).join('');
  const llm = tr.offloaded ? '<span class="badge llmb">LLM offload</span>' : (tr.offloaded===false?'<span class="badge">slm</span>':'');
  const pp = tr.postprocess ? `<span class="badge pp">pp:${{esc(tr.postprocess)}}</span>` : '';
  const cmds = (tr.cmds&&tr.cmds.length) ? tr.cmds.map(c => `<div class="cmd">${{esc(c)}}</div>`).join('') : (tr.tool_calls?`<div class="cmd">${{esc(tr.tool_calls)}}</div>`:'');
  const a = Number(tr.advantage);
  const isSpin = (tr.spin&&tr.spin.length) ? 1 : 0;
  const hasPp = tr.postprocess ? 1 : 0;
  const klass = ['turn', tr.offloaded?'llm':'', isSpin?'spin':'', (!isSpin && a<0)?'low':''].filter(Boolean).join(' ');
  const raw = (tr.score_raw!=null && Math.abs(Number(tr.score_raw)-Number(tr.score))>1e-9)
    ? `<span class="raw">raw ${{fmt(tr.score_raw)}}</span>` : '';
  const rc = (tr.rcs&&tr.rcs.length) ? `<span class="badge">rc=${{esc(tr.rcs.join(','))}}</span>` : '';
  return `<div class="${{klass}}" data-spin="${{isSpin}}" data-nega="${{a<0?1:0}}" data-pp="${{hasPp}}">
    <div class="turn-h">
      <strong>#${{tr.turn}}</strong>
      <span class="score ${{clsNum(tr.score)}}">r ${{fmt(tr.score)}}</span>
      ${{raw}}
      <span class="adv ${{clsNum(a)}}">A_t ${{fmt(a)}}</span>
      ${{barHtml(a, advScale)}}
      ${{tags}}${{spin}}${{pp}}${{llm}}${{rc}}
    </div>
    <div class="reason">${{esc(tr.reason||'')}}</div>
    ${{cmds}}
  </div>`;
}}
function modelCard(name, m) {{
  const mean = m.mean_r ?? (m.token_totals||{{}}).mean_score;
  const spinN = (m.llm_turn_rewards||[]).filter(x => x.spin && x.spin.length).length;
  const ppN = m.n_postprocess_adjusted || (m.llm_turn_rewards||[]).filter(x => x.postprocess).length;
  return `<div class="card"><h3>${{esc(name)}}</h3>
    <div class="stat">exit <b>${{esc(m.exit_status)}}</b> · harness <b>${{m.solved?'✓':'✗'}}</b> · submitted <b>${{m.submitted?'✓':'✗'}}</b> · turns <b>${{m.n_turns}}</b></div>
    <div class="stat">mean r <b class="${{clsNum(mean)}}">${{fmt(mean)}}</b> · spin <b>${{spinN}}</b> · pp下调 <b>${{ppN}}</b></div>
    <div class="stat">judge <b>${{esc(m.judge_mode)}}</b></div>
    ${{spark(m.llm_turn_rewards||[], 'advantage')}}
  </div>`;
}}
function applyFilters() {{
  const onlySpin = document.getElementById('onlySpin').checked;
  const onlyNeg = document.getElementById('onlyNegA').checked;
  const onlyPp = document.getElementById('onlyPp').checked;
  document.querySelectorAll('.case.active .turn').forEach(t => {{
    const okSpin = !onlySpin || t.dataset.spin==='1';
    const okNeg = !onlyNeg || t.dataset.nega==='1';
    const okPp = !onlyPp || t.dataset.pp==='1';
    t.style.display = (okSpin && okNeg && okPp) ? '' : 'none';
  }});
}}
function render() {{
  const tabs = document.getElementById('tabs');
  const wrap = document.getElementById('cases');
  tabs.innerHTML = ''; wrap.innerHTML = '';
  (report.cases||[]).forEach((c, idx) => {{
    const ns = names.map(n => (c.models[n]||{{}}).n_turns || 0);
    const btn = document.createElement('button');
    btn.className = 'tab' + (idx===0?' active':'');
    btn.textContent = c.inst.split('__').pop() + ` (${{ns.join('/')}})`;
    btn.onclick = () => {{
      document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
      document.querySelectorAll('.case').forEach(x=>x.classList.remove('active'));
      btn.classList.add('active');
      document.querySelector('.case[data-i="'+idx+'"]').classList.add('active');
      applyFilters();
      bindSync();
    }};
    tabs.appendChild(btn);
    const sec = document.createElement('section');
    sec.className = 'case' + (idx===0?' active':'');
    sec.dataset.i = idx;
    sec.innerHTML = `
      <div class="note"><b>${{esc(c.inst)}}</b> — ${{esc(c.note)}}</div>
      <div class="meta">${{names.map(n => modelCard(n, c.models[n]||{{}})).join('')}}</div>
      <div class="grid">${{names.map(n => {{
        const m = c.models[n]||{{}};
        const rewards = m.llm_turn_rewards||[];
        const advScale = Math.max(0.2, ...rewards.map(x => Math.abs(Number(x.advantage)||0)));
        return `<div class="col" data-col="${{n}}"><h2>${{n}} · ${{m.n_turns}} turns · mean r ${{fmt(m.mean_r)}}</h2>
          ${{rewards.map(tr => turnHtml(tr, advScale)).join('')}}</div>`;
      }}).join('')}}</div>`;
    wrap.appendChild(sec);
  }});
}}
render();
document.getElementById('onlySpin').onchange = applyFilters;
document.getElementById('onlyNegA').onchange = applyFilters;
document.getElementById('onlyPp').onchange = applyFilters;
let syncing=false;
function bindSync() {{
  document.querySelectorAll('.case.active .col').forEach(col => {{
    col.onscroll = () => {{
      if (!document.getElementById('syncScroll').checked || syncing) return;
      syncing = true;
      const ratio = col.scrollTop / (col.scrollHeight - col.clientHeight || 1);
      document.querySelectorAll('.case.active .col').forEach(o => {{
        if (o!==col) o.scrollTop = ratio * (o.scrollHeight - o.clientHeight);
      }});
      syncing=false;
    }};
  }});
}}
bindSync();
</script>
</body>
</html>
"""


def main() -> None:
    src = Path("/workspace/work/mjy/traj_compare_judge/compare_judge.json")
    report = json.loads(src.read_text(encoding="utf-8"))
    html_txt = render_html(report)
    for p in [
        Path("/workspace/work/mjy/traj_compare_judge/compare_judge.html"),
        Path("/workspace/work/mjy/traj_compare_dp_sft_qwen.html"),
    ]:
        p.write_text(html_txt, encoding="utf-8")
        print("wrote", p, p.stat().st_size)


if __name__ == "__main__":
    main()
