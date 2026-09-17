#!/usr/bin/env python3
"""GiGPO vs LLM-judge on selected groups from a phase2 offload run.

True GiGPO (matches docker / training narrative):
  A = A_E + w * A_S
  A_E = R - mean(R|siblings)
  G_t = R * γ^{n-1-t}
  T#  = intent · tool   → A_S = G_t - mean(G|same T#)   (per-step, discounted)

LLM judge: reuse llm_turn_credit_assign.judge_trajectory (needs DASHSCOPE_*).

Example:
  python examples/coding_agent_rl/compare_gigpo_llm_cases.py \\
    --run-dir runs/agent_offload_pyrodash4b_phase2_sft03_prob_20260824_113854 \\
    --pick /tmp/gigpo_llm_case_pick.json \\
    --out-dir /workspace/work/mjy/gigpo_llm_compare_phase2_sft03
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent.parent))  # repo root for examples.*

import llm_turn_credit_assign as judge  # noqa: E402
from examples.coding_agent_rl.gigpo_advantage import (  # noqa: E402
    classify_message_tools,
)

GAMMA = 0.95
W = 1.0


def unique_trajs_by_index(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one full traj per GRPO index (samples with agent + turn_costs)."""
    by: dict[int, dict[str, Any]] = {}
    for s in samples:
        md = s.get("metadata") or {}
        if not md.get("turn_costs") or not md.get("agent"):
            continue
        by[int(s.get("index", -1))] = s
    return [by[i] for i in sorted(by)]


def group_key_from_turn(tc: dict[str, Any]) -> tuple[str, str, str]:
    if tc.get("gigpo_T"):
        key = str(tc["gigpo_T"])
        # "intent · family"
        if " · " in key:
            intent, tool = key.split(" · ", 1)
            return key, intent, tool
        return key, "其他", key
    am = tc.get("sft_assistant_message") if isinstance(tc.get("sft_assistant_message"), dict) else {}
    if am:
        return classify_message_tools(am)
    return "其他 · 无 tool", "其他", "无 tool"

def git_diff_key(md: dict[str, Any], turn_i: int) -> str:
    diffs = md.get("turn_git_diffs") or []
    if turn_i < len(diffs) and isinstance(diffs[turn_i], dict):
        g = diffs[turn_i].get("git_diff") or ""
        # normalize whitespace for segment id
        g = re.sub(r"\s+", " ", g).strip()
        return g[:240] if g else "<empty>"
    return "<empty>"


def compute_gigpo_group(
    traj_samples: list[dict[str, Any]],
    *,
    gamma: float = GAMMA,
    w: float = W,
) -> dict[str, Any]:
    trajs: list[dict[str, Any]] = []
    for ti, s in enumerate(traj_samples):
        md = s.get("metadata") or {}
        tc_list = list(md.get("turn_costs") or [])
        R = float(s.get("reward") or 0.0)
        # binary solve for display; keep R for G
        solved = float(md.get("solved") or 0) > 0.5
        n = len(tc_list)
        turns = []
        for i, tc in enumerate(tc_list):
            gkey, intent, tool = group_key_from_turn(tc)
            G = R * (gamma ** max(0, n - 1 - i))
            turns.append(
                {
                    "turn": i,
                    "T": gkey,
                    "intent": intent,
                    "tool": tool,
                    "S": git_diff_key(md, i),
                    "G": G,
                    "cmd": _tool_brief(tc)[:220],
                    "offload": bool(tc.get("valid_offload")),
                    "train_r": float((md.get("turn_rewards") or [0] * n)[i]) if i < len(md.get("turn_costs") or []) else None,
                }
            )
        trajs.append(
            {
                "traj_index": ti,
                "index": s.get("index"),
                "solved": solved,
                "R": R,
                "agent": md.get("agent"),
                "n_turns": n,
                "turns": turns,
            }
        )

    mean_R = statistics.mean(t["R"] for t in trajs) if trajs else 0.0
    for t in trajs:
        t["A_E"] = t["R"] - mean_R

    # A_S: across trajs, same T#
    by_T: dict[str, list[float]] = defaultdict(list)
    for t in trajs:
        for tr in t["turns"]:
            by_T[tr["T"]].append(tr["G"])
    T_bar = {k: statistics.mean(v) for k, v in by_T.items()}

    for t in trajs:
        ae = t["A_E"]
        for tr in t["turns"]:
            a_s = tr["G"] - T_bar[tr["T"]]
            tr["A_S"] = a_s
            tr["A"] = ae + w * a_s
            tr["A_E"] = ae

    # legend
    t_counts = Counter(tr["T"] for t in trajs for tr in t["turns"])
    legend = [{"T": k, "count": c, "n_traj": len({ti for ti, t in enumerate(trajs) if any(x["T"] == k for x in t["turns"])})} for k, c in t_counts.most_common()]

    return {
        "mean_R": mean_R,
        "gamma": gamma,
        "w": w,
        "n_traj": len(trajs),
        "n_solved": sum(1 for t in trajs if t["solved"]),
        "trajectories": trajs,
        "legend": legend,
    }


def paint_llm_advantages(traj: dict[str, Any]) -> None:
    items = list(traj.get("llm_turn_rewards") or [])
    if not items:
        return
    scores = [float(x.get("score") or 0) for x in items]
    mean_r = statistics.mean(scores) if scores else 0.0
    a_s = float(traj.get("A_E_outcome") or 0.0)  # use same A_E as GiGPO for fair compare
    traj["mean_r"] = mean_r
    for it in items:
        r = float(it.get("score") or 0)
        it["residual"] = r - mean_r
        it["a_s"] = a_s
        it["advantage"] = a_s + (r - mean_r)


def judge_one(
    sample: dict[str, Any],
    *,
    args: argparse.Namespace,
    a_e: float,
) -> dict[str, Any]:
    out = judge.judge_trajectory(
        sample,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        text_limit=args.text_limit,
        normalize=args.normalize,
        heuristic=args.heuristic,
    )
    out["A_E_outcome"] = a_e
    paint_llm_advantages(out)
    return out


def run_cases(args: argparse.Namespace) -> dict[str, Any]:
    pick = json.loads(Path(args.pick).read_text())
    cases_out = []
    for spec in pick:
        rid, gi = int(spec["rollout"]), int(spec["group"])
        print(f"\n=== load r{rid} g{gi} {spec.get('inst')} ===", flush=True)
        dump = judge.load_rollout(args.run_dir, rid)
        samples = [s for s in dump["samples"] if int(s.get("group_index", 0)) == gi]
        trajs = unique_trajs_by_index(samples)
        print(f"  unique trajs={len(trajs)} solved={sum(1 for s in trajs if float((s.get('metadata') or {}).get('solved') or 0)>0.5)}", flush=True)

        gigpo = compute_gigpo_group(trajs, gamma=args.gamma, w=args.w)

        # LLM judge in parallel
        llm_trajs: list[dict[str, Any] | None] = [None] * len(trajs)

        def _one(i_s: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
            i, sample = i_s
            a_e = gigpo["trajectories"][i]["A_E"]
            t0 = time.time()
            try:
                out = judge_one(sample, args=args, a_e=a_e)
            except Exception as exc:  # noqa: BLE001
                md = sample.get("metadata") or {}
                turns = judge.compact_turns(sample, text_limit=args.text_limit)
                solved = float(md.get("solved") or 0) > 0.5
                outcome = float(sample.get("reward") or 0)
                fb = judge.heuristic_scores(turns, solved=solved, outcome_reward=outcome)
                scored = judge.normalize_scores(list(fb["turns"]), n_turns=len(turns), outcome_reward=outcome, mode=args.normalize)
                for t, c in zip(scored, turns):
                    t["context"] = c
                out = {
                    "instance_id": md.get("instance_id"),
                    "agent": md.get("agent"),
                    "solved": solved,
                    "outcome_reward": outcome,
                    "n_turns": len(turns),
                    "judge_mode": "error_fallback",
                    "judge_summary": f"judge failed: {exc}",
                    "llm_turn_rewards": scored,
                    "error": str(exc),
                    "A_E_outcome": a_e,
                }
                paint_llm_advantages(out)
            out["latency_s"] = round(time.time() - t0, 2)
            out["traj_index"] = i
            return i, out

        workers = max(1, int(args.concurrency))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_one, (i, s)) for i, s in enumerate(trajs)]
            for fut in concurrent.futures.as_completed(futs):
                i, out = fut.result()
                llm_trajs[i] = out
                err = f" ERR={out.get('error')}" if out.get("error") else ""
                print(
                    f"  [llm] traj#{i} solved={out.get('solved')} turns={out.get('n_turns')} "
                    f"mode={out.get('judge_mode')} {out.get('latency_s')}s{err}",
                    flush=True,
                )

        # align llm cmds onto gigpo turns for HTML
        for i, gt in enumerate(gigpo["trajectories"]):
            lt = llm_trajs[i] or {}
            gt["llm"] = {
                "judge_mode": lt.get("judge_mode"),
                "summary": lt.get("judge_summary"),
                "mean_r": lt.get("mean_r"),
                "turns": lt.get("llm_turn_rewards") or [],
            }

        analysis = analyze_case(gigpo)
        cases_out.append(
            {
                "rollout": rid,
                "group": gi,
                "inst": spec.get("inst"),
                "note": f"{gigpo['n_solved']}/{gigpo['n_traj']} solved · agents={Counter(t['agent'] for t in gigpo['trajectories'])}",
                "gigpo": gigpo,
                "analysis": analysis,
            }
        )
        print(f"  analysis: {analysis.get('headline')}", flush=True)

    return {
        "title": "GiGPO vs LLM judge · phase2_sft03",
        "run_dir": str(args.run_dir),
        "formula_gigpo": "A=A_E+w*A_S; A_S=G_t−mean(G|T#); G_t=R·γ^{n−1−t}",
        "formula_llm": "A_t=A_E+(r_i−mean r); r_i from LLM process judge",
        "gamma": args.gamma,
        "w": args.w,
        "heuristic": bool(args.heuristic),
        "model": args.model,
        "cases": cases_out,
    }


def analyze_case(gigpo: dict[str, Any]) -> dict[str, Any]:
    trajs = gigpo["trajectories"]
    all_turns = [tr for t in trajs for tr in t["turns"]]
    if not all_turns:
        return {"headline": "empty"}

    def mean_abs(xs: list[float]) -> float:
        return statistics.mean(map(abs, xs)) if xs else 0.0

    a_s = [tr["A_S"] for tr in all_turns]
    a = [tr["A"] for tr in all_turns]
    zero_as = sum(1 for x in a_s if abs(x) < 1e-9) / len(a_s)

    # early vs late A_S within traj
    early, late = [], []
    for t in trajs:
        n = len(t["turns"])
        if n < 3:
            continue
        for i, tr in enumerate(t["turns"]):
            if i < n * 0.4:
                early.append(tr["A_S"])
            if i >= n * 0.7:
                late.append(tr["A_S"])

    # same T within one traj: is A_S varying?
    within_var = []
    for t in trajs:
        by_t: dict[str, list[float]] = defaultdict(list)
        for tr in t["turns"]:
            by_t[tr["T"]].append(tr["A_S"])
        for xs in by_t.values():
            if len(xs) >= 2:
                within_var.append(statistics.pstdev(xs))

    # LLM residuals if present
    llm_res = []
    llm_spinish = 0
    llm_n = 0
    for t in trajs:
        for tr in (t.get("llm") or {}).get("turns") or []:
            llm_n += 1
            llm_res.append(float(tr.get("residual") or 0))
            reason = str(tr.get("reason") or "").lower()
            if any(k in reason for k in ["重复", "空转", "浪费", "无进展", "spin"]):
                llm_spinish += 1

    # correlation: solved traj mean A vs unsolved
    ok_a = [statistics.mean(tr["A"] for tr in t["turns"]) for t in trajs if t["solved"] and t["turns"]]
    bad_a = [statistics.mean(tr["A"] for tr in t["turns"]) for t in trajs if (not t["solved"]) and t["turns"]]

    headline = (
        f"|A_S|μ={mean_abs(a_s):.3f} zero={zero_as:.0%} "
        f"A_S early={statistics.mean(early) if early else 0:+.3f} late={statistics.mean(late) if late else 0:+.3f} "
        f"sameT σμ={statistics.mean(within_var) if within_var else 0:.3f}"
    )
    return {
        "headline": headline,
        "abs_A_S": mean_abs(a_s),
        "abs_A": mean_abs(a),
        "A_S_zero_rate": zero_as,
        "A_S_early": statistics.mean(early) if early else None,
        "A_S_late": statistics.mean(late) if late else None,
        "same_T_A_S_std_mean": statistics.mean(within_var) if within_var else None,
        "mean_A_solved": statistics.mean(ok_a) if ok_a else None,
        "mean_A_failed": statistics.mean(bad_a) if bad_a else None,
        "llm_abs_residual": mean_abs(llm_res) if llm_res else None,
        "llm_spinish_rate": (llm_spinish / llm_n) if llm_n else None,
        "llm_n_turns": llm_n,
    }


def render_html(report: dict[str, Any]) -> str:
    payload = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
    tpl = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>GiGPO vs LLM judge</title>
<style>
:root { --bg:#0f1115; --panel:#171a21; --border:#2a3140; --text:#e7eaf0; --muted:#9aa3b2;
  --good:#3dd68c; --bad:#ff7b72; --accent:#6ea8fe; }
* { box-sizing:border-box; }
body { margin:0; font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--text); }
header { position:sticky; top:0; z-index:20; background:#0c0e12f2; border-bottom:1px solid var(--border); padding:12px 18px; }
h1 { margin:0 0 6px; font-size:18px; }
.sub { color:var(--muted); font-size:12px; line-height:1.45; }
.tabs { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.tab { border:1px solid var(--border); background:#12151c; color:var(--text); padding:6px 10px; border-radius:8px; cursor:pointer; font-size:12px; }
.tab.active { border-color:var(--accent); background:#182033; }
.case { display:none; padding:14px 18px 40px; } .case.active { display:block; }
.note { margin:0 0 10px; padding:10px 12px; background:#1a2030; border-left:3px solid var(--accent); border-radius:0 8px 8px 0; font-size:13px; }
.stats { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:10px 12px; font-size:12px; color:var(--muted); }
.card b { color:var(--text); font-family:ui-monospace,Menlo,monospace; }
.pos { color:var(--good); } .neg { color:var(--bad); }
.grid { display:grid; grid-template-columns:repeat(8, minmax(220px, 1fr)); gap:8px; overflow-x:auto; padding-bottom:6px; }
.traj { background:var(--panel); border:1px solid var(--border); border-radius:8px; max-height:70vh; overflow:auto; min-width:220px; }
.traj-head { position:sticky; top:0; background:#141821; border-bottom:1px solid var(--border); z-index:2; }
.traj h3 { margin:0; padding:8px 10px 4px; font-size:12px; }
.chart { padding:2px 8px 8px; }
.chart-label { font-size:10px; color:var(--muted); margin-bottom:4px; display:flex; justify-content:space-between; gap:8px; }
.chart-bars { display:flex; align-items:stretch; gap:1px; height:56px; overflow-x:auto; padding-bottom:2px; }
.chart-bars .col { flex:1 0 4px; min-width:3px; max-width:12px; height:100%; display:flex; flex-direction:column; cursor:pointer; opacity:.95; }
.chart-bars .col.dim { opacity:.18; }
.chart-bars .col .up, .chart-bars .col .dn { flex:1; width:100%; display:flex; }
.chart-bars .col .up { align-items:flex-end; }
.chart-bars .col .dn { align-items:flex-start; }
.chart-bars .col i { display:block; width:100%; border-radius:1px 1px 0 0; min-height:0; }
.chart-bars .col .dn i { border-radius:0 0 1px 1px; }
.chart-bars .col.hl { outline:1px solid #fff; outline-offset:-1px; }
.zero { height:0; border-top:1px dashed #3a4558; margin:-1px 0 0; position:relative; top:28px; pointer-events:none; z-index:0; }
.turn { border-bottom:1px solid #222836; padding:7px 9px; font-size:11px; }
.turn.dim { opacity:.2; }
.row { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.badge { font-size:10px; padding:1px 6px; border-radius:999px; border:1px solid var(--border); color:#dbe7ff; }
.gid { color:#fff; border:none; font-weight:700; }
.mono { font-family:ui-monospace,Menlo,monospace; }
.cmd { margin-top:4px; background:#0e1218; border:1px solid #243044; border-radius:6px; padding:5px 7px; white-space:pre-wrap; word-break:break-word; max-height:70px; overflow:auto; color:#d6deea; }
.bar { flex:1; min-width:60px; height:5px; background:#2a3140; border-radius:99px; position:relative; overflow:hidden; }
.bar i { position:absolute; top:0; bottom:0; }
.bar i.p { left:50%; background:var(--good); } .bar i.n { right:50%; background:var(--bad); }
.legend { display:flex; flex-wrap:wrap; gap:5px; margin:8px 0; }
.legend span { font-size:10px; padding:2px 7px; border-radius:999px; color:#fff; cursor:pointer; }
</style></head><body>
<header>
  <h1>GiGPO vs LLM judge</h1>
  <div class="sub" id="sub"></div>
  <div class="tabs" id="tabs"></div>
</header>
<div id="root"></div>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const report = JSON.parse(document.getElementById('data').textContent);
let active=0, activeT=null;
const COLORS=["#1f6feb","#2ea043","#d2a8ff","#e3b341","#f85149","#79c0ff","#56d364","#db61dd","#ffa657","#a371f7"];
document.getElementById('sub').textContent =
  `GiGPO: ${report.formula_gigpo}  |  LLM: ${report.formula_llm}  |  γ=${report.gamma} w=${report.w}  |  judge=${report.heuristic?'heuristic':report.model}`;

function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmt(n,d=3){ if(n==null||Number.isNaN(n)) return '—'; const x=Number(n); return (x>=0?'+':'')+x.toFixed(d); }
function cls(n){ return Number(n)>=0?'pos':'neg'; }
function bar(v,scale){
  const s=Math.max(-scale, Math.min(scale, Number(v)||0));
  const pct=Math.abs(s)/Math.max(scale,1e-6)*50;
  return s>=0 ? `<div class="bar"><i class="p" style="width:${pct}%"></i></div>`
              : `<div class="bar"><i class="n" style="width:${pct}%"></i></div>`;
}
function colorT(T, legend){
  const i=legend.findIndex(x=>x.T===T); return COLORS[(i>=0?i:0)%COLORS.length];
}
function gigpoChart(turns, legend, scaleA){
  const maxAbs=Math.max(scaleA, ...turns.map(tr=>Math.abs(Number(tr.A)||0)), 1e-6);
  const meanA=turns.length?turns.reduce((s,tr)=>s+(Number(tr.A)||0),0)/turns.length:0;
  const bars=turns.map(tr=>{
    const a=Number(tr.A)||0;
    const pct=Math.max(2, Math.round(Math.abs(a)/maxAbs*100));
    const dim=activeT && tr.T!==activeT ? 'dim':'';
    const hl=activeT && tr.T===activeT ? 'hl':'';
    const col=colorT(tr.T, legend);
    const fill=a>=0? 'var(--good)':'var(--bad)';
    // color by T#, but tint intensity via height; border-left hint via bg mix
    const style=`background:linear-gradient(${a>=0?'180deg':'0deg'}, ${col}, ${fill})`;
    const bar=`<i style="height:${pct}%;${style}"></i>`;
    return `<div class="col ${dim} ${hl}" data-t="${esc(tr.T)}" title="t${tr.turn} ${tr.T} A=${fmt(a)} A_S=${fmt(tr.A_S)}">
      <div class="up">${a>=0?bar:''}</div>
      <div class="dn">${a<0?bar:''}</div>
    </div>`;
  }).join('');
  return `<div class="chart">
    <div class="chart-label"><span>GiGPO A · zero mid · scale ±${maxAbs.toFixed(2)}</span>
      <span class="mono">μA ${fmt(meanA)}</span></div>
    <div style="position:relative">
      <div class="zero"></div>
      <div class="chart-bars">${bars}</div>
    </div>
  </div>`;
}

function render(){
  const tabs=document.getElementById('tabs');
  tabs.innerHTML=report.cases.map((c,i)=>`<button class="tab ${i===active?'active':''}" data-i="${i}">${esc((c.inst||'').split('/').pop().slice(0,28))} · ${c.gigpo.n_solved}/${c.gigpo.n_traj}</button>`).join('');
  tabs.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{active=+b.dataset.i; activeT=null; render();});

  const c=report.cases[active];
  const g=c.gigpo; const an=c.analysis||{};
  const legend=g.legend||[];
  const scale=Math.max(0.2, ...g.trajectories.flatMap(t=>t.turns.map(x=>Math.abs(x.A||0))),
    ...g.trajectories.flatMap(t=>(t.llm?.turns||[]).map(x=>Math.abs(x.advantage||0))));

  const legendHtml=`<div class="legend">${legend.slice(0,16).map(x=>{
    const on=activeT===x.T?'outline:2px solid #fff;':'';
    return `<span style="background:${colorT(x.T,legend)};${on}" data-t="${esc(x.T)}">${esc(x.T)} ×${x.count}</span>`;
  }).join('')}</div>`;

  const stats=`<div class="stats">
    <div class="card"><b>GiGPO</b><br/>${esc(an.headline||'')}<br/>
      mean A solved=<b class="${cls(an.mean_A_solved)}">${fmt(an.mean_A_solved)}</b>
      failed=<b class="${cls(an.mean_A_failed)}">${fmt(an.mean_A_failed)}</b>
    </div>
    <div class="card"><b>LLM</b><br/>
      |r−r̄|μ=<b>${fmt(an.llm_abs_residual)}</b>
      · spinish≈<b>${an.llm_spinish_rate==null?'—':(100*an.llm_spinish_rate).toFixed(0)+'%'}</b>
      · turns=<b>${an.llm_n_turns||0}</b><br/>
      同组内 A_S 有方差(折扣) σμ=<b>${fmt(an.same_T_A_S_std_mean)}</b>
    </div>
  </div>`;

  const cols=g.trajectories.map(t=>{
    const llmTurns=t.llm?.turns||[];
    const llmBy={}; llmTurns.forEach(x=>llmBy[x.turn]=x);
    const turnsHtml=t.turns.map(tr=>{
      const dim=activeT && tr.T!==activeT;
      const L=llmBy[tr.turn]||{};
      return `<div class="turn ${dim?'dim':''}" data-t="${esc(tr.T)}">
        <div class="row">
          <span class="badge">t${tr.turn}</span>
          <span class="badge gid" style="background:${colorT(tr.T,legend)}">${esc(tr.T)}</span>
          ${tr.offload?'<span class="badge" style="background:#5a4510;color:#ffd27a">offload</span>':''}
        </div>
        <div class="row mono" style="margin-top:3px">
          <span class="${cls(tr.A)}">GiGPO A ${fmt(tr.A)}</span>${bar(tr.A,scale)}
        </div>
        <div class="row mono" style="color:#9aa3b2">A_E=${fmt(tr.A_E)} A_S=${fmt(tr.A_S)} G=${fmt(tr.G,4)}</div>
        <div class="row mono" style="margin-top:2px">
          <span class="${cls(L.advantage)}" style="color:#e3b341">LLM A_t ${fmt(L.advantage)}</span>
          <span style="color:#9aa3b2">r=${L.score!=null?Number(L.score).toFixed(3):'—'} r−r̄=${fmt(L.residual)}</span>
        </div>
        ${L.reason?`<div style="color:#c9d4e8;margin-top:2px">${esc(L.reason)}</div>`:''}
        ${tr.cmd?`<div class="cmd">${esc(tr.cmd)}</div>`:''}
      </div>`;
    }).join('');
    return `<div class="traj"><div class="traj-head">
      <h3>#${t.traj_index} ${t.solved?'✓':'✗'} R=${t.R.toFixed(3)} A_E=${fmt(t.A_E)} · ${esc(t.agent||'')} · ${t.n_turns}t
        <div style="font-weight:400;color:#9aa3b2">LLM mean_r=${t.llm?.mean_r!=null?Number(t.llm.mean_r).toFixed(3):'—'} · ${esc(t.llm?.judge_mode||'')}</div></h3>
      ${gigpoChart(t.turns, legend, scale)}
    </div>${turnsHtml}</div>`;
  }).join('');

  document.getElementById('root').innerHTML=`<div class="case active">
    <div class="note"><b>${esc(c.inst)}</b> · r${c.rollout}/g${c.group} · ${esc(c.note||'')}</div>
    ${legendHtml}${stats}<div class="grid">${cols}</div>
  </div>`;

  document.querySelectorAll('.legend span').forEach(el=>el.onclick=()=>{
    const t=el.getAttribute('data-t'); activeT=(activeT===t)?null:t; render();
  });
  document.querySelectorAll('.turn, .chart-bars .col').forEach(el=>el.onclick=()=>{
    const t=el.getAttribute('data-t'); activeT=(activeT===t)?null:t; render();
  });
}
render();
</script></body></html>"""
    return tpl.replace("__PAYLOAD__", payload)



def write_analysis_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# GiGPO vs LLM judge · 对比分析",
        "",
        f"- run: `{report['run_dir']}`",
        f"- GiGPO: `{report['formula_gigpo']}`",
        f"- LLM: `{report['formula_llm']}`",
        f"- judge: `{'heuristic' if report.get('heuristic') else report.get('model')}`",
        "",
        "## Case 一览",
        "",
        "| instance | solve | |A_S|μ | A_S zero | early→late A_S | sameT σ | |r−r̄|μ | spinish |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for c in report["cases"]:
        a = c.get("analysis") or {}
        g = c["gigpo"]
        lines.append(
            f"| {c['inst']} | {g['n_solved']}/{g['n_traj']} | {a.get('abs_A_S',0):.3f} | "
            f"{100*(a.get('A_S_zero_rate') or 0):.0f}% | "
            f"{(a.get('A_S_early') or 0):+.3f}→{(a.get('A_S_late') or 0):+.3f} | "
            f"{(a.get('same_T_A_S_std_mean') or 0):.3f} | "
            f"{(a.get('llm_abs_residual') if a.get('llm_abs_residual') is not None else float('nan')):.3f} | "
            f"{100*(a.get('llm_spinish_rate') or 0):.0f}% |"
        )
    lines += [
        "",
        "## 解读要点",
        "",
        "1. **同组有折扣**：`sameT σ` > 0 说明同 T# 内逐步 A_S 不同（越靠后通常越高），不是组级广播。",
        "2. **A_S early→late**：若 early 低、late 高，存在位置偏置；LLM 残差不应系统性随位置单调。",
        "3. **混合成败组**：GiGPO 主要靠 A_E + 同意图桶相对 G；LLM 能在失败长轨内标空转/有害步。",
        "4. **全对/全错**：A_E≈0 时 GiGPO 只剩 A_S；全错时 G=0 居多，A_S 信息弱。",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--pick", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--w", type=float, default=W)
    p.add_argument("--heuristic", action="store_true")
    p.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", judge.DEFAULT_BASE_URL))
    p.add_argument("--api-key", default=judge.DEFAULT_API_KEY)
    p.add_argument("--model", default=os.environ.get("DASHSCOPE_MODEL", judge.DEFAULT_MODEL))
    p.add_argument("--normalize", default="mean_to_outcome", choices=("raw", "sum_to_outcome", "mean_to_outcome"))
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--text-limit", type=int, default=420)
    args = p.parse_args()

    if not args.heuristic and not args.api_key:
        print("WARN: no DASHSCOPE_API_KEY — falling back to --heuristic for LLM side", flush=True)
        args.heuristic = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = run_cases(args)
    json_path = args.out_dir / "compare.json"
    html_path = args.out_dir / "compare.html"
    md_path = args.out_dir / "analysis.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    write_analysis_md(report, md_path)
    print(f"\nwrote {json_path}")
    print(f"wrote {html_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
