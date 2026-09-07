#!/usr/bin/env python3
"""LLM process-supervision credit assignment over coding-agent trajectories.

Reads ``runs/<exp>/rollout_dumps/rollout_*.pt``, keeps one sample per
``session_id`` (8 trajectories per prompt group), asks a judge LLM to score
each turn, and writes JSON + a self-contained HTML report.

Example::

    export DASHSCOPE_BASE_URL=http://208.64.254.189:8001/v1
    export DASHSCOPE_API_KEY=sk-...
    export DASHSCOPE_MODEL=deepseek-v4-flash-0731

    python examples/coding_agent_rl/llm_turn_credit_assign.py \\
      --run-dir runs/agent_offload_pyrodash4b_phase2_sft00902_00_prob_5k_offload_insert_20260906_171737 \\
      --rollout-id 0 \\
      --group-index 1 \\
      --out-dir /tmp/credit_assign_r0_g1

Dry-run (no API; heuristic scores for HTML layout)::

    python examples/coding_agent_rl/llm_turn_credit_assign.py \\
      --run-dir runs/.../171737 --rollout-id 0 --group-index 1 --heuristic
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "http://208.64.254.189:8001/v1")
DEFAULT_MODEL = os.environ.get("DASHSCOPE_MODEL", "deepseek-v4-flash-0731")
DEFAULT_API_KEY = (
    os.environ.get("DASHSCOPE_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
).strip()

JUDGE_SYSTEM = """你是 coding-agent 长轨迹的过程监督（process supervision）评判员。
给定一个软件工程任务、最终是否解题、以及按时间顺序的逐步动作摘要，请为**每一轮**分配奖励。

评分原则：
1. score ∈ [-1.0, 1.0]：正分=推进任务；0=中性/空转；负分=有害/误导/浪费。
2. 关注因果贡献，而不是“看起来忙不忙”。解题关键步骤应拿更高分；失败路径上的错误方向给负分。
3. 若该轮触发了 <|llm_offload|>（向大模型求助）：
   - 自己本可轻松完成却求助 → 偏低或负；
   - 卡住/高不确定时合理求助并随后推进 → 正；
   - 求助后仍无进展 / 忽略建议 → 偏低。
4. 最终 solved=false 时，多数轮次应偏低；仅对真正有信息增益的探索给小正分。
5. 必须对输入中的每一轮都打分；reason 用一句中文说明。

只输出一个 JSON 对象，不要 markdown 围栏：
{"turns":[{"turn":0,"score":0.3,"reason":"..."}, ...], "summary":"一两句总评"}
"""


def _trunc(text: Any, n: int) -> str:
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    s = s.replace("\r\n", "\n").strip()
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


def _tool_calls_brief(am: dict[str, Any] | None, limit: int = 400) -> str:
    if not isinstance(am, dict):
        return ""
    calls = am.get("tool_calls") or []
    if not calls:
        return ""
    parts: list[str] = []
    for c in calls[:4]:
        if not isinstance(c, dict):
            continue
        fn = c.get("function") if isinstance(c.get("function"), dict) else c
        name = (fn or {}).get("name") or "?"
        args = (fn or {}).get("arguments")
        parts.append(f"{name}({_trunc(args, 180)})")
    return _trunc("; ".join(parts), limit)


def _obs_after_turn(turn_costs: list[dict[str, Any]], i: int, limit: int = 350) -> str:
    """Tool / user observations that appeared after turn i's assistant message."""
    if i + 1 >= len(turn_costs):
        return ""
    hist_i = turn_costs[i].get("sft_history_messages") or []
    hist_next = turn_costs[i + 1].get("sft_history_messages") or []
    if not isinstance(hist_i, list) or not isinstance(hist_next, list):
        return ""
    # hist_i = messages before generating turn i; +1 assistant ≈ start of new obs.
    start = len(hist_i) + 1
    obs = hist_next[start:] if start < len(hist_next) else []
    chunks: list[str] = []
    for m in obs:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "?"
        chunks.append(f"[{role}] {_trunc(m.get('content') or '', limit)}")
    return _trunc("\n".join(chunks), limit * 2)


def extract_problem(sample: dict[str, Any]) -> str:
    prompt = sample.get("prompt") or []
    if isinstance(prompt, list) and prompt:
        last = prompt[-1]
        if isinstance(last, dict) and last.get("content"):
            return str(last["content"])
    # fallback: first user message in turn0 history
    tc = (sample.get("metadata") or {}).get("turn_costs") or []
    if tc:
        for m in tc[0].get("sft_history_messages") or []:
            if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                return str(m["content"])
    return ""


def compact_turns(sample: dict[str, Any], *, text_limit: int = 420) -> list[dict[str, Any]]:
    md = sample.get("metadata") or {}
    turn_costs = list(md.get("turn_costs") or [])
    turn_rewards = list(md.get("turn_rewards") or [])
    diffs = list(md.get("turn_git_diffs") or [])
    out: list[dict[str, Any]] = []
    for i, tc in enumerate(turn_costs):
        am = tc.get("sft_assistant_message") if isinstance(tc.get("sft_assistant_message"), dict) else {}
        reason = (am or {}).get("reasoning_content") or ""
        content = (am or {}).get("content") or ""
        raw = tc.get("slm_raw_output") or ""
        git = ""
        if i < len(diffs) and isinstance(diffs[i], dict):
            git = diffs[i].get("git_diff") or ""
        out.append(
            {
                "turn": i,
                "valid_offload": bool(tc.get("valid_offload")),
                "forced_offload": bool(tc.get("forced_offload")),
                "outside_think": bool(tc.get("outside_think")),
                "response_token_len": int(tc.get("response_token_len") or 0),
                "train_turn_reward": float(turn_rewards[i]) if i < len(turn_rewards) else None,
                "reasoning": _trunc(reason, text_limit),
                "content": _trunc(content, text_limit // 2),
                "tool_calls": _tool_calls_brief(am, text_limit),
                "raw_preview": _trunc(raw, min(220, text_limit)),
                "observation": _obs_after_turn(turn_costs, i, limit=text_limit // 2),
                "git_diff": _trunc(git, 500) if git else "",
            }
        )
    return out


def unique_trajectories(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One representative sample per session_id, preserving dump order."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for s in samples:
        sid = str(s.get("session_id") or (s.get("metadata") or {}).get("session_id") or "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(s)
    return out


def load_rollout(run_dir: Path, rollout_id: int) -> dict[str, Any]:
    path = run_dir / "rollout_dumps" / f"rollout_{rollout_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "samples" not in obj:
        raise ValueError(f"unexpected dump format: {path}")
    return obj


def group_samples(samples: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by[int(s.get("group_index", 0))].append(s)
    return dict(sorted(by.items()))


def build_judge_user_payload(
    *,
    problem: str,
    solved: bool,
    outcome_reward: float,
    instance_id: str,
    agent: str,
    turns: list[dict[str, Any]],
    problem_limit: int = 2500,
) -> str:
    slim_turns = []
    for t in turns:
        slim_turns.append(
            {
                "turn": t["turn"],
                "offload": t["valid_offload"],
                "forced_offload": t["forced_offload"],
                "reasoning": t["reasoning"],
                "content": t["content"],
                "tool_calls": t["tool_calls"],
                "observation": t["observation"],
                "git_diff": t["git_diff"],
            }
        )
    payload = {
        "instance_id": instance_id,
        "agent": agent,
        "solved": solved,
        "outcome_reward": outcome_reward,
        "n_turns": len(turns),
        "problem": _trunc(problem, problem_limit),
        "turns": slim_turns,
    }
    return (
        "请对下列轨迹做逐步奖励分配。\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _extract_json_obj(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"judge response is not JSON: {text[:400]}")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("judge JSON root must be object")
    return obj


def call_chat(
    messages: list[dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:600]
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    payload = json.loads(raw)
    msg = (payload.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or ""
    if not content and msg.get("reasoning_content"):
        content = str(msg.get("reasoning_content"))
    return str(content)


def heuristic_scores(turns: list[dict[str, Any]], *, solved: bool, outcome_reward: float) -> dict[str, Any]:
    """Offline fallback so HTML can be checked without an API key."""
    n = max(len(turns), 1)
    base = float(outcome_reward) if solved else min(0.0, float(outcome_reward))
    scored = []
    for t in turns:
        i = int(t["turn"])
        score = base * (0.4 + 0.6 * (i + 1) / n)
        if t.get("valid_offload"):
            score *= 0.85 if solved else 0.5
        if t.get("forced_offload") and not solved:
            score -= 0.05
        if not t.get("tool_calls") and not t.get("content") and not t.get("valid_offload"):
            score *= 0.2
        score = max(-1.0, min(1.0, float(score)))
        reason = "heuristic: late turns / offload / empty-action shaped"
        scored.append({"turn": i, "score": round(score, 4), "reason": reason})
    return {"turns": scored, "summary": "heuristic fallback (no LLM)", "mode": "heuristic"}


def normalize_scores(
    scored_turns: list[dict[str, Any]],
    *,
    n_turns: int,
    outcome_reward: float,
    mode: str,
) -> list[dict[str, Any]]:
    by_turn = {int(t["turn"]): t for t in scored_turns if "turn" in t and "score" in t}
    filled: list[dict[str, Any]] = []
    for i in range(n_turns):
        if i in by_turn:
            item = dict(by_turn[i])
            item["score"] = float(item["score"])
            item.setdefault("reason", "")
            filled.append(item)
        else:
            filled.append({"turn": i, "score": 0.0, "reason": "missing from judge"})

    if mode == "raw":
        return filled

    scores = [float(t["score"]) for t in filled]
    s = sum(scores)
    if mode == "sum_to_outcome":
        if abs(s) < 1e-8:
            # spread outcome uniformly
            u = float(outcome_reward) / max(n_turns, 1)
            for t in filled:
                t["score"] = u
                t["normalized"] = mode
            return filled
        scale = float(outcome_reward) / s
        for t in filled:
            t["score"] = float(t["score"]) * scale
            t["normalized"] = mode
        return filled

    if mode == "mean_to_outcome":
        mean = s / max(n_turns, 1)
        if abs(mean) < 1e-8:
            for t in filled:
                t["score"] = float(outcome_reward)
                t["normalized"] = mode
            return filled
        scale = float(outcome_reward) / mean
        for t in filled:
            t["score"] = float(t["score"]) * scale
            t["normalized"] = mode
        return filled

    raise ValueError(f"unknown normalize mode: {mode}")


def judge_trajectory(
    sample: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    text_limit: int,
    normalize: str,
    heuristic: bool,
) -> dict[str, Any]:
    md = sample.get("metadata") or {}
    turns = compact_turns(sample, text_limit=text_limit)
    solved = bool(md.get("grading_solved") if md.get("grading_solved") is not None else md.get("solved"))
    outcome = float(sample.get("reward") or 0.0)
    problem = extract_problem(sample)

    if heuristic or not api_key:
        judged = heuristic_scores(turns, solved=solved, outcome_reward=outcome)
    else:
        user = build_judge_user_payload(
            problem=problem,
            solved=solved,
            outcome_reward=outcome,
            instance_id=str(md.get("instance_id") or sample.get("label") or ""),
            agent=str(md.get("agent") or ""),
            turns=turns,
        )
        content = call_chat(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        judged = _extract_json_obj(content)
        judged["mode"] = "llm"
        judged["raw_response"] = content

    scored = normalize_scores(
        list(judged.get("turns") or []),
        n_turns=len(turns),
        outcome_reward=outcome,
        mode=normalize,
    )
    # attach compact turn context for HTML
    for t, c in zip(scored, turns):
        t["context"] = c

    off = md.get("offload_stats") or {}
    return {
        "session_id": sample.get("session_id") or md.get("session_id"),
        "instance_id": md.get("instance_id") or sample.get("label"),
        "agent": md.get("agent"),
        "protocol": md.get("protocol"),
        "solved": solved,
        "outcome_reward": outcome,
        "train_turn_rewards": list(md.get("turn_rewards") or []),
        "n_turns": len(turns),
        "offload_count": off.get("offload_count"),
        "forced_offload_count": off.get("forced_offload_count"),
        "judge_mode": judged.get("mode"),
        "judge_summary": judged.get("summary") or "",
        "llm_turn_rewards": scored,
        "normalize": normalize,
        "problem": problem,
    }


def render_html(report: dict[str, Any]) -> str:
    title = html.escape(str(report.get("title") or "LLM Turn Credit Assignment"))
    payload = json.dumps(report, ensure_ascii=False)
    # Escape for embedding inside <script type="application/json">
    payload_safe = payload.replace("<", "\\u003c")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Outfit:wght@600;700&family=Source+Sans+3:wght@400;600&display=swap');
:root {{
  --bg:#e9eef3; --ink:#142033; --muted:#5b6b7c; --line:#c5d0db;
  --panel:#f7fafc; --side:#122033; --accent:#0b6e6a; --good:#1a6b4a;
  --bad:#a33b3b; --warn:#8a5a12;
  --sans:"Source Sans 3","Noto Sans SC",sans-serif;
  --display:"Outfit",var(--sans); --mono:"IBM Plex Mono",ui-monospace,monospace;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:var(--sans); color:var(--ink); background:
  linear-gradient(135deg,#f4f8fb 0%,transparent 40%),
  radial-gradient(circle at 1px 1px,rgba(18,40,55,.07) 1px,transparent 0) var(--bg);
  background-size:auto,18px 18px; }}
.app {{ display:grid; grid-template-columns:17rem 1fr; min-height:100vh; }}
.side {{ background:var(--side); color:#c9d5e2; padding:1rem; overflow:auto; }}
.side h1 {{ font-family:var(--display); color:#fff; font-size:1.05rem; margin:0 0 .4rem; }}
.side .meta {{ font-family:var(--mono); font-size:.68rem; color:#7f91a6; line-height:1.45; }}
.nav-item {{ display:block; width:100%; text-align:left; margin:.25rem 0; padding:.45rem .55rem;
  border:1px solid #24364d; background:#18283c; color:#d5e0eb; border-radius:2px; cursor:pointer; }}
.nav-item.active {{ border-color:var(--accent); background:#0f3f3d; }}
.nav-item .id {{ font-family:var(--mono); font-size:.72rem; }}
.main {{ padding:1rem 1.25rem 2rem; overflow:auto; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:2px; padding:1rem; margin-bottom:1rem; }}
.row {{ display:flex; flex-wrap:wrap; gap:.6rem; }}
.stat {{ min-width:7.5rem; background:#eef3f7; border:1px solid var(--line); padding:.5rem .65rem; }}
.stat b {{ display:block; font-family:var(--mono); font-size:1rem; }}
.stat span {{ color:var(--muted); font-size:.75rem; }}
.tabs {{ display:flex; flex-wrap:wrap; gap:.35rem; margin:.6rem 0 1rem; }}
.tab {{ font-family:var(--mono); font-size:.75rem; padding:.35rem .55rem; border:1px solid var(--line);
  background:#fff; cursor:pointer; }}
.tab.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.tab.ok {{ box-shadow: inset 0 -2px 0 var(--good); }}
.tab.fail {{ box-shadow: inset 0 -2px 0 var(--bad); }}
.problem {{ white-space:pre-wrap; font-size:.85rem; max-height:10rem; overflow:auto;
  background:#eef3f7; border:1px solid var(--line); padding:.6rem; }}
.summary {{ color:var(--muted); margin:.4rem 0 .8rem; }}
.turn {{ border-top:1px solid var(--line); padding:.7rem 0; }}
.turn-head {{ display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; }}
.badge {{ font-family:var(--mono); font-size:.68rem; padding:.1rem .35rem; border:1px solid var(--line); background:#fff; }}
.badge.off {{ background:#d5ebe9; border-color:#7fb6b2; }}
.badge.forced {{ background:#f7e8c8; border-color:#d2a85a; }}
.bar-wrap {{ flex:1; min-width:10rem; height:.55rem; background:#dde5ec; position:relative; }}
.bar {{ position:absolute; top:0; bottom:0; }}
.bar.pos {{ left:50%; background:var(--good); }}
.bar.neg {{ right:50%; background:var(--bad); }}
.score {{ font-family:var(--mono); font-weight:500; min-width:4.2rem; }}
.score.pos {{ color:var(--good); }}
.score.neg {{ color:var(--bad); }}
.reason {{ margin:.35rem 0; font-size:.88rem; }}
.details {{ font-family:var(--mono); font-size:.72rem; color:var(--muted); white-space:pre-wrap;
  background:#f1f5f8; border:1px solid var(--line); padding:.45rem; max-height:9rem; overflow:auto; }}
.compare {{ width:100%; border-collapse:collapse; font-size:.82rem; }}
.compare th,.compare td {{ border:1px solid var(--line); padding:.35rem .45rem; text-align:left; }}
.compare th {{ background:#eef3f7; position:sticky; top:0; }}
.hidden {{ display:none; }}
</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <h1>Turn Credit</h1>
    <div class="meta" id="side-meta"></div>
    <div id="nav" style="margin-top:1rem"></div>
  </aside>
  <main class="main" id="main"></main>
</div>
<script id="report-data" type="application/json">{payload_safe}</script>
<script>
const report = JSON.parse(document.getElementById('report-data').textContent);
const nav = document.getElementById('nav');
const main = document.getElementById('main');
const sideMeta = document.getElementById('side-meta');
sideMeta.innerHTML = [
  'run: ' + (report.run_dir || ''),
  'rollout: ' + report.rollout_id,
  'normalize: ' + report.normalize,
  'judge: ' + report.model,
].map(x => '<div>'+esc(x)+'</div>').join('');

let activeGroup = 0;
let activeTraj = 0;

function esc(s) {{
  return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function scoreClass(x) {{ return Number(x) >= 0 ? 'pos' : 'neg'; }}
function barHtml(score) {{
  const s = Math.max(-1, Math.min(1, Number(score) || 0));
  const pct = Math.abs(s) * 50;
  if (s >= 0) return `<div class="bar-wrap"><div class="bar pos" style="width:${{pct}}%"></div></div>`;
  return `<div class="bar-wrap"><div class="bar neg" style="width:${{pct}}%"></div></div>`;
}}

function render() {{
  const groups = report.groups || [];
  nav.innerHTML = groups.map((g, i) => {{
    const n = (g.trajectories||[]).length;
    const sol = (g.trajectories||[]).filter(t => t.solved).length;
    return `<button class="nav-item ${{i===activeGroup?'active':''}}" data-i="${{i}}">
      <div class="id">group ${{g.group_index}}</div>
      <div>${{esc(g.instance_id||'')}}</div>
      <div class="id">${{sol}}/${{n}} solved</div>
    </button>`;
  }}).join('');
  nav.querySelectorAll('.nav-item').forEach(btn => btn.onclick = () => {{
    activeGroup = Number(btn.dataset.i); activeTraj = 0; render();
  }});

  const g = groups[activeGroup];
  if (!g) {{ main.innerHTML = '<div class="card">no group</div>'; return; }}
  const trajs = g.trajectories || [];
  const t = trajs[activeTraj] || trajs[0];
  if (!t) {{ main.innerHTML = '<div class="card">no trajectories</div>'; return; }}

  const tabs = trajs.map((tr, i) =>
    `<button class="tab ${{i===activeTraj?'active':''}} ${{tr.solved?'ok':'fail'}}" data-i="${{i}}">
      #${{i}} ${{tr.solved?'✓':'✗'}} r=${{Number(tr.outcome_reward).toFixed(3)}} T=${{tr.n_turns}} off=${{tr.offload_count??0}}
    </button>`
  ).join('');

  const maxT = Math.max(...trajs.map(x => x.n_turns||0), 0);
  let cmpRows = '';
  for (let i=0;i<maxT;i++) {{
    cmpRows += `<tr><td>T${{i}}</td>` + trajs.map(tr => {{
      const cell = (tr.llm_turn_rewards||[])[i];
      if (!cell) return '<td></td>';
      const sc = Number(cell.score);
      return `<td class="${{scoreClass(sc)}}" title="${{esc(cell.reason||'')}}">${{sc.toFixed(3)}}${{cell.context&&cell.context.valid_offload?' ★':''}}</td>`;
    }}).join('') + '</tr>';
  }}
  const cmpHead = '<tr><th>turn</th>' + trajs.map((_,i)=>`<th>#${{i}}</th>`).join('') + '</tr>';

  const turnsHtml = (t.llm_turn_rewards||[]).map(tr => {{
    const c = tr.context || {{}};
    const badges = [
      c.valid_offload ? '<span class="badge off">offload</span>' : '',
      c.forced_offload ? '<span class="badge forced">forced</span>' : '',
      c.train_turn_reward!=null ? `<span class="badge">train r=${{Number(c.train_turn_reward).toFixed(3)}}</span>` : '',
    ].join('');
    const detail = [
      c.reasoning && ('reasoning: ' + c.reasoning),
      c.content && ('content: ' + c.content),
      c.tool_calls && ('tools: ' + c.tool_calls),
      c.observation && ('obs: ' + c.observation),
      c.git_diff && ('diff: ' + c.git_diff),
    ].filter(Boolean).join('\\n\\n');
    return `<div class="turn">
      <div class="turn-head">
        <span class="badge">T${{tr.turn}}</span>
        <span class="score ${{scoreClass(tr.score)}}">${{Number(tr.score).toFixed(3)}}</span>
        ${{barHtml(tr.score)}}
        ${{badges}}
      </div>
      <div class="reason">${{esc(tr.reason||'')}}</div>
      ${{detail ? `<pre class="details">${{esc(detail)}}</pre>` : ''}}
    </div>`;
  }}).join('');

  main.innerHTML = `
    <div class="card">
      <div class="row">
        <div class="stat"><span>instance</span><b>${{esc(g.instance_id||'')}}</b></div>
        <div class="stat"><span>traj</span><b>#${{activeTraj}} / ${{trajs.length}}</b></div>
        <div class="stat"><span>solved</span><b>${{t.solved}}</b></div>
        <div class="stat"><span>outcome r</span><b>${{Number(t.outcome_reward).toFixed(4)}}</b></div>
        <div class="stat"><span>turns</span><b>${{t.n_turns}}</b></div>
        <div class="stat"><span>offload</span><b>${{t.offload_count??0}}</b></div>
      </div>
      <div class="tabs">${{tabs}}</div>
      <div class="summary">${{esc(t.judge_summary||'')}}</div>
      <details><summary>problem</summary><pre class="problem">${{esc(t.problem||'')}}</pre></details>
    </div>
    <div class="card">
      <strong>8 轨迹逐步分数对照</strong>
      <div style="max-height:16rem;overflow:auto;margin-top:.5rem">
        <table class="compare"><thead>${{cmpHead}}</thead><tbody>${{cmpRows}}</tbody></table>
      </div>
    </div>
    <div class="card">
      <strong>轨迹 #${{activeTraj}} · session ${{esc((t.session_id||'').slice(0,8))}}</strong>
      ${{turnsHtml}}
    </div>`;

  main.querySelectorAll('.tab').forEach(btn => btn.onclick = () => {{
    activeTraj = Number(btn.dataset.i); render();
  }});
}}
render();
</script>
</body>
</html>
"""


def score_group(
    group_index: int,
    samples: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trajs = unique_trajectories(samples)
    if args.max_trajs is not None:
        trajs = trajs[: max(0, args.max_trajs)]

    results: list[dict[str, Any] | None] = [None] * len(trajs)

    def _one(idx_sample: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        idx, sample = idx_sample
        t0 = time.time()
        md = sample.get("metadata") or {}
        try:
            out = judge_trajectory(
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
        except Exception as exc:  # noqa: BLE001 - keep other trajs going
            turns = compact_turns(sample, text_limit=args.text_limit)
            solved = bool(
                md.get("grading_solved")
                if md.get("grading_solved") is not None
                else md.get("solved")
            )
            outcome = float(sample.get("reward") or 0.0)
            fallback = heuristic_scores(turns, solved=solved, outcome_reward=outcome)
            scored = normalize_scores(
                list(fallback["turns"]),
                n_turns=len(turns),
                outcome_reward=outcome,
                mode=args.normalize,
            )
            for t, c in zip(scored, turns):
                t["context"] = c
            out = {
                "session_id": sample.get("session_id") or md.get("session_id"),
                "instance_id": md.get("instance_id") or sample.get("label"),
                "agent": md.get("agent"),
                "protocol": md.get("protocol"),
                "solved": solved,
                "outcome_reward": outcome,
                "train_turn_rewards": list(md.get("turn_rewards") or []),
                "n_turns": len(turns),
                "offload_count": (md.get("offload_stats") or {}).get("offload_count"),
                "forced_offload_count": (md.get("offload_stats") or {}).get(
                    "forced_offload_count"
                ),
                "judge_mode": "error_fallback",
                "judge_summary": f"judge failed: {exc}",
                "llm_turn_rewards": scored,
                "normalize": args.normalize,
                "problem": extract_problem(sample),
                "error": str(exc),
            }
        out["latency_s"] = round(time.time() - t0, 2)
        out["traj_index"] = idx
        return idx, out

    workers = max(1, int(args.concurrency))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, (i, s)) for i, s in enumerate(trajs)]
        for fut in concurrent.futures.as_completed(futs):
            idx, out = fut.result()
            results[idx] = out
            err = f" ERR={out.get('error')}" if out.get("error") else ""
            print(
                f"[group {group_index}] traj#{idx} "
                f"sid={(out.get('session_id') or '')[:8]} "
                f"solved={out.get('solved')} turns={out.get('n_turns')} "
                f"mode={out.get('judge_mode')} {out.get('latency_s')}s{err}",
                flush=True,
            )

    instance_id = ""
    for r in results:
        if r and r.get("instance_id"):
            instance_id = str(r["instance_id"])
            break
    return {
        "group_index": group_index,
        "instance_id": instance_id,
        "trajectories": [r for r in results if r is not None],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="Experiment run root containing rollout_dumps/")
    p.add_argument("--rollout-id", type=int, default=0)
    p.add_argument("--group-index", type=int, nargs="*", default=None, help="Only these groups (default: all)")
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument("--max-trajs", type=int, default=None, help="Cap trajectories per group (debug)")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--normalize", choices=("raw", "sum_to_outcome", "mean_to_outcome"), default="raw")
    p.add_argument("--heuristic", action="store_true", help="No LLM; synthetic scores for HTML")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--text-limit", type=int, default=420, help="Per-field truncation for judge prompt")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout", type=float, default=300.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    dump = load_rollout(run_dir, args.rollout_id)
    samples = list(dump.get("samples") or [])
    groups = group_samples(samples)

    selected = sorted(groups.keys())
    if args.group_index is not None:
        want = set(args.group_index)
        selected = [g for g in selected if g in want]
    if args.max_groups is not None:
        selected = selected[: max(0, args.max_groups)]
    if not selected:
        print("no groups selected", file=sys.stderr)
        return 2

    if not args.heuristic and not args.api_key:
        print(
            "ERROR: set DASHSCOPE_API_KEY / OPENAI_API_KEY, or pass --heuristic",
            file=sys.stderr,
        )
        return 2

    out_dir = args.out_dir
    if out_dir is None:
        tag = f"r{args.rollout_id}_g{'-'.join(map(str, selected[:5]))}"
        out_dir = run_dir / "credit_assign" / tag
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_groups = [
        score_group(gi, groups[gi], args=args) for gi in selected
    ]

    report = {
        "title": f"LLM Turn Credit · {run_dir.name} · rollout {args.rollout_id}",
        "run_dir": str(run_dir),
        "rollout_id": args.rollout_id,
        "normalize": args.normalize,
        "model": args.model if not args.heuristic else "heuristic",
        "base_url": args.base_url,
        "groups": scored_groups,
    }

    json_path = out_dir / "credit_assign.json"
    html_path = out_dir / "credit_assign.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
