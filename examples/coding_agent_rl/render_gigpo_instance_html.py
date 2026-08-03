#!/usr/bin/env python3
"""Render all 8 trajectories of one instance as HTML, showing per-step grouping.

Each SLM turn is a row. Steps with the same anchor_obs (within the same
group_index) get the same group color → you can see which steps matched
across the 8 siblings and which are singletons.

Usage:
  PYTHONPATH=. python examples/coding_agent_rl/render_gigpo_instance_html.py \\
    --rollout runs/.../rollout_dumps/rollout_0.pt \\
    --instance adamtheturtle_sybil-extras_pr296 \\
    --out    runs/.../gigpo_instance_sybil_pr296.html
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

# 12 distinct-ish colors for group ids.
GROUP_COLORS = [
    "#1f6feb", "#2ea043", "#d2a8ff", "#e3b341", "#f85149", "#79c0ff",
    "#56d364", "#db61dd", "#ffa657", "#8b949e", "#79c0ff", "#a371f7",
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
                    "branch_key": t["branch_key"],
                    "prev_tool_call": (
                        {"name": prev_tool_name, "input": prev_tool_input}
                        if prev_tool_name
                        else None
                    ),
                    "prev_tool_response_preview": (prev_tool_result or "")[:500],
                    "this_turn_tool_calls": [
                        {"name": n, "input": inp} for n, inp in this_calls[:4]
                    ],
                    "assistant_preview": msgs[ai][1][:600],
                    "reward": float(s.get("reward") or 0.0),
                    "solved": float(md.get("solved", 1.0 if md.get("grading_solved") else 0.0) or 0.0),
                    "offload_count": int(stats.get("offload_count") or 0),
                    "response_length": int(s.get("response_length") or 0),
                    "group_index": s.get("group_index"),
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
    """Map anchor_obs → stable color id (only for matched anchors, size>=2)."""
    by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_anchor[r["anchor_obs"]].append(r)
    # Only color matched groups (size>=2). Singletons get -1 (grey).
    anchor_to_gid: dict[str, int] = {}
    gid = 0
    for anchor, members in sorted(by_anchor.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(members) >= 2:
            anchor_to_gid[anchor] = gid
            gid += 1
    return anchor_to_gid


CSS = """
body { font: 12px/1.45 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
       margin: 0; background: #0d1117; color: #c9d1d9; }
header { padding: 14px 20px; background: #161b22; border-bottom: 1px solid #30363d; position: sticky; top: 0; z-index: 1; }
header h1 { margin: 0 0 4px 0; font-size: 15px; }
header .sub { color: #8b949e; font-size: 11px; }
.legend { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }
.legend .chip { font-size: 10px; padding: 1px 6px; border-radius: 8px; color: #fff; }
main { padding: 12px 16px 60px; }
.traj { background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin: 8px 0; }
.traj-head { padding: 6px 10px; background: #21262d; border-radius: 8px 8px 0 0; font-weight: 600; display: flex; gap: 8px; align-items: center; }
.traj-head .st { font-size: 11px; }
.step { display: grid; grid-template-columns: 36px 1fr; gap: 8px; padding: 5px 10px; border-top: 1px solid #21262d; }
.step .gcol { text-align: center; font-weight: 600; font-size: 11px; padding-top: 2px; border-radius: 4px; }
.step .body { font-size: 11px; }
.step .tn { color: #79c0ff; font-weight: 600; }
.step .tv { color: #d2a8ff; }
.step .muted { color: #6e7681; }
.step .gid { display: inline-block; min-width: 18px; padding: 0 4px; border-radius: 8px; font-size: 10px; color: #fff; margin-right: 4px; }
.badge { font-size: 10px; padding: 1px 5px; border-radius: 8px; background: #21262d; color: #c9d1d9; }
.badge.yes { background: #1b3a23; color: #56d364; }
.badge.no { background: #3a1b1b; color: #f85149; }
.badge.off { background: #3b2e10; color: #e3b341; }
.badge.len { background: #21262d; color: #8b949e; }
.resp { margin: 2px 0; background: #010409; border: 1px solid #21262d; border-radius: 4px;
  padding: 4px 6px; font-size: 10px; color: #b1bac4; white-space: pre-wrap; max-height: 80px; overflow: auto; }
.asst { margin-top: 3px; color: #8b949e; font-style: italic; max-height: 60px; overflow: auto; }
"""


def _chip(gid: int) -> str:
    if gid < 0:
        return "<span class='gid' style='background:#6e7681'>·</span>"
    color = GROUP_COLORS[gid % len(GROUP_COLORS)]
    return f"<span class='gid' style='background:{color}'>G{gid}</span>"


def _fmt_calls(calls: list[dict[str, Any]] | None) -> str:
    if not calls:
        return "<span class='muted'>—</span>"
    parts = []
    for c in calls:
        name = c.get("name") or "?"
        inp = c.get("input") or {}
        key = "command" if "command" in inp else ("file_path" if "file_path" in inp else None)
        if key:
            val = str(inp.get(key))[:120]
            parts.append(f"<span class='tn'>{html.escape(name)}</span>(<span class='tv'>{html.escape(val)}</span>)")
        else:
            parts.append(f"<span class='tn'>{html.escape(name)}</span>(…)")
    return ", ".join(parts)


def _fmt_input(inp: dict[str, Any] | None) -> str:
    if not inp:
        return "<span class='muted'>—</span>"
    out = []
    if "file_path" in inp:
        out.append(f"file=<span class='tv'>{html.escape(str(inp['file_path']))}</span>")
    if "command" in inp:
        out.append(f"cmd=<span class='tv'>{html.escape(str(inp['command'])[:140])}</span>")
    if "old_string" in inp:
        out.append(f"old=<span class='tv'>{html.escape(str(inp['old_string'])[:60])}</span>")
    if "new_string" in inp:
        out.append(f"new=<span class='tv'>{html.escape(str(inp['new_string'])[:60])}</span>")
    if "pattern" in inp:
        out.append(f"pat=<span class='tv'>{html.escape(str(inp['pattern']))}</span>")
    return ", ".join(out) if out else ""


def render_traj(
    sample_index: int,
    steps: list[dict[str, Any]],
    anchor_to_gid: dict[str, int],
) -> str:
    s0 = steps[0]
    solved_cls = "yes" if s0["solved"] > 0 else "no"
    solved_txt = "✓" if s0["solved"] > 0 else "✗"
    offload_badge = f"<span class='badge off'>off×{s0['offload_count']}</span>" if s0["offload_count"] else ""
    head = (
        f"<div class='traj-head'>"
        f"<span>sample {sample_index}</span>"
        f"<span class='st'>turns={len(steps)}</span>"
        f"<span class='badge {solved_cls}'>solved {solved_txt}</span>"
        f"<span class='badge'>R={s0['reward']:.3f}</span>"
        f"{offload_badge}"
        f"</div>"
    )
    rows_html = []
    for m in steps:
        gid = anchor_to_gid.get(m["anchor_obs"], -1)
        tool = _tool_of(m["anchor_obs"])
        prev = m.get("prev_tool_call") or {}
        prev_name = prev.get("name") if prev else None
        prev_inp = prev.get("input") if prev else None
        row = f"""
      <div class='step'>
        <div class='gcol' title='group {gid}'>{_chip(gid)}</div>
        <div class='body'>
          <div><span class='muted'>t{m['turn_index']}</span> {html.escape(tool)}
           {_fmt_calls(m.get('this_turn_tool_calls'))}
           <span class='badge len'>resp {m['response_length']}</span>
          </div>
          <div><span class='muted'>prev:</span> <span class='tn'>{html.escape(str(prev_name))}</span>({_fmt_input(prev_inp)})</div>
          <div class='resp'>{html.escape((m.get('prev_tool_response_preview') or '')[:200])}</div>
        </div>
      </div>"""
        rows_html.append(row)
    return f"<div class='traj'>{head}{''.join(rows_html)}</div>"


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
    # Legend: matched groups in size order.
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
            f"<span class='chip' style='background:{color}'>G{gid} {html.escape(tool)} ×{len(members)}</span>"
        )
    legend = " ".join(legend_parts) or "<span class='muted'>无成组</span>"

    trajs_html = "".join(
        render_traj(idx, steps, anchor_to_gid) for idx, steps in trajs
    )
    match_pct = (n_matched_steps / n_total_steps * 100) if n_total_steps else 0
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>GiGPO {html.escape(instance_id)} · rollout {rollout_id}</title>
<style>{CSS}</style></head>
<body>
<header>
  <h1>GiGPO 入组情况 — {html.escape(instance_id)}</h1>
  <div class="sub">rollout {rollout_id} · {rollout_path} · {len(trajs)} 条 traj ·
    成组 step {n_matched_steps}/{n_total_steps} ({match_pct:.0f}%)</div>
  <div class="legend">{legend}</div>
</header>
<main>
  <p style='color:#8b949e;font-size:11px'>每行 = 一个 SLM turn；左侧色块 = 该步所属 step-group（灰=未成组）。
  同色的 step 共享同一 anchor_obs，会互相算 step-level advantage。</p>
{trajs_html}
</main>
</body></html>"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--instance", required=True, help="instance_id to filter")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hf-checkpoint", default="/workspace/models/pyromind/PyroDash-4B-SFT-07313")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True, local_files_only=True)
    data = torch.load(args.rollout, map_location="cpu", weights_only=False)
    rollout_id = int(data.get("rollout_id") or args.rollout.stem.split("_")[1])
    samples = list(data.get("samples") or [])

    # Filter to the requested instance (across any group_index that has it).
    target = [s for s in samples if str((s.get("metadata") or {}).get("instance_id") or s.get("label") or "") == args.instance]
    if not target:
        print(f"ERROR: no samples with instance_id={args.instance}", file=sys.stderr)
        return 1

    rows = build_rows(target, tok=tok)
    anchor_to_gid = assign_group_ids(rows)

    # Group rows by sample_index, keep turn order.
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
