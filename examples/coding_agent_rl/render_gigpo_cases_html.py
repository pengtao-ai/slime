#!/usr/bin/env python3
"""Render GiGPO matched + singleton step-groups as a standalone HTML report.

Combines:
  - matched cases (size >= 2, cross-sample): from dump_gigpo_matched_cases_all_members
  - singleton cases (size == 1, "无法聚合"): sampled here

Outputs a single self-contained HTML file (no external assets).

Usage:
  PYTHONPATH=. python examples/coding_agent_rl/render_gigpo_cases_html.py \\
    --rollout runs/.../rollout_dumps/rollout_0.pt \\
    --out    runs/.../gigpo_cases.html \\
    --max-matched 8 --max-singletons 12
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


def _load_samples(path: Path) -> list[dict[str, Any]]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return list(d.get("samples") or [])


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
                    "assistant_preview": msgs[ai][1][:400],
                    "reward": float(s.get("reward") or 0.0),
                    "solved": float(md.get("solved", 1.0 if md.get("grading_solved") else 0.0) or 0.0),
                    "offload_count": int(stats.get("offload_count") or 0),
                    "response_length": int(s.get("response_length") or 0),
                    "group_index": s.get("group_index"),
                    "instance_id": instance_id,
                }
            )
    return rows


def cluster(rows: list[dict[str, Any]]) -> dict[tuple[Any, str], list[dict[str, Any]]]:
    by = defaultdict(list)
    for r in rows:
        by[(r["group_index"], r["anchor_obs"])].append(r)
    return by


def _tool_of(anchor: str) -> str:
    try:
        return (json.loads(anchor).get("obs") or ["?"])[0]
    except json.JSONDecodeError:
        return "?"


def _is_init(anchor: str) -> bool:
    try:
        return (json.loads(anchor).get("obs") or ["?"])[0] == "__init__"
    except json.JSONDecodeError:
        return False


def pick_cases(
    by: dict[tuple[Any, str], list[dict[str, Any]]],
    *,
    max_matched: int,
    max_singletons: int,
    exclude_init: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matched: list[dict[str, Any]] = []
    singletons: list[dict[str, Any]] = []

    # Matched: size >= 2, prefer cross-sample and larger size.
    items = sorted(by.items(), key=lambda kv: (-len({m["sample_index"] for m in kv[1]}), -len(kv[1])))
    for (gidx, anchor), members in items:
        if len(members) < 2:
            continue
        if exclude_init and _is_init(anchor):
            continue
        distinct = {m["sample_index"] for m in members}
        if len(distinct) < 2:
            continue
        matched.append(
            {
                "group_index": gidx,
                "anchor": anchor,
                "tool": _tool_of(anchor),
                "size": len(members),
                "n_distinct_samples": len(distinct),
                "instance_id": members[0]["instance_id"],
                "members": sorted(members, key=lambda m: (m["sample_index"], m["turn_index"])),
            }
        )
        if len(matched) >= max_matched:
            break

    # Singletons: size == 1, prefer non-init, mix of tools and instances.
    singleton_items = [
        ((gidx, anchor), members)
        for (gidx, anchor), members in by.items()
        if len(members) == 1 and not (exclude_init and _is_init(anchor))
    ]
    # Spread by instance: sort by (instance_id, tool) and stride-sample.
    singleton_items.sort(key=lambda kv: (kv[1][0]["instance_id"], _tool_of(kv[0][1])))
    step = max(1, len(singleton_items) // max(max_singletons, 1))
    for i in range(0, len(singleton_items), step):
        (gidx, anchor), members = singleton_items[i]
        singletons.append(
            {
                "group_index": gidx,
                "anchor": anchor,
                "tool": _tool_of(anchor),
                "size": 1,
                "n_distinct_samples": 1,
                "instance_id": members[0]["instance_id"],
                "members": members,
            }
        )
        if len(singletons) >= max_singletons:
            break

    return matched, singletons


def _fmt_calls(calls: list[dict[str, Any]] | None) -> str:
    if not calls:
        return "<span class='muted'>—</span>"
    parts = []
    for c in calls:
        name = c.get("name") or "?"
        inp = c.get("input") or {}
        # Compact: show file_path / command first
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
    # Show path/command + short old_string preview for Edit
    out = []
    if "file_path" in inp:
        out.append(f"file_path=<span class='tv'>{html.escape(str(inp['file_path']))}</span>")
    if "command" in inp:
        out.append(f"cmd=<span class='tv'>{html.escape(str(inp['command'])[:160])}</span>")
    if "old_string" in inp:
        out.append(f"old=<span class='tv'>{html.escape(str(inp['old_string'])[:80])}</span>")
    if "new_string" in inp:
        out.append(f"new=<span class='tv'>{html.escape(str(inp['new_string'])[:80])}</span>")
    if "pattern" in inp:
        out.append(f"pattern=<span class='tv'>{html.escape(str(inp['pattern']))}</span>")
    return ", ".join(out) if out else html.escape(json.dumps(inp, ensure_ascii=False)[:200])


def render_member(m: dict[str, Any]) -> str:
    solved_cls = "yes" if m["solved"] > 0 else "no"
    solved_txt = "✓" if m["solved"] > 0 else "✗"
    offload_badge = f"<span class='badge off'>offload×{m['offload_count']}</span>" if m["offload_count"] else ""
    prev = m.get("prev_tool_call") or {}
    prev_name = prev.get("name") if prev else None
    prev_inp = prev.get("input") if prev else None
    return f"""
    <div class="member">
      <div class="m-head">
        <span class="m-id">sample {m['sample_index']} · turn {m['turn_index']}</span>
        <span class="badge {solved_cls}">solved {solved_txt}</span>
        <span class="badge R">R={m['reward']:.3f}</span>
        {offload_badge}
        <span class="badge len">resp {m['response_length']}</span>
      </div>
      <div class="m-row"><span class="lbl">prev obs</span>
        <span class="tn">{html.escape(str(prev_name))}</span>({_fmt_input(prev_inp)})
      </div>
      <div class="m-row"><span class="lbl">prev resp</span>
        <pre class="resp">{html.escape((m.get('prev_tool_response_preview') or '')[:300])}</pre>
      </div>
      <div class="m-row"><span class="lbl">this action</span>
        {_fmt_calls(m.get('this_turn_tool_calls'))}
      </div>
    </div>"""


def render_case(c: dict[str, Any], idx: int, *, kind: str) -> str:
    members_html = "".join(render_member(m) for m in c["members"])
    anchor_short = c["anchor"][:200] + ("…" if len(c["anchor"]) > 200 else "")
    return f"""
  <section class="case {kind}">
    <h3>#{idx} [{kind}] tool={html.escape(c['tool'])} size={c['size']} samples={c['n_distinct_samples']}
      <span class="inst">{html.escape(c['instance_id'])}</span>
      <span class="gidx">g={c['group_index']}</span>
    </h3>
    <details class="anchor"><summary>anchor_obs</summary><code>{html.escape(anchor_short)}</code></details>
    <div class="members">{members_html}</div>
  </section>"""


CSS = """
body { font: 13px/1.5 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
       margin: 0; background: #0d1117; color: #c9d1d9; }
header { padding: 14px 20px; background: #161b22; border-bottom: 1px solid #30363d; position: sticky; top: 0; z-index: 1; }
header h1 { margin: 0 0 4px 0; font-size: 16px; }
header .sub { color: #8b949e; font-size: 12px; }
.tabs { display: flex; gap: 8px; margin-top: 10px; }
.tabs button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px;
  padding: 4px 12px; cursor: pointer; font-size: 12px; }
.tabs button.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
main { padding: 16px 20px 60px; }
h2 { font-size: 14px; margin: 18px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #21262d; color: #8b949e; }
.case { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 14px; margin: 10px 0; }
.case h3 { margin: 0 0 8px 0; font-size: 13px; font-weight: 600; }
.case.matched { border-left: 3px solid #2ea043; }
.case.singleton { border-left: 3px solid #f85149; }
.case h3 .inst { color: #79c0ff; font-weight: 400; margin-left: 8px; }
.case h3 .gidx { color: #8b949e; font-weight: 400; margin-left: 6px; }
.anchor { margin: 4px 0 8px; }
.anchor summary { cursor: pointer; color: #8b949e; font-size: 12px; }
.anchor code { color: #d2a8ff; word-break: break-all; font-size: 11px; }
.members { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 8px; }
.member { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 8px 10px; }
.m-head { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 6px; }
.m-id { font-weight: 600; color: #e6edf3; }
.m-row { margin: 3px 0; font-size: 12px; }
.lbl { display: inline-block; min-width: 64px; color: #8b949e; }
.tn { color: #79c0ff; font-weight: 600; }
.tv { color: #d2a8ff; }
.muted { color: #6e7681; }
.badge { font-size: 11px; padding: 1px 6px; border-radius: 10px; background: #21262d; color: #c9d1d9; }
.badge.yes { background: #1b3a23; color: #56d364; }
.badge.no { background: #3a1b1b; color: #f85149; }
.badge.R { background: #1f2937; color: #79c0ff; }
.badge.off { background: #3b2e10; color: #e3b341; }
.badge.len { background: #21262d; color: #8b949e; }
.resp { margin: 2px 0 0; background: #010409; border: 1px solid #21262d; border-radius: 4px;
  padding: 6px; font-size: 11px; color: #b1bac4; white-space: pre-wrap; max-height: 120px; overflow: auto; }
.hidden { display: none; }
"""


def render_html(
    *,
    rollout_id: int,
    rollout_path: str,
    matched: list[dict[str, Any]],
    singletons: list[dict[str, Any]],
    n_samples: int,
    n_rows: int,
    n_matched_groups: int,
    n_singleton_groups: int,
) -> str:
    matched_html = "".join(render_case(c, i + 1, kind="matched") for i, c in enumerate(matched))
    singleton_html = "".join(render_case(c, i + 1, kind="singleton") for i, c in enumerate(singletons))
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>GiGPO groups · rollout {rollout_id}</title>
<style>{CSS}</style></head>
<body>
<header>
  <h1>GiGPO step-group 入组情况</h1>
  <div class="sub">rollout {rollout_id} · {rollout_path} · samples={n_samples} step_rows={n_rows}
    · matched_groups={n_matched_groups} singleton_groups={n_singleton_groups}
    · 显示: {len(matched)} matched + {len(singletons)} singleton</div>
  <div class="tabs">
    <button class="active" onclick="showTab('matched',this)">可聚合 ({len(matched)})</button>
    <button onclick="showTab('singleton',this)">无法聚合 ({len(singletons)})</button>
  </div>
</header>
<main>
  <div id="matched"><h2>可聚合 (size ≥ 2, 跨 sample)</h2>{matched_html}</div>
  <div id="singleton" class="hidden"><h2>无法聚合 (singleton, 只有一条 traj 经历该观测)</h2>{singleton_html}</div>
</main>
<script>
function showTab(name, btn) {{
  document.getElementById('matched').classList.toggle('hidden', name !== 'matched');
  document.getElementById('singleton').classList.toggle('hidden', name !== 'singleton');
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}}
</script>
</body></html>"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-matched", type=int, default=8)
    p.add_argument("--max-singletons", type=int, default=12)
    p.add_argument("--exclude-init", action="store_true")
    p.add_argument("--hf-checkpoint", default="/workspace/models/pyromind/PyroDash-4B-SFT-07313")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True, local_files_only=True)
    data = torch.load(args.rollout, map_location="cpu", weights_only=False)
    rollout_id = int(data.get("rollout_id") or args.rollout.stem.split("_")[1])
    samples = list(data.get("samples") or [])
    rows = build_rows(samples, tok=tok)
    by = cluster(rows)

    n_matched_groups = sum(1 for v in by.values() if len(v) >= 2)
    n_singleton_groups = sum(1 for v in by.values() if len(v) == 1)

    matched, singletons = pick_cases(
        by,
        max_matched=args.max_matched,
        max_singletons=args.max_singletons,
        exclude_init=args.exclude_init,
    )

    html_str = render_html(
        rollout_id=rollout_id,
        rollout_path=str(args.rollout),
        matched=matched,
        singletons=singletons,
        n_samples=len(samples),
        n_rows=len(rows),
        n_matched_groups=n_matched_groups,
        n_singleton_groups=n_singleton_groups,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_str, encoding="utf-8")
    print(
        f"wrote {args.out}: {len(matched)} matched + {len(singletons)} singletons "
        f"(rollout {rollout_id}, {len(rows)} rows, {len(samples)} samples)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
