#!/usr/bin/env python3
"""8 SWE cases × 3 models in the same layout as gigpo_llm_compare_phase2_sft03/compare.html.

GiGPO (matches training / ``gigpo_advantage``; per-step, not group-broadcast):
  A = A_E + w * A_S
  G_t = R · γ^{n−1−t}
  A_S = G_t − mean(G | same T# across DeepSeek/SFT/Qwen)

If R=0 (unsolved), G_t=0 for every turn → A_S is constant inside a T# on that traj.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ANALYSIS_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _ANALYSIS_DIR.parent
_REPO_ROOT = _EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(_ANALYSIS_DIR))
sys.path.insert(0, str(_REPO_ROOT))

import compare_gigpo_llm_cases as cmp  # noqa: E402
from examples.coding_agent_rl.gigpo_advantage import (  # noqa: E402
    DEFAULT_GIGPO_GAMMA,
    _intent_from_family,
    classify_tool_call,
)

GAMMA = DEFAULT_GIGPO_GAMMA
MODEL_ORDER = ["DeepSeek", "SFT", "Qwen"]
W = 1.0


def _cmd(tr: dict[str, Any]) -> str:
    cmds = tr.get("cmds") or []
    if cmds:
        return str(cmds[0])[:220]
    return str(tr.get("tool_calls") or "")[:220]


def _parse_tool_calls_blob(blob: str) -> list[tuple[str, str]]:
    """Parse judge ``tool_calls`` like ``bash: cmd; Edit: path`` into ``[(name, args), ...]``."""
    s = (blob or "").strip()
    if not s:
        return []
    parts = re.split(r"\s*;\s*(?=[A-Za-z_][\w\-]*\s*:)", s)
    out: list[tuple[str, str]] = []
    for p in parts:
        p = p.strip()
        m = re.match(r"^([A-Za-z_][\w\-]*)\s*:\s*(.*)$", p, flags=re.S)
        if m:
            out.append((m.group(1), m.group(2).strip()))
        elif p:
            out.append(("bash", p))
    return out


def group_key(tr: dict[str, Any]) -> tuple[str, str, str]:
    """T# via ``gigpo_advantage.classify_tool_call`` (same families as training)."""
    cmds = [str(c) for c in (tr.get("cmds") or []) if str(c).strip()]
    if cmds:
        name, args = "bash", cmds[0]
    else:
        parsed = _parse_tool_calls_blob(str(tr.get("tool_calls") or ""))
        if not parsed:
            return "其他 · 无 tool", "其他", "无 tool"
        name, args = parsed[0]
    family = classify_tool_call(name, args)
    intent = _intent_from_family(family, args)
    return f"{intent} · {family}", intent, family


def build_case(case: dict[str, Any], *, gamma: float, w: float) -> dict[str, Any]:
    trajs: list[dict[str, Any]] = []
    for name in MODEL_ORDER:
        m = case["models"].get(name)
        if not m:
            continue
        solved = bool(m.get("solved"))
        R = float(m.get("outcome_reward") if m.get("outcome_reward") is not None else (1.0 if solved else 0.0))
        items = list(m.get("llm_turn_rewards") or [])
        n = int(m.get("n_turns") or len(items))
        turns: list[dict[str, Any]] = []
        llm_turns: list[dict[str, Any]] = []
        scores = [float(x.get("score") or 0) for x in items]
        mean_r = statistics.mean(scores) if scores else 0.0
        for tr in items:
            gkey, intent, tool = group_key(tr)
            i = int(tr.get("turn", 0))
            G = R * (gamma ** max(0, n - 1 - i))
            residual = float(tr.get("residual") if tr.get("residual") is not None else (float(tr.get("score") or 0) - mean_r))
            turns.append(
                {
                    "turn": i,
                    "T": gkey,
                    "intent": intent,
                    "tool": tool,
                    "G": G,
                    "cmd": _cmd(tr),
                    "offload": bool(tr.get("offloaded")),
                }
            )
            llm_turns.append(
                {
                    "turn": i,
                    "score": float(tr.get("score") or 0),
                    "residual": residual,
                    "reason": tr.get("reason") or "",
                }
            )
        trajs.append(
            {
                "traj_index": len(trajs),
                "agent": name,
                "solved": solved,
                "R": R,
                "n_turns": n,
                "turns": turns,
                "llm": {
                    "judge_mode": m.get("judge_mode") or "llm",
                    "summary": m.get("judge_summary"),
                    "mean_r": mean_r,
                    "turns": llm_turns,
                },
            }
        )

    mean_R = statistics.mean(t["R"] for t in trajs) if trajs else 0.0
    for t in trajs:
        t["A_E"] = t["R"] - mean_R

    by_T: dict[str, list[float]] = defaultdict(list)
    for t in trajs:
        for tr in t["turns"]:
            by_T[tr["T"]].append(float(tr["G"]))
    t_bar = {k: statistics.mean(v) for k, v in by_T.items()}

    for t in trajs:
        ae = t["A_E"]
        for tr in t["turns"]:
            tr["A_E"] = ae
            tr["A_S"] = float(tr["G"]) - t_bar[tr["T"]]
            tr["A"] = ae + w * tr["A_S"]
        for lt in t["llm"]["turns"]:
            lt["a_s"] = ae
            lt["advantage"] = ae + float(lt["residual"])

    t_counts = Counter(tr["T"] for t in trajs for tr in t["turns"])
    legend = [
        {
            "T": k,
            "count": c,
            "n_traj": len({i for i, t in enumerate(trajs) if any(x["T"] == k for x in t["turns"])}),
        }
        for k, c in t_counts.most_common()
    ]
    gigpo = {
        "mean_R": mean_R,
        "gamma": gamma,
        "w": w,
        "n_traj": len(trajs),
        "n_solved": sum(1 for t in trajs if t["solved"]),
        "trajectories": trajs,
        "legend": legend,
    }
    return {
        "rollout": "swe8",
        "group": 0,
        "inst": case.get("inst"),
        "note": case.get("note") or "",
        "gigpo": gigpo,
        "analysis": cmp.analyze_case(gigpo),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("/workspace/work/mjy/traj_compare_judge/compare_judge.json"))
    p.add_argument("--out", type=Path, default=Path("/workspace/work/mjy/traj_compare_gigpo_llm.html"))
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--w", type=float, default=W)
    args = p.parse_args()

    src = json.loads(args.src.read_text())
    cases = [build_case(c, gamma=args.gamma, w=args.w) for c in src.get("cases") or []]
    report = {
        "title": "GiGPO vs LLM judge · SWE 8 cases",
        "run_dir": str(args.src),
        "formula_gigpo": "A=A_E+w*A_S；T#=intent·tool（gigpo_advantage 新映射）；A_S=G_t−mean(G|T#)；G_t=R·γ^{n−1−t}",
        "formula_llm": "A_t=A_E+(r_i−r̄)；r_i 来自已有 LLM judge",
        "gamma": args.gamma,
        "w": args.w,
        "heuristic": False,
        "model": src.get("model") or "llm-judge",
        "cases": cases,
    }
    args.out.write_text(cmp.render_html(report), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    for c in cases:
        a = c["analysis"]
        g = c["gigpo"]
        print(
            f"  {c['inst']}: {g['n_solved']}/{g['n_traj']} "
            f"sameTσ={a.get('same_T_A_S_std_mean'):.3f} |A_S|={a.get('abs_A_S'):.3f}"
        )


if __name__ == "__main__":
    main()
