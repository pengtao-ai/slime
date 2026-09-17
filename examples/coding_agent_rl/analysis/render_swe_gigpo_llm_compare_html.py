#!/usr/bin/env python3
"""8 SWE cases × 3 models in the same layout as gigpo_llm_compare_phase2_sft03/compare.html.

GiGPO (per-step, not group-broadcast):
  A = A_E + A_S + A_I
  G_t = R · γ^{n−1−t}
  A_S = G_t − mean(G | same T# across DeepSeek/SFT/Qwen)
  A_I = G_t − mean(G | same S# on this traj)
  S#  = edit-segment proxy (no git dump)

If R=0 (unsolved), G_t=0 for every turn → A_S is constant inside a T# on that traj.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import compare_gigpo_llm_cases as cmp  # noqa: E402
from render_compare_group_advantage_html import (  # noqa: E402
    GAMMA,
    MODEL_ORDER,
    _is_edit_turn,
    group_key,
)

W = 1.0


def _cmd(tr: dict[str, Any]) -> str:
    cmds = tr.get("cmds") or []
    if cmds:
        return str(cmds[0])[:220]
    return str(tr.get("tool_calls") or "")[:220]


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
        seg = 0
        llm_turns: list[dict[str, Any]] = []
        scores = [float(x.get("score") or 0) for x in items]
        mean_r = statistics.mean(scores) if scores else 0.0
        for tr in items:
            gkey, intent, tool = group_key(tr)
            i = int(tr.get("turn", 0))
            G = R * (gamma ** max(0, n - 1 - i))
            s_label = "<empty>" if seg == 0 else f"S{seg}"
            residual = float(tr.get("residual") if tr.get("residual") is not None else (float(tr.get("score") or 0) - mean_r))
            turns.append(
                {
                    "turn": i,
                    "T": gkey,
                    "intent": intent,
                    "tool": tool,
                    "S": s_label,
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
            if _is_edit_turn(tr, tool):
                seg += 1
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
        by_S: dict[str, list[float]] = defaultdict(list)
        for tr in t["turns"]:
            by_S[tr["S"]].append(float(tr["G"]))
        s_bar = {k: statistics.mean(v) for k, v in by_S.items()}
        ae = t["A_E"]
        for tr in t["turns"]:
            tr["A_E"] = ae
            tr["A_S"] = float(tr["G"]) - t_bar[tr["T"]]
            tr["A_I"] = float(tr["G"]) - s_bar[tr["S"]]
            tr["A"] = ae + w * (tr["A_S"] + tr["A_I"])
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
        "formula_gigpo": "A=A_E+A_S+A_I；A_S=G_t−mean(G|T#) 逐步非广播；G_t=R·γ^{n−1−t}。失败轨 R=0 ⇒ G=0 ⇒ 同T# 的 A_S 全相同",
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
