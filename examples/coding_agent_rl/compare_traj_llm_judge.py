#!/usr/bin/env python3
"""LLM-judge mini-swe trajectories (DeepSeek / SFT / Qwen) and emit a compare HTML.

Reuses scoring from ``score_miniswe_traj_html.py``. Every turn gets a score
(chunked judge calls). Also reports per-turn / final token advantage vs DeepSeek.
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
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import score_miniswe_traj_html as scorer  # noqa: E402

SLM_IN, SLM_OUT = 0.016, 0.23  # Qwen3.5-4B USD / MTok
LLM_IN, LLM_OUT = 0.44, 1.32  # DeepSeek-V4-Flash USD / MTok

CASES = [
    ("astropy__astropy-14508", "全对 ✓✓✓"),
    ("django__django-12262", "全对 ✓✓✓"),
    ("astropy__astropy-14182", "仅 SFT 对 ✗✓✗"),
    ("django__django-14122", "SFT/Qwen 错 ✓✗✗（SFT LimitsExceeded）"),
    ("sympy__sympy-14531", "SFT/Qwen 错 ✓✗✗（SFT LimitsExceeded）"),
    ("django__django-15268", "仅 Qwen 错 ✓✓✗（Qwen TimeExceeded）"),
    ("django__django-16631", "仅 Qwen 错 ✓✓✗"),
    ("django__django-10999", "全错 ✗✗✗"),
]

DEFAULT_RUNS = {
    "DeepSeek": Path(
        "/workspace/work/mjy/eval/swe-bench/output/runs_output/deepseek/"
        "deeepseek_all_nolimit_maxthinking"
    ),
    "SFT": Path(
        "/workspace/work/mjy/swe_run_offload/output/runs/"
        "a_sft_20260910_162110_c200_miniswe_offload_20filter_0"
    ),
    "Qwen": Path("/workspace/work/mjy/eval/swe-bench/output/runs_output/qwen/qwen_20"),
}

# SWE-bench harness report.json roots (resolved=true/false)
DEFAULT_VERIFIED = {
    "DeepSeek": Path(
        "/workspace/work/mjy/eval/swe-bench/output/verified_output/deepseek/"
        "deeepseek_all_nolimit_maxthinking/logs"
    ),
    "SFT": Path(
        "/workspace/work/mjy/swe_run_offload/output/verified/"
        "a_sft_20260910_162110_c200_miniswe_offload_20filter_0/logs"
    ),
    "Qwen": Path(
        "/workspace/work/mjy/eval/swe-bench/output/verified_output/qwen/"
        "qwen3.5_4b_all_nnnnnew/logs"
    ),
}


def call_chat_no_think(
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
        "chat_template_kwargs": {"thinking": False},
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


scorer.call_chat = call_chat_no_think  # type: ignore[method-assign]


def _cmds(msg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for a in (msg.get("extra") or {}).get("actions") or []:
        if isinstance(a, dict) and "command" in a:
            out.append(str(a["command"]))
    if out:
        return out
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        args = (tc.get("function") or {}).get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(args, dict) and "command" in args:
            out.append(str(args["command"]))
    return out


def _tag(cmd: str) -> str:
    low = cmd.lower()
    if re.search(r"\b(pytest|unittest|tox|runtests)\b", low):
        return "test"
    if "cat >" in low or "tee " in low or re.search(r"\b(sed\s+-i|apply_patch)\b", low):
        return "edit"
    if re.search(r"\b(grep|rg\b|find\b|git\s+grep)\b", low):
        return "search"
    if re.search(r"\b(cat\s|head\s|tail\s|sed\s+-n|nl\s|awk\s)\b", low):
        return "read"
    if "python -c" in low or "python - <<" in low or "python <<" in low:
        return "probe"
    if "git " in low:
        return "git"
    if "pip " in low or "conda " in low or "setup.py" in low:
        return "env"
    return "other"


def extract_turn_tokens(traj_path: Path) -> list[dict[str, Any]]:
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    msgs = list(data.get("messages") or [])
    rows: list[dict[str, Any]] = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") != "assistant":
            i += 1
            continue
        usage = {}
        extra = m.get("extra") or {}
        resp = extra.get("response") if isinstance(extra, dict) else None
        if isinstance(resp, dict) and isinstance(resp.get("usage"), dict):
            usage = resp["usage"]
        cmds = _cmds(m)
        j = i + 1
        rcs: list[int | None] = []
        while j < len(msgs) and msgs[j].get("role") == "tool":
            c = msgs[j].get("content") or ""
            mrc = re.search(r"<returncode>(-?\d+)</returncode>", c)
            rcs.append(int(mrc.group(1)) if mrc else None)
            j += 1
        rows.append(
            {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "cmds": cmds,
                "rcs": rcs,
                "tags": sorted({_tag(c) for c in cmds}) or ["other"],
            }
        )
        i = j if j > i + 1 else i + 1
    return rows


def _cmd_fingerprint(cmd: str) -> str:
    c = re.sub(r"\s+", " ", cmd.strip())
    c = re.sub(r"sed\s+-n\s+'?\d+,\d+p'?", "sed -n 'N,Mp'", c)
    c = re.sub(r"NR>=?\d+\s*&&\s*NR<=?\d+", "NR>=N && NR<=M", c)
    c = re.sub(r"head\s+-n?\s*\d+", "head -N", c)
    c = re.sub(r"tail\s+-n?\s*\d+", "tail -N", c)
    return c[:120]


def _file_key(cmd: str) -> str | None:
    m = re.findall(r"(?:/testbed/|\./)?([\w./-]+\.py)", cmd)
    return m[0] if m else None


def _is_read(cmd: str) -> bool:
    low = cmd.lower()
    if re.search(r"\b(pytest|unittest|tox|runtests)\b", low):
        return False
    if re.search(r"\bpython(?:3)?\s+(-m|-c)\b", low):
        return False
    return bool(re.search(r"(^|[;&]\s*)(cat\s+-n|cat\s|head\s|tail\s|sed\s+-n|nl\s|awk\s)", low))


def _is_edit(cmd: str) -> bool:
    low = cmd.lower()
    return bool(
        "cat >" in low
        or "tee " in low
        or re.search(r"\b(sed\s+-i|apply_patch|git\s+apply)\b", low)
        or re.search(r"open\([^)]+,\s*['\"]w", low)
        or "with open(" in low and ", 'w" in low
    )


def _is_verify_like(cmd: str) -> bool:
    if _is_edit(cmd):
        return False
    low = cmd.lower()
    if re.search(r"\b(pytest|unittest|tox|runtests)\b", low):
        return True
    if "reproduce" in low:
        return True
    # scripted repro / check (not an edit)
    if re.search(r"\bpython(?:3)?\s+\S+\.py\b", low):
        return True
    if "python -c" in low or "python - <<" in low or "python <<" in low:
        return True
    return False


def postprocess_scores(scored: dict[str, Any], token_rows: list[dict[str, Any]]) -> None:
    """Clamp LLM scores for repeats / rereads / repeated failures.

    Exemptions (do not clamp):
    - first verify-like cmd after an edit (one shot per edit generation)
    - same fingerprint transitioning fail → pass
    """
    from collections import Counter

    rewards = scored.get("llm_turn_rewards") or []
    seen_cmds: Counter[str] = Counter()
    fail_cmds: Counter[str] = Counter()
    last_fail: dict[str, bool] = {}
    last_file: str | None = None
    file_streak = 0
    edit_gen = 0
    verify_used_gen = -1
    n_adj = 0
    n_exempt = 0
    for i, item in enumerate(rewards):
        row = token_rows[i] if i < len(token_rows) else {}
        cmds = list(item.get("cmds") or row.get("cmds") or [])
        rcs = list(row.get("rcs") or [])
        raw = float(item.get("score_raw") if item.get("score_raw") is not None else item.get("score") or 0.0)
        item["score_raw"] = raw
        score = raw
        notes: list[str] = []

        if any(_is_edit(c) for c in cmds):
            edit_gen += 1

        has_fail = any(rc not in (0, None) for rc in rcs)
        all_ok = bool(rcs) and all(rc in (0, None) for rc in rcs)

        for cmd in cmds:
            fp = _cmd_fingerprint(cmd)
            seen_cmds[fp] += 1
            n = seen_cmds[fp]

            fail2pass = bool(last_fail.get(fp)) and all_ok and not has_fail and _is_verify_like(cmd)
            post_edit_first = (
                edit_gen > 0
                and _is_verify_like(cmd)
                and verify_used_gen < edit_gen
            )
            exempt = False
            if post_edit_first:
                verify_used_gen = edit_gen
                exempt = True
                notes.append("verify_exempt")
                n_exempt += 1
            elif fail2pass:
                exempt = True
                notes.append("fail2pass_exempt")
                n_exempt += 1

            if not exempt:
                if n == 2 and score > 0.0:
                    score = 0.0
                    notes.append("repeat2<=0")
                elif n >= 3 and score > -0.3:
                    score = -0.3
                    notes.append(f"repeat{n}<=-0.3")

            if _is_read(cmd):
                fk = _file_key(cmd)
                if fk and fk == last_file:
                    file_streak += 1
                    if not exempt:
                        if file_streak >= 2 and score > 0.0:
                            score = min(score, 0.0)
                            notes.append("reread<=0")
                        if file_streak >= 3 and score > -0.2:
                            score = min(score, -0.2)
                            notes.append("reread3<=-0.2")
                else:
                    last_file = fk
                    file_streak = 1 if fk else 0
            else:
                last_file = None
                file_streak = 0

            last_fail[fp] = has_fail

        if has_fail and cmds:
            fp = _cmd_fingerprint(cmds[0])
            fail_cmds[fp] += 1
            n = fail_cmds[fp]
            if n >= 2 and score > 0.0:
                score = 0.0
                notes.append("fail_repeat<=0")
            if n >= 3 and score > -0.3:
                score = -0.3
                notes.append("fail_repeat3<=-0.3")

        score = max(-1.0, min(1.0, float(score)))
        # strip old pp tags then re-apply
        reason = re.sub(r"\s*\[pp:[^\]]*\]", "", str(item.get("reason") or "")).strip()
        item.pop("postprocess", None)
        if abs(score - raw) > 1e-9 or any(x.endswith("_exempt") for x in notes):
            if abs(score - raw) > 1e-9:
                n_adj += 1
            # dedupe notes
            uniq: list[str] = []
            for n in notes:
                if n not in uniq:
                    uniq.append(n)
            item["postprocess"] = ",".join(uniq) if uniq else "clamped"
            tag = f"[pp:{item['postprocess']}]"
            item["reason"] = (reason + " " + tag).strip()
        else:
            item["reason"] = reason
        item["score"] = round(score, 4)
        item["rcs"] = rcs

    scored["n_postprocess_adjusted"] = n_adj
    scored["n_verify_exempt"] = n_exempt
    scores = [float(x.get("score") or 0.0) for x in rewards]
    tot = scored.setdefault("token_totals", {})
    tot["mean_score"] = round(sum(scores) / max(len(scores), 1), 4)
    tot["sum_score"] = round(sum(scores), 4)
    tot["pos_score"] = round(sum(max(s, 0.0) for s in scores), 4)
    scored["mean_r"] = tot["mean_score"]


def detect_spin_marks(scored: dict[str, Any]) -> None:
    from collections import Counter

    rewards = scored.get("llm_turn_rewards") or []
    seen_cmd: Counter[str] = Counter()
    last_file: str | None = None
    file_streak = 0
    for item in rewards:
        marks: list[str] = []
        for cmd in item.get("cmds") or []:
            key = re.sub(r"\s+", " ", cmd.strip())[:80]
            seen_cmd[key] += 1
            if seen_cmd[key] >= 3:
                marks.append("spin-repeat")
            if _is_read(cmd):
                fk = _file_key(cmd)
                if fk and fk == last_file:
                    file_streak += 1
                    if file_streak >= 2:
                        marks.append("spin-reread")
                else:
                    last_file = fk
                    file_streak = 1 if fk else 0
            else:
                last_file = None
                file_streak = 0
        out: list[str] = []
        for m in marks:
            if m not in out:
                out.append(m)
        item["spin"] = out
        item["spin_mark"] = out[0] if out else ""


def load_sft_offload(sft_root: Path) -> dict[int, list[dict[str, Any]]]:
    summary_path = sft_root / "adapter_stats" / "summary.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    by_turns: dict[int, list[dict[str, Any]]] = {}
    for sess in summary.get("sessions") or []:
        by_turns.setdefault(int(sess["turns"]), []).append(sess)
    return by_turns


def load_harness_resolved(verified_root: Path, instance_id: str) -> bool | None:
    """Return harness ``resolved`` from report.json, or None if missing."""
    if not verified_root.exists():
        return None
    hits = list(verified_root.rglob(f"{instance_id}/report.json"))
    if not hits:
        return None
    try:
        data = json.loads(hits[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        if instance_id in data and isinstance(data[instance_id], dict):
            if "resolved" in data[instance_id]:
                return bool(data[instance_id]["resolved"])
        if "resolved" in data:
            return bool(data["resolved"])
    return None


def attach_tokens(
    scored: dict[str, Any],
    *,
    model_name: str,
    token_rows: list[dict[str, Any]],
    sft_sessions: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    rewards = scored["llm_turn_rewards"]
    n = min(len(rewards), len(token_rows))
    offload = [None] * len(rewards)
    slm_ps = [0] * len(rewards)
    slm_cs = [0] * len(rewards)
    llm_ps = [0] * len(rewards)
    llm_cs = [0] * len(rewards)
    if model_name == "SFT":
        cands = sft_sessions.get(len(rewards), [])
        if len(cands) == 1:
            for t in cands[0].get("per_turn") or []:
                idx = int(t.get("turn") or 0)
                if 0 <= idx < len(rewards):
                    offload[idx] = bool(t.get("offloaded"))
                    slm_ps[idx] = int(t.get("slm_prompt_tokens") or 0)
                    slm_cs[idx] = int(t.get("slm_completion_tokens") or 0)
                    llm_ps[idx] = int(t.get("llm_prompt_tokens") or 0)
                    llm_cs[idx] = int(t.get("llm_completion_tokens") or 0)

    cum_tok = 0.0
    cum_cost = 0.0
    cum_score = 0.0
    tot_prompt = tot_comp = 0
    tot_slm_p = tot_slm_c = tot_llm_p = tot_llm_c = 0
    for i, item in enumerate(rewards):
        row = token_rows[i] if i < n else {}
        prompt = int(row.get("prompt_tokens") or 0)
        comp = int(row.get("completion_tokens") or 0)
        if model_name == "SFT" and (slm_ps[i] or llm_ps[i]):
            cost = (
                slm_ps[i] / 1e6 * SLM_IN
                + slm_cs[i] / 1e6 * SLM_OUT
                + llm_ps[i] / 1e6 * LLM_IN
                + llm_cs[i] / 1e6 * LLM_OUT
            )
            tok = slm_ps[i] + slm_cs[i] + llm_ps[i] + llm_cs[i]
            tot_slm_p += slm_ps[i]
            tot_slm_c += slm_cs[i]
            tot_llm_p += llm_ps[i]
            tot_llm_c += llm_cs[i]
        elif model_name == "Qwen":
            cost = prompt / 1e6 * SLM_IN + comp / 1e6 * SLM_OUT
            tok = prompt + comp
        else:
            cost = prompt / 1e6 * LLM_IN + comp / 1e6 * LLM_OUT
            tok = prompt + comp
        tot_prompt += prompt
        tot_comp += comp
        score = float(item.get("score") or 0)
        cum_tok += tok
        cum_cost += cost
        cum_score += score
        item["prompt_tokens"] = prompt
        item["completion_tokens"] = comp
        item["tokens"] = tok
        item["cost_usd"] = round(cost, 6)
        item["cum_tokens"] = int(cum_tok)
        item["cum_cost_usd"] = round(cum_cost, 6)
        item["cum_score"] = round(cum_score, 4)
        item["score_per_1k_tok"] = round(score / max(tok, 1) * 1000, 4)
        item["offloaded"] = offload[i]
        item["slm_prompt_tokens"] = slm_ps[i]
        item["slm_completion_tokens"] = slm_cs[i]
        item["llm_prompt_tokens"] = llm_ps[i]
        item["llm_completion_tokens"] = llm_cs[i]
        item["cmds"] = row.get("cmds") or []
        item["tags"] = row.get("tags") or []
        ctx = item.get("context") or {}
        item["tool_calls"] = ctx.get("tool_calls") or ""
        item["observation"] = ctx.get("observation") or ""
        item["reasoning"] = ctx.get("reasoning") or ""
        # drop bulky context for HTML payload
        item.pop("context", None)

    scored["token_totals"] = {
        "prompt_tokens": tot_prompt,
        "completion_tokens": tot_comp,
        "tokens": tot_prompt + tot_comp if model_name != "SFT" else int(cum_tok),
        "slm_prompt_tokens": tot_slm_p,
        "slm_completion_tokens": tot_slm_c,
        "llm_prompt_tokens": tot_llm_p,
        "llm_completion_tokens": tot_llm_c,
        "cost_usd": round(cum_cost, 4),
        "mean_score": round(cum_score / max(len(rewards), 1), 4),
        "sum_score": round(cum_score, 4),
        "pos_score": round(sum(max(float(x.get("score") or 0), 0) for x in rewards), 4),
    }
    return scored


def paint_advantages(models: dict[str, dict[str, Any]]) -> None:
    """A_t = r_i - mean(r) with A_s = 0 (independent trajs, no GRPO group)."""
    for rec in models.values():
        items = list(rec.get("llm_turn_rewards") or [])
        scores = [float(x.get("score") or 0.0) for x in items]
        mean_r = (sum(scores) / len(scores)) if scores else 0.0
        rec["mean_r"] = round(mean_r, 6)
        rec["a_s"] = 0.0
        for item, score in zip(items, scores):
            residual = float(score) - mean_r
            item["residual"] = round(residual, 6)
            item["a_s"] = 0.0
            item["advantage"] = round(residual, 6)


def render_html(report: dict[str, Any]) -> str:
    payload = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trajectory Compare + LLM Judge</title>
<style>
:root {{
  --bg:#0f1115; --panel:#171a21; --border:#2a3140; --text:#e7eaf0; --muted:#9aa3b2;
  --accent:#6ea8fe; --spin:#5a2a2a; --llm:#3a2f14; --llm-b:#ffb020;
  --good:#3dd68c; --bad:#ff7b72;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--text); }}
header {{ position:sticky; top:0; z-index:20; background:#0c0e12ee; border-bottom:1px solid var(--border); padding:12px 18px; }}
header h1 {{ margin:0 0 6px; font-size:18px; }}
header p {{ margin:0; color:var(--muted); font-size:13px; }}
.tabs, .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
.tab, .toolbar label {{ border:1px solid var(--border); background:#12151c; color:var(--text); padding:6px 10px; border-radius:8px; cursor:pointer; font-size:12px; }}
.tab.active {{ border-color:var(--accent); background:#182033; }}
.toolbar {{ color:var(--muted); align-items:center; }}
.toolbar label {{ display:flex; gap:6px; align-items:center; border:0; background:transparent; padding:0; }}
.case {{ display:none; padding:14px 18px 48px; }}
.case.active {{ display:block; }}
.note {{ margin:0 0 12px; padding:10px 12px; background:#1a2030; border-left:3px solid var(--accent); border-radius:0 8px 8px 0; font-size:13px; color:#c9d4e8; }}
.meta, .adv {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-bottom:12px; }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:10px 12px; }}
.card h3 {{ margin:0 0 6px; font-size:14px; }}
.stat {{ color:var(--muted); font-size:12px; line-height:1.45; }}
.stat b {{ color:var(--text); font-family: ui-monospace, Menlo, Consolas, monospace; }}
.pos {{ color:var(--good); }} .neg {{ color:var(--bad); }}
.grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; align-items:start; }}
.col {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; max-height:calc(100vh - 280px); overflow:auto; }}
.col h2 {{ position:sticky; top:0; margin:0; padding:10px 12px; background:#141821; border-bottom:1px solid var(--border); font-size:13px; z-index:5; }}
.turn {{ border-bottom:1px solid #222836; padding:8px 10px; font-size:12px; }}
.turn.llm {{ background:var(--llm); box-shadow: inset 3px 0 0 var(--llm-b); }}
.turn.low {{ background:#2a1616; }}
.turn-h {{ display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-bottom:4px; }}
.badge {{ font-size:10px; padding:1px 6px; border-radius:999px; border:1px solid var(--border); color:var(--muted); }}
.score {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-weight:600; min-width:3.6rem; }}
.bar-wrap {{ flex:1; min-width:6rem; height:6px; background:#2a3140; position:relative; border-radius:99px; overflow:hidden; }}
.bar {{ position:absolute; top:0; bottom:0; }}
.bar.p {{ left:50%; background:var(--good); }}
.bar.n {{ right:50%; background:var(--bad); }}
.cmd {{ font-family: ui-monospace, Menlo, Consolas, monospace; white-space:pre-wrap; word-break:break-word; background:#0e1218; border:1px solid #243044; border-radius:6px; padding:6px 8px; margin:4px 0; color:#d6deea; line-height:1.35; }}
.reason {{ color:#c9d4e8; margin:4px 0; }}
.tok {{ color:var(--muted); font-size:11px; font-family: ui-monospace, Menlo, Consolas, monospace; }}
.think,.out {{ color:var(--muted); font-size:11px; white-space:pre-wrap; display:none; margin-top:4px; }}
.show-think .think, .show-out .out {{ display:block; }}
.spark {{ display:flex; align-items:flex-end; gap:1px; height:36px; overflow:auto; margin-top:6px; }}
.spark i {{ display:block; width:3px; min-width:2px; background:var(--good); }}
.spark i.neg {{ background:var(--bad); }}
table.advtab {{ width:100%; border-collapse:collapse; font-size:12px; margin-bottom:12px; }}
table.advtab th, table.advtab td {{ border:1px solid var(--border); padding:6px 8px; text-align:right; }}
table.advtab th:first-child, table.advtab td:first-child {{ text-align:left; }}
@media (max-width: 1100px) {{ .grid,.meta,.adv {{ grid-template-columns:1fr; }} .col {{ max-height:none; }} }}
</style>
</head>
<body>
<header>
  <h1>Trajectory Compare + LLM Judge · DeepSeek vs SFT vs Qwen</h1>
  <p id="hdr"></p>
  <div class="tabs" id="tabs"></div>
  <div class="toolbar">
    <label><input type="checkbox" id="onlyLow"/> 只看低分轮 (score&lt;0)</label>
    <label><input type="checkbox" id="showThink"/> thinking</label>
    <label><input type="checkbox" id="showOut"/> observation</label>
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
function fmt(n, d=2) {{
  if (n==null || Number.isNaN(n)) return '—';
  const x = Number(n);
  if (Math.abs(x) >= 1000000) return (x/1e6).toFixed(2)+'M';
  if (Math.abs(x) >= 10000) return Math.round(x).toLocaleString();
  return x.toFixed(d);
}}
function clsNum(n) {{ return Number(n) >= 0 ? 'pos' : 'neg'; }}
function barHtml(score) {{
  const s = Math.max(-1, Math.min(1, Number(score)||0));
  const pct = Math.abs(s)*50;
  if (s>=0) return `<div class="bar-wrap"><div class="bar p" style="width:${{pct}}%"></div></div>`;
  return `<div class="bar-wrap"><div class="bar n" style="width:${{pct}}%"></div></div>`;
}}
function spark(rewards) {{
  const maxAbs = Math.max(...rewards.map(r => Math.abs(Number(r.score)||0)), 1e-6);
  return '<div class="spark">' + rewards.map(r => {{
    const s = Number(r.score)||0;
    const h = Math.max(2, Math.round(Math.abs(s)/maxAbs*34));
    return `<i class="${{s<0?'neg':''}}" style="height:${{h}}px" title="T${{r.turn}}=${{s.toFixed(3)}}"></i>`;
  }}).join('') + '</div>';
}}
function turnHtml(tr) {{
  const tags = (tr.tags||[]).map(t => `<span class="badge">${{esc(t)}}</span>`).join('');
  const llm = tr.offloaded ? '<span class="badge">LLM offload</span>' : (tr.offloaded===false?'<span class="badge">slm</span>':'');
  const cmds = (tr.cmds&&tr.cmds.length) ? tr.cmds.map(c => `<div class="cmd">${{esc(c)}}</div>`).join('') : (tr.tool_calls?`<div class="cmd">${{esc(tr.tool_calls)}}</div>`:'');
  const adv = (tr.adv_score_vs_dp==null) ? '' :
    `Δscore vs DP <span class="${{clsNum(tr.adv_score_vs_dp)}}">${{fmt(tr.adv_score_vs_dp,3)}}</span>
     · ΔcumTok <span class="${{clsNum(-(tr.adv_cumtok_vs_dp||0))}}">${{fmt(tr.adv_cumtok_vs_dp,0)}}</span>
     · Δcum$ <span class="${{clsNum(-(tr.adv_cumcost_vs_dp||0))}}">${{fmt(tr.adv_cumcost_vs_dp,4)}}</span>`;
  const klass = ['turn', tr.offloaded?'llm':'', Number(tr.score)<0?'low':''].filter(Boolean).join(' ');
  return `<div class="${{klass}}" data-low="${{Number(tr.score)<0?1:0}}">
    <div class="turn-h">
      <strong>#${{tr.turn}}</strong>
      <span class="score ${{clsNum(tr.score)}}">${{fmt(tr.score,3)}}</span>
      ${{barHtml(tr.score)}}
      ${{tags}}${{llm}}
    </div>
    <div class="reason">${{esc(tr.reason||'')}}</div>
    ${{cmds}}
    <div class="tok">tok in/out ${{fmt(tr.prompt_tokens,0)}}/${{fmt(tr.completion_tokens,0)}}
      · turn tok ${{fmt(tr.tokens,0)}} · $ ${{fmt(tr.cost_usd,5)}}
      · cumTok ${{fmt(tr.cum_tokens,0)}} · cum$ ${{fmt(tr.cum_cost_usd,4)}}
      · r/1kTok ${{fmt(tr.score_per_1k_tok,3)}}
      ${{adv ? ' · '+adv : ''}}</div>
    ${{tr.reasoning ? `<div class="think">think: ${{esc(tr.reasoning)}}</div>` : ''}}
    ${{tr.observation ? `<div class="out">obs: ${{esc(tr.observation)}}</div>` : ''}}
  </div>`;
}}
function modelCard(name, m) {{
  const t = m.token_totals || {{}};
  const a = m.final_advantage_vs_dp || {{}};
  return `<div class="card"><h3>${{esc(name)}}</h3>
    <div class="stat">exit <b>${{esc(m.exit_status)}}</b> · harness <b>${{m.solved?'✓':'✗'}}</b> · submitted <b>${{m.submitted?'✓':'✗'}}</b> · judge <b>${{esc(m.judge_mode)}}</b></div>
    <div class="stat">turns <b>${{m.n_turns}}</b> · mean r <b class="${{clsNum(t.mean_score)}}">${{fmt(t.mean_score,3)}}</b> · Σr <b>${{fmt(t.sum_score,2)}}</b></div>
    <div class="stat">tokens <b>${{fmt(t.tokens,0)}}</b> (in ${{fmt(t.prompt_tokens,0)}} / out ${{fmt(t.completion_tokens,0)}})</div>
    <div class="stat">cost <b>$${{fmt(t.cost_usd,3)}}</b>
      ${{name==='SFT' ? `· SLM in/out ${{fmt(t.slm_prompt_tokens,0)}}/${{fmt(t.slm_completion_tokens,0)}} · LLM in/out ${{fmt(t.llm_prompt_tokens,0)}}/${{fmt(t.llm_completion_tokens,0)}}` : ''}}
    </div>
    <div class="stat">vs DP: Δturns <b class="${{clsNum(-a.delta_turns)}}">${{a.delta_turns>0?'+':''}}${{a.delta_turns}}</b>
      · Δtok <b class="${{clsNum(-a.delta_tokens)}}">${{fmt(a.delta_tokens,0)}}</b>
      · Δ$ <b class="${{clsNum(-a.delta_cost_usd)}}">${{fmt(a.delta_cost_usd,3)}}</b>
      · Δmean r <b class="${{clsNum(a.delta_mean_score)}}">${{fmt(a.delta_mean_score,3)}}</b>
    </div>
    ${{spark(m.llm_turn_rewards||[])}}
  </div>`;
}}
function render() {{
  document.getElementById('hdr').textContent =
    `judge=${{report.model}} · normalize=${{report.normalize}} · 红/低分=负贡献；琥珀=SFT offload；Δ 均相对 DeepSeek（token/cost 越负越贵）。`;
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
    }};
    tabs.appendChild(btn);
    const sec = document.createElement('section');
    sec.className = 'case' + (idx===0?' active':'');
    sec.dataset.i = idx;
    const rows = names.map(n => {{
      const m = c.models[n]||{{}}; const t=m.token_totals||{{}}; const a=m.final_advantage_vs_dp||{{}};
      return `<tr>
        <td>${{n}}</td><td>${{m.n_turns}}</td><td>${{fmt(t.tokens,0)}}</td><td>${{fmt(t.cost_usd,3)}}</td>
        <td class="${{clsNum(t.mean_score)}}">${{fmt(t.mean_score,3)}}</td>
        <td class="${{clsNum(-a.delta_tokens)}}">${{fmt(a.delta_tokens,0)}}</td>
        <td class="${{clsNum(-a.delta_cost_usd)}}">${{fmt(a.delta_cost_usd,3)}}</td>
        <td class="${{clsNum(a.delta_mean_score)}}">${{fmt(a.delta_mean_score,3)}}</td>
      </tr>`;
    }}).join('');
    sec.innerHTML = `
      <div class="note"><b>${{esc(c.inst)}}</b> — ${{esc(c.note)}}<div style="margin-top:4px;color:#9aa3b2">${{esc((c.models.DeepSeek||{{}}).judge_summary||'')}}</div></div>
      <table class="advtab">
        <thead><tr><th>模型</th><th>轮次</th><th>tokens</th><th>cost $</th><th>mean score</th><th>token优势 vs DP</th><th>$优势 vs DP</th><th>mean r 优势 vs DP</th></tr></thead>
        <tbody>${{rows}}</tbody>
      </table>
      <div class="adv">${{names.map(n => modelCard(n, c.models[n]||{{}})).join('')}}</div>
      <div class="grid">${{names.map(n => {{
        const m = c.models[n]||{{}};
        return `<div class="col" data-col="${{n}}"><h2>${{n}} · ${{m.n_turns}} turns · mean r ${{fmt((m.token_totals||{{}}).mean_score,3)}}</h2>
          ${{(m.llm_turn_rewards||[]).map(turnHtml).join('')}}</div>`;
      }}).join('')}}</div>`;
    wrap.appendChild(sec);
  }});
}}
render();
document.getElementById('showThink').onchange = e => document.body.classList.toggle('show-think', e.target.checked);
document.getElementById('showOut').onchange = e => document.body.classList.toggle('show-out', e.target.checked);
document.getElementById('onlyLow').onchange = e => {{
  document.querySelectorAll('.turn').forEach(t => t.style.display = (!e.target.checked || t.dataset.low==='1') ? '' : 'none');
}};
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
document.getElementById('tabs').addEventListener('click', () => setTimeout(bindSync, 0));
</script>
</body>
</html>
"""


def score_one(
    *,
    model_name: str,
    traj_path: Path,
    args: argparse.Namespace,
    sft_sessions: dict[int, list[dict[str, Any]]],
    resolved: bool | None = None,
) -> dict[str, Any]:
    traj = scorer.load_traj(traj_path, resolved=resolved)
    print(
        f"judge {model_name} {traj['instance_id']} turns={traj['n_turns']} "
        f"resolved={traj['solved']} src={traj.get('resolved_source')} "
        f"submitted={traj.get('submitted')}",
        flush=True,
    )
    scored = scorer.score_traj(traj, args)
    token_rows = extract_turn_tokens(traj_path)
    attach_tokens(
        scored,
        model_name=model_name,
        token_rows=token_rows,
        sft_sessions=sft_sessions,
    )
    for item in scored.get("llm_turn_rewards") or []:
        item["score_raw"] = float(item.get("score") or 0.0)
    scored["n_postprocess_adjusted"] = 0
    scored["n_verify_exempt"] = 0
    detect_spin_marks(scored)
    scored["model_name"] = model_name
    return scored


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("/workspace/work/mjy/traj_compare_judge"))
    p.add_argument("--normalize", choices=("raw", "sum_to_outcome", "mean_to_outcome"), default="raw")
    p.add_argument("--chunk-size", type=int, default=8, help="turns per judge call; each listed turn is scored")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--heuristic", action="store_true")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", "http://208.64.254.189:8000/v1"))
    p.add_argument("--api-key", default=(os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip())
    p.add_argument("--model", default=os.environ.get("DASHSCOPE_MODEL", "deepseek-v4-flash-0731"))
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--timeout", type=float, default=180.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.heuristic and not args.api_key:
        print("ERROR: set DASHSCOPE_API_KEY", file=sys.stderr)
        return 2
    sft_sessions = load_sft_offload(DEFAULT_RUNS["SFT"])
    jobs: list[tuple[str, str, Path, bool | None]] = []
    for inst, _note in CASES:
        for name, root in DEFAULT_RUNS.items():
            p = root / inst / f"{inst}.traj.json"
            if not p.exists():
                print(f"missing {p}", file=sys.stderr)
                return 2
            resolved = load_harness_resolved(DEFAULT_VERIFIED[name], inst)
            if resolved is None:
                print(f"WARN: no harness report for {name} {inst}; fallback Submitted", flush=True)
            jobs.append((inst, name, p, resolved))

    results: dict[str, dict[str, dict[str, Any]]] = {inst: {} for inst, _ in CASES}

    def _run(job: tuple[str, str, Path, bool | None]) -> tuple[str, str, dict[str, Any]]:
        inst, name, path, resolved = job
        return inst, name, score_one(
            model_name=name,
            traj_path=path,
            args=args,
            sft_sessions=sft_sessions,
            resolved=resolved,
        )

    workers = max(1, int(args.concurrency))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run, j) for j in jobs]
        for fut in concurrent.futures.as_completed(futs):
            inst, name, scored = fut.result()
            results[inst][name] = scored
            tot = scored.get("token_totals") or {}
            print(
                f"done {name} {inst} mode={scored.get('judge_mode')} "
                f"mean_r={tot.get('mean_score')} tok={tot.get('tokens')} "
                f"${tot.get('cost_usd')} {scored.get('latency_s')}s",
                flush=True,
            )

    cases_out = []
    for inst, note in CASES:
        models = results[inst]
        paint_advantages(models)
        cases_out.append({"inst": inst, "note": note, "models": models})

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "title": "Trajectory Compare · judge score + A_t (prompt rubric)",
        "model": args.model if not args.heuristic else "heuristic",
        "normalize": args.normalize,
        "base_url": args.base_url,
        "advantage_def": "A_t = r_i - mean(r), A_s=0; no rule clamp, LLM scores from rubric prompt",
        "solved_source": "harness report.json resolved (fallback Submitted)",
        "pricing": {
            "slm_in": SLM_IN,
            "slm_out": SLM_OUT,
            "llm_in": LLM_IN,
            "llm_out": LLM_OUT,
        },
        "cases": cases_out,
    }
    json_path = out_dir / "compare_judge.json"
    html_path = out_dir / "compare_judge.html"
    public_html = Path("/workspace/work/mjy/traj_compare_dp_sft_qwen.html")
    json_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    from render_compare_judge_html import render_html as render_simple_html

    html_txt = render_simple_html(report)
    html_path.write_text(html_txt, encoding="utf-8")
    public_html.write_text(html_txt, encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")
    print(f"wrote {public_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
