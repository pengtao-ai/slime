#!/usr/bin/env python3
"""Web viewer for coding-agent / offload rollout ``.pt`` dumps.

Browse trajectories, fork segments, and decoded chat messages (incl. offload /
GLM embed). Unlike ``trace_timeline_viewer.py``, this does not need ``sample.trace``.

Examples::

    python tools/rollout_traj_viewer.py \\
        runs/agent_offload_pyrodash4b_docker_async_20260728_040357/rollout_dumps/rollout_1.pt \\
        --tokenizer /workspace/models/pyromind/PyroDash-4B-SFT-0727_pad248320 \\
        --port 8765

    # Or a dump directory (pick dump in UI):
    python tools/rollout_traj_viewer.py runs/.../rollout_dumps \\
        --tokenizer /path/to/hf --port 8765
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

OFFLOAD_OPEN = "<|llm_offload|>"
OFFLOAD_CLOSE = "<|/llm_offload|>"
OFF_SPAN_RE = re.compile(re.escape(OFFLOAD_OPEN) + r"(\d)" + re.escape(OFFLOAD_CLOSE))
IM_BLOCK_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL)
TOOL_XML_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
PARAM_RE = re.compile(r"<parameter=([^>]+)>\n?(.*?)</parameter>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
JSON_TOOL_CALLS_RE = re.compile(r"\[tool_calls\]\s*(\[.*\])\s*$", re.DOTALL)

DEFAULT_OFFLOAD_CLOSE_ID = 248078


# ---------------------------------------------------------------------------
# Dump helpers
# ---------------------------------------------------------------------------


def _sample_dict(sample: Any) -> dict[str, Any]:
    if hasattr(sample, "to_dict"):
        sample = sample.to_dict()
    if not isinstance(sample, dict):
        raise TypeError(type(sample))
    return sample


def _meta(s: dict[str, Any]) -> dict[str, Any]:
    m = s.get("metadata") or {}
    return m if isinstance(m, dict) else {}


def _offload_stats(s: dict[str, Any]) -> dict[str, Any]:
    st = _meta(s).get("offload_stats") or {}
    return st if isinstance(st, dict) else {}


def _traj_key(s: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (s.get("label"), s.get("group_index"), s.get("index"))


def _reward(segs: list[dict[str, Any]]) -> Any:
    for s in reversed(segs):
        if s.get("reward") is not None:
            return s.get("reward")
    return None


def _is_solved(segs: list[dict[str, Any]]) -> bool | None:
    meta = _meta(segs[0])
    if "grading_solved" in meta:
        return bool(meta["grading_solved"])
    if "solved" in meta:
        v = meta["solved"]
        return bool(v > 0) if isinstance(v, (int, float)) else bool(v)
    r = _reward(segs)
    if isinstance(r, (int, float)):
        return r > 0
    return None


def _group_trajs(samples: list[dict[str, Any]]) -> dict[tuple[Any, Any, Any], list[dict[str, Any]]]:
    trajs: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        trajs[_traj_key(s)].append(s)
    for segs in trajs.values():
        segs.sort(key=lambda x: len(x.get("tokens") or []) - int(x.get("response_length") or 0))
    return trajs


def reasoning_from_n(n: int) -> tuple[bool, str | None]:
    if n <= 0:
        return False, None
    if n <= 5:
        return True, "high"
    return True, "max"


def _common_prefix_len(a: list[int], b: list[int], chunk: int = 4096) -> int:
    limit = min(len(a), len(b))
    matched = 0
    while matched < limit:
        chunk_end = min(matched + chunk, limit)
        if a[matched:chunk_end] == b[matched:chunk_end]:
            matched = chunk_end
        else:
            while matched < chunk_end and a[matched] == b[matched]:
                matched += 1
            return matched
    return matched


def infer_failure_reason(
    segs: list[dict[str, Any]],
    *,
    reward: Any = None,
) -> dict[str, Any]:
    """Summarize why a trajectory did not count as solved (from dump metadata)."""
    meta = _meta(segs[0]) if segs else {}
    st = _offload_stats(segs[0]) if segs else {}
    if reward is None:
        reward = _reward(segs)
    solved = _is_solved(segs)
    grading = meta.get("grading_solved")
    empty_patch = bool(meta.get("empty_patch"))
    exit_code = meta.get("agent_exit_code")
    truncated = bool(meta.get("truncated"))
    ill_formed = bool(meta.get("ill_formed"))
    applied = meta.get("applied_cleanly")

    codes: list[str] = []
    notes: list[str] = []

    if solved is True and isinstance(reward, (int, float)) and reward > 0:
        return {
            "passed": True,
            "code": "passed",
            "summary": "评测通过（grading_solved / reward>0）",
            "codes": ["passed"],
            "notes": [],
            "details": {
                "grading_solved": grading,
                "empty_patch": empty_patch,
                "agent_exit_code": exit_code,
                "reward": reward,
            },
        }

    # Order: most actionable first
    if exit_code is not None and int(exit_code) < 0:
        codes.append("time_budget")
        notes.append(f"agent_exit_code={exit_code}：超时 / time budget exceeded（未正常跑完）")
    elif exit_code is not None and int(exit_code) != 0:
        codes.append("cli_error")
        notes.append(f"agent_exit_code={exit_code}：harness/CLI 非零退出")

    if empty_patch:
        codes.append("empty_patch")
        notes.append("empty_patch=True：最终 git diff 为空，没有可提交的代码改动")

    if truncated:
        codes.append("truncated")
        notes.append("truncated=True：最后一轮生成因 length 截断")

    if ill_formed:
        codes.append("ill_formed")
        notes.append("ill_formed=True：存在格式异常的 assistant turn")

    if applied is False:
        codes.append("patch_apply_failed")
        notes.append("applied_cleanly=False：评测侧 patch 未能干净应用")

    if grading is True and empty_patch:
        codes.append("grader_empty_conflict")
        notes.append("grader 曾判 solved 但 patch 为空（运行时会强制 solved=0）")

    if grading is True and isinstance(reward, (int, float)) and reward <= 0 and not empty_patch:
        codes.append("efficiency_zero")
        notes.append(
            "grading_solved=True 但 train reward≤0：可能被 offload cost-aware 效率项压到 0"
            f"（offload_count={st.get('offload_count')}, glm_out={st.get('glm_output_tokens')}）"
        )

    if grading is False and not empty_patch:
        codes.append("tests_failed")
        notes.append("有 patch（empty_patch=False）但 grading_solved=False：改动未通过 SWE 评测")

    if grading is False and empty_patch and "empty_patch" in codes:
        # already covered; add grading note
        notes.append("grading_solved=False：未解决该 instance")

    if not codes:
        codes.append("unsolved_unknown")
        notes.append("未解决，但 metadata 未给出更细原因（检查 reward / grader 日志）")

    primary = codes[0]
    summary_map = {
        "time_budget": "超时未跑完",
        "cli_error": "Agent CLI 异常退出",
        "empty_patch": "无有效 patch（未改代码）",
        "truncated": "生成被截断",
        "ill_formed": "输出格式异常",
        "patch_apply_failed": "patch 无法应用",
        "efficiency_zero": "评测通过但效率奖励为 0",
        "tests_failed": "有改动但评测未通过",
        "grader_empty_conflict": "空 patch 与 grader 冲突",
        "unsolved_unknown": "未通过（原因不明）",
    }
    return {
        "passed": False,
        "code": primary,
        "summary": summary_map.get(primary, primary),
        "codes": codes,
        "notes": notes,
        "details": {
            "grading_solved": grading,
            "solved_meta": meta.get("solved"),
            "empty_patch": empty_patch,
            "agent_exit_code": exit_code,
            "truncated": truncated,
            "ill_formed": ill_formed,
            "applied_cleanly": applied,
            "reward": reward,
            "offload_count": st.get("offload_count"),
        },
    }


def infer_segment_reasons(
    segs: list[dict[str, Any]],
    *,
    offload_close_id: int = DEFAULT_OFFLOAD_CLOSE_ID,
) -> list[dict[str, Any]]:
    """Infer why each fork segment exists (TOKEN_FORK / TREE_LEAF) via token LCP.

    Dump does not store fork kind; this matches the offline analysis used in
    ``examples/coding_agent_rl/data/fork_traj_*/README.md``.
    """
    infos: list[dict[str, Any]] = []
    token_cache = [list(s.get("tokens") or []) for s in segs]
    prompt_lens = [
        max(0, len(token_cache[i]) - int(segs[i].get("response_length") or 0)) for i in range(len(segs))
    ]

    for i, s in enumerate(segs):
        tokens = token_cache[i]
        prompt_len = prompt_lens[i]
        resp_len = int(s.get("response_length") or 0)
        prompt_ids = tokens[:prompt_len]
        resp_ids = tokens[prompt_len:] if resp_len else []

        offload_hits = sum(1 for t in resp_ids if t == offload_close_id)
        after_close = 0
        if offload_close_id in resp_ids:
            pos = len(resp_ids) - 1 - resp_ids[::-1].index(offload_close_id)
            after_close = len(resp_ids) - pos - 1

        if i == 0:
            infos.append(
                {
                    "segment_id": i,
                    "kind": "FIRST",
                    "branch": "main",
                    "parent_seg": None,
                    "lcp": None,
                    "parent_full": None,
                    "drift": None,
                    "offload_in_response": offload_hits,
                    "after_close_tokens": after_close,
                    "summary": "轨迹第一条 Sample（无前驱）",
                    "detail": "Episode 起点。后续 TOKEN_FORK / TREE_LEAF 都相对这条（或同支前驱）切开。",
                }
            )
            continue

        best_j = 0
        best_lcp = -1
        for j in range(i):
            lcp = _common_prefix_len(token_cache[j], prompt_ids)
            if lcp > best_lcp:
                best_lcp = lcp
                best_j = j
        parent_full = len(token_cache[best_j])
        drift = parent_full - best_lcp
        main_lcp = _common_prefix_len(token_cache[0], prompt_ids)
        main_prompt = prompt_lens[0]
        main_affinity = main_lcp / max(main_prompt, 1)

        # Offload expand: prev ends near offload close and this prompt continues far past prev full
        prev_resp = token_cache[best_j][prompt_lens[best_j] :]
        prev_had_offload = offload_close_id in prev_resp
        offload_expand = bool(prev_had_offload and best_lcp >= parent_full - 8 and prompt_len > parent_full + 16)

        if offload_expand:
            kind = "TOKEN_FORK"
            branch = infos[best_j].get("branch") or "main"
            summary = f"TOKEN_FORK←seg{best_j}（offload 续写扩展）"
            detail = (
                f"相对 seg{best_j}：LCP={best_lcp}/{parent_full}（drift={drift}）。"
                f"前一段 response 含 offload；本段 prompt 比前段 full 更长（+{prompt_len - parent_full} tok），"
                "SampleBuilder 无法 REALIGN 吸收扩展的 assistant 回放 → FORK 新开 Sample。"
            )
        elif best_lcp < 128 or (main_affinity < 0.25 and best_lcp < prompt_len * 0.35):
            kind = "TREE_LEAF"
            branch = f"subagent_from_seg{best_j}"
            summary = f"TREE_LEAF 旁支（相对 seg{best_j}）"
            detail = (
                f"与最佳前驱 seg{best_j} 的 LCP 仅 {best_lcp}/{parent_full}；"
                f"与 main(seg0) LCP={main_lcp}/{main_prompt}（affinity={main_affinity:.2f}）。"
                "更像 Agent/subagent 旁支 leaf，而不是主链 TOKEN_FORK 切段。"
            )
        else:
            kind = "TOKEN_FORK"
            branch = infos[best_j].get("branch") or "main"
            summary = f"TOKEN_FORK←seg{best_j}（token 漂移）"
            detail = (
                f"同支续写但 SampleBuilder FORK：相对 seg{best_j} LCP={best_lcp}/{parent_full}"
                f"（drift={drift} tok）。通常来自 TITO/tool_result 重渲或超 fork_threshold 的漂移，"
                "无法 CLEAN/REALIGN → 新开训练 Sample。"
            )
            if after_close:
                detail += f" 本段 response 在 offload close 后还有 {after_close} tok（GLM embed / 续写）。"

        infos.append(
            {
                "segment_id": i,
                "kind": kind,
                "branch": branch,
                "parent_seg": best_j,
                "lcp": best_lcp,
                "parent_full": parent_full,
                "drift": drift,
                "main_lcp": main_lcp,
                "offload_in_response": offload_hits,
                "after_close_tokens": after_close,
                "summary": summary,
                "detail": detail,
            }
        )
    return infos


# ---------------------------------------------------------------------------
# Message parse (chat template -> messages)
# ---------------------------------------------------------------------------


def _parse_tool_calls_xml(text: str) -> list[dict]:
    out = []
    for i, m in enumerate(TOOL_XML_RE.finditer(text)):
        name = m.group(1).strip()
        params = {pm.group(1): pm.group(2) for pm in PARAM_RE.finditer(m.group(2))}
        out.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": params},
            }
        )
    return out


def _parse_tool_calls_json(text: str) -> list[dict] | None:
    m = JSON_TOOL_CALLS_RE.search((text or "").strip())
    if not m:
        return None
    try:
        raw = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    out = []
    for i, tc in enumerate(raw):
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args_obj = json.loads(args)
            except json.JSONDecodeError:
                args_obj = {"_raw": args}
        else:
            args_obj = args or {}
        out.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": fn.get("name") or "unknown", "arguments": args_obj},
            }
        )
    return out


def _extract_tool_calls(content: str) -> tuple[str, list[dict]]:
    tool_calls = _parse_tool_calls_xml(content)
    if tool_calls:
        return TOOL_XML_RE.sub("", content).strip(), tool_calls
    jtc = _parse_tool_calls_json(content)
    if jtc:
        return JSON_TOOL_CALLS_RE.sub("", content).strip(), jtc
    return (content or "").strip(), []


def _parse_glm_suffix(suffix: str) -> dict[str, Any]:
    glm_think = ""
    glm_content = ""
    if "</think>" in suffix:
        glm_think, _, rest = suffix.partition("</think>")
        glm_content = rest.lstrip("\n")
    elif "[tool_calls]" in suffix or "<tool_call>" in suffix:
        glm_content = suffix.lstrip("\n")
    else:
        glm_think = suffix
    glm_content, tool_calls = _extract_tool_calls(glm_content)
    out: dict[str, Any] = {
        "reasoning_content": glm_think if glm_think else None,
        "content": glm_content if glm_content else None,
        "source": "embedded_loss_mask_0",
    }
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def parse_assistant_message(body: str) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant"}
    off_m = OFF_SPAN_RE.search(body)
    if off_m:
        n = int(off_m.group(1))
        _, effort = reasoning_from_n(n)
        slm_part = body[: off_m.end()]
        glm_part = body[off_m.end() :]
        slm_think = slm_part
        if "<think>" in slm_think:
            slm_think = slm_think.split("<think>", 1)[1]
        slm_think = slm_think.replace("</think>", "")
        msg["reasoning_content"] = slm_think
        msg["content"] = None
        msg["offload"] = {"n": n, "in_think": True, "reasoning_effort": effort}
        glm = _parse_glm_suffix(glm_part)
        msg["glm_response"] = glm
        if glm.get("tool_calls"):
            msg["tool_calls"] = glm["tool_calls"]
        if glm.get("content"):
            msg["content"] = glm["content"]
        return msg

    m = THINK_RE.search(body)
    if m:
        reasoning = m.group(1)
        content = (body[: m.start()] + body[m.end() :]).strip()
    elif "<think>" in body and "</think>" not in body:
        idx = body.index("<think>")
        reasoning = body[idx + len("<think>") :]
        content = body[:idx].strip()
        msg["think_unclosed"] = True
    elif "</think>" in body and "<think>" not in body:
        reasoning, _, content = body.partition("</think>")
        content = content.strip()
    else:
        reasoning, content = None, body.strip()

    content, tool_calls = _extract_tool_calls(content or "")
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    msg["content"] = content if content else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def parse_user_or_tool(body: str) -> dict[str, Any]:
    body = body.strip()
    tr = re.search(r"<tool_response>\n?(.*?)\n?</tool_response>", body, re.DOTALL)
    if tr:
        return {"role": "tool", "content": tr.group(1)}
    if body.startswith("<tool_response>"):
        return {
            "role": "tool",
            "content": body[len("<tool_response>") :].replace("</tool_response>", "").strip(),
        }
    return {"role": "user", "content": body}


def tokens_to_messages(full_text: str) -> tuple[str, list[dict[str, Any]]]:
    system_prompt = ""
    messages: list[dict[str, Any]] = []
    for m in IM_BLOCK_RE.finditer(full_text):
        role, body = m.group(1), m.group(2)
        if role == "system":
            system_prompt = (system_prompt + "\n" + body.strip()).strip() if system_prompt else body.strip()
            continue
        if role == "assistant":
            messages.append(parse_assistant_message(body))
        elif role == "user":
            messages.append(parse_user_or_tool(body))
        else:
            messages.append({"role": role, "content": body.strip()})
    return system_prompt, messages


def find_offloads(text: str) -> list[dict[str, Any]]:
    out = []
    for m in OFF_SPAN_RE.finditer(text or ""):
        n = int(m.group(1))
        _, effort = reasoning_from_n(n)
        out.append({"n": n, "span": m.group(0), "reasoning_effort": effort})
    return out


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DumpStore:
    def __init__(
        self,
        pt_paths: list[Path],
        tokenizer_path: str | None,
        offload_close_id: int,
        cache_size: int = 32,
    ) -> None:
        self.pt_paths = [p.resolve() for p in pt_paths]
        self.tokenizer_path = tokenizer_path
        self.offload_close_id = offload_close_id
        self._tokenizer = None
        self._lock = threading.RLock()
        self._dumps: dict[str, dict[str, Any]] = {}
        self._detail_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_size = cache_size

    def list_dumps(self) -> list[dict[str, Any]]:
        return [{"id": str(p), "name": p.name, "path": str(p)} for p in self.pt_paths]

    def _ensure_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if not self.tokenizer_path:
            raise RuntimeError("No --tokenizer provided; cannot decode trajectories")
        from transformers import AutoTokenizer

        print(f"[traj_viewer] loading tokenizer: {self.tokenizer_path}", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        return self._tokenizer

    def _load_dump(self, dump_id: str) -> dict[str, Any]:
        with self._lock:
            if dump_id in self._dumps:
                return self._dumps[dump_id]
        path = Path(dump_id)
        if path.resolve() not in self.pt_paths:
            # allow name match
            matches = [p for p in self.pt_paths if p.name == dump_id or str(p) == dump_id]
            if not matches:
                raise KeyError(f"unknown dump: {dump_id}")
            path = matches[0]
            dump_id = str(path)

        print(f"[traj_viewer] loading {path} ...", flush=True)
        t0 = time.time()
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "samples" in obj:
            samples = [_sample_dict(s) for s in obj["samples"]]
            rollout_id = obj.get("rollout_id")
        elif isinstance(obj, list):
            samples = [_sample_dict(s) for s in obj]
            rollout_id = None
        else:
            raise ValueError(f"bad dump format: {type(obj)}")

        trajs = _group_trajs(samples)
        index_rows = []
        n_offload = n_solved = n_unsolved = 0
        for (label, g, i), segs in trajs.items():
            st = _offload_stats(segs[0])
            oc = int(st.get("offload_count") or 0)
            solved = _is_solved(segs)
            if oc:
                n_offload += 1
            if solved is True:
                n_solved += 1
            elif solved is False:
                n_unsolved += 1
            meta = _meta(segs[0])
            reward = _reward(segs)
            fail = infer_failure_reason(segs, reward=reward)
            # Cheap seg-kind histogram (no decode) for list view
            seg_kinds = infer_segment_reasons(segs, offload_close_id=self.offload_close_id)
            kind_counts: dict[str, int] = defaultdict(int)
            for info in seg_kinds:
                kind_counts[str(info["kind"])] += 1
            index_rows.append(
                {
                    "label": label,
                    "group_index": g,
                    "index": i,
                    "n_seg": len(segs),
                    "reward": reward,
                    "solved": solved,
                    "empty_patch": meta.get("empty_patch"),
                    "offload_count": oc,
                    "glm_output_tokens": st.get("glm_output_tokens"),
                    "small_output_tokens": st.get("small_output_tokens"),
                    "resp_sum": sum(int(s.get("response_length") or 0) for s in segs),
                    "agent_exit_code": meta.get("agent_exit_code"),
                    "has_trace": any(isinstance(s.get("trace"), dict) for s in segs),
                    "fail_code": fail["code"],
                    "fail_summary": fail["summary"],
                    "fail_passed": fail["passed"],
                    "seg_kinds": dict(kind_counts),
                }
            )
        index_rows.sort(key=lambda r: (-(r["offload_count"] or 0), -(r["n_seg"] or 0), str(r["label"])))

        packed = {
            "id": str(path),
            "path": str(path),
            "name": path.name,
            "rollout_id": rollout_id,
            "n_samples": len(samples),
            "n_trajs": len(trajs),
            "n_offload": n_offload,
            "n_solved": n_solved,
            "n_unsolved": n_unsolved,
            "n_with_trace": sum(1 for s in samples if isinstance(s.get("trace"), dict)),
            "index": index_rows,
            "trajs": trajs,
            "loaded_at": time.time(),
            "load_sec": round(time.time() - t0, 2),
        }
        with self._lock:
            self._dumps[str(path)] = packed
        print(f"[traj_viewer] loaded {path.name}: {len(samples)} segs / {len(trajs)} trajs in {packed['load_sec']}s", flush=True)
        return packed

    def summary(self, dump_id: str) -> dict[str, Any]:
        d = self._load_dump(dump_id)
        return {
            "id": d["id"],
            "path": d["path"],
            "name": d["name"],
            "rollout_id": d["rollout_id"],
            "n_samples": d["n_samples"],
            "n_trajs": d["n_trajs"],
            "n_offload": d["n_offload"],
            "n_solved": d["n_solved"],
            "n_unsolved": d["n_unsolved"],
            "n_with_trace": d["n_with_trace"],
            "load_sec": d["load_sec"],
            "tokenizer": self.tokenizer_path,
        }

    def list_trajs(
        self,
        dump_id: str,
        *,
        offload: bool = False,
        unsolved: bool = False,
        solved: bool = False,
        q: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        d = self._load_dump(dump_id)
        rows = []
        for r in d["index"]:
            if offload and not (r["offload_count"] or 0):
                continue
            if unsolved and r["solved"] is not False:
                continue
            if solved and r["solved"] is not True:
                continue
            if q and q.lower() not in str(r["label"]).lower():
                continue
            rows.append(r)
            if len(rows) >= limit:
                break
        return rows

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            if key in self._detail_cache:
                self._detail_cache.move_to_end(key)
                return self._detail_cache[key]
        return None

    def _cache_put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._detail_cache[key] = value
            self._detail_cache.move_to_end(key)
            while len(self._detail_cache) > self._cache_size:
                self._detail_cache.popitem(last=False)

    def get_traj(
        self,
        dump_id: str,
        *,
        label: str,
        group_index: int,
        index: int,
        seg: int | None = None,
        include_system: bool = False,
        max_tool_chars: int = 8000,
    ) -> dict[str, Any]:
        d = self._load_dump(dump_id)
        key = (label, group_index, index)
        # fuzzy label match if exact missing
        segs = d["trajs"].get(key)
        if segs is None:
            for k, v in d["trajs"].items():
                if k[1] == group_index and k[2] == index and label in str(k[0]):
                    key = k
                    segs = v
                    break
        if not segs:
            raise KeyError(f"traj not found: {label} g={group_index} i={index}")

        cache_key = f"v2|{d['id']}|{key}|{seg}|{include_system}|{max_tool_chars}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        tok = self._ensure_tokenizer()
        meta = _meta(segs[0])
        st = _offload_stats(segs[0])
        reward = _reward(segs)
        failure = infer_failure_reason(segs, reward=reward)
        seg_reasons = infer_segment_reasons(segs, offload_close_id=self.offload_close_id)
        reason_by_id = {r["segment_id"]: r for r in seg_reasons}

        out_segs = []
        for si, s in enumerate(segs):
            if seg is not None and si != seg:
                continue
            tokens = list(s.get("tokens") or [])
            resp_len = int(s.get("response_length") or 0)
            prompt_len = len(tokens) - resp_len
            resp_ids = tokens[-resp_len:] if resp_len else []
            mask = list(s.get("loss_mask") or [])
            full_text = tok.decode(tokens, skip_special_tokens=False)
            resp_text = tok.decode(resp_ids, skip_special_tokens=False) if resp_ids else ""
            system_prompt, messages = tokens_to_messages(full_text)

            # truncate huge tool payloads for UI
            slim_messages = []
            for m in messages:
                mm = dict(m)
                if mm.get("role") == "tool" and isinstance(mm.get("content"), str):
                    c = mm["content"]
                    if len(c) > max_tool_chars:
                        mm["content"] = c[:max_tool_chars] + f"\n...<truncated {len(c) - max_tool_chars} chars>"
                        mm["truncated"] = True
                slim_messages.append(mm)

            after_close = 0
            mask0_after = 0
            if self.offload_close_id in resp_ids:
                pos = len(resp_ids) - 1 - resp_ids[::-1].index(self.offload_close_id)
                after_close = len(resp_ids) - pos - 1
                if mask and pos + 1 < len(mask):
                    mask0_after = mask[pos + 1 :].count(0)

            why = reason_by_id.get(si) or {}
            seg_doc: dict[str, Any] = {
                "segment_id": si,
                "why_segment": why,
                "stats": {
                    "n_tokens": len(tokens),
                    "prompt_tokens": prompt_len,
                    "response_length": resp_len,
                    "loss_mask_sum": int(sum(mask)) if mask else 0,
                    "loss_mask_zeros": int(mask.count(0)) if mask else 0,
                    "offload_in_response": find_offloads(resp_text),
                    "after_close_tokens": after_close,
                    "mask0_after_close": mask0_after,
                    "reward": s.get("reward"),
                    "status": s.get("status"),
                },
                "messages": slim_messages,
                "response_preview": resp_text[:2000],
            }
            if include_system:
                seg_doc["system_prompt"] = system_prompt
            else:
                seg_doc["system_prompt_chars"] = len(system_prompt)
            out_segs.append(seg_doc)

        result = {
            "dump": d["name"],
            "label": key[0],
            "group_index": key[1],
            "index": key[2],
            "reward": reward,
            "solved": _is_solved(segs),
            "empty_patch": meta.get("empty_patch"),
            "grading_solved": meta.get("grading_solved"),
            "failure": failure,
            "segment_reasons": seg_reasons,
            "offload_stats": st,
            "metadata": {k: meta[k] for k in meta if k != "offload_stats"},
            "n_segments": len(segs),
            "segments": out_segs,
        }
        self._cache_put(cache_key, result)
        return result


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Rollout Trajectory Viewer</title>
<style>
  :root {
    --bg: #f4f1ea;
    --panel: #fffdf8;
    --ink: #1c1917;
    --muted: #78716c;
    --line: #e7e5e4;
    --accent: #0f766e;
    --user: #1d4ed8;
    --assistant: #0f766e;
    --tool: #a16207;
    --offload: #b45309;
    --glm: #7c3aed;
    --danger: #b91c1c;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "IBM Plex Sans", "Source Sans 3", "Noto Sans SC", sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #dde9e4 0%, transparent 55%),
                radial-gradient(900px 500px at 100% 0%, #efe6d6 0%, transparent 50%),
                var(--bg);
    color: var(--ink); height: 100vh; display: grid;
    grid-template-columns: 360px 1fr; grid-template-rows: auto 1fr;
  }
  header {
    grid-column: 1 / -1; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
    padding: 12px 16px; border-bottom: 1px solid var(--line); background: rgba(255,253,248,.9);
    backdrop-filter: blur(8px);
  }
  header h1 { font-size: 16px; margin: 0; letter-spacing: .02em; }
  header .meta { color: var(--muted); font-size: 12px; }
  select, input, button {
    font: inherit; border: 1px solid var(--line); border-radius: 8px; padding: 6px 10px;
    background: #fff; color: var(--ink);
  }
  button { cursor: pointer; background: var(--accent); color: #fff; border-color: transparent; }
  button.secondary { background: #fff; color: var(--ink); border-color: var(--line); }
  label.chk { display: inline-flex; gap: 6px; align-items: center; font-size: 13px; color: var(--muted); }
  #sidebar {
    border-right: 1px solid var(--line); overflow: auto; background: rgba(255,253,248,.75);
    padding: 8px;
  }
  .traj {
    padding: 10px 12px; border-radius: 10px; cursor: pointer; border: 1px solid transparent;
    margin-bottom: 4px;
  }
  .traj:hover { background: #fff; border-color: var(--line); }
  .traj.active { background: #fff; border-color: var(--accent); box-shadow: 0 0 0 1px rgba(15,118,110,.2); }
  .traj .title { font-size: 13px; font-weight: 600; word-break: break-all; }
  .traj .sub { font-size: 11px; color: var(--muted); margin-top: 4px; display: flex; flex-wrap: wrap; gap: 6px; }
  .badge {
    display: inline-block; padding: 1px 6px; border-radius: 999px; font-size: 10px;
    background: #e7e5e4; color: #44403c;
  }
  .badge.off { background: #ffedd5; color: var(--offload); }
  .badge.ok { background: #dcfce7; color: #166534; }
  .badge.bad { background: #fee2e2; color: var(--danger); }
  .badge.fork { background: #e0e7ff; color: #3730a3; }
  .badge.leaf { background: #fce7f3; color: #9d174d; }
  .fail-box {
    border: 1px solid #fecaca; background: #fff1f2; border-radius: 14px;
    padding: 12px 14px; margin-bottom: 14px;
  }
  .fail-box.pass { border-color: #bbf7d0; background: #f0fdf4; }
  .fail-box h2 { margin: 0 0 6px; font-size: 14px; }
  .fail-box ul { margin: 6px 0 0; padding-left: 18px; font-size: 12px; }
  .why-box {
    border-left: 3px solid #6366f1; background: #eef2ff; padding: 8px 10px;
    border-radius: 0 8px 8px 0; font-size: 12px; margin-top: 8px; white-space: pre-wrap;
  }
  #main { overflow: auto; padding: 16px 20px 40px; }
  .empty { color: var(--muted); padding: 40px; text-align: center; }
  .seg-tabs { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0; }
  .seg-tabs button { background: #fff; color: var(--ink); border: 1px solid var(--line); }
  .seg-tabs button.active { background: var(--accent); color: #fff; border-color: transparent; }
  .seg-tabs button.has-off { outline: 2px solid #fdba74; outline-offset: 1px; }
  .panel {
    background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
    padding: 14px 16px; margin-bottom: 14px;
  }
  .panel h2 { margin: 0 0 8px; font-size: 14px; }
  .kv { display: grid; grid-template-columns: 140px 1fr; gap: 4px 10px; font-size: 12px; }
  .kv .k { color: var(--muted); }
  .msg {
    border: 1px solid var(--line); border-radius: 12px; margin: 10px 0; overflow: hidden;
    background: #fff;
  }
  .msg .bar {
    display: flex; gap: 8px; align-items: center; padding: 6px 10px; font-size: 12px;
    font-weight: 600; border-bottom: 1px solid var(--line);
  }
  .msg.user .bar { background: #eff6ff; color: var(--user); }
  .msg.assistant .bar { background: #ecfdf5; color: var(--assistant); }
  .msg.tool .bar { background: #fffbeb; color: var(--tool); }
  .msg .body { padding: 10px 12px; font-size: 13px; white-space: pre-wrap; word-break: break-word; line-height: 1.45; }
  .msg .think {
    margin: 8px 12px; padding: 8px 10px; border-left: 3px solid #a8a29e; background: #fafaf9;
    color: #57534e; font-size: 12px; white-space: pre-wrap;
  }
  .msg .glm {
    margin: 8px 12px 12px; padding: 8px 10px; border-left: 3px solid var(--glm); background: #f5f3ff;
    color: #5b21b6; font-size: 12px; white-space: pre-wrap;
  }
  .msg .tools {
    margin: 0 12px 12px; padding: 8px 10px; background: #f8fafc; border-radius: 8px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px;
    white-space: pre-wrap; overflow: auto; max-height: 280px;
  }
  .sys {
    max-height: 160px; overflow: auto; font-size: 11px; color: var(--muted);
    white-space: pre-wrap; border: 1px dashed var(--line); border-radius: 8px; padding: 8px;
  }
  .err { color: var(--danger); padding: 12px; }
</style>
</head>
<body>
<header>
  <h1>Rollout Trajectory Viewer</h1>
  <select id="dumpSelect"></select>
  <input id="q" type="search" placeholder="filter label..." style="width:180px"/>
  <label class="chk"><input type="checkbox" id="offOnly"/> offload</label>
  <label class="chk"><input type="checkbox" id="unsolvedOnly"/> unsolved</label>
  <label class="chk"><input type="checkbox" id="showSystem"/> system prompt</label>
  <button id="reloadBtn" class="secondary">Reload list</button>
  <span class="meta" id="summaryMeta"></span>
</header>
<aside id="sidebar"><div class="empty">Loading dumps…</div></aside>
<main id="main"><div class="empty">Select a trajectory on the left</div></main>
<script>
const state = {
  dumps: [],
  dumpId: null,
  trajs: [],
  activeKey: null,
  detail: null,
  activeSeg: 0,
};

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.json();
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function keyOf(r) {
  return `${r.label}||${r.group_index}||${r.index}`;
}

async function boot() {
  const data = await api('/api/dumps');
  state.dumps = data.dumps || [];
  const sel = document.getElementById('dumpSelect');
  sel.innerHTML = state.dumps.map(d => `<option value="${esc(d.id)}">${esc(d.name)}</option>`).join('');
  if (state.dumps.length) {
    state.dumpId = state.dumps[0].id;
    sel.value = state.dumpId;
    await refreshList();
  } else {
    document.getElementById('sidebar').innerHTML = '<div class="empty">No .pt dumps</div>';
  }
  sel.onchange = async () => { state.dumpId = sel.value; state.activeKey = null; await refreshList(); };
  document.getElementById('reloadBtn').onclick = refreshList;
  document.getElementById('q').oninput = debounce(refreshList, 250);
  document.getElementById('offOnly').onchange = refreshList;
  document.getElementById('unsolvedOnly').onchange = refreshList;
  document.getElementById('showSystem').onchange = () => {
    if (state.activeKey) openTraj(...state.activeKey.split('||'));
  };
}

function debounce(fn, ms) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

async function refreshList() {
  if (!state.dumpId) return;
  const q = document.getElementById('q').value.trim();
  const off = document.getElementById('offOnly').checked;
  const un = document.getElementById('unsolvedOnly').checked;
  const params = new URLSearchParams({ dump: state.dumpId, limit: '300' });
  if (q) params.set('q', q);
  if (off) params.set('offload', '1');
  if (un) params.set('unsolved', '1');
  const [sum, list] = await Promise.all([
    api('/api/summary?' + new URLSearchParams({ dump: state.dumpId })),
    api('/api/trajs?' + params),
  ]);
  document.getElementById('summaryMeta').textContent =
    `${sum.name} · trajs ${sum.n_trajs} · segs ${sum.n_samples} · offload ${sum.n_offload} · solved ${sum.n_solved}/${sum.n_trajs} · trace ${sum.n_with_trace}`;
  state.trajs = list.trajs || [];
  renderSidebar();
}

function renderSidebar() {
  const box = document.getElementById('sidebar');
  if (!state.trajs.length) {
    box.innerHTML = '<div class="empty">No trajectories match</div>';
    return;
  }
  box.innerHTML = state.trajs.map(r => {
    const k = keyOf(r);
    const active = state.activeKey === k ? 'active' : '';
    const badges = [
      r.offload_count ? `<span class="badge off">off×${r.offload_count}</span>` : '',
      r.fail_passed ? `<span class="badge ok">passed</span>` : `<span class="badge bad">${esc(r.fail_summary || 'unsolved')}</span>`,
      r.empty_patch ? `<span class="badge">empty_patch</span>` : '',
      (r.seg_kinds && r.seg_kinds.TOKEN_FORK) ? `<span class="badge fork">fork×${r.seg_kinds.TOKEN_FORK}</span>` : '',
      (r.seg_kinds && r.seg_kinds.TREE_LEAF) ? `<span class="badge leaf">leaf×${r.seg_kinds.TREE_LEAF}</span>` : '',
    ].join('');
    return `<div class="traj ${active}" data-key="${esc(k)}">
      <div class="title">${esc(r.label)}</div>
      <div class="sub">
        <span>g${esc(r.group_index)} i${esc(r.index)}</span>
        <span>segs ${esc(r.n_seg)}</span>
        <span>rew ${_fmt(r.reward)}</span>
        <span>respΣ ${esc(r.resp_sum)}</span>
        ${badges}
      </div>
    </div>`;
  }).join('');
  box.querySelectorAll('.traj').forEach(el => {
    el.onclick = () => {
      const [label, g, i] = el.dataset.key.split('||');
      openTraj(label, g, i);
    };
  });
}

function _fmt(v) {
  if (v == null) return '-';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return esc(v);
}

async function openTraj(label, g, i) {
  state.activeKey = `${label}||${g}||${i}`;
  renderSidebar();
  const main = document.getElementById('main');
  main.innerHTML = '<div class="empty">Decoding trajectory…</div>';
  const params = new URLSearchParams({
    dump: state.dumpId, label, group: g, index: i,
  });
  if (document.getElementById('showSystem').checked) params.set('system', '1');
  try {
    const detail = await api('/api/traj?' + params);
    state.detail = detail;
    state.activeSeg = 0;
    // prefer first seg with offload
    const idx = (detail.segments || []).findIndex(s => (s.stats.offload_in_response || []).length);
    if (idx >= 0) state.activeSeg = idx;
    renderDetail();
  } catch (e) {
    main.innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
}

function renderDetail() {
  const d = state.detail;
  const main = document.getElementById('main');
  if (!d) return;
  const segs = d.segments || [];
  const fail = d.failure || {};
  const tabs = segs.map(s => {
    const hasOff = (s.stats.offload_in_response || []).length > 0;
    const why = s.why_segment || {};
    const active = s.segment_id === state.activeSeg ? 'active' : '';
    const kind = why.kind || '';
    return `<button class="${active} ${hasOff ? 'has-off' : ''}" data-seg="${s.segment_id}" title="${esc(why.summary || '')}">
      seg ${String(s.segment_id).padStart(2,'0')} · ${esc(kind || '?')} · resp ${s.stats.response_length}${hasOff ? ' · OFF' : ''}
    </button>`;
  }).join('');

  const seg = segs.find(s => s.segment_id === state.activeSeg) || segs[0];
  const st = seg?.stats || {};
  const why = seg?.why_segment || {};
  const offs = (st.offload_in_response || []).map(o => `N=${o.n}(${o.reasoning_effort})`).join(', ') || '—';

  const failClass = fail.passed ? 'pass' : '';
  const failNotes = (fail.notes || []).map(n => `<li>${esc(n)}</li>`).join('');
  const failHtml = `<div class="fail-box ${failClass}">
    <h2>${fail.passed ? '✓ Case 通过' : '✗ Case 未通过'} — ${esc(fail.summary || '')}</h2>
    <div class="kv">
      <div class="k">codes</div><div>${esc((fail.codes || []).join(', '))}</div>
      <div class="k">grading / empty / exit</div>
      <div>${esc(d.grading_solved)} / ${esc(d.empty_patch)} / ${esc((d.metadata||{}).agent_exit_code)}</div>
      <div class="k">reward</div><div>${_fmt(d.reward)}</div>
    </div>
    ${failNotes ? `<ul>${failNotes}</ul>` : ''}
  </div>`;

  let sysHtml = '';
  if (seg?.system_prompt) {
    sysHtml = `<div class="panel"><h2>System prompt</h2><div class="sys">${esc(seg.system_prompt)}</div></div>`;
  } else if (seg?.system_prompt_chars) {
    sysHtml = `<div class="panel"><h2>System prompt</h2><div class="meta">hidden (${seg.system_prompt_chars} chars) — enable “system prompt”</div></div>`;
  }

  const msgs = (seg?.messages || []).map(renderMsg).join('');
  const whyHtml = why.detail ? `<div class="why-box"><strong>为何有这个 seg（${esc(why.kind)} / ${esc(why.branch)}）</strong>\n${esc(why.detail)}</div>` : '';

  main.innerHTML = `
    <div class="panel">
      <h2>${esc(d.label)}</h2>
      <div class="kv">
        <div class="k">group / index</div><div>${esc(d.group_index)} / ${esc(d.index)}</div>
        <div class="k">reward / solved</div><div>${_fmt(d.reward)} / ${esc(d.solved)} (empty_patch=${esc(d.empty_patch)})</div>
        <div class="k">offload_stats</div><div><code>${esc(JSON.stringify(d.offload_stats || {}))}</code></div>
        <div class="k">segments</div><div>${esc(d.n_segments)}</div>
      </div>
    </div>
    ${failHtml}
    <div class="seg-tabs">${tabs}</div>
    <div class="panel">
      <h2>Segment ${String(seg?.segment_id).padStart(2,'0')} · ${esc(why.kind || '')} · ${esc(why.summary || '')}</h2>
      <div class="kv">
        <div class="k">tokens</div><div>prompt ${esc(st.prompt_tokens)} + resp ${esc(st.response_length)} = ${esc(st.n_tokens)}</div>
        <div class="k">loss_mask</div><div>1→${esc(st.loss_mask_sum)} / 0→${esc(st.loss_mask_zeros)}</div>
        <div class="k">offload in resp</div><div>${esc(offs)}</div>
        <div class="k">after close / mask0</div><div>${esc(st.after_close_tokens)} / ${esc(st.mask0_after_close)}</div>
        <div class="k">parent / LCP / drift</div>
        <div>seg${esc(why.parent_seg)} / ${esc(why.lcp)} / ${esc(why.drift)}</div>
      </div>
      ${whyHtml}
    </div>
    ${sysHtml}
    <div class="panel"><h2>Messages (${(seg?.messages||[]).length})</h2>${msgs || '<div class="empty">No messages parsed</div>'}</div>
  `;
  main.querySelectorAll('.seg-tabs button').forEach(b => {
    b.onclick = () => { state.activeSeg = Number(b.dataset.seg); renderDetail(); };
  });
}

function renderMsg(m) {
  const role = m.role || 'unknown';
  let badges = '';
  if (m.offload) badges += `<span class="badge off">offload N=${esc(m.offload.n)} ${esc(m.offload.reasoning_effort||'')}</span>`;
  if (m.truncated) badges += `<span class="badge">truncated</span>`;
  let body = '';
  if (m.reasoning_content) {
    body += `<div class="think"><strong>think / SLM</strong>\n${esc(m.reasoning_content)}</div>`;
  }
  if (m.glm_response) {
    const g = m.glm_response;
    let gtxt = '';
    if (g.reasoning_content) gtxt += `think:\n${g.reasoning_content}\n\n`;
    if (g.content) gtxt += `content:\n${g.content}`;
    if (!gtxt && g.tool_calls) gtxt = '(tool_calls only)';
    body += `<div class="glm"><strong>GLM embed</strong> <span class="badge">${esc(g.source||'')}</span>\n${esc(gtxt)}</div>`;
  }
  if (m.content) body += `<div class="body">${esc(m.content)}</div>`;
  if (m.tool_calls) {
    body += `<div class="tools">${esc(JSON.stringify(m.tool_calls, null, 2))}</div>`;
  }
  if (!body) body = `<div class="body" style="color:var(--muted)">(empty)</div>`;
  return `<div class="msg ${esc(role)}"><div class="bar">${esc(role)} ${badges}</div>${body}</div>`;
}

boot().catch(e => {
  document.getElementById('main').innerHTML = `<div class="err">${esc(e)}</div>`;
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class ViewerHandler(BaseHTTPRequestHandler):
    store: DumpStore

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _error(self, code: int, msg: str) -> None:
        self._json(code, {"error": msg})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        try:
            if path in ("/", "/index.html"):
                self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/dumps":
                self._json(200, {"dumps": self.store.list_dumps()})
                return
            if path == "/api/summary":
                dump = (qs.get("dump") or [None])[0]
                if not dump:
                    return self._error(400, "missing dump")
                self._json(200, self.store.summary(dump))
                return
            if path == "/api/trajs":
                dump = (qs.get("dump") or [None])[0]
                if not dump:
                    return self._error(400, "missing dump")
                self._json(
                    200,
                    {
                        "trajs": self.store.list_trajs(
                            dump,
                            offload=(qs.get("offload") or ["0"])[0] in ("1", "true", "yes"),
                            unsolved=(qs.get("unsolved") or ["0"])[0] in ("1", "true", "yes"),
                            solved=(qs.get("solved") or ["0"])[0] in ("1", "true", "yes"),
                            q=(qs.get("q") or [""])[0],
                            limit=int((qs.get("limit") or ["200"])[0]),
                        )
                    },
                )
                return
            if path == "/api/traj":
                dump = (qs.get("dump") or [None])[0]
                label = (qs.get("label") or [None])[0]
                if not dump or not label:
                    return self._error(400, "missing dump/label")
                group = int((qs.get("group") or qs.get("group_index") or ["0"])[0])
                index = int((qs.get("index") or ["0"])[0])
                seg_raw = (qs.get("seg") or [None])[0]
                seg = int(seg_raw) if seg_raw not in (None, "") else None
                include_system = (qs.get("system") or ["0"])[0] in ("1", "true", "yes")
                detail = self.store.get_traj(
                    dump,
                    label=label,
                    group_index=group,
                    index=index,
                    seg=seg,
                    include_system=include_system,
                )
                self._json(200, detail)
                return
            self._error(404, f"not found: {path}")
        except KeyError as exc:
            self._error(404, str(exc))
        except Exception as exc:  # noqa: BLE001
            self._error(500, f"{type(exc).__name__}: {exc}")


def resolve_pt_paths(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_file():
        if path.suffix != ".pt":
            raise SystemExit(f"expected .pt file, got {path}")
        return [path]
    if path.is_dir():
        pts = sorted(path.glob("rollout_*.pt")) or sorted(path.glob("*.pt"))
        if not pts:
            raise SystemExit(f"no .pt files in {path}")
        return pts
    raise SystemExit(f"path not found: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="A rollout_*.pt file or a directory of them")
    p.add_argument(
        "--tokenizer",
        "-t",
        type=str,
        default=None,
        help="HF tokenizer path (required to decode messages)",
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--offload-close-id", type=int, default=DEFAULT_OFFLOAD_CLOSE_ID)
    p.add_argument("--preload", action="store_true", help="Load first dump at startup")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pts = resolve_pt_paths(args.path)
    if not args.tokenizer:
        # try infer from common env / sibling
        guess = Path("/workspace/models/pyromind/PyroDash-4B-SFT-0727_pad248320")
        if guess.exists():
            args.tokenizer = str(guess)
            print(f"[traj_viewer] using default tokenizer {args.tokenizer}", flush=True)
        else:
            print("WARNING: --tokenizer not set; opening a traj will fail until provided", flush=True)

    store = DumpStore(pts, args.tokenizer, args.offload_close_id)
    if args.preload and pts:
        store.summary(str(pts[0]))

    ViewerHandler.store = store
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"Trajectory viewer: http://127.0.0.1:{args.port}/", flush=True)
    print(f"dumps ({len(pts)}):", flush=True)
    for p in pts:
        print(f"  - {p}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)


if __name__ == "__main__":
    main()
