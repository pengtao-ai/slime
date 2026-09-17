#!/usr/bin/env python3
"""Score mini-swe-agent ``*.traj.json`` trajectories with an LLM and emit HTML.

Example::

    export DASHSCOPE_BASE_URL=http://208.64.254.189:8001/v1
    export DASHSCOPE_API_KEY=sk-...
    export DASHSCOPE_MODEL=deepseek-v4-flash-0731

    python examples/coding_agent_rl/analysis/score_miniswe_traj_html.py \\
      --traj-dir examples/coding_agent_rl/data/test_traj \\
      --out-dir examples/coding_agent_rl/data/test_traj/credit_assign
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
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "http://208.64.254.189:8000/v1")
DEFAULT_MODEL = os.environ.get("DASHSCOPE_MODEL", "deepseek-v4-flash-0731")
DEFAULT_API_KEY = (
    os.environ.get("DASHSCOPE_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
).strip()

JUDGE_SYSTEM = """你是 coding-agent 长轨迹的过程监督（process supervision）评判员。

你的任务是：对输入中列出的每一轮 agent 行为分别评分。
你只能依据：
1. `problem`（任务/PR 描述，用来判断是否任务相关）
2. 当前轮的 command / observation / returncode（以及必要时的 reasoning）
3. `prior_chunk` 中更早轮次的命令摘要、重复统计和成败
4. `solved` 与 `outcome_reward`（harness resolved，不是 Submitted）

不要根据“看起来像好行为”主观加分。
每一轮都必须独立给出一个 score，但评分时必须考虑历史上下文和跨轮重复。
命中多档时取最高一档；未验证的 edit 不超过 +0.5。后续验证成功记在验证那一轮，不回写前面的 edit。

==================================================
一、核心原则
==================================================

评分目标不是评价代码质量，而是评价“这一轮动作对解决任务的过程贡献”。

优先级从高到低：

1. 是否直接推动任务解决
2. 是否产生新的、任务相关的信息
3. 是否减少关键不确定性
4. 是否只是重复已有工作
5. 是否产生负面影响或导致回退

注意：
- “有输出” ≠ “有信息增益”
- “执行了命令” ≠ “推进”
- “最终 solved” ≠ “所有前面的动作都有价值”
- 错误尝试本身不一定扣分，只要它是有意且有信息的探索/验证
- 只有第一次出现的行为可以获得对应的信息增益奖励
- 重复行为必须随着次数增加而降分

==================================================
二、评分范围
==================================================

score ∈ [-1.0, 1.0]

【+0.7 ~ +1.0 关键推进】
满足以下任一条件：
- 有效修改任务相关源码，并明显朝正确修复方向推进
- 修复后验证由 fail → pass
- 完成与任务直接相关且正确的 submit
- 直接完成导致 SWE-bench harness resolved 的关键动作
- 本轮 observation 已能证明修复有效（例如本轮测试 fail→pass）；不要用尚未发生的后续轮次给当前 edit 加分

【+0.4 ~ +0.6 明显推进】
- 首次定位核心根因
- 首次复现目标 bug，并获得关键错误信息
- 第一次有效 edit，且修改明显针对根因
- edit 后第一次有信息量的验证
- 新测试明确缩小问题范围
- 获得对后续修复具有直接指导意义的新证据

【+0.1 ~ +0.3 有价值探索】
- 首次阅读新的相关文件
- 首次阅读同一文件中的新函数/新代码区间
- 首次 grep 新模式或搜索新的相关符号
- 首次环境检查，并获得任务相关信息
- 首次运行相关测试但结果没有直接定位根因
- 首次探索一个合理但尚未证实的方向

正常探索通常不要超过 +0.3。

【0.0 中性】
- 第二次实质相同的动作
- 纯确认、重复查看已知信息
- 没有产生新信息但也没有明显危害
- 常规命令执行但对解决问题没有新增贡献
- 错误尝试没有产生新信息，但尚未达到明显空转程度

【-0.2 ~ -0.3 空转】
- 第三次及以后重复同一实质意图
- 连续重复读取相同内容且没有新结论
- 相同失败反复出现
- 明显无效的试错
- 已经获得足够信息后仍重复相同探索
- 长时间停留在已经排除的方向

【-0.5 ~ -1.0 有害】
- 明显破坏已有正确修改
- 无理由整文件覆盖、还原、破坏环境
- 修复 → 错误回退 → 再修复的明显死循环
- 删除/修改大量无关代码导致任务恶化
- 明确导致已有测试从 pass 变 fail
- 其他对任务造成实质负面影响的行为

==================================================
三、重复判定
==================================================

重复必须按“意图”判断，而不是简单字符串匹配。

【第一次】
第一次执行某种实质行为时：
- 按实际信息增益评分
- 即使后面大量重复，也不能回头修改第一次的分数

【第二次】
同一实质行为第二次出现：
- score ≤ 0.0
- 如果第二次确实产生了新的任务相关信息，可以给 0.0 ~ +0.1，但通常不要正分

【第三次及以后】
同一实质行为第三次或更多：
- score ≤ -0.3
- 如果只是重复确认，优先 -0.3
- 如果明显浪费大量轨迹或持续重复失败，可进一步降低

以下情况视为“实质相同”：
- 同一脚本
- 同一 pytest 节点
- 同一段 python -c / python 逻辑
- 同一文件、同一函数、同一代码区域的重复查看
- 同一 grep 意图
- 同一失败测试的重复运行
- 同一修复方向的无实质变化重复尝试

以下情况不要判为重复：
- sed/cat/read 使用不同且明显不重叠的代码区间
- 阅读同一文件中的不同函数
- grep 使用新的搜索模式
- grep 针对不同文件
- pytest 从不同测试节点切换
- edit 后第一次验证
- 同一个测试由 fail → pass
- 同一命令参数发生实质变化，并且变化产生了新的信息

注意：
pytest / python 命令后使用 head/tail 等截断输出，
只是改变输出展示方式，不代表读取了新的信息，因此仍属于同一意图。

“再确认一下”“重新看看”“再跑一次”不能自动获得正分，
仍然按照第 2 / 第 3 次重复规则处理。

==================================================
四、碎片化阅读
==================================================

重点识别“为了看而看”的碎读行为。

如果连续多轮：
- 阅读同一个文件
- 阅读窗口高度重叠
- 没有产生新的结论
- 没有伴随新的搜索假设或修复动作

则：
- 第一次：正常按探索评分
- 第二次：≤ 0
- 第三次及以后：≤ -0.2

但是：
同一文件不同函数、不同代码区域、不同调用链上的渐进式阅读，
如果能够减少关键不确定性，应视为新的探索，可以给 +0.1 ~ +0.3。

==================================================
五、失败与错误尝试
==================================================

失败不能简单等同于负分。

以下情况可以给正分：
- 首次有意复现 bug
- 首次运行测试并暴露新的错误
- 首次获得新的 traceback
- 首次验证某个假设并明确证明该方向错误
- 首次发现环境/依赖/接口问题

如果失败结果提供了新的任务相关信息：
→ +0.1 ~ +0.5

如果只是重复之前已经看到的失败：
→ 第二次 ≤ 0
→ 第三次及以后 ≤ -0.3

以下情况应明显扣分：
- 打错路径后机械重复
- 同一错误命令反复执行
- 明知环境有问题却持续重复相同操作
- 修改文件导致错误后又原地重复相同修改
- 没有新假设，只是盲目 retry

==================================================
六、代码修改与验证
==================================================

代码修改必须看“是否有证据支持”，不能仅因为出现 edit 就高分。

【高价值 edit】
- 修改与已定位根因直接相关
- 修改范围合理
- 修改后测试结果支持该方向
→ +0.4 ~ +0.8

【普通 edit】
- 针对合理假设进行小范围修改
- 但尚未验证
→ +0.3 ~ +0.5

【低价值 edit】
- 没有明确根因依据
- 大量试探性修改
- 修改与任务关系弱
→ 0 ~ +0.2

【有害 edit】
- 修改无关代码
- 覆盖大量文件
- 破坏已有修复
- 导致原本通过的测试失败
→ -0.5 ~ -1.0

验证具有特殊优先级：

如果 edit 之后第一次验证：
- fail → pass：+0.7 ~ +1.0
- 暴露新的、更接近根因的错误：+0.3 ~ +0.6
- fail 且没有新信息：0 ~ +0.1

如果验证已经 pass 后再次重复验证：
→ 不因为“确认正确”而继续给高分。

==================================================
七、submit 与 solved
==================================================

`solved` 指：
SWE-bench harness resolved。

不是：
- Submitted
- Patch generated
- Test locally passed
- Agent 宣称完成

如果 `solved = false`：
- 不得因为 submit、完成描述或“看起来正确”而给关键步高分
- 错误 submit 通常 ≤ +0.2，除非产生新的有效信息
- 有信息增益的探索/首次失败仍按第二～五节给分，不要因为最终未 resolved 就把前面合理探索一律压低

如果 `solved = true`：
- 导致 resolved 的关键动作必须明显高于普通探索和空转
- 最终正确 submit / 关键修复 / 通过 harness 验证通常应在 +0.7 ~ +1.0
- 不能因为最终 solved，就把之前所有探索轮自动提高分数
- 如果 solved 之前存在大量重复/空转，仍然照常扣分
- solved 后的重复操作不应获得额外奖励

特别注意：
如果某轮只是“提交”，但真正的修复早已完成，
则应根据该轮实际贡献评分，而不是机械给 +1.0。

==================================================
八、跨 chunk 历史
==================================================

`prior_chunk` 表示上一段轨迹的命令摘要、重复统计和上下文。

必须使用 `prior_chunk` 判断：
- 当前动作是否已经在之前 chunk 出现
- 当前搜索/测试是否属于跨 chunk 重复
- 某个失败是否已经发生过
- 某个文件/函数是否已经被充分探索
- 当前动作是否只是重新执行之前的工作

如果当前动作在 `prior_chunk` 中已经明确出现：
→ 不应再次获得“首次探索”奖励。

但：
- 如果当前动作虽然形式相同，但上下文发生实质变化
- 或者之前 fail，现在出现新的证据 / fail → pass

则可以重新按实际信息增益评分。

非常重要：
不能因为历史中存在相似命令，就回头修改历史轮次的评分。
每一轮只根据“截至该轮可获得的信息”评分。

==================================================
九、评分决策顺序
==================================================

对每一轮严格按照以下顺序判断：

Step 1：判断是否有明显有害行为
→ 有：优先考虑 -0.5 ~ -1.0，结束本轮

Step 2：先排除「不要判为重复」的情形（第三节）：不同代码窗口、新 grep、不同测试节点、edit 后第一次验证、fail→pass、参数实质变化且有新信息
→ 属于豁免：跳到 Step 4，不要按重复次数压分

Step 3：若非豁免，判断是否实质重复
→ 第 1 次：不因“后面还会重复”而扣分，继续 Step 4
→ 第 2 次：≤ 0（通常不要正分）
→ 第 3 次+：≤ -0.3

Step 4：判断是否产生新的任务相关信息 / 根因定位 / 合理 edit
→ 探索 +0.1 ~ +0.3；首次根因/复现/有信息验证 +0.4 ~ +0.6；未验证 edit ≤ +0.5

Step 5：判断本轮 observation 是否直接表明 fail → pass / 正确 submit
→ 有：+0.7 ~ +1.0

Step 6：如果没有明显贡献
→ 0 或轻微负分

最终选择最接近的一档，不要为了“看起来公平”而人为平均分布。

==================================================
十、重要边界
==================================================

1. 不要把“运行测试”本身视为推进，关键是测试是否产生新信息。
2. 不要把“读代码”本身视为高价值，关键是是否缩小了问题范围。
3. 不要把“修改代码”自动视为正确推进，必须结合修改目的和后续证据。
4. 不要因为 agent 最终 solved 就给前面的所有动作奖励。
5. 不要因为一次失败就扣分；有信息的首次失败通常是正向过程信号。
6. 不要因为命令不同就认为不是重复，要按意图判断。
7. 不要因为命令字符串相同就一定判重复，如果上下文/目标发生实质变化，应重新判断。
8. 不要回溯修改之前轮次的评分。
9. 正分应该稀缺，尤其是 +0.5 以上的分数。
10. 长轨迹中大量无效探索、重复测试和碎读，应能够体现为明显的负奖励。

==================================================
输出格式
==================================================

必须给输入中列出的每一轮打分。
turn 编号必须保持原值。
reason 必须是一句中文，短于 40 个字。

只输出一个 JSON 对象，不要解释，不要 markdown，不要代码围栏。

格式：
{
  "turns": [
    {"turn": 0, "score": 0.3, "reason": "首次定位相关函数"},
    {"turn": 1, "score": 0.0, "reason": "重复确认已有信息"}
  ],
  "summary": "总体推进情况简述。一两句话。"
}
"""


def _trunc(text: Any, n: int) -> str:
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    s = s.replace("\r\n", "\n").strip()
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


def _extract_json_obj(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    candidates: list[str] = [text]
    # Prefer an object that contains "turns"
    for m in re.finditer(r"\{[^{}]*\"turns\"[\s\S]*", text):
        candidates.append(m.group(0))
    # Balanced-ish: from first { to last }
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])

    errors: list[str] = []
    for cand in candidates:
        for variant in (cand, re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", cand))):
            try:
                obj = json.loads(variant)
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
                continue
            if isinstance(obj, dict) and isinstance(obj.get("turns"), list):
                return obj
            if isinstance(obj, dict):
                return obj
    raise ValueError(f"judge response is not JSON: {text[:400]} | errs={errors[:2]}")


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


def _tool_calls_brief(tool_calls: Any, limit: int = 350) -> str:
    if not tool_calls:
        return ""
    parts: list[str] = []
    for c in tool_calls[:4]:
        if not isinstance(c, dict):
            continue
        fn = c.get("function") if isinstance(c.get("function"), dict) else c
        name = (fn or {}).get("name") or "bash"
        args = (fn or {}).get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        if isinstance(args, dict) and "command" in args:
            parts.append(f"{name}: {_trunc(args.get('command'), 220)}")
        else:
            parts.append(f"{name}({_trunc(args, 180)})")
    return _trunc("; ".join(parts), limit)


def _cmds_from_message(msg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    extra = msg.get("extra") if isinstance(msg.get("extra"), dict) else {}
    for a in extra.get("actions") or []:
        if isinstance(a, dict) and a.get("command"):
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
        if isinstance(args, dict) and args.get("command"):
            out.append(str(args["command"]))
    return out


def _returncodes(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"<returncode>(-?\d+)</returncode>", text or "")]


def _head_tail(text: str, *, head: int = 180, tail: int = 160) -> str:
    s = (text or "").replace("\r\n", "\n").strip()
    if len(s) <= head + tail + 1:
        return s
    return s[:head] + "\n…\n" + s[-tail:]


def extract_problem(messages: list[dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            text = str(m["content"])
            # Prefer PR description block
            m_pr = re.search(
                r"<pr_description>\s*(.*?)\s*</pr_description>",
                text,
                flags=re.S | re.I,
            )
            if m_pr:
                return m_pr.group(1).strip()
            return text
    return ""


def messages_to_turns(messages: list[dict[str, Any]], *, text_limit: int = 360) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if m.get("role") != "assistant":
            i += 1
            continue
        reason = m.get("reasoning_content") or m.get("content") or ""
        tools = _tool_calls_brief(m.get("tool_calls"))
        cmds = _cmds_from_message(m)
        obs_raw: list[str] = []
        j = i + 1
        while j < n and messages[j].get("role") == "tool":
            obs_raw.append(str(messages[j].get("content") or ""))
            j += 1
        obs_full = "\n".join(obs_raw)
        rcs = _returncodes(obs_full)
        turns.append(
            {
                "turn": len(turns),
                "reasoning": _trunc(reason, text_limit),
                "content": _trunc(m.get("content") or "", text_limit // 2),
                "tool_calls": tools,
                "cmds": cmds,
                "rcs": rcs,
                "observation": _head_tail(obs_full, head=180, tail=160) or _trunc(obs_full, text_limit),
                "valid_offload": "<|llm_offload|>" in str(reason) or "<|llm_offload|>" in tools,
            }
        )
        i = j if j > i + 1 else i + 1
    return turns


def load_traj(path: Path, *, resolved: bool | None = None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "messages" not in data:
        raise ValueError(f"not a mini-swe traj: {path}")
    info = data.get("info") or {}
    exit_status = str(info.get("exit_status") or "")
    submission = str(info.get("submission") or "")
    submitted = exit_status == "Submitted" and bool(submission.strip())
    if resolved is None:
        solved = submitted
        resolved_source = "submitted"
    else:
        solved = bool(resolved)
        resolved_source = "harness"
    messages = list(data.get("messages") or [])
    problem = extract_problem(messages)
    turns = messages_to_turns(messages)
    return {
        "path": str(path),
        "instance_id": data.get("instance_id") or path.stem.replace(".traj", ""),
        "exit_status": exit_status,
        "submitted": submitted,
        "solved": solved,
        "resolved_source": resolved_source,
        "outcome_reward": 1.0 if solved else 0.0,
        "submission": submission,
        "problem": problem,
        "n_turns": len(turns),
        "turns": turns,
        "api_calls": (info.get("model_stats") or {}).get("api_calls"),
    }


def _intent_key(cmd: str) -> str:
    """Intent fingerprint: keep sed windows; strip trailing head/tail (output truncation)."""
    c = re.sub(r"\s+", " ", (cmd or "").strip())
    c = re.sub(r"\s*\|\s*(?:head|tail)\s+-n?\s*\d+\s*$", "", c, flags=re.I)
    return c[:120]


def _file_from_cmd(cmd: str) -> str | None:
    m = re.findall(r"(?:/testbed/|\./)?([\w./-]+\.py)", cmd or "")
    return m[0] if m else None


def _is_read_cmd(cmd: str) -> bool:
    low = (cmd or "").lower()
    if re.search(r"\b(pytest|unittest|tox|runtests)\b", low):
        return False
    if re.search(r"\bpython(?:3)?\s+(-m|-c)\b", low):
        return False
    if re.search(r"\bpython(?:3)?\s+\S+\.py\b", low):
        return False
    return bool(re.search(r"(^|[;&]\s*)(cat\s+-n|cat\s|nl\s|sed\s+-n|head\s|tail\s|awk\s)", low))


def _turn_cmd(t: dict[str, Any]) -> str:
    cmds = t.get("cmds") or []
    if cmds:
        return str(cmds[0])
    return str(t.get("tool_calls") or "")


def build_prior_chunk_context(prev_turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Cumulative history for the next judge chunk (cmd digest + repeat counters)."""
    from collections import Counter

    cmd_keys: Counter[str] = Counter()
    file_hits: Counter[str] = Counter()
    digests: list[dict[str, Any]] = []
    for t in prev_turns:
        cmd = _turn_cmd(t)
        key = _intent_key(cmd) if cmd else ""
        if key:
            cmd_keys[key] += 1
        fk = _file_from_cmd(cmd) if cmd and _is_read_cmd(cmd) else None
        if fk:
            file_hits[fk] += 1
        rcs = t.get("rcs") or []
        digests.append(
            {
                "turn": t.get("turn"),
                "cmd": _trunc(cmd, 120),
                "rc": rcs[-1] if rcs else None,
                "n": cmd_keys[key] if key else 0,
            }
        )
    top_rep = [{"cmd": k, "n": n} for k, n in cmd_keys.most_common(8) if n >= 2]
    top_files = [{"file": f, "n": n} for f, n in file_hits.most_common(5) if n >= 2]
    return {
        "turn_range": [prev_turns[0]["turn"], prev_turns[-1]["turn"]] if prev_turns else [],
        "n_turns": len(prev_turns),
        "cmd_digest": digests[-20:],
        "spin": {
            "repeat_cmds_ge2": sum(1 for n in cmd_keys.values() if n >= 2),
            "repeat_cmds_ge3": sum(1 for n in cmd_keys.values() if n >= 3),
            "reread_files_ge2": sum(1 for n in file_hits.values() if n >= 2),
            "top_repeated_cmds": top_rep,
            "top_reread_files": top_files,
        },
        "hint": "以上为更早轮次摘要。按意图判断是否重复；不同 sed 窗口 / 新 grep / edit 后首次验证 / fail→pass 不要当重复。",
    }


def normalize_scores(
    scored: list[dict[str, Any]],
    *,
    n_turns: int,
    outcome_reward: float,
    mode: str,
) -> list[dict[str, Any]]:
    by = {int(t["turn"]): t for t in scored if "turn" in t and "score" in t}
    filled: list[dict[str, Any]] = []
    for i in range(n_turns):
        if i in by:
            item = dict(by[i])
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
            u = float(outcome_reward) / max(n_turns, 1)
            for t in filled:
                t["score"] = u
            return filled
        scale = float(outcome_reward) / s
        for t in filled:
            t["score"] = float(t["score"]) * scale
        return filled
    if mode == "mean_to_outcome":
        mean = s / max(n_turns, 1)
        if abs(mean) < 1e-8:
            for t in filled:
                t["score"] = float(outcome_reward)
            return filled
        scale = float(outcome_reward) / mean
        for t in filled:
            t["score"] = float(t["score"]) * scale
        return filled
    raise ValueError(mode)


def heuristic_chunk(turns: list[dict[str, Any]], *, solved: bool, outcome: float) -> list[dict[str, Any]]:
    n = max(len(turns), 1)
    out = []
    for t in turns:
        i = int(t["turn"])
        # later turns get slightly more weight if solved; idle-looking steps lower
        score = (0.4 + 0.6 * (i + 1) / n) * (outcome if solved else 0.15)
        tools = t.get("tool_calls") or ""
        if "cat " in tools or "sed " in tools or "python" in tools:
            score += 0.05
        if not tools:
            score *= 0.3
        if not solved and i > n * 0.7:
            score -= 0.1
        score = max(-1.0, min(1.0, float(score)))
        out.append({"turn": i, "score": round(score, 4), "reason": "heuristic fallback"})
    return out


def judge_chunk(
    *,
    problem: str,
    solved: bool,
    outcome_reward: float,
    instance_id: str,
    n_turns: int,
    chunk: list[dict[str, Any]],
    chunk_idx: int,
    n_chunks: int,
    args: argparse.Namespace,
    prior_chunk: dict[str, Any] | None = None,
    resolved_source: str = "submitted",
) -> tuple[list[dict[str, Any]], str]:
    slim = []
    for t in chunk:
        cmds = list(t.get("cmds") or [])
        row: dict[str, Any] = {
            "turn": t["turn"],
            "reasoning": _trunc(t.get("reasoning") or "", 280),
            "cmd": _trunc(_turn_cmd(t), 260),
            "cmds": [_trunc(c, 220) for c in cmds[:3]],
            "rcs": t.get("rcs") or [],
            "observation": t.get("observation") or "",
        }
        if not cmds and t.get("content"):
            row["content"] = t["content"]
        slim.append(row)
    payload = {
        "instance_id": instance_id,
        "solved": solved,
        "resolved_source": resolved_source,
        "outcome_reward": outcome_reward,
        "n_turns_total": n_turns,
        "chunk_index": chunk_idx,
        "n_chunks": n_chunks,
        "turn_range": [chunk[0]["turn"], chunk[-1]["turn"]],
        "problem": _trunc(problem, 1800),
        "prior_chunk": prior_chunk,
        "turns": slim,
    }
    user = (
        f"这是第 {chunk_idx + 1}/{n_chunks} 段（共 {n_turns} 轮）。"
        "只给本段列出的 turn 打分，turn 编号保持原值。"
        "每轮含 cmd / rcs / observation（头尾，含 returncode）。"
        "prior_chunk 是本段之前全部历史的摘要，按 system 第三节判断是否重复。"
        "reason 短于40字。只输出 JSON。\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    if args.heuristic or not args.api_key:
        return heuristic_chunk(chunk, solved=solved, outcome=outcome_reward), "heuristic"

    last_err: Exception | None = None
    for attempt in range(max(1, int(args.retries))):
        try:
            content = call_chat(
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature if attempt == 0 else min(0.4, args.temperature + 0.1),
                timeout=args.timeout,
            )
            obj = _extract_json_obj(content)
            scored = list(obj.get("turns") or [])
            if not scored:
                raise ValueError("empty turns in judge JSON")
            return scored, str(obj.get("summary") or "")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(
                f"    retry chunk {chunk_idx + 1} attempt {attempt + 1}: {exc}",
                flush=True,
            )
    assert last_err is not None
    print(f"    chunk {chunk_idx + 1} fallback heuristic: {last_err}", flush=True)
    return heuristic_chunk(chunk, solved=solved, outcome=outcome_reward), f"chunk_fallback:{last_err}"


def score_traj(traj: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    turns = traj["turns"]
    chunk_size = max(1, int(args.chunk_size))
    chunks = [turns[i : i + chunk_size] for i in range(0, len(turns), chunk_size)]
    all_scored: list[dict[str, Any]] = []
    summaries: list[str] = []
    n_llm_chunks = 0
    n_fb_chunks = 0
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        prior = build_prior_chunk_context(turns[: chunk[0]["turn"]]) if ci > 0 else None
        scored, summary = judge_chunk(
            problem=traj["problem"],
            solved=traj["solved"],
            outcome_reward=traj["outcome_reward"],
            instance_id=traj["instance_id"],
            n_turns=traj["n_turns"],
            chunk=chunk,
            chunk_idx=ci,
            n_chunks=len(chunks),
            args=args,
            prior_chunk=prior,
            resolved_source=str(traj.get("resolved_source") or "submitted"),
        )
        all_scored.extend(scored)
        if summary.startswith("chunk_fallback") or summary == "heuristic":
            n_fb_chunks += 1
        else:
            n_llm_chunks += 1
            if summary:
                summaries.append(summary)
        print(
            f"  [{traj['instance_id']}] chunk {ci + 1}/{len(chunks)} "
            f"turns {chunk[0]['turn']}-{chunk[-1]['turn']} (+{len(scored)} scores)",
            flush=True,
        )
    normalized = normalize_scores(
        all_scored,
        n_turns=len(turns),
        outcome_reward=float(traj["outcome_reward"]),
        mode=args.normalize,
    )
    for item, ctx in zip(normalized, turns):
        item["context"] = ctx
    if args.heuristic or not args.api_key:
        mode = "heuristic"
    elif n_fb_chunks == 0:
        mode = "llm"
    elif n_llm_chunks == 0:
        mode = "heuristic_fallback"
    else:
        mode = "llm_partial"
    return {
        "instance_id": traj["instance_id"],
        "path": traj["path"],
        "exit_status": traj["exit_status"],
        "submitted": traj.get("submitted"),
        "solved": traj["solved"],
        "resolved_source": traj.get("resolved_source") or "submitted",
        "outcome_reward": traj["outcome_reward"],
        "n_turns": traj["n_turns"],
        "api_calls": traj.get("api_calls"),
        "problem": traj["problem"],
        "submission": traj.get("submission") or "",
        "judge_mode": mode,
        "judge_summary": " | ".join(summaries[:6]),
        "n_llm_chunks": n_llm_chunks,
        "n_fallback_chunks": n_fb_chunks,
        "normalize": args.normalize,
        "latency_s": round(time.time() - t0, 2),
        "llm_turn_rewards": normalized,
    }


def paint_turn_advantages(traj: dict[str, Any], *, a_s: float) -> None:
    items = list(traj.get("llm_turn_rewards") or [])
    scores = [float(x.get("score") or 0.0) for x in items]
    mean_r = (sum(scores) / len(scores)) if scores else 0.0
    traj["a_s"] = float(a_s)
    traj["mean_r"] = float(mean_r)
    for item, score in zip(items, scores):
        residual = float(score) - mean_r
        item["residual"] = residual
        item["a_s"] = float(a_s)
        item["advantage"] = float(a_s) + residual


def render_html(report: dict[str, Any]) -> str:
    title = html.escape(str(report.get("title") or "Mini-SWE Traj Credit"))
    payload = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
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
  --panel:#f7fafc; --side:#122033; --accent:#0b6e6a; --good:#1a6b4a; --bad:#a33b3b;
  --sans:"Source Sans 3","Noto Sans SC",sans-serif;
  --display:"Outfit",var(--sans); --mono:"IBM Plex Mono",ui-monospace,monospace;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:var(--sans); color:var(--ink); background:
  linear-gradient(135deg,#f4f8fb 0%,transparent 40%),
  radial-gradient(circle at 1px 1px, rgba(18,40,55,.07) 1px,transparent 0) var(--bg);
  background-size:auto,18px 18px; }}
.app {{ display:grid; grid-template-columns:18rem 1fr; min-height:100vh; }}
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
.stat b {{ display:block; font-family:var(--mono); font-size:.95rem; }}
.stat span {{ color:var(--muted); font-size:.75rem; }}
.summary {{ color:var(--muted); margin:.4rem 0 .8rem; }}
.problem {{ white-space:pre-wrap; font-size:.85rem; max-height:10rem; overflow:auto;
  background:#eef3f7; border:1px solid var(--line); padding:.6rem; }}
.turn {{ border-top:1px solid var(--line); padding:.7rem 0; }}
.turn-head {{ display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; }}
.badge {{ font-family:var(--mono); font-size:.68rem; padding:.1rem .35rem; border:1px solid var(--line); background:#fff; }}
.bar-wrap {{ flex:1; min-width:10rem; height:.55rem; background:#dde5ec; position:relative; }}
.bar {{ position:absolute; top:0; bottom:0; }}
.bar.pos {{ left:50%; background:var(--good); }}
.bar.neg {{ right:50%; background:var(--bad); }}
.score {{ font-family:var(--mono); font-weight:500; min-width:4.2rem; }}
.score.pos {{ color:var(--good); }}
.score.neg {{ color:var(--bad); }}
.adv {{ font-family:var(--mono); font-size:.78rem; color:var(--muted); }}
.reason {{ margin:.35rem 0; font-size:.88rem; }}
.details {{ font-family:var(--mono); font-size:.72rem; color:var(--muted); white-space:pre-wrap;
  background:#f1f5f8; border:1px solid var(--line); padding:.45rem; max-height:9rem; overflow:auto; }}
.spark {{ display:flex; align-items:flex-end; gap:1px; height:48px; margin:.5rem 0 0; overflow:auto; }}
.spark i {{ display:block; width:4px; min-width:3px; background:var(--good); }}
.spark i.neg {{ background:var(--bad); }}
</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <h1>Mini-SWE Credit</h1>
    <div class="meta" id="side-meta"></div>
    <div id="nav" style="margin-top:1rem"></div>
  </aside>
  <main class="main" id="main"></main>
</div>
<script id="report-data" type="application/json">{payload}</script>
<script>
const report = JSON.parse(document.getElementById('report-data').textContent);
const nav = document.getElementById('nav');
const main = document.getElementById('main');
const sideMeta = document.getElementById('side-meta');
sideMeta.innerHTML = [
  'model: ' + (report.model||''),
  'normalize: ' + (report.normalize||''),
  'trajs: ' + ((report.trajectories||[]).length),
  'A_s=0（不同题无 group）',
  'A_t = r_i − r̄',
].map(x => '<div>'+esc(x)+'</div>').join('');
let active = 0;
function esc(s) {{
  return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function scoreClass(x) {{ return Number(x) >= 0 ? 'pos' : 'neg'; }}
function barHtml(score, scale) {{
  const lim = Math.max(1, Number(scale) || 1);
  const s = Math.max(-lim, Math.min(lim, Number(score) || 0));
  const pct = (Math.abs(s) / lim) * 50;
  if (s >= 0) return `<div class="bar-wrap"><div class="bar pos" style="width:${{pct}}%"></div></div>`;
  return `<div class="bar-wrap"><div class="bar neg" style="width:${{pct}}%"></div></div>`;
}}
function fmt(x) {{
  const n = Number(x);
  if (!Number.isFinite(n)) return '—';
  return (n>=0?'+':'') + n.toFixed(3);
}}
function sparkHtml(rewards) {{
  const maxAbs = Math.max(...rewards.map(r => Math.abs(Number(r.advantage ?? r.score)||0)), 1e-6);
  return '<div class="spark">' + rewards.map(r => {{
    const s = Number(r.advantage ?? r.score)||0;
    const h = Math.max(2, Math.round(Math.abs(s)/maxAbs*46));
    return `<i class="${{s<0?'neg':''}}" style="height:${{h}}px" title="T${{r.turn}} A_t=${{fmt(s)}} r=${{Number(r.score).toFixed(3)}}"></i>`;
  }}).join('') + '</div>';
}}
function render() {{
  const trajs = report.trajectories || [];
  nav.innerHTML = trajs.map((t,i) => `<button class="nav-item ${{i===active?'active':''}}" data-i="${{i}}">
    <div class="id">${{esc(t.instance_id)}}</div>
    <div>${{t.solved?'✓ solved':'✗ '+esc(t.exit_status)}} · T=${{t.n_turns}}</div>
  </button>`).join('');
  nav.querySelectorAll('.nav-item').forEach(btn => btn.onclick = () => {{ active=Number(btn.dataset.i); render(); }});
  const t = trajs[active];
  if (!t) {{ main.innerHTML = '<div class="card">no traj</div>'; return; }}
  const rewards = t.llm_turn_rewards || [];
  const mean = Number(t.mean_r);
  const meanDisp = Number.isFinite(mean)
    ? mean
    : (rewards.length ? rewards.reduce((a,b)=>a+Number(b.score),0)/rewards.length : 0);
  const advScale = Math.max(1, ...rewards.map(x => Math.abs(Number(x.advantage)||0)));
  const turnsHtml = rewards.map(tr => {{
    const c = tr.context || {{}};
    const detail = [
      c.reasoning && ('reasoning: ' + c.reasoning),
      c.tool_calls && ('tools: ' + c.tool_calls),
      c.observation && ('obs: ' + c.observation),
    ].filter(Boolean).join('\\n\\n');
    return `<div class="turn" id="t${{tr.turn}}">
      <div class="turn-head">
        <span class="badge">T${{tr.turn}}</span>
        <span class="score ${{scoreClass(tr.advantage)}}">A_t ${{fmt(tr.advantage)}}</span>
        ${{barHtml(tr.advantage, advScale)}}
      </div>
      <div class="adv">r_i=${{Number(tr.score).toFixed(3)}} · r_i−r̄=${{fmt(tr.residual)}} · A_s=${{fmt(tr.a_s)}}</div>
      <div class="reason">${{esc(tr.reason||'')}}</div>
      ${{detail ? `<pre class="details">${{esc(detail)}}</pre>` : ''}}
    </div>`;
  }}).join('');
  main.innerHTML = `
    <div class="card">
      <div class="row">
        <div class="stat"><span>instance</span><b>${{esc(t.instance_id)}}</b></div>
        <div class="stat"><span>exit</span><b>${{esc(t.exit_status)}}</b></div>
        <div class="stat"><span>solved</span><b>${{t.solved}}</b></div>
        <div class="stat"><span>outcome</span><b>${{Number(t.outcome_reward).toFixed(2)}}</b></div>
        <div class="stat"><span>A_s</span><b>${{fmt(t.a_s)}}</b></div>
        <div class="stat"><span>turns</span><b>${{t.n_turns}}</b></div>
        <div class="stat"><span>mean r</span><b>${{meanDisp.toFixed(3)}}</b></div>
        <div class="stat"><span>latency</span><b>${{t.latency_s||0}}s</b></div>
      </div>
      <div class="summary">${{esc(t.judge_summary||'')}}</div>
      ${{sparkHtml(rewards)}}
      <details style="margin-top:.6rem"><summary>problem</summary><pre class="problem">${{esc(t.problem||'')}}</pre></details>
      ${{t.submission ? `<details><summary>submission</summary><pre class="problem">${{esc(t.submission)}}</pre></details>` : ''}}
    </div>
    <div class="card">
      <strong>逐步优势 A_t = r_i − r̄（无 group，A_s=0）</strong>
      ${{turnsHtml}}
    </div>`;
}}
render();
</script>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj-dir", type=Path, default=None)
    p.add_argument("--traj", type=Path, nargs="*", default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--normalize", choices=("raw", "sum_to_outcome", "mean_to_outcome"), default="raw")
    p.add_argument("--chunk-size", type=int, default=15)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--heuristic", action="store_true")
    p.add_argument("--concurrency", type=int, default=1, help="Parallel trajs (chunks stay sequential per traj)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--timeout", type=float, default=300.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths: list[Path] = []
    if args.traj:
        paths.extend(args.traj)
    if args.traj_dir:
        paths.extend(sorted(args.traj_dir.glob("*.traj.json")))
    # unique preserve order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    if not uniq:
        print("no traj files", file=sys.stderr)
        return 2
    if not args.heuristic and not args.api_key:
        print("ERROR: set DASHSCOPE_API_KEY or pass --heuristic", file=sys.stderr)
        return 2

    trajs = [load_traj(p) for p in uniq]
    for t in trajs:
        print(
            f"loaded {t['instance_id']}: turns={t['n_turns']} "
            f"exit={t['exit_status']} solved={t['solved']}",
            flush=True,
        )

    results: list[dict[str, Any] | None] = [None] * len(trajs)

    def _one(i_t: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        i, traj = i_t
        try:
            return i, score_traj(traj, args)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {traj['instance_id']}: {exc}", flush=True)
            # heuristic fallback for whole traj
            scored = heuristic_chunk(
                traj["turns"], solved=traj["solved"], outcome=traj["outcome_reward"]
            )
            normalized = normalize_scores(
                scored,
                n_turns=traj["n_turns"],
                outcome_reward=traj["outcome_reward"],
                mode=args.normalize,
            )
            for item, ctx in zip(normalized, traj["turns"]):
                item["context"] = ctx
            return i, {
                **{k: traj[k] for k in (
                    "instance_id", "path", "exit_status", "solved",
                    "outcome_reward", "n_turns", "api_calls", "problem",
                )},
                "submission": _trunc(traj.get("submission") or "", 2000),
                "judge_summary": f"error fallback: {exc}",
                "judge_mode": "error_fallback",
                "normalize": args.normalize,
                "latency_s": 0.0,
                "llm_turn_rewards": normalized,
                "error": str(exc),
            }

    workers = max(1, int(args.concurrency))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, (i, t)) for i, t in enumerate(trajs)]
        for fut in concurrent.futures.as_completed(futs):
            i, out = fut.result()
            results[i] = out
            print(
                f"done {out['instance_id']} mode={out['judge_mode']} "
                f"mean={sum(x['score'] for x in out['llm_turn_rewards'])/max(out['n_turns'],1):.3f} "
                f"{out['latency_s']}s",
                flush=True,
            )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "title": "Mini-SWE Traj LLM Turn Credit",
        "model": args.model if not args.heuristic else "heuristic",
        "normalize": args.normalize,
        "base_url": args.base_url,
        "trajectories": [r for r in results if r is not None],
    }
    for traj in report["trajectories"]:
        # Independent instances: no GRPO group, A_s = 0, A_t = r_i - mean(r).
        paint_turn_advantages(traj, a_s=0.0)
    json_path = out_dir / "credit_assign.json"
    html_path = out_dir / "credit_assign.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
