#!/usr/bin/env python3
"""Simulate a multi-turn coding-agent session with mid-turn offload; dump trajectory.

CPU-only (FakeSGLang + MockGLM). Saves:
  - ``trajectory.json``   human-readable agent rounds + GLM payloads
  - ``samples.pt``        trainable Sample segments (same shape as rollout dumps)
  - ``summary.json``      offload_stats / reward / loss_mask sums

Run::

    python examples/coding_agent_rl/sim_offload_multiturn_traj.py
    # optional: OUT_DIR=/tmp/my_traj
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

os.environ["SLIME_AGENT_OFFLOAD"] = "1"
os.environ["OFFLOAD_EFFICIENCY_LAMBDA"] = "0.05"
os.environ["OFFLOAD_MAX_TOKENS"] = "256"

from examples.coding_agent_rl import offload  # noqa: E402
from examples.coding_agent_rl.generate import CodingAnthropicAdapter  # noqa: E402
from slime.utils.types import Sample  # noqa: E402
from tests.test_agent._fakes import FakeSGLangServer, ScriptedTokenizer  # noqa: E402

# ---- scripted SLM turns ----
# turn0: explore (no offload)
# turn1: offload N=3
# turn2: continue after tool
# turn3: final answer
TURN0_IDS = (8001, 8002, 8003, 8004)
TURN1_IDS = (8101, 8102, 8103, 8104, 8105)
TURN2_IDS = (8201, 8202, 8203)
TURN3_IDS = (8301, 8302)

TURN0_TEXT = "I'll inspect the repo layout and locate from_preliz."
TURN1_TEXT = (
    f"<think>\nFound the helper but truncated-dist handling is ambiguous; "
    f"requesting help {offload.OFFLOAD_OPEN}3{offload.OFFLOAD_CLOSE}\n"
)
TURN2_TEXT = "Applying the suggested patch to distributions.py."
TURN3_TEXT = "Done. Added Truncated* unwrap and a regression test."

GLM_CONTENT = (
    "Next: open preliz/distributions.py around from_preliz, "
    "strip a Truncated prefix before getattr(pymc, name), then re-run the repro."
)
GLM_THINK = "Continue from part_think; do not re-derive the file search."


class _MockGLM:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.url = ""
        self._runner = None

    async def __aenter__(self) -> _MockGLM:
        app = web.Application()

        async def chat(request: web.Request) -> web.Response:
            body = await request.json()
            self.requests.append(body)
            return web.json_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": GLM_CONTENT,
                                "reasoning_content": GLM_THINK,
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 240 + 40 * len(self.requests),
                        "completion_tokens": 48,
                    },
                }
            )

        app.router.add_post("/v1/chat/completions", chat)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/v1"
        return self

    async def __aexit__(self, *exc) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


def _agent_system() -> str:
    return (
        "You are Claude Code, an agentic coding assistant.\n"
        "Working directory: /workspace/preliz\n"
        f"{offload.OFFLOAD_SYSTEM_PROMPT_APPEND}"
    )


def _blocks_to_parts(data: dict) -> tuple[str, str]:
    texts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    thinks = [b.get("thinking", "") for b in data.get("content", []) if b.get("type") == "thinking"]
    return "\n".join(texts), "".join(thinks)


def _expanding_prompt(seed: list[int], pad: int) -> list[int]:
    """Grow prompt ids across turns so offload expansion FORKs (not REALIGN-wipe)."""
    return list(seed) + list(range(1000 + pad, 1000 + pad + 80 + 20 * pad))


def _sample_to_dict(s: Sample) -> dict:
    return {
        "index": s.index,
        "group_index": s.group_index,
        "rollout_id": s.rollout_id,
        "status": s.status.value if hasattr(s.status, "value") else str(s.status),
        "reward": s.reward,
        "response": s.response,
        "response_length": s.response_length,
        "loss_mask_sum": int(sum(s.loss_mask or [])),
        "loss_mask_len": len(s.loss_mask or []),
        "tokens_len": len(s.tokens or []),
        "prompt_preview": (s.prompt if isinstance(s.prompt, str) else json.dumps(s.prompt, ensure_ascii=False))[
            :200
        ],
    }


async def _run(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_rounds: list[dict] = []
    history: list[dict] = []
    stats: dict = {}
    reward = 0.0
    samples: list[Sample] = []

    async with _MockGLM() as glm:
        os.environ["DASHSCOPE_BASE_URL"] = glm.url
        os.environ["DASHSCOPE_API_KEY"] = "sim-key"
        os.environ["DASHSCOPE_MODEL"] = "glm-sim"

        prompts = [
            _expanding_prompt([10, 11, 12], 0),
            _expanding_prompt([10, 11, 12], 1),
            _expanding_prompt([10, 11, 12], 2),
            _expanding_prompt([10, 11, 12], 3),
        ]
        turns = [
            [(0.0, tid) for tid in TURN0_IDS],
            [(0.0, tid) for tid in TURN1_IDS],
            [(0.0, tid) for tid in TURN2_IDS],
            [(0.0, tid) for tid in TURN3_IDS],
        ]
        tok = ScriptedTokenizer(
            prompts=prompts,
            outputs={
                TURN0_IDS: TURN0_TEXT,
                TURN1_IDS: TURN1_TEXT,
                TURN2_IDS: TURN2_TEXT,
                TURN3_IDS: TURN3_TEXT,
            },
        )

        async with FakeSGLangServer(turns) as sglang:
            adapter = CodingAnthropicAdapter(tokenizer=tok, sglang_url=sglang.url)
            sid = "sim-offload-multiturn"
            adapter.open_session(sid, sampling_defaults={"max_new_tokens": 128})

            server = TestServer(adapter.app)
            async with server:
                client = TestClient(server)
                await client.start_server()
                headers = {
                    "Authorization": f"Bearer {sid}",
                    "Content-Type": "application/json",
                }
                system = _agent_system()

                async def _agent_round(user_content: str, *, expect_offload: bool) -> dict:
                    n_glm_before = len(glm.requests)
                    messages = list(history) + [{"role": "user", "content": user_content}]
                    body = {
                        "model": "slime",
                        "max_tokens": 128,
                        "system": system,
                        "messages": messages,
                    }
                    resp = await client.post("/v1/messages", json=body, headers=headers)
                    raw = await resp.text()
                    assert resp.status == 200, raw
                    data = json.loads(raw)
                    text, think = _blocks_to_parts(data)
                    did_offload = len(glm.requests) > n_glm_before
                    assert did_offload == expect_offload, (
                        f"offload mismatch: got={did_offload} expect={expect_offload}"
                    )
                    glm_payload = glm.requests[-1] if did_offload else None
                    if did_offload:
                        # CC keeps the offload span; GLM request must strip it.
                        assert offload.OFFLOAD_OPEN in (text + think)
                        assert all(
                            offload.OFFLOAD_OPEN not in m.get("content", "")
                            for m in glm_payload["messages"]
                        )
                        assert offload.CODING_HANDOFF_PROMPT in glm_payload["messages"][0]["content"]
                        assert (
                            offload.OFFLOAD_SYSTEM_PROMPT_APPEND
                            not in glm_payload["messages"][0]["content"]
                        )

                    # Update agent-visible history with composed assistant turn.
                    asst_content: list[dict] = []
                    if think:
                        asst_content.append({"type": "thinking", "thinking": think})
                    asst_content.append({"type": "text", "text": text})
                    history.append({"role": "user", "content": user_content})
                    history.append({"role": "assistant", "content": asst_content})

                    record = {
                        "round": len(traj_rounds),
                        "user": user_content,
                        "assistant_text": text,
                        "assistant_think": think,
                        "offload_triggered": did_offload,
                        "glm_request": (
                            {
                                "model": glm_payload.get("model"),
                                "max_tokens": glm_payload.get("max_tokens"),
                                "reasoning_effort": glm_payload.get("reasoning_effort"),
                                "chat_template_kwargs": glm_payload.get("chat_template_kwargs"),
                                "messages": glm_payload.get("messages"),
                            }
                            if glm_payload
                            else None
                        ),
                        "anthropic_response_content": data.get("content"),
                    }
                    traj_rounds.append(record)
                    return record

                # Round 0: explore
                await _agent_round(
                    "Bug: from_preliz fails on truncated distributions. Investigate.",
                    expect_offload=False,
                )
                # Round 1: SLM offloads
                await _agent_round(
                    "[tool:Bash] rg -n 'def from_preliz' -g '*.py'\n"
                    "preliz/distributions.py:120:def from_preliz(dist):",
                    expect_offload=True,
                )
                # Round 2: continue with tool result after composed reply
                await _agent_round(
                    "[tool:Bash] sed -n '110,140p' preliz/distributions.py\n"
                    "... name = dist.__class__.__name__ ...",
                    expect_offload=False,
                )
                # Round 3: finish
                await _agent_round(
                    "Tests passed for TruncatedNormal. Summarize the fix.",
                    expect_offload=False,
                )

                stats = dict(adapter.store[sid].offload_stats)
                reward = offload.cost_aware_reward(1.0, stats, usage=None)
                samples = await adapter.finish_session(
                    sid,
                    base_sample=Sample(
                        index=0,
                        prompt="sim multiturn offload",
                        label="sim_offload_multiturn",
                    ),
                    reward=reward,
                )
                await client.close()

    # ---- persist ----
    traj_path = out_dir / "trajectory.json"
    traj_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sid": "sim-offload-multiturn",
                "agent_system_for_slm": _agent_system(),
                "contract": {
                    "slm_system": "agent_system + OFFLOAD_SYSTEM_PROMPT_APPEND",
                    "glm_system": "agent_system + CODING_HANDOFF_PROMPT (append stripped)",
                },
                "rounds": traj_rounds,
                "final_history": history,
                "offload_stats": stats,
                "train_reward": reward,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pt_path = out_dir / "samples.pt"
    torch.save(
        {
            "rollout_id": 0,
            "samples": samples,
            "offload_stats": stats,
            "train_reward": reward,
        },
        pt_path,
    )

    summary = {
        "ok": True,
        "out_dir": str(out_dir),
        "n_rounds": len(traj_rounds),
        "offload_rounds": [r["round"] for r in traj_rounds if r["offload_triggered"]],
        "offload_stats": stats,
        "train_reward": reward,
        "n_sample_segments": len(samples),
        "loss_mask_sums": [int(sum(s.loss_mask or [])) for s in samples],
        "loss_mask_total": int(sum(sum(s.loss_mask or []) for s in samples)),
        "small_output_tokens": int(stats.get("small_output_tokens", 0)),
        "segments": [_sample_to_dict(s) for s in samples],
        "files": {
            "trajectory": str(traj_path),
            "samples_pt": str(pt_path),
        },
    }
    assert summary["loss_mask_total"] == summary["small_output_tokens"], summary
    assert stats.get("offload_count") == 1, stats

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["files"]["summary"] = str(summary_path)
    return summary


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = _REPO / "examples" / "coding_agent_rl" / "data" / f"offload_traj_sim_{stamp}"
    out_dir = Path(os.environ.get("OUT_DIR", str(default_out)))
    summary = asyncio.run(_run(out_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
