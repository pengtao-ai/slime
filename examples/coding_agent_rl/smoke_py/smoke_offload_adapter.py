#!/usr/bin/env python3
"""CPU smoke for coding-agent mid-turn offload (no GPU / no Docker / no Ray).

Validates the intended contract with a tiny multi-turn session:

1. Agent round 1 -> SLM emits ``<|llm_offload|>7<|/llm_offload|>``
2. Adapter calls GLM with (agent system + handoff) and chat history
3. Composed SLM+GLM reply is returned to the agent
4. Agent round 2 carries that history; SLM answers without offload
5. Session token stats feed ``cost_aware_reward``

Run::

    python examples/coding_agent_rl/smoke_offload_adapter.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

os.environ["SLIME_AGENT_OFFLOAD"] = "1"
os.environ["OFFLOAD_EFFICIENCY_LAMBDA"] = "0.6"
# Force (not setdefault): parent shells may export a larger OFFLOAD_MAX_TOKENS.
os.environ["OFFLOAD_MAX_TOKENS"] = "100"

from examples.coding_agent_rl import offload  # noqa: E402
from examples.coding_agent_rl.generate import CodingAnthropicAdapter  # noqa: E402
from slime.utils.types import Sample  # noqa: E402
from tests.test_agent._fakes import FakeSGLangServer, ScriptedTokenizer  # noqa: E402


OFFLOAD_TEXT = (
    f"<think>\npartial plan {offload.OFFLOAD_OPEN}7{offload.OFFLOAD_CLOSE}\n"
)
TURN1_IDS = (9001, 9002, 9003)
TURN2_IDS = (9101, 9102)


class _MockGLM:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.url = ""
        self._runner = None
        self._site = None

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
                                "content": "GLM next step: edit foo.py",
                                "reasoning_content": "remote-why",
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 111, "completion_tokens": 22},
                }
            )

        app.router.add_post("/v1/chat/completions", chat)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()
        port = self._site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/v1"
        return self

    async def __aexit__(self, *exc) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


async def _run() -> None:
    async with _MockGLM() as glm:
        os.environ["DASHSCOPE_BASE_URL"] = glm.url
        os.environ["DASHSCOPE_API_KEY"] = "smoke-key"
        os.environ["DASHSCOPE_MODEL"] = "glm-smoke"

        turns = [
            [(0.0, tid) for tid in TURN1_IDS],
            [(0.0, tid) for tid in TURN2_IDS],
        ]
        # Turn-2 prompt shares the turn-1 prompt prefix, then expands a lot
        # (simulating chat-template re-render of the longer SLM+GLM assistant echo).
        # That expansion must be large enough to FORK instead of REALIGN-wipe.
        turn2_prompt = [10, 11, 12] + list(range(100, 180))
        tok = ScriptedTokenizer(
            prompts=[[10, 11, 12], turn2_prompt],
            outputs={
                TURN1_IDS: OFFLOAD_TEXT,
                TURN2_IDS: "ok continue",
            },
        )

        async with FakeSGLangServer(turns) as sglang:
            adapter = CodingAnthropicAdapter(tokenizer=tok, sglang_url=sglang.url)
            sid = "smoke-offload-1"
            adapter.open_session(sid, sampling_defaults={"max_new_tokens": 64})

            server = TestServer(adapter.app)
            async with server:
                client = TestClient(server)
                await client.start_server()
                headers = {"Authorization": f"Bearer {sid}", "Content-Type": "application/json"}

                # ---- round 1: SLM offloads ----
                body1 = {
                    "model": "slime",
                    "max_tokens": 64,
                    # Mimic Claude Code: system ends with gitStatus; adapter appends offload after it.
                    "system": (
                        "You are Claude Code agent.\n"
                        "gitStatus: This is the git status at the start of the conversation.\n"
                        "Current branch: main"
                    ),
                    "messages": [{"role": "user", "content": "Fix the bug in foo.py"}],
                }
                resp1 = await client.post("/v1/messages", json=body1, headers=headers)
                assert resp1.status == 200, await resp1.text()
                # Adapter inject must place offload text after gitStatus (not via CC append).
                assert body1["system"].index("gitStatus:") < body1["system"].index(
                    offload.OFFLOAD_SYSTEM_PROMPT_APPEND
                ), body1["system"][-400:]
                data1 = await resp1.json()
                blocks1 = data1["content"]
                texts1 = [b.get("text", "") for b in blocks1 if b.get("type") == "text"]
                thinks1 = [b.get("thinking", "") for b in blocks1 if b.get("type") == "thinking"]
                joined_text = "\n".join(texts1)
                joined_think = "".join(thinks1)
                assert "GLM next step" in joined_text, data1
                # Protocol: span was in <think>; CC reply keeps it (may sit in text
                # when smoke adapter has no reasoning_parser).
                assert f"{offload.OFFLOAD_OPEN}7{offload.OFFLOAD_CLOSE}" in (
                    joined_text + joined_think
                )
                assert "remote-why" in joined_think
                assert "partial plan" in (joined_text + joined_think)
                assert len(glm.requests) == 1
                glm_msgs = glm.requests[0]["messages"]
                assert glm_msgs[0]["role"] == "system"
                assert "You are Claude Code agent." in glm_msgs[0]["content"]
                assert offload.CODING_HANDOFF_PROMPT in glm_msgs[0]["content"]
                # SLM-only append must not leak into the GLM system prompt.
                assert offload.OFFLOAD_SYSTEM_PROMPT_APPEND not in glm_msgs[0]["content"]
                assert all(offload.OFFLOAD_OPEN not in (m.get("content") or "") for m in glm_msgs)
                assert any("<part_think>" in (m.get("content") or "") for m in glm_msgs)
                # budget = OFFLOAD_MAX_TOKENS - len(SLM output ids)
                assert glm.requests[0]["max_tokens"] == 100 - len(TURN1_IDS)
                assert glm.requests[0].get("reasoning_effort") == "max"  # N=7

                # ---- round 2: history includes composed assistant ----
                body2 = {
                    "model": "slime",
                    "max_tokens": 64,
                    "system": f"You are Claude Code agent.\n{offload.OFFLOAD_SYSTEM_PROMPT_APPEND}",
                    "messages": [
                        {"role": "user", "content": "Fix the bug in foo.py"},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "thinking", "thinking": joined_think},
                                {"type": "text", "text": joined_text},
                            ],
                        },
                        {"role": "user", "content": "Continue."},
                    ],
                }
                resp2 = await client.post("/v1/messages", json=body2, headers=headers)
                assert resp2.status == 200, await resp2.text()
                data2 = await resp2.json()
                text2 = "".join(b.get("text", "") for b in data2["content"] if b.get("type") == "text")
                assert "ok continue" in text2
                assert len(glm.requests) == 1  # no second offload

                stats = dict(adapter.store[sid].offload_stats)
                assert stats["offload_count"] == 1
                assert stats["last_offload_n"] == 7
                assert stats["last_reasoning_effort"] == "max"
                assert stats["small_prompt_tokens"] > 0
                assert stats["small_output_tokens"] == len(TURN1_IDS) + len(TURN2_IDS)
                assert stats["glm_input_tokens"] == 111
                assert stats["glm_output_tokens"] == 22

                reward = offload.cost_aware_reward(1.0, stats, usage=None)
                assert reward < 1.0, reward
                assert int(stats.get("offload_outside_think_count", 0)) == 0
                # Outside-think span: no GLM, but solved reward takes format penalty.
                bad = {
                    "offload_outside_think_count": 1,
                    "small_prompt_tokens": 0,
                    "small_output_tokens": 0,
                    "glm_input_tokens": 0,
                    "glm_output_tokens": 0,
                }
                assert offload.cost_aware_reward(1.0, bad, usage=None, format_penalty=0.25) == 0.75
                # Failures must not get a length/cost gradient (EOS-collapse fix).
                assert offload.cost_aware_reward(0.0, stats, usage=None) == 0.0
                assert (
                    offload.cost_aware_reward(
                        0.0,
                        {
                            "small_prompt_tokens": 10_000_000,
                            "small_output_tokens": 50_000,
                            "glm_input_tokens": 0,
                            "glm_output_tokens": 0,
                        },
                        usage=None,
                    )
                    == 0.0
                )

                samples = await adapter.finish_session(
                    sid, base_sample=Sample(index=0, prompt="x"), reward=reward
                )
                assert samples, "expected trainable samples"
                # SLM tokens stay loss_mask=1 (offload expansion FORKs instead of REALIGN-wipe).
                loss_sum = sum(sum(s.loss_mask or []) for s in samples)
                assert loss_sum == stats["small_output_tokens"], (
                    f"loss_mask sum {loss_sum} != slm_output {stats['small_output_tokens']}; "
                    f"segments={[sum(s.loss_mask or []) for s in samples]}"
                )

                print(
                    json.dumps(
                        {
                            "ok": True,
                            "offload_stats": stats,
                            "train_reward": reward,
                            "glm_max_tokens": glm.requests[0]["max_tokens"],
                            "segments": len(samples),
                            "loss_mask_sums": [sum(s.loss_mask or []) for s in samples],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                await client.close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
