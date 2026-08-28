"""Unit tests for OpenAI-protocol GLM offload message building."""

from __future__ import annotations

import json

from examples.coding_agent_rl import offload
from slime.agent.adapters.common import Reply

NUM_GPUS = 0


def test_build_offload_messages_openai_tool_protocol():
    translated = [
        {
            "role": "system",
            "content": f"You are Claude Code.\n{offload.OFFLOAD_SYSTEM_PROMPT_APPEND}",
        },
        {"role": "user", "content": "Fix MetadataEndpoint."},
        {
            "role": "assistant",
            "content": "I'll start by exploring the codebase.",
            "reasoning_content": "Let me start by exploring the codebase.",
            "tool_calls": [
                {
                    "id": "chatcmpl-tool-8a09ddebc7f20d0d",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {"command": 'find /testbed -type f -name "*.py" | grep -i metadata'},
                    },
                },
                {
                    "id": "chatcmpl-tool-8846002a37e76d6b",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {"command": 'find /testbed -type f -name "*.py" | head -50'},
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "chatcmpl-tool-8a09ddebc7f20d0d",
            "content": "<returncode>0</returncode>\n<output>\n/testbed/.../metadata.py\n</output>",
        },
        {
            "role": "tool",
            "tool_call_id": "chatcmpl-tool-8846002a37e76d6b",
            "content": "<returncode>0</returncode>\n<output>\n/testbed/tests/__init__.py\n</output>",
        },
    ]
    raw = f"<think>\nNeed a careful read {offload.OFFLOAD_OPEN}5{offload.OFFLOAD_CLOSE}"

    messages = offload.build_offload_messages(translated, raw)

    assert messages[0]["role"] == "system"
    assert offload.CODING_HANDOFF_PROMPT in messages[0]["content"]
    assert offload.OFFLOAD_SYSTEM_PROMPT_APPEND not in messages[0]["content"]

    asst = messages[2]
    assert asst["role"] == "assistant"
    assert asst["content"].startswith("<part_think>")
    assert "</part_think>" in asst["content"]
    inner = asst["content"].split("<part_think>", 1)[1].split("</part_think>", 1)[0]
    assert "Let me start by exploring" in inner
    assert "I'll start by exploring" in inner
    assert "<think>" not in asst["content"]
    assert "[tool_calls]" not in (asst["content"] or "")
    assert len(asst["tool_calls"]) == 2
    assert asst["tool_calls"][0]["id"] == "chatcmpl-tool-8a09ddebc7f20d0d"
    assert asst["tool_calls"][0]["type"] == "function"
    assert asst["tool_calls"][0]["function"]["name"] == "bash"
    args0 = asst["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args0, str)
    assert json.loads(args0)["command"].startswith("find /testbed")

    tool0, tool1 = messages[3], messages[4]
    assert tool0["role"] == "tool"
    assert tool0["tool_call_id"] == "chatcmpl-tool-8a09ddebc7f20d0d"
    assert "metadata.py" in tool0["content"]
    assert tool1["role"] == "tool"
    assert tool1["tool_call_id"] == "chatcmpl-tool-8846002a37e76d6b"

    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"].startswith("<part_think>")
    assert offload.OFFLOAD_OPEN not in messages[-1]["content"]


def test_build_offload_messages_pairs_tools_without_ids_in_order():
    translated = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {"type": "function", "function": {"name": "Bash", "arguments": {"command": "ls"}}},
                {"type": "function", "function": {"name": "Bash", "arguments": {"command": "pwd"}}},
            ],
        },
        {"role": "tool", "content": "a"},
        {"role": "tool", "content": "b"},
    ]
    messages = offload.build_offload_messages(
        translated, f"<think>\nx {offload.OFFLOAD_OPEN}2{offload.OFFLOAD_CLOSE}"
    )
    asst = next(m for m in messages if m.get("tool_calls"))
    ids = [tc["id"] for tc in asst["tool_calls"]]
    tools = [m for m in messages if m["role"] == "tool"]
    assert [t["tool_call_id"] for t in tools] == ids


def test_build_offload_messages_keeps_glm_content_outside_part_think():
    translated = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix foo"},
        {
            "role": "assistant",
            "content": "edit foo.py",
            "reasoning_content": f"<think>\npartial plan {offload.OFFLOAD_OPEN}5{offload.OFFLOAD_CLOSE}\nremote-why",
        },
        {"role": "user", "content": "continue"},
    ]
    messages = offload.build_offload_messages(
        translated, f"<think>\nnow {offload.OFFLOAD_OPEN}2{offload.OFFLOAD_CLOSE}"
    )
    asst = next(m for m in messages if m["role"] == "assistant" and "edit foo.py" in (m.get("content") or ""))
    assert asst["content"].startswith("<part_think>")
    inner, rest = asst["content"].split("</part_think>", 1)
    assert "partial plan" in inner
    assert "remote-why" in inner
    assert offload.OFFLOAD_OPEN not in asst["content"]
    assert "edit foo.py" in rest
    assert "edit foo.py" not in inner


def test_normalize_openai_tools_and_request_body_includes_tools(monkeypatch):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "run shell",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
    ]
    normalized = offload._normalize_openai_tools(tools)
    assert normalized and normalized[0]["function"]["name"] == "Bash"

    captured: dict = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "ok", "reasoning_content": "why"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["body"] = json
        return _Resp()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setattr(offload.requests, "post", _fake_post)

    content, think, usage, tcs = offload._call_remote_chat_sync(
        [{"role": "user", "content": "hi"}],
        max_tokens=16,
        enable_thinking=False,
        reasoning_effort=None,
        tools=tools,
    )
    assert content == "ok"
    assert think == "why"
    assert usage == {"prompt_tokens": 1, "completion_tokens": 1}
    assert tcs == []
    assert captured["body"]["tools"][0]["function"]["name"] == "Bash"
    assert "messages" in captured["body"]


def test_amend_reply_with_glm_tool_calls():
    reply = Reply(
        manager_message={"role": "assistant", "content": "partial"},
        finish_reason="stop",
        wire=([{"type": "text", "text": "partial"}], "end_turn"),
    )
    glm_tcs = [
        {
            "id": "chatcmpl-tool-1",
            "type": "function",
            "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
        }
    ]
    out = offload.amend_reply_with_offload(
        reply,
        raw_output=f"<think>\nx {offload.OFFLOAD_OPEN}2{offload.OFFLOAD_CLOSE}",
        glm_content="next",
        glm_think="why",
        glm_tool_calls=glm_tcs,
    )
    assert out.finish_reason == "tool_calls"
    blocks, stop = out.wire
    assert stop == "tool_use"
    assert any(b.get("type") == "tool_use" and b.get("name") == "Bash" for b in blocks)
    assert out.manager_message["tool_calls"][0]["function"]["name"] == "Bash"
