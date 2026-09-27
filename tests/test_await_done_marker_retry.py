"""done-marker poll retries once after host cmd timeout."""

from __future__ import annotations

import asyncio

import pytest

from slime.agent import sandbox as sandbox_mod


class _FakeSb:
    def __init__(self, sequence: list[object]) -> None:
        self.sequence = list(sequence)
        self.calls: list[int] = []

    async def exec(self, cmd, *, user="root", env=None, timeout=120, check=False, idempotent=True):
        del cmd, user, env, check, idempotent
        self.calls.append(int(timeout))
        item = self.sequence.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    async def _noop(_delay: float) -> None:
        return None

    monkeypatch.setattr(sandbox_mod.asyncio, "sleep", _noop)


def test_await_done_marker_retries_timeout_once_with_20s():
    sb = _FakeSb(
        [
            RuntimeError("host cmd timed out after 15s: ['docker', 'exec']..."),
            (0, "0\n", ""),
        ]
    )

    code = asyncio.run(
        sandbox_mod._await_done_marker(sb, "/tmp/.run.done", user="agent", time_budget_sec=30)
    )
    assert code == 0
    assert sb.calls == [15, 20]


def test_await_done_marker_reraises_non_timeout():
    sb = _FakeSb([RuntimeError("docker exec failed (exit=1): boom")])

    with pytest.raises(RuntimeError, match="docker exec failed"):
        asyncio.run(
            sandbox_mod._await_done_marker(sb, "/tmp/.run.done", user="agent", time_budget_sec=30)
        )
