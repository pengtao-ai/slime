"""Unit tests for sandbox pool + RolloutDataSource.reserve_samples."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# CPU CI may lack transformers; RolloutDataSource import chain pulls it.
if "transformers" not in sys.modules:
    _tf_stub = types.ModuleType("transformers")
    for _name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(_tf_stub, _name, type(_name, (), {}))
    sys.modules["transformers"] = _tf_stub

from examples.coding_agent_rl import sandbox_pool as sp  # noqa: E402
from slime.rollout.data_source import RolloutDataSource  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0


def _args(**kwargs):
    base = dict(
        rollout_global_dataset=False,
        prompt_data=None,
        n_samples_per_prompt=2,
        rollout_shuffle=False,
        save="/tmp/unused",
        load=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _reset_pool(monkeypatch):
    monkeypatch.setenv("SANDBOX_POOL", "1")
    monkeypatch.delenv("SANDBOX_POOL_MAX", raising=False)
    sp.reset_pool_for_tests()
    sp.set_boot_sem(asyncio.Semaphore(8))
    yield
    sp.reset_pool_for_tests()


def test_reserve_then_get_samples_matches_sequential():
    ds_a = RolloutDataSource(_args())
    ds_b = RolloutDataSource(_args())

    first = ds_a.get_samples(2)
    reserved = ds_a.reserve_samples(2)
    second = ds_a.get_samples(2)

    seq1 = ds_b.get_samples(2)
    seq2 = ds_b.get_samples(2)

    assert len(first) == len(seq1) == 2
    assert len(reserved) == len(second) == len(seq2) == 2
    assert [s.index for g in reserved for s in g] == [s.index for g in seq2 for s in g]
    assert [s.index for g in second for s in g] == [s.index for g in seq2 for s in g]
    # Second reserve while parked returns the same object until taken.
    ds_c = RolloutDataSource(_args())
    ds_c.get_samples(1)
    r1 = ds_c.reserve_samples(1)
    r2 = ds_c.reserve_samples(1)
    assert r1 is r2


def test_get_samples_consumes_reserved():
    ds = RolloutDataSource(_args())
    ds.reserve_samples(1)
    assert ds._reserved is not None
    out = ds.get_samples(1)
    assert ds._reserved is None
    assert len(out) == 1
    assert len(out[0]) == 2  # n_samples_per_prompt


class _FakeSandbox:
    def __init__(self, image: str, name: str):
        self.image = image
        self.sandbox_id = name
        self._container = name
        self.remove = True
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *args):
        self.exited = True


class _FakeHarness:
    name = "claude_code"
    install_calls: list[str] = []

    def __init__(self):
        pass

    async def install_cli(self, sb):
        _FakeHarness.install_calls.append(sb.sandbox_id)
        await asyncio.sleep(0)


def _sample(image: str, agent: str = "claude_code") -> Sample:
    return Sample(
        prompt="fix",
        label="id",
        metadata={"image": image, "workdir": "/repo", "instance_id": "i", "agent": agent},
    )


def test_pool_acquire_hit_and_release(monkeypatch):
    _FakeHarness.install_calls = []
    created: list[_FakeSandbox] = []

    def fake_make(image, *, timeout=None):
        sb = _FakeSandbox(image, f"c-{len(created)}")
        created.append(sb)
        return sb

    monkeypatch.setattr(sp, "make_sandbox", fake_make)
    monkeypatch.setattr(sp, "resolve_agent", lambda name=None: SimpleNamespace(name="claude_code", harness_cls=_FakeHarness))

    async def run_case():
        pool = sp.get_pool()
        pool.configure(max_size=4)
        groups = [[_sample("img-a"), _sample("img-a")]]
        await pool.prefetch(groups)
        assert len(pool._ready[("img-a", "claude_code")]) == 2
        assert len(_FakeHarness.install_calls) == 2

        hit = await pool.acquire("img-a", "claude_code", timeout=1.0)
        assert hit is not None
        assert pool.hits == 1

        # No ready sandbox and pool full → wait times out (no capacity for on-demand).
        pool.configure(max_size=2)  # size still 2 (one ready + one acquired)
        timed_out = await pool.acquire("img-missing", "claude_code", timeout=0.05)
        assert timed_out is None
        assert pool.wait_timeouts >= 1

        await pool.release(hit)
        assert hit.exited

    asyncio.run(run_case())


def test_acquire_waits_for_in_flight_warm(monkeypatch):
    """acquire blocks until an in-flight warm finishes (no immediate cold miss)."""
    started = asyncio.Event()
    release_warm = asyncio.Event()

    class SlowSandbox(_FakeSandbox):
        async def __aenter__(self):
            started.set()
            await release_warm.wait()
            self.entered = True
            return self

    n = {"i": 0}

    def fake_make(image, *, timeout=None):
        sb = SlowSandbox(image, f"w-{n['i']}")
        n["i"] += 1
        return sb

    monkeypatch.setattr(sp, "make_sandbox", fake_make)
    monkeypatch.setattr(
        sp,
        "resolve_agent",
        lambda name=None: SimpleNamespace(name="claude_code", harness_cls=_FakeHarness),
    )

    async def run_case():
        pool = sp.get_pool()
        pool.configure(max_size=2)
        prefetch_task = asyncio.create_task(pool.prefetch([[_sample("img-wait")]]))
        await started.wait()

        acquire_task = asyncio.create_task(pool.acquire("img-wait", "claude_code", timeout=2.0))
        await asyncio.sleep(0.05)
        assert not acquire_task.done(), "acquire should wait for warm"
        release_warm.set()
        hit = await acquire_task
        await prefetch_task
        assert hit is not None
        assert pool.hits == 1
        await pool.release(hit)

    asyncio.run(run_case())


def test_acquire_ondemand_warm_when_empty(monkeypatch):
    """With nothing prefetched, acquire starts a warm and waits for it."""
    created: list[_FakeSandbox] = []

    def fake_make(image, *, timeout=None):
        sb = _FakeSandbox(image, f"od-{len(created)}")
        created.append(sb)
        return sb

    monkeypatch.setattr(sp, "make_sandbox", fake_make)
    monkeypatch.setattr(
        sp,
        "resolve_agent",
        lambda name=None: SimpleNamespace(name="claude_code", harness_cls=_FakeHarness),
    )

    async def run_case():
        pool = sp.get_pool()
        pool.configure(max_size=2)
        hit = await pool.acquire("img-od", "claude_code", timeout=2.0)
        assert hit is not None
        assert len(created) == 1
        await pool.release(hit)

    asyncio.run(run_case())


def test_same_image_parallel_after_first_warm(monkeypatch):
    """First warm per image is gated; after success, same-image warms overlap."""
    active = {"img-a": 0, "img-b": 0}
    max_active = {"img-a": 0, "img-b": 0}
    overlap_different = {"seen": False}
    lock = asyncio.Lock()

    class SlowSandbox(_FakeSandbox):
        async def __aenter__(self):
            img = self.image
            async with lock:
                active[img] += 1
                max_active[img] = max(max_active[img], active[img])
                if active["img-a"] > 0 and active["img-b"] > 0:
                    overlap_different["seen"] = True
            await asyncio.sleep(0.05)
            async with lock:
                active[img] -= 1
            self.entered = True
            return self

    class SlowHarness(_FakeHarness):
        async def install_cli(self, sb):
            await asyncio.sleep(0.01)

    n = {"i": 0}

    def fake_make(image, *, timeout=None):
        sb = SlowSandbox(image, f"s-{n['i']}")
        n["i"] += 1
        return sb

    monkeypatch.setattr(sp, "make_sandbox", fake_make)
    monkeypatch.setattr(
        sp,
        "resolve_agent",
        lambda name=None: SimpleNamespace(name="claude_code", harness_cls=SlowHarness),
    )

    async def run_case():
        pool = sp.get_pool()
        pool.configure(max_size=8)
        # 4 of img-a + 2 of img-b: after first img-a succeeds, remaining a's parallel.
        groups = [
            [_sample("img-a"), _sample("img-a"), _sample("img-a"), _sample("img-a")],
            [_sample("img-b"), _sample("img-b")],
        ]
        await pool.prefetch(groups)

        assert max_active["img-a"] >= 2, "same image should parallelize after first warm"
        assert overlap_different["seen"], "different images should warm in parallel"
        assert "img-a" in pool._images_warmed_ok
        assert "img-b" in pool._images_warmed_ok
        assert len(pool._ready[("img-a", "claude_code")]) == 4
        assert len(pool._ready[("img-b", "claude_code")]) == 2

    asyncio.run(run_case())


def test_pool_enabled_defaults_on(monkeypatch):
    monkeypatch.delenv("SANDBOX_POOL", raising=False)
    assert sp.pool_enabled() is True
    monkeypatch.setenv("SANDBOX_POOL", "0")
    assert sp.pool_enabled() is False


def test_docker_sandbox_from_container_skips_run(monkeypatch):
    from slime.agent.sandbox import DockerSandbox

    sb = DockerSandbox.from_container("img", "slime-sb-deadbeef")
    assert sb._existing_name == "slime-sb-deadbeef"
    assert sb.sandbox_id == "slime-sb-deadbeef"

    async def _noop_wait(name: str) -> None:
        return None

    monkeypatch.setattr(sb, "_wait_until_running", _noop_wait)

    async def _enter():
        # Should not call docker run; just return self after ready wait.
        out = await sb.__aenter__()
        assert out is sb
        assert sb._container == "slime-sb-deadbeef"

    asyncio.run(_enter())


def test_docker_sandbox_waits_until_running(monkeypatch):
    from slime.agent.sandbox import DockerSandbox

    sb = DockerSandbox("img:test")
    calls = {"n": 0}

    async def fake_run_host(cmd, *, timeout, check):
        if cmd[:2] == ["docker", "run"]:
            return 0, "", ""
        if cmd[:2] == ["docker", "exec"] and cmd[-1] == "true":
            # Fail twice (Created), then succeed (Running).
            calls["n"] += 1
            if calls["n"] < 3:
                return 1, "", "Error: Container is not running"
            return 0, "", ""
        return 0, "", ""

    monkeypatch.setattr(sb, "_run_host", fake_run_host)

    async def _enter():
        out = await sb.__aenter__()
        assert out is sb
        assert calls["n"] == 3

    asyncio.run(_enter())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
