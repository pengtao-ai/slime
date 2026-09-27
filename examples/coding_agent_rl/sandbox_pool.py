"""Warm sandbox pool for coding-agent rollouts.

Prefetch next-batch containers (``docker run`` + ``install_cli``) so the
following ``boot_agent_sandbox`` can wait for a ready sandbox.

Concurrency rules:
  * same image — first successful warm is gated (serial); after that image has
    been pulled up once, further warms of that image run in parallel
  * different images — parallel from the start
  * global ``SWE_BOOT_CONCURRENCY`` semaphore still caps total boots

Env:
  * ``SANDBOX_POOL`` — default ``1`` (on); ``0``/``false``/``off`` disables
  * ``SANDBOX_POOL_MAX`` — max warm+in-use containers (default set by rollout
    hook to ``2 * rollout_batch_size * n_samples_per_prompt`` so current and
    next-step warms can coexist)
  * ``SANDBOX_POOL_ACQUIRE_TIMEOUT_SEC`` — how long ``acquire`` waits for a
    warm sandbox before giving up (default ``600``); ``0`` waits forever
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from collections.abc import Iterable

from slime.agent.harness.common import BaseHarness
from slime.agent.sandbox import Sandbox, make_sandbox
from slime.utils.types import Sample

from . import swe
from .agents_registry import resolve_agent

logger = logging.getLogger(__name__)

_PoolKey = tuple[str, str]  # (image, harness_name)

_boot_sem: asyncio.Semaphore | None = None
_POOL: "SandboxPool | None" = None


def pool_enabled() -> bool:
    raw = (os.environ.get("SANDBOX_POOL") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def set_boot_sem(sem: asyncio.Semaphore) -> None:
    """Share the coding-agent boot semaphore with the pool."""
    global _boot_sem
    _boot_sem = sem


def get_boot_sem() -> asyncio.Semaphore:
    global _boot_sem
    if _boot_sem is None:
        n = int(os.environ.get("SWE_BOOT_CONCURRENCY", "16"))
        _boot_sem = asyncio.Semaphore(max(1, n))
    return _boot_sem


def _default_max_size() -> int:
    raw = (os.environ.get("SANDBOX_POOL_MAX") or "").strip()
    if raw:
        return max(1, int(raw))
    # Fallback when configured before rollout args are known: one-rollout-sized.
    # rollout_with_pool overrides to 2× via configure(max_size=...).
    return 32


def _acquire_timeout_sec() -> float | None:
    raw = (os.environ.get("SANDBOX_POOL_ACQUIRE_TIMEOUT_SEC") or "600").strip()
    try:
        sec = float(raw)
    except ValueError:
        return 600.0
    if sec <= 0:
        return None  # wait forever
    return sec


_ACQUIRE_TIMEOUT_DEFAULT = object()


class SandboxPool:
    """In-process warm pool keyed by ``(image, harness_name)``."""

    def __init__(self, *, max_size: int | None = None) -> None:
        self.max_size = int(max_size if max_size is not None else _default_max_size())
        self._ready: dict[_PoolKey, list[Sandbox]] = defaultdict(list)
        self._image_locks: dict[str, asyncio.Lock] = {}
        # Images that have completed at least one successful warm (layer/cache hot).
        self._images_warmed_ok: set[str] = set()
        self._lock = asyncio.Lock()
        self._cv = asyncio.Condition(self._lock)
        self._warm_tasks: set[asyncio.Task] = set()
        self._inflight: dict[_PoolKey, int] = defaultdict(int)
        self._size = 0
        self.hits = 0
        self.misses = 0
        self.wait_timeouts = 0

    def configure(self, *, max_size: int | None = None) -> None:
        if max_size is not None:
            self.max_size = max(1, int(max_size))

    def _lock_for_image(self, image: str) -> asyncio.Lock:
        lock = self._image_locks.get(image)
        if lock is None:
            lock = asyncio.Lock()
            self._image_locks[image] = lock
        return lock

    def _slots_from_groups(
        self,
        groups: Iterable[list[Sample]],
        *,
        protocol: str,
    ) -> list[tuple[str, str]]:
        """Flatten groups into ``(image, harness_name)`` slots (one per sample)."""
        slots: list[tuple[str, str]] = []
        for group in groups:
            for sample in group:
                raw_protocol = (sample.metadata or {}).get("protocol") or protocol
                try:
                    md = swe.get_metadata(sample, raw_protocol)
                except Exception as e:
                    logger.warning("[sandbox_pool] skip sample metadata: %s", e)
                    continue
                image = md.get("image")
                if not image:
                    continue
                harness_name = resolve_agent(md.get("agent")).name
                slots.append((str(image), harness_name))
        return slots

    def _track_task(self, task: asyncio.Task) -> None:
        self._warm_tasks.add(task)
        task.add_done_callback(self._warm_tasks.discard)

    async def prefetch(
        self,
        groups: list[list[Sample]],
        *,
        protocol: str = swe.PROTOCOL_SCALESWE,
    ) -> None:
        """Warm sandboxes for ``groups``.

        First successful warm per image is gated; after that, same-image warms
        run concurrently (still under ``SWE_BOOT_CONCURRENCY``).
        """
        if not pool_enabled() or not groups:
            return

        slots = self._slots_from_groups(groups, protocol=protocol)
        if not slots:
            return

        want: dict[_PoolKey, int] = defaultdict(int)
        for image, harness_name in slots:
            want[(image, harness_name)] += 1

        async with self._lock:
            free = max(0, self.max_size - self._size)
            if free <= 0:
                logger.info(
                    "[sandbox_pool] prefetch skipped: pool full size=%d max=%d",
                    self._size,
                    self.max_size,
                )
                return
            to_schedule: list[_PoolKey] = []
            scheduled = 0
            for key, count in want.items():
                have = len(self._ready.get(key, [])) + self._inflight.get(key, 0)
                missing = max(0, count - have)
                for _ in range(missing):
                    if scheduled >= free:
                        break
                    to_schedule.append(key)
                    scheduled += 1
                    self._size += 1
                    self._inflight[key] += 1
                if scheduled >= free:
                    break

        if not to_schedule:
            return

        tasks = [
            asyncio.create_task(
                self._warm_one_gated(key, reserved=True),
                name=f"sandbox_pool_warm:{key[0][:40]}",
            )
            for key in to_schedule
        ]
        for t in tasks:
            self._track_task(t)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _warm_one_gated(self, key: _PoolKey, *, reserved: bool) -> bool:
        """Gate the first warm per image; allow parallel warms after success."""
        image, _ = key
        if image in self._images_warmed_ok:
            return await self._warm_one(key, reserved=reserved)

        async with self._lock_for_image(image):
            if image not in self._images_warmed_ok:
                return await self._warm_one(key, reserved=reserved)
        # Peer finished the first warm while we waited for the image lock.
        return await self._warm_one(key, reserved=reserved)

    async def _warm_one(self, key: _PoolKey, *, reserved: bool) -> bool:
        """Warm a single sandbox. If ``reserved``, capacity/inflight already booked."""
        image, harness_name = key
        if not reserved:
            async with self._lock:
                if self._size >= self.max_size:
                    return False
                self._size += 1
                self._inflight[key] += 1

        sem = get_boot_sem()
        await sem.acquire()
        sb: Sandbox | None = None
        ok = False
        try:
            harness: BaseHarness = resolve_agent(harness_name).harness_cls()
            cand = make_sandbox(image)
            if hasattr(cand, "remove"):
                cand.remove = False  # type: ignore[attr-defined]
            await cand.__aenter__()
            await harness.install_cli(cand)
            sb = cand
            async with self._cv:
                self._inflight[key] = max(0, self._inflight[key] - 1)
                self._ready[key].append(sb)
                self._images_warmed_ok.add(image)
                self._cv.notify_all()
            logger.info(
                "[sandbox_pool] warmed image=%s harness=%s sandbox_id=%s ready=%d size=%d/%d",
                image,
                harness_name,
                getattr(sb, "sandbox_id", ""),
                len(self._ready[key]),
                self._size,
                self.max_size,
            )
            ok = True
            return True
        except Exception as e:
            logger.warning(
                "[sandbox_pool] warm failed image=%s harness=%s: %s: %s",
                image,
                harness_name,
                type(e).__name__,
                str(e)[:200],
            )
            if sb is not None:
                try:
                    if hasattr(sb, "remove"):
                        sb.remove = True  # type: ignore[attr-defined]
                    await sb.__aexit__(None, None, None)
                except Exception:
                    pass
            async with self._cv:
                self._inflight[key] = max(0, self._inflight[key] - 1)
                self._size = max(0, self._size - 1)
                self._cv.notify_all()
            return False
        finally:
            sem.release()

    def _ensure_warm_task(self, key: _PoolKey) -> None:
        """Start an on-demand warm if none ready and none in flight (caller holds lock)."""
        if self._ready.get(key) or self._inflight.get(key, 0) > 0:
            return
        if self._size >= self.max_size:
            return
        self._size += 1
        self._inflight[key] += 1
        image, _ = key

        async def _run() -> None:
            await self._warm_one_gated(key, reserved=True)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._inflight[key] = max(0, self._inflight[key] - 1)
            self._size = max(0, self._size - 1)
            return
        task = loop.create_task(_run(), name=f"sandbox_pool_ondemand:{image[:48]}")
        self._track_task(task)

    async def acquire(
        self,
        image: str,
        harness_name: str,
        *,
        timeout: float | None | object = _ACQUIRE_TIMEOUT_DEFAULT,
    ) -> Sandbox | None:
        """Wait for a pre-warmed sandbox; start one on demand if needed.

        Returns ``None`` only on timeout (caller may cold-boot). Default timeout
        from ``SANDBOX_POOL_ACQUIRE_TIMEOUT_SEC`` (``0`` = wait forever).
        """
        if timeout is _ACQUIRE_TIMEOUT_DEFAULT:
            timeout = _acquire_timeout_sec()
        assert timeout is None or isinstance(timeout, (int, float))
        key = (image, harness_name)
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else (loop.time() + float(timeout))

        async with self._cv:
            while True:
                q = self._ready.get(key)
                if q:
                    sb = q.pop(0)
                    self.hits += 1
                    logger.info(
                        "[sandbox_pool] acquire HIT image=%s harness=%s sandbox_id=%s "
                        "hits=%d misses=%d timeouts=%d",
                        image,
                        harness_name,
                        getattr(sb, "sandbox_id", ""),
                        self.hits,
                        self.misses,
                        self.wait_timeouts,
                    )
                    return sb

                self._ensure_warm_task(key)

                if deadline is None:
                    await self._cv.wait()
                    continue

                remaining = deadline - loop.time()
                if remaining <= 0:
                    self.misses += 1
                    self.wait_timeouts += 1
                    logger.warning(
                        "[sandbox_pool] acquire TIMEOUT image=%s harness=%s "
                        "hits=%d misses=%d timeouts=%d",
                        image,
                        harness_name,
                        self.hits,
                        self.misses,
                        self.wait_timeouts,
                    )
                    return None
                try:
                    await asyncio.wait_for(self._cv.wait(), timeout=remaining)
                except TimeoutError:
                    self.misses += 1
                    self.wait_timeouts += 1
                    logger.warning(
                        "[sandbox_pool] acquire TIMEOUT image=%s harness=%s "
                        "hits=%d misses=%d timeouts=%d",
                        image,
                        harness_name,
                        self.hits,
                        self.misses,
                        self.wait_timeouts,
                    )
                    return None

    async def release(self, sb: Sandbox) -> None:
        """Destroy a used sandbox (never return dirty containers to the pool)."""
        async with self._cv:
            self._size = max(0, self._size - 1)
            self._cv.notify_all()
        try:
            if hasattr(sb, "remove"):
                sb.remove = True  # type: ignore[attr-defined]
            await sb.__aexit__(None, None, None)
        except Exception as e:
            logger.warning("[sandbox_pool] release failed: %s", e)

    async def close(self) -> None:
        """Cancel in-flight warms and rm all unused ready containers."""
        tasks = list(self._warm_tasks)
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._cv:
            ready = self._ready
            self._ready = defaultdict(list)
            self._inflight.clear()
            self._images_warmed_ok.clear()
            self._size = 0
            self._cv.notify_all()
        for q in ready.values():
            for sb in q:
                try:
                    if hasattr(sb, "remove"):
                        sb.remove = True  # type: ignore[attr-defined]
                    await sb.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning("[sandbox_pool] close cleanup failed: %s", e)


def get_pool() -> SandboxPool:
    global _POOL
    if _POOL is None:
        _POOL = SandboxPool()
    return _POOL


def reset_pool_for_tests() -> None:
    """Drop the singleton (unit tests only)."""
    global _POOL
    _POOL = None


def schedule_prefetch(
    groups: list[list[Sample]],
    *,
    protocol: str = swe.PROTOCOL_SCALESWE,
    max_size: int | None = None,
) -> asyncio.Task | None:
    """Fire-and-forget prefetch on the running event loop."""
    if not pool_enabled() or not groups:
        return None
    pool = get_pool()
    if max_size is not None:
        pool.configure(max_size=max_size)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("[sandbox_pool] schedule_prefetch: no running loop")
        return None
    task = loop.create_task(pool.prefetch(groups, protocol=protocol), name="sandbox_pool_prefetch")
    pool._track_task(task)
    return task
