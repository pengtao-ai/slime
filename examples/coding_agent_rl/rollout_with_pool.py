"""Coding-agent rollout entry that prefetches the next sandbox batch.

Wire via::

    --rollout-function-path examples.coding_agent_rl.rollout_with_pool.generate_rollout

When ``SANDBOX_POOL`` is enabled (default), each ``get_samples`` call also
``reserve_samples`` the following batch and schedules ``SandboxPool.prefetch``
for both the current batch and the next. ``SANDBOX_POOL_MAX`` defaults to
``2 * rollout_batch_size * n_samples_per_prompt`` so current in-flight
sandboxes and the next-step warm set can coexist.
"""

from __future__ import annotations

import atexit
import logging
import os
from argparse import Namespace
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.sglang_rollout import eval_rollout, generate_rollout_async
from slime.utils.async_utils import run

from . import sandbox_pool, swe
from .generate import CONFIG

logger = logging.getLogger(__name__)

_ATEXIT_REGISTERED = False


def _rollout_sample_count(args: Namespace) -> int:
    return max(1, int(args.rollout_batch_size) * int(args.n_samples_per_prompt))


def _pool_max_from_args(args: Namespace) -> int:
    """Pool capacity: default = 2× one rollout (current + next warm)."""
    env = (os.environ.get("SANDBOX_POOL_MAX") or "").strip()
    if env:
        return max(1, int(env))
    return 2 * _rollout_sample_count(args)


def _ensure_atexit_close() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    _ATEXIT_REGISTERED = True

    def _close() -> None:
        if not sandbox_pool.pool_enabled():
            return
        try:
            run(sandbox_pool.get_pool().close())
        except Exception as e:
            logger.warning("[rollout_with_pool] atexit pool.close failed: %s", e)

    atexit.register(_close)


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Same contract as ``slime.rollout.sglang_rollout.generate_rollout``."""
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    pool_on = sandbox_pool.pool_enabled()
    one_rollout = _rollout_sample_count(args)
    pool_max = _pool_max_from_args(args)
    if pool_on:
        sandbox_pool.get_pool().configure(max_size=pool_max)
        _ensure_atexit_close()
        logger.info(
            "[rollout_with_pool] SANDBOX_POOL on max=%d "
            "(one_rollout=%d ≈ current+next; rollout_batch_size=%s n_samples_per_prompt=%s)",
            pool_max,
            one_rollout,
            args.rollout_batch_size,
            args.n_samples_per_prompt,
        )

    protocol = CONFIG.train_protocol or swe.PROTOCOL_SCALESWE

    def get_samples(n: int):
        groups = data_source.get_samples(n)
        if pool_on:
            # Warm the batch we just took (so acquire can wait on in-flight warms)
            # and reserve+prefetch the following batch for the next step.
            # pool_max defaults to 2×rollout so both fit without skipping next.
            sandbox_pool.schedule_prefetch(groups, protocol=protocol, max_size=pool_max)
            if hasattr(data_source, "reserve_samples"):
                nxt = data_source.reserve_samples(n)
                sandbox_pool.schedule_prefetch(nxt, protocol=protocol, max_size=pool_max)
        return groups

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, get_samples))
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output
