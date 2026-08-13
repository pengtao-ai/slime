#!/usr/bin/env python3
"""Inference-only for Tmax: Claude Code → Anthropic adapter → remote LLM + inplace eval.

Same GLM-only path as ``infer_cc_offload_traj.py``, but dataset protocol is
``tmax`` and ``--eval`` grades final sandbox state via ``swe.grade_tmax_inplace``
(deferred ``test_sh``), matching ``generate.py``.

Batch mode (default): first ``--limit`` jsonl rows with ``--concurrency`` parallel
docker+CC sessions sharing one adapter. Per-sample outputs live under::

    OUT_DIR/summary.json
    OUT_DIR/i000_<instance_id>/
      requests/req_XX.json
      trajectory.json / turns.jsonl / summary.json

Prereqs::

    export DASHSCOPE_API_KEY=...
    export DASHSCOPE_BASE_URL=https://.../v1
    export DASHSCOPE_MODEL=deepseek-v4-flash-0731   # optional

Then::

    python examples/coding_agent_rl/infer_cc_tmax_traj.py \\
      --jsonl examples/coding_agent_rl/data/tmax_smoke_3.jsonl \\
      --out-dir /tmp/cc_tmax_traj --limit 3 --eval --time-budget 900

Or::

    bash examples/coding_agent_rl/run_infer_cc_tmax_traj.sh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EXAMPLE_DIR))

import infer_cc_offload_traj as base  # noqa: E402
import smoke_claude_code_docker as smoke  # noqa: E402
import swe  # noqa: E402
from slime.agent.sandbox import DockerSandbox, ensure_agent_user  # noqa: E402
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

from agents_registry import resolve_agent  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_cc_tmax")

DEFAULT_JSONL = _EXAMPLE_DIR / "data" / "tmax_smoke_3.jsonl"
DEFAULT_PROMPT = (
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the task. "
    "Use tools as needed. When finished, print a one-line summary and exit."
)


def _resolve_protocol(sample: Sample) -> str:
    raw = (sample.metadata or {}).get("protocol")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return swe.PROTOCOL_TMAX


async def _run_one_sample(
    *,
    args: argparse.Namespace,
    adapters: base.DualInferAdapters,
    adapter_port: int,
    index: int,
    sample: Sample,
    sample_dir: Path,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        protocol = _resolve_protocol(sample)
        md = swe.get_metadata(sample, protocol)
        agent_spec = resolve_agent(md.get("agent"))
        harness = agent_spec.harness_cls()
        image = args.image or md["image"]
        workdir = md["workdir"]
        instance_id = md["instance_id"]
        if not image or not workdir:
            raise SystemExit(f"[infer:{index}] missing image/workdir for {instance_id}")
        uneval = swe.evaluability_check(md)
        if args.eval and uneval:
            raise SystemExit(f"[infer:{index}] unevaluable ({uneval}): {instance_id}")

        session_id = f"infer-tmax-{agent_spec.name}-{index}-{secrets.token_hex(6)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        requests_dir = sample_dir / "requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        adapters.open_session(
            session_id,
            protocol=agent_spec.adapter_protocol,
            sampling_defaults={"temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
            max_context_tokens=args.max_context_len,
        )
        adapters.bind_session_outdir(session_id, requests_dir)

        agent_exit_code = -999
        harness_traj = ""
        patch_diff = ""
        reward = 0.0
        applied = False
        error: str | None = None
        t0 = time.monotonic()
        try:
            print(
                f"[infer:{index}] start agent={agent_spec.name} instance={instance_id} "
                f"protocol={md.get('protocol')} image={image}",
                flush=True,
            )
            async with DockerSandbox(image) as sb:
                await ensure_agent_user(sb, workdir)
                code, out, err = await sb.exec("id -u; id -un", user="agent", timeout=30)
                uid = (out or "").strip().splitlines()[0] if (out or "").strip() else ""
                if code != 0 or uid == "0" or "root" in (out or "").split():
                    raise RuntimeError(f"sandbox still root under user=agent out={out!r} err={err!r}")

                adapter_host = await smoke._pick_adapter_host(
                    sb, port=adapter_port, preferred=(args.public_host or None)
                )
                adapter_url = f"http://{adapter_host}:{adapter_port}"
                await harness.install_cli(sb)
                await swe.prepare_workspace(sb, workdir, md)
                if args.shrink_problem:
                    await sb.write_file(
                        f"{workdir}/PROBLEM_STATEMENT.md",
                        "# Infer smoke\n\nInvestigate briefly with tools, then stop.\n",
                        user="agent",
                    )
                agent_exit_code = await harness.run(
                    sb,
                    workdir=workdir,
                    session_id=session_id,
                    adapter_url=adapter_url,
                    time_budget_sec=args.time_budget,
                    prompt=args.prompt,
                )
                _, traj_out, _ = await sb.exec(
                    f"cat {workdir}/.harness/trajectory.jsonl 2>/dev/null || true",
                    timeout=60,
                )
                harness_traj = traj_out or ""

                is_tmax = md.get("protocol") == swe.PROTOCOL_TMAX
                if not is_tmax:
                    patch_diff = await swe.git_diff(sb, workdir)

                # Tmax scores final env state on the *agent* sandbox (must stay open).
                if args.eval:
                    if is_tmax:
                        reward, applied = await swe.grade_tmax_inplace(
                            sb, md, timeout_sec=args.eval_timeout
                        )
                    else:
                        reward, applied = await swe.run_evaluation(
                            md, diff_text=patch_diff, timeout_sec=args.eval_timeout
                        )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "[infer:%s] failed instance=%s agent=%s", index, instance_id, agent_spec.name
            )
        finally:
            traj_turns = adapters.pop_session_traj(session_id)
            await adapters.drop_session(session_id, wait_timeout=10)

        base_url, model = base._glm_endpoint()
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ok": bool(traj_turns) and error is None,
            "mode": "glm_only_tmax",
            "protocol": md.get("protocol"),
            "agent": agent_spec.name,
            "adapter_protocol": agent_spec.adapter_protocol,
            "index": index,
            "instance_id": instance_id,
            "session_id": session_id,
            "agent_exit_code": agent_exit_code,
            "turns": len(traj_turns),
            "tool_use_turns": base._count_tool_use_turns(traj_turns),
            "reward": reward,
            "eval_applied": applied,
            "eval": bool(args.eval),
            "patch_chars": len(patch_diff or ""),
            "elapsed_sec": round(time.monotonic() - t0, 3),
            "error": error,
            "dashscope_base_url": base_url,
            "dashscope_model": model,
            "prompt": args.prompt,
            "time_budget_sec": args.time_budget,
        }
        base._save_outputs(
            out_dir=sample_dir,
            traj_turns=traj_turns,
            harness_traj=harness_traj,
            patch_diff=patch_diff or "",
            summary=summary,
        )
        print(
            f"[infer:{index}] done agent={agent_spec.name} ok={summary['ok']} turns={summary['turns']} "
            f"reward={reward} exit={agent_exit_code} err={error}",
            flush=True,
        )
        return summary


async def run_infer(args: argparse.Namespace) -> None:
    os.environ["SLIME_AGENT_OFFLOAD"] = "0"
    base._require_glm_env()
    smoke._setup_docker_env(network=args.network)
    base.setup_agent_tarball_envs(node_tarball=args.node_tarball, cc_tarball=args.cc_tarball)
    os.environ["SWE_CC_PROMPT"] = args.prompt
    if args.pull:
        os.environ["SLIME_AGENT_DOCKER_PULL"] = "1"

    samples = base._load_samples(args.jsonl, limit=args.limit, offset=args.offset)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force and not args.resume:
        raise SystemExit(
            f"--out-dir already exists and is non-empty: {out_dir} (pass --force or --resume)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[int, Sample, Path]] = []
    done_results: list[dict[str, Any]] = []
    for index, sample in samples:
        instance_id = str(
            (sample.metadata or {}).get("instance_id") or sample.label or f"row{index}"
        )
        sample_dir = base._sample_out_dir(out_dir, index, instance_id)
        existing = base._load_done_summary(sample_dir) if args.resume else None
        if existing is not None:
            done_results.append(existing)
            print(
                f"[infer:{index}] skip (resume) ok={existing.get('ok')} "
                f"turns={existing.get('turns')} reward={existing.get('reward')} "
                f"instance={instance_id}",
                flush=True,
            )
            continue
        if args.resume and sample_dir.exists():
            shutil.rmtree(sample_dir)
            print(f"[infer:{index}] cleared incomplete dir {sample_dir.name}", flush=True)
        pending.append((index, sample, sample_dir))

    print(
        f"[infer] samples={len(samples)} resume_skip={len(done_results)} "
        f"pending={len(pending)} concurrency={args.concurrency} eval={args.eval}",
        flush=True,
    )

    base_url, model = base._glm_endpoint()
    new_results: list[dict[str, Any]] = []
    if pending:
        adapters = base.DualInferAdapters(
            enable_thinking=not args.no_thinking,
            reasoning_effort=args.reasoning_effort or None,
        )
        handle = run_app_in_thread(
            adapters.app,
            host=args.bind_host,
            port=args.bind_port,
            thread_name="infer-tmax-adapter",
            runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
        )
        print(
            f"[infer] adapter {args.bind_host}:{handle.port}  glm={model} @ {base_url}",
            flush=True,
        )
        sem = asyncio.Semaphore(max(1, int(args.concurrency)))
        try:
            tasks = [
                _run_one_sample(
                    args=args,
                    adapters=adapters,
                    adapter_port=handle.port,
                    index=index,
                    sample=sample,
                    sample_dir=sample_dir,
                    sem=sem,
                )
                for index, sample, sample_dir in pending
            ]
            new_results = list(await asyncio.gather(*tasks))
        finally:
            handle.stop()

    results = sorted(done_results + new_results, key=lambda r: int(r.get("index", -1)))

    root_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "glm_only_tmax_batch",
        "jsonl": str(args.jsonl),
        "offset": args.offset,
        "limit": args.limit,
        "concurrency": args.concurrency,
        "eval": bool(args.eval),
        "resume": bool(args.resume),
        "resume_skipped": len(done_results),
        "n": len(results),
        "ok": sum(1 for r in results if r.get("ok")),
        "failed": sum(1 for r in results if not r.get("ok")),
        "solved": sum(1 for r in results if float(r.get("reward") or 0.0) >= 1.0),
        "mean_reward": (
            sum(float(r.get("reward") or 0.0) for r in results) / len(results) if results else 0.0
        ),
        "dashscope_base_url": base_url,
        "dashscope_model": model,
        "time_budget_sec": args.time_budget,
        "results": results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(root_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[infer] batch done ok={root_summary['ok']}/{root_summary['n']} "
        f"failed={root_summary['failed']} solved={root_summary['solved']} "
        f"mean_reward={root_summary['mean_reward']:.3f} out={out_dir}",
        flush=True,
    )
    if root_summary["ok"] == 0:
        raise SystemExit("FAIL: all samples failed / empty")
    if args.require_exit_zero and any(int(r.get("agent_exit_code", -1)) != 0 for r in results):
        raise SystemExit("FAIL: some agent_exit_code != 0")
    print("[infer] PASS", flush=True)


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = _REPO_ROOT / "runs" / f"infer_cc_tmax_{stamp}"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    p.add_argument("--image", default=None)
    p.add_argument("--pull", action="store_true")
    p.add_argument("--node-tarball", type=Path, default=Path(os.environ.get("SLIME_AGENT_NODE_TARBALL", smoke.DEFAULT_NODE)))
    p.add_argument("--cc-tarball", type=Path, default=Path(os.environ.get("SLIME_AGENT_CC_TARBALL", smoke.DEFAULT_CC)))
    p.add_argument("--bind-host", default=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"))
    p.add_argument("--bind-port", type=int, default=int(os.environ.get("ADAPTER_PORT", "18021")))
    p.add_argument("--public-host", default=os.environ.get("ADAPTER_PUBLIC_HOST", ""))
    p.add_argument("--network", default=os.environ.get("SLIME_AGENT_DOCKER_NETWORK", "bridge"))
    p.add_argument("--time-budget", type=int, default=int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "900")))
    p.add_argument("--max-context-len", type=int, default=int(os.environ.get("SMOKE_MAX_CONTEXT_LEN", "96000")))
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "8192")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("SMOKE_TEMPERATURE", "1.0")))
    p.add_argument("--prompt", default=os.environ.get("SWE_CC_PROMPT", DEFAULT_PROMPT))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("INFER_OUT_DIR", str(default_out))),
    )
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse --out-dir: skip samples that already have summary.json; re-run incomplete dirs",
    )
    p.add_argument("--shrink-problem", action="store_true")
    p.add_argument(
        "--eval",
        action="store_true",
        help="Run grader after agent exit (tmax: grade_tmax_inplace on agent sandbox)",
    )
    p.add_argument("--eval-timeout", type=int, default=600)
    p.add_argument("--require-exit-zero", action="store_true")
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable chat_template_kwargs.thinking",
    )
    p.add_argument(
        "--reasoning-effort",
        default=os.environ.get("INFER_REASONING_EFFORT", ""),
        help="Optional chat_template_kwargs.reasoning_effort (e.g. high)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("INFER_LIMIT", "3")),
        help="Number of jsonl rows to run (from --offset)",
    )
    p.add_argument(
        "--offset",
        type=int,
        default=int(os.environ.get("INFER_OFFSET", "0")),
        help="Skip this many jsonl rows before --limit",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("INFER_CONCURRENCY", "3")),
        help="Max concurrent docker+CC sessions",
    )
    args = p.parse_args()

    smoke._require_file(args.node_tarball, "SLIME_AGENT_NODE_TARBALL")
    smoke._require_file(args.cc_tarball, "SLIME_AGENT_CC_TARBALL")
    smoke._require_file(args.jsonl, "jsonl")
    if args.limit <= 0:
        raise SystemExit("--limit must be > 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")

    asyncio.run(run_infer(args))


if __name__ == "__main__":
    main()
