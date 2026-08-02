#!/usr/bin/env python3
"""Offline regrade: score existing ``patch.diff`` without re-running the agent.

Reads per-sample dirs under ``--out-dir`` (as written by ``infer_cc_offload_traj.py``)::

    OUT_DIR/i851_<instance_id>/
      summary.json
      patch.diff

Looks up scaleswe metadata from the original ``--jsonl`` by ``summary.index``,
calls ``swe.run_evaluation``, then updates ``reward`` / ``eval_applied`` in place.

No GLM / Claude Code / adapter. Needs docker (or docker-rt) and the instance images.

Example::

    CONCURRENCY=4 python examples/coding_agent_rl/regrade_patches.py \\
      --out-dir runs/infer_cc_glm_20260728_141118 \\
      --jsonl examples/coding_agent_rl/data/swe_train_scaleswe.jsonl \\
      --only-with-patch

Or::

    bash examples/coding_agent_rl/run_regrade_patches.sh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EXAMPLE_DIR))

import swe  # noqa: E402
from slime.utils.types import Sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("regrade_patches")

DEFAULT_JSONL = _EXAMPLE_DIR / "data" / "swe_train_scaleswe.jsonl"
DEFAULT_OUT = _REPO_ROOT / "runs" / "infer_cc_glm_20260728_141118"


def _setup_docker_env(*, network: str, name_prefix: str) -> None:
    os.environ.setdefault("SLIME_AGENT_SANDBOX_BACKEND", "docker")
    os.environ["SLIME_AGENT_DOCKER_NETWORK"] = network
    os.environ["SLIME_AGENT_DOCKER_NAME_PREFIX"] = name_prefix
    if network == "host":
        os.environ.pop("SLIME_AGENT_DOCKER_ADD_HOST", None)
    else:
        os.environ.setdefault("SLIME_AGENT_DOCKER_ADD_HOST", "host.docker.internal:host-gateway")


def _require_docker(*, dry_run: bool) -> None:
    """Fail fast when the host cannot talk to a docker / docker-rt daemon.

    Eval sandboxes are created via ``docker run`` (``DockerSandbox``). A jupyter
    pod without ``/var/run/docker.sock`` (and without ``DOCKER_HOST``) will fail
    every sample the same way — catch that before the batch starts.
    """
    if dry_run:
        return
    import shutil
    import subprocess

    docker = shutil.which("docker")
    if not docker:
        raise SystemExit(
            "ERROR: `docker` not found on PATH. Regrade needs the same docker/docker-rt "
            "CLI as training/infer (SLIME_AGENT_SANDBOX_BACKEND=docker)."
        )
    sock = Path("/var/run/docker.sock")
    docker_host = (os.environ.get("DOCKER_HOST") or "").strip()
    if not docker_host and not sock.exists():
        raise SystemExit(
            "ERROR: no docker daemon reachable.\n"
            f"  DOCKER_HOST is unset and {sock} does not exist.\n"
            "  This jupyter/dev pod cannot create eval sandboxes.\n"
            "  Run regrade on the same host/pod where infer/train docker sandboxes work,\n"
            "  or set DOCKER_HOST to your docker-rt endpoint and retry.\n"
            "  Quick check: `docker info`"
        )
    try:
        proc = subprocess.run(
            [docker, "info"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: failed to run `docker info`: {exc}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(
            "ERROR: `docker info` failed — cannot create eval sandboxes.\n"
            f"  {(err[:500] if err else f'exit={proc.returncode}')}\n"
            "  Fix docker/docker-rt access (same env as infer), then retry."
        )
    print(f"[regrade] docker ok ({docker})", flush=True)


def _load_jsonl_by_index(jsonl: Path) -> dict[int, Sample]:
    """Same 0-based non-empty-line indexing as ``infer_cc_offload_traj._load_samples``."""
    out: dict[int, Sample] = {}
    data_i = -1
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data_i += 1
            row = json.loads(line)
            out[data_i] = Sample(
                index=data_i,
                prompt=row.get("prompt") or "",
                label=row.get("label"),
                metadata=row.get("metadata") or {},
            )
    if not out:
        raise SystemExit(f"no samples loaded from {jsonl}")
    return out


def _discover_sample_dirs(out_dir: Path) -> list[Path]:
    dirs = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("i")]
    return sorted(dirs, key=lambda p: p.name)


def _read_summary(sample_dir: Path) -> dict[str, Any] | None:
    path = sample_dir / "summary.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_patch(sample_dir: Path) -> str:
    path = sample_dir / "patch.diff"
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _should_skip(
    summary: dict[str, Any],
    patch: str,
    *,
    only_with_patch: bool,
    skip_eval_applied: bool,
    force: bool,
    only_regraded: bool,
) -> str | None:
    if only_with_patch and not (patch or "").strip():
        return "empty_patch"
    has_regrade = isinstance(summary.get("regrade"), dict)
    if only_regraded and not has_regrade:
        return "not_previously_regraded"
    # Prior regrade pass may have used a buggy index mapping — always redo.
    if has_regrade:
        return None
    if skip_eval_applied and summary.get("eval_applied") and not force:
        return "already_eval_applied"
    return None


async def _grade_one(
    *,
    sample_dir: Path,
    sample: Sample,
    summary: dict[str, Any],
    patch: str,
    timeout_sec: int,
    image_override: str | None,
    dry_run: bool,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        index = int(summary.get("index", sample.index or -1))
        md = swe.get_metadata(sample, swe.PROTOCOL_SCALESWE)
        if image_override:
            md = {**md, "image": image_override}
        instance_id = md.get("instance_id") or summary.get("instance_id")
        reason = swe.evaluability_check(md)
        if reason:
            result = {
                "index": index,
                "instance_id": instance_id,
                "sample_dir": sample_dir.name,
                "skipped": False,
                "ok": False,
                "error": f"unevaluatable:{reason}",
                "reward": 0.0,
                "eval_applied": False,
                "patch_chars": len(patch or ""),
                "elapsed_sec": 0.0,
            }
            logger.warning("[regrade:%s] skip grade: %s", index, reason)
            return result

        print(
            f"[regrade:{index}] start instance={instance_id} "
            f"patch_chars={len(patch or '')} dry_run={dry_run}",
            flush=True,
        )
        t0 = time.monotonic()
        if dry_run:
            return {
                "index": index,
                "instance_id": instance_id,
                "sample_dir": sample_dir.name,
                "skipped": False,
                "ok": True,
                "dry_run": True,
                "reward": summary.get("reward"),
                "eval_applied": summary.get("eval_applied"),
                "patch_chars": len(patch or ""),
                "elapsed_sec": 0.0,
                "error": None,
            }

        error: str | None = None
        reward = 0.0
        applied = False
        try:
            ev = await swe.run_evaluation(md, diff_text=patch, timeout_sec=timeout_sec)
            reward, applied = float(ev.reward), bool(ev.applied_cleanly)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("[regrade:%s] failed instance=%s", index, instance_id)

        elapsed = round(time.monotonic() - t0, 3)
        summary["reward"] = reward
        summary["eval_applied"] = applied
        summary["patch_chars"] = len(patch or "")
        summary["regrade"] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": elapsed,
            "error": error,
        }
        (sample_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[regrade:{index}] done reward={reward} applied={applied} "
            f"elapsed={elapsed}s err={error}",
            flush=True,
        )
        return {
            "index": index,
            "instance_id": instance_id,
            "sample_dir": sample_dir.name,
            "skipped": False,
            "ok": error is None,
            "reward": reward,
            "eval_applied": applied,
            "patch_chars": len(patch or ""),
            "elapsed_sec": elapsed,
            "error": error,
        }


async def run_regrade(args: argparse.Namespace) -> None:
    _setup_docker_env(network=args.network, name_prefix=args.name_prefix)
    if args.pull:
        os.environ["SLIME_AGENT_DOCKER_PULL"] = "1"
    _require_docker(dry_run=bool(args.dry_run))

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise SystemExit(f"--out-dir not found: {out_dir}")

    jsonl = Path(args.jsonl)
    if not jsonl.is_file():
        raise SystemExit(f"--jsonl not found: {jsonl}")

    by_index = _load_jsonl_by_index(jsonl)
    sample_dirs = _discover_sample_dirs(out_dir)

    index_filter: set[int] | None = None
    if args.indices:
        index_filter = {int(x) for x in args.indices.split(",") if x.strip()}

    pending: list[tuple[Path, Sample, dict[str, Any], str]] = []
    skipped: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        summary = _read_summary(sample_dir)
        if summary is None:
            skipped.append({"sample_dir": sample_dir.name, "reason": "no_summary"})
            continue
        try:
            index = int(summary["index"])
        except (KeyError, TypeError, ValueError):
            skipped.append({"sample_dir": sample_dir.name, "reason": "bad_index"})
            continue
        if index_filter is not None and index not in index_filter:
            skipped.append({"sample_dir": sample_dir.name, "index": index, "reason": "index_filter"})
            continue
        if args.offset and index < args.offset:
            skipped.append({"sample_dir": sample_dir.name, "index": index, "reason": "offset"})
            continue
        if args.min_index is not None and index < args.min_index:
            skipped.append({"sample_dir": sample_dir.name, "index": index, "reason": "min_index"})
            continue
        if args.max_index is not None and index > args.max_index:
            skipped.append({"sample_dir": sample_dir.name, "index": index, "reason": "max_index"})
            continue

        sample = by_index.get(index)
        if sample is None:
            skipped.append({"sample_dir": sample_dir.name, "index": index, "reason": "missing_jsonl_row"})
            continue

        patch = _read_patch(sample_dir)
        reason = _should_skip(
            summary,
            patch,
            only_with_patch=args.only_with_patch,
            skip_eval_applied=args.skip_eval_applied,
            force=args.force,
            only_regraded=args.only_regraded,
        )
        if reason:
            skipped.append(
                {
                    "sample_dir": sample_dir.name,
                    "index": index,
                    "reason": reason,
                    "reward": summary.get("reward"),
                    "eval_applied": summary.get("eval_applied"),
                    "patch_chars": len(patch or ""),
                }
            )
            continue
        pending.append((sample_dir, sample, summary, patch))
        if args.limit is not None and len(pending) >= args.limit:
            break

    print(
        f"[regrade] out_dir={out_dir} jsonl={jsonl} "
        f"discovered={len(sample_dirs)} pending={len(pending)} skipped={len(skipped)} "
        f"concurrency={args.concurrency} dry_run={args.dry_run}",
        flush=True,
    )
    if not pending:
        print("[regrade] nothing to grade", flush=True)
        return

    sem = asyncio.Semaphore(max(1, args.concurrency))
    tasks = [
        _grade_one(
            sample_dir=sample_dir,
            sample=sample,
            summary=summary,
            patch=patch,
            timeout_sec=args.eval_timeout,
            image_override=args.image,
            dry_run=args.dry_run,
            sem=sem,
        )
        for sample_dir, sample, summary, patch in pending
    ]
    results = list(await asyncio.gather(*tasks))

    graded = [r for r in results if not r.get("skipped")]
    ok = [r for r in graded if r.get("ok")]
    solved = [r for r in graded if float(r.get("reward") or 0.0) >= 1.0]
    applied = [r for r in graded if r.get("eval_applied")]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "regrade_patches",
        "out_dir": str(out_dir),
        "jsonl": str(jsonl),
        "dry_run": bool(args.dry_run),
        "only_with_patch": bool(args.only_with_patch),
        "skip_eval_applied": bool(args.skip_eval_applied),
        "force": bool(args.force),
        "concurrency": args.concurrency,
        "eval_timeout_sec": args.eval_timeout,
        "discovered": len(sample_dirs),
        "pending": len(pending),
        "skipped": len(skipped),
        "graded": len(graded),
        "ok": len(ok),
        "eval_applied_true": len(applied),
        "solved": len(solved),
        "pass_rate_among_graded": (len(solved) / len(graded) if graded else 0.0),
        "pass_rate_among_applied": (len(solved) / len(applied) if applied else 0.0),
        "skipped_breakdown": _count_reasons(skipped),
        "results": sorted(graded, key=lambda r: int(r.get("index", -1))),
        "skipped_samples": skipped,
    }
    report_path = out_dir / "regrade_summary.json"
    if not args.dry_run:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[regrade] done graded={len(graded)} ok={len(ok)} "
        f"applied={len(applied)} solved={len(solved)} "
        f"pass@graded={report['pass_rate_among_graded']:.3f} "
        f"pass@applied={report['pass_rate_among_applied']:.3f} "
        f"report={report_path if not args.dry_run else '(dry-run)'}",
        flush=True,
    )


def _count_reasons(skipped: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in skipped:
        key = str(row.get("reason") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=Path(os.environ.get("OUT_DIR", str(DEFAULT_OUT))))
    p.add_argument("--jsonl", type=Path, default=Path(os.environ.get("PROMPT_DATA", str(DEFAULT_JSONL))))
    p.add_argument("--image", default=None, help="Override docker image for all samples")
    p.add_argument("--pull", action="store_true", help="Allow docker pull if image missing")
    p.add_argument(
        "--network",
        default=os.environ.get("SLIME_AGENT_DOCKER_NETWORK", "bridge"),
    )
    p.add_argument(
        "--name-prefix",
        default=os.environ.get("SLIME_AGENT_DOCKER_NAME_PREFIX", "cc-regrade"),
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("CONCURRENCY", os.environ.get("REGRADE_CONCURRENCY", "4"))),
    )
    p.add_argument(
        "--eval-timeout",
        type=int,
        default=int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "600")),
    )
    p.add_argument("--limit", type=int, default=None, help="Max samples to grade this run")
    p.add_argument("--offset", type=int, default=0, help="Skip summary.index < offset")
    p.add_argument("--min-index", type=int, default=None)
    p.add_argument("--max-index", type=int, default=None)
    p.add_argument("--indices", default=None, help="Comma-separated summary.index list")
    p.add_argument(
        "--only-with-patch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only grade samples with non-empty patch.diff (default: true)",
    )
    p.add_argument(
        "--skip-eval-applied",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip samples already marked eval_applied=true (default: true)",
    )
    p.add_argument("--force", action="store_true", help="Re-grade even if eval_applied=true")
    p.add_argument(
        "--only-regraded",
        action="store_true",
        help="Only re-run samples that already have a summary.regrade block (redo prior pass)",
    )
    p.add_argument("--dry-run", action="store_true", help="List pending samples without grading")
    args = p.parse_args()
    asyncio.run(run_regrade(args))


if __name__ == "__main__":
    main()
