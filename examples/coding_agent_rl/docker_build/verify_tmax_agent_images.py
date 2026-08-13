#!/usr/bin/env python3
"""Verify baked Tmax agent images via ``crane export``.

Checks (per unique ``metadata.image``):

* ``/etc/passwd`` exists and is non-empty
* ``/home/user`` exists
* no ``/tmp_build``, no ``/tests`` (deferred verifier must stay out)
* if ``/tmp`` exists: mode ``1777``, no leftover bake artifacts
* pre-baked CLIs under ``/usr/local/bin``: node, npm, claude, opencode, pi, mini

Example::

    python3 examples/coding_agent_rl/docker_build/verify_tmax_agent_images.py --limit 1
"""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bake_common import AGENT_ASSET_NAMES, check_prebaked_clis, resolve_crane

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR.parent
_VERIFY_TMP_ROOT = Path(os.environ.get("VERIFY_TMP_ROOT", "/tmp/slime-verify-tmax"))
_EXPECTED_TMP_MODE = 0o1777
_FORBIDDEN_TMP_LEFTOVERS = AGENT_ASSET_NAMES
_FORBIDDEN_PATHS = ("tmp_build", "tests")


@dataclass(frozen=True)
class VerifyJob:
    instance_id: str
    image: str


def cleanup_verify_tmp(root: Path = _VERIFY_TMP_ROOT) -> int:
    if not root.exists():
        return 0
    removed = 0
    try:
        for child in list(root.iterdir()):
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
        try:
            root.rmdir()
        except OSError:
            pass
    except OSError:
        shutil.rmtree(root, ignore_errors=True)
        removed += 1
    if removed:
        print(f"[verify] cleaned {removed} path(s) under {root}", flush=True)
    return removed


def _instance_id_of(row: dict[str, Any]) -> str:
    md = row.get("metadata") or {}
    iid = md.get("instance_id") or row.get("label")
    if not iid:
        raise ValueError("row missing instance_id / label")
    return str(iid)


def _target_image_of(row: dict[str, Any]) -> str:
    md = row.get("metadata") or {}
    image = md.get("image")
    if not image:
        raise ValueError(f"row {_instance_id_of(row)!r} missing metadata.image")
    return str(image)


def load_jobs(path: Path) -> list[VerifyJob]:
    jobs: list[VerifyJob] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image = _target_image_of(row)
            if image in seen:
                continue
            seen.add(image)
            jobs.append(VerifyJob(instance_id=_instance_id_of(row), image=image))
    return jobs


def export_image(crane_bin: str, image: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    export = subprocess.Popen(
        [crane_bin, "export", image],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert export.stdout is not None
    extract = subprocess.Popen(
        ["tar", "-x", "-C", str(dest)],
        stdin=export.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    export.stdout.close()
    _exp_out, exp_err = export.communicate()
    _tar_out, tar_err = extract.communicate()
    if export.returncode != 0:
        raise RuntimeError(
            f"crane export failed for {image}:\n{(exp_err or b'').decode(errors='replace')[-1000:]}"
        )
    if extract.returncode != 0:
        raise RuntimeError(
            f"tar extract failed for {image}:\n{(tar_err or b'').decode(errors='replace')[-1000:]}"
        )


def _check_tmp_layout(root: Path, errors: list[str]) -> None:
    tmp = root / "tmp"
    if not tmp.is_dir():
        return
    mode = tmp.stat().st_mode & 0o7777
    if mode != _EXPECTED_TMP_MODE:
        errors.append(f"/tmp mode is {mode:04o}, expected {_EXPECTED_TMP_MODE:04o}")
    for name in _FORBIDDEN_TMP_LEFTOVERS:
        leftover = tmp / name
        if leftover.exists():
            errors.append(f"forbidden bake leftover: /tmp/{name}")


def verify_extracted(job: VerifyJob, root: Path) -> dict[str, Any]:
    errors: list[str] = []

    passwd = root / "etc" / "passwd"
    if not passwd.is_file():
        errors.append("missing etc/passwd")
    elif passwd.stat().st_size == 0:
        errors.append("etc/passwd is empty")

    home_user = root / "home" / "user"
    if not home_user.is_dir():
        errors.append("missing /home/user")

    for name in _FORBIDDEN_PATHS:
        if (root / name).exists():
            errors.append(f"forbidden path present: /{name}/")

    _check_tmp_layout(root, errors)
    check_prebaked_clis(root, errors)

    return {
        "instance_id": job.instance_id,
        "image": job.image,
        "ok": not errors,
        "errors": errors,
    }


def verify_job(job: VerifyJob, *, crane_bin: str) -> dict[str, Any]:
    _VERIFY_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = Path(
        tempfile.mkdtemp(
            prefix=f"{job.instance_id}-",
            dir=str(_VERIFY_TMP_ROOT),
        )
    )
    try:
        print(f"[verify] export {job.image} → {tmp}", flush=True)
        export_image(crane_bin, job.image, tmp)
        return verify_extracted(job, tmp)
    except Exception as e:
        return {
            "instance_id": job.instance_id,
            "image": job.image,
            "ok": False,
            "errors": [f"{type(e).__name__}: {e}"],
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=_EXAMPLE_DIR / "data" / "tmax_train_200_baked.jsonl",
    )
    p.add_argument("--limit", type=int, default=0, help="Max unique images (0=all)")
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("VERIFY_WORKERS", "4")),
    )
    p.add_argument(
        "--failures-out",
        type=Path,
        default=_SCRIPT_DIR / "verify_tmax_failures.jsonl",
    )
    p.add_argument("--crane", default=os.environ.get("CRANE", ""))
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"missing input: {args.input}")

    crane_bin = resolve_crane(args.crane or None)
    cleanup_verify_tmp()
    atexit.register(cleanup_verify_tmp)

    jobs = load_jobs(args.input)
    print(f"[verify] loaded {len(jobs)} unique images from {args.input}", flush=True)
    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]
        print(f"[verify] --limit {args.limit} → {len(jobs)} images", flush=True)

    def _run(job: VerifyJob) -> dict[str, Any]:
        return verify_job(job, crane_bin=crane_bin)

    try:
        workers = max(1, int(args.workers))
        if workers == 1 or len(jobs) <= 1:
            results = [_run(j) for j in jobs]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_run, jobs))
    finally:
        cleanup_verify_tmp()

    failures = [r for r in results if not r.get("ok")]
    for r in results:
        if r.get("ok"):
            print(f"[verify] OK {r['instance_id']} {r['image']}", flush=True)
        else:
            print(
                f"[verify] FAIL {r['instance_id']}: {'; '.join(r.get('errors') or [])}",
                flush=True,
            )

    args.failures_out.parent.mkdir(parents=True, exist_ok=True)
    with args.failures_out.open("w", encoding="utf-8") as ff:
        for item in failures:
            ff.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"[verify] done ok={len(results) - len(failures)} fail={len(failures)} "
        f"→ {args.failures_out}; tmp={_VERIFY_TMP_ROOT} cleaned",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
