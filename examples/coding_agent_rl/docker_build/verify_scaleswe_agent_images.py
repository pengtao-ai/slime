#!/usr/bin/env python3
"""Verify baked ScaleSWE agent images via ``crane export``.

Checks (per unique ``metadata.image``):

* ``/etc/passwd`` exists and is non-empty
* no ``deepswe/`` or ``scaleswe/`` path prefixes in the image rootfs
* no ``/tmp_build`` (bake scratch must be removed)
* if ``/tmp`` exists in the image: mode must be ``1777``, and no leftover bake
  artifacts under it (node/agent tarballs, ``pre_commands.sh``). Missing
  ``/tmp`` is OK (runtime creates it).
* pre-baked CLIs exist under ``/usr/local/bin``: node, npm, claude, opencode,
  pi, mini (``cli_prebaked`` = Node + four agents)
* ``metadata.workdir`` is a git work tree on branch ``scaleswe``; porcelain
  entries matching ScaleSWE ``pre_commands`` keep-list (``*.egg-info``,
  ``.tox``, ``.venv``) and generated ``version.py`` are ignored

Temporary extract dirs under ``/tmp/slime-verify-scaleswe`` are removed after
each job and again on process exit.

Example::

    python3 examples/coding_agent_rl/docker_build/verify_scaleswe_agent_images.py \\
      --limit 1
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

from bake_common import AGENT_ASSET_NAMES, check_prebaked_clis

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR.parent
_VERIFY_TMP_ROOT = Path(os.environ.get("VERIFY_TMP_ROOT", "/tmp/slime-verify-scaleswe"))
_FORBIDDEN_TOP_DIRS = ("deepswe", "scaleswe")
_EXPECTED_BRANCH = "scaleswe"
_EXPECTED_TMP_MODE = 0o1777
_FORBIDDEN_TMP_LEFTOVERS = (*AGENT_ASSET_NAMES, "pre_commands.sh")
# ScaleSWE pre_commands use: git clean -fd -e '*.egg-info' -e '.tox' -e '.venv'
_IGNORED_DIRTY_DIR_NAMES = frozenset({".tox", ".venv", "venv", ".eggs"})
_IGNORED_DIRTY_FILE_NAMES = frozenset({"version.py"})


def _porcelain_path(line: str) -> str:
    """Extract path from a ``git status --porcelain`` line."""
    text = line.rstrip("\n")
    if " -> " in text:
        text = text.split(" -> ", 1)[-1]
    # status is two chars + space; paths may be quoted
    path = text[3:] if len(text) >= 3 else text
    return path.strip().strip('"')


def _is_ignorable_dirty_path(path: str) -> bool:
    """Return True for dirty paths ScaleSWE intentionally retains."""
    if not path:
        return False
    p = Path(path)
    if p.name in _IGNORED_DIRTY_FILE_NAMES:
        return True
    for part in p.parts:
        if part in _IGNORED_DIRTY_DIR_NAMES or part.endswith(".egg-info"):
            return True
    return False


def _meaningful_dirty_lines(porcelain: str) -> list[str]:
    lines = [ln for ln in porcelain.splitlines() if ln.strip()]
    return [ln for ln in lines if not _is_ignorable_dirty_path(_porcelain_path(ln))]


@dataclass(frozen=True)
class VerifyJob:
    instance_id: str
    image: str
    workdir: str


def resolve_crane(explicit: str | None = None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("CRANE", "").strip()
    if env:
        candidates.append(env)
    candidates.extend(["crane", "/usr/local/bin/crane"])
    for c in candidates:
        path = c if "/" in c else shutil.which(c)
        if path and Path(path).is_file() and os.access(path, os.X_OK):
            return path
    raise RuntimeError(
        "crane not found. Install:\n"
        "  curl -fsSL https://github.com/google/go-containerregistry/releases/"
        "download/v0.20.3/go-containerregistry_Linux_x86_64.tar.gz "
        "| tar -xz -C /usr/local/bin crane"
    )


def cleanup_verify_tmp(root: Path = _VERIFY_TMP_ROOT) -> int:
    """Remove leftover verify extract trees. Returns number of paths removed."""
    if not root.exists():
        return 0
    removed = 0
    try:
        for child in list(root.iterdir()):
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
        # Drop empty root if possible.
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
    rem = md.get("remote_env_info") or {}
    iid = md.get("instance_id") or rem.get("instance_id") or row.get("label")
    if not iid:
        raise ValueError("row missing instance_id / label")
    return str(iid)


def _target_image_of(row: dict[str, Any]) -> str:
    """Baked agent image (``metadata.image``), not the original ScaleSWE base."""
    md = row.get("metadata") or {}
    rem = md.get("remote_env_info") or {}
    image = md.get("image") or rem.get("image_url")
    if not image:
        raise ValueError(f"row {_instance_id_of(row)!r} missing metadata.image")
    return str(image)


def _workdir_of(row: dict[str, Any]) -> str:
    md = row.get("metadata") or {}
    rem = md.get("remote_env_info") or {}
    workdir = md.get("workdir") or rem.get("workdir")
    if not workdir:
        raise ValueError(f"row {_instance_id_of(row)!r} missing workdir")
    return str(workdir)


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
            jobs.append(
                VerifyJob(
                    instance_id=_instance_id_of(row),
                    image=image,
                    workdir=_workdir_of(row),
                )
            )
    return jobs


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # crane|tar preserves image UIDs (often ``agent``); verify runs as root, so
    # plain git hits "dubious ownership" and looks like "not a git work tree".
    return subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(cwd), *args],
        check=False,
        capture_output=True,
        text=True,
    )


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
    """Bake must not leave /tmp_build; /tmp must stay agent-writable."""
    tmp_build = root / "tmp_build"
    if tmp_build.exists():
        errors.append("forbidden path present: /tmp_build/ (bake scratch not removed)")

    tmp = root / "tmp"
    if not tmp.is_dir():
        errors.append("missing /tmp directory")
        return

    mode = tmp.stat().st_mode & 0o7777
    if mode != _EXPECTED_TMP_MODE:
        errors.append(f"/tmp mode is {mode:04o}, expected {_EXPECTED_TMP_MODE:04o} (sticky+world-writable)")

    for name in _FORBIDDEN_TMP_LEFTOVERS:
        leftover = tmp / name
        if leftover.exists():
            errors.append(f"forbidden bake leftover: /tmp/{name}")


def verify_extracted(job: VerifyJob, root: Path) -> dict[str, Any]:
    """Run checks on an already-exported rootfs. Returns a result dict."""
    errors: list[str] = []
    head = ""
    branch = ""

    passwd = root / "etc" / "passwd"
    if not passwd.is_file():
        errors.append("missing etc/passwd")
    elif passwd.stat().st_size == 0:
        errors.append("etc/passwd is empty")

    for name in _FORBIDDEN_TOP_DIRS:
        bad = root / name
        if bad.exists():
            errors.append(f"forbidden path present: /{name}/")

    _check_tmp_layout(root, errors)
    check_prebaked_clis(root, errors)

    rel = job.workdir.lstrip("/")
    if not rel:
        errors.append(f"invalid workdir: {job.workdir!r}")
        return {
            "instance_id": job.instance_id,
            "image": job.image,
            "workdir": job.workdir,
            "ok": False,
            "errors": errors,
            "head": head,
            "branch": branch,
        }

    wd = root / rel
    if not wd.is_dir():
        errors.append(f"missing workdir directory: {job.workdir}")
    else:
        inside = _git(wd, "rev-parse", "--is-inside-work-tree")
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            errors.append(f"not a git work tree: {job.workdir}")
        else:
            br = _git(wd, "branch", "--show-current")
            branch = (br.stdout or "").strip()
            if br.returncode != 0:
                errors.append(f"git branch --show-current failed: {(br.stderr or '').strip()}")
            elif branch != _EXPECTED_BRANCH:
                errors.append(f"expected branch {_EXPECTED_BRANCH!r}, got {branch!r}")

            st = _git(wd, "status", "--porcelain")
            if st.returncode != 0:
                errors.append(f"git status failed: {(st.stderr or '').strip()}")
            else:
                dirty = _meaningful_dirty_lines(st.stdout or "")
                if dirty:
                    preview = "; ".join(dirty[:5])
                    more = f" (+{len(dirty) - 5} more)" if len(dirty) > 5 else ""
                    errors.append(f"working tree dirty: {preview}{more}")

            hd = _git(wd, "rev-parse", "HEAD")
            if hd.returncode == 0:
                head = (hd.stdout or "").strip()
            else:
                errors.append(f"git rev-parse HEAD failed: {(hd.stderr or '').strip()}")

    return {
        "instance_id": job.instance_id,
        "image": job.image,
        "workdir": job.workdir,
        "ok": not errors,
        "errors": errors,
        "head": head,
        "branch": branch,
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
            "workdir": job.workdir,
            "ok": False,
            "errors": [f"{type(e).__name__}: {e}"],
            "head": "",
            "branch": "",
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=_EXAMPLE_DIR / "data" / "swe_train_scaleswe_200_baked.jsonl",
    )
    p.add_argument("--limit", type=int, default=0, help="Max unique images (0=all)")
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("VERIFY_WORKERS", "4")),
        help="Parallel crane exports (also VERIFY_WORKERS). Watch /tmp quota.",
    )
    p.add_argument(
        "--failures-out",
        type=Path,
        default=_SCRIPT_DIR / "verify_failures.jsonl",
    )
    p.add_argument(
        "--crane",
        default=os.environ.get("CRANE", ""),
        help="Path to crane binary (or set CRANE)",
    )
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

    results: list[dict[str, Any]] = []

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
            print(
                f"[verify] OK {r['instance_id']} branch={r.get('branch')} "
                f"HEAD={(r.get('head') or '')[:12]}",
                flush=True,
            )
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
