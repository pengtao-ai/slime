#!/usr/bin/env python3
"""Bake ScaleSWE agent images with Node22 + claude/opencode/pi/miniswe + pre_commands.

Builds and pushes with **Kaniko only** (no ``docker`` CLI, no BuildKit /
``buildctl``). Registry auth comes from ``DOCKERHUB_USERNAME`` +
``DOCKERHUB_TOKEN`` written into ``~/.docker/config.json`` for Kaniko
(Kaniko's config path; does not call ``docker login``).

``cli_prebaked=True`` means Node + claude / opencode / pi / mini are in the image.

Install the executor once::

    mkdir -p /opt/kaniko
    crane export gcr.io/kaniko-project/executor:v1.23.2 \\
      | tar -xO kaniko/executor > /opt/kaniko/executor
    chmod +x /opt/kaniko/executor

Example::

    export DOCKERHUB_USERNAME=...
    export DOCKERHUB_TOKEN=...
    python examples/coding_agent_rl/docker_build/bake_scaleswe_agent_images.py \\
      --input examples/coding_agent_rl/data/swe_train_scaleswe_200.jsonl \\
      --output examples/coding_agent_rl/data/swe_train_scaleswe_200_baked.jsonl \\
      --limit 1 --skip-existing
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bake_common import (
    add_agent_asset_args,
    cleanup_kaniko_tmp,
    ensure_assets_from_args,
    ensure_hub_repo_public,
    ensure_registry_auth_from_env,
    kaniko_build,
    parse_tag_prefix,
    render_dockerfile,
    resolve_crane,
    resolve_kaniko_executor,
    should_skip_existing_remote_tag,
    target_ref,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR.parent
_TEMPLATE = _SCRIPT_DIR / "Dockerfile.template"


@dataclass(frozen=True)
class BakeJob:
    instance_id: str
    base_image: str
    workdir: str
    pre_commands: str
    source_row_indices: tuple[int, ...]


def _normalize_pre_commands(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        body = "\n".join(str(c) for c in raw if c)
    else:
        body = str(raw).replace("\\n", "\n")
    return body.strip() + ("\n" if body.strip() else "")


def _instance_id_of(row: dict[str, Any]) -> str:
    md = row.get("metadata") or {}
    rem = md.get("remote_env_info") or {}
    iid = md.get("instance_id") or rem.get("instance_id") or row.get("label")
    if not iid:
        raise ValueError("row missing instance_id / label")
    return str(iid)


def _base_image_of(row: dict[str, Any]) -> str:
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


def _pre_commands_of(row: dict[str, Any]) -> str:
    md = row.get("metadata") or {}
    rem = md.get("remote_env_info") or {}
    return _normalize_pre_commands(md.get("pre_commands") or rem.get("pre_commands"))


def load_jobs(path: Path) -> tuple[list[dict[str, Any]], list[BakeJob]]:
    rows: list[dict[str, Any]] = []
    by_image: dict[str, BakeJob] = {}
    order: list[str] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            image = _base_image_of(row)
            iid = _instance_id_of(row)
            if image in by_image:
                prev = by_image[image]
                by_image[image] = BakeJob(
                    instance_id=prev.instance_id,
                    base_image=prev.base_image,
                    workdir=prev.workdir,
                    pre_commands=prev.pre_commands,
                    source_row_indices=prev.source_row_indices + (idx,),
                )
                continue
            by_image[image] = BakeJob(
                instance_id=iid,
                base_image=image,
                workdir=_workdir_of(row),
                pre_commands=_pre_commands_of(row),
                source_row_indices=(idx,),
            )
            order.append(image)
    jobs = [by_image[k] for k in order]
    return rows, jobs


def write_instance_files(
    job: BakeJob,
    *,
    dockerfiles_dir: Path,
    template: str,
) -> tuple[Path, Path]:
    dockerfiles_dir.mkdir(parents=True, exist_ok=True)
    df_path = dockerfiles_dir / f"{job.instance_id}.Dockerfile"
    pre_path = dockerfiles_dir / f"{job.instance_id}.pre.sh"
    pre_body = "set -euo pipefail\n" + (job.pre_commands or "true\n")
    pre_path.write_text(pre_body, encoding="utf-8")
    df_path.write_text(
        render_dockerfile(
            template,
            base_image=job.base_image,
            workdir=job.workdir,
            instance_id=job.instance_id,
        ),
        encoding="utf-8",
    )
    return df_path, pre_path


def restore_workspace_from_base(
    job: BakeJob,
    *,
    context_dir: Path,
    crane: str | None = None,
) -> Path:
    """Extract ``workspace/`` from the base image into the Kaniko context."""
    dest = context_dir / "restored" / job.instance_id
    marker = dest / ".restored_from"
    workspace_dir = dest / "workspace"
    if (
        workspace_dir.is_dir()
        and marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == job.base_image
    ):
        return dest

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    crane_bin = resolve_crane(crane)
    print(f"[bake] restoring workspace/ from {job.base_image} → {dest}", flush=True)
    export = subprocess.Popen(
        [crane_bin, "export", job.base_image],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert export.stdout is not None
    extract = subprocess.Popen(
        ["tar", "-x", "-C", str(dest), "workspace"],
        stdin=export.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    export.stdout.close()
    _exp_out, exp_err = export.communicate()
    _tar_out, tar_err = extract.communicate()
    if export.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(
            f"crane export failed for {job.base_image}:\n{(exp_err or b'').decode(errors='replace')[-1000:]}"
        )
    if extract.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(
            f"failed extracting workspace/ from {job.base_image} "
            f"(does the image contain /workspace?):\n{(tar_err or b'').decode(errors='replace')[-1000:]}"
        )
    if not workspace_dir.is_dir():
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(f"no workspace/ directory in {job.base_image}")
    marker.write_text(job.base_image + "\n", encoding="utf-8")
    return dest


def baked_row(row: dict[str, Any], *, tag_prefix: str) -> dict[str, Any]:
    out = json.loads(json.dumps(row))  # deep copy via JSON
    md = out.setdefault("metadata", {})
    rem = md.setdefault("remote_env_info", {}) if isinstance(md.get("remote_env_info"), dict) else {}
    if "remote_env_info" not in md or not isinstance(md.get("remote_env_info"), dict):
        md["remote_env_info"] = rem
    else:
        rem = md["remote_env_info"]

    iid = _instance_id_of(row)
    original = _base_image_of(row)
    original_pre = _pre_commands_of(row)

    md["docker_image"] = original
    md["image"] = f"{tag_prefix.rstrip('/')}:{iid}"
    md["workdir"] = _workdir_of(row)
    md["cli_prebaked"] = True
    md["pre_commands_prebaked"] = True
    if original_pre:
        md["pre_commands_original"] = original_pre
    md["pre_commands"] = ""
    rem["image_url"] = md["image"]
    rem["workdir"] = md["workdir"]
    rem["pre_commands"] = ""
    if original_pre:
        rem["pre_commands_original"] = original_pre
    return out


def process_job(
    job: BakeJob,
    *,
    context_dir: Path,
    dockerfiles_dir: Path,
    template: str,
    tag_prefix: str,
    push: bool,
    skip_existing: bool,
    skip_existing_max_age_hours: float | None,
    generate_only: bool,
    kaniko_executor: str | None,
    hub_user: str | None,
    hub_pass: str | None,
) -> dict[str, Any]:
    write_instance_files(job, dockerfiles_dir=dockerfiles_dir, template=template)
    # pre.sh must exist for ScaleSWE Dockerfile COPY
    pre_path = dockerfiles_dir / f"{job.instance_id}.pre.sh"
    if not pre_path.is_file():
        raise FileNotFoundError(pre_path)
    ref = target_ref(tag_prefix, job.instance_id)
    if generate_only:
        return {
            "instance_id": job.instance_id,
            "base_image": job.base_image,
            "target": ref,
            "skipped": True,
            "generate_only": True,
            "ok": True,
        }

    skipped = False
    if skip_existing and should_skip_existing_remote_tag(
        ref,
        username=hub_user,
        password=hub_pass,
        max_age_hours=skip_existing_max_age_hours,
    ):
        skipped = True
    else:
        restore_workspace_from_base(job, context_dir=context_dir)
        executor = resolve_kaniko_executor(kaniko_executor)
        kaniko_build(
            tag=job.instance_id,
            dockerfile_name=f"{job.instance_id}.Dockerfile",
            context_dir=context_dir,
            dockerfiles_dir=dockerfiles_dir,
            tag_prefix=tag_prefix,
            push=push,
            executor=executor,
            restored_instance_id=job.instance_id,
        )
        if push and hub_user and hub_pass:
            ns, repo = parse_tag_prefix(tag_prefix)
            try:
                ensure_hub_repo_public(
                    namespace=ns, repository=repo, username=hub_user, password=hub_pass
                )
            except Exception as e:
                print(f"[bake] hub visibility pending ({e}); retrying once", flush=True)
                ensure_hub_repo_public(
                    namespace=ns, repository=repo, username=hub_user, password=hub_pass
                )
    return {
        "instance_id": job.instance_id,
        "base_image": job.base_image,
        "target": ref,
        "skipped": skipped,
        "backend": "kaniko",
        "ok": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=_EXAMPLE_DIR / "data" / "swe_train_scaleswe_200.jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_EXAMPLE_DIR / "data" / "swe_train_scaleswe_200_baked.jsonl",
    )
    p.add_argument("--tag-prefix", default="pyrominddynamics/scaleswe-agent")
    p.add_argument("--dockerfiles-dir", type=Path, default=_SCRIPT_DIR / "dockerfiles")
    p.add_argument("--assets-dir", type=Path, default=_SCRIPT_DIR / "assets")
    p.add_argument(
        "--context-dir",
        type=Path,
        default=_SCRIPT_DIR,
        help="Kaniko build context root (must contain assets/ + dockerfiles/)",
    )
    add_agent_asset_args(p)
    p.add_argument("--limit", type=int, default=0, help="Max unique images (0=all)")
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("BAKE_WORKERS", "1")),
        help="Parallel Kaniko builds (each job gets its own /tmp guest root). "
        "Also set via BAKE_WORKERS. Watch disk/CPU; ~2-4 is usually enough.",
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--skip-existing-max-age-hours",
        type=float,
        default=None,
        help="With --skip-existing: only skip Hub tags newer than this many hours; "
        "older tags are rebuilt. Default: skip any existing tag.",
    )
    p.add_argument(
        "--no-push",
        action="store_true",
        help="Do not push; write image tarball under <context>/kaniko_out/",
    )
    p.add_argument(
        "--kaniko-executor",
        default=os.environ.get("KANIKO_EXECUTOR", ""),
        help="Path to kaniko executor binary (or set KANIKO_EXECUTOR)",
    )
    p.add_argument(
        "--generate-only",
        action="store_true",
        help="Only write dockerfiles + baked JSONL; no image build",
    )
    p.add_argument(
        "--failures",
        type=Path,
        default=_SCRIPT_DIR / "bake_failures.jsonl",
    )
    p.add_argument(
        "--hub-username",
        default=os.environ.get("DOCKERHUB_USERNAME") or os.environ.get("DOCKER_USERNAME") or "",
    )
    p.add_argument(
        "--hub-password",
        default=os.environ.get("DOCKERHUB_TOKEN")
        or os.environ.get("DOCKERHUB_PASSWORD")
        or os.environ.get("DOCKER_PASSWORD")
        or "",
    )
    args = p.parse_args()

    if not args.generate_only:
        resolve_kaniko_executor(args.kaniko_executor or None)
        cleanup_kaniko_tmp()
    if not _TEMPLATE.is_file():
        raise SystemExit(f"missing template: {_TEMPLATE}")

    hub_user = args.hub_username.strip() or None
    hub_pass = args.hub_password.strip() or None
    ensure_registry_auth_from_env(username=hub_user, password=hub_pass)

    if not args.generate_only:
        ensure_assets_from_args(args, assets_dir=args.assets_dir)
    template = _TEMPLATE.read_text(encoding="utf-8")
    rows, jobs = load_jobs(args.input)
    print(f"[bake] loaded {len(rows)} rows → {len(jobs)} unique images from {args.input}", flush=True)
    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]
        print(f"[bake] --limit {args.limit} → {len(jobs)} images", flush=True)

    push = not args.no_push
    if push and not args.generate_only and (not hub_user or not hub_pass):
        print(
            "[bake] WARNING: no DOCKERHUB_USERNAME/DOCKERHUB_TOKEN; "
            "Kaniko needs registry auth in ~/.docker/config.json to push",
            flush=True,
        )

    failures: list[dict[str, Any]] = []

    def _run(job: BakeJob) -> tuple[BakeJob, dict[str, Any] | None, str | None]:
        try:
            info = process_job(
                job,
                context_dir=args.context_dir,
                dockerfiles_dir=args.dockerfiles_dir,
                template=template,
                tag_prefix=args.tag_prefix,
                push=push,
                skip_existing=args.skip_existing,
                skip_existing_max_age_hours=args.skip_existing_max_age_hours,
                generate_only=args.generate_only,
                kaniko_executor=args.kaniko_executor or None,
                hub_user=hub_user,
                hub_pass=hub_pass,
            )
            return job, info, None
        except Exception as e:
            return job, None, f"{type(e).__name__}: {e}"

    try:
        workers = max(1, int(args.workers))
        if workers == 1:
            outcomes = [_run(j) for j in jobs]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                outcomes = list(ex.map(_run, jobs))
    finally:
        if not args.generate_only:
            cleanup_kaniko_tmp()

    for job, info, err in outcomes:
        if err is None and info is not None:
            print(
                f"[bake] OK {job.instance_id} skipped={info.get('skipped')} "
                f"generate_only={info.get('generate_only', False)}",
                flush=True,
            )
        else:
            failures.append(
                {
                    "instance_id": job.instance_id,
                    "base_image": job.base_image,
                    "error": err,
                }
            )
            print(f"[bake] FAIL {job.instance_id}: {err}", flush=True)

    built_images = {j.base_image for j, info, err in outcomes if err is None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for row in rows:
            image = _base_image_of(row)
            if image in built_images:
                fout.write(json.dumps(baked_row(row, tag_prefix=args.tag_prefix), ensure_ascii=False) + "\n")
                n_out += 1

    args.failures.parent.mkdir(parents=True, exist_ok=True)
    with args.failures.open("w", encoding="utf-8") as ff:
        for item in failures:
            ff.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"[bake] wrote {n_out} baked rows → {args.output}; "
        f"failures={len(failures)} → {args.failures}; "
        f"dockerfiles → {args.dockerfiles_dir}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
