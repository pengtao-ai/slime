#!/usr/bin/env python3
"""Bake Tmax agent images with Node22 + claude/opencode/pi/miniswe.

No workspace restore and no pre_commands. Verifier (``test_sh``) stays out of
the image. Output tags: ``pyrominddynamics/tmax-agent:<base-image-tag>``.

Example::

    export DOCKERHUB_USERNAME=...
    export DOCKERHUB_TOKEN=...
    python examples/coding_agent_rl/docker_build/bake_tmax_agent_images.py \\
      --input examples/coding_agent_rl/data/mixed_agents_bake_smoke_tmax.jsonl \\
      --output examples/coding_agent_rl/data/mixed_agents_bake_smoke_tmax_baked.jsonl \\
      --limit 1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
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
    resolve_kaniko_executor,
    should_skip_existing_remote_tag,
    target_ref,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR.parent
_TEMPLATE = _SCRIPT_DIR / "Dockerfile.tmax.template"
_SAFE_TAG_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class BakeJob:
    instance_id: str
    base_image: str
    image_tag: str
    source_row_indices: tuple[int, ...]


def _instance_id_of(row: dict[str, Any]) -> str:
    md = row.get("metadata") or {}
    iid = md.get("instance_id") or row.get("label")
    if not iid:
        raise ValueError("row missing instance_id / label")
    return str(iid)


def _base_image_of(row: dict[str, Any]) -> str:
    md = row.get("metadata") or {}
    image = md.get("image") or md.get("docker_image")
    if not image:
        raise ValueError(f"row {_instance_id_of(row)!r} missing metadata.image")
    return str(image)


def _image_tag_of(base_image: str) -> str:
    if ":" not in base_image:
        raise ValueError(f"base image missing tag: {base_image!r}")
    tag = base_image.rsplit(":", 1)[-1].strip()
    if not tag:
        raise ValueError(f"base image empty tag: {base_image!r}")
    return tag


def _dockerfile_stem(image_tag: str) -> str:
    safe = _SAFE_TAG_RE.sub("_", image_tag)
    return safe or "tmax"


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
                    image_tag=prev.image_tag,
                    source_row_indices=prev.source_row_indices + (idx,),
                )
                continue
            by_image[image] = BakeJob(
                instance_id=iid,
                base_image=image,
                image_tag=_image_tag_of(image),
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
) -> Path:
    dockerfiles_dir.mkdir(parents=True, exist_ok=True)
    stem = _dockerfile_stem(job.image_tag)
    df_path = dockerfiles_dir / f"{stem}.Dockerfile"
    df_path.write_text(
        render_dockerfile(template, base_image=job.base_image),
        encoding="utf-8",
    )
    return df_path


def baked_row(row: dict[str, Any], *, tag_prefix: str) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    md = out.setdefault("metadata", {})
    original = _base_image_of(row)
    tag = _image_tag_of(original)
    md["docker_image"] = original
    md["image"] = f"{tag_prefix.rstrip('/')}:{tag}"
    md["cli_prebaked"] = True
    md.setdefault("protocol", "tmax")
    md.setdefault("workdir", "/home/user")
    # Keep test_sh / problem_statement untouched.
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
    df_path = write_instance_files(job, dockerfiles_dir=dockerfiles_dir, template=template)
    ref = target_ref(tag_prefix, job.image_tag)
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
        executor = resolve_kaniko_executor(kaniko_executor)
        kaniko_build(
            tag=job.image_tag,
            dockerfile_name=df_path.name,
            context_dir=context_dir,
            dockerfiles_dir=dockerfiles_dir,
            tag_prefix=tag_prefix,
            push=push,
            executor=executor,
            restored_instance_id=None,
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
        default=_EXAMPLE_DIR / "data" / "tmax_train_200.jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_EXAMPLE_DIR / "data" / "tmax_train_200_baked.jsonl",
    )
    p.add_argument("--tag-prefix", default="pyrominddynamics/tmax-agent")
    p.add_argument("--dockerfiles-dir", type=Path, default=_SCRIPT_DIR / "dockerfiles_tmax")
    p.add_argument("--assets-dir", type=Path, default=_SCRIPT_DIR / "assets")
    p.add_argument(
        "--context-dir",
        type=Path,
        default=_SCRIPT_DIR,
        help="Kaniko build context root (must contain assets/ + dockerfiles_tmax/)",
    )
    add_agent_asset_args(p)
    p.add_argument("--limit", type=int, default=0, help="Max unique images (0=all)")
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("BAKE_WORKERS", "1")),
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--skip-existing-max-age-hours",
        type=float,
        default=None,
        help="With --skip-existing: only skip Hub tags newer than this many hours; "
        "older tags are rebuilt. Default: skip any existing tag.",
    )
    p.add_argument("--no-push", action="store_true")
    p.add_argument(
        "--kaniko-executor",
        default=os.environ.get("KANIKO_EXECUTOR", ""),
    )
    p.add_argument("--generate-only", action="store_true")
    p.add_argument(
        "--failures",
        type=Path,
        default=_SCRIPT_DIR / "bake_tmax_failures.jsonl",
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
                f"[bake] OK {job.instance_id} tag={job.image_tag} "
                f"skipped={info.get('skipped')} generate_only={info.get('generate_only', False)}",
                flush=True,
            )
        else:
            failures.append(
                {
                    "instance_id": job.instance_id,
                    "base_image": job.base_image,
                    "image_tag": job.image_tag,
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
