#!/usr/bin/env python3
"""Bake ScaleSWE agent images with Node22 + Claude Code + full pre_commands.

Builds and pushes with **Kaniko only** (no ``docker`` CLI, no BuildKit /
``buildctl``). Registry auth comes from ``DOCKERHUB_USERNAME`` +
``DOCKERHUB_TOKEN`` written into ``~/.docker/config.json`` for Kaniko
(Kaniko's config path; does not call ``docker login``).

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
import base64
import concurrent.futures
import json
import lzma
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR.parent
_DEFAULT_TARBALLS = _EXAMPLE_DIR / "tarballs"
_DEFAULT_NODE_XZ = _DEFAULT_TARBALLS / "node-v22.20.0-linux-x64.tar.xz"
_DEFAULT_CC_TGZ = _DEFAULT_TARBALLS / "anthropic-ai-claude-code-local-linux-x64.tgz"
_TEMPLATE = _SCRIPT_DIR / "Dockerfile.template"

_hub_public_lock = threading.Lock()
_hub_public_done = False


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


def ensure_assets(*, node_xz: Path, cc_tgz: Path, assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    node_tar = assets_dir / "node22.tar"
    cc_dst = assets_dir / "claude-code-local.tgz"
    if not node_tar.exists():
        if not node_xz.is_file():
            raise SystemExit(f"missing node tarball: {node_xz}")
        tmp = node_tar.with_suffix(".tar.partial")
        with lzma.open(node_xz, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp.replace(node_tar)
        print(f"[bake] prepared {node_tar} ({node_tar.stat().st_size} bytes)", flush=True)
    if not cc_dst.exists() or (cc_tgz.is_file() and cc_dst.stat().st_size != cc_tgz.stat().st_size):
        if not cc_tgz.is_file():
            raise SystemExit(f"missing claude-code tarball: {cc_tgz}")
        shutil.copy2(cc_tgz, cc_dst)
        print(f"[bake] prepared {cc_dst}", flush=True)


def ensure_registry_auth_from_env(*, username: str | None, password: str | None) -> None:
    """Write ~/.docker/config.json for Kaniko. Does not call ``docker login``."""
    if not username or not password:
        return
    auth = base64.b64encode(f"{username}:{password}".encode()).decode()
    cfg_path = Path.home() / ".docker" / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cfg = {}
    auths = cfg.setdefault("auths", {})
    for host in ("https://index.docker.io/v1/", "registry-1.docker.io", "docker.io"):
        auths[host] = {"auth": auth}
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"[bake] wrote registry auth for {username!r} → {cfg_path} (no docker login)", flush=True)


def prepare_filtered_proc(dest: Path) -> Path:
    """Build a fake ``/proc`` without host JuiceFS mounts.

    Kaniko reads ``/proc/self/mountinfo`` and skips snapshotting those mount
    points. On docker-rt, ``/workspace`` is JuiceFS, so a real ``/proc`` would
    drop ScaleSWE repos from the image. Runtime workdir must stay ``/workspace/...``.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "cpuinfo",
        "meminfo",
        "stat",
        "uptime",
        "loadavg",
        "version",
        "filesystems",
        "diskstats",
        "partitions",
    ):
        src = Path("/proc") / name
        if src.is_file():
            try:
                shutil.copyfile(src, dest / name)
            except OSError:
                pass
    mountinfo = (
        "1 0 0:1 / / rw,relatime - overlay overlay rw\n"
        "2 1 0:2 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw\n"
        "3 1 0:3 / /dev rw - tmpfs tmpfs rw\n"
        "4 1 0:4 / /sys rw - sysfs sysfs ro\n"
        "5 1 0:5 / /tmp rw - tmpfs tmpfs rw\n"
    )
    mounts = (
        "overlay / overlay rw 0 0\n"
        "proc /proc proc rw 0 0\n"
        "tmpfs /dev tmpfs rw 0 0\n"
        "sysfs /sys sysfs ro 0 0\n"
        "tmpfs /tmp tmpfs rw 0 0\n"
    )
    (dest / "mountinfo").write_text(mountinfo, encoding="utf-8")
    (dest / "mounts").write_text(mounts, encoding="utf-8")
    self_dir = dest / "self"
    self_dir.mkdir(parents=True, exist_ok=True)
    (self_dir / "mountinfo").write_text(mountinfo, encoding="utf-8")
    (self_dir / "status").write_text("Name:\tkaniko\nPid:\t1\n", encoding="utf-8")
    (self_dir / "stat").write_text("1 (kaniko) R 0 0 0 0 0 0 0 0 0 0\n", encoding="utf-8")
    (self_dir / "cmdline").write_text("kaniko\x00", encoding="utf-8")
    (self_dir / "fd").mkdir(exist_ok=True)
    for i in range(3):
        try:
            (self_dir / "fd" / str(i)).symlink_to("/dev/null")
        except OSError:
            pass
    try:
        (self_dir / "exe").symlink_to("/kaniko/executor")
    except OSError:
        pass
    return dest


def render_dockerfile(template: str, *, base_image: str, workdir: str, instance_id: str) -> str:
    return (
        template.replace("__BASE_IMAGE__", base_image)
        .replace("__WORKDIR__", workdir)
        .replace("__INSTANCE_ID__", instance_id)
    )


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


def target_ref(tag_prefix: str, instance_id: str) -> str:
    prefix = tag_prefix.strip().rstrip("/")
    name = f"{prefix}:{instance_id}"
    if name.startswith("docker.io/"):
        return name
    return f"docker.io/{name}"


def parse_docker_hub_ref(ref: str) -> tuple[str, str, str]:
    """Parse ``docker.io/ns/repo:tag`` → (namespace, repository, tag)."""
    name = ref.strip().removeprefix("docker.io/")
    if ":" not in name or "/" not in name:
        raise ValueError(f"expected docker.io/ns/repo:tag, got {ref!r}")
    repo_path, tag = name.rsplit(":", 1)
    namespace, repository = repo_path.split("/", 1)
    if "/" in repository:
        raise ValueError(f"expected docker.io/ns/repo:tag, got {ref!r}")
    return namespace, repository, tag


def remote_tag_exists(
    ref: str,
    *,
    username: str | None = None,
    password: str | None = None,
) -> bool:
    """Return True if the registry already has ``ref`` (Docker Hub via registry API)."""
    try:
        ns, repo, tag = parse_docker_hub_ref(ref)
    except ValueError:
        return False
    repository = f"{ns}/{repo}"
    auth_url = (
        "https://auth.docker.io/token"
        f"?service=registry.docker.io&scope=repository:{repository}:pull"
    )
    auth_req = urllib.request.Request(auth_url)
    if username and password:
        basic = base64.b64encode(f"{username}:{password}".encode()).decode()
        auth_req.add_header("Authorization", f"Basic {basic}")
    try:
        with urllib.request.urlopen(auth_req, timeout=30) as resp:
            token = json.loads(resp.read().decode()).get("token")
        if not token:
            return False
        man_req = urllib.request.Request(
            f"https://registry-1.docker.io/v2/{repository}/manifests/{urllib.parse.quote(tag, safe='')}",
            method="HEAD",
        )
        man_req.add_header("Authorization", f"Bearer {token}")
        man_req.add_header(
            "Accept",
            ", ".join(
                [
                    "application/vnd.docker.distribution.manifest.v2+json",
                    "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.oci.image.index.v1+json",
                ]
            ),
        )
        with urllib.request.urlopen(man_req, timeout=30) as resp:
            return 200 <= int(resp.status) < 300
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        return False
    except Exception:
        return False


def hub_login_token(username: str, password: str) -> str:
    req = urllib.request.Request(
        "https://hub.docker.com/v2/users/login/",
        data=json.dumps({"username": username, "password": password}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    token = payload.get("token")
    if not token:
        raise RuntimeError("Docker Hub login did not return a token")
    return str(token)


def ensure_hub_repo_public(*, namespace: str, repository: str, username: str, password: str) -> None:
    global _hub_public_done
    with _hub_public_lock:
        if _hub_public_done:
            return
        token = hub_login_token(username, password)
        url = f"https://hub.docker.com/v2/repositories/{namespace}/{repository}/"
        req = urllib.request.Request(
            url,
            data=json.dumps({"is_private": False}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"JWT {token}",
            },
            method="PATCH",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                _ = resp.read()
            print(f"[bake] set {namespace}/{repository} is_private=false", flush=True)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            # Repo may not exist until first push; caller retries after push.
            raise RuntimeError(f"Hub PATCH visibility failed ({e.code}): {body}") from e
        _hub_public_done = True


def parse_tag_prefix(tag_prefix: str) -> tuple[str, str]:
    """Return (namespace, repository) for Hub API."""
    prefix = tag_prefix.strip().removeprefix("docker.io/").rstrip("/")
    if "/" not in prefix:
        raise ValueError(f"tag prefix must be namespace/repo, got {tag_prefix!r}")
    namespace, repository = prefix.split("/", 1)
    if "/" in repository:
        raise ValueError(f"tag prefix must be namespace/repo, got {tag_prefix!r}")
    return namespace, repository


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
        "crane not found (needed to restore /workspace from base images when "
        "the host mounts /workspace). Install:\n"
        "  curl -fsSL https://github.com/google/go-containerregistry/releases/"
        "download/v0.20.3/go-containerregistry_Linux_x86_64.tar.gz "
        "| tar -xz -C /usr/local/bin crane"
    )


def restore_workspace_from_base(
    job: BakeJob,
    *,
    context_dir: Path,
    crane: str | None = None,
) -> Path:
    """Extract ``workspace/`` from the base image into the Kaniko context.

    Kaniko skips unpacking into host mountpoints. On docker-rt, ``/workspace`` is
    often JuiceFS, so the ScaleSWE repo tree never appears unless we COPY it back.
    """
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


def resolve_kaniko_executor(explicit: str | None = None) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("KANIKO_EXECUTOR", "").strip()
    if env:
        candidates.append(env)
    candidates.extend(
        [
            "/opt/kaniko/executor",
            "/kaniko/executor",
            "executor",
            "kaniko-executor",
        ]
    )
    for c in candidates:
        if c in ("executor", "kaniko-executor") and shutil.which(c):
            return c
        if c not in ("executor", "kaniko-executor") and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise RuntimeError(
        "Kaniko executor not found. Install via:\n"
        "  crane export gcr.io/kaniko-project/executor:v1.23.2 | tar -xO kaniko/executor "
        "> /opt/kaniko/executor && chmod +x /opt/kaniko/executor\n"
        "or set KANIKO_EXECUTOR=/path/to/executor"
    )


def resolve_proot(explicit: str | None = None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("PROOT", "").strip()
    if env:
        candidates.append(env)
    candidates.extend(["proot", "/usr/local/bin/proot"])
    for c in candidates:
        path = c if "/" in c else shutil.which(c)
        if path and Path(path).is_file() and os.access(path, os.X_OK):
            return path
    raise RuntimeError(
        "proot not found (needed to isolate Kaniko from JuiceFS /workspace mounts). "
        "Install:\n"
        "  curl -fsSL -o /usr/local/bin/proot "
        "https://github.com/proot-me/proot/releases/download/v5.3.0/proot-v5.3.0-x86_64-static "
        "&& chmod +x /usr/local/bin/proot"
    )


def kaniko_guest_root_base() -> Path:
    return Path(os.environ.get("KANIKO_GUEST_ROOT", "/tmp/slime-kaniko-root"))


def cleanup_kaniko_tmp() -> int:
    """Remove leftover Kaniko guest roots under ``/tmp`` (quota-sensitive)."""
    removed = 0
    targets: list[Path] = []
    base = kaniko_guest_root_base()
    targets.append(base)
    tmp = Path("/tmp")
    if tmp.is_dir():
        targets.extend(sorted(tmp.glob("slime-kaniko*")))

    seen: set[Path] = set()
    for path in targets:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                pass
        if not path.exists():
            removed += 1
    if removed:
        print(f"[bake] cleaned {removed} leftover /tmp kaniko path(s)", flush=True)
    return removed


def ensure_kaniko_guest_root(guest_root: Path) -> Path:
    """Prepare a minimal guest root under ``/tmp`` for proot+Kaniko.

    Build context (assets / dockerfiles / restored) is bind-mounted via proot so
    we do not copy hundreds of MB into the quota-limited ``/tmp``.
    """
    guest_root.mkdir(parents=True, exist_ok=True)
    for rel in (
        "kaniko",
        "build/assets",
        "build/dockerfiles",
        "build/restored",
        "build/kaniko_out",
        "root/.docker",
        "etc/ssl/certs",
        "dev",
        "tmp",
        "proc",
    ):
        (guest_root / rel).mkdir(parents=True, exist_ok=True)
    return guest_root


def _validate_build_context(
    *,
    context_dir: Path,
    dockerfiles_dir: Path,
    job: BakeJob,
) -> tuple[Path, Path, Path]:
    assets_src = context_dir / "assets"
    for name in ("node22.tar", "claude-code-local.tgz"):
        src = assets_src / name
        if not src.is_file():
            raise FileNotFoundError(src)
    for name in (f"{job.instance_id}.Dockerfile", f"{job.instance_id}.pre.sh"):
        src = dockerfiles_dir / name
        if not src.is_file():
            raise FileNotFoundError(src)
    restored_src = context_dir / "restored" / job.instance_id
    if not (restored_src / "workspace").is_dir():
        raise FileNotFoundError(restored_src / "workspace")
    return assets_src, dockerfiles_dir, restored_src


def kaniko_build(
    job: BakeJob,
    *,
    context_dir: Path,
    dockerfiles_dir: Path,
    tag_prefix: str,
    push: bool,
    executor: str,
    tar_path: Path | None = None,
    guest_root: Path | None = None,
    proot: str | None = None,
) -> None:
    """Build with Kaniko inside a proot guest (no Docker daemon / BuildKit).

    Running the executor directly on docker-rt mutates the host root and collides
    with JuiceFS mounts at ``/workspace``. proot gives Kaniko an isolated rootfs
    under ``/tmp`` while still providing a real ``/proc`` (needed by Claude Code).

    Context dirs are bind-mounted (not copied) to keep ``/tmp`` quota small.
    """
    ref = target_ref(tag_prefix, job.instance_id)
    dockerfile = dockerfiles_dir / f"{job.instance_id}.Dockerfile"
    if not dockerfile.is_file():
        raise FileNotFoundError(dockerfile)

    proot_bin = resolve_proot(proot)
    assets_src, dockerfiles_src, restored_src = _validate_build_context(
        context_dir=context_dir,
        dockerfiles_dir=dockerfiles_dir,
        job=job,
    )

    # Per-job guest root so --workers > 1 do not wipe each other.
    root_base = kaniko_guest_root_base()
    root = guest_root or (root_base / job.instance_id)
    if root.exists():
        shutil.rmtree(root)
    ensure_kaniko_guest_root(root)
    # Mount point for this instance's restored tree.
    (root / "build" / "restored" / job.instance_id).mkdir(parents=True, exist_ok=True)
    # Filtered /proc so Kaniko will snapshot /workspace (not treat host JuiceFS as a mount).
    fake_proc = root / "fake_proc"
    prepare_filtered_proc(fake_proc)

    docker_cfg = Path.home() / ".docker" / "config.json"
    if docker_cfg.is_file():
        # Tiny file; copy so proot always sees a regular file path.
        shutil.copy2(docker_cfg, root / "root" / ".docker" / "config.json")

    # Regular-file copies (not bind mounts) so DNS/TLS work before/without unpacking
    # over busy host kubelet mounts.
    etc = root / "etc"
    etc.mkdir(parents=True, exist_ok=True)
    if Path("/etc/resolv.conf").is_file():
        shutil.copy2("/etc/resolv.conf", etc / "resolv.conf")
    certs_src = Path("/etc/ssl/certs")
    if certs_src.is_dir():
        certs_dst = etc / "ssl" / "certs"
        if certs_dst.exists():
            shutil.rmtree(certs_dst)
        shutil.copytree(certs_src, certs_dst, symlinks=True)

    out_host: Path | None = None
    if not push:
        if tar_path is None:
            tar_path = context_dir / "kaniko_out" / f"{job.instance_id}.tar"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        out_host = tar_path.parent

    try:
        kaniko_cmd = [
            "/kaniko/executor",
            "--force",
            f"--dockerfile=/build/dockerfiles/{job.instance_id}.Dockerfile",
            "--context=dir:///build",
            "--verbosity=info",
            "--ignore-path=/build",
            "--ignore-path=/kaniko",
            "--ignore-path=/fake_proc",
            # Host kubelet bind-mounts; with filtered mountinfo Kaniko no longer
            # auto-skips them and would fail with "device or resource busy".
            "--ignore-path=/etc/resolv.conf",
            "--ignore-path=/etc/hosts",
            "--ignore-path=/etc/hostname",
        ]
        if push:
            kaniko_cmd.append(f"--destination={ref}")
        else:
            assert tar_path is not None
            kaniko_cmd.append(f"--tarPath=/build/kaniko_out/{tar_path.name}")
            kaniko_cmd.append("--no-push")
            kaniko_cmd.append(f"--destination={ref}")

        binds = [
            f"{fake_proc.resolve()}:/proc",
            "/dev",
            f"{assets_src.resolve()}:/build/assets",
            f"{dockerfiles_src.resolve()}:/build/dockerfiles",
            f"{restored_src.resolve()}:/build/restored/{job.instance_id}",
            f"{Path(executor).resolve()}:/kaniko/executor",
        ]
        # Bind host DNS/TLS paths for registry access. Paired with --ignore-path
        # above so unpack does not try to unlink these busy kubelet mounts.
        if Path("/etc/resolv.conf").is_file():
            binds.append("/etc/resolv.conf:/etc/resolv.conf")
        if Path("/etc/ssl/certs").is_dir():
            binds.append("/etc/ssl/certs:/etc/ssl/certs")
        if out_host is not None:
            binds.append(f"{out_host.resolve()}:/build/kaniko_out")

        cmd = [proot_bin, "-0", "-r", str(root), "-w", "/"]
        for b in binds:
            cmd.extend(["-b", b])
        cmd.extend(kaniko_cmd)

        print(f"[bake] kaniko(proot) → {ref}", flush=True)
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=os.environ.copy())
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError(f"kaniko failed for {job.instance_id}:\n{log[-4000:]}")
        if not push and tar_path is not None and not tar_path.is_file():
            raise RuntimeError(f"kaniko --no-push did not write {tar_path}")
        if log.strip():
            print(log[-800:], flush=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        # Drop empty parent if last worker finished.
        try:
            if root_base.is_dir() and not any(root_base.iterdir()):
                root_base.rmdir()
        except OSError:
            pass


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
    generate_only: bool,
    kaniko_executor: str | None,
    hub_user: str | None,
    hub_pass: str | None,
) -> dict[str, Any]:
    write_instance_files(job, dockerfiles_dir=dockerfiles_dir, template=template)
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
    if skip_existing and remote_tag_exists(ref, username=hub_user, password=hub_pass):
        print(f"[bake] skip-existing {ref}", flush=True)
        skipped = True
    else:
        restore_workspace_from_base(job, context_dir=context_dir)
        executor = resolve_kaniko_executor(kaniko_executor)
        kaniko_build(
            job,
            context_dir=context_dir,
            dockerfiles_dir=dockerfiles_dir,
            tag_prefix=tag_prefix,
            push=push,
            executor=executor,
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
    p.add_argument("--node-xz", type=Path, default=_DEFAULT_NODE_XZ)
    p.add_argument("--cc-tgz", type=Path, default=_DEFAULT_CC_TGZ)
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
        ensure_assets(node_xz=args.node_xz, cc_tgz=args.cc_tgz, assets_dir=args.assets_dir)
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
