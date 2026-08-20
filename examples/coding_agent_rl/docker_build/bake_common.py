"""Shared helpers for ScaleSWE / Tmax agent image bake (Kaniko + proot)."""

from __future__ import annotations

import base64
import json
import lzma
import os
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR.parent
DEFAULT_TARBALLS = _EXAMPLE_DIR / "tarballs"
DEFAULT_NODE_XZ = DEFAULT_TARBALLS / "node-v22.20.0-linux-x64.tar.xz"
DEFAULT_CC_TGZ = DEFAULT_TARBALLS / "anthropic-ai-claude-code-local-linux-x64.tgz"
DEFAULT_OPENCODE_TGZ = DEFAULT_TARBALLS / "opencode-ai-local-linux-x64.tgz"
DEFAULT_PI_TGZ = DEFAULT_TARBALLS / "pi-coding-agent-local.tgz"
DEFAULT_MINISWE_WHEELS_TAR = DEFAULT_TARBALLS / "miniswe-wheels.tar"

# Filenames under assets/ (build context) and leftover names to reject in verify.
AGENT_ASSET_NAMES = (
    "node22.tar",
    "claude-code-local.tgz",
    "opencode-ai-local.tgz",
    "pi-coding-agent-local.tgz",
    "miniswe-wheels.tar",
)
PREBAKED_CLI_BINS = (
    "node",
    "npm",
    "claude",
    "opencode",
    "pi",
    "mini",
)

_hub_public_lock = threading.Lock()
_hub_public_done: set[str] = set()


def ensure_assets(
    *,
    node_xz: Path,
    cc_tgz: Path,
    opencode_tgz: Path,
    pi_tgz: Path,
    miniswe_wheels_tar: Path,
    assets_dir: Path,
) -> None:
    """Sync Node + four agent packages into the Kaniko assets/ directory."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    node_tar = assets_dir / "node22.tar"
    if not node_tar.exists():
        if not node_xz.is_file():
            raise SystemExit(f"missing node tarball: {node_xz}")
        tmp = node_tar.with_suffix(".tar.partial")
        with lzma.open(node_xz, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp.replace(node_tar)
        print(f"[bake] prepared {node_tar} ({node_tar.stat().st_size} bytes)", flush=True)

    copies = (
        (cc_tgz, assets_dir / "claude-code-local.tgz", "claude-code"),
        (opencode_tgz, assets_dir / "opencode-ai-local.tgz", "opencode"),
        (pi_tgz, assets_dir / "pi-coding-agent-local.tgz", "pi"),
        (miniswe_wheels_tar, assets_dir / "miniswe-wheels.tar", "miniswe-wheels"),
    )
    for src, dst, label in copies:
        if not dst.exists() or (src.is_file() and dst.stat().st_size != src.stat().st_size):
            if not src.is_file():
                raise SystemExit(f"missing {label} tarball: {src}")
            shutil.copy2(src, dst)
            print(f"[bake] prepared {dst}", flush=True)


def _registry_auth_payload(*, username: str, password: str) -> dict[str, Any]:
    auth = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {
        "auths": {
            host: {"auth": auth}
            for host in ("https://index.docker.io/v1/", "registry-1.docker.io", "docker.io")
        }
    }


def write_registry_auth_file(dest: Path, *, username: str, password: str) -> None:
    """Write a Kaniko-readable Docker config to ``dest`` (does not call ``docker login``)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(_registry_auth_payload(username=username, password=password), indent=2) + "\n", encoding="utf-8")


def hub_credentials_from_env() -> tuple[str | None, str | None]:
    username = (
        os.environ.get("DOCKERHUB_USERNAME") or os.environ.get("DOCKER_USERNAME") or ""
    ).strip() or None
    password = (
        os.environ.get("DOCKERHUB_TOKEN")
        or os.environ.get("DOCKERHUB_PASSWORD")
        or os.environ.get("DOCKER_PASSWORD")
        or ""
    ).strip() or None
    return username, password


def ensure_registry_auth_from_env(*, username: str | None, password: str | None) -> None:
    """Write ~/.docker/config.json for Kaniko. Does not call ``docker login``."""
    if not username or not password:
        return
    cfg_path = Path.home() / ".docker" / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cfg = {}
    auths = cfg.setdefault("auths", {})
    auths.update(_registry_auth_payload(username=username, password=password)["auths"])
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


def render_dockerfile(template: str, *, base_image: str, workdir: str = "", instance_id: str = "") -> str:
    return (
        template.replace("__BASE_IMAGE__", base_image)
        .replace("__WORKDIR__", workdir)
        .replace("__INSTANCE_ID__", instance_id)
    )


def target_ref(tag_prefix: str, tag: str) -> str:
    prefix = tag_prefix.strip().rstrip("/")
    name = f"{prefix}:{tag}"
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


def _parse_hub_datetime(value: str) -> datetime:
    """Parse Hub ``last_updated`` (fractional seconds length varies)."""
    text = value.replace("Z", "+00:00")
    m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?([+-]\d{2}:\d{2})", text)
    if not m:
        raise ValueError(f"unrecognized Hub datetime: {value!r}")
    base, frac, off = m.group(1), m.group(2) or "", m.group(3)
    if frac:
        frac = (frac + "000000")[:7]  # '.' + 6 digits
    return datetime.fromisoformat(base + frac + off)


_hub_tag_mtime_lock = threading.Lock()
_hub_tag_mtime_cache: dict[str, datetime | None] = {}
_hub_jwt_lock = threading.Lock()
_hub_jwt_cache: dict[str, str] = {}


def _hub_jwt(username: str, password: str) -> str:
    key = f"{username}:{password}"
    with _hub_jwt_lock:
        cached = _hub_jwt_cache.get(key)
        if cached:
            return cached
    token = hub_login_token(username, password)
    with _hub_jwt_lock:
        _hub_jwt_cache[key] = token
    return token


def hub_tag_last_updated(
    ref: str,
    *,
    username: str | None = None,
    password: str | None = None,
) -> datetime | None:
    """Return Hub ``last_updated`` for ``ref``, or None if the tag is missing."""
    try:
        ns, repo, tag = parse_docker_hub_ref(ref)
    except ValueError:
        return None
    cache_key = f"{ns}/{repo}:{tag}"
    with _hub_tag_mtime_lock:
        if cache_key in _hub_tag_mtime_cache:
            return _hub_tag_mtime_cache[cache_key]

    if not username or not password:
        # Anonymous Hub tag GETs are flaky / rate-limited; fall back to exists-only.
        exists = remote_tag_exists(ref, username=username, password=password)
        with _hub_tag_mtime_lock:
            _hub_tag_mtime_cache[cache_key] = datetime.now(timezone.utc) if exists else None
            return _hub_tag_mtime_cache[cache_key]

    try:
        token = _hub_jwt(username, password)
        url = (
            f"https://hub.docker.com/v2/repositories/{ns}/{repo}/tags/"
            f"{urllib.parse.quote(tag, safe='')}/"
        )
        req = urllib.request.Request(url, headers={"Authorization": f"JWT {token}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code != 401:
                raise
            with _hub_jwt_lock:
                _hub_jwt_cache.pop(f"{username}:{password}", None)
            token = _hub_jwt(username, password)
            req = urllib.request.Request(url, headers={"Authorization": f"JWT {token}"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
        updated = _parse_hub_datetime(str(payload["last_updated"]))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            updated = None
        else:
            # On Hub API errors, treat as missing so bake rebuilds rather than skipping.
            print(f"[bake] hub tag mtime failed for {ref}: HTTP {e.code}; will rebuild", flush=True)
            updated = None
    except Exception as e:
        print(f"[bake] hub tag mtime failed for {ref}: {e}; will rebuild", flush=True)
        updated = None

    with _hub_tag_mtime_lock:
        _hub_tag_mtime_cache[cache_key] = updated
    return updated


def should_skip_existing_remote_tag(
    ref: str,
    *,
    username: str | None = None,
    password: str | None = None,
    max_age_hours: float | None = None,
) -> bool:
    """Whether ``--skip-existing`` should skip building ``ref``.

    - ``max_age_hours is None``: skip whenever the tag exists (legacy behavior).
    - otherwise: skip only if the Hub tag exists **and** ``last_updated`` is
      within ``max_age_hours``; older tags are rebuilt.
    """
    if max_age_hours is None:
        if remote_tag_exists(ref, username=username, password=password):
            print(f"[bake] skip-existing {ref}", flush=True)
            return True
        return False

    updated = hub_tag_last_updated(ref, username=username, password=password)
    if updated is None:
        return False
    age = datetime.now(timezone.utc) - updated
    if age > timedelta(hours=max_age_hours):
        print(
            f"[bake] rebuild-stale {ref} age={age.total_seconds()/3600:.1f}h > {max_age_hours:g}h",
            flush=True,
        )
        return False
    print(
        f"[bake] skip-existing {ref} age={age.total_seconds()/3600:.1f}h <= {max_age_hours:g}h",
        flush=True,
    )
    return True


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
    """Make Hub repo publicly pullable.

    Docker Hub often keeps a newly pushed org repo private. A bare PATCH
    ``is_private=false`` can return 200 while leaving the repo private; the
    ``/privacy/`` endpoint is what actually flips visibility. Always verify
    with a follow-up GET before treating the repo as public.
    """
    global _hub_public_done
    key = f"{namespace}/{repository}"
    with _hub_public_lock:
        if key in _hub_public_done:
            return
        token = hub_login_token(username, password)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"JWT {token}",
        }
        base = f"https://hub.docker.com/v2/repositories/{namespace}/{repository}/"
        # Prefer privacy endpoint; also PATCH as a belt-and-suspenders.
        # A bare PATCH can return 200 while leaving the repo private.
        attempts = (
            ("POST", f"{base}privacy/", {"is_private": False}),
            ("PATCH", base, {"is_private": False}),
        )
        last_err: Exception | None = None
        for method, url, payload in attempts:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers=headers,
                method=method,
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    _ = resp.read()
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:500]
                last_err = RuntimeError(f"Hub {method} visibility failed ({e.code}): {body}")
                continue

        get_req = urllib.request.Request(base, headers={"Authorization": f"JWT {token}"})
        try:
            with urllib.request.urlopen(get_req, timeout=60) as resp:
                info = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"Hub GET after visibility change failed ({e.code}): {body}") from e

        if info.get("is_private") is not False:
            hint = f"; last write error: {last_err}" if last_err else ""
            raise RuntimeError(
                f"Hub repo {key} is still private after visibility change{hint}. "
                "Open https://hub.docker.com/repository/docker/"
                f"{key}/settings and set Visibility to Public."
            )
        print(f"[bake] set {key} is_private=false (verified)", flush=True)
        _hub_public_done.add(key)


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
        "crane not found. Install:\n"
        "  curl -fsSL https://github.com/google/go-containerregistry/releases/"
        "download/v0.20.3/go-containerregistry_Linux_x86_64.tar.gz "
        "| tar -xz -C /usr/local/bin crane"
    )


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
    """Prepare a minimal guest root under ``/tmp`` for proot+Kaniko."""
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


def check_prebaked_clis(root: Path, errors: list[str]) -> None:
    """Assert Node + agent CLIs exist under an exported rootfs.

    Absolute symlinks (e.g. ``/usr/local/bin/node`` → ``/opt/node22/bin/node``)
    must be resolved against ``root``, not the host filesystem.
    """
    for name in PREBAKED_CLI_BINS:
        path = root / "usr" / "local" / "bin" / name
        if path.is_symlink():
            target = Path(os.readlink(path))
            if target.is_absolute():
                resolved = root.joinpath(*target.parts[1:])
            else:
                resolved = path.parent / target
            # Follow one level of relative node-style links (npm → npm-cli.js).
            if resolved.is_symlink():
                t2 = Path(os.readlink(resolved))
                resolved = (
                    root.joinpath(*t2.parts[1:])
                    if t2.is_absolute()
                    else (resolved.parent / t2)
                )
            if not resolved.exists():
                errors.append(f"broken symlink /usr/local/bin/{name} -> {target}")
            continue
        if not path.is_file():
            errors.append(f"missing /usr/local/bin/{name}")
            continue
        if not os.access(path, os.X_OK):
            errors.append(f"not executable: /usr/local/bin/{name}")


def add_agent_asset_args(parser: Any) -> None:
    """Register --node-xz / --cc-tgz / opencode / pi / miniswe path flags."""
    parser.add_argument("--node-xz", type=Path, default=DEFAULT_NODE_XZ)
    parser.add_argument("--cc-tgz", type=Path, default=DEFAULT_CC_TGZ)
    parser.add_argument("--opencode-tgz", type=Path, default=DEFAULT_OPENCODE_TGZ)
    parser.add_argument("--pi-tgz", type=Path, default=DEFAULT_PI_TGZ)
    parser.add_argument("--miniswe-wheels-tar", type=Path, default=DEFAULT_MINISWE_WHEELS_TAR)


def ensure_assets_from_args(args: Any, *, assets_dir: Path) -> None:
    ensure_assets(
        node_xz=args.node_xz,
        cc_tgz=args.cc_tgz,
        opencode_tgz=args.opencode_tgz,
        pi_tgz=args.pi_tgz,
        miniswe_wheels_tar=args.miniswe_wheels_tar,
        assets_dir=assets_dir,
    )


def kaniko_build(
    *,
    tag: str,
    dockerfile_name: str,
    context_dir: Path,
    dockerfiles_dir: Path,
    tag_prefix: str,
    push: bool,
    executor: str,
    restored_instance_id: str | None = None,
    tar_path: Path | None = None,
    guest_root: Path | None = None,
    proot: str | None = None,
) -> None:
    """Build with Kaniko inside a proot guest (no Docker daemon / BuildKit).

    When ``restored_instance_id`` is set (ScaleSWE), bind-mount
    ``restored/<id>`` into the build context. Tmax omits restore.
    """
    ref = target_ref(tag_prefix, tag)
    dockerfile = dockerfiles_dir / dockerfile_name
    if not dockerfile.is_file():
        raise FileNotFoundError(dockerfile)

    assets_src = context_dir / "assets"
    for name in AGENT_ASSET_NAMES:
        src = assets_src / name
        if not src.is_file():
            raise FileNotFoundError(src)
    if not (dockerfiles_dir / dockerfile_name).is_file():
        raise FileNotFoundError(dockerfiles_dir / dockerfile_name)

    restored_src: Path | None = None
    if restored_instance_id is not None:
        restored_src = context_dir / "restored" / restored_instance_id
        if not (restored_src / "workspace").is_dir():
            raise FileNotFoundError(restored_src / "workspace")

    proot_bin = resolve_proot(proot)
    root_base = kaniko_guest_root_base()
    root = guest_root or (root_base / tag.replace("/", "_"))
    if root.exists():
        shutil.rmtree(root)
    ensure_kaniko_guest_root(root)
    if restored_instance_id is not None:
        (root / "build" / "restored" / restored_instance_id).mkdir(parents=True, exist_ok=True)

    fake_proc = root / "fake_proc"
    prepare_filtered_proc(fake_proc)

    # Write Hub auth into the proot guest. Do not copy ~/.docker/config.json:
    # that host file has vanished mid-bake (TOCTOU with parallel workers).
    hub_user, hub_pass = hub_credentials_from_env()
    guest_cfg = root / "root" / ".docker" / "config.json"
    if hub_user and hub_pass:
        write_registry_auth_file(guest_cfg, username=hub_user, password=hub_pass)
    else:
        docker_cfg = Path.home() / ".docker" / "config.json"
        if docker_cfg.is_file():
            guest_cfg.write_bytes(docker_cfg.read_bytes())

    etc = root / "etc"
    etc.mkdir(parents=True, exist_ok=True)
    if Path("/etc/resolv.conf").is_file():
        shutil.copy2("/etc/resolv.conf", etc / "resolv.conf")
    certs_src = Path("/etc/ssl/certs")
    if certs_src.is_dir():
        certs_dst = etc / "ssl" / "certs"
        if certs_dst.exists():
            shutil.rmtree(certs_dst)
        shutil.copytree(certs_src, certs_dst, symlinks=True, ignore_dangling_symlinks=True)

    out_host: Path | None = None
    if not push:
        if tar_path is None:
            tar_path = context_dir / "kaniko_out" / f"{tag}.tar"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        out_host = tar_path.parent

    try:
        kaniko_cmd = [
            "/kaniko/executor",
            "--force",
            f"--dockerfile=/build/dockerfiles/{dockerfile_name}",
            "--context=dir:///build",
            "--verbosity=info",
            "--ignore-path=/build",
            "--ignore-path=/kaniko",
            "--ignore-path=/fake_proc",
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
            f"{dockerfiles_dir.resolve()}:/build/dockerfiles",
            f"{Path(executor).resolve()}:/kaniko/executor",
        ]
        if restored_src is not None and restored_instance_id is not None:
            binds.append(f"{restored_src.resolve()}:/build/restored/{restored_instance_id}")
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
            raise RuntimeError(f"kaniko failed for {tag}:\n{log[-4000:]}")
        if not push and tar_path is not None and not tar_path.is_file():
            raise RuntimeError(f"kaniko --no-push did not write {tar_path}")
        if log.strip():
            print(log[-800:], flush=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        try:
            if root_base.is_dir() and not any(root_base.iterdir()):
                root_base.rmdir()
        except OSError:
            pass
