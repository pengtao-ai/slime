#!/usr/bin/env python3
"""Build an offline-installable pi coding-agent npm tarball for slime sandboxes.

Packs ``@earendil-works/pi-coding-agent`` (bin: ``pi``) into
``tarballs/pi-coding-agent-local.tgz`` for ``npm install -g``.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tarballs"
# Prefer the actively published earendil package; falls back documented in README.
PKG = "@earendil-works/pi-coding-agent@0.84.1"


def npm_pack(workdir: Path, pkg: str) -> Path:
    before = {p.name for p in workdir.glob("*.tgz")}
    subprocess.run(["npm", "pack", pkg, "--silent"], cwd=workdir, check=True)
    after = [p for p in workdir.glob("*.tgz") if p.name not in before]
    if not after:
        raise RuntimeError(f"npm pack produced no new tarball for {pkg}")
    after.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"packed {after[0].name} ({after[0].stat().st_size} bytes)")
    return after[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="pi-local-"))
    print("workdir", workdir)

    tgz = npm_pack(workdir, PKG)
    cached = OUT / tgz.name
    shutil.copy2(tgz, cached)

    # Re-pack as a plain package/ tree named for slime install_npm_cli.
    root = workdir / "extract"
    root.mkdir()
    with tarfile.open(tgz) as tar:
        tar.extractall(root)
    pkg = root / "package"
    if not (pkg / "package.json").exists():
        raise RuntimeError(f"unexpected npm pack layout under {root}")

    out_tgz = OUT / "pi-coding-agent-local.tgz"
    with tarfile.open(out_tgz, "w:gz") as tar:
        tar.add(pkg, arcname="package")
    print(f"wrote {out_tgz} ({out_tgz.stat().st_size} bytes)")
    print(f"cached upstream pack at {cached}")


if __name__ == "__main__":
    main()
