#!/usr/bin/env python3
"""Build an offline-installable Codex npm tarball for slime sandboxes.

Pins ``@openai/codex@0.50.0`` — the last convenient full-package release that
still accepts ``wire_api = "chat"`` (hits ``/v1/chat/completions``). Newer
Codex only speaks the Responses API, which slime's OpenAIAdapter does not
implement yet.

Strips non-linux-x64 vendor trees to keep the tarball smaller, then writes
``tarballs/openai-codex-local.tgz`` for ``npm install -g``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tarballs"
VER = "0.50.0"
WRAPPER = f"@openai/codex@{VER}"
KEEP_TRIPLE = "x86_64-unknown-linux-musl"


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
    workdir = Path(tempfile.mkdtemp(prefix="codex-local-"))
    print("workdir", workdir)

    wrapper_cached = OUT / f"openai-codex-{VER}.tgz"
    if wrapper_cached.exists():
        wrapper_tgz = workdir / wrapper_cached.name
        shutil.copy2(wrapper_cached, wrapper_tgz)
        print(f"reuse {wrapper_cached}")
    else:
        wrapper_tgz = npm_pack(workdir, WRAPPER)
        shutil.copy2(wrapper_tgz, wrapper_cached)

    wrap_root = workdir / "wrap"
    wrap_root.mkdir()
    with tarfile.open(wrapper_tgz) as tar:
        tar.extractall(wrap_root)
    pkg = wrap_root / "package"
    if not (pkg / "package.json").exists():
        raise RuntimeError(f"unexpected npm pack layout under {wrap_root}")

    vendor = pkg / "vendor"
    if not vendor.is_dir():
        raise RuntimeError(f"no vendor/ in {pkg} (expected codex {VER} layout)")

    # Drop other platform trees (darwin/win/arm) to shrink the offline artifact.
    for child in list(vendor.iterdir()):
        if child.name != KEEP_TRIPLE:
            print(f"strip vendor/{child.name}")
            shutil.rmtree(child)

    # 0.50 layout: vendor/<triple>/codex/codex
    codex_bin = vendor / KEEP_TRIPLE / "codex" / "codex"
    if not codex_bin.is_file():
        # newer split packages used vendor/<triple>/bin/codex
        alt = vendor / KEEP_TRIPLE / "bin" / "codex"
        if alt.is_file():
            codex_bin = alt
        else:
            raise RuntimeError(f"missing baked binary under {vendor / KEEP_TRIPLE}")
    codex_bin.chmod(0o755)
    print(f"kept {codex_bin.relative_to(pkg)} ({codex_bin.stat().st_size} bytes)")

    pj = json.loads((pkg / "package.json").read_text())
    pj["optionalDependencies"] = {}
    scripts = pj.get("scripts") or {}
    scripts.pop("postinstall", None)
    scripts.pop("prepare", None)
    pj["scripts"] = scripts
    (pkg / "package.json").write_text(json.dumps(pj, indent=2) + "\n")

    out_tgz = OUT / "openai-codex-local.tgz"
    with tarfile.open(out_tgz, "w:gz") as tar:
        tar.add(pkg, arcname="package")
    print(f"wrote {out_tgz} ({out_tgz.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
