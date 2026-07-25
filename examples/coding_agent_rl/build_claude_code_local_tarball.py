#!/usr/bin/env python3
"""Build an offline-installable Claude Code npm tarball for slime sandboxes.

Downloads @anthropic-ai/claude-code + @anthropic-ai/claude-code-linux-x64,
bakes the native binary into bin/claude.exe, and writes
anthropic-ai-claude-code-local-linux-x64.tgz (npm install -g compatible).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tarballs"
VER = "2.1.217"
WRAPPER = f"@anthropic-ai/claude-code@{VER}"
BINARY_PKG = f"@anthropic-ai/claude-code-linux-x64@{VER}"


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
    workdir = Path(tempfile.mkdtemp(prefix="cc-local-"))
    print("workdir", workdir)

    wrapper_cached = OUT / f"anthropic-ai-claude-code-{VER}.tgz"
    if wrapper_cached.exists():
        wrapper_tgz = workdir / wrapper_cached.name
        shutil.copy2(wrapper_cached, wrapper_tgz)
        print(f"reuse {wrapper_cached}")
    else:
        wrapper_tgz = npm_pack(workdir, WRAPPER)
        shutil.copy2(wrapper_tgz, wrapper_cached)

    bin_tgz = npm_pack(workdir, BINARY_PKG)
    shutil.copy2(bin_tgz, OUT / bin_tgz.name)

    wrap_root = workdir / "wrap"
    wrap_root.mkdir()
    with tarfile.open(wrapper_tgz) as tar:
        tar.extractall(wrap_root)
    pkg = wrap_root / "package"

    bin_root = workdir / "binpkg"
    bin_root.mkdir()
    with tarfile.open(bin_tgz) as tar:
        tar.extractall(bin_root)
    bpkg = bin_root / "package"
    files = [p for p in bpkg.rglob("*") if p.is_file()]
    print("binary package files:", len(files))
    candidates = [p for p in files if p.name == "claude"]
    if not candidates:
        candidates = [p for p in files if "claude" in p.name.lower() and p.suffix not in {".md", ".json", ".txt"}]
    if not candidates:
        raise RuntimeError(f"no claude binary in {bpkg}; sample={[str(p.relative_to(bpkg)) for p in files[:20]]}")
    binary = max(candidates, key=lambda p: p.stat().st_size)
    print(f"using binary {binary.relative_to(bpkg)} ({binary.stat().st_size} bytes)")

    dest = pkg / "bin" / "claude.exe"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, dest)
    dest.chmod(0o755)

    pj = json.loads((pkg / "package.json").read_text())
    pj["optionalDependencies"] = {}
    scripts = pj.get("scripts") or {}
    scripts.pop("postinstall", None)
    scripts.pop("prepare", None)
    pj["scripts"] = scripts
    (pkg / "package.json").write_text(json.dumps(pj, indent=2) + "\n")

    out_tgz = OUT / "anthropic-ai-claude-code-local-linux-x64.tgz"
    with tarfile.open(out_tgz, "w:gz") as tar:
        tar.add(pkg, arcname="package")
    print(f"wrote {out_tgz} ({out_tgz.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
