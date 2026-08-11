#!/usr/bin/env python3
"""Download mini-swe-agent + deps as wheels for offline sandbox pip install.

Writes ``tarballs/miniswe-wheels/*.whl`` (directory pointed at by
``SLIME_AGENT_MINISWE_WHEEL``).

Sandboxes differ: SWE-Smith images are CPython 3.10; some ScaleSWE smoke
images are 3.11. Download both ABIs into the same find-links dir.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tarballs" / "miniswe-wheels"
PKG = "mini-swe-agent==2.4.6"
# Host Python is often 3.11+, so marker-gated deps for 3.10 (e.g. async-timeout
# via aiohttp) can be missed; pin them explicitly.
EXTRA_PKGS = ("async-timeout>=4.0,<6.0",)
PLATFORM = "manylinux2014_x86_64"
TARGETS = (
    ("310", "cp310"),
    ("311", "cp311"),
)


def _download(py_ver: str, abi: str, packages: tuple[str, ...]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        *packages,
        "-d",
        str(OUT),
        "--only-binary=:all:",
        f"--python-version={py_ver}",
        f"--platform={PLATFORM}",
        "--implementation=cp",
        f"--abi={abi}",
    ]
    print("running", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        cmd_sdist = [
            sys.executable,
            "-m",
            "pip",
            "download",
            *packages,
            "-d",
            str(OUT),
            f"--python-version={py_ver}",
            f"--platform={PLATFORM}",
        ]
        print("retry with sdists allowed:", " ".join(cmd_sdist))
        subprocess.run(cmd_sdist, check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in list(OUT.iterdir()):
        if old.suffix == ".whl" or old.name.endswith(".tar.gz") or old.suffix == ".zip":
            old.unlink()
    for py_ver, abi in TARGETS:
        _download(py_ver, abi, (PKG, *EXTRA_PKGS))
    wheels = sorted(OUT.glob("*.whl")) + sorted(OUT.glob("*.tar.gz"))
    print(f"wrote {len(wheels)} artifacts under {OUT}")
    for w in wheels[:25]:
        print(f"  {w.name} ({w.stat().st_size} bytes)")
    if len(wheels) > 25:
        print(f"  ... and {len(wheels) - 25} more")
    has_310 = any("cp310" in w.name for w in OUT.glob("*.whl"))
    has_311 = any("cp311" in w.name for w in OUT.glob("*.whl"))
    if not (has_310 and has_311):
        raise SystemExit(f"expected both cp310 and cp311 wheels (310={has_310} 311={has_311})")
    if not any(w.name.startswith("async_timeout") or w.name.startswith("async-timeout") for w in OUT.glob("*.whl")):
        raise SystemExit("missing async-timeout wheel required for CPython 3.10 aiohttp")


if __name__ == "__main__":
    main()
