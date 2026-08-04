#!/usr/bin/env python3
"""Smoke: compare bare ``docker run`` vs slime boot_sandbox / prepare_workspace.

``generate.boot_agent_sandbox``'s ``boot_sandbox`` span includes BOTH:

  1. ``DockerSandbox.__aenter__``  (docker run -d --entrypoint sleep ...)
  2. ``Harness.install_cli``       (upload node/cc tarballs + npm install)

So a ~90s ``boot_sandbox`` is usually install_cli, not container create.
This script times them separately and also runs a bare docker baseline.

Run (micromamba slime)::

    micromamba run -n slime python examples/coding_agent_rl/smoke_boot_prepare_timeline.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

os.environ.setdefault("SLIME_AGENT_SANDBOX_BACKEND", "docker")
os.environ.setdefault("SLIME_AGENT_E2B_USE_TEMPLATE", "0")
os.environ.setdefault("SLIME_AGENT_DOCKER_PULL", "0")
os.environ.setdefault(
    "SLIME_AGENT_NODE_TARBALL",
    str(_SCRIPT_DIR / "tarballs" / "node-v22.20.0-linux-x64.tar.xz"),
)
os.environ.setdefault(
    "SLIME_AGENT_CC_TARBALL",
    str(_SCRIPT_DIR / "tarballs" / "anthropic-ai-claude-code-local-linux-x64.tgz"),
)

from slime.agent.chrome_trace import chrome_span  # noqa: E402
from slime.agent.sandbox import make_sandbox  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

from examples.coding_agent_rl import swe  # noqa: E402
from examples.coding_agent_rl.generate import HARNESS_CLS  # noqa: E402


def _load_sample(path: Path, *, index: int = 0) -> Sample:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"empty prompt data: {path}")
    if index < 0 or index >= len(lines):
        raise SystemExit(f"--index {index} out of range (n={len(lines)})")
    row = json.loads(lines[index])
    return Sample(
        prompt=row.get("prompt") or "",
        label=row.get("label"),
        metadata=row.get("metadata") or {},
        index=index,
        group_index=0,
    )


def _pair_spans(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stack: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for ev in events:
        ph = ev.get("ph")
        if ph == "B":
            stack.append(ev)
            continue
        if ph != "E":
            continue
        name, tid = ev.get("name"), ev.get("tid")
        for i in range(len(stack) - 1, -1, -1):
            beg = stack[i]
            if beg.get("name") == name and beg.get("tid") == tid:
                stack.pop(i)
                pairs.append(
                    {
                        "name": name,
                        "tid": tid,
                        "cat": beg.get("cat"),
                        "wall_start_us": float(beg["ts"]),
                        "wall_end_us": float(ev["ts"]),
                        "args": beg.get("args") or {},
                        "status": (ev.get("args") or {}).get("status"),
                    }
                )
                break
    return pairs


def _print_timeline(pairs: list[dict[str, Any]], *, title: str) -> None:
    if not pairs:
        print(f"[{title}] no B/E pairs", flush=True)
        return
    t0 = min(p["wall_start_us"] for p in pairs)
    print(f"======== {title} ========", flush=True)
    print(f"{'name':22} {'start_us':>18} {'end_us':>18} {'dt_s':>10}  notes", flush=True)
    for p in pairs:
        dt_s = (p["wall_end_us"] - p["wall_start_us"]) / 1e6
        rel_s = (p["wall_start_us"] - t0) / 1e6
        note = f"rel+{rel_s:.3f}s"
        if p.get("status"):
            note += f" status={p['status']}"
        print(
            f"{p['name']:22} {p['wall_start_us']:18.0f} {p['wall_end_us']:18.0f} {dt_s:10.3f}  {note}",
            flush=True,
        )


def _write_chrome_json(path: Path, events: list[dict[str, Any]], *, thread_name: str) -> None:
    doc = {
        "traceEvents": [
            {"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "smoke_boot_prepare"}},
            {"name": "thread_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": thread_name}},
            *events,
        ],
        "displayTimeUnit": "ms",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _docker_run_cmd(image: str, name: str) -> list[str]:
    """Same argv shape as ``DockerSandbox.__aenter__`` (no pull)."""
    network = (os.environ.get("SLIME_AGENT_DOCKER_NETWORK") or "bridge").lower()
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--entrypoint",
        "sleep",
        "--network",
        network,
    ]
    if network != "host":
        add_host = os.environ.get("SLIME_AGENT_DOCKER_ADD_HOST", "host.docker.internal:host-gateway")
        if add_host:
            cmd.extend(["--add-host", add_host])
    cmd.extend([image, "infinity"])
    return cmd


def baseline_bare_docker_run(image: str) -> float:
    """Time a bare ``docker run`` + immediate ``rm -f`` (no install_cli)."""
    name = f"slime-baseline-{uuid.uuid4().hex[:10]}"
    cmd = _docker_run_cmd(image, name)
    print(f"[baseline] cmd={' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        dt = time.perf_counter() - t0
        # Prove the container is up.
        ps = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(f"[baseline] running={ps.stdout.strip()} docker_run_sec={dt:.3f}", flush=True)
        return dt
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True, timeout=60)


async def run_slime_path(
    *,
    image: str,
    workdir: str,
    md: dict[str, Any],
    instance_id: str,
    events: list[dict[str, Any]],
    tid: int,
) -> None:
    """Split what ``boot_sandbox`` bundles: docker __aenter__ vs install_cli."""
    sb = make_sandbox(image)
    try:
        with chrome_span(
            events,
            "docker_aenter",
            cat="baseline",
            tid=tid,
            args={"instance_id": instance_id, "image": image},
        ):
            await sb.__aenter__()
        print(f"[slime] sandbox_id={sb.sandbox_id}", flush=True)

        with chrome_span(
            events,
            "install_cli",
            cat="outer",
            tid=tid,
            args={"instance_id": instance_id, "harness": getattr(HARNESS_CLS, "name", HARNESS_CLS.__name__)},
        ):
            await HARNESS_CLS().install_cli(sb)

        with chrome_span(
            events,
            "prepare_workspace",
            cat="outer",
            tid=tid,
            args={"instance_id": instance_id, "workdir": workdir},
        ):
            await swe.prepare_workspace(sb, workdir, md)

        code, out_txt, err = await sb.exec(
            f"test -f {workdir}/PROBLEM_STATEMENT.md && echo PS_OK || echo PS_MISSING",
            timeout=30,
        )
        print(f"[slime] post-prepare check exit={code} {(out_txt or err or '').strip()}", flush=True)
        if "PS_OK" not in (out_txt or ""):
            raise SystemExit("PROBLEM_STATEMENT.md missing after prepare_workspace")
    finally:
        await sb.__aexit__(None, None, None)


async def run(
    *,
    prompt_data: Path,
    index: int,
    image: str | None,
    out: Path,
    skip_baseline: bool,
) -> None:
    sample = _load_sample(prompt_data, index=index)
    md = swe.get_metadata(sample, swe.PROTOCOL_SCALESWE)
    instance_id = md["instance_id"]
    workdir = md["workdir"]
    img = image or md["image"]
    if not img or not workdir:
        raise SystemExit(f"missing image/workdir: image={img!r} workdir={workdir!r}")

    for key in ("SLIME_AGENT_NODE_TARBALL", "SLIME_AGENT_CC_TARBALL"):
        p = Path(os.environ[key])
        if not p.is_file():
            raise SystemExit(f"{key} not found: {p}")

    print(
        f"[smoke] instance={instance_id} image={img} workdir={workdir}\n"
        f"[smoke] node={os.environ['SLIME_AGENT_NODE_TARBALL']}\n"
        f"[smoke] cc={os.environ['SLIME_AGENT_CC_TARBALL']}\n"
        f"[smoke] SLIME_AGENT_DOCKER_PULL={os.environ.get('SLIME_AGENT_DOCKER_PULL')}",
        flush=True,
    )

    baseline_sec: float | None = None
    if not skip_baseline:
        print("-------- bare docker run (no install_cli) --------", flush=True)
        baseline_sec = baseline_bare_docker_run(img)

    tid = 1
    events: list[dict[str, Any]] = []
    # Record baseline as an Instant + synthetic Complete-like B/E for the table.
    if baseline_sec is not None:
        # Synthetic span at the start of the slime path for side-by-side printing.
        # Use wall clock around a no-op sleep of 0 after we already measured.
        with chrome_span(
            events,
            "bare_docker_run",
            cat="baseline",
            tid=tid,
            args={"image": img, "measured_sec": round(baseline_sec, 3)},
        ):
            # Span duration here is ~0; real seconds are in args.measured_sec /
            # the printed baseline line. Keep event for chrome JSON metadata.
            pass

    print("-------- slime docker_aenter + install_cli + prepare --------", flush=True)
    await run_slime_path(
        image=img,
        workdir=workdir,
        md=md,
        instance_id=instance_id,
        events=events,
        tid=tid,
    )

    pairs = _pair_spans([e for e in events if e.get("name") != "bare_docker_run"])
    _print_timeline(pairs, title="slime timeline (wall timestamps)")

    by_name = {p["name"]: (p["wall_end_us"] - p["wall_start_us"]) / 1e6 for p in pairs}
    print("======== comparison ========", flush=True)
    if baseline_sec is not None:
        print(f"bare docker run:              {baseline_sec:8.3f}s", flush=True)
    print(f"slime docker_aenter:          {by_name.get('docker_aenter', float('nan')):8.3f}s", flush=True)
    print(f"slime install_cli:            {by_name.get('install_cli', float('nan')):8.3f}s", flush=True)
    print(f"slime prepare_workspace:      {by_name.get('prepare_workspace', float('nan')):8.3f}s", flush=True)
    if baseline_sec is not None and "docker_aenter" in by_name:
        print(
            f"delta (docker_aenter - bare): {by_name['docker_aenter'] - baseline_sec:+8.3f}s",
            flush=True,
        )
    print(
        "note: generate.boot_sandbox ≈ docker_aenter + install_cli "
        f"= {by_name.get('docker_aenter', 0) + by_name.get('install_cli', 0):.3f}s",
        flush=True,
    )

    _write_chrome_json(out, events, thread_name=f"{instance_id}#smoke")
    print(f"[smoke] wrote chrome trace: {out}", flush=True)
    print("[smoke] PASS", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--prompt-data",
        type=Path,
        default=_SCRIPT_DIR / "data" / "swe_smoke_preliz_docker.jsonl",
    )
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--image", default=None, help="Override metadata.image")
    p.add_argument(
        "--out",
        type=Path,
        default=_SCRIPT_DIR / "timelines" / "smoke_boot_prepare.json",
    )
    p.add_argument("--skip-baseline", action="store_true", help="Skip bare docker run comparison")
    args = p.parse_args()
    asyncio.run(
        run(
            prompt_data=args.prompt_data,
            index=args.index,
            image=args.image,
            out=args.out,
            skip_baseline=args.skip_baseline,
        )
    )


if __name__ == "__main__":
    main()
