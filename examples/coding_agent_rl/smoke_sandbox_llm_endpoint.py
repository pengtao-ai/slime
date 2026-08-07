#!/usr/bin/env python3
"""From inside a Docker sandbox, POST to an OpenAI-compatible chat endpoint.

Also detects the local docker-rt failure mode we hit in training: ``docker exec``
returns exit=0 with empty stdout and does not actually run the command (so
response files never appear).

Example::

    export SMOKE_LLM_API_KEY='sk-...'
    python examples/coding_agent_rl/smoke_sandbox_llm_endpoint.py

    # Skip sandbox; verify the LLM from the host only:
    python examples/coding_agent_rl/smoke_sandbox_llm_endpoint.py --host-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slime.agent.sandbox import DockerSandbox


DEFAULT_BASE_URL = "http://208.64.254.187:8001"
DEFAULT_MODEL = "deepseek-v4-flash-0731"
DEFAULT_PROMPT = "Reply with exactly: pong"


def _build_payload(*, model: str, prompt: str, stream: bool, enable_thinking: bool) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": stream,
        "max_tokens": 64,
    }
    if enable_thinking:
        body["chat_template_kwargs"] = {"thinking": True, "reasoning_effort": "high"}
    return body


def _host_request(endpoint: str, api_key: str, payload: dict, timeout: int) -> tuple[int, bytes]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as e:
        return int(e.code), e.read() or b""


def _tcp_check(url: str, timeout: float = 5.0) -> str:
    u = urlparse(url)
    host = u.hostname or ""
    port = u.port or (443 if u.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return f"OK tcp {host}:{port}"
    except OSError as e:
        return f"FAIL tcp {host}:{port}: {e}"


async def _preflight_exec(sb: DockerSandbox) -> None:
    """Fail fast if docker exec is the broken no-op / empty-capture mode."""
    code, out, err = await sb.exec("echo SMOKE_EXEC_OK; pwd; id", timeout=30)
    text = (out or "").strip()
    print(f"[smoke] preflight exec exit={code} stdout={text!r} stderr={(err or '')[:200]!r}", flush=True)
    if code != 0 or "SMOKE_EXEC_OK" not in text:
        raise SystemExit(
            "FAIL: docker exec is broken or empty (same class of bug as training "
            "`connection reset by peer` / fake exit). LLM smoke cannot proceed "
            "until `docker exec <container> echo ok` works reliably on this host "
            f"(docker-rt socket). raw exit={code} out={text!r} err={(err or '')[:300]!r}"
        )

    await sb.write_file("/tmp/smoke_write_probe.txt", "probe-ok\n")
    # Prefer docker cp round-trip via read_file; also verify exec cat.
    via_api = await sb.read_file("/tmp/smoke_write_probe.txt")
    code2, via_exec, err2 = await sb.exec("cat /tmp/smoke_write_probe.txt", timeout=30)
    print(
        f"[smoke] preflight write/read api={via_api!r} exec_cat exit={code2} out={via_exec!r}",
        flush=True,
    )
    if "probe-ok" not in (via_api or "") and "probe-ok" not in (via_exec or ""):
        raise SystemExit(
            "FAIL: docker cp/exec file round-trip broken "
            f"api={via_api!r} exec={via_exec!r} err={(err2 or '')[:200]!r}"
        )


async def smoke_sandbox(
    *,
    image: str,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    stream: bool,
    enable_thinking: bool,
    pull: bool,
    timeout: int,
    network: str | None,
) -> None:
    os.environ["SLIME_AGENT_SANDBOX_BACKEND"] = "docker"
    if network:
        os.environ["SLIME_AGENT_DOCKER_NETWORK"] = network

    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = _build_payload(model=model, prompt=prompt, stream=stream, enable_thinking=enable_thinking)
    payload_json = json.dumps(payload, ensure_ascii=False)

    print(f"[smoke] image={image!r} network={os.environ.get('SLIME_AGENT_DOCKER_NETWORK', 'bridge')}", flush=True)
    print(f"[smoke] endpoint={endpoint}", flush=True)
    print(f"[smoke] host_tcp: {_tcp_check(endpoint)}", flush=True)

    async with DockerSandbox(image, pull=pull) as sb:
        print(f"[smoke] sandbox_id={sb.sandbox_id}", flush=True)
        await _preflight_exec(sb)

        # Network check from inside the sandbox (file-based so empty stdout cannot hide it).
        host = urlparse(endpoint).hostname or ""
        port = urlparse(endpoint).port or 80
        net_py = f"""
import socket
s = socket.socket()
s.settimeout(5)
try:
    s.connect(({host!r}, {port}))
    open("/tmp/smoke_tcp.txt","w").write("OK")
    print("TCP_OK")
except Exception as e:
    open("/tmp/smoke_tcp.txt","w").write("FAIL:"+str(e))
    print("TCP_FAIL", e)
    raise SystemExit(2)
finally:
    s.close()
"""
        await sb.write_file("/tmp/smoke_tcp_check.py", net_py)
        ncode, nout, nerr = await sb.exec("python3 /tmp/smoke_tcp_check.py", timeout=30)
        nfile = await sb.read_file("/tmp/smoke_tcp.txt")
        print(f"[smoke] sandbox_tcp exit={ncode} file={nfile!r} stdout={nout!r} stderr={(nerr or '')[:200]!r}", flush=True)
        if "OK" not in (nfile or "") and "TCP_OK" not in (nout or ""):
            raise SystemExit(
                "FAIL: sandbox cannot TCP-connect to the LLM endpoint. "
                "Try --network host, or ensure docker bridge has egress to "
                f"{host}:{port}. detail={nfile or nout or nerr!r}"
            )

        await sb.write_file("/tmp/smoke_llm_payload.json", payload_json + "\n")
        # docker-rt often drops `docker exec -e`; pass the key via a mode-600 file
        # written with docker cp (same path training uses for tarballs).
        await sb.write_file("/tmp/smoke_llm_key.txt", api_key)
        await sb.exec("chmod 600 /tmp/smoke_llm_key.txt", timeout=15)
        req_py = f"""
import json, urllib.error, urllib.request
url = {endpoint!r}
key = open("/tmp/smoke_llm_key.txt", "r", encoding="utf-8").read().strip()
body = open("/tmp/smoke_llm_payload.json", "rb").read()
open("/tmp/smoke_llm_progress.txt", "w").write("start bytes=%d key_len=%d\\n" % (len(body), len(key)))
if not key:
    open("/tmp/smoke_llm_progress.txt", "a").write("error:empty api key file\\n")
    raise SystemExit(3)
req = urllib.request.Request(
    url, data=body, method="POST",
    headers={{"Content-Type": "application/json", "Authorization": "Bearer " + key}},
)
try:
    with urllib.request.urlopen(req, timeout={timeout}) as resp:
        raw = resp.read()
        code = resp.status
except urllib.error.HTTPError as e:
    raw = e.read() or b""
    code = e.code
except Exception as e:
    open("/tmp/smoke_llm_progress.txt", "a").write("error:%s\\n" % e)
    open("/tmp/smoke_llm_http_code.txt", "w").write("err")
    open("/tmp/smoke_llm_resp.bin", "wb").write(b"")
    print("SMOKE_HTTP_ERROR", e)
    raise SystemExit(1)
open("/tmp/smoke_llm_resp.bin", "wb").write(raw)
open("/tmp/smoke_llm_http_code.txt", "w").write(str(code))
open("/tmp/smoke_llm_progress.txt", "a").write("done http=%s bytes=%d\\n" % (code, len(raw)))
print("SMOKE_HTTP", code, "BYTES", len(raw))
print(raw.decode("utf-8", errors="replace"))
"""
        await sb.write_file("/tmp/smoke_llm_req.py", req_py)
        code, out, err = await sb.exec("python3 /tmp/smoke_llm_req.py", timeout=timeout + 30)
        progress = await sb.read_file("/tmp/smoke_llm_progress.txt")
        file_body = await sb.read_file("/tmp/smoke_llm_resp.bin")
        http_code = (await sb.read_file("/tmp/smoke_llm_http_code.txt")).strip()

        print(f"[smoke] http_client exit={code}", flush=True)
        print(f"[smoke] progress:\n{(progress or '').strip() or '<empty>'}", flush=True)
        if err and err.strip():
            print(f"[smoke] stderr:\n{err.strip()[:800]}", flush=True)
        print(
            f"[smoke] http_code={http_code or '?'} body_bytes={len(file_body or '')} stdout_bytes={len(out or '')}",
            flush=True,
        )

        text = (file_body or out or "").strip()
        if not text:
            raise SystemExit(
                "empty response from endpoint after preflight passed. "
                f"progress={progress!r} stdout={out!r} stderr={(err or '')[:300]!r}"
            )
        preview = text if len(text) <= 2000 else text[:2000] + "\n...[truncated]..."
        print(f"[smoke] response preview:\n{preview}", flush=True)

        if code != 0 and "choices" not in text:
            raise SystemExit(f"request failed (exit={code})")
        if not stream:
            brace = text.find("{")
            json_text = text[brace:] if brace >= 0 else text
            data = json.loads(json_text)
            choices = data.get("choices") or []
            if not choices:
                raise SystemExit(f"no choices in response keys={list(data.keys())}")
            msg = (choices[0].get("message") or {}).get("content") or ""
            reasoning = (choices[0].get("message") or {}).get("reasoning_content") or ""
            print(f"[smoke] content_chars={len(msg)} reasoning_chars={len(reasoning)}", flush=True)
            if not msg and not reasoning:
                raise SystemExit("choices present but content/reasoning empty")

    print("[smoke] PASS", flush=True)


def smoke_host_only(*, base_url: str, model: str, api_key: str, prompt: str, stream: bool, enable_thinking: bool, timeout: int) -> None:
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = _build_payload(model=model, prompt=prompt, stream=stream, enable_thinking=enable_thinking)
    print(f"[smoke] host-only endpoint={endpoint}", flush=True)
    print(f"[smoke] host_tcp: {_tcp_check(endpoint)}", flush=True)
    status, raw = _host_request(endpoint, api_key, payload, timeout)
    text = raw.decode("utf-8", errors="replace")
    print(f"[smoke] http_code={status} body_bytes={len(raw)}", flush=True)
    print(f"[smoke] response preview:\n{text[:2000]}", flush=True)
    if status >= 400:
        raise SystemExit(f"host-only request HTTP {status}")
    if not stream:
        data = json.loads(text)
        if not (data.get("choices") or []):
            raise SystemExit(f"no choices: keys={list(data.keys())}")
    print("[smoke] PASS (host-only)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--image",
        default=os.environ.get("SMOKE_DOCKER_IMAGE", "aweaiteam/scaleswe:arviz-devs_preliz_pr249"),
    )
    p.add_argument("--base-url", default=os.environ.get("SMOKE_LLM_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--model", default=os.environ.get("SMOKE_LLM_MODEL", DEFAULT_MODEL))
    p.add_argument("--api-key", default=os.environ.get("SMOKE_LLM_API_KEY", ""))
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--pull", action="store_true")
    p.add_argument("--host-only", action="store_true", help="Call LLM from host; skip Docker sandbox")
    p.add_argument(
        "--network",
        default=os.environ.get("SLIME_AGENT_DOCKER_NETWORK", "bridge"),
        help="Docker network for the sandbox (try 'host' if bridge has no egress)",
    )
    args = p.parse_args()
    if not args.api_key:
        raise SystemExit("set --api-key or SMOKE_LLM_API_KEY")

    if args.host_only:
        smoke_host_only(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            prompt=args.prompt,
            stream=args.stream,
            enable_thinking=args.thinking,
            timeout=args.timeout,
        )
        return

    asyncio.run(
        smoke_sandbox(
            image=args.image,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            prompt=args.prompt,
            stream=args.stream,
            enable_thinking=args.thinking,
            pull=args.pull,
            timeout=args.timeout,
            network=args.network,
        )
    )


if __name__ == "__main__":
    main()
