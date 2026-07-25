"""Sandbox backends for agent rollouts.

The public sandbox contract is intentionally small: async context management,
command execution, and file read/write. Agent examples can build task-specific
setup, runner, and evaluator logic on top of this without depending directly on
one sandbox provider.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import random
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


ExecResult = tuple[int, str, str]
FileContent = str | bytes | Path


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async sandbox interface used by agent rollouts.

    ``write_file`` accepts either in-memory content (``str``/``bytes``) or a
    host ``Path`` to stream into the sandbox.

    Retry/idempotency is deliberately *not* part of this contract: whether a
    severed RPC is safe to re-send is a backend transport concern (see
    ``E2BSandbox._rpc_retry``), not something abstraction consumers reason about.
    """

    sandbox_id: str

    async def __aenter__(self) -> Sandbox: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult: ...

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None: ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str: ...


EXIT_TIME_BUDGET_EXCEEDED = -1


async def _await_done_marker(sb: Sandbox, done_file: str, *, user: str, time_budget_sec: int) -> int:
    """Poll a detached command's exit-code marker until it appears, returning the
    exit code (or ``EXIT_TIME_BUDGET_EXCEEDED`` if the budget runs out first).

    The 5s ``test -f && cat`` polls are deliberately short, idempotent RPCs --
    they keep the sandbox alive against idle GC while the detached command runs
    over a stream the gateway can't sever.
    """
    deadline = time.time() + time_budget_sec
    while time.time() < deadline:
        await asyncio.sleep(5)
        ec, out, _ = await sb.exec(f"test -f {done_file} && cat {done_file}", user=user, timeout=15, check=False)
        if ec == 0 and (out or "").strip():
            return int(out.strip())
    return EXIT_TIME_BUDGET_EXCEEDED


async def exec_and_wait(
    sb: Sandbox,
    *,
    cmd: str,
    time_budget_sec: int,
    tag: str,
    user: str = "root",
    env: dict[str, str] | None = None,
    workdir: str | None = None,
    out_file: str | None = None,
    want_output: bool = False,
) -> tuple[int, str]:
    """Run ``cmd`` to completion detached, returning ``(exit_code, output)``.

    A plain ``sb.exec`` keeps an HTTP/2 stream open for the command's whole
    runtime, so a long-running command (build, test suite) outlives what the
    E2B gateway will hold a single response stream open for: the stream gets
    severed mid-run and we lose the exit code with no safe way to retry a
    non-idempotent command. Instead we ``setsid`` the command fully detached,
    redirect its output to a file, and have it drop its exit code into a marker
    file. The caller side then becomes a sequence of short, idempotent RPCs --
    write the launcher, fire-and-forget the spawn, then poll for the marker (see
    ``_await_done_marker``) -- none of which depend on a stream staying alive,
    and the polling doubles as an idle-GC keepalive while the command runs.
    """
    import shlex

    out_file = out_file or f"/tmp/.{tag}.out"
    done_file = f"/tmp/.{tag}.done"
    launcher = f"/tmp/.{tag}.sh"
    lock_dir = f"/tmp/.{tag}.spawned"
    prefix = f"cd {workdir}\nexport HOME=/home/{user}\n" if workdir else ""
    # Bake env into the launcher. Detached `setsid bash` must not rely on
    # `docker exec -e` (docker-rt drops -e; Claude Code then never sees
    # ANTHROPIC_BASE_URL and the host adapter gets adapter_session_empty).
    exports = ""
    if env:
        exports = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
    launcher_body = f"#!/bin/bash\n{exports}{prefix}{cmd}\necho $? > {done_file}\n"
    await sb.write_file(launcher, launcher_body, user=user)

    await sb.exec(
        f"chmod +x {launcher}; "
        f"mkdir {lock_dir} 2>/dev/null || exit 0; "
        f"rm -f {out_file} {done_file}; "
        f"setsid bash {launcher} < /dev/null > {out_file} 2>&1 &",
        user=user,
        timeout=30,
        check=True,
        idempotent=True,
    )
    exit_code = await _await_done_marker(sb, done_file, user=user, time_budget_sec=time_budget_sec)
    if exit_code == 0 and not want_output:
        return exit_code, ""
    if want_output:
        return exit_code, await sb.read_file(out_file, user=user)
    _, tail, _ = await sb.exec(f"tail -c 512 {out_file} 2>/dev/null", user=user, timeout=15, check=False)
    return exit_code, tail or ""


def _getenv(*names: str, default: str = "") -> str:
    """First non-empty environment value among ``names`` (else ``default``).

    Lets a setting carry a primary name plus legacy aliases: list the canonical
    ``SLIME_AGENT_*`` name first, older names after."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return default


class E2BSandbox:
    """Async context manager around e2b.AsyncSandbox."""

    image_metadata_key_env = ("SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY", "SWE_SANDBOX_IMAGE_METADATA_KEY")
    lifetime_sec_env = ("SLIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC")
    rpc_retries_env = ("SLIME_AGENT_SANDBOX_RPC_RETRIES", "SWE_RPC_RETRIES")
    size_env = ("SLIME_AGENT_E2B_SANDBOX_SIZE", "SWE_E2B_SANDBOX_SIZE")

    default_lifetime_sec = 3600
    default_rpc_retries = 6
    default_size = "md"
    rpc_backoff_base_sec = 1.0
    rpc_backoff_cap_sec = 32.0

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        image_metadata_key: str | None = None,
        rpc_retries: int | None = None,
        size: str | None = None,
    ) -> None:
        self.image = image
        self.timeout = timeout if timeout is not None else self._lifetime_sec_from_env()
        self.image_metadata_key = image_metadata_key or self._image_metadata_key_from_env()
        self.rpc_retries = rpc_retries if rpc_retries is not None else self._rpc_retries_from_env()
        self.size = size if size is not None else self._size_from_env()
        self._sb = None
        self.sandbox_id = ""

    @classmethod
    def _image_metadata_key_from_env(cls) -> str | None:
        return _getenv(*cls.image_metadata_key_env) or None

    @classmethod
    def _lifetime_sec_from_env(cls) -> int:
        return int(_getenv(*cls.lifetime_sec_env, default=str(cls.default_lifetime_sec)))

    @classmethod
    def _rpc_retries_from_env(cls) -> int:
        return int(_getenv(*cls.rpc_retries_env, default=str(cls.default_rpc_retries)))

    @classmethod
    def _size_from_env(cls) -> str:
        return _getenv(*cls.size_env, default=cls.default_size)

    # Transient client-side failures safe to retry.
    _TRANSIENT_RPC_ERRORS = frozenset(
        {
            "ProtocolError",
            "LocalProtocolError",
            "WriteError",
            "ReadError",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "RemoteProtocolError",
            "SSLError",
        }
    )

    @classmethod
    def _is_transient_rpc_error(cls, e: BaseException) -> bool:
        """True if e is a transient E2B client-side failure safe to retry."""
        name = type(e).__name__
        if name in cls._TRANSIENT_RPC_ERRORS:
            return True
        msg = str(e)
        if name == "SandboxException":
            if "does not exist" in msg or "STOPPED state" in msg:
                return False
            return True
        return False

    async def _rpc_retry(self, op_name: str, coro_factory, *, idempotent: bool = True):
        """Run coro_factory() with retries for transient E2B RPC failures.

        :param idempotent: When False, a transient failure is re-raised instead
            of retried: re-running a non-idempotent op (e.g. a process-spawning
            exec) after a severed response could double-execute it. Idempotent
            ops (the default: create / read_file / write_file / short read-only
            execs) retry as before.
        """
        last_err = None
        for attempt in range(self.rpc_retries):
            try:
                return await coro_factory()
            except Exception as e:
                if not self._is_transient_rpc_error(e):
                    raise
                if not idempotent:
                    raise
                last_err = e
                if attempt + 1 < self.rpc_retries:
                    await self._reset_conn_pool()
                    ceiling = min(self.rpc_backoff_cap_sec, self.rpc_backoff_base_sec * (2**attempt))
                    backoff = random.uniform(0.0, ceiling)
                    logger.debug(
                        "[agent.sandbox] %s transient %s, retry %d/%d in %.1fs: %s",
                        op_name,
                        type(e).__name__,
                        attempt + 1,
                        self.rpc_retries,
                        backoff,
                        str(e)[:120],
                    )
                    await asyncio.sleep(backoff)
        assert last_err is not None
        raise last_err

    async def _reset_conn_pool(self) -> None:
        """Tear down the sandbox's httpcore pool so the next RPC reconnects."""
        try:
            pool = self._sb._transport.pool  # httpcore.AsyncConnectionPool
            await pool.aclose()
        except Exception as e:
            logger.debug("[agent.sandbox] conn-pool reset skipped: %s", e)

    async def __aenter__(self) -> E2BSandbox:
        from e2b import AsyncSandbox  # type: ignore

        # Public e2b.dev selects the environment via ``template=`` (an E2B
        # template name/alias). Internal gateways instead route on metadata
        # (``SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY`` → Docker image).
        use_template = (_getenv("SLIME_AGENT_E2B_USE_TEMPLATE", default="") or "").lower() in (
            "1",
            "true",
            "yes",
        )
        if use_template:
            self._sb = await self._rpc_retry(
                "create",
                lambda: AsyncSandbox.create(template=self.image, timeout=self.timeout),
            )
        else:
            if self.image_metadata_key is None:
                raise RuntimeError(
                    "SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY is not set. Export it "
                    "to the metadata key your E2B gateway uses for image routing. "
                    "The legacy SWE_SANDBOX_IMAGE_METADATA_KEY name is also "
                    "accepted for coding-agent examples. For public e2b.dev, "
                    "export SLIME_AGENT_E2B_USE_TEMPLATE=1 and pass an E2B "
                    "template name as metadata.image instead."
                )
            md = {self.image_metadata_key: self.image}

            if self.size:
                prefix = self.image_metadata_key.rsplit("/", 1)[0] if "/" in self.image_metadata_key else ""
                size_key = f"{prefix}/size" if prefix else "size"
                md[size_key] = self.size

            self._sb = await self._rpc_retry(
                "create", lambda: AsyncSandbox.create(timeout=self.timeout, metadata=md)
            )
        self.sandbox_id = self._sb.sandbox_id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._sb is not None:
                await self._sb.kill()
        except Exception as e:
            logger.warning("[agent.sandbox] kill %s failed: %s", self.sandbox_id[:8], e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        from e2b.sandbox.commands.command_handle import CommandExitException

        try:
            res = await self._rpc_retry(
                f"exec({cmd[:60]!r})",
                lambda: self._sb.commands.run(
                    cmd,
                    user=user,
                    envs=env,
                    timeout=timeout,
                    on_stdout=lambda s: None,
                    on_stderr=lambda s: None,
                ),
                idempotent=idempotent,
            )
            return res.exit_code, res.stdout or "", res.stderr or ""
        except CommandExitException as e:
            if check:
                raise RuntimeError(
                    f"e2b exec failed (exit={e.exit_code}): {cmd[:120]}\n{(e.stderr or '')[:400]}"
                ) from None
            return e.exit_code, e.stdout or "", e.stderr or ""

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            host_path = content

            async def _do_path():
                with open(host_path, "rb") as fp:
                    await self._sb.files.write(
                        sandbox_path,
                        fp,
                        user=user,
                        gzip=False,
                        use_octet_stream=True,
                        request_timeout=600,
                    )

            await self._rpc_retry(f"write_file({sandbox_path} <- {host_path.name})", _do_path)
            return

        if isinstance(content, bytes):

            async def _do_bytes():
                await self._sb.files.write(
                    sandbox_path,
                    io.BytesIO(content),
                    user=user,
                    gzip=False,
                    use_octet_stream=True,
                    request_timeout=600,
                )

            await self._rpc_retry(f"write_file({sandbox_path}, bytes={len(content)})", _do_bytes)
            return

        await self._rpc_retry(
            f"write_file({sandbox_path})",
            lambda: self._sb.files.write(sandbox_path, content, user=user),
        )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            return await self._rpc_retry(
                f"read_file({sandbox_path})",
                lambda: self._sb.files.read(sandbox_path, user=user),
            )
        except Exception:
            return ""


async def ensure_agent_user(sb: Sandbox, workdir: str) -> None:
    """Create the unprivileged 'agent' user that owns workdir + can git diff."""
    # ScaleSWE / SWE images must ship ``workdir`` already (repo checkout).
    # If it is missing, chown fails obscurely — surface image-routing issues.
    ec, _, _ = await sb.exec(f"test -d {workdir}", user="root", check=False, timeout=30)
    if ec != 0:
        _, listing, _ = await sb.exec(
            "ls -la /workspace 2>/dev/null || ls -la / 2>/dev/null | head -40",
            user="root",
            check=False,
            timeout=30,
        )
        raise RuntimeError(
            f"workdir {workdir!r} does not exist in the sandbox. "
            "The E2B gateway likely did not boot the sample's Docker image "
            f"(check SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY / metadata.image). "
            f"Listing:\n{(listing or '').strip()[:500]}"
        )
    await sb.exec(
        f"id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent && "
        f"chown -R agent:agent /home/agent {workdir} && "
        f"git config --system --add safe.directory '*' && id agent",
        user="root",
        check=True,
        timeout=60,
    )


class DockerSandbox:
    """Local Docker / docker-rt sandbox implementing the ``Sandbox`` protocol.

    Env knobs:
      * ``SLIME_AGENT_DOCKER_NETWORK`` (default ``bridge``)
      * ``SLIME_AGENT_DOCKER_ADD_HOST`` (default ``host.docker.internal:host-gateway``)
      * ``SLIME_AGENT_DOCKER_PULL`` (``1``/``true`` to pull before run)
      * ``SLIME_AGENT_DOCKER_RUN_TIMEOUT_SEC`` (default ``300``)
    """

    image_metadata_key_env = E2BSandbox.image_metadata_key_env
    lifetime_sec_env = E2BSandbox.lifetime_sec_env
    default_lifetime_sec = E2BSandbox.default_lifetime_sec

    def __init__(self, image: str, *, timeout: int | None = None) -> None:
        self.image = image
        self.timeout = int(timeout if timeout is not None else (_getenv(*self.lifetime_sec_env) or self.default_lifetime_sec))
        self.network = (_getenv("SLIME_AGENT_DOCKER_NETWORK", default="bridge") or "bridge").lower()
        self.pull = (_getenv("SLIME_AGENT_DOCKER_PULL", default="") or "").lower() in ("1", "true", "yes")
        self.remove = True
        self.sandbox_id = ""
        self._container = ""

    async def __aenter__(self) -> DockerSandbox:
        import uuid

        name = f"slime-sb-{uuid.uuid4().hex[:12]}"
        if self.pull:
            await self._run_host(["docker", "pull", self.image], timeout=600, check=True)

        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            # Override image ENTRYPOINT so sleep works on SWE images.
            "--entrypoint",
            "sleep",
            "--network",
            self.network,
        ]
        if self.network != "host":
            add_host = _getenv("SLIME_AGENT_DOCKER_ADD_HOST", default="host.docker.internal:host-gateway")
            if add_host:
                cmd.extend(["--add-host", add_host])
        cmd.extend([self.image, "infinity"])

        # docker-rt / k8s can take longer than a local daemon to schedule a pod.
        run_timeout = int(_getenv("SLIME_AGENT_DOCKER_RUN_TIMEOUT_SEC", default="300") or "300")
        try:
            await self._run_host(cmd, timeout=run_timeout, check=True)
        except Exception:
            # Client timeout/failure does not mean the daemon failed: a container
            # may already exist under ``name``. ``async with`` skips __aexit__ when
            # __aenter__ raises, so best-effort rm here to avoid quota leaks.
            try:
                await self._run_host(["docker", "rm", "-f", name], timeout=60, check=False)
            except Exception as cleanup_err:
                logger.warning(
                    "[agent.sandbox] orphan cleanup docker rm -f %s failed: %s",
                    name,
                    cleanup_err,
                )
            raise
        self._container = name
        self.sandbox_id = name
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        name = self._container
        self._container = ""
        if not name or not self.remove:
            return
        try:
            await self._run_host(["docker", "rm", "-f", name], timeout=60, check=False)
        except Exception as e:
            logger.warning("[agent.sandbox] docker cleanup %s failed: %s", name, e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        import shlex

        if not self._container:
            raise RuntimeError("DockerSandbox is not started")

        # Bake env into the shell (docker-rt often drops `docker exec -e`).
        exports = "".join(f"export {shlex.quote(k)}={shlex.quote(v)}; " for k, v in (env or {}).items())
        inner = exports + cmd

        if user and user != "root":
            # docker-rt often ignores `docker exec -u`; switch inside the container.
            uq = shlex.quote(user)
            cq = shlex.quote(inner)
            launcher = (
                f"if command -v runuser >/dev/null 2>&1; then runuser -u {uq} -- env HOME=/home/{user} sh -lc {cq}; "
                f"else su -s /bin/bash {uq} -c {cq}; fi"
            )
            docker_cmd = ["docker", "exec", self._container, "sh", "-lc", launcher]
        else:
            docker_cmd = ["docker", "exec", "-u", "root", self._container, "sh", "-lc", inner]

        code, out, err = await self._run_host(docker_cmd, timeout=timeout, check=False)
        if check and code != 0:
            raise RuntimeError(f"docker exec failed (exit={code}): {cmd[:120]}\n{err[:400]}")
        return code, out, err

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        import shlex
        import tempfile

        if not self._container:
            raise RuntimeError("DockerSandbox is not started")

        parent = str(Path(sandbox_path).parent)
        await self.exec(f"mkdir -p {shlex.quote(parent)}", user="root", check=True, timeout=60)

        tmp_path: str | None = None
        try:
            if isinstance(content, Path):
                host_src = str(content)
            else:
                data = content.encode() if isinstance(content, str) else content
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                host_src = tmp_path

            await self._run_host(
                ["docker", "cp", host_src, f"{self._container}:{sandbox_path}"],
                timeout=600,
                check=True,
            )
            if user and user != "root":
                await self.exec(f"chown {shlex.quote(user)} {shlex.quote(sandbox_path)}", user="root", check=False)
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        import shlex

        try:
            code, out, _ = await self.exec(f"cat {shlex.quote(sandbox_path)}", user=user, check=False)
            return out if code == 0 else ""
        except Exception:
            return ""

    async def _run_host(self, cmd: list[str], *, timeout: int, check: bool) -> ExecResult:
        import subprocess

        def _call() -> ExecResult:
            try:
                p = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"host cmd timed out after {timeout}s: {cmd[:6]}...") from e
            out = (p.stdout or b"").decode("utf-8", "replace")
            err = (p.stderr or b"").decode("utf-8", "replace")
            if check and p.returncode != 0:
                raise RuntimeError(f"host cmd failed (exit={p.returncode}): {' '.join(cmd[:6])}...\n{err[:400]}")
            return p.returncode, out, err

        return await asyncio.to_thread(_call)


def make_sandbox(image: str, *, timeout: int | None = None) -> Sandbox:
    """Construct the configured sandbox backend (``E2BSandbox`` or ``DockerSandbox``).

    ``SLIME_AGENT_SANDBOX_BACKEND``:

    * ``e2b`` (default) — remote E2B / E2B-compatible gateway
    * ``docker`` / ``local`` — local Docker engine
    """
    backend = (_getenv("SLIME_AGENT_SANDBOX_BACKEND", default="e2b") or "e2b").lower()
    if backend in ("docker", "local"):
        return DockerSandbox(image, timeout=timeout)
    if backend in ("e2b", ""):
        return E2BSandbox(image, timeout=timeout)
    raise ValueError(f"Unknown SLIME_AGENT_SANDBOX_BACKEND={backend!r}; expected 'e2b' or 'docker'")
