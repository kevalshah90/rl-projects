"""Modal sandboxes as a verifiers runtime plugin (for prime-rl's verifiers, a12316c and later).

Why it exists: verifiers removed its built-in Modal runtime in 682e8dfb1 (2026-10-19); other
providers now load as plugins. Our agent and grader sandboxes run on Modal (A10G, the task
images), as in the baseline and dev eval E0, so training keeps the same hardware this way.

This is verifiers' own Modal runtime as of da75527 (the version our baseline/E0 ran with),
moved onto the new plugin contract. Behavior is unchanged; the interface changes are:
    old `Runtime` subclass, type = "modal"   ->  `RuntimeBackend` + `CustomRuntimeConfig`,
                                                 type = "custom", import_path = this class
    open_process(argv, env)                 ->  exec(runtime, argv, env)
    run / write / read / run_background     ->  take the `Runtime` view first (env bindings)
    read: the base class honors max_bytes   ->  same: native read only for whole files
    prepare_execution(routes)               ->  prepare_execution(routes, network=None);
                                                 `network` replaces our policy (grader box)

Use (host side; this module must be importable wherever the config is loaded):
    [env.agent.runtime]
    type = "custom"
    import_path = "modal_runtime.ModalBackend"
"""

import asyncio
import contextlib

import re
import shlex
import uuid
from collections.abc import AsyncIterator
from ipaddress import ip_address
from pathlib import PurePosixPath
from typing import ClassVar

from pydantic import model_validator

from verifiers.v1.configs.runtime import CustomRuntimeConfig, NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.backends.custom import CustomRuntimeInfo
from verifiers.v1.runtimes.base import (
    SERVICE_PORT,
    ProgramResult,
    RuntimeBackend,
    RuntimeProcess,
)
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.logging import get_logger
from verifiers.v1.utils.network import parse_network_rule
from verifiers.v1.utils.scope import run_scope

# Under verifiers' logger namespace, so vf-run / prime-rl print our lines (they configure
# only `verifiers.*`). Each sandbox start is logged with its image, GPU and network policy,
# which is how a run shows the agent box and the separate grader box.
logger = get_logger("verifiers.v1.runtimes.modal_plugin")

_APP_NAME = "verifiers-v1"   # same shared Modal app the old runtime attached sandboxes to


def _egress_domain(rule: str, *, framework: bool = False) -> str | None:
    """Translate only domains Modal can filter without broadening the rule (unchanged)."""
    parsed, host, port = parse_network_rule(rule)
    try:
        address = ip_address(host)
    except ValueError:
        address = None
    # Colocated tools are container-local; they need no external network grant.
    if (
        framework
        and parsed.scheme in ("http", "https")
        and (host == "localhost" or (address is not None and address.is_loopback))
    ):
        return None
    if (
        parsed.scheme not in (("https",) if framework else ("", "https"))
        or port not in (None, 443)
        or parsed.username is not None
        or address is not None
        or not re.fullmatch(
            r"(?:\*\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
            r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
            host,
        )
        or (framework and "*" in host)
    ):
        raise ValueError(
            f"Modal egress rule {rule!r} is unsupported; use DNS names or HTTPS "
            "origins on port 443 (only a leading *. wildcard is supported)"
        )
    return host


class ModalConfig(CustomRuntimeConfig, NetworkPolicyConfig):
    # The fields of the old ModalConfig. image/workdir are required for task images, and
    # cpu/memory/gpu/disk are where a task's [environment] resources land (same names as
    # TaskResources), e.g. gpu = "A10G:1" from task.toml.
    import_path: str = "modal_runtime.ModalBackend"
    vm: bool = False
    """Use a VM sandbox for workloads requiring a Docker daemon."""
    image: str = "python:3.11-slim"
    workdir: str | None = None
    """Working directory override; None uses the task's workdir, or /app."""
    network_access: bool = True
    """Allow network access at creation. False blocks all egress, including setup;
    it cannot be combined with execution-time allow/block policies."""
    region: str | None = None
    cpu: float = 1.0
    memory: float = 2.0
    """Memory in GB."""
    gpu: str | None = None
    """GPU spec, e.g. "A10G" or "A10G:1"."""
    disk: float = 5.0
    """Accepted so tasks can declare it; Modal sandboxes have no disk knob."""
    creates_per_sec: float | None = 40.0
    """Pace sandbox creation run-wide (None/<= 0 disables it)."""
    sandbox_timeout: int = 2 * 60 * 60
    """Seconds before Modal kills the sandbox no matter what (was a fixed 24 h). Normally our
    teardown terminates it within seconds; this caps the bill when the owning process is
    hard-killed and teardown never runs. Must exceed agent (900 s) + grading time."""

    @model_validator(mode="after")
    def _validate_egress(self) -> "ModalConfig":
        if not self.network_restricted:
            return self
        if not self.network_access:
            raise ValueError(
                "Modal allow/block policies require network_access=true for trusted setup"
            )
        if self.allow == ["*"]:
            raise ValueError(
                "Modal does not support egress deny lists; use an allowlist or allow=[]"
            )
        for rule in self.allow:
            _egress_domain(rule)
        return self

    @model_validator(mode="after")
    def _validate_vm(self) -> "ModalConfig":
        if self.vm and self.gpu:
            raise ValueError("Modal VM sandboxes support CPU workloads only")
        return self


class ModalProcess(RuntimeProcess):
    """A process started with `sandbox.exec`; signals go through `kill` by PID (unchanged)."""

    def __init__(self, process, sandbox, pid: int, workdir: str) -> None:
        self._process = process
        self._sandbox = sandbox
        self._pid = pid
        self._workdir = workdir
        self.stdout: AsyncIterator[bytes] = process.stdout
        self.stderr: AsyncIterator[bytes] = process.stderr

    async def write(self, data: bytes) -> None:
        self._process.stdin.write(data)
        await self._process.stdin.drain.aio()

    async def wait(self) -> int:
        return await self._process.wait.aio()

    async def poll(self) -> int | None:
        return await self._process.poll.aio()

    async def terminate(self) -> None:
        await self._signal("TERM")

    async def kill(self) -> None:
        await self._signal("KILL")

    async def _signal(self, signal: str) -> None:
        if await self._process.poll.aio() is not None:
            return
        proc = await self._sandbox.exec.aio(
            "sh", "-c", 'kill -"$1" "$2"', "vf-signal", signal, str(self._pid),
            workdir=self._workdir,
        )
        stderr = await proc.stderr.read.aio()
        exit_code = await proc.wait.aio()
        if exit_code != 0 and await self._process.poll.aio() is None:
            raise SandboxError(f"modal exec process signal failed: {stderr.strip()}")


class ModalBackend(RuntimeBackend):
    config_class = ModalConfig
    is_local: ClassVar[bool] = False
    # What the old runtime did, now declared: boots task images (needed for isolated
    # grading) and forwards SERVICE_PORT out. No user switching, no Compose (as before).
    SUPPORTS_TASK_IMAGES: ClassVar[bool] = True
    SUPPORTS_PORT_EXPOSURE: ClassVar[bool] = True

    def __init__(self, config: ModalConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config.model_copy(update={"workdir": config.workdir or "/app"})
        self.info = CustomRuntimeInfo(**self.config.model_dump())
        self._sandbox = None

    @property
    def published_port(self) -> int | None:
        return SERVICE_PORT

    async def start(self) -> None:
        import modal

        try:
            app = await modal.App.lookup.aio(_APP_NAME, create_if_missing=True)
            async with (
                creation_limiter(self.config.creates_per_sec, "modal-sandbox", run_scope())
                or contextlib.nullcontext()
            ):
                await run_shielded(self._create_sandbox(app))
            self.info.id = self._sandbox.object_id
            logger.info("modal: sandbox %s up (name=%s image=%s gpu=%s allow=%s)",
                        self.info.id, self.name, self.config.image, self.config.gpu,
                        self.config.allow)
            await self._sandbox.filesystem.make_directory.aio(self.config.workdir)
        except Exception as e:  # provisioning failure is one rollout's problem, not the run's
            raise SandboxError(f"modal sandbox provisioning failed: {e}") from e

    async def _create_sandbox(self, app) -> None:
        """Create the sandbox and take ownership of the handle in one shielded step, so a
        cancellation can't leave a billed sandbox nobody tears down (unchanged)."""
        import modal

        # Modal requires both allowlist types at creation before they can be updated.
        # Trusted setup runs open; prepare_execution removes the broad CIDR grant.
        self._sandbox = await modal.Sandbox.create.aio(
            "sleep", "infinity",  # keep-alive entrypoint; the harness runs via `exec`
            app=app,
            name=self.name,
            image=modal.Image.from_registry(self.config.image).entrypoint([]),
            workdir=self.config.workdir,
            env=self.env,
            cpu=self.config.cpu,
            memory=int(self.config.memory * 1024),  # Modal memory is MB
            gpu=self.config.gpu,
            region=self.config.region,
            block_network=not self.config.network_access,
            outbound_domain_allowlist=["*"] if self.network_restricted else None,
            outbound_cidr_allowlist=["0.0.0.0/0"] if self.network_restricted else None,
            timeout=self.config.sandbox_timeout,
            encrypted_ports=[SERVICE_PORT],
            experimental_options={"vm_runtime": True} if self.config.vm else {},
        )

    async def prepare_execution(
        self, routes: list[str] | None, network: NetworkPolicyConfig | None = None
    ) -> None:
        """Apply TLS domain filtering after setup, keeping framework endpoints (`routes`).
        `network` (new) replaces our own policy, e.g. a grader box's scoring network.
        Modal filters TLS SNI on port 443, not HTTP Host headers or URL paths."""
        if not self.network_restricted:
            return
        policy = network or self.config
        try:
            if routes is None:
                domains, cidrs = ["*"], ["0.0.0.0/0"]
            else:
                domains = [_egress_domain(route, framework=True) for route in routes]
                domains.extend(_egress_domain(rule) for rule in policy.allow)
                domains = list(dict.fromkeys(d for d in domains if d is not None))
                cidrs = []
            # Always send both lists: leaving the setup CIDR grant would bypass domains.
            async with asyncio.timeout(60):
                await self._sandbox._experimental_set_outbound_network_policy.aio(
                    outbound_domain_allowlist=domains,
                    outbound_cidr_allowlist=cidrs,
                )
        except Exception as e:
            raise SandboxError(f"modal egress policy failed: {e}") from e
        logger.info("modal: egress policy on sandbox %s (domains=%s cidrs=%s)",
                    self.info.id, domains, cidrs)

    async def expose(self, port: int) -> str | None:
        # Only SERVICE_PORT (declared at creation) has a Modal tunnel.
        if self._sandbox is None:
            return None
        try:
            tunnels = await self._sandbox.tunnels.aio()
        except Exception as e:
            raise SandboxError(f"modal tunnels unavailable (port {port}): {e}") from e
        tunnel = tunnels.get(port)
        return str(tunnel.url).rstrip("/") if tunnel else None

    async def run(self, runtime: Runtime, argv: list[str], env: dict[str, str]) -> ProgramResult:
        # Close stdin (as the base class's run does): a command that reads stdin, e.g. a bare
        # `cat` or `python`, must see EOF and return, not wait forever. The old Modal runtime
        # passed argv straight to exec and left stdin open; verifiers' test_run caught it.
        try:
            proc = await self._sandbox.exec.aio(
                "sh", "-c", 'exec "$@" < /dev/null', "sh", *argv,
                workdir=self.config.workdir, env=runtime.process_env(env),
            )
            # Drain both pipes concurrently so a large stderr can't deadlock stdout.
            stdout, stderr = await asyncio.gather(proc.stdout.read.aio(), proc.stderr.read.aio())
            await proc.wait.aio()
        except Exception as e:  # a sandbox/API failure is one rollout's problem
            raise SandboxError(f"modal exec failed: {e}") from e
        return ProgramResult(exit_code=proc.returncode or 0, stdout=stdout or "", stderr=stderr or "")

    async def exec(self, runtime: Runtime, argv: list[str], env: dict[str, str]) -> RuntimeProcess:
        """Start argv streaming its stdio. The shell wrapper records the PID, because Modal's
        process handle has no signal API (terminate/kill send `kill PID`)."""
        pidfile = f".vf-process-{uuid.uuid4().hex}.pid"
        wrapper = 'echo $$ > "$1"; shift; exec "$@"'
        try:
            proc = await self._sandbox.exec.aio(
                "sh", "-c", wrapper, "vf-process", pidfile, *argv,
                workdir=self.config.workdir, env=runtime.process_env(env), text=False,
            )
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 5
            while True:
                ready = await self.run(runtime, ["cat", pidfile], {})
                if ready.exit_code == 0 and ready.stdout.strip().isdigit():
                    return ModalProcess(proc, self._sandbox, int(ready.stdout.strip()),
                                        self.config.workdir)
                returncode = await proc.poll.aio()
                if returncode is not None or loop.time() >= deadline:
                    if returncode is None:
                        # A live process without a readable PID can't be signalled:
                        # fail the sandbox closed instead of abandoning it.
                        sandbox = self._sandbox
                        try:
                            await sandbox.terminate.aio()
                        except Exception:
                            logger.warning("modal: failed to terminate sandbox %s after live "
                                           "process startup timed out", self.info.id, exc_info=True)
                        else:
                            self._sandbox = None
                    raise SandboxError("modal live process failed to start: "
                                       f"{ready.stderr.strip() or 'PID unavailable'}")
                await asyncio.sleep(0.05)
        except SandboxError:
            raise
        except Exception as e:
            raise SandboxError(f"modal live process failed to start: {e}") from e

    async def run_background(self, runtime: Runtime, argv: list[str], env: dict[str, str],
                             log: str) -> None:
        # Detached with nohup: lives until the sandbox is terminated, holds no connection.
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 &"
        result = await self.run(runtime, ["sh", "-c", inner], env)
        if result.exit_code != 0:
            raise SandboxError(f"modal background launch failed: {result.stderr.strip()}")

    def _abs(self, path: str) -> str:
        # Modal's filesystem API only accepts absolute paths; resolve against the workdir.
        if path.startswith("/"):
            return path
        return f"{self.config.workdir.rstrip('/')}/{path}"

    async def read(self, runtime: Runtime, path: str, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None:   # the base class reads at most max_bytes at the source
            return await super().read(runtime, path, max_bytes)
        try:
            return await self._sandbox.filesystem.read_bytes.aio(self._abs(path))
        except Exception as e:
            raise SandboxError(f"read {path!r}: {e}") from e

    async def write(self, runtime: Runtime, path: str, data: bytes) -> None:
        target = self._abs(path)   # Modal's write does not create parent directories
        try:
            await self._sandbox.filesystem.make_directory.aio(str(PurePosixPath(target).parent))
            await self._sandbox.filesystem.write_bytes.aio(data, target)
        except Exception as e:
            raise SandboxError(f"write {path!r}: {e}") from e

    def cleanup(self) -> None:
        # Synchronous atexit backstop: terminate so a paid sandbox isn't left to its 24h limit.
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:
            with contextlib.suppress(Exception):
                sandbox.terminate()

    async def teardown(self) -> None:
        # Normal path (shielded by `stop`). `_sandbox` is cleared only after terminate, so the
        # atexit backstop can still find it if the loop dies mid-await.
        sandbox = self._sandbox
        if sandbox is None:
            return
        try:
            await sandbox.terminate.aio()
        except Exception as e:  # noqa: BLE001 - provider teardown is best-effort
            logger.warning("modal: failed to terminate sandbox %s: %s", self.info.id, e)
        self._sandbox = None
