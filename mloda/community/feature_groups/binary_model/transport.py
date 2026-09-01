"""Process transport for the binary-model mixin: the minimal subprocess environment, the private
per-invocation directory, and running the binary itself over stdin/stdout or file transport
(contract: Invocation, License, Data handling).
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess  # nosec
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

from mloda.community.feature_groups.binary_model.errors import (
    BinaryTerminatedError,
    BinaryUnavailableError,
    BinaryUsageError,
    OutputContractError,
    error_from_exit,
)

logger = logging.getLogger(__name__)

TEMP_PARENT_NAME = "mloda-binary"

_SIBLING_PID_PATTERN = re.compile(r"^(\d+)-")


def minimal_environment(
    *,
    license_file: str | None = None,
    license_key: str | None = None,
    source_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the minimal environment passed to the binary (contract: Data handling): ``PATH``, a
    fixed UTF-8 locale on POSIX, ``SYSTEMROOT`` on Windows when present, and the license
    variables (an explicit argument wins over ``source_env``, which itself defaults to
    ``os.environ``). ``MLODA_LICENSE_FILE`` is absolutized against the caller's own cwd, since the
    binary itself runs with its private invocation directory as its cwd."""
    source = os.environ if source_env is None else source_env
    env: dict[str, str] = {"PATH": source.get("PATH") or os.defpath}

    if os.name == "nt":
        systemroot = source.get("SYSTEMROOT")
        if systemroot is not None:
            env["SYSTEMROOT"] = systemroot
    else:
        env["LC_ALL"] = "C.UTF-8"
        env["LANG"] = "C.UTF-8"

    resolved_file = license_file if license_file is not None else source.get("MLODA_LICENSE_FILE")
    if resolved_file:
        env["MLODA_LICENSE_FILE"] = os.path.abspath(resolved_file)

    resolved_key = license_key if license_key is not None else source.get("MLODA_LICENSE_KEY")
    if resolved_key:
        env["MLODA_LICENSE_KEY"] = resolved_key

    return env


def pid_is_alive(pid: int) -> bool:
    """Whether ``pid`` names a live process (contract: Data handling, orphan detection)."""
    if os.name == "nt":
        # os.kill on Windows terminates the target process rather than merely signalling it, so a
        # liveness check must never call it there; report conservatively alive to never reap.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class InvocationDirectory:
    """A private, owner-only directory for one binary invocation, created under a fixed parent
    and reaping dead siblings on entry (contract: Data handling)."""

    def __init__(self, parent: Path | None = None) -> None:
        self.parent = parent if parent is not None else Path(tempfile.gettempdir()) / TEMP_PARENT_NAME
        self.path: Path

    def __enter__(self) -> InvocationDirectory:
        self.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt":
            self._validate_parent()

        self._reap_dead_siblings()

        name = f"{os.getpid()}-{secrets.token_hex(4)}"
        path = self.parent / name
        path.mkdir(mode=0o700)
        path.chmod(0o700)
        self.path = path
        return self

    def _validate_parent(self) -> None:
        """Refuse a parent not owned by the current user, world-writable, or writable by a group
        other than the current process's own (contract: Data handling): a directory shared with
        the process's own group, the common user-private-group scheme, is not a foreign-write
        risk, but world-writable or a foreign group is."""
        stat_result = self.parent.stat()
        if stat_result.st_uid != os.getuid():
            raise BinaryUnavailableError(f"refusing to use {self.parent}: not owned by the current user")
        if stat_result.st_mode & 0o002:
            raise BinaryUnavailableError(f"refusing to use {self.parent}: world writable")
        if stat_result.st_mode & 0o020 and stat_result.st_gid != os.getgid():
            raise BinaryUnavailableError(f"refusing to use {self.parent}: writable by a group other than our own")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    def _reap_dead_siblings(self) -> None:
        try:
            entries = list(self.parent.iterdir())
        except OSError:
            return
        for entry in entries:
            match = _SIBLING_PID_PATTERN.match(entry.name)
            if match is None:
                continue
            pid = int(match.group(1))
            if pid_is_alive(pid):
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                try:
                    os.unlink(entry)
                except OSError:
                    continue


def _find_offending_parameter_key(config: Mapping[str, Any]) -> str | None:
    """The key of the first ``parameters`` entry that is not JSON-serializable, found by
    serializing each entry one by one so the offending value itself is never included in a message
    (contract: Data handling)."""
    parameters = config.get("parameters")
    if not isinstance(parameters, Mapping):
        return None
    for key, value in parameters.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            return str(key)
    return None


def _terminate_timed_out_process(proc: subprocess.Popen[bytes]) -> None:
    """Terminate a hung binary after ``communicate`` times out (contract: Errors, Data handling):
    on POSIX, the whole process group started with the child, via ``start_new_session=True``, so a
    descendant it spawned does not outlive it; on Windows, the child process alone."""
    if os.name == "nt":
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run_binary(
    argv: Sequence[str],
    env: Mapping[str, str],
    config: Mapping[str, Any],
    input_bytes: bytes,
    *,
    timeout: float | None,
    file_transport_threshold: int,
    invocation_dir: Path,
) -> bytes:
    """Run one ``run --config <path>`` invocation, choosing stdin/stdout or file transport based
    on ``input_bytes`` size, and returning the output bytes (contract: Invocation, Data)."""
    config_path = invocation_dir / "config.json"
    try:
        payload = json.dumps(dict(config))
    except (TypeError, ValueError) as exc:
        offending_key = _find_offending_parameter_key(config)
        if offending_key is not None:
            raise BinaryUsageError(f"parameter {offending_key!r} is not JSON-serializable") from exc
        raise BinaryUsageError(f"config contains a value that is not JSON-serializable: {exc}") from exc

    fd = os.open(str(config_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)

    args = ["run", "--config", str(config_path)]
    output_path: Path | None = None
    if len(input_bytes) > file_transport_threshold:
        input_path = invocation_dir / "input.arrows"
        input_path.write_bytes(input_bytes)
        output_path = invocation_dir / "output.arrows"
        args += ["--input", str(input_path), "--output", str(output_path)]
        stdin_bytes = b""
    else:
        stdin_bytes = input_bytes

    try:
        proc = subprocess.Popen(  # nosec B603
            [*argv, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env),
            cwd=str(invocation_dir),
            start_new_session=os.name != "nt",
        )
    except (PermissionError, FileNotFoundError) as exc:
        raise BinaryUnavailableError(f"cannot spawn binary {argv[0]!r}: {exc}") from exc

    try:
        stdout, stderr = proc.communicate(stdin_bytes, timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_timed_out_process(proc)
        raise BinaryTerminatedError(f"binary timed out after {timeout}s and was terminated")

    logger.debug("binary exited with code %s", proc.returncode)

    if proc.returncode != 0:
        raise error_from_exit(proc.returncode, stderr)

    if output_path is not None:
        try:
            return output_path.read_bytes()
        except OSError as exc:
            raise OutputContractError("binary exited 0 but wrote no --output file") from exc
    return stdout
