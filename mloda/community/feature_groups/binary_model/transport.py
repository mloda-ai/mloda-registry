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
import subprocess  # nosec
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

from mloda.community.feature_groups.binary_model.errors import (
    BinaryTerminatedError,
    BinaryUnavailableError,
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
    ``os.environ``)."""
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
        env["MLODA_LICENSE_FILE"] = resolved_file

    resolved_key = license_key if license_key is not None else source.get("MLODA_LICENSE_KEY")
    if resolved_key:
        env["MLODA_LICENSE_KEY"] = resolved_key

    return env


def pid_is_alive(pid: int) -> bool:
    """Whether ``pid`` names a live process (contract: Data handling, orphan detection)."""
    if os.name == "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
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
        created_parent = not self.parent.exists()
        self.parent.mkdir(parents=True, exist_ok=True)
        if created_parent:
            self.parent.chmod(0o700)

        self._reap_dead_siblings()

        name = f"{os.getpid()}-{secrets.token_hex(4)}"
        path = self.parent / name
        path.mkdir(mode=0o700)
        path.chmod(0o700)
        self.path = path
        return self

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
            try:
                shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                continue


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
    config_path.write_text(json.dumps(dict(config)), encoding="utf-8")

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
        )
    except (PermissionError, FileNotFoundError) as exc:
        raise BinaryUnavailableError(f"cannot spawn binary {argv[0]!r}: {exc}") from exc

    try:
        stdout, stderr = proc.communicate(stdin_bytes, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise BinaryTerminatedError(f"binary timed out after {timeout}s and was terminated")

    logger.debug("binary exited with code %s", proc.returncode)

    if proc.returncode != 0:
        raise error_from_exit(proc.returncode, stderr)

    if output_path is not None:
        return output_path.read_bytes()
    return stdout
