"""Process transport for the binary-model mixin: the minimal subprocess environment, the private
per-invocation directory, and running the binary itself over stdin/stdout or file transport
(contract: Invocation, License, Data handling).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess  # nosec
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO

from mloda.community.feature_groups.binary_model.contract import stderr_excerpt
from mloda.community.feature_groups.binary_model.errors import (
    BinaryTerminatedError,
    BinaryUnavailableError,
    BinaryUsageError,
    OutputContractError,
    error_from_exit,
)

if os.name != "nt":
    import fcntl

logger = logging.getLogger(__name__)

# Paths of live directories owned by this process. The reaper must not even open their lock files:
# flock may be emulated as per-process record locks (e.g. NFS), so closing any fd drops the lock.
_OWNED_PATHS: set[Path] = set()
_OWNED_LOCK = threading.Lock()

TEMP_PARENT_NAME = "mloda-binary"
LOCK_FILE_NAME = ".lock"

_SIBLING_PID_PATTERN = re.compile(r"^(\d+)-")
_STAGING_PATTERN = re.compile(r"^\.tmp-[0-9a-f]+$")
LOCK_INIT_FILE_NAME = ".lock-init"
# A lock-less staging dir is held for microseconds by a live owner; older ones are abandoned.
_STAGING_GRACE_SECONDS = 600.0


def _windows_pid_alive(pid: int) -> bool:
    """Whether ``pid`` is a live process on Windows; failures other than a missing pid read alive."""
    if sys.platform != "win32":
        return True
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: no such process
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _is_reparse_point(path: Path) -> bool:
    """Whether ``path`` is a Windows reparse point (symlink or junction)."""
    return bool(getattr(os.lstat(path), "st_file_attributes", 0) & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT


def _windows_owned_by_current_user(path: Path) -> bool:
    """Whether ``path`` is owned by the current user (token user or owner SID), Administrators or
    LocalSystem on Windows; raises ``OSError`` when an API call fails."""
    if sys.platform != "win32":
        return True
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    pvoid = ctypes.c_void_p
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(pvoid),
        ctypes.POINTER(pvoid),
        ctypes.POINTER(pvoid),
        ctypes.POINTER(pvoid),
        ctypes.POINTER(pvoid),
    ]
    kernel32.LocalFree.restype = pvoid
    kernel32.LocalFree.argtypes = [pvoid]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        pvoid,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.CreateWellKnownSid.restype = wintypes.BOOL
    advapi32.CreateWellKnownSid.argtypes = [wintypes.DWORD, pvoid, pvoid, ctypes.POINTER(wintypes.DWORD)]
    advapi32.EqualSid.restype = wintypes.BOOL
    advapi32.EqualSid.argtypes = [pvoid, pvoid]

    def token_sid(token: Any, info_class: int) -> tuple[Any, Any]:
        """The SID pointer of a token information class, plus its backing buffer."""
        size = wintypes.DWORD(0)
        advapi32.GetTokenInformation(token, info_class, None, 0, ctypes.byref(size))  # sizing call, fails by design
        if size.value == 0:
            raise ctypes.WinError(ctypes.get_last_error())
        buf = ctypes.create_string_buffer(size.value)
        if not advapi32.GetTokenInformation(token, info_class, buf, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        return ctypes.cast(buf, ctypes.POINTER(pvoid))[0], buf

    def well_known_sid(sid_type: int) -> Any:
        buf = ctypes.create_string_buffer(68)  # SECURITY_MAX_SID_SIZE
        size = wintypes.DWORD(68)
        if not advapi32.CreateWellKnownSid(sid_type, None, buf, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        return buf

    owner = pvoid()
    descriptor = pvoid()
    error = advapi32.GetNamedSecurityInfoW(
        str(path), 1, 1, ctypes.byref(owner), None, None, None, ctypes.byref(descriptor)
    )  # SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION
    if error:
        raise OSError(None, ctypes.FormatError(error), str(path), error)
    try:
        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):  # TOKEN_QUERY
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            user_sid, user_buf = token_sid(token, 1)  # TokenUser
            owner_sid, owner_buf = token_sid(token, 4)  # TokenOwner
        finally:
            kernel32.CloseHandle(token)
        admins = well_known_sid(26)  # WinBuiltinAdministratorsSid
        system = well_known_sid(22)  # WinLocalSystemSid
        trusted = (user_sid, owner_sid, ctypes.addressof(admins), ctypes.addressof(system))
        return any(bool(advapi32.EqualSid(owner.value, sid)) for sid in trusted)
    finally:
        kernel32.LocalFree(descriptor)


def minimal_environment(
    *,
    license_file: str | None = None,
    license_key: str | None = None,
    source_env: Mapping[str, str] | None = None,
    inherit_license: bool = True,
) -> dict[str, str]:
    """Build the minimal environment passed to the binary (contract: Data handling): ``PATH``, a
    fixed UTF-8 locale on POSIX, ``SYSTEMROOT`` on Windows when present, and the license
    variables (an explicit argument wins over the value inherited from ``source_env``, which
    itself defaults to ``os.environ``). ``source_env`` also supplies ``PATH`` and ``SYSTEMROOT``,
    so ``source_env={}`` drops them too; ``inherit_license=False`` (the canonical spelling; an
    explicit empty ``license_file`` / ``license_key`` does the same) suppresses only the license
    variables. A non-empty explicit ``license_file`` or ``license_key`` means neither variable is
    inherited (the binary reads the file first, so an inherited file would mask an explicit key).
    ``MLODA_LICENSE_FILE`` is absolutized against the caller's own cwd, since the
    binary itself runs with its private invocation directory as its cwd."""
    source = os.environ if source_env is None else source_env
    explicit = bool(license_file) or bool(license_key)
    inherited: Mapping[str, str] = source if inherit_license and not explicit else {}
    env: dict[str, str] = {"PATH": source.get("PATH") or os.defpath}

    if os.name == "nt":
        systemroot = source.get("SYSTEMROOT")
        if systemroot is not None:
            env["SYSTEMROOT"] = systemroot
    else:
        env["LC_ALL"] = "C.UTF-8"
        env["LANG"] = "C.UTF-8"

    resolved_file = license_file if license_file is not None else inherited.get("MLODA_LICENSE_FILE")
    if resolved_file:
        env["MLODA_LICENSE_FILE"] = os.path.abspath(resolved_file)

    resolved_key = license_key if license_key is not None else inherited.get("MLODA_LICENSE_KEY")
    if resolved_key:
        env["MLODA_LICENSE_KEY"] = resolved_key

    return env


def default_parent() -> Path:
    """The default parent directory: per-user on POSIX so users never share one."""
    base = Path(tempfile.gettempdir())
    if os.name == "nt":
        return base / TEMP_PARENT_NAME
    return base / f"{TEMP_PARENT_NAME}-{os.getuid()}"


class InvocationDirectory:
    """A private, owner-only directory for one binary invocation, created under a per-user parent
    (or the given one) and reaping dead siblings on entry (contract: Data handling). Liveness is an exclusive
    ``flock`` on a lock file inside the directory, held until exit (POSIX), or the owner pid (Windows)."""

    def __init__(self, parent: Path | None = None) -> None:
        self.parent = parent if parent is not None else default_parent()
        self.path: Path
        self._lock_fd: int | None = None

    def __enter__(self) -> InvocationDirectory:
        try:
            self.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise BinaryUnavailableError(f"cannot create {self.parent}: {exc}") from exc
        self._validate_parent()

        self._reap_dead_siblings()

        name = f"{os.getpid()}-{secrets.token_hex(4)}"
        path = self.parent / name
        if os.name == "nt":
            with _OWNED_LOCK:
                _OWNED_PATHS.add(path)
            try:
                path.mkdir(mode=0o700)
                path.chmod(0o700)
            except BaseException as exc:
                with _OWNED_LOCK:
                    _OWNED_PATHS.discard(path)
                shutil.rmtree(path, ignore_errors=True)
                if isinstance(exc, OSError):
                    raise BinaryUnavailableError(
                        f"cannot create an invocation directory under {self.parent}: {exc}"
                    ) from exc
                raise
            self.path = path
            return self
        # Staged under a non-pid name so a reaper never sees a pid-named dir before it is locked.
        staging = self.parent / f".tmp-{secrets.token_hex(8)}"
        try:
            with _OWNED_LOCK:
                _OWNED_PATHS.add(staging)
                _OWNED_PATHS.add(path)
            staging.mkdir(mode=0o700)
            staging.chmod(0o700)
            self._lock_fd = _lock_file_fd(staging)
            try:
                os.rename(staging, path)
            finally:
                with _OWNED_LOCK:
                    _OWNED_PATHS.discard(staging)
        except BaseException as exc:
            with _OWNED_LOCK:
                _OWNED_PATHS.discard(path)
                _OWNED_PATHS.discard(staging)
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            shutil.rmtree(staging, ignore_errors=True)
            if isinstance(exc, OSError):
                raise BinaryUnavailableError(f"cannot lock an invocation directory under {self.parent}: {exc}") from exc
            raise
        self.path = path
        return self

    def _validate_parent(self) -> None:
        """On Windows refuse a parent that is a symlink or junction or not owned by the current user.
        On POSIX refuse a parent that is a symlink, not owned by the current user, world-writable, or
        writable by a group other than the current process's own (contract: Data handling): a
        directory shared with the process's own group, the common user-private-group scheme, is
        not a foreign-write risk, but world-writable or a foreign group is."""
        if os.name == "nt":
            if _is_reparse_point(self.parent):
                raise BinaryUnavailableError(f"refusing to use {self.parent}: a symlink or junction")
            try:
                owned = _windows_owned_by_current_user(self.parent)
            except OSError as exc:
                raise BinaryUnavailableError(f"refusing to use {self.parent}: cannot read its owner: {exc}") from exc
            if not owned:
                raise BinaryUnavailableError(
                    f"refusing to use {self.parent}: not owned by the current user; delete it so it is recreated"
                )
            return
        stat_result = os.lstat(self.parent)
        if stat.S_ISLNK(stat_result.st_mode):
            raise BinaryUnavailableError(f"refusing to use {self.parent}: a symlink")
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
        with _OWNED_LOCK:
            _OWNED_PATHS.discard(self.path)
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def _reap_dead_siblings(self) -> None:
        try:
            entries = list(self.parent.iterdir())
        except OSError:
            return
        for entry in entries:
            match = _SIBLING_PID_PATTERN.match(entry.name)
            is_staging = os.name != "nt" and _STAGING_PATTERN.match(entry.name) is not None
            if match is None and not is_staging:
                continue
            try:
                is_dir = stat.S_ISDIR(os.lstat(entry).st_mode)
            except OSError:
                continue
            if match is None:
                if is_dir:
                    _reap_if_unlocked(entry, require_lock=True)
                continue
            if not is_dir:
                try:
                    os.unlink(entry)
                except OSError:
                    pass
                continue
            if os.name == "nt":
                # Liveness is by pid here, so a TEMP shared across hosts is unsupported.
                with _OWNED_LOCK:
                    if entry in _OWNED_PATHS:
                        continue
                pid = int(match.group(1))
                if pid == os.getpid() or not _windows_pid_alive(pid):
                    shutil.rmtree(entry, ignore_errors=True)
                continue
            _reap_if_unlocked(entry)


def _lock_file_fd(directory: Path) -> int:
    """Create, flock and rename ``.lock-init`` to ``.lock``, so ``.lock`` is never visible unlocked."""
    init_path = directory / LOCK_INIT_FILE_NAME
    fd = os.open(str(init_path), os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.rename(init_path, directory / LOCK_FILE_NAME)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _reap_if_unlocked(entry: Path, *, require_lock: bool = False) -> None:
    """Remove ``entry`` if its lock can be taken (owner dead) or it has no lock file (staging
    guarantees a live directory is locked once visible); keep it when locked or unreadable. With
    ``require_lock`` a directory without a lock file is kept until it is older than the staging grace period."""
    with _OWNED_LOCK:
        if entry in _OWNED_PATHS:
            return
    try:
        fd = os.open(str(entry / LOCK_FILE_NAME), os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        if require_lock:
            try:
                stale = time.time() - os.lstat(entry).st_mtime > _STAGING_GRACE_SECONDS
            except OSError:
                return
            if not stale:
                return
        shutil.rmtree(entry, ignore_errors=True)
        return
    except OSError:
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return
        shutil.rmtree(entry, ignore_errors=True)
    finally:
        os.close(fd)


def _find_offending_parameter_key(config: Mapping[str, Any]) -> str | None:
    """The key of the first ``parameters`` entry that is not JSON-serializable, found by
    serializing each entry one by one so the offending value itself is never included in a message
    (contract: Data handling)."""
    parameters = config.get("parameters")
    if not isinstance(parameters, Mapping):
        return None
    for key, value in parameters.items():
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            return str(key)
    return None


_GRACE_WAIT_SECONDS = 1.0
_REAP_WAIT_SECONDS = 5.0
_STDERR_WARNING_LINES = 5


def _close_posix_pipes(proc: subprocess.Popen[bytes]) -> None:
    """Close the parent's own pipe ends to the child, after the reap whether or not it completed
    (contract: Data handling); POSIX only, since on Windows a reader thread may still block in
    ``read()`` and closing there can hang."""
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass


def _stop_process(proc: subprocess.Popen[bytes], *, hard: bool) -> None:
    if os.name == "nt":
        if hard:
            proc.kill()
        else:
            proc.terminate()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL if hard else signal.SIGTERM)
        except OSError:
            pass


def _terminate_timed_out_process(proc: subprocess.Popen[bytes]) -> None:
    """Terminate a hung binary after ``communicate`` times out, or on any exceptional exit (contract:
    Errors, Data handling): soft-stops then hard-kills the whole process group on POSIX, or just the
    child on Windows. The hard kill always runs, even if the soft stop's grace wait is interrupted,
    and the final reap is bounded, logging a warning rather than blocking forever."""
    try:
        _stop_process(proc, hard=False)
        try:
            proc.wait(timeout=_GRACE_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    finally:
        try:
            _stop_process(proc, hard=True)
            try:
                proc.wait(timeout=_REAP_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning("process %s did not exit after being killed", proc.pid)
        finally:
            if os.name != "nt":
                _close_posix_pipes(proc)


def communicate_or_terminate(
    proc: subprocess.Popen[bytes], input_bytes: bytes | None, timeout: float | None
) -> tuple[bytes, bytes]:
    """Run ``proc.communicate``; on any exception terminate ``proc`` and re-raise: its whole process
    group on POSIX (so ``proc`` must start with ``start_new_session=True``), only the child on Windows
    (contract: Errors, Data handling)."""
    try:
        return proc.communicate(input_bytes, timeout=timeout)
    except BaseException:
        # Unconditional: the process group persists while it has members, so this still reaches a
        # descendant that outlives an already-exited leader; the only residual risk is the same
        # pid-recycling race already accepted on an emptied group.
        _terminate_timed_out_process(proc)
        raise


def run_binary(
    argv: Sequence[str],
    env: Mapping[str, str],
    config: Mapping[str, Any],
    write_input: Callable[[BinaryIO], object],
    input_size: int,
    *,
    timeout: float | None,
    file_transport_threshold: int,
    invocation_dir: Path,
) -> bytes:
    """Run one ``run --config <path>`` invocation, choosing stdin/stdout or file transport based
    on ``input_size``, and returning the output bytes (contract: Invocation, Data). ``write_input`` writes
    the input stream into the given binary file: straight to ``input.arrows`` for file transport (never
    held in memory), else into a buffer sent on stdin."""
    config_path = invocation_dir / "config.json"
    try:
        payload = json.dumps(dict(config), allow_nan=False)
    except (TypeError, ValueError) as exc:
        offending_key = _find_offending_parameter_key(config)
        if offending_key is not None:
            raise BinaryUsageError(f"parameter {offending_key!r} is not JSON-serializable") from exc
        raise BinaryUsageError("config contains a value that is not JSON-serializable") from exc

    args = ["run", "--config", str(config_path)]
    output_path: Path | None = None
    stdin_bytes: bytes
    try:
        fd = os.open(str(config_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        if input_size > file_transport_threshold:
            input_path = invocation_dir / "input.arrows"
            input_fd = os.open(
                str(input_path),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
                0o600,
            )
            with os.fdopen(input_fd, "wb") as input_handle:
                write_input(input_handle)
            output_path = invocation_dir / "output.arrows"
            args += ["--input", str(input_path), "--output", str(output_path)]
            stdin_bytes = b""
    except OSError as exc:
        raise BinaryUnavailableError(f"cannot write to the invocation directory {invocation_dir}: {exc}") from exc
    if output_path is None:
        buffer = io.BytesIO()
        write_input(buffer)
        stdin_bytes = buffer.getvalue()

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
    except OSError as exc:
        raise BinaryUnavailableError(f"cannot spawn binary {argv[0]!r}: {exc}") from exc

    try:
        stdout, stderr = communicate_or_terminate(proc, stdin_bytes, timeout)
    except subprocess.TimeoutExpired:
        raise BinaryTerminatedError(f"binary timed out after {timeout}s and was terminated")

    logger.debug("binary exited with code %s", proc.returncode)

    if proc.returncode != 0:
        raise error_from_exit(proc.returncode, stderr)

    excerpt = stderr_excerpt(stderr, _STDERR_WARNING_LINES)
    if excerpt is not None:
        logger.warning("binary %s wrote to stderr on a successful run: %r", os.path.basename(argv[0]), excerpt)

    if output_path is not None:
        try:
            return output_path.read_bytes()
        except OSError as exc:
            raise OutputContractError("binary exited 0 but wrote no --output file") from exc
    return stdout
