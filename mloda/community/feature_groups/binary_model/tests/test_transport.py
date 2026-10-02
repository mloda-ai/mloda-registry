"""Tests for ``transport.py``: the minimal subprocess environment, the per-invocation directory,
and ``run_binary`` itself, exercised against the well-behaved simulated binary
(``mloda.testing.binary_model``) for the happy paths and against the deliberately misbehaving
``faulty_binary.py`` fixture for every termination/error path (contract: Invocation, License, Data
handling, Errors).
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import signal
import stat
import subprocess  # nosec
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from mloda.community.feature_groups.binary_model import transport
from mloda.community.feature_groups.binary_model.errors import (
    BinaryInternalError,
    BinaryTerminatedError,
    BinaryUnavailableError,
    BinaryUsageError,
    LicenseInvalidError,
    LicenseMissingError,
    OutputContractError,
    UnsupportedError,
)
from mloda.community.feature_groups.binary_model.tests.process_helpers import (
    kill_descendant_if_running,
    pid_is_alive,
    pid_running,
)
from mloda.community.feature_groups.binary_model.transport import (
    TEMP_PARENT_NAME,
    InvocationDirectory,
    _terminate_timed_out_process,
    minimal_environment,
    run_binary,
)
from mloda.testing.binary_model.arrow import arrow_stream_bytes, read_arrow_stream
from mloda.testing.binary_model.hash_reference import compute_expected_hash_column
from mloda.testing.binary_model.license_vectors import expired_license_token, valid_license_token

if os.name == "posix":
    import fcntl

STUB_CMD = [sys.executable, "-m", "mloda.testing.binary_model.simulated_binary"]
FAULTY_CMD = [sys.executable, "-m", "mloda.community.feature_groups.binary_model.tests.faulty_binary"]
PLUGIN_ID = "example_binary"


def _run_binary(
    argv: Sequence[str],
    env: Mapping[str, str],
    config: Mapping[str, Any],
    input_bytes: bytes,
    invocation_dir: Path,
    *,
    timeout: float | None = 10.0,
    file_transport_threshold: int = 10_000_000,
) -> bytes:
    return run_binary(
        argv,
        env,
        config,
        lambda sink: sink.write(input_bytes),
        len(input_bytes),
        timeout=timeout,
        file_transport_threshold=file_transport_threshold,
        invocation_dir=invocation_dir,
    )


def _hash_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "input_columns": ["col_a"],
        "operation": "hash",
        "parameters": {},
        "output_columns": {"result": "col_a_hash"},
    }
    config.update(overrides)
    return config


def _dead_child_pid() -> int:
    """Spawn and wait for a subprocess so it is fully reaped: its pid is guaranteed dead."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])  # nosec B603
    proc.wait()
    return proc.pid


def _hold_lock(lock_file: Path) -> int:
    """Open ``lock_file`` (creating it) and take an exclusive flock on the returned fd; the caller closes it."""
    fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _as_windows(monkeypatch: pytest.MonkeyPatch, alive: Callable[[int], bool]) -> None:
    """Patch ``os.name`` to ``nt``; call after every ``Path`` is built, as ``Path`` may refuse to instantiate under ``nt``."""
    monkeypatch.setattr(transport, "_windows_pid_alive", alive)
    monkeypatch.setattr(os, "name", "nt")


def _own_zombie_children() -> list[int]:
    """Zombie (state ``Z``) children of the current process, read from ``/proc`` (Linux-only;
    returns an empty list, i.e. no assertion power, on any platform without ``/proc``)."""
    proc_dir = Path("/proc")
    if not proc_dir.is_dir():
        return []
    my_pid = os.getpid()
    zombies: list[int] = []
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text()
        except OSError:
            continue
        closing = stat_text.rfind(")")
        if closing == -1:
            continue
        fields = stat_text[closing + 2 :].split()
        if len(fields) < 2:
            continue
        state, ppid = fields[0], fields[1]
        if state == "Z" and int(ppid) == my_pid:
            zombies.append(int(entry.name))
    return zombies


def _interrupted_popen_class(
    spawned: list[subprocess.Popen[bytes]],
    *,
    communicate_raises: type[BaseException] | None = None,
    wait_before_raise: Any = None,
) -> type[subprocess.Popen[bytes]]:
    """A ``subprocess.Popen`` subclass recording every spawned instance into ``spawned``; when
    ``communicate_raises`` is given, ``communicate`` calls ``wait_before_raise(self)`` (if given)
    and then raises it instead of running normally (shared by every test that needs to observe or
    interrupt the spawned process)."""

    class RecordingPopen(subprocess.Popen):  # type: ignore[type-arg]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            spawned.append(self)

        def communicate(self, *args: Any, **kwargs: Any) -> tuple[bytes, bytes]:
            if communicate_raises is not None:
                if wait_before_raise is not None:
                    wait_before_raise(self)
                raise communicate_raises
            return super().communicate(*args, **kwargs)

    return RecordingPopen


class TestMinimalEnvironment:
    @pytest.mark.skipif(os.name != "posix", reason="asserts the POSIX environment shape")
    def test_exact_key_set_on_posix_with_unrelated_env_vars(self) -> None:
        source_env = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/someone",
            "USER": "someone",
            "RANDOM_VAR": "leak-me-not",
        }
        result = minimal_environment(source_env=source_env)
        assert result == {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}

    def test_path_falls_back_to_defpath_when_absent(self) -> None:
        result = minimal_environment(source_env={"HOME": "/home/someone"})
        assert result["PATH"] == os.defpath

    def test_explicit_license_file_overrides_source_env(self) -> None:
        result = minimal_environment(
            license_file="/explicit/license.txt",
            source_env={"PATH": "/usr/bin", "MLODA_LICENSE_FILE": "/from/env/license.txt"},
        )
        assert result["MLODA_LICENSE_FILE"] == "/explicit/license.txt"
        assert "MLODA_LICENSE_KEY" not in result

    def test_explicit_license_key_overrides_source_env(self) -> None:
        result = minimal_environment(
            license_key="explicit-key",
            source_env={"PATH": "/usr/bin", "MLODA_LICENSE_KEY": "env-key"},
        )
        assert result["MLODA_LICENSE_KEY"] == "explicit-key"

    def test_license_file_falls_back_to_source_env_when_no_override(self) -> None:
        result = minimal_environment(source_env={"PATH": "/usr/bin", "MLODA_LICENSE_FILE": "/from/env/license.txt"})
        assert result["MLODA_LICENSE_FILE"] == "/from/env/license.txt"

    def test_empty_string_license_values_in_source_env_are_dropped(self) -> None:
        result = minimal_environment(source_env={"PATH": "/usr/bin", "MLODA_LICENSE_FILE": "", "MLODA_LICENSE_KEY": ""})
        assert "MLODA_LICENSE_FILE" not in result
        assert "MLODA_LICENSE_KEY" not in result

    def test_both_license_keys_present_when_both_set(self) -> None:
        result = minimal_environment(
            license_file="/f/license.txt", license_key="inline-key", source_env={"PATH": "/usr/bin"}
        )
        assert result["MLODA_LICENSE_FILE"] == "/f/license.txt"
        assert result["MLODA_LICENSE_KEY"] == "inline-key"

    def test_source_env_defaults_to_os_environ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", "/from/os/environ")
        result = minimal_environment()
        assert result["PATH"] == "/from/os/environ"

    def test_relative_license_file_is_absolutized(self) -> None:
        """The binary runs with its own private invocation directory as its cwd, so a relative
        ``MLODA_LICENSE_FILE`` must be made absolute here, against the caller's own cwd, before it
        is ever handed to the subprocess (contract: License, Data handling)."""
        result = minimal_environment(license_file="license.txt", source_env={"PATH": "/usr/bin"})
        assert os.path.isabs(result["MLODA_LICENSE_FILE"])
        assert result["MLODA_LICENSE_FILE"] == str(Path("license.txt").resolve())

    def test_inherit_license_false_drops_ambient_license_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", "/from/os/environ")
        monkeypatch.setenv("MLODA_LICENSE_FILE", "/ambient/license.txt")
        monkeypatch.setenv("MLODA_LICENSE_KEY", "ambient-key")
        result = minimal_environment(inherit_license=False)
        assert result["PATH"] == "/from/os/environ"
        assert "MLODA_LICENSE_FILE" not in result
        assert "MLODA_LICENSE_KEY" not in result

    def test_inherit_license_false_keeps_path_and_systemroot_on_nt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", "C:\\Windows\\System32")
        monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
        monkeypatch.setenv("MLODA_LICENSE_FILE", "C:\\ambient\\license.txt")
        monkeypatch.setenv("MLODA_LICENSE_KEY", "ambient-key")
        monkeypatch.setattr(os, "name", "nt")
        result = minimal_environment(inherit_license=False)
        assert result["PATH"] == "C:\\Windows\\System32"
        assert result["SYSTEMROOT"] == "C:\\Windows"
        assert "MLODA_LICENSE_FILE" not in result
        assert "MLODA_LICENSE_KEY" not in result

    def test_explicit_license_file_applies_when_inherit_license_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLODA_LICENSE_KEY", "ambient-key")
        result = minimal_environment(license_file="/f/license.txt", inherit_license=False)
        assert result["MLODA_LICENSE_FILE"] == "/f/license.txt"
        assert "MLODA_LICENSE_KEY" not in result

    def test_empty_explicit_license_values_suppress_ambient_license_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", "/from/os/environ")
        monkeypatch.setenv("MLODA_LICENSE_FILE", "/ambient/license.txt")
        monkeypatch.setenv("MLODA_LICENSE_KEY", "ambient-key")
        result = minimal_environment(license_file="", license_key="")
        assert result["PATH"] == "/from/os/environ"
        assert "MLODA_LICENSE_FILE" not in result
        assert "MLODA_LICENSE_KEY" not in result


class TestPidIsAlive:
    def test_current_pid_is_alive(self) -> None:
        assert pid_is_alive(os.getpid()) is True

    def test_exited_pid_is_not_alive(self) -> None:
        assert pid_is_alive(_dead_child_pid()) is False


class TestPidRunning:
    @pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="asserts the /proc stat read")
    def test_reaped_pid_is_not_running_when_zero_signal_probe_saw_zombie(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pid reaped between the zero-signal probe and the /proc read must read as not running."""
        monkeypatch.setattr(
            "mloda.community.feature_groups.binary_model.tests.process_helpers.pid_is_alive", lambda pid: True
        )
        assert pid_running(_dead_child_pid()) is False

    @pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="asserts the /proc stat read")
    def test_live_pid_is_running_when_stat_read_is_denied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A denied stat read (e.g. procfs hidepid) must not report a live process as dead."""
        monkeypatch.setattr(
            "mloda.community.feature_groups.binary_model.tests.process_helpers.pid_is_alive", lambda pid: True
        )

        def denied(self: Path, *args: object, **kwargs: object) -> str:
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "read_text", denied)
        assert pid_running(os.getpid()) is True


class TestInvocationDirectory:
    def test_created_under_given_parent(self, tmp_path: Path) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        with InvocationDirectory(parent=parent) as inv:
            assert inv.path.parent == parent
            assert inv.path.is_dir()

    def test_name_matches_pid_and_hex_token(self, tmp_path: Path) -> None:
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            match = re.fullmatch(r"(\d+)-[0-9a-f]{8}", inv.path.name)
            assert match is not None, f"unexpected directory name: {inv.path.name!r}"
            assert int(match.group(1)) == os.getpid()

    def test_directory_mode_is_owner_only(self, tmp_path: Path) -> None:
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            assert stat.S_IMODE(inv.path.stat().st_mode) == 0o700

    def test_parent_created_when_missing(self, tmp_path: Path) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        assert not parent.exists()
        with InvocationDirectory(parent=parent):
            assert parent.is_dir()

    def test_parent_created_by_the_class_has_owner_only_mode(self, tmp_path: Path) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        assert not parent.exists()
        with InvocationDirectory(parent=parent):
            assert stat.S_IMODE(parent.stat().st_mode) == 0o700

    @pytest.mark.skipif(os.name != "posix", reason="asserts POSIX permission bits")
    def test_group_or_world_writable_parent_is_refused(self, tmp_path: Path) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(mode=0o777)
        parent.chmod(0o777)
        with pytest.raises(BinaryUnavailableError):
            with InvocationDirectory(parent=parent):
                pass

    @pytest.mark.skipif(os.name != "posix", reason="asserts POSIX permission bits")
    def test_owner_only_parent_is_accepted(self, tmp_path: Path) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(mode=0o700)
        parent.chmod(0o700)
        with InvocationDirectory(parent=parent) as inv:
            assert inv.path.is_dir()

    @pytest.mark.skipif(os.name != "posix", reason="asserts POSIX ownership")
    def test_parent_not_owned_by_the_current_user_is_refused_when_checkable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ownership can't be forged without root, so this simulates a foreign owner by wrapping
        ``os.lstat`` and reporting a different ``st_uid`` for the parent path only -- every other
        call (including pytest's and pathlib's own bookkeeping) goes through unmodified."""
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(mode=0o700)
        real_lstat = os.lstat

        def fake_lstat(path: Any, *, dir_fd: Any = None) -> os.stat_result:
            result = real_lstat(path, dir_fd=dir_fd)
            if Path(os.fspath(path)) != parent:
                return result
            fields = list(result)
            fields[stat.ST_UID] = result.st_uid + 1
            return os.stat_result(fields)

        monkeypatch.setattr(os, "lstat", fake_lstat)
        with pytest.raises(BinaryUnavailableError):
            with InvocationDirectory(parent=parent):
                pass

    @pytest.mark.skipif(os.name != "posix", reason="asserts POSIX symlinks")
    def test_symlinked_parent_is_refused_and_its_target_left_untouched(self, tmp_path: Path) -> None:
        victim = tmp_path / "victim"
        victim.mkdir()
        dead_dir = victim / f"{_dead_child_pid()}-taxes"
        dead_dir.mkdir()
        (dead_dir / "return.pdf").write_text("precious")
        (dead_dir / transport.LOCK_FILE_NAME).write_text("")
        (victim / "01-notes.md").write_text("notes")
        before = sorted(victim.iterdir())
        os.symlink(victim, tmp_path / TEMP_PARENT_NAME)
        with pytest.raises(BinaryUnavailableError, match="symlink"):
            with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME):
                pass
        assert sorted(victim.iterdir()) == before
        assert (dead_dir / "return.pdf").exists()

    @pytest.mark.skipif(os.name != "posix", reason="asserts POSIX symlinks")
    def test_dangling_symlink_parent_is_refused_as_unavailable(self, tmp_path: Path) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        os.symlink(tmp_path / "missing", parent)
        with pytest.raises(BinaryUnavailableError):
            with InvocationDirectory(parent=parent):
                pass

    @pytest.mark.skipif(os.name != "posix", reason="asserts the per-user POSIX parent name")
    def test_default_parent_is_per_user_under_the_temp_dir(self) -> None:
        assert transport.default_parent() == Path(tempfile.gettempdir()) / f"{TEMP_PARENT_NAME}-{os.getuid()}"

    @pytest.mark.skipif(os.name != "posix", reason="asserts POSIX ownership")
    def test_other_users_default_parent_does_not_block_the_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        foreign = tmp_path / TEMP_PARENT_NAME
        foreign.mkdir(mode=0o700)
        real_lstat = os.lstat

        def fake_lstat(path: Any, *, dir_fd: Any = None) -> os.stat_result:
            result = real_lstat(path, dir_fd=dir_fd)
            if Path(os.fspath(path)) != foreign:
                return result
            fields = list(result)
            fields[stat.ST_UID] = result.st_uid + 1
            return os.stat_result(fields)

        monkeypatch.setattr(os, "lstat", fake_lstat)
        with InvocationDirectory() as inv:
            assert inv.path.parent == tmp_path / f"{TEMP_PARENT_NAME}-{os.getuid()}"
            assert inv.path.is_dir()

    @pytest.mark.parametrize(
        "platform",
        [
            pytest.param(
                "posix", marks=pytest.mark.skipif(os.name != "posix", reason="asserts the POSIX lock-file path")
            ),
            "nt",
        ],
    )
    def test_dead_pid_file_sibling_is_reaped_on_enter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
    ) -> None:
        """A sibling matching the ``<pid>-`` naming that is a regular FILE (not a directory) with a
        dead pid must also be removed: ``shutil.rmtree`` alone cannot delete a plain file."""
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        dead_sibling_file = parent / f"{_dead_child_pid()}-deadfile"
        dead_sibling_file.write_text("stray file")
        if platform == "nt":
            _as_windows(monkeypatch, lambda queried: False)
        with InvocationDirectory(parent=parent):
            assert not dead_sibling_file.exists()

    def test_removed_after_normal_exit(self, tmp_path: Path) -> None:
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            path = inv.path
            assert path.is_dir()
        assert not path.exists()

    def test_removed_when_block_raises(self, tmp_path: Path) -> None:
        captured_path: Path | None = None
        with pytest.raises(RuntimeError):
            with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
                captured_path = inv.path
                raise RuntimeError("boom")
        assert captured_path is not None
        assert not captured_path.exists()

    @pytest.mark.skipif(os.name != "posix", reason="asserts the POSIX lock-file path")
    def test_unlocked_sibling_is_reaped_on_enter(self, tmp_path: Path) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        dead_sibling = parent / f"{_dead_child_pid()}-deadbeef"
        dead_sibling.mkdir()
        (dead_sibling / transport.LOCK_FILE_NAME).write_text("")
        with InvocationDirectory(parent=parent):
            assert not dead_sibling.exists()

    @pytest.mark.skipif(os.name != "posix", reason="asserts flock-based liveness")
    def test_alive_pid_sibling_is_kept(self, tmp_path: Path) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        alive_sibling = parent / f"{os.getpid()}-aliveaaa"
        alive_sibling.mkdir()
        fd = _hold_lock(alive_sibling / transport.LOCK_FILE_NAME)
        try:
            with InvocationDirectory(parent=parent):
                assert alive_sibling.is_dir()
        finally:
            os.close(fd)

    @pytest.mark.skipif(os.name != "posix", reason="asserts flock-based liveness")
    def test_locked_sibling_with_a_dead_pid_name_is_kept(self, tmp_path: Path) -> None:
        """Liveness is the lock, not the pid: a holder in another pid namespace looks dead by name."""
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        sibling = parent / f"{_dead_child_pid()}-otherns"
        sibling.mkdir()
        fd = _hold_lock(sibling / transport.LOCK_FILE_NAME)
        try:
            with InvocationDirectory(parent=parent):
                assert sibling.is_dir()
        finally:
            os.close(fd)

    @pytest.mark.skipif(os.name != "posix", reason="asserts flock-based liveness")
    def test_sibling_without_a_lock_file_is_reaped(self, tmp_path: Path) -> None:
        """Staging locks a live directory before it is visible, so a lockless one is dead."""
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        sibling = parent / f"{_dead_child_pid()}-nolock"
        sibling.mkdir()
        with InvocationDirectory(parent=parent):
            assert not sibling.exists()

    @pytest.mark.skipif(os.name != "posix", reason="asserts flock-based liveness")
    def test_lock_file_is_held_exclusively_while_inside_the_block(self, tmp_path: Path) -> None:
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            lock_file = inv.path / transport.LOCK_FILE_NAME
            assert lock_file.exists()
            fd = os.open(str(lock_file), os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
        assert not inv.path.exists()

    @pytest.mark.skipif(os.name != "posix", reason="asserts flock-based liveness")
    def test_no_unlocked_entry_is_visible_and_staging_is_owned_before_locking(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reaper must never see an unlocked <pid>- sibling or ``.lock``, nor reap the owner's own staging."""
        parent = tmp_path / TEMP_PARENT_NAME
        real_flock = fcntl.flock
        visible_at_lock: list[str] = []
        lock_visible_at_lock: list[Path] = []
        unowned_staging_at_lock: list[Path] = []
        calls: list[int] = []

        def spy_flock(fd: int, operation: int) -> None:
            if operation & fcntl.LOCK_EX and parent.exists():
                calls.append(operation)
                visible_at_lock.extend(p.name for p in parent.iterdir() if re.match(r"^\d+-", p.name))
                lock_visible_at_lock.extend(parent.rglob(transport.LOCK_FILE_NAME))
                unowned_staging_at_lock.extend(p for p in parent.glob(".tmp-*") if p not in transport._OWNED_PATHS)
            real_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", spy_flock)
        with InvocationDirectory(parent=parent) as inv:
            assert re.match(r"^\d+-", inv.path.name)
            assert (inv.path / transport.LOCK_FILE_NAME).exists()
        assert calls
        assert visible_at_lock == []
        assert lock_visible_at_lock == []
        assert unowned_staging_at_lock == []
        assert not [p for p in transport._OWNED_PATHS if p.name.startswith(".tmp-")]

    @pytest.mark.skipif(os.name != "posix", reason="asserts flock-based liveness")
    def test_owner_lock_is_non_blocking(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        real_flock = fcntl.flock
        operations: list[int] = []

        def spy_flock(fd: int, operation: int) -> None:
            operations.append(operation)
            real_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", spy_flock)
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME):
            pass
        assert operations
        assert all(op & fcntl.LOCK_NB for op in operations)

    @pytest.mark.skipif(os.name != "posix", reason="asserts flock-based liveness")
    def test_lock_failure_is_unavailable_and_leaves_no_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parent = tmp_path / TEMP_PARENT_NAME

        def failing_flock(fd: int, operation: int) -> None:
            raise OSError(errno.ENOLCK, "no locks")

        monkeypatch.setattr(fcntl, "flock", failing_flock)
        with pytest.raises(BinaryUnavailableError, match="lock"):
            with InvocationDirectory(parent=parent):
                pass
        assert list(parent.iterdir()) == []

    @pytest.mark.skipif(os.name != "posix", reason="asserts flock-based liveness")
    def test_nested_instance_does_not_reap_the_outer_under_per_process_locks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Where flock is emulated as per-process record locks (NFS), the same process can retake its own lock."""
        monkeypatch.setattr(fcntl, "flock", fcntl.lockf)
        parent = tmp_path / TEMP_PARENT_NAME
        with InvocationDirectory(parent=parent) as outer:
            with InvocationDirectory(parent=parent):
                assert outer.path.is_dir()
            assert outer.path.is_dir()

    def test_non_matching_name_sibling_is_kept(self, tmp_path: Path) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        stray = parent / "not-a-pid-dir"
        stray.mkdir()
        with InvocationDirectory(parent=parent):
            assert stray.is_dir()

    def test_two_sequential_instances_never_collide(self, tmp_path: Path) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        with InvocationDirectory(parent=parent) as first:
            first_path = first.path
        with InvocationDirectory(parent=parent) as second:
            assert second.path != first_path

    def test_nested_instances_do_not_collide_or_interfere(self, tmp_path: Path) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        with InvocationDirectory(parent=parent) as outer:
            with InvocationDirectory(parent=parent) as inner:
                assert outer.path != inner.path
                assert outer.path.is_dir()
                assert inner.path.is_dir()
            assert outer.path.is_dir()
        assert not outer.path.exists()

    @pytest.mark.skipif(os.name != "posix", reason="asserts the POSIX lock-file path")
    @pytest.mark.parametrize(
        ("kind", "kept"),
        [
            ("unlocked", False),
            ("held", True),
            ("absent", True),
            ("absent_old", False),
            ("lock_init_old", False),
            ("lock_init_fresh", True),
            ("plain_file", True),
        ],
    )
    def test_staging_sibling_is_reaped_only_when_its_lock_is_free(self, tmp_path: Path, kind: str, kept: bool) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        sibling = parent / ".tmp-0123abcd"
        fd: int | None = None
        if kind == "plain_file":
            sibling.write_text("stray")
        else:
            sibling.mkdir()
            if kind == "unlocked":
                (sibling / transport.LOCK_FILE_NAME).write_text("")
            elif kind == "held":
                fd = _hold_lock(sibling / transport.LOCK_FILE_NAME)
            elif kind.startswith("lock_init"):
                (sibling / ".lock-init").write_text("")
            if kind.endswith("_old"):
                old = time.time() - 2 * getattr(transport, "_STAGING_GRACE_SECONDS", 600.0)
                os.utime(sibling, (old, old))
        try:
            with InvocationDirectory(parent=parent):
                assert sibling.exists() is kept
        finally:
            if fd is not None:
                os.close(fd)

    @pytest.mark.parametrize(
        ("pid_kind", "owned", "alive", "kept"),
        [
            ("dead", False, False, False),
            ("other", False, True, True),
            ("own", False, True, False),
            ("own", True, False, True),
        ],
    )
    def test_windows_sibling_is_reaped_by_pid_liveness(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_kind: str, owned: bool, alive: bool, kept: bool
    ) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        pid = os.getpid() if pid_kind == "own" else _dead_child_pid()
        sibling = parent / f"{pid}-winsibl"
        sibling.mkdir()
        monkeypatch.setattr(transport, "_OWNED_PATHS", {sibling} if owned else set())
        queried_pids: list[int] = []

        def stub_alive(queried: int) -> bool:
            queried_pids.append(queried)
            return alive

        _as_windows(monkeypatch, stub_alive)
        with InvocationDirectory(parent=parent) as inv:
            assert inv.path.is_dir()
            assert sibling.exists() is kept
        assert queried_pids == ([] if pid_kind == "own" else [pid])

    def test_windows_directory_is_owned_before_it_is_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        real_mkdir = Path.mkdir
        owned_at_mkdir: list[bool] = []

        def spy_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
            if re.match(r"^\d+-", self.name):
                owned_at_mkdir.append(self in transport._OWNED_PATHS)
            real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", spy_mkdir)
        _as_windows(monkeypatch, lambda queried: True)
        with InvocationDirectory(parent=parent):
            pass
        assert owned_at_mkdir == [True]

    def test_windows_mkdir_failure_is_unavailable_and_leaves_no_owned_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        real_mkdir = Path.mkdir
        owned_before = set(transport._OWNED_PATHS)

        def failing_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
            if re.match(r"^\d+-", self.name):
                raise OSError(errno.ENOSPC, "no space")
            real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", failing_mkdir)
        _as_windows(monkeypatch, lambda queried: True)
        with pytest.raises(BinaryUnavailableError, match="cannot create an invocation directory"):
            with InvocationDirectory(parent=parent):
                pass
        assert transport._OWNED_PATHS == owned_before
        assert list(parent.iterdir()) == []

    @pytest.mark.parametrize("failure", [OSError(errno.EPERM, "denied"), KeyboardInterrupt()])
    def test_windows_chmod_failure_leaves_no_directory_or_owned_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
    ) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        owned_before = set(transport._OWNED_PATHS)

        def failing_chmod(self: Path, *args: Any, **kwargs: Any) -> None:
            raise failure

        monkeypatch.setattr(Path, "chmod", failing_chmod)
        _as_windows(monkeypatch, lambda queried: True)
        if isinstance(failure, OSError):
            with pytest.raises(BinaryUnavailableError, match="cannot create an invocation directory"):
                with InvocationDirectory(parent=parent):
                    pass
        else:
            with pytest.raises(KeyboardInterrupt):
                with InvocationDirectory(parent=parent):
                    pass
        assert transport._OWNED_PATHS == owned_before
        assert list(parent.iterdir()) == []

    def test_windows_non_oserror_from_mkdir_propagates_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        real_mkdir = Path.mkdir
        owned_before = set(transport._OWNED_PATHS)

        def interrupted_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
            if re.match(r"^\d+-", self.name):
                raise KeyboardInterrupt
            real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", interrupted_mkdir)
        _as_windows(monkeypatch, lambda queried: True)
        with pytest.raises(KeyboardInterrupt):
            with InvocationDirectory(parent=parent):
                pass
        assert transport._OWNED_PATHS == owned_before


@pytest.mark.skipif(os.name != "nt", reason="exercises the real Windows process API")
class TestWindowsPidAlive:
    def test_current_pid_is_alive(self) -> None:
        assert transport._windows_pid_alive(os.getpid()) is True

    def test_exited_pid_is_not_alive(self) -> None:
        assert transport._windows_pid_alive(_dead_child_pid()) is False


class TestRunBinary:
    def test_happy_path_over_stdin_returns_parseable_output(self, tmp_path: Path) -> None:
        schema = pa.schema([pa.field("col_a", pa.string())])
        rows = {"col_a": ["alpha", "beta", "gamma"]}
        input_bytes = arrow_stream_bytes(schema, rows)
        env = {"PATH": os.defpath, "MLODA_LICENSE_KEY": valid_license_token([PLUGIN_ID])}
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            output_bytes = _run_binary(STUB_CMD, env, _hash_config(), input_bytes, inv.path)
            assert (inv.path / "config.json").is_file()
        table = read_arrow_stream(output_bytes)
        expected = compute_expected_hash_column(rows, ["col_a"], None)
        assert table.column("col_a_hash").to_pylist() == expected
        assert table.num_rows == 3

    @pytest.mark.skipif(os.name != "posix", reason="asserts POSIX permission bits")
    def test_config_json_has_owner_only_mode_after_run(self, tmp_path: Path) -> None:
        schema = pa.schema([pa.field("col_a", pa.string())])
        input_bytes = arrow_stream_bytes(schema, {"col_a": ["alpha"]})
        env = {"PATH": os.defpath, "MLODA_LICENSE_KEY": valid_license_token([PLUGIN_ID])}
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            _run_binary(STUB_CMD, env, _hash_config(), input_bytes, inv.path)
            assert stat.S_IMODE((inv.path / "config.json").stat().st_mode) == 0o600

    def test_file_transport_used_above_threshold(self, tmp_path: Path) -> None:
        """Named for the actual condition: a threshold of 0 with non-empty input means
        ``len(input_bytes) > file_transport_threshold``, the ABOVE-threshold path (file transport),
        not below it."""
        schema = pa.schema([pa.field("col_a", pa.string())])
        rows = {"col_a": ["alpha", "beta"]}
        input_bytes = arrow_stream_bytes(schema, rows)
        env = {"PATH": os.defpath, "MLODA_LICENSE_KEY": valid_license_token([PLUGIN_ID])}
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            output_bytes = _run_binary(STUB_CMD, env, _hash_config(), input_bytes, inv.path, file_transport_threshold=0)
            assert (inv.path / "input.arrows").is_file()
            assert (inv.path / "output.arrows").is_file()
            assert (inv.path / "output.arrows").read_bytes() == output_bytes
        table = read_arrow_stream(output_bytes)
        expected = compute_expected_hash_column(rows, ["col_a"], None)
        assert table.column("col_a_hash").to_pylist() == expected

    def test_stdin_stdout_used_below_threshold_leaves_no_transport_files(self, tmp_path: Path) -> None:
        """The counterpart of ``test_file_transport_used_above_threshold``: a threshold larger
        than the input uses stdin/stdout and leaves no ``input.arrows``/``output.arrows`` behind."""
        schema = pa.schema([pa.field("col_a", pa.string())])
        rows = {"col_a": ["alpha", "beta"]}
        input_bytes = arrow_stream_bytes(schema, rows)
        env = {"PATH": os.defpath, "MLODA_LICENSE_KEY": valid_license_token([PLUGIN_ID])}
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            output_bytes = _run_binary(
                STUB_CMD, env, _hash_config(), input_bytes, inv.path, file_transport_threshold=len(input_bytes) + 1
            )
            assert not (inv.path / "input.arrows").exists()
            assert not (inv.path / "output.arrows").exists()
        table = read_arrow_stream(output_bytes)
        expected = compute_expected_hash_column(rows, ["col_a"], None)
        assert table.column("col_a_hash").to_pylist() == expected

    def test_missing_license_raises_license_missing(self, tmp_path: Path) -> None:
        input_bytes = arrow_stream_bytes(pa.schema([pa.field("col_a", pa.string())]), {"col_a": ["alpha"]})
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(LicenseMissingError) as excinfo:
                _run_binary(STUB_CMD, {"PATH": os.defpath}, _hash_config(), input_bytes, inv.path)
        assert excinfo.value.code == 2
        assert "MLODA_LICENSE_FILE" in excinfo.value.message

    def test_expired_license_raises_license_invalid(self, tmp_path: Path) -> None:
        input_bytes = arrow_stream_bytes(pa.schema([pa.field("col_a", pa.string())]), {"col_a": ["alpha"]})
        env = {"PATH": os.defpath, "MLODA_LICENSE_KEY": expired_license_token([PLUGIN_ID])}
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(LicenseInvalidError) as excinfo:
                _run_binary(STUB_CMD, env, _hash_config(), input_bytes, inv.path)
        assert excinfo.value.code == 3

    def test_unknown_operation_raises_unsupported(self, tmp_path: Path) -> None:
        input_bytes = arrow_stream_bytes(pa.schema([pa.field("col_a", pa.string())]), {"col_a": ["alpha"]})
        env = {"PATH": os.defpath, "MLODA_LICENSE_KEY": valid_license_token([PLUGIN_ID])}
        config = _hash_config(operation="no-such-operation")
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(UnsupportedError) as excinfo:
                _run_binary(STUB_CMD, env, config, input_bytes, inv.path)
        assert excinfo.value.code == 4

    def test_non_finite_config_fallback_message_never_names_the_value(self, tmp_path: Path) -> None:
        """A NaN outside parameters raises BinaryUsageError whose message does not echo the value."""
        config = _hash_config(extra=float("nan"))
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(BinaryUsageError) as excinfo:
                _run_binary(STUB_CMD, {"PATH": os.defpath}, config, b"", inv.path)
        assert "nan" not in str(excinfo.value).lower()

    def test_hanging_binary_is_terminated_on_timeout_without_a_zombie(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spawned: list[subprocess.Popen[bytes]] = []
        monkeypatch.setattr(subprocess, "Popen", _interrupted_popen_class(spawned))
        started = time.monotonic()
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(BinaryTerminatedError) as excinfo:
                _run_binary(
                    [*FAULTY_CMD, "--mode", "hang"], {"PATH": os.defpath}, _hash_config(), b"", inv.path, timeout=0.5
                )
        elapsed = time.monotonic() - started
        assert excinfo.value.code == 6
        assert elapsed < 5.0, f"expected termination well before the 60s hang, took {elapsed}s"
        assert _own_zombie_children() == []
        assert len(spawned) == 1
        if os.name == "posix":
            assert spawned[0].stdin is not None and spawned[0].stdin.closed
            assert spawned[0].stdout is not None and spawned[0].stdout.closed
            assert spawned[0].stderr is not None and spawned[0].stderr.closed

    @pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, RuntimeError])
    def test_exceptional_exit_during_communicate_terminates_and_reaps_the_child(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: type[BaseException]
    ) -> None:
        spawned: list[subprocess.Popen[bytes]] = []
        monkeypatch.setattr(subprocess, "Popen", _interrupted_popen_class(spawned, communicate_raises=exc))
        try:
            with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
                with pytest.raises(exc):
                    _run_binary([*FAULTY_CMD, "--mode", "hang"], {"PATH": os.defpath}, _hash_config(), b"", inv.path)
                assert len(spawned) == 1
                assert spawned[0].poll() is not None, f"child was left running after {exc.__name__}"
                assert _own_zombie_children() == []
                if os.name == "posix":
                    assert spawned[0].stdin is not None and spawned[0].stdin.closed
                    assert spawned[0].stdout is not None and spawned[0].stdout.closed
                    assert spawned[0].stderr is not None and spawned[0].stderr.closed
        finally:
            monkeypatch.undo()
            for proc in spawned:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()

    @pytest.mark.skipif(os.name != "posix", reason="process-group orphan only reachable on POSIX")
    def test_exceptional_exit_after_leader_already_exited_still_kills_descendant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The leader (``exit_leaving_child``) spawns a descendant, writes its pid, and exits before
        ``run_binary``'s exceptional-exit handler runs; the descendant must still be killed even
        though the leader is already dead by then."""
        pid_file = tmp_path / "child.pid"
        spawned: list[subprocess.Popen[bytes]] = []

        def _wait_for_leader_exit_and_pid_file(proc: subprocess.Popen[bytes]) -> None:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if pid_file.exists():
                    try:
                        proc.wait(timeout=0.05)
                        return
                    except subprocess.TimeoutExpired:
                        continue
                time.sleep(0.02)

        monkeypatch.setattr(
            subprocess,
            "Popen",
            _interrupted_popen_class(
                spawned, communicate_raises=KeyboardInterrupt, wait_before_raise=_wait_for_leader_exit_and_pid_file
            ),
        )
        child_pid: int | None = None
        try:
            with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
                with pytest.raises(KeyboardInterrupt):
                    _run_binary(
                        [*FAULTY_CMD, "--mode", "exit_leaving_child"],
                        {"PATH": os.defpath},
                        _hash_config(parameters={"pid_file": str(pid_file)}),
                        b"",
                        inv.path,
                    )
            assert pid_file.exists(), "faulty_binary never wrote the descendant's pid"
            child_pid = int(pid_file.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and pid_running(child_pid):
                time.sleep(0.05)
            assert not pid_running(child_pid), "descendant of an already-exited leader was left alive"
        finally:
            monkeypatch.undo()
            kill_descendant_if_running(pid_file, child_pid)

    def test_exit_before_reading_with_large_input_does_not_raise_broken_pipe(self, tmp_path: Path) -> None:
        large_input = os.urandom(4 * 1024 * 1024)
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(LicenseMissingError) as excinfo:
                _run_binary(
                    [*FAULTY_CMD, "--mode", "exit_before_reading"],
                    {"PATH": os.defpath},
                    _hash_config(),
                    large_input,
                    inv.path,
                )
        assert excinfo.value.code == 2

    def test_garbage_stderr_raises_binary_internal_error(self, tmp_path: Path) -> None:
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(BinaryInternalError) as excinfo:
                _run_binary(
                    [*FAULTY_CMD, "--mode", "garbage_stderr"], {"PATH": os.defpath}, _hash_config(), b"", inv.path
                )
        assert excinfo.value.code == 6

    def test_signal_terminated_binary_raises_binary_internal_error(self, tmp_path: Path) -> None:
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(BinaryInternalError) as excinfo:
                _run_binary([*FAULTY_CMD, "--mode", "signal"], {"PATH": os.defpath}, _hash_config(), b"", inv.path)
        assert excinfo.value.code == 6

    def test_process_receives_exactly_the_given_environment(self, tmp_path: Path) -> None:
        # LC_ALL/LANG must be set (matching minimal_environment's own POSIX shape): otherwise
        # CPython's PEP 538 locale coercion injects its own LC_CTYPE into the child's os.environ,
        # which would make this assertion fail for a reason unrelated to run_binary's own env
        # handling.
        env = {"PATH": os.defpath, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "FOO": "bar"}
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            output_bytes = _run_binary([*FAULTY_CMD, "--mode", "echo_env"], env, _hash_config(), b"", inv.path)
        assert json.loads(output_bytes) == sorted(env)

    @pytest.mark.parametrize(
        "case",
        [
            "non_executable_file",
            pytest.param(
                "executable_without_shebang_or_elf_header",
                marks=pytest.mark.skipif(
                    os.name != "posix", reason="relies on the kernel rejecting an unrecognised executable format"
                ),
            ),
            "popen_raises_emfile",
        ],
    )
    def test_spawn_failure_raises_binary_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
    ) -> None:
        cmd = STUB_CMD
        if case == "non_executable_file":
            not_executable = tmp_path / "not-a-binary"
            not_executable.write_text("not a script")
            not_executable.chmod(0o600)
            cmd = [str(not_executable)]
        elif case == "executable_without_shebang_or_elf_header":
            not_a_program = tmp_path / "not-a-program"
            not_a_program.write_bytes(b"\x00\x01\x02 neither a shebang nor an ELF header\n")
            not_a_program.chmod(0o700)
            cmd = [str(not_a_program)]
        else:

            def failing_popen(*args: Any, **kwargs: Any) -> Any:
                raise OSError(errno.EMFILE, "too many open files")

            monkeypatch.setattr(subprocess, "Popen", failing_popen)
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(BinaryUnavailableError, match="cannot spawn binary"):
                _run_binary(cmd, {"PATH": os.defpath}, _hash_config(), b"", inv.path)

    @pytest.mark.parametrize("target", ["config", "input", "write_input"])
    def test_write_failure_in_the_invocation_directory_raises_binary_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
    ) -> None:
        real_open = os.open
        failing_name = {"config": "config.json", "input": "input.arrows"}.get(target)

        def failing_open(path: Any, *args: Any, **kwargs: Any) -> int:
            if failing_name is not None and str(path).endswith(failing_name):
                raise OSError(errno.ENOSPC, "no space left on device")
            return real_open(path, *args, **kwargs)

        def failing_write_input(sink: Any) -> None:
            sink.write(b"abc")
            raise OSError(errno.ENOSPC, "no space left on device")

        input_bytes = arrow_stream_bytes(pa.schema([pa.field("col_a", pa.string())]), {"col_a": ["alpha"]})
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            monkeypatch.setattr(os, "open", failing_open)
            with pytest.raises(BinaryUnavailableError, match="cannot write to the invocation directory"):
                if target == "write_input":
                    run_binary(
                        STUB_CMD,
                        {"PATH": os.defpath},
                        _hash_config(),
                        failing_write_input,
                        len(input_bytes),
                        timeout=10.0,
                        file_transport_threshold=0,
                        invocation_dir=inv.path,
                    )
                else:
                    _run_binary(
                        STUB_CMD, {"PATH": os.defpath}, _hash_config(), input_bytes, inv.path, file_transport_threshold=0
                    )

    def test_nonexistent_path_raises_binary_unavailable(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(BinaryUnavailableError):
                _run_binary([str(missing)], {"PATH": os.defpath}, _hash_config(), b"", inv.path)

    def test_exit_zero_without_writing_the_output_file_raises_output_contract_error(self, tmp_path: Path) -> None:
        """A binary that exits 0 but writes no ``--output`` file must be reported as an output
        contract violation, not an uncaught filesystem error, and the message must not leak the
        (private, per-invocation) directory path (contract: Data, Data handling)."""
        input_bytes = arrow_stream_bytes(pa.schema([pa.field("col_a", pa.string())]), {"col_a": ["alpha"]})
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(OutputContractError) as excinfo:
                _run_binary(
                    [*FAULTY_CMD, "--mode", "no_output_file"],
                    {"PATH": os.defpath},
                    _hash_config(),
                    input_bytes,
                    inv.path,
                    file_transport_threshold=0,
                )
        assert str(inv.path) not in excinfo.value.message


class _FakePipe:
    """Stand-in for a pipe handle, recording whether ``close()`` was called."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    """Tiny stand-in for ``subprocess.Popen`` used to unit-test ``_terminate_timed_out_process``
    without spawning a real process."""

    def __init__(
        self, wait_effects: Sequence[BaseException] = (), *, always_timeout: bool = False, pid: int = 4321
    ) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self._wait_effects = list(wait_effects)
        self._always_timeout = always_timeout
        self.wait_calls: list[float | None] = []
        self.terminate_called = False
        self.kill_called = False
        self.stdin = _FakePipe()
        self.stdout = _FakePipe()
        self.stderr = _FakePipe()

    def wait(self, timeout: float | None = None) -> int | None:
        self.wait_calls.append(timeout)
        if self._always_timeout:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout if timeout is not None else 0)
        if self._wait_effects:
            effect = self._wait_effects.pop(0)
            raise effect
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_called = True

    def kill(self) -> None:
        self.kill_called = True


class TestTerminateTimedOutProcess:
    """Unit tests for ``_terminate_timed_out_process`` against a fake process, parametrized over
    POSIX and Windows (``os.name``), so no real subprocess is spawned."""

    @pytest.mark.parametrize(
        "os_name",
        [
            pytest.param(
                "posix",
                marks=pytest.mark.skipif(
                    not hasattr(signal, "SIGKILL"), reason="asserts signal.SIGKILL, absent on a real Windows host"
                ),
            ),
            "nt",
        ],
    )
    def test_hard_kill_is_issued_even_when_the_grace_wait_is_interrupted(
        self, monkeypatch: pytest.MonkeyPatch, os_name: str
    ) -> None:
        monkeypatch.setattr(os, "name", os_name)
        killpg_calls: list[tuple[int, int]] = []
        monkeypatch.setattr(os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)), raising=False)
        proc = _FakeProcess(wait_effects=[KeyboardInterrupt()])
        with pytest.raises(KeyboardInterrupt):
            _terminate_timed_out_process(proc)  # type: ignore[arg-type]
        if os_name == "posix":
            assert (proc.pid, signal.SIGKILL) in killpg_calls, (
                f"hard kill must be issued even when the grace wait is interrupted; calls were {killpg_calls}"
            )
        else:
            assert proc.kill_called, "kill() must be called even when the grace wait is interrupted"

    @pytest.mark.parametrize("os_name", ["posix", "nt"])
    def test_final_wait_is_bounded_and_logs_a_warning_on_expiry(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, os_name: str
    ) -> None:
        monkeypatch.setattr(os, "name", os_name)
        monkeypatch.setattr(os, "killpg", lambda pid, sig: None, raising=False)
        proc = _FakeProcess(always_timeout=True, pid=9999)
        caplog.set_level(logging.WARNING)
        _terminate_timed_out_process(proc)  # type: ignore[arg-type]
        assert proc.wait_calls, "expected at least one wait() call"
        assert all(t is not None for t in proc.wait_calls), (
            f"wait() must always use a bounded timeout, got {proc.wait_calls}"
        )
        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert any("9999" in record.getMessage() for record in warnings), (
            f"expected a WARNING naming pid 9999; got {[record.getMessage() for record in warnings]}"
        )

    @pytest.mark.parametrize("os_name", ["posix", "nt"])
    def test_pipes_closed_on_posix_only_after_normal_termination(
        self, monkeypatch: pytest.MonkeyPatch, os_name: str
    ) -> None:
        monkeypatch.setattr(os, "name", os_name)
        monkeypatch.setattr(os, "killpg", lambda pid, sig: None, raising=False)
        proc = _FakeProcess()
        _terminate_timed_out_process(proc)  # type: ignore[arg-type]
        if os_name == "posix":
            assert proc.stdin.closed and proc.stdout.closed and proc.stderr.closed
        else:
            assert not proc.stdin.closed and not proc.stdout.closed and not proc.stderr.closed
