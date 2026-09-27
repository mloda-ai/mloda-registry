"""Tests for ``transport.py``: the minimal subprocess environment, the per-invocation directory,
and ``run_binary`` itself, exercised against the well-behaved simulated binary
(``mloda.testing.binary_model``) for the happy paths and against the deliberately misbehaving
``faulty_binary.py`` fixture for every termination/error path (contract: Invocation, License, Data
handling, Errors).
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import stat
import subprocess  # nosec
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from mloda.community.feature_groups.binary_model.errors import (
    BinaryInternalError,
    BinaryTerminatedError,
    BinaryUnavailableError,
    LicenseInvalidError,
    LicenseMissingError,
    OutputContractError,
    UnsupportedError,
)
from mloda.community.feature_groups.binary_model.tests.process_helpers import pid_running
from mloda.community.feature_groups.binary_model.transport import (
    TEMP_PARENT_NAME,
    InvocationDirectory,
    _terminate_timed_out_process,
    minimal_environment,
    pid_is_alive,
    run_binary,
)
from mloda.testing.binary_model.arrow import arrow_stream_bytes, read_arrow_stream
from mloda.testing.binary_model.hash_reference import compute_expected_hash_column
from mloda.testing.binary_model.license_vectors import expired_license_token, valid_license_token

STUB_CMD = [sys.executable, "-m", "mloda.testing.binary_model.simulated_binary"]
FAULTY_CMD = [sys.executable, "-m", "mloda.community.feature_groups.binary_model.tests.faulty_binary"]
PLUGIN_ID = "example_binary"


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
        ``os.stat`` and reporting a different ``st_uid`` for the parent path only -- every other
        call (including pytest's and pathlib's own bookkeeping) goes through unmodified."""
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(mode=0o700)
        real_stat = os.stat

        def fake_stat(path: Any, *, dir_fd: Any = None, follow_symlinks: bool = True) -> os.stat_result:
            result = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
            if Path(os.fspath(path)) != parent:
                return result
            fields = list(result)
            fields[stat.ST_UID] = result.st_uid + 1
            return os.stat_result(fields)

        monkeypatch.setattr(os, "stat", fake_stat)
        with pytest.raises(BinaryUnavailableError):
            with InvocationDirectory(parent=parent):
                pass

    def test_dead_pid_file_sibling_is_reaped_on_enter(self, tmp_path: Path) -> None:
        """A sibling matching the ``<pid>-`` naming that is a regular FILE (not a directory) with a
        dead pid must also be removed: ``shutil.rmtree`` alone cannot delete a plain file."""
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        dead_sibling_file = parent / f"{_dead_child_pid()}-deadfile"
        dead_sibling_file.write_text("stray file")
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

    def test_dead_pid_sibling_is_reaped_on_enter(self, tmp_path: Path) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        dead_sibling = parent / f"{_dead_child_pid()}-deadbeef"
        dead_sibling.mkdir()
        with InvocationDirectory(parent=parent):
            assert not dead_sibling.exists()

    def test_alive_pid_sibling_is_kept(self, tmp_path: Path) -> None:
        parent = tmp_path / TEMP_PARENT_NAME
        parent.mkdir(parents=True)
        alive_sibling = parent / f"{os.getpid()}-aliveaaa"
        alive_sibling.mkdir()
        with InvocationDirectory(parent=parent):
            assert alive_sibling.is_dir()

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


class TestRunBinary:
    def test_happy_path_over_stdin_returns_parseable_output(self, tmp_path: Path) -> None:
        schema = pa.schema([pa.field("col_a", pa.string())])
        rows = {"col_a": ["alpha", "beta", "gamma"]}
        input_bytes = arrow_stream_bytes(schema, rows)
        env = {"PATH": os.defpath, "MLODA_LICENSE_KEY": valid_license_token([PLUGIN_ID])}
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            output_bytes = run_binary(
                STUB_CMD,
                env,
                _hash_config(),
                input_bytes,
                timeout=10.0,
                file_transport_threshold=10_000_000,
                invocation_dir=inv.path,
            )
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
            run_binary(
                STUB_CMD,
                env,
                _hash_config(),
                input_bytes,
                timeout=10.0,
                file_transport_threshold=10_000_000,
                invocation_dir=inv.path,
            )
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
            output_bytes = run_binary(
                STUB_CMD,
                env,
                _hash_config(),
                input_bytes,
                timeout=10.0,
                file_transport_threshold=0,
                invocation_dir=inv.path,
            )
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
            output_bytes = run_binary(
                STUB_CMD,
                env,
                _hash_config(),
                input_bytes,
                timeout=10.0,
                file_transport_threshold=len(input_bytes) + 1,
                invocation_dir=inv.path,
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
                run_binary(
                    STUB_CMD,
                    {"PATH": os.defpath},
                    _hash_config(),
                    input_bytes,
                    timeout=10.0,
                    file_transport_threshold=10_000_000,
                    invocation_dir=inv.path,
                )
        assert excinfo.value.code == 2
        assert "MLODA_LICENSE_FILE" in excinfo.value.message

    def test_expired_license_raises_license_invalid(self, tmp_path: Path) -> None:
        input_bytes = arrow_stream_bytes(pa.schema([pa.field("col_a", pa.string())]), {"col_a": ["alpha"]})
        env = {"PATH": os.defpath, "MLODA_LICENSE_KEY": expired_license_token([PLUGIN_ID])}
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(LicenseInvalidError) as excinfo:
                run_binary(
                    STUB_CMD,
                    env,
                    _hash_config(),
                    input_bytes,
                    timeout=10.0,
                    file_transport_threshold=10_000_000,
                    invocation_dir=inv.path,
                )
        assert excinfo.value.code == 3

    def test_unknown_operation_raises_unsupported(self, tmp_path: Path) -> None:
        input_bytes = arrow_stream_bytes(pa.schema([pa.field("col_a", pa.string())]), {"col_a": ["alpha"]})
        env = {"PATH": os.defpath, "MLODA_LICENSE_KEY": valid_license_token([PLUGIN_ID])}
        config = _hash_config(operation="no-such-operation")
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(UnsupportedError) as excinfo:
                run_binary(
                    STUB_CMD,
                    env,
                    config,
                    input_bytes,
                    timeout=10.0,
                    file_transport_threshold=10_000_000,
                    invocation_dir=inv.path,
                )
        assert excinfo.value.code == 4

    def test_hanging_binary_is_terminated_on_timeout_without_a_zombie(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spawned: list[subprocess.Popen[bytes]] = []
        monkeypatch.setattr(subprocess, "Popen", _interrupted_popen_class(spawned))
        started = time.monotonic()
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(BinaryTerminatedError) as excinfo:
                run_binary(
                    [*FAULTY_CMD, "--mode", "hang"],
                    {"PATH": os.defpath},
                    _hash_config(),
                    b"",
                    timeout=0.5,
                    file_transport_threshold=10_000_000,
                    invocation_dir=inv.path,
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
                    run_binary(
                        [*FAULTY_CMD, "--mode", "hang"],
                        {"PATH": os.defpath},
                        _hash_config(),
                        b"",
                        timeout=10.0,
                        file_transport_threshold=10_000_000,
                        invocation_dir=inv.path,
                    )
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
                    run_binary(
                        [*FAULTY_CMD, "--mode", "exit_leaving_child"],
                        {"PATH": os.defpath},
                        _hash_config(parameters={"pid_file": str(pid_file)}),
                        b"",
                        timeout=10.0,
                        file_transport_threshold=10_000_000,
                        invocation_dir=inv.path,
                    )
            assert pid_file.exists(), "faulty_binary never wrote the descendant's pid"
            child_pid = int(pid_file.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and pid_running(child_pid):
                time.sleep(0.05)
            assert not pid_running(child_pid), "descendant of an already-exited leader was left alive"
        finally:
            monkeypatch.undo()
            if child_pid is None and pid_file.exists():
                child_pid = int(pid_file.read_text(encoding="utf-8"))
            if child_pid is not None and pid_running(child_pid):
                os.kill(child_pid, signal.SIGKILL)

    def test_exit_before_reading_with_large_input_does_not_raise_broken_pipe(self, tmp_path: Path) -> None:
        large_input = os.urandom(4 * 1024 * 1024)
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(LicenseMissingError) as excinfo:
                run_binary(
                    [*FAULTY_CMD, "--mode", "exit_before_reading"],
                    {"PATH": os.defpath},
                    _hash_config(),
                    large_input,
                    timeout=10.0,
                    file_transport_threshold=10_000_000,
                    invocation_dir=inv.path,
                )
        assert excinfo.value.code == 2

    def test_garbage_stderr_raises_binary_internal_error(self, tmp_path: Path) -> None:
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(BinaryInternalError) as excinfo:
                run_binary(
                    [*FAULTY_CMD, "--mode", "garbage_stderr"],
                    {"PATH": os.defpath},
                    _hash_config(),
                    b"",
                    timeout=10.0,
                    file_transport_threshold=10_000_000,
                    invocation_dir=inv.path,
                )
        assert excinfo.value.code == 6

    def test_signal_terminated_binary_raises_binary_internal_error(self, tmp_path: Path) -> None:
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(BinaryInternalError) as excinfo:
                run_binary(
                    [*FAULTY_CMD, "--mode", "signal"],
                    {"PATH": os.defpath},
                    _hash_config(),
                    b"",
                    timeout=10.0,
                    file_transport_threshold=10_000_000,
                    invocation_dir=inv.path,
                )
        assert excinfo.value.code == 6

    def test_process_receives_exactly_the_given_environment(self, tmp_path: Path) -> None:
        # LC_ALL/LANG must be set (matching minimal_environment's own POSIX shape): otherwise
        # CPython's PEP 538 locale coercion injects its own LC_CTYPE into the child's os.environ,
        # which would make this assertion fail for a reason unrelated to run_binary's own env
        # handling.
        env = {"PATH": os.defpath, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "FOO": "bar"}
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            output_bytes = run_binary(
                [*FAULTY_CMD, "--mode", "echo_env"],
                env,
                _hash_config(),
                b"",
                timeout=10.0,
                file_transport_threshold=10_000_000,
                invocation_dir=inv.path,
            )
        assert json.loads(output_bytes) == sorted(env)

    def test_non_executable_regular_file_raises_binary_unavailable(self, tmp_path: Path) -> None:
        not_executable = tmp_path / "not-a-binary"
        not_executable.write_text("not a script")
        not_executable.chmod(0o600)
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(BinaryUnavailableError):
                run_binary(
                    [str(not_executable)],
                    {"PATH": os.defpath},
                    _hash_config(),
                    b"",
                    timeout=10.0,
                    file_transport_threshold=10_000_000,
                    invocation_dir=inv.path,
                )

    def test_nonexistent_path_raises_binary_unavailable(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(BinaryUnavailableError):
                run_binary(
                    [str(missing)],
                    {"PATH": os.defpath},
                    _hash_config(),
                    b"",
                    timeout=10.0,
                    file_transport_threshold=10_000_000,
                    invocation_dir=inv.path,
                )

    def test_exit_zero_without_writing_the_output_file_raises_output_contract_error(self, tmp_path: Path) -> None:
        """A binary that exits 0 but writes no ``--output`` file must be reported as an output
        contract violation, not an uncaught filesystem error, and the message must not leak the
        (private, per-invocation) directory path (contract: Data, Data handling)."""
        input_bytes = arrow_stream_bytes(pa.schema([pa.field("col_a", pa.string())]), {"col_a": ["alpha"]})
        with InvocationDirectory(parent=tmp_path / TEMP_PARENT_NAME) as inv:
            with pytest.raises(OutputContractError) as excinfo:
                run_binary(
                    [*FAULTY_CMD, "--mode", "no_output_file"],
                    {"PATH": os.defpath},
                    _hash_config(),
                    input_bytes,
                    timeout=10.0,
                    file_transport_threshold=0,
                    invocation_dir=inv.path,
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
