"""Tests for ``transport.py``: the minimal subprocess environment, the per-invocation directory,
and ``run_binary`` itself, exercised against the well-behaved simulated binary
(``mloda.testing.binary_model``) for the happy paths and against the deliberately misbehaving
``faulty_binary.py`` fixture for every termination/error path (contract: Invocation, License, Data
handling, Errors).
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess  # nosec
import sys
import time
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
    UnsupportedError,
)
from mloda.community.feature_groups.binary_model.transport import (
    TEMP_PARENT_NAME,
    InvocationDirectory,
    minimal_environment,
    pid_is_alive,
    run_binary,
)
from mloda.testing.binary_model.arrow import arrow_stream_bytes, read_arrow_stream
from mloda.testing.binary_model.hash_reference import compute_expected_hash_column
from mloda.testing.binary_model.license_vectors import license_token_text

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
        env = {"PATH": os.defpath, "MLODA_LICENSE_KEY": license_token_text("valid", [PLUGIN_ID])}
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

    def test_file_transport_used_below_threshold(self, tmp_path: Path) -> None:
        schema = pa.schema([pa.field("col_a", pa.string())])
        rows = {"col_a": ["alpha", "beta"]}
        input_bytes = arrow_stream_bytes(schema, rows)
        env = {"PATH": os.defpath, "MLODA_LICENSE_KEY": license_token_text("valid", [PLUGIN_ID])}
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
        env = {"PATH": os.defpath, "MLODA_LICENSE_KEY": license_token_text("expired", [PLUGIN_ID])}
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
        env = {"PATH": os.defpath, "MLODA_LICENSE_KEY": license_token_text("valid", [PLUGIN_ID])}
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

    def test_hanging_binary_is_terminated_on_timeout_without_a_zombie(self, tmp_path: Path) -> None:
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
