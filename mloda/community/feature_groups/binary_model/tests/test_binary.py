"""Tests for ``binary.py``: resolving a binary from an explicit override or the installed wheel,
probing ``--version``/``--capabilities`` up front (contract: Invocation, Capabilities), and
caching the probe result per process so a warm binary is never re-probed.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess  # nosec
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mloda.community.feature_groups.binary_model import binary, contract, transport
from mloda.community.feature_groups.binary_model.errors import BinaryUnavailableError
from mloda.community.feature_groups.binary_model.tests.process_helpers import pid_running

STUB_CMD = [sys.executable, "-m", "mloda.testing.binary_model.simulated_binary"]
FAULTY_CMD = [sys.executable, "-m", "mloda.community.feature_groups.binary_model.tests.faulty_binary"]
PLUGIN_ID = "example_binary"


@pytest.fixture(autouse=True)
def _clear_capability_cache_before_each_test() -> None:
    binary.clear_capability_cache()


class _CountingPopen:
    """Wraps the real ``subprocess.Popen`` to count constructions, so cache-hit tests can assert no
    process was spawned. Patches the stdlib ``subprocess`` module object directly (not
    ``binary.subprocess``, which mypy's ``--strict`` (no implicit re-export) rejects from outside
    the module): ``binary.py`` doing ``import subprocess`` and constructing ``subprocess.Popen(...)``
    (directly, or indirectly via ``subprocess.run``) shares this very same module object at
    runtime, so patching it here is equally effective.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        real_popen = subprocess.Popen

        def counting_popen(*args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return real_popen(*args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", counting_popen)


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, plugin_id: str, binary_path: Callable[[], Path]) -> None:
    """Injects a fake importable module into ``sys.modules`` exposing ``binary_path()``, so
    ``resolve_binary(plugin_id, None, ...)`` resolves it via ``importlib.import_module`` without
    a real wheel installed (contract: Platform naming and wheel binary path)."""
    module = types.ModuleType(plugin_id)
    module.binary_path = binary_path  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, plugin_id, module)


class TestResolveBinary:
    def test_stub_override_list_resolves(self) -> None:
        resolved = binary.resolve_binary(PLUGIN_ID, STUB_CMD, env={"PATH": os.defpath}, timeout=10.0)
        assert os.path.isabs(resolved.argv[0])
        assert resolved.capabilities == binary.BinaryCapabilities(
            contract=1,
            plugin_id=PLUGIN_ID,
            version="1.0.0",
            operations=frozenset({"hash"}),
            column_types=contract.COLUMN_TYPES,
        )

    def test_str_path_override_missing_file_is_unavailable(self, tmp_path: Path) -> None:
        with pytest.raises(BinaryUnavailableError):
            binary.resolve_binary("whatever", str(tmp_path / "missing-binary"), env={"PATH": os.defpath}, timeout=10.0)

    def test_override_none_missing_module_is_unavailable(self) -> None:
        with pytest.raises(BinaryUnavailableError) as excinfo:
            binary.resolve_binary(PLUGIN_ID, None, env={"PATH": os.defpath}, timeout=10.0)
        assert PLUGIN_ID in str(excinfo.value)

    def test_override_none_binary_path_raising_filenotfound_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise() -> Path:
            raise FileNotFoundError("binary data file missing from the wheel")

        _install_fake_module(monkeypatch, PLUGIN_ID, _raise)
        with pytest.raises(BinaryUnavailableError):
            binary.resolve_binary(PLUGIN_ID, None, env={"PATH": os.defpath}, timeout=10.0)

    def test_override_none_resolves_via_fake_module_wrapper_script(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        wrapper = tmp_path / "example_binary_wrapper.sh"
        wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} -m mloda.testing.binary_model.simulated_binary "$@"\n')
        wrapper.chmod(0o700)
        _install_fake_module(monkeypatch, PLUGIN_ID, lambda: wrapper)
        resolved = binary.resolve_binary(PLUGIN_ID, None, env={"PATH": os.defpath}, timeout=10.0)
        assert resolved.argv == (str(wrapper),)
        assert resolved.capabilities.plugin_id == PLUGIN_ID

    def test_plugin_id_mismatch_is_unavailable(self) -> None:
        with pytest.raises(BinaryUnavailableError):
            binary.resolve_binary("other_binary", STUB_CMD, env={"PATH": os.defpath}, timeout=10.0)

    def test_contract_mismatch_message_names_both_versions(self) -> None:
        with pytest.raises(BinaryUnavailableError) as excinfo:
            binary.resolve_binary(
                "faulty_binary", [*FAULTY_CMD, "--mode", "contract_2"], env={"PATH": os.defpath}, timeout=10.0
            )
        message = str(excinfo.value)
        assert "2" in message
        assert "1" in message

    def test_bad_capabilities_missing_contract_key_is_unavailable(self) -> None:
        with pytest.raises(BinaryUnavailableError):
            binary.resolve_binary(
                "faulty_binary", [*FAULTY_CMD, "--mode", "bad_capabilities"], env={"PATH": os.defpath}, timeout=10.0
            )

    @pytest.mark.parametrize(
        "mode, match",
        [
            pytest.param("capabilities_not_json", "not valid JSON", id="not_json"),
            pytest.param("capabilities_oversized_int", "not valid JSON|must print a JSON object", id="oversized_int"),
            pytest.param("capabilities_deeply_nested", "not valid JSON", id="deeply_nested"),
        ],
    )
    def test_capabilities_not_json_is_unavailable(self, mode: str, match: str) -> None:
        """A ``--capabilities`` line that ``json.loads`` cannot parse -- not JSON at all, an int
        past the interpreter's string-conversion limit, or JSON deep enough to raise
        ``RecursionError`` -- is uniformly a ``BinaryUnavailableError``, not an escaping
        ``ValueError``/``RecursionError`` (contract: Capabilities). ``oversized_int`` accepts either
        message: with ``PYTHONINTMAXSTRDIGITS=0`` (or on CPython < 3.10.7) the digit string parses as
        a plain ``int``, so rejection instead happens on the "not a JSON object" check."""
        with pytest.raises(BinaryUnavailableError, match=match):
            binary.resolve_binary(
                "faulty_binary",
                [*FAULTY_CMD, "--mode", mode],
                env={"PATH": os.defpath},
                timeout=10.0,
            )

    @pytest.mark.parametrize(
        "mode",
        ["capabilities_unicode_line_separator", "capabilities_crlf"],
        ids=["unicode_line_separator", "crlf"],
    )
    def test_capabilities_line_split_tolerates_non_lf_line_content(self, mode: str) -> None:
        """The capabilities parser splits output via ``contract.split_output_lines``, the same
        "one line" notion as the conformance kit's own
        ``test_capabilities_prints_single_json_object_no_license_required`` check: split on
        ``b"\\n"`` only (contract: Capabilities). ``unicode_line_separator`` puts a raw U+2028
        inside a tolerated unknown extra value, which must not be treated as a second line.
        ``crlf`` is a regression guard for a ``\\r\\n``-terminated line, also accepted."""
        resolved = binary.resolve_binary(
            "faulty_binary", [*FAULTY_CMD, "--mode", mode], env={"PATH": os.defpath}, timeout=10.0
        )
        assert "hash" in resolved.capabilities.operations

    def test_version_two_lines_is_unavailable(self) -> None:
        with pytest.raises(BinaryUnavailableError):
            binary.resolve_binary(
                "faulty_binary", [*FAULTY_CMD, "--mode", "version_two_lines"], env={"PATH": os.defpath}, timeout=10.0
            )


class TestCapabilityCache:
    def test_second_call_with_same_argv_spawns_no_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counter = _CountingPopen(monkeypatch)
        binary.resolve_binary(PLUGIN_ID, STUB_CMD, env={"PATH": os.defpath}, timeout=10.0)
        warm_calls = counter.count
        assert warm_calls > 0
        binary.resolve_binary(PLUGIN_ID, STUB_CMD, env={"PATH": os.defpath}, timeout=10.0)
        assert counter.count == warm_calls

    def test_clear_capability_cache_forces_reprobe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counter = _CountingPopen(monkeypatch)
        binary.resolve_binary(PLUGIN_ID, STUB_CMD, env={"PATH": os.defpath}, timeout=10.0)
        warm_calls = counter.count
        binary.clear_capability_cache()
        binary.resolve_binary(PLUGIN_ID, STUB_CMD, env={"PATH": os.defpath}, timeout=10.0)
        assert counter.count > warm_calls

    def test_different_argv_suffix_is_a_different_cache_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counter = _CountingPopen(monkeypatch)
        binary.resolve_binary(PLUGIN_ID, STUB_CMD, env={"PATH": os.defpath}, timeout=10.0)
        warm_calls = counter.count
        with pytest.raises(BinaryUnavailableError):
            binary.resolve_binary(PLUGIN_ID, [*STUB_CMD, "--mode", "unused"], env={"PATH": os.defpath}, timeout=10.0)
        assert counter.count > warm_calls

    def test_different_plugin_id_with_the_same_argv_is_a_different_cache_key(self) -> None:
        """Warming the cache under one ``plugin_id`` must not let a second, unrelated
        ``plugin_id`` resolve the very same argv from that cache entry without being re-probed and
        checked against its own reported ``plugin_id`` (contract: Capabilities, Identifier)."""
        binary.resolve_binary(PLUGIN_ID, STUB_CMD, env={"PATH": os.defpath}, timeout=10.0)
        with pytest.raises(BinaryUnavailableError):
            binary.resolve_binary("other_binary", STUB_CMD, env={"PATH": os.defpath}, timeout=10.0)


class TestBareOverrideLookupOnlyViaWhich:
    """A bare override name (no path separator) must be looked up only through ``shutil.which``,
    never treated as a path relative to the current directory (contract: Platform naming)."""

    def test_non_executable_file_in_the_current_directory_is_not_used(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        not_executable = tmp_path / PLUGIN_ID
        not_executable.write_text("not a script")
        not_executable.chmod(0o600)
        with pytest.raises(BinaryUnavailableError):
            binary.resolve_binary(PLUGIN_ID, PLUGIN_ID, env={"PATH": os.defpath}, timeout=10.0)

    def test_executable_wrapper_script_in_the_current_directory_is_not_used(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        wrapper = tmp_path / PLUGIN_ID
        wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} -m mloda.testing.binary_model.simulated_binary "$@"\n')
        wrapper.chmod(0o700)
        with pytest.raises(BinaryUnavailableError):
            binary.resolve_binary(PLUGIN_ID, PLUGIN_ID, env={"PATH": os.defpath}, timeout=10.0)


class TestBareOverrideResolvedAgainstGivenEnvPath:
    """A bare override name must be looked up through the ``env["PATH"]`` argument itself, not the
    parent process's own ``PATH`` (contract: Platform naming, Data handling): probing and running a
    binary must be reproducible from the given environment alone."""

    @staticmethod
    def _write_stub_wrapper(tmp_path: Path) -> Path:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        wrapper = bin_dir / "stub-binary"
        wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} -m mloda.testing.binary_model.simulated_binary "$@"\n')
        wrapper.chmod(0o700)
        return bin_dir

    def test_bare_override_resolves_against_the_given_env_path(self, tmp_path: Path) -> None:
        bin_dir = self._write_stub_wrapper(tmp_path)
        resolved = binary.resolve_binary(PLUGIN_ID, "stub-binary", env={"PATH": str(bin_dir)}, timeout=10.0)
        assert resolved.capabilities.plugin_id == PLUGIN_ID

    def test_bare_override_ignores_the_parent_processs_own_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The parent process's own ``PATH`` is monkeypatched to contain the wrapper, while the
        ``env`` argument passed to ``resolve_binary`` does not: resolution must still fail,
        proving the parent process's PATH is never consulted."""
        bin_dir = self._write_stub_wrapper(tmp_path)
        monkeypatch.setenv("PATH", str(bin_dir))
        with pytest.raises(BinaryUnavailableError):
            binary.resolve_binary(PLUGIN_ID, "stub-binary", env={"PATH": os.defpath}, timeout=10.0)

    def test_bare_override_found_via_a_relative_path_entry_is_made_absolute(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``shutil.which`` returns a relative path for a relative ``PATH`` entry. The resolved argv
        is executed later from a different working directory, so it must be absolute by then."""
        bin_dir = self._write_stub_wrapper(tmp_path)
        monkeypatch.chdir(tmp_path)
        resolved = binary.resolve_binary(PLUGIN_ID, "stub-binary", env={"PATH": "bin"}, timeout=10.0)
        assert os.path.isabs(resolved.argv[0])
        assert resolved.argv[0] == str(bin_dir / "stub-binary")


class TestVersionMustBeSemVer:
    """The ``--version`` line's second token must be SemVer, the same pattern the conformance kit
    itself requires of every binary (contract: Invocation)."""

    @pytest.mark.parametrize(
        "mode",
        ["version_not_semver", "version_empty", "version_no_second_token", "version_non_ascii_digits"],
        ids=["not_semver", "empty", "no_second_token", "non_ascii_digits"],
    )
    def test_non_semver_version_is_unavailable(self, mode: str) -> None:
        with pytest.raises(BinaryUnavailableError) as excinfo:
            binary.resolve_binary(
                "faulty_binary", [*FAULTY_CMD, "--mode", mode], env={"PATH": os.defpath}, timeout=10.0
            )
        assert "--version" in str(excinfo.value)

    def test_prerelease_semver_version_is_accepted(self) -> None:
        resolved = binary.resolve_binary(
            "faulty_binary", [*FAULTY_CMD, "--mode", "version_prerelease"], env={"PATH": os.defpath}, timeout=10.0
        )
        assert resolved.capabilities.version == "0.0.1-rc.1+build.5"


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("a\n", ["a"], id="single_line_trailing_newline"),
        pytest.param("a", ["a"], id="single_line_no_trailing_newline"),
        pytest.param("a\u2028b\n", ["a\u2028b"], id="line_separator_not_split"),
        pytest.param("a\u2029b\n", ["a\u2029b"], id="paragraph_separator_not_split"),
        pytest.param("a\u0085b\n", ["a\u0085b"], id="next_line_not_split"),
        pytest.param("a\nb\n", ["a", "b"], id="two_lines"),
        pytest.param("a\n\n", ["a", ""], id="trailing_blank_line_kept"),
        pytest.param("", [], id="empty_text"),
    ],
)
def test_split_output_lines_drops_one_trailing_empty_element(text: str, expected: list[str]) -> None:
    """``contract.split_output_lines`` is the single source for the ``--capabilities`` line split
    (contract: Capabilities)."""
    assert contract.split_output_lines(text) == expected


class TestProbeTimeoutKillsDescendants:
    """A probe (``--version``/``--capabilities``) that times out must kill the whole process
    group, not just the probed leader, so a helper the leader spawned before hanging does not
    leak past the timeout (contract: Invocation, orphan detection)."""

    @pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX-only")
    def test_version_timeout_kills_a_child_spawned_by_the_probed_binary(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "child.pid"
        timeout = 3.0
        pid: int | None = None
        try:
            started = time.monotonic()
            with pytest.raises(BinaryUnavailableError, match="timed out probing --version"):
                binary.resolve_binary(
                    "faulty_binary",
                    [*FAULTY_CMD, "--mode", "version_hang_with_child"],
                    env={"PATH": os.defpath, "FAULTY_PID_FILE": str(pid_file)},
                    timeout=timeout,
                )
            elapsed = time.monotonic() - started
            assert elapsed < timeout + transport._GRACE_WAIT_SECONDS + transport._REAP_WAIT_SECONDS
            assert pid_file.exists(), "faulty_binary never wrote the child pid before the probe timed out"
            pid = int(pid_file.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 2.0
            while pid_running(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not pid_running(pid)
        finally:
            if pid is None and pid_file.exists():
                pid = int(pid_file.read_text(encoding="utf-8"))
            if pid is not None and pid_running(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass


class TestPublicParseFunctions:
    """``_parse_capabilities``/``_parse_version`` become the mixin's public parsing API
    (``binary.parse_capabilities``/``binary.parse_version``), so the conformance kit can delegate to
    them instead of duplicating the same checks and drifting out of sync (contract: Invocation,
    Capabilities)."""

    def test_parse_capabilities_is_public(self) -> None:
        """``binary.parse_capabilities`` must exist as a public name."""
        assert hasattr(binary, "parse_capabilities"), "binary.parse_capabilities does not exist yet"

    def test_parse_version_is_public(self) -> None:
        """``binary.parse_version`` must exist as a public name."""
        assert hasattr(binary, "parse_version"), "binary.parse_version does not exist yet"

    def test_parse_capabilities_rejects_boolean_contract_value(self) -> None:
        """A ``"contract": true`` value must be rejected: ``True == 1`` in Python, but the
        contract's own ``contract`` value is never a bool (contract: Capabilities)."""
        payload = (
            json.dumps(
                {
                    "contract": True,
                    "plugin_id": PLUGIN_ID,
                    "operations": ["hash"],
                    "column_types": sorted(contract.COLUMN_TYPES),
                }
            )
            + "\n"
        ).encode("utf-8")
        with pytest.raises(BinaryUnavailableError):
            binary.parse_capabilities(STUB_CMD, PLUGIN_ID, payload)

    def test_parse_capabilities_rejects_non_string_operations_element(self) -> None:
        """Every element of ``operations`` must be a string; a non-string element is rejected
        (contract: Capabilities)."""
        payload = (
            json.dumps(
                {
                    "contract": contract.CONTRACT_VERSION,
                    "plugin_id": PLUGIN_ID,
                    "operations": ["hash", 1],
                    "column_types": sorted(contract.COLUMN_TYPES),
                }
            )
            + "\n"
        ).encode("utf-8")
        with pytest.raises(BinaryUnavailableError):
            binary.parse_capabilities(STUB_CMD, PLUGIN_ID, payload)
