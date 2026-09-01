"""Tests for ``binary.py``: resolving a binary from an explicit override or the installed wheel,
probing ``--version``/``--capabilities`` up front (contract: Invocation, Capabilities), and
caching the probe result per process so a warm binary is never re-probed.
"""

from __future__ import annotations

import os
import subprocess  # nosec
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mloda.community.feature_groups.binary_model import binary
from mloda.community.feature_groups.binary_model.errors import BinaryUnavailableError

STUB_CMD = [sys.executable, "-m", "mloda.testing.binary_model.simulated_binary"]
FAULTY_CMD = [sys.executable, "-m", "mloda.community.feature_groups.binary_model.tests.faulty_binary"]
PLUGIN_ID = "example_binary"


@pytest.fixture(autouse=True)
def _clear_capability_cache_before_each_test() -> None:
    binary.clear_capability_cache()


class _CountingRun:
    """Wraps the real ``subprocess.run`` to count calls, so cache-hit tests can assert no process
    was spawned. Patches the stdlib ``subprocess`` module object directly (not
    ``binary.subprocess``, which mypy's ``--strict`` (no implicit re-export) rejects from outside
    the module): ``binary.py`` doing ``import subprocess`` and calling ``subprocess.run(...)``
    shares this very same module object at runtime, so patching it here is equally effective.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        real_run = subprocess.run

        def counting_run(*args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return real_run(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", counting_run)


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
            column_types=binary.COLUMN_TYPE_VOCABULARY,
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

    def test_capabilities_not_json_is_unavailable(self) -> None:
        with pytest.raises(BinaryUnavailableError):
            binary.resolve_binary(
                "faulty_binary",
                [*FAULTY_CMD, "--mode", "capabilities_not_json"],
                env={"PATH": os.defpath},
                timeout=10.0,
            )

    def test_version_two_lines_is_unavailable(self) -> None:
        with pytest.raises(BinaryUnavailableError):
            binary.resolve_binary(
                "faulty_binary", [*FAULTY_CMD, "--mode", "version_two_lines"], env={"PATH": os.defpath}, timeout=10.0
            )


class TestCapabilityCache:
    def test_second_call_with_same_argv_spawns_no_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counter = _CountingRun(monkeypatch)
        binary.resolve_binary(PLUGIN_ID, STUB_CMD, env={"PATH": os.defpath}, timeout=10.0)
        warm_calls = counter.count
        assert warm_calls > 0
        binary.resolve_binary(PLUGIN_ID, STUB_CMD, env={"PATH": os.defpath}, timeout=10.0)
        assert counter.count == warm_calls

    def test_clear_capability_cache_forces_reprobe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counter = _CountingRun(monkeypatch)
        binary.resolve_binary(PLUGIN_ID, STUB_CMD, env={"PATH": os.defpath}, timeout=10.0)
        warm_calls = counter.count
        binary.clear_capability_cache()
        binary.resolve_binary(PLUGIN_ID, STUB_CMD, env={"PATH": os.defpath}, timeout=10.0)
        assert counter.count > warm_calls

    def test_different_argv_suffix_is_a_different_cache_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counter = _CountingRun(monkeypatch)
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
