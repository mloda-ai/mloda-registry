"""Meta-tests for the win32 guards in ``BinaryModelConformanceBase``: monkeypatch ``sys.platform``
to ``"win32"`` and call the two affected conformance methods directly on a bare subclass instance,
so no real win32 runner is needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mloda.testing.binary_model.conformance import BinaryModelConformanceBase, write_text


class _ConformanceHarness(BinaryModelConformanceBase):
    """Not collected by pytest (no ``Test`` prefix): a bare instance to call conformance methods
    directly instead of re-running the whole suite under a new class."""


def test_input_path_not_readable_skips_on_win32(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`chmod(0o000)` is a no-op on Windows, so the check must skip when ``sys.platform ==
    "win32"``."""
    conformance = _ConformanceHarness()
    license_path = write_text(tmp_path / "license.txt", conformance.valid_license_text)
    valid_license_env = {"MLODA_LICENSE_FILE": str(license_path)}
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(pytest.skip.Exception):
        conformance.test_input_path_not_readable_is_usage_error(valid_license_env, tmp_path)


def test_minimal_environment_forwards_systemroot_on_win32(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """On ``sys.platform == "win32"`` with ``SYSTEMROOT`` set, the env passed to ``run_binary``
    must carry it."""
    conformance = _ConformanceHarness()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    captured_env: dict[str, str] = {}

    def fake_run_binary(
        cmd: list[str],
        args: list[str],
        env: dict[str, str],
        input_bytes: bytes = b"",
        timeout: float = 10.0,
        cwd: Path | None = None,
    ) -> None:
        captured_env.update(env)
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr("mloda.testing.binary_model.conformance.run_binary", fake_run_binary)
    with pytest.raises(RuntimeError, match="stop-after-capture"):
        conformance.test_minimal_environment_allowlist_only(tmp_path)
    assert captured_env.get("SYSTEMROOT") == "C:\\Windows"
