"""Meta-tests for the Windows-environment handling in ``BinaryModelConformanceBase``: monkeypatch
``os.name`` (this codebase's convention, since ``sys.platform`` comparisons are dead-code-eliminated
by ``mypy --strict`` on this Linux CI) and call the affected conformance methods directly on a bare
subclass instance, so no real win32 runner is needed."""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pytest

from mloda.testing.binary_model.conformance import BinaryModelConformanceBase, write_text


class _ConformanceHarness(BinaryModelConformanceBase):
    """Not collected by pytest (no ``Test`` prefix): a bare instance to call conformance methods
    directly instead of re-running the whole suite under a new class."""


def test_input_path_not_readable_skips_on_win32(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No POSIX ``os.geteuid`` (as on real Windows) must skip for that specific reason, not fall
    through to exercising the chmod path."""
    conformance = _ConformanceHarness()
    license_path = write_text(tmp_path / "license.txt", conformance.valid_license_text)
    valid_license_env = {"MLODA_LICENSE_FILE": str(license_path)}
    monkeypatch.delattr(os, "geteuid", raising=False)
    with pytest.raises(pytest.skip.Exception, match="no POSIX file permissions"):
        conformance.test_input_path_not_readable_is_usage_error(valid_license_env, tmp_path)


def test_minimal_environment_forwards_systemroot_on_win32(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """On ``os.name == "nt"`` with ``SYSTEMROOT`` set, the env passed to ``run_binary`` must carry
    exactly the license key, ``PATH``, and ``SYSTEMROOT`` -- nothing else."""
    conformance = _ConformanceHarness()
    # warm pyarrow's lazy sysconfig-touching init under the real os.name before faking it below
    pa.array([0], type=pa.int64())
    monkeypatch.setattr(os, "name", "nt")
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
    assert captured_env == {
        "MLODA_LICENSE_KEY": conformance.valid_license_text,
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": "C:\\Windows",
    }


def test_platform_env_adds_systemroot_on_nt(monkeypatch: pytest.MonkeyPatch) -> None:
    """``platform_env`` adds ``SYSTEMROOT`` from the host environment when ``os.name == "nt"``,
    without mutating its input."""
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    conformance = _ConformanceHarness()
    source_env = {"FOO": "bar"}
    result = conformance.platform_env(source_env)
    assert result == {"FOO": "bar", "SYSTEMROOT": "C:\\Windows"}
    assert source_env == {"FOO": "bar"}


def test_platform_env_noop_without_systemroot_on_nt(monkeypatch: pytest.MonkeyPatch) -> None:
    """``platform_env`` leaves the env untouched on ``os.name == "nt"`` when ``SYSTEMROOT`` is
    unset."""
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.delenv("SYSTEMROOT", raising=False)
    conformance = _ConformanceHarness()
    result = conformance.platform_env({"FOO": "bar"})
    assert result == {"FOO": "bar"}


def test_platform_env_noop_off_nt(monkeypatch: pytest.MonkeyPatch) -> None:
    """``platform_env`` never adds ``SYSTEMROOT`` off ``os.name == "nt"``, even if it happens to be
    set already."""
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    conformance = _ConformanceHarness()
    result = conformance.platform_env({"FOO": "bar"})
    assert result == {"FOO": "bar"}
