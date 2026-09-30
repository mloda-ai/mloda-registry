"""Meta-tests for the Windows-environment handling in ``BinaryModelConformanceBase``: a monkeypatched
``os.name`` (this codebase's convention, since ``sys.platform`` comparisons are dead-code-eliminated by
``mypy --strict``) lets them run on any host, including the Windows CI job; the POSIX-shape case skips
off POSIX."""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pytest

from mloda.community.feature_groups.binary_model.transport import minimal_environment
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
    monkeypatch.delenv("PATH", raising=False)
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
        "PATH": os.defpath,
        "SYSTEMROOT": "C:\\Windows",
    }


def test_platform_env_layers_env_on_minimal_environment_on_nt(monkeypatch: pytest.MonkeyPatch) -> None:
    """On ``os.name == "nt"`` with ``SYSTEMROOT`` set, ``platform_env`` is the production minimal
    environment plus the given entries, without mutating its input."""
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    monkeypatch.setenv("PATH", "/host/bin")
    conformance = _ConformanceHarness()
    source_env = {"FOO": "bar"}
    result = conformance.platform_env(source_env)
    assert result == {**minimal_environment(inherit_license=False), "FOO": "bar"}
    assert result["SYSTEMROOT"] == "C:\\Windows"
    assert "PATH" in result
    assert source_env == {"FOO": "bar"}


def test_platform_env_omits_systemroot_when_unset_on_nt(monkeypatch: pytest.MonkeyPatch) -> None:
    """On ``os.name == "nt"`` without ``SYSTEMROOT``, ``platform_env`` adds ``PATH`` but no
    ``SYSTEMROOT`` and keeps the given entries."""
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.delenv("SYSTEMROOT", raising=False)
    monkeypatch.setenv("PATH", "/host/bin")
    conformance = _ConformanceHarness()
    result = conformance.platform_env({"FOO": "bar"})
    assert "SYSTEMROOT" not in result
    assert "PATH" in result
    assert result["FOO"] == "bar"


@pytest.mark.skipif(os.name != "posix", reason="asserts the POSIX-shaped minimal environment")
def test_platform_env_delegates_to_minimal_environment_off_nt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off ``os.name == "nt"``, a host ``SYSTEMROOT`` never appears and the result is the production
    minimal environment plus the given entries."""
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    conformance = _ConformanceHarness()
    result = conformance.platform_env({"FOO": "bar"})
    assert "SYSTEMROOT" not in result
    assert result == {**minimal_environment(inherit_license=False), "FOO": "bar"}


def test_platform_env_explicit_entries_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entries passed to ``platform_env`` override the minimal-environment defaults."""
    monkeypatch.setenv("PATH", "/host/bin")
    explicit = {"PATH": "/explicit", "MLODA_LICENSE_KEY": "", "MLODA_LICENSE_FILE": ""}
    result = _ConformanceHarness().platform_env(explicit)
    for key, value in explicit.items():
        assert result[key] == value


def test_platform_env_does_not_leak_ambient_license(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host license variables never reach the env built by ``platform_env``."""
    monkeypatch.setenv("MLODA_LICENSE_KEY", "ambient-key")
    monkeypatch.setenv("MLODA_LICENSE_FILE", "/ambient/license.txt")
    result = _ConformanceHarness().platform_env({})
    assert "MLODA_LICENSE_KEY" not in result
    assert "MLODA_LICENSE_FILE" not in result
