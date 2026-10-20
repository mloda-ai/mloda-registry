"""Tests for the shared clean-venv install, probe and wheel-build helpers in scripts/verify_common.py."""

from __future__ import annotations

import subprocess  # nosec
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.script_loader import load_script
from tests.test_end2end.test_verify_independent_installs import _FakeCompletedProcess

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMON_PATH = _REPO_ROOT / "scripts" / "verify_common.py"


def _common() -> ModuleType:
    """The shared helper module."""
    assert _COMMON_PATH.is_file(), f"{_COMMON_PATH} does not exist: the shared verify helpers are missing"
    return load_script("verify_common", _COMMON_PATH)


def _record(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, _FakeCompletedProcess] | None = None
) -> list[tuple[list[str], dict[str, Any]]]:
    """Patch subprocess.run to record (command, kwargs); a command containing a results key gets that result."""
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        calls.append((command, kwargs))
        for key, result in (results or {}).items():
            if key in command:
                return result
        return _FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_install_runs_venv_then_install_in_tmpdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two commands, both in tmpdir; the venv lives at <tmpdir>/venv; the install ends with every requirement."""
    common = _common()
    calls = _record(monkeypatch)

    error = common.install(["pkg-a==1", "pkg-b"], str(tmp_path))

    assert error is None, f"install reported {error}"
    assert len(calls) == 2, f"expected a venv and an install command, got {[c for c, _ in calls]}"
    assert str(tmp_path / "venv") in calls[0][0], f"the venv command must target <tmpdir>/venv: {calls[0][0]}"
    assert calls[1][0][-2:] == ["pkg-a==1", "pkg-b"], f"the install must end with all requirements: {calls[1][0]}"
    for _, kwargs in calls:
        assert Path(kwargs["cwd"]) == tmp_path, f"every command must run with cwd == tmpdir, got {kwargs.get('cwd')}"


def test_failure_message_falls_back_to_stdout_when_stderr_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing probe with empty stderr reports its stdout."""
    common = _common()
    _record(monkeypatch, {"-c": _FakeCompletedProcess(returncode=1, stdout="from stdout", stderr="")})

    error = common.install_and_probe(["pkg-a"], "import pkg_a")

    assert error is not None and "from stdout" in error, f"stdout must be used when stderr is empty, got {error!r}"


def test_build_wheel_success_returns_the_wheel_built_in_the_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """build_wheel returns (wheel, None) for the single wheel written, building with cwd at the repo root."""
    common = _common()
    calls: list[dict[str, Any]] = []

    def fake_run(command: list[str], *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        calls.append(kwargs)
        (tmp_path / "mloda_community_demo-1.0-py3-none-any.whl").write_bytes(b"")
        return _FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)

    wheel, error = common.build_wheel("mloda-community-demo", tmp_path)

    assert error is None, f"build_wheel reported {error}"
    assert wheel == tmp_path / "mloda_community_demo-1.0-py3-none-any.whl", f"unexpected wheel {wheel}"
    assert len(calls) == 1 and Path(calls[0]["cwd"]) == _REPO_ROOT, f"build must run in the repo root: {calls}"


def test_build_wheel_failure_keeps_the_stderr_tail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A failed build returns (None, error) naming the package and keeping only the last 500 chars of stderr."""
    common = _common()
    stderr = "HEAD" + "x" * 600 + "TAIL"
    _record(monkeypatch, {"build": _FakeCompletedProcess(returncode=1, stderr=stderr)})

    wheel, error = common.build_wheel("mloda-community-demo", tmp_path)

    assert wheel is None, f"a failed build must return no wheel, got {wheel}"
    assert error is not None and "mloda-community-demo" in error and "build failed" in error, f"got {error!r}"
    assert error.endswith("TAIL"), f"the stderr tail must be kept: {error!r}"
    assert "HEAD" not in error, f"only the last 500 chars of stderr belong in the error: {error!r}"
