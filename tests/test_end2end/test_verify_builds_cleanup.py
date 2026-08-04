"""Stale build-artifact cleanup for scripts/verify_builds.py. setuptools leaves a ``build/lib`` copy
of the sources behind, and the next build copies it into the wheel, so a build can ship files the
source tree no longer has."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERIFY_BUILDS_PATH = _REPO_ROOT / "scripts" / "verify_builds.py"

# Configured package paths that get a stale build/ tree in the fake workspace.
_STALE_BUILD_PACKAGES = ["mloda/registry", "mloda/community", "mloda/testing"]

vb = load_script("verify_builds", _VERIFY_BUILDS_PATH)


def _cleanup_build_dirs() -> Callable[[], int]:
    """The stale-build-tree cleanup that verify_builds must expose."""
    cleanup: Callable[[], int] | None = getattr(vb, "cleanup_build_dirs", None)
    assert callable(cleanup), "verify_builds.cleanup_build_dirs must be a callable"
    return cleanup


def _write(path: Path, content: str = "") -> Path:
    """Create a file and its parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _make_workspace(root: Path) -> tuple[list[Path], list[Path]]:
    """Fake workspace with stale build trees; returns ``(build dirs, paths that must survive)``."""
    # The real config is copied in because the script's paths are CWD relative.
    (root / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(_REPO_ROOT / "config" / "packages.toml", root / "config" / "packages.toml")

    # setuptools mirrors the sources into build/lib/<dotted path>.
    _write(root / "mloda/community/build/lib/mloda/community/py.typed")
    _write(root / "mloda/registry/build/lib/mloda/registry/discover.py", "def discover() -> None: ...\n")
    (root / "mloda/testing/build").mkdir(parents=True, exist_ok=True)

    build_dirs = [root / pkg_path / "build" for pkg_path in _STALE_BUILD_PACKAGES]
    survivors = [
        root / "config" / "packages.toml",
        _write(root / "mloda/community/py.typed"),
        _write(root / "mloda/community/feature_groups/example/manifest.py", "FEATURE_GROUPS = []\n"),
        _write(root / "mloda/registry/pyproject.toml", "[project]\n"),
        _write(root / "mloda/testing/py.typed"),
    ]
    return build_dirs, survivors


def test_cleanup_build_dirs_removes_stale_trees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_dirs, _survivors = _make_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    removed = _cleanup_build_dirs()()

    for build_dir in build_dirs:
        assert not build_dir.exists(), f"{build_dir.relative_to(tmp_path)} survived cleanup_build_dirs()"
    assert removed == len(build_dirs), f"expected {len(build_dirs)} removed build directories, got {removed!r}"


def test_cleanup_build_dirs_leaves_sources_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _build_dirs, survivors = _make_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    _cleanup_build_dirs()()

    for path in survivors:
        assert path.exists(), f"cleanup_build_dirs() removed non-artifact {path.relative_to(tmp_path)}"


def test_cleanup_build_dirs_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _build_dirs, survivors = _make_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    cleanup = _cleanup_build_dirs()
    cleanup()

    assert cleanup() == 0, "a second cleanup_build_dirs() run must report zero removed directories"
    for path in survivors:
        assert path.exists(), f"the second cleanup_build_dirs() run removed {path.relative_to(tmp_path)}"


def test_cleanup_build_dirs_tolerates_missing_package_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config").mkdir()
    shutil.copy(_REPO_ROOT / "config" / "packages.toml", tmp_path / "config" / "packages.toml")
    monkeypatch.chdir(tmp_path)

    assert _cleanup_build_dirs()() == 0


class _FakeCompletedProcess:
    """Stand-in for subprocess.CompletedProcess; main() reads returncode and stderr."""

    def __init__(self, returncode: int = 1) -> None:
        self.returncode = returncode
        self.stdout = ""
        self.stderr = "stubbed: the pytest gate never runs a real build"


def test_main_cleans_build_dirs_before_building(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cleaning after the loop still lets a run read the previous run's artifacts."""
    assert callable(getattr(vb, "cleanup_build_dirs", None)), "verify_builds.cleanup_build_dirs must be a callable"

    calls: list[str] = []

    def _record_cleanup() -> int:
        calls.append("cleanup_build_dirs")
        return 0

    def _record_run(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        calls.append("build")
        return _FakeCompletedProcess()

    monkeypatch.setattr(vb, "cleanup_build_dirs", _record_cleanup)
    monkeypatch.setattr(vb.subprocess, "run", _record_run)

    vb.main()

    assert "build" in calls, "fixture assumption: main() must reach the build loop"
    assert calls[0] == "cleanup_build_dirs", (
        f"main() must remove stale build/ trees before the first build, call order was {calls[:3]}"
    )


def test_main_leaves_no_build_dirs_after_a_successful_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Trees left by a green run make the next pytest run collect phantom ``build.lib.*`` modules."""
    build_dirs, _survivors = _make_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    built_packages: list[str] = []

    def _fake_build(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        """Write what a real build writes: a build/lib tree, plus a wheel in --out-dir."""
        if not built_packages:
            for build_dir in build_dirs:
                assert not build_dir.exists(), (
                    f"fixture assumption: {build_dir.relative_to(tmp_path)} must be gone before the first build"
                )
        package = cmd[cmd.index("--package") + 1]
        built_packages.append(package)
        for build_dir in build_dirs:
            _write(build_dir / "lib" / "mloda" / "community" / "py.typed")
        _write(Path(cmd[cmd.index("--out-dir") + 1]) / f"{package.replace('-', '_')}-0.0.0-py3-none-any.whl")
        return _FakeCompletedProcess(returncode=0)

    def _consistent_versions() -> tuple[bool, str]:
        return True, "0.0.0"

    def _version_matches(wheel_path: Path, expected: str) -> bool:
        return True

    def _no_errors(*args: Any, **kwargs: Any) -> list[str]:
        return []

    monkeypatch.setattr(vb.subprocess, "run", _fake_build)
    monkeypatch.setattr(vb, "check_version_consistency", _consistent_versions)
    monkeypatch.setattr(vb, "verify_wheel_version", _version_matches)
    for verifier in (
        "verify_dependency_relationships",
        "verify_wheel_metadata",
        "verify_entry_points",
        "verify_py_typed_markers",
        "verify_pep420_source_compliance",
    ):
        monkeypatch.setattr(vb, verifier, _no_errors)

    exit_code = vb.main()

    assert built_packages, "fixture assumption: main() must reach the build loop"
    assert exit_code == 0, "fixture assumption: the stubs must drive main() down its success path"
    for build_dir in build_dirs:
        assert not build_dir.exists(), f"a successful main() left {build_dir.relative_to(tmp_path)} behind"
