"""Wheel-to-package binding in scripts/verify_builds.py. Every package builds into one shared temp
directory, so a prefix glob lets a package adopt a sibling's wheel: ``mloda_community*.whl`` also
matches ``mloda_community_offset-0.4.0-py3-none-any.whl``."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERIFY_BUILDS_PATH = _REPO_ROOT / "scripts" / "verify_builds.py"

_VERSION = "0.4.0"

# Children before the bundle whose name is their prefix, the inverse of config/packages.toml today.
_REORDERED_NAMES = ["mloda-community-example", "mloda-community-offset", "mloda-community", "mloda-registry"]


def _load_module(name: str, path: Path) -> ModuleType:
    """Import a loose script by file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"could not load spec for {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vb = _load_module("verify_builds", _VERIFY_BUILDS_PATH)


class _FakeCompletedProcess:
    """Stand-in for subprocess.CompletedProcess; main() reads returncode and stderr."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.stdout = ""
        self.stderr = "stubbed: the pytest gate never runs a real build"


def _wheel_name(pkg_name: str) -> str:
    """Wheel filename a real build produces for a distribution."""
    return f"{pkg_name.replace('-', '_')}-{_VERSION}-py3-none-any.whl"


def _sandbox(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Run main() against a copy of the config; returns the reordered (name, pyproject path) entries."""
    (root / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(_REPO_ROOT / "config" / "packages.toml", root / "config" / "packages.toml")
    monkeypatch.chdir(root)

    paths = dict(vb.load_packages_from_config())
    missing = [name for name in _REORDERED_NAMES if name not in paths]
    assert not missing, f"fixture assumption: config/packages.toml must declare {missing}"
    return [(name, paths[name]) for name in _REORDERED_NAMES]


def _stub_build(
    monkeypatch: pytest.MonkeyPatch,
    packages: list[tuple[str, str]],
    skip_wheel: str | None = None,
) -> dict[str, Path]:
    """Drive main() with fake builds; returns the built_wheels mapping, filled in once main() runs."""
    captured: dict[str, Path] = {}

    def _fake_build(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        package = cmd[cmd.index("--package") + 1]
        if package != skip_wheel:
            (Path(cmd[cmd.index("--out-dir") + 1]) / _wheel_name(package)).write_bytes(b"")
        return _FakeCompletedProcess()

    def _consistent_versions() -> tuple[bool, str]:
        return True, _VERSION

    def _version_matches(wheel_path: Path, expected: str) -> bool:
        return True

    def _capture_wheels(built_wheels: dict[str, Path]) -> list[str]:
        captured.update(built_wheels)
        return []

    def _no_errors(*args: Any, **kwargs: Any) -> list[str]:
        return []

    monkeypatch.setattr(vb, "PACKAGES", packages)
    monkeypatch.setattr(vb.subprocess, "run", _fake_build)
    monkeypatch.setattr(vb, "check_version_consistency", _consistent_versions)
    monkeypatch.setattr(vb, "verify_wheel_version", _version_matches)
    monkeypatch.setattr(vb, "verify_entry_points", _capture_wheels)
    for verifier in (
        "verify_dependency_relationships",
        "verify_wheel_metadata",
        "verify_py_typed_markers",
        "verify_pep420_source_compliance",
    ):
        monkeypatch.setattr(vb, verifier, _no_errors)
    return captured


def test_every_package_binds_its_own_wheel_under_a_reordered_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config order must not decide which wheel a package is verified against."""
    packages = _sandbox(tmp_path, monkeypatch)
    built_wheels = _stub_build(monkeypatch, packages)

    exit_code = vb.main()

    assert exit_code == 0, "fixture assumption: the stubs must drive main() down its success path"
    assert sorted(built_wheels) == sorted(_REORDERED_NAMES), (
        f"main() bound wheels for {sorted(built_wheels)}, expected {sorted(_REORDERED_NAMES)}"
    )
    for pkg_name, wheel_path in built_wheels.items():
        assert wheel_path.name == _wheel_name(pkg_name), (
            f"{pkg_name}: bound {wheel_path.name}, a wheel carrying another distribution's name; "
            f"expected {_wheel_name(pkg_name)}"
        )


def test_a_prefix_sibling_wheel_is_never_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With its own wheel absent, a package must report no wheel instead of adopting a sibling's."""
    packages = _sandbox(tmp_path, monkeypatch)
    built_wheels = _stub_build(monkeypatch, packages, skip_wheel="mloda-community")

    exit_code = vb.main()
    output = capsys.readouterr().out

    assert "mloda-community" not in built_wheels, (
        f"main() bound {built_wheels.get('mloda-community')} to mloda-community, which built no wheel"
    )
    assert "mloda-community: no wheel produced" in output, (
        f"main() must report the missing mloda-community wheel, printed:\n{output}"
    )
    assert exit_code == 1, f"main() must fail when a package produces no wheel, returned {exit_code!r}"
