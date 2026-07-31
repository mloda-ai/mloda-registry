"""PEP 561 ``py.typed`` marker tests. A marker reaches a wheel only when its dotted path is listed
in ``[tool.setuptools] packages``, and ``mloda/community`` and ``mloda/enterprise`` are PEP 420
portions that ``discover_packages`` misses."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"
_VERIFY_BUILDS_PATH = _REPO_ROOT / "scripts" / "verify_builds.py"

# The distributions whose wheels must carry a PEP 561 marker.
_TYPED_PACKAGES = ["mloda-registry", "mloda-testing", "mloda-community", "mloda-enterprise"]


def _load_module(name: str, path: Path) -> ModuleType:
    """Import a loose script by file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"could not load spec for {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_module("generate_pyproject", _GEN_PATH)
vb = _load_module("verify_builds", _VERIFY_BUILDS_PATH)


def _packages() -> dict[str, dict[str, Any]]:
    """Config-declared packages, freshly loaded."""
    _shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    return packages


def _dotted_path(pkg_name: str) -> str:
    """Dotted import path of a configured package."""
    path: str = _packages()[pkg_name]["path"]
    return path.replace("/", ".")


def _setuptools_table(source: str, content: str) -> dict[str, Any]:
    """Return the parsed ``[tool.setuptools]`` table of a pyproject document."""
    data = tomllib.loads(content)
    table: dict[str, Any] = data.get("tool", {}).get("setuptools", {})
    assert table, f"{source}: [tool.setuptools] table is missing"
    return table


def _generated_setuptools(pkg_name: str) -> dict[str, Any]:
    """``[tool.setuptools]`` table the generator produces for a package."""
    shared, packages_config = gen.load_configs()
    packages = packages_config["packages"]
    content: str = gen.generate_pyproject(pkg_name, packages[pkg_name], shared, packages)
    return _setuptools_table(f"generated {pkg_name}", content)


def _committed_setuptools(pkg_name: str) -> dict[str, Any]:
    """``[tool.setuptools]`` table of the committed per-package pyproject.toml."""
    pyproject_path = _REPO_ROOT / _packages()[pkg_name]["path"] / "pyproject.toml"
    assert pyproject_path.exists(), f"{pyproject_path} is missing (run scripts/generate_pyproject.py)"
    return _setuptools_table(str(pyproject_path), pyproject_path.read_text())


@pytest.mark.parametrize("pkg_name", _TYPED_PACKAGES)
def test_config_flags_package_as_py_typed(pkg_name: str) -> None:
    pkg_config = _packages()[pkg_name]
    assert pkg_config.get("py_typed") is True, (
        f"{pkg_name}: config/packages.toml must declare 'py_typed = true', got {pkg_config.get('py_typed')!r}"
    )


@pytest.mark.parametrize("pkg_name", _TYPED_PACKAGES)
def test_marker_file_is_committed(pkg_name: str) -> None:
    marker = _REPO_ROOT / _packages()[pkg_name]["path"] / "py.typed"
    assert marker.is_file(), f"{pkg_name}: missing PEP 561 marker file {_packages()[pkg_name]['path']}/py.typed"


@pytest.mark.parametrize("pkg_name", _TYPED_PACKAGES)
def test_generator_emits_package_data_for_marker(pkg_name: str) -> None:
    dotted = _dotted_path(pkg_name)
    package_data = _generated_setuptools(pkg_name).get("package-data")
    assert package_data is not None, f"{pkg_name}: generated pyproject has no package-data table for {dotted}"
    assert package_data.get(dotted) == ["py.typed"], (
        f'{pkg_name}: [tool.setuptools.package-data] must contain "{dotted}" = ["py.typed"], got {package_data!r}'
    )


@pytest.mark.parametrize("pkg_name", _TYPED_PACKAGES)
def test_generator_lists_typed_path_as_package(pkg_name: str) -> None:
    """setuptools drops package-data for paths not listed in ``packages``."""
    dotted = _dotted_path(pkg_name)
    packages = _generated_setuptools(pkg_name).get("packages", [])
    assert dotted in packages, f"{pkg_name}: {dotted!r} is missing from [tool.setuptools] packages, got {packages!r}"


@pytest.mark.parametrize("pkg_name", _TYPED_PACKAGES)
def test_committed_pyproject_ships_marker_config(pkg_name: str) -> None:
    """``tox -e check-generated`` runs only in the package-integrity workflow, not in this gate."""
    dotted = _dotted_path(pkg_name)
    table = _committed_setuptools(pkg_name)
    assert table.get("package-data", {}).get(dotted) == ["py.typed"], (
        f'{pkg_name}: committed pyproject.toml lacks "{dotted}" = ["py.typed"] (run scripts/generate_pyproject.py)'
    )
    assert dotted in table.get("packages", []), (
        f"{pkg_name}: committed pyproject.toml does not list {dotted!r} (run scripts/generate_pyproject.py)"
    )


def test_package_data_is_emitted_only_for_flagged_packages() -> None:
    shared, packages_config = gen.load_configs()
    all_packages: dict[str, dict[str, Any]] = packages_config["packages"]

    flagged = sorted(name for name, cfg in all_packages.items() if cfg.get("py_typed"))
    assert flagged == sorted(_TYPED_PACKAGES), (
        f"config/packages.toml must flag exactly {sorted(_TYPED_PACKAGES)} with py_typed = true, got {flagged}"
    )

    for pkg_name, pkg_config in all_packages.items():
        content: str = gen.generate_pyproject(pkg_name, pkg_config, shared, all_packages)
        table = _setuptools_table(f"generated {pkg_name}", content)
        assert ("package-data" in table) == (pkg_name in flagged), (
            f"{pkg_name}: package-data table present={'package-data' in table}, "
            f"but py_typed is {pkg_config.get('py_typed')!r}"
        )


def _write_wheel(path: Path, names: list[str]) -> Path:
    """Write a zip standing in for a wheel holding exactly ``names``."""
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, "")
    return path


def _verifier() -> Callable[[dict[str, Path]], list[str]]:
    """The wheel-level marker check that verify_builds must expose."""
    verifier: Callable[[dict[str, Path]], list[str]] | None = getattr(vb, "verify_py_typed_markers", None)
    assert callable(verifier), "verify_builds.verify_py_typed_markers must be a callable"
    return verifier


def test_verify_py_typed_markers_reports_missing_marker(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path / "mloda_registry-0.0.0-py3-none-any.whl", ["mloda/registry/__init__.py"])

    errors = _verifier()({"mloda-registry": wheel})

    assert errors, "verify_py_typed_markers must report a wheel missing mloda/registry/py.typed"
    joined = " ".join(errors)
    assert "mloda-registry" in joined and "py.typed" in joined, joined


def test_verify_py_typed_markers_accepts_present_marker(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / "mloda_registry-0.0.0-py3-none-any.whl",
        ["mloda/registry/__init__.py", "mloda/registry/py.typed"],
    )

    assert _verifier()({"mloda-registry": wheel}) == []


def test_verify_py_typed_markers_ignores_unflagged_packages(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / "mloda_community_ffill-0.0.0-py3-none-any.whl",
        ["mloda/community/feature_groups/data_operations/row_preserving/ffill/__init__.py"],
    )

    assert _verifier()({"mloda-community-ffill": wheel}) == []
