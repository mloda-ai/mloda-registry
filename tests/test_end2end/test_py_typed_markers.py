"""PEP 561 ``py.typed`` marker tests. A marker reaches a wheel only when its dotted path is listed
in ``[tool.setuptools] packages``, and ``mloda/community`` and ``mloda/enterprise`` are PEP 420
portions that ``discover_packages`` misses."""

from __future__ import annotations

import re
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest

from tests.script_loader import load_script, version_tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"
_VERIFY_BUILDS_PATH = _REPO_ROOT / "scripts" / "verify_builds.py"
_PUBLISHED_PACKAGES_PATH = _REPO_ROOT / "scripts" / "published_packages.py"

# The bundle distributions, always part of the released set.
_BUNDLES = ["mloda-registry", "mloda-testing", "mloda-community", "mloda-enterprise"]

# The distributions whose wheels must carry a PEP 561 marker. mypy returns at the first py.typed on the
# module path, so the two ancestor markers also type the leaf distributions shipped from below them.
_TYPED_PACKAGES = [*_BUNDLES, "mloda-community-data-operations", "mloda-community-example", "mloda-community-otel"]

# Confirmed by inspecting the published wheels directly: py.typed first appears in each base's 0.4.1 wheel.
_MARKER_FLOORS = {
    "mloda-community-data-operations": "0.4.1",
    "mloda-community-example": "0.4.1",
    "mloda-community-otel": "0.4.6",
}


gen = load_script("generate_pyproject", _GEN_PATH)
vb = load_script("verify_builds", _VERIFY_BUILDS_PATH)


def _packages() -> dict[str, dict[str, Any]]:
    """Config-declared packages, freshly loaded."""
    _shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    return packages


def _dotted_path(pkg_name: str) -> str:
    """Dotted import path of a configured package."""
    path: str = _packages()[pkg_name]["path"]
    return path.replace("/", ".")


def _published_packages() -> list[str]:
    """Distribution names flagged ``published = true`` in config/packages.toml, the released set."""
    assert _PUBLISHED_PACKAGES_PATH.exists(), (
        f"{_PUBLISHED_PACKAGES_PATH} is missing; it reads the released set from config/packages.toml"
    )
    pub = load_script("published_packages", _PUBLISHED_PACKAGES_PATH)
    names: list[str] = pub.published_packages(_packages())
    assert set(_BUNDLES) <= set(names), (
        f"config/packages.toml lost the published flag on bundles {sorted(set(_BUNDLES) - set(names))}"
    )
    return names


def _dependency_closure(pkg_name: str, packages: dict[str, dict[str, Any]]) -> set[str]:
    """Configured packages transitively reachable through ``dependencies``, extras excluded."""
    seen: set[str] = set()
    queue = [pkg_name]
    while queue:
        node = queue.pop()
        for dep in packages.get(node, {}).get("dependencies", []):
            match = re.match(r"[A-Za-z0-9._-]+", dep.strip())
            name = re.sub(r"[-_.]+", "-", match.group(0)).lower() if match else ""
            if name in packages and name not in seen:
                seen.add(name)
                queue.append(name)
    return seen


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


@pytest.mark.parametrize("pkg_name", _published_packages())
def test_every_published_distribution_has_a_typed_ancestor(pkg_name: str) -> None:
    """Every published distribution needs a py.typed at or above its path, from itself or a dependency."""
    # test_package_data_is_emitted_only_for_flagged_packages pins the flagged set, so a newly published
    # package can ship untyped without failing it.
    packages = _packages()
    pkg_path: str = packages[pkg_name]["path"]
    reachable = {pkg_name} | _dependency_closure(pkg_name, packages)
    typed = {name: cfg["path"] for name, cfg in packages.items() if cfg.get("py_typed")}

    covering = [
        name
        for name, typed_path in typed.items()
        if name in reachable and (pkg_path == typed_path or pkg_path.startswith(typed_path + "/"))
    ]
    assert covering, (
        f"{pkg_name} ({pkg_path}) ships untyped: none of its own or dependency-reachable packages carries a "
        f"py.typed at or above that path. It needs either 'py_typed = true' plus a committed "
        f"{pkg_path}/py.typed, or a dependency on a package that already ships an ancestor marker."
    )


def test_marker_floors_cover_exactly_the_non_bundle_typed_packages() -> None:
    """_MARKER_FLOORS must track _TYPED_PACKAGES minus the bundles: those are the only "shared base" typed
    packages leaves declare a dependency floor on. If a third typed base is ever added to _TYPED_PACKAGES
    without a matching _MARKER_FLOORS entry, this must fail instead of the floor-enforcement test below
    silently covering fewer packages than it should."""
    assert set(_MARKER_FLOORS) == set(_TYPED_PACKAGES) - set(_BUNDLES), (
        f"_MARKER_FLOORS {sorted(_MARKER_FLOORS)} must equal _TYPED_PACKAGES minus _BUNDLES "
        f"{sorted(set(_TYPED_PACKAGES) - set(_BUNDLES))}"
    )


def test_marker_floor_versions_are_at_or_below_the_shared_project_version() -> None:
    """Every sibling floor the generator emits is [project].version itself, so the only invariant left
    to check is that the shared version hasn't regressed below a recorded py.typed marker floor."""
    shared, _packages_config = gen.load_configs()
    current = version_tuple(shared["project"]["version"])
    for base_name, first in _MARKER_FLOORS.items():
        assert current >= version_tuple(first), (
            f"{base_name}: [project].version {shared['project']['version']!r} is below the recorded py.typed "
            f"marker floor {first!r}"
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
