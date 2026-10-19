"""Backend extras declared by data-operation leaf distributions must match what their manifest registers."""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from tests.script_loader import load_script
from tests.toml_loader import load_toml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"
_script = load_script("verify_backend_extras", _REPO_ROOT / "scripts" / "verify_backend_extras.py")

# Extras that never name a compute framework, so they're excluded from the backend comparison.
_NON_FRAMEWORK_EXTRAS = frozenset({"all", "dev"})

# Backends whose extra depends on exactly their own pip package.
_PACKAGED_BACKENDS = {"pyarrow", "polars", "pandas"}

# Backends whose extra must mirror mloda core's same-named extra exactly (duckdb and sqlite need pyarrow too).
_CORE_MIRRORED_BACKENDS = {"duckdb", "sqlite"}

# Per-package polars floors above mloda core's, where a leaf needs a newer polars than core requires.
# frame-aggregate: time-window rolling_*_by rejects null values below polars 1.38.
_POLARS_FLOOR_OVERRIDES: dict[str, str] = {"mloda-community-frame-aggregate": ">=1.38"}


def _dep_name(spec: str) -> str:
    """Bare package name from a PEP 508 requirement string, e.g. 'pandas>=2.2' -> 'pandas'."""
    return re.split(r"[<>=!~;\s\[(@]", spec.strip(), maxsplit=1)[0]


def _min_version(specifier: SpecifierSet) -> Version:
    """Version of the '>=' specifier in a SpecifierSet, e.g. '>=1.21' -> Version('1.21')."""
    for spec in specifier:
        if spec.operator == ">=":
            return Version(spec.version)
    raise AssertionError(f"specifier {specifier} has no '>=' floor to compare")


def _core_extra_requirements(extra: str) -> set[Requirement]:
    """Mloda core's requirements for its '<extra>' extra, from installed metadata, marker cleared."""
    selected: set[Requirement] = set()
    for dep in importlib.metadata.requires("mloda") or []:
        req = Requirement(dep)
        # Textual match, not marker evaluation: a clause such as python_version < "3.11" must not hide a requirement.
        if req.marker is None or f'extra == "{extra}"' not in str(req.marker):
            continue
        assert str(req.marker) == f'extra == "{extra}"', (
            f"mloda core requirement {dep!r} has a marker beyond the '{extra}' extra; extend this helper"
        )
        req.marker = None
        selected.add(req)
    if not selected:
        raise AssertionError(f"mloda core declares no requirements for its '{extra}' extra; guard is vacuous")
    return selected


def _core_extra_requirement(name: str) -> Requirement:
    """Mloda core's own '<name>' extra requirement on <name>, read from installed distribution metadata."""
    for req in _core_extra_requirements(name):
        if req.name == name:
            return req
    raise AssertionError(f"mloda core does not declare a '{name}' extra requirement on {name}; guard is vacuous")


def _packages() -> dict[str, dict[str, Any]]:
    data = load_toml(_PACKAGES_CONFIG)
    packages: dict[str, dict[str, Any]] = data["packages"]
    return packages


_LEAF_PACKAGES = _script.leaf_packages(_packages())
assert _LEAF_PACKAGES, "no data_operations leaf packages found under config/packages.toml; guard is vacuous"
_CORE_POLARS_REQUIREMENT = _core_extra_requirement("polars")
_CORE_PYARROW_REQUIREMENT = _core_extra_requirement("pyarrow")
_CORE_MIRRORED_REQUIREMENTS = {backend: _core_extra_requirements(backend) for backend in _CORE_MIRRORED_BACKENDS}
# sqlite has no pip package of its own, so only duckdb is checked for naming its backend package.
assert "duckdb" in {req.name for req in _CORE_MIRRORED_REQUIREMENTS["duckdb"]}, (
    "mloda core's 'duckdb' extra no longer requires duckdb itself; the mirror guard would be vacuous"
)


@pytest.mark.parametrize("package_name,entry", _LEAF_PACKAGES, ids=[name for name, _ in _LEAF_PACKAGES])
def test_leaf_package_extras_match_registered_backends(package_name: str, entry: dict[str, Any]) -> None:
    specs = _script.manifest_specs(_REPO_ROOT / entry["path"] / "manifest.py")
    backends = {_script.backend_for_module(module) for module, _ in specs}
    actual = entry.get("optional_dependencies", {})

    declared = set(actual) - _NON_FRAMEWORK_EXTRAS
    assert declared == backends, (
        f"{package_name}: config/packages.toml declares extras {sorted(declared)} "
        f"but the manifest registers backends {sorted(backends)}"
    )
    for backend in backends - _PACKAGED_BACKENDS - _CORE_MIRRORED_BACKENDS:
        assert actual[backend] == [], f"{package_name}: {backend} needs no pip package, its extra must be []"
    for backend in backends & _CORE_MIRRORED_BACKENDS:
        deps = actual[backend]
        assert deps, f"{package_name}: {backend} extra must not be empty"
        core_set = _CORE_MIRRORED_REQUIREMENTS[backend]
        assert {Requirement(dep) for dep in deps} == core_set, (
            f"{package_name}: {backend} extra must mirror mloda core's '{backend}' extra "
            f"{sorted(str(r) for r in core_set)}, got {deps!r}"
        )
    for backend in backends & _PACKAGED_BACKENDS:
        deps = actual[backend]
        assert deps, f"{package_name}: {backend} extra must not be empty"
        assert {_dep_name(dep) for dep in deps} == {backend}, (
            f"{package_name}: {backend} extra must depend on the '{backend}' package, got {deps!r}"
        )
        if backend == "polars":
            override = _POLARS_FLOOR_OVERRIDES.get(package_name)
            expected = SpecifierSet(override) if override is not None else _CORE_POLARS_REQUIREMENT.specifier
            if override is not None:
                assert _min_version(expected) >= _min_version(_CORE_POLARS_REQUIREMENT.specifier), (
                    f"{package_name}: polars floor override {expected} must not be below mloda core's floor "
                    f"({_CORE_POLARS_REQUIREMENT.specifier})"
                )
            for dep in deps:
                specifier = Requirement(dep).specifier
                source = f"override for {package_name}" if override else "mloda core's polars extra floor"
                assert specifier == expected, (
                    f"{package_name}: polars extra must require {expected} ({source}), got {specifier}"
                )
        if backend == "pyarrow":
            expected = _CORE_PYARROW_REQUIREMENT.specifier
            for dep in deps:
                specifier = Requirement(dep).specifier
                assert specifier == expected, (
                    f"{package_name}: pyarrow extra must require {expected} (mloda core's pyarrow extra floor), "
                    f"got {specifier}"
                )

    expected_all = {dep for backend in backends for dep in actual[backend]}
    assert set(actual.get("all", [])) == expected_all, f"{package_name}: 'all' extra must match its backends' deps"
