"""Backend extras declared by data-operation leaf distributions must match what their manifest registers."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"
_DATA_OPERATIONS_REL = "mloda/community/feature_groups/data_operations"

# Backend module prefixes used by data_operations manifests, longest first so
# "polars_lazy_"/"python_dict_" aren't swallowed by a shorter unrelated prefix.
_BACKEND_PREFIXES = [
    ("polars_lazy_", "polars"),
    ("python_dict_", "python_dict"),
    ("pyarrow_", "pyarrow"),
    ("duckdb_", "duckdb"),
    ("pandas_", "pandas"),
    ("sqlite_", "sqlite"),
]

# Backends needing their own pip package; python_dict and sqlite are always available.
_PACKAGED_BACKENDS = {"pyarrow", "duckdb", "polars", "pandas"}


def _packages() -> dict[str, dict[str, Any]]:
    with open(_PACKAGES_CONFIG, "rb") as f:
        data = tomllib.load(f)
    packages: dict[str, dict[str, Any]] = data["packages"]
    return packages


def _leaf_packages() -> list[tuple[str, dict[str, Any]]]:
    """data_operations packages with their own manifest.py (excludes the shared base package)."""
    leaves = []
    for name, entry in _packages().items():
        path = entry.get("path", "")
        if path != _DATA_OPERATIONS_REL and not path.startswith(_DATA_OPERATIONS_REL + "/"):
            continue
        if (_REPO_ROOT / path / "manifest.py").is_file():
            leaves.append((name, entry))
    return leaves


def _backend_for_module(module_name: str) -> str:
    for prefix, backend in _BACKEND_PREFIXES:
        if module_name.startswith(prefix):
            return backend
    raise AssertionError(f"unrecognized backend module name {module_name!r}")


def _registered_backends(manifest_path: Path) -> set[str]:
    """Backend extras keys a manifest.py registers, read from its load_plugin_classes specs."""
    tree = ast.parse(manifest_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_plugin_classes":
            specs = node.args[1]
            assert isinstance(specs, ast.List), f"{manifest_path}: load_plugin_classes specs must be a list literal"
            modules = []
            for elt in specs.elts:
                assert isinstance(elt, ast.Tuple), f"{manifest_path}: each spec must be a (module, class) tuple"
                submodule = elt.elts[0]
                assert isinstance(submodule, ast.Constant) and isinstance(submodule.value, str)
                modules.append(submodule.value)
            return {_backend_for_module(m) for m in modules}
    raise AssertionError(f"{manifest_path}: no load_plugin_classes(...) call found")


_LEAF_PACKAGES = _leaf_packages()


@pytest.mark.parametrize("package_name,entry", _LEAF_PACKAGES, ids=[name for name, _ in _LEAF_PACKAGES])
def test_leaf_package_extras_match_registered_backends(package_name: str, entry: dict[str, Any]) -> None:
    backends = _registered_backends(_REPO_ROOT / entry["path"] / "manifest.py")
    actual = entry.get("optional_dependencies", {})

    declared = set(actual) - {"all"}
    assert declared == backends, (
        f"{package_name}: config/packages.toml declares extras {sorted(declared)} "
        f"but the manifest registers backends {sorted(backends)}"
    )
    for backend in backends - _PACKAGED_BACKENDS:
        assert actual[backend] == [], f"{package_name}: {backend} needs no pip package, its extra must be []"
    for backend in backends & _PACKAGED_BACKENDS:
        assert actual[backend], f"{package_name}: {backend} extra must not be empty"

    expected_all = {dep for backend in backends & _PACKAGED_BACKENDS for dep in actual[backend]}
    assert set(actual.get("all", [])) == expected_all, f"{package_name}: 'all' extra must match its packaged backends"
