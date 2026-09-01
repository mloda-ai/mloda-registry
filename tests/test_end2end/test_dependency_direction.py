"""Dependency-direction guard: community and enterprise plugin packages must never depend on
``mloda-testing`` or on a binary wheel (see ``docs/guides/feature-group-patterns/28-binary-backed-
features.md``), and no runtime module under ``mloda/community/`` or ``mloda/enterprise/`` may import
``mloda.testing`` at any depth. Mirrors the resolution and TOML-loading style of
``tests/test_end2end/test_dev_dependencies.py`` and ``tests/test_end2end/test_manifest_resilience.py``.
"""

from __future__ import annotations

import ast
import importlib
import re
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

from mloda.community.feature_groups.binary_model.mixin import BinaryModelMixin

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"

_GROUP_ATTR: dict[str, str] = {
    "mloda.feature_groups": "FEATURE_GROUPS",
    "mloda.compute_frameworks": "COMPUTE_FRAMEWORKS",
    "mloda.extenders": "EXTENDERS",
}

_SKIP_DIR_NAMES = frozenset({"__pycache__", "build", "dist"})


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _dep_name(spec: str) -> str:
    """Extract the bare package name from a PEP 508 requirement string."""
    return re.split(r"[<>=!~;\s\[(@]", spec.strip(), maxsplit=1)[0]


def _is_community_or_enterprise_path(path: str) -> bool:
    return (
        path == "mloda/community"
        or path.startswith("mloda/community/")
        or path == "mloda/enterprise"
        or path.startswith("mloda/enterprise/")
    )


def _enterprise_plugin_packages(packages: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (name, cfg)
        for name, cfg in packages.items()
        if cfg.get("path", "").startswith("mloda/enterprise/") and cfg.get("entry_point_groups")
    ]


def _licensed_plugin_wheel_distribution_names(packages: dict[str, dict[str, Any]]) -> set[str]:
    """PyPI distribution names for every binary wheel a BinaryModelMixin subclass needs, derived
    from BINARY_PLUGIN_ID (normalizing '_' to '-') across every enterprise manifest."""
    names: set[str] = set()
    for _name, cfg in _enterprise_plugin_packages(packages):
        dotted = cfg["path"].replace("/", ".")
        module = importlib.import_module(f"{dotted}.manifest")
        for group in cfg.get("entry_point_groups", []):
            attr_name = _GROUP_ATTR.get(group)
            if attr_name is None:
                continue
            for cls in getattr(module, attr_name, []):
                if issubclass(cls, BinaryModelMixin):
                    names.add(cls.BINARY_PLUGIN_ID.replace("_", "-"))
    return names


def test_no_community_or_enterprise_package_depends_on_mloda_testing_or_a_binary_wheel() -> None:
    packages: dict[str, dict[str, Any]] = _load_toml(_PACKAGES_CONFIG).get("packages", {})
    wheel_names = _licensed_plugin_wheel_distribution_names(packages)
    assert wheel_names, "expected at least one BINARY_PLUGIN_ID-derived wheel name; check is vacuous"

    violations: list[str] = []
    for name, cfg in packages.items():
        if not _is_community_or_enterprise_path(cfg.get("path", "")):
            continue
        for dep in cfg.get("dependencies", []):
            dep_name = _dep_name(dep)
            if dep.startswith("mloda-testing") or dep_name in wheel_names:
                violations.append(f"{name}: {dep}")
    assert not violations, (
        f"community/enterprise packages must not depend on mloda-testing or a binary wheel: {violations}"
    )


def test_new_binary_packages_are_registered_with_dev_extra() -> None:
    """Fails now: config/packages.toml does not yet declare these two new packages."""
    packages: dict[str, dict[str, Any]] = _load_toml(_PACKAGES_CONFIG).get("packages", {})
    expected_paths = {
        "mloda-community-binary-model": "mloda/community/feature_groups/binary_model",
        "mloda-enterprise-binary-example": "mloda/enterprise/feature_groups/binary_example",
    }
    for pkg_name, expected_path in expected_paths.items():
        assert pkg_name in packages, f"{pkg_name} missing from config/packages.toml"
        cfg = packages[pkg_name]
        assert cfg.get("path") == expected_path, f"{pkg_name}: expected path {expected_path!r}, got {cfg.get('path')!r}"
        dev_deps = cfg.get("optional_dependencies", {}).get("dev", [])
        assert "mloda-testing[binary-model]" in dev_deps, (
            f"{pkg_name}: optional_dependencies.dev must contain 'mloda-testing[binary-model]', got {dev_deps!r}"
        )


def _is_mloda_testing_import(node: ast.AST) -> bool:
    """True for an Import whose dotted name starts with 'mloda.testing', or an ImportFrom whose
    module starts with 'mloda.testing'. No exemption for function scope or TYPE_CHECKING blocks:
    the caller walks the whole tree with ast.walk, not a scope-limited traversal."""
    if isinstance(node, ast.Import):
        return any(alias.name == "mloda.testing" or alias.name.startswith("mloda.testing.") for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        if node.module is None:
            return False
        return node.module == "mloda.testing" or node.module.startswith("mloda.testing.")
    return False


def _files_importing_mloda_testing(root_dir: Path) -> list[Path]:
    """Every .py file under root_dir, outside any tests/ directory, that imports mloda.testing at
    any depth (module level, function level, or inside a TYPE_CHECKING block)."""
    offenders: list[Path] = []
    for py_file in sorted(root_dir.rglob("*.py")):
        rel_path = py_file.relative_to(root_dir)
        parts = rel_path.parts
        if "tests" in parts or any(part in _SKIP_DIR_NAMES or part.endswith(".egg-info") for part in parts):
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        if any(_is_mloda_testing_import(node) for node in ast.walk(tree)):
            offenders.append(rel_path)
    return offenders


def test_no_community_or_enterprise_runtime_file_imports_mloda_testing() -> None:
    offenders: list[str] = []
    for root_name in ("community", "enterprise"):
        root_dir = _REPO_ROOT / "mloda" / root_name
        assert root_dir.is_dir(), f"expected {root_dir} to exist"
        offenders.extend(f"mloda/{root_name}/{path.as_posix()}" for path in _files_importing_mloda_testing(root_dir))
    assert not offenders, f"mloda.testing imported outside tests/: {offenders}"


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_walker_catches_function_level_import(tmp_path: Path) -> None:
    _write(tmp_path / "m.py", "def f() -> None:\n    from mloda.testing.base import FeatureGroupTestBase\n")
    assert _files_importing_mloda_testing(tmp_path) == [Path("m.py")]


def test_walker_catches_type_checking_import(tmp_path: Path) -> None:
    """No exemption for TYPE_CHECKING blocks, unlike test_manifest_resilience.py's own walker."""
    body = (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from mloda.testing.base import FeatureGroupTestBase\n"
    )
    _write(tmp_path / "m.py", body)
    assert _files_importing_mloda_testing(tmp_path) == [Path("m.py")]


def test_walker_ignores_import_inside_a_tests_directory(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write(tests_dir / "test_m.py", "from mloda.testing.base import FeatureGroupTestBase\n")
    assert _files_importing_mloda_testing(tmp_path) == []
