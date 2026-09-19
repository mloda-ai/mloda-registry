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

import pytest

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


def _licensed_plugin_classes_by_package(
    packages: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any], type[BinaryModelMixin]]]:
    """(package name, its config, BinaryModelMixin subclass) for every enterprise manifest entry
    mixing in BinaryModelMixin, so a packages.toml check can be scoped to the owning package."""
    result: list[tuple[str, dict[str, Any], type[BinaryModelMixin]]] = []
    for name, cfg in _enterprise_plugin_packages(packages):
        dotted = cfg["path"].replace("/", ".")
        module = importlib.import_module(f"{dotted}.manifest")
        for group in cfg.get("entry_point_groups", []):
            attr_name = _GROUP_ATTR.get(group)
            if attr_name is None:
                continue
            for cls in getattr(module, attr_name, []):
                if issubclass(cls, BinaryModelMixin):
                    result.append((name, cfg, cls))
    return result


def _licensed_plugin_wheel_distribution_names(packages: dict[str, dict[str, Any]]) -> set[str]:
    """PyPI distribution names (``BINARY_WHEEL_DISTRIBUTION``) for every binary wheel a
    BinaryModelMixin subclass needs, across every enterprise manifest."""
    return {cls.BINARY_WHEEL_DISTRIBUTION for _name, _cfg, cls in _licensed_plugin_classes_by_package(packages)}


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


def test_binary_wheel_distribution_is_an_optional_dependency_with_a_version_specifier() -> None:
    """Every enterprise package whose FeatureGroup(s) mix in BinaryModelMixin must declare
    BINARY_WHEEL_DISTRIBUTION as a version-pinned optional dependency, never a hard one, never
    exposed only through the ``dev`` extra (tox's ``[testenv] extras = dev`` and ``uv sync
    --all-extras`` both install it by default -- see ``_assert_wheel_optional_dependency_is_safe``
    below), and with a real version specifier living before any environment marker (``;``)."""
    packages: dict[str, dict[str, Any]] = _load_toml(_PACKAGES_CONFIG).get("packages", {})
    entries = _licensed_plugin_classes_by_package(packages)
    assert entries, "expected at least one BinaryModelMixin-derived enterprise plugin; check is vacuous"

    for name, cfg, cls in entries:
        distribution = cls.BINARY_WHEEL_DISTRIBUTION
        hard_dep_names = {_dep_name(dep) for dep in cfg.get("dependencies", [])}
        assert distribution not in hard_dep_names, f"{name}: {distribution} must not be a hard dependency"

        _assert_wheel_optional_dependency_is_safe(name, distribution, cfg.get("optional_dependencies", {}))


def _assert_wheel_optional_dependency_is_safe(
    name: str, distribution: str, optional_dependencies: dict[str, list[str]]
) -> None:
    """Assert `distribution` is declared as an optional dependency: at least one extra names it,
    ``dev`` is never one of the declaring extras (tox's ``[testenv] extras = dev`` and ``uv sync
    --all-extras`` both install it by default, so a wheel declared there is never truly optional),
    and every declaring requirement string carries a real version specifier in the portion before
    any environment marker (``;``)."""
    declaring_extras = [
        extra for extra, deps in optional_dependencies.items() if any(_dep_name(dep) == distribution for dep in deps)
    ]
    assert declaring_extras, (
        f"{name}: expected an optional_dependencies extra declaring {distribution!r}, got {optional_dependencies!r}"
    )
    assert "dev" not in declaring_extras, (
        f"{name}: {distribution} must not be declared under the 'dev' extra, got {declaring_extras!r}"
    )
    matches = [dep for deps in optional_dependencies.values() for dep in deps if _dep_name(dep) == distribution]
    for dep in matches:
        pre_marker = dep.split(";", 1)[0]
        assert re.search(r"[<>=!~]", pre_marker), (
            f"{name}: {distribution} optional dependency entries must carry a version specifier before any "
            f"environment marker, got {dep!r}"
        )


_MARKER_ONLY_DEPENDENCY = "mloda-example-binary; python_version >= '3.10'"


class TestWheelOptionalDependencyIsSafe:
    """Exercises `_assert_wheel_optional_dependency_is_safe` against synthetic data, so a
    regression is caught even when the real config/packages.toml is already compliant."""

    def test_accepts_a_version_pinned_dependency_under_a_non_dev_extra(self) -> None:
        optional_dependencies = {"wheel": ["mloda-example-binary>=0.1.0,<0.2.0"]}
        _assert_wheel_optional_dependency_is_safe("pkg", "mloda-example-binary", optional_dependencies)

    def test_rejects_a_dependency_declared_only_under_dev(self) -> None:
        """tox's `[testenv] extras = dev` and `uv sync --all-extras` both install the dev extra by
        default, so a wheel declared only there is never truly optional."""
        optional_dependencies = {"dev": ["mloda-example-binary>=0.1.0,<0.2.0"]}
        with pytest.raises(AssertionError, match="dev"):
            _assert_wheel_optional_dependency_is_safe("pkg", "mloda-example-binary", optional_dependencies)

    def test_rejects_a_marker_only_dependency_with_no_real_version_specifier(self) -> None:
        """The version specifier must live before any environment marker (`;`);
        `_MARKER_ONLY_DEPENDENCY` carries only a marker, no real pin."""
        optional_dependencies = {"wheel": [_MARKER_ONLY_DEPENDENCY]}
        with pytest.raises(AssertionError, match="version specifier"):
            _assert_wheel_optional_dependency_is_safe("pkg", "mloda-example-binary", optional_dependencies)

    def test_rejects_when_the_distribution_is_declared_nowhere(self) -> None:
        with pytest.raises(AssertionError, match="expected an optional_dependencies extra"):
            _assert_wheel_optional_dependency_is_safe("pkg", "mloda-example-binary", {})


def test_new_binary_packages_are_registered_with_dev_extra() -> None:
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
