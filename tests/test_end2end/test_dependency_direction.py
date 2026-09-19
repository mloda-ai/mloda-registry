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
from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"

gen = load_script("generate_pyproject", _REPO_ROOT / "scripts" / "generate_pyproject.py")

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
    """PEP 503 normalized bare package name of a PEP 508 requirement string. Falls back to the
    stripped spec itself for a ``{...}`` template placeholder, which never normalizes to a name
    and so never equals a real distribution name."""
    return gen.normalize_dependency_name(spec) or spec.strip()


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


def _assert_binary_wheel_distribution(name: str, cls: type[BinaryModelMixin]) -> str:
    assert hasattr(cls, "BINARY_WHEEL_DISTRIBUTION"), f"{name}: {cls.__name__} must declare BINARY_WHEEL_DISTRIBUTION"
    return cls.BINARY_WHEEL_DISTRIBUTION


def _licensed_plugin_wheel_distribution_names(packages: dict[str, dict[str, Any]]) -> set[str]:
    """PyPI distribution names (``BINARY_WHEEL_DISTRIBUTION``) for every binary wheel a
    BinaryModelMixin subclass needs, across every enterprise manifest."""
    return {
        _assert_binary_wheel_distribution(name, cls)
        for name, _cfg, cls in _licensed_plugin_classes_by_package(packages)
    }


def _community_or_enterprise_wheel_or_testing_dependency_violations(
    packages: dict[str, dict[str, Any]], wheel_names: set[str]
) -> list[str]:
    normalized_wheel_names = {_dep_name(wheel_name) for wheel_name in wheel_names}
    violations: list[str] = []
    for name, cfg in packages.items():
        if not _is_community_or_enterprise_path(cfg.get("path", "")):
            continue
        for dep in cfg.get("dependencies", []):
            dep_name = _dep_name(dep)
            if dep.startswith("mloda-testing") or dep_name in normalized_wheel_names:
                violations.append(f"{name}: {dep}")
    return violations


def test_no_community_or_enterprise_package_depends_on_mloda_testing_or_a_binary_wheel() -> None:
    packages: dict[str, dict[str, Any]] = _load_toml(_PACKAGES_CONFIG).get("packages", {})
    wheel_names = _licensed_plugin_wheel_distribution_names(packages)
    assert wheel_names, "expected at least one BINARY_WHEEL_DISTRIBUTION-derived wheel name; check is vacuous"

    violations = _community_or_enterprise_wheel_or_testing_dependency_violations(packages, wheel_names)
    assert not violations, (
        f"community/enterprise packages must not depend on mloda-testing or a binary wheel: {violations}"
    )


def test_hard_dependency_violation_detected_for_an_underscore_spelled_distribution() -> None:
    """PEP 503 normalisation must fold an underscore-spelled hard dependency onto its wheel distribution."""
    packages = {
        "mloda-sandbox-enterprise-plugin": {
            "path": "mloda/enterprise/feature_groups/sandbox",
            "dependencies": ["mloda_example_binary>=0.1.0"],
        },
    }
    violations = _community_or_enterprise_wheel_or_testing_dependency_violations(packages, {"mloda-example-binary"})
    assert violations == ["mloda-sandbox-enterprise-plugin: mloda_example_binary>=0.1.0"], violations


def test_binary_wheel_distribution_is_an_optional_dependency_with_a_version_specifier() -> None:
    """Every enterprise package whose FeatureGroup(s) mix in BinaryModelMixin must declare
    BINARY_WHEEL_DISTRIBUTION as a version-pinned optional dependency, never a hard one, never
    exposed only through the ``dev`` extra (dev is what developer and CI flows install wholesale,
    so a wheel declared there is not meaningfully optional -- see
    ``_assert_wheel_optional_dependency_is_safe`` below), and with a real version specifier living
    before any environment marker (``;``)."""
    packages: dict[str, dict[str, Any]] = _load_toml(_PACKAGES_CONFIG).get("packages", {})
    entries = _licensed_plugin_classes_by_package(packages)
    assert entries, "expected at least one BinaryModelMixin-derived enterprise plugin; check is vacuous"

    for name, cfg, cls in entries:
        distribution = _assert_binary_wheel_distribution(name, cls)
        hard_dep_names = {_dep_name(dep) for dep in cfg.get("dependencies", [])}
        assert _dep_name(distribution) not in hard_dep_names, f"{name}: {distribution} must not be a hard dependency"

        _assert_wheel_optional_dependency_is_safe(name, distribution, cfg.get("optional_dependencies", {}))


class _MissingWheelDistributionModel(BinaryModelMixin):
    """Subclass omitting BINARY_WHEEL_DISTRIBUTION, the required ClassVar with no default."""


def test_licensed_plugin_wheel_distribution_names_reports_missing_attribute_actionably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A BinaryModelMixin subclass omitting BINARY_WHEEL_DISTRIBUTION must fail with an
    AssertionError naming the class and the attribute, not a bare AttributeError."""
    fake_entries: list[tuple[str, dict[str, Any], type[BinaryModelMixin]]] = [
        ("mloda-sandbox-enterprise-plugin", {}, _MissingWheelDistributionModel)
    ]
    monkeypatch.setattr(sys.modules[__name__], "_licensed_plugin_classes_by_package", lambda packages: fake_entries)

    with pytest.raises(AssertionError, match="_MissingWheelDistributionModel") as exc_info:
        _licensed_plugin_wheel_distribution_names({})
    assert "BINARY_WHEEL_DISTRIBUTION" in str(exc_info.value)


def test_binary_wheel_distribution_test_reports_missing_attribute_actionably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The BINARY_WHEEL_DISTRIBUTION read inside the version-specifier test must also fail with
    an actionable AssertionError, not a bare AttributeError."""
    fake_entries: list[tuple[str, dict[str, Any], type[BinaryModelMixin]]] = [
        ("mloda-sandbox-enterprise-plugin", {}, _MissingWheelDistributionModel)
    ]
    monkeypatch.setattr(sys.modules[__name__], "_licensed_plugin_classes_by_package", lambda packages: fake_entries)

    with pytest.raises(AssertionError, match="_MissingWheelDistributionModel") as exc_info:
        test_binary_wheel_distribution_is_an_optional_dependency_with_a_version_specifier()
    assert "BINARY_WHEEL_DISTRIBUTION" in str(exc_info.value)


_VERSION_OPERATOR_RE = re.compile(r"===|~=|==|<=|>=|!=|<|>")
_UPPER_BOUND_OPERATORS = frozenset({"<", "<=", "==", "===", "~="})


def _assert_wheel_optional_dependency_is_safe(
    name: str, distribution: str, optional_dependencies: dict[str, list[str]]
) -> None:
    """Assert `distribution` is declared as an optional dependency: at least one extra names it,
    ``dev`` is never one of the declaring extras (dev is what developer and CI flows install
    wholesale, so a wheel declared there is not meaningfully optional), and every declaring
    requirement string carries a version specifier with an upper bound before any environment
    marker (``;``)."""
    distribution_name = _dep_name(distribution)
    declaring_extras = [
        extra
        for extra, deps in optional_dependencies.items()
        if any(_dep_name(dep) == distribution_name for dep in deps)
    ]
    assert declaring_extras, (
        f"{name}: expected an optional_dependencies extra declaring {distribution!r}, got {optional_dependencies!r}"
    )
    assert "dev" not in declaring_extras, (
        f"{name}: {distribution} must not be declared under the 'dev' extra, got {declaring_extras!r}"
    )
    matches = [dep for deps in optional_dependencies.values() for dep in deps if _dep_name(dep) == distribution_name]
    for dep in matches:
        pre_marker = dep.split(";", 1)[0]
        operators = [m.group() for clause in pre_marker.split(",") if (m := _VERSION_OPERATOR_RE.search(clause))]
        assert operators, (
            f"{name}: {distribution} optional dependency entries must carry a version specifier before any "
            f"environment marker, got {dep!r}"
        )
        assert any(op in _UPPER_BOUND_OPERATORS for op in operators), (
            f"{name}: {distribution} optional dependency entries must carry a version specifier with an upper "
            f"bound before any environment marker, got {dep!r}"
        )


_MARKER_ONLY_DEPENDENCY = "mloda-example-binary; python_version >= '3.10'"


class TestWheelOptionalDependencyIsSafe:
    """Exercises `_assert_wheel_optional_dependency_is_safe` against synthetic data, so a
    regression is caught even when the real config/packages.toml is already compliant."""

    def test_accepts_a_version_pinned_dependency_under_a_non_dev_extra(self) -> None:
        optional_dependencies = {"wheel": ["mloda-example-binary>=0.1.0,<0.2.0"]}
        _assert_wheel_optional_dependency_is_safe("pkg", "mloda-example-binary", optional_dependencies)

    def test_rejects_a_dependency_declared_only_under_dev(self) -> None:
        """dev is what developer and CI flows install wholesale, so a wheel declared only there is
        never truly optional."""
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

    def test_rejects_a_lower_bound_only_dependency(self) -> None:
        """A lower bound alone is not a version range; an upper bound is required."""
        optional_dependencies = {"wheel": ["mloda-example-binary>=0.1.0"]}
        with pytest.raises(AssertionError, match="upper bound"):
            _assert_wheel_optional_dependency_is_safe("pkg", "mloda-example-binary", optional_dependencies)

    def test_accepts_an_exact_version_pin(self) -> None:
        optional_dependencies = {"wheel": ["mloda-example-binary==0.1.0"]}
        _assert_wheel_optional_dependency_is_safe("pkg", "mloda-example-binary", optional_dependencies)

    def test_accepts_a_compatible_release_pin(self) -> None:
        optional_dependencies = {"wheel": ["mloda-example-binary~=0.1.0"]}
        _assert_wheel_optional_dependency_is_safe("pkg", "mloda-example-binary", optional_dependencies)

    def test_accepts_an_arbitrary_equality_pin(self) -> None:
        optional_dependencies = {"wheel": ["mloda-example-binary===0.1.0"]}
        _assert_wheel_optional_dependency_is_safe("pkg", "mloda-example-binary", optional_dependencies)

    def test_accepts_an_underscore_spelling_as_the_same_distribution(self) -> None:
        """PEP 503 folds `_`, `-` and `.` together; the underscore spelling is the same distribution."""
        optional_dependencies = {"wheel": ["mloda_example_binary>=0.1.0,<0.2.0"]}
        _assert_wheel_optional_dependency_is_safe("pkg", "mloda-example-binary", optional_dependencies)


def _assert_no_published_package_exposes_an_index_pinned_distribution(
    packages: dict[str, dict[str, Any]],
) -> None:
    """A built wheel's metadata carries only the bare requirement string, so a `published` package
    must not declare, in any extra, a distribution some package pins to a non-default index via
    `optional_dependency_indexes`."""
    index_pinned_names = {
        _dep_name(dep_name) for cfg in packages.values() for dep_name in cfg.get("optional_dependency_indexes", {})
    }
    for name, cfg in packages.items():
        if not cfg.get("published"):
            continue
        for extra, deps in cfg.get("optional_dependencies", {}).items():
            for dep in deps:
                assert _dep_name(dep) not in index_pinned_names, (
                    f"{name}: published package's {extra!r} extra declares {dep!r}, which is pinned to a "
                    "non-default index by another package's optional_dependency_indexes"
                )


class TestNoPublishedPackageExposesAnIndexPinnedDistribution:
    """Exercises `_assert_no_published_package_exposes_an_index_pinned_distribution` against
    synthetic data, so a regression is caught even when the real config is already compliant."""

    def test_rejects_a_published_package_exposing_an_index_pinned_distribution(self) -> None:
        packages: dict[str, dict[str, Any]] = {
            "mloda-sandbox-index-owner": {
                "path": "mloda/enterprise/feature_groups/sandbox_owner",
                "dependencies": ["{core_dependency}"],
                "optional_dependency_indexes": {"mloda-example-binary": "testpypi"},
            },
            "mloda-sandbox-published-bundle": {
                "path": "mloda/enterprise",
                "published": True,
                "optional_dependencies": {"wheel": ["mloda-example-binary>=0.1.0,<0.2.0"]},
            },
        }
        with pytest.raises(AssertionError, match="mloda-example-binary"):
            _assert_no_published_package_exposes_an_index_pinned_distribution(packages)

    def test_accepts_a_published_package_with_a_non_index_pinned_dependency(self) -> None:
        packages: dict[str, dict[str, Any]] = {
            "mloda-sandbox-index-owner": {
                "path": "mloda/enterprise/feature_groups/sandbox_owner",
                "dependencies": ["{core_dependency}"],
                "optional_dependency_indexes": {"mloda-example-binary": "testpypi"},
            },
            "mloda-sandbox-published-bundle": {
                "path": "mloda/enterprise",
                "published": True,
                "optional_dependencies": {"all": ["pyarrow>=25"]},
            },
        }
        _assert_no_published_package_exposes_an_index_pinned_distribution(packages)

    def test_accepts_a_non_published_package_exposing_the_index_pinned_distribution(self) -> None:
        """Mirrors mloda-enterprise-binary-example: not published, owns the index pin, and
        declares the distribution in its own wheel extra."""
        packages = {
            "mloda-sandbox-binary-example": {
                "path": "mloda/enterprise/feature_groups/binary_example",
                "dependencies": ["{core_dependency}"],
                "optional_dependencies": {"wheel": ["mloda-example-binary>=0.1.0,<0.2.0"]},
                "optional_dependency_indexes": {"mloda-example-binary": "testpypi"},
            },
        }
        _assert_no_published_package_exposes_an_index_pinned_distribution(packages)

    def test_real_config_has_no_published_package_exposing_an_index_pinned_distribution(self) -> None:
        packages = _load_toml(_PACKAGES_CONFIG).get("packages", {})
        _assert_no_published_package_exposes_an_index_pinned_distribution(packages)


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
