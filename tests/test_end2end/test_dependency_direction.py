"""Dependency-direction guard: community and enterprise plugin packages must never depend on
``mloda-testing`` or on a binary wheel (see ``docs/guides/feature-group-patterns/29-binary-backed-
features.md``), and no runtime module under ``mloda/community/`` or ``mloda/enterprise/`` may import
``mloda.testing`` at any depth. Enterprise code may use ``mloda.community.extenders.shared`` only
through public names (no private imports, subclassing, or private attributes). Binary wheel floors must
also exclude releases that trust no production key. The ``mloda-testing`` kit must declare every registry
package and third-party distribution it imports. Mirrors the resolution
and TOML-loading style of ``tests/test_end2end/test_dev_dependencies.py`` and
``tests/test_end2end/test_manifest_resilience.py``.
"""

from __future__ import annotations

import ast
import importlib
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from mloda.community.feature_groups.binary_model.mixin import BinaryModelMixin
from tests.import_roots import (
    FIRST_PARTY_IMPORT_ROOTS,
    STDLIB_ROOTS,
    is_type_checking_block,
    trusted_type_checking_alias,
)
from tests.script_loader import load_script
from tests.toml_loader import load_toml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"

gen = load_script("generate_pyproject", _REPO_ROOT / "scripts" / "generate_pyproject.py")

_GROUP_ATTR: dict[str, str] = {
    "mloda.feature_groups": "FEATURE_GROUPS",
    "mloda.compute_frameworks": "COMPUTE_FRAMEWORKS",
    "mloda.extenders": "EXTENDERS",
}

_SKIP_DIR_NAMES = frozenset({"__pycache__", "build", "dist"})


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
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
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
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
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


_INSTALL_EXTRA_RE = re.compile(r"^(?P<package>[^\[\]]+)\[(?P<extra>[^\[\]]+)\]$")


def _assert_install_extra_brings_the_wheel(
    name: str, cls: type[BinaryModelMixin], packages: dict[str, dict[str, Any]]
) -> None:
    """`BINARY_INSTALL_EXTRA` (when set) must be `<package>[<extra>]`, naming a configured package whose
    extra requires the class's `BINARY_WHEEL_DISTRIBUTION`."""
    install_extra = cls.BINARY_INSTALL_EXTRA
    if install_extra is None:
        return
    match = _INSTALL_EXTRA_RE.match(install_extra)
    assert match, f"{name}: {cls.__name__}.BINARY_INSTALL_EXTRA {install_extra!r} must look like '<package>[<extra>]'"
    package, extra = match.group("package"), match.group("extra")
    assert package in packages, f"{name}: BINARY_INSTALL_EXTRA names unknown package {package!r}"
    deps = packages[package].get("optional_dependencies", {}).get(extra)
    assert deps is not None, f"{name}: package {package!r} has no extra {extra!r}"
    wheel = _dep_name(_assert_binary_wheel_distribution(name, cls))
    assert any(_dep_name(dep) == wheel for dep in deps), (
        f"{name}: extra {install_extra!r} does not require {cls.BINARY_WHEEL_DISTRIBUTION!r}, got {deps!r}"
    )


def test_binary_install_extra_names_an_extra_that_brings_the_wheel() -> None:
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
    entries = _licensed_plugin_classes_by_package(packages)
    assert any(cls.BINARY_INSTALL_EXTRA is not None for _n, _c, cls in entries), "no class sets BINARY_INSTALL_EXTRA"
    for name, _cfg, cls in entries:
        _assert_install_extra_brings_the_wheel(name, cls, packages)


class _InstallExtraModel(BinaryModelMixin):
    BINARY_WHEEL_DISTRIBUTION = "mloda-example-binary"
    BINARY_INSTALL_EXTRA = "pkg[wheel]"


class TestInstallExtraGuard:
    """Exercises `_assert_install_extra_brings_the_wheel` against synthetic config."""

    def test_accepts_an_extra_requiring_the_wheel(self) -> None:
        packages = {"pkg": {"optional_dependencies": {"wheel": ["mloda_example_binary>=0.1,<0.2"]}}}
        _assert_install_extra_brings_the_wheel("pkg", _InstallExtraModel, packages)

    def test_rejects_an_extra_requiring_a_different_wheel(self) -> None:
        packages = {"pkg": {"optional_dependencies": {"wheel": ["other-binary>=0.1,<0.2"]}}}
        with pytest.raises(AssertionError, match="does not require"):
            _assert_install_extra_brings_the_wheel("pkg", _InstallExtraModel, packages)

    def test_rejects_an_unknown_package(self) -> None:
        with pytest.raises(AssertionError, match="unknown package"):
            _assert_install_extra_brings_the_wheel("pkg", _InstallExtraModel, {})

    def test_rejects_an_unknown_extra(self) -> None:
        with pytest.raises(AssertionError, match="no extra"):
            _assert_install_extra_brings_the_wheel("pkg", _InstallExtraModel, {"pkg": {"optional_dependencies": {}}})

    def test_rejects_a_malformed_value(self) -> None:
        class _Bad(_InstallExtraModel):
            BINARY_INSTALL_EXTRA = "pkg-without-extra"

        with pytest.raises(AssertionError, match="must look like"):
            _assert_install_extra_brings_the_wheel("pkg", _Bad, {})


# Releases that trust no production license key and reject every license, per normalized distribution.
# Source: the mloda-binary-wrapper releases that embed no production key.
_KEYLESS_WHEEL_RELEASES: dict[str, list[str]] = {
    "mloda-anonymizer-binary": ["0.1.0"],
    "mloda-example-binary": ["0.1.0", "0.1.1"],
}


def _keyless_wheel_release_violations(packages: dict[str, dict[str, Any]]) -> list[str]:
    """Requirements on a binary wheel whose specifier admits a release listed in `_KEYLESS_WHEEL_RELEASES`."""
    violations: list[str] = []
    for name, cfg in packages.items():
        requirements = {"dependencies": cfg.get("dependencies", [])}
        requirements.update(cfg.get("optional_dependencies", {}))
        for where, deps in requirements.items():
            for dep in deps:
                keyless = _KEYLESS_WHEEL_RELEASES.get(_dep_name(dep))
                if keyless is None:
                    continue
                specifier = Requirement(dep).specifier
                violations.extend(
                    f"{name}.{where}: {dep} admits {version}"
                    for version in keyless
                    if specifier.contains(Version(version), prereleases=True)
                )
    return violations


def test_binary_wheel_floors_exclude_releases_that_trust_no_key() -> None:
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
    declared = {_dep_name(wheel) for wheel in _licensed_plugin_wheel_distribution_names(packages)}
    assert set(_KEYLESS_WHEEL_RELEASES) == declared, (
        f"_KEYLESS_WHEEL_RELEASES must have one entry per binary wheel {sorted(declared)}"
    )
    violations = _keyless_wheel_release_violations(packages)
    assert violations == [], f"binary wheel floors admit releases that trust no production key: {violations}"


class TestKeylessWheelReleaseViolations:
    """Exercises `_keyless_wheel_release_violations` against synthetic config."""

    def test_reports_a_floor_admitting_a_keyless_release(self) -> None:
        packages = {"pkg": {"optional_dependencies": {"wheel": ["mloda-example-binary>=0.1.0,<0.2.0"]}}}
        assert _keyless_wheel_release_violations(packages) == [
            "pkg.wheel: mloda-example-binary>=0.1.0,<0.2.0 admits 0.1.0",
            "pkg.wheel: mloda-example-binary>=0.1.0,<0.2.0 admits 0.1.1",
        ]

    def test_accepts_a_floor_above_every_keyless_release(self) -> None:
        packages = {"pkg": {"optional_dependencies": {"wheel": ["mloda-example-binary>=0.1.2,<0.2.0"]}}}
        assert _keyless_wheel_release_violations(packages) == []

    @pytest.mark.parametrize(
        ("requirement", "admitted"),
        [
            ("mloda-example-binary>0.1.0,<0.2.0", ["0.1.1"]),
            ("mloda-example-binary==0.1.1", ["0.1.1"]),
            ("mloda-example-binary~=0.1.0", ["0.1.0", "0.1.1"]),
            ("mloda-example-binary", ["0.1.0", "0.1.1"]),
        ],
    )
    def test_reports_operator_edge_cases(self, requirement: str, admitted: list[str]) -> None:
        packages = {"pkg": {"optional_dependencies": {"wheel": [requirement]}}}
        assert _keyless_wheel_release_violations(packages) == [
            f"pkg.wheel: {requirement} admits {version}" for version in admitted
        ]

    def test_parses_a_requirement_with_an_environment_marker(self) -> None:
        packages = {"pkg": {"dependencies": ["mloda_anonymizer_binary>=0.1.0,<0.2.0; python_version >= '3.10'"]}}
        violations = _keyless_wheel_release_violations(packages)
        assert [v.rsplit(" ", 1)[1] for v in violations] == ["0.1.0"], violations


def test_new_binary_packages_are_registered_with_dev_extra() -> None:
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
    expected_paths = {
        "mloda-community-binary-model": "mloda/community/feature_groups/binary_model",
        "mloda-enterprise-binary-example": "mloda/enterprise/feature_groups/binary_example",
        "mloda-enterprise-anonymizer": "mloda/enterprise/feature_groups/anonymizer",
    }
    for pkg_name, expected_path in expected_paths.items():
        assert pkg_name in packages, f"{pkg_name} missing from config/packages.toml"
        cfg = packages[pkg_name]
        assert cfg.get("path") == expected_path, f"{pkg_name}: expected path {expected_path!r}, got {cfg.get('path')!r}"
        dev_deps = cfg.get("optional_dependencies", {}).get("dev", [])
        assert "mloda-testing[binary-model]" in dev_deps, (
            f"{pkg_name}: optional_dependencies.dev must contain 'mloda-testing[binary-model]', got {dev_deps!r}"
        )


def test_no_configured_package_names_the_community_bundle_outside_dev() -> None:
    """Consumers depend on the mixin package, never on the mloda-community bundle (exact name, not prefixes)."""
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
    violations: list[str] = []
    for pkg_name, cfg in packages.items():
        specs = {"dependencies": cfg.get("dependencies", [])}
        for extra, extra_deps in cfg.get("optional_dependencies", {}).items():
            if extra != "dev":
                specs[f"extra {extra}"] = extra_deps
        for where, deps in specs.items():
            violations.extend(f"{pkg_name} ({where}): {dep}" for dep in deps if _dep_name(dep) == "mloda-community")
    assert violations == [], f"no package may depend on the mloda-community bundle: {violations}"


def test_consumers_name_the_binary_model_package_instead_of_the_bundle() -> None:
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
    mixin = "mloda-community-binary-model"
    for pkg_name in ("mloda-enterprise", "mloda-enterprise-binary-example", "mloda-enterprise-anonymizer"):
        names = [_dep_name(dep) for dep in packages[pkg_name].get("dependencies", [])]
        assert mixin in names, f"{pkg_name}: dependencies must name {mixin}, got {names!r}"
    extra = [
        _dep_name(dep) for dep in packages["mloda-testing"].get("optional_dependencies", {}).get("binary-model", [])
    ]
    assert mixin in extra, f"mloda-testing: the binary-model extra must name {mixin}, got {extra!r}"


def test_mloda_testing_binary_model_extra_pins_the_enterprise_license_token_extra() -> None:
    """The raw spec is compared: ``_dep_name`` strips extras and would miss a dropped ``[ed25519]``."""
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
    extra = packages["mloda-testing"].get("optional_dependencies", {}).get("binary-model", [])
    spec = "mloda-enterprise[ed25519]>={version}"
    assert spec in extra, f"mloda-testing: the binary-model extra must contain {spec!r}, got {extra!r}"


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


def _runtime_py_files(root_dir: Path) -> list[Path]:
    """Sorted .py files under root_dir, relative to it, outside tests/, skipped dirs and egg-info."""
    files: list[Path] = []
    for py_file in sorted(root_dir.rglob("*.py")):
        rel_path = py_file.relative_to(root_dir)
        parts = rel_path.parts
        if "tests" in parts or any(part in _SKIP_DIR_NAMES or part.endswith(".egg-info") for part in parts):
            continue
        files.append(rel_path)
    return files


def _files_with_import(root_dir: Path, predicate: Callable[[ast.AST], bool]) -> list[Path]:
    """Every runtime .py file under root_dir with a node matching predicate at any depth
    (module level, function level, or inside a TYPE_CHECKING block)."""
    offenders: list[Path] = []
    for rel_path in _runtime_py_files(root_dir):
        tree = ast.parse((root_dir / rel_path).read_text(encoding="utf-8"))
        if any(predicate(node) for node in ast.walk(tree)):
            offenders.append(rel_path)
    return offenders


def _files_importing_mloda_testing(root_dir: Path) -> list[Path]:
    return _files_with_import(root_dir, _is_mloda_testing_import)


_EXTENDERS_SHARED = "mloda.community.extenders.shared"


def _is_private_name(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _is_under_extenders_shared(module: str) -> bool:
    return module == _EXTENDERS_SHARED or module.startswith(_EXTENDERS_SHARED + ".")


def _names_bound_from_extenders_shared(tree: ast.AST) -> set[str]:
    """Local names bound by imports from (or of) the extenders-shared package."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                if _is_under_extenders_shared(node.module) or (node.module + "." + alias.name == _EXTENDERS_SHARED):
                    bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            bound.update(
                alias.asname for alias in node.names if alias.asname and _is_under_extenders_shared(alias.name)
            )
    return bound


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _rides_on_shared_private_seams(tree: ast.AST) -> bool:
    """True if a class subclasses a shared name, or code reads a private attribute of a shared name."""
    bound = _names_bound_from_extenders_shared(tree)
    if not bound:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(_root_name(base) in bound for base in node.bases):
            return True
        if (
            isinstance(node, ast.Attribute)
            and _is_private_name(node.attr)
            and isinstance(node.value, ast.Name)
            and node.value.id in bound
        ):
            return True
    return False


def _uses_extenders_shared_privately(node: ast.AST) -> bool:
    """True for a private import from extenders-shared, or (on a Module) subclassing or private attribute access."""
    if isinstance(node, ast.Module):
        return _rides_on_shared_private_seams(node)
    if isinstance(node, ast.Import):
        return any(
            (alias.name == _EXTENDERS_SHARED or alias.name.startswith(_EXTENDERS_SHARED + "."))
            and any(_is_private_name(segment) for segment in alias.name.split("."))
            for alias in node.names
        )
    if isinstance(node, ast.ImportFrom) and node.module is not None:
        if node.module != _EXTENDERS_SHARED and not node.module.startswith(_EXTENDERS_SHARED + "."):
            return False
        return any(_is_private_name(segment) for segment in node.module.split(".")) or any(
            _is_private_name(alias.name) for alias in node.names
        )
    return False


def test_enterprise_runtime_code_uses_extenders_shared_only_through_public_names() -> None:
    root_dir = _REPO_ROOT / "mloda" / "enterprise"
    assert root_dir.is_dir(), f"expected {root_dir} to exist"
    offenders = [
        f"mloda/enterprise/{p.as_posix()}" for p in _files_with_import(root_dir, _uses_extenders_shared_privately)
    ]
    assert not offenders, (
        "enterprise runtime code relies on private mloda-community-extenders-shared seams (private import, "
        "subclassing a shared class, or a private attribute of a shared name). Either make the name public in "
        "extenders-shared, or pin mloda-community-extenders-shared~={version} and accept that this caps the "
        "mloda-community bundle at enterprise's minor (see docs/packaging.md#sibling-dependency-floors): "
        f"{offenders}"
    )


def test_no_community_or_enterprise_runtime_file_imports_mloda_testing() -> None:
    offenders: list[str] = []
    for root_name in ("community", "enterprise"):
        root_dir = _REPO_ROOT / "mloda" / root_name
        assert root_dir.is_dir(), f"expected {root_dir} to exist"
        offenders.extend(f"mloda/{root_name}/{path.as_posix()}" for path in _files_importing_mloda_testing(root_dir))
    assert not offenders, f"mloda.testing imported outside tests/: {offenders}"


# Kit paths (relative to the kit root) whose imports may be declared in a named extra of mloda-testing.
_KIT_EXTRAS = {
    "binary_model": "binary-model",
    "feature_groups/data_operations": "data-operations",
    "extenders/otel.py": "otel",
    "extenders/openlineage.py": "openlineage",
}
_KIT_PACKAGE = "mloda-testing"
_KIT_MODULE = "mloda.testing"
# Import root -> PyPI distribution. A new third-party import in the kit must be added here.
_IMPORT_DISTRIBUTIONS = {
    "pytest": "pytest",
    "pyarrow": "pyarrow",
    "dateutil": "python-dateutil",
    "opentelemetry": "opentelemetry-sdk",
    "openlineage": "openlineage-python",
}
# The compute backend the plugin author brings: allowed at function scope only, never declared by the kit.
_USER_BACKENDS = frozenset({"pandas", "polars", "duckdb"})


def _scope_importorskips(scope: ast.AST) -> list[tuple[int, str]]:
    """(line, dotted name) of each ``importorskip("<literal>")`` call in a scope, nested functions excluded."""
    found: list[tuple[int, str]] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "importorskip"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pytest"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.append((node.lineno, node.args[0].value))
        stack.extend(ast.iter_child_nodes(node))
    return found


def _kit_imports(tree: ast.Module) -> list[tuple[str, bool, list[str]]]:
    """(dotted module, at function scope, earlier importorskip names in scope) of every import at any depth, in
    source order. A relative import is reported as its leading dots plus module (``..pkg``).

    Skips imports inside a trusted TYPE_CHECKING block (never executed at runtime, one rule for every check) and
    imports whose root is named by an earlier ``pytest.importorskip("<dotted>")`` in the same function or module
    scope. The exemption covers only that root and never a ``mloda.*`` import (the caller decides those from the
    returned importorskip names). Known gaps: a module-level ``try: import x / except ImportError`` optional
    import is flagged; the kit has none. ``importlib.import_module`` calls are not seen."""
    alias = trusted_type_checking_alias(tree)
    found: list[tuple[int, str, bool, list[str]]] = []

    def handle(child: ast.AST, in_function: bool, skips: list[tuple[int, str]]) -> None:
        if isinstance(child, ast.If) and is_type_checking_block(child, alias):
            for orelse in child.orelse:
                handle(orelse, in_function, skips)
            return
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.iter_child_nodes(child):
                handle(sub, True, _scope_importorskips(child))
            return
        names: list[str] = []
        if isinstance(child, ast.Import):
            names = [a.name for a in child.names]
        elif isinstance(child, ast.ImportFrom) and child.level > 0:
            names = ["." * child.level + (child.module or "")]
        elif isinstance(child, ast.ImportFrom) and child.module is not None:
            names = [f"{child.module}.{a.name}" for a in child.names]
        for name in names:
            root = name.split(".")[0]
            line: int = getattr(child, "lineno", 0)
            earlier = [d for skip_line, d in skips if skip_line < line]
            skipped = root not in ("mloda", "") and any(
                d == root or d.startswith(root + ".") or root.startswith(d + ".") for d in earlier
            )
            if not skipped:
                found.append((line, name, in_function, earlier))
        for sub in ast.iter_child_nodes(child):
            handle(sub, in_function, skips)

    def visit(node: ast.AST, in_function: bool, skips: list[tuple[int, str]]) -> None:
        for child in ast.iter_child_nodes(node):
            handle(child, in_function, skips)

    visit(tree, False, _scope_importorskips(tree))
    return [(name, in_function, earlier) for _line, name, in_function, earlier in sorted(found, key=lambda f: f[:2])]


def _owning_package(module: str, packages: dict[str, dict[str, Any]]) -> str | None:
    """Configured package with the longest path prefix of the dotted module."""
    module_path = module.replace(".", "/")
    owners = [
        (len(path), name)
        for name, cfg in packages.items()
        if (path := cfg.get("path", "")) and (module_path == path or module_path.startswith(path + "/"))
    ]
    return max(owners)[1] if owners else None


def _kit_gate(rel: str, kit_extras: dict[str, str]) -> str | None:
    """The kit_extras key gating a kit-relative path (file or dotted module turned into a path), if any."""
    stem = rel.removesuffix(".py")
    return next(
        (
            key
            for key in kit_extras
            if stem == key.removesuffix(".py")
            or stem.startswith(key.removesuffix(".py") + "/")
            or stem.startswith(key + "/")
        ),
        None,
    )


def _undeclared_kit_imports(
    kit_root: Path, packages: dict[str, dict[str, Any]], kit_extras: dict[str, str] = _KIT_EXTRAS
) -> list[str]:
    """Imports of the kit its mloda-testing declaration (dependencies plus the extra gating the file) does not cover."""
    kit_cfg = packages.get(_KIT_PACKAGE, {})
    optional = kit_cfg.get("optional_dependencies", {})
    violations: list[str] = []
    for rel_path in _runtime_py_files(kit_root):
        rel = rel_path.as_posix()
        gate = _kit_gate(rel, kit_extras)
        extra = kit_extras[gate] if gate else None
        deps = list(kit_cfg.get("dependencies", [])) + (list(optional.get(extra, [])) if extra else [])
        declared = {_dep_name(dep) for dep in deps}
        allowed_packages = gen.dependency_closure({"dependencies": deps}, packages)
        target = f"the {extra!r} extra" if extra else "dependencies"
        tree = ast.parse((kit_root / rel_path).read_text(encoding="utf-8"))
        seen: set[str] = set()
        for module, in_function, importorskips in _kit_imports(tree):
            root = module.split(".")[0]
            problem: str | None = None
            key = root
            if module.startswith("."):
                key = module
                problem = f"uses the relative import {module!r}; use an absolute import"
            elif module in ("mloda.community", "mloda.enterprise") or module.startswith(
                ("mloda.community.", "mloda.enterprise.")
            ):
                owner = _owning_package(module, packages)
                key = owner or module
                if owner is not None and owner not in allowed_packages:
                    problem = f"imports {module} (package {owner}); declare it in {target} of {_KIT_PACKAGE}"
            elif module == _KIT_MODULE or module.startswith(_KIT_MODULE + "."):
                target_gate = _kit_gate(module.removeprefix(_KIT_MODULE + ".").replace(".", "/"), kit_extras)
                key = module
                target_declared = (
                    {_dep_name(dep) for dep in optional.get(kit_extras[target_gate], [])} if target_gate else set()
                )
                guarded = any(
                    _dep_name(_IMPORT_DISTRIBUTIONS.get(d.split(".")[0], "")) in target_declared for d in importorskips
                )
                if target_gate is not None and target_gate != gate and not guarded:
                    problem = (
                        f"imports {module}, which sits under the {kit_extras[target_gate]!r} extra gate "
                        f"({target_gate}); move this module under that gate in _KIT_EXTRAS or stop importing it"
                    )
            elif root in FIRST_PARTY_IMPORT_ROOTS or root in STDLIB_ROOTS or root == "__future__":
                continue
            elif root in _USER_BACKENDS:
                if not in_function:
                    problem = (
                        f"imports {module} at module scope; the {root} backend is the plugin author's, "
                        "import it inside a function"
                    )
            elif root in _IMPORT_DISTRIBUTIONS:
                dist = _IMPORT_DISTRIBUTIONS[root]
                if _dep_name(dist) not in declared:
                    problem = f"imports {module} (distribution {dist}); declare it in {target} of {_KIT_PACKAGE}"
            else:
                problem = (
                    f"imports {module}, an unknown third-party name {root!r}; add it to _IMPORT_DISTRIBUTIONS "
                    f"and declare its distribution in {target} of {_KIT_PACKAGE}"
                    + ("" if gate else " (or add the subtree to _KIT_EXTRAS)")
                )
            if problem is not None and key not in seen:
                seen.add(key)
                violations.append(f"{rel}: {problem}")
    return violations


def test_kit_extras_gate_map_matches_the_non_dev_mloda_testing_extras() -> None:
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
    extras = set(packages[_KIT_PACKAGE].get("optional_dependencies", {})) - {"dev"}
    assert set(_KIT_EXTRAS.values()) == extras, (
        f"_KIT_EXTRAS must gate exactly the non-dev extras of {_KIT_PACKAGE} {sorted(extras)}, "
        f"got {sorted(set(_KIT_EXTRAS.values()))}"
    )


def test_mloda_testing_declares_everything_it_imports() -> None:
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
    kit_root = _REPO_ROOT / "mloda" / "testing"
    assert kit_root.is_dir(), f"expected {kit_root} to exist"
    violations = _undeclared_kit_imports(kit_root, packages)
    assert not violations, (
        "mloda-testing imports what it does not declare. Declare each in config/packages.toml under "
        f"[packages.mloda-testing] (dependencies or the gating extra), then regenerate: {violations}"
    )


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


@pytest.mark.parametrize(
    ("body", "flagged"),
    [
        pytest.param("from mloda.community.extenders.shared.teardown import _force_flush\n", True, id="private-name"),
        pytest.param("from mloda.community.extenders.shared._internal import force_flush\n", True, id="private-module"),
        pytest.param("import mloda.community.extenders.shared._internal\n", True, id="private-plain-import"),
        pytest.param("from mloda.community.extenders.shared.teardown import force_flush\n", False, id="public-name"),
        pytest.param(
            "from mloda.community.extenders.shared.open_invocations import OpenInvocationStack\n"
            "class S(OpenInvocationStack):\n    pass\n",
            True,
            id="subclass-imported-class",
        ),
        pytest.param(
            "from mloda.community.extenders import shared\nclass S(shared.open_invocations.OpenInvocationStack):\n"
            "    pass\n",
            True,
            id="subclass-via-module-alias",
        ),
        pytest.param(
            "from mloda.community.extenders.shared import teardown\nteardown._x\n",
            True,
            id="private-attribute-of-shared-module",
        ),
        pytest.param("from mloda.community.extenders.shared import __version__\n", False, id="dunder-import"),
        pytest.param("from mloda.community.extenders.shared_other._x import y\n", False, id="prefix-collision"),
        pytest.param(
            "from mloda.community.extenders.openlineage.openlineage_extender import _x\n",
            False,
            id="unrelated-private-import",
        ),
        pytest.param(
            "from mloda.community.extenders.shared.open_invocations import OpenInvocationStack\n"
            "s = OpenInvocationStack()\n",
            False,
            id="plain-instantiation",
        ),
    ],
)
def test_extenders_shared_private_use_predicate(tmp_path: Path, body: str, flagged: bool) -> None:
    _write(tmp_path / "m.py", body)
    expected = [Path("m.py")] if flagged else []
    assert _files_with_import(tmp_path, _uses_extenders_shared_privately) == expected


_LEAF_IMPORT = "def f() -> None:\n    from mloda.community.feature_groups.leaf.impl import Thing\n"
_SYNTHETIC_KIT_EXTRAS = {"gated": "x", "gated2": "y", "ext/otel.py": "z"}


def _synthetic_kit_packages() -> dict[str, dict[str, Any]]:
    return {
        _KIT_PACKAGE: {
            "path": "mloda/testing",
            "dependencies": ["pytest>=9"],
            "optional_dependencies": {"x": ["leaf>=1", "python-dateutil>=2"], "y": [], "z": ["opentelemetry-sdk>=1"]},
        },
        "leaf": {"path": "mloda/community/feature_groups/leaf", "dependencies": []},
        "bundle": {"path": "mloda/community", "dependencies": []},
    }


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param({"gated/m.py": _LEAF_IMPORT}, [], id="leaf-inside-gated-subtree"),
        pytest.param({"other/m.py": _LEAF_IMPORT}, ["other/m.py", "package leaf", "dependencies"], id="leaf-outside"),
        pytest.param({"gated2/m.py": _LEAF_IMPORT}, ["gated2/m.py", "package leaf", "'y' extra"], id="leaf-wrong-gate"),
        pytest.param(
            {"gated/m.py": "from mloda.community.feature_groups import leaf\n"}, [], id="from-import-not-bundle-owned"
        ),
        pytest.param(
            {"other/m.py": "def f() -> None:\n    import dateutil.parser\n"},
            ["other/m.py", "dateutil.parser", "python-dateutil"],
            id="undeclared-third-party-function-level",
        ),
        pytest.param({"gated/m.py": "def f() -> None:\n    import dateutil.parser\n"}, [], id="third-party-gated"),
        pytest.param(
            {"other/m.py": "import requests\n"},
            ["other/m.py", "requests", "add it to _IMPORT_DISTRIBUTIONS"],
            id="unknown",
        ),
        pytest.param(
            {
                "other/m.py": "import pytest\ndef f() -> None:\n    pytest.importorskip('dateutil.parser')\n    import dateutil.parser\n"
            },
            [],
            id="importorskip-exempts-its-root",
        ),
        pytest.param(
            {
                "other/m.py": "import pytest\ndef f() -> None:\n    pytest.importorskip('dateutil')\n"
                "    import dateutil.parser\n    import opentelemetry.sdk\n"
            },
            ["opentelemetry.sdk", "opentelemetry-sdk"],
            id="importorskip-not-another-root",
        ),
        pytest.param(
            {
                "other/m.py": "import pytest\ndef g() -> None:\n    pytest.importorskip('dateutil')\n"
                "def f() -> None:\n    import dateutil.parser\n"
            },
            ["dateutil.parser"],
            id="importorskip-other-scope",
        ),
        pytest.param(
            {
                "other/m.py": "import pytest\ndef f() -> None:\n    import dateutil.parser\n"
                "    pytest.importorskip('dateutil')\n"
            },
            ["dateutil.parser"],
            id="importorskip-after-import",
        ),
        pytest.param(
            {
                "other/m.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\nelse:\n    import requests\n"
            },
            ["requests"],
            id="type-checking-else-branch-flagged",
        ),
        pytest.param({"other/m.py": "import pandas\n"}, ["other/m.py", "pandas", "module scope"], id="backend-module"),
        pytest.param({"other/m.py": "def f() -> None:\n    import pandas\n"}, [], id="backend-function"),
        pytest.param(
            {"other/m.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import requests\n"},
            [],
            id="type-checking-ignored",
        ),
        pytest.param(
            {"gated/helper.py": "", "gated2/m.py": "from mloda.testing.gated.helper import X\n"},
            ["gated2/m.py", "mloda.testing.gated.helper", "'x' extra gate"],
            id="cross-gate-kit-import",
        ),
        pytest.param(
            {"gated/helper.py": "", "other/m.py": "from mloda.testing import gated\n"},
            ["other/m.py", "mloda.testing.gated", "'x' extra gate"],
            id="ungated-imports-gated",
        ),
        pytest.param(
            {
                "gated/helper.py": "",
                "gated/m.py": "from mloda.testing.gated.helper import X\nimport mloda.testing.base\n",
            },
            [],
            id="same-gate-or-ungated-kit-import",
        ),
        pytest.param(
            {"ext/otel.py": "", "other/m.py": "from mloda.testing.ext.otel import X\n"},
            ["other/m.py", "mloda.testing.ext.otel.X", "'z' extra gate"],
            id="file-key-gate-cross-import",
        ),
        pytest.param(
            {
                "ext/otel.py": "",
                "other/m.py": "import pytest\ndef f() -> None:\n    pytest.importorskip('opentelemetry.sdk.trace')\n"
                "    from mloda.testing.ext.otel import X\n",
            },
            [],
            id="file-key-gate-importorskip-target-dist",
        ),
        pytest.param(
            {
                "ext/otel.py": "",
                "other/m.py": "import pytest\ndef f() -> None:\n    pytest.importorskip('dateutil')\n"
                "    from mloda.testing.ext.otel import X\n",
            },
            ["other/m.py", "mloda.testing.ext.otel.X", "'z' extra gate"],
            id="file-key-gate-importorskip-other-dist",
        ),
        pytest.param(
            {"other/m.py": "import mloda.community\n"},
            ["other/m.py", "mloda.community", "package bundle"],
            id="bare-bundle-import",
        ),
        pytest.param(
            {"other/m.py": "from mloda import community\n"},
            ["other/m.py", "mloda.community", "package bundle"],
            id="from-mloda-import-bundle",
        ),
        pytest.param(
            {"gated/helper.py": "", "gated/m.py": "from . import helper\n"},
            ["gated/m.py", "absolute import"],
            id="relative-import-sibling",
        ),
        pytest.param(
            {"gated/helper.py": "", "gated2/m.py": "from ..gated import helper\n"},
            ["gated2/m.py", "absolute import"],
            id="relative-import-parent",
        ),
        pytest.param(
            {"other/m.py": "def f() -> None:\n    other.importorskip('requests')\n    import requests\n"},
            ["other/m.py", "requests"],
            id="importorskip-not-from-pytest",
        ),
        pytest.param(
            {
                "other/m.py": "import pytest\ndef f() -> None:\n    pytest.importorskip('mloda.community')\n"
                "    from mloda.community.feature_groups.leaf.impl import Thing\n"
            },
            ["other/m.py", "package leaf", "dependencies"],
            id="importorskip-never-exempts-mloda-community",
        ),
        pytest.param(
            {
                "other/m.py": "import pytest\ndef f() -> None:\n    pytest.importorskip('mloda')\n"
                "    from mloda.community.feature_groups.leaf.impl import Thing\n"
            },
            ["other/m.py", "package leaf", "dependencies"],
            id="importorskip-never-exempts-mloda-root",
        ),
    ],
)
def test_undeclared_kit_imports_walker(tmp_path: Path, files: dict[str, str], expected: list[str]) -> None:
    for rel, body in files.items():
        _write(tmp_path / rel, body)
    violations = _undeclared_kit_imports(tmp_path, _synthetic_kit_packages(), _SYNTHETIC_KIT_EXTRAS)
    assert len(violations) == (1 if expected else 0), violations
    assert all(part in violations[0] for part in expected), violations
