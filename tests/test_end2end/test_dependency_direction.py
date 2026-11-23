"""Dependency-direction guard: community and enterprise plugin packages must never depend on
``mloda-testing`` or on a binary wheel (see ``docs/guides/feature-group-patterns/29-binary-backed-
features.md``), and no runtime module under ``mloda/community/`` or ``mloda/enterprise/`` may import
``mloda.testing`` or ``mloda.enterprise.licensing.testing`` at any depth (only
``mloda/enterprise/licensing/testing/`` is exempt). The kit imports
no ``mloda.enterprise`` module anywhere and ``mloda-testing`` names no enterprise distribution in any
extra. Enterprise code may use ``mloda.community.extenders.shared`` only
through public names (no private imports, subclassing, or private attributes). Binary wheel floors must
also exclude releases that trust no production key. Mirrors the resolution
and TOML-loading style of ``tests/test_end2end/test_dev_dependencies.py`` and
``tests/test_end2end/test_manifest_resilience.py``.
"""

from __future__ import annotations

import ast
import importlib
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from mloda.community.feature_groups.binary_model.mixin import BinaryModelMixin
from tests.script_loader import load_script
from tests.test_end2end.test_manifest_resilience import (
    _FIRST_PARTY_IMPORT_ROOTS,
    _STDLIB_ROOTS,
    _is_type_checking_if,
    _trusted_type_checking_alias,
)
from tests.toml_loader import load_toml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"
_SHARED_CONFIG = _REPO_ROOT / "config" / "shared.toml"
_TESTING_PYPROJECT = _REPO_ROOT / "mloda" / "testing" / "pyproject.toml"

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


def _mloda_testing_plugin_distribution_violations(packages: dict[str, dict[str, Any]]) -> list[str]:
    """Enterprise distributions that mloda-testing names anywhere, and community distributions outside its
    binary-model extra."""
    cfg = packages.get("mloda-testing", {})
    specs = {"dependencies": cfg.get("dependencies", [])}
    for extra, extra_deps in cfg.get("optional_dependencies", {}).items():
        specs[f"extra {extra}"] = extra_deps
    return [
        f"mloda-testing ({where}): {dep}"
        for where, deps in specs.items()
        for dep in deps
        if _dep_name(dep).startswith("mloda-enterprise")
        or (_dep_name(dep).startswith("mloda-community") and where != "extra binary-model")
    ]


def test_mloda_testing_names_no_enterprise_distribution_and_community_only_in_binary_model_extra() -> None:
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
    assert "mloda-testing" in packages, "mloda-testing missing from config/packages.toml"
    violations = _mloda_testing_plugin_distribution_violations(packages)
    assert violations == [], f"the test kit must not depend on enterprise code: {violations}"


def test_mloda_testing_plugin_distribution_violations_are_detected() -> None:
    hard = {"mloda-testing": {"dependencies": ["mloda_community_aggregation>=1"]}}
    assert _mloda_testing_plugin_distribution_violations(hard) == [
        "mloda-testing (dependencies): mloda_community_aggregation>=1"
    ]
    extra = {"mloda-testing": {"optional_dependencies": {"dev": ["mloda-enterprise[ed25519]>=1"]}}}
    assert _mloda_testing_plugin_distribution_violations(extra) == [
        "mloda-testing (extra dev): mloda-enterprise[ed25519]>=1"
    ]
    allowed = {
        "mloda-testing": {
            "dependencies": ["pyarrow"],
            "optional_dependencies": {"binary-model": ["pyarrow>=25", "mloda-community-binary-model>=1"]},
        }
    }
    assert _mloda_testing_plugin_distribution_violations(allowed) == []
    enterprise_in_binary_model = {
        "mloda-testing": {"optional_dependencies": {"binary-model": ["mloda-enterprise[ed25519]>=1"]}}
    }
    assert _mloda_testing_plugin_distribution_violations(enterprise_in_binary_model) == [
        "mloda-testing (extra binary-model): mloda-enterprise[ed25519]>=1"
    ]
    bare_enterprise_in_binary_model = {
        "mloda-testing": {"optional_dependencies": {"binary-model": ["mloda-enterprise>=1"]}}
    }
    assert _mloda_testing_plugin_distribution_violations(bare_enterprise_in_binary_model) == [
        "mloda-testing (extra binary-model): mloda-enterprise>=1"
    ]
    community_elsewhere = {"mloda-testing": {"optional_dependencies": {"otel": ["mloda-community-binary-model>=1"]}}}
    assert _mloda_testing_plugin_distribution_violations(community_elsewhere) == [
        "mloda-testing (extra otel): mloda-community-binary-model>=1"
    ]


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


def test_mloda_testing_has_no_hard_pyarrow_dependency() -> None:
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
    names = [_dep_name(dep) for dep in packages["mloda-testing"].get("dependencies", [])]
    assert "pyarrow" not in names, f"mloda-testing: pyarrow must be an extra, not a dependency, got {names!r}"


@pytest.mark.parametrize("extra", ["pyarrow", "otel", "openlineage", "binary-model", "dev"])
def test_mloda_testing_extra_carries_pyarrow(extra: str) -> None:
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
    deps = packages["mloda-testing"].get("optional_dependencies", {}).get(extra)
    assert deps is not None, f"mloda-testing: no extra {extra!r}"
    assert "pyarrow" in [_dep_name(dep) for dep in deps], (
        f"mloda-testing: the {extra} extra must name pyarrow, got {deps!r}"
    )


def test_shared_default_dev_extra_installs_mloda_testing_with_pyarrow() -> None:
    dev = load_toml(_SHARED_CONFIG)["defaults"]["optional_dependencies"]["dev"]
    assert "mloda-testing[pyarrow]" in dev, (
        f"the shared default dev extra must name mloda-testing[pyarrow], got {dev!r}"
    )
    assert "mloda-testing" not in dev, f"the bare mloda-testing would drop the pyarrow extra, got {dev!r}"


def test_generated_mloda_testing_pyproject_does_not_depend_on_itself() -> None:
    project = load_toml(_TESTING_PYPROJECT)["project"]
    specs = list(project.get("dependencies", []))
    for deps in project.get("optional-dependencies", {}).values():
        specs.extend(deps)
    assert [spec for spec in specs if _dep_name(spec) == "mloda-testing"] == []


def _is_import_of(node: ast.AST, dotted: str, from_parent: bool = False) -> bool:
    """True for an Import or ImportFrom of ``dotted`` or a submodule; with from_parent, also ``from <parent> import <leaf>``.
    No scope exemption: callers walk the whole tree with ast.walk."""
    if isinstance(node, ast.Import):
        return any(alias.name == dotted or alias.name.startswith(dotted + ".") for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        if node.module is None:
            return False
        parent, _, leaf = dotted.rpartition(".")
        if from_parent and node.module == parent:
            return any(alias.name == leaf for alias in node.names)
        return node.module == dotted or node.module.startswith(dotted + ".")
    return False


def _is_mloda_testing_import(node: ast.AST) -> bool:
    """True for an import of ``mloda.testing`` or a submodule, at any depth (no TYPE_CHECKING exemption)."""
    return _is_import_of(node, "mloda.testing")


def _runtime_modules(root_dir: Path, include_tests: bool = False) -> Iterator[tuple[Path, ast.Module]]:
    """(relative path, parsed tree) of every .py file under root_dir, outside any tests/ directory (unless
    include_tests), build artifact directory and egg-info directory."""
    for py_file in sorted(root_dir.rglob("*.py")):
        rel_path = py_file.relative_to(root_dir)
        parts = rel_path.parts
        if ("tests" in parts and not include_tests) or any(
            part in _SKIP_DIR_NAMES or part.endswith(".egg-info") for part in parts
        ):
            continue
        yield rel_path, ast.parse(py_file.read_text(encoding="utf-8"))


def _files_with_import(root_dir: Path, predicate: Callable[[ast.AST], bool], include_tests: bool = False) -> list[Path]:
    """Every .py file under root_dir, outside any tests/ directory unless include_tests, with a node matching
    predicate at any depth (module level, function level, or inside a TYPE_CHECKING block)."""
    return [
        rel_path
        for rel_path, tree in _runtime_modules(root_dir, include_tests)
        if any(predicate(n) for n in ast.walk(tree))
    ]


_PLUGIN_PACKAGES = ("mloda.community", "mloda.enterprise")


def _plugin_package_of(module: str) -> str | None:
    """The mloda.community / mloda.enterprise package `module` is, or lives under (exact dotted prefix)."""
    return next((p for p in _PLUGIN_PACKAGES if module == p or module.startswith(p + ".")), None)


def _plugin_modules_imported_by(node: ast.AST) -> list[str]:
    """Modules of mloda.community / mloda.enterprise that `node` imports: an Import, an ImportFrom
    (``from mloda import community`` records ``mloda.community``), or an import_module / importorskip
    call with a string literal first argument. A dynamic argument cannot be checked."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names if _plugin_package_of(a.name)]
    if isinstance(node, ast.ImportFrom):
        if node.module is None or node.level:
            return []
        if _plugin_package_of(node.module):
            return [node.module]
        if node.module == "mloda":
            return [f"mloda.{a.name}" for a in node.names if f"mloda.{a.name}" in _PLUGIN_PACKAGES]
        return []
    if isinstance(node, ast.Call) and node.args:
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        first = node.args[0]
        if (
            name in ("import_module", "importorskip")
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and _plugin_package_of(first.value)
        ):
            return [first.value]
    return []


_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda


def _walk_with_scope(
    node: ast.AST,
    enclosing: _FunctionNode | None = None,
    skip_type_checking: bool = False,
    type_checking_alias: str | None = None,
) -> Iterator[tuple[ast.AST, _FunctionNode | None]]:
    """Every node under `node` with its enclosing function (None at module level or in a class body);
    skip_type_checking drops the body, not the orelse, of a trusted ``if TYPE_CHECKING:``."""
    if skip_type_checking and isinstance(node, ast.Module):
        type_checking_alias = _trusted_type_checking_alias(node)
    yield node, enclosing
    inside = node if isinstance(node, _FunctionNode) else enclosing
    skipped = (
        node.body
        if skip_type_checking and isinstance(node, ast.If) and _is_type_checking_if(node, type_checking_alias)
        else []
    )
    for child in ast.iter_child_nodes(node):
        if child not in skipped:
            yield from _walk_with_scope(child, inside, skip_type_checking, type_checking_alias)


def _kit_plugin_imports(root_dir: Path, *, exempt_binary_model: bool = True) -> set[tuple[str, str]]:
    """(posix path relative to root_dir, imported module) for every mloda.community / mloda.enterprise
    import outside tests/, and outside the top-level binary_model/ directory unless exempt_binary_model is
    False. A dynamic import_module argument cannot be checked."""
    found: set[tuple[str, str]] = set()
    for rel_path, tree in _runtime_modules(root_dir):
        if exempt_binary_model and rel_path.parts[0] == "binary_model":
            continue
        for node in ast.walk(tree):
            found.update((rel_path.as_posix(), module) for module in _plugin_modules_imported_by(node))
    return found


def _kit_enterprise_imports(root_dir: Path) -> set[tuple[str, str]]:
    """(posix path relative to root_dir, imported module) for every mloda.enterprise import outside tests/,
    with no directory exemption."""
    return {
        (path, module)
        for path, module in _kit_plugin_imports(root_dir, exempt_binary_model=False)
        if _plugin_package_of(module) == "mloda.enterprise"
    }


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


_LICENSING_TESTING_DIR = Path("mloda/enterprise/licensing/testing")


def _importers_outside_exempt_dir(root_name: str, root_dir: Path, predicate: Callable[[ast.AST], bool]) -> list[str]:
    """Repo-relative paths of files under root_dir with a node matching predicate, outside the exempt directory."""
    paths = (Path("mloda") / root_name / path for path in _files_with_import(root_dir, predicate))
    return [path.as_posix() for path in paths if _LICENSING_TESTING_DIR not in path.parents]


def _is_licensing_testing_import(node: ast.AST) -> bool:
    """True for an import of ``mloda.enterprise.licensing.testing`` or a submodule, at any depth."""
    return _is_import_of(node, "mloda.enterprise.licensing.testing", from_parent=True)


@pytest.mark.parametrize(
    ("predicate", "message"),
    [
        pytest.param(_is_mloda_testing_import, "mloda.testing imported outside tests/", id="mloda_testing"),
        pytest.param(
            _is_licensing_testing_import,
            "mloda.enterprise.licensing.testing imported outside tests/",
            id="licensing_test_tooling",
        ),
    ],
)
def test_no_community_or_enterprise_runtime_file_imports_the_test_tooling(
    predicate: Callable[[ast.AST], bool], message: str
) -> None:
    offenders: list[str] = []
    for root_name in ("community", "enterprise"):
        root_dir = _REPO_ROOT / "mloda" / root_name
        assert root_dir.is_dir(), f"expected {root_dir} to exist"
        offenders.extend(_importers_outside_exempt_dir(root_name, root_dir, predicate))
    assert not offenders, f"{message}: {offenders}"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("from mloda.enterprise.licensing.testing import license_vectors\n", id="from-package-import"),
        pytest.param("import mloda.enterprise.licensing.testing.licensed_binary\n", id="plain-submodule-import"),
        pytest.param("from mloda.enterprise.licensing import testing\n", id="from-parent-import"),
    ],
)
def test_licensing_test_tooling_rule_flags_a_runtime_importer(tmp_path: Path, body: str) -> None:
    _write(tmp_path / "feature_groups" / "x" / "manifest.py", body)
    assert _importers_outside_exempt_dir("enterprise", tmp_path, _is_licensing_testing_import) == [
        "mloda/enterprise/feature_groups/x/manifest.py"
    ]


def test_licensing_test_tooling_rule_ignores_the_exempt_directory_and_tests(tmp_path: Path) -> None:
    _write(
        tmp_path / "licensing" / "testing" / "a.py", "from mloda.enterprise.licensing.testing import license_vectors\n"
    )
    _write(
        tmp_path / "feature_groups" / "x" / "tests" / "test_m.py", "from mloda.enterprise.licensing import testing\n"
    )
    assert _importers_outside_exempt_dir("enterprise", tmp_path, _is_licensing_testing_import) == []


def _is_core_testing_import(node: ast.AST) -> bool:
    """True for an import of ``mloda._testing`` (core's non-public conformance kit) at any depth."""
    return _is_import_of(node, "mloda._testing", from_parent=True)


def test_no_plugin_file_imports_core_testing_directly() -> None:
    """Plugins, tests included, reach core's suites only through mloda.testing."""
    offenders: list[str] = []
    for root_name in ("community", "enterprise", "registry"):
        root_dir = _REPO_ROOT / "mloda" / root_name
        assert root_dir.is_dir(), f"expected {root_dir} to exist"
        offenders.extend(
            f"mloda/{root_name}/{path.as_posix()}"
            for path in _files_with_import(root_dir, _is_core_testing_import, include_tests=True)
        )
    assert not offenders, f"mloda._testing imported by plugin code, use mloda.testing instead: {offenders}"


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_mloda_testing_rule_exempts_only_the_enterprise_licensing_testing_directory(tmp_path: Path) -> None:
    body = "from mloda.testing.base import FeatureGroupTestBase\n"
    for rel in ("licensing/testing/a.py", "licensing/other.py", "licensing/testing_extra/b.py", "other/c.py"):
        _write(tmp_path / rel, body)
    assert _importers_outside_exempt_dir("enterprise", tmp_path, _is_mloda_testing_import) == [
        "mloda/enterprise/licensing/other.py",
        "mloda/enterprise/licensing/testing_extra/b.py",
        "mloda/enterprise/other/c.py",
    ]


def test_walker_catches_function_level_import(tmp_path: Path) -> None:
    _write(tmp_path / "m.py", "def f() -> None:\n    from mloda.testing.base import FeatureGroupTestBase\n")
    assert _files_with_import(tmp_path, _is_mloda_testing_import) == [Path("m.py")]


def test_walker_catches_type_checking_import(tmp_path: Path) -> None:
    """No exemption for TYPE_CHECKING blocks, unlike test_manifest_resilience.py's own walker."""
    body = (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from mloda.testing.base import FeatureGroupTestBase\n"
    )
    _write(tmp_path / "m.py", body)
    assert _files_with_import(tmp_path, _is_mloda_testing_import) == [Path("m.py")]


def test_walker_ignores_import_inside_a_tests_directory(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    _write(tests_dir / "test_m.py", "from mloda.testing.base import FeatureGroupTestBase\n")
    assert _files_with_import(tmp_path, _is_mloda_testing_import) == []


@pytest.mark.parametrize(
    ("body", "flagged"),
    [
        pytest.param("import mloda._testing.suite\n", True, id="plain-import"),
        pytest.param("from mloda._testing.suite import Suite\n", True, id="from-import"),
        pytest.param("from mloda import _testing\n", True, id="from-package-import"),
        pytest.param("def f() -> None:\n    from mloda._testing import result\n", True, id="function-level"),
        pytest.param("from mloda.testing.suite import Suite\n", False, id="public-door"),
        pytest.param("from mloda._testing_other import x\n", False, id="prefix-collision"),
    ],
)
def test_core_testing_import_predicate_inside_tests(tmp_path: Path, body: str, flagged: bool) -> None:
    _write(tmp_path / "tests" / "test_m.py", body)
    expected = [Path("tests/test_m.py")] if flagged else []
    assert _files_with_import(tmp_path, _is_core_testing_import, include_tests=True) == expected


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


_KIT_ROOT = _REPO_ROOT / "mloda" / "testing"


def test_kit_imports_no_community_or_enterprise() -> None:
    assert _KIT_ROOT.is_dir(), f"expected {_KIT_ROOT} to exist"
    offenders = sorted(_kit_plugin_imports(_KIT_ROOT))
    assert offenders == [], (
        f"mloda/testing must not import mloda.community or mloda.enterprise at any depth: {offenders}"
    )


@pytest.mark.parametrize(
    ("body", "module"),
    [
        pytest.param("from mloda.community.x import y\n", "mloda.community.x", id="module-level"),
        pytest.param(
            "def f() -> None:\n    from mloda.community.x import y\n", "mloda.community.x", id="function-level"
        ),
        pytest.param(
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from mloda.community.x import y\n",
            "mloda.community.x",
            id="type-checking",
        ),
        pytest.param("class C:\n    from mloda.community.x import y\n", "mloda.community.x", id="class-body"),
        pytest.param("import mloda.enterprise.x\n", "mloda.enterprise.x", id="plain-import"),
        pytest.param("from mloda import community\n", "mloda.community", id="from-mloda-import-community"),
        pytest.param("from mloda import enterprise\n", "mloda.enterprise", id="from-mloda-import-enterprise"),
        pytest.param(
            'import importlib\nimportlib.import_module("mloda.community.x")\n',
            "mloda.community.x",
            id="import-module-literal",
        ),
        pytest.param(
            'import pytest\npytest.importorskip("mloda.community.x")\n',
            "mloda.community.x",
            id="importorskip-literal",
        ),
        pytest.param(
            'from importlib import import_module\nimport_module("mloda.enterprise")\n',
            "mloda.enterprise",
            id="bare-import-module-literal",
        ),
    ],
)
def test_kit_plugin_import_collector_collects(tmp_path: Path, body: str, module: str) -> None:
    _write(tmp_path / "m.py", body)
    assert _kit_plugin_imports(tmp_path) == {("m.py", module)}


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("from mloda.community_other.x import y\n", id="prefix-from"),
        pytest.param("import mloda.enterprise_other\n", id="prefix-import"),
        pytest.param('import importlib\nimportlib.import_module("mloda.community_other")\n', id="prefix-literal"),
        pytest.param("import importlib\nname = 'x'\nimportlib.import_module(name)\n", id="dynamic-argument"),
        pytest.param("from mloda import user\n", id="other-mloda-name"),
    ],
)
def test_kit_plugin_import_collector_ignores(tmp_path: Path, body: str) -> None:
    _write(tmp_path / "m.py", body)
    assert _kit_plugin_imports(tmp_path) == set()


def test_kit_plugin_import_collector_skips_tests_and_top_level_binary_model(tmp_path: Path) -> None:
    body = "from mloda.community.x import y\n"
    _write(tmp_path / "tests" / "test_m.py", body)
    _write(tmp_path / "pkg" / "tests" / "test_m.py", body)
    _write(tmp_path / "binary_model" / "m.py", body)
    _write(tmp_path / "binary_model" / "sub" / "m.py", body)
    assert _kit_plugin_imports(tmp_path) == set()


def test_kit_plugin_import_collector_scans_a_nested_binary_model_directory(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "binary_model" / "m.py", "from mloda.community.x import y\n")
    assert _kit_plugin_imports(tmp_path) == {("pkg/binary_model/m.py", "mloda.community.x")}


def test_kit_imports_no_enterprise_module_including_binary_model() -> None:
    offenders = sorted(_kit_enterprise_imports(_KIT_ROOT))
    assert offenders == [], f"mloda/testing must not import mloda.enterprise anywhere: {offenders}"


def test_kit_enterprise_import_collector_scans_binary_model_and_skips_community_and_tests(tmp_path: Path) -> None:
    _write(tmp_path / "binary_model" / "m.py", "from mloda.enterprise.x import y\n")
    _write(tmp_path / "binary_model" / "c.py", "from mloda.community.x import y\n")
    _write(tmp_path / "binary_model" / "tests" / "test_m.py", "from mloda.enterprise.x import y\n")
    _write(tmp_path / "tests" / "test_m.py", "from mloda.enterprise.x import y\n")
    assert _kit_enterprise_imports(tmp_path) == {("binary_model/m.py", "mloda.enterprise.x")}


kit_map = load_script("kit_modules", _REPO_ROOT / "scripts" / "kit_modules.py")
verify = load_script("verify_independent_installs", _REPO_ROOT / "scripts" / "verify_independent_installs.py")


def _kit_modules_flat() -> dict[str, str | None]:
    """Dotted name of every shipped kit module, mapped to its gating extra."""
    return {module: gate for gate, modules in kit_map.kit_modules().items() for module in modules}


def test_kit_gates_extras_equal_the_non_dev_extras_of_mloda_testing() -> None:
    extras = set(_mloda_testing_config().get("optional_dependencies", {})) - {"dev"}
    assert set(kit_map.KIT_GATES.values()) == extras, (
        f"KIT_GATES extras {sorted(set(kit_map.KIT_GATES.values()))} must equal mloda-testing's non-dev extras "
        f"{sorted(extras)}"
    )


def test_every_kit_gate_prefix_matches_a_shipped_module() -> None:
    rel_paths = [rel.as_posix() for rel, _tree in _runtime_modules(_REPO_ROOT / kit_map.KIT_PATH)]
    unmatched = [prefix for prefix in kit_map.KIT_GATES if not any(p.startswith(prefix) for p in rel_paths)]
    assert unmatched == [], f"KIT_GATES prefixes that match no shipped kit module: {unmatched}"


def test_child_kit_gate_extra_lists_every_distribution_of_its_parent_gate_extra() -> None:
    extras = _mloda_testing_config().get("optional_dependencies", {})
    nested = [(p, c) for p in kit_map.KIT_GATES for c in kit_map.KIT_GATES if c != p and c.startswith(p)]
    assert nested, "no KIT_GATES prefix extends another; check is vacuous"
    for parent, child in nested:
        missing = {_dep_name(d) for d in extras[kit_map.KIT_GATES[parent]]} - {
            _dep_name(d) for d in extras[kit_map.KIT_GATES[child]]
        }
        assert missing == set(), f"the {child} extra lacks {sorted(missing)} of its parent {parent} extra"


def test_real_kit_has_a_binary_model_module_and_no_tests_module() -> None:
    flat = _kit_modules_flat()
    assert "binary-model" in set(flat.values()), "no shipped kit module is gated by binary-model"
    assert all(g == "binary-model" for m, g in flat.items() if m.startswith("mloda.testing.binary_model.")), flat
    assert [m for m in flat if "tests" in m.split(".")] == [], "a tests module is listed as a shipped kit module"


@pytest.mark.parametrize(
    ("rel", "gate"),
    [
        pytest.param("extenders/otel.py", "otel", id="longest-prefix-file"),
        pytest.param("extenders/openlineage.py", "openlineage", id="longest-prefix-other-file"),
        pytest.param("extenders/runners.py", "pyarrow", id="shorter-prefix"),
        pytest.param("binary_model/a/b.py", "binary-model", id="directory-prefix"),
        pytest.param("feature_groups/data_operations/x.py", None, id="data-operations"),
        pytest.param("feature_groups/other.py", None, id="sibling-directory-ungated"),
        pytest.param("base.py", None, id="top-level-ungated"),
    ],
)
def test_kit_gate_picks_the_longest_matching_prefix(rel: str, gate: str | None) -> None:
    assert kit_map.kit_gate(rel) == gate


def test_kit_modules_groups_sorts_and_names_packages(tmp_path: Path) -> None:
    _write(tmp_path / "__init__.py", "")
    _write(tmp_path / "b.py", "")
    _write(tmp_path / "a.py", "")
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(tmp_path / "pkg" / "m.py", "")
    _write(tmp_path / "extenders" / "otel.py", "")
    _write(tmp_path / "extenders" / "runners.py", "")
    assert kit_map.kit_modules(tmp_path) == {
        None: ["mloda.testing", "mloda.testing.a", "mloda.testing.b", "mloda.testing.pkg", "mloda.testing.pkg.m"],
        "otel": ["mloda.testing.extenders.otel"],
        "pyarrow": ["mloda.testing.extenders.runners"],
    }


@pytest.mark.parametrize(
    "skipped",
    [
        pytest.param("tests/test_m.py", id="tests"),
        pytest.param("pkg/tests/test_m.py", id="nested-tests"),
        pytest.param("__pycache__/m.py", id="pycache"),
        pytest.param("build/m.py", id="build"),
        pytest.param("dist/m.py", id="dist"),
        pytest.param("x.egg-info/m.py", id="egg-info"),
    ],
)
def test_kit_modules_skips_build_and_test_directories(tmp_path: Path, skipped: str) -> None:
    _write(tmp_path / "keep.py", "")
    _write(tmp_path / skipped, "")
    assert kit_map.kit_modules(tmp_path) == {None: ["mloda.testing.keep"]}


# Import root -> the distribution that provides it.
_KIT_IMPORT_DISTRIBUTIONS = {
    "pytest": "pytest",
    "pyarrow": "pyarrow",
    "opentelemetry": "opentelemetry-sdk",
    "openlineage": "openlineage-python",
    "cryptography": "cryptography",
}

# Allowed inside a function in any kit module (pyarrow because data_creator/pyarrow.py imports it lazily
# for an author's pyarrow backend).
_AUTHOR_BACKENDS = frozenset({"pandas", "polars", "duckdb", "pyarrow"})


def _mloda_testing_config() -> dict[str, Any]:
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG).get("packages", {})
    return packages["mloda-testing"]


def _kit_declared_distributions(cfg: dict[str, Any], gate: str | None) -> set[str]:
    """Normalized distributions a package config declares: its dependencies plus the named extra."""
    specs = list(cfg.get("dependencies", []))
    if gate is not None:
        specs.extend(cfg.get("optional_dependencies", {}).get(gate, []))
    return {_dep_name(spec) for spec in specs}


def _imports_root_via_importorskip(fn: _FunctionNode, root: str, before_line: int) -> bool:
    """True if a top-level statement of `fn` is importorskip("<root>[.sub]") with a literal, before `before_line`."""
    if isinstance(fn, ast.Lambda):
        return False
    for stmt in fn.body:
        call = stmt.value if isinstance(stmt, ast.Expr) else None
        if not (isinstance(call, ast.Call) and call.args and call.lineno < before_line):
            continue
        func = call.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        first = call.args[0]
        if (
            name == "importorskip"
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and first.value.split(".")[0] == root
        ):
            return True
    return False


def _import_roots_of(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [node.module.split(".")[0]]
    return []


def _import_violation_text(
    root: str,
    node: ast.AST,
    fn: _FunctionNode | None,
    declared: set[str],
    label: str,
    author_backends: frozenset[str],
) -> str | None:
    """Why importing `root` at `node` is not covered by `declared` (None when it is)."""
    if root in _STDLIB_ROOTS or root in _FIRST_PARTY_IMPORT_ROOTS:
        return None
    if root in author_backends and fn is not None:
        return None
    distribution = _KIT_IMPORT_DISTRIBUTIONS.get(root)
    if distribution is None and root in author_backends:
        return f"author backend {root!r} must be imported inside a function"
    if distribution is None:
        return f"unknown import name {root!r}"
    if _dep_name(distribution) in declared:
        return None
    if fn is not None and _imports_root_via_importorskip(fn, root, getattr(node, "lineno", 0)):
        return None
    return f"import {root!r} needs {distribution!r}, not declared for {label}"


def _dotted_import_targets(node: ast.AST, package: list[str]) -> list[str]:
    """Dotted modules an import statement may load, relative imports resolved against `package`."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if node.level:
        if node.level - 1 >= len(package):
            return []
        base = package[: len(package) - (node.level - 1)] + (node.module.split(".") if node.module else [])
    else:
        base = node.module.split(".") if node.module else []
    if not base:
        return []
    return [".".join(base)] + [".".join([*base, alias.name]) for alias in node.names if alias.name != "*"]


def _files_of_dotted_module(dotted: str, import_root: Path) -> list[Path]:
    """Existing files under `import_root` that importing `dotted` runs: its module and every parent package."""
    parts = dotted.split(".")
    files = [import_root.joinpath(*parts[:i], "__init__.py") for i in range(1, len(parts))]
    files += [import_root.joinpath(*parts).with_suffix(".py"), import_root.joinpath(*parts, "__init__.py")]
    return [f for f in files if f.is_file()]


def _followed_import_violations(
    kit_file: Path,
    kit_rel: str,
    import_root: Path,
    declared: set[str],
    label: str,
    author_backends: frozenset[str],
) -> list[str]:
    """Violations in the files `kit_file` and its ancestor packages reach through module-level first-party imports,
    judged against `declared`.

    Conservative: a followed file's own distribution dependencies do not count; guarded module-level imports
    are still flagged."""
    violations: list[str] = []
    kit_dotted = ".".join(kit_file.relative_to(import_root).with_suffix("").parts)
    queue = [kit_file, *(f for f in _files_of_dotted_module(kit_dotted, import_root) if f != kit_file)]
    seen = set(queue)
    while queue:
        current = queue.pop(0)
        rel_parts = current.relative_to(import_root).with_suffix("").parts
        package = list(rel_parts[:-1])
        tree = ast.parse(current.read_text(encoding="utf-8"))
        for node, fn in _walk_with_scope(tree, skip_type_checking=True):
            if fn is not None:
                continue
            if current != kit_file:
                for root in _import_roots_of(node):
                    text = _import_violation_text(root, node, fn, declared, label, author_backends)
                    if text is not None:
                        where = f"{current.relative_to(import_root).as_posix()}:{getattr(node, 'lineno', 0)}"
                        violations.append(f"{kit_rel}: via {where}: {text}")
            for dotted in _dotted_import_targets(node, package):
                for target in _files_of_dotted_module(dotted, import_root):
                    if target not in seen:
                        seen.add(target)
                        queue.append(target)
    return violations


def _third_party_import_violations(
    root_dir: Path,
    declared_for: Callable[[str], tuple[set[str], str]],
    *,
    author_backends: frozenset[str],
    import_root: Path,
) -> list[str]:
    """Third-party imports in shipped modules, and in the in-repo modules they import at module level, that
    `declared_for(rel_posix)` (declared set, label) does not cover.

    Only `author_backends` may be imported inside a function without being declared."""
    import_root = import_root.resolve()
    violations: list[str] = []
    for rel_path, tree in _runtime_modules(root_dir):
        declared, label = declared_for(rel_path.as_posix())
        for node, fn in _walk_with_scope(tree, skip_type_checking=True):
            for root in _import_roots_of(node):
                text = _import_violation_text(root, node, fn, declared, label, author_backends)
                if text is not None:
                    violations.append(f"{rel_path.as_posix()}:{getattr(node, 'lineno', 0)}: {text}")
        violations += _followed_import_violations(
            (root_dir / rel_path).resolve(), rel_path.as_posix(), import_root, declared, label, author_backends
        )
    return violations


def _kit_third_party_import_violations(
    root_dir: Path, cfg: dict[str, Any] | None = None, import_root: Path = _REPO_ROOT
) -> list[str]:
    """Third-party imports in shipped kit modules that mloda-testing does not declare for that module."""
    cfg = _mloda_testing_config() if cfg is None else cfg

    def declared_for(rel_posix: str) -> tuple[set[str], str]:
        gate = kit_map.kit_gate(rel_posix)
        return _kit_declared_distributions(cfg, gate), f"gate {gate!r}"

    return _third_party_import_violations(
        root_dir, declared_for, author_backends=_AUTHOR_BACKENDS, import_root=import_root
    )


def _license_kit_declared_distributions(packages: dict[str, dict[str, Any]]) -> set[str]:
    """Normalized distributions the license kit job's requirements declare: dependencies plus each named extra."""
    declared: set[str] = set()
    for spec in verify.LICENSE_KIT_REQUIREMENTS:
        req = Requirement(spec)
        for extra in req.extras or (None,):
            declared |= _kit_declared_distributions(packages[req.name], extra)
    return declared


def _license_kit_third_party_import_violations(
    root_dir: Path, packages: dict[str, dict[str, Any]] | None = None, import_root: Path = _REPO_ROOT
) -> list[str]:
    """Third-party imports in the license kit that its job's requirements do not declare."""
    packages = load_toml(_PACKAGES_CONFIG).get("packages", {}) if packages is None else packages
    declared = _license_kit_declared_distributions(packages)
    label = " + ".join(verify.LICENSE_KIT_REQUIREMENTS)
    return _third_party_import_violations(
        root_dir, lambda _rel: (declared, label), author_backends=frozenset(), import_root=import_root
    )


def test_kit_imports_only_third_party_distributions_mloda_testing_declares() -> None:
    assert _KIT_ROOT.is_dir(), f"expected {_KIT_ROOT} to exist"
    violations = _kit_third_party_import_violations(_KIT_ROOT)
    assert violations == [], f"mloda/testing imports a distribution it does not declare for that module: {violations}"


_SKIP_THEN_IMPORT = (
    'import pytest\n\n\ndef f() -> None:\n    pytest.importorskip("opentelemetry")\n    import opentelemetry\n'
)


@pytest.mark.parametrize(
    ("rel", "body", "fragment"),
    [
        pytest.param(
            "m.py",
            "import pandas\n",
            "author backend 'pandas' must be imported inside a function",
            id="module-level-pandas-ungated",
        ),
        pytest.param("m.py", "def f() -> None:\n    import pandas\n", None, id="function-level-pandas"),
        pytest.param("m.py", "def f() -> None:\n    import pyarrow\n", None, id="function-level-pyarrow-ungated"),
        pytest.param("m.py", "import pyarrow\n", "import 'pyarrow'", id="module-level-pyarrow-ungated"),
        pytest.param("extenders/m.py", "import pyarrow\n", None, id="module-level-pyarrow-gated"),
        pytest.param(
            "feature_groups/data_operations/m.py",
            "import pyarrow\n",
            "import 'pyarrow'",
            id="module-level-pyarrow-in-data-operations",
        ),
        pytest.param(
            "m.py", "def f() -> None:\n    import requests\n", "unknown import name", id="unknown-in-function"
        ),
        pytest.param("m.py", "import opentelemetry\n", "import 'opentelemetry'", id="module-level-otel-ungated"),
        pytest.param("extenders/otel.py", "import opentelemetry\n", None, id="module-level-otel-at-otel-gate"),
        pytest.param(
            "extenders/runners.py",
            "import opentelemetry\n",
            "import 'opentelemetry'",
            id="module-level-otel-at-pyarrow-gate",
        ),
        pytest.param(
            "m.py",
            "class C:\n    import pandas\n",
            "author backend 'pandas' must be imported inside a function",
            id="class-body-counts-as-module-level",
        ),
        pytest.param("m.py", "import pytest\n", None, id="declared-dependency"),
        pytest.param("m.py", _SKIP_THEN_IMPORT, None, id="importorskip-then-import"),
        pytest.param(
            "m.py",
            'import pytest\n\n\ndef f() -> None:\n    pytest.importorskip("opentelemetry.sdk")\n'
            "    from opentelemetry.sdk import trace\n",
            None,
            id="importorskip-dotted-matches-root",
        ),
        pytest.param(
            "m.py",
            "def f() -> None:\n    import opentelemetry\n",
            "import 'opentelemetry'",
            id="import-without-importorskip",
        ),
        pytest.param(
            "m.py",
            'import pytest\n\n\ndef g() -> None:\n    pytest.importorskip("opentelemetry")\n\n\n'
            "def f() -> None:\n    import opentelemetry\n",
            "import 'opentelemetry'",
            id="importorskip-in-another-function",
        ),
        pytest.param(
            "m.py",
            'import pytest\n\n\ndef f() -> None:\n    import opentelemetry\n    pytest.importorskip("opentelemetry")\n',
            "import 'opentelemetry'",
            id="importorskip-after-import",
        ),
        pytest.param(
            "m.py",
            'import pytest\n\n\ndef f() -> None:\n    pytest.importorskip("openlineage")\n    import opentelemetry\n',
            "import 'opentelemetry'",
            id="importorskip-of-another-root",
        ),
        pytest.param(
            "m.py",
            'import pytest\n\n\ndef f() -> None:\n    def g() -> None:\n        pytest.importorskip("opentelemetry")\n'
            "    import opentelemetry\n",
            "import 'opentelemetry'",
            id="importorskip-in-nested-def",
        ),
        pytest.param(
            "m.py",
            'import pytest\n\n\ndef f(c: bool) -> None:\n    if c:\n        pytest.importorskip("opentelemetry")\n'
            "    import opentelemetry\n",
            "import 'opentelemetry'",
            id="importorskip-in-branch",
        ),
        pytest.param(
            "m.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import pandas\n",
            None,
            id="type-checking-import-exempt",
        ),
        pytest.param(
            "m.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\nelse:\n    import pandas\n",
            "author backend 'pandas' must be imported inside a function",
            id="type-checking-else-branch-checked",
        ),
        pytest.param(
            "m.py",
            "TYPE_CHECKING = True\nif TYPE_CHECKING:\n    import pandas\n",
            "author backend 'pandas' must be imported inside a function",
            id="spoofed-type-checking",
        ),
        pytest.param("m.py", "import mloda_plugins.x\nfrom mloda.core import y\n", None, id="first-party-ignored"),
        pytest.param("m.py", "import json\nfrom os import path\n", None, id="stdlib-ignored"),
        pytest.param("m.py", "from . import x\n", None, id="relative-import-ignored"),
    ],
)
def test_kit_third_party_import_collector(tmp_path: Path, rel: str, body: str, fragment: str | None) -> None:
    _write(tmp_path / rel, body)
    violations = _kit_third_party_import_violations(tmp_path, import_root=tmp_path)
    if fragment is None:
        assert violations == []
    else:
        assert len(violations) == 1 and fragment in violations[0], violations


def test_kit_third_party_import_collector_skips_tests_directories(tmp_path: Path) -> None:
    _write(tmp_path / "tests" / "test_m.py", "import pandas\n")
    assert _kit_third_party_import_violations(tmp_path, import_root=tmp_path) == []


def test_license_kit_imports_only_third_party_distributions_its_job_declares() -> None:
    license_kit_root = _REPO_ROOT / verify.LICENSE_KIT_PATH
    assert license_kit_root.is_dir(), f"expected {license_kit_root} to exist"
    violations = _license_kit_third_party_import_violations(license_kit_root)
    assert violations == [], f"the license kit imports a distribution its job does not declare: {violations}"


@pytest.mark.parametrize(
    ("body", "fragments"),
    [
        pytest.param("import pytest\n", None, id="mloda-testing-dependency"),
        pytest.param("import pyarrow\n", None, id="binary-model-extra"),
        pytest.param("import cryptography\n", None, id="ed25519-extra"),
        pytest.param(
            "import openlineage\n",
            ("import 'openlineage'", "mloda-enterprise[ed25519]"),
            id="extra-outside-the-job",
        ),
        pytest.param("import requests\n", ("unknown import name",), id="unknown-name"),
        pytest.param(
            "def f() -> None:\n    import pandas\n",
            ("unknown import name",),
            id="function-level-author-backend-not-exempt",
        ),
    ],
)
def test_license_kit_third_party_import_collector(tmp_path: Path, body: str, fragments: tuple[str, ...] | None) -> None:
    _write(tmp_path / "m.py", body)
    violations = _license_kit_third_party_import_violations(tmp_path, import_root=tmp_path)
    if fragments is None:
        assert violations == []
    else:
        assert len(violations) == 1 and all(f in violations[0] for f in fragments), violations


def test_license_kit_declared_set_follows_the_job_requirements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify, "LICENSE_KIT_REQUIREMENTS", ("mloda-testing[binary-model]",))
    _write(tmp_path / "m.py", "import cryptography\n")
    violations = _license_kit_third_party_import_violations(tmp_path, import_root=tmp_path)
    assert len(violations) == 1 and "import 'cryptography'" in violations[0], violations
    real = _license_kit_third_party_import_violations(_REPO_ROOT / verify.LICENSE_KIT_PATH)
    assert any("via mloda/enterprise/licensing/license_token.py" in v and "import 'cryptography'" in v for v in real), (
        real
    )


_LICENSE_KIT_REL = "mloda/enterprise/licensing/testing"
_TESTING_KIT_REL = "mloda/testing"
_OL = "import openlineage\n"
_TYPE_CHECKING_OL = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import openlineage\n"


@pytest.mark.parametrize(
    ("kit_rel", "files", "fragments"),
    [
        pytest.param(
            _LICENSE_KIT_REL,
            {"m.py": "import mloda.enterprise.x\n", "../../x.py": _OL},
            ("m.py: via mloda/enterprise/x.py:1", "import 'openlineage'"),
            id="module-level-first-party-import-is-followed",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {"m.py": "import mloda.enterprise.x\n", "../../x.py": "import cryptography\n"},
            None,
            id="followed-import-declared",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {
                "m.py": "import mloda.enterprise.a\n",
                "../../a.py": "import mloda.enterprise.b\n",
                "../../b.py": _OL,
            },
            ("m.py: via mloda/enterprise/b.py:1", "import 'openlineage'"),
            id="transitive-chain-is-followed",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {"m.py": "import mloda.enterprise.x\n", "../../x.py": f"def f() -> None:\n    {_OL}"},
            None,
            id="followed-function-level-import-clean",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {"m.py": "def f() -> None:\n    import mloda.enterprise.x\n", "../../x.py": _OL},
            None,
            id="function-level-first-party-import-not-followed",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {
                "m.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import mloda.enterprise.x\n",
                "../../x.py": _OL,
            },
            None,
            id="type-checking-first-party-import-not-followed",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {"m.py": "from mloda.enterprise import x\n", "../../x.py": _OL},
            ("m.py: via mloda/enterprise/x.py:1", "import 'openlineage'"),
            id="from-package-import-submodule-is-followed",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {"m.py": "import mloda.enterprise.x\n", "../../x.py": "", "../../__init__.py": _OL},
            ("m.py: via mloda/enterprise/__init__.py:1", "import 'openlineage'"),
            id="parent-package-init-is-followed",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {
                "m.py": "import mloda.enterprise.x\n",
                "../../x.py": "from .mod import y\n",
                "../../mod.py": _OL,
            },
            ("m.py: via mloda/enterprise/mod.py:1", "import 'openlineage'"),
            id="single-dot-relative-import-is-followed",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {
                "m.py": "import mloda.enterprise.sub.x\n",
                "../../sub/x.py": "from ..mod import y\n",
                "../../mod.py": _OL,
            },
            ("m.py: via mloda/enterprise/mod.py:1", "import 'openlineage'"),
            id="double-dot-relative-import-is-followed",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {
                "m.py": "import mloda.enterprise.a\n",
                "../../a.py": "import cryptography\nimport mloda.enterprise.b\n",
                "../../b.py": "import cryptography\nimport mloda.enterprise.a\n",
            },
            None,
            id="import-cycle-terminates",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {
                "m.py": "import mloda.enterprise.x\n",
                "../../x.py": "from ...y import z\n",
                "../../../../y.py": _OL,
            },
            None,
            id="relative-import-above-the-root-is-skipped",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {"m.py": "", "../__init__.py": _OL},
            ("m.py: via mloda/enterprise/licensing/__init__.py:1", "import 'openlineage'"),
            id="kit-module-ancestor-package-init-is-followed",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {"m.py": "import mloda.enterprise.x\n", "../../x.py": "from . import mod\n", "../../mod.py": _OL},
            ("m.py: via mloda/enterprise/mod.py:1", "import 'openlineage'"),
            id="bare-relative-from-import-is-followed",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {"m.py": "import mloda.enterprise.sub\n", "../../sub/__init__.py": _OL},
            ("m.py: via mloda/enterprise/sub/__init__.py:1", "import 'openlineage'"),
            id="package-init-as-the-import-target-is-followed",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {
                "m.py": "import mloda.enterprise.x\n",
                "../../x.py": "def f() -> None:\n    import mloda.enterprise.y\n",
                "../../y.py": _OL,
            },
            None,
            id="followed-function-level-first-party-import-not-followed",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {"m.py": "import mloda.enterprise.x\n", "../../x.py": _TYPE_CHECKING_OL},
            None,
            id="followed-type-checking-import-clean",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {"m.py": "from mloda.core import y\nimport mloda.steward\n"},
            None,
            id="first-party-module-absent-under-import-root-skipped",
        ),
        pytest.param(
            _LICENSE_KIT_REL,
            {"m.py": "import mloda.enterprise.x\n", "../../x.py": "import requests\n"},
            ("m.py: via mloda/enterprise/x.py:1", "unknown import name 'requests'"),
            id="followed-unknown-import-name",
        ),
        pytest.param(
            _TESTING_KIT_REL,
            {"m.py": "import mloda.community.x\n", "../community/x.py": "import pandas\n"},
            ("m.py: via mloda/community/x.py:1", "author backend 'pandas' must be imported inside a function"),
            id="followed-author-backend-at-module-level",
        ),
        pytest.param(
            _TESTING_KIT_REL,
            {"m.py": "import mloda.testing.extenders.otel\n", "extenders/otel.py": "import opentelemetry\n"},
            ("m.py: via mloda/testing/extenders/otel.py:1", "import 'opentelemetry'"),
            id="followed-kit-module-checked-against-the-importers-gate",
        ),
    ],
)
def test_kit_third_party_import_follower(
    tmp_path: Path, kit_rel: str, files: dict[str, str], fragments: tuple[str, ...] | None
) -> None:
    kit_dir = tmp_path / kit_rel
    for rel, body in files.items():
        _write((kit_dir / rel).resolve(), body)
    if kit_rel == _LICENSE_KIT_REL:
        violations = _license_kit_third_party_import_violations(kit_dir, import_root=tmp_path)
    else:
        violations = _kit_third_party_import_violations(kit_dir, import_root=tmp_path)
    if fragments is None:
        assert violations == []
    else:
        assert len(violations) == 1 and all(f in violations[0] for f in fragments), violations
