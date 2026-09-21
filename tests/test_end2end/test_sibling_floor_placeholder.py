"""Sibling (in-repo) dependency floors in config/packages.toml. A dependency on another package of
the same ``packages`` table must be written ``"<sibling>>={version}"``, which the generator expands
to ``shared["project"]["version"]``; a hand-written numeric floor on a sibling must be rejected.

The generator lives at ``scripts/generate_pyproject.py`` (a script, not an installed package), so
it is loaded here by file path.
"""

from __future__ import annotations

import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"

gen = load_script("generate_pyproject", _GEN_PATH)

# _NAME_RE / _normalize are a deliberate independent oracle of DEP_NAME_RE / normalize_package_name.
# The distribution name a PEP 508 dependency string starts with; placeholders like {core_dependency} match none.
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

_DEP = "mloda-community-data-operations"
_LEAF = "mloda-community-aggregation"
_DEPENDENT = "mloda-synthetic-dependent"
_TOP_LEVEL_PATH = "mloda/synthetic_dependent"
_WORKSPACE = {"workspace": True}


def _normalize(name: str) -> str:
    """PEP 503 normal form: lowercase, runs of '-', '_', '.' collapsed to '-'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _synthetic_packages(leaf_dependency: str) -> dict[str, dict[str, Any]]:
    """A minimal two-package table: a base plus a leaf depending on it, shaped like config/packages.toml."""
    return {
        _DEP: {
            "description": "base",
            "dependencies": ["{core_dependency}"],
            "path": "mloda/community/feature_groups/data_operations",
            "published": True,
        },
        _LEAF: {
            "description": "leaf",
            "dependencies": [leaf_dependency],
            "path": "mloda/community/feature_groups/data_operations/aggregation",
            "published": True,
        },
    }


def _synthetic_packages_with_base_extra(optional_dependencies: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    """Like ``_synthetic_packages``, but the base package carries the given ``optional_dependencies`` table."""
    packages = _synthetic_packages(f"{_DEP}>={{version}}")
    packages[_DEP]["optional_dependencies"] = optional_dependencies
    return packages


def _generated_dependencies(pkg_name: str, packages: dict[str, dict[str, Any]], shared: dict[str, Any]) -> list[str]:
    """Parsed ``[project].dependencies`` of the pyproject the generator produces for ``pkg_name``."""
    content = gen.generate_pyproject(pkg_name, packages[pkg_name], shared, packages)
    deps: list[str] = tomllib.loads(content)["project"]["dependencies"]
    return deps


def _generated_optional_dependencies(
    pkg_name: str, packages: dict[str, dict[str, Any]], shared: dict[str, Any]
) -> dict[str, list[str]]:
    """Parsed ``[project.optional-dependencies]`` of the pyproject the generator produces for ``pkg_name``."""
    content = gen.generate_pyproject(pkg_name, packages[pkg_name], shared, packages)
    opts: dict[str, list[str]] = tomllib.loads(content).get("project", {}).get("optional-dependencies", {})
    return opts


def _generated_uv_sources(pkg_name: str, packages: dict[str, dict[str, Any]], shared: dict[str, Any]) -> dict[str, Any]:
    """Parsed ``[tool.uv.sources]`` of the generated pyproject ({} when absent); a duplicate header is invalid TOML."""
    content = gen.generate_pyproject(pkg_name, packages[pkg_name], shared, packages)
    sources: dict[str, Any] = tomllib.loads(content).get("tool", {}).get("uv", {}).get("sources", {})
    return sources


def _synthetic_packages_with_dependent(path: str, dependencies: list[str], **extra: Any) -> dict[str, dict[str, Any]]:
    """``_synthetic_packages`` plus a ``_DEPENDENT`` package at ``path`` with the given runtime ``dependencies``."""
    packages = _synthetic_packages(f"{_DEP}>={{version}}")
    packages[_DEPENDENT] = {
        "description": "dependent",
        "dependencies": dependencies,
        "path": path,
        "published": True,
        **extra,
    }
    return packages


def test_version_placeholder_expands_to_shared_version() -> None:
    """A leaf writing ``{version}`` gets the shared project version substituted, verbatim floor operator kept."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages(f"{_DEP}>={{version}}")

    deps = _generated_dependencies(_LEAF, packages, shared)

    expected = f"{_DEP}>={shared['project']['version']}"
    assert deps == [expected], f"expected exactly [{expected!r}], got {deps!r}"
    assert not any("{version}" in dep for dep in deps), f"literal {{version}} placeholder leaked into {deps!r}"


@pytest.mark.parametrize(
    "dependency",
    [
        f"{_DEP}>=0.4.4",
        f"{_DEP}==0.4.4",
        _DEP,
        "mloda_community_data_operations>=0.4.4",
        "MLODA-Community-Data-Operations>=0.4.4",
        f'{_DEP}>=0.4.4; python_version>="3.11"',
    ],
    ids=["gte-pin", "exact-pin", "bare", "underscores", "mixed-case", "env-marker"],
)
def test_hand_pinned_sibling_dependency_is_rejected(dependency: str) -> None:
    """A sibling dependency written without the {version} placeholder must raise ValueError naming
    both the leaf package and the offending dependency string."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages(dependency)

    with pytest.raises(ValueError) as exc_info:
        gen.generate_pyproject(_LEAF, packages[_LEAF], shared, packages)

    message = str(exc_info.value)
    assert _LEAF in message, f"error message must name the leaf package {_LEAF!r}, got: {message}"
    assert dependency in message, f"error message must name the offending dependency {dependency!r}, got: {message}"


@pytest.mark.parametrize(
    "dependency",
    [
        f"{_DEP}>=0.4.0,<={{version}}",
        f"{_DEP}>=0.4.4,<{{version}}",
        f"{_DEP}~={{version}}",
        f"{_DEP}=={{version}}",
        f'{_DEP}>=0.4.4; python_version>="{{version}}"',
    ],
    ids=["extra-upper-bound", "extra-lower-bound", "tilde-operator", "exact-operator", "placeholder-in-marker-only"],
)
def test_malformed_version_placeholder_specifier_is_rejected(dependency: str) -> None:
    """{version} alone isn't enough: the specifier (marker stripped) must be exactly '<name>[extras]>={version}'."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages(dependency)

    with pytest.raises(ValueError) as exc_info:
        gen.generate_pyproject(_LEAF, packages[_LEAF], shared, packages)

    message = str(exc_info.value)
    assert _LEAF in message, f"error message must name the leaf package {_LEAF!r}, got: {message}"
    assert dependency in message, f"error message must name the offending dependency {dependency!r}, got: {message}"


@pytest.mark.parametrize(
    "dependency",
    [
        "pytest{version}",
        "{version}",
        "mloda-comunity-data-operations>={version}",
    ],
    ids=["placeholder-in-external-name", "placeholder-alone", "typo-sibling-name"],
)
def test_version_placeholder_outside_a_sibling_floor_is_rejected(dependency: str) -> None:
    """{version} is only valid inside a sibling floor '<sibling>[extras]>={version}'; used anywhere
    else (an unrelated external name, standalone, or a misspelled sibling name) it must raise ValueError."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages(dependency)

    with pytest.raises(ValueError) as exc_info:
        gen.generate_pyproject(_LEAF, packages[_LEAF], shared, packages)

    message = str(exc_info.value)
    assert _LEAF in message, f"error message must name the leaf package {_LEAF!r}, got: {message}"
    assert dependency in message, f"error message must name the offending dependency {dependency!r}, got: {message}"


def test_version_placeholder_with_extras_is_accepted() -> None:
    """A sibling requirement may carry extras before the floor operator: '<name>[extras]>={version}'."""
    shared, _packages_config = gen.load_configs()
    dependency = f"{_DEP}[all]>={{version}}"
    packages = _synthetic_packages(dependency)

    deps = _generated_dependencies(_LEAF, packages, shared)

    expected = f"{_DEP}[all]>={shared['project']['version']}"
    assert deps == [expected], f"expected exactly [{expected!r}], got {deps!r}"


def test_optional_dependency_bare_sibling_is_unchanged() -> None:
    """A sibling named without any specifier in an extra needs no {version} placeholder and passes through."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages_with_base_extra({"all": [_LEAF]})

    opts = _generated_optional_dependencies(_DEP, packages, shared)

    assert opts.get("all") == [_LEAF], f"expected the 'all' extra to list {_LEAF!r} unchanged, got {opts.get('all')!r}"


def test_optional_dependency_version_placeholder_expands() -> None:
    """A sibling floor written in an extra must expand the {version} placeholder, same as a plain dependency."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages_with_base_extra({"all": [f"{_LEAF}>={{version}}"]})

    opts = _generated_optional_dependencies(_DEP, packages, shared)

    expected = f"{_LEAF}>={shared['project']['version']}"
    assert opts.get("all") == [expected], (
        f"expected the 'all' extra to expand to [{expected!r}], got {opts.get('all')!r}"
    )
    assert not any("{version}" in dep for deps in opts.values() for dep in deps), (
        f"literal {{version}} placeholder leaked into optional-dependencies {opts!r}"
    )


@pytest.mark.parametrize("dependency", [f"{_LEAF}>=0.4.0", f"{_LEAF}==0.4.0"], ids=["gte-pin", "exact-pin"])
def test_hand_pinned_optional_sibling_dependency_is_rejected(dependency: str) -> None:
    """A sibling floor hand-written in an extra must be rejected the same as one in plain dependencies."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages_with_base_extra({"all": [dependency]})

    with pytest.raises(ValueError) as exc_info:
        gen.generate_pyproject(_DEP, packages[_DEP], shared, packages)

    message = str(exc_info.value)
    assert _DEP in message, f"error message must name the package {_DEP!r}, got: {message}"
    assert dependency in message, f"error message must name the offending entry {dependency!r}, got: {message}"


@pytest.mark.parametrize(
    "dependency",
    [
        "pytest{version}",
        "{version}",
        "mloda-comunity-data-operations>={version}",
    ],
    ids=["placeholder-in-external-name", "placeholder-alone", "typo-sibling-name"],
)
def test_optional_version_placeholder_outside_a_sibling_floor_is_rejected(dependency: str) -> None:
    """Same guard as test_version_placeholder_outside_a_sibling_floor_is_rejected, applied to an
    optional-dependencies group instead of plain dependencies."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages_with_base_extra({"all": [dependency]})

    with pytest.raises(ValueError) as exc_info:
        gen.generate_pyproject(_DEP, packages[_DEP], shared, packages)

    message = str(exc_info.value)
    assert _DEP in message, f"error message must name the package {_DEP!r}, got: {message}"
    assert dependency in message, f"error message must name the offending dependency {dependency!r}, got: {message}"


def test_optional_dependency_non_sibling_entry_is_untouched() -> None:
    """An extra naming no sibling package must pass through unchanged, same as an external plain dependency."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages_with_base_extra({"pandas": ["pandas>=2.2"]})

    opts = _generated_optional_dependencies(_DEP, packages, shared)

    assert opts.get("pandas") == ["pandas>=2.2"], f"expected 'pandas' extra unchanged, got {opts.get('pandas')!r}"


def test_optional_dependency_published_children_combination_does_not_raise() -> None:
    """The new sibling guard must not reject the pre-existing {published_children} expansion."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages_with_base_extra({"all": ["{published_children}"]})

    gen.generate_pyproject(_DEP, packages[_DEP], shared, packages)  # must not raise


def test_core_dependency_placeholder_does_not_silently_absorb_the_registry_version() -> None:
    """If {core_dependency} expanded before {version}, a literal "{version}" inside it would be silently absorbed."""
    shared, _packages_config = gen.load_configs()
    shared = deepcopy(shared)
    shared["defaults"]["core_dependency"] = "mloda>={version}"
    pkg_name = _DEP
    packages = {
        pkg_name: {
            "description": "base",
            "dependencies": ["{core_dependency}"],
            "path": "mloda/community/feature_groups/data_operations",
            "published": True,
        }
    }

    with pytest.raises(ValueError) as exc_info:
        gen.generate_pyproject(pkg_name, packages[pkg_name], shared, packages)

    assert pkg_name in str(exc_info.value), f"error message must name the package {pkg_name!r}, got: {exc_info.value}"


def test_core_dependency_expanding_to_a_hand_pinned_sibling_is_rejected() -> None:
    """Sibling-spelling validation must run on the {core_dependency}-expanded string: a shared
    core_dependency default that hand-pins a sibling floor must be rejected, not emitted verbatim."""
    shared, _packages_config = gen.load_configs()
    shared = deepcopy(shared)
    shared["defaults"]["core_dependency"] = f"{_DEP}>=0.0.1"
    packages = _synthetic_packages("{core_dependency}")

    with pytest.raises(ValueError) as exc_info:
        gen.generate_pyproject(_LEAF, packages[_LEAF], shared, packages)

    message = str(exc_info.value)
    assert _LEAF in message, f"error message must name the leaf package {_LEAF!r}, got: {message}"


def test_typo_placeholder_is_rejected_as_a_hand_written_floor() -> None:
    """{versoin} is a typo, not the {version} placeholder, so it must be rejected like any hand-written floor."""
    shared, _packages_config = gen.load_configs()
    dependency = f"{_DEP}>={{versoin}}"
    packages = _synthetic_packages(dependency)

    with pytest.raises(ValueError) as exc_info:
        gen.generate_pyproject(_LEAF, packages[_LEAF], shared, packages)

    message = str(exc_info.value)
    assert "{version}" in message or dependency in message, (
        f"error message must mention the {{version}} placeholder or the offending dependency, got: {message}"
    )


@pytest.mark.parametrize(
    "dependency",
    [f"{_DEP} >= {{version}}", f'{_DEP}>={{version}}; python_version>="3.11"'],
    ids=["whitespace", "env-marker"],
)
def test_placeholder_accepted_with_pep508_variants(dependency: str) -> None:
    """PEP 508 whitespace and environment markers around the {version} placeholder must not trip the guard."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages(dependency)

    deps = _generated_dependencies(_LEAF, packages, shared)

    assert len(deps) == 1, f"expected exactly one dependency, got {deps!r}"
    assert shared["project"]["version"] in deps[0], f"expected the shared version substituted into {deps[0]!r}"
    assert "{version}" not in deps[0], f"literal {{version}} placeholder leaked into {deps[0]!r}"


@pytest.mark.parametrize(
    "dependency",
    ["{core_dependency}", "mloda>=0.10.0,<0.11.0", "pytest>=9.0.3", "opentelemetry-api>=1.30,<2"],
    ids=["core-placeholder", "mloda-core-pin", "pytest", "opentelemetry-api"],
)
def test_external_dependency_is_untouched(dependency: str) -> None:
    """A dependency naming no sibling in the packages table must neither raise nor be rewritten,
    except for the pre-existing {core_dependency} expansion."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages(dependency)

    deps = _generated_dependencies(_LEAF, packages, shared)

    assert len(deps) == 1, f"expected exactly one dependency, got {deps!r}"
    if dependency == "{core_dependency}":
        core_dep = shared["defaults"]["core_dependency"]
        assert deps[0] == core_dep, f"expected {{core_dependency}} to expand to {core_dep!r}, got {deps[0]!r}"
    else:
        assert deps[0] == dependency, f"external dependency must pass through unchanged, got {deps[0]!r}"


def test_real_sibling_dependencies_use_the_version_placeholder() -> None:
    """Every config/packages.toml dependency naming a sibling package must carry the {version}
    placeholder, so its floor stays generator-derived rather than hand-written."""
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    canonical = {_normalize(name) for name in packages}

    offending = []
    for pkg_name, cfg in packages.items():
        for dep in cfg.get("dependencies", []):
            requirement = dep.split(";", 1)[0]
            match = _NAME_RE.match(requirement)
            if match is None:
                continue
            if _normalize(match.group(1)) in canonical and "{version}" not in dep:
                offending.append(f"{pkg_name}: {dep!r}")

    assert not offending, (
        "every sibling dependency must use the {version} placeholder so its floor is generator-derived, "
        "but found hand-pinned floors:\n" + "\n".join(offending)
    )

    for pkg_name, cfg in packages.items():
        content = gen.generate_pyproject(pkg_name, cfg, shared, packages)
        assert "{version}" not in content, f"{pkg_name}: generated pyproject still contains a literal {{version}}"


def test_generate_raises_when_version_missing_for_placeholder_dependency() -> None:
    """Mirrors the {core_dependency} missing-key guard: a {version} placeholder dependency with no
    [project].version in shared config must raise ValueError, not fail some other way.
    """
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages(f"{_DEP}>={{version}}")
    del shared["project"]["version"]

    with pytest.raises(ValueError):
        gen.generate_pyproject(_LEAF, packages[_LEAF], shared, packages)


@pytest.mark.parametrize("entry_point_bundle", [True, False], ids=["bundle", "plain"])
@pytest.mark.parametrize(
    "dependency",
    [f"{_DEP}>={{version}}", f"{_DEP}[all]>={{version}}", f'{_DEP}>={{version}}; python_version>="3.11"'],
    ids=["floor", "extras", "env-marker"],
)
def test_top_level_package_gets_a_workspace_source_for_a_sibling_dependency(
    dependency: str, entry_point_bundle: bool
) -> None:
    """A sibling in a top-level package's runtime dependencies gets a ``{ workspace = true }`` source (uv needs it)."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages_with_dependent(
        _TOP_LEVEL_PATH, ["{core_dependency}", dependency], entry_point_bundle=entry_point_bundle
    )

    sources = _generated_uv_sources(_DEPENDENT, packages, shared)

    assert sources.get(_DEP) == _WORKSPACE, f"expected sources[{_DEP!r}] == {_WORKSPACE!r}, got {sources!r}"


@pytest.mark.parametrize(
    "path",
    ["mloda/community/synthetic_dependent", "mloda/community/feature_groups/synthetic_dependent"],
    ids=["depth-3", "depth-4"],
)
def test_nested_package_gets_no_workspace_source_for_a_sibling_dependency(path: str) -> None:
    """Nested members (path depth > 2) resolve a sibling without a source, and uv rejects one there."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages_with_dependent(path, ["{core_dependency}", f"{_DEP}>={{version}}"])

    sources = _generated_uv_sources(_DEPENDENT, packages, shared)

    assert sources == {}, f"a nested package must emit no [tool.uv.sources] entries, got {sources!r}"


def test_top_level_package_with_default_dev_deps_lists_mloda_testing_and_its_sibling_in_one_table() -> None:
    """Default dev deps keep the ``mloda-testing`` source beside the sibling sources, which stay sorted."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages_with_dependent(
        _TOP_LEVEL_PATH, ["{core_dependency}", f"{_DEP}>={{version}}", f"{_LEAF}>={{version}}"]
    )

    sources = _generated_uv_sources(_DEPENDENT, packages, shared)

    expected = {"mloda-testing": _WORKSPACE, _DEP: _WORKSPACE, _LEAF: _WORKSPACE}
    assert sources == expected, f"expected exactly {expected!r}, got {sources!r}"
    siblings = [name for name in sources if name != "mloda-testing"]
    assert siblings == sorted(siblings), f"sibling sources must be in sorted order, got {siblings!r}"


def test_top_level_package_with_default_dev_deps_and_no_sibling_keeps_only_mloda_testing() -> None:
    """External and core dependencies get no source, so only ``mloda-testing`` is listed."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_packages_with_dependent(_TOP_LEVEL_PATH, ["{core_dependency}", "pytest>=9.0.3"])

    sources = _generated_uv_sources(_DEPENDENT, packages, shared)

    assert sources == {"mloda-testing": _WORKSPACE}, f"expected only the mloda-testing source, got {sources!r}"


def test_real_bundles_workspace_sources_follow_their_sibling_dependencies() -> None:
    """Both bundles need a source for the shared extenders package or uv lock fails."""
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    sibling = "mloda-community-extenders-shared"

    enterprise_sources = _generated_uv_sources("mloda-enterprise", packages, shared)
    community_sources = _generated_uv_sources("mloda-community", packages, shared)

    assert enterprise_sources.get(sibling) == _WORKSPACE, (
        f"expected mloda-enterprise sources[{sibling!r}] == {_WORKSPACE!r}, got {enterprise_sources!r}"
    )
    assert community_sources == {sibling: _WORKSPACE}, (
        f"mloda-community depends on {sibling!r}, expected exactly that workspace source, got {community_sources!r}"
    )


# A synthetic bundle over the real mloda/community tree: _LEAF (aggregation) and _DEP (data_operations)
# are configured packages nested under it, so the generator's filesystem discovery has real modules to list.
_BUNDLE_PATH = "mloda/community"
_LEAF_DOTTED = "mloda.community.feature_groups.data_operations.aggregation"
_DEP_DOTTED = "mloda.community.feature_groups.data_operations"


def _synthetic_bundle(dependencies: list[str]) -> dict[str, dict[str, Any]]:
    """``_synthetic_packages`` plus a ``_DEPENDENT`` entry-point bundle at ``mloda/community``."""
    return _synthetic_packages_with_dependent(_BUNDLE_PATH, dependencies, entry_point_bundle=True)


def _generated_wheel_packages(pkg_name: str, packages: dict[str, dict[str, Any]], shared: dict[str, Any]) -> list[str]:
    """Parsed ``[tool.setuptools] packages`` of the generated pyproject."""
    content = gen.generate_pyproject(pkg_name, packages[pkg_name], shared, packages)
    listed: list[str] = tomllib.loads(content)["tool"]["setuptools"]["packages"]
    return listed


def test_bundle_wheel_excludes_a_nested_package_it_depends_on() -> None:
    """A bundle naming a nested package in its own dependencies leaves that package's files to its own wheel."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_bundle(["{core_dependency}", f"{_LEAF}>={{version}}"])

    listed = _generated_wheel_packages(_DEPENDENT, packages, shared)

    leaked = [entry for entry in listed if entry == _LEAF_DOTTED or entry.startswith(_LEAF_DOTTED + ".")]
    assert leaked == [], f"the bundle wheel must not ship the nested dependency {_LEAF!r}, but lists {leaked!r}"


def test_bundle_wheel_still_ships_a_nested_package_it_does_not_depend_on() -> None:
    """The exclusion is scoped to the bundle's own dependencies: another nested package is still shipped."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_bundle(["{core_dependency}", f"{_LEAF}>={{version}}"])

    listed = _generated_wheel_packages(_DEPENDENT, packages, shared)

    assert _DEP_DOTTED in listed, f"the bundle wheel must still ship non-dependency {_DEP!r}, got {listed!r}"
    # Control: with no dependency on the leaf, the bundle ships it, so the exclusion above is dependency-driven.
    control = _generated_wheel_packages(_DEPENDENT, _synthetic_bundle(["{core_dependency}"]), shared)
    assert _LEAF_DOTTED in control, f"without the dependency the bundle must ship {_LEAF!r}, got {control!r}"


def test_bundle_dependency_on_an_unpublished_nested_package_is_rejected() -> None:
    """The nested dependency owns the files, so it must be a published distribution or nothing ships them."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_bundle(["{core_dependency}", f"{_LEAF}>={{version}}"])
    packages[_LEAF]["published"] = False

    with pytest.raises(ValueError) as exc_info:
        gen.generate_pyproject(_DEPENDENT, packages[_DEPENDENT], shared, packages)

    message = str(exc_info.value)
    assert _DEPENDENT in message, f"error message must name the bundle {_DEPENDENT!r}, got: {message}"
    assert _LEAF in message, f"error message must name the nested dependency {_LEAF!r}, got: {message}"


def test_bundle_dependency_on_a_nested_package_with_entry_point_groups_is_rejected() -> None:
    """A nested dependency that declares entry_point_groups would be registered twice (its wheel and the bundle)."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_bundle(["{core_dependency}", f"{_LEAF}>={{version}}"])
    packages[_LEAF]["entry_point_groups"] = ["mloda.feature_groups"]

    with pytest.raises(ValueError) as exc_info:
        gen.generate_pyproject(_DEPENDENT, packages[_DEPENDENT], shared, packages)

    message = str(exc_info.value)
    assert _DEPENDENT in message, f"error message must name the bundle {_DEPENDENT!r}, got: {message}"
    assert _LEAF in message, f"error message must name the nested dependency {_LEAF!r}, got: {message}"


def test_bundle_dependency_on_a_nested_package_with_its_own_nested_packages_is_rejected() -> None:
    """A nested dependency that itself has packages nested under it would leave those without an owner."""
    shared, _packages_config = gen.load_configs()
    packages = _synthetic_bundle(["{core_dependency}", f"{_DEP}>={{version}}"])

    with pytest.raises(ValueError) as exc_info:
        gen.generate_pyproject(_DEPENDENT, packages[_DEPENDENT], shared, packages)

    message = str(exc_info.value)
    assert _DEPENDENT in message, f"error message must name the bundle {_DEPENDENT!r}, got: {message}"
    assert _DEP in message, f"error message must name the nested dependency {_DEP!r}, got: {message}"
