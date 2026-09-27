"""mloda-community must not hard-require an optional plugin's own third-party dependency
(openlineage-python, opentelemetry-api, ...); each leaf ships as its own bundle extra, exactly pinned
`mloda-community-<leaf>=={version}`, itself the sole owner of the third-party pin in its own
`dependencies` (mloda-community-otel/mloda-community-openlineage are published, nested siblings the
bundle owns through the extra). mloda-enterprise[openlineage] does the same for the first-party sibling
mloda-community-openlineage, whose floor is spelled `{version}` in the extra and equally in the leaf's
dev entry.
"""

from __future__ import annotations

import importlib
import logging
import re
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest
from mloda.user import PluginLoader

from mloda.testing.import_isolation import block_root, evict_package, evict_root
from tests.script_loader import load_script

# Logger name PluginLoader.load_entry_points() itself logs WARNINGs under when it skips an entry point
# for a missing optional dependency (see mloda.core.abstract_plugins.plugin_loader.plugin_loader).
_PLUGIN_LOADER_LOGGER = "mloda.core.abstract_plugins.plugin_loader.plugin_loader"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"
_COMMUNITY_PYPROJECT = _REPO_ROOT / "mloda" / "community" / "pyproject.toml"
_ENTERPRISE_PYPROJECT = _REPO_ROOT / "mloda" / "enterprise" / "pyproject.toml"
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"

# A first-party sibling floor: the {version} placeholder, equal to the leaf's dev entry.
_LINEAGE_EMITTER_FLOOR = "mloda-community-openlineage>={version}"

# One row per mloda-community extra-only dependency: (extra name, distribution name, leaf
# package name, import root, exposed extender names). Extend this when the bundle gains another
# extra-only plugin dependency.
_ROWS: list[tuple[str, str, str, str, list[str]]] = [
    ("openlineage", "openlineage-python", "mloda-community-openlineage", "openlineage", ["OpenLineageExtender"]),
    ("otel", "opentelemetry-api", "mloda-community-otel", "opentelemetry", ["OtelExtender"]),
]

# Distribution name -> the import root that provides it, derived from _ROWS. Extend _ROWS above (not
# here) when a bundle covers a new nested leaf's dependency only through an extra.
_IMPORT_ROOT_OF_DISTRIBUTION = {distribution_name: root for _, distribution_name, _, root, _ in _ROWS}

# Leaf package name -> attribute name(s) its own __init__ must not expose when its extra-only
# dependency is absent, derived from _ROWS.
_EXPOSED_EXTENDER_NAMES: dict[str, list[str]] = {leaf_name: exposed for _, _, leaf_name, _, exposed in _ROWS}

gen = load_script("generate_pyproject", _GEN_PATH)


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _dep_name(spec: str) -> str:
    """Extract the bare package name from a PEP 508 requirement string."""
    return re.split(r"[<>=!~;\s\[(@]", spec.strip(), maxsplit=1)[0]


def _packages() -> dict[str, dict[str, Any]]:
    packages: dict[str, dict[str, Any]] = _load_toml(_PACKAGES_CONFIG)["packages"]
    return packages


def _external_dependency_names(deps: list[str], packages: dict[str, dict[str, Any]]) -> set[str]:
    """Bare distribution names among ``deps``, dropping placeholders (e.g. ``{core_dependency}``) and
    names that resolve to a configured package."""
    names = set()
    for dep in deps:
        if dep.strip().startswith("{"):
            continue
        name = _dep_name(dep)
        if name in packages or name == "mloda":
            continue
        names.add(name)
    return names


def _bundle_owned_names(bundle_name: str, packages: dict[str, dict[str, Any]]) -> set[str]:
    """Configured packages a bundle owns: named in its own dependencies or a non-dev extra."""
    bundle_cfg = packages[bundle_name]
    raw = list(bundle_cfg.get("dependencies", []))
    for extra_name, deps in bundle_cfg.get("optional_dependencies", {}).items():
        if extra_name != "dev":
            raw.extend(deps)
    return {_dep_name(dep) for dep in raw if _dep_name(dep) in packages}


@pytest.mark.parametrize("extra_name, distribution_name, leaf_name, root, exposed", _ROWS)
def test_mloda_community_dependencies_do_not_pin_extra_only_distribution(
    extra_name: str, distribution_name: str, leaf_name: str, root: str, exposed: list[str]
) -> None:
    community = _packages()["mloda-community"]
    hard_deps = community.get("dependencies", [])
    offending = [dep for dep in hard_deps if _dep_name(dep) in (distribution_name, leaf_name)]
    assert not offending, (
        f"packages.mloda-community.dependencies must not pin {distribution_name} or {leaf_name} directly, "
        f"found {offending!r}; {leaf_name} belongs in optional_dependencies.{extra_name} instead, and "
        f"{distribution_name} belongs in {leaf_name}'s own dependencies"
    )


@pytest.mark.parametrize("extra_name, distribution_name, leaf_name, root, exposed", _ROWS)
def test_mloda_community_declares_extra_matching_pin_source(
    extra_name: str, distribution_name: str, leaf_name: str, root: str, exposed: list[str]
) -> None:
    """The bundle extra owns the leaf exactly (`{leaf}=={version}`); the leaf's own dependencies own the
    third-party pin, the extra's only route to it."""
    packages = _packages()
    community_extra = packages["mloda-community"].get("optional_dependencies", {}).get(extra_name)
    assert community_extra is not None, f"packages.mloda-community.optional_dependencies.{extra_name} must be declared"

    expected_extra = [f"{leaf_name}=={{version}}"]
    assert community_extra == expected_extra, (
        f"packages.mloda-community.optional_dependencies.{extra_name} must be exactly "
        f"{expected_extra!r}, got {community_extra!r}"
    )

    leaf_deps = packages[leaf_name]["dependencies"]
    leaf_matching = [dep for dep in leaf_deps if _dep_name(dep) == distribution_name]
    assert len(leaf_matching) == 1, (
        f"packages.{leaf_name}.dependencies must declare exactly one {distribution_name} entry, got {leaf_deps!r}"
    )


def _bundle_extra_leaf_dev_pairs(packages: dict[str, dict[str, Any]]) -> list[tuple[str, str, str, str, str]]:
    """``(bundle, extra, extra spec, leaf, dev spec)`` for each bundle extra entry a nested leaf also lists in its
    ``dev`` extra."""
    pairs: list[tuple[str, str, str, str, str]] = []
    for bundle_name, bundle_cfg in packages.items():
        if bundle_cfg.get("entry_point_bundle") is not True:
            continue
        prefix = bundle_cfg["path"] + "/"

        for extra_name, extra_deps in bundle_cfg.get("optional_dependencies", {}).items():
            if extra_name in ("all", "dev"):
                continue
            for extra_spec in extra_deps:
                for leaf_name, leaf_cfg in packages.items():
                    if not leaf_cfg["path"].startswith(prefix):
                        continue
                    for dev_spec in leaf_cfg.get("optional_dependencies", {}).get("dev", []):
                        if _dep_name(dev_spec) == _dep_name(extra_spec):
                            pairs.append((bundle_name, extra_name, extra_spec, leaf_name, dev_spec))
    return pairs


def test_bundle_extra_floor_matches_leaf_dev_entry() -> None:
    """A bundle extra's floor for an external dependency must equal the same dependency's entry in the
    ``dev`` extra of every nested leaf that lists it, so the two places cannot drift apart."""
    packages = _packages()
    checked = 0

    for bundle_name, extra_name, extra_spec, leaf_name, dev_spec in _bundle_extra_leaf_dev_pairs(packages):
        if not _external_dependency_names([extra_spec], packages):
            continue
        checked += 1
        assert extra_spec == dev_spec, (
            f"the {_dep_name(extra_spec)} floor must match: packages.{bundle_name}.optional_dependencies."
            f"{extra_name} ({extra_spec!r}) must equal packages.{leaf_name}."
            f"optional_dependencies.dev ({dev_spec!r})"
        )

    assert checked, (
        "expected at least one bundle extra dependency that a nested leaf also lists in its dev extra "
        "(e.g. mloda-enterprise[ed25519] / mloda-enterprise-audit)"
    )


def test_bundle_extra_sibling_floor_matches_leaf_dev_entry() -> None:
    """The guard above skips first-party siblings, so the ``{version}`` spelling of a sibling in a bundle extra
    and in the leaf's ``dev`` entry is pinned here."""
    packages = _packages()
    checked = 0

    for bundle_name, extra_name, extra_spec, leaf_name, dev_spec in _bundle_extra_leaf_dev_pairs(packages):
        if _dep_name(extra_spec) not in packages:
            continue
        checked += 1
        assert extra_spec == dev_spec, (
            f"the {_dep_name(extra_spec)} floor must match: packages.{bundle_name}.optional_dependencies."
            f"{extra_name} ({extra_spec!r}) must equal packages.{leaf_name}."
            f"optional_dependencies.dev ({dev_spec!r})"
        )

    assert checked, (
        "expected at least one bundle extra sibling that a nested leaf also lists in its dev extra "
        "(e.g. mloda-enterprise[openlineage] / mloda-enterprise-lineage)"
    )


def test_mloda_enterprise_openlineage_extra_carries_the_community_emitter_for_the_lineage_leaf() -> None:
    """The bundle offers the community emitter as an extra, never a hard dependency, and the leaf's dev extra
    matches it."""
    packages = _packages()

    extra = packages["mloda-enterprise"].get("optional_dependencies", {}).get("openlineage")
    assert extra == [_LINEAGE_EMITTER_FLOOR], (
        f"packages.mloda-enterprise.optional_dependencies.openlineage must be [{_LINEAGE_EMITTER_FLOOR!r}], "
        f"got {extra!r}"
    )

    hard_deps = packages["mloda-enterprise"].get("dependencies", [])
    offending = [dep for dep in hard_deps if _dep_name(dep) in ("mloda-community-openlineage", "openlineage-python")]
    assert not offending, (
        f"packages.mloda-enterprise.dependencies must not pin the openlineage emitter directly, found "
        f"{offending!r}; it belongs in optional_dependencies.openlineage instead"
    )

    lineage_dev = packages["mloda-enterprise-lineage"].get("optional_dependencies", {}).get("dev", [])
    assert lineage_dev.count(_LINEAGE_EMITTER_FLOOR) == 1, (
        f"packages.mloda-enterprise-lineage.optional_dependencies.dev must list {_LINEAGE_EMITTER_FLOOR!r} "
        f"exactly once so the leaf's own tests run against the emitter, got {lineage_dev!r}"
    )


def test_mloda_enterprise_pyproject_lists_the_openlineage_extra_with_the_shared_floor() -> None:
    """The generator's output and the committed pyproject.toml both carry the extra with ``{version}`` expanded
    (``tox -e check-generated`` is not in this gate)."""
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    expected = [f"mloda-community-openlineage>={shared['project']['version']}"]

    generated = tomllib.loads(
        gen.generate_pyproject("mloda-enterprise", packages["mloda-enterprise"], shared, packages)
    )
    assert _ENTERPRISE_PYPROJECT.is_file(), f"committed pyproject not found at {_ENTERPRISE_PYPROJECT}"
    committed = _load_toml(_ENTERPRISE_PYPROJECT)

    for source, project in (("generated", generated["project"]), ("committed", committed["project"])):
        actual = project.get("optional-dependencies", {}).get("openlineage")
        assert actual == expected, (
            f"the {source} mloda-enterprise pyproject must declare optional-dependencies.openlineage = "
            f"{expected!r}, got {actual!r} (run scripts/generate_pyproject.py)"
        )


@pytest.mark.parametrize("extra_name, distribution_name, leaf_name, root, exposed", _ROWS)
def test_generated_pyproject_lists_distribution_only_under_optional_extra(
    extra_name: str, distribution_name: str, leaf_name: str, root: str, exposed: list[str]
) -> None:
    """The generated community pyproject lists the leaf, pinned '==<version>', only under its own extra
    (and 'all'); the third-party distribution the leaf depends on never appears in the community pyproject."""
    assert _COMMUNITY_PYPROJECT.is_file(), f"generated pyproject not found at {_COMMUNITY_PYPROJECT}"
    project = _load_toml(_COMMUNITY_PYPROJECT)["project"]

    hard_deps = project.get("dependencies", [])
    offending = [dep for dep in hard_deps if _dep_name(dep) in (distribution_name, leaf_name)]
    assert not offending, (
        f"{_COMMUNITY_PYPROJECT}: [project].dependencies must not pin {distribution_name} or {leaf_name}, "
        f"found {offending!r}"
    )

    optional_deps: dict[str, list[str]] = project.get("optional-dependencies", {})
    distribution_appears = any(_dep_name(dep) == distribution_name for deps in optional_deps.values() for dep in deps)
    assert not distribution_appears, (
        f"{_COMMUNITY_PYPROJECT}: {distribution_name} must not appear anywhere; it is pinned only in "
        f"{leaf_name}'s own dependencies, not the community pyproject"
    )

    # "all" is expected to re-list every extra's entries (see the union test below), so it is not
    # itself a violation of "only under <extra_name>".
    groups_with_leaf = {
        group: deps
        for group, deps in optional_deps.items()
        if group != "all" and any(_dep_name(dep) == leaf_name for dep in deps)
    }
    assert groups_with_leaf == {extra_name: optional_deps.get(extra_name, [])}, (
        f"{_COMMUNITY_PYPROJECT}: {leaf_name} must appear only under "
        f"[project.optional-dependencies].{extra_name}, found it under {sorted(groups_with_leaf)}"
    )
    assert optional_deps.get(extra_name, []) == [f"{leaf_name}=={project['version']}"], (
        f"{_COMMUNITY_PYPROJECT}: [project.optional-dependencies].{extra_name} must be exactly "
        f"[{leaf_name!r}==<version>], got {optional_deps.get(extra_name)!r}"
    )


def test_mloda_community_declares_all_extra_as_union_of_per_extra_entries() -> None:
    """mloda-community[all] must install every optional plugin dependency."""
    optional = _packages()["mloda-community"].get("optional_dependencies", {})
    assert "all" in optional, "packages.mloda-community.optional_dependencies.all must be declared"

    expected: list[str] = []
    seen: set[str] = set()
    for extra_name, deps in optional.items():
        if extra_name in ("all", "dev"):
            continue
        for dep in deps:
            if dep not in seen:
                seen.add(dep)
                expected.append(dep)

    assert sorted(optional["all"]) == sorted(expected), (
        "packages.mloda-community.optional_dependencies.all must be exactly the union of every other "
        f"extra's entries, expected {sorted(expected)!r}, got {sorted(optional['all'])!r}"
    )


@pytest.mark.parametrize("extra_name, distribution_name, leaf_name, root, exposed", _ROWS)
def test_extra_only_bundle_dependencies_skip_via_plugin_loader_with_a_warning(
    extra_name: str,
    distribution_name: str,
    leaf_name: str,
    root: str,
    exposed: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A leaf reached only through a bundle extra is optional at runtime: PluginLoader is the sole guard,
    so the degrade is observed through load_entry_points(), not a direct manifest import. _ROWS is the
    single source of which leaves the bundle covers only through an extra."""
    packages = _packages()
    leaf_cfg = packages[leaf_name]
    dotted = leaf_cfg["path"].replace("/", ".")
    groups = leaf_cfg.get("entry_point_groups")
    assert groups, f"fixture assumption: {leaf_name} declares entry_point_groups"
    # PluginLoader.load_entry_points(group=...) only accepts plugin-type groups.
    groups = [g for g in groups if g != gen.OPTIONAL_DEPENDENCIES_GROUP]

    with pytest.MonkeyPatch.context() as mp:
        block_root(mp, root)
        evict_package(mp, dotted)

        # PluginLoader._skipped is process-wide and dedupes identical (entry point, dependency)
        # warnings, so a fresh reset keeps this test's warning assertion independent of whichever
        # other test last blocked the same distribution for the same entry point.
        PluginLoader.reset_cache()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=_PLUGIN_LOADER_LOGGER):
            for group in groups:
                keys = PluginLoader().load_entry_points(group=group)
                assert not any(key.startswith(f"{dotted}.") for key in keys), (
                    f"PluginLoader registered a class from {dotted}'s manifest even though "
                    f"{distribution_name} was blocked; the entry point should have been skipped"
                )

        plugin_loader_warnings = [
            record
            for record in caplog.records
            if record.name == _PLUGIN_LOADER_LOGGER
            and record.levelno == logging.WARNING
            and f"{dotted}.manifest:" in record.getMessage()
        ]
        assert plugin_loader_warnings, (
            f"PluginLoader.load_entry_points() logged no WARNING mentioning {dotted}.manifest: "
            f"while {distribution_name} was blocked"
        )

        leaf_module = importlib.import_module(dotted)
        for attr in exposed:
            assert attr not in vars(leaf_module), (
                f"{dotted} still exposes {attr!r} after blocking {distribution_name}; {leaf_name} is "
                f"covered only through the mloda-community.{extra_name} extra, so the package's __init__ "
                "must not import it eagerly"
            )


def test_bundle_shipped_leaf_with_extra_only_dependency_skips_via_plugin_loader(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Future guard, derived from the config rather than _ROWS: a nested leaf the bundle SHIPS (does not
    own) whose external dependency is covered only by one of the bundle's own extras must still degrade
    import-safely through PluginLoader, not a direct manifest import. Owned leaves are covered by the
    _ROWS-parametrized test above instead."""
    packages = _packages()

    for bundle_name, bundle_cfg in packages.items():
        if bundle_cfg.get("entry_point_bundle") is not True:
            continue
        prefix = bundle_cfg["path"] + "/"
        owned = _bundle_owned_names(bundle_name, packages)

        hard_names = _external_dependency_names(bundle_cfg.get("dependencies", []), packages)
        extra_names: set[str] = set()
        for extra_name, extra_deps in bundle_cfg.get("optional_dependencies", {}).items():
            if extra_name == "dev":
                continue
            extra_names |= _external_dependency_names(extra_deps, packages)
        extra_only = extra_names - hard_names

        for leaf_name, leaf_cfg in packages.items():
            if leaf_name == bundle_name or not leaf_cfg["path"].startswith(prefix) or leaf_name in owned:
                continue
            groups = leaf_cfg.get("entry_point_groups")
            if not groups:
                continue
            # PluginLoader.load_entry_points(group=...) only accepts plugin-type groups.
            groups = [g for g in groups if g != gen.OPTIONAL_DEPENDENCIES_GROUP]

            leaf_names = _external_dependency_names(leaf_cfg.get("dependencies", []), packages)
            covered_only_by_extra = sorted(leaf_names & extra_only)
            if not covered_only_by_extra:
                continue

            dotted = leaf_cfg["path"].replace("/", ".")

            # Scoped per leaf so a block/evict from one iteration never leaks into the next.
            with pytest.MonkeyPatch.context() as mp:
                for dist_name in covered_only_by_extra:
                    root = _IMPORT_ROOT_OF_DISTRIBUTION.get(dist_name)
                    assert root is not None, (
                        f"{leaf_name} depends on {dist_name!r}, which {bundle_name} covers only through its "
                        f"'{bundle_name}' optional extra; add {dist_name!r} to _IMPORT_ROOT_OF_DISTRIBUTION "
                        "in this test so its manifest can be checked for an import-safe degrade"
                    )
                    block_root(mp, root)

                evict_package(mp, dotted)

                # See the matching comment in test_extra_only_bundle_dependencies_skip_via_plugin_loader_with_a_warning:
                # PluginLoader._skipped dedupes identical warnings process-wide, across tests.
                PluginLoader.reset_cache()
                caplog.clear()
                with caplog.at_level(logging.WARNING, logger=_PLUGIN_LOADER_LOGGER):
                    for group in groups:
                        keys = PluginLoader().load_entry_points(group=group)
                        assert not any(key.startswith(f"{dotted}.") for key in keys), (
                            f"PluginLoader registered a class from {dotted}'s manifest even though "
                            f"{covered_only_by_extra} was blocked; the entry point should have been skipped"
                        )

                plugin_loader_warnings = [
                    record
                    for record in caplog.records
                    if record.name == _PLUGIN_LOADER_LOGGER
                    and record.levelno == logging.WARNING
                    and f"{dotted}.manifest:" in record.getMessage()
                ]
                assert plugin_loader_warnings, (
                    f"PluginLoader.load_entry_points() logged no WARNING mentioning {dotted}.manifest: "
                    f"while {covered_only_by_extra} was blocked"
                )

                exposed_attrs = _EXPOSED_EXTENDER_NAMES.get(leaf_name)
                assert exposed_attrs is not None, (
                    f"{leaf_name} has no entry in _EXPOSED_EXTENDER_NAMES in this test; add the attribute "
                    "name(s) its package __init__ must not expose when the extra-only dependency is absent"
                )
                leaf_module = importlib.import_module(dotted)
                for attr in exposed_attrs:
                    assert attr not in vars(leaf_module), (
                        f"{dotted} still exposes {attr!r} after blocking {covered_only_by_extra}; "
                        f"{leaf_name} is covered only through {bundle_name}'s extra, so the package's "
                        "__init__ must not import it eagerly"
                    )

    # No bundle-shipped (unowned) leaf has such a dependency today; the _ROWS-parametrized test above
    # exercises the same mechanism non-vacuously for the owned mloda-community-otel/-openlineage leaves.


# root -> a real transitive dependency of that root's own distribution (not the root itself), used to
# simulate "root is installed but one of its own dependencies is missing" (e.g. too old an install).
_TRANSITIVE_DEPENDENCY_OF_ROOT: dict[str, str] = {
    "openlineage": "attr",
    "opentelemetry": "typing_extensions",
}


@pytest.mark.parametrize("extra_name, distribution_name, leaf_name, root, exposed", _ROWS)
def test_plugin_loader_skips_entry_point_with_warning_when_transitive_dependency_is_missing(
    extra_name: str,
    distribution_name: str,
    leaf_name: str,
    root: str,
    exposed: list[str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``root`` is installed but one of its transitive dependencies is missing, so the failure is named
    after that dependency, not ``root``. PluginLoader must skip only this entry point with a WARNING,
    via the declared marker plus traceback blame."""
    transitive_dependency = _TRANSITIVE_DEPENDENCY_OF_ROOT[root]
    packages = _packages()
    leaf_cfg = packages[leaf_name]
    dotted = leaf_cfg["path"].replace("/", ".")
    # PluginLoader.load_entry_points(group=...) only accepts plugin-type groups.
    groups = [g for g in leaf_cfg["entry_point_groups"] if g != gen.OPTIONAL_DEPENDENCIES_GROUP]

    evict_root(monkeypatch, root)
    monkeypatch.setitem(sys.modules, transitive_dependency, None)
    evict_package(monkeypatch, dotted)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=_PLUGIN_LOADER_LOGGER):
        for group in groups:
            keys = PluginLoader().load_entry_points(group=group)
            assert not any(key.startswith(f"{dotted}.") for key in keys), (
                f"{dotted} was registered despite {transitive_dependency!r} being blocked"
            )

    plugin_loader_warnings = [
        record
        for record in caplog.records
        if record.name == _PLUGIN_LOADER_LOGGER
        and record.levelno == logging.WARNING
        and f"{dotted}.manifest:" in record.getMessage()
    ]
    assert plugin_loader_warnings, f"no WARNING mentioning {dotted}.manifest: logged for skipping {dotted}"
