"""mloda-community must not hard-require an optional plugin's own third-party dependency
(openlineage-python, opentelemetry-api, ...); each ships as its own bundle extra, pinned in
exactly one place: the leaf package's own dependency.
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

# Companion marker group (mloda.core...plugin_loader.OPTIONAL_DEPENDENCY_ENTRY_POINT_GROUP):
# PluginLoader consults it internally on every load_entry_points() call regardless of which
# group was requested, so it is never itself a valid `group=` argument to that method.
_OPTIONAL_DEPENDENCY_MARKER_GROUP = "mloda.optional_dependencies"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"
_COMMUNITY_PYPROJECT = _REPO_ROOT / "mloda" / "community" / "pyproject.toml"
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"

# One row per mloda-community extra-only dependency: (extra name, distribution name, leaf
# package name, import root, exposed extender names). Extend this when the bundle gains another
# extra-only plugin dependency.
_ROWS: list[tuple[str, str, str, str, list[str]]] = [
    ("openlineage", "openlineage-python", "mloda-community-openlineage", "openlineage", ["OpenLineageExtender"]),
    ("otel", "opentelemetry-api", "mloda-community-otel", "opentelemetry", ["OtelExtender"]),
]

# Distribution name -> the import root that provides it, derived from _ROWS. Extend _ROWS above
# when a bundle covers a new nested leaf's dependency only through an extra.
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


@pytest.mark.parametrize("extra_name, distribution_name, leaf_name, root, exposed", _ROWS)
def test_mloda_community_dependencies_do_not_pin_extra_only_distribution(
    extra_name: str, distribution_name: str, leaf_name: str, root: str, exposed: list[str]
) -> None:
    community = _packages()["mloda-community"]
    hard_deps = community.get("dependencies", [])
    offending = [dep for dep in hard_deps if _dep_name(dep) == distribution_name]
    assert not offending, (
        f"packages.mloda-community.dependencies must not pin {distribution_name} directly, found "
        f"{offending!r}; it belongs in optional_dependencies.{extra_name} instead"
    )


@pytest.mark.parametrize("extra_name, distribution_name, leaf_name, root, exposed", _ROWS)
def test_mloda_community_declares_extra_matching_pin_source(
    extra_name: str, distribution_name: str, leaf_name: str, root: str, exposed: list[str]
) -> None:
    packages = _packages()
    community_extra = packages["mloda-community"].get("optional_dependencies", {}).get(extra_name)
    assert community_extra is not None, f"packages.mloda-community.optional_dependencies.{extra_name} must be declared"

    matching = [dep for dep in community_extra if _dep_name(dep) == distribution_name]
    assert len(matching) == 1, (
        f"packages.mloda-community.optional_dependencies.{extra_name} must contain exactly one "
        f"{distribution_name} entry, got {community_extra!r}"
    )

    leaf_deps = packages[leaf_name]["dependencies"]
    leaf_matching = [dep for dep in leaf_deps if _dep_name(dep) == distribution_name]
    assert len(leaf_matching) == 1, (
        f"packages.{leaf_name}.dependencies must declare exactly one {distribution_name} entry, got {leaf_deps!r}"
    )
    assert matching[0] == leaf_matching[0], (
        f"the {distribution_name} pin must live in one place: packages.mloda-community."
        f"optional_dependencies.{extra_name} ({matching[0]!r}) must equal packages.{leaf_name}.dependencies "
        f"({leaf_matching[0]!r})"
    )


@pytest.mark.parametrize("extra_name, distribution_name, leaf_name, root, exposed", _ROWS)
def test_generated_pyproject_lists_distribution_only_under_optional_extra(
    extra_name: str, distribution_name: str, leaf_name: str, root: str, exposed: list[str]
) -> None:
    assert _COMMUNITY_PYPROJECT.is_file(), f"generated pyproject not found at {_COMMUNITY_PYPROJECT}"
    project = _load_toml(_COMMUNITY_PYPROJECT)["project"]

    hard_deps = project.get("dependencies", [])
    offending = [dep for dep in hard_deps if _dep_name(dep) == distribution_name]
    assert not offending, (
        f"{_COMMUNITY_PYPROJECT}: [project].dependencies must not pin {distribution_name}, found {offending!r}"
    )

    optional_deps: dict[str, list[str]] = project.get("optional-dependencies", {})
    # "all" is expected to re-list every extra's entries (see the union test below), so it is not
    # itself a violation of "only under <extra_name>".
    groups_with_distribution = {
        group: deps
        for group, deps in optional_deps.items()
        if group != "all" and any(_dep_name(dep) == distribution_name for dep in deps)
    }
    assert groups_with_distribution == {extra_name: optional_deps.get(extra_name, [])}, (
        f"{_COMMUNITY_PYPROJECT}: {distribution_name} must appear only under "
        f"[project.optional-dependencies].{extra_name}, found it under {sorted(groups_with_distribution)}"
    )
    assert len(optional_deps.get(extra_name, [])) == 1


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


def test_extra_only_bundle_dependencies_skip_via_plugin_loader_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A nested leaf's runtime dependency covered only through its bundle's own extra is optional at
    runtime: PluginLoader.load_entry_points(), not a direct ``<leaf>.manifest`` import, is the sole
    guard tolerating that. A direct import of the manifest with the dependency missing now raises by
    design, so the only way to observe the intended degrade is through PluginLoader itself.
    """
    packages = _packages()
    checked = 0

    for bundle_name, bundle_cfg in packages.items():
        if bundle_cfg.get("entry_point_bundle") is not True:
            continue
        prefix = bundle_cfg["path"] + "/"

        hard_names = _external_dependency_names(bundle_cfg.get("dependencies", []), packages)
        extra_names: set[str] = set()
        for extra_name, extra_deps in bundle_cfg.get("optional_dependencies", {}).items():
            if extra_name == "dev":
                continue
            extra_names |= _external_dependency_names(extra_deps, packages)
        extra_only = extra_names - hard_names

        for leaf_name, leaf_cfg in packages.items():
            if leaf_name == bundle_name or not leaf_cfg["path"].startswith(prefix):
                continue
            groups = leaf_cfg.get("entry_point_groups")
            if not groups:
                continue
            # PluginLoader.load_entry_points(group=...) only accepts plugin-type groups.
            groups = [g for g in groups if g != _OPTIONAL_DEPENDENCY_MARKER_GROUP]

            leaf_names = _external_dependency_names(leaf_cfg.get("dependencies", []), packages)
            covered_only_by_extra = sorted(leaf_names & extra_only)
            if not covered_only_by_extra:
                continue

            checked += 1
            dotted = leaf_cfg["path"].replace("/", ".")

            for dist_name in covered_only_by_extra:
                root = _IMPORT_ROOT_OF_DISTRIBUTION.get(dist_name)
                assert root is not None, (
                    f"{leaf_name} depends on {dist_name!r}, which {bundle_name} covers only through its "
                    f"'{bundle_name}' optional extra; add {dist_name!r} to _IMPORT_ROOT_OF_DISTRIBUTION "
                    "in this test so its manifest can be checked for an import-safe degrade"
                )
                block_root(monkeypatch, root)

            evict_package(monkeypatch, dotted)

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
                if record.name == _PLUGIN_LOADER_LOGGER and record.levelno == logging.WARNING
            ]
            assert plugin_loader_warnings, (
                f"PluginLoader.load_entry_points() logged no WARNING skipping {dotted}'s entry point "
                f"while {covered_only_by_extra} was blocked; manifest.py must no longer swallow this "
                "itself now that PluginLoader is the sole guard"
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

    assert checked, (
        "expected at least one entry-point-bundle nested leaf whose dependency is covered only through "
        "the bundle's own optional extra (e.g. mloda-community-openlineage / openlineage-python)"
    )


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
    """``root`` is genuinely installed, but one of its own transitive dependencies is missing (as if too
    old a version): the failure surfaces from a frame inside ``root``'s own code, named by that
    transitive dependency, not by ``root`` itself. PluginLoader.load_entry_points() must not abort
    discovery for every plugin over this, it must skip only this one entry point with a WARNING.

    Both leaves declare ``mloda.optional_dependencies`` (this package's own ``_optional_dependencies.py``
    marker), so PluginLoader resolves the leaf's own declared root (``openlineage``/``opentelemetry``)
    instead of falling back to its hardcoded OPTIONAL_PLUGIN_DEPENDENCIES allowlist. Traceback-frame
    blame (``_traceback_blames_root``) then attributes the failure, which surfaces under the transitive
    dependency's own name, back to that declared root, so discovery survives with a clean skip and
    WARNING rather than a re-raise.
    """
    transitive_dependency = _TRANSITIVE_DEPENDENCY_OF_ROOT[root]
    packages = _packages()
    leaf_cfg = packages[leaf_name]
    dotted = leaf_cfg["path"].replace("/", ".")
    # PluginLoader.load_entry_points(group=...) only accepts plugin-type groups.
    groups = [g for g in leaf_cfg["entry_point_groups"] if g != _OPTIONAL_DEPENDENCY_MARKER_GROUP]

    evict_root(monkeypatch, root)
    monkeypatch.setitem(sys.modules, transitive_dependency, None)
    evict_package(monkeypatch, dotted)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=_PLUGIN_LOADER_LOGGER):
        for group in groups:
            keys = PluginLoader().load_entry_points(group=group)
            assert not any(key.startswith(f"{dotted}.") for key in keys), (
                f"PluginLoader registered a class from {dotted}'s manifest even though its transitive "
                f"dependency {transitive_dependency!r} (of {root!r}) was blocked"
            )

    plugin_loader_warnings = [
        record
        for record in caplog.records
        if record.name == _PLUGIN_LOADER_LOGGER and record.levelno == logging.WARNING
    ]
    assert plugin_loader_warnings, (
        f"PluginLoader.load_entry_points() logged no WARNING skipping {dotted}'s entry point while its "
        f"transitive dependency {transitive_dependency!r} (of {root!r}) was blocked"
    )
