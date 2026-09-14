"""Entry-point declaration tests for issue #271.

mloda 0.9.0 discovers installed plugins through the entry-point groups
``mloda.feature_groups``, ``mloda.compute_frameworks`` and ``mloda.extenders``.
Each plugin package ships a ``manifest.py`` module exposing a list of concrete
plugin classes under a per-group attribute (``FEATURE_GROUPS``,
``COMPUTE_FRAMEWORKS`` or ``EXTENDERS``). The generator
``scripts/generate_pyproject.py`` must emit a
``[project.entry-points."<group>"]`` table whose entry name is the distribution
label and whose value is the canonical ``<dotted>.manifest:<ATTR>`` target.
Bundle packages (``mloda-community`` / ``mloda-enterprise``) aggregate the entry
points of every nested plugin package under their path.

Both the generator and the ``scripts/verify_builds.py`` script live as loose
scripts (not installed packages), so they are loaded here by file path through
``tests.script_loader``.
"""

from __future__ import annotations

import importlib
import inspect
import re
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest
from mloda.provider import ComputeFramework, FeatureGroup
from mloda.steward import Extender

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"
_VERIFY_BUILDS_PATH = _REPO_ROOT / "scripts" / "verify_builds.py"

# The three valid entry-point groups mapped to (manifest attribute, base type).
_GROUP_INFO: dict[str, tuple[str, type]] = {
    "mloda.feature_groups": ("FEATURE_GROUPS", FeatureGroup),
    "mloda.compute_frameworks": ("COMPUTE_FRAMEWORKS", ComputeFramework),
    "mloda.extenders": ("EXTENDERS", Extender),
}

# Companion marker group: targets a dependency-free sibling module (not manifest.py) and carries
# a tuple of import roots, not a list of plugin classes, so it doesn't fit _GROUP_INFO's shape.
# Covered separately by test_optional_dependencies_* and test_verify_builds_accepts_optional_
# dependencies_marker_target below.
_OPTIONAL_DEPENDENCY_MARKER_GROUP = "mloda.optional_dependencies"

_VALUE_PATTERN = re.compile(
    r"^(mloda\.community\.|mloda\.enterprise\.).*\.manifest:(FEATURE_GROUPS|COMPUTE_FRAMEWORKS|EXTENDERS)$"
)


gen = load_script("generate_pyproject", _GEN_PATH)
vb = load_script("verify_builds", _VERIFY_BUILDS_PATH)


def _generate(pkg_name: str) -> str:
    """Load configs and generate the pyproject text for a single package."""
    shared, packages_config = gen.load_configs()
    packages = packages_config["packages"]
    return str(gen.generate_pyproject(pkg_name, packages[pkg_name], shared, packages))


# (package, entry-point group, exact entry line it must emit), one package per group.
_PLUGIN_ENTRY_POINTS = [
    pytest.param(
        "mloda-community-ffill",
        "mloda.feature_groups",
        'mloda-community-ffill = "mloda.community.feature_groups.data_operations.row_preserving.ffill.manifest:FEATURE_GROUPS"',
        id="feature_group",
    ),
    pytest.param(
        "mloda-community-compute-frameworks-example",
        "mloda.compute_frameworks",
        'mloda-community-compute-frameworks-example = "mloda.community.compute_frameworks.example.manifest:COMPUTE_FRAMEWORKS"',
        id="compute_framework",
    ),
    pytest.param(
        "mloda-community-extenders-example",
        "mloda.extenders",
        'mloda-community-extenders-example = "mloda.community.extenders.example.manifest:EXTENDERS"',
        id="extender",
    ),
]


@pytest.mark.parametrize(("pkg_name", "group", "entry"), _PLUGIN_ENTRY_POINTS)
def test_plugin_package_declares_entry_point(pkg_name: str, group: str, entry: str) -> None:
    """A plugin package must declare its entry point under the group matching its plugin type."""
    content = _generate(pkg_name)
    assert f'[project.entry-points."{group}"]' in content, content
    assert entry in content, content


def test_bundle_aggregates_child_entry_points() -> None:
    """Bundle packages must aggregate the entry points of all nested plugin packages."""
    community = _generate("mloda-community")

    assert '[project.entry-points."mloda.feature_groups"]' in community, community
    assert (
        'mloda-community-ffill = "mloda.community.feature_groups.data_operations.row_preserving.ffill.manifest:FEATURE_GROUPS"'
        in community
    ), community
    assert 'mloda-community-example = "mloda.community.feature_groups.example.manifest:FEATURE_GROUPS"' in community, (
        community
    )

    assert '[project.entry-points."mloda.compute_frameworks"]' in community, community
    assert (
        'mloda-community-compute-frameworks-example = "mloda.community.compute_frameworks.example.manifest:COMPUTE_FRAMEWORKS"'
        in community
    ), community

    assert '[project.entry-points."mloda.extenders"]' in community, community
    assert 'mloda-community-extenders-example = "mloda.community.extenders.example.manifest:EXTENDERS"' in community, (
        community
    )

    enterprise = _generate("mloda-enterprise")

    assert '[project.entry-points."mloda.feature_groups"]' in enterprise, enterprise
    assert (
        'mloda-enterprise-example = "mloda.enterprise.feature_groups.example.manifest:FEATURE_GROUPS"' in enterprise
    ), enterprise

    assert '[project.entry-points."mloda.compute_frameworks"]' in enterprise, enterprise
    assert (
        'mloda-enterprise-compute-frameworks-example = "mloda.enterprise.compute_frameworks.example.manifest:COMPUTE_FRAMEWORKS"'
        in enterprise
    ), enterprise

    assert '[project.entry-points."mloda.extenders"]' in enterprise, enterprise
    assert (
        'mloda-enterprise-extenders-example = "mloda.enterprise.extenders.example.manifest:EXTENDERS"' in enterprise
    ), enterprise


@pytest.mark.parametrize(
    "pkg_name",
    ["mloda-registry", "mloda-testing", "mloda-community-data-operations"],
)
def test_non_plugin_packages_have_no_entry_points(pkg_name: str) -> None:
    """Non-plugin packages (tools, test utilities, shared bases) declare no entry points."""
    content = _generate(pkg_name)
    assert "[project.entry-points." not in content, content


def test_all_entry_point_values_are_namespaced_manifests() -> None:
    """Every emitted entry-point target must be a namespaced ``.manifest:<ATTR>`` value."""
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]

    for pkg_name, pkg_config in packages.items():
        content = gen.generate_pyproject(pkg_name, pkg_config, shared, packages)
        data = tomllib.loads(content)
        entry_points = data.get("project", {}).get("entry-points")
        if not entry_points:
            continue
        for group, mapping in entry_points.items():
            if group == _OPTIONAL_DEPENDENCY_MARKER_GROUP:
                continue
            assert group in _GROUP_INFO, f"{pkg_name}: unexpected entry-point group {group!r}"
            for name, value in mapping.items():
                assert _VALUE_PATTERN.match(value), (
                    f"{pkg_name}: entry point {name!r} in group {group!r} has non-namespaced-manifest value {value!r}"
                )


def _plugin_packages() -> list[tuple[str, dict[str, Any]]]:
    """Config-driven list of plugin packages (those declaring entry_point_groups)."""
    _shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    return [(name, cfg) for name, cfg in packages.items() if cfg.get("entry_point_groups")]


def test_manifest_modules_list_only_concrete_plugins() -> None:
    """Each manifest attribute must list only concrete, correctly based, non-shared plugin classes."""
    plugin_packages = _plugin_packages()
    assert plugin_packages, "expected at least one package with entry_point_groups declared in config"

    for pkg_name, pkg_config in plugin_packages:
        dotted = pkg_config["path"].replace("/", ".")
        manifest_name = f"{dotted}.manifest"
        module = importlib.import_module(manifest_name)

        for group in pkg_config["entry_point_groups"]:
            if group == _OPTIONAL_DEPENDENCY_MARKER_GROUP:
                continue
            assert group in _GROUP_INFO, f"{pkg_name}: unexpected entry-point group {group!r}"
            attr_name, base_type = _GROUP_INFO[group]
            assert hasattr(module, attr_name), f"{manifest_name}: missing attribute {attr_name}"
            plugins = getattr(module, attr_name)

            assert isinstance(plugins, list), f"{manifest_name}.{attr_name} must be a list"
            assert plugins, f"{manifest_name}.{attr_name} must be non-empty"

            for cls in plugins:
                assert inspect.isclass(cls), f"{manifest_name}.{attr_name}: {cls!r} is not a class"
                assert not inspect.isabstract(cls), f"{manifest_name}.{attr_name}: {cls!r} is abstract"
                assert issubclass(cls, base_type), (
                    f"{manifest_name}.{attr_name}: {cls!r} is not a subclass of {base_type.__name__}"
                )
                assert not (cls.__module__.endswith(".base") or cls.__module__.endswith("_base")), (
                    f"{manifest_name}.{attr_name}: {cls!r} is a shared base class ({cls.__module__})"
                )


def test_bundle_and_groups_are_mutually_exclusive() -> None:
    """A package config declaring both entry_point_bundle and entry_point_groups must be rejected."""
    pkg_config: dict[str, Any] = {
        "path": "mloda/community",
        "entry_point_bundle": True,
        "entry_point_groups": ["mloda.feature_groups"],
    }
    all_packages: dict[str, dict[str, Any]] = {"mloda-bogus": pkg_config}

    with pytest.raises(ValueError, match="mutually exclusive"):
        gen.compute_entry_points("mloda-bogus", pkg_config, all_packages)


def test_optional_dependencies_group_is_registered_in_entry_point_attrs() -> None:
    """The new ``mloda.optional_dependencies`` group must map to an ``OPTIONAL_DEPENDENCIES`` manifest
    attribute in ENTRY_POINT_ATTRS, exactly like the three existing plugin groups map to their own
    attribute, so a declaring package's generated pyproject.toml can carry it."""
    assert "mloda.optional_dependencies" in gen.ENTRY_POINT_ATTRS
    assert gen.ENTRY_POINT_ATTRS["mloda.optional_dependencies"] == "OPTIONAL_DEPENDENCIES"


def test_optional_dependencies_entry_point_targets_a_dependency_free_sibling_module_not_manifest() -> None:
    """OPTIONAL_DEPENDENCIES cannot live inside manifest.py itself: PluginLoader only consults the
    ``mloda.optional_dependencies`` marker *inside* the except-ImportError handler for the guarded entry
    point, i.e. after manifest.py's own import has already failed. If the marker lived in manifest.py,
    loading it would re-attempt that same failing import and fail too, so the declaration would never be
    readable exactly when it is needed. It must live in a separate, dependency-free sibling module
    instead (e.g. ``_optional_dependencies.py``), which ``compute_entry_points`` must target instead of
    the ``<path>.manifest:<ATTR>`` value every other group uses today.
    """
    pkg_config: dict[str, Any] = {
        "path": "mloda/community/extenders/openlineage",
        "entry_point_groups": ["mloda.extenders", "mloda.optional_dependencies"],
    }
    all_packages: dict[str, dict[str, Any]] = {"mloda-community-openlineage": pkg_config}

    entry_points = gen.compute_entry_points("mloda-community-openlineage", pkg_config, all_packages)

    assert "mloda.optional_dependencies" in entry_points
    [(label, value)] = entry_points["mloda.optional_dependencies"]
    assert label == "mloda-community-openlineage"
    assert value == "mloda.community.extenders.openlineage._optional_dependencies:OPTIONAL_DEPENDENCIES"
    assert not value.startswith("mloda.community.extenders.openlineage.manifest:"), (
        "OPTIONAL_DEPENDENCIES must not live in manifest.py: PluginLoader only reads the "
        "mloda.optional_dependencies marker after manifest.py's own import already failed, so a marker "
        "living inside manifest.py could never be read exactly when it is needed"
    )


def test_verify_builds_namespace_helper() -> None:
    """verify_builds must expose namespaced_entry_point_error validating entry-point targets."""
    helper = getattr(vb, "namespaced_entry_point_error", None)
    assert callable(helper), "verify_builds.namespaced_entry_point_error must be a callable"

    # Valid target -> no error.
    assert (
        helper(
            "mloda.feature_groups",
            "mloda-community-ffill",
            "mloda.community.feature_groups.data_operations.row_preserving.ffill.manifest:FEATURE_GROUPS",
        )
        is None
    )

    # Value not under the mloda namespace -> error.
    assert helper("mloda.feature_groups", "some-pkg", "some_pkg.manifest:FEATURE_GROUPS") is not None

    # Module does not end in .manifest -> error.
    assert (
        helper("mloda.feature_groups", "mloda-community-foo", "mloda.community.foo.plugins:FEATURE_GROUPS") is not None
    )

    # Attribute not one of the three allowed -> error.
    assert helper("mloda.feature_groups", "mloda-community-foo", "mloda.community.foo.manifest:PLUGINS") is not None


def test_verify_builds_valid_entry_point_attrs_include_optional_dependencies() -> None:
    """OPTIONAL_DEPENDENCIES must join FEATURE_GROUPS/COMPUTE_FRAMEWORKS/EXTENDERS as a valid attribute,
    or a built wheel declaring the new marker fails verify_builds.py's own consistency check."""
    assert "OPTIONAL_DEPENDENCIES" in vb._VALID_ENTRY_POINT_ATTRS


def test_verify_builds_accepts_optional_dependencies_marker_target() -> None:
    """namespaced_entry_point_error must accept the mloda.optional_dependencies group's target module,
    which is a dependency-free sibling of manifest.py (not manifest.py itself, see the placement
    constraint documented on test_optional_dependencies_entry_point_targets_a_dependency_free_sibling_module_not_manifest
    above): it cannot keep requiring every entry point's module to end with '.manifest'."""
    assert (
        vb.namespaced_entry_point_error(
            "mloda.optional_dependencies",
            "mloda-community-openlineage",
            "mloda.community.extenders.openlineage._optional_dependencies:OPTIONAL_DEPENDENCIES",
        )
        is None
    )
