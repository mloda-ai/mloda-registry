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
scripts (not installed packages), so they are loaded here by file path using the
same ``importlib.util`` pattern as ``test_generate_pyproject_guards.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"
_VERIFY_BUILDS_PATH = _REPO_ROOT / "scripts" / "verify_builds.py"

# The three valid entry-point groups mapped to (manifest attribute, base type).
_GROUP_INFO: dict[str, tuple[str, type]] = {
    "mloda.feature_groups": ("FEATURE_GROUPS", FeatureGroup),
    "mloda.compute_frameworks": ("COMPUTE_FRAMEWORKS", ComputeFramework),
    "mloda.extenders": ("EXTENDERS", Extender),
}

_VALUE_PATTERN = re.compile(
    r"^(mloda\.community\.|mloda\.enterprise\.).*\.manifest:(FEATURE_GROUPS|COMPUTE_FRAMEWORKS|EXTENDERS)$"
)


def _load_module(name: str, path: Path) -> ModuleType:
    """Import a loose script by file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"could not load spec for {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_module("generate_pyproject", _GEN_PATH)
vb = _load_module("verify_builds", _VERIFY_BUILDS_PATH)


def _generate(pkg_name: str) -> str:
    """Load configs and generate the pyproject text for a single package."""
    shared, packages_config = gen.load_configs()
    packages = packages_config["packages"]
    return str(gen.generate_pyproject(pkg_name, packages[pkg_name], shared, packages))


def test_feature_group_package_declares_entry_point() -> None:
    """A FeatureGroup plugin package must declare a mloda.feature_groups entry point."""
    content = _generate("mloda-community-ffill")
    assert '[project.entry-points."mloda.feature_groups"]' in content, content
    assert (
        'mloda-community-ffill = "mloda.community.feature_groups.data_operations.row_preserving.ffill.manifest:FEATURE_GROUPS"'
        in content
    ), content


def test_compute_framework_package_declares_entry_point() -> None:
    """A ComputeFramework plugin package must declare a mloda.compute_frameworks entry point."""
    content = _generate("mloda-community-compute-frameworks-example")
    assert '[project.entry-points."mloda.compute_frameworks"]' in content, content
    assert (
        'mloda-community-compute-frameworks-example = "mloda.community.compute_frameworks.example.manifest:COMPUTE_FRAMEWORKS"'
        in content
    ), content


def test_extender_package_declares_entry_point() -> None:
    """An Extender plugin package must declare a mloda.extenders entry point."""
    content = _generate("mloda-community-extenders-example")
    assert '[project.entry-points."mloda.extenders"]' in content, content
    assert 'mloda-community-extenders-example = "mloda.community.extenders.example.manifest:EXTENDERS"' in content, (
        content
    )


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


# 0.3.3 is the first registry release shipping manifest_utils, which the
# data_operations manifest.py files import; older floors permit installs
# whose entry points fail silently at discovery time.
_MANIFEST_UTILS_MIN = (0, 3, 3)

_DATA_OPERATIONS_PREFIX = "mloda/community/feature_groups/data_operations/"
_DATA_OPERATIONS_DEP_PREFIX = "mloda-community-data-operations>="


def test_data_operations_plugins_require_manifest_utils_capable_floor() -> None:
    """data_operations plugin packages must require a manifest_utils-capable base package."""
    _shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]

    checked = 0
    for pkg_name, pkg_config in packages.items():
        if not pkg_config.get("entry_point_groups"):
            continue
        if not pkg_config["path"].startswith(_DATA_OPERATIONS_PREFIX):
            continue
        for dep in pkg_config.get("dependencies", []):
            if not dep.startswith(_DATA_OPERATIONS_DEP_PREFIX):
                continue
            checked += 1
            floor = tuple(int(part) for part in dep.removeprefix(_DATA_OPERATIONS_DEP_PREFIX).split("."))
            assert floor >= _MANIFEST_UTILS_MIN, (
                f"{pkg_name}: dependency {dep!r} permits versions without manifest_utils; "
                f"the floor must be >= {'.'.join(str(p) for p in _MANIFEST_UTILS_MIN)}"
            )

    assert checked, "expected at least one data_operations plugin package with a base-package dependency"


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
