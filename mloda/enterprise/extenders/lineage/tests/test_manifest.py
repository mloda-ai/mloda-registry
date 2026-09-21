"""Manifest resilience for mloda-enterprise-lineage. Unlike the community emitter (whose contract lives in
OptionalDependencyPackageTestMixin), this manifest swallows a missing community emitter on purpose:
PluginLoader re-raises a missing ``mloda.*`` module because its root equals the entry point's own root, so an
unguarded import would break discovery of every enterprise plugin on an install without the extra.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

import pytest
from mloda.user import PluginLoader

from mloda.testing.import_isolation import block_root, evict_package, evict_root

_PACKAGE = "mloda.enterprise.extenders.lineage"
_MANIFEST = f"{_PACKAGE}.manifest"
_EXTENDER_MODULE = f"{_PACKAGE}.lineage_extender"
_EXTENDER_NAME = "LineageFacetsExtender"
_COMMUNITY = "mloda.community.extenders.openlineage"
_SHARED = "mloda.community.extenders.shared"

# The two shapes of "the community emitter is unavailable": its package is not installed, or it is
# installed but openlineage-python is not.
_MISSING_ROOTS = [_COMMUNITY, "openlineage"]

# Hand-built so the discovery test needs neither installed metadata nor the community entry point.
_ENTRY_POINT = importlib.metadata.EntryPoint(
    name="mloda-enterprise-lineage", value=f"{_MANIFEST}:EXTENDERS", group="mloda.extenders"
)


def _cold_import_without(monkeypatch: pytest.MonkeyPatch, missing_root: str) -> None:
    """Evict the lineage and community packages, then poison ``missing_root`` so the next import of the
    manifest re-executes and hits it. Evict first: evicting after blocking would lift the block."""
    evict_package(monkeypatch, _PACKAGE)
    evict_package(monkeypatch, _COMMUNITY)
    block_root(monkeypatch, missing_root)


def test_manifest_lists_the_extender_when_the_community_emitter_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    evict_package(monkeypatch, _PACKAGE)

    manifest = importlib.import_module(_MANIFEST)
    extender = getattr(importlib.import_module(_EXTENDER_MODULE), _EXTENDER_NAME)

    assert manifest.EXTENDERS == [extender]


@pytest.mark.parametrize("missing_root", _MISSING_ROOTS)
def test_manifest_is_empty_when_the_community_emitter_is_missing(
    missing_root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cold_import_without(monkeypatch, missing_root)

    manifest = importlib.import_module(_MANIFEST)

    assert manifest.EXTENDERS == []


def test_manifest_reraises_a_missing_module_unrelated_to_the_community_emitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard is scoped to openlineage: a sibling under the same ``mloda.community.extenders`` namespace
    that the emitter imports first must stay a loud failure."""
    _cold_import_without(monkeypatch, _SHARED)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module(_MANIFEST)

    # block_root poisons the cached submodules too, so the re-raised name is the first one the emitter imports.
    missing = excinfo.value.name
    assert missing is not None
    assert missing == _SHARED or missing.startswith(f"{_SHARED}."), (
        f"expected a module under {_SHARED!r}, got {missing!r}"
    )


def test_package_import_does_not_import_openlineage_or_the_extender(monkeypatch: pytest.MonkeyPatch) -> None:
    """The package re-exports lazily: the manifest guard cannot help an eager import of the emitter, and the
    package must stay importable without the extra."""
    evict_root(monkeypatch, "openlineage")
    evict_package(monkeypatch, _COMMUNITY)
    evict_package(monkeypatch, _PACKAGE)

    module = importlib.import_module(_PACKAGE)

    assert _EXTENDER_MODULE not in sys.modules
    assert "openlineage" not in sys.modules
    assert not [name for name in sys.modules if name == _COMMUNITY or name.startswith(f"{_COMMUNITY}.")]

    extender = getattr(module, _EXTENDER_NAME)

    assert _EXTENDER_MODULE in sys.modules
    assert extender is getattr(sys.modules[_EXTENDER_MODULE], _EXTENDER_NAME)


@pytest.mark.parametrize("missing_root", _MISSING_ROOTS)
def test_plugin_loader_skips_the_entry_point_quietly_when_the_community_emitter_is_missing(
    missing_root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The discovery break the guard prevents: without it PluginLoader re-raises the missing ``mloda.*``
    module (own root ``mloda``) and no extender registers on an install without the extra."""
    _cold_import_without(monkeypatch, missing_root)

    def only_the_lineage_entry_point(*, group: str) -> list[importlib.metadata.EntryPoint]:
        return [_ENTRY_POINT] if group == "mloda.extenders" else []

    monkeypatch.setattr(importlib.metadata, "entry_points", only_the_lineage_entry_point)

    keys = PluginLoader().load_entry_points(group="mloda.extenders")

    assert keys == []
