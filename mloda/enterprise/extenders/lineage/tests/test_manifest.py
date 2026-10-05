"""This manifest swallows a missing community emitter on purpose: PluginLoader re-raises a missing ``mloda.*``
module (its root equals the entry point's own), so an unguarded import would break every enterprise plugin on an
install without the extra.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
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

# Community emitter unavailable: its package is missing, or openlineage-python (or attr, only installed through it) is.
_MISSING_ROOTS = [_COMMUNITY, "openlineage", "attr"]

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


_COMMUNITY_DIST = "mloda-community-openlineage"
_ENTERPRISE_DIST = "mloda-enterprise"
_ENTERPRISE_VERSION = "1.4.2"
_EXTRA = "mloda-enterprise[openlineage]"


def _fake_versions(monkeypatch: pytest.MonkeyPatch, versions: dict[str, str | None]) -> None:
    """Patch ``importlib.metadata.version``: a ``None`` value raises PackageNotFoundError, unlisted names
    delegate to the real function."""
    real_version = importlib.metadata.version

    def fake_version(distribution_name: str) -> str:
        if distribution_name in versions:
            version = versions[distribution_name]
            if version is None:
                raise importlib.metadata.PackageNotFoundError(distribution_name)
            return version
        return real_version(distribution_name)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)


def _cold_import_with_versions(
    monkeypatch: pytest.MonkeyPatch, community: str | None, enterprise: str | None = _ENTERPRISE_VERSION
) -> None:
    evict_package(monkeypatch, _PACKAGE)
    evict_package(monkeypatch, _COMMUNITY)
    _fake_versions(monkeypatch, {_COMMUNITY_DIST: community, _ENTERPRISE_DIST: enterprise})


def _manifest_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _MANIFEST and r.levelno == logging.WARNING]


@pytest.mark.parametrize("community", ["1.5.0", "1.3.7"])
def test_manifest_is_empty_and_warns_once_when_the_community_minor_differs(
    community: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _cold_import_with_versions(monkeypatch, community=community)

    with caplog.at_level(logging.WARNING, logger=_MANIFEST):
        manifest = importlib.import_module(_MANIFEST)

    assert manifest.EXTENDERS == []
    warnings = _manifest_warnings(caplog)
    assert len(warnings) == 1
    assert community in warnings[0]
    assert _ENTERPRISE_VERSION in warnings[0]
    assert _EXTRA in warnings[0]
    assert _EXTENDER_MODULE not in sys.modules


@pytest.mark.parametrize(
    ("community", "enterprise"),
    [
        pytest.param("1.4.9", _ENTERPRISE_VERSION, id="community-patch-differs"),
        pytest.param(None, _ENTERPRISE_VERSION, id="community-metadata-missing"),
        pytest.param("1.5.0", None, id="enterprise-metadata-missing"),
    ],
)
def test_manifest_lists_the_extender_when_the_versions_match_on_minor_or_cannot_be_compared(
    community: str | None, enterprise: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A patch difference is tolerated; editable or source checkouts without installed metadata skip the check."""
    _cold_import_with_versions(monkeypatch, community=community, enterprise=enterprise)

    manifest = importlib.import_module(_MANIFEST)
    extender = getattr(importlib.import_module(_EXTENDER_MODULE), _EXTENDER_NAME)

    assert manifest.EXTENDERS == [extender]


def test_plugin_loader_registers_nothing_when_the_community_minor_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    _cold_import_with_versions(monkeypatch, community="1.5.0")

    def only_the_lineage_entry_point(*, group: str) -> list[importlib.metadata.EntryPoint]:
        return [_ENTRY_POINT] if group == "mloda.extenders" else []

    monkeypatch.setattr(importlib.metadata, "entry_points", only_the_lineage_entry_point)

    keys = PluginLoader().load_entry_points(group="mloda.extenders")

    assert keys == []


def test_importing_the_extender_directly_raises_when_the_community_minor_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cold_import_with_versions(monkeypatch, community="1.5.0")

    with pytest.raises(ImportError) as excinfo:
        importlib.import_module(_EXTENDER_MODULE)

    message = str(excinfo.value)
    assert "1.5.0" in message
    assert _ENTERPRISE_VERSION in message
    assert _EXTRA in message
