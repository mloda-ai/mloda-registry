"""Manifest resilience: mloda's entry-point loader only tolerates a missing ``openlineage``
root (OPTIONAL_PLUGIN_DEPENDENCIES does not list it, so the manifest itself must degrade to an
empty EXTENDERS list), while any other import error must still propagate.
"""

from __future__ import annotations

import importlib
import sys

import pytest

_PACKAGE = "mloda.community.extenders.openlineage"


def _evict(monkeypatch: pytest.MonkeyPatch, dotted: str) -> None:
    """Drop dotted and everything cached under it from sys.modules."""
    for name in list(sys.modules):
        if name == dotted or name.startswith(f"{dotted}."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def _block_openlineage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poison sys.modules so any import of the openlineage root raises ModuleNotFoundError."""
    monkeypatch.setitem(sys.modules, "openlineage", None)
    monkeypatch.setitem(sys.modules, "openlineage.client", None)
    for name in list(sys.modules):
        if name == "openlineage" or name.startswith("openlineage."):
            monkeypatch.setitem(sys.modules, name, None)


def test_manifest_lists_the_extender_when_openlineage_is_installed() -> None:
    from mloda.community.extenders.openlineage import manifest
    from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender

    assert manifest.EXTENDERS == [OpenLineageExtender]


def test_manifest_is_empty_when_openlineage_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_openlineage(monkeypatch)
    _evict(monkeypatch, _PACKAGE)

    module = importlib.import_module(f"{_PACKAGE}.manifest")

    assert module.EXTENDERS == []


def test_manifest_reraises_unrelated_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, f"{_PACKAGE}.openlineage_extender", None)
    monkeypatch.delitem(sys.modules, f"{_PACKAGE}.manifest", raising=False)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{_PACKAGE}.manifest")


def test_package_imports_without_openlineage_and_hides_the_extender(monkeypatch: pytest.MonkeyPatch) -> None:
    """The package must import cleanly without openlineage and must not expose the extender."""
    _block_openlineage(monkeypatch)
    _evict(monkeypatch, _PACKAGE)

    module = importlib.import_module(_PACKAGE)

    assert hasattr(module, "OpenLineageExtender") is False

    with pytest.raises(ImportError):
        from mloda.community.extenders.openlineage import OpenLineageExtender  # noqa: F401
