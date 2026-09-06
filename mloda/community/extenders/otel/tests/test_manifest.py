"""Manifest resilience: mloda's entry-point loader only tolerates a missing ``opentelemetry``
root (OPTIONAL_PLUGIN_DEPENDENCIES does not list it, so the manifest itself must degrade to an
empty EXTENDERS list), while any other import error must still propagate.
"""

from __future__ import annotations

import importlib
import sys

import pytest

_PACKAGE = "mloda.community.extenders.otel"


def _evict(monkeypatch: pytest.MonkeyPatch, dotted: str) -> None:
    """Drop dotted and everything cached under it from sys.modules."""
    for name in list(sys.modules):
        if name == dotted or name.startswith(f"{dotted}."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def _block_opentelemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poison sys.modules so any import of the opentelemetry root raises ModuleNotFoundError."""
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", None)
    for name in list(sys.modules):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            monkeypatch.setitem(sys.modules, name, None)


def test_manifest_lists_the_extender_when_opentelemetry_is_installed() -> None:
    from mloda.community.extenders.otel import manifest
    from mloda.community.extenders.otel.otel_extender import OtelExtender

    assert manifest.EXTENDERS == [OtelExtender]


def test_manifest_is_empty_when_opentelemetry_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_opentelemetry(monkeypatch)
    _evict(monkeypatch, _PACKAGE)

    module = importlib.import_module(f"{_PACKAGE}.manifest")

    assert module.EXTENDERS == []


def test_manifest_reraises_unrelated_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, f"{_PACKAGE}.otel_extender", None)
    monkeypatch.delitem(sys.modules, f"{_PACKAGE}.manifest", raising=False)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{_PACKAGE}.manifest")


def test_package_imports_without_opentelemetry_and_hides_the_extender(monkeypatch: pytest.MonkeyPatch) -> None:
    """The package must import cleanly without opentelemetry and must not expose the extender."""
    _block_opentelemetry(monkeypatch)
    _evict(monkeypatch, _PACKAGE)

    module = importlib.import_module(_PACKAGE)

    assert hasattr(module, "OtelExtender") is False

    with pytest.raises(ImportError):
        from mloda.community.extenders.otel import OtelExtender  # noqa: F401
