"""Manifest resilience: mloda's entry-point loader only tolerates a missing ``opentelemetry``
root (OPTIONAL_PLUGIN_DEPENDENCIES does not list it, so the manifest itself must degrade to an
empty EXTENDERS list), while any other import error must still propagate.
"""

from __future__ import annotations

import importlib
import logging
import sys

import pytest

from mloda.testing.import_isolation import block_root, evict_package

_PACKAGE = "mloda.community.extenders.otel"
_ROOT = "opentelemetry"


def test_manifest_lists_the_extender_when_opentelemetry_is_installed() -> None:
    from mloda.community.extenders.otel import manifest
    from mloda.community.extenders.otel.otel_extender import OtelExtender

    assert manifest.EXTENDERS == [OtelExtender]


def test_manifest_is_empty_when_opentelemetry_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    block_root(monkeypatch, _ROOT)
    evict_package(monkeypatch, _PACKAGE)

    module = importlib.import_module(f"{_PACKAGE}.manifest")

    assert module.EXTENDERS == []


def test_manifest_reraises_unrelated_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, f"{_PACKAGE}.otel_extender", None)
    monkeypatch.delitem(sys.modules, f"{_PACKAGE}.manifest", raising=False)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{_PACKAGE}.manifest")


def test_package_imports_without_opentelemetry_and_hides_the_extender(monkeypatch: pytest.MonkeyPatch) -> None:
    """The package must import cleanly without opentelemetry and must not expose the extender."""
    block_root(monkeypatch, _ROOT)
    evict_package(monkeypatch, _PACKAGE)

    module = importlib.import_module(_PACKAGE)

    assert "OtelExtender" not in vars(module)

    with pytest.raises(ImportError):
        from mloda.community.extenders.otel import OtelExtender  # noqa: F401


def test_missing_dependency_import_error_names_distribution_and_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    block_root(monkeypatch, _ROOT)
    evict_package(monkeypatch, _PACKAGE)

    with pytest.raises(ImportError) as info:
        from mloda.community.extenders.otel import OtelExtender  # noqa: F401

    assert "opentelemetry-api" in str(info.value)
    assert "mloda-community[otel]" in str(info.value)

    module = importlib.import_module(_PACKAGE)
    assert getattr(module, "some_other_name", None) is None


def test_manifest_logs_when_dependency_is_missing(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    block_root(monkeypatch, _ROOT)
    evict_package(monkeypatch, _PACKAGE)

    with caplog.at_level(logging.INFO):
        importlib.import_module(f"{_PACKAGE}.manifest")

    matching = [
        record
        for record in caplog.records
        if record.name.startswith("mloda.community.extenders.")
        and record.levelno >= logging.INFO
        and "mloda-community[otel]" in record.getMessage()
    ]
    assert matching, "expected an INFO+ log record naming the mloda-community[otel] extra"


def test_blocking_tests_leave_no_degraded_module_behind() -> None:
    """A prior test blocking opentelemetry must not leak its degraded module into later imports."""
    root_package = "mloda.community.extenders"
    leaf_attr = _PACKAGE.rsplit(".", 1)[1]
    manifest_name = f"{_PACKAGE}.manifest"

    if _PACKAGE not in sys.modules:
        importlib.import_module(_PACKAGE)
    if manifest_name in sys.modules:
        del sys.modules[manifest_name]

    before_package = sys.modules.get(_PACKAGE)
    before_manifest = sys.modules.get(manifest_name)

    with pytest.MonkeyPatch.context() as mp:
        block_root(mp, _ROOT)
        evict_package(mp, _PACKAGE)
        importlib.import_module(manifest_name)

    assert sys.modules.get(manifest_name) is before_manifest
    assert sys.modules.get(_PACKAGE) is before_package

    from mloda.community.extenders.otel.otel_extender import OtelExtender

    module = importlib.import_module(manifest_name)
    assert module.EXTENDERS == [OtelExtender]

    restored_leaf = getattr(importlib.import_module(root_package), leaf_attr, None)
    assert restored_leaf is not None
    assert hasattr(restored_leaf, "OtelExtender")
