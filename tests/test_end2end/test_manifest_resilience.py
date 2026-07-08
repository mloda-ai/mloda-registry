"""Unit tests for the resilient manifest loader helper (issue #271).

``load_plugin_classes`` builds an entry-point manifest's class list by importing
each backend module individually. A backend whose optional compute framework
(pandas, polars, duckdb, pyarrow) is not installed must be skipped so the rest
still register, while any other import error must stay loud. These tests drive
that behaviour by monkeypatching the helper module's ``importlib.import_module``,
so they need no optional framework installed and touch no network.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes

_IMPORT_MODULE_TARGET = "mloda.community.feature_groups.data_operations.manifest_utils.importlib.import_module"


class _KeptClass:
    """Placeholder class returned by a successfully imported backend."""


def test_skips_backend_with_missing_optional_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    kept_module = SimpleNamespace(KeptClass=_KeptClass)

    def fake_import(name: str) -> Any:
        if name.endswith("polars_backend"):
            raise ModuleNotFoundError("No module named 'polars'", name="polars")
        return kept_module

    monkeypatch.setattr(_IMPORT_MODULE_TARGET, fake_import)

    classes = load_plugin_classes(
        "pkg",
        [
            ("polars_backend", "PolarsClass"),
            ("pandas_backend", "KeptClass"),
        ],
    )

    assert [c.__name__ for c in classes] == ["_KeptClass"]


def test_reraises_non_optional_module_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> ModuleType:
        raise ModuleNotFoundError(
            "No module named 'mloda.community.foo.bar'",
            name="mloda.community.foo.bar",
        )

    monkeypatch.setattr(_IMPORT_MODULE_TARGET, fake_import)

    with pytest.raises(ModuleNotFoundError):
        load_plugin_classes("pkg", [("missing_backend", "MissingClass")])


def test_empty_specs_returns_empty_list() -> None:
    assert load_plugin_classes("pkg", []) == []
