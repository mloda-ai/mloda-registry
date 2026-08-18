"""Unit tests for manifest_utils optional-backend import skipping."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

from mloda.community.feature_groups.data_operations import manifest_utils
from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes


def _raise_module_not_found(missing_root: str) -> Any:
    def _fake_import_module(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        raise ModuleNotFoundError(f"No module named {missing_root!r}", name=missing_root)

    return _fake_import_module


class TestLoadPluginClasses:
    def test_skips_pandas_rooted_module_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(manifest_utils.importlib, "import_module", _raise_module_not_found("pandas"))

        result = load_plugin_classes("some.package", [("pandas_backend", "SomeClass")])

        assert result == []

    def test_skips_numpy_rooted_module_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # pandas_binning.py imports numpy directly before pandas, so a missing
        # numpy install fails the import with exc.name == "numpy" before the
        # pandas import is ever reached. numpy should be skipped exactly like
        # the other optional backend roots (pandas, polars, duckdb, pyarrow).
        monkeypatch.setattr(manifest_utils.importlib, "import_module", _raise_module_not_found("numpy"))

        result = load_plugin_classes("some.package", [("pandas_backend", "SomeClass")])

        assert result == []

    def test_reraises_unrelated_module_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            manifest_utils.importlib,
            "import_module",
            _raise_module_not_found("totally_bogus_required_module"),
        )

        with pytest.raises(ModuleNotFoundError, match="totally_bogus_required_module"):
            load_plugin_classes("some.package", [("pandas_backend", "SomeClass")])
