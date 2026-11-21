"""Tests for PandasStringOps compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.string import pandas_string
from mloda.community.feature_groups.data_operations.string.pandas_string import (
    PandasStringOps,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import (
    PandasNullableIntTestMixin,
    PandasObjectStringTestMixin,
    PandasPythonStringTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.string.string import (
    StringTestBase,
)


class TestPandasStringOps(PandasTestMixin, StringTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasStringOps


class TestPandasStringOpsNullableInt(PandasNullableIntTestMixin, TestPandasStringOps):
    """Runs the string suite on nullable pandas int columns."""


class TestPandasStringOpsObjectString(PandasObjectStringTestMixin, TestPandasStringOps):
    """Runs the string suite on object string storage."""

    def test_upper_lower_raise_without_pyarrow_while_trim_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pandas_string, "_pyarrow_available", lambda: False)
        for op in ("upper", "lower"):
            with pytest.raises(ValueError, match="pyarrow"):
                calculate_with(self.implementation_class(), self.test_data, make_feature_set(f"name__{op}"))
        result = calculate_with(self.implementation_class(), self.test_data, make_feature_set("name__trim"))
        assert self.extract_column(result, "name__trim") == self.expected_trim()


class TestPandasStringOpsPythonString(PandasPythonStringTestMixin, TestPandasStringOps):
    """Runs the string suite on string[python] storage."""
