"""Tests for PandasDateTimeExtraction compute implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import DATETIME_OPS
from mloda.community.feature_groups.data_operations.row_preserving.datetime.pandas_datetime import (
    PandasDateTimeExtraction,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_preserving.datetime.datetime import (
    DateTimeTestBase,
)


class TestPandasDateTimeExtraction(PandasTestMixin, DateTimeTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasDateTimeExtraction


class TestPandasDateTimeExtractionNullableInt(PandasNullableIntTestMixin, TestPandasDateTimeExtraction):
    """Runs the datetime suite on nullable pandas int columns."""


class TestPandasDateTimeOutputDtype:
    """Pins the pandas-only output dtype: always nullable Int64, <NA> on null source rows."""

    @pytest.mark.parametrize("op", sorted(DATETIME_OPS))
    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([datetime(2023, 5, 1, 10), None], id="with_null"),
            pytest.param([datetime(2023, 5, 1, 10), datetime(2023, 5, 6)], id="no_null"),
        ],
    )
    def test_output_dtype_pinned(self, op: str, values: list[datetime | None]) -> None:
        df = pd.DataFrame({"ts": values})
        col = f"ts__{op}"
        result = PandasDateTimeExtraction._compute_datetime(df, col, "ts", op)
        assert str(result[col].dtype) == "Int64"
        if values[1] is None:
            assert result[col].iloc[1] is pd.NA


class TestPandasDateTimeNoDeprecatedAccessor:
    """dayofweek and is_weekend must not call Series.dt.dayofweek (deprecated in pandas 3.1)."""

    @pytest.mark.parametrize("op", ["dayofweek", "is_weekend"])
    def test_no_pandas4_warning(self, op: str) -> None:
        import warnings

        pandas4_warning = getattr(pd.errors, "Pandas4Warning", None)
        if pandas4_warning is None:
            pytest.skip("pandas without Pandas4Warning")
        df = pd.DataFrame({"ts": [datetime(2023, 5, 6)]})
        col = f"ts__{op}"
        with warnings.catch_warnings():
            warnings.simplefilter("error", pandas4_warning)
            PandasDateTimeExtraction._compute_datetime(df, col, "ts", op)
