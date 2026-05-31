"""Tests for PolarsLazyEma compute implementation.

Also includes an explicit pandas-vs-polars cross-compute agreement test: there
is no live PyArrow reference oracle for EMA, so this directly compares the two
native backends against each other (in addition to both being asserted against
the same pinned literals via ``EmaTestBase``).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

pytest.importorskip("polars")
pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.ema.polars_lazy_ema import (
    PolarsLazyEma,
)
from mloda.community.feature_groups.data_operations.row_preserving.ema.pandas_ema import (
    PandasEma,
)
from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.ema.ema import (
    EmaTestBase,
    _create_ema_arrow_table,
)


class TestPolarsLazyEma(PolarsLazyTestMixin, EmaTestBase):
    """All value/semantics/error tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyEma


class TestPandasPolarsAgreement:
    """The two native backends must produce identical EMA output.

    This is the cross-compute check that replaces the (impossible) PyArrow
    reference oracle: pandas and polars share the same span -> alpha mapping,
    so their results must agree elementwise (null-aware) on the same fixture.
    """

    @staticmethod
    def _norm_null(value: Any) -> Any:
        """Normalize pandas NaN (and polars None) to a single null sentinel.

        The bare ``extract_column`` helper does NOT convert pandas NaN to None,
        so null rows surface as ``nan`` from pandas but ``None`` from polars.
        Normalize both before comparing so the test checks numeric agreement,
        not a NaN-vs-None mismatch on null-input rows.
        """
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    @pytest.mark.parametrize("span", [2, 3])
    def test_pandas_and_polars_agree(self, span: int) -> None:
        import polars as pl

        arrow_table = _create_ema_arrow_table()
        feature_name = f"value__ema_{span}"
        fs = make_feature_set(feature_name, partition_by=["region"], order_by="ts")

        pandas_result = PandasEma.calculate_feature(arrow_table.to_pandas(), fs)
        pandas_col = [self._norm_null(v) for v in extract_column(pandas_result, feature_name)]

        lazy = pl.from_arrow(arrow_table)
        assert isinstance(lazy, pl.DataFrame)
        polars_result = PolarsLazyEma.calculate_feature(lazy.lazy(), fs)
        polars_col = [self._norm_null(v) for v in extract_column(polars_result, feature_name)]

        assert len(pandas_col) == len(polars_col)
        for i, (p, q) in enumerate(zip(pandas_col, polars_col)):
            if p is None or q is None:
                assert p is None and q is None, f"row {i}: pandas={p!r} polars={q!r}"
            else:
                assert float(p) == pytest.approx(float(q), rel=1e-9), f"row {i}: pandas={p!r} polars={q!r}"
