"""Tests for EmaFeatureGroup base class.

Covers the arity contract of the scalar ``order_by`` key: core unwraps a
one-element container when it reads a property value, so ``("timestamp",)`` is
valid caller syntax for one column and must dispatch to ``"timestamp"`` rather
than to the container's string form. Match tests use ``PandasEma`` (the ema
convention, mirroring ``TestEmaMatchFeatureGroupCriteria`` in test_integration).
"""

from __future__ import annotations

from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.ema.base import EmaFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.ema.pandas_ema import PandasEma


class TestOrderByArity:
    """``order_by`` is a scalar key: one column, bare or in a single-element container."""

    def _options(self, order_by: Any) -> Options:
        return Options(context={"order_by": order_by, "partition_by": ["region"]})

    @pytest.mark.parametrize("order_by", ["timestamp", ("timestamp",), ["timestamp"]])
    def test_singleton_matches(self, order_by: Any) -> None:
        assert PandasEma.match_feature_group_criteria("value_float__ema_2", self._options(order_by)) is True

    @pytest.mark.parametrize("order_by", [["timestamp", "region"], ("timestamp", "region")])
    def test_multi_element_rejected(self, order_by: Any) -> None:
        assert PandasEma.match_feature_group_criteria("value_float__ema_2", self._options(order_by)) is False

    def test_wrong_type_rejected(self) -> None:
        assert PandasEma.match_feature_group_criteria("value_float__ema_2", self._options(123)) is False

    @pytest.mark.parametrize("order_by", ["timestamp", ("timestamp",), ["timestamp"]])
    def test_extract_order_by_unwraps_to_bare_column(self, order_by: Any) -> None:
        feature = Feature("value_float__ema_2", options=self._options(order_by))
        assert EmaFeatureGroup._extract_order_by(feature) == "timestamp"

    def test_extract_order_by_raises_when_missing(self) -> None:
        feature = Feature("value_float__ema_2", options=Options())
        with pytest.raises(ValueError, match="order_by"):
            EmaFeatureGroup._extract_order_by(feature)

    def test_singleton_order_by_dispatches_like_bare(self) -> None:
        def _compute(order_by: Any) -> list[Any]:
            df = pd.DataFrame(DataOperationsTestDataCreator.get_raw_data())
            fs = FeatureSet()
            fs.add(Feature("value_float__ema_2", options=self._options(order_by)))
            return list(PandasEma.calculate_feature(df, fs)["value_float__ema_2"].astype(str))

        assert _compute(("timestamp",)) == _compute("timestamp")
