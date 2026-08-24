"""Tests for ReferenceWindowAggregation (mloda.testing reference implementation)."""

from __future__ import annotations

from typing import Any


class TestReferenceWindowAggregationNoFutureWarning:
    """first/last with partition_by must not raise pyarrow's null_placement FutureWarning."""

    @staticmethod
    def _agg_no_warning(agg_type: str, expected: list[Any]) -> None:
        import warnings

        from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set
        from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.reference import (
            ReferenceWindowAggregation,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        fs = make_feature_set(f"value_int__{agg_type}_window", ["region"], order_by="value_int")

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = ReferenceWindowAggregation.calculate_feature(arrow_table, fs)

        result_col = extract_column(result, f"value_int__{agg_type}_window")
        assert result_col == expected

    def test_first_window_no_future_warning(self) -> None:
        from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
            EXPECTED_FIRST_BY_REGION,
        )

        self._agg_no_warning("first", EXPECTED_FIRST_BY_REGION)

    def test_last_window_no_future_warning(self) -> None:
        from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
            EXPECTED_LAST_BY_REGION,
        )

        self._agg_no_warning("last", EXPECTED_LAST_BY_REGION)
