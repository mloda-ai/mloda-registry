"""Integration tests for frame aggregate feature group.

Uses PandasFrameAggregate as the backend (PyArrow implementation was removed
because PyArrow lacks native window frame functions). Inherits from
DataOpsIntegrationTestBase with Pandas adapter overrides.
"""

from __future__ import annotations

from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
    PandasFrameAggregate,
)


def _extract_pandas_column(df: Any, col: str) -> list[Any]:
    """Extract a column from a DataFrame as a Python list with None for NaN."""
    return [None if pd.isna(v) else v for v in df[col].tolist()]


class TestFrameAggregateIntegration(DataOpsIntegrationTestBase):
    """Standard integration tests inherited from the base class.

    Overrides the framework adapter methods to work with Pandas DataFrames
    instead of the default PyArrow Tables.
    """

    # -- Pandas adapter overrides ---------------------------------------------

    def _is_framework_result(self, obj: Any, feature_name: str) -> bool:
        return isinstance(obj, pd.DataFrame) and feature_name in obj.columns

    def _extract_result_column(self, result: Any, feature_name: str) -> list[Any]:
        return _extract_pandas_column(result, feature_name)

    def _get_result_row_count(self, result: Any) -> int:
        return len(result)

    # -- Abstract method implementations --------------------------------------

    @classmethod
    def feature_group_class(cls) -> type:
        return PandasFrameAggregate

    @classmethod
    def data_creator_class(cls) -> type:
        return PandasDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PandasDataFrame

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__sum_rolling_3"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"], "order_by": "value_int"}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        # Rolling sum (window 3) on value_int, partitioned by region, ordered by value_int.
        return [5, -5, -5, 30, 110, 80, 30, 140, 15, 30, 70, -10]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__cumsum"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"], "order_by": "value_int"}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        # Cumulative sum on value_int, partitioned by region, ordered by value_int.
        return [5, -5, -5, 25, 140, 80, 30, 140, 15, 30, 70, -10]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return [
            "value_int__sum_rolling_3",
            "value_int__avg_rolling_5",
            "value_int__cumsum",
            "value_int__cummax",
            "value_int__cumavg",
            "value_int__expanding_avg",
        ]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return [
            "value_int__sum_window",
            "value_int",
            "plain_feature",
        ]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "value_int"})

    @classmethod
    def expected_row_count(cls) -> int:
        return 12


class TestFrameAggregateMultiFeature:
    """Test multiple frame aggregate features in a single pipeline run."""

    def test_rolling_and_cumulative_together(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, PandasFrameAggregate})

        features: list[Feature | str] = [
            Feature("value_int__sum_rolling_3", Options(context={"partition_by": ["region"], "order_by": "value_int"})),
            Feature("value_int__cumsum", Options(context={"partition_by": ["region"], "order_by": "value_int"})),
        ]

        results = mloda.run_all(
            features,
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        rolling_found = False
        cumsum_found = False
        for table in results:
            if not isinstance(table, pd.DataFrame):
                continue
            if "value_int__sum_rolling_3" in table.columns:
                col = _extract_pandas_column(table, "value_int__sum_rolling_3")
                assert col == [5, -5, -5, 30, 110, 80, 30, 140, 15, 30, 70, -10]
                rolling_found = True
            if "value_int__cumsum" in table.columns:
                col = _extract_pandas_column(table, "value_int__cumsum")
                assert col == [5, -5, -5, 25, 140, 80, 30, 140, 15, 30, 70, -10]
                cumsum_found = True

        assert rolling_found, "sum_rolling_3 result not found in any result DataFrame"
        assert cumsum_found, "cumsum result not found in any result DataFrame"

    def test_expanding_and_rolling_different_aggs(self) -> None:
        """Request expanding avg and rolling min in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, PandasFrameAggregate})

        features: list[Feature | str] = [
            Feature(
                "value_int__expanding_avg",
                Options(context={"partition_by": ["region"], "order_by": "value_int"}),
            ),
            Feature(
                "value_int__min_rolling_2",
                Options(context={"partition_by": ["region"], "order_by": "value_int"}),
            ),
        ]

        results = mloda.run_all(
            features,
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        expected_expanding_avg = [
            5.0 / 3.0,
            -5.0,
            -2.5,
            6.25,
            140.0 / 3.0,
            40.0,
            30.0,
            140.0 / 3.0,
            15.0,
            15.0,
            70.0 / 3.0,
            -10.0,
        ]
        expected_rolling_min_2 = [0, -5, -5, 10, 60, 30, 30, 50, 15, 15, 15, -10]

        expanding_found = False
        rolling_min_found = False
        for table in results:
            if not isinstance(table, pd.DataFrame):
                continue
            if "value_int__expanding_avg" in table.columns:
                col = _extract_pandas_column(table, "value_int__expanding_avg")
                for i, (actual, expected) in enumerate(zip(col, expected_expanding_avg)):
                    assert actual == pytest.approx(expected, rel=1e-3), f"expanding_avg row {i}: {actual} != {expected}"
                expanding_found = True
            if "value_int__min_rolling_2" in table.columns:
                col = _extract_pandas_column(table, "value_int__min_rolling_2")
                assert col == expected_rolling_min_2
                rolling_min_found = True

        assert expanding_found, "expanding_avg result not found in any result DataFrame"
        assert rolling_min_found, "min_rolling_2 result not found in any result DataFrame"
