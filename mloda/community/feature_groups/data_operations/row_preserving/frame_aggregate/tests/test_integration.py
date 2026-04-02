"""Integration tests for frame aggregate feature group.

Uses PandasFrameAggregate as the backend (PyArrow implementation was removed
because PyArrow lacks native window frame functions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pd = pytest.importorskip("pandas")

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
    PandasFrameAggregate,
)

if TYPE_CHECKING:
    import pandas


def _extract_column(df: pandas.DataFrame, col: str) -> list[Any]:
    """Extract a column as a Python list with None for NaN."""
    return [None if pd.isna(v) else v for v in df[col].tolist()]


def _run_single_feature(name: str, options_context: dict[str, Any]) -> pandas.DataFrame:
    """Run a single feature through the pipeline and return the result DataFrame."""
    plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, PandasFrameAggregate})
    feature = Feature(name, options=Options(context=options_context))
    results = mloda.run_all(
        [feature],
        compute_frameworks={PandasDataFrame},
        plugin_collector=plugin_collector,
    )
    assert len(results) >= 1
    for table in results:
        if isinstance(table, pd.DataFrame) and name in table.columns:
            return table
    raise AssertionError(f"No result DataFrame with {name} found")


class TestFrameAggregateIntegration:
    """Standard integration tests (previously inherited from DataOpsIntegrationTestBase)."""

    _opts = {"partition_by": ["region"], "order_by": "value_int"}

    # Rolling sum (window 3) on value_int, partitioned by region, ordered by value_int.
    _primary_expected = [5, -5, -5, 30, 110, 80, 30, 140, 15, 30, 70, -10]
    # Cumulative sum on value_int, partitioned by region, ordered by value_int.
    _secondary_expected = [5, -5, -5, 25, 140, 80, 30, 140, 15, 30, 70, -10]

    def test_primary_feature_through_pipeline(self) -> None:
        result = _run_single_feature("value_int__sum_rolling_3", self._opts)
        assert len(result) == 12
        assert _extract_column(result, "value_int__sum_rolling_3") == self._primary_expected

    def test_secondary_feature_through_pipeline(self) -> None:
        result = _run_single_feature("value_int__cumsum", self._opts)
        assert len(result) == 12
        assert _extract_column(result, "value_int__cumsum") == self._secondary_expected

    def test_feature_group_is_discoverable(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, PandasFrameAggregate})
        assert plugin_collector.applicable_feature_group_class(PandasFrameAggregate)
        assert plugin_collector.applicable_feature_group_class(PandasDataOpsTestDataCreator)

    def test_disabled_feature_group_blocks_execution(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator})
        feature = Feature("value_int__sum_rolling_3", options=Options(context=self._opts))
        with pytest.raises(ValueError):
            mloda.run_all(
                [feature],
                compute_frameworks={PandasDataFrame},
                plugin_collector=plugin_collector,
            )

    def test_match_feature_group_criteria_valid(self) -> None:
        options = Options(context=self._opts)
        valid_names = [
            "value_int__sum_rolling_3",
            "value_int__avg_rolling_5",
            "value_int__cumsum",
            "value_int__cummax",
            "value_int__cumavg",
            "value_int__expanding_avg",
        ]
        for name in valid_names:
            assert PandasFrameAggregate.match_feature_group_criteria(name, options), (
                f"Expected {name} to match PandasFrameAggregate"
            )

    def test_match_rejects_invalid_feature_names(self) -> None:
        options = Options(context=self._opts)
        invalid_names = [
            "value_int__sum_window",
            "value_int",
            "plain_feature",
        ]
        for name in invalid_names:
            assert not PandasFrameAggregate.match_feature_group_criteria(name, options), (
                f"Expected {name} to NOT match PandasFrameAggregate"
            )

    def test_match_rejects_missing_config(self) -> None:
        options = Options()
        assert not PandasFrameAggregate.match_feature_group_criteria("value_int__sum_rolling_3", options)


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
                col = _extract_column(table, "value_int__sum_rolling_3")
                assert col == [5, -5, -5, 30, 110, 80, 30, 140, 15, 30, 70, -10]
                rolling_found = True
            if "value_int__cumsum" in table.columns:
                col = _extract_column(table, "value_int__cumsum")
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
                col = _extract_column(table, "value_int__expanding_avg")
                for i, (actual, expected) in enumerate(zip(col, expected_expanding_avg)):
                    assert actual == pytest.approx(expected, rel=1e-3), f"expanding_avg row {i}: {actual} != {expected}"
                expanding_found = True
            if "value_int__min_rolling_2" in table.columns:
                col = _extract_column(table, "value_int__min_rolling_2")
                assert col == expected_rolling_min_2
                rolling_min_found = True

        assert expanding_found, "expanding_avg result not found in any result DataFrame"
        assert rolling_min_found, "min_rolling_2 result not found in any result DataFrame"
