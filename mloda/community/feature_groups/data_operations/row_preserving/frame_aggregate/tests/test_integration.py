"""Integration tests for frame aggregate feature group.

Uses the shared DataOpsIntegrationTestBase from the testing library.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pyarrow_frame_aggregate import (
    PyArrowFrameAggregate,
)


class TestFrameAggregateIntegration(DataOpsIntegrationTestBase):
    """Standard integration tests inherited from the base class.

    Uses the unified DataOpsIntegrationTestBase framework with real expected
    values so base-class test methods (primary/secondary pipeline, discovery,
    pattern matching) run without overrides.
    """

    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowFrameAggregate

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__sum_rolling_3"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"], "order_by": "value_int"}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        # Rolling sum (window 3) on value_int, partitioned by region, ordered by value_int.
        # Region A sorted: -5, 0, 10, 20
        # Region B sorted: 30, 50, 60, None
        # Region C sorted: 15, 15, 40
        # Region None: -10
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
            "value_int__sum_groupby",
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
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowFrameAggregate}
        )

        features: list[Feature | str] = [
            Feature("value_int__sum_rolling_3", Options(context={"partition_by": ["region"], "order_by": "value_int"})),
            Feature("value_int__cumsum", Options(context={"partition_by": ["region"], "order_by": "value_int"})),
        ]

        results = mloda.run_all(
            features,
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        rolling_found = False
        cumsum_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__sum_rolling_3" in table.column_names:
                col = table.column("value_int__sum_rolling_3").to_pylist()
                assert col == [5, -5, -5, 30, 110, 80, 30, 140, 15, 30, 70, -10]
                rolling_found = True
            if "value_int__cumsum" in table.column_names:
                col = table.column("value_int__cumsum").to_pylist()
                assert col == [5, -5, -5, 25, 140, 80, 30, 140, 15, 30, 70, -10]
                cumsum_found = True

        assert rolling_found, "sum_rolling_3 result not found in any result table"
        assert cumsum_found, "cumsum result not found in any result table"

    def test_expanding_and_rolling_different_aggs(self) -> None:
        """Request expanding avg and rolling min in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowFrameAggregate}
        )

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
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        # Expected expanding avg: see FrameAggregateTestBase EXPECTED_EXPANDING_AVG
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
        # Expected rolling min (window 2) on value_int, partitioned by region, ordered by value_int.
        expected_rolling_min_2 = [0, -5, -5, 10, 60, 30, 30, 50, 15, 15, 15, -10]

        expanding_found = False
        rolling_min_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__expanding_avg" in table.column_names:
                col = table.column("value_int__expanding_avg").to_pylist()
                for i, (actual, expected) in enumerate(zip(col, expected_expanding_avg)):
                    assert actual == pytest.approx(expected, rel=1e-3), f"expanding_avg row {i}: {actual} != {expected}"
                expanding_found = True
            if "value_int__min_rolling_2" in table.column_names:
                col = table.column("value_int__min_rolling_2").to_pylist()
                assert col == expected_rolling_min_2
                rolling_min_found = True

        assert expanding_found, "expanding_avg result not found in any result table"
        assert rolling_min_found, "min_rolling_2 result not found in any result table"
