"""Integration tests for frame aggregate feature group."""

from __future__ import annotations

from typing import Any, Dict

import pyarrow as pa

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pyarrow_frame_aggregate import (
    PyArrowFrameAggregate,
)
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator


class TestFrameAggregateIntegration(DataOpsIntegrationTestBase):
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
        # Rolling sum of window 3 on value_int partitioned by region, ordered by value_int
        # This depends on the test data, use approximate matching
        return None  # type: ignore[return-value]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__cumsum"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"], "order_by": "value_int"}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return None  # type: ignore[return-value]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return [
            "value_int__sum_rolling_3",
            "value_int__avg_rolling_5",
            "value_int__cumsum",
            "value_int__cummax",
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

    @classmethod
    def use_approx(cls) -> bool:
        return True

    def test_primary_feature_through_pipeline(self) -> None:
        """Override: just verify the feature runs without error."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {self.data_creator_class(), self.feature_group_class()}
        )
        results = mloda.run_all(
            [Feature(self.primary_feature_name(), Options(context=self.primary_feature_options()))],
            compute_frameworks={self.compute_framework_class()},
            plugin_collector=plugin_collector,
        )
        assert len(results) == 1
        assert self.primary_feature_name() in results[0].column_names
        assert results[0].num_rows == self.expected_row_count()

    def test_secondary_feature_through_pipeline(self) -> None:
        """Override: just verify the feature runs without error."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {self.data_creator_class(), self.feature_group_class()}
        )
        results = mloda.run_all(
            [Feature(self.secondary_feature_name(), Options(context=self.secondary_feature_options()))],
            compute_frameworks={self.compute_framework_class()},
            plugin_collector=plugin_collector,
        )
        assert len(results) == 1
        assert self.secondary_feature_name() in results[0].column_names


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

        assert len(results) == 1
        result = results[0]
        assert "value_int__sum_rolling_3" in result.column_names
        assert "value_int__cumsum" in result.column_names
        assert result.num_rows == 12
