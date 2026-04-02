"""Integration tests for offset through mloda's full pipeline.

Uses the ReferenceOffset implementation (a test utility that accepts PyArrow
tables and computes in Python) because PyArrow lacks native window functions.
The tests verify that offset operations work end-to-end through mloda's
runtime, including plugin discovery, feature resolution, and PluginCollector.
"""

from __future__ import annotations

import pyarrow as pa

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.testing.feature_groups.data_operations.row_preserving.offset.reference import ReferenceOffset


class TestIntegrationBasic:
    def test_lag_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, ReferenceOffset})

        feature = Feature(
            "value_int__lag_1_offset",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__lag_1_offset" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 12

    def test_first_value_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, ReferenceOffset})

        feature = Feature(
            "value_int__first_value_offset",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__first_value_offset" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 12


class TestIntegrationPluginDiscovery:
    def test_feature_group_is_discoverable(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, ReferenceOffset})
        assert plugin_collector.applicable_feature_group_class(ReferenceOffset)

    def test_match_feature_group_criteria(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        assert ReferenceOffset.match_feature_group_criteria("value_int__lag_1_offset", options)
        assert ReferenceOffset.match_feature_group_criteria("value_int__first_value_offset", options)

    def test_match_rejects_missing_order_by(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        assert not ReferenceOffset.match_feature_group_criteria("value_int__lag_1_offset", options)
