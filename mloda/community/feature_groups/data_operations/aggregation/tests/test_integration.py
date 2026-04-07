"""Integration tests for aggregation through mloda's full pipeline.

These tests verify that aggregation feature groups work end-to-end
through mloda's runtime, including plugin discovery, feature resolution,
and the PluginCollector mechanism.

Uses the shared DataOpsIntegrationTestBase from the testing library,
plus custom multi-feature tests for aggregation specifics.

Note: mloda's pipeline only returns requested feature columns, so
integration tests verify values as sets (not keyed by partition columns).
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

from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
    PyArrowAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation.reference import (
    ReferenceAggregation,
)


class TestAggregationIntegration(DataOpsIntegrationTestBase):
    """Standard integration tests inherited from the base class."""

    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowAggregation

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__sum_agg"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return [25, 140, 70, -10]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__avg_agg"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return [6.25, 46.667, 23.333, -10.0]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_int__sum_agg", "value_int__avg_agg", "value_int__count_agg"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["value_int", "value_int__sum"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"]})

    @classmethod
    def expected_row_count(cls) -> int:
        return 4

    @classmethod
    def compare_sorted(cls) -> bool:
        return True

    @classmethod
    def use_approx(cls) -> bool:
        return True


class TestIntegrationAdditionalAggTypes:
    """Test additional aggregation types through the full mloda pipeline.

    Aggregation reduces rows, so each feature must be requested
    in a separate run_all call (mloda cannot merge different row-count
    outputs without explicit Links).
    """

    def test_min_agg_through_pipeline(self) -> None:
        """Run value_int__min_agg through run_all and verify the result."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowAggregation})

        feature = Feature(
            "value_int__min_agg",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__min_agg" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4
        min_col = sorted(result_table.column("value_int__min_agg").to_pylist())
        assert min_col == sorted([-5, 30, 15, -10])

    def test_max_agg_through_pipeline(self) -> None:
        """Run value_int__max_agg through run_all and verify the result."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowAggregation})

        feature = Feature(
            "value_int__max_agg",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__max_agg" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4
        max_col = sorted(result_table.column("value_int__max_agg").to_pylist())
        assert max_col == sorted([20, 60, 40, -10])

    def test_std_agg_through_pipeline(self) -> None:
        """Run value_int__std_agg through run_all and verify the result."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowAggregation})

        feature = Feature(
            "value_int__std_agg",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__std_agg" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4

    def test_nunique_agg_through_pipeline(self) -> None:
        """Run value_int__nunique_agg through run_all and verify the result."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowAggregation})

        feature = Feature(
            "value_int__nunique_agg",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__nunique_agg" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4


class TestIntegrationReferenceOnlyAggTypes:
    """Pipeline test for aggregation types not natively supported by PyArrow.

    PyArrow lacks native grouped median and mode, so those operations were
    removed from PyArrowAggregation. The ReferenceAggregation (a test utility
    that computes in Python) still supports them. This test verifies that
    median continues to work end-to-end through mloda's pipeline, ensuring
    no regression for frameworks that do support it natively (Pandas, Polars,
    SQLite, DuckDB).
    """

    def test_median_agg_through_pipeline(self) -> None:
        """Run median aggregation through run_all using the reference implementation."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, ReferenceAggregation})

        feature = Feature(
            "value_int__median_agg",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__median_agg" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4

        median_col = sorted(result_table.column("value_int__median_agg").to_pylist())
        # Regions: A=[−5,0,10,20] median=5.0, B=[30,50,60,None] median=50.0,
        #          C=[15,15,40] median=15.0, None=[−10] median=−10.0
        assert median_col == pytest.approx(sorted([-10.0, 5.0, 15.0, 50.0]))


class TestIntegrationMultipleFeatures:
    """Test multiple aggregation features in separate pipeline runs.

    Aggregation reduces rows, so each feature must be requested
    in a separate run_all call (mloda cannot merge different row-count
    outputs without explicit Links). These tests verify that different
    aggregation types produce consistent, correct results from the
    same source data.
    """

    def test_sum_and_avg_consistent(self) -> None:
        """Run sum and avg separately and verify both produce correct results."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowAggregation})

        f_sum = Feature(
            "value_int__sum_agg",
            options=Options(context={"partition_by": ["region"]}),
        )
        sum_results = mloda.run_all(
            [f_sum],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        sum_table = None
        for table in sum_results:
            if isinstance(table, pa.Table) and "value_int__sum_agg" in table.column_names:
                sum_table = table
                break
        assert sum_table is not None
        sum_col = sorted(sum_table.column("value_int__sum_agg").to_pylist())
        assert sum_col == sorted([25, 140, 70, -10])

        f_avg = Feature(
            "value_int__avg_agg",
            options=Options(context={"partition_by": ["region"]}),
        )
        avg_results = mloda.run_all(
            [f_avg],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        avg_table = None
        for table in avg_results:
            if isinstance(table, pa.Table) and "value_int__avg_agg" in table.column_names:
                avg_table = table
                break
        assert avg_table is not None
        avg_col = sorted(avg_table.column("value_int__avg_agg").to_pylist())
        expected_avg = sorted([6.25, 46.667, 23.333, -10.0])
        assert avg_col == pytest.approx(expected_avg, rel=1e-3)

        assert sum_table.num_rows == avg_table.num_rows == 4

    def test_min_and_max_consistent(self) -> None:
        """Run min and max separately and verify both produce correct results."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowAggregation})

        f_min = Feature(
            "value_int__min_agg",
            options=Options(context={"partition_by": ["region"]}),
        )
        min_results = mloda.run_all(
            [f_min],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        min_table = None
        for table in min_results:
            if isinstance(table, pa.Table) and "value_int__min_agg" in table.column_names:
                min_table = table
                break
        assert min_table is not None
        min_col = sorted(min_table.column("value_int__min_agg").to_pylist())
        assert min_col == sorted([-5, 30, 15, -10])

        f_max = Feature(
            "value_int__max_agg",
            options=Options(context={"partition_by": ["region"]}),
        )
        max_results = mloda.run_all(
            [f_max],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        max_table = None
        for table in max_results:
            if isinstance(table, pa.Table) and "value_int__max_agg" in table.column_names:
                max_table = table
                break
        assert max_table is not None
        max_col = sorted(max_table.column("value_int__max_agg").to_pylist())
        assert max_col == sorted([20, 60, 40, -10])

        assert min_table.num_rows == max_table.num_rows == 4


def _extract_result_column(results: list[Any], feature_name: str) -> list[Any]:
    for table in results:
        if isinstance(table, pa.Table) and feature_name in table.column_names:
            result: list[Any] = table.column(feature_name).to_pylist()
            return result
    raise AssertionError(f"No result table with {feature_name} found")


class TestAggregationMaskIntegration:
    """Integration tests for aggregation with conditional mask."""

    def test_mask_single_condition(self) -> None:
        """Masked sum_agg through full pipeline: only category='X' rows contribute."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowAggregation})
        feature = Feature(
            "value_int__sum_agg",
            options=Options(context={"partition_by": ["region"], "mask": ("category", "equal", "X")}),
        )
        results = mloda.run_all([feature], compute_frameworks={PyArrowTable}, plugin_collector=plugin_collector)
        result_col = _extract_result_column(results, "value_int__sum_agg")
        assert sorted(v for v in result_col if v is not None) == [-10, 10, 15, 60]

    def test_mask_multiple_conditions(self) -> None:
        """AND-combined mask: category='X' AND value_int >= 10."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowAggregation})
        feature = Feature(
            "value_int__sum_agg",
            options=Options(
                context={
                    "partition_by": ["region"],
                    "mask": [("category", "equal", "X"), ("value_int", "greater_equal", 10)],
                }
            ),
        )
        results = mloda.run_all([feature], compute_frameworks={PyArrowTable}, plugin_collector=plugin_collector)
        result_col = _extract_result_column(results, "value_int__sum_agg")
        non_null = sorted(v for v in result_col if v is not None)
        assert non_null == [10, 15, 60]
        assert None in result_col  # None group has all values masked out

    def test_mask_is_in_operator(self) -> None:
        """is_in operator in an actual aggregation: only region in ['A', 'C'] contributes."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowAggregation})
        feature = Feature(
            "value_int__sum_agg",
            options=Options(
                context={
                    "partition_by": ["region"],
                    "mask": ("region", "is_in", ["A", "C"]),
                }
            ),
        )
        results = mloda.run_all([feature], compute_frameworks={PyArrowTable}, plugin_collector=plugin_collector)
        result_col = _extract_result_column(results, "value_int__sum_agg")
        non_null = sorted(v for v in result_col if v is not None)
        # Region A: all rows match, sum=[10,-5,0,20]=25
        # Region B: no rows match -> None
        # Region C: all rows match, sum=[15,15,40]=70
        # None: no rows match -> None
        assert non_null == [25, 70]
        assert result_col.count(None) == 2

    def test_mask_mode_aggregation(self) -> None:
        """Masked mode_agg through pipeline using reference implementation.

        Region C with mask category='Y': values [15, 40], mode depends on impl.
        Region A with mask category='X': values [10, 0], both unique.
        """
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, ReferenceAggregation})
        feature = Feature(
            "value_int__mode_agg",
            options=Options(context={"partition_by": ["region"], "mask": ("category", "equal", "X")}),
        )
        results = mloda.run_all([feature], compute_frameworks={PyArrowTable}, plugin_collector=plugin_collector)
        result_col = _extract_result_column(results, "value_int__mode_agg")
        # Verify results are not all None and contain valid values
        non_null = [v for v in result_col if v is not None]
        assert len(non_null) >= 3  # At least A, C, None group have X values

    def test_mask_fully_masked_partition_returns_none(self) -> None:
        """All rows masked out in every partition should produce None for sum."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowAggregation})
        feature = Feature(
            "value_int__sum_agg",
            options=Options(context={"partition_by": ["region"], "mask": ("category", "equal", "Z")}),
        )
        results = mloda.run_all([feature], compute_frameworks={PyArrowTable}, plugin_collector=plugin_collector)
        result_col = _extract_result_column(results, "value_int__sum_agg")
        assert all(v is None for v in result_col)

    def test_mask_greater_than_operator(self) -> None:
        """greater_than operator in an actual aggregation."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowAggregation})
        feature = Feature(
            "value_int__sum_agg",
            options=Options(
                context={
                    "partition_by": ["region"],
                    "mask": ("value_int", "greater_than", 10),
                }
            ),
        )
        results = mloda.run_all([feature], compute_frameworks={PyArrowTable}, plugin_collector=plugin_collector)
        result_col = _extract_result_column(results, "value_int__sum_agg")
        non_null = sorted(v for v in result_col if v is not None)
        # Region A: values > 10 -> [20], sum=20
        # Region B: values > 10 -> [50, 30, 60], sum=140
        # Region C: values > 10 -> [15, 15, 40], sum=70
        # None: values > 10 -> None (value_int=-10 does not pass)
        assert non_null == [20, 70, 140]
