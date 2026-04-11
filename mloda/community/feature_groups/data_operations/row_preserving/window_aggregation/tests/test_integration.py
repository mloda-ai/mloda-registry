"""Integration tests for window aggregation through mloda's full pipeline.

These tests verify that window aggregation feature groups work end-to-end
through mloda's runtime, including plugin discovery, feature resolution,
and the PluginCollector mechanism.

Uses the shared DataOpsIntegrationTestBase from the testing library.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.testing.feature_groups.data_operations.mixins.mask_integration import MaskIntegrationTestMixin
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
    PyArrowWindowAggregation,
)


class TestWindowAggregationIntegration(MaskIntegrationTestMixin, DataOpsIntegrationTestBase):
    """Standard integration tests inherited from the base class."""

    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowWindowAggregation

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__sum_window"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return [25, 25, 25, 25, 140, 140, 140, 140, 70, 70, 70, -10]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__avg_window"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return [6.25, 6.25, 6.25, 6.25, 46.667, 46.667, 46.667, 46.667, 23.333, 23.333, 23.333, -10.0]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_int__sum_window", "value_int__avg_window", "value_int__count_window"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["value_int", "value_int__sum"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"]})

    @classmethod
    def use_approx(cls) -> bool:
        return True

    # -- MaskIntegrationTestMixin configuration --------------------------------

    @classmethod
    def mask_integration_feature_name(cls) -> str:
        return "value_int__sum_window"

    @classmethod
    def mask_integration_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def mask_integration_expected(cls) -> list[Any]:
        # category='X': A: 10+0=10, B: 60, C: 15, None: -10
        return [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, -10]

    @classmethod
    def mask_integration_complex_expected(cls) -> list[Any]:
        # category='X' AND value_int>=10: A: 10, B: 60, C: 15, None: None
        return [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, None]

    @classmethod
    def mask_integration_unmasked_expected(cls) -> list[Any]:
        return [25, 25, 25, 25, 140, 140, 140, 140, 70, 70, 70, -10]


class TestIntegrationMultipleFeatures:
    """Test multiple window aggregation features in a single run_all call."""

    def test_sum_and_avg_together(self) -> None:
        """Request both sum and avg features in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowWindowAggregation}
        )

        f_sum = Feature(
            "value_int__sum_window",
            options=Options(context={"partition_by": ["region"]}),
        )
        f_avg = Feature(
            "value_int__avg_window",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [f_sum, f_avg],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        sum_found = False
        avg_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__sum_window" in table.column_names:
                sum_col = table.column("value_int__sum_window").to_pylist()
                expected_sum = [25, 25, 25, 25, 140, 140, 140, 140, 70, 70, 70, -10]
                assert sum_col == expected_sum
                sum_found = True
            if "value_int__avg_window" in table.column_names:
                avg_col = table.column("value_int__avg_window").to_pylist()
                expected_avg = [
                    6.25,
                    6.25,
                    6.25,
                    6.25,
                    46.667,
                    46.667,
                    46.667,
                    46.667,
                    23.333,
                    23.333,
                    23.333,
                    -10.0,
                ]
                assert avg_col == pytest.approx(expected_avg, rel=1e-3)
                avg_found = True

        assert sum_found, "sum_window result not found in any result table"
        assert avg_found, "avg_window result not found in any result table"

    def test_different_aggregation_types(self) -> None:
        """Request min and max features in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowWindowAggregation}
        )

        f_min = Feature(
            "value_int__min_window",
            options=Options(context={"partition_by": ["region"]}),
        )
        f_max = Feature(
            "value_int__max_window",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [f_min, f_max],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        min_found = False
        max_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__min_window" in table.column_names:
                min_col = table.column("value_int__min_window").to_pylist()
                expected_min = [-5, -5, -5, -5, 30, 30, 30, 30, 15, 15, 15, -10]
                assert min_col == expected_min
                min_found = True
            if "value_int__max_window" in table.column_names:
                max_col = table.column("value_int__max_window").to_pylist()
                expected_max = [20, 20, 20, 20, 60, 60, 60, 60, 40, 40, 40, -10]
                assert max_col == expected_max
                max_found = True

        assert min_found, "min_window result not found in any result table"
        assert max_found, "max_window result not found in any result table"
