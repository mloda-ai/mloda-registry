"""Integration tests for percentile through mloda's full pipeline.

Uses the ReferencePercentile implementation (a test utility that accepts PyArrow
tables and computes in Python) because PyArrow lacks native grouped percentile.
The tests verify that percentile operations work end-to-end through mloda's
runtime, including plugin discovery, feature resolution, and PluginCollector.
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

from mloda.testing.feature_groups.data_operations.row_preserving.percentile.reference import (
    ReferencePercentile,
)


class TestPercentileIntegration(MaskIntegrationTestMixin, DataOpsIntegrationTestBase):
    @classmethod
    def feature_group_class(cls) -> type:
        return ReferencePercentile

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__p50_percentile"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return [5.0, 5.0, 5.0, 5.0, 50.0, 50.0, 50.0, 50.0, 15.0, 15.0, 15.0, -10.0]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__p75_percentile"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return [12.5, 12.5, 12.5, 12.5, 55.0, 55.0, 55.0, 55.0, 27.5, 27.5, 27.5, -10.0]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_int__p25_percentile", "value_int__p50_percentile", "value_int__p75_percentile"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["value_int", "value_int__p50"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"]})

    @classmethod
    def use_approx(cls) -> bool:
        return True

    # -- MaskIntegrationTestMixin configuration --------------------------------

    @classmethod
    def mask_integration_feature_name(cls) -> str:
        return "value_int__p50_percentile"

    @classmethod
    def mask_integration_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def mask_integration_expected(cls) -> list[Any]:
        # category='X': A=[10,0] p50=5.0, B=[60] p50=60.0, C=[15] p50=15.0, None=[-10] p50=-10.0
        return [5.0, 5.0, 5.0, 5.0, 60.0, 60.0, 60.0, 60.0, 15.0, 15.0, 15.0, -10.0]

    @classmethod
    def mask_integration_complex_expected(cls) -> list[Any]:
        # category='X' AND value_int>=10: A=[10] p50=10, B=[60] p50=60, C=[15] p50=15, None=[] p50=None
        return [10.0, 10.0, 10.0, 10.0, 60.0, 60.0, 60.0, 60.0, 15.0, 15.0, 15.0, None]

    @classmethod
    def mask_integration_unmasked_expected(cls) -> list[Any]:
        return [5.0, 5.0, 5.0, 5.0, 50.0, 50.0, 50.0, 50.0, 15.0, 15.0, 15.0, -10.0]

    @classmethod
    def mask_integration_use_approx(cls) -> bool:
        return True


class TestIntegrationMultipleFeatures:
    def test_p50_and_p75_together(self) -> None:
        """Request both p50 and p75 in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, ReferencePercentile})

        f_p50 = Feature(
            "value_int__p50_percentile",
            options=Options(context={"partition_by": ["region"]}),
        )
        f_p75 = Feature(
            "value_int__p75_percentile",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [f_p50, f_p75],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        p50_found = False
        p75_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__p50_percentile" in table.column_names:
                p50_col = table.column("value_int__p50_percentile").to_pylist()
                expected_p50 = [5.0, 5.0, 5.0, 5.0, 50.0, 50.0, 50.0, 50.0, 15.0, 15.0, 15.0, -10.0]
                assert p50_col == pytest.approx(expected_p50, rel=1e-3)
                p50_found = True
            if "value_int__p75_percentile" in table.column_names:
                p75_col = table.column("value_int__p75_percentile").to_pylist()
                expected_p75 = [12.5, 12.5, 12.5, 12.5, 55.0, 55.0, 55.0, 55.0, 27.5, 27.5, 27.5, -10.0]
                assert p75_col == pytest.approx(expected_p75, rel=1e-3)
                p75_found = True

        assert p50_found, "p50_percentile result not found in any result table"
        assert p75_found, "p75_percentile result not found in any result table"

    def test_different_percentiles(self) -> None:
        """Request p25 and p100 in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, ReferencePercentile})

        f_p25 = Feature(
            "value_int__p25_percentile",
            options=Options(context={"partition_by": ["region"]}),
        )
        f_p100 = Feature(
            "value_int__p100_percentile",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [f_p25, f_p100],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        p25_found = False
        p100_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__p25_percentile" in table.column_names:
                p25_col = table.column("value_int__p25_percentile").to_pylist()
                expected_p25 = [-1.25, -1.25, -1.25, -1.25, 40.0, 40.0, 40.0, 40.0, 15.0, 15.0, 15.0, -10.0]
                assert p25_col == pytest.approx(expected_p25, rel=1e-3)
                p25_found = True
            if "value_int__p100_percentile" in table.column_names:
                p100_col = table.column("value_int__p100_percentile").to_pylist()
                expected_p100 = [20.0, 20.0, 20.0, 20.0, 60.0, 60.0, 60.0, 60.0, 40.0, 40.0, 40.0, -10.0]
                assert p100_col == pytest.approx(expected_p100, rel=1e-3)
                p100_found = True

        assert p25_found, "p25_percentile result not found in any result table"
        assert p100_found, "p100_percentile result not found in any result table"
