"""Integration tests for binning through mloda's full pipeline."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.testing.feature_groups.data_operations.row_preserving.binning import (
    EXPECTED_BIN_3,
    EXPECTED_QBIN_3,
)
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.binning.pyarrow_binning import (
    PyArrowBinning,
)


class TestBinningIntegration(DataOpsIntegrationTestBase):
    """Standard integration tests inherited from the base class."""

    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowBinning

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__bin_3"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return EXPECTED_BIN_3

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__bin_5"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        # value_int = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]
        # min=-10, max=60, range=70, bin_width=70/5=14
        # val=10:  (10-(-10))/14 = 20/14 = 1.428 -> 1
        # val=-5:  (5)/14 = 0.357 -> 0
        # val=0:   (10)/14 = 0.714 -> 0
        # val=20:  (30)/14 = 2.142 -> 2
        # val=None: None
        # val=50:  (60)/14 = 4.285 -> 4
        # val=30:  (40)/14 = 2.857 -> 2
        # val=60:  (70)/14 = 5.0 -> min(5, 4) = 4
        # val=15:  (25)/14 = 1.785 -> 1
        # val=15:  same -> 1
        # val=40:  (50)/14 = 3.571 -> 3
        # val=-10: (0)/14 = 0.0 -> 0
        return [1, 0, 0, 2, None, 4, 2, 4, 1, 1, 3, 0]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_int__bin_3", "value_int__bin_5", "value_int__qbin_3"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["value_int", "value_int__bin"]

    @classmethod
    def match_options(cls) -> Options:
        return Options()

    def test_match_rejects_missing_config(self) -> None:
        """Override: binning requires no config for pattern-based features.

        Unlike window_aggregation (which needs partition_by), binning
        extracts all parameters from the feature name pattern itself.
        An empty Options() still matches ``value_int__bin_3``.
        """
        options = Options()
        fg_cls = self.feature_group_class()
        assert fg_cls.match_feature_group_criteria(self.primary_feature_name(), options)  # type: ignore[attr-defined]


class TestIntegrationMultipleFeatures:
    """Test multiple binning features in a single run_all call."""

    def test_bin_3_and_bin_5_together(self) -> None:
        """Request both bin_3 and bin_5 features in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowBinning})

        f_bin3 = Feature("value_int__bin_3", options=Options())
        f_bin5 = Feature("value_int__bin_5", options=Options())

        results = mloda.run_all(
            [f_bin3, f_bin5],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        bin3_found = False
        bin5_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__bin_3" in table.column_names:
                result_col = table.column("value_int__bin_3").to_pylist()
                assert result_col == EXPECTED_BIN_3
                bin3_found = True
            if "value_int__bin_5" in table.column_names:
                bin5_found = True

        assert bin3_found, "value_int__bin_3 result not found"
        assert bin5_found, "value_int__bin_5 result not found"

    def test_qbin_3_integration(self) -> None:
        """Request qbin_3 feature through the full pipeline."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowBinning})

        f_qbin3 = Feature("value_int__qbin_3", options=Options())

        results = mloda.run_all(
            [f_qbin3],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        qbin3_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__qbin_3" in table.column_names:
                result_col = table.column("value_int__qbin_3").to_pylist()
                assert result_col == EXPECTED_QBIN_3
                qbin3_found = True

        assert qbin3_found, "value_int__qbin_3 result not found"

    def test_bin_and_qbin_together(self) -> None:
        """Request both bin and qbin features in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowBinning})

        f_bin3 = Feature("value_int__bin_3", options=Options())
        f_qbin3 = Feature("value_int__qbin_3", options=Options())

        results = mloda.run_all(
            [f_bin3, f_qbin3],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        bin3_found = False
        qbin3_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__bin_3" in table.column_names:
                result_col = table.column("value_int__bin_3").to_pylist()
                assert result_col == EXPECTED_BIN_3
                bin3_found = True
            if "value_int__qbin_3" in table.column_names:
                result_col = table.column("value_int__qbin_3").to_pylist()
                assert result_col == EXPECTED_QBIN_3
                qbin3_found = True

        assert bin3_found, "value_int__bin_3 result not found"
        assert qbin3_found, "value_int__qbin_3 result not found"
