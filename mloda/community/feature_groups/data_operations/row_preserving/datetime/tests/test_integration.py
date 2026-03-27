"""Integration tests for datetime extraction through mloda's full pipeline.

These tests verify that datetime extraction feature groups work end-to-end
through mloda's runtime, including plugin discovery, feature resolution,
and the PluginCollector mechanism.

Uses the shared DataOpsIntegrationTestBase from the testing library,
following the same pattern as the window_aggregation integration tests.
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

from mloda.community.feature_groups.data_operations.row_preserving.datetime.pyarrow_datetime import (
    PyArrowDateTimeExtraction,
)


class TestDateTimeIntegration(DataOpsIntegrationTestBase):
    """Standard integration tests inherited from the base class.

    Datetime operations use pattern-based matching (``<source>__<op>``),
    so ``match_feature_group_criteria`` succeeds with empty Options when
    the feature name contains the pattern. The ``test_match_rejects_missing_config``
    test is overridden to use a non-pattern name that requires config-based matching.
    """

    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowDateTimeExtraction

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "timestamp__year"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, None, 2023]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "timestamp__day"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return [1, 2, 3, 5, 6, 6, 7, 8, 9, 10, None, 12]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return [
            "timestamp__year",
            "timestamp__month",
            "timestamp__day",
            "timestamp__hour",
            "timestamp__minute",
            "timestamp__second",
            "timestamp__dayofweek",
            "timestamp__is_weekend",
            "timestamp__quarter",
        ]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["timestamp", "timestamp__weekday", "year", "timestamp__sum_groupby"]

    @classmethod
    def match_options(cls) -> Options:
        return Options()

    def test_match_rejects_missing_config(self) -> None:
        """Datetime uses pattern-based matching. A non-pattern name without config should fail."""
        options = Options()
        assert not PyArrowDateTimeExtraction.match_feature_group_criteria("my_custom_result", options)


class TestIntegrationMultipleFeatures:
    """Test multiple datetime extraction features in a single run_all call."""

    def test_year_and_month_together(self) -> None:
        """Request both year and month features in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowDateTimeExtraction}
        )

        f_year = Feature("timestamp__year", options=Options())
        f_month = Feature("timestamp__month", options=Options())

        results = mloda.run_all(
            [f_year, f_month],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        year_found = False
        month_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "timestamp__year" in table.column_names:
                year_col = table.column("timestamp__year").to_pylist()
                assert year_col == [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, None, 2023]
                year_found = True
            if "timestamp__month" in table.column_names:
                month_col = table.column("timestamp__month").to_pylist()
                assert month_col == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None, 1]
                month_found = True

        assert year_found, "timestamp__year result not found in any result table"
        assert month_found, "timestamp__month result not found in any result table"

    def test_dayofweek_and_is_weekend_together(self) -> None:
        """Request dayofweek and is_weekend features in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowDateTimeExtraction}
        )

        f_dow = Feature("timestamp__dayofweek", options=Options())
        f_weekend = Feature("timestamp__is_weekend", options=Options())

        results = mloda.run_all(
            [f_dow, f_weekend],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        dow_found = False
        weekend_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "timestamp__dayofweek" in table.column_names:
                dow_col = table.column("timestamp__dayofweek").to_pylist()
                assert dow_col == [6, 0, 1, 3, 4, 4, 5, 6, 0, 1, None, 3]
                dow_found = True
            if "timestamp__is_weekend" in table.column_names:
                weekend_col = table.column("timestamp__is_weekend").to_pylist()
                assert weekend_col == [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, None, 0]
                weekend_found = True

        assert dow_found, "timestamp__dayofweek result not found in any result table"
        assert weekend_found, "timestamp__is_weekend result not found in any result table"
