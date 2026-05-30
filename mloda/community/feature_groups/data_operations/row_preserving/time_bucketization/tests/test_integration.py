"""Integration tests for time bucketization through mloda's full pipeline.

Verifies that the feature group resolves through ``mloda.run_all`` with a
``PluginCollector`` (string-pattern and option-based features both).

The canonical 12-row dataset's timestamps are all at midnight on January 2023
days, so ``floor_1_day`` is a no-op on values. That is acceptable here: the
integration test exercises plugin discovery, feature-name resolution, and
pipeline plumbing, not the bucketization arithmetic (which is fully covered
by the dedicated-fixture tests in the framework-specific test modules).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pyarrow as pa

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.pyarrow_time_bucketization import (
    PyArrowTimeBucketization,
)


# Expected values on the canonical dataset (timestamp column). Midnight days
# floor to themselves; the null row stays null.
_CANONICAL_FLOOR_1_DAY: list[Any] = [
    datetime(2023, 1, 1, tzinfo=timezone.utc),
    datetime(2023, 1, 2, tzinfo=timezone.utc),
    datetime(2023, 1, 3, tzinfo=timezone.utc),
    datetime(2023, 1, 5, tzinfo=timezone.utc),
    datetime(2023, 1, 6, tzinfo=timezone.utc),
    datetime(2023, 1, 6, tzinfo=timezone.utc),
    datetime(2023, 1, 7, tzinfo=timezone.utc),
    datetime(2023, 1, 8, tzinfo=timezone.utc),
    datetime(2023, 1, 9, tzinfo=timezone.utc),
    datetime(2023, 1, 10, tzinfo=timezone.utc),
    None,
    datetime(2023, 1, 12, tzinfo=timezone.utc),
]

# All timestamps are in January 2023 (or null), so month floor collapses to
# 2023-01-01.
_CANONICAL_FLOOR_1_MONTH: list[Any] = [datetime(2023, 1, 1, tzinfo=timezone.utc)] * 10 + [
    None,
    datetime(2023, 1, 1, tzinfo=timezone.utc),
]


class TestTimeBucketizationIntegration(DataOpsIntegrationTestBase):
    """Standard integration tests inherited from the base class.

    Time-bucketization uses pattern-based matching, so
    ``match_feature_group_criteria`` succeeds with empty Options when the
    feature name contains the pattern. ``test_match_rejects_missing_config``
    is overridden below to use a non-pattern name that requires config-based
    matching.
    """

    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowTimeBucketization

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "timestamp__floor_1_day"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return list(_CANONICAL_FLOOR_1_DAY)

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "timestamp__floor_1_month"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return list(_CANONICAL_FLOOR_1_MONTH)

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return [
            "timestamp__floor_1_day",
            "timestamp__ceil_1_day",
            "timestamp__round_1_day",
            "timestamp__floor_5_minute",
            "timestamp__ceil_15_minute",
            "timestamp__floor_1_week",
            "timestamp__floor_1_month",
            "timestamp__floor_1_year",
        ]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return [
            "timestamp",
            "timestamp__floor",
            "timestamp__truncate_1_day",
            "timestamp__floor_1_century",
            "timestamp__floor_0_day",
            "floor_1_day",
        ]

    @classmethod
    def match_options(cls) -> Options:
        return Options()

    def test_match_rejects_missing_config(self) -> None:
        """Pattern-based: a non-pattern name without config must fail to match."""
        options = Options()
        assert not PyArrowTimeBucketization.match_feature_group_criteria("my_custom_result", options)


class TestIntegrationMultipleFeatures:
    """Run multiple bucketization features in a single ``run_all`` call."""

    def test_floor_day_and_floor_month_together(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowTimeBucketization}
        )

        f_day = Feature("timestamp__floor_1_day", options=Options())
        f_month = Feature("timestamp__floor_1_month", options=Options())

        results = mloda.run_all(
            [f_day, f_month],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        day_found = False
        month_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "timestamp__floor_1_day" in table.column_names:
                col = table.column("timestamp__floor_1_day").to_pylist()
                assert col == _CANONICAL_FLOOR_1_DAY
                day_found = True
            if "timestamp__floor_1_month" in table.column_names:
                col = table.column("timestamp__floor_1_month").to_pylist()
                assert col == _CANONICAL_FLOOR_1_MONTH
                month_found = True

        assert day_found, "timestamp__floor_1_day result not found"
        assert month_found, "timestamp__floor_1_month result not found"


class TestIntegrationOptionBasedConfig:
    """Integration tests for option-based (non-string-pattern) configuration."""

    def test_option_based_floor_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowTimeBucketization}
        )

        feature = Feature(
            "my_day_floor",
            options=Options(
                context={
                    "bucket_op": "floor_1_day",
                    "in_features": "timestamp",
                }
            ),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "my_day_floor" in table.column_names:
                col = table.column("my_day_floor").to_pylist()
                assert col == _CANONICAL_FLOOR_1_DAY
                assert len(col) == 12
                found = True

        assert found, "option-based floor_1_day result not found"

    def test_string_and_option_features_together(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowTimeBucketization}
        )

        f_pattern = Feature("timestamp__floor_1_day", options=Options())
        f_option = Feature(
            "my_month_floor",
            options=Options(
                context={
                    "bucket_op": "floor_1_month",
                    "in_features": "timestamp",
                }
            ),
        )

        results = mloda.run_all(
            [f_pattern, f_option],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        pattern_found = False
        option_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "timestamp__floor_1_day" in table.column_names:
                col = table.column("timestamp__floor_1_day").to_pylist()
                assert col == _CANONICAL_FLOOR_1_DAY
                pattern_found = True
            if "my_month_floor" in table.column_names:
                col = table.column("my_month_floor").to_pylist()
                assert col == _CANONICAL_FLOOR_1_MONTH
                option_found = True

        assert pattern_found, "string-pattern floor_1_day not found"
        assert option_found, "option-based floor_1_month not found"
