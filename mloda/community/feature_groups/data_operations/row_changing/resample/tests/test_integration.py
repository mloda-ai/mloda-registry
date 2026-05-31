"""Integration tests for resample through mloda's full pipeline.

Verifies that the resample feature group resolves through ``mloda.run_all`` with
a ``PluginCollector``, including plugin discovery, feature-name resolution
(``match_feature_group_criteria``), and pipeline plumbing.

resample is ROW-CHANGING: it collapses event rows onto a regular time grid, so
the output row count differs from the 12-row input and the output row order is
unspecified. The shared ``DataOpsIntegrationTestBase`` assumes a row-preserving
12-row exact-column compare, so it does NOT fit; this module uses bespoke
``run_all`` tests instead (mirroring time_bucketization's
``TestIntegrationMultipleFeatures``).

The public pipeline returns only the requested aggregate column (the partition
and bucket-start columns are not part of the selected output), so the assertions
pin the SORTED MULTISET of aggregate values plus the bucket count. Expected
multisets are computed OFFLINE on the canonical dataset (floor-to-day, then
groupby ``(region, bucket)`` with ``dropna=False``) and pinned as literals; the
production feature group is not imported to generate its own oracle.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_changing.resample.pyarrow_resample import (
    PyArrowResample,
)

# Daily resample of ``value_float`` grouped by ``(region, bucket_start)`` on the
# canonical dataset (dropna=False, so the null-timestamp row and the null-region
# row each form their own bucket -> 11 non-empty buckets total). All-null buckets
# emit mean=None / count=0. Output order is unspecified, so the SORTED multiset
# is pinned (None sorts last for the mean column).
_NUM_BUCKETS = 11
_MEAN_SORTED: list[Any] = [0.0, 0.0, 1e-15, 1.18, 1.5, 2.5, 6.5, 7.5, 100.0, None, None]
_COUNT_SORTED: list[Any] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2]


def _sort_none_last(values: list[Any]) -> list[Any]:
    """Sort values with ``None`` ordered last (mirrors the base-class trick)."""
    return sorted(values, key=lambda x: (x is None, x))


def _run_resample(name: str, context: dict[str, Any]) -> pa.Table:
    """Run a single resample feature through ``run_all`` and return its table."""
    plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowResample})
    feature = Feature(name, options=Options(context=context))

    results = mloda.run_all(
        [feature],
        compute_frameworks={PyArrowTable},
        plugin_collector=plugin_collector,
    )

    for table in results:
        if isinstance(table, pa.Table) and name in table.column_names:
            return table

    raise AssertionError(f"No result table with {name} found")


class TestResampleIntegration:
    """Bespoke ``run_all`` integration tests for the row-changing resample FG."""

    def test_daily_mean_through_pipeline(self) -> None:
        name = "value_float__resample_1_day_mean"
        table = _run_resample(name, {"time_column": "timestamp", "partition_by": ["region"]})

        assert table.num_rows == _NUM_BUCKETS

        col = table.column(name).to_pylist()
        assert _sort_none_last(col) == _MEAN_SORTED

    def test_daily_count_through_pipeline(self) -> None:
        name = "value_float__resample_1_day_count"
        table = _run_resample(name, {"time_column": "timestamp", "partition_by": ["region"]})

        assert table.num_rows == _NUM_BUCKETS

        col = table.column(name).to_pylist()
        assert sorted(col) == _COUNT_SORTED


class TestResampleMatchFeatureGroupCriteria:
    """match_feature_group_criteria routing for the resample pattern."""

    def test_valid_name_matches(self) -> None:
        options = Options(context={"time_column": "timestamp", "partition_by": ["region"]})
        assert PyArrowResample.match_feature_group_criteria("value_float__resample_1_hour_mean", options)

    def test_invalid_names_do_not_match(self) -> None:
        options = Options(context={"time_column": "timestamp", "partition_by": ["region"]})
        # median is not in the v1 agg set; ffill belongs to a different FG.
        assert not PyArrowResample.match_feature_group_criteria("value_float__resample_1_hour_median", options)
        assert not PyArrowResample.match_feature_group_criteria("value_float__ffill", options)
