"""Integration tests for resample through mloda's full pipeline.

Verifies that the resample feature group resolves through ``mloda.run_all`` with
a ``PluginCollector``, including plugin discovery, feature-name resolution
(``match_feature_group_criteria``), and pipeline plumbing.

resample is ROW-CHANGING: it collapses event rows onto a regular time grid, so
the output row count differs from the 12-row input and the output row order is
unspecified. The shared ``DataOpsIntegrationTestBase`` runs with a reduced
``expected_row_count`` and ``compare_sorted``, and
``OperationGroupIntegrationTestMixin`` adds the operation-group checks.

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
import pytest
from mloda.user import DataType, Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_changing.resample.base import (
    ResampleFeatureGroup,
)
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.testing.feature_groups.data_operations.mixins.operation_group_integration import (
    PYARROW_TABLE,
    OperationGroupIntegrationTestMixin,
    RoutingCase,
)

# Daily resample of ``value_float`` grouped by ``(region, bucket_start)`` on the
# canonical dataset (dropna=False, so the null-timestamp row and the null-region
# row each form their own bucket -> 11 non-empty buckets total). All-null buckets
# emit mean=None / count=0. Output order is unspecified, so the SORTED multiset
# is pinned (None sorts last for the mean column).
_NUM_BUCKETS = 11
_MEAN_SORTED: list[Any] = [0.0, 0.0, 1e-15, 1.18, 1.5, 2.5, 6.5, 7.5, 100.0, None, None]
_COUNT_SORTED: list[Any] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2]
# Daily sum per region bucket, computed with plain pyarrow.
_SUM_SORTED: list[Any] = [0.0, 0.0, 1e-15, 1.5, 2.36, 2.5, 6.5, 7.5, 100.0, None, None]


def _sort_none_last(values: list[Any]) -> list[Any]:
    """Sort values with ``None`` ordered last (mirrors the base-class trick)."""
    return sorted(values, key=lambda x: (x is None, x))


def _run_resample(name: str, context: dict[str, Any], *more_names: str) -> pa.Table:
    """Run resample features (all sharing ``context``) through ``run_all`` and return the table holding them."""
    plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, ResampleFeatureGroup})
    names = (name, *more_names)
    features = [Feature(n, options=Options(context=context)) for n in names]

    results = mloda.run_all(
        features,
        compute_frameworks=[PyArrowTable],
        plugin_collector=plugin_collector,
    )

    for table in results:
        if isinstance(table, pa.Table) and all(n in table.column_names for n in names):
            return table

    raise AssertionError(f"No result table with {names} found")


class TestResampleIntegration(OperationGroupIntegrationTestMixin, DataOpsIntegrationTestBase):
    @classmethod
    def feature_group_class(cls) -> type:
        return ResampleFeatureGroup

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def expected_row_count(cls) -> int:
        return _NUM_BUCKETS

    @classmethod
    def compare_sorted(cls) -> bool:
        return True

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_float__resample_1_day_mean"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"time_column": "timestamp", "partition_by": ["region"]}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return _MEAN_SORTED

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_float__resample_1_day_count"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"time_column": "timestamp", "partition_by": ["region"]}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return _COUNT_SORTED

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_float__resample_1_hour_mean"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        # median is not in the v1 agg set; ffill belongs to a different FG.
        return ["value_float__resample_1_hour_median", "value_float__ffill"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"time_column": "timestamp", "partition_by": ["region"]})

    @classmethod
    def divergent_routing_cases(cls) -> list[RoutingCase]:
        """Pandas declares every resample agg divergent (it skips NaN), so the PyArrow reference serves them."""
        return [
            RoutingCase(
                "value_float__resample_1_day_mean",
                cls.primary_feature_options(),
                implementation="mloda.community.feature_groups.data_operations.row_changing.resample."
                "pyarrow_resample:PyArrowResample",
                compute_framework=PYARROW_TABLE,
            )
        ]

    def test_daily_mean_and_sum_in_one_run(self) -> None:
        mean_name = "value_float__resample_1_day_mean"
        sum_name = "value_float__resample_1_day_sum"
        table = _run_resample(mean_name, {"time_column": "timestamp", "partition_by": ["region"]}, sum_name)

        assert table.num_rows == _NUM_BUCKETS
        assert _sort_none_last(table.column(mean_name).to_pylist()) == _MEAN_SORTED
        sums = _sort_none_last(table.column(sum_name).to_pylist())
        assert sums.count(None) == _SUM_SORTED.count(None)
        assert [v for v in sums if v is not None] == pytest.approx([v for v in _SUM_SORTED if v is not None], abs=1e-12)


class TestResampleSplitFeatureSets:
    """Data-type-split requests batch into one step; distinct groups and chained names also compute."""

    _CONTEXT = {"time_column": "timestamp", "partition_by": ["region"]}
    _MEAN = "value_float__resample_1_day_mean"
    _COUNT = "value_float__resample_1_day_count"

    def _run(self, features: list[Feature]) -> list[Any]:
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, ResampleFeatureGroup})
        return mloda.run_all(features, compute_frameworks=[PyArrowTable], plugin_collector=plugin_collector)

    @pytest.mark.parametrize("order", ["mean_first", "count_first"])
    @pytest.mark.parametrize("count_source", ["value_float", "timestamp"])
    def test_split_by_data_type_requests_compute_in_one_step(self, count_source: str, order: str) -> None:
        """Requests that differ only in data type batch into one step, so the later one sees the unreduced frame."""
        mean = Feature(self._MEAN, options=Options(context=self._CONTEXT), data_type=DataType.DOUBLE)
        count_name = f"{count_source}__resample_1_day_count"
        count = Feature(count_name, options=Options(context=self._CONTEXT), data_type=DataType.INT64)
        features = [mean, count] if order == "mean_first" else [count, mean]
        results = self._run(features)

        table = next(t for t in results if isinstance(t, pa.Table) and self._MEAN in t.column_names)
        assert _sort_none_last(table.column(self._MEAN).to_pylist()) == _MEAN_SORTED
        # The null-timestamp row has no timestamp to count, so that source counts one bucket fewer than value_float.
        expected_counts = _COUNT_SORTED if count_source == "value_float" else [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
        assert sorted(table.column(count_name).to_pylist()) == expected_counts

    def test_chained_resample_runs_without_group_error(self) -> None:
        """A chained name whose input is an output of the same group is one FeatureSet and must not raise."""
        name = "value_float__resample_1_day_sum__resample_2_day_max"
        options = Options(context=self._CONTEXT, cascade_context_keys=frozenset(self._CONTEXT))
        results = self._run([Feature(name, options=options)])
        assert any(isinstance(t, pa.Table) and name in t.column_names for t in results)

    def test_split_with_distinct_groups_work(self) -> None:
        features = [
            Feature(
                self._MEAN,
                options=Options(group={"g": "mean"}, context=self._CONTEXT),
                data_type=DataType.DOUBLE,
            ),
            Feature(
                self._COUNT,
                options=Options(group={"g": "count"}, context=self._CONTEXT),
                data_type=DataType.INT64,
            ),
        ]
        results = self._run(features)

        mean_table = next(t for t in results if isinstance(t, pa.Table) and self._MEAN in t.column_names)
        count_table = next(t for t in results if isinstance(t, pa.Table) and self._COUNT in t.column_names)
        assert _sort_none_last(mean_table.column(self._MEAN).to_pylist()) == _MEAN_SORTED
        assert sorted(count_table.column(self._COUNT).to_pylist()) == _COUNT_SORTED


class TestChainedNameDropsContext:
    """A chained resample name drops context keys on the child; the error must name time_column."""

    def test_chained_child_missing_time_column_named_in_resolution_error(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, ResampleFeatureGroup})
        feature = Feature(
            "value_float__resample_1_hour_mean__resample_2_hour_sum",
            options=Options(context={"time_column": "timestamp"}),
        )

        with pytest.raises(ValueError, match=r"required option 'time_column'"):
            mloda.run_all(
                [feature],
                compute_frameworks=[PyArrowTable],
                plugin_collector=plugin_collector,
            )


class TestResampleMatchFeatureGroupCriteria:
    """match_feature_group_criteria routing for the resample pattern."""

    def test_zero_bucket_names_do_not_match(self) -> None:
        # n must be a positive integer (n >= 1); a zero bucket size is not selectable.
        # resample_0_* matches PREFIX_PATTERN (n=0 is accepted so _validate_string_match has
        # real work to do) but is rejected there via _parse_resample_op; resample_00_* fails
        # PREFIX_PATTERN outright (no leading-zero form of n is accepted).
        options = Options(context={"time_column": "timestamp", "partition_by": ["region"]})
        assert not ResampleFeatureGroup.match_feature_group_criteria("value_float__resample_0_hour_mean", options)
        assert not ResampleFeatureGroup.match_feature_group_criteria("value_float__resample_0_minute_count", options)
        assert not ResampleFeatureGroup.match_feature_group_criteria("value_float__resample_00_hour_sum", options)
        # Regression guard: a positive bucket size still matches.
        assert ResampleFeatureGroup.match_feature_group_criteria("value_float__resample_1_hour_mean", options)

    def test_config_garbage_resample_op_does_not_match(self) -> None:
        # A config-style feature carrying an unparseable resample_op token must be
        # rejected by the resample_op validator (never routed to this FG).
        opts_bad = Options(context={"time_column": "ts", "resample_op": "not_a_token"})
        assert ResampleFeatureGroup.match_feature_group_criteria("my_resampled", opts_bad, None) is False

    def test_missing_time_column_does_not_match(self) -> None:
        """Name path: a pattern name without time_column is a non-match."""
        options = Options(context={"partition_by": ["region"]})
        assert not ResampleFeatureGroup.match_feature_group_criteria("value_float__resample_1_hour_mean", options)

    def test_config_path_missing_time_column_does_not_match(self) -> None:
        """Config path: a config feature without time_column stays a non-match."""
        options = Options(context={"in_features": "value_float", "resample_op": "1_hour_mean"})
        assert ResampleFeatureGroup.match_feature_group_criteria("my_resampled", options, None) is False

    def test_config_valid_resample_op_not_rejected_by_validator(self) -> None:
        # Exercise the actual match_guard wired into PROPERTY_MAPPING: a valid token
        # passes, a garbage token and a non-string are rejected.
        from mloda.community.feature_groups.data_operations.row_changing.resample.base import _is_valid_resample_op

        assert _is_valid_resample_op("1_hour_mean") is True
        assert _is_valid_resample_op("not_a_token") is False
        assert _is_valid_resample_op(123) is False
