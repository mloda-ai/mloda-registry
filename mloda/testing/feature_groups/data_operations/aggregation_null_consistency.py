"""Shared test base class for cross-framework null consistency in column aggregation.

All test logic lives here. Framework-specific test files subclass and
provide an ``all_adapters`` fixture that yields a list of FrameworkAdapter
objects (one per framework under test).

Expected values are from the canonical 12-row dataset in
DataOperationsTestDataCreator.

See: https://github.com/mloda-ai/mloda-registry/issues/74
"""

from __future__ import annotations

from typing import Any

from mloda.testing.feature_groups.data_operations.null_consistency import (
    FrameworkAdapter,
    assert_all_frameworks_agree,
    make_feature_set,
)


class AggregationNullConsistencyBase:
    """Abstract base for column aggregation null consistency tests.

    Subclasses must ensure an ``all_adapters`` pytest fixture is available
    that yields ``list[FrameworkAdapter]`` with PyArrow as the first entry.
    """

    # -- All-null column (score) ------------------------------------------------

    def test_sum_all_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Sum of all-null column: PyArrow/Polars/DuckDB/SQLite return None.

        NOTE: Known divergence. Pandas returns 0 (identity element convention:
        sum of zero valid values = 0, via skipna=True with min_count=0 default).
        All other frameworks return None (SQL null-propagation convention).
        """
        fs = make_feature_set("score__sum_aggr")
        non_pandas = [a for a in all_adapters if a.name != "pandas"]
        results = assert_all_frameworks_agree(non_pandas, fs, "score__sum_aggr")
        assert all(v is None for v in results[non_pandas[0].name])
        # Verify Pandas divergence is documented, not silent
        pandas_adapters = [a for a in all_adapters if a.name == "pandas"]
        if pandas_adapters:
            raw = pandas_adapters[0].calculate(fs)
            pandas_col = pandas_adapters[0].extract(raw, "score__sum_aggr")
            assert all(v == 0 for v in pandas_col), "Pandas sum(all-null) should return 0"

    def test_min_all_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Min of all-null column should produce None for every row."""
        fs = make_feature_set("score__min_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "score__min_aggr")
        assert all(v is None for v in results[all_adapters[0].name])

    def test_max_all_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Max of all-null column should produce None for every row."""
        fs = make_feature_set("score__max_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "score__max_aggr")
        assert all(v is None for v in results[all_adapters[0].name])

    def test_avg_all_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Avg of all-null column should produce None for every row."""
        fs = make_feature_set("score__avg_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "score__avg_aggr")
        assert all(v is None for v in results[all_adapters[0].name])

    def test_count_all_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Count of all-null column should produce 0 for every row."""
        fs = make_feature_set("score__count_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "score__count_aggr")
        assert all(v == 0 for v in results[all_adapters[0].name])

    # -- Partial-null column (value_int: 1 null at row 4) ----------------------

    def test_sum_partial_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_int has 1 null. All frameworks should skip it and agree on sum=225."""
        fs = make_feature_set("value_int__sum_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "value_int__sum_aggr")
        assert all(v == 225 for v in results[all_adapters[0].name])

    def test_count_partial_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_int has 1 null. Count of non-null values should be 11."""
        fs = make_feature_set("value_int__count_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "value_int__count_aggr")
        assert all(v == 11 for v in results[all_adapters[0].name])

    def test_min_partial_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_int has 1 null. Min should be -10."""
        fs = make_feature_set("value_int__min_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "value_int__min_aggr")
        assert all(v == -10 for v in results[all_adapters[0].name])

    def test_max_partial_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_int has 1 null. Max should be 60."""
        fs = make_feature_set("value_int__max_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "value_int__max_aggr")
        assert all(v == 60 for v in results[all_adapters[0].name])

    def test_avg_partial_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_int has 1 null. Avg should be 225/11."""
        fs = make_feature_set("value_int__avg_aggr")
        assert_all_frameworks_agree(all_adapters, fs, "value_int__avg_aggr", use_approx=True)

    # -- Multi-null column (value_float: nulls at rows 2 and 11) ---------------

    def test_sum_multi_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_float has 2 nulls. All frameworks should skip them."""
        fs = make_feature_set("value_float__sum_aggr")
        assert_all_frameworks_agree(all_adapters, fs, "value_float__sum_aggr", use_approx=True)

    def test_count_multi_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_float has 2 nulls. Count should be 10."""
        fs = make_feature_set("value_float__count_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "value_float__count_aggr")
        assert all(v == 10 for v in results[all_adapters[0].name])

    def test_min_multi_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_float has 2 nulls. Min should be -3.14."""
        fs = make_feature_set("value_float__min_aggr")
        assert_all_frameworks_agree(all_adapters, fs, "value_float__min_aggr", use_approx=True)

    def test_max_multi_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_float has 2 nulls. Max should be 100.0."""
        fs = make_feature_set("value_float__max_aggr")
        assert_all_frameworks_agree(all_adapters, fs, "value_float__max_aggr", use_approx=True)

    def test_avg_multi_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """value_float has 2 nulls. Avg should use only non-null values."""
        fs = make_feature_set("value_float__avg_aggr")
        assert_all_frameworks_agree(all_adapters, fs, "value_float__avg_aggr", use_approx=True)

    # -- Amount column (nulls at rows 1 and 7) ---------------------------------

    def test_sum_amount(self, all_adapters: list[FrameworkAdapter]) -> None:
        """amount has nulls at rows 1 and 7. All frameworks should skip them."""
        fs = make_feature_set("amount__sum_aggr")
        assert_all_frameworks_agree(all_adapters, fs, "amount__sum_aggr", use_approx=True)

    def test_count_amount(self, all_adapters: list[FrameworkAdapter]) -> None:
        """amount has 2 nulls. Count should be 10."""
        fs = make_feature_set("amount__count_aggr")
        results = assert_all_frameworks_agree(all_adapters, fs, "amount__count_aggr")
        assert all(v == 10 for v in results[all_adapters[0].name])
