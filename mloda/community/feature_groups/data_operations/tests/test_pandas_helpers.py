"""Tests for shared pandas helper utilities used by group and window aggregation."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.pandas_helpers import (
    PANDAS_AGG_FUNCS,
    apply_null_safe_agg,
    coerce_count_dtype,
    null_safe_groupby,
)


class TestPandasAggFuncs:
    def test_pandas_agg_funcs_contains_base_mappings(self) -> None:
        """PANDAS_AGG_FUNCS must contain exactly the 5 canonical aggregation mappings."""
        expected = {
            "sum": "sum",
            "avg": "mean",
            "count": "count",
            "min": "min",
            "max": "max",
        }
        assert PANDAS_AGG_FUNCS == expected


class TestNullSafeGroupby:
    def test_null_safe_groupby_preserves_null_keys(self) -> None:
        """Null values in partition columns must form their own group, not be dropped."""
        df = pd.DataFrame(
            {
                "key": ["a", "a", None, None],
                "val": [1.0, 2.0, 3.0, 4.0],
            }
        )
        grouped = null_safe_groupby(df, ["key"], "val")
        group_keys = [k for k, _ in grouped]
        # There should be a group for the null key (represented as NaN in pandas).
        assert any(pd.isna(k) for k in group_keys), (
            "Null group key was dropped; expected null keys to form their own group."
        )

    def test_null_safe_groupby_selects_column(self) -> None:
        """The returned object must be a SeriesGroupBy over the specified column."""
        df = pd.DataFrame(
            {
                "key": ["a", "b"],
                "val": [10, 20],
                "other": [99, 99],
            }
        )
        grouped = null_safe_groupby(df, ["key"], "val")
        assert isinstance(grouped, pd.core.groupby.SeriesGroupBy)


class TestApplyNullSafeAgg:
    def test_apply_null_safe_agg_sum_uses_min_count(self) -> None:
        """Sum of an all-NaN group must return NaN, not 0.

        pandas sum defaults to 0 for empty/all-null groups. Passing min_count=1
        ensures NaN is returned instead, matching PyArrow semantics.
        """
        df = pd.DataFrame(
            {
                "key": ["a", "a"],
                "val": [float("nan"), float("nan")],
            }
        )
        grouped = df.groupby(["key"], dropna=False)["val"]
        result = apply_null_safe_agg(grouped, "sum", "sum")
        assert result.isna().all(), "Expected NaN for all-NaN group under sum with min_count=1, got non-NaN."

    def test_apply_null_safe_agg_non_sum_no_min_count(self) -> None:
        """Non-sum aggregation types (avg, count, min, max) must work without min_count."""
        df = pd.DataFrame(
            {
                "key": ["a", "a", "b", "b"],
                "val": [1.0, 3.0, 2.0, 4.0],
            }
        )
        grouped = df.groupby(["key"], dropna=False)["val"]

        avg_result = apply_null_safe_agg(grouped, "mean", "avg")
        assert avg_result.loc["a"] == pytest.approx(2.0)
        assert avg_result.loc["b"] == pytest.approx(3.0)

        count_result = apply_null_safe_agg(grouped, "count", "count")
        assert count_result.loc["a"] == 2
        assert count_result.loc["b"] == 2

        min_result = apply_null_safe_agg(grouped, "min", "min")
        assert min_result.loc["a"] == pytest.approx(1.0)

        max_result = apply_null_safe_agg(grouped, "max", "max")
        assert max_result.loc["b"] == pytest.approx(4.0)

    def test_apply_null_safe_agg_transform_method(self) -> None:
        """method='transform' must produce row-preserving results (same length as input)."""
        df = pd.DataFrame(
            {
                "key": ["a", "a", "b", "b", "b"],
                "val": [10.0, 20.0, 1.0, 2.0, 3.0],
            }
        )
        grouped = df.groupby(["key"], dropna=False)["val"]
        result = apply_null_safe_agg(grouped, "sum", "sum", method="transform")
        # transform preserves the original number of rows
        assert len(result) == len(df)
        # Each row should carry its group's aggregate
        assert result.iloc[0] == pytest.approx(30.0)  # group "a" sum
        assert result.iloc[1] == pytest.approx(30.0)  # group "a" sum
        assert result.iloc[2] == pytest.approx(6.0)  # group "b" sum
        assert result.iloc[3] == pytest.approx(6.0)  # group "b" sum
        assert result.iloc[4] == pytest.approx(6.0)  # group "b" sum


class TestCoerceCountDtype:
    def test_coerce_count_dtype_converts_to_int64(self) -> None:
        """When agg_type is 'count', the target column must be cast to int64."""
        df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        assert df["feature"].dtype == "float64"
        coerce_count_dtype(df, "feature", "count")
        assert df["feature"].dtype == "int64"

    def test_coerce_count_dtype_noop_for_non_count(self) -> None:
        """For non-count agg types the column dtype must remain unchanged."""
        df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        original_dtype = df["feature"].dtype
        coerce_count_dtype(df, "feature", "sum")
        assert df["feature"].dtype == original_dtype
