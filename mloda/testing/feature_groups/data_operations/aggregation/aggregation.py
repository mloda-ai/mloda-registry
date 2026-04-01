"""Shared test base class, data, and helpers for aggregation tests.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, partitioned by the 'region' column
(A=rows 0-3, B=rows 4-7, C=rows 8-10, None=row 11).

Aggregation reduces to one row per unique group key. With
partition_by=["region"], the 12 input rows collapse to 4 output rows
(A, B, C, None).

The ``AggregationTestBase`` class provides concrete test methods
that any framework implementation inherits by subclassing and implementing
5 abstract methods.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set


# ---------------------------------------------------------------------------
# Expected values (module-level constants)
# ---------------------------------------------------------------------------
# After GROUP BY region, output has 4 rows: A, B, C, None
# Order depends on implementation but values per group are deterministic.

EXPECTED_SUM_BY_REGION: dict[Any, int] = {"A": 25, "B": 140, "C": 70, None: -10}
EXPECTED_AVG_BY_REGION: dict[Any, float] = {"A": 6.25, "B": 46.667, "C": 23.333, None: -10.0}
EXPECTED_COUNT_BY_REGION: dict[Any, int] = {"A": 4, "B": 3, "C": 3, None: 1}
EXPECTED_MIN_BY_REGION: dict[Any, int] = {"A": -5, "B": 30, "C": 15, None: -10}
EXPECTED_MAX_BY_REGION: dict[Any, int] = {"A": 20, "B": 60, "C": 40, None: -10}


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _build_result_map(
    region_col: list[Any],
    value_col: list[Any],
) -> dict[Any, Any]:
    """Build a {region: value} mapping from result columns."""
    return {region_col[i]: value_col[i] for i in range(len(region_col))}


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class AggregationTestBase(DataOpsTestBase):
    """Abstract base class for aggregation framework tests.

    Subclasses implement 5 abstract methods to wire up their framework,
    then inherit concrete test methods for free.

    Simple frameworks (PyArrow, Pandas, Polars) need ~15 lines.
    Connection-based frameworks (DuckDB, SQLite) need ~25 lines
    (add setup_method/teardown_method for connection lifecycle).
    """

    # -- Overridable methods --------------------------------------------------

    ALL_AGG_TYPES = {
        "sum",
        "avg",
        "count",
        "min",
        "max",
        "std",
        "var",
        "std_pop",
        "std_samp",
        "var_pop",
        "var_samp",
        "median",
        "mode",
        "nunique",
        "first",
        "last",
    }

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        """Aggregation types this framework supports. Override to restrict."""
        return cls.ALL_AGG_TYPES

    # -- PyArrow reference (for cross-framework comparison) -------------------

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        """Return the PyArrow implementation class (reference for cross-framework comparison)."""
        from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
            PyArrowAggregation,
        )

        return PyArrowAggregation

    # -- Concrete test methods (inherited for free) --------------------------

    def test_sum_agg_region(self) -> None:
        """Sum of value_int grouped by region."""
        fs = make_feature_set("value_int__sum_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 4

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__sum_agg")
        result_map = _build_result_map(region_col, result_col)
        for region, expected in EXPECTED_SUM_BY_REGION.items():
            assert result_map[region] == expected, f"region={region}"

    def test_avg_agg_region(self) -> None:
        """Average of value_int grouped by region."""
        fs = make_feature_set("value_int__avg_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 4

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__avg_agg")
        result_map = _build_result_map(region_col, result_col)
        for region, expected in EXPECTED_AVG_BY_REGION.items():
            assert result_map[region] == pytest.approx(expected, rel=1e-3), f"region={region}"

    def test_count_agg_region(self) -> None:
        """Count of non-null value_int grouped by region."""
        fs = make_feature_set("value_int__count_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 4

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__count_agg")
        result_map = _build_result_map(region_col, result_col)
        for region, expected in EXPECTED_COUNT_BY_REGION.items():
            assert result_map[region] == expected, f"region={region}"

    def test_min_agg_region(self) -> None:
        """Minimum of value_int grouped by region."""
        fs = make_feature_set("value_int__min_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 4

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__min_agg")
        result_map = _build_result_map(region_col, result_col)
        for region, expected in EXPECTED_MIN_BY_REGION.items():
            assert result_map[region] == expected, f"region={region}"

    def test_max_agg_region(self) -> None:
        """Maximum of value_int grouped by region."""
        fs = make_feature_set("value_int__max_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 4

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__max_agg")
        result_map = _build_result_map(region_col, result_col)
        for region, expected in EXPECTED_MAX_BY_REGION.items():
            assert result_map[region] == expected, f"region={region}"

    def test_null_policy_skip_avg_with_null_values(self) -> None:
        """NullPolicy.SKIP: Group B has a null value_int at row 4. Avg should skip it."""
        fs = make_feature_set("value_int__avg_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__avg_agg")
        result_map = _build_result_map(region_col, result_col)
        # B has [None, 50, 30, 60] -> avg of non-null = 140/3
        assert result_map["B"] == pytest.approx(140.0 / 3.0, rel=1e-6)

    def test_null_policy_null_is_group(self) -> None:
        """NullPolicy.NULL_IS_GROUP: Row 11 has region=None. It should form its own group."""
        fs = make_feature_set("value_int__sum_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__sum_agg")
        result_map = _build_result_map(region_col, result_col)
        assert result_map[None] == -10

    def test_output_rows_reduced(self) -> None:
        """Output must have exactly 4 rows (one per group), fewer than 12 input rows."""
        fs = make_feature_set("value_int__sum_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 4

    def test_new_column_added(self) -> None:
        """The aggregation result column should be present in the output."""
        fs = make_feature_set("value_int__max_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_cols = self.extract_column(result, "value_int__max_agg")
        assert len(result_cols) == 4

    def test_result_has_correct_type(self) -> None:
        """The result of calculate_feature must be the expected framework type."""
        fs = make_feature_set("value_int__min_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())

    # -- Cross-framework comparison (matches PyArrow reference) --------------

    def _compare_agg_with_pyarrow(self, feature_name: str, partition_by: list[str], use_approx: bool = False) -> None:
        """Run the feature through this framework and PyArrow, assert results match."""
        fs = make_feature_set(feature_name, partition_by)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ref = self.pyarrow_implementation_class().calculate_feature(self._arrow_table, fs)

        region_col = self.extract_column(result, partition_by[0])
        result_col = self.extract_column(result, feature_name)
        result_map = _build_result_map(region_col, result_col)

        ref_region_col = extract_column(ref, partition_by[0])
        ref_col = extract_column(ref, feature_name)
        ref_map = _build_result_map(ref_region_col, ref_col)

        assert len(result_map) == len(ref_map), f"group count {len(result_map)} != reference {len(ref_map)}"
        for key in ref_map:
            if use_approx:
                if ref_map[key] is None:
                    assert result_map[key] is None, f"group {key}: expected None, got {result_map[key]}"
                else:
                    assert result_map[key] == pytest.approx(ref_map[key], rel=1e-6), (
                        f"group {key}: {result_map[key]} != reference {ref_map[key]}"
                    )
            else:
                assert result_map[key] == ref_map[key], f"group {key}: {result_map[key]} != reference {ref_map[key]}"

    def test_cross_framework_sum(self) -> None:
        """Sum must match PyArrow reference."""
        self._compare_agg_with_pyarrow("value_int__sum_agg", ["region"])

    def test_cross_framework_avg(self) -> None:
        """Avg must match PyArrow reference."""
        self._compare_agg_with_pyarrow("value_int__avg_agg", ["region"], use_approx=True)

    def test_cross_framework_count(self) -> None:
        """Count must match PyArrow reference."""
        self._compare_agg_with_pyarrow("value_int__count_agg", ["region"])

    def test_cross_framework_min(self) -> None:
        """Min must match PyArrow reference."""
        self._compare_agg_with_pyarrow("value_int__min_agg", ["region"])

    def test_cross_framework_max(self) -> None:
        """Max must match PyArrow reference."""
        self._compare_agg_with_pyarrow("value_int__max_agg", ["region"])

    def test_cross_framework_std(self) -> None:
        """Std must match PyArrow reference."""
        self._skip_if_unsupported("std")
        self._compare_agg_with_pyarrow("value_int__std_agg", ["region"], use_approx=True)

    def test_cross_framework_var(self) -> None:
        """Var must match PyArrow reference."""
        self._skip_if_unsupported("var")
        self._compare_agg_with_pyarrow("value_int__var_agg", ["region"], use_approx=True)

    def test_cross_framework_median(self) -> None:
        """Median must match PyArrow reference."""
        self._skip_if_unsupported("median")
        self._compare_agg_with_pyarrow("value_int__median_agg", ["region"], use_approx=True)

    def test_cross_framework_nunique(self) -> None:
        """Nunique must match PyArrow reference."""
        self._skip_if_unsupported("nunique")
        self._compare_agg_with_pyarrow("value_int__nunique_agg", ["region"])

    # -- Statistical aggregation tests (skipped if unsupported) --------------

    def test_std_agg_region(self) -> None:
        """Population standard deviation of value_int grouped by region (ddof=0)."""
        self._skip_if_unsupported("std")
        fs = make_feature_set("value_int__std_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__std_agg")
        result_map = _build_result_map(region_col, result_col)
        # Group A values: [10, -5, 0, 20] -> population std (ddof=0)
        a_vals = [10, -5, 0, 20]
        a_std = (sum((x - 6.25) ** 2 for x in a_vals) / len(a_vals)) ** 0.5
        assert result_map["A"] == pytest.approx(a_std, rel=1e-6)

    def test_var_agg_region(self) -> None:
        """Population variance of value_int grouped by region (ddof=0)."""
        self._skip_if_unsupported("var")
        fs = make_feature_set("value_int__var_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__var_agg")
        result_map = _build_result_map(region_col, result_col)
        a_vals = [10, -5, 0, 20]
        a_var = sum((x - 6.25) ** 2 for x in a_vals) / len(a_vals)
        assert result_map["A"] == pytest.approx(a_var, rel=1e-6)

    def test_std_pop_agg_region(self) -> None:
        """Explicit population standard deviation (ddof=0), identical to std."""
        self._skip_if_unsupported("std_pop")
        fs = make_feature_set("value_int__std_pop_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__std_pop_agg")
        result_map = _build_result_map(region_col, result_col)
        a_vals = [10, -5, 0, 20]
        a_std = (sum((x - 6.25) ** 2 for x in a_vals) / len(a_vals)) ** 0.5
        assert result_map["A"] == pytest.approx(a_std, rel=1e-6)

    def test_std_samp_agg_region(self) -> None:
        """Sample standard deviation (ddof=1)."""
        self._skip_if_unsupported("std_samp")
        fs = make_feature_set("value_int__std_samp_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__std_samp_agg")
        result_map = _build_result_map(region_col, result_col)
        a_vals = [10, -5, 0, 20]
        a_std = (sum((x - 6.25) ** 2 for x in a_vals) / (len(a_vals) - 1)) ** 0.5
        assert result_map["A"] == pytest.approx(a_std, rel=1e-6)

    def test_var_pop_agg_region(self) -> None:
        """Explicit population variance (ddof=0), identical to var."""
        self._skip_if_unsupported("var_pop")
        fs = make_feature_set("value_int__var_pop_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__var_pop_agg")
        result_map = _build_result_map(region_col, result_col)
        a_vals = [10, -5, 0, 20]
        a_var = sum((x - 6.25) ** 2 for x in a_vals) / len(a_vals)
        assert result_map["A"] == pytest.approx(a_var, rel=1e-6)

    def test_var_samp_agg_region(self) -> None:
        """Sample variance (ddof=1)."""
        self._skip_if_unsupported("var_samp")
        fs = make_feature_set("value_int__var_samp_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__var_samp_agg")
        result_map = _build_result_map(region_col, result_col)
        a_vals = [10, -5, 0, 20]
        a_var = sum((x - 6.25) ** 2 for x in a_vals) / (len(a_vals) - 1)
        assert result_map["A"] == pytest.approx(a_var, rel=1e-6)

    def test_median_agg_region(self) -> None:
        """Median of value_int grouped by region."""
        self._skip_if_unsupported("median")
        fs = make_feature_set("value_int__median_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__median_agg")
        result_map = _build_result_map(region_col, result_col)
        assert result_map["A"] == pytest.approx(5.0, rel=1e-6)
        assert result_map["B"] == pytest.approx(50.0, rel=1e-6)
        assert result_map["C"] == pytest.approx(15.0, rel=1e-6)
        assert result_map[None] == pytest.approx(-10.0, rel=1e-6)

    # -- Advanced aggregation tests (skipped if unsupported) -----------------

    def test_mode_agg_region(self) -> None:
        """Mode of value_int grouped by region."""
        self._skip_if_unsupported("mode")
        fs = make_feature_set("value_int__mode_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__mode_agg")
        result_map = _build_result_map(region_col, result_col)
        # Group C has 15 appearing twice, so mode = 15
        assert result_map["C"] == 15
        # None group: single value -10
        assert result_map[None] == -10

    def test_nunique_agg_region(self) -> None:
        """Count of unique value_int values grouped by region."""
        self._skip_if_unsupported("nunique")
        fs = make_feature_set("value_int__nunique_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__nunique_agg")
        result_map = _build_result_map(region_col, result_col)
        assert result_map["A"] == 4  # {10, -5, 0, 20}
        assert result_map["B"] == 3  # {50, 30, 60} - nulls excluded
        assert result_map["C"] == 2  # {15, 40}
        assert result_map[None] == 1  # {-10}

    def test_first_agg_region(self) -> None:
        """First value of value_int grouped by region.

        Group values (insertion order): A=[10,-5,0,20], B=[None,50,30,60],
        C=[15,15,40], None=[-10]. For groups without leading nulls, all
        frameworks agree. For B, PyArrow returns the first non-null (50)
        while Polars/DuckDB return None (literal first value).
        """
        self._skip_if_unsupported("first")
        fs = make_feature_set("value_int__first_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__first_agg")
        result_map = _build_result_map(region_col, result_col)

        assert result_map["A"] == 10
        assert result_map["C"] == 15
        assert result_map[None] == -10
        # B has a leading null: PyArrow skips nulls (50), others return None
        assert result_map["B"] in {None, 50}

    def test_last_agg_region(self) -> None:
        """Last value of value_int grouped by region.

        Group values (insertion order): A=[10,-5,0,20], B=[None,50,30,60],
        C=[15,15,40], None=[-10]. No group has a trailing null, so all
        frameworks agree on the last value.
        """
        self._skip_if_unsupported("last")
        fs = make_feature_set("value_int__last_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__last_agg")
        result_map = _build_result_map(region_col, result_col)

        assert result_map["A"] == 20
        assert result_map["B"] == 60
        assert result_map["C"] == 40
        assert result_map[None] == -10

    # -- Null edge case tests ------------------------------------------------

    def test_null_policy_skip_all_null_column(self) -> None:
        """NullPolicy.SKIP: score column is all null. Aggregation should produce all nulls."""
        fs = make_feature_set("score__sum_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__sum_agg")
        assert all(v is None for v in result_col)

    def test_all_null_column_count_per_group(self) -> None:
        """score is all-null. Count per group should be 0."""
        fs = make_feature_set("score__count_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__count_agg")
        assert all(v == 0 for v in result_col)

    def test_all_null_column_min_per_group(self) -> None:
        """score is all-null. Min per group should be None."""
        fs = make_feature_set("score__min_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__min_agg")
        assert all(v is None for v in result_col)

    def test_all_null_column_max_per_group(self) -> None:
        """score is all-null. Max per group should be None."""
        fs = make_feature_set("score__max_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__max_agg")
        assert all(v is None for v in result_col)

    def test_all_null_column_avg_per_group(self) -> None:
        """score is all-null. Avg per group should be None."""
        fs = make_feature_set("score__avg_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__avg_agg")
        assert all(v is None for v in result_col)

    def test_all_null_column_median_per_group(self) -> None:
        """score is all-null. Median per group should be None."""
        self._skip_if_unsupported("median")
        fs = make_feature_set("score__median_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__median_agg")
        assert all(v is None for v in result_col)

    def test_all_null_column_mode_per_group(self) -> None:
        """score is all-null. Mode per group should be None."""
        self._skip_if_unsupported("mode")
        fs = make_feature_set("score__mode_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__mode_agg")
        assert all(v is None for v in result_col)

    # -- Cross-framework null comparisons ------------------------------------

    def test_cross_framework_all_null_sum(self) -> None:
        """score sum aggregation must match PyArrow reference."""
        self._compare_agg_with_pyarrow("score__sum_agg", ["region"])

    def test_cross_framework_all_null_count(self) -> None:
        """score count aggregation must match PyArrow reference."""
        self._compare_agg_with_pyarrow("score__count_agg", ["region"])

    # -- Multi-key partition tests -------------------------------------------

    def test_multi_key_partition_sum(self) -> None:
        """Sum of value_int grouped by [region, category]."""
        fs = make_feature_set("value_int__sum_agg", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        category_col = self.extract_column(result, "category")
        result_col = self.extract_column(result, "value_int__sum_agg")
        result_map = {(region_col[i], category_col[i]): result_col[i] for i in range(len(region_col))}

        assert result_map[("A", "X")] == 10  # [10, 0]
        assert result_map[("A", "Y")] == 15  # [-5, 20]
        assert result_map[("B", "X")] == 60  # [None, 60]
        assert result_map[("B", "Y")] == 50  # [50]
        assert result_map[("B", None)] == 30  # [30]

    def test_multi_key_partition_count(self) -> None:
        """Count of non-null value_int grouped by [region, category]."""
        fs = make_feature_set("value_int__count_agg", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        category_col = self.extract_column(result, "category")
        result_col = self.extract_column(result, "value_int__count_agg")
        result_map = {(region_col[i], category_col[i]): result_col[i] for i in range(len(region_col))}

        assert result_map[("A", "X")] == 2
        assert result_map[("A", "Y")] == 2
        assert result_map[("B", "X")] == 1  # row 4 is null

    def test_multi_key_float_avg(self) -> None:
        """Avg of value_float grouped by [region, category]."""
        fs = make_feature_set("value_float__avg_agg", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        category_col = self.extract_column(result, "category")
        result_col = self.extract_column(result, "value_float__avg_agg")
        result_map = {(region_col[i], category_col[i]): result_col[i] for i in range(len(region_col))}

        # A/X: [1.5, None] -> avg = 1.5
        assert result_map[("A", "X")] == pytest.approx(1.5, rel=1e-6)
        # A/Y: [2.5, 0.0] -> avg = 1.25
        assert result_map[("A", "Y")] == pytest.approx(1.25, rel=1e-6)

    # -- Unsupported operation raises ----------------------------------------

    def test_unsupported_aggregation_type_raises(self) -> None:
        """Calling calculate_feature with an unknown aggregation type should raise."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "value_int__evil_type_agg",
            options=Options(
                context={
                    "partition_by": ["region"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises((ValueError, KeyError)):
            self.implementation_class().calculate_feature(self.test_data, fs)
