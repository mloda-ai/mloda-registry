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
from mloda.testing.feature_groups.data_operations.mixins.mask import MaskTestMixin


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


class AggregationTestBase(MaskTestMixin, DataOpsTestBase):
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

    # -- MaskTestMixin configuration -------------------------------------------

    @classmethod
    def mask_feature_name(cls) -> str:
        return "value_int__sum_agg"

    @classmethod
    def mask_partition_by(cls) -> list[str] | None:
        return ["region"]

    @classmethod
    def mask_expected_row_count(cls) -> int:
        return 4

    @classmethod
    def mask_is_reducing(cls) -> bool:
        return True

    @classmethod
    def mask_equal_expected(cls) -> dict[Any, Any]:
        # category='X': A: 10+0=10, B: 60, C: 15, None: -10
        return {"A": 10, "B": 60, "C": 15, None: -10}

    @classmethod
    def mask_multiple_conditions_expected(cls) -> dict[Any, Any]:
        # category='X' AND value_int>=10: A: 10, B: 60, C: 15, None: None (-10<10)
        return {"A": 10, "B": 60, "C": 15, None: None}

    @classmethod
    def mask_is_in_expected(cls) -> dict[Any, Any]:
        # region is_in ['A','C']: A=25 (all match), B=None, C=70, None=None
        return {"A": 25, "B": None, "C": 70, None: None}

    @classmethod
    def mask_greater_than_expected(cls) -> dict[Any, Any]:
        # value_int > 10: A: [20]=20, B: [50,30,60]=140, C: [15,15,40]=70, None: None
        return {"A": 20, "B": 140, "C": 70, None: None}

    @classmethod
    def mask_no_mask_expected(cls) -> dict[Any, Any]:
        return dict(EXPECTED_SUM_BY_REGION)

    # -- Reference implementation (for cross-framework comparison) -------------------

    @classmethod
    def reference_implementation_class(cls) -> Any:
        """Return the reference implementation class for cross-framework comparison."""
        from mloda.testing.feature_groups.data_operations.aggregation.reference import (
            ReferenceAggregation,
        )

        return ReferenceAggregation

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

    # -- Cross-framework comparison (matches reference) --------------

    def _compare_agg_with_reference(
        self,
        feature_name: str,
        partition_by: list[str],
        use_approx: bool = False,
    ) -> None:
        """Run the feature through this framework and the reference, assert results match.

        The reference implementation produces the canonical results.  Every
        framework must produce identical results.  If a framework cannot match
        the reference for a given operation, it should exclude that operation
        from supported_agg_types().
        """
        fs = make_feature_set(feature_name, partition_by)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ref = self.reference_implementation_class().calculate_feature(self._arrow_table, fs)

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
        """Sum must match reference."""
        self._compare_agg_with_reference("value_int__sum_agg", ["region"])

    def test_cross_framework_avg(self) -> None:
        """Avg must match reference."""
        self._compare_agg_with_reference("value_int__avg_agg", ["region"], use_approx=True)

    def test_cross_framework_count(self) -> None:
        """Count must match reference."""
        self._compare_agg_with_reference("value_int__count_agg", ["region"])

    def test_cross_framework_min(self) -> None:
        """Min must match reference."""
        self._compare_agg_with_reference("value_int__min_agg", ["region"])

    def test_cross_framework_max(self) -> None:
        """Max must match reference."""
        self._compare_agg_with_reference("value_int__max_agg", ["region"])

    def test_cross_framework_std(self) -> None:
        """Std must match reference."""
        self._skip_if_unsupported("std")
        self._compare_agg_with_reference("value_int__std_agg", ["region"], use_approx=True)

    def test_cross_framework_var(self) -> None:
        """Var must match reference."""
        self._skip_if_unsupported("var")
        self._compare_agg_with_reference("value_int__var_agg", ["region"], use_approx=True)

    def test_cross_framework_median(self) -> None:
        """Median must match reference."""
        self._skip_if_unsupported("median")
        self._compare_agg_with_reference("value_int__median_agg", ["region"], use_approx=True)

    def test_cross_framework_nunique(self) -> None:
        """Nunique must match reference."""
        self._skip_if_unsupported("nunique")
        self._compare_agg_with_reference("value_int__nunique_agg", ["region"])

    def test_cross_framework_first(self) -> None:
        """First must match reference.

        PyArrow skips nulls (Group B returns 50, not None).
        Frameworks that cannot match must exclude 'first' from supported_agg_types().
        """
        self._skip_if_unsupported("first")
        self._compare_agg_with_reference("value_int__first_agg", ["region"])

    def test_cross_framework_last(self) -> None:
        """Last must match reference."""
        self._skip_if_unsupported("last")
        self._compare_agg_with_reference("value_int__last_agg", ["region"])

    def test_cross_framework_mode(self) -> None:
        """Mode must match reference.

        PyArrow picks the first-encountered value on ties (Group A=10,
        Group B=50).  Frameworks that use different tie-breaking must
        exclude 'mode' from supported_agg_types().
        """
        self._skip_if_unsupported("mode")
        self._compare_agg_with_reference("value_int__mode_agg", ["region"])

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
        """Mode of value_int grouped by region.

        reference: A=10, B=50, C=15, None=-10.
        On ties (all-unique groups A, B), PyArrow picks the first-encountered value.
        """
        self._skip_if_unsupported("mode")
        fs = make_feature_set("value_int__mode_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__mode_agg")
        result_map = _build_result_map(region_col, result_col)
        assert result_map["A"] == 10
        assert result_map["B"] == 50
        assert result_map["C"] == 15
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

        reference: A=10, B=50, C=15, None=-10.
        PyArrow skips nulls, so Group B returns 50 (first non-null), not None.
        Frameworks that cannot match must exclude 'first' from supported_agg_types().
        """
        self._skip_if_unsupported("first")
        fs = make_feature_set("value_int__first_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__first_agg")
        result_map = _build_result_map(region_col, result_col)

        assert result_map["A"] == 10
        assert result_map["B"] == 50
        assert result_map["C"] == 15
        assert result_map[None] == -10

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

    def test_all_null_column_std_per_group(self) -> None:
        """score is all-null. Std per group should be None."""
        self._skip_if_unsupported("std")
        fs = make_feature_set("score__std_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__std_agg")
        assert all(v is None for v in result_col)

    def test_all_null_column_var_per_group(self) -> None:
        """score is all-null. Var per group should be None."""
        self._skip_if_unsupported("var")
        fs = make_feature_set("score__var_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__var_agg")
        assert all(v is None for v in result_col)

    def test_all_null_column_first_per_group(self) -> None:
        """score is all-null. First per group should be None."""
        self._skip_if_unsupported("first")
        fs = make_feature_set("score__first_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__first_agg")
        assert all(v is None for v in result_col)

    def test_all_null_column_last_per_group(self) -> None:
        """score is all-null. Last per group should be None."""
        self._skip_if_unsupported("last")
        fs = make_feature_set("score__last_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__last_agg")
        assert all(v is None for v in result_col)

    def test_all_null_column_nunique_per_group(self) -> None:
        """score is all-null. Nunique per group should be 0 (no distinct non-null values).

        reference: count_distinct excludes nulls, returning 0.
        """
        self._skip_if_unsupported("nunique")
        fs = make_feature_set("score__nunique_agg", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__nunique_agg")
        assert all(v == 0 for v in result_col)

    # -- Cross-framework null comparisons ------------------------------------

    def test_cross_framework_all_null_sum(self) -> None:
        """score sum aggregation must match reference."""
        self._compare_agg_with_reference("score__sum_agg", ["region"])

    def test_cross_framework_all_null_count(self) -> None:
        """score count aggregation must match reference."""
        self._compare_agg_with_reference("score__count_agg", ["region"])

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

    def _build_multi_key_map(self, result: Any, feature_name: str) -> dict[tuple[Any, ...], Any]:
        """Build a {(region, category): value} map from a multi-key aggregation result."""
        region_col = self.extract_column(result, "region")
        category_col = self.extract_column(result, "category")
        result_col = self.extract_column(result, feature_name)
        return {(region_col[i], category_col[i]): result_col[i] for i in range(len(region_col))}

    def test_multi_key_partition_median(self) -> None:
        """Median of value_int grouped by [region, category]."""
        self._skip_if_unsupported("median")
        fs = make_feature_set("value_int__median_agg", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_map = self._build_multi_key_map(result, "value_int__median_agg")

        # (A,X): [10, 0] -> median = 5.0
        assert result_map[("A", "X")] == pytest.approx(5.0, rel=1e-6)
        # (A,Y): [-5, 20] -> median = 7.5
        assert result_map[("A", "Y")] == pytest.approx(7.5, rel=1e-6)
        # (C,Y): [15, 40] -> median = 27.5
        assert result_map[("C", "Y")] == pytest.approx(27.5, rel=1e-6)
        # Single-value groups
        assert result_map[("B", "Y")] == pytest.approx(50.0, rel=1e-6)
        assert result_map[("C", "X")] == pytest.approx(15.0, rel=1e-6)

    def test_multi_key_partition_std(self) -> None:
        """Population std of value_int grouped by [region, category].

        Only groups with >= 2 non-null values are checked. Single-value
        groups produce 0 (population std) but the assertion is kept
        to groups where the result is non-trivial.
        """
        self._skip_if_unsupported("std")
        fs = make_feature_set("value_int__std_agg", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_map = self._build_multi_key_map(result, "value_int__std_agg")

        # (A,X): [10, 0] -> pop std = 5.0
        assert result_map[("A", "X")] == pytest.approx(5.0, rel=1e-6)
        # (A,Y): [-5, 20] -> pop std = 12.5
        assert result_map[("A", "Y")] == pytest.approx(12.5, rel=1e-6)
        # (C,Y): [15, 40] -> pop std = 12.5
        assert result_map[("C", "Y")] == pytest.approx(12.5, rel=1e-6)

    def test_multi_key_partition_var(self) -> None:
        """Population variance of value_int grouped by [region, category]."""
        self._skip_if_unsupported("var")
        fs = make_feature_set("value_int__var_agg", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_map = self._build_multi_key_map(result, "value_int__var_agg")

        # (A,X): [10, 0] -> pop var = 25.0
        assert result_map[("A", "X")] == pytest.approx(25.0, rel=1e-6)
        # (A,Y): [-5, 20] -> pop var = 156.25
        assert result_map[("A", "Y")] == pytest.approx(156.25, rel=1e-6)
        # (C,Y): [15, 40] -> pop var = 156.25
        assert result_map[("C", "Y")] == pytest.approx(156.25, rel=1e-6)

    def test_multi_key_partition_nunique(self) -> None:
        """Nunique of value_int grouped by [region, category]."""
        self._skip_if_unsupported("nunique")
        fs = make_feature_set("value_int__nunique_agg", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_map = self._build_multi_key_map(result, "value_int__nunique_agg")

        assert result_map[("A", "X")] == 2  # {10, 0}
        assert result_map[("A", "Y")] == 2  # {-5, 20}
        assert result_map[("B", "X")] == 1  # {60} (null excluded)
        assert result_map[("B", "Y")] == 1  # {50}
        assert result_map[("C", "Y")] == 2  # {15, 40}
        assert result_map[("C", "X")] == 1  # {15}

    def test_multi_key_partition_mode(self) -> None:
        """Mode of value_int grouped by [region, category].

        reference: on ties, picks the first-encountered value.
        """
        self._skip_if_unsupported("mode")
        fs = make_feature_set("value_int__mode_agg", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_map = self._build_multi_key_map(result, "value_int__mode_agg")

        assert result_map[("A", "X")] == 10
        assert result_map[("A", "Y")] == -5
        assert result_map[("B", "X")] == 60
        assert result_map[("B", "Y")] == 50
        assert result_map[("B", None)] == 30
        assert result_map[("C", "X")] == 15
        assert result_map[("C", "Y")] == 15
        assert result_map[(None, "X")] == -10

    def test_multi_key_partition_first(self) -> None:
        """First value of value_int grouped by [region, category] (insertion order).

        PyArrow skips nulls: (B,X) has [None, 60], first non-null = 60.
        """
        self._skip_if_unsupported("first")
        fs = make_feature_set("value_int__first_agg", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_map = self._build_multi_key_map(result, "value_int__first_agg")

        assert result_map[("A", "X")] == 10
        assert result_map[("A", "Y")] == -5
        assert result_map[("B", "X")] == 60
        assert result_map[("B", "Y")] == 50
        assert result_map[("B", None)] == 30
        assert result_map[("C", "Y")] == 15
        assert result_map[("C", "X")] == 15
        assert result_map[(None, "X")] == -10

    def test_multi_key_partition_last(self) -> None:
        """Last value of value_int grouped by [region, category] (insertion order).

        No multi-key group has a trailing null, so all frameworks agree.
        """
        self._skip_if_unsupported("last")
        fs = make_feature_set("value_int__last_agg", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_map = self._build_multi_key_map(result, "value_int__last_agg")

        assert result_map[("A", "X")] == 0
        assert result_map[("A", "Y")] == 20
        assert result_map[("B", "X")] == 60
        assert result_map[("B", "Y")] == 50
        assert result_map[("B", None)] == 30
        assert result_map[("C", "Y")] == 40
        assert result_map[("C", "X")] == 15
        assert result_map[(None, "X")] == -10

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

    # -- Mask (conditional aggregation) tests ----------------------------------

    def test_mask_sum_agg_equal(self) -> None:
        """Sum of value_int where category='X', grouped by region."""
        fs = make_feature_set("value_int__sum_agg", ["region"], mask=("category", "equal", "X"))
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 4
        result_col = self.extract_column(result, "value_int__sum_agg")
        result_map = _build_result_map(self.extract_column(result, "region"), result_col)
        assert result_map["A"] == 10
        assert result_map["B"] == 60
        assert result_map["C"] == 15
        assert result_map[None] == -10

    def test_mask_count_agg_equal(self) -> None:
        """Count of non-null value_int where category='X', grouped by region."""
        fs = make_feature_set("value_int__count_agg", ["region"], mask=("category", "equal", "X"))
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 4
        result_col = self.extract_column(result, "value_int__count_agg")
        result_map = _build_result_map(self.extract_column(result, "region"), result_col)
        assert result_map["A"] == 2
        assert result_map["B"] == 1
        assert result_map["C"] == 1
        assert result_map[None] == 1

    def test_mask_multiple_conditions_agg(self) -> None:
        """Sum with AND-combined mask: category='X' AND value_int >= 10."""
        fs = make_feature_set(
            "value_int__sum_agg",
            ["region"],
            mask=[("category", "equal", "X"), ("value_int", "greater_equal", 10)],
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 4
        result_col = self.extract_column(result, "value_int__sum_agg")
        # A: only row 0 (10) -> 10, B: only row 7 (60) -> 60, C: only row 9 (15) -> 15, None: row 11 (-10<10) -> None
        result_pairs = sorted(
            zip(self.extract_column(result, "region"), result_col),
            key=lambda x: (x[0] is None, x[0] or ""),
        )
        assert result_pairs[0] == ("A", 10)
        assert result_pairs[1] == ("B", 60)
        assert result_pairs[2] == ("C", 15)
        assert result_pairs[3][0] is None
        assert result_pairs[3][1] is None

    def test_mask_is_in_agg(self) -> None:
        """Sum of value_int where region is_in ['A', 'C'], grouped by region."""
        fs = make_feature_set("value_int__sum_agg", ["region"], mask=("region", "is_in", ["A", "C"]))
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 4
        result_col = self.extract_column(result, "value_int__sum_agg")
        result_map = _build_result_map(self.extract_column(result, "region"), result_col)
        assert result_map["A"] == 25  # All A rows match
        assert result_map["C"] == 70  # All C rows match
        assert result_map["B"] is None  # No B rows match
        assert result_map[None] is None  # None group doesn't match

    def test_mask_fully_masked_agg(self) -> None:
        """All rows masked out (category='Z') should produce None for every group."""
        fs = make_feature_set("value_int__sum_agg", ["region"], mask=("category", "equal", "Z"))
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 4
        result_col = self.extract_column(result, "value_int__sum_agg")
        assert all(v is None for v in result_col)

    def test_mask_greater_than_agg(self) -> None:
        """Sum of value_int where value_int > 10, grouped by region."""
        fs = make_feature_set("value_int__sum_agg", ["region"], mask=("value_int", "greater_than", 10))
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 4
        result_col = self.extract_column(result, "value_int__sum_agg")
        result_map = _build_result_map(self.extract_column(result, "region"), result_col)
        assert result_map["A"] == 20  # Only [20] > 10
        assert result_map["B"] == 140  # [50, 30, 60] > 10
        assert result_map["C"] == 70  # [15, 15, 40] > 10
        assert result_map[None] is None  # [-10] not > 10

    def test_mask_mode_agg_equal(self) -> None:
        """Mode of value_int where category='X', grouped by region."""
        self._skip_if_unsupported("mode")
        fs = make_feature_set("value_int__mode_agg", ["region"], mask=("category", "equal", "X"))
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 4
        result_col = self.extract_column(result, "value_int__mode_agg")
        result_map = _build_result_map(self.extract_column(result, "region"), result_col)
        assert result_map["A"] == 10
        assert result_map["B"] == 60
        assert result_map["C"] == 15
        assert result_map[None] == -10
