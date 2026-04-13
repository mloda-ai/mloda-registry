"""Shared test base class, data, and helpers for window aggregation tests.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, partitioned by the 'region' column
(A=rows 0-3, B=rows 4-7, C=rows 8-10, None=row 11).

The ``WindowAggregationTestBase`` class provides 16 concrete test methods
(10 per-framework + 5 cross-framework comparison + 1 column check) that
any framework implementation inherits by subclassing and implementing
5 abstract methods. This follows the same pattern as mloda core's
``DataFrameTestBase`` in ``tests/test_plugins/compute_framework/test_tooling/``.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.mask import MaskTestMixin


# ---------------------------------------------------------------------------
# Expected values (module-level constants, also usable standalone)
# ---------------------------------------------------------------------------

EXPECTED_SUM_BY_REGION: list[int] = [25, 25, 25, 25, 140, 140, 140, 140, 70, 70, 70, -10]
EXPECTED_AVG_BY_REGION: list[float] = [
    6.25,
    6.25,
    6.25,
    6.25,
    46.667,
    46.667,
    46.667,
    46.667,
    23.333,
    23.333,
    23.333,
    -10.0,
]
EXPECTED_COUNT_BY_REGION: list[int] = [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 1]
EXPECTED_MIN_BY_REGION: list[int] = [-5, -5, -5, -5, 30, 30, 30, 30, 15, 15, 15, -10]
EXPECTED_MAX_BY_REGION: list[int] = [20, 20, 20, 20, 60, 60, 60, 60, 40, 40, 40, -10]

# First/last non-null value_int per region, ordered by value_int (ascending).
# Group A: sorted non-null = [-5, 0, 10, 20] -> first=-5, last=20
# Group B: sorted non-null = [30, 50, 60] -> first=30, last=60
# Group C: sorted non-null = [15, 15, 40] -> first=15, last=40
# None:    sorted non-null = [-10] -> first=-10, last=-10
EXPECTED_FIRST_BY_REGION: list[int] = [-5, -5, -5, -5, 30, 30, 30, 30, 15, 15, 15, -10]
EXPECTED_LAST_BY_REGION: list[int] = [20, 20, 20, 20, 60, 60, 60, 60, 40, 40, 40, -10]

# EC-016: Group B avg when one member (row 4) is null.
GROUP_B_AVG_EXPECTED: float = 140.0 / 3.0

# EC-019: Row 11 has region=None, value_int=-10. Forms its own group.
NULL_GROUP_SUM_EXPECTED: int = -10


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class WindowAggregationTestBase(MaskTestMixin, DataOpsTestBase):
    """Abstract base class for window aggregation framework tests."""

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
        return "value_int__sum_window"

    @classmethod
    def mask_partition_by(cls) -> list[str] | None:
        return ["region"]

    @classmethod
    def mask_equal_expected(cls) -> list[Any]:
        # category='X': A: 10+0=10, B: 60, C: 15, None: -10
        return [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, -10]

    @classmethod
    def mask_multiple_conditions_expected(cls) -> list[Any]:
        # category='X' AND value_int>=10: A: 10, B: 60, C: 15, None: None (-10<10)
        return [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, None]

    @classmethod
    def mask_is_in_expected(cls) -> list[Any]:
        # region is_in ['A','C']: A=all match sum=25, B=none match=None, C=all match sum=70, None=no match=None
        return [25, 25, 25, 25, None, None, None, None, 70, 70, 70, None]

    @classmethod
    def mask_greater_than_expected(cls) -> list[Any]:
        # value_int > 10: A: [20]=20, B: [50,30,60]=140, C: [15,15,40]=70, None: []=None
        return [20, 20, 20, 20, 140, 140, 140, 140, 70, 70, 70, None]

    @classmethod
    def mask_no_mask_expected(cls) -> list[Any]:
        return list(EXPECTED_SUM_BY_REGION)

    @classmethod
    def reference_implementation_class(cls) -> Any:
        from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.reference import (
            ReferenceWindowAggregation,
        )

        return ReferenceWindowAggregation

    # -- Concrete test methods (inherited for free) --------------------------

    def test_sum_window_region(self) -> None:
        """Sum of value_int partitioned by region, broadcast back to every row."""
        fs = make_feature_set("value_int__sum_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__sum_window")
        assert result_col == EXPECTED_SUM_BY_REGION

    def test_avg_window_region(self) -> None:
        """Average of value_int partitioned by region, broadcast back to every row."""
        fs = make_feature_set("value_int__avg_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__avg_window")
        assert result_col == pytest.approx(EXPECTED_AVG_BY_REGION, rel=1e-3)

    def test_count_window_region(self) -> None:
        """Count of non-null value_int partitioned by region."""
        fs = make_feature_set("value_int__count_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__count_window")
        assert result_col == EXPECTED_COUNT_BY_REGION

    def test_min_window_region(self) -> None:
        """Minimum of value_int partitioned by region."""
        fs = make_feature_set("value_int__min_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__min_window")
        assert result_col == EXPECTED_MIN_BY_REGION

    def test_max_window_region(self) -> None:
        """Maximum of value_int partitioned by region."""
        fs = make_feature_set("value_int__max_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__max_window")
        assert result_col == EXPECTED_MAX_BY_REGION

    def test_null_policy_skip_avg_with_null_values(self) -> None:
        """NullPolicy.SKIP: Group B has a null value_int at row 4. Avg should skip it."""
        fs = make_feature_set("value_int__avg_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__avg_window")
        assert result_col[4] == pytest.approx(GROUP_B_AVG_EXPECTED, rel=1e-6)
        assert result_col[5] == pytest.approx(GROUP_B_AVG_EXPECTED, rel=1e-6)
        assert result_col[6] == pytest.approx(GROUP_B_AVG_EXPECTED, rel=1e-6)
        assert result_col[7] == pytest.approx(GROUP_B_AVG_EXPECTED, rel=1e-6)

    def test_null_policy_null_is_group(self) -> None:
        """NullPolicy.NULL_IS_GROUP: Row 11 has region=None. It should form its own group."""
        fs = make_feature_set("value_int__sum_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__sum_window")
        assert result_col[11] == NULL_GROUP_SUM_EXPECTED

    def test_output_rows_equal_input_rows(self) -> None:
        """Output must have exactly 12 rows, same as input."""
        fs = make_feature_set("value_int__sum_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

    def test_new_column_added(self) -> None:
        """The aggregation result column should be added to the output."""
        fs = make_feature_set("value_int__max_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_cols = self.extract_column(result, "value_int__max_window")
        assert len(result_cols) == 12

    def test_result_has_correct_type(self) -> None:
        """The result of calculate_feature must be the expected framework type."""
        fs = make_feature_set("value_int__min_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())

    # -- Cross-framework comparison (matches reference) --------------

    def test_cross_framework_sum(self) -> None:
        """Sum must match reference."""
        self._compare_with_reference("value_int__sum_window", partition_by=["region"])

    def test_cross_framework_avg(self) -> None:
        """Avg must match reference."""
        self._compare_with_reference("value_int__avg_window", partition_by=["region"], use_approx=True)

    def test_cross_framework_count(self) -> None:
        """Count must match reference."""
        self._compare_with_reference("value_int__count_window", partition_by=["region"])

    def test_cross_framework_min(self) -> None:
        """Min must match reference."""
        self._compare_with_reference("value_int__min_window", partition_by=["region"])

    def test_cross_framework_max(self) -> None:
        """Max must match reference."""
        self._compare_with_reference("value_int__max_window", partition_by=["region"])

    def test_cross_framework_std(self) -> None:
        """Std must match reference."""
        self._skip_if_unsupported("std")
        self._compare_with_reference("value_int__std_window", partition_by=["region"], use_approx=True)

    def test_cross_framework_var(self) -> None:
        """Var must match reference."""
        self._skip_if_unsupported("var")
        self._compare_with_reference("value_int__var_window", partition_by=["region"], use_approx=True)

    def test_cross_framework_median(self) -> None:
        """Median must match reference."""
        self._skip_if_unsupported("median")
        self._compare_with_reference("value_int__median_window", partition_by=["region"], use_approx=True)

    def test_cross_framework_nunique(self) -> None:
        """Nunique must match reference."""
        self._skip_if_unsupported("nunique")
        self._compare_with_reference("value_int__nunique_window", partition_by=["region"])

    def test_cross_framework_first(self) -> None:
        """First must match reference (with order_by)."""
        self._skip_if_unsupported("first")
        self._compare_with_reference("value_int__first_window", partition_by=["region"], order_by="value_int")

    def test_cross_framework_last(self) -> None:
        """Last must match reference (with order_by)."""
        self._skip_if_unsupported("last")
        self._compare_with_reference("value_int__last_window", partition_by=["region"], order_by="value_int")

    def test_cross_framework_mode(self) -> None:
        """Mode must match reference.

        PyArrow picks the first-encountered value on ties.
        Frameworks that use different tie-breaking must exclude
        'mode' from supported_agg_types().
        """
        self._skip_if_unsupported("mode")
        self._compare_with_reference("value_int__mode_window", partition_by=["region"])

    # -- Statistical aggregation tests (skipped if unsupported) --------------

    def test_std_window_region(self) -> None:
        """Population standard deviation of value_int partitioned by region (ddof=0)."""
        self._skip_if_unsupported("std")
        fs = make_feature_set("value_int__std_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__std_window")
        # Group A values: [10, -5, 0, 20] -> population std (ddof=0)
        a_vals = [10, -5, 0, 20]
        a_std = (sum((x - 6.25) ** 2 for x in a_vals) / len(a_vals)) ** 0.5
        assert result_col[0] == pytest.approx(a_std, rel=1e-6)
        assert result_col[1] == pytest.approx(a_std, rel=1e-6)
        assert result_col[2] == pytest.approx(a_std, rel=1e-6)
        assert result_col[3] == pytest.approx(a_std, rel=1e-6)

    def test_var_window_region(self) -> None:
        """Population variance of value_int partitioned by region (ddof=0)."""
        self._skip_if_unsupported("var")
        fs = make_feature_set("value_int__var_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__var_window")
        a_vals = [10, -5, 0, 20]
        a_var = sum((x - 6.25) ** 2 for x in a_vals) / len(a_vals)
        assert result_col[0] == pytest.approx(a_var, rel=1e-6)
        assert result_col[1] == pytest.approx(a_var, rel=1e-6)

    def test_std_pop_window_region(self) -> None:
        """Explicit population standard deviation (ddof=0), identical to std."""
        self._skip_if_unsupported("std_pop")
        fs = make_feature_set("value_int__std_pop_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__std_pop_window")
        a_vals = [10, -5, 0, 20]
        a_std = (sum((x - 6.25) ** 2 for x in a_vals) / len(a_vals)) ** 0.5
        assert result_col[0] == pytest.approx(a_std, rel=1e-6)

    def test_std_samp_window_region(self) -> None:
        """Sample standard deviation (ddof=1)."""
        self._skip_if_unsupported("std_samp")
        fs = make_feature_set("value_int__std_samp_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__std_samp_window")
        a_vals = [10, -5, 0, 20]
        a_std = (sum((x - 6.25) ** 2 for x in a_vals) / (len(a_vals) - 1)) ** 0.5
        assert result_col[0] == pytest.approx(a_std, rel=1e-6)

    def test_var_pop_window_region(self) -> None:
        """Explicit population variance (ddof=0), identical to var."""
        self._skip_if_unsupported("var_pop")
        fs = make_feature_set("value_int__var_pop_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__var_pop_window")
        a_vals = [10, -5, 0, 20]
        a_var = sum((x - 6.25) ** 2 for x in a_vals) / len(a_vals)
        assert result_col[0] == pytest.approx(a_var, rel=1e-6)

    def test_var_samp_window_region(self) -> None:
        """Sample variance (ddof=1)."""
        self._skip_if_unsupported("var_samp")
        fs = make_feature_set("value_int__var_samp_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__var_samp_window")
        a_vals = [10, -5, 0, 20]
        a_var = sum((x - 6.25) ** 2 for x in a_vals) / (len(a_vals) - 1)
        assert result_col[0] == pytest.approx(a_var, rel=1e-6)

    def test_median_window_region(self) -> None:
        """Median of value_int partitioned by region."""
        self._skip_if_unsupported("median")
        fs = make_feature_set("value_int__median_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__median_window")
        # Group A sorted: [-5, 0, 10, 20] -> median = 5.0
        assert result_col[0] == pytest.approx(5.0, rel=1e-6)
        # Group B sorted (non-null): [30, 50, 60] -> median = 50
        assert result_col[4] == pytest.approx(50.0, rel=1e-6)
        # Group C sorted: [15, 15, 40] -> median = 15
        assert result_col[8] == pytest.approx(15.0, rel=1e-6)
        # None group: [-10] -> median = -10
        assert result_col[11] == pytest.approx(-10.0, rel=1e-6)

    # -- Advanced aggregation tests (skipped if unsupported) -----------------

    def test_mode_window_region(self) -> None:
        """Mode of value_int partitioned by region.

        reference: [10,10,10,10, 50,50,50,50, 15,15,15, -10].
        On ties (all-unique groups A, B), PyArrow picks the first-encountered value.
        """
        self._skip_if_unsupported("mode")
        fs = make_feature_set("value_int__mode_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__mode_window")
        assert result_col == [10, 10, 10, 10, 50, 50, 50, 50, 15, 15, 15, -10]

    def test_nunique_window_region(self) -> None:
        """Count of unique non-null value_int values partitioned by region.

        reference: nulls are excluded from the distinct count.
        Group B has {None, 50, 30, 60}: 3 distinct non-null values.
        """
        self._skip_if_unsupported("nunique")
        fs = make_feature_set("value_int__nunique_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__nunique_window")
        assert result_col[0] == 4  # Group A: {10, -5, 0, 20} (no nulls)
        assert result_col[4] == 3  # Group B: 3 non-null unique values (null skipped)
        assert result_col[8] == 2  # Group C: {15, 40} (no nulls in non-null values)
        assert result_col[11] == 1  # None group: {-10}

    def test_first_window_region(self) -> None:
        """First non-null value_int per region, ordered by value_int ascending.

        Group A: sorted non-null = [-5, 0, 10, 20] -> first=-5
        Group B: sorted non-null = [30, 50, 60] -> first=30
        Group C: sorted non-null = [15, 15, 40] -> first=15
        None:    sorted non-null = [-10] -> first=-10
        """
        self._skip_if_unsupported("first")
        fs = make_feature_set("value_int__first_window", ["region"], order_by="value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__first_window")
        assert result_col == EXPECTED_FIRST_BY_REGION

    def test_last_window_region(self) -> None:
        """Last non-null value_int per region, ordered by value_int ascending.

        Group A: sorted non-null = [-5, 0, 10, 20] -> last=20
        Group B: sorted non-null = [30, 50, 60] -> last=60
        Group C: sorted non-null = [15, 15, 40] -> last=40
        None:    sorted non-null = [-10] -> last=-10
        """
        self._skip_if_unsupported("last")
        fs = make_feature_set("value_int__last_window", ["region"], order_by="value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__last_window")
        assert result_col == EXPECTED_LAST_BY_REGION

    # -- Null edge case tests ------------------------------------------------

    def test_null_policy_skip_all_null_column(self) -> None:
        """NullPolicy.SKIP: score column is all null. Aggregation should produce all nulls or zeros.

        PyArrow/Polars/DuckDB/SQLite return null for sum of all-null group.
        Pandas returns 0 (groupby.transform("sum") treats all-null as 0).
        """
        fs = make_feature_set("score__sum_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__sum_window")
        assert all(v is None for v in result_col)

    def test_all_null_column_std(self) -> None:
        """score is all-null. Std per group should be None."""
        self._skip_if_unsupported("std")
        fs = make_feature_set("score__std_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__std_window")
        assert all(v is None for v in result_col)

    def test_all_null_column_var(self) -> None:
        """score is all-null. Var per group should be None."""
        self._skip_if_unsupported("var")
        fs = make_feature_set("score__var_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__var_window")
        assert all(v is None for v in result_col)

    def test_all_null_column_first(self) -> None:
        """score is all-null. First per group should be None."""
        self._skip_if_unsupported("first")
        fs = make_feature_set("score__first_window", ["region"], order_by="score")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__first_window")
        assert all(v is None for v in result_col)

    def test_all_null_column_last(self) -> None:
        """score is all-null. Last per group should be None."""
        self._skip_if_unsupported("last")
        fs = make_feature_set("score__last_window", ["region"], order_by="score")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__last_window")
        assert all(v is None for v in result_col)

    def test_all_null_column_nunique(self) -> None:
        """score is all-null. Nunique per group should be 0 (no distinct non-null values).

        reference: count_distinct excludes nulls, returning 0.
        """
        self._skip_if_unsupported("nunique")
        fs = make_feature_set("score__nunique_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__nunique_window")
        assert all(v == 0 for v in result_col)

    # -- Multi-key partition tests -------------------------------------------

    def test_multi_key_partition_sum(self) -> None:
        """Sum of value_int partitioned by [region, category]."""
        fs = make_feature_set("value_int__sum_window", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__sum_window")
        assert result_col[0] == 10  # A/X: [10, 0]
        assert result_col[2] == 10
        assert result_col[1] == 15  # A/Y: [-5, 20]
        assert result_col[3] == 15
        assert result_col[4] == 60  # B/X: [None, 60]
        assert result_col[7] == 60
        assert result_col[5] == 50  # B/Y: [50]
        assert result_col[6] == 30  # B/None: [30]

    def test_multi_key_partition_count(self) -> None:
        """Count of non-null value_int partitioned by [region, category]."""
        fs = make_feature_set("value_int__count_window", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__count_window")
        assert result_col[0] == 2  # A/X: 2 non-null
        assert result_col[2] == 2
        assert result_col[1] == 2  # A/Y: 2 non-null
        assert result_col[3] == 2
        assert result_col[4] == 1  # B/X: 1 non-null (row 4 is null)
        assert result_col[7] == 1

    def test_multiple_null_order_by_first(self) -> None:
        """Two or more null order_by values must not crash when using first aggregation."""
        self._skip_if_unsupported("first")
        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "ts": [None, 1, None, 2],
                "value": [100, 10, 200, 20],
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("value__first_window", ["region"], order_by="ts")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "value__first_window")

        # first with order_by sorts by ts: [1->10, 2->20, None->100, None->200]
        # first non-null = 10, broadcast to all rows
        assert result_col == [10, 10, 10, 10]

    def test_multiple_null_order_by_last(self) -> None:
        """Two or more null order_by values must not crash when using last aggregation."""
        self._skip_if_unsupported("last")
        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "ts": [None, 1, None, 2],
                "value": [100, 10, 200, 20],
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("value__last_window", ["region"], order_by="ts")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "value__last_window")

        # last with order_by sorts by ts: [1->10, 2->20, None->100, None->200]
        # last non-null = 200, broadcast to all rows
        assert result_col == [200, 200, 200, 200]

    def test_multi_key_float_avg(self) -> None:
        """Avg of value_float partitioned by [region, category]."""
        fs = make_feature_set("value_float__avg_window", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_float__avg_window")
        # A/X: [1.5, None] -> avg = 1.5
        assert result_col[0] == pytest.approx(1.5, rel=1e-6)
        assert result_col[2] == pytest.approx(1.5, rel=1e-6)
        # A/Y: [2.5, 0.0] -> avg = 1.25
        assert result_col[1] == pytest.approx(1.25, rel=1e-6)
        assert result_col[3] == pytest.approx(1.25, rel=1e-6)

    def test_multi_key_partition_median(self) -> None:
        """Median of value_int partitioned by [region, category]."""
        self._skip_if_unsupported("median")
        fs = make_feature_set("value_int__median_window", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__median_window")
        # (A,X) rows 0,2: median([10, 0]) = 5.0
        assert result_col[0] == pytest.approx(5.0, rel=1e-6)
        assert result_col[2] == pytest.approx(5.0, rel=1e-6)
        # (A,Y) rows 1,3: median([-5, 20]) = 7.5
        assert result_col[1] == pytest.approx(7.5, rel=1e-6)
        assert result_col[3] == pytest.approx(7.5, rel=1e-6)
        # (C,Y) rows 8,10: median([15, 40]) = 27.5
        assert result_col[8] == pytest.approx(27.5, rel=1e-6)
        assert result_col[10] == pytest.approx(27.5, rel=1e-6)

    def test_multi_key_partition_std(self) -> None:
        """Population std of value_int partitioned by [region, category].

        Only groups with >= 2 non-null values are checked.
        """
        self._skip_if_unsupported("std")
        fs = make_feature_set("value_int__std_window", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__std_window")
        # (A,X) rows 0,2: std([10, 0]) = 5.0
        assert result_col[0] == pytest.approx(5.0, rel=1e-6)
        assert result_col[2] == pytest.approx(5.0, rel=1e-6)
        # (A,Y) rows 1,3: std([-5, 20]) = 12.5
        assert result_col[1] == pytest.approx(12.5, rel=1e-6)
        assert result_col[3] == pytest.approx(12.5, rel=1e-6)
        # (C,Y) rows 8,10: std([15, 40]) = 12.5
        assert result_col[8] == pytest.approx(12.5, rel=1e-6)
        assert result_col[10] == pytest.approx(12.5, rel=1e-6)

    def test_multi_key_partition_var(self) -> None:
        """Population variance of value_int partitioned by [region, category]."""
        self._skip_if_unsupported("var")
        fs = make_feature_set("value_int__var_window", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__var_window")
        # (A,X) rows 0,2: var([10, 0]) = 25.0
        assert result_col[0] == pytest.approx(25.0, rel=1e-6)
        assert result_col[2] == pytest.approx(25.0, rel=1e-6)
        # (A,Y) rows 1,3: var([-5, 20]) = 156.25
        assert result_col[1] == pytest.approx(156.25, rel=1e-6)
        assert result_col[3] == pytest.approx(156.25, rel=1e-6)
        # (C,Y) rows 8,10: var([15, 40]) = 156.25
        assert result_col[8] == pytest.approx(156.25, rel=1e-6)
        assert result_col[10] == pytest.approx(156.25, rel=1e-6)

    def test_multi_key_partition_nunique(self) -> None:
        """Nunique of value_int partitioned by [region, category]."""
        self._skip_if_unsupported("nunique")
        fs = make_feature_set("value_int__nunique_window", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__nunique_window")
        assert result_col[0] == 2  # (A,X): {10, 0}
        assert result_col[2] == 2
        assert result_col[1] == 2  # (A,Y): {-5, 20}
        assert result_col[3] == 2
        assert result_col[4] == 1  # (B,X): {60} (null excluded)
        assert result_col[7] == 1
        assert result_col[8] == 2  # (C,Y): {15, 40}
        assert result_col[10] == 2

    def test_multi_key_partition_mode(self) -> None:
        """Mode of value_int partitioned by [region, category].

        reference: [10,-5,10,-5, 60,50,30,60, 15,15,15, -10].
        On ties, PyArrow picks the first-encountered value.
        """
        self._skip_if_unsupported("mode")
        fs = make_feature_set("value_int__mode_window", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__mode_window")
        assert result_col == [10, -5, 10, -5, 60, 50, 30, 60, 15, 15, 15, -10]

    def test_multi_key_partition_first(self) -> None:
        """First non-null value_int per [region, category], ordered by value_int."""
        self._skip_if_unsupported("first")
        fs = make_feature_set("value_int__first_window", ["region", "category"], order_by="value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__first_window")
        # (A,X) rows 0,2: sorted [0, 10] -> first = 0
        assert result_col[0] == 0
        assert result_col[2] == 0
        # (A,Y) rows 1,3: sorted [-5, 20] -> first = -5
        assert result_col[1] == -5
        assert result_col[3] == -5
        # (B,X) rows 4,7: sorted [60] (null excluded) -> first = 60
        assert result_col[7] == 60
        # (C,Y) rows 8,10: sorted [15, 40] -> first = 15
        assert result_col[8] == 15
        assert result_col[10] == 15

    def test_multi_key_partition_last(self) -> None:
        """Last non-null value_int per [region, category], ordered by value_int."""
        self._skip_if_unsupported("last")
        fs = make_feature_set("value_int__last_window", ["region", "category"], order_by="value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__last_window")
        # (A,X) rows 0,2: sorted [0, 10] -> last = 10
        assert result_col[0] == 10
        assert result_col[2] == 10
        # (A,Y) rows 1,3: sorted [-5, 20] -> last = 20
        assert result_col[1] == 20
        assert result_col[3] == 20
        # (B,X) rows 4,7: sorted [60] -> last = 60
        assert result_col[7] == 60
        # (C,Y) rows 8,10: sorted [15, 40] -> last = 40
        assert result_col[8] == 40
        assert result_col[10] == 40

    # -- Mask tests ------------------------------------------------------------

    def test_mask_sum_window_equal(self) -> None:
        """Sum of value_int where category='X', partitioned by region."""
        fs = make_feature_set("value_int__sum_window", ["region"], mask=("category", "equal", "X"))
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__sum_window")
        assert result_col == [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, -10]

    def test_mask_avg_window_equal(self) -> None:
        """Avg of value_int where category='X', partitioned by region."""
        fs = make_feature_set("value_int__avg_window", ["region"], mask=("category", "equal", "X"))
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__avg_window")
        assert result_col == pytest.approx(
            [5.0, 5.0, 5.0, 5.0, 60.0, 60.0, 60.0, 60.0, 15.0, 15.0, 15.0, -10.0], rel=1e-3
        )

    def test_mask_count_window_equal(self) -> None:
        """Count of non-null value_int where category='X', partitioned by region."""
        fs = make_feature_set("value_int__count_window", ["region"], mask=("category", "equal", "X"))
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__count_window")
        assert result_col == [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]

    def test_mask_min_window_equal(self) -> None:
        """Min of value_int where category='X', partitioned by region."""
        fs = make_feature_set("value_int__min_window", ["region"], mask=("category", "equal", "X"))
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__min_window")
        assert result_col == [0, 0, 0, 0, 60, 60, 60, 60, 15, 15, 15, -10]

    def test_mask_max_window_equal(self) -> None:
        """Max of value_int where category='X', partitioned by region."""
        fs = make_feature_set("value_int__max_window", ["region"], mask=("category", "equal", "X"))
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__max_window")
        assert result_col == [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, -10]

    def test_mask_multiple_conditions(self) -> None:
        """Sum with AND-combined mask: category='X' AND value_int >= 10."""
        fs = make_feature_set(
            "value_int__sum_window",
            ["region"],
            mask=[("category", "equal", "X"), ("value_int", "greater_equal", 10)],
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__sum_window")
        # Region A: only row 0 (val=10, cat=X, >=10) -> 10
        # Region B: only row 7 (val=60, cat=X, >=10) -> 60
        # Region C: only row 9 (val=15, cat=X, >=10) -> 15
        # None: row 11 (val=-10, cat=X, <10) -> None (all masked out)
        assert result_col == [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, None]

    def test_mask_no_mask_unchanged(self) -> None:
        """Without mask, results match the standard unmasked test."""
        fs = make_feature_set("value_int__sum_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__sum_window")
        assert result_col == [25, 25, 25, 25, 140, 140, 140, 140, 70, 70, 70, -10]

    def test_mask_first_window_equal(self) -> None:
        """First of value_int where category='X', ordered by timestamp, partitioned by region."""
        self._skip_if_unsupported("first")
        fs = make_feature_set(
            "value_int__first_window", ["region"], order_by="timestamp", mask=("category", "equal", "X")
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__first_window")
        assert result_col == [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, -10]

    def test_mask_last_window_equal(self) -> None:
        """Last of value_int where category='X', ordered by timestamp, partitioned by region."""
        self._skip_if_unsupported("last")
        fs = make_feature_set(
            "value_int__last_window", ["region"], order_by="timestamp", mask=("category", "equal", "X")
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__last_window")
        assert result_col == [0, 0, 0, 0, 60, 60, 60, 60, 15, 15, 15, -10]

    def test_mask_nunique_window_equal(self) -> None:
        """Nunique of value_int where category='X', partitioned by region."""
        self._skip_if_unsupported("nunique")
        fs = make_feature_set("value_int__nunique_window", ["region"], mask=("category", "equal", "X"))
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__nunique_window")
        assert result_col == [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]

    def test_mask_fully_masked_window(self) -> None:
        """All rows masked out (category='Z') should produce None for every row."""
        fs = make_feature_set("value_int__sum_window", ["region"], mask=("category", "equal", "Z"))
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__sum_window")
        assert all(v is None for v in result_col)

    def test_mask_mode_window_equal(self) -> None:
        """Mode of value_int where category='X', partitioned by region (row-preserving)."""
        self._skip_if_unsupported("mode")
        fs = make_feature_set("value_int__mode_window", ["region"], mask=("category", "equal", "X"))
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__mode_window")
        assert result_col == [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, -10]

    # -- Option-based config tests -------------------------------------------

    def test_option_based_sum_window(self) -> None:
        """Option-based configuration (not string pattern) produces the same result."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "my_window_sum",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "in_features": "value_int",
                    "partition_by": ["region"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_window_sum")
        assert result_col == EXPECTED_SUM_BY_REGION

    def test_unsupported_aggregation_type_raises(self) -> None:
        """Calling calculate_feature with an unknown aggregation type should raise."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "value_int__evil_type_window",
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

    # -- Row-order preservation ------------------------------------------------

    def test_row_order_preserved_first(self) -> None:
        """Original columns must remain in input row order after first.

        PyArrow parity: DuckDB first/last uses ORDER BY which reorders
        rows; the ROW_NUMBER + .order() pattern must restore input order.

        Compares (value_int, category) tuples to detect tied-row swaps
        (rows 8 and 9 both have value_int=15).
        """
        self._skip_if_unsupported("first")
        fs = make_feature_set("value_int__first_window", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        input_id = list(
            zip(
                self.extract_column(self.test_data, "value_int"),
                self.extract_column(self.test_data, "category"),
            )
        )
        output_id = list(
            zip(
                self.extract_column(result, "value_int"),
                self.extract_column(result, "category"),
            )
        )
        assert output_id == input_id

    def test_row_order_preserved_sum(self) -> None:
        """Original columns must remain in input row order after sum.

        Compares (value_int, category) tuples to detect tied-row swaps.
        """
        fs = make_feature_set("value_int__sum_window", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        input_id = list(
            zip(
                self.extract_column(self.test_data, "value_int"),
                self.extract_column(self.test_data, "category"),
            )
        )
        output_id = list(
            zip(
                self.extract_column(result, "value_int"),
                self.extract_column(result, "category"),
            )
        )
        assert output_id == input_id
