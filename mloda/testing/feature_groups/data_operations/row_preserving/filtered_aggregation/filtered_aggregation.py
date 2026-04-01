"""Shared test base class, data, and helpers for filtered aggregation tests.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, partitioned by the 'region' column
(A=rows 0-3, B=rows 4-7, C=rows 8-10, None=row 11), and filtered by
category='X'.

The ``FilteredAggregationTestBase`` class provides concrete test methods
that any framework implementation inherits by subclassing and implementing
5 abstract adapter methods.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import extract_column as _extract_column
from mloda.user import Feature


# ---------------------------------------------------------------------------
# Expected values (module-level constants)
# ---------------------------------------------------------------------------
# filter_column='category', filter_value='X', partition_by=['region']
#
# Region A: matching rows 0(value_int=10), 2(value_int=0)
# Region B: matching rows 4(value_int=None), 7(value_int=60)
# Region C: matching row 9(value_int=15)
# Region None: matching row 11(value_int=-10)

EXPECTED_SUM_FILTERED: list[Any] = [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, -10]
EXPECTED_AVG_FILTERED: list[Any] = [5.0, 5.0, 5.0, 5.0, 60.0, 60.0, 60.0, 60.0, 15.0, 15.0, 15.0, -10.0]
EXPECTED_COUNT_FILTERED: list[int] = [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
EXPECTED_MIN_FILTERED: list[Any] = [0, 0, 0, 0, 60, 60, 60, 60, 15, 15, 15, -10]
EXPECTED_MAX_FILTERED: list[Any] = [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, -10]

# NullPolicy.SKIP: Group B has null value_int at row 4 where category='X'.
# Avg should skip it: only row 7 (value_int=60) counts.
GROUP_B_AVG_FILTERED_EXPECTED: float = 60.0

# NullPolicy.NULL_IS_GROUP: Row 11 has region=None, forms its own group.
NULL_GROUP_SUM_FILTERED_EXPECTED: int = -10

# Default filter parameters for tests
DEFAULT_FILTER_COLUMN = "category"
DEFAULT_FILTER_VALUE = "X"
DEFAULT_PARTITION_BY = ["region"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_filtered_feature_set(
    feature_name: str,
    partition_by: list[str] | None = None,
    filter_column: str | None = None,
    filter_value: Any = None,
) -> FeatureSet:
    """Build a FeatureSet with partition_by, filter_column, and filter_value options."""
    context: dict[str, Any] = {}
    if partition_by is not None:
        context["partition_by"] = partition_by
    if filter_column is not None:
        context["filter_column"] = filter_column
    if filter_value is not None:
        context["filter_value"] = filter_value
    feature = Feature(feature_name, options=Options(context=context))
    fs = FeatureSet()
    fs.add(feature)
    return fs


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class FilteredAggregationTestBase(DataOpsTestBase):
    """Abstract base class for filtered aggregation framework tests."""

    ALL_AGG_TYPES = {"sum", "avg", "count", "min", "max"}

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        """Aggregation types this framework supports. Override to restrict."""
        return cls.ALL_AGG_TYPES

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        from mloda.community.feature_groups.data_operations.row_preserving.filtered_aggregation.pyarrow_filtered_aggregation import (
            PyArrowFilteredAggregation,
        )

        return PyArrowFilteredAggregation

    def _compare_with_pyarrow(  # type: ignore[override]
        self,
        feature_name: str,
        *,
        partition_by: list[str] | None = None,
        filter_column: str | None = None,
        filter_value: Any = None,
        use_approx: bool = False,
        rel: float = 1e-6,
    ) -> None:
        """Run a feature through this framework and PyArrow, assert results match."""
        fs = make_filtered_feature_set(
            feature_name,
            partition_by=partition_by,
            filter_column=filter_column,
            filter_value=filter_value,
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ref = self.pyarrow_implementation_class().calculate_feature(self._arrow_table, fs)

        result_col = self.extract_column(result, feature_name)
        ref_col = _extract_column(ref, feature_name)

        assert len(result_col) == len(ref_col), f"row count {len(result_col)} != reference {len(ref_col)}"
        if use_approx:
            for i, (ref_val, fw_val) in enumerate(zip(ref_col, result_col)):
                if ref_val is None:
                    assert fw_val is None, f"row {i}: expected None, got {fw_val}"
                else:
                    assert fw_val == pytest.approx(ref_val, rel=rel), f"row {i}: {fw_val} != reference {ref_val}"
        else:
            assert result_col == ref_col

    # -- Concrete test methods (inherited for free) --------------------------

    def test_sum_filtered_region(self) -> None:
        """Sum of value_int where category='X', partitioned by region."""
        fs = make_filtered_feature_set(
            "value_int__sum_filtered_groupby", DEFAULT_PARTITION_BY, DEFAULT_FILTER_COLUMN, DEFAULT_FILTER_VALUE
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__sum_filtered_groupby")
        assert result_col == EXPECTED_SUM_FILTERED

    def test_avg_filtered_region(self) -> None:
        """Average of value_int where category='X', partitioned by region."""
        fs = make_filtered_feature_set(
            "value_int__avg_filtered_groupby", DEFAULT_PARTITION_BY, DEFAULT_FILTER_COLUMN, DEFAULT_FILTER_VALUE
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__avg_filtered_groupby")
        assert result_col == pytest.approx(EXPECTED_AVG_FILTERED, rel=1e-3)

    def test_count_filtered_region(self) -> None:
        """Count of non-null value_int where category='X', partitioned by region."""
        fs = make_filtered_feature_set(
            "value_int__count_filtered_groupby", DEFAULT_PARTITION_BY, DEFAULT_FILTER_COLUMN, DEFAULT_FILTER_VALUE
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__count_filtered_groupby")
        assert result_col == EXPECTED_COUNT_FILTERED

    def test_min_filtered_region(self) -> None:
        """Minimum of value_int where category='X', partitioned by region."""
        fs = make_filtered_feature_set(
            "value_int__min_filtered_groupby", DEFAULT_PARTITION_BY, DEFAULT_FILTER_COLUMN, DEFAULT_FILTER_VALUE
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__min_filtered_groupby")
        assert result_col == EXPECTED_MIN_FILTERED

    def test_max_filtered_region(self) -> None:
        """Maximum of value_int where category='X', partitioned by region."""
        fs = make_filtered_feature_set(
            "value_int__max_filtered_groupby", DEFAULT_PARTITION_BY, DEFAULT_FILTER_COLUMN, DEFAULT_FILTER_VALUE
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__max_filtered_groupby")
        assert result_col == EXPECTED_MAX_FILTERED

    def test_null_policy_skip_avg_with_null_values(self) -> None:
        """NullPolicy.SKIP: Group B has null value_int at row 4 where category='X'. Avg should skip it."""
        fs = make_filtered_feature_set(
            "value_int__avg_filtered_groupby", DEFAULT_PARTITION_BY, DEFAULT_FILTER_COLUMN, DEFAULT_FILTER_VALUE
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__avg_filtered_groupby")
        assert result_col[4] == pytest.approx(GROUP_B_AVG_FILTERED_EXPECTED, rel=1e-6)
        assert result_col[5] == pytest.approx(GROUP_B_AVG_FILTERED_EXPECTED, rel=1e-6)
        assert result_col[6] == pytest.approx(GROUP_B_AVG_FILTERED_EXPECTED, rel=1e-6)
        assert result_col[7] == pytest.approx(GROUP_B_AVG_FILTERED_EXPECTED, rel=1e-6)

    def test_null_policy_null_is_group(self) -> None:
        """NullPolicy.NULL_IS_GROUP: Row 11 has region=None. It should form its own group."""
        fs = make_filtered_feature_set(
            "value_int__sum_filtered_groupby", DEFAULT_PARTITION_BY, DEFAULT_FILTER_COLUMN, DEFAULT_FILTER_VALUE
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__sum_filtered_groupby")
        assert result_col[11] == NULL_GROUP_SUM_FILTERED_EXPECTED

    def test_output_rows_equal_input_rows(self) -> None:
        """Output must have exactly 12 rows, same as input."""
        fs = make_filtered_feature_set(
            "value_int__sum_filtered_groupby", DEFAULT_PARTITION_BY, DEFAULT_FILTER_COLUMN, DEFAULT_FILTER_VALUE
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

    def test_new_column_added(self) -> None:
        """The aggregation result column should be added to the output."""
        fs = make_filtered_feature_set(
            "value_int__max_filtered_groupby", DEFAULT_PARTITION_BY, DEFAULT_FILTER_COLUMN, DEFAULT_FILTER_VALUE
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_cols = self.extract_column(result, "value_int__max_filtered_groupby")
        assert len(result_cols) == 12

    def test_result_has_correct_type(self) -> None:
        """The result of calculate_feature must be the expected framework type."""
        fs = make_filtered_feature_set(
            "value_int__min_filtered_groupby", DEFAULT_PARTITION_BY, DEFAULT_FILTER_COLUMN, DEFAULT_FILTER_VALUE
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())

    # -- Cross-framework comparison (matches PyArrow reference) --------------

    def test_cross_framework_sum(self) -> None:
        """Sum must match PyArrow reference."""
        self._compare_with_pyarrow(
            "value_int__sum_filtered_groupby",
            partition_by=DEFAULT_PARTITION_BY,
            filter_column=DEFAULT_FILTER_COLUMN,
            filter_value=DEFAULT_FILTER_VALUE,
        )

    def test_cross_framework_avg(self) -> None:
        """Avg must match PyArrow reference."""
        self._compare_with_pyarrow(
            "value_int__avg_filtered_groupby",
            partition_by=DEFAULT_PARTITION_BY,
            filter_column=DEFAULT_FILTER_COLUMN,
            filter_value=DEFAULT_FILTER_VALUE,
            use_approx=True,
        )

    def test_cross_framework_count(self) -> None:
        """Count must match PyArrow reference."""
        self._compare_with_pyarrow(
            "value_int__count_filtered_groupby",
            partition_by=DEFAULT_PARTITION_BY,
            filter_column=DEFAULT_FILTER_COLUMN,
            filter_value=DEFAULT_FILTER_VALUE,
        )

    def test_cross_framework_min(self) -> None:
        """Min must match PyArrow reference."""
        self._compare_with_pyarrow(
            "value_int__min_filtered_groupby",
            partition_by=DEFAULT_PARTITION_BY,
            filter_column=DEFAULT_FILTER_COLUMN,
            filter_value=DEFAULT_FILTER_VALUE,
        )

    def test_cross_framework_max(self) -> None:
        """Max must match PyArrow reference."""
        self._compare_with_pyarrow(
            "value_int__max_filtered_groupby",
            partition_by=DEFAULT_PARTITION_BY,
            filter_column=DEFAULT_FILTER_COLUMN,
            filter_value=DEFAULT_FILTER_VALUE,
        )

    # -- Null edge case tests ------------------------------------------------

    def test_null_policy_skip_all_null_column(self) -> None:
        """NullPolicy.SKIP: score column is all null. Filtered aggregation should produce all nulls."""
        fs = make_filtered_feature_set(
            "score__sum_filtered_groupby", DEFAULT_PARTITION_BY, DEFAULT_FILTER_COLUMN, DEFAULT_FILTER_VALUE
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__sum_filtered_groupby")
        assert all(v is None for v in result_col)

    # -- Multi-key partition tests -------------------------------------------

    def test_multi_key_partition_sum(self) -> None:
        """Sum of value_int where category='X', partitioned by [region, category].

        When partition_by includes the filter column, partitions where
        category != 'X' have no matching rows and receive null.
        """
        fs = make_filtered_feature_set(
            "value_int__sum_filtered_groupby",
            ["region", "category"],
            DEFAULT_FILTER_COLUMN,
            DEFAULT_FILTER_VALUE,
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__sum_filtered_groupby")
        # A/X (rows 0, 2): both match filter -> sum of [10, 0] = 10
        assert result_col[0] == 10
        assert result_col[2] == 10
        # A/Y (rows 1, 3): no matching rows -> null
        assert result_col[1] is None
        assert result_col[3] is None
        # B/X (rows 4, 7): both match -> sum of [None, 60] = 60 (null skipped)
        assert result_col[4] == 60
        assert result_col[7] == 60
        # B/Y (row 5): no match -> null
        assert result_col[5] is None
        # B/None (row 6): no match (None != 'X') -> null
        assert result_col[6] is None

    # -- Option-based config tests -------------------------------------------

    def test_option_based_sum_filtered(self) -> None:
        """Option-based configuration (not string pattern) produces the same result."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "my_filtered_sum",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "in_features": "value_int",
                    "partition_by": ["region"],
                    "filter_column": "category",
                    "filter_value": "X",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_filtered_sum")
        assert result_col == EXPECTED_SUM_FILTERED

    def test_unsupported_aggregation_type_raises(self) -> None:
        """Calling calculate_feature with an unknown aggregation type should raise."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "value_int__evil_type_filtered_groupby",
            options=Options(
                context={
                    "partition_by": ["region"],
                    "filter_column": "category",
                    "filter_value": "X",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises((ValueError, KeyError)):
            self.implementation_class().calculate_feature(self.test_data, fs)
