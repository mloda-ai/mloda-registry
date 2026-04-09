"""Shared test base class and expected values for percentile tests.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, partitioned by the 'region' column
(A=rows 0-3, B=rows 4-7, C=rows 8-10, None=row 11).

Uses PERCENTILE_CONT with linear interpolation.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.user import Feature


# ---------------------------------------------------------------------------
# Expected values (module-level constants)
# ---------------------------------------------------------------------------

# value_int per region (non-null, sorted):
# A (rows 0-3): [-5, 0, 10, 20]
# B (rows 4-7): [30, 50, 60]  (row 4 is null)
# C (rows 8-10): [15, 15, 40]
# None (row 11): [-10]
#
# PERCENTILE_CONT linear interpolation:
#   index = (n-1) * q, then interpolate between surrounding values.

EXPECTED_P50_BY_REGION: list[float] = [5.0, 5.0, 5.0, 5.0, 50.0, 50.0, 50.0, 50.0, 15.0, 15.0, 15.0, -10.0]
EXPECTED_P25_BY_REGION: list[float] = [-1.25, -1.25, -1.25, -1.25, 40.0, 40.0, 40.0, 40.0, 15.0, 15.0, 15.0, -10.0]
EXPECTED_P75_BY_REGION: list[float] = [12.5, 12.5, 12.5, 12.5, 55.0, 55.0, 55.0, 55.0, 27.5, 27.5, 27.5, -10.0]
EXPECTED_P0_BY_REGION: list[float] = [-5.0, -5.0, -5.0, -5.0, 30.0, 30.0, 30.0, 30.0, 15.0, 15.0, 15.0, -10.0]
EXPECTED_P100_BY_REGION: list[float] = [20.0, 20.0, 20.0, 20.0, 60.0, 60.0, 60.0, 60.0, 40.0, 40.0, 40.0, -10.0]

# Multi-key partition [region, category] p50 expected values:
# A/X: rows 0,2 values [10,0] sorted=[0,10] p50=5.0
# A/Y: rows 1,3 values [-5,20] sorted=[-5,20] p50=7.5
# B/X: rows 4,7 values [None,60] sorted=[60] p50=60.0
# B/Y: row 5 values [50] p50=50.0
# B/None: row 6 values [30] p50=30.0
# C/Y: rows 8,10 values [15,40] sorted=[15,40] p50=27.5
# C/X: row 9 values [15] p50=15.0
# None/X: row 11 values [-10] p50=-10.0


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class PercentileTestBase(DataOpsTestBase):
    """Abstract base class for percentile framework tests."""

    @classmethod
    def reference_implementation_class(cls) -> Any:
        from mloda.testing.feature_groups.data_operations.row_preserving.percentile.reference import (
            ReferencePercentile,
        )

        return ReferencePercentile

    # -- Concrete test methods (inherited for free) --------------------------

    def test_p50_percentile_region(self) -> None:
        """P50 of value_int partitioned by region, broadcast back to every row."""
        fs = make_feature_set("value_int__p50_percentile", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__p50_percentile")
        assert result_col == pytest.approx(EXPECTED_P50_BY_REGION, rel=1e-6)

    def test_p25_percentile_region(self) -> None:
        """P25 of value_int partitioned by region."""
        fs = make_feature_set("value_int__p25_percentile", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__p25_percentile")
        assert result_col == pytest.approx(EXPECTED_P25_BY_REGION, rel=1e-6)

    def test_p75_percentile_region(self) -> None:
        """P75 of value_int partitioned by region."""
        fs = make_feature_set("value_int__p75_percentile", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__p75_percentile")
        assert result_col == pytest.approx(EXPECTED_P75_BY_REGION, rel=1e-6)

    def test_p0_percentile_region(self) -> None:
        """P0 equals the minimum of value_int per region."""
        fs = make_feature_set("value_int__p0_percentile", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__p0_percentile")
        assert result_col == pytest.approx(EXPECTED_P0_BY_REGION, rel=1e-6)

    def test_p100_percentile_region(self) -> None:
        """P100 equals the maximum of value_int per region."""
        fs = make_feature_set("value_int__p100_percentile", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__p100_percentile")
        assert result_col == pytest.approx(EXPECTED_P100_BY_REGION, rel=1e-6)

    def test_null_policy_skip_p50_with_null_values(self) -> None:
        """NullPolicy.SKIP: Group B has a null value_int at row 4. P50 should skip it."""
        fs = make_feature_set("value_int__p50_percentile", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__p50_percentile")
        assert result_col[4] == pytest.approx(50.0, rel=1e-6)
        assert result_col[5] == pytest.approx(50.0, rel=1e-6)
        assert result_col[6] == pytest.approx(50.0, rel=1e-6)
        assert result_col[7] == pytest.approx(50.0, rel=1e-6)

    def test_null_policy_null_is_group(self) -> None:
        """NullPolicy.NULL_IS_GROUP: Row 11 has region=None. It should form its own group."""
        fs = make_feature_set("value_int__p50_percentile", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__p50_percentile")
        assert result_col[11] == pytest.approx(-10.0, rel=1e-6)

    def test_output_rows_equal_input_rows(self) -> None:
        """Output must have exactly 12 rows, same as input."""
        fs = make_feature_set("value_int__p50_percentile", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

    def test_new_column_added(self) -> None:
        """The percentile result column should be added to the output."""
        fs = make_feature_set("value_int__p50_percentile", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__p50_percentile")
        assert len(result_col) == 12

    def test_result_has_correct_type(self) -> None:
        """The result of calculate_feature must be the expected framework type."""
        fs = make_feature_set("value_int__p50_percentile", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())

    def test_null_policy_skip_all_null_column(self) -> None:
        """NullPolicy.SKIP: score column is all null. Percentile should produce all nulls."""
        fs = make_feature_set("score__p50_percentile", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "score__p50_percentile")
        assert all(v is None for v in result_col)

    def test_multi_key_partition_p50(self) -> None:
        """P50 of value_int partitioned by [region, category]."""
        fs = make_feature_set("value_int__p50_percentile", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__p50_percentile")
        assert result_col[0] == pytest.approx(5.0, rel=1e-6)  # A/X
        assert result_col[2] == pytest.approx(5.0, rel=1e-6)  # A/X
        assert result_col[1] == pytest.approx(7.5, rel=1e-6)  # A/Y
        assert result_col[3] == pytest.approx(7.5, rel=1e-6)  # A/Y
        assert result_col[4] == pytest.approx(60.0, rel=1e-6)  # B/X (null skipped)
        assert result_col[7] == pytest.approx(60.0, rel=1e-6)  # B/X
        assert result_col[5] == pytest.approx(50.0, rel=1e-6)  # B/Y
        assert result_col[6] == pytest.approx(30.0, rel=1e-6)  # B/None
        assert result_col[8] == pytest.approx(27.5, rel=1e-6)  # C/Y
        assert result_col[10] == pytest.approx(27.5, rel=1e-6)  # C/Y
        assert result_col[9] == pytest.approx(15.0, rel=1e-6)  # C/X
        assert result_col[11] == pytest.approx(-10.0, rel=1e-6)  # None/X

    def test_option_based_p50_percentile(self) -> None:
        """Option-based configuration (not string pattern) produces the same result."""
        feature = Feature(
            "my_custom_percentile",
            options=Options(
                context={
                    "percentile": 0.5,
                    "in_features": "value_int",
                    "partition_by": ["region"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_custom_percentile")
        assert result_col == pytest.approx(EXPECTED_P50_BY_REGION, rel=1e-6)

    # -- Float column tests ----------------------------------------------------

    def test_multi_key_float_p50(self) -> None:
        """P50 of value_float partitioned by [region, category]."""
        fs = make_feature_set("value_float__p50_percentile", ["region", "category"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_float__p50_percentile")
        # A/X: [1.5, None] -> p50 = 1.5
        assert result_col[0] == pytest.approx(1.5, rel=1e-6)
        assert result_col[2] == pytest.approx(1.5, rel=1e-6)
        # A/Y: [2.5, 0.0] -> sorted = [0.0, 2.5] -> p50 = 1.25
        assert result_col[1] == pytest.approx(1.25, rel=1e-6)
        assert result_col[3] == pytest.approx(1.25, rel=1e-6)

    # -- Multi-column in_features rejection ------------------------------------

    def test_multi_column_in_features_rejected_at_calculate(self) -> None:
        """calculate_feature must reject features with multiple in_features."""
        feature = Feature(
            "bad_multi",
            options=Options(
                context={
                    "percentile": 0.5,
                    "in_features": ["value_int", "value_float"],
                    "partition_by": ["region"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match="at most 1"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    # -- Null consistency tests (multi-null columns) ---------------------------

    def test_multi_null_column_p50(self) -> None:
        """value_float has 2 nulls (rows 2, 11). Percentile should skip them."""
        fs = make_feature_set("value_float__p50_percentile", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_float__p50_percentile")
        # A: value_float non-null = [1.5, 0.0, 2.5] -> sorted [0.0, 1.5, 2.5] -> p50 = 1.5
        assert result_col[0] == pytest.approx(1.5, rel=1e-6)
        # None group: value_float = [None] -> all null -> result is None
        assert result_col[11] is None

    def test_amount_p50_by_region(self) -> None:
        """amount has 2 nulls (rows 1, 7). Percentile should skip them."""
        fs = make_feature_set("amount__p50_percentile", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "amount__p50_percentile")
        # A: amount non-null = [100.0, 250.0, 75.0] -> sorted [75.0, 100.0, 250.0] -> p50 = 100.0
        assert result_col[0] == pytest.approx(100.0, rel=1e-6)

    # -- Cross-framework comparison (matches reference) ----------------

    def test_cross_framework_p50(self) -> None:
        """P50 must match reference."""
        self._compare_with_reference("value_int__p50_percentile", partition_by=["region"], use_approx=True)

    def test_cross_framework_p25(self) -> None:
        """P25 must match reference."""
        self._compare_with_reference("value_int__p25_percentile", partition_by=["region"], use_approx=True)

    def test_cross_framework_p75(self) -> None:
        """P75 must match reference."""
        self._compare_with_reference("value_int__p75_percentile", partition_by=["region"], use_approx=True)

    # -- Cross-framework null comparisons --------------------------------------

    def test_cross_framework_all_null_p50(self) -> None:
        """All-null column p50 must match reference."""
        self._compare_with_reference("score__p50_percentile", partition_by=["region"])

    def test_cross_framework_multi_null_p50(self) -> None:
        """Multi-null column (value_float) p50 must match reference."""
        self._compare_with_reference("value_float__p50_percentile", partition_by=["region"], use_approx=True)

    def test_cross_framework_amount_p50(self) -> None:
        """Amount column (2 nulls) p50 must match reference."""
        self._compare_with_reference("amount__p50_percentile", partition_by=["region"], use_approx=True)

    # -- Edge case tests -------------------------------------------------------

    def test_empty_data_returns_zero_rows(self) -> None:
        """Percentile on an empty table (0 rows) should return 0 rows with the new column."""
        import pyarrow as pa

        empty_arrow = pa.table(
            {
                "region": pa.array([], type=pa.string()),
                "value_int": pa.array([], type=pa.int64()),
            }
        )
        empty_data = self.create_test_data(empty_arrow)
        fs = make_feature_set("value_int__p50_percentile", ["region"])
        result = self.implementation_class().calculate_feature(empty_data, fs)
        assert self.get_row_count(result) == 0

    def test_single_row_partitions_return_own_value(self) -> None:
        """Each partition has 1 row: percentile at any level equals the single value."""
        import pyarrow as pa

        # 3 rows, each in its own partition
        arrow = pa.table(
            {
                "region": pa.array(["A", "B", "C"], type=pa.string()),
                "value_int": pa.array([10, 20, 30], type=pa.int64()),
            }
        )
        custom_data = self.create_test_data(arrow)

        for pctl in ("p0", "p50", "p100"):
            feature_name = f"value_int__{pctl}_percentile"
            fs = make_feature_set(feature_name, ["region"])
            result = self.implementation_class().calculate_feature(custom_data, fs)

            result_col = self.extract_column(result, feature_name)
            assert result_col == pytest.approx([10.0, 20.0, 30.0], rel=1e-6), (
                f"{pctl}: single-row partitions should return the value itself"
            )

    # -- Mask tests ------------------------------------------------------------

    def test_mask_p50_percentile_equal(self) -> None:
        """P50 of value_int where category='X', partitioned by region."""
        fs = make_feature_set("value_int__p50_percentile", ["region"], mask=("category", "equal", "X"))
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__p50_percentile")
        assert result_col == pytest.approx(
            [5.0, 5.0, 5.0, 5.0, 60.0, 60.0, 60.0, 60.0, 15.0, 15.0, 15.0, -10.0], rel=1e-3
        )

    def test_mask_multiple_conditions_percentile(self) -> None:
        """P50 with AND-combined mask: category='X' AND value_int >= 10."""
        fs = make_feature_set(
            "value_int__p50_percentile",
            ["region"],
            mask=[("category", "equal", "X"), ("value_int", "greater_equal", 10)],
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__p50_percentile")
        # A: [10] -> 10.0, B: [60] -> 60.0, C: [15] -> 15.0, None: [] -> None
        assert result_col[:4] == pytest.approx([10.0, 10.0, 10.0, 10.0], rel=1e-3)
        assert result_col[4:8] == pytest.approx([60.0, 60.0, 60.0, 60.0], rel=1e-3)
        assert result_col[8:11] == pytest.approx([15.0, 15.0, 15.0], rel=1e-3)
        assert result_col[11] is None

    def test_unicode_column_names(self) -> None:
        """Unicode characters in source and partition_by column names must work."""
        import pyarrow as pa

        arrow = pa.table(
            {
                "r\u00e9gion": pa.array(["A", "A", "B", "B"], type=pa.string()),
                "w\u00e9rt": pa.array([10, 20, 30, 40], type=pa.int64()),
            }
        )
        custom_data = self.create_test_data(arrow)

        fs = make_feature_set("w\u00e9rt__p50_percentile", ["r\u00e9gion"])
        result = self.implementation_class().calculate_feature(custom_data, fs)

        result_col = self.extract_column(result, "w\u00e9rt__p50_percentile")
        # A: [10, 20] -> p50 = 15.0, B: [30, 40] -> p50 = 35.0
        assert result_col == pytest.approx([15.0, 15.0, 35.0, 35.0], rel=1e-6)
