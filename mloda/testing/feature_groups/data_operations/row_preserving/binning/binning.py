"""Shared test base class, data, and helpers for binning tests.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, using the 'value_int' column.

value_int = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]

The ``BinningTestBase`` class provides concrete test methods that any
framework implementation inherits by subclassing and implementing
5 abstract methods. This follows the same pattern as WindowAggregationTestBase.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set


# ---------------------------------------------------------------------------
# Expected values (module-level constants)
# ---------------------------------------------------------------------------

# value_int non-null: [10, -5, 0, 20, 50, 30, 60, 15, 15, 40, -10]
# min=-10, max=60, range=70, bin_width = 70 / 3 = 23.333...
# val=10:  (10 - (-10)) / 23.333 = 20/23.333 = 0.857 -> int(0.857) = 0
# val=-5:  (-5 - (-10)) / 23.333 = 5/23.333 = 0.214 -> int(0.214) = 0
# val=0:   (0 - (-10)) / 23.333 = 10/23.333 = 0.428 -> int(0.428) = 0
# val=20:  (20 - (-10)) / 23.333 = 30/23.333 = 1.285 -> int(1.285) = 1
# val=None: None
# val=50:  (50 - (-10)) / 23.333 = 60/23.333 = 2.571 -> int(2.571) = 2
# val=30:  (30 - (-10)) / 23.333 = 40/23.333 = 1.714 -> int(1.714) = 1
# val=60:  (60 - (-10)) / 23.333 = 70/23.333 = 3.0 -> min(3, 2) = 2
# val=15:  (15 - (-10)) / 23.333 = 25/23.333 = 1.071 -> int(1.071) = 1
# val=15:  same -> 1
# val=40:  (40 - (-10)) / 23.333 = 50/23.333 = 2.142 -> int(2.142) = 2
# val=-10: (-10 - (-10)) / 23.333 = 0/23.333 = 0.0 -> int(0.0) = 0

# Corrected:
EXPECTED_BIN_3 = [0, 0, 0, 1, None, 2, 1, 2, 1, 1, 2, 0]


# ---------------------------------------------------------------------------
# qbin expected values (rank-based NTILE semantics)
# ---------------------------------------------------------------------------
# Non-null sorted with original indices:
#   (-10, 11), (-5, 1), (0, 2), (10, 0), (15, 8), (15, 9),
#   (20, 3), (30, 6), (40, 10), (50, 5), (60, 7)
# 11 values, n_bins=3: bin = rank * 3 // 11
#   rank 0 -> 0, rank 1 -> 0, rank 2 -> 0, rank 3 -> 0,
#   rank 4 -> 1, rank 5 -> 1, rank 6 -> 1, rank 7 -> 1,
#   rank 8 -> 2, rank 9 -> 2, rank 10 -> 2
EXPECTED_QBIN_3: list[Any] = [0, 0, 0, 1, None, 2, 1, 2, 1, 1, 2, 0]

# 11 values, n_bins=5: bin = rank * 5 // 11
#   rank 0 -> 0, rank 1 -> 0, rank 2 -> 0, rank 3 -> 1,
#   rank 4 -> 1, rank 5 -> 2, rank 6 -> 2, rank 7 -> 3,
#   rank 8 -> 3, rank 9 -> 4, rank 10 -> 4
EXPECTED_QBIN_5: list[Any] = [1, 0, 0, 2, None, 4, 3, 4, 1, 2, 3, 0]


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class BinningTestBase(DataOpsTestBase):
    """Abstract base class for binning framework tests."""

    @classmethod
    def supported_ops(cls) -> set[str]:
        return {"bin", "qbin"}

    @classmethod
    def reference_implementation_class(cls) -> Any:
        from mloda.community.feature_groups.data_operations.row_preserving.binning.pyarrow_binning import (
            PyArrowBinning,
        )

        return PyArrowBinning

    # -- Concrete test methods -----------------------------------------------

    def test_bin_3_value_int(self) -> None:
        """Equal-width binning of value_int into 3 bins."""
        fs = make_feature_set("value_int__bin_3")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__bin_3")
        assert result_col == EXPECTED_BIN_3

    def test_null_propagation(self) -> None:
        """EC-013: Null values produce null bin assignments."""
        fs = make_feature_set("value_int__bin_3")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__bin_3")
        # Row 4 has value_int=None
        assert result_col[4] is None

    def test_output_rows_equal_input_rows(self) -> None:
        """Row-preserving contract: output rows == input rows."""
        fs = make_feature_set("value_int__bin_3")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

    def test_result_has_correct_type(self) -> None:
        """The result must be the expected framework type."""
        fs = make_feature_set("value_int__bin_3")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())

    def test_new_column_added(self) -> None:
        """The binning result column should be added to the output."""
        fs = make_feature_set("value_int__bin_3")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__bin_3")
        assert len(result_col) == 12

    def test_bin_values_in_range(self) -> None:
        """All non-null bin values must be in [0, n_bins-1]."""
        fs = make_feature_set("value_int__bin_5")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__bin_5")
        for val in result_col:
            if val is not None:
                assert 0 <= val < 5, f"Bin value {val} out of range [0, 4]"

    # -- qbin concrete tests -------------------------------------------------

    def test_qbin_3_value_int(self) -> None:
        """Quantile binning of value_int into 3 bins."""
        self._skip_if_unsupported("qbin")
        fs = make_feature_set("value_int__qbin_3")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__qbin_3")
        assert result_col == EXPECTED_QBIN_3

    def test_qbin_5_value_int(self) -> None:
        """Quantile binning of value_int into 5 bins."""
        self._skip_if_unsupported("qbin")
        fs = make_feature_set("value_int__qbin_5")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__qbin_5")
        assert result_col == EXPECTED_QBIN_5

    def test_qbin_null_propagation(self) -> None:
        """Null values produce null qbin assignments."""
        self._skip_if_unsupported("qbin")
        fs = make_feature_set("value_int__qbin_3")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__qbin_3")
        assert result_col[4] is None

    def test_qbin_values_in_range(self) -> None:
        """All non-null qbin values must be in [0, n_bins-1]."""
        self._skip_if_unsupported("qbin")
        fs = make_feature_set("value_int__qbin_5")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__qbin_5")
        for val in result_col:
            if val is not None:
                assert 0 <= val < 5, f"Qbin value {val} out of range [0, 4]"

    # -- Cross-framework comparison (matches reference) --------------

    def test_cross_framework_bin_3(self) -> None:
        """bin_3 must match reference."""
        self._compare_with_reference("value_int__bin_3")

    def test_cross_framework_bin_5(self) -> None:
        """bin_5 must match reference."""
        self._compare_with_reference("value_int__bin_5")

    def test_cross_framework_qbin_3(self) -> None:
        """qbin_3 must match reference."""
        self._skip_if_unsupported("qbin")
        self._compare_with_reference("value_int__qbin_3")

    def test_cross_framework_qbin_5(self) -> None:
        """qbin_5 must match reference."""
        self._skip_if_unsupported("qbin")
        self._compare_with_reference("value_int__qbin_5")

    # -- Edge case tests -----------------------------------------------------

    def test_ec011_all_identical_values(self) -> None:
        """EC-011: All identical values should produce one bin, no division-by-zero."""
        identical_data = {"value_int": [5, 5, 5, 5, 5]}
        arrow_table = pa.table(identical_data)
        test_data = self.create_test_data(arrow_table)

        fs = make_feature_set("value_int__bin_3")
        result = self.implementation_class().calculate_feature(test_data, fs)

        result_col = self.extract_column(result, "value_int__bin_3")
        assert all(v == 0 for v in result_col)

    def test_ec015_single_value(self) -> None:
        """EC-015: Single value in column should produce one bin."""
        single_data = {"value_int": [42]}
        arrow_table = pa.table(single_data)
        test_data = self.create_test_data(arrow_table)

        fs = make_feature_set("value_int__bin_3")
        result = self.implementation_class().calculate_feature(test_data, fs)

        result_col = self.extract_column(result, "value_int__bin_3")
        assert result_col == [0]

    def test_qbin_all_identical_values(self) -> None:
        """Quantile binning with all identical values should produce bin 0."""
        self._skip_if_unsupported("qbin")
        identical_data = {"value_int": [5, 5, 5, 5, 5]}
        arrow_table = pa.table(identical_data)
        test_data = self.create_test_data(arrow_table)

        fs = make_feature_set("value_int__qbin_3")
        result = self.implementation_class().calculate_feature(test_data, fs)

        result_col = self.extract_column(result, "value_int__qbin_3")
        for val in result_col:
            if val is not None:
                assert 0 <= val < 3, f"Qbin value {val} out of range [0, 2]"

    def test_n_bins_zero_rejected(self) -> None:
        """n_bins=0 must raise ValueError."""
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            self.implementation_class().get_binning_params("value_int__bin_0")

    def test_qbin_single_value(self) -> None:
        """Quantile binning with a single value."""
        self._skip_if_unsupported("qbin")
        single_data = {"value_int": [42]}
        arrow_table = pa.table(single_data)
        test_data = self.create_test_data(arrow_table)

        fs = make_feature_set("value_int__qbin_3")
        result = self.implementation_class().calculate_feature(test_data, fs)

        result_col = self.extract_column(result, "value_int__qbin_3")
        assert result_col == [0]

    def test_n_bins_1(self) -> None:
        """n_bins=1: all non-null values should map to bin 0."""
        fs = make_feature_set("value_int__bin_1")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__bin_1")
        for i, val in enumerate(result_col):
            if val is None:
                assert i == 4, f"unexpected None at row {i}"
            else:
                assert val == 0, f"row {i}: expected 0, got {val}"

    def test_qbin_n_bins_1(self) -> None:
        """qbin with n_bins=1: all non-null values should map to bin 0."""
        self._skip_if_unsupported("qbin")
        fs = make_feature_set("value_int__qbin_1")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__qbin_1")
        for i, val in enumerate(result_col):
            if val is None:
                assert i == 4, f"unexpected None at row {i}"
            else:
                assert val == 0, f"row {i}: expected 0, got {val}"

    def test_n_bins_greater_than_row_count(self) -> None:
        """n_bins > row count: should not error, bins are sparse."""
        small_data = {"value_int": [10, 20, 30, 40, 50]}
        arrow_table = pa.table(small_data)
        test_data = self.create_test_data(arrow_table)

        fs = make_feature_set("value_int__bin_100")
        result = self.implementation_class().calculate_feature(test_data, fs)

        result_col = self.extract_column(result, "value_int__bin_100")
        assert len(result_col) == 5
        for val in result_col:
            assert 0 <= val < 100, f"bin value {val} out of range [0, 99]"

    def test_all_null_column(self) -> None:
        """All-null column should produce all-null bin assignments."""
        arrow_table = pa.table({"value_int": pa.array([None, None, None], type=pa.int64())})
        test_data = self.create_test_data(arrow_table)

        fs = make_feature_set("value_int__bin_3")
        result = self.implementation_class().calculate_feature(test_data, fs)

        result_col = self.extract_column(result, "value_int__bin_3")
        assert all(v is None for v in result_col), f"expected all None, got {result_col}"

    def test_all_null_column_qbin(self) -> None:
        """All-null column should produce all-null qbin assignments."""
        self._skip_if_unsupported("qbin")
        arrow_table = pa.table({"value_int": pa.array([None, None, None], type=pa.int64())})
        test_data = self.create_test_data(arrow_table)

        fs = make_feature_set("value_int__qbin_3")
        result = self.implementation_class().calculate_feature(test_data, fs)

        result_col = self.extract_column(result, "value_int__qbin_3")
        assert all(v is None for v in result_col), f"expected all None, got {result_col}"

    def test_bin_boundary_precision(self) -> None:
        """Values exactly on bin boundaries should be consistent across frameworks.

        [0.0, 10.0, 20.0, 30.0] with n_bins=3: width=10.0
        All backends use left-closed [a, b) intervals.
        val=0:  (0-0)/10 = 0.0 -> bin 0
        val=10: (10-0)/10 = 1.0 -> bin 1
        val=20: (20-0)/10 = 2.0 -> bin 2
        val=30: (30-0)/10 = 3.0 -> clamped to bin 2
        """
        boundary_data = {"value_int": [0, 10, 20, 30]}
        arrow_table = pa.table(boundary_data)
        test_data = self.create_test_data(arrow_table)

        fs = make_feature_set("value_int__bin_3")
        result = self.implementation_class().calculate_feature(test_data, fs)

        result_col = self.extract_column(result, "value_int__bin_3")
        assert result_col == [0, 1, 2, 2]

    # -- NaN handling tests ----------------------------------------------------

    def test_nan_treated_as_null_bin(self) -> None:
        """NaN values in float columns must be treated as null for bin.

        Data: [1.0, NaN, 3.0, 5.0, 7.0]
        Non-null (NaN excluded): [1.0, 3.0, 5.0, 7.0]
        min=1.0, max=7.0, range=6.0, width=2.0
        val=1: (1-1)/2=0.0 -> 0, val=NaN: None, val=3: (3-1)/2=1.0 -> 1,
        val=5: (5-1)/2=2.0 -> 2, val=7: (7-1)/2=3.0 -> clamped to 2
        """
        arrow_table = pa.table({"value_int": pa.array([1.0, float("nan"), 3.0, 5.0, 7.0])})
        test_data = self.create_test_data(arrow_table)

        fs = make_feature_set("value_int__bin_3")
        result = self.implementation_class().calculate_feature(test_data, fs)

        result_col = self.extract_column(result, "value_int__bin_3")
        assert result_col == [0, None, 1, 2, 2]

    def test_nan_treated_as_null_qbin(self) -> None:
        """NaN values in float columns must be treated as null for qbin.

        Data: [1.0, NaN, 3.0, 5.0, 7.0]
        Non-null sorted: [1.0, 3.0, 5.0, 7.0], N=4, n_bins=3
        rank 0 -> 0*3//4=0, rank 1 -> 3//4=0,
        rank 2 -> 6//4=1, rank 3 -> 9//4=2
        """
        self._skip_if_unsupported("qbin")
        arrow_table = pa.table({"value_int": pa.array([1.0, float("nan"), 3.0, 5.0, 7.0])})
        test_data = self.create_test_data(arrow_table)

        fs = make_feature_set("value_int__qbin_3")
        result = self.implementation_class().calculate_feature(test_data, fs)

        result_col = self.extract_column(result, "value_int__qbin_3")
        assert result_col == [0, None, 0, 1, 2]

    # -- Option-based config tests -------------------------------------------

    def test_option_based_bin_3(self) -> None:
        """Option-based configuration (not string pattern) produces the same result."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "my_bin_result",
            options=Options(
                context={
                    "binning_op": "bin",
                    "n_bins": 3,
                    "in_features": "value_int",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_bin_result")
        assert result_col == EXPECTED_BIN_3

    def test_unsupported_binning_type_raises(self) -> None:
        """Calling calculate_feature with an unknown binning type should raise."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "value_int__evil_3",
            options=Options(
                context={
                    "binning_op": "evil",
                    "n_bins": 3,
                    "in_features": "value_int",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises((ValueError, KeyError)):
            self.implementation_class().calculate_feature(self.test_data, fs)
