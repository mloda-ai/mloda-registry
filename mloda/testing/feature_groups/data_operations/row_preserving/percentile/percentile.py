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
EXPECTED_P50_MULTI_KEY: list[float] = [5.0, 7.5, 5.0, 7.5, 60.0, 50.0, 30.0, 60.0, 27.5, 15.0, 27.5, -10.0]


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class PercentileTestBase(DataOpsTestBase):
    """Abstract base class for percentile framework tests."""

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        from mloda.community.feature_groups.data_operations.row_preserving.percentile.pyarrow_percentile import (
            PyArrowPercentile,
        )

        return PyArrowPercentile

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
                    "aggregation_type": "percentile",
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

    # -- Cross-framework comparison (matches PyArrow reference) --------------

    def test_cross_framework_p50(self) -> None:
        """P50 must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__p50_percentile", partition_by=["region"], use_approx=True)

    def test_cross_framework_p25(self) -> None:
        """P25 must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__p25_percentile", partition_by=["region"], use_approx=True)

    def test_cross_framework_p75(self) -> None:
        """P75 must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__p75_percentile", partition_by=["region"], use_approx=True)
