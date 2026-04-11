"""Reusable mask (conditional aggregation) test mixin for data-operations feature groups.

Provides 6 standardized test methods that verify masking works correctly
across all feature groups that support FilterMask. Each test base class
mixes this in and overrides the abstract configuration methods to adapt
the generic tests to its specific semantics (feature names, partition keys,
expected values, reducing vs row-preserving).

Test methods use a ``test_mixin_mask_`` prefix to avoid name collisions
with existing inline mask tests on each feature group's test base.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from mloda.testing.feature_groups.data_operations.helpers import make_feature_set


def _is_null(value: Any) -> bool:
    """Check if a value is null (None or NaN)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


class MaskTestMixin:
    """Mixin providing standardized mask (conditional aggregation) tests.

    Feature-group test bases mix this in and implement the configuration
    methods below to adapt the generic tests to their semantics.

    Requires the host class to provide (from DataOpsTestBase):
    - ``implementation_class()``
    - ``test_data`` attribute (set in setup_method)
    - ``extract_column(result, column_name)``
    - ``get_row_count(result)``
    """

    # -- Configuration methods (override per feature group) --------------------

    @classmethod
    def mask_feature_name(cls) -> str:
        """Feature name for mask tests (e.g. 'value_int__sum_window')."""
        raise NotImplementedError

    @classmethod
    def mask_partition_by(cls) -> list[str] | None:
        """Partition key(s) for mask tests. None for scalar aggregate."""
        raise NotImplementedError

    @classmethod
    def mask_order_by(cls) -> str | None:
        """Order key for mask tests. Only frame aggregate needs this."""
        return None

    @classmethod
    def mask_expected_row_count(cls) -> int:
        """Expected rows: 12 for row-preserving, 4 for aggregation."""
        return 12

    @classmethod
    def mask_is_reducing(cls) -> bool:
        """True for aggregation (reduces rows), False for row-preserving."""
        return False

    @classmethod
    def mask_use_approx(cls) -> bool:
        """True if results are floating-point and need approximate comparison."""
        return False

    @classmethod
    def mask_equal_expected(cls) -> list[Any] | dict[Any, Any]:
        """Expected result for basic equal mask (category='X')."""
        raise NotImplementedError

    @classmethod
    def mask_multiple_conditions_expected(cls) -> list[Any] | dict[Any, Any]:
        """Expected result for AND mask (category='X' AND value_int>=10)."""
        raise NotImplementedError

    @classmethod
    def mask_is_in_expected(cls) -> list[Any] | dict[Any, Any]:
        """Expected result for is_in mask (region is_in ['A', 'C'])."""
        raise NotImplementedError

    @classmethod
    def mask_greater_than_expected(cls) -> list[Any] | dict[Any, Any]:
        """Expected result for greater_than mask (value_int > 10)."""
        raise NotImplementedError

    @classmethod
    def mask_no_mask_expected(cls) -> list[Any] | dict[Any, Any]:
        """Expected result without any mask (baseline)."""
        raise NotImplementedError

    # -- Assertion helper ------------------------------------------------------

    def _assert_mask_values(self, result: Any, expected: list[Any] | dict[Any, Any]) -> None:
        """Compare mask test results against expected values.

        Dispatches between row-preserving (list) and reducing (dict) modes
        based on ``mask_is_reducing()``.
        """
        feature_name = self.mask_feature_name()

        if self.mask_is_reducing():
            region_col = self.extract_column(result, "region")  # type: ignore[attr-defined]
            result_col = self.extract_column(result, feature_name)  # type: ignore[attr-defined]
            result_map = {region_col[i]: result_col[i] for i in range(len(region_col))}
            for key, exp in expected.items():  # type: ignore[union-attr]
                actual = result_map[key]
                if _is_null(exp):
                    assert _is_null(actual), f"region={key}: expected null, got {actual}"
                elif self.mask_use_approx() and isinstance(exp, float):
                    assert actual == pytest.approx(exp, rel=1e-3), f"region={key}: {actual} != {exp}"
                else:
                    assert actual == exp, f"region={key}: {actual} != {exp}"
        else:
            result_col = self.extract_column(result, feature_name)  # type: ignore[attr-defined]
            assert len(result_col) == len(expected), f"length {len(result_col)} != {len(expected)}"
            for i, (actual, exp) in enumerate(zip(result_col, expected)):
                if _is_null(exp):
                    assert _is_null(actual), f"row {i}: expected null, got {actual}"
                elif self.mask_use_approx() and isinstance(exp, float):
                    assert actual == pytest.approx(exp, rel=1e-3), f"row {i}: {actual} != {exp}"
                else:
                    assert actual == exp, f"row {i}: {actual} != {exp}"

    # -- Concrete test methods -------------------------------------------------

    def test_mixin_mask_equal(self) -> None:
        """Mixin: basic equal mask (category='X')."""
        fs = make_feature_set(
            self.mask_feature_name(),
            self.mask_partition_by(),
            self.mask_order_by(),
            mask=("category", "equal", "X"),
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)  # type: ignore[attr-defined]
        assert self.get_row_count(result) == self.mask_expected_row_count()  # type: ignore[attr-defined]
        self._assert_mask_values(result, self.mask_equal_expected())

    def test_mixin_mask_multiple_conditions(self) -> None:
        """Mixin: AND-combined mask (category='X' AND value_int >= 10)."""
        fs = make_feature_set(
            self.mask_feature_name(),
            self.mask_partition_by(),
            self.mask_order_by(),
            mask=[("category", "equal", "X"), ("value_int", "greater_equal", 10)],
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)  # type: ignore[attr-defined]
        assert self.get_row_count(result) == self.mask_expected_row_count()  # type: ignore[attr-defined]
        self._assert_mask_values(result, self.mask_multiple_conditions_expected())

    def test_mixin_mask_is_in(self) -> None:
        """Mixin: is_in mask (region is_in ['A', 'C'])."""
        fs = make_feature_set(
            self.mask_feature_name(),
            self.mask_partition_by(),
            self.mask_order_by(),
            mask=("region", "is_in", ["A", "C"]),
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)  # type: ignore[attr-defined]
        assert self.get_row_count(result) == self.mask_expected_row_count()  # type: ignore[attr-defined]
        self._assert_mask_values(result, self.mask_is_in_expected())

    def test_mixin_mask_greater_than(self) -> None:
        """Mixin: greater_than mask (value_int > 10)."""
        fs = make_feature_set(
            self.mask_feature_name(),
            self.mask_partition_by(),
            self.mask_order_by(),
            mask=("value_int", "greater_than", 10),
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)  # type: ignore[attr-defined]
        assert self.get_row_count(result) == self.mask_expected_row_count()  # type: ignore[attr-defined]
        self._assert_mask_values(result, self.mask_greater_than_expected())

    def test_mixin_mask_fully_masked(self) -> None:
        """Mixin: all rows masked out (category='Z') should produce None for every value."""
        fs = make_feature_set(
            self.mask_feature_name(),
            self.mask_partition_by(),
            self.mask_order_by(),
            mask=("category", "equal", "Z"),
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)  # type: ignore[attr-defined]
        assert self.get_row_count(result) == self.mask_expected_row_count()  # type: ignore[attr-defined]
        result_col = self.extract_column(result, self.mask_feature_name())  # type: ignore[attr-defined]
        assert all(_is_null(v) for v in result_col)

    def test_mixin_mask_no_mask_baseline(self) -> None:
        """Mixin: without mask, results match the standard unmasked value."""
        fs = make_feature_set(
            self.mask_feature_name(),
            self.mask_partition_by(),
            self.mask_order_by(),
        )
        result = self.implementation_class().calculate_feature(self.test_data, fs)  # type: ignore[attr-defined]
        assert self.get_row_count(result) == self.mask_expected_row_count()  # type: ignore[attr-defined]
        self._assert_mask_values(result, self.mask_no_mask_expected())
