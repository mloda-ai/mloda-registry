"""Reusable mask (conditional aggregation) test mixin for data-operations feature groups.

Provides standardized test methods that verify masking works correctly
across all feature groups that support FilterMask. Each test base class
mixes this in and overrides the abstract configuration methods to adapt
the generic tests to its specific semantics (feature names, partition keys,
expected values, reducing vs row-preserving).

Test methods use a ``test_mixin_mask_`` prefix to avoid name collisions
with existing inline mask tests on each feature group's test base.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.testing.feature_groups.data_operations.helpers import is_null, make_feature_set


class MaskTestMixin:
    """Mixin providing standardized mask (conditional aggregation) tests.

    Feature-group test bases mix this in and implement the configuration
    methods below to adapt the generic tests to their semantics.

    Requires the host class to provide (from DataOpsTestBase):
    - ``implementation_class()``
    - ``create_test_data(arrow_table)``
    - ``test_data`` attribute (set in setup_method)
    - ``_arrow_table`` attribute (set in setup_method)
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
                if is_null(exp):
                    assert is_null(actual), f"region={key}: expected null, got {actual}"
                elif self.mask_use_approx() and isinstance(exp, float):
                    assert actual == pytest.approx(exp, rel=1e-3), f"region={key}: {actual} != {exp}"
                else:
                    assert actual == exp, f"region={key}: {actual} != {exp}"
        else:
            result_col = self.extract_column(result, feature_name)  # type: ignore[attr-defined]
            assert len(result_col) == len(expected), f"length {len(result_col)} != {len(expected)}"
            for i, (actual, exp) in enumerate(zip(result_col, expected)):
                if is_null(exp):
                    assert is_null(actual), f"row {i}: expected null, got {actual}"
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
        assert all(is_null(v) for v in result_col)

    def test_mixin_mask_missing_column_rejected(self) -> None:
        """Mixin: a mask on an unknown column raises a ValueError naming it."""
        fs = make_feature_set(
            self.mask_feature_name(),
            self.mask_partition_by(),
            self.mask_order_by(),
            mask=("no_such_col", "equal", "X"),
        )
        with pytest.raises(ValueError, match="mask column 'no_such_col' is not present"):
            self.implementation_class().calculate_feature(self.test_data, fs)  # type: ignore[attr-defined]

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

    @pytest.mark.parametrize(
        "metric_values, control_values, mask_spec, control_mask_spec",
        [
            pytest.param(
                # Mix of passing (5.0) and failing (0.0) rows in every region so a bug
                # that over-masks an entire group whenever it contains a NaN/null is caught.
                [5.0, 0.0, 0.0, float("nan"), 5.0, 0.0, 0.0, 0.0, 5.0, None, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                ("mask_metric", "greater_equal", 1.0),
                ("mask_metric", "greater_equal", 1.0),
                id="greater_equal",
            ),
            pytest.param(
                # Same idea as greater_equal, passing rows at different positions per region.
                [0.0, 5.0, 0.0, float("nan"), 0.0, 0.0, 5.0, 0.0, 0.0, None, 5.0, 0.0],
                [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0],
                ("mask_metric", "greater_than", 1.0),
                ("mask_metric", "greater_than", 1.0),
                id="greater_than",
            ),
            pytest.param(
                [5.0, 7.0, 7.0, float("nan"), 7.0, 7.0, 7.0, 7.0, 7.0, None, 7.0, 7.0],
                [5.0, 7.0, 7.0, -999.0, 7.0, 7.0, 7.0, 7.0, 7.0, -999.0, 7.0, 7.0],
                ("mask_metric", "equal"),
                ("mask_metric", "equal", -999.0),
                id="equal_two_element",
            ),
            pytest.param(
                [5.0, 7.0, 7.0, float("nan"), 7.0, 7.0, 7.0, 7.0, 7.0, None, 7.0, 7.0],
                [5.0, 7.0, 7.0, -999.0, 7.0, 7.0, 7.0, 7.0, 7.0, -999.0, 7.0, 7.0],
                ("mask_metric", "is_in", [None, 5.0]),
                ("mask_metric", "is_in", [-999.0, 5.0]),
                id="is_in_none",
            ),
            pytest.param(
                # AND-combined mask: a null (row 2, value_int=0) and a NaN (row 7, value_int=60)
                # each sit in a category='X' group that also has a passing row (row 0 / row 4).
                # value_int is nonzero at row 7, so wrongly letting the NaN row through (or
                # over-masking the whole group) changes the sum there too.
                [5.0, 0.0, None, 0.0, 5.0, 0.0, 0.0, float("nan"), 0.0, 5.0, 0.0, 5.0],
                [5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 5.0],
                [("mask_metric", "greater_equal", 1.0), ("category", "equal", "X")],
                [("mask_metric", "greater_equal", 1.0), ("category", "equal", "X")],
                id="and_combined_missing_values",
            ),
        ],
    )
    def test_mixin_mask_missing_values_follow_core_rule(
        self,
        metric_values: list[Any],
        control_values: list[Any],
        mask_spec: tuple[Any, ...] | list[tuple[Any, ...]],
        control_mask_spec: tuple[Any, ...] | list[tuple[Any, ...]],
    ) -> None:
        """Mixin: a null/NaN mask column follows core's missing-value rule for every feature group."""
        # Expectations come from a control run (missing values replaced by finite stand-ins) rather
        # than static per-group config, so the same test runs unchanged across every feature group.
        actual_table = self._arrow_table.append_column(  # type: ignore[attr-defined]
            "mask_metric", pa.array(metric_values, type=pa.float64())
        )
        control_table = self._arrow_table.append_column(  # type: ignore[attr-defined]
            "mask_metric", pa.array(control_values, type=pa.float64())
        )

        actual_data = self.create_test_data(actual_table)  # type: ignore[attr-defined]
        control_data = self.create_test_data(control_table)  # type: ignore[attr-defined]

        fs_actual = make_feature_set(
            self.mask_feature_name(), self.mask_partition_by(), self.mask_order_by(), mask=mask_spec
        )
        fs_control = make_feature_set(
            self.mask_feature_name(), self.mask_partition_by(), self.mask_order_by(), mask=control_mask_spec
        )

        actual_result = self.implementation_class().calculate_feature(actual_data, fs_actual)  # type: ignore[attr-defined]
        control_result = self.implementation_class().calculate_feature(control_data, fs_control)  # type: ignore[attr-defined]

        feature_name = self.mask_feature_name()
        expected: list[Any] | dict[Any, Any]
        if self.mask_is_reducing():
            region_col = self.extract_column(control_result, "region")  # type: ignore[attr-defined]
            value_col = self.extract_column(control_result, feature_name)  # type: ignore[attr-defined]
            expected = {region_col[i]: value_col[i] for i in range(len(region_col))}
        else:
            expected = self.extract_column(control_result, feature_name)  # type: ignore[attr-defined]

        assert self.get_row_count(actual_result) == self.mask_expected_row_count()  # type: ignore[attr-defined]
        self._assert_mask_values(actual_result, expected)
