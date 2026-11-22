"""Reusable mask integration test mixin for data-operations feature groups.

Mix into an integration test class alongside DataOpsIntegrationTestBase and override the configuration methods.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.testing.data_creator.base import column_selective
from mloda.testing.feature_groups.data_operations.helpers import extract_column, is_null
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase


class MaskIntegrationTestMixin(DataOpsIntegrationTestBase):
    """Mixin providing mask integration tests through mloda's full pipeline."""

    # -- Configuration methods (override per feature group) --------------------

    @classmethod
    def mask_integration_feature_name(cls) -> str:
        """Feature name for mask integration tests."""
        raise NotImplementedError("Subclass must override mask_integration_feature_name")

    @classmethod
    def mask_integration_options(cls) -> dict[str, Any]:
        """Options context dict for mask integration tests (without mask key)."""
        raise NotImplementedError("Subclass must override mask_integration_options")

    @classmethod
    def mask_integration_spec(cls) -> tuple[str, str, Any]:
        """Mask spec for the basic mask integration test."""
        return ("category", "equal", "X")

    @classmethod
    def mask_integration_expected(cls) -> list[Any]:
        """Expected column values for the basic masked feature."""
        raise NotImplementedError("Subclass must override mask_integration_expected")

    @classmethod
    def mask_integration_complex_spec(cls) -> list[tuple[str, str, Any]]:
        """Multi-condition mask spec for the complex mask test."""
        return [("category", "equal", "X"), ("value_int", "greater_equal", 10)]

    @classmethod
    def mask_integration_complex_expected(cls) -> list[Any]:
        """Expected column values for the complex masked feature."""
        raise NotImplementedError("Subclass must override mask_integration_complex_expected")

    @classmethod
    def mask_integration_unmasked_expected(cls) -> list[Any]:
        """Expected column values for the unmasked feature (for the together test)."""
        raise NotImplementedError("Subclass must override mask_integration_unmasked_expected")

    @classmethod
    def mask_integration_expected_row_count(cls) -> int:
        """Expected number of rows. Default: 12 (row-preserving)."""
        return 12

    @classmethod
    def mask_integration_use_approx(cls) -> bool:
        """Whether to use approximate comparison for floating-point results."""
        return False

    @classmethod
    def mask_integration_compare_sorted(cls) -> bool:
        """Whether to sort values before comparing (for reducing operations)."""
        return False

    # -- Assertion helpers -----------------------------------------------------

    def _assert_mask_integration_values(self, actual: list[Any], expected: list[Any]) -> None:
        """Compare actual vs expected, handling nulls, approx, and sorting."""
        if self.mask_integration_compare_sorted():
            actual = sorted(actual, key=lambda x: (x is None, x if x is not None else 0))
            expected = sorted(expected, key=lambda x: (x is None, x if x is not None else 0))

        assert len(actual) == len(expected), f"length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            if is_null(e):
                assert is_null(a), f"row {i}: expected null, got {a}"
            elif self.mask_integration_use_approx() and isinstance(e, float):
                assert a == pytest.approx(e, rel=1e-3), f"row {i}: {a} != {e}"
            else:
                assert a == e, f"row {i}: {a} != {e}"

    def _run_masked_feature(
        self, name: str, options_ctx: dict[str, Any], mask_spec: Any, creator: type | None = None
    ) -> Any:
        """Run a single masked feature through the pipeline and return the framework result holding it."""
        return self._run_single_feature(name, {**options_ctx, "mask": mask_spec}, creator)

    # -- Concrete test methods -------------------------------------------------

    def _assert_masked(self, spec: Any, expected: list[Any], creator: type | None = None) -> None:
        """Run the masked feature, then check the row count and values."""
        name = self.mask_integration_feature_name()
        result_col = extract_column(
            self._run_masked_feature(name, self.mask_integration_options(), spec, creator), name
        )
        assert len(result_col) == self.mask_integration_expected_row_count()
        self._assert_mask_integration_values(result_col, expected)

    def test_mask_feature_through_pipeline(self) -> None:
        """Run a masked feature through run_all and verify values."""
        self._assert_masked(self.mask_integration_spec(), self.mask_integration_expected())

    def test_mask_feature_with_column_selective_source(self) -> None:
        """Mask columns must be requested from a source that serves only requested columns."""
        self._assert_masked(
            self.mask_integration_complex_spec(),
            self.mask_integration_complex_expected(),
            column_selective(self.data_creator_class()),
        )

    def test_mask_complex_conditions_through_pipeline(self) -> None:
        """Run a multi-condition masked feature through run_all and verify values."""
        self._assert_masked(self.mask_integration_complex_spec(), self.mask_integration_complex_expected())

    def test_masked_and_unmasked_consistent(self) -> None:
        """Run unmasked and masked in separate calls: same-type features share a group instance (mask leakage)."""
        name = self.mask_integration_feature_name()
        unmasked_col = extract_column(self._run_single_feature(name, self.mask_integration_options()), name)
        assert len(unmasked_col) == self.mask_integration_expected_row_count()
        self._assert_mask_integration_values(unmasked_col, self.mask_integration_unmasked_expected())
        self._assert_masked(self.mask_integration_spec(), self.mask_integration_expected())
