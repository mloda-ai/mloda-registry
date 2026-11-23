"""Run a group's hand-computed pinning rows through its backend under test."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.case_parametrization import CaseParametrizationTestMixin
from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable
from mloda.testing.typed_data import TypedTestData


class PinnedRowTestMixin(DataOpsTestBase, CaseParametrizationTestMixin):
    """Share collection and execution; hosts supply the table and any declared divergent expectations."""

    _case_fixtures: ClassVar[dict[str, str]] = {"pinned_row_case": "pinned_row_cases"}

    @classmethod
    def pinning_table(cls) -> PinningTable:
        """Hand-computed rows of the group under test. Required."""
        raise NotImplementedError("Subclass must override pinning_table")

    @classmethod
    def pinned_row_cases(cls) -> dict[str, PinningRow]:
        return {f"{i}-{row.case}-{row.feature}": row for i, row in enumerate(cls.pinning_table().rows)}

    def expected_values(self, row: PinningRow) -> dict[str, list[Any]]:
        """Expected columns; override for declared divergences or to skip unsupported operations."""
        return {column: list(values) for column, (_, values) in row.expected.items()}

    def rejection_match(self, row: PinningRow) -> str | None:
        """Regex of the up-front ValueError a declared rejection raises for this row; None computes it."""
        return None

    def assert_pinned_values(self, row: PinningRow, column: str, actual: list[Any], expected: list[Any]) -> None:
        """Compare one column; hosts may retain a stricter numeric comparison or backend cast."""
        assert actual == expected

    def test_mixin_pinned_row(self, pinned_row_case: str) -> None:
        """A row matches the backend result; rejection_match takes precedence over expected_values."""
        row = self.pinned_row_cases()[pinned_row_case]

        def run() -> Any:
            return calculate_with(
                self.implementation_class(),
                self.create_test_data(TypedTestData(row.inputs)),
                make_feature_set(row.feature, **row.context),
            )

        pattern = self.rejection_match(row)
        if pattern is not None:
            with pytest.raises(ValueError, match=pattern):
                run()
            return
        expected = self.expected_values(row)
        result = run()
        for column, values in expected.items():
            self.assert_pinned_values(row, column, self.extract_column(result, column), values)
