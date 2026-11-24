"""Tests for DuckdbOffset compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.offset.duckdb_offset import DuckdbOffset
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase


class TestDuckdbOffset(DuckdbTestMixin, OffsetTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbOffset

    def test_mixin_pinned_row(self, pinned_row_case: str) -> None:
        # Reproduced native failure (not a rejection): narrow integer subtraction overflows before widening.
        # The relation is lazy, so the error surfaces on extraction, which the mixin's rejection branch does not reach.
        row = self.pinned_row_cases()[pinned_row_case]
        if row.feature != "v__diff_1_offset" or row.case != "dtype":
            super().test_mixin_pinned_row(pinned_row_case)
            return
        result = calculate_with(
            self.implementation_class(), self._create_pinned_data(row), make_feature_set(row.feature, **row.context)
        )
        with pytest.raises(duckdb.OutOfRangeException, match="Overflow in subtraction of (UINT8|INT8)"):
            self.extract_column(result, "v__diff_1_offset")
