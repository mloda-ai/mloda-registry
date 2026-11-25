"""Tests for DuckdbOffset compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.offset.duckdb_offset import DuckdbOffset
from mloda.testing.feature_groups.data_operations.mixins.frameworks import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.pinning import PinningRow
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase

# Reproduced native failure (DuckDB narrow-int subtraction overflows before widening), not an up-front rejection.
_NATIVE_OVERFLOW_ROWS = {"20-dtype-v__diff_1_offset", "21-dtype-v__diff_1_offset"}


class TestDuckdbOffset(DuckdbTestMixin, OffsetTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbOffset

    def rejection_match(self, row: PinningRow) -> str | None:
        if self._row_id(row) in _NATIVE_OVERFLOW_ROWS:
            return "Overflow in subtraction of (UINT8|INT8)"
        return super().rejection_match(row)

    def rejection_type(self, row: PinningRow) -> type[Exception]:
        if self._row_id(row) in _NATIVE_OVERFLOW_ROWS:
            return duckdb.OutOfRangeException  # type: ignore[no-any-return]
        return super().rejection_type(row)
