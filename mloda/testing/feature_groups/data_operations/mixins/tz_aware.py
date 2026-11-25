"""Flags and pinned-row hooks for temporal groups that read a tz-aware source column."""

from __future__ import annotations

from datetime import timezone
from typing import Any

import pytest
from mloda.provider import DivergenceKind, Implementation
from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import (
    arrow_table_of,
    extract_column,
    has_numeric_fixed_offset,
    make_feature_set,
)
from mloda.testing.feature_groups.data_operations.mixins.pinned_rows import PinnedRowTestMixin
from mloda.testing.feature_groups.data_operations.pinning import PinningRow
from mloda.testing.typed_data import TypedTestData


class TzAwareSourceTestMixin(DataOpsTestBase):
    """Flags for a backend's handling of a tz-aware source column, and the check that a UTC reader declares a GAP."""

    @classmethod
    def tz_aware_source_reads_utc(cls) -> bool:
        """True when a tz-aware source column is read on its UTC instant, not its own zone (a declared GAP)."""
        return False

    @classmethod
    def rejects_numeric_offset_source(cls) -> bool:
        """True when a source with a numeric fixed-offset timezone is rejected with a ValueError."""
        return False

    def test_divergent_flags_are_declared(self) -> None:
        impl = self.implementation_class()
        if not (isinstance(impl, type) and issubclass(impl, Implementation)):
            pytest.skip("implementation under test is not an Implementation class")
        if self.tz_aware_source_reads_utc():
            assert any(record.kind is DivergenceKind.GAP for record in impl.divergences)


class TzAwarePinningTestMixin(TzAwareSourceTestMixin, PinnedRowTestMixin):
    """Pinned rows for a table whose rows have a single tz-aware input column."""

    def expected_values(self, row: PinningRow) -> dict[str, list[Any]]:
        """Hand-computed rows; a UTC-reading backend is pinned to the reference on UTC instants."""
        expected = super().expected_values(row)
        column = next(iter(row.inputs))
        data_type, values = row.inputs[column]
        if data_type is DataType.TIMESTAMP_MICROS_TZ and self.tz_aware_source_reads_utc():
            utc_values = [None if v is None else v.astimezone(timezone.utc) for v in values]
            ref_inputs = {column: (data_type, utc_values)}
            fs = make_feature_set(row.feature, **row.context)
            ref = self.reference_calculate(arrow_table_of(TypedTestData(ref_inputs)), fs)
            expected[row.feature] = extract_column(ref, row.feature)
        return expected

    def rejection_match(self, row: PinningRow) -> str | None:
        """A numeric fixed-offset row on a rejecting backend must raise up front instead of computing."""
        values = next(iter(row.inputs.values()))[1]
        if self.rejects_numeric_offset_source() and has_numeric_fixed_offset(values):
            return r"(?i)fixed.offset"
        return super().rejection_match(row)
