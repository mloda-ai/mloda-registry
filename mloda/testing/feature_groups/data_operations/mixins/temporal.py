"""Shared timezone expectations for datetime, time-bucketization and resample kits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from mloda.provider import DivergenceKind, Implementation

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import has_numeric_fixed_offset


class TemporalSourceTestMixin(DataOpsTestBase):
    """Declare UTC-reading gaps and numeric fixed-offset source rejection once per backend."""

    @classmethod
    def tz_aware_source_reads_utc(cls) -> bool:
        """True when a tz-aware source uses UTC rather than its own zone (a declared GAP)."""
        return False

    @classmethod
    def rejects_numeric_offset_source(cls) -> bool:
        """True when a source with a numeric fixed-offset timezone is rejected with a ValueError."""
        return False

    @classmethod
    def numeric_offset_rejection_match(cls, values: Sequence[Any]) -> str | None:
        """Regex of the expected fixed-offset rejection, or None for sources the backend accepts."""
        if cls.rejects_numeric_offset_source() and has_numeric_fixed_offset(values):
            return r"(?i)fixed.offset"
        return None

    def test_divergent_flags_are_declared(self) -> None:
        impl = self.implementation_class()
        if not (isinstance(impl, type) and issubclass(impl, Implementation)):
            pytest.skip("implementation under test is not an Implementation class")
        if self.tz_aware_source_reads_utc():
            assert any(record.kind is DivergenceKind.GAP for record in impl.divergences)


class PyArrowTemporalSourceTestMixin(TemporalSourceTestMixin):
    """Temporal-source expectations for PyArrow hosts; mix in before the operation base."""

    @classmethod
    def rejects_numeric_offset_source(cls) -> bool:
        """PyArrow before 22 cannot handle numeric fixed-offset zones."""
        import pyarrow as pa

        return bool(pa.cpp_version_info.major < 22)
