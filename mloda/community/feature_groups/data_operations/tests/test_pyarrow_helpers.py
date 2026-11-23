"""Tests for the shared PyArrow temporal timezone helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest

from mloda.community.feature_groups.data_operations.pyarrow_helpers import (
    assert_temporal_timezone_supported,
    fixed_offset_unsupported,
)

NUMERIC_ZONES = ["+00:00", "+05:00", "-03:30"]


@pytest.mark.parametrize(
    ("arrow_type", "numeric_fixed_offset"),
    [
        *[pytest.param(pa.timestamp("us", tz=zone), False, id=str(zone)) for zone in (None, "UTC", "Etc/GMT-5")],
        *[pytest.param(pa.timestamp("us", tz=zone), True, id=zone) for zone in NUMERIC_ZONES],
        pytest.param(pa.date32(), False, id="date32"),
        pytest.param(pa.string(), False, id="string"),
        pytest.param(pa.int64(), False, id="int64"),
    ],
)
@pytest.mark.parametrize(("major", "gate_active"), [(21, True), (22, False)])
def test_fixed_offset_unsupported_by_zone(
    monkeypatch: pytest.MonkeyPatch, arrow_type: Any, numeric_fixed_offset: bool, major: int, gate_active: bool
) -> None:
    """Only numeric fixed-offset timestamps are unsupported, and only before PyArrow 22."""
    monkeypatch.setattr(pa, "cpp_version_info", SimpleNamespace(major=major))
    unsupported = numeric_fixed_offset and gate_active
    assert fixed_offset_unsupported(arrow_type) is unsupported
    if unsupported:
        with pytest.raises(ValueError, match="22.0.0") as error:
            assert_temporal_timezone_supported(arrow_type, "event_ts")
        assert arrow_type.tz in str(error.value)
    else:
        assert_temporal_timezone_supported(arrow_type, "event_ts")
