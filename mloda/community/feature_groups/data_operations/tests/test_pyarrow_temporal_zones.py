"""Pin the PyArrow temporal timezone contract on the minimum and current releases."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pyarrow as pa
import pytest

from mloda.community.feature_groups.data_operations.pyarrow_helpers import fixed_offset_unsupported
from mloda.community.feature_groups.data_operations.row_preserving.datetime.pyarrow_datetime import (
    PyArrowDateTimeExtraction,
)
from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.pyarrow_time_bucketization import (
    PyArrowTimeBucketization,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set


@pytest.mark.parametrize(
    ("implementation", "operation", "expected"),
    [
        pytest.param(PyArrowTimeBucketization, "floor_1_day", datetime(2024, 3, 31), id="floor"),
        pytest.param(PyArrowTimeBucketization, "round_1_month", datetime(2024, 4, 1), id="round-month"),
        pytest.param(PyArrowDateTimeExtraction, "year", 2024, id="year"),
        pytest.param(PyArrowDateTimeExtraction, "hour", 23, id="hour"),
    ],
)
@pytest.mark.parametrize(
    ("zone", "offset_minutes"),
    [
        pytest.param(None, 0, id="naive"),
        pytest.param("UTC", 0, id="UTC"),
        pytest.param("Etc/GMT-5", 300, id="named-offset"),
        pytest.param("+00:00", 0, id="zero-offset"),
        pytest.param("+05:00", 300, id="positive-offset"),
        pytest.param("-03:30", -210, id="negative-sub-hour-offset"),
    ],
)
@pytest.mark.parametrize("shape", ["values", "all-null", "empty"])
def test_temporal_timezone_support(
    implementation: Any,
    operation: str,
    expected: datetime | int,
    zone: str | None,
    offset_minutes: int,
    shape: str,
) -> None:
    """Numeric fixed-offset zones are rejected up front on Arrow < 22; other zones keep working."""
    local_zone = None if zone is None else timezone(timedelta(minutes=offset_minutes))
    timestamp = datetime(2024, 3, 31, 23, 45, 50, tzinfo=local_zone)
    values = {"values": [timestamp, None], "all-null": [None], "empty": []}[shape]
    timestamp_type = pa.timestamp("us", tz=zone)
    table = pa.table({"event_ts": pa.array(values, type=timestamp_type)})
    feature = f"event_ts__{operation}"
    features = make_feature_set(feature)

    if fixed_offset_unsupported(timestamp_type):
        with pytest.raises(ValueError, match=r"(?i)fixed.offset") as error:
            calculate_with(implementation, table, features)
        assert "22.0.0" in str(error.value)
        assert zone is not None
        assert zone in str(error.value)
        return

    result = calculate_with(implementation, table, features)
    output_type: pa.DataType
    if isinstance(expected, datetime):
        expected = expected.replace(tzinfo=local_zone)
        output_type = timestamp_type
    else:
        output_type = pa.int64()
    expected_values = {"values": [expected, None], "all-null": [None], "empty": []}[shape]
    expected_column = pa.chunked_array([pa.array(expected_values, type=output_type)])
    assert result.column(feature).equals(expected_column)
