"""Pure Python reference for datetime extraction: runs the group's loop with a Python compute.

Fields come from Python datetime attributes of ``to_pylist()`` values; aware values keep their column timezone.
DATE sources work for date-level ops; ns timestamps are outside the reference.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from mloda.provider import FeatureSet

if TYPE_CHECKING:
    import pyarrow as pa

DATETIME_OP_NAMES = frozenset(
    {"year", "month", "day", "hour", "minute", "second", "dayofweek", "is_weekend", "quarter"}
)
TIME_OF_DAY_OPS = frozenset({"hour", "minute", "second"})


def _element(value: datetime | date, op: str) -> int:
    if op == "dayofweek":
        return value.weekday()
    if op == "is_weekend":
        return 1 if value.weekday() >= 5 else 0
    if op == "quarter":
        return (value.month - 1) // 3 + 1
    return int(getattr(value, op))


def _compute(table: pa.Table, feature_name: str, source_col: str, op: str) -> pa.Table:
    import pyarrow as pa

    if op not in DATETIME_OP_NAMES:
        raise ValueError(f"Unsupported datetime operation {op!r}. Supported: {sorted(DATETIME_OP_NAMES)}")
    arrow_type = table.column(source_col).type
    is_date = pa.types.is_date(arrow_type)
    if not (is_date or pa.types.is_timestamp(arrow_type)):
        raise ValueError(
            f"Datetime operation {op!r} requires a timestamp or date source column, but {source_col!r} is {arrow_type}"
        )
    if is_date and op in TIME_OF_DAY_OPS:
        raise ValueError(f"Datetime operation {op!r} needs a time of day, but {source_col!r} is a DATE ({arrow_type})")
    if not is_date and arrow_type.unit == "ns":
        raise ValueError(f"Source column {source_col!r} has nanosecond precision, which Python datetime cannot hold")
    values = [None if v is None else _element(v, op) for v in table.column(source_col).to_pylist()]
    return table.append_column(feature_name, pa.array(values, type=pa.int64()))


def reference_datetime(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """The table plus one pure Python column per datetime feature."""
    return group.extract_datetime(table, features, compute_datetime=_compute)
