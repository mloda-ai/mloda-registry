"""Shared constants and utilities for all data operation feature groups."""

from __future__ import annotations

from enum import Enum


# ---------------------------------------------------------------------------
# Shared config key constants
# ---------------------------------------------------------------------------
# These are the keys used in Options to configure data operation feature groups.
# Core's DefaultOptionKeys defines standard keys like "order_by".
# Use DefaultOptionKeys directly for keys that exist in core.
# The constants below are data-operations-specific keys not in DefaultOptionKeys.

PARTITION_BY = "partition_by"
"""Config key for partitioning columns (list of column names).

Used by: window_aggregation, group_aggregation, rank, offset, frame_aggregate.
"""

FRAME_TYPE = "frame_type"
"""Config key for frame type in frame_aggregate operations.

Valid values: "rows", "time", "expanding", "cumulative".
"""

FRAME_SIZE = "frame_size"
"""Config key for frame size (positive integer).

Used by frame_aggregate with frame_type "rows" or "time".
"""

FRAME_UNIT = "frame_unit"
"""Config key for time unit in time-interval frames.

Used by frame_aggregate with frame_type "time".
Valid values: second, minute, hour, day, week, month, year.
"""


# ---------------------------------------------------------------------------
# Null handling policy constants
# ---------------------------------------------------------------------------
# These document the data operations null handling contract.
# Implementations must match these defaults. PyArrow behavior is the reference.


class NullPolicy(str, Enum):
    """Null handling behavior constants for data operations.

    Each value describes a null handling rule. Implementations must match
    PyArrow's behavior as the reference. Where a framework diverges from
    these defaults, add explicit convergence code (e.g. pandas groupby
    needs ``dropna=False``; SQLite rank needs an explicit null-last clause).

    These constants are documentation and configuration anchors, not
    runtime enforcement. Each package's ``calculate_feature`` is responsible
    for honoring the policy.
    """

    PROPAGATE = "propagate"
    """Element-wise operations return null for null input (null in, null out).

    Applies to: datetime, string, binning.
    """

    SKIP = "skip"
    """Aggregations skip null values (e.g. SUM ignores nulls).

    Applies to: window_aggregation, group_aggregation, frame_aggregate.
    """

    NULL_IS_GROUP = "null_is_group"
    """Null is a valid group key in partitioned operations.

    Applies to: window_aggregation, group_aggregation, rank, offset, frame_aggregate.
    Pandas divergence: pass ``dropna=False`` to ``groupby()``.
    """

    NULLS_LAST = "nulls_last"
    """Nulls rank last in ordered operations.

    Applies to: rank.
    SQLite divergence: add ``CASE WHEN col IS NULL THEN 1 ELSE 0 END`` to ORDER BY.
    """

    EDGE_NULL = "edge_null"
    """Out-of-range positions produce null (e.g. lag/lead at table edges).

    Applies to: offset.
    """
