"""Shared constants and utilities for all data operation feature groups."""

from __future__ import annotations

import re
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

Used by: window_aggregation, aggregation, rank, offset, frame_aggregate.
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

    Applies to: window_aggregation, aggregation, frame_aggregate.
    """

    NULL_IS_GROUP = "null_is_group"
    """Null is a valid group key in partitioned operations.

    Applies to: window_aggregation, aggregation, rank, offset, frame_aggregate.
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


# ---------------------------------------------------------------------------
# Shared PROPERTY_MAPPING guards
# ---------------------------------------------------------------------------


def is_op_token(value: object) -> bool:
    """True for exactly one operation token: a non-empty string, bare or in a single-element container.

    The guard is about arity, not Python syntax. Core unwraps a singleton container when it
    reads a property value (``_unpack_property_value``, ``FeatureGroup.resolve_subtype``), so
    ``("sum",)`` is valid caller syntax for one token. Multi-element and empty values are rejected:
    strict membership checks the elements one by one and would otherwise wrongly match a composite
    value such as ``["sum", "max"]``.
    """
    if isinstance(value, (list, tuple, set, frozenset)):
        if len(value) != 1:
            return False
        (value,) = value
    return isinstance(value, str) and bool(value)


def op_token_value(value: object) -> str:
    """The single token of a value is_op_token accepts, unwrapped from its container."""
    if isinstance(value, (list, tuple, set, frozenset)) and len(value) == 1:
        (value,) = value
    return str(value)


def _is_feature_ref(value: object) -> bool:
    """One source-feature reference: a non-empty string, or a Feature (duck-typed as core's converter does)."""
    if isinstance(value, str):
        return bool(value)
    return hasattr(value, "options")


def is_in_features_value(value: object) -> bool:
    """True for source-feature references core can resolve: non-empty str or Feature, or a container of those."""
    if isinstance(value, (list, tuple, set, frozenset)):
        # An empty container passes: core's in-feature count check owns it as a plain non-match.
        return all(_is_feature_ref(item) for item in value)
    return _is_feature_ref(value)


def is_positive_int(value: object) -> bool:
    """True only for a positive int (rejects bool, non-int, and n < 1)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


#: ASCII decimal >= 1. str.isdigit also accepts superscripts (int() raises) and non-ASCII digits.
_PARAMETRIC_SUFFIX_PATTERN = re.compile(r"[1-9][0-9]*")


def is_parametric_suffix(suffix: str) -> bool:
    """True for the ASCII positive-integer suffix of a parametric operation token (e.g. the 4 in ntile_4)."""
    return _PARAMETRIC_SUFFIX_PATTERN.fullmatch(suffix) is not None
