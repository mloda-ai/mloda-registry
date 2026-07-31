"""Shared constants and utilities for all data operation feature groups."""

from __future__ import annotations

import re
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from mloda.core.abstract_plugins.components.options import Options

T = TypeVar("T")


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
# Shared PROPERTY_MAPPING guards and their read-site unwrappers
# ---------------------------------------------------------------------------


def _unwrap_singleton(value: Any) -> Any:
    """The one element of a single-element container, or the value itself.

    The arity rule every scalar guard and unwrapper below shares, in one place. Core unwraps a
    singleton container when it reads a property value (``_unpack_property_value``,
    ``FeatureGroup.resolve_subtype``), so a one-element container is valid caller syntax for one
    value. A multi-element or empty container is returned unchanged, which fails every guard's
    type check: strict membership checks the elements one by one and would otherwise wrongly
    match a composite value such as ``["sum", "max"]``.
    """
    if isinstance(value, (list, tuple, set, frozenset)) and len(value) == 1:
        (element,) = value
        return element
    return value


def option_value(options: Options, key: str, unwrap: Callable[[Any], T]) -> T | None:
    """The option under ``key`` unwrapped from its container; None when the option is absent.

    Absent stays None instead of being routed through an unwrapper (which would turn it into
    the string ``"None"``), so a missing option keeps its family's absent-value behavior.
    """
    value = options.get(key)
    return None if value is None else unwrap(value)


def is_op_token(value: object) -> bool:
    """True for exactly one operation token: a non-empty string, bare or in a single-element container.

    The guard is about arity, not Python syntax: ``("sum",)`` is one token, while ``["sum", "max"]``
    is a composite value that strict membership would otherwise wrongly match element by element.
    """
    token = _unwrap_singleton(value)
    return isinstance(token, str) and bool(token)


def op_token_value(value: object) -> str:
    """The single token of a value is_op_token accepts, unwrapped from its container."""
    return str(_unwrap_singleton(value))


def is_column_ref(value: object) -> bool:
    """True for exactly one column name: a non-empty string, bare or in a single-element container.

    Same value space and arity rule as is_op_token, delegated rather than copied; the second name
    exists for the declaration sites, where a column reference reads nothing like an op token.
    """
    return is_op_token(value)


def column_ref_value(value: object) -> str:
    """The single column name of a value is_column_ref accepts, unwrapped from its container."""
    return op_token_value(value)


def is_scalar_number(value: object) -> bool:
    """True for exactly one int or float, bare or in a single-element container (bool is not a number)."""
    number = _unwrap_singleton(value)
    return isinstance(number, (int, float)) and not isinstance(number, bool)


def scalar_number_value(value: object) -> int | float:
    """The single number of a value is_scalar_number accepts, unwrapped from its container."""
    number: int | float = _unwrap_singleton(value)
    return number


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
    """True only for exactly one positive int, bare or in a single-element container (rejects bool, non-int, n < 1)."""
    n = _unwrap_singleton(value)
    return isinstance(n, int) and not isinstance(n, bool) and n >= 1


def positive_int_value(value: object) -> int:
    """The single int of a value is_positive_int accepts, unwrapped from its container."""
    n: int = _unwrap_singleton(value)
    return n


#: ASCII decimal >= 1. str.isdigit also accepts superscripts (int() raises) and non-ASCII digits.
_PARAMETRIC_SUFFIX_PATTERN = re.compile(r"[1-9][0-9]*")


def is_parametric_suffix(suffix: str) -> bool:
    """True for the ASCII positive-integer suffix of a parametric operation token (e.g. the 4 in ntile_4)."""
    return _PARAMETRIC_SUFFIX_PATTERN.fullmatch(suffix) is not None
