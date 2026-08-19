"""Shared PythonDict helpers for group keys, sorting, reductions, and tz math.

Used across the PythonDict data-operation backends so null/NaN, DST offsets, and
numeric-column checks are handled identically everywhere.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error

# Sentinel substituted for any NaN partition-key value so NaN-valued rows of a partition
# column land in one shared group, matching PyArrow's Table.group_by() (which merges all
# NaN keys into a single group, distinct from None).
_NAN_KEY = object()


def is_nan(value: Any) -> bool:
    """True for any NaN-like value (float, Decimal, numpy float32/64, ...).

    Uses the universal self-inequality NaN test (``value != value``), IEEE-754's own
    definition of NaN, so it works across every NaN-capable numeric type without
    enumerating them one by one. A pathological ``__ne__`` that raises is treated as
    "not NaN" rather than propagating.
    """
    try:
        return bool(value != value)
    except Exception:
        return False


def is_null_like(value: Any) -> bool:
    """True for None or NaN, the two order-value tiers rank treats as one tied null group."""
    return value is None or is_nan(value)


def group_key_value(value: Any) -> Any:
    """Map a partition-column value to a hashable, NaN-safe group-key component.

    ``float('nan')`` never equals another NaN, so used raw as a dict key it would split
    every NaN-valued row into its own singleton group; substitute a shared sentinel instead.
    """
    if is_nan(value):
        return _NAN_KEY
    return value


def nulls_last_sort_key(value: Any) -> tuple[int, Any]:
    """Total order placing both ``None`` and NaN last, safe to pass to ``sorted()``.

    Both map to the same last-place sentinel, so ``sorted()`` never compares a NaN or a
    None against a real value of a different type (e.g. a datetime) with ``<``. Use this
    for ORDER-value sorting (e.g. rank's ``order_by``), where None and NaN are meant to
    tie at one rank. For PARTITION-key sorting, where None and NaN must stay distinct,
    contiguous groups, use ``partition_sort_key`` instead.
    """
    if value is None or is_nan(value):
        return (1, 0)
    return (0, value)


def partition_sort_key(value: Any) -> tuple[int, Any]:
    """Total order for PARTITION-key sorting: real values first, then a distinct, contiguous tier each for NaN and None.

    Unlike ``nulls_last_sort_key`` (which merges None and NaN into one tier for order-value
    ties), this keeps every None-keyed row contiguous and separate from every NaN-keyed row,
    matching PyArrow's ``Table.group_by()`` treating null and NaN as distinct groups.
    """
    if is_nan(value):
        return (1, 0)
    if value is None:
        return (2, 0)
    return (0, value)


def values_equal(a: Any, b: Any) -> bool:
    """NaN-safe equality: NaN equals NaN here, unlike Python's own ``==``.

    Recurses into tuples element-wise, so this also works as a comparator for
    group-key tuples.
    """
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    if is_nan(a) and is_nan(b):
        return True
    return bool(a == b)


def order_values_equal(a: Any, b: Any) -> bool:
    """Order-value equality for rank run detection: any two null-likes (None or NaN) tie.

    Unlike ``values_equal`` (where NaN equals NaN but not None), this also ties None
    against NaN, so a run of mixed null-like ``order_by`` values sorts and ranks together.
    """
    if is_null_like(a) and is_null_like(b):
        return True
    return values_equal(a, b)


# All aggregation types supported by the PythonDict aggregation/window_aggregation backends.
SUPPORTED_AGG_TYPES: frozenset[str] = frozenset(
    {
        "sum",
        "avg",
        "mean",
        "count",
        "min",
        "max",
        "std",
        "var",
        "std_pop",
        "std_samp",
        "var_pop",
        "var_samp",
        "median",
        "mode",
        "nunique",
        "first",
        "last",
    }
)

# ddof (delta degrees of freedom) per std/var variant. std/var/std_pop/var_pop are
# population (ddof=0); std_samp/var_samp are sample (ddof=1).
VARIANCE_DDOF: dict[str, int] = {
    "std": 0,
    "var": 0,
    "std_pop": 0,
    "var_pop": 0,
    "std_samp": 1,
    "var_samp": 1,
}

# std_* variants take a square root of the variance; var_* variants return the variance.
STD_AGG_TYPES: frozenset[str] = frozenset({"std", "std_pop", "std_samp"})


def mode(values: list[Any]) -> Any:
    """Most frequent non-null value; ties broken by first occurrence in *values*.

    Counts are keyed by ``group_key_value``, so distinct NaN objects (as produced by
    ``Table.to_pylist()``) merge into one candidate instead of each comparing unequal to
    itself and forming its own singleton count. Dict insertion order preserves
    first-occurrence order, so a strict ``count > best_count`` comparison keeps the
    earlier value on a tie.
    """
    counts: dict[Any, int] = {}
    representative: dict[Any, Any] = {}
    for v in values:
        if v is None:
            continue
        key = group_key_value(v)
        if key not in counts:
            representative[key] = v
        counts[key] = counts.get(key, 0) + 1

    best_key: Any = None
    best_count = -1
    for key, count in counts.items():
        if count > best_count:
            best_key = key
            best_count = count
    return representative.get(best_key)


def variance(non_null: list[float], *, ddof: int, as_std: bool) -> float | None:
    """Population (ddof=0) or sample (ddof=1) variance/std of *non_null*.

    Returns ``None`` when there are fewer than ``ddof + 1`` values (e.g. an
    empty group, or a single-value group for the sample variant).
    """
    n = len(non_null)
    if n - ddof <= 0:
        return None
    m = sum(non_null) / n
    var = sum((x - m) ** 2 for x in non_null) / (n - ddof)
    return var**0.5 if as_std else var


def reduce_agg(agg_type: str, values: list[Any]) -> Any:
    """Reduce one group's raw (possibly null-containing) values per *agg_type*.

    NaN is skipped (in addition to None) for ``min``/``max``, matching PyArrow's
    ``pc.min``/``pc.max``.
    """
    non_null = [v for v in values if v is not None]

    if agg_type == "count":
        return len(non_null)
    if agg_type == "nunique":
        # Normalize through group_key_value so distinct NaN objects merge into one
        # distinct value, matching PyArrow's pc.count_distinct.
        return len({group_key_value(v) for v in non_null})
    if agg_type == "mode":
        return mode(values)
    if agg_type == "first":
        return non_null[0] if non_null else None
    if agg_type == "last":
        return non_null[-1] if non_null else None
    if agg_type == "median":
        return statistics.median(non_null) if non_null else None
    if agg_type in VARIANCE_DDOF:
        return variance(non_null, ddof=VARIANCE_DDOF[agg_type], as_std=agg_type in STD_AGG_TYPES)

    if not non_null:
        return None
    if agg_type == "sum":
        return sum(non_null)
    if agg_type in ("avg", "mean"):
        return sum(non_null) / len(non_null)
    if agg_type == "min":
        finite = [v for v in non_null if not is_nan(v)]
        return min(finite) if finite else None
    if agg_type == "max":
        finite = [v for v in non_null if not is_nan(v)]
        return max(finite) if finite else None

    raise unsupported_agg_type_error(agg_type, SUPPORTED_AGG_TYPES, framework="PythonDict")


_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Fixed-duration units bucketed by an epoch-anchored floor (see floor_fixed_duration).
SECONDS_PER_UNIT: dict[str, int] = {"minute": 60, "hour": 3600, "day": 86400}


def attach_tzinfo(naive: datetime, tzinfo: Any) -> datetime:
    """Attach *tzinfo* to a naive datetime, resolving its DST offset correctly for its own date.

    ``zoneinfo.ZoneInfo`` and fixed-offset ``datetime.timezone`` resolve their offset
    dynamically, so a plain ``.replace(tzinfo=...)`` is correct for them. ``pytz``'s
    ``DstTzInfo`` pins its offset at construction time and does not recompute it on
    ``.replace()``, so it needs its ``localize`` re-localization API instead (duck-typed here,
    since every other tzinfo type's ``.replace()`` is already correct).
    """
    if tzinfo is None:
        return naive
    localize = getattr(tzinfo, "localize", None)
    if callable(localize):
        localized: datetime = localize(naive)
        return localized
    return naive.replace(tzinfo=tzinfo)


def wall_clock_epoch_seconds(dt: datetime) -> int:
    """Whole seconds between 1970-01-01T00:00:00 and *dt*'s wall-clock fields.

    Computed by discarding ``dt``'s real ``tzinfo`` and treating its calendar/clock fields as
    if they were already UTC, since PyArrow's ``floor/ceil/round_temporal`` bucket by a
    timestamp's *local* wall-clock representation, not its true UTC instant. Anchoring this
    wall-clock value to the Unix epoch (rather than resetting within the enclosing hour/day)
    makes the floor correct for any ``n``, including values that don't evenly divide 60
    (minute) or 24 (hour).
    """
    delta = dt.replace(tzinfo=timezone.utc) - _EPOCH_UTC
    return delta.days * 86400 + delta.seconds


def floor_fixed_duration(dt: datetime, n: int, unit: str) -> datetime:
    """Floor ``dt`` to its epoch-anchored ``(n, unit)`` bucket start, preserving tzinfo.

    ``unit`` must be a key of ``SECONDS_PER_UNIT`` (minute/hour/day); the caller validates
    the unit before calling. ``dt.tzinfo`` is reattached only once the final floored
    wall-clock date is known, so a floor that crosses a DST transition gets the correct
    offset for its own date.
    """
    bucket_seconds = n * SECONDS_PER_UNIT[unit]
    floored_seconds = (wall_clock_epoch_seconds(dt) // bucket_seconds) * bucket_seconds
    naive = (_EPOCH_UTC + timedelta(seconds=floored_seconds)).replace(tzinfo=None)
    return attach_tzinfo(naive, dt.tzinfo)


def input_columns_and_framework(data: dict[str, list[Any]]) -> tuple[list[str], str]:
    """All dict keys are input columns; framework label is always ``"PythonDict"``."""
    return list(data.keys()), "PythonDict"


def non_numeric_descriptor(data: dict[str, list[Any]], source_col: str) -> object | None:
    """Type-name descriptor for ``source_col``'s first non-numeric value, else ``None`` (bool counts as non-numeric)."""
    for value in data[source_col]:
        if value is None:
            continue
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, (int, float)):
            continue
        return type(value).__name__
    return None
