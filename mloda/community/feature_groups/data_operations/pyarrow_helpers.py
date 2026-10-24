"""Shared PyArrow helper utilities for row-preserving partitioned operations."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc


def nan_safe_not_equal(
    curr: pa.Array | pa.ChunkedArray, prev: pa.Array | pa.ChunkedArray
) -> pa.Array | pa.ChunkedArray:
    """Like ``pc.not_equal``, but two NaN neighbours compare equal (unlike native ``!=``).

    ``pc.not_equal(nan, nan)`` is ``True`` (Arrow follows IEEE-754 float comparisons),
    which would split a NaN-keyed partition into one row per NaN -- unlike
    ``Table.group_by()``, which merges all NaN keys of a column into a single group. Null
    handling is untouched (a null operand still yields a null result); only the NaN-vs-NaN
    case flips from "changed" to "unchanged". ``curr`` and ``prev`` must be same-typed
    slices of the same column (the float check reads ``curr.type`` only).
    """
    result = pc.not_equal(curr, prev)
    if pa.types.is_floating(curr.type):
        both_nan = pc.and_(pc.fill_null(pc.is_nan(curr), False), pc.fill_null(pc.is_nan(prev), False))
        result = pc.and_(result, pc.invert(both_nan))
    return result


def rename_aggregates_by_position(grouped: pa.Table, keys: list[str], names: list[str]) -> pa.Table:
    """Rename the aggregate columns of a ``group_by().aggregate()`` result by position.

    Auto names collide for a shared source + func. Newer PyArrow puts the keys first, older
    puts them last; anything else is an internal invariant violation.
    """
    columns = list(grouped.column_names)
    k = len(keys)
    if columns[:k] == keys:
        offset = k
    elif columns[len(columns) - k :] == keys:
        offset = 0
    else:
        raise ValueError(f"Unexpected group_by output columns {columns}: keys {keys} are neither first nor last.")
    for i, name in enumerate(names):
        columns[offset + i] = name
    return grouped.rename_columns(columns)
