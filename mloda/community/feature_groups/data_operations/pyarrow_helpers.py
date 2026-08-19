"""Shared PyArrow helper utilities for row-preserving partitioned operations."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc


def nan_safe_not_equal(curr: pa.Array, prev: pa.Array) -> pa.Array:
    """Like ``pc.not_equal``, but two NaN neighbours compare equal (unlike native ``!=``).

    ``pc.not_equal(nan, nan)`` is ``True`` (Arrow follows IEEE-754 float comparisons),
    which would split a NaN-keyed partition into one row per NaN -- unlike
    ``Table.group_by()``, which merges all NaN keys of a column into a single group. Null
    handling is untouched (a null operand still yields a null result); only the NaN-vs-NaN
    case flips from "changed" to "unchanged".
    """
    result = pc.not_equal(curr, prev)
    if pa.types.is_floating(curr.type):
        both_nan = pc.and_(pc.fill_null(pc.is_nan(curr), False), pc.fill_null(pc.is_nan(prev), False))
        result = pc.and_(result, pc.invert(both_nan))
    return result
