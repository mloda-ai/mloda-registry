"""Declared-attribute helpers shared by the OpenLineage, audit and OTel extenders."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_INT_MIN = -(2**63)
_INT_MAX = 2**63 - 1


def declared_scalars(declared: Mapping[str, Any] | None) -> dict[str, str | int | float | bool]:
    """A fresh dict of the str, bool, int and finite float declarations, types unchanged."""
    if declared is None:
        return {}
    # Dropped: values that are not portable JSON numbers (non-finite floats, which allow_nan=False rejects and
    # OpenLineage would emit as NaN; ints beyond 64 bits, which also break json.dumps past Python's int digit limit).
    return {
        key: value
        for key, value in declared.items()
        if isinstance(value, (str, bool))
        or (isinstance(value, int) and _INT_MIN <= value <= _INT_MAX)
        or (isinstance(value, float) and math.isfinite(value))
    }


def same_scalar(left: object, right: object) -> bool:
    """True only when both have exactly the same type and are equal."""
    return type(left) is type(right) and left == right
