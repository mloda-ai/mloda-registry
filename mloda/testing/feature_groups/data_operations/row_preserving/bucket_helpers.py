"""Shared by the pure Python references of binning and time bucketization."""

from __future__ import annotations

import math


def bucket_index(offset: int | float, width: int | float) -> int:
    """Floor of offset / width: exact for ints; floats divide first because ``1.0 // 0.1 == 9.0``."""
    if isinstance(offset, int) and isinstance(width, int):
        return offset // width
    return math.floor(offset / width)
