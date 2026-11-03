"""Shared tie layouts (group, order key, value) for tied-order tests across row-preserving operations."""

from __future__ import annotations

# small: a 3-row case where Polars' small-input sort is unstable.
TIE_SMALL = (["a", "a", "a"], [2, 2, 1], [1.0, 2.0, 3.0])
# long: all rows tied; DuckDB reorders ties at this size.
TIE_LONG = (["a"] * 25, [1] * 25, [float(i) for i in range(25)])
