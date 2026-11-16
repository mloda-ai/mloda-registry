"""Format of the pinning tables that ship next to a reference: hand-computed inputs and expected outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class PinningRow:
    """One pinned case: a feature over input columns gives the expected output columns."""

    case: str
    feature: str
    inputs: Mapping[str, tuple[Any, ...]]
    expected: Mapping[str, tuple[Any, ...]]
    context: Mapping[str, Any] = field(default_factory=dict)
    types: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PinningTable:
    """The pinning rows of one operation and the rule text of each case label."""

    operation: str
    rules: Mapping[str, str]
    rows: tuple[PinningRow, ...]


AGGREGATE_RULES: Mapping[str, str] = MappingProxyType(
    {
        "none": "Skipped by every agg type; an empty or all-None group gives 0 for count and nunique, None for the rest.",
        "nan": (
            "A value, distinct from None: sum, avg, mean, std* and var* propagate it; count counts it; min and max skip it "
            "(all-NaN returns NaN); median skips it (all-NaN returns None); mode and nunique count it as one value; "
            "first and last skip only None."
        ),
        "signed_zero": "0.0 and -0.0 are one value (nunique, mode) and one partition key.",
        "std_var": (
            "Population for std, var, std_pop and var_pop; ddof=1 for the *_samp types; fewer than ddof+1 values gives None."
        ),
        "mode": "Ties keep the first occurrence.",
        "partition_keys": (
            "NaN keys form one group, None is its own key, groups come in first-occurrence order; "
            "partition_by=[] is one whole-table group, also on an empty table."
        ),
        "dtype": (
            "count and nunique are int64; avg, mean, median, std* and var* are float64; sum widens signed integers to "
            "int64, unsigned integers and bool to uint64 and floats to float64; min, max, first, last and mode keep the source type."
        ),
        "order": (
            "Within a group, order_by (window first and last, frame) sorts ascending: real values, then NaN, then None, "
            "stable."
        ),
        "row_order": "Window, frame and scalar keep input order.",
    }
)
