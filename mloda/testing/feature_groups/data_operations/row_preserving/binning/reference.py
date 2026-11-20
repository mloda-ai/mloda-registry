"""Pure Python reference for binning: runs the group's loop with a Python compute.

NaN and None are missing; ``bin`` computes in float64 on ``to_pylist()`` values, ``qbin`` ranks by value.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.row_preserving.bucket_helpers import bucket_index

if TYPE_CHECKING:
    import pyarrow as pa

BINNING_OP_NAMES = frozenset({"bin", "qbin"})
_MAX_EXACT_INT = 2**53


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _equal_width(values: list[Any], n_bins: int) -> list[int | None]:
    present = [v for v in values if not _is_missing(v)]
    for v in present:
        if isinstance(v, float) and math.isinf(v):
            raise ValueError("bin cannot split a range with an infinite value")
        if isinstance(v, int) and abs(v) > _MAX_EXACT_INT:
            raise ValueError(f"bin needs integers within 2**53 to stay exact in float64, got {v}")
    if not present:
        return [None] * len(values)
    lo, hi = min(present), max(present)
    if lo == hi:
        return [None if _is_missing(v) else 0 for v in values]
    width = (float(hi) - float(lo)) / n_bins
    return [None if _is_missing(v) else min(bucket_index(float(v) - float(lo), width), n_bins - 1) for v in values]


def _quantile(values: list[Any], n_bins: int) -> list[int | None]:
    order = sorted((i for i, v in enumerate(values) if not _is_missing(v)), key=lambda i: values[i])
    result: list[int | None] = [None] * len(values)
    for rank, index in enumerate(order):
        result[index] = rank * n_bins // len(order)
    return result


def _compute(table: pa.Table, feature_name: str, source_col: str, op: str, n_bins: int) -> pa.Table:
    import pyarrow as pa

    if op not in BINNING_OP_NAMES:
        raise ValueError(f"Unsupported binning operation {op!r}. Supported: {sorted(BINNING_OP_NAMES)}")
    arrow_type = table.column(source_col).type
    if not (pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type)):
        raise ValueError(f"Binning needs an integer or floating source column, but {source_col!r} is {arrow_type}")
    values = table.column(source_col).to_pylist()
    result = _equal_width(values, n_bins) if op == "bin" else _quantile(values, n_bins)
    return table.append_column(feature_name, pa.array(result, type=pa.int64()))


def reference_binning(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """The table plus one pure Python column per binning feature."""
    return group.assign_bins(table, features, compute_binning=_compute)
