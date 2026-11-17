"""Pure Python reference for point and scalar arithmetic: runs the group's loop with a kernel-backed compute.

pa.Table is only the I/O container; the element kernel runs on ``to_pylist()`` values.
"""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
from mloda.provider import FeatureSet

ARITHMETIC_OPS = frozenset({"add", "subtract", "multiply", "divide"})

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _non_numeric(table: pa.Table, col: str) -> pa.DataType | None:
    """The column type when it is not integer or floating (bool, string, null and decimal included), else None."""
    arrow_type = table.column(col).type
    if pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type):
        return None
    return arrow_type


def _declared_type(op: str, *is_float: bool) -> pa.DataType:
    return pa.float64() if op == "divide" or any(is_float) else pa.int64()


def _divide(a: float, b: float) -> float:
    if b == 0:
        if a == 0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, math.copysign(1.0, a) * math.copysign(1.0, b))
    return a / b


def _element(a: Any, b: Any, op: str) -> Any:
    """One result value: None in either operand wins, NaN propagates, divide follows IEEE 754."""
    if op not in ARITHMETIC_OPS:
        raise ValueError(f"Unsupported arithmetic operation {op!r}. Supported: {sorted(ARITHMETIC_OPS)}")
    if a is None or b is None:
        return None
    if op == "divide":
        return _divide(float(a), float(b))
    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    return a * b


def _append(table: pa.Table, feature_name: str, op: str, a: list[Any], b: list[Any], declared: pa.DataType) -> pa.Table:
    values = [_element(x, y, op) for x, y in zip(a, b)]
    if pa.types.is_integer(declared):
        for value in values:
            if value is not None and not _INT64_MIN <= value <= _INT64_MAX:
                raise ValueError(f"Integer result {value} of feature {feature_name!r} is outside int64")
    return table.append_column(feature_name, pa.array(values, type=declared))


def _compute_point(table: pa.Table, feature_name: str, col_a: str, col_b: str, op: str) -> pa.Table:
    declared = _declared_type(op, *(pa.types.is_floating(table.column(col).type) for col in (col_a, col_b)))
    return _append(table, feature_name, op, table.column(col_a).to_pylist(), table.column(col_b).to_pylist(), declared)


def _compute_scalar(table: pa.Table, feature_name: str, source_col: str, op: str, constant: int | float) -> pa.Table:
    declared = _declared_type(op, pa.types.is_floating(table.column(source_col).type), isinstance(constant, float))
    values = table.column(source_col).to_pylist()
    return _append(table, feature_name, op, values, [constant] * len(values), declared)


def reference_point_arithmetic(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """The table plus one pure Python column per point arithmetic feature."""
    return group.apply_arithmetic(table, features, non_numeric=_non_numeric, compute_arithmetic=_compute_point)


def reference_scalar_arithmetic(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """The table plus one pure Python column per scalar arithmetic feature."""
    return group.apply_arithmetic(table, features, non_numeric=_non_numeric, compute_arithmetic=_compute_scalar)
