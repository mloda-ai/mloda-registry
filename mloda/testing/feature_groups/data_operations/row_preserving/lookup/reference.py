"""Pure Python reference for key lookup: runs the group's loop with a kernel-backed compute.

pa.Table is only the I/O container; the kernel runs on ``to_pylist()`` values.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from mloda.provider import FeatureSet

if TYPE_CHECKING:
    import pyarrow as pa

_SAMPLE_SIZE = 10


def _source_family(arrow_type: pa.DataType) -> str:
    import pyarrow as pa

    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "str"
    if pa.types.is_integer(arrow_type) and not pa.types.is_uint64(arrow_type):
        return "int"
    return "other"


def _family(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _output_type(family: str) -> pa.DataType:
    import pyarrow as pa

    return {"str": pa.string(), "int": pa.int64(), "float": pa.float64(), "bool": pa.bool_()}[family]


def _unmatched_error(
    group: Any, feature_name: str, rows: int, distinct: set[Any], reasons: Mapping[Any, str]
) -> Exception:
    keys = sorted(distinct)
    sample = tuple(keys[:_SAMPLE_SIZE])
    sample_reasons = {key: reasons[key] for key in sample if key in reasons}
    row_noun = f"{rows} row" if rows == 1 else f"{rows} rows"
    key_noun = f"{len(keys)} distinct key" if len(keys) == 1 else f"{len(keys)} distinct keys"
    message = (
        f"{feature_name}: {row_noun}, {key_noun} not in the mapping. "
        f"A sorted sample of up to {_SAMPLE_SIZE} keys is on the error's sample attribute."
    )
    error: Exception = group.UNMATCHED_ERROR(message, feature_name, rows, len(keys), sample, sample_reasons)
    return error


def reference_lookup(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """The table plus one pure Python column per lookup feature."""
    import pyarrow as pa

    def _compute(
        data: pa.Table,
        feature_name: str,
        source_col: str,
        mapping: Mapping[Any, Any],
        policy: str,
        reasons: Mapping[Any, str],
    ) -> pa.Table:
        source = data.column(source_col)
        out_type = _output_type(_family(next(iter(mapping.values()))))
        values = source.to_pylist()
        if pa.types.is_null(source.type):
            return data.append_column(feature_name, pa.array([None] * len(values), type=out_type))
        expected = _family(next(iter(mapping)))
        actual = _source_family(source.type)
        if actual != expected:
            raise ValueError(
                f"{feature_name}: mapping keys are {expected} but source column '{source_col}' has type {source.type}"
            )
        result: list[Any] = []
        unmatched_rows = 0
        unmatched_keys: set[Any] = set()
        for value in values:
            if value is None:
                result.append(None)
            elif value in mapping:
                result.append(mapping[value])
            else:
                unmatched_rows += 1
                unmatched_keys.add(value)
                result.append(value if policy == "keep" else None)
        if policy == "raise" and unmatched_rows:
            raise _unmatched_error(group, feature_name, unmatched_rows, unmatched_keys, reasons)
        return data.append_column(feature_name, pa.array(result, type=out_type))

    return group.lookup(table, features, compute_lookup=_compute)
