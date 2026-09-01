"""The "hash" operation's reference algorithm: one independent implementation, imported by both
``simulated_binary.py`` (so the stub's output is produced by exactly this code) and
``conformance.py``'s ``HashOperationConformanceMixin`` (so expected values never derive from
whatever the binary happens to do). No dependency back on either caller, which is what makes that
one-way sharing possible.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Callable

import pyarrow as pa

# Sentinel/delimiter for the "hash" algorithm: the sentinel embeds NUL bytes so it can't collide
# with real utf8 input; the delimiter is the ASCII unit separator (0x1F), also unlikely in text.
HASH_NULL_SENTINEL = "\x00__NULL__\x00"
HASH_FIELD_DELIMITER = "\x1f"


def _hash_value_token(value: Any) -> str:
    """One row value's token (see ``compute_expected_hash``). ``bool`` is checked before ``int``
    since ``bool`` is an ``int`` subclass in Python.

    float64 encodes the raw IEEE-754 binary64 bytes, big-endian, as lowercase hex, not a decimal
    string: a decimal repr is CPython-specific and not reproducible by another language's
    implementation of this operation (e.g. Rust formats ``0.0`` as ``"0"``, ``1e16`` as
    ``"10000000000000000"``). Two deliberate consequences of hashing raw bits: ``-0.0``/``0.0``
    hash differently (distinct sign bit), and NaN bit patterns are used as-is, uncanonicalized.
    """
    if value is None:
        return HASH_NULL_SENTINEL
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return struct.pack(">d", value).hex()
    if isinstance(value, str):
        return value
    raise TypeError(f"unsupported value type for the hash reference algorithm: {type(value)!r}")


def compute_expected_hash(key: str | None, row_values: list[Any]) -> int:
    """Independent reference implementation of "hash" (the contract leaves the operation itself
    operation-defined): tokenize each value of the row, in ``input_columns`` order, via
    ``_hash_value_token``; join with ``HASH_FIELD_DELIMITER``; prepend ``key`` (or ``""`` if
    ``None`` -- absent and empty-string keys must hash identically, no other falsy value counts as
    empty) plus one more delimiter; UTF-8 encode; BLAKE2b digest (``digest_size=8``); interpret the
    8 bytes as a big-endian signed int64.
    """
    row_text = HASH_FIELD_DELIMITER.join(_hash_value_token(value) for value in row_values)
    message = f"{key if key is not None else ''}{HASH_FIELD_DELIMITER}{row_text}".encode("utf-8")
    digest = hashlib.blake2b(message, digest_size=8).digest()
    result: int = struct.unpack(">q", digest)[0]
    return result


def compute_expected_hash_column(rows: dict[str, list[Any]], input_columns: list[str], key: str | None) -> list[int]:
    """Apply ``compute_expected_hash`` row by row, reading each row's values in ``input_columns``
    order from ``rows`` (column name -> one Python value per row)."""
    num_rows = len(next(iter(rows.values())))
    return [
        compute_expected_hash(key, [rows[column][row_index] for column in input_columns])
        for row_index in range(num_rows)
    ]


def hash_multi_column_case(
    *,
    key: str | None,
    output_column_name: str,
    make_config: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Build one self-contained "hash" test case: a small multi-column, multi-row dataset (every
    vocabulary type, one null) with its Arrow schema, a config built via ``make_config``, and the
    expected output computed independently via ``compute_expected_hash_column`` (contract:
    Configuration "hash" operation shape). ``key`` is forwarded to both the config and the
    independent computation. ``id`` varies per row so a row-order bug is caught even though the
    hash also depends on every other column; ``amount`` is null on one row to exercise the
    null-sentinel path.
    """
    input_columns = ["id", "count", "amount", "active", "name"]
    rows: dict[str, list[Any]] = {
        "id": ["row-0", "row-1", "row-2", "row-3"],
        "count": [10, -5, 0, 42],
        "amount": [1.5, None, -3.25, 0.0],
        "active": [True, False, True, False],
        "name": ["alpha", "beta", "gamma", "delta"],
    }
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("count", pa.int64()),
            pa.field("amount", pa.float64()),
            pa.field("active", pa.bool_()),
            pa.field("name", pa.string()),
        ]
    )
    output_columns = {"result": output_column_name}
    parameters: dict[str, Any] = {} if key is None else {"key": key}
    config = make_config(input_columns=input_columns, parameters=parameters, output_columns=output_columns)
    expected = compute_expected_hash_column(rows, input_columns, key)
    return {
        "input_columns": input_columns,
        "rows": rows,
        "schema": schema,
        "output_columns": output_columns,
        "config": config,
        "expected": expected,
    }
