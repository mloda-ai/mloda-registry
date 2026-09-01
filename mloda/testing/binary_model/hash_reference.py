"""The "hash" operation's reference algorithm: one independent implementation, imported both by
``simulated_binary.py`` (so the CLI stub's own output is produced by exactly this code, never a
second hand-written copy that could drift) and by ``conformance.py``'s
``HashOperationConformanceMixin`` (so a test's expected value is never derived from whatever the
binary happens to do). Keeping the algorithm in one place with no dependency back on either caller
is what makes that one-way sharing possible.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Callable

import pyarrow as pa

# Fixed sentinel/delimiter for the "hash" reference algorithm. The sentinel embeds NUL bytes so it
# can never collide with a real utf8 value a caller could plausibly send; the delimiter is the
# ASCII unit separator (0x1F), likewise not expected in ordinary text input.
HASH_NULL_SENTINEL = "\x00__NULL__\x00"
HASH_FIELD_DELIMITER = "\x1f"


def _hash_value_token(value: Any) -> str:
    """One row value's token in the "hash" reference algorithm (see ``compute_expected_hash``).
    Order matters: ``bool`` is checked before ``int`` since ``bool`` is an ``int`` subclass in
    Python.

    The float64 branch encodes the value's raw IEEE-754 binary64 bytes, big-endian, as lowercase
    hex (``struct.pack(">d", value).hex()``), never a decimal string. A decimal ``repr()`` is
    CPython-specific and not reproducible by a Rust (or any other language) implementation of the
    same operation: ``0.0`` reprs as ``"0.0"`` in Python but formats as ``"0"`` under Rust's
    default ``Display``, and ``1e16`` reprs as ``"1e+16"`` vs Rust's ``"10000000000000000"``. Since
    this conformance kit exists precisely so the three implementations (stub, wrapper, real
    binary) cannot drift apart, the token must be an unambiguous encoding of the exact bit
    pattern instead. Two consequences of hashing the raw bits rather than the decimal value,
    both deliberate and not bugs:
    - ``-0.0`` and ``0.0`` have distinct IEEE-754 bit patterns (the sign bit is the only
      difference) and therefore hash to different tokens; a caller that wants them to hash
      identically must normalize before calling.
    - Any NaN bit pattern is used exactly as given, unmodified: this reference implementation
      does not canonicalize NaN, so two NaN values with different bit patterns (a signalling vs
      quiet NaN, or a different payload) hash to different tokens. This is a deliberate,
      documented choice, not an oversight.
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
    """Independent reference implementation of the "hash" operation, computed with Python's
    ``hashlib`` so a test's expected value is never derived from whatever the binary happens to
    do. This is the algorithm's own specification (the contract leaves "hash" operation-defined):

    1. For each value of the row, in ``input_columns`` order (the operation's input contract, per
       contract: Data -- the order fields happen to appear in the stream is not), produce a token:
       - null -> ``HASH_NULL_SENTINEL``;
       - boolean -> ``"true"`` / ``"false"``;
       - int64 -> ``str(value)``;
       - float64 -> the value's raw IEEE-754 binary64 bytes, big-endian, hex-encoded (see
         ``_hash_value_token`` for the exact rules, including ``-0.0``/NaN treatment) -- not a
         decimal string representation, which is not portable across implementations;
       - utf8 -> the string value, unmodified.
    2. Join the row's tokens with ``HASH_FIELD_DELIMITER``.
    3. Prepend ``key`` (``parameters.key``, or the empty string if the operation was invoked
       without one -- ``key`` entirely absent and ``key: ""`` both mean "no key" and must produce
       the same digest; only ``None`` (absent) is treated as empty, no other falsy value) followed
       by one more ``HASH_FIELD_DELIMITER``.
    4. UTF-8 encode the resulting string and hash it with BLAKE2b, ``digest_size=8`` (a 64-bit /
       8-byte digest).
    5. Interpret the 8 raw digest bytes as a big-endian *signed* 64-bit integer
       (``struct.unpack(">q", digest)[0]``); this is the row's single "result" output value, of
       column type int64.
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
    vocabulary type) with a null in one column, its Arrow schema, a structurally valid config for
    it (built via ``make_config``, so a subclass's overridden config shape is honoured), and the
    expected output column computed independently via ``compute_expected_hash_column`` (contract:
    Configuration "hash" operation shape). ``key`` is forwarded to both the config's
    ``parameters.key`` and the independent computation, so calling this with a different key
    exercises the "with parameters.key" variant of the operation with the same rows.

    ``id`` gets a distinct value per row so a row-order bug is caught even though the hash of a
    row also depends on every other column; ``amount`` is null on one row to exercise the
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
