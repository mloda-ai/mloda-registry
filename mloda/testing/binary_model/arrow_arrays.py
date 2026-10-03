"""Builds output arrays for the fake binaries' ``run`` path straight from Arrow buffers.

pyarrow's Python-to-Arrow constructors (``pa.array(list)``, ``pa.array(numpy)``, ``pa.table(dict)``)
lazily import pandas on first use, costing every fake-binary invocation a pandas import it never
uses; ``pa.Array.from_buffers`` does not. Kept apart from ``arrow.py`` so ``simulated_binary.py``
can use it without importing the mixin.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from typing import Any

import pyarrow as pa

# struct format character per supported fixed-width integer type (little-endian, as Arrow stores it).
_INT_FORMATS: dict[str, str] = {"int32": "i", "int64": "q"}


def _utf8_array(values: Sequence[str | None]) -> pa.Array:
    offsets = [0]
    chunks: list[bytes] = []
    validity = bytearray((len(values) + 7) // 8)
    for index, value in enumerate(values):
        if value is not None:
            validity[index >> 3] |= 1 << (index & 7)
            chunks.append(value.encode("utf-8"))
        offsets.append(offsets[-1] + (len(chunks[-1]) if value is not None else 0))
    validity_buffer = pa.py_buffer(bytes(validity)) if None in values else None
    offsets_buffer = pa.py_buffer(struct.pack(f"<{len(offsets)}i", *offsets))
    return pa.Array.from_buffers(
        pa.string(), len(values), [validity_buffer, offsets_buffer, pa.py_buffer(b"".join(chunks))]
    )


def array_from_values(values: Sequence[Any], arrow_type: pa.DataType) -> pa.Array:
    """Build a null-free ``int32``, ``int64`` or ``bool`` array, or a nullable ``utf8`` one, holding ``values``,
    equal to ``pa.array(values, type=arrow_type)`` but without pyarrow's pandas-importing sequence conversion."""
    if pa.types.is_string(arrow_type):
        return _utf8_array(values)
    if pa.types.is_boolean(arrow_type):
        bits = bytearray((len(values) + 7) // 8)
        for index, value in enumerate(values):
            if value:
                bits[index >> 3] |= 1 << (index & 7)
        data = bytes(bits)
    elif str(arrow_type) in _INT_FORMATS:
        data = struct.pack(f"<{len(values)}{_INT_FORMATS[str(arrow_type)]}", *values)
    else:
        raise ValueError(f"unsupported arrow type: {arrow_type}")
    return pa.Array.from_buffers(arrow_type, len(values), [None, pa.py_buffer(data)])
