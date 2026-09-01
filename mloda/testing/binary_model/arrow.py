"""Arrow IPC stream mechanics for the binary-model conformance kit: building input streams (single
batch, multi-batch, schema-only, IPC file/Feather format), reading output back into a table, and
raw-bytes/raw-message helpers for conditions pyarrow's own reader can't distinguish (schema-only vs
zero-row batch) or works around (compressed bodies, corrupted messages after a valid schema).

The pytest-facing surface (fixtures, assertions, conformance-check classes) lives in
``conformance.py``, which imports and re-exports these helpers.
"""

from __future__ import annotations

import io
import struct
from typing import Any

import pyarrow as pa

from mloda.testing.binary_model import IPC_END_OF_STREAM_MARKER


def arrow_stream_bytes_from_arrays(
    schema: pa.Schema, arrays: list[pa.Array] | None, *, options: pa.ipc.IpcWriteOptions | None = None
) -> bytes:
    """Write a single record batch built from ``arrays`` (aligned to ``schema``'s fields in order)
    to Arrow IPC stream bytes, or a schema-only stream if ``arrays`` is ``None`` (contract: Data).
    Lower-level than ``arrow_stream_bytes``: accepts duplicate field names, which a
    column-name-keyed mapping cannot represent."""
    buf = io.BytesIO()
    with pa.ipc.new_stream(buf, schema, options=options) as writer:
        if arrays is not None:
            writer.write_batch(pa.record_batch(arrays, schema=schema))
    return buf.getvalue()


def arrow_stream_bytes(
    schema: pa.Schema, rows: dict[str, list[Any]] | None, *, options: pa.ipc.IpcWriteOptions | None = None
) -> bytes:
    """Write ``rows`` (column name -> one Python value per row) as a single record batch to Arrow
    IPC stream format bytes, or a schema-only stream if ``rows`` is ``None``."""
    arrays = None if rows is None else [pa.array(rows[field.name], type=field.type) for field in schema]
    return arrow_stream_bytes_from_arrays(schema, arrays, options=options)


def arrow_file_format_bytes(schema: pa.Schema, rows: dict[str, list[Any]]) -> bytes:
    """Write ``rows`` to the Arrow IPC *file*/Feather format (``ARROW1`` magic bytes), used to
    test rejection of that format on the streaming-only input contract (contract: Data)."""
    arrays = [pa.array(rows[field.name], type=field.type) for field in schema]
    buf = io.BytesIO()
    with pa.ipc.new_file(buf, schema) as writer:
        writer.write_batch(pa.record_batch(arrays, schema=schema))
    return buf.getvalue()


def arrow_stream_bytes_multi_batch(schema: pa.Schema, batches_rows: list[dict[str, list[Any]]]) -> bytes:
    """Write ``batches_rows`` (one column-name-keyed rows dict per batch, in order) as multiple
    record batches in a single Arrow IPC stream, so a binary must combine every batch, not just the
    first (contract: Data -- "batch boundaries may differ")."""
    buf = io.BytesIO()
    with pa.ipc.new_stream(buf, schema) as writer:
        for rows in batches_rows:
            arrays = [pa.array(rows[field.name], type=field.type) for field in schema]
            writer.write_batch(pa.record_batch(arrays, schema=schema))
    return buf.getvalue()


def read_arrow_stream(data: bytes) -> pa.Table:
    """Parse Arrow IPC stream bytes back into a table, for asserting on a binary's output
    (contract: Data)."""
    return pa.ipc.open_stream(data).read_all()


def assert_ends_with_ipc_eos_marker(data: bytes) -> None:
    """Assert the raw bytes end with the IPC end-of-stream marker: checked directly since pyarrow's
    own reader tolerates a stream missing it (contract: Data)."""
    tail = data[-len(IPC_END_OF_STREAM_MARKER) :]
    assert tail == IPC_END_OF_STREAM_MARKER, (
        f"expected output to end with the IPC end-of-stream marker {IPC_END_OF_STREAM_MARKER!r}, "
        f"got trailing bytes {tail!r} (total length {len(data)})"
    )


def enumerate_ipc_message_types(data: bytes) -> list[str]:
    """Enumerate an Arrow IPC stream's raw messages by type, in order (e.g. ``["schema"]`` vs
    ``["schema", "record batch"]``): distinguishes wire shapes that
    ``pa.ipc.open_stream(...).read_all()`` cannot, since both parse to the same zero-row table
    (contract: Data)."""
    message_reader = pa.ipc.MessageReader.open_stream(pa.py_buffer(data))
    types: list[str] = []
    while True:
        try:
            message = message_reader.read_next_message()
        except StopIteration:
            return types
        types.append(message.type)


def corrupt_record_batch_message_after_schema(data: bytes) -> bytes:
    """Corrupt the record-batch message following the schema in a single-batch stream so
    ``open_stream()`` still parses the schema but ``RecordBatchReader.read_all()`` fails: overwrites
    that message's metadata-length prefix (4 little-endian bytes after its ``0xFFFFFFFF``
    continuation marker) with an implausibly large value (contract: Data). Requires ``data`` to be
    exactly one schema message followed by one record batch message."""
    assert data[0:4] == b"\xff\xff\xff\xff", "expected data to start with the IPC continuation marker"
    schema_metadata_len = struct.unpack_from("<I", data, 4)[0]
    record_batch_message_start = 8 + schema_metadata_len
    assert data[record_batch_message_start : record_batch_message_start + 4] == b"\xff\xff\xff\xff", (
        "expected a second IPC message (record batch) immediately after the schema message"
    )
    corrupted = bytearray(data)
    struct.pack_into("<I", corrupted, record_batch_message_start + 4, 0x7FFFFFFF)
    return bytes(corrupted)
