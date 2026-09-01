"""Arrow IPC stream mechanics for the binary-model conformance kit.

Building input streams (single batch, multi-batch, schema-only, or the IPC file/Feather format
used to test its rejection), reading a binary's Arrow IPC stream output back into a table, and the
lower-level raw-bytes / raw-message-metadata helpers needed to test conditions pyarrow's own reader
either cannot distinguish (a schema-only stream vs. a zero-row record-batch message) or actively
works around (a compressed record-batch body, or a schema message immediately followed by a
corrupted record-batch message). Genuinely Arrow-IPC-specific mechanics; the pytest-facing surface
(fixtures, assertions, the conformance-check classes) lives in ``conformance.py``, which imports
and re-exports the helpers below.
"""

from __future__ import annotations

import io
import struct
from typing import Any

import pyarrow as pa

# Continuation marker (0xFFFFFFFF) followed by a zero-length (0x00000000) message: the Arrow IPC
# end-of-stream marker (contract: Data). pyarrow's own stream reader tolerates a stream missing
# this, so it is checked on the raw trailing bytes instead (contract: Data, Conformance).
IPC_END_OF_STREAM_MARKER = b"\xff\xff\xff\xff\x00\x00\x00\x00"


def arrow_stream_bytes_from_arrays(
    schema: pa.Schema, arrays: list[pa.Array] | None, *, options: pa.ipc.IpcWriteOptions | None = None
) -> bytes:
    """Write a single record batch built from ``arrays`` (aligned to ``schema``'s fields, in
    order) to Arrow IPC *stream* format bytes, or a schema-only stream (zero record batches, then
    the end-of-stream marker) if ``arrays`` is ``None`` (contract: Data). Lower-level than
    ``arrow_stream_bytes`` below: it accepts a schema with duplicate field names, which a
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
    """Write ``batches_rows`` (one column-name-keyed rows dict per record batch, in order) as
    multiple record batches within a single Arrow IPC stream, for testing that a binary combines
    every batch rather than only the first (contract: Data -- "batch boundaries may differ" is a
    normal, valid shape on both sides of the wire)."""
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
    """Assert the raw bytes end with the IPC end-of-stream marker, checked on the raw trailing
    bytes rather than through pyarrow's own reader, which tolerates a stream missing it (contract:
    Data)."""
    tail = data[-len(IPC_END_OF_STREAM_MARKER) :]
    assert tail == IPC_END_OF_STREAM_MARKER, (
        f"expected output to end with the IPC end-of-stream marker {IPC_END_OF_STREAM_MARKER!r}, "
        f"got trailing bytes {tail!r} (total length {len(data)})"
    )


def enumerate_ipc_message_types(data: bytes) -> list[str]:
    """Enumerate an Arrow IPC stream's raw messages by type, in order (e.g. ``["schema"]`` for a
    schema-only stream with no record-batch message at all, vs ``["schema", "record batch"]``).
    Used to assert on the exact wire *shape* of a stream, which
    ``pa.ipc.open_stream(...).read_all()`` cannot distinguish: both shapes above parse to the same
    zero-row table (contract: Data)."""
    message_reader = pa.ipc.MessageReader.open_stream(pa.py_buffer(data))
    types: list[str] = []
    while True:
        try:
            message = message_reader.read_next_message()
        except StopIteration:
            return types
        types.append(message.type)


def corrupt_record_batch_message_after_schema(data: bytes) -> bytes:
    """Corrupt the second Arrow IPC message (the record batch immediately following the schema
    message) of a single-batch stream so that the schema-parsing step
    (``pa.ipc.open_stream(...)``) still succeeds, but reading the record batch body
    (``RecordBatchReader.read_all()``) fails -- distinct from "malformed bytes from the very
    start", which fails at ``open_stream()`` itself. Overwrites the record batch message's
    metadata-length prefix (the 4 little-endian bytes right after that message's continuation
    marker, ``0xFFFFFFFF``) with an implausibly large value, so pyarrow reports a metadata-length
    mismatch once it tries to read the batch (contract: Data). Requires ``data`` to be a
    single-batch stream (a schema message followed by exactly one record batch message)."""
    assert data[0:4] == b"\xff\xff\xff\xff", "expected data to start with the IPC continuation marker"
    schema_metadata_len = struct.unpack_from("<I", data, 4)[0]
    record_batch_message_start = 8 + schema_metadata_len
    assert data[record_batch_message_start : record_batch_message_start + 4] == b"\xff\xff\xff\xff", (
        "expected a second IPC message (record batch) immediately after the schema message"
    )
    corrupted = bytearray(data)
    struct.pack_into("<I", corrupted, record_batch_message_start + 4, 0x7FFFFFFF)
    return bytes(corrupted)
