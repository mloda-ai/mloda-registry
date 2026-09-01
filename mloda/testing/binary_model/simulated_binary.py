"""Simulated binary: a pure-Python CLI stand-in for a future Rust-compiled binary model.

Part of the ``mloda-testing`` package's conformance kit, so it can be run as
``python -m mloda.testing.binary_model.simulated_binary`` by both this repository's own tests and,
later, external consumers (a private wrapper repo, binary vendors, consumer FeatureGroup repos)
exercising ``mloda.testing.binary_model.conformance`` against their own binary: ``--version``,
``--capabilities``, ``run --config <path> [--input <path>] [--output <path>]``, the license gate,
config validation, and the Arrow IPC data path.
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from mloda.testing.binary_model import (
    COLUMN_TYPES,
    CONTRACT_VERSION,
    DATA_ERROR,
    INTERNAL_ERROR,
    LICENSE_INVALID,
    LICENSE_MISSING,
    MESSAGE_MAX_BYTES,
    UNSUPPORTED,
    USAGE_ERROR,
)
from mloda.testing.binary_model.hash_reference import compute_expected_hash

PLUGIN_ID = "example_binary"
VERSION = "1.0.0"
CAPABILITY_OPERATIONS = ["hash"]
RESERVED_INTERNAL_ERROR_OPERATION = "_conformance_internal_error"

# Outputs each operation defines, in the order they must be written (contract: Configuration).
_OPERATION_OUTPUTS: dict[str, tuple[str, ...]] = {"hash": ("result",)}

_REQUIRED_CONFIG_KEYS = frozenset({"input_columns", "operation", "parameters", "output_columns"})

# Continuation marker (0xFFFFFFFF) followed by a zero-length (0x00000000) message: the Arrow IPC
# end-of-stream marker (contract: Data). pyarrow's own stream reader tolerates a stream missing
# this, so it is checked on the raw trailing bytes instead (contract: Data, Conformance).
IPC_END_OF_STREAM_MARKER = b"\xff\xff\xff\xff\x00\x00\x00\x00"


def _classify_column_type(arrow_type: pa.DataType) -> str | None:
    """Map an Arrow type to this contract's type vocabulary, or ``None`` if it is outside it
    (contract: Capabilities). Only bare ``pa.string()`` counts as this contract's ``utf8``:
    ``pa.large_string()`` and ``pa.string_view()`` are distinct Arrow types outside the
    vocabulary, not accepted as an alias for it."""
    if pa.types.is_int64(arrow_type):
        return "int64"
    if pa.types.is_float64(arrow_type):
        return "float64"
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if pa.types.is_string(arrow_type):
        return "utf8"
    return None


# ---------------------------------------------------------------------------
# Compressed record batch detection.
#
# pyarrow's stream reader transparently decompresses lz4/zstd record batch bodies instead of
# rejecting them (verified against pyarrow 25), so a compressed body cannot be detected by reading
# the data through pyarrow. Instead this walks the Arrow IPC message's raw flatbuffer metadata by
# hand -- a stable, versioned wire format -- to see the ``RecordBatch.compression`` field, which
# the pyarrow Python API does not expose.
# ---------------------------------------------------------------------------

# org.apache.arrow.flatbuf.MessageHeader union tag for a RecordBatch message (stable wire format).
_MESSAGE_HEADER_RECORD_BATCH = 3


def _fb_bounds_check(buf: bytes, pos: int, width: int) -> None:
    """Guard against a negative or out-of-range computed flatbuffer offset. ``struct.unpack_from``
    accepts a negative offset (reading from the end of the buffer instead of raising), which is a
    latent correctness hazard even though pyarrow's own message verifier currently runs first and
    makes this unreachable in practice; raising here turns any future gap into a clean data error
    (code 5) instead of an uncaught ``struct.error`` (code 6)."""
    if not (0 <= pos <= len(buf) - width):
        raise _CliError(
            DATA_ERROR, "malformed input: flatbuffer offset out of range while scanning for compressed record batches"
        )


def _fb_u32(buf: bytes, pos: int) -> int:
    _fb_bounds_check(buf, pos, 4)
    value: int = struct.unpack_from("<I", buf, pos)[0]
    return value


def _fb_i32(buf: bytes, pos: int) -> int:
    _fb_bounds_check(buf, pos, 4)
    value: int = struct.unpack_from("<i", buf, pos)[0]
    return value


def _fb_u16(buf: bytes, pos: int) -> int:
    _fb_bounds_check(buf, pos, 2)
    value: int = struct.unpack_from("<H", buf, pos)[0]
    return value


def _fb_field_slot(buf: bytes, table_pos: int, field_index: int) -> int:
    """The vtable-declared byte offset (within the table at ``table_pos``) of ``field_index``, or
    0 if that field is absent. Flatbuffers wire format: a table's first 4 bytes are a signed
    offset back to its vtable; the vtable starts with its own size and the table's size (2 bytes
    each), followed by one 2-byte offset per declared field, in declaration order."""
    soffset = _fb_i32(buf, table_pos)
    vtable_pos = table_pos - soffset
    vtable_size = _fb_u16(buf, vtable_pos)
    slot = 4 + field_index * 2
    if slot >= vtable_size:
        return 0
    return _fb_u16(buf, vtable_pos + slot)


def _fb_offset_field(buf: bytes, table_pos: int, field_index: int) -> int | None:
    """Follow a table/vector/string offset field to its absolute position, or ``None`` if the
    field is absent."""
    slot = _fb_field_slot(buf, table_pos, field_index)
    if slot == 0:
        return None
    field_pos = table_pos + slot
    return field_pos + _fb_u32(buf, field_pos)


def _fb_scalar_u8_field(buf: bytes, table_pos: int, field_index: int) -> int:
    slot = _fb_field_slot(buf, table_pos, field_index)
    if slot == 0:
        return 0
    field_pos = table_pos + slot
    _fb_bounds_check(buf, field_pos, 1)
    return buf[field_pos]


def _message_metadata_is_compressed_record_batch(metadata: bytes) -> bool:
    """Whether an Arrow IPC message's raw flatbuffer metadata describes a compressed record batch:
    ``Message.header_type`` (field 1) must be RecordBatch; ``Message.header`` (field 2) then
    points at the RecordBatch table, whose ``compression`` field (field 3), if present, means the
    batch is compressed (contract: Data)."""
    buf = bytes(metadata)
    message_pos = _fb_u32(buf, 0)
    header_type = _fb_scalar_u8_field(buf, message_pos, 1)
    if header_type != _MESSAGE_HEADER_RECORD_BATCH:
        return False
    record_batch_pos = _fb_offset_field(buf, message_pos, 2)
    if record_batch_pos is None:
        return False
    return _fb_offset_field(buf, record_batch_pos, 3) is not None


def _truncate_message(text: str, max_bytes: int = MESSAGE_MAX_BYTES) -> str:
    """Bound ``text`` to at most ``max_bytes`` UTF-8 bytes, cutting only on a UTF-8 character
    boundary (contract: Data handling)."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    suffix = "...(truncated)"
    budget = max_bytes - len(suffix.encode("utf-8"))
    if budget <= 0:
        return encoded[:max_bytes].decode("utf-8", errors="ignore")
    return encoded[:budget].decode("utf-8", errors="ignore") + suffix


class _CliError(Exception):
    """Carries the exit code and stderr message for one contract error class. ``message`` is
    always run through ``_truncate_message`` so every error path respects the 1024-byte cap, since
    several messages interpolate caller-controlled data whose length is not otherwise bounded."""

    def __init__(self, code: int, message: str) -> None:
        message = _truncate_message(message)
        super().__init__(message)
        self.code = code
        self.message = message


class _ReservedConformanceInternalError(Exception):
    """Raised by the reserved conformance-only operation to deliberately reach ``main()``'s
    generic unexpected-exception catch-all, so the resulting stderr message is exactly
    ``f"internal error: {type(exc).__name__}"`` -- the same shape produced for any other genuinely
    unexpected exception -- rather than a hand-written message that bypasses that catch-all
    (contract: Data handling, Errors, Conformance)."""


@dataclass
class _RunArgs:
    config_path: Path
    input_path: Path | None
    output_path: Path | None


def _capabilities() -> dict[str, Any]:
    return {
        "contract": CONTRACT_VERSION,
        "plugin_id": PLUGIN_ID,
        "operations": list(CAPABILITY_OPERATIONS),
        "column_types": sorted(COLUMN_TYPES),
    }


def _parse_run_args(args: list[str]) -> _RunArgs:
    config_value: str | None = None
    input_value: str | None = None
    output_value: str | None = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--config", "--input", "--output") and i + 1 < len(args):
            i += 1
            if arg == "--config":
                config_value = args[i]
            elif arg == "--input":
                input_value = args[i]
            else:
                output_value = args[i]
        else:
            raise _CliError(USAGE_ERROR, f"unrecognized run argument: {arg!r}")
        i += 1

    if config_value is None:
        raise _CliError(USAGE_ERROR, "run requires --config <path>")
    config_path = Path(config_value)
    if not config_path.is_file() or not os.access(config_path, os.R_OK):
        raise _CliError(USAGE_ERROR, f"--config path does not exist or is not readable: {config_value}")

    return _RunArgs(
        config_path=config_path,
        input_path=Path(input_value) if input_value is not None else None,
        output_path=Path(output_value) if output_value is not None else None,
    )


def _license_source() -> tuple[str, str]:
    """Return (source_name, token_text), the first of MLODA_LICENSE_FILE / MLODA_LICENSE_KEY set
    to a non-empty value (contract: License). Raises code 2 if neither is usable."""
    file_value = os.environ.get("MLODA_LICENSE_FILE", "")
    if file_value:
        path = Path(file_value)
        if not path.is_file():
            raise _CliError(LICENSE_MISSING, f"MLODA_LICENSE_FILE {file_value}: not found")
        try:
            return "MLODA_LICENSE_FILE", path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # UnicodeDecodeError is a ValueError subclass, not an OSError, but a file whose content
            # cannot be decoded as UTF-8 is exactly "set but unusable" (contract: License), the same
            # class as any other unreadable-content failure -- code 3, not a bare uncaught crash.
            raise _CliError(LICENSE_INVALID, f"MLODA_LICENSE_FILE {file_value}: not readable ({exc})") from exc

    key_value = os.environ.get("MLODA_LICENSE_KEY", "")
    if key_value:
        return "MLODA_LICENSE_KEY", key_value

    raise _CliError(LICENSE_MISSING, "no license source set (MLODA_LICENSE_FILE or MLODA_LICENSE_KEY)")


def _check_license() -> None:
    source_name, text = _license_source()
    text = text.strip()
    try:
        token = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _CliError(LICENSE_INVALID, f"{source_name}: license token is not valid JSON") from exc
    if not isinstance(token, dict):
        raise _CliError(LICENSE_INVALID, f"{source_name}: license token is not a JSON object")

    status = token.get("status")
    plugins = token.get("plugins")
    if status is None or plugins is None:
        raise _CliError(LICENSE_INVALID, f"{source_name}: license token missing 'status' or 'plugins'")
    if status != "valid":
        raise _CliError(LICENSE_INVALID, f"{source_name}: license {status}")
    if not isinstance(plugins, list) or PLUGIN_ID not in plugins:
        raise _CliError(LICENSE_INVALID, f"{source_name}: plugin_id {PLUGIN_ID!r} not entitled")


def _load_config(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # UnicodeDecodeError is a ValueError subclass, not an OSError, but decoding failure is a
        # config parse failure just like malformed JSON: a usage error, not an uncaught crash
        # (contract: Errors -- "config parse and structural validation ...; code 1").
        raise _CliError(USAGE_ERROR, f"--config not readable: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _CliError(USAGE_ERROR, f"--config is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _CliError(USAGE_ERROR, "--config must be a JSON object")
    return data


def _validate_config_structure(config: dict[str, Any]) -> None:
    """Required/unknown keys, input_columns rules, output_columns uniqueness and collision with
    input_columns -- everything checked "with the document", before the operation is consulted
    (contract: Configuration, Errors)."""
    unknown = set(config) - _REQUIRED_CONFIG_KEYS
    if unknown:
        raise _CliError(USAGE_ERROR, f"unknown config keys: {sorted(unknown)}")
    missing = _REQUIRED_CONFIG_KEYS - set(config)
    if missing:
        raise _CliError(USAGE_ERROR, f"missing config keys: {sorted(missing)}")

    input_columns = config["input_columns"]
    if not isinstance(input_columns, list) or not all(isinstance(c, str) for c in input_columns):
        raise _CliError(USAGE_ERROR, "input_columns must be a list of strings")
    if not input_columns:
        raise _CliError(USAGE_ERROR, "input_columns must name at least one column")
    if len(set(input_columns)) != len(input_columns):
        raise _CliError(USAGE_ERROR, "input_columns must not contain duplicates")

    if not isinstance(config["operation"], str):
        raise _CliError(USAGE_ERROR, "operation must be a string")
    if not isinstance(config["parameters"], dict):
        raise _CliError(USAGE_ERROR, "parameters must be an object")

    output_columns = config["output_columns"]
    if not isinstance(output_columns, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in output_columns.items()
    ):
        raise _CliError(USAGE_ERROR, "output_columns must be an object of string to string")
    written_names = list(output_columns.values())
    if len(set(written_names)) != len(written_names):
        raise _CliError(USAGE_ERROR, "output_columns written names must be unique")
    if set(written_names) & set(input_columns):
        raise _CliError(USAGE_ERROR, "output_columns written names must not collide with input_columns")


def _check_operation_capability(operation: str) -> None:
    if operation == RESERVED_INTERNAL_ERROR_OPERATION:
        return
    if operation not in CAPABILITY_OPERATIONS:
        raise _CliError(UNSUPPORTED, f"unsupported operation: {operation!r}")


def _validate_hash_parameters(operation: str, parameters: dict[str, Any]) -> None:
    """ "hash"-specific ``parameters`` validation (contract: Configuration, Errors): an operation
    validates its own parameter shape exhaustively, not just the keys it recognizes, so any other
    key present is rejected too. The reserved conformance-only operation has no defined parameter
    shape and bypasses this, same as the operation-output checks."""
    if operation != "hash":
        return
    unknown = set(parameters) - {"key"}
    if unknown:
        raise _CliError(USAGE_ERROR, f"unknown hash parameters: {sorted(unknown)}")
    key = parameters.get("key")
    if key is not None and not isinstance(key, str):
        raise _CliError(USAGE_ERROR, "parameters.key must be a string")


def _check_output_columns_completeness(operation: str, output_columns: dict[str, str]) -> None:
    """The reserved conformance operation bypasses this too: it has no defined output list, and
    its purpose is to reach the data stage regardless of what output_columns the kit supplies."""
    if operation == RESERVED_INTERNAL_ERROR_OPERATION:
        return
    expected = set(_OPERATION_OUTPUTS.get(operation, ()))
    actual = set(output_columns)
    if actual != expected:
        raise _CliError(
            USAGE_ERROR,
            f"output_columns must map exactly the operation's outputs {sorted(expected)}, got {sorted(actual)}",
        )


def _open_input_output(input_path: Path | None, output_path: Path | None) -> None:
    """Eagerly opens ``--input`` for reading and ``--output`` for writing, so a permissions problem
    or an ``--output`` naming an existing directory fails here, as a usage error (contract: Errors
    -- "opening --input and creating --output (code 1 if either fails)"), rather than surfacing
    later as an uncaught exception at read/write time. Opening ``--output`` for writing also
    truncates an existing file immediately, which is the contract's "an existing --output file is
    overwritten by the binary" (Data handling)."""
    if input_path is not None:
        if not input_path.is_file():
            raise _CliError(USAGE_ERROR, f"--input path does not exist: {input_path}")
        try:
            with input_path.open("rb"):
                pass
        except OSError as exc:
            raise _CliError(USAGE_ERROR, f"--input path is not readable: {exc}") from exc
    if output_path is not None:
        try:
            with output_path.open("wb"):
                pass
        except OSError as exc:
            raise _CliError(USAGE_ERROR, f"--output path is not creatable: {exc}") from exc


def _read_input_bytes(input_path: Path | None) -> bytes:
    if input_path is not None:
        return input_path.read_bytes()
    return sys.stdin.buffer.read()


def _open_ipc_stream_reader(source: pa.BufferReader) -> pa.RecordBatchReader:
    """Parse ``source`` (a position-tracking ``pa.BufferReader`` over the raw input bytes) as an
    Arrow IPC *stream* (not file/Feather format). Anything that is not a well-formed IPC stream
    (zero bytes handled by the caller; garbage bytes; the IPC file/Feather format's ``ARROW1``
    magic) fails pyarrow's own parsing, which is reported as a data error (contract: Data). A
    ``pa.BufferReader`` is used (rather than passing ``raw`` bytes directly) so its ``.tell()``
    afterward reports how many bytes the one logical stream actually consumed, which
    ``_assert_no_trailing_data`` below needs to reject a second stream or garbage concatenated
    after a complete one."""
    try:
        return pa.ipc.open_stream(source)
    except pa.ArrowException as exc:
        raise _CliError(DATA_ERROR, f"input is not a valid Arrow IPC stream: {exc}") from exc


def _read_all_batches(reader: pa.RecordBatchReader) -> pa.Table:
    """Malformed record-batch data appearing after a valid schema message (the schema parses, but
    the batch body does not) fails here rather than at ``open_stream()``, and must be reported the
    same way: a data error (contract: Data)."""
    try:
        return reader.read_all()
    except pa.ArrowException as exc:
        raise _CliError(DATA_ERROR, f"malformed record batch data: {exc}") from exc


def _assert_no_trailing_data(buffer_reader: pa.BufferReader, raw: bytes) -> None:
    """The one logical IPC stream must consume the whole input buffer; anything left over (a
    second, concatenated stream, or trailing garbage after a complete stream's own end-of-stream
    marker) is a data error (contract: Data). This is distinct from, and does not replace,
    ``_assert_ends_with_eos_marker``'s truncation check: a truncated buffer's reader position also
    ends up at ``len(raw)`` (via physical end of file rather than a formal end-of-stream token), so
    that failure mode is caught by the marker check instead, before this one ever runs."""
    if buffer_reader.tell() != len(raw):
        raise _CliError(DATA_ERROR, "trailing data after the end of the Arrow IPC stream")


def _assert_ends_with_eos_marker(raw: bytes) -> None:
    """ "truncated" means end of file without the end-of-stream marker, not "no more batches"
    (contract: Data); checked on the raw trailing bytes since pyarrow's own reader accepts a
    stream without it."""
    if raw[-len(IPC_END_OF_STREAM_MARKER) :] != IPC_END_OF_STREAM_MARKER:
        raise _CliError(DATA_ERROR, "input stream is truncated: missing the end-of-stream marker")


def _assert_no_compressed_record_batch(raw: bytes) -> None:
    """Malformed message metadata encountered while sweeping the raw bytes for a compressed
    record-batch body must be reported the same way as any other malformed record-batch data: a
    data error (contract: Data), not an uncaught ``pa.ArrowException``."""
    try:
        message_reader = pa.ipc.MessageReader.open_stream(pa.py_buffer(raw))
        while True:
            try:
                message = message_reader.read_next_message()
            except StopIteration:
                return
            if message.type == "record batch" and _message_metadata_is_compressed_record_batch(message.metadata):
                raise _CliError(DATA_ERROR, "input contains a compressed record batch body, which is not supported")
    except pa.ArrowException as exc:
        raise _CliError(DATA_ERROR, f"malformed input while scanning for compressed record batches: {exc}") from exc


def _validate_input_schema(schema: pa.Schema, input_columns: list[str]) -> None:
    """The stream's schema must contain exactly ``input_columns``, in any order, without
    duplicates (data error); each field's type must then be from the vocabulary (unsupported
    error); presence errors precede type errors (contract: Data)."""
    names = list(schema.names)
    if len(set(names)) != len(names):
        raise _CliError(DATA_ERROR, f"input schema has duplicate field names: {names}")
    if set(names) != set(input_columns):
        raise _CliError(
            DATA_ERROR,
            f"input schema must contain exactly input_columns {sorted(input_columns)}, got {sorted(names)}",
        )
    for field in schema:
        if _classify_column_type(field.type) is None:
            raise _CliError(UNSUPPORTED, f"column {field.name!r} has an unsupported type: {field.type!r}")


def _compute_hash_output(table: pa.Table, config: dict[str, Any]) -> tuple[pa.Schema, list[pa.Array]]:
    """Run the "hash" operation row by row, reading each row's values in ``input_columns`` order
    (the operation's input contract, not stream field order), and produce the single "result"
    output column under its configured written name, typed int64 (contract: Data, Configuration).
    Uses ``compute_expected_hash`` -- the same reference algorithm the conformance kit's
    ``HashOperationConformanceMixin`` computes its expected values with -- rather than a second,
    hand-written copy that could drift from it. A fresh output schema/table is built from scratch
    here, so none of the input's Arrow metadata (schema- or field-level) is ever propagated to the
    output (contract: Data)."""
    input_columns = config["input_columns"]
    key: str | None = config["parameters"].get("key")
    written_name = config["output_columns"]["result"]
    columns = {name: table.column(name).to_pylist() for name in input_columns}
    values = [
        compute_expected_hash(key, [columns[name][row_index] for name in input_columns])
        for row_index in range(table.num_rows)
    ]
    output_schema = pa.schema([pa.field(written_name, pa.int64())])
    return output_schema, [pa.array(values, type=pa.int64())]


def _build_ipc_stream_bytes(schema: pa.Schema, arrays: list[pa.Array]) -> bytes:
    """Write ``arrays`` (aligned to ``schema``) to Arrow IPC stream format bytes, ending with the
    end-of-stream marker (``pa.ipc.new_stream``'s context manager writes it on close). Zero rows
    produces a schema-only stream (schema message + end-of-stream marker, no record-batch message
    at all): a caller's zero-row output must not carry a record-batch message with zero rows
    (contract: Data -- "a schema-only input ... yields a schema-only output")."""
    num_rows = len(arrays[0]) if arrays else 0
    buf = io.BytesIO()
    with pa.ipc.new_stream(buf, schema) as writer:
        if num_rows > 0:
            writer.write_batch(pa.record_batch(arrays, schema=schema))
    return buf.getvalue()


def _write_output_bytes(data: bytes, output_path: Path | None) -> None:
    """Shared write path for both transports (contract: Invocation), so stdin/stdout and
    ``--input``/``--output`` behave identically."""
    if output_path is not None:
        output_path.write_bytes(data)
        return
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _run_data_stage(raw: bytes, config: dict[str, Any], output_path: Path | None) -> None:
    """Parse the input as an Arrow IPC stream and run the configured operation, writing the
    result to stdout or ``--output`` (contract: Data).

    Note: reads the whole input into memory before writing any output; a real Rust binary should
    stream batches instead (contract: Invocation), but for this small stub that simplicity is
    acceptable.
    """
    if len(raw) == 0:
        raise _CliError(DATA_ERROR, "input is zero bytes, not an Arrow IPC stream")

    operation = config["operation"]
    if operation == RESERVED_INTERNAL_ERROR_OPERATION:
        # Reserved conformance-only operation: reaches the data stage and deliberately fails via a
        # genuine, distinctly-named exception, letting it fall through to main()'s generic
        # catch-all -- the same path any other unexpected exception takes -- so the kit can
        # provoke code 6 on demand with that catch-all's own message shape (contract: Conformance).
        raise _ReservedConformanceInternalError("deliberate internal error for conformance testing")

    buffer_reader = pa.BufferReader(pa.py_buffer(raw))
    reader = _open_ipc_stream_reader(buffer_reader)
    _assert_ends_with_eos_marker(raw)
    _validate_input_schema(reader.schema, config["input_columns"])
    _assert_no_compressed_record_batch(raw)
    table = _read_all_batches(reader)
    _assert_no_trailing_data(buffer_reader, raw)

    output_schema, output_arrays = _compute_hash_output(table, config)
    _write_output_bytes(_build_ipc_stream_bytes(output_schema, output_arrays), output_path)


def _run_command(args: list[str]) -> int:
    run_args = _parse_run_args(args)
    _check_license()
    config = _load_config(run_args.config_path)
    _validate_config_structure(config)
    operation = config["operation"]
    _check_operation_capability(operation)
    _validate_hash_parameters(operation, config["parameters"])
    _check_output_columns_completeness(operation, config["output_columns"])
    _open_input_output(run_args.input_path, run_args.output_path)
    raw = _read_input_bytes(run_args.input_path)
    _run_data_stage(raw, config, run_args.output_path)
    return 0


def _dispatch(argv: list[str]) -> int:
    if argv == ["--version"]:
        print(f"{PLUGIN_ID} {VERSION}")
        return 0
    if argv == ["--capabilities"]:
        print(json.dumps(_capabilities()))
        return 0
    if argv and argv[0] == "run":
        return _run_command(argv[1:])
    raise _CliError(
        USAGE_ERROR, "usage: --version | --capabilities | run --config <path> [--input <path>] [--output <path>]"
    )


def _emit_error(code: int, message: str) -> None:
    print(json.dumps({"code": code, "message": message}), file=sys.stderr)


def _reconfigure_streams_utf8_lf() -> None:
    """Everything textual the binary writes (stderr; stdout for --version/--capabilities) is
    UTF-8 with `\\n` line endings, regardless of platform locale; the binary configures its own
    streams for that and never depends on the locale it inherits (contract: Invocation). Guarded
    with a broad ``except`` since ``sys.stdout``/``sys.stderr`` may have been replaced by something
    that is not a real ``TextIOWrapper`` (e.g. under a test runner's own capture) and therefore has
    no ``reconfigure`` method at all."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", newline="\n")
        except (AttributeError, ValueError, OSError):
            pass


def main() -> int:
    _reconfigure_streams_utf8_lf()
    try:
        return _dispatch(sys.argv[1:])
    except _CliError as exc:
        _emit_error(exc.code, exc.message)
        return exc.code
    except Exception as exc:
        # Only the exception's class name is reported, never its full string representation
        # (``str(exc)``), which is not bounded by design for an arbitrary, unexpected exception and
        # could otherwise leak caller data (contract: Data handling).
        _emit_error(INTERNAL_ERROR, _truncate_message(f"internal error: {type(exc).__name__}"))
        return INTERNAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
