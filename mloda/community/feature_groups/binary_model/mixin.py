"""``BinaryModelMixin``: the entry point a FeatureGroup mixes in to run an external binary as a
model over Arrow IPC (contract: Capabilities, Data, Configuration, License, Data handling,
Errors). Combines the building blocks from ``binary.py`` (resolution and probing) and
``transport.py`` (the private per-invocation directory and process transport) with the mixin's own
responsibility: every up-front rejection, projecting/casting/batching the outgoing data, and
verifying the binary's output against the contract before it reaches the caller.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.compute as pc

from mloda.community.feature_groups.binary_model.binary import COLUMN_TYPE_VOCABULARY, ResolvedBinary, resolve_binary
from mloda.community.feature_groups.binary_model.errors import (
    BinaryModelError,
    BinaryUsageError,
    DataError,
    OutputContractError,
    UnsupportedError,
)
from mloda.community.feature_groups.binary_model.transport import InvocationDirectory, minimal_environment, run_binary

logger = logging.getLogger(__name__)

_STRING_CELL_BYTE_LIMIT = 2**31 - 1


def max_string_length(column: pa.ChunkedArray) -> int:
    """Largest byte length of a string cell in ``column`` (0 for an all-null or empty column)."""
    source = column.cast(pa.large_string()) if pa.types.is_string_view(column.type) else column
    longest = pc.max(pc.binary_length(source)).as_py()
    return int(longest) if longest is not None else 0


def _classify_column_type(arrow_type: pa.DataType) -> str | None:
    """Map an Arrow type to the contract's column-type vocabulary (contract: Capabilities), or
    ``None`` if it falls outside it."""
    if pa.types.is_int64(arrow_type):
        name = "int64"
    elif pa.types.is_float64(arrow_type):
        name = "float64"
    elif pa.types.is_boolean(arrow_type):
        name = "boolean"
    elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type) or pa.types.is_string_view(arrow_type):
        name = "utf8"
    else:
        return None
    return name if name in COLUMN_TYPE_VOCABULARY else None


def _assert_parameters_mapping(parameters: Any) -> None:
    if not isinstance(parameters, Mapping) or not all(isinstance(key, str) for key in parameters):
        raise BinaryUsageError("parameters must be a mapping with str keys")


def _assert_output_columns_shape(output_columns: Any) -> None:
    if not isinstance(output_columns, Mapping) or not output_columns:
        raise BinaryUsageError("output_columns must be a non-empty mapping of str to str")
    for key, value in output_columns.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise BinaryUsageError("output_columns must be a non-empty mapping of str to str")


def _assert_no_duplicate_table_columns(table: pa.Table) -> None:
    if len(set(table.column_names)) != len(table.column_names):
        raise BinaryUsageError(f"table must not contain duplicate column names: {table.column_names}")


def _assert_input_columns(input_columns: Sequence[str], table: pa.Table) -> None:
    if not input_columns:
        raise BinaryUsageError("input_columns must name at least one column")
    if len(set(input_columns)) != len(input_columns):
        raise BinaryUsageError(f"input_columns must not contain duplicates: {list(input_columns)}")
    missing = [name for name in input_columns if name not in table.column_names]
    if missing:
        raise BinaryUsageError(f"input_columns not present in the table: {missing}")


def _assert_output_columns(output_columns: Mapping[str, str], table: pa.Table) -> None:
    written_names = list(output_columns.values())
    if len(set(written_names)) != len(written_names):
        raise BinaryUsageError(f"output_columns written names must be unique: {written_names}")
    colliding = set(written_names) & set(table.column_names)
    if colliding:
        raise BinaryUsageError(f"output_columns written names collide with existing table columns: {sorted(colliding)}")


def _check_input_column_types(table: pa.Table, input_columns: Sequence[str], resolved: ResolvedBinary) -> None:
    """Classify each input column, reject it if outside the vocabulary or outside this binary's own
    advertised ``column_types``, then reject an oversized string cell (contract: Capabilities,
    Data)."""
    for name in input_columns:
        arrow_type = table.schema.field(name).type
        vocabulary_name = _classify_column_type(arrow_type)
        if vocabulary_name is None:
            raise UnsupportedError(f"column {name!r} has an unsupported type: {arrow_type!r}")
        if vocabulary_name not in resolved.capabilities.column_types:
            raise UnsupportedError(
                f"column {name!r} is classified as {vocabulary_name!r}, which binary "
                f"{resolved.capabilities.plugin_id!r} does not advertise in its column_types"
            )
        if vocabulary_name == "utf8" and max_string_length(table.column(name)) >= _STRING_CELL_BYTE_LIMIT:
            raise DataError(f"column {name!r} contains a string cell at or above the 2 GiB limit")


def _build_outgoing_table(table: pa.Table, input_columns: Sequence[str]) -> pa.Table:
    """Project ``table`` to ``input_columns`` (in that order) and strip all schema- and
    field-level metadata (contract: Data). ``large_string``/``string_view`` columns keep their own
    type here; the ``utf8`` cast happens later, per batch, in ``_write_ipc_stream``."""
    projected = table.select(list(input_columns))
    fields = [pa.field(field.name, field.type, nullable=field.nullable) for field in projected.schema]
    return projected.cast(pa.schema(fields))


def _rows_per_batch(table: pa.Table, max_batch_bytes: int) -> int:
    num_rows: int = table.num_rows
    num_bytes: int = table.nbytes
    bytes_per_row = max(1, num_bytes // max(1, num_rows))
    return max(1, max_batch_bytes // bytes_per_row)


def _split_oversized_batch(batch: pa.RecordBatch, max_batch_bytes: int) -> list[pa.RecordBatch]:
    """Halve ``batch`` by rows, recursively, until every piece fits in ``max_batch_bytes`` or holds
    a single row (contract: Capabilities): the mean-based estimate in ``_rows_per_batch`` can badly
    underestimate a batch containing an outlier row, so this is the backstop that actually
    guarantees the limit."""
    if batch.num_rows <= 1 or batch.nbytes <= max_batch_bytes:
        return [batch]
    midpoint = batch.num_rows // 2
    left = batch.slice(0, midpoint)
    right = batch.slice(midpoint)
    return _split_oversized_batch(left, max_batch_bytes) + _split_oversized_batch(right, max_batch_bytes)


def _wire_field(field: pa.Field) -> pa.Field:
    """Map a ``large_string``/``string_view`` field to plain ``utf8`` for the wire schema, keeping
    its name and nullability and dropping any metadata (contract: Data); every other field passes
    through unchanged."""
    field_type = (
        pa.string() if pa.types.is_large_string(field.type) or pa.types.is_string_view(field.type) else field.type
    )
    return pa.field(field.name, field_type, nullable=field.nullable)


def _cast_batch_to_wire_schema(batch: pa.RecordBatch, wire_schema: pa.Schema) -> pa.RecordBatch:
    """Cast ``batch`` to ``wire_schema``, falling back to rebuilding it column by column if the
    installed pyarrow has no ``RecordBatch.cast``."""
    cast_method = getattr(batch, "cast", None)
    if cast_method is not None:
        result: pa.RecordBatch = cast_method(wire_schema)
        return result
    arrays = [batch.column(index).cast(field.type) for index, field in enumerate(wire_schema)]
    return pa.RecordBatch.from_arrays(arrays, schema=wire_schema)


def _write_ipc_stream(table: pa.Table, max_batch_bytes: int) -> bytes:
    """Write ``table`` to Arrow IPC stream bytes, batched small enough that no single array exceeds
    ``max_batch_bytes`` (contract: Capabilities); a zero-row table writes a schema-only stream. The
    ``large_string``/``string_view`` -> ``utf8`` cast happens here, per batch, after splitting on
    ``table``'s own, still-large-typed batches, since casting the whole table up front could
    overflow ``utf8``'s 32-bit offsets even though no individual cell is oversized."""
    wire_schema = pa.schema([_wire_field(field) for field in table.schema])
    rows_per_batch = _rows_per_batch(table, max_batch_bytes)
    buffer = io.BytesIO()
    with pa.ipc.new_stream(buffer, wire_schema) as writer:
        for batch in table.to_batches(max_chunksize=rows_per_batch):
            for piece in _split_oversized_batch(batch, max_batch_bytes):
                writer.write_batch(_cast_batch_to_wire_schema(piece, wire_schema))
    return buffer.getvalue()


def _parse_output_stream(data: bytes) -> pa.Table:
    try:
        return pa.ipc.open_stream(data).read_all()
    except (pa.ArrowException, ValueError, OSError) as exc:
        raise OutputContractError(f"binary output is not a valid Arrow IPC stream: {exc}") from exc


def _verify_output_contract(
    result: pa.Table, output_columns: Mapping[str, str], expected_rows: int, column_types: frozenset[str]
) -> None:
    """Verify the binary's output against the contract (contract: Data): no duplicate field names,
    the column-name set, every type in this binary's own advertised ``column_types``, and the row
    count, each reported by name only, never by value."""
    if len(result.column_names) != len(set(result.column_names)):
        raise OutputContractError(f"binary output contains duplicate column names: {result.column_names}")
    expected_names = set(output_columns.values())
    actual_names = set(result.column_names)
    if actual_names != expected_names:
        raise OutputContractError(
            f"binary output column names {sorted(actual_names)} do not match expected {sorted(expected_names)}"
        )
    for field in result.schema:
        vocabulary_name = _classify_column_type(field.type)
        if vocabulary_name is None or vocabulary_name not in column_types:
            raise OutputContractError(f"binary output column {field.name!r} has an unsupported type: {field.type!r}")
    if result.num_rows != expected_rows:
        raise OutputContractError(
            f"binary output row count {result.num_rows} does not match input row count {expected_rows}"
        )


def _finalize_output(result: pa.Table, original_table: pa.Table) -> pa.Table:
    """Strip metadata and cast ``utf8`` output columns to ``large_string`` when the caller's frame
    itself uses ``large_string`` (contract: Capabilities), keeping the output column order the
    binary wrote."""
    frame_uses_large_string = any(pa.types.is_large_string(field.type) for field in original_table.schema)
    fields: list[pa.Field] = []
    arrays: list[pa.ChunkedArray] = []
    for field in result.schema:
        column = result.column(field.name)
        field_type = field.type
        if (
            frame_uses_large_string
            and _classify_column_type(field_type) == "utf8"
            and not pa.types.is_large_string(field_type)
        ):
            column = column.cast(pa.large_string())
            field_type = pa.large_string()
        fields.append(pa.field(field.name, field_type, nullable=field.nullable))
        arrays.append(column)
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


class BinaryModelMixin:
    """Mixed into a FeatureGroup to run an external binary as a model over Arrow IPC (contract:
    Invocation, Capabilities, Data, Configuration, License, Data handling, Errors)."""

    BINARY_PLUGIN_ID: ClassVar[str]
    BINARY_COMMAND_OVERRIDE: ClassVar[Sequence[str] | str | None] = None
    LICENSE_FILE_OVERRIDE: ClassVar[str | None] = None
    LICENSE_KEY_OVERRIDE: ClassVar[str | None] = None
    BINARY_TIMEOUT_SECONDS: ClassVar[float | None] = 600.0
    FILE_TRANSPORT_THRESHOLD_BYTES: ClassVar[int] = 64 * 1024 * 1024
    MAX_BATCH_BYTES: ClassVar[int] = 1 << 30

    @classmethod
    def binary_environment(cls) -> dict[str, str]:
        """The minimal subprocess environment for this model's binary (contract: Data handling)."""
        return minimal_environment(license_file=cls.LICENSE_FILE_OVERRIDE, license_key=cls.LICENSE_KEY_OVERRIDE)

    @classmethod
    def resolved_binary(cls) -> ResolvedBinary:
        """Resolve and probe this model's binary (contract: Invocation, Capabilities)."""
        return resolve_binary(
            cls.BINARY_PLUGIN_ID,
            cls.BINARY_COMMAND_OVERRIDE,
            env=cls.binary_environment(),
            timeout=cls.BINARY_TIMEOUT_SECONDS,
        )

    @classmethod
    def run_binary_model(
        cls,
        table: pa.Table,
        input_columns: Sequence[str],
        operation: str,
        parameters: Mapping[str, Any],
        output_columns: Mapping[str, str],
    ) -> pa.Table:
        """Run ``operation`` on ``table`` through the resolved binary, returning a table of only the
        output columns, row-aligned to ``table`` (contract: Data, Configuration, Errors)."""
        resolved = cls.resolved_binary()
        _assert_parameters_mapping(parameters)
        _assert_output_columns_shape(output_columns)
        _assert_no_duplicate_table_columns(table)
        _assert_input_columns(input_columns, table)
        _assert_output_columns(output_columns, table)
        if operation not in resolved.capabilities.operations:
            raise UnsupportedError(
                f"binary {resolved.capabilities.plugin_id!r} does not support operation {operation!r}"
            )
        _check_input_column_types(table, input_columns, resolved)

        outgoing = _build_outgoing_table(table, input_columns)
        stream_bytes = _write_ipc_stream(outgoing, cls.MAX_BATCH_BYTES)
        config = {
            "input_columns": list(input_columns),
            "operation": operation,
            "parameters": dict(parameters),
            "output_columns": dict(output_columns),
        }

        with InvocationDirectory() as invocation:
            try:
                output_bytes = run_binary(
                    resolved.argv,
                    cls.binary_environment(),
                    config,
                    stream_bytes,
                    timeout=cls.BINARY_TIMEOUT_SECONDS,
                    file_transport_threshold=cls.FILE_TRANSPORT_THRESHOLD_BYTES,
                    invocation_dir=invocation.path,
                )
            except BinaryModelError as exc:
                logger.debug(
                    "binary %s version %s failed with code %s",
                    resolved.capabilities.plugin_id,
                    resolved.capabilities.version,
                    exc.code,
                )
                raise

        logger.debug(
            "binary %s version %s exited with code 0", resolved.capabilities.plugin_id, resolved.capabilities.version
        )

        result = _parse_output_stream(output_bytes)
        _verify_output_contract(result, output_columns, table.num_rows, resolved.capabilities.column_types)
        return _finalize_output(result, table)
