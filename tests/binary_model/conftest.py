"""Shared fixtures and helpers for the binary-model conformance kit.

``simulated_binary.py`` is a pure-Python CLI stand-in for a future Rust-compiled binary model; a
private test fixture, never published as a package. ``conformance_kit.py``'s test functions are
meant to run, unmodified, against a real binary later by overriding the ``binary_cmd`` fixture
below, so nothing here may hardcode a binary path. The concrete ``plugin_id`` and operation used
throughout ("example_binary" / "hash") are the contract's own worked example, not a scope
limitation of the contract itself.

License tokens: the real signed-token format does not exist yet, so the contract has the binary
accept a fixed placeholder format for now: plain JSON text (not signed) shaped as
``{"status": "valid" | "expired", "plugins": [<plugin_id>, ...]}``. The helpers below build every
state the contract's License section distinguishes: missing, valid, expired, wrong-plugin, and
tampered (unparseable text, or valid JSON missing a required key).

"hash" operation: ``compute_expected_hash`` below is an independent reference implementation using
Python's ``hashlib``, so a test's expected value is never derived from whatever the binary happens
to do.

Also covers the contract's "Data handling" section and the remaining stderr/diagnostics rules from
"Errors": no network, no files outside ``--config``/``--input``/``MLODA_LICENSE_FILE``/
``--output``, the minimal environment, data-free diagnostics, and the stderr/message size caps
(the latter folded into ``assert_error_response`` itself, so every error-path test that uses it
gets this check for free).
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import subprocess  # nosec
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

# ---------------------------------------------------------------------------
# Design parameters (contract vocabulary + this kit's concrete worked example)
# ---------------------------------------------------------------------------

PLUGIN_ID = "example_binary"
WRONG_PLUGIN_ID = "some_other_plugin"
CONTRACT_VERSION = 1
CAPABILITY_OPERATIONS = ["hash"]
RESERVED_INTERNAL_ERROR_OPERATION = "_conformance_internal_error"
COLUMN_TYPES = frozenset({"int64", "float64", "utf8", "boolean"})

SIMULATED_BINARY_PATH = Path(__file__).resolve().parent / "simulated_binary.py"

# Contract "Errors" table.
USAGE_ERROR = 1
LICENSE_MISSING = 2
LICENSE_INVALID = 3
UNSUPPORTED = 4
DATA_ERROR = 5
INTERNAL_ERROR = 6

# A distinctive marker string, chosen so it would never otherwise appear in this suite's input,
# output or diagnostics, used to test that a marked input cell's value never leaks into stderr
# (contract: Data handling).
DATA_FREE_MARKER = "SECRET_MARKER_YlZ9qX7"

# The contract's stderr/message size caps (contract: Data handling). Enforced inside
# ``assert_error_response`` below so every error-path test that uses it gets this check for free.
MESSAGE_MAX_BYTES = 1024
STDERR_SOFT_CAP_BYTES = 64 * 1024


def _license_token_text(status: str, plugins: list[str]) -> str:
    return json.dumps({"status": status, "plugins": plugins})


VALID_LICENSE_TEXT = _license_token_text("valid", [PLUGIN_ID])
EXPIRED_LICENSE_TEXT = _license_token_text("expired", [PLUGIN_ID])
WRONG_PLUGIN_LICENSE_TEXT = _license_token_text("valid", [WRONG_PLUGIN_ID])
TAMPERED_UNPARSEABLE_TEXT = "{this is not json"
TAMPERED_MISSING_STATUS_TEXT = json.dumps({"plugins": [PLUGIN_ID]})
TAMPERED_MISSING_PLUGINS_TEXT = json.dumps({"status": "valid"})


# ---------------------------------------------------------------------------
# Plain helpers (importable directly by conformance_kit.py, not pytest fixtures)
# ---------------------------------------------------------------------------


def write_text(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` and return the path, for config/license file fixtures."""
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, data: dict[str, Any]) -> Path:
    """Write ``data`` as JSON text to ``path`` and return the path."""
    return write_text(path, json.dumps(data))


def make_config(
    *,
    input_columns: list[str] | None = None,
    operation: str | None = None,
    parameters: dict[str, Any] | None = None,
    output_columns: dict[str, str] | None = None,
) -> dict[str, Any]:
    """A structurally valid ``--config`` document for the "hash" operation, with every field
    overridable so a test can mutate exactly the one field under test."""
    return {
        "input_columns": ["col_a"] if input_columns is None else input_columns,
        "operation": CAPABILITY_OPERATIONS[0] if operation is None else operation,
        "parameters": {} if parameters is None else parameters,
        "output_columns": {"result": "col_a_hash"} if output_columns is None else output_columns,
    }


def run_binary(
    cmd: list[str],
    args: list[str],
    env: dict[str, str],
    input_bytes: bytes = b"",
    timeout: float = 10.0,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Invoke the binary with a fully-controlled argv and environment.

    ``env`` replaces the child's environment outright (never merged with the ambient one), so
    tests are hermetic and never accidentally inherit a license from the calling shell. ``stdin``
    always gets an explicit (possibly empty) byte string so the process never blocks the test
    suite waiting on an open pipe. ``cwd``, if given, is the child's working directory (used by
    the read-only-cwd check, contract: Data handling); the caller's own cwd otherwise.
    """
    return subprocess.run(  # nosec B603
        [*cmd, *args],
        env=dict(env),
        input=input_bytes,
        capture_output=True,
        timeout=timeout,
        cwd=cwd,
    )


def stderr_error_object(stderr: bytes) -> dict[str, Any]:
    """Parse the last non-empty line of stderr as the contract's ``{"code": ..., "message": ...}``
    object; earlier lines, if any, are free-form diagnostics (contract: Errors)."""
    text = stderr.decode("utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines, f"expected at least one non-empty stderr line, got {stderr!r}"
    obj = json.loads(lines[-1])
    assert isinstance(obj, dict), f"last non-empty stderr line is not a JSON object: {lines[-1]!r}"
    return obj


def assert_error_response(result: subprocess.CompletedProcess[bytes], expected_code: int) -> dict[str, Any]:
    """Assert the process exited with ``expected_code`` and that stderr's last non-empty line is
    exactly one parseable JSON object whose ``code`` matches, with a non-empty ``message`` string
    (contract: Errors). Also asserts the contract's Data handling size caps -- ``message`` at most
    ``MESSAGE_MAX_BYTES`` (1024) UTF-8 bytes, total stderr at most ``STDERR_SOFT_CAP_BYTES``
    (64 KiB) -- so every error-path test that goes through this helper checks the caps for free.
    Returns the parsed error object for further assertions."""
    assert result.returncode == expected_code, (
        f"expected exit code {expected_code}, got {result.returncode}; stderr={result.stderr!r}"
    )
    error = stderr_error_object(result.stderr)
    assert error.get("code") == expected_code, f"error object code mismatch: {error!r}"
    message = error.get("message")
    assert isinstance(message, str) and message, f"error object missing a non-empty message: {error!r}"
    message_bytes = len(message.encode("utf-8"))
    assert message_bytes <= MESSAGE_MAX_BYTES, (
        f"error message exceeds the {MESSAGE_MAX_BYTES}-byte cap: {message_bytes} bytes ({message[:200]!r}...)"
    )
    assert len(result.stderr) <= STDERR_SOFT_CAP_BYTES, (
        f"stderr exceeds the {STDERR_SOFT_CAP_BYTES}-byte soft cap: {len(result.stderr)} bytes"
    )
    return error


def assert_not_rejected_with(result: subprocess.CompletedProcess[bytes], forbidden_codes: set[int]) -> None:
    """Assert the process did not exit with any of ``forbidden_codes``, without asserting what a
    successful outcome looks like. A non-zero exit for some other reason must still satisfy the
    stderr JSON format rule (contract: Errors)."""
    assert result.returncode not in forbidden_codes, (
        f"unexpectedly rejected with code {result.returncode}; stderr={result.stderr!r}"
    )
    if result.returncode != 0:
        error = stderr_error_object(result.stderr)
        assert error.get("code") == result.returncode, f"error object code mismatch: {error!r}"
        message = error.get("message")
        assert isinstance(message, str) and message, f"error object missing a non-empty message: {error!r}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_cmd() -> list[str]:
    """Invocation prefix for our own simulated binary. A future conformance run against a real
    binary (the wrapper, or the end-to-end run) overrides this fixture to point at that
    executable instead, reusing every test in ``conformance_kit.py`` unmodified."""
    return [sys.executable, str(SIMULATED_BINARY_PATH)]


@pytest.fixture
def hermetic_env() -> dict[str, str]:
    """A minimal, controlled environment: no ambient shell variables, no license variables."""
    return {}


@pytest.fixture
def valid_config_dict() -> dict[str, Any]:
    return make_config()


@pytest.fixture
def valid_config_path(tmp_path: Path, valid_config_dict: dict[str, Any]) -> Path:
    """A structurally valid ``--config`` file, used to isolate the license gate and the
    invocation surface from config validation itself."""
    return write_json(tmp_path / "config.json", valid_config_dict)


@pytest.fixture
def valid_license_file(tmp_path: Path) -> Path:
    return write_text(tmp_path / "license.txt", VALID_LICENSE_TEXT)


@pytest.fixture
def valid_license_env(valid_license_file: Path) -> dict[str, str]:
    """``MLODA_LICENSE_FILE`` pointed at a valid token; used by config-validation tests to
    isolate config behaviour from the license gate."""
    return {"MLODA_LICENSE_FILE": str(valid_license_file)}


# ---------------------------------------------------------------------------
# Arrow IPC data pipeline -- design parameters, the "hash" algorithm, and helpers
# ---------------------------------------------------------------------------

# The vocabulary's Arrow types (contract: Capabilities). ``utf8`` is pyarrow's 32-bit-offset
# string type, ``pa.string()`` -- not ``pa.large_string()`` or ``pa.string_view()``.
COLUMN_TYPE_TO_ARROW: dict[str, pa.DataType] = {
    "int64": pa.int64(),
    "float64": pa.float64(),
    "utf8": pa.string(),
    "boolean": pa.bool_(),
}

# Continuation marker (0xFFFFFFFF) followed by a zero-length (0x00000000) message: the Arrow IPC
# end-of-stream marker (contract: Data).
IPC_END_OF_STREAM_MARKER = b"\xff\xff\xff\xff\x00\x00\x00\x00"

# Fixed sentinel/delimiter for the "hash" reference algorithm below. The sentinel embeds NUL bytes
# so it can never collide with a real utf8 value a caller could plausibly send; the delimiter is
# the ASCII unit separator (0x1F), likewise not expected in ordinary text input.
HASH_NULL_SENTINEL = "\x00__NULL__\x00"
HASH_FIELD_DELIMITER = "\x1f"


def _hash_value_token(value: Any) -> str:
    """One row value's token in the "hash" reference algorithm (see ``compute_expected_hash``).
    Order matters: ``bool`` is checked before ``int`` since ``bool`` is an ``int`` subclass in
    Python."""
    if value is None:
        return HASH_NULL_SENTINEL
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
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
       - float64 -> ``repr(value)`` (Python's shortest round-tripping representation);
       - utf8 -> the string value, unmodified.
    2. Join the row's tokens with ``HASH_FIELD_DELIMITER``.
    3. Prepend ``key`` (``parameters.key``, or the empty string if the operation was invoked
       without one) followed by one more ``HASH_FIELD_DELIMITER``.
    4. UTF-8 encode the resulting string and hash it with BLAKE2b, ``digest_size=8`` (a 64-bit /
       8-byte digest).
    5. Interpret the 8 raw digest bytes as a big-endian *signed* 64-bit integer
       (``struct.unpack(">q", digest)[0]``); this is the row's single "result" output value, of
       column type int64.
    """
    row_text = HASH_FIELD_DELIMITER.join(_hash_value_token(value) for value in row_values)
    message = f"{key or ''}{HASH_FIELD_DELIMITER}{row_text}".encode("utf-8")
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


def hash_multi_column_case(key: str | None = None) -> dict[str, Any]:
    """Build one self-contained "hash" test case: a small multi-column, multi-row dataset (every
    vocabulary type) with a null in one column, its Arrow schema, a structurally valid config for
    it (written output name "hash_out", distinct from every input column), and the expected output
    column computed independently via ``compute_expected_hash_column`` (contract: Configuration
    "hash" operation shape). ``key`` is forwarded to both the config's ``parameters.key`` and the
    independent computation, so calling this with a different key exercises the "with
    parameters.key" variant of the operation with the same rows.

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
    output_columns = {"result": "hash_out"}
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


def run_binary_with_transport(
    cmd: list[str],
    env: dict[str, str],
    config_path: Path,
    input_bytes: bytes,
    *,
    use_input_file: bool,
    use_output_file: bool,
    tmp_path: Path,
    timeout: float = 10.0,
) -> tuple[subprocess.CompletedProcess[bytes], bytes]:
    """Run the binary via one of the four transport combinations (contract: Invocation, Data).
    Returns ``(result, output_bytes)``, where ``output_bytes`` is read from stdout or from the
    ``--output`` file depending on ``use_output_file``, so callers can assert on the data
    identically regardless of transport."""
    args = ["run", "--config", str(config_path)]
    stdin_bytes = input_bytes
    if use_input_file:
        input_path = tmp_path / "input.arrows"
        input_path.write_bytes(input_bytes)
        args = [*args, "--input", str(input_path)]
        stdin_bytes = b""
    output_path: Path | None = None
    if use_output_file:
        output_path = tmp_path / "output.arrows"
        args = [*args, "--output", str(output_path)]
    result = run_binary(cmd, args, env, stdin_bytes, timeout=timeout)
    if use_output_file:
        assert output_path is not None
        output_bytes = output_path.read_bytes() if output_path.exists() else b""
    else:
        output_bytes = result.stdout
    return result, output_bytes
