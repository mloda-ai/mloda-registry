"""Reusable, pip-installable binary-model conformance kit (``mloda-testing[binary-model]``):
subprocess/Arrow-IPC plumbing, the contract's fixed error codes, and the two class-based
conformance-check suites, meant to be imported both by this repository's own tests and by external
consumers (a private wrapper repo, binary vendors, consumer FeatureGroup repos) that subclass
``BinaryModelConformanceBase``/``HashOperationConformanceMixin`` to run the same checks against
their own binary.

- ``BinaryModelConformanceBase``: every CONTRACT-GENERIC check (invocation surface, capabilities,
  license gate, config structural validation, transport combinations, column-type vocabulary /
  exact-column-set rule, schema-only round trip, EOS marker, malformed-input rejection, the
  reserved internal-error operation, data-handling/diagnostics, error classification, trailing-data
  rejection, and Arrow metadata handling). A plain class (NOT a ``unittest.TestCase``), so pytest
  collects it directly once subclassed; it is not itself named ``Test*`` so pytest's default
  collection does not try to run it standalone.
- ``HashOperationConformanceMixin``: every HASH-OPERATION-SPECIFIC check (the reference algorithm,
  output schema/type, row-order preservation, ``parameters.key`` validation and null/empty-string
  equivalence, field-order independence, multi-batch input, and unknown-parameters rejection).
  Mixed in alongside ``BinaryModelConformanceBase`` to assemble a full conformance suite for a
  binary whose worked example is the "hash" operation.

Every overridable input (which binary to invoke, its plugin_id, its operations, its column-type
vocabulary, its license fixture texts, and the shape of a generically-valid config) is a class
attribute or an overridable method reached through ``self``, never a module-level constant and
never a pytest fixture, so a future conformance run against a real binary (or a different
operation) reuses these classes unmodified, just by subclassing with different class attributes.

The lower-level, non-test-specific mechanics these classes build on live in this package's other
modules and are re-exported here so a single import statement covers everything a subclass or a
wiring test module needs: Arrow IPC stream mechanics (``arrow.py``), the "hash" reference algorithm
(``hash_reference.py``), and the placeholder license-token text builders (``license_vectors.py``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess  # nosec
import sys
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pytest

from mloda.testing.binary_model import (
    CONTRACT_VERSION,
    DATA_ERROR,
    INTERNAL_ERROR,
    LICENSE_INVALID,
    LICENSE_MISSING,
    MESSAGE_MAX_BYTES,
    UNSUPPORTED,
    USAGE_ERROR,
)
from mloda.testing.binary_model import hash_reference, license_vectors
from mloda.testing.binary_model.arrow import (
    IPC_END_OF_STREAM_MARKER,
    arrow_file_format_bytes,
    arrow_stream_bytes,
    arrow_stream_bytes_from_arrays,
    arrow_stream_bytes_multi_batch,
    assert_ends_with_ipc_eos_marker,
    corrupt_record_batch_message_after_schema,
    enumerate_ipc_message_types,
    read_arrow_stream,
)

__all__ = [
    "CONTRACT_VERSION",
    "DATA_ERROR",
    "DATA_FREE_MARKER",
    "INTERNAL_ERROR",
    "IPC_END_OF_STREAM_MARKER",
    "LICENSE_INVALID",
    "LICENSE_MISSING",
    "UNSUPPORTED",
    "USAGE_ERROR",
    "arrow_file_format_bytes",
    "arrow_stream_bytes",
    "arrow_stream_bytes_from_arrays",
    "arrow_stream_bytes_multi_batch",
    "assert_ends_with_ipc_eos_marker",
    "assert_error_response",
    "assert_not_rejected_with",
    "corrupt_record_batch_message_after_schema",
    "enumerate_ipc_message_types",
    "read_arrow_stream",
    "run_binary",
    "run_binary_with_transport",
    "stderr_error_object",
    "write_json",
    "write_text",
    "BinaryModelConformanceBase",
    "HashOperationConformanceMixin",
]

# A distinctive marker string, chosen so it would never otherwise appear in this suite's input,
# output or diagnostics, used to test that a marked input cell's value never leaks into stderr
# (contract: Data handling).
DATA_FREE_MARKER = "SECRET_MARKER_YlZ9qX7"

# The contract's stderr soft cap (contract: Data handling). ``MESSAGE_MAX_BYTES`` (the `message`
# field's own cap) is a contract constant shared with the binary itself, so it lives in
# ``mloda.testing.binary_model``; this soft cap is a test-kit-only sanity check on total stderr
# size, not itself part of the contract.
STDERR_SOFT_CAP_BYTES = 64 * 1024


# =============================================================================
# Plain helpers -- subprocess invocation, error-object parsing/assertions
# =============================================================================


def write_text(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` and return the path, for config/license file fixtures."""
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, data: dict[str, Any]) -> Path:
    """Write ``data`` as JSON text to ``path`` and return the path."""
    return write_text(path, json.dumps(data))


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
    object; earlier lines, if any, are free-form diagnostics (contract: Errors). Decodes with
    ``errors="replace"`` rather than the strict default: a binary emitting non-UTF-8 stderr must
    fail an assertion here (the JSON parse below, or a caller's own assertion on the object),
    never crash the test suite itself with an unhandled ``UnicodeDecodeError``."""
    text = stderr.decode("utf-8", errors="replace")
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


# =============================================================================
# BinaryModelConformanceBase -- contract-generic checks
# =============================================================================


class BinaryModelConformanceBase:
    """Every contract-generic conformance check, plus the class attributes / overridable methods
    a subclass needs to point the whole kit at a different binary, plugin_id, operation set,
    column-type vocabulary, or license fixture shape.

    Deliberately not a ``unittest.TestCase``: a plain class so pytest collects ``test_*`` methods
    on any subclass whose own name matches pytest's ``Test*`` collection pattern (this class itself
    does not, so it is never collected standalone).
    """

    # -- Class attributes a subclass overrides to point this kit at a different binary/contract --
    binary_cmd: ClassVar[list[str]] = [sys.executable, "-m", "mloda.testing.binary_model.simulated_binary"]
    plugin_id: ClassVar[str] = "example_binary"
    wrong_plugin_id: ClassVar[str] = "some_other_plugin"
    operations: ClassVar[list[str]] = ["hash"]
    reserved_internal_error_operation: ClassVar[str] = "_conformance_internal_error"
    column_types: ClassVar[frozenset[str]] = frozenset({"int64", "float64", "utf8", "boolean"})

    # A generically-valid single-column config shape, used by every test that needs *some* valid
    # config/data but must not hardcode a hash-specific output name (that would leak an
    # operation-specific detail into contract-generic tests).
    default_input_columns: ClassVar[list[str]] = ["col_a"]
    default_output_columns: ClassVar[dict[str, str]] = {"result": "col_a_hash"}

    # License fixture texts (contract: License), built via ``license_vectors``'s reusable builders
    # from ``plugin_id``/``wrong_plugin_id`` at class-definition time; a subclass overriding
    # ``plugin_id`` alone would need to also override these to stay consistent, same tradeoff as
    # any other class-attribute default.
    valid_license_text: ClassVar[str] = license_vectors.license_token_text("valid", [plugin_id])
    expired_license_text: ClassVar[str] = license_vectors.license_token_text("expired", [plugin_id])
    wrong_plugin_license_text: ClassVar[str] = license_vectors.license_token_text("valid", [wrong_plugin_id])
    tampered_unparseable_text: ClassVar[str] = license_vectors.TAMPERED_UNPARSEABLE_TEXT
    tampered_missing_status_text: ClassVar[str] = license_vectors.tampered_missing_status_text([plugin_id])
    tampered_missing_plugins_text: ClassVar[str] = license_vectors.tampered_missing_plugins_text()

    # -- Overridable helpers for building a generically-valid config/data case --

    def make_config(
        self,
        *,
        input_columns: list[str] | None = None,
        operation: str | None = None,
        parameters: dict[str, Any] | None = None,
        output_columns: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """A structurally valid ``--config`` document for ``self.operations[0]``, every field
        overridable so a test can mutate exactly the one field under test (mirrors the old
        ``conftest.make_config``, sourced from instance state instead of module constants)."""
        return {
            "input_columns": list(self.default_input_columns) if input_columns is None else input_columns,
            "operation": self.operations[0] if operation is None else operation,
            "parameters": {} if parameters is None else parameters,
            "output_columns": dict(self.default_output_columns) if output_columns is None else output_columns,
        }

    def default_input_schema(self) -> pa.Schema:
        """A schema matching ``default_input_columns``, generic enough that any operation over the
        column-type vocabulary can run against it successfully."""
        return pa.schema([pa.field(self.default_input_columns[0], pa.string())])

    def default_input_rows(self) -> dict[str, list[Any]]:
        return {self.default_input_columns[0]: ["alpha", "beta", "gamma"]}

    def default_output_column_type(self) -> pa.DataType:
        """The Arrow type of ``self.operations[0]``'s single output column, for contract-generic
        tests (schema-only round trip) that need to assert on it without hardcoding a hash-specific
        detail. Defaults to int64, matching this contract's own worked example."""
        return pa.int64()

    @property
    def default_output_column_name(self) -> str:
        return next(iter(self.default_output_columns.values()))

    # -- Fixtures (equivalent to the old conftest.py's binary_cmd-independent fixtures) --

    @pytest.fixture
    def hermetic_env(self) -> dict[str, str]:
        """A minimal, controlled environment: no ambient shell variables, no license variables."""
        return {}

    @pytest.fixture
    def valid_config_path(self, tmp_path: Path) -> Path:
        """A structurally valid ``--config`` file, used to isolate the license gate and the
        invocation surface from config validation itself."""
        return write_json(tmp_path / "config.json", self.make_config())

    @pytest.fixture
    def valid_license_file(self, tmp_path: Path) -> Path:
        return write_text(tmp_path / "license.txt", self.valid_license_text)

    @pytest.fixture
    def valid_license_env(self, valid_license_file: Path) -> dict[str, str]:
        """``MLODA_LICENSE_FILE`` pointed at a valid token; used by config-validation tests to
        isolate config behaviour from the license gate."""
        return {"MLODA_LICENSE_FILE": str(valid_license_file)}

    # -------------------------------------------------------------------------------------------
    # 1. Invocation surface (contract: Invocation, Capabilities)
    # -------------------------------------------------------------------------------------------

    def test_version_prints_single_line_no_license_required(self, hermetic_env: dict[str, str]) -> None:
        """`--version` prints exactly one `<plugin_id> <semver>` line to stdout and exits 0, with no
        license variables set at all (contract: Invocation)."""
        version_pattern = re.compile(rf"^{re.escape(self.plugin_id)} \d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?$")
        result = run_binary(self.binary_cmd, ["--version"], hermetic_env)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        lines = result.stdout.decode("utf-8").splitlines()
        assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
        assert version_pattern.fullmatch(lines[0]), f"line does not match '<plugin_id> <semver>': {lines[0]!r}"

    def test_capabilities_prints_single_json_object_no_license_required(self, hermetic_env: dict[str, str]) -> None:
        """`--capabilities` prints exactly one JSON object followed by at most one trailing newline,
        nothing else, and exits 0, with no license variables set. The required keys (`contract`,
        `plugin_id`, `operations`, `column_types`) are present and correct; unknown extra keys are
        tolerated (contract: Invocation, Capabilities)."""
        result = run_binary(self.binary_cmd, ["--capabilities"], hermetic_env)
        assert result.returncode == 0, f"stderr={result.stderr!r}"

        stdout = result.stdout
        assert stdout.count(b"\n") <= 1, f"expected at most one trailing newline, got {stdout!r}"
        body_bytes = stdout[:-1] if stdout.endswith(b"\n") else stdout
        assert b"\n" not in body_bytes, f"expected exactly one JSON object on stdout, got {stdout!r}"

        body = json.loads(body_bytes.decode("utf-8"))
        assert isinstance(body, dict), f"expected a JSON object, got {body!r}"
        assert body.get("contract") == CONTRACT_VERSION, f"unexpected contract value: {body!r}"
        assert body.get("plugin_id") == self.plugin_id, f"unexpected plugin_id: {body!r}"
        operations = body.get("operations")
        assert isinstance(operations, list), f"operations must be a list: {body!r}"
        for op in self.operations:
            assert op in operations, f"expected operation {op!r} in {operations!r}"
        column_types = body.get("column_types")
        assert isinstance(column_types, list), f"column_types must be a list: {body!r}"
        assert set(column_types) == set(self.column_types), f"column_types mismatch: {column_types!r}"

    def test_no_arguments_is_usage_error(self, hermetic_env: dict[str, str]) -> None:
        """No arguments at all is a usage error: exit 1, no stdout, and no license required since a
        flag-parsing error happens before any license check (contract: Invocation)."""
        result = run_binary(self.binary_cmd, [], hermetic_env)
        assert_error_response(result, USAGE_ERROR)
        assert result.stdout == b"", f"expected no stdout data, got {result.stdout!r}"

    def test_help_flag_is_usage_error(self, hermetic_env: dict[str, str]) -> None:
        """`--help` is a usage error too: the binary is machine-invoked and has no interactive help
        (contract: Invocation)."""
        result = run_binary(self.binary_cmd, ["--help"], hermetic_env)
        assert_error_response(result, USAGE_ERROR)
        assert result.stdout == b"", f"expected no stdout data, got {result.stdout!r}"

    @pytest.mark.parametrize(
        "args",
        [
            pytest.param(["--bogus-flag"], id="unknown_flag"),
            pytest.param(["--version", "--capabilities"], id="conflicting_flags"),
            pytest.param(["--capabilities", "extra-positional"], id="extra_positional_after_capabilities"),
        ],
    )
    def test_unrecognized_flag_combination_is_usage_error(self, hermetic_env: dict[str, str], args: list[str]) -> None:
        """Any argument combination beyond the three documented invocations is a usage error
        (contract: Invocation)."""
        result = run_binary(self.binary_cmd, args, hermetic_env)
        assert_error_response(result, USAGE_ERROR)
        assert result.stdout == b"", f"expected no stdout data, got {result.stdout!r}"

    def test_run_without_config_is_usage_error(self, hermetic_env: dict[str, str]) -> None:
        """`run` requires `--config`; without it, usage error (contract: Invocation)."""
        result = run_binary(self.binary_cmd, ["run"], hermetic_env)
        assert_error_response(result, USAGE_ERROR)

    def test_run_with_nonexistent_config_path_is_usage_error(
        self, hermetic_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A `--config` path that does not exist fails flag parsing (`--config` must exist and be
        readable), before any license check (contract: Invocation, Errors)."""
        missing_path = tmp_path / "does-not-exist.json"
        result = run_binary(self.binary_cmd, ["run", "--config", str(missing_path)], hermetic_env)
        assert_error_response(result, USAGE_ERROR)

    # -------------------------------------------------------------------------------------------
    # 2. License gate (contract: License, Errors)
    # -------------------------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "env",
        [
            pytest.param({}, id="unset"),
            pytest.param({"MLODA_LICENSE_FILE": "", "MLODA_LICENSE_KEY": ""}, id="empty_strings"),
        ],
    )
    def test_license_missing_when_no_source_set(self, valid_config_path: Path, env: dict[str, str]) -> None:
        """Neither license variable set to a non-empty value: exit 2, license missing (contract:
        License)."""
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env)
        assert_error_response(result, LICENSE_MISSING)

    def test_license_missing_when_file_path_nonexistent(self, valid_config_path: Path, tmp_path: Path) -> None:
        """`MLODA_LICENSE_FILE` naming a file that does not exist: exit 2 (contract: License)."""
        env = {"MLODA_LICENSE_FILE": str(tmp_path / "no-such-license.txt")}
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env)
        assert_error_response(result, LICENSE_MISSING)

    def test_license_missing_message_names_file_source(self, valid_config_path: Path, tmp_path: Path) -> None:
        """The code 2 `message` names the source that was set (contract: License)."""
        env = {"MLODA_LICENSE_FILE": str(tmp_path / "no-such-license.txt")}
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env)
        error = assert_error_response(result, LICENSE_MISSING)
        assert "MLODA_LICENSE_FILE" in error["message"], f"message does not name the source: {error!r}"

    def test_license_accepted_via_license_file(
        self, valid_config_path: Path, valid_license_env: dict[str, str]
    ) -> None:
        """A valid token via `MLODA_LICENSE_FILE` proceeds past the license check; whatever happens
        next is never code 2 or 3 (contract: License)."""
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], valid_license_env)
        assert_not_rejected_with(result, {LICENSE_MISSING, LICENSE_INVALID})

    def test_license_accepted_via_license_key_inline(self, valid_config_path: Path) -> None:
        """A valid token via `MLODA_LICENSE_KEY` (inline) is accepted the same as a file (contract:
        License)."""
        env = {"MLODA_LICENSE_KEY": self.valid_license_text}
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env)
        assert_not_rejected_with(result, {LICENSE_MISSING, LICENSE_INVALID})

    def test_license_file_wins_over_license_key(self, valid_config_path: Path, valid_license_file: Path) -> None:
        """When both are set, `MLODA_LICENSE_FILE` wins with no fallback to `MLODA_LICENSE_KEY`: a
        valid file plus garbage inline key is still accepted (contract: License)."""
        env = {"MLODA_LICENSE_FILE": str(valid_license_file), "MLODA_LICENSE_KEY": "not-json-and-must-not-be-used"}
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env)
        assert_not_rejected_with(result, {LICENSE_MISSING, LICENSE_INVALID})

    def test_license_expired_is_invalid(self, valid_config_path: Path, tmp_path: Path) -> None:
        """An expired token: exit 3, license invalid (contract: License)."""
        license_path = write_text(tmp_path / "license.txt", self.expired_license_text)
        env = {"MLODA_LICENSE_FILE": str(license_path)}
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env)
        assert_error_response(result, LICENSE_INVALID)

    def test_license_wrong_plugin_is_invalid(self, valid_config_path: Path) -> None:
        """A token whose `plugins` entitlement list omits this `plugin_id`: exit 3. The message also
        names the source (contract: License)."""
        env = {"MLODA_LICENSE_KEY": self.wrong_plugin_license_text}
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env)
        error = assert_error_response(result, LICENSE_INVALID)
        assert "MLODA_LICENSE_KEY" in error["message"], f"message does not name the source: {error!r}"

    @pytest.mark.parametrize(
        "attr_name",
        [
            pytest.param("tampered_unparseable_text", id="unparseable_json"),
            pytest.param("tampered_missing_status_text", id="missing_status_key"),
            pytest.param("tampered_missing_plugins_text", id="missing_plugins_key"),
        ],
    )
    def test_license_tampered_is_invalid(self, valid_config_path: Path, tmp_path: Path, attr_name: str) -> None:
        """A tampered token (unparseable text, or valid JSON missing a required key): exit 3
        (contract: License). Parametrized by attribute name (looked up via ``getattr`` at test-run
        time) rather than by value, since the tampered-text fixtures are instance/class attributes,
        not module constants available at decoration time."""
        tampered_text = getattr(self, attr_name)
        license_path = write_text(tmp_path / "license.txt", tampered_text)
        env = {"MLODA_LICENSE_FILE": str(license_path)}
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env)
        assert_error_response(result, LICENSE_INVALID)

    def test_license_checked_before_config_valid_license_broken_config(
        self, valid_license_file: Path, tmp_path: Path
    ) -> None:
        """A valid license with a syntactically broken config gets past the license stage and fails
        on config parsing instead: exit 1, never 2 or 3 (contract: Errors, check order)."""
        config_path = write_text(tmp_path / "config.json", "{not valid json")
        env = {"MLODA_LICENSE_FILE": str(valid_license_file)}
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], env)
        assert_error_response(result, USAGE_ERROR)

    def test_license_checked_before_config_invalid_license_valid_config(
        self, valid_config_path: Path, tmp_path: Path
    ) -> None:
        """An invalid license with an otherwise valid config still exits 2 or 3, not something else,
        proving license is checked before config (contract: Errors, check order)."""
        license_path = write_text(tmp_path / "license.txt", self.expired_license_text)
        env = {"MLODA_LICENSE_FILE": str(license_path)}
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env)
        assert result.returncode in (LICENSE_MISSING, LICENSE_INVALID), (
            f"expected 2 or 3, got {result.returncode}; stderr={result.stderr!r}"
        )
        error = stderr_error_object(result.stderr)
        assert error.get("code") == result.returncode, f"error object code mismatch: {error!r}"

    def test_license_file_broken_key_valid_no_fallback_missing_file(
        self, valid_config_path: Path, tmp_path: Path
    ) -> None:
        """`MLODA_LICENSE_FILE` naming a missing file, with `MLODA_LICENSE_KEY` simultaneously set to
        a genuinely valid token: the missing file wins with no fallback to the key, so this is still
        code 2, never a fallback-to-key success (contract: License)."""
        env = {
            "MLODA_LICENSE_FILE": str(tmp_path / "no-such-license.txt"),
            "MLODA_LICENSE_KEY": self.valid_license_text,
        }
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env)
        assert_error_response(result, LICENSE_MISSING)

    def test_license_file_broken_key_valid_no_fallback_tampered_file(
        self, valid_config_path: Path, tmp_path: Path
    ) -> None:
        """`MLODA_LICENSE_FILE` naming a file with tampered, unparseable content, with
        `MLODA_LICENSE_KEY` simultaneously set to a genuinely valid token: still code 3, never a
        fallback-to-key success (contract: License)."""
        license_path = write_text(tmp_path / "license.txt", self.tampered_unparseable_text)
        env = {"MLODA_LICENSE_FILE": str(license_path), "MLODA_LICENSE_KEY": self.valid_license_text}
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env)
        assert_error_response(result, LICENSE_INVALID)

    # -------------------------------------------------------------------------------------------
    # 3. Config structural validation (contract: Configuration, Errors)
    # -------------------------------------------------------------------------------------------

    def test_config_json_syntax_error(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """A config file that is not valid JSON: exit 1 (contract: Configuration, Errors)."""
        config_path = write_text(tmp_path / "config.json", "{not valid json")
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_unknown_top_level_key(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """An unknown top-level config key: exit 1 (contract: Configuration)."""
        config = self.make_config()
        config["unexpected_top_level_key"] = "value"
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    @pytest.mark.parametrize("missing_key", ["input_columns", "operation", "parameters", "output_columns"])
    def test_config_missing_required_key(
        self, valid_license_env: dict[str, str], tmp_path: Path, missing_key: str
    ) -> None:
        """Each of the four required top-level keys is mandatory; missing any one is exit 1 (contract:
        Configuration)."""
        config = self.make_config()
        del config[missing_key]
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_input_columns_empty(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """`input_columns` must name at least one column: an empty list is exit 1 (contract:
        Configuration)."""
        config = self.make_config(input_columns=[])
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_input_columns_duplicate(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """`input_columns` without duplicates: a repeated name is exit 1 (contract: Configuration)."""
        config = self.make_config(input_columns=["col_a", "col_a"])
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_output_columns_written_names_not_unique(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Written names in `output_columns` must be unique among themselves; this is checked
        structurally, before the operation's own output list is consulted (contract: Configuration,
        Errors)."""
        config = self.make_config(output_columns={"result": "dup_name", "an_extra_output_key": "dup_name"})
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_output_columns_collides_with_input_columns(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Every written output name must be distinct from every `input_columns` entry: colliding with
        one is exit 1 (contract: Configuration)."""
        config = self.make_config(input_columns=["col_a"], output_columns={"result": "col_a"})
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_operation_not_in_capabilities(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """An `operation` outside `capabilities.operations` (and not the reserved conformance-only
        operation) is unsupported: exit 4 (contract: Configuration, Errors)."""
        config = self.make_config(operation="not_a_real_operation")
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, UNSUPPORTED)

    def test_reserved_internal_error_operation_not_rejected_as_unknown(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """The reserved internal-error operation is not listed in `capabilities.operations` but must
        still be accepted at the operation-capability-check step, bypassing that check specifically
        for this literal string, so the kit can provoke code 6 on demand (contract: Conformance)."""
        config = self.make_config(operation=self.reserved_internal_error_operation)
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_not_rejected_with(result, {UNSUPPORTED})

    def test_config_output_columns_missing_operation_output(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """`output_columns` must map every output the operation defines; an empty mapping is missing
        it: exit 1, checked after the operation check (contract: Configuration, Errors)."""
        config = self.make_config(output_columns={})
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_output_columns_extra_unmapped_output(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """An output name the operation does not define is also exit 1, checked after the operation
        check (contract: Configuration, Errors)."""
        config = self.make_config(
            output_columns={"result": self.default_output_column_name, "not_a_real_output": "col_b_out"}
        )
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_unknown_operation_with_bad_output_columns_reports_operation_error_first(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """The operation-capability check (code 4) runs before the output_columns completeness check
        (code 1): an unknown operation combined with an incomplete output mapping is exit 4, not exit
        1 (contract: Errors, check order)."""
        config = self.make_config(operation="not_a_real_operation", output_columns={})
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, UNSUPPORTED)

    def test_config_parameters_empty_object_accepted_structurally(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """`parameters: {}` is structurally accepted for an operation whose parameters are all
        optional: this does not itself cause exit 1 (contract: Configuration). Final success is out
        of scope here, so nothing else is asserted."""
        config = self.make_config(parameters={})
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert result.returncode != USAGE_ERROR, (
            f"empty parameters object unexpectedly caused a usage error; stderr={result.stderr!r}"
        )

    # -------------------------------------------------------------------------------------------
    # 4. Transport combinations (contract: Invocation, Data) -- generic byte-identical-output check
    # -------------------------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "use_input_file, use_output_file",
        [
            pytest.param(False, False, id="stdin_stdout"),
            pytest.param(False, True, id="stdin_output_file"),
            pytest.param(True, False, id="input_file_stdout"),
            pytest.param(True, True, id="input_file_output_file"),
        ],
    )
    def test_all_transport_combinations_produce_identical_output(
        self,
        valid_license_env: dict[str, str],
        tmp_path: Path,
        use_input_file: bool,
        use_output_file: bool,
    ) -> None:
        """All four transport combinations (`--input`/stdin crossed with `--output`/stdout) must
        produce byte-identical output for the same input data and config, regardless of which
        operation is under test (contract: Invocation, Data). Compares each transport's raw output
        bytes to a stdin/stdout baseline run rather than asserting any operation-specific value, so
        this stays contract-generic; hash-specific correctness is covered separately by
        ``HashOperationConformanceMixin.test_hash_transport_combinations_match_reference_algorithm``."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        input_bytes = arrow_stream_bytes(self.default_input_schema(), self.default_input_rows())
        result, output_bytes = run_binary_with_transport(
            self.binary_cmd,
            valid_license_env,
            config_path,
            input_bytes,
            use_input_file=use_input_file,
            use_output_file=use_output_file,
            tmp_path=tmp_path,
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        baseline_result, baseline_output = run_binary_with_transport(
            self.binary_cmd,
            valid_license_env,
            config_path,
            input_bytes,
            use_input_file=False,
            use_output_file=False,
            tmp_path=tmp_path,
        )
        assert baseline_result.returncode == 0, f"stderr={baseline_result.stderr!r}"
        assert output_bytes == baseline_output, "transport combination produced different output bytes"

    def test_output_file_transport_leaves_stdout_empty(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """With `--output <path>` used, stdout stays empty (contract: Invocation)."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        input_bytes = arrow_stream_bytes(self.default_input_schema(), self.default_input_rows())
        result, output_bytes = run_binary_with_transport(
            self.binary_cmd,
            valid_license_env,
            config_path,
            input_bytes,
            use_input_file=False,
            use_output_file=True,
            tmp_path=tmp_path,
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert result.stdout == b"", f"expected empty stdout when --output is used, got {result.stdout!r}"
        table = read_arrow_stream(output_bytes)
        assert table.num_rows == len(next(iter(self.default_input_rows().values())))

    def test_input_file_given_ignores_stdin(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """When `--input <path>` is used, stdin is never read (contract: Invocation): deliberately
        garbage bytes fed on stdin alongside a valid `--input` file must not prevent success -- a
        binary that read stdin instead of the file would fail here, since the garbage is not a valid
        Arrow IPC stream, so success alone proves stdin was ignored (no operation-specific output
        value needs to be checked)."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        input_bytes = arrow_stream_bytes(self.default_input_schema(), self.default_input_rows())
        input_path = tmp_path / "input.arrows"
        input_path.write_bytes(input_bytes)
        garbage_stdin = b"this-is-not-an-arrow-stream-and-must-be-ignored" * 4
        result = run_binary(
            self.binary_cmd,
            ["run", "--config", str(config_path), "--input", str(input_path)],
            valid_license_env,
            garbage_stdin,
        )
        assert result.returncode == 0, (
            f"expected success reading from --input despite garbage stdin; stderr={result.stderr!r}"
        )

    def test_zero_length_input_file_is_data_error(
        self, valid_config_path: Path, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A zero-length `--input` file is not an Arrow IPC stream at all, exactly like zero-byte
        stdin (contract: Data)."""
        input_path = tmp_path / "empty-input.arrows"
        input_path.write_bytes(b"")
        result = run_binary(
            self.binary_cmd,
            ["run", "--config", str(valid_config_path), "--input", str(input_path)],
            valid_license_env,
        )
        assert_error_response(result, DATA_ERROR)

    # -------------------------------------------------------------------------------------------
    # 5. Exact-column-set rule and column type vocabulary (contract: Data, Capabilities)
    # -------------------------------------------------------------------------------------------

    def test_input_schema_missing_column_is_data_error(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """The input stream's schema must contain exactly `input_columns`; a missing name is a data
        error (contract: Data)."""
        config = self.make_config(input_columns=["col_a", "col_b"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("col_a", pa.int64())])  # col_b missing entirely
        input_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2]})
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert_error_response(result, DATA_ERROR)

    def test_input_schema_extra_column_is_data_error(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """An extra column beyond `input_columns` is a data error (contract: Data)."""
        config = self.make_config(input_columns=["col_a", "col_b"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema(
            [pa.field("col_a", pa.int64()), pa.field("col_b", pa.int64()), pa.field("col_c", pa.int64())]
        )
        input_bytes = arrow_stream_bytes(schema, {"col_a": [1], "col_b": [2], "col_c": [3]})
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert_error_response(result, DATA_ERROR)

    def test_input_schema_duplicate_field_name_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Two distinct Arrow fields sharing the same name is a "duplicate", a data error, even though
        the set of distinct names equals `input_columns` (contract: Data)."""
        config = self.make_config(input_columns=["col_a", "col_b"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema(
            [pa.field("col_a", pa.int64()), pa.field("col_a", pa.int64()), pa.field("col_b", pa.int64())]
        )
        input_bytes = arrow_stream_bytes_from_arrays(schema, [pa.array([1, 2]), pa.array([3, 4]), pa.array([5, 6])])
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert_error_response(result, DATA_ERROR)

    @pytest.mark.parametrize(
        "bad_type, sample_values",
        [
            pytest.param(pa.int32(), [1, 2], id="int32_not_in_vocabulary"),
            pytest.param(pa.timestamp("us"), [0, 1], id="timestamp_not_in_vocabulary"),
        ],
    )
    def test_input_column_type_outside_vocabulary_is_unsupported(
        self, valid_license_env: dict[str, str], tmp_path: Path, bad_type: pa.DataType, sample_values: list[Any]
    ) -> None:
        """A column typed outside `column_types` (int32, a timestamp type, ...) is code 4 (contract:
        Capabilities)."""
        config = self.make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("col_a", bad_type)])
        input_bytes = arrow_stream_bytes(schema, {"col_a": sample_values})
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert_error_response(result, UNSUPPORTED)

    def test_input_column_large_string_type_is_rejected_as_unsupported(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """NEW/stricter: only bare `pa.string()` counts as this contract's `utf8`; `pa.large_string()`
        must be rejected as an unsupported column type (code 4) (contract: Capabilities). The old
        stub incorrectly accepted `large_string` as `utf8` via
        `pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t)`, so this
        currently fails against that logic."""
        column = self.default_input_columns[0]
        config = self.make_config(input_columns=[column], output_columns={"result": self.default_output_column_name})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field(column, pa.large_string())])
        input_bytes = arrow_stream_bytes(schema, {column: ["alpha", "beta"]})
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert_error_response(result, UNSUPPORTED)

    def test_input_column_string_view_type_is_rejected_as_unsupported(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """NEW/stricter: same rule for `pa.string_view()` (contract: Capabilities). Currently fails
        against the old stub for the same reason as the `large_string` case above."""
        column = self.default_input_columns[0]
        config = self.make_config(input_columns=[column], output_columns={"result": self.default_output_column_name})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field(column, pa.string_view())])
        input_bytes = arrow_stream_bytes(schema, {column: ["alpha", "beta"]})
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert_error_response(result, UNSUPPORTED)

    def test_input_schema_presence_error_precedes_type_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A presence violation (missing `col_b`) combined with a type violation (`col_a` sent as
        int32 instead of int64) is a data error, not code 4: presence is checked first (contract:
        Data)."""
        config = self.make_config(input_columns=["col_a", "col_b"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("col_a", pa.int32())])  # col_b missing; col_a also wrong type
        input_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2]})
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert_error_response(result, DATA_ERROR)

    def test_dictionary_encoded_column_is_unsupported_type(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A dictionary-encoded column is an unsupported column type (code 4), decided by the same
        type check as any other type outside the vocabulary (contract: Data)."""
        config = self.make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        dict_array = pa.array(["x", "y", "x"]).dictionary_encode()
        schema = pa.schema([pa.field("col_a", dict_array.type)])
        input_bytes = arrow_stream_bytes_from_arrays(schema, [dict_array])
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert_error_response(result, UNSUPPORTED)

    # -------------------------------------------------------------------------------------------
    # 6. Schema-only round trip (contract: Data)
    # -------------------------------------------------------------------------------------------

    def test_schema_only_input_valid_license_produces_schema_only_output(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Zero record batches then the end-of-stream marker is valid input; the output is
        schema-only too, but already carries the output columns and types (contract: Data)."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        input_bytes = arrow_stream_bytes(self.default_input_schema(), None)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        table = read_arrow_stream(result.stdout)
        assert table.num_rows == 0
        assert table.schema.names == [self.default_output_column_name]
        assert table.schema.field(self.default_output_column_name).type == self.default_output_column_type()

    def test_schema_only_input_bad_license_still_rejected(self, valid_config_path: Path, tmp_path: Path) -> None:
        """A schema-only input with a bad license still exits 2 or 3, not 0: the license check applies
        before any data is read, schema-only input included (contract: Data, License)."""
        input_bytes = arrow_stream_bytes(self.default_input_schema(), None)
        license_path = write_text(tmp_path / "license.txt", self.expired_license_text)
        env = {"MLODA_LICENSE_FILE": str(license_path)}
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env, input_bytes)
        assert result.returncode in (LICENSE_MISSING, LICENSE_INVALID), (
            f"expected 2 or 3, got {result.returncode}; stderr={result.stderr!r}"
        )
        error = stderr_error_object(result.stderr)
        assert error.get("code") == result.returncode, f"error object code mismatch: {error!r}"

    def test_schema_only_output_has_no_record_batch_message(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A schema-only output (for a schema-only input) must be schema + end-of-stream marker only:
        no record-batch message at all, not a record-batch message with zero rows (contract: Data).
        `pa.ipc.open_stream(...).read_all()` cannot distinguish the two wire shapes, so this
        enumerates the raw output's message sequence instead."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        input_bytes = arrow_stream_bytes(self.default_input_schema(), None)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        message_types = enumerate_ipc_message_types(result.stdout)
        assert "record batch" not in message_types, (
            f"expected a schema-only output to carry no record-batch message, got {message_types!r}"
        )

    # -------------------------------------------------------------------------------------------
    # 7. End-of-stream marker on the raw output bytes (contract: Data, Conformance)
    # -------------------------------------------------------------------------------------------

    def test_output_stream_raw_bytes_end_with_ipc_eos_marker(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """The raw output bytes end with the IPC end-of-stream marker, checked on the bytes
        themselves rather than through pyarrow's own reader, which tolerates a stream missing it
        (contract: Data)."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        input_bytes = arrow_stream_bytes(self.default_input_schema(), self.default_input_rows())
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert_ends_with_ipc_eos_marker(result.stdout)

    # -------------------------------------------------------------------------------------------
    # 8. Malformed input rejections (contract: Data)
    # -------------------------------------------------------------------------------------------

    def test_zero_byte_input_is_data_error(self, valid_config_path: Path, valid_license_env: dict[str, str]) -> None:
        """Zero bytes is not an Arrow IPC stream at all: a data error (contract: Data)."""
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], valid_license_env, b"")
        assert_error_response(result, DATA_ERROR)

    def test_truncated_stream_missing_end_of_stream_marker_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """ "truncated" means end of file without the end-of-stream marker, not "no more batches": a
        stream cut off right before that marker is a data error (contract: Data)."""
        config = self.make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("col_a", pa.int64())])
        full_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2, 3]})
        assert full_bytes.endswith(IPC_END_OF_STREAM_MARKER), "test setup: expected the writer to emit the EOS marker"
        truncated = full_bytes[: -len(IPC_END_OF_STREAM_MARKER)]
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, truncated)
        assert_error_response(result, DATA_ERROR)

    def test_ipc_file_format_instead_of_stream_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """The IPC file/Feather format (`ARROW1` magic bytes) is not the streaming format the contract
        requires: a data error (contract: Data)."""
        config = self.make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("col_a", pa.int64())])
        input_bytes = arrow_file_format_bytes(schema, {"col_a": [1, 2, 3]})
        assert input_bytes[:6] == b"ARROW1", "test setup: expected the IPC file magic at the start"
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert_error_response(result, DATA_ERROR)

    def test_compressed_record_batch_body_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A compressed record batch body is a data error (contract: Data)."""
        config = self.make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("col_a", pa.int64())])
        options = pa.ipc.IpcWriteOptions(compression="lz4")
        input_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2, 3]}, options=options)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert_error_response(result, DATA_ERROR)

    def test_malformed_record_batch_after_valid_schema_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Malformed/corrupted record-batch data appearing *after* a valid schema message is a data
        error, exit 5 (contract: Data): distinct from "malformed bytes from the very start"; the
        schema-parsing step succeeds here, but reading the record batch fails on the corrupted second
        message."""
        config = self.make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("col_a", pa.int64())])
        valid_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2, 3]})
        corrupted = corrupt_record_batch_message_after_schema(valid_bytes)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, corrupted)
        assert_error_response(result, DATA_ERROR)

    def test_output_path_is_existing_directory_is_usage_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """`--output` naming an existing directory (not a file) is a usage error, exit 1: "opening
        --input and creating --output (code 1 if either fails)" (contract: Errors)."""
        config = self.make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        output_dir = tmp_path / "output_is_a_directory"
        output_dir.mkdir()
        result = run_binary(
            self.binary_cmd, ["run", "--config", str(config_path), "--output", str(output_dir)], valid_license_env
        )
        assert_error_response(result, USAGE_ERROR)

    def test_input_path_not_readable_is_usage_error(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """An `--input` path that exists but is not readable (chmod 000) is a usage error, exit 1
        (contract: Errors). Skipped when running as root, which ignores file read permissions."""
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("running as root ignores file read permissions")
        config = self.make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("col_a", pa.int64())])
        input_path = tmp_path / "input.arrows"
        input_path.write_bytes(arrow_stream_bytes(schema, {"col_a": [1, 2, 3]}))
        input_path.chmod(0o000)
        try:
            result = run_binary(
                self.binary_cmd,
                ["run", "--config", str(config_path), "--input", str(input_path)],
                valid_license_env,
            )
        finally:
            input_path.chmod(0o644)
        assert_error_response(result, USAGE_ERROR)

    # -------------------------------------------------------------------------------------------
    # 9. The reserved internal-error operation (contract: Conformance)
    # -------------------------------------------------------------------------------------------

    def test_reserved_internal_error_operation_with_data_triggers_code_6(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """The reserved internal-error operation, given a structurally valid config and valid input
        data, reaches the data stage and deliberately produces code 6, with stderr's last non-empty
        line still a valid `{"code": 6, "message": ...}` object, not a bare traceback (contract:
        Conformance)."""
        config = self.make_config(operation=self.reserved_internal_error_operation)
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("col_a", pa.int64())])
        input_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2, 3]})
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert_error_response(result, INTERNAL_ERROR)

    def test_internal_error_message_reports_only_exception_class_name(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """NEW: any code-6 message must be exactly `f"internal error: {ExceptionClassName}"` -- only
        the exception's class name, never its full string representation, which could otherwise leak
        caller data (contract: Data handling, Errors). Mechanism: reuses the reserved internal-error
        operation with valid data (the only reliable way to provoke code 6 from this binary's public
        surface) -- currently that operation raises `_CliError` directly with a fixed message
        ("reserved conformance operation: deliberate internal error"), bypassing the generic
        unexpected-exception catch-all and not matching this shape at all, so this currently fails."""
        config = self.make_config(operation=self.reserved_internal_error_operation)
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("col_a", pa.int64())])
        input_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2, 3]})
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        error = assert_error_response(result, INTERNAL_ERROR)
        message = error["message"]
        # Mechanism: the reserved operation is the only way to reach code 6 from the public CLI
        # surface; the message must be exactly "internal error: <ExceptionClassName>" (no other text).
        assert re.fullmatch(r"internal error: [A-Za-z_][A-Za-z0-9_]*", message), (
            f"expected 'internal error: <ExceptionClassName>' with no other text, got {message!r}"
        )

    # -------------------------------------------------------------------------------------------
    # 10. Trailing-data / concatenated-stream rejection (contract: Data)
    # -------------------------------------------------------------------------------------------

    def test_two_concatenated_ipc_streams_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Two complete, valid Arrow IPC streams concatenated back-to-back must be rejected as a data
        error, not silently accepted with only the first stream's rows returned (contract: Data)."""
        config = self.make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("col_a", pa.int64())])
        stream_one = arrow_stream_bytes(schema, {"col_a": [1, 2, 3]})
        stream_two = arrow_stream_bytes(schema, {"col_a": [4, 5, 6]})
        result = run_binary(
            self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, stream_one + stream_two
        )
        assert_error_response(result, DATA_ERROR)

    def test_trailing_garbage_bytes_after_eos_marker_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Valid stream bytes followed by arbitrary trailing garbage (not another full stream) after
        the end-of-stream marker is equally a data error (contract: Data)."""
        config = self.make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("col_a", pa.int64())])
        valid_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2, 3]})
        garbage = b"trailing-garbage-bytes-not-a-stream-0123456789"
        assert garbage[-len(IPC_END_OF_STREAM_MARKER) :] != IPC_END_OF_STREAM_MARKER, (
            "test setup: the garbage suffix must not coincidentally end with the EOS marker"
        )
        result = run_binary(
            self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, valid_bytes + garbage
        )
        assert_error_response(result, DATA_ERROR)

    # -------------------------------------------------------------------------------------------
    # 11. Data handling: data-free diagnostics, size caps, no network, no incidental files, the
    #     minimal environment (contract: Data handling, Errors, Conformance)
    # -------------------------------------------------------------------------------------------

    def test_diagnostics_never_leak_marked_cell_value_on_success(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A distinctive marker cell value, sent through a normal successful run: stderr never
        contains it. stdout on a successful run legitimately carries the caller's own data by design,
        so this scopes the assertion to stderr only (contract: Data handling)."""
        column = self.default_input_columns[0]
        config = self.make_config(input_columns=[column], output_columns={"result": self.default_output_column_name})
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field(column, pa.string())])
        input_bytes = arrow_stream_bytes(schema, {column: [DATA_FREE_MARKER]})
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert DATA_FREE_MARKER.encode("utf-8") not in result.stderr, (
            f"marker cell value leaked into stderr on a successful run: {result.stderr!r}"
        )

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param("missing_column_data_error", id="missing_column_data_error"),
            pytest.param("reserved_internal_error_operation", id="reserved_internal_error_operation"),
        ],
    )
    def test_diagnostics_never_leak_marked_cell_value_on_failure(
        self, valid_license_env: dict[str, str], tmp_path: Path, case: str
    ) -> None:
        """The same marker cell value, sent through two failing cases that still reach the data
        stage with the marked input present: a data error from a wrong schema, and the reserved
        internal-error operation. Neither case's stderr contains the marker (contract: Data handling,
        Conformance)."""
        column = self.default_input_columns[0]
        if case == "missing_column_data_error":
            input_columns = [column, "col_b_missing"]
            operation = self.operations[0]
            output_columns = self.default_output_columns
        else:
            input_columns = [column]
            operation = self.reserved_internal_error_operation
            output_columns = {}
        config = self.make_config(input_columns=input_columns, operation=operation, output_columns=output_columns)
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field(column, pa.string())])
        input_bytes = arrow_stream_bytes(schema, {column: [DATA_FREE_MARKER]})
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert result.returncode != 0, f"expected this case to fail, got exit 0; stdout={result.stdout!r}"
        assert DATA_FREE_MARKER.encode("utf-8") not in result.stderr, (
            f"marker cell value leaked into stderr: {result.stderr!r}"
        )

    def test_error_message_stays_under_size_cap_for_long_garbage_operation(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A very long garbage `operation` string provokes the unsupported-operation error message to
        echo it back; `assert_error_response`'s shared size-cap assertion enforces that the resulting
        `message` still stays within the contract's 1024-byte cap (contract: Data handling,
        Conformance)."""
        long_garbage_operation = "not_a_real_operation_" + "x" * 2000
        config = self.make_config(operation=long_garbage_operation)
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, UNSUPPORTED)

    def test_no_network_dependency_under_unshare_net(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """A normal, successful run wrapped in `unshare --net` (a network-denied Linux namespace)
        must still succeed and produce a structurally valid output, proving the binary makes no
        network calls it depends on (contract: Data handling, Conformance). Exact per-operation
        output correctness is covered elsewhere; this test stays scoped to the sandboxing question.
        Skipped if `unshare`/`unshare --net` is not usable in this sandbox."""
        if shutil.which("unshare") is None:
            pytest.skip("unshare is not available on this host")
        probe = subprocess.run(  # nosec B603 B607
            ["unshare", "--net", "--", "/bin/true"], capture_output=True, timeout=10.0
        )
        if probe.returncode != 0:
            pytest.skip(f"unshare --net is not usable in this sandbox: {probe.stderr!r}")

        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        rows = self.default_input_rows()
        input_bytes = arrow_stream_bytes(self.default_input_schema(), rows)
        env = {**valid_license_env, "PATH": os.environ.get("PATH", os.defpath)}
        result = run_binary(
            ["unshare", "--net", "--", *self.binary_cmd], ["run", "--config", str(config_path)], env, input_bytes
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        table = read_arrow_stream(result.stdout)
        assert table.num_rows == len(next(iter(rows.values())))

    def test_no_files_created_outside_output_in_read_only_cwd(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A normal, successful run over stdin/stdout run with its working directory set to a fresh
        read-only directory must still succeed and create no files there, proving the binary writes
        nothing of its own outside `--output` (contract: Data handling, Conformance). Skipped on a
        platform with no POSIX directory-permission concept, or when running as root."""
        if not hasattr(os, "geteuid"):
            pytest.skip("no POSIX directory-permission concept on this platform")
        if os.geteuid() == 0:
            pytest.skip("running as root ignores directory write permissions")

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        config = self.make_config()
        config_path = write_json(work_dir / "config.json", config)
        rows = self.default_input_rows()
        input_bytes = arrow_stream_bytes(self.default_input_schema(), rows)

        readonly_cwd = tmp_path / "readonly_cwd"
        readonly_cwd.mkdir()
        original_mode = readonly_cwd.stat().st_mode
        readonly_cwd.chmod(0o500)  # read + execute only: no write, so nothing can be created here
        try:
            result = run_binary(
                self.binary_cmd,
                ["run", "--config", str(config_path)],
                valid_license_env,
                input_bytes,
                cwd=readonly_cwd,
            )
        finally:
            readonly_cwd.chmod(original_mode)

        assert result.returncode == 0, f"stderr={result.stderr!r}"
        table = read_arrow_stream(result.stdout)
        assert table.num_rows == len(next(iter(rows.values())))
        leftover = list(readonly_cwd.iterdir())
        assert leftover == [], f"expected no files created in the read-only cwd, found {leftover!r}"

    def test_minimal_environment_allowlist_only(self, tmp_path: Path) -> None:
        """A normal, successful run with an environment containing only the license variable and
        `PATH` -- no `HOME`, no `LANG`, nothing else -- must still succeed, proving the binary reads
        no environment variable outside the license variables and what its runtime needs to start
        (contract: Data handling, Conformance)."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        rows = self.default_input_rows()
        input_bytes = arrow_stream_bytes(self.default_input_schema(), rows)
        env = {"MLODA_LICENSE_KEY": self.valid_license_text, "PATH": os.environ.get("PATH", "")}
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], env, input_bytes)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        table = read_arrow_stream(result.stdout)
        assert table.num_rows == len(next(iter(rows.values())))

    # -------------------------------------------------------------------------------------------
    # 12. Corrected error classification (contract: Errors, License, Configuration, Data)
    # -------------------------------------------------------------------------------------------

    def test_config_invalid_utf8_bytes_is_usage_error(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """A `--config` file containing bytes that are not valid UTF-8 is a usage error, exit 1:
        config decode/parse failures are usage errors (contract: Errors, Invocation), never code 6."""
        config_path = tmp_path / "config.json"
        config_path.write_bytes(b"\xff\xfe\x00invalid-utf8-config")
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_license_file_invalid_utf8_bytes_is_license_invalid(self, valid_config_path: Path, tmp_path: Path) -> None:
        """A file named by `MLODA_LICENSE_FILE` containing bytes that are not valid UTF-8 is code 3,
        not code 2 and not code 6: the source is set and the file exists, but its content cannot be
        read as the license token (contract: License)."""
        license_path = tmp_path / "license.txt"
        license_path.write_bytes(b"\xff\xfe\x00invalid-utf8-license")
        env = {"MLODA_LICENSE_FILE": str(license_path)}
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env)
        assert_error_response(result, LICENSE_INVALID)

    # -------------------------------------------------------------------------------------------
    # 13. Arrow metadata handling (contract: Data)
    # -------------------------------------------------------------------------------------------

    def test_input_arrow_metadata_schema_and_field_level_accepted_and_stripped_from_output(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """NEW: schema-level and field-level Arrow metadata attached to the INPUT stream is accepted
        without failing the run, and the OUTPUT stream carries none of it: metadata is
        stripped/ignored, never echoed back onto either the output schema or any output field
        (contract: Data). Asserts the input actually carries non-empty metadata first, ruling out a
        trivial always-empty-output-metadata pass."""
        column = self.default_input_columns[0]
        field = pa.field(column, pa.string(), metadata={b"field_meta_key": b"field_meta_value"})
        schema = pa.schema([field]).with_metadata({b"schema_meta_key": b"schema_meta_value"})
        assert schema.metadata, "test setup: expected the input schema to carry schema-level metadata"
        assert schema.field(column).metadata, "test setup: expected the input field to carry field-level metadata"
        config = self.make_config(input_columns=[column], output_columns={"result": self.default_output_column_name})
        config_path = write_json(tmp_path / "config.json", config)
        input_bytes = arrow_stream_bytes(schema, {column: ["alpha", "beta"]})
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert result.returncode == 0, f"input metadata unexpectedly caused a failure; stderr={result.stderr!r}"
        table = read_arrow_stream(result.stdout)
        assert not table.schema.metadata, f"output schema unexpectedly carries metadata: {table.schema.metadata!r}"
        for output_field in table.schema:
            assert not output_field.metadata, (
                f"output field {output_field.name!r} unexpectedly carries metadata: {output_field.metadata!r}"
            )


# =============================================================================
# HashOperationConformanceMixin -- "hash" operation-specific checks
# =============================================================================


class HashOperationConformanceMixin(BinaryModelConformanceBase):
    """Every check specific to the "hash" operation's own shape: the reference algorithm, output
    schema/type, row-order preservation, `parameters.key` validation and null/empty-string
    equivalence, field-order independence, and multi-batch input. Mixed in alongside
    ``BinaryModelConformanceBase`` (which carries every contract-generic check) to assemble a full
    conformance suite for a binary whose worked example is the "hash" operation.

    The reference algorithm itself (``compute_expected_hash``/``compute_expected_hash_column``) and
    the multi-column test-case builder (``hash_multi_column_case``) delegate to
    ``mloda.testing.binary_model.hash_reference``, the single implementation also imported by
    ``simulated_binary.py``, so the two can never silently drift apart."""

    operations: ClassVar[list[str]] = ["hash"]
    default_output_columns: ClassVar[dict[str, str]] = {"result": "hash_out"}

    # -- The "hash" operation's reference algorithm (independent of whatever the binary does) --

    compute_expected_hash = staticmethod(hash_reference.compute_expected_hash)
    compute_expected_hash_column = staticmethod(hash_reference.compute_expected_hash_column)

    def hash_multi_column_case(self, key: str | None = None) -> dict[str, Any]:
        """Build one self-contained "hash" test case: a small multi-column, multi-row dataset (every
        vocabulary type) with a null in one column, its Arrow schema, a structurally valid config for
        it, and the expected output column computed independently via `compute_expected_hash_column`
        (contract: Configuration "hash" operation shape). Delegates the dataset/expected-value
        construction to ``hash_reference.hash_multi_column_case``, passing ``self.make_config`` and
        ``self.default_output_column_name`` through so a subclass's overrides are honoured."""
        return hash_reference.hash_multi_column_case(
            key=key, output_column_name=self.default_output_column_name, make_config=self.make_config
        )

    # -------------------------------------------------------------------------------------------
    # H1. "hash" reference algorithm (contract: Configuration "hash" operation shape; Data)
    # -------------------------------------------------------------------------------------------

    def test_hash_multi_column_with_null_matches_reference_algorithm(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """ "hash" with no `parameters.key`, multiple input columns covering every vocabulary type,
        and a null in one column matches the independent reference computation byte for byte, row for
        row (contract: Configuration "hash" shape; Data)."""
        case = self.hash_multi_column_case()
        config_path = write_json(tmp_path / "config.json", case["config"])
        input_bytes = arrow_stream_bytes(case["schema"], case["rows"])
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        table = read_arrow_stream(result.stdout)
        assert table.num_rows == len(case["expected"]), f"row count mismatch: {table.num_rows}"
        assert table.column(self.default_output_column_name).to_pylist() == case["expected"]

    def test_hash_with_key_parameter_matches_reference_algorithm_and_changes_result(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """ "hash" with a `parameters.key` produces the reference algorithm's value computed with that
        key, which differs from the no-key result for the same rows (contract: Configuration)."""
        key = "s3cr3t-key"
        case = self.hash_multi_column_case(key=key)
        config_path = write_json(tmp_path / "config.json", case["config"])
        input_bytes = arrow_stream_bytes(case["schema"], case["rows"])
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        table = read_arrow_stream(result.stdout)
        actual = table.column(self.default_output_column_name).to_pylist()
        assert actual == case["expected"]

        no_key_case = self.hash_multi_column_case(key=None)
        assert actual != no_key_case["expected"], "the parameters.key must change the hash result"

    def test_hash_row_count_and_row_order_preserved(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """Row count and row order are preserved: distinct `id` values per row make the expected hash
        order-sensitive, so a binary that drops, adds or reorders rows fails this even if it computes
        correct hashes for the wrong positions (contract: Data, Conformance)."""
        case = self.hash_multi_column_case()
        config_path = write_json(tmp_path / "config.json", case["config"])
        input_bytes = arrow_stream_bytes(case["schema"], case["rows"])
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        table = read_arrow_stream(result.stdout)
        assert table.num_rows == len(case["rows"]["id"])
        actual = table.column(self.default_output_column_name).to_pylist()
        for row_index, (expected_value, row_id) in enumerate(zip(case["expected"], case["rows"]["id"])):
            assert actual[row_index] == expected_value, (
                f"row {row_index} ({row_id!r}) hash mismatch: expected {expected_value}, got {actual[row_index]}"
            )

    def test_hash_output_schema_exact_type_int64_no_input_columns_echoed(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Output schema is exactly the `output_columns` written name, typed int64; no input column
        is echoed (contract: Data)."""
        case = self.hash_multi_column_case()
        config_path = write_json(tmp_path / "config.json", case["config"])
        input_bytes = arrow_stream_bytes(case["schema"], case["rows"])
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        table = read_arrow_stream(result.stdout)
        assert table.schema.names == [self.default_output_column_name], (
            f"unexpected output schema field names: {table.schema.names!r}"
        )
        assert pa.types.is_int64(table.schema.field(self.default_output_column_name).type), (
            f"expected int64 output type, got {table.schema.field(self.default_output_column_name).type!r}"
        )

    @pytest.mark.parametrize(
        "use_input_file, use_output_file",
        [
            pytest.param(False, False, id="stdin_stdout"),
            pytest.param(False, True, id="stdin_output_file"),
            pytest.param(True, False, id="input_file_stdout"),
            pytest.param(True, True, id="input_file_output_file"),
        ],
    )
    def test_hash_transport_combinations_match_reference_algorithm(
        self,
        valid_license_env: dict[str, str],
        tmp_path: Path,
        use_input_file: bool,
        use_output_file: bool,
    ) -> None:
        """All four transport combinations must produce the exact reference-algorithm hash values
        for the same input data and config (contract: Invocation, Data). Complements
        ``BinaryModelConformanceBase.test_all_transport_combinations_produce_identical_output``,
        which checks byte-identical output across transports without asserting on operation
        semantics."""
        case = self.hash_multi_column_case()
        config_path = write_json(tmp_path / "config.json", case["config"])
        input_bytes = arrow_stream_bytes(case["schema"], case["rows"])
        result, output_bytes = run_binary_with_transport(
            self.binary_cmd,
            valid_license_env,
            config_path,
            input_bytes,
            use_input_file=use_input_file,
            use_output_file=use_output_file,
            tmp_path=tmp_path,
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        table = read_arrow_stream(output_bytes)
        assert table.column(self.default_output_column_name).to_pylist() == case["expected"]

    def test_hash_field_order_independent_of_stream_schema_order(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """The stream schema's field order need not match `input_columns`' order: the operation must
        read each row's values by column *name*, in `input_columns` order, not by stream position
        (contract: Data). Config declares `input_columns: ["b", "a"]`, the reverse of the stream
        schema's field order `[a, b]`."""
        input_columns = ["b", "a"]
        config = self.make_config(
            input_columns=input_columns, output_columns={"result": self.default_output_column_name}
        )
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.int64())])
        rows = {"a": [1, 2, 3], "b": [10, 20, 30]}
        input_bytes = arrow_stream_bytes(schema, rows)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        table = read_arrow_stream(result.stdout)
        expected = self.compute_expected_hash_column(rows, input_columns, key=None)
        assert table.column(self.default_output_column_name).to_pylist() == expected

    def test_hash_multi_batch_input_processes_all_batches(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """An IPC stream containing more than one record batch: the combined row count, row order,
        and hash values must all be correct across the batch boundary, catching an implementation
        that only processes the first batch (contract: Data)."""
        input_columns = ["id", "value"]
        config = self.make_config(
            input_columns=input_columns, output_columns={"result": self.default_output_column_name}
        )
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("id", pa.string()), pa.field("value", pa.int64())])
        batch_one: dict[str, list[Any]] = {"id": ["row-0", "row-1"], "value": [1, 2]}
        batch_two: dict[str, list[Any]] = {"id": ["row-2", "row-3", "row-4"], "value": [3, 4, 5]}
        input_bytes = arrow_stream_bytes_multi_batch(schema, [batch_one, batch_two])
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        table = read_arrow_stream(result.stdout)
        combined_rows = {
            "id": batch_one["id"] + batch_two["id"],
            "value": batch_one["value"] + batch_two["value"],
        }
        assert table.num_rows == len(combined_rows["id"]), f"row count mismatch: {table.num_rows}"
        expected = self.compute_expected_hash_column(combined_rows, input_columns, key=None)
        assert table.column(self.default_output_column_name).to_pylist() == expected

    # -------------------------------------------------------------------------------------------
    # H2. `parameters.key` validation and null/falsy handling (contract: Configuration)
    # -------------------------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "bad_key",
        [
            pytest.param(0, id="key_zero_int"),
            pytest.param(False, id="key_false_bool"),
            pytest.param({"a": 1}, id="key_object"),
        ],
    )
    def test_config_parameters_key_non_string_is_usage_error(
        self, valid_license_env: dict[str, str], tmp_path: Path, bad_key: Any
    ) -> None:
        """`parameters.key` for the "hash" operation must be a string (or absent entirely); a
        non-string value -- however falsy -- is an operation-specific parameter-validation usage
        error (contract: Configuration, Errors)."""
        config = self.make_config(parameters={"key": bad_key})
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_hash_key_absent_equals_key_empty_string(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """`parameters.key` entirely absent and `parameters.key: ""` explicitly given both mean "no
        key" and must produce the same hash for the same row data (contract: Configuration). Only
        `key` being entirely absent (`None`) is "no key", not any other falsy value."""
        case_absent = self.hash_multi_column_case(key=None)
        case_empty = self.hash_multi_column_case(key="")
        assert case_absent["expected"] == case_empty["expected"], (
            "test setup: the reference algorithm must already treat absent and empty-string keys identically"
        )
        config_path_absent = write_json(tmp_path / "config_absent.json", case_absent["config"])
        config_path_empty = write_json(tmp_path / "config_empty.json", case_empty["config"])
        input_bytes = arrow_stream_bytes(case_absent["schema"], case_absent["rows"])

        result_absent = run_binary(
            self.binary_cmd, ["run", "--config", str(config_path_absent)], valid_license_env, input_bytes
        )
        assert result_absent.returncode == 0, f"stderr={result_absent.stderr!r}"
        table_absent = read_arrow_stream(result_absent.stdout)

        result_empty = run_binary(
            self.binary_cmd, ["run", "--config", str(config_path_empty)], valid_license_env, input_bytes
        )
        assert result_empty.returncode == 0, f"stderr={result_empty.stderr!r}"
        table_empty = read_arrow_stream(result_empty.stdout)

        assert (
            table_absent.column(self.default_output_column_name).to_pylist()
            == table_empty.column(self.default_output_column_name).to_pylist()
        )

    def test_hash_parameters_rejects_unknown_extra_key(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """NEW: the "hash" operation's `parameters` object rejects an unknown/extra key with a usage
        error (code 1): `parameters` is operation-defined, and an operation must validate its own
        parameter shape exhaustively, not just the keys it recognizes (contract: Configuration).
        Today's stub's parameter validation only inspects `parameters.get("key")` and silently
        ignores any other key present, so this currently fails (the run proceeds instead of being
        rejected)."""
        config = self.make_config(parameters={"key": "x", "unexpected_extra_key": 1})
        config_path = write_json(tmp_path / "config.json", config)
        result = run_binary(self.binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
        assert_error_response(result, USAGE_ERROR)
