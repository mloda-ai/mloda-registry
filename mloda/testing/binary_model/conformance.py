"""Reusable, pip-installable binary-model conformance kit (``mloda-testing[binary-model]``):
subprocess/Arrow-IPC plumbing and two class-based conformance suites, for this repository's own
tests and external consumers (wrapper repos, binary vendors, consumer FeatureGroup repos) that
subclass them to run the same checks against their own binary.

- ``BinaryModelConformanceBase``: every contract-generic check. A plain class, not a
  ``unittest.TestCase``, so pytest collects it once subclassed under a ``Test*`` name.
- ``HashOperationConformanceMixin``: every "hash"-operation-specific check, mixed in alongside the
  base class for a binary whose worked example is "hash".
- ``HmacSha256OperationConformanceMixin``: every "hmac_sha256"-operation-specific check (one utf8
  input column, a required 64-hex ``key``, lowercase-hex utf8 output), mixed in the same way.

Every overridable input (binary command, plugin_id, operations, column-type vocabulary, license
fixtures, config shape) is a class attribute or method on ``self``, never a module constant or
pytest fixture, so subclassing with different attributes retargets the whole kit.

Re-exports the lower-level mechanics these classes build on: Arrow IPC stream mechanics
(``arrow.py``), the "hash" and "hmac_sha256" reference algorithms (``hash_reference.py``,
``hmac_sha256_reference.py``), and the signed
license-token vectors and builders (``license_vectors.py``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess  # nosec
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pytest

from mloda.community.feature_groups.binary_model import binary
from mloda.community.feature_groups.binary_model.contract import last_non_empty_stderr_line
from mloda.community.feature_groups.binary_model.errors import BinaryUnavailableError, reported_error
from mloda.community.feature_groups.binary_model.mixin import classify_column_type
from mloda.community.feature_groups.binary_model.transport import minimal_environment
from mloda.testing.binary_model import (
    COLUMN_TYPES,
    DATA_ERROR,
    INTERNAL_ERROR,
    IPC_END_OF_STREAM_MARKER,
    LICENSE_INVALID,
    LICENSE_MISSING,
    MESSAGE_MAX_BYTES,
    UNSUPPORTED,
    USAGE_ERROR,
    hash_reference,
    hmac_sha256_reference,
    license_vectors,
)

# Re-exported via __all__ below for external consumers, unused in this module's own logic now that
# the version/capabilities shape checks delegate to binary.parse_version/parse_capabilities.
from mloda.testing.binary_model import CONTRACT_VERSION as CONTRACT_VERSION
from mloda.testing.binary_model import VERSION_PATTERN as VERSION_PATTERN
from mloda.testing.binary_model.arrow import (
    arrow_file_format_bytes,
    arrow_stream_bytes,
    arrow_stream_bytes_from_arrays,
    arrow_stream_bytes_invalid_utf8,
    arrow_stream_bytes_multi_batch,
    assert_ends_with_ipc_eos_marker,
    assert_output_contract,
    corrupt_record_batch_message_after_schema,
    enumerate_ipc_message_types,
    read_arrow_stream,
)

__all__ = [
    "CONTRACT_VERSION",
    "COLUMN_TYPES",
    "DATA_ERROR",
    "DATA_FREE_MARKER",
    "INTERNAL_ERROR",
    "IPC_END_OF_STREAM_MARKER",
    "LICENSE_INVALID",
    "LICENSE_MISSING",
    "MESSAGE_MAX_BYTES",
    "STDERR_SOFT_CAP_BYTES",
    "UNSUPPORTED",
    "USAGE_ERROR",
    "VERSION_PATTERN",
    "arrow_file_format_bytes",
    "arrow_stream_bytes",
    "arrow_stream_bytes_from_arrays",
    "arrow_stream_bytes_invalid_utf8",
    "arrow_stream_bytes_multi_batch",
    "assert_ends_with_ipc_eos_marker",
    "assert_error_response",
    "assert_not_rejected_with",
    "assert_output_contract",
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
    "HmacSha256OperationConformanceMixin",
]

# A distinctive marker string, chosen so it would never otherwise appear in this suite's input,
# output or diagnostics, used to test that a marked input cell's value never leaks into stderr
# (contract: Data handling).
DATA_FREE_MARKER = "SECRET_MARKER_YlZ9qX7"

# The int64 counterpart of the marker, for binaries without a utf8 column type.
_DATA_FREE_INT_MARKER = 7391046218553917
_UTF8_MARKER_CELL = (pa.string(), DATA_FREE_MARKER, DATA_FREE_MARKER.encode("utf-8"))
_NO_MARKER_SKIP_REASON = (
    "the default input schema's first field is not utf8 or int64: float text is not canonical across languages "
    "and booleans carry no distinctive value"
)

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
    """Invoke the binary with a fully-controlled argv and environment. ``env`` replaces (never
    merges with) the ambient environment, for hermetic tests; ``stdin`` always gets an explicit
    byte string so the process never blocks waiting on an open pipe; ``cwd`` defaults to the
    caller's own (contract: Data handling)."""
    return subprocess.run(  # nosec B603
        [*cmd, *args],
        env=dict(env),
        input=input_bytes,
        capture_output=True,
        timeout=timeout,
        cwd=cwd,
    )


def stderr_error_object(stderr: bytes) -> dict[str, Any]:
    """Parse the last non-empty stderr line as the contract's ``{"code": ..., "message": ...}``
    object; earlier lines are free-form diagnostics (contract: Errors). Uses the mixin's own
    ``contract.last_non_empty_stderr_line``, so this matches its 64 KiB tail window exactly."""
    line = last_non_empty_stderr_line(stderr)
    assert line is not None, f"expected at least one non-empty stderr line, got {stderr!r}"
    try:
        obj = json.loads(line)
    except (ValueError, RecursionError) as exc:
        raise AssertionError(f"last non-empty stderr line is not valid JSON: {line[:200]!r}") from exc
    assert isinstance(obj, dict), f"last non-empty stderr line is not a JSON object: {line!r}"
    return obj


def assert_error_response(result: subprocess.CompletedProcess[bytes], expected_code: int) -> dict[str, Any]:
    """Assert the process exited with ``expected_code`` and stderr's last line is a parseable error
    object with a matching, non-empty ``message`` within the contract's size caps (``message`` <=
    ``MESSAGE_MAX_BYTES`` (1024) UTF-8 bytes, stderr <= ``STDERR_SOFT_CAP_BYTES`` (64 KiB))
    (contract: Errors, Data handling). Returns the parsed error object."""
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
    assert reported_error(result.returncode, result.stderr) is not None, (
        f"error object is not a mixin-parseable error (the mixin would treat it as unparseable): {error!r}"
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
        assert reported_error(result.returncode, result.stderr) is not None, (
            f"error object is not a mixin-parseable error (the mixin would treat it as unparseable): {error!r}"
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


# =============================================================================
# BinaryModelConformanceBase -- contract-generic checks
# =============================================================================


class BinaryModelConformanceBase:
    """Every contract-generic conformance check, plus the class attributes / overridable methods a
    subclass needs to point the whole kit at a different binary/contract.

    Deliberately not a ``unittest.TestCase``: pytest collects ``test_*`` methods on any subclass
    named ``Test*``; this class itself isn't, so it's never collected standalone.

    The default license fixtures are signed with the published test key
    (``license_vectors.TEST_KID``); a binary built with only production keys must override these
    fixture attributes/properties with real licenses (spec: Keys, kid, rotation).
    """

    # -- Class attributes a subclass overrides to point this kit at a different binary/contract --
    binary_cmd: ClassVar[list[str]] = [sys.executable, "-m", "mloda.testing.binary_model.simulated_binary"]
    plugin_id: ClassVar[str] = "example_binary"
    wrong_plugin_id: ClassVar[str] = "some_other_plugin"
    operations: ClassVar[list[str]] = ["hash"]
    reserved_internal_error_operation: ClassVar[str] = "_conformance_internal_error"
    column_types: ClassVar[frozenset[str]] = COLUMN_TYPES
    # Per-invocation subprocess timeout (contract: Invocation); a subclass exercising a slower
    # real binary raises this.
    binary_timeout_seconds: ClassVar[float] = 10.0

    # A generically-valid single-column config shape, used by every test that needs *some* valid
    # config/data but must not hardcode a hash-specific output name (that would leak an
    # operation-specific detail into contract-generic tests).
    default_input_columns: ClassVar[list[str]] = ["col_a"]
    default_output_columns: ClassVar[dict[str, str]] = {"result": "col_a_hash"}

    # A column name distinct from every ``default_input_columns`` entry, for the negative-test
    # variants that need one extra (missing/unexpected/duplicate) input column beyond the default
    # single-column shape.
    extra_input_column: ClassVar[str] = "extra_input_column"

    # None means no limit; an operation taking one column sets 1.
    max_input_columns: ClassVar[int | None] = None

    # License fixture texts (contract: License), computed lazily from ``self.plugin_id``/
    # ``self.wrong_plugin_id`` at the point of use (not at class-definition time), so a subclass
    # overriding ``plugin_id`` alone gets consistent fixtures automatically.
    @property
    def valid_license_text(self) -> str:
        return license_vectors.valid_license_token([self.plugin_id])

    @property
    def expired_license_text(self) -> str:
        return license_vectors.expired_license_token([self.plugin_id])

    @property
    def wrong_plugin_license_text(self) -> str:
        return license_vectors.valid_license_token([self.wrong_plugin_id])

    tampered_unparseable_text: ClassVar[str] = license_vectors.TAMPERED_UNPARSEABLE_TEXT

    @property
    def tampered_signature_text(self) -> str:
        return license_vectors.tampered_signature_token([self.plugin_id])

    missing_plugins_claim_text: ClassVar[str] = license_vectors.missing_plugins_claim_token()

    # ``in_grace_license_text`` is time-relative: an override must return a token currently inside its grace window.
    @property
    def in_grace_license_text(self) -> str:
        return license_vectors.in_grace_license_token([self.plugin_id])

    @property
    def not_yet_valid_license_text(self) -> str:
        return license_vectors.not_yet_valid_license_token([self.plugin_id])

    @property
    def unknown_kid_license_text(self) -> str:
        return license_vectors.unknown_kid_license_token([self.plugin_id])

    # -- Overridable helpers for building a generically-valid config/data case --

    def required_parameters(self) -> dict[str, Any]:
        """The minimal `parameters` ``self.operations[0]`` needs."""
        return {}

    def _kit_two_input_columns_allowed(self) -> bool:
        return self.max_input_columns is None or self.max_input_columns >= 2

    def make_config(
        self,
        *,
        input_columns: list[str] | None = None,
        operation: str | None = None,
        parameters: dict[str, Any] | None = None,
        output_columns: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """A structurally valid ``--config`` document for ``self.operations[0]``, every field
        overridable so a test can mutate exactly the one field under test."""
        if parameters is None:
            parameters = self.required_parameters() if operation is None or operation == self.operations[0] else {}
        return {
            "input_columns": list(self.default_input_columns) if input_columns is None else input_columns,
            "operation": self.operations[0] if operation is None else operation,
            "parameters": parameters,
            "output_columns": dict(self.default_output_columns) if output_columns is None else output_columns,
        }

    def default_input_schema(self) -> pa.Schema:
        """A schema matching ``default_input_columns``, generic enough that any operation over the
        column-type vocabulary can run against it successfully. Must use only types in
        ``self.column_types``: a binary without utf8 overrides this and ``default_input_rows()``.
        Schema and rows must cover every ``default_input_columns`` entry."""
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

    @property
    def default_output_column_key(self) -> str:
        """The single output key ``self.operations[0]`` defines, e.g. ``"result"`` for the "hash"
        worked example; used by generic tests that need the real key rather than a hardcoded one."""
        return next(iter(self.default_output_columns))

    def _kit_run(
        self, args: list[str], env: dict[str, str], input_bytes: bytes = b""
    ) -> subprocess.CompletedProcess[bytes]:
        return run_binary(self.binary_cmd, args, env, input_bytes, timeout=self.binary_timeout_seconds)

    def _kit_run_with_config(
        self,
        config_path: Path,
        env: dict[str, str],
        input_bytes: bytes = b"",
        *,
        extra_args: Sequence[str] = (),
    ) -> subprocess.CompletedProcess[bytes]:
        return self._kit_run(["run", "--config", str(config_path), *extra_args], env, input_bytes)

    def _kit_assert_success_table(
        self, result: subprocess.CompletedProcess[bytes], output_columns: dict[str, str], expected_rows: int
    ) -> pa.Table:
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        table = read_arrow_stream(result.stdout)
        assert_output_contract(table, output_columns, expected_rows, self.column_types)
        return table

    def _kit_assert_licensed_run_succeeds(self, config_path: Path, env: dict[str, str]) -> None:
        rows = self.default_input_rows()
        input_bytes = arrow_stream_bytes(self.default_input_schema(), rows)
        result = self._kit_run_with_config(config_path, env, input_bytes)
        self._kit_assert_success_table(result, self.make_config()["output_columns"], len(next(iter(rows.values()))))

    def _kit_marker_cell(self) -> tuple[pa.DataType, Any, bytes] | None:
        """The marker cell (type, value, stderr needle) for the default input schema's first field, or None."""
        first_type = self.default_input_schema()[0].type
        if classify_column_type(first_type, strict=True) == "utf8":
            return _UTF8_MARKER_CELL
        if pa.types.is_int64(first_type):
            return pa.int64(), _DATA_FREE_INT_MARKER, str(_DATA_FREE_INT_MARKER).encode()
        return None

    # -- Fixtures --

    def platform_env(self, env: dict[str, str]) -> dict[str, str]:
        """Production's minimal environment without license variables, overridden by ``env``.

        An override should build on ``super().platform_env(env)`` so kit runs keep ``PATH``.
        """
        return {**minimal_environment(inherit_license=False), **env}

    @pytest.fixture
    def hermetic_env(self) -> dict[str, str]:
        """Production's minimal environment with no license variables."""
        return self.platform_env({})

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
        return self.platform_env({"MLODA_LICENSE_FILE": str(valid_license_file)})

    # -------------------------------------------------------------------------------------------
    # 1. Invocation surface (contract: Invocation, Capabilities)
    # -------------------------------------------------------------------------------------------

    def test_version_prints_single_line_no_license_required(self, hermetic_env: dict[str, str]) -> None:
        """`--version` prints exactly one `<plugin_id> <semver>` line to stdout and exits 0, with no
        license variables set at all (contract: Invocation). Delegates the line/shape rule to the
        mixin's own ``binary.parse_version``, so the kit checks exactly what the mixin accepts."""
        result = self._kit_run(["--version"], hermetic_env)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        try:
            binary.parse_version(self.binary_cmd, self.plugin_id, result.stdout)
        except BinaryUnavailableError as exc:
            raise AssertionError(str(exc)) from exc

    def test_default_input_schema_uses_advertised_column_types(self) -> None:
        """Fails fast so a missing hook override is not reported as many code-4 failures (contract: Capabilities)."""
        for field in self.default_input_schema():
            classified = classify_column_type(field.type, strict=True)
            if classified is None or classified not in self.column_types:
                raise AssertionError(
                    f"default_input_schema() field {field.name!r} has Arrow type {field.type}, which is not an "
                    f"advertised column type (column_types={sorted(self.column_types)}). Override "
                    "default_input_schema() and default_input_rows() with advertised types, and set column_types "
                    "to what the binary advertises (utf8 must be pa.string(), its wire type)."
                )

    def test_capabilities_prints_single_json_object_no_license_required(self, hermetic_env: dict[str, str]) -> None:
        """Unknown extra keys in the capabilities object are tolerated (contract: Invocation,
        Capabilities). Delegates the shape rule to the mixin's own ``binary.parse_capabilities``,
        so the kit checks exactly what the mixin accepts, then asserts on the kit-specific fixture
        values."""
        result = self._kit_run(["--capabilities"], hermetic_env)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        try:
            capabilities = binary.parse_capabilities(self.binary_cmd, self.plugin_id, result.stdout)
        except BinaryUnavailableError as exc:
            raise AssertionError(str(exc)) from exc

        for op in self.operations:
            assert op in capabilities.operations, f"expected operation {op!r} in {capabilities.operations!r}"
        assert capabilities.column_types == set(self.column_types), (
            f"column_types mismatch: {capabilities.column_types!r}"
        )

    def test_no_arguments_is_usage_error(self, hermetic_env: dict[str, str]) -> None:
        """No license required: a flag-parsing error happens before any license check (contract:
        Invocation)."""
        result = self._kit_run([], hermetic_env)
        assert_error_response(result, USAGE_ERROR)
        assert result.stdout == b"", f"expected no stdout data, got {result.stdout!r}"

    def test_help_flag_is_usage_error(self, hermetic_env: dict[str, str]) -> None:
        """`--help` is a usage error too: the binary is machine-invoked and has no interactive help
        (contract: Invocation)."""
        result = self._kit_run(["--help"], hermetic_env)
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
        result = self._kit_run(args, hermetic_env)
        assert_error_response(result, USAGE_ERROR)
        assert result.stdout == b"", f"expected no stdout data, got {result.stdout!r}"

    @pytest.mark.parametrize(
        "flag",
        [
            pytest.param("--config", id="config"),
            pytest.param("--input", id="input"),
            pytest.param("--output", id="output"),
        ],
    )
    def test_repeated_flag_is_usage_error(
        self, valid_license_env: dict[str, str], valid_config_path: Path, tmp_path: Path, flag: str
    ) -> None:
        """Repeating any of `--config`/`--input`/`--output` is a usage error, not "last one wins"
        (contract: Invocation). Real Arrow IPC data is supplied on stdin in every branch, so a
        regression that drops the repeated-flag check surfaces as an accepted run (exit 0), never
        masked by an unrelated zero-byte-input data error."""
        input_bytes = arrow_stream_bytes(self.default_input_schema(), self.default_input_rows())
        args = ["run", "--config", str(valid_config_path)]
        if flag == "--config":
            args = [*args, "--config", str(valid_config_path)]
        elif flag == "--input":
            input_path = tmp_path / "input.arrows"
            input_path.write_bytes(input_bytes)
            args = [*args, "--input", str(input_path), "--input", str(input_path)]
        else:
            output_path = tmp_path / "out.arrows"
            args = [*args, "--output", str(output_path), "--output", str(output_path)]
        result = self._kit_run(args, valid_license_env, input_bytes)
        assert_error_response(result, USAGE_ERROR)
        assert result.stdout == b"", f"expected no stdout data, got {result.stdout!r}"

    def test_run_without_config_is_usage_error(self, hermetic_env: dict[str, str]) -> None:
        """`run` requires `--config`; without it, usage error (contract: Invocation)."""
        result = self._kit_run(["run"], hermetic_env)
        assert_error_response(result, USAGE_ERROR)

    def test_run_with_nonexistent_config_path_is_usage_error(
        self, hermetic_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A `--config` path that does not exist fails flag parsing (`--config` must exist and be
        readable), before any license check (contract: Invocation, Errors)."""
        missing_path = tmp_path / "does-not-exist.json"
        result = self._kit_run_with_config(missing_path, hermetic_env)
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
        """Neither license variable set to a non-empty value (contract: License)."""
        result = self._kit_run_with_config(valid_config_path, self.platform_env(env))
        assert_error_response(result, LICENSE_MISSING)

    def test_license_missing_when_file_path_nonexistent(self, valid_config_path: Path, tmp_path: Path) -> None:
        """`MLODA_LICENSE_FILE` naming a file that does not exist: exit 2 (contract: License)."""
        env = self.platform_env({"MLODA_LICENSE_FILE": str(tmp_path / "no-such-license.txt")})
        result = self._kit_run_with_config(valid_config_path, env)
        assert_error_response(result, LICENSE_MISSING)

    def test_license_invalid_when_file_path_is_a_directory(self, valid_config_path: Path, tmp_path: Path) -> None:
        """`MLODA_LICENSE_FILE` naming an existing directory, not a file: exit 3, not exit 2 -- the
        path exists, it just can't be read as license content (contract: License)."""
        license_dir = tmp_path / "some-name"
        license_dir.mkdir()
        env = self.platform_env({"MLODA_LICENSE_FILE": str(license_dir)})
        result = self._kit_run_with_config(valid_config_path, env)
        error = assert_error_response(result, LICENSE_INVALID)
        assert "MLODA_LICENSE_FILE" in error["message"], f"message does not name the source: {error!r}"

    def test_license_file_named_pipe_is_license_invalid_not_hung(self, valid_config_path: Path, tmp_path: Path) -> None:
        """`MLODA_LICENSE_FILE` naming a FIFO with no writer must return promptly, never block
        reading an unopened pipe (contract: License, Invocation)."""
        if not hasattr(os, "mkfifo"):
            pytest.skip("os.mkfifo is not available on this platform")
        fifo_path = tmp_path / "license-fifo"
        os.mkfifo(fifo_path)
        env = self.platform_env({"MLODA_LICENSE_FILE": str(fifo_path)})
        result = run_binary(self.binary_cmd, ["run", "--config", str(valid_config_path)], env, timeout=5.0)
        assert_error_response(result, LICENSE_INVALID)

    def test_license_missing_message_names_file_source(self, valid_config_path: Path, tmp_path: Path) -> None:
        """The code 2 `message` names the source that was set (contract: License)."""
        env = self.platform_env({"MLODA_LICENSE_FILE": str(tmp_path / "no-such-license.txt")})
        result = self._kit_run_with_config(valid_config_path, env)
        error = assert_error_response(result, LICENSE_MISSING)
        assert "MLODA_LICENSE_FILE" in error["message"], f"message does not name the source: {error!r}"

    def test_license_accepted_via_license_file(
        self, valid_config_path: Path, valid_license_env: dict[str, str]
    ) -> None:
        """A valid token via `MLODA_LICENSE_FILE` is accepted and the run succeeds with
        contract-conforming output (contract: License)."""
        self._kit_assert_licensed_run_succeeds(valid_config_path, valid_license_env)

    def test_license_accepted_via_license_key_inline(self, valid_config_path: Path) -> None:
        """A valid token via `MLODA_LICENSE_KEY` (inline) is accepted the same as a file and the run
        succeeds (contract: License)."""
        env = self.platform_env({"MLODA_LICENSE_KEY": self.valid_license_text})
        self._kit_assert_licensed_run_succeeds(valid_config_path, env)

    def test_license_file_wins_over_license_key(self, valid_config_path: Path, valid_license_file: Path) -> None:
        """When both are set, `MLODA_LICENSE_FILE` wins with no fallback to `MLODA_LICENSE_KEY`: a
        valid file plus garbage inline key is accepted and the run succeeds (contract: License)."""
        env = self.platform_env(
            {"MLODA_LICENSE_FILE": str(valid_license_file), "MLODA_LICENSE_KEY": "not-json-and-must-not-be-used"}
        )
        self._kit_assert_licensed_run_succeeds(valid_config_path, env)

    def test_license_expired_is_invalid(self, valid_config_path: Path, tmp_path: Path) -> None:
        """An expired token: exit 3, license invalid (contract: License)."""
        license_path = write_text(tmp_path / "license.txt", self.expired_license_text)
        env = self.platform_env({"MLODA_LICENSE_FILE": str(license_path)})
        result = self._kit_run_with_config(valid_config_path, env)
        assert_error_response(result, LICENSE_INVALID)

    def test_license_wrong_plugin_is_invalid(self, valid_config_path: Path) -> None:
        """A token whose `plugins` entitlement list omits this `plugin_id`: exit 3. The message also
        names the source (contract: License)."""
        env = self.platform_env({"MLODA_LICENSE_KEY": self.wrong_plugin_license_text})
        result = self._kit_run_with_config(valid_config_path, env)
        error = assert_error_response(result, LICENSE_INVALID)
        assert "MLODA_LICENSE_KEY" in error["message"], f"message does not name the source: {error!r}"

    @pytest.mark.parametrize(
        "attr_name",
        [
            pytest.param("tampered_unparseable_text", id="unparseable_text"),
            pytest.param("tampered_signature_text", id="tampered_signature"),
            pytest.param("missing_plugins_claim_text", id="missing_plugins_claim"),
        ],
    )
    def test_license_tampered_is_invalid(self, valid_config_path: Path, tmp_path: Path, attr_name: str) -> None:
        """A rejected token body (text that is not a token at all, a broken signature, or a
        well-signed payload missing the required ``plugins`` claim): exit 3 (spec: Verification
        steps 2, 4, 5; contract: License). Parametrized by attribute name, looked up via
        ``getattr`` at test-run time, since the fixtures are instance attributes, not module
        constants."""
        tampered_text = getattr(self, attr_name)
        license_path = write_text(tmp_path / "license.txt", tampered_text)
        env = self.platform_env({"MLODA_LICENSE_FILE": str(license_path)})
        result = self._kit_run_with_config(valid_config_path, env)
        assert_error_response(result, LICENSE_INVALID)

    def test_license_in_grace_is_accepted(self, valid_config_path: Path) -> None:
        """An in-grace token (past ``exp``, inside ``grace_days``) is accepted and the run succeeds
        with contract-conforming output (spec: Verification step 6; contract: License)."""
        env = self.platform_env({"MLODA_LICENSE_KEY": self.in_grace_license_text})
        self._kit_assert_licensed_run_succeeds(valid_config_path, env)

    def test_license_not_yet_valid_is_invalid(self, valid_config_path: Path) -> None:
        """A token whose ``nbf`` lies in the future: exit 3, not yet valid (spec: Verification
        step 6; contract: License)."""
        env = self.platform_env({"MLODA_LICENSE_KEY": self.not_yet_valid_license_text})
        result = self._kit_run_with_config(valid_config_path, env)
        assert_error_response(result, LICENSE_INVALID)

    def test_license_unknown_kid_is_invalid(self, valid_config_path: Path) -> None:
        """A well-signed token under a ``kid`` the verifier's key map does not contain: exit 3
        (spec: Verification step 3; contract: License)."""
        env = self.platform_env({"MLODA_LICENSE_KEY": self.unknown_kid_license_text})
        result = self._kit_run_with_config(valid_config_path, env)
        assert_error_response(result, LICENSE_INVALID)

    def test_license_checked_before_config_valid_license_broken_config(
        self, valid_license_file: Path, tmp_path: Path
    ) -> None:
        """A valid license with a syntactically broken config gets past the license stage and fails
        on config parsing instead: exit 1, never 2 or 3 (contract: Errors, check order)."""
        config_path = write_text(tmp_path / "config.json", "{not valid json")
        env = self.platform_env({"MLODA_LICENSE_FILE": str(valid_license_file)})
        result = self._kit_run_with_config(config_path, env)
        assert_error_response(result, USAGE_ERROR)

    def test_license_checked_before_config_invalid_license_valid_config(
        self, valid_config_path: Path, tmp_path: Path
    ) -> None:
        """An invalid license with an otherwise valid config still exits 2 or 3, not something else,
        proving license is checked before config (contract: Errors, check order)."""
        license_path = write_text(tmp_path / "license.txt", self.expired_license_text)
        env = self.platform_env({"MLODA_LICENSE_FILE": str(license_path)})
        result = self._kit_run_with_config(valid_config_path, env)
        assert result.returncode in (LICENSE_MISSING, LICENSE_INVALID), (
            f"expected 2 or 3, got {result.returncode}; stderr={result.stderr!r}"
        )
        error = stderr_error_object(result.stderr)
        assert error.get("code") == result.returncode, f"error object code mismatch: {error!r}"

    def test_license_file_broken_key_valid_no_fallback_missing_file(
        self, valid_config_path: Path, tmp_path: Path
    ) -> None:
        """`MLODA_LICENSE_FILE` naming a missing file wins over a simultaneously valid
        `MLODA_LICENSE_KEY`: still code 2, no fallback (contract: License)."""
        env = self.platform_env(
            {
                "MLODA_LICENSE_FILE": str(tmp_path / "no-such-license.txt"),
                "MLODA_LICENSE_KEY": self.valid_license_text,
            }
        )
        result = self._kit_run_with_config(valid_config_path, env)
        assert_error_response(result, LICENSE_MISSING)

    def test_license_file_broken_key_valid_no_fallback_tampered_file(
        self, valid_config_path: Path, tmp_path: Path
    ) -> None:
        """`MLODA_LICENSE_FILE` naming a file with tampered content wins over a simultaneously valid
        `MLODA_LICENSE_KEY`: still code 3, no fallback (contract: License)."""
        license_path = write_text(tmp_path / "license.txt", self.tampered_unparseable_text)
        env = self.platform_env({"MLODA_LICENSE_FILE": str(license_path), "MLODA_LICENSE_KEY": self.valid_license_text})
        result = self._kit_run_with_config(valid_config_path, env)
        assert_error_response(result, LICENSE_INVALID)

    # -------------------------------------------------------------------------------------------
    # 3. Config structural validation (contract: Configuration, Errors)
    # -------------------------------------------------------------------------------------------

    def test_config_json_syntax_error(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """A config file that is not valid JSON: exit 1 (contract: Configuration, Errors)."""
        config_path = write_text(tmp_path / "config.json", "{not valid json")
        result = self._kit_run_with_config(config_path, valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_unknown_top_level_key(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """An unknown top-level config key: exit 1 (contract: Configuration)."""
        config = self.make_config()
        config["unexpected_top_level_key"] = "value"
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    @pytest.mark.parametrize("missing_key", ["input_columns", "operation", "parameters", "output_columns"])
    def test_config_missing_required_key(
        self, valid_license_env: dict[str, str], tmp_path: Path, missing_key: str
    ) -> None:
        """Each of the four required top-level keys is mandatory; missing any one is exit 1 (contract:
        Configuration)."""
        config = self.make_config()
        del config[missing_key]
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_input_columns_empty(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """`input_columns` must name at least one column: an empty list is exit 1 (contract:
        Configuration)."""
        config = self.make_config(input_columns=[])
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_input_columns_duplicate(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """`input_columns` without duplicates: a repeated name is exit 1 (contract: Configuration)."""
        column = self.default_input_columns[0]
        config = self.make_config(input_columns=[column, column])
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_output_columns_written_names_not_unique(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Written names in `output_columns` must be unique among themselves; this is checked
        structurally, before the operation's own output list is consulted (contract: Configuration,
        Errors)."""
        output_key = self.default_output_column_key
        config = self.make_config(output_columns={output_key: "dup_name", "an_extra_output_key": "dup_name"})
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_output_columns_collides_with_input_columns(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Every written output name must be distinct from every `input_columns` entry: colliding with
        one is exit 1 (contract: Configuration)."""
        column = self.default_input_columns[0]
        output_key = self.default_output_column_key
        config = self.make_config(input_columns=[column], output_columns={output_key: column})
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_operation_not_in_capabilities(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """An `operation` outside `capabilities.operations` (and not the reserved conformance-only
        operation) is unsupported: exit 4 (contract: Configuration, Errors)."""
        config = self.make_config(operation="not_a_real_operation")
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, UNSUPPORTED)

    def test_reserved_internal_error_operation_not_rejected_as_unknown(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """The reserved internal-error operation is not listed in `capabilities.operations` but must
        still be accepted at the operation-capability-check step, bypassing that check specifically
        for this literal string, so the kit can provoke code 6 on demand (contract: Conformance)."""
        config = self.make_config(operation=self.reserved_internal_error_operation)
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_not_rejected_with(result, {UNSUPPORTED})

    def test_config_output_columns_missing_operation_output(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """`output_columns` must map every output the operation defines; an empty mapping is missing
        it: exit 1, checked after the operation check (contract: Configuration, Errors)."""
        config = self.make_config(output_columns={})
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_config_output_columns_extra_unmapped_output(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """An output name the operation does not define is also exit 1, checked after the operation
        check (contract: Configuration, Errors)."""
        output_key = self.default_output_column_key
        config = self.make_config(
            output_columns={output_key: self.default_output_column_name, "not_a_real_output": "col_b_out"}
        )
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_unknown_operation_with_bad_output_columns_reports_operation_error_first(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """The operation-capability check (code 4) runs before the output_columns completeness check
        (code 1): an unknown operation combined with an incomplete output mapping is exit 4, not exit
        1 (contract: Errors, check order)."""
        config = self.make_config(operation="not_a_real_operation", output_columns={})
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, UNSUPPORTED)

    def test_config_parameters_empty_object_accepted_structurally(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """`parameters` holding only the required keys (`{}` when all are optional) is structurally
        accepted: this alone does not cause exit 1 (contract: Configuration)."""
        config = self.make_config(parameters=self.required_parameters())
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert result.returncode != USAGE_ERROR, (
            f"minimal required parameters unexpectedly caused a usage error; stderr={result.stderr!r}"
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
        """All four transport combinations must produce equivalent output (same schema, row count,
        row order) for the same input and config (contract: Invocation, Data). Compares parsed
        tables rather than raw bytes since "batch boundaries may differ" (contract: Data) is still
        conforming. Contract-generic; hash-specific values are covered by
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
            timeout=self.binary_timeout_seconds,
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
            timeout=self.binary_timeout_seconds,
        )
        assert baseline_result.returncode == 0, f"stderr={baseline_result.stderr!r}"
        table = read_arrow_stream(output_bytes)
        expected_rows = len(next(iter(self.default_input_rows().values())))
        assert_output_contract(table, config["output_columns"], expected_rows, self.column_types)
        baseline_table = read_arrow_stream(baseline_output)
        assert table.schema.equals(baseline_table.schema), (
            f"transport combination produced a different schema: {table.schema!r} vs {baseline_table.schema!r}"
        )
        assert table.num_rows == baseline_table.num_rows, (
            f"transport combination produced a different row count: {table.num_rows} vs {baseline_table.num_rows}"
        )
        for name in table.schema.names:
            assert table.column(name).to_pylist() == baseline_table.column(name).to_pylist(), (
                f"transport combination produced different values for column {name!r}"
            )

    def test_zero_row_batches_do_not_change_the_result(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """Zero-row batches interleaved in one stream must not change the output (contract: Data);
        the mixin sends them for chunked frames."""
        schema = self.default_input_schema()
        rows = self.default_input_rows()
        total = len(next(iter(rows.values())))
        if total == 0:
            pytest.skip("needs at least one default input row to split around zero-row batches")
        split = total // 2
        first = {name: values[:split] for name, values in rows.items()}
        rest = {name: values[split:] for name, values in rows.items()}
        empty: dict[str, list[Any]] = {name: [] for name in schema.names}
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        chunked = arrow_stream_bytes_multi_batch(schema, [empty, first, empty, rest, empty])
        assert enumerate_ipc_message_types(chunked).count("record batch") == 5, (
            "test setup: the stream must carry five record-batch messages"
        )
        chunked_result = self._kit_run_with_config(config_path, valid_license_env, chunked)
        chunked_table = self._kit_assert_success_table(chunked_result, config["output_columns"], total)
        baseline_result = self._kit_run_with_config(config_path, valid_license_env, arrow_stream_bytes(schema, rows))
        baseline_table = self._kit_assert_success_table(baseline_result, config["output_columns"], total)
        for name in config["output_columns"].values():
            assert chunked_table.column(name).to_pylist() == baseline_table.column(name).to_pylist(), (
                f"zero-row batches changed the values of output column {name!r}"
            )

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
            timeout=self.binary_timeout_seconds,
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert result.stdout == b"", f"expected empty stdout when --output is used, got {result.stdout!r}"
        table = read_arrow_stream(output_bytes)
        expected_rows = len(next(iter(self.default_input_rows().values())))
        assert_output_contract(table, config["output_columns"], expected_rows, self.column_types)

    def test_input_file_given_ignores_stdin(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """`--input <path>` means stdin is never read (contract: Invocation): garbage fed on stdin
        alongside a valid `--input` file must not prevent success, since a binary that read stdin
        instead would fail on the garbage -- success alone proves stdin was ignored."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        input_bytes = arrow_stream_bytes(self.default_input_schema(), self.default_input_rows())
        input_path = tmp_path / "input.arrows"
        input_path.write_bytes(input_bytes)
        garbage_stdin = b"this-is-not-an-arrow-stream-and-must-be-ignored" * 4
        result = self._kit_run_with_config(
            config_path, valid_license_env, garbage_stdin, extra_args=["--input", str(input_path)]
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
        result = self._kit_run_with_config(
            valid_config_path, valid_license_env, extra_args=["--input", str(input_path)]
        )
        assert_error_response(result, DATA_ERROR)

    # -------------------------------------------------------------------------------------------
    # 5. Exact-column-set rule and column type vocabulary (contract: Data, Capabilities)
    # -------------------------------------------------------------------------------------------

    def test_input_schema_missing_column_is_data_error(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """The input stream's schema must contain exactly `input_columns`; a missing name is a data
        error (contract: Data)."""
        column = self.default_input_columns[0]
        column_type = self.default_input_schema().field(0).type
        values = self.default_input_rows()[column]
        if self._kit_two_input_columns_allowed():
            config = self.make_config(input_columns=[column, self.extra_input_column])
            present = column  # extra column missing
        else:
            config = self.make_config(input_columns=[column])
            present = self.extra_input_column  # the configured column missing
        config_path = write_json(tmp_path / "config.json", config)
        input_bytes = arrow_stream_bytes(pa.schema([pa.field(present, column_type)]), {present: values})
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert_error_response(result, DATA_ERROR)

    def test_input_schema_extra_column_is_data_error(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """An extra column beyond `input_columns` is a data error (contract: Data)."""
        column = self.default_input_columns[0]
        config = self.make_config(input_columns=[column])
        config_path = write_json(tmp_path / "config.json", config)
        column_type = self.default_input_schema().field(0).type
        values = self.default_input_rows()[column]
        schema = pa.schema([pa.field(column, column_type), pa.field(self.extra_input_column, column_type)])
        input_bytes = arrow_stream_bytes(schema, {column: values, self.extra_input_column: values})
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert_error_response(result, DATA_ERROR)

    def test_input_schema_duplicate_field_name_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """The one configured column carried twice (two Arrow fields sharing a name) is a "duplicate",
        a data error (contract: Data)."""
        column = self.default_input_columns[0]
        config_path = write_json(tmp_path / "config.json", self.make_config(input_columns=[column]))
        column_type = self.default_input_schema().field(0).type
        schema = pa.schema([pa.field(column, column_type), pa.field(column, column_type)])
        values = pa.array(self.default_input_rows()[column], type=column_type)
        input_bytes = arrow_stream_bytes_from_arrays(schema, [values, values])
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
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
        column = self.default_input_columns[0]
        config = self.make_config(input_columns=[column])
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field(column, bad_type)])
        input_bytes = arrow_stream_bytes(schema, {column: sample_values})
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert_error_response(result, UNSUPPORTED)

    def test_input_column_large_string_type_is_rejected_as_unsupported(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Only bare `pa.string()` counts as this contract's `utf8`; `pa.large_string()` is a
        distinct Arrow type outside the vocabulary and must be rejected as an unsupported column
        type (code 4) (contract: Capabilities)."""
        column = self.default_input_columns[0]
        config = self.make_config(input_columns=[column])
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field(column, pa.large_string())])
        input_bytes = arrow_stream_bytes(schema, {column: ["alpha", "beta"]})
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert_error_response(result, UNSUPPORTED)

    def test_input_column_string_view_type_is_rejected_as_unsupported(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Same rule for `pa.string_view()` (contract: Capabilities): outside the vocabulary,
        rejected as an unsupported column type (code 4)."""
        column = self.default_input_columns[0]
        config = self.make_config(input_columns=[column])
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field(column, pa.string_view())])
        input_bytes = arrow_stream_bytes(schema, {column: ["alpha", "beta"]})
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert_error_response(result, UNSUPPORTED)

    def test_input_schema_presence_error_precedes_type_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A presence violation (a missing column, or an extra field when one column is configured)
        plus a wrongly typed field is a data error, not code 4: presence is checked first
        (contract: Data)."""
        column = self.default_input_columns[0]
        config_path = write_json(tmp_path / "config.json", self.make_config(input_columns=[column]))
        fields = [pa.field(column, pa.int32()), pa.field(self.extra_input_column, pa.int32())]
        rows: dict[str, list[Any]] = {column: [1, 2], self.extra_input_column: [1, 2]}
        if self._kit_two_input_columns_allowed():
            # extra_input_column missing; column also wrong type
            config_path = write_json(
                tmp_path / "config.json", self.make_config(input_columns=[column, self.extra_input_column])
            )
            fields, rows = fields[:1], {column: [1, 2]}
        input_bytes = arrow_stream_bytes(pa.schema(fields), rows)
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert_error_response(result, DATA_ERROR)

    def test_dictionary_encoded_column_is_unsupported_type(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A dictionary-encoded column is an unsupported column type (code 4), decided by the same
        type check as any other type outside the vocabulary (contract: Data)."""
        column = self.default_input_columns[0]
        config = self.make_config(input_columns=[column])
        config_path = write_json(tmp_path / "config.json", config)
        dict_array = pa.array(["x", "y", "x"]).dictionary_encode()
        schema = pa.schema([pa.field(column, dict_array.type)])
        input_bytes = arrow_stream_bytes_from_arrays(schema, [dict_array])
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
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
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        table = self._kit_assert_success_table(result, config["output_columns"], 0)
        assert table.schema.field(self.default_output_column_name).type == self.default_output_column_type()

    def test_schema_only_input_bad_license_still_rejected(self, valid_config_path: Path, tmp_path: Path) -> None:
        """A schema-only input with a bad license still exits 2 or 3, not 0: the license check applies
        before any data is read, schema-only input included (contract: Data, License)."""
        input_bytes = arrow_stream_bytes(self.default_input_schema(), None)
        license_path = write_text(tmp_path / "license.txt", self.expired_license_text)
        env = self.platform_env({"MLODA_LICENSE_FILE": str(license_path)})
        result = self._kit_run_with_config(valid_config_path, env, input_bytes)
        assert result.returncode in (LICENSE_MISSING, LICENSE_INVALID), (
            f"expected 2 or 3, got {result.returncode}; stderr={result.stderr!r}"
        )
        error = stderr_error_object(result.stderr)
        assert error.get("code") == result.returncode, f"error object code mismatch: {error!r}"

    def test_schema_only_output_has_no_record_batch_message(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A schema-only output must carry no record-batch message at all, not one with zero rows
        (contract: Data). ``read_all()`` can't distinguish the two wire shapes, so this enumerates
        the raw output's message sequence instead."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        input_bytes = arrow_stream_bytes(self.default_input_schema(), None)
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
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
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert_ends_with_ipc_eos_marker(result.stdout)

    # -------------------------------------------------------------------------------------------
    # 8. Malformed input rejections (contract: Data)
    # -------------------------------------------------------------------------------------------

    def test_zero_byte_input_is_data_error(self, valid_config_path: Path, valid_license_env: dict[str, str]) -> None:
        """Zero bytes is not an Arrow IPC stream at all: a data error (contract: Data)."""
        result = self._kit_run_with_config(valid_config_path, valid_license_env, b"")
        assert_error_response(result, DATA_ERROR)

    def test_truncated_stream_missing_end_of_stream_marker_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """ "truncated" means end of file without the end-of-stream marker, not "no more batches": a
        stream cut off right before that marker is a data error (contract: Data)."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        schema = self.default_input_schema()
        full_bytes = arrow_stream_bytes(schema, self.default_input_rows())
        assert full_bytes.endswith(IPC_END_OF_STREAM_MARKER), "test setup: expected the writer to emit the EOS marker"
        truncated = full_bytes[: -len(IPC_END_OF_STREAM_MARKER)]
        result = self._kit_run_with_config(config_path, valid_license_env, truncated)
        assert_error_response(result, DATA_ERROR)

    def test_truncated_stream_with_unsupported_column_type_reports_type_error_first(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A stream cut off right before the end-of-stream marker, whose single column is typed
        outside the vocabulary, is unsupported (code 4), not a data error: the vocabulary check
        must run before the truncation check (contract: Data, Capabilities)."""
        column = self.default_input_columns[0]
        config = self.make_config(input_columns=[column])
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field(column, pa.int32())])
        full_bytes = arrow_stream_bytes(schema, {column: [1, 2, 3]})
        assert full_bytes.endswith(IPC_END_OF_STREAM_MARKER), "test setup: expected the writer to emit the EOS marker"
        truncated = full_bytes[: -len(IPC_END_OF_STREAM_MARKER)]
        result = self._kit_run_with_config(config_path, valid_license_env, truncated)
        assert_error_response(result, UNSUPPORTED)

    def test_ipc_file_format_instead_of_stream_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """The IPC file/Feather format (`ARROW1` magic bytes) is not the streaming format the contract
        requires: a data error (contract: Data)."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        schema = self.default_input_schema()
        input_bytes = arrow_file_format_bytes(schema, self.default_input_rows())
        assert input_bytes[:6] == b"ARROW1", "test setup: expected the IPC file magic at the start"
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert_error_response(result, DATA_ERROR)

    def test_compressed_record_batch_body_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A compressed record batch body is a data error (contract: Data)."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        schema = self.default_input_schema()
        options = pa.ipc.IpcWriteOptions(compression="lz4")
        input_bytes = arrow_stream_bytes(schema, self.default_input_rows(), options=options)
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert_error_response(result, DATA_ERROR)

    def test_malformed_record_batch_after_valid_schema_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Corrupted record-batch data *after* a valid schema message is a data error (contract:
        Data): distinct from malformed bytes from the very start, since the schema parses but
        reading the batch fails on the corrupted message."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        schema = self.default_input_schema()
        valid_bytes = arrow_stream_bytes(schema, self.default_input_rows())
        corrupted = corrupt_record_batch_message_after_schema(valid_bytes)
        result = self._kit_run_with_config(config_path, valid_license_env, corrupted)
        assert_error_response(result, DATA_ERROR)

    def test_output_path_is_existing_directory_is_usage_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """`--output` naming an existing directory (not a file) is a usage error, exit 1: "opening
        --input and creating --output (code 1 if either fails)" (contract: Errors)."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        output_dir = tmp_path / "output_is_a_directory"
        output_dir.mkdir()
        result = self._kit_run_with_config(config_path, valid_license_env, extra_args=["--output", str(output_dir)])
        assert_error_response(result, USAGE_ERROR)

    def test_input_path_not_readable_is_usage_error(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """An `--input` path that exists but is not readable (chmod 000) is a usage error, exit 1
        (contract: Errors). Skipped without POSIX file permissions (e.g. Windows), or when running
        as root, which ignores file read permissions."""
        if not hasattr(os, "geteuid"):
            pytest.skip("no POSIX file permissions on this platform")
        if os.geteuid() == 0:
            pytest.skip("running as root ignores file read permissions")
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        schema = self.default_input_schema()
        input_path = tmp_path / "input.arrows"
        input_path.write_bytes(arrow_stream_bytes(schema, self.default_input_rows()))
        original_mode = input_path.stat().st_mode
        input_path.chmod(0o000)
        try:
            result = self._kit_run_with_config(config_path, valid_license_env, extra_args=["--input", str(input_path)])
        finally:
            input_path.chmod(original_mode)
        assert_error_response(result, USAGE_ERROR)

    # -------------------------------------------------------------------------------------------
    # 9. The reserved internal-error operation (contract: Conformance)
    # -------------------------------------------------------------------------------------------

    def test_reserved_internal_error_operation_with_data_triggers_code_6(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """The reserved internal-error operation, given valid config/input, reaches the data stage
        and deliberately produces code 6, with stderr still a valid error object, not a bare
        traceback (contract: Conformance)."""
        config = self.make_config(operation=self.reserved_internal_error_operation)
        config_path = write_json(tmp_path / "config.json", config)
        schema = self.default_input_schema()
        input_bytes = arrow_stream_bytes(schema, self.default_input_rows())
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert_error_response(result, INTERNAL_ERROR)

    def test_internal_error_message_reports_only_exception_class_name(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A code-6 message names only the exception's class, never its unbounded `str()` (contract:
        Data handling, Errors). The contract doesn't mandate a specific wrapper text, so this only
        checks the message's final token is a bare identifier shape, not the exact wording. Uses
        `output_columns={}`: the reserved operation produces no normal output (contract:
        Conformance)."""
        config = self.make_config(operation=self.reserved_internal_error_operation, output_columns={})
        config_path = write_json(tmp_path / "config.json", config)
        schema = self.default_input_schema()
        input_bytes = arrow_stream_bytes(schema, self.default_input_rows())
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        error = assert_error_response(result, INTERNAL_ERROR)
        message = error["message"]
        assert message, "expected a non-empty code-6 message"
        class_name_token = message.rsplit(maxsplit=1)[-1]
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", class_name_token), (
            f"expected the message to end in a bare exception-class-name token, got {message!r}"
        )

    # -------------------------------------------------------------------------------------------
    # 10. Trailing-data / concatenated-stream rejection (contract: Data)
    # -------------------------------------------------------------------------------------------

    def test_two_concatenated_ipc_streams_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Two complete, valid Arrow IPC streams concatenated back-to-back must be rejected as a data
        error, not silently accepted with only the first stream's rows returned (contract: Data)."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        schema = self.default_input_schema()
        stream_one = arrow_stream_bytes(schema, self.default_input_rows())
        stream_two = arrow_stream_bytes(schema, self.default_input_rows())
        result = self._kit_run_with_config(config_path, valid_license_env, stream_one + stream_two)
        assert_error_response(result, DATA_ERROR)

    def test_trailing_garbage_bytes_after_eos_marker_is_data_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Valid stream bytes followed by arbitrary trailing garbage (not another full stream) after
        the end-of-stream marker is equally a data error (contract: Data)."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        schema = self.default_input_schema()
        valid_bytes = arrow_stream_bytes(schema, self.default_input_rows())
        garbage = b"trailing-garbage-bytes-not-a-stream-0123456789"
        assert garbage[-len(IPC_END_OF_STREAM_MARKER) :] != IPC_END_OF_STREAM_MARKER, (
            "test setup: the garbage suffix must not coincidentally end with the EOS marker"
        )
        result = self._kit_run_with_config(config_path, valid_license_env, valid_bytes + garbage)
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
        marker = self._kit_marker_cell()
        if marker is None:
            pytest.skip(_NO_MARKER_SKIP_REASON)
        marker_type, marker_value, needle = marker
        column = self.default_input_columns[0]
        config = self.make_config(input_columns=[column])
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field(column, marker_type)])
        input_bytes = arrow_stream_bytes(schema, {column: [marker_value]})
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert needle not in result.stderr, (
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
        """The marker cell value through two failing cases that still reach the data stage: a data
        error from a wrong schema, and the reserved internal-error operation. Neither leaks the
        marker into stderr (contract: Data handling, Conformance)."""
        marker = self._kit_marker_cell() or _UTF8_MARKER_CELL
        marker_type, marker_value, needle = marker
        column = self.default_input_columns[0]
        missing_column = case == "missing_column_data_error"
        if missing_column:
            input_columns = [column, "col_b_missing"] if self._kit_two_input_columns_allowed() else [column]
            operation = self.operations[0]
            output_columns = self.default_output_columns
            data_column = column if self._kit_two_input_columns_allowed() else self.extra_input_column
        else:
            input_columns = [column]
            operation = self.reserved_internal_error_operation
            output_columns = {}
            data_column = column
        config = self.make_config(input_columns=input_columns, operation=operation, output_columns=output_columns)
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field(data_column, marker_type)])
        input_bytes = arrow_stream_bytes(schema, {data_column: [marker_value]})
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        if missing_column:
            assert_error_response(result, DATA_ERROR)
        else:
            assert result.returncode != 0, f"expected this case to fail, got exit 0; stdout={result.stdout!r}"
        assert needle not in result.stderr, f"marker cell value leaked into stderr: {result.stderr!r}"

    def test_error_message_stays_under_size_cap_for_long_garbage_operation(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A very long garbage `operation` string, echoed back into the error message, still stays
        within the contract's 1024-byte cap via `assert_error_response`'s shared size-cap assertion
        (contract: Data handling, Conformance)."""
        long_garbage_operation = "not_a_real_operation_" + "x" * 2000
        config = self.make_config(operation=long_garbage_operation)
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, UNSUPPORTED)

    def test_no_network_dependency_under_unshare_net(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """A normal run wrapped in `unshare --net` must still succeed, proving the binary makes no
        network calls it depends on (contract: Data handling, Conformance). Skipped if
        `unshare --net` is not usable in this sandbox."""
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
        result = run_binary(
            ["unshare", "--net", "--", *self.binary_cmd],
            ["run", "--config", str(config_path)],
            valid_license_env,
            input_bytes,
            timeout=self.binary_timeout_seconds,
        )
        self._kit_assert_success_table(result, config["output_columns"], len(next(iter(rows.values()))))

    def test_no_files_created_outside_output_in_read_only_cwd(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """A normal run with its cwd set to a fresh read-only directory must still succeed and
        create no files there, proving the binary writes nothing outside `--output` (contract: Data
        handling, Conformance). Skipped without POSIX directory permissions, or as root."""
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
                timeout=self.binary_timeout_seconds,
            )
        finally:
            readonly_cwd.chmod(original_mode)

        self._kit_assert_success_table(result, config["output_columns"], len(next(iter(rows.values()))))
        leftover = list(readonly_cwd.iterdir())
        assert leftover == [], f"expected no files created in the read-only cwd, found {leftover!r}"

    def test_minimal_environment_allowlist_only(self, tmp_path: Path) -> None:
        """A normal run with only production's minimal environment plus the license key must still succeed,
        proving the binary reads no other environment variable (contract: Data handling, Conformance)."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        rows = self.default_input_rows()
        input_bytes = arrow_stream_bytes(self.default_input_schema(), rows)
        env = self.platform_env({"MLODA_LICENSE_KEY": self.valid_license_text})
        result = self._kit_run_with_config(config_path, env, input_bytes)
        self._kit_assert_success_table(result, config["output_columns"], len(next(iter(rows.values()))))

    # -------------------------------------------------------------------------------------------
    # 12. Corrected error classification (contract: Errors, License, Configuration, Data)
    # -------------------------------------------------------------------------------------------

    def test_config_invalid_utf8_bytes_is_usage_error(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """A `--config` file containing bytes that are not valid UTF-8 is a usage error, exit 1:
        config decode/parse failures are usage errors (contract: Errors, Invocation), never code 6."""
        config_path = tmp_path / "config.json"
        config_path.write_bytes(b"\xff\xfe\x00invalid-utf8-config")
        result = self._kit_run_with_config(config_path, valid_license_env)
        assert_error_response(result, USAGE_ERROR)

    def test_license_file_invalid_utf8_bytes_is_license_invalid(self, valid_config_path: Path, tmp_path: Path) -> None:
        """A file named by `MLODA_LICENSE_FILE` containing bytes that are not valid UTF-8 is code 3,
        not code 2 and not code 6: the source is set and the file exists, but its content cannot be
        read as the license token (contract: License)."""
        license_path = tmp_path / "license.txt"
        license_path.write_bytes(b"\xff\xfe\x00invalid-utf8-license")
        env = self.platform_env({"MLODA_LICENSE_FILE": str(license_path)})
        result = self._kit_run_with_config(valid_config_path, env)
        assert_error_response(result, LICENSE_INVALID)

    # -------------------------------------------------------------------------------------------
    # 13. Arrow metadata handling (contract: Data)
    # -------------------------------------------------------------------------------------------

    def test_input_arrow_metadata_schema_and_field_level_accepted_and_stripped_from_output(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Arrow metadata (schema- and field-level) on the INPUT stream is accepted, and stripped
        entirely from the OUTPUT stream (contract: Data). Asserts the input actually carries
        non-empty metadata first, ruling out a trivial always-empty-output pass."""
        base_schema = self.default_input_schema()
        fields = [field.with_metadata({b"field_meta_key": b"field_meta_value"}) for field in base_schema]
        schema = pa.schema(fields).with_metadata({b"schema_meta_key": b"schema_meta_value"})
        assert schema.metadata, "test setup: expected the input schema to carry schema-level metadata"
        assert schema.field(0).metadata, "test setup: expected the input field to carry field-level metadata"
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        rows = self.default_input_rows()
        input_bytes = arrow_stream_bytes(schema, rows)
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert result.returncode == 0, f"input metadata unexpectedly caused a failure; stderr={result.stderr!r}"
        table = read_arrow_stream(result.stdout)
        assert_output_contract(table, config["output_columns"], len(next(iter(rows.values()))), self.column_types)
        assert not table.schema.metadata, f"output schema unexpectedly carries metadata: {table.schema.metadata!r}"
        for output_field in table.schema:
            assert not output_field.metadata, (
                f"output field {output_field.name!r} unexpectedly carries metadata: {output_field.metadata!r}"
            )


# =============================================================================
# HashOperationConformanceMixin -- "hash" operation-specific checks
# =============================================================================


class HashOperationConformanceMixin(BinaryModelConformanceBase):
    """Every check specific to the "hash" operation's shape. Mixed in alongside
    ``BinaryModelConformanceBase`` to assemble a full conformance suite for a binary whose worked
    example is "hash".

    Delegates the reference algorithm to ``mloda.testing.binary_model.hash_reference``, the single
    implementation also imported by ``simulated_binary.py``, so the two can never silently drift
    apart. A subclass may list only this class, or both explicitly with the mixin first
    (``class T(HashOperationConformanceMixin, BinaryModelConformanceBase)``) -- never the reverse
    order, which fails Python's MRO resolution."""

    operations: ClassVar[list[str]] = ["hash"]
    default_output_columns: ClassVar[dict[str, str]] = {"result": "hash_out"}

    # -- The "hash" operation's reference algorithm (independent of whatever the binary does) --

    compute_expected_hash = staticmethod(hash_reference.compute_expected_hash)
    compute_expected_hash_column = staticmethod(hash_reference.compute_expected_hash_column)

    def hash_multi_column_case(self, key: str | None = None) -> dict[str, Any]:
        """Build one self-contained "hash" test case: a small multi-column, multi-row dataset (every
        vocabulary type, one null) with its Arrow schema, a config built via ``self.make_config``,
        and the expected output computed independently via ``compute_expected_hash_column``
        (contract: Configuration "hash" operation shape). ``key`` is forwarded to both the config
        and the independent computation. ``id`` varies per row so a row-order bug is caught even
        though the hash also depends on every other column; ``amount`` is null on one row to
        exercise the null-sentinel path."""
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
        output_columns = {"result": self.default_output_column_name}
        parameters: dict[str, Any] = {} if key is None else {"key": key}
        config = self.make_config(input_columns=input_columns, parameters=parameters, output_columns=output_columns)
        expected = hash_reference.compute_expected_hash_column(rows, input_columns, key)
        return {
            "input_columns": input_columns,
            "rows": rows,
            "schema": schema,
            "output_columns": output_columns,
            "config": config,
            "expected": expected,
        }

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
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        table = self._kit_assert_success_table(result, case["config"]["output_columns"], len(case["expected"]))
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
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        table = self._kit_assert_success_table(result, case["config"]["output_columns"], len(case["expected"]))
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
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        table = self._kit_assert_success_table(result, case["config"]["output_columns"], len(case["rows"]["id"]))
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
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        table = self._kit_assert_success_table(result, case["config"]["output_columns"], len(case["rows"]["id"]))
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
        (contract: Invocation, Data). Complements
        ``BinaryModelConformanceBase.test_all_transport_combinations_produce_identical_output``,
        which checks output equivalence without asserting operation semantics."""
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
            timeout=self.binary_timeout_seconds,
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        table = read_arrow_stream(output_bytes)
        assert_output_contract(table, case["config"]["output_columns"], len(case["expected"]), self.column_types)
        assert table.column(self.default_output_column_name).to_pylist() == case["expected"]

    def test_hash_field_order_independent_of_stream_schema_order(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """The stream schema's field order need not match `input_columns`: values are read by
        column name in `input_columns` order, not stream position (contract: Data). Config declares
        `input_columns: ["b", "a"]`, reversed from the schema's `[a, b]`."""
        input_columns = ["b", "a"]
        config = self.make_config(
            input_columns=input_columns, output_columns={"result": self.default_output_column_name}
        )
        config_path = write_json(tmp_path / "config.json", config)
        schema = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.int64())])
        rows = {"a": [1, 2, 3], "b": [10, 20, 30]}
        input_bytes = arrow_stream_bytes(schema, rows)
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        table = self._kit_assert_success_table(result, config["output_columns"], len(rows["a"]))
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
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        combined_rows = {
            "id": batch_one["id"] + batch_two["id"],
            "value": batch_one["value"] + batch_two["value"],
        }
        table = self._kit_assert_success_table(result, config["output_columns"], len(combined_rows["id"]))
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
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
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

        result_absent = self._kit_run_with_config(config_path_absent, valid_license_env, input_bytes)
        table_absent = self._kit_assert_success_table(
            result_absent, case_absent["config"]["output_columns"], len(case_absent["expected"])
        )

        result_empty = self._kit_run_with_config(config_path_empty, valid_license_env, input_bytes)
        table_empty = self._kit_assert_success_table(
            result_empty, case_empty["config"]["output_columns"], len(case_empty["expected"])
        )

        assert (
            table_absent.column(self.default_output_column_name).to_pylist()
            == table_empty.column(self.default_output_column_name).to_pylist()
        )

    def test_hash_parameters_rejects_unknown_extra_key(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """The "hash" operation's `parameters` object rejects an unknown/extra key with a usage
        error (code 1): `parameters` is operation-defined, and an operation must validate its own
        parameter shape exhaustively, not just the keys it recognizes (contract: Configuration)."""
        config = self.make_config(parameters={"key": "x", "unexpected_extra_key": 1})
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, USAGE_ERROR)


# =============================================================================
# HmacSha256OperationConformanceMixin -- "hmac_sha256" operation-specific checks
# =============================================================================


class HmacSha256OperationConformanceMixin(BinaryModelConformanceBase):
    """Every check specific to "hmac_sha256": a required 64-hex ``key``, one utf8 input column, lowercase-hex
    utf8 output, nulls stay null. Subclass order matches ``HashOperationConformanceMixin``: mixin first."""

    operations: ClassVar[list[str]] = ["hmac_sha256"]
    default_output_columns: ClassVar[dict[str, str]] = {"result": "col_a_token"}
    max_input_columns: ClassVar[int | None] = 1
    hmac_key: ClassVar[str] = "42" * 32

    def required_parameters(self) -> dict[str, Any]:
        return {hmac_sha256_reference.KEY_PARAMETER: self.hmac_key}

    def default_output_column_type(self) -> pa.DataType:
        return pa.string()

    def _assert_key_not_echoed(self, result: subprocess.CompletedProcess[bytes], key: str) -> None:
        """The key never appears in stderr, nor in stdout when the run failed."""
        assert key.encode() not in result.stderr, "stderr echoes the key"
        if result.returncode != 0:
            assert key.encode() not in result.stdout, "stdout echoes the key"

    def _hmac_run(
        self, env: dict[str, str], tmp_path: Path, rows: list[str | None], key: str | None = None
    ) -> pa.Table:
        """Run the default config (optionally with ``key``) over one utf8 column and return the output table."""
        parameters = None if key is None else {hmac_sha256_reference.KEY_PARAMETER: key}
        config = self.make_config(parameters=parameters)
        column = self.default_input_columns[0]
        input_bytes = arrow_stream_bytes(pa.schema([pa.field(column, pa.string())]), {column: rows})
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), env, input_bytes)
        self._assert_key_not_echoed(result, key or self.hmac_key)
        return self._kit_assert_success_table(result, config["output_columns"], len(rows))

    # -------------------------------------------------------------------------------------------
    # M1. "hmac_sha256" values (contract: Configuration "hmac_sha256" operation shape; Data)
    # -------------------------------------------------------------------------------------------

    def test_hmac_known_answer(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """A fixed key and value give the literal known digest (contract: Configuration)."""
        table = self._hmac_run(
            valid_license_env,
            tmp_path,
            [hmac_sha256_reference.KNOWN_ANSWER_VALUE],
            key=hmac_sha256_reference.KNOWN_ANSWER_KEY,
        )
        expected = [hmac_sha256_reference.KNOWN_ANSWER_DIGEST]
        assert table.column(self.default_output_column_name).to_pylist() == expected

    def test_hmac_matches_reference(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """Output matches the reference row for row; nulls stay null, digests are 64 lowercase hex (contract: Data)."""
        rows: list[str | None] = ["alice", None, "", "über", "alice"]
        actual = self._hmac_run(valid_license_env, tmp_path, rows).column(self.default_output_column_name).to_pylist()
        assert actual == hmac_sha256_reference.compute_expected_hmac_sha256_column(rows, self.hmac_key)
        assert actual[1] is None
        assert actual[0] == actual[4]
        for token in (value for value in actual if value is not None):
            assert re.fullmatch(r"[0-9a-f]{64}", token), f"not 64 lowercase hex characters: {token!r}"

    def test_hmac_is_deterministic(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """The same input run twice gives byte-identical stdout (contract: Data)."""
        config = self.make_config()
        config_path = write_json(tmp_path / "config.json", config)
        column = self.default_input_columns[0]
        input_bytes = arrow_stream_bytes(pa.schema([pa.field(column, pa.string())]), {column: ["alice", None, "bob"]})
        first = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        second = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert first.returncode == 0, f"stderr={first.stderr!r}"
        assert first.stdout == second.stdout
        self._assert_key_not_echoed(first, self.hmac_key)
        self._assert_key_not_echoed(second, self.hmac_key)

    def test_hmac_multi_batch_input_processes_all_batches(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Two record batches give the reference values across the boundary (contract: Data)."""
        config = self.make_config()
        column = self.default_input_columns[0]
        schema = pa.schema([pa.field(column, pa.string())])
        batch_one: list[str | None] = ["alice", None]
        batch_two: list[str | None] = ["bob", "über", "alice"]
        input_bytes = arrow_stream_bytes_multi_batch(schema, [{column: batch_one}, {column: batch_two}])
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env, input_bytes)
        self._assert_key_not_echoed(result, self.hmac_key)
        table = self._kit_assert_success_table(result, config["output_columns"], len(batch_one) + len(batch_two))
        expected = hmac_sha256_reference.compute_expected_hmac_sha256_column(batch_one + batch_two, self.hmac_key)
        assert table.column(self.default_output_column_name).to_pylist() == expected

    def test_hmac_different_key_changes_every_token(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """A different key changes every non-null token (contract: Configuration)."""
        rows: list[str | None] = ["alice", None, "bob"]
        first = self._hmac_run(valid_license_env, tmp_path, rows).column(self.default_output_column_name).to_pylist()
        other_key = "43" * 32
        second = (
            self._hmac_run(valid_license_env, tmp_path, rows, key=other_key)
            .column(self.default_output_column_name)
            .to_pylist()
        )
        assert second[1] is None
        for before, after in zip(first, second):
            assert before == after if before is None else before != after

    def test_hmac_uppercase_hex_key_equals_lowercase(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """An uppercase hex key gives the same tokens as its lowercase form (contract: Configuration)."""
        key = "ab" * 32
        rows: list[str | None] = ["alice", None, "bob"]
        lower = self._hmac_run(valid_license_env, tmp_path, rows, key=key)
        upper = self._hmac_run(valid_license_env, tmp_path, rows, key=key.upper())
        name = self.default_output_column_name
        assert upper.column(name).to_pylist() == lower.column(name).to_pylist()

    def test_hmac_output_schema_exact_type_utf8_no_input_columns_echoed(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Output schema is exactly the written name, typed utf8; no input column is echoed (contract: Data)."""
        table = self._hmac_run(valid_license_env, tmp_path, ["alice", "bob"])
        assert table.schema.names == [self.default_output_column_name]
        assert pa.types.is_string(table.schema.field(self.default_output_column_name).type)

    # -------------------------------------------------------------------------------------------
    # M2. Input and `parameters.key` rejection (contract: Configuration, Capabilities, Errors)
    # -------------------------------------------------------------------------------------------

    def test_hmac_non_utf8_input_column_is_unsupported(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """An int64 input column is rejected as unsupported (contract: Capabilities)."""
        column = self.default_input_columns[0]
        config_path = write_json(tmp_path / "config.json", self.make_config())
        input_bytes = arrow_stream_bytes(pa.schema([pa.field(column, pa.int64())]), {column: [1, 2]})
        result = self._kit_run_with_config(config_path, valid_license_env, input_bytes)
        assert_error_response(result, UNSUPPORTED)
        self._assert_key_not_echoed(result, self.hmac_key)

    def test_hmac_two_input_columns_is_usage_error(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """ "hmac_sha256" takes exactly one input column (contract: Configuration)."""
        config = self.make_config(input_columns=[self.default_input_columns[0], self.extra_input_column])
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, USAGE_ERROR)
        self._assert_key_not_echoed(result, self.hmac_key)

    @pytest.mark.parametrize(
        "bad_key",
        [
            pytest.param("11" * 31 + "1", id="63_hex_chars"),
            pytest.param("11" * 32 + "1", id="65_hex_chars"),
            pytest.param("zz" * 32, id="non_hex_chars"),
            pytest.param("11 " * 21 + "1", id="embedded_spaces"),
            pytest.param(1234, id="non_string"),
            pytest.param(None, id="null"),
            pytest.param("", id="empty"),
        ],
    )
    def test_hmac_bad_key_is_usage_error_and_never_echoed(
        self, valid_license_env: dict[str, str], tmp_path: Path, bad_key: Any
    ) -> None:
        """A key that is not exactly 64 hex characters is a usage error, and stderr never contains it
        (contract: Configuration, Errors, Data handling)."""
        config = self.make_config(parameters={hmac_sha256_reference.KEY_PARAMETER: bad_key})
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, USAGE_ERROR)
        if isinstance(bad_key, str) and bad_key:
            self._assert_key_not_echoed(result, bad_key)

    def test_hmac_unknown_extra_parameter_is_usage_error(
        self, valid_license_env: dict[str, str], tmp_path: Path
    ) -> None:
        """An unknown parameter next to a valid key is a usage error (contract: Configuration)."""
        config = self.make_config(
            parameters={hmac_sha256_reference.KEY_PARAMETER: self.hmac_key, "unexpected_extra_key": 1}
        )
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, USAGE_ERROR)
        self._assert_key_not_echoed(result, self.hmac_key)

    def test_hmac_missing_key_is_usage_error(self, valid_license_env: dict[str, str], tmp_path: Path) -> None:
        """`parameters: {}` lacks the required key, a usage error (contract: Configuration)."""
        config = self.make_config(parameters={})
        result = self._kit_run_with_config(write_json(tmp_path / "config.json", config), valid_license_env)
        assert_error_response(result, USAGE_ERROR)
