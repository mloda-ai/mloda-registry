"""Shared fixtures and helpers for the binary-model conformance kit.

The simulated binary (``simulated_binary.py``) is a pure-Python CLI stand-in for a future
Rust-compiled binary model; it is never published as a package, only used as a private test
fixture. Its test suite is the conformance kit for the binary-model interface contract (see the
contract document in the epic's rust-crate-binary-feature-group folder): the same
``conformance_kit.py`` test functions are meant to run, unmodified, against a real binary later by
overriding the ``binary_cmd`` fixture below, so nothing here may hardcode a binary path.

This module covers only what cycle 1 of that contract needs: invocation surface, the license
gate, and config validation, all of it before any Arrow IPC data is touched. The concrete
``plugin_id`` and operation used throughout ("example_binary" / "hash") are the contract's own
worked example, not a scope limitation of the contract itself.

License tokens: the real signed-token format does not exist yet, so the contract has the binary
accept a fixed placeholder format for now: plain JSON text (not signed) shaped as
``{"status": "valid" | "expired", "plugins": [<plugin_id>, ...]}``. The helpers below build every
state the contract's License section distinguishes: missing, valid, expired, wrong-plugin, and
tampered (unparseable text, or valid JSON missing a required key).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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
) -> subprocess.CompletedProcess[bytes]:
    """Invoke the binary with a fully-controlled argv and environment.

    ``env`` replaces the child's environment outright (never merged with the ambient one), so
    tests are hermetic and never accidentally inherit a license from the calling shell. ``stdin``
    always gets an explicit (possibly empty) byte string so an implementation that reaches the
    data stage never blocks the test suite waiting on an open pipe.
    """
    return subprocess.run(
        [*cmd, *args],
        env=dict(env),
        input=input_bytes,
        capture_output=True,
        timeout=timeout,
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
    (contract: Errors). Returns the parsed error object for further assertions."""
    assert result.returncode == expected_code, (
        f"expected exit code {expected_code}, got {result.returncode}; stderr={result.stderr!r}"
    )
    error = stderr_error_object(result.stderr)
    assert error.get("code") == expected_code, f"error object code mismatch: {error!r}"
    message = error.get("message")
    assert isinstance(message, str) and message, f"error object missing a non-empty message: {error!r}"
    return error


def assert_not_rejected_with(result: subprocess.CompletedProcess[bytes], forbidden_codes: set[int]) -> None:
    """Assert the process did not exit with any of ``forbidden_codes``, without asserting what a
    successful outcome looks like. If it exited non-zero for some other reason, the stderr JSON
    format rule (contract: Errors) still must hold, so a not-yet-implemented stub still fails
    this."""
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
