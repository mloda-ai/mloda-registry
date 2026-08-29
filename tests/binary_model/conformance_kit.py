"""Conformance kit, cycle 1: invocation surface, license gate, config validation.

These ``test_*`` functions are not collected directly (this file's name does not match pytest's
``test_*.py`` / ``*_test.py`` discovery pattern) -- a wiring module imports them by name (or with
``import *``) so pytest collects them as part of that module instead, using whatever
``binary_cmd`` fixture is in scope there. ``test_simulated_binary.py`` in this same directory
wires this kit to our own ``simulated_binary.py`` via the ``binary_cmd`` fixture in
``conftest.py``. A future conformance run against a real binary (the wrapper, or the end-to-end
run) is meant to reuse these same functions unmodified by supplying its own ``binary_cmd`` fixture
instead: nothing here may hardcode a binary path.

Scope: this cycle covers everything before any Arrow IPC data is touched -- the three documented
invocations, the license gate, and config structural/capability validation. Data flow, operation
output correctness, transport combinations, and process-hygiene/sandbox checks belong to later
cycles and are not covered here.

Every check below cites the interface contract section it comes from; see that document (in the
epic's rust-crate-binary-feature-group folder) for the authoritative text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.binary_model.conftest import (
    CAPABILITY_OPERATIONS,
    COLUMN_TYPES,
    CONTRACT_VERSION,
    EXPIRED_LICENSE_TEXT,
    PLUGIN_ID,
    RESERVED_INTERNAL_ERROR_OPERATION,
    TAMPERED_MISSING_PLUGINS_TEXT,
    TAMPERED_MISSING_STATUS_TEXT,
    TAMPERED_UNPARSEABLE_TEXT,
    VALID_LICENSE_TEXT,
    WRONG_PLUGIN_LICENSE_TEXT,
    assert_error_response,
    assert_not_rejected_with,
    make_config,
    run_binary,
    stderr_error_object,
    write_json,
    write_text,
)

# ---------------------------------------------------------------------------
# 1. Invocation surface (contract: Invocation, Capabilities)
# ---------------------------------------------------------------------------

_VERSION_PATTERN = re.compile(rf"^{re.escape(PLUGIN_ID)} \d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?$")


def test_version_prints_single_line_no_license_required(binary_cmd: list[str], hermetic_env: dict[str, str]) -> None:
    """`--version` prints exactly one `<plugin_id> <semver>` line to stdout and exits 0, with no
    license variables set at all (contract: Invocation)."""
    result = run_binary(binary_cmd, ["--version"], hermetic_env)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    stdout_text = result.stdout.decode("utf-8")
    lines = stdout_text.splitlines()
    assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
    assert _VERSION_PATTERN.fullmatch(lines[0]), f"line does not match '<plugin_id> <semver>': {lines[0]!r}"


def test_capabilities_prints_single_json_object_no_license_required(
    binary_cmd: list[str], hermetic_env: dict[str, str]
) -> None:
    """`--capabilities` prints exactly one JSON object followed by at most one trailing newline,
    nothing else, and exits 0, with no license variables set. The required keys (`contract`,
    `plugin_id`, `operations`, `column_types`) are present and correct; unknown extra keys are
    tolerated (contract: Invocation, Capabilities)."""
    result = run_binary(binary_cmd, ["--capabilities"], hermetic_env)
    assert result.returncode == 0, f"stderr={result.stderr!r}"

    stdout = result.stdout
    assert stdout.count(b"\n") <= 1, f"expected at most one trailing newline, got {stdout!r}"
    body_bytes = stdout[:-1] if stdout.endswith(b"\n") else stdout
    assert b"\n" not in body_bytes, f"expected exactly one JSON object on stdout, got {stdout!r}"

    body = json.loads(body_bytes.decode("utf-8"))
    assert isinstance(body, dict), f"expected a JSON object, got {body!r}"
    assert body.get("contract") == CONTRACT_VERSION, f"unexpected contract value: {body!r}"
    assert body.get("plugin_id") == PLUGIN_ID, f"unexpected plugin_id: {body!r}"
    operations = body.get("operations")
    assert isinstance(operations, list), f"operations must be a list: {body!r}"
    for op in CAPABILITY_OPERATIONS:
        assert op in operations, f"expected operation {op!r} in {operations!r}"
    column_types = body.get("column_types")
    assert isinstance(column_types, list), f"column_types must be a list: {body!r}"
    assert set(column_types) == set(COLUMN_TYPES), f"column_types mismatch: {column_types!r}"


def test_no_arguments_is_usage_error(binary_cmd: list[str], hermetic_env: dict[str, str]) -> None:
    """No arguments at all is a usage error: exit 1, no stdout, and no license required since a
    flag-parsing error happens before any license check (contract: Invocation)."""
    result = run_binary(binary_cmd, [], hermetic_env)
    assert_error_response(result, 1)
    assert result.stdout == b"", f"expected no stdout data, got {result.stdout!r}"


def test_help_flag_is_usage_error(binary_cmd: list[str], hermetic_env: dict[str, str]) -> None:
    """`--help` is a usage error too: the binary is machine-invoked and has no interactive help
    (contract: Invocation)."""
    result = run_binary(binary_cmd, ["--help"], hermetic_env)
    assert_error_response(result, 1)
    assert result.stdout == b"", f"expected no stdout data, got {result.stdout!r}"


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--bogus-flag"], id="unknown_flag"),
        pytest.param(["--version", "--capabilities"], id="conflicting_flags"),
        pytest.param(["--capabilities", "extra-positional"], id="extra_positional_after_capabilities"),
    ],
)
def test_unrecognized_flag_combination_is_usage_error(
    binary_cmd: list[str], hermetic_env: dict[str, str], args: list[str]
) -> None:
    """Any argument combination beyond the three documented invocations is a usage error
    (contract: Invocation)."""
    result = run_binary(binary_cmd, args, hermetic_env)
    assert_error_response(result, 1)
    assert result.stdout == b"", f"expected no stdout data, got {result.stdout!r}"


def test_run_without_config_is_usage_error(binary_cmd: list[str], hermetic_env: dict[str, str]) -> None:
    """`run` requires `--config`; without it, usage error (contract: Invocation)."""
    result = run_binary(binary_cmd, ["run"], hermetic_env)
    assert_error_response(result, 1)


def test_run_with_nonexistent_config_path_is_usage_error(
    binary_cmd: list[str], hermetic_env: dict[str, str], tmp_path: Path
) -> None:
    """A `--config` path that does not exist fails flag parsing (`--config` must exist and be
    readable), before any license check (contract: Invocation, Errors)."""
    missing_path = tmp_path / "does-not-exist.json"
    result = run_binary(binary_cmd, ["run", "--config", str(missing_path)], hermetic_env)
    assert_error_response(result, 1)


# ---------------------------------------------------------------------------
# 2. License gate (contract: License, Errors)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env",
    [
        pytest.param({}, id="unset"),
        pytest.param({"MLODA_LICENSE_FILE": "", "MLODA_LICENSE_KEY": ""}, id="empty_strings"),
    ],
)
def test_license_missing_when_no_source_set(
    binary_cmd: list[str], valid_config_path: Path, env: dict[str, str]
) -> None:
    """Neither license variable set to a non-empty value: exit 2, license missing (contract:
    License)."""
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env)
    assert_error_response(result, 2)


def test_license_missing_when_file_path_nonexistent(
    binary_cmd: list[str], valid_config_path: Path, tmp_path: Path
) -> None:
    """`MLODA_LICENSE_FILE` naming a file that does not exist: exit 2 (contract: License)."""
    env = {"MLODA_LICENSE_FILE": str(tmp_path / "no-such-license.txt")}
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env)
    assert_error_response(result, 2)


def test_license_missing_message_names_file_source(
    binary_cmd: list[str], valid_config_path: Path, tmp_path: Path
) -> None:
    """The code 2 `message` names the source that was set (contract: License: "the error message
    names the source and the reason")."""
    env = {"MLODA_LICENSE_FILE": str(tmp_path / "no-such-license.txt")}
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env)
    error = assert_error_response(result, 2)
    assert "MLODA_LICENSE_FILE" in error["message"], f"message does not name the source: {error!r}"


def test_license_accepted_via_license_file(
    binary_cmd: list[str], valid_config_path: Path, valid_license_env: dict[str, str]
) -> None:
    """A valid token via `MLODA_LICENSE_FILE` proceeds past the license check; whatever happens
    next (config/data stages, out of scope for this cycle) is never code 2 or 3 (contract:
    License)."""
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], valid_license_env)
    assert_not_rejected_with(result, {2, 3})


def test_license_accepted_via_license_key_inline(binary_cmd: list[str], valid_config_path: Path) -> None:
    """A valid token via `MLODA_LICENSE_KEY` (inline) is accepted the same as a file (contract:
    License)."""
    env = {"MLODA_LICENSE_KEY": VALID_LICENSE_TEXT}
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env)
    assert_not_rejected_with(result, {2, 3})


def test_license_file_wins_over_license_key(
    binary_cmd: list[str], valid_config_path: Path, valid_license_file: Path
) -> None:
    """When both are set, `MLODA_LICENSE_FILE` wins with no fallback to `MLODA_LICENSE_KEY`: a
    valid file plus garbage inline key is still accepted (contract: License: "there is no
    fallback from a set but unusable `MLODA_LICENSE_FILE` to `MLODA_LICENSE_KEY`")."""
    env = {"MLODA_LICENSE_FILE": str(valid_license_file), "MLODA_LICENSE_KEY": "not-json-and-must-not-be-used"}
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env)
    assert_not_rejected_with(result, {2, 3})


def test_license_expired_is_invalid(binary_cmd: list[str], valid_config_path: Path, tmp_path: Path) -> None:
    """An expired token: exit 3, license invalid (contract: License)."""
    license_path = write_text(tmp_path / "license.txt", EXPIRED_LICENSE_TEXT)
    env = {"MLODA_LICENSE_FILE": str(license_path)}
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env)
    assert_error_response(result, 3)


def test_license_wrong_plugin_is_invalid(binary_cmd: list[str], valid_config_path: Path) -> None:
    """A token whose `plugins` entitlement list omits this `plugin_id`: exit 3. The message also
    names the source (contract: License)."""
    env = {"MLODA_LICENSE_KEY": WRONG_PLUGIN_LICENSE_TEXT}
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env)
    error = assert_error_response(result, 3)
    assert "MLODA_LICENSE_KEY" in error["message"], f"message does not name the source: {error!r}"


@pytest.mark.parametrize(
    "tampered_text",
    [
        pytest.param(TAMPERED_UNPARSEABLE_TEXT, id="unparseable_json"),
        pytest.param(TAMPERED_MISSING_STATUS_TEXT, id="missing_status_key"),
        pytest.param(TAMPERED_MISSING_PLUGINS_TEXT, id="missing_plugins_key"),
    ],
)
def test_license_tampered_is_invalid(
    binary_cmd: list[str], valid_config_path: Path, tmp_path: Path, tampered_text: str
) -> None:
    """A tampered token (unparseable text, or valid JSON missing a required key): exit 3
    (contract: License)."""
    license_path = write_text(tmp_path / "license.txt", tampered_text)
    env = {"MLODA_LICENSE_FILE": str(license_path)}
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env)
    assert_error_response(result, 3)


def test_license_checked_before_config_valid_license_broken_config(
    binary_cmd: list[str], valid_license_file: Path, tmp_path: Path
) -> None:
    """A valid license with a syntactically broken config gets past the license stage and fails
    on config parsing instead: exit 1, never 2 or 3 (contract: Errors, check order: "license;
    config parse and structural validation")."""
    config_path = write_text(tmp_path / "config.json", "{not valid json")
    env = {"MLODA_LICENSE_FILE": str(valid_license_file)}
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], env)
    assert_error_response(result, 1)


def test_license_checked_before_config_invalid_license_valid_config(
    binary_cmd: list[str], valid_config_path: Path, tmp_path: Path
) -> None:
    """An invalid license with an otherwise valid config still exits 2 or 3, not something else,
    proving license is checked before config (contract: Errors, check order)."""
    license_path = write_text(tmp_path / "license.txt", EXPIRED_LICENSE_TEXT)
    env = {"MLODA_LICENSE_FILE": str(license_path)}
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env)
    assert result.returncode in (2, 3), f"expected 2 or 3, got {result.returncode}; stderr={result.stderr!r}"
    error = stderr_error_object(result.stderr)
    assert error.get("code") == result.returncode, f"error object code mismatch: {error!r}"


# ---------------------------------------------------------------------------
# 3. Config validation (contract: Configuration, Errors)
# ---------------------------------------------------------------------------


def test_config_json_syntax_error(binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path) -> None:
    """A config file that is not valid JSON: exit 1 (contract: Configuration, Errors)."""
    config_path = write_text(tmp_path / "config.json", "{not valid json")
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, 1)


def test_config_unknown_top_level_key(binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path) -> None:
    """An unknown top-level config key: exit 1 (contract: Configuration)."""
    config = make_config()
    config["unexpected_top_level_key"] = "value"
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, 1)


@pytest.mark.parametrize("missing_key", ["input_columns", "operation", "parameters", "output_columns"])
def test_config_missing_required_key(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path, missing_key: str
) -> None:
    """Each of the four required top-level keys is mandatory; missing any one is exit 1 (contract:
    Configuration)."""
    config = make_config()
    del config[missing_key]
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, 1)


def test_config_input_columns_empty(binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path) -> None:
    """`input_columns` must name at least one column: an empty list is exit 1 (contract:
    Configuration)."""
    config = make_config(input_columns=[])
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, 1)


def test_config_input_columns_duplicate(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """`input_columns` without duplicates: a repeated name is exit 1 (contract: Configuration)."""
    config = make_config(input_columns=["col_a", "col_a"])
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, 1)


def test_config_output_columns_written_names_not_unique(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """Written names in `output_columns` must be unique among themselves; this is checked
    structurally, before the operation's own output list is consulted, so it does not depend on
    whether the extra key is a real output of "hash" (contract: Configuration, Errors)."""
    config = make_config(output_columns={"result": "dup_name", "an_extra_output_key": "dup_name"})
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, 1)


def test_config_output_columns_collides_with_input_columns(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """Every written output name must be distinct from every `input_columns` entry: colliding with
    one is exit 1 (contract: Configuration)."""
    config = make_config(input_columns=["col_a"], output_columns={"result": "col_a"})
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, 1)


def test_config_operation_not_in_capabilities(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """An `operation` outside `capabilities.operations` (and not the reserved conformance-only
    operation) is unsupported: exit 4 (contract: Configuration, Errors)."""
    config = make_config(operation="not_a_real_operation")
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, 4)


def test_reserved_internal_error_operation_not_rejected_as_unknown(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """The reserved `_conformance_internal_error` operation is not listed in
    `capabilities.operations` but must still be accepted at the operation-capability-check step,
    bypassing that check specifically for this literal string, so the kit can provoke code 6 on
    demand (contract: Conformance). Actually triggering code 6 from it needs data flow and belongs
    to cycle 3; here we only assert it is not rejected as an unknown operation (never code 4),
    whatever the run does afterward."""
    config = make_config(operation=RESERVED_INTERNAL_ERROR_OPERATION)
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_not_rejected_with(result, {4})


def test_config_output_columns_missing_operation_output(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """`output_columns` must map every output the operation defines; "hash" defines exactly one
    ("result"), so an empty mapping is missing it: exit 1, checked after the operation check
    (contract: Configuration, Errors)."""
    config = make_config(output_columns={})
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, 1)


def test_config_output_columns_extra_unmapped_output(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """An output name the operation does not define is also exit 1, checked after the operation
    check (contract: Configuration, Errors)."""
    config = make_config(output_columns={"result": "col_a_hash", "not_a_real_output": "col_b_out"})
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, 1)


def test_unknown_operation_with_bad_output_columns_reports_operation_error_first(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """The operation-capability check (code 4) runs before the output_columns completeness check
    (code 1): an unknown operation combined with an incomplete output mapping is exit 4, not exit
    1 (contract: Errors, check order)."""
    config = make_config(operation="not_a_real_operation", output_columns={})
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, 4)


def test_config_parameters_empty_object_accepted_structurally(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """`parameters: {}` is structurally accepted for an operation whose parameters are all
    optional ("hash"'s `key` is optional): this does not itself cause exit 1 (contract:
    Configuration). Final success is out of scope for this cycle, so nothing else is asserted."""
    config = make_config(parameters={})
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert result.returncode != 1, (
        f"empty parameters object unexpectedly caused a usage error; stderr={result.stderr!r}"
    )
