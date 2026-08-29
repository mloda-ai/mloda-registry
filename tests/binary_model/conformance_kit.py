"""Conformance kit: invocation surface, license gate, config validation, the Arrow IPC data
pipeline, and the contract's "Data handling" / remaining "Errors" diagnostics rules.

These ``test_*`` functions are not collected directly (this file's name does not match pytest's
``test_*.py`` / ``*_test.py`` discovery pattern) -- a wiring module imports them by name (or with
``import *``) so pytest collects them as part of that module instead, using whatever
``binary_cmd`` fixture is in scope there. ``test_simulated_binary.py`` in this same directory
wires this kit to our own ``simulated_binary.py`` via the ``binary_cmd`` fixture in
``conftest.py``. A future conformance run against a real binary is meant to reuse these same
functions unmodified by supplying its own ``binary_cmd`` fixture instead: nothing here may
hardcode a binary path.

Every check below cites the interface contract section it comes from.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess  # nosec
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from tests.binary_model.conftest import (
    CAPABILITY_OPERATIONS,
    COLUMN_TYPES,
    CONTRACT_VERSION,
    DATA_ERROR,
    DATA_FREE_MARKER,
    EXPIRED_LICENSE_TEXT,
    INTERNAL_ERROR,
    IPC_END_OF_STREAM_MARKER,
    PLUGIN_ID,
    RESERVED_INTERNAL_ERROR_OPERATION,
    TAMPERED_MISSING_PLUGINS_TEXT,
    TAMPERED_MISSING_STATUS_TEXT,
    TAMPERED_UNPARSEABLE_TEXT,
    UNSUPPORTED,
    VALID_LICENSE_TEXT,
    WRONG_PLUGIN_LICENSE_TEXT,
    arrow_file_format_bytes,
    arrow_stream_bytes,
    arrow_stream_bytes_from_arrays,
    arrow_stream_bytes_multi_batch,
    assert_ends_with_ipc_eos_marker,
    assert_error_response,
    assert_not_rejected_with,
    compute_expected_hash_column,
    corrupt_record_batch_message_after_schema,
    enumerate_ipc_message_types,
    hash_multi_column_case,
    make_config,
    read_arrow_stream,
    run_binary,
    run_binary_with_transport,
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
    """The code 2 `message` names the source that was set (contract: License)."""
    env = {"MLODA_LICENSE_FILE": str(tmp_path / "no-such-license.txt")}
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env)
    error = assert_error_response(result, 2)
    assert "MLODA_LICENSE_FILE" in error["message"], f"message does not name the source: {error!r}"


def test_license_accepted_via_license_file(
    binary_cmd: list[str], valid_config_path: Path, valid_license_env: dict[str, str]
) -> None:
    """A valid token via `MLODA_LICENSE_FILE` proceeds past the license check; whatever happens
    next is never code 2 or 3 (contract: License)."""
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
    valid file plus garbage inline key is still accepted (contract: License)."""
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
    on config parsing instead: exit 1, never 2 or 3 (contract: Errors, check order)."""
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
    demand (contract: Conformance). Actually triggering code 6 from it needs data flow, checked
    elsewhere; here we only assert it is not rejected as an unknown operation (never code 4),
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
    Configuration). Final success is out of scope here, so nothing else is asserted."""
    config = make_config(parameters={})
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert result.returncode != 1, (
        f"empty parameters object unexpectedly caused a usage error; stderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# 4. The "hash" operation reference algorithm (contract: Configuration "hash" operation shape;
#    the concrete algorithm itself is defined in conftest.compute_expected_hash, not the contract,
#    since operations are opaque to mloda)
# ---------------------------------------------------------------------------


def test_hash_multi_column_with_null_matches_reference_algorithm(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """ "hash" with no `parameters.key`, multiple input columns covering every vocabulary type, and
    a null in one column (`amount`) matches the independent reference computation
    (`compute_expected_hash`) byte for byte, row for row (contract: Configuration "hash" shape;
    Data)."""
    case = hash_multi_column_case()
    config_path = write_json(tmp_path / "config.json", case["config"])
    input_bytes = arrow_stream_bytes(case["schema"], case["rows"])
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    table = read_arrow_stream(result.stdout)
    assert table.num_rows == len(case["expected"]), f"row count mismatch: {table.num_rows}"
    assert table.column("hash_out").to_pylist() == case["expected"]


def test_hash_with_key_parameter_matches_reference_algorithm_and_changes_result(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """ "hash" with a `parameters.key` produces the reference algorithm's value computed with that
    key, which differs from the no-key result for the same rows (contract: Configuration; the key
    is folded into the digest per `compute_expected_hash`)."""
    key = "s3cr3t-key"
    case = hash_multi_column_case(key=key)
    config_path = write_json(tmp_path / "config.json", case["config"])
    input_bytes = arrow_stream_bytes(case["schema"], case["rows"])
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    table = read_arrow_stream(result.stdout)
    actual = table.column("hash_out").to_pylist()
    assert actual == case["expected"]

    no_key_case = hash_multi_column_case(key=None)
    assert actual != no_key_case["expected"], "the parameters.key must change the hash result"


def test_hash_row_count_and_row_order_preserved(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """Row count and row order are preserved: distinct `id` values per row make the expected hash
    order-sensitive, so a binary that drops, adds or reorders rows fails this even if it computes
    correct hashes for the wrong positions (contract: Data, Conformance)."""
    case = hash_multi_column_case()
    config_path = write_json(tmp_path / "config.json", case["config"])
    input_bytes = arrow_stream_bytes(case["schema"], case["rows"])
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    table = read_arrow_stream(result.stdout)
    assert table.num_rows == len(case["rows"]["id"])
    actual = table.column("hash_out").to_pylist()
    for row_index, (expected_value, row_id) in enumerate(zip(case["expected"], case["rows"]["id"])):
        assert actual[row_index] == expected_value, (
            f"row {row_index} ({row_id!r}) hash mismatch: expected {expected_value}, got {actual[row_index]}"
        )


def test_hash_output_schema_exact_type_int64_no_input_columns_echoed(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """Output schema is exactly the `output_columns` written names ("hash_out"), typed int64; no
    input column is echoed (contract: Data)."""
    case = hash_multi_column_case()
    config_path = write_json(tmp_path / "config.json", case["config"])
    input_bytes = arrow_stream_bytes(case["schema"], case["rows"])
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    table = read_arrow_stream(result.stdout)
    assert table.schema.names == ["hash_out"], f"unexpected output schema field names: {table.schema.names!r}"
    assert pa.types.is_int64(table.schema.field("hash_out").type), (
        f"expected int64 output type, got {table.schema.field('hash_out').type!r}"
    )


# ---------------------------------------------------------------------------
# 5. Transport combinations (contract: Invocation, Data)
# ---------------------------------------------------------------------------


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
    binary_cmd: list[str],
    valid_license_env: dict[str, str],
    tmp_path: Path,
    use_input_file: bool,
    use_output_file: bool,
) -> None:
    """All four transport combinations (`--input`/stdin crossed with `--output`/stdout) must work
    identically for the same input data and config (contract: Invocation, Data)."""
    case = hash_multi_column_case()
    config_path = write_json(tmp_path / "config.json", case["config"])
    input_bytes = arrow_stream_bytes(case["schema"], case["rows"])
    result, output_bytes = run_binary_with_transport(
        binary_cmd,
        valid_license_env,
        config_path,
        input_bytes,
        use_input_file=use_input_file,
        use_output_file=use_output_file,
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    table = read_arrow_stream(output_bytes)
    assert table.column("hash_out").to_pylist() == case["expected"]


# ---------------------------------------------------------------------------
# 6. Exact-column-set rule and column type vocabulary (contract: Data)
# ---------------------------------------------------------------------------


def test_input_schema_missing_column_is_data_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """The input stream's schema must contain exactly `input_columns`; a missing name is a data
    error (contract: Data)."""
    config = make_config(input_columns=["col_a", "col_b"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int64())])  # col_b missing entirely
    input_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2]})
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert_error_response(result, DATA_ERROR)


def test_input_schema_extra_column_is_data_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """An extra column beyond `input_columns` is a data error (contract: Data)."""
    config = make_config(input_columns=["col_a", "col_b"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int64()), pa.field("col_b", pa.int64()), pa.field("col_c", pa.int64())])
    input_bytes = arrow_stream_bytes(schema, {"col_a": [1], "col_b": [2], "col_c": [3]})
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert_error_response(result, DATA_ERROR)


def test_input_schema_duplicate_field_name_is_data_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """Two distinct Arrow fields sharing the same name is a "duplicate", a data error, even though
    the set of distinct names equals `input_columns` (contract: Data)."""
    config = make_config(input_columns=["col_a", "col_b"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int64()), pa.field("col_a", pa.int64()), pa.field("col_b", pa.int64())])
    input_bytes = arrow_stream_bytes_from_arrays(schema, [pa.array([1, 2]), pa.array([3, 4]), pa.array([5, 6])])
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert_error_response(result, DATA_ERROR)


@pytest.mark.parametrize(
    "bad_type, sample_values",
    [
        pytest.param(pa.int32(), [1, 2], id="int32_not_in_vocabulary"),
        pytest.param(pa.timestamp("us"), [0, 1], id="timestamp_not_in_vocabulary"),
    ],
)
def test_input_column_type_outside_vocabulary_is_unsupported(
    binary_cmd: list[str],
    valid_license_env: dict[str, str],
    tmp_path: Path,
    bad_type: pa.DataType,
    sample_values: list[Any],
) -> None:
    """A column typed outside `column_types` (int32, a timestamp type, ...) is code 4 (contract:
    Capabilities)."""
    config = make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", bad_type)])
    input_bytes = arrow_stream_bytes(schema, {"col_a": sample_values})
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert_error_response(result, UNSUPPORTED)


def test_input_schema_presence_error_precedes_type_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """A presence violation (missing `col_b`) combined with a type violation (`col_a` sent as
    int32 instead of int64) is a data error, not code 4: presence is checked first (contract:
    Data)."""
    config = make_config(input_columns=["col_a", "col_b"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int32())])  # col_b missing; col_a also wrong type
    input_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2]})
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert_error_response(result, DATA_ERROR)


# ---------------------------------------------------------------------------
# 7. Schema-only round trip (contract: Data)
# ---------------------------------------------------------------------------


def test_schema_only_input_valid_license_produces_schema_only_output(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """Zero record batches then the end-of-stream marker is valid input; the output is
    schema-only too, but already carries the output columns and types (contract: Data)."""
    config = make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.string())])
    input_bytes = arrow_stream_bytes(schema, None)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    table = read_arrow_stream(result.stdout)
    assert table.num_rows == 0
    assert table.schema.names == ["hash_out"]
    assert pa.types.is_int64(table.schema.field("hash_out").type)


def test_schema_only_input_bad_license_still_rejected(
    binary_cmd: list[str], valid_config_path: Path, tmp_path: Path
) -> None:
    """A schema-only input with a bad license still exits 2 or 3, not 0: the license check applies
    before any data is read, schema-only input included (contract: Data, License)."""
    schema = pa.schema([pa.field("col_a", pa.string())])
    input_bytes = arrow_stream_bytes(schema, None)
    license_path = write_text(tmp_path / "license.txt", EXPIRED_LICENSE_TEXT)
    env = {"MLODA_LICENSE_FILE": str(license_path)}
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env, input_bytes)
    assert result.returncode in (2, 3), f"expected 2 or 3, got {result.returncode}; stderr={result.stderr!r}"
    error = stderr_error_object(result.stderr)
    assert error.get("code") == result.returncode, f"error object code mismatch: {error!r}"


# ---------------------------------------------------------------------------
# 8. End-of-stream marker on the raw output bytes (contract: Data, Conformance)
# ---------------------------------------------------------------------------


def test_output_stream_raw_bytes_end_with_ipc_eos_marker(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """The raw output bytes end with the IPC end-of-stream marker, checked on the bytes
    themselves rather than through pyarrow's own reader, which tolerates a stream missing it
    (contract: Data)."""
    case = hash_multi_column_case()
    config_path = write_json(tmp_path / "config.json", case["config"])
    input_bytes = arrow_stream_bytes(case["schema"], case["rows"])
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert_ends_with_ipc_eos_marker(result.stdout)


# ---------------------------------------------------------------------------
# 9. Malformed input rejections (contract: Data)
# ---------------------------------------------------------------------------


def test_zero_byte_input_is_data_error(
    binary_cmd: list[str], valid_config_path: Path, valid_license_env: dict[str, str]
) -> None:
    """Zero bytes is not an Arrow IPC stream at all: a data error (contract: Data)."""
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], valid_license_env, b"")
    assert_error_response(result, DATA_ERROR)


def test_truncated_stream_missing_end_of_stream_marker_is_data_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """ "truncated" means end of file without the end-of-stream marker, not "no more batches": a
    stream cut off right before that marker is a data error (contract: Data)."""
    config = make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int64())])
    full_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2, 3]})
    assert full_bytes.endswith(IPC_END_OF_STREAM_MARKER), "test setup: expected the writer to emit the EOS marker"
    truncated = full_bytes[: -len(IPC_END_OF_STREAM_MARKER)]
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, truncated)
    assert_error_response(result, DATA_ERROR)


def test_ipc_file_format_instead_of_stream_is_data_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """The IPC file/Feather format (`ARROW1` magic bytes) is not the streaming format the contract
    requires: a data error (contract: Data)."""
    config = make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int64())])
    input_bytes = arrow_file_format_bytes(schema, {"col_a": [1, 2, 3]})
    assert input_bytes[:6] == b"ARROW1", "test setup: expected the IPC file magic at the start"
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert_error_response(result, DATA_ERROR)


def test_compressed_record_batch_body_is_data_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """A compressed record batch body is a data error (contract: Data)."""
    config = make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int64())])
    options = pa.ipc.IpcWriteOptions(compression="lz4")
    input_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2, 3]}, options=options)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert_error_response(result, DATA_ERROR)


def test_dictionary_encoded_column_is_unsupported_type(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """A dictionary-encoded column is an unsupported column type (code 4), decided by the same
    type check as any other type outside the vocabulary (contract: Data)."""
    config = make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    dict_array = pa.array(["x", "y", "x"]).dictionary_encode()
    schema = pa.schema([pa.field("col_a", dict_array.type)])
    input_bytes = arrow_stream_bytes_from_arrays(schema, [dict_array])
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert_error_response(result, UNSUPPORTED)


# ---------------------------------------------------------------------------
# 10. The reserved `_conformance_internal_error` operation with real data (contract: Conformance)
# ---------------------------------------------------------------------------


def test_reserved_internal_error_operation_with_data_triggers_code_6(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """The reserved `_conformance_internal_error` operation, given a structurally valid config and
    valid input data, reaches the data stage and deliberately produces code 6, with stderr's last
    non-empty line still a valid `{"code": 6, "message": ...}` object, not a bare traceback
    (contract: Conformance)."""
    config = make_config(operation=RESERVED_INTERNAL_ERROR_OPERATION)
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int64())])
    input_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2, 3]})
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert_error_response(result, INTERNAL_ERROR)


# ---------------------------------------------------------------------------
# 11. `--capabilities` classification nice-to-have (contract: Capabilities) -- skipped.
# ---------------------------------------------------------------------------
# Not included: per contract, casting large_string/string_view to utf8 is a mixin-side
# responsibility that happens before data ever reaches the binary. A binary-level test sending
# large_string or string_view directly would really be exercising mixin behavior, out of scope for
# this binary-focused conformance kit, so this nice-to-have is skipped rather than over-scoped.


# ---------------------------------------------------------------------------
# 12. Data handling: data-free diagnostics, size caps, no network, no incidental files, the
#     minimal environment (contract: Data handling, Errors, Conformance)
# ---------------------------------------------------------------------------
#
# "no state between invocations" and "removes its own temp directory" (contract: Invocation, Data
# handling) describe the mixin's private-per-invocation-directory behavior, not the binary's: the
# binary itself carries no invocation-tracking state, so that obligation is covered here only as
# "no files created outside --output". The mixin side of both bullets is a different piece of
# work, out of scope for this kit.
#
# The "any exit code not in this table is treated as code 6" rule (contract: Errors) is a
# caller-side interpretation rule for a mixin receiving a response from *some* binary; it is not
# something this binary's own conformance kit can meaningfully test against itself (the binary
# always reports one of its own table's codes, or the top-level exception handler folds any
# uncaught error into a well-formed code 6, already covered below).


def test_diagnostics_never_leak_marked_cell_value_on_success(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """A distinctive marker cell value that would never otherwise appear in this suite's own
    input, output or diagnostics, sent through a normal successful "hash" run: stderr never
    contains it. stdout on a successful run legitimately carries the caller's own data by design,
    so this scopes the assertion to stderr only, not stdout (contract: Data handling)."""
    config = make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.string())])
    input_bytes = arrow_stream_bytes(schema, {"col_a": [DATA_FREE_MARKER]})
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert DATA_FREE_MARKER.encode("utf-8") not in result.stderr, (
        f"marker cell value leaked into stderr on a successful run: {result.stderr!r}"
    )


@pytest.mark.parametrize(
    "input_columns, operation, output_columns",
    [
        pytest.param(
            ["col_a", "col_b"], CAPABILITY_OPERATIONS[0], {"result": "hash_out"}, id="missing_column_data_error"
        ),
        pytest.param(["col_a"], RESERVED_INTERNAL_ERROR_OPERATION, {}, id="reserved_internal_error_operation"),
    ],
)
def test_diagnostics_never_leak_marked_cell_value_on_failure(
    binary_cmd: list[str],
    valid_license_env: dict[str, str],
    tmp_path: Path,
    input_columns: list[str],
    operation: str,
    output_columns: dict[str, str],
) -> None:
    """The same marker cell value, sent through two failing cases that still reach the data
    stage with the marked input present: a data error from a wrong schema (`col_b` declared in
    `input_columns` but absent from the stream), and the reserved
    `_conformance_internal_error` operation. Neither case's stderr contains the marker (contract:
    Data handling, Conformance)."""
    config = make_config(input_columns=input_columns, operation=operation, output_columns=output_columns)
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.string())])
    input_bytes = arrow_stream_bytes(schema, {"col_a": [DATA_FREE_MARKER]})
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert result.returncode != 0, f"expected this case to fail, got exit 0; stdout={result.stdout!r}"
    assert DATA_FREE_MARKER.encode("utf-8") not in result.stderr, (
        f"marker cell value leaked into stderr: {result.stderr!r}"
    )


def test_error_message_stays_under_size_cap_for_long_garbage_operation(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """A very long garbage `operation` string provokes the unsupported-operation error message to
    echo it back (`unsupported operation: {operation!r}`); `assert_error_response`'s shared
    size-cap assertion enforces that the resulting `message` still stays within the contract's
    1024-byte cap (contract: Data handling, Conformance)."""
    long_garbage_operation = "not_a_real_operation_" + "x" * 2000
    config = make_config(operation=long_garbage_operation)
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, UNSUPPORTED)


def test_no_network_dependency_under_unshare_net(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """A normal, successful "hash" run wrapped in `unshare --net` (a network-denied Linux
    namespace) must still succeed and produce the correct output, proving the binary makes no
    network calls it depends on (contract: Data handling, Conformance). Skipped if `unshare` is
    not on `PATH`, or if `unshare --net` itself cannot be used in this sandbox (some CI/containers
    block user/network namespaces), so the test never false-fails on an environment limitation
    unrelated to the binary."""
    if shutil.which("unshare") is None:
        pytest.skip("unshare is not available on this host")
    probe = subprocess.run(  # nosec B603 B607
        ["unshare", "--net", "--", "/bin/true"], capture_output=True, timeout=10.0
    )
    if probe.returncode != 0:
        pytest.skip(f"unshare --net is not usable in this sandbox: {probe.stderr!r}")

    case = hash_multi_column_case()
    config_path = write_json(tmp_path / "config.json", case["config"])
    input_bytes = arrow_stream_bytes(case["schema"], case["rows"])
    # `shutil.which("unshare")` above resolves against the ambient PATH, but the actual run below
    # replaces the child's environment outright (contract: Data handling, minimal environment);
    # without PATH in that replacement env, `unshare` itself might resolve differently (or not at
    # all) than the guard just checked, so PATH is added here to keep the guard and the run in
    # agreement.
    env = {**valid_license_env, "PATH": os.environ.get("PATH", os.defpath)}
    result = run_binary(
        ["unshare", "--net", "--", *binary_cmd], ["run", "--config", str(config_path)], env, input_bytes
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    table = read_arrow_stream(result.stdout)
    assert table.column("hash_out").to_pylist() == case["expected"]


def test_no_files_created_outside_output_in_read_only_cwd(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """A normal, successful "hash" run over stdin/stdout (no `--input`/`--output`, so nothing
    should touch the filesystem at all) run with its working directory set to a fresh read-only
    directory must still succeed and create no files there, proving the binary writes nothing of
    its own outside `--output` (contract: Data handling, Conformance). `--config` and the license
    file live in a separate, writable directory so only the cwd itself is read-only. Skipped on a
    platform with no POSIX directory-permission concept, or when running as root, which ignores
    directory write permissions."""
    if not hasattr(os, "geteuid"):
        pytest.skip("no POSIX directory-permission concept on this platform")
    if os.geteuid() == 0:
        pytest.skip("running as root ignores directory write permissions")

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    case = hash_multi_column_case()
    config_path = write_json(work_dir / "config.json", case["config"])
    input_bytes = arrow_stream_bytes(case["schema"], case["rows"])

    readonly_cwd = tmp_path / "readonly_cwd"
    readonly_cwd.mkdir()
    original_mode = readonly_cwd.stat().st_mode
    readonly_cwd.chmod(0o500)  # read + execute only: no write, so nothing can be created here
    try:
        result = run_binary(
            binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes, cwd=readonly_cwd
        )
    finally:
        readonly_cwd.chmod(original_mode)

    assert result.returncode == 0, f"stderr={result.stderr!r}"
    table = read_arrow_stream(result.stdout)
    assert table.column("hash_out").to_pylist() == case["expected"]
    leftover = list(readonly_cwd.iterdir())
    assert leftover == [], f"expected no files created in the read-only cwd, found {leftover!r}"


def test_minimal_environment_allowlist_only(binary_cmd: list[str], tmp_path: Path) -> None:
    """A normal, successful "hash" run with an environment containing only the license variable
    and `PATH` (the mixin's own allowlist also carries a fixed locale and, on Windows,
    `SYSTEMROOT`, but the license variable and `PATH` alone are enough to prove the binary needs
    nothing ambient) -- no `HOME`, no `LANG`, nothing else -- must still succeed and produce the
    correct output, proving the binary reads no environment variable outside the license
    variables and what its runtime needs to start (contract: Data handling, Conformance)."""
    case = hash_multi_column_case()
    config_path = write_json(tmp_path / "config.json", case["config"])
    input_bytes = arrow_stream_bytes(case["schema"], case["rows"])
    env = {"MLODA_LICENSE_KEY": VALID_LICENSE_TEXT, "PATH": os.environ.get("PATH", "")}
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], env, input_bytes)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    table = read_arrow_stream(result.stdout)
    assert table.column("hash_out").to_pylist() == case["expected"]


# ---------------------------------------------------------------------------
# 13. Corrected error classification (contract: Errors, License, Configuration, Data) -- these
#     cases were previously misclassified as code 6 (a bare uncaught exception falling through to
#     the top-level handler) instead of the contract's own, more specific code for each; the
#     erroneous former `test_uncaught_exception_still_produces_well_formed_code_6` (which asserted
#     code 6 for the invalid-UTF-8-config case below) is superseded by
#     `test_config_invalid_utf8_bytes_is_usage_error`.
# ---------------------------------------------------------------------------


def test_config_invalid_utf8_bytes_is_usage_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """A `--config` file containing bytes that are not valid UTF-8 is a usage error, exit 1: config
    decode/parse failures are usage errors (contract: Errors -- "config parse and structural
    validation ...; code 1"; Invocation -- "everything textual the binary reads or writes (config,
    ...) is UTF-8"), never code 6."""
    config_path = tmp_path / "config.json"
    config_path.write_bytes(b"\xff\xfe\x00invalid-utf8-config")
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, 1)


def test_license_file_invalid_utf8_bytes_is_license_invalid(
    binary_cmd: list[str], valid_config_path: Path, tmp_path: Path
) -> None:
    """A file named by `MLODA_LICENSE_FILE` containing bytes that are not valid UTF-8 is code 3,
    not code 2 and not code 6: the source is set and the file exists, but its content cannot be
    read as the license token, which is exactly "set but unusable" (contract: License -- "if a
    source is set but the license cannot be read or verified ..., the binary exits with code 3")."""
    license_path = tmp_path / "license.txt"
    license_path.write_bytes(b"\xff\xfe\x00invalid-utf8-license")
    env = {"MLODA_LICENSE_FILE": str(license_path)}
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env)
    assert_error_response(result, 3)


def test_output_path_is_existing_directory_is_usage_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """`--output` naming an existing directory (not a file) is a usage error, exit 1: "opening
    --input and creating --output (code 1 if either fails)" (contract: Errors)."""
    config = make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    output_dir = tmp_path / "output_is_a_directory"
    output_dir.mkdir()
    result = run_binary(
        binary_cmd, ["run", "--config", str(config_path), "--output", str(output_dir)], valid_license_env
    )
    assert_error_response(result, 1)


def test_input_path_not_readable_is_usage_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """An `--input` path that exists but is not readable (chmod 000) is a usage error, exit 1: the
    same "opening --input ... (code 1 if either fails)" rule as the `--output`-is-a-directory case
    above (contract: Errors). Skipped when running as root, which ignores file read permissions."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("running as root ignores file read permissions")
    config = make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int64())])
    input_path = tmp_path / "input.arrows"
    input_path.write_bytes(arrow_stream_bytes(schema, {"col_a": [1, 2, 3]}))
    input_path.chmod(0o000)
    try:
        result = run_binary(
            binary_cmd, ["run", "--config", str(config_path), "--input", str(input_path)], valid_license_env
        )
    finally:
        input_path.chmod(0o644)
    assert_error_response(result, 1)


def test_malformed_record_batch_after_valid_schema_is_data_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """Malformed/corrupted record-batch data appearing *after* a valid schema message is a data
    error, exit 5 (contract: Data): distinct from "malformed bytes from the very start" (already
    covered by `test_ipc_file_format_instead_of_stream_is_data_error` and friends), the
    schema-parsing step of `open_stream()` here succeeds, but reading the record batch
    (`read_all()`) fails on the corrupted second message."""
    config = make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int64())])
    valid_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2, 3]})
    corrupted = corrupt_record_batch_message_after_schema(valid_bytes)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, corrupted)
    assert_error_response(result, DATA_ERROR)


# ---------------------------------------------------------------------------
# 14. Trailing-data / concatenated-stream rejection (contract: Data)
# ---------------------------------------------------------------------------


def test_two_concatenated_ipc_streams_is_data_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """Two complete, valid Arrow IPC streams concatenated back-to-back must be rejected as a data
    error, not silently accepted with only the first stream's rows returned. The current
    truncation check only looks at whether the *last* 8 bytes of the whole buffer are the EOS
    marker, which this case satisfies coincidentally (the second stream's own EOS marker), so a
    correct implementation must track how many bytes the one logical stream actually consumed
    (e.g. a `pa.BufferReader`'s position after reading) and reject anything left over in the
    buffer (contract: Data)."""
    config = make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int64())])
    stream_one = arrow_stream_bytes(schema, {"col_a": [1, 2, 3]})
    stream_two = arrow_stream_bytes(schema, {"col_a": [4, 5, 6]})
    concatenated = stream_one + stream_two
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, concatenated)
    assert_error_response(result, DATA_ERROR)


def test_trailing_garbage_bytes_after_eos_marker_is_data_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """Valid stream bytes followed by arbitrary trailing garbage (not another full stream) after
    the end-of-stream marker is equally a data error, for the same "the whole buffer must be fully
    consumed by the one logical stream" reason as the concatenated-streams case above (contract:
    Data)."""
    config = make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int64())])
    valid_bytes = arrow_stream_bytes(schema, {"col_a": [1, 2, 3]})
    garbage = b"trailing-garbage-bytes-not-a-stream-0123456789"
    assert garbage[-len(IPC_END_OF_STREAM_MARKER) :] != IPC_END_OF_STREAM_MARKER, (
        "test setup: the garbage suffix must not coincidentally end with the EOS marker"
    )
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, valid_bytes + garbage)
    assert_error_response(result, DATA_ERROR)


# ---------------------------------------------------------------------------
# 15. Schema-only output wire shape, strengthened (contract: Data)
# ---------------------------------------------------------------------------


def test_schema_only_output_has_no_record_batch_message(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """A schema-only output (for a schema-only input) must be schema + end-of-stream marker only:
    no record-batch message at all, not a record-batch message with zero rows (contract: Data --
    "a schema-only input ... is valid and yields a schema-only output that already carries the
    output columns"). `pa.ipc.open_stream(...).read_all()` cannot distinguish the two wire shapes
    (both parse to a zero-row table), so this enumerates the raw output's message sequence
    instead."""
    config = make_config(input_columns=["col_a"], output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.string())])
    input_bytes = arrow_stream_bytes(schema, None)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    message_types = enumerate_ipc_message_types(result.stdout)
    assert "record batch" not in message_types, (
        f"expected a schema-only output to carry no record-batch message at all, got message types {message_types!r}"
    )


# ---------------------------------------------------------------------------
# 16. `parameters.key` validation and null/falsy handling (contract: Configuration)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    [
        pytest.param(0, id="key_zero_int"),
        pytest.param(False, id="key_false_bool"),
        pytest.param({"a": 1}, id="key_object"),
    ],
)
def test_config_parameters_key_non_string_is_usage_error(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path, bad_key: Any
) -> None:
    """`parameters.key` for the "hash" operation must be a string (or absent entirely); a
    non-string value -- however falsy (`0`, `false`, an object) -- is an operation-specific
    parameter-validation usage error, checked with the rest of the config document, not silently
    treated as "no key" (contract: Configuration -- "parameters: operation-specific ... checked
    with the document"; Errors -- config structural validation is code 1)."""
    config = make_config(parameters={"key": bad_key})
    config_path = write_json(tmp_path / "config.json", config)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env)
    assert_error_response(result, 1)


def test_hash_key_absent_equals_key_empty_string(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """`parameters.key` entirely absent and `parameters.key: ""` explicitly given both mean "no
    key" and must produce the same hash for the same row data (contract: Configuration). A
    regression guard for after the null-handling fix: only `key` being entirely absent (`None`) is
    "no key", not any other falsy value (`0`, `false`, `{}`), which
    `test_config_parameters_key_non_string_is_usage_error` above now rejects instead of silently
    treating as "no key"."""
    case_absent = hash_multi_column_case(key=None)
    case_empty = hash_multi_column_case(key="")
    assert case_absent["expected"] == case_empty["expected"], (
        "test setup: the independent reference algorithm must already treat absent and empty-string keys identically"
    )
    config_path_absent = write_json(tmp_path / "config_absent.json", case_absent["config"])
    config_path_empty = write_json(tmp_path / "config_empty.json", case_empty["config"])
    input_bytes = arrow_stream_bytes(case_absent["schema"], case_absent["rows"])

    result_absent = run_binary(binary_cmd, ["run", "--config", str(config_path_absent)], valid_license_env, input_bytes)
    assert result_absent.returncode == 0, f"stderr={result_absent.stderr!r}"
    table_absent = read_arrow_stream(result_absent.stdout)

    result_empty = run_binary(binary_cmd, ["run", "--config", str(config_path_empty)], valid_license_env, input_bytes)
    assert result_empty.returncode == 0, f"stderr={result_empty.stderr!r}"
    table_empty = read_arrow_stream(result_empty.stdout)

    # Compared to each other, not to `case_absent["expected"]`/`case_empty["expected"]` directly:
    # this test's scope is the absent-vs-empty-string key equivalence, independent of whatever
    # value-tokenization scheme (see the float-token portability fix elsewhere in this kit) the
    # binary under test currently implements.
    assert table_absent.column("hash_out").to_pylist() == table_empty.column("hash_out").to_pylist()


# ---------------------------------------------------------------------------
# 17. Additional coverage: field-order independence, multi-batch input, license precedence, and
#     transport edge cases (contract: Data, License, Invocation)
# ---------------------------------------------------------------------------


def test_hash_field_order_independent_of_stream_schema_order(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """The stream schema's field order need not match `input_columns`' order: the operation must
    read each row's values by column *name*, in `input_columns` order, not by stream position
    (contract: Data -- "the order of the input_columns list is part of the operation's input
    contract ...; the order of fields in the stream is not"). Config declares
    `input_columns: ["b", "a"]`, the reverse of the stream schema's field order `[a, b]`; this
    catches an implementation that reads columns positionally instead of by name."""
    input_columns = ["b", "a"]
    config = make_config(input_columns=input_columns, output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.int64())])
    rows = {"a": [1, 2, 3], "b": [10, 20, 30]}
    input_bytes = arrow_stream_bytes(schema, rows)
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    table = read_arrow_stream(result.stdout)
    expected = compute_expected_hash_column(rows, input_columns, key=None)
    assert table.column("hash_out").to_pylist() == expected


def test_hash_multi_batch_input_processes_all_batches(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """An IPC stream containing more than one record batch: the combined row count, row order, and
    hash values must all be correct across the batch boundary, catching an implementation that
    only processes the first batch (contract: Data)."""
    input_columns = ["id", "value"]
    config = make_config(input_columns=input_columns, output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("id", pa.string()), pa.field("value", pa.int64())])
    batch_one: dict[str, list[Any]] = {"id": ["row-0", "row-1"], "value": [1, 2]}
    batch_two: dict[str, list[Any]] = {"id": ["row-2", "row-3", "row-4"], "value": [3, 4, 5]}
    input_bytes = arrow_stream_bytes_multi_batch(schema, [batch_one, batch_two])
    result = run_binary(binary_cmd, ["run", "--config", str(config_path)], valid_license_env, input_bytes)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    table = read_arrow_stream(result.stdout)
    combined_rows = {
        "id": batch_one["id"] + batch_two["id"],
        "value": batch_one["value"] + batch_two["value"],
    }
    assert table.num_rows == len(combined_rows["id"]), f"row count mismatch: {table.num_rows}"
    expected = compute_expected_hash_column(combined_rows, input_columns, key=None)
    assert table.column("hash_out").to_pylist() == expected


def test_license_file_broken_key_valid_no_fallback_missing_file(
    binary_cmd: list[str], valid_config_path: Path, tmp_path: Path
) -> None:
    """`MLODA_LICENSE_FILE` naming a missing file, with `MLODA_LICENSE_KEY` simultaneously set to
    a genuinely valid token: the missing file wins with no fallback to the key, so this is still
    code 2, never a fallback-to-key success (contract: License -- "there is no fallback from a set
    but unusable MLODA_LICENSE_FILE to MLODA_LICENSE_KEY"). Complements
    `test_license_file_wins_over_license_key`, which only tests the reverse (valid file + garbled
    key)."""
    env = {
        "MLODA_LICENSE_FILE": str(tmp_path / "no-such-license.txt"),
        "MLODA_LICENSE_KEY": VALID_LICENSE_TEXT,
    }
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env)
    assert_error_response(result, 2)


def test_license_file_broken_key_valid_no_fallback_tampered_file(
    binary_cmd: list[str], valid_config_path: Path, tmp_path: Path
) -> None:
    """`MLODA_LICENSE_FILE` naming a file with tampered, unparseable content, with
    `MLODA_LICENSE_KEY` simultaneously set to a genuinely valid token: still code 3, never a
    fallback-to-key success (contract: License). The other broken-file shape (missing entirely,
    code 2) is `test_license_file_broken_key_valid_no_fallback_missing_file` above."""
    license_path = write_text(tmp_path / "license.txt", TAMPERED_UNPARSEABLE_TEXT)
    env = {"MLODA_LICENSE_FILE": str(license_path), "MLODA_LICENSE_KEY": VALID_LICENSE_TEXT}
    result = run_binary(binary_cmd, ["run", "--config", str(valid_config_path)], env)
    assert_error_response(result, 3)


def test_output_file_transport_leaves_stdout_empty(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """With `--output <path>` used, stdout stays empty (contract: Invocation -- "without it, the
    binary writes stdout; with it, stdout stays empty"). `run_binary_with_transport` reads the
    output bytes from the `--output` file when that transport is used but does not itself assert
    that stdout was left empty, so this asserts on `result.stdout` directly. Uses a plain int64
    case (not `hash_multi_column_case`, which has a float column) so this test's pass/fail is
    scoped to the stdout-emptiness question, independent of the float-token portability fix
    elsewhere in this kit."""
    input_columns = ["col_a"]
    config = make_config(input_columns=input_columns, output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int64())])
    rows = {"col_a": [1, 2, 3]}
    input_bytes = arrow_stream_bytes(schema, rows)
    result, output_bytes = run_binary_with_transport(
        binary_cmd,
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
    expected = compute_expected_hash_column(rows, input_columns, key=None)
    assert table.column("hash_out").to_pylist() == expected


def test_input_file_given_ignores_stdin(
    binary_cmd: list[str], valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """When `--input <path>` is used, stdin is never read (contract: Invocation -- "without
    --input, the binary reads stdin; with it, stdin is never read"): deliberately different,
    garbage bytes fed on stdin alongside a valid `--input` file must not affect the result, proving
    stdin was ignored rather than merged with or preferred over the file. Uses a plain int64 case
    (not `hash_multi_column_case`, which has a float column) so this test's pass/fail is scoped to
    the stdin-ignoring question, independent of the float-token portability fix elsewhere in this
    kit."""
    input_columns = ["col_a"]
    config = make_config(input_columns=input_columns, output_columns={"result": "hash_out"})
    config_path = write_json(tmp_path / "config.json", config)
    schema = pa.schema([pa.field("col_a", pa.int64())])
    rows = {"col_a": [1, 2, 3]}
    input_bytes = arrow_stream_bytes(schema, rows)
    input_path = tmp_path / "input.arrows"
    input_path.write_bytes(input_bytes)
    garbage_stdin = b"this-is-not-an-arrow-stream-and-must-be-ignored" * 4
    result = run_binary(
        binary_cmd,
        ["run", "--config", str(config_path), "--input", str(input_path)],
        valid_license_env,
        garbage_stdin,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    table = read_arrow_stream(result.stdout)
    expected = compute_expected_hash_column(rows, input_columns, key=None)
    assert table.column("hash_out").to_pylist() == expected


def test_zero_length_input_file_is_data_error(
    binary_cmd: list[str], valid_config_path: Path, valid_license_env: dict[str, str], tmp_path: Path
) -> None:
    """A zero-length `--input` file is not an Arrow IPC stream at all, exactly like zero-byte
    stdin (contract: Data -- "a data error (code 5), as is any input that is not an IPC stream at
    all: zero bytes, a zero-length --input file")."""
    input_path = tmp_path / "empty-input.arrows"
    input_path.write_bytes(b"")
    result = run_binary(
        binary_cmd, ["run", "--config", str(valid_config_path), "--input", str(input_path)], valid_license_env
    )
    assert_error_response(result, DATA_ERROR)
