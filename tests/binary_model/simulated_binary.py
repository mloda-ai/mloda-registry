"""Simulated binary: a pure-Python CLI stand-in for a future Rust-compiled binary model.

This is a private test fixture, never published as a package. It exists so the registry can build
and test the (future) FeatureGroup mixin against the binary-model interface contract without any
real binary or Rust toolchain. See the contract document in the epic's
rust-crate-binary-feature-group folder for the full specification: ``--version``,
``--capabilities``, ``run --config <path> [--input <path>] [--output <path>]``, the license gate,
config validation, and the Arrow IPC data path.

Cycle 1 scope (implemented here): invocation surface, license gate, config validation, all of it
before any Arrow IPC data is touched. Cycle 2 placeholder: the data stage only distinguishes a
zero-byte input (code 5, per contract) from anything else, which fails as code 6 ("Arrow IPC data
path not implemented yet") rather than actually parsing Arrow IPC -- that parsing is a future TDD
cycle.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLUGIN_ID = "example_binary"
VERSION = "1.0.0"
CONTRACT_VERSION = 1
CAPABILITY_OPERATIONS = ["hash"]
RESERVED_INTERNAL_ERROR_OPERATION = "_conformance_internal_error"
COLUMN_TYPES = frozenset({"int64", "float64", "utf8", "boolean"})

# Outputs each operation defines, in the order they must be written (contract: Configuration).
_OPERATION_OUTPUTS: dict[str, tuple[str, ...]] = {"hash": ("result",)}

_REQUIRED_CONFIG_KEYS = frozenset({"input_columns", "operation", "parameters", "output_columns"})

# Contract "Errors" table.
USAGE_ERROR = 1
LICENSE_MISSING = 2
LICENSE_INVALID = 3
UNSUPPORTED = 4
DATA_ERROR = 5
INTERNAL_ERROR = 6


class _CliError(Exception):
    """Carries the exit code and stderr message for one contract error class."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


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
        except OSError as exc:
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
    except OSError as exc:
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
    if input_path is not None and not input_path.is_file():
        raise _CliError(USAGE_ERROR, f"--input path does not exist: {input_path}")
    if output_path is not None:
        try:
            output_path.touch()
        except OSError as exc:
            raise _CliError(USAGE_ERROR, f"--output path is not creatable: {exc}") from exc


def _read_input_bytes(input_path: Path | None) -> bytes:
    if input_path is not None:
        return input_path.read_bytes()
    return sys.stdin.buffer.read()


def _run_data_stage(raw: bytes) -> None:
    """Cycle 2 placeholder: only the zero-byte case (contract: Data, code 5) is distinguished.
    Real Arrow IPC parsing is a future TDD cycle, so any non-empty input fails loudly as an
    internal error instead of being silently accepted or crashing with a bare traceback."""
    if len(raw) == 0:
        raise _CliError(DATA_ERROR, "input is zero bytes, not an Arrow IPC stream")
    raise _CliError(INTERNAL_ERROR, "Arrow IPC data path not implemented yet")


def _run_command(args: list[str]) -> int:
    run_args = _parse_run_args(args)
    _check_license()
    config = _load_config(run_args.config_path)
    _validate_config_structure(config)
    operation = config["operation"]
    _check_operation_capability(operation)
    _check_output_columns_completeness(operation, config["output_columns"])
    _open_input_output(run_args.input_path, run_args.output_path)
    raw = _read_input_bytes(run_args.input_path)
    _run_data_stage(raw)
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


def main() -> int:
    try:
        return _dispatch(sys.argv[1:])
    except _CliError as exc:
        _emit_error(exc.code, exc.message)
        return exc.code
    except Exception as exc:
        _emit_error(INTERNAL_ERROR, f"internal error: {exc}")
        return INTERNAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
