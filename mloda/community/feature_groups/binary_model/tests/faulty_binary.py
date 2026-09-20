"""Test-only faulty binary: a deliberately misbehaving CLI stand-in exercising rejection and error
paths of ``transport.py`` and ``binary.py`` that the well-behaved simulated binary never triggers.

Best-effort test tooling, not a conformance-kit binary: no license gate, no config validation.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess  # nosec
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa

from mloda.testing.binary_model import COLUMN_TYPES
from mloda.testing.binary_model.arrow import arrow_stream_bytes_from_arrays, read_arrow_stream

PLUGIN_ID = "faulty_binary"
VERSION = "0.0.1"


def _emit_error(code: int, message: str) -> int:
    print(json.dumps({"code": code, "message": message}), file=sys.stderr)
    return code


def _license_env_present() -> bool:
    """Whether either license environment variable is set (contract: License), used by
    ``reject_license_at_probe`` to prove probing never receives the license."""
    return bool(os.environ.get("MLODA_LICENSE_FILE") or os.environ.get("MLODA_LICENSE_KEY"))


def _version(mode: str) -> int:
    if mode == "reject_license_at_probe" and _license_env_present():
        return _emit_error(1, "probing must not receive a license (simulated by faulty_binary reject_license_at_probe)")
    if mode == "version_not_semver":
        print(f"{PLUGIN_ID} 1")
        return 0
    if mode == "version_empty":
        print(f"{PLUGIN_ID} ")
        return 0
    if mode == "version_no_second_token":
        print(PLUGIN_ID)
        return 0
    if mode == "version_prerelease":
        print(f"{PLUGIN_ID} 0.0.1-rc.1+build.5")
        return 0
    print(f"{PLUGIN_ID} {VERSION}")
    if mode == "version_two_lines":
        print("unexpected second line")
    return 0


def _capabilities(mode: str) -> int:
    if mode == "reject_license_at_probe" and _license_env_present():
        return _emit_error(1, "probing must not receive a license (simulated by faulty_binary reject_license_at_probe)")
    if mode == "contract_2":
        print(
            json.dumps(
                {"contract": 2, "plugin_id": PLUGIN_ID, "operations": ["hash"], "column_types": sorted(COLUMN_TYPES)}
            )
        )
    elif mode == "bad_capabilities":
        print(json.dumps({"plugin_id": PLUGIN_ID}))
    elif mode == "capabilities_not_json":
        print("oops")
    else:
        print(
            json.dumps(
                {"contract": 1, "plugin_id": PLUGIN_ID, "operations": ["hash"], "column_types": sorted(COLUMN_TYPES)}
            )
        )
    return 0


def _parse_run_args(args: list[str]) -> tuple[Path | None, Path | None, Path | None]:
    config_path: Path | None = None
    input_path: Path | None = None
    output_path: Path | None = None
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            i += 1
            config_path = Path(args[i])
        elif args[i] == "--input" and i + 1 < len(args):
            i += 1
            input_path = Path(args[i])
        elif args[i] == "--output" and i + 1 < len(args):
            i += 1
            output_path = Path(args[i])
        i += 1
    return config_path, input_path, output_path


def _load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    data: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    return data


def _read_input_bytes(input_path: Path | None) -> bytes:
    if input_path is not None:
        return input_path.read_bytes()
    return sys.stdin.buffer.read()


def _write_output(data: bytes, output_path: Path | None) -> None:
    if output_path is not None:
        output_path.write_bytes(data)
        return
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _run(mode: str, args: list[str]) -> int:
    config_path, input_path, output_path = _parse_run_args(args)

    if mode == "hang":
        time.sleep(60)
        return 0
    if mode == "hang_with_child":
        loaded_config = _load_config(config_path)
        pid_path = Path(loaded_config["parameters"]["pid_file"])
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])  # nosec B603
        pid_path.write_text(str(child.pid), encoding="utf-8")
        time.sleep(60)
        return 0
    if mode == "exit_before_reading":
        return _emit_error(2, "license missing (simulated by faulty_binary exit_before_reading)")
    if mode == "signal":
        os.kill(os.getpid(), signal.SIGKILL)
        return 6
    if mode == "garbage_stderr":
        print("not json at all", file=sys.stderr)
        return 5
    if mode == "garbage_output":
        _write_output(b"this is not arrow", output_path)
        return 0
    if mode == "echo_env":
        _write_output(json.dumps(sorted(os.environ)).encode("utf-8"), output_path)
        return 0
    if mode == "no_output_file" and output_path is not None:
        # Exits 0 without writing the file --output names at all (contract: Invocation, Data).
        return 0

    config = _load_config(config_path)
    output_columns = config.get("output_columns") or {"result": "result"}
    written_name = next(iter(output_columns.values()))
    raw = _read_input_bytes(input_path)
    num_rows = read_arrow_stream(raw).num_rows if raw else 0

    field_name = written_name
    column_type: pa.DataType = pa.int64()
    if mode == "wrong_row_count":
        num_rows += 1
    elif mode == "wrong_schema":
        field_name = "unexpected_name"
    elif mode == "wrong_type":
        column_type = pa.int32()

    if mode == "duplicate_output_names":
        # A valid stream whose schema carries the written name twice (contract: Data); built via
        # arrow_stream_bytes_from_arrays, which accepts duplicate field names.
        dup_schema = pa.schema([pa.field(field_name, column_type), pa.field(field_name, column_type)])
        dup_data = arrow_stream_bytes_from_arrays(
            dup_schema, [pa.array([0] * num_rows, type=column_type), pa.array([0] * num_rows, type=column_type)]
        )
        _write_output(dup_data, output_path)
        return 0

    schema = pa.schema([pa.field(field_name, column_type)])
    data = arrow_stream_bytes_from_arrays(schema, [pa.array([0] * num_rows, type=column_type)])
    if mode == "missing_eos":
        data = data[:-8]
    _write_output(data, output_path)
    return 0


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2 or args[0] != "--mode":
        return _emit_error(1, "usage: --mode <mode> (--version | --capabilities | run --config <path> ...)")
    mode = args[1]
    rest = args[2:]
    if rest == ["--version"]:
        return _version(mode)
    if rest == ["--capabilities"]:
        return _capabilities(mode)
    if rest and rest[0] == "run":
        return _run(mode, rest[1:])
    return _emit_error(1, f"unrecognized arguments: {rest!r}")


if __name__ == "__main__":
    sys.exit(main())
