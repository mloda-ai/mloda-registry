"""Test-only binary stand-ins for two ``BinaryModelMixin`` behaviours neither the well-behaved
simulated binary (``mloda.testing.binary_model``) nor ``faulty_binary.py`` can exercise:

- ``restricted_columns``: a per-binary ``column_types`` narrower than the contract's own full
  vocabulary (omits ``"boolean"``), so a test can distinguish "Arrow type outside the contract's
  vocabulary entirely" from "Arrow type in the vocabulary but not in this binary's own advertised
  ``column_types``" (contract: Capabilities).
- ``echo_utf8``: one operation, ``"echo"``, whose single output column is typed ``utf8`` -- unlike
  ``"hash"``, whose output is always ``int64`` -- needed to exercise the mixin's
  utf8-output-to-``large_string`` cast (contract: Capabilities).

Run as ``python -m mloda.community.feature_groups.binary_model.tests.mixin_fixtures --variant
<restricted_columns|echo_utf8> (--version | --capabilities | run --config <path> [--input <path>]
[--output <path>])``. Best-effort test tooling, not a conformance-kit binary: no license gate, no
config validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa

from mloda.testing.binary_model import CONTRACT_VERSION
from mloda.testing.binary_model.arrow import arrow_stream_bytes_from_arrays, read_arrow_stream

VERSION = "1.0.0"

_PLUGIN_ID: dict[str, str] = {"restricted_columns": "restricted_binary", "echo_utf8": "echo_utf8_binary"}
_OPERATIONS: dict[str, list[str]] = {"restricted_columns": ["hash"], "echo_utf8": ["echo"]}
# "restricted_columns" deliberately omits "boolean" from the contract's full vocabulary
# (contract: Capabilities).
_COLUMN_TYPES: dict[str, list[str]] = {
    "restricted_columns": ["int64", "float64", "utf8"],
    "echo_utf8": ["int64", "float64", "utf8", "boolean"],
}


def _capabilities(variant: str) -> dict[str, Any]:
    return {
        "contract": CONTRACT_VERSION,
        "plugin_id": _PLUGIN_ID[variant],
        "operations": list(_OPERATIONS[variant]),
        "column_types": list(_COLUMN_TYPES[variant]),
    }


def _parse_run_args(args: list[str]) -> tuple[Path, Path | None, Path | None]:
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
    if config_path is None:
        raise SystemExit("run requires --config <path>")
    return config_path, input_path, output_path


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


def _run_echo_utf8(args: list[str]) -> int:
    """``"echo"``: exactly one input column, echoed back cast to plain ``pa.string()`` (utf8),
    under its one configured output name (used to exercise the mixin's utf8-output-to-
    ``large_string`` cast, contract: Capabilities)."""
    config_path, input_path, output_path = _parse_run_args(args)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_column = config["input_columns"][0]
    written_name = next(iter(config["output_columns"].values()))
    table = read_arrow_stream(_read_input_bytes(input_path))
    values = table.column(input_column).cast(pa.string()).to_pylist()
    schema = pa.schema([pa.field(written_name, pa.string())])
    data = arrow_stream_bytes_from_arrays(schema, [pa.array(values, type=pa.string())])
    _write_output(data, output_path)
    return 0


def _run_restricted_columns(args: list[str]) -> int:
    """Never expected to be reached by this fixture's own tests: ``BinaryModelMixin`` rejects an
    input column outside this binary's advertised ``column_types`` before spawning ``run``
    (contract: Errors, check order)."""
    _parse_run_args(args)
    print(json.dumps({"code": 1, "message": "unexpected invocation of restricted_columns run"}), file=sys.stderr)
    return 1


_RUN_HANDLERS = {"restricted_columns": _run_restricted_columns, "echo_utf8": _run_echo_utf8}


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2 or args[0] != "--variant":
        print(json.dumps({"code": 1, "message": "usage: --variant <name> ..."}), file=sys.stderr)
        return 1
    variant = args[1]
    rest = args[2:]
    if rest == ["--version"]:
        print(f"{_PLUGIN_ID[variant]} {VERSION}")
        return 0
    if rest == ["--capabilities"]:
        print(json.dumps(_capabilities(variant)))
        return 0
    if rest and rest[0] == "run":
        return _RUN_HANDLERS[variant](rest[1:])
    print(json.dumps({"code": 1, "message": f"unrecognized arguments: {rest!r}"}), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
