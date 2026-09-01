"""Standalone entry point standing up a second, minimal conforming binary
(``second_fake_binary``/``frobnicate``) for ``test_second_binary_conformance.py``'s red-phase proof
that ``BinaryModelConformanceBase``'s contract-generic checks are not actually
operation/output-name agnostic (see that module's docstring for the full rationale).

Reuses every mechanic of ``simulated_binary.py`` (CLI parsing, license gate, config validation,
Arrow IPC plumbing) unchanged, by monkeypatching only the handful of module-level names that carry
"hash"'s own operation identity: ``PLUGIN_ID``, ``CAPABILITY_OPERATIONS``, ``_OPERATION_OUTPUTS``,
and the function that reads the operation's single output column under the literal key
``"result"``. Every ``simulated_binary.py`` function that reads these looks them up as a module
global at call time (ordinary Python name resolution, not something bound at def time), so
patching the module's attributes from outside -- rather than editing its source, which is off
limits for this task -- is sufficient; no other CLI/license/Arrow-IPC logic is duplicated.

Not itself a test module: run only via ``python -m mloda.testing.tests._second_fake_binary``,
mirroring how ``binary_cmd`` invokes ``simulated_binary.py`` itself. Each invocation is its own
fresh subprocess, so this monkeypatching never leaks into the plain ``simulated_binary.py``
subprocess invocations other tests in this repository make.
"""

from __future__ import annotations

import sys
from typing import Any

import pyarrow as pa

from mloda.testing.binary_model import simulated_binary
from mloda.testing.binary_model.hash_reference import compute_expected_hash

# This binary's identity: deliberately not "example_binary" / "hash" / "result" (contract:
# Identifier, Capabilities, Configuration), so a conformance suite pointed at it proves whether
# BinaryModelConformanceBase's checks are truly contract-generic, as the contract's Conformance
# section promises ("inherits every applicable check unmodified").
PLUGIN_ID = "second_fake_binary"
OPERATION = "frobnicate"
OUTPUT_KEY = "value"


def _compute_frobnicate_output(table: pa.Table, config: dict[str, Any]) -> tuple[pa.Schema, list[pa.Array]]:
    """Drop-in replacement for ``simulated_binary._compute_hash_output``: identical except it
    reads the operation's single output under ``OUTPUT_KEY`` ("value") instead of the hash-specific
    literal ``"result"``. Reuses the same reference algorithm; the computed values themselves are
    not the point under test here, only the operation/output identifiers are."""
    input_columns = config["input_columns"]
    key: str | None = config["parameters"].get("key")
    written_name = config["output_columns"][OUTPUT_KEY]
    columns = {name: table.column(name).to_pylist() for name in input_columns}
    values = [
        compute_expected_hash(key, [columns[name][row_index] for name in input_columns])
        for row_index in range(table.num_rows)
    ]
    output_schema = pa.schema([pa.field(written_name, pa.int64())])
    return output_schema, [pa.array(values, type=pa.int64())]


def _install_second_binary_identity() -> None:
    simulated_binary.PLUGIN_ID = PLUGIN_ID
    simulated_binary.CAPABILITY_OPERATIONS = [OPERATION]
    simulated_binary._OPERATION_OUTPUTS = {OPERATION: (OUTPUT_KEY,)}
    simulated_binary._compute_hash_output = _compute_frobnicate_output


if __name__ == "__main__":
    _install_second_binary_identity()
    sys.exit(simulated_binary.main())
