"""Standalone entry point for a second, minimal conforming binary (``second_fake_binary`` /
``frobnicate``), used by ``test_second_binary_conformance.py`` to prove
``BinaryModelConformanceBase``'s checks are truly operation/output-name agnostic.

Reuses ``simulated_binary.py``'s CLI/license/Arrow-IPC mechanics unchanged by monkeypatching only
the module globals that carry "hash"'s own identity (``PLUGIN_ID``, ``CAPABILITY_OPERATIONS``,
``_OPERATION_OUTPUTS``, and the output-computing function): every ``simulated_binary.py`` function
looks these up as a module global at call time, so patching them from outside is sufficient.

Not a test module: run only via ``python -m mloda.testing.tests._second_fake_binary``, one fresh
subprocess per invocation, so the monkeypatching never leaks into other tests' own
``simulated_binary.py`` subprocess runs.
"""

from __future__ import annotations

import sys
from typing import Any

import pyarrow as pa

from mloda.testing.binary_model import simulated_binary
from mloda.testing.binary_model.hash_reference import compute_expected_hash

# Deliberately not "example_binary" / "hash" / "result" (contract: Identifier, Capabilities,
# Configuration), so this proves BinaryModelConformanceBase's checks are truly contract-generic.
PLUGIN_ID = "second_fake_binary"
OPERATION = "frobnicate"
OUTPUT_KEY = "value"


def _compute_frobnicate_output(table: pa.Table, config: dict[str, Any]) -> tuple[pa.Schema, list[pa.Array]]:
    """Drop-in replacement for ``simulated_binary._compute_hash_output``: reads the output under
    ``OUTPUT_KEY`` ("value") instead of the hash-specific literal ``"result"``. Reuses the same
    reference algorithm; only the operation/output identifiers are under test here."""
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
