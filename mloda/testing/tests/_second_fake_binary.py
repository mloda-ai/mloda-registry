"""Standalone entry point for a second, minimal conforming binary (``second_fake_binary`` /
``frobnicate``), used by ``test_second_binary_conformance.py`` to prove
``BinaryModelConformanceBase``'s checks are truly operation/output-name agnostic.

Reuses ``simulated_binary.py``'s CLI/license/Arrow-IPC mechanics unchanged by monkeypatching only
the module globals that carry "hash"'s own identity (``PLUGIN_ID``, ``CAPABILITY_OPERATIONS``
and ``_OPERATION_OUTPUTS``, from which the shared hash computation reads its output key), plus the
column vocabulary (utf8 only) and the output column type (utf8): every
``simulated_binary.py`` function looks these up as a module global at call time, so patching them from
outside is sufficient.

Not a test module: run only via ``python -m mloda.testing.tests._second_fake_binary``, one fresh
subprocess per invocation, so the monkeypatching never leaks into other tests' own
``simulated_binary.py`` subprocess runs.
"""

from __future__ import annotations

import sys
from typing import Any

from mloda.testing.binary_model import simulated_binary

# Deliberately not "example_binary" / "hash" / "result" (contract: Identifier, Capabilities,
# Configuration), so this proves BinaryModelConformanceBase's checks are truly contract-generic.
PLUGIN_ID = "second_fake_binary"
OPERATION = "frobnicate"
OUTPUT_KEY = "value"
# Valid semver with both a pre-release and a build-metadata part; the production parser accepts it, so the
# kit's --version check must too.
VERSION = "1.2.3-rc.1+build.7"
# Advertised column vocabulary: utf8 only, unlike the default binary (contract: Capabilities).
COLUMN_TYPES = frozenset({"utf8"})


def _utf8_output(original: Any) -> Any:
    def compute(table: Any, config: dict[str, Any]) -> Any:
        import pyarrow as pa

        schema, arrays = original(table, config)
        schema = pa.schema([pa.field(field.name, pa.string()) for field in schema])
        return schema, [array.cast(pa.string()) for array in arrays]

    return compute


def _install_second_binary_identity() -> None:
    simulated_binary.PLUGIN_ID = PLUGIN_ID
    simulated_binary.VERSION = VERSION
    simulated_binary.CAPABILITY_OPERATIONS = [OPERATION]
    simulated_binary._OPERATION_OUTPUTS = {OPERATION: (OUTPUT_KEY,)}
    setattr(simulated_binary, "COLUMN_TYPES", COLUMN_TYPES)
    simulated_binary._compute_hash_output = _utf8_output(simulated_binary._compute_hash_output)


if __name__ == "__main__":
    _install_second_binary_identity()
    sys.exit(simulated_binary.main())
