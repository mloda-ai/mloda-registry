"""Standalone entry point for a second, minimal conforming binary (``second_fake_binary`` /
``frobnicate``), used by ``test_second_binary_conformance.py`` to prove
``BinaryModelConformanceBase``'s checks are truly operation/output-name agnostic.

Passes a ``FakeBinary`` to ``simulated_binary.main``: the shared hash computation under another
operation name and output key, a utf8-only column vocabulary and a utf8 output column type.

Not a test module: run via ``python -m mloda.testing.tests._second_fake_binary``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mloda.testing.binary_model import simulated_binary
from mloda.testing.binary_model.simulated_binary import HASH_OPERATION, FakeBinary, FakeOperation

if TYPE_CHECKING:
    import pyarrow as pa

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


def _utf8_output(
    original: Callable[[pa.Table, dict[str, Any]], tuple[pa.Schema, list[pa.Array]]],
) -> Callable[[pa.Table, dict[str, Any]], tuple[pa.Schema, list[pa.Array]]]:
    def compute(table: pa.Table, config: dict[str, Any]) -> tuple[pa.Schema, list[pa.Array]]:
        import pyarrow as pa

        schema, arrays = original(table, config)
        schema = pa.schema([pa.field(field.name, pa.string()) for field in schema])
        return schema, [array.cast(pa.string()) for array in arrays]

    return compute


SECOND_BINARY = FakeBinary(
    PLUGIN_ID, (FakeOperation(OPERATION, (OUTPUT_KEY,), _utf8_output(HASH_OPERATION.compute)),), COLUMN_TYPES, VERSION
)


if __name__ == "__main__":
    sys.exit(simulated_binary.main(SECOND_BINARY))
