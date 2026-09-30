"""Proves ``BinaryModelConformanceBase``'s contract-generic checks are genuinely reusable against a
binary shaped differently from the "hash" worked example, not just superficially generic.

Stands up a second, minimal binary (``second_fake_binary``, operation "frobnicate", output key
"value") via the monkeypatch shim in ``_second_fake_binary.py``, then points every overridable
class attribute at it, the same way ``TestBinaryModelConformance`` targets the hash example.
"""

from __future__ import annotations

import sys
from typing import ClassVar

import pyarrow as pa

from mloda.testing.binary_model.conformance import BinaryModelConformanceBase
from mloda.testing.tests._second_fake_binary import COLUMN_TYPES, OPERATION, OUTPUT_KEY, PLUGIN_ID


class TestSecondBinaryConformance(BinaryModelConformanceBase):
    """Every contract-generic check, run against the second fake binary (contract: Conformance)."""

    binary_cmd: ClassVar[list[str]] = [sys.executable, "-m", "mloda.testing.tests._second_fake_binary"]
    plugin_id: ClassVar[str] = PLUGIN_ID
    operations: ClassVar[list[str]] = [OPERATION]
    default_input_columns: ClassVar[list[str]] = ["source_col"]
    default_output_columns: ClassVar[dict[str, str]] = {OUTPUT_KEY: "frobnicated_out"}
    column_types: ClassVar[frozenset[str]] = COLUMN_TYPES

    def default_output_column_type(self) -> pa.DataType:
        return pa.string()
