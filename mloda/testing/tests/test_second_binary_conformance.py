"""Regression test proving ``BinaryModelConformanceBase``'s contract-generic checks are reusable,
unmodified, against a binary with a different operation/output name, per the "Conformance" section
of ``docs/binary-model-contract.md`` ("a binary implementation subclasses them, points those
attributes at itself, and inherits every applicable check unmodified").

Stands up a second, minimal conforming binary (``second_fake_binary``, one operation
"frobnicate", single output key "value") via ``_second_fake_binary.py``, a thin monkeypatch shim
over ``simulated_binary.py`` that changes only the binary's identity, not its CLI/license/Arrow-IPC
mechanics. Subclassing only ``BinaryModelConformanceBase`` here (not
``HashOperationConformanceMixin``: this binary's one operation is not literally "hash") and
pointing every documented overridable class attribute (``binary_cmd``, ``plugin_id``,
``operations``, ``default_input_columns``, ``default_output_columns``) at this second binary makes
every contract-generic check pass unmodified -- the same way ``TestBinaryModelConformance`` in
``test_binary_model_conformance.py`` reuses the same base class against the hash worked example.

This test previously caught a real defect: roughly a dozen and a half of
``BinaryModelConformanceBase``'s own test bodies built their config with an explicit
``output_columns={"result": ...}`` override (sometimes paired with an explicit
``input_columns=["col_a", ...]``) instead of deriving the key from ``self.default_output_columns``
-- "result" is ``simulated_binary.py``'s hash-specific output identifier (see
``_OPERATION_OUTPUTS = {"hash": ("result",)}`` there), not a contract-generic default, so every
such config failed the config-stage ``output_columns`` completeness check against this binary
before reaching the path each test actually meant to exercise. Fixed by deriving
``output_columns``/``input_columns`` from ``self.default_output_columns``/
``self.default_input_columns`` throughout those test bodies; this module stays as the regression
test for that fix.

The license-fixture texts need no override here: ``BinaryModelConformanceBase`` computes them
lazily from ``self.plugin_id``/``self.wrong_plugin_id`` at the point of use, so overriding
``plugin_id`` alone (below) is enough.
"""

from __future__ import annotations

import sys
from typing import ClassVar

from mloda.testing.binary_model.conformance import BinaryModelConformanceBase
from mloda.testing.tests._second_fake_binary import OPERATION, OUTPUT_KEY, PLUGIN_ID


class TestSecondBinaryConformance(BinaryModelConformanceBase):
    """Every ``BinaryModelConformanceBase`` contract-generic check, run against the second fake
    binary, proving those checks are truly contract-generic (contract: Conformance)."""

    binary_cmd: ClassVar[list[str]] = [sys.executable, "-m", "mloda.testing.tests._second_fake_binary"]
    plugin_id: ClassVar[str] = PLUGIN_ID
    operations: ClassVar[list[str]] = [OPERATION]
    default_input_columns: ClassVar[list[str]] = ["source_col"]
    default_output_columns: ClassVar[dict[str, str]] = {OUTPUT_KEY: "frobnicated_out"}
