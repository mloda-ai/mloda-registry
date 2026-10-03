"""Runs ``HmacSha256OperationConformanceMixin`` against the minimal hmac fake binary in
``_hmac_sha256_fake_binary.py``, plus one known-answer check of the in-process reference."""

from __future__ import annotations

import sys
from typing import ClassVar

from mloda.testing.binary_model.conformance import BinaryModelConformanceBase, HmacSha256OperationConformanceMixin
from mloda.testing.binary_model.hmac_sha256_reference import (
    KNOWN_ANSWER_KEY,
    KNOWN_ANSWER_VALUE,
    compute_expected_hmac_sha256,
)
from mloda.testing.tests._hmac_sha256_fake_binary import COLUMN_TYPES, PLUGIN_ID


class TestHmacSha256Conformance(HmacSha256OperationConformanceMixin, BinaryModelConformanceBase):
    """Every contract-generic and hmac_sha256-specific check, run against the hmac fake binary."""

    binary_cmd: ClassVar[list[str]] = [sys.executable, "-m", "mloda.testing.tests._hmac_sha256_fake_binary"]
    plugin_id: ClassVar[str] = PLUGIN_ID
    column_types: ClassVar[frozenset[str]] = COLUMN_TYPES


def test_reference_known_answer() -> None:
    assert (
        compute_expected_hmac_sha256(KNOWN_ANSWER_KEY, KNOWN_ANSWER_VALUE)
        == "ea497bac56964937ed2d5a6a77241026a609b5fd149f6eea3d41316e94983325"
    )
