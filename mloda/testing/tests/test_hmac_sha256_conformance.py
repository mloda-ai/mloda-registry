"""Runs ``HmacSha256OperationConformanceMixin`` against the minimal hmac fake binary in
``mloda.testing.binary_model.hmac_fake_binary``, plus one known-answer check of the in-process reference."""

from __future__ import annotations

import json
import os
import sys
from typing import ClassVar

import pytest

from mloda.testing.binary_model import simulated_binary
from mloda.testing.binary_model.conformance import (
    USAGE_ERROR,
    BinaryModelConformanceBase,
    HmacSha256OperationConformanceMixin,
    assert_error_response,
    run_binary,
)
from mloda.testing.binary_model.hmac_fake_binary import COLUMN_TYPES, PLUGIN_ID
from mloda.testing.binary_model.hmac_sha256_reference import (
    KNOWN_ANSWER_KEY,
    KNOWN_ANSWER_VALUE,
    compute_expected_hmac_sha256,
)


class TestHmacSha256Conformance(HmacSha256OperationConformanceMixin, BinaryModelConformanceBase):
    """Every contract-generic and hmac_sha256-specific check, run against the hmac fake binary."""

    binary_cmd: ClassVar[list[str]] = [sys.executable, "-m", "mloda.testing.binary_model.hmac_fake_binary"]
    plugin_id: ClassVar[str] = PLUGIN_ID
    column_types: ClassVar[frozenset[str]] = COLUMN_TYPES


def test_reference_known_answer() -> None:
    assert (
        compute_expected_hmac_sha256(KNOWN_ANSWER_KEY, KNOWN_ANSWER_VALUE)
        == "ea497bac56964937ed2d5a6a77241026a609b5fd149f6eea3d41316e94983325"
    )


_PLUGIN_ID_CMD = [
    sys.executable,
    "-m",
    "mloda.testing.binary_model.hmac_fake_binary",
    "--plugin-id",
    "anonymizer_binary",
]


@pytest.mark.parametrize(
    "flag", [pytest.param("--version", id="version"), pytest.param("--capabilities", id="capabilities")]
)
def test_plugin_id_flag_renames_the_reported_identity(flag: str) -> None:
    """`--plugin-id` renames the identity in both the `--version` line and the `--capabilities` JSON."""
    result = run_binary(_PLUGIN_ID_CMD, [flag], dict(os.environ))
    assert result.returncode == 0
    if flag == "--version":
        assert result.stdout.decode().strip() == f"anonymizer_binary {simulated_binary.VERSION}"
    else:
        assert json.loads(result.stdout)["plugin_id"] == "anonymizer_binary"


def test_bare_plugin_id_flag_is_a_usage_error_without_a_traceback() -> None:
    result = run_binary(_PLUGIN_ID_CMD[:3], ["--plugin-id"], dict(os.environ))
    assert_error_response(result, USAGE_ERROR)
    assert b"Traceback" not in result.stderr
