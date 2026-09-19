"""Exit-code classification and probe-request building for ``test_real_wheel.py``: whether an
installed real wheel accepts the shared test-signed license vectors, a release build's rejection
of them as an unknown key id, or anything else, which must fail loudly instead of being guessed at.
"""

from __future__ import annotations

import subprocess  # nosec
import tempfile
from pathlib import Path

import pyarrow as pa

from mloda.community.feature_groups.binary_model.transport import minimal_environment
from mloda.enterprise.feature_groups.binary_example.binary_example_feature_group import BinaryExampleFeatureGroup
from mloda.testing.binary_model import LICENSE_INVALID
from mloda.testing.binary_model.arrow import arrow_stream_bytes
from mloda.testing.binary_model.conformance import run_binary, stderr_error_object, write_json
from mloda.testing.binary_model.license_vectors import valid_license_token

# Wording owned by the mloda-binary-wrapper repo's stub/binary, not this repo's contract; verified
# against the real wheel but not contractually pinned, so a future wording change is expected to
# break this (and test_real_wheel.py's matching assertion) loudly rather than silently.
UNKNOWN_TEST_KEY_MESSAGE = "unknown license key id"

_PLUGIN_ID = BinaryExampleFeatureGroup.BINARY_PLUGIN_ID


def classify_test_key_probe(result: "subprocess.CompletedProcess[bytes]") -> bool:
    """True if ``result`` is the expected success (exit 0); False if it is the expected rejection
    (``LICENSE_INVALID`` for an unknown test key id, a release build trusting only
    ``PRODUCTION_KEYS``). Anything else raises ``AssertionError`` naming the actual exit code and
    stderr, never silently misclassified as either outcome."""
    if result.returncode == 0:
        return True
    if result.returncode == LICENSE_INVALID:
        try:
            error = stderr_error_object(result.stderr)
        except (AssertionError, ValueError) as exc:
            raise AssertionError(f"unexpected probe exit code {result.returncode}, stderr={result.stderr!r}") from exc
        message = error.get("message", "")
        if not isinstance(message, str):
            raise AssertionError(f"unexpected probe exit code {result.returncode}, error object: {error!r}")
        if UNKNOWN_TEST_KEY_MESSAGE in message:
            return False
        raise AssertionError(f"unexpected LICENSE_INVALID rejection unrelated to an unknown test key: {message!r}")
    raise AssertionError(f"unexpected probe exit code {result.returncode}, stderr={result.stderr!r}")


def probe_environment(license_key: str | None = None) -> dict[str, str]:
    """Minimal probe environment: real PATH/locale/SYSTEMROOT from the caller's own environment,
    plus only the given test license_key, never the caller's own license variables."""
    return minimal_environment(license_file="", license_key=license_key or "")


def probe_accepts_test_key(cmd: list[str]) -> bool:
    """True if ``cmd`` accepts the shared test-signed license vectors from ``license_vectors``;
    False for a release build, which trusts only ``PRODUCTION_KEYS`` (currently empty, so a
    test-signed token is always an unknown ``kid``). Delegates exit-code/message interpretation to
    ``classify_test_key_probe``, which raises loudly for anything else."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = write_json(
            Path(tmp_dir) / "config.json",
            {"input_columns": ["col_a"], "operation": "hash", "parameters": {}, "output_columns": {"result": "out"}},
        )
        input_bytes = arrow_stream_bytes(pa.schema([pa.field("col_a", pa.string())]), {"col_a": ["alpha"]})
        env = probe_environment(valid_license_token([_PLUGIN_ID]))
        result = run_binary(cmd, ["run", "--config", str(config_path)], env, input_bytes)
    return classify_test_key_probe(result)
