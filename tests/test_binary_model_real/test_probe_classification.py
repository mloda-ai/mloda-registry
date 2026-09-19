"""Tests for ``probe_classification.py``: the exit-code classification, tested with synthetic
``subprocess.CompletedProcess`` results, and the probe itself, run against the simulated binary.
``test_real_wheel.py`` is ``pytest.importorskip("example_binary")``-gated, so this module must
never import from it, or the import would propagate that module-level skip here.
"""

from __future__ import annotations

import subprocess  # nosec
import sys

import pytest

from tests.test_binary_model_real.probe_classification import classify_test_key_probe, probe_accepts_test_key

# Same spelling as mloda/enterprise/tests/test_licensing_invariants.py and
# mloda/enterprise/feature_groups/binary_example/tests/test_binary_example_feature_group.py.
STUB_CMD = [sys.executable, "-m", "mloda.testing.binary_model.simulated_binary"]


def _completed_process(
    returncode: int, stdout: bytes = b"", stderr: bytes = b""
) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.CompletedProcess(args=["binary"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_classify_test_key_probe_true_on_expected_success() -> None:
    """Exit 0 is the expected success outcome: the test key was accepted."""
    result = _completed_process(0, stdout=b"arrow-ipc-stream-bytes")
    assert classify_test_key_probe(result) is True


def test_classify_test_key_probe_false_on_expected_license_invalid() -> None:
    """Exit 3 with an "unknown license key id" message is the expected rejection: a release build
    trusting only PRODUCTION_KEYS (currently empty)."""
    stderr = b'{"code": 3, "message": "unknown license key id: test-kid"}\n'
    result = _completed_process(3, stderr=stderr)
    assert classify_test_key_probe(result) is False


def test_classify_test_key_probe_raises_on_unexpected_outcome() -> None:
    """A crash / wrong exit code must raise loudly, never be misclassified as "accepted"."""
    result = _completed_process(1, stderr=b'{"code": 1, "message": "usage error"}\n')
    with pytest.raises(AssertionError, match="1"):
        classify_test_key_probe(result)


def test_classify_test_key_probe_raises_on_license_invalid_for_an_unrelated_reason() -> None:
    """Exit 3 for a reason other than an unknown test key must also raise, not be silently
    treated as either "accepted" or "the expected rejection"."""
    stderr = b'{"code": 3, "message": "license expired"}\n'
    result = _completed_process(3, stderr=stderr)
    with pytest.raises(AssertionError, match="license expired"):
        classify_test_key_probe(result)


def test_probe_accepts_test_key_true_for_accepting_binary() -> None:
    assert probe_accepts_test_key(STUB_CMD) is True


def test_probe_accepts_test_key_ignores_ambient_license_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLODA_LICENSE_KEY", "not-a-real-key")
    monkeypatch.setenv("MLODA_LICENSE_FILE", "/nonexistent/license/file")

    assert probe_accepts_test_key(STUB_CMD) is True


def test_classify_test_key_probe_raises_on_unparseable_stderr_at_exit_3() -> None:
    """Invalid JSON in stderr must still raise AssertionError, not a raw JSON decode error."""
    result = _completed_process(3, stderr=b"not json at all\n")
    with pytest.raises(AssertionError, match="3"):
        classify_test_key_probe(result)
    with pytest.raises(AssertionError, match="not json at all"):
        classify_test_key_probe(result)


def test_classify_test_key_probe_raises_on_empty_stderr_at_exit_3() -> None:
    result = _completed_process(3, stderr=b"")
    with pytest.raises(AssertionError, match="3"):
        classify_test_key_probe(result)
    with pytest.raises(AssertionError, match=r"b''"):
        classify_test_key_probe(result)
