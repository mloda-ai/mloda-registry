"""Unit tests for the pure exit-code classification behind
``test_real_wheel.py::_probe_accepts_test_key``.

Today that function does ``if result.returncode != LICENSE_INVALID: return True``, so a crash, an
unrelated non-zero exit, or any wrong exit code gets misclassified as "this build accepts the
test key" -- which would then incorrectly un-skip ``test_real_binary_end_to_end_with_valid_test_
license``. ``test_real_wheel.py`` is entirely ``pytest.importorskip("example_binary")``-gated, so
it cannot exercise this logic without the real wheel installed; these tests exercise the
classification directly against fake ``subprocess.CompletedProcess`` results instead, independent
of the wheel.

``tests.test_binary_model_real.probe_classification`` does not exist yet (Green phase adds it,
and refactors ``_probe_accepts_test_key`` to delegate to it): every test below fails with
``ModuleNotFoundError`` until it defines ``classify_test_key_probe(result) -> bool``:

- returns ``True`` for the expected success outcome (exit 0);
- returns ``False`` for the expected rejection (exit ``LICENSE_INVALID`` with an "unknown license
  key id" message -- a release build trusting only ``PRODUCTION_KEYS``);
- raises ``AssertionError`` naming the actual exit code/stderr for anything else (a crash, a
  wrong exit code, or a ``LICENSE_INVALID`` for an unrelated reason).
"""

from __future__ import annotations

import subprocess  # nosec

import pytest

from tests.test_binary_model_real.probe_classification import classify_test_key_probe


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
