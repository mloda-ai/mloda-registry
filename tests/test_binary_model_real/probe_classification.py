"""Pure exit-code classification for ``test_real_wheel.py::_probe_accepts_test_key``: whether an
installed real wheel accepts the shared test-signed license vectors, a release build's rejection
of them as an unknown key id, or anything else, which must fail loudly instead of being guessed at.
"""

from __future__ import annotations

import subprocess  # nosec

from mloda.testing.binary_model import LICENSE_INVALID
from mloda.testing.binary_model.conformance import stderr_error_object

# Wording owned by the mloda-binary-wrapper repo's stub/binary, not this repo's contract; verified
# against the real wheel but not contractually pinned, so a future wording change is expected to
# break this (and test_real_wheel.py's matching assertion) loudly rather than silently.
UNKNOWN_TEST_KEY_MESSAGE = "unknown license key id"


def classify_test_key_probe(result: "subprocess.CompletedProcess[bytes]") -> bool:
    """True if ``result`` is the expected success (exit 0); False if it is the expected rejection
    (``LICENSE_INVALID`` for an unknown test key id, a release build trusting only
    ``PRODUCTION_KEYS``). Anything else raises ``AssertionError`` naming the actual exit code and
    stderr, never silently misclassified as either outcome."""
    if result.returncode == 0:
        return True
    if result.returncode == LICENSE_INVALID:
        error = stderr_error_object(result.stderr)
        message = error.get("message", "")
        if UNKNOWN_TEST_KEY_MESSAGE in message:
            return False
        raise AssertionError(f"unexpected LICENSE_INVALID rejection unrelated to an unknown test key: {message!r}")
    raise AssertionError(f"unexpected probe exit code {result.returncode}, stderr={result.stderr!r}")
