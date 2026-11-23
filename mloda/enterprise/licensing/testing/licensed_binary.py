"""Fake ``example_binary`` that verifies signed license tokens against the published test key.

Run: ``python -m mloda.enterprise.licensing.testing.licensed_binary``. Needs ``mloda-testing[binary-model]``
and ``mloda-enterprise[ed25519]``."""

from __future__ import annotations

import dataclasses
import sys
from datetime import date

from mloda.testing.binary_model import simulated_binary
from mloda.testing.binary_model.simulated_binary import EXAMPLE_BINARY, FakeLicenseError

RELEASE_DATE = date(2026, 9, 1)
"""The binary's own release date, the build-time constant compared against the license's
``max_release_date`` claim (spec: Verification step 8)."""


def check_license_token(text: str, plugin_id: str) -> bool:
    """Verify ``text`` as a signed license token for ``plugin_id`` against the published test keys. Returns
    True when accepted inside its grace window (spec: Verification step 6); a rejection is a
    ``FakeLicenseError`` carrying the verifier's reason."""
    # Imported inside the call so --version and --capabilities skip the cryptography import.
    from mloda.enterprise.licensing import license_token
    from mloda.enterprise.licensing.testing import license_vectors

    try:
        verified = license_token.verify_license_token(
            text,
            keys=license_vectors.TEST_PUBLIC_KEYS,
            plugin_id=plugin_id,
            release_date=RELEASE_DATE,
        )
    except license_token.LicenseVerificationError as exc:
        raise FakeLicenseError(exc.reason) from exc
    return verified.in_grace


def main() -> int:
    return simulated_binary.main(dataclasses.replace(EXAMPLE_BINARY, check_license=check_license_token))


if __name__ == "__main__":
    sys.exit(main())
