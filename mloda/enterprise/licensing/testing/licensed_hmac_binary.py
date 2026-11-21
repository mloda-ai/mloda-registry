"""Fake ``hmac_sha256`` binary that verifies signed license tokens against the published test key.

Run: ``python -m mloda.enterprise.licensing.testing.licensed_hmac_binary [--plugin-id <id>]``. Needs
``mloda-testing[binary-model]`` and ``mloda-enterprise[ed25519]``."""

from __future__ import annotations

import sys

from mloda.enterprise.licensing.testing.licensed_binary import check_license_token
from mloda.testing.binary_model import hmac_fake_binary


def main() -> int:
    return hmac_fake_binary.main(check_license=check_license_token)


if __name__ == "__main__":
    sys.exit(main())
