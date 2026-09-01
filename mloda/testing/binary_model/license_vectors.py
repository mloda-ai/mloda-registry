"""Signed license-token test vectors (contract: License; spec: Test vectors): the deliberately
public test keypair (``test-2026-01``), per-state builders on ``license_token.sign_license_token``,
and literal tokens for every time-stable state the conformance kit distinguishes. Ed25519 signing
is deterministic, so the literals are re-derived from the builders by the tests and cannot drift.

The legacy placeholder API (plain-JSON ``{"status": ..., "plugins": [...]}`` texts) at the bottom
predates the signed format and is still consumed by the stub and the conformance suite; a follow-up
replaces those consumers and removes it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from mloda.testing.binary_model.license_token import TOKEN_SCHEMA_VERSION, sign_license_token

# =============================================================================
# The published test keypair (spec: Keys, kid, rotation)
# =============================================================================

TEST_KID = "test-2026-01"

# DELIBERATELY PUBLIC, TEST-ONLY keypair: the private seed below is published on purpose so any
# consumer can re-sign vectors. It must NEVER appear in a release build's key map; int-045's
# release checks assert the test kid's absence (spec: Keys, kid, rotation).
TEST_SECRET_SEED = bytes.fromhex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
TEST_PUBLIC_KEY = bytes.fromhex("03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8")
TEST_PUBLIC_KEYS = {TEST_KID: TEST_PUBLIC_KEY}

# =============================================================================
# Per-state builders (spec: Test vectors)
# =============================================================================

_LICENSE_ID = "lic-test-001"
_CUSTOMER_ID = "cust-test-001"
_ISSUED_2020 = "2020-01-01T00:00:00+00:00"


def _base_claims(plugins: list[str]) -> dict[str, Any]:
    """A fully valid claim set: issued 2020, expiring 2036, 14 grace days (spec: Claims)."""
    return {
        "v": TOKEN_SCHEMA_VERSION,
        "license_id": _LICENSE_ID,
        "customer_id": _CUSTOMER_ID,
        "plugins": plugins,
        "iat": _ISSUED_2020,
        "nbf": _ISSUED_2020,
        "exp": "2036-01-01T00:00:00+00:00",
        "grace_days": 14,
    }


def _signed(claims: dict[str, Any]) -> str:
    return sign_license_token(claims, kid=TEST_KID, secret_seed=TEST_SECRET_SEED)


def valid_license_token(plugins: list[str]) -> str:
    """A token valid until 2036 for the given plugins (spec: Test vectors)."""
    return _signed(_base_claims(plugins))


def expired_license_token(plugins: list[str]) -> str:
    """A token expired 2021, beyond its 14 grace days (spec: Test vectors)."""
    claims = _base_claims(plugins)
    claims["exp"] = "2021-01-01T00:00:00+00:00"
    return _signed(claims)


def in_grace_license_token(plugins: list[str], *, now: datetime | None = None) -> str:
    """The one time-relative state a constant cannot express: ``exp`` one day before ``now``,
    inside 7 grace days (spec: Test vectors; spec: Verification step 6)."""
    moment = now if now is not None else datetime.now(timezone.utc)
    claims = _base_claims(plugins)
    claims["exp"] = (moment.astimezone(timezone.utc) - timedelta(days=1)).isoformat()
    claims["grace_days"] = 7
    return _signed(claims)


def not_yet_valid_license_token(plugins: list[str]) -> str:
    """A token whose ``nbf`` is 2036 (spec: Test vectors)."""
    claims = _base_claims(plugins)
    claims["nbf"] = "2036-01-01T00:00:00+00:00"
    claims["exp"] = "2037-01-01T00:00:00+00:00"
    return _signed(claims)


def unknown_kid_license_token(plugins: list[str]) -> str:
    """A well-signed token under a ``kid`` no verifier key map contains (spec: Verification
    step 3)."""
    return sign_license_token(_base_claims(plugins), kid="test-unknown", secret_seed=TEST_SECRET_SEED)


def missing_plugins_claim_token() -> str:
    """A well-signed payload missing the required ``plugins`` claim (spec: Verification step 5)."""
    claims = _base_claims([])
    del claims["plugins"]
    return _signed(claims)


def tampered_signature_token(plugins: list[str]) -> str:
    """A valid token with one payload base64url character swapped: the container still decodes but
    the Ed25519 signature no longer matches (spec: Verification step 4)."""
    parts = valid_license_token(plugins).split(".")
    body = parts[2]
    index = 5
    replacement = "A" if body[index] != "A" else "B"
    parts[2] = body[:index] + replacement + body[index + 1 :]
    return ".".join(parts)


# =============================================================================
# Literal tokens for the time-stable states (spec: Test vectors)
# =============================================================================

VALID_TOKEN = (  # nosec B105
    "v4.public.eyJjdXN0b21lcl9pZCI6ImN1c3QtdGVzdC0wMDEiLCJleHAiOiIyMDM2LTAxLTAxVDAwOjAwOjAwKzAwOjAwIiwiZ3JhY2"
    "VfZGF5cyI6MTQsImlhdCI6IjIwMjAtMDEtMDFUMDA6MDA6MDArMDA6MDAiLCJsaWNlbnNlX2lkIjoibGljLXRlc3QtMDAxIiwibmJmIj"
    "oiMjAyMC0wMS0wMVQwMDowMDowMCswMDowMCIsInBsdWdpbnMiOlsiZXhhbXBsZV9iaW5hcnkiXSwidiI6MX1GJGqn4eqTvdTfeZBrIw"
    "_qpXJVwIy1oeV1d74yzCeVTL80CjZFdgQFJtQ5XQNzGPcb9VAbsJWeaMIgc6442ckC.eyJraWQiOiJ0ZXN0LTIwMjYtMDEifQ"
)

EXPIRED_TOKEN = (  # nosec B105
    "v4.public.eyJjdXN0b21lcl9pZCI6ImN1c3QtdGVzdC0wMDEiLCJleHAiOiIyMDIxLTAxLTAxVDAwOjAwOjAwKzAwOjAwIiwiZ3JhY2"
    "VfZGF5cyI6MTQsImlhdCI6IjIwMjAtMDEtMDFUMDA6MDA6MDArMDA6MDAiLCJsaWNlbnNlX2lkIjoibGljLXRlc3QtMDAxIiwibmJmIj"
    "oiMjAyMC0wMS0wMVQwMDowMDowMCswMDowMCIsInBsdWdpbnMiOlsiZXhhbXBsZV9iaW5hcnkiXSwidiI6MX1Vp6-EpQxr4mvpuggLbD"
    "VSQXZrftZMlYllBwvSkirwGSDmmw5vMtaMxekDc3l1SUil7rLjeDTLauhENgbvteII.eyJraWQiOiJ0ZXN0LTIwMjYtMDEifQ"
)

WRONG_PLUGIN_TOKEN = (  # nosec B105
    "v4.public.eyJjdXN0b21lcl9pZCI6ImN1c3QtdGVzdC0wMDEiLCJleHAiOiIyMDM2LTAxLTAxVDAwOjAwOjAwKzAwOjAwIiwiZ3JhY2"
    "VfZGF5cyI6MTQsImlhdCI6IjIwMjAtMDEtMDFUMDA6MDA6MDArMDA6MDAiLCJsaWNlbnNlX2lkIjoibGljLXRlc3QtMDAxIiwibmJmIj"
    "oiMjAyMC0wMS0wMVQwMDowMDowMCswMDowMCIsInBsdWdpbnMiOlsic29tZV9vdGhlcl9wbHVnaW4iXSwidiI6MX0K1CFROSfBQ_D90m"
    "JCBkDTQyEYOTb7M2Hwybu_Rpl_Sz5i0L9W-NB6wfn8AbrI1fi-k3U0x4v91Krx1CAYiQsG.eyJraWQiOiJ0ZXN0LTIwMjYtMDEifQ"
)

NOT_YET_VALID_TOKEN = (  # nosec B105
    "v4.public.eyJjdXN0b21lcl9pZCI6ImN1c3QtdGVzdC0wMDEiLCJleHAiOiIyMDM3LTAxLTAxVDAwOjAwOjAwKzAwOjAwIiwiZ3JhY2"
    "VfZGF5cyI6MTQsImlhdCI6IjIwMjAtMDEtMDFUMDA6MDA6MDArMDA6MDAiLCJsaWNlbnNlX2lkIjoibGljLXRlc3QtMDAxIiwibmJmIj"
    "oiMjAzNi0wMS0wMVQwMDowMDowMCswMDowMCIsInBsdWdpbnMiOlsiZXhhbXBsZV9iaW5hcnkiXSwidiI6MX3jNx3V01bzN5HD4wKCPL"
    "a0LceVD4mRI7xA6ObWRWP7TlZP7h7VlhRNJ_jkMrvxGVnNn8KTQDnqO1KzH3pq0OsI.eyJraWQiOiJ0ZXN0LTIwMjYtMDEifQ"
)

UNKNOWN_KID_TOKEN = (  # nosec B105
    "v4.public.eyJjdXN0b21lcl9pZCI6ImN1c3QtdGVzdC0wMDEiLCJleHAiOiIyMDM2LTAxLTAxVDAwOjAwOjAwKzAwOjAwIiwiZ3JhY2"
    "VfZGF5cyI6MTQsImlhdCI6IjIwMjAtMDEtMDFUMDA6MDA6MDArMDA6MDAiLCJsaWNlbnNlX2lkIjoibGljLXRlc3QtMDAxIiwibmJmIj"
    "oiMjAyMC0wMS0wMVQwMDowMDowMCswMDowMCIsInBsdWdpbnMiOlsiZXhhbXBsZV9iaW5hcnkiXSwidiI6MX2mJ0_Wmkjk2JItN31w84"
    "ZjWAQ5qkKU5fQw12ycNRDs5DV1ivj3ZnyBTxwfLbdwwzPsz4ETIJVXaYncAPEoE1UB.eyJraWQiOiJ0ZXN0LXVua25vd24ifQ"
)

TAMPERED_SIGNATURE_TOKEN = (  # nosec B105
    "v4.public.eyJjdAN0b21lcl9pZCI6ImN1c3QtdGVzdC0wMDEiLCJleHAiOiIyMDM2LTAxLTAxVDAwOjAwOjAwKzAwOjAwIiwiZ3JhY2"
    "VfZGF5cyI6MTQsImlhdCI6IjIwMjAtMDEtMDFUMDA6MDA6MDArMDA6MDAiLCJsaWNlbnNlX2lkIjoibGljLXRlc3QtMDAxIiwibmJmIj"
    "oiMjAyMC0wMS0wMVQwMDowMDowMCswMDowMCIsInBsdWdpbnMiOlsiZXhhbXBsZV9iaW5hcnkiXSwidiI6MX1GJGqn4eqTvdTfeZBrIw"
    "_qpXJVwIy1oeV1d74yzCeVTL80CjZFdgQFJtQ5XQNzGPcb9VAbsJWeaMIgc6442ckC.eyJraWQiOiJ0ZXN0LTIwMjYtMDEifQ"
)

MISSING_PLUGINS_TOKEN = (  # nosec B105
    "v4.public.eyJjdXN0b21lcl9pZCI6ImN1c3QtdGVzdC0wMDEiLCJleHAiOiIyMDM2LTAxLTAxVDAwOjAwOjAwKzAwOjAwIiwiZ3JhY2"
    "VfZGF5cyI6MTQsImlhdCI6IjIwMjAtMDEtMDFUMDA6MDA6MDArMDA6MDAiLCJsaWNlbnNlX2lkIjoibGljLXRlc3QtMDAxIiwibmJmIj"
    "oiMjAyMC0wMS0wMVQwMDowMDowMCswMDowMCIsInYiOjF9l92xpaLwrk2NvcQ_bMJ2i7h_KXTt2DcBu409JPmDEtczJ6BRms0Z4sS_Pf"
    "zghqH9ET4xl92NGrgOvRaNko9LCA.eyJraWQiOiJ0ZXN0LTIwMjYtMDEifQ"
)

# =============================================================================
# Legacy placeholder API (predates the signed format; removal is a follow-up)
# =============================================================================


def license_token_text(status: str, plugins: list[str]) -> str:
    """A placeholder (unsigned) license token: plain JSON shaped as
    ``{"status": ..., "plugins": [...]}`` (contract: License)."""
    return json.dumps({"status": status, "plugins": plugins})


# A tampered token that is not even valid JSON (contract: License).
TAMPERED_UNPARSEABLE_TEXT = "{this is not json"


def tampered_missing_status_text(plugins: list[str]) -> str:
    """Valid JSON missing the required ``status`` key (contract: License)."""
    return json.dumps({"plugins": plugins})


def tampered_missing_plugins_text() -> str:
    """Valid JSON missing the required ``plugins`` key (contract: License)."""
    return json.dumps({"status": "valid"})
