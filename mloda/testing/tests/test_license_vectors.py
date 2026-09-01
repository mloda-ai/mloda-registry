"""Tests for the signed license-token vectors in ``mloda.testing.binary_model.license_vectors``:
the deliberately public test keypair (``test-2026-01``), the per-state builders, and the literal
tokens pinning every time-stable state the conformance kit distinguishes (spec: Test vectors;
spec: Keys, kid, rotation)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from mloda.testing.binary_model.license_token import LicenseVerificationError, verify_license_token
from mloda.testing.binary_model.license_vectors import (
    EXPIRED_TOKEN,
    MISSING_PLUGINS_TOKEN,
    NOT_YET_VALID_TOKEN,
    TAMPERED_SIGNATURE_TOKEN,
    TAMPERED_UNPARSEABLE_TEXT,
    TEST_KID,
    TEST_PUBLIC_KEY,
    TEST_PUBLIC_KEYS,
    TEST_SECRET_SEED,
    UNKNOWN_KID_TOKEN,
    VALID_TOKEN,
    WRONG_PLUGIN_TOKEN,
    expired_license_token,
    in_grace_license_token,
    missing_plugins_claim_token,
    not_yet_valid_license_token,
    tampered_signature_token,
    unknown_kid_license_token,
    valid_license_token,
)

PLUGIN_ID = "example_binary"
OTHER_PLUGIN_ID = "some_other_plugin"

# A fixed aware "now" inside VALID_TOKEN's validity window (nbf 2020, exp 2036).
NOW_2026 = datetime(2026, 3, 1, tzinfo=timezone.utc)

ALL_TOKEN_CONSTANTS = [
    pytest.param(VALID_TOKEN, id="valid"),
    pytest.param(EXPIRED_TOKEN, id="expired"),
    pytest.param(WRONG_PLUGIN_TOKEN, id="wrong_plugin"),
    pytest.param(NOT_YET_VALID_TOKEN, id="not_yet_valid"),
    pytest.param(UNKNOWN_KID_TOKEN, id="unknown_kid"),
    pytest.param(TAMPERED_SIGNATURE_TOKEN, id="tampered_signature"),
    pytest.param(MISSING_PLUGINS_TOKEN, id="missing_plugins"),
]


class TestPublishedTestKeypair:
    """The throwaway test keypair, embedded as constants and marked test-only (spec: Keys, kid,
    rotation)."""

    def test_test_kid_value(self) -> None:
        assert TEST_KID == "test-2026-01"

    def test_secret_seed_is_32_bytes(self) -> None:
        assert isinstance(TEST_SECRET_SEED, bytes)
        assert len(TEST_SECRET_SEED) == 32

    def test_public_key_is_32_bytes(self) -> None:
        assert isinstance(TEST_PUBLIC_KEY, bytes)
        assert len(TEST_PUBLIC_KEY) == 32
        assert TEST_PUBLIC_KEY != TEST_SECRET_SEED

    def test_public_keys_map(self) -> None:
        assert TEST_PUBLIC_KEYS == {TEST_KID: TEST_PUBLIC_KEY}


class TestConstantsMatchBuilders:
    """Ed25519 signing is deterministic, so each literal token must equal its builder's output;
    constants and builders cannot drift (spec: Test vectors)."""

    @pytest.mark.parametrize(
        "constant, build",
        [
            pytest.param(VALID_TOKEN, lambda: valid_license_token([PLUGIN_ID]), id="valid"),
            pytest.param(EXPIRED_TOKEN, lambda: expired_license_token([PLUGIN_ID]), id="expired"),
            pytest.param(WRONG_PLUGIN_TOKEN, lambda: valid_license_token([OTHER_PLUGIN_ID]), id="wrong_plugin"),
            pytest.param(NOT_YET_VALID_TOKEN, lambda: not_yet_valid_license_token([PLUGIN_ID]), id="not_yet_valid"),
            pytest.param(UNKNOWN_KID_TOKEN, lambda: unknown_kid_license_token([PLUGIN_ID]), id="unknown_kid"),
            pytest.param(
                TAMPERED_SIGNATURE_TOKEN, lambda: tampered_signature_token([PLUGIN_ID]), id="tampered_signature"
            ),
            pytest.param(MISSING_PLUGINS_TOKEN, lambda: missing_plugins_claim_token(), id="missing_plugins"),
        ],
    )
    def test_constant_equals_builder_output(self, constant: str, build: Callable[[], str]) -> None:
        assert constant == build()


class TestTokenShape:
    """Every literal token is one line of ASCII in the PASETO v4.public container (spec:
    Container)."""

    @pytest.mark.parametrize("token", ALL_TOKEN_CONSTANTS)
    def test_single_ascii_line_with_v4_public_header(self, token: str) -> None:
        assert isinstance(token, str)
        assert token.isascii()
        assert token == token.strip()
        assert "\n" not in token
        assert "\r" not in token
        assert token.startswith("v4.public.")

    def test_tampered_unparseable_text_is_not_a_token(self) -> None:
        assert isinstance(TAMPERED_UNPARSEABLE_TEXT, str)
        assert not TAMPERED_UNPARSEABLE_TEXT.startswith("v4.public.")


class TestVerification:
    """The vectors drive ``verify_license_token`` into every state the conformance kit needs
    (spec: Test vectors; spec: Verification)."""

    def test_valid_token_accepted_not_in_grace(self) -> None:
        result = verify_license_token(VALID_TOKEN, keys=TEST_PUBLIC_KEYS, plugin_id=PLUGIN_ID, now=NOW_2026)
        assert result.in_grace is False
        assert PLUGIN_ID in result.claims["plugins"]

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param(EXPIRED_TOKEN, id="expired"),
            pytest.param(WRONG_PLUGIN_TOKEN, id="wrong_plugin"),
            pytest.param(NOT_YET_VALID_TOKEN, id="not_yet_valid"),
            pytest.param(UNKNOWN_KID_TOKEN, id="unknown_kid"),
            pytest.param(TAMPERED_SIGNATURE_TOKEN, id="tampered_signature"),
            pytest.param(MISSING_PLUGINS_TOKEN, id="missing_plugins"),
            pytest.param(TAMPERED_UNPARSEABLE_TEXT, id="unparseable_text"),
        ],
    )
    def test_rejection_states_raise(self, text: str) -> None:
        with pytest.raises(LicenseVerificationError):
            verify_license_token(text, keys=TEST_PUBLIC_KEYS, plugin_id=PLUGIN_ID, now=NOW_2026)

    def test_in_grace_builder_verifies_in_grace_at_current_time(self) -> None:
        """The one time-relative state a constant cannot express: exp in the near past, inside
        ``grace_days`` (spec: Test vectors)."""
        token = in_grace_license_token([PLUGIN_ID])
        result = verify_license_token(token, keys=TEST_PUBLIC_KEYS, plugin_id=PLUGIN_ID)
        assert result.in_grace is True

    def test_in_grace_builder_honors_explicit_now(self) -> None:
        token = in_grace_license_token([PLUGIN_ID], now=NOW_2026)
        result = verify_license_token(token, keys=TEST_PUBLIC_KEYS, plugin_id=PLUGIN_ID, now=NOW_2026)
        assert result.in_grace is True

    def test_wrong_plugin_token_accepted_for_its_own_plugin(self) -> None:
        """WRONG_PLUGIN_TOKEN is only wrong relative to ``example_binary``; it is a perfectly valid
        token for ``some_other_plugin`` (spec: Verification step 7)."""
        result = verify_license_token(
            WRONG_PLUGIN_TOKEN, keys=TEST_PUBLIC_KEYS, plugin_id=OTHER_PLUGIN_ID, now=NOW_2026
        )
        assert result.in_grace is False
