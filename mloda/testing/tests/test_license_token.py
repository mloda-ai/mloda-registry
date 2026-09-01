"""Tests for ``mloda.testing.binary_model.license_token``: the PASETO v4.public sign/verify layer
and the claim-level license verifier defined by the license token specification (int-043).

The sign/verify layer is pinned against the official PASETO v4.public test vectors 4-S-1 to 4-S-3,
embedded verbatim below; Ed25519 signing is deterministic, so ``sign_v4_public`` must reproduce the
official tokens byte for byte (spec: Container, Test vectors). ``verify_license_token`` is then
exercised state by state against the spec's Verification steps 2-8 with tokens built inline through
``sign_license_token`` and the official vector keypair as a throwaway signing key.
"""

from __future__ import annotations

import dataclasses
import json
import struct
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

from mloda.testing.binary_model.license_token import (
    TOKEN_SCHEMA_VERSION,
    LicenseVerificationError,
    VerifiedLicense,
    pae,
    sign_license_token,
    sign_v4_public,
    verify_license_token,
    verify_v4_public,
)

# =============================================================================
# Official PASETO v4.public test vectors 4-S-1 to 4-S-3, embedded verbatim
# =============================================================================

OFFICIAL_SECRET_SEED = bytes.fromhex("b4cbfb43df4ce210727d953e4a713307fa19bb7d9f85041438d9e11b942a3774")
OFFICIAL_PUBLIC_KEY = bytes.fromhex("1eb9dbbbbc047c03fd70604e0071f0987e16b28b757225c11f00415d0e20b1a2")
OFFICIAL_PAYLOAD = b'{"data":"this is a signed message","exp":"2022-01-01T00:00:00+00:00"}'
OFFICIAL_FOOTER = b'{"kid":"zVhMiPBP9fRf2snEcT7gFTioeA9COcNy9DfgL1W60haN"}'
OFFICIAL_IMPLICIT_4_S_3 = b'{"test-vector":"4-S-3"}'

TOKEN_4_S_1 = (
    "v4.public.eyJkYXRhIjoidGhpcyBpcyBhIHNpZ25lZCBtZXNzYWdlIiwiZXhwIjoiMjAyMi0wMS0wMVQwMDowMDowMCswMDowMCJ9"
    "bg_XBBzds8lTZShVlwwKSgeKpLT3yukTw6JUz3W4h_ExsQV-P0V54zemZDcAxFaSeef1QlXEFtkqxT1ciiQEDA"
)
TOKEN_4_S_2 = (
    "v4.public.eyJkYXRhIjoidGhpcyBpcyBhIHNpZ25lZCBtZXNzYWdlIiwiZXhwIjoiMjAyMi0wMS0wMVQwMDowMDowMCswMDowMCJ9"
    "v3Jt8mx_TdM2ceTGoqwrh4yDFn0XsHvvV_D0DtwQxVrJEBMl0F2caAdgnpKlt4p7xBnx1HcO-SPo8FPp214HDw"
    ".eyJraWQiOiJ6VmhNaVBCUDlmUmYyc25FY1Q3Z0ZUaW9lQTlDT2NOeTlEZmdMMVc2MGhhTiJ9"
)
TOKEN_4_S_3 = (
    "v4.public.eyJkYXRhIjoidGhpcyBpcyBhIHNpZ25lZCBtZXNzYWdlIiwiZXhwIjoiMjAyMi0wMS0wMVQwMDowMDowMCswMDowMCJ9"
    "NPWciuD3d0o5eXJXG5pJy-DiVEoyPYWs1YSTwWHNJq6DZD3je5gf-0M4JR9ipdUSJbIovzmBECeaWmaqcaP0DQ"
    ".eyJraWQiOiJ6VmhNaVBCUDlmUmYyc25FY1Q3Z0ZUaW9lQTlDT2NOeTlEZmdMMVc2MGhhTiJ9"
)

# RFC 8032 TEST 1 public key: a known-valid Ed25519 point that is not the official vector's key,
# for the wrong-public-key rejection.
RFC8032_TEST1_PUBLIC_KEY = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")

# =============================================================================
# Claim-level fixtures: the official vector keypair doubles as a throwaway signing key
# =============================================================================

UNIT_KID = "unit-2026-01"
UNIT_KEYS: dict[str, bytes] = {UNIT_KID: OFFICIAL_PUBLIC_KEY}
PLUGIN_ID = "example_binary"

NBF = datetime(2026, 1, 1, tzinfo=timezone.utc)
EXP = datetime(2026, 6, 1, tzinfo=timezone.utc)
GRACE_END = datetime(2026, 6, 15, tzinfo=timezone.utc)  # EXP plus grace_days = 14
NOW_VALID = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _le64(value: int) -> bytes:
    """The 8-byte little-endian length encoding PAE uses (spec: Container)."""
    return struct.pack("<Q", value)


def _claims(**overrides: Any) -> dict[str, Any]:
    """A fully valid claim set matching NBF/EXP/GRACE_END (spec: Claims); keyword arguments
    override individual claims."""
    claims: dict[str, Any] = {
        "v": TOKEN_SCHEMA_VERSION,
        "license_id": "lic-unit-001",
        "customer_id": "cust-unit-001",
        "plugins": [PLUGIN_ID],
        "iat": "2026-01-01T00:00:00+00:00",
        "nbf": "2026-01-01T00:00:00+00:00",
        "exp": "2026-06-01T00:00:00+00:00",
        "grace_days": 14,
    }
    claims.update(overrides)
    return claims


def _signed(claims: dict[str, Any]) -> str:
    return sign_license_token(claims, kid=UNIT_KID, secret_seed=OFFICIAL_SECRET_SEED)


def _flip_payload_char(token: str) -> str:
    """Replace one character inside the payload segment with a different base64url character, so
    the Ed25519 signature no longer matches (spec: Verification step 4)."""
    parts = token.split(".")
    body = parts[2]
    index = 5
    replacement = "A" if body[index] != "A" else "B"
    parts[2] = body[:index] + replacement + body[index + 1 :]
    return ".".join(parts)


def test_token_schema_version_is_one() -> None:
    """This module implements token schema version 1 (spec: Claims)."""
    assert TOKEN_SCHEMA_VERSION == 1


class TestPae:
    """Hand-computed PAE examples: LE64 piece count, then LE64 length plus bytes per piece
    (spec: Container)."""

    def test_empty_list(self) -> None:
        assert pae([]) == b"\x00" * 8

    def test_single_empty_piece(self) -> None:
        assert pae([b""]) == _le64(1) + _le64(0)

    def test_single_piece(self) -> None:
        assert pae([b"test"]) == _le64(1) + _le64(4) + b"test"

    def test_two_pieces(self) -> None:
        assert pae([b"a", b"bc"]) == _le64(2) + _le64(1) + b"a" + _le64(2) + b"bc"


class TestSignV4Public:
    """``sign_v4_public`` reproduces the official PASETO v4.public tokens byte for byte; Ed25519
    signing is deterministic (spec: Container, Test vectors)."""

    def test_vector_4_s_1_empty_footer(self) -> None:
        token = sign_v4_public(OFFICIAL_PAYLOAD, b"", secret_seed=OFFICIAL_SECRET_SEED)
        assert token == TOKEN_4_S_1

    def test_vector_4_s_2_kid_footer(self) -> None:
        token = sign_v4_public(OFFICIAL_PAYLOAD, OFFICIAL_FOOTER, secret_seed=OFFICIAL_SECRET_SEED)
        assert token == TOKEN_4_S_2

    def test_vector_4_s_3_implicit_assertion(self) -> None:
        token = sign_v4_public(
            OFFICIAL_PAYLOAD, OFFICIAL_FOOTER, secret_seed=OFFICIAL_SECRET_SEED, implicit=OFFICIAL_IMPLICIT_4_S_3
        )
        assert token == TOKEN_4_S_3

    def test_empty_footer_omits_footer_segment(self) -> None:
        """With an empty footer the token has exactly the three ``v4.public.<body>`` segments, no
        trailing dot (spec: Container)."""
        token = sign_v4_public(OFFICIAL_PAYLOAD, b"", secret_seed=OFFICIAL_SECRET_SEED)
        assert token.count(".") == 2


class TestVerifyV4Public:
    """Container parsing and signature verification (spec: Container, Verification steps 2 and 4)."""

    def test_vector_4_s_1_round_trip(self) -> None:
        payload, footer = verify_v4_public(TOKEN_4_S_1, public_key=OFFICIAL_PUBLIC_KEY)
        assert payload == OFFICIAL_PAYLOAD
        assert footer == b""

    def test_vector_4_s_2_round_trip(self) -> None:
        payload, footer = verify_v4_public(TOKEN_4_S_2, public_key=OFFICIAL_PUBLIC_KEY)
        assert payload == OFFICIAL_PAYLOAD
        assert footer == OFFICIAL_FOOTER

    def test_vector_4_s_3_round_trip_with_implicit(self) -> None:
        payload, footer = verify_v4_public(
            TOKEN_4_S_3, public_key=OFFICIAL_PUBLIC_KEY, implicit=OFFICIAL_IMPLICIT_4_S_3
        )
        assert payload == OFFICIAL_PAYLOAD
        assert footer == OFFICIAL_FOOTER

    def test_tampered_payload_rejected(self) -> None:
        with pytest.raises(LicenseVerificationError):
            verify_v4_public(_flip_payload_char(TOKEN_4_S_1), public_key=OFFICIAL_PUBLIC_KEY)

    def test_wrong_public_key_rejected(self) -> None:
        with pytest.raises(LicenseVerificationError):
            verify_v4_public(TOKEN_4_S_1, public_key=RFC8032_TEST1_PUBLIC_KEY)

    @pytest.mark.parametrize("header", ["v4.local.", "v3.public."])
    def test_wrong_header_rejected(self, header: str) -> None:
        """The header must be exactly ``v4.public.`` (spec: Verification step 2)."""
        wrong = header + TOKEN_4_S_1[len("v4.public.") :]
        with pytest.raises(LicenseVerificationError):
            verify_v4_public(wrong, public_key=OFFICIAL_PUBLIC_KEY)

    def test_undecodable_base64_rejected(self) -> None:
        with pytest.raises(LicenseVerificationError):
            verify_v4_public("v4.public.!!!not-base64url!!!", public_key=OFFICIAL_PUBLIC_KEY)

    def test_implicit_mismatch_rejected(self) -> None:
        """A token signed with a non-empty implicit assertion must not verify with an empty one:
        the implicit assertion is part of the PAE signing input (spec: Container)."""
        with pytest.raises(LicenseVerificationError):
            verify_v4_public(TOKEN_4_S_3, public_key=OFFICIAL_PUBLIC_KEY)


class TestLicenseVerificationError:
    """The error carries a single human-readable line in ``reason`` and stringifies to it."""

    def test_reason_attribute_and_str(self) -> None:
        error = LicenseVerificationError("token expired")
        assert error.reason == "token expired"
        assert str(error) == "token expired"

    def test_is_an_exception(self) -> None:
        assert issubclass(LicenseVerificationError, Exception)


class TestSignLicenseToken:
    """``sign_license_token`` serializes claims as compact sorted JSON with exactly a
    ``{"kid": ...}`` footer and an empty implicit assertion (spec: Container)."""

    def test_payload_is_compact_sorted_json_and_footer_carries_kid(self) -> None:
        claims: dict[str, Any] = {"b": 2, "a": 1}
        token = sign_license_token(claims, kid=UNIT_KID, secret_seed=OFFICIAL_SECRET_SEED)
        payload, footer = verify_v4_public(token, public_key=OFFICIAL_PUBLIC_KEY)
        assert payload == b'{"a":1,"b":2}'
        assert footer == b'{"kid":"unit-2026-01"}'


class TestVerifyLicenseTokenAcceptance:
    """Clock windows and claim sets the verifier accepts (spec: Verification steps 5 and 6)."""

    def test_valid_token_accepted(self) -> None:
        result = verify_license_token(_signed(_claims()), keys=UNIT_KEYS, plugin_id=PLUGIN_ID, now=NOW_VALID)
        assert isinstance(result, VerifiedLicense)
        assert result.in_grace is False
        assert result.claims == _claims()

    def test_now_equal_to_nbf_is_valid(self) -> None:
        """``nbf <= now`` includes the boundary (spec: Verification step 6)."""
        result = verify_license_token(_signed(_claims()), keys=UNIT_KEYS, plugin_id=PLUGIN_ID, now=NBF)
        assert result.in_grace is False

    def test_now_equal_to_exp_is_valid_not_in_grace(self) -> None:
        """``now <= exp`` includes the boundary and is not yet grace (spec: Verification step 6)."""
        result = verify_license_token(_signed(_claims()), keys=UNIT_KEYS, plugin_id=PLUGIN_ID, now=EXP)
        assert result.in_grace is False

    def test_past_exp_within_grace_is_in_grace(self) -> None:
        now = EXP + timedelta(days=1)
        result = verify_license_token(_signed(_claims()), keys=UNIT_KEYS, plugin_id=PLUGIN_ID, now=now)
        assert result.in_grace is True

    def test_now_equal_to_grace_end_is_still_in_grace(self) -> None:
        """``now <= exp + grace_days`` includes the boundary (spec: Verification step 6)."""
        result = verify_license_token(_signed(_claims()), keys=UNIT_KEYS, plugin_id=PLUGIN_ID, now=GRACE_END)
        assert result.in_grace is True

    def test_unknown_claims_are_ignored(self) -> None:
        """Unknown claims must be ignored for forward compatibility (spec: Claims)."""
        token = _signed(_claims(seat_count=5, unknown_extra="x"))
        result = verify_license_token(token, keys=UNIT_KEYS, plugin_id=PLUGIN_ID, now=NOW_VALID)
        assert result.in_grace is False

    def test_verified_license_is_frozen(self) -> None:
        result = verify_license_token(_signed(_claims()), keys=UNIT_KEYS, plugin_id=PLUGIN_ID, now=NOW_VALID)
        with pytest.raises(dataclasses.FrozenInstanceError):
            # setattr instead of plain assignment so mypy does not reject the file once the frozen
            # dataclass exists; the runtime rejection is exactly what is under test.
            setattr(result, "in_grace", True)


class TestVerifyLicenseTokenRejections:
    """Every rejection state raises ``LicenseVerificationError`` (spec: Verification steps 2-7)."""

    def _reject(self, token: str, now: datetime = NOW_VALID) -> LicenseVerificationError:
        with pytest.raises(LicenseVerificationError) as excinfo:
            verify_license_token(token, keys=UNIT_KEYS, plugin_id=PLUGIN_ID, now=now)
        return excinfo.value

    def test_one_second_past_grace_rejected_as_expired(self) -> None:
        error = self._reject(_signed(_claims()), now=GRACE_END + timedelta(seconds=1))
        assert "expired" in error.reason.lower()

    def test_now_before_nbf_rejected(self) -> None:
        self._reject(_signed(_claims()), now=NBF - timedelta(seconds=1))

    def test_unknown_kid_rejected(self) -> None:
        """A ``kid`` outside the verifier's key map is a rejection (spec: Verification step 3)."""
        token = sign_license_token(_claims(), kid="unit-9999-99", secret_seed=OFFICIAL_SECRET_SEED)
        error = self._reject(token)
        assert "kid" in error.reason

    def test_missing_plugins_claim_rejected(self) -> None:
        claims = _claims()
        del claims["plugins"]
        self._reject(_signed(claims))

    def test_empty_plugins_list_rejected(self) -> None:
        self._reject(_signed(_claims(plugins=[])))

    def test_non_string_plugins_entries_rejected(self) -> None:
        self._reject(_signed(_claims(plugins=[1, 2])))

    def test_plugin_id_not_entitled_rejected(self) -> None:
        """The verifying binary's ``plugin_id`` must be in ``plugins`` (spec: Verification
        step 7)."""
        with pytest.raises(LicenseVerificationError) as excinfo:
            verify_license_token(_signed(_claims()), keys=UNIT_KEYS, plugin_id="some_other_plugin", now=NOW_VALID)
        assert "entitle" in excinfo.value.reason.lower()

    def test_schema_version_two_rejected(self) -> None:
        self._reject(_signed(_claims(v=2)))

    def test_missing_required_claim_rejected(self) -> None:
        claims = _claims()
        del claims["exp"]
        self._reject(_signed(claims))

    def test_wrongly_typed_grace_days_rejected(self) -> None:
        self._reject(_signed(_claims(grace_days="14")))

    def test_negative_grace_days_rejected(self) -> None:
        self._reject(_signed(_claims(grace_days=-1)))

    def test_footer_without_kid_rejected(self) -> None:
        """A correctly signed token whose footer object lacks a string ``kid`` (spec: Verification
        step 2)."""
        payload = json.dumps(_claims(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        token = sign_v4_public(payload, b'{"note":"no kid"}', secret_seed=OFFICIAL_SECRET_SEED)
        error = self._reject(token)
        assert "kid" in error.reason

    def test_footer_not_a_json_object_rejected(self) -> None:
        payload = json.dumps(_claims(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        token = sign_v4_public(payload, b"not a json object", secret_seed=OFFICIAL_SECRET_SEED)
        self._reject(token)

    def test_payload_not_a_json_object_rejected(self) -> None:
        """A signed payload that is valid JSON but not an object (spec: Verification step 5)."""
        footer = json.dumps({"kid": UNIT_KID}, separators=(",", ":")).encode("utf-8")
        token = sign_v4_public(b'["not", "an", "object"]', footer, secret_seed=OFFICIAL_SECRET_SEED)
        self._reject(token)

    def test_not_a_token_text_rejected(self) -> None:
        self._reject("this is not a paseto token")

    def test_tampered_payload_rejected(self) -> None:
        self._reject(_flip_payload_char(_signed(_claims())))


class TestMaxReleaseDate:
    """The optional ``max_release_date`` claim against the binary's release date (spec:
    Verification step 8)."""

    def test_release_date_after_max_rejected(self) -> None:
        token = _signed(_claims(max_release_date="2026-03-01"))
        with pytest.raises(LicenseVerificationError):
            verify_license_token(
                token, keys=UNIT_KEYS, plugin_id=PLUGIN_ID, now=NOW_VALID, release_date=date(2026, 3, 2)
            )

    def test_release_date_on_max_accepted(self) -> None:
        token = _signed(_claims(max_release_date="2026-03-01"))
        result = verify_license_token(
            token, keys=UNIT_KEYS, plugin_id=PLUGIN_ID, now=NOW_VALID, release_date=date(2026, 3, 1)
        )
        assert result.in_grace is False

    def test_absent_claim_ignores_release_date(self) -> None:
        result = verify_license_token(
            _signed(_claims()), keys=UNIT_KEYS, plugin_id=PLUGIN_ID, now=NOW_VALID, release_date=date(2099, 1, 1)
        )
        assert result.in_grace is False
