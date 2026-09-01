"""PASETO v4.public license tokens (contract: License): PAE and Ed25519 sign/verify built on
``cryptography``, plus the claim-level verifier implementing the license token specification's
Verification steps 2-8 (spec: Container, Claims, Verification). The signing helpers double as the
issuance tooling (spec: Issuer decision); the sign/verify layer is cross-checked against the
official PASETO v4.public test vectors by this module's tests (spec: Test vectors).
"""

from __future__ import annotations

import base64
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

# The token schema version this module signs and the only one it verifies (spec: Claims).
TOKEN_SCHEMA_VERSION = 1

_HEADER = "v4.public."
_SIGNATURE_LENGTH = 64


class LicenseVerificationError(Exception):
    """A license token failed verification; ``reason`` is one human-readable line naming the
    failure (spec: Verification)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class VerifiedLicense:
    """A successfully verified license: the full claim set and whether the token is only valid
    through its grace window (spec: Verification step 6)."""

    claims: dict[str, Any]
    in_grace: bool


def pae(pieces: list[bytes]) -> bytes:
    """Pre-Authentication Encoding: LE64 piece count, then LE64 length plus bytes per piece
    (spec: Container)."""
    encoded = struct.pack("<Q", len(pieces))
    for piece in pieces:
        encoded += struct.pack("<Q", len(piece)) + piece
    return encoded


def _b64url_encode(data: bytes) -> str:
    """base64url without padding, the encoding of both token segments (spec: Container)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    """Decode a base64url token segment, re-adding padding; a segment outside the base64url
    alphabet is a rejection (spec: Verification step 2)."""
    padded = segment + "=" * (-len(segment) % 4)
    try:
        return base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
    except ValueError as error:
        raise LicenseVerificationError(f"license token segment is not base64url: {error}") from error


def sign_v4_public(payload: bytes, footer: bytes, *, secret_seed: bytes, implicit: bytes = b"") -> str:
    """Sign a PASETO v4.public token: Ed25519 over ``pae([header, payload, footer, implicit])``,
    with the footer segment present only when the footer is non-empty (spec: Container)."""
    private_key = Ed25519PrivateKey.from_private_bytes(secret_seed)
    signature = private_key.sign(pae([_HEADER.encode("ascii"), payload, footer, implicit]))
    token = _HEADER + _b64url_encode(payload + signature)
    if footer:
        token += "." + _b64url_encode(footer)
    return token


def _split_token(token: str) -> tuple[bytes, bytes]:
    """Parse the container into decoded ``(body, footer)`` bytes; the header must be exactly
    ``v4.public.`` and both segments must decode (spec: Verification step 2)."""
    if not token.startswith(_HEADER):
        raise LicenseVerificationError("license token is not a PASETO v4.public token")
    segments = token[len(_HEADER) :].split(".")
    if len(segments) not in (1, 2):
        raise LicenseVerificationError("license token has more segments than 'v4.public.<body>.<footer>'")
    body = _b64url_decode(segments[0])
    footer = _b64url_decode(segments[1]) if len(segments) == 2 else b""
    if len(body) < _SIGNATURE_LENGTH:
        raise LicenseVerificationError("license token body is shorter than an Ed25519 signature")
    return body, footer


def verify_v4_public(token: str, *, public_key: bytes, implicit: bytes = b"") -> tuple[bytes, bytes]:
    """Verify a PASETO v4.public token and return ``(payload, footer)``; every failure raises
    ``LicenseVerificationError`` (spec: Verification steps 2 and 4)."""
    body, footer = _split_token(token)
    payload, signature = body[:-_SIGNATURE_LENGTH], body[-_SIGNATURE_LENGTH:]
    try:
        key = Ed25519PublicKey.from_public_bytes(public_key)
    except ValueError as error:
        raise LicenseVerificationError(f"invalid Ed25519 public key: {error}") from error
    try:
        key.verify(signature, pae([_HEADER.encode("ascii"), payload, footer, implicit]))
    except InvalidSignature as error:
        raise LicenseVerificationError("license token signature does not verify") from error
    return payload, footer


def sign_license_token(claims: dict[str, Any], *, kid: str, secret_seed: bytes) -> str:
    """Sign a license token: compact sorted-JSON claims payload, exactly a ``{"kid": ...}`` footer
    and an empty implicit assertion (spec: Container)."""
    payload = json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8")
    footer = json.dumps({"kid": kid}, separators=(",", ":")).encode("utf-8")
    return sign_v4_public(payload, footer, secret_seed=secret_seed)


def _parse_footer_kid(footer: bytes) -> str:
    """The footer must be a JSON object with a string ``kid`` (spec: Verification step 2)."""
    try:
        parsed = json.loads(footer.decode("utf-8"))
    except ValueError as error:
        raise LicenseVerificationError(f"license footer is not JSON: {error}") from error
    kid = parsed.get("kid") if isinstance(parsed, dict) else None
    if not isinstance(kid, str):
        raise LicenseVerificationError("license footer must be a JSON object with a string 'kid'")
    return kid


def _parse_payload(payload: bytes) -> dict[str, Any]:
    """The signed payload must be one JSON object (spec: Verification step 5)."""
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except ValueError as error:
        raise LicenseVerificationError(f"license payload is not JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise LicenseVerificationError("license payload must be a JSON object")
    return parsed


def _parse_timestamp(claims: dict[str, Any], name: str) -> datetime:
    """A required RFC 3339 timestamp claim; a ``Z`` suffix is normalized to ``+00:00`` and a naive
    timestamp is a rejection (spec: Claims)."""
    value = claims.get(name)
    if not isinstance(value, str):
        raise LicenseVerificationError(f"claim '{name}' is missing or not an RFC 3339 timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LicenseVerificationError(f"claim '{name}' is not an RFC 3339 timestamp: {error}") from error
    if parsed.tzinfo is None:
        raise LicenseVerificationError(f"claim '{name}' must carry a UTC offset")
    return parsed


def _parse_max_release_date(claims: dict[str, Any]) -> date | None:
    """The optional ``max_release_date`` claim, a ``YYYY-MM-DD`` date when present (spec:
    Claims)."""
    value = claims.get("max_release_date")
    if value is None:
        return None
    if not isinstance(value, str):
        raise LicenseVerificationError("claim 'max_release_date' must be a YYYY-MM-DD date string")
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise LicenseVerificationError(f"claim 'max_release_date' is not a YYYY-MM-DD date: {error}") from error


def _validate_claims(claims: dict[str, Any]) -> tuple[datetime, datetime, int, list[str], date | None]:
    """Check ``v == 1`` and the required claims' presence and types; unknown claims are ignored
    (spec: Claims; spec: Verification step 5)."""
    version = claims.get("v")
    if isinstance(version, bool) or version != TOKEN_SCHEMA_VERSION:
        raise LicenseVerificationError(f"unsupported token schema version {version!r}; this verifier expects v == 1")
    for name in ("license_id", "customer_id"):
        if not isinstance(claims.get(name), str):
            raise LicenseVerificationError(f"claim '{name}' is missing or not a string")
    plugins = claims.get("plugins")
    if not isinstance(plugins, list) or not plugins or not all(isinstance(entry, str) for entry in plugins):
        raise LicenseVerificationError("claim 'plugins' is missing or not a non-empty list of strings")
    _parse_timestamp(claims, "iat")
    nbf = _parse_timestamp(claims, "nbf")
    exp = _parse_timestamp(claims, "exp")
    grace_days = claims.get("grace_days")
    if isinstance(grace_days, bool) or not isinstance(grace_days, int) or grace_days < 0:
        raise LicenseVerificationError("claim 'grace_days' is missing or not a non-negative integer")
    return nbf, exp, grace_days, plugins, _parse_max_release_date(claims)


def _check_clock(now: datetime, *, nbf: datetime, exp: datetime, grace_days: int) -> bool:
    """The clock windows, boundaries inclusive; returns whether the token is only valid through its
    grace window (spec: Verification step 6)."""
    if now < nbf:
        raise LicenseVerificationError(f"license not yet valid: nbf is {nbf.isoformat()}")
    if now <= exp:
        return False
    if now <= exp + timedelta(days=grace_days):
        return True
    raise LicenseVerificationError(f"license expired: exp was {exp.isoformat()} plus {grace_days} grace days")


def verify_license_token(
    text: str,
    *,
    keys: Mapping[str, bytes],
    plugin_id: str,
    now: datetime | None = None,
    release_date: date | None = None,
) -> VerifiedLicense:
    """Verify a license token against the specification's Verification steps 2-8, in order:
    container and footer ``kid``, key lookup, signature, claims, clock rules, entitlement,
    ``max_release_date`` (contract: License; spec: Verification)."""
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise ValueError("now must be a timezone-aware datetime")
    token = text.strip()
    _, footer = _split_token(token)
    kid = _parse_footer_kid(footer)
    if kid not in keys:
        raise LicenseVerificationError(f"unknown license key id (kid) '{kid}'")
    payload, _ = verify_v4_public(token, public_key=keys[kid])
    claims = _parse_payload(payload)
    nbf, exp, grace_days, plugins, max_release_date = _validate_claims(claims)
    in_grace = _check_clock(moment, nbf=nbf, exp=exp, grace_days=grace_days)
    if plugin_id not in plugins:
        raise LicenseVerificationError(f"license does not entitle plugin '{plugin_id}'")
    if max_release_date is not None and release_date is not None and release_date > max_release_date:
        raise LicenseVerificationError(
            f"binary release date {release_date.isoformat()} is after the license's "
            f"max_release_date {max_release_date.isoformat()}"
        )
    return VerifiedLicense(claims=claims, in_grace=in_grace)
