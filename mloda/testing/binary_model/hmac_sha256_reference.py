"""The "hmac_sha256" operation's reference algorithm: one independent implementation, imported by both
the hmac fake binary and ``conformance.py``'s ``HmacSha256OperationConformanceMixin`` (so expected
values never derive from whatever the binary happens to do). No dependency back on either caller.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Sequence

KEY_PARAMETER = "key"

# Known answer for HMAC-SHA256 over the utf8 bytes of KNOWN_ANSWER_VALUE (lowercase hex digest).
KNOWN_ANSWER_KEY = "11" * 32
KNOWN_ANSWER_VALUE = "alice"
KNOWN_ANSWER_DIGEST = "ea497bac56964937ed2d5a6a77241026a609b5fd149f6eea3d41316e94983325"

_KEY_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def parse_hmac_key(key: object) -> bytes:
    """The 32 key bytes of a 64-character hex string; anything else raises ``ValueError`` without
    echoing the key. Stricter than ``bytes.fromhex``, which tolerates whitespace."""
    if not isinstance(key, str) or _KEY_PATTERN.fullmatch(key) is None:
        raise ValueError("hmac key must be a 64-character hex string")
    return bytes.fromhex(key)


def _digest(key: bytes, value: str | None) -> str | None:
    if value is None:
        return None
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()


def compute_expected_hmac_sha256(key_hex: str, value: str | None) -> str | None:
    """HMAC-SHA256 of ``value``'s utf8 bytes as lowercase hex; ``None`` stays ``None``."""
    return _digest(parse_hmac_key(key_hex), value)


def compute_expected_hmac_sha256_column(values: Sequence[str | None], key_hex: str) -> list[str | None]:
    """Apply ``compute_expected_hmac_sha256`` row by row, parsing the key once."""
    key = parse_hmac_key(key_hex)
    return [_digest(key, value) for value in values]
