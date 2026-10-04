"""Run manifest signers: the signing protocol and the HMAC and Ed25519 implementations."""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Iterable
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

_MIN_KEY_BYTES = 32
_ED25519_KEY_BYTES = 32
_ED25519_SIGNATURE = re.compile(r"[0-9a-f]{128}")


class ManifestSigner(Protocol):
    """Signs and verifies manifest payloads; verify returns False instead of raising."""

    algorithm: str
    key_id: str

    def sign(self, payload: bytes) -> str: ...

    def verify(self, payload: bytes, signature: str) -> bool: ...


def _check_key_id(owner: str, key_id: object) -> None:
    if not isinstance(key_id, str) or not key_id.strip():
        raise ValueError(f"{owner} key_id must be a non-blank string")


class HmacSha256Signer:
    """HMAC-SHA256 over a shared key of at least 32 bytes."""

    algorithm = "HMAC-SHA256"

    def __init__(self, key: bytes, key_id: str) -> None:
        if not isinstance(key, bytes) or len(key) < _MIN_KEY_BYTES:
            raise ValueError(f"HmacSha256Signer key must be bytes of at least {_MIN_KEY_BYTES} bytes")
        _check_key_id("HmacSha256Signer", key_id)
        self._key = key
        self.key_id = key_id

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key_id={self.key_id!r})"

    def sign(self, payload: bytes) -> str:
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature: str) -> bool:
        if not isinstance(signature, str):
            return False
        # compare_digest raises TypeError on a non-ASCII str.
        return signature.isascii() and hmac.compare_digest(self.sign(payload), signature)


def _ed25519() -> ModuleType:
    """The cryptography ed25519 module, imported per call because the extra is optional."""
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:
        raise ImportError(
            "Ed25519Signer needs the 'cryptography' package: pip install mloda-enterprise[ed25519]"
        ) from exc
    return ed25519


def _check_ed25519_key(name: str, key: object) -> None:
    if not isinstance(key, bytes) or len(key) != _ED25519_KEY_BYTES:
        raise ValueError(f"Ed25519Signer {name} must be bytes of exactly {_ED25519_KEY_BYTES} bytes")


class Ed25519Signer:
    """Ed25519 over raw 32-byte keys (`private_bytes_raw()`, `public_bytes_raw()`); needs the `ed25519` extra."""

    algorithm = "Ed25519"

    def __init__(self, private_key: bytes, key_id: str) -> None:
        _check_ed25519_key("private_key", private_key)
        _check_key_id("Ed25519Signer", key_id)
        private: Ed25519PrivateKey = _ed25519().Ed25519PrivateKey.from_private_bytes(private_key)
        self._private: Ed25519PrivateKey | None = private
        self._public: Ed25519PublicKey = private.public_key()
        self.key_id = key_id

    @classmethod
    def from_public_key(cls, public_key: bytes, key_id: str) -> Ed25519Signer:
        """A verify-only signer: `sign` raises ValueError."""
        _check_ed25519_key("public_key", public_key)
        _check_key_id("Ed25519Signer", key_id)
        signer = cls.__new__(cls)
        signer._private = None
        signer._public = _ed25519().Ed25519PublicKey.from_public_bytes(public_key)
        signer.key_id = key_id
        return signer

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key_id={self.key_id!r})"

    def sign(self, payload: bytes) -> str:
        if self._private is None:
            raise ValueError(f"{type(self).__name__} {self.key_id!r} holds only a public key and cannot sign")
        return self._private.sign(payload).hex()

    def verify(self, payload: bytes, signature: str) -> bool:
        if not isinstance(signature, str) or not _ED25519_SIGNATURE.fullmatch(signature):
            return False
        from cryptography.exceptions import InvalidSignature

        try:
            self._public.verify(bytes.fromhex(signature), payload)
        except InvalidSignature:
            return False
        return True


def _signer_map(signer: ManifestSigner, previous_signers: Iterable[ManifestSigner]) -> dict[str, ManifestSigner]:
    """Map key_id to signer, from `signer` plus `previous_signers`; raise ValueError for a shared key_id."""
    signers: dict[str, ManifestSigner] = {signer.key_id: signer}
    for previous in previous_signers:
        if previous.key_id in signers:
            raise ValueError(f"previous_signers repeats key_id {previous.key_id!r}")
        signers[previous.key_id] = previous
    return signers
