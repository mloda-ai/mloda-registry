"""Tests for the manifest signers."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from mloda.enterprise.extenders.audit import (
    Ed25519Signer,
    HmacSha256Signer,
    KeyAlreadyCurrentError,
    ManifestVerificationError,
    manifest_hash,
    rotate_manifest_key,
    seal_ndjson_runs,
    verify_ndjson_log,
    verify_ndjson_log_coverage,
)
from mloda.enterprise.extenders.audit.tests.manifest_helpers import (
    _BAD_SIGNATURE,
    _KEY,
    _OTHER_KEY,
    _THIRD_KEY,
    _both_algorithms,
    _log_with_rotation_entry,
    _public_key,
    _quarantine,
    _quarantine_from_entry,
    _read_lines,
    _record,
    _rotate,
    _rotated_three_key_log,
    _sealed_log,
    _signer,
    _signer_class,
    _snapshot,
    _spans,
    _summary,
    _torn_manifest_log,
    _verify_only,
    _write_records,
)
from mloda.testing.import_isolation import block_root, evict_package

# RFC 8032 section 7.1, test 1: an empty message.
_RFC8032_SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")


_RFC8032_PUBLIC_KEY = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")


_RFC8032_SIGNATURE = (
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555"
    "fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
)


# With `_both_algorithms`, narrows a class to Ed25519 (public-key signers exist only there).
_ed25519_only = pytest.mark.parametrize("algorithm", ["ed25519"], indirect=True)


# Signatures that are not a str at all, one of which is the valid signature as bytes.
_NON_STR_SIGNATURES: dict[str, Callable[[str], Any]] = {
    "none": lambda signature: None,
    "bytes": lambda signature: signature.encode("ascii"),
    "int": lambda signature: 123,
    "list": lambda signature: ["a"],
}


@_both_algorithms
class TestManifestSignerContract:
    def test_verify_accepts_its_own_signature(self) -> None:
        signer = _signer()

        assert signer.verify(b"payload", signer.sign(b"payload")) is True

    def test_verify_rejects_a_tampered_payload(self) -> None:
        signer = _signer()

        assert signer.verify(b"payload-tampered", signer.sign(b"payload")) is False

    def test_verify_rejects_a_tampered_signature(self) -> None:
        signer = _signer()
        signature = signer.sign(b"payload")
        tampered = ("1" if signature[0] == "0" else "0") + signature[1:]

        assert signer.verify(b"payload", tampered) is False

    def test_verify_rejects_the_signature_of_a_different_key(self) -> None:
        signature = _signer(_OTHER_KEY).sign(b"payload")

        assert _signer().verify(b"payload", signature) is False

    @pytest.mark.parametrize("signature", ["é" * 64, chr(0xD800)], ids=["non-ascii", "lone-surrogate"])
    def test_verify_returns_false_for_a_signature_it_cannot_compare(self, signature: str) -> None:
        assert _signer().verify(b"payload", signature) is False

    @pytest.mark.parametrize("make", list(_NON_STR_SIGNATURES.values()), ids=list(_NON_STR_SIGNATURES))
    def test_verify_returns_false_and_never_raises_for_a_signature_that_is_not_a_str(
        self, make: Callable[[str], Any]
    ) -> None:
        signer = _signer()
        # Control: the original verifies.
        assert signer.verify(b"payload", signer.sign(b"payload")) is True

        assert signer.verify(b"payload", make(signer.sign(b"payload"))) is False

    @pytest.mark.parametrize("key_id", ["", "   ", "\t\n"])
    def test_blank_key_id_raises_value_error(self, key_id: str) -> None:
        with pytest.raises(ValueError):
            _signer_class()(_KEY, key_id)

    def test_non_str_key_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _signer_class()(_KEY, None)  # type: ignore[arg-type]

    def test_repr_does_not_contain_the_key(self) -> None:
        key = b"do-not-print-this-signing-key-01"
        signer = _signer_class()(key, "key-1")

        for text in (repr(signer), str(signer)):
            assert key.decode("utf-8") not in text
            assert key.hex() not in text


class TestHmacSha256Signer:
    def test_algorithm_and_key_id(self) -> None:
        signer = HmacSha256Signer(_KEY, "key-2026-01")

        assert signer.algorithm == "HMAC-SHA256"
        assert signer.key_id == "key-2026-01"

    def test_sign_is_the_lowercase_hex_hmac_sha256(self) -> None:
        payload = b'{"a": 1}'

        assert _signer().sign(payload) == hmac.new(_KEY, payload, hashlib.sha256).hexdigest()

    @pytest.mark.parametrize("key", [b"", b"k" * 31])
    def test_short_key_raises_value_error(self, key: bytes) -> None:
        with pytest.raises(ValueError):
            HmacSha256Signer(key, "key-1")

    def test_non_bytes_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            HmacSha256Signer("k" * 32, "key-1")  # type: ignore[arg-type]

    def test_repr_does_not_contain_the_key(self) -> None:
        key = b"do-not-print-this-signing-key-0123456789"
        signer = HmacSha256Signer(key, "key-1")

        for text in (repr(signer), str(signer)):
            assert key.decode("utf-8") not in text
            assert key.hex() not in text


# Forms of a valid signature that verify must refuse, some of which a lenient hex parser would accept.
_MANGLED_SIGNATURES: dict[str, Callable[[str], str]] = {
    "uppercase": str.upper,
    "leading-space": lambda signature: " " + signature,
    "trailing-newline": lambda signature: signature + "\n",
    "spaced-byte-pairs": lambda signature: " ".join(signature[i : i + 2] for i in range(0, len(signature), 2)),
    "127-chars": lambda signature: signature[:-1],
    "129-chars": lambda signature: signature + "0",
    "non-hex": lambda signature: "g" + signature[1:],
    "trailing-spaces-at-128": lambda signature: signature[:-2] + "  ",
    "empty": lambda signature: "",
    "non-ascii": lambda signature: "é" * len(signature),
    "lone-surrogate": lambda signature: chr(0xD800) * len(signature),
}


_not_a_32_byte_key = pytest.mark.parametrize(
    "key", [b"", b"k" * 31, b"k" * 33, "k" * 32], ids=["empty", "31-bytes", "33-bytes", "str"]
)


class TestEd25519Signer:
    def test_algorithm_and_key_id(self) -> None:
        signer = Ed25519Signer(_KEY, "key-2026-01")

        assert signer.algorithm == "Ed25519"
        assert signer.key_id == "key-2026-01"

    def test_sign_matches_the_rfc_8032_test_vector(self) -> None:
        signer = Ed25519Signer(_RFC8032_SEED, "rfc-8032")

        assert signer.sign(b"") == _RFC8032_SIGNATURE
        assert signer.verify(b"", _RFC8032_SIGNATURE) is True

    def test_a_public_key_signer_verifies_the_rfc_8032_test_vector(self) -> None:
        verifier = Ed25519Signer.from_public_key(_RFC8032_PUBLIC_KEY, "rfc-8032")

        assert verifier.verify(b"", _RFC8032_SIGNATURE) is True
        assert verifier.verify(b"tampered", _RFC8032_SIGNATURE) is False

    def test_sign_returns_128_lowercase_hex_characters(self) -> None:
        signature = Ed25519Signer(_KEY, "key-1").sign(b"payload")

        assert re.fullmatch(r"[0-9a-f]{128}", signature)

    @pytest.mark.parametrize("public_only", [False, True], ids=["private", "public-only"])
    @pytest.mark.parametrize("mangle", list(_MANGLED_SIGNATURES.values()), ids=list(_MANGLED_SIGNATURES))
    def test_verify_returns_false_for_anything_but_128_lowercase_hex_characters(
        self, mangle: Callable[[str], str], public_only: bool
    ) -> None:
        signature = Ed25519Signer(_KEY, "key-1").sign(b"payload")
        signer = _verify_only() if public_only else Ed25519Signer(_KEY, "key-1")
        mangled = mangle(signature)
        # Control: the original verifies and the mangled form differs.
        assert signer.verify(b"payload", signature) is True
        assert mangled != signature

        assert signer.verify(b"payload", mangled) is False

    @pytest.mark.parametrize("make", list(_NON_STR_SIGNATURES.values()), ids=list(_NON_STR_SIGNATURES))
    def test_a_public_key_signer_returns_false_and_never_raises_for_a_signature_that_is_not_a_str(
        self, make: Callable[[str], Any]
    ) -> None:
        signature = Ed25519Signer(_KEY, "key-1").sign(b"payload")
        signer = _verify_only()
        # Control: the original verifies.
        assert signer.verify(b"payload", signature) is True

        assert signer.verify(b"payload", make(signature)) is False

    @_not_a_32_byte_key
    def test_a_private_key_that_is_not_32_bytes_raises_value_error(self, key: Any) -> None:
        with pytest.raises(ValueError):
            Ed25519Signer(key, "key-1")

    @pytest.mark.parametrize("key_id", ["", "   ", "\t\n"])
    def test_a_blank_key_id_raises_value_error_for_a_public_key_too(self, key_id: str) -> None:
        with pytest.raises(ValueError):
            Ed25519Signer.from_public_key(_RFC8032_PUBLIC_KEY, key_id)

    def test_a_non_str_key_id_raises_value_error_for_a_public_key_too(self) -> None:
        with pytest.raises(ValueError):
            Ed25519Signer.from_public_key(_RFC8032_PUBLIC_KEY, None)  # type: ignore[arg-type]

    @_not_a_32_byte_key
    def test_a_public_key_that_is_not_32_bytes_raises_value_error(self, key: Any) -> None:
        with pytest.raises(ValueError):
            Ed25519Signer.from_public_key(key, "key-1")

    def test_a_public_key_signer_has_the_algorithm_and_key_id(self) -> None:
        verifier = Ed25519Signer.from_public_key(_RFC8032_PUBLIC_KEY, "key-2026-01")

        assert verifier.algorithm == "Ed25519"
        assert verifier.key_id == "key-2026-01"

    def test_a_public_key_signer_verifies_the_signature_of_the_matching_private_signer(self) -> None:
        signature = Ed25519Signer(_KEY, "key-1").sign(b"payload")
        verifier = _verify_only()

        assert verifier.verify(b"payload", signature) is True
        assert verifier.verify(b"payload-tampered", signature) is False
        assert _verify_only(_OTHER_KEY).verify(b"payload", signature) is False

    def test_a_public_key_signer_cannot_sign(self) -> None:
        with pytest.raises(ValueError, match="public key"):
            _verify_only().sign(b"payload")

    def test_repr_shows_the_key_id_and_no_key_material(self) -> None:
        signer = Ed25519Signer(_KEY, "key-1")

        for text in (repr(signer), str(signer)):
            assert "key-1" in text
            assert _KEY.decode("utf-8") not in text
            assert _KEY.hex() not in text
            assert _public_key(_KEY).hex() not in text

    def test_a_public_key_signer_repr_shows_the_key_id_and_no_key_material(self) -> None:
        verifier = _verify_only()

        for text in (repr(verifier), str(verifier)):
            assert "key-1" in text
            assert _public_key(_KEY).hex() not in text


@_both_algorithms
@_ed25519_only
class TestEd25519PublicKeyOnly:
    """A host that holds only public keys can verify and inspect a log, and can never write to it."""

    def test_a_whole_rotated_log_verifies_with_public_keys_alone_and_returns_the_same_head(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path)
        head = verify_ndjson_log(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])
        current = _verify_only(_THIRD_KEY, "key-3")
        previous = [_verify_only(_KEY, "key-1"), _verify_only(_OTHER_KEY, "key-2")]

        assert verify_ndjson_log(audit_path, manifest_path, signer=current, previous_signers=previous) == head
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=current, previous_signers=previous)
        assert coverage.head == head
        assert coverage.sealed_runs == 3
        assert head == manifest_hash(_read_lines(manifest_path)[-1])

    def test_a_public_key_that_is_not_the_sealing_keys_is_rejected(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        # Control: the sealing key's public key verifies.
        verify_ndjson_log(audit_path, manifest_path, signer=_verify_only())

        with pytest.raises(ManifestVerificationError, match=re.escape(_BAD_SIGNATURE)):
            verify_ndjson_log(audit_path, manifest_path, signer=_verify_only(_OTHER_KEY))

    def test_a_dry_run_quarantine_reports_the_damage_and_changes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path, signer=_verify_only(), dry_run=True)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [4])
        assert _snapshot(tmp_path) == before

    def test_a_repair_cannot_sign_its_trace_so_it_raises_and_leaves_both_logs_alone(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError, match="public key"):
            _quarantine(tmp_path, audit_path, manifest_path, signer=_verify_only())

        assert _snapshot(tmp_path) == before

    def test_a_rotation_entry_repair_cannot_sign_its_trace_so_it_raises_and_leaves_the_log_alone(
        self, tmp_path: Path
    ) -> None:
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError, match="public key"):
            _quarantine_from_entry(tmp_path, manifest_path, signer=_verify_only(), expected_head=anchor)

        assert _snapshot(tmp_path) == before

    def test_sealing_a_pending_run_raises_value_error_and_writes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = manifest_path.read_bytes()

        with pytest.raises(ValueError, match="public key"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_verify_only())

        assert manifest_path.read_bytes() == before

    def test_rotating_to_a_new_public_key_raises_value_error_and_writes_nothing(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = manifest_path.read_bytes()
        head = manifest_hash(_read_lines(manifest_path)[-1])

        with pytest.raises(ValueError, match="public key") as excinfo:
            rotate_manifest_key(
                manifest_path,
                signer=_verify_only(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=head,
            )

        assert not isinstance(excinfo.value, (ManifestVerificationError, KeyAlreadyCurrentError))
        assert manifest_path.read_bytes() == before

    def test_rotating_away_from_a_key_known_only_by_its_public_key_raises_value_error_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = manifest_path.read_bytes()
        head = manifest_hash(_read_lines(manifest_path)[-1])

        with pytest.raises(ValueError, match="public key") as excinfo:
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_verify_only()],
                expected_head=head,
            )

        assert not isinstance(excinfo.value, (ManifestVerificationError, KeyAlreadyCurrentError))
        assert manifest_path.read_bytes() == before

    def test_a_rotated_log_whose_outgoing_keys_are_public_verifies(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path)
        entries = [line for line in _read_lines(manifest_path) if line.get("kind") == "key_rotation"]

        assert [entry["previous_key_signature"]["key_id"] for entry in entries] == ["key-1", "key-2"]
        previous = [_verify_only(_KEY, "key-1"), _verify_only(_OTHER_KEY, "key-2")]
        verify_ndjson_log(
            audit_path, manifest_path, signer=_verify_only(_THIRD_KEY, "key-3"), previous_signers=previous
        )

    def test_a_public_key_as_a_retired_key_still_lets_the_private_signer_seal(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_public, new_signer = _verify_only(), _signer(_OTHER_KEY, "key-2")
        _rotate(manifest_path, new_signer, _signer())
        _write_records(audit_path, [_record("run-d", 7)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_public])

        assert [manifest["run_id"] for manifest in manifests] == ["run-d"]
        head = verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_public])
        assert head == manifest_hash(manifests[-1])


_ED25519_EXTRA = re.escape("mloda-enterprise[ed25519]")


def _audit_package_without_cryptography(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The audit package, cold-imported while `cryptography` cannot be imported."""
    block_root(monkeypatch, "cryptography")
    evict_package(monkeypatch, "mloda.enterprise.extenders.audit")
    return importlib.import_module("mloda.enterprise.extenders.audit")


class TestEd25519WithoutCryptography:
    """cryptography is an optional extra: only building an Ed25519Signer needs it."""

    def test_the_package_imports_and_hmac_signing_still_works(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        package = _audit_package_without_cryptography(monkeypatch)
        signer = package.HmacSha256Signer(_KEY, "key-1")
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])

        manifests = package.seal_ndjson_runs(audit_path, manifest_path, signer=signer)

        assert package.verify_ndjson_log(audit_path, manifest_path, signer=signer) == manifest_hash(manifests[-1])
        assert "Ed25519Signer" in package.__all__

    def test_building_an_ed25519_signer_raises_import_error_naming_the_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        package = _audit_package_without_cryptography(monkeypatch)

        with pytest.raises(ImportError, match=_ED25519_EXTRA):
            package.Ed25519Signer(_KEY, "k")
        with pytest.raises(ImportError, match=_ED25519_EXTRA):
            package.Ed25519Signer.from_public_key(_RFC8032_PUBLIC_KEY, "k")

    def test_the_import_is_not_cached_so_building_works_again_once_cryptography_is_back(self) -> None:
        public_key = _public_key(_KEY)
        # Built once before the block: a cached import would let the blocked builds below through.
        Ed25519Signer(_KEY, "k")
        Ed25519Signer.from_public_key(public_key, "k")

        with pytest.MonkeyPatch.context() as blocked:
            block_root(blocked, "cryptography")
            with pytest.raises(ImportError, match=_ED25519_EXTRA):
                Ed25519Signer(_KEY, "k")
            with pytest.raises(ImportError, match=_ED25519_EXTRA):
                Ed25519Signer.from_public_key(public_key, "k")

        signature = Ed25519Signer(_KEY, "k").sign(b"payload")
        assert Ed25519Signer.from_public_key(public_key, "k").verify(b"payload", signature) is True
