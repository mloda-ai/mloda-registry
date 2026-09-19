"""Tests for run_manifest: HmacSha256Signer, the pure seal_run / manifest_hash / verify_manifest trio and the
NDJSON sealing and verification built on them; file tests write their audit records through NdjsonAuditSink."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
import stat
import sys
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from mloda.steward import verified_context
from mloda.user import ParallelizationMode

import mloda.enterprise.extenders.audit as audit_package
import mloda.enterprise.extenders.audit.run_manifest as run_manifest_module
from mloda.enterprise.extenders.audit import (
    AuditExtender,
    HmacSha256Signer,
    ManifestSigner,
    ManifestVerificationError,
    NdjsonAuditSink,
    RunNotPendingError,
    manifest_hash,
    seal_ndjson_runs,
    seal_run,
    verify_manifest,
    verify_ndjson_log,
)
from mloda.testing.extenders.runners import expected_value_int, run_value_int

_KEY = b"k" * 32
_OTHER_KEY = b"o" * 32

_EXPECTED_MANIFEST_KEYS = {
    "manifest_version",
    "run_id",
    "sealed_at",
    "hash_algorithm",
    "record_count",
    "record_hashes",
    "compliant",
    "previous_manifest_hash",
    "signature",
}


def _signer(key: bytes = _KEY) -> HmacSha256Signer:
    return HmacSha256Signer(key, "key-1")


def _record(run_id: str | None = "run-1", second: int = 0, *, tenant_id: str | None = "tenant-1") -> dict[str, Any]:
    """A plausible slice of an AuditExtender record; `second` keeps the records of one run distinct and a
    missing tenant_id makes it a deny record."""
    compliant = tenant_id is not None
    return {
        "record_version": 1,
        "event_time": f"2026-01-01T00:00:{second:02d}.000000Z",
        "run_id": run_id,
        "tenant_id": tenant_id,
        "decision": "allow" if compliant else "deny",
        "compliant": compliant,
        "deny_reason": None if compliant else "missing_tenant_id",
        "feature_group_class": "my.module.MyFeatureGroup",
        "feature_names": ["value_int"],
        "status": "success",
    }


def _canonical(obj: Mapping[str, Any]) -> bytes:
    """The canonical JSON the spec pins: exactly the line NdjsonAuditSink writes, without the newline."""
    return json.dumps(obj, sort_keys=True).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unsigned(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "signature"}


def _write_records(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    sink = NdjsonAuditSink(path)
    for record in records:
        sink.write(record)


def _read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rewrite_lines(path: Path, objects: Iterable[Mapping[str, Any]]) -> None:
    """Replace the file's content, keeping the one-sorted-JSON-object-per-line shape of NdjsonAuditSink."""
    path.write_text("".join(json.dumps(obj, sort_keys=True) + "\n" for obj in objects), encoding="utf-8")


def _append_line(path: Path, line: str | bytes) -> None:
    """Append raw text or bytes, for a line json.dumps cannot produce or NdjsonAuditSink would never write."""
    data = line.encode("utf-8") if isinstance(line, str) else line
    with open(path, "ab") as file:
        file.write(data + b"\n")


def _resigned(manifest: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    """`manifest` with `changes` applied and a fresh valid signature, so only a check of the content can reject it."""
    changed = {**_unsigned(manifest), **changes}
    changed["signature"] = {**manifest["signature"], "value": _signer().sign(_canonical(changed))}
    return changed


def _sealed_log(directory: Path) -> tuple[Path, Path]:
    """Write three runs (plus one record without a run_id) into an audit file and seal them all."""
    audit_path = directory / "audit.ndjson"
    manifest_path = directory / "manifests.ndjson"
    _write_records(
        audit_path,
        [
            _record("run-a", 1),
            _record("run-a", 2),
            _record(None, 3),
            _record("run-b", 4),
            _record("run-b", 5),
            _record("run-c", 6),
        ],
    )
    manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
    assert [manifest["run_id"] for manifest in manifests] == ["run-a", "run-b", "run-c"]
    return audit_path, manifest_path


def _truncated_log(directory: Path) -> tuple[Path, Path, str]:
    """The tail truncation attack: seal runs a, b and c, note the head, cut the log back to its first line and
    edit a run-b record, which now looks like the record of a run that was never sealed."""
    audit_path, manifest_path = _sealed_log(directory)
    manifests = _read_lines(manifest_path)
    head = manifest_hash(manifests[-1])
    _rewrite_lines(manifest_path, manifests[:1])
    records = _read_lines(audit_path)
    assert records[3]["run_id"] == "run-b"
    records[3]["tenant_id"] = "tenant-other"
    _rewrite_lines(audit_path, records)
    return audit_path, manifest_path, head


class _PrefixSigner:
    """Minimal non-HMAC signer that inherits from nothing: sealing and verifying rely on the protocol only."""

    algorithm = "TEST-SHA256"
    key_id = "test-key"

    def sign(self, payload: bytes) -> str:
        return _sha256(b"prefix:" + payload)

    def verify(self, payload: bytes, signature: str) -> bool:
        # The protocol's contract: a signature that cannot be compared is a wrong one, never an exception.
        try:
            return hmac.compare_digest(self.sign(payload).encode("ascii"), signature.encode("ascii"))
        except (AttributeError, UnicodeEncodeError):
            return False


class TestRunManifestPublicApi:
    """The package root exports every run_manifest name; a missing one already fails this module's import."""

    def test_run_manifest_names_are_in_the_package_all(self) -> None:
        assert {
            "ManifestSigner",
            "HmacSha256Signer",
            "ManifestVerificationError",
            "RunNotPendingError",
            "seal_run",
            "manifest_hash",
            "verify_manifest",
            "seal_ndjson_runs",
            "verify_ndjson_log",
        } <= set(audit_package.__all__)

    def test_run_not_pending_error_is_a_value_error_but_not_a_verification_error(self) -> None:
        # An idempotent `except RunNotPendingError` must not swallow a broken chain.
        assert issubclass(RunNotPendingError, ValueError)
        assert not issubclass(RunNotPendingError, ManifestVerificationError)


class TestHmacSha256Signer:
    """HmacSha256Signer signs with HMAC-SHA256, validates its key material and never shows the key."""

    def test_algorithm_and_key_id(self) -> None:
        signer = HmacSha256Signer(_KEY, "key-2026-01")

        assert signer.algorithm == "HMAC-SHA256"
        assert signer.key_id == "key-2026-01"

    def test_sign_is_the_lowercase_hex_hmac_sha256(self) -> None:
        payload = b'{"a": 1}'

        assert _signer().sign(payload) == hmac.new(_KEY, payload, hashlib.sha256).hexdigest()

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
        # json.loads hands both to a verifier: compare_digest refuses non-ASCII text, utf-8 a lone surrogate.
        assert _signer().verify(b"payload", signature) is False

    @pytest.mark.parametrize("key", [b"", b"k" * 31])
    def test_short_key_raises_value_error(self, key: bytes) -> None:
        with pytest.raises(ValueError):
            HmacSha256Signer(key, "key-1")

    def test_non_bytes_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            HmacSha256Signer("k" * 32, "key-1")  # type: ignore[arg-type]

    @pytest.mark.parametrize("key_id", ["", "   ", "\t\n"])
    def test_blank_key_id_raises_value_error(self, key_id: str) -> None:
        with pytest.raises(ValueError):
            HmacSha256Signer(_KEY, key_id)

    def test_non_str_key_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            HmacSha256Signer(_KEY, None)  # type: ignore[arg-type]

    def test_repr_does_not_contain_the_key(self) -> None:
        key = b"do-not-print-this-signing-key-0123456789"
        signer = HmacSha256Signer(key, "key-1")

        for text in (repr(signer), str(signer)):
            assert key.decode("utf-8") not in text
            assert key.hex() not in text


class TestSealRun:
    """seal_run turns the records of one run into a signed, order-independent manifest."""

    def test_manifest_has_exactly_the_expected_keys(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())

        assert set(manifest) == _EXPECTED_MANIFEST_KEYS

    def test_version_run_id_and_hash_algorithm(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())

        assert manifest["manifest_version"] == 1
        assert manifest["run_id"] == "run-1"
        assert manifest["hash_algorithm"] == "sha256"

    def test_sealed_at_is_rfc3339_utc(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())

        sealed_at = manifest["sealed_at"]
        assert sealed_at.endswith("Z")
        datetime.fromisoformat(sealed_at.removesuffix("Z"))

    def test_record_hash_matches_the_line_ndjson_audit_sink_writes(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.ndjson"
        record = _record(tenant_id="tenant-ü")
        NdjsonAuditSink(path).write(record)

        manifest = seal_run([record], run_id="run-1", signer=_signer())

        assert manifest["record_hashes"] == [_sha256(path.read_bytes().removesuffix(b"\n"))]

    def test_record_hashes_are_sorted(self) -> None:
        records = [_record(second=second) for second in range(5)]

        manifest = seal_run(records, run_id="run-1", signer=_signer())

        assert manifest["record_hashes"] == sorted(_sha256(_canonical(record)) for record in records)

    def test_identical_records_are_each_counted(self) -> None:
        manifest = seal_run([_record(), _record()], run_id="run-1", signer=_signer())

        assert manifest["record_count"] == 2
        assert manifest["record_hashes"] == [_sha256(_canonical(_record()))] * 2

    def test_a_one_shot_iterable_of_records_is_accepted(self) -> None:
        records = [_record(second=1), _record(second=2)]

        manifest = seal_run((record for record in records), run_id="run-1", signer=_signer())

        assert manifest["record_count"] == 2

    def test_records_are_not_mutated(self) -> None:
        records = [_record(second=1), _record(second=2)]
        before = copy.deepcopy(records)

        seal_run(records, run_id="run-1", signer=_signer())

        assert records == before

    def test_all_compliant_records_make_a_compliant_manifest(self) -> None:
        manifest = seal_run([_record(second=1), _record(second=2)], run_id="run-1", signer=_signer())

        assert manifest["compliant"] is True

    def test_one_deny_record_makes_a_non_compliant_manifest(self) -> None:
        records = [_record(second=1), _record(second=2, tenant_id=None), _record(second=3)]

        manifest = seal_run(records, run_id="run-1", signer=_signer())

        assert manifest["compliant"] is False

    def test_record_without_a_compliant_key_makes_a_non_compliant_manifest(self) -> None:
        incomplete = {key: value for key, value in _record(second=2).items() if key != "compliant"}

        manifest = seal_run([_record(second=1), incomplete], run_id="run-1", signer=_signer())

        assert manifest["compliant"] is False

    def test_signature_value_verifies_against_the_manifest_without_its_signature(self) -> None:
        signer = _signer()

        manifest = seal_run([_record()], run_id="run-1", signer=signer, previous_manifest_hash="ab" * 32)

        payload = _canonical(_unsigned(manifest))
        assert signer.verify(payload, manifest["signature"]["value"]) is True
        assert manifest["signature"]["value"] == hmac.new(_KEY, payload, hashlib.sha256).hexdigest()

    def test_any_manifest_signer_can_seal(self) -> None:
        signer: ManifestSigner = _PrefixSigner()

        manifest = seal_run([_record()], run_id="run-1", signer=signer)

        assert set(manifest["signature"]) == {"algorithm", "key_id", "value"}
        assert manifest["signature"]["algorithm"] == "TEST-SHA256"
        assert manifest["signature"]["key_id"] == "test-key"
        assert signer.verify(_canonical(_unsigned(manifest)), manifest["signature"]["value"]) is True

    def test_no_records_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            seal_run([], run_id="run-1", signer=_signer())

    @pytest.mark.parametrize("run_id", ["", "   "])
    def test_blank_run_id_raises_value_error(self, run_id: str) -> None:
        with pytest.raises(ValueError):
            seal_run([_record(run_id)], run_id=run_id, signer=_signer())

    def test_non_str_run_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            seal_run([{**_record(), "run_id": 5}], run_id=5, signer=_signer())  # type: ignore[arg-type]

    @pytest.mark.parametrize("other_run_id", ["run-2", None])
    def test_record_of_a_different_run_raises_value_error(self, other_run_id: str | None) -> None:
        records = [_record("run-1", 1), _record(other_run_id, 2)]

        with pytest.raises(ValueError):
            seal_run(records, run_id="run-1", signer=_signer())


class TestManifestHash:
    """manifest_hash covers the whole manifest, signature included."""

    def test_is_the_sha256_of_the_full_canonical_manifest(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())

        assert manifest_hash(manifest) == hashlib.sha256(_canonical(manifest)).hexdigest()

    def test_does_not_depend_on_key_order_or_a_json_round_trip(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())
        reordered = dict(reversed(list(manifest.items())))

        assert manifest_hash(reordered) == manifest_hash(manifest)
        assert manifest_hash(json.loads(json.dumps(manifest))) == manifest_hash(manifest)

    def test_changing_the_signature_changes_the_hash(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())

        assert manifest_hash({**manifest, "signature": "tampered"}) != manifest_hash(manifest)


class TestVerifyManifest:
    """verify_manifest accepts an untouched manifest with its records and names what broke otherwise."""

    def test_manifest_verification_error_is_a_value_error(self) -> None:
        assert issubclass(ManifestVerificationError, ValueError)

    def test_untouched_manifest_and_records_verify(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        verify_manifest(manifest, records, signer=_signer())

    def test_record_order_does_not_matter(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        verify_manifest(manifest, list(reversed(records)), signer=_signer())

    def test_manifest_and_records_survive_a_json_round_trip(self) -> None:
        records = [_record(second=second, tenant_id="tenant-ü") for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer(), previous_manifest_hash="ab" * 32)

        verify_manifest(json.loads(json.dumps(manifest)), json.loads(json.dumps(records)), signer=_signer())

    def test_any_manifest_signer_can_verify(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_PrefixSigner())

        verify_manifest(manifest, records, signer=_PrefixSigner())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("compliant", True),
            ("hash_algorithm", "md5"),
            ("record_count", 4),
            ("record_hashes", []),
            ("run_id", "run-2"),
            ("sealed_at", "2000-01-01T00:00:00.000000Z"),
            ("previous_manifest_hash", "ab" * 32),
        ],
    )
    def test_changed_manifest_field_fails_the_signature_check(self, field: str, value: Any) -> None:
        records = [_record(second=1), _record(second=2, tenant_id=None), _record(second=3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        assert manifest[field] != value

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest({**manifest, field: value}, records, signer=_signer())

    def test_changed_signature_value_fails_the_signature_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        other_value = _signer().sign(b"something else")

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(
                {**manifest, "signature": {**manifest["signature"], "value": other_value}}, records, signer=_signer()
            )

    def test_non_ascii_signature_value_fails_the_signature_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        # hmac.compare_digest raises TypeError on a non-ASCII str; tampered input must not surface as one.
        tampered = {**manifest, "signature": {**manifest["signature"], "value": "é" * 64}}

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(tampered, records, signer=_signer())

    @pytest.mark.parametrize(("field", "value"), [("key_id", "key-other"), ("algorithm", "HMAC-SHA512")])
    def test_changed_signature_block_field_fails_the_signature_check(self, field: str, value: str) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        assert manifest["signature"][field] != value
        # key_id and algorithm sit outside the signed payload: only a comparison with the signer exposes them.
        tampered = {**manifest, "signature": {**manifest["signature"], field: value}}

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(tampered, records, signer=_signer())

    def test_removed_signature_fails_the_signature_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(_unsigned(manifest), records, signer=_signer())

    def test_extra_key_in_the_signature_block_fails_the_signature_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        # The block sits outside the signed payload, so an unknown member could carry anything unnoticed.
        tampered = {**manifest, "signature": {**manifest["signature"], "note": "never signed"}}

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(tampered, records, signer=_signer())

    def test_altered_record_fails_the_record_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        altered = [records[0], {**records[1], "tenant_id": "tenant-other"}, records[2]]

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_manifest(manifest, altered, signer=_signer())

    def test_removed_record_fails_the_record_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_manifest(manifest, records[:-1], signer=_signer())

    def test_added_record_fails_the_record_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_manifest(manifest, [*records, _record(second=9)], signer=_signer())

    def test_repeated_record_fails_the_record_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_manifest(manifest, [*records, records[0]], signer=_signer())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("manifest_version", 2),
            ("hash_algorithm", "md5"),
            ("record_count", 4),
            ("run_id", 5),
            ("compliant", False),
            ("compliant", 1),
        ],
    )
    def test_correctly_signed_manifest_with_an_unacceptable_field_fails(self, field: str, value: Any) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        # Re-signed: the signature holds and the records still match, so only the check of this field can object.
        resigned = _resigned(manifest, **{field: value})

        with pytest.raises(ManifestVerificationError, match=field):
            verify_manifest(resigned, records, signer=_signer())

    def test_correctly_signed_compliant_manifest_over_a_deny_record_fails(self) -> None:
        records = [_record(second=1), _record(second=2, tenant_id=None), _record(second=3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        assert manifest["compliant"] is False

        with pytest.raises(ManifestVerificationError, match="compliant"):
            verify_manifest(_resigned(manifest, compliant=True), records, signer=_signer())

    @pytest.mark.parametrize("field", ["record_count", "manifest_version"])
    @pytest.mark.parametrize("value", [True, 1.0], ids=["bool", "float"])
    def test_correctly_signed_manifest_with_a_non_int_number_fails(self, field: str, value: Any) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        # Equal to the expected 1, so only a type check can object.
        assert manifest[field] == value

        with pytest.raises(ManifestVerificationError, match=field):
            verify_manifest(_resigned(manifest, **{field: value}), records, signer=_signer())


class TestSealNdjsonRuns:
    """seal_ndjson_runs seals every unsealed run of an audit file into a hash-chained manifest log."""

    def test_seals_every_run_of_the_audit_file(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2), _record("run-a", 3)])

        manifests = seal_ndjson_runs(audit_path, tmp_path / "manifests.ndjson", signer=_signer())

        assert [(manifest["run_id"], manifest["record_count"]) for manifest in manifests] == [
            ("run-a", 2),
            ("run-b", 1),
        ]

    def test_runs_are_sealed_in_order_of_first_appearance_in_the_audit_file(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        # run-b appears first, although run-a sorts first by name and by every event_time.
        _write_records(audit_path, [_record("run-b", 5), _record("run-a", 1), _record("run-b", 6)])

        manifests = seal_ndjson_runs(audit_path, tmp_path / "manifests.ndjson", signer=_signer())

        assert [manifest["run_id"] for manifest in manifests] == ["run-b", "run-a"]

    @pytest.mark.parametrize("event_time", [{}, {"event_time": 5}], ids=["missing", "not-a-string"])
    def test_record_without_a_usable_event_time_does_not_block_sealing(
        self, tmp_path: Path, event_time: dict[str, Any]
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        record = {key: value for key, value in _record("run-a", 1).items() if key != "event_time"}
        _write_records(audit_path, [{**record, **event_time}, _record("run-b", 2)])

        manifests = seal_ndjson_runs(audit_path, tmp_path / "manifests.ndjson", signer=_signer())

        assert [manifest["run_id"] for manifest in manifests] == ["run-a", "run-b"]

    @pytest.mark.parametrize(
        ("run_id", "drop_key"),
        [(None, False), ("", False), ("   ", False), (None, True)],
        ids=["null", "empty", "blank", "absent"],
    )
    def test_records_without_a_run_id_are_ignored(self, tmp_path: Path, run_id: str | None, drop_key: bool) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"

        def unusable(second: int) -> dict[str, Any]:
            record = _record(run_id, second)
            if drop_key:
                del record["run_id"]
            return record

        _write_records(audit_path, [unusable(1), _record("run-a", 2), unusable(3)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert [(manifest["run_id"], manifest["record_count"]) for manifest in manifests] == [("run-a", 1)]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_run_with_a_deny_record_is_sealed_as_non_compliant(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2), _record("run-b", 3, tenant_id=None)])

        manifests = seal_ndjson_runs(audit_path, tmp_path / "manifests.ndjson", signer=_signer())

        assert [(manifest["run_id"], manifest["compliant"]) for manifest in manifests] == [
            ("run-a", True),
            ("run-b", False),
        ]

    def test_appends_one_json_line_per_manifest_and_accepts_str_paths(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])

        manifests = seal_ndjson_runs(str(audit_path), str(manifest_path), signer=_signer())

        assert len(manifests) == 2
        assert _read_lines(manifest_path) == manifests
        verify_ndjson_log(str(audit_path), str(manifest_path), signer=_signer())

    def test_new_manifest_file_has_owner_only_permissions(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])

        seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600

    def test_first_manifest_has_no_previous_hash_and_each_later_one_chains(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2), _record("run-c", 3)])

        manifests = seal_ndjson_runs(audit_path, tmp_path / "manifests.ndjson", signer=_signer())

        assert len(manifests) == 3
        assert manifests[0]["previous_manifest_hash"] is None
        assert manifests[1]["previous_manifest_hash"] == manifest_hash(manifests[0])
        assert manifests[2]["previous_manifest_hash"] == manifest_hash(manifests[1])

    def test_chain_continues_across_calls_and_sealed_runs_are_skipped(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])
        first = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        _write_records(audit_path, [_record("run-b", 2)])

        second = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert [manifest["run_id"] for manifest in second] == ["run-b"]
        assert second[0]["previous_manifest_hash"] == manifest_hash(first[0])
        assert second[0]["previous_manifest_hash"] == manifest_hash(_read_lines(manifest_path)[0])
        assert _read_lines(manifest_path) == [*first, *second]

    def test_nothing_new_returns_empty_and_leaves_the_manifest_file_unchanged(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        before = manifest_path.read_bytes()

        assert seal_ndjson_runs(audit_path, manifest_path, signer=_signer()) == []
        assert manifest_path.read_bytes() == before

    def test_explicit_run_id_seals_only_that_run(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2), _record("run-c", 3)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-b")

        assert [manifest["run_id"] for manifest in manifests] == ["run-b"]
        assert manifests[0]["previous_manifest_hash"] is None
        assert _read_lines(manifest_path) == manifests

    def test_explicit_run_id_without_records_raises_run_not_pending_error(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])

        with pytest.raises(RunNotPendingError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-missing")

    def test_explicit_run_id_already_sealed_raises_run_not_pending_error(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-a")
        before = manifest_path.read_bytes()

        with pytest.raises(RunNotPendingError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-a")

        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("forged_run_id", ["run-d", "run-never-written"])
    def test_forged_unsigned_manifest_line_stops_sealing(self, tmp_path: Path, forged_run_id: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        # Unverified, the first line would pass run-d off as sealed and the second would be chained onto.
        _write_records(manifest_path, [{"run_id": forged_run_id}])
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert manifest_path.read_bytes() == before

    def test_edited_audit_line_of_a_sealed_run_does_not_stop_sealing(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        records = _read_lines(audit_path)
        assert records[0]["run_id"] == "run-a"
        records[0]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, [*records, _record("run-d", 7)])

        # The sealer checks the manifest log, not the records of runs it signs nothing about: sealing on top
        # of the edit makes nothing worse, and verification still reports it.
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert [(manifest["run_id"], manifest["previous_manifest_hash"]) for manifest in manifests] == [("run-d", head)]
        with pytest.raises(ManifestVerificationError, match="record"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_signing_failure_writes_nothing_and_a_retry_seals_every_run(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _write_records(audit_path, [_record("run-d", 7), _record("run-e", 8), _record("run-f", 9)])
        before = manifest_path.read_bytes()

        class _FailingSigner(HmacSha256Signer):
            def sign(self, payload: bytes) -> str:
                # Fails on the last new manifest; verifying the sealed log signs too but never meets run-f.
                if json.loads(payload).get("run_id") == "run-f":
                    raise RuntimeError("signing boom")
                return super().sign(payload)

        with pytest.raises(RuntimeError, match="signing boom"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_FailingSigner(_KEY, "key-1"), expected_head=head)

        assert manifest_path.read_bytes() == before
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head)
        assert [manifest["run_id"] for manifest in manifests] == ["run-d", "run-e", "run-f"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("file_name", ["manifests.ndjson", "audit.ndjson"])
    def test_unterminated_final_line_fails_and_leaves_both_files_untouched(
        self, tmp_path: Path, file_name: str
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        # Pending run-d: sealing it would concatenate its JSON onto an unterminated manifest line.
        _write_records(audit_path, [_record("run-d", 7)])
        path = tmp_path / file_name
        path.write_bytes(path.read_bytes().removesuffix(b"\n"))
        audit_before, manifest_before = audit_path.read_bytes(), manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match="newline"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        with pytest.raises(ManifestVerificationError, match="newline"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert audit_path.read_bytes() == audit_before
        assert manifest_path.read_bytes() == manifest_before

    def test_sealing_holds_an_exclusive_lock_on_the_manifest_log(self, tmp_path: Path) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        refused: list[bool] = []

        class _LockProbeSigner(HmacSha256Signer):
            def sign(self, payload: bytes) -> str:
                # Verifying the sealed log signs too. A second descriptor is a second lock owner, so it is refused.
                fd = os.open(manifest_path, os.O_RDONLY)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    refused.append(False)
                except BlockingIOError:
                    refused.append(True)
                finally:
                    os.close(fd)
                return super().sign(payload)

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_LockProbeSigner(_KEY, "key-1"))

        assert [manifest["run_id"] for manifest in manifests] == ["run-d"]
        assert refused
        assert all(refused)
        fd = os.open(manifest_path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    def test_sealing_and_verifying_work_where_fcntl_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "fcntl", None)
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert [manifest["run_id"] for manifest in manifests] == ["run-a", "run-b"]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == manifest_hash(manifests[-1])

    def test_nothing_pending_on_a_fresh_manifest_path_leaves_an_empty_log(self, tmp_path: Path) -> None:
        pytest.importorskip("fcntl")  # The lock creates the log; where fcntl is missing there is none.
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record(None, 1), _record(None, 2)])

        assert seal_ndjson_runs(audit_path, manifest_path, signer=_signer()) == []

        assert manifest_path.read_bytes() == b""
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) is None


class TestVerifyNdjsonLog:
    """verify_ndjson_log checks every sealed run, the manifest chain and the uniqueness of each run_id."""

    def test_untouched_log_verifies(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_returns_the_manifest_hash_of_the_last_manifest(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        head = verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert head == manifest_hash(_read_lines(manifest_path)[-1])

    def test_records_of_a_run_that_is_not_sealed_yet_are_fine(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7), _record("run-d", 8)])

        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_log_sealed_out_of_audit_file_order_verifies(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2), _record("run-c", 3)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-b")

        rest = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert [manifest["run_id"] for manifest in rest] == ["run-a", "run-c"]
        assert [manifest["run_id"] for manifest in _read_lines(manifest_path)] == ["run-b", "run-a", "run-c"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_signer_with_a_different_key_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(_OTHER_KEY))

    def test_edited_audit_line_of_a_sealed_run_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[0]["run_id"] == "run-a"
        records[0]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, records)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_deleted_audit_line_of_a_sealed_run_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[0]["run_id"] == "run-a"
        _rewrite_lines(audit_path, records[1:])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_sealed_run_without_any_audit_line_left_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert [record["run_id"] for record in records].count("run-c") == 1
        _rewrite_lines(audit_path, [record for record in records if record["run_id"] != "run-c"])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_record_appended_to_a_sealed_run_fails_and_the_run_is_never_resealed(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        # By design: a second session.run() on one prepared session writes the sealed run_id again.
        _write_records(audit_path, [_record("run-a", 9)])
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        # The sealer neither absorbs the late record nor refuses to go on: a benign rerun must not brick the log.
        assert seal_ndjson_runs(audit_path, manifest_path, signer=_signer()) == []
        assert manifest_path.read_bytes() == before

    def test_edited_last_manifest_line_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        manifests = _read_lines(manifest_path)
        # Only the signature can expose an edit of the last line; no later manifest chains to it yet.
        manifests[-1]["compliant"] = not manifests[-1]["compliant"]
        _rewrite_lines(manifest_path, manifests)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_manifest_line_with_an_unhashable_run_id_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        manifests = _read_lines(manifest_path)
        assert manifests[0]["run_id"] == "run-a"
        # A JSON list parses to an unhashable value; looking the run up must not surface as a TypeError.
        # Re-signed, so the run_id guard is reached and not the signature check.
        manifests[0] = _resigned(manifests[0], run_id=["run-a"])
        _rewrite_lines(manifest_path, manifests)

        with pytest.raises(ManifestVerificationError, match="run_id"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("index", [0, 1])
    def test_deleted_manifest_line_breaks_the_chain(self, tmp_path: Path, index: int) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        manifests = _read_lines(manifest_path)
        del manifests[index]
        _rewrite_lines(manifest_path, manifests)

        with pytest.raises(ManifestVerificationError, match="chain"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_manifest_removed_together_with_its_records_breaks_the_chain(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        manifests = _read_lines(manifest_path)
        assert manifests[1]["run_id"] == "run-b"
        # By design: with the manifest and the records both gone, only the chain still remembers run-b.
        _rewrite_lines(manifest_path, [manifests[0], manifests[2]])
        _rewrite_lines(audit_path, [record for record in _read_lines(audit_path) if record["run_id"] != "run-b"])

        with pytest.raises(ManifestVerificationError, match="chain"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize(("left", "right"), [(0, 1), (1, 2)])
    def test_swapped_manifest_lines_break_the_chain(self, tmp_path: Path, left: int, right: int) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        manifests = _read_lines(manifest_path)
        manifests[left], manifests[right] = manifests[right], manifests[left]
        _rewrite_lines(manifest_path, manifests)

        with pytest.raises(ManifestVerificationError, match="chain"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_same_run_id_in_two_manifests_fails(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        records = [_record("run-a", 1), _record("run-a", 2)]
        _write_records(audit_path, records)
        # Both manifests are correctly signed, chained and match the records; only the repeated run_id is wrong.
        first = seal_run(records, run_id="run-a", signer=_signer())
        second = seal_run(records, run_id="run-a", signer=_signer(), previous_manifest_hash=manifest_hash(first))
        _write_records(manifest_path, [first, second])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize(
        "dump_options",
        [{"sort_keys": True, "separators": (",", ":")}, {"sort_keys": False}],
        ids=["compact", "reversed-keys"],
    )
    def test_reformatted_audit_line_of_a_sealed_run_fails(self, tmp_path: Path, dump_options: dict[str, Any]) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        lines = audit_path.read_text(encoding="utf-8").splitlines()
        original = json.loads(lines[0])
        assert original["run_id"] == "run-a"
        # The same JSON value in other bytes: the seal covers the bytes the sink wrote.
        reformatted = json.dumps(dict(reversed(original.items())), **dump_options)
        assert reformatted != lines[0]
        assert json.loads(reformatted) == original
        lines[0] = reformatted
        audit_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("log_bytes", [None, b""], ids=["missing", "empty"])
    def test_missing_or_empty_manifest_log_returns_none(self, tmp_path: Path, log_bytes: bytes | None) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])
        if log_bytes is not None:
            manifest_path.write_bytes(log_bytes)

        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) is None
        assert manifest_path.exists() == (log_bytes is not None)

    def test_missing_manifest_log_fails_against_an_expected_head(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])

        with pytest.raises(ManifestVerificationError, match="head"):
            verify_ndjson_log(audit_path, tmp_path / "manifests.ndjson", signer=_signer(), expected_head="ab" * 32)

    def test_every_failing_run_is_reported_in_one_error(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[5]["run_id"] == "run-c"
        records[5]["tenant_id"] = "tenant-other"
        # A late record on run-a and an edit on run-c: the first failure must not hide the second.
        _rewrite_lines(audit_path, [*records, _record("run-a", 9)])

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        message = str(excinfo.value)
        assert "run-a" in message
        assert "run-c" in message
        assert "run-b" not in message

    def test_a_head_mismatch_is_reported_together_with_a_record_mismatch(self, tmp_path: Path) -> None:
        audit_path, manifest_path, head = _truncated_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[0]["run_id"] == "run-a"
        records[0]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, records)

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), expected_head=head)

        message = str(excinfo.value)
        assert "head" in message
        assert "record" in message

    def test_manifest_log_is_read_before_the_audit_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        original = run_manifest_module._read_ndjson
        reads: list[str] = []

        def spy(path: str | Path) -> Any:
            reads.append(Path(path).name)
            return original(path)

        monkeypatch.setattr(run_manifest_module, "_read_ndjson", spy)

        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        # A run sealed between the two reads must not look like tampering: the log has to be the older snapshot.
        assert reads == ["manifests.ndjson", "audit.ndjson"]

    def test_edited_audit_line_without_a_run_id_still_verifies(self, tmp_path: Path) -> None:
        # Documents the limit: a record without a run_id is outside every seal.
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[2]["run_id"] is None
        records[2]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, records)

        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_verification_waits_for_a_sealer_holding_the_lock(self, tmp_path: Path) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])

        with ThreadPoolExecutor(max_workers=1) as pool:
            fd = os.open(manifest_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                future = pool.submit(verify_ndjson_log, audit_path, manifest_path, signer=_signer())
                done, _ = wait([future], timeout=0.3)
            finally:
                os.close(fd)

            assert not done
            assert future.result(timeout=10) == head


class TestExpectedHead:
    """expected_head anchors the log outside itself, the only defence against cutting manifests off its end."""

    def test_matching_expected_head_verifies_and_lets_sealing_continue(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _write_records(audit_path, [_record("run-d", 7)])

        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer(), expected_head=head) == head
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head)

        assert [(manifest["run_id"], manifest["previous_manifest_hash"]) for manifest in manifests] == [("run-d", head)]

    def test_truncated_log_fails_verification_against_the_expected_head(self, tmp_path: Path) -> None:
        audit_path, manifest_path, head = _truncated_log(tmp_path)

        with pytest.raises(ManifestVerificationError, match="head"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), expected_head=head)

    def test_truncated_log_stops_sealing_against_the_expected_head(self, tmp_path: Path) -> None:
        audit_path, manifest_path, head = _truncated_log(tmp_path)
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match="head"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head)

        assert manifest_path.read_bytes() == before

    def test_truncated_log_verifies_without_an_expected_head(self, tmp_path: Path) -> None:
        # Documents the limit: nothing inside the log can tell that its newest manifests were cut off.
        audit_path, manifest_path, _ = _truncated_log(tmp_path)

        verify_ndjson_log(audit_path, manifest_path, signer=_signer())


class TestMalformedNdjsonLines:
    """A line that is ambiguous, or is not the object it should be, is a verification error in both readers."""

    @pytest.mark.parametrize(("file_name", "key"), [("audit.ndjson", "tenant_id"), ("manifests.ndjson", "run_id")])
    def test_line_with_a_duplicate_json_key_fails_verification(self, tmp_path: Path, file_name: str, key: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        path = tmp_path / file_name
        lines = path.read_text(encoding="utf-8").splitlines()
        # Raw text, json.dumps cannot write it: Python keeps the last duplicate, a first-wins reader sees EVIL.
        lines[0] = f'{{"{key}": "EVIL", {lines[0][1:]}'
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        assert json.loads(lines[0])[key] != "EVIL"

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize(
        ("file_name", "line"),
        [("audit.ndjson", "[]"), ("audit.ndjson", "null"), ("manifests.ndjson", '"x"'), ("manifests.ndjson", "5")],
    )
    def test_json_line_that_is_not_an_object_fails(self, tmp_path: Path, file_name: str, line: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _append_line(tmp_path / file_name, line)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        with pytest.raises(ManifestVerificationError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("run_id", [["x"], 5], ids=["list", "number"])
    def test_audit_record_whose_run_id_is_neither_a_string_nor_null_fails(self, tmp_path: Path, run_id: Any) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [{**_record(), "run_id": run_id}])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        with pytest.raises(ManifestVerificationError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize(("file_name", "line_number"), [("audit.ndjson", 7), ("manifests.ndjson", 4)])
    @pytest.mark.parametrize(
        "line",
        [b"", b'{"run_id": "run-d"', b"\xff\xfe", b"[" * 100000],
        ids=["blank", "truncated", "bad-utf8", "deep-nesting"],
    )
    def test_undecodable_line_fails_naming_the_file_and_line(
        self, tmp_path: Path, file_name: str, line_number: int, line: bytes
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        path = tmp_path / file_name
        _append_line(path, line)
        # The first line after what _sealed_log wrote.
        location = re.escape(f"{path} line {line_number}")

        with pytest.raises(ManifestVerificationError, match=location):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        with pytest.raises(ManifestVerificationError, match=location):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())


class TestRunManifestRunAll:
    """run_all round trips: the audit file a real run leaves behind seals into one verifiable manifest."""

    @pytest.mark.parametrize(
        "mode",
        [ParallelizationMode.SYNC, ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING],
    )
    def test_run_all_audit_file_seals_and_verifies(
        self, mode: ParallelizationMode, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        # Only MULTIPROCESSING needs (and pays for) the repo-wide flight_server fixture, as in the shared contract.
        flight_server = (
            request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
        )
        extender = AuditExtender(sink=NdjsonAuditSink(audit_path))

        with verified_context(tenant_id="tenant-42", project_id="project-7", principal="svc"):
            values = run_value_int(extender, parallelization_modes={mode}, flight_server=flight_server)

        assert values == expected_value_int()
        records = _read_lines(audit_path)
        assert records
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert len(manifests) == 1
        manifest = manifests[0]
        assert manifest["compliant"] is True
        assert manifest["record_count"] == len(records)
        assert {record["run_id"] for record in records} == {manifest["run_id"]}

    def test_run_all_without_verified_context_seals_a_non_compliant_manifest(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"

        values = run_value_int(AuditExtender(sink=NdjsonAuditSink(audit_path)))

        assert values == expected_value_int()
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert len(manifests) == 1
        assert manifests[0]["compliant"] is False
