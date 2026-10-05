"""Tests for sealing a run, verifying manifests and the manifest log chain."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import re
import stat
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import mloda.enterprise.extenders.audit.run_manifest as run_manifest_module
from mloda.enterprise.extenders.audit import (
    Ed25519Signer,
    HmacSha256Signer,
    KeyAlreadyCurrentError,
    ManifestSigner,
    ManifestVerificationError,
    NdjsonAuditSink,
    NdjsonHeadAnchor,
    RunAlreadySealedError,
    _core,
    manifest_hash,
    rotate_manifest_key,
    seal_ndjson_runs,
    seal_run,
    verify_manifest,
    verify_ndjson_log,
    verify_ndjson_log_coverage,
)
from mloda.enterprise.extenders.audit.tests.manifest_helpers import (
    _BAD_SIGNATURE,
    _DAMAGED_LINES,
    _EXPECTED_GENESIS_KEYS,
    _FORGED_ENTRIES,
    _KEY,
    _MISSING,
    _OTHER_KEY,
    _PREDECESSOR,
    _THIRD_KEY,
    _append_line,
    _as_v1,
    _assert_names,
    _assert_raises_and_unchanged,
    _assert_refused,
    _both_algorithms,
    _both_recoveries,
    _build_log,
    _canonical,
    _cap,
    _genesis_entry,
    _genesis_log,
    _insert_line,
    _log_heads,
    _ndjson_anchor,
    _oversized_line,
    _patch_bindings,
    _PrefixSigner,
    _public_key,
    _quarantine,
    _read_lines,
    _record,
    _RecordingAnchor,
    _Recovery,
    _reference_signature,
    _resigned,
    _rewrite_lines,
    _rotate,
    _rotated_three_key_log,
    _rotation_entry,
    _sealed_log,
    _sha256,
    _signer,
    _signing_payload_ref,
    _snapshot,
    _spans,
    _successor_log,
    _summary,
    _torn,
    _trace,
    _trace_chain,
    _truncated_log,
    _under_key,
    _unsigned,
    _verify_only,
    _verify_quarantine,
    _write_records,
)
from mloda.enterprise.extenders.audit.tests.test_audit_extender import (
    _ReadSpy,
)

_EXPECTED_MANIFEST_KEYS = {
    "manifest_version",
    "run_id",
    "sealed_at",
    "hash_algorithm",
    "record_count",
    "record_hashes",
    "compliant",
    "sealed_late",
    "previous_manifest_hash",
    "signature",
}


_current_key_required = pytest.mark.parametrize(
    "call", [verify_ndjson_log, verify_ndjson_log_coverage, seal_ndjson_runs], ids=["verify", "coverage", "seal"]
)


_BAD_SHAPE = "must be a 'key_rotation' entry with exactly"


# The `match` fragment of the error each forged entry must raise, so none passes for the wrong reason.
_FORGED_ENTRY_REASONS: dict[str, str] = {
    "wrong-key-material": _BAD_SIGNATURE,
    "unknown-key-id": "signature key_id 'key-unknown' matches no known key",
    "wrong-chain": "manifest chain is broken at a key rotation entry",
    "tampered-rotated-at": _BAD_SIGNATURE,
    "wrong-algorithm": "signature algorithm 'OTHER' is not the signer's",
    "wrong-version": "unsupported manifest_version 3",
    "extra-key": _BAD_SHAPE,
    "missing-key": _BAD_SHAPE,
    "unknown-kind": _BAD_SHAPE,
    "null-kind": _BAD_SHAPE,
    "unhashable-kind": _BAD_SHAPE,
    "manifest-with-kind": _BAD_SHAPE,
    "missing-co-signature": _BAD_SHAPE,
    "co-signed-by-a-non-current-key": "previous_key_signature key_id 'key-2' is not the current key 'key-1'",
    "forged-co-signature-value": "previous_key_signature does not match the manifest",
}


def _mixed_algorithm_log(directory: Path) -> tuple[Path, Path, ManifestSigner, ManifestSigner, ManifestSigner]:
    """One run each sealed by HMAC key-1, Ed25519 key-2 and HMAC key-3 in turn, with two rotations."""
    audit_path = directory / "audit.ndjson"
    manifest_path = directory / "manifests.ndjson"
    key_1 = HmacSha256Signer(_KEY, "key-1")
    key_2 = Ed25519Signer(_OTHER_KEY, "key-2")
    key_3 = HmacSha256Signer(_THIRD_KEY, "key-3")
    _write_records(audit_path, [_record("run-a", 1)])
    seal_ndjson_runs(audit_path, manifest_path, signer=key_1)
    _rotate(manifest_path, key_2, key_1)
    _write_records(audit_path, [_record("run-b", 2)])
    seal_ndjson_runs(audit_path, manifest_path, signer=key_2, previous_signers=[key_1])
    _rotate(manifest_path, key_3, key_1, key_2)
    _write_records(audit_path, [_record("run-c", 3)])
    seal_ndjson_runs(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])
    return audit_path, manifest_path, key_1, key_2, key_3


@_both_algorithms
class TestSealRun:
    def test_manifest_has_exactly_the_expected_keys(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())

        assert set(manifest) == _EXPECTED_MANIFEST_KEYS

    def test_version_run_id_and_hash_algorithm(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())

        assert manifest["manifest_version"] == 2
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

        payload = _signing_payload_ref(manifest)
        assert signer.verify(payload, manifest["signature"]["value"]) is True
        assert manifest["signature"]["value"] == _reference_signature(_KEY, payload)

    def test_any_manifest_signer_can_seal(self) -> None:
        signer: ManifestSigner = _PrefixSigner()

        manifest = seal_run([_record()], run_id="run-1", signer=signer)

        assert set(manifest["signature"]) == {"algorithm", "key_id", "value"}
        assert manifest["signature"]["algorithm"] == "TEST-SHA256"
        assert manifest["signature"]["key_id"] == "test-key"
        assert signer.verify(_signing_payload_ref(manifest), manifest["signature"]["value"]) is True

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


@_both_algorithms
class TestVerifyManifest:
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
        tampered = {**manifest, "signature": {**manifest["signature"], "value": "é" * 64}}

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(tampered, records, signer=_signer())

    @pytest.mark.parametrize(("field", "value"), [("key_id", "key-other"), ("algorithm", "HMAC-SHA512")])
    def test_changed_signature_block_field_fails_the_signature_check(self, field: str, value: str) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        assert manifest["signature"][field] != value
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
        tampered = {**manifest, "signature": {**manifest["signature"], "note": "never signed"}}

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(tampered, records, signer=_signer())

    def test_unhashable_signature_key_id_fails_without_crashing(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        tampered = {**manifest, "signature": {**manifest["signature"], "key_id": []}}

        with pytest.raises(ManifestVerificationError):
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

    def test_added_record_beyond_the_seal_names_the_run_and_the_count(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_manifest(manifest, [*records, _record(second=9)], signer=_signer())

        message = str(excinfo.value)
        assert "run-1" in message
        assert "has 1 record(s) beyond its seal" in message
        assert "do not match the record_hashes" not in message

    def test_added_record_together_with_an_edited_record_fails_the_generic_check(self) -> None:
        """Count grows by one, but an edit replaces a sealed hash: not a pure late append."""
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        mixed = [{**records[0], "tenant_id": "tenant-other"}, records[1], records[2], _record(second=9)]

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_manifest(manifest, mixed, signer=_signer())

        message = str(excinfo.value)
        assert "do not match the record_hashes" in message
        assert "beyond its seal" not in message

    def test_repeated_record_fails_the_record_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_manifest(manifest, [*records, records[0]], signer=_signer())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("manifest_version", 3),
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
        resigned = _resigned(manifest, **{field: value})

        with pytest.raises(ManifestVerificationError, match=field):
            verify_manifest(resigned, records, signer=_signer())

    @pytest.mark.parametrize("record_hashes", [[[]], [1]], ids=["unhashable", "hashable-non-string"])
    def test_correctly_signed_manifest_with_a_non_string_record_hash_fails_without_crashing(
        self, record_hashes: list[Any]
    ) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        resigned = _resigned(manifest, record_hashes=record_hashes, record_count=len(record_hashes))

        with pytest.raises(ManifestVerificationError):
            verify_manifest(resigned, records, signer=_signer())

    def test_correctly_signed_compliant_manifest_over_a_deny_record_fails(self) -> None:
        records = [_record(second=1), _record(second=2, tenant_id=None), _record(second=3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        assert manifest["compliant"] is False

        with pytest.raises(ManifestVerificationError, match="compliant"):
            verify_manifest(_resigned(manifest, compliant=True), records, signer=_signer())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("record_count", True),
            ("record_count", 1.0),
            ("manifest_version", True),
            ("manifest_version", 2.0),
        ],
    )
    def test_correctly_signed_manifest_with_a_non_int_number_fails(self, field: str, value: Any) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        # Where the value equals the genuine one (1 == True == 1.0, 2 == 2.0) only its type is wrong. The
        # manifest_version True case (True != 2) only guards that a bool is not accepted as a version.
        if (field, value) != ("manifest_version", True):
            assert manifest[field] == value

        with pytest.raises(ManifestVerificationError, match=field):
            verify_manifest(_resigned(manifest, **{field: value}), records, signer=_signer())

    def test_previous_signers_verifies_a_manifest_signed_by_an_old_key(self) -> None:
        records = [_record(second=second) for second in range(3)]
        old_signer = _signer()
        manifest = seal_run(records, run_id="run-1", signer=old_signer)
        new_signer = _signer(_OTHER_KEY, "key-2")

        verify_manifest(manifest, records, signer=new_signer, previous_signers=[old_signer])

    def test_previous_signers_accepts_any_manifest_signer_protocol_implementation(self) -> None:
        records = [_record()]
        old_signer = _PrefixSigner()
        manifest = seal_run(records, run_id="run-1", signer=old_signer)
        new_signer = _signer(_OTHER_KEY, "key-2")

        verify_manifest(manifest, records, signer=new_signer, previous_signers=[old_signer])

    def test_unknown_key_id_fails_the_signature_check_and_names_it(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer(b"u" * 32, "key-unknown"))

        with pytest.raises(ManifestVerificationError, match="signature") as excinfo:
            verify_manifest(manifest, records, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()])

        assert "key-unknown" in str(excinfo.value)

    def test_unknown_key_id_with_previous_signers_names_no_known_key(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer(b"u" * 32, "key-unknown"))

        with pytest.raises(ManifestVerificationError, match="signature") as excinfo:
            verify_manifest(manifest, records, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()])

        message = str(excinfo.value)
        assert "key-unknown" in message
        assert "is not the signer's" not in message

    def test_unknown_key_id_without_previous_signers_keeps_the_exact_wording(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer(b"u" * 32, "key-unknown"))

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_manifest(manifest, records, signer=_signer())

        assert str(excinfo.value) == "signature key_id 'key-unknown' is not the signer's 'key-1'"

    @pytest.mark.parametrize(
        "previous_signers",
        [[_signer()], [_signer(key_id="key-3"), _signer(_OTHER_KEY, "key-3")]],
        ids=["signer-and-previous", "two-previous"],
    )
    def test_previous_signers_with_a_duplicate_key_id_is_a_value_error(
        self, previous_signers: list[ManifestSigner]
    ) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ValueError) as excinfo:
            verify_manifest(manifest, records, signer=_signer(_OTHER_KEY), previous_signers=previous_signers)

        assert not isinstance(excinfo.value, ManifestVerificationError)

    def test_a_rotation_entry_is_not_a_run_manifest_and_fails_without_crashing(self) -> None:
        with pytest.raises(ManifestVerificationError, match="unsupported hash_algorithm"):
            verify_manifest(_rotation_entry(_signer(), None), [_record()], signer=_signer())


@_both_algorithms
class TestKeyRotationEntries:
    """Key order comes from the log's rotation entries, not from previous_signers."""

    @pytest.mark.parametrize("trailing_seal", [False, True], ids=["ends-with-entry", "seal-after-entry"])
    def test_a_hand_built_rotation_entry_verifies_and_the_next_line_chains_to_it(
        self, tmp_path: Path, trailing_seal: bool
    ) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        steps = [("seal", old_signer), ("rotate", new_signer), *([("seal", new_signer)] if trailing_seal else [])]
        audit_path, manifest_path = _build_log(tmp_path, *steps)

        head = verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert head == manifest_hash(_read_lines(manifest_path)[-1])

    def test_a_new_key_manifest_without_a_rotation_entry_is_rejected(self, tmp_path: Path) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        audit_path, manifest_path = _build_log(tmp_path, ("seal", old_signer), ("seal", new_signer))

        with pytest.raises(ManifestVerificationError, match="run-2") as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        _assert_names(excinfo, "key-2", "key-1")

    def test_a_retired_key_manifest_after_a_rotation_is_rejected_before_the_new_key_ever_sealed(
        self, tmp_path: Path
    ) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        steps = [("seal", old_signer), ("rotate", new_signer), ("seal", old_signer)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)

        with pytest.raises(ManifestVerificationError, match="run-3") as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        _assert_names(excinfo, "key-1", "key-2")

    @pytest.mark.parametrize("seal_between", [True, False], ids=["seal-between", "back-to-back"])
    @pytest.mark.parametrize("newest_first", [False, True], ids=["oldest-first", "newest-first"])
    def test_the_order_of_previous_signers_does_not_matter(
        self, tmp_path: Path, seal_between: bool, newest_first: bool
    ) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path, seal_between=seal_between)
        previous_signers = [key_2, key_1] if newest_first else [key_1, key_2]

        head = verify_ndjson_log(audit_path, manifest_path, signer=key_3, previous_signers=previous_signers)

        assert head == manifest_hash(_read_lines(manifest_path)[-1])

    @pytest.mark.parametrize("signed_by", ["active", "retired"])
    def test_a_rotation_entry_signed_by_the_active_or_a_retired_key_is_rejected(
        self, tmp_path: Path, signed_by: str
    ) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        signer = new_signer if signed_by == "active" else old_signer
        steps = [("seal", old_signer), ("rotate", new_signer), ("rotate", signer)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)

        with pytest.raises(ManifestVerificationError, match=signer.key_id):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    def test_a_rotation_entry_as_the_first_line_is_rejected(self, tmp_path: Path) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        audit_path, manifest_path = _build_log(tmp_path, ("seal", old_signer), ("rotate", new_signer))
        # The control: after a seal, a rotation entry is fine.
        verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        _rewrite_lines(manifest_path, [_rotation_entry(new_signer, None)])

        with pytest.raises(ManifestVerificationError, match="has no earlier manifest to rotate from"):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    @pytest.mark.parametrize(
        ("forge", "reason"),
        [(forge, _FORGED_ENTRY_REASONS[name]) for name, forge in _FORGED_ENTRIES.items()],
        ids=list(_FORGED_ENTRIES),
    )
    def test_a_forged_rotation_entry_is_rejected_and_never_skipped(
        self, tmp_path: Path, forge: Callable[[str], dict[str, Any]], reason: str
    ) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        audit_path, manifest_path = _build_log(tmp_path, ("seal", old_signer), ("rotate", new_signer))
        # The control: the unforged entry verifies.
        verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        seal, entry = _read_lines(manifest_path)
        _rewrite_lines(manifest_path, [seal, forge(entry["previous_manifest_hash"])])
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match=reason):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        with pytest.raises(ManifestVerificationError, match=reason):
            seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert manifest_path.read_bytes() == before

    def test_a_v2_rotation_entry_carries_the_co_signature_of_the_key_it_retires(self, tmp_path: Path) -> None:
        old_signer, new_signer, third = _signer(), _signer(_OTHER_KEY, "key-2"), _signer(_THIRD_KEY, "key-3")
        steps = [("seal", old_signer), ("rotate", new_signer), ("rotate", third)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)

        verify_ndjson_log(audit_path, manifest_path, signer=third, previous_signers=[old_signer, new_signer])

        _, first, second = _read_lines(manifest_path)
        assert [first["previous_key_signature"]["key_id"], second["previous_key_signature"]["key_id"]] == [
            "key-1",
            "key-2",
        ]

    def test_a_v1_rotation_without_a_co_signature_verifies_in_a_v1_prefix_before_a_v2_rotation(
        self, tmp_path: Path
    ) -> None:
        old_signer, new_signer, third = _signer(), _signer(_OTHER_KEY, "key-2"), _signer(_THIRD_KEY, "key-3")
        steps = [("seal-v1", old_signer), ("rotate-v1", new_signer), ("rotate", third)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)
        assert "previous_key_signature" not in _read_lines(manifest_path)[1]

        head = verify_ndjson_log(audit_path, manifest_path, signer=third, previous_signers=[old_signer, new_signer])

        assert head == manifest_hash(_read_lines(manifest_path)[-1])

    def test_sealing_skips_rotation_entries_like_sealed_runs(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path)
        before = manifest_path.read_bytes()

        assert seal_ndjson_runs(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2]) == []
        with pytest.raises(RunAlreadySealedError):
            seal_ndjson_runs(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2], run_id="run-a")

        assert manifest_path.read_bytes() == before


@_both_algorithms
class TestCurrentKeyRequirement:
    """Sealing and verifying need the log's current key to be the signer."""

    @_current_key_required
    def test_a_log_under_the_old_key_is_rejected_for_the_new_signer_even_with_the_old_key_as_previous_signer(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError, match=_under_key("key-1", "key-2")):
            call(audit_path, manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()])

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("call", [verify_ndjson_log, verify_ndjson_log_coverage], ids=["verify", "coverage"])
    def test_the_check_is_raised_before_the_collected_record_and_head_problems(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        records[0]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, records)

        with pytest.raises(ManifestVerificationError, match=_under_key("key-1", "key-2")):
            call(
                audit_path,
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head="0" * 64,
            )

    @_current_key_required
    def test_an_entry_to_a_keyring_key_the_log_never_used_does_not_take_the_log_over(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        old_signer, new_signer, extra_signer = _signer(), _signer(_OTHER_KEY, "key-2"), _signer(b"x" * 32, "key-x")
        steps = [("seal", old_signer), ("rotate", new_signer), ("seal", new_signer), ("rotate", extra_signer)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError, match=_under_key("key-x", "key-2")):
            call(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer, extra_signer])

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("log_bytes", [None, b""], ids=["missing", "empty"])
    def test_an_empty_log_has_no_current_key_and_the_first_seal_defines_it(
        self, tmp_path: Path, log_bytes: bytes | None
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])
        if log_bytes is not None:
            manifest_path.write_bytes(log_bytes)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")

        assert verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer]) is None
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert manifests[0]["signature"]["key_id"] == "key-2"
        verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])


@_both_algorithms
class TestExpectedHead:
    """expected_head anchors the log outside itself."""

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
        audit_path, manifest_path, _ = _truncated_log(tmp_path)

        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_after_a_rotation_the_entry_hash_anchors_and_an_older_head_does_not(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        old_head = manifest_hash(_read_lines(manifest_path)[-1])
        entry_head = manifest_hash(_rotate(manifest_path, new_signer, old_signer))
        _write_records(audit_path, [_record("run-d", 7)])

        assert (
            verify_ndjson_log(
                audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=entry_head
            )
            == entry_head
        )
        with pytest.raises(ManifestVerificationError, match="head"):
            verify_ndjson_log(
                audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=old_head
            )
        manifests = seal_ndjson_runs(
            audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=entry_head
        )
        assert [manifest["previous_manifest_hash"] for manifest in manifests] == [entry_head]


_ANCHORED_CALLS: dict[str, Callable[..., Any]] = {
    "verify": lambda a, m, **kw: verify_ndjson_log(a, m, signer=_signer(), **kw),
    "coverage": lambda a, m, **kw: verify_ndjson_log_coverage(a, m, signer=_signer(), **kw),
    "seal": lambda a, m, **kw: seal_ndjson_runs(a, m, signer=_signer(), **kw),
}


_each_anchored_call = pytest.mark.parametrize("call", list(_ANCHORED_CALLS.values()), ids=list(_ANCHORED_CALLS))


@_both_algorithms
class TestAnchoredHeads:
    """anchored_heads: every anchor must be the hash of a verified line, so a rolled back or replaced log is caught."""

    @_each_anchored_call
    def test_every_line_hash_of_the_log_passes_as_an_anchor(self, tmp_path: Path, call: Callable[..., Any]) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        heads = _log_heads(manifest_path)
        assert len(heads) == 4

        call(audit_path, manifest_path, log_id="log-a", anchored_heads=heads)

    @_each_anchored_call
    def test_the_predecessor_head_of_a_successor_genesis_passes_as_an_anchor(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        audit_path, manifest_path = _successor_log(tmp_path)

        call(audit_path, manifest_path, log_id="log-a", anchored_heads=[_PREDECESSOR, *_log_heads(manifest_path)])

    @_each_anchored_call
    def test_an_anchored_heads_iterator_is_accepted(self, tmp_path: Path, call: Callable[..., Any]) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        call(audit_path, manifest_path, anchored_heads=iter(_log_heads(manifest_path)))

    @_each_anchored_call
    def test_no_anchored_heads_keeps_todays_behaviour(self, tmp_path: Path, call: Callable[..., Any]) -> None:
        audit_path, manifest_path, _ = _truncated_log(tmp_path)

        call(audit_path, manifest_path, anchored_heads=[])

    @_each_anchored_call
    def test_an_unknown_anchor_fails_and_is_named(self, tmp_path: Path, call: Callable[..., Any]) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        unknown = "0" * 64

        with pytest.raises(ManifestVerificationError) as excinfo:
            call(audit_path, manifest_path, anchored_heads=[unknown])

        _assert_names(excinfo, unknown)

    @_each_anchored_call
    def test_one_unknown_anchor_among_valid_ones_fails(self, tmp_path: Path, call: Callable[..., Any]) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        unknown = "1" * 64

        with pytest.raises(ManifestVerificationError) as excinfo:
            call(audit_path, manifest_path, anchored_heads=[*_log_heads(manifest_path), unknown])

        _assert_names(excinfo, unknown)

    def test_a_rotation_entry_hash_passes_as_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        old_head = manifest_hash(_read_lines(manifest_path)[-1])
        entry_head = manifest_hash(_rotate(manifest_path, new_signer, old_signer))

        head = verify_ndjson_log(
            audit_path,
            manifest_path,
            signer=new_signer,
            previous_signers=[old_signer],
            anchored_heads=[old_head, entry_head],
        )

        assert head == entry_head

    def test_an_anchor_that_is_the_hash_of_a_forged_unsigned_manifest_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        forged = {**_read_lines(manifest_path)[-1]}
        del forged["signature"]
        forged["run_id"] = "run-never-sealed"
        anchor = manifest_hash(forged)

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), anchored_heads=[anchor])

        _assert_names(excinfo, anchor)

    def test_expected_head_keeps_equality_semantics_next_to_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        heads = _log_heads(manifest_path)

        # Control: an older head is a valid anchor but not the expected head.
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), anchored_heads=[heads[0]])
        with pytest.raises(ManifestVerificationError, match="head"):
            verify_ndjson_log(
                audit_path, manifest_path, signer=_signer(), expected_head=heads[0], anchored_heads=[heads[0]]
            )

    def test_the_rollback_that_equality_alone_cannot_catch_is_caught_by_an_older_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_head = _log_heads(manifest_path)[-1]
        _rewrite_lines(manifest_path, _read_lines(manifest_path)[:1])
        # Growth after the rollback: the cut runs are sealed again, so the log is as long as before.
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), sealed_late=False)
        grown = verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        assert grown != old_head
        assert len(_read_lines(manifest_path)) == 3
        # Control: today's checks accept the regrown log, and it is not the old head either way.
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), expected_head=grown)

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), anchored_heads=[old_head])

        _assert_names(excinfo, old_head)

    @pytest.mark.parametrize("which", ["verify", "coverage"])
    def test_dropping_the_newest_run_with_its_records_is_detected(self, tmp_path: Path, which: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = _log_heads(manifest_path)[-1]
        _rewrite_lines(manifest_path, _read_lines(manifest_path)[:-1])
        _rewrite_lines(audit_path, [r for r in _read_lines(audit_path) if r["run_id"] != "run-c"])
        call = _ANCHORED_CALLS[which]
        # Control: the shortened log is a consistent log without the anchor.
        call(audit_path, manifest_path)

        with pytest.raises(ManifestVerificationError) as excinfo:
            call(audit_path, manifest_path, anchored_heads=[head])

        _assert_names(excinfo, head)

    @pytest.mark.parametrize("which", ["verify", "coverage"])
    def test_deleting_both_files_is_detected_by_an_anchor(self, tmp_path: Path, which: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = _log_heads(manifest_path)[-1]
        audit_path.unlink()
        manifest_path.unlink()

        with pytest.raises(ManifestVerificationError) as excinfo:
            _ANCHORED_CALLS[which](audit_path, manifest_path, anchored_heads=[head])

        _assert_names(excinfo, head)

    def test_an_empty_manifest_log_fails_with_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = _log_heads(manifest_path)[-1]
        manifest_path.write_bytes(b"")

        for call in (
            lambda: verify_ndjson_log(audit_path, manifest_path, signer=_signer(), anchored_heads=[head]),
            lambda: seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), anchored_heads=[head]),
            lambda: rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), expected_head=None, anchored_heads=[head]
            ),
        ):
            with pytest.raises(ManifestVerificationError) as excinfo:
                call()
            _assert_names(excinfo, head)
        assert manifest_path.read_bytes() == b""

    def test_a_missing_manifest_log_fails_sealing_with_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = _log_heads(manifest_path)[-1]
        manifest_path.unlink()

        with pytest.raises(ManifestVerificationError) as excinfo:
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), anchored_heads=[head])

        _assert_names(excinfo, head)
        assert not manifest_path.exists() or not manifest_path.read_bytes()

    def test_substituting_another_log_signed_with_the_same_key_is_detected(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        _, manifest_path = _sealed_log(real)
        head = _log_heads(manifest_path)[-1]
        other = tmp_path / "other"
        other.mkdir()
        other_audit, other_manifest = _build_log(other, ("seal", _signer()), ("seal", _signer()), ("seal", _signer()))
        # Control: the substitute is a valid log under the same key.
        verify_ndjson_log(other_audit, other_manifest, signer=_signer())

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(other_audit, other_manifest, signer=_signer(), anchored_heads=[head])

        _assert_names(excinfo, head)

    def test_rolling_back_then_sealing_again_refuses_and_writes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path, old_head = _truncated_log(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError) as excinfo:
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), anchored_heads=[old_head])

        _assert_names(excinfo, old_head)
        assert _snapshot(tmp_path) == before

    def test_rotation_refuses_a_rolled_back_log_with_an_anchor_and_writes_nothing(self, tmp_path: Path) -> None:
        _, manifest_path, old_head = _truncated_log(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError) as excinfo:
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=None,
                anchored_heads=[old_head],
            )

        _assert_names(excinfo, old_head)
        assert _snapshot(tmp_path) == before


@_both_algorithms
class TestHeadAnchor:
    """The HeadAnchor protocol, NdjsonHeadAnchor and the head_anchor hook of sealing, rotation and quarantine."""

    def test_write_appends_a_head_and_anchored_at_line_and_latest_returns_the_last(self, tmp_path: Path) -> None:
        anchor = _ndjson_anchor(tmp_path / "anchor.ndjson")

        anchor.write("a" * 64)
        anchor.write("b" * 64)

        lines = _read_lines(tmp_path / "anchor.ndjson")
        assert [line["head"] for line in lines] == ["a" * 64, "b" * 64]
        for line in lines:
            assert set(line) == {"head", "anchored_at"}
            assert line["anchored_at"].endswith("Z")
            datetime.fromisoformat(line["anchored_at"].removesuffix("Z"))
        assert anchor.latest() == "b" * 64

    def test_write_appends_to_an_existing_anchor_file_and_keeps_it_owner_only(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.ndjson"
        _rewrite_lines(path, [{"head": "a" * 64, "anchored_at": "2026-01-01T00:00:00.000000Z"}])

        _ndjson_anchor(path).write("b" * 64)

        assert [line["head"] for line in _read_lines(path)] == ["a" * 64, "b" * 64]
        fresh = tmp_path / "fresh.ndjson"
        _ndjson_anchor(fresh).write("c" * 64)
        assert stat.S_IMODE(fresh.stat().st_mode) == 0o600

    def test_latest_is_none_for_a_missing_or_empty_file(self, tmp_path: Path) -> None:
        assert _ndjson_anchor(tmp_path / "missing.ndjson").latest() is None
        (tmp_path / "empty.ndjson").write_bytes(b"")
        assert _ndjson_anchor(tmp_path / "empty.ndjson").latest() is None

    def test_latest_ignores_a_torn_last_line_and_returns_the_last_terminated_head(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.ndjson"
        _rewrite_lines(path, [{"head": "a" * 64, "anchored_at": "2026-01-01T00:00:00.000000Z"}])
        _torn(path, b'{"head": "bb')

        assert _ndjson_anchor(path).latest() == "a" * 64

    def test_latest_is_none_when_the_only_line_is_torn(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.ndjson"
        path.write_bytes(b'{"head": "bb')

        assert _ndjson_anchor(path).latest() is None

    @pytest.mark.parametrize("earlier", [1, 5000], ids=["one-line", "many-lines"])
    def test_latest_ignores_an_oversized_last_line_like_a_torn_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, earlier: int
    ) -> None:
        path = tmp_path / "anchor.ndjson"
        if earlier == 1:
            _rewrite_lines(path, [{"head": "a" * 64, "anchored_at": "2026-01-01T00:00:00.000000Z"}])
            previous = "a" * 64
        else:
            self._many_heads(path, earlier)
            previous = f"{earlier - 1:064x}"
        _cap(monkeypatch)
        _append_line(path, _oversized_line({"head": "b" * 64}))

        head, seen = self._latest_with_read_spy(path, monkeypatch)

        assert head == previous
        if earlier > 1:
            assert sum(seen) <= 64 * 1024

    def test_latest_is_none_when_the_only_line_is_oversized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "anchor.ndjson"
        _cap(monkeypatch)
        _append_line(path, _oversized_line({"head": "b" * 64}))

        assert _ndjson_anchor(path).latest() is None

    @staticmethod
    def _latest_with_read_spy(path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str | None, list[int]]:
        seen: list[int] = []
        real_open = open

        def spy_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            handle = real_open(file, mode, *args, **kwargs)
            return _ReadSpy(handle, seen) if mode == "rb" else handle

        _patch_bindings(monkeypatch, "open", spy_open)
        return _ndjson_anchor(path).latest(), seen

    @staticmethod
    def _many_heads(path: Path, count: int) -> None:
        _rewrite_lines(
            path,
            [{"head": f"{number:064x}", "anchored_at": "2026-01-01T00:00:00.000000Z"} for number in range(count)],
        )

    def test_latest_reads_a_bounded_tail_not_the_whole_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "anchor.ndjson"
        self._many_heads(path, 5000)
        assert path.stat().st_size > 400_000

        head, seen = self._latest_with_read_spy(path, monkeypatch)

        assert head == f"{4999:064x}"
        assert seen
        assert sum(seen) <= 64 * 1024

    def test_latest_read_volume_does_not_grow_with_the_number_of_earlier_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        small, large = tmp_path / "small.ndjson", tmp_path / "large.ndjson"
        self._many_heads(small, 50)
        self._many_heads(large, 5000)

        _, seen_small = self._latest_with_read_spy(small, monkeypatch)
        _, seen_large = self._latest_with_read_spy(large, monkeypatch)

        assert sum(seen_large) <= max(sum(seen_small), 1) + 32 * 1024

    def test_latest_ignores_a_torn_tail_after_many_lines_without_reading_them_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "anchor.ndjson"
        self._many_heads(path, 5000)
        _torn(path, b'{"head": "bb')

        head, seen = self._latest_with_read_spy(path, monkeypatch)

        assert head == f"{4999:064x}"
        assert sum(seen) <= 64 * 1024

    def test_latest_returns_a_last_line_longer_than_any_read_chunk(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.ndjson"
        self._many_heads(path, 20)
        long_head = "h" * 100_000
        _append_line(path, json.dumps({"head": long_head, "anchored_at": "2026-01-01T00:00:00.000000Z"}).encode())

        assert _ndjson_anchor(path).latest() == long_head

    def test_latest_returns_the_previous_long_head_when_a_long_tail_is_torn(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.ndjson"
        self._many_heads(path, 20)
        long_head = "h" * 100_000
        _append_line(path, json.dumps({"head": long_head, "anchored_at": "2026-01-01T00:00:00.000000Z"}).encode())
        _torn(path, b'{"head": "' + b"t" * 50_000)

        assert _ndjson_anchor(path).latest() == long_head

    @pytest.mark.parametrize("line", [b'{"head": 5}', b'{"anchored_at": "x"}'], ids=["non-string", "missing"])
    def test_latest_fails_closed_for_a_terminated_line_without_a_string_head(self, tmp_path: Path, line: bytes) -> None:
        path = tmp_path / "anchor.ndjson"
        _rewrite_lines(path, [{"head": "a" * 64, "anchored_at": "2026-01-01T00:00:00.000000Z"}])
        _append_line(path, line)

        with pytest.raises(ManifestVerificationError, match=re.escape(str(path))):
            _ndjson_anchor(path).latest()

    def test_write_fsyncs_the_anchor_file_and_its_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "anchor.ndjson"
        real_fsync = os.fsync
        fsynced: list[str] = []

        def spy_fsync(fd: int) -> None:
            if os.path.samestat(os.fstat(fd), path.stat()):
                fsynced.append("file")
            if os.path.samestat(os.fstat(fd), tmp_path.stat()):
                fsynced.append("dir")
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", spy_fsync)

        _ndjson_anchor(path).write("a" * 64)

        assert fsynced == ["file", "dir"]

    @pytest.mark.parametrize(
        "last",
        [raw for name, raw in _DAMAGED_LINES.items() if name != "blank"],
        ids=[n for n in _DAMAGED_LINES if n != "blank"],
    )
    def test_latest_fails_closed_for_an_undecodable_last_line(self, tmp_path: Path, last: bytes) -> None:
        path = tmp_path / "anchor.ndjson"
        _rewrite_lines(path, [{"head": "a" * 64, "anchored_at": "2026-01-01T00:00:00.000000Z"}])
        _append_line(path, last.rstrip(b"\n"))

        with pytest.raises(ManifestVerificationError, match=re.escape(str(path))):
            _ndjson_anchor(path).latest()

    def test_seal_writes_the_new_head_after_the_durable_append(self, tmp_path: Path) -> None:
        audit_path, manifest_path = tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])
        anchor = _RecordingAnchor(manifest_path)

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), head_anchor=anchor)

        assert anchor.heads == [manifest_hash(manifests[-1])] == anchor.last_lines_seen

    def test_seal_with_a_genesis_writes_the_hash_of_the_last_appended_line(self, tmp_path: Path) -> None:
        audit_path, manifest_path = tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])
        anchor = _RecordingAnchor(manifest_path)

        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a", head_anchor=anchor)

        lines = _read_lines(manifest_path)
        assert [line.get("kind") for line in lines] == ["genesis", None]
        assert anchor.heads == [manifest_hash(lines[-1])]

    def test_seal_with_nothing_to_append_does_not_write_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        anchor = _RecordingAnchor()

        assert seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), head_anchor=anchor) == []

        assert anchor.heads == []

    def test_a_refused_seal_does_not_write_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path, head = _truncated_log(tmp_path)
        anchor = _RecordingAnchor()

        with pytest.raises(ManifestVerificationError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head, head_anchor=anchor)

        assert anchor.heads == []

    def test_a_failing_anchor_write_leaves_the_seal_on_disk_and_propagates(self, tmp_path: Path) -> None:
        audit_path, manifest_path = tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])

        with pytest.raises(RuntimeError, match="anchor down"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), head_anchor=_RecordingAnchor(fail=True))

        assert [line["run_id"] for line in _read_lines(manifest_path)] == ["run-a", "run-b"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_rotation_writes_the_entry_hash_after_the_durable_append(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        head = _log_heads(manifest_path)[-1]
        anchor = _RecordingAnchor(manifest_path)

        entry = rotate_manifest_key(
            manifest_path,
            signer=_signer(_OTHER_KEY, "key-2"),
            previous_signers=[_signer()],
            expected_head=head,
            head_anchor=anchor,
        )

        assert anchor.heads == [manifest_hash(entry)] == anchor.last_lines_seen

    def test_a_failing_anchor_write_leaves_the_rotation_on_disk_and_propagates(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        head = _log_heads(manifest_path)[-1]

        with pytest.raises(RuntimeError, match="anchor down"):
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=head,
                head_anchor=_RecordingAnchor(fail=True),
            )

        assert _read_lines(manifest_path)[-1]["kind"] == "key_rotation"

    def test_a_rotation_that_fails_does_not_write_an_anchor(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        anchor = _RecordingAnchor()

        with pytest.raises(KeyAlreadyCurrentError):
            rotate_manifest_key(manifest_path, signer=_signer(), expected_head=None, head_anchor=anchor)

        assert anchor.heads == []

    def test_an_ndjson_anchor_catches_a_later_rollback(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        anchor = _ndjson_anchor(tmp_path / "anchor.ndjson")
        _write_records(audit_path, [_record("run-d", 7)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), head_anchor=anchor)
        latest = anchor.latest()
        assert latest == _log_heads(manifest_path)[-1]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer(), anchored_heads=[latest]) == latest

        _rewrite_lines(manifest_path, _read_lines(manifest_path)[:-1])

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), anchored_heads=[anchor.latest()])
        _assert_names(excinfo, str(latest))

    @_both_recoveries
    def test_a_repair_writes_the_hash_of_the_last_trace_line_before_touching_the_logs(
        self, tmp_path: Path, recovery: _Recovery
    ) -> None:
        manifest_path, repair, _ = recovery(tmp_path)
        before = _snapshot(tmp_path)
        anchor = _RecordingAnchor(_trace(tmp_path))
        repaired_when_written: list[bool] = []
        write = anchor.write

        def write_and_look(head: str) -> None:
            on_disk = {name: data for name, data in _snapshot(tmp_path).items() if name != _trace(tmp_path).name}
            repaired_when_written.append(on_disk != before)
            write(head)

        anchor.write = write_and_look  # type: ignore[method-assign]

        repair(tmp_path, head_anchor=anchor)

        assert anchor.heads == [_sha256(_canonical(_read_lines(_trace(tmp_path))[-1]))] == anchor.last_lines_seen
        assert repaired_when_written == [False]
        assert manifest_path.read_bytes() != before[manifest_path.name]

    @_both_recoveries
    def test_a_dry_run_writes_no_anchor(self, tmp_path: Path, recovery: _Recovery) -> None:
        _, repair, _ = recovery(tmp_path)
        anchor = _RecordingAnchor()

        repair(tmp_path, dry_run=True, head_anchor=anchor)

        assert anchor.heads == []

    def test_nothing_to_repair_writes_no_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        anchor = _RecordingAnchor()

        assert _quarantine(tmp_path, audit_path, manifest_path, head_anchor=anchor) == []

        assert anchor.heads == []

    @_both_recoveries
    def test_a_refused_repair_writes_no_anchor(self, tmp_path: Path, recovery: _Recovery) -> None:
        _, repair, _ = recovery(tmp_path)
        _rewrite_lines(_trace(tmp_path), _trace_chain(_signer(_OTHER_KEY, "key-2")))
        anchor = _RecordingAnchor()

        _assert_raises_and_unchanged(tmp_path, lambda: repair(tmp_path, head_anchor=anchor))

        assert anchor.heads == []

    def test_a_repair_that_raises_after_the_trace_append_has_anchored_the_trace(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")
        _insert_line(audit_path, 5, b"\n")
        anchor = _RecordingAnchor()

        with patch("os.replace", side_effect=OSError(errno.EIO, "replace failed")):
            with pytest.raises(OSError, match="replace failed"):
                _quarantine(tmp_path, audit_path, manifest_path, head_anchor=anchor)

        assert _trace(tmp_path).exists()
        assert anchor.heads == [_sha256(_canonical(_read_lines(_trace(tmp_path))[-1]))]

    @_both_recoveries
    def test_a_failing_anchor_write_keeps_the_trace_leaves_the_logs_untouched_and_propagates(
        self, tmp_path: Path, recovery: _Recovery
    ) -> None:
        manifest_path, repair, dropped = recovery(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(RuntimeError, match="anchor down"):
            repair(tmp_path, head_anchor=_RecordingAnchor(fail=True))

        assert _verify_quarantine(_trace(tmp_path), _signer()) is not None
        assert len(_read_lines(_trace(tmp_path))) == dropped
        assert manifest_path.read_bytes() == before[manifest_path.name]
        assert {n: d for n, d in _snapshot(tmp_path).items() if n != _trace(tmp_path).name} == {
            n: d for n, d in before.items() if n != _trace(tmp_path).name
        }

    @_both_recoveries
    def test_an_ndjson_anchor_catches_a_later_trace_rollback(self, tmp_path: Path, recovery: _Recovery) -> None:
        _, repair, _ = recovery(tmp_path)
        anchor = _ndjson_anchor(tmp_path / "anchor.ndjson")

        repair(tmp_path, head_anchor=anchor)

        latest = anchor.latest()
        assert latest is not None
        assert _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=[latest]) == latest
        _rewrite_lines(_trace(tmp_path), _read_lines(_trace(tmp_path))[:-1])
        with pytest.raises(ManifestVerificationError) as excinfo:
            _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=[latest])
        _assert_names(excinfo, latest)


@_both_algorithms
class TestGenesis:
    """A log opened with a log_id starts with a signed genesis line that names it."""

    def test_the_first_seal_writes_a_genesis_with_exactly_the_expected_keys_and_chains_onto_it(
        self, tmp_path: Path
    ) -> None:
        _, manifest_path = _genesis_log(tmp_path)

        genesis, *seals = _read_lines(manifest_path)

        assert set(genesis) == _EXPECTED_GENESIS_KEYS
        assert (genesis["manifest_version"], genesis["kind"], genesis["log_id"]) == (2, "genesis", "log-a")
        assert genesis["previous_manifest_hash"] is None
        assert genesis["created_at"].endswith("Z")
        datetime.fromisoformat(genesis["created_at"].removesuffix("Z"))
        signature = genesis["signature"]
        assert set(signature) == {"algorithm", "key_id", "value"}
        assert (signature["algorithm"], signature["key_id"]) == (_signer().algorithm, "key-1")
        assert signature["value"] == _reference_signature(_KEY, _signing_payload_ref(genesis))
        assert [seal["run_id"] for seal in seals] == ["run-a", "run-b", "run-c"]
        assert seals[0]["previous_manifest_hash"] == manifest_hash(genesis)

    def test_nothing_is_written_when_there_is_nothing_to_seal(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        audit_path.write_bytes(b"")

        assert seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a") == []

        # The seal lock creates an empty log (see the fresh-manifest-path test); only a genesis line must be absent.
        assert not manifest_path.exists() or manifest_path.read_bytes() == b""

    def test_a_later_seal_batch_adds_no_second_genesis(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])

        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a")

        assert [line.get("kind") for line in _read_lines(manifest_path)] == ["genesis", None, None, None, None]

    def test_a_genesis_log_verifies_with_and_without_the_log_id(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])

        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a") == head
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer(), log_id="log-a")
        assert (coverage.head, coverage.sealed_runs) == (head, 3)

    def test_the_genesis_hash_is_a_valid_expected_head_for_a_genesis_only_log(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        audit_path.write_bytes(b"")
        genesis = _genesis_entry(_signer())
        _write_records(manifest_path, [genesis])
        head = manifest_hash(genesis)

        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer(), expected_head=head) == head
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a") == head

    def test_a_genesis_only_log_takes_the_first_seal_chained_onto_it(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        genesis = _genesis_entry(_signer())
        _write_records(manifest_path, [genesis])
        _write_records(audit_path, [_record("run-a", 1)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a")

        assert [manifest["previous_manifest_hash"] for manifest in manifests] == [manifest_hash(genesis)]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a")

    def test_the_genesis_sets_the_active_key(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = manifest_path.read_bytes()
        other = _signer(_OTHER_KEY, "key-2")

        with pytest.raises(ManifestVerificationError, match=_under_key("key-1", "key-2")):
            seal_ndjson_runs(audit_path, manifest_path, signer=other, previous_signers=[_signer()], log_id="log-a")

        assert manifest_path.read_bytes() == before

    def test_a_genesis_only_log_can_be_rotated_and_sealed_under_the_new_key(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        genesis = _genesis_entry(old_signer)
        _write_records(manifest_path, [genesis])

        entry = rotate_manifest_key(
            manifest_path,
            signer=new_signer,
            previous_signers=[old_signer],
            expected_head=manifest_hash(genesis),
            log_id="log-a",
        )

        assert entry["previous_manifest_hash"] == manifest_hash(genesis)
        assert entry["previous_key_signature"]["key_id"] == "key-1"
        _write_records(audit_path, [_record("run-a", 1)])
        manifests = seal_ndjson_runs(
            audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], log_id="log-a"
        )
        head = verify_ndjson_log(
            audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], log_id="log-a"
        )
        assert head == manifest_hash(manifests[-1])

    def test_a_genesis_headed_log_rotates_after_sealing_and_still_verifies_by_log_id(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")

        rotate_manifest_key(
            manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=None, log_id="log-a"
        )

        verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], log_id="log-a")

    @pytest.mark.parametrize("call", [verify_ndjson_log, verify_ndjson_log_coverage, seal_ndjson_runs])
    def test_another_log_id_under_the_same_key_is_rejected(self, tmp_path: Path, call: Callable[..., Any]) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path, log_id="log-a")
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError) as excinfo:
            call(audit_path, manifest_path, signer=_signer(), log_id="log-b")

        _assert_names(excinfo, "log-a", "log-b")
        assert _snapshot(tmp_path) == before

    def test_rotating_another_log_id_under_the_same_key_is_rejected(self, tmp_path: Path) -> None:
        _, manifest_path = _genesis_log(tmp_path, log_id="log-a")
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError) as excinfo:
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=None,
                log_id="log-b",
            )

        _assert_names(excinfo, "log-a", "log-b")
        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("call", [verify_ndjson_log, verify_ndjson_log_coverage, seal_ndjson_runs])
    def test_a_log_without_a_genesis_is_rejected_when_a_log_id_is_given(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError, match="genesis"):
            call(audit_path, manifest_path, signer=_signer(), log_id="log-a")

        assert _snapshot(tmp_path) == before

    def test_rotating_a_log_without_a_genesis_is_rejected_when_a_log_id_is_given(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match="genesis"):
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=None,
                log_id="log-a",
            )

        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("log_id", [None, "log-a"], ids=["no-log-id", "log-id"])
    @pytest.mark.parametrize("call", [verify_ndjson_log, verify_ndjson_log_coverage, seal_ndjson_runs])
    @pytest.mark.parametrize("position", ["after-a-seal", "second-genesis"])
    def test_a_genesis_anywhere_but_line_1_is_rejected(
        self, tmp_path: Path, call: Callable[..., Any], position: str, log_id: str | None
    ) -> None:
        if position == "second-genesis":
            audit_path, manifest_path = _genesis_log(tmp_path)
        else:
            audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _append_line(manifest_path, _canonical(_genesis_entry(_signer(), previous_manifest_hash=head)))
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError, match="genesis"):
            call(audit_path, manifest_path, signer=_signer(), **({} if log_id is None else {"log_id": log_id}))

        assert _snapshot(tmp_path) == before

    def test_rotating_a_log_with_a_misplaced_genesis_is_rejected(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _append_line(manifest_path, _canonical(_genesis_entry(_signer(), previous_manifest_hash=head)))
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match="genesis"):
            rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()], expected_head=None
            )

        assert manifest_path.read_bytes() == before

    def test_sealing_with_a_log_id_onto_a_non_empty_log_without_that_genesis_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match="genesis"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a")

        assert manifest_path.read_bytes() == before

    def test_sealing_without_a_log_id_onto_a_genesis_log_keeps_the_genesis_and_chains(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _write_records(audit_path, [_record("run-d", 7)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert [manifest["previous_manifest_hash"] for manifest in manifests] == [head]
        assert _read_lines(manifest_path)[0]["kind"] == "genesis"

    @pytest.mark.parametrize("log_id", ["", "   ", 5, b"log-a"], ids=["empty", "blank", "int", "bytes"])
    @pytest.mark.parametrize(
        "call", [verify_ndjson_log, verify_ndjson_log_coverage, seal_ndjson_runs], ids=["verify", "coverage", "seal"]
    )
    def test_a_blank_or_non_str_log_id_is_a_value_error_and_writes_nothing(
        self, tmp_path: Path, call: Callable[..., Any], log_id: Any
    ) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError, match="log_id") as excinfo:
            call(audit_path, manifest_path, signer=_signer(), log_id=log_id)

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("log_id", ["", "   ", 5], ids=["empty", "blank", "int"])
    def test_a_blank_or_non_str_log_id_is_a_value_error_for_a_rotation(self, tmp_path: Path, log_id: Any) -> None:
        _, manifest_path = _genesis_log(tmp_path)
        before = manifest_path.read_bytes()

        with pytest.raises(ValueError, match="log_id") as excinfo:
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=None,
                log_id=log_id,
            )

        assert not isinstance(excinfo.value, (ManifestVerificationError, KeyAlreadyCurrentError))
        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("log_id", [None, "log-b"], ids=["no-log-id", "new-id"])
    def test_a_genesis_with_a_changed_log_id_and_the_old_signature_fails(
        self, tmp_path: Path, log_id: str | None
    ) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        lines = _read_lines(manifest_path)
        lines[0]["log_id"] = "log-b"
        _rewrite_lines(manifest_path, lines)

        with pytest.raises(ManifestVerificationError, match=re.escape(_BAD_SIGNATURE)):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id=log_id)

    @pytest.mark.parametrize(
        "forged",
        [
            lambda: _genesis_entry(_signer(b"x" * 32, "key-1")),
            lambda: _genesis_entry(_signer(), created_at=_MISSING),
            lambda: _genesis_entry(_signer(), run_id="run-x"),
            lambda: _genesis_entry(_signer(), previous_manifest_hash="z" * 64),
            lambda: _genesis_entry(_signer(), previous_manifest_hash=5),
            lambda: _genesis_entry(_signer(), previous_manifest_hash="a" * 63),
            lambda: _genesis_entry(_signer(), previous_manifest_hash="A" * 64),
            lambda: _genesis_entry(_signer(), manifest_version=3),
            lambda: _genesis_entry(_signer(), log_id=""),
            lambda: _genesis_entry(_signer(), log_id=5),
        ],
        ids=[
            "wrong-key-material",
            "missing-key",
            "extra-key",
            "non-hex-previous-hash",
            "non-str-previous-hash",
            "short-previous-hash",
            "uppercase-previous-hash",
            "wrong-version",
            "blank-log-id",
            "non-str-log-id",
        ],
    )
    def test_a_malformed_or_forged_genesis_is_rejected(
        self, tmp_path: Path, forged: Callable[[], dict[str, Any]]
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        audit_path.write_bytes(b"")
        _write_records(manifest_path, [forged()])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("log_id", [None, "log-a"], ids=["no-log-id", "log-id"])
    def test_a_genesis_naming_a_predecessor_head_verifies_standalone(self, tmp_path: Path, log_id: str | None) -> None:
        audit_path, manifest_path = _successor_log(tmp_path)
        genesis, *seals = _read_lines(manifest_path)
        assert genesis["previous_manifest_hash"] == _PREDECESSOR

        head = verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id=log_id)

        assert head == manifest_hash(seals[-1])
        assert seals[0]["previous_manifest_hash"] == manifest_hash(genesis)

    def test_sealing_continues_a_successor_segment(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _successor_log(tmp_path)
        _write_records(audit_path, [_record("run-c", 3)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a")

        assert [manifest["run_id"] for manifest in manifests] == ["run-c"]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a") == manifest_hash(
            manifests[-1]
        )

    def test_a_successor_segment_rotates_its_key(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _successor_log(tmp_path)

        entry = _rotate(manifest_path, _signer(_OTHER_KEY, "key-2"), _signer())

        assert entry["kind"] == "key_rotation"
        head = verify_ndjson_log(
            audit_path, manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()]
        )
        assert head == manifest_hash(entry)

    def test_deleting_both_files_and_sealing_with_a_log_id_but_no_anchor_silently_starts_a_new_log(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        audit_path.unlink()
        manifest_path.unlink()
        _write_records(audit_path, [_record("run-x", 1)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a")

        assert [manifest["run_id"] for manifest in manifests] == ["run-x"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a")
        assert _read_lines(manifest_path)[0]["kind"] == "genesis"


class TestMalformedNdjsonLines:
    """Ambiguous or non-object lines fail verification in both readers."""

    @pytest.mark.parametrize(("file_name", "key"), [("audit.ndjson", "tenant_id"), ("manifests.ndjson", "run_id")])
    def test_line_with_a_duplicate_json_key_fails_verification(self, tmp_path: Path, file_name: str, key: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        path = tmp_path / file_name
        lines = path.read_text(encoding="utf-8").splitlines()
        # Raw text: json.dumps cannot write a duplicate key.
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
        location = re.escape(f"{path} line {line_number}")

        with pytest.raises(ManifestVerificationError, match=location):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        with pytest.raises(ManifestVerificationError, match=location):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize(("file_name", "line_number"), [("audit.ndjson", 7), ("manifests.ndjson", 4)])
    def test_oversized_line_fails_naming_the_file_and_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_name: str, line_number: int
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _cap(monkeypatch)
        path = tmp_path / file_name
        _append_line(path, _oversized_line())
        location = re.escape(f"{path} line {line_number}")

        with pytest.raises(ManifestVerificationError, match=location):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        with pytest.raises(ManifestVerificationError, match=location):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

    def test_max_line_bytes_defaults_to_64_mib(self) -> None:
        assert getattr(run_manifest_module, "MAX_LINE_BYTES", None) == 64 * 1024 * 1024

    def test_the_sealed_lookup_skips_an_oversized_line(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        _cap(monkeypatch)
        _append_line(manifest_path, _oversized_line({"run_id": "run-z"}))

        assert run_manifest_module._is_run_sealed_unverified(manifest_path, "run-z") is False
        assert run_manifest_module._is_run_sealed_unverified(manifest_path, "run-a") is True

    @pytest.mark.parametrize("reader", ["read-ndjson", "sealed-lookup", "anchor-latest", "quarantine"])
    def test_no_reader_reads_an_oversized_line_whole(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str
    ) -> None:
        cap = _cap(monkeypatch, 2_000_000)
        (tmp_path / "sealed").mkdir()
        _, manifest_path = _sealed_log(tmp_path / "sealed")
        path = tmp_path / "log.ndjson"
        path.write_bytes(b'{"head": "' + b"a" * 64 + b'"}\n' + _oversized_line({"run_id": "run-z"}, 3 * cap) + b"\n")
        seen: list[int] = []
        real_open = open

        def spy_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            handle = real_open(file, mode, *args, **kwargs)
            return _ReadSpy(handle, seen) if mode == "rb" else handle

        _patch_bindings(monkeypatch, "open", spy_open)
        calls: dict[str, Callable[[], object]] = {
            "read-ndjson": lambda: list(_core._read_ndjson(path)),
            "sealed-lookup": lambda: run_manifest_module._is_run_sealed_unverified(path, "run-z"),
            "anchor-latest": lambda: NdjsonHeadAnchor(path).latest(),
            "quarantine": lambda: _quarantine(tmp_path, path, manifest_path, dry_run=True),
        }

        with suppress(ManifestVerificationError):
            calls[reader]()

        assert seen
        assert max(seen) <= cap + 1


class TestMixedAlgorithmRotation:
    """A log can mix algorithms: each key_id resolves to one signer, and a signature's algorithm must be its own."""

    def test_a_log_rotated_from_an_hmac_key_to_an_ed25519_key_seals_and_verifies(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        hmac_key, ed25519_key = HmacSha256Signer(_KEY, "key-1"), Ed25519Signer(_OTHER_KEY, "key-2")
        _write_records(audit_path, [_record("run-a", 1)])
        seal_ndjson_runs(audit_path, manifest_path, signer=hmac_key)
        _rotate(manifest_path, ed25519_key, hmac_key)
        _write_records(audit_path, [_record("run-b", 2)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=ed25519_key, previous_signers=[hmac_key])

        assert [line["signature"]["algorithm"] for line in _read_lines(manifest_path)] == [
            "HMAC-SHA256",
            "Ed25519",
            "Ed25519",
        ]
        head = verify_ndjson_log(audit_path, manifest_path, signer=ed25519_key, previous_signers=[hmac_key])
        assert head == manifest_hash(manifests[-1])

    def test_the_log_then_rotates_back_to_a_new_hmac_key_and_verifies_with_the_full_keyring(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _mixed_algorithm_log(tmp_path)

        assert [line["signature"]["algorithm"] for line in _read_lines(manifest_path)] == [
            "HMAC-SHA256",
            "Ed25519",
            "Ed25519",
            "HMAC-SHA256",
            "HMAC-SHA256",
        ]
        head = verify_ndjson_log(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])
        assert head == manifest_hash(_read_lines(manifest_path)[-1])
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])
        assert (coverage.head, coverage.sealed_runs) == (head, 3)

    def test_each_rotation_entry_is_co_signed_with_the_outgoing_keys_own_algorithm(self, tmp_path: Path) -> None:
        _, manifest_path, *_ = _mixed_algorithm_log(tmp_path)

        entries = [line for line in _read_lines(manifest_path) if line.get("kind") == "key_rotation"]

        assert [
            (entry["previous_key_signature"]["key_id"], entry["previous_key_signature"]["algorithm"])
            for entry in entries
        ] == [
            ("key-1", "HMAC-SHA256"),
            ("key-2", "Ed25519"),
        ]
        assert [entry["signature"]["algorithm"] for entry in entries] == ["Ed25519", "HMAC-SHA256"]

    def test_a_co_signature_relabelled_to_another_algorithm_is_rejected(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _mixed_algorithm_log(tmp_path)
        lines = _read_lines(manifest_path)
        lines[1]["previous_key_signature"]["algorithm"] = "Ed25519"
        _rewrite_lines(manifest_path, lines)

        with pytest.raises(ManifestVerificationError, match="algorithm"):
            verify_ndjson_log(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])

    @pytest.mark.parametrize("retired", [0, 1], ids=["hmac-key", "ed25519-key"])
    def test_a_retired_key_cannot_seal_after_its_rotation_entry(self, tmp_path: Path, retired: int) -> None:
        audit_path, manifest_path, *keys = _mixed_algorithm_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 4)])
        before = _snapshot(tmp_path)
        others = [key for index, key in enumerate(keys) if index != retired]

        with pytest.raises(ManifestVerificationError, match=_under_key("key-3", keys[retired].key_id)):
            seal_ndjson_runs(audit_path, manifest_path, signer=keys[retired], previous_signers=others)

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("retired", [0, 1], ids=["hmac-key", "ed25519-key"])
    def test_a_seal_by_a_retired_key_after_the_last_rotation_fails_verification(
        self, tmp_path: Path, retired: int
    ) -> None:
        audit_path, manifest_path, *keys = _mixed_algorithm_log(tmp_path)
        record = _record("run-d", 4)
        _write_records(audit_path, [record])
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _write_records(
            manifest_path,
            [seal_run([record], run_id="run-d", signer=keys[retired], previous_manifest_hash=head)],
        )

        with pytest.raises(ManifestVerificationError, match="run-d") as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=keys[2], previous_signers=keys[:2])

        _assert_names(excinfo, keys[retired].key_id, "key-3")

    @pytest.mark.parametrize("retired", [0, 1], ids=["hmac-key", "ed25519-key"])
    def test_rotating_back_to_a_retired_key_is_refused_whatever_its_algorithm(
        self, tmp_path: Path, retired: int
    ) -> None:
        _, manifest_path, *keys = _mixed_algorithm_log(tmp_path)
        target = keys[retired]
        before = manifest_path.read_bytes()
        head = manifest_hash(_read_lines(manifest_path)[-1])

        with pytest.raises(ManifestVerificationError, match=target.key_id):
            rotate_manifest_key(
                manifest_path,
                signer=target,
                previous_signers=[key for key in keys if key is not target],
                expected_head=head,
            )

        assert manifest_path.read_bytes() == before

    @_current_key_required
    def test_an_ed25519_key_listed_beside_an_hmac_key_of_the_same_key_id_is_a_value_error(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _mixed_algorithm_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 4)])
        before = _snapshot(tmp_path)
        clash = Ed25519Signer(_THIRD_KEY, "key-1")

        with pytest.raises(ValueError, match="previous_signers repeats key_id 'key-1'") as excinfo:
            call(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2, clash])

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    def test_rotating_to_an_ed25519_key_listed_beside_the_hmac_key_of_the_same_key_id_is_a_value_error(
        self, tmp_path: Path
    ) -> None:
        _, manifest_path = _build_log(tmp_path, ("seal", HmacSha256Signer(_KEY, "key-1")))
        before = manifest_path.read_bytes()
        head = manifest_hash(_read_lines(manifest_path)[-1])

        with pytest.raises(ValueError, match="previous_signers repeats key_id 'key-1'") as excinfo:
            rotate_manifest_key(
                manifest_path,
                signer=Ed25519Signer(_OTHER_KEY, "key-1"),
                previous_signers=[HmacSha256Signer(_KEY, "key-1")],
                expected_head=head,
            )

        assert not isinstance(excinfo.value, (ManifestVerificationError, KeyAlreadyCurrentError))
        assert manifest_path.read_bytes() == before

    @_current_key_required
    def test_an_ed25519_signer_reusing_the_key_id_of_an_hmac_sealed_log_fails_on_the_algorithm(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        audit_path, manifest_path = _build_log(tmp_path, ("seal", HmacSha256Signer(_KEY, "key-1")))
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)
        mismatch = "signature algorithm 'HMAC-SHA256' is not the signer's 'Ed25519'"

        with pytest.raises(ManifestVerificationError, match=re.escape(mismatch)):
            call(audit_path, manifest_path, signer=Ed25519Signer(_KEY, "key-1"))

        assert _snapshot(tmp_path) == before

    def test_rotating_an_hmac_sealed_log_to_an_ed25519_key_of_the_same_key_id_fails_on_the_algorithm(
        self, tmp_path: Path
    ) -> None:
        _, manifest_path = _build_log(tmp_path, ("seal", HmacSha256Signer(_KEY, "key-1")))
        before = manifest_path.read_bytes()
        head = manifest_hash(_read_lines(manifest_path)[-1])
        mismatch = "signature algorithm 'HMAC-SHA256' is not the signer's 'Ed25519'"

        with pytest.raises(ManifestVerificationError, match=re.escape(mismatch)):
            rotate_manifest_key(
                manifest_path, signer=Ed25519Signer(_OTHER_KEY, "key-1"), previous_signers=[], expected_head=head
            )

        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("public_only", [False, True], ids=["private", "public-only"])
    def test_a_manifest_hmac_signed_under_the_public_key_is_rejected_by_the_ed25519_signer(
        self, public_only: bool
    ) -> None:
        """An attacker who knows only the public key HMACs with it and claims the Ed25519 key_id."""
        records = [_record()]
        verifier = _verify_only() if public_only else Ed25519Signer(_KEY, "key-1")
        forged = seal_run(records, run_id="run-1", signer=HmacSha256Signer(_public_key(_KEY), "key-1"))
        assert forged["signature"]["algorithm"] == "HMAC-SHA256"

        with pytest.raises(ManifestVerificationError, match="algorithm"):
            verify_manifest(forged, records, signer=verifier)

    def test_an_ed25519_manifest_is_rejected_by_an_hmac_signer_of_the_same_key_id(self) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=Ed25519Signer(_KEY, "key-1"))

        with pytest.raises(ManifestVerificationError, match="algorithm"):
            verify_manifest(manifest, records, signer=HmacSha256Signer(_KEY, "key-1"))


_NON_FINITE_LINES: dict[str, bytes] = {
    "nan": b'{"run_id": "run-d", "x": NaN}\n',
    "infinity": b'{"run_id": "run-d", "x": Infinity}\n',
    "negative-infinity": b'{"run_id": "run-d", "x": -Infinity}\n',
}


@_both_algorithms
class TestSignedPayloadV2:
    """A v2 signature covers the signature block's key_id and algorithm."""

    def test_relabeled_key_id_sharing_one_secret_fails_verification(self) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer(key_id="key-1"))
        relabeled = {**manifest, "signature": {**manifest["signature"], "key_id": "key-2"}}
        verifier = _signer(key_id="key-2")

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(relabeled, records, signer=verifier, previous_signers=[_signer(key_id="key-1")])

    def test_relabeled_key_id_in_a_log_fails_verification(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        lines = _read_lines(manifest_path)
        lines[-1]["signature"]["key_id"] = "key-2"
        _rewrite_lines(manifest_path, lines)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(
                audit_path, manifest_path, signer=_signer(key_id="key-2"), previous_signers=[_signer(key_id="key-1")]
            )

    def test_changed_signature_algorithm_fails_verification(self) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        tampered = {**manifest, "signature": {**manifest["signature"], "algorithm": "OTHER"}}

        with pytest.raises(ManifestVerificationError, match="signature|algorithm"):
            verify_manifest(tampered, records, signer=_signer())

    def test_signature_covers_key_id_and_algorithm_but_not_the_value(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())
        other = {**manifest, "signature": {**manifest["signature"], "key_id": "key-2"}}

        assert _signing_payload_ref(manifest) != _signing_payload_ref(other)
        assert _signing_payload_ref(manifest) == _signing_payload_ref(
            {**manifest, "signature": {**manifest["signature"], "value": "x"}}
        )

    def test_relabeled_rotation_entry_key_id_fails_verification(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        _rotate(manifest_path, new_signer, old_signer)
        lines = _read_lines(manifest_path)
        lines[-1]["signature"]["key_id"] = "key-3"
        _rewrite_lines(manifest_path, lines)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(
                audit_path, manifest_path, signer=_signer(_OTHER_KEY, "key-3"), previous_signers=[old_signer]
            )

    def test_a_previous_key_signature_member_on_a_non_rotation_entry_stays_in_the_signed_bytes(self) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        extra = {"algorithm": "x", "key_id": "k", "value": "original"}
        entry = {**manifest, "previous_key_signature": extra}
        reduced = {**entry, "signature": {"algorithm": entry["signature"]["algorithm"], "key_id": "key-1"}}
        entry["signature"] = {**entry["signature"], "value": _signer().sign(_canonical(reduced))}
        verify_manifest(entry, records, signer=_signer())  # control: correctly signed with the member as is

        tampered = {**entry, "previous_key_signature": {**extra, "value": "changed"}}

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(tampered, records, signer=_signer())

    def test_manifest_version_3_reports_unsupported_even_with_a_bad_signature(self) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        forged = {**manifest, "manifest_version": 3}

        with pytest.raises(ManifestVerificationError, match="unsupported manifest_version 3"):
            verify_manifest(forged, records, signer=_signer())

    def test_manifest_version_3_in_a_log_reports_unsupported(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        lines = _read_lines(manifest_path)
        lines[-1]["manifest_version"] = 3
        _rewrite_lines(manifest_path, lines)

        with pytest.raises(ManifestVerificationError, match="unsupported manifest_version 3"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())


@_both_algorithms
class TestV1ReadPath:
    """v1 lines verify only as a prefix before the first v2 line, and new lines are v2."""

    def test_a_v1_log_with_a_v1_rotation_verifies(self, tmp_path: Path) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        steps = [("seal-v1", old_signer), ("seal-v1", old_signer), ("rotate-v1", new_signer)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)
        assert {line["manifest_version"] for line in _read_lines(manifest_path)} == {1}
        head = manifest_hash(_read_lines(manifest_path)[-1])

        result = verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert result == head

    def test_new_seals_appended_to_a_v1_log_are_v2_and_the_whole_log_verifies(self, tmp_path: Path) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        steps = [("seal-v1", old_signer), ("seal-v1", old_signer), ("rotate-v1", new_signer)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)
        _write_records(audit_path, [_record("run-4", 4)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert [(manifest["run_id"], manifest["manifest_version"]) for manifest in manifests] == [("run-4", 2)]
        assert [line["manifest_version"] for line in _read_lines(manifest_path)] == [1, 1, 1, 2]
        verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    def test_a_v1_seal_verifies_alone(self) -> None:
        records = [_record()]
        manifest = _as_v1(seal_run(records, run_id="run-1", signer=_signer()), _signer())

        verify_manifest(manifest, records, signer=_signer())

    def test_a_v1_line_after_a_v2_line_is_rejected(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _build_log(tmp_path, ("seal", _signer()), ("seal-v1", _signer()))

        with pytest.raises(ManifestVerificationError, match="manifest_version"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_a_v1_rotation_after_a_v2_line_is_rejected(self, tmp_path: Path) -> None:
        new_signer = _signer(_OTHER_KEY, "key-2")
        audit_path, manifest_path = _build_log(tmp_path, ("seal", _signer()), ("rotate-v1", new_signer))

        with pytest.raises(ManifestVerificationError, match="manifest_version"):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[_signer()])

    def test_a_v1_line_after_a_v2_line_stops_sealing(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _build_log(tmp_path, ("seal", _signer()), ("seal-v1", _signer()))
        _write_records(audit_path, [_record("run-9", 9)])
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert manifest_path.read_bytes() == before


@_both_algorithms
class TestNonFiniteNumbers:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
    def test_seal_run_rejects_a_non_finite_record(self, value: float) -> None:
        with pytest.raises(ValueError):
            seal_run([{**_record(), "x": value}], run_id="run-1", signer=_signer())

    def test_verify_manifest_maps_a_nan_record_to_a_verification_error(self) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError):
            verify_manifest(manifest, [{**records[0], "x": float("nan")}], signer=_signer())

    @pytest.mark.parametrize("damage", list(_NON_FINITE_LINES.values()), ids=list(_NON_FINITE_LINES))
    def test_a_non_finite_audit_line_fails_verification(self, tmp_path: Path, damage: bytes) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 2, damage)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("damage", list(_NON_FINITE_LINES.values()), ids=list(_NON_FINITE_LINES))
    def test_a_non_finite_audit_line_counts_as_damaged_for_quarantine(self, tmp_path: Path, damage: bytes) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        clean = audit_path.read_bytes()
        _insert_line(audit_path, 2, damage)
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [3])
        assert audit_path.read_bytes() == clean
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head

    @pytest.mark.parametrize("damage", list(_NON_FINITE_LINES.values()), ids=list(_NON_FINITE_LINES))
    def test_a_non_finite_manifest_line_is_not_repaired(self, tmp_path: Path, damage: bytes) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(manifest_path, 1, damage)

        _assert_refused(tmp_path, audit_path, manifest_path)
