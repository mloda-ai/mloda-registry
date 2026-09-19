"""Tests for run_manifest: HmacSha256Signer, the pure seal_run / manifest_hash / verify_manifest trio and the
NDJSON sealing and verification built on them; file tests write their audit records through NdjsonAuditSink."""

from __future__ import annotations

import base64
import copy
import dataclasses
import errno
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
from unittest.mock import patch

import pytest
from mloda.steward import verified_context
from mloda.user import ParallelizationMode

import mloda.enterprise.extenders.audit as audit_package
import mloda.enterprise.extenders.audit.run_manifest as run_manifest_module
from mloda.enterprise.extenders.audit import (
    AuditExtender,
    HmacSha256Signer,
    LogCoverage,
    ManifestSigner,
    ManifestVerificationError,
    NdjsonAuditSink,
    QuarantinedLine,
    RunAlreadySealedError,
    RunNotPendingError,
    manifest_hash,
    quarantine_damaged_lines,
    seal_ndjson_runs,
    seal_run,
    verify_manifest,
    verify_ndjson_log,
    verify_ndjson_log_coverage,
)
from mloda.enterprise.extenders.audit.audit_extender import _append_records
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

_EXPECTED_QUARANTINE_KEYS = {
    "quarantine_version",
    "quarantined_at",
    "file",
    "path",
    "line",
    "offset",
    "length",
    "sha256",
    "reason",
    "raw_base64",
    "signature",
}

# Lines both readers reject; the last is the one that raises RecursionError.
_DAMAGED_LINES: dict[str, bytes] = {
    "blank": b"\n",
    "bad-utf8": b"\xff\xfe\n",
    "truncated": b'{"run_id": "run-d"\n',
    "not-an-object": b"[]\n",
    "duplicate-key": b'{"run_id": "run-d", "run_id": "run-e"}\n',
    "deep-nesting": b"[" * 100000 + b"\n",
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


def _torn(path: Path, tail: bytes) -> None:
    """Append `tail` without a newline, as a crash mid-append leaves it."""
    path.write_bytes(path.read_bytes() + tail)


def _insert_line(path: Path, index: int, line: bytes) -> None:
    """Put `line` before the 0-based line `index`; every other byte of the file stays as it was."""
    lines = path.read_bytes().splitlines(keepends=True)
    lines.insert(index, line)
    path.write_bytes(b"".join(lines))


def _torn_manifest_log(directory: Path) -> tuple[Path, Path, str]:
    """A sealed log whose manifest log ends in half of a manifest line (line 4), and the head before the damage."""
    audit_path, manifest_path = _sealed_log(directory)
    head = manifest_hash(_read_lines(manifest_path)[-1])
    last = manifest_path.read_bytes().splitlines(keepends=True)[-1]
    _torn(manifest_path, last[: len(last) // 2])
    return audit_path, manifest_path, head


def _damaged_log(directory: Path) -> tuple[Path, Path, str]:
    """A torn manifest log plus an undecodable audit line (line 4) and a torn last audit line (line 8)."""
    audit_path, manifest_path, head = _torn_manifest_log(directory)
    _insert_line(audit_path, 3, b"\xff\xfe\n")
    _torn(audit_path, b'{"run_id": "run-d", "tenant')
    return audit_path, manifest_path, head


def _log_cut_mid_write(directory: Path) -> tuple[Path, Path, list[str]]:
    """Seal run-a alone, then run-b and run-c in one write that a crash cut inside run-c's line (line 3). Returns the
    heads after a, b and c; the log keeps the first two manifests and half of the third."""
    audit_path = directory / "audit.ndjson"
    manifest_path = directory / "manifests.ndjson"
    _write_records(audit_path, [_record("run-a", 1)])
    heads = [manifest_hash(manifest) for manifest in seal_ndjson_runs(audit_path, manifest_path, signer=_signer())]
    _write_records(audit_path, [_record("run-b", 2), _record("run-c", 3)])
    heads += [manifest_hash(manifest) for manifest in seal_ndjson_runs(audit_path, manifest_path, signer=_signer())]
    assert len(heads) == 3
    lines = manifest_path.read_bytes().splitlines(keepends=True)
    manifest_path.write_bytes(b"".join(lines[:2]) + lines[2][: len(lines[2]) // 2])
    return audit_path, manifest_path, heads


def _trace(directory: Path) -> Path:
    """Where `_quarantine` writes the quarantine log."""
    return directory / "quarantine.ndjson"


def _quarantine(
    directory: Path,
    audit_path: Path,
    manifest_path: Path,
    *,
    signer: ManifestSigner | None = None,
    expected_head: str | None = None,
    dry_run: bool = False,
) -> list[QuarantinedLine]:
    return quarantine_damaged_lines(
        audit_path,
        manifest_path,
        quarantine_path=_trace(directory),
        signer=signer or _signer(),
        expected_head=expected_head,
        dry_run=dry_run,
    )


def _snapshot(directory: Path) -> dict[str, bytes]:
    """Every file of `directory` with its bytes: equal snapshots mean nothing was changed, created or left behind."""
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def _spans(file: str, data: bytes, numbers: Iterable[int]) -> list[tuple[str, int, int, int, str]]:
    """What a recovery reports for the 1-based lines `numbers` of a file holding `data`, worked out from the bytes."""
    lines = data.splitlines(keepends=True)
    return [(file, n, len(b"".join(lines[: n - 1])), len(lines[n - 1]), _sha256(lines[n - 1])) for n in numbers]


def _summary(removed: Iterable[QuarantinedLine]) -> list[tuple[str, int, int, int, str]]:
    return [(item.file, item.line, item.offset, item.length, item.sha256) for item in removed]


def _assert_refused(
    directory: Path,
    audit_path: Path,
    manifest_path: Path,
    *,
    signer: ManifestSigner | None = None,
    expected_head: str | None = None,
    dry_run: bool = False,
) -> None:
    """Recovery must raise ManifestVerificationError and leave the directory exactly as it was."""
    before = _snapshot(directory)

    with pytest.raises(ManifestVerificationError):
        _quarantine(directory, audit_path, manifest_path, signer=signer, expected_head=expected_head, dry_run=dry_run)

    assert _snapshot(directory) == before


def _lock_refused(path: Path) -> bool:
    """Whether a second descriptor is refused an exclusive lock on `path`, so another owner holds a lock on it."""
    fcntl = pytest.importorskip("fcntl")
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def _flock_unsupported(fd: int, operation: int) -> None:
    """The error flock raises on a file system without lock support, such as NFS without a lock manager."""
    raise OSError(errno.ENOLCK, "no locks")


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
            "RunAlreadySealedError",
            "seal_run",
            "manifest_hash",
            "verify_manifest",
            "seal_ndjson_runs",
            "verify_ndjson_log",
            "LogCoverage",
            "verify_ndjson_log_coverage",
            "QuarantinedLine",
            "quarantine_damaged_lines",
        } <= set(audit_package.__all__)

    def test_run_not_pending_error_is_a_value_error_but_not_a_verification_error(self) -> None:
        # An idempotent `except RunNotPendingError` must not swallow a broken chain.
        assert issubclass(RunNotPendingError, ValueError)
        assert not issubclass(RunNotPendingError, ManifestVerificationError)

    def test_run_already_sealed_error_is_a_run_not_pending_error(self) -> None:
        # Catching RunNotPendingError still covers it; catching RunAlreadySealedError alone is the idempotent retry.
        assert issubclass(RunAlreadySealedError, RunNotPendingError)


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

        with pytest.raises(RunNotPendingError) as excinfo:
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-missing")

        # A caller retrying idempotently on RunAlreadySealedError must not swallow a run_id that never had records.
        assert not isinstance(excinfo.value, RunAlreadySealedError)

    def test_explicit_run_id_already_sealed_raises_run_already_sealed_error(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-a")
        before = manifest_path.read_bytes()

        with pytest.raises(RunAlreadySealedError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-a")

        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("run_id", ["", "   ", 123], ids=["empty", "blank", "non-str"])
    def test_unusable_explicit_run_id_raises_value_error_and_creates_no_manifest_log(
        self, tmp_path: Path, run_id: Any
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])

        with pytest.raises(ValueError, match="run_id") as excinfo:
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id=run_id)

        # A usage error is not a run that is merely not pending, and it fails before the lock creates the log.
        assert not isinstance(excinfo.value, RunNotPendingError)
        assert not manifest_path.exists()

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

    def test_failed_append_is_rolled_back_and_a_retry_seals_every_run(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _write_records(audit_path, [_record("run-d", 7), _record("run-e", 8), _record("run-f", 9)])
        before = manifest_path.read_bytes()
        real_write = os.write
        landed: list[int] = []

        def disk_fills_after_the_first_manifest(fd: int, data: bytes | memoryview) -> int:
            payload = bytes(data)
            if b"record_hashes" not in payload:
                return real_write(fd, data)
            if landed:
                raise OSError(errno.ENOSPC, "disk full")
            # A short write: only the first manifest line reaches the file, then the disk is full.
            landed.append(real_write(fd, payload[: payload.index(b"\n") + 1]))
            return landed[0]

        with patch("os.write", side_effect=disk_fills_after_the_first_manifest):
            with pytest.raises(OSError):
                seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head)

        assert landed
        # The sealer holds the lock and is the only writer, so it removes the partial bytes it left behind.
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

    def test_sealing_holds_an_exclusive_lock_on_the_manifest_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        refused: list[bool] = []
        appends: list[bool] = []

        def probe() -> bool:
            # A second descriptor is a second lock owner, so it is refused while the sealer holds the lock.
            fd = os.open(manifest_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return False
            except BlockingIOError:
                return True
            finally:
                os.close(fd)

        class _LockProbeSigner(HmacSha256Signer):
            def sign(self, payload: bytes) -> str:
                # Verifying the sealed log signs too.
                refused.append(probe())
                return super().sign(payload)

        def spy(*args: Any, **kwargs: Any) -> None:
            # The write itself must run under the lock, not only the signing before it.
            refused.append(probe())
            appends.append(True)
            _append_records(*args, **kwargs)

        monkeypatch.setattr(run_manifest_module, "_append_records", spy)

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_LockProbeSigner(_KEY, "key-1"))

        assert [manifest["run_id"] for manifest in manifests] == ["run-d"]
        assert appends == [True]
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

    def test_sealing_fails_when_the_exclusive_lock_is_unsupported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = manifest_path.read_bytes()
        monkeypatch.setattr(fcntl, "flock", _flock_unsupported)

        # Sealing without the lock could fork the chain, so unlike verifying it never proceeds unlocked.
        with pytest.raises(OSError) as excinfo:
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert excinfo.value.errno == errno.ENOLCK
        assert manifest_path.read_bytes() == before

    def test_missing_audit_file_still_fails_sealing(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        audit_path.unlink()
        before = manifest_path.read_bytes()

        # A typo in the audit path must stay loud, unlike verification, which reads a missing file as empty.
        with pytest.raises(FileNotFoundError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert manifest_path.read_bytes() == before

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

    def test_at_most_twenty_failing_runs_are_reported_and_the_rest_are_counted(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record(f"run-{number:02d}", number) for number in range(25)])
        assert len(seal_ndjson_runs(audit_path, manifest_path, signer=_signer())) == 25
        # Emptied, not deleted, so that only the cap is under test; a deleted file has its own test.
        audit_path.write_bytes(b"")

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        message = str(excinfo.value)
        assert "and 5 more" in message
        assert message.count("do not match the record_hashes") == 20

    def test_a_head_mismatch_is_reported_even_when_more_than_twenty_runs_fail(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record(f"run-{number:02d}", number) for number in range(25)])
        assert len(seal_ndjson_runs(audit_path, manifest_path, signer=_signer())) == 25
        manifests = _read_lines(manifest_path)
        head = manifest_hash(manifests[-1])
        _rewrite_lines(manifest_path, manifests[:-1])
        audit_path.write_bytes(b"")

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), expected_head=head)

        # Only the head detects the cut log, so the 20 problem cap must never hide it: 24 records plus it is 25.
        message = str(excinfo.value)
        assert "expected head" in message
        assert message.count("do not match the record_hashes") == 19
        assert "and 5 more" in message

    def test_deleted_audit_file_fails_every_sealed_run(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        audit_path.unlink()

        # A missing audit file is an empty one: the seals stay and nothing backs them, which is a tamper alarm.
        with pytest.raises(ManifestVerificationError, match="record") as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        message = str(excinfo.value)
        assert all(run_id in message for run_id in ("run-a", "run-b", "run-c"))

    def test_neither_audit_file_nor_manifest_log_returns_none(self, tmp_path: Path) -> None:
        assert verify_ndjson_log(tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson", signer=_signer()) is None

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

    def test_verification_proceeds_unlocked_when_the_shared_lock_is_unsupported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        monkeypatch.setattr(fcntl, "flock", _flock_unsupported)

        # The shared lock is best effort: a file system that cannot lock must not make verifying impossible.
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head


class TestVerifyNdjsonLogCoverage:
    """verify_ndjson_log_coverage verifies exactly like verify_ndjson_log and also counts what the seals leave out."""

    def test_fully_sealed_log_covers_every_line(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2), _record("run-a", 3)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert coverage == LogCoverage(
            head=manifest_hash(_read_lines(manifest_path)[-1]),
            sealed_runs=2,
            sealed_lines=3,
            unattributed_lines=0,
            unsealed_lines={},
        )

    def test_head_is_what_verify_ndjson_log_returns(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert coverage.head == verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        assert coverage.head == manifest_hash(_read_lines(manifest_path)[-1])

    @pytest.mark.parametrize(
        ("run_id", "drop_key"),
        [(None, False), ("", False), ("   ", False), (None, True)],
        ids=["null", "empty", "blank", "absent"],
    )
    def test_lines_without_a_run_id_are_unattributed(self, tmp_path: Path, run_id: str | None, drop_key: bool) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        records = [_record(run_id, 1), _record("run-a", 2), _record(run_id, 3)]
        if drop_key:
            del records[0]["run_id"]
            del records[2]["run_id"]
        _write_records(audit_path, records)
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert coverage == LogCoverage(
            head=manifest_hash(_read_lines(manifest_path)[-1]),
            sealed_runs=1,
            sealed_lines=1,
            unattributed_lines=2,
            unsealed_lines={},
        )

    def test_lines_of_runs_that_are_not_sealed_yet_are_counted_per_run(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        # run-e appears first although run-d sorts first.
        _write_records(
            audit_path,
            [_record("run-e", 7), _record("run-d", 8), _record("run-d", 9), _record("run-e", 10), _record("run-d", 11)],
        )

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert list(coverage.unsealed_lines.items()) == [("run-e", 2), ("run-d", 3)]

    def test_mixed_log_counts_every_kind_of_line_once(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        absent = _record(None, 10)
        del absent["run_id"]
        _write_records(
            audit_path,
            [_record("run-d", 7), _record("   ", 8), _record("run-e", 9), absent, _record("run-d", 11)],
        )

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert coverage.head == manifest_hash(_read_lines(manifest_path)[-1])
        assert coverage.sealed_runs == 3
        assert coverage.sealed_lines == 5
        assert coverage.unattributed_lines == 3
        assert list(coverage.unsealed_lines.items()) == [("run-d", 2), ("run-e", 1)]
        total = coverage.sealed_lines + coverage.unattributed_lines + sum(coverage.unsealed_lines.values())
        assert total == len(_read_lines(audit_path))

    @pytest.mark.parametrize("log_bytes", [None, b""], ids=["missing", "empty"])
    def test_missing_or_empty_manifest_log_leaves_every_line_unsealed_or_unattributed(
        self, tmp_path: Path, log_bytes: bytes | None
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-a", 2), _record(None, 3), _record("run-b", 4)])
        if log_bytes is not None:
            manifest_path.write_bytes(log_bytes)

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert coverage == LogCoverage(
            head=None, sealed_runs=0, sealed_lines=0, unattributed_lines=1, unsealed_lines={"run-a": 2, "run-b": 1}
        )
        assert manifest_path.exists() == (log_bytes is not None)

    @pytest.mark.parametrize("log_bytes", [None, b""], ids=["missing", "empty"])
    def test_missing_audit_file_without_manifests_is_an_empty_coverage(
        self, tmp_path: Path, log_bytes: bytes | None
    ) -> None:
        manifest_path = tmp_path / "manifests.ndjson"
        if log_bytes is not None:
            manifest_path.write_bytes(log_bytes)

        coverage = verify_ndjson_log_coverage(tmp_path / "audit.ndjson", manifest_path, signer=_signer())

        assert coverage == LogCoverage(
            head=None, sealed_runs=0, sealed_lines=0, unattributed_lines=0, unsealed_lines={}
        )

    def test_untouched_log_with_a_matching_expected_head_returns_the_coverage(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer(), expected_head=head)

        assert coverage.head == head
        assert coverage.sealed_runs == 3

    def test_edited_audit_line_of_a_sealed_run_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[0]["run_id"] == "run-a"
        records[0]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, records)

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

    def test_signer_with_a_different_key_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer(_OTHER_KEY))

    def test_wrong_expected_head_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        with pytest.raises(ManifestVerificationError, match="head"):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer(), expected_head="ab" * 32)

    def test_deleted_audit_file_fails_every_sealed_run(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        audit_path.unlink()

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("run_id", [["x"], 5], ids=["list", "number"])
    def test_audit_record_whose_run_id_is_neither_a_string_nor_null_fails(self, tmp_path: Path, run_id: Any) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [{**_record(), "run_id": run_id}])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("file_name", ["audit.ndjson", "manifests.ndjson"])
    def test_json_line_that_is_not_an_object_fails(self, tmp_path: Path, file_name: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _append_line(tmp_path / file_name, "[]")

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

    def test_verification_waits_for_a_sealer_holding_the_lock(self, tmp_path: Path) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _sealed_log(tmp_path)

        with ThreadPoolExecutor(max_workers=1) as pool:
            fd = os.open(manifest_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                future = pool.submit(verify_ndjson_log_coverage, audit_path, manifest_path, signer=_signer())
                done, _ = wait([future], timeout=0.3)
            finally:
                os.close(fd)

            assert not done
            assert future.result(timeout=10).sealed_runs == 3

    def test_verify_ndjson_log_still_returns_only_the_head(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7), _record("", 8)])

        head = verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert isinstance(head, str)
        assert head == verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer()).head

    def test_log_coverage_is_frozen(self) -> None:
        coverage = LogCoverage(head=None, sealed_runs=0, sealed_lines=0, unattributed_lines=0, unsealed_lines={})

        with pytest.raises(dataclasses.FrozenInstanceError):
            coverage.sealed_runs = 1  # type: ignore[misc]


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


class TestQuarantineDamagedLines:
    """quarantine_damaged_lines removes what the readers reject from the audit file and a torn tail from the manifest
    log, records every removed byte in a signed quarantine log first, and refuses any repair a seal could hide."""

    def test_quarantined_line_is_frozen(self) -> None:
        line = QuarantinedLine(file="audit", line=1, offset=0, length=1, sha256="0" * 64, reason="not valid JSON")

        with pytest.raises(dataclasses.FrozenInstanceError):
            line.line = 2  # type: ignore[misc]

    def test_quarantined_line_has_exactly_the_specified_fields(self) -> None:
        assert [field.name for field in dataclasses.fields(QuarantinedLine)] == [
            "file",
            "line",
            "offset",
            "length",
            "sha256",
            "reason",
        ]

    def test_torn_manifest_tail_is_reported_by_a_dry_run_and_nothing_changes(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path, dry_run=True)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [4])
        assert all(item.reason for item in removed)
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("anchored", [False, True], ids=["unanchored", "anchored"])
    def test_torn_manifest_tail_is_truncated_and_the_chain_continues(self, tmp_path: Path, anchored: bool) -> None:
        audit_path, manifest_path, head = _torn_manifest_log(tmp_path)
        before = _snapshot(tmp_path)
        manifest_before = before["manifests.ndjson"]

        removed = _quarantine(tmp_path, audit_path, manifest_path, expected_head=head if anchored else None)

        assert _summary(removed) == _spans("manifest", manifest_before, [4])
        assert manifest_path.read_bytes() == manifest_before[: manifest_before.rindex(b"\n") + 1]
        assert audit_path.read_bytes() == before["audit.ndjson"]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head
        _write_records(audit_path, [_record("run-d", 7)])
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head)
        assert [(manifest["run_id"], manifest["previous_manifest_hash"]) for manifest in manifests] == [("run-d", head)]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == manifest_hash(manifests[-1])

    def test_trace_has_one_entry_per_removed_line_in_removal_order(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert len(removed) == len(entries) == 3
        assert [
            (entry["file"], entry["line"], entry["offset"], entry["length"], entry["sha256"]) for entry in entries
        ] == [
            *_spans("manifest", before["manifests.ndjson"], [4]),
            *_spans("audit", before["audit.ndjson"], [4, 8]),
        ]
        for entry, item in zip(entries, removed, strict=True):
            assert set(entry) == _EXPECTED_QUARANTINE_KEYS
            assert entry["quarantine_version"] == 1
            assert entry["path"] == str(manifest_path if item.file == "manifest" else audit_path)
            assert entry["reason"] == item.reason
        # Written through _append_records, so a line is the canonical JSON of its entry.
        assert _trace(tmp_path).read_bytes() == b"".join(_canonical(entry) + b"\n" for entry in entries)

    def test_trace_keeps_the_removed_bytes_so_they_can_be_inspected_or_restored(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        audit_lines = audit_path.read_bytes().splitlines(keepends=True)
        manifest_lines = manifest_path.read_bytes().splitlines(keepends=True)

        _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        removed_bytes = [manifest_lines[3], audit_lines[3], audit_lines[7]]
        assert [base64.b64decode(entry["raw_base64"], validate=True) for entry in entries] == removed_bytes
        assert [entry["sha256"] for entry in entries] == [_sha256(raw) for raw in removed_bytes]

    def test_trace_signature_covers_the_entry_without_its_signature(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)

        _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert entries
        for entry in entries:
            signature = entry["signature"]
            payload = _canonical(_unsigned(entry))
            assert set(signature) == {"algorithm", "key_id", "value"}
            assert (signature["algorithm"], signature["key_id"]) == (_signer().algorithm, _signer().key_id)
            assert signature["value"] == hmac.new(_KEY, payload, hashlib.sha256).hexdigest()
            tampered = _canonical(_unsigned({**entry, "line": entry["line"] + 1}))
            assert _signer().verify(tampered, signature["value"]) is False

    def test_quarantined_at_is_rfc3339_utc(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)

        _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert entries
        for entry in entries:
            quarantined_at = entry["quarantined_at"]
            assert quarantined_at.endswith("Z")
            datetime.fromisoformat(quarantined_at.removesuffix("Z"))

    def test_new_trace_file_has_owner_only_permissions(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)

        _quarantine(tmp_path, audit_path, manifest_path)

        assert stat.S_IMODE(_trace(tmp_path).stat().st_mode) == 0o600

    def test_a_second_recovery_appends_to_the_trace(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        first = _trace(tmp_path).read_bytes()
        _append_line(audit_path, b"[]")
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [7])
        trace = _trace(tmp_path).read_bytes()
        assert trace.startswith(first)
        assert len(_read_lines(_trace(tmp_path))) == 4

    def test_str_paths_are_accepted_and_recorded_as_given(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)

        removed = quarantine_damaged_lines(
            str(audit_path), str(manifest_path), quarantine_path=str(_trace(tmp_path)), signer=_signer()
        )

        assert [item.file for item in removed] == ["manifest", "audit", "audit"]
        entries = _read_lines(_trace(tmp_path))
        assert [entry["path"] for entry in entries] == [str(manifest_path), str(audit_path), str(audit_path)]

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    @pytest.mark.parametrize(
        "tail",
        [b"{}", b"[]", b"null", b'{"run_id": "run-d"}'],
        ids=["empty-object", "array", "null", "object"],
    )
    def test_unterminated_manifest_tail_that_is_valid_json_is_not_repaired(
        self, tmp_path: Path, tail: bytes, dry_run: bool
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")
        _torn(manifest_path, tail)

        _assert_refused(tmp_path, audit_path, manifest_path, dry_run=dry_run)

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_a_seal_missing_only_its_newline_is_not_repaired(self, tmp_path: Path, dry_run: bool) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")
        manifest_path.write_bytes(manifest_path.read_bytes().removesuffix(b"\n"))

        _assert_refused(tmp_path, audit_path, manifest_path, dry_run=dry_run)

    @pytest.mark.parametrize("index", [1, 3], ids=["middle", "last"])
    @pytest.mark.parametrize("damage", list(_DAMAGED_LINES.values()), ids=list(_DAMAGED_LINES))
    def test_terminated_damaged_manifest_line_is_not_repaired(self, tmp_path: Path, damage: bytes, index: int) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")
        _insert_line(manifest_path, index, damage)

        _assert_refused(tmp_path, audit_path, manifest_path)

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    @pytest.mark.parametrize("failure", ["edited-manifest", "broken-chain", "other-key", "wrong-head"])
    def test_manifest_log_that_does_not_verify_is_not_repaired(
        self, tmp_path: Path, failure: str, dry_run: bool
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        lines = manifest_path.read_bytes().splitlines(keepends=True)
        if failure == "edited-manifest":
            edited = lines[0].replace(b'"compliant": true', b'"compliant": false')
            assert edited != lines[0]
            lines[0] = edited
        if failure == "broken-chain":
            del lines[1]
        manifest_path.write_bytes(b"".join(lines))

        _assert_refused(
            tmp_path,
            audit_path,
            manifest_path,
            signer=_signer(_OTHER_KEY) if failure == "other-key" else None,
            expected_head="0" * 64 if failure == "wrong-head" else None,
            dry_run=dry_run,
        )

    @pytest.mark.parametrize("anchor", [0, 1], ids=["first-manifest", "second-manifest"])
    def test_expected_head_of_any_complete_manifest_repairs_a_cut_multi_seal_write(
        self, tmp_path: Path, anchor: int
    ) -> None:
        audit_path, manifest_path, heads = _log_cut_mid_write(tmp_path)
        before = _snapshot(tmp_path)
        manifest_before = before["manifests.ndjson"]

        removed = _quarantine(tmp_path, audit_path, manifest_path, expected_head=heads[anchor])

        # Only the fragment goes: run-b's complete manifest stays, so the log ends at its head.
        assert _summary(removed) == _spans("manifest", manifest_before, [3])
        assert manifest_path.read_bytes() == manifest_before[: manifest_before.rindex(b"\n") + 1]
        assert audit_path.read_bytes() == before["audit.ndjson"]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == heads[1]
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=heads[1])
        assert [(manifest["run_id"], manifest["previous_manifest_hash"]) for manifest in manifests] == [
            ("run-c", heads[1])
        ]

    def test_dry_run_accepts_the_expected_head_of_an_earlier_manifest(self, tmp_path: Path) -> None:
        audit_path, manifest_path, heads = _log_cut_mid_write(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path, expected_head=heads[0], dry_run=True)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [3])
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    @pytest.mark.parametrize("anchor", ["unrelated", "cut-manifest"])
    def test_expected_head_of_no_complete_manifest_is_not_repaired(
        self, tmp_path: Path, anchor: str, dry_run: bool
    ) -> None:
        audit_path, manifest_path, heads = _log_cut_mid_write(tmp_path)

        # The cut manifest's own hash is no complete manifest's hash: accepting it would drop a seal it anchors.
        _assert_refused(
            tmp_path,
            audit_path,
            manifest_path,
            expected_head="0" * 64 if anchor == "unrelated" else heads[2],
            dry_run=dry_run,
        )

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_a_signer_with_another_key_is_refused_whatever_the_expected_head(
        self, tmp_path: Path, dry_run: bool
    ) -> None:
        audit_path, manifest_path, heads = _log_cut_mid_write(tmp_path)

        _assert_refused(
            tmp_path, audit_path, manifest_path, signer=_signer(_OTHER_KEY), expected_head=heads[0], dry_run=dry_run
        )

    def test_a_broken_chain_is_refused_even_with_the_head_of_a_manifest_before_the_break(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        lines = manifest_path.read_bytes().splitlines(keepends=True)
        del lines[1]
        manifest_path.write_bytes(b"".join(lines))

        _assert_refused(tmp_path, audit_path, manifest_path, expected_head=manifest_hash(json.loads(lines[0])))

    @pytest.mark.parametrize("anchor", ["unrelated", "sealed-manifest"])
    def test_any_expected_head_is_refused_when_the_log_has_no_complete_manifest(
        self, tmp_path: Path, anchor: str
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        first = manifest_path.read_bytes().splitlines(keepends=True)[0]
        manifest_path.write_bytes(first[: len(first) // 2])

        _assert_refused(
            tmp_path,
            audit_path,
            manifest_path,
            expected_head="0" * 64 if anchor == "unrelated" else manifest_hash(json.loads(first)),
        )

    def test_a_log_that_is_only_a_torn_fragment_is_truncated_to_empty(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        first = manifest_path.read_bytes().splitlines(keepends=True)[0]
        manifest_path.write_bytes(first[: len(first) // 2])
        manifest_before = manifest_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("manifest", manifest_before, [1])
        assert manifest_path.read_bytes() == b""
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) is None

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_clean_files_return_nothing_and_change_nothing(self, tmp_path: Path, dry_run: bool) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)

        assert _quarantine(tmp_path, audit_path, manifest_path, dry_run=dry_run) == []

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize(
        "tail",
        [b'{"run_id": "run-d", "tenant', b"\xff\xfe", b'{"run_id": "run-d", "tenant": "\xc3'],
        ids=["torn", "bad-utf8", "torn-mid-character"],
    )
    def test_unterminated_audit_tail_that_does_not_parse_is_removed(self, tmp_path: Path, tail: bytes) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        clean = audit_path.read_bytes()
        _torn(audit_path, tail)
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [7])
        assert removed[0].length == len(tail)
        assert audit_path.read_bytes() == clean
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    @pytest.mark.parametrize(
        "tail",
        [_canonical(_record("run-d", 7)), b"[]", b"{}", b"null", b"5", b'"x"'],
        ids=["complete-record", "array", "empty-object", "null", "number", "string"],
    )
    def test_unterminated_audit_tail_that_is_valid_json_is_not_repaired(
        self, tmp_path: Path, tail: bytes, dry_run: bool
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        # A line the recovery would otherwise remove: the refusal covers the whole file.
        _insert_line(audit_path, 3, b"[]\n")
        _torn(audit_path, tail)

        _assert_refused(tmp_path, audit_path, manifest_path, dry_run=dry_run)

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_a_sealed_record_missing_only_its_newline_is_not_repaired(self, tmp_path: Path, dry_run: bool) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        audit_path.write_bytes(audit_path.read_bytes().removesuffix(b"\n"))

        _assert_refused(tmp_path, audit_path, manifest_path, dry_run=dry_run)

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_a_refused_audit_tail_also_stops_the_manifest_repair(self, tmp_path: Path, dry_run: bool) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        _torn(audit_path, b"[]")

        _assert_refused(tmp_path, audit_path, manifest_path, dry_run=dry_run)

    @pytest.mark.parametrize("damage", list(_DAMAGED_LINES.values()), ids=list(_DAMAGED_LINES))
    def test_damaged_audit_line_is_removed_and_the_others_keep_their_bytes(self, tmp_path: Path, damage: bytes) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        clean = audit_path.read_bytes()
        _insert_line(audit_path, 2, damage)
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [3])
        assert all(item.reason for item in removed)
        assert audit_path.read_bytes() == clean
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head

    def test_every_kind_of_damaged_audit_line_is_removed_in_file_order(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        clean = audit_path.read_bytes()
        lines = clean.splitlines(keepends=True)
        damages = list(_DAMAGED_LINES.values())
        interleaved = [piece for pair in zip(lines, damages, strict=False) for piece in pair] + lines[len(damages) :]
        audit_path.write_bytes(b"".join(interleaved))
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, range(2, 2 * len(damages) + 1, 2))
        assert audit_path.read_bytes() == clean
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head

    @pytest.mark.parametrize(
        ("index", "damage", "run_id"),
        [
            (3, lambda line: b"garbled\n", "run-b"),
            (5, lambda line: line[:20], "run-c"),
        ],
        ids=["garbled", "torn-last-line"],
    )
    def test_damaged_line_of_a_sealed_run_is_removed_but_verification_still_fails(
        self, tmp_path: Path, index: int, damage: Any, run_id: str
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        lines = audit_path.read_bytes().splitlines(keepends=True)
        lines[index] = damage(lines[index])
        audit_path.write_bytes(b"".join(lines))
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        # The seal covers the exact bytes of the line, so removing it is never hidden, and the trace names it.
        assert _summary(removed) == _spans("audit", damaged, [index + 1])
        with pytest.raises(ManifestVerificationError, match=run_id):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        assert base64.b64decode(_read_lines(_trace(tmp_path))[0]["raw_base64"]) == lines[index]

    @pytest.mark.parametrize("run_id", [["x"], 5], ids=["list", "number"])
    def test_audit_line_with_a_non_string_run_id_is_not_damage_and_still_fails_verification(
        self, tmp_path: Path, run_id: Any
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [{**_record(), "run_id": run_id}])
        kept = audit_path.read_bytes()
        _append_line(audit_path, "")
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [8])
        assert audit_path.read_bytes() == kept
        with pytest.raises(ManifestVerificationError, match="run_id"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_missing_audit_file_is_nothing_to_repair_and_the_manifest_tail_is_still_repaired(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        audit_path.unlink()
        manifest_before = manifest_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("manifest", manifest_before, [4])
        assert manifest_path.read_bytes() == manifest_before[: manifest_before.rindex(b"\n") + 1]
        assert not audit_path.exists()

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_neither_audit_file_nor_manifest_log_returns_nothing_and_creates_nothing(
        self, tmp_path: Path, dry_run: bool
    ) -> None:
        audit_path, manifest_path = tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson"

        assert _quarantine(tmp_path, audit_path, manifest_path, dry_run=dry_run) == []

        assert _snapshot(tmp_path) == {}

    def test_audit_file_is_repaired_before_anything_was_sealed(self, tmp_path: Path) -> None:
        audit_path, manifest_path = tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])
        clean = audit_path.read_bytes()
        _torn(audit_path, b'{"run_id": "run-c", "tena')
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [3])
        assert audit_path.read_bytes() == clean
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        assert [manifest["run_id"] for manifest in manifests] == ["run-a", "run-b"]

    def test_both_files_are_repaired_manifest_entry_first_and_sealing_works_again(self, tmp_path: Path) -> None:
        audit_path, manifest_path, head = _damaged_log(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path, expected_head=head)

        assert _summary(removed) == [
            *_spans("manifest", before["manifests.ndjson"], [4]),
            *_spans("audit", before["audit.ndjson"], [4, 8]),
        ]
        entries = _read_lines(_trace(tmp_path))
        assert [(entry["file"], entry["line"]) for entry in entries] == [("manifest", 4), ("audit", 4), ("audit", 8)]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head
        _write_records(audit_path, [_record("run-d", 7)])
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head)
        assert [(manifest["run_id"], manifest["previous_manifest_hash"]) for manifest in manifests] == [("run-d", head)]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_dry_run_returns_what_a_real_run_removes_and_changes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        before = _snapshot(tmp_path)

        planned = _quarantine(tmp_path, audit_path, manifest_path, dry_run=True)

        assert len(planned) == 3
        assert _snapshot(tmp_path) == before
        assert planned == _quarantine(tmp_path, audit_path, manifest_path)

    def test_dry_run_creates_no_file_when_the_manifest_log_does_not_exist(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])
        _torn(audit_path, b'{"run_id": "run-a", "tena')
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, tmp_path / "manifests.ndjson", dry_run=True)

        assert _summary(removed) == _spans("audit", before["audit.ndjson"], [2])
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("mode", [0o600, 0o640], ids=["owner-only", "group-readable"])
    def test_rewritten_audit_file_keeps_its_permission_bits_and_leaves_no_temporary_file(
        self, tmp_path: Path, mode: int
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")
        audit_path.chmod(mode)

        _quarantine(tmp_path, audit_path, manifest_path)

        assert stat.S_IMODE(audit_path.stat().st_mode) == mode
        assert sorted(_snapshot(tmp_path)) == ["audit.ndjson", "manifests.ndjson", "quarantine.ndjson"]

    def test_failed_audit_replacement_keeps_the_trace_and_the_original_audit_file(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")
        _insert_line(audit_path, 5, b"\n")
        before = _snapshot(tmp_path)

        with patch("os.replace", side_effect=OSError(errno.EIO, "replace failed")):
            with pytest.raises(OSError, match="replace failed"):
                _quarantine(tmp_path, audit_path, manifest_path)

        after = _snapshot(tmp_path)
        assert after.pop("quarantine.ndjson")
        assert after == before
        assert [(entry["file"], entry["line"]) for entry in _read_lines(_trace(tmp_path))] == [
            ("audit", 4),
            ("audit", 6),
        ]

    def test_failed_trace_write_leaves_both_files_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        before = _snapshot(tmp_path)

        def disk_full(*args: Any, **kwargs: Any) -> None:
            raise OSError(errno.ENOSPC, "disk full")

        monkeypatch.setattr(run_manifest_module, "_append_records", disk_full)

        with pytest.raises(OSError, match="disk full"):
            _quarantine(tmp_path, audit_path, manifest_path)

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("existing", [False, True], ids=["new-trace", "existing-trace"])
    def test_failed_trace_append_is_rolled_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        if existing:
            _trace(tmp_path).write_bytes(b'{"quarantine_version": 1}\n')
        before = _snapshot(tmp_path)

        def partial_append(path: str | Path, records: Any) -> None:
            with open(path, "ab") as file:
                file.write(b'{"quarantine_version": 1, "fi')
            raise OSError(errno.ENOSPC, "disk full")

        monkeypatch.setattr(run_manifest_module, "_append_records", partial_append)

        with pytest.raises(OSError, match="disk full"):
            _quarantine(tmp_path, audit_path, manifest_path)

        # The partial bytes go, and a trace that did not exist is absent again; audit and manifest were never touched.
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_quarantine_log_without_a_final_newline_is_refused_and_nothing_changes(
        self, tmp_path: Path, dry_run: bool
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        # An appended entry would run into this line and make both unreadable.
        _trace(tmp_path).write_bytes(b'{"quarantine_version": 1, "fi')
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError, match=re.escape(str(_trace(tmp_path)))):
            _quarantine(tmp_path, audit_path, manifest_path, dry_run=dry_run)

        assert _snapshot(tmp_path) == before

    def test_an_empty_quarantine_log_is_appended_to(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _trace(tmp_path).write_bytes(b"")

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert len(removed) == len(_read_lines(_trace(tmp_path))) == 3

    def test_trace_is_fsynced_before_either_file_is_modified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        events: list[tuple[str, int]] = []
        real_fsync, real_replace, real_truncate = os.fsync, os.replace, os.truncate

        def spy_fsync(fd: int) -> None:
            try:
                is_trace = os.path.samestat(os.fstat(fd), _trace(tmp_path).stat())
            except FileNotFoundError:
                is_trace = False
            if is_trace:
                events.append(("fsync-trace", _trace(tmp_path).stat().st_size))
            real_fsync(fd)

        def spy_replace(*args: Any, **kwargs: Any) -> None:
            if str(args[1]) == str(audit_path):
                events.append(("replace-audit", 0))
            real_replace(*args, **kwargs)

        def spy_truncate(*args: Any, **kwargs: Any) -> None:
            if str(args[0]) == str(manifest_path):
                events.append(("truncate-manifest", 0))
            real_truncate(*args, **kwargs)

        monkeypatch.setattr(os, "fsync", spy_fsync)
        monkeypatch.setattr(os, "replace", spy_replace)
        monkeypatch.setattr(os, "truncate", spy_truncate)

        _quarantine(tmp_path, audit_path, manifest_path)

        names = [name for name, _ in events]
        assert {"replace-audit", "truncate-manifest"} <= set(names)
        modified = min(names.index("replace-audit"), names.index("truncate-manifest"))
        # Every entry is in the file when it is fsynced, and that happens before either file changes.
        assert ("fsync-trace", _trace(tmp_path).stat().st_size) in events[:modified]

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    @pytest.mark.parametrize("target", ["audit.ndjson", "manifests.ndjson"])
    @pytest.mark.parametrize("spelling", ["same-path", "relative", "symlink"])
    def test_quarantine_path_that_resolves_to_a_log_is_refused_with_a_plain_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str, target: str, dry_run: bool
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        quarantine_path: Path | str = tmp_path / target
        if spelling == "relative":
            monkeypatch.chdir(tmp_path)
            quarantine_path = f"./{target}"
        if spelling == "symlink":
            quarantine_path = tmp_path / "trace-link.ndjson"
            quarantine_path.symlink_to(tmp_path / target)
        before = _snapshot(tmp_path)

        # Trace lines appended to a log would brick it.
        with pytest.raises(ValueError) as excinfo:
            quarantine_damaged_lines(
                audit_path, manifest_path, quarantine_path=quarantine_path, signer=_signer(), dry_run=dry_run
            )

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    def test_a_symlinked_audit_path_repairs_the_real_file_and_stays_a_symlink(self, tmp_path: Path) -> None:
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        real_audit, manifest_path = _sealed_log(real_dir)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        clean = real_audit.read_bytes()
        _insert_line(real_audit, 3, b"[]\n")
        damaged = real_audit.read_bytes()
        link = tmp_path / "audit-link.ndjson"
        link.symlink_to(real_audit)

        removed = _quarantine(tmp_path, link, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [4])
        assert link.is_symlink()
        assert link.resolve() == real_audit.resolve()
        assert not real_audit.is_symlink()
        assert real_audit.read_bytes() == clean
        assert [entry["path"] for entry in _read_lines(_trace(tmp_path))] == [str(link)]
        assert verify_ndjson_log(link, manifest_path, signer=_signer()) == head
        assert sorted(path.name for path in real_dir.iterdir()) == ["audit.ndjson", "manifests.ndjson"]

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_recovery_waits_for_a_sealer_holding_the_lock(self, tmp_path: Path, dry_run: bool) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")

        with ThreadPoolExecutor(max_workers=1) as pool:
            fd = os.open(manifest_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                future = pool.submit(_quarantine, tmp_path, audit_path, manifest_path, dry_run=dry_run)
                done, _ = wait([future], timeout=0.3)
            finally:
                os.close(fd)

            assert not done
            assert len(future.result(timeout=10)) == 1

    def test_dry_run_does_not_wait_for_a_reader_holding_a_shared_lock(self, tmp_path: Path) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")

        with ThreadPoolExecutor(max_workers=1) as pool:
            fd = os.open(manifest_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
                future = pool.submit(_quarantine, tmp_path, audit_path, manifest_path, dry_run=True)
                done, _ = wait([future], timeout=5)
            finally:
                os.close(fd)

            assert done
            assert len(future.result()) == 1

    def test_recovery_holds_an_exclusive_lock_on_the_manifest_log_while_it_repairs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        probes: list[tuple[str, bool]] = []
        real_replace = os.replace

        class _LockProbeSigner(HmacSha256Signer):
            def sign(self, payload: bytes) -> str:
                # Verifying the manifest log and signing the trace entries both sign.
                probes.append(("sign", _lock_refused(manifest_path)))
                return super().sign(payload)

        def spy_append(*args: Any, **kwargs: Any) -> None:
            probes.append(("append", _lock_refused(manifest_path)))
            _append_records(*args, **kwargs)

        def spy_replace(*args: Any, **kwargs: Any) -> None:
            probes.append(("replace", _lock_refused(manifest_path)))
            real_replace(*args, **kwargs)

        monkeypatch.setattr(run_manifest_module, "_append_records", spy_append)
        monkeypatch.setattr(os, "replace", spy_replace)

        removed = _quarantine(tmp_path, audit_path, manifest_path, signer=_LockProbeSigner(_KEY, "key-1"))

        assert len(removed) == 3
        assert {stage for stage, _ in probes} == {"sign", "append", "replace"}
        assert all(refused for _, refused in probes)
        fd = os.open(manifest_path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


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
