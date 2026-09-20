"""Tests for run_manifest and the NDJSON sealing built on it."""

from __future__ import annotations

import base64
import copy
import dataclasses
import errno
import hashlib
import hmac
import json
import os
import pickle  # nosec
import re
import stat
import sys
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from mloda.steward import verified_context
from mloda.user import ParallelizationMode

import mloda.enterprise.extenders.audit as audit_package
import mloda.enterprise.extenders.audit.audit_extender as audit_extender_module
import mloda.enterprise.extenders.audit.run_manifest as run_manifest_module
from mloda.enterprise.extenders.audit import (
    AuditExtender,
    HmacSha256Signer,
    IdentityRequiredError,
    KeyAlreadyCurrentError,
    LogCoverage,
    ManifestSigner,
    ManifestVerificationError,
    NdjsonAuditSink,
    QuarantinedLine,
    RunAlreadySealedError,
    RunNotPendingError,
    manifest_hash,
    quarantine_damaged_lines,
    quarantine_from_rotation_entry,
    rotate_manifest_key,
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

_EXPECTED_ROTATION_KEYS = {"manifest_version", "kind", "rotated_at", "previous_manifest_hash", "signature"}

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

_DAMAGED_LINES: dict[str, bytes] = {
    "blank": b"\n",
    "bad-utf8": b"\xff\xfe\n",
    "truncated": b'{"run_id": "run-d"\n',
    "not-an-object": b"[]\n",
    "duplicate-key": b'{"run_id": "run-d", "run_id": "run-e"}\n',
    "deep-nesting": b"[" * 100000 + b"\n",
}

# The manifest log's writers and the run_id of each line they append (None for a rotation entry).
_WRITERS = ["seal", "rotate"]
_PENDING_RUN_IDS: dict[str, list[str | None]] = {"seal": ["run-d", "run-e", "run-f"], "rotate": [None]}

# As a `_rotation_entry` change, drops the key.
_MISSING: Any = object()

_current_key_required = pytest.mark.parametrize(
    "call", [verify_ndjson_log, verify_ndjson_log_coverage, seal_ndjson_runs], ids=["verify", "coverage", "seal"]
)


def _signer(key: bytes = _KEY, key_id: str = "key-1") -> HmacSha256Signer:
    return HmacSha256Signer(key, key_id)


def _record(run_id: str | None = "run-1", second: int = 0, *, tenant_id: str | None = "tenant-1") -> dict[str, Any]:
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
    """The bytes NdjsonAuditSink writes, without the newline."""
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
    path.write_text("".join(json.dumps(obj, sort_keys=True) + "\n" for obj in objects), encoding="utf-8")


def _append_line(path: Path, line: str | bytes) -> None:
    data = line.encode("utf-8") if isinstance(line, str) else line
    with open(path, "ab") as file:
        file.write(data + b"\n")


def _resigned(manifest: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    changed = {**_unsigned(manifest), **changes}
    changed["signature"] = {**manifest["signature"], "value": _signer().sign(_canonical(changed))}
    return changed


def _rotation_entry(signer: ManifestSigner, previous_manifest_hash: str | None, **changes: Any) -> dict[str, Any]:
    """A rotation entry signed by any `signer`, for forgeries; `changes` apply before signing."""
    entry: dict[str, Any] = {
        "manifest_version": 1,
        "kind": "key_rotation",
        "rotated_at": "2026-01-01T00:00:00.000000Z",
        "previous_manifest_hash": previous_manifest_hash,
        **changes,
    }
    entry = {key: value for key, value in entry.items() if value is not _MISSING}
    entry["signature"] = {
        "algorithm": signer.algorithm,
        "key_id": signer.key_id,
        "value": signer.sign(_canonical(entry)),
    }
    return entry


def _build_log(directory: Path, *steps: tuple[str, ManifestSigner]) -> tuple[Path, Path]:
    """Hand-build a manifest log: ("seal", signer) seals a new run-N, ("rotate", signer) appends a rotation entry."""
    audit_path = directory / "audit.ndjson"
    manifest_path = directory / "manifests.ndjson"
    head: str | None = None
    for number, (action, signer) in enumerate(steps, start=1):
        if action == "seal":
            record = _record(f"run-{number}", number)
            _write_records(audit_path, [record])
            line = seal_run([record], run_id=f"run-{number}", signer=signer, previous_manifest_hash=head)
        else:
            line = _rotation_entry(signer, head)
        _write_records(manifest_path, [line])
        head = manifest_hash(line)
    return audit_path, manifest_path


def _under_key(current: str, signer: str) -> str:
    """A `match` pattern for the log-under-another-key error."""
    return "^" + re.escape(f"manifest log is under key '{current}', not the signer's '{signer}'") + "$"


def _assert_names(excinfo: pytest.ExceptionInfo[ManifestVerificationError], *names: str) -> None:
    message = str(excinfo.value)
    assert all(name in message for name in names), message


def _key_2_entry(head: str, **changes: Any) -> dict[str, Any]:
    return _rotation_entry(_signer(_OTHER_KEY, "key-2"), head, **changes)


# Rotation entries that must not verify, each built from the head it should chain onto.
_FORGED_ENTRIES: dict[str, Callable[[str], dict[str, Any]]] = {
    "wrong-key-material": lambda head: _rotation_entry(_signer(b"x" * 32, "key-2"), head),
    "unknown-key-id": lambda head: _rotation_entry(_signer(b"u" * 32, "key-unknown"), head),
    "wrong-chain": lambda head: _key_2_entry("0" * 64),
    "tampered-rotated-at": lambda head: {**_key_2_entry(head), "rotated_at": "2030-01-01T00:00:00.000000Z"},
    "wrong-algorithm": lambda head: {
        **_key_2_entry(head),
        "signature": {**_key_2_entry(head)["signature"], "algorithm": "OTHER"},
    },
    "wrong-version": lambda head: _key_2_entry(head, manifest_version=2),
    "extra-key": lambda head: _key_2_entry(head, run_id="run-x"),
    "missing-key": lambda head: _key_2_entry(head, rotated_at=_MISSING),
    "unknown-kind": lambda head: _key_2_entry(head, kind="seal"),
    "null-kind": lambda head: _key_2_entry(head, kind=None),
    "unhashable-kind": lambda head: _key_2_entry(head, kind=["key_rotation"]),
    "manifest-with-kind": lambda head: _key_2_entry(
        head,
        run_id="run-1",
        sealed_at="2026-01-01T00:00:00.000000Z",
        hash_algorithm="sha256",
        record_count=0,
        record_hashes=[],
        compliant=True,
    ),
}

_BAD_SIGNATURE = "signature does not match the manifest"
_BAD_SHAPE = "must be a 'key_rotation' entry with exactly"
# The `match` fragment of the error each forged entry must raise, so none passes for the wrong reason.
_FORGED_ENTRY_REASONS: dict[str, str] = {
    "wrong-key-material": _BAD_SIGNATURE,
    "unknown-key-id": "signature key_id 'key-unknown' matches no known key",
    "wrong-chain": "manifest chain is broken at a key rotation entry",
    "tampered-rotated-at": _BAD_SIGNATURE,
    "wrong-algorithm": "signature algorithm 'OTHER' is not the signer's",
    "wrong-version": "unsupported manifest_version 2",
    "extra-key": _BAD_SHAPE,
    "missing-key": _BAD_SHAPE,
    "unknown-kind": _BAD_SHAPE,
    "null-kind": _BAD_SHAPE,
    "unhashable-kind": _BAD_SHAPE,
    "manifest-with-kind": _BAD_SHAPE,
}


def _sealed_log(directory: Path) -> tuple[Path, Path]:
    """Three runs plus one record without a run_id, all sealed."""
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


def _rotate(manifest_path: Path, signer: ManifestSigner, *previous_signers: ManifestSigner) -> dict[str, Any]:
    """Rotate the log to `signer`, anchored on its current head."""
    head = manifest_hash(_read_lines(manifest_path)[-1])
    return rotate_manifest_key(manifest_path, signer=signer, previous_signers=previous_signers, expected_head=head)


def _rotated_three_key_log(
    directory: Path, *, seal_between: bool = True
) -> tuple[Path, Path, HmacSha256Signer, HmacSha256Signer, HmacSha256Signer]:
    """One run each sealed by three keys in turn, with two rotations; `seal_between=False` skips the key-2 run."""
    audit_path = directory / "audit.ndjson"
    manifest_path = directory / "manifests.ndjson"
    key_1, key_2, key_3 = _signer(key_id="key-1"), _signer(_OTHER_KEY, "key-2"), _signer(b"t" * 32, "key-3")
    _write_records(audit_path, [_record("run-a", 1)])
    seal_ndjson_runs(audit_path, manifest_path, signer=key_1)
    _rotate(manifest_path, key_2, key_1)
    if seal_between:
        _write_records(audit_path, [_record("run-b", 2)])
        seal_ndjson_runs(audit_path, manifest_path, signer=key_2, previous_signers=[key_1])
    _rotate(manifest_path, key_3, key_1, key_2)
    _write_records(audit_path, [_record("run-c", 3)])
    seal_ndjson_runs(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])
    return audit_path, manifest_path, key_1, key_2, key_3


def _pending_write(
    directory: Path, writer: str
) -> tuple[Path, Callable[..., list[dict[str, Any]]], Callable[[], str | None]]:
    """A sealed log with one pending write ("seal": runs d, e, f; "rotate": key-1 to key-2). Returns the manifest path,
    `write(signer_type)` (the appended lines; the type lets a test spy on signing) and `verify()` (the new head)."""
    audit_path, manifest_path = _sealed_log(directory)
    head = manifest_hash(_read_lines(manifest_path)[-1])
    key, key_id = (_KEY, "key-1") if writer == "seal" else (_OTHER_KEY, "key-2")
    if writer == "seal":
        _write_records(
            audit_path, [_record(f"run-{name}", second) for name, second in zip("def", (7, 8, 9), strict=True)]
        )

    def write(signer_type: type[HmacSha256Signer] = HmacSha256Signer) -> list[dict[str, Any]]:
        signer = signer_type(key, key_id)
        if writer == "seal":
            return seal_ndjson_runs(audit_path, manifest_path, signer=signer, expected_head=head)
        old_signer = signer_type(_KEY, "key-1")
        return [rotate_manifest_key(manifest_path, signer=signer, previous_signers=[old_signer], expected_head=head)]

    def verify() -> str | None:
        previous = [] if writer == "seal" else [_signer()]
        return verify_ndjson_log(audit_path, manifest_path, signer=_signer(key, key_id), previous_signers=previous)

    return manifest_path, write, verify


def _truncated_log(directory: Path) -> tuple[Path, Path, str]:
    """Seal runs a, b, c, cut the log to its first line, edit a run-b record; returns the head before the cut."""
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
    path.write_bytes(path.read_bytes() + tail)


def _insert_line(path: Path, index: int, line: bytes) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    lines.insert(index, line)
    path.write_bytes(b"".join(lines))


def _torn_manifest_log(directory: Path) -> tuple[Path, Path, str]:
    """A manifest log ending in half a manifest line (line 4); returns the head before the damage."""
    audit_path, manifest_path = _sealed_log(directory)
    head = manifest_hash(_read_lines(manifest_path)[-1])
    last = manifest_path.read_bytes().splitlines(keepends=True)[-1]
    _torn(manifest_path, last[: len(last) // 2])
    return audit_path, manifest_path, head


def _torn_rotation_log(directory: Path) -> tuple[Path, Path, str]:
    """A manifest log ending in half a key-2 rotation entry; returns the head before the damage."""
    audit_path, manifest_path = _sealed_log(directory)
    head = manifest_hash(_read_lines(manifest_path)[-1])
    entry = _canonical(_rotation_entry(_signer(_OTHER_KEY, "key-2"), head)) + b"\n"
    _torn(manifest_path, entry[: len(entry) // 2])
    return audit_path, manifest_path, head


def _damaged_log(directory: Path) -> tuple[Path, Path, str]:
    """A torn manifest log plus an undecodable audit line (line 4) and a torn last audit line (line 8)."""
    audit_path, manifest_path, head = _torn_manifest_log(directory)
    _insert_line(audit_path, 3, b"\xff\xfe\n")
    _torn(audit_path, b'{"run_id": "run-d", "tenant')
    return audit_path, manifest_path, head


def _log_cut_mid_write(directory: Path) -> tuple[Path, Path, list[str]]:
    """Seal run-a, then run-b and run-c in one write cut inside run-c's line; returns the three heads."""
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


def _log_with_rotation_entry(directory: Path) -> tuple[Path, Path, str]:
    """Two key-1 seals, a key-2 rotation entry (line 3) and a key-2 seal (line 4); returns the head before the entry."""
    key_1, key_2 = _signer(), _signer(_OTHER_KEY, "key-2")
    audit_path, manifest_path = _build_log(
        directory, ("seal", key_1), ("seal", key_1), ("rotate", key_2), ("seal", key_2)
    )
    return audit_path, manifest_path, manifest_hash(_read_lines(manifest_path)[1])


def _trace(directory: Path) -> Path:
    return directory / "quarantine.ndjson"


def _quarantine(
    directory: Path,
    audit_path: Path,
    manifest_path: Path,
    *,
    signer: ManifestSigner | None = None,
    previous_signers: Iterable[ManifestSigner] = (),
    expected_head: str | None = None,
    dry_run: bool = False,
) -> list[QuarantinedLine]:
    return quarantine_damaged_lines(
        audit_path,
        manifest_path,
        quarantine_path=_trace(directory),
        signer=signer or _signer(),
        previous_signers=previous_signers,
        expected_head=expected_head,
        dry_run=dry_run,
    )


def _quarantine_from_entry(
    directory: Path,
    manifest_path: Path,
    *,
    expected_head: str,
    signer: ManifestSigner | None = None,
    previous_signers: Iterable[ManifestSigner] = (),
    dry_run: bool = False,
) -> list[QuarantinedLine]:
    return quarantine_from_rotation_entry(
        manifest_path,
        quarantine_path=_trace(directory),
        signer=signer or _signer(),
        previous_signers=previous_signers,
        expected_head=expected_head,
        dry_run=dry_run,
    )


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def _aliased(tmp_path: Path, target: Path, spelling: str) -> Path:
    """`target` itself, or a symlink to it: two spellings that resolve to the same file."""
    if spelling == "same-path":
        return target
    link = tmp_path / "aliased.ndjson"
    link.symlink_to(target)
    return link


def _spans(file: str, data: bytes, numbers: Iterable[int]) -> list[tuple[str, int, int, int, str]]:
    """What a recovery reports for the 1-based lines `numbers` of `data`."""
    lines = data.splitlines(keepends=True)
    return [(file, n, len(b"".join(lines[: n - 1])), len(lines[n - 1]), _sha256(lines[n - 1])) for n in numbers]


def _summary(removed: Iterable[QuarantinedLine]) -> list[tuple[str, int, int, int, str]]:
    return [(item.file, item.line, item.offset, item.length, item.sha256) for item in removed]


def _assert_raises_and_unchanged(
    directory: Path, call: Callable[[], object]
) -> pytest.ExceptionInfo[ManifestVerificationError]:
    """`call` raises ManifestVerificationError and changes nothing in `directory`."""
    before = _snapshot(directory)

    with pytest.raises(ManifestVerificationError) as excinfo:
        call()

    assert _snapshot(directory) == before
    return excinfo


def _assert_refused(
    directory: Path,
    audit_path: Path,
    manifest_path: Path,
    *,
    signer: ManifestSigner | None = None,
    expected_head: str | None = None,
    dry_run: bool = False,
) -> None:
    """Recovery raises ManifestVerificationError and changes nothing."""
    _assert_raises_and_unchanged(
        directory,
        lambda: _quarantine(
            directory, audit_path, manifest_path, signer=signer, expected_head=expected_head, dry_run=dry_run
        ),
    )


def _assert_traced(
    directory: Path,
    manifest_path: Path,
    manifest_before: bytes,
    removed: list[QuarantinedLine],
    signer: ManifestSigner | None = None,
) -> None:
    """The trace holds one entry per dropped line: its report, its raw bytes and a signature by `signer`."""
    signer = signer or _signer()
    lines = manifest_before.splitlines(keepends=True)
    entries = _read_lines(_trace(directory))
    assert len(entries) == len(removed)
    for entry, item in zip(entries, removed, strict=True):
        raw = base64.b64decode(entry["raw_base64"], validate=True)
        assert set(entry) == _EXPECTED_QUARANTINE_KEYS
        assert entry["quarantine_version"] == 1
        assert entry["path"] == str(manifest_path)
        assert {field: entry[field] for field in dataclasses.asdict(item)} == dataclasses.asdict(item)
        assert raw == lines[item.line - 1]
        assert entry["sha256"] == _sha256(raw)
        signature = entry["signature"]
        assert (signature["algorithm"], signature["key_id"]) == (signer.algorithm, signer.key_id)
        assert signature["value"] == signer.sign(_canonical(_unsigned(entry)))


def _lock_refused(path: Path) -> bool:
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
    raise OSError(errno.ENOLCK, "no locks")


class _PrefixSigner:
    """A signer that inherits from nothing, so only the protocol is relied on."""

    algorithm = "TEST-SHA256"
    key_id = "test-key"

    def sign(self, payload: bytes) -> str:
        return _sha256(b"prefix:" + payload)

    def verify(self, payload: bytes, signature: str) -> bool:
        try:
            return hmac.compare_digest(self.sign(payload).encode("ascii"), signature.encode("ascii"))
        except (AttributeError, UnicodeEncodeError):
            return False


class TestRunManifestPublicApi:
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
            "quarantine_from_rotation_entry",
            "IdentityRequiredError",
            "KeyAlreadyCurrentError",
            "rotate_manifest_key",
        } <= set(audit_package.__all__)

    def test_run_not_pending_error_is_a_value_error_but_not_a_verification_error(self) -> None:
        assert issubclass(RunNotPendingError, ValueError)
        assert not issubclass(RunNotPendingError, ManifestVerificationError)

    def test_key_already_current_error_is_a_value_error_but_not_a_verification_error(self) -> None:
        assert issubclass(KeyAlreadyCurrentError, ValueError)
        assert not issubclass(KeyAlreadyCurrentError, ManifestVerificationError)

    def test_identity_required_error_is_a_runtime_error_but_not_a_value_error(self) -> None:
        assert issubclass(IdentityRequiredError, RuntimeError)
        assert not issubclass(IdentityRequiredError, ValueError)

    def test_run_already_sealed_error_is_a_run_not_pending_error(self) -> None:
        assert issubclass(RunAlreadySealedError, RunNotPendingError)

    def test_append_records_and_canonical_json_come_from_one_shared_private_records_module(self) -> None:
        import mloda.enterprise.extenders.audit._records as records_module

        # getattr: run_manifest and audit_extender import these, they do not define them.
        assert getattr(run_manifest_module, "_append_records") is records_module._append_records
        assert getattr(run_manifest_module, "_canonical_json") is records_module._canonical_json
        assert getattr(audit_extender_module, "_append_records") is records_module._append_records
        assert getattr(audit_extender_module, "_canonical_json") is records_module._canonical_json
        assert records_module._is_blank("") is True
        assert records_module._is_blank("value") is False


class TestHmacSha256Signer:
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

    @pytest.mark.parametrize("field", ["record_count", "manifest_version"])
    @pytest.mark.parametrize("value", [True, 1.0], ids=["bool", "float"])
    def test_correctly_signed_manifest_with_a_non_int_number_fails(self, field: str, value: Any) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
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


class TestSealNdjsonRuns:
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

        assert not isinstance(excinfo.value, RunNotPendingError)
        assert not manifest_path.exists()

    @pytest.mark.parametrize("forged_run_id", ["run-d", "run-never-written"])
    def test_forged_unsigned_manifest_line_stops_sealing(self, tmp_path: Path, forged_run_id: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
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
                # Fails on run-f, which only sealing meets.
                if json.loads(payload).get("run_id") == "run-f":
                    raise RuntimeError("signing boom")
                return super().sign(payload)

        with pytest.raises(RuntimeError, match="signing boom"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_FailingSigner(_KEY, "key-1"), expected_head=head)

        assert manifest_path.read_bytes() == before
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head)
        assert [manifest["run_id"] for manifest in manifests] == ["run-d", "run-e", "run-f"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    # Sealing and rotation share one append path, so its tests run against both (`_pending_write`).
    @pytest.mark.parametrize("writer", _WRITERS)
    def test_failed_append_is_rolled_back_and_a_retry_appends_every_pending_line(
        self, tmp_path: Path, writer: str
    ) -> None:
        manifest_path, write, verify = _pending_write(tmp_path, writer)
        before = manifest_path.read_bytes()
        real_write = os.write

        def disk_is_full(fd: int, data: bytes | memoryview) -> int:
            # Every appended line carries the chain link.
            if b"previous_manifest_hash" in bytes(data):
                raise OSError(errno.ENOSPC, "disk full")
            return real_write(fd, data)

        with patch("os.write", side_effect=disk_is_full):
            with pytest.raises(OSError):
                write()

        assert manifest_path.read_bytes() == before
        assert [line.get("run_id") for line in write()] == _PENDING_RUN_IDS[writer]
        verify()

    @pytest.mark.parametrize("writer", _WRITERS)
    def test_failed_append_with_a_genuine_short_write_is_rolled_back_and_a_retry_appends_every_pending_line(
        self, tmp_path: Path, writer: str
    ) -> None:
        """Unlike the raise-based case above, os.write here really writes a truncated prefix and returns the
        short count: bytes land on disk before _append_records raises, so only the truncate saves the log."""
        manifest_path, write, verify = _pending_write(tmp_path, writer)
        before = manifest_path.read_bytes()
        real_write = os.write

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                write()

        assert manifest_path.read_bytes() == before
        assert [line.get("run_id") for line in write()] == _PENDING_RUN_IDS[writer]
        verify()

    @pytest.mark.parametrize("fcntl_available", [True, False], ids=["fcntl", "no-fcntl"])
    def test_failed_first_append_leaves_the_new_manifest_log_as_an_empty_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fcntl_available: bool
    ) -> None:
        if fcntl_available:
            pytest.importorskip("fcntl")
        else:
            monkeypatch.setitem(sys.modules, "fcntl", None)
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])
        real_write = os.write

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert manifest_path.read_bytes() == b""

    @pytest.mark.parametrize("writer", _WRITERS)
    def test_failed_append_fsyncs_the_manifest_log_and_its_directory_after_the_rollback_truncate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
    ) -> None:
        """The rollback truncate must be durable: a crash after it must not resurrect lines reported as failed."""
        manifest_path, write, _ = _pending_write(tmp_path, writer)
        real_write = os.write
        real_truncate = os.truncate
        real_fsync = os.fsync
        events: list[str] = []

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        def spy_truncate(path: str | Path, length: int) -> None:
            events.append("truncate")
            real_truncate(path, length)

        def spy_fsync(fd: int) -> None:
            try:
                if os.path.samestat(os.fstat(fd), manifest_path.stat()):
                    events.append("fsync-manifest")
            except FileNotFoundError:
                pass
            try:
                if os.path.samestat(os.fstat(fd), manifest_path.parent.stat()):
                    events.append("fsync-manifest-dir")
            except FileNotFoundError:
                pass
            real_fsync(fd)

        monkeypatch.setattr(os, "truncate", spy_truncate)
        monkeypatch.setattr(os, "fsync", spy_fsync)

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                write()

        assert "truncate" in events
        assert "fsync-manifest" in events
        assert "fsync-manifest-dir" in events
        assert events.index("truncate") < events.index("fsync-manifest")
        assert events.index("truncate") < events.index("fsync-manifest-dir")

    @pytest.mark.parametrize("writer", _WRITERS)
    def test_append_fsyncs_the_manifest_log_and_its_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
    ) -> None:
        manifest_path, write, _ = _pending_write(tmp_path, writer)
        fsynced: list[Path] = []
        real_fsync = run_manifest_module._fsync

        def spy(path: str | Path) -> None:
            fsynced.append(Path(path))
            real_fsync(path)

        monkeypatch.setattr(run_manifest_module, "_fsync", spy)

        write()

        assert manifest_path in fsynced
        assert manifest_path.parent in fsynced

    @pytest.mark.parametrize("writer", _WRITERS)
    def test_append_fsync_failure_rolls_back_the_new_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
    ) -> None:
        manifest_path, write, _ = _pending_write(tmp_path, writer)
        before = _snapshot(tmp_path)
        real_fsync = run_manifest_module._fsync

        def fsync_boom(path: str | Path) -> None:
            if Path(path) == manifest_path:
                raise OSError(errno.EIO, "fsync failed")
            real_fsync(path)

        monkeypatch.setattr(run_manifest_module, "_fsync", fsync_boom)

        with pytest.raises(OSError, match="fsync failed"):
            write()

        assert _snapshot(tmp_path) == before

    def test_audit_file_is_fsynced_before_the_manifest_log_is_appended(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        events: list[str] = []
        real_fsync = os.fsync

        def spy_fsync(fd: int) -> None:
            try:
                is_audit = os.path.samestat(os.fstat(fd), audit_path.stat())
            except FileNotFoundError:
                is_audit = False
            if is_audit:
                events.append("fsync-audit")
            real_fsync(fd)

        def spy_append(*args: Any, **kwargs: Any) -> None:
            events.append("append")
            _append_records(*args, **kwargs)

        monkeypatch.setattr(os, "fsync", spy_fsync)
        monkeypatch.setattr(run_manifest_module, "_append_records", spy_append)

        seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert "fsync-audit" in events
        assert events.index("fsync-audit") < events.index("append")

    @pytest.mark.parametrize("file_name", ["manifests.ndjson", "audit.ndjson"])
    def test_unterminated_final_line_fails_and_leaves_both_files_untouched(
        self, tmp_path: Path, file_name: str
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
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

    @pytest.mark.parametrize("writer", _WRITERS)
    def test_appending_holds_an_exclusive_lock_on_the_manifest_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        manifest_path, write, _ = _pending_write(tmp_path, writer)
        refused: list[bool] = []
        appends: list[bool] = []

        def probe() -> bool:
            # A shared request is refused only by an exclusive holder.
            fd = os.open(manifest_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                return False
            except BlockingIOError:
                return True
            finally:
                os.close(fd)

        class _LockProbeSigner(HmacSha256Signer):
            def sign(self, payload: bytes) -> str:
                # Verifying signs too.
                refused.append(probe())
                return super().sign(payload)

        def spy(*args: Any, **kwargs: Any) -> None:
            refused.append(probe())
            appends.append(True)
            _append_records(*args, **kwargs)

        monkeypatch.setattr(run_manifest_module, "_append_records", spy)

        appended = write(_LockProbeSigner)

        assert [line.get("run_id") for line in appended] == _PENDING_RUN_IDS[writer]
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

    @pytest.mark.parametrize("writer", _WRITERS)
    def test_appending_fails_when_the_exclusive_lock_is_unsupported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        manifest_path, write, _ = _pending_write(tmp_path, writer)
        before = manifest_path.read_bytes()
        monkeypatch.setattr(fcntl, "flock", _flock_unsupported)

        with pytest.raises(OSError) as excinfo:
            write()

        assert excinfo.value.errno == errno.ENOLCK
        assert manifest_path.read_bytes() == before

    def test_missing_audit_file_still_fails_sealing(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        audit_path.unlink()
        before = manifest_path.read_bytes()

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

    def test_seal_after_rotation_chains_onto_the_rotation_entry_and_signs_with_the_new_key(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        head = manifest_hash(_rotate(manifest_path, new_signer, old_signer))
        _write_records(audit_path, [_record("run-d", 7)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert [manifest["run_id"] for manifest in manifests] == ["run-d"]
        assert manifests[0]["previous_manifest_hash"] == head
        assert manifests[0]["signature"]["key_id"] == "key-2"

    @pytest.mark.parametrize(
        "previous_signers",
        [[_signer()], [_signer(key_id="key-3"), _signer(_OTHER_KEY, "key-3")]],
        ids=["signer-and-previous", "two-previous"],
    )
    def test_previous_signers_with_a_duplicate_key_id_is_a_value_error(
        self, tmp_path: Path, previous_signers: list[ManifestSigner]
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = manifest_path.read_bytes()

        with pytest.raises(ValueError) as excinfo:
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(_OTHER_KEY), previous_signers=previous_signers)

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert manifest_path.read_bytes() == before


class TestRotateManifestKey:
    """Lock, fsync and rollback are covered in TestSealNdjsonRuns."""

    def test_entry_has_exactly_the_expected_keys_kind_and_version(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)

        entry = _rotate(manifest_path, _signer(_OTHER_KEY, "key-2"), _signer())

        assert set(entry) == _EXPECTED_ROTATION_KEYS
        assert entry["manifest_version"] == 1
        assert entry["kind"] == "key_rotation"

    def test_rotated_at_is_rfc3339_utc(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)

        rotated_at = _rotate(manifest_path, _signer(_OTHER_KEY, "key-2"), _signer())["rotated_at"]

        assert rotated_at.endswith("Z")
        datetime.fromisoformat(rotated_at.removesuffix("Z"))

    def test_entry_is_signed_by_the_new_key_over_the_unsigned_entry(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        new_signer = _signer(_OTHER_KEY, "key-2")

        entry = _rotate(manifest_path, new_signer, _signer())

        signature = entry["signature"]
        assert set(signature) == {"algorithm", "key_id", "value"}
        assert (signature["algorithm"], signature["key_id"]) == (new_signer.algorithm, "key-2")
        assert signature["value"] == hmac.new(_OTHER_KEY, _canonical(_unsigned(entry)), hashlib.sha256).hexdigest()

    def test_entry_chains_onto_the_head_and_is_the_only_line_appended(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = _read_lines(manifest_path)

        entry = _rotate(manifest_path, _signer(_OTHER_KEY, "key-2"), _signer())

        assert entry["previous_manifest_hash"] == manifest_hash(before[-1])
        assert _read_lines(manifest_path) == [*before, entry]

    @pytest.mark.parametrize("log_bytes", [None, b""], ids=["missing", "empty"])
    def test_a_log_without_manifests_raises_value_error_and_creates_no_file(
        self, tmp_path: Path, log_bytes: bytes | None
    ) -> None:
        manifest_path = tmp_path / "manifests.ndjson"
        if log_bytes is not None:
            manifest_path.write_bytes(log_bytes)

        with pytest.raises(ValueError) as excinfo:
            rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()], expected_head=None
            )

        assert not isinstance(excinfo.value, (ManifestVerificationError, KeyAlreadyCurrentError))
        assert manifest_path.exists() == (log_bytes is not None)

    @pytest.mark.parametrize("rotated_before", [False, True], ids=["fresh-log", "retry-after-rotation"])
    def test_a_log_already_under_the_signer_raises_key_already_current_error_and_writes_nothing(
        self, tmp_path: Path, rotated_before: bool
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        signer, previous_signers = (new_signer, [old_signer]) if rotated_before else (old_signer, [])
        if rotated_before:
            _rotate(manifest_path, new_signer, old_signer)
        before = manifest_path.read_bytes()
        head = manifest_hash(_read_lines(manifest_path)[-1])

        with pytest.raises(KeyAlreadyCurrentError) as excinfo:
            rotate_manifest_key(manifest_path, signer=signer, previous_signers=previous_signers, expected_head=head)

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("anchor", ["pre-rotation-head", "none", "post-rotation-head"])
    def test_a_retry_after_a_rotation_fails_on_a_stale_head_and_is_current_otherwise(
        self, tmp_path: Path, anchor: str
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        pre_rotation_head = manifest_hash(_read_lines(manifest_path)[-1])
        _rotate(manifest_path, new_signer, old_signer)
        heads = {
            "pre-rotation-head": pre_rotation_head,
            "none": None,
            "post-rotation-head": manifest_hash(_read_lines(manifest_path)[-1]),
        }
        before = manifest_path.read_bytes()
        stale = anchor == "pre-rotation-head"

        with pytest.raises(
            ManifestVerificationError if stale else KeyAlreadyCurrentError,
            match="is not the expected head" if stale else "already under key",
        ) as excinfo:
            rotate_manifest_key(
                manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=heads[anchor]
            )

        assert isinstance(excinfo.value, ManifestVerificationError) is stale
        assert isinstance(excinfo.value, KeyAlreadyCurrentError) is not stale
        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("seal_between", [True, False], ids=["seal-between", "back-to-back"])
    @pytest.mark.parametrize("retired", [0, 1], ids=["oldest-key", "middle-key"])
    def test_rotating_back_to_a_retired_key_raises_manifest_verification_error_and_writes_nothing(
        self, tmp_path: Path, retired: int, seal_between: bool
    ) -> None:
        _, manifest_path, *keys = _rotated_three_key_log(tmp_path, seal_between=seal_between)
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

    @pytest.mark.parametrize(
        "previous_signers",
        [[_signer()], [_signer(key_id="key-3"), _signer(_OTHER_KEY, "key-3")]],
        ids=["signer-and-previous", "two-previous"],
    )
    def test_previous_signers_with_a_duplicate_key_id_is_a_value_error(
        self, tmp_path: Path, previous_signers: list[ManifestSigner]
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = manifest_path.read_bytes()

        with pytest.raises(ValueError) as excinfo:
            rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY), previous_signers=previous_signers, expected_head=None
            )

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("anchor", ["unrelated", "older-manifest"])
    def test_an_expected_head_that_is_not_the_head_raises_and_writes_nothing(self, tmp_path: Path, anchor: str) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        expected_head = "0" * 64 if anchor == "unrelated" else manifest_hash(_read_lines(manifest_path)[0])
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match="head"):
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=expected_head,
            )

        assert manifest_path.read_bytes() == before

    def test_expected_head_none_means_no_anchor_and_is_accepted(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])

        entry = rotate_manifest_key(
            manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()], expected_head=None
        )

        assert entry["previous_manifest_hash"] == head

    def test_omitting_expected_head_is_a_type_error(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = manifest_path.read_bytes()

        with pytest.raises(TypeError):
            rotate_manifest_key(  # type: ignore[call-arg]
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()]
            )

        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("damage", ["torn-tail", "edited-manifest"])
    def test_a_torn_or_tampered_log_raises_manifest_verification_error_and_writes_nothing(
        self, tmp_path: Path, damage: str
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        if damage == "torn-tail":
            _torn(manifest_path, b'{"run_id": "run-d"')
        else:
            manifests = _read_lines(manifest_path)
            manifests[0]["compliant"] = not manifests[0]["compliant"]
            _rewrite_lines(manifest_path, manifests)
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError):
            rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()], expected_head=None
            )

        assert manifest_path.read_bytes() == before

    def test_rotating_without_the_retired_signer_raises_manifest_verification_error_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError):
            rotate_manifest_key(manifest_path, signer=_signer(_OTHER_KEY, "key-2"), expected_head=None)

        assert manifest_path.read_bytes() == before

    def test_the_existing_log_is_verified_only_once(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        verified: list[bytes] = []

        class _CountingSigner(HmacSha256Signer):
            def verify(self, payload: bytes, signature: str) -> bool:
                verified.append(payload)
                return super().verify(payload, signature)

        _rotate(manifest_path, _signer(_OTHER_KEY, "key-2"), _CountingSigner(_KEY, "key-1"))

        assert len(verified) == len(_read_lines(manifest_path)) - 1

    def test_rotating_works_where_fcntl_is_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "fcntl", None)
        _, write, verify = _pending_write(tmp_path, "rotate")

        (entry,) = write()

        assert verify() == manifest_hash(entry)


class TestVerifyNdjsonLog:
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

    def test_previous_signers_verifies_a_log_sealed_before_rotation(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        expected_head = manifest_hash(_rotate(manifest_path, new_signer, old_signer))

        head = verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert head == expected_head

    def test_mixed_key_log_verifies_after_a_further_seal_with_the_new_key(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        _rotate(manifest_path, new_signer, old_signer)
        _write_records(audit_path, [_record("run-d", 7)])
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        expected_head = manifest_hash(manifests[-1])

        head = verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert head == expected_head

    def test_rotated_log_fails_without_the_old_signer_in_previous_signers(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        _rotate(manifest_path, new_signer, old_signer)
        _write_records(audit_path, [_record("run-d", 7)])
        seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer)

    def test_a_previous_key_manifest_appended_after_the_current_key_is_rejected(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        _write_records(audit_path, [_record("run-a", 1)])
        seal_ndjson_runs(audit_path, manifest_path, signer=old_signer)
        _rotate(manifest_path, new_signer, old_signer)
        _write_records(audit_path, [_record("run-b", 2)])
        seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        head = manifest_hash(_read_lines(manifest_path)[-1])
        record = _record("run-c", 3)
        _write_records(audit_path, [record])
        forged = seal_run([record], run_id="run-c", signer=old_signer, previous_manifest_hash=head)
        _write_records(manifest_path, [forged])

        with pytest.raises(ManifestVerificationError, match="run-c"):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    def test_three_key_rotation_verifies_with_the_current_and_both_previous_signers(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path)

        verify_ndjson_log(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])

    def test_three_key_rotation_rejects_a_retired_key_named_as_the_current_signer(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path)

        with pytest.raises(ManifestVerificationError, match=_under_key("key-3", "key-1")):
            verify_ndjson_log(audit_path, manifest_path, signer=key_1, previous_signers=[key_2, key_3])

    @pytest.mark.parametrize("seal_between", [True, False], ids=["seal-between", "back-to-back"])
    @pytest.mark.parametrize("retired", [0, 1], ids=["oldest-key", "middle-key"])
    def test_three_key_rotation_rejects_a_retired_key_manifest_after_the_last_rotation(
        self, tmp_path: Path, retired: int, seal_between: bool
    ) -> None:
        audit_path, manifest_path, *keys = _rotated_three_key_log(tmp_path, seal_between=seal_between)
        record = _record("run-d", 4)
        _write_records(audit_path, [record])
        head = manifest_hash(_read_lines(manifest_path)[-1])
        forged = seal_run([record], run_id="run-d", signer=keys[retired], previous_manifest_hash=head)
        _write_records(manifest_path, [forged])

        with pytest.raises(ManifestVerificationError, match="run-d") as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=keys[2], previous_signers=keys[:2])

        _assert_names(excinfo, keys[retired].key_id, "key-3")

    def test_previous_signer_with_a_different_algorithm_verifies_against_its_own_algorithm(
        self, tmp_path: Path
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        old_signer = _PrefixSigner()
        new_signer = _signer(_OTHER_KEY, "key-2")
        _write_records(audit_path, [_record("run-a", 1)])
        seal_ndjson_runs(audit_path, manifest_path, signer=old_signer)
        _rotate(manifest_path, new_signer, old_signer)

        verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    def test_previous_signer_manifest_with_a_mismatched_algorithm_is_rejected(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        old_signer = _PrefixSigner()
        new_signer = _signer(_OTHER_KEY, "key-2")
        _write_records(audit_path, [_record("run-a", 1)])
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=old_signer)
        # Forged after rotating: rotating verifies the log first.
        entry = _rotate(manifest_path, new_signer, old_signer)
        forged = {**manifests[0], "signature": {**manifests[0]["signature"], "algorithm": "HMAC-SHA256"}}
        _rewrite_lines(manifest_path, [forged, entry])

        with pytest.raises(ManifestVerificationError, match="algorithm"):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    @pytest.mark.parametrize(
        "previous_signers",
        [[_signer()], [_signer(key_id="key-3"), _signer(_OTHER_KEY, "key-3")]],
        ids=["signer-and-previous", "two-previous"],
    )
    def test_previous_signers_with_a_duplicate_key_id_is_a_value_error(
        self, tmp_path: Path, previous_signers: list[ManifestSigner]
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(_OTHER_KEY), previous_signers=previous_signers)

        assert not isinstance(excinfo.value, ManifestVerificationError)

    def test_edited_audit_line_of_a_sealed_run_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[0]["run_id"] == "run-a"
        records[0]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, records)

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        assert "do not match the record_hashes" in str(excinfo.value)

    def test_deleted_audit_line_of_a_sealed_run_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[0]["run_id"] == "run-a"
        _rewrite_lines(audit_path, records[1:])

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        assert "do not match the record_hashes" in str(excinfo.value)

    def test_sealed_run_without_any_audit_line_left_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert [record["run_id"] for record in records].count("run-c") == 1
        _rewrite_lines(audit_path, [record for record in records if record["run_id"] != "run-c"])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("appended", [1, 2])
    def test_records_appended_to_a_sealed_run_fail_and_the_run_is_never_resealed(
        self, tmp_path: Path, appended: int
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-a", 9 + offset) for offset in range(appended)])
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        message = str(excinfo.value)
        assert "run-a" in message
        assert f"has {appended} record(s) beyond its seal" in message
        assert "do not match the record_hashes" not in message
        assert seal_ndjson_runs(audit_path, manifest_path, signer=_signer()) == []
        assert manifest_path.read_bytes() == before

    def test_edited_last_manifest_line_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        manifests = _read_lines(manifest_path)
        manifests[-1]["compliant"] = not manifests[-1]["compliant"]
        _rewrite_lines(manifest_path, manifests)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_manifest_line_with_an_unhashable_run_id_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        manifests = _read_lines(manifest_path)
        assert manifests[0]["run_id"] == "run-a"
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

        message = str(excinfo.value)
        assert "expected head" in message
        assert message.count("do not match the record_hashes") == 19
        assert "and 5 more" in message

    def test_deleted_audit_file_fails_every_sealed_run(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        audit_path.unlink()

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

        # Log first: a run sealed in between looks unsealed, not tampered.
        assert reads == ["manifests.ndjson", "audit.ndjson"]

    def test_edited_audit_line_without_a_run_id_still_verifies(self, tmp_path: Path) -> None:
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

        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head


class TestVerifyNdjsonLogCoverage:
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

    def test_previous_signers_verifies_coverage_of_a_log_sealed_before_rotation(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        expected_head = manifest_hash(_rotate(manifest_path, new_signer, old_signer))

        coverage = verify_ndjson_log_coverage(
            audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer]
        )

        assert coverage.head == expected_head
        assert coverage.sealed_runs == 3

    @pytest.mark.parametrize(
        ("seal_between", "sealed_runs"), [(True, 3), (False, 2)], ids=["seal-between", "back-to-back"]
    )
    def test_rotation_entries_are_not_sealed_runs_and_never_unsealed(
        self, tmp_path: Path, seal_between: bool, sealed_runs: int
    ) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path, seal_between=seal_between)
        _write_records(audit_path, [_record("run-z", 9)])

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])

        assert coverage == LogCoverage(
            head=manifest_hash(_read_lines(manifest_path)[-1]),
            sealed_runs=sealed_runs,
            sealed_lines=sealed_runs,
            unattributed_lines=0,
            unsealed_lines={"run-z": 1},
        )

    @pytest.mark.parametrize(
        "previous_signers",
        [[_signer()], [_signer(key_id="key-3"), _signer(_OTHER_KEY, "key-3")]],
        ids=["signer-and-previous", "two-previous"],
    )
    def test_previous_signers_with_a_duplicate_key_id_is_a_value_error(
        self, tmp_path: Path, previous_signers: list[ManifestSigner]
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            verify_ndjson_log_coverage(
                audit_path, manifest_path, signer=_signer(_OTHER_KEY), previous_signers=previous_signers
            )

        assert not isinstance(excinfo.value, ManifestVerificationError)

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

    def test_coverage_is_hashable_and_hash_respects_equality(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        first = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())
        second = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        hash(first)
        assert first == second
        assert hash(first) == hash(second)
        assert len({first, second}) == 1

    def test_coverage_survives_dataclasses_asdict(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        as_dict = dataclasses.asdict(coverage)

        assert as_dict["head"] == coverage.head
        assert as_dict["unsealed_lines"] == coverage.unsealed_lines

    def test_coverage_survives_a_deepcopy(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert copy.deepcopy(coverage) == coverage

    def test_coverage_survives_a_pickle_round_trip(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert pickle.loads(pickle.dumps(coverage)) == coverage  # nosec

    def test_unsealed_lines_returned_is_a_private_copy(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())
        coverage.unsealed_lines["run-z"] = 999

        again = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert "run-z" not in again.unsealed_lines


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

    def test_sealing_skips_rotation_entries_like_sealed_runs(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path)
        before = manifest_path.read_bytes()

        assert seal_ndjson_runs(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2]) == []
        with pytest.raises(RunAlreadySealedError):
            seal_ndjson_runs(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2], run_id="run-a")

        assert manifest_path.read_bytes() == before


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


class TestPathAliasingGuards:
    """audit_path and manifest_path must not resolve to the same file: that is a caller mistake, not tampering."""

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_seal_ndjson_runs_refuses_an_aliased_audit_and_manifest_path(self, tmp_path: Path, spelling: str) -> None:
        audit_path, _manifest_path = _sealed_log(tmp_path)
        manifest_path = _aliased(tmp_path, audit_path, spelling)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_verify_ndjson_log_refuses_an_aliased_audit_and_manifest_path(self, tmp_path: Path, spelling: str) -> None:
        audit_path, _manifest_path = _sealed_log(tmp_path)
        manifest_path = _aliased(tmp_path, audit_path, spelling)

        with pytest.raises(ValueError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert not isinstance(excinfo.value, ManifestVerificationError)

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_verify_ndjson_log_coverage_refuses_an_aliased_audit_and_manifest_path(
        self, tmp_path: Path, spelling: str
    ) -> None:
        audit_path, _manifest_path = _sealed_log(tmp_path)
        manifest_path = _aliased(tmp_path, audit_path, spelling)

        with pytest.raises(ValueError) as excinfo:
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert not isinstance(excinfo.value, ManifestVerificationError)

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_quarantine_damaged_lines_refuses_an_aliased_audit_and_manifest_path(
        self, tmp_path: Path, spelling: str
    ) -> None:
        audit_path, _manifest_path = _sealed_log(tmp_path)
        manifest_path = _aliased(tmp_path, audit_path, spelling)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            quarantine_damaged_lines(audit_path, manifest_path, quarantine_path=_trace(tmp_path), signer=_signer())

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_quarantine_damaged_lines_refuses_an_aliased_manifest_and_quarantine_path(
        self, tmp_path: Path, spelling: str
    ) -> None:
        """quarantine_path aliasing manifest_path would nest _flock(quarantine_path) inside _flock(manifest_path)
        on the same file: a self-deadlock, not tampering."""
        audit_path, manifest_path = _sealed_log(tmp_path)
        quarantine_path = _aliased(tmp_path, manifest_path, spelling)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            quarantine_damaged_lines(audit_path, manifest_path, quarantine_path=quarantine_path, signer=_signer())

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_quarantine_damaged_lines_refuses_an_aliased_audit_and_quarantine_path(
        self, tmp_path: Path, spelling: str
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        quarantine_path = _aliased(tmp_path, audit_path, spelling)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            quarantine_damaged_lines(audit_path, manifest_path, quarantine_path=quarantine_path, signer=_signer())

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_quarantine_from_rotation_entry_refuses_an_aliased_manifest_and_quarantine_path(
        self, tmp_path: Path, spelling: str
    ) -> None:
        _, manifest_path, head = _log_with_rotation_entry(tmp_path)
        quarantine_path = _aliased(tmp_path, manifest_path, spelling)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            quarantine_from_rotation_entry(
                manifest_path, quarantine_path=quarantine_path, signer=_signer(), expected_head=head
            )

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before


class TestQuarantineDamagedLines:
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

    def test_failed_trace_append_to_an_existing_trace_log_is_rolled_back_and_a_retry_quarantines_everything(
        self, tmp_path: Path
    ) -> None:
        """os.write really writes a truncated prefix into the trace and returns the short count: bytes land on
        disk before _append_records raises, so only the truncate back to the previous trace content saves it."""
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        trace_before = _trace(tmp_path).read_bytes()
        manifest_before = manifest_path.read_bytes()
        _append_line(audit_path, b"[]")
        damaged = audit_path.read_bytes()
        real_write = os.write

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                _quarantine(tmp_path, audit_path, manifest_path)

        assert _trace(tmp_path).read_bytes() == trace_before
        assert audit_path.read_bytes() == damaged
        assert manifest_path.read_bytes() == manifest_before

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [7])
        assert len(_read_lines(_trace(tmp_path))) == 4

    def test_failed_trace_append_to_a_new_trace_log_leaves_no_trace_file_and_a_retry_quarantines_everything(
        self, tmp_path: Path
    ) -> None:
        """Same as above, but the trace log did not exist before: the rollback unlinks it instead of truncating."""
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        audit_before = audit_path.read_bytes()
        manifest_before = manifest_path.read_bytes()
        assert not _trace(tmp_path).exists()
        real_write = os.write

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                _quarantine(tmp_path, audit_path, manifest_path)

        assert not _trace(tmp_path).exists()
        assert audit_path.read_bytes() == audit_before
        assert manifest_path.read_bytes() == manifest_before

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert len(removed) == len(entries) == 3

    def test_failed_trace_append_to_an_existing_trace_log_fsyncs_it_and_its_directory_after_the_rollback_truncate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rollback truncate of the trace log must be durable too, or a crash right after it can resurrect
        bytes of an entry this recovery already reported as failed via the raised OSError."""
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        _append_line(audit_path, b"[]")
        trace_path = _trace(tmp_path)
        real_write = os.write
        real_truncate = os.truncate
        real_fsync = os.fsync
        events: list[str] = []

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        def spy_truncate(path: str | Path, length: int) -> None:
            events.append("truncate")
            real_truncate(path, length)

        def spy_fsync(fd: int) -> None:
            try:
                if os.path.samestat(os.fstat(fd), trace_path.stat()):
                    events.append("fsync-trace")
            except FileNotFoundError:
                pass
            try:
                if os.path.samestat(os.fstat(fd), trace_path.parent.stat()):
                    events.append("fsync-trace-dir")
            except FileNotFoundError:
                pass
            real_fsync(fd)

        monkeypatch.setattr(os, "truncate", spy_truncate)
        monkeypatch.setattr(os, "fsync", spy_fsync)

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                _quarantine(tmp_path, audit_path, manifest_path)

        assert "truncate" in events
        assert "fsync-trace" in events
        assert "fsync-trace-dir" in events
        assert events.index("truncate") < events.index("fsync-trace")
        assert events.index("truncate") < events.index("fsync-trace-dir")

    def test_failed_trace_append_to_a_new_trace_log_fsyncs_its_directory_after_the_rollback_unlink(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same as above, but the trace log did not exist before: the rollback unlinks it, and that unlink must
        itself be durable, so its directory entry is fsynced after the unlink."""
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        trace_path = _trace(tmp_path)
        assert not trace_path.exists()
        real_write = os.write
        real_unlink = os.unlink
        real_fsync = os.fsync
        events: list[str] = []

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        def spy_unlink(path: str | Path) -> None:
            events.append("unlink")
            real_unlink(path)

        def spy_fsync(fd: int) -> None:
            try:
                if os.path.samestat(os.fstat(fd), trace_path.parent.stat()):
                    events.append("fsync-trace-dir")
            except FileNotFoundError:
                pass
            real_fsync(fd)

        monkeypatch.setattr(os, "unlink", spy_unlink)
        monkeypatch.setattr(os, "fsync", spy_fsync)

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                _quarantine(tmp_path, audit_path, manifest_path)

        assert not trace_path.exists()
        assert "unlink" in events
        assert "fsync-trace-dir" in events
        assert events.index("unlink") < events.index("fsync-trace-dir")

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

    def test_previous_signers_repairs_a_torn_tail_signed_by_an_old_key(self, tmp_path: Path) -> None:
        audit_path, manifest_path, head = _torn_manifest_log(tmp_path)
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        manifest_before = manifest_path.read_bytes()

        removed = quarantine_damaged_lines(
            audit_path,
            manifest_path,
            quarantine_path=_trace(tmp_path),
            signer=new_signer,
            previous_signers=[old_signer],
            expected_head=head,
        )

        assert _summary(removed) == _spans("manifest", manifest_before, [4])
        # The repair does not rotate: the log is still under the old key.
        assert verify_ndjson_log(audit_path, manifest_path, signer=old_signer) == head
        with pytest.raises(ManifestVerificationError, match=_under_key("key-1", "key-2")):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    def test_torn_rotation_entry_tail_is_reported_by_a_dry_run_and_nothing_changes(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _torn_rotation_log(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path, dry_run=True)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [4])
        assert _snapshot(tmp_path) == before

    def test_torn_rotation_entry_tail_is_quarantined_and_rotating_again_restores_the_new_key(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, head = _torn_rotation_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        manifest_before = manifest_path.read_bytes()

        removed = _quarantine(
            tmp_path, audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=head
        )

        assert _summary(removed) == _spans("manifest", manifest_before, [4])
        assert manifest_path.read_bytes() == manifest_before[: manifest_before.rindex(b"\n") + 1]
        with pytest.raises(ManifestVerificationError, match=_under_key("key-1", "key-2")):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        with pytest.raises(ManifestVerificationError, match=_under_key("key-1", "key-2")):
            seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        entry = rotate_manifest_key(manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=head)
        assert verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer]) == (
            manifest_hash(entry)
        )

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_unterminated_rotation_entry_tail_that_is_valid_json_is_not_repaired(
        self, tmp_path: Path, dry_run: bool
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _torn(manifest_path, _canonical(_rotation_entry(_signer(_OTHER_KEY, "key-2"), head)))

        _assert_refused(tmp_path, audit_path, manifest_path, dry_run=dry_run)

    def test_expected_head_of_a_rotation_entry_repairs_a_torn_seal_after_it(self, tmp_path: Path) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        audit_path, manifest_path = _build_log(tmp_path, ("seal", old_signer), ("rotate", new_signer))
        entry_head = manifest_hash(_read_lines(manifest_path)[-1])
        record = _record("run-3", 3)
        _write_records(audit_path, [record])
        seal = _canonical(seal_run([record], run_id="run-3", signer=new_signer, previous_manifest_hash=entry_head))
        _torn(manifest_path, seal[: len(seal) // 2])
        manifest_before = manifest_path.read_bytes()

        removed = _quarantine(
            tmp_path,
            audit_path,
            manifest_path,
            signer=new_signer,
            previous_signers=[old_signer],
            expected_head=entry_head,
        )

        assert _summary(removed) == _spans("manifest", manifest_before, [3])
        verified = verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        assert verified == entry_head

    @pytest.mark.parametrize(
        "previous_signers",
        [[_signer()], [_signer(key_id="key-3"), _signer(_OTHER_KEY, "key-3")]],
        ids=["signer-and-previous", "two-previous"],
    )
    def test_previous_signers_with_a_duplicate_key_id_is_a_value_error(
        self, tmp_path: Path, previous_signers: list[ManifestSigner]
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            quarantine_damaged_lines(
                audit_path,
                manifest_path,
                quarantine_path=_trace(tmp_path),
                signer=_signer(_OTHER_KEY),
                previous_signers=previous_signers,
            )

        assert not isinstance(excinfo.value, ManifestVerificationError)

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
        assert manifest_path.exists()
        assert manifest_path.read_bytes() == b""
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

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_quarantine_log_without_a_final_newline_is_refused_and_nothing_changes(
        self, tmp_path: Path, dry_run: bool
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
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
                # Verifying and the trace both sign.
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

    def test_quarantine_fsyncs_the_manifest_log_and_its_directory_after_truncating(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        events: list[str] = []
        real_fsync = os.fsync

        def spy_fsync(fd: int) -> None:
            try:
                if os.path.samestat(os.fstat(fd), manifest_path.stat()):
                    events.append("fsync-manifest")
            except FileNotFoundError:
                pass
            try:
                if os.path.samestat(os.fstat(fd), manifest_path.parent.stat()):
                    events.append("fsync-manifest-dir")
            except FileNotFoundError:
                pass
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", spy_fsync)

        quarantine_damaged_lines(
            audit_path, manifest_path, quarantine_path=trace_dir / "quarantine.ndjson", signer=_signer()
        )

        assert "fsync-manifest" in events
        assert "fsync-manifest-dir" in events

    def test_recovery_holds_an_exclusive_lock_on_the_quarantine_log_during_the_trace_append(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("fcntl")
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        quarantine_path = _trace(tmp_path)
        quarantine_path.write_bytes(b"")
        refused: list[bool] = []

        def spy_append(*args: Any, **kwargs: Any) -> None:
            refused.append(_lock_refused(quarantine_path))
            _append_records(*args, **kwargs)

        monkeypatch.setattr(run_manifest_module, "_append_records", spy_append)

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert len(removed) == 3
        assert refused
        assert all(refused)

    def test_failed_trace_append_under_the_exclusive_lock_still_leaves_no_file_when_none_existed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("fcntl")
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        quarantine_path = _trace(tmp_path)
        assert not quarantine_path.exists()

        def partial_append(path: str | Path, records: Any) -> None:
            with open(path, "ab") as file:
                file.write(b'{"quarantine_version": 1, "fi')
            raise OSError(errno.ENOSPC, "disk full")

        monkeypatch.setattr(run_manifest_module, "_append_records", partial_append)

        with pytest.raises(OSError, match="disk full"):
            _quarantine(tmp_path, audit_path, manifest_path)

        assert not quarantine_path.exists()

    def test_repairs_a_torn_manifest_tail_where_fcntl_is_missing_even_when_the_quarantine_log_does_not_exist_yet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "fcntl", None)
        audit_path, manifest_path, head = _torn_manifest_log(tmp_path)
        manifest_before = manifest_path.read_bytes()
        assert not _trace(tmp_path).exists()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("manifest", manifest_before, [4])
        assert manifest_path.read_bytes() == manifest_before[: manifest_before.rindex(b"\n") + 1]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head
        assert len(_read_lines(_trace(tmp_path))) == 1


class TestQuarantineFromRotationEntry:
    """Drops a rotation entry and everything after it, anchored on the head just before the entry."""

    def test_the_entry_and_everything_after_it_are_dropped_and_the_honest_key_continues(self, tmp_path: Path) -> None:
        audit_path, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        # The control: the entry wedges the log for the honest key.
        with pytest.raises(ManifestVerificationError, match=_under_key("key-2", "key-1")):
            verify_ndjson_log(
                audit_path, manifest_path, signer=_signer(), previous_signers=[_signer(_OTHER_KEY, "key-2")]
            )
        before = _snapshot(tmp_path)
        manifest_before = before["manifests.ndjson"]

        removed = _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

        assert _summary(removed) == _spans("manifest", manifest_before, [3, 4])
        assert all("line 3" in item.reason for item in removed)
        assert manifest_path.read_bytes() == manifest_before[: removed[0].offset]
        assert audit_path.read_bytes() == before["audit.ndjson"]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == anchor
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())
        assert coverage.unsealed_lines == {"run-4": 1}
        _write_records(audit_path, [_record("run-5", 9)])
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=anchor, run_id="run-5")
        assert [(manifest["run_id"], manifest["previous_manifest_hash"]) for manifest in manifests] == [
            ("run-5", anchor)
        ]

    def test_trace_has_one_signed_entry_per_dropped_line_with_its_raw_bytes(self, tmp_path: Path) -> None:
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        manifest_before = manifest_path.read_bytes()

        removed = _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

        assert len(removed) == 2
        _assert_traced(tmp_path, manifest_path, manifest_before, removed)

    @pytest.mark.parametrize("forge", list(_FORGED_ENTRIES.values()), ids=list(_FORGED_ENTRIES))
    def test_an_entry_is_dropped_whatever_is_wrong_with_it(
        self, tmp_path: Path, forge: Callable[[str], dict[str, Any]]
    ) -> None:
        audit_path, manifest_path = _build_log(tmp_path, ("seal", _signer()), ("seal", _signer()))
        anchor = manifest_hash(_read_lines(manifest_path)[-1])
        _append_line(manifest_path, _canonical(forge(anchor)))
        before = _snapshot(tmp_path)

        removed = _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [3])
        assert manifest_path.read_bytes() == before["manifests.ndjson"][: removed[0].offset]
        _assert_traced(tmp_path, manifest_path, before["manifests.ndjson"], removed)
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == anchor

    @pytest.mark.parametrize(
        "after",
        [*_DAMAGED_LINES.values(), b'{"run_id": "run-x", "sig', b'{"run_id": "run-x"}'],
        ids=[*_DAMAGED_LINES, "unterminated-torn", "unterminated-valid-json"],
    )
    def test_a_line_after_the_entry_is_dropped_without_being_parsed(self, tmp_path: Path, after: bytes) -> None:
        steps = [("seal", _signer()), ("seal", _signer()), ("rotate", _signer(_OTHER_KEY, "key-2"))]
        audit_path, manifest_path = _build_log(tmp_path, *steps)
        anchor = manifest_hash(_read_lines(manifest_path)[1])
        _torn(manifest_path, after)
        before = _snapshot(tmp_path)

        removed = _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [3, 4])
        assert removed[1].length == len(after)
        assert manifest_path.read_bytes() == before["manifests.ndjson"][: removed[0].offset]
        _assert_traced(tmp_path, manifest_path, before["manifests.ndjson"], removed)
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == anchor

    def test_previous_signers_verify_a_prefix_sealed_by_retired_keys(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, key_2, _ = _rotated_three_key_log(tmp_path)
        anchor = manifest_hash(_read_lines(manifest_path)[2])
        before = _snapshot(tmp_path)

        removed = _quarantine_from_entry(
            tmp_path, manifest_path, signer=key_2, previous_signers=[key_1], expected_head=anchor
        )

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [4, 5])
        assert manifest_path.read_bytes() == before["manifests.ndjson"][: removed[0].offset]
        _assert_traced(tmp_path, manifest_path, before["manifests.ndjson"], removed, key_2)
        assert verify_ndjson_log(audit_path, manifest_path, signer=key_2, previous_signers=[key_1]) == anchor

    def test_a_retired_key_can_drop_the_honest_rotations_and_seals_after_its_own(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, *_ = _rotated_three_key_log(tmp_path)
        anchor = manifest_hash(_read_lines(manifest_path)[0])
        before = _snapshot(tmp_path)

        # The documented hazard: a retired key with write access drops honest lines too.
        removed = _quarantine_from_entry(tmp_path, manifest_path, signer=key_1, expected_head=anchor)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [2, 3, 4, 5])
        assert manifest_path.read_bytes() == before["manifests.ndjson"][: removed[0].offset]
        _assert_traced(tmp_path, manifest_path, before["manifests.ndjson"], removed, key_1)
        assert verify_ndjson_log(audit_path, manifest_path, signer=key_1) == anchor
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=key_1)
        assert coverage.unsealed_lines == {"run-b": 1, "run-c": 1}

    def test_trace_is_fsynced_before_the_manifest_is_truncated_and_the_manifest_and_its_directory_after(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        trace_path = _trace(trace_dir)
        events: list[tuple[str, int]] = []
        real_fsync, real_truncate = os.fsync, os.truncate

        def spy_fsync(fd: int) -> None:
            for name, target in (("trace", trace_path), ("manifest", manifest_path), ("manifest-dir", tmp_path)):
                try:
                    if os.path.samestat(os.fstat(fd), target.stat()):
                        events.append((f"fsync-{name}", target.stat().st_size))
                except FileNotFoundError:
                    pass
            real_fsync(fd)

        def spy_truncate(path: str | Path, length: int) -> None:
            events.append(("truncate", length))
            real_truncate(path, length)

        monkeypatch.setattr(os, "fsync", spy_fsync)
        monkeypatch.setattr(os, "truncate", spy_truncate)

        removed = _quarantine_from_entry(trace_dir, manifest_path, expected_head=anchor)

        assert len(removed) == 2
        names = [name for name, _ in events]
        cut = names.index("truncate")
        assert ("fsync-trace", trace_path.stat().st_size) in events[:cut]
        assert "fsync-manifest" in names[cut:]
        assert "fsync-manifest-dir" in names[cut:]

    def test_failed_trace_write_leaves_the_manifest_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        before = _snapshot(tmp_path)

        def disk_full(*args: Any, **kwargs: Any) -> None:
            raise OSError(errno.ENOSPC, "disk full")

        monkeypatch.setattr(run_manifest_module, "_append_records", disk_full)

        with pytest.raises(OSError, match="disk full"):
            _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

        assert _snapshot(tmp_path) == before

    def test_dry_run_returns_what_a_real_run_drops_and_changes_nothing(self, tmp_path: Path) -> None:
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        before = _snapshot(tmp_path)

        planned = _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor, dry_run=True)

        assert _summary(planned) == _spans("manifest", before["manifests.ndjson"], [3, 4])
        assert _snapshot(tmp_path) == before
        assert planned == _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

    def test_dry_run_does_not_wait_for_a_reader_holding_a_shared_lock(self, tmp_path: Path) -> None:
        fcntl = pytest.importorskip("fcntl")
        _audit_path, manifest_path, anchor = _log_with_rotation_entry(tmp_path)

        with ThreadPoolExecutor(max_workers=1) as pool:
            fd = os.open(manifest_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
                future = pool.submit(
                    _quarantine_from_entry, tmp_path, manifest_path, expected_head=anchor, dry_run=True
                )
                done, _ = wait([future], timeout=5)
            finally:
                os.close(fd)

            assert done
            assert len(future.result()) == 2

    def test_a_real_run_holds_an_exclusive_lock_on_the_manifest_log_while_it_repairs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        probes: list[tuple[str, bool]] = []
        real_truncate = os.truncate

        class _LockProbeSigner(HmacSha256Signer):
            def sign(self, payload: bytes) -> str:
                # Verifying and the trace both sign.
                probes.append(("sign", _lock_refused(manifest_path)))
                return super().sign(payload)

        def spy_append(*args: Any, **kwargs: Any) -> None:
            probes.append(("append", _lock_refused(manifest_path)))
            _append_records(*args, **kwargs)

        def spy_truncate(*args: Any, **kwargs: Any) -> None:
            probes.append(("truncate", _lock_refused(manifest_path)))
            real_truncate(*args, **kwargs)

        monkeypatch.setattr(run_manifest_module, "_append_records", spy_append)
        monkeypatch.setattr(os, "truncate", spy_truncate)

        removed = _quarantine_from_entry(
            tmp_path, manifest_path, signer=_LockProbeSigner(_KEY, "key-1"), expected_head=anchor
        )

        assert len(removed) == 2
        assert {stage for stage, _ in probes} == {"sign", "append", "truncate"}
        assert all(refused for _, refused in probes)
        fd = os.open(manifest_path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    def test_a_real_run_holds_an_exclusive_lock_on_the_quarantine_log_during_the_trace_append(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("fcntl")
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        quarantine_path = _trace(tmp_path)
        quarantine_path.write_bytes(b"")
        refused: list[bool] = []

        def spy_append(*args: Any, **kwargs: Any) -> None:
            refused.append(_lock_refused(quarantine_path))
            _append_records(*args, **kwargs)

        monkeypatch.setattr(run_manifest_module, "_append_records", spy_append)

        removed = _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

        assert len(removed) == 2
        assert refused
        assert all(refused)

    def test_drops_the_entry_where_fcntl_is_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "fcntl", None)
        audit_path, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        manifest_before = manifest_path.read_bytes()
        assert not _trace(tmp_path).exists()

        removed = _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

        assert _summary(removed) == _spans("manifest", manifest_before, [3, 4])
        assert manifest_path.read_bytes() == manifest_before[: removed[0].offset]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == anchor
        assert len(_read_lines(_trace(tmp_path))) == 2

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_an_anchor_with_nothing_after_it_returns_nothing_and_creates_no_trace(
        self, tmp_path: Path, dry_run: bool
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        before = _snapshot(tmp_path)

        assert _quarantine_from_entry(tmp_path, manifest_path, expected_head=head, dry_run=dry_run) == []

        assert _snapshot(tmp_path) == before
        assert not _trace(tmp_path).exists()

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    @pytest.mark.parametrize(
        "refusal",
        [
            "anchor-not-in-the-log",
            "first-dropped-line-is-a-seal",
            "first-dropped-line-is-torn",
            "first-dropped-line-is-an-unterminated-entry",
            "first-dropped-line-is-not-json",
            "first-dropped-line-is-not-an-entry",
            "signer-is-not-current-at-the-anchor",
            "prefix-does-not-verify",
            "quarantine-log-without-a-final-newline",
        ],
    )
    def test_a_log_that_cannot_be_dropped_from_is_refused_and_nothing_changes(
        self, tmp_path: Path, refusal: str, dry_run: bool
    ) -> None:
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        lines = manifest_path.read_bytes().splitlines(keepends=True)
        prefix, entry = b"".join(lines[:2]), lines[2]
        signer: ManifestSigner = _signer()
        previous_signers: list[ManifestSigner] = []
        names: list[str] = []
        if refusal == "anchor-not-in-the-log":
            anchor = "0" * 64
            names = [anchor]
        if refusal == "first-dropped-line-is-a-seal":
            anchor = manifest_hash(json.loads(lines[0]))
        if refusal == "first-dropped-line-is-torn":
            manifest_path.write_bytes(prefix + entry[: len(entry) // 2])
        if refusal == "first-dropped-line-is-an-unterminated-entry":
            manifest_path.write_bytes(prefix + entry.removesuffix(b"\n"))
        if refusal == "first-dropped-line-is-not-json":
            manifest_path.write_bytes(prefix + b"garbage\n" + entry)
        if refusal == "first-dropped-line-is-not-an-entry":
            manifest_path.write_bytes(prefix + b'{"run_id": "run-x"}\n' + entry)
        if refusal == "signer-is-not-current-at-the-anchor":
            signer = _signer(_OTHER_KEY, "key-2")
            previous_signers = [_signer()]
            names = ["key-1", "key-2"]
        if refusal == "prefix-does-not-verify":
            edited = lines[0].replace(b'"compliant": true', b'"compliant": false')
            assert edited != lines[0]
            manifest_path.write_bytes(edited + b"".join(lines[1:]))
        if refusal == "quarantine-log-without-a-final-newline":
            _trace(tmp_path).write_bytes(b'{"quarantine_version": 1, "fi')
            names = [str(_trace(tmp_path))]

        excinfo = _assert_raises_and_unchanged(
            tmp_path,
            lambda: _quarantine_from_entry(
                tmp_path,
                manifest_path,
                signer=signer,
                previous_signers=previous_signers,
                expected_head=anchor,
                dry_run=dry_run,
            ),
        )

        _assert_names(excinfo, *names)

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_a_missing_manifest_log_is_refused_and_no_file_is_created(self, tmp_path: Path, dry_run: bool) -> None:
        _assert_raises_and_unchanged(
            tmp_path,
            lambda: _quarantine_from_entry(
                tmp_path, tmp_path / "manifests.ndjson", expected_head="0" * 64, dry_run=dry_run
            ),
        )

        assert _snapshot(tmp_path) == {}

    def test_omitting_expected_head_is_a_type_error(self, tmp_path: Path) -> None:
        manifest_path = _log_with_rotation_entry(tmp_path)[1]
        before = _snapshot(tmp_path)

        with pytest.raises(TypeError):
            quarantine_from_rotation_entry(  # type: ignore[call-arg]
                manifest_path, quarantine_path=_trace(tmp_path), signer=_signer()
            )

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("expected_head", [None, b"0" * 64], ids=["none", "bytes"])
    def test_a_non_string_expected_head_is_a_plain_value_error(self, tmp_path: Path, expected_head: Any) -> None:
        manifest_path = _log_with_rotation_entry(tmp_path)[1]
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            _quarantine_from_entry(tmp_path, manifest_path, expected_head=expected_head)

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before


class TestRunManifestRunAll:
    """A real run's audit file seals into one verifiable manifest."""

    @pytest.mark.parametrize(
        ("mode", "fail_closed"),
        [
            (ParallelizationMode.SYNC, False),
            (ParallelizationMode.THREADING, False),
            (ParallelizationMode.MULTIPROCESSING, False),
            (ParallelizationMode.MULTIPROCESSING, True),
        ],
    )
    def test_run_all_audit_file_seals_and_verifies(
        self, mode: ParallelizationMode, fail_closed: bool, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        # Only MULTIPROCESSING needs the flight_server fixture.
        flight_server = (
            request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
        )
        extender = AuditExtender(sink=NdjsonAuditSink(audit_path), fail_closed=fail_closed)

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

    def test_run_all_fail_closed_refusal_seals_a_non_compliant_manifest(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"

        with pytest.raises(IdentityRequiredError):
            run_value_int(AuditExtender(sink=NdjsonAuditSink(audit_path), fail_closed=True))

        records = _read_lines(audit_path)
        assert len(records) == 1
        assert records[0]["decision"] == "deny"
        assert records[0]["hook"] == "FEATURE_GROUP_MATCHED"
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert len(manifests) == 1
        assert manifests[0]["compliant"] is False
