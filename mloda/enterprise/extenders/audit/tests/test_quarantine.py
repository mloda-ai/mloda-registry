"""Tests for quarantining damaged lines and tracing the chain."""

from __future__ import annotations

import base64
import dataclasses
import errno
import json
import os
import re
import stat
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mloda.enterprise.extenders.audit import (
    ManifestSigner,
    ManifestVerificationError,
    QuarantinedLine,
    manifest_hash,
    quarantine_damaged_lines,
    quarantine_from_rotation_entry,
    rotate_manifest_key,
    seal_ndjson_runs,
    seal_run,
    verify_ndjson_log,
    verify_ndjson_log_coverage,
)
from mloda.enterprise.extenders.audit.audit_extender import _append_records
from mloda.enterprise.extenders.audit.tests.manifest_helpers import (
    _BAD_SIGNATURE,
    _CAP,
    _DAMAGED_LINES,
    _FORGED_ENTRIES,
    _KEY,
    _OTHER_KEY,
    _append_line,
    _assert_names,
    _assert_raises_and_unchanged,
    _assert_refused,
    _both_algorithms,
    _both_recoveries,
    _build_log,
    _canonical,
    _cap,
    _damaged_log,
    _HookedSigner,
    _insert_line,
    _lock_refused,
    _log_with_rotation_entry,
    _oversized_line,
    _patch_bindings,
    _quarantine,
    _quarantine_from_entry,
    _read_lines,
    _record,
    _Recovery,
    _reference_signature,
    _rewrite_lines,
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
    _torn_manifest_log,
    _trace,
    _trace_chain,
    _trace_entry_ref,
    _under_key,
    _unsigned,
    _verify_quarantine,
    _write_records,
)

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
    "previous_entry_hash",
    "signature",
}


_NOT_AN_ENTRY = "is not a key rotation entry"


_UNTERMINATED = "does not end with a newline"


# The message fragment of the error each refusal of quarantine_from_rotation_entry must raise, so none passes for the
# wrong reason.
_REFUSAL_REASONS: dict[str, str] = {
    "anchor-not-in-the-log": "has the head",
    "first-dropped-line-is-a-seal": _NOT_AN_ENTRY,
    "first-dropped-line-is-torn": _UNTERMINATED,
    "first-dropped-line-is-an-unterminated-entry": _UNTERMINATED,
    "first-dropped-line-is-not-json": "is not valid JSON",
    "first-dropped-line-is-not-an-entry": _NOT_AN_ENTRY,
    "signer-is-not-current-at-the-anchor": "not the signer's",
    "prefix-does-not-verify": _BAD_SIGNATURE,
    "quarantine-log-without-a-final-newline": _UNTERMINATED,
}


def _torn_rotation_log(directory: Path) -> tuple[Path, Path, str]:
    """A manifest log ending in half a key-2 rotation entry; returns the head before the damage."""
    audit_path, manifest_path = _sealed_log(directory)
    head = manifest_hash(_read_lines(manifest_path)[-1])
    entry = _canonical(_rotation_entry(_signer(_OTHER_KEY, "key-2"), head)) + b"\n"
    _torn(manifest_path, entry[: len(entry) // 2])
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


def _assert_traced(
    directory: Path,
    manifest_path: Path,
    manifest_before: bytes,
    removed: list[QuarantinedLine],
    signer: ManifestSigner | None = None,
    *,
    audit: tuple[Path, bytes] | None = None,
) -> None:
    """The trace holds one entry per dropped line: its report, its raw bytes and a signature by `signer`.
    `audit` is the audit file and its bytes before the recovery, if the recovery may drop audit lines."""
    signer = signer or _signer()
    logs = {"manifest": (manifest_path, manifest_before)}
    if audit:
        logs["audit"] = audit
    entries = _read_lines(_trace(directory))
    assert len(entries) == len(removed)
    assert _trace(directory).read_bytes() == b"".join(_canonical(entry) + b"\n" for entry in entries)
    for entry, item in zip(entries, removed, strict=True):
        path, before = logs[item.file]
        raw = base64.b64decode(entry["raw_base64"], validate=True)
        assert set(entry) == _EXPECTED_QUARANTINE_KEYS
        assert entry["quarantine_version"] == 2
        assert entry["path"] == str(path)
        assert {field: entry[field] for field in dataclasses.asdict(item)} == dataclasses.asdict(item)
        assert raw == before.splitlines(keepends=True)[item.line - 1]
        assert entry["sha256"] == _sha256(raw)
        signature = entry["signature"]
        assert set(signature) == {"algorithm", "key_id", "value"}
        assert (signature["algorithm"], signature["key_id"]) == (signer.algorithm, signer.key_id)
        assert signature["value"] == signer.sign(_signing_payload_ref(entry))
    assert [entry["previous_entry_hash"] for entry in entries] == [
        None,
        *(_sha256(_canonical(e)) for e in entries[:-1]),
    ]


@_both_algorithms
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

    def test_torn_manifest_tail_of_a_successor_segment_is_truncated(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _successor_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        before = manifest_path.read_bytes()
        _torn(manifest_path, b'{"manifest_version": 2, "run_')

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert [item.file for item in removed] == ["manifest"]
        assert manifest_path.read_bytes() == before
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head

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
        _assert_traced(
            tmp_path, manifest_path, before["manifests.ndjson"], removed, audit=(audit_path, before["audit.ndjson"])
        )

    def test_trace_keeps_the_removed_bytes_so_they_can_be_inspected_or_restored(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        before = _snapshot(tmp_path)
        audit_lines = before["audit.ndjson"].splitlines(keepends=True)
        manifest_lines = before["manifests.ndjson"].splitlines(keepends=True)

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        _assert_traced(
            tmp_path, manifest_path, before["manifests.ndjson"], removed, audit=(audit_path, before["audit.ndjson"])
        )
        entries = _read_lines(_trace(tmp_path))
        removed_bytes = [manifest_lines[3], audit_lines[3], audit_lines[7]]
        assert [base64.b64decode(entry["raw_base64"], validate=True) for entry in entries] == removed_bytes
        assert [entry["sha256"] for entry in entries] == [_sha256(raw) for raw in removed_bytes]

    def test_trace_signature_covers_the_entry_without_its_signature(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert removed
        _assert_traced(
            tmp_path, manifest_path, before["manifests.ndjson"], removed, audit=(audit_path, before["audit.ndjson"])
        )
        for entry in _read_lines(_trace(tmp_path)):
            payload = _signing_payload_ref(entry)
            signature = entry["signature"]
            assert set(signature) == {"algorithm", "key_id", "value"}
            assert (signature["algorithm"], signature["key_id"]) == (_signer().algorithm, _signer().key_id)
            assert signature["value"] == _reference_signature(_KEY, payload)
            tampered = _signing_payload_ref({**entry, "line": entry["line"] + 1})
            assert _signer().verify(tampered, entry["signature"]["value"]) is False

    def test_quarantined_at_is_rfc3339_utc(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)

        _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert entries
        for entry in entries:
            quarantined_at = entry["quarantined_at"]
            assert quarantined_at.endswith("Z")
            datetime.fromisoformat(quarantined_at.removesuffix("Z"))

    @pytest.mark.parametrize("position", ["terminated", "unterminated"])
    def test_an_oversized_audit_line_is_dropped_and_its_trace_entry_stays_within_the_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, position: str
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _cap(monkeypatch)
        if position == "terminated":
            big = _oversized_line() + b"\n"
            _insert_line(audit_path, 3, big)
            number = 4
        else:
            big = b"x" * (_CAP + 1)
            _torn(audit_path, big)
            number = 7
        before = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", before, [number])
        assert audit_path.read_bytes() == before.replace(big, b"")
        trace_lines = _trace(tmp_path).read_bytes().splitlines()
        assert [len(line) <= _CAP for line in trace_lines] == [True]
        entry = json.loads(trace_lines[0])
        assert entry.get("raw_base64") is None
        assert (entry["sha256"], entry["length"]) == (_sha256(big), len(big))
        assert _verify_quarantine(_trace(tmp_path), _signer()) is not None

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

        _patch_bindings(monkeypatch, "_append_records", disk_full)

        with pytest.raises(OSError, match="disk full"):
            _quarantine(tmp_path, audit_path, manifest_path)

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("existing", [False, True], ids=["new-trace", "existing-trace"])
    def test_failed_trace_append_is_rolled_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        if existing:
            _rewrite_lines(_trace(tmp_path), [_trace_entry_ref(_signer(), None)])
        before = _snapshot(tmp_path)

        def partial_append(path: str | Path, records: Any) -> None:
            with open(path, "ab") as file:
                file.write(b'{"quarantine_version": 1, "fi')
            raise OSError(errno.ENOSPC, "disk full")

        _patch_bindings(monkeypatch, "_append_records", partial_append)

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

    @_both_recoveries
    def test_trace_is_fsynced_before_any_file_is_modified_and_the_manifest_and_its_directory_after_the_cut(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recovery: _Recovery
    ) -> None:
        manifest_path, repair, dropped = recovery(tmp_path)
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        trace_path = _trace(trace_dir)
        events: list[tuple[str, int]] = []
        real_fsync, real_replace, real_truncate = os.fsync, os.replace, os.truncate

        def spy_fsync(fd: int) -> None:
            for name, target in (("trace", trace_path), ("manifest", manifest_path), ("manifest-dir", tmp_path)):
                try:
                    if os.path.samestat(os.fstat(fd), target.stat()):
                        events.append((f"fsync-{name}", target.stat().st_size))
                except FileNotFoundError:
                    pass
            real_fsync(fd)

        def spy_replace(*args: Any, **kwargs: Any) -> None:
            events.append(("replace", 0))
            real_replace(*args, **kwargs)

        def spy_truncate(path: str | Path, length: int) -> None:
            events.append(("truncate", length))
            real_truncate(path, length)

        monkeypatch.setattr(os, "fsync", spy_fsync)
        monkeypatch.setattr(os, "replace", spy_replace)
        monkeypatch.setattr(os, "truncate", spy_truncate)

        removed = repair(trace_dir)

        assert len(removed) == dropped
        names = [name for name, _ in events]
        # The manifest log is cut, and the audit file replaced if the repair dropped audit lines.
        modifications = {"truncate" if item.file == "manifest" else "replace" for item in removed}
        assert "truncate" in modifications
        assert modifications <= set(names)
        first = min(names.index(name) for name in modifications)
        assert ("fsync-trace", trace_path.stat().st_size) in events[:first]
        cut = names.index("truncate")
        assert "fsync-manifest" in names[cut:]
        assert "fsync-manifest-dir" in names[cut:]

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

        def probe_signing(payload: bytes) -> None:
            probes.append(("sign", _lock_refused(manifest_path)))

        def spy_append(*args: Any, **kwargs: Any) -> None:
            probes.append(("append", _lock_refused(manifest_path)))
            _append_records(*args, **kwargs)

        def spy_replace(*args: Any, **kwargs: Any) -> None:
            probes.append(("replace", _lock_refused(manifest_path)))
            real_replace(*args, **kwargs)

        _patch_bindings(monkeypatch, "_append_records", spy_append)
        monkeypatch.setattr(os, "replace", spy_replace)

        # Verifying and the trace both use the signer, so the wrapper probes both hooks.
        probing = _HookedSigner(_signer(), on_sign=probe_signing, on_verify=probe_signing)
        removed = _quarantine(tmp_path, audit_path, manifest_path, signer=probing)

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

    @_both_recoveries
    def test_recovery_holds_an_exclusive_lock_on_the_quarantine_log_during_the_trace_append(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recovery: _Recovery
    ) -> None:
        pytest.importorskip("fcntl")
        _, repair, dropped = recovery(tmp_path)
        quarantine_path = _trace(tmp_path)
        quarantine_path.write_bytes(b"")
        refused: list[bool] = []

        def spy_append(*args: Any, **kwargs: Any) -> None:
            refused.append(_lock_refused(quarantine_path))
            _append_records(*args, **kwargs)

        _patch_bindings(monkeypatch, "_append_records", spy_append)

        removed = repair(tmp_path)

        assert len(removed) == dropped
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

        _patch_bindings(monkeypatch, "_append_records", partial_append)

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


@_both_algorithms
class TestQuarantineTraceChain:
    """Trace entries are v2: chained by previous_entry_hash and signed over the v2 payload."""

    def test_the_first_entry_has_no_predecessor_and_each_later_one_hashes_the_line_before(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)

        _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert len(entries) == 3
        assert entries[0]["previous_entry_hash"] is None
        assert [entry["previous_entry_hash"] for entry in entries[1:]] == [_sha256(_canonical(e)) for e in entries[:2]]

    def test_a_second_recovery_chains_onto_the_existing_trace(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        _append_line(audit_path, b"[]")

        _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert len(entries) == 4
        assert entries[3]["previous_entry_hash"] == _sha256(_canonical(entries[2]))
        assert _verify_quarantine(_trace(tmp_path), _signer()) == _sha256(_canonical(entries[3]))

    def test_a_recovery_chains_onto_a_legacy_v1_trace(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        legacy = _trace_entry_ref(_signer(), None, v1=True)
        _rewrite_lines(_trace(tmp_path), [legacy])

        _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert entries[0] == legacy
        assert entries[1]["quarantine_version"] == 2
        assert entries[1]["previous_entry_hash"] == _sha256(_canonical(legacy))
        assert _verify_quarantine(_trace(tmp_path), _signer()) == _sha256(_canonical(entries[-1]))

    def test_a_tampered_trace_refuses_the_repair_and_changes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        _append_line(audit_path, b"[]")
        entries = _read_lines(_trace(tmp_path))
        entries[1]["line"] += 1
        _rewrite_lines(_trace(tmp_path), entries)

        _assert_raises_and_unchanged(tmp_path, lambda: _quarantine(tmp_path, audit_path, manifest_path))

    def test_a_trace_with_a_deleted_line_refuses_the_repair_and_changes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        _append_line(audit_path, b"[]")
        entries = _read_lines(_trace(tmp_path))
        _rewrite_lines(_trace(tmp_path), [entries[0], entries[2]])

        _assert_raises_and_unchanged(tmp_path, lambda: _quarantine(tmp_path, audit_path, manifest_path))

    def test_a_trace_under_an_unknown_key_refuses_the_repair_and_changes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _rewrite_lines(_trace(tmp_path), _trace_chain(_signer(_OTHER_KEY, "key-2")))

        _assert_raises_and_unchanged(tmp_path, lambda: _quarantine(tmp_path, audit_path, manifest_path))

    def test_a_trace_signed_by_a_retired_key_is_extended_with_previous_signers(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _rewrite_lines(_trace(tmp_path), _trace_chain(_signer()))
        key_2 = _signer(_OTHER_KEY, "key-2")

        removed = _quarantine(tmp_path, audit_path, manifest_path, signer=key_2, previous_signers=[_signer()])

        assert len(removed) == 3
        assert len(_read_lines(_trace(tmp_path))) == 4
        assert _verify_quarantine(_trace(tmp_path), key_2, _signer()) is not None

    @pytest.mark.parametrize("existing", [False, True], ids=["new-trace", "existing-trace"])
    def test_a_signing_failure_leaves_no_trace_file_it_created_and_changes_nothing(
        self, tmp_path: Path, existing: bool
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        if existing:
            _rewrite_lines(_trace(tmp_path), _trace_chain(_signer()))
        calls: list[bytes] = []

        def fail_on_second(payload: bytes) -> None:
            calls.append(payload)
            if len(calls) == 2:
                raise RuntimeError("signing failed")

        before = _snapshot(tmp_path)

        with pytest.raises(RuntimeError, match="signing failed"):
            _quarantine(tmp_path, audit_path, manifest_path, signer=_HookedSigner(_signer(), on_sign=fail_on_second))

        assert len(calls) == 2
        assert _snapshot(tmp_path) == before
        assert existing or not _trace(tmp_path).exists()

    def test_verify_returns_none_for_a_missing_or_empty_trace(self, tmp_path: Path) -> None:
        assert _verify_quarantine(_trace(tmp_path), _signer()) is None
        _trace(tmp_path).write_bytes(b"")
        assert _verify_quarantine(_trace(tmp_path), _signer()) is None

    def test_verify_returns_the_hash_of_the_last_line_as_the_head(self, tmp_path: Path) -> None:
        entries = _trace_chain(_signer(), _signer(), _signer())
        _rewrite_lines(_trace(tmp_path), entries)

        assert _verify_quarantine(_trace(tmp_path), _signer()) == _sha256(_canonical(entries[-1]))

    def test_verify_accepts_lines_signed_by_a_retired_key_through_previous_signers(self, tmp_path: Path) -> None:
        key_2 = _signer(_OTHER_KEY, "key-2")
        entries = _trace_chain(_signer(), key_2)
        _rewrite_lines(_trace(tmp_path), entries)

        assert _verify_quarantine(_trace(tmp_path), key_2, _signer()) == _sha256(_canonical(entries[-1]))
        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), key_2)

    def test_verify_rejects_a_trace_under_another_key(self, tmp_path: Path) -> None:
        _rewrite_lines(_trace(tmp_path), _trace_chain(_signer()))

        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), _signer(_OTHER_KEY, "key-1"))

    @pytest.mark.parametrize("attack", ["delete-first", "delete-middle", "reorder", "edit", "relabel-key-id"])
    def test_verify_detects_a_deleted_reordered_or_edited_line(self, tmp_path: Path, attack: str) -> None:
        entries = _trace_chain(_signer(), _signer(), _signer())
        attacked = {
            "delete-first": [entries[1], entries[2]],
            "delete-middle": [entries[0], entries[2]],
            "reorder": [entries[0], entries[2], entries[1]],
            "edit": [entries[0], {**entries[1], "reason": "other"}, entries[2]],
            "relabel-key-id": [
                entries[0],
                {**entries[1], "signature": {**entries[1]["signature"], "key_id": "x"}},
                entries[2],
            ],
        }[attack]
        _rewrite_lines(_trace(tmp_path), attacked)

        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_verify_rejects_a_line_with_the_wrong_chain_value_even_when_correctly_signed(self, tmp_path: Path) -> None:
        entries = _trace_chain(_signer(), _signer())
        entries[1] = _trace_entry_ref(_signer(), "0" * 64, line=2)
        _rewrite_lines(_trace(tmp_path), entries)

        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_verify_rejects_a_first_line_that_names_a_predecessor(self, tmp_path: Path) -> None:
        _rewrite_lines(_trace(tmp_path), [_trace_entry_ref(_signer(), "0" * 64)])

        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_verify_rejects_an_unknown_version(self, tmp_path: Path) -> None:
        _rewrite_lines(_trace(tmp_path), [_trace_entry_ref(_signer(), None, quarantine_version=3)])

        with pytest.raises(ManifestVerificationError, match="3"):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_verify_rejects_an_unterminated_last_line(self, tmp_path: Path) -> None:
        _rewrite_lines(_trace(tmp_path), _trace_chain(_signer()))
        _torn(_trace(tmp_path), b'{"quarantine_version": 2, "fi')

        with pytest.raises(ManifestVerificationError, match=re.escape(str(_trace(tmp_path)))):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_legacy_v1_lines_verify_as_a_prefix_before_v2_lines(self, tmp_path: Path) -> None:
        legacy = [
            _trace_entry_ref(_signer(), None, v1=True, line=1),
            _trace_entry_ref(_signer(), None, v1=True, line=2),
        ]
        v2 = _trace_entry_ref(_signer(), _sha256(_canonical(legacy[-1])), line=3)
        _rewrite_lines(_trace(tmp_path), legacy)
        assert _verify_quarantine(_trace(tmp_path), _signer()) == _sha256(_canonical(legacy[-1]))

        _rewrite_lines(_trace(tmp_path), [*legacy, v2])

        assert _verify_quarantine(_trace(tmp_path), _signer()) == _sha256(_canonical(v2))

    def test_a_v1_line_after_a_v2_line_is_rejected(self, tmp_path: Path) -> None:
        first = _trace_entry_ref(_signer(), None, line=1)
        late_v1 = _trace_entry_ref(_signer(), None, v1=True, line=2)
        _rewrite_lines(_trace(tmp_path), [first, late_v1])

        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_a_v1_line_signed_over_the_v2_payload_is_rejected(self, tmp_path: Path) -> None:
        v1 = _trace_entry_ref(_signer(), None, v1=True)
        block = {"algorithm": v1["signature"]["algorithm"], "key_id": v1["signature"]["key_id"]}
        v1["signature"]["value"] = _signer().sign(_signing_payload_ref({**_unsigned(v1), "signature": block}))
        _rewrite_lines(_trace(tmp_path), [v1])

        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_every_line_hash_of_the_trace_passes_as_an_anchor(self, tmp_path: Path) -> None:
        entries = _trace_chain(_signer(), _signer(), _signer())
        _rewrite_lines(_trace(tmp_path), entries)
        heads = [_sha256(_canonical(entry)) for entry in entries]

        assert _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=heads) == heads[-1]
        assert _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=iter(heads)) == heads[-1]

    def test_an_anchor_on_a_legacy_v1_line_hash_passes(self, tmp_path: Path) -> None:
        legacy = _trace_entry_ref(_signer(), None, v1=True)
        _rewrite_lines(_trace(tmp_path), [legacy])

        assert _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=[_sha256(_canonical(legacy))]) == _sha256(
            _canonical(legacy)
        )

    def test_one_unknown_anchor_among_valid_ones_fails_and_is_named(self, tmp_path: Path) -> None:
        entries = _trace_chain(_signer(), _signer())
        _rewrite_lines(_trace(tmp_path), entries)
        unknown = "1" * 64

        with pytest.raises(ManifestVerificationError) as excinfo:
            _verify_quarantine(
                _trace(tmp_path), _signer(), anchored_heads=[*(_sha256(_canonical(e)) for e in entries), unknown]
            )

        _assert_names(excinfo, unknown, "not a line of the quarantine trace")

    def test_a_trace_with_its_newest_line_dropped_fails_only_with_the_old_head_anchored(self, tmp_path: Path) -> None:
        entries = _trace_chain(_signer(), _signer(), _signer())
        old_head = _sha256(_canonical(entries[-1]))
        _rewrite_lines(_trace(tmp_path), entries[:-1])

        # Control: without the anchor the cut trace verifies.
        assert _verify_quarantine(_trace(tmp_path), _signer()) == _sha256(_canonical(entries[1]))
        with pytest.raises(ManifestVerificationError) as excinfo:
            _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=[old_head])
        _assert_names(excinfo, old_head, "not a line of the quarantine trace")

    @pytest.mark.parametrize("state", ["missing", "empty"])
    def test_a_missing_or_empty_trace_fails_with_an_anchor(self, tmp_path: Path, state: str) -> None:
        if state == "empty":
            _trace(tmp_path).write_bytes(b"")

        with pytest.raises(ManifestVerificationError) as excinfo:
            _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=["2" * 64])

        _assert_names(excinfo, "2" * 64, "not a line of the quarantine trace")

    def test_a_trace_cut_and_regrown_by_a_repair_fails_with_the_old_head_anchored(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        entries = _read_lines(_trace(tmp_path))
        old_head = _sha256(_canonical(entries[-1]))
        _rewrite_lines(_trace(tmp_path), entries[:-1])
        _append_line(audit_path, b"[]")
        _quarantine(tmp_path, audit_path, manifest_path)
        assert len(_read_lines(_trace(tmp_path))) == len(entries)

        with pytest.raises(ManifestVerificationError) as excinfo:
            _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=[old_head])

        _assert_names(excinfo, old_head)

    @_both_recoveries
    def test_a_cut_trace_with_the_old_head_anchored_refuses_the_repair(
        self, tmp_path: Path, recovery: _Recovery
    ) -> None:
        _, repair, _ = recovery(tmp_path)
        entries = _trace_chain(_signer(), _signer(), _signer())
        _rewrite_lines(_trace(tmp_path), entries[:-1])
        old_head = _sha256(_canonical(entries[-1]))

        excinfo = _assert_raises_and_unchanged(tmp_path, lambda: repair(tmp_path, anchored_heads=[old_head]))

        _assert_names(excinfo, old_head, "not a line of the quarantine trace")

    @_both_recoveries
    def test_the_current_head_anchored_lets_the_repair_proceed(self, tmp_path: Path, recovery: _Recovery) -> None:
        _, repair, dropped = recovery(tmp_path)
        entries = _trace_chain(_signer(), _signer())
        _rewrite_lines(_trace(tmp_path), entries)

        removed = repair(tmp_path, anchored_heads=[_sha256(_canonical(entries[-1]))])

        assert len(removed) == dropped
        assert len(_read_lines(_trace(tmp_path))) == 2 + dropped


@_both_algorithms
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
    def test_a_line_after_the_entry_is_dropped_whatever_it_holds(self, tmp_path: Path, after: bytes) -> None:
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

    def test_failed_trace_write_leaves_the_manifest_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        before = _snapshot(tmp_path)

        def disk_full(*args: Any, **kwargs: Any) -> None:
            raise OSError(errno.ENOSPC, "disk full")

        _patch_bindings(monkeypatch, "_append_records", disk_full)

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

        def probe_signing(payload: bytes) -> None:
            probes.append(("sign", _lock_refused(manifest_path)))

        def spy_append(*args: Any, **kwargs: Any) -> None:
            probes.append(("append", _lock_refused(manifest_path)))
            _append_records(*args, **kwargs)

        def spy_truncate(*args: Any, **kwargs: Any) -> None:
            probes.append(("truncate", _lock_refused(manifest_path)))
            real_truncate(*args, **kwargs)

        _patch_bindings(monkeypatch, "_append_records", spy_append)
        monkeypatch.setattr(os, "truncate", spy_truncate)

        # Verifying and the trace both use the signer, so the wrapper probes both hooks.
        probing = _HookedSigner(_signer(), on_sign=probe_signing, on_verify=probe_signing)
        removed = _quarantine_from_entry(tmp_path, manifest_path, signer=probing, expected_head=anchor)

        assert len(removed) == 2
        assert {stage for stage, _ in probes} == {"sign", "append", "truncate"}
        assert all(refused for _, refused in probes)
        fd = os.open(manifest_path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

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
    @pytest.mark.parametrize("refusal", list(_REFUSAL_REASONS))
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

        _assert_names(excinfo, _REFUSAL_REASONS[refusal], *names)

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
