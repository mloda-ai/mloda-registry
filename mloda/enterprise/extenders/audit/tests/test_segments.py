"""Tests for segment rotation and segment verification."""

from __future__ import annotations

import builtins
import json
import os
import threading
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import mloda.enterprise.extenders.audit.run_manifest as run_manifest_module
from mloda.enterprise.extenders.audit import (
    LogCoverage,
    ManifestVerificationError,
    NdjsonAuditSink,
    RunAlreadySealedError,
    _segments,
    manifest_hash,
    rotate_manifest_key,
    seal_ndjson_runs,
    seal_run,
    verify_ndjson_log,
    verify_ndjson_log_coverage,
    verify_ndjson_segments,
)
from mloda.enterprise.extenders.audit.tests.manifest_helpers import (
    _EXPECTED_GENESIS_KEYS,
    _OTHER_KEY,
    _append_line,
    _assert_raises_and_unchanged,
    _both_algorithms,
    _canonical,
    _genesis_entry,
    _genesis_log,
    _lock_refused,
    _locked_during_digest,
    _ndjson_anchor,
    _patch_bindings,
    _quarantine,
    _quarantine_from_entry,
    _read_lines,
    _record,
    _RecordingAnchor,
    _rewrite_lines,
    _rotate,
    _rotate_segment,
    _sealed_log,
    _sha256,
    _signer,
    _snapshot,
    _write_records,
)


def _archive(path: Path, number: int = 1) -> Path:
    return path.with_name(f"{path.name}.{number:06d}")


def _archives(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.[0-9][0-9][0-9][0-9][0-9][0-9]"))


def _same_inode(first: Path, second: Path) -> bool:
    return first.stat().st_ino == second.stat().st_ino


def _pending_log(directory: Path) -> tuple[Path, Path]:
    """`_genesis_log` plus a pending run-p (two lines), an unattributed line and a blank-run_id line."""
    audit_path, manifest_path = _genesis_log(directory)
    _write_records(audit_path, [_record("run-p", 4), _record(None, 5), _record("run-p", 6), _record("   ", 7)])
    return audit_path, manifest_path


def _rotated_with_stray(directory: Path) -> tuple[Path, Path]:
    """`_pending_log` rotated, then a stray record of run-a, which the archived segment sealed."""
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    _write_records(audit_path, [_record("run-a", 9)])
    return audit_path, manifest_path


class _Crash(BaseException):
    """Stands for the process dying: no `except Exception` cleanup runs."""


def _crash_on_replace(call_number: int) -> Callable[[Any, Any], None]:
    """An os.replace that raises _Crash on its `call_number`th call and otherwise replaces."""
    real_replace = os.replace
    calls: list[Any] = []

    def replace(source: Any, target: Any) -> None:
        calls.append(source)
        if len(calls) == call_number:
            raise _Crash()
        real_replace(source, target)

    return replace


def _crashed_rotation(directory: Path, call_number: int) -> tuple[Path, Path, dict[str, bytes]]:
    """A pending log whose rotation dies on its `call_number`th os.replace; returns the paths and the files before."""
    audit_path, manifest_path = _pending_log(directory)
    before = _snapshot(directory)
    with patch("os.replace", _crash_on_replace(call_number)):
        with pytest.raises(_Crash):
            _rotate_segment(audit_path, manifest_path)
    return audit_path, manifest_path, before


class TestCheckRunAgainstArchivedSeal:
    """A run sealed in an archived segment is checked against the archived manifest and audit file."""

    def test_a_run_sealed_in_an_archived_segment_passes_when_the_archived_pair_matches(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        _rotate_segment(audit_path, manifest_path)

        run_manifest_module._check_run_against_seal(audit_path, manifest_path, "run-a", signer=_signer())

    def test_a_run_sealed_in_an_archived_segment_fails_when_its_archived_records_were_tampered(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        _rotate_segment(audit_path, manifest_path)
        archived_audit = _archive(audit_path)
        lines = _read_lines(archived_audit)
        lines[0]["tenant_id"] = "tampered"
        _rewrite_lines(archived_audit, lines)

        with pytest.raises(ManifestVerificationError) as excinfo:
            run_manifest_module._check_run_against_seal(audit_path, manifest_path, "run-a", signer=_signer())

        assert "has no manifest" not in str(excinfo.value)


def _no_log_id(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    audit_path, manifest_path = _pending_log(directory)
    return audit_path, manifest_path, {"log_id": None}


def _blank_log_id(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    audit_path, manifest_path = _pending_log(directory)
    return audit_path, manifest_path, {"log_id": "   "}


def _missing_manifest_log(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    _write_records(directory / "audit.ndjson", [_record("run-p", 1)])
    return directory / "audit.ndjson", directory / "manifests.ndjson", {}


def _empty_manifest_log(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    audit_path, manifest_path, kwargs = _missing_manifest_log(directory)
    manifest_path.touch()
    return audit_path, manifest_path, kwargs


def _legacy_log(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    audit_path, manifest_path = _sealed_log(directory)
    return audit_path, manifest_path, {}


def _symlinked_audit(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    audit_path, manifest_path = _pending_log(directory)
    real = directory / "real-audit.ndjson"
    audit_path.rename(real)
    audit_path.symlink_to(real)
    return audit_path, manifest_path, {}


def _symlinked_manifest(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    audit_path, manifest_path = _pending_log(directory)
    real = directory / "real-manifests.ndjson"
    manifest_path.rename(real)
    manifest_path.symlink_to(real)
    return audit_path, manifest_path, {}


_UNROTATABLE = {
    "log-id-none": _no_log_id,
    "log-id-blank": _blank_log_id,
    "manifest-missing": _missing_manifest_log,
    "manifest-empty": _empty_manifest_log,
    "legacy-log-without-genesis": _legacy_log,
    "audit-path-is-a-symlink": _symlinked_audit,
    "manifest-path-is-a-symlink": _symlinked_manifest,
}


def _tampered_record(audit_path: Path, manifest_path: Path) -> dict[str, Any]:
    lines = _read_lines(audit_path)
    lines[0]["tenant_id"] = "tampered"
    _rewrite_lines(audit_path, lines)
    return {}


def _stray_record_of_a_sealed_run(audit_path: Path, manifest_path: Path) -> dict[str, Any]:
    _write_records(audit_path, [_record("run-a", 9)])
    return {}


_UNVERIFIABLE = {
    "tampered-record": _tampered_record,
    "stray-record-of-a-sealed-run": _stray_record_of_a_sealed_run,
    "wrong-expected-head": lambda a, m: {"expected_head": _sha256(b"not the head")},
    "unknown-anchored-head": lambda a, m: {"anchored_heads": [_sha256(b"unknown")]},
    "signer-not-current": lambda a, m: {"signer": _signer(_OTHER_KEY, "key-2")},
}


@_both_algorithms
class TestSegmentRotation:
    """rotate_ndjson_segment: archive the outgoing audit and manifest pair, start a new segment chained onto it."""

    def test_the_outgoing_pair_is_archived_byte_for_byte_next_to_the_live_files(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        audit_before, manifest_before = audit_path.read_bytes(), manifest_path.read_bytes()

        _rotate_segment(audit_path, manifest_path)

        assert _archive(audit_path).name == "audit.ndjson.000001"
        assert _archive(manifest_path).name == "manifests.ndjson.000001"
        assert _archive(audit_path).read_bytes() == audit_before
        assert _archive(manifest_path).read_bytes() == manifest_before
        assert audit_path.exists() and manifest_path.exists()

    def test_a_second_rotation_archives_under_the_next_number_and_keeps_the_first_archive(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        _rotate_segment(audit_path, manifest_path)
        first = (_archive(audit_path).read_bytes(), _archive(manifest_path).read_bytes())
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-p")
        audit_before, manifest_before = audit_path.read_bytes(), manifest_path.read_bytes()

        _rotate_segment(audit_path, manifest_path)

        assert (_archive(audit_path).read_bytes(), _archive(manifest_path).read_bytes()) == first
        assert _archive(audit_path, 2).read_bytes() == audit_before
        assert _archive(manifest_path, 2).read_bytes() == manifest_before

    def test_the_next_number_is_one_above_the_highest_archived_manifest(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        _archive(manifest_path, 4).write_bytes(b"retained")

        _rotate_segment(audit_path, manifest_path)

        assert _archives(tmp_path) == ["audit.ndjson.000005", "manifests.ndjson.000004", "manifests.ndjson.000005"]

    def test_the_new_manifest_log_is_one_genesis_chained_onto_the_outgoing_head(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        old_head = manifest_hash(_read_lines(manifest_path)[-1])

        genesis = _rotate_segment(audit_path, manifest_path)

        assert _read_lines(manifest_path) == [genesis]
        assert set(genesis) == _EXPECTED_GENESIS_KEYS
        assert genesis["kind"] == "genesis"
        assert genesis["log_id"] == "log-a"
        assert genesis["previous_manifest_hash"] == old_head
        assert genesis["signature"]["key_id"] == _signer().key_id

    def test_the_new_pair_and_the_archived_pair_each_verify_on_their_own(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        old_head = manifest_hash(_read_lines(manifest_path)[-1])

        genesis = _rotate_segment(audit_path, manifest_path)

        new_head = verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a")
        assert new_head == manifest_hash(genesis)
        archived = verify_ndjson_log(_archive(audit_path), _archive(manifest_path), signer=_signer(), log_id="log-a")
        assert archived == old_head

    def test_only_the_pending_attributed_lines_are_carried_forward_byte_for_byte_in_order(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        lines = audit_path.read_bytes().splitlines(keepends=True)
        pending = [line for line in lines if json.loads(line)["run_id"] == "run-p"]
        assert len(pending) == 2 and len(lines) == 7

        _rotate_segment(audit_path, manifest_path)

        assert audit_path.read_bytes() == b"".join(pending)

    def test_a_carried_run_seals_in_the_new_segment_with_its_carried_and_new_lines(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        _rotate_segment(audit_path, manifest_path)
        _write_records(audit_path, [_record("run-p", 8)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-p")

        assert [manifest["run_id"] for manifest in manifests] == ["run-p"]
        assert manifests[0]["record_count"] == 3
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a")

    @pytest.mark.parametrize("case", list(_UNROTATABLE.values()), ids=list(_UNROTATABLE))
    def test_a_log_that_cannot_rotate_is_refused_and_nothing_changes(
        self, tmp_path: Path, case: Callable[[Path], tuple[Path, Path, dict[str, Any]]]
    ) -> None:
        audit_path, manifest_path, kwargs = case(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError):
            _rotate_segment(audit_path, manifest_path, **kwargs)

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("damage", list(_UNVERIFIABLE.values()), ids=list(_UNVERIFIABLE))
    def test_an_outgoing_segment_that_fails_verification_is_refused_and_nothing_changes(
        self, tmp_path: Path, damage: Callable[[Path, Path], dict[str, Any]]
    ) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        kwargs = damage(audit_path, manifest_path)

        _assert_raises_and_unchanged(tmp_path, lambda: _rotate_segment(audit_path, manifest_path, **kwargs))

    def test_the_head_anchor_gets_the_genesis_hash_once_the_log_holds_it(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        recording = _RecordingAnchor(manifest_path)

        genesis = _rotate_segment(audit_path, manifest_path, head_anchor=recording)

        assert recording.heads == [manifest_hash(genesis)]
        assert recording.last_lines_seen == [manifest_hash(genesis)]

    def test_the_new_segment_accepts_the_old_final_head_as_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        anchor = _ndjson_anchor(tmp_path / "anchor.ndjson")
        old_head = manifest_hash(_read_lines(manifest_path)[-1])
        anchor.write(old_head)

        genesis = _rotate_segment(audit_path, manifest_path, head_anchor=anchor)
        manifests = seal_ndjson_runs(
            audit_path, manifest_path, signer=_signer(), run_id="run-p", anchored_heads=[old_head]
        )

        assert anchor.latest() == manifest_hash(genesis)
        assert [manifest["run_id"] for manifest in manifests] == ["run-p"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a", anchored_heads=[old_head])

    @pytest.mark.parametrize("indexed", [False, True], ids=["no-index", "index"])
    def test_a_targeted_seal_of_a_run_sealed_in_an_archived_segment_is_refused(
        self, tmp_path: Path, indexed: bool
    ) -> None:
        audit_path, manifest_path = _rotated_with_stray(tmp_path)
        extra: dict[str, Any] = {"seal_index_path": tmp_path / "index.sqlite"} if indexed else {}
        if indexed:
            _write_records(audit_path, [_record("run-q", 10)])
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-q", **extra)
        before = (audit_path.read_bytes(), manifest_path.read_bytes())

        with pytest.raises(RunAlreadySealedError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-a", **extra)

        assert (audit_path.read_bytes(), manifest_path.read_bytes()) == before

    @pytest.mark.parametrize(
        "sweep", [{}, {"older_than": timedelta(days=1)}], ids=["blanket-sweep", "older-than-sweep"]
    )
    def test_a_sweep_skips_the_stray_records_of_a_run_sealed_in_an_archived_segment(
        self, tmp_path: Path, sweep: dict[str, Any]
    ) -> None:
        audit_path, manifest_path = _rotated_with_stray(tmp_path)

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), **sweep)

        assert [manifest["run_id"] for manifest in manifests] == ["run-p"]
        assert "run-a" not in {line.get("run_id") for line in _read_lines(manifest_path)}

    def test_a_second_rotation_does_not_carry_the_strays_of_a_run_sealed_in_an_archived_segment(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _rotated_with_stray(tmp_path)
        _write_records(audit_path, [_record("run-q", 10)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-p")

        _rotate_segment(audit_path, manifest_path)

        assert audit_path.read_bytes() == _canonical(_record("run-q", 10)) + b"\n"
        assert _archive(audit_path, 2).exists()

    @pytest.mark.parametrize("indexed", [False, True], ids=["no-index", "index"])
    def test_the_index_answers_the_archive_lookup_without_opening_the_archived_manifest(
        self, tmp_path: Path, indexed: bool
    ) -> None:
        audit_path, manifest_path = _rotated_with_stray(tmp_path)
        index_path = tmp_path / "index.sqlite"
        _write_records(audit_path, [_record("run-q", 10)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-q", seal_index_path=index_path)
        opened: list[str] = []
        real_open = builtins.open

        def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        with patch("builtins.open", spy):
            sealed = run_manifest_module._is_run_sealed_unverified(
                manifest_path, "run-a", index_path if indexed else None
            )

        assert sealed is True
        assert (str(_archive(manifest_path)) in opened) is (not indexed)

    @pytest.mark.parametrize("live", ["audit", "manifest"])
    @pytest.mark.parametrize("operation", ["seal", "rotate-key"])
    def test_an_unrelated_hard_link_to_a_live_file_does_not_block_writers(
        self, tmp_path: Path, live: str, operation: str
    ) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        os.link(audit_path if live == "audit" else manifest_path, tmp_path / "backup-link")
        head = manifest_hash(_read_lines(manifest_path)[-1])

        if operation == "seal":
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-p")
        else:
            rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()], expected_head=head
            )

        assert _read_lines(manifest_path)[-1] != _read_lines(manifest_path)[0]

    @pytest.mark.parametrize("repair", ["damaged-lines", "rotation-entry", "verify-segments"])
    def test_an_interrupted_rotation_refuses_recovery_and_segment_verification(
        self, tmp_path: Path, repair: str
    ) -> None:
        audit_path, manifest_path, _ = _crashed_rotation(tmp_path, 1)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        calls: dict[str, Callable[[], object]] = {
            "damaged-lines": lambda: _quarantine(tmp_path, audit_path, manifest_path),
            "rotation-entry": lambda: _quarantine_from_entry(tmp_path, manifest_path, expected_head=head),
            "verify-segments": lambda: _verify_segments(audit_path, manifest_path),
        }
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError, match="rotate_ndjson_segment"):
            calls[repair]()

        assert _snapshot(tmp_path) == before

    def test_a_sealer_cannot_append_to_the_new_manifest_before_the_genesis_reached_the_anchor(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        locked: list[bool] = []

        class LockProbingAnchor(_RecordingAnchor):
            def write(self, head: str) -> None:
                locked.append(_lock_refused(manifest_path))
                super().write(head)

        _rotate_segment(audit_path, manifest_path, head_anchor=LockProbingAnchor())

        assert locked == [True]

    def test_a_missing_audit_file_with_sealed_runs_is_a_verification_failure_and_changes_nothing(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        audit_path.unlink()

        _assert_raises_and_unchanged(tmp_path, lambda: _rotate_segment(audit_path, manifest_path))

    def test_a_record_of_a_sealed_run_appended_after_verification_refuses_the_rotation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        stray = _canonical(_record("run-a", 9))
        before = _snapshot(tmp_path)
        real_carry = _segments._carry_pending
        appended: list[bool] = []

        def carry(*args: Any, **kwargs: Any) -> Any:
            if not appended:
                appended.append(True)
                _append_line(audit_path, stray)
            return real_carry(*args, **kwargs)

        _patch_bindings(monkeypatch, "_carry_pending", carry)

        with pytest.raises((ManifestVerificationError, ValueError)):
            _rotate_segment(audit_path, manifest_path)

        assert appended
        assert _snapshot(tmp_path) == {**before, "audit.ndjson": before["audit.ndjson"] + stray + b"\n"}
        assert _archives(tmp_path) == []

    def test_a_partial_last_line_completed_before_the_audit_lock_is_carried_when_its_run_is_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _pending_log(tmp_path)
        _append_line(audit_path, _canonical(_record("run-p", 8)))
        data = audit_path.read_bytes()
        audit_path.write_bytes(data[:-1])
        audit_inode = audit_path.stat().st_ino
        completed: list[bool] = []
        real_flock = fcntl.flock

        def flock(fd: int, operation: int) -> None:
            if operation & fcntl.LOCK_EX and os.fstat(fd).st_ino == audit_inode and not completed:
                completed.append(True)
                with open(audit_path, "ab") as file:
                    file.write(b"\n")
            real_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", flock)

        _rotate_segment(audit_path, manifest_path)

        assert completed
        assert [line["run_id"] for line in _read_lines(audit_path)] == ["run-p", "run-p", "run-p"]
        assert audit_path.read_bytes().endswith(b"\n")

    @pytest.mark.parametrize("blocked", ["seal", "rotate-key"])
    def test_a_crash_before_the_first_swap_leaves_both_live_files_linked_and_blocks_writers(
        self, tmp_path: Path, blocked: str
    ) -> None:
        audit_path, manifest_path, _ = _crashed_rotation(tmp_path, 1)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        calls: dict[str, Callable[[], object]] = {
            "seal": lambda: seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-p"),
            "rotate-key": lambda: rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()], expected_head=head
            ),
        }
        before = _snapshot(tmp_path)

        for live in (audit_path, manifest_path):
            assert live.stat().st_nlink > 1
            assert _same_inode(live, _archive(live))
        with pytest.raises(ValueError, match="rotate_ndjson_segment"):
            calls[blocked]()

        assert _snapshot(tmp_path) == before

    def test_rotating_again_after_a_crash_before_the_first_swap_completes_under_the_same_number(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, before = _crashed_rotation(tmp_path, 1)
        old_head = manifest_hash(_read_lines(manifest_path)[-1])

        genesis = _rotate_segment(audit_path, manifest_path)

        assert _archives(tmp_path) == [
            "audit.ndjson.000001",
            "manifests.ndjson.000001",
        ]
        assert _archive(audit_path).read_bytes() == before["audit.ndjson"]
        assert _archive(manifest_path).read_bytes() == before["manifests.ndjson"]
        assert _read_lines(manifest_path) == [genesis]
        assert genesis["previous_manifest_hash"] == old_head
        assert audit_path.stat().st_nlink == manifest_path.stat().st_nlink == 1
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a")

    def test_after_a_crash_between_the_swaps_writes_land_in_the_new_audit_and_sealing_is_blocked(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, before = _crashed_rotation(tmp_path, 2)

        assert not _same_inode(audit_path, _archive(audit_path))
        assert _same_inode(manifest_path, _archive(manifest_path))
        _write_records(audit_path, [_record("run-n", 20)])
        archived_audit = _archive(audit_path).read_bytes()
        manifest_before = manifest_path.read_bytes()
        with pytest.raises(ValueError, match="rotate_ndjson_segment"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-n")

        assert archived_audit == before["audit.ndjson"]
        assert manifest_path.read_bytes() == manifest_before
        assert [line["run_id"] for line in _read_lines(audit_path)] == ["run-p", "run-p", "run-n"]

    def test_rotating_again_after_a_crash_between_the_swaps_chains_the_genesis_and_loses_no_record(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, before = _crashed_rotation(tmp_path, 2)
        _write_records(audit_path, [_record("run-n", 20)])
        archived_head = manifest_hash(_read_lines(_archive(manifest_path))[-1])

        genesis = _rotate_segment(audit_path, manifest_path)

        assert genesis["previous_manifest_hash"] == archived_head
        assert _read_lines(manifest_path) == [genesis]
        assert _archive(manifest_path).read_bytes() == before["manifests.ndjson"]
        assert _archive(audit_path).read_bytes() == before["audit.ndjson"]
        assert _archives(tmp_path) == ["audit.ndjson.000001", "manifests.ndjson.000001"]
        assert [line["run_id"] for line in _read_lines(audit_path)] == ["run-p", "run-p", "run-n"]
        assert manifest_path.stat().st_nlink == 1
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a")

    def test_a_sink_write_that_opened_the_old_audit_file_before_the_swap_lands_in_the_new_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _pending_log(tmp_path)
        before = audit_path.read_bytes()
        old_inode = audit_path.stat().st_ino
        entered, started = threading.Event(), threading.Event()
        errors: list[BaseException] = []

        def write_record() -> None:
            try:
                NdjsonAuditSink(audit_path).write(_record("run-sink", 20))
            except BaseException as error:  # noqa: BLE001 - surfaced by the assertion below
                errors.append(error)

        writer = threading.Thread(target=write_record)
        real_flock = fcntl.flock

        def flock(fd: int, operation: int) -> None:
            on_old_file = os.fstat(fd).st_ino == old_inode
            if on_old_file and threading.current_thread() is writer:
                entered.set()
            real_flock(fd, operation)
            if on_old_file and operation & fcntl.LOCK_EX and not started.is_set():
                # Rotation holds the audit file exclusively: start a writer that opened the old file, wait until it
                # reached its lock, and let the rotation go on to swap the file.
                started.set()
                writer.start()
                entered.wait(5)

        monkeypatch.setattr(fcntl, "flock", flock)

        _rotate_segment(audit_path, manifest_path)
        writer.join(10)

        assert not errors
        assert started.is_set() and not writer.is_alive()
        assert _archive(audit_path).read_bytes() == before
        assert [line["run_id"] for line in _read_lines(audit_path)] == ["run-p", "run-p", "run-sink"]

    def test_a_record_of_a_run_sealed_in_the_outgoing_segment_during_the_copy_pass_refuses_the_rotation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _pending_log(tmp_path)
        stray = _canonical(_record("run-a", 9))
        before = _snapshot(tmp_path)
        audit_inode = audit_path.stat().st_ino
        appended: list[bool] = []
        real_flock = fcntl.flock

        def flock(fd: int, operation: int) -> None:
            if operation & fcntl.LOCK_EX and os.fstat(fd).st_ino == audit_inode and not appended:
                # The copy pass is over and the audit lock is about to be taken: a writer lands a stray line now.
                appended.append(True)
                _append_line(audit_path, stray)
            real_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", flock)

        with pytest.raises(ValueError):
            _rotate_segment(audit_path, manifest_path)

        assert appended
        assert _snapshot(tmp_path) == {**before, "audit.ndjson": before["audit.ndjson"] + stray + b"\n"}
        assert _archives(tmp_path) == []


def _verify_segments(audit_path: Path, manifest_path: Path, **kwargs: Any) -> LogCoverage:
    """verify_ndjson_segments with the default signer."""
    coverage: LogCoverage = verify_ndjson_segments(audit_path, manifest_path, **{"signer": _signer(), **kwargs})
    return coverage


def _three_segments(directory: Path) -> tuple[Path, Path]:
    """Two rotations. Segment 1 seals a, b, c; segment 2 seals run-q; the live segment seals run-p, which was pending
    in segments 1 and 2 and was carried across both rotations, and leaves run-r pending."""
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    _write_records(audit_path, [_record("run-q", 10), _record(None, 11)])
    seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-q")
    _rotate_segment(audit_path, manifest_path)
    _write_records(audit_path, [_record("run-p", 12), _record("run-r", 13)])
    seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-p")
    return audit_path, manifest_path


def _head_of(manifest_path: Path, index: int = -1) -> str:
    return manifest_hash(_read_lines(manifest_path)[index])


def _middle_segment_deleted(directory: Path) -> dict[str, Any]:
    audit_path, manifest_path = _three_segments(directory)
    _archive(audit_path, 2).unlink()
    _archive(manifest_path, 2).unlink()
    return {}


def _genesis_under_another_key(directory: Path) -> dict[str, Any]:
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    key_2 = _signer(_OTHER_KEY, "key-2")
    manifest_path.unlink()
    _write_records(manifest_path, [_genesis_entry(key_2, previous_manifest_hash=_head_of(_archive(manifest_path)))])
    return {"signer": key_2, "previous_signers": [_signer()]}


def _run_sealed_twice(directory: Path) -> dict[str, Any]:
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    record = _record("run-a", 9)
    _write_records(audit_path, [record])
    seal = seal_run([record], run_id="run-a", signer=_signer(), previous_manifest_hash=_head_of(manifest_path))
    _append_line(manifest_path, _canonical(seal))
    return {}


def _record_stranded_in_the_archive(directory: Path) -> dict[str, Any]:
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    _append_line(_archive(audit_path), _canonical(_record("run-p", 30)))
    return {}


def _carried_line_dropped(directory: Path) -> dict[str, Any]:
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    audit_path.write_bytes(audit_path.read_bytes().splitlines(keepends=True)[0])
    return {}


def _sealed_run_reappears(directory: Path) -> dict[str, Any]:
    _rotated_with_stray(directory)
    return {}


def _unknown_anchor(directory: Path) -> dict[str, Any]:
    _three_segments(directory)
    return {"anchored_heads": [_sha256(b"unknown")]}


def _wrong_expected_head(directory: Path) -> dict[str, Any]:
    _three_segments(directory)
    return {"expected_head": _sha256(b"not the head")}


def _old_segment_head_as_expected_head(directory: Path) -> dict[str, Any]:
    _, manifest_path = _three_segments(directory)
    return {"expected_head": _head_of(_archive(manifest_path))}


def _wrong_log_id(directory: Path) -> dict[str, Any]:
    _three_segments(directory)
    return {"log_id": "log-other"}


def _signer_not_current(directory: Path) -> dict[str, Any]:
    _three_segments(directory)
    return {"signer": _signer(_OTHER_KEY, "key-2")}


def _successor_genesis_names_another_log_id(directory: Path) -> dict[str, Any]:
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    manifest_path.unlink()
    genesis = _genesis_entry(_signer(), "log-b", previous_manifest_hash=_head_of(_archive(manifest_path)))
    _write_records(manifest_path, [genesis])
    return {"log_id": None}


_SEGMENT_FAILURES = {
    "middle-segment-deleted": (_middle_segment_deleted, "manifest chain is broken"),
    "genesis-under-another-key": (_genesis_under_another_key, "genesis entry is signed by key"),
    "run-sealed-in-two-segments": (_run_sealed_twice, "sealed by more than one manifest"),
    "carried-run-stranded-in-the-archive": (_record_stranded_in_the_archive, "stranded"),
    "carried-line-dropped-from-the-next-segment": (_carried_line_dropped, "stranded"),
    "sealed-run-reappears-in-a-later-segment": (_sealed_run_reappears, "run_id 'run-a' has records"),
    "unknown-anchored-head": (_unknown_anchor, "anchored head .* is not a line"),
    "wrong-expected-head": (_wrong_expected_head, "is not the expected head"),
    "old-segment-head-is-not-the-expected-head": (_old_segment_head_as_expected_head, "is not the expected head"),
    "wrong-log-id": (_wrong_log_id, "manifest log is for log_id"),
    "signer-not-current": (_signer_not_current, "not the signer's"),
    "successor-genesis-names-another-log-id": (_successor_genesis_names_another_log_id, "log_id"),
}


@_both_algorithms
class TestVerifyNdjsonSegments:
    """verify_ndjson_segments: the retained archived pairs in order plus the live pair, as one chain."""

    def test_three_segments_with_a_run_carried_across_both_rotations_pass_and_aggregate(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _three_segments(tmp_path)

        coverage = _verify_segments(audit_path, manifest_path, log_id="log-a")

        assert coverage == LogCoverage(
            head=_head_of(manifest_path),
            sealed_runs=5,
            sealed_lines=7,
            unattributed_lines=3,
            unsealed_lines={"run-r": 1},
        )

    def test_without_a_rotation_it_matches_verify_ndjson_log_coverage(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        _write_records(audit_path, [_record("run-p", 4), _record(None, 5)])

        coverage = _verify_segments(audit_path, manifest_path)

        assert coverage == verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

    def test_dropping_the_oldest_pair_by_retention_still_passes(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _three_segments(tmp_path)
        _archive(audit_path).unlink()
        _archive(manifest_path).unlink()

        coverage = _verify_segments(audit_path, manifest_path)

        assert coverage.head == _head_of(manifest_path)
        assert coverage.sealed_runs == 2

    def test_a_key_rotation_inside_a_segment_followed_by_a_segment_rotation_passes(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        key_1, key_2 = _signer(), _signer(_OTHER_KEY, "key-2")
        _rotate(manifest_path, key_2, key_1)
        _write_records(audit_path, [_record("run-d", 4)])
        seal_ndjson_runs(audit_path, manifest_path, signer=key_2, previous_signers=[key_1])
        _rotate_segment(audit_path, manifest_path, signer=key_2, previous_signers=[key_1])
        _write_records(audit_path, [_record("run-e", 5)])
        seal_ndjson_runs(audit_path, manifest_path, signer=key_2, previous_signers=[key_1])

        coverage = _verify_segments(audit_path, manifest_path, signer=key_2, previous_signers=[key_1])

        assert coverage.head == _head_of(manifest_path)
        assert coverage.sealed_runs == 5

    @pytest.mark.parametrize("build, message", list(_SEGMENT_FAILURES.values()), ids=list(_SEGMENT_FAILURES))
    def test_a_broken_history_fails_and_changes_nothing(
        self, tmp_path: Path, build: Callable[[Path], dict[str, Any]], message: str
    ) -> None:
        kwargs = build(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError, match=message):
            _verify_segments(tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson", **kwargs)

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize(
        "anchor",
        [
            lambda a, m: _head_of(_archive(m, 1)),
            lambda a, m: _head_of(_archive(m, 1), 0),
            lambda a, m: _head_of(_archive(m, 2)),
            lambda a, m: _head_of(m, 0),
        ],
        ids=["first-segment-head", "first-genesis", "middle-segment-head", "live-genesis"],
    )
    def test_an_anchored_head_may_be_a_line_of_any_retained_segment(
        self, tmp_path: Path, anchor: Callable[[Path, Path], str]
    ) -> None:
        audit_path, manifest_path = _three_segments(tmp_path)

        coverage = _verify_segments(
            audit_path, manifest_path, anchored_heads=[anchor(audit_path, manifest_path)], log_id="log-a"
        )

        assert coverage.head == _head_of(manifest_path)

    def test_the_expected_head_is_the_live_segments_head(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _three_segments(tmp_path)

        coverage = _verify_segments(audit_path, manifest_path, expected_head=_head_of(manifest_path))

        assert coverage.head == _head_of(manifest_path)

    def test_the_live_manifest_is_shared_locked_while_the_archives_and_audit_files_are_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path = _three_segments(tmp_path)
        locked_during_read = _locked_during_digest(monkeypatch, manifest_path)

        _verify_segments(audit_path, manifest_path)

        assert locked_during_read and all(locked_during_read)
