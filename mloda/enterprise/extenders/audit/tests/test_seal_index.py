"""Tests for the opt-in seal index."""

from __future__ import annotations

import json
import logging
import os
import stat
import sys
from collections.abc import Callable, Iterable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from mloda.enterprise.extenders.audit import (
    ManifestSigner,
    ManifestVerificationError,
    RunAlreadySealedError,
    RunNotPendingError,
    _core,
    _seal_index,
    seal_ndjson_runs,
    verify_ndjson_log,
)
from mloda.enterprise.extenders.audit.tests.manifest_helpers import (
    _OTHER_KEY,
    _aliased,
    _append_line,
    _assert_raises_and_unchanged,
    _both_algorithms,
    _HookedSigner,
    _log_heads,
    _ndjson_anchor,
    _patch_bindings,
    _quarantine,
    _read_lines,
    _record,
    _RecordingAnchor,
    _rotate,
    _rotate_segment,
    _signer,
    _snapshot,
    _unpatch_bindings,
    _write_records,
)
from mloda.enterprise.extenders.audit.tests.test_audit_extender import (
    _ReadSpy,
)

_INDEX = "seal.index"


_AUDIT_LOGGER = "mloda.enterprise.extenders.audit.run_manifest"


def _indexed_log(directory: Path, prior: int = 4, *, indexed: bool = True) -> tuple[Path, Path, Path]:
    """`prior` runs, each sealed on its own (through seal_index_path if `indexed`); returns audit, manifest and index paths."""
    directory.mkdir()
    audit_path, manifest_path, index_path = (
        directory / "audit.ndjson",
        directory / "manifests.ndjson",
        directory / _INDEX,
    )
    extra: dict[str, Any] = {"seal_index_path": index_path} if indexed else {}
    for number in range(prior):
        _write_records(audit_path, [_record(f"run-{number}", number)])
        seal_ndjson_runs(
            audit_path,
            manifest_path,
            signer=_signer(),
            run_id=f"run-{number}",
            **extra,
        )
    return audit_path, manifest_path, index_path


def _copy_dir(source: Path, target: Path) -> None:
    target.mkdir()
    for path in source.iterdir():
        (target / path.name).write_bytes(path.read_bytes())


def _count_seal(
    audit_path: Path, manifest_path: Path, run_id: str, *, signer: ManifestSigner | None = None, **kwargs: Any
) -> int:
    """Signature verifications one seal of `run_id` performs."""
    calls: list[bytes] = []
    hooked = _HookedSigner(signer or _signer(), on_verify=calls.append)
    seal_ndjson_runs(audit_path, manifest_path, signer=hooked, run_id=run_id, **kwargs)
    return len(calls)


def _full_count(directory: Path, run_id: str, **kwargs: Any) -> int:
    """Verifications a seal of `run_id` needs without the index, measured on a copy of `directory`."""
    reference = directory.parent / f"ref-{directory.name}"
    _copy_dir(directory, reference)
    return _count_seal(reference / "audit.ndjson", reference / "manifests.ndjson", run_id, **kwargs)


def _last_line_start(path: Path) -> int:
    return path.read_bytes().rstrip(b"\n").rfind(b"\n") + 1


def _replace_manifest_by_copy(audit_path: Path, manifest_path: Path, index_path: Path) -> None:
    copy = manifest_path.with_name("copy.tmp")
    copy.write_bytes(manifest_path.read_bytes())
    os.replace(copy, manifest_path)


def _truncate_below_checkpoint(audit_path: Path, manifest_path: Path, index_path: Path) -> None:
    os.truncate(manifest_path, _last_line_start(manifest_path))


def _swap_last_line_in_place(audit_path: Path, manifest_path: Path, index_path: Path) -> None:
    """Same inode, same size, same prefix, but a different last line than the checkpoint head."""
    size = manifest_path.stat().st_size
    os.truncate(manifest_path, _last_line_start(manifest_path))
    _write_records(audit_path, [_record("xun-3", 3)])
    seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="xun-3")
    assert manifest_path.stat().st_size == size


_INDEX_FALLBACKS: dict[str, Callable[[Path, Path, Path], object]] = {
    "missing-index": lambda a, m, i: i.unlink(),
    "garbage-index": lambda a, m, i: i.write_bytes(b"SQLite format 3\x00" + b"\xff" * 4000),
    "manifest-replaced-by-a-copy": _replace_manifest_by_copy,
    "manifest-truncated-below-the-checkpoint": _truncate_below_checkpoint,
    "head-differs-from-the-line-at-the-offset": _swap_last_line_in_place,
}


@_both_algorithms
class TestSealIndex:
    """seal_index_path: opt-in signed checkpoint, so a seal verifies only the manifest lines after it."""

    def test_a_seal_verifies_only_lines_after_the_checkpoint_whatever_the_history(self, tmp_path: Path) -> None:
        counts = []
        for prior in (2, 8):
            audit_path, manifest_path, index_path = _indexed_log(tmp_path / f"p{prior}", prior)
            _write_records(audit_path, [_record("run-next", 99)])
            counts.append(_count_seal(audit_path, manifest_path, "run-next", seal_index_path=index_path))

        assert counts[0] == counts[1]

    def test_without_seal_index_path_the_verifications_grow_with_the_history(self, tmp_path: Path) -> None:
        counts = []
        for prior in (2, 8):
            audit_path, manifest_path, _ = _indexed_log(tmp_path / f"p{prior}", prior, indexed=False)
            _write_records(audit_path, [_record("run-next", 99)])
            counts.append(_count_seal(audit_path, manifest_path, "run-next"))

        assert counts[0] < counts[1]

    def test_the_seal_line_equals_the_full_scan_one_except_sealed_at_and_signature(self, tmp_path: Path) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        _write_records(audit_path, [_record("run-next", 99)])
        _copy_dir(tmp_path / "live", tmp_path / "ref")

        fast = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), seal_index_path=index_path)
        full = seal_ndjson_runs(
            tmp_path / "ref" / "audit.ndjson", tmp_path / "ref" / "manifests.ndjson", signer=_signer()
        )

        def stable(manifest: Mapping[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in manifest.items() if k not in ("sealed_at", "signature")}

        assert [stable(m) for m in fast] == [stable(m) for m in full]
        assert stable(_read_lines(manifest_path)[-1]) == stable(fast[0])
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("tamper", list(_INDEX_FALLBACKS.values()), ids=list(_INDEX_FALLBACKS))
    def test_an_unusable_index_falls_back_to_full_verification_and_is_rebuilt(
        self, tmp_path: Path, tamper: Callable[[Path, Path, Path], object]
    ) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        tamper(audit_path, manifest_path, index_path)
        _write_records(audit_path, [_record("run-next", 99), _record("run-after", 100)])
        full = _full_count(tmp_path / "live", "run-next")

        count = _count_seal(audit_path, manifest_path, "run-next", seal_index_path=index_path)

        assert count >= full
        assert _read_lines(manifest_path)[-1]["run_id"] == "run-next"
        assert _count_seal(audit_path, manifest_path, "run-after", seal_index_path=index_path) < count

    def test_a_checkpoint_signed_by_a_retired_key_falls_back_to_full_verification(self, tmp_path: Path) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        key_1, key_2 = _signer(), _signer(_OTHER_KEY, "key-2")
        _rotate(manifest_path, key_2, key_1)
        _write_records(audit_path, [_record("run-next", 99), _record("run-after", 100)])
        extra: dict[str, Any] = {"signer": key_2, "previous_signers": [key_1]}
        full = _full_count(tmp_path / "live", "run-next", **extra)

        count = _count_seal(audit_path, manifest_path, "run-next", seal_index_path=index_path, **extra)

        assert count >= full
        assert _read_lines(manifest_path)[-1]["signature"]["key_id"] == "key-2"
        assert _count_seal(audit_path, manifest_path, "run-after", seal_index_path=index_path, **extra) < count

    @pytest.mark.parametrize("anchor", ["all-heads", "latest-head"])
    def test_a_rollback_with_an_older_index_copy_is_caught_by_an_anchor(self, tmp_path: Path, anchor: str) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        old_manifest, old_index = manifest_path.read_bytes(), index_path.read_bytes()
        recording = _RecordingAnchor()
        for name in ("run-x", "run-y"):
            _write_records(audit_path, [_record(name, 50)])
            seal_ndjson_runs(
                audit_path,
                manifest_path,
                signer=_signer(),
                run_id=name,
                seal_index_path=index_path,
                head_anchor=recording,
            )
        manifest_path.write_bytes(old_manifest)
        index_path.write_bytes(old_index)
        _write_records(audit_path, [_record("run-z", 60)])
        heads = recording.heads if anchor == "all-heads" else recording.heads[-1:]

        _assert_raises_and_unchanged(
            tmp_path / "live",
            lambda: seal_ndjson_runs(
                audit_path,
                manifest_path,
                signer=_signer(),
                run_id="run-z",
                seal_index_path=index_path,
                anchored_heads=heads,
            ),
        )

    def test_a_line_tampered_before_the_checkpoint_is_missed_by_the_seal_but_caught_offline(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        manifest_path.write_bytes(manifest_path.read_bytes().replace(b"run-0", b"run-7", 1))
        _write_records(audit_path, [_record("run-next", 99)])
        _copy_dir(tmp_path / "live", tmp_path / "ref")
        with pytest.raises(ManifestVerificationError):
            seal_ndjson_runs(
                tmp_path / "ref" / "audit.ndjson",
                tmp_path / "ref" / "manifests.ndjson",
                signer=_signer(),
                run_id="run-next",
            )

        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-next", seal_index_path=index_path)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_an_anchor_older_than_the_checkpoint_falls_back_to_full_verification(self, tmp_path: Path) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        heads = _log_heads(manifest_path)
        _write_records(audit_path, [_record("run-next", 99)])
        full = _full_count(tmp_path / "live", "run-next")

        count = _count_seal(
            audit_path, manifest_path, "run-next", seal_index_path=index_path, anchored_heads=[heads[1]]
        )

        assert count >= full
        assert _read_lines(manifest_path)[-1]["run_id"] == "run-next"

    def test_an_anchor_at_the_checkpoint_head_keeps_the_fast_path(self, tmp_path: Path) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        heads = _log_heads(manifest_path)
        _write_records(audit_path, [_record("run-next", 99)])
        full = _full_count(tmp_path / "live", "run-next")

        count = _count_seal(
            audit_path, manifest_path, "run-next", seal_index_path=index_path, anchored_heads=[heads[-1]]
        )

        assert count < full

    def test_a_crash_between_the_append_and_the_index_commit_is_healed(self, tmp_path: Path) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        stale_index = index_path.read_bytes()
        _write_records(audit_path, [_record("run-crash", 98)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-crash", seal_index_path=index_path)
        index_path.write_bytes(stale_index)
        _write_records(audit_path, [_record("run-next", 99)])
        full = _full_count(tmp_path / "live", "run-next")

        count = _count_seal(audit_path, manifest_path, "run-next", seal_index_path=index_path)

        assert count < full
        assert [m["run_id"] for m in _read_lines(manifest_path)][-2:] == ["run-crash", "run-next"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        with pytest.raises(RunAlreadySealedError):
            seal_ndjson_runs(
                audit_path, manifest_path, signer=_signer(), run_id="run-crash", seal_index_path=index_path
            )

    def test_an_index_write_error_after_the_append_is_logged_and_does_not_fail_the_seal(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        audit_path, manifest_path, _ = _indexed_log(tmp_path / "live")
        broken = tmp_path / "live" / "index-is-a-directory"
        broken.mkdir()
        _write_records(audit_path, [_record("run-next", 99)])

        with caplog.at_level(logging.WARNING, logger=_AUDIT_LOGGER):
            manifests = seal_ndjson_runs(
                audit_path, manifest_path, signer=_signer(), run_id="run-next", seal_index_path=broken
            )

        assert [m["run_id"] for m in manifests] == ["run-next"]
        assert _read_lines(manifest_path)[-1]["run_id"] == "run-next"
        assert len([r for r in caplog.records if r.name == _AUDIT_LOGGER and r.levelno == logging.WARNING]) == 1

    @pytest.mark.parametrize("index_exists", [True, False], ids=["index-present", "index-absent"])
    @pytest.mark.parametrize("failure", ["not-pending", "bad-tail-line", "already-sealed"])
    def test_a_failing_seal_neither_creates_nor_modifies_the_index(
        self, tmp_path: Path, failure: str, index_exists: bool
    ) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        if not index_exists:
            index_path.unlink()
        run_id = {"not-pending": "run-nowhere", "bad-tail-line": "run-0", "already-sealed": "run-0"}[failure]
        if failure == "bad-tail-line":
            _append_line(manifest_path, "not json")
        expected = {
            "not-pending": RunNotPendingError,
            "bad-tail-line": ManifestVerificationError,
            "already-sealed": RunAlreadySealedError,
        }[failure]
        before = _snapshot(tmp_path / "live")

        with pytest.raises(expected):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id=run_id, seal_index_path=index_path)

        assert _snapshot(tmp_path / "live") == before

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    @pytest.mark.parametrize("target", ["audit", "manifest", "anchor"])
    def test_an_index_path_aliasing_another_path_is_refused(self, tmp_path: Path, target: str, spelling: str) -> None:
        audit_path, manifest_path = tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson"
        anchor_path = tmp_path / "anchor.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])
        files = {"audit": audit_path, "manifest": manifest_path, "anchor": anchor_path}
        files[target].touch()
        index_path = _aliased(tmp_path, files[target], spelling)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            seal_ndjson_runs(
                audit_path,
                manifest_path,
                signer=_signer(),
                seal_index_path=index_path,
                head_anchor=_ndjson_anchor(anchor_path),
            )

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    def test_a_run_sealed_before_the_checkpoint_is_refused_on_the_fast_path(self, tmp_path: Path) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        calls: list[bytes] = []
        hooked = _HookedSigner(_signer(), on_verify=calls.append)

        with pytest.raises(RunAlreadySealedError):
            seal_ndjson_runs(audit_path, manifest_path, signer=hooked, run_id="run-0", seal_index_path=index_path)

        assert len(calls) < 4

    def test_a_hint_that_points_at_a_line_not_naming_the_run_falls_back_to_the_full_scan(self, tmp_path: Path) -> None:
        import sqlite3

        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        connection = sqlite3.connect(index_path)
        try:
            connection.execute("UPDATE hint SET line_start = 0 WHERE run_id = ?", ("run-3",))
            connection.commit()
        finally:
            connection.close()

        with pytest.raises(RunAlreadySealedError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-3", seal_index_path=index_path)

    def test_no_wal_or_shm_files_are_left_next_to_the_index(self, tmp_path: Path) -> None:
        _indexed_log(tmp_path / "live")

        names = {path.name for path in (tmp_path / "live").iterdir()}

        assert names == {"audit.ndjson", "manifests.ndjson", _INDEX}

    # Audit side: the index also remembers how far the audit file was scanned and where each unsealed run starts.

    @staticmethod
    def _count_audit_parses(monkeypatch: pytest.MonkeyPatch, audit_path: Path) -> list[str]:
        """Every audit line decoded from now on (the decode of any other file is not counted)."""
        seen: list[str] = []
        real = _core._decode_line

        def counting(where: str, line: bytes) -> Any:
            if where.startswith(str(audit_path)):
                seen.append(where)
            return real(where, line)

        _patch_bindings(monkeypatch, "_decode_line", counting)
        return seen

    @staticmethod
    def _stable(manifests: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in m.items() if k not in ("sealed_at", "signature")} for m in manifests]

    @staticmethod
    def _chainless(manifests: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in m.items() if k != "previous_manifest_hash"} for m in TestSealIndex._stable(manifests)]

    def _audit_parses_of_one_seal(
        self, monkeypatch: pytest.MonkeyPatch, directory: Path, prior: int, *, indexed: bool
    ) -> int:
        audit_path, manifest_path, index_path = _indexed_log(directory, prior, indexed=indexed)
        _write_records(audit_path, [_record("run-next", 99)])
        seen = self._count_audit_parses(monkeypatch, audit_path)
        extra: dict[str, Any] = {"seal_index_path": index_path} if indexed else {}
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-next", **extra)
        return len(seen)

    def test_a_seal_parses_only_the_audit_lines_of_the_run_whatever_the_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        counts = [
            self._audit_parses_of_one_seal(monkeypatch, tmp_path / f"p{prior}", prior, indexed=True) for prior in (2, 8)
        ]

        assert counts[0] == counts[1]

    def test_without_seal_index_path_the_audit_lines_parsed_grow_with_the_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        counts = [
            self._audit_parses_of_one_seal(monkeypatch, tmp_path / f"p{prior}", prior, indexed=False)
            for prior in (2, 8)
        ]

        assert counts[0] < counts[1]

    def test_a_run_interleaved_with_an_earlier_sealed_run_still_gets_all_its_records_sealed(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        _write_records(
            audit_path,
            [_record("run-a", 1), _record("run-b", 2), _record("run-a", 3), _record("run-b", 4)],
        )
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-b", seal_index_path=index_path)
        _write_records(audit_path, [_record("run-a", 5)])
        _copy_dir(tmp_path / "live", tmp_path / "ref")

        fast = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-a", seal_index_path=index_path)
        full = seal_ndjson_runs(
            tmp_path / "ref" / "audit.ndjson", tmp_path / "ref" / "manifests.ndjson", signer=_signer(), run_id="run-a"
        )

        assert fast[0]["record_count"] == 3
        assert self._stable(fast) == self._stable(full)
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("later", ["older-than-sweep", "targeted-seal"])
    def test_a_crashed_run_stays_pending_and_can_still_be_sealed_with_all_its_records(
        self, tmp_path: Path, later: str
    ) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        _write_records(audit_path, [_record("run-crash", 10), _record("run-crash", 11)])
        for number in range(2):
            _write_records(audit_path, [_record(f"run-late-{number}", 20 + number)])
            seal_ndjson_runs(
                audit_path,
                manifest_path,
                signer=_signer(),
                run_id=f"run-late-{number}",
                seal_index_path=index_path,
            )
        assert "run-crash" not in [m["run_id"] for m in _read_lines(manifest_path)]
        _copy_dir(tmp_path / "live", tmp_path / "ref")
        kwargs: dict[str, Any] = (
            {"older_than": timedelta(0)} if later == "older-than-sweep" else {"run_id": "run-crash"}
        )

        fast = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), seal_index_path=index_path, **kwargs)
        full = seal_ndjson_runs(
            tmp_path / "ref" / "audit.ndjson", tmp_path / "ref" / "manifests.ndjson", signer=_signer(), **kwargs
        )

        assert [m["run_id"] for m in fast] == ["run-crash"]
        assert fast[0]["record_count"] == 2
        assert self._stable(fast) == self._stable(full)

    @pytest.mark.parametrize("stale", ["replaced-by-a-copy", "truncated-below-the-offset", "last-line-changed"])
    def test_a_stale_audit_checkpoint_falls_back_to_a_full_audit_scan_with_the_same_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale: str
    ) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        if stale == "replaced-by-a-copy":
            copy = audit_path.with_name("copy.tmp")
            copy.write_bytes(audit_path.read_bytes())
            os.replace(copy, audit_path)
        elif stale == "truncated-below-the-offset":
            os.truncate(audit_path, _last_line_start(audit_path))
        else:
            audit_path.write_bytes(audit_path.read_bytes().replace(b"run-3", b"run-9"))
        _write_records(audit_path, [_record("run-next", 99)])
        _copy_dir(tmp_path / "live", tmp_path / "ref")
        reference = tmp_path / "ref" / "audit.ndjson"
        full_seen = self._count_audit_parses(monkeypatch, reference)
        full = seal_ndjson_runs(reference, tmp_path / "ref" / "manifests.ndjson", signer=_signer(), run_id="run-next")
        seen = self._count_audit_parses(monkeypatch, audit_path)

        fast = seal_ndjson_runs(
            audit_path, manifest_path, signer=_signer(), run_id="run-next", seal_index_path=index_path
        )

        assert len(seen) >= len(full_seen) > 2
        assert self._stable(fast) == self._stable(full)

    def test_a_seal_after_a_quarantine_repair_rewrote_the_audit_file_is_correct(self, tmp_path: Path) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        _append_line(audit_path, "not json")
        _write_records(audit_path, [_record("run-next", 99)])
        _quarantine(tmp_path / "live", audit_path, manifest_path)
        _write_records(audit_path, [_record("run-next", 100)])
        _copy_dir(tmp_path / "live", tmp_path / "ref")

        fast = seal_ndjson_runs(
            audit_path, manifest_path, signer=_signer(), run_id="run-next", seal_index_path=index_path
        )
        full = seal_ndjson_runs(
            tmp_path / "ref" / "audit.ndjson",
            tmp_path / "ref" / "manifests.ndjson",
            signer=_signer(),
            run_id="run-next",
        )

        assert fast[0]["record_count"] == 2
        assert self._stable(fast) == self._stable(full)
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_a_blanket_sweep_with_the_index_seals_everything_and_the_next_targeted_seal_is_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        _write_records(audit_path, [_record("run-a", 30), _record("run-b", 31), _record("run-a", 32)])
        _copy_dir(tmp_path / "live", tmp_path / "ref")

        fast = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), seal_index_path=index_path)
        full = seal_ndjson_runs(
            tmp_path / "ref" / "audit.ndjson", tmp_path / "ref" / "manifests.ndjson", signer=_signer()
        )
        _write_records(audit_path, [_record("run-c", 40)])
        seen = self._count_audit_parses(monkeypatch, audit_path)
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-c", seal_index_path=index_path)

        assert self._chainless(fast) == self._chainless(full)
        assert len(seen) <= 2
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("indexed", [True, False], ids=["with-index", "without-index"])
    def test_an_unterminated_audit_tail_is_refused_as_without_the_index(self, tmp_path: Path, indexed: bool) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live", indexed=indexed)
        _write_records(audit_path, [_record("run-next", 99)])
        with open(audit_path, "ab") as file:
            file.write(b'{"run_id": "run-z"')
        extra: dict[str, Any] = {"seal_index_path": index_path} if indexed else {}
        before = _snapshot(tmp_path / "live")

        with pytest.raises(ManifestVerificationError, match="newline"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-next", **extra)

        assert _snapshot(tmp_path / "live") == before

    def test_the_whole_seal_costs_the_same_whatever_the_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        costs = []
        for prior in (2, 8):
            audit_path, manifest_path, index_path = _indexed_log(tmp_path / f"p{prior}", prior)
            _write_records(audit_path, [_record("run-next", 99)])
            verifications: list[bytes] = []
            parses = self._count_audit_parses(monkeypatch, audit_path)
            read: list[int] = []
            real_open = open

            def spy_open(
                file: Any, mode: str = "r", *args: Any, _p: Path = manifest_path, _r: list[int] = read, **kw: Any
            ) -> Any:
                handle = real_open(file, mode, *args, **kw)
                return _ReadSpy(handle, _r) if mode == "rb" and str(file) == str(_p) else handle

            _patch_bindings(monkeypatch, "open", spy_open)
            hooked = _HookedSigner(_signer(), on_verify=verifications.append)
            seal_ndjson_runs(audit_path, manifest_path, signer=hooked, run_id="run-next", seal_index_path=index_path)
            _unpatch_bindings(monkeypatch, "open")
            costs.append((len(verifications), len(parses), sum(read)))

        assert costs[0] == costs[1]
        assert min(costs[0]) > 0

    @staticmethod
    def _seal_after_checkpoint(audit_path: Path, manifest_path: Path, run_id: str = "run-tail") -> None:
        """A seal written without the index, so it lies past the checkpoint end offset."""
        _write_records(audit_path, [_record(run_id, 50)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id=run_id)

    # Review fixes.

    @staticmethod
    def _checkpoint_body(index_path: Path) -> dict[str, Any]:
        import sqlite3

        connection = sqlite3.connect(index_path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            row = connection.execute("SELECT body FROM checkpoint WHERE id = 1").fetchone()
        finally:
            connection.close()
        body: dict[str, Any] = json.loads(row[0])
        return body

    def test_a_seal_written_without_the_index_between_indexed_seals_stays_refused(self, tmp_path: Path) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live", 2)
        self._seal_after_checkpoint(audit_path, manifest_path, "run-tail")
        _write_records(audit_path, [_record("run-c", 70)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-c", seal_index_path=index_path)

        with pytest.raises(RunAlreadySealedError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-tail", seal_index_path=index_path)

    def test_an_audit_file_replaced_by_a_copy_keeps_sealed_runs_out_of_the_pending_set(self, tmp_path: Path) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        copy = audit_path.with_name("copy.tmp")
        copy.write_bytes(audit_path.read_bytes())
        os.replace(copy, audit_path)
        sealed = {f"run-{number}" for number in range(4)}

        for name in ("run-next", "run-after"):
            _write_records(audit_path, [_record(name, 99)])
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id=name, seal_index_path=index_path)
            sealed.add(name)

            assert not sealed & set(self._checkpoint_body(index_path)["audit_pending"])

    @pytest.mark.parametrize("kind", ["blob", "null"])
    def test_a_checkpoint_body_that_is_not_text_falls_back_in_the_seal(self, tmp_path: Path, kind: str) -> None:
        import sqlite3

        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        body = self._checkpoint_body(index_path)
        connection = sqlite3.connect(index_path)
        try:
            connection.execute("DROP TABLE checkpoint")
            connection.execute("CREATE TABLE checkpoint (id INTEGER PRIMARY KEY, body)")
            value = json.dumps(body).encode("utf-8") if kind == "blob" else None
            connection.execute("INSERT INTO checkpoint (id, body) VALUES (1, ?)", (value,))
            connection.commit()
        finally:
            connection.close()
        _write_records(audit_path, [_record("run-next", 99)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-next", seal_index_path=index_path)

        assert _read_lines(manifest_path)[-1]["run_id"] == "run-next"
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_a_non_string_run_id_before_the_run_fails_the_indexed_seal_as_the_plain_one(self, tmp_path: Path) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        _write_records(audit_path, [_record(5, 98), _record("run-next", 99)])  # type: ignore[arg-type]
        _copy_dir(tmp_path / "live", tmp_path / "ref")
        _assert_raises_and_unchanged(
            tmp_path / "ref",
            lambda: seal_ndjson_runs(
                tmp_path / "ref" / "audit.ndjson", tmp_path / "ref" / "manifests.ndjson", signer=_signer()
            ),
        )

        _assert_raises_and_unchanged(
            tmp_path / "live",
            lambda: seal_ndjson_runs(
                audit_path, manifest_path, signer=_signer(), run_id="run-next", seal_index_path=index_path
            ),
        )

    def test_a_file_at_the_index_path_that_is_not_sqlite_survives_a_seal_untouched(self, tmp_path: Path) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live", indexed=False)
        index_path.write_bytes(b"precious notes, not a database\n" * 50)
        before = index_path.read_bytes()
        _write_records(audit_path, [_record("run-next", 99)])

        manifests = seal_ndjson_runs(
            audit_path, manifest_path, signer=_signer(), run_id="run-next", seal_index_path=index_path
        )

        assert [m["run_id"] for m in manifests] == ["run-next"]
        assert index_path.read_bytes() == before

    def test_without_sqlite3_an_indexed_seal_matches_an_unindexed_one_and_warns_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live", 2, indexed=False)
        _write_records(audit_path, [_record("run-next", 99), _record("run-after", 100)])
        _copy_dir(tmp_path / "live", tmp_path / "ref")
        monkeypatch.setitem(sys.modules, "sqlite3", None)
        monkeypatch.setattr(_seal_index, "_sqlite_missing_logged", False)

        ref = [
            seal_ndjson_runs(
                tmp_path / "ref" / "audit.ndjson",
                tmp_path / "ref" / "manifests.ndjson",
                signer=_signer(),
                run_id=run_id,
            )[0]
            for run_id in ("run-next", "run-after")
        ]

        with caplog.at_level(logging.WARNING, logger=_AUDIT_LOGGER):
            live = [
                seal_ndjson_runs(
                    audit_path, manifest_path, signer=_signer(), run_id=run_id, seal_index_path=index_path
                )[0]
                for run_id in ("run-next", "run-after")
            ]

        assert self._chainless(live) == self._chainless(ref)
        assert live[0]["previous_manifest_hash"] == ref[0]["previous_manifest_hash"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        assert sorted(p.name for p in (tmp_path / "live").iterdir()) == ["audit.ndjson", "manifests.ndjson"]
        warnings = [r for r in caplog.records if r.name == _AUDIT_LOGGER and r.levelno == logging.WARNING]
        assert [r.getMessage() for r in warnings] == ["seal_index_path is ignored: sqlite3 is not available"]

    def test_the_index_file_is_created_with_owner_only_permissions(self, tmp_path: Path) -> None:
        _, _, index_path = _indexed_log(tmp_path / "live", 1)

        assert stat.S_IMODE(index_path.stat().st_mode) == 0o600

    def test_the_next_indexed_seal_after_a_segment_rotation_succeeds_and_rebuilds_the_index(
        self, tmp_path: Path
    ) -> None:
        tmp_path.joinpath("live").mkdir()
        audit_path, manifest_path = tmp_path / "live" / "audit.ndjson", tmp_path / "live" / "manifests.ndjson"
        index_path = tmp_path / "live" / _INDEX
        for number in range(3):
            _write_records(audit_path, [_record(f"run-{number}", number)])
            seal_ndjson_runs(
                audit_path,
                manifest_path,
                signer=_signer(),
                run_id=f"run-{number}",
                log_id="log-a",
                seal_index_path=index_path,
            )
        _write_records(audit_path, [_record("run-next", 50)])
        _rotate_segment(audit_path, manifest_path)
        _write_records(audit_path, [_record("run-after", 51)])

        for run_id in ("run-next", "run-after"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id=run_id, seal_index_path=index_path)

        assert [line["run_id"] for line in _read_lines(manifest_path)[1:]] == ["run-next", "run-after"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a")
