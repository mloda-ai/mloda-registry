"""Run manifest seal index: checkpoints, the SQLite index and archived-run lookups."""

from __future__ import annotations

import os
from collections.abc import Container, Iterable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

from mloda.enterprise.extenders.audit._core import (
    _GENESIS_KIND,
    _SIGNATURE_KEYS,
    MAX_LINE_BYTES,
    ManifestVerificationError,
    _AuditScan,
    _decode_line,
    _LogState,
    _parse_ndjson_line,
    _read_lines,
    _read_ndjson,
    _record_run_id,
    _sha256,
    _signature_block,
    _signing_payload,
    _unverified_anchor,
    _verify_log,
    logger,
    manifest_hash,
)
from mloda.enterprise.extenders.audit._records import (
    _canonical_json,
)
from mloda.enterprise.extenders.audit._signers import (
    ManifestSigner,
)

_CHECKPOINT_KIND = "seal_checkpoint"
_CHECKPOINT_INTS = ("st_dev", "st_ino", "end", "lines", "head_start")
_AUDIT_INTS = ("audit_dev", "audit_ino", "audit_end", "audit_lines", "audit_last_start")
_CHECKPOINT_KEYS = {
    *_CHECKPOINT_INTS,
    *_AUDIT_INTS,
    "audit_last_sha256",
    "audit_pending",
    "kind",
    "head",
    "active",
    "retired",
    "log_id",
    "seen_v2",
    "signature",
}
_SQLITE_HEADER = b"SQLite format 3\x00"
_sqlite_missing_logged = False


def _sqlite() -> ModuleType | None:
    """The sqlite3 module, or None (logged once) where it is missing: the index is then off."""
    global _sqlite_missing_logged
    try:
        import sqlite3
    except ImportError:
        if not _sqlite_missing_logged:
            _sqlite_missing_logged = True
            logger.warning("seal_index_path is ignored: sqlite3 is not available")
        return None
    return sqlite3


class _Scan:
    """Walks manifest lines from `end`, tracking line numbers, the last line's start and each seal's start offset."""

    def __init__(self, end: int = 0, count: int = 0, last: int | None = None) -> None:
        self.end, self.count, self.last = end, count, last
        self.seals: dict[str, int] = {}

    def manifests(self, path: str | Path) -> Iterator[dict[str, Any]]:
        try:
            with open(path, "rb") as file:
                file.seek(self.end)
                for raw in _read_lines(file):
                    self.count += 1
                    manifest = _parse_ndjson_line(path, self.count, raw)[1]
                    self.last = self.end
                    if "kind" not in manifest and isinstance(manifest.get("run_id"), str):
                        self.seals[manifest["run_id"]] = self.end
                    self.end += len(raw)
                    yield manifest
        except FileNotFoundError:
            return


def _checkpoint_current(entry: Any, signer: ManifestSigner, manifest_path: str | Path) -> bool:
    """True iff `entry` is a checkpoint signed by the current key that still describes the manifest file."""
    if not isinstance(entry, dict) or set(entry) != _CHECKPOINT_KEYS or entry["kind"] != _CHECKPOINT_KIND:
        return False
    signature = entry["signature"]
    if not isinstance(signature, Mapping) or set(signature) != _SIGNATURE_KEYS:
        return False
    value = signature["value"]
    # Authenticate before trusting any field.
    if not isinstance(value, str) or not signer.verify(_signing_payload(entry), value):
        return False
    if {signature["key_id"], entry["active"]} != {signer.key_id} or signature["algorithm"] != signer.algorithm:
        return False
    return _checkpoint_fresh(entry, manifest_path)


def _checkpoint_fresh(entry: Any, manifest_path: str | Path) -> bool:
    """True iff `entry` has the checkpoint shape and still describes the manifest file (signature not checked)."""
    if not isinstance(entry, dict) or set(entry) != _CHECKPOINT_KEYS or entry["kind"] != _CHECKPOINT_KIND:
        return False
    shape = (
        all(type(entry[key]) is int for key in _CHECKPOINT_INTS)
        and isinstance(entry["head"], str)
        and isinstance(entry["retired"], list)
        and all(isinstance(key, str) for key in entry["retired"])
        and (entry["log_id"] is None or isinstance(entry["log_id"], str))
        and type(entry["seen_v2"]) is bool
    )
    start, end = entry["head_start"], entry["end"]
    if not shape or not 0 <= start < end <= start + MAX_LINE_BYTES + 1:
        return False
    stat_result = os.stat(manifest_path)
    if (stat_result.st_dev, stat_result.st_ino) != (entry["st_dev"], entry["st_ino"]) or stat_result.st_size < end:
        return False
    with open(manifest_path, "rb") as file:
        file.seek(start)
        raw = file.read(end - start)
    try:
        return raw.endswith(b"\n") and manifest_hash(_decode_line("seal index head line", raw[:-1])) == entry["head"]
    except ManifestVerificationError:
        return False


def _load_checkpoint(
    index_path: str | Path, manifest_path: str | Path, run_id: str, signer: ManifestSigner
) -> tuple[dict[str, Any], int | None] | None:
    """The valid checkpoint and the run's hinted seal offset, or None for anything missing, unreadable or stale."""
    db = _sqlite()
    if db is None or not os.path.exists(index_path):
        return None
    try:
        entry, hint = _read_index(db, index_path, run_id)
        if not _checkpoint_current(entry, signer, manifest_path):
            return None
    except Exception:  # an untrusted index must never fail the seal: any error means a full verify
        return None
    return entry, hint


@contextmanager
def _readonly_index(db: ModuleType, index_path: str | Path) -> Iterator[Any]:
    with closing(db.connect(Path(index_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0)) as conn:
        conn.execute("PRAGMA trusted_schema=OFF")
        yield conn


def _indexed_archive(db: ModuleType, index_path: str | Path, run_id: str) -> bool:
    """Whether the index's `archived` table holds run_id; a missing table raises (the index is unusable)."""
    with _readonly_index(db, index_path) as conn:
        return conn.execute("SELECT 1 FROM archived WHERE run_id = ?", (run_id,)).fetchone() is not None


def _read_index(db: ModuleType, index_path: str | Path, run_id: str) -> tuple[Any, int | None]:
    """The stored checkpoint entry and the run's hinted offset, read-only and without waiting on a writer."""
    with _readonly_index(db, index_path) as conn:
        row = conn.execute(
            "SELECT body FROM checkpoint WHERE id = 1 AND length(CAST(body AS BLOB)) <= ?", (MAX_LINE_BYTES,)
        ).fetchone()
        hint = conn.execute("SELECT line_start FROM hint WHERE run_id = ?", (run_id,)).fetchone()
    if row is None or not isinstance(row[0], str):
        raise ValueError("seal index checkpoint is missing, oversized or not text")
    entry = _decode_line("seal index checkpoint", row[0].encode("utf-8"))
    return entry, None if hint is None else hint[0]


def _audit_scan_from(entry: Mapping[str, Any], audit_path: str | Path) -> _AuditScan | None:
    """The audit scan a checkpoint recorded, or None when the audit file no longer matches it."""
    try:
        dev, ino, end, count, last = (entry[key] for key in _AUDIT_INTS)
        sha, pending = entry["audit_last_sha256"], entry["audit_pending"]
        if not all(type(value) is int for value in (dev, ino, end, count, last)) or not isinstance(sha, str):
            return None
        first = {run: (offset, number) for run, (offset, number) in pending.items()}
        stat_result = os.stat(audit_path)
        if (stat_result.st_dev, stat_result.st_ino) != (dev, ino) or stat_result.st_size < end:
            return None
        if end:
            if not 0 <= last < end <= last + MAX_LINE_BYTES + 1:
                return None
            with open(audit_path, "rb") as file:
                file.seek(last)
                if _sha256(file.read(end - last)) != sha:
                    return None
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        return None
    scan = _AuditScan(dev, ino)
    scan.end, scan.count, scan.last_start, scan.last_sha256, scan.first = end, count, last, sha, first
    return scan


def _scan_audit_tail(audit_path: str | Path, scan: _AuditScan, sealed: Container[str]) -> None:
    """Advance `scan` over the terminated audit lines after it, noting where unsealed runs start."""
    for line, record in _read_ndjson(audit_path, scan.end, scan.count + 1):
        scan.note(line, _record_run_id(record), sealed)


def _hint_names_run(manifest_path: str | Path, offset: object, run_id: str) -> bool:
    """True iff the manifest line at `offset` is a seal of `run_id`."""
    if type(offset) is not int:
        return False
    with open(manifest_path, "rb") as file:
        file.seek(offset)
        raw = next(_read_lines(file), b"")
    try:
        line = _decode_line("seal index hint", raw.rstrip(b"\n")) if raw.endswith(b"\n") else None
    except ManifestVerificationError:
        return False
    return isinstance(line, dict) and "kind" not in line and line.get("run_id") == run_id


def _is_sqlite_or_empty(path: str | Path) -> bool:
    with open(path, "rb") as file:
        head = file.read(len(_SQLITE_HEADER))
    return not head or head == _SQLITE_HEADER


def _store_checkpoint(
    index_path: str | Path,
    entry: dict[str, Any],
    hints: Mapping[str, int],
    *,
    rebuild: bool,
    archived: Iterable[str] = (),
) -> None:
    """Commit `entry`, `hints` and `archived` (all replaced if `rebuild`); a corrupt SQLite file is deleted and
    recreated, any other file is left alone and raises."""
    db = _sqlite()
    if db is None:
        return
    body = _canonical_json(entry).decode("utf-8")
    for attempt in (0, 1):
        try:
            if not os.path.exists(index_path):
                os.close(os.open(index_path, os.O_WRONLY | os.O_CREAT, 0o600))
            with closing(db.connect(os.fspath(index_path), timeout=5)) as conn:
                conn.execute("PRAGMA trusted_schema=OFF")
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.execute("CREATE TABLE IF NOT EXISTS checkpoint (id INTEGER PRIMARY KEY, body TEXT NOT NULL)")
                conn.execute("CREATE TABLE IF NOT EXISTS hint (run_id TEXT PRIMARY KEY, line_start INTEGER NOT NULL)")
                conn.execute("CREATE TABLE IF NOT EXISTS archived (run_id TEXT PRIMARY KEY)")
                with conn:
                    if rebuild:
                        conn.execute("DELETE FROM hint")
                        conn.execute("DELETE FROM archived")
                        conn.executemany("INSERT OR REPLACE INTO archived VALUES (?)", [(run,) for run in archived])
                    conn.executemany("INSERT OR REPLACE INTO hint VALUES (?, ?)", list(hints.items()))
                    conn.execute("INSERT OR REPLACE INTO checkpoint VALUES (1, ?)", (body,))
            return
        except db.OperationalError:
            raise
        except db.DatabaseError:
            if attempt:
                raise
            if not _is_sqlite_or_empty(index_path):
                raise ValueError(f"{index_path} is not a SQLite file; leaving it alone") from None
            os.unlink(index_path)


def _checkpoint_entry(
    manifest_path: str | Path,
    state: _LogState,
    head_start: int,
    signer: ManifestSigner,
    audit: _AuditScan,
    done: Container[str],
) -> dict[str, Any]:
    stat_result = os.stat(manifest_path)
    entry: dict[str, Any] = {
        "kind": _CHECKPOINT_KIND,
        "st_dev": stat_result.st_dev,
        "st_ino": stat_result.st_ino,
        "end": stat_result.st_size,
        "lines": state.lines,
        "head_start": head_start,
        "head": state.head,
        "active": state.active,
        "retired": sorted(state.retired),
        "log_id": state.log_id,
        "seen_v2": state.seen_v2,
        "audit_dev": audit.dev,
        "audit_ino": audit.ino,
        "audit_end": audit.end,
        "audit_lines": audit.count,
        "audit_last_start": audit.last_start,
        "audit_last_sha256": audit.last_sha256,
        "audit_pending": {run: list(where) for run, where in audit.first.items() if run not in done},
    }
    entry["signature"] = _signature_block(entry, signer)
    return entry


def _verify_from_checkpoint(
    index_path: str | Path,
    manifest_path: str | Path,
    run_id: str,
    *,
    signer: ManifestSigner,
    signers: Mapping[str, ManifestSigner],
    expected_head: str | None,
    log_id: str | None,
    anchors: Sequence[str],
) -> tuple[_LogState, _Scan, dict[str, Any]] | None:
    """Verify only the manifest lines after a valid checkpoint; None means verify the whole log."""
    loaded = _load_checkpoint(index_path, manifest_path, run_id, signer)
    if loaded is None:
        return None
    checkpoint, hint = loaded
    scan = _Scan(checkpoint["end"], checkpoint["lines"], checkpoint["head_start"])
    head = checkpoint["head"]
    seed = _LogState(
        set(),
        head,
        checkpoint["active"],
        frozenset(checkpoint["retired"]),
        {head},
        checkpoint["seen_v2"],
        checkpoint["log_id"],
        checkpoint["lines"],
    )
    state = _verify_log(
        scan.manifests(manifest_path),
        signer=signer,
        signers=signers,
        expected_head=expected_head,
        require_current=True,
        log_id=log_id,
        seed=seed,
    )
    if log_id is not None and state.log_id != log_id:
        return None
    if _unverified_anchor(anchors, state.hashes) is not None:
        logger.info("seal index skipped: an anchored head is older than the checkpoint")
        return None
    if hint is not None:
        if not _hint_names_run(manifest_path, hint, run_id):
            return None
        state.sealed.add(run_id)
    return state, scan, checkpoint


def _update_index(
    index_path: str | Path,
    manifest_path: str | Path,
    state: _LogState,
    scan: _Scan,
    audit: _AuditScan,
    appended: Sequence[Mapping[str, Any]],
    signer: ManifestSigner,
    *,
    rebuild: bool,
    archived: Iterable[str] = (),
) -> None:
    """Commit a fresh checkpoint and the new seals' hints after an append; a failure is logged, never raised."""
    try:
        offset = os.path.getsize(manifest_path) - sum(len(_canonical_json(entry)) + 1 for entry in appended)
        hints, head_start = dict(scan.seals), scan.last
        for entry in appended:
            head_start = offset
            if "kind" not in entry:
                hints[entry["run_id"]] = offset
            offset += len(_canonical_json(entry)) + 1
        genesis = next((entry["log_id"] for entry in appended if entry.get("kind") == _GENESIS_KIND), state.log_id)
        head = manifest_hash(appended[-1]) if appended else state.head
        after = replace(
            state,
            head=head,
            active=state.active or signer.key_id,
            seen_v2=state.seen_v2 or bool(appended),
            log_id=genesis,
            lines=state.lines + len(appended),
        )
        done = state.sealed | {entry["run_id"] for entry in appended if "kind" not in entry}
        entry = _checkpoint_entry(manifest_path, after, head_start or 0, signer, audit, done)
        _store_checkpoint(index_path, entry, hints, rebuild=rebuild, archived=archived)
    except Exception as exc:
        logger.warning("seal index %s was not updated: %s", index_path, exc)
