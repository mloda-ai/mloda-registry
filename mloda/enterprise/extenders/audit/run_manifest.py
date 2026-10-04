"""Run manifests: a signed, hash-chained seal over the audit records of one run.

Sealing is a post-run step: seal a run only once it is finished. A seal is final, so records reaching a sealed run
later fail verification by design. `head_anchor` emits each new head outside the log, and `anchored_heads` checks that
every anchored head is an ancestor of the log (catches truncation, rollback, deletion and substitution). The
quarantine functions take their own `head_anchor` and `anchored_heads` for the quarantine trace.
verify_ndjson_log returns the head, which `expected_head` can pin exactly.

Version 1 lines (older payload, no `sealed_late`) still verify, but only before the first version 2 line.
A `log_id` opts into a signed genesis first line naming the log; a verifier passing it fails a non-empty log without the
matching genesis; an empty or missing log passes (only anchors catch deletion).
A run that crashed stays unsealed until an operator sweeps with `seal_ndjson_runs(older_than=...)`, which seals only
stale runs and marks them `sealed_late`; the sweep is manual.

Limits:
- HMAC is symmetric: integrity only. Ed25519Signer adds non-repudiation: only the private key holder can seal. A
  public-key signer (`Ed25519Signer.from_public_key`) verifies but cannot seal, rotate or repair
  (`quarantine_damaged_lines(dry_run=True)` works). HMAC-era seals stay repudiable, and moving a log from HMAC to
  Ed25519 needs a new key_id. A log that began under an HMAC key still needs that retired key in `previous_signers` on
  every verifying host, so its verifier holds an HMAC secret. Distributing the public key is out of scope. A
  KMS-backed signer plugs in through the unchanged `ManifestSigner` protocol.
- Records without a usable run_id and unsealed runs sit outside every seal (verify_ndjson_log_coverage counts them).
- One manifest log per audit file; sealers serialise on flock (POSIX), but not at all where fcntl is missing, so
  concurrent sealers, rotations and recoveries then race. `previous_signers` verifies manifests a retired key sealed
  before its rotation entry (it cannot seal after it).
- Key rotation is an entry in the log (rotate_manifest_key): key order comes from the entries, not `previous_signers`.
  Rotate every log when changing keys; seals under the retired key before its entry stay valid. Anchor
  `expected_head` on every rotation. Keep retired keys while their seals must verify (rotating verifies the log with
  them too). A v2 rotation entry carries a co-signature by the outgoing current key, so only the current
  key can rotate: rotating needs its private key, and a lost current key cannot be rotated away from (start a new
  log). A v1 rotation entry (no co-signature) in a v1 prefix can still wedge the log for the real current key
  (availability only, not integrity). Recover by rotating forward to a fresh key when that entry's key is in the
  keyring; use quarantine_from_rotation_entry for an entry signed outside it or to keep the honest current key, then
  re-seal with seal_ndjson_runs(expected_head=<anchor>) and replace any external anchor recorded past it. The same
  repair lets any key that was current at an anchored head, even one retired since, drop the honest lines after it.
  Still a hand repair (truncate the log back to an anchored head): a terminated undecodable or duplicate-key line
  mid-log, a rotation entry lacking only its newline, a seal signed by a non-current keyring key, and a rotation
  entry as the first line (no anchored head to go back to).
- Without a seal index every auto-seal verifies the whole manifest log and parses the whole audit file. With
  `seal_index_path` a seal resumes from a signed checkpoint and no longer re-verifies lines before it, so run
  verify_ndjson_log(anchored_heads=...) on a schedule; an anchor lagging behind the checkpoint falls back to full
  verification. Re-running a sealed run and manual sweeps stay full scans. The digest cost is the bytes from the
  run's first record to EOF; runs left pending (crashed, or refused at plan time) stay so until swept. Negative
  lookups trust the index's unsigned hint rows: whoever can write the index can make a sealed run look unsealed and
  cause a duplicate seal, so keep it under the logs' write protection and verify offline on a schedule.
- Segments: `rotate_ndjson_segment` archives `<path>.<NNNNNN>` pairs and carries pending runs over;
  `verify_ndjson_segments` verifies the retained history. Only anchors detect deleted archives. Runs sealed in
  retained archives count as sealed via unverified reads. Safe only for writers taking the shared flock
  (NdjsonAuditSink), and not without fcntl.
- A line longer than MAX_LINE_BYTES (64 MiB) fails verification and is refused on write, which bounds a seal to
  roughly a million records per run. Logs written by earlier releases with a seal line over the cap no longer verify.
- Sealing and verifying need the log's current key to be `signer` (an archived log verifies with its current key).
  The check is load-bearing: it stops an unused keyring key from taking the log over.
- verify_manifest on a single manifest cannot order keys.
- A torn or undecodable line fails sealing and verification until quarantine_damaged_lines repairs it. It repairs
  only a torn manifest tail and the audit lines the readers reject; anything else stays a hard failure.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mloda.enterprise.extenders.audit._quarantine import QuarantinedLine as QuarantinedLine
from mloda.enterprise.extenders.audit._quarantine import quarantine_damaged_lines as quarantine_damaged_lines
from mloda.enterprise.extenders.audit._quarantine import (
    quarantine_from_rotation_entry as quarantine_from_rotation_entry,
)
from mloda.enterprise.extenders.audit._quarantine import verify_quarantine_log as verify_quarantine_log
from mloda.enterprise.extenders.audit._records import _is_blank
from mloda.enterprise.extenders.audit._seal_index import (
    _audit_scan_from,
    _indexed_lookup,
    _Scan,
    _scan_audit_tail,
    _sqlite,
    _update_index,
    _verify_from_checkpoint,
)
from mloda.enterprise.extenders.audit._segments import (
    _archived_sealed_ids,
    _archived_segments,
    _refuse_interrupted_rotation,
    _sealed_in_archives,
    _segment_path,
)
from mloda.enterprise.extenders.audit._segments import rotate_ndjson_segment as rotate_ndjson_segment
from mloda.enterprise.extenders.audit._segments import verify_ndjson_segments as verify_ndjson_segments
from mloda.enterprise.extenders.audit._signers import Ed25519Signer as Ed25519Signer
from mloda.enterprise.extenders.audit._signers import HmacSha256Signer as HmacSha256Signer
from mloda.enterprise.extenders.audit._signers import ManifestSigner as ManifestSigner
from mloda.enterprise.extenders.audit._signers import _signer_map
from mloda.enterprise.extenders.audit._verify import MAX_LINE_BYTES as MAX_LINE_BYTES
from mloda.enterprise.extenders.audit._verify import HeadAnchor as HeadAnchor
from mloda.enterprise.extenders.audit._verify import KeyAlreadyCurrentError as KeyAlreadyCurrentError
from mloda.enterprise.extenders.audit._verify import LogCoverage as LogCoverage
from mloda.enterprise.extenders.audit._verify import ManifestVerificationError as ManifestVerificationError
from mloda.enterprise.extenders.audit._verify import NdjsonHeadAnchor as NdjsonHeadAnchor
from mloda.enterprise.extenders.audit._verify import RunAlreadySealedError as RunAlreadySealedError
from mloda.enterprise.extenders.audit._verify import RunNotPendingError as RunNotPendingError
from mloda.enterprise.extenders.audit._verify import (
    _append_with_rollback,
    _AuditScan,
    _check_line_cap,
    _check_log_id,
    _digest_runs,
    _flock,
    _fsync,
    _genesis_entry,
    _iter_manifests,
    _read_manifests,
    _reject_aliased_paths,
    _rotation_entry,
    _rotation_transition,
    _RunDigest,
    _scan_for_run,
    _seal,
    _Uncovered,
    _unlink_durably,
    _verify_digest,
    _verify_log,
    _verify_manifest_fields,
)
from mloda.enterprise.extenders.audit._verify import manifest_hash as manifest_hash
from mloda.enterprise.extenders.audit._verify import seal_run as seal_run
from mloda.enterprise.extenders.audit._verify import verify_manifest as verify_manifest

# Keep reporting the facade as the defining module (tracebacks, pickles, _error_type), as before the split.
for _public in (
    QuarantinedLine,
    quarantine_damaged_lines,
    quarantine_from_rotation_entry,
    verify_quarantine_log,
    rotate_ndjson_segment,
    verify_ndjson_segments,
    Ed25519Signer,
    HmacSha256Signer,
    ManifestSigner,
    HeadAnchor,
    KeyAlreadyCurrentError,
    LogCoverage,
    ManifestVerificationError,
    NdjsonHeadAnchor,
    RunAlreadySealedError,
    RunNotPendingError,
    manifest_hash,
    seal_run,
    verify_manifest,
):
    setattr(_public, "__module__", __name__)

logger = logging.getLogger(__name__)


def _is_run_sealed_unverified(manifest_path: str | Path, run_id: str, index_path: str | Path | None = None) -> bool:
    """Unverified read without the lock: a seal holds the exclusive lock while it digests the audit log, so a
    calculation must not wait on it. True iff a line naming run_id decodes to a JSON object with
    that run_id; a line that does not decode (torn, or mid-append) is skipped, not treated as sealed; a
    decodable one, even unterminated, counts. With `index_path` (read-only, no signature check) a hint hit is
    confirmed at its offset and only the bytes after the checkpoint are scanned; anything unusable scans it all."""
    if index_path is not None:
        found = _indexed_lookup(manifest_path, run_id, index_path)
        if found is not None:
            return found
    return _scan_for_run(manifest_path, run_id) or _sealed_in_archives(manifest_path, run_id)


def _check_run_against_seal(
    audit_path: str | Path,
    manifest_path: str | Path,
    run_id: str,
    *,
    signer: ManifestSigner,
    previous_signers: Iterable[ManifestSigner] = (),
) -> None:
    """Raise ManifestVerificationError when run_id's manifest fails its signature/field checks, or when
    audit_path's records of run_id no longer match it."""
    with _flock(manifest_path, exclusive=False):
        manifest = next((m for m in _iter_manifests(manifest_path) if m.get("run_id") == run_id), None)
        sealed_audit: str | Path = audit_path
        if manifest is None:
            for number, archive in reversed(_archived_segments(manifest_path)):
                manifest = next((m for m in _iter_manifests(archive) if m.get("run_id") == run_id), None)
                if manifest is not None:
                    sealed_audit = _segment_path(audit_path, number)
                    break
        if manifest is None:
            raise ManifestVerificationError(f"run_id {run_id!r} has no manifest in {manifest_path}")
        _verify_manifest_fields(manifest, signer, _signer_map(signer, previous_signers))
        _verify_digest(manifest, _digest_runs(sealed_audit, run_id.__eq__).get(run_id, _RunDigest()))


def rotate_manifest_key(
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    expected_head: str | None,
    previous_signers: Iterable[ManifestSigner] = (),
    log_id: str | None = None,
    anchored_heads: Iterable[str] = (),
    head_anchor: HeadAnchor | None = None,
) -> dict[str, Any]:
    """Append a signed rotation entry making `signer` the log's current key; return it. Verifies the log first
    (`previous_signers` must hold its retired keys) and writes nothing on failure. The outgoing current key co-signs
    the entry, so it must be in `previous_signers` with its private key (ValueError for a public-key one). Raises
    ValueError for a log with no manifests, KeyAlreadyCurrentError when `signer` is already current (a retry needs
    `expected_head=None` or the post-rotation head) and ManifestVerificationError when `signer` is retired. Pass the
    anchored head as `expected_head`: the entry chains onto it, so it commits to every earlier line. A wrongly
    appended entry is dropped with quarantine_from_rotation_entry. Every `anchored_heads` entry must be a line of the
    log; `head_anchor` gets the entry's hash after the append."""
    signers = _signer_map(signer, previous_signers)
    anchors = list(anchored_heads)
    _check_log_id("rotate_manifest_key", log_id)
    nothing_to_rotate = f"{manifest_path} has no manifests to rotate; the first seal defines the key"
    # Checked before the lock: an exclusive _flock would create the file.
    if not os.path.exists(manifest_path):
        _verify_log([], signer=signer, signers=signers, expected_head=None, anchored_heads=anchors)
        raise ValueError(nothing_to_rotate)
    with _flock(manifest_path, exclusive=True):
        _refuse_interrupted_rotation(manifest_path)
        state = _verify_log(
            _iter_manifests(manifest_path),
            signer=signer,
            signers=signers,
            expected_head=expected_head,
            log_id=log_id,
            anchored_heads=anchors,
        )
        if state.active is None:
            raise ValueError(nothing_to_rotate)
        if state.active == signer.key_id:
            raise KeyAlreadyCurrentError(f"manifest log is already under key {signer.key_id!r}")
        # Called for its raise only: it raises when `signer` is a retired key.
        _rotation_transition(signer.key_id, state.active, state.retired)
        outgoing = signers[state.active]
        try:
            entry = _rotation_entry(signer, outgoing, state.head)
        except ValueError as exc:
            raise ValueError(
                f"cannot rotate away from key {outgoing.key_id!r}: it holds only a public key and cannot co-sign; "
                "start a new log"
            ) from exc
        _append_with_rollback(manifest_path, [entry], existed=True)
        if head_anchor is not None:
            head_anchor.write(manifest_hash(entry))
        return entry


def seal_ndjson_runs(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    previous_signers: Iterable[ManifestSigner] = (),
    run_id: str | None = None,
    expected_head: str | None = None,
    sealed_late: bool = True,
    log_id: str | None = None,
    anchored_heads: Iterable[str] = (),
    head_anchor: HeadAnchor | None = None,
    older_than: timedelta | None = None,
    seal_index_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Seal every unsealed run (or only `run_id`). Seal only after a run's writers stop, or sealing a still-live run
    fails its verification for good; prefer AuditExtender's automatic sealing from Extender.on_run_complete when
    available, since it fires only once a run's writers have stopped, for a run that is run() exactly once
    (AuditExtender refuses a calculation under a run_id already sealed in its manifest log: see its class docstring).
    Pass `run_id` when other runs may still be live; omit it to sweep an audit file no writer is appending to, but
    never as a substitute for targeting a specific unsealed run_id when other runs are still live, since a blanket
    sweep could seal one of those.
    Raises RunAlreadySealedError for an already-sealed `run_id` (catch that, not ValueError, for an idempotent retry)
    and RunNotPendingError when it has no records. `signer` must be the log's current key: call rotate_manifest_key
    after a key change.
    `previous_signers` covers a retired signing key during rotation (see module docstring).
    `log_id` writes a genesis line before the first seal batch of an empty log; a non-empty log must already have it.
    Every `anchored_heads` entry must be a line of the log; `head_anchor` gets the new head after the append, under the
    lock (its failure leaves the seals written).
    `older_than` (a non-negative timedelta, not with `run_id`) is the manual stale sweep for crashed runs: it seals only
    runs whose newest record `event_time` is older than now minus it. A run with a record lacking a parseable
    `event_time` is skipped, and one warning gives the count.
    `seal_index_path` (opt-in sqlite file) keeps a signed checkpoint of the verified log, so a `run_id` seal verifies
    only the manifest lines after it; any stale or unusable index means full verification, and an index error after
    the append is logged, never raised."""
    signers = _signer_map(signer, previous_signers)
    _reject_aliased_paths(audit_path=audit_path, manifest_path=manifest_path)
    if seal_index_path is not None:
        anchor_path = head_anchor._path if isinstance(head_anchor, NdjsonHeadAnchor) else None
        _reject_aliased_paths(
            audit_path=audit_path,
            manifest_path=manifest_path,
            seal_index_path=seal_index_path,
            **({"head_anchor": anchor_path} if anchor_path is not None else {}),
        )
        if _sqlite() is None:
            seal_index_path = None
    anchors = list(anchored_heads)
    if run_id is not None and (not isinstance(run_id, str) or _is_blank(run_id)):
        raise ValueError("seal_ndjson_runs run_id must be a non-blank string")
    if older_than is not None:
        if not isinstance(older_than, timedelta) or older_than < timedelta(0):
            raise ValueError("seal_ndjson_runs older_than must be a non-negative timedelta")
        if run_id is not None:
            raise ValueError("seal_ndjson_runs older_than cannot be combined with run_id")
    _check_log_id("seal_ndjson_runs", log_id)
    existed = os.path.exists(manifest_path)
    with _flock(manifest_path, exclusive=True):
        _refuse_interrupted_rotation(manifest_path, audit_path)
        fast = checkpoint = None
        if seal_index_path is not None and run_id is not None:
            fast = _verify_from_checkpoint(
                seal_index_path,
                manifest_path,
                run_id,
                signer=signer,
                signers=signers,
                expected_head=expected_head,
                log_id=log_id,
                anchors=anchors,
            )
        state, scan = (fast[0], fast[1]) if fast else (None, _Scan())
        audit = None
        if fast:
            checkpoint = fast[2]
            audit = _audit_scan_from(checkpoint, audit_path)
            if audit is None:
                # The audit cursor cannot resume, so rebuild everything instead of carrying sealed runs as pending.
                fast = checkpoint = state = None
                scan = _Scan()
        if state is None:
            state = _verify_log(
                scan.manifests(manifest_path),
                signer=signer,
                signers=signers,
                expected_head=expected_head,
                require_current=True,
                log_id=log_id,
                anchored_heads=anchors,
            )
        sealed, head = state.sealed, state.head
        if run_id is not None and run_id in sealed:
            raise RunAlreadySealedError(f"run_id {run_id!r} is already sealed in {manifest_path}")
        # Runs sealed in an archived segment count as sealed: a sweep skips them, a targeted seal refuses them.
        archived: set[str] | None = None
        if run_id is None or (seal_index_path is not None and not fast):
            archived = _archived_sealed_ids(manifest_path)
        if run_id is not None and (
            run_id in archived if archived is not None else _sealed_in_archives(manifest_path, run_id, seal_index_path)
        ):
            raise RunAlreadySealedError(
                f"run_id {run_id!r} is already sealed in an archived segment of {manifest_path}"
            )
        skip = sealed | archived if archived else sealed
        # A manifest must not name records that never reached disk; a missing file is left for _digest_runs
        # to raise, as before.
        with suppress(FileNotFoundError):
            _fsync(audit_path)
        incremental = audit is not None and run_id is not None
        first = (0, 1)
        if audit is not None and incremental:
            _scan_audit_tail(audit_path, audit, sealed)
            first = audit.first.get(run_id, (audit.end, audit.count + 1)) if run_id else first
        else:
            audit = _AuditScan.fresh(audit_path) if seal_index_path is not None else None
        digests = _digest_runs(
            audit_path,
            lambda candidate: candidate == run_id if run_id is not None else candidate not in skip,
            track_times=older_than is not None,
            start=first[0],
            number=first[1],
            scan=None if incremental else audit,
        )
        if older_than is not None:
            digests = _stale_runs(digests, older_than)
        if run_id is not None and run_id not in digests:
            raise RunNotPendingError(f"no audit records for run_id {run_id!r} in {audit_path}")

        genesis = [_genesis_entry(signer, log_id)] if log_id is not None and head is None and digests else []
        if genesis:
            head = manifest_hash(genesis[0])
        manifests = []
        for pending_run_id, digest in digests.items():
            manifest = _seal(
                pending_run_id, digest, signer=signer, previous_manifest_hash=head, sealed_late=sealed_late
            )
            manifests.append(manifest)
            head = manifest_hash(manifest)
        appended = [*genesis, *manifests]
        try:
            _check_line_cap(appended)
        except ValueError:
            if not existed and os.path.getsize(manifest_path) == 0:
                with suppress(OSError):
                    _unlink_durably(manifest_path)
            raise
        _append_with_rollback(manifest_path, appended, existed=True)
        if seal_index_path is not None and audit is not None and (appended or (not fast and state.head is not None)):
            _update_index(
                seal_index_path,
                manifest_path,
                state,
                scan,
                audit,
                appended,
                signer,
                rebuild=not fast,
                archived=archived or (),
            )
        if head_anchor is not None and appended:
            head_anchor.write(manifest_hash(appended[-1]))
        return manifests


def _stale_runs(digests: dict[str, _RunDigest], older_than: timedelta) -> dict[str, _RunDigest]:
    """The digests whose newest event_time is older than now minus `older_than`; warn once with the undated count."""
    cutoff = datetime.now(timezone.utc) - older_than
    skipped = sum(digest.undated or digest.newest is None for digest in digests.values())
    if skipped:
        logger.warning("seal_ndjson_runs skipped %d run(s) with a record lacking a parseable event_time", skipped)
    return {
        run: digest
        for run, digest in digests.items()
        if not digest.undated and digest.newest is not None and digest.newest < cutoff
    }


def verify_ndjson_log_coverage(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    previous_signers: Iterable[ManifestSigner] = (),
    expected_head: str | None = None,
    log_id: str | None = None,
    anchored_heads: Iterable[str] = (),
) -> LogCoverage:
    """Verify like verify_ndjson_log; return a LogCoverage (head, sealed_runs, sealed_lines, unattributed_lines,
    unsealed_lines). `signer` must be the log's current key: call rotate_manifest_key after a key change.
    `previous_signers` covers a retired signing key during rotation (see module docstring)."""
    signers = _signer_map(signer, previous_signers)
    _reject_aliased_paths(audit_path=audit_path, manifest_path=manifest_path)
    _check_log_id("verify_ndjson_log_coverage", log_id)
    uncovered = _Uncovered()
    # Log first: a run sealed after this read looks unsealed, not tampered. The shared lock spans both reads.
    with _flock(manifest_path, exclusive=False):
        manifests = _read_manifests(manifest_path)
        sealed = {manifest["run_id"] for manifest in manifests if isinstance(manifest.get("run_id"), str)}
        try:
            digests = _digest_runs(audit_path, sealed.__contains__, uncovered)
        except FileNotFoundError:
            digests = {}
    state = _verify_log(
        manifests,
        signer=signer,
        signers=signers,
        expected_head=expected_head,
        digests=digests,
        require_current=True,
        log_id=log_id,
        anchored_heads=anchored_heads,
    )
    return LogCoverage(
        head=state.head,
        sealed_runs=len(sealed),
        sealed_lines=sum(len(digest.hashes) for digest in digests.values()),
        unattributed_lines=uncovered.unattributed,
        unsealed_lines=dict(uncovered.by_run),
    )


def verify_ndjson_log(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    previous_signers: Iterable[ManifestSigner] = (),
    expected_head: str | None = None,
    log_id: str | None = None,
    anchored_heads: Iterable[str] = (),
) -> str | None:
    """Raise ManifestVerificationError unless every manifest matches the audit bytes of its run and every
    `anchored_heads` entry is a line of the log. Returns the head.
    `signer` must be the log's current key: call rotate_manifest_key after a key change.
    `previous_signers` covers a retired signing key during rotation (see module docstring)."""
    return verify_ndjson_log_coverage(
        audit_path,
        manifest_path,
        signer=signer,
        previous_signers=previous_signers,
        expected_head=expected_head,
        log_id=log_id,
        anchored_heads=anchored_heads,
    ).head
