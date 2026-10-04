"""Run manifest quarantine: damaged-line quarantine, its trace chain and repair."""

from __future__ import annotations

import base64
import os
import stat
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mloda.enterprise.extenders.audit._records import (
    _canonical_json,
    _utc_now,
)
from mloda.enterprise.extenders.audit._segments import (
    _refuse_interrupted_rotation,
)
from mloda.enterprise.extenders.audit._signers import (
    ManifestSigner,
    _signer_map,
)
from mloda.enterprise.extenders.audit._verify import (
    _V1,
    MAX_LINE_BYTES,
    HeadAnchor,
    ManifestVerificationError,
    _append_with_rollback,
    _check_signature,
    _decode_line,
    _flock,
    _fsync,
    _is_terminated,
    _line_size,
    _OversizedLine,
    _parse_ndjson_line,
    _read_lines,
    _read_ndjson,
    _reject_aliased_paths,
    _require_terminated,
    _sha256,
    _signature_block,
    _truncate_durably,
    _unlink_durably,
    _unverified_anchor,
    _verify_log,
    manifest_hash,
)

_QUARANTINE_VERSION = 2


@dataclass(frozen=True)
class QuarantinedLine:
    """A line removed by a quarantine repair; line (1-based) and offset are from before the repair."""

    file: str
    line: int
    offset: int
    length: int
    sha256: str
    reason: str


_DamageEntry = tuple[QuarantinedLine, bytes, Path]
_Damage = list[_DamageEntry]


def _damage_entry(file: str, path: str | Path, number: int, offset: int, raw: bytes, reason: str) -> _DamageEntry:
    digest = raw.sha256 if isinstance(raw, _OversizedLine) else _sha256(raw)
    return QuarantinedLine(file, number, offset, _line_size(raw), digest, reason), raw, Path(path)


def _refuse_json_tail(path: str | Path, number: int, tail: bytes) -> None:
    where = f"{path} line {number}"
    if isinstance(tail, _OversizedLine):
        return
    try:
        _decode_line(where, tail)
    except ManifestVerificationError:
        return
    raise ManifestVerificationError(f"{where} is unterminated but valid JSON, so it may be a real seal or record")


def _damaged_lines(file: str, path: str | Path, raws: Iterable[bytes], *, number: int = 1, offset: int = 0) -> _Damage:
    damage: _Damage = []
    for raw in raws:
        try:
            _parse_ndjson_line(path, number, raw)
        except ManifestVerificationError as exc:
            if not _is_terminated(raw):
                _refuse_json_tail(path, number, raw)
            damage.append(_damage_entry(file, path, number, offset, raw, str(exc)))
        number += 1
        offset += _line_size(raw)
    return damage


def _damaged_audit_lines(path: str | Path) -> _Damage:
    try:
        with open(path, "rb") as file:
            return _damaged_lines("audit", path, _read_lines(file))
    except FileNotFoundError:
        return []


def _split_unterminated_tail(path: str | Path) -> tuple[list[bytes], bytes]:
    """The terminated raw lines of `path` (none if it is missing), and its unterminated last line (empty if none)."""
    try:
        with open(path, "rb") as file:
            lines = list(_read_lines(file))
    except FileNotFoundError:
        lines = []
    tail = lines.pop() if lines and not _is_terminated(lines[-1]) else b""
    return lines, tail


def _parsed_manifests(path: str | Path, lines: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Parse `lines` lazily, so a caller that stops early never parses the lines after."""
    for number, raw in enumerate(lines, start=1):
        yield _parse_ndjson_line(path, number, raw)[1]


def _damaged_manifest_tail(
    path: str | Path, *, signer: ManifestSigner, signers: Mapping[str, ManifestSigner], expected_head: str | None
) -> _Damage:
    """`expected_head` may be any complete manifest's head: a torn multi-seal write leaves complete ones past it."""
    lines, tail = _split_unterminated_tail(path)
    manifests = list(_parsed_manifests(path, lines))
    state = _verify_log(manifests, signer=signer, signers=signers, expected_head=None)
    head = state.head
    if expected_head is not None and _unverified_anchor([expected_head], state.hashes) is not None:
        raise ManifestVerificationError(
            f"manifest log head {head!r} is not the expected head {expected_head!r}, and no earlier manifest has it"
        )
    if not tail:
        return []
    return _damaged_lines("manifest", path, [tail], number=len(lines) + 1, offset=sum(map(_line_size, lines)))


def _trace_entry(
    item: QuarantinedLine, raw: bytes, path: str | Path, signer: ManifestSigner, previous_entry_hash: str | None
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "quarantine_version": _QUARANTINE_VERSION,
        "quarantined_at": _utc_now(),
        **asdict(item),
        "path": str(path),
        "raw_base64": None if isinstance(raw, _OversizedLine) else base64.b64encode(raw).decode("ascii"),
        "previous_entry_hash": previous_entry_hash,
    }
    entry["signature"] = _signature_block(entry, signer)
    if entry["raw_base64"] is not None and len(_canonical_json(entry)) > MAX_LINE_BYTES:
        # Only the sha256 and length stay when the encoded bytes would push the trace line over the cap.
        entry["raw_base64"] = None
        entry["signature"] = _signature_block(entry, signer)
    return entry


def _verify_trace(
    path: str | Path,
    signer: ManifestSigner,
    signers: Mapping[str, ManifestSigner],
    anchored_heads: Iterable[str] = (),
) -> str | None:
    """Verify the trace lines (v1 only as a prefix), chain and `anchored_heads` without locking; return the head."""
    head: str | None = None
    hashes: set[str] = set()
    seen_v2 = False
    try:
        for _, entry in _read_ndjson(path):
            version = entry.get("quarantine_version")
            if type(version) is not int or version not in (_V1, _QUARANTINE_VERSION):
                raise ManifestVerificationError(f"unsupported quarantine_version {version!r}")
            if version == _V1 and seen_v2:
                raise ManifestVerificationError("quarantine_version 1 is not allowed after a quarantine_version 2 line")
            _check_signature(entry, signer, signers, version)
            if version != _V1 and entry.get("previous_entry_hash") != head:
                raise ManifestVerificationError(f"quarantine trace chain is broken in {path}")
            head = _sha256(_canonical_json(entry))
            hashes.add(head)
            seen_v2 = seen_v2 or version != _V1
    except FileNotFoundError:
        pass
    missing = _unverified_anchor(anchored_heads, hashes)
    if missing is not None:
        raise ManifestVerificationError(
            f"anchored head {missing!r} is not a line of the quarantine trace (head {head!r})"
        )
    return head


def verify_quarantine_log(
    quarantine_path: str | Path,
    *,
    signer: ManifestSigner,
    previous_signers: Iterable[ManifestSigner] = (),
    anchored_heads: Iterable[str] = (),
) -> str | None:
    """Raise ManifestVerificationError unless every trace line verifies and the chain is intact. Returns the hash of
    the last line (None for a missing or empty trace). Legacy v1 lines verify only as a prefix; they are unchained, so
    an anchor on one does not cover the lines before it. Every `anchored_heads` entry must be a line of the trace
    (catches truncation)."""
    signers = _signer_map(signer, previous_signers)
    with _flock(quarantine_path, exclusive=False):
        return _verify_trace(quarantine_path, signer, signers, anchored_heads)


def _trace_damage(
    damage: _Damage,
    *,
    quarantine_path: str | Path,
    signer: ManifestSigner,
    signers: Mapping[str, ManifestSigner],
    dry_run: bool,
    anchored_heads: Iterable[str] = (),
    head_anchor: HeadAnchor | None = None,
) -> None:
    """Trace `damage` to the signed, chained `quarantine_path` log, then anchor its head; a dry run only checks that
    log is terminated."""
    # Captured before the lock: an exclusive _flock opens O_CREAT, so afterwards the file always exists.
    existed = os.path.exists(quarantine_path)
    with _flock(quarantine_path, exclusive=not dry_run):
        _require_terminated(quarantine_path)
        if dry_run:
            return
        # Signed under the lock (each entry needs the previous hash): a failure must not leave a file it created.
        try:
            previous = _verify_trace(quarantine_path, signer, signers, anchored_heads)
            entries = []
            for item, raw, path in damage:
                entries.append(_trace_entry(item, raw, path, signer, previous))
                previous = _sha256(_canonical_json(entries[-1]))
        except BaseException:
            if not existed:
                with suppress(OSError):
                    _unlink_durably(quarantine_path)
            raise
        _append_with_rollback(quarantine_path, entries, existed=existed)
        if head_anchor is not None and previous is not None:
            head_anchor.write(previous)


def _rewrite_without(path: str | Path, drop: set[int]) -> None:
    """Atomically replace the real file behind `path` (a symlink stays one) with a copy minus the lines in `drop`."""
    target = Path(os.path.realpath(path))
    fd, temp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "wb") as out, open(target, "rb") as source:
            for number, raw in enumerate(_read_lines(source), start=1):
                if number in drop:
                    continue
                if isinstance(raw, _OversizedLine):
                    raise ValueError(f"{path} line {number} is longer than {MAX_LINE_BYTES} bytes and cannot be kept")
                out.write(raw)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temp, stat.S_IMODE(os.stat(target).st_mode))
        os.replace(temp, target)
    except BaseException:
        with suppress(OSError):
            os.unlink(temp)
        raise
    _fsync(target.parent)


def quarantine_damaged_lines(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    quarantine_path: str | Path,
    signer: ManifestSigner,
    previous_signers: Iterable[ManifestSigner] = (),
    expected_head: str | None = None,
    dry_run: bool = False,
    anchored_heads: Iterable[str] = (),
    head_anchor: HeadAnchor | None = None,
) -> list[QuarantinedLine]:
    """Repair a torn manifest tail and the audit lines the readers reject.

    - `dry_run=True` only reports; nothing is repaired or written.
    - A removed line is traced to the signed `quarantine_path` log first, fsynced before either file is touched.
      That write is not atomic with the repair itself: an interrupted run can leave a line traced but still
      present, and re-running it then appends a second trace entry for it.
    - The existing trace is verified before appending, so `previous_signers` must hold every key that signed it.
    - A repair creates `manifest_path` when it is missing, since the exclusive lock needs a writable file;
      `dry_run` creates nothing.
    - Stop every audit-file writer first: the sink appends without a lock.
    - A valid-JSON unterminated last line is refused in both files (it may be a real seal or record).
    - An anchored `expected_head` must match one of the complete manifests, which bounds how far back the log can
      have been rewound; re-verify the repaired log against the freshest anchor you track yourself.
    - A torn rotation entry is a torn manifest tail; call rotate_manifest_key again after the repair. A complete but
      wrongly appended one is quarantine_from_rotation_entry's job.
    - `anchored_heads` must all be lines of the existing trace (checked only when a repair appends, not on a dry run
      or with nothing to repair); `head_anchor` (the trace's own, never the manifest log's) gets the new trace head
      under the trace lock before either log is touched, so its failure leaves the logs unrepaired.
    - Anything else raises and changes nothing.

    `previous_signers` covers a retired signing key during rotation (see module docstring)."""
    signers = _signer_map(signer, previous_signers)
    _reject_aliased_paths(audit_path=audit_path, manifest_path=manifest_path, quarantine_path=quarantine_path)
    if not (os.path.exists(audit_path) or os.path.exists(manifest_path)):
        return []
    with _flock(manifest_path, exclusive=not dry_run):
        _refuse_interrupted_rotation(manifest_path, audit_path)
        manifest_damage = _damaged_manifest_tail(
            manifest_path, signer=signer, signers=signers, expected_head=expected_head
        )
        audit_damage = _damaged_audit_lines(audit_path)
        damage = manifest_damage + audit_damage
        if not damage:
            return []
        _trace_damage(
            damage,
            quarantine_path=quarantine_path,
            signer=signer,
            signers=signers,
            dry_run=dry_run,
            anchored_heads=anchored_heads,
            head_anchor=head_anchor,
        )
        if not dry_run:
            if audit_damage:
                _rewrite_without(audit_path, {item.line for item, _, _ in audit_damage})
            if manifest_damage:
                _truncate_durably(manifest_path, manifest_damage[0][0].offset)
        return [item for item, _, _ in damage]


def _manifests_through(path: str | Path, lines: Sequence[bytes], head: str) -> list[dict[str, Any]]:
    """Parse `lines` up to and including the manifest whose hash is `head`; raise if none has it."""
    manifests: list[dict[str, Any]] = []
    for manifest in _parsed_manifests(path, lines):
        manifests.append(manifest)
        if manifest_hash(manifest) == head:
            return manifests
    raise ManifestVerificationError(f"no complete manifest in {path} has the head {head!r}")


def quarantine_from_rotation_entry(
    manifest_path: str | Path,
    *,
    quarantine_path: str | Path,
    signer: ManifestSigner,
    previous_signers: Iterable[ManifestSigner] = (),
    expected_head: str,
    dry_run: bool = False,
    anchored_heads: Iterable[str] = (),
    head_anchor: HeadAnchor | None = None,
) -> list[QuarantinedLine]:
    """Drop a key rotation entry and everything after it from the manifest log; return the dropped lines.

    - `expected_head`: everything after this head is dropped, so dry-run first. It is the head of the line just before
      the entry (the entry's own `previous_manifest_hash` only if its chain is intact); an older head is refused
      unless a rotation entry follows it. `signer` must be the key current there, and the prefix must verify;
      re-verify the repaired log against the freshest anchor you track yourself.
    - The first dropped line must be a terminated JSON object with a `kind` field (what the log reads as a rotation
      entry); a torn tail is quarantine_damaged_lines' job. Neither its signature nor any later line is verified, so
      an entry signed outside the keyring can be dropped.
    - Each dropped line is traced to the signed `quarantine_path` log (its own file) before the manifest log is cut,
      and survives only there: keep that log on separate or append-only storage. A retired key with write access
      can drop honest seals too. The interrupted-run caveat of quarantine_damaged_lines applies.
    - The existing trace is verified before appending, so `previous_signers` must hold every key that signed it.
    - Runs sealed by a dropped line are unsealed again: review them against the trace before re-sealing with
      `seal_ndjson_runs(expected_head=<anchor>)`, and replace any external anchor recorded past `expected_head`.
    - `dry_run=True` only reports; nothing is written.
    - `anchored_heads` must all be lines of the existing trace (checked only when a repair appends, not on a dry run
      or with nothing to repair); `head_anchor` (the trace's own, never the manifest log's) gets the new trace head
      under the trace lock before either log is touched, so its failure leaves the logs unrepaired.
    - Anything else raises and changes nothing.

    `previous_signers` covers a retired signing key during rotation (see module docstring)."""
    signers = _signer_map(signer, previous_signers)
    if not isinstance(expected_head, str):
        raise ValueError("quarantine_from_rotation_entry expected_head must be a string")
    _reject_aliased_paths(manifest_path=manifest_path, quarantine_path=quarantine_path)
    # Checked before the lock: an exclusive _flock would create the file.
    if not os.path.exists(manifest_path):
        raise ManifestVerificationError(f"{manifest_path} does not exist, so there is no head to anchor on")
    with _flock(manifest_path, exclusive=not dry_run):
        _refuse_interrupted_rotation(manifest_path)
        lines, tail = _split_unterminated_tail(manifest_path)
        prefix = _manifests_through(manifest_path, lines, expected_head)
        _verify_log(prefix, signer=signer, signers=signers, expected_head=None, require_current=True)
        dropped = [*lines[len(prefix) :], *([tail] if tail else [])]
        if not dropped:
            return []
        number, offset = len(prefix) + 1, sum(map(_line_size, lines[: len(prefix)]))
        # Parsing refuses an unterminated first line; later lines, even a valid-JSON tail, are dropped unread.
        if "kind" not in _parse_ndjson_line(manifest_path, number, dropped[0])[1]:
            raise ManifestVerificationError(f"{manifest_path} line {number} is not a key rotation entry")
        reason = f"dropped from the key rotation entry at {manifest_path} line {number}"
        damage: _Damage = []
        for raw in dropped:
            damage.append(_damage_entry("manifest", manifest_path, number, offset, raw, reason))
            number += 1
            offset += _line_size(raw)
        _trace_damage(
            damage,
            quarantine_path=quarantine_path,
            signer=signer,
            signers=signers,
            dry_run=dry_run,
            anchored_heads=anchored_heads,
            head_anchor=head_anchor,
        )
        if not dry_run:
            _truncate_durably(manifest_path, damage[0][0].offset)
        return [item for item, _, _ in damage]
