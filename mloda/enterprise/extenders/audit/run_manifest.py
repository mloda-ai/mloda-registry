"""Run manifests: a signed, hash-chained seal over the audit records of one run.

Sealing is a post-run step: seal a run only once it is finished. A seal is final, so records reaching a sealed run
later fail verification by design. verify_ndjson_log returns the head: anchor it outside the log and pass it back
as expected_head, because removing the newest manifests or the whole log is otherwise undetectable.

Limits:
- HMAC is symmetric: integrity, not non-repudiation.
- Records without a usable run_id and unsealed runs sit outside every seal (verify_ndjson_log_coverage counts them).
- One key and one manifest log per audit file; sealers serialise on flock (POSIX).
- A torn or undecodable line fails sealing and verification until quarantine_damaged_lines repairs it. It repairs
  only a torn manifest tail and the audit lines the readers reject; anything else stays a hard failure.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import stat
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from mloda.enterprise.extenders.audit._records import _append_records, _canonical_json, _is_blank, _utc_now

_MANIFEST_VERSION = 1
_QUARANTINE_VERSION = 1
_HASH_ALGORITHM = "sha256"
_MIN_KEY_BYTES = 32
_SIGNATURE_KEYS = {"algorithm", "key_id", "value"}
_MAX_PROBLEMS = 20


class ManifestSigner(Protocol):
    """Signs and verifies manifest payloads; verify returns False instead of raising."""

    algorithm: str
    key_id: str

    def sign(self, payload: bytes) -> str: ...

    def verify(self, payload: bytes, signature: str) -> bool: ...


class HmacSha256Signer:
    """HMAC-SHA256 over a shared key of at least 32 bytes."""

    algorithm = "HMAC-SHA256"

    def __init__(self, key: bytes, key_id: str) -> None:
        if not isinstance(key, bytes) or len(key) < _MIN_KEY_BYTES:
            raise ValueError(f"HmacSha256Signer key must be bytes of at least {_MIN_KEY_BYTES} bytes")
        if not isinstance(key_id, str) or not key_id.strip():
            raise ValueError("HmacSha256Signer key_id must be a non-blank string")
        self._key = key
        self.key_id = key_id

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key_id={self.key_id!r})"

    def sign(self, payload: bytes) -> str:
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature: str) -> bool:
        # compare_digest raises TypeError on a non-ASCII str.
        return signature.isascii() and hmac.compare_digest(self.sign(payload), signature)


class ManifestVerificationError(ValueError):
    """A sealed manifest, record or line does not verify."""


class RunNotPendingError(ValueError):
    """The run_id has no audit records, or is already sealed."""


class RunAlreadySealedError(RunNotPendingError):
    """The run_id is already sealed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _RunDigest:
    def __init__(self) -> None:
        self.hashes: list[str] = []
        self.compliant = True

    def add(self, canonical: bytes, record: Mapping[str, Any]) -> None:
        self.hashes.append(_sha256(canonical))
        self.compliant = self.compliant and record.get("compliant") is True


class _Uncovered:
    def __init__(self) -> None:
        self.unattributed = 0
        self.by_run: Counter[str] = Counter()


def _digest_of(records: Iterable[Mapping[str, Any]]) -> _RunDigest:
    digest = _RunDigest()
    for record in records:
        digest.add(_canonical_json(record), record)
    return digest


def _unsigned(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "signature"}


def _signature_block(payload: Mapping[str, Any], signer: ManifestSigner) -> dict[str, str]:
    return {"algorithm": signer.algorithm, "key_id": signer.key_id, "value": signer.sign(_canonical_json(payload))}


def _seal(
    run_id: str, digest: _RunDigest, *, signer: ManifestSigner, previous_manifest_hash: str | None
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "manifest_version": _MANIFEST_VERSION,
        "run_id": run_id,
        "sealed_at": _utc_now(),
        "hash_algorithm": _HASH_ALGORITHM,
        "record_count": len(digest.hashes),
        # Sorted: writers append in any order.
        "record_hashes": sorted(digest.hashes),
        "compliant": digest.compliant,
        "previous_manifest_hash": previous_manifest_hash,
    }
    manifest["signature"] = _signature_block(manifest, signer)
    return manifest


def seal_run(
    records: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    signer: ManifestSigner,
    previous_manifest_hash: str | None = None,
) -> dict[str, Any]:
    records = list(records)
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("seal_run run_id must be a non-blank string")
    if not records:
        raise ValueError(f"seal_run got no records for run_id {run_id!r}")
    if any(record.get("run_id") != run_id for record in records):
        raise ValueError(f"seal_run got a record that does not belong to run_id {run_id!r}")
    return _seal(run_id, _digest_of(records), signer=signer, previous_manifest_hash=previous_manifest_hash)


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """The value the next manifest chains to."""
    return _sha256(_canonical_json(manifest))


def _verify_manifest_fields(manifest: Mapping[str, Any], signer: ManifestSigner) -> None:
    signature = manifest.get("signature")
    # The block is unsigned, so an unknown member could carry anything.
    if not isinstance(signature, Mapping) or set(signature) != _SIGNATURE_KEYS:
        raise ManifestVerificationError(f"manifest signature block must have exactly {sorted(_SIGNATURE_KEYS)}")
    for field in ("algorithm", "key_id"):
        if signature[field] != getattr(signer, field):
            raise ManifestVerificationError(
                f"signature {field} {signature[field]!r} is not the signer's {getattr(signer, field)!r}"
            )
    value = signature["value"]
    if not isinstance(value, str) or not signer.verify(_canonical_json(_unsigned(manifest)), value):
        raise ManifestVerificationError("signature does not match the manifest")
    # type() is int: True and 1.0 equal 1.
    version = manifest.get("manifest_version")
    if type(version) is not int or version != _MANIFEST_VERSION:
        raise ManifestVerificationError(f"unsupported manifest_version {version!r}")
    if manifest.get("hash_algorithm") != _HASH_ALGORITHM:
        raise ManifestVerificationError(f"unsupported hash_algorithm {manifest.get('hash_algorithm')!r}")
    hashes = manifest.get("record_hashes")
    count = manifest.get("record_count")
    if not isinstance(hashes, list) or type(count) is not int or count != len(hashes):
        raise ManifestVerificationError(f"record_count {count!r} is not the number of hashes")
    if not isinstance(manifest.get("run_id"), str):
        raise ManifestVerificationError(f"run_id {manifest.get('run_id')!r} is not a string")


def _verify_digest(manifest: Mapping[str, Any], digest: _RunDigest) -> None:
    # Lists, not sets: a repeated record must not pass.
    if sorted(digest.hashes) != manifest["record_hashes"]:
        raise ManifestVerificationError(f"records of run_id {manifest['run_id']!r} do not match the record_hashes")
    if digest.compliant is not manifest.get("compliant"):
        raise ManifestVerificationError(
            f"compliant {manifest.get('compliant')!r} of run_id {manifest['run_id']!r} is not what its records give"
        )


def verify_manifest(
    manifest: Mapping[str, Any], records: Iterable[Mapping[str, Any]], *, signer: ManifestSigner
) -> None:
    """Raise ManifestVerificationError unless signed by `signer` and matching `records`."""
    _verify_manifest_fields(manifest, signer)
    _verify_digest(manifest, _digest_of(records))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # json.loads keeps the last duplicate, a first-wins reader the first: reject.
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError(f"JSON object repeats a key among {sorted(obj)}")
    return obj


def _decode_line(where: str, line: bytes) -> Any:
    try:
        return json.loads(line.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (ValueError, RecursionError) as exc:
        raise ManifestVerificationError(f"{where} is not valid JSON: {exc}") from exc


def _parse_ndjson_line(path: str | Path, number: int, raw: bytes) -> tuple[bytes, dict[str, Any]]:
    where = f"{path} line {number}"
    if not raw.endswith(b"\n"):
        raise ManifestVerificationError(f"{where} does not end with a newline")
    line = raw[:-1]
    obj = _decode_line(where, line)
    if not isinstance(obj, dict):
        raise ManifestVerificationError(f"{where} is not a JSON object")
    return line, obj


def _read_ndjson(path: str | Path) -> Iterator[tuple[bytes, dict[str, Any]]]:
    with open(path, "rb") as file:
        for number, raw in enumerate(file, start=1):
            yield _parse_ndjson_line(path, number, raw)


def _iter_manifests(path: str | Path) -> Iterator[dict[str, Any]]:
    try:
        for _, manifest in _read_ndjson(path):
            yield manifest
    except FileNotFoundError:
        return


def _read_manifests(path: str | Path) -> list[dict[str, Any]]:
    return list(_iter_manifests(path))


def _digest_runs(
    audit_path: str | Path, keep: Callable[[str], bool], uncovered: _Uncovered | None = None
) -> dict[str, _RunDigest]:
    if uncovered is None:
        uncovered = _Uncovered()
    digests: dict[str, _RunDigest] = {}
    for line, record in _read_ndjson(audit_path):
        run_id = record.get("run_id")
        if run_id is not None and not isinstance(run_id, str):
            raise ManifestVerificationError(f"audit record run_id {run_id!r} is neither a string nor null")
        if run_id is None or _is_blank(run_id):
            uncovered.unattributed += 1
            continue
        if not keep(run_id):
            uncovered.by_run[run_id] += 1
            continue
        # The raw line, not a re-serialisation: the seal covers the written bytes.
        digests.setdefault(run_id, _RunDigest()).add(line, record)
    return digests


@contextmanager
def _flock(path: str | Path, *, exclusive: bool) -> Iterator[None]:
    """flock `path`; a shared lock is best-effort, an exclusive one raises on OSError. No lock without fcntl."""
    fd: int | None = None
    try:
        try:
            import fcntl

            # NFS needs a writable descriptor for an exclusive lock.
            fd = os.open(path, (os.O_RDWR | os.O_CREAT) if exclusive else os.O_RDONLY, 0o600)
        except ImportError:
            pass
        except FileNotFoundError:
            if exclusive:
                raise
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            except OSError:
                if exclusive:
                    raise
        yield
    finally:
        if fd is not None:
            os.close(fd)


def _verify_log(
    log: Iterable[Mapping[str, Any]],
    *,
    signer: ManifestSigner,
    expected_head: str | None,
    digests: Mapping[str, _RunDigest] | None = None,
) -> tuple[set[str], str | None]:
    """Record and head mismatches are collected into one error (head first); a structural failure raises at once."""
    sealed: set[str] = set()
    head: str | None = None
    problems: list[str] = []
    for manifest in log:
        _verify_manifest_fields(manifest, signer)
        run_id: str = manifest["run_id"]
        if manifest.get("previous_manifest_hash") != head:
            raise ManifestVerificationError(f"manifest chain is broken at run_id {run_id!r}")
        if run_id in sealed:
            raise ManifestVerificationError(f"run_id {run_id!r} is sealed by more than one manifest")
        if digests is not None:
            try:
                _verify_digest(manifest, digests.get(run_id, _RunDigest()))
            except ManifestVerificationError as exc:
                problems.append(str(exc))
        sealed.add(run_id)
        head = manifest_hash(manifest)
    if expected_head is not None and head != expected_head:
        problems.insert(0, f"manifest log head {head!r} is not the expected head {expected_head!r}")
    if problems:
        message = "; ".join(problems[:_MAX_PROBLEMS])
        if len(problems) > _MAX_PROBLEMS:
            message += f"; and {len(problems) - _MAX_PROBLEMS} more"
        raise ManifestVerificationError(message)
    return sealed, head


def seal_ndjson_runs(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    run_id: str | None = None,
    expected_head: str | None = None,
) -> list[dict[str, Any]]:
    """Seal every unsealed run (or only `run_id`). Raises RunAlreadySealedError for a sealed `run_id` (catch that, not
    ValueError, for an idempotent retry) and RunNotPendingError when it has no records."""
    if run_id is not None and (not isinstance(run_id, str) or _is_blank(run_id)):
        raise ValueError("seal_ndjson_runs run_id must be a non-blank string")
    with _flock(manifest_path, exclusive=True):
        sealed, head = _verify_log(_iter_manifests(manifest_path), signer=signer, expected_head=expected_head)
        if run_id is not None and run_id in sealed:
            raise RunAlreadySealedError(f"run_id {run_id!r} is already sealed in {manifest_path}")
        digests = _digest_runs(
            audit_path, lambda candidate: candidate == run_id if run_id is not None else candidate not in sealed
        )
        if run_id is not None and run_id not in digests:
            raise RunNotPendingError(f"no audit records for run_id {run_id!r} in {audit_path}")

        manifests = []
        for pending_run_id, digest in digests.items():
            manifest = _seal(pending_run_id, digest, signer=signer, previous_manifest_hash=head)
            manifests.append(manifest)
            head = manifest_hash(manifest)
        size = os.path.getsize(manifest_path) if os.path.exists(manifest_path) else 0
        try:
            _append_records(manifest_path, manifests)
        except BaseException:
            # Under the lock, so only this sealer's partial bytes go.
            with suppress(OSError):
                os.truncate(manifest_path, size)
            raise
        return manifests


@dataclass(frozen=True)
class LogCoverage:
    """What a verification covered; unsealed_lines maps each unsealed run_id to its line count."""

    head: str | None
    sealed_runs: int
    sealed_lines: int
    unattributed_lines: int
    unsealed_lines: dict[str, int]

    def __hash__(self) -> int:
        return hash(
            (
                self.head,
                self.sealed_runs,
                self.sealed_lines,
                self.unattributed_lines,
                tuple(sorted(self.unsealed_lines.items())),
            )
        )


def verify_ndjson_log_coverage(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    expected_head: str | None = None,
) -> LogCoverage:
    """Verify like verify_ndjson_log; return a LogCoverage (head, sealed_runs, sealed_lines, unattributed_lines,
    unsealed_lines)."""
    # Log first: a run sealed after this read looks unsealed, not tampered.
    with _flock(manifest_path, exclusive=False):
        manifests = _read_manifests(manifest_path)
    sealed = {manifest["run_id"] for manifest in manifests if isinstance(manifest.get("run_id"), str)}
    uncovered = _Uncovered()
    try:
        digests = _digest_runs(audit_path, sealed.__contains__, uncovered)
    except FileNotFoundError:
        digests = {}
    _, head = _verify_log(manifests, signer=signer, expected_head=expected_head, digests=digests)
    return LogCoverage(
        head=head,
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
    expected_head: str | None = None,
) -> str | None:
    """Raise ManifestVerificationError unless every manifest matches the audit bytes of its run. Returns the head:
    anchor it outside the log and pass it back as `expected_head`, or a truncated log goes undetected."""
    return verify_ndjson_log_coverage(audit_path, manifest_path, signer=signer, expected_head=expected_head).head


@dataclass(frozen=True)
class QuarantinedLine:
    """A line removed by quarantine_damaged_lines; line (1-based) and offset are from before the repair."""

    file: str
    line: int
    offset: int
    length: int
    sha256: str
    reason: str


_Damage = list[tuple[QuarantinedLine, bytes, Path]]


def _refuse_json_tail(path: str | Path, number: int, tail: bytes) -> None:
    where = f"{path} line {number}"
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
            if not raw.endswith(b"\n"):
                _refuse_json_tail(path, number, raw)
            damage.append((QuarantinedLine(file, number, offset, len(raw), _sha256(raw), str(exc)), raw, Path(path)))
        number += 1
        offset += len(raw)
    return damage


def _damaged_audit_lines(path: str | Path) -> _Damage:
    try:
        with open(path, "rb") as file:
            return _damaged_lines("audit", path, file)
    except FileNotFoundError:
        return []


def _damaged_manifest_tail(path: str | Path, *, signer: ManifestSigner, expected_head: str | None) -> _Damage:
    """`expected_head` may be any complete manifest's head: a torn multi-seal write leaves complete ones past it."""
    try:
        with open(path, "rb") as file:
            lines = list(file)
    except FileNotFoundError:
        lines = []
    tail = lines.pop() if lines and not lines[-1].endswith(b"\n") else b""
    manifests = [_parse_ndjson_line(path, number, raw)[1] for number, raw in enumerate(lines, start=1)]
    _, head = _verify_log(manifests, signer=signer, expected_head=None)
    if expected_head is not None and expected_head not in {manifest_hash(manifest) for manifest in manifests}:
        raise ManifestVerificationError(
            f"manifest log head {head!r} is not the expected head {expected_head!r}, and no earlier manifest has it"
        )
    if not tail:
        return []
    return _damaged_lines("manifest", path, [tail], number=len(lines) + 1, offset=sum(map(len, lines)))


def _trace_entry(item: QuarantinedLine, raw: bytes, path: str | Path, signer: ManifestSigner) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "quarantine_version": _QUARANTINE_VERSION,
        "quarantined_at": _utc_now(),
        **asdict(item),
        "path": str(path),
        "raw_base64": base64.b64encode(raw).decode("ascii"),
    }
    entry["signature"] = _signature_block(entry, signer)
    return entry


def _fsync(path: str | Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _require_terminated(path: str | Path) -> None:
    try:
        with open(path, "rb") as file:
            if file.seek(0, os.SEEK_END):
                file.seek(-1, os.SEEK_END)
                if file.read(1) != b"\n":
                    raise ManifestVerificationError(f"{path} does not end with a newline; repair it before a recovery")
    except FileNotFoundError:
        return


def _append_trace(path: str | Path, entries: list[dict[str, Any]]) -> None:
    """Append and fsync; a failure restores the file, or removes it if it did not exist."""
    size = os.path.getsize(path) if os.path.exists(path) else None
    try:
        _append_records(path, entries)
        _fsync(path)
        _fsync(Path(path).parent)
    except BaseException:
        with suppress(OSError):
            if size is None:
                os.unlink(path)
            else:
                os.truncate(path, size)
        raise


def _rewrite_without(path: str | Path, drop: set[int]) -> None:
    """Atomically replace the real file behind `path` (a symlink stays one) with a copy minus the lines in `drop`."""
    target = Path(os.path.realpath(path))
    fd, temp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "wb") as out, open(target, "rb") as source:
            out.writelines(raw for number, raw in enumerate(source, start=1) if number not in drop)
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
    expected_head: str | None = None,
    dry_run: bool = False,
) -> list[QuarantinedLine]:
    """Repair a torn manifest tail and the audit lines the readers reject; `dry_run` only reports.

    Every removed line goes first to the signed `quarantine_path` log. Stop every audit-file writer first: the sink
    appends without a lock. A valid-JSON unterminated last line is refused in both files (it may be a real seal or
    record). Without an anchored `expected_head`, truncating a torn manifest tail is indistinguishable from cutting
    the newest manifest. Anything else raises and changes nothing."""
    if os.path.realpath(quarantine_path) in {os.path.realpath(audit_path), os.path.realpath(manifest_path)}:
        raise ValueError("quarantine_path must not be the audit file or the manifest log")
    if not (os.path.exists(audit_path) or os.path.exists(manifest_path)):
        return []
    with _flock(manifest_path, exclusive=not dry_run):
        manifest_damage = _damaged_manifest_tail(manifest_path, signer=signer, expected_head=expected_head)
        audit_damage = _damaged_audit_lines(audit_path)
        damage = manifest_damage + audit_damage
        if not damage:
            return []
        _require_terminated(quarantine_path)
        if not dry_run:
            _append_trace(quarantine_path, [_trace_entry(item, raw, path, signer) for item, raw, path in damage])
            if audit_damage:
                _rewrite_without(audit_path, {item.line for item, _, _ in audit_damage})
            if manifest_damage:
                os.truncate(manifest_path, manifest_damage[0][0].offset)
        return [item for item, _, _ in damage]
