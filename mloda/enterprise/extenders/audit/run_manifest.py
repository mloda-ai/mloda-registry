"""Run manifests: a signed, hash-chained seal over the audit records of one run.

Sealing is a post-run step. Workers hold their own extender copy and records arrive unordered from several
writers, so the chain cannot run through the records. It links per-run manifests instead, written by a sealer
in the parent process after run_all returns. Sealers of one manifest log serialise on an exclusive flock (POSIX;
the wait has no timeout). Where fcntl is missing they race and verify_ndjson_log reports a forked chain.
Verifying takes a shared lock on an existing log.

A seal is final, so seal a run_id only once it is finished. Records that reach a sealed run later fail
verification by design: a second session.run() on one prepared session reuses the run_id, and a run still in
flight in another thread or process that shares the audit file keeps writing. For a prepared session pass
run_id=session.run_id. The sealer verifies the manifest log but not the records of sealed runs;
verify_ndjson_log does, over the bytes of each audit line where verify_manifest compares records.

Limits:
- HMAC is symmetric, so HmacSha256Signer proves integrity to key holders and not non-repudiation; an asymmetric
  or KMS signer plugs in through ManifestSigner.
- Records without a usable run_id (null, absent, blank) and runs that are not sealed yet are outside every seal;
  verify_ndjson_log_coverage counts them.
- Removing the newest manifests or the whole log cannot be detected from the files alone, and a truncated log
  lets the next sealing run re-seal altered records. Anchor the head outside the log (verify_ndjson_log returns
  it, manifest_hash of the last sealed manifest gives it) and pass it back as expected_head.
- A failed append is rolled back; a crash mid-append can still leave a torn last line. An unterminated or
  undecodable line in the manifest log or the audit file fails sealing and verification for every run.
  quarantine_damaged_lines repairs a torn manifest tail and the audit lines the readers reject, and refuses the
  rest. A deleted audit file fails verification of every sealed run.
- One key and one manifest log per audit file: the payload carries no log identity and a log cannot span a key
  rotation. Start a new audit file and manifest log for a new key.
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

from mloda.enterprise.extenders.audit.audit_extender import _append_records, _canonical_json, _is_missing, _utc_now

_MANIFEST_VERSION = 1
_QUARANTINE_VERSION = 1
_HASH_ALGORITHM = "sha256"
_MIN_KEY_BYTES = 32
_SIGNATURE_KEYS = {"algorithm", "key_id", "value"}
_MAX_PROBLEMS = 20


class ManifestSigner(Protocol):
    """Signs and verifies manifest payloads; algorithm and key_id are recorded next to each signature.
    verify returns False for a signature it cannot compare and never raises."""

    algorithm: str
    key_id: str

    def sign(self, payload: bytes) -> str: ...

    def verify(self, payload: bytes, signature: str) -> bool: ...


class HmacSha256Signer:
    """HMAC-SHA256 over a shared key of at least 32 bytes; signatures are lowercase hex."""

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
        # compare_digest raises TypeError on a non-ASCII str; a hex signature is ASCII, so anything else is wrong.
        return signature.isascii() and hmac.compare_digest(self.sign(payload), signature)


class ManifestVerificationError(ValueError):
    """A manifest, its records, the manifest chain or a line of either file is not what was sealed."""


class RunNotPendingError(ValueError):
    """The run_id given to seal_ndjson_runs has no audit records or is already sealed."""


class RunAlreadySealedError(RunNotPendingError):
    """The run_id given to seal_ndjson_runs is already sealed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _RunDigest:
    """What a manifest seals of one run: a hash per record line and whether every record is compliant."""

    def __init__(self) -> None:
        self.hashes: list[str] = []
        self.compliant = True

    def add(self, canonical: bytes, record: Mapping[str, Any]) -> None:
        self.hashes.append(_sha256(canonical))
        self.compliant = self.compliant and record.get("compliant") is True


class _Uncovered:
    """The audit lines a scan leaves out of every digest: those without a usable run_id, and per run those `keep`
    refused."""

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
        # Sorted, so the hashes do not depend on the order the writers happened to append in.
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
    """Build the signed manifest of one run; the signature covers every key except `signature` itself."""
    records = list(records)
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("seal_run run_id must be a non-blank string")
    if not records:
        raise ValueError(f"seal_run got no records for run_id {run_id!r}")
    if any(record.get("run_id") != run_id for record in records):
        raise ValueError(f"seal_run got a record that does not belong to run_id {run_id!r}")
    return _seal(run_id, _digest_of(records), signer=signer, previous_manifest_hash=previous_manifest_hash)


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """The value the next manifest chains to; covers the signature as well."""
    return _sha256(_canonical_json(manifest))


def _verify_manifest_fields(manifest: Mapping[str, Any], signer: ManifestSigner) -> None:
    """The checks that need no records: the signature first, then the signed fields."""
    signature = manifest.get("signature")
    # The block sits outside the signed payload: an unknown member could carry anything unnoticed, and
    # algorithm and key_id can only be compared with the signer directly.
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
    # type() is int, not ==: True and 1.0 equal 1 but are not what the sealer wrote.
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
    # Sorted lists, not sets: a repeated record must not pass.
    if sorted(digest.hashes) != manifest["record_hashes"]:
        raise ManifestVerificationError(f"records of run_id {manifest['run_id']!r} do not match the record_hashes")
    if digest.compliant is not manifest.get("compliant"):
        raise ManifestVerificationError(
            f"compliant {manifest.get('compliant')!r} of run_id {manifest['run_id']!r} is not what its records give"
        )


def verify_manifest(
    manifest: Mapping[str, Any], records: Iterable[Mapping[str, Any]], *, signer: ManifestSigner
) -> None:
    """Raise ManifestVerificationError unless the manifest is signed by `signer` and matches `records`."""
    _verify_manifest_fields(manifest, signer)
    _verify_digest(manifest, _digest_of(records))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # json.loads keeps the last duplicate while a first-wins reader sees another object, so neither is trusted.
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
    """The line without its newline, plus the object it parses to; the one check both readers and the recovery
    apply. Raises ManifestVerificationError naming the file and line."""
    where = f"{path} line {number}"
    if not raw.endswith(b"\n"):
        raise ManifestVerificationError(f"{where} does not end with a newline")
    line = raw[:-1]
    obj = _decode_line(where, line)
    if not isinstance(obj, dict):
        raise ManifestVerificationError(f"{where} is not a JSON object")
    return line, obj


def _read_ndjson(path: str | Path) -> Iterator[tuple[bytes, dict[str, Any]]]:
    """Yield each line without its newline, plus the object it parses to; every line must end in a newline."""
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
    """Keyed in order of first appearance in the audit file, which is the sealing order. Streams the file, keeping
    only the runs `keep` accepts; records without a usable run_id are outside every seal. The lines it leaves out
    are counted into `uncovered` when given."""
    if uncovered is None:
        uncovered = _Uncovered()
    digests: dict[str, _RunDigest] = {}
    for line, record in _read_ndjson(audit_path):
        run_id = record.get("run_id")
        if run_id is not None and not isinstance(run_id, str):
            raise ManifestVerificationError(f"audit record run_id {run_id!r} is neither a string nor null")
        if run_id is None or _is_missing(run_id):
            uncovered.unattributed += 1
            continue
        if not keep(run_id):
            uncovered.by_run[run_id] += 1
            continue
        # The raw line, not a re-serialisation: the seal covers the bytes that were written.
        digests.setdefault(run_id, _RunDigest()).add(line, record)
    return digests


@contextmanager
def _flock(path: str | Path, *, exclusive: bool) -> Iterator[None]:
    """Block on an flock of `path`. Exclusive creates the file and raises when it cannot lock. No lock where fcntl
    is missing, for a shared lock on a file that does not exist, nor for one the file system refuses (ENOLCK,
    ENOSYS, EOPNOTSUPP)."""
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
    """The one pass over a manifest log: every manifest, the chain, no repeated run_id and the head. Compares
    the records as well when `digests` is given; record and head mismatches are collected into one error (the
    head first, so the problem cap never hides it), a structural failure raises at once. Returns the sealed run
    ids and the head."""
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
    """Seal every unsealed run of the audit file (or only `run_id`), in order of first appearance, and append
    the manifests to the log in one write under an exclusive lock, rolled back if the write fails. All are signed
    before any is written. The log is verified first; the records of sealed runs are not. Raises
    RunAlreadySealedError when `run_id` is already sealed: catch that, not ValueError, for an idempotent retry
    (ManifestVerificationError is a ValueError too). Raises RunNotPendingError when `run_id` has no records,
    usually a bug, and FileNotFoundError when the audit file is missing."""
    if run_id is not None and (not isinstance(run_id, str) or _is_missing(run_id)):
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
            # The lock makes this sealer the only writer, so this removes only its own partial bytes.
            with suppress(OSError):
                os.truncate(manifest_path, size)
            raise
        return manifests


@dataclass(frozen=True)
class LogCoverage:
    """What a successful verification covered. head is what verify_ndjson_log returns; sealed_lines are the audit
    lines of the sealed_runs. Every other line is outside every seal: unattributed_lines have a null, absent or
    blank run_id, and unsealed_lines maps each run that is not sealed yet to its line count, in order of first
    appearance."""

    head: str | None
    sealed_runs: int
    sealed_lines: int
    unattributed_lines: int
    unsealed_lines: dict[str, int]


def verify_ndjson_log_coverage(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    expected_head: str | None = None,
) -> LogCoverage:
    """Verify exactly like verify_ndjson_log (same checks, errors and lock) and return the head together with
    the counts of the audit lines the seals cover and leave out."""
    # The log first, under a shared lock so no sealer is mid-append; a run sealed after this read then looks
    # unsealed, where the reverse order would make it look like tampering.
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
    """Raise ManifestVerificationError unless every manifest verifies against the audit bytes of its run,
    chains to its predecessor and seals a run_id of its own; every failing run is named in the one error.
    Runs that are not sealed yet are ignored, and a missing audit file is treated as empty. Returns the head, the
    manifest_hash of the last manifest (None for an empty or missing log), to anchor outside the log;
    verify_ndjson_log_coverage also reports how many lines the seals cover."""
    return verify_ndjson_log_coverage(audit_path, manifest_path, signer=signer, expected_head=expected_head).head


@dataclass(frozen=True)
class QuarantinedLine:
    """A line quarantine_damaged_lines removes. file is "audit" or "manifest"; line is 1-based and offset is the
    byte where the line starts, both before the repair; length counts the bytes removed, newline included when
    the line had one, and sha256 is the hex digest of exactly those bytes; reason is the reader's failure message."""

    file: str
    line: int
    offset: int
    length: int
    sha256: str
    reason: str


_Damage = list[tuple[QuarantinedLine, bytes]]


def _refuse_json_tail(path: str | Path, number: int, tail: bytes) -> None:
    """An unterminated last line that parses as JSON may be a real seal or record that only lost its newline."""
    where = f"{path} line {number}"
    try:
        _decode_line(where, tail)
    except ManifestVerificationError:
        return
    raise ManifestVerificationError(f"{where} is unterminated but valid JSON, so it may be a real seal or record")


def _damaged_lines(file: str, path: str | Path, raws: Iterable[bytes], *, number: int = 1, offset: int = 0) -> _Damage:
    """The `raws` the readers reject, which start at line `number` and byte `offset` of the file, with their bytes.
    Raises for an unterminated line that is valid JSON, in either file."""
    damage: _Damage = []
    for raw in raws:
        try:
            _parse_ndjson_line(path, number, raw)
        except ManifestVerificationError as exc:
            if not raw.endswith(b"\n"):
                _refuse_json_tail(path, number, raw)
            damage.append((QuarantinedLine(file, number, offset, len(raw), _sha256(raw), str(exc)), raw))
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
    """Verify the complete lines of the manifest log, then return its unterminated last fragment as the one damage
    that may be repaired. `expected_head` may be the head of any complete manifest: a torn write of several seals
    leaves complete ones past the anchor."""
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
    """An entry appended to a file without a final newline would run into its last line."""
    try:
        with open(path, "rb") as file:
            if file.seek(0, os.SEEK_END):
                file.seek(-1, os.SEEK_END)
                if file.read(1) != b"\n":
                    raise ManifestVerificationError(f"{path} does not end with a newline; repair it before a recovery")
    except FileNotFoundError:
        return


def _append_trace(path: str | Path, entries: list[dict[str, Any]]) -> None:
    """Append `entries` and fsync the file and its directory. A failure restores the file, or removes it when it did
    not exist, so no partial line stays."""
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
    """Replace the real file behind `path` (a symlink stays one) by a copy without the 1-based lines `drop`: an
    owner-only temporary file beside it, given the original permission bits, fsynced and renamed over it, then the
    directory fsynced. The temporary file is removed again when anything fails."""
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
    """Repair a torn manifest tail and the audit lines the readers reject, so sealing and verification work again,
    and record every removed byte. In the audit file every line the readers reject is removed and the others keep
    their bytes and order; a missing file has nothing to repair, and a non-string run_id is not damage. In the
    manifest log only an unterminated last fragment is truncated, once every complete line and the chain verify
    against `signer`. `expected_head` may be the head of any complete manifest. Anything else raises
    ManifestVerificationError and changes nothing: a manifest log that does not verify, a damaged manifest line
    that ends in a newline, a quarantine log without a final newline, and in either file an unterminated last line
    that parses as JSON (it may be a real seal or record: check it and append its newline by hand). A
    quarantine_path that is the audit file or the manifest log raises ValueError.

    Without an anchored `expected_head`, truncating a torn manifest tail is indistinguishable from someone cutting
    the newest manifest, and the next sealing run re-seals that run: anchor the head and pass it.

    One signed line per removed line, holding its bytes, is appended to `quarantine_path` and fsynced before either
    file changes; then the audit file is replaced atomically and the manifest log truncated. A removed line of a
    sealed run still fails verification, and the trace names it. Returns the removed lines, manifest first, then
    audit in file order. dry_run returns and raises the same but writes and creates nothing. The exclusive lock on
    the manifest log (shared for dry_run) covers verifying, the trace write and the swap.

    Stop every writer of the audit file first: the sink appends without a lock, so a write racing the swap is lost."""
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
            paths = {"manifest": manifest_path, "audit": audit_path}
            _append_trace(quarantine_path, [_trace_entry(item, raw, paths[item.file], signer) for item, raw in damage])
            if audit_damage:
                _rewrite_without(audit_path, {item.line for item, _ in audit_damage})
            if manifest_damage:
                os.truncate(manifest_path, manifest_damage[0][0].offset)
        return [item for item, _ in damage]
