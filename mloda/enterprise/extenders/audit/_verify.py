"""Run manifest core: errors, line IO, sealing a run, verification and head anchors."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from collections.abc import Callable, Container, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import IO, Any, Protocol

from mloda.enterprise.extenders.audit._records import (
    _append_records,
    _canonical_json,
    _is_blank,
    _open_locked,
    _parse_event_time,
    _utc_now,
)
from mloda.enterprise.extenders.audit._signers import (
    ManifestSigner,
    _signer_map,
)

logger = logging.getLogger("mloda.enterprise.extenders.audit.run_manifest")


_MANIFEST_VERSION = 2
MAX_LINE_BYTES = 64 * 1024 * 1024
_V1 = 1
_HASH_ALGORITHM = "sha256"
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_SIGNATURE_KEYS = {"algorithm", "key_id", "value"}
_ROTATION_KIND = "key_rotation"
_ROTATION_KEYS_V1 = {"manifest_version", "kind", "rotated_at", "previous_manifest_hash", "signature"}
_ROTATION_KEYS = _ROTATION_KEYS_V1 | {"previous_key_signature"}
_GENESIS_KIND = "genesis"
_GENESIS_KEYS = {"manifest_version", "kind", "log_id", "created_at", "previous_manifest_hash", "signature"}
_MAX_PROBLEMS = 20


class ManifestVerificationError(ValueError):
    """A sealed manifest, record or line does not verify."""


class RunNotPendingError(ValueError):
    """The run_id has no audit records, or is already sealed."""


class RunAlreadySealedError(RunNotPendingError):
    """The run_id is already sealed."""


class KeyAlreadyCurrentError(ValueError):
    """The manifest log is already under the signer's key."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _RunDigest:
    def __init__(self) -> None:
        self.hashes: list[str] = []
        self.compliant = True
        self.newest: datetime | None = None
        self.undated = False

    def note_time(self, record: Mapping[str, Any]) -> None:
        """Track the newest event_time; one record without a parseable one makes the run undated."""
        try:
            moment = _parse_event_time(record.get("event_time"))
        except ValueError:
            self.undated = True
            return
        self.newest = moment if self.newest is None or moment > self.newest else self.newest

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


def _signing_payload(entry: Mapping[str, Any]) -> bytes:
    """The v2 signed bytes: the entry with each signature block reduced to its algorithm and key_id."""
    reduced = dict(entry)
    fields = ("signature", "previous_key_signature") if entry.get("kind") == _ROTATION_KIND else ("signature",)
    for field in fields:
        block = reduced.get(field)
        if isinstance(block, Mapping):
            reduced[field] = {"algorithm": block.get("algorithm"), "key_id": block.get("key_id")}
    return _canonical_json(reduced)


def _signature_block(payload: Mapping[str, Any], signer: ManifestSigner) -> dict[str, str]:
    block = {"algorithm": signer.algorithm, "key_id": signer.key_id}
    return {**block, "value": signer.sign(_signing_payload({**payload, "signature": block}))}


def _seal(
    run_id: str,
    digest: _RunDigest,
    *,
    signer: ManifestSigner,
    previous_manifest_hash: str | None,
    sealed_late: bool = True,
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
        "sealed_late": sealed_late,
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
    sealed_late: bool = True,
) -> dict[str, Any]:
    records = list(records)
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("seal_run run_id must be a non-blank string")
    if not records:
        raise ValueError(f"seal_run got no records for run_id {run_id!r}")
    if any(record.get("run_id") != run_id for record in records):
        raise ValueError(f"seal_run got a record that does not belong to run_id {run_id!r}")
    return _seal(
        run_id,
        _digest_of(records),
        signer=signer,
        previous_manifest_hash=previous_manifest_hash,
        sealed_late=sealed_late,
    )


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """The value the next manifest chains to."""
    return _sha256(_canonical_json(manifest))


def _verify_signed(
    manifest: Mapping[str, Any],
    signer: ManifestSigner,
    signers: Mapping[str, ManifestSigner],
    *,
    allow_v1: bool = True,
) -> None:
    """The manifest_version and signature checks shared by manifests and rotation entries."""
    version = manifest.get("manifest_version")
    # type() is int: True and 1.0 equal 1.
    if type(version) is not int or version not in (_V1, _MANIFEST_VERSION):
        raise ManifestVerificationError(f"unsupported manifest_version {version!r}")
    if version == _V1 and not allow_v1:
        raise ManifestVerificationError("manifest_version 1 is not allowed after a manifest_version 2 line")
    _check_signature(manifest, signer, signers, version)


def _check_signature(
    manifest: Mapping[str, Any], signer: ManifestSigner, signers: Mapping[str, ManifestSigner], version: int
) -> None:
    """The signature block and value checks; a v1 line signs the entry minus its signature, v2 `_signing_payload`."""
    signature = manifest.get("signature")
    # The block is unsigned, so an unknown member could carry anything.
    if not isinstance(signature, Mapping) or set(signature) != _SIGNATURE_KEYS:
        raise ManifestVerificationError(f"manifest signature block must have exactly {sorted(_SIGNATURE_KEYS)}")
    key_id = signature["key_id"]
    resolved: ManifestSigner | None
    try:
        resolved = signers.get(key_id)
    except TypeError:
        resolved = None
    if resolved is None:
        if len(signers) > 1:
            raise ManifestVerificationError(f"signature key_id {key_id!r} matches no known key")
        raise ManifestVerificationError(f"signature key_id {key_id!r} is not the signer's {signer.key_id!r}")
    if signature["algorithm"] != resolved.algorithm:
        raise ManifestVerificationError(
            f"signature algorithm {signature['algorithm']!r} is not the signer's {resolved.algorithm!r}"
        )
    value = signature["value"]
    payload = _canonical_json(_unsigned(manifest)) if version == _V1 else _signing_payload(manifest)
    if not isinstance(value, str) or not resolved.verify(payload, value):
        raise ManifestVerificationError("signature does not match the manifest")


def _verify_manifest_fields(
    manifest: Mapping[str, Any],
    signer: ManifestSigner,
    signers: Mapping[str, ManifestSigner],
    *,
    allow_v1: bool = True,
) -> None:
    _verify_signed(manifest, signer, signers, allow_v1=allow_v1)
    if manifest.get("hash_algorithm") != _HASH_ALGORITHM:
        raise ManifestVerificationError(f"unsupported hash_algorithm {manifest.get('hash_algorithm')!r}")
    hashes = manifest.get("record_hashes")
    count = manifest.get("record_count")
    if not isinstance(hashes, list) or type(count) is not int or count != len(hashes):
        raise ManifestVerificationError(f"record_count {count!r} is not the number of hashes")
    if not isinstance(manifest.get("run_id"), str):
        raise ManifestVerificationError(f"run_id {manifest.get('run_id')!r} is not a string")
    if manifest["manifest_version"] != _V1 and type(manifest.get("sealed_late")) is not bool:
        raise ManifestVerificationError(f"sealed_late {manifest.get('sealed_late')!r} is not a bool")


def _verify_rotation_fields(
    entry: Mapping[str, Any],
    signer: ManifestSigner,
    signers: Mapping[str, ManifestSigner],
    *,
    allow_v1: bool = True,
    current: str | None = None,
) -> None:
    keys = _ROTATION_KEYS_V1 if entry.get("manifest_version") == _V1 else _ROTATION_KEYS
    if entry.get("kind") != _ROTATION_KIND or set(entry) != keys:
        raise ManifestVerificationError(
            f"a manifest log line with a kind must be a {_ROTATION_KIND!r} entry with exactly {sorted(keys)}"
        )
    outgoing = signers[current] if current is not None and entry["manifest_version"] == _MANIFEST_VERSION else None
    if outgoing is not None:
        # Header before the new key's signature, which covers it: a relabel reports as one.
        _check_co_signature_header(entry, outgoing)
    _verify_signed(entry, signer, signers, allow_v1=allow_v1)
    if outgoing is not None:
        value = entry["previous_key_signature"]["value"]
        if not isinstance(value, str) or not outgoing.verify(_signing_payload(entry), value):
            raise ManifestVerificationError("previous_key_signature does not match the manifest")


def _check_co_signature_header(entry: Mapping[str, Any], current: ManifestSigner) -> None:
    """The outgoing current key must co-sign a v2 rotation entry: check the block shape, key_id and algorithm."""
    block = entry["previous_key_signature"]
    if not isinstance(block, Mapping) or set(block) != _SIGNATURE_KEYS:
        raise ManifestVerificationError(f"previous_key_signature must have exactly {sorted(_SIGNATURE_KEYS)}")
    if block["key_id"] != current.key_id:
        raise ManifestVerificationError(
            f"previous_key_signature key_id {block['key_id']!r} is not the current key {current.key_id!r}"
        )
    if block["algorithm"] != current.algorithm:
        raise ManifestVerificationError(
            f"previous_key_signature algorithm {block['algorithm']!r} is not the key's {current.algorithm!r}"
        )


def _verify_genesis_fields(
    entry: Mapping[str, Any], signer: ManifestSigner, signers: Mapping[str, ManifestSigner]
) -> None:
    if set(entry) != _GENESIS_KEYS:
        raise ManifestVerificationError(f"a genesis entry must have exactly {sorted(_GENESIS_KEYS)}")
    _verify_signed(entry, signer, signers)
    if entry["manifest_version"] != _MANIFEST_VERSION:
        raise ManifestVerificationError("a genesis entry must be manifest_version 2")
    if not isinstance(entry["log_id"], str) or _is_blank(entry["log_id"]):
        raise ManifestVerificationError(f"genesis log_id {entry['log_id']!r} is not a non-blank string")
    previous = entry["previous_manifest_hash"]
    if previous is not None and not (isinstance(previous, str) and _SHA256_HEX.fullmatch(previous)):
        raise ManifestVerificationError("a genesis previous_manifest_hash must be null or a sha256 hex digest")


def _rotation_transition(key_id: str, active: str | None, retired: frozenset[str]) -> tuple[str, frozenset[str]]:
    """The (active, retired) keys after a rotation to `key_id`; raises if the log may not rotate to it."""
    if active is None:
        raise ManifestVerificationError("a key rotation entry has no earlier manifest to rotate from")
    if key_id == active or key_id in retired:
        raise ManifestVerificationError(f"a key rotation entry is signed by key {key_id!r}, which the log already used")
    return key_id, retired | {active}


def _record_mismatch(manifest: Mapping[str, Any], digest: _RunDigest) -> str:
    """The message for a record_hashes mismatch: a distinct wording when the run merely has extra records."""
    run_id = manifest["run_id"]
    sealed_hashes = manifest["record_hashes"]
    generic = f"records of run_id {run_id!r} do not match the record_hashes"
    try:
        sealed_counts: Counter[Any] = Counter(sealed_hashes)
        actual_counts: Counter[Any] = Counter(digest.hashes)
    except TypeError:
        return generic
    extra = len(digest.hashes) - len(sealed_hashes)
    if extra > 0 and all(actual_counts[key] >= count for key, count in sealed_counts.items()):
        return f"run_id {run_id!r} has {extra} record(s) beyond its seal"
    return generic


def _verify_digest(manifest: Mapping[str, Any], digest: _RunDigest) -> None:
    # Lists, not sets: a repeated record must not pass.
    if sorted(digest.hashes) != manifest["record_hashes"]:
        raise ManifestVerificationError(_record_mismatch(manifest, digest))
    if digest.compliant is not manifest.get("compliant"):
        raise ManifestVerificationError(
            f"compliant {manifest.get('compliant')!r} of run_id {manifest['run_id']!r} is not what its records give"
        )


def verify_manifest(
    manifest: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    *,
    signer: ManifestSigner,
    previous_signers: Iterable[ManifestSigner] = (),
) -> None:
    """Raise ManifestVerificationError unless signed by `signer` and matching `records`. `previous_signers` covers
    a retired signing key during rotation (see module docstring)."""
    signers = _signer_map(signer, previous_signers)
    _verify_manifest_fields(manifest, signer, signers)
    try:
        digest = _digest_of(records)
    except ValueError as exc:
        raise ManifestVerificationError(f"a record cannot be digested: {exc}") from exc
    _verify_digest(manifest, digest)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # json.loads keeps the last duplicate, a first-wins reader the first: reject.
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError(f"JSON object repeats a key among {sorted(obj)}")
    return obj


def _reject_constant(token: str) -> Any:
    raise ValueError(f"JSON constant {token} is not allowed")


def _decode_line(where: str, line: bytes) -> Any:
    try:
        return json.loads(
            line.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
        )
    except (ValueError, RecursionError) as exc:
        raise ManifestVerificationError(f"{where} is not valid JSON: {exc}") from exc


class _OversizedLine(bytes):
    """A line longer than the cap: no content, only its full size, sha256 and whether a newline ended it."""

    size: int
    sha256: str
    terminated: bool

    def __bool__(self) -> bool:
        return True


def _line_size(raw: bytes) -> int:
    return raw.size if isinstance(raw, _OversizedLine) else len(raw)


def _is_terminated(raw: bytes) -> bool:
    return raw.terminated if isinstance(raw, _OversizedLine) else raw.endswith(b"\n")


def _read_lines(file: IO[bytes]) -> Iterator[bytes]:
    """Yield the raw lines of `file`, never holding more than MAX_LINE_BYTES + 1 bytes of one line."""
    chunk = MAX_LINE_BYTES + 1
    while raw := file.readline(chunk):
        if len(raw) < chunk or raw.endswith(b"\n"):
            yield raw
            continue
        digest, size, last = hashlib.sha256(raw), len(raw), raw
        while last and not last.endswith(b"\n"):
            last = file.readline(chunk)
            digest.update(last)
            size += len(last)
        line = _OversizedLine()
        line.size, line.sha256, line.terminated = size, digest.hexdigest(), last.endswith(b"\n")
        yield line


def _parse_ndjson_line(path: str | Path, number: int | str, raw: bytes) -> tuple[bytes, dict[str, Any]]:
    where = f"{path} line {number}" if isinstance(number, int) else f"{path} {number}"
    if isinstance(raw, _OversizedLine):
        raise ManifestVerificationError(f"{where} is longer than {MAX_LINE_BYTES} bytes")
    if not raw.endswith(b"\n"):
        raise ManifestVerificationError(f"{where} does not end with a newline")
    line = raw[:-1]
    obj = _decode_line(where, line)
    if not isinstance(obj, dict):
        raise ManifestVerificationError(f"{where} is not a JSON object")
    return line, obj


def _read_ndjson(path: str | Path, start: int = 0, number: int = 1) -> Iterator[tuple[bytes, dict[str, Any]]]:
    with open(path, "rb") as file:
        file.seek(start)
        for number, raw in enumerate(_read_lines(file), start=number):
            yield _parse_ndjson_line(path, number, raw)


def _iter_manifests(path: str | Path) -> Iterator[dict[str, Any]]:
    try:
        for _, manifest in _read_ndjson(path):
            yield manifest
    except FileNotFoundError:
        return


def _scan_for_run(manifest_path: str | Path, run_id: str, start: int = 0) -> bool:
    """Byte-scan the manifest lines from `start` for a decodable JSON object with `run_id`."""
    needle = json.dumps(run_id).encode("utf-8")
    try:
        with open(manifest_path, "rb") as file:
            file.seek(start)
            for raw in _read_lines(file):
                if isinstance(raw, _OversizedLine) or needle not in raw:
                    continue
                try:
                    manifest = _decode_line(str(manifest_path), raw.rstrip(b"\n"))
                except ManifestVerificationError:
                    continue
                if isinstance(manifest, dict) and manifest.get("run_id") == run_id:
                    return True
    except FileNotFoundError:
        return False
    return False


def _read_manifests(path: str | Path) -> list[dict[str, Any]]:
    return list(_iter_manifests(path))


class _AuditScan:
    """How far the audit file was scanned: the end offset and line count, the last line's start and sha256, and the
    first record (offset, line number) of each run seen."""

    def __init__(self, dev: int, ino: int) -> None:
        self.dev, self.ino = dev, ino
        self.end = self.count = self.last_start = 0
        self.last_sha256 = ""
        self.first: dict[str, tuple[int, int]] = {}

    @classmethod
    def fresh(cls, path: str | Path) -> _AuditScan:
        stat_result = os.stat(path)
        return cls(stat_result.st_dev, stat_result.st_ino)

    def note(self, line: bytes, run_id: str | None, sealed: Container[str] = ()) -> None:
        self.count += 1
        if run_id is not None and not _is_blank(run_id) and run_id not in sealed:
            self.first.setdefault(run_id, (self.end, self.count))
        self.last_start = self.end
        self.last_sha256 = _sha256(line + b"\n")
        self.end += len(line) + 1


def _record_run_id(record: Mapping[str, Any]) -> str | None:
    run_id = record.get("run_id")
    if run_id is not None and not isinstance(run_id, str):
        raise ManifestVerificationError(f"audit record run_id {run_id!r} is neither a string nor null")
    return run_id


def _digest_runs(
    audit_path: str | Path,
    keep: Callable[[str], bool],
    uncovered: _Uncovered | None = None,
    *,
    track_times: bool = False,
    start: int = 0,
    number: int = 1,
    scan: _AuditScan | None = None,
) -> dict[str, _RunDigest]:
    if uncovered is None:
        uncovered = _Uncovered()
    digests: dict[str, _RunDigest] = {}
    for line, record in _read_ndjson(audit_path, *((start, number) if start else ())):
        run_id = _record_run_id(record)
        if scan is not None:
            scan.note(line, run_id)
        if run_id is None or _is_blank(run_id):
            uncovered.unattributed += 1
            continue
        if not keep(run_id):
            uncovered.by_run[run_id] += 1
            continue
        # The raw line, not a re-serialisation: the seal covers the written bytes.
        digest = digests.setdefault(run_id, _RunDigest())
        digest.add(line, record)
        if track_times:
            digest.note_time(record)
    return digests


def _reject_aliased_paths(**named_paths: str | Path) -> None:
    """Raise ValueError, never ManifestVerificationError, when two of the given paths resolve to the same file."""
    seen: dict[str, str] = {}
    for name, path in named_paths.items():
        real = os.path.realpath(path)
        if real in seen:
            raise ValueError(f"{seen[real]} and {name} must not be the same file, but both resolve to {real}")
        seen[real] = name


@contextmanager
def _flock(path: str | Path, *, exclusive: bool) -> Iterator[None]:
    """flock `path`; a shared lock is best-effort, an exclusive one raises on OSError. No lock without fcntl."""
    fd: int | None = None
    try:
        try:
            # NFS needs a writable descriptor for an exclusive lock.
            fd = _open_locked(path, (os.O_RDWR | os.O_CREAT) if exclusive else os.O_RDONLY, exclusive=exclusive)
        except FileNotFoundError:
            if exclusive:
                raise
        yield
    finally:
        if fd is not None:
            os.close(fd)


@dataclass
class _LogState:
    sealed: set[str]
    head: str | None
    active: str | None
    retired: frozenset[str]
    hashes: set[str]
    seen_v2: bool = False
    log_id: str | None = None
    lines: int = 0


def _unverified_anchor(anchors: Iterable[str], hashes: set[str]) -> str | None:
    """The first anchor that is not the hash of a verified line, or None."""
    return next((anchor for anchor in anchors if anchor not in hashes), None)


def _verify_log(
    log: Iterable[Mapping[str, Any]],
    *,
    signer: ManifestSigner,
    signers: Mapping[str, ManifestSigner],
    expected_head: str | None,
    digests: Mapping[str, _RunDigest] | None = None,
    require_current: bool = False,
    log_id: str | None = None,
    anchored_heads: Iterable[str] = (),
    seed: _LogState | None = None,
) -> _LogState:
    """Record and head mismatches are collected into one error (head first); a structural failure raises at once.
    The first line's key is current until a rotation entry replaces it. `log_id` demands a matching genesis first.
    Every anchored head must be the hash of a verified line. A `seed` resumes after its lines: `log` holds the rest."""
    seed = seed or _LogState(set(), None, None, frozenset(), set())
    sealed: set[str] = set(seed.sealed)
    hashes: set[str] = set(seed.hashes)
    head, active, retired, seen_v2, genesis_id = seed.head, seed.active, seed.retired, seed.seen_v2, seed.log_id
    problems: list[str] = []
    lines = seed.lines
    for number, manifest in enumerate(log, start=seed.lines):
        is_seal = "kind" not in manifest
        if manifest.get("kind") == _GENESIS_KIND:
            if number:
                raise ManifestVerificationError(f"a genesis entry is only allowed as line 1, not line {number + 1}")
            _verify_genesis_fields(manifest, signer, signers)
            if log_id is not None and manifest["log_id"] != log_id:
                raise ManifestVerificationError(f"manifest log is for log_id {manifest['log_id']!r}, not {log_id!r}")
            if seed.head is None and manifest["previous_manifest_hash"] is not None:
                # A successor segment starts the chain at its predecessor's head.
                head = manifest["previous_manifest_hash"]
                hashes.add(head)
            genesis_key = manifest["signature"]["key_id"]
            if active is not None and genesis_key != active:
                raise ManifestVerificationError(
                    f"the genesis entry is signed by key {genesis_key!r}, not the log's current key {active!r}"
                )
            active = genesis_key
            genesis_id = manifest["log_id"]
            where = "the genesis entry"
        elif log_id is not None and number == 0:
            raise ManifestVerificationError(f"manifest log line 1 is not a genesis entry for log_id {log_id!r}")
        elif not is_seal:
            _verify_rotation_fields(manifest, signer, signers, allow_v1=not seen_v2, current=active)
            active, retired = _rotation_transition(manifest["signature"]["key_id"], active, retired)
            where = "a key rotation entry"
        else:
            _verify_manifest_fields(manifest, signer, signers, allow_v1=not seen_v2)
            run_id: str = manifest["run_id"]
            key_id = manifest["signature"]["key_id"]
            if active is None:
                active = key_id
            elif key_id != active:
                raise ManifestVerificationError(
                    f"run_id {run_id!r} is signed by key {key_id!r}, not the log's current key {active!r}"
                )
            where = f"run_id {run_id!r}"
        if manifest.get("previous_manifest_hash") != head:
            raise ManifestVerificationError(f"manifest chain is broken at {where}")
        if is_seal:
            if run_id in sealed:
                raise ManifestVerificationError(f"run_id {run_id!r} is sealed by more than one manifest")
            if digests is not None:
                try:
                    _verify_digest(manifest, digests.get(run_id, _RunDigest()))
                except ManifestVerificationError as exc:
                    problems.append(str(exc))
            sealed.add(run_id)
        head = manifest_hash(manifest)
        hashes.add(head)
        seen_v2 = seen_v2 or manifest["manifest_version"] == _MANIFEST_VERSION
        lines = number + 1
    if require_current and active is not None and active != signer.key_id:
        raise ManifestVerificationError(f"manifest log is under key {active!r}, not the signer's {signer.key_id!r}")
    if expected_head is not None and head != expected_head:
        problems.insert(0, f"manifest log head {head!r} is not the expected head {expected_head!r}")
    missing = _unverified_anchor(anchored_heads, hashes)
    if missing is not None:
        problems.append(f"anchored head {missing!r} is not a line of the manifest log (head {head!r})")
    if problems:
        message = "; ".join(problems[:_MAX_PROBLEMS])
        if len(problems) > _MAX_PROBLEMS:
            message += f"; and {len(problems) - _MAX_PROBLEMS} more"
        raise ManifestVerificationError(message)
    return _LogState(sealed, head, active, retired, hashes, seen_v2, genesis_id, lines)


class HeadAnchor(Protocol):
    """Where manifest log heads are anchored: `write` records one, `latest` reads the newest (None if unknown)."""

    def write(self, head: str) -> None: ...

    def latest(self) -> str | None: ...


_TAIL_CHUNK = 8 * 1024


def _read_lines_reversed(file: IO[bytes]) -> Iterator[bytes | None]:
    """Yield the lines of `file` last to first, reading bounded chunks and keeping at most MAX_LINE_BYTES bytes of
    a line. The unterminated tail (if any) comes first without a newline; a line over the cap is yielded as None."""

    def emit(segs: list[bytes], size: int, terminated: bool) -> bytes | None:
        return None if size > MAX_LINE_BYTES else b"".join(reversed(segs)) + (b"\n" if terminated else b"")

    pos = file.seek(0, os.SEEK_END)
    segs: list[bytes] = []
    size, tail = 0, True
    while pos > 0:
        step = min(_TAIL_CHUNK, pos)
        pos -= step
        file.seek(pos)
        parts = file.read(step).split(b"\n")
        for i in range(len(parts) - 1, -1, -1):
            size += len(parts[i])
            if size <= MAX_LINE_BYTES:
                segs.append(parts[i])
            else:
                segs.clear()
            if i:
                if size or not tail:
                    yield emit(segs, size, not tail)
                segs, size, tail = [], 0, False
    if size or not tail:
        yield emit(segs, size, not tail)


class NdjsonHeadAnchor:
    """A HeadAnchor appending {"head", "anchored_at"} lines to a file; keep it on storage the log writer cannot
    rewrite."""

    def __init__(self, path: str | Path) -> None:
        self._path = path

    def write(self, head: str) -> None:
        _append_and_fsync(self._path, [{"head": head, "anchored_at": _utc_now()}])

    def latest(self) -> str | None:
        """The last terminated line's head; None for a missing file or one without a terminated line. An unterminated, torn
        or oversized last line (crash residue) is ignored; a terminated bad line raises ManifestVerificationError."""
        try:
            with open(self._path, "rb") as file:
                lines = list(islice(_read_lines_reversed(file), 2))
        except FileNotFoundError:
            return None
        if lines and (lines[0] is None or not lines[0].endswith(b"\n")):
            lines.pop(0)
        if not lines:
            return None
        raw = _OversizedLine() if lines[0] is None else lines[0]
        head = _parse_ndjson_line(self._path, "last terminated line", raw)[1].get("head")
        if not isinstance(head, str):
            raise ManifestVerificationError(f"{self._path} last terminated line has no string head")
        return head


def _check_line_cap(records: Sequence[Mapping[str, Any]]) -> None:
    """Raise ValueError before anything is written if a record's line is longer than MAX_LINE_BYTES."""
    for record in records:
        size = len(_canonical_json(record))
        if size > MAX_LINE_BYTES:
            raise ValueError(f"a line of {size} bytes exceeds the {MAX_LINE_BYTES} byte line cap")


def _append_and_fsync(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Append, then fsync the file and its parent directory."""
    _check_line_cap(records)
    _append_records(path, records)
    _fsync(path)
    _fsync(Path(path).parent)


def _rotation_entry(
    signer: ManifestSigner, outgoing: ManifestSigner, previous_manifest_hash: str | None
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "manifest_version": _MANIFEST_VERSION,
        "kind": _ROTATION_KIND,
        "rotated_at": _utc_now(),
        "previous_manifest_hash": previous_manifest_hash,
        "previous_key_signature": {"algorithm": outgoing.algorithm, "key_id": outgoing.key_id},
    }
    entry["signature"] = _signature_block(entry, signer)
    entry["previous_key_signature"]["value"] = outgoing.sign(_signing_payload(entry))
    return entry


def _genesis_entry(signer: ManifestSigner, log_id: str, previous: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "manifest_version": _MANIFEST_VERSION,
        "kind": _GENESIS_KIND,
        "log_id": log_id,
        "created_at": _utc_now(),
        "previous_manifest_hash": previous,
    }
    entry["signature"] = _signature_block(entry, signer)
    return entry


def _check_log_id(owner: str, log_id: object) -> None:
    if log_id is not None and (not isinstance(log_id, str) or _is_blank(log_id)):
        raise ValueError(f"{owner} log_id must be a non-blank string")


@dataclass(frozen=True)
class LogCoverage:
    """What a verification covered; unsealed_lines maps each unsealed run_id to its line count. It is a private
    copy: do not mutate it if you hash the value."""

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


def _fsync(path: str | Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _truncate_durably(path: str | Path, length: int) -> None:
    os.truncate(path, length)
    _fsync(path)
    _fsync(Path(path).parent)


def _require_terminated(path: str | Path) -> None:
    try:
        with open(path, "rb") as file:
            if file.seek(0, os.SEEK_END):
                file.seek(-1, os.SEEK_END)
                if file.read(1) != b"\n":
                    raise ManifestVerificationError(f"{path} does not end with a newline; repair it before a recovery")
    except FileNotFoundError:
        return


def _unlink_durably(path: str | Path) -> None:
    os.unlink(path)
    _fsync(Path(path).parent)


def _append_with_rollback(path: str | Path, entries: Sequence[Mapping[str, Any]], *, existed: bool) -> None:
    """Append and fsync under the caller's lock on `path`; a failure restores the file, or removes it if the lock
    created it. Capture `existed` before acquiring the lock: its open(O_CREAT) would otherwise always find the
    file already there. Sealing and rotation pass `existed=True`, so a failed append leaves the log file (truncated)
    instead of removing it."""
    size = os.path.getsize(path) if os.path.exists(path) else 0
    try:
        _append_and_fsync(path, entries)
    except BaseException:
        with suppress(OSError):
            if not existed and size == 0:
                _unlink_durably(path)
            else:
                _truncate_durably(path, size)
        raise
