"""Run manifests: a signed, hash-chained seal over the audit records of one run.

Sealing is a post-run step. Workers hold their own extender copy and records arrive unordered from several
writers, so the chain cannot run through the records. It links per-run manifests instead, written by a sealer
in the parent process after run_all returns. Sealers of one manifest log serialise on an exclusive flock (POSIX;
the wait has no timeout). Where fcntl is missing they race and verify_ndjson_log reports a forked chain.
Verifying takes a shared lock on the log.

A seal is final, so seal a run_id only once it is finished. Records that reach a sealed run later fail
verification by design: a second session.run() on one prepared session reuses the run_id, and a run still in
flight in another thread or process that shares the audit file keeps writing. For a prepared session pass
run_id=session.run_id. The sealer verifies the manifest log but not the records of sealed runs;
verify_ndjson_log does, over the bytes of each audit line where verify_manifest compares records.

Limits:
- HMAC is symmetric, so HmacSha256Signer proves integrity to key holders and not non-repudiation; an asymmetric
  or KMS signer plugs in through ManifestSigner.
- Records without a usable run_id (null, absent, blank) and runs that are not sealed yet are outside every seal
  and are not reported.
- Removing the newest manifests or the whole log cannot be detected from the files alone, and a truncated log
  lets the next sealing run re-seal altered records. Anchor the head outside the log (verify_ndjson_log returns
  it, manifest_hash of the last sealed manifest gives it) and pass it back as expected_head.
- A failed append can leave a torn last line; the log then fails closed and must be repaired by hand.
- One key and one manifest log per audit file: the payload carries no log identity and a log cannot span a key
  rotation. Start a new audit file and manifest log for a new key.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from mloda.enterprise.extenders.audit.audit_extender import _append_records, _canonical_json, _is_missing, _utc_now

_MANIFEST_VERSION = 1
_HASH_ALGORITHM = "sha256"
_MIN_KEY_BYTES = 32
_SIGNATURE_KEYS = {"algorithm", "key_id", "value"}


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


def _digest_of(records: Iterable[Mapping[str, Any]]) -> _RunDigest:
    digest = _RunDigest()
    for record in records:
        digest.add(_canonical_json(record), record)
    return digest


def _unsigned(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "signature"}


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
    manifest["signature"] = {
        "algorithm": signer.algorithm,
        "key_id": signer.key_id,
        "value": signer.sign(_canonical_json(manifest)),
    }
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


def _read_ndjson(path: str | Path) -> Iterator[tuple[bytes, dict[str, Any]]]:
    """Yield each line without its newline, plus the object it parses to; every line must end in a newline."""
    with open(path, "rb") as file:
        for number, raw in enumerate(file, start=1):
            if not raw.endswith(b"\n"):
                raise ManifestVerificationError(f"{path} line {number} does not end with a newline")
            line = raw[:-1]
            try:
                obj = json.loads(line.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
            except (ValueError, RecursionError) as exc:
                raise ManifestVerificationError(f"{path} line {number} is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ManifestVerificationError(f"{path} line {number} is not a JSON object")
            yield line, obj


def _read_manifests(path: str | Path) -> list[dict[str, Any]]:
    try:
        return [manifest for _, manifest in _read_ndjson(path)]
    except FileNotFoundError:
        return []


def _digest_runs(audit_path: str | Path, keep: Callable[[str], bool] | None = None) -> dict[str, _RunDigest]:
    """Keyed in order of first appearance in the audit file, which is the sealing order. Streams the file, keeping
    only the runs `keep` accepts (all when None); records without a usable run_id are outside every seal."""
    digests: dict[str, _RunDigest] = {}
    for line, record in _read_ndjson(audit_path):
        run_id = record.get("run_id")
        if run_id is not None and not isinstance(run_id, str):
            raise ManifestVerificationError(f"audit record run_id {run_id!r} is neither a string nor null")
        if run_id is None or _is_missing(run_id) or (keep is not None and not keep(run_id)):
            continue
        # The raw line, not a re-serialisation: the seal covers the bytes that were written.
        digests.setdefault(run_id, _RunDigest()).add(line, record)
    return digests


@contextmanager
def _flock(path: str | Path, *, exclusive: bool) -> Iterator[None]:
    """Block on an flock of `path`. Exclusive creates the file. No lock where fcntl is missing, nor for a shared
    lock on a file that does not exist."""
    fd: int | None = None
    try:
        try:
            import fcntl

            fd = os.open(path, os.O_RDONLY | (os.O_CREAT if exclusive else 0), 0o600)
        except ImportError:
            pass
        except FileNotFoundError:
            if exclusive:
                raise
        else:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
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
    the records as well when `digests` is given; record and head mismatches are collected into one error, a
    structural failure raises at once. Returns the sealed run ids and the head."""
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
        problems.append(f"manifest log head {head!r} is not the expected head {expected_head!r}")
    if problems:
        raise ManifestVerificationError("; ".join(problems))
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
    the manifests to the log in one write under an exclusive lock. All are signed before any is written. The log
    is verified first; the records of sealed runs are not. Raises RunNotPendingError when `run_id` has no
    records or is already sealed; catch it, not ValueError, for an idempotent retry (ManifestVerificationError
    is a ValueError too)."""
    with _flock(manifest_path, exclusive=True):
        sealed, head = _verify_log(_read_manifests(manifest_path), signer=signer, expected_head=expected_head)
        if run_id is not None and run_id in sealed:
            raise RunNotPendingError(f"run_id {run_id!r} is already sealed in {manifest_path}")
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
        _append_records(manifest_path, manifests)
        return manifests


def verify_ndjson_log(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    expected_head: str | None = None,
) -> str | None:
    """Raise ManifestVerificationError unless every manifest verifies against the audit bytes of its run,
    chains to its predecessor and seals a run_id of its own; every failing run is named in the one error.
    Runs that are not sealed yet are ignored. Returns the head, the manifest_hash of the last manifest (None for
    an empty or missing log), to anchor outside the log."""
    # The log first, under a shared lock so no sealer is mid-append; a run sealed after this read then looks
    # unsealed, where the reverse order would make it look like tampering.
    with _flock(manifest_path, exclusive=False):
        manifests = _read_manifests(manifest_path)
    sealed = {manifest["run_id"] for manifest in manifests if isinstance(manifest.get("run_id"), str)}
    digests = _digest_runs(audit_path, sealed.__contains__)
    _, head = _verify_log(manifests, signer=signer, expected_head=expected_head, digests=digests)
    return head
