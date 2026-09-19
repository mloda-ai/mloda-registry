"""Run manifests: a signed, hash-chained seal over the audit records of one run.

Sealing is a post-run step. Workers hold their own extender copy and records arrive unordered from several
writers, so the chain cannot run through the records. It links per-run manifests instead, written by one sealer
in the parent process after run_all returns. Use one sealer per manifest log: two racing sealers fork the chain,
and verify_ndjson_log reports it.

A seal is final, so seal a run_id only once it is finished. Records that reach a sealed run later fail
verification by design: a second session.run() on one prepared session reuses the run_id, and a run still in
flight in another thread or process that shares the audit file keeps writing. For a prepared session pass
run_id=session.run_id. The sealer verifies the manifest log but not the records of sealed runs;
verify_ndjson_log does.

Limits:
- HMAC is symmetric, so HmacSha256Signer proves integrity to key holders and not non-repudiation; an asymmetric
  or KMS signer plugs in through ManifestSigner.
- Removing the newest manifests cannot be detected from the files alone, and a truncated log lets the next
  sealing run re-seal altered records. Anchor the head outside the log (verify_ndjson_log returns it,
  manifest_hash of the last sealed manifest gives it) and pass it back as expected_head.
- One key and one manifest log per audit file: the payload carries no log identity and a log cannot span a key
  rotation. Start a new audit file and manifest log for a new key.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from mloda.enterprise.extenders.audit.audit_extender import NdjsonAuditSink, _canonical_json, _utc_now

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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record_hashes(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Sorted, so the hashes do not depend on the order the writers happened to append in."""
    return sorted(_sha256(_canonical_json(record)) for record in records)


def _unsigned(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "signature"}


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
    manifest: dict[str, Any] = {
        "manifest_version": _MANIFEST_VERSION,
        "run_id": run_id,
        "sealed_at": _utc_now(),
        "hash_algorithm": _HASH_ALGORITHM,
        "record_count": len(records),
        "record_hashes": _record_hashes(records),
        "compliant": all(record.get("compliant") is True for record in records),
        "previous_manifest_hash": previous_manifest_hash,
    }
    manifest["signature"] = {
        "algorithm": signer.algorithm,
        "key_id": signer.key_id,
        "value": signer.sign(_canonical_json(manifest)),
    }
    return manifest


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
    if manifest.get("manifest_version") != _MANIFEST_VERSION:
        raise ManifestVerificationError(f"unsupported manifest_version {manifest.get('manifest_version')!r}")
    if manifest.get("hash_algorithm") != _HASH_ALGORITHM:
        raise ManifestVerificationError(f"unsupported hash_algorithm {manifest.get('hash_algorithm')!r}")
    hashes = manifest.get("record_hashes")
    if not isinstance(hashes, list) or manifest.get("record_count") != len(hashes):
        raise ManifestVerificationError(f"record_count {manifest.get('record_count')!r} is not the number of hashes")
    if not isinstance(manifest.get("run_id"), str):
        raise ManifestVerificationError(f"run_id {manifest.get('run_id')!r} is not a string")


def _verify_records(manifest: Mapping[str, Any], records: Iterable[Mapping[str, Any]]) -> None:
    # Sorted lists, not sets: a repeated record must not pass.
    if _record_hashes(records) != manifest["record_hashes"]:
        raise ManifestVerificationError(f"records of run_id {manifest['run_id']!r} do not match the record_hashes")


def verify_manifest(
    manifest: Mapping[str, Any], records: Iterable[Mapping[str, Any]], *, signer: ManifestSigner
) -> None:
    """Raise ManifestVerificationError unless the manifest is signed by `signer` and matches `records`."""
    _verify_manifest_fields(manifest, signer)
    _verify_records(manifest, records)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # json.loads keeps the last duplicate while a first-wins reader sees another object, so neither is trusted.
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ManifestVerificationError(f"JSON object repeats a key among {sorted(obj)}")
    return obj


def _read_ndjson(path: str | Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            obj = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            if not isinstance(obj, dict):
                raise ManifestVerificationError(f"{path} line {number} is not a JSON object")
            objects.append(obj)
    return objects


def _records_by_run(audit_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Keyed in order of first appearance in the audit file, which is the sealing order."""
    runs: dict[str, list[dict[str, Any]]] = {}
    for record in _read_ndjson(audit_path):
        run_id = record.get("run_id")
        if isinstance(run_id, str):
            runs.setdefault(run_id, []).append(record)
        elif run_id is not None:
            raise ManifestVerificationError(f"audit record run_id {run_id!r} is neither a string nor null")
    return runs


def _verify_log(
    log: Iterable[Mapping[str, Any]],
    *,
    signer: ManifestSigner,
    expected_head: str | None,
    runs: Mapping[str, list[dict[str, Any]]] | None = None,
) -> tuple[set[str], str | None]:
    """The one pass over a manifest log: every manifest, the chain, no repeated run_id and the head. Compares
    the records as well when `runs` is given. Returns the sealed run ids and the head."""
    sealed: set[str] = set()
    head: str | None = None
    for manifest in log:
        _verify_manifest_fields(manifest, signer)
        run_id: str = manifest["run_id"]
        if runs is not None:
            _verify_records(manifest, runs.get(run_id, []))
        if manifest.get("previous_manifest_hash") != head:
            raise ManifestVerificationError(f"manifest chain is broken at run_id {run_id!r}")
        if run_id in sealed:
            raise ManifestVerificationError(f"run_id {run_id!r} is sealed by more than one manifest")
        sealed.add(run_id)
        head = manifest_hash(manifest)
    if expected_head is not None and head != expected_head:
        raise ManifestVerificationError(f"manifest log head {head!r} is not the expected head {expected_head!r}")
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
    the manifests to the log. The log is verified first; the records of sealed runs are not."""
    runs = _records_by_run(audit_path)
    log = _read_ndjson(manifest_path) if Path(manifest_path).exists() else []
    sealed, head = _verify_log(log, signer=signer, expected_head=expected_head)
    if run_id is None:
        pending = [candidate for candidate in runs if candidate not in sealed]
    elif run_id not in runs:
        raise ValueError(f"no audit records for run_id {run_id!r} in {audit_path}")
    elif run_id in sealed:
        raise ValueError(f"run_id {run_id!r} is already sealed in {manifest_path}")
    else:
        pending = [run_id]

    sink = NdjsonAuditSink(manifest_path)
    manifests = []
    for pending_run_id in pending:
        manifest = seal_run(runs[pending_run_id], run_id=pending_run_id, signer=signer, previous_manifest_hash=head)
        sink.write(manifest)
        manifests.append(manifest)
        head = manifest_hash(manifest)
    return manifests


def verify_ndjson_log(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    expected_head: str | None = None,
) -> str | None:
    """Raise ManifestVerificationError unless every manifest verifies against the audit records of its run,
    chains to its predecessor and seals a run_id of its own. Runs that are not sealed yet are ignored.
    Returns the head, the manifest_hash of the last manifest (None for an empty log), to anchor outside the log."""
    runs = _records_by_run(audit_path)
    _, head = _verify_log(_read_ndjson(manifest_path), signer=signer, expected_head=expected_head, runs=runs)
    return head
