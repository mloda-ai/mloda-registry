"""Run manifests: a signed, hash-chained seal over the audit records of one run.

Sealing is a post-run step: seal a run only once it is finished. A seal is final, so records reaching a sealed run
later fail verification by design. verify_ndjson_log returns the head: anchor it outside the log and pass it back
as expected_head, because removing the newest manifests or the whole log is otherwise undetectable.

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
  them too). Any keyring key can append a complete rotation entry, which wedges the log for the real current key
  (availability only, not integrity). Quarantine repairs only a torn tail, so recovery is truncating the log back to
  an anchored head.
- Sealing and verifying need the log's current key to be `signer` (an archived log verifies with its current key).
  The check is load-bearing: it stops an unused keyring key from taking the log over.
- verify_manifest on a single manifest cannot order keys.
- A torn or undecodable line fails sealing and verification until quarantine_damaged_lines repairs it. It repairs
  only a torn manifest tail and the audit lines the readers reject; anything else stays a hard failure.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import stat
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

from mloda.enterprise.extenders.audit._records import _append_records, _canonical_json, _is_blank, _utc_now

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

_MANIFEST_VERSION = 1
_QUARANTINE_VERSION = 1
_HASH_ALGORITHM = "sha256"
_MIN_KEY_BYTES = 32
_ED25519_KEY_BYTES = 32
_ED25519_SIGNATURE = re.compile(r"[0-9a-f]{128}")
_SIGNATURE_KEYS = {"algorithm", "key_id", "value"}
_ROTATION_KIND = "key_rotation"
_ROTATION_KEYS = {"manifest_version", "kind", "rotated_at", "previous_manifest_hash", "signature"}
_MAX_PROBLEMS = 20


class ManifestSigner(Protocol):
    """Signs and verifies manifest payloads; verify returns False instead of raising."""

    algorithm: str
    key_id: str

    def sign(self, payload: bytes) -> str: ...

    def verify(self, payload: bytes, signature: str) -> bool: ...


def _check_key_id(owner: str, key_id: object) -> None:
    if not isinstance(key_id, str) or not key_id.strip():
        raise ValueError(f"{owner} key_id must be a non-blank string")


class HmacSha256Signer:
    """HMAC-SHA256 over a shared key of at least 32 bytes."""

    algorithm = "HMAC-SHA256"

    def __init__(self, key: bytes, key_id: str) -> None:
        if not isinstance(key, bytes) or len(key) < _MIN_KEY_BYTES:
            raise ValueError(f"HmacSha256Signer key must be bytes of at least {_MIN_KEY_BYTES} bytes")
        _check_key_id("HmacSha256Signer", key_id)
        self._key = key
        self.key_id = key_id

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key_id={self.key_id!r})"

    def sign(self, payload: bytes) -> str:
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature: str) -> bool:
        if not isinstance(signature, str):
            return False
        # compare_digest raises TypeError on a non-ASCII str.
        return signature.isascii() and hmac.compare_digest(self.sign(payload), signature)


def _ed25519() -> ModuleType:
    """The cryptography ed25519 module, imported per call because the extra is optional."""
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:
        raise ImportError(
            "Ed25519Signer needs the 'cryptography' package: pip install mloda-enterprise[ed25519]"
        ) from exc
    return ed25519


def _check_ed25519_key(name: str, key: object) -> None:
    if not isinstance(key, bytes) or len(key) != _ED25519_KEY_BYTES:
        raise ValueError(f"Ed25519Signer {name} must be bytes of exactly {_ED25519_KEY_BYTES} bytes")


class Ed25519Signer:
    """Ed25519 over raw 32-byte keys (`private_bytes_raw()`, `public_bytes_raw()`); needs the `ed25519` extra."""

    algorithm = "Ed25519"

    def __init__(self, private_key: bytes, key_id: str) -> None:
        _check_ed25519_key("private_key", private_key)
        _check_key_id("Ed25519Signer", key_id)
        private: Ed25519PrivateKey = _ed25519().Ed25519PrivateKey.from_private_bytes(private_key)
        self._private: Ed25519PrivateKey | None = private
        self._public: Ed25519PublicKey = private.public_key()
        self.key_id = key_id

    @classmethod
    def from_public_key(cls, public_key: bytes, key_id: str) -> Ed25519Signer:
        """A verify-only signer: `sign` raises ValueError."""
        _check_ed25519_key("public_key", public_key)
        _check_key_id("Ed25519Signer", key_id)
        signer = cls.__new__(cls)
        signer._private = None
        signer._public = _ed25519().Ed25519PublicKey.from_public_bytes(public_key)
        signer.key_id = key_id
        return signer

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key_id={self.key_id!r})"

    def sign(self, payload: bytes) -> str:
        if self._private is None:
            raise ValueError(f"{type(self).__name__} {self.key_id!r} holds only a public key and cannot sign")
        return self._private.sign(payload).hex()

    def verify(self, payload: bytes, signature: str) -> bool:
        if not isinstance(signature, str) or not _ED25519_SIGNATURE.fullmatch(signature):
            return False
        from cryptography.exceptions import InvalidSignature

        try:
            self._public.verify(bytes.fromhex(signature), payload)
        except InvalidSignature:
            return False
        return True


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


def _signer_map(signer: ManifestSigner, previous_signers: Iterable[ManifestSigner]) -> dict[str, ManifestSigner]:
    """Map key_id to signer, from `signer` plus `previous_signers`; raise ValueError for a shared key_id."""
    signers: dict[str, ManifestSigner] = {signer.key_id: signer}
    for previous in previous_signers:
        if previous.key_id in signers:
            raise ValueError(f"previous_signers repeats key_id {previous.key_id!r}")
        signers[previous.key_id] = previous
    return signers


def _verify_signed(manifest: Mapping[str, Any], signer: ManifestSigner, signers: Mapping[str, ManifestSigner]) -> None:
    """The signature and manifest_version checks shared by manifests and rotation entries."""
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
    if not isinstance(value, str) or not resolved.verify(_canonical_json(_unsigned(manifest)), value):
        raise ManifestVerificationError("signature does not match the manifest")
    # type() is int: True and 1.0 equal 1.
    version = manifest.get("manifest_version")
    if type(version) is not int or version != _MANIFEST_VERSION:
        raise ManifestVerificationError(f"unsupported manifest_version {version!r}")


def _verify_manifest_fields(
    manifest: Mapping[str, Any], signer: ManifestSigner, signers: Mapping[str, ManifestSigner]
) -> None:
    _verify_signed(manifest, signer, signers)
    if manifest.get("hash_algorithm") != _HASH_ALGORITHM:
        raise ManifestVerificationError(f"unsupported hash_algorithm {manifest.get('hash_algorithm')!r}")
    hashes = manifest.get("record_hashes")
    count = manifest.get("record_count")
    if not isinstance(hashes, list) or type(count) is not int or count != len(hashes):
        raise ManifestVerificationError(f"record_count {count!r} is not the number of hashes")
    if not isinstance(manifest.get("run_id"), str):
        raise ManifestVerificationError(f"run_id {manifest.get('run_id')!r} is not a string")


def _verify_rotation_fields(
    entry: Mapping[str, Any], signer: ManifestSigner, signers: Mapping[str, ManifestSigner]
) -> None:
    if entry.get("kind") != _ROTATION_KIND or set(entry) != _ROTATION_KEYS:
        raise ManifestVerificationError(
            f"a manifest log line with a kind must be a {_ROTATION_KIND!r} entry with exactly {sorted(_ROTATION_KEYS)}"
        )
    _verify_signed(entry, signer, signers)


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


@dataclass
class _LogState:
    sealed: set[str]
    head: str | None
    active: str | None
    retired: frozenset[str]


def _verify_log(
    log: Iterable[Mapping[str, Any]],
    *,
    signer: ManifestSigner,
    signers: Mapping[str, ManifestSigner],
    expected_head: str | None,
    digests: Mapping[str, _RunDigest] | None = None,
    require_current: bool = False,
) -> _LogState:
    """Record and head mismatches are collected into one error (head first); a structural failure raises at once.
    The first line's key is current until a rotation entry replaces it."""
    sealed: set[str] = set()
    head: str | None = None
    problems: list[str] = []
    active: str | None = None
    retired: frozenset[str] = frozenset()
    for manifest in log:
        is_rotation = "kind" in manifest
        if is_rotation:
            _verify_rotation_fields(manifest, signer, signers)
            active, retired = _rotation_transition(manifest["signature"]["key_id"], active, retired)
            where = "a key rotation entry"
        else:
            _verify_manifest_fields(manifest, signer, signers)
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
        if not is_rotation:
            if run_id in sealed:
                raise ManifestVerificationError(f"run_id {run_id!r} is sealed by more than one manifest")
            if digests is not None:
                try:
                    _verify_digest(manifest, digests.get(run_id, _RunDigest()))
                except ManifestVerificationError as exc:
                    problems.append(str(exc))
            sealed.add(run_id)
        head = manifest_hash(manifest)
    if require_current and active is not None and active != signer.key_id:
        raise ManifestVerificationError(f"manifest log is under key {active!r}, not the signer's {signer.key_id!r}")
    if expected_head is not None and head != expected_head:
        problems.insert(0, f"manifest log head {head!r} is not the expected head {expected_head!r}")
    if problems:
        message = "; ".join(problems[:_MAX_PROBLEMS])
        if len(problems) > _MAX_PROBLEMS:
            message += f"; and {len(problems) - _MAX_PROBLEMS} more"
        raise ManifestVerificationError(message)
    return _LogState(sealed, head, active, retired)


def _append_and_fsync(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Append, then fsync the file and its parent directory."""
    _append_records(path, records)
    _fsync(path)
    _fsync(Path(path).parent)


def _rotation_entry(signer: ManifestSigner, previous_manifest_hash: str | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "manifest_version": _MANIFEST_VERSION,
        "kind": _ROTATION_KIND,
        "rotated_at": _utc_now(),
        "previous_manifest_hash": previous_manifest_hash,
    }
    entry["signature"] = _signature_block(entry, signer)
    return entry


def rotate_manifest_key(
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    expected_head: str | None,
    previous_signers: Iterable[ManifestSigner] = (),
) -> dict[str, Any]:
    """Append a signed rotation entry making `signer` the log's current key; return it. Verifies the log first
    (`previous_signers` must hold its retired keys) and writes nothing on failure. Raises ValueError for a log with no
    manifests, KeyAlreadyCurrentError when `signer` is already current (a retry needs `expected_head=None` or the
    post-rotation head) and ManifestVerificationError when `signer` is retired. Pass the anchored head as
    `expected_head`: the entry chains onto it, so it commits to every earlier line."""
    signers = _signer_map(signer, previous_signers)
    nothing_to_rotate = f"{manifest_path} has no manifests to rotate; the first seal defines the key"
    # Checked before the lock: an exclusive _flock would create the file.
    if not os.path.exists(manifest_path):
        raise ValueError(nothing_to_rotate)
    with _flock(manifest_path, exclusive=True):
        state = _verify_log(_iter_manifests(manifest_path), signer=signer, signers=signers, expected_head=expected_head)
        if state.active is None:
            raise ValueError(nothing_to_rotate)
        if state.active == signer.key_id:
            raise KeyAlreadyCurrentError(f"manifest log is already under key {signer.key_id!r}")
        # Called for its raise only: it raises when `signer` is a retired key.
        _rotation_transition(signer.key_id, state.active, state.retired)
        entry = _rotation_entry(signer, state.head)
        _append_with_rollback(manifest_path, [entry], existed=True)
        return entry


def seal_ndjson_runs(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    previous_signers: Iterable[ManifestSigner] = (),
    run_id: str | None = None,
    expected_head: str | None = None,
) -> list[dict[str, Any]]:
    """Seal every unsealed run (or only `run_id`). Seal only after a run's writers stop, or sealing a still-live run
    fails its verification for good; prefer AuditExtender's automatic sealing from Extender.on_run_complete when
    available, since it fires only once a run's writers have stopped, for a run that is run() exactly once (a
    re-run session that reuses the same run_id, including a retry after a failed run, is not this case: see
    AuditExtender's class docstring). Pass `run_id` when other runs may still be live; omit it to sweep an audit
    file no writer is appending to, but never as a substitute for targeting a specific unsealed run_id when other
    runs are still live, since a blanket sweep could seal one of those. Raises RunAlreadySealedError for an
    already-sealed `run_id` (catch that, not ValueError, for an idempotent retry) and RunNotPendingError when it
    has no records. `signer` must be the log's current key: call rotate_manifest_key after a key change.
    `previous_signers` covers a retired signing key during rotation (see module docstring)."""
    signers = _signer_map(signer, previous_signers)
    _reject_aliased_paths(audit_path=audit_path, manifest_path=manifest_path)
    if run_id is not None and (not isinstance(run_id, str) or _is_blank(run_id)):
        raise ValueError("seal_ndjson_runs run_id must be a non-blank string")
    with _flock(manifest_path, exclusive=True):
        state = _verify_log(
            _iter_manifests(manifest_path),
            signer=signer,
            signers=signers,
            expected_head=expected_head,
            require_current=True,
        )
        sealed, head = state.sealed, state.head
        if run_id is not None and run_id in sealed:
            raise RunAlreadySealedError(f"run_id {run_id!r} is already sealed in {manifest_path}")
        # A manifest must not name records that never reached disk; a missing file is left for _digest_runs
        # to raise, as before.
        with suppress(FileNotFoundError):
            _fsync(audit_path)
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
        _append_with_rollback(manifest_path, manifests, existed=True)
        return manifests


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


def verify_ndjson_log_coverage(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    previous_signers: Iterable[ManifestSigner] = (),
    expected_head: str | None = None,
) -> LogCoverage:
    """Verify like verify_ndjson_log; return a LogCoverage (head, sealed_runs, sealed_lines, unattributed_lines,
    unsealed_lines). `signer` must be the log's current key: call rotate_manifest_key after a key change.
    `previous_signers` covers a retired signing key during rotation (see module docstring)."""
    signers = _signer_map(signer, previous_signers)
    _reject_aliased_paths(audit_path=audit_path, manifest_path=manifest_path)
    # Log first: a run sealed after this read looks unsealed, not tampered.
    with _flock(manifest_path, exclusive=False):
        manifests = _read_manifests(manifest_path)
    sealed = {manifest["run_id"] for manifest in manifests if isinstance(manifest.get("run_id"), str)}
    uncovered = _Uncovered()
    try:
        digests = _digest_runs(audit_path, sealed.__contains__, uncovered)
    except FileNotFoundError:
        digests = {}
    state = _verify_log(
        manifests, signer=signer, signers=signers, expected_head=expected_head, digests=digests, require_current=True
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
) -> str | None:
    """Raise ManifestVerificationError unless every manifest matches the audit bytes of its run. Returns the head:
    anchor it outside the log and pass it back as `expected_head`, or a truncated log goes undetected.
    `signer` must be the log's current key: call rotate_manifest_key after a key change.
    `previous_signers` covers a retired signing key during rotation (see module docstring)."""
    return verify_ndjson_log_coverage(
        audit_path, manifest_path, signer=signer, previous_signers=previous_signers, expected_head=expected_head
    ).head


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


def _damaged_manifest_tail(
    path: str | Path, *, signer: ManifestSigner, signers: Mapping[str, ManifestSigner], expected_head: str | None
) -> _Damage:
    """`expected_head` may be any complete manifest's head: a torn multi-seal write leaves complete ones past it."""
    try:
        with open(path, "rb") as file:
            lines = list(file)
    except FileNotFoundError:
        lines = []
    tail = lines.pop() if lines and not lines[-1].endswith(b"\n") else b""
    manifests = [_parse_ndjson_line(path, number, raw)[1] for number, raw in enumerate(lines, start=1)]
    head = _verify_log(manifests, signer=signer, signers=signers, expected_head=None).head
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
                os.unlink(path)
                _fsync(Path(path).parent)
            else:
                os.truncate(path, size)
                _fsync(path)
                _fsync(Path(path).parent)
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
    previous_signers: Iterable[ManifestSigner] = (),
    expected_head: str | None = None,
    dry_run: bool = False,
) -> list[QuarantinedLine]:
    """Repair a torn manifest tail and the audit lines the readers reject.

    - `dry_run=True` only reports; nothing is repaired or written.
    - A removed line is traced to the signed `quarantine_path` log first, fsynced before either file is touched.
      That write is not atomic with the repair itself: an interrupted run can leave a line traced but still
      present, and re-running it then appends a second trace entry for it.
    - A repair creates `manifest_path` when it is missing, since the exclusive lock needs a writable file;
      `dry_run` creates nothing.
    - Stop every audit-file writer first: the sink appends without a lock.
    - A valid-JSON unterminated last line is refused in both files (it may be a real seal or record).
    - An anchored `expected_head` must match one of the complete manifests, which bounds how far back the log can
      have been rewound; re-verify the repaired log against the freshest anchor you track yourself.
    - A torn rotation entry is a torn manifest tail; call rotate_manifest_key again after the repair.
    - Anything else raises and changes nothing.

    `previous_signers` covers a retired signing key during rotation (see module docstring)."""
    signers = _signer_map(signer, previous_signers)
    _reject_aliased_paths(audit_path=audit_path, manifest_path=manifest_path, quarantine_path=quarantine_path)
    if not (os.path.exists(audit_path) or os.path.exists(manifest_path)):
        return []
    with _flock(manifest_path, exclusive=not dry_run):
        manifest_damage = _damaged_manifest_tail(
            manifest_path, signer=signer, signers=signers, expected_head=expected_head
        )
        audit_damage = _damaged_audit_lines(audit_path)
        damage = manifest_damage + audit_damage
        if not damage:
            return []
        # Captured before the lock: an exclusive _flock opens O_CREAT, so afterwards the file always exists.
        existed = os.path.exists(quarantine_path)
        # Signed before the quarantine lock: a signing failure must not leave the file it creates; a dry run never signs.
        entries = [] if dry_run else [_trace_entry(item, raw, path, signer) for item, raw, path in damage]
        with _flock(quarantine_path, exclusive=not dry_run):
            _require_terminated(quarantine_path)
            if not dry_run:
                _append_with_rollback(quarantine_path, entries, existed=existed)
        if not dry_run:
            if audit_damage:
                _rewrite_without(audit_path, {item.line for item, _, _ in audit_damage})
            if manifest_damage:
                os.truncate(manifest_path, manifest_damage[0][0].offset)
                _fsync(manifest_path)
                _fsync(Path(manifest_path).parent)
        return [item for item, _, _ in damage]
