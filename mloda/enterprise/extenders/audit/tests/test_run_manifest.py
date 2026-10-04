"""Tests for run_manifest and the NDJSON sealing built on it."""

from __future__ import annotations

import ast
import base64
import builtins
import copy
import dataclasses
import errno
import hashlib
import hmac
import importlib
import json
import logging
import os
import pickle  # nosec
import re
import stat
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from mloda.steward import Extender, verified_context
from mloda.user import ParallelizationMode

import mloda.enterprise.extenders.audit as audit_package
import mloda.enterprise.extenders.audit.audit_extender as audit_extender_module
import mloda.enterprise.extenders.audit.run_manifest as run_manifest_module
from mloda.enterprise.extenders.audit import (
    AuditExtender,
    Ed25519Signer,
    HeadAnchor,
    HmacSha256Signer,
    IdentityRequiredError,
    KeyAlreadyCurrentError,
    LogCoverage,
    ManifestSigner,
    ManifestVerificationError,
    NdjsonAuditSink,
    NdjsonHeadAnchor,
    QuarantinedLine,
    RunAlreadySealedError,
    RunNotPendingError,
    SealedRunRefusedError,
    TeeAuditSink,
    _segments,
    _verify,
    manifest_hash,
    quarantine_damaged_lines,
    quarantine_from_rotation_entry,
    rotate_manifest_key,
    rotate_ndjson_segment,
    seal_ndjson_runs,
    seal_run,
    verify_manifest,
    verify_ndjson_log,
    verify_ndjson_log_coverage,
    verify_ndjson_segments,
    verify_quarantine_log,
)
from mloda.enterprise.extenders.audit import _quarantine as _quarantine_module
from mloda.enterprise.extenders.audit._records import _parse_event_time
from mloda.enterprise.extenders.audit.audit_extender import _append_records
from mloda.enterprise.extenders.audit.tests.manifest_helpers import _patch_bindings, _unpatch_bindings
from mloda.enterprise.extenders.audit.tests.test_audit_extender import (
    _POLICY_VERSION,
    BufferingNdjsonAuditSink,
    InMemoryAuditSink,
    _ReadSpy,
)
from mloda.testing.extenders.runners import expected_value_int, prepare_value_int, run_value_int
from mloda.testing.import_isolation import block_root, evict_package

_KEY = b"k" * 32
_OTHER_KEY = b"o" * 32
_THIRD_KEY = b"t" * 32

# RFC 8032 section 7.1, test 1: an empty message.
_RFC8032_SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
_RFC8032_PUBLIC_KEY = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
_RFC8032_SIGNATURE = (
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555"
    "fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
)

# The signing algorithm `_signer` builds; the `algorithm` fixture flips it for a test.
_ALGORITHM = "hmac"

_EXPECTED_MANIFEST_KEYS = {
    "manifest_version",
    "run_id",
    "sealed_at",
    "hash_algorithm",
    "record_count",
    "record_hashes",
    "compliant",
    "sealed_late",
    "previous_manifest_hash",
    "signature",
}

_EXPECTED_ROTATION_KEYS = {
    "manifest_version",
    "kind",
    "rotated_at",
    "previous_manifest_hash",
    "previous_key_signature",
    "signature",
}

_EXPECTED_GENESIS_KEYS = {"manifest_version", "kind", "log_id", "created_at", "previous_manifest_hash", "signature"}

_EXPECTED_QUARANTINE_KEYS = {
    "quarantine_version",
    "quarantined_at",
    "file",
    "path",
    "line",
    "offset",
    "length",
    "sha256",
    "reason",
    "raw_base64",
    "previous_entry_hash",
    "signature",
}

_DAMAGED_LINES: dict[str, bytes] = {
    "blank": b"\n",
    "bad-utf8": b"\xff\xfe\n",
    "truncated": b'{"run_id": "run-d"\n',
    "not-an-object": b"[]\n",
    "duplicate-key": b'{"run_id": "run-d", "run_id": "run-e"}\n',
    "deep-nesting": b"[" * 100000 + b"\n",
}

# The manifest log's writers and the run_id of each line they append (None for a rotation entry).
_WRITERS = ["seal", "rotate"]
_PENDING_RUN_IDS: dict[str, list[str | None]] = {"seal": ["run-d", "run-e", "run-f"], "rotate": [None]}

# As a `_rotation_entry` change, drops the key.
_MISSING: Any = object()

_current_key_required = pytest.mark.parametrize(
    "call", [verify_ndjson_log, verify_ndjson_log_coverage, seal_ndjson_runs], ids=["verify", "coverage", "seal"]
)

# Runs a class once per algorithm.
_both_algorithms = pytest.mark.usefixtures("algorithm")
# With `_both_algorithms`, narrows a class to Ed25519 (public-key signers exist only there).
_ed25519_only = pytest.mark.parametrize("algorithm", ["ed25519"], indirect=True)


@pytest.fixture(params=["hmac", "ed25519"])
def algorithm(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "_ALGORITHM", request.param)


def _signer_class() -> type[HmacSha256Signer] | type[Ed25519Signer]:
    classes: dict[str, type[HmacSha256Signer] | type[Ed25519Signer]] = {
        "hmac": HmacSha256Signer,
        "ed25519": Ed25519Signer,
    }
    return classes[_ALGORITHM]


def _signer(key: bytes = _KEY, key_id: str = "key-1") -> ManifestSigner:
    return _signer_class()(key, key_id)


def _reference_signature(key: bytes, payload: bytes) -> str:
    """The signature recomputed without the signer under test."""
    if _ALGORITHM == "ed25519":
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        signature: bytes = Ed25519PrivateKey.from_private_bytes(key).sign(payload)
        return signature.hex()
    assert _ALGORITHM == "hmac", _ALGORITHM
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _public_key(private_key: bytes) -> bytes:
    """The raw public key, derived without the signer under test."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    public_key = Ed25519PrivateKey.from_private_bytes(private_key).public_key()
    raw: bytes = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    return raw


def _verify_only(private_key: bytes = _KEY, key_id: str = "key-1") -> Ed25519Signer:
    """The public-key signer matching `_signer(private_key, key_id)`."""
    return Ed25519Signer.from_public_key(_public_key(private_key), key_id)


# Signatures that are not a str at all, one of which is the valid signature as bytes.
_NON_STR_SIGNATURES: dict[str, Callable[[str], Any]] = {
    "none": lambda signature: None,
    "bytes": lambda signature: signature.encode("ascii"),
    "int": lambda signature: 123,
    "list": lambda signature: ["a"],
}


def _record(run_id: str | None = "run-1", second: int = 0, *, tenant_id: str | None = "tenant-1") -> dict[str, Any]:
    compliant = tenant_id is not None
    return {
        "record_version": 1,
        "event_time": f"2026-01-01T00:00:{second:02d}.000000Z",
        "run_id": run_id,
        "tenant_id": tenant_id,
        "decision": "allow" if compliant else "deny",
        "compliant": compliant,
        "deny_reason": None if compliant else "missing_tenant_id",
        "feature_group_class": "my.module.MyFeatureGroup",
        "feature_names": ["value_int"],
        "status": "success",
    }


def _canonical(obj: Mapping[str, Any]) -> bytes:
    """The bytes NdjsonAuditSink writes, without the newline."""
    return json.dumps(obj, sort_keys=True).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unsigned(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "signature"}


def _write_records(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    sink = NdjsonAuditSink(path)
    for record in records:
        sink.write(record)


def _read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rewrite_lines(path: Path, objects: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(obj, sort_keys=True) + "\n" for obj in objects), encoding="utf-8")


def _append_line(path: Path, line: str | bytes) -> None:
    data = line.encode("utf-8") if isinstance(line, str) else line
    with open(path, "ab") as file:
        file.write(data + b"\n")


def _signing_payload_ref(entry: Mapping[str, Any]) -> bytes:
    """The v2 signed bytes, recomputed without the module: every signature block reduced to algorithm and key_id."""
    reduced = dict(entry)
    for field in ("signature", "previous_key_signature"):
        if field in reduced:
            block = reduced[field]
            reduced[field] = {"algorithm": block["algorithm"], "key_id": block["key_id"]}
    return _canonical(reduced)


def _resigned(manifest: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    changed = {**manifest, **changes}
    value = _signer().sign(_signing_payload_ref(changed))
    changed["signature"] = {**manifest["signature"], "value": value}
    return changed


def _as_v1(manifest: Mapping[str, Any], signer: ManifestSigner) -> dict[str, Any]:
    """A v1 seal: no sealed_late, signed over the entry minus its signature."""
    entry = {key: value for key, value in _unsigned(manifest).items() if key != "sealed_late"}
    entry["manifest_version"] = 1
    entry["signature"] = {
        "algorithm": signer.algorithm,
        "key_id": signer.key_id,
        "value": signer.sign(_canonical(entry)),
    }
    return entry


def _rotation_entry(
    signer: ManifestSigner,
    previous_manifest_hash: str | None,
    *,
    v1: bool = False,
    outgoing: ManifestSigner | None = None,
    **changes: Any,
) -> dict[str, Any]:
    """A rotation entry signed by any `signer`, for forgeries; `changes` apply before signing; `v1` builds a v1 one.

    A v2 entry is co-signed by `outgoing` (default: the key-1 signer); `previous_key_signature=_MISSING` drops the
    co-signature and a changed `previous_key_signature["value"]` forges it.
    """
    entry: dict[str, Any] = {
        "manifest_version": 1 if v1 else 2,
        "kind": "key_rotation",
        "rotated_at": "2026-01-01T00:00:00.000000Z",
        "previous_manifest_hash": previous_manifest_hash,
        **changes,
    }
    co_signer = outgoing or _signer()
    if not v1 and "previous_key_signature" not in changes:
        entry["previous_key_signature"] = {"algorithm": co_signer.algorithm, "key_id": co_signer.key_id}
    entry = {key: value for key, value in entry.items() if value is not _MISSING}
    block = {"algorithm": signer.algorithm, "key_id": signer.key_id}
    payload = _canonical(entry) if v1 else _signing_payload_ref({**entry, "signature": block})
    entry["signature"] = {**block, "value": signer.sign(payload)}
    co_block = entry.get("previous_key_signature")
    if isinstance(co_block, dict) and "value" not in co_block:
        co_block["value"] = co_signer.sign(payload)
    return entry


def _build_log(directory: Path, *steps: tuple[str, ManifestSigner]) -> tuple[Path, Path]:
    """Hand-build a manifest log: ("seal", signer) seals a new run-N, ("rotate", signer) appends a rotation entry.

    A rotation entry is co-signed by the signer of the previous step. The "seal-v1" and "rotate-v1" actions append the
    same lines in the v1 format.
    """
    audit_path = directory / "audit.ndjson"
    manifest_path = directory / "manifests.ndjson"
    head: str | None = None
    current: ManifestSigner | None = None
    for number, (action, signer) in enumerate(steps, start=1):
        if action in ("seal", "seal-v1"):
            record = _record(f"run-{number}", number)
            _write_records(audit_path, [record])
            line = seal_run([record], run_id=f"run-{number}", signer=signer, previous_manifest_hash=head)
            if action == "seal-v1":
                line = _as_v1(line, signer)
        else:
            line = _rotation_entry(signer, head, v1=action == "rotate-v1", outgoing=current)
        _write_records(manifest_path, [line])
        head = manifest_hash(line)
        current = signer
    return audit_path, manifest_path


def _genesis_entry(signer: ManifestSigner, log_id: Any = "log-a", **changes: Any) -> dict[str, Any]:
    """A v2 genesis line signed by `signer`; `changes` apply before signing (`_MISSING` drops a key)."""
    entry: dict[str, Any] = {
        "manifest_version": 2,
        "kind": "genesis",
        "log_id": log_id,
        "created_at": "2026-01-01T00:00:00.000000Z",
        "previous_manifest_hash": None,
        **changes,
    }
    entry = {key: value for key, value in entry.items() if value is not _MISSING}
    block = {"algorithm": signer.algorithm, "key_id": signer.key_id}
    entry["signature"] = {**block, "value": signer.sign(_signing_payload_ref({**entry, "signature": block}))}
    return entry


def _genesis_log(directory: Path, log_id: str = "log-a") -> tuple[Path, Path]:
    """Runs a, b, c sealed under `log_id`, so the log starts with a genesis line."""
    audit_path = directory / "audit.ndjson"
    manifest_path = directory / "manifests.ndjson"
    _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2), _record("run-c", 3)])
    manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id=log_id)
    assert [manifest["run_id"] for manifest in manifests] == ["run-a", "run-b", "run-c"]
    return audit_path, manifest_path


_PREDECESSOR = _sha256(b"previous-segment-final-head")


def _successor_log(directory: Path, predecessor: Any = _PREDECESSOR) -> tuple[Path, Path]:
    """A successor segment: a genesis naming `predecessor`, then runs a and b sealed onto it."""
    audit_path = directory / "audit.ndjson"
    manifest_path = directory / "manifests.ndjson"
    _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])
    _write_records(manifest_path, [_genesis_entry(_signer(), previous_manifest_hash=predecessor)])
    manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a")
    assert [manifest["run_id"] for manifest in manifests] == ["run-a", "run-b"]
    return audit_path, manifest_path


def _under_key(current: str, signer: str) -> str:
    """A `match` pattern for the log-under-another-key error."""
    return "^" + re.escape(f"manifest log is under key '{current}', not the signer's '{signer}'") + "$"


def _assert_names(excinfo: pytest.ExceptionInfo[ManifestVerificationError], *names: str) -> None:
    message = str(excinfo.value)
    assert all(name in message for name in names), message


def _key_2_entry(head: str, **changes: Any) -> dict[str, Any]:
    return _rotation_entry(_signer(_OTHER_KEY, "key-2"), head, **changes)


def _forged_co_signature(entry: dict[str, Any]) -> dict[str, Any]:
    """The entry with its co-signature value replaced by a signature over other bytes."""
    entry["previous_key_signature"]["value"] = _signer().sign(b"other bytes")
    return entry


# Rotation entries that must not verify, each built from the head it should chain onto.
_FORGED_ENTRIES: dict[str, Callable[[str], dict[str, Any]]] = {
    "wrong-key-material": lambda head: _rotation_entry(_signer(b"x" * 32, "key-2"), head),
    "unknown-key-id": lambda head: _rotation_entry(_signer(b"u" * 32, "key-unknown"), head),
    "wrong-chain": lambda head: _key_2_entry("0" * 64),
    "tampered-rotated-at": lambda head: {**_key_2_entry(head), "rotated_at": "2030-01-01T00:00:00.000000Z"},
    "wrong-algorithm": lambda head: {
        **_key_2_entry(head),
        "signature": {**_key_2_entry(head)["signature"], "algorithm": "OTHER"},
    },
    "wrong-version": lambda head: _key_2_entry(head, manifest_version=3),
    "extra-key": lambda head: _key_2_entry(head, run_id="run-x"),
    "missing-key": lambda head: _key_2_entry(head, rotated_at=_MISSING),
    "unknown-kind": lambda head: _key_2_entry(head, kind="seal"),
    "null-kind": lambda head: _key_2_entry(head, kind=None),
    "unhashable-kind": lambda head: _key_2_entry(head, kind=["key_rotation"]),
    "manifest-with-kind": lambda head: _key_2_entry(
        head,
        run_id="run-1",
        sealed_at="2026-01-01T00:00:00.000000Z",
        hash_algorithm="sha256",
        record_count=0,
        record_hashes=[],
        compliant=True,
    ),
    "missing-co-signature": lambda head: _key_2_entry(head, previous_key_signature=_MISSING),
    "co-signed-by-a-non-current-key": lambda head: _key_2_entry(head, outgoing=_signer(_OTHER_KEY, "key-2")),
    "forged-co-signature-value": lambda head: _forged_co_signature(_key_2_entry(head)),
}

_BAD_SIGNATURE = "signature does not match the manifest"
_BAD_SHAPE = "must be a 'key_rotation' entry with exactly"
# The `match` fragment of the error each forged entry must raise, so none passes for the wrong reason.
_FORGED_ENTRY_REASONS: dict[str, str] = {
    "wrong-key-material": _BAD_SIGNATURE,
    "unknown-key-id": "signature key_id 'key-unknown' matches no known key",
    "wrong-chain": "manifest chain is broken at a key rotation entry",
    "tampered-rotated-at": _BAD_SIGNATURE,
    "wrong-algorithm": "signature algorithm 'OTHER' is not the signer's",
    "wrong-version": "unsupported manifest_version 3",
    "extra-key": _BAD_SHAPE,
    "missing-key": _BAD_SHAPE,
    "unknown-kind": _BAD_SHAPE,
    "null-kind": _BAD_SHAPE,
    "unhashable-kind": _BAD_SHAPE,
    "manifest-with-kind": _BAD_SHAPE,
    "missing-co-signature": _BAD_SHAPE,
    "co-signed-by-a-non-current-key": "previous_key_signature key_id 'key-2' is not the current key 'key-1'",
    "forged-co-signature-value": "previous_key_signature does not match the manifest",
}

_NOT_AN_ENTRY = "is not a key rotation entry"
_UNTERMINATED = "does not end with a newline"
# The message fragment of the error each refusal of quarantine_from_rotation_entry must raise, so none passes for the
# wrong reason.
_REFUSAL_REASONS: dict[str, str] = {
    "anchor-not-in-the-log": "has the head",
    "first-dropped-line-is-a-seal": _NOT_AN_ENTRY,
    "first-dropped-line-is-torn": _UNTERMINATED,
    "first-dropped-line-is-an-unterminated-entry": _UNTERMINATED,
    "first-dropped-line-is-not-json": "is not valid JSON",
    "first-dropped-line-is-not-an-entry": _NOT_AN_ENTRY,
    "signer-is-not-current-at-the-anchor": "not the signer's",
    "prefix-does-not-verify": _BAD_SIGNATURE,
    "quarantine-log-without-a-final-newline": _UNTERMINATED,
}


def _sealed_log(directory: Path) -> tuple[Path, Path]:
    """Three runs plus one record without a run_id, all sealed."""
    audit_path = directory / "audit.ndjson"
    manifest_path = directory / "manifests.ndjson"
    _write_records(
        audit_path,
        [
            _record("run-a", 1),
            _record("run-a", 2),
            _record(None, 3),
            _record("run-b", 4),
            _record("run-b", 5),
            _record("run-c", 6),
        ],
    )
    manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
    assert [manifest["run_id"] for manifest in manifests] == ["run-a", "run-b", "run-c"]
    return audit_path, manifest_path


def _rotate(manifest_path: Path, signer: ManifestSigner, *previous_signers: ManifestSigner) -> dict[str, Any]:
    """Rotate the log to `signer`, anchored on its current head."""
    head = manifest_hash(_read_lines(manifest_path)[-1])
    return rotate_manifest_key(manifest_path, signer=signer, previous_signers=previous_signers, expected_head=head)


def _rotated_three_key_log(
    directory: Path, *, seal_between: bool = True
) -> tuple[Path, Path, ManifestSigner, ManifestSigner, ManifestSigner]:
    """One run each sealed by three keys in turn, with two rotations; `seal_between=False` skips the key-2 run."""
    audit_path = directory / "audit.ndjson"
    manifest_path = directory / "manifests.ndjson"
    key_1, key_2, key_3 = _signer(key_id="key-1"), _signer(_OTHER_KEY, "key-2"), _signer(_THIRD_KEY, "key-3")
    _write_records(audit_path, [_record("run-a", 1)])
    seal_ndjson_runs(audit_path, manifest_path, signer=key_1)
    _rotate(manifest_path, key_2, key_1)
    if seal_between:
        _write_records(audit_path, [_record("run-b", 2)])
        seal_ndjson_runs(audit_path, manifest_path, signer=key_2, previous_signers=[key_1])
    _rotate(manifest_path, key_3, key_1, key_2)
    _write_records(audit_path, [_record("run-c", 3)])
    seal_ndjson_runs(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])
    return audit_path, manifest_path, key_1, key_2, key_3


def _mixed_algorithm_log(directory: Path) -> tuple[Path, Path, ManifestSigner, ManifestSigner, ManifestSigner]:
    """One run each sealed by HMAC key-1, Ed25519 key-2 and HMAC key-3 in turn, with two rotations."""
    audit_path = directory / "audit.ndjson"
    manifest_path = directory / "manifests.ndjson"
    key_1 = HmacSha256Signer(_KEY, "key-1")
    key_2 = Ed25519Signer(_OTHER_KEY, "key-2")
    key_3 = HmacSha256Signer(_THIRD_KEY, "key-3")
    _write_records(audit_path, [_record("run-a", 1)])
    seal_ndjson_runs(audit_path, manifest_path, signer=key_1)
    _rotate(manifest_path, key_2, key_1)
    _write_records(audit_path, [_record("run-b", 2)])
    seal_ndjson_runs(audit_path, manifest_path, signer=key_2, previous_signers=[key_1])
    _rotate(manifest_path, key_3, key_1, key_2)
    _write_records(audit_path, [_record("run-c", 3)])
    seal_ndjson_runs(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])
    return audit_path, manifest_path, key_1, key_2, key_3


def _unwrapped(signer: ManifestSigner) -> ManifestSigner:
    return signer


def _pending_write(
    directory: Path, writer: str
) -> tuple[Path, Callable[..., list[dict[str, Any]]], Callable[[], str | None]]:
    """A sealed log with one pending write ("seal": runs d, e, f; "rotate": key-1 to key-2). Returns the manifest path,
    `write(wrap)` (the appended lines; `wrap` maps each signer, to spy on signing) and `verify()` (the new head)."""
    audit_path, manifest_path = _sealed_log(directory)
    head = manifest_hash(_read_lines(manifest_path)[-1])
    key, key_id = (_KEY, "key-1") if writer == "seal" else (_OTHER_KEY, "key-2")
    if writer == "seal":
        _write_records(
            audit_path, [_record(f"run-{name}", second) for name, second in zip("def", (7, 8, 9), strict=True)]
        )

    def write(wrap: Callable[[ManifestSigner], ManifestSigner] = _unwrapped) -> list[dict[str, Any]]:
        signer = wrap(_signer(key, key_id))
        if writer == "seal":
            return seal_ndjson_runs(audit_path, manifest_path, signer=signer, expected_head=head)
        old_signer = wrap(_signer(_KEY, "key-1"))
        return [rotate_manifest_key(manifest_path, signer=signer, previous_signers=[old_signer], expected_head=head)]

    def verify() -> str | None:
        previous = [] if writer == "seal" else [_signer()]
        return verify_ndjson_log(audit_path, manifest_path, signer=_signer(key, key_id), previous_signers=previous)

    return manifest_path, write, verify


def _truncated_log(directory: Path) -> tuple[Path, Path, str]:
    """Seal runs a, b, c, cut the log to its first line, edit a run-b record; returns the head before the cut."""
    audit_path, manifest_path = _sealed_log(directory)
    manifests = _read_lines(manifest_path)
    head = manifest_hash(manifests[-1])
    _rewrite_lines(manifest_path, manifests[:1])
    records = _read_lines(audit_path)
    assert records[3]["run_id"] == "run-b"
    records[3]["tenant_id"] = "tenant-other"
    _rewrite_lines(audit_path, records)
    return audit_path, manifest_path, head


def _torn(path: Path, tail: bytes) -> None:
    path.write_bytes(path.read_bytes() + tail)


def _insert_line(path: Path, index: int, line: bytes) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    lines.insert(index, line)
    path.write_bytes(b"".join(lines))


# Small line cap for the tests that monkeypatch run_manifest.MAX_LINE_BYTES; every regular line stays under it.
_CAP = 2048


def _cap(monkeypatch: pytest.MonkeyPatch, cap: int = _CAP) -> int:
    _patch_bindings(monkeypatch, "MAX_LINE_BYTES", cap)
    return cap


def _oversized_line(extra: Mapping[str, Any] | None = None, cap: int = _CAP) -> bytes:
    """A valid JSON object line (no newline) longer than `cap`, so only the cap makes it bad."""
    line = json.dumps({**(extra or {}), "pad": "x" * cap}, sort_keys=True).encode("utf-8")
    assert len(line) > cap
    return line


def _torn_manifest_log(directory: Path) -> tuple[Path, Path, str]:
    """A manifest log ending in half a manifest line (line 4); returns the head before the damage."""
    audit_path, manifest_path = _sealed_log(directory)
    head = manifest_hash(_read_lines(manifest_path)[-1])
    last = manifest_path.read_bytes().splitlines(keepends=True)[-1]
    _torn(manifest_path, last[: len(last) // 2])
    return audit_path, manifest_path, head


def _torn_rotation_log(directory: Path) -> tuple[Path, Path, str]:
    """A manifest log ending in half a key-2 rotation entry; returns the head before the damage."""
    audit_path, manifest_path = _sealed_log(directory)
    head = manifest_hash(_read_lines(manifest_path)[-1])
    entry = _canonical(_rotation_entry(_signer(_OTHER_KEY, "key-2"), head)) + b"\n"
    _torn(manifest_path, entry[: len(entry) // 2])
    return audit_path, manifest_path, head


def _damaged_log(directory: Path) -> tuple[Path, Path, str]:
    """A torn manifest log plus an undecodable audit line (line 4) and a torn last audit line (line 8)."""
    audit_path, manifest_path, head = _torn_manifest_log(directory)
    _insert_line(audit_path, 3, b"\xff\xfe\n")
    _torn(audit_path, b'{"run_id": "run-d", "tenant')
    return audit_path, manifest_path, head


def _log_cut_mid_write(directory: Path) -> tuple[Path, Path, list[str]]:
    """Seal run-a, then run-b and run-c in one write cut inside run-c's line; returns the three heads."""
    audit_path = directory / "audit.ndjson"
    manifest_path = directory / "manifests.ndjson"
    _write_records(audit_path, [_record("run-a", 1)])
    heads = [manifest_hash(manifest) for manifest in seal_ndjson_runs(audit_path, manifest_path, signer=_signer())]
    _write_records(audit_path, [_record("run-b", 2), _record("run-c", 3)])
    heads += [manifest_hash(manifest) for manifest in seal_ndjson_runs(audit_path, manifest_path, signer=_signer())]
    assert len(heads) == 3
    lines = manifest_path.read_bytes().splitlines(keepends=True)
    manifest_path.write_bytes(b"".join(lines[:2]) + lines[2][: len(lines[2]) // 2])
    return audit_path, manifest_path, heads


def _log_with_rotation_entry(directory: Path) -> tuple[Path, Path, str]:
    """Two key-1 seals, a key-2 rotation entry (line 3) and a key-2 seal (line 4); returns the head before the entry."""
    key_1, key_2 = _signer(), _signer(_OTHER_KEY, "key-2")
    audit_path, manifest_path = _build_log(
        directory, ("seal", key_1), ("seal", key_1), ("rotate", key_2), ("seal", key_2)
    )
    return audit_path, manifest_path, manifest_hash(_read_lines(manifest_path)[1])


def _trace(directory: Path) -> Path:
    return directory / "quarantine.ndjson"


def _trace_entry_ref(
    signer: ManifestSigner, previous_entry_hash: str | None, *, v1: bool = False, **changes: Any
) -> dict[str, Any]:
    """A hand-built trace line signed by `signer`: v2 (chained, v2 payload) or, with `v1`, the legacy v1 one."""
    entry: dict[str, Any] = {
        "quarantine_version": 1 if v1 else 2,
        "quarantined_at": "2026-01-01T00:00:00.000000Z",
        "file": "audit",
        "path": "audit.ndjson",
        "line": 4,
        "offset": 0,
        "length": 3,
        "sha256": _sha256(b"abc"),
        "reason": "not valid JSON",
        "raw_base64": base64.b64encode(b"abc").decode("ascii"),
        **changes,
    }
    if not v1:
        entry["previous_entry_hash"] = previous_entry_hash
    block = {"algorithm": signer.algorithm, "key_id": signer.key_id}
    payload = _canonical(entry) if v1 else _signing_payload_ref({**entry, "signature": block})
    entry["signature"] = {**block, "value": signer.sign(payload)}
    return entry


def _trace_chain(*signers: ManifestSigner) -> list[dict[str, Any]]:
    """A valid v2 trace, one line per signer."""
    entries: list[dict[str, Any]] = []
    for number, signer in enumerate(signers, start=1):
        previous = _sha256(_canonical(entries[-1])) if entries else None
        entries.append(_trace_entry_ref(signer, previous, line=number))
    return entries


def _verify_quarantine(
    path: Path,
    signer: ManifestSigner,
    *previous_signers: ManifestSigner,
    anchored_heads: Iterable[str] = (),
) -> str | None:
    return verify_quarantine_log(path, signer=signer, previous_signers=previous_signers, anchored_heads=anchored_heads)


def _ndjson_anchor(path: Path) -> Any:
    return NdjsonHeadAnchor(path)


def _quarantine(
    directory: Path,
    audit_path: Path,
    manifest_path: Path,
    *,
    signer: ManifestSigner | None = None,
    previous_signers: Iterable[ManifestSigner] = (),
    expected_head: str | None = None,
    dry_run: bool = False,
    anchored_heads: Iterable[str] = (),
    head_anchor: HeadAnchor | None = None,
) -> list[QuarantinedLine]:
    return quarantine_damaged_lines(
        audit_path,
        manifest_path,
        quarantine_path=_trace(directory),
        signer=signer or _signer(),
        previous_signers=previous_signers,
        expected_head=expected_head,
        dry_run=dry_run,
        anchored_heads=anchored_heads,
        head_anchor=head_anchor,
    )


def _quarantine_from_entry(
    directory: Path,
    manifest_path: Path,
    *,
    expected_head: str,
    signer: ManifestSigner | None = None,
    previous_signers: Iterable[ManifestSigner] = (),
    dry_run: bool = False,
    anchored_heads: Iterable[str] = (),
    head_anchor: HeadAnchor | None = None,
) -> list[QuarantinedLine]:
    return quarantine_from_rotation_entry(
        manifest_path,
        quarantine_path=_trace(directory),
        signer=signer or _signer(),
        previous_signers=previous_signers,
        expected_head=expected_head,
        dry_run=dry_run,
        anchored_heads=anchored_heads,
        head_anchor=head_anchor,
    )


# The hook of each recovery function for the tests both share: builds a log with something to drop under a directory,
# and returns the manifest log the repair cuts, the repair (given the directory for its quarantine log) and how many
# lines it drops.
_Repair = Callable[..., list[QuarantinedLine]]
_Recovery = Callable[[Path], tuple[Path, _Repair, int]]


def _damaged_lines_recovery(directory: Path) -> tuple[Path, _Repair, int]:
    audit_path, manifest_path, _ = _damaged_log(directory)
    return manifest_path, lambda trace_dir, **kwargs: _quarantine(trace_dir, audit_path, manifest_path, **kwargs), 3


def _rotation_entry_recovery(directory: Path) -> tuple[Path, _Repair, int]:
    _, manifest_path, anchor = _log_with_rotation_entry(directory)
    return (
        manifest_path,
        lambda trace_dir, **kwargs: _quarantine_from_entry(trace_dir, manifest_path, expected_head=anchor, **kwargs),
        2,
    )


_RECOVERIES: dict[str, _Recovery] = {
    "damaged-lines": _damaged_lines_recovery,
    "rotation-entry": _rotation_entry_recovery,
}
_both_recoveries = pytest.mark.parametrize("recovery", list(_RECOVERIES.values()), ids=list(_RECOVERIES))


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def _aliased(tmp_path: Path, target: Path, spelling: str) -> Path:
    """`target` itself, or a symlink to it: two spellings that resolve to the same file."""
    if spelling == "same-path":
        return target
    link = tmp_path / "aliased.ndjson"
    link.symlink_to(target)
    return link


def _spans(file: str, data: bytes, numbers: Iterable[int]) -> list[tuple[str, int, int, int, str]]:
    """What a recovery reports for the 1-based lines `numbers` of `data`."""
    lines = data.splitlines(keepends=True)
    return [(file, n, len(b"".join(lines[: n - 1])), len(lines[n - 1]), _sha256(lines[n - 1])) for n in numbers]


def _summary(removed: Iterable[QuarantinedLine]) -> list[tuple[str, int, int, int, str]]:
    return [(item.file, item.line, item.offset, item.length, item.sha256) for item in removed]


def _assert_raises_and_unchanged(
    directory: Path, call: Callable[[], object]
) -> pytest.ExceptionInfo[ManifestVerificationError]:
    """`call` raises ManifestVerificationError and changes nothing in `directory`."""
    before = _snapshot(directory)

    with pytest.raises(ManifestVerificationError) as excinfo:
        call()

    assert _snapshot(directory) == before
    return excinfo


def _assert_refused(
    directory: Path,
    audit_path: Path,
    manifest_path: Path,
    *,
    signer: ManifestSigner | None = None,
    expected_head: str | None = None,
    dry_run: bool = False,
) -> None:
    """Recovery raises ManifestVerificationError and changes nothing."""
    _assert_raises_and_unchanged(
        directory,
        lambda: _quarantine(
            directory, audit_path, manifest_path, signer=signer, expected_head=expected_head, dry_run=dry_run
        ),
    )


def _assert_traced(
    directory: Path,
    manifest_path: Path,
    manifest_before: bytes,
    removed: list[QuarantinedLine],
    signer: ManifestSigner | None = None,
    *,
    audit: tuple[Path, bytes] | None = None,
) -> None:
    """The trace holds one entry per dropped line: its report, its raw bytes and a signature by `signer`.
    `audit` is the audit file and its bytes before the recovery, if the recovery may drop audit lines."""
    signer = signer or _signer()
    logs = {"manifest": (manifest_path, manifest_before)}
    if audit:
        logs["audit"] = audit
    entries = _read_lines(_trace(directory))
    assert len(entries) == len(removed)
    assert _trace(directory).read_bytes() == b"".join(_canonical(entry) + b"\n" for entry in entries)
    for entry, item in zip(entries, removed, strict=True):
        path, before = logs[item.file]
        raw = base64.b64decode(entry["raw_base64"], validate=True)
        assert set(entry) == _EXPECTED_QUARANTINE_KEYS
        assert entry["quarantine_version"] == 2
        assert entry["path"] == str(path)
        assert {field: entry[field] for field in dataclasses.asdict(item)} == dataclasses.asdict(item)
        assert raw == before.splitlines(keepends=True)[item.line - 1]
        assert entry["sha256"] == _sha256(raw)
        signature = entry["signature"]
        assert set(signature) == {"algorithm", "key_id", "value"}
        assert (signature["algorithm"], signature["key_id"]) == (signer.algorithm, signer.key_id)
        assert signature["value"] == signer.sign(_signing_payload_ref(entry))
    assert [entry["previous_entry_hash"] for entry in entries] == [
        None,
        *(_sha256(_canonical(e)) for e in entries[:-1]),
    ]


def _lock_refused(path: Path) -> bool:
    fcntl = pytest.importorskip("fcntl")
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def _locked_during_digest(monkeypatch: pytest.MonkeyPatch, manifest_path: Path) -> list[bool]:
    """Spy on `_digest_runs`: the returned list gets, per call, whether `manifest_path` was locked at that moment."""
    locked: list[bool] = []
    real_digest_runs = _verify._digest_runs

    def digest_runs(*args: Any, **kwargs: Any) -> Any:
        locked.append(_lock_refused(manifest_path))
        return real_digest_runs(*args, **kwargs)

    _patch_bindings(monkeypatch, "_digest_runs", digest_runs)
    return locked


def _flock_unsupported(fd: int, operation: int) -> None:
    raise OSError(errno.ENOLCK, "no locks")


def _swap_on_first_flock(monkeypatch: pytest.MonkeyPatch, path: Path, tmp_path: Path) -> tuple[Path, list[int], int]:
    """Make the first flock on `path`'s current inode first replace `path` with a copy (a new inode), as a rotation
    would after the caller opened the old file but before it locked it. Returns the hard-linked archive of the old
    file, the inode of every flock on the old or new file in order, and the new file's inode."""
    fcntl = pytest.importorskip("fcntl")
    archive = tmp_path / "archive.ndjson"
    os.link(path, archive)
    new = tmp_path / "new.ndjson"
    new.write_bytes(path.read_bytes())
    old_ino, new_ino = path.stat().st_ino, new.stat().st_ino
    locked: list[int] = []
    real_flock = fcntl.flock

    def flock(fd: int, operation: int) -> None:
        ino = os.fstat(fd).st_ino
        if ino == old_ino and not locked:
            os.replace(new, path)
        if ino in (old_ino, new_ino):
            locked.append(ino)
        real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", flock)
    return archive, locked, new_ino


def _rotate_segment(audit_path: Path, manifest_path: Path, **kwargs: Any) -> dict[str, Any]:
    """rotate_ndjson_segment with the default signer and log_id."""
    entry: dict[str, Any] = rotate_ndjson_segment(
        audit_path, manifest_path, **{"signer": _signer(), "log_id": "log-a", **kwargs}
    )
    return entry


def _archive(path: Path, number: int = 1) -> Path:
    return path.with_name(f"{path.name}.{number:06d}")


def _archives(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.[0-9][0-9][0-9][0-9][0-9][0-9]"))


def _same_inode(first: Path, second: Path) -> bool:
    return first.stat().st_ino == second.stat().st_ino


def _pending_log(directory: Path) -> tuple[Path, Path]:
    """`_genesis_log` plus a pending run-p (two lines), an unattributed line and a blank-run_id line."""
    audit_path, manifest_path = _genesis_log(directory)
    _write_records(audit_path, [_record("run-p", 4), _record(None, 5), _record("run-p", 6), _record("   ", 7)])
    return audit_path, manifest_path


def _rotated_with_stray(directory: Path) -> tuple[Path, Path]:
    """`_pending_log` rotated, then a stray record of run-a, which the archived segment sealed."""
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    _write_records(audit_path, [_record("run-a", 9)])
    return audit_path, manifest_path


class _Crash(BaseException):
    """Stands for the process dying: no `except Exception` cleanup runs."""


def _crash_on_replace(call_number: int) -> Callable[[Any, Any], None]:
    """An os.replace that raises _Crash on its `call_number`th call and otherwise replaces."""
    real_replace = os.replace
    calls: list[Any] = []

    def replace(source: Any, target: Any) -> None:
        calls.append(source)
        if len(calls) == call_number:
            raise _Crash()
        real_replace(source, target)

    return replace


def _crashed_rotation(directory: Path, call_number: int) -> tuple[Path, Path, dict[str, bytes]]:
    """A pending log whose rotation dies on its `call_number`th os.replace; returns the paths and the files before."""
    audit_path, manifest_path = _pending_log(directory)
    before = _snapshot(directory)
    with patch("os.replace", _crash_on_replace(call_number)):
        with pytest.raises(_Crash):
            _rotate_segment(audit_path, manifest_path)
    return audit_path, manifest_path, before


class _PrefixSigner:
    """A signer that inherits from nothing, so only the protocol is relied on."""

    algorithm = "TEST-SHA256"
    key_id = "test-key"

    def sign(self, payload: bytes) -> str:
        return _sha256(b"prefix:" + payload)

    def verify(self, payload: bytes, signature: str) -> bool:
        try:
            return hmac.compare_digest(self.sign(payload).encode("ascii"), signature.encode("ascii"))
        except (AttributeError, UnicodeEncodeError):
            return False


class _RecordingAnchor:
    """A HeadAnchor that records each written head and what the log held at that moment."""

    def __init__(self, log_path: Path | None = None, *, fail: bool = False) -> None:
        self.heads: list[str] = []
        self.last_lines_seen: list[str] = []
        self._log_path = log_path
        self._fail = fail

    def write(self, head: str) -> None:
        if self._log_path is not None:
            self.last_lines_seen.append(manifest_hash(_read_lines(self._log_path)[-1]))
        self.heads.append(head)
        if self._fail:
            raise RuntimeError("anchor down")

    def latest(self) -> str | None:
        return self.heads[-1] if self.heads else None


class _HookedSigner:
    """Delegates to `inner` after running a hook, so it wraps any algorithm."""

    def __init__(
        self,
        inner: ManifestSigner,
        *,
        on_sign: Callable[[bytes], None] | None = None,
        on_verify: Callable[[bytes], None] | None = None,
    ) -> None:
        self._inner = inner
        self._on_sign = on_sign
        self._on_verify = on_verify
        self.algorithm = inner.algorithm
        self.key_id = inner.key_id

    def sign(self, payload: bytes) -> str:
        if self._on_sign is not None:
            self._on_sign(payload)
        return self._inner.sign(payload)

    def verify(self, payload: bytes, signature: str) -> bool:
        if self._on_verify is not None:
            self._on_verify(payload)
        return self._inner.verify(payload, signature)


_LAYERS = [
    "_records",
    "_signers",
    "_verify",
    "_seal_index",
    "_segments",
    "_quarantine",
    "run_manifest",
    "audit_extender",
    "otel_log_sink",
    "__init__",
]
_FACADE_HOMES = [
    *((n, "_signers") for n in ("ManifestSigner", "HmacSha256Signer", "Ed25519Signer")),
    *(
        (n, "_verify")
        for n in (
            "ManifestVerificationError",
            "RunNotPendingError",
            "RunAlreadySealedError",
            "KeyAlreadyCurrentError",
            "HeadAnchor",
            "NdjsonHeadAnchor",
            "LogCoverage",
            "MAX_LINE_BYTES",
            "seal_run",
            "manifest_hash",
            "verify_manifest",
        )
    ),
    *((n, "run_manifest") for n in ("rotate_manifest_key", "seal_ndjson_runs", "verify_ndjson_log")),
    ("verify_ndjson_log_coverage", "run_manifest"),
    *(
        (n, "_quarantine")
        for n in (
            "QuarantinedLine",
            "verify_quarantine_log",
            "quarantine_damaged_lines",
            "quarantine_from_rotation_entry",
        )
    ),
    *((n, "_segments") for n in ("rotate_ndjson_segment", "verify_ndjson_segments")),
]


class TestRunManifestPublicApi:
    def test_run_manifest_names_are_in_the_package_all(self) -> None:
        assert {
            "ManifestSigner",
            "HmacSha256Signer",
            "Ed25519Signer",
            "ManifestVerificationError",
            "RunNotPendingError",
            "RunAlreadySealedError",
            "SealedRunRefusedError",
            "seal_run",
            "manifest_hash",
            "verify_manifest",
            "seal_ndjson_runs",
            "verify_ndjson_log",
            "LogCoverage",
            "verify_ndjson_log_coverage",
            "QuarantinedLine",
            "quarantine_damaged_lines",
            "quarantine_from_rotation_entry",
            "IdentityRequiredError",
            "KeyAlreadyCurrentError",
            "rotate_manifest_key",
            "rotate_ndjson_segment",
            "verify_ndjson_segments",
            "HeadAnchor",
            "NdjsonHeadAnchor",
            "verify_quarantine_log",
        } <= set(audit_package.__all__)

    @pytest.mark.parametrize("name", ["HeadAnchor", "NdjsonHeadAnchor", "verify_quarantine_log"])
    def test_anchor_and_quarantine_names_are_the_run_manifest_objects(self, name: str) -> None:
        assert getattr(audit_package, name) is getattr(run_manifest_module, name)

    def test_run_not_pending_error_is_a_value_error_but_not_a_verification_error(self) -> None:
        assert issubclass(RunNotPendingError, ValueError)
        assert not issubclass(RunNotPendingError, ManifestVerificationError)

    def test_key_already_current_error_is_a_value_error_but_not_a_verification_error(self) -> None:
        assert issubclass(KeyAlreadyCurrentError, ValueError)
        assert not issubclass(KeyAlreadyCurrentError, ManifestVerificationError)

    def test_identity_required_error_is_a_runtime_error_but_not_a_value_error(self) -> None:
        assert issubclass(IdentityRequiredError, RuntimeError)
        assert not issubclass(IdentityRequiredError, ValueError)

    def test_run_already_sealed_error_is_a_run_not_pending_error(self) -> None:
        assert issubclass(RunAlreadySealedError, RunNotPendingError)

    def test_sealed_run_refused_error_is_a_runtime_error_but_not_a_value_error(self) -> None:
        assert issubclass(SealedRunRefusedError, RuntimeError)
        assert not issubclass(SealedRunRefusedError, ValueError)

    def test_append_records_and_canonical_json_come_from_one_shared_private_records_module(self) -> None:
        import mloda.enterprise.extenders.audit._records as records_module

        # getattr: _verify and audit_extender import these, they do not define them.
        assert getattr(_verify, "_append_records") is records_module._append_records
        assert getattr(_verify, "_canonical_json") is records_module._canonical_json
        assert getattr(audit_extender_module, "_append_records") is records_module._append_records
        assert getattr(audit_extender_module, "_canonical_json") is records_module._canonical_json
        assert records_module._is_blank("") is True
        assert records_module._is_blank("value") is False

    @pytest.mark.parametrize("name, home", _FACADE_HOMES)
    def test_facade_names_are_the_objects_their_defining_module_defines(self, name: str, home: str) -> None:
        defining = importlib.import_module(f"mloda.enterprise.extenders.audit.{home}")
        assert getattr(run_manifest_module, name) is getattr(defining, name)
        if hasattr(audit_package, name):
            assert getattr(audit_package, name) is getattr(run_manifest_module, name)

    def test_audit_source_modules_import_only_downward_and_only_from_the_defining_module(self) -> None:
        package = Path(audit_package.__file__ or "").parent
        trees = {m: ast.parse((package / f"{m}.py").read_text()) for m in _LAYERS}
        defined = {
            m: {
                n
                for node in tree.body
                for n in (
                    [node.name]
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                    else [t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)]
                    + (
                        [node.target.id]
                        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                        else []
                    )
                )
            }
            for m, tree in trees.items()
        }
        problems = []
        for module, tree in trees.items():
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not (
                    node.level or "mloda.enterprise" in (node.module or "")
                ):
                    continue
                last = (node.module or "").rsplit(".", 1)[-1]
                for alias in node.names:
                    source = last if last in _LAYERS else alias.name
                    if source not in _LAYERS:
                        continue
                    if _LAYERS.index(source) > _LAYERS.index(module):
                        problems.append(f"{module} imports upward from {source}")
                    private = alias.name.startswith("_") and alias.name != source
                    if private and alias.name not in defined[source] and alias.asname != alias.name:
                        problems.append(f"{module} imports {alias.name} from {source}, which does not define it")
        assert problems == []


@_both_algorithms
class TestManifestSignerContract:
    def test_verify_accepts_its_own_signature(self) -> None:
        signer = _signer()

        assert signer.verify(b"payload", signer.sign(b"payload")) is True

    def test_verify_rejects_a_tampered_payload(self) -> None:
        signer = _signer()

        assert signer.verify(b"payload-tampered", signer.sign(b"payload")) is False

    def test_verify_rejects_a_tampered_signature(self) -> None:
        signer = _signer()
        signature = signer.sign(b"payload")
        tampered = ("1" if signature[0] == "0" else "0") + signature[1:]

        assert signer.verify(b"payload", tampered) is False

    def test_verify_rejects_the_signature_of_a_different_key(self) -> None:
        signature = _signer(_OTHER_KEY).sign(b"payload")

        assert _signer().verify(b"payload", signature) is False

    @pytest.mark.parametrize("signature", ["é" * 64, chr(0xD800)], ids=["non-ascii", "lone-surrogate"])
    def test_verify_returns_false_for_a_signature_it_cannot_compare(self, signature: str) -> None:
        assert _signer().verify(b"payload", signature) is False

    @pytest.mark.parametrize("make", list(_NON_STR_SIGNATURES.values()), ids=list(_NON_STR_SIGNATURES))
    def test_verify_returns_false_and_never_raises_for_a_signature_that_is_not_a_str(
        self, make: Callable[[str], Any]
    ) -> None:
        signer = _signer()
        # Control: the original verifies.
        assert signer.verify(b"payload", signer.sign(b"payload")) is True

        assert signer.verify(b"payload", make(signer.sign(b"payload"))) is False

    @pytest.mark.parametrize("key_id", ["", "   ", "\t\n"])
    def test_blank_key_id_raises_value_error(self, key_id: str) -> None:
        with pytest.raises(ValueError):
            _signer_class()(_KEY, key_id)

    def test_non_str_key_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _signer_class()(_KEY, None)  # type: ignore[arg-type]

    def test_repr_does_not_contain_the_key(self) -> None:
        key = b"do-not-print-this-signing-key-01"
        signer = _signer_class()(key, "key-1")

        for text in (repr(signer), str(signer)):
            assert key.decode("utf-8") not in text
            assert key.hex() not in text


class TestHmacSha256Signer:
    def test_algorithm_and_key_id(self) -> None:
        signer = HmacSha256Signer(_KEY, "key-2026-01")

        assert signer.algorithm == "HMAC-SHA256"
        assert signer.key_id == "key-2026-01"

    def test_sign_is_the_lowercase_hex_hmac_sha256(self) -> None:
        payload = b'{"a": 1}'

        assert _signer().sign(payload) == hmac.new(_KEY, payload, hashlib.sha256).hexdigest()

    @pytest.mark.parametrize("key", [b"", b"k" * 31])
    def test_short_key_raises_value_error(self, key: bytes) -> None:
        with pytest.raises(ValueError):
            HmacSha256Signer(key, "key-1")

    def test_non_bytes_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            HmacSha256Signer("k" * 32, "key-1")  # type: ignore[arg-type]

    def test_repr_does_not_contain_the_key(self) -> None:
        key = b"do-not-print-this-signing-key-0123456789"
        signer = HmacSha256Signer(key, "key-1")

        for text in (repr(signer), str(signer)):
            assert key.decode("utf-8") not in text
            assert key.hex() not in text


# Forms of a valid signature that verify must refuse, some of which a lenient hex parser would accept.
_MANGLED_SIGNATURES: dict[str, Callable[[str], str]] = {
    "uppercase": str.upper,
    "leading-space": lambda signature: " " + signature,
    "trailing-newline": lambda signature: signature + "\n",
    "spaced-byte-pairs": lambda signature: " ".join(signature[i : i + 2] for i in range(0, len(signature), 2)),
    "127-chars": lambda signature: signature[:-1],
    "129-chars": lambda signature: signature + "0",
    "non-hex": lambda signature: "g" + signature[1:],
    "trailing-spaces-at-128": lambda signature: signature[:-2] + "  ",
    "empty": lambda signature: "",
    "non-ascii": lambda signature: "é" * len(signature),
    "lone-surrogate": lambda signature: chr(0xD800) * len(signature),
}

_not_a_32_byte_key = pytest.mark.parametrize(
    "key", [b"", b"k" * 31, b"k" * 33, "k" * 32], ids=["empty", "31-bytes", "33-bytes", "str"]
)


class TestEd25519Signer:
    def test_algorithm_and_key_id(self) -> None:
        signer = Ed25519Signer(_KEY, "key-2026-01")

        assert signer.algorithm == "Ed25519"
        assert signer.key_id == "key-2026-01"

    def test_sign_matches_the_rfc_8032_test_vector(self) -> None:
        signer = Ed25519Signer(_RFC8032_SEED, "rfc-8032")

        assert signer.sign(b"") == _RFC8032_SIGNATURE
        assert signer.verify(b"", _RFC8032_SIGNATURE) is True

    def test_a_public_key_signer_verifies_the_rfc_8032_test_vector(self) -> None:
        verifier = Ed25519Signer.from_public_key(_RFC8032_PUBLIC_KEY, "rfc-8032")

        assert verifier.verify(b"", _RFC8032_SIGNATURE) is True
        assert verifier.verify(b"tampered", _RFC8032_SIGNATURE) is False

    def test_sign_returns_128_lowercase_hex_characters(self) -> None:
        signature = Ed25519Signer(_KEY, "key-1").sign(b"payload")

        assert re.fullmatch(r"[0-9a-f]{128}", signature)

    @pytest.mark.parametrize("public_only", [False, True], ids=["private", "public-only"])
    @pytest.mark.parametrize("mangle", list(_MANGLED_SIGNATURES.values()), ids=list(_MANGLED_SIGNATURES))
    def test_verify_returns_false_for_anything_but_128_lowercase_hex_characters(
        self, mangle: Callable[[str], str], public_only: bool
    ) -> None:
        signature = Ed25519Signer(_KEY, "key-1").sign(b"payload")
        signer = _verify_only() if public_only else Ed25519Signer(_KEY, "key-1")
        mangled = mangle(signature)
        # Control: the original verifies and the mangled form differs.
        assert signer.verify(b"payload", signature) is True
        assert mangled != signature

        assert signer.verify(b"payload", mangled) is False

    @pytest.mark.parametrize("make", list(_NON_STR_SIGNATURES.values()), ids=list(_NON_STR_SIGNATURES))
    def test_a_public_key_signer_returns_false_and_never_raises_for_a_signature_that_is_not_a_str(
        self, make: Callable[[str], Any]
    ) -> None:
        signature = Ed25519Signer(_KEY, "key-1").sign(b"payload")
        signer = _verify_only()
        # Control: the original verifies.
        assert signer.verify(b"payload", signature) is True

        assert signer.verify(b"payload", make(signature)) is False

    @_not_a_32_byte_key
    def test_a_private_key_that_is_not_32_bytes_raises_value_error(self, key: Any) -> None:
        with pytest.raises(ValueError):
            Ed25519Signer(key, "key-1")

    @pytest.mark.parametrize("key_id", ["", "   ", "\t\n"])
    def test_a_blank_key_id_raises_value_error_for_a_public_key_too(self, key_id: str) -> None:
        with pytest.raises(ValueError):
            Ed25519Signer.from_public_key(_RFC8032_PUBLIC_KEY, key_id)

    def test_a_non_str_key_id_raises_value_error_for_a_public_key_too(self) -> None:
        with pytest.raises(ValueError):
            Ed25519Signer.from_public_key(_RFC8032_PUBLIC_KEY, None)  # type: ignore[arg-type]

    @_not_a_32_byte_key
    def test_a_public_key_that_is_not_32_bytes_raises_value_error(self, key: Any) -> None:
        with pytest.raises(ValueError):
            Ed25519Signer.from_public_key(key, "key-1")

    def test_a_public_key_signer_has_the_algorithm_and_key_id(self) -> None:
        verifier = Ed25519Signer.from_public_key(_RFC8032_PUBLIC_KEY, "key-2026-01")

        assert verifier.algorithm == "Ed25519"
        assert verifier.key_id == "key-2026-01"

    def test_a_public_key_signer_verifies_the_signature_of_the_matching_private_signer(self) -> None:
        signature = Ed25519Signer(_KEY, "key-1").sign(b"payload")
        verifier = _verify_only()

        assert verifier.verify(b"payload", signature) is True
        assert verifier.verify(b"payload-tampered", signature) is False
        assert _verify_only(_OTHER_KEY).verify(b"payload", signature) is False

    def test_a_public_key_signer_cannot_sign(self) -> None:
        with pytest.raises(ValueError, match="public key"):
            _verify_only().sign(b"payload")

    def test_repr_shows_the_key_id_and_no_key_material(self) -> None:
        signer = Ed25519Signer(_KEY, "key-1")

        for text in (repr(signer), str(signer)):
            assert "key-1" in text
            assert _KEY.decode("utf-8") not in text
            assert _KEY.hex() not in text
            assert _public_key(_KEY).hex() not in text

    def test_a_public_key_signer_repr_shows_the_key_id_and_no_key_material(self) -> None:
        verifier = _verify_only()

        for text in (repr(verifier), str(verifier)):
            assert "key-1" in text
            assert _public_key(_KEY).hex() not in text


@_both_algorithms
class TestSealRun:
    def test_manifest_has_exactly_the_expected_keys(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())

        assert set(manifest) == _EXPECTED_MANIFEST_KEYS

    def test_version_run_id_and_hash_algorithm(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())

        assert manifest["manifest_version"] == 2
        assert manifest["run_id"] == "run-1"
        assert manifest["hash_algorithm"] == "sha256"

    def test_sealed_at_is_rfc3339_utc(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())

        sealed_at = manifest["sealed_at"]
        assert sealed_at.endswith("Z")
        datetime.fromisoformat(sealed_at.removesuffix("Z"))

    def test_record_hash_matches_the_line_ndjson_audit_sink_writes(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.ndjson"
        record = _record(tenant_id="tenant-ü")
        NdjsonAuditSink(path).write(record)

        manifest = seal_run([record], run_id="run-1", signer=_signer())

        assert manifest["record_hashes"] == [_sha256(path.read_bytes().removesuffix(b"\n"))]

    def test_record_hashes_are_sorted(self) -> None:
        records = [_record(second=second) for second in range(5)]

        manifest = seal_run(records, run_id="run-1", signer=_signer())

        assert manifest["record_hashes"] == sorted(_sha256(_canonical(record)) for record in records)

    def test_identical_records_are_each_counted(self) -> None:
        manifest = seal_run([_record(), _record()], run_id="run-1", signer=_signer())

        assert manifest["record_count"] == 2
        assert manifest["record_hashes"] == [_sha256(_canonical(_record()))] * 2

    def test_a_one_shot_iterable_of_records_is_accepted(self) -> None:
        records = [_record(second=1), _record(second=2)]

        manifest = seal_run((record for record in records), run_id="run-1", signer=_signer())

        assert manifest["record_count"] == 2

    def test_records_are_not_mutated(self) -> None:
        records = [_record(second=1), _record(second=2)]
        before = copy.deepcopy(records)

        seal_run(records, run_id="run-1", signer=_signer())

        assert records == before

    def test_all_compliant_records_make_a_compliant_manifest(self) -> None:
        manifest = seal_run([_record(second=1), _record(second=2)], run_id="run-1", signer=_signer())

        assert manifest["compliant"] is True

    def test_one_deny_record_makes_a_non_compliant_manifest(self) -> None:
        records = [_record(second=1), _record(second=2, tenant_id=None), _record(second=3)]

        manifest = seal_run(records, run_id="run-1", signer=_signer())

        assert manifest["compliant"] is False

    def test_record_without_a_compliant_key_makes_a_non_compliant_manifest(self) -> None:
        incomplete = {key: value for key, value in _record(second=2).items() if key != "compliant"}

        manifest = seal_run([_record(second=1), incomplete], run_id="run-1", signer=_signer())

        assert manifest["compliant"] is False

    def test_signature_value_verifies_against_the_manifest_without_its_signature(self) -> None:
        signer = _signer()

        manifest = seal_run([_record()], run_id="run-1", signer=signer, previous_manifest_hash="ab" * 32)

        payload = _signing_payload_ref(manifest)
        assert signer.verify(payload, manifest["signature"]["value"]) is True
        assert manifest["signature"]["value"] == _reference_signature(_KEY, payload)

    def test_any_manifest_signer_can_seal(self) -> None:
        signer: ManifestSigner = _PrefixSigner()

        manifest = seal_run([_record()], run_id="run-1", signer=signer)

        assert set(manifest["signature"]) == {"algorithm", "key_id", "value"}
        assert manifest["signature"]["algorithm"] == "TEST-SHA256"
        assert manifest["signature"]["key_id"] == "test-key"
        assert signer.verify(_signing_payload_ref(manifest), manifest["signature"]["value"]) is True

    def test_no_records_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            seal_run([], run_id="run-1", signer=_signer())

    @pytest.mark.parametrize("run_id", ["", "   "])
    def test_blank_run_id_raises_value_error(self, run_id: str) -> None:
        with pytest.raises(ValueError):
            seal_run([_record(run_id)], run_id=run_id, signer=_signer())

    def test_non_str_run_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            seal_run([{**_record(), "run_id": 5}], run_id=5, signer=_signer())  # type: ignore[arg-type]

    @pytest.mark.parametrize("other_run_id", ["run-2", None])
    def test_record_of_a_different_run_raises_value_error(self, other_run_id: str | None) -> None:
        records = [_record("run-1", 1), _record(other_run_id, 2)]

        with pytest.raises(ValueError):
            seal_run(records, run_id="run-1", signer=_signer())


class TestManifestHash:
    def test_is_the_sha256_of_the_full_canonical_manifest(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())

        assert manifest_hash(manifest) == hashlib.sha256(_canonical(manifest)).hexdigest()

    def test_does_not_depend_on_key_order_or_a_json_round_trip(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())
        reordered = dict(reversed(list(manifest.items())))

        assert manifest_hash(reordered) == manifest_hash(manifest)
        assert manifest_hash(json.loads(json.dumps(manifest))) == manifest_hash(manifest)

    def test_changing_the_signature_changes_the_hash(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())

        assert manifest_hash({**manifest, "signature": "tampered"}) != manifest_hash(manifest)


@_both_algorithms
class TestVerifyManifest:
    def test_manifest_verification_error_is_a_value_error(self) -> None:
        assert issubclass(ManifestVerificationError, ValueError)

    def test_untouched_manifest_and_records_verify(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        verify_manifest(manifest, records, signer=_signer())

    def test_record_order_does_not_matter(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        verify_manifest(manifest, list(reversed(records)), signer=_signer())

    def test_manifest_and_records_survive_a_json_round_trip(self) -> None:
        records = [_record(second=second, tenant_id="tenant-ü") for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer(), previous_manifest_hash="ab" * 32)

        verify_manifest(json.loads(json.dumps(manifest)), json.loads(json.dumps(records)), signer=_signer())

    def test_any_manifest_signer_can_verify(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_PrefixSigner())

        verify_manifest(manifest, records, signer=_PrefixSigner())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("compliant", True),
            ("hash_algorithm", "md5"),
            ("record_count", 4),
            ("record_hashes", []),
            ("run_id", "run-2"),
            ("sealed_at", "2000-01-01T00:00:00.000000Z"),
            ("previous_manifest_hash", "ab" * 32),
        ],
    )
    def test_changed_manifest_field_fails_the_signature_check(self, field: str, value: Any) -> None:
        records = [_record(second=1), _record(second=2, tenant_id=None), _record(second=3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        assert manifest[field] != value

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest({**manifest, field: value}, records, signer=_signer())

    def test_changed_signature_value_fails_the_signature_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        other_value = _signer().sign(b"something else")

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(
                {**manifest, "signature": {**manifest["signature"], "value": other_value}}, records, signer=_signer()
            )

    def test_non_ascii_signature_value_fails_the_signature_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        tampered = {**manifest, "signature": {**manifest["signature"], "value": "é" * 64}}

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(tampered, records, signer=_signer())

    @pytest.mark.parametrize(("field", "value"), [("key_id", "key-other"), ("algorithm", "HMAC-SHA512")])
    def test_changed_signature_block_field_fails_the_signature_check(self, field: str, value: str) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        assert manifest["signature"][field] != value
        tampered = {**manifest, "signature": {**manifest["signature"], field: value}}

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(tampered, records, signer=_signer())

    def test_removed_signature_fails_the_signature_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(_unsigned(manifest), records, signer=_signer())

    def test_extra_key_in_the_signature_block_fails_the_signature_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        tampered = {**manifest, "signature": {**manifest["signature"], "note": "never signed"}}

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(tampered, records, signer=_signer())

    def test_unhashable_signature_key_id_fails_without_crashing(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        tampered = {**manifest, "signature": {**manifest["signature"], "key_id": []}}

        with pytest.raises(ManifestVerificationError):
            verify_manifest(tampered, records, signer=_signer())

    def test_altered_record_fails_the_record_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        altered = [records[0], {**records[1], "tenant_id": "tenant-other"}, records[2]]

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_manifest(manifest, altered, signer=_signer())

    def test_removed_record_fails_the_record_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_manifest(manifest, records[:-1], signer=_signer())

    def test_added_record_fails_the_record_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_manifest(manifest, [*records, _record(second=9)], signer=_signer())

    def test_added_record_beyond_the_seal_names_the_run_and_the_count(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_manifest(manifest, [*records, _record(second=9)], signer=_signer())

        message = str(excinfo.value)
        assert "run-1" in message
        assert "has 1 record(s) beyond its seal" in message
        assert "do not match the record_hashes" not in message

    def test_added_record_together_with_an_edited_record_fails_the_generic_check(self) -> None:
        """Count grows by one, but an edit replaces a sealed hash: not a pure late append."""
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        mixed = [{**records[0], "tenant_id": "tenant-other"}, records[1], records[2], _record(second=9)]

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_manifest(manifest, mixed, signer=_signer())

        message = str(excinfo.value)
        assert "do not match the record_hashes" in message
        assert "beyond its seal" not in message

    def test_repeated_record_fails_the_record_check(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_manifest(manifest, [*records, records[0]], signer=_signer())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("manifest_version", 3),
            ("hash_algorithm", "md5"),
            ("record_count", 4),
            ("run_id", 5),
            ("compliant", False),
            ("compliant", 1),
        ],
    )
    def test_correctly_signed_manifest_with_an_unacceptable_field_fails(self, field: str, value: Any) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        resigned = _resigned(manifest, **{field: value})

        with pytest.raises(ManifestVerificationError, match=field):
            verify_manifest(resigned, records, signer=_signer())

    @pytest.mark.parametrize("record_hashes", [[[]], [1]], ids=["unhashable", "hashable-non-string"])
    def test_correctly_signed_manifest_with_a_non_string_record_hash_fails_without_crashing(
        self, record_hashes: list[Any]
    ) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        resigned = _resigned(manifest, record_hashes=record_hashes, record_count=len(record_hashes))

        with pytest.raises(ManifestVerificationError):
            verify_manifest(resigned, records, signer=_signer())

    def test_correctly_signed_compliant_manifest_over_a_deny_record_fails(self) -> None:
        records = [_record(second=1), _record(second=2, tenant_id=None), _record(second=3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        assert manifest["compliant"] is False

        with pytest.raises(ManifestVerificationError, match="compliant"):
            verify_manifest(_resigned(manifest, compliant=True), records, signer=_signer())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("record_count", True),
            ("record_count", 1.0),
            ("manifest_version", True),
            ("manifest_version", 2.0),
        ],
    )
    def test_correctly_signed_manifest_with_a_non_int_number_fails(self, field: str, value: Any) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        # Where the value equals the genuine one (1 == True == 1.0, 2 == 2.0) only its type is wrong. The
        # manifest_version True case (True != 2) only guards that a bool is not accepted as a version.
        if (field, value) != ("manifest_version", True):
            assert manifest[field] == value

        with pytest.raises(ManifestVerificationError, match=field):
            verify_manifest(_resigned(manifest, **{field: value}), records, signer=_signer())

    def test_previous_signers_verifies_a_manifest_signed_by_an_old_key(self) -> None:
        records = [_record(second=second) for second in range(3)]
        old_signer = _signer()
        manifest = seal_run(records, run_id="run-1", signer=old_signer)
        new_signer = _signer(_OTHER_KEY, "key-2")

        verify_manifest(manifest, records, signer=new_signer, previous_signers=[old_signer])

    def test_previous_signers_accepts_any_manifest_signer_protocol_implementation(self) -> None:
        records = [_record()]
        old_signer = _PrefixSigner()
        manifest = seal_run(records, run_id="run-1", signer=old_signer)
        new_signer = _signer(_OTHER_KEY, "key-2")

        verify_manifest(manifest, records, signer=new_signer, previous_signers=[old_signer])

    def test_unknown_key_id_fails_the_signature_check_and_names_it(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer(b"u" * 32, "key-unknown"))

        with pytest.raises(ManifestVerificationError, match="signature") as excinfo:
            verify_manifest(manifest, records, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()])

        assert "key-unknown" in str(excinfo.value)

    def test_unknown_key_id_with_previous_signers_names_no_known_key(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer(b"u" * 32, "key-unknown"))

        with pytest.raises(ManifestVerificationError, match="signature") as excinfo:
            verify_manifest(manifest, records, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()])

        message = str(excinfo.value)
        assert "key-unknown" in message
        assert "is not the signer's" not in message

    def test_unknown_key_id_without_previous_signers_keeps_the_exact_wording(self) -> None:
        records = [_record(second=second) for second in range(3)]
        manifest = seal_run(records, run_id="run-1", signer=_signer(b"u" * 32, "key-unknown"))

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_manifest(manifest, records, signer=_signer())

        assert str(excinfo.value) == "signature key_id 'key-unknown' is not the signer's 'key-1'"

    @pytest.mark.parametrize(
        "previous_signers",
        [[_signer()], [_signer(key_id="key-3"), _signer(_OTHER_KEY, "key-3")]],
        ids=["signer-and-previous", "two-previous"],
    )
    def test_previous_signers_with_a_duplicate_key_id_is_a_value_error(
        self, previous_signers: list[ManifestSigner]
    ) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ValueError) as excinfo:
            verify_manifest(manifest, records, signer=_signer(_OTHER_KEY), previous_signers=previous_signers)

        assert not isinstance(excinfo.value, ManifestVerificationError)

    def test_a_rotation_entry_is_not_a_run_manifest_and_fails_without_crashing(self) -> None:
        with pytest.raises(ManifestVerificationError, match="unsupported hash_algorithm"):
            verify_manifest(_rotation_entry(_signer(), None), [_record()], signer=_signer())


@_both_algorithms
class TestSealNdjsonRuns:
    def test_seals_every_run_of_the_audit_file(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2), _record("run-a", 3)])

        manifests = seal_ndjson_runs(audit_path, tmp_path / "manifests.ndjson", signer=_signer())

        assert [(manifest["run_id"], manifest["record_count"]) for manifest in manifests] == [
            ("run-a", 2),
            ("run-b", 1),
        ]

    def test_seal_refuses_a_line_over_the_cap_and_changes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _cap(monkeypatch)
        _write_records(audit_path, [_record("run-d", second) for second in range(40)])
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert _snapshot(tmp_path) == before

    def test_a_refused_first_seal_does_not_create_the_manifest_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _cap(monkeypatch)
        _write_records(audit_path, [_record("run-a", second) for second in range(40)])

        with pytest.raises(ValueError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert not manifest_path.exists()

    def test_runs_are_sealed_in_order_of_first_appearance_in_the_audit_file(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_record("run-b", 5), _record("run-a", 1), _record("run-b", 6)])

        manifests = seal_ndjson_runs(audit_path, tmp_path / "manifests.ndjson", signer=_signer())

        assert [manifest["run_id"] for manifest in manifests] == ["run-b", "run-a"]

    @pytest.mark.parametrize("event_time", [{}, {"event_time": 5}], ids=["missing", "not-a-string"])
    def test_record_without_a_usable_event_time_does_not_block_sealing(
        self, tmp_path: Path, event_time: dict[str, Any]
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        record = {key: value for key, value in _record("run-a", 1).items() if key != "event_time"}
        _write_records(audit_path, [{**record, **event_time}, _record("run-b", 2)])

        manifests = seal_ndjson_runs(audit_path, tmp_path / "manifests.ndjson", signer=_signer())

        assert [manifest["run_id"] for manifest in manifests] == ["run-a", "run-b"]

    @pytest.mark.parametrize(
        ("run_id", "drop_key"),
        [(None, False), ("", False), ("   ", False), (None, True)],
        ids=["null", "empty", "blank", "absent"],
    )
    def test_records_without_a_run_id_are_ignored(self, tmp_path: Path, run_id: str | None, drop_key: bool) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"

        def unusable(second: int) -> dict[str, Any]:
            record = _record(run_id, second)
            if drop_key:
                del record["run_id"]
            return record

        _write_records(audit_path, [unusable(1), _record("run-a", 2), unusable(3)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert [(manifest["run_id"], manifest["record_count"]) for manifest in manifests] == [("run-a", 1)]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_run_with_a_deny_record_is_sealed_as_non_compliant(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2), _record("run-b", 3, tenant_id=None)])

        manifests = seal_ndjson_runs(audit_path, tmp_path / "manifests.ndjson", signer=_signer())

        assert [(manifest["run_id"], manifest["compliant"]) for manifest in manifests] == [
            ("run-a", True),
            ("run-b", False),
        ]

    def test_appends_one_json_line_per_manifest_and_accepts_str_paths(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])

        manifests = seal_ndjson_runs(str(audit_path), str(manifest_path), signer=_signer())

        assert len(manifests) == 2
        assert _read_lines(manifest_path) == manifests
        verify_ndjson_log(str(audit_path), str(manifest_path), signer=_signer())

    def test_new_manifest_file_has_owner_only_permissions(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])

        seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600

    def test_first_manifest_has_no_previous_hash_and_each_later_one_chains(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2), _record("run-c", 3)])

        manifests = seal_ndjson_runs(audit_path, tmp_path / "manifests.ndjson", signer=_signer())

        assert len(manifests) == 3
        assert manifests[0]["previous_manifest_hash"] is None
        assert manifests[1]["previous_manifest_hash"] == manifest_hash(manifests[0])
        assert manifests[2]["previous_manifest_hash"] == manifest_hash(manifests[1])

    def test_chain_continues_across_calls_and_sealed_runs_are_skipped(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])
        first = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        _write_records(audit_path, [_record("run-b", 2)])

        second = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert [manifest["run_id"] for manifest in second] == ["run-b"]
        assert second[0]["previous_manifest_hash"] == manifest_hash(first[0])
        assert second[0]["previous_manifest_hash"] == manifest_hash(_read_lines(manifest_path)[0])
        assert _read_lines(manifest_path) == [*first, *second]

    def test_nothing_new_returns_empty_and_leaves_the_manifest_file_unchanged(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        before = manifest_path.read_bytes()

        assert seal_ndjson_runs(audit_path, manifest_path, signer=_signer()) == []
        assert manifest_path.read_bytes() == before

    def test_explicit_run_id_seals_only_that_run(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2), _record("run-c", 3)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-b")

        assert [manifest["run_id"] for manifest in manifests] == ["run-b"]
        assert manifests[0]["previous_manifest_hash"] is None
        assert _read_lines(manifest_path) == manifests

    def test_explicit_run_id_without_records_raises_run_not_pending_error(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])

        with pytest.raises(RunNotPendingError) as excinfo:
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-missing")

        assert not isinstance(excinfo.value, RunAlreadySealedError)

    def test_explicit_run_id_already_sealed_raises_run_already_sealed_error(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-a")
        before = manifest_path.read_bytes()

        with pytest.raises(RunAlreadySealedError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-a")

        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("run_id", ["", "   ", 123], ids=["empty", "blank", "non-str"])
    def test_unusable_explicit_run_id_raises_value_error_and_creates_no_manifest_log(
        self, tmp_path: Path, run_id: Any
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])

        with pytest.raises(ValueError, match="run_id") as excinfo:
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id=run_id)

        assert not isinstance(excinfo.value, RunNotPendingError)
        assert not manifest_path.exists()

    @pytest.mark.parametrize("forged_run_id", ["run-d", "run-never-written"])
    def test_forged_unsigned_manifest_line_stops_sealing(self, tmp_path: Path, forged_run_id: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        _write_records(manifest_path, [{"run_id": forged_run_id}])
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert manifest_path.read_bytes() == before

    def test_edited_audit_line_of_a_sealed_run_does_not_stop_sealing(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        records = _read_lines(audit_path)
        assert records[0]["run_id"] == "run-a"
        records[0]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, [*records, _record("run-d", 7)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert [(manifest["run_id"], manifest["previous_manifest_hash"]) for manifest in manifests] == [("run-d", head)]
        with pytest.raises(ManifestVerificationError, match="record"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_signing_failure_writes_nothing_and_a_retry_seals_every_run(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _write_records(audit_path, [_record("run-d", 7), _record("run-e", 8), _record("run-f", 9)])
        before = manifest_path.read_bytes()

        def fail_on_run_f(payload: bytes) -> None:
            # Fails on run-f, which only sealing meets.
            if json.loads(payload).get("run_id") == "run-f":
                raise RuntimeError("signing boom")

        failing = _HookedSigner(_signer(), on_sign=fail_on_run_f)

        with pytest.raises(RuntimeError, match="signing boom"):
            seal_ndjson_runs(audit_path, manifest_path, signer=failing, expected_head=head)

        assert manifest_path.read_bytes() == before
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head)
        assert [manifest["run_id"] for manifest in manifests] == ["run-d", "run-e", "run-f"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    # Sealing and rotation share one append path, so its tests run against both (`_pending_write`).
    @pytest.mark.parametrize("writer", _WRITERS)
    def test_failed_append_is_rolled_back_and_a_retry_appends_every_pending_line(
        self, tmp_path: Path, writer: str
    ) -> None:
        manifest_path, write, verify = _pending_write(tmp_path, writer)
        before = manifest_path.read_bytes()
        real_write = os.write

        def disk_is_full(fd: int, data: bytes | memoryview) -> int:
            # Every appended line carries the chain link.
            if b"previous_manifest_hash" in bytes(data):
                raise OSError(errno.ENOSPC, "disk full")
            return real_write(fd, data)

        with patch("os.write", side_effect=disk_is_full):
            with pytest.raises(OSError):
                write()

        assert manifest_path.read_bytes() == before
        assert [line.get("run_id") for line in write()] == _PENDING_RUN_IDS[writer]
        verify()

    @pytest.mark.parametrize("writer", _WRITERS)
    def test_failed_append_with_a_genuine_short_write_is_rolled_back_and_a_retry_appends_every_pending_line(
        self, tmp_path: Path, writer: str
    ) -> None:
        """Unlike the raise-based case above, os.write here really writes a truncated prefix and returns the
        short count: bytes land on disk before _append_records raises, so only the truncate saves the log."""
        manifest_path, write, verify = _pending_write(tmp_path, writer)
        before = manifest_path.read_bytes()
        real_write = os.write

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                write()

        assert manifest_path.read_bytes() == before
        assert [line.get("run_id") for line in write()] == _PENDING_RUN_IDS[writer]
        verify()

    @pytest.mark.parametrize("fcntl_available", [True, False], ids=["fcntl", "no-fcntl"])
    def test_failed_first_append_leaves_the_new_manifest_log_as_an_empty_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fcntl_available: bool
    ) -> None:
        if fcntl_available:
            pytest.importorskip("fcntl")
        else:
            monkeypatch.setitem(sys.modules, "fcntl", None)
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])
        real_write = os.write

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert manifest_path.read_bytes() == b""

    @pytest.mark.parametrize("writer", _WRITERS)
    def test_failed_append_fsyncs_the_manifest_log_and_its_directory_after_the_rollback_truncate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
    ) -> None:
        """The rollback truncate must be durable: a crash after it must not resurrect lines reported as failed."""
        manifest_path, write, _ = _pending_write(tmp_path, writer)
        real_write = os.write
        real_truncate = os.truncate
        real_fsync = os.fsync
        events: list[str] = []

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        def spy_truncate(path: str | Path, length: int) -> None:
            events.append("truncate")
            real_truncate(path, length)

        def spy_fsync(fd: int) -> None:
            try:
                if os.path.samestat(os.fstat(fd), manifest_path.stat()):
                    events.append("fsync-manifest")
            except FileNotFoundError:
                pass
            try:
                if os.path.samestat(os.fstat(fd), manifest_path.parent.stat()):
                    events.append("fsync-manifest-dir")
            except FileNotFoundError:
                pass
            real_fsync(fd)

        monkeypatch.setattr(os, "truncate", spy_truncate)
        monkeypatch.setattr(os, "fsync", spy_fsync)

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                write()

        assert "truncate" in events
        assert "fsync-manifest" in events
        assert "fsync-manifest-dir" in events
        assert events.index("truncate") < events.index("fsync-manifest")
        assert events.index("truncate") < events.index("fsync-manifest-dir")

    @pytest.mark.parametrize("writer", _WRITERS)
    def test_append_fsyncs_the_manifest_log_and_its_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
    ) -> None:
        manifest_path, write, _ = _pending_write(tmp_path, writer)
        fsynced: list[Path] = []
        real_fsync = _verify._fsync

        def spy(path: str | Path) -> None:
            fsynced.append(Path(path))
            real_fsync(path)

        _patch_bindings(monkeypatch, "_fsync", spy)

        write()

        assert manifest_path in fsynced
        assert manifest_path.parent in fsynced

    @pytest.mark.parametrize("writer", _WRITERS)
    def test_append_fsync_failure_rolls_back_the_new_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
    ) -> None:
        manifest_path, write, _ = _pending_write(tmp_path, writer)
        before = _snapshot(tmp_path)
        real_fsync = _verify._fsync

        def fsync_boom(path: str | Path) -> None:
            if Path(path) == manifest_path:
                raise OSError(errno.EIO, "fsync failed")
            real_fsync(path)

        _patch_bindings(monkeypatch, "_fsync", fsync_boom)

        with pytest.raises(OSError, match="fsync failed"):
            write()

        assert _snapshot(tmp_path) == before

    def test_audit_file_is_fsynced_before_the_manifest_log_is_appended(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        events: list[str] = []
        real_fsync = os.fsync

        def spy_fsync(fd: int) -> None:
            try:
                is_audit = os.path.samestat(os.fstat(fd), audit_path.stat())
            except FileNotFoundError:
                is_audit = False
            if is_audit:
                events.append("fsync-audit")
            real_fsync(fd)

        def spy_append(*args: Any, **kwargs: Any) -> None:
            events.append("append")
            _append_records(*args, **kwargs)

        monkeypatch.setattr(os, "fsync", spy_fsync)
        _patch_bindings(monkeypatch, "_append_records", spy_append)

        seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert "fsync-audit" in events
        assert events.index("fsync-audit") < events.index("append")

    @pytest.mark.parametrize("file_name", ["manifests.ndjson", "audit.ndjson"])
    def test_unterminated_final_line_fails_and_leaves_both_files_untouched(
        self, tmp_path: Path, file_name: str
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        path = tmp_path / file_name
        path.write_bytes(path.read_bytes().removesuffix(b"\n"))
        audit_before, manifest_before = audit_path.read_bytes(), manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match="newline"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        with pytest.raises(ManifestVerificationError, match="newline"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert audit_path.read_bytes() == audit_before
        assert manifest_path.read_bytes() == manifest_before

    @pytest.mark.parametrize("writer", _WRITERS)
    def test_appending_holds_an_exclusive_lock_on_the_manifest_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        manifest_path, write, _ = _pending_write(tmp_path, writer)
        refused: list[bool] = []
        appends: list[bool] = []

        def probe() -> bool:
            # A shared request is refused only by an exclusive holder.
            fd = os.open(manifest_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                return False
            except BlockingIOError:
                return True
            finally:
                os.close(fd)

        def probe_signing(payload: bytes) -> None:
            refused.append(probe())

        def spy(*args: Any, **kwargs: Any) -> None:
            refused.append(probe())
            appends.append(True)
            _append_records(*args, **kwargs)

        _patch_bindings(monkeypatch, "_append_records", spy)

        # The wrapper probes verify as well as sign: its verify does not call its sign.
        appended = write(lambda signer: _HookedSigner(signer, on_sign=probe_signing, on_verify=probe_signing))

        assert [line.get("run_id") for line in appended] == _PENDING_RUN_IDS[writer]
        assert appends == [True]
        assert refused
        assert all(refused)
        fd = os.open(manifest_path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    def test_sealing_and_verifying_work_where_fcntl_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "fcntl", None)
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert [manifest["run_id"] for manifest in manifests] == ["run-a", "run-b"]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == manifest_hash(manifests[-1])

    @pytest.mark.parametrize("writer", _WRITERS)
    def test_appending_fails_when_the_exclusive_lock_is_unsupported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        manifest_path, write, _ = _pending_write(tmp_path, writer)
        before = manifest_path.read_bytes()
        monkeypatch.setattr(fcntl, "flock", _flock_unsupported)

        with pytest.raises(OSError) as excinfo:
            write()

        assert excinfo.value.errno == errno.ENOLCK
        assert manifest_path.read_bytes() == before

    def test_a_sealer_locking_a_replaced_manifest_relocks_and_seals_in_the_new_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        archive, locked, new_ino = _swap_on_first_flock(monkeypatch, manifest_path, tmp_path)
        archived = archive.read_bytes()
        _write_records(audit_path, [_record("run-d", 7)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert [manifest["run_id"] for manifest in manifests] == ["run-d"]
        assert locked[-1] == new_ino
        assert [line["run_id"] for line in _read_lines(manifest_path)][-1] == "run-d"
        assert archive.read_bytes() == archived

    def test_missing_audit_file_still_fails_sealing(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        audit_path.unlink()
        before = manifest_path.read_bytes()

        with pytest.raises(FileNotFoundError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert manifest_path.read_bytes() == before

    def test_nothing_pending_on_a_fresh_manifest_path_leaves_an_empty_log(self, tmp_path: Path) -> None:
        pytest.importorskip("fcntl")  # The lock creates the log; where fcntl is missing there is none.
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record(None, 1), _record(None, 2)])

        assert seal_ndjson_runs(audit_path, manifest_path, signer=_signer()) == []

        assert manifest_path.read_bytes() == b""
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) is None

    def test_seal_after_rotation_chains_onto_the_rotation_entry_and_signs_with_the_new_key(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        head = manifest_hash(_rotate(manifest_path, new_signer, old_signer))
        _write_records(audit_path, [_record("run-d", 7)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert [manifest["run_id"] for manifest in manifests] == ["run-d"]
        assert manifests[0]["previous_manifest_hash"] == head
        assert manifests[0]["signature"]["key_id"] == "key-2"

    @pytest.mark.parametrize(
        "previous_signers",
        [[_signer()], [_signer(key_id="key-3"), _signer(_OTHER_KEY, "key-3")]],
        ids=["signer-and-previous", "two-previous"],
    )
    def test_previous_signers_with_a_duplicate_key_id_is_a_value_error(
        self, tmp_path: Path, previous_signers: list[ManifestSigner]
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = manifest_path.read_bytes()

        with pytest.raises(ValueError) as excinfo:
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(_OTHER_KEY), previous_signers=previous_signers)

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert manifest_path.read_bytes() == before


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
        assert run_manifest_module._is_run_sealed_unverified(manifest_path, "run-crash", index_path) is True
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

    def test_no_wal_or_shm_files_are_left_next_to_the_index(self, tmp_path: Path) -> None:
        _indexed_log(tmp_path / "live")

        names = {path.name for path in (tmp_path / "live").iterdir()}

        assert names == {"audit.ndjson", "manifests.ndjson", _INDEX}

    # Audit side: the index also remembers how far the audit file was scanned and where each unsealed run starts.

    @staticmethod
    def _count_audit_parses(monkeypatch: pytest.MonkeyPatch, audit_path: Path) -> list[str]:
        """Every audit line decoded from now on (the decode of any other file is not counted)."""
        seen: list[str] = []
        real = _verify._decode_line

        def counting(where: str, line: bytes) -> Any:
            if where.startswith(str(audit_path)):
                seen.append(where)
            return real(where, line)

        _patch_bindings(monkeypatch, "_decode_line", counting)
        return seen

    @staticmethod
    def _stable(manifests: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in m.items() if k not in ("sealed_at", "signature")} for m in manifests]

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

        def chainless(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{k: v for k, v in m.items() if k != "previous_manifest_hash"} for m in self._stable(manifests)]

        assert chainless(fast) == chainless(full)
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

    # Sealed-run lookup through the index (workers have no signer: no signature checks).

    @staticmethod
    def _lookups(manifest_path: Path, index_path: Path | None, run_ids: Iterable[str]) -> dict[str, bool]:
        return {
            run_id: run_manifest_module._is_run_sealed_unverified(manifest_path, run_id, index_path)
            for run_id in run_ids
        }

    @staticmethod
    def _seal_after_checkpoint(audit_path: Path, manifest_path: Path, run_id: str = "run-tail") -> None:
        """A seal written without the index, so it lies past the checkpoint end offset."""
        _write_records(audit_path, [_record(run_id, 50)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id=run_id)

    def test_the_lookup_finds_runs_sealed_before_and_after_the_checkpoint_and_not_unsealed_ones(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        self._seal_after_checkpoint(audit_path, manifest_path)
        _write_records(audit_path, [_record("run-open", 60)])

        found = self._lookups(manifest_path, index_path, ["run-0", "run-3", "run-tail", "run-open", "run-nowhere"])

        assert found == {"run-0": True, "run-3": True, "run-tail": True, "run-open": False, "run-nowhere": False}

    def test_the_lookup_reads_the_same_manifest_bytes_whatever_the_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        costs = []
        for prior in (2, 8):
            audit_path, manifest_path, index_path = _indexed_log(tmp_path / f"p{prior}", prior)
            self._seal_after_checkpoint(audit_path, manifest_path)
            read: list[int] = []
            real_open = open

            def spy_open(
                file: Any, mode: str = "r", *args: Any, _p: Path = manifest_path, _r: list[int] = read, **kw: Any
            ) -> Any:
                handle = real_open(file, mode, *args, **kw)
                return _ReadSpy(handle, _r) if mode == "rb" and str(file) == str(_p) else handle

            _patch_bindings(monkeypatch, "open", spy_open)
            found = self._lookups(manifest_path, index_path, ["run-0", "run-tail", "run-nowhere"])
            _unpatch_bindings(monkeypatch, "open")
            assert found == {"run-0": True, "run-tail": True, "run-nowhere": False}
            costs.append(sum(read))

        assert costs[0] == costs[1]

    @pytest.mark.parametrize("case", list(_INDEX_FALLBACKS))
    def test_a_missing_garbage_or_stale_index_falls_back_to_the_full_scan_answer(
        self, tmp_path: Path, case: str
    ) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        _INDEX_FALLBACKS[case](audit_path, manifest_path, index_path)
        run_ids = ["run-0", "run-2", "run-3", "xun-3", "run-nowhere"]
        expected = self._lookups(manifest_path, None, run_ids)

        found = self._lookups(manifest_path, index_path, run_ids)

        assert found == expected
        assert expected["run-0"] is True

    def test_a_hint_that_points_at_a_line_not_naming_the_run_falls_back_to_the_full_scan(self, tmp_path: Path) -> None:
        import sqlite3

        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        connection = sqlite3.connect(index_path)
        try:
            connection.execute("UPDATE hint SET line_start = 0 WHERE run_id = ?", ("run-3",))
            connection.execute("INSERT OR REPLACE INTO hint (run_id, line_start) VALUES (?, 0)", ("run-ghost",))
            connection.commit()
        finally:
            connection.close()

        found = self._lookups(manifest_path, index_path, ["run-0", "run-3", "run-ghost"])

        assert found == {"run-0": True, "run-3": True, "run-ghost": False}

    @pytest.mark.parametrize("case", ["missing", "garbage", "current"])
    def test_the_lookup_never_creates_or_modifies_the_index(self, tmp_path: Path, case: str) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        if case == "missing":
            index_path.unlink()
        elif case == "garbage":
            index_path.write_bytes(b"not a database" * 100)
        before = _snapshot(tmp_path / "live")

        self._lookups(manifest_path, index_path, ["run-0", "run-nowhere"])

        assert _snapshot(tmp_path / "live") == before

    def test_the_lookup_never_waits_on_a_writer_holding_the_index(self, tmp_path: Path) -> None:
        import sqlite3
        import threading

        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live")
        expected = self._lookups(manifest_path, None, ["run-0", "run-nowhere"])
        found: list[dict[str, bool]] = []
        writer = sqlite3.connect(index_path, isolation_level=None)
        try:
            writer.execute("BEGIN EXCLUSIVE")
            thread = threading.Thread(
                target=lambda: found.append(self._lookups(manifest_path, index_path, ["run-0", "run-nowhere"]))
            )
            thread.start()
            thread.join(timeout=1.5)
            finished_while_held = not thread.is_alive()
            writer.execute("ROLLBACK")
        finally:
            writer.close()
        thread.join()

        assert finished_while_held
        assert found == [expected]

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

    def test_a_seal_written_without_the_index_between_indexed_seals_stays_found_and_refused(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, index_path = _indexed_log(tmp_path / "live", 2)
        self._seal_after_checkpoint(audit_path, manifest_path, "run-tail")
        _write_records(audit_path, [_record("run-c", 70)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-c", seal_index_path=index_path)

        found = self._lookups(manifest_path, index_path, ["run-0", "run-tail", "run-c", "run-nowhere"])

        assert found == {"run-0": True, "run-tail": True, "run-c": True, "run-nowhere": False}
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
    def test_a_checkpoint_body_that_is_not_text_falls_back_in_the_lookup_and_the_seal(
        self, tmp_path: Path, kind: str
    ) -> None:
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
        run_ids = ["run-0", "run-3", "run-nowhere"]

        assert self._lookups(manifest_path, index_path, run_ids) == {"run-0": True, "run-3": True, "run-nowhere": False}
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

    def test_the_index_file_is_created_with_owner_only_permissions(self, tmp_path: Path) -> None:
        _, _, index_path = _indexed_log(tmp_path / "live", 1)

        assert stat.S_IMODE(index_path.stat().st_mode) == 0o600

    def test_rewrite_without_refuses_to_keep_an_oversized_line_and_leaves_the_file_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _cap(monkeypatch)
        path = tmp_path / "audit.ndjson"
        path.write_bytes(b'{"a": 1}\n' + _oversized_line() + b'\n{"b": 2}\n')
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError):
            _quarantine_module._rewrite_without(path, {3})

        assert _snapshot(tmp_path) == before

    def test_rewrite_without_still_drops_an_oversized_line_that_is_listed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _cap(monkeypatch)
        path = tmp_path / "audit.ndjson"
        path.write_bytes(b'{"a": 1}\n' + _oversized_line() + b'\n{"b": 2}\n')

        _quarantine_module._rewrite_without(path, {2})

        assert path.read_bytes() == b'{"a": 1}\n{"b": 2}\n'

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


class TestCheckRunAgainstSeal:
    """_check_run_against_seal is what AuditExtender.on_run_complete calls when seal_ndjson_runs raises
    RunAlreadySealedError, to tell an untouched re-run from a stray record written after the seal. It must
    verify the manifest's signature and fields, not just compare digests against an unverified read."""

    def test_untouched_sealed_run_does_not_raise(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-1", 1)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        # must not raise
        run_manifest_module._check_run_against_seal(audit_path, manifest_path, "run-1", signer=_signer())

    def test_the_manifest_is_shared_locked_while_the_audit_file_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        locked_during_read = _locked_during_digest(monkeypatch, manifest_path)

        run_manifest_module._check_run_against_seal(audit_path, manifest_path, "run-a", signer=_signer())

        assert locked_during_read == [True]

    def test_a_manifest_edited_to_hide_a_stray_record_fails_the_signature_check(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-1", 1)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        _write_records(audit_path, [_record("run-1", 2)])  # a stray record, written after the seal
        stray_hash = _sha256(audit_path.read_bytes().splitlines()[-1])
        manifest = _read_lines(manifest_path)[0]
        # The digest now covers the stray too, but the signature block is untouched: it no longer matches.
        tampered = {
            **manifest,
            "record_count": manifest["record_count"] + 1,
            "record_hashes": sorted([*manifest["record_hashes"], stray_hash]),
        }
        _rewrite_lines(manifest_path, [tampered])

        with pytest.raises(ManifestVerificationError, match="signature"):
            run_manifest_module._check_run_against_seal(audit_path, manifest_path, "run-1", signer=_signer())


class TestCheckRunAgainstArchivedSeal:
    """A run sealed in an archived segment is checked against the archived manifest and audit file."""

    def test_a_run_sealed_in_an_archived_segment_passes_when_the_archived_pair_matches(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        _rotate_segment(audit_path, manifest_path)

        run_manifest_module._check_run_against_seal(audit_path, manifest_path, "run-a", signer=_signer())

    def test_a_run_sealed_in_an_archived_segment_fails_when_its_archived_records_were_tampered(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        _rotate_segment(audit_path, manifest_path)
        archived_audit = _archive(audit_path)
        lines = _read_lines(archived_audit)
        lines[0]["tenant_id"] = "tampered"
        _rewrite_lines(archived_audit, lines)

        with pytest.raises(ManifestVerificationError) as excinfo:
            run_manifest_module._check_run_against_seal(audit_path, manifest_path, "run-a", signer=_signer())

        assert "has no manifest" not in str(excinfo.value)


def _no_log_id(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    audit_path, manifest_path = _pending_log(directory)
    return audit_path, manifest_path, {"log_id": None}


def _blank_log_id(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    audit_path, manifest_path = _pending_log(directory)
    return audit_path, manifest_path, {"log_id": "   "}


def _missing_manifest_log(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    _write_records(directory / "audit.ndjson", [_record("run-p", 1)])
    return directory / "audit.ndjson", directory / "manifests.ndjson", {}


def _empty_manifest_log(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    audit_path, manifest_path, kwargs = _missing_manifest_log(directory)
    manifest_path.touch()
    return audit_path, manifest_path, kwargs


def _legacy_log(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    audit_path, manifest_path = _sealed_log(directory)
    return audit_path, manifest_path, {}


def _symlinked_audit(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    audit_path, manifest_path = _pending_log(directory)
    real = directory / "real-audit.ndjson"
    audit_path.rename(real)
    audit_path.symlink_to(real)
    return audit_path, manifest_path, {}


def _symlinked_manifest(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    audit_path, manifest_path = _pending_log(directory)
    real = directory / "real-manifests.ndjson"
    manifest_path.rename(real)
    manifest_path.symlink_to(real)
    return audit_path, manifest_path, {}


_UNROTATABLE = {
    "log-id-none": _no_log_id,
    "log-id-blank": _blank_log_id,
    "manifest-missing": _missing_manifest_log,
    "manifest-empty": _empty_manifest_log,
    "legacy-log-without-genesis": _legacy_log,
    "audit-path-is-a-symlink": _symlinked_audit,
    "manifest-path-is-a-symlink": _symlinked_manifest,
}


def _tampered_record(audit_path: Path, manifest_path: Path) -> dict[str, Any]:
    lines = _read_lines(audit_path)
    lines[0]["tenant_id"] = "tampered"
    _rewrite_lines(audit_path, lines)
    return {}


def _stray_record_of_a_sealed_run(audit_path: Path, manifest_path: Path) -> dict[str, Any]:
    _write_records(audit_path, [_record("run-a", 9)])
    return {}


_UNVERIFIABLE = {
    "tampered-record": _tampered_record,
    "stray-record-of-a-sealed-run": _stray_record_of_a_sealed_run,
    "wrong-expected-head": lambda a, m: {"expected_head": _sha256(b"not the head")},
    "unknown-anchored-head": lambda a, m: {"anchored_heads": [_sha256(b"unknown")]},
    "signer-not-current": lambda a, m: {"signer": _signer(_OTHER_KEY, "key-2")},
}


@_both_algorithms
class TestSegmentRotation:
    """rotate_ndjson_segment: archive the outgoing audit and manifest pair, start a new segment chained onto it."""

    def test_the_outgoing_pair_is_archived_byte_for_byte_next_to_the_live_files(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        audit_before, manifest_before = audit_path.read_bytes(), manifest_path.read_bytes()

        _rotate_segment(audit_path, manifest_path)

        assert _archive(audit_path).name == "audit.ndjson.000001"
        assert _archive(manifest_path).name == "manifests.ndjson.000001"
        assert _archive(audit_path).read_bytes() == audit_before
        assert _archive(manifest_path).read_bytes() == manifest_before
        assert audit_path.exists() and manifest_path.exists()

    def test_a_second_rotation_archives_under_the_next_number_and_keeps_the_first_archive(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        _rotate_segment(audit_path, manifest_path)
        first = (_archive(audit_path).read_bytes(), _archive(manifest_path).read_bytes())
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-p")
        audit_before, manifest_before = audit_path.read_bytes(), manifest_path.read_bytes()

        _rotate_segment(audit_path, manifest_path)

        assert (_archive(audit_path).read_bytes(), _archive(manifest_path).read_bytes()) == first
        assert _archive(audit_path, 2).read_bytes() == audit_before
        assert _archive(manifest_path, 2).read_bytes() == manifest_before

    def test_the_next_number_is_one_above_the_highest_archived_manifest(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        _archive(manifest_path, 4).write_bytes(b"retained")

        _rotate_segment(audit_path, manifest_path)

        assert _archives(tmp_path) == ["audit.ndjson.000005", "manifests.ndjson.000004", "manifests.ndjson.000005"]

    def test_the_new_manifest_log_is_one_genesis_chained_onto_the_outgoing_head(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        old_head = manifest_hash(_read_lines(manifest_path)[-1])

        genesis = _rotate_segment(audit_path, manifest_path)

        assert _read_lines(manifest_path) == [genesis]
        assert set(genesis) == _EXPECTED_GENESIS_KEYS
        assert genesis["kind"] == "genesis"
        assert genesis["log_id"] == "log-a"
        assert genesis["previous_manifest_hash"] == old_head
        assert genesis["signature"]["key_id"] == _signer().key_id

    def test_the_new_pair_and_the_archived_pair_each_verify_on_their_own(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        old_head = manifest_hash(_read_lines(manifest_path)[-1])

        genesis = _rotate_segment(audit_path, manifest_path)

        new_head = verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a")
        assert new_head == manifest_hash(genesis)
        archived = verify_ndjson_log(_archive(audit_path), _archive(manifest_path), signer=_signer(), log_id="log-a")
        assert archived == old_head

    def test_only_the_pending_attributed_lines_are_carried_forward_byte_for_byte_in_order(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        lines = audit_path.read_bytes().splitlines(keepends=True)
        pending = [line for line in lines if json.loads(line)["run_id"] == "run-p"]
        assert len(pending) == 2 and len(lines) == 7

        _rotate_segment(audit_path, manifest_path)

        assert audit_path.read_bytes() == b"".join(pending)

    def test_a_carried_run_seals_in_the_new_segment_with_its_carried_and_new_lines(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        _rotate_segment(audit_path, manifest_path)
        _write_records(audit_path, [_record("run-p", 8)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-p")

        assert [manifest["run_id"] for manifest in manifests] == ["run-p"]
        assert manifests[0]["record_count"] == 3
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a")

    @pytest.mark.parametrize("case", list(_UNROTATABLE.values()), ids=list(_UNROTATABLE))
    def test_a_log_that_cannot_rotate_is_refused_and_nothing_changes(
        self, tmp_path: Path, case: Callable[[Path], tuple[Path, Path, dict[str, Any]]]
    ) -> None:
        audit_path, manifest_path, kwargs = case(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError):
            _rotate_segment(audit_path, manifest_path, **kwargs)

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("damage", list(_UNVERIFIABLE.values()), ids=list(_UNVERIFIABLE))
    def test_an_outgoing_segment_that_fails_verification_is_refused_and_nothing_changes(
        self, tmp_path: Path, damage: Callable[[Path, Path], dict[str, Any]]
    ) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        kwargs = damage(audit_path, manifest_path)

        _assert_raises_and_unchanged(tmp_path, lambda: _rotate_segment(audit_path, manifest_path, **kwargs))

    def test_the_head_anchor_gets_the_genesis_hash_once_the_log_holds_it(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        recording = _RecordingAnchor(manifest_path)

        genesis = _rotate_segment(audit_path, manifest_path, head_anchor=recording)

        assert recording.heads == [manifest_hash(genesis)]
        assert recording.last_lines_seen == [manifest_hash(genesis)]

    def test_the_new_segment_accepts_the_old_final_head_as_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        anchor = _ndjson_anchor(tmp_path / "anchor.ndjson")
        old_head = manifest_hash(_read_lines(manifest_path)[-1])
        anchor.write(old_head)

        genesis = _rotate_segment(audit_path, manifest_path, head_anchor=anchor)
        manifests = seal_ndjson_runs(
            audit_path, manifest_path, signer=_signer(), run_id="run-p", anchored_heads=[old_head]
        )

        assert anchor.latest() == manifest_hash(genesis)
        assert [manifest["run_id"] for manifest in manifests] == ["run-p"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a", anchored_heads=[old_head])

    @pytest.mark.parametrize("indexed", [False, True], ids=["no-index", "index"])
    def test_a_targeted_seal_of_a_run_sealed_in_an_archived_segment_is_refused(
        self, tmp_path: Path, indexed: bool
    ) -> None:
        audit_path, manifest_path = _rotated_with_stray(tmp_path)
        extra: dict[str, Any] = {"seal_index_path": tmp_path / "index.sqlite"} if indexed else {}
        if indexed:
            _write_records(audit_path, [_record("run-q", 10)])
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-q", **extra)
        before = (audit_path.read_bytes(), manifest_path.read_bytes())

        with pytest.raises(RunAlreadySealedError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-a", **extra)

        assert (audit_path.read_bytes(), manifest_path.read_bytes()) == before

    @pytest.mark.parametrize(
        "sweep", [{}, {"older_than": timedelta(days=1)}], ids=["blanket-sweep", "older-than-sweep"]
    )
    def test_a_sweep_skips_the_stray_records_of_a_run_sealed_in_an_archived_segment(
        self, tmp_path: Path, sweep: dict[str, Any]
    ) -> None:
        audit_path, manifest_path = _rotated_with_stray(tmp_path)

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), **sweep)

        assert [manifest["run_id"] for manifest in manifests] == ["run-p"]
        assert "run-a" not in {line.get("run_id") for line in _read_lines(manifest_path)}

    def test_a_second_rotation_does_not_carry_the_strays_of_a_run_sealed_in_an_archived_segment(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _rotated_with_stray(tmp_path)
        _write_records(audit_path, [_record("run-q", 10)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-p")

        _rotate_segment(audit_path, manifest_path)

        assert audit_path.read_bytes() == _canonical(_record("run-q", 10)) + b"\n"
        assert _archive(audit_path, 2).exists()

    @pytest.mark.parametrize("indexed", [False, True], ids=["no-index", "index"])
    def test_the_index_answers_the_archive_lookup_without_opening_the_archived_manifest(
        self, tmp_path: Path, indexed: bool
    ) -> None:
        audit_path, manifest_path = _rotated_with_stray(tmp_path)
        index_path = tmp_path / "index.sqlite"
        _write_records(audit_path, [_record("run-q", 10)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-q", seal_index_path=index_path)
        opened: list[str] = []
        real_open = builtins.open

        def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        with patch("builtins.open", spy):
            sealed = run_manifest_module._is_run_sealed_unverified(
                manifest_path, "run-a", index_path if indexed else None
            )

        assert sealed is True
        assert (str(_archive(manifest_path)) in opened) is (not indexed)

    @pytest.mark.parametrize("live", ["audit", "manifest"])
    @pytest.mark.parametrize("operation", ["seal", "rotate-key"])
    def test_an_unrelated_hard_link_to_a_live_file_does_not_block_writers(
        self, tmp_path: Path, live: str, operation: str
    ) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        os.link(audit_path if live == "audit" else manifest_path, tmp_path / "backup-link")
        head = manifest_hash(_read_lines(manifest_path)[-1])

        if operation == "seal":
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-p")
        else:
            rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()], expected_head=head
            )

        assert _read_lines(manifest_path)[-1] != _read_lines(manifest_path)[0]

    @pytest.mark.parametrize("repair", ["damaged-lines", "rotation-entry", "verify-segments"])
    def test_an_interrupted_rotation_refuses_recovery_and_segment_verification(
        self, tmp_path: Path, repair: str
    ) -> None:
        audit_path, manifest_path, _ = _crashed_rotation(tmp_path, 1)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        calls: dict[str, Callable[[], object]] = {
            "damaged-lines": lambda: _quarantine(tmp_path, audit_path, manifest_path),
            "rotation-entry": lambda: _quarantine_from_entry(tmp_path, manifest_path, expected_head=head),
            "verify-segments": lambda: _verify_segments(audit_path, manifest_path),
        }
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError, match="rotate_ndjson_segment"):
            calls[repair]()

        assert _snapshot(tmp_path) == before

    def test_a_sealer_cannot_append_to_the_new_manifest_before_the_genesis_reached_the_anchor(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        locked: list[bool] = []

        class LockProbingAnchor(_RecordingAnchor):
            def write(self, head: str) -> None:
                locked.append(_lock_refused(manifest_path))
                super().write(head)

        _rotate_segment(audit_path, manifest_path, head_anchor=LockProbingAnchor())

        assert locked == [True]

    def test_a_missing_audit_file_with_sealed_runs_is_a_verification_failure_and_changes_nothing(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        audit_path.unlink()

        _assert_raises_and_unchanged(tmp_path, lambda: _rotate_segment(audit_path, manifest_path))

    def test_a_record_of_a_sealed_run_appended_after_verification_refuses_the_rotation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path = _pending_log(tmp_path)
        stray = _canonical(_record("run-a", 9))
        before = _snapshot(tmp_path)
        real_carry = _segments._carry_pending
        appended: list[bool] = []

        def carry(*args: Any, **kwargs: Any) -> Any:
            if not appended:
                appended.append(True)
                _append_line(audit_path, stray)
            return real_carry(*args, **kwargs)

        _patch_bindings(monkeypatch, "_carry_pending", carry)

        with pytest.raises((ManifestVerificationError, ValueError)):
            _rotate_segment(audit_path, manifest_path)

        assert appended
        assert _snapshot(tmp_path) == {**before, "audit.ndjson": before["audit.ndjson"] + stray + b"\n"}
        assert _archives(tmp_path) == []

    def test_a_partial_last_line_completed_before_the_audit_lock_is_carried_when_its_run_is_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _pending_log(tmp_path)
        _append_line(audit_path, _canonical(_record("run-p", 8)))
        data = audit_path.read_bytes()
        audit_path.write_bytes(data[:-1])
        audit_inode = audit_path.stat().st_ino
        completed: list[bool] = []
        real_flock = fcntl.flock

        def flock(fd: int, operation: int) -> None:
            if operation & fcntl.LOCK_EX and os.fstat(fd).st_ino == audit_inode and not completed:
                completed.append(True)
                with open(audit_path, "ab") as file:
                    file.write(b"\n")
            real_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", flock)

        _rotate_segment(audit_path, manifest_path)

        assert completed
        assert [line["run_id"] for line in _read_lines(audit_path)] == ["run-p", "run-p", "run-p"]
        assert audit_path.read_bytes().endswith(b"\n")

    @pytest.mark.parametrize("blocked", ["seal", "rotate-key"])
    def test_a_crash_before_the_first_swap_leaves_both_live_files_linked_and_blocks_writers(
        self, tmp_path: Path, blocked: str
    ) -> None:
        audit_path, manifest_path, _ = _crashed_rotation(tmp_path, 1)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        calls: dict[str, Callable[[], object]] = {
            "seal": lambda: seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-p"),
            "rotate-key": lambda: rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()], expected_head=head
            ),
        }
        before = _snapshot(tmp_path)

        for live in (audit_path, manifest_path):
            assert live.stat().st_nlink > 1
            assert _same_inode(live, _archive(live))
        with pytest.raises(ValueError, match="rotate_ndjson_segment"):
            calls[blocked]()

        assert _snapshot(tmp_path) == before

    def test_rotating_again_after_a_crash_before_the_first_swap_completes_under_the_same_number(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, before = _crashed_rotation(tmp_path, 1)
        old_head = manifest_hash(_read_lines(manifest_path)[-1])

        genesis = _rotate_segment(audit_path, manifest_path)

        assert _archives(tmp_path) == [
            "audit.ndjson.000001",
            "manifests.ndjson.000001",
        ]
        assert _archive(audit_path).read_bytes() == before["audit.ndjson"]
        assert _archive(manifest_path).read_bytes() == before["manifests.ndjson"]
        assert _read_lines(manifest_path) == [genesis]
        assert genesis["previous_manifest_hash"] == old_head
        assert audit_path.stat().st_nlink == manifest_path.stat().st_nlink == 1
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a")

    def test_after_a_crash_between_the_swaps_writes_land_in_the_new_audit_and_sealing_is_blocked(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, before = _crashed_rotation(tmp_path, 2)

        assert not _same_inode(audit_path, _archive(audit_path))
        assert _same_inode(manifest_path, _archive(manifest_path))
        _write_records(audit_path, [_record("run-n", 20)])
        archived_audit = _archive(audit_path).read_bytes()
        manifest_before = manifest_path.read_bytes()
        with pytest.raises(ValueError, match="rotate_ndjson_segment"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-n")

        assert archived_audit == before["audit.ndjson"]
        assert manifest_path.read_bytes() == manifest_before
        assert [line["run_id"] for line in _read_lines(audit_path)] == ["run-p", "run-p", "run-n"]

    def test_rotating_again_after_a_crash_between_the_swaps_chains_the_genesis_and_loses_no_record(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, before = _crashed_rotation(tmp_path, 2)
        _write_records(audit_path, [_record("run-n", 20)])
        archived_head = manifest_hash(_read_lines(_archive(manifest_path))[-1])

        genesis = _rotate_segment(audit_path, manifest_path)

        assert genesis["previous_manifest_hash"] == archived_head
        assert _read_lines(manifest_path) == [genesis]
        assert _archive(manifest_path).read_bytes() == before["manifests.ndjson"]
        assert _archive(audit_path).read_bytes() == before["audit.ndjson"]
        assert _archives(tmp_path) == ["audit.ndjson.000001", "manifests.ndjson.000001"]
        assert [line["run_id"] for line in _read_lines(audit_path)] == ["run-p", "run-p", "run-n"]
        assert manifest_path.stat().st_nlink == 1
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a")

    def test_a_sink_write_that_opened_the_old_audit_file_before_the_swap_lands_in_the_new_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _pending_log(tmp_path)
        before = audit_path.read_bytes()
        old_inode = audit_path.stat().st_ino
        entered, started = threading.Event(), threading.Event()
        errors: list[BaseException] = []

        def write_record() -> None:
            try:
                NdjsonAuditSink(audit_path).write(_record("run-sink", 20))
            except BaseException as error:  # noqa: BLE001 - surfaced by the assertion below
                errors.append(error)

        writer = threading.Thread(target=write_record)
        real_flock = fcntl.flock

        def flock(fd: int, operation: int) -> None:
            on_old_file = os.fstat(fd).st_ino == old_inode
            if on_old_file and threading.current_thread() is writer:
                entered.set()
            real_flock(fd, operation)
            if on_old_file and operation & fcntl.LOCK_EX and not started.is_set():
                # Rotation holds the audit file exclusively: start a writer that opened the old file, wait until it
                # reached its lock, and let the rotation go on to swap the file.
                started.set()
                writer.start()
                entered.wait(5)

        monkeypatch.setattr(fcntl, "flock", flock)

        _rotate_segment(audit_path, manifest_path)
        writer.join(10)

        assert not errors
        assert started.is_set() and not writer.is_alive()
        assert _archive(audit_path).read_bytes() == before
        assert [line["run_id"] for line in _read_lines(audit_path)] == ["run-p", "run-p", "run-sink"]

    def test_a_record_of_a_run_sealed_in_the_outgoing_segment_during_the_copy_pass_refuses_the_rotation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _pending_log(tmp_path)
        stray = _canonical(_record("run-a", 9))
        before = _snapshot(tmp_path)
        audit_inode = audit_path.stat().st_ino
        appended: list[bool] = []
        real_flock = fcntl.flock

        def flock(fd: int, operation: int) -> None:
            if operation & fcntl.LOCK_EX and os.fstat(fd).st_ino == audit_inode and not appended:
                # The copy pass is over and the audit lock is about to be taken: a writer lands a stray line now.
                appended.append(True)
                _append_line(audit_path, stray)
            real_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", flock)

        with pytest.raises(ValueError):
            _rotate_segment(audit_path, manifest_path)

        assert appended
        assert _snapshot(tmp_path) == {**before, "audit.ndjson": before["audit.ndjson"] + stray + b"\n"}
        assert _archives(tmp_path) == []


def _verify_segments(audit_path: Path, manifest_path: Path, **kwargs: Any) -> LogCoverage:
    """verify_ndjson_segments with the default signer."""
    coverage: LogCoverage = verify_ndjson_segments(audit_path, manifest_path, **{"signer": _signer(), **kwargs})
    return coverage


def _three_segments(directory: Path) -> tuple[Path, Path]:
    """Two rotations. Segment 1 seals a, b, c; segment 2 seals run-q; the live segment seals run-p, which was pending
    in segments 1 and 2 and was carried across both rotations, and leaves run-r pending."""
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    _write_records(audit_path, [_record("run-q", 10), _record(None, 11)])
    seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-q")
    _rotate_segment(audit_path, manifest_path)
    _write_records(audit_path, [_record("run-p", 12), _record("run-r", 13)])
    seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-p")
    return audit_path, manifest_path


def _head_of(manifest_path: Path, index: int = -1) -> str:
    return manifest_hash(_read_lines(manifest_path)[index])


def _middle_segment_deleted(directory: Path) -> dict[str, Any]:
    audit_path, manifest_path = _three_segments(directory)
    _archive(audit_path, 2).unlink()
    _archive(manifest_path, 2).unlink()
    return {}


def _genesis_under_another_key(directory: Path) -> dict[str, Any]:
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    key_2 = _signer(_OTHER_KEY, "key-2")
    manifest_path.unlink()
    _write_records(manifest_path, [_genesis_entry(key_2, previous_manifest_hash=_head_of(_archive(manifest_path)))])
    return {"signer": key_2, "previous_signers": [_signer()]}


def _run_sealed_twice(directory: Path) -> dict[str, Any]:
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    record = _record("run-a", 9)
    _write_records(audit_path, [record])
    seal = seal_run([record], run_id="run-a", signer=_signer(), previous_manifest_hash=_head_of(manifest_path))
    _append_line(manifest_path, _canonical(seal))
    return {}


def _record_stranded_in_the_archive(directory: Path) -> dict[str, Any]:
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    _append_line(_archive(audit_path), _canonical(_record("run-p", 30)))
    return {}


def _carried_line_dropped(directory: Path) -> dict[str, Any]:
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    audit_path.write_bytes(audit_path.read_bytes().splitlines(keepends=True)[0])
    return {}


def _sealed_run_reappears(directory: Path) -> dict[str, Any]:
    _rotated_with_stray(directory)
    return {}


def _unknown_anchor(directory: Path) -> dict[str, Any]:
    _three_segments(directory)
    return {"anchored_heads": [_sha256(b"unknown")]}


def _wrong_expected_head(directory: Path) -> dict[str, Any]:
    _three_segments(directory)
    return {"expected_head": _sha256(b"not the head")}


def _old_segment_head_as_expected_head(directory: Path) -> dict[str, Any]:
    _, manifest_path = _three_segments(directory)
    return {"expected_head": _head_of(_archive(manifest_path))}


def _wrong_log_id(directory: Path) -> dict[str, Any]:
    _three_segments(directory)
    return {"log_id": "log-other"}


def _signer_not_current(directory: Path) -> dict[str, Any]:
    _three_segments(directory)
    return {"signer": _signer(_OTHER_KEY, "key-2")}


def _successor_genesis_names_another_log_id(directory: Path) -> dict[str, Any]:
    audit_path, manifest_path = _pending_log(directory)
    _rotate_segment(audit_path, manifest_path)
    manifest_path.unlink()
    genesis = _genesis_entry(_signer(), "log-b", previous_manifest_hash=_head_of(_archive(manifest_path)))
    _write_records(manifest_path, [genesis])
    return {"log_id": None}


_SEGMENT_FAILURES = {
    "middle-segment-deleted": (_middle_segment_deleted, "manifest chain is broken"),
    "genesis-under-another-key": (_genesis_under_another_key, "genesis entry is signed by key"),
    "run-sealed-in-two-segments": (_run_sealed_twice, "sealed by more than one manifest"),
    "carried-run-stranded-in-the-archive": (_record_stranded_in_the_archive, "stranded"),
    "carried-line-dropped-from-the-next-segment": (_carried_line_dropped, "stranded"),
    "sealed-run-reappears-in-a-later-segment": (_sealed_run_reappears, "run_id 'run-a' has records"),
    "unknown-anchored-head": (_unknown_anchor, "anchored head .* is not a line"),
    "wrong-expected-head": (_wrong_expected_head, "is not the expected head"),
    "old-segment-head-is-not-the-expected-head": (_old_segment_head_as_expected_head, "is not the expected head"),
    "wrong-log-id": (_wrong_log_id, "manifest log is for log_id"),
    "signer-not-current": (_signer_not_current, "not the signer's"),
    "successor-genesis-names-another-log-id": (_successor_genesis_names_another_log_id, "log_id"),
}


@_both_algorithms
class TestVerifyNdjsonSegments:
    """verify_ndjson_segments: the retained archived pairs in order plus the live pair, as one chain."""

    def test_three_segments_with_a_run_carried_across_both_rotations_pass_and_aggregate(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _three_segments(tmp_path)

        coverage = _verify_segments(audit_path, manifest_path, log_id="log-a")

        assert coverage == LogCoverage(
            head=_head_of(manifest_path),
            sealed_runs=5,
            sealed_lines=7,
            unattributed_lines=3,
            unsealed_lines={"run-r": 1},
        )

    def test_without_a_rotation_it_matches_verify_ndjson_log_coverage(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        _write_records(audit_path, [_record("run-p", 4), _record(None, 5)])

        coverage = _verify_segments(audit_path, manifest_path)

        assert coverage == verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

    def test_dropping_the_oldest_pair_by_retention_still_passes(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _three_segments(tmp_path)
        _archive(audit_path).unlink()
        _archive(manifest_path).unlink()

        coverage = _verify_segments(audit_path, manifest_path)

        assert coverage.head == _head_of(manifest_path)
        assert coverage.sealed_runs == 2

    def test_a_key_rotation_inside_a_segment_followed_by_a_segment_rotation_passes(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        key_1, key_2 = _signer(), _signer(_OTHER_KEY, "key-2")
        _rotate(manifest_path, key_2, key_1)
        _write_records(audit_path, [_record("run-d", 4)])
        seal_ndjson_runs(audit_path, manifest_path, signer=key_2, previous_signers=[key_1])
        _rotate_segment(audit_path, manifest_path, signer=key_2, previous_signers=[key_1])
        _write_records(audit_path, [_record("run-e", 5)])
        seal_ndjson_runs(audit_path, manifest_path, signer=key_2, previous_signers=[key_1])

        coverage = _verify_segments(audit_path, manifest_path, signer=key_2, previous_signers=[key_1])

        assert coverage.head == _head_of(manifest_path)
        assert coverage.sealed_runs == 5

    @pytest.mark.parametrize("build, message", list(_SEGMENT_FAILURES.values()), ids=list(_SEGMENT_FAILURES))
    def test_a_broken_history_fails_and_changes_nothing(
        self, tmp_path: Path, build: Callable[[Path], dict[str, Any]], message: str
    ) -> None:
        kwargs = build(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError, match=message):
            _verify_segments(tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson", **kwargs)

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize(
        "anchor",
        [
            lambda a, m: _head_of(_archive(m, 1)),
            lambda a, m: _head_of(_archive(m, 1), 0),
            lambda a, m: _head_of(_archive(m, 2)),
            lambda a, m: _head_of(m, 0),
        ],
        ids=["first-segment-head", "first-genesis", "middle-segment-head", "live-genesis"],
    )
    def test_an_anchored_head_may_be_a_line_of_any_retained_segment(
        self, tmp_path: Path, anchor: Callable[[Path, Path], str]
    ) -> None:
        audit_path, manifest_path = _three_segments(tmp_path)

        coverage = _verify_segments(
            audit_path, manifest_path, anchored_heads=[anchor(audit_path, manifest_path)], log_id="log-a"
        )

        assert coverage.head == _head_of(manifest_path)

    def test_the_expected_head_is_the_live_segments_head(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _three_segments(tmp_path)

        coverage = _verify_segments(audit_path, manifest_path, expected_head=_head_of(manifest_path))

        assert coverage.head == _head_of(manifest_path)

    def test_the_live_manifest_is_shared_locked_while_the_archives_and_audit_files_are_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path = _three_segments(tmp_path)
        locked_during_read = _locked_during_digest(monkeypatch, manifest_path)

        _verify_segments(audit_path, manifest_path)

        assert locked_during_read and all(locked_during_read)


@_both_algorithms
class TestRotateManifestKey:
    """Lock, fsync and rollback are covered in TestSealNdjsonRuns."""

    def test_entry_has_exactly_the_expected_keys_kind_and_version(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)

        entry = _rotate(manifest_path, _signer(_OTHER_KEY, "key-2"), _signer())

        assert set(entry) == _EXPECTED_ROTATION_KEYS
        assert entry["manifest_version"] == 2
        assert entry["kind"] == "key_rotation"

    def test_rotated_at_is_rfc3339_utc(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)

        rotated_at = _rotate(manifest_path, _signer(_OTHER_KEY, "key-2"), _signer())["rotated_at"]

        assert rotated_at.endswith("Z")
        datetime.fromisoformat(rotated_at.removesuffix("Z"))

    def test_entry_is_signed_by_the_new_key_over_the_unsigned_entry(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        new_signer = _signer(_OTHER_KEY, "key-2")

        entry = _rotate(manifest_path, new_signer, _signer())

        signature = entry["signature"]
        assert set(signature) == {"algorithm", "key_id", "value"}
        assert (signature["algorithm"], signature["key_id"]) == (new_signer.algorithm, "key-2")
        assert signature["value"] == _reference_signature(_OTHER_KEY, _signing_payload_ref(entry))

    def test_entry_is_co_signed_by_the_outgoing_key_over_the_same_payload(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        old_signer = _signer()

        entry = _rotate(manifest_path, _signer(_OTHER_KEY, "key-2"), old_signer)

        co_signature = entry["previous_key_signature"]
        assert set(co_signature) == {"algorithm", "key_id", "value"}
        assert (co_signature["algorithm"], co_signature["key_id"]) == (old_signer.algorithm, "key-1")
        assert co_signature["value"] == _reference_signature(_KEY, _signing_payload_ref(entry))

    def test_the_outgoing_key_is_found_by_the_logs_current_key_id_among_several_previous_signers(
        self, tmp_path: Path
    ) -> None:
        _, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path)
        key_4 = _signer(b"f" * 32, "key-4")

        entry = _rotate(manifest_path, key_4, key_1, key_3, key_2)

        assert entry["previous_key_signature"]["key_id"] == "key-3"
        assert entry["previous_key_signature"]["value"] == _reference_signature(_THIRD_KEY, _signing_payload_ref(entry))

    def test_entry_chains_onto_the_head_and_is_the_only_line_appended(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = _read_lines(manifest_path)

        entry = _rotate(manifest_path, _signer(_OTHER_KEY, "key-2"), _signer())

        assert entry["previous_manifest_hash"] == manifest_hash(before[-1])
        assert _read_lines(manifest_path) == [*before, entry]

    @pytest.mark.parametrize("log_bytes", [None, b""], ids=["missing", "empty"])
    def test_a_log_without_manifests_raises_value_error_and_creates_no_file(
        self, tmp_path: Path, log_bytes: bytes | None
    ) -> None:
        manifest_path = tmp_path / "manifests.ndjson"
        if log_bytes is not None:
            manifest_path.write_bytes(log_bytes)

        with pytest.raises(ValueError) as excinfo:
            rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()], expected_head=None
            )

        assert not isinstance(excinfo.value, (ManifestVerificationError, KeyAlreadyCurrentError))
        assert manifest_path.exists() == (log_bytes is not None)

    @pytest.mark.parametrize("rotated_before", [False, True], ids=["fresh-log", "retry-after-rotation"])
    def test_a_log_already_under_the_signer_raises_key_already_current_error_and_writes_nothing(
        self, tmp_path: Path, rotated_before: bool
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        signer, previous_signers = (new_signer, [old_signer]) if rotated_before else (old_signer, [])
        if rotated_before:
            _rotate(manifest_path, new_signer, old_signer)
        before = manifest_path.read_bytes()
        head = manifest_hash(_read_lines(manifest_path)[-1])

        with pytest.raises(KeyAlreadyCurrentError) as excinfo:
            rotate_manifest_key(manifest_path, signer=signer, previous_signers=previous_signers, expected_head=head)

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("anchor", ["pre-rotation-head", "none", "post-rotation-head"])
    def test_a_retry_after_a_rotation_fails_on_a_stale_head_and_is_current_otherwise(
        self, tmp_path: Path, anchor: str
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        pre_rotation_head = manifest_hash(_read_lines(manifest_path)[-1])
        _rotate(manifest_path, new_signer, old_signer)
        heads = {
            "pre-rotation-head": pre_rotation_head,
            "none": None,
            "post-rotation-head": manifest_hash(_read_lines(manifest_path)[-1]),
        }
        before = manifest_path.read_bytes()
        stale = anchor == "pre-rotation-head"

        with pytest.raises(
            ManifestVerificationError if stale else KeyAlreadyCurrentError,
            match="is not the expected head" if stale else "already under key",
        ) as excinfo:
            rotate_manifest_key(
                manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=heads[anchor]
            )

        assert isinstance(excinfo.value, ManifestVerificationError) is stale
        assert isinstance(excinfo.value, KeyAlreadyCurrentError) is not stale
        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("seal_between", [True, False], ids=["seal-between", "back-to-back"])
    @pytest.mark.parametrize("retired", [0, 1], ids=["oldest-key", "middle-key"])
    def test_rotating_back_to_a_retired_key_raises_manifest_verification_error_and_writes_nothing(
        self, tmp_path: Path, retired: int, seal_between: bool
    ) -> None:
        _, manifest_path, *keys = _rotated_three_key_log(tmp_path, seal_between=seal_between)
        target = keys[retired]
        before = manifest_path.read_bytes()
        head = manifest_hash(_read_lines(manifest_path)[-1])

        with pytest.raises(ManifestVerificationError, match=target.key_id):
            rotate_manifest_key(
                manifest_path,
                signer=target,
                previous_signers=[key for key in keys if key is not target],
                expected_head=head,
            )

        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize(
        "previous_signers",
        [[_signer()], [_signer(key_id="key-3"), _signer(_OTHER_KEY, "key-3")]],
        ids=["signer-and-previous", "two-previous"],
    )
    def test_previous_signers_with_a_duplicate_key_id_is_a_value_error(
        self, tmp_path: Path, previous_signers: list[ManifestSigner]
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = manifest_path.read_bytes()

        with pytest.raises(ValueError) as excinfo:
            rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY), previous_signers=previous_signers, expected_head=None
            )

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("anchor", ["unrelated", "older-manifest"])
    def test_an_expected_head_that_is_not_the_head_raises_and_writes_nothing(self, tmp_path: Path, anchor: str) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        expected_head = "0" * 64 if anchor == "unrelated" else manifest_hash(_read_lines(manifest_path)[0])
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match="head"):
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=expected_head,
            )

        assert manifest_path.read_bytes() == before

    def test_expected_head_none_means_no_anchor_and_is_accepted(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])

        entry = rotate_manifest_key(
            manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()], expected_head=None
        )

        assert entry["previous_manifest_hash"] == head

    def test_omitting_expected_head_is_a_type_error(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = manifest_path.read_bytes()

        with pytest.raises(TypeError):
            rotate_manifest_key(  # type: ignore[call-arg]
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()]
            )

        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("damage", ["torn-tail", "edited-manifest"])
    def test_a_torn_or_tampered_log_raises_manifest_verification_error_and_writes_nothing(
        self, tmp_path: Path, damage: str
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        if damage == "torn-tail":
            _torn(manifest_path, b'{"run_id": "run-d"')
        else:
            manifests = _read_lines(manifest_path)
            manifests[0]["compliant"] = not manifests[0]["compliant"]
            _rewrite_lines(manifest_path, manifests)
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError):
            rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()], expected_head=None
            )

        assert manifest_path.read_bytes() == before

    def test_rotating_without_the_retired_signer_raises_manifest_verification_error_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError):
            rotate_manifest_key(manifest_path, signer=_signer(_OTHER_KEY, "key-2"), expected_head=None)

        assert manifest_path.read_bytes() == before

    def test_the_outgoing_signer_signs_exactly_once_per_rotation(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        signed: list[bytes] = []

        _rotate(manifest_path, _signer(_OTHER_KEY, "key-2"), _HookedSigner(_signer(), on_sign=signed.append))

        assert len(signed) == 1
        assert _read_lines(manifest_path)[-1]["previous_key_signature"]["key_id"] == "key-1"

    def test_the_existing_log_is_verified_only_once(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        verified: list[bytes] = []

        _rotate(manifest_path, _signer(_OTHER_KEY, "key-2"), _HookedSigner(_signer(), on_verify=verified.append))

        assert len(verified) == len(_read_lines(manifest_path)) - 1

    def test_rotating_works_where_fcntl_is_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "fcntl", None)
        _, write, verify = _pending_write(tmp_path, "rotate")

        (entry,) = write()

        assert verify() == manifest_hash(entry)

    def test_rotating_to_a_fresh_key_restores_a_log_wedged_by_a_forged_rotation_entry(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        key_1, key_2, key_3 = _signer(), _signer(_OTHER_KEY, "key-2"), _signer(b"t" * 32, "key-3")
        _append_line(manifest_path, _canonical(_key_2_entry(manifest_hash(_read_lines(manifest_path)[-1]))))
        _write_records(audit_path, [_record("run-d", 7)])
        # The complete forged entry moves the log to key-2: the honest key-1 can neither verify nor seal.
        with pytest.raises(ManifestVerificationError, match=_under_key("key-2", "key-1")):
            verify_ndjson_log(audit_path, manifest_path, signer=key_1, previous_signers=[key_2])
        with pytest.raises(ManifestVerificationError, match=_under_key("key-2", "key-1")):
            seal_ndjson_runs(audit_path, manifest_path, signer=key_1, previous_signers=[key_2])

        head = manifest_hash(_rotate(manifest_path, key_3, key_1, key_2))

        assert verify_ndjson_log(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2]) == head
        manifests = seal_ndjson_runs(
            audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2], expected_head=head
        )
        assert [(manifest["run_id"], manifest["previous_manifest_hash"]) for manifest in manifests] == [("run-d", head)]
        verified = verify_ndjson_log(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])
        assert verified == manifest_hash(manifests[-1])


@_both_algorithms
class TestVerifyNdjsonLog:
    def test_untouched_log_verifies(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_returns_the_manifest_hash_of_the_last_manifest(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        head = verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert head == manifest_hash(_read_lines(manifest_path)[-1])

    def test_records_of_a_run_that_is_not_sealed_yet_are_fine(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7), _record("run-d", 8)])

        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_log_sealed_out_of_audit_file_order_verifies(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2), _record("run-c", 3)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-b")

        rest = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert [manifest["run_id"] for manifest in rest] == ["run-a", "run-c"]
        assert [manifest["run_id"] for manifest in _read_lines(manifest_path)] == ["run-b", "run-a", "run-c"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_signer_with_a_different_key_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(_OTHER_KEY))

    def test_previous_signers_verifies_a_log_sealed_before_rotation(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        expected_head = manifest_hash(_rotate(manifest_path, new_signer, old_signer))

        head = verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert head == expected_head

    def test_mixed_key_log_verifies_after_a_further_seal_with_the_new_key(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        _rotate(manifest_path, new_signer, old_signer)
        _write_records(audit_path, [_record("run-d", 7)])
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        expected_head = manifest_hash(manifests[-1])

        head = verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert head == expected_head

    def test_rotated_log_fails_without_the_old_signer_in_previous_signers(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        _rotate(manifest_path, new_signer, old_signer)
        _write_records(audit_path, [_record("run-d", 7)])
        seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer)

    def test_a_previous_key_manifest_appended_after_the_current_key_is_rejected(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        _write_records(audit_path, [_record("run-a", 1)])
        seal_ndjson_runs(audit_path, manifest_path, signer=old_signer)
        _rotate(manifest_path, new_signer, old_signer)
        _write_records(audit_path, [_record("run-b", 2)])
        seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        head = manifest_hash(_read_lines(manifest_path)[-1])
        record = _record("run-c", 3)
        _write_records(audit_path, [record])
        forged = seal_run([record], run_id="run-c", signer=old_signer, previous_manifest_hash=head)
        _write_records(manifest_path, [forged])

        with pytest.raises(ManifestVerificationError, match="run-c"):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    def test_three_key_rotation_verifies_with_the_current_and_both_previous_signers(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path)

        verify_ndjson_log(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])

    def test_three_key_rotation_rejects_a_retired_key_named_as_the_current_signer(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path)

        with pytest.raises(ManifestVerificationError, match=_under_key("key-3", "key-1")):
            verify_ndjson_log(audit_path, manifest_path, signer=key_1, previous_signers=[key_2, key_3])

    @pytest.mark.parametrize("seal_between", [True, False], ids=["seal-between", "back-to-back"])
    @pytest.mark.parametrize("retired", [0, 1], ids=["oldest-key", "middle-key"])
    def test_three_key_rotation_rejects_a_retired_key_manifest_after_the_last_rotation(
        self, tmp_path: Path, retired: int, seal_between: bool
    ) -> None:
        audit_path, manifest_path, *keys = _rotated_three_key_log(tmp_path, seal_between=seal_between)
        record = _record("run-d", 4)
        _write_records(audit_path, [record])
        head = manifest_hash(_read_lines(manifest_path)[-1])
        forged = seal_run([record], run_id="run-d", signer=keys[retired], previous_manifest_hash=head)
        _write_records(manifest_path, [forged])

        with pytest.raises(ManifestVerificationError, match="run-d") as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=keys[2], previous_signers=keys[:2])

        _assert_names(excinfo, keys[retired].key_id, "key-3")

    def test_previous_signer_with_a_different_algorithm_verifies_against_its_own_algorithm(
        self, tmp_path: Path
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        old_signer = _PrefixSigner()
        new_signer = _signer(_OTHER_KEY, "key-2")
        _write_records(audit_path, [_record("run-a", 1)])
        seal_ndjson_runs(audit_path, manifest_path, signer=old_signer)
        _rotate(manifest_path, new_signer, old_signer)

        verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    def test_previous_signer_manifest_with_a_mismatched_algorithm_is_rejected(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        old_signer = _PrefixSigner()
        new_signer = _signer(_OTHER_KEY, "key-2")
        _write_records(audit_path, [_record("run-a", 1)])
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=old_signer)
        # Forged after rotating: rotating verifies the log first.
        entry = _rotate(manifest_path, new_signer, old_signer)
        forged = {**manifests[0], "signature": {**manifests[0]["signature"], "algorithm": "HMAC-SHA256"}}
        _rewrite_lines(manifest_path, [forged, entry])

        with pytest.raises(ManifestVerificationError, match="algorithm"):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    @pytest.mark.parametrize(
        "previous_signers",
        [[_signer()], [_signer(key_id="key-3"), _signer(_OTHER_KEY, "key-3")]],
        ids=["signer-and-previous", "two-previous"],
    )
    def test_previous_signers_with_a_duplicate_key_id_is_a_value_error(
        self, tmp_path: Path, previous_signers: list[ManifestSigner]
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(_OTHER_KEY), previous_signers=previous_signers)

        assert not isinstance(excinfo.value, ManifestVerificationError)

    def test_edited_audit_line_of_a_sealed_run_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[0]["run_id"] == "run-a"
        records[0]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, records)

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        assert "do not match the record_hashes" in str(excinfo.value)

    def test_deleted_audit_line_of_a_sealed_run_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[0]["run_id"] == "run-a"
        _rewrite_lines(audit_path, records[1:])

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        assert "do not match the record_hashes" in str(excinfo.value)

    def test_sealed_run_without_any_audit_line_left_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert [record["run_id"] for record in records].count("run-c") == 1
        _rewrite_lines(audit_path, [record for record in records if record["run_id"] != "run-c"])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("appended", [1, 2])
    def test_records_appended_to_a_sealed_run_fail_and_the_run_is_never_resealed(
        self, tmp_path: Path, appended: int
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-a", 9 + offset) for offset in range(appended)])
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        message = str(excinfo.value)
        assert "run-a" in message
        assert f"has {appended} record(s) beyond its seal" in message
        assert "do not match the record_hashes" not in message
        assert seal_ndjson_runs(audit_path, manifest_path, signer=_signer()) == []
        assert manifest_path.read_bytes() == before

    def test_edited_last_manifest_line_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        manifests = _read_lines(manifest_path)
        manifests[-1]["compliant"] = not manifests[-1]["compliant"]
        _rewrite_lines(manifest_path, manifests)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_manifest_line_with_an_unhashable_run_id_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        manifests = _read_lines(manifest_path)
        assert manifests[0]["run_id"] == "run-a"
        manifests[0] = _resigned(manifests[0], run_id=["run-a"])
        _rewrite_lines(manifest_path, manifests)

        with pytest.raises(ManifestVerificationError, match="run_id"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("index", [0, 1])
    def test_deleted_manifest_line_breaks_the_chain(self, tmp_path: Path, index: int) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        manifests = _read_lines(manifest_path)
        del manifests[index]
        _rewrite_lines(manifest_path, manifests)

        with pytest.raises(ManifestVerificationError, match="chain"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_manifest_removed_together_with_its_records_breaks_the_chain(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        manifests = _read_lines(manifest_path)
        assert manifests[1]["run_id"] == "run-b"
        _rewrite_lines(manifest_path, [manifests[0], manifests[2]])
        _rewrite_lines(audit_path, [record for record in _read_lines(audit_path) if record["run_id"] != "run-b"])

        with pytest.raises(ManifestVerificationError, match="chain"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize(("left", "right"), [(0, 1), (1, 2)])
    def test_swapped_manifest_lines_break_the_chain(self, tmp_path: Path, left: int, right: int) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        manifests = _read_lines(manifest_path)
        manifests[left], manifests[right] = manifests[right], manifests[left]
        _rewrite_lines(manifest_path, manifests)

        with pytest.raises(ManifestVerificationError, match="chain"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_same_run_id_in_two_manifests_fails(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        records = [_record("run-a", 1), _record("run-a", 2)]
        _write_records(audit_path, records)
        first = seal_run(records, run_id="run-a", signer=_signer())
        second = seal_run(records, run_id="run-a", signer=_signer(), previous_manifest_hash=manifest_hash(first))
        _write_records(manifest_path, [first, second])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize(
        "dump_options",
        [{"sort_keys": True, "separators": (",", ":")}, {"sort_keys": False}],
        ids=["compact", "reversed-keys"],
    )
    def test_reformatted_audit_line_of_a_sealed_run_fails(self, tmp_path: Path, dump_options: dict[str, Any]) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        lines = audit_path.read_text(encoding="utf-8").splitlines()
        original = json.loads(lines[0])
        assert original["run_id"] == "run-a"
        reformatted = json.dumps(dict(reversed(original.items())), **dump_options)
        assert reformatted != lines[0]
        assert json.loads(reformatted) == original
        lines[0] = reformatted
        audit_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("log_bytes", [None, b""], ids=["missing", "empty"])
    def test_missing_or_empty_manifest_log_returns_none(self, tmp_path: Path, log_bytes: bytes | None) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])
        if log_bytes is not None:
            manifest_path.write_bytes(log_bytes)

        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) is None
        assert manifest_path.exists() == (log_bytes is not None)

    def test_missing_manifest_log_fails_against_an_expected_head(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])

        with pytest.raises(ManifestVerificationError, match="head"):
            verify_ndjson_log(audit_path, tmp_path / "manifests.ndjson", signer=_signer(), expected_head="ab" * 32)

    def test_every_failing_run_is_reported_in_one_error(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[5]["run_id"] == "run-c"
        records[5]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, [*records, _record("run-a", 9)])

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        message = str(excinfo.value)
        assert "run-a" in message
        assert "run-c" in message
        assert "run-b" not in message

    def test_at_most_twenty_failing_runs_are_reported_and_the_rest_are_counted(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record(f"run-{number:02d}", number) for number in range(25)])
        assert len(seal_ndjson_runs(audit_path, manifest_path, signer=_signer())) == 25
        audit_path.write_bytes(b"")

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        message = str(excinfo.value)
        assert "and 5 more" in message
        assert message.count("do not match the record_hashes") == 20

    def test_a_head_mismatch_is_reported_even_when_more_than_twenty_runs_fail(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record(f"run-{number:02d}", number) for number in range(25)])
        assert len(seal_ndjson_runs(audit_path, manifest_path, signer=_signer())) == 25
        manifests = _read_lines(manifest_path)
        head = manifest_hash(manifests[-1])
        _rewrite_lines(manifest_path, manifests[:-1])
        audit_path.write_bytes(b"")

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), expected_head=head)

        message = str(excinfo.value)
        assert "expected head" in message
        assert message.count("do not match the record_hashes") == 19
        assert "and 5 more" in message

    def test_deleted_audit_file_fails_every_sealed_run(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        audit_path.unlink()

        with pytest.raises(ManifestVerificationError, match="record") as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        message = str(excinfo.value)
        assert all(run_id in message for run_id in ("run-a", "run-b", "run-c"))

    def test_neither_audit_file_nor_manifest_log_returns_none(self, tmp_path: Path) -> None:
        assert verify_ndjson_log(tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson", signer=_signer()) is None

    def test_a_head_mismatch_is_reported_together_with_a_record_mismatch(self, tmp_path: Path) -> None:
        audit_path, manifest_path, head = _truncated_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[0]["run_id"] == "run-a"
        records[0]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, records)

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), expected_head=head)

        message = str(excinfo.value)
        assert "head" in message
        assert "record" in message

    def test_manifest_log_is_read_before_the_audit_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        original = _verify._read_ndjson
        reads: list[str] = []

        def spy(path: str | Path) -> Any:
            reads.append(Path(path).name)
            return original(path)

        _patch_bindings(monkeypatch, "_read_ndjson", spy)

        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        # Log first: a run sealed in between looks unsealed, not tampered.
        assert reads == ["manifests.ndjson", "audit.ndjson"]

    def test_edited_audit_line_without_a_run_id_still_verifies(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[2]["run_id"] is None
        records[2]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, records)

        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_verification_waits_for_a_sealer_holding_the_lock(self, tmp_path: Path) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])

        with ThreadPoolExecutor(max_workers=1) as pool:
            fd = os.open(manifest_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                future = pool.submit(verify_ndjson_log, audit_path, manifest_path, signer=_signer())
                done, _ = wait([future], timeout=0.3)
            finally:
                os.close(fd)

            assert not done
            assert future.result(timeout=10) == head

    def test_verification_proceeds_unlocked_when_the_shared_lock_is_unsupported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        monkeypatch.setattr(fcntl, "flock", _flock_unsupported)

        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head


@_both_algorithms
class TestVerifyNdjsonLogCoverage:
    def test_a_verifier_locking_a_replaced_manifest_relocks_the_new_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        archive, locked, new_ino = _swap_on_first_flock(monkeypatch, manifest_path, tmp_path)
        archived = archive.read_bytes()

        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert locked[-1] == new_ino
        assert archive.read_bytes() == archived

    def test_the_manifest_is_shared_locked_while_the_audit_file_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        locked_during_read = _locked_during_digest(monkeypatch, manifest_path)

        verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert locked_during_read == [True]

    def test_fully_sealed_log_covers_every_line(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2), _record("run-a", 3)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert coverage == LogCoverage(
            head=manifest_hash(_read_lines(manifest_path)[-1]),
            sealed_runs=2,
            sealed_lines=3,
            unattributed_lines=0,
            unsealed_lines={},
        )

    def test_head_is_what_verify_ndjson_log_returns(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert coverage.head == verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        assert coverage.head == manifest_hash(_read_lines(manifest_path)[-1])

    @pytest.mark.parametrize(
        ("run_id", "drop_key"),
        [(None, False), ("", False), ("   ", False), (None, True)],
        ids=["null", "empty", "blank", "absent"],
    )
    def test_lines_without_a_run_id_are_unattributed(self, tmp_path: Path, run_id: str | None, drop_key: bool) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        records = [_record(run_id, 1), _record("run-a", 2), _record(run_id, 3)]
        if drop_key:
            del records[0]["run_id"]
            del records[2]["run_id"]
        _write_records(audit_path, records)
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert coverage == LogCoverage(
            head=manifest_hash(_read_lines(manifest_path)[-1]),
            sealed_runs=1,
            sealed_lines=1,
            unattributed_lines=2,
            unsealed_lines={},
        )

    def test_lines_of_runs_that_are_not_sealed_yet_are_counted_per_run(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(
            audit_path,
            [_record("run-e", 7), _record("run-d", 8), _record("run-d", 9), _record("run-e", 10), _record("run-d", 11)],
        )

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert list(coverage.unsealed_lines.items()) == [("run-e", 2), ("run-d", 3)]

    def test_mixed_log_counts_every_kind_of_line_once(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        absent = _record(None, 10)
        del absent["run_id"]
        _write_records(
            audit_path,
            [_record("run-d", 7), _record("   ", 8), _record("run-e", 9), absent, _record("run-d", 11)],
        )

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert coverage.head == manifest_hash(_read_lines(manifest_path)[-1])
        assert coverage.sealed_runs == 3
        assert coverage.sealed_lines == 5
        assert coverage.unattributed_lines == 3
        assert list(coverage.unsealed_lines.items()) == [("run-d", 2), ("run-e", 1)]
        total = coverage.sealed_lines + coverage.unattributed_lines + sum(coverage.unsealed_lines.values())
        assert total == len(_read_lines(audit_path))

    @pytest.mark.parametrize("log_bytes", [None, b""], ids=["missing", "empty"])
    def test_missing_or_empty_manifest_log_leaves_every_line_unsealed_or_unattributed(
        self, tmp_path: Path, log_bytes: bytes | None
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-a", 2), _record(None, 3), _record("run-b", 4)])
        if log_bytes is not None:
            manifest_path.write_bytes(log_bytes)

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert coverage == LogCoverage(
            head=None, sealed_runs=0, sealed_lines=0, unattributed_lines=1, unsealed_lines={"run-a": 2, "run-b": 1}
        )
        assert manifest_path.exists() == (log_bytes is not None)

    @pytest.mark.parametrize("log_bytes", [None, b""], ids=["missing", "empty"])
    def test_missing_audit_file_without_manifests_is_an_empty_coverage(
        self, tmp_path: Path, log_bytes: bytes | None
    ) -> None:
        manifest_path = tmp_path / "manifests.ndjson"
        if log_bytes is not None:
            manifest_path.write_bytes(log_bytes)

        coverage = verify_ndjson_log_coverage(tmp_path / "audit.ndjson", manifest_path, signer=_signer())

        assert coverage == LogCoverage(
            head=None, sealed_runs=0, sealed_lines=0, unattributed_lines=0, unsealed_lines={}
        )

    def test_untouched_log_with_a_matching_expected_head_returns_the_coverage(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer(), expected_head=head)

        assert coverage.head == head
        assert coverage.sealed_runs == 3

    def test_edited_audit_line_of_a_sealed_run_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        assert records[0]["run_id"] == "run-a"
        records[0]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, records)

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

    def test_signer_with_a_different_key_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer(_OTHER_KEY))

    def test_previous_signers_verifies_coverage_of_a_log_sealed_before_rotation(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        expected_head = manifest_hash(_rotate(manifest_path, new_signer, old_signer))

        coverage = verify_ndjson_log_coverage(
            audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer]
        )

        assert coverage.head == expected_head
        assert coverage.sealed_runs == 3

    @pytest.mark.parametrize(
        ("seal_between", "sealed_runs"), [(True, 3), (False, 2)], ids=["seal-between", "back-to-back"]
    )
    def test_rotation_entries_are_not_sealed_runs_and_never_unsealed(
        self, tmp_path: Path, seal_between: bool, sealed_runs: int
    ) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path, seal_between=seal_between)
        _write_records(audit_path, [_record("run-z", 9)])

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])

        assert coverage == LogCoverage(
            head=manifest_hash(_read_lines(manifest_path)[-1]),
            sealed_runs=sealed_runs,
            sealed_lines=sealed_runs,
            unattributed_lines=0,
            unsealed_lines={"run-z": 1},
        )

    @pytest.mark.parametrize(
        "previous_signers",
        [[_signer()], [_signer(key_id="key-3"), _signer(_OTHER_KEY, "key-3")]],
        ids=["signer-and-previous", "two-previous"],
    )
    def test_previous_signers_with_a_duplicate_key_id_is_a_value_error(
        self, tmp_path: Path, previous_signers: list[ManifestSigner]
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            verify_ndjson_log_coverage(
                audit_path, manifest_path, signer=_signer(_OTHER_KEY), previous_signers=previous_signers
            )

        assert not isinstance(excinfo.value, ManifestVerificationError)

    def test_wrong_expected_head_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        with pytest.raises(ManifestVerificationError, match="head"):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer(), expected_head="ab" * 32)

    def test_deleted_audit_file_fails_every_sealed_run(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        audit_path.unlink()

        with pytest.raises(ManifestVerificationError, match="record"):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("run_id", [["x"], 5], ids=["list", "number"])
    def test_audit_record_whose_run_id_is_neither_a_string_nor_null_fails(self, tmp_path: Path, run_id: Any) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [{**_record(), "run_id": run_id}])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("file_name", ["audit.ndjson", "manifests.ndjson"])
    def test_json_line_that_is_not_an_object_fails(self, tmp_path: Path, file_name: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _append_line(tmp_path / file_name, "[]")

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

    def test_verification_waits_for_a_sealer_holding_the_lock(self, tmp_path: Path) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _sealed_log(tmp_path)

        with ThreadPoolExecutor(max_workers=1) as pool:
            fd = os.open(manifest_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                future = pool.submit(verify_ndjson_log_coverage, audit_path, manifest_path, signer=_signer())
                done, _ = wait([future], timeout=0.3)
            finally:
                os.close(fd)

            assert not done
            assert future.result(timeout=10).sealed_runs == 3

    def test_verify_ndjson_log_still_returns_only_the_head(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7), _record("", 8)])

        head = verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert isinstance(head, str)
        assert head == verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer()).head

    def test_log_coverage_is_frozen(self) -> None:
        coverage = LogCoverage(head=None, sealed_runs=0, sealed_lines=0, unattributed_lines=0, unsealed_lines={})

        with pytest.raises(dataclasses.FrozenInstanceError):
            coverage.sealed_runs = 1  # type: ignore[misc]

    def test_coverage_is_hashable_and_hash_respects_equality(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        first = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())
        second = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        hash(first)
        assert first == second
        assert hash(first) == hash(second)
        assert len({first, second}) == 1

    def test_coverage_survives_dataclasses_asdict(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        as_dict = dataclasses.asdict(coverage)

        assert as_dict["head"] == coverage.head
        assert as_dict["unsealed_lines"] == coverage.unsealed_lines

    def test_coverage_survives_a_deepcopy(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert copy.deepcopy(coverage) == coverage

    def test_coverage_survives_a_pickle_round_trip(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert pickle.loads(pickle.dumps(coverage)) == coverage  # nosec

    def test_unsealed_lines_returned_is_a_private_copy(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())
        coverage.unsealed_lines["run-z"] = 999

        again = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert "run-z" not in again.unsealed_lines


@_both_algorithms
class TestKeyRotationEntries:
    """Key order comes from the log's rotation entries, not from previous_signers."""

    @pytest.mark.parametrize("trailing_seal", [False, True], ids=["ends-with-entry", "seal-after-entry"])
    def test_a_hand_built_rotation_entry_verifies_and_the_next_line_chains_to_it(
        self, tmp_path: Path, trailing_seal: bool
    ) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        steps = [("seal", old_signer), ("rotate", new_signer), *([("seal", new_signer)] if trailing_seal else [])]
        audit_path, manifest_path = _build_log(tmp_path, *steps)

        head = verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert head == manifest_hash(_read_lines(manifest_path)[-1])

    def test_a_new_key_manifest_without_a_rotation_entry_is_rejected(self, tmp_path: Path) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        audit_path, manifest_path = _build_log(tmp_path, ("seal", old_signer), ("seal", new_signer))

        with pytest.raises(ManifestVerificationError, match="run-2") as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        _assert_names(excinfo, "key-2", "key-1")

    def test_a_retired_key_manifest_after_a_rotation_is_rejected_before_the_new_key_ever_sealed(
        self, tmp_path: Path
    ) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        steps = [("seal", old_signer), ("rotate", new_signer), ("seal", old_signer)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)

        with pytest.raises(ManifestVerificationError, match="run-3") as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        _assert_names(excinfo, "key-1", "key-2")

    @pytest.mark.parametrize("seal_between", [True, False], ids=["seal-between", "back-to-back"])
    @pytest.mark.parametrize("newest_first", [False, True], ids=["oldest-first", "newest-first"])
    def test_the_order_of_previous_signers_does_not_matter(
        self, tmp_path: Path, seal_between: bool, newest_first: bool
    ) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path, seal_between=seal_between)
        previous_signers = [key_2, key_1] if newest_first else [key_1, key_2]

        head = verify_ndjson_log(audit_path, manifest_path, signer=key_3, previous_signers=previous_signers)

        assert head == manifest_hash(_read_lines(manifest_path)[-1])

    @pytest.mark.parametrize("signed_by", ["active", "retired"])
    def test_a_rotation_entry_signed_by_the_active_or_a_retired_key_is_rejected(
        self, tmp_path: Path, signed_by: str
    ) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        signer = new_signer if signed_by == "active" else old_signer
        steps = [("seal", old_signer), ("rotate", new_signer), ("rotate", signer)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)

        with pytest.raises(ManifestVerificationError, match=signer.key_id):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    def test_a_rotation_entry_as_the_first_line_is_rejected(self, tmp_path: Path) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        audit_path, manifest_path = _build_log(tmp_path, ("seal", old_signer), ("rotate", new_signer))
        # The control: after a seal, a rotation entry is fine.
        verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        _rewrite_lines(manifest_path, [_rotation_entry(new_signer, None)])

        with pytest.raises(ManifestVerificationError, match="has no earlier manifest to rotate from"):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    @pytest.mark.parametrize(
        ("forge", "reason"),
        [(forge, _FORGED_ENTRY_REASONS[name]) for name, forge in _FORGED_ENTRIES.items()],
        ids=list(_FORGED_ENTRIES),
    )
    def test_a_forged_rotation_entry_is_rejected_and_never_skipped(
        self, tmp_path: Path, forge: Callable[[str], dict[str, Any]], reason: str
    ) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        audit_path, manifest_path = _build_log(tmp_path, ("seal", old_signer), ("rotate", new_signer))
        # The control: the unforged entry verifies.
        verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        seal, entry = _read_lines(manifest_path)
        _rewrite_lines(manifest_path, [seal, forge(entry["previous_manifest_hash"])])
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match=reason):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        with pytest.raises(ManifestVerificationError, match=reason):
            seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert manifest_path.read_bytes() == before

    def test_a_v2_rotation_entry_carries_the_co_signature_of_the_key_it_retires(self, tmp_path: Path) -> None:
        old_signer, new_signer, third = _signer(), _signer(_OTHER_KEY, "key-2"), _signer(_THIRD_KEY, "key-3")
        steps = [("seal", old_signer), ("rotate", new_signer), ("rotate", third)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)

        verify_ndjson_log(audit_path, manifest_path, signer=third, previous_signers=[old_signer, new_signer])

        _, first, second = _read_lines(manifest_path)
        assert [first["previous_key_signature"]["key_id"], second["previous_key_signature"]["key_id"]] == [
            "key-1",
            "key-2",
        ]

    def test_a_v1_rotation_without_a_co_signature_verifies_in_a_v1_prefix_before_a_v2_rotation(
        self, tmp_path: Path
    ) -> None:
        old_signer, new_signer, third = _signer(), _signer(_OTHER_KEY, "key-2"), _signer(_THIRD_KEY, "key-3")
        steps = [("seal-v1", old_signer), ("rotate-v1", new_signer), ("rotate", third)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)
        assert "previous_key_signature" not in _read_lines(manifest_path)[1]

        head = verify_ndjson_log(audit_path, manifest_path, signer=third, previous_signers=[old_signer, new_signer])

        assert head == manifest_hash(_read_lines(manifest_path)[-1])

    def test_sealing_skips_rotation_entries_like_sealed_runs(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path)
        before = manifest_path.read_bytes()

        assert seal_ndjson_runs(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2]) == []
        with pytest.raises(RunAlreadySealedError):
            seal_ndjson_runs(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2], run_id="run-a")

        assert manifest_path.read_bytes() == before


@_both_algorithms
class TestCurrentKeyRequirement:
    """Sealing and verifying need the log's current key to be the signer."""

    @_current_key_required
    def test_a_log_under_the_old_key_is_rejected_for_the_new_signer_even_with_the_old_key_as_previous_signer(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError, match=_under_key("key-1", "key-2")):
            call(audit_path, manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()])

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("call", [verify_ndjson_log, verify_ndjson_log_coverage], ids=["verify", "coverage"])
    def test_the_check_is_raised_before_the_collected_record_and_head_problems(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        records = _read_lines(audit_path)
        records[0]["tenant_id"] = "tenant-other"
        _rewrite_lines(audit_path, records)

        with pytest.raises(ManifestVerificationError, match=_under_key("key-1", "key-2")):
            call(
                audit_path,
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head="0" * 64,
            )

    @_current_key_required
    def test_an_entry_to_a_keyring_key_the_log_never_used_does_not_take_the_log_over(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        old_signer, new_signer, extra_signer = _signer(), _signer(_OTHER_KEY, "key-2"), _signer(b"x" * 32, "key-x")
        steps = [("seal", old_signer), ("rotate", new_signer), ("seal", new_signer), ("rotate", extra_signer)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError, match=_under_key("key-x", "key-2")):
            call(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer, extra_signer])

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("log_bytes", [None, b""], ids=["missing", "empty"])
    def test_an_empty_log_has_no_current_key_and_the_first_seal_defines_it(
        self, tmp_path: Path, log_bytes: bytes | None
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])
        if log_bytes is not None:
            manifest_path.write_bytes(log_bytes)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")

        assert verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer]) is None
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert manifests[0]["signature"]["key_id"] == "key-2"
        verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])


@_both_algorithms
class TestExpectedHead:
    """expected_head anchors the log outside itself."""

    def test_matching_expected_head_verifies_and_lets_sealing_continue(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _write_records(audit_path, [_record("run-d", 7)])

        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer(), expected_head=head) == head
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head)

        assert [(manifest["run_id"], manifest["previous_manifest_hash"]) for manifest in manifests] == [("run-d", head)]

    def test_truncated_log_fails_verification_against_the_expected_head(self, tmp_path: Path) -> None:
        audit_path, manifest_path, head = _truncated_log(tmp_path)

        with pytest.raises(ManifestVerificationError, match="head"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), expected_head=head)

    def test_truncated_log_stops_sealing_against_the_expected_head(self, tmp_path: Path) -> None:
        audit_path, manifest_path, head = _truncated_log(tmp_path)
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match="head"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head)

        assert manifest_path.read_bytes() == before

    def test_truncated_log_verifies_without_an_expected_head(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _truncated_log(tmp_path)

        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_after_a_rotation_the_entry_hash_anchors_and_an_older_head_does_not(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        old_head = manifest_hash(_read_lines(manifest_path)[-1])
        entry_head = manifest_hash(_rotate(manifest_path, new_signer, old_signer))
        _write_records(audit_path, [_record("run-d", 7)])

        assert (
            verify_ndjson_log(
                audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=entry_head
            )
            == entry_head
        )
        with pytest.raises(ManifestVerificationError, match="head"):
            verify_ndjson_log(
                audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=old_head
            )
        manifests = seal_ndjson_runs(
            audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=entry_head
        )
        assert [manifest["previous_manifest_hash"] for manifest in manifests] == [entry_head]


def _log_heads(manifest_path: Path) -> list[str]:
    return [manifest_hash(line) for line in _read_lines(manifest_path)]


_ANCHORED_CALLS: dict[str, Callable[..., Any]] = {
    "verify": lambda a, m, **kw: verify_ndjson_log(a, m, signer=_signer(), **kw),
    "coverage": lambda a, m, **kw: verify_ndjson_log_coverage(a, m, signer=_signer(), **kw),
    "seal": lambda a, m, **kw: seal_ndjson_runs(a, m, signer=_signer(), **kw),
}
_each_anchored_call = pytest.mark.parametrize("call", list(_ANCHORED_CALLS.values()), ids=list(_ANCHORED_CALLS))


@_both_algorithms
class TestAnchoredHeads:
    """anchored_heads: every anchor must be the hash of a verified line, so a rolled back or replaced log is caught."""

    @_each_anchored_call
    def test_every_line_hash_of_the_log_passes_as_an_anchor(self, tmp_path: Path, call: Callable[..., Any]) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        heads = _log_heads(manifest_path)
        assert len(heads) == 4

        call(audit_path, manifest_path, log_id="log-a", anchored_heads=heads)

    @_each_anchored_call
    def test_the_predecessor_head_of_a_successor_genesis_passes_as_an_anchor(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        audit_path, manifest_path = _successor_log(tmp_path)

        call(audit_path, manifest_path, log_id="log-a", anchored_heads=[_PREDECESSOR, *_log_heads(manifest_path)])

    @_each_anchored_call
    def test_an_anchored_heads_iterator_is_accepted(self, tmp_path: Path, call: Callable[..., Any]) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        call(audit_path, manifest_path, anchored_heads=iter(_log_heads(manifest_path)))

    @_each_anchored_call
    def test_no_anchored_heads_keeps_todays_behaviour(self, tmp_path: Path, call: Callable[..., Any]) -> None:
        audit_path, manifest_path, _ = _truncated_log(tmp_path)

        call(audit_path, manifest_path, anchored_heads=[])

    @_each_anchored_call
    def test_an_unknown_anchor_fails_and_is_named(self, tmp_path: Path, call: Callable[..., Any]) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        unknown = "0" * 64

        with pytest.raises(ManifestVerificationError) as excinfo:
            call(audit_path, manifest_path, anchored_heads=[unknown])

        _assert_names(excinfo, unknown)

    @_each_anchored_call
    def test_one_unknown_anchor_among_valid_ones_fails(self, tmp_path: Path, call: Callable[..., Any]) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        unknown = "1" * 64

        with pytest.raises(ManifestVerificationError) as excinfo:
            call(audit_path, manifest_path, anchored_heads=[*_log_heads(manifest_path), unknown])

        _assert_names(excinfo, unknown)

    def test_a_rotation_entry_hash_passes_as_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        old_head = manifest_hash(_read_lines(manifest_path)[-1])
        entry_head = manifest_hash(_rotate(manifest_path, new_signer, old_signer))

        head = verify_ndjson_log(
            audit_path,
            manifest_path,
            signer=new_signer,
            previous_signers=[old_signer],
            anchored_heads=[old_head, entry_head],
        )

        assert head == entry_head

    def test_an_anchor_that_is_the_hash_of_a_forged_unsigned_manifest_fails(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        forged = {**_read_lines(manifest_path)[-1]}
        del forged["signature"]
        forged["run_id"] = "run-never-sealed"
        anchor = manifest_hash(forged)

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), anchored_heads=[anchor])

        _assert_names(excinfo, anchor)

    def test_expected_head_keeps_equality_semantics_next_to_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        heads = _log_heads(manifest_path)

        # Control: an older head is a valid anchor but not the expected head.
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), anchored_heads=[heads[0]])
        with pytest.raises(ManifestVerificationError, match="head"):
            verify_ndjson_log(
                audit_path, manifest_path, signer=_signer(), expected_head=heads[0], anchored_heads=[heads[0]]
            )

    def test_the_rollback_that_equality_alone_cannot_catch_is_caught_by_an_older_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_head = _log_heads(manifest_path)[-1]
        _rewrite_lines(manifest_path, _read_lines(manifest_path)[:1])
        # Growth after the rollback: the cut runs are sealed again, so the log is as long as before.
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), sealed_late=False)
        grown = verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        assert grown != old_head
        assert len(_read_lines(manifest_path)) == 3
        # Control: today's checks accept the regrown log, and it is not the old head either way.
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), expected_head=grown)

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), anchored_heads=[old_head])

        _assert_names(excinfo, old_head)

    @pytest.mark.parametrize("which", ["verify", "coverage"])
    def test_dropping_the_newest_run_with_its_records_is_detected(self, tmp_path: Path, which: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = _log_heads(manifest_path)[-1]
        _rewrite_lines(manifest_path, _read_lines(manifest_path)[:-1])
        _rewrite_lines(audit_path, [r for r in _read_lines(audit_path) if r["run_id"] != "run-c"])
        call = _ANCHORED_CALLS[which]
        # Control: the shortened log is a consistent log without the anchor.
        call(audit_path, manifest_path)

        with pytest.raises(ManifestVerificationError) as excinfo:
            call(audit_path, manifest_path, anchored_heads=[head])

        _assert_names(excinfo, head)

    @pytest.mark.parametrize("which", ["verify", "coverage"])
    def test_deleting_both_files_is_detected_by_an_anchor(self, tmp_path: Path, which: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = _log_heads(manifest_path)[-1]
        audit_path.unlink()
        manifest_path.unlink()

        with pytest.raises(ManifestVerificationError) as excinfo:
            _ANCHORED_CALLS[which](audit_path, manifest_path, anchored_heads=[head])

        _assert_names(excinfo, head)

    def test_an_empty_manifest_log_fails_with_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = _log_heads(manifest_path)[-1]
        manifest_path.write_bytes(b"")

        for call in (
            lambda: verify_ndjson_log(audit_path, manifest_path, signer=_signer(), anchored_heads=[head]),
            lambda: seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), anchored_heads=[head]),
            lambda: rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), expected_head=None, anchored_heads=[head]
            ),
        ):
            with pytest.raises(ManifestVerificationError) as excinfo:
                call()
            _assert_names(excinfo, head)
        assert manifest_path.read_bytes() == b""

    def test_a_missing_manifest_log_fails_sealing_with_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = _log_heads(manifest_path)[-1]
        manifest_path.unlink()

        with pytest.raises(ManifestVerificationError) as excinfo:
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), anchored_heads=[head])

        _assert_names(excinfo, head)
        assert not manifest_path.exists() or not manifest_path.read_bytes()

    def test_substituting_another_log_signed_with_the_same_key_is_detected(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        _, manifest_path = _sealed_log(real)
        head = _log_heads(manifest_path)[-1]
        other = tmp_path / "other"
        other.mkdir()
        other_audit, other_manifest = _build_log(other, ("seal", _signer()), ("seal", _signer()), ("seal", _signer()))
        # Control: the substitute is a valid log under the same key.
        verify_ndjson_log(other_audit, other_manifest, signer=_signer())

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(other_audit, other_manifest, signer=_signer(), anchored_heads=[head])

        _assert_names(excinfo, head)

    def test_rolling_back_then_sealing_again_refuses_and_writes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path, old_head = _truncated_log(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError) as excinfo:
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), anchored_heads=[old_head])

        _assert_names(excinfo, old_head)
        assert _snapshot(tmp_path) == before

    def test_rotation_refuses_a_rolled_back_log_with_an_anchor_and_writes_nothing(self, tmp_path: Path) -> None:
        _, manifest_path, old_head = _truncated_log(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError) as excinfo:
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=None,
                anchored_heads=[old_head],
            )

        _assert_names(excinfo, old_head)
        assert _snapshot(tmp_path) == before


@_both_algorithms
class TestHeadAnchor:
    """The HeadAnchor protocol, NdjsonHeadAnchor and the head_anchor hook of sealing, rotation and quarantine."""

    def test_write_appends_a_head_and_anchored_at_line_and_latest_returns_the_last(self, tmp_path: Path) -> None:
        anchor = _ndjson_anchor(tmp_path / "anchor.ndjson")

        anchor.write("a" * 64)
        anchor.write("b" * 64)

        lines = _read_lines(tmp_path / "anchor.ndjson")
        assert [line["head"] for line in lines] == ["a" * 64, "b" * 64]
        for line in lines:
            assert set(line) == {"head", "anchored_at"}
            assert line["anchored_at"].endswith("Z")
            datetime.fromisoformat(line["anchored_at"].removesuffix("Z"))
        assert anchor.latest() == "b" * 64

    def test_write_appends_to_an_existing_anchor_file_and_keeps_it_owner_only(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.ndjson"
        _rewrite_lines(path, [{"head": "a" * 64, "anchored_at": "2026-01-01T00:00:00.000000Z"}])

        _ndjson_anchor(path).write("b" * 64)

        assert [line["head"] for line in _read_lines(path)] == ["a" * 64, "b" * 64]
        fresh = tmp_path / "fresh.ndjson"
        _ndjson_anchor(fresh).write("c" * 64)
        assert stat.S_IMODE(fresh.stat().st_mode) == 0o600

    def test_latest_is_none_for_a_missing_or_empty_file(self, tmp_path: Path) -> None:
        assert _ndjson_anchor(tmp_path / "missing.ndjson").latest() is None
        (tmp_path / "empty.ndjson").write_bytes(b"")
        assert _ndjson_anchor(tmp_path / "empty.ndjson").latest() is None

    def test_latest_ignores_a_torn_last_line_and_returns_the_last_terminated_head(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.ndjson"
        _rewrite_lines(path, [{"head": "a" * 64, "anchored_at": "2026-01-01T00:00:00.000000Z"}])
        _torn(path, b'{"head": "bb')

        assert _ndjson_anchor(path).latest() == "a" * 64

    def test_latest_is_none_when_the_only_line_is_torn(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.ndjson"
        path.write_bytes(b'{"head": "bb')

        assert _ndjson_anchor(path).latest() is None

    @pytest.mark.parametrize("earlier", [1, 5000], ids=["one-line", "many-lines"])
    def test_latest_ignores_an_oversized_last_line_like_a_torn_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, earlier: int
    ) -> None:
        path = tmp_path / "anchor.ndjson"
        if earlier == 1:
            _rewrite_lines(path, [{"head": "a" * 64, "anchored_at": "2026-01-01T00:00:00.000000Z"}])
            previous = "a" * 64
        else:
            self._many_heads(path, earlier)
            previous = f"{earlier - 1:064x}"
        _cap(monkeypatch)
        _append_line(path, _oversized_line({"head": "b" * 64}))

        head, seen = self._latest_with_read_spy(path, monkeypatch)

        assert head == previous
        if earlier > 1:
            assert sum(seen) <= 64 * 1024

    def test_latest_is_none_when_the_only_line_is_oversized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "anchor.ndjson"
        _cap(monkeypatch)
        _append_line(path, _oversized_line({"head": "b" * 64}))

        assert _ndjson_anchor(path).latest() is None

    @staticmethod
    def _latest_with_read_spy(path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str | None, list[int]]:
        seen: list[int] = []
        real_open = open

        def spy_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            handle = real_open(file, mode, *args, **kwargs)
            return _ReadSpy(handle, seen) if mode == "rb" else handle

        _patch_bindings(monkeypatch, "open", spy_open)
        return _ndjson_anchor(path).latest(), seen

    @staticmethod
    def _many_heads(path: Path, count: int) -> None:
        _rewrite_lines(
            path,
            [{"head": f"{number:064x}", "anchored_at": "2026-01-01T00:00:00.000000Z"} for number in range(count)],
        )

    def test_latest_reads_a_bounded_tail_not_the_whole_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "anchor.ndjson"
        self._many_heads(path, 5000)
        assert path.stat().st_size > 400_000

        head, seen = self._latest_with_read_spy(path, monkeypatch)

        assert head == f"{4999:064x}"
        assert seen
        assert sum(seen) <= 64 * 1024

    def test_latest_read_volume_does_not_grow_with_the_number_of_earlier_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        small, large = tmp_path / "small.ndjson", tmp_path / "large.ndjson"
        self._many_heads(small, 50)
        self._many_heads(large, 5000)

        _, seen_small = self._latest_with_read_spy(small, monkeypatch)
        _, seen_large = self._latest_with_read_spy(large, monkeypatch)

        assert sum(seen_large) <= max(sum(seen_small), 1) + 32 * 1024

    def test_latest_ignores_a_torn_tail_after_many_lines_without_reading_them_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "anchor.ndjson"
        self._many_heads(path, 5000)
        _torn(path, b'{"head": "bb')

        head, seen = self._latest_with_read_spy(path, monkeypatch)

        assert head == f"{4999:064x}"
        assert sum(seen) <= 64 * 1024

    def test_latest_returns_a_last_line_longer_than_any_read_chunk(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.ndjson"
        self._many_heads(path, 20)
        long_head = "h" * 100_000
        _append_line(path, json.dumps({"head": long_head, "anchored_at": "2026-01-01T00:00:00.000000Z"}).encode())

        assert _ndjson_anchor(path).latest() == long_head

    def test_latest_returns_the_previous_long_head_when_a_long_tail_is_torn(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.ndjson"
        self._many_heads(path, 20)
        long_head = "h" * 100_000
        _append_line(path, json.dumps({"head": long_head, "anchored_at": "2026-01-01T00:00:00.000000Z"}).encode())
        _torn(path, b'{"head": "' + b"t" * 50_000)

        assert _ndjson_anchor(path).latest() == long_head

    @pytest.mark.parametrize("line", [b'{"head": 5}', b'{"anchored_at": "x"}'], ids=["non-string", "missing"])
    def test_latest_fails_closed_for_a_terminated_line_without_a_string_head(self, tmp_path: Path, line: bytes) -> None:
        path = tmp_path / "anchor.ndjson"
        _rewrite_lines(path, [{"head": "a" * 64, "anchored_at": "2026-01-01T00:00:00.000000Z"}])
        _append_line(path, line)

        with pytest.raises(ManifestVerificationError, match=re.escape(str(path))):
            _ndjson_anchor(path).latest()

    def test_write_fsyncs_the_anchor_file_and_its_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "anchor.ndjson"
        real_fsync = os.fsync
        fsynced: list[str] = []

        def spy_fsync(fd: int) -> None:
            if os.path.samestat(os.fstat(fd), path.stat()):
                fsynced.append("file")
            if os.path.samestat(os.fstat(fd), tmp_path.stat()):
                fsynced.append("dir")
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", spy_fsync)

        _ndjson_anchor(path).write("a" * 64)

        assert fsynced == ["file", "dir"]

    @pytest.mark.parametrize(
        "last",
        [raw for name, raw in _DAMAGED_LINES.items() if name != "blank"],
        ids=[n for n in _DAMAGED_LINES if n != "blank"],
    )
    def test_latest_fails_closed_for_an_undecodable_last_line(self, tmp_path: Path, last: bytes) -> None:
        path = tmp_path / "anchor.ndjson"
        _rewrite_lines(path, [{"head": "a" * 64, "anchored_at": "2026-01-01T00:00:00.000000Z"}])
        _append_line(path, last.rstrip(b"\n"))

        with pytest.raises(ManifestVerificationError, match=re.escape(str(path))):
            _ndjson_anchor(path).latest()

    def test_seal_writes_the_new_head_after_the_durable_append(self, tmp_path: Path) -> None:
        audit_path, manifest_path = tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])
        anchor = _RecordingAnchor(manifest_path)

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), head_anchor=anchor)

        assert anchor.heads == [manifest_hash(manifests[-1])] == anchor.last_lines_seen

    def test_seal_with_a_genesis_writes_the_hash_of_the_last_appended_line(self, tmp_path: Path) -> None:
        audit_path, manifest_path = tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])
        anchor = _RecordingAnchor(manifest_path)

        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a", head_anchor=anchor)

        lines = _read_lines(manifest_path)
        assert [line.get("kind") for line in lines] == ["genesis", None]
        assert anchor.heads == [manifest_hash(lines[-1])]

    def test_seal_with_nothing_to_append_does_not_write_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        anchor = _RecordingAnchor()

        assert seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), head_anchor=anchor) == []

        assert anchor.heads == []

    def test_a_refused_seal_does_not_write_an_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path, head = _truncated_log(tmp_path)
        anchor = _RecordingAnchor()

        with pytest.raises(ManifestVerificationError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head, head_anchor=anchor)

        assert anchor.heads == []

    def test_a_failing_anchor_write_leaves_the_seal_on_disk_and_propagates(self, tmp_path: Path) -> None:
        audit_path, manifest_path = tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])

        with pytest.raises(RuntimeError, match="anchor down"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), head_anchor=_RecordingAnchor(fail=True))

        assert [line["run_id"] for line in _read_lines(manifest_path)] == ["run-a", "run-b"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_rotation_writes_the_entry_hash_after_the_durable_append(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        head = _log_heads(manifest_path)[-1]
        anchor = _RecordingAnchor(manifest_path)

        entry = rotate_manifest_key(
            manifest_path,
            signer=_signer(_OTHER_KEY, "key-2"),
            previous_signers=[_signer()],
            expected_head=head,
            head_anchor=anchor,
        )

        assert anchor.heads == [manifest_hash(entry)] == anchor.last_lines_seen

    def test_a_failing_anchor_write_leaves_the_rotation_on_disk_and_propagates(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        head = _log_heads(manifest_path)[-1]

        with pytest.raises(RuntimeError, match="anchor down"):
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=head,
                head_anchor=_RecordingAnchor(fail=True),
            )

        assert _read_lines(manifest_path)[-1]["kind"] == "key_rotation"

    def test_a_rotation_that_fails_does_not_write_an_anchor(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        anchor = _RecordingAnchor()

        with pytest.raises(KeyAlreadyCurrentError):
            rotate_manifest_key(manifest_path, signer=_signer(), expected_head=None, head_anchor=anchor)

        assert anchor.heads == []

    def test_an_ndjson_anchor_catches_a_later_rollback(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        anchor = _ndjson_anchor(tmp_path / "anchor.ndjson")
        _write_records(audit_path, [_record("run-d", 7)])
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), head_anchor=anchor)
        latest = anchor.latest()
        assert latest == _log_heads(manifest_path)[-1]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer(), anchored_heads=[latest]) == latest

        _rewrite_lines(manifest_path, _read_lines(manifest_path)[:-1])

        with pytest.raises(ManifestVerificationError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), anchored_heads=[anchor.latest()])
        _assert_names(excinfo, str(latest))

    @_both_recoveries
    def test_a_repair_writes_the_hash_of_the_last_trace_line_before_touching_the_logs(
        self, tmp_path: Path, recovery: _Recovery
    ) -> None:
        manifest_path, repair, _ = recovery(tmp_path)
        before = _snapshot(tmp_path)
        anchor = _RecordingAnchor(_trace(tmp_path))
        repaired_when_written: list[bool] = []
        write = anchor.write

        def write_and_look(head: str) -> None:
            on_disk = {name: data for name, data in _snapshot(tmp_path).items() if name != _trace(tmp_path).name}
            repaired_when_written.append(on_disk != before)
            write(head)

        anchor.write = write_and_look  # type: ignore[method-assign]

        repair(tmp_path, head_anchor=anchor)

        assert anchor.heads == [_sha256(_canonical(_read_lines(_trace(tmp_path))[-1]))] == anchor.last_lines_seen
        assert repaired_when_written == [False]
        assert manifest_path.read_bytes() != before[manifest_path.name]

    @_both_recoveries
    def test_a_dry_run_writes_no_anchor(self, tmp_path: Path, recovery: _Recovery) -> None:
        _, repair, _ = recovery(tmp_path)
        anchor = _RecordingAnchor()

        repair(tmp_path, dry_run=True, head_anchor=anchor)

        assert anchor.heads == []

    def test_nothing_to_repair_writes_no_anchor(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        anchor = _RecordingAnchor()

        assert _quarantine(tmp_path, audit_path, manifest_path, head_anchor=anchor) == []

        assert anchor.heads == []

    @_both_recoveries
    def test_a_refused_repair_writes_no_anchor(self, tmp_path: Path, recovery: _Recovery) -> None:
        _, repair, _ = recovery(tmp_path)
        _rewrite_lines(_trace(tmp_path), _trace_chain(_signer(_OTHER_KEY, "key-2")))
        anchor = _RecordingAnchor()

        _assert_raises_and_unchanged(tmp_path, lambda: repair(tmp_path, head_anchor=anchor))

        assert anchor.heads == []

    def test_a_repair_that_raises_after_the_trace_append_has_anchored_the_trace(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")
        _insert_line(audit_path, 5, b"\n")
        anchor = _RecordingAnchor()

        with patch("os.replace", side_effect=OSError(errno.EIO, "replace failed")):
            with pytest.raises(OSError, match="replace failed"):
                _quarantine(tmp_path, audit_path, manifest_path, head_anchor=anchor)

        assert _trace(tmp_path).exists()
        assert anchor.heads == [_sha256(_canonical(_read_lines(_trace(tmp_path))[-1]))]

    @_both_recoveries
    def test_a_failing_anchor_write_keeps_the_trace_leaves_the_logs_untouched_and_propagates(
        self, tmp_path: Path, recovery: _Recovery
    ) -> None:
        manifest_path, repair, dropped = recovery(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(RuntimeError, match="anchor down"):
            repair(tmp_path, head_anchor=_RecordingAnchor(fail=True))

        assert _verify_quarantine(_trace(tmp_path), _signer()) is not None
        assert len(_read_lines(_trace(tmp_path))) == dropped
        assert manifest_path.read_bytes() == before[manifest_path.name]
        assert {n: d for n, d in _snapshot(tmp_path).items() if n != _trace(tmp_path).name} == {
            n: d for n, d in before.items() if n != _trace(tmp_path).name
        }

    @_both_recoveries
    def test_an_ndjson_anchor_catches_a_later_trace_rollback(self, tmp_path: Path, recovery: _Recovery) -> None:
        _, repair, _ = recovery(tmp_path)
        anchor = _ndjson_anchor(tmp_path / "anchor.ndjson")

        repair(tmp_path, head_anchor=anchor)

        latest = anchor.latest()
        assert latest is not None
        assert _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=[latest]) == latest
        _rewrite_lines(_trace(tmp_path), _read_lines(_trace(tmp_path))[:-1])
        with pytest.raises(ManifestVerificationError) as excinfo:
            _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=[latest])
        _assert_names(excinfo, latest)


@_both_algorithms
class TestGenesis:
    """A log opened with a log_id starts with a signed genesis line that names it."""

    def test_the_first_seal_writes_a_genesis_with_exactly_the_expected_keys_and_chains_onto_it(
        self, tmp_path: Path
    ) -> None:
        _, manifest_path = _genesis_log(tmp_path)

        genesis, *seals = _read_lines(manifest_path)

        assert set(genesis) == _EXPECTED_GENESIS_KEYS
        assert (genesis["manifest_version"], genesis["kind"], genesis["log_id"]) == (2, "genesis", "log-a")
        assert genesis["previous_manifest_hash"] is None
        assert genesis["created_at"].endswith("Z")
        datetime.fromisoformat(genesis["created_at"].removesuffix("Z"))
        signature = genesis["signature"]
        assert set(signature) == {"algorithm", "key_id", "value"}
        assert (signature["algorithm"], signature["key_id"]) == (_signer().algorithm, "key-1")
        assert signature["value"] == _reference_signature(_KEY, _signing_payload_ref(genesis))
        assert [seal["run_id"] for seal in seals] == ["run-a", "run-b", "run-c"]
        assert seals[0]["previous_manifest_hash"] == manifest_hash(genesis)

    def test_nothing_is_written_when_there_is_nothing_to_seal(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        audit_path.write_bytes(b"")

        assert seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a") == []

        # The seal lock creates an empty log (see the fresh-manifest-path test); only a genesis line must be absent.
        assert not manifest_path.exists() or manifest_path.read_bytes() == b""

    def test_a_later_seal_batch_adds_no_second_genesis(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])

        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a")

        assert [line.get("kind") for line in _read_lines(manifest_path)] == ["genesis", None, None, None, None]

    def test_a_genesis_log_verifies_with_and_without_the_log_id(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])

        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a") == head
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer(), log_id="log-a")
        assert (coverage.head, coverage.sealed_runs) == (head, 3)

    def test_the_genesis_hash_is_a_valid_expected_head_for_a_genesis_only_log(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        audit_path.write_bytes(b"")
        genesis = _genesis_entry(_signer())
        _write_records(manifest_path, [genesis])
        head = manifest_hash(genesis)

        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer(), expected_head=head) == head
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a") == head

    def test_a_genesis_only_log_takes_the_first_seal_chained_onto_it(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        genesis = _genesis_entry(_signer())
        _write_records(manifest_path, [genesis])
        _write_records(audit_path, [_record("run-a", 1)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a")

        assert [manifest["previous_manifest_hash"] for manifest in manifests] == [manifest_hash(genesis)]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a")

    def test_the_genesis_sets_the_active_key(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = manifest_path.read_bytes()
        other = _signer(_OTHER_KEY, "key-2")

        with pytest.raises(ManifestVerificationError, match=_under_key("key-1", "key-2")):
            seal_ndjson_runs(audit_path, manifest_path, signer=other, previous_signers=[_signer()], log_id="log-a")

        assert manifest_path.read_bytes() == before

    def test_a_genesis_only_log_can_be_rotated_and_sealed_under_the_new_key(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        genesis = _genesis_entry(old_signer)
        _write_records(manifest_path, [genesis])

        entry = rotate_manifest_key(
            manifest_path,
            signer=new_signer,
            previous_signers=[old_signer],
            expected_head=manifest_hash(genesis),
            log_id="log-a",
        )

        assert entry["previous_manifest_hash"] == manifest_hash(genesis)
        assert entry["previous_key_signature"]["key_id"] == "key-1"
        _write_records(audit_path, [_record("run-a", 1)])
        manifests = seal_ndjson_runs(
            audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], log_id="log-a"
        )
        head = verify_ndjson_log(
            audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], log_id="log-a"
        )
        assert head == manifest_hash(manifests[-1])

    def test_a_genesis_headed_log_rotates_after_sealing_and_still_verifies_by_log_id(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")

        rotate_manifest_key(
            manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=None, log_id="log-a"
        )

        verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], log_id="log-a")

    @pytest.mark.parametrize("call", [verify_ndjson_log, verify_ndjson_log_coverage, seal_ndjson_runs])
    def test_another_log_id_under_the_same_key_is_rejected(self, tmp_path: Path, call: Callable[..., Any]) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path, log_id="log-a")
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError) as excinfo:
            call(audit_path, manifest_path, signer=_signer(), log_id="log-b")

        _assert_names(excinfo, "log-a", "log-b")
        assert _snapshot(tmp_path) == before

    def test_rotating_another_log_id_under_the_same_key_is_rejected(self, tmp_path: Path) -> None:
        _, manifest_path = _genesis_log(tmp_path, log_id="log-a")
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError) as excinfo:
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=None,
                log_id="log-b",
            )

        _assert_names(excinfo, "log-a", "log-b")
        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("call", [verify_ndjson_log, verify_ndjson_log_coverage, seal_ndjson_runs])
    def test_a_log_without_a_genesis_is_rejected_when_a_log_id_is_given(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError, match="genesis"):
            call(audit_path, manifest_path, signer=_signer(), log_id="log-a")

        assert _snapshot(tmp_path) == before

    def test_rotating_a_log_without_a_genesis_is_rejected_when_a_log_id_is_given(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match="genesis"):
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=None,
                log_id="log-a",
            )

        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("log_id", [None, "log-a"], ids=["no-log-id", "log-id"])
    @pytest.mark.parametrize("call", [verify_ndjson_log, verify_ndjson_log_coverage, seal_ndjson_runs])
    @pytest.mark.parametrize("position", ["after-a-seal", "second-genesis"])
    def test_a_genesis_anywhere_but_line_1_is_rejected(
        self, tmp_path: Path, call: Callable[..., Any], position: str, log_id: str | None
    ) -> None:
        if position == "second-genesis":
            audit_path, manifest_path = _genesis_log(tmp_path)
        else:
            audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _append_line(manifest_path, _canonical(_genesis_entry(_signer(), previous_manifest_hash=head)))
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError, match="genesis"):
            call(audit_path, manifest_path, signer=_signer(), **({} if log_id is None else {"log_id": log_id}))

        assert _snapshot(tmp_path) == before

    def test_rotating_a_log_with_a_misplaced_genesis_is_rejected(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _append_line(manifest_path, _canonical(_genesis_entry(_signer(), previous_manifest_hash=head)))
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match="genesis"):
            rotate_manifest_key(
                manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()], expected_head=None
            )

        assert manifest_path.read_bytes() == before

    def test_sealing_with_a_log_id_onto_a_non_empty_log_without_that_genesis_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError, match="genesis"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a")

        assert manifest_path.read_bytes() == before

    def test_sealing_without_a_log_id_onto_a_genesis_log_keeps_the_genesis_and_chains(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _write_records(audit_path, [_record("run-d", 7)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert [manifest["previous_manifest_hash"] for manifest in manifests] == [head]
        assert _read_lines(manifest_path)[0]["kind"] == "genesis"

    @pytest.mark.parametrize("log_id", ["", "   ", 5, b"log-a"], ids=["empty", "blank", "int", "bytes"])
    @pytest.mark.parametrize(
        "call", [verify_ndjson_log, verify_ndjson_log_coverage, seal_ndjson_runs], ids=["verify", "coverage", "seal"]
    )
    def test_a_blank_or_non_str_log_id_is_a_value_error_and_writes_nothing(
        self, tmp_path: Path, call: Callable[..., Any], log_id: Any
    ) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError, match="log_id") as excinfo:
            call(audit_path, manifest_path, signer=_signer(), log_id=log_id)

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("log_id", ["", "   ", 5], ids=["empty", "blank", "int"])
    def test_a_blank_or_non_str_log_id_is_a_value_error_for_a_rotation(self, tmp_path: Path, log_id: Any) -> None:
        _, manifest_path = _genesis_log(tmp_path)
        before = manifest_path.read_bytes()

        with pytest.raises(ValueError, match="log_id") as excinfo:
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=None,
                log_id=log_id,
            )

        assert not isinstance(excinfo.value, (ManifestVerificationError, KeyAlreadyCurrentError))
        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("log_id", [None, "log-b"], ids=["no-log-id", "new-id"])
    def test_a_genesis_with_a_changed_log_id_and_the_old_signature_fails(
        self, tmp_path: Path, log_id: str | None
    ) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        lines = _read_lines(manifest_path)
        lines[0]["log_id"] = "log-b"
        _rewrite_lines(manifest_path, lines)

        with pytest.raises(ManifestVerificationError, match=re.escape(_BAD_SIGNATURE)):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id=log_id)

    @pytest.mark.parametrize(
        "forged",
        [
            lambda: _genesis_entry(_signer(b"x" * 32, "key-1")),
            lambda: _genesis_entry(_signer(), created_at=_MISSING),
            lambda: _genesis_entry(_signer(), run_id="run-x"),
            lambda: _genesis_entry(_signer(), previous_manifest_hash="z" * 64),
            lambda: _genesis_entry(_signer(), previous_manifest_hash=5),
            lambda: _genesis_entry(_signer(), previous_manifest_hash="a" * 63),
            lambda: _genesis_entry(_signer(), previous_manifest_hash="A" * 64),
            lambda: _genesis_entry(_signer(), manifest_version=3),
            lambda: _genesis_entry(_signer(), log_id=""),
            lambda: _genesis_entry(_signer(), log_id=5),
        ],
        ids=[
            "wrong-key-material",
            "missing-key",
            "extra-key",
            "non-hex-previous-hash",
            "non-str-previous-hash",
            "short-previous-hash",
            "uppercase-previous-hash",
            "wrong-version",
            "blank-log-id",
            "non-str-log-id",
        ],
    )
    def test_a_malformed_or_forged_genesis_is_rejected(
        self, tmp_path: Path, forged: Callable[[], dict[str, Any]]
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        audit_path.write_bytes(b"")
        _write_records(manifest_path, [forged()])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("log_id", [None, "log-a"], ids=["no-log-id", "log-id"])
    def test_a_genesis_naming_a_predecessor_head_verifies_standalone(self, tmp_path: Path, log_id: str | None) -> None:
        audit_path, manifest_path = _successor_log(tmp_path)
        genesis, *seals = _read_lines(manifest_path)
        assert genesis["previous_manifest_hash"] == _PREDECESSOR

        head = verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id=log_id)

        assert head == manifest_hash(seals[-1])
        assert seals[0]["previous_manifest_hash"] == manifest_hash(genesis)

    def test_sealing_continues_a_successor_segment(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _successor_log(tmp_path)
        _write_records(audit_path, [_record("run-c", 3)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a")

        assert [manifest["run_id"] for manifest in manifests] == ["run-c"]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a") == manifest_hash(
            manifests[-1]
        )

    def test_a_successor_segment_rotates_its_key(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _successor_log(tmp_path)

        entry = _rotate(manifest_path, _signer(_OTHER_KEY, "key-2"), _signer())

        assert entry["kind"] == "key_rotation"
        head = verify_ndjson_log(
            audit_path, manifest_path, signer=_signer(_OTHER_KEY, "key-2"), previous_signers=[_signer()]
        )
        assert head == manifest_hash(entry)

    def test_deleting_both_files_and_sealing_with_a_log_id_but_no_anchor_silently_starts_a_new_log(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path = _genesis_log(tmp_path)
        audit_path.unlink()
        manifest_path.unlink()
        _write_records(audit_path, [_record("run-x", 1)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a")

        assert [manifest["run_id"] for manifest in manifests] == ["run-x"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a")
        assert _read_lines(manifest_path)[0]["kind"] == "genesis"


class TestMalformedNdjsonLines:
    """Ambiguous or non-object lines fail verification in both readers."""

    @pytest.mark.parametrize(("file_name", "key"), [("audit.ndjson", "tenant_id"), ("manifests.ndjson", "run_id")])
    def test_line_with_a_duplicate_json_key_fails_verification(self, tmp_path: Path, file_name: str, key: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        path = tmp_path / file_name
        lines = path.read_text(encoding="utf-8").splitlines()
        # Raw text: json.dumps cannot write a duplicate key.
        lines[0] = f'{{"{key}": "EVIL", {lines[0][1:]}'
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        assert json.loads(lines[0])[key] != "EVIL"

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize(
        ("file_name", "line"),
        [("audit.ndjson", "[]"), ("audit.ndjson", "null"), ("manifests.ndjson", '"x"'), ("manifests.ndjson", "5")],
    )
    def test_json_line_that_is_not_an_object_fails(self, tmp_path: Path, file_name: str, line: str) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _append_line(tmp_path / file_name, line)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        with pytest.raises(ManifestVerificationError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("run_id", [["x"], 5], ids=["list", "number"])
    def test_audit_record_whose_run_id_is_neither_a_string_nor_null_fails(self, tmp_path: Path, run_id: Any) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [{**_record(), "run_id": run_id}])

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        with pytest.raises(ManifestVerificationError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize(("file_name", "line_number"), [("audit.ndjson", 7), ("manifests.ndjson", 4)])
    @pytest.mark.parametrize(
        "line",
        [b"", b'{"run_id": "run-d"', b"\xff\xfe", b"[" * 100000],
        ids=["blank", "truncated", "bad-utf8", "deep-nesting"],
    )
    def test_undecodable_line_fails_naming_the_file_and_line(
        self, tmp_path: Path, file_name: str, line_number: int, line: bytes
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        path = tmp_path / file_name
        _append_line(path, line)
        location = re.escape(f"{path} line {line_number}")

        with pytest.raises(ManifestVerificationError, match=location):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        with pytest.raises(ManifestVerificationError, match=location):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize(("file_name", "line_number"), [("audit.ndjson", 7), ("manifests.ndjson", 4)])
    def test_oversized_line_fails_naming_the_file_and_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_name: str, line_number: int
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _cap(monkeypatch)
        path = tmp_path / file_name
        _append_line(path, _oversized_line())
        location = re.escape(f"{path} line {line_number}")

        with pytest.raises(ManifestVerificationError, match=location):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        with pytest.raises(ManifestVerificationError, match=location):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

    def test_max_line_bytes_defaults_to_64_mib(self) -> None:
        assert getattr(run_manifest_module, "MAX_LINE_BYTES", None) == 64 * 1024 * 1024

    def test_the_sealed_lookup_skips_an_oversized_line(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        _cap(monkeypatch)
        _append_line(manifest_path, _oversized_line({"run_id": "run-z"}))

        assert run_manifest_module._is_run_sealed_unverified(manifest_path, "run-z") is False
        assert run_manifest_module._is_run_sealed_unverified(manifest_path, "run-a") is True

    @pytest.mark.parametrize("reader", ["read-ndjson", "sealed-lookup", "anchor-latest", "quarantine"])
    def test_no_reader_reads_an_oversized_line_whole(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str
    ) -> None:
        cap = _cap(monkeypatch, 2_000_000)
        (tmp_path / "sealed").mkdir()
        _, manifest_path = _sealed_log(tmp_path / "sealed")
        path = tmp_path / "log.ndjson"
        path.write_bytes(b'{"head": "' + b"a" * 64 + b'"}\n' + _oversized_line({"run_id": "run-z"}, 3 * cap) + b"\n")
        seen: list[int] = []
        real_open = open

        def spy_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            handle = real_open(file, mode, *args, **kwargs)
            return _ReadSpy(handle, seen) if mode == "rb" else handle

        _patch_bindings(monkeypatch, "open", spy_open)
        calls: dict[str, Callable[[], object]] = {
            "read-ndjson": lambda: list(_verify._read_ndjson(path)),
            "sealed-lookup": lambda: run_manifest_module._is_run_sealed_unverified(path, "run-z"),
            "anchor-latest": lambda: NdjsonHeadAnchor(path).latest(),
            "quarantine": lambda: _quarantine(tmp_path, path, manifest_path, dry_run=True),
        }

        with suppress(ManifestVerificationError):
            calls[reader]()

        assert seen
        assert max(seen) <= cap + 1


class TestPathAliasingGuards:
    """audit_path and manifest_path must not resolve to the same file: that is a caller mistake, not tampering."""

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_seal_ndjson_runs_refuses_an_aliased_audit_and_manifest_path(self, tmp_path: Path, spelling: str) -> None:
        audit_path, _manifest_path = _sealed_log(tmp_path)
        manifest_path = _aliased(tmp_path, audit_path, spelling)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_verify_ndjson_log_refuses_an_aliased_audit_and_manifest_path(self, tmp_path: Path, spelling: str) -> None:
        audit_path, _manifest_path = _sealed_log(tmp_path)
        manifest_path = _aliased(tmp_path, audit_path, spelling)

        with pytest.raises(ValueError) as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert not isinstance(excinfo.value, ManifestVerificationError)

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_verify_ndjson_log_coverage_refuses_an_aliased_audit_and_manifest_path(
        self, tmp_path: Path, spelling: str
    ) -> None:
        audit_path, _manifest_path = _sealed_log(tmp_path)
        manifest_path = _aliased(tmp_path, audit_path, spelling)

        with pytest.raises(ValueError) as excinfo:
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())

        assert not isinstance(excinfo.value, ManifestVerificationError)

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_quarantine_damaged_lines_refuses_an_aliased_audit_and_manifest_path(
        self, tmp_path: Path, spelling: str
    ) -> None:
        audit_path, _manifest_path = _sealed_log(tmp_path)
        manifest_path = _aliased(tmp_path, audit_path, spelling)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            quarantine_damaged_lines(audit_path, manifest_path, quarantine_path=_trace(tmp_path), signer=_signer())

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_quarantine_damaged_lines_refuses_an_aliased_manifest_and_quarantine_path(
        self, tmp_path: Path, spelling: str
    ) -> None:
        """quarantine_path aliasing manifest_path would nest _flock(quarantine_path) inside _flock(manifest_path)
        on the same file: a self-deadlock, not tampering."""
        audit_path, manifest_path = _sealed_log(tmp_path)
        quarantine_path = _aliased(tmp_path, manifest_path, spelling)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            quarantine_damaged_lines(audit_path, manifest_path, quarantine_path=quarantine_path, signer=_signer())

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_quarantine_damaged_lines_refuses_an_aliased_audit_and_quarantine_path(
        self, tmp_path: Path, spelling: str
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        quarantine_path = _aliased(tmp_path, audit_path, spelling)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            quarantine_damaged_lines(audit_path, manifest_path, quarantine_path=quarantine_path, signer=_signer())

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("spelling", ["same-path", "symlink"])
    def test_quarantine_from_rotation_entry_refuses_an_aliased_manifest_and_quarantine_path(
        self, tmp_path: Path, spelling: str
    ) -> None:
        _, manifest_path, head = _log_with_rotation_entry(tmp_path)
        quarantine_path = _aliased(tmp_path, manifest_path, spelling)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            quarantine_from_rotation_entry(
                manifest_path, quarantine_path=quarantine_path, signer=_signer(), expected_head=head
            )

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before


@_both_algorithms
class TestQuarantineDamagedLines:
    def test_quarantined_line_is_frozen(self) -> None:
        line = QuarantinedLine(file="audit", line=1, offset=0, length=1, sha256="0" * 64, reason="not valid JSON")

        with pytest.raises(dataclasses.FrozenInstanceError):
            line.line = 2  # type: ignore[misc]

    def test_quarantined_line_has_exactly_the_specified_fields(self) -> None:
        assert [field.name for field in dataclasses.fields(QuarantinedLine)] == [
            "file",
            "line",
            "offset",
            "length",
            "sha256",
            "reason",
        ]

    def test_torn_manifest_tail_is_reported_by_a_dry_run_and_nothing_changes(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path, dry_run=True)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [4])
        assert all(item.reason for item in removed)
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("anchored", [False, True], ids=["unanchored", "anchored"])
    def test_torn_manifest_tail_is_truncated_and_the_chain_continues(self, tmp_path: Path, anchored: bool) -> None:
        audit_path, manifest_path, head = _torn_manifest_log(tmp_path)
        before = _snapshot(tmp_path)
        manifest_before = before["manifests.ndjson"]

        removed = _quarantine(tmp_path, audit_path, manifest_path, expected_head=head if anchored else None)

        assert _summary(removed) == _spans("manifest", manifest_before, [4])
        assert manifest_path.read_bytes() == manifest_before[: manifest_before.rindex(b"\n") + 1]
        assert audit_path.read_bytes() == before["audit.ndjson"]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head
        _write_records(audit_path, [_record("run-d", 7)])
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head)
        assert [(manifest["run_id"], manifest["previous_manifest_hash"]) for manifest in manifests] == [("run-d", head)]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == manifest_hash(manifests[-1])

    def test_torn_manifest_tail_of_a_successor_segment_is_truncated(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _successor_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        before = manifest_path.read_bytes()
        _torn(manifest_path, b'{"manifest_version": 2, "run_')

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert [item.file for item in removed] == ["manifest"]
        assert manifest_path.read_bytes() == before
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head

    def test_trace_has_one_entry_per_removed_line_in_removal_order(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert len(removed) == len(entries) == 3
        assert [
            (entry["file"], entry["line"], entry["offset"], entry["length"], entry["sha256"]) for entry in entries
        ] == [
            *_spans("manifest", before["manifests.ndjson"], [4]),
            *_spans("audit", before["audit.ndjson"], [4, 8]),
        ]
        _assert_traced(
            tmp_path, manifest_path, before["manifests.ndjson"], removed, audit=(audit_path, before["audit.ndjson"])
        )

    def test_trace_keeps_the_removed_bytes_so_they_can_be_inspected_or_restored(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        before = _snapshot(tmp_path)
        audit_lines = before["audit.ndjson"].splitlines(keepends=True)
        manifest_lines = before["manifests.ndjson"].splitlines(keepends=True)

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        _assert_traced(
            tmp_path, manifest_path, before["manifests.ndjson"], removed, audit=(audit_path, before["audit.ndjson"])
        )
        entries = _read_lines(_trace(tmp_path))
        removed_bytes = [manifest_lines[3], audit_lines[3], audit_lines[7]]
        assert [base64.b64decode(entry["raw_base64"], validate=True) for entry in entries] == removed_bytes
        assert [entry["sha256"] for entry in entries] == [_sha256(raw) for raw in removed_bytes]

    def test_trace_signature_covers_the_entry_without_its_signature(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert removed
        _assert_traced(
            tmp_path, manifest_path, before["manifests.ndjson"], removed, audit=(audit_path, before["audit.ndjson"])
        )
        for entry in _read_lines(_trace(tmp_path)):
            payload = _signing_payload_ref(entry)
            signature = entry["signature"]
            assert set(signature) == {"algorithm", "key_id", "value"}
            assert (signature["algorithm"], signature["key_id"]) == (_signer().algorithm, _signer().key_id)
            assert signature["value"] == _reference_signature(_KEY, payload)
            tampered = _signing_payload_ref({**entry, "line": entry["line"] + 1})
            assert _signer().verify(tampered, entry["signature"]["value"]) is False

    def test_quarantined_at_is_rfc3339_utc(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)

        _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert entries
        for entry in entries:
            quarantined_at = entry["quarantined_at"]
            assert quarantined_at.endswith("Z")
            datetime.fromisoformat(quarantined_at.removesuffix("Z"))

    @pytest.mark.parametrize("position", ["terminated", "unterminated"])
    def test_an_oversized_audit_line_is_dropped_and_its_trace_entry_stays_within_the_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, position: str
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _cap(monkeypatch)
        if position == "terminated":
            big = _oversized_line() + b"\n"
            _insert_line(audit_path, 3, big)
            number = 4
        else:
            big = b"x" * (_CAP + 1)
            _torn(audit_path, big)
            number = 7
        before = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", before, [number])
        assert audit_path.read_bytes() == before.replace(big, b"")
        trace_lines = _trace(tmp_path).read_bytes().splitlines()
        assert [len(line) <= _CAP for line in trace_lines] == [True]
        entry = json.loads(trace_lines[0])
        assert entry.get("raw_base64") is None
        assert (entry["sha256"], entry["length"]) == (_sha256(big), len(big))
        assert _verify_quarantine(_trace(tmp_path), _signer()) is not None

    def test_new_trace_file_has_owner_only_permissions(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)

        _quarantine(tmp_path, audit_path, manifest_path)

        assert stat.S_IMODE(_trace(tmp_path).stat().st_mode) == 0o600

    def test_a_second_recovery_appends_to_the_trace(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        first = _trace(tmp_path).read_bytes()
        _append_line(audit_path, b"[]")
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [7])
        trace = _trace(tmp_path).read_bytes()
        assert trace.startswith(first)
        assert len(_read_lines(_trace(tmp_path))) == 4

    def test_failed_trace_append_to_an_existing_trace_log_is_rolled_back_and_a_retry_quarantines_everything(
        self, tmp_path: Path
    ) -> None:
        """os.write really writes a truncated prefix into the trace and returns the short count: bytes land on
        disk before _append_records raises, so only the truncate back to the previous trace content saves it."""
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        trace_before = _trace(tmp_path).read_bytes()
        manifest_before = manifest_path.read_bytes()
        _append_line(audit_path, b"[]")
        damaged = audit_path.read_bytes()
        real_write = os.write

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                _quarantine(tmp_path, audit_path, manifest_path)

        assert _trace(tmp_path).read_bytes() == trace_before
        assert audit_path.read_bytes() == damaged
        assert manifest_path.read_bytes() == manifest_before

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [7])
        assert len(_read_lines(_trace(tmp_path))) == 4

    def test_failed_trace_append_to_a_new_trace_log_leaves_no_trace_file_and_a_retry_quarantines_everything(
        self, tmp_path: Path
    ) -> None:
        """Same as above, but the trace log did not exist before: the rollback unlinks it instead of truncating."""
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        audit_before = audit_path.read_bytes()
        manifest_before = manifest_path.read_bytes()
        assert not _trace(tmp_path).exists()
        real_write = os.write

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                _quarantine(tmp_path, audit_path, manifest_path)

        assert not _trace(tmp_path).exists()
        assert audit_path.read_bytes() == audit_before
        assert manifest_path.read_bytes() == manifest_before

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert len(removed) == len(entries) == 3

    def test_failed_trace_append_to_an_existing_trace_log_fsyncs_it_and_its_directory_after_the_rollback_truncate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rollback truncate of the trace log must be durable too, or a crash right after it can resurrect
        bytes of an entry this recovery already reported as failed via the raised OSError."""
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        _append_line(audit_path, b"[]")
        trace_path = _trace(tmp_path)
        real_write = os.write
        real_truncate = os.truncate
        real_fsync = os.fsync
        events: list[str] = []

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        def spy_truncate(path: str | Path, length: int) -> None:
            events.append("truncate")
            real_truncate(path, length)

        def spy_fsync(fd: int) -> None:
            try:
                if os.path.samestat(os.fstat(fd), trace_path.stat()):
                    events.append("fsync-trace")
            except FileNotFoundError:
                pass
            try:
                if os.path.samestat(os.fstat(fd), trace_path.parent.stat()):
                    events.append("fsync-trace-dir")
            except FileNotFoundError:
                pass
            real_fsync(fd)

        monkeypatch.setattr(os, "truncate", spy_truncate)
        monkeypatch.setattr(os, "fsync", spy_fsync)

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                _quarantine(tmp_path, audit_path, manifest_path)

        assert "truncate" in events
        assert "fsync-trace" in events
        assert "fsync-trace-dir" in events
        assert events.index("truncate") < events.index("fsync-trace")
        assert events.index("truncate") < events.index("fsync-trace-dir")

    def test_failed_trace_append_to_a_new_trace_log_fsyncs_its_directory_after_the_rollback_unlink(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same as above, but the trace log did not exist before: the rollback unlinks it, and that unlink must
        itself be durable, so its directory entry is fsynced after the unlink."""
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        trace_path = _trace(tmp_path)
        assert not trace_path.exists()
        real_write = os.write
        real_unlink = os.unlink
        real_fsync = os.fsync
        events: list[str] = []

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        def spy_unlink(path: str | Path) -> None:
            events.append("unlink")
            real_unlink(path)

        def spy_fsync(fd: int) -> None:
            try:
                if os.path.samestat(os.fstat(fd), trace_path.parent.stat()):
                    events.append("fsync-trace-dir")
            except FileNotFoundError:
                pass
            real_fsync(fd)

        monkeypatch.setattr(os, "unlink", spy_unlink)
        monkeypatch.setattr(os, "fsync", spy_fsync)

        with patch("os.write", side_effect=short_write):
            with pytest.raises(OSError):
                _quarantine(tmp_path, audit_path, manifest_path)

        assert not trace_path.exists()
        assert "unlink" in events
        assert "fsync-trace-dir" in events
        assert events.index("unlink") < events.index("fsync-trace-dir")

    def test_str_paths_are_accepted_and_recorded_as_given(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)

        removed = quarantine_damaged_lines(
            str(audit_path), str(manifest_path), quarantine_path=str(_trace(tmp_path)), signer=_signer()
        )

        assert [item.file for item in removed] == ["manifest", "audit", "audit"]
        entries = _read_lines(_trace(tmp_path))
        assert [entry["path"] for entry in entries] == [str(manifest_path), str(audit_path), str(audit_path)]

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    @pytest.mark.parametrize(
        "tail",
        [b"{}", b"[]", b"null", b'{"run_id": "run-d"}'],
        ids=["empty-object", "array", "null", "object"],
    )
    def test_unterminated_manifest_tail_that_is_valid_json_is_not_repaired(
        self, tmp_path: Path, tail: bytes, dry_run: bool
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")
        _torn(manifest_path, tail)

        _assert_refused(tmp_path, audit_path, manifest_path, dry_run=dry_run)

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_a_seal_missing_only_its_newline_is_not_repaired(self, tmp_path: Path, dry_run: bool) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")
        manifest_path.write_bytes(manifest_path.read_bytes().removesuffix(b"\n"))

        _assert_refused(tmp_path, audit_path, manifest_path, dry_run=dry_run)

    @pytest.mark.parametrize("index", [1, 3], ids=["middle", "last"])
    @pytest.mark.parametrize("damage", list(_DAMAGED_LINES.values()), ids=list(_DAMAGED_LINES))
    def test_terminated_damaged_manifest_line_is_not_repaired(self, tmp_path: Path, damage: bytes, index: int) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")
        _insert_line(manifest_path, index, damage)

        _assert_refused(tmp_path, audit_path, manifest_path)

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    @pytest.mark.parametrize("failure", ["edited-manifest", "broken-chain", "other-key", "wrong-head"])
    def test_manifest_log_that_does_not_verify_is_not_repaired(
        self, tmp_path: Path, failure: str, dry_run: bool
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        lines = manifest_path.read_bytes().splitlines(keepends=True)
        if failure == "edited-manifest":
            edited = lines[0].replace(b'"compliant": true', b'"compliant": false')
            assert edited != lines[0]
            lines[0] = edited
        if failure == "broken-chain":
            del lines[1]
        manifest_path.write_bytes(b"".join(lines))

        _assert_refused(
            tmp_path,
            audit_path,
            manifest_path,
            signer=_signer(_OTHER_KEY) if failure == "other-key" else None,
            expected_head="0" * 64 if failure == "wrong-head" else None,
            dry_run=dry_run,
        )

    @pytest.mark.parametrize("anchor", [0, 1], ids=["first-manifest", "second-manifest"])
    def test_expected_head_of_any_complete_manifest_repairs_a_cut_multi_seal_write(
        self, tmp_path: Path, anchor: int
    ) -> None:
        audit_path, manifest_path, heads = _log_cut_mid_write(tmp_path)
        before = _snapshot(tmp_path)
        manifest_before = before["manifests.ndjson"]

        removed = _quarantine(tmp_path, audit_path, manifest_path, expected_head=heads[anchor])

        assert _summary(removed) == _spans("manifest", manifest_before, [3])
        assert manifest_path.read_bytes() == manifest_before[: manifest_before.rindex(b"\n") + 1]
        assert audit_path.read_bytes() == before["audit.ndjson"]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == heads[1]
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=heads[1])
        assert [(manifest["run_id"], manifest["previous_manifest_hash"]) for manifest in manifests] == [
            ("run-c", heads[1])
        ]

    def test_dry_run_accepts_the_expected_head_of_an_earlier_manifest(self, tmp_path: Path) -> None:
        audit_path, manifest_path, heads = _log_cut_mid_write(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path, expected_head=heads[0], dry_run=True)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [3])
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    @pytest.mark.parametrize("anchor", ["unrelated", "cut-manifest"])
    def test_expected_head_of_no_complete_manifest_is_not_repaired(
        self, tmp_path: Path, anchor: str, dry_run: bool
    ) -> None:
        audit_path, manifest_path, heads = _log_cut_mid_write(tmp_path)

        _assert_refused(
            tmp_path,
            audit_path,
            manifest_path,
            expected_head="0" * 64 if anchor == "unrelated" else heads[2],
            dry_run=dry_run,
        )

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_a_signer_with_another_key_is_refused_whatever_the_expected_head(
        self, tmp_path: Path, dry_run: bool
    ) -> None:
        audit_path, manifest_path, heads = _log_cut_mid_write(tmp_path)

        _assert_refused(
            tmp_path, audit_path, manifest_path, signer=_signer(_OTHER_KEY), expected_head=heads[0], dry_run=dry_run
        )

    def test_previous_signers_repairs_a_torn_tail_signed_by_an_old_key(self, tmp_path: Path) -> None:
        audit_path, manifest_path, head = _torn_manifest_log(tmp_path)
        old_signer = _signer()
        new_signer = _signer(_OTHER_KEY, "key-2")
        manifest_before = manifest_path.read_bytes()

        removed = quarantine_damaged_lines(
            audit_path,
            manifest_path,
            quarantine_path=_trace(tmp_path),
            signer=new_signer,
            previous_signers=[old_signer],
            expected_head=head,
        )

        assert _summary(removed) == _spans("manifest", manifest_before, [4])
        # The repair does not rotate: the log is still under the old key.
        assert verify_ndjson_log(audit_path, manifest_path, signer=old_signer) == head
        with pytest.raises(ManifestVerificationError, match=_under_key("key-1", "key-2")):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    def test_torn_rotation_entry_tail_is_reported_by_a_dry_run_and_nothing_changes(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _torn_rotation_log(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path, dry_run=True)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [4])
        assert _snapshot(tmp_path) == before

    def test_torn_rotation_entry_tail_is_quarantined_and_rotating_again_restores_the_new_key(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, head = _torn_rotation_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        manifest_before = manifest_path.read_bytes()

        removed = _quarantine(
            tmp_path, audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=head
        )

        assert _summary(removed) == _spans("manifest", manifest_before, [4])
        assert manifest_path.read_bytes() == manifest_before[: manifest_before.rindex(b"\n") + 1]
        with pytest.raises(ManifestVerificationError, match=_under_key("key-1", "key-2")):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        with pytest.raises(ManifestVerificationError, match=_under_key("key-1", "key-2")):
            seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        entry = rotate_manifest_key(manifest_path, signer=new_signer, previous_signers=[old_signer], expected_head=head)
        assert verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer]) == (
            manifest_hash(entry)
        )

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_unterminated_rotation_entry_tail_that_is_valid_json_is_not_repaired(
        self, tmp_path: Path, dry_run: bool
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _torn(manifest_path, _canonical(_rotation_entry(_signer(_OTHER_KEY, "key-2"), head)))

        _assert_refused(tmp_path, audit_path, manifest_path, dry_run=dry_run)

    def test_expected_head_of_a_rotation_entry_repairs_a_torn_seal_after_it(self, tmp_path: Path) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        audit_path, manifest_path = _build_log(tmp_path, ("seal", old_signer), ("rotate", new_signer))
        entry_head = manifest_hash(_read_lines(manifest_path)[-1])
        record = _record("run-3", 3)
        _write_records(audit_path, [record])
        seal = _canonical(seal_run([record], run_id="run-3", signer=new_signer, previous_manifest_hash=entry_head))
        _torn(manifest_path, seal[: len(seal) // 2])
        manifest_before = manifest_path.read_bytes()

        removed = _quarantine(
            tmp_path,
            audit_path,
            manifest_path,
            signer=new_signer,
            previous_signers=[old_signer],
            expected_head=entry_head,
        )

        assert _summary(removed) == _spans("manifest", manifest_before, [3])
        verified = verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])
        assert verified == entry_head

    @pytest.mark.parametrize(
        "previous_signers",
        [[_signer()], [_signer(key_id="key-3"), _signer(_OTHER_KEY, "key-3")]],
        ids=["signer-and-previous", "two-previous"],
    )
    def test_previous_signers_with_a_duplicate_key_id_is_a_value_error(
        self, tmp_path: Path, previous_signers: list[ManifestSigner]
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            quarantine_damaged_lines(
                audit_path,
                manifest_path,
                quarantine_path=_trace(tmp_path),
                signer=_signer(_OTHER_KEY),
                previous_signers=previous_signers,
            )

        assert not isinstance(excinfo.value, ManifestVerificationError)

    def test_a_broken_chain_is_refused_even_with_the_head_of_a_manifest_before_the_break(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        lines = manifest_path.read_bytes().splitlines(keepends=True)
        del lines[1]
        manifest_path.write_bytes(b"".join(lines))

        _assert_refused(tmp_path, audit_path, manifest_path, expected_head=manifest_hash(json.loads(lines[0])))

    @pytest.mark.parametrize("anchor", ["unrelated", "sealed-manifest"])
    def test_any_expected_head_is_refused_when_the_log_has_no_complete_manifest(
        self, tmp_path: Path, anchor: str
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        first = manifest_path.read_bytes().splitlines(keepends=True)[0]
        manifest_path.write_bytes(first[: len(first) // 2])

        _assert_refused(
            tmp_path,
            audit_path,
            manifest_path,
            expected_head="0" * 64 if anchor == "unrelated" else manifest_hash(json.loads(first)),
        )

    def test_a_log_that_is_only_a_torn_fragment_is_truncated_to_empty(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        first = manifest_path.read_bytes().splitlines(keepends=True)[0]
        manifest_path.write_bytes(first[: len(first) // 2])
        manifest_before = manifest_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("manifest", manifest_before, [1])
        assert manifest_path.read_bytes() == b""
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) is None

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_clean_files_return_nothing_and_change_nothing(self, tmp_path: Path, dry_run: bool) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)

        assert _quarantine(tmp_path, audit_path, manifest_path, dry_run=dry_run) == []

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize(
        "tail",
        [b'{"run_id": "run-d", "tenant', b"\xff\xfe", b'{"run_id": "run-d", "tenant": "\xc3'],
        ids=["torn", "bad-utf8", "torn-mid-character"],
    )
    def test_unterminated_audit_tail_that_does_not_parse_is_removed(self, tmp_path: Path, tail: bytes) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        clean = audit_path.read_bytes()
        _torn(audit_path, tail)
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [7])
        assert removed[0].length == len(tail)
        assert audit_path.read_bytes() == clean
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    @pytest.mark.parametrize(
        "tail",
        [_canonical(_record("run-d", 7)), b"[]", b"{}", b"null", b"5", b'"x"'],
        ids=["complete-record", "array", "empty-object", "null", "number", "string"],
    )
    def test_unterminated_audit_tail_that_is_valid_json_is_not_repaired(
        self, tmp_path: Path, tail: bytes, dry_run: bool
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")
        _torn(audit_path, tail)

        _assert_refused(tmp_path, audit_path, manifest_path, dry_run=dry_run)

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_a_sealed_record_missing_only_its_newline_is_not_repaired(self, tmp_path: Path, dry_run: bool) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        audit_path.write_bytes(audit_path.read_bytes().removesuffix(b"\n"))

        _assert_refused(tmp_path, audit_path, manifest_path, dry_run=dry_run)

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_a_refused_audit_tail_also_stops_the_manifest_repair(self, tmp_path: Path, dry_run: bool) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        _torn(audit_path, b"[]")

        _assert_refused(tmp_path, audit_path, manifest_path, dry_run=dry_run)

    @pytest.mark.parametrize("damage", list(_DAMAGED_LINES.values()), ids=list(_DAMAGED_LINES))
    def test_damaged_audit_line_is_removed_and_the_others_keep_their_bytes(self, tmp_path: Path, damage: bytes) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        clean = audit_path.read_bytes()
        _insert_line(audit_path, 2, damage)
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [3])
        assert all(item.reason for item in removed)
        assert audit_path.read_bytes() == clean
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head

    def test_every_kind_of_damaged_audit_line_is_removed_in_file_order(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        clean = audit_path.read_bytes()
        lines = clean.splitlines(keepends=True)
        damages = list(_DAMAGED_LINES.values())
        interleaved = [piece for pair in zip(lines, damages, strict=False) for piece in pair] + lines[len(damages) :]
        audit_path.write_bytes(b"".join(interleaved))
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, range(2, 2 * len(damages) + 1, 2))
        assert audit_path.read_bytes() == clean
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head

    @pytest.mark.parametrize(
        ("index", "damage", "run_id"),
        [
            (3, lambda line: b"garbled\n", "run-b"),
            (5, lambda line: line[:20], "run-c"),
        ],
        ids=["garbled", "torn-last-line"],
    )
    def test_damaged_line_of_a_sealed_run_is_removed_but_verification_still_fails(
        self, tmp_path: Path, index: int, damage: Any, run_id: str
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        lines = audit_path.read_bytes().splitlines(keepends=True)
        lines[index] = damage(lines[index])
        audit_path.write_bytes(b"".join(lines))
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [index + 1])
        with pytest.raises(ManifestVerificationError, match=run_id):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())
        assert base64.b64decode(_read_lines(_trace(tmp_path))[0]["raw_base64"]) == lines[index]

    @pytest.mark.parametrize("run_id", [["x"], 5], ids=["list", "number"])
    def test_audit_line_with_a_non_string_run_id_is_not_damage_and_still_fails_verification(
        self, tmp_path: Path, run_id: Any
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [{**_record(), "run_id": run_id}])
        kept = audit_path.read_bytes()
        _append_line(audit_path, "")
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [8])
        assert audit_path.read_bytes() == kept
        with pytest.raises(ManifestVerificationError, match="run_id"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_missing_audit_file_is_nothing_to_repair_and_the_manifest_tail_is_still_repaired(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        audit_path.unlink()
        manifest_before = manifest_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("manifest", manifest_before, [4])
        assert manifest_path.read_bytes() == manifest_before[: manifest_before.rindex(b"\n") + 1]
        assert not audit_path.exists()

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_neither_audit_file_nor_manifest_log_returns_nothing_and_creates_nothing(
        self, tmp_path: Path, dry_run: bool
    ) -> None:
        audit_path, manifest_path = tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson"

        assert _quarantine(tmp_path, audit_path, manifest_path, dry_run=dry_run) == []

        assert _snapshot(tmp_path) == {}

    def test_audit_file_is_repaired_before_anything_was_sealed(self, tmp_path: Path) -> None:
        audit_path, manifest_path = tmp_path / "audit.ndjson", tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])
        clean = audit_path.read_bytes()
        _torn(audit_path, b'{"run_id": "run-c", "tena')
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [3])
        assert audit_path.read_bytes() == clean
        assert manifest_path.exists()
        assert manifest_path.read_bytes() == b""
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        assert [manifest["run_id"] for manifest in manifests] == ["run-a", "run-b"]

    def test_both_files_are_repaired_manifest_entry_first_and_sealing_works_again(self, tmp_path: Path) -> None:
        audit_path, manifest_path, head = _damaged_log(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path, expected_head=head)

        assert _summary(removed) == [
            *_spans("manifest", before["manifests.ndjson"], [4]),
            *_spans("audit", before["audit.ndjson"], [4, 8]),
        ]
        entries = _read_lines(_trace(tmp_path))
        assert [(entry["file"], entry["line"]) for entry in entries] == [("manifest", 4), ("audit", 4), ("audit", 8)]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head
        _write_records(audit_path, [_record("run-d", 7)])
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=head)
        assert [(manifest["run_id"], manifest["previous_manifest_hash"]) for manifest in manifests] == [("run-d", head)]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_dry_run_returns_what_a_real_run_removes_and_changes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        before = _snapshot(tmp_path)

        planned = _quarantine(tmp_path, audit_path, manifest_path, dry_run=True)

        assert len(planned) == 3
        assert _snapshot(tmp_path) == before
        assert planned == _quarantine(tmp_path, audit_path, manifest_path)

    def test_dry_run_creates_no_file_when_the_manifest_log_does_not_exist(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])
        _torn(audit_path, b'{"run_id": "run-a", "tena')
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, tmp_path / "manifests.ndjson", dry_run=True)

        assert _summary(removed) == _spans("audit", before["audit.ndjson"], [2])
        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("mode", [0o600, 0o640], ids=["owner-only", "group-readable"])
    def test_rewritten_audit_file_keeps_its_permission_bits_and_leaves_no_temporary_file(
        self, tmp_path: Path, mode: int
    ) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")
        audit_path.chmod(mode)

        _quarantine(tmp_path, audit_path, manifest_path)

        assert stat.S_IMODE(audit_path.stat().st_mode) == mode
        assert sorted(_snapshot(tmp_path)) == ["audit.ndjson", "manifests.ndjson", "quarantine.ndjson"]

    def test_failed_audit_replacement_keeps_the_trace_and_the_original_audit_file(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")
        _insert_line(audit_path, 5, b"\n")
        before = _snapshot(tmp_path)

        with patch("os.replace", side_effect=OSError(errno.EIO, "replace failed")):
            with pytest.raises(OSError, match="replace failed"):
                _quarantine(tmp_path, audit_path, manifest_path)

        after = _snapshot(tmp_path)
        assert after.pop("quarantine.ndjson")
        assert after == before
        assert [(entry["file"], entry["line"]) for entry in _read_lines(_trace(tmp_path))] == [
            ("audit", 4),
            ("audit", 6),
        ]

    def test_failed_trace_write_leaves_both_files_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        before = _snapshot(tmp_path)

        def disk_full(*args: Any, **kwargs: Any) -> None:
            raise OSError(errno.ENOSPC, "disk full")

        _patch_bindings(monkeypatch, "_append_records", disk_full)

        with pytest.raises(OSError, match="disk full"):
            _quarantine(tmp_path, audit_path, manifest_path)

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("existing", [False, True], ids=["new-trace", "existing-trace"])
    def test_failed_trace_append_is_rolled_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        if existing:
            _rewrite_lines(_trace(tmp_path), [_trace_entry_ref(_signer(), None)])
        before = _snapshot(tmp_path)

        def partial_append(path: str | Path, records: Any) -> None:
            with open(path, "ab") as file:
                file.write(b'{"quarantine_version": 1, "fi')
            raise OSError(errno.ENOSPC, "disk full")

        _patch_bindings(monkeypatch, "_append_records", partial_append)

        with pytest.raises(OSError, match="disk full"):
            _quarantine(tmp_path, audit_path, manifest_path)

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_quarantine_log_without_a_final_newline_is_refused_and_nothing_changes(
        self, tmp_path: Path, dry_run: bool
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _trace(tmp_path).write_bytes(b'{"quarantine_version": 1, "fi')
        before = _snapshot(tmp_path)

        with pytest.raises(ManifestVerificationError, match=re.escape(str(_trace(tmp_path)))):
            _quarantine(tmp_path, audit_path, manifest_path, dry_run=dry_run)

        assert _snapshot(tmp_path) == before

    def test_an_empty_quarantine_log_is_appended_to(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _trace(tmp_path).write_bytes(b"")

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert len(removed) == len(_read_lines(_trace(tmp_path))) == 3

    @_both_recoveries
    def test_trace_is_fsynced_before_any_file_is_modified_and_the_manifest_and_its_directory_after_the_cut(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recovery: _Recovery
    ) -> None:
        manifest_path, repair, dropped = recovery(tmp_path)
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        trace_path = _trace(trace_dir)
        events: list[tuple[str, int]] = []
        real_fsync, real_replace, real_truncate = os.fsync, os.replace, os.truncate

        def spy_fsync(fd: int) -> None:
            for name, target in (("trace", trace_path), ("manifest", manifest_path), ("manifest-dir", tmp_path)):
                try:
                    if os.path.samestat(os.fstat(fd), target.stat()):
                        events.append((f"fsync-{name}", target.stat().st_size))
                except FileNotFoundError:
                    pass
            real_fsync(fd)

        def spy_replace(*args: Any, **kwargs: Any) -> None:
            events.append(("replace", 0))
            real_replace(*args, **kwargs)

        def spy_truncate(path: str | Path, length: int) -> None:
            events.append(("truncate", length))
            real_truncate(path, length)

        monkeypatch.setattr(os, "fsync", spy_fsync)
        monkeypatch.setattr(os, "replace", spy_replace)
        monkeypatch.setattr(os, "truncate", spy_truncate)

        removed = repair(trace_dir)

        assert len(removed) == dropped
        names = [name for name, _ in events]
        # The manifest log is cut, and the audit file replaced if the repair dropped audit lines.
        modifications = {"truncate" if item.file == "manifest" else "replace" for item in removed}
        assert "truncate" in modifications
        assert modifications <= set(names)
        first = min(names.index(name) for name in modifications)
        assert ("fsync-trace", trace_path.stat().st_size) in events[:first]
        cut = names.index("truncate")
        assert "fsync-manifest" in names[cut:]
        assert "fsync-manifest-dir" in names[cut:]

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    @pytest.mark.parametrize("target", ["audit.ndjson", "manifests.ndjson"])
    @pytest.mark.parametrize("spelling", ["same-path", "relative", "symlink"])
    def test_quarantine_path_that_resolves_to_a_log_is_refused_with_a_plain_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str, target: str, dry_run: bool
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        quarantine_path: Path | str = tmp_path / target
        if spelling == "relative":
            monkeypatch.chdir(tmp_path)
            quarantine_path = f"./{target}"
        if spelling == "symlink":
            quarantine_path = tmp_path / "trace-link.ndjson"
            quarantine_path.symlink_to(tmp_path / target)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            quarantine_damaged_lines(
                audit_path, manifest_path, quarantine_path=quarantine_path, signer=_signer(), dry_run=dry_run
            )

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    def test_a_symlinked_audit_path_repairs_the_real_file_and_stays_a_symlink(self, tmp_path: Path) -> None:
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        real_audit, manifest_path = _sealed_log(real_dir)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        clean = real_audit.read_bytes()
        _insert_line(real_audit, 3, b"[]\n")
        damaged = real_audit.read_bytes()
        link = tmp_path / "audit-link.ndjson"
        link.symlink_to(real_audit)

        removed = _quarantine(tmp_path, link, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [4])
        assert link.is_symlink()
        assert link.resolve() == real_audit.resolve()
        assert not real_audit.is_symlink()
        assert real_audit.read_bytes() == clean
        assert [entry["path"] for entry in _read_lines(_trace(tmp_path))] == [str(link)]
        assert verify_ndjson_log(link, manifest_path, signer=_signer()) == head
        assert sorted(path.name for path in real_dir.iterdir()) == ["audit.ndjson", "manifests.ndjson"]

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_recovery_waits_for_a_sealer_holding_the_lock(self, tmp_path: Path, dry_run: bool) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")

        with ThreadPoolExecutor(max_workers=1) as pool:
            fd = os.open(manifest_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                future = pool.submit(_quarantine, tmp_path, audit_path, manifest_path, dry_run=dry_run)
                done, _ = wait([future], timeout=0.3)
            finally:
                os.close(fd)

            assert not done
            assert len(future.result(timeout=10)) == 1

    def test_dry_run_does_not_wait_for_a_reader_holding_a_shared_lock(self, tmp_path: Path) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 3, b"[]\n")

        with ThreadPoolExecutor(max_workers=1) as pool:
            fd = os.open(manifest_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
                future = pool.submit(_quarantine, tmp_path, audit_path, manifest_path, dry_run=True)
                done, _ = wait([future], timeout=5)
            finally:
                os.close(fd)

            assert done
            assert len(future.result()) == 1

    def test_recovery_holds_an_exclusive_lock_on_the_manifest_log_while_it_repairs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        probes: list[tuple[str, bool]] = []
        real_replace = os.replace

        def probe_signing(payload: bytes) -> None:
            probes.append(("sign", _lock_refused(manifest_path)))

        def spy_append(*args: Any, **kwargs: Any) -> None:
            probes.append(("append", _lock_refused(manifest_path)))
            _append_records(*args, **kwargs)

        def spy_replace(*args: Any, **kwargs: Any) -> None:
            probes.append(("replace", _lock_refused(manifest_path)))
            real_replace(*args, **kwargs)

        _patch_bindings(monkeypatch, "_append_records", spy_append)
        monkeypatch.setattr(os, "replace", spy_replace)

        # Verifying and the trace both use the signer, so the wrapper probes both hooks.
        probing = _HookedSigner(_signer(), on_sign=probe_signing, on_verify=probe_signing)
        removed = _quarantine(tmp_path, audit_path, manifest_path, signer=probing)

        assert len(removed) == 3
        assert {stage for stage, _ in probes} == {"sign", "append", "replace"}
        assert all(refused for _, refused in probes)
        fd = os.open(manifest_path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    def test_quarantine_fsyncs_the_manifest_log_and_its_directory_after_truncating(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        events: list[str] = []
        real_fsync = os.fsync

        def spy_fsync(fd: int) -> None:
            try:
                if os.path.samestat(os.fstat(fd), manifest_path.stat()):
                    events.append("fsync-manifest")
            except FileNotFoundError:
                pass
            try:
                if os.path.samestat(os.fstat(fd), manifest_path.parent.stat()):
                    events.append("fsync-manifest-dir")
            except FileNotFoundError:
                pass
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", spy_fsync)

        quarantine_damaged_lines(
            audit_path, manifest_path, quarantine_path=trace_dir / "quarantine.ndjson", signer=_signer()
        )

        assert "fsync-manifest" in events
        assert "fsync-manifest-dir" in events

    @_both_recoveries
    def test_recovery_holds_an_exclusive_lock_on_the_quarantine_log_during_the_trace_append(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recovery: _Recovery
    ) -> None:
        pytest.importorskip("fcntl")
        _, repair, dropped = recovery(tmp_path)
        quarantine_path = _trace(tmp_path)
        quarantine_path.write_bytes(b"")
        refused: list[bool] = []

        def spy_append(*args: Any, **kwargs: Any) -> None:
            refused.append(_lock_refused(quarantine_path))
            _append_records(*args, **kwargs)

        _patch_bindings(monkeypatch, "_append_records", spy_append)

        removed = repair(tmp_path)

        assert len(removed) == dropped
        assert refused
        assert all(refused)

    def test_failed_trace_append_under_the_exclusive_lock_still_leaves_no_file_when_none_existed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("fcntl")
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        quarantine_path = _trace(tmp_path)
        assert not quarantine_path.exists()

        def partial_append(path: str | Path, records: Any) -> None:
            with open(path, "ab") as file:
                file.write(b'{"quarantine_version": 1, "fi')
            raise OSError(errno.ENOSPC, "disk full")

        _patch_bindings(monkeypatch, "_append_records", partial_append)

        with pytest.raises(OSError, match="disk full"):
            _quarantine(tmp_path, audit_path, manifest_path)

        assert not quarantine_path.exists()

    def test_repairs_a_torn_manifest_tail_where_fcntl_is_missing_even_when_the_quarantine_log_does_not_exist_yet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "fcntl", None)
        audit_path, manifest_path, head = _torn_manifest_log(tmp_path)
        manifest_before = manifest_path.read_bytes()
        assert not _trace(tmp_path).exists()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("manifest", manifest_before, [4])
        assert manifest_path.read_bytes() == manifest_before[: manifest_before.rindex(b"\n") + 1]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head
        assert len(_read_lines(_trace(tmp_path))) == 1


@_both_algorithms
class TestQuarantineTraceChain:
    """Trace entries are v2: chained by previous_entry_hash and signed over the v2 payload."""

    def test_the_first_entry_has_no_predecessor_and_each_later_one_hashes_the_line_before(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)

        _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert len(entries) == 3
        assert entries[0]["previous_entry_hash"] is None
        assert [entry["previous_entry_hash"] for entry in entries[1:]] == [_sha256(_canonical(e)) for e in entries[:2]]

    def test_a_second_recovery_chains_onto_the_existing_trace(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        _append_line(audit_path, b"[]")

        _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert len(entries) == 4
        assert entries[3]["previous_entry_hash"] == _sha256(_canonical(entries[2]))
        assert _verify_quarantine(_trace(tmp_path), _signer()) == _sha256(_canonical(entries[3]))

    def test_a_recovery_chains_onto_a_legacy_v1_trace(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        legacy = _trace_entry_ref(_signer(), None, v1=True)
        _rewrite_lines(_trace(tmp_path), [legacy])

        _quarantine(tmp_path, audit_path, manifest_path)

        entries = _read_lines(_trace(tmp_path))
        assert entries[0] == legacy
        assert entries[1]["quarantine_version"] == 2
        assert entries[1]["previous_entry_hash"] == _sha256(_canonical(legacy))
        assert _verify_quarantine(_trace(tmp_path), _signer()) == _sha256(_canonical(entries[-1]))

    def test_a_tampered_trace_refuses_the_repair_and_changes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        _append_line(audit_path, b"[]")
        entries = _read_lines(_trace(tmp_path))
        entries[1]["line"] += 1
        _rewrite_lines(_trace(tmp_path), entries)

        _assert_raises_and_unchanged(tmp_path, lambda: _quarantine(tmp_path, audit_path, manifest_path))

    def test_a_trace_with_a_deleted_line_refuses_the_repair_and_changes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        _append_line(audit_path, b"[]")
        entries = _read_lines(_trace(tmp_path))
        _rewrite_lines(_trace(tmp_path), [entries[0], entries[2]])

        _assert_raises_and_unchanged(tmp_path, lambda: _quarantine(tmp_path, audit_path, manifest_path))

    def test_a_trace_under_an_unknown_key_refuses_the_repair_and_changes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _rewrite_lines(_trace(tmp_path), _trace_chain(_signer(_OTHER_KEY, "key-2")))

        _assert_raises_and_unchanged(tmp_path, lambda: _quarantine(tmp_path, audit_path, manifest_path))

    def test_a_trace_signed_by_a_retired_key_is_extended_with_previous_signers(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _rewrite_lines(_trace(tmp_path), _trace_chain(_signer()))
        key_2 = _signer(_OTHER_KEY, "key-2")

        removed = _quarantine(tmp_path, audit_path, manifest_path, signer=key_2, previous_signers=[_signer()])

        assert len(removed) == 3
        assert len(_read_lines(_trace(tmp_path))) == 4
        assert _verify_quarantine(_trace(tmp_path), key_2, _signer()) is not None

    @pytest.mark.parametrize("existing", [False, True], ids=["new-trace", "existing-trace"])
    def test_a_signing_failure_leaves_no_trace_file_it_created_and_changes_nothing(
        self, tmp_path: Path, existing: bool
    ) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        if existing:
            _rewrite_lines(_trace(tmp_path), _trace_chain(_signer()))
        calls: list[bytes] = []

        def fail_on_second(payload: bytes) -> None:
            calls.append(payload)
            if len(calls) == 2:
                raise RuntimeError("signing failed")

        before = _snapshot(tmp_path)

        with pytest.raises(RuntimeError, match="signing failed"):
            _quarantine(tmp_path, audit_path, manifest_path, signer=_HookedSigner(_signer(), on_sign=fail_on_second))

        assert len(calls) == 2
        assert _snapshot(tmp_path) == before
        assert existing or not _trace(tmp_path).exists()

    def test_verify_returns_none_for_a_missing_or_empty_trace(self, tmp_path: Path) -> None:
        assert _verify_quarantine(_trace(tmp_path), _signer()) is None
        _trace(tmp_path).write_bytes(b"")
        assert _verify_quarantine(_trace(tmp_path), _signer()) is None

    def test_verify_returns_the_hash_of_the_last_line_as_the_head(self, tmp_path: Path) -> None:
        entries = _trace_chain(_signer(), _signer(), _signer())
        _rewrite_lines(_trace(tmp_path), entries)

        assert _verify_quarantine(_trace(tmp_path), _signer()) == _sha256(_canonical(entries[-1]))

    def test_verify_accepts_lines_signed_by_a_retired_key_through_previous_signers(self, tmp_path: Path) -> None:
        key_2 = _signer(_OTHER_KEY, "key-2")
        entries = _trace_chain(_signer(), key_2)
        _rewrite_lines(_trace(tmp_path), entries)

        assert _verify_quarantine(_trace(tmp_path), key_2, _signer()) == _sha256(_canonical(entries[-1]))
        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), key_2)

    def test_verify_rejects_a_trace_under_another_key(self, tmp_path: Path) -> None:
        _rewrite_lines(_trace(tmp_path), _trace_chain(_signer()))

        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), _signer(_OTHER_KEY, "key-1"))

    @pytest.mark.parametrize("attack", ["delete-first", "delete-middle", "reorder", "edit", "relabel-key-id"])
    def test_verify_detects_a_deleted_reordered_or_edited_line(self, tmp_path: Path, attack: str) -> None:
        entries = _trace_chain(_signer(), _signer(), _signer())
        attacked = {
            "delete-first": [entries[1], entries[2]],
            "delete-middle": [entries[0], entries[2]],
            "reorder": [entries[0], entries[2], entries[1]],
            "edit": [entries[0], {**entries[1], "reason": "other"}, entries[2]],
            "relabel-key-id": [
                entries[0],
                {**entries[1], "signature": {**entries[1]["signature"], "key_id": "x"}},
                entries[2],
            ],
        }[attack]
        _rewrite_lines(_trace(tmp_path), attacked)

        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_verify_rejects_a_line_with_the_wrong_chain_value_even_when_correctly_signed(self, tmp_path: Path) -> None:
        entries = _trace_chain(_signer(), _signer())
        entries[1] = _trace_entry_ref(_signer(), "0" * 64, line=2)
        _rewrite_lines(_trace(tmp_path), entries)

        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_verify_rejects_a_first_line_that_names_a_predecessor(self, tmp_path: Path) -> None:
        _rewrite_lines(_trace(tmp_path), [_trace_entry_ref(_signer(), "0" * 64)])

        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_verify_rejects_an_unknown_version(self, tmp_path: Path) -> None:
        _rewrite_lines(_trace(tmp_path), [_trace_entry_ref(_signer(), None, quarantine_version=3)])

        with pytest.raises(ManifestVerificationError, match="3"):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_verify_rejects_an_unterminated_last_line(self, tmp_path: Path) -> None:
        _rewrite_lines(_trace(tmp_path), _trace_chain(_signer()))
        _torn(_trace(tmp_path), b'{"quarantine_version": 2, "fi')

        with pytest.raises(ManifestVerificationError, match=re.escape(str(_trace(tmp_path)))):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_legacy_v1_lines_verify_as_a_prefix_before_v2_lines(self, tmp_path: Path) -> None:
        legacy = [
            _trace_entry_ref(_signer(), None, v1=True, line=1),
            _trace_entry_ref(_signer(), None, v1=True, line=2),
        ]
        v2 = _trace_entry_ref(_signer(), _sha256(_canonical(legacy[-1])), line=3)
        _rewrite_lines(_trace(tmp_path), legacy)
        assert _verify_quarantine(_trace(tmp_path), _signer()) == _sha256(_canonical(legacy[-1]))

        _rewrite_lines(_trace(tmp_path), [*legacy, v2])

        assert _verify_quarantine(_trace(tmp_path), _signer()) == _sha256(_canonical(v2))

    def test_a_v1_line_after_a_v2_line_is_rejected(self, tmp_path: Path) -> None:
        first = _trace_entry_ref(_signer(), None, line=1)
        late_v1 = _trace_entry_ref(_signer(), None, v1=True, line=2)
        _rewrite_lines(_trace(tmp_path), [first, late_v1])

        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_a_v1_line_signed_over_the_v2_payload_is_rejected(self, tmp_path: Path) -> None:
        v1 = _trace_entry_ref(_signer(), None, v1=True)
        block = {"algorithm": v1["signature"]["algorithm"], "key_id": v1["signature"]["key_id"]}
        v1["signature"]["value"] = _signer().sign(_signing_payload_ref({**_unsigned(v1), "signature": block}))
        _rewrite_lines(_trace(tmp_path), [v1])

        with pytest.raises(ManifestVerificationError):
            _verify_quarantine(_trace(tmp_path), _signer())

    def test_every_line_hash_of_the_trace_passes_as_an_anchor(self, tmp_path: Path) -> None:
        entries = _trace_chain(_signer(), _signer(), _signer())
        _rewrite_lines(_trace(tmp_path), entries)
        heads = [_sha256(_canonical(entry)) for entry in entries]

        assert _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=heads) == heads[-1]
        assert _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=iter(heads)) == heads[-1]

    def test_an_anchor_on_a_legacy_v1_line_hash_passes(self, tmp_path: Path) -> None:
        legacy = _trace_entry_ref(_signer(), None, v1=True)
        _rewrite_lines(_trace(tmp_path), [legacy])

        assert _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=[_sha256(_canonical(legacy))]) == _sha256(
            _canonical(legacy)
        )

    def test_one_unknown_anchor_among_valid_ones_fails_and_is_named(self, tmp_path: Path) -> None:
        entries = _trace_chain(_signer(), _signer())
        _rewrite_lines(_trace(tmp_path), entries)
        unknown = "1" * 64

        with pytest.raises(ManifestVerificationError) as excinfo:
            _verify_quarantine(
                _trace(tmp_path), _signer(), anchored_heads=[*(_sha256(_canonical(e)) for e in entries), unknown]
            )

        _assert_names(excinfo, unknown, "not a line of the quarantine trace")

    def test_a_trace_with_its_newest_line_dropped_fails_only_with_the_old_head_anchored(self, tmp_path: Path) -> None:
        entries = _trace_chain(_signer(), _signer(), _signer())
        old_head = _sha256(_canonical(entries[-1]))
        _rewrite_lines(_trace(tmp_path), entries[:-1])

        # Control: without the anchor the cut trace verifies.
        assert _verify_quarantine(_trace(tmp_path), _signer()) == _sha256(_canonical(entries[1]))
        with pytest.raises(ManifestVerificationError) as excinfo:
            _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=[old_head])
        _assert_names(excinfo, old_head, "not a line of the quarantine trace")

    @pytest.mark.parametrize("state", ["missing", "empty"])
    def test_a_missing_or_empty_trace_fails_with_an_anchor(self, tmp_path: Path, state: str) -> None:
        if state == "empty":
            _trace(tmp_path).write_bytes(b"")

        with pytest.raises(ManifestVerificationError) as excinfo:
            _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=["2" * 64])

        _assert_names(excinfo, "2" * 64, "not a line of the quarantine trace")

    def test_a_trace_cut_and_regrown_by_a_repair_fails_with_the_old_head_anchored(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _damaged_log(tmp_path)
        _quarantine(tmp_path, audit_path, manifest_path)
        entries = _read_lines(_trace(tmp_path))
        old_head = _sha256(_canonical(entries[-1]))
        _rewrite_lines(_trace(tmp_path), entries[:-1])
        _append_line(audit_path, b"[]")
        _quarantine(tmp_path, audit_path, manifest_path)
        assert len(_read_lines(_trace(tmp_path))) == len(entries)

        with pytest.raises(ManifestVerificationError) as excinfo:
            _verify_quarantine(_trace(tmp_path), _signer(), anchored_heads=[old_head])

        _assert_names(excinfo, old_head)

    @_both_recoveries
    def test_a_cut_trace_with_the_old_head_anchored_refuses_the_repair(
        self, tmp_path: Path, recovery: _Recovery
    ) -> None:
        _, repair, _ = recovery(tmp_path)
        entries = _trace_chain(_signer(), _signer(), _signer())
        _rewrite_lines(_trace(tmp_path), entries[:-1])
        old_head = _sha256(_canonical(entries[-1]))

        excinfo = _assert_raises_and_unchanged(tmp_path, lambda: repair(tmp_path, anchored_heads=[old_head]))

        _assert_names(excinfo, old_head, "not a line of the quarantine trace")

    @_both_recoveries
    def test_the_current_head_anchored_lets_the_repair_proceed(self, tmp_path: Path, recovery: _Recovery) -> None:
        _, repair, dropped = recovery(tmp_path)
        entries = _trace_chain(_signer(), _signer())
        _rewrite_lines(_trace(tmp_path), entries)

        removed = repair(tmp_path, anchored_heads=[_sha256(_canonical(entries[-1]))])

        assert len(removed) == dropped
        assert len(_read_lines(_trace(tmp_path))) == 2 + dropped


@_both_algorithms
@_ed25519_only
class TestEd25519PublicKeyOnly:
    """A host that holds only public keys can verify and inspect a log, and can never write to it."""

    def test_a_whole_rotated_log_verifies_with_public_keys_alone_and_returns_the_same_head(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path)
        head = verify_ndjson_log(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])
        current = _verify_only(_THIRD_KEY, "key-3")
        previous = [_verify_only(_KEY, "key-1"), _verify_only(_OTHER_KEY, "key-2")]

        assert verify_ndjson_log(audit_path, manifest_path, signer=current, previous_signers=previous) == head
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=current, previous_signers=previous)
        assert coverage.head == head
        assert coverage.sealed_runs == 3
        assert head == manifest_hash(_read_lines(manifest_path)[-1])

    def test_a_public_key_that_is_not_the_sealing_keys_is_rejected(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        # Control: the sealing key's public key verifies.
        verify_ndjson_log(audit_path, manifest_path, signer=_verify_only())

        with pytest.raises(ManifestVerificationError, match=re.escape(_BAD_SIGNATURE)):
            verify_ndjson_log(audit_path, manifest_path, signer=_verify_only(_OTHER_KEY))

    def test_a_dry_run_quarantine_reports_the_damage_and_changes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        before = _snapshot(tmp_path)

        removed = _quarantine(tmp_path, audit_path, manifest_path, signer=_verify_only(), dry_run=True)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [4])
        assert _snapshot(tmp_path) == before

    def test_a_repair_cannot_sign_its_trace_so_it_raises_and_leaves_both_logs_alone(self, tmp_path: Path) -> None:
        audit_path, manifest_path, _ = _torn_manifest_log(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError, match="public key"):
            _quarantine(tmp_path, audit_path, manifest_path, signer=_verify_only())

        assert _snapshot(tmp_path) == before

    def test_a_rotation_entry_repair_cannot_sign_its_trace_so_it_raises_and_leaves_the_log_alone(
        self, tmp_path: Path
    ) -> None:
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError, match="public key"):
            _quarantine_from_entry(tmp_path, manifest_path, signer=_verify_only(), expected_head=anchor)

        assert _snapshot(tmp_path) == before

    def test_sealing_a_pending_run_raises_value_error_and_writes_nothing(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 7)])
        before = manifest_path.read_bytes()

        with pytest.raises(ValueError, match="public key"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_verify_only())

        assert manifest_path.read_bytes() == before

    def test_rotating_to_a_new_public_key_raises_value_error_and_writes_nothing(self, tmp_path: Path) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = manifest_path.read_bytes()
        head = manifest_hash(_read_lines(manifest_path)[-1])

        with pytest.raises(ValueError, match="public key") as excinfo:
            rotate_manifest_key(
                manifest_path,
                signer=_verify_only(_OTHER_KEY, "key-2"),
                previous_signers=[_signer()],
                expected_head=head,
            )

        assert not isinstance(excinfo.value, (ManifestVerificationError, KeyAlreadyCurrentError))
        assert manifest_path.read_bytes() == before

    def test_rotating_away_from_a_key_known_only_by_its_public_key_raises_value_error_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        before = manifest_path.read_bytes()
        head = manifest_hash(_read_lines(manifest_path)[-1])

        with pytest.raises(ValueError, match="public key") as excinfo:
            rotate_manifest_key(
                manifest_path,
                signer=_signer(_OTHER_KEY, "key-2"),
                previous_signers=[_verify_only()],
                expected_head=head,
            )

        assert not isinstance(excinfo.value, (ManifestVerificationError, KeyAlreadyCurrentError))
        assert manifest_path.read_bytes() == before

    def test_a_rotated_log_whose_outgoing_keys_are_public_verifies(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _rotated_three_key_log(tmp_path)
        entries = [line for line in _read_lines(manifest_path) if line.get("kind") == "key_rotation"]

        assert [entry["previous_key_signature"]["key_id"] for entry in entries] == ["key-1", "key-2"]
        previous = [_verify_only(_KEY, "key-1"), _verify_only(_OTHER_KEY, "key-2")]
        verify_ndjson_log(
            audit_path, manifest_path, signer=_verify_only(_THIRD_KEY, "key-3"), previous_signers=previous
        )

    def test_a_public_key_as_a_retired_key_still_lets_the_private_signer_seal(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_public, new_signer = _verify_only(), _signer(_OTHER_KEY, "key-2")
        _rotate(manifest_path, new_signer, _signer())
        _write_records(audit_path, [_record("run-d", 7)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_public])

        assert [manifest["run_id"] for manifest in manifests] == ["run-d"]
        head = verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_public])
        assert head == manifest_hash(manifests[-1])


class TestMixedAlgorithmRotation:
    """A log can mix algorithms: each key_id resolves to one signer, and a signature's algorithm must be its own."""

    def test_a_log_rotated_from_an_hmac_key_to_an_ed25519_key_seals_and_verifies(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        hmac_key, ed25519_key = HmacSha256Signer(_KEY, "key-1"), Ed25519Signer(_OTHER_KEY, "key-2")
        _write_records(audit_path, [_record("run-a", 1)])
        seal_ndjson_runs(audit_path, manifest_path, signer=hmac_key)
        _rotate(manifest_path, ed25519_key, hmac_key)
        _write_records(audit_path, [_record("run-b", 2)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=ed25519_key, previous_signers=[hmac_key])

        assert [line["signature"]["algorithm"] for line in _read_lines(manifest_path)] == [
            "HMAC-SHA256",
            "Ed25519",
            "Ed25519",
        ]
        head = verify_ndjson_log(audit_path, manifest_path, signer=ed25519_key, previous_signers=[hmac_key])
        assert head == manifest_hash(manifests[-1])

    def test_the_log_then_rotates_back_to_a_new_hmac_key_and_verifies_with_the_full_keyring(
        self, tmp_path: Path
    ) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _mixed_algorithm_log(tmp_path)

        assert [line["signature"]["algorithm"] for line in _read_lines(manifest_path)] == [
            "HMAC-SHA256",
            "Ed25519",
            "Ed25519",
            "HMAC-SHA256",
            "HMAC-SHA256",
        ]
        head = verify_ndjson_log(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])
        assert head == manifest_hash(_read_lines(manifest_path)[-1])
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])
        assert (coverage.head, coverage.sealed_runs) == (head, 3)

    def test_each_rotation_entry_is_co_signed_with_the_outgoing_keys_own_algorithm(self, tmp_path: Path) -> None:
        _, manifest_path, *_ = _mixed_algorithm_log(tmp_path)

        entries = [line for line in _read_lines(manifest_path) if line.get("kind") == "key_rotation"]

        assert [
            (entry["previous_key_signature"]["key_id"], entry["previous_key_signature"]["algorithm"])
            for entry in entries
        ] == [
            ("key-1", "HMAC-SHA256"),
            ("key-2", "Ed25519"),
        ]
        assert [entry["signature"]["algorithm"] for entry in entries] == ["Ed25519", "HMAC-SHA256"]

    def test_a_co_signature_relabelled_to_another_algorithm_is_rejected(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _mixed_algorithm_log(tmp_path)
        lines = _read_lines(manifest_path)
        lines[1]["previous_key_signature"]["algorithm"] = "Ed25519"
        _rewrite_lines(manifest_path, lines)

        with pytest.raises(ManifestVerificationError, match="algorithm"):
            verify_ndjson_log(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2])

    @pytest.mark.parametrize("retired", [0, 1], ids=["hmac-key", "ed25519-key"])
    def test_a_retired_key_cannot_seal_after_its_rotation_entry(self, tmp_path: Path, retired: int) -> None:
        audit_path, manifest_path, *keys = _mixed_algorithm_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 4)])
        before = _snapshot(tmp_path)
        others = [key for index, key in enumerate(keys) if index != retired]

        with pytest.raises(ManifestVerificationError, match=_under_key("key-3", keys[retired].key_id)):
            seal_ndjson_runs(audit_path, manifest_path, signer=keys[retired], previous_signers=others)

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("retired", [0, 1], ids=["hmac-key", "ed25519-key"])
    def test_a_seal_by_a_retired_key_after_the_last_rotation_fails_verification(
        self, tmp_path: Path, retired: int
    ) -> None:
        audit_path, manifest_path, *keys = _mixed_algorithm_log(tmp_path)
        record = _record("run-d", 4)
        _write_records(audit_path, [record])
        head = manifest_hash(_read_lines(manifest_path)[-1])
        _write_records(
            manifest_path,
            [seal_run([record], run_id="run-d", signer=keys[retired], previous_manifest_hash=head)],
        )

        with pytest.raises(ManifestVerificationError, match="run-d") as excinfo:
            verify_ndjson_log(audit_path, manifest_path, signer=keys[2], previous_signers=keys[:2])

        _assert_names(excinfo, keys[retired].key_id, "key-3")

    @pytest.mark.parametrize("retired", [0, 1], ids=["hmac-key", "ed25519-key"])
    def test_rotating_back_to_a_retired_key_is_refused_whatever_its_algorithm(
        self, tmp_path: Path, retired: int
    ) -> None:
        _, manifest_path, *keys = _mixed_algorithm_log(tmp_path)
        target = keys[retired]
        before = manifest_path.read_bytes()
        head = manifest_hash(_read_lines(manifest_path)[-1])

        with pytest.raises(ManifestVerificationError, match=target.key_id):
            rotate_manifest_key(
                manifest_path,
                signer=target,
                previous_signers=[key for key in keys if key is not target],
                expected_head=head,
            )

        assert manifest_path.read_bytes() == before

    @_current_key_required
    def test_an_ed25519_key_listed_beside_an_hmac_key_of_the_same_key_id_is_a_value_error(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        audit_path, manifest_path, key_1, key_2, key_3 = _mixed_algorithm_log(tmp_path)
        _write_records(audit_path, [_record("run-d", 4)])
        before = _snapshot(tmp_path)
        clash = Ed25519Signer(_THIRD_KEY, "key-1")

        with pytest.raises(ValueError, match="previous_signers repeats key_id 'key-1'") as excinfo:
            call(audit_path, manifest_path, signer=key_3, previous_signers=[key_1, key_2, clash])

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before

    def test_rotating_to_an_ed25519_key_listed_beside_the_hmac_key_of_the_same_key_id_is_a_value_error(
        self, tmp_path: Path
    ) -> None:
        _, manifest_path = _build_log(tmp_path, ("seal", HmacSha256Signer(_KEY, "key-1")))
        before = manifest_path.read_bytes()
        head = manifest_hash(_read_lines(manifest_path)[-1])

        with pytest.raises(ValueError, match="previous_signers repeats key_id 'key-1'") as excinfo:
            rotate_manifest_key(
                manifest_path,
                signer=Ed25519Signer(_OTHER_KEY, "key-1"),
                previous_signers=[HmacSha256Signer(_KEY, "key-1")],
                expected_head=head,
            )

        assert not isinstance(excinfo.value, (ManifestVerificationError, KeyAlreadyCurrentError))
        assert manifest_path.read_bytes() == before

    @_current_key_required
    def test_an_ed25519_signer_reusing_the_key_id_of_an_hmac_sealed_log_fails_on_the_algorithm(
        self, tmp_path: Path, call: Callable[..., Any]
    ) -> None:
        audit_path, manifest_path = _build_log(tmp_path, ("seal", HmacSha256Signer(_KEY, "key-1")))
        _write_records(audit_path, [_record("run-d", 7)])
        before = _snapshot(tmp_path)
        mismatch = "signature algorithm 'HMAC-SHA256' is not the signer's 'Ed25519'"

        with pytest.raises(ManifestVerificationError, match=re.escape(mismatch)):
            call(audit_path, manifest_path, signer=Ed25519Signer(_KEY, "key-1"))

        assert _snapshot(tmp_path) == before

    def test_rotating_an_hmac_sealed_log_to_an_ed25519_key_of_the_same_key_id_fails_on_the_algorithm(
        self, tmp_path: Path
    ) -> None:
        _, manifest_path = _build_log(tmp_path, ("seal", HmacSha256Signer(_KEY, "key-1")))
        before = manifest_path.read_bytes()
        head = manifest_hash(_read_lines(manifest_path)[-1])
        mismatch = "signature algorithm 'HMAC-SHA256' is not the signer's 'Ed25519'"

        with pytest.raises(ManifestVerificationError, match=re.escape(mismatch)):
            rotate_manifest_key(
                manifest_path, signer=Ed25519Signer(_OTHER_KEY, "key-1"), previous_signers=[], expected_head=head
            )

        assert manifest_path.read_bytes() == before

    @pytest.mark.parametrize("public_only", [False, True], ids=["private", "public-only"])
    def test_a_manifest_hmac_signed_under_the_public_key_is_rejected_by_the_ed25519_signer(
        self, public_only: bool
    ) -> None:
        """An attacker who knows only the public key HMACs with it and claims the Ed25519 key_id."""
        records = [_record()]
        verifier = _verify_only() if public_only else Ed25519Signer(_KEY, "key-1")
        forged = seal_run(records, run_id="run-1", signer=HmacSha256Signer(_public_key(_KEY), "key-1"))
        assert forged["signature"]["algorithm"] == "HMAC-SHA256"

        with pytest.raises(ManifestVerificationError, match="algorithm"):
            verify_manifest(forged, records, signer=verifier)

    def test_an_ed25519_manifest_is_rejected_by_an_hmac_signer_of_the_same_key_id(self) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=Ed25519Signer(_KEY, "key-1"))

        with pytest.raises(ManifestVerificationError, match="algorithm"):
            verify_manifest(manifest, records, signer=HmacSha256Signer(_KEY, "key-1"))


_ED25519_EXTRA = re.escape("mloda-enterprise[ed25519]")


_NON_FINITE_LINES: dict[str, bytes] = {
    "nan": b'{"run_id": "run-d", "x": NaN}\n',
    "infinity": b'{"run_id": "run-d", "x": Infinity}\n',
    "negative-infinity": b'{"run_id": "run-d", "x": -Infinity}\n',
}


@_both_algorithms
class TestSignedPayloadV2:
    """A v2 signature covers the signature block's key_id and algorithm."""

    def test_relabeled_key_id_sharing_one_secret_fails_verification(self) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer(key_id="key-1"))
        relabeled = {**manifest, "signature": {**manifest["signature"], "key_id": "key-2"}}
        verifier = _signer(key_id="key-2")

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(relabeled, records, signer=verifier, previous_signers=[_signer(key_id="key-1")])

    def test_relabeled_key_id_in_a_log_fails_verification(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        lines = _read_lines(manifest_path)
        lines[-1]["signature"]["key_id"] = "key-2"
        _rewrite_lines(manifest_path, lines)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(
                audit_path, manifest_path, signer=_signer(key_id="key-2"), previous_signers=[_signer(key_id="key-1")]
            )

    def test_changed_signature_algorithm_fails_verification(self) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        tampered = {**manifest, "signature": {**manifest["signature"], "algorithm": "OTHER"}}

        with pytest.raises(ManifestVerificationError, match="signature|algorithm"):
            verify_manifest(tampered, records, signer=_signer())

    def test_signature_covers_key_id_and_algorithm_but_not_the_value(self) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer())
        other = {**manifest, "signature": {**manifest["signature"], "key_id": "key-2"}}

        assert _signing_payload_ref(manifest) != _signing_payload_ref(other)
        assert _signing_payload_ref(manifest) == _signing_payload_ref(
            {**manifest, "signature": {**manifest["signature"], "value": "x"}}
        )

    def test_relabeled_rotation_entry_key_id_fails_verification(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        _rotate(manifest_path, new_signer, old_signer)
        lines = _read_lines(manifest_path)
        lines[-1]["signature"]["key_id"] = "key-3"
        _rewrite_lines(manifest_path, lines)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(
                audit_path, manifest_path, signer=_signer(_OTHER_KEY, "key-3"), previous_signers=[old_signer]
            )

    def test_a_previous_key_signature_member_on_a_non_rotation_entry_stays_in_the_signed_bytes(self) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        extra = {"algorithm": "x", "key_id": "k", "value": "original"}
        entry = {**manifest, "previous_key_signature": extra}
        reduced = {**entry, "signature": {"algorithm": entry["signature"]["algorithm"], "key_id": "key-1"}}
        entry["signature"] = {**entry["signature"], "value": _signer().sign(_canonical(reduced))}
        verify_manifest(entry, records, signer=_signer())  # control: correctly signed with the member as is

        tampered = {**entry, "previous_key_signature": {**extra, "value": "changed"}}

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest(tampered, records, signer=_signer())

    def test_manifest_version_3_reports_unsupported_even_with_a_bad_signature(self) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())
        forged = {**manifest, "manifest_version": 3}

        with pytest.raises(ManifestVerificationError, match="unsupported manifest_version 3"):
            verify_manifest(forged, records, signer=_signer())

    def test_manifest_version_3_in_a_log_reports_unsupported(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        lines = _read_lines(manifest_path)
        lines[-1]["manifest_version"] = 3
        _rewrite_lines(manifest_path, lines)

        with pytest.raises(ManifestVerificationError, match="unsupported manifest_version 3"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())


@_both_algorithms
class TestV1ReadPath:
    """v1 lines verify only as a prefix before the first v2 line, and new lines are v2."""

    def test_a_v1_log_with_a_v1_rotation_verifies(self, tmp_path: Path) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        steps = [("seal-v1", old_signer), ("seal-v1", old_signer), ("rotate-v1", new_signer)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)
        assert {line["manifest_version"] for line in _read_lines(manifest_path)} == {1}
        head = manifest_hash(_read_lines(manifest_path)[-1])

        result = verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert result == head

    def test_new_seals_appended_to_a_v1_log_are_v2_and_the_whole_log_verifies(self, tmp_path: Path) -> None:
        old_signer, new_signer = _signer(), _signer(_OTHER_KEY, "key-2")
        steps = [("seal-v1", old_signer), ("seal-v1", old_signer), ("rotate-v1", new_signer)]
        audit_path, manifest_path = _build_log(tmp_path, *steps)
        _write_records(audit_path, [_record("run-4", 4)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

        assert [(manifest["run_id"], manifest["manifest_version"]) for manifest in manifests] == [("run-4", 2)]
        assert [line["manifest_version"] for line in _read_lines(manifest_path)] == [1, 1, 1, 2]
        verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    def test_a_v1_seal_verifies_alone(self) -> None:
        records = [_record()]
        manifest = _as_v1(seal_run(records, run_id="run-1", signer=_signer()), _signer())

        verify_manifest(manifest, records, signer=_signer())

    def test_a_v1_line_after_a_v2_line_is_rejected(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _build_log(tmp_path, ("seal", _signer()), ("seal-v1", _signer()))

        with pytest.raises(ManifestVerificationError, match="manifest_version"):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    def test_a_v1_rotation_after_a_v2_line_is_rejected(self, tmp_path: Path) -> None:
        new_signer = _signer(_OTHER_KEY, "key-2")
        audit_path, manifest_path = _build_log(tmp_path, ("seal", _signer()), ("rotate-v1", new_signer))

        with pytest.raises(ManifestVerificationError, match="manifest_version"):
            verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[_signer()])

    def test_a_v1_line_after_a_v2_line_stops_sealing(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _build_log(tmp_path, ("seal", _signer()), ("seal-v1", _signer()))
        _write_records(audit_path, [_record("run-9", 9)])
        before = manifest_path.read_bytes()

        with pytest.raises(ManifestVerificationError):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer())

        assert manifest_path.read_bytes() == before


@_both_algorithms
class TestSealedLate:
    def test_seal_run_defaults_to_late(self) -> None:
        assert seal_run([_record()], run_id="run-1", signer=_signer())["sealed_late"] is True

    @pytest.mark.parametrize("value", [True, False])
    def test_seal_run_records_the_argument(self, value: bool) -> None:
        manifest = seal_run([_record()], run_id="run-1", signer=_signer(), sealed_late=value)

        assert manifest["sealed_late"] is value

    def test_seal_ndjson_runs_defaults_to_late(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)

        assert [line["sealed_late"] for line in _read_lines(manifest_path)] == [True, True, True]

    def test_seal_ndjson_runs_records_the_argument(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _record("run-b", 2)])

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), sealed_late=False)

        assert [manifest["sealed_late"] for manifest in manifests] == [False, False]
        assert [line["sealed_late"] for line in _read_lines(manifest_path)] == [False, False]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("value", [True, False])
    def test_flipping_sealed_late_breaks_the_signature(self, value: bool) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer(), sealed_late=value)

        with pytest.raises(ManifestVerificationError, match="signature"):
            verify_manifest({**manifest, "sealed_late": not value}, records, signer=_signer())

    @pytest.mark.parametrize("value", [1, 0, "yes", None, [True]], ids=["one", "zero", "str", "none", "list"])
    def test_correctly_signed_non_bool_sealed_late_is_rejected(self, value: Any) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError, match="sealed_late"):
            verify_manifest(_resigned(manifest, sealed_late=value), records, signer=_signer())


_FUTURE_TIME = "2999-01-01T00:00:00.000000Z"


def _at(record: dict[str, Any], event_time: Any) -> dict[str, Any]:
    return {**record, "event_time": event_time}


@_both_algorithms
class TestSealNdjsonRunsOlderThan:
    """The manual stale-run sweep: seal only runs whose newest record is older than `older_than`."""

    _LOGGER = "mloda.enterprise.extenders.audit.run_manifest"

    def _stale_and_fresh(self, tmp_path: Path) -> tuple[Path, Path]:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(
            audit_path,
            [_record("run-old", 1), _at(_record("run-fresh", 2), _FUTURE_TIME), _record("run-old", 3)],
        )
        return audit_path, tmp_path / "manifests.ndjson"

    def test_seals_only_runs_older_than_the_threshold(self, tmp_path: Path) -> None:
        audit_path, manifest_path = self._stale_and_fresh(tmp_path)

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), older_than=timedelta(days=1))

        assert [(manifest["run_id"], manifest["record_count"]) for manifest in manifests] == [("run-old", 2)]
        assert [line["run_id"] for line in _read_lines(manifest_path)] == ["run-old"]

    def test_a_later_sweep_seals_the_runs_left_unsealed(self, tmp_path: Path) -> None:
        audit_path, manifest_path = self._stale_and_fresh(tmp_path)
        seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), older_than=timedelta(days=1))

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), older_than=timedelta(0))
        assert [manifest["run_id"] for manifest in manifests] == []

        later = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        assert [manifest["run_id"] for manifest in later] == ["run-fresh"]

    def test_the_newest_record_decides_not_the_first(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_record("run-a", 1), _at(_record("run-a", 2), _FUTURE_TIME)])

        manifests = seal_ndjson_runs(
            audit_path, tmp_path / "manifests.ndjson", signer=_signer(), older_than=timedelta(days=1)
        )

        assert manifests == []

    def test_a_threshold_longer_than_the_age_of_every_run_seals_nothing(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])

        manifests = seal_ndjson_runs(
            audit_path, tmp_path / "manifests.ndjson", signer=_signer(), older_than=timedelta(days=365 * 100)
        )

        assert manifests == []

    def test_sweep_manifests_are_sealed_late_by_default(self, tmp_path: Path) -> None:
        audit_path, manifest_path = self._stale_and_fresh(tmp_path)

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), older_than=timedelta(days=1))

        assert [manifest["sealed_late"] for manifest in manifests] == [True]

    def test_is_an_ordinary_seal_with_anchored_heads_and_log_id(self, tmp_path: Path) -> None:
        audit_path, manifest_path = self._stale_and_fresh(tmp_path)
        first = seal_ndjson_runs(
            audit_path, manifest_path, signer=_signer(), log_id="log-a", older_than=timedelta(days=1)
        )
        assert [manifest["run_id"] for manifest in first] == ["run-old"]
        anchored = _log_heads(manifest_path)

        later = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), log_id="log-a", anchored_heads=anchored)

        assert [manifest["run_id"] for manifest in later] == ["run-fresh"]
        verify_ndjson_log(audit_path, manifest_path, signer=_signer(), log_id="log-a", anchored_heads=anchored)

    def test_the_z_suffixed_event_time_of_audit_records_is_parsed(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(audit_path, [_at(_record("run-a", 1), "2026-01-01T00:00:01Z")])

        manifests = seal_ndjson_runs(
            audit_path, tmp_path / "manifests.ndjson", signer=_signer(), older_than=timedelta(days=1)
        )

        assert [manifest["run_id"] for manifest in manifests] == ["run-a"]

    @pytest.mark.parametrize("event_time", ["not a time", 5, None, ""], ids=["garbage", "int", "none", "empty"])
    def test_a_run_with_an_unparseable_event_time_is_skipped_with_one_warning_giving_the_count(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, event_time: Any
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        _write_records(
            audit_path,
            [
                _record("run-ok", 1),
                _record("run-bad-1", 2),
                _at(_record("run-bad-1", 3), event_time),
                _at(_record("run-bad-2", 4), event_time),
            ],
        )

        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            manifests = seal_ndjson_runs(
                audit_path, tmp_path / "manifests.ndjson", signer=_signer(), older_than=timedelta(days=1)
            )

        assert [manifest["run_id"] for manifest in manifests] == ["run-ok"]
        warnings = [r for r in caplog.records if r.name == self._LOGGER and r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "2" in warnings[0].getMessage()

    @pytest.mark.parametrize(
        ("event_time", "expected"),
        [
            ("2026-10-03T12:00:00-05:00", datetime(2026, 10, 3, 17, 0, tzinfo=timezone.utc)),
            ("2026-10-03T12:00:00+02:00", datetime(2026, 10, 3, 10, 0, tzinfo=timezone.utc)),
            ("2026-10-03T12:00:00", datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)),
            ("2026-10-03T12:00:00Z", datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)),
        ],
        ids=["negative-offset", "positive-offset", "naive", "z-suffix"],
    )
    def test_event_time_offsets_are_converted_to_utc(self, event_time: str, expected: datetime) -> None:
        assert _parse_event_time(event_time) == expected

    def test_a_negative_offset_event_time_is_compared_as_the_utc_instant(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        # The wall clock is 30h old, but at -12:00 the instant is 18h old: newer than the one day threshold.
        wall = (datetime.now(timezone.utc) - timedelta(hours=30)).replace(tzinfo=None)
        _write_records(audit_path, [_at(_record("run-a", 1), wall.isoformat() + "-12:00")])

        manifests = seal_ndjson_runs(
            audit_path, tmp_path / "manifests.ndjson", signer=_signer(), older_than=timedelta(days=1)
        )

        assert manifests == []

    def test_a_run_without_an_event_time_key_is_skipped(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        bare = {key: value for key, value in _record("run-bad", 1).items() if key != "event_time"}
        _write_records(audit_path, [bare, _record("run-ok", 2)])

        manifests = seal_ndjson_runs(
            audit_path, tmp_path / "manifests.ndjson", signer=_signer(), older_than=timedelta(days=1)
        )

        assert [manifest["run_id"] for manifest in manifests] == ["run-ok"]

    def test_no_warning_when_every_run_has_an_event_time(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        audit_path, manifest_path = self._stale_and_fresh(tmp_path)

        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), older_than=timedelta(days=1))

        assert [r for r in caplog.records if r.name == self._LOGGER] == []

    def test_requires_no_run_id(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])

        with pytest.raises(ValueError, match="older_than"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), run_id="run-a", older_than=timedelta(days=1))

        assert not manifest_path.exists()

    @pytest.mark.parametrize(
        "value", [timedelta(seconds=-1), 3600, 1.5, "1d", True], ids=["negative", "int", "float", "str", "bool"]
    )
    def test_rejects_a_non_timedelta_or_negative_value(self, tmp_path: Path, value: Any) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])

        with pytest.raises(ValueError, match="older_than"):
            seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), older_than=value)

        assert not manifest_path.exists()


@_both_algorithms
class TestNonFiniteNumbers:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
    def test_seal_run_rejects_a_non_finite_record(self, value: float) -> None:
        with pytest.raises(ValueError):
            seal_run([{**_record(), "x": value}], run_id="run-1", signer=_signer())

    def test_verify_manifest_maps_a_nan_record_to_a_verification_error(self) -> None:
        records = [_record()]
        manifest = seal_run(records, run_id="run-1", signer=_signer())

        with pytest.raises(ManifestVerificationError):
            verify_manifest(manifest, [{**records[0], "x": float("nan")}], signer=_signer())

    @pytest.mark.parametrize("damage", list(_NON_FINITE_LINES.values()), ids=list(_NON_FINITE_LINES))
    def test_a_non_finite_audit_line_fails_verification(self, tmp_path: Path, damage: bytes) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(audit_path, 2, damage)

        with pytest.raises(ManifestVerificationError):
            verify_ndjson_log(audit_path, manifest_path, signer=_signer())

    @pytest.mark.parametrize("damage", list(_NON_FINITE_LINES.values()), ids=list(_NON_FINITE_LINES))
    def test_a_non_finite_audit_line_counts_as_damaged_for_quarantine(self, tmp_path: Path, damage: bytes) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        clean = audit_path.read_bytes()
        _insert_line(audit_path, 2, damage)
        damaged = audit_path.read_bytes()

        removed = _quarantine(tmp_path, audit_path, manifest_path)

        assert _summary(removed) == _spans("audit", damaged, [3])
        assert audit_path.read_bytes() == clean
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == head

    @pytest.mark.parametrize("damage", list(_NON_FINITE_LINES.values()), ids=list(_NON_FINITE_LINES))
    def test_a_non_finite_manifest_line_is_not_repaired(self, tmp_path: Path, damage: bytes) -> None:
        audit_path, manifest_path = _sealed_log(tmp_path)
        _insert_line(manifest_path, 1, damage)

        _assert_refused(tmp_path, audit_path, manifest_path)


def _audit_package_without_cryptography(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The audit package, cold-imported while `cryptography` cannot be imported."""
    block_root(monkeypatch, "cryptography")
    evict_package(monkeypatch, "mloda.enterprise.extenders.audit")
    return importlib.import_module("mloda.enterprise.extenders.audit")


class TestEd25519WithoutCryptography:
    """cryptography is an optional extra: only building an Ed25519Signer needs it."""

    def test_the_package_imports_and_hmac_signing_still_works(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        package = _audit_package_without_cryptography(monkeypatch)
        signer = package.HmacSha256Signer(_KEY, "key-1")
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        _write_records(audit_path, [_record("run-a", 1)])

        manifests = package.seal_ndjson_runs(audit_path, manifest_path, signer=signer)

        assert package.verify_ndjson_log(audit_path, manifest_path, signer=signer) == manifest_hash(manifests[-1])
        assert "Ed25519Signer" in package.__all__

    def test_building_an_ed25519_signer_raises_import_error_naming_the_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        package = _audit_package_without_cryptography(monkeypatch)

        with pytest.raises(ImportError, match=_ED25519_EXTRA):
            package.Ed25519Signer(_KEY, "k")
        with pytest.raises(ImportError, match=_ED25519_EXTRA):
            package.Ed25519Signer.from_public_key(_RFC8032_PUBLIC_KEY, "k")

    def test_the_import_is_not_cached_so_building_works_again_once_cryptography_is_back(self) -> None:
        public_key = _public_key(_KEY)
        # Built once before the block: a cached import would let the blocked builds below through.
        Ed25519Signer(_KEY, "k")
        Ed25519Signer.from_public_key(public_key, "k")

        with pytest.MonkeyPatch.context() as blocked:
            block_root(blocked, "cryptography")
            with pytest.raises(ImportError, match=_ED25519_EXTRA):
                Ed25519Signer(_KEY, "k")
            with pytest.raises(ImportError, match=_ED25519_EXTRA):
                Ed25519Signer.from_public_key(public_key, "k")

        signature = Ed25519Signer(_KEY, "k").sign(b"payload")
        assert Ed25519Signer.from_public_key(public_key, "k").verify(b"payload", signature) is True


@_both_algorithms
class TestQuarantineFromRotationEntry:
    """Drops a rotation entry and everything after it, anchored on the head just before the entry."""

    def test_the_entry_and_everything_after_it_are_dropped_and_the_honest_key_continues(self, tmp_path: Path) -> None:
        audit_path, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        # The control: the entry wedges the log for the honest key.
        with pytest.raises(ManifestVerificationError, match=_under_key("key-2", "key-1")):
            verify_ndjson_log(
                audit_path, manifest_path, signer=_signer(), previous_signers=[_signer(_OTHER_KEY, "key-2")]
            )
        before = _snapshot(tmp_path)
        manifest_before = before["manifests.ndjson"]

        removed = _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

        assert _summary(removed) == _spans("manifest", manifest_before, [3, 4])
        assert all("line 3" in item.reason for item in removed)
        assert manifest_path.read_bytes() == manifest_before[: removed[0].offset]
        assert audit_path.read_bytes() == before["audit.ndjson"]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == anchor
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=_signer())
        assert coverage.unsealed_lines == {"run-4": 1}
        _write_records(audit_path, [_record("run-5", 9)])
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer(), expected_head=anchor, run_id="run-5")
        assert [(manifest["run_id"], manifest["previous_manifest_hash"]) for manifest in manifests] == [
            ("run-5", anchor)
        ]

    def test_trace_has_one_signed_entry_per_dropped_line_with_its_raw_bytes(self, tmp_path: Path) -> None:
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        manifest_before = manifest_path.read_bytes()

        removed = _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

        assert len(removed) == 2
        _assert_traced(tmp_path, manifest_path, manifest_before, removed)

    @pytest.mark.parametrize("forge", list(_FORGED_ENTRIES.values()), ids=list(_FORGED_ENTRIES))
    def test_an_entry_is_dropped_whatever_is_wrong_with_it(
        self, tmp_path: Path, forge: Callable[[str], dict[str, Any]]
    ) -> None:
        audit_path, manifest_path = _build_log(tmp_path, ("seal", _signer()), ("seal", _signer()))
        anchor = manifest_hash(_read_lines(manifest_path)[-1])
        _append_line(manifest_path, _canonical(forge(anchor)))
        before = _snapshot(tmp_path)

        removed = _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [3])
        assert manifest_path.read_bytes() == before["manifests.ndjson"][: removed[0].offset]
        _assert_traced(tmp_path, manifest_path, before["manifests.ndjson"], removed)
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == anchor

    @pytest.mark.parametrize(
        "after",
        [*_DAMAGED_LINES.values(), b'{"run_id": "run-x", "sig', b'{"run_id": "run-x"}'],
        ids=[*_DAMAGED_LINES, "unterminated-torn", "unterminated-valid-json"],
    )
    def test_a_line_after_the_entry_is_dropped_whatever_it_holds(self, tmp_path: Path, after: bytes) -> None:
        steps = [("seal", _signer()), ("seal", _signer()), ("rotate", _signer(_OTHER_KEY, "key-2"))]
        audit_path, manifest_path = _build_log(tmp_path, *steps)
        anchor = manifest_hash(_read_lines(manifest_path)[1])
        _torn(manifest_path, after)
        before = _snapshot(tmp_path)

        removed = _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [3, 4])
        assert removed[1].length == len(after)
        assert manifest_path.read_bytes() == before["manifests.ndjson"][: removed[0].offset]
        _assert_traced(tmp_path, manifest_path, before["manifests.ndjson"], removed)
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == anchor

    def test_previous_signers_verify_a_prefix_sealed_by_retired_keys(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, key_2, _ = _rotated_three_key_log(tmp_path)
        anchor = manifest_hash(_read_lines(manifest_path)[2])
        before = _snapshot(tmp_path)

        removed = _quarantine_from_entry(
            tmp_path, manifest_path, signer=key_2, previous_signers=[key_1], expected_head=anchor
        )

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [4, 5])
        assert manifest_path.read_bytes() == before["manifests.ndjson"][: removed[0].offset]
        _assert_traced(tmp_path, manifest_path, before["manifests.ndjson"], removed, key_2)
        assert verify_ndjson_log(audit_path, manifest_path, signer=key_2, previous_signers=[key_1]) == anchor

    def test_a_retired_key_can_drop_the_honest_rotations_and_seals_after_its_own(self, tmp_path: Path) -> None:
        audit_path, manifest_path, key_1, *_ = _rotated_three_key_log(tmp_path)
        anchor = manifest_hash(_read_lines(manifest_path)[0])
        before = _snapshot(tmp_path)

        # The documented hazard: a retired key with write access drops honest lines too.
        removed = _quarantine_from_entry(tmp_path, manifest_path, signer=key_1, expected_head=anchor)

        assert _summary(removed) == _spans("manifest", before["manifests.ndjson"], [2, 3, 4, 5])
        assert manifest_path.read_bytes() == before["manifests.ndjson"][: removed[0].offset]
        _assert_traced(tmp_path, manifest_path, before["manifests.ndjson"], removed, key_1)
        assert verify_ndjson_log(audit_path, manifest_path, signer=key_1) == anchor
        coverage = verify_ndjson_log_coverage(audit_path, manifest_path, signer=key_1)
        assert coverage.unsealed_lines == {"run-b": 1, "run-c": 1}

    def test_failed_trace_write_leaves_the_manifest_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        before = _snapshot(tmp_path)

        def disk_full(*args: Any, **kwargs: Any) -> None:
            raise OSError(errno.ENOSPC, "disk full")

        _patch_bindings(monkeypatch, "_append_records", disk_full)

        with pytest.raises(OSError, match="disk full"):
            _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

        assert _snapshot(tmp_path) == before

    def test_dry_run_returns_what_a_real_run_drops_and_changes_nothing(self, tmp_path: Path) -> None:
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        before = _snapshot(tmp_path)

        planned = _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor, dry_run=True)

        assert _summary(planned) == _spans("manifest", before["manifests.ndjson"], [3, 4])
        assert _snapshot(tmp_path) == before
        assert planned == _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

    def test_dry_run_does_not_wait_for_a_reader_holding_a_shared_lock(self, tmp_path: Path) -> None:
        fcntl = pytest.importorskip("fcntl")
        _audit_path, manifest_path, anchor = _log_with_rotation_entry(tmp_path)

        with ThreadPoolExecutor(max_workers=1) as pool:
            fd = os.open(manifest_path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
                future = pool.submit(
                    _quarantine_from_entry, tmp_path, manifest_path, expected_head=anchor, dry_run=True
                )
                done, _ = wait([future], timeout=5)
            finally:
                os.close(fd)

            assert done
            assert len(future.result()) == 2

    def test_a_real_run_holds_an_exclusive_lock_on_the_manifest_log_while_it_repairs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        probes: list[tuple[str, bool]] = []
        real_truncate = os.truncate

        def probe_signing(payload: bytes) -> None:
            probes.append(("sign", _lock_refused(manifest_path)))

        def spy_append(*args: Any, **kwargs: Any) -> None:
            probes.append(("append", _lock_refused(manifest_path)))
            _append_records(*args, **kwargs)

        def spy_truncate(*args: Any, **kwargs: Any) -> None:
            probes.append(("truncate", _lock_refused(manifest_path)))
            real_truncate(*args, **kwargs)

        _patch_bindings(monkeypatch, "_append_records", spy_append)
        monkeypatch.setattr(os, "truncate", spy_truncate)

        # Verifying and the trace both use the signer, so the wrapper probes both hooks.
        probing = _HookedSigner(_signer(), on_sign=probe_signing, on_verify=probe_signing)
        removed = _quarantine_from_entry(tmp_path, manifest_path, signer=probing, expected_head=anchor)

        assert len(removed) == 2
        assert {stage for stage, _ in probes} == {"sign", "append", "truncate"}
        assert all(refused for _, refused in probes)
        fd = os.open(manifest_path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    def test_drops_the_entry_where_fcntl_is_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "fcntl", None)
        audit_path, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        manifest_before = manifest_path.read_bytes()
        assert not _trace(tmp_path).exists()

        removed = _quarantine_from_entry(tmp_path, manifest_path, expected_head=anchor)

        assert _summary(removed) == _spans("manifest", manifest_before, [3, 4])
        assert manifest_path.read_bytes() == manifest_before[: removed[0].offset]
        assert verify_ndjson_log(audit_path, manifest_path, signer=_signer()) == anchor
        assert len(_read_lines(_trace(tmp_path))) == 2

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_an_anchor_with_nothing_after_it_returns_nothing_and_creates_no_trace(
        self, tmp_path: Path, dry_run: bool
    ) -> None:
        _, manifest_path = _sealed_log(tmp_path)
        head = manifest_hash(_read_lines(manifest_path)[-1])
        before = _snapshot(tmp_path)

        assert _quarantine_from_entry(tmp_path, manifest_path, expected_head=head, dry_run=dry_run) == []

        assert _snapshot(tmp_path) == before
        assert not _trace(tmp_path).exists()

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    @pytest.mark.parametrize("refusal", list(_REFUSAL_REASONS))
    def test_a_log_that_cannot_be_dropped_from_is_refused_and_nothing_changes(
        self, tmp_path: Path, refusal: str, dry_run: bool
    ) -> None:
        _, manifest_path, anchor = _log_with_rotation_entry(tmp_path)
        lines = manifest_path.read_bytes().splitlines(keepends=True)
        prefix, entry = b"".join(lines[:2]), lines[2]
        signer: ManifestSigner = _signer()
        previous_signers: list[ManifestSigner] = []
        names: list[str] = []
        if refusal == "anchor-not-in-the-log":
            anchor = "0" * 64
            names = [anchor]
        if refusal == "first-dropped-line-is-a-seal":
            anchor = manifest_hash(json.loads(lines[0]))
        if refusal == "first-dropped-line-is-torn":
            manifest_path.write_bytes(prefix + entry[: len(entry) // 2])
        if refusal == "first-dropped-line-is-an-unterminated-entry":
            manifest_path.write_bytes(prefix + entry.removesuffix(b"\n"))
        if refusal == "first-dropped-line-is-not-json":
            manifest_path.write_bytes(prefix + b"garbage\n" + entry)
        if refusal == "first-dropped-line-is-not-an-entry":
            manifest_path.write_bytes(prefix + b'{"run_id": "run-x"}\n' + entry)
        if refusal == "signer-is-not-current-at-the-anchor":
            signer = _signer(_OTHER_KEY, "key-2")
            previous_signers = [_signer()]
            names = ["key-1", "key-2"]
        if refusal == "prefix-does-not-verify":
            edited = lines[0].replace(b'"compliant": true', b'"compliant": false')
            assert edited != lines[0]
            manifest_path.write_bytes(edited + b"".join(lines[1:]))
        if refusal == "quarantine-log-without-a-final-newline":
            _trace(tmp_path).write_bytes(b'{"quarantine_version": 1, "fi')
            names = [str(_trace(tmp_path))]

        excinfo = _assert_raises_and_unchanged(
            tmp_path,
            lambda: _quarantine_from_entry(
                tmp_path,
                manifest_path,
                signer=signer,
                previous_signers=previous_signers,
                expected_head=anchor,
                dry_run=dry_run,
            ),
        )

        _assert_names(excinfo, _REFUSAL_REASONS[refusal], *names)

    @pytest.mark.parametrize("dry_run", [False, True], ids=["repair", "dry-run"])
    def test_a_missing_manifest_log_is_refused_and_no_file_is_created(self, tmp_path: Path, dry_run: bool) -> None:
        _assert_raises_and_unchanged(
            tmp_path,
            lambda: _quarantine_from_entry(
                tmp_path, tmp_path / "manifests.ndjson", expected_head="0" * 64, dry_run=dry_run
            ),
        )

        assert _snapshot(tmp_path) == {}

    def test_omitting_expected_head_is_a_type_error(self, tmp_path: Path) -> None:
        manifest_path = _log_with_rotation_entry(tmp_path)[1]
        before = _snapshot(tmp_path)

        with pytest.raises(TypeError):
            quarantine_from_rotation_entry(  # type: ignore[call-arg]
                manifest_path, quarantine_path=_trace(tmp_path), signer=_signer()
            )

        assert _snapshot(tmp_path) == before

    @pytest.mark.parametrize("expected_head", [None, b"0" * 64], ids=["none", "bytes"])
    def test_a_non_string_expected_head_is_a_plain_value_error(self, tmp_path: Path, expected_head: Any) -> None:
        manifest_path = _log_with_rotation_entry(tmp_path)[1]
        before = _snapshot(tmp_path)

        with pytest.raises(ValueError) as excinfo:
            _quarantine_from_entry(tmp_path, manifest_path, expected_head=expected_head)

        assert not isinstance(excinfo.value, ManifestVerificationError)
        assert _snapshot(tmp_path) == before


# (mode, fail_closed) combos for TestRunManifestRunAll.
_RUN_ALL_MODE_AND_FAIL_CLOSED = [
    (ParallelizationMode.SYNC, False),
    (ParallelizationMode.THREADING, False),
    (ParallelizationMode.MULTIPROCESSING, False),
    (ParallelizationMode.MULTIPROCESSING, True),
]


class TestRunManifestRunAll:
    """A real run's audit file seals into one verifiable manifest."""

    @pytest.mark.parametrize(("mode", "fail_closed"), _RUN_ALL_MODE_AND_FAIL_CLOSED)
    def test_run_all_audit_file_seals_and_verifies(
        self, mode: ParallelizationMode, fail_closed: bool, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        # Only MULTIPROCESSING needs the flight_server fixture.
        flight_server = (
            request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
        )
        extender = AuditExtender(sink=NdjsonAuditSink(audit_path), fail_closed=fail_closed)

        with verified_context(tenant_id="tenant-42", project_id="project-7", principal="svc"):
            values = run_value_int(extender, parallelization_modes={mode}, flight_server=flight_server)

        assert values == expected_value_int()
        records = _read_lines(audit_path)
        assert records
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        head = verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert len(manifests) == 1
        manifest = manifests[0]
        assert head == manifest_hash(manifest)
        assert manifest["compliant"] is True
        assert manifest["record_count"] == len(records)
        assert {record["run_id"] for record in records} == {manifest["run_id"]}
        assert [record["policy_version"] for record in records] == [extender.policy_version] * len(records)

    def test_run_all_tee_of_ndjson_and_memory_sinks_seals_and_verifies(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        memory = InMemoryAuditSink()
        extender = AuditExtender(sink=TeeAuditSink(NdjsonAuditSink(audit_path), memory), policy_version=_POLICY_VERSION)

        with verified_context(tenant_id="tenant-42", project_id="project-7", principal="svc"):
            values = run_value_int(extender)

        assert values == expected_value_int()
        records = _read_lines(audit_path)
        assert records
        assert records == memory.records
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        head = verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert len(manifests) == 1
        assert head == manifest_hash(manifests[0])
        assert manifests[0]["record_count"] == len(records)
        assert extender.policy_version == _POLICY_VERSION
        assert [record["policy_version"] for record in records] == [extender.policy_version] * len(records)

    def test_run_all_without_verified_context_seals_a_non_compliant_manifest(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"

        values = run_value_int(AuditExtender(sink=NdjsonAuditSink(audit_path)))

        assert values == expected_value_int()
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert len(manifests) == 1
        assert manifests[0]["compliant"] is False

    def test_run_all_fail_closed_refusal_seals_a_non_compliant_manifest(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"

        with pytest.raises(IdentityRequiredError):
            run_value_int(AuditExtender(sink=NdjsonAuditSink(audit_path), fail_closed=True))

        records = _read_lines(audit_path)
        assert len(records) == 1
        assert records[0]["decision"] == "deny"
        assert records[0]["hook"] == "FEATURE_GROUP_MATCHED"
        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=_signer())
        verify_ndjson_log(audit_path, manifest_path, signer=_signer())

        assert len(manifests) == 1
        assert manifests[0]["compliant"] is False

    @pytest.mark.parametrize(("mode", "fail_closed"), _RUN_ALL_MODE_AND_FAIL_CLOSED)
    def test_run_all_auto_seals_via_on_run_complete_without_a_manual_seal_call(
        self, mode: ParallelizationMode, fail_closed: bool, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        """AuditExtender.on_run_complete seals the run itself; no seal_ndjson_runs call is made here."""
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        # Only MULTIPROCESSING needs the flight_server fixture.
        flight_server = (
            request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
        )
        signer = _signer()
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path),
            audit_path=audit_path,
            manifest_path=manifest_path,
            signer=signer,
            fail_closed=fail_closed,
        )

        with verified_context(tenant_id="tenant-42", project_id="project-7", principal="svc"):
            values = run_value_int(extender, parallelization_modes={mode}, flight_server=flight_server)

        assert values == expected_value_int()
        records = _read_lines(audit_path)
        assert records
        manifests = _read_lines(manifest_path)
        assert len(manifests) == 1
        manifest = manifests[0]
        assert manifest["record_count"] == len(records)
        assert manifest["compliant"] is True
        head = verify_ndjson_log(audit_path, manifest_path, signer=signer)
        assert head == manifest_hash(manifest)
        assert {record["run_id"] for record in records} == {manifest["run_id"]}
        assert [record["policy_version"] for record in records] == [extender.policy_version] * len(records)

    @pytest.mark.parametrize(("mode", "fail_closed"), _RUN_ALL_MODE_AND_FAIL_CLOSED)
    @pytest.mark.parametrize("rerun_with", ["same_instance", "fresh_instance"])
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_run_all_second_run_of_a_prepared_session_is_refused_and_leaves_the_seal_untouched(
        self,
        mode: ParallelizationMode,
        fail_closed: bool,
        rerun_with: str,
        tmp_path: Path,
        request: pytest.FixtureRequest,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A prepared session's run() reuses its run_id, so a second run() must be refused, not resealed,
        whether the refusal comes from the same extender instance or a fresh one built over the same
        sealing config (audit_path/manifest_path/signer), as after a process restart."""
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        # Only MULTIPROCESSING needs the flight_server fixture.
        flight_server = (
            request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
        )
        signer = _signer()
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path),
            audit_path=audit_path,
            manifest_path=manifest_path,
            signer=signer,
            fail_closed=fail_closed,
        )

        with verified_context(tenant_id="tenant-42", project_id="project-7", principal="svc"):
            session = prepare_value_int(extender, parallelization_modes={mode})
            session.run(parallelization_modes={mode}, flight_server=flight_server)

            assert len(_read_lines(manifest_path)) == 1
            audit_before = audit_path.read_bytes()
            manifest_before = manifest_path.read_bytes()

            rerun_extenders: set[Extender] | None = (
                None
                if rerun_with == "same_instance"
                else {
                    AuditExtender(
                        sink=NdjsonAuditSink(audit_path),
                        audit_path=audit_path,
                        manifest_path=manifest_path,
                        signer=signer,
                        fail_closed=fail_closed,
                    )
                }
            )

            with caplog.at_level(logging.INFO):
                with pytest.raises(SealedRunRefusedError):
                    session.run(
                        parallelization_modes={mode},
                        flight_server=flight_server,
                        function_extender=rerun_extenders,
                    )

        assert audit_path.read_bytes() == audit_before
        assert manifest_path.read_bytes() == manifest_before
        verify_ndjson_log(audit_path, manifest_path, signer=signer)
        # A refused re-run wrote nothing new: on_run_complete must not log an ERROR for it.
        assert not any(r.levelno >= logging.ERROR and r.name == audit_extender_module.__name__ for r in caplog.records)
        # It logs an audit_extender INFO instead: core's own ERROR log of the raised SealedRunRefusedError
        # (a different logger) must not stand in for it.
        infos = [r for r in caplog.records if r.levelno == logging.INFO and r.name == audit_extender_module.__name__]
        assert any("was not sealed again" in r.getMessage() for r in infos)

    @_both_algorithms
    def test_run_all_multiprocessing_auto_seal_waits_for_workers_to_be_joined(
        self, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        """BufferingNdjsonAuditSink only reaches disk on a worker's graceful-exit close(): an auto-seal that
        ran before the workers were joined would see an empty (or partial) audit file."""
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        flight_server = request.getfixturevalue("flight_server")
        signer = _signer()
        sink = BufferingNdjsonAuditSink(audit_path)
        extender = AuditExtender(sink=sink, audit_path=audit_path, manifest_path=manifest_path, signer=signer)

        with verified_context(tenant_id="tenant-42"):
            values = run_value_int(
                extender, parallelization_modes={ParallelizationMode.MULTIPROCESSING}, flight_server=flight_server
            )

        assert values == expected_value_int()
        # The sink instance held in THIS (parent) process must never itself have received a direct write():
        # under MULTIPROCESSING, the actual calculation and every sink.write() call happen inside a worker's
        # pickled copy of the sink, and only that worker copy's flush() at graceful exit puts records on disk.
        # If the parent's own sink._buffer were non-empty here, records would have been written synchronously
        # in the parent instead of by a worker, which would prove nothing about waiting for the join.
        assert sink._buffer == []
        records = _read_lines(audit_path)
        assert records
        manifests = _read_lines(manifest_path)
        assert len(manifests) == 1
        manifest = manifests[0]
        assert manifest["record_count"] == len(records)
        assert manifest["compliant"] is True
        head = verify_ndjson_log(audit_path, manifest_path, signer=signer)
        assert head == manifest_hash(manifest)
        assert {record["run_id"] for record in records} == {manifest["run_id"]}
        assert [record["policy_version"] for record in records] == [extender.policy_version] * len(records)

    def test_run_all_auto_seal_threads_previous_signers_through_to_seal_ndjson_runs(self, tmp_path: Path) -> None:
        """A key rotated before the run starts; the extender is given the new signer plus the retired one in
        previous_signers, which seal_ndjson_runs (and thus verify_ndjson_log) needs to verify the log's
        earlier, still-old-key-signed entries. If previous_signers were dropped before reaching
        seal_ndjson_runs, verification below would fail."""
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        old_signer = _signer(key_id="key-1")
        new_signer = _signer(_OTHER_KEY, "key-2")
        _write_records(audit_path, [_record("run-seed")])
        seal_ndjson_runs(audit_path, manifest_path, signer=old_signer)
        _rotate(manifest_path, new_signer, old_signer)
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path),
            audit_path=audit_path,
            manifest_path=manifest_path,
            signer=new_signer,
            previous_signers=(old_signer,),
        )

        with verified_context(tenant_id="tenant-42"):
            values = run_value_int(extender)

        assert values == expected_value_int()
        verify_ndjson_log(audit_path, manifest_path, signer=new_signer, previous_signers=[old_signer])

    def test_run_all_fail_closed_plan_time_refusal_is_not_auto_sealed_but_a_manual_sweep_still_seals_it(
        self, tmp_path: Path
    ) -> None:
        """Documents the limitation: on_run_complete never fires for a plan-time refusal (core's hook contract),
        so the deny record it wrote stays unsealed until a manual seal_ndjson_runs sweep, which still works."""
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifests.ndjson"
        signer = _signer()
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path),
            audit_path=audit_path,
            manifest_path=manifest_path,
            signer=signer,
            fail_closed=True,
        )

        with pytest.raises(IdentityRequiredError):
            run_value_int(extender)

        records = _read_lines(audit_path)
        assert len(records) == 1
        assert records[0]["decision"] == "deny"
        assert records[0]["hook"] == "FEATURE_GROUP_MATCHED"
        assert not manifest_path.exists()

        manifests = seal_ndjson_runs(audit_path, manifest_path, signer=signer)
        verify_ndjson_log(audit_path, manifest_path, signer=signer)

        assert len(manifests) == 1
        assert manifests[0]["compliant"] is False
