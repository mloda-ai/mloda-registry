"""Shared test helpers for the run manifest tests (not collected)."""

from __future__ import annotations

import base64
import builtins
import hashlib
import hmac
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from mloda.enterprise.extenders.audit import (
    Ed25519Signer,
    HeadAnchor,
    HmacSha256Signer,
    ManifestSigner,
    ManifestVerificationError,
    NdjsonAuditSink,
    NdjsonHeadAnchor,
    QuarantinedLine,
    _seal_index,
    _segments,
    _signers,
    _verify,
    manifest_hash,
    quarantine_damaged_lines,
    quarantine_from_rotation_entry,
    rotate_manifest_key,
    rotate_ndjson_segment,
    run_manifest,
    seal_ndjson_runs,
    seal_run,
    verify_quarantine_log,
)
from mloda.enterprise.extenders.audit import _quarantine as _quarantine_module

_SOURCE_MODULES: tuple[ModuleType, ...] = (_signers, _verify, _seal_index, _segments, _quarantine_module, run_manifest)


def _patch_bindings(monkeypatch: pytest.MonkeyPatch, name: str, value: Any) -> None:
    """Patch `name` on each source module sharing the original binding; shadow a builtin on all of them."""
    bound = [module for module in _SOURCE_MODULES if name in vars(module)]
    if not bound:
        assert hasattr(builtins, name), f"no audit source module binds {name!r}"
        for module in _SOURCE_MODULES:
            monkeypatch.setattr(module, name, value, raising=False)
        return
    original = vars(bound[0])[name]
    for module in bound:
        if vars(module)[name] is original:
            monkeypatch.setattr(module, name, value)


def _unpatch_bindings(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Undo a builtin shadow set by `_patch_bindings`."""
    for module in _SOURCE_MODULES:
        monkeypatch.delattr(module, name, raising=False)


_KEY = b"k" * 32


_OTHER_KEY = b"o" * 32


_THIRD_KEY = b"t" * 32


# The signing algorithm `_signer` builds; the `algorithm` fixture flips it for a test.
_ALGORITHM = "hmac"


_EXPECTED_GENESIS_KEYS = {"manifest_version", "kind", "log_id", "created_at", "previous_manifest_hash", "signature"}


_DAMAGED_LINES: dict[str, bytes] = {
    "blank": b"\n",
    "bad-utf8": b"\xff\xfe\n",
    "truncated": b'{"run_id": "run-d"\n',
    "not-an-object": b"[]\n",
    "duplicate-key": b'{"run_id": "run-d", "run_id": "run-e"}\n',
    "deep-nesting": b"[" * 100000 + b"\n",
}


# As a `_rotation_entry` change, drops the key.
_MISSING: Any = object()


# Runs a class once per algorithm.
_both_algorithms = pytest.mark.usefixtures("algorithm")


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


def _damaged_log(directory: Path) -> tuple[Path, Path, str]:
    """A torn manifest log plus an undecodable audit line (line 4) and a torn last audit line (line 8)."""
    audit_path, manifest_path, head = _torn_manifest_log(directory)
    _insert_line(audit_path, 3, b"\xff\xfe\n")
    _torn(audit_path, b'{"run_id": "run-d", "tenant')
    return audit_path, manifest_path, head


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


def _rotate_segment(audit_path: Path, manifest_path: Path, **kwargs: Any) -> dict[str, Any]:
    """rotate_ndjson_segment with the default signer and log_id."""
    entry: dict[str, Any] = rotate_ndjson_segment(
        audit_path, manifest_path, **{"signer": _signer(), "log_id": "log-a", **kwargs}
    )
    return entry


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


def _log_heads(manifest_path: Path) -> list[str]:
    return [manifest_hash(line) for line in _read_lines(manifest_path)]
