"""Tests for the run_manifest public API and the NDJSON sealing built on it."""

from __future__ import annotations

import ast
import copy
import dataclasses
import errno
import importlib
import inspect
import json
import logging
import os
import pickle  # nosec
import stat
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
    IdentityRequiredError,
    KeyAlreadyCurrentError,
    LogCoverage,
    ManifestSigner,
    ManifestVerificationError,
    NdjsonAuditSink,
    RunAlreadySealedError,
    RunNotPendingError,
    SealedRunRefusedError,
    TeeAuditSink,
    _verify,
    manifest_hash,
    quarantine_damaged_lines,
    quarantine_from_rotation_entry,
    rotate_manifest_key,
    seal_ndjson_runs,
    seal_run,
    verify_manifest,
    verify_ndjson_log,
    verify_ndjson_log_coverage,
)
from mloda.enterprise.extenders.audit._records import _parse_event_time
from mloda.enterprise.extenders.audit.audit_extender import _append_records
from mloda.enterprise.extenders.audit.tests.manifest_helpers import (
    _KEY,
    _OTHER_KEY,
    _THIRD_KEY,
    _aliased,
    _append_line,
    _assert_names,
    _both_algorithms,
    _canonical,
    _cap,
    _HookedSigner,
    _key_2_entry,
    _locked_during_digest,
    _log_heads,
    _log_with_rotation_entry,
    _patch_bindings,
    _PrefixSigner,
    _read_lines,
    _record,
    _reference_signature,
    _resigned,
    _rewrite_lines,
    _rotate,
    _rotated_three_key_log,
    _sealed_log,
    _sha256,
    _signer,
    _signing_payload_ref,
    _snapshot,
    _torn,
    _trace,
    _truncated_log,
    _under_key,
    _write_records,
)
from mloda.enterprise.extenders.audit.tests.test_audit_extender import (
    _POLICY_VERSION,
    BufferingNdjsonAuditSink,
    InMemoryAuditSink,
)
from mloda.testing.extenders.runners import expected_value_int, prepare_value_int, run_value_int

_EXPECTED_ROTATION_KEYS = {
    "manifest_version",
    "kind",
    "rotated_at",
    "previous_manifest_hash",
    "previous_key_signature",
    "signature",
}


# The manifest log's writers and the run_id of each line they append (None for a rotation entry).
_WRITERS = ["seal", "rotate"]


_PENDING_RUN_IDS: dict[str, list[str | None]] = {"seal": ["run-d", "run-e", "run-f"], "rotate": [None]}


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


def _defined_names(tree: ast.Module) -> set[str]:
    """Names a module defines at top level (def, class, assignment)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


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

    def test_facade_homes_cover_exactly_the_public_audit_names_of_run_manifest(self) -> None:
        prefix = "mloda.enterprise.extenders.audit"
        public = {
            n
            for n, obj in vars(run_manifest_module).items()
            if not n.startswith("_")
            and (inspect.isclass(obj) or inspect.isfunction(obj))
            and getattr(obj, "__module__", "").startswith(prefix)
        }
        assert {n for n, _ in _FACADE_HOMES} == public | {"MAX_LINE_BYTES"}

    @pytest.mark.parametrize("name, home", _FACADE_HOMES)
    def test_facade_names_are_the_objects_their_defining_module_defines(self, name: str, home: str) -> None:
        defining = importlib.import_module(f"mloda.enterprise.extenders.audit.{home}")
        package = Path(audit_package.__file__ or "").parent
        assert name in _defined_names(ast.parse((package / f"{home}.py").read_text()))
        assert getattr(run_manifest_module, name) is getattr(defining, name)
        if hasattr(audit_package, name):
            assert getattr(audit_package, name) is getattr(run_manifest_module, name)

    @pytest.mark.parametrize("name, home", [(n, h) for n, h in _FACADE_HOMES if n != "MAX_LINE_BYTES"])
    def test_facade_names_report_the_facade_as_their_module(self, name: str, home: str) -> None:
        assert getattr(run_manifest_module, name).__module__ == "mloda.enterprise.extenders.audit.run_manifest"

    def test_audit_source_modules_import_only_downward_and_only_from_the_defining_module(self) -> None:
        package = Path(audit_package.__file__ or "").parent
        root = "mloda.enterprise.extenders.audit"
        trees = {m: ast.parse((package / f"{m}.py").read_text()) for m in _LAYERS}
        defined = {m: _defined_names(tree) for m, tree in trees.items()}
        problems = []
        for module, tree in trees.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and module != "__init__":
                    problems += [f"{module} uses `import {a.name}`" for a in node.names if a.name.startswith(root)]
                if not isinstance(node, ast.ImportFrom) or not (node.level or root in (node.module or "")):
                    continue
                if node.module == root and module != "__init__":
                    problems.append(f"{module} imports from the package root")
                last = (node.module or "").rsplit(".", 1)[-1]
                for alias in node.names:
                    source = last if last in _LAYERS else alias.name
                    if source not in _LAYERS:
                        continue
                    if _LAYERS.index(source) > _LAYERS.index(module):
                        problems.append(f"{module} imports upward from {source}")
                    if (
                        module != "__init__"
                        and alias.name != source
                        and last in _LAYERS
                        and alias.name not in defined[source]
                    ):
                        problems.append(f"{module} imports {alias.name} from {source}, which does not define it")
        assert problems == []


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
