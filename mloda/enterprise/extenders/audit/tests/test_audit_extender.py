"""Tests for AuditExtender: contract compliance, record shape, allow/deny decisions, the fail_closed
refusal, error handling and NdjsonAuditSink; __call__ tests build a HookContext manually, mirroring
core's own instrumentation."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import logging
import os
import pickle  # nosec
import re
import socket
import stat
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pyarrow as pa
import pytest
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.steward import (
    CompositeExtender,
    Extender,
    ExtenderHook,
    HookContext,
    LifecycleOutcome,
    PlanContext,
    PlanStep,
    RunContext,
    verified_context,
)
from mloda.user import Feature, FeatureName, Options, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader

from mloda.community.extenders.shared import termination
from mloda.community.extenders.shared.classification import CLASSIFICATION_KEY
from mloda.community.extenders.shared.step_run_id import owner_name, step_run_id
from mloda.enterprise.extenders.audit import (
    AuditExtender,
    ClassificationDeniedError,
    ClassificationPolicy,
    Ed25519Signer,
    HmacSha256Signer,
    IdentityRequiredError,
    ManifestVerificationError,
    NdjsonAuditSink,
    NdjsonHeadAnchor,
    RunNotPendingError,
    TeeAuditSink,
    _core,
    manifest_hash,
    rotate_ndjson_segment,
    seal_ndjson_runs,
    verify_ndjson_log,
    verify_ndjson_log_coverage,
    verify_ndjson_segments,
)
from mloda.enterprise.extenders.audit import audit_extender as audit_extender_module
from mloda.enterprise.extenders.audit._core import _flock
from mloda.enterprise.extenders.audit._records import _append_records, _canonical_json, _parse_event_time
from mloda.enterprise.extenders.audit.tests import manifest_helpers
from mloda.enterprise.extenders.audit.tests.manifest_helpers import _Crash, _crash_on_replace, _patch_bindings
from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.runners import (
    CountingExtender,
    expected_value_int,
    prepare_value_int,
    run_csv_feature,
    run_value_int,
)

_BOTH_POSTURES = pytest.mark.parametrize("fail_closed", [False, True])


def _fail_closed_default(
    sink: Any,
    *,
    audit_path: Path | None = None,
    manifest_path: Path | None = None,
    signer: Any | None = None,
) -> AuditExtender:
    return AuditExtender(sink=sink, fail_closed=True, audit_path=audit_path, manifest_path=manifest_path, signer=signer)


def _fail_closed_raise_on_error_false_set_after_construction(
    sink: Any,
    *,
    audit_path: Path | None = None,
    manifest_path: Path | None = None,
    signer: Any | None = None,
) -> AuditExtender:
    extender = _fail_closed_default(sink, audit_path=audit_path, manifest_path=manifest_path, signer=signer)
    extender.raise_on_error = False
    return extender


# How raise_on_error is configured on a fail_closed=True gate; a refusal must propagate in every case.
# The in-constructor variant is covered directly by test_fail_closed_with_raise_on_error_false_is_accepted.
_FAIL_CLOSED_RAISE_ON_ERROR_POSTURES = pytest.mark.parametrize(
    "make_gate_extender",
    [
        pytest.param(_fail_closed_default, id="default"),
        pytest.param(
            _fail_closed_raise_on_error_false_set_after_construction, id="raise_on_error_false_set_after_construction"
        ),
    ],
)

_IDENTITY_REQUIRED_ERROR_TYPE = "mloda.enterprise.extenders.audit.audit_extender.IdentityRequiredError"

# Recognisable values for a present identity, so a refusal message that leaks one is caught.
_TENANT = "tenant-marker-7f3a"
_PROJECT = "project-marker-7f3a"
_PRINCIPAL = "principal-marker-7f3a"

_MISSING_IDENTITY_CASES = [
    (None, _PROJECT, _PRINCIPAL, "missing_tenant_id"),
    (_TENANT, _PROJECT, None, "missing_principal"),
    (None, _PROJECT, None, "missing_tenant_id_and_principal"),
    ("", _PROJECT, _PRINCIPAL, "missing_tenant_id"),
    (_TENANT, _PROJECT, "   ", "missing_principal"),
]

_EXPECTED_RECORD_KEYS = {
    "record_version",
    "event_time",
    "run_id",
    "plan_id",
    "tenant_id",
    "project_id",
    "principal",
    "decision",
    "compliant",
    "deny_reason",
    "hook",
    "feature_group_class",
    "feature_group_version",
    "plugin_version",
    "feature_names",
    "input_features",
    "compute_framework_name",
    "rows_out",
    "duration_seconds",
    "status",
    "error_type",
    "data_access_identity",
    "data_access_format",
    "data_access_identity_is_fallback",
    "policy_version",
    "enforced",
    "phase",
    "input_feature_edges",
    "host",
    "worker_index",
    "start_time",
    "step_run_id",
    "structure_hash",
    "trace_id",
    "span_id",
    "classification",
}

_FINGERPRINT = re.compile(r"[0-9a-f]{12}")

# Neither 12 characters nor hex, so a truncated or hashed value would not equal it.
_POLICY_VERSION = "policy-2026-09-rev-3"


# Classification gate fixtures. Module-level so MULTIPROCESSING can pickle them by path; the MlodaTestingClass prefix
# and the column name keep them from colliding with other plugins during resolution.
_PII_COLUMN = "mloda_testing_class_pii_value"
_CLASS_STEP_UUID = uuid.UUID("6f1c2d3e-4a5b-4c6d-8e7f-0123456789ab")
_OTHER_STEP_UUID = uuid.UUID("0a1b2c3d-4e5f-4a6b-8c7d-0123456789cd")


class _PlainGroup(FeatureGroup):
    """Declares nothing, so it takes the policy's undeclared level."""


class MlodaTestingClassPiiRoot(FeatureGroup):
    """Root source declared pii."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({_PII_COLUMN})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, str | int | float | bool]:
        return {CLASSIFICATION_KEY: "pii"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({_PII_COLUMN: [1, 2, 3]})


class MlodaTestingClassDerived(FeatureGroup):
    """Declares nothing: inherits pii from its input."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(_PII_COLUMN)}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): data[_PII_COLUMN].to_pylist()}


class MlodaTestingClassMasked(FeatureGroup):
    """Masking step declaring internal."""

    masking = True

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(_PII_COLUMN)}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, str | int | float | bool]:
        return {CLASSIFICATION_KEY: "internal"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [None for _ in data[_PII_COLUMN].to_pylist()]}


_DERIVED = MlodaTestingClassDerived.get_class_name()
_MASKED = MlodaTestingClassMasked.get_class_name()
_CLASS_GROUPS = {MlodaTestingClassPiiRoot, MlodaTestingClassDerived, MlodaTestingClassMasked}


def _clear_internal(tenant: str | None, principal: str | None) -> str:
    return "internal"


def _clear_pii(tenant: str | None, principal: str | None) -> str:
    return "pii"


def _classified(
    sink: Any, clearance: Callable[[str | None, str | None], str | None] | str | None = "pii", **kwargs: Any
) -> AuditExtender:
    """A fail_closed gate with a classification policy; a str clearance is a fixed level for every caller."""
    grant = clearance if callable(clearance) else (lambda tenant, principal: clearance)
    options: dict[str, Any] = {
        "fail_closed": True,
        "required_identity": ("tenant_id", "principal"),
        "policy_version": _POLICY_VERSION,
        "classification": ClassificationPolicy(clearance=grant, undeclared="public"),
        **kwargs,
    }
    return AuditExtender(sink=sink, **options)


def _class_step(
    group: type[FeatureGroup],
    names: tuple[str, ...],
    edges: Mapping[str, tuple[str, ...]] | None = None,
    *,
    requested: tuple[str, ...] | None = None,
    step_uuid: uuid.UUID | None = _CLASS_STEP_UUID,
) -> PlanStep:
    return PlanStep(
        step_kind="compute",
        feature_names=names,
        feature_group=group,
        compute_framework=None,
        source_feature_group=None,
        source_compute_framework=None,
        requested_feature_names=names if requested is None else requested,
        input_feature_edges=edges or {},
        step_uuid=step_uuid,
    )


def _pii_chain(requested: tuple[str, ...] = ("derived",)) -> tuple[PlanStep, ...]:
    """A pii root and a derived feature that inherits it; `requested` names the user-requested ones."""
    root = _class_step(MlodaTestingClassPiiRoot, ("root",), requested=() if "root" not in requested else ("root",))
    derived = _class_step(
        _PlainGroup,
        ("derived",),
        {"derived": ("root",)},
        requested=("derived",) if "derived" in requested else (),
        step_uuid=_OTHER_STEP_UUID,
    )
    return (root, derived)


def _run_start(extender: AuditExtender, steps: tuple[PlanStep, ...], **identity: Any) -> None:
    run = RunContext(run_id=_RUN_UUID, plan_id="plan-1", **{"tenant_id": "t", "principal": "svc", **identity})
    extender.on_run_start(run, Mock(plan_id="plan-1", structure_hash=None), steps)


def _run_class_features(
    features: list[Feature | str],
    *extenders: Extender,
    mode: ParallelizationMode = ParallelizationMode.SYNC,
    **kwargs: Any,
) -> list[Any]:
    return mloda.run_all(
        features,
        compute_frameworks=[PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups(_CLASS_GROUPS),
        function_extender=set(extenders),
        parallelization_modes={mode},
        **kwargs,
    )


def _prepare_class_feature(feature: str, *extenders: Extender) -> mloda:
    return mloda.prepare(
        [feature],
        compute_frameworks=[PyArrowTable],
        plugin_collector=PluginCollector.enabled_feature_groups(_CLASS_GROUPS),
        function_extender=set(extenders),
    )


def _ndjson(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# Sealing (AuditExtender.on_run_complete) fixtures: deterministic keys, never used for anything but these tests.
_SEALING_KEY = b"k" * 32
_SEALING_KEY_B = b"o" * 32
_SUCCEEDED = LifecycleOutcome(status="succeeded")


def _hmac_signer(key_id: str = "seal-key-1", key: bytes = _SEALING_KEY) -> HmacSha256Signer:
    return HmacSha256Signer(key, key_id)


def _ed25519_signer(key_id: str = "seal-key-1") -> Ed25519Signer:
    return Ed25519Signer(hashlib.sha256(key_id.encode("utf-8")).digest(), key_id)


def _minimal_audit_record(run_id: str | None, *, compliant: bool = True) -> dict[str, Any]:
    """The smallest record shape seal_ndjson_runs needs: run_id and compliant."""
    return {
        "record_version": 1,
        "run_id": run_id,
        "tenant_id": "tenant-1",
        "decision": "allow" if compliant else "deny",
        "compliant": compliant,
        "deny_reason": None if compliant else "missing_tenant_id",
        "feature_group_class": "my.module.MyFeatureGroup",
        "feature_names": ["value_int"],
        "status": "success",
    }


def _sealing_config(tmp_path: Path) -> tuple[Path, Path]:
    """audit_path, manifest_path for one sealing config, shared to avoid duplicating this construction."""
    return tmp_path / "audit.ndjson", tmp_path / "manifest.ndjson"


def _extender_with_run_1_sealed(
    tmp_path: Path, make: Callable[..., AuditExtender] = AuditExtender, **kwargs: Any
) -> tuple[AuditExtender, Path]:
    """Seal run-1 via on_run_complete; return (extender, audit_path)."""
    audit_path, manifest_path = _sealing_config(tmp_path)
    _append_records(audit_path, [_minimal_audit_record("run-1")])
    signer = _hmac_signer()
    extender = make(
        NdjsonAuditSink(audit_path), audit_path=audit_path, manifest_path=manifest_path, signer=signer, **kwargs
    )
    extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)
    return extender, audit_path


def _second_extender_over_same_sealing_config(
    audit_path: Path,
    manifest_path: Path,
    signer: Any,
    **kwargs: Any,
) -> AuditExtender:
    """A second, fresh AuditExtender over the same sealing config."""
    return AuditExtender(
        NdjsonAuditSink(audit_path), audit_path=audit_path, manifest_path=manifest_path, signer=signer, **kwargs
    )


def _sealing_instance(tmp_path: Path, fail_closed: bool) -> tuple[AuditExtender, Path]:
    """The instance that sealed run-1 itself."""
    return _extender_with_run_1_sealed(tmp_path, fail_closed=fail_closed)


def _second_instance(tmp_path: Path, fail_closed: bool) -> tuple[AuditExtender, Path]:
    """A fresh AuditExtender over the same sealing config, which never sealed run-1 itself."""
    sealing_instance, audit_path = _extender_with_run_1_sealed(tmp_path, fail_closed=fail_closed)
    _, manifest_path = _sealing_config(tmp_path)
    second = _second_extender_over_same_sealing_config(
        audit_path, manifest_path, _find_signer_attr(sealing_instance), fail_closed=fail_closed
    )
    return second, audit_path


def _rotated_after_run_1(tmp_path: Path, auto: bool = False, **kwargs: Any) -> tuple[AuditExtender, Path, Path]:
    """An extender (genesis log_id "log-a") that sealed run-1, then a segment rotation under the same signer:
    manual, or (auto) the extender's own after the seal via segment_max_bytes=1."""
    audit_path, manifest_path = _sealing_config(tmp_path)
    _append_records(audit_path, [_minimal_audit_record("run-1")])
    extra: dict[str, Any] = {"segment_max_bytes": 1} if auto else {}
    extender = AuditExtender(
        NdjsonAuditSink(audit_path),
        audit_path=audit_path,
        manifest_path=manifest_path,
        signer=_hmac_signer(),
        log_id="log-a",
        **extra,
        **kwargs,
    )
    extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)
    if not auto:
        rotate_ndjson_segment(audit_path, manifest_path, signer=_hmac_signer(), log_id="log-a")
    return extender, audit_path, manifest_path


# The extenders that hold a signer: the sealing instance and a fresh second instance over the same config.
_SIGNER_HOLDING_INSTANCES = [
    pytest.param(_sealing_instance, id="sealing_instance"),
    pytest.param(_second_instance, id="second_instance"),
]


def _signer_like(value: Any) -> bool:
    return hasattr(value, "sign") and hasattr(value, "verify") and hasattr(value, "key_id")


def _find_signer_attr(obj: Any) -> Any:
    """The first attribute on obj that looks like a ManifestSigner, else None; name-agnostic on purpose."""
    for value in vars(obj).values():
        if _signer_like(value):
            return value
    return None


def _find_signer_tuple_attr(obj: Any) -> tuple[Any, ...] | None:
    """The first non-empty tuple attribute of signer-like objects on obj, else None; name-agnostic on purpose."""
    for value in vars(obj).values():
        if isinstance(value, tuple) and value and all(_signer_like(item) for item in value):
            return value
    return None


def _holds_none_of(obj: Any, run_ids: Iterable[str]) -> bool:
    """Whether no set, frozenset or dict attribute of obj contains any of run_ids; name-agnostic on purpose."""
    for value in vars(obj).values():
        if isinstance(value, (set, frozenset, dict)) and any(run_id in value for run_id in run_ids):
            return False
    return True


class BufferingNdjsonAuditSink:
    """Buffers written records in memory; flush() appends them to path as NDJSON, so the marker file
    only exists once flush() actually ran with something to write (proving a worker's graceful-exit
    close(), not merely that it wrote a record). Module-level so it survives pickling into a spawned
    worker. Writes via _append_records, the same helper NdjsonAuditSink uses, which already skips an
    empty batch (no os.open at all) rather than creating an empty marker file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._buffer: list[dict[str, Any]] = []

    def write(self, record: Mapping[str, Any]) -> None:
        self._buffer.append(dict(record))

    def flush(self) -> None:
        _append_records(self.path, self._buffer)
        self._buffer = []


class InMemoryAuditSink:
    """Collects every written record in memory, in call order; module-level so it survives pickling."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: Mapping[str, Any]) -> None:
        self.records.append(dict(record))


class _LoggingSink:
    """Appends (name, record) to a log shared between sinks, so the call order across sinks is observable."""

    def __init__(self, name: str, log: list[tuple[str, Mapping[str, Any]]]) -> None:
        self.name = name
        self.log = log

    def write(self, record: Mapping[str, Any]) -> None:
        self.log.append((self.name, record))


class _FailingSink:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def write(self, record: Mapping[str, Any]) -> None:
        raise self.error


class _DiskFullSink:
    def write(self, record: Mapping[str, Any]) -> None:
        raise OSError("disk full")


class _LockedAnchor:
    """A head anchor holding a threading.Lock, so it is unpicklable (TypeError) on every Python version."""

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def write(self, head: str) -> None:
        raise AssertionError("a pickled copy never seals")

    def latest(self) -> str | None:
        raise AssertionError("a pickled copy never seals")


class _ReadSpy:
    """A binary file that records the size of every piece a read hands back."""

    def __init__(self, file: Any, seen: list[int]) -> None:
        self._file = file
        self._seen = seen

    def __enter__(self) -> _ReadSpy:
        return self

    def __exit__(self, *exc: object) -> None:
        self._file.close()

    def __iter__(self) -> Iterator[bytes]:
        for raw in self._file:
            self._seen.append(len(raw))
            yield raw

    def __getattr__(self, name: str) -> Any:
        return getattr(self._file, name)

    def _note(self, data: bytes) -> bytes:
        self._seen.append(len(data))
        return data

    def read(self, *args: Any) -> bytes:
        return self._note(self._file.read(*args))

    def read1(self, *args: Any) -> bytes:
        return self._note(self._file.read1(*args))

    def readline(self, *args: Any) -> bytes:
        return self._note(self._file.readline(*args))

    def readlines(self, *args: Any) -> list[bytes]:
        return [self._note(raw) for raw in self._file.readlines(*args)]

    def readinto(self, buffer: Any) -> int:
        count = int(self._file.readinto(buffer))
        self._seen.append(count)
        return count


class _CountingCall:
    """A wrapped call that counts how often it ran."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        return 42


_BUCKET_KEY = "s3://bucket/key.parquet"

# Stands in for the FeatureSet core passes after data_access.
_FEATURES_PLACEHOLDER = object()

_OUTER_CLASS = "my.module.OuterFeatureGroup"
_INNER_CLASS = "my.module.InnerFeatureGroup"


def _load_context(identity: str | None, data_format: str | None = None, is_fallback: bool | None = None) -> HookContext:
    """An INPUT_DATA_LOAD context without a tenant_id, so a load that got gated would be refused."""
    return make_hook_context(
        hook=ExtenderHook.INPUT_DATA_LOAD,
        data_access_identity=identity,
        data_access_format=data_format,
        data_access_identity_is_fallback=is_fallback,
    )


def _load(
    extender: Callable[..., Any],
    identity: str | None,
    data_format: str | None = None,
    *,
    args: tuple[Any, ...] = (),
    is_fallback: bool | None = None,
) -> Any:
    """Run one wrapped load, passing args to the wrapped call; call it from inside a calculate call."""
    with _load_context(identity, data_format, is_fallback).activate():
        return extender(lambda *_: "loaded", *args)


def _calculate(extender: Callable[..., Any], body: Callable[[], Any], feature_group_class: str = _OUTER_CLASS) -> Any:
    """Run body as one wrapped calculate call under a present tenant_id."""
    with make_hook_context(feature_group_class=feature_group_class, tenant_id="tenant-1").activate():
        return extender(body)


def _record_for_loads(
    loads: Sequence[tuple[str | None, str | None] | tuple[str | None, str | None, bool | None]],
    args: tuple[Any, ...] = (),
) -> dict[str, Any]:
    """The record of one calculate call that ran the given (identity, format[, is_fallback]) loads, each called
    with args."""
    sink = InMemoryAuditSink()
    extender = AuditExtender(sink=sink)

    def body() -> None:
        for identity, data_format, *rest in loads:
            _load(extender, identity, data_format, args=args, is_fallback=rest[0] if rest else None)

    _calculate(extender, body)

    assert len(sink.records) == 1
    return sink.records[0]


def _assert_logged_no_enclosing_calculate(caplog: pytest.LogCaptureFixture) -> None:
    """Exactly one DEBUG message from the audit extender's logger says the load had no open calculate."""
    matching = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG
        and r.name == audit_extender_module.__name__
        and "AuditExtender" in r.getMessage()
        and "calculate" in r.getMessage().lower()
        and ("enclosing" in r.getMessage().lower() or "open" in r.getMessage().lower())
    ]
    assert len(matching) == 1, [(r.name, r.levelno, r.getMessage()) for r in caplog.records]


# (raw data_access passed as args[0], core's recorded identity, secret markers that must not survive anywhere
# in the record). Expected is a hardcoded literal, pinned once against core's BaseInputData.data_access_identity.
_DATA_ACCESS_IDENTITY_CASES = [
    pytest.param(_BUCKET_KEY, _BUCKET_KEY, (), id="clean_s3_unchanged"),
    pytest.param(
        "https://user:pw@host/p/a?sig=SECRET#frag", "https://host/p/a", ("SECRET", "pw"), id="userinfo_query_fragment"
    ),
    pytest.param("postgresql://user:pw@db:5432/mydb", "postgresql://db:5432/mydb", ("pw",), id="userinfo_with_port"),
    pytest.param(
        "jdbc:postgresql://h/db?password=SECRET", "jdbc:postgresql://h", ("SECRET",), id="compound_scheme_query"
    ),
    pytest.param(
        "https://b.com&token=SECRET", "str", ("SECRET", "token"), id="ampersand_tail_in_authority_is_unparseable"
    ),
    pytest.param("https://[::1]:8080/x?k=v", "https://[::1]:8080/x", ("k=v",), id="ipv6_host_with_port"),
    pytest.param("file:///tmp/data.csv", "file:///tmp/data.csv", (), id="file_uri_empty_authority_unchanged"),
    pytest.param("https://host?x=1", "https://host", ("x=1",), id="query_directly_after_host"),
    pytest.param("https://host/p?", "https://host/p", (), id="empty_query"),
    pytest.param("https://u:p@ss@host/db", "https://host/db", ("p@ss",), id="synthetic_at_sign_inside_userinfo"),
    pytest.param(
        "jdbc:hive2://h:10000/default;user=u;password=SECRET",
        "str",
        ("SECRET",),
        id="jdbc_semicolon_params_are_unparseable",
    ),
    pytest.param(
        "jdbc:sqlserver://h:1433;password=SECRET",
        "str",
        ("SECRET",),
        id="jdbc_semicolon_in_authority_is_unparseable",
    ),
    pytest.param("my_scheme://host?tok=SECRET", "str", ("SECRET",), id="underscore_in_scheme_is_unparseable"),
    pytest.param("https://b.com/x&sig=SECRET", "str", ("SECRET", "sig"), id="ampersand_tail_in_path_is_unparseable"),
    pytest.param(
        "mongodb://h1:27017,h2:27017/db?replicaSet=rs",
        "str",
        ("replicaSet",),
        id="multi_host_authority_is_unparseable",
    ),
    pytest.param("s3://münchen-bucket/key.parquet", "s3://münchen-bucket/key.parquet", (), id="non_ascii_host_kept"),
    pytest.param(
        "s3://bucket/t/dt=2026-09-20/part.parquet",
        "s3://bucket/t/dt=2026-09-20/part.parquet",
        (),
        id="hive_style_equals_in_path_kept",
    ),
    pytest.param("postgresql://user:pa/ss@host/db", "str", ("pa/ss",), id="synthetic_slash_in_password_is_unparseable"),
    pytest.param("postgresql://u:p&q@host/db", "postgresql://host/db", ("p&q",), id="synthetic_ampersand_in_password"),
    pytest.param("https://host=SECRET/p/a", "str", ("SECRET",), id="equals_in_authority_is_unparseable"),
    pytest.param("https://host SECRET/p/a", "str", ("SECRET",), id="whitespace_in_authority_is_unparseable"),
    pytest.param(
        "https://host/p?email=a@b.com/x&sig=SECRET",
        "str",
        ("SECRET", "a@b.com", "b.com"),
        id="at_sign_in_query_value",
    ),
    pytest.param(
        "https://host/dl?url=https://user@o.example/f&token=SECRET",
        "str",
        ("SECRET", "o.example"),
        id="uri_in_query_value",
    ),
    pytest.param(
        "postgresql://host/db?user=u&password=p@ss/word",
        "str",
        ("p@ss", "ss/word", "password"),
        id="password_with_at_sign_and_slash_in_query",
    ),
    pytest.param("/data/dir?/file#1.csv", "str", (), id="path_with_query_and_fragment_chars"),
    pytest.param(
        "jdbc:postgresql://h/db",
        "jdbc:postgresql://h",
        (),
        id="jdbc_path_is_dropped_by_core_projection",
    ),
    pytest.param("s3://bucket/my file.parquet", "str", (), id="space_in_path_is_unparseable"),
    pytest.param("{host, port}", "str", (), id="core_dict_form_string_is_a_type_name"),
    pytest.param("str", "str", (), id="core_default_deny_type_name_is_a_fixed_point"),
    pytest.param("host=db user=u password=hunter2", "str", ("hunter2",), id="keyword_dsn_with_password"),
    pytest.param("Server=x;Uid=u;Pwd=hunter2;", "str", ("hunter2",), id="odbc_connection_string"),
    pytest.param(
        {"host": "h", "port": 5432, "api_key": "SECRET"},
        "{api_key, host, port}",
        ("SECRET",),
        id="mapping_identity_is_sorted_keys",
    ),
]


class TestAuditExtenderContract(ExtenderContractTestMixin):
    """AuditExtender satisfies the shared Extender contract."""

    fail_closed = False

    @classmethod
    def extender_class(cls) -> type[Extender]:
        return AuditExtender

    @classmethod
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD}

    @classmethod
    def has_backend_sink(cls) -> bool:
        return False

    @classmethod
    def supports_real_worker_sink(cls) -> bool:
        return True

    def make_extender(self, *, raise_on_error: bool | None = None) -> AuditExtender:
        sink = InMemoryAuditSink()
        if raise_on_error is None:
            return AuditExtender(sink=sink, fail_closed=self.fail_closed)
        return AuditExtender(sink=sink, raise_on_error=raise_on_error, fail_closed=self.fail_closed)

    def own_failure(self) -> AbstractContextManager[Any]:
        return patch.object(InMemoryAuditSink, "write", side_effect=RuntimeError("extender boom"))

    def make_real_worker_extender_and_marker(self, tmp_path: Path) -> tuple[Extender, Path]:
        marker_path = tmp_path / "audit.ndjson"
        return AuditExtender(sink=NdjsonAuditSink(marker_path), fail_closed=self.fail_closed), marker_path

    @classmethod
    def supports_real_worker_buffered_sink(cls) -> bool:
        return True

    def make_real_worker_buffered_extender_and_marker(self, tmp_path: Path) -> tuple[Extender, Path]:
        marker_path = tmp_path / "audit_buffered.ndjson"
        sink = BufferingNdjsonAuditSink(marker_path)
        return AuditExtender(sink=sink, fail_closed=self.fail_closed), marker_path


class TestAuditExtenderFailClosedContract(TestAuditExtenderContract):
    """The fail_closed posture satisfies the same Extender contract, under a present identity."""

    fail_closed = True

    @classmethod
    def supports_warning_only(cls) -> bool:
        return False

    @classmethod
    def supports_real_worker_buffered_sink(cls) -> bool:
        # fail_closed's close() is AuditExtender.close(), the exact same sink.flush() path the parent
        # (non-fail_closed) host already exercises; the extra spawned real-worker run would be redundant.
        return False

    @classmethod
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        return {
            ExtenderHook.FEATURE_GROUP_MATCHED,
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.INPUT_DATA_LOAD,
        }

    @classmethod
    def context_identity(cls) -> dict[str, str]:
        return {"tenant_id": _TENANT}


class TestAuditExtenderClassificationContract(TestAuditExtenderFailClosedContract):
    """A fail_closed gate with a classification policy satisfies the same contract; the policy is dropped on pickle."""

    @classmethod
    def context_identity(cls) -> dict[str, str]:
        return {"tenant_id": _TENANT, "principal": _PRINCIPAL}

    def make_extender(self, *, raise_on_error: bool | None = None) -> AuditExtender:
        extra = {} if raise_on_error is None else {"raise_on_error": raise_on_error}
        return _classified(InMemoryAuditSink(), "pii", **extra)

    def make_real_worker_extender_and_marker(self, tmp_path: Path) -> tuple[Extender, Path]:
        marker_path = tmp_path / "audit.ndjson"
        return _classified(NdjsonAuditSink(marker_path), "pii"), marker_path


class TestAuditExtenderSealingContract(TestAuditExtenderContract):
    """The shared Extender contract also holds with auto-sealing, a genesis log_id and a head anchor configured."""

    @pytest.fixture(autouse=True)
    def _sealing_dir(self, tmp_path: Path) -> None:
        self.sealing_dir = tmp_path

    def _sealing_kwargs(self, audit_path: Path) -> dict[str, Any]:
        return {
            "audit_path": audit_path,
            "manifest_path": self.sealing_dir / "sealed_manifest.ndjson",
            "signer": _hmac_signer(),
            "log_id": "contract-log",
            "head_anchor": NdjsonHeadAnchor(self.sealing_dir / "sealed_anchor.ndjson"),
        }

    def make_extender(self, *, raise_on_error: bool | None = None) -> AuditExtender:
        sink = InMemoryAuditSink()
        kwargs = self._sealing_kwargs(self.sealing_dir / "sealed_audit.ndjson")
        if raise_on_error is None:
            return AuditExtender(sink=sink, fail_closed=self.fail_closed, **kwargs)
        return AuditExtender(sink=sink, raise_on_error=raise_on_error, fail_closed=self.fail_closed, **kwargs)

    def make_real_worker_extender_and_marker(self, tmp_path: Path) -> tuple[Extender, Path]:
        marker_path = tmp_path / "audit.ndjson"
        extender = AuditExtender(
            sink=NdjsonAuditSink(marker_path), fail_closed=self.fail_closed, **self._sealing_kwargs(marker_path)
        )
        return extender, marker_path

    def make_real_worker_buffered_extender_and_marker(self, tmp_path: Path) -> tuple[Extender, Path]:
        marker_path = tmp_path / "audit_buffered.ndjson"
        extender = AuditExtender(
            sink=BufferingNdjsonAuditSink(marker_path),
            fail_closed=self.fail_closed,
            **self._sealing_kwargs(marker_path),
        )
        return extender, marker_path


class TestAuditExtenderRaisePolicySealingContract(TestAuditExtenderSealingContract):
    """The same contract suite with seal_failure_policy="raise" on a healthy sealing extender."""

    def _sealing_kwargs(self, audit_path: Path) -> dict[str, Any]:
        return {**super()._sealing_kwargs(audit_path), "seal_failure_policy": "raise"}


class TestAuditExtenderSealingIndexedContract(TestAuditExtenderSealingContract):
    """The same contract suite with seal_index_path set."""

    def _sealing_kwargs(self, audit_path: Path) -> dict[str, Any]:
        return {**super()._sealing_kwargs(audit_path), "seal_index_path": self.sealing_dir / "sealed_index.sqlite"}


class TestAuditExtenderConstruction:
    """sink and required_identity are validated once, at construction time."""

    @pytest.mark.parametrize("name", ["tenant_id", "project_id", "principal"])
    def test_known_identity_name_is_accepted(self, name: str) -> None:
        AuditExtender(sink=InMemoryAuditSink(), required_identity=(name,))

    def test_unknown_identity_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            AuditExtender(sink=InMemoryAuditSink(), required_identity=("bogus",))

    def test_duplicate_required_identity_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            AuditExtender(sink=InMemoryAuditSink(), required_identity=("tenant_id", "tenant_id"))

    def test_invalid_sink_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            AuditExtender(sink=object())  # type: ignore[arg-type]

    def test_fail_closed_with_empty_required_identity_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            AuditExtender(sink=InMemoryAuditSink(), fail_closed=True, required_identity=())

    def test_classification_with_every_requirement_is_accepted(self) -> None:
        _classified(InMemoryAuditSink())

    def test_classification_without_fail_closed_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _classified(InMemoryAuditSink(), fail_closed=False)

    def test_classification_without_principal_in_required_identity_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _classified(InMemoryAuditSink(), required_identity=("tenant_id",))

    @pytest.mark.parametrize("policy_version", [None, "", "   "], ids=["none", "empty", "blank"])
    def test_classification_without_an_explicit_policy_version_raises_value_error(
        self, policy_version: str | None
    ) -> None:
        with pytest.raises(ValueError):
            _classified(InMemoryAuditSink(), policy_version=policy_version)

    def test_a_classification_policy_with_an_unknown_undeclared_level_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ClassificationPolicy(clearance=lambda tenant, principal: "pii", undeclared="secret")

    def test_a_classification_policy_is_frozen(self) -> None:
        policy = ClassificationPolicy(clearance=lambda tenant, principal: "pii", undeclared="public")

        with pytest.raises(AttributeError):
            policy.undeclared = "pii"  # type: ignore[misc]

    def test_the_classification_policy_is_read_only(self) -> None:
        extender = _classified(InMemoryAuditSink())

        with pytest.raises(AttributeError):
            extender.classification = None  # type: ignore[misc]

    @pytest.mark.parametrize(
        "value", ["pii", object(), lambda tenant, principal: "pii"], ids=["str", "object", "callable"]
    )
    def test_a_classification_that_is_not_a_policy_raises(self, value: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            _classified(InMemoryAuditSink(), classification=value)

    def test_a_classification_denied_error_is_a_runtime_error(self) -> None:
        assert issubclass(ClassificationDeniedError, RuntimeError)

    def test_fail_closed_with_raise_on_error_false_is_accepted(self) -> None:
        extender = AuditExtender(sink=InMemoryAuditSink(), fail_closed=True, raise_on_error=False)

        assert extender.raise_on_error is False
        assert extender.never_fall_back is True

    def test_fail_closed_with_defaults_is_accepted_wraps_the_matched_hook_sorts_outermost_and_never_falls_back(
        self,
    ) -> None:
        extender = AuditExtender(sink=InMemoryAuditSink(), fail_closed=True)

        assert extender.wraps() == {
            ExtenderHook.FEATURE_GROUP_MATCHED,
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.INPUT_DATA_LOAD,
        }
        assert extender.priority == 0
        assert extender.never_fall_back is True
        default_posture = AuditExtender(sink=InMemoryAuditSink())
        assert default_posture.priority == 100
        assert default_posture.never_fall_back is False

    @_BOTH_POSTURES
    def test_fail_closed_is_read_only_after_construction(self, fail_closed: bool) -> None:
        extender = AuditExtender(sink=InMemoryAuditSink(), fail_closed=fail_closed)

        with pytest.raises(AttributeError):
            extender.fail_closed = not fail_closed  # type: ignore[misc]

        assert extender.fail_closed is fail_closed

    def test_default_posture_wraps_the_calculate_and_input_data_load_hooks(self) -> None:
        extender = AuditExtender(sink=InMemoryAuditSink())

        assert extender.wraps() == {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD}

    @pytest.mark.parametrize("policy_version", ["", "   ", 3], ids=["empty", "blank", "non_str"])
    def test_blank_or_non_str_policy_version_raises_value_error(self, policy_version: Any) -> None:
        with pytest.raises(ValueError):
            AuditExtender(sink=InMemoryAuditSink(), policy_version=policy_version)

    def test_signer_without_paths_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            AuditExtender(sink=InMemoryAuditSink(), signer=_hmac_signer())

    def test_audit_path_without_signer_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            AuditExtender(sink=InMemoryAuditSink(), audit_path=tmp_path / "audit.ndjson")

    def test_manifest_path_alone_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            AuditExtender(sink=InMemoryAuditSink(), manifest_path=tmp_path / "manifest.ndjson")

    def test_audit_and_manifest_path_without_signer_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            AuditExtender(
                sink=InMemoryAuditSink(),
                audit_path=tmp_path / "audit.ndjson",
                manifest_path=tmp_path / "manifest.ndjson",
            )

    def test_previous_signers_without_signer_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            AuditExtender(sink=InMemoryAuditSink(), previous_signers=(_hmac_signer("previous-key"),))

    def test_aliased_audit_and_manifest_path_raises_value_error(self, tmp_path: Path) -> None:
        same_path = tmp_path / "same.ndjson"
        with pytest.raises(ValueError):
            AuditExtender(
                sink=InMemoryAuditSink(),
                audit_path=same_path,
                manifest_path=same_path,
                signer=_hmac_signer(),
            )

    def test_duplicate_key_id_between_signer_and_previous_signers_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            AuditExtender(
                sink=InMemoryAuditSink(),
                audit_path=tmp_path / "audit.ndjson",
                manifest_path=tmp_path / "manifest.ndjson",
                signer=_hmac_signer("shared-key-id"),
                previous_signers=(_hmac_signer("shared-key-id", key=_SEALING_KEY_B),),
            )

    def test_non_signer_shaped_signer_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            AuditExtender(
                sink=InMemoryAuditSink(),
                audit_path=tmp_path / "audit.ndjson",
                manifest_path=tmp_path / "manifest.ndjson",
                signer=object(),  # type: ignore[arg-type]
            )

    def test_str_signer_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            AuditExtender(
                sink=InMemoryAuditSink(),
                audit_path=tmp_path / "audit.ndjson",
                manifest_path=tmp_path / "manifest.ndjson",
                signer="not-a-signer",  # type: ignore[arg-type]
            )

    def test_non_signer_shaped_previous_signer_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            AuditExtender(
                sink=InMemoryAuditSink(),
                audit_path=tmp_path / "audit.ndjson",
                manifest_path=tmp_path / "manifest.ndjson",
                signer=_hmac_signer(),
                previous_signers=("not-a-signer",),  # type: ignore[arg-type]
            )

    def test_valid_sealing_config_constructs_and_leaves_existing_behavior_unaffected(self, tmp_path: Path) -> None:
        extender = AuditExtender(
            sink=InMemoryAuditSink(),
            audit_path=tmp_path / "audit.ndjson",
            manifest_path=tmp_path / "manifest.ndjson",
            signer=_hmac_signer(),
            previous_signers=(_hmac_signer("previous-key", key=_SEALING_KEY_B),),
            fail_closed=True,
        )

        assert extender.fail_closed is True
        assert extender.never_fall_back is True
        assert extender.priority == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"log_id": "log-a"},
            {"seal_failure_policy": "raise"},
            {"seal_failure_policy": lambda run_id, exc: None},
            {"segment_max_bytes": 1},
            {"segment_max_age": timedelta(hours=1)},
        ],
        ids=[
            "log_id",
            "seal_failure_policy_raise",
            "seal_failure_policy_callable",
            "segment_max_bytes",
            "segment_max_age",
        ],
    )
    def test_seal_options_without_the_sealing_config_raise_value_error(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            AuditExtender(sink=InMemoryAuditSink(), **kwargs)

    def test_head_anchor_without_the_sealing_config_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            AuditExtender(sink=InMemoryAuditSink(), head_anchor=NdjsonHeadAnchor(tmp_path / "anchor.ndjson"))

    @pytest.mark.parametrize("policy", ["warn", "", None, 3, object()], ids=["warn", "empty", "none", "int", "object"])
    def test_invalid_seal_failure_policy_raises_value_error(self, tmp_path: Path, policy: Any) -> None:
        audit_path, manifest_path = _sealing_config(tmp_path)
        with pytest.raises(ValueError):
            AuditExtender(
                sink=InMemoryAuditSink(),
                audit_path=audit_path,
                manifest_path=manifest_path,
                signer=_hmac_signer(),
                seal_failure_policy=policy,
            )

    @pytest.mark.parametrize("policy", ["log", "raise", lambda run_id, exc: None], ids=["log", "raise", "callable"])
    def test_valid_seal_failure_policy_is_accepted(self, tmp_path: Path, policy: Any) -> None:
        audit_path, manifest_path = _sealing_config(tmp_path)
        extender = AuditExtender(
            sink=InMemoryAuditSink(),
            audit_path=audit_path,
            manifest_path=manifest_path,
            signer=_hmac_signer(),
            seal_failure_policy=policy,
        )
        assert extender.seal_failures == 0
        assert extender.raise_on_run_complete is (policy == "raise")
        unsealed = AuditExtender(sink=InMemoryAuditSink())  # no sealing config
        assert unsealed.seal_failures == 0
        assert unsealed.raise_on_run_complete is False

    @pytest.mark.parametrize("log_id", ["", "   ", 5], ids=["empty", "blank", "int"])
    def test_blank_or_non_str_log_id_raises_value_error(self, tmp_path: Path, log_id: Any) -> None:
        audit_path, manifest_path = _sealing_config(tmp_path)
        with pytest.raises(ValueError):
            AuditExtender(
                sink=InMemoryAuditSink(),
                audit_path=audit_path,
                manifest_path=manifest_path,
                signer=_hmac_signer(),
                log_id=log_id,
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"segment_max_bytes": 0},
            {"segment_max_bytes": -1},
            {"segment_max_bytes": True},
            {"segment_max_bytes": "10"},
            {"segment_max_bytes": 1.5},
            {"segment_max_age": timedelta(0)},
            {"segment_max_age": timedelta(seconds=-1)},
            {"segment_max_age": 3600},
        ],
        ids=["zero", "negative", "bool", "str", "float", "zero_age", "negative_age", "int_age"],
    )
    def test_invalid_segment_rotation_threshold_raises_value_error(
        self, tmp_path: Path, kwargs: dict[str, Any]
    ) -> None:
        audit_path, manifest_path = _sealing_config(tmp_path)
        with pytest.raises(ValueError):
            AuditExtender(
                sink=InMemoryAuditSink(),
                audit_path=audit_path,
                manifest_path=manifest_path,
                signer=_hmac_signer(),
                log_id="log-a",
                **kwargs,
            )

    @pytest.mark.parametrize(
        "kwargs",
        [{"segment_max_bytes": 1}, {"segment_max_age": timedelta(hours=1)}],
        ids=["segment_max_bytes", "segment_max_age"],
    )
    def test_segment_rotation_threshold_without_log_id_raises_value_error(
        self, tmp_path: Path, kwargs: dict[str, Any]
    ) -> None:
        audit_path, manifest_path = _sealing_config(tmp_path)
        with pytest.raises(ValueError):
            AuditExtender(
                sink=InMemoryAuditSink(),
                audit_path=audit_path,
                manifest_path=manifest_path,
                signer=_hmac_signer(),
                **kwargs,
            )

    @pytest.mark.parametrize(
        "anchor",
        [object(), SimpleNamespace(write=lambda head: None), SimpleNamespace(latest=lambda: None), "anchor"],
        ids=["bare_object", "no_latest", "no_write", "str"],
    )
    def test_head_anchor_without_callable_write_and_latest_raises_value_error(
        self, tmp_path: Path, anchor: Any
    ) -> None:
        audit_path, manifest_path = _sealing_config(tmp_path)
        with pytest.raises(ValueError):
            AuditExtender(
                sink=InMemoryAuditSink(),
                audit_path=audit_path,
                manifest_path=manifest_path,
                signer=_hmac_signer(),
                head_anchor=anchor,
            )


class TestAuditExtenderRecord:
    """The audit record's shape and the allow/deny/error decisions that fill it."""

    @_BOTH_POSTURES
    def test_call_without_hook_context_writes_nothing(self, fail_closed: bool) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=fail_closed)

        assert extender(lambda a, b: a + b, 3, 4) == 7
        assert sink.records == []

    def test_call_writes_exactly_one_record(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        with make_hook_context(tenant_id="tenant-1").activate():
            assert extender(lambda a, b: a + b, 3, 4) == 7

        assert len(sink.records) == 1

    @_BOTH_POSTURES
    def test_record_has_exactly_the_expected_keys(self, fail_closed: bool) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=fail_closed)

        with make_hook_context(tenant_id="tenant-1").activate():
            extender(lambda: None)

        assert set(sink.records[0]) == _EXPECTED_RECORD_KEYS

    def test_event_time_is_rfc3339_utc(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        with make_hook_context(tenant_id="tenant-1").activate():
            extender(lambda: None)

        event_time = sink.records[0]["event_time"]
        assert event_time.endswith("Z")
        datetime.fromisoformat(event_time.removesuffix("Z"))

    def test_identity_and_feature_group_fields_are_copied_from_context(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)
        context = make_hook_context(
            feature_group_class="my.module.MyFeatureGroup",
            feature_group_version="3",
            plugin_version="1.2.3",
            feature_names=("value_int", "value_str"),
            input_features=frozenset({"b_feature", "a_feature"}),
            compute_framework_name="PyArrowTable",
            tenant_id="tenant-1",
            project_id="project-1",
            principal="svc-1",
            run_id="run-123",
            plan_id="plan-456",
        )

        with context.activate():
            extender(lambda: None)

        record = sink.records[0]
        assert record["feature_group_class"] == "my.module.MyFeatureGroup"
        assert record["feature_group_version"] == "3"
        assert record["plugin_version"] == "1.2.3"
        assert record["feature_names"] == ["value_int", "value_str"]
        assert record["input_features"] == ["a_feature", "b_feature"]
        assert record["compute_framework_name"] == "PyArrowTable"
        assert record["tenant_id"] == "tenant-1"
        assert record["project_id"] == "project-1"
        assert record["principal"] == "svc-1"
        assert record["run_id"] == "run-123"
        assert record["plan_id"] == "plan-456"
        assert record["hook"] == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE.name

    def test_input_features_none_stays_none(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        with make_hook_context(tenant_id="tenant-1", input_features=None).activate():
            extender(lambda: None)

        assert sink.records[0]["input_features"] is None

    @_BOTH_POSTURES
    def test_all_required_identity_present_allows(self, fail_closed: bool) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(
            sink=sink, required_identity=("tenant_id", "project_id", "principal"), fail_closed=fail_closed
        )

        with make_hook_context(tenant_id="t", project_id="p", principal="s").activate():
            extender(lambda: None)

        assert len(sink.records) == 1
        record = sink.records[0]
        assert record["decision"] == "allow"
        assert record["compliant"] is True
        assert record["deny_reason"] is None
        assert record["policy_version"] == extender.policy_version

    @pytest.mark.parametrize(("tenant_id", "project_id", "principal", "expected_reason"), _MISSING_IDENTITY_CASES)
    def test_missing_required_identity_denies_but_still_runs(
        self,
        tenant_id: str | None,
        project_id: str | None,
        principal: str | None,
        expected_reason: str,
    ) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, required_identity=("tenant_id", "project_id", "principal"))
        call = _CountingCall()

        with make_hook_context(tenant_id=tenant_id, project_id=project_id, principal=principal).activate():
            result = extender(call)

        assert result == 42
        assert call.calls == 1
        record = sink.records[0]
        assert record["decision"] == "deny"
        assert record["compliant"] is False
        assert record["deny_reason"] == expected_reason
        assert record["policy_version"] == extender.policy_version

    @pytest.mark.parametrize(("tenant_id", "project_id", "principal", "expected_reason"), _MISSING_IDENTITY_CASES)
    def test_missing_required_identity_refuses_when_fail_closed(
        self,
        tenant_id: str | None,
        project_id: str | None,
        principal: str | None,
        expected_reason: str,
    ) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(
            sink=sink, required_identity=("tenant_id", "project_id", "principal"), fail_closed=True
        )
        call = _CountingCall()

        with make_hook_context(tenant_id=tenant_id, project_id=project_id, principal=principal).activate():
            with pytest.raises(IdentityRequiredError) as excinfo:
                extender(call)

        assert call.calls == 0
        assert len(sink.records) == 1
        record = sink.records[0]
        assert record["decision"] == "deny"
        assert record["compliant"] is False
        assert record["deny_reason"] == expected_reason
        assert record["status"] == "error"
        assert record["error_type"] == _IDENTITY_REQUIRED_ERROR_TYPE
        assert record["hook"] == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE.name
        assert record["policy_version"] == extender.policy_version
        assert set(record) == _EXPECTED_RECORD_KEYS
        message = str(excinfo.value)
        for name in expected_reason.removeprefix("missing_").split("_and_"):
            assert name in message
        for value in (tenant_id, project_id, principal):
            if value and value.strip():
                assert value not in message

    def test_wrapped_failure_records_error_status_and_propagates(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)
        calls = 0
        marker = "SENSITIVE_ROW_VALUE_xyz123"

        def func() -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError(f"inner boom: {marker}")

        with make_hook_context(tenant_id="tenant-1").activate():
            with pytest.raises(RuntimeError, match="inner boom"):
                extender(func)

        assert calls == 1
        record = sink.records[0]
        assert record["status"] == "error"
        assert record["error_type"] == "builtins.RuntimeError"
        assert record["policy_version"] == extender.policy_version
        assert marker not in json.dumps(record)

    def test_successful_call_records_success_and_no_error_type(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        with make_hook_context(tenant_id="tenant-1").activate():
            extender(lambda: None)

        record = sink.records[0]
        assert record["status"] == "success"
        assert record["error_type"] is None

    def test_keyboard_interrupt_records_error_and_propagates(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        def func() -> None:
            raise KeyboardInterrupt()

        with make_hook_context(tenant_id="tenant-1").activate():
            with pytest.raises(KeyboardInterrupt):
                extender(func)

        assert len(sink.records) == 1
        record = sink.records[0]
        assert record["status"] == "error"
        assert record["error_type"] == "builtins.KeyboardInterrupt"

    def test_wrapped_failure_and_sink_failure_propagates_original_and_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _AlwaysFailingSink:
            def write(self, record: Mapping[str, Any]) -> None:
                raise RuntimeError("sink boom")

        extender = AuditExtender(sink=_AlwaysFailingSink())

        def func() -> None:
            raise RuntimeError("inner boom")

        with make_hook_context(tenant_id="tenant-1").activate():
            with caplog.at_level(logging.WARNING):
                with pytest.raises(RuntimeError, match="inner boom"):
                    extender(func)

        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("AuditExtender" in message for message in warnings)
        assert "sink boom" not in caplog.text


_RUN_UUID = "3f2b8c1e-9d4a-4e6f-8a21-5b7c0d9e1f23"
_STRUCTURE_HASH = "s" * 64


def _plan_ctx(structure_hash: str | None = _STRUCTURE_HASH) -> PlanContext:
    return PlanContext(
        plan_id="plan-1",
        tenant_id=None,
        project_id=None,
        principal=None,
        created_at=datetime.now(),
        structure_hash=structure_hash,
    )


def _calculate_in_run(extender: AuditExtender, run_id: str) -> None:
    with make_hook_context(run_id=run_id, plan_id="plan-1").activate():
        extender(lambda: None)


_CARRIER_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
_CARRIER_SPAN_ID = "00f067aa0ba902b7"
_TRACEPARENT = f"00-{_CARRIER_TRACE_ID}-{_CARRIER_SPAN_ID}-01"


def _write_gate_record(kind: str, fail_closed: bool, identity_present: bool) -> dict[str, Any] | None:
    """The one record a calculate, FEATURE_GROUP_MATCHED or RUN_START call writes (None when it writes nothing);
    a refusal is swallowed here. Identity present means tenant, project and principal."""
    sink = InMemoryAuditSink()
    extender = AuditExtender(sink=sink, fail_closed=fail_closed)
    identity: dict[str, Any] = {"tenant_id": "t", "project_id": "p", "principal": "svc"} if identity_present else {}
    with suppress(IdentityRequiredError):
        if kind == "RUN_START":
            extender.on_run_start(
                RunContext(run_id=_RUN_UUID, plan_id="plan-1", **identity),
                Mock(plan_id="plan-1", structure_hash=None),
                (),
            )
        else:
            hook = (
                ExtenderHook.FEATURE_GROUP_MATCHED
                if kind == "MATCHED"
                else ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE
            )
            unresolved: dict[str, Any] = (
                {"feature_group_class": None, "feature_group_version": None, "compute_framework_name": None}
                if kind == "MATCHED"
                else {}
            )
            with make_hook_context(hook=hook, run_id=_RUN_UUID, plan_id="plan-1", **unresolved, **identity).activate():
                extender(lambda: None)
    assert len(sink.records) <= 1
    return sink.records[0] if sink.records else None


# (kind, fail_closed, identity_present, (decision, enforced, phase)); None expects no record.
_GATE_MATRIX = [
    pytest.param("calculate", False, True, ("allow", False, "run"), id="calculate-open-present"),
    pytest.param("calculate", False, False, ("deny", False, "run"), id="calculate-open-missing"),
    pytest.param("calculate", True, True, ("allow", False, "run"), id="calculate-closed-present"),
    pytest.param("calculate", True, False, ("deny", True, "run"), id="calculate-closed-missing"),
    pytest.param("MATCHED", True, True, None, id="matched-closed-present"),
    pytest.param("MATCHED", True, False, ("deny", True, "plan"), id="matched-closed-missing"),
    pytest.param("RUN_START", False, False, None, id="run_start-open-missing"),
    pytest.param("RUN_START", True, True, None, id="run_start-closed-present"),
    pytest.param("RUN_START", True, False, ("deny", True, "run"), id="run_start-closed-missing"),
]


def _calculate_record(**context: Any) -> dict[str, Any]:
    sink = InMemoryAuditSink()
    with make_hook_context(**{"tenant_id": "t", "principal": "svc", **context}).activate():
        AuditExtender(sink=sink)(lambda: None)
    return sink.records[0]


class TestAuditExtenderStructureHash:
    def test_a_run_record_carries_the_structure_hash_of_its_run_until_on_run_complete(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        extender.on_run_start(RunContext(run_id=_RUN_UUID, plan_id="plan-1"), _plan_ctx(), ())
        _calculate_in_run(extender, _RUN_UUID)
        extender.on_run_complete(RunContext(run_id=_RUN_UUID), _SUCCEEDED)
        _calculate_in_run(extender, _RUN_UUID)

        assert [record["structure_hash"] for record in sink.records] == [_STRUCTURE_HASH, None]

    @pytest.mark.parametrize("setup", ["never_started", "started_without_hash", "plan_time_matched"])
    def test_a_record_without_a_run_structure_hash_has_none(self, setup: str) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)
        if setup == "started_without_hash":
            extender.on_run_start(RunContext(run_id=_RUN_UUID, plan_id="plan-1"), _plan_ctx(None), ())
        if setup == "plan_time_matched":
            extender.on_run_start(RunContext(run_id=_RUN_UUID, plan_id="plan-1"), _plan_ctx(), ())
            with make_hook_context(
                hook=ExtenderHook.FEATURE_GROUP_MATCHED,
                feature_group_class=None,
                feature_group_version=None,
                compute_framework_name=None,
            ).activate():
                extender(lambda: None)
            assert sink.records[0]["run_id"] is None
        else:
            _calculate_in_run(extender, _RUN_UUID)

        assert sink.records[0]["structure_hash"] is None

    def test_concurrent_runs_keep_their_own_hash(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)
        other = str(uuid.uuid4())

        extender.on_run_start(RunContext(run_id=_RUN_UUID, plan_id="plan-1"), _plan_ctx("a" * 64), ())
        extender.on_run_start(RunContext(run_id=other, plan_id="plan-1"), _plan_ctx("b" * 64), ())
        _calculate_in_run(extender, _RUN_UUID)
        _calculate_in_run(extender, other)

        assert [record["structure_hash"] for record in sink.records] == ["a" * 64, "b" * 64]

    def test_a_pickled_copy_still_stamps_the_hash(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)
        extender.on_run_start(RunContext(run_id=_RUN_UUID, plan_id="plan-1"), _plan_ctx(), ())

        copy = pickle.loads(pickle.dumps(extender))  # nosec
        _calculate_in_run(copy, _RUN_UUID)

        assert copy.sink.records[0]["structure_hash"] == _STRUCTURE_HASH

    def test_the_fail_closed_run_start_refusal_record_carries_the_plan_hash(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=True)
        run = RunContext(run_id="run-1", plan_id="plan-1", project_id="p", principal="svc")

        with pytest.raises(IdentityRequiredError):
            extender.on_run_start(run, _plan_ctx(), ())

        assert sink.records[0]["hook"] == "RUN_START"
        assert sink.records[0]["structure_hash"] == _STRUCTURE_HASH


class TestAuditExtenderRecordV2:
    """record_version 2: enforced, phase, edges, host, worker_index, start_time, step_run_id and trace correlation."""

    @pytest.mark.parametrize(("kind", "fail_closed", "identity_present", "expected"), _GATE_MATRIX)
    def test_decision_enforced_and_phase_matrix(
        self, kind: str, fail_closed: bool, identity_present: bool, expected: tuple[str, bool, str] | None
    ) -> None:
        record = _write_gate_record(kind, fail_closed, identity_present)

        if expected is None:
            assert record is None
            return
        assert record is not None
        assert record["record_version"] == 2
        assert (record["decision"], record["enforced"], record["phase"]) == expected
        assert record["compliant"] is identity_present
        if record["enforced"]:
            assert record["start_time"] == record["event_time"]

    @pytest.mark.parametrize("kind", ["MATCHED", "RUN_START"])
    def test_plan_and_run_start_refusals_have_no_step_run_id(self, kind: str) -> None:
        record = _write_gate_record(kind, True, False)

        assert record is not None
        assert record["step_run_id"] is None

    @pytest.mark.parametrize(
        ("tenant_id", "principal", "required", "decision", "compliant"),
        [
            pytest.param("t", "svc", ("tenant_id",), "allow", True, id="principal-present"),
            pytest.param("t", None, ("tenant_id",), "allow", False, id="principal-absent-not-required"),
            pytest.param("t", "   ", ("tenant_id",), "allow", False, id="principal-blank-not-required"),
            pytest.param("t", None, ("tenant_id", "principal"), "deny", False, id="principal-absent-required"),
            pytest.param(None, "svc", ("tenant_id",), "deny", False, id="tenant-missing"),
        ],
    )
    def test_compliant_needs_the_required_identity_and_a_non_blank_principal(
        self, tenant_id: str | None, principal: str | None, required: tuple[str, ...], decision: str, compliant: bool
    ) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, required_identity=required)

        with make_hook_context(tenant_id=tenant_id, principal=principal).activate():
            extender(lambda: None)

        assert sink.records[0]["decision"] == decision
        assert sink.records[0]["compliant"] is compliant

    def test_default_config_without_a_principal_is_non_compliant(self) -> None:
        assert _calculate_record(principal=None)["compliant"] is False

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            pytest.param(
                {"input_feature_edges": {"out_b": ("z", "a"), "out_a": ("m",)}},
                {"input_feature_edges": {"out_b": ["a", "z"], "out_a": ["m"]}},
                id="edges_sorted_per_feature",
            ),
            pytest.param(
                {"input_feature_edges": None, "input_features": frozenset({"b", "a"})},
                {"input_feature_edges": None, "input_features": ["a", "b"]},
                id="edges_none_keeps_input_features",
            ),
        ],
    )
    def test_input_feature_edges(self, kwargs: dict[str, Any], expected: dict[str, Any]) -> None:
        record = _calculate_record(**kwargs)

        for key, value in expected.items():
            assert record[key] == value

    def test_host_is_the_hostname_and_resolved_at_most_once(self) -> None:
        with patch.object(socket, "gethostname", return_value="patched-host") as gethostname:
            first = _calculate_record()
            second = _calculate_record()

        assert first["host"] == second["host"]
        assert gethostname.call_count <= 1
        assert _calculate_record()["host"] in {socket.gethostname(), "patched-host"}

    @pytest.mark.parametrize("worker_index", [None, 0, 3])
    def test_worker_index_is_copied_from_the_context(self, worker_index: int | None) -> None:
        assert _calculate_record(worker_index=worker_index)["worker_index"] == worker_index

    @pytest.mark.parametrize("fails", [False, True], ids=["succeeds", "fails"])
    def test_start_time_is_taken_before_the_wrapped_call(self, fails: bool) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        def call() -> None:
            time.sleep(0.05)
            if fails:
                raise RuntimeError("boom")

        with make_hook_context(tenant_id="t", principal="svc").activate():
            if fails:
                with pytest.raises(RuntimeError):
                    extender(call)
            else:
                extender(call)

        record = sink.records[0]
        start, end = _parse_event_time(record["start_time"]), _parse_event_time(record["event_time"])
        assert record["start_time"].endswith("Z")
        assert (end - start).total_seconds() >= 0.04

    def test_step_run_id_is_the_shared_helper_over_the_owner_name(self) -> None:
        step_uuid = uuid.uuid4()
        context = make_hook_context(
            run_id=_RUN_UUID,
            feature_group_class="my.module.MyFeatureGroup",
            feature_names=("b", "a"),
            compute_framework_name="PyArrowTable",
            step_uuid=step_uuid,
        )
        sink = InMemoryAuditSink()

        with context.activate():
            AuditExtender(sink=sink)(lambda: None)

        expected = step_run_id(_RUN_UUID, owner_name(context, lambda: None), ("b", "a"), "PyArrowTable", step_uuid)
        assert expected is not None
        assert sink.records[0]["step_run_id"] == expected

    def test_step_run_id_is_none_without_a_uuid_run_id(self) -> None:
        assert _calculate_record(run_id="run-1")["step_run_id"] is None

    def test_trace_ids_of_the_active_span_are_hex_strings(self) -> None:
        pytest.importorskip("opentelemetry.sdk.trace")
        from opentelemetry.sdk.trace import TracerProvider

        tracer = TracerProvider().get_tracer("audit-test")
        with tracer.start_as_current_span("caller") as span:
            record = _calculate_record(carrier={"traceparent": _TRACEPARENT})
            ctx = span.get_span_context()

        assert record["trace_id"] == format(ctx.trace_id, "032x")
        assert record["span_id"] == format(ctx.span_id, "016x")
        assert len(record["trace_id"]) == 32
        assert len(record["span_id"]) == 16

    @pytest.mark.parametrize(
        ("carrier", "trace_id"),
        [
            pytest.param({"traceparent": _TRACEPARENT}, _CARRIER_TRACE_ID, id="traceparent"),
            pytest.param(None, None, id="none"),
            pytest.param({}, None, id="empty"),
            pytest.param({"traceparent": "garbage"}, None, id="malformed"),
        ],
    )
    def test_trace_id_comes_from_the_carrier_traceparent_with_no_span_id(
        self, carrier: dict[str, str] | None, trace_id: str | None
    ) -> None:
        record = _calculate_record(carrier=carrier)

        assert record["trace_id"] == trace_id
        assert record["span_id"] is None

    def test_refusal_record_carries_the_trace_ids_of_the_active_span(self) -> None:
        pytest.importorskip("opentelemetry.sdk.trace")
        from opentelemetry.sdk.trace import TracerProvider

        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=True)
        tracer = TracerProvider().get_tracer("audit-test")

        with tracer.start_as_current_span("caller") as span:
            with make_hook_context().activate():
                with pytest.raises(IdentityRequiredError):
                    extender(lambda: None)
            ctx = span.get_span_context()

        assert sink.records[0]["trace_id"] == format(ctx.trace_id, "032x")
        assert sink.records[0]["span_id"] == format(ctx.span_id, "016x")


class TestAuditExtenderFailClosed:
    """fail_closed also fires on FEATURE_GROUP_MATCHED, and the refusal record is written unguarded first."""

    def test_matched_hook_with_missing_identity_refuses_and_records_the_hook(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=True)
        call = _CountingCall()
        context = make_hook_context(
            hook=ExtenderHook.FEATURE_GROUP_MATCHED,
            feature_group_class=None,
            feature_group_version=None,
            compute_framework_name=None,
        )

        with context.activate():
            with pytest.raises(IdentityRequiredError):
                extender(call)

        assert call.calls == 0
        assert len(sink.records) == 1
        record = sink.records[0]
        assert record["feature_group_class"] is None
        assert record["feature_group_version"] is None
        assert record["compute_framework_name"] is None
        assert record["hook"] == ExtenderHook.FEATURE_GROUP_MATCHED.name
        assert record["decision"] == "deny"
        assert record["deny_reason"] == "missing_tenant_id"
        assert record["policy_version"] == extender.policy_version
        assert set(record) == _EXPECTED_RECORD_KEYS

    def test_matched_hook_with_identity_present_returns_the_result_and_writes_nothing(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=True)
        call = _CountingCall()
        context = make_hook_context(
            hook=ExtenderHook.FEATURE_GROUP_MATCHED,
            feature_group_class=None,
            feature_group_version=None,
            compute_framework_name=None,
            tenant_id="tenant-1",
        )

        with context.activate():
            assert extender(call) == 42

        assert call.calls == 1
        assert sink.records == []

    def test_on_run_start_without_the_tenant_refuses_and_writes_one_deny_record(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=True)
        run = RunContext(run_id="run-1", plan_id="plan-1", project_id="p", principal="svc")

        with pytest.raises(IdentityRequiredError):
            extender.on_run_start(run, Mock(plan_id="plan-1", structure_hash=None), ())

        assert len(sink.records) == 1
        record = sink.records[0]
        assert record["hook"] == "RUN_START"
        assert record["decision"] == "deny"
        assert record["status"] == "error"
        assert record["deny_reason"] == "missing_tenant_id"
        assert record["run_id"] == "run-1"
        assert record["plan_id"] == "plan-1"

    def test_on_run_start_with_the_identity_present_allows_and_writes_nothing(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=True)
        run = RunContext(run_id="run-1", plan_id="plan-1", tenant_id="t", project_id="p", principal="svc")

        extender.on_run_start(run, Mock(plan_id="plan-1", structure_hash=None), ())

        assert sink.records == []

    def test_on_run_start_without_fail_closed_is_a_no_op(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=False)

        extender.on_run_start(
            RunContext(run_id="run-1", plan_id="plan-1"), Mock(plan_id="plan-1", structure_hash=None), ()
        )

        assert sink.records == []

    @pytest.mark.parametrize(
        "hook", [ExtenderHook.FEATURE_GROUP_MATCHED, ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE], ids=lambda h: h.name
    )
    @_FAIL_CLOSED_RAISE_ON_ERROR_POSTURES
    def test_sink_failure_on_the_refusal_path_propagates_and_the_call_never_runs(
        self, hook: ExtenderHook, make_gate_extender: Callable[[Any], AuditExtender]
    ) -> None:
        extender = make_gate_extender(_DiskFullSink())
        call = _CountingCall()

        unresolved: dict[str, Any] = (
            {"feature_group_class": None, "feature_group_version": None, "compute_framework_name": None}
            if hook is ExtenderHook.FEATURE_GROUP_MATCHED
            else {}
        )
        with make_hook_context(hook=hook, **unresolved).activate():
            with pytest.raises(OSError, match="disk full") as excinfo:
                CompositeExtender([extender])(call)

        assert call.calls == 0
        assert isinstance(excinfo.value.__context__, IdentityRequiredError)


class TestAuditExtenderClassificationGate:
    """With a classification policy, on_run_start refuses a run whose caller is not cleared for a requested feature."""

    def test_a_requested_feature_above_the_clearance_is_refused_with_one_deny_record(self) -> None:
        sink = InMemoryAuditSink()
        extender = _classified(sink, "internal")

        with pytest.raises(ClassificationDeniedError):
            _run_start(extender, _pii_chain())

        assert len(sink.records) == 1
        record = sink.records[0]
        assert record["hook"] == "RUN_START"
        assert record["phase"] == "run"
        assert record["status"] == "error"
        assert record["run_id"] == _RUN_UUID
        assert record["decision"] == "deny"
        assert record["enforced"] is True
        assert record["compliant"] is True
        assert record["deny_reason"] == "classification_above_clearance"
        assert record["feature_names"] == ["derived"]
        assert record["classification"] == "pii"
        assert record["policy_version"] == _POLICY_VERSION
        assert set(record) == _EXPECTED_RECORD_KEYS

    def test_only_requested_offending_names_are_listed_sorted_with_their_most_restrictive_level(self) -> None:
        sink = InMemoryAuditSink()
        extender = _classified(sink, "internal")
        steps = (
            *_pii_chain(requested=("derived",)),
            _class_step(MlodaTestingClassPiiRoot, ("b", "a"), step_uuid=uuid.uuid4()),
            _class_step(MlodaTestingClassMasked, ("ok",), {"ok": ("root",)}, step_uuid=uuid.uuid4()),
        )

        with pytest.raises(ClassificationDeniedError):
            _run_start(extender, steps)

        assert sink.records[0]["feature_names"] == ["a", "b", "derived"]
        assert sink.records[0]["classification"] == "pii"

    def test_an_unrequested_restricted_intermediate_does_not_deny(self) -> None:
        sink = InMemoryAuditSink()
        steps = (
            _class_step(MlodaTestingClassPiiRoot, ("root",), requested=()),
            _class_step(MlodaTestingClassMasked, ("masked",), {"masked": ("root",)}, step_uuid=_OTHER_STEP_UUID),
        )

        _run_start(_classified(sink, "internal"), steps)

        assert sink.records == []

    def test_an_undeclared_step_takes_the_policy_undeclared_level(self) -> None:
        sink = InMemoryAuditSink()
        extender = _classified(
            sink,
            "internal",
            classification=ClassificationPolicy(clearance=lambda tenant, principal: "internal", undeclared="pii"),
        )

        with pytest.raises(ClassificationDeniedError):
            _run_start(extender, (_class_step(_PlainGroup, ("a",)),))

        assert sink.records[0]["classification"] == "pii"

    def test_a_clearance_at_the_level_allows_and_writes_nothing(self) -> None:
        sink = InMemoryAuditSink()

        _run_start(_classified(sink, "pii"), _pii_chain())

        assert sink.records == []

    def test_the_clearance_is_called_with_the_run_tenant_and_principal(self) -> None:
        clearance = Mock(return_value="pii")

        _run_start(_classified(InMemoryAuditSink(), clearance), _pii_chain(), tenant_id="t-1", principal="alice")

        clearance.assert_called_once_with("t-1", "alice")

    @pytest.mark.parametrize(
        "clearance",
        [
            pytest.param(Mock(side_effect=RuntimeError("clearance boom")), id="raises"),
            pytest.param(lambda tenant, principal: "secret", id="unknown_level"),
        ],
    )
    def test_an_unresolvable_clearance_is_refused_as_unresolved(self, clearance: Callable[..., Any]) -> None:
        sink = InMemoryAuditSink()

        with pytest.raises(ClassificationDeniedError):
            _run_start(_classified(sink, clearance), _pii_chain())

        assert len(sink.records) == 1
        assert sink.records[0]["decision"] == "deny"
        assert sink.records[0]["enforced"] is True
        assert sink.records[0]["deny_reason"] == "classification_unresolved"

    def test_a_none_clearance_is_cleared_for_nothing_and_every_requested_feature_offends(self) -> None:
        sink = InMemoryAuditSink()
        steps = (*_pii_chain(requested=("derived", "root")), _class_step(_PlainGroup, ("z",), step_uuid=uuid.uuid4()))

        with pytest.raises(ClassificationDeniedError):
            _run_start(_classified(sink, None), steps)

        assert len(sink.records) == 1
        assert sink.records[0]["deny_reason"] == "classification_above_clearance"
        assert sink.records[0]["feature_names"] == ["derived", "root", "z"]
        assert sink.records[0]["classification"] == "pii"

    def test_a_plan_whose_levels_cannot_be_resolved_is_refused_as_unresolved(self) -> None:
        sink = InMemoryAuditSink()
        steps = (_class_step(_PlainGroup, ("d",), {"d": ("ghost",)}),)

        with pytest.raises(ClassificationDeniedError):
            _run_start(_classified(sink, "pii"), steps)

        assert sink.records[0]["deny_reason"] == "classification_unresolved"

    def test_missing_identity_is_refused_first_and_its_record_is_unchanged(self) -> None:
        sink = InMemoryAuditSink()
        clearance = Mock(return_value="public")
        extender = _classified(sink, clearance)

        with pytest.raises(IdentityRequiredError):
            extender.on_run_start(
                RunContext(run_id=_RUN_UUID, plan_id="plan-1", principal="svc"),
                Mock(plan_id="plan-1", structure_hash=None),
                _pii_chain(),
            )

        clearance.assert_not_called()
        assert len(sink.records) == 1
        assert sink.records[0]["deny_reason"] == "missing_tenant_id"
        assert sink.records[0]["classification"] is None

    def test_the_gate_is_checked_again_on_every_run_start(self) -> None:
        sink = InMemoryAuditSink()
        extender = _classified(sink, lambda tenant, principal: "pii" if principal == "alice" else "internal")

        _run_start(extender, _pii_chain(), principal="alice")
        with pytest.raises(ClassificationDeniedError):
            _run_start(extender, _pii_chain(), principal="bob")

        assert [record["deny_reason"] for record in sink.records] == ["classification_above_clearance"]

    def test_a_sink_failure_on_the_classification_refusal_propagates_chained_to_the_refusal(self) -> None:
        extender = _classified(_DiskFullSink(), "internal")

        with pytest.raises(OSError, match="disk full") as excinfo:
            _run_start(extender, _pii_chain())

        assert isinstance(excinfo.value.__context__, ClassificationDeniedError)

    def test_an_allowed_runs_call_records_carry_the_level_of_their_step_until_on_run_complete(self) -> None:
        sink = InMemoryAuditSink()
        extender = _classified(sink, "pii")
        _run_start(extender, _pii_chain())

        for step_uuid in (_CLASS_STEP_UUID, _OTHER_STEP_UUID, uuid.uuid4()):
            with make_hook_context(
                run_id=_RUN_UUID, plan_id="plan-1", step_uuid=step_uuid, tenant_id="t", principal="svc"
            ).activate():
                extender(lambda: None)
        extender.on_run_complete(RunContext(run_id=_RUN_UUID), _SUCCEEDED)
        with make_hook_context(
            run_id=_RUN_UUID, plan_id="plan-1", step_uuid=_CLASS_STEP_UUID, tenant_id="t", principal="svc"
        ).activate():
            extender(lambda: None)

        assert [record["classification"] for record in sink.records] == ["pii", "pii", None, None]

    def test_a_step_level_is_the_most_restrictive_of_its_feature_names(self) -> None:
        sink = InMemoryAuditSink()
        extender = _classified(sink, "pii")
        steps = (
            _class_step(MlodaTestingClassPiiRoot, ("root",), requested=()),
            _class_step(MlodaTestingClassMasked, ("masked",), {"masked": ("root",)}, step_uuid=_OTHER_STEP_UUID),
        )
        _run_start(extender, steps)

        with make_hook_context(
            run_id=_RUN_UUID, plan_id="plan-1", step_uuid=_OTHER_STEP_UUID, tenant_id="t", principal="svc"
        ).activate():
            extender(lambda: None)

        assert sink.records[0]["classification"] == "internal"

    def test_a_call_record_without_a_policy_or_step_uuid_has_a_none_classification(self) -> None:
        assert _calculate_record()["classification"] is None
        assert _calculate_record(step_uuid=_CLASS_STEP_UUID)["classification"] is None

        sink = InMemoryAuditSink()
        extender = _classified(sink, "pii")
        _run_start(extender, _pii_chain())
        with make_hook_context(run_id=_RUN_UUID, plan_id="plan-1", tenant_id="t", principal="svc").activate():
            extender(lambda: None)

        assert sink.records[0]["classification"] is None

    def test_the_pickled_extender_keeps_the_run_levels_and_refuses_a_new_run_when_the_policy_is_unpicklable(
        self,
    ) -> None:
        sink = InMemoryAuditSink()
        extender = _classified(sink, "pii")
        _run_start(extender, _pii_chain())

        worker = pickle.loads(pickle.dumps(extender))  # nosec
        with make_hook_context(
            run_id=_RUN_UUID, plan_id="plan-1", step_uuid=_CLASS_STEP_UUID, tenant_id="t", principal="svc"
        ).activate():
            worker(lambda: None)
        with pytest.raises(ClassificationDeniedError):
            _run_start(worker, _pii_chain())

        assert worker._classifications == extender._classifications
        assert worker.sink.records[0]["classification"] == "pii"
        assert worker.sink.records[1]["deny_reason"] == "classification_unresolved"

    @pytest.mark.parametrize(
        "duplicate",
        [copy.copy, copy.deepcopy, lambda extender: pickle.loads(pickle.dumps(extender))],  # nosec
        ids=["copy", "deepcopy", "pickle"],
    )
    def test_a_copy_with_a_picklable_clearance_still_enforces_on_run_start(
        self, duplicate: Callable[[AuditExtender], AuditExtender]
    ) -> None:
        duplicated = duplicate(_classified(InMemoryAuditSink(), _clear_internal))

        with pytest.raises(ClassificationDeniedError):
            _run_start(duplicated, _pii_chain())

        assert cast(InMemoryAuditSink, duplicated.sink).records[0]["deny_reason"] == "classification_above_clearance"

    @pytest.mark.parametrize("duplicate", [copy.copy, copy.deepcopy], ids=["copy", "deepcopy"])
    def test_a_copy_with_a_picklable_clearance_still_allows_a_cleared_run(
        self, duplicate: Callable[[AuditExtender], AuditExtender]
    ) -> None:
        duplicated = duplicate(_classified(InMemoryAuditSink(), _clear_pii))

        _run_start(duplicated, _pii_chain())

        assert cast(InMemoryAuditSink, duplicated.sink).records == []

    @pytest.mark.parametrize(
        "duplicate",
        [copy.copy, copy.deepcopy, lambda extender: pickle.loads(pickle.dumps(extender))],  # nosec
        ids=["copy", "deepcopy", "pickle"],
    )
    def test_a_copy_with_an_unpicklable_clearance_refuses_as_unresolved_even_when_it_would_be_cleared(
        self, duplicate: Callable[[AuditExtender], AuditExtender]
    ) -> None:
        extender = _classified(InMemoryAuditSink(), "pii")
        _run_start(extender, _pii_chain())
        cast(InMemoryAuditSink, extender.sink).records.clear()
        duplicated = duplicate(extender)

        with pytest.raises(ClassificationDeniedError):
            _run_start(duplicated, _pii_chain())

        assert len(cast(InMemoryAuditSink, duplicated.sink).records) == 1
        assert cast(InMemoryAuditSink, duplicated.sink).records[0]["decision"] == "deny"
        assert cast(InMemoryAuditSink, duplicated.sink).records[0]["deny_reason"] == "classification_unresolved"

    def test_a_classification_deny_record_is_sealed_under_its_run_id(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealing_config(tmp_path)
        extender = _classified(
            NdjsonAuditSink(audit_path),
            "internal",
            audit_path=audit_path,
            manifest_path=manifest_path,
            signer=_hmac_signer(),
        )

        with pytest.raises(ClassificationDeniedError):
            _run_start(extender, _pii_chain())
        extender.on_run_complete(RunContext(run_id=_RUN_UUID), _SUCCEEDED)

        assert _ndjson(audit_path)[0]["deny_reason"] == "classification_above_clearance"
        manifests = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        assert [manifest["run_id"] for manifest in manifests] == [_RUN_UUID]


class TestAuditExtenderClose:
    """close() flushes the sink if it defines flush(); a sink without one is a no-op, never an error."""

    def test_close_calls_sink_flush_once(self) -> None:
        sink = Mock(spec=["write", "flush"])
        extender = AuditExtender(sink=sink)

        extender.close()

        sink.flush.assert_called_once()

    @pytest.mark.parametrize("make_sink", [InMemoryAuditSink, lambda: NdjsonAuditSink(Path("unused.ndjson"))])
    def test_close_is_a_noop_when_sink_has_no_flush(self, make_sink: Callable[[], Any]) -> None:
        extender = AuditExtender(sink=make_sink())

        extender.close()  # must not raise

    def test_close_propagates_a_raising_sink_flush(self) -> None:
        sink = Mock(spec=["write", "flush"])
        sink.flush.side_effect = RuntimeError("flush boom")
        extender = AuditExtender(sink=sink)

        with pytest.raises(RuntimeError, match="flush boom"):
            extender.close()


class TestAuditExtenderSealing:
    """on_run_complete(run, outcome) auto-seals a finished run when audit_path/manifest_path/signer are configured;
    otherwise (or with run_id=None) it is a no-op. Direct construction and direct on_run_complete calls, no
    mloda.run_all: the seal machinery itself is exercised end-to-end in the run manifest test modules."""

    def test_run_id_none_is_a_noop_even_when_sealing_is_configured(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        sink = NdjsonAuditSink(audit_path)
        _append_records(audit_path, [_minimal_audit_record("run-1")])
        extender = AuditExtender(sink=sink, audit_path=audit_path, manifest_path=manifest_path, signer=_hmac_signer())

        extender.on_run_complete(RunContext(run_id=None), _SUCCEEDED)

        assert not manifest_path.exists()

    @pytest.mark.parametrize(
        "outcome",
        [
            LifecycleOutcome(status="succeeded"),
            LifecycleOutcome(status="failed", error_type="RuntimeError"),
            LifecycleOutcome(status="cancelled"),
        ],
        ids=["succeeded", "failed", "cancelled"],
    )
    def test_closes_the_sink_before_sealing_so_a_buffered_record_is_included(
        self, tmp_path: Path, outcome: LifecycleOutcome
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        sink = BufferingNdjsonAuditSink(audit_path)
        sink.write(_minimal_audit_record("run-1"))
        assert not audit_path.exists()  # buffered only, proving close() (not something else) puts it on disk
        extender = AuditExtender(sink=sink, audit_path=audit_path, manifest_path=manifest_path, signer=_hmac_signer())

        extender.on_run_complete(RunContext(run_id="run-1"), outcome)

        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
        assert manifest["run_id"] == "run-1"
        assert manifest["record_count"] == 1

    def test_while_terminating_flushes_the_sink_but_skips_the_seal_and_warns_with_the_run_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        sink = BufferingNdjsonAuditSink(audit_path)
        sink.write(_minimal_audit_record("run-1"))
        extender = AuditExtender(sink=sink, audit_path=audit_path, manifest_path=manifest_path, signer=_hmac_signer())
        monkeypatch.setattr(termination, "_terminating", True)

        with caplog.at_level(logging.WARNING):
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert audit_path.exists()  # the sink was flushed
        assert not manifest_path.exists()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("run-1" in r.getMessage() for r in warnings)

    def test_zero_records_and_missing_audit_file_is_a_noop_and_creates_no_manifest(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        assert not audit_path.exists()
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path), audit_path=audit_path, manifest_path=manifest_path, signer=_hmac_signer()
        )

        with caplog.at_level(logging.WARNING):
            extender.on_run_complete(RunContext(run_id="run-never-wrote-anything"), _SUCCEEDED)  # must not raise

        assert not manifest_path.exists()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(str(audit_path) in r.getMessage() for r in warnings)

    def test_zero_records_for_this_run_but_audit_path_has_other_runs_logs_warning_and_seals_nothing_for_it(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        _append_records(audit_path, [_minimal_audit_record("run-other")])
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path), audit_path=audit_path, manifest_path=manifest_path, signer=_hmac_signer()
        )

        with caplog.at_level(logging.WARNING):
            extender.on_run_complete(RunContext(run_id="run-missing"), _SUCCEEDED)  # must not raise

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("run-missing" in r.getMessage() and str(audit_path) in r.getMessage() for r in warnings)
        manifests = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        assert "run-missing" not in {m.get("run_id") for m in manifests}

    @pytest.mark.parametrize("make_instance", _SIGNER_HOLDING_INSTANCES)
    def test_on_run_complete_for_an_already_sealed_run_with_no_new_record_logs_no_error_and_does_not_raise(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        make_instance: Callable[..., tuple[AuditExtender, Path]],
    ) -> None:
        instance, audit_path = make_instance(tmp_path, False)
        _, manifest_path = _sealing_config(tmp_path)
        before = manifest_path.read_bytes()

        with caplog.at_level(logging.INFO):
            instance.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)  # must not raise

        assert not any(r.levelno >= logging.ERROR and r.name == audit_extender_module.__name__ for r in caplog.records)
        infos = [r for r in caplog.records if r.levelno == logging.INFO and r.name == audit_extender_module.__name__]
        assert any("run-1" in r.getMessage() and str(manifest_path) in r.getMessage() for r in infos)
        assert manifest_path.read_bytes() == before

    def test_second_call_after_a_new_record_for_the_same_run_id_does_not_reseal_it(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        _append_records(audit_path, [_minimal_audit_record("run-1")])
        signer = _hmac_signer()
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path), audit_path=audit_path, manifest_path=manifest_path, signer=signer
        )
        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)
        # Stands for a record from a writer that cannot be refused, e.g. an AuditExtender without sealing
        # config.
        _append_records(audit_path, [_minimal_audit_record("run-1")])

        with caplog.at_level(logging.ERROR):
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)  # must not raise

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any(
            "run-1" in r.getMessage() and str(manifest_path) in r.getMessage() and str(audit_path) in r.getMessage()
            for r in errors
        )
        assert any("beyond its seal" in r.getMessage() for r in errors)
        manifests = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        assert len(manifests) == 1
        assert manifests[0]["record_count"] == 1
        with pytest.raises(ManifestVerificationError, match="beyond its seal"):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=signer)

    def test_a_torn_line_in_audit_path_after_the_run_is_sealed_logs_error_and_does_not_raise(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        extender, audit_path = _extender_with_run_1_sealed(tmp_path)
        _, manifest_path = _sealing_config(tmp_path)
        before = manifest_path.read_bytes()
        # A torn line: the record-hash check itself fails to read audit_path.
        with open(audit_path, "ab") as audit_file:
            audit_file.write(b'{"run_id": "run-1", "comp')

        with caplog.at_level(logging.ERROR):
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)  # must not raise

        errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.name == audit_extender_module.__name__]
        assert any("run-1" in r.getMessage() for r in errors)
        assert manifest_path.read_bytes() == before

    def test_audit_path_unreadable_after_the_run_is_sealed_logs_error_and_does_not_raise(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        extender, audit_path = _extender_with_run_1_sealed(tmp_path)
        _, manifest_path = _sealing_config(tmp_path)
        before = manifest_path.read_bytes()
        # audit_path is now a directory: the record-hash check cannot even open it as a file.
        audit_path.unlink()
        audit_path.mkdir()

        with caplog.at_level(logging.ERROR):
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)  # must not raise

        errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.name == audit_extender_module.__name__]
        assert any("run-1" in r.getMessage() for r in errors)
        assert manifest_path.read_bytes() == before

    def test_manifest_verification_error_from_seal_ndjson_runs_propagates(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        _append_records(audit_path, [_minimal_audit_record("run-1")])
        # Seals run-1 under a different key than the extender below is configured with, so the manifest
        # log's current key mismatches the extender's signer.
        other_signer = _hmac_signer("other-key", key=_SEALING_KEY_B)
        seal_ndjson_runs(audit_path, manifest_path, signer=other_signer, run_id="run-1")
        _append_records(audit_path, [_minimal_audit_record("run-2")])
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path),
            audit_path=audit_path,
            manifest_path=manifest_path,
            signer=_hmac_signer(),
            seal_failure_policy="raise",
        )

        with pytest.raises(ManifestVerificationError):
            extender.on_run_complete(RunContext(run_id="run-2"), _SUCCEEDED)

    def test_pickled_copy_drops_the_signer_and_previous_signers(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        extender = AuditExtender(
            sink=InMemoryAuditSink(),
            audit_path=audit_path,
            manifest_path=manifest_path,
            signer=_hmac_signer(),
            previous_signers=(_hmac_signer("previous-key", key=_SEALING_KEY_B),),
        )
        assert _find_signer_attr(extender) is not None
        assert _find_signer_tuple_attr(extender) is not None

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        assert _find_signer_attr(copy) is None
        assert _find_signer_tuple_attr(copy) is None

    def test_pickled_copy_no_longer_auto_seals_while_the_original_still_does(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        _append_records(audit_path, [_minimal_audit_record("run-1")])
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path), audit_path=audit_path, manifest_path=manifest_path, signer=_hmac_signer()
        )
        copy = pickle.loads(pickle.dumps(extender))  # nosec

        copy.on_run_complete(
            RunContext(run_id="run-1"), _SUCCEEDED
        )  # must not raise, and must not seal: the copy has no signer

        assert not manifest_path.exists()

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)  # the original still seals

        assert manifest_path.exists()

    def test_pickled_copy_warns_once_on_run_complete_that_sealing_config_was_dropped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        _append_records(audit_path, [_minimal_audit_record("run-1")])
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path), audit_path=audit_path, manifest_path=manifest_path, signer=_hmac_signer()
        )
        copy = pickle.loads(pickle.dumps(extender))  # nosec

        with caplog.at_level(logging.WARNING):
            copy.on_run_complete(
                RunContext(run_id="run-1"), _SUCCEEDED
            )  # must not raise, and must not seal: the copy has no signer

        assert not manifest_path.exists()
        matching = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and re.search(r"pickl|copi", r.getMessage(), re.IGNORECASE)
        ]
        assert len(matching) == 1, [(r.levelno, r.getMessage()) for r in caplog.records]

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            copy.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)  # same copy again: must not warn a second time

        assert not manifest_path.exists()
        assert not any(re.search(r"pickl|copi", r.getMessage(), re.IGNORECASE) for r in caplog.records)

    def test_flush_failure_inside_on_run_complete_propagates_and_aborts_sealing(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        _append_records(audit_path, [_minimal_audit_record("run-1")])
        sink = Mock(spec=["write", "flush"])
        sink.flush.side_effect = RuntimeError("flush boom")
        extender = AuditExtender(sink=sink, audit_path=audit_path, manifest_path=manifest_path, signer=_hmac_signer())

        with pytest.raises(RuntimeError, match="flush boom"):
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert not manifest_path.exists()

    def test_ed25519_signer_never_enters_the_pickle_stream(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        signer = _ed25519_signer()
        # Contrast: pickling the raw signer alone fails, proving this test would catch a missing __getstate__.
        with pytest.raises(TypeError):
            pickle.dumps(signer)  # nosec
        extender = AuditExtender(
            sink=InMemoryAuditSink(), audit_path=audit_path, manifest_path=manifest_path, signer=signer
        )

        pickle.dumps(extender)  # nosec  # must not raise

    def test_an_unpicklable_head_anchor_and_callable_policy_never_enter_the_pickle_stream(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealing_config(tmp_path)
        _append_records(audit_path, [_minimal_audit_record("run-1")])
        anchor = _LockedAnchor()
        extender = AuditExtender(
            sink=InMemoryAuditSink(),
            audit_path=audit_path,
            manifest_path=manifest_path,
            signer=_hmac_signer(),
            head_anchor=anchor,
            seal_failure_policy=lambda run_id, exc: None,
        )
        with pytest.raises((TypeError, pickle.PicklingError)):
            pickle.dumps(anchor)  # nosec  # contrast: the anchor alone is unpicklable

        copy = pickle.loads(pickle.dumps(extender))  # nosec  # must not raise

        assert copy._head_anchor is None
        assert copy._seal_failure_policy == "log"
        copy.on_run_complete(
            RunContext(run_id="run-1"), _SUCCEEDED
        )  # the copy never seals and never touches the anchor
        assert not manifest_path.exists()
        assert copy.seal_failures == 0

    def test_a_long_lived_instance_holds_no_run_ids_once_their_runs_completed(self, tmp_path: Path) -> None:
        audit_path, manifest_path = _sealing_config(tmp_path)
        extender = AuditExtender(
            NdjsonAuditSink(audit_path), audit_path=audit_path, manifest_path=manifest_path, signer=_hmac_signer()
        )
        run_ids = ["run-alpha", "run-bravo", "run-charlie"]
        for run_id in run_ids:
            with make_hook_context(run_id=run_id, tenant_id=_TENANT).activate():
                extender(_CountingCall())
            extender.on_run_complete(RunContext(run_id=run_id), _SUCCEEDED)

        extender.on_run_complete(RunContext(run_id="run-alpha"), _SUCCEEDED)

        assert _holds_none_of(extender, run_ids)
        pickled = pickle.dumps(extender)  # nosec
        assert not any(run_id.encode("utf-8") in pickled for run_id in run_ids)

    def test_on_run_complete_for_a_run_sealed_in_an_archived_segment_is_not_a_seal_failure(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _, audit_path, manifest_path = _rotated_after_run_1(tmp_path)
        second = _second_extender_over_same_sealing_config(audit_path, manifest_path, _hmac_signer(), log_id="log-a")

        with caplog.at_level(logging.INFO):
            second.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert second.seal_failures == 0
        assert any(
            r.levelno == logging.INFO
            and "already sealed" in r.getMessage()
            and "was not sealed again" in r.getMessage()
            for r in caplog.records
        )

    def test_an_auto_seal_after_a_rotation_succeeds_while_the_anchor_still_holds_the_old_head(
        self, tmp_path: Path
    ) -> None:
        anchor = NdjsonHeadAnchor(tmp_path / "anchor.ndjson")
        extender, audit_path, manifest_path = _rotated_after_run_1(tmp_path, head_anchor=anchor)
        old_head = anchor.latest()
        assert old_head is not None

        with make_hook_context(run_id="run-2", tenant_id=_TENANT).activate():
            extender(_CountingCall())
        extender.on_run_complete(RunContext(run_id="run-2"), _SUCCEEDED)

        assert extender.seal_failures == 0
        assert json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[-1])["run_id"] == "run-2"
        assert anchor.latest() != old_head
        verify_ndjson_log(audit_path, manifest_path, signer=_hmac_signer(), log_id="log-a", anchored_heads=[old_head])

    def test_a_calculation_never_waits_on_an_exclusive_holder_of_the_manifest_lock(self, tmp_path: Path) -> None:
        sealing_instance, audit_path = _extender_with_run_1_sealed(tmp_path)
        _, manifest_path = _sealing_config(tmp_path)
        second = _second_extender_over_same_sealing_config(audit_path, manifest_path, _hmac_signer())
        call = _CountingCall()
        finished = threading.Event()

        def run_under_run_2() -> None:
            with make_hook_context(run_id="run-2", tenant_id=_TENANT).activate():
                second(call)
            finished.set()

        with _flock(manifest_path, exclusive=True):
            thread = threading.Thread(target=run_under_run_2)
            thread.start()
            thread.join(timeout=3)
            finished_while_locked = finished.is_set()
        thread.join()  # the lock is released now; never leave the thread running past the test

        assert finished_while_locked
        assert call.calls == 1

    def test_call_under_a_different_run_id_after_sealing_still_runs_and_writes(self, tmp_path: Path) -> None:
        extender, audit_path = _extender_with_run_1_sealed(tmp_path)
        before = audit_path.read_bytes()
        call = _CountingCall()

        with make_hook_context(run_id="run-2", tenant_id=_TENANT).activate():
            result = extender(call)

        assert result == 42
        assert call.calls == 1
        after = audit_path.read_bytes()
        assert after != before
        assert after.startswith(before)

    @pytest.mark.parametrize(
        "seed",
        [
            pytest.param(lambda audit_path: None, id="missing_audit_file"),
            pytest.param(
                lambda audit_path: _append_records(audit_path, [_minimal_audit_record("run-other")]),
                id="run_not_pending",
            ),
        ],
    )
    def test_on_run_complete_that_sealed_nothing_does_not_remember_the_run_id(
        self, tmp_path: Path, seed: Callable[[Path], None]
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        seed(audit_path)
        signer = _hmac_signer()
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path), audit_path=audit_path, manifest_path=manifest_path, signer=signer
        )

        extender.on_run_complete(
            RunContext(run_id="run-1"), _SUCCEEDED
        )  # nothing was sealed: missing file or no records for run-1

        call = _CountingCall()
        with make_hook_context(run_id="run-1", tenant_id=_TENANT).activate():
            result = extender(call)

        assert result == 42
        assert call.calls == 1

    def test_call_under_an_already_sealed_run_id_runs_and_its_stray_record_counts_as_one_seal_failure(
        self, tmp_path: Path
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        _append_records(audit_path, [_minimal_audit_record("run-1")])
        signer = _hmac_signer()
        seal_ndjson_runs(audit_path, manifest_path, signer=signer, run_id="run-1")
        captured: list[BaseException] = []
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path),
            audit_path=audit_path,
            manifest_path=manifest_path,
            signer=signer,
            seal_failure_policy=lambda run_id, exc: captured.append(exc),
        )

        before = audit_path.read_bytes()
        call = _CountingCall()
        with make_hook_context(run_id="run-1", tenant_id=_TENANT).activate():
            result = extender(call)

        assert result == 42
        assert call.calls == 1
        after = audit_path.read_bytes()
        assert after != before
        assert after.startswith(before)

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert extender.seal_failures == 1
        assert len(captured) == 1
        assert isinstance(captured[0], ManifestVerificationError)

    def test_extender_without_sealing_config_seal_is_a_noop_and_the_call_still_runs(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)  # a no-op: no sealing config

        call = _CountingCall()
        with make_hook_context(run_id="run-1", tenant_id=_TENANT).activate():
            result = extender(call)

        assert result == 42
        assert call.calls == 1
        assert len(sink.records) == 1

    # --- head anchoring, genesis, seal_failures and seal_failure_policy ---

    @staticmethod
    def _anchored_extender(tmp_path: Path, **kwargs: Any) -> tuple[AuditExtender, Path, Path, NdjsonHeadAnchor]:
        audit_path, manifest_path = _sealing_config(tmp_path)
        anchor = NdjsonHeadAnchor(tmp_path / "anchor.ndjson")
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path),
            audit_path=audit_path,
            manifest_path=manifest_path,
            signer=_hmac_signer(),
            head_anchor=anchor,
            **kwargs,
        )
        return extender, audit_path, manifest_path, anchor

    @staticmethod
    def _failing_seal_extender(tmp_path: Path, **kwargs: Any) -> AuditExtender:
        audit_path, manifest_path = _sealing_config(tmp_path)
        _append_records(audit_path, [_minimal_audit_record("run-1")])
        return AuditExtender(
            sink=NdjsonAuditSink(audit_path),
            audit_path=audit_path,
            manifest_path=manifest_path,
            signer=_hmac_signer(),
            **kwargs,
        )

    def test_auto_seal_passes_sealed_late_false_and_the_manifest_says_so(self, tmp_path: Path) -> None:
        extender = self._failing_seal_extender(tmp_path)
        _, manifest_path = _sealing_config(tmp_path)

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
        assert manifest["sealed_late"] is False

    def test_auto_seal_forwards_log_id_head_anchor_and_the_latest_anchored_head(self, tmp_path: Path) -> None:
        anchor = Mock(spec=["write", "latest"])
        anchor.latest.return_value = "a" * 64
        extender = self._failing_seal_extender(tmp_path, log_id="log-a", head_anchor=anchor)

        with patch.object(audit_extender_module, "seal_ndjson_runs", return_value=[]) as seal:
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        kwargs = seal.call_args.kwargs
        assert kwargs["sealed_late"] is False
        assert kwargs["log_id"] == "log-a"
        assert kwargs["head_anchor"] is anchor
        assert list(kwargs["anchored_heads"]) == ["a" * 64]

    def test_auto_seal_passes_no_anchored_heads_when_latest_returns_none(self, tmp_path: Path) -> None:
        anchor = Mock(spec=["write", "latest"])
        anchor.latest.return_value = None
        extender = self._failing_seal_extender(tmp_path, head_anchor=anchor)

        with patch.object(audit_extender_module, "seal_ndjson_runs", return_value=[]) as seal:
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert list(seal.call_args.kwargs.get("anchored_heads", ())) == []

    def test_auto_seal_writes_a_genesis_with_the_log_id_and_anchors_the_new_head(self, tmp_path: Path) -> None:
        extender, audit_path, manifest_path, anchor = self._anchored_extender(tmp_path, log_id="log-a")
        _append_records(audit_path, [_minimal_audit_record("run-1")])

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        lines = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        assert [line.get("kind") for line in lines] == ["genesis", None]
        assert lines[0]["log_id"] == "log-a"
        assert anchor.latest() == manifest_hash(lines[-1])
        assert extender.seal_failures == 0

    def test_a_second_auto_seal_passes_the_first_anchored_head_and_succeeds(self, tmp_path: Path) -> None:
        extender, audit_path, manifest_path, anchor = self._anchored_extender(tmp_path)
        _append_records(audit_path, [_minimal_audit_record("run-1")])
        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)
        first_head = anchor.latest()
        _append_records(audit_path, [_minimal_audit_record("run-2")])

        extender.on_run_complete(RunContext(run_id="run-2"), _SUCCEEDED)

        assert anchor.latest() != first_head
        assert extender.seal_failures == 0

    # --- segment rotation after an auto-seal ---

    @staticmethod
    def _rotating_extender(tmp_path: Path, **kwargs: Any) -> tuple[AuditExtender, Path, Path, NdjsonHeadAnchor]:
        _, manifest_path = _sealing_config(tmp_path)
        anchor = NdjsonHeadAnchor(tmp_path / "anchor.ndjson")
        extender = TestAuditExtenderSealing._failing_seal_extender(
            tmp_path, log_id="log-a", head_anchor=anchor, **kwargs
        )
        return extender, tmp_path / "audit.ndjson", manifest_path, anchor

    @staticmethod
    def _archive(path: Path) -> Path:
        return path.with_name(path.name + ".000001")

    def test_no_rotation_when_neither_threshold_is_passed(self, tmp_path: Path) -> None:
        extender, audit_path, manifest_path, _ = self._rotating_extender(
            tmp_path, segment_max_bytes=10**9, segment_max_age=timedelta(days=365)
        )

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert extender.seal_failures == 0
        assert json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[-1])["run_id"] == "run-1"
        assert not self._archive(audit_path).exists()
        assert not self._archive(manifest_path).exists()

    def test_segment_max_bytes_passed_rotates_after_the_seal_and_anchors_the_new_genesis(self, tmp_path: Path) -> None:
        extender, audit_path, manifest_path, anchor = self._rotating_extender(tmp_path, segment_max_bytes=1)

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert self._archive(audit_path).exists()
        assert self._archive(manifest_path).exists()
        live = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        assert [line.get("kind") for line in live] == ["genesis"]
        assert anchor.latest() == manifest_hash(live[0])
        heads = [json.loads(line)["head"] for line in (tmp_path / "anchor.ndjson").read_text().splitlines()]
        verify_ndjson_segments(audit_path, manifest_path, signer=_hmac_signer(), log_id="log-a", anchored_heads=heads)
        assert extender.seal_failures == 0

    def test_segment_max_age_passed_rotates_after_the_seal(self, tmp_path: Path) -> None:
        extender, audit_path, manifest_path, _ = self._rotating_extender(tmp_path, segment_max_age=timedelta(hours=1))
        genesis = manifest_helpers._genesis_entry(
            _hmac_signer(), log_id="log-a", created_at="2020-01-01T00:00:00.000000Z"
        )
        manifest_path.write_bytes(_canonical_json(genesis) + b"\n")

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert extender.seal_failures == 0
        assert self._archive(audit_path).exists()
        assert self._archive(manifest_path).exists()

    def test_segment_max_age_not_passed_on_a_fresh_log_does_not_rotate(self, tmp_path: Path) -> None:
        extender, audit_path, manifest_path, _ = self._rotating_extender(tmp_path, segment_max_age=timedelta(hours=1))

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert extender.seal_failures == 0
        assert not self._archive(audit_path).exists()
        assert not self._archive(manifest_path).exists()

    def test_no_rotation_when_the_run_was_already_sealed(self, tmp_path: Path) -> None:
        extender, audit_path, manifest_path, _ = self._rotating_extender(tmp_path)
        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)
        rotating = _second_extender_over_same_sealing_config(
            audit_path, manifest_path, _hmac_signer(), log_id="log-a", segment_max_bytes=1
        )

        rotating.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert rotating.seal_failures == 0
        assert not self._archive(audit_path).exists()
        assert not self._archive(manifest_path).exists()

    def test_no_rotation_when_the_run_is_not_pending(self, tmp_path: Path) -> None:
        extender, audit_path, manifest_path, _ = self._rotating_extender(tmp_path, segment_max_bytes=1)

        with patch.object(audit_extender_module, "seal_ndjson_runs", side_effect=RunNotPendingError("none")):
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert extender.seal_failures == 0
        assert not self._archive(audit_path).exists()
        assert not self._archive(manifest_path).exists()

    def test_carried_pending_lines_alone_over_segment_max_bytes_never_rotate(self, tmp_path: Path) -> None:
        extender, audit_path, manifest_path, _ = self._rotating_extender(tmp_path, segment_max_bytes=1000)
        _append_records(audit_path, [_minimal_audit_record("run-x") for _ in range(20)])
        assert audit_path.stat().st_size > 1000

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)
        _append_records(audit_path, [_minimal_audit_record("run-2")])
        extender.on_run_complete(RunContext(run_id="run-2"), _SUCCEEDED)

        assert extender.seal_failures == 0
        assert not self._archive(audit_path).exists()
        assert not self._archive(manifest_path).exists()

    @staticmethod
    def _interrupted_rotation(tmp_path: Path, call_number: int = 1, **kwargs: Any) -> tuple[AuditExtender, Path, Path]:
        """run-1 sealed under "log-a", a pending run-2 record, a manual rotation crashed at its `call_number`th os.replace;
        returns a fresh extender over that state."""
        _, audit_path, manifest_path, _ = TestAuditExtenderSealing._rotating_extender(tmp_path)
        _append_records(audit_path, [_minimal_audit_record("run-2")])
        sealer = _second_extender_over_same_sealing_config(audit_path, manifest_path, _hmac_signer(), log_id="log-a")
        sealer.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)
        with patch("os.replace", _crash_on_replace(call_number)):
            with pytest.raises(_Crash):
                rotate_ndjson_segment(audit_path, manifest_path, signer=_hmac_signer(), log_id="log-a")
        extender = _second_extender_over_same_sealing_config(
            audit_path, manifest_path, _hmac_signer(), log_id="log-a", **kwargs
        )
        return extender, audit_path, manifest_path

    @pytest.mark.parametrize(
        ("auto", "call_number"), [(True, 1), (True, 2), (False, 1)], ids=["auto-1", "auto-2", "no_auto"]
    )
    def test_an_interrupted_rotation_is_finished_and_the_seal_retried_only_under_auto_rotation(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, auto: bool, call_number: int
    ) -> None:
        extender, audit_path, manifest_path = self._interrupted_rotation(
            tmp_path, call_number, **({"segment_max_bytes": 10**9} if auto else {})
        )

        with caplog.at_level(logging.WARNING):
            extender.on_run_complete(RunContext(run_id="run-2"), _SUCCEEDED)

        last = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[-1])
        warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING and "interrupted rotation" in r.getMessage()
        ]
        if auto:
            assert extender.seal_failures == 0
            assert self._archive(audit_path).exists()
            assert last["run_id"] == "run-2"
            assert len(warnings) == 1
            assert "run-2" in warnings[0].getMessage()
            verify_ndjson_segments(audit_path, manifest_path, signer=_hmac_signer(), log_id="log-a")
        else:
            assert extender.seal_failures == 1
            assert last.get("run_id") != "run-2"

    @pytest.mark.parametrize("policy", ["callable", "log", "raise"])
    def test_a_failure_finishing_an_interrupted_rotation_is_one_seal_failure_under_the_policy(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, policy: str
    ) -> None:
        mock = Mock()
        extra: dict[str, Any] = {"seal_failure_policy": mock} if policy == "callable" else {}
        if policy == "raise":
            extra = {"seal_failure_policy": "raise"}
        extender, _, _ = self._interrupted_rotation(tmp_path, segment_max_bytes=10**9, **extra)
        error = RuntimeError("finish boom")

        with patch.object(audit_extender_module, "_rotate", side_effect=error):
            with caplog.at_level(logging.ERROR):
                if policy == "raise":
                    with pytest.raises(RuntimeError, match="finish boom"):
                        extender.on_run_complete(RunContext(run_id="run-2"), _SUCCEEDED)
                else:
                    extender.on_run_complete(RunContext(run_id="run-2"), _SUCCEEDED)

        assert extender.seal_failures == 1
        if policy == "raise":
            return
        if policy == "callable":
            mock.assert_called_once_with("run-2", error)
        else:
            errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
            assert any("interrupted rotation" in m and "RuntimeError" in m for m in errors)
            assert not any("finish boom" in m for m in errors)

    @pytest.mark.parametrize("policy", ["log", "raise"])
    def test_not_a_failure_run_not_pending(self, tmp_path: Path, policy: Any) -> None:
        extender = self._failing_seal_extender(tmp_path, seal_failure_policy=policy)
        with patch.object(audit_extender_module, "seal_ndjson_runs", side_effect=RunNotPendingError("none")):
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)  # must not raise even under "raise"

        assert extender.seal_failures == 0

    @pytest.mark.parametrize("policy", ["log", "raise"])
    def test_not_a_failure_missing_audit_file(self, tmp_path: Path, policy: Any) -> None:
        audit_path, manifest_path = _sealing_config(tmp_path)
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path),
            audit_path=audit_path,
            manifest_path=manifest_path,
            signer=_hmac_signer(),
            seal_failure_policy=policy,
        )

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert extender.seal_failures == 0

    @pytest.mark.parametrize("policy", ["log", "raise"])
    def test_not_a_failure_already_sealed_run_whose_records_match(self, tmp_path: Path, policy: Any) -> None:
        extender, _ = _extender_with_run_1_sealed(tmp_path, seal_failure_policy=policy)

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)  # must not raise

        assert extender.seal_failures == 0

    def test_mismatch_after_seal_counts_as_a_failure_and_logs_at_error_by_default(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        extender, audit_path = _extender_with_run_1_sealed(tmp_path)
        _append_records(audit_path, [_minimal_audit_record("run-1")])

        with caplog.at_level(logging.ERROR):
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert extender.seal_failures == 1
        assert any(r.levelno == logging.ERROR and "beyond its seal" in r.getMessage() for r in caplog.records)

    def test_any_error_checking_an_already_sealed_run_counts_as_a_seal_failure_under_the_policy(
        self, tmp_path: Path
    ) -> None:
        calls: list[tuple[str, BaseException]] = []
        extender, _ = _extender_with_run_1_sealed(
            tmp_path, seal_failure_policy=lambda run_id, exc: calls.append((run_id, exc))
        )

        with patch.object(audit_extender_module, "_check_run_against_seal", side_effect=RuntimeError("check boom")):
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert extender.seal_failures == 1
        assert [(run_id, type(exc)) for run_id, exc in calls] == [("run-1", RuntimeError)]

    def test_mismatch_after_seal_under_raise_raises_manifest_verification_error(self, tmp_path: Path) -> None:
        extender, audit_path = _extender_with_run_1_sealed(tmp_path, seal_failure_policy="raise")
        _append_records(audit_path, [_minimal_audit_record("run-1")])

        with pytest.raises(ManifestVerificationError):
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert extender.seal_failures == 1

    def test_mismatch_after_seal_calls_a_callable_policy_once(self, tmp_path: Path) -> None:
        calls: list[tuple[str, BaseException]] = []
        extender, audit_path = _extender_with_run_1_sealed(
            tmp_path, seal_failure_policy=lambda run_id, exc: calls.append((run_id, exc))
        )
        _append_records(audit_path, [_minimal_audit_record("run-1")])

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert len(calls) == 1
        assert calls[0][0] == "run-1"
        assert isinstance(calls[0][1], ManifestVerificationError)
        assert extender.seal_failures == 1

    @pytest.mark.parametrize(
        "error",
        [
            ManifestVerificationError("rolled back"),
            RuntimeError("anchor boom"),
            OSError("disk"),
        ],
        ids=["manifest_verification", "runtime", "os"],
    )
    def test_any_exception_from_seal_ndjson_runs_counts_once_per_failed_seal(
        self, tmp_path: Path, error: Exception
    ) -> None:
        extender = self._failing_seal_extender(tmp_path)

        with patch.object(audit_extender_module, "seal_ndjson_runs", side_effect=error):
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)
            assert extender.seal_failures == 1
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert extender.seal_failures == 2

    @staticmethod
    def _failing_stage_extender(tmp_path: Path, stage: str, **kwargs: Any) -> tuple[AuditExtender, str]:
        """The extender whose run-1 auto-seal or post-seal rotation is patched to fail, and the name to patch."""
        if stage == "seal":
            return TestAuditExtenderSealing._failing_seal_extender(tmp_path, **kwargs), "seal_ndjson_runs"
        extender = TestAuditExtenderSealing._rotating_extender(tmp_path, segment_max_bytes=1, **kwargs)[0]
        return extender, "_rotate"

    @pytest.mark.parametrize(
        ("stage", "error"),
        [
            ("seal", RuntimeError("secret-record-data")),
            ("rotate", RuntimeError("secret-record-data")),
            ("rotate", ManifestVerificationError("rolled back")),
        ],
        ids=["seal", "rotate", "rotate_manifest_verification"],
    )
    def test_log_policy_logs_at_error_naming_the_exception_type_and_not_the_message(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, stage: str, error: Exception
    ) -> None:
        extender, target = self._failing_stage_extender(tmp_path, stage)

        with patch.object(audit_extender_module, target, side_effect=error):
            with caplog.at_level(logging.ERROR):
                extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)  # must not raise

        errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.name == audit_extender_module.__name__]
        assert any(type(error).__name__ in r.getMessage() and "run-1" in r.getMessage() for r in errors)
        if isinstance(error, RuntimeError):
            assert not any("secret-record-data" in r.getMessage() for r in errors)
        if stage == "rotate":
            _, manifest_path = _sealing_config(tmp_path)
            assert any("rotat" in r.getMessage() for r in errors)
            assert extender.seal_failures == 1
            assert json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[-1])["run_id"] == "run-1"

    @pytest.mark.parametrize("stage", ["seal", "rotate"])
    def test_callable_policy_is_called_once_with_run_id_and_the_exception_and_nothing_is_raised(
        self, tmp_path: Path, stage: str
    ) -> None:
        callback = Mock()
        extender, target = self._failing_stage_extender(tmp_path, stage, seal_failure_policy=callback)
        boom = RuntimeError("boom")

        with patch.object(audit_extender_module, target, side_effect=boom):
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        callback.assert_called_once_with("run-1", boom)
        assert extender.seal_failures == 1

    @pytest.mark.parametrize("stage", ["seal", "rotate"])
    def test_raise_policy_reraises_the_exception_and_still_counts(self, tmp_path: Path, stage: str) -> None:
        extender, target = self._failing_stage_extender(tmp_path, stage, seal_failure_policy="raise")

        with patch.object(audit_extender_module, target, side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert extender.seal_failures == 1

    def test_anchor_write_failure_counts_as_a_seal_failure(self, tmp_path: Path) -> None:
        anchor = Mock(spec=["write", "latest"])
        anchor.latest.return_value = None
        anchor.write.side_effect = OSError("anchor down")
        extender = self._failing_seal_extender(tmp_path, head_anchor=anchor, seal_failure_policy="raise")

        with pytest.raises(OSError, match="anchor down"):
            extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert extender.seal_failures == 1

    def test_anchor_latest_failure_counts_as_a_seal_failure_and_seals_nothing(self, tmp_path: Path) -> None:
        anchor = Mock(spec=["write", "latest"])
        anchor.latest.side_effect = ManifestVerificationError("torn anchor")
        calls: list[tuple[str, BaseException]] = []
        extender = self._failing_seal_extender(
            tmp_path, head_anchor=anchor, seal_failure_policy=lambda run_id, exc: calls.append((run_id, exc))
        )
        _, manifest_path = _sealing_config(tmp_path)

        extender.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)

        assert extender.seal_failures == 1
        assert [type(exc) for _, exc in calls] == [ManifestVerificationError]
        assert not manifest_path.exists() or manifest_path.read_text(encoding="utf-8") == ""

    @pytest.mark.parametrize("policy", ["log", "callable", "raise"])
    @pytest.mark.parametrize("how", ["truncate", "delete_both"])
    def test_a_rolled_back_or_deleted_manifest_log_is_caught_via_the_anchor_through_run_all(
        self, tmp_path: Path, how: str, policy: str
    ) -> None:
        audit_path, manifest_path = _sealing_config(tmp_path)
        anchor_path = tmp_path / "anchor.ndjson"

        def fresh(**kwargs: Any) -> AuditExtender:
            return AuditExtender(
                sink=NdjsonAuditSink(audit_path),
                audit_path=audit_path,
                manifest_path=manifest_path,
                signer=_hmac_signer(),
                log_id="log-a",
                head_anchor=NdjsonHeadAnchor(anchor_path),
                **kwargs,
            )

        with verified_context(tenant_id="tenant-1"):
            first = fresh()
            assert run_value_int(first) == expected_value_int()
        assert first.seal_failures == 0
        assert NdjsonHeadAnchor(anchor_path).latest() is not None

        if how == "truncate":
            manifest_path.write_bytes(b"")
        else:
            manifest_path.unlink()
            audit_path.unlink()

        failures: list[tuple[str, BaseException]] = []
        with verified_context(tenant_id="tenant-1"):
            if policy == "callable":
                second = fresh(seal_failure_policy=lambda run_id, exc: failures.append((run_id, exc)))
            else:
                second = fresh(seal_failure_policy=policy)
            if policy == "raise":
                # raise_on_run_complete makes core re-raise the failed auto-seal on an otherwise successful run
                with pytest.raises(ManifestVerificationError):
                    run_value_int(second)
                values = None
            else:
                values = run_value_int(second)

        if policy != "raise":
            assert values == expected_value_int()  # "log" and "callable" keep the run result
        assert second.seal_failures == 1
        if policy == "callable":
            assert len(failures) == 1
            assert isinstance(failures[0][1], ManifestVerificationError)

    def test_a_pickled_copy_with_an_ndjson_head_anchor_unpickles_and_never_seals(self, tmp_path: Path) -> None:
        extender, audit_path, manifest_path, _ = self._anchored_extender(
            tmp_path, log_id="log-a", seal_failure_policy="raise"
        )
        _append_records(audit_path, [_minimal_audit_record("run-1")])

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        copy.on_run_complete(RunContext(run_id="run-1"), _SUCCEEDED)  # no signer: skipped, never a failure
        assert not manifest_path.exists()
        assert copy.seal_failures == 0

    # seal_index_path: forwarded to seal_ndjson_runs.

    @staticmethod
    def _indexed_pair(tmp_path: Path, **kwargs: Any) -> tuple[AuditExtender, Path, Path, Path]:
        """run-1 sealed by an extender with an index; returns it with audit, manifest and index paths."""
        index_path = tmp_path / "seal.index"
        extender, audit_path = _extender_with_run_1_sealed(tmp_path, seal_index_path=index_path, **kwargs)
        return extender, audit_path, _sealing_config(tmp_path)[1], index_path

    def test_auto_seal_builds_the_index_when_seal_index_path_is_given(self, tmp_path: Path) -> None:
        _, _, _, index_path = self._indexed_pair(tmp_path)

        assert index_path.exists()

    def test_seal_index_path_survives_a_pickle_round_trip(self, tmp_path: Path) -> None:
        first, audit_path, manifest_path, index_path = self._indexed_pair(tmp_path)
        second = _second_extender_over_same_sealing_config(
            audit_path, manifest_path, _find_signer_attr(first), seal_index_path=index_path
        )

        copy = pickle.loads(pickle.dumps(second))  # nosec

        assert any(str(value) == str(index_path) for value in vars(copy).values() if isinstance(value, (str, Path)))

    @pytest.mark.parametrize("alias", ["audit", "manifest", "anchor"])
    def test_seal_index_path_aliasing_another_log_file_raises_value_error(self, tmp_path: Path, alias: str) -> None:
        audit_path, manifest_path = _sealing_config(tmp_path)
        anchor_path = tmp_path / "anchor.ndjson"
        target = {"audit": audit_path, "manifest": manifest_path, "anchor": anchor_path}[alias]

        with pytest.raises(ValueError):
            AuditExtender(
                sink=NdjsonAuditSink(audit_path),
                audit_path=audit_path,
                manifest_path=manifest_path,
                signer=_hmac_signer(),
                head_anchor=NdjsonHeadAnchor(anchor_path),
                seal_index_path=target,
            )

    def test_seal_index_path_without_the_sealing_config_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            AuditExtender(sink=InMemoryAuditSink(), seal_index_path=tmp_path / "seal.index")

    @pytest.mark.parametrize("policy", ["log", "raise"])
    def test_an_index_write_error_is_not_a_seal_failure_and_never_triggers_the_policy(
        self, tmp_path: Path, policy: Any
    ) -> None:
        broken = tmp_path / "index-is-a-directory"
        broken.mkdir()
        failures: list[str] = []
        chosen = policy if policy == "raise" else (lambda run_id, exc: failures.append(run_id))

        extender, _ = _extender_with_run_1_sealed(tmp_path, seal_index_path=broken, seal_failure_policy=chosen)

        assert extender.seal_failures == 0
        assert failures == []
        assert json.loads(_sealing_config(tmp_path)[1].read_text(encoding="utf-8").splitlines()[0])["run_id"] == "run-1"

    # Scaling: the whole auto-seal path costs the run, not the history, with the index.

    @staticmethod
    def _scaling_cost(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior: int, *, indexed: bool
    ) -> tuple[int, int, int]:
        """(signature verifications, audit lines parsed, manifest bytes read) of one run's first calculation lookup
        plus on_run_complete, after `prior` sealed runs."""
        tmp_path.mkdir()
        audit_path, manifest_path = _sealing_config(tmp_path)
        extra: dict[str, Any] = {"seal_index_path": tmp_path / "seal.index"} if indexed else {}
        verifications: list[bytes] = []

        class CountingSigner(HmacSha256Signer):
            def verify(self, payload: bytes, signature: str) -> bool:
                verifications.append(payload)
                return super().verify(payload, signature)

        def build() -> AuditExtender:
            return AuditExtender(
                NdjsonAuditSink(audit_path),
                audit_path=audit_path,
                manifest_path=manifest_path,
                signer=CountingSigner(_SEALING_KEY, "seal-key-1"),
                head_anchor=NdjsonHeadAnchor(tmp_path / "anchor.ndjson"),
                **extra,
            )

        for number in range(prior):
            _append_records(audit_path, [_minimal_audit_record(f"run-{number}")])
            build().on_run_complete(RunContext(run_id=f"run-{number}"), _SUCCEEDED)
        verifications.clear()

        parses: list[str] = []
        real_decode = _core._decode_line

        def counting_decode(where: str, line: bytes) -> Any:
            if where.startswith(str(audit_path)):
                parses.append(where)
            return real_decode(where, line)

        read: list[int] = []
        real_open = open

        def spy_open(file: Any, mode: str = "r", *args: Any, **kw: Any) -> Any:
            handle = real_open(file, mode, *args, **kw)
            if mode != "rb" or str(file) != str(manifest_path):
                return handle
            return _ReadSpy(handle, read)

        with monkeypatch.context() as patch:
            _patch_bindings(patch, "_decode_line", counting_decode)
            _patch_bindings(patch, "open", spy_open)
            patch.setattr(audit_extender_module, "open", spy_open, raising=False)
            extender = build()
            with make_hook_context(run_id="run-next", tenant_id=_TENANT).activate():
                extender(_CountingCall())
            extender.on_run_complete(RunContext(run_id="run-next"), _SUCCEEDED)
        return len(verifications), len(parses), sum(read)

    def test_the_whole_auto_seal_costs_the_same_whatever_the_history_with_the_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        costs = [self._scaling_cost(tmp_path / f"p{prior}", monkeypatch, prior, indexed=True) for prior in (2, 8)]

        assert costs[0] == costs[1]
        assert costs[0][0] > 0 or costs[0][1] > 0 or costs[0][2] > 0

    def test_without_seal_index_path_the_whole_auto_seal_cost_grows_with_the_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        small = self._scaling_cost(tmp_path / "p2", monkeypatch, 2, indexed=False)
        large = self._scaling_cost(tmp_path / "p8", monkeypatch, 8, indexed=False)

        assert all(big > little for big, little in zip(large, small))


class TestTeeAuditSinkFlush:
    """flush() fans out to every child that defines one (in order), skips children without one,
    continues past a failing child, and re-raises the first error only after every child ran."""

    def test_flush_calls_every_child_that_has_one_in_order(self) -> None:
        calls: list[str] = []

        class _FlushingSink:
            def __init__(self, name: str) -> None:
                self.name = name

            def write(self, record: Mapping[str, Any]) -> None:
                pass

            def flush(self) -> None:
                calls.append(self.name)

        tee = TeeAuditSink(_FlushingSink("first"), _FlushingSink("second"))

        tee.flush()

        assert calls == ["first", "second"]

    def test_flush_skips_children_without_a_flush_method(self) -> None:
        calls: list[str] = []

        class _FlushingSink:
            def write(self, record: Mapping[str, Any]) -> None:
                pass

            def flush(self) -> None:
                calls.append("flushing")

        tee = TeeAuditSink(InMemoryAuditSink(), _FlushingSink())

        tee.flush()  # InMemoryAuditSink has no flush(); must not raise or be called

        assert calls == ["flushing"]

    def test_flush_continues_past_a_failing_child_and_reraises_the_first_error(self) -> None:
        calls: list[str] = []

        class _FlushRecordingSink:
            def __init__(self, name: str, error: Exception | None = None) -> None:
                self.name = name
                self.error = error

            def write(self, record: Mapping[str, Any]) -> None:
                pass

            def flush(self) -> None:
                calls.append(self.name)
                if self.error is not None:
                    raise self.error

        first_error = RuntimeError("first boom")
        second_error = RuntimeError("second boom")
        tee = TeeAuditSink(
            _FlushRecordingSink("a", first_error),
            _FlushRecordingSink("b", second_error),
            _FlushRecordingSink("c"),
        )

        with pytest.raises(RuntimeError, match="first boom") as excinfo:
            tee.flush()

        assert excinfo.value is first_error
        assert calls == ["a", "b", "c"]


class TestAuditExtenderPolicyVersion:
    """policy_version is an explicit label or, by default, a fingerprint of the configured gate, on every record."""

    @_BOTH_POSTURES
    def test_default_is_a_12_char_lowercase_hex_fingerprint(self, fail_closed: bool) -> None:
        extender = AuditExtender(sink=InMemoryAuditSink(), fail_closed=fail_closed)

        assert isinstance(extender.policy_version, str)
        assert _FINGERPRINT.fullmatch(extender.policy_version)

    @_BOTH_POSTURES
    def test_same_policy_gives_the_same_fingerprint(self, fail_closed: bool) -> None:
        first = AuditExtender(
            sink=InMemoryAuditSink(), required_identity=("tenant_id", "principal"), fail_closed=fail_closed
        )
        second = AuditExtender(
            sink=InMemoryAuditSink(), required_identity=("tenant_id", "principal"), fail_closed=fail_closed
        )

        assert first.policy_version == second.policy_version

    @pytest.mark.parametrize(
        "changed",
        [
            {"required_identity": ("tenant_id", "principal")},
            {"required_identity": ("principal",)},
            {"fail_closed": True},
        ],
        ids=["extra_identity_name", "other_identity_name", "fail_closed"],
    )
    def test_a_different_gate_gives_a_different_fingerprint(self, changed: dict[str, Any]) -> None:
        baseline = AuditExtender(sink=InMemoryAuditSink())

        assert AuditExtender(sink=InMemoryAuditSink(), **changed).policy_version != baseline.policy_version

    @pytest.mark.parametrize(
        ("options", "expected"),
        [
            ({}, "4003968045ae"),
            ({"fail_closed": True, "required_identity": ("tenant_id", "principal")}, "809a86028b49"),
        ],
        ids=["defaults", "fail_closed"],
    )
    def test_the_default_fingerprint_without_a_classification_policy_is_unchanged(
        self, options: dict[str, Any], expected: str
    ) -> None:
        assert AuditExtender(sink=InMemoryAuditSink(), classification=None, **options).policy_version == expected

    @_BOTH_POSTURES
    def test_required_identity_order_does_not_change_the_fingerprint(self, fail_closed: bool) -> None:
        forward = AuditExtender(
            sink=InMemoryAuditSink(), required_identity=("tenant_id", "principal"), fail_closed=fail_closed
        )
        backward = AuditExtender(
            sink=InMemoryAuditSink(), required_identity=("principal", "tenant_id"), fail_closed=fail_closed
        )

        assert forward.policy_version == backward.policy_version

    def test_explicit_value_is_kept_as_given(self) -> None:
        extender = AuditExtender(sink=InMemoryAuditSink(), policy_version=_POLICY_VERSION)

        assert extender.policy_version == _POLICY_VERSION

    @_BOTH_POSTURES
    def test_default_fingerprint_is_recorded_on_an_allow_record(self, fail_closed: bool) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=fail_closed)

        with make_hook_context(tenant_id="tenant-1").activate():
            extender(lambda: None)

        assert _FINGERPRINT.fullmatch(sink.records[0]["policy_version"])
        assert sink.records[0]["policy_version"] == extender.policy_version

    @_BOTH_POSTURES
    def test_explicit_value_is_recorded_on_an_allow_record(self, fail_closed: bool) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=fail_closed, policy_version=_POLICY_VERSION)

        with make_hook_context(tenant_id="tenant-1").activate():
            extender(lambda: None)

        assert sink.records[0]["decision"] == "allow"
        assert sink.records[0]["policy_version"] == _POLICY_VERSION

    def test_explicit_value_is_recorded_on_a_deny_record(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, policy_version=_POLICY_VERSION)

        with make_hook_context().activate():
            extender(_CountingCall())

        assert sink.records[0]["decision"] == "deny"
        assert sink.records[0]["policy_version"] == _POLICY_VERSION

    @pytest.mark.parametrize(
        "hook", [ExtenderHook.FEATURE_GROUP_MATCHED, ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE], ids=lambda h: h.name
    )
    def test_explicit_value_is_recorded_on_a_fail_closed_refusal(self, hook: ExtenderHook) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=True, policy_version=_POLICY_VERSION)

        unresolved: dict[str, Any] = (
            {"feature_group_class": None, "feature_group_version": None, "compute_framework_name": None}
            if hook is ExtenderHook.FEATURE_GROUP_MATCHED
            else {}
        )
        with make_hook_context(hook=hook, **unresolved).activate():
            with pytest.raises(IdentityRequiredError):
                extender(_CountingCall())

        assert sink.records[0]["hook"] == hook.name
        assert sink.records[0]["policy_version"] == _POLICY_VERSION


class TestAuditExtenderDataAccess:
    """A load nested in a calculate call is recorded on that call's record with core's identity, never its own."""

    def test_calculate_without_a_load_records_empty_lists(self) -> None:
        record = _record_for_loads([])

        assert record["data_access_identity"] == []
        assert record["data_access_format"] == []
        assert record["data_access_identity_is_fallback"] == []

    def test_refusal_record_carries_empty_data_access_lists(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=True)

        with make_hook_context().activate():
            with pytest.raises(IdentityRequiredError):
                extender(_CountingCall())

        assert sink.records[0]["data_access_identity"] == []
        assert sink.records[0]["data_access_format"] == []
        assert sink.records[0]["data_access_identity_is_fallback"] == []

    def test_nested_load_lands_on_the_enclosing_record_and_writes_no_record_of_its_own(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)
        results: list[Any] = []

        def body() -> None:
            results.append(_load(extender, _BUCKET_KEY, "ParquetReader"))

        _calculate(extender, body)

        assert results == ["loaded"]
        assert len(sink.records) == 1
        record = sink.records[0]
        assert record["hook"] == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE.name
        assert record["record_version"] == 2
        assert record["data_access_identity"] == [_BUCKET_KEY]
        assert record["data_access_format"] == ["ParquetReader"]

    def test_format_none_stays_none_inside_the_list(self) -> None:
        record = _record_for_loads([(_BUCKET_KEY, None)])

        assert record["data_access_identity"] == [_BUCKET_KEY]
        assert record["data_access_format"] == [None]

    def test_load_without_an_identity_is_skipped(self) -> None:
        record = _record_for_loads([(None, "CsvReader", True)])

        assert record["data_access_identity"] == []
        assert record["data_access_format"] == []
        assert record["data_access_identity_is_fallback"] == []

    def test_failing_load_appears_on_the_error_record_and_the_exception_propagates(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)
        calls = 0

        def failing_load() -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError("load boom")

        def body() -> None:
            with _load_context(_BUCKET_KEY, "ParquetReader").activate():
                extender(failing_load)

        with pytest.raises(RuntimeError, match="load boom"):
            _calculate(extender, body)

        assert calls == 1
        assert len(sink.records) == 1
        record = sink.records[0]
        assert record["status"] == "error"
        assert record["error_type"] == "builtins.RuntimeError"
        assert record["data_access_identity"] == [_BUCKET_KEY]
        assert record["data_access_format"] == ["ParquetReader"]

    def test_failing_load_swallowed_by_the_calculate_body_is_still_recorded(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        def failing_load() -> None:
            raise RuntimeError("load boom")

        def body() -> None:
            with _load_context(_BUCKET_KEY).activate():
                with pytest.raises(RuntimeError, match="load boom"):
                    extender(failing_load)

        _calculate(extender, body)

        assert len(sink.records) == 1
        assert sink.records[0]["status"] == "success"
        assert sink.records[0]["data_access_identity"] == [_BUCKET_KEY]

    def test_repeated_identical_loads_are_deduplicated(self) -> None:
        record = _record_for_loads([(_BUCKET_KEY, "ParquetReader")] * 3)

        assert record["data_access_identity"] == [_BUCKET_KEY]
        assert record["data_access_format"] == ["ParquetReader"]

    def test_distinct_loads_keep_first_seen_order_and_stay_index_aligned(self) -> None:
        other = "s3://bucket/other.csv"

        record = _record_for_loads(
            [
                (_BUCKET_KEY, "ParquetReader", False),
                (other, "CsvReader", True),
                (_BUCKET_KEY, "ParquetReader", False),
            ]
        )

        assert record["data_access_identity"] == [_BUCKET_KEY, other]
        assert record["data_access_format"] == ["ParquetReader", "CsvReader"]
        assert record["data_access_identity_is_fallback"] == [False, True]

    def test_same_identity_and_format_with_different_flags_gives_two_entries(self) -> None:
        record = _record_for_loads(
            [("str", "CsvReader", True), ("str", "CsvReader", False), ("str", "CsvReader", True)]
        )

        assert record["data_access_identity"] == ["str", "str"]
        assert record["data_access_format"] == ["CsvReader", "CsvReader"]
        assert record["data_access_identity_is_fallback"] == [True, False]

    def test_hand_built_context_without_the_flag_records_none(self) -> None:
        record = _record_for_loads([(_BUCKET_KEY, "CsvReader")])

        assert record["data_access_identity_is_fallback"] == [None]

    def test_same_identity_under_two_formats_gives_two_entries(self) -> None:
        record = _record_for_loads([(_BUCKET_KEY, "CsvReader"), (_BUCKET_KEY, "ParquetReader")])

        assert record["data_access_identity"] == [_BUCKET_KEY, _BUCKET_KEY]
        assert record["data_access_format"] == ["CsvReader", "ParquetReader"]

    def test_loads_whose_core_identity_matches_are_deduplicated_even_when_raw_args_differ(self) -> None:
        # Raw args[0] differ in userinfo and query; core projects both to the same identity.
        raw_one = "https://a:1@host/x?sig=one"
        raw_two = "https://b:2@host/x?sig=two"
        assert BaseInputData.data_access_identity(raw_one) == BaseInputData.data_access_identity(raw_two)
        context_identity = BaseInputData.data_access_identity(raw_one)
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        def body() -> None:
            _load(extender, context_identity, "CsvReader", args=(raw_one, _FEATURES_PLACEHOLDER))
            _load(extender, context_identity, "CsvReader", args=(raw_two, _FEATURES_PLACEHOLDER))

        _calculate(extender, body)

        assert len(sink.records) == 1
        record = sink.records[0]
        assert record["data_access_identity"] == ["https://host/x"]
        assert record["data_access_format"] == ["CsvReader"]

    def test_two_azure_containers_on_one_account_and_path_stay_distinct_entries(self) -> None:
        marker = "SECRET"
        raw_raw_container = f"abfss://raw@acct.dfs.core.windows.net/p?sv=1&sig={marker}"
        raw_curated_container = f"abfss://curated@acct.dfs.core.windows.net/p?sv=1&sig={marker}"
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        def body() -> None:
            for raw in (raw_raw_container, raw_curated_container):
                context_identity = BaseInputData.data_access_identity(raw)
                _load(extender, context_identity, "CsvReader", args=(raw, _FEATURES_PLACEHOLDER))

        _calculate(extender, body)

        assert len(sink.records) == 1
        record = sink.records[0]
        assert record["data_access_identity"] == [
            "abfss://raw@acct.dfs.core.windows.net/p",
            "abfss://curated@acct.dfs.core.windows.net/p",
        ]
        assert marker not in json.dumps(record)

    def test_nested_calculate_attributes_each_load_to_its_own_level(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        def inner_body() -> None:
            _load(extender, "s3://bucket/inner.parquet")

        def outer_body() -> None:
            _calculate(extender, inner_body, feature_group_class=_INNER_CLASS)
            _load(extender, "s3://bucket/outer.parquet")

        _calculate(extender, outer_body, feature_group_class=_OUTER_CLASS)

        by_class = {record["feature_group_class"]: record for record in sink.records}
        assert len(sink.records) == 2
        assert by_class[_INNER_CLASS]["data_access_identity"] == ["s3://bucket/inner.parquet"]
        assert by_class[_OUTER_CLASS]["data_access_identity"] == ["s3://bucket/outer.parquet"]

    def test_load_after_a_raising_inner_calculate_attaches_to_the_outer_call(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        def inner_body() -> None:
            _load(extender, "s3://bucket/inner.parquet")
            raise RuntimeError("inner boom")

        def outer_body() -> None:
            with pytest.raises(RuntimeError, match="inner boom"):
                _calculate(extender, inner_body, feature_group_class=_INNER_CLASS)
            _load(extender, "s3://bucket/outer.parquet")

        _calculate(extender, outer_body, feature_group_class=_OUTER_CLASS)

        by_class = {record["feature_group_class"]: record for record in sink.records}
        assert len(sink.records) == 2
        assert by_class[_INNER_CLASS]["status"] == "error"
        assert by_class[_INNER_CLASS]["data_access_identity"] == ["s3://bucket/inner.parquet"]
        assert by_class[_OUTER_CLASS]["data_access_identity"] == ["s3://bucket/outer.parquet"]

    def test_stack_is_restored_after_a_calculate_that_raises(self, caplog: pytest.LogCaptureFixture) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        def failing_body() -> None:
            raise RuntimeError("calculate boom")

        with pytest.raises(RuntimeError, match="calculate boom"):
            _calculate(extender, failing_body)
        with caplog.at_level(logging.DEBUG):
            assert _load(extender, _BUCKET_KEY) == "loaded"

        assert len(sink.records) == 1
        assert sink.records[0]["data_access_identity"] == []
        _assert_logged_no_enclosing_calculate(caplog)

    @_BOTH_POSTURES
    def test_load_without_an_enclosing_calculate_passes_through_writes_nothing_and_logs_debug(
        self, fail_closed: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=fail_closed)
        call = _CountingCall()

        with caplog.at_level(logging.DEBUG):
            with _load_context(_BUCKET_KEY, "ParquetReader").activate():
                result = extender(call)

        assert result == 42
        assert call.calls == 1
        assert sink.records == []
        _assert_logged_no_enclosing_calculate(caplog)

    @_BOTH_POSTURES
    def test_nested_load_passes_through_untouched_even_without_a_tenant_id_on_the_load(self, fail_closed: bool) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=fail_closed)
        call = _CountingCall()
        results: list[int] = []

        def body() -> None:
            with _load_context(_BUCKET_KEY).activate():
                results.append(extender(call))

        _calculate(extender, body)

        assert results == [42]
        assert call.calls == 1
        assert len(sink.records) == 1
        record = sink.records[0]
        assert record["hook"] == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE.name
        assert record["decision"] == "allow"
        assert record["data_access_identity"] == [_BUCKET_KEY]

    def test_composite_of_two_extenders_attributes_each_load_to_the_extender_that_saw_it(self) -> None:
        sink_a = InMemoryAuditSink()
        sink_b = InMemoryAuditSink()
        extender_a = AuditExtender(sink=sink_a)
        extender_b = AuditExtender(sink=sink_b)
        composite = CompositeExtender([extender_a, extender_b])

        def body() -> None:
            _load(composite, "s3://bucket/both.parquet")
            _load(extender_a, "s3://bucket/only-a.parquet")

        _calculate(composite, body)

        assert [r["data_access_identity"] for r in sink_a.records] == [
            ["s3://bucket/both.parquet", "s3://bucket/only-a.parquet"]
        ]
        assert [r["data_access_identity"] for r in sink_b.records] == [["s3://bucket/both.parquet"]]

    @pytest.mark.parametrize(("raw", "expected", "secrets"), _DATA_ACCESS_IDENTITY_CASES)
    def test_the_core_identity_is_recorded_never_the_raw_data_access(
        self, raw: Any, expected: str, secrets: tuple[str, ...]
    ) -> None:
        context_identity = BaseInputData.data_access_identity(raw)
        record = _record_for_loads([(context_identity, "CsvReader")], args=(raw, _FEATURES_PLACEHOLDER))

        assert record["data_access_identity"] == [expected]
        assert record["data_access_format"] == ["CsvReader"]
        for secret in secrets:
            assert secret not in json.dumps(record)

    def test_a_load_without_a_context_identity_is_skipped_even_with_a_uri_first_argument(self) -> None:
        record = _record_for_loads([(None, "CsvReader")], args=("https://host/p?sig=SECRET", _FEATURES_PLACEHOLDER))

        assert record["data_access_identity"] == []
        assert record["data_access_format"] == []

    def test_a_reader_overridden_context_identity_is_recorded_exactly_as_given(self) -> None:
        # Models a reader whose data_access_identity() override diverges from core's own projection of args[0].
        overridden_identity = "https://api.example/x?station=7"
        record = _record_for_loads(
            [(overridden_identity, "CsvReader")], args=("s3://bucket/raw/key.parquet", _FEATURES_PLACEHOLDER)
        )

        assert record["data_access_identity"] == [overridden_identity]


class TestNdjsonAuditSink:
    """NdjsonAuditSink appends one JSON line per record, surviving a pickle round trip."""

    def test_appends_one_json_line_per_write(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.ndjson"
        sink = NdjsonAuditSink(path)

        sink.write({"a": 1})
        sink.write({"a": 2})

        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == [{"a": 1}, {"a": 2}]

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
    def test_non_finite_number_raises_value_error_and_writes_nothing(self, tmp_path: Path, value: float) -> None:
        path = tmp_path / "audit.ndjson"
        sink = NdjsonAuditSink(path)

        with pytest.raises(ValueError):
            sink.write({"a": value})

        assert not path.exists() or path.read_bytes() == b""

    def test_pickled_copy_appends_to_the_same_file(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.ndjson"
        sink = NdjsonAuditSink(path)
        sink.write({"a": 1})

        copy = pickle.loads(pickle.dumps(sink))  # nosec
        copy.write({"a": 2})

        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == [{"a": 1}, {"a": 2}]

    def test_new_file_has_owner_only_permissions(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.ndjson"
        sink = NdjsonAuditSink(path)

        sink.write({"a": 1})

        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_oversized_record_lands_as_one_exact_line(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.ndjson"
        sink = NdjsonAuditSink(path)
        record = {"feature_names": ["f" * 50] * 500}
        assert len(json.dumps(record)) > 16 * 1024

        sink.write(record)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == record

    def test_record_over_the_line_cap_is_refused_before_writing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "audit.ndjson"
        sink = NdjsonAuditSink(path)
        sink.write({"a": 1})
        before = path.read_bytes()
        _patch_bindings(monkeypatch, "MAX_LINE_BYTES", 100)

        with pytest.raises(ValueError):
            sink.write({"a": "x" * 100})

        assert path.read_bytes() == before

    def test_a_write_whose_file_was_swapped_after_opening_lands_in_the_new_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        path = tmp_path / "audit.ndjson"
        archive = tmp_path / "audit.ndjson.000001"
        path.write_bytes(b'{"old": 1}\n')
        new = tmp_path / "new.ndjson"
        new.write_bytes(b'{"carried": 1}\n')
        swapped: list[int] = []
        real_flock = fcntl.flock

        def flock(fd: int, operation: int) -> None:
            if not swapped:
                os.link(path, archive)
                os.replace(new, path)
                swapped.append(operation)
            real_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", flock)

        NdjsonAuditSink(path).write({"a": 1})

        assert swapped == [fcntl.LOCK_SH]
        assert path.read_bytes() == b'{"carried": 1}\n{"a": 1}\n'
        assert archive.read_bytes() == b'{"old": 1}\n'

    @pytest.mark.parametrize("failure", ["no-fcntl", "flock-error"])
    def test_the_record_is_written_without_a_lock_when_locking_is_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
    ) -> None:
        fcntl = pytest.importorskip("fcntl")
        path = tmp_path / "audit.ndjson"
        if failure == "no-fcntl":
            monkeypatch.setitem(sys.modules, "fcntl", None)
        else:

            def unsupported(fd: int, operation: int) -> None:
                raise OSError(errno.ENOLCK, "no locks")

            monkeypatch.setattr(fcntl, "flock", unsupported)

        NdjsonAuditSink(path).write({"a": 1})

        assert path.read_bytes() == b'{"a": 1}\n'

    def test_short_write_raises_os_error_naming_the_path(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.ndjson"
        sink = NdjsonAuditSink(path)
        record = {"a": "x" * 40}
        real_write = os.write

        def short_write(fd: int, data: bytes | memoryview) -> int:
            return real_write(fd, data[:7])

        with patch("os.write", side_effect=short_write) as mock_write:
            with pytest.raises(OSError) as excinfo:
                sink.write(record)

        assert str(path) in str(excinfo.value)
        assert mock_write.call_count == 1
        line = json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"
        assert path.read_bytes() == line[:7]

    def test_zero_byte_write_raises_os_error(self, tmp_path: Path) -> None:
        sink = NdjsonAuditSink(tmp_path / "audit.ndjson")

        with patch("os.write", return_value=0) as mock_write:
            with pytest.raises(OSError):
                sink.write({"a": 1})

        assert mock_write.call_count == 1

    def test_small_record_is_one_write_carrying_the_whole_line(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.ndjson"
        sink = NdjsonAuditSink(path)
        line = json.dumps({"a": 1}, sort_keys=True).encode("utf-8") + b"\n"

        with patch("os.write", wraps=os.write) as mock_write:
            sink.write({"a": 1})

        assert mock_write.call_count == 1
        assert [bytes(call.args[1]) for call in mock_write.call_args_list] == [line]
        assert path.read_bytes() == line


class TestTeeAuditSink:
    """TeeAuditSink forwards each record to every sink in order and stops at the first failure."""

    def test_is_defined_in_the_audit_extender_module(self) -> None:
        assert TeeAuditSink is audit_extender_module.TeeAuditSink

    def test_no_sinks_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            TeeAuditSink()

    @pytest.mark.parametrize(
        "invalid",
        [NdjsonAuditSink, None, "not-a-sink", SimpleNamespace(write="not-callable")],
        ids=["class_instead_of_instance", "none", "str", "write_not_callable"],
    )
    def test_a_sink_without_a_callable_write_raises_value_error(self, invalid: Any) -> None:
        with pytest.raises(ValueError, match="write"):
            TeeAuditSink(invalid)

    def test_an_invalid_sink_after_a_valid_one_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="write"):
            TeeAuditSink(InMemoryAuditSink(), None)  # type: ignore[arg-type]

    def test_valid_sinks_construct_and_keep_their_order(self, tmp_path: Path) -> None:
        memory = InMemoryAuditSink()
        ndjson = NdjsonAuditSink(tmp_path / "audit.ndjson")

        assert TeeAuditSink(memory, ndjson).sinks == (memory, ndjson)

    def test_writes_the_same_record_to_every_sink_in_the_order_given(self) -> None:
        log: list[tuple[str, Mapping[str, Any]]] = []
        tee = TeeAuditSink(_LoggingSink("first", log), _LoggingSink("second", log), _LoggingSink("third", log))
        record = {"a": 1}

        tee.write(record)

        assert [name for name, _ in log] == ["first", "second", "third"]
        assert all(written is record for _, written in log)

    def test_first_failure_propagates_unchanged_and_later_sinks_are_not_called(self) -> None:
        error = OSError("disk full")
        before = InMemoryAuditSink()
        after = InMemoryAuditSink()
        tee = TeeAuditSink(before, _FailingSink(error), after)

        with pytest.raises(OSError, match="disk full") as excinfo:
            tee.write({"a": 1})

        assert excinfo.value is error
        assert before.records == [{"a": 1}]
        assert after.records == []

    def test_pickled_copy_appends_to_every_file(self, tmp_path: Path) -> None:
        paths = [tmp_path / "first.ndjson", tmp_path / "second.ndjson"]
        tee = TeeAuditSink(NdjsonAuditSink(paths[0]), NdjsonAuditSink(paths[1]))
        tee.write({"a": 1})

        copy = pickle.loads(pickle.dumps(tee))  # nosec
        copy.write({"a": 2})

        for path in paths:
            lines = path.read_text(encoding="utf-8").splitlines()
            assert [json.loads(line) for line in lines] == [{"a": 1}, {"a": 2}]

    def test_extender_record_reaches_the_memory_and_the_ndjson_sink(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.ndjson"
        memory = InMemoryAuditSink()
        extender = AuditExtender(sink=TeeAuditSink(memory, NdjsonAuditSink(path)))

        with make_hook_context(tenant_id="tenant-1").activate():
            extender(lambda: None)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(memory.records) == 1
        assert [json.loads(line) for line in lines] == memory.records


class TestAuditExtenderRunAll:
    """run_all round trips: identity resolved through verified_context, or missing entirely."""

    @_BOTH_POSTURES
    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    def test_run_all_with_verified_context_allows_and_records(
        self, mode: ParallelizationMode, fail_closed: bool
    ) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=fail_closed)

        with verified_context(tenant_id="tenant-42", project_id="project-7", principal="svc"):
            values = run_value_int(extender, parallelization_modes={mode})

        assert values == expected_value_int()
        assert sink.records
        for record in sink.records:
            assert record["tenant_id"] == "tenant-42"
            assert record["decision"] == "allow"
            assert record["status"] == "success"
            assert record["policy_version"] == extender.policy_version
            assert record["run_id"] is not None
            assert record["hook"] == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE.name

    @_BOTH_POSTURES
    def test_run_outside_a_verified_context_inherits_the_plan_time_identity(self, fail_closed: bool) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=fail_closed)

        with verified_context(tenant_id="tenant-42", project_id="project-7", principal="svc"):
            session = prepare_value_int(extender)
        session.run()

        assert sink.records
        for record in sink.records:
            assert record["tenant_id"] == "tenant-42"
            assert record["decision"] == "allow"

    @pytest.mark.parametrize(
        "mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING]
    )
    def test_run_all_csv_read_records_the_load_identity_and_format(
        self, mode: ParallelizationMode, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        extender = AuditExtender(sink=NdjsonAuditSink(audit_path))
        # Only MULTIPROCESSING needs the flight_server fixture.
        flight_server = (
            request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
        )

        with verified_context(tenant_id="tenant-42"):
            run_csv_feature(tmp_path, extender, parallelization_modes={mode}, flight_server=flight_server)

        read_class = f"{ReadFileFeature.__module__}.{ReadFileFeature.__qualname__}"
        sink_records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        records = [record for record in sink_records if record["feature_group_class"] == read_class]
        assert len(records) == 1
        run_hashes = {record["structure_hash"] for record in sink_records if record["run_id"] is not None}
        assert len(run_hashes) == 1
        assert all(run_hashes)
        assert records[0]["data_access_identity"] == [str(tmp_path / "data.csv")]
        assert records[0]["data_access_format"] == ["CsvReader"]
        assert records[0]["data_access_identity_is_fallback"] == [False]

    def test_run_all_csv_read_with_a_fallback_identity_records_the_flag_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A reader that does not override the identity falls back to the type name of its data access ("str").
        monkeypatch.setattr(
            CsvReader, "data_access_identity", classmethod(lambda cls, data_access: type(data_access).__name__)
        )
        audit_path = tmp_path / "audit.ndjson"
        extender = AuditExtender(sink=NdjsonAuditSink(audit_path))

        with verified_context(tenant_id="tenant-42"):
            run_csv_feature(tmp_path, extender)

        read_class = f"{ReadFileFeature.__module__}.{ReadFileFeature.__qualname__}"
        sink_records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        records = [record for record in sink_records if record["feature_group_class"] == read_class]
        assert len(records) == 1
        assert records[0]["data_access_identity"] == ["str"]
        assert records[0]["data_access_identity_is_fallback"] == [True]

    def test_run_all_without_verified_context_denies_but_still_runs(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        values = run_value_int(extender)

        assert values == expected_value_int()
        assert sink.records
        for record in sink.records:
            assert record["decision"] == "deny"
            assert record["deny_reason"] == "missing_tenant_id"

    @_FAIL_CLOSED_RAISE_ON_ERROR_POSTURES
    def test_run_all_fail_closed_without_verified_context_refuses_at_plan_time(
        self, make_gate_extender: Callable[[Any], AuditExtender]
    ) -> None:
        sink = InMemoryAuditSink()
        counting = CountingExtender()

        with pytest.raises(IdentityRequiredError):
            run_value_int(make_gate_extender(sink), counting)

        assert len(sink.records) == 1
        record = sink.records[0]
        assert record["hook"] == ExtenderHook.FEATURE_GROUP_MATCHED.name
        assert record["decision"] == "deny"
        assert record["feature_group_class"] is None
        assert record["feature_group_version"] is None
        assert record["compute_framework_name"] is None
        assert counting.calls == 0

    @_FAIL_CLOSED_RAISE_ON_ERROR_POSTURES
    @pytest.mark.parametrize(
        "mode",
        [
            ParallelizationMode.SYNC,
            pytest.param(
                ParallelizationMode.THREADING,
                marks=pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning"),
            ),
            ParallelizationMode.MULTIPROCESSING,
        ],
    )
    def test_run_all_fail_closed_refuses_at_run_start_a_run_whose_identity_lost_the_tenant_since_prepare(
        self,
        mode: ParallelizationMode,
        make_gate_extender: Callable[[Any], AuditExtender],
        tmp_path: Path,
        request: pytest.FixtureRequest,
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        marker_path = tmp_path / "counting_calls"
        counting = CountingExtender(marker_path=marker_path)
        # Lower than the default 100: it runs outside the gate unless the gate sorts itself outermost.
        counting.priority = 50
        # Only MULTIPROCESSING needs the flight_server fixture.
        flight_server = (
            request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
        )
        # For MULTIPROCESSING, never_fall_back must survive pickling into the worker.
        gate = make_gate_extender(NdjsonAuditSink(audit_path))

        with verified_context(tenant_id="t"):
            session = prepare_value_int(gate, counting, parallelization_modes={mode})

        # An explicit run-time verified context replaces the plan-time identity; the gate refuses it at run start.
        with verified_context(project_id="p"):
            with pytest.raises(IdentityRequiredError):
                session.run(parallelization_modes={mode}, flight_server=flight_server)

        lines = audit_path.read_text(encoding="utf-8").splitlines() if audit_path.exists() else []
        records = [json.loads(line) for line in lines]
        assert len(records) == 1
        record = records[0]
        assert record["hook"] == "RUN_START"
        assert record["decision"] == "deny"
        assert record["status"] == "error"
        assert record["run_id"] is not None
        assert counting.calls == 0
        # Marker file covers MULTIPROCESSING: a worker's own `calls` copy would be invisible here.
        assert not marker_path.exists()

    @_BOTH_POSTURES
    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING])
    def test_run_all_fail_closed_allows_a_run_that_switches_to_another_tenant_since_prepare(
        self, mode: ParallelizationMode, fail_closed: bool, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        # Only MULTIPROCESSING needs the flight_server fixture.
        flight_server = (
            request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
        )
        extender = AuditExtender(sink=NdjsonAuditSink(audit_path), fail_closed=fail_closed)

        with verified_context(tenant_id="tenant-a", project_id="project-7", principal="svc"):
            session = prepare_value_int(extender, parallelization_modes={mode})
        with verified_context(tenant_id="tenant-b", project_id="project-7", principal="svc"):
            results = session.run(parallelization_modes={mode}, flight_server=flight_server)

        values = next(t.to_pydict()["value_int"] for t in results if "value_int" in t.column_names)
        assert values == expected_value_int()
        records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        calculate = [r for r in records if r["hook"] == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE.name]
        assert calculate
        for record in calculate:
            assert record["tenant_id"] == "tenant-b"
            assert record["decision"] == "allow"


class TestAuditExtenderClassificationRunAll:
    """run_all round trips of the classification gate under verified_context."""

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING])
    def test_a_run_requesting_an_uncleared_feature_is_refused_before_any_row(
        self, mode: ParallelizationMode, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        marker_path = tmp_path / "counting_calls"
        counting = CountingExtender(marker_path=marker_path)
        counting.priority = 50
        # Only MULTIPROCESSING needs the flight_server fixture.
        flight_server = (
            request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
        )
        gate = _classified(NdjsonAuditSink(audit_path), "internal")

        with verified_context(tenant_id="t", principal="svc"):
            with pytest.raises(ClassificationDeniedError):
                _run_class_features([_DERIVED], gate, counting, mode=mode, flight_server=flight_server)

        records = _ndjson(audit_path)
        assert len(records) == 1
        assert records[0]["hook"] == "RUN_START"
        assert records[0]["decision"] == "deny"
        assert records[0]["deny_reason"] == "classification_above_clearance"
        assert records[0]["feature_names"] == [_DERIVED]
        assert records[0]["classification"] == "pii"
        assert counting.calls == 0
        # Marker file covers MULTIPROCESSING: a worker's own `calls` copy would be invisible here.
        assert not marker_path.exists()

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING])
    def test_a_cleared_principal_runs_and_every_allow_record_carries_the_level(
        self, mode: ParallelizationMode, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        # Only MULTIPROCESSING needs the flight_server fixture.
        flight_server = (
            request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
        )
        gate = _classified(NdjsonAuditSink(audit_path), "pii")

        with verified_context(tenant_id="t", principal="svc"):
            _run_class_features([_DERIVED], gate, mode=mode, flight_server=flight_server)

        records = [r for r in _ndjson(audit_path) if r["hook"] == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE.name]
        assert len(records) == 2
        for record in records:
            assert record["decision"] == "allow"
            assert record["classification"] == "pii"

    def test_a_masked_derivative_of_pii_is_allowed_for_an_internal_principal(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        gate = _classified(NdjsonAuditSink(audit_path), "internal")

        with verified_context(tenant_id="t", principal="svc"):
            _run_class_features([_MASKED], gate)

        records = [r for r in _ndjson(audit_path) if r["hook"] == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE.name]
        assert {r["decision"] for r in records} == {"allow"}
        assert {r["classification"] for r in records} == {"pii", "internal"}

    def test_a_rerun_of_one_prepared_session_is_checked_per_principal(self) -> None:
        sink = InMemoryAuditSink()
        gate = _classified(sink, lambda tenant, principal: "pii" if principal == "alice" else "internal")

        with verified_context(tenant_id="t", principal="alice"):
            session = _prepare_class_feature(_DERIVED, gate)
        session.run()
        with verified_context(tenant_id="t", principal="bob"):
            with pytest.raises(ClassificationDeniedError):
                session.run()

        denies = [r for r in sink.records if r["decision"] == "deny"]
        assert [r["deny_reason"] for r in denies] == ["classification_above_clearance"]
        assert denies[0]["principal"] == "bob"

    @pytest.mark.parametrize(("clearance", "denied"), [("internal", True), ("pii", False)])
    def test_a_reader_declared_pii_root_is_gated_through_run_csv_feature(
        self, clearance: str, denied: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            CsvReader, "declared_attributes", classmethod(lambda cls, features: {CLASSIFICATION_KEY: "pii"})
        )
        sink = InMemoryAuditSink()
        gate = _classified(sink, clearance)

        with verified_context(tenant_id="t", principal="svc"):
            if denied:
                with pytest.raises(ClassificationDeniedError):
                    run_csv_feature(tmp_path, gate)
            else:
                run_csv_feature(tmp_path, gate)

        if denied:
            assert [r["deny_reason"] for r in sink.records] == ["classification_above_clearance"]
            assert sink.records[0]["feature_names"] == ["alpha"]
        else:
            assert {r["classification"] for r in sink.records} == {"pii"}
