"""Tests for AuditExtender: contract compliance, record shape, allow/deny decisions, the fail_closed
refusal, error handling and NdjsonAuditSink; __call__ tests build a HookContext manually, mirroring
core's own instrumentation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle  # nosec
import re
import stat
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
from mloda.steward import CompositeExtender, Extender, ExtenderHook, HookContext, verified_context
from mloda.user import ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature

from mloda.enterprise.extenders.audit import (
    AuditExtender,
    Ed25519Signer,
    HmacSha256Signer,
    IdentityRequiredError,
    ManifestVerificationError,
    NdjsonAuditSink,
    RunAlreadySealedError,
    TeeAuditSink,
    seal_ndjson_runs,
    verify_ndjson_log_coverage,
)
from mloda.enterprise.extenders.audit import audit_extender as audit_extender_module
from mloda.enterprise.extenders.audit._records import _append_records
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.runners import (
    CountingExtender,
    expected_value_int,
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
    "policy_version",
}

_FINGERPRINT = re.compile(r"[0-9a-f]{12}")

# Neither 12 characters nor hex, so a truncated or hashed value would not equal it.
_POLICY_VERSION = "policy-2026-09-rev-3"


# Sealing (AuditExtender.on_run_complete) fixtures: deterministic keys, never used for anything but these tests.
_SEALING_KEY = b"k" * 32
_SEALING_KEY_B = b"o" * 32


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


def _extender_with_run_1_sealed(
    tmp_path: Path, make: Callable[..., AuditExtender] = AuditExtender, **kwargs: Any
) -> tuple[AuditExtender, Path]:
    """Seal run-1 via on_run_complete; return (extender, audit_path)."""
    audit_path = tmp_path / "audit.ndjson"
    manifest_path = tmp_path / "manifest.ndjson"
    _append_records(audit_path, [_minimal_audit_record("run-1")])
    signer = _hmac_signer()
    extender = make(
        NdjsonAuditSink(audit_path), audit_path=audit_path, manifest_path=manifest_path, signer=signer, **kwargs
    )
    extender.on_run_complete("run-1")
    return extender, audit_path


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


def _load_context(identity: str | None, data_format: str | None = None) -> HookContext:
    """An INPUT_DATA_LOAD context without a tenant_id, so a load that got gated would be refused."""
    return make_hook_context(
        hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=identity, data_access_format=data_format
    )


def _load(
    extender: Callable[..., Any], identity: str | None, data_format: str | None = None, *, args: tuple[Any, ...] = ()
) -> Any:
    """Run one wrapped load, passing args to the wrapped call; call it from inside a calculate call."""
    with _load_context(identity, data_format).activate():
        return extender(lambda *_: "loaded", *args)


def _calculate(extender: Callable[..., Any], body: Callable[[], Any], feature_group_class: str = _OUTER_CLASS) -> Any:
    """Run body as one wrapped calculate call under a present tenant_id."""
    with make_hook_context(feature_group_class=feature_group_class, tenant_id="tenant-1").activate():
        return extender(body)


def _record_for_loads(loads: list[tuple[str | None, str | None]], args: tuple[Any, ...] = ()) -> dict[str, Any]:
    """The record of one calculate call that ran the given (identity, format) loads, each called with args."""
    sink = InMemoryAuditSink()
    extender = AuditExtender(sink=sink)

    def body() -> None:
        for identity, data_format in loads:
            _load(extender, identity, data_format, args=args)

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


# (raw identity, sanitized identity, secret markers that must not survive anywhere in the record)
_URI_SANITIZER_CASES = [
    pytest.param(_BUCKET_KEY, _BUCKET_KEY, (), id="clean_s3_unchanged"),
    pytest.param(
        "https://user:pw@host/p/a?sig=SECRET#frag", "https://host/p/a", ("SECRET", "pw"), id="userinfo_query_fragment"
    ),
    pytest.param("postgresql://user:pw@db:5432/mydb", "postgresql://db:5432/mydb", ("pw",), id="userinfo_with_port"),
    pytest.param(
        "jdbc:postgresql://h/db?password=SECRET", "jdbc:postgresql://h/db", ("SECRET",), id="compound_scheme_query"
    ),
    pytest.param(
        "https://b.com&token=SECRET",
        "https://b.com",
        ("SECRET", "token"),
        id="ampersand_tail_in_authority",
    ),
    pytest.param("https://[::1]:8080/x?k=v", "https://[::1]:8080/x", ("k=v",), id="ipv6_host_with_port"),
    pytest.param("file:///tmp/data.csv", "file:///tmp/data.csv", (), id="file_uri_empty_authority_unchanged"),
    pytest.param("https://host?x=1", "https://host", ("x=1",), id="query_directly_after_host"),
    pytest.param("https://host/p?", "https://host/p", (), id="empty_query"),
    pytest.param("https://u:p@ss@host/db", "https://host/db", ("p@ss",), id="synthetic_at_sign_inside_userinfo"),
    pytest.param(
        "jdbc:hive2://h:10000/default;user=u;password=SECRET",
        "jdbc:hive2://h:10000/default",
        ("SECRET",),
        id="jdbc_semicolon_params_cut_from_path",
    ),
    pytest.param(
        "jdbc:sqlserver://h:1433;password=SECRET",
        "jdbc:sqlserver://h:1433",
        ("SECRET",),
        id="jdbc_semicolon_params_cut_from_authority",
    ),
    pytest.param("my_scheme://host?tok=SECRET", "my_scheme://host", ("SECRET",), id="underscore_in_scheme"),
    pytest.param(
        "https://b.com/x&sig=SECRET",
        "https://b.com/x",
        ("SECRET", "sig"),
        id="ampersand_tail_in_path",
    ),
    pytest.param(
        "mongodb://h1:27017,h2:27017/db?replicaSet=rs",
        "mongodb://h1:27017,h2:27017/db",
        ("replicaSet",),
        id="multi_host_authority_stays_intact",
    ),
    pytest.param("s3://münchen-bucket/key.parquet", "s3://münchen-bucket/key.parquet", (), id="non_ascii_host_kept"),
    pytest.param(
        "s3://bucket/t/dt=2026-09-20/part.parquet",
        "s3://bucket/t/dt=2026-09-20/part.parquet",
        (),
        id="hive_style_equals_in_path_kept",
    ),
    pytest.param(
        "postgresql://user:pa/ss@host/db", "postgresql://host/db", ("pa/ss",), id="synthetic_slash_in_password"
    ),
    pytest.param("postgresql://u:p&q@host/db", "postgresql://host/db", ("p&q",), id="synthetic_ampersand_in_password"),
    pytest.param(
        "https://host=SECRET/p/a", "https://host", ("SECRET",), id="equals_in_authority_cuts_it_and_drops_the_path"
    ),
    pytest.param(
        "https://host SECRET/p/a", "https://host", ("SECRET",), id="whitespace_in_authority_cuts_it_and_drops_the_path"
    ),
]

# (raw data_access, core's default-deny identity on the load context, expected recorded identity, secret markers absent from the record)
_RAW_FIRST_CASES = [
    pytest.param(
        "https://host/p?email=a@b.com/x&sig=SECRET",
        "str",
        "https://host/p",
        ("SECRET", "a@b.com", "b.com"),
        id="at_sign_in_query_value",
    ),
    pytest.param(
        "https://host/dl?url=https://user@o.example/f&token=SECRET",
        "str",
        "https://host/dl",
        ("SECRET", "o.example"),
        id="uri_in_query_value",
    ),
    pytest.param(
        "postgresql://host/db?user=u&password=p@ss/word",
        "str",
        "postgresql://host/db",
        ("p@ss", "ss/word", "password"),
        id="password_with_at_sign_and_slash_in_query",
    ),
    pytest.param(
        "/data/dir?/file#1.csv",
        "str",
        "/data/dir?/file#1.csv",
        (),
        id="non_uri_str_recorded_as_given",
    ),
    pytest.param(
        "jdbc:postgresql://h/db",
        "jdbc:postgresql://h",
        "jdbc:postgresql://h/db",
        (),
        id="clean_uri_with_differing_core_projection_uses_sanitized_raw",
    ),
    pytest.param(
        "s3://bucket/my file.parquet",
        "str",
        "s3://bucket/my file.parquet",
        (),
        id="clean_uri_with_space_and_core_type_name_uses_sanitized_raw",
    ),
    pytest.param(
        "Server=db;Password=SECRET",
        "str",
        "Server=db;Password=SECRET",
        (),
        id="keyword_dsn_raw_arg_recorded_as_given",
    ),
]

_NON_URI_IDENTITIES = [
    pytest.param("/data/dir?/file#1.csv", id="path_with_query_and_fragment_chars"),
    pytest.param("{host, port}", id="core_dict_form"),
    pytest.param("host=h user=u password=SECRET", id="keyword_dsn_passes_through"),
    pytest.param("str", id="core_default_deny_type_name"),
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


class TestAuditExtenderFailClosed:
    """fail_closed also fires on FEATURE_GROUP_MATCHED, and the refusal record is written unguarded first."""

    def test_matched_hook_with_missing_identity_refuses_and_records_the_hook(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=True)
        call = _CountingCall()
        context = make_hook_context(
            hook=ExtenderHook.FEATURE_GROUP_MATCHED, feature_group_class="", compute_framework_name=""
        )

        with context.activate():
            with pytest.raises(IdentityRequiredError):
                extender(call)

        assert call.calls == 0
        assert len(sink.records) == 1
        record = sink.records[0]
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
            feature_group_class="",
            compute_framework_name="",
            tenant_id="tenant-1",
        )

        with context.activate():
            assert extender(call) == 42

        assert call.calls == 1
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

        with make_hook_context(hook=hook).activate():
            with pytest.raises(OSError, match="disk full") as excinfo:
                CompositeExtender([extender])(call)

        assert call.calls == 0
        assert isinstance(excinfo.value.__context__, IdentityRequiredError)


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
    """on_run_complete(run_id) auto-seals a finished run when audit_path/manifest_path/signer are configured;
    otherwise (or with run_id=None) it is a no-op. Direct construction and direct on_run_complete calls, no
    mloda.run_all: the seal machinery itself is exercised end-to-end in test_run_manifest.py."""

    def test_run_id_none_is_a_noop_even_when_sealing_is_configured(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        sink = NdjsonAuditSink(audit_path)
        _append_records(audit_path, [_minimal_audit_record("run-1")])
        extender = AuditExtender(sink=sink, audit_path=audit_path, manifest_path=manifest_path, signer=_hmac_signer())

        extender.on_run_complete(None)

        assert not manifest_path.exists()

    def test_closes_the_sink_before_sealing_so_a_buffered_record_is_included(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        sink = BufferingNdjsonAuditSink(audit_path)
        sink.write(_minimal_audit_record("run-1"))
        assert not audit_path.exists()  # buffered only, proving close() (not something else) puts it on disk
        extender = AuditExtender(sink=sink, audit_path=audit_path, manifest_path=manifest_path, signer=_hmac_signer())

        extender.on_run_complete("run-1")

        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
        assert manifest["run_id"] == "run-1"
        assert manifest["record_count"] == 1

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
            extender.on_run_complete("run-never-wrote-anything")  # must not raise

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
            extender.on_run_complete("run-missing")  # must not raise

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("run-missing" in r.getMessage() and str(audit_path) in r.getMessage() for r in warnings)
        manifests = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        assert "run-missing" not in {m.get("run_id") for m in manifests}

    def test_second_call_for_an_already_sealed_run_with_no_new_records_logs_error_and_does_not_raise(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        _append_records(audit_path, [_minimal_audit_record("run-1")])
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path), audit_path=audit_path, manifest_path=manifest_path, signer=_hmac_signer()
        )
        extender.on_run_complete("run-1")
        before = manifest_path.read_text(encoding="utf-8")

        with caplog.at_level(logging.ERROR):
            extender.on_run_complete("run-1")  # must not raise

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("run-1" in r.getMessage() and str(manifest_path) in r.getMessage() for r in errors)
        assert manifest_path.read_text(encoding="utf-8") == before

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
        extender.on_run_complete("run-1")
        # Stands for a record written by a writer outside this extender, since this extender now refuses run-1.
        _append_records(audit_path, [_minimal_audit_record("run-1")])

        with caplog.at_level(logging.ERROR):
            extender.on_run_complete("run-1")  # must not raise

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("run-1" in r.getMessage() and str(manifest_path) in r.getMessage() for r in errors)
        manifests = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        assert len(manifests) == 1
        assert manifests[0]["record_count"] == 1
        with pytest.raises(ManifestVerificationError, match="beyond its seal"):
            verify_ndjson_log_coverage(audit_path, manifest_path, signer=signer)

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
            sink=NdjsonAuditSink(audit_path), audit_path=audit_path, manifest_path=manifest_path, signer=_hmac_signer()
        )

        with pytest.raises(ManifestVerificationError):
            extender.on_run_complete("run-2")

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

        copy.on_run_complete("run-1")  # must not raise, and must not seal: the copy has no signer

        assert not manifest_path.exists()

        extender.on_run_complete("run-1")  # the original still seals

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
            copy.on_run_complete("run-1")  # must not raise, and must not seal: the copy has no signer

        assert not manifest_path.exists()
        matching = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and re.search(r"pickl|copi", r.getMessage(), re.IGNORECASE)
        ]
        assert len(matching) == 1, [(r.levelno, r.getMessage()) for r in caplog.records]

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            copy.on_run_complete("run-1")  # same copy again: must not warn a second time

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
            extender.on_run_complete("run-1")

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

    @_BOTH_POSTURES
    @pytest.mark.parametrize("tenant_id", [_TENANT, None], ids=["identity_present", "identity_missing"])
    def test_call_under_the_sealed_run_id_raises_run_already_sealed_error_before_the_fail_closed_gate(
        self, tmp_path: Path, fail_closed: bool, tenant_id: str | None
    ) -> None:
        extender, audit_path = _extender_with_run_1_sealed(tmp_path, fail_closed=fail_closed)
        before = audit_path.read_bytes()
        call = _CountingCall()

        with make_hook_context(run_id="run-1", tenant_id=tenant_id).activate():
            with pytest.raises(RunAlreadySealedError) as excinfo:
                extender(call)

        assert "run-1" in str(excinfo.value)
        assert "new session" in str(excinfo.value)
        assert call.calls == 0
        assert audit_path.read_bytes() == before  # no deny record either, in either posture

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

        extender.on_run_complete("run-1")  # nothing was sealed: missing file or no records for run-1

        call = _CountingCall()
        with make_hook_context(run_id="run-1", tenant_id=_TENANT).activate():
            result = extender(call)

        assert result == 42
        assert call.calls == 1

    def test_call_under_a_run_sealed_elsewhere_is_refused(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.ndjson"
        manifest_path = tmp_path / "manifest.ndjson"
        _append_records(audit_path, [_minimal_audit_record("run-1")])
        signer = _hmac_signer()
        seal_ndjson_runs(audit_path, manifest_path, signer=signer, run_id="run-1")
        extender = AuditExtender(
            sink=NdjsonAuditSink(audit_path), audit_path=audit_path, manifest_path=manifest_path, signer=signer
        )

        extender.on_run_complete("run-1")  # hits the RunAlreadySealedError branch, logged at ERROR

        before = audit_path.read_bytes()
        call = _CountingCall()
        with make_hook_context(run_id="run-1", tenant_id=_TENANT).activate():
            with pytest.raises(RunAlreadySealedError):
                extender(call)

        assert call.calls == 0
        assert audit_path.read_bytes() == before

    def test_pickled_copy_made_after_sealing_still_refuses_a_call_under_the_sealed_run_id(self, tmp_path: Path) -> None:
        extender, audit_path = _extender_with_run_1_sealed(tmp_path)
        copy = pickle.loads(pickle.dumps(extender))  # nosec

        before = audit_path.read_bytes()
        call = _CountingCall()
        with make_hook_context(run_id="run-1", tenant_id=_TENANT).activate():
            with pytest.raises(RunAlreadySealedError):
                copy(call)

        assert call.calls == 0
        assert audit_path.read_bytes() == before

    def test_extender_without_sealing_config_never_refuses(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        extender.on_run_complete("run-1")  # a no-op: no sealing config

        call = _CountingCall()
        with make_hook_context(run_id="run-1", tenant_id=_TENANT).activate():
            result = extender(call)

        assert result == 42
        assert call.calls == 1
        assert len(sink.records) == 1

    @_FAIL_CLOSED_RAISE_ON_ERROR_POSTURES
    def test_sealed_run_refusal_through_composite_extender_propagates_and_the_call_never_runs(
        self, make_gate_extender: Callable[..., AuditExtender], tmp_path: Path
    ) -> None:
        extender, audit_path = _extender_with_run_1_sealed(tmp_path, make=make_gate_extender)
        before = audit_path.read_bytes()
        call = _CountingCall()

        with make_hook_context(run_id="run-1", tenant_id=_TENANT).activate():
            with pytest.raises(RunAlreadySealedError):
                CompositeExtender([extender])(call)

        assert call.calls == 0
        assert audit_path.read_bytes() == before

    def test_sealed_run_refusal_with_fail_closed_false_and_raise_on_error_false_runs_once_via_warning_only_fallback(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        extender, audit_path = _extender_with_run_1_sealed(tmp_path, raise_on_error=False)
        before = audit_path.read_bytes()
        call = _CountingCall()

        with make_hook_context(run_id="run-1", tenant_id=_TENANT).activate():
            with caplog.at_level(logging.WARNING):
                result = CompositeExtender([extender])(call)

        assert result == 42
        assert call.calls == 1
        assert audit_path.read_bytes() == before
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("run-1" in r.getMessage() for r in warnings)


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

        with make_hook_context(hook=hook).activate():
            with pytest.raises(IdentityRequiredError):
                extender(_CountingCall())

        assert sink.records[0]["hook"] == hook.name
        assert sink.records[0]["policy_version"] == _POLICY_VERSION


class TestAuditExtenderDataAccess:
    """A load nested in a calculate call is recorded on that call's record, sanitized, never on a record of its own."""

    def test_calculate_without_a_load_records_empty_lists(self) -> None:
        record = _record_for_loads([])

        assert record["data_access_identity"] == []
        assert record["data_access_format"] == []

    def test_refusal_record_carries_empty_data_access_lists(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink, fail_closed=True)

        with make_hook_context().activate():
            with pytest.raises(IdentityRequiredError):
                extender(_CountingCall())

        assert sink.records[0]["data_access_identity"] == []
        assert sink.records[0]["data_access_format"] == []

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
        assert record["record_version"] == 1
        assert record["data_access_identity"] == [_BUCKET_KEY]
        assert record["data_access_format"] == ["ParquetReader"]

    def test_format_none_stays_none_inside_the_list(self) -> None:
        record = _record_for_loads([(_BUCKET_KEY, None)])

        assert record["data_access_identity"] == [_BUCKET_KEY]
        assert record["data_access_format"] == [None]

    def test_load_without_an_identity_is_skipped(self) -> None:
        record = _record_for_loads([(None, "CsvReader")])

        assert record["data_access_identity"] == []
        assert record["data_access_format"] == []

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
            [(_BUCKET_KEY, "ParquetReader"), (other, "CsvReader"), (_BUCKET_KEY, "ParquetReader")]
        )

        assert record["data_access_identity"] == [_BUCKET_KEY, other]
        assert record["data_access_format"] == ["ParquetReader", "CsvReader"]

    def test_same_identity_under_two_formats_gives_two_entries(self) -> None:
        record = _record_for_loads([(_BUCKET_KEY, "CsvReader"), (_BUCKET_KEY, "ParquetReader")])

        assert record["data_access_identity"] == [_BUCKET_KEY, _BUCKET_KEY]
        assert record["data_access_format"] == ["CsvReader", "ParquetReader"]

    def test_loads_that_sanitize_to_the_same_identity_are_deduplicated(self) -> None:
        record = _record_for_loads(
            [("https://a:1@host/x?sig=one", "CsvReader"), ("https://b:2@host/x?sig=two", "CsvReader")]
        )

        assert record["data_access_identity"] == ["https://host/x"]
        assert record["data_access_format"] == ["CsvReader"]

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

    @pytest.mark.parametrize(("raw", "expected", "secrets"), _URI_SANITIZER_CASES)
    def test_uri_identity_is_sanitized_before_it_is_stored(
        self, raw: str, expected: str, secrets: tuple[str, ...]
    ) -> None:
        record = _record_for_loads([(raw, None)])

        assert record["data_access_identity"] == [expected]
        for secret in secrets:
            assert secret not in json.dumps(record)

    @pytest.mark.parametrize("raw", _NON_URI_IDENTITIES)
    def test_non_uri_identity_passes_through_unchanged(self, raw: str) -> None:
        record = _record_for_loads([(raw, None)])

        assert record["data_access_identity"] == [raw]

    @pytest.mark.parametrize(("raw", "context_identity", "expected", "secrets"), _RAW_FIRST_CASES)
    def test_a_str_first_argument_is_recorded_sanitized_unless_its_scheme_is_malformed(
        self, raw: str, context_identity: str, expected: str, secrets: tuple[str, ...]
    ) -> None:
        record = _record_for_loads([(context_identity, "CsvReader")], args=(raw, _FEATURES_PLACEHOLDER))

        assert record["data_access_identity"] == [expected]
        assert record["data_access_format"] == ["CsvReader"]
        for secret in secrets:
            assert secret not in json.dumps(record)

    def test_a_non_str_first_argument_falls_back_to_the_context_identity(self) -> None:
        connection_params = {"host": "h", "port": 5432, "api_key": "SECRET"}

        record = _record_for_loads([("{host, port}", None)], args=(connection_params, _FEATURES_PLACEHOLDER))

        assert record["data_access_identity"] == ["{host, port}"]
        assert "SECRET" not in json.dumps(record)

    def test_a_load_without_a_context_identity_is_skipped_even_with_a_uri_first_argument(self) -> None:
        record = _record_for_loads([(None, "CsvReader")], args=("https://host/p?sig=SECRET", _FEATURES_PLACEHOLDER))

        assert record["data_access_identity"] == []
        assert record["data_access_format"] == []


class TestNdjsonAuditSink:
    """NdjsonAuditSink appends one JSON line per record, surviving a pickle round trip."""

    def test_appends_one_json_line_per_write(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.ndjson"
        sink = NdjsonAuditSink(path)

        sink.write({"a": 1})
        sink.write({"a": 2})

        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == [{"a": 1}, {"a": 2}]

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
        assert records[0]["data_access_identity"] == [str(tmp_path / "data.csv")]
        assert records[0]["data_access_format"] == ["CsvReader"]

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
    def test_run_all_fail_closed_refuses_at_calculate_when_identity_is_gone_by_run_time(
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
            session = mloda.prepare(
                ["value_int"],
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator}),
                function_extender={gate, counting},
                parallelization_modes={mode},
            )

        with pytest.raises(IdentityRequiredError):
            session.run(parallelization_modes={mode}, flight_server=flight_server)

        records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 1
        record = records[0]
        assert record["hook"] == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE.name
        assert record["decision"] == "deny"
        assert record["status"] == "error"
        assert counting.calls == 0
        # Marker file covers MULTIPROCESSING: a worker's own `calls` copy would be invisible here.
        assert not marker_path.exists()
