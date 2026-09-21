"""Tests for OtelLogAuditSink. They patch the provider resolver instead of setting a global LoggerProvider."""

from __future__ import annotations

import calendar
import hashlib
import hmac
import importlib
import io
import json
import logging
import pickle  # nosec
import re
from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("opentelemetry.sdk")

from mloda.steward import ExtenderHook, verified_context
from opentelemetry import _logs
from opentelemetry._logs import NoOpLoggerProvider, SeverityNumber
from opentelemetry.sdk._logs import Logger as SdkLogger
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor

try:
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
except ImportError:  # SDKs that predate the rename only ship InMemoryLogExporter
    from opentelemetry.sdk._logs.export import InMemoryLogExporter as InMemoryLogRecordExporter

try:
    from opentelemetry.sdk._logs.export import ConsoleLogRecordExporter
except ImportError:  # SDKs that predate the rename only ship ConsoleLogExporter
    from opentelemetry.sdk._logs.export import ConsoleLogExporter as ConsoleLogRecordExporter

import mloda.enterprise.extenders.audit as audit_package
from mloda.enterprise.extenders.audit import (
    AuditExtender,
    IdentityRequiredError,
    NdjsonAuditSink,
    OtelLogAuditSink,
    TeeAuditSink,
)
from mloda.enterprise.extenders.audit import otel_log_sink as otel_log_sink_module
from mloda.enterprise.extenders.audit import run_manifest as run_manifest_module
from mloda.enterprise.extenders.audit.tests.test_audit_extender import (
    _IDENTITY_REQUIRED_ERROR_TYPE,
    _MISSING_IDENTITY_CASES,
    _POLICY_VERSION,
    _PRINCIPAL,
    _PROJECT,
    _TENANT,
    InMemoryAuditSink,
)
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.runners import expected_value_int, run_value_int
from mloda.testing.import_isolation import block_root, evict_package

_GET_LOGGER_PROVIDER = "opentelemetry._logs.get_logger_provider"

_SCOPE_NAME = "mloda_enterprise_audit"

_ALL_IDENTITY = ("tenant_id", "project_id", "principal")

_OTEL_EXTRA = re.escape("mloda-enterprise[otel]")

_STR_ATTRIBUTES = {
    "deny_reason": "mloda.audit.deny_reason",
    "policy_version": "mloda.audit.policy_version",
    "hook": "mloda.audit.hook",
    "run_id": "mloda.run.id",
    "tenant_id": "mloda.tenant.id",
    "project_id": "mloda.project.id",
    "feature_group_class": "mloda.feature_group.name",
    "error_type": "error.type",
}

_BLANK_OMITTED_ATTRIBUTES = {**_STR_ATTRIBUTES, "principal": "user.hash"}

_ALLOWED_ATTRIBUTES = {*_STR_ATTRIBUTES.values(), "mloda.audit.decision", "user.hash", "mloda.feature.names"}

_SEVERITY_BY_DECISION = {"allow": SeverityNumber.INFO, "deny": SeverityNumber.WARN}

_EVENT_TIME_CASES = [
    pytest.param("2026-09-21T10:11:12.123456Z", 1789985472123456000, id="microseconds"),
    pytest.param("2026-09-21T10:11:12Z", 1789985472000000000, id="no_fraction"),
    pytest.param("2026-09-21T10:11:12.000042Z", 1789985472000042000, id="leading_zero_microseconds"),
    pytest.param("2026-09-21T23:59:59.999999Z", 1790035199999999000, id="last_microsecond"),
]

# "abc" is the published SHA-256 test vector.
_PRINCIPAL_HASH_CASES = [
    pytest.param("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", id="known_vector"),
    pytest.param(_PRINCIPAL, hashlib.sha256(_PRINCIPAL.encode("utf-8")).hexdigest(), id="ascii"),
    pytest.param(
        "prïncipal-é",
        "12e76f92790454204ea2601beb3c2dabdf4d68a77a5f85cbc0ed80094d321b8e",
        id="non_ascii_utf8",
    ),
]

_KEY = b"k" * 32
_OTHER_KEY = b"o" * 32
# Distinctive ASCII so a leak of the key, as text or as a bytes repr, is findable.
_LEAK_KEY = b"key-marker-7f3a-0123456789abcdef-xyz"

_RFC_4231_CASE_6_KEY = b"\xaa" * 131
_RFC_4231_CASE_6_MESSAGE = "Test Using Larger Than Block-Size Key - Hash Key First"
_RFC_4231_CASE_6_DIGEST = "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"


def _hmac_sha256(key: bytes, value: str) -> str:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()


_KEYED_PRINCIPAL_HASH_CASES = [
    pytest.param(
        _RFC_4231_CASE_6_KEY, _RFC_4231_CASE_6_MESSAGE, _RFC_4231_CASE_6_DIGEST, id="rfc_4231_case_6_long_key"
    ),
    pytest.param(_KEY, _PRINCIPAL, _hmac_sha256(_KEY, _PRINCIPAL), id="ascii_min_length_key"),
    pytest.param(_OTHER_KEY, _PRINCIPAL, _hmac_sha256(_OTHER_KEY, _PRINCIPAL), id="ascii_other_key"),
    pytest.param(_KEY, "prïncipal-é", _hmac_sha256(_KEY, "prïncipal-é"), id="non_ascii_utf8"),
]

_INVALID_KEYS = [
    pytest.param("k" * 32, id="str_key"),
    pytest.param(b"", id="empty_bytes"),
    pytest.param(b"k" * 31, id="31_bytes"),
    pytest.param(bytearray(b"k" * 32), id="bytearray"),
    pytest.param(memoryview(b"k" * 32), id="memoryview"),
    pytest.param(12345, id="int"),
    pytest.param(["k"] * 32, id="list"),
]

_EVENT_TIME = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?Z")

_UNFORWARDED = {
    "extra_marker": "extra-marker-9c1d",
    "data_access_identity": ["s3://bucket-marker-9c1d/key"],
    "data_access_format": ["format-marker-9c1d"],
    "input_features": ["input-marker-9c1d"],
    "rows_out": 987654321,
    "duration_seconds": 12.3456789,
    "status": "status-marker-9c1d",
    "compliant": True,
    "feature_group_version": "fgv-marker-9c1d",
    "plugin_version": "pv-marker-9c1d",
    "compute_framework_name": "cfw-marker-9c1d",
}


def make_log_capture() -> tuple[LoggerProvider, InMemoryLogRecordExporter]:
    exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call,unused-ignore]
    provider = LoggerProvider(shutdown_on_exit=False)
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    return provider, exporter


def make_console_capture() -> tuple[LoggerProvider, io.StringIO]:
    out = io.StringIO()
    provider = LoggerProvider(shutdown_on_exit=False)
    provider.add_log_record_processor(SimpleLogRecordProcessor(ConsoleLogRecordExporter(out=out)))
    return provider, out


@pytest.fixture
def log_exporter() -> Iterator[InMemoryLogRecordExporter]:
    """An in-memory exporter behind the provider the patched resolver hands out; no global provider is set."""
    provider, exporter = make_log_capture()
    with patch(_GET_LOGGER_PROVIDER, return_value=provider):
        yield exporter


@pytest.fixture(autouse=True)
def _reset_no_sdk_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(otel_log_sink_module, "_no_sdk_warned", False)


def _audit_record(*, fail_closed: bool = False, **context: Any) -> dict[str, Any]:
    """The record a real AuditExtender builds for one call; project_id and principal default to present."""
    memory = InMemoryAuditSink()
    extender = AuditExtender(
        sink=memory, required_identity=_ALL_IDENTITY, fail_closed=fail_closed, policy_version=_POLICY_VERSION
    )
    with make_hook_context(**{"project_id": _PROJECT, "principal": _PRINCIPAL, **context}).activate():
        with suppress(IdentityRequiredError):
            extender(lambda: None)
    return memory.records[0]


def _single_log(exporter: InMemoryLogRecordExporter) -> Any:
    logs = exporter.get_finished_logs()
    assert len(logs) == 1, logs
    return logs[0]


def _make_sink(key: bytes | None) -> OtelLogAuditSink:
    """No key means the constructor default, so the default-path tests do not depend on the new keyword."""
    return OtelLogAuditSink() if key is None else OtelLogAuditSink(user_hash_key=key)


def _write_one(exporter: InMemoryLogRecordExporter, record: dict[str, Any], key: bytes | None = None) -> Any:
    _make_sink(key).write(record)
    return _single_log(exporter)


def _attributes(log: Any) -> dict[str, Any]:
    attributes = log.log_record.attributes
    assert attributes is not None
    return dict(attributes)


def _everything(log: Any) -> str:
    """The body, severity text and every attribute name and value of a log, as one string to search."""
    record = log.log_record
    parts = [record.body, record.severity_text, getattr(record, "event_name", None), dict(record.attributes or {})]
    return repr(parts)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _epoch_ns(event_time: str) -> int:
    """Nanoseconds since the epoch, exact to the microsecond, by integer arithmetic only."""
    match = _EVENT_TIME.fullmatch(event_time)
    assert match, event_time
    year, month, day, hour, minute, second = (int(part) for part in match.groups()[:6])
    microseconds = int((match.group(7) or "").ljust(6, "0"))
    return (calendar.timegm((year, month, day, hour, minute, second)) * 1_000_000 + microseconds) * 1000


def _module_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == otel_log_sink_module.__name__ and r.levelno == logging.WARNING
    ]


def _emit_through_console(*records: dict[str, Any], key: bytes | None = None) -> tuple[str, list[dict[str, Any]]]:
    """The console exporter's raw text and the JSON object it printed per log (pretty-printed, so decoded in turn)."""
    provider, out = make_console_capture()
    with patch(_GET_LOGGER_PROVIDER, return_value=provider):
        for record in records:
            _make_sink(key).write(record)
    text = out.getvalue()
    decoder = json.JSONDecoder()
    logs: list[dict[str, Any]] = []
    rest = text.strip()
    while rest:
        log, end = decoder.raw_decode(rest)
        logs.append(log)
        rest = rest[end:].lstrip()
    return text, logs


def _read_ndjson(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_channels_agree(records: list[dict[str, Any]], logs: tuple[Any, ...]) -> None:
    """The NDJSON records and the emitted logs are the same records, in the same order."""
    assert records
    assert len(logs) == len(records)
    for record, log in zip(records, logs, strict=True):
        attributes = _attributes(log)
        assert log.log_record.body == record["decision"]
        assert log.log_record.severity_number == _SEVERITY_BY_DECISION[record["decision"]]
        assert attributes["mloda.audit.decision"] == record["decision"]
        assert attributes.get("mloda.run.id") == record["run_id"]
        assert attributes["mloda.audit.policy_version"] == record["policy_version"]
        assert attributes["mloda.audit.hook"] == record["hook"]
        assert attributes.get("mloda.feature_group.name") == (record["feature_group_class"] or None)
        assert log.log_record.timestamp == _epoch_ns(record["event_time"])


def _audit_package_without_opentelemetry(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The audit package, cold-imported while `opentelemetry` cannot be imported."""
    block_root(monkeypatch, "opentelemetry")
    evict_package(monkeypatch, "mloda.enterprise.extenders.audit")
    return importlib.import_module("mloda.enterprise.extenders.audit")


_NO_SDK_PROVIDERS = [
    pytest.param(NoOpLoggerProvider, id="noop"),
    # What the API hands out when nothing was configured: a proxy, not a NoOpLoggerProvider.
    pytest.param(_logs.get_logger_provider, id="api_default"),
]


class TestOtelLogAuditSinkExport:
    """The sink lives in its own module and is exported from the package."""

    def test_is_defined_in_the_otel_log_sink_module_and_exported(self) -> None:
        assert OtelLogAuditSink is otel_log_sink_module.OtelLogAuditSink
        assert "OtelLogAuditSink" in audit_package.__all__


class TestOtelLogAuditSinkWithoutOpentelemetry:
    """opentelemetry is an optional extra: only building an OtelLogAuditSink needs it."""

    def test_the_package_imports_and_the_other_sinks_still_work(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        package = _audit_package_without_opentelemetry(monkeypatch)
        memory = InMemoryAuditSink()
        audit_path = tmp_path / "audit.ndjson"

        for sink in (memory, package.NdjsonAuditSink(audit_path)):
            extender = package.AuditExtender(sink=sink)
            with make_hook_context(tenant_id="tenant-1").activate():
                assert extender(lambda a, b: a + b, 3, 4) == 7

        assert len(memory.records) == 1
        assert len(_read_ndjson(audit_path)) == 1
        assert "OtelLogAuditSink" in package.__all__

    def test_building_the_sink_raises_import_error_naming_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        package = _audit_package_without_opentelemetry(monkeypatch)

        with pytest.raises(ImportError, match=_OTEL_EXTRA):
            package.OtelLogAuditSink()

    @pytest.mark.parametrize("key", ["short", b"k" * 31, b"k" * 32], ids=["str", "short", "valid"])
    def test_the_import_error_wins_over_a_bad_or_valid_key(self, monkeypatch: pytest.MonkeyPatch, key: Any) -> None:
        package = _audit_package_without_opentelemetry(monkeypatch)

        with pytest.raises(ImportError, match=_OTEL_EXTRA):
            package.OtelLogAuditSink(user_hash_key=key)


class TestOtelLogAuditSinkProvider:
    """The provider is resolved through the public API on every write; the scope is mloda_enterprise_audit."""

    def test_the_log_carries_the_audit_instrumentation_scope(self, log_exporter: InMemoryLogRecordExporter) -> None:
        log = _write_one(log_exporter, _audit_record(tenant_id="tenant-1"))

        assert log.instrumentation_scope.name == _SCOPE_NAME

    def test_every_write_resolves_the_provider_at_call_time(self) -> None:
        sink = OtelLogAuditSink()
        first_provider, first_exporter = make_log_capture()
        second_provider, second_exporter = make_log_capture()
        record = _audit_record(tenant_id="tenant-1")

        with patch(_GET_LOGGER_PROVIDER, return_value=first_provider):
            sink.write(record)
        with patch(_GET_LOGGER_PROVIDER, return_value=second_provider):
            sink.write(record)

        assert len(first_exporter.get_finished_logs()) == 1
        assert len(second_exporter.get_finished_logs()) == 1

    @pytest.mark.parametrize(
        "make_provider", [lambda: make_log_capture()[0], NoOpLoggerProvider], ids=["sdk", "no_sdk"]
    )
    def test_write_never_sets_a_global_logger_provider(self, make_provider: Callable[[], Any]) -> None:
        with (
            patch(_GET_LOGGER_PROVIDER, return_value=make_provider()),
            patch("opentelemetry._logs.set_logger_provider") as public_setter,
            patch("opentelemetry._logs._internal.set_logger_provider") as internal_setter,
        ):
            OtelLogAuditSink().write(_audit_record(tenant_id="tenant-1"))

        public_setter.assert_not_called()
        internal_setter.assert_not_called()


class TestOtelLogAuditSinkMapping:
    """One log per audit record: body, severity, timestamp and an allowlist of attributes."""

    @pytest.mark.parametrize(
        ("tenant_id", "decision", "severity_text"),
        [("tenant-1", "allow", "INFO"), (None, "deny", "WARN")],
        ids=["allow", "deny"],
    )
    def test_body_is_the_decision_and_severity_follows_it(
        self,
        log_exporter: InMemoryLogRecordExporter,
        tenant_id: str | None,
        decision: str,
        severity_text: str,
    ) -> None:
        log = _write_one(log_exporter, _audit_record(tenant_id=tenant_id))

        assert log.log_record.body == decision
        assert log.log_record.severity_number == _SEVERITY_BY_DECISION[decision]
        assert log.log_record.severity_text == severity_text

    @pytest.mark.parametrize(("event_time", "expected_ns"), _EVENT_TIME_CASES)
    def test_timestamp_is_the_event_time_in_epoch_nanoseconds(
        self, log_exporter: InMemoryLogRecordExporter, event_time: str, expected_ns: int
    ) -> None:
        record = {**_audit_record(tenant_id="tenant-1"), "event_time": event_time}

        log = _write_one(log_exporter, record)

        assert isinstance(log.log_record.timestamp, int)
        assert log.log_record.timestamp == expected_ns

    def test_a_record_without_event_time_still_emits(self, log_exporter: InMemoryLogRecordExporter) -> None:
        record = _audit_record(tenant_id="tenant-1")
        del record["event_time"]

        log = _write_one(log_exporter, record)

        assert log.log_record.body == "allow"

    def test_an_allow_record_maps_to_exactly_the_allowlisted_attributes(
        self, log_exporter: InMemoryLogRecordExporter
    ) -> None:
        record = _audit_record(
            tenant_id="tenant-1",
            project_id="project-1",
            principal="svc-1",
            run_id="run-123",
            feature_group_class="my.module.MyFeatureGroup",
            feature_names=("value_int", "value_str"),
        )

        log = _write_one(log_exporter, record)

        assert _attributes(log) == {
            "mloda.audit.decision": "allow",
            "mloda.audit.policy_version": _POLICY_VERSION,
            "mloda.audit.hook": "FEATURE_GROUP_CALCULATE_FEATURE",
            "mloda.run.id": "run-123",
            "mloda.tenant.id": "tenant-1",
            "mloda.project.id": "project-1",
            "user.hash": _sha256("svc-1"),
            "mloda.feature_group.name": "my.module.MyFeatureGroup",
            # The SDK stores a sequence attribute as a tuple.
            "mloda.feature.names": ("value_int", "value_str"),
        }

    def test_a_refusal_record_adds_the_deny_reason_and_the_error_type(
        self, log_exporter: InMemoryLogRecordExporter
    ) -> None:
        record = _audit_record(
            fail_closed=True,
            hook=ExtenderHook.FEATURE_GROUP_MATCHED,
            tenant_id=None,
            project_id="project-1",
            principal="svc-1",
            run_id="run-123",
            feature_group_class="my.module.MyFeatureGroup",
        )

        log = _write_one(log_exporter, record)

        assert log.log_record.body == "deny"
        assert _attributes(log) == {
            "mloda.audit.decision": "deny",
            "mloda.audit.deny_reason": "missing_tenant_id",
            "mloda.audit.policy_version": _POLICY_VERSION,
            "mloda.audit.hook": "FEATURE_GROUP_MATCHED",
            "mloda.run.id": "run-123",
            "mloda.project.id": "project-1",
            "user.hash": _sha256("svc-1"),
            "mloda.feature_group.name": "my.module.MyFeatureGroup",
            "mloda.feature.names": ("value_int",),
            "error.type": _IDENTITY_REQUIRED_ERROR_TYPE,
        }

    @pytest.mark.parametrize(("principal", "expected_hash"), _PRINCIPAL_HASH_CASES)
    def test_the_principal_is_the_lowercase_sha256_hex_of_its_utf8_bytes(
        self, log_exporter: InMemoryLogRecordExporter, principal: str, expected_hash: str
    ) -> None:
        log = _write_one(log_exporter, _audit_record(tenant_id="tenant-1", principal=principal))

        attributes = _attributes(log)
        assert attributes["user.hash"] == expected_hash
        assert re.fullmatch(r"[0-9a-f]{64}", attributes["user.hash"])
        assert "principal" not in attributes

    @pytest.mark.parametrize(("key", "principal", "expected_hash"), _KEYED_PRINCIPAL_HASH_CASES)
    def test_a_keyed_principal_is_the_lowercase_hmac_sha256_hex_of_its_utf8_bytes(
        self, log_exporter: InMemoryLogRecordExporter, key: bytes, principal: str, expected_hash: str
    ) -> None:
        log = _write_one(log_exporter, _audit_record(tenant_id="tenant-1", principal=principal), key=key)

        attributes = _attributes(log)
        assert attributes["user.hash"] == expected_hash
        assert attributes["user.hash"] != _sha256(principal)
        assert re.fullmatch(r"[0-9a-f]{64}", attributes["user.hash"])
        assert "principal" not in attributes

    def test_the_key_changes_the_hash_and_the_same_key_is_stable(self, log_exporter: InMemoryLogRecordExporter) -> None:
        record = _audit_record(tenant_id="tenant-1")

        hashes = []
        for key in (None, _KEY, _KEY, _OTHER_KEY):
            log_exporter.clear()
            hashes.append(_attributes(_write_one(log_exporter, record, key=key))["user.hash"])

        assert hashes[1] == hashes[2]
        assert len({hashes[0], hashes[1], hashes[3]}) == 3

    def test_an_explicit_none_key_keeps_the_plain_sha256(self, log_exporter: InMemoryLogRecordExporter) -> None:
        log = _write_one(log_exporter, _audit_record(tenant_id="tenant-1"), key=None)

        assert _attributes(log)["user.hash"] == _sha256(_PRINCIPAL)

    def test_a_key_does_not_change_the_other_attributes(self, log_exporter: InMemoryLogRecordExporter) -> None:
        record = _audit_record(tenant_id="tenant-1", project_id="project-1", run_id="run-123")
        plain = _attributes(_write_one(log_exporter, record))
        log_exporter.clear()

        keyed = _attributes(_write_one(log_exporter, record, key=_KEY))

        assert {k: v for k, v in keyed.items() if k != "user.hash"} == {
            k: v for k, v in plain.items() if k != "user.hash"
        }

    @pytest.mark.parametrize("blank", [None, "", "   "], ids=["none", "empty", "whitespace"])
    def test_a_blank_principal_leaves_user_hash_out_with_a_key(
        self, log_exporter: InMemoryLogRecordExporter, blank: str | None
    ) -> None:
        log = _write_one(log_exporter, {"decision": "deny", "principal": blank}, key=_KEY)

        assert "user.hash" not in _attributes(log)

    def test_a_record_with_only_a_decision_emits_only_that_attribute(
        self, log_exporter: InMemoryLogRecordExporter
    ) -> None:
        log = _write_one(log_exporter, {"decision": "allow"})

        assert log.log_record.body == "allow"
        assert _attributes(log) == {"mloda.audit.decision": "allow"}

    @pytest.mark.parametrize(("key", "attribute"), sorted(_BLANK_OMITTED_ATTRIBUTES.items()))
    @pytest.mark.parametrize("blank", [None, "", "   "], ids=["none", "empty", "whitespace"])
    def test_a_none_or_blank_value_leaves_its_attribute_out(
        self, log_exporter: InMemoryLogRecordExporter, key: str, attribute: str, blank: str | None
    ) -> None:
        log = _write_one(log_exporter, {"decision": "deny", key: blank})

        assert attribute not in _attributes(log)

    @pytest.mark.parametrize("feature_names", [[], None], ids=["empty", "none"])
    def test_empty_feature_names_leave_the_attribute_out(
        self, log_exporter: InMemoryLogRecordExporter, feature_names: list[str] | None
    ) -> None:
        log = _write_one(log_exporter, {"decision": "allow", "feature_names": feature_names})

        assert "mloda.feature.names" not in _attributes(log)

    @pytest.mark.parametrize(("tenant_id", "project_id", "principal", "expected_reason"), _MISSING_IDENTITY_CASES)
    def test_a_missing_identity_is_omitted_and_the_rest_is_kept(
        self,
        log_exporter: InMemoryLogRecordExporter,
        tenant_id: str | None,
        project_id: str | None,
        principal: str | None,
        expected_reason: str,
    ) -> None:
        record = _audit_record(tenant_id=tenant_id, project_id=project_id, principal=principal)

        log = _write_one(log_exporter, record)

        attributes = _attributes(log)
        assert log.log_record.body == "deny"
        assert attributes["mloda.audit.deny_reason"] == expected_reason
        assert attributes["mloda.project.id"] == project_id
        assert attributes.get("mloda.tenant.id") == (tenant_id if tenant_id is not None and tenant_id.strip() else None)
        assert attributes.get("user.hash") == (
            _sha256(principal) if principal is not None and principal.strip() else None
        )
        assert all(value.strip() for value in attributes.values() if isinstance(value, str))

    def test_a_match_time_refusal_with_an_empty_feature_group_class_omits_the_name(
        self, log_exporter: InMemoryLogRecordExporter
    ) -> None:
        record = _audit_record(
            fail_closed=True,
            hook=ExtenderHook.FEATURE_GROUP_MATCHED,
            feature_group_class="",
            compute_framework_name="",
        )

        log = _write_one(log_exporter, record)

        attributes = _attributes(log)
        assert attributes["mloda.audit.hook"] == "FEATURE_GROUP_MATCHED"
        assert "mloda.feature_group.name" not in attributes

    def test_no_other_record_key_is_forwarded_and_the_raw_principal_appears_nowhere(
        self, log_exporter: InMemoryLogRecordExporter
    ) -> None:
        record = {
            **_audit_record(tenant_id=_TENANT, project_id=_PROJECT, principal=_PRINCIPAL, run_id="run-123"),
            **_UNFORWARDED,
        }

        log = _write_one(log_exporter, record)

        attributes = _attributes(log)
        assert set(attributes) <= _ALLOWED_ATTRIBUTES
        assert attributes["user.hash"] == _sha256(_PRINCIPAL)
        haystack = _everything(log)
        assert _PRINCIPAL not in haystack
        for name in [*_UNFORWARDED, "record_version", "event_time"]:
            assert name not in haystack
        for value in _UNFORWARDED.values():
            markers = value if isinstance(value, list) else [value]
            for marker in markers:
                assert str(marker) not in haystack

    def test_with_a_key_neither_the_raw_principal_nor_the_key_appears_anywhere(
        self, log_exporter: InMemoryLogRecordExporter
    ) -> None:
        record = {
            **_audit_record(tenant_id=_TENANT, project_id=_PROJECT, principal=_PRINCIPAL, run_id="run-123"),
            **_UNFORWARDED,
        }

        log = _write_one(log_exporter, record, key=_LEAK_KEY)

        attributes = _attributes(log)
        assert set(attributes) <= _ALLOWED_ATTRIBUTES
        assert attributes["user.hash"] == _hmac_sha256(_LEAK_KEY, _PRINCIPAL)
        haystack = _everything(log)
        assert _PRINCIPAL not in haystack
        assert _LEAK_KEY.decode("ascii") not in haystack
        assert repr(_LEAK_KEY) not in haystack


class TestOtelLogAuditSinkUserHashKey:
    """user_hash_key switches user.hash to a keyed HMAC-SHA256; it must be bytes of at least 32 bytes."""

    @pytest.mark.parametrize("key", _INVALID_KEYS)
    def test_an_invalid_key_is_rejected_naming_the_sink(self, key: Any) -> None:
        with pytest.raises(ValueError, match="OtelLogAuditSink"):
            OtelLogAuditSink(user_hash_key=key)

    @pytest.mark.parametrize("key", [None, b"k" * 32, b"k" * 33, b"\xaa" * 131], ids=["none", "32", "33", "131"])
    def test_none_and_bytes_of_at_least_32_bytes_are_accepted(self, key: bytes | None) -> None:
        assert isinstance(OtelLogAuditSink(user_hash_key=key), OtelLogAuditSink)

    def test_the_minimum_is_the_one_the_run_manifest_signer_uses(self) -> None:
        minimum = run_manifest_module._MIN_KEY_BYTES

        OtelLogAuditSink(user_hash_key=b"k" * minimum)
        with pytest.raises(ValueError, match="OtelLogAuditSink"):
            OtelLogAuditSink(user_hash_key=b"k" * (minimum - 1))

    def test_the_repr_does_not_print_the_key(self) -> None:
        sink = OtelLogAuditSink(user_hash_key=_LEAK_KEY)

        text = f"{sink!r} {sink}"
        assert _LEAK_KEY.decode("ascii") not in text
        assert repr(_LEAK_KEY) not in text
        assert _LEAK_KEY.hex() not in text

    def test_a_failing_emit_logs_neither_the_key_nor_the_raw_principal(
        self, log_exporter: InMemoryLogRecordExporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        record = _audit_record(tenant_id="tenant-1", principal=_PRINCIPAL)

        with patch.object(SdkLogger, "emit", side_effect=RuntimeError("boom-marker")):
            with caplog.at_level(logging.DEBUG):
                OtelLogAuditSink(user_hash_key=_LEAK_KEY).write(record)

        assert len(_module_warnings(caplog)) == 1
        assert _LEAK_KEY.decode("ascii") not in caplog.text
        assert repr(_LEAK_KEY) not in caplog.text
        assert _PRINCIPAL not in caplog.text


class TestOtelLogAuditSinkMatchTimeRefusal:
    """A fail_closed refusal at FEATURE_GROUP_MATCHED still reaches the log channel."""

    def test_the_refusal_emits_one_warn_record(self, log_exporter: InMemoryLogRecordExporter) -> None:
        extender = AuditExtender(sink=OtelLogAuditSink(), fail_closed=True)

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_MATCHED).activate():
            with pytest.raises(IdentityRequiredError):
                extender(lambda: None)

        log = _single_log(log_exporter)
        attributes = _attributes(log)
        assert log.log_record.severity_number == SeverityNumber.WARN
        assert attributes["mloda.audit.hook"] == "FEATURE_GROUP_MATCHED"
        assert attributes["mloda.audit.decision"] == "deny"
        assert "IdentityRequiredError" in attributes["error.type"]


class TestOtelLogAuditSinkRealExporter:
    """A real SDK exporter converts each log itself; the in-memory exporter keeps the object and never does."""

    def test_an_allow_record_is_serialised_with_its_body_severity_and_attributes(self) -> None:
        record = _audit_record(
            tenant_id="tenant-1",
            project_id="project-1",
            principal=_PRINCIPAL,
            run_id="run-123",
            feature_group_class="my.module.MyFeatureGroup",
            feature_names=("value_int", "value_str"),
        )

        text, logs = _emit_through_console(record)

        assert len(logs) == 1
        assert logs[0]["body"] == "allow"
        assert logs[0]["severity_text"] == "INFO"
        assert logs[0]["attributes"] == {
            "mloda.audit.decision": "allow",
            "mloda.audit.policy_version": _POLICY_VERSION,
            "mloda.audit.hook": "FEATURE_GROUP_CALCULATE_FEATURE",
            "mloda.run.id": "run-123",
            "mloda.tenant.id": "tenant-1",
            "mloda.project.id": "project-1",
            "user.hash": _sha256(_PRINCIPAL),
            "mloda.feature_group.name": "my.module.MyFeatureGroup",
            "mloda.feature.names": ["value_int", "value_str"],
        }
        assert _PRINCIPAL not in text

    def test_a_deny_record_is_serialised_with_its_reason_feature_names_and_hashed_principal(self) -> None:
        record = _audit_record(
            fail_closed=True,
            hook=ExtenderHook.FEATURE_GROUP_MATCHED,
            tenant_id=None,
            project_id="project-1",
            principal=_PRINCIPAL,
            run_id="run-123",
            feature_group_class="my.module.MyFeatureGroup",
            feature_names=("value_int", "value_str"),
        )

        text, logs = _emit_through_console(record)

        assert len(logs) == 1
        assert logs[0]["body"] == "deny"
        assert logs[0]["severity_text"] == "WARN"
        assert logs[0]["attributes"] == {
            "mloda.audit.decision": "deny",
            "mloda.audit.deny_reason": "missing_tenant_id",
            "mloda.audit.policy_version": _POLICY_VERSION,
            "mloda.audit.hook": "FEATURE_GROUP_MATCHED",
            "mloda.run.id": "run-123",
            "mloda.project.id": "project-1",
            "user.hash": _sha256(_PRINCIPAL),
            "mloda.feature_group.name": "my.module.MyFeatureGroup",
            "mloda.feature.names": ["value_int", "value_str"],
            "error.type": _IDENTITY_REQUIRED_ERROR_TYPE,
        }
        assert _PRINCIPAL not in text

    def test_a_keyed_record_is_serialised_with_the_hmac_and_without_the_key_or_principal(self) -> None:
        record = _audit_record(tenant_id="tenant-1", project_id="project-1", principal=_PRINCIPAL)

        text, logs = _emit_through_console(record, key=_LEAK_KEY)

        assert len(logs) == 1
        assert logs[0]["attributes"]["user.hash"] == _hmac_sha256(_LEAK_KEY, _PRINCIPAL)
        assert _PRINCIPAL not in text
        assert _LEAK_KEY.decode("ascii") not in text


class TestOtelLogAuditSinkFailureIsolation:
    """write never raises an Exception; a failure is logged as a warning naming the sink and the error type."""

    def test_a_failing_emit_is_logged_and_not_raised(
        self, log_exporter: InMemoryLogRecordExporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        record = _audit_record(tenant_id="tenant-1")

        with patch.object(SdkLogger, "emit", side_effect=RuntimeError("boom-marker")):
            with caplog.at_level(logging.WARNING, logger=otel_log_sink_module.__name__):
                OtelLogAuditSink().write(record)

        warnings = _module_warnings(caplog)
        assert len(warnings) == 1
        assert "OtelLogAuditSink" in warnings[0]
        assert "RuntimeError" in warnings[0]

    def test_a_malformed_event_time_is_logged_and_not_raised(
        self, log_exporter: InMemoryLogRecordExporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        record = {**_audit_record(tenant_id="tenant-1"), "event_time": "not-a-time"}

        with caplog.at_level(logging.WARNING, logger=otel_log_sink_module.__name__):
            OtelLogAuditSink().write(record)

        warnings = _module_warnings(caplog)
        assert len(warnings) == 1
        assert "OtelLogAuditSink" in warnings[0]
        assert "ValueError" in warnings[0]

    def test_a_failing_provider_lookup_is_logged_and_not_raised(self, caplog: pytest.LogCaptureFixture) -> None:
        record = _audit_record(tenant_id="tenant-1")

        with patch(_GET_LOGGER_PROVIDER, side_effect=RuntimeError("boom-marker")):
            with caplog.at_level(logging.WARNING, logger=otel_log_sink_module.__name__):
                OtelLogAuditSink().write(record)

        warnings = _module_warnings(caplog)
        assert len(warnings) == 1
        assert "OtelLogAuditSink" in warnings[0]
        assert "RuntimeError" in warnings[0]

    def test_a_wrapped_call_returns_its_result_unchanged_when_the_emit_fails(
        self, log_exporter: InMemoryLogRecordExporter
    ) -> None:
        extender = AuditExtender(sink=OtelLogAuditSink())

        with patch.object(SdkLogger, "emit", side_effect=RuntimeError("boom-marker")):
            with make_hook_context(tenant_id="tenant-1").activate():
                assert extender(lambda a, b: a + b, 3, 4) == 7

    def test_a_refusal_still_raises_identity_required_when_the_emit_fails(
        self, log_exporter: InMemoryLogRecordExporter
    ) -> None:
        extender = AuditExtender(sink=OtelLogAuditSink(), fail_closed=True)

        with patch.object(SdkLogger, "emit", side_effect=RuntimeError("boom-marker")):
            with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_MATCHED).activate():
                with pytest.raises(IdentityRequiredError):
                    extender(lambda: None)


class TestOtelLogAuditSinkWithoutSdkProvider:
    """Without an SDK LoggerProvider the logs go nowhere, so the first write warns once per process."""

    @pytest.mark.parametrize("make_provider", _NO_SDK_PROVIDERS)
    def test_the_first_write_warns_once_naming_the_provider_and_child_bootstrap(
        self, make_provider: Callable[[], Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = make_provider()
        record = _audit_record(tenant_id="tenant-1")

        with patch(_GET_LOGGER_PROVIDER, return_value=provider):
            with caplog.at_level(logging.WARNING, logger=otel_log_sink_module.__name__):
                OtelLogAuditSink().write(record)

        warnings = _module_warnings(caplog)
        assert len(warnings) == 1
        assert "LoggerProvider" in warnings[0]
        assert "child_bootstrap" in warnings[0]

    @pytest.mark.parametrize("make_provider", _NO_SDK_PROVIDERS)
    def test_later_writes_and_a_fresh_sink_do_not_warn_again(
        self, make_provider: Callable[[], Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = make_provider()
        record = _audit_record(tenant_id="tenant-1")
        first = OtelLogAuditSink()

        with patch(_GET_LOGGER_PROVIDER, return_value=provider):
            with caplog.at_level(logging.WARNING, logger=otel_log_sink_module.__name__):
                first.write(record)
                assert len(_module_warnings(caplog)) == 1
                caplog.clear()
                first.write(record)
                OtelLogAuditSink().write(record)

        assert _module_warnings(caplog) == []

    def test_an_sdk_provider_logs_no_such_warning(
        self, log_exporter: InMemoryLogRecordExporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=otel_log_sink_module.__name__):
            _write_one(log_exporter, _audit_record(tenant_id="tenant-1"))

        assert _module_warnings(caplog) == []


class TestOtelLogAuditSinkPickle:
    """The sink holds no provider, so it pickles and the copy resolves its own provider."""

    def test_a_pickled_copy_still_emits_into_the_patched_provider(
        self, log_exporter: InMemoryLogRecordExporter
    ) -> None:
        sink = OtelLogAuditSink()
        record = _audit_record(tenant_id="tenant-1")
        sink.write(record)

        copy = pickle.loads(pickle.dumps(sink))  # nosec
        copy.write(record)

        assert isinstance(copy, OtelLogAuditSink)
        assert len(log_exporter.get_finished_logs()) == 2

    def test_a_pickled_keyed_copy_emits_the_same_user_hash_as_the_original(
        self, log_exporter: InMemoryLogRecordExporter
    ) -> None:
        sink = OtelLogAuditSink(user_hash_key=_KEY)
        record = _audit_record(tenant_id="tenant-1")
        sink.write(record)

        copy = pickle.loads(pickle.dumps(sink))  # nosec
        copy.write(record)

        original_log, copy_log = log_exporter.get_finished_logs()
        assert _attributes(copy_log)["user.hash"] == _attributes(original_log)["user.hash"]
        assert _attributes(copy_log)["user.hash"] == _hmac_sha256(_KEY, _PRINCIPAL)
        assert _attributes(copy_log)["user.hash"] != _sha256(_PRINCIPAL)


class TestOtelLogAuditSinkRunAll:
    """run_all through a tee of the NDJSON sink and the log sink: both channels hold the same records."""

    def test_a_verified_run_writes_matching_allow_records_to_both_channels(
        self, tmp_path: Path, log_exporter: InMemoryLogRecordExporter
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        extender = AuditExtender(sink=TeeAuditSink(NdjsonAuditSink(audit_path), OtelLogAuditSink()))

        with verified_context(tenant_id="tenant-42", principal="svc"):
            values = run_value_int(extender)

        assert values == expected_value_int()
        records = _read_ndjson(audit_path)
        logs = log_exporter.get_finished_logs()
        _assert_channels_agree(records, logs)
        for record, log in zip(records, logs, strict=True):
            attributes = _attributes(log)
            assert record["decision"] == "allow"
            assert record["run_id"]
            assert log.log_record.severity_number == SeverityNumber.INFO
            assert log.log_record.severity_text == "INFO"
            assert attributes["mloda.tenant.id"] == "tenant-42"
            assert attributes["user.hash"] == _sha256("svc")

    def test_an_unverified_run_writes_matching_deny_records_and_still_completes(
        self, tmp_path: Path, log_exporter: InMemoryLogRecordExporter
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        extender = AuditExtender(sink=TeeAuditSink(NdjsonAuditSink(audit_path), OtelLogAuditSink()))

        values = run_value_int(extender)

        assert values == expected_value_int()
        records = _read_ndjson(audit_path)
        logs = log_exporter.get_finished_logs()
        _assert_channels_agree(records, logs)
        for record, log in zip(records, logs, strict=True):
            assert record["decision"] == "deny"
            assert log.log_record.severity_number == SeverityNumber.WARN
            assert log.log_record.severity_text == "WARN"
            assert _attributes(log)["mloda.audit.deny_reason"] == "missing_tenant_id"

    def test_a_fail_closed_refusal_reaches_both_channels(
        self, tmp_path: Path, log_exporter: InMemoryLogRecordExporter
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        extender = AuditExtender(sink=TeeAuditSink(NdjsonAuditSink(audit_path), OtelLogAuditSink()), fail_closed=True)

        with pytest.raises(IdentityRequiredError):
            run_value_int(extender)

        records = _read_ndjson(audit_path)
        log = _single_log(log_exporter)
        assert len(records) == 1
        assert records[0]["hook"] == ExtenderHook.FEATURE_GROUP_MATCHED.name
        assert records[0]["decision"] == "deny"
        _assert_channels_agree(records, (log,))
        assert log.log_record.severity_number == SeverityNumber.WARN
        assert _attributes(log)["mloda.audit.hook"] == "FEATURE_GROUP_MATCHED"
