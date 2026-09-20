"""Tests for AuditExtender: contract compliance, record shape, allow/deny decisions, the fail_closed
refusal, error handling and NdjsonAuditSink; __call__ tests build a HookContext manually, mirroring
core's own instrumentation."""

from __future__ import annotations

import json
import logging
import os
import pickle  # nosec
import stat
from collections.abc import Mapping
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from mloda.steward import Extender, ExtenderHook, verified_context
from mloda.user import ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.enterprise.extenders.audit import AuditExtender, IdentityRequiredError, NdjsonAuditSink
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.runners import CountingExtender, expected_value_int, run_value_int

_BOTH_POSTURES = pytest.mark.parametrize("fail_closed", [False, True])

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
}


class InMemoryAuditSink:
    """Collects every written record in memory, in call order; module-level so it survives pickling."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: Mapping[str, Any]) -> None:
        self.records.append(dict(record))


class _CountingCall:
    """A wrapped call that counts how often it ran."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        return 42


class TestAuditExtenderContract(ExtenderContractTestMixin):
    """AuditExtender satisfies the shared Extender contract."""

    fail_closed = False

    @classmethod
    def extender_class(cls) -> type[Extender]:
        return AuditExtender

    @classmethod
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

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


class TestAuditExtenderFailClosedContract(TestAuditExtenderContract):
    """The fail_closed posture satisfies the same Extender contract, under a present identity."""

    fail_closed = True

    @classmethod
    def supports_warning_only(cls) -> bool:
        return False

    @classmethod
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        return {ExtenderHook.FEATURE_GROUP_MATCHED, ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

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

    @pytest.mark.parametrize(
        "kwargs",
        [{"raise_on_error": False}, {"required_identity": ()}],
        ids=["raise_on_error_false", "empty_required_identity"],
    )
    def test_fail_closed_that_could_not_be_enforced_raises_value_error(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            AuditExtender(sink=InMemoryAuditSink(), fail_closed=True, **kwargs)

    def test_fail_closed_with_defaults_is_accepted_wraps_the_matched_hook_and_sorts_outermost(self) -> None:
        extender = AuditExtender(sink=InMemoryAuditSink(), fail_closed=True)

        assert extender.wraps() == {ExtenderHook.FEATURE_GROUP_MATCHED, ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}
        assert extender.priority == 0
        assert AuditExtender(sink=InMemoryAuditSink()).priority == 100


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

    def test_record_has_exactly_the_expected_keys(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

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
    def test_sink_failure_on_the_refusal_path_propagates_and_the_call_never_runs(self, hook: ExtenderHook) -> None:
        class _DiskFullSink:
            def write(self, record: Mapping[str, Any]) -> None:
                raise OSError("disk full")

        extender = AuditExtender(sink=_DiskFullSink(), fail_closed=True)
        call = _CountingCall()

        with make_hook_context(hook=hook).activate():
            with pytest.raises(OSError, match="disk full") as excinfo:
                extender(call)

        assert call.calls == 0
        assert isinstance(excinfo.value.__context__, IdentityRequiredError)


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
            assert record["run_id"] is not None
            assert record["hook"] == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE.name

    def test_run_all_without_verified_context_denies_but_still_runs(self) -> None:
        sink = InMemoryAuditSink()
        extender = AuditExtender(sink=sink)

        values = run_value_int(extender)

        assert values == expected_value_int()
        assert sink.records
        for record in sink.records:
            assert record["decision"] == "deny"
            assert record["deny_reason"] == "missing_tenant_id"

    def test_run_all_fail_closed_without_verified_context_refuses_at_plan_time(self) -> None:
        sink = InMemoryAuditSink()
        counting = CountingExtender()

        with pytest.raises(IdentityRequiredError):
            run_value_int(AuditExtender(sink=sink, fail_closed=True), counting)

        assert len(sink.records) == 1
        record = sink.records[0]
        assert record["hook"] == ExtenderHook.FEATURE_GROUP_MATCHED.name
        assert record["decision"] == "deny"
        assert counting.calls == 0

    @pytest.mark.parametrize(
        "mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING]
    )
    def test_run_all_fail_closed_refuses_at_calculate_when_identity_is_gone_by_run_time(
        self, mode: ParallelizationMode, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        audit_path = tmp_path / "audit.ndjson"
        counting = CountingExtender()
        # Lower than the default 100: it runs outside the gate unless the gate sorts itself outermost.
        counting.priority = 50
        # Only MULTIPROCESSING needs the flight_server fixture.
        flight_server = (
            request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
        )

        with verified_context(tenant_id="t"):
            session = mloda.prepare(
                ["value_int"],
                compute_frameworks={PyArrowTable},
                plugin_collector=PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator}),
                function_extender={AuditExtender(sink=NdjsonAuditSink(audit_path), fail_closed=True), counting},
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
