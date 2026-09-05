"""Tests for OpenLineageExtender.

Every direct __call__ is wrapped in a manually built HookContext.activate() scope, mirroring
core's INPUT_DATA_LOAD nesting inside the enclosing CALCULATE_FEATURE HookContext.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pytest
from mloda.core.abstract_plugins.function_extender import _CompositeExtender  # no public equivalent yet
from mloda.steward import Extender, ExtenderHook, HookContext
from mloda.user import PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from openlineage.client.client import Event, OpenLineageClient
from openlineage.client.event_v2 import InputDataset, OutputDataset, RunEvent, RunState
from openlineage.client.facet_v2 import parent_run, schema_dataset
from openlineage.client.transport.transport import Config, Transport


class RecordingTransport(Transport):
    """In-memory Transport double: the OpenLineage equivalent of InMemorySpanExporter."""

    kind = "recording"
    config_class = Config

    def __init__(self, config: Config | None = None) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: Event) -> None:
        # OpenLineageExtender only ever emits RunEvent; Event's other union members are unused here.
        assert isinstance(event, RunEvent)
        self.events.append(event)


@pytest.fixture
def ol_capture() -> Iterator[tuple[OpenLineageClient, RecordingTransport]]:
    """A fresh, isolated (client, transport) pair per test."""
    transport = RecordingTransport()
    client = OpenLineageClient(transport=transport)
    yield client, transport


def _make_context(
    *,
    hook: ExtenderHook = ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
    feature_group_class: str = "tests.openlineage.DummyFeatureGroup",
    feature_group_version: str = "1",
    plugin_version: str | None = None,
    feature_names: tuple[str, ...] = ("value_int",),
    input_features: frozenset[str] | None = None,
    compute_framework_name: str = "PyArrowTable",
    rows_in: int | None = None,
    rows_out: int | None = None,
    run_id: str | None = None,
    data_access_identity: str | None = None,
) -> HookContext:
    """Build a HookContext with sane defaults; every field is overridable per test."""
    return HookContext(
        hook=hook,
        feature_group_class=feature_group_class,
        feature_group_version=feature_group_version,
        plugin_version=plugin_version,
        feature_names=feature_names,
        input_features=input_features,
        compute_framework_name=compute_framework_name,
        rows_in=rows_in,
        rows_out=rows_out,
        run_id=run_id,
        data_access_identity=data_access_identity,
    )


class TestOpenLineageExtenderImport:
    def test_import_from_package(self) -> None:
        from mloda.community.extenders.openlineage import OpenLineageExtender

        assert OpenLineageExtender is not None

    def test_class_is_accessible(self) -> None:
        from mloda.community.extenders.openlineage import OpenLineageExtender

        assert isinstance(OpenLineageExtender, type)


class TestOpenLineageExtenderInheritance:
    def test_inherits_from_extender(self) -> None:
        assert issubclass(OpenLineageExtender, Extender)

    def test_instance_is_extender(self) -> None:
        assert isinstance(OpenLineageExtender(), Extender)


class TestOpenLineageExtenderErrorContract:
    """raise_on_error and wraps(): observability-only, so it must default to warning-only."""

    def test_raise_on_error_defaults_to_false(self) -> None:
        assert OpenLineageExtender().raise_on_error is False

    def test_raise_on_error_can_be_enabled(self) -> None:
        assert OpenLineageExtender(raise_on_error=True).raise_on_error is True

    def test_raise_on_error_explicit_false(self) -> None:
        assert OpenLineageExtender(raise_on_error=False).raise_on_error is False

    def test_wraps_returns_calculate_and_input_data_load_only(self) -> None:
        expected = {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.INPUT_DATA_LOAD,
        }
        assert OpenLineageExtender().wraps() == expected


class TestOpenLineageExtenderConstructorOptions:
    """client injection: the seam that keeps tests off any real OpenLineage backend."""

    def test_client_is_used_to_emit_events(self, ol_capture: tuple[OpenLineageClient, RecordingTransport]) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        context = _make_context()

        with context.activate():
            extender(lambda: None)

        assert len(transport.events) >= 1

    def test_default_client_is_none_and_call_still_works(self) -> None:
        extender = OpenLineageExtender()
        context = _make_context()

        with context.activate():
            result = extender(lambda: 42)

        assert result == 42


class TestOpenLineageExtenderNoAmbientContext:
    def test_func_called_once_and_result_returned_with_no_events(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        calls = {"n": 0}

        def func() -> str:
            calls["n"] += 1
            return "value"

        assert HookContext.current() is None
        result = extender(func)

        assert result == "value"
        assert calls["n"] == 1
        assert transport.events == []


class TestOpenLineageExtenderStartEvent:
    """One RunEvent(START), emitted before func runs, with empty inputs/outputs."""

    def test_job_namespace_and_name_use_defaults(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = _make_context(feature_group_class="pkg.mod.MyFeatureGroup")
        extender = OpenLineageExtender(client=client)

        with context.activate():
            extender(lambda: None)

        assert len(transport.events) == 2
        event = transport.events[0]
        assert event.eventType == RunState.START
        assert event.job.namespace == "mloda"
        assert event.job.name == "pkg.mod.MyFeatureGroup"
        assert event.inputs == []
        assert event.outputs == []

    def test_job_namespace_uses_constructor_override(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = _make_context()
        extender = OpenLineageExtender(client=client, job_namespace="custom-ns")

        with context.activate():
            extender(lambda: None)

        assert transport.events[0].job.namespace == "custom-ns"

    def test_start_event_present_before_func_is_called(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = _make_context()
        extender = OpenLineageExtender(client=client)
        observed: dict[str, int] = {}

        def func() -> str:
            observed["count_at_call_time"] = len(transport.events)
            return "result"

        with context.activate():
            result = extender(func)

        assert observed["count_at_call_time"] == 1
        assert result == "result"


class TestOpenLineageExtenderCompleteEvent:
    """After a successful func: outputs (one per feature name), then RunEvent(COMPLETE)."""

    def test_exactly_two_events_start_then_complete_on_success(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = _make_context()
        extender = OpenLineageExtender(client=client)

        with context.activate():
            extender(lambda: None)

        assert len(transport.events) == 2
        assert transport.events[0].eventType == RunState.START
        assert transport.events[1].eventType == RunState.COMPLETE

    def test_output_dataset_per_feature_name(self, ol_capture: tuple[OpenLineageClient, RecordingTransport]) -> None:
        client, transport = ol_capture
        context = _make_context(feature_names=("value_int", "value_str"))
        extender = OpenLineageExtender(client=client, dataset_namespace="custom-ds")

        with context.activate():
            extender(lambda: None)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        assert [ds.name for ds in complete_event.outputs] == ["value_int", "value_str"]
        for dataset in complete_event.outputs:
            assert isinstance(dataset, OutputDataset)
            assert dataset.namespace == "custom-ds"

    def test_schema_facet_present_when_result_exposes_pyarrow_schema(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = _make_context(feature_names=("value_int", "value_str"))
        extender = OpenLineageExtender(client=client)
        table = pa.table({"value_int": [1, 2, 3], "value_str": ["a", "b", "c"]})

        with context.activate():
            extender(lambda: table)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        for dataset in complete_event.outputs:
            assert dataset.facets is not None
            schema_facet = dataset.facets.get("schema")
            assert isinstance(schema_facet, schema_dataset.SchemaDatasetFacet)
            assert schema_facet.fields is not None
            field_names = [f.name for f in schema_facet.fields]
            field_types = [f.type for f in schema_facet.fields]
            assert field_names == ["value_int", "value_str"]
            assert field_types == ["int64", "string"]

    def test_schema_facet_absent_when_result_has_no_introspectable_schema(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = _make_context()
        extender = OpenLineageExtender(client=client)

        with context.activate():
            extender(lambda: 42)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        output = complete_event.outputs[0]
        assert output.facets is not None
        assert "schema" not in output.facets


class TestOpenLineageExtenderParentRunFacet:
    """ParentRunFacet is present iff context.run_id is not None."""

    def test_parent_facet_present_with_default_namespace_and_root_job_name(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        run_id = str(uuid.uuid4())
        context = _make_context(run_id=run_id)
        extender = OpenLineageExtender(client=client)

        with context.activate():
            extender(lambda: None)

        event = transport.events[0]
        assert event.run.facets is not None
        parent = event.run.facets.get("parent")
        assert isinstance(parent, parent_run.ParentRunFacet)
        assert parent.run.runId == run_id
        assert parent.job.namespace == "mloda"
        assert parent.job.name == "mloda.run_all"

    def test_parent_facet_uses_custom_job_namespace_and_root_job_name(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        run_id = str(uuid.uuid4())
        context = _make_context(run_id=run_id)
        extender = OpenLineageExtender(client=client, job_namespace="custom-ns", root_job_name="custom.root")

        with context.activate():
            extender(lambda: None)

        event = transport.events[0]
        assert event.run.facets is not None
        parent = event.run.facets.get("parent")
        assert isinstance(parent, parent_run.ParentRunFacet)
        assert parent.job.namespace == "custom-ns"
        assert parent.job.name == "custom.root"

    def test_parent_facet_absent_when_run_id_is_none(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = _make_context(run_id=None)
        extender = OpenLineageExtender(client=client)

        with context.activate():
            extender(lambda: None)

        event = transport.events[0]
        assert event.run.facets is not None
        assert "parent" not in event.run.facets


class TestOpenLineageExtenderFailureHandling:
    """When the wrapped function raises: never swallow, always propagate, but observe first."""

    def test_call_propagates_func_failure_and_invokes_func_exactly_once(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, _ = ol_capture
        context = _make_context()
        extender = OpenLineageExtender(client=client)
        calls = {"n": 0}

        def func() -> None:
            calls["n"] += 1
            raise RuntimeError("inner boom")

        with context.activate():
            with pytest.raises(RuntimeError, match="inner boom"):
                extender(func)

        assert calls["n"] == 1

    def test_call_emits_exactly_one_fail_event_after_the_start_event(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = _make_context()
        extender = OpenLineageExtender(client=client)

        def func() -> None:
            raise RuntimeError("inner boom")

        with context.activate():
            with pytest.raises(RuntimeError):
                extender(func)

        assert len(transport.events) == 2
        assert transport.events[0].eventType == RunState.START
        assert transport.events[1].eventType == RunState.FAIL
        assert transport.events[1].outputs == []

    def test_call_logs_warning_naming_extender_on_func_failure(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, _ = ol_capture
        context = _make_context()
        extender = OpenLineageExtender(client=client)

        def func() -> None:
            raise RuntimeError("inner boom")

        with caplog.at_level(logging.WARNING):
            with context.activate():
                with pytest.raises(RuntimeError):
                    extender(func)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OpenLineageExtender" in r.message and "inner boom" in r.message for r in warnings), warnings

    def test_fail_event_emit_failure_does_not_mask_original_exception(
        self,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A broken transport on the FAIL-emit path must never replace func's real exception."""
        client, _ = ol_capture
        context = _make_context()
        extender = OpenLineageExtender(client=client)

        original_emit = client.emit
        counts = {"n": 0}

        def flaky_emit(event: Any) -> None:
            counts["n"] += 1
            if counts["n"] == 1:
                original_emit(event)
                return
            raise RuntimeError("transport boom")

        monkeypatch.setattr(client, "emit", flaky_emit)

        def func() -> None:
            raise ValueError("inner boom")

        with context.activate():
            with pytest.raises(ValueError, match="inner boom"):
                extender(func)


class TestOpenLineageExtenderInputDataLoadCorrelation:
    """INPUT_DATA_LOAD fires nested inside an already-open CALCULATE_FEATURE invocation."""

    def test_no_event_emitted_for_input_data_load_and_input_recorded_on_complete(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client, dataset_namespace="custom-ds")
        outer_context = _make_context()
        inner_context = _make_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="s3://bucket/key.parquet")

        def inner_func() -> str:
            return "loaded-data"

        def outer_func() -> str:
            with inner_context.activate():
                inner_result = extender(inner_func)
            assert inner_result == "loaded-data"
            return "calculate-result"

        with outer_context.activate():
            result = extender(outer_func)

        assert result == "calculate-result"
        assert len(transport.events) == 2
        assert [e.eventType for e in transport.events] == [RunState.START, RunState.COMPLETE]

        complete_event = transport.events[1]
        assert complete_event.inputs is not None
        assert len(complete_event.inputs) == 1
        input_dataset = complete_event.inputs[0]
        assert isinstance(input_dataset, InputDataset)
        assert input_dataset.namespace == "custom-ds"
        assert input_dataset.name == "s3://bucket/key.parquet"

    def test_fail_event_carries_accumulated_input_from_nested_input_data_load(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        outer_context = _make_context()
        inner_context = _make_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="s3://bucket/key.parquet")

        def inner_func() -> str:
            return "loaded-data"

        def outer_func() -> None:
            with inner_context.activate():
                extender(inner_func)
            raise RuntimeError("outer boom")

        with outer_context.activate():
            with pytest.raises(RuntimeError, match="outer boom"):
                extender(outer_func)

        assert len(transport.events) == 2
        fail_event = transport.events[1]
        assert fail_event.eventType == RunState.FAIL
        assert fail_event.inputs is not None
        assert len(fail_event.inputs) == 1
        assert fail_event.inputs[0].name == "s3://bucket/key.parquet"

    def test_input_data_load_without_enclosing_calculate_does_not_raise_or_emit(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        context = _make_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="standalone")

        with context.activate():
            result = extender(lambda: "loaded")

        assert result == "loaded"
        assert transport.events == []


class TestOpenLineageExtenderPostCallInstrumentationFailure:
    """A bug in the extender's own post-success code must not corrupt a successful result."""

    def test_completion_emit_failure_does_not_corrupt_successful_result(
        self,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, transport = ol_capture
        context = _make_context()
        extender = OpenLineageExtender(client=client)

        original_emit = client.emit
        counts = {"n": 0}

        def flaky_emit(event: Any) -> None:
            counts["n"] += 1
            if counts["n"] == 2:
                raise RuntimeError("completion emit boom")
            original_emit(event)

        monkeypatch.setattr(client, "emit", flaky_emit)

        func_calls = {"n": 0}

        def func() -> list[int]:
            func_calls["n"] += 1
            return [1, 2, 3]

        with context.activate():
            result = extender(func)

        assert result == [1, 2, 3]
        assert func_calls["n"] == 1


class TestOpenLineageExtenderCompositeChaining:
    """OpenLineageExtender chains via _CompositeExtender; faults are injected on OpenLineageClient.emit."""

    def test_own_failure_falls_back_when_raise_on_error_false(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, _ = ol_capture
        context = _make_context()
        extender = OpenLineageExtender(raise_on_error=False, client=client)
        composite = _CompositeExtender([extender])

        def func(x: int, y: int) -> int:
            return x + y

        with patch(
            "openlineage.client.client.OpenLineageClient.emit",
            side_effect=RuntimeError("openlineage instrumentation boom"),
        ):
            with caplog.at_level(logging.WARNING):
                with context.activate():
                    result = composite(func, 3, 4)

        assert result == 7, "Failing OpenLineageExtender must fall back to the wrapped function result"
        assert any(r.levelno == logging.WARNING and "OpenLineageExtender" in r.message for r in caplog.records), (
            caplog.records
        )

    def test_own_failure_propagates_when_raise_on_error_true(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, _ = ol_capture
        context = _make_context()
        extender = OpenLineageExtender(raise_on_error=True, client=client)
        composite = _CompositeExtender([extender])

        def func(x: int, y: int) -> int:
            return x + y

        with patch(
            "openlineage.client.client.OpenLineageClient.emit",
            side_effect=RuntimeError("openlineage instrumentation boom"),
        ):
            with context.activate():
                with pytest.raises(RuntimeError, match="openlineage instrumentation boom"):
                    composite(func, 3, 4)


class TestOpenLineageExtenderRunAll:
    """End-to-end wiring through mloda.user.mloda.run_all: real RunEvents, unmodified results."""

    def test_run_all_produces_expected_events_and_leaves_result_unchanged(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator})

        results = mloda.run_all(
            ["value_int"],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
            function_extender={OpenLineageExtender(client=client)},
        )

        values = None
        for table in results:
            if "value_int" in table.column_names:
                values = table.to_pydict()["value_int"]
        expected_values = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]
        assert values == expected_values

        expected_job_name = f"{PyArrowDataOpsTestDataCreator.__module__}.{PyArrowDataOpsTestDataCreator.__qualname__}"
        matching_events = [e for e in transport.events if e.job.name == expected_job_name]
        assert any(e.eventType == RunState.START for e in matching_events)
        assert any(e.eventType == RunState.COMPLETE for e in matching_events)

        # Core mints one real run_id per run_all() call, so every calculate invocation's
        # ParentRunFacet must correlate back to that single shared run_id.
        parent_run_ids = set()
        for event in matching_events:
            assert event.run.facets is not None
            parent = event.run.facets.get("parent")
            assert isinstance(parent, parent_run.ParentRunFacet)
            parent_run_ids.add(parent.run.runId)
        assert len(parent_run_ids) == 1, parent_run_ids
