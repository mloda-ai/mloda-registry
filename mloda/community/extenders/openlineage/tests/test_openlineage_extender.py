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
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
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

    def test_default_client_is_none_and_call_still_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Disable ambient OpenLineage env vars (e.g. OPENLINEAGE_URL) so the default client always
        resolves to NoopTransport here, regardless of the environment running the suite."""
        monkeypatch.setenv("OPENLINEAGE_DISABLED", "true")
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
        datasets_by_name = {ds.name: ds for ds in complete_event.outputs}

        int_facet = datasets_by_name["value_int"].facets
        assert int_facet is not None
        int_schema = int_facet.get("schema")
        assert isinstance(int_schema, schema_dataset.SchemaDatasetFacet)
        assert int_schema.fields is not None
        assert [f.name for f in int_schema.fields] == ["value_int"]
        assert [f.type for f in int_schema.fields] == ["int64"]

        str_facet = datasets_by_name["value_str"].facets
        assert str_facet is not None
        str_schema = str_facet.get("schema")
        assert isinstance(str_schema, schema_dataset.SchemaDatasetFacet)
        assert str_schema.fields is not None
        assert [f.name for f in str_schema.fields] == ["value_str"]
        assert [f.type for f in str_schema.fields] == ["string"]

    def test_schema_facet_never_leaks_columns_outside_feature_names(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        """Each OutputDataset's schema facet must be narrowed to its own field only, never the
        whole frame's schema; a column absent from feature_names must never appear anywhere."""
        client, transport = ol_capture
        context = _make_context(feature_names=("value_int", "value_str"))
        extender = OpenLineageExtender(client=client)
        table = pa.table(
            {
                "value_int": [1, 2, 3],
                "value_str": ["a", "b", "c"],
                "internal_secret_col": [1, 2, 3],
            }
        )

        with context.activate():
            extender(lambda: table)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        for dataset in complete_event.outputs:
            assert dataset.facets is not None
            schema_facet = dataset.facets.get("schema")
            if schema_facet is not None:
                assert isinstance(schema_facet, schema_dataset.SchemaDatasetFacet)
                assert schema_facet.fields is not None
                field_names = [f.name for f in schema_facet.fields]
                assert "internal_secret_col" not in field_names

        from openlineage.client.serde import Serde

        for event in transport.events:
            serialized = Serde.to_json(event)
            assert "internal_secret_col" not in serialized

    def test_schema_facet_field_names_are_strings_for_pandas_rangeindex_columns(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        """A pandas DataFrame with the default integer RangeIndex columns (labels 0, 1, ...) must
        still produce string field names in the schema facet, never raw ints."""
        import pandas as pd

        client, transport = ol_capture
        context = _make_context(feature_names=("0", "1"))
        extender = OpenLineageExtender(client=client)
        frame = pd.DataFrame([[1, "a"]])

        with context.activate():
            extender(lambda: frame)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        datasets_by_name = {ds.name: ds for ds in complete_event.outputs}

        for expected_name in ("0", "1"):
            dataset = datasets_by_name[expected_name]
            assert dataset.facets is not None
            schema_facet = dataset.facets.get("schema")
            assert isinstance(schema_facet, schema_dataset.SchemaDatasetFacet)
            assert schema_facet.fields is not None
            for f in schema_facet.fields:
                assert isinstance(f.name, str)
            assert [f.name for f in schema_facet.fields] == [expected_name]

    def test_schema_facet_type_not_garbage_for_duplicate_pandas_column_names(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        """A duplicate pandas column name ('x', 'x') must not make dtypes[c] return a Series whose
        str() is a multi-line garbage repr; every field for name 'x' must report the real dtype."""
        import pandas as pd

        client, transport = ol_capture
        context = _make_context(feature_names=("x",))
        extender = OpenLineageExtender(client=client)
        frame = pd.DataFrame([[1, 2]], columns=["x", "x"])

        with context.activate():
            extender(lambda: frame)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        output = complete_event.outputs[0]
        assert output.name == "x"
        assert output.facets is not None
        schema_facet = output.facets.get("schema")
        assert isinstance(schema_facet, schema_dataset.SchemaDatasetFacet)
        assert schema_facet.fields is not None
        assert len(schema_facet.fields) >= 1
        for f in schema_facet.fields:
            assert f.name == "x"
            assert f.type == "int64"

    def test_schema_facet_types_come_from_schema_fields_for_spark_shaped_result(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        """A pyspark-shaped result (columns, (name, type) dtypes tuples, schema.fields with dataType) must
        report str(field.dataType) per column, never the stringified dtypes tuple."""

        class _DataType:
            def __init__(self, rendered: str) -> None:
                self._rendered = rendered

            def __str__(self) -> str:
                return self._rendered

        class _StructField:
            def __init__(self, name: str, data_type: _DataType) -> None:
                self.name = name
                self.dataType = data_type

        class _StructType:
            def __init__(self, fields: list[_StructField]) -> None:
                self.fields = fields
                self.names = [f.name for f in fields]

        class _SparkFrame:
            def __init__(self, columns: list[str], dtypes: list[tuple[str, str]], schema: _StructType) -> None:
                self.columns = columns
                self.dtypes = dtypes
                self.schema = schema

        client, transport = ol_capture
        context = _make_context(feature_names=("value_int", "value_str"))
        extender = OpenLineageExtender(client=client)
        frame = _SparkFrame(
            columns=["value_int", "value_str"],
            dtypes=[("value_int", "bigint"), ("value_str", "string")],
            schema=_StructType(
                [
                    _StructField("value_int", _DataType("LongType()")),
                    _StructField("value_str", _DataType("StringType()")),
                ]
            ),
        )

        with context.activate():
            extender(lambda: frame)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        datasets_by_name = {ds.name: ds for ds in complete_event.outputs}

        expected_types = {"value_int": "LongType()", "value_str": "StringType()"}
        for expected_name, expected_type in expected_types.items():
            dataset = datasets_by_name[expected_name]
            assert dataset.facets is not None
            schema_facet = dataset.facets.get("schema")
            assert isinstance(schema_facet, schema_dataset.SchemaDatasetFacet)
            assert schema_facet.fields is not None
            assert [f.name for f in schema_facet.fields] == [expected_name]
            assert [f.type for f in schema_facet.fields] == [expected_type]

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

    def test_base_exception_still_emits_fail_event_and_propagates(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        """A BaseException that is not an Exception (e.g. KeyboardInterrupt-like) must still get a
        FAIL event emitted before propagating, never be swallowed by an `except Exception` guard."""
        client, transport = ol_capture
        context = _make_context()
        extender = OpenLineageExtender(client=client)

        class _CustomBaseException(BaseException):
            pass

        def func() -> None:
            raise _CustomBaseException("base boom")

        with context.activate():
            with pytest.raises(_CustomBaseException):
                extender(func)

        assert transport.events[-1].eventType == RunState.FAIL

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

        from openlineage.client.facet_v2 import datasource_dataset

        assert input_dataset.facets is not None
        data_source_facet = input_dataset.facets["dataSource"]
        assert isinstance(data_source_facet, datasource_dataset.DatasourceDatasetFacet)
        assert data_source_facet.name == "s3://bucket/key.parquet"

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
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        context = _make_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="standalone")

        with caplog.at_level(logging.DEBUG):
            with context.activate():
                result = extender(lambda: "loaded")

        assert result == "loaded"
        assert transport.events == []

        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(
            "OpenLineageExtender" in r.message
            and "calculate" in r.message.lower()
            and ("enclosing" in r.message.lower() or "open" in r.message.lower())
            for r in debug_records
        ), debug_records


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
        calls = {"n": 0}

        def func(x: int, y: int) -> int:
            calls["n"] += 1
            return x + y

        with patch(
            "openlineage.client.client.OpenLineageClient.emit",
            side_effect=RuntimeError("openlineage instrumentation boom"),
        ):
            with caplog.at_level(logging.WARNING):
                with context.activate():
                    result = composite(func, 3, 4)

        assert result == 7, "Failing OpenLineageExtender must fall back to the wrapped function result"
        assert calls["n"] == 1
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


class FailingCalculateFeatureGroup(FeatureGroup):
    """Root feature group whose calculate_feature always raises, counting its invocations."""

    calls = 0

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"openlineage_boom_feature"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.calls += 1
        raise RuntimeError("inner boom")


def _run_boom_feature(*extenders: Extender) -> Any:
    """Run the always-failing ``openlineage_boom_feature`` through run_all with the given extenders."""
    plugin_collector = PluginCollector.enabled_feature_groups({FailingCalculateFeatureGroup})

    return mloda.run_all(
        ["openlineage_boom_feature"],
        compute_frameworks={PyArrowTable},
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
    )


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

    def test_run_all_wrapped_function_failure_propagates_and_runs_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """A failure of the wrapped function propagates through the real _CompositeExtender/core
        dispatch path, regardless of raise_on_error, without a re-run."""
        FailingCalculateFeatureGroup.calls = 0

        with caplog.at_level(logging.WARNING):
            with pytest.raises(Exception, match="inner boom"):
                _run_boom_feature(OpenLineageExtender(raise_on_error=False))

        assert FailingCalculateFeatureGroup.calls == 1

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("OpenLineageExtender" in message and "inner boom" in message for message in warnings), warnings


class TestOpenLineageExtenderContentIsolation:
    """Emitted events must never carry the wrapped function's exception message text."""

    def test_func_exception_message_does_not_leak_into_emitted_events(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        from openlineage.client.serde import Serde

        client, transport = ol_capture
        context = _make_context()
        extender = OpenLineageExtender(client=client)
        marker = "SENSITIVE_ROW_VALUE_xyz123"

        def func() -> None:
            raise ValueError(f"invalid value found: {marker}")

        with context.activate():
            with pytest.raises(ValueError):
                extender(func)

        assert len(transport.events) >= 1
        for event in transport.events:
            serialized = Serde.to_json(event)
            assert marker not in serialized, f"event {event.eventType} leaked the func exception's message ({marker!r})"
