"""Tests for OpenLineageExtender: contract compliance via OpenLineageExtenderTestMixin, plus
schema-facet, namespace-override, dedupe and per-instance-attribution checks not covered by the
mixin.

Direct __call__ tests below wrap calls in a manually built HookContext.activate() scope, mirroring
core's INPUT_DATA_LOAD nesting inside the enclosing CALCULATE_FEATURE HookContext.
"""

from __future__ import annotations

import atexit
import logging
import pickle  # nosec
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pytest
from mloda.core.abstract_plugins.function_extender import _CompositeExtender
from mloda.steward import ExtenderHook

from mloda.community.extenders.openlineage import openlineage_extender as openlineage_extender_module
from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.openlineage import OpenLineageExtenderTestMixin, RecordingTransport, make_recording_client
from mloda.testing.extenders.runners import run_value_int
from openlineage.client.client import OpenLineageClient
from openlineage.client.event_v2 import RunState
from openlineage.client.facet_v2 import parent_run, schema_dataset
from openlineage.client.transport.transport import Config, Transport


class _LockHoldingTransport(Transport):
    """A Transport whose lock attribute cannot survive plain pickling."""

    kind = "lock-holding"
    config_class = Config

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def emit(self, event: Any) -> None:
        pass


class _IncompleteFlushTransport(Transport):
    """A Transport whose close() reports that not everything was flushed in time."""

    kind = "incomplete-flush"
    config_class = Config

    def emit(self, event: Any) -> None:
        pass

    def close(self, timeout: float = -1.0) -> bool:
        return False


@pytest.fixture
def ol_capture() -> Iterator[tuple[OpenLineageClient, RecordingTransport]]:
    """A fresh, isolated (client, transport) pair per test."""
    yield make_recording_client()


class TestOpenLineageExtenderContract(OpenLineageExtenderTestMixin):
    """OpenLineageExtender must satisfy the shared Extender contract and the OpenLineage RunEvent contract."""

    @classmethod
    def extender_class(cls) -> type[OpenLineageExtender]:
        return OpenLineageExtender

    def make_openlineage_extender(
        self, client: OpenLineageClient, *, raise_on_error: bool | None = None
    ) -> OpenLineageExtender:
        if raise_on_error is None:
            return OpenLineageExtender(client=client)
        return OpenLineageExtender(client=client, raise_on_error=raise_on_error)

    @classmethod
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        return {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.INPUT_DATA_LOAD,
        }

    @classmethod
    def emits_schema_facets(cls) -> bool:
        return True


class TestOpenLineageExtenderConstructorOptions:
    """client injection: the seam that keeps tests off any real OpenLineage backend."""

    def test_client_is_used_to_emit_events(self, ol_capture: tuple[OpenLineageClient, RecordingTransport]) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        context = make_hook_context()

        with context.activate():
            extender(lambda: None)

        assert len(transport.events) >= 1

    def test_default_client_is_none_and_call_still_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENLINEAGE_DISABLED", "true")
        extender = OpenLineageExtender()
        context = make_hook_context()

        with context.activate():
            result = extender(lambda: 42)

        assert result == 42


class TestOpenLineageExtenderPickling:
    """`_client` must never make the extender itself unpicklable."""

    def test_pickle_round_trip_keeps_config(self) -> None:
        extender = OpenLineageExtender(
            client=OpenLineageClient(transport=_LockHoldingTransport()),
            raise_on_error=True,
            job_namespace="custom-ns",
            dataset_namespace="custom-ds",
            root_job_name="custom.root",
        )

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        assert copy.raise_on_error is True
        assert copy.job_namespace == "custom-ns"
        assert copy.dataset_namespace == "custom-ds"
        assert copy.root_job_name == "custom.root"

    def test_pickled_copy_can_still_build_a_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        extender = OpenLineageExtender(use_sdk_defaults=True)

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        monkeypatch.setenv("OPENLINEAGE_DISABLED", "true")
        first = copy._get_client()
        second = copy._get_client()

        assert isinstance(first, OpenLineageClient)
        assert first is second


class TestOpenLineageExtenderLazyClientInit:
    """`_get_client()`'s lazy default-construction must be race-free across concurrent first calls."""

    def test_concurrent_first_calls_build_exactly_one_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        build_count = 0
        count_lock = threading.Lock()

        class _FakeOpenLineageClient:
            def __init__(self) -> None:
                nonlocal build_count
                time.sleep(0.05)
                with count_lock:
                    build_count += 1
                self.transport = None

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FakeOpenLineageClient)
        monkeypatch.setattr(atexit, "register", lambda *args, **kwargs: None)

        extender = OpenLineageExtender(use_sdk_defaults=True)
        thread_count = 16
        barrier = threading.Barrier(thread_count)
        results: list[Any] = [None] * thread_count

        def worker(index: int) -> None:
            barrier.wait(timeout=5)
            results[index] = extender._get_client()

        threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert build_count == 1
        assert all(result is results[0] for result in results)


class TestOpenLineageExtenderClose:
    """close() flushes the underlying OpenLineageClient/transport and registers an atexit hook
    only for a client this extender built itself; a caller-injected client is never touched by
    atexit, and closing before any client exists must not build one."""

    def test_close_delegates_to_injected_client_and_flushes_transport(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)

        result = extender.close()

        assert result is True
        assert transport.close_calls == 1

    def test_close_passes_timeout_argument_through_to_client_close(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = ol_capture
        extender = OpenLineageExtender(client=client)
        received: list[float] = []

        def fake_close(timeout: float = -1.0) -> bool:
            received.append(timeout)
            return True

        monkeypatch.setattr(client, "close", fake_close)

        extender.close(timeout=5.5)

        assert received == [5.5]

    def test_close_delegates_to_lazily_built_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeClientWithClose:
            def __init__(self) -> None:
                self.close_calls: list[float] = []

            def close(self, timeout: float = -1.0) -> bool:
                self.close_calls.append(timeout)
                return True

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FakeClientWithClose)
        extender = OpenLineageExtender(use_sdk_defaults=True)
        built: Any = extender._get_client()

        result = extender.close(timeout=9.0)

        assert result is True
        assert built.close_calls == [9.0]

    def test_close_without_a_built_client_is_noop_returning_true_and_never_constructs_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FailingClient:
            def __init__(self) -> None:
                raise AssertionError("OpenLineageClient must not be constructed as a side effect of close()")

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FailingClient)
        registered: list[Any] = []
        monkeypatch.setattr(atexit, "register", lambda func: registered.append(func))
        extender = OpenLineageExtender()

        result = extender.close()

        assert result is True
        assert extender._client is None
        assert registered == []

    def test_atexit_registered_exactly_once_when_client_lazily_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeClient:
            def close(self, timeout: float = -1.0) -> bool:
                return True

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FakeClient)
        registered: list[Any] = []
        monkeypatch.setattr(atexit, "register", lambda *args, **kwargs: registered.append(args[0]))
        extender = OpenLineageExtender(use_sdk_defaults=True)

        extender._get_client()
        extender._get_client()

        assert len(registered) == 1
        assert registered[0].__self__ is extender
        assert registered[0].__func__ is OpenLineageExtender.close

    def test_atexit_not_registered_when_client_injected_via_constructor(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = ol_capture
        registered: list[Any] = []
        monkeypatch.setattr(atexit, "register", lambda func: registered.append(func))
        extender = OpenLineageExtender(client=client)

        assert extender._get_client() is client
        extender.close()

        assert registered == []


class TestOpenLineageExtenderCloseIdempotencyAndReuse:
    """close() must be idempotent, unregister its own atexit hook, use a bounded atexit timeout,
    warn on incomplete flush, and reject reuse of a closed extender with a RuntimeError that the
    standard warning-only _CompositeExtender fallback degrades gracefully instead of silently
    emitting into a dead client."""

    def test_close_is_idempotent_only_invokes_client_close_once(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)

        first = extender.close()
        second = extender.close()

        assert first is True
        assert second is True
        assert transport.close_calls == 1

    def test_close_unregisters_atexit_hook_when_client_lazily_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeClient:
            def close(self, timeout: float = -1.0) -> bool:
                return True

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FakeClient)
        registered: list[Any] = []
        unregistered: list[Any] = []
        monkeypatch.setattr(atexit, "register", lambda *args, **kwargs: registered.append((args, kwargs)))
        monkeypatch.setattr(atexit, "unregister", lambda func: unregistered.append(func))
        extender = OpenLineageExtender(use_sdk_defaults=True)

        extender._get_client()
        extender.close()

        assert len(unregistered) == 1
        assert unregistered[0].__self__ is extender
        assert unregistered[0].__func__ is OpenLineageExtender.close

    def test_atexit_registered_close_uses_bounded_timeout_not_blocking_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeClient:
            def close(self, timeout: float = -1.0) -> bool:
                return True

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FakeClient)
        captured: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        monkeypatch.setattr(atexit, "register", lambda *args, **kwargs: captured.append((args, kwargs)))
        extender = OpenLineageExtender(use_sdk_defaults=True)

        extender._get_client()

        assert len(captured) == 1
        args, kwargs = captured[0]
        assert args[0].__self__ is extender
        assert args[0].__func__ is OpenLineageExtender.close
        if len(args) > 1:
            timeout = args[1]
        else:
            assert "timeout" in kwargs, "atexit.register must pass an explicit bounded timeout"
            timeout = kwargs["timeout"]
        assert isinstance(timeout, float)
        assert timeout != -1.0
        assert 0 < timeout < float("inf")

    def test_close_logs_warning_naming_extender_when_flush_incomplete(self, caplog: pytest.LogCaptureFixture) -> None:
        extender = OpenLineageExtender(client=OpenLineageClient(transport=_IncompleteFlushTransport()))

        with caplog.at_level(logging.WARNING):
            result = extender.close()

        assert result is False
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OpenLineageExtender" in message for message in warnings)

    def test_get_client_after_close_raises_runtime_error(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, _ = ol_capture
        extender = OpenLineageExtender(client=client)

        extender.close()

        with pytest.raises(RuntimeError):
            extender._get_client()

    def test_reuse_after_close_falls_back_to_func_and_logs_warning_via_composite(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, _ = ol_capture
        extender = OpenLineageExtender(client=client)
        extender.close()
        composite = _CompositeExtender([extender])
        sentinel = object()

        def func() -> object:
            return sentinel

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            with caplog.at_level(logging.WARNING):
                result = composite(func)

        assert result is sentinel
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OpenLineageExtender" in message for message in warnings)


class TestOpenLineageExtenderGetClientBoundary:
    def test_unconfigured_extender_returns_none_and_builds_no_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        build_count = 0

        class _CountingOpenLineageClient:
            def __init__(self) -> None:
                nonlocal build_count
                build_count += 1

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _CountingOpenLineageClient)

        extender = OpenLineageExtender()

        assert extender._get_client() is None
        assert build_count == 0


class TestOpenLineageExtenderPickledInertLogging:
    def test_pickled_copy_logs_its_own_inert_state(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, _ = ol_capture
        extender = OpenLineageExtender(client=client)
        extender._logged_inert = True

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        with caplog.at_level(logging.WARNING):
            with make_hook_context().activate():
                result = copy(lambda: 42)

        assert result == 42
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("OpenLineageExtender" in r.message and "inert" in r.message.lower() for r in warning_records), (
            warning_records
        )


class TestOpenLineageExtenderConcurrentInertLogging:
    def test_concurrent_first_calls_log_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        extender = OpenLineageExtender()
        original_info = openlineage_extender_module.logger.info

        def slow_info(msg: str, *args: Any, **kwargs: Any) -> None:
            time.sleep(0.05)
            original_info(msg, *args, **kwargs)

        monkeypatch.setattr(openlineage_extender_module.logger, "info", slow_info)

        thread_count = 32
        barrier = threading.Barrier(thread_count)

        def worker() -> None:
            barrier.wait(timeout=5)
            with make_hook_context().activate():
                extender(lambda: None)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(thread_count)]
        with caplog.at_level(logging.INFO):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        inert_records = [
            r for r in caplog.records if "OpenLineageExtender" in r.message and "inert" in r.message.lower()
        ]
        assert len(inert_records) == 1, inert_records


class TestOpenLineageExtenderStartEvent:
    """One RunEvent(START), emitted before func runs, with empty inputs/outputs."""

    def test_job_namespace_and_name_use_defaults(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = make_hook_context()
        extender = OpenLineageExtender(client=client)

        with context.activate():
            extender(lambda: None)

        start_event = transport.events[0]
        assert start_event.job.namespace == "mloda"
        assert start_event.job.name == context.feature_group_class
        assert start_event.inputs == []
        assert start_event.outputs == []

    def test_job_namespace_uses_constructor_override(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = make_hook_context()
        extender = OpenLineageExtender(client=client, job_namespace="custom-ns")

        with context.activate():
            extender(lambda: None)

        assert transport.events[0].job.namespace == "custom-ns"


class TestOpenLineageExtenderCompleteEvent:
    """After a successful func: outputs (one per feature name), then RunEvent(COMPLETE)."""

    def test_schema_facet_present_when_result_exposes_pyarrow_schema(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = make_hook_context(feature_names=("value_int", "value_str"))
        extender = OpenLineageExtender(client=client, dataset_namespace="custom-ds")
        table = pa.table({"value_int": [1, 2, 3], "value_str": ["a", "b", "c"]})

        with context.activate():
            extender(lambda: table)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        datasets_by_name = {ds.name: ds for ds in complete_event.outputs}

        int_dataset = datasets_by_name["value_int"]
        assert int_dataset.namespace == "custom-ds"
        int_facet = int_dataset.facets
        assert int_facet is not None
        int_schema = int_facet.get("schema")
        assert isinstance(int_schema, schema_dataset.SchemaDatasetFacet)
        assert int_schema.fields is not None
        assert [f.name for f in int_schema.fields] == ["value_int"]
        assert [f.type for f in int_schema.fields] == ["int64"]

        str_dataset = datasets_by_name["value_str"]
        assert str_dataset.namespace == "custom-ds"
        str_facet = str_dataset.facets
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
        context = make_hook_context(feature_names=("value_int", "value_str"))
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
        context = make_hook_context(feature_names=("0", "1"))
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
        context = make_hook_context(feature_names=("x",))
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
        assert len(schema_facet.fields) == 1
        for f in schema_facet.fields:
            assert f.name == "x"
            assert f.type == "int64"

    def test_schema_facet_types_come_from_schema_fields_for_spark_shaped_result(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        """schema.fields must win over schema.names/schema.types when a fake result exposes both shapes."""

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
            def __init__(self, fields: list[_StructField], types: list[tuple[str, str]]) -> None:
                self.fields = fields
                self.names = [f.name for f in fields]
                self.types = types

        class _SparkFrame:
            def __init__(self, columns: list[str], dtypes: list[tuple[str, str]], schema: _StructType) -> None:
                self.columns = columns
                self.dtypes = dtypes
                self.schema = schema

        client, transport = ol_capture
        context = make_hook_context(feature_names=("value_int", "value_str"))
        extender = OpenLineageExtender(client=client)
        frame = _SparkFrame(
            columns=["value_int", "value_str"],
            dtypes=[("value_int", "bigint"), ("value_str", "string")],
            schema=_StructType(
                fields=[
                    _StructField("value_int", _DataType("LongType()")),
                    _StructField("value_str", _DataType("StringType()")),
                ],
                types=[("value_int", "bigint"), ("value_str", "string")],
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
        context = make_hook_context()
        extender = OpenLineageExtender(client=client)

        with context.activate():
            extender(lambda: 42)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        output = complete_event.outputs[0]
        assert output.facets is not None
        assert "schema" not in output.facets

    def test_collect_schema_used_without_resolving_schema_property(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        """A lazy frame's `schema` property must never be resolved when `collect_schema()` is available."""

        class _LazyFrame:
            def collect_schema(self) -> dict[str, str]:
                return {"value_int": "Int64"}

            @property
            def schema(self) -> Any:
                raise AssertionError("schema must not be resolved")

        client, transport = ol_capture
        context = make_hook_context(feature_names=("value_int",))
        extender = OpenLineageExtender(client=client)

        with context.activate():
            extender(lambda: _LazyFrame())

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        output = complete_event.outputs[0]
        assert output.facets is not None
        schema_facet = output.facets.get("schema")
        assert isinstance(schema_facet, schema_dataset.SchemaDatasetFacet)
        assert schema_facet.fields is not None
        assert [f.type for f in schema_facet.fields] == ["Int64"]

    def test_schema_facet_present_for_pandas_frame_with_column_named_schema(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        """A pandas column literally named "schema" must not hijack schema inference for other columns."""
        import pandas as pd

        client, transport = ol_capture
        context = make_hook_context(feature_names=("value_int",))
        extender = OpenLineageExtender(client=client)
        frame = pd.DataFrame({"schema": [1], "value_int": [2]})

        with context.activate():
            extender(lambda: frame)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        output = complete_event.outputs[0]
        assert output.name == "value_int"
        assert output.facets is not None
        schema_facet = output.facets.get("schema")
        assert isinstance(schema_facet, schema_dataset.SchemaDatasetFacet)
        assert schema_facet.fields is not None
        assert [f.name for f in schema_facet.fields] == ["value_int"]
        assert [f.type for f in schema_facet.fields] == ["int64"]


class TestOpenLineageExtenderParentRunFacet:
    """ParentRunFacet is present iff context.run_id is not None."""

    def test_parent_facet_present_with_default_namespace_and_root_job_name(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        run_id = str(uuid.uuid4())
        context = make_hook_context(run_id=run_id)
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
        context = make_hook_context(run_id=run_id)
        extender = OpenLineageExtender(client=client, job_namespace="custom-ns", root_job_name="custom.root")

        with context.activate():
            extender(lambda: None)

        event = transport.events[0]
        assert event.run.facets is not None
        parent = event.run.facets.get("parent")
        assert isinstance(parent, parent_run.ParentRunFacet)
        assert parent.job.namespace == "custom-ns"
        assert parent.job.name == "custom.root"


class TestOpenLineageExtenderInputDataLoadCorrelation:
    """INPUT_DATA_LOAD fires nested inside an already-open CALCULATE_FEATURE invocation."""

    def test_recorded_input_uses_dataset_namespace_and_data_source_facet(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client, dataset_namespace="custom-ds")
        outer_context = make_hook_context()
        inner_context = make_hook_context(
            hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="s3://bucket/key.parquet"
        )

        def inner_func() -> str:
            return "loaded-data"

        def outer_func() -> str:
            with inner_context.activate():
                extender(inner_func)
            return "calculate-result"

        with outer_context.activate():
            extender(outer_func)

        complete_event = transport.events[1]
        assert complete_event.inputs is not None
        input_dataset = complete_event.inputs[0]
        assert input_dataset.namespace == "custom-ds"

        from openlineage.client.facet_v2 import datasource_dataset

        assert input_dataset.facets is not None
        data_source_facet = input_dataset.facets["dataSource"]
        assert isinstance(data_source_facet, datasource_dataset.DatasourceDatasetFacet)
        assert data_source_facet.name == "s3://bucket/key.parquet"

    def test_input_data_load_without_enclosing_calculate_does_not_raise_or_emit(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="standalone")

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


class TestOpenLineageExtenderPerInstanceAttribution:
    """Nested INPUT_DATA_LOAD must attribute to its own enclosing instance, not any open one."""

    def test_composite_of_two_extenders_each_see_their_own_single_input(self) -> None:
        transport_a = RecordingTransport()
        transport_b = RecordingTransport()
        extender_a = OpenLineageExtender(client=OpenLineageClient(transport=transport_a))
        extender_b = OpenLineageExtender(client=OpenLineageClient(transport=transport_b))
        composite = _CompositeExtender([extender_a, extender_b])

        inner_context = make_hook_context(
            hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="s3://bucket/key.parquet"
        )

        def inner_loader() -> str:
            return "loaded-data"

        def calculate_body() -> str:
            with inner_context.activate():
                composite(inner_loader)
            return "calculated"

        with make_hook_context().activate():
            composite(calculate_body)

        for transport in (transport_a, transport_b):
            complete_events = [e for e in transport.events if e.eventType == RunState.COMPLETE]
            assert len(complete_events) == 1
            inputs = complete_events[0].inputs
            assert inputs is not None
            assert len(inputs) == 1
            assert inputs[0].name == "s3://bucket/key.parquet"


class TestOpenLineageExtenderInputDedupe:
    """Loading the same data_access_identity twice must not duplicate the OpenLineage input."""

    def test_repeated_input_data_load_produces_single_input(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        inner_context = make_hook_context(
            hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="s3://bucket/key.parquet"
        )

        def inner_loader() -> str:
            return "loaded-data"

        def calculate_body() -> str:
            with inner_context.activate():
                extender(inner_loader)
            with inner_context.activate():
                extender(inner_loader)
            return "calculated"

        with make_hook_context().activate():
            extender(calculate_body)

        complete_event = transport.events[-1]
        assert complete_event.eventType == RunState.COMPLETE
        assert complete_event.inputs is not None
        assert len(complete_event.inputs) == 1


class TestOpenLineageExtenderRunAll:
    """End-to-end wiring through mloda.user.mloda.run_all: RunEvents carry the real feature group's job name."""

    def test_run_all_events_use_feature_group_qualified_name_as_job_name(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture

        run_value_int(OpenLineageExtender(client=client))

        expected_job_name = f"{PyArrowDataOpsTestDataCreator.__module__}.{PyArrowDataOpsTestDataCreator.__qualname__}"
        assert any(e.job.name == expected_job_name for e in transport.events)
