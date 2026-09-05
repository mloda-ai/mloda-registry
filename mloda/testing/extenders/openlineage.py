"""In-memory OpenLineage transport capture plus a contract mixin for extenders that emit RunEvents."""

from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager
from typing import Any
from unittest.mock import patch

import pytest
from mloda.steward import Extender, ExtenderHook
from openlineage.client.client import Event, OpenLineageClient
from openlineage.client.event_v2 import InputDataset, OutputDataset, RunEvent, RunState
from openlineage.client.facet_v2 import parent_run
from openlineage.client.serde import Serde
from openlineage.client.transport.transport import Config, Transport

from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.runners import expected_value_int, run_value_int


class RecordingTransport(Transport):
    """Records every emitted RunEvent in memory instead of sending it anywhere."""

    kind = "recording"
    config_class = Config

    def __init__(self, config: Config | None = None) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: Event) -> None:
        if not isinstance(event, RunEvent):
            raise TypeError(f"RecordingTransport only records RunEvent, got {type(event).__name__}")
        self.events.append(event)


def make_recording_client() -> tuple[OpenLineageClient, RecordingTransport]:
    """An OpenLineageClient wired to a fresh RecordingTransport."""
    transport = RecordingTransport()
    return OpenLineageClient(transport=transport), transport


class OpenLineageExtenderTestMixin(ExtenderContractTestMixin):
    """Contract for extenders that emit OpenLineage RunEvents. Host provides extender_class and make_openlineage_extender."""

    def make_openlineage_extender(self, client: OpenLineageClient, *, raise_on_error: bool | None = None) -> Extender:
        raise NotImplementedError

    @classmethod
    def raise_on_error_default(cls) -> bool:
        return False

    def make_extender(self, *, raise_on_error: bool | None = None) -> Extender:
        client, _ = make_recording_client()
        return self.make_openlineage_extender(client, raise_on_error=raise_on_error)

    def own_failure(self) -> AbstractContextManager[Any]:
        return patch.object(OpenLineageClient, "emit", side_effect=RuntimeError("openlineage instrumentation boom"))

    def pickled_copy_environment(self) -> AbstractContextManager[Any]:
        return patch.dict(os.environ, {"OPENLINEAGE_DISABLED": "true"})

    def test_openlineage_no_ambient_context_emits_nothing(self) -> None:
        client, transport = make_recording_client()
        extender = self.make_openlineage_extender(client)
        calls = 0

        def func() -> int:
            nonlocal calls
            calls += 1
            return 42

        result = extender(func)

        assert result == 42
        assert calls == 1
        assert transport.events == []

    def test_openlineage_success_emits_start_then_complete(self) -> None:
        client, transport = make_recording_client()
        extender = self.make_openlineage_extender(client)

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            extender(lambda: None)

        assert len(transport.events) == 2
        assert transport.events[0].eventType == RunState.START
        assert transport.events[1].eventType == RunState.COMPLETE

    def test_openlineage_start_event_precedes_wrapped_call(self) -> None:
        client, transport = make_recording_client()
        extender = self.make_openlineage_extender(client)
        observed: list[int] = []

        def func() -> None:
            observed.append(len(transport.events))

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            extender(func)

        assert observed == [1]

    def test_openlineage_complete_event_has_output_per_feature_name(self) -> None:
        client, transport = make_recording_client()
        extender = self.make_openlineage_extender(client)
        feature_names = ("value_int", "value_str")

        with make_hook_context(
            hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, feature_names=feature_names
        ).activate():
            extender(lambda: None)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        assert [output.name for output in complete_event.outputs] == list(feature_names)
        for output in complete_event.outputs:
            assert isinstance(output, OutputDataset)

    def test_openlineage_wrapped_failure_emits_start_then_fail_and_propagates(self) -> None:
        client, transport = make_recording_client()
        extender = self.make_openlineage_extender(client)

        def func() -> None:
            raise RuntimeError("inner boom")

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            with pytest.raises(RuntimeError, match="inner boom"):
                extender(func)

        assert len(transport.events) == 2
        assert transport.events[0].eventType == RunState.START
        assert transport.events[1].eventType == RunState.FAIL
        assert transport.events[1].outputs == []

    def test_openlineage_base_exception_emits_fail_and_propagates(self) -> None:
        client, transport = make_recording_client()
        extender = self.make_openlineage_extender(client)

        class _Boom(BaseException):
            pass

        def func() -> None:
            raise _Boom("base exception boom")

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            with pytest.raises(_Boom):
                extender(func)

        assert transport.events[-1].eventType == RunState.FAIL

    def test_openlineage_fail_emit_error_never_masks_wrapped_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, transport = make_recording_client()
        extender = self.make_openlineage_extender(client)
        original_emit = client.emit
        calls = 0

        def flaky_emit(event: Event) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                original_emit(event)
                return
            raise RuntimeError("transport boom")

        monkeypatch.setattr(client, "emit", flaky_emit)

        def func() -> None:
            raise ValueError("inner boom")

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            with pytest.raises(ValueError, match="inner boom"):
                extender(func)

    def test_openlineage_wrapped_failure_logs_warning_naming_extender(self, caplog: pytest.LogCaptureFixture) -> None:
        client, _ = make_recording_client()
        extender = self.make_openlineage_extender(client)

        def func() -> None:
            raise RuntimeError("inner boom")

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            with caplog.at_level(logging.WARNING):
                with pytest.raises(RuntimeError, match="inner boom"):
                    extender(func)

        extender_name = self.extender_class().__name__
        assert any(extender_name in message and "inner boom" in message for message in caplog.messages)

    def test_openlineage_exception_message_never_leaks_into_events(self) -> None:
        client, transport = make_recording_client()
        extender = self.make_openlineage_extender(client)
        marker = "SENSITIVE_ROW_VALUE_xyz123"

        def func() -> None:
            raise ValueError(f"invalid value found: {marker}")

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            with pytest.raises(ValueError):
                extender(func)

        for event in transport.events:
            assert marker not in Serde.to_json(event)

    def test_openlineage_parent_facet_run_id_matches_context(self) -> None:
        client, transport = make_recording_client()
        extender = self.make_openlineage_extender(client)
        run_id = "018f1e4a-7c3b-7c3b-8c3b-1234567890ab"

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, run_id=run_id).activate():
            extender(lambda: None)

        first_event = transport.events[0]
        assert first_event.run.facets is not None
        parent = first_event.run.facets.get("parent")
        assert isinstance(parent, parent_run.ParentRunFacet)
        assert parent.run.runId == run_id

        client_no_run, transport_no_run = make_recording_client()
        extender_no_run = self.make_openlineage_extender(client_no_run)
        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, run_id=None).activate():
            extender_no_run(lambda: None)

        first_event_no_run = transport_no_run.events[0]
        run_facets = first_event_no_run.run.facets or {}
        assert "parent" not in run_facets

    def test_openlineage_nested_input_data_load_becomes_input(self) -> None:
        client, transport = make_recording_client()
        extender = self.make_openlineage_extender(client)
        if ExtenderHook.INPUT_DATA_LOAD not in extender.wraps():
            pytest.skip("extender does not wrap INPUT_DATA_LOAD")

        inner_context = make_hook_context(
            hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="s3://bucket/key.parquet"
        )

        def outer_func() -> None:
            with inner_context.activate():
                extender(lambda: "loaded-data")

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            extender(outer_func)

        complete_event = transport.events[1]
        assert complete_event.eventType == RunState.COMPLETE
        assert complete_event.inputs is not None
        assert len(complete_event.inputs) == 1
        input_dataset = complete_event.inputs[0]
        assert isinstance(input_dataset, InputDataset)
        assert input_dataset.name == "s3://bucket/key.parquet"
        assert len(transport.events) == 2

    def test_openlineage_fail_event_carries_nested_inputs(self) -> None:
        client, transport = make_recording_client()
        extender = self.make_openlineage_extender(client)
        if ExtenderHook.INPUT_DATA_LOAD not in extender.wraps():
            pytest.skip("extender does not wrap INPUT_DATA_LOAD")

        inner_context = make_hook_context(
            hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="s3://bucket/key.parquet"
        )

        def outer_func() -> None:
            with inner_context.activate():
                extender(lambda: "loaded-data")
            raise RuntimeError("inner boom")

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            with pytest.raises(RuntimeError, match="inner boom"):
                extender(outer_func)

        fail_event = transport.events[-1]
        assert fail_event.eventType == RunState.FAIL
        assert fail_event.inputs is not None
        assert [i.name for i in fail_event.inputs] == ["s3://bucket/key.parquet"]

    def test_openlineage_completion_emit_failure_keeps_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, _ = make_recording_client()
        extender = self.make_openlineage_extender(client)
        original_emit = client.emit
        calls = 0

        def flaky_emit(event: Event) -> None:
            nonlocal calls
            calls += 1
            original_emit(event)
            if calls == 2:
                raise RuntimeError("transport boom")

        monkeypatch.setattr(client, "emit", flaky_emit)
        func_calls = 0

        def func() -> list[int]:
            nonlocal func_calls
            func_calls += 1
            return [1, 2, 3]

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            result = extender(func)

        assert result == [1, 2, 3]
        assert func_calls == 1

    def test_openlineage_run_all_events_share_one_parent_run_id(self) -> None:
        client, transport = make_recording_client()
        assert run_value_int(self.make_openlineage_extender(client)) == expected_value_int()

        event_types = {event.eventType for event in transport.events}
        assert RunState.START in event_types
        assert RunState.COMPLETE in event_types

        parent_run_ids = set()
        for event in transport.events:
            run_facets = event.run.facets or {}
            parent = run_facets.get("parent")
            if isinstance(parent, parent_run.ParentRunFacet):
                parent_run_ids.add(parent.run.runId)
        assert len(parent_run_ids) == 1
