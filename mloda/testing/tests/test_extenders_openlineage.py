"""Self-tests for mloda.testing.extenders.openlineage helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

pytest.importorskip("openlineage.client")

from openlineage.client.client import OpenLineageClient
from openlineage.client.event_v2 import Job, Run, RunEvent, RunState

from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.openlineage import OpenLineageExtenderTestMixin, RecordingTransport, make_recording_client


def _build_run_event() -> RunEvent:
    return RunEvent(
        eventType=RunState.START,
        eventTime=datetime.now(timezone.utc).isoformat(),
        run=Run(runId=str(uuid.uuid4())),
        job=Job(namespace="mloda", name="job"),
        producer="test",
        inputs=[],
        outputs=[],
    )


class TestRecordingTransport:
    def test_records_run_event_passed_to_emit(self) -> None:
        transport = RecordingTransport()
        event = _build_run_event()

        transport.emit(event)

        assert transport.events == [event]

    def test_raises_type_error_for_non_run_event(self) -> None:
        transport = RecordingTransport()
        with pytest.raises(TypeError):
            transport.emit(object())  # type: ignore[arg-type]


class TestMakeRecordingClient:
    def test_emit_lands_event_in_returned_transport(self) -> None:
        client, transport = make_recording_client()
        event = _build_run_event()

        client.emit(event)

        assert isinstance(client, OpenLineageClient)
        assert transport.events == [event]


class TestOpenLineageExtenderTestMixinShape:
    def test_is_extender_contract_subclass(self) -> None:
        assert issubclass(OpenLineageExtenderTestMixin, ExtenderContractTestMixin)

    def test_raise_on_error_default_is_false(self) -> None:
        assert OpenLineageExtenderTestMixin.raise_on_error_default() is False

    @pytest.mark.parametrize(
        "name",
        [
            "test_openlineage_no_ambient_context_emits_nothing",
            "test_openlineage_success_emits_start_then_complete",
            "test_openlineage_start_event_precedes_wrapped_call",
            "test_openlineage_complete_event_has_output_per_feature_name",
            "test_openlineage_wrapped_failure_emits_start_then_fail_and_propagates",
            "test_openlineage_base_exception_emits_fail_and_propagates",
            "test_openlineage_fail_emit_error_never_masks_wrapped_exception",
            "test_openlineage_wrapped_failure_logs_warning_naming_extender",
            "test_openlineage_exception_message_never_leaks_into_events",
            "test_openlineage_parent_facet_run_id_matches_context",
            "test_openlineage_nested_input_data_load_becomes_input",
            "test_openlineage_fail_event_carries_nested_inputs",
            "test_openlineage_completion_emit_failure_keeps_result",
            "test_openlineage_run_all_events_share_one_parent_run_id",
        ],
    )
    def test_openlineage_test_methods_exist(self, name: str) -> None:
        assert hasattr(OpenLineageExtenderTestMixin, name)
