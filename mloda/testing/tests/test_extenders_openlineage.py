"""Self-tests for mloda.testing.extenders.openlineage helpers, plus a negative test proving the mixin's default
own_failure() detects a fault."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("openlineage.client")

from mloda.steward import Extender, ExtenderHook, HookContext
from openlineage.client.client import OpenLineageClient
from openlineage.client.event_v2 import Job, Run, RunEvent, RunState

from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.openlineage import (
    FileTransport,
    OpenLineageExtenderTestMixin,
    RecordingTransport,
    make_recording_client,
)

_PRODUCER = "mloda-testing-probe-openlineage"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


class TestFileTransport:
    def test_emits_event_type_line_to_marker_file(self, tmp_path: Path) -> None:
        marker_path = tmp_path / "marker.txt"
        transport = FileTransport(marker_path)
        event = _build_run_event()

        transport.emit(event)

        assert event.eventType is not None
        assert marker_path.read_text() == f"{event.eventType.value}\n"

    def test_raises_type_error_for_non_run_event(self, tmp_path: Path) -> None:
        transport = FileTransport(tmp_path / "marker.txt")
        with pytest.raises(TypeError):
            transport.emit(object())  # type: ignore[arg-type]


class TestMakeRecordingClient:
    def test_emit_lands_event_in_returned_transport(self) -> None:
        client, transport = make_recording_client()
        event = _build_run_event()

        client.emit(event)

        assert isinstance(client, OpenLineageClient)
        assert transport.events == [event]

    def test_records_events_even_when_openlineage_is_disabled_in_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENLINEAGE_DISABLED", "true")

        client, transport = make_recording_client()
        event = _build_run_event()
        client.emit(event)

        assert client.transport is transport
        assert transport.events == [event]

    def test_records_events_even_when_openlineage_config_filters_are_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "openlineage.yml"
        config_path.write_text('filters:\n  - type: regex\n    regex: ".*"\n', encoding="utf-8")
        monkeypatch.setenv("OPENLINEAGE_CONFIG", str(config_path))

        client, transport = make_recording_client()
        event = _build_run_event()
        client.emit(event)

        assert transport.events == [event]


class TestOpenLineageExtenderTestMixinShape:
    def test_is_extender_contract_subclass(self) -> None:
        assert issubclass(OpenLineageExtenderTestMixin, ExtenderContractTestMixin)

    def test_raise_on_error_default_is_false(self) -> None:
        assert OpenLineageExtenderTestMixin.raise_on_error_default() is False


class _DirectTransportProbeOpenLineageExtender(Extender):
    """Emits straight to the RecordingTransport, bypassing OpenLineageClient.emit entirely."""

    def __init__(self, client: OpenLineageClient | None = None, raise_on_error: bool = False) -> None:
        self.raise_on_error = raise_on_error
        self._transport = client.transport if client is not None else RecordingTransport()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_transport"] = None
        return state

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        job_name = context.feature_group_class if context is not None else "unknown"
        job = Job(namespace="probe", name=job_name)
        run = Run(runId=str(uuid.uuid4()))

        self._transport.emit(
            RunEvent(
                eventType=RunState.START,
                eventTime=_now_iso(),
                run=run,
                job=job,
                producer=_PRODUCER,
                inputs=[],
                outputs=[],
            )
        )
        result = func(*args, **kwargs)
        self._transport.emit(
            RunEvent(
                eventType=RunState.COMPLETE,
                eventTime=_now_iso(),
                run=run,
                job=job,
                producer=_PRODUCER,
                inputs=[],
                outputs=[],
            )
        )
        return result


class TestOwnFailureDefaultDetectsNoFault:
    """Proves the chained own_failure() test is no longer vacuous for a probe that never touches the client."""

    def test_default_own_failure_fails_loudly_when_nothing_is_faulted(self, caplog: pytest.LogCaptureFixture) -> None:
        class _Host(OpenLineageExtenderTestMixin):
            @classmethod
            def extender_class(cls) -> type[Extender]:
                return _DirectTransportProbeOpenLineageExtender

            def make_openlineage_extender(
                self, client: OpenLineageClient, *, raise_on_error: bool | None = None
            ) -> Extender:
                if raise_on_error is None:
                    return _DirectTransportProbeOpenLineageExtender(client=client)
                return _DirectTransportProbeOpenLineageExtender(client=client, raise_on_error=raise_on_error)

        with pytest.raises(AssertionError, match="own_failure"):
            _Host().test_contract_own_failure_does_not_stop_chained_extender(caplog)
