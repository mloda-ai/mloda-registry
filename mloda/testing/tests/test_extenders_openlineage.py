"""Self-tests for mloda.testing.extenders.openlineage helpers, plus a negative test proving the mixin's default
own_failure() detects a fault."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("openlineage.client")

from mloda.steward import Extender, ExtenderHook, HookContext
from openlineage.client.client import OpenLineageClient
from openlineage.client.event_v2 import InputDataset, Job, Run, RunEvent, RunState
from openlineage.client.facet_v2 import parent_run

from mloda.community.extenders.openlineage import OpenLineageExtender
from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.openlineage import (
    FileTransport,
    OpenLineageExtenderTestMixin,
    RecordingTransport,
    make_recording_client,
)

_PRODUCER = "mloda-testing-probe-openlineage"
_NESTED_JOB_SUFFIX = ".validate_output_feature"


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

    def test_ambient_sink_captured_reads_events_from_the_spy_transports(self) -> None:
        client, transport = make_recording_client()
        event = _build_run_event()
        client.emit(event)

        mixin = OpenLineageExtenderTestMixin()
        assert mixin.ambient_sink_captured([transport]) == [event]
        assert mixin.ambient_sink_captured([RecordingTransport()]) == []

    def test_calculate_run_events_defaults_to_identity(self) -> None:
        events = [_build_run_event(), _build_run_event()]

        mixin = OpenLineageExtenderTestMixin()
        assert mixin.calculate_run_events(events) == events
        assert mixin.calculate_run_events([]) == []


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


class _NestedRunProbeOpenLineageExtender(Extender):
    """Delegates to OpenLineageExtender, then emits one extra nested run after every calculate run, like a validation
    run would."""

    def __init__(self, client: OpenLineageClient, raise_on_error: bool = False) -> None:
        self.raise_on_error = raise_on_error
        self._client = client
        self._emitter = OpenLineageExtender(client=client, raise_on_error=raise_on_error)

    def wraps(self) -> set[ExtenderHook]:
        return self._emitter.wraps()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        result = self._emitter(func, *args, **kwargs)
        if context is not None and context.hook == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE:
            self._emit_nested_run(context)
        return result

    def _emit_nested_run(self, context: HookContext) -> None:
        job = Job(namespace="mloda", name=f"{context.feature_group_class}{_NESTED_JOB_SUFFIX}")
        parent = parent_run.ParentRunFacet(
            run=parent_run.Run(runId=str(uuid.uuid4())),
            job=parent_run.Job(namespace="mloda", name="probe.nested_parent"),
            producer=_PRODUCER,
        )
        run = Run(runId=str(uuid.uuid4()), facets={"parent": parent})
        for state, inputs in (
            (RunState.START, []),
            (RunState.COMPLETE, [InputDataset(namespace="mloda", name="nested_probe_input")]),
        ):
            self._client.emit(
                RunEvent(
                    eventType=state,
                    eventTime=_now_iso(),
                    run=run,
                    job=job,
                    producer=_PRODUCER,
                    inputs=inputs,
                    outputs=[],
                )
            )


class _NestedRunHost(OpenLineageExtenderTestMixin):
    """Keeps the default calculate_run_events, so the probe's nested run is counted by the mixin assertions."""

    @classmethod
    def extender_class(cls) -> type[Extender]:
        return _NestedRunProbeOpenLineageExtender

    def make_openlineage_extender(self, client: OpenLineageClient, *, raise_on_error: bool | None = None) -> Extender:
        if raise_on_error is None:
            return _NestedRunProbeOpenLineageExtender(client=client)
        return _NestedRunProbeOpenLineageExtender(client=client, raise_on_error=raise_on_error)


class _FilteringNestedRunHost(_NestedRunHost):
    def calculate_run_events(self, events: list[RunEvent]) -> list[RunEvent]:
        return [event for event in events if not event.job.name.endswith(_NESTED_JOB_SUFFIX)]


class _RealOpenLineageExtenderHost(OpenLineageExtenderTestMixin):
    @classmethod
    def extender_class(cls) -> type[Extender]:
        return OpenLineageExtender

    def make_openlineage_extender(self, client: OpenLineageClient, *, raise_on_error: bool | None = None) -> Extender:
        if raise_on_error is None:
            return OpenLineageExtender(client=client)
        return OpenLineageExtender(client=client, raise_on_error=raise_on_error)


_COUNT_SENSITIVE_MIXIN_TESTS: list[Any] = [
    pytest.param(
        lambda host, tmp_path: host.test_openlineage_run_all_derived_feature_reports_input_feature(),
        id="derived-feature-reports-input-feature",
    ),
    pytest.param(
        lambda host, tmp_path: host.test_openlineage_run_all_reader_backed_feature_reports_input_dataset(tmp_path),
        id="reader-backed-feature-reports-input-dataset",
    ),
    pytest.param(
        lambda host, tmp_path: host.test_openlineage_run_all_events_share_one_parent_run_id(),
        id="events-share-one-parent-run-id",
    ),
]


class TestCalculateRunEventsHook:
    """calculate_run_events lets a host drop non-calculate runs before the count-sensitive run_all assertions."""

    @pytest.mark.parametrize("mixin_test", _COUNT_SENSITIVE_MIXIN_TESTS)
    def test_nested_run_breaks_the_assertion_without_a_filtering_override(
        self, mixin_test: Callable[[OpenLineageExtenderTestMixin, Path], None], tmp_path: Path
    ) -> None:
        with pytest.raises(AssertionError):
            mixin_test(_NestedRunHost(), tmp_path)

    @pytest.mark.parametrize("mixin_test", _COUNT_SENSITIVE_MIXIN_TESTS)
    def test_filtering_override_restores_the_assertion(
        self, mixin_test: Callable[[OpenLineageExtenderTestMixin, Path], None], tmp_path: Path
    ) -> None:
        mixin_test(_FilteringNestedRunHost(), tmp_path)


class TestQueryStringIdentityContract:
    """Proves the query-string contract test fails for an extender that publishes the raw identity or drops it."""

    @pytest.mark.parametrize(
        ("host_class", "resolver", "message"),
        [
            pytest.param(
                _RealOpenLineageExtenderHost,
                lambda args, context_identity: context_identity,
                "URI query string reached an event",
                id="publishes-raw-identity",
            ),
            pytest.param(
                _NestedRunHost,
                lambda args, context_identity: None,
                "not attributed as an input",
                id="drops-identity-despite-other-inputs",
            ),
        ],
    )
    def test_non_compliant_identity_handling_is_detected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        host_class: type[OpenLineageExtenderTestMixin],
        resolver: Callable[..., str | None],
        message: str,
    ) -> None:
        monkeypatch.setattr(
            "mloda.community.extenders.openlineage.openlineage_extender.resolve_data_access_identity",
            resolver,
        )

        with pytest.raises(AssertionError, match=message):
            host_class().test_openlineage_input_data_load_query_string_never_reaches_events()
