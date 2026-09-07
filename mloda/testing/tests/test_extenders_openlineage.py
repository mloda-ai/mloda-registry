"""Self-tests for mloda.testing.extenders.openlineage helpers, plus a minimal probe extender that
exercises the full OpenLineageExtenderTestMixin contract independently of the real registry extender."""

from __future__ import annotations

import itertools
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("openlineage.client")

from mloda.steward import Extender, ExtenderHook, HookContext
from openlineage.client.client import OpenLineageClient
from openlineage.client.event_v2 import InputDataset, Job, OutputDataset, Run, RunEvent, RunState
from openlineage.client.facet_v2 import parent_run

from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.openlineage import OpenLineageExtenderTestMixin, RecordingTransport, make_recording_client

logger = logging.getLogger(__name__)

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


# Process-local stash for the probe's injected client, mirroring OpenLineageExtender's pid-scoped token resolution.
_probe_openlineage_registry: dict[str, OpenLineageClient] = {}
_probe_openlineage_token_ids = itertools.count()


class _ProbeOpenLineageExtender(Extender):
    """Minimal OpenLineage probe: START/COMPLETE|FAIL|ABORT per calculate, correlating nested input loads."""

    def __init__(
        self,
        client: OpenLineageClient | None = None,
        raise_on_error: bool = False,
        use_sdk_defaults: bool = False,
    ) -> None:
        self.raise_on_error = raise_on_error
        self.use_sdk_defaults = use_sdk_defaults
        self._client = client
        self._open_inputs: list[InputDataset] | None = None
        self._logged_inert = False
        self._token: str | None = None
        if client is not None:
            self._token = f"{os.getpid()}:{next(_probe_openlineage_token_ids)}"
            _probe_openlineage_registry[self._token] = client

    def _get_client(self) -> OpenLineageClient:
        if self._client is None:
            self._client = OpenLineageClient()
        return self._client

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_client"] = None
        state["_open_inputs"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self._token is not None and self._token.split(":", 1)[0] == str(os.getpid()):
            self._client = _probe_openlineage_registry.get(self._token)

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        if self._client is None and not self.use_sdk_defaults:
            if not self._logged_inert:
                logger.info("_ProbeOpenLineageExtender is inert: no injected client and use_sdk_defaults is False")
                self._logged_inert = True
            return func(*args, **kwargs)

        context = HookContext.current()
        if context is None:
            return func(*args, **kwargs)
        if context.hook == ExtenderHook.INPUT_DATA_LOAD:
            return self._call_input_data_load(context, func, *args, **kwargs)
        return self._call_calculate(context, func, *args, **kwargs)

    def _call_input_data_load(self, context: HookContext, func: Any, *args: Any, **kwargs: Any) -> Any:
        if self._open_inputs is not None and context.data_access_identity is not None:
            already_present = any(i.name == context.data_access_identity for i in self._open_inputs)
            if not already_present:
                self._open_inputs.append(InputDataset(namespace="probe", name=context.data_access_identity))
        return func(*args, **kwargs)

    def _call_calculate(self, context: HookContext, func: Any, *args: Any, **kwargs: Any) -> Any:
        run_facets: dict[str, Any] = {}
        if context.run_id is not None:
            run_facets["parent"] = parent_run.ParentRunFacet(
                run=parent_run.Run(runId=context.run_id),
                job=parent_run.Job(namespace="probe", name="probe.run_all"),
                producer=_PRODUCER,
            )
        job = Job(namespace="probe", name=context.feature_group_class)
        run = Run(runId=str(uuid.uuid4()), facets=run_facets)

        # Unguarded on purpose: this must propagate naturally so _CompositeExtender's raise_on_error
        # fallback machinery sees the real failure and never double-invokes func.
        self._get_client().emit(
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

        previous_inputs = self._open_inputs
        self._open_inputs = []
        current_inputs: list[InputDataset] = []
        try:
            result = func(*args, **kwargs)
            current_inputs = list(self._open_inputs)
        except BaseException as exc:
            event_state = RunState.FAIL if isinstance(exc, Exception) else RunState.ABORT
            try:
                self._get_client().emit(
                    RunEvent(
                        eventType=event_state,
                        eventTime=_now_iso(),
                        run=run,
                        job=job,
                        producer=_PRODUCER,
                        inputs=list(self._open_inputs),
                        outputs=[],
                    )
                )
            except Exception as emit_exc:
                logger.warning(
                    "_ProbeOpenLineageExtender failed to emit %s event: %s: %s",
                    event_state.name,
                    type(emit_exc).__name__,
                    emit_exc,
                )
            outcome = "failure" if event_state == RunState.FAIL else "abort"
            logger.warning(
                "_ProbeOpenLineageExtender observed %s %s: %s: %s", job.name, outcome, type(exc).__name__, exc
            )
            raise
        finally:
            self._open_inputs = previous_inputs

        try:
            outputs = [OutputDataset(namespace="probe", name=name) for name in context.feature_names]
            self._get_client().emit(
                RunEvent(
                    eventType=RunState.COMPLETE,
                    eventTime=_now_iso(),
                    run=run,
                    job=job,
                    producer=_PRODUCER,
                    inputs=current_inputs,
                    outputs=outputs,
                )
            )
        except Exception as exc:
            logger.warning(
                "_ProbeOpenLineageExtender post-call instrumentation failed: %s: %s", type(exc).__name__, exc
            )

        return result


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


class TestProbeOpenLineageExtenderContract(OpenLineageExtenderTestMixin):
    """Self-test: _ProbeOpenLineageExtender must satisfy every OpenLineage contract test the mixin defines."""

    @classmethod
    def extender_class(cls) -> type[Extender]:
        return _ProbeOpenLineageExtender

    def make_openlineage_extender(self, client: OpenLineageClient, *, raise_on_error: bool | None = None) -> Extender:
        if raise_on_error is None:
            return _ProbeOpenLineageExtender(client=client)
        return _ProbeOpenLineageExtender(client=client, raise_on_error=raise_on_error)

    @classmethod
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD}


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
