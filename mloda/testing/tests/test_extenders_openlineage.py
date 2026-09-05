"""Self-tests for mloda.testing.extenders.openlineage helpers, plus a minimal probe extender that
exercises the full OpenLineageExtenderTestMixin contract independently of the real registry extender."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
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


class _ProbeOpenLineageExtender(Extender):
    """Minimal OpenLineage probe: START/COMPLETE|FAIL per calculate, correlating nested input loads."""

    def __init__(self, client: OpenLineageClient | None = None, raise_on_error: bool = False) -> None:
        self.raise_on_error = raise_on_error
        self._client = client
        self._open_inputs: list[InputDataset] | None = None

    def _get_client(self) -> OpenLineageClient:
        if self._client is None:
            self._client = OpenLineageClient()
        return self._client

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_client"] = None
        state["_open_inputs"] = None
        return state

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is None:
            return func(*args, **kwargs)
        if context.hook == ExtenderHook.INPUT_DATA_LOAD:
            return self._call_input_data_load(context, func, *args, **kwargs)
        return self._call_calculate(context, func, *args, **kwargs)

    def _call_input_data_load(self, context: HookContext, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        if self._open_inputs is not None and context.data_access_identity is not None:
            already_present = any(i.name == context.data_access_identity for i in self._open_inputs)
            if not already_present:
                self._open_inputs.append(InputDataset(namespace="probe", name=context.data_access_identity))
        return result

    def _call_calculate(self, context: HookContext, func: Any, *args: Any, **kwargs: Any) -> Any:
        run_facets: dict[str, Any] = {}
        if context.run_id is not None:
            run_facets["parent"] = parent_run.ParentRunFacet(
                run=parent_run.Run(runId=context.run_id),
                job=parent_run.Job(namespace="probe", name="probe.run_all"),
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
            try:
                self._get_client().emit(
                    RunEvent(
                        eventType=RunState.FAIL,
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
                    "_ProbeOpenLineageExtender failed to emit FAIL event: %s: %s", type(emit_exc).__name__, emit_exc
                )
            logger.warning("_ProbeOpenLineageExtender observed %s failure: %s: %s", job.name, type(exc).__name__, exc)
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
