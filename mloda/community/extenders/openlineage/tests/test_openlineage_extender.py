"""Tests for OpenLineageExtender: contract compliance via OpenLineageExtenderTestMixin, plus
schema-facet, namespace-override, dedupe and per-instance-attribution checks not covered by the
mixin.

Direct __call__ tests below wrap calls in a manually built HookContext.activate() scope, mirroring
core's INPUT_DATA_LOAD nesting inside the enclosing CALCULATE_FEATURE HookContext.
"""

from __future__ import annotations

import atexit
import copy
import gc
import json
import logging
import os
import pickle  # nosec
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import patch

import pyarrow as pa
import pytest
import requests
from mloda.core.abstract_plugins.hook_context import instrument  # no public equivalent yet
from mloda.provider import BaseInputData
from mloda.steward import (
    CompositeExtender,
    Extender,
    ExtenderHook,
    HookContext,
    LifecycleOutcome,
    PlanContext,
    RunContext,
)
from mloda.user import PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.extenders.openlineage import openlineage_extender as openlineage_extender_module
from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender, _open_invocations
from mloda.community.extenders.shared.step_run_id import step_run_id
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.extenders.flush import active_close_context
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.openlineage import (
    OPENLINEAGE_EXTENDER_SEAMS,
    BufferingFileTransport,
    FileTransport,
    LockHoldingTransport,
    OpenLineageExtenderTestMixin,
    RecordingTransport,
    assert_openlineage_extender_seams,
    make_recording_client,
)
from mloda.testing.extenders.runners import MlodaTestingValueIntPlusOne, expected_value_int, run_value_int
from openlineage.client.client import OpenLineageClient
from openlineage.client.event_v2 import InputDataset, RunState
from openlineage.client.facet_v2 import documentation_dataset, nominal_time_run, parent_run, schema_dataset
from openlineage.client.serde import Serde
from openlineage.client.transport.transport import Config, Transport

_DEFAULT_PRODUCER = "https://github.com/mloda-ai/mloda-registry/tree/main/mloda/community/extenders/openlineage"
_CUSTOM_PRODUCER = "https://example.invalid/custom-lineage-producer"


class _IncompleteFlushTransport(Transport):
    """A Transport whose close() reports that not everything was flushed in time."""

    kind = "incomplete-flush"
    config_class = Config

    def __init__(self) -> None:
        self.close_calls = 0

    def emit(self, event: Any) -> None:
        pass

    def close(self, timeout: float = -1.0) -> bool:
        self.close_calls += 1
        return False


class _BlockingFlushTransport(Transport):
    """A Transport whose close() blocks on an Event until released, then reports an incomplete flush."""

    kind = "blocking-flush"
    config_class = Config

    def __init__(self) -> None:
        self.close_calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def emit(self, event: Any) -> None:
        pass

    def close(self, timeout: float = -1.0) -> bool:
        self.close_calls += 1
        self.entered.set()
        self.release.wait(timeout=5)
        return False


class _PicklableFakeClient:
    """Module-level so pickle can find it by qualified name when a copy is unpickled."""

    def close(self, timeout: float = -1.0) -> bool:
        return True


class _EqDefiningClient(OpenLineageClient):
    """Defines __eq__ without __hash__, so Python sets __hash__ = None: unhashable."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _EqDefiningClient)


class _EqualityCollidingClient(OpenLineageClient):
    """Two distinct instances compare and hash equal, unlike real client object identity."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _EqualityCollidingClient)

    def __hash__(self) -> int:
        return 0


class _SlottedDuckTypeClient:
    """A minimal duck-typed client with __slots__ and no __weakref__ slot: it cannot be weak-referenced."""

    __slots__ = ("close_calls",)

    def __init__(self) -> None:
        self.close_calls: list[float] = []

    def close(self, timeout: float = -1.0) -> bool:
        self.close_calls.append(timeout)
        return True


_EMIT_ERROR_MESSAGE = "emit-error-message-must-not-be-logged"
_CLOSE_ERROR_MESSAGE = "close-error-message-must-not-be-logged"
_RUN_A = "00000000-0000-4000-8000-00000000000a"
_RUN_B = "00000000-0000-4000-8000-00000000000b"
_RUN_X = "00000000-0000-4000-8000-0000000000ff"
_RUN_OTHER = "00000000-0000-4000-8000-000000000001"
_OPENLINEAGE_ENV_PREFIXES = ("OPENLINEAGE_", "OPENLINEAGE__")
_PARENT_ID = f"airflow/dag.task/{_RUN_A}"
_ROOT_PARENT_ID = f"airflow/dag/{_RUN_B}"
_OTHER_PARENT_ID = f"other-ns/other.job/{_RUN_OTHER}"


def _parent_facet(event: Any) -> parent_run.ParentRunFacet:
    assert event.run.facets is not None
    parent = event.run.facets.get("parent")
    assert isinstance(parent, parent_run.ParentRunFacet)
    return parent


def _start_root(extender: OpenLineageExtender, transport: RecordingTransport) -> Any:
    """Emit the root START and return it."""
    extender.on_run_start(RunContext(run_id=_RUN_X, plan_id="plan-0001"), _plan(), ())
    return transport.events[0]


class _SinkTransport(Transport):
    """Picklable transport whose events survive in a class-level list, so a pickled copy's emits are observable."""

    kind = "sink"
    config_class = Config
    events: ClassVar[list[Any]] = []

    def emit(self, event: Any) -> None:
        type(self).events.append(event)

    def close(self, timeout: float = -1.0) -> bool:
        return True


class _FailingEmitTransport(Transport):
    """A Transport whose emit() raises error_factory() (default RuntimeError) and counts every attempt.
    The first succeed_first attempts succeed; set failing = False to let later attempts succeed."""

    kind = "failing-emit"
    config_class = Config

    def __init__(self, error_factory: Callable[[], BaseException] | None = None, succeed_first: int = 0) -> None:
        self.emit_attempts = 0
        self.error_factory = error_factory or (lambda: RuntimeError(_EMIT_ERROR_MESSAGE))
        self.succeed_first = succeed_first
        self.failing = True

    def emit(self, event: Any) -> None:
        self.emit_attempts += 1
        if self.emit_attempts <= self.succeed_first or not self.failing:
            return
        raise self.error_factory()

    def close(self, timeout: float = -1.0) -> bool:
        return True


class _CustomProducerExtender(OpenLineageExtender):
    producer = _CUSTOM_PRODUCER


def _connection_error() -> BaseException:
    return ConnectionError(_EMIT_ERROR_MESSAGE)


def _http_error(status_code: int) -> BaseException:
    return requests.HTTPError(_EMIT_ERROR_MESSAGE, response=cast(Any, SimpleNamespace(status_code=status_code)))


def _runtime_error_from_connection_error() -> BaseException:
    try:
        raise ConnectionError(_EMIT_ERROR_MESSAGE)
    except ConnectionError as cause:
        wrapped = RuntimeError(_EMIT_ERROR_MESSAGE)
        wrapped.__cause__ = cause
        return wrapped


def _runtime_error_in_handler_of_connection_error() -> BaseException:
    """A RuntimeError raised while handling a ConnectionError, chained only implicitly (no `from`)."""
    try:
        try:
            raise ConnectionError(_EMIT_ERROR_MESSAGE)
        except ConnectionError:
            raise RuntimeError(_EMIT_ERROR_MESSAGE)
    except RuntimeError as wrapped:
        return wrapped


def _runtime_error_from_none_after_connection_error() -> BaseException:
    """A RuntimeError raised `from None` while handling a ConnectionError (context suppressed)."""
    try:
        try:
            raise ConnectionError(_EMIT_ERROR_MESSAGE)
        except ConnectionError:
            raise RuntimeError(_EMIT_ERROR_MESSAGE) from None
    except RuntimeError as wrapped:
        return wrapped


def _spy_on_finalize(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every OpenLineageExtender that gets a weakref.finalize, delegating to the real one."""
    created: list[Any] = []
    real_finalize = weakref.finalize

    def spy(obj: Any, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        if isinstance(obj, OpenLineageExtender):
            created.append(obj)
        return real_finalize(obj, func, *args, **kwargs)

    monkeypatch.setattr(weakref, "finalize", spy)
    return created


class _RecordingDispatchExtender(OpenLineageExtender):
    """Records every _dispatch call, then routes it through the default dispatch."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dispatched: list[tuple[HookContext, Any, tuple[Any, ...], dict[str, Any]]] = []

    def _dispatch(self, context: HookContext, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self.dispatched.append((context, func, args, kwargs))
        return super()._dispatch(context, func, args, kwargs)


class _RunFacetExtender(OpenLineageExtender):
    """Adds one run facet next to the default parent facet and records what the seam received."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_facet_calls: list[tuple[HookContext, Any, tuple[Any, ...]]] = []

    def _calculate_run_facets(self, context: HookContext, func: Any, args: tuple[Any, ...]) -> dict[str, Any]:
        self.run_facet_calls.append((context, func, args))
        facets = super()._calculate_run_facets(context, func, args)
        facets["probe"] = nominal_time_run.NominalTimeRunFacet(
            nominalStartTime="2026-01-01T00:00:00+00:00", producer=self.producer
        )
        return facets


class _OutputFacetExtender(OpenLineageExtender):
    """Adds one dataset facet to every output and records what the seam received."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.output_facet_calls: list[tuple[HookContext, Any, tuple[Any, ...], str, list[InputDataset]]] = []

    def _calculate_output_facets(
        self, context: HookContext, func: Any, args: tuple[Any, ...], name: str, inputs: list[InputDataset]
    ) -> dict[str, Any]:
        self.output_facet_calls.append((context, func, args, name, inputs))
        return {
            "probe": documentation_dataset.DocumentationDatasetFacet(
                description=f"probe:{name}", producer=self.producer
            )
        }


class _FailingOutputFacetExtender(OpenLineageExtender):
    def _calculate_output_facets(
        self, context: HookContext, func: Any, args: tuple[Any, ...], name: str, inputs: list[InputDataset]
    ) -> dict[str, Any]:
        raise RuntimeError("output facet boom")


class _OverridesNonSeamExtender(OpenLineageExtender):
    def _get_client(self) -> Any:
        return super()._get_client()


class _ReshapedSeamExtender(OpenLineageExtender):
    def _calculate_run_facets(self, ctx: HookContext, func: Any, args: tuple[Any, ...]) -> dict[str, Any]:
        return super()._calculate_run_facets(ctx, func, args)


class _ExtraSeamParameterExtender(OpenLineageExtender):
    def _calculate_run_facets(  # type: ignore[override]
        self, context: HookContext, func: Any, args: tuple[Any, ...], extra: int
    ) -> dict[str, Any]:
        return super()._calculate_run_facets(context, func, args)


class _DropsDatasetNamespaceExtender(OpenLineageExtender):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        del self.dataset_namespace


class _AddsPrivateMethodExtender(OpenLineageExtender):
    def _call_validator(self) -> None:
        return None


class _OverridesGetstateExtender(OpenLineageExtender):
    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__()


class _CallsPrivateEmitExtender(OpenLineageExtender):
    def emit_start(self) -> Any:
        return self._emit_event(RunState.START)  # type: ignore[call-arg]


class _ReadsPrivateClientExtender(OpenLineageExtender):
    def peek_client(self) -> Any:
        return self._client


class _UsesPrivateModuleGlobalExtender(OpenLineageExtender):
    def peek_stack(self) -> Any:
        return openlineage_extender_module._open_invocations


class _UsesPrivateModuleNameExtender(OpenLineageExtender):
    def peek_stack(self) -> Any:
        return _open_invocations


class _DefinesDeepcopyExtender(OpenLineageExtender):
    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        return self


class _OverridesSetattrExtender(OpenLineageExtender):
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)


class _CallsPrivateViaBaseClassExtender(OpenLineageExtender):
    def peek_client(self) -> Any:
        return OpenLineageExtender._get_client(self)


class _ReadsPrivateClassAttributeExtender(OpenLineageExtender):
    def peek_timeout(self) -> Any:
        return type(self)._ATEXIT_CLOSE_TIMEOUT


class _OverridesPrivateConstantExtender(OpenLineageExtender):
    _BREAKER_RETRY_AFTER = 0.0


class _StaticmethodOverrideExtender(OpenLineageExtender):
    @staticmethod
    def _log_inert_once() -> None:
        return None


class _PropertyOverrideExtender(OpenLineageExtender):
    @property
    def _log_inert_once(self) -> Any:
        return None


class _DropsSeamDefaultExtender(OpenLineageExtender):
    def _run_with_events(  # type: ignore[override]
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        job: Any,
        run_facets: dict[str, Any],
        declared_inputs: list[Any],
        build_inputs: Any,
        build_outputs: Any,
    ) -> Any:
        return None


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

    @classmethod
    def supports_real_worker_sink(cls) -> bool:
        return True

    def make_real_worker_extender_and_marker(self, tmp_path: Path) -> tuple[Extender, Path]:
        marker_path = tmp_path / "openlineage_real_worker_events.txt"
        client = OpenLineageClient(transport=FileTransport(marker_path))
        extender = self.make_openlineage_extender(client)
        return extender, marker_path

    @classmethod
    def supports_real_worker_buffered_sink(cls) -> bool:
        return True

    def make_real_worker_buffered_extender_and_marker(self, tmp_path: Path) -> tuple[Extender, Path]:
        marker_path = tmp_path / "openlineage_real_worker_buffered_events.txt"
        client = OpenLineageClient(transport=BufferingFileTransport(marker_path))
        extender = self.make_openlineage_extender(client)
        return extender, marker_path


class TestOpenLineageExtenderPriority:
    """Default priority 110 (outside-in after OtelExtender at 100) without overriding a subclass's own priority."""

    def test_a_plain_instance_reports_110(self) -> None:
        assert OpenLineageExtender().priority == 110

    def test_a_subclass_class_attribute_priority_wins(self) -> None:
        class _Fifty(OpenLineageExtender):
            priority = 50

        assert _Fifty().priority == 50

    def test_a_subclass_property_priority_wins(self) -> None:
        class _Sixty(OpenLineageExtender):
            @property  # type: ignore[misc]  # read-only override is the regression under test
            def priority(self) -> int:
                return 60

        assert _Sixty().priority == 60

    def test_priority_can_be_assigned(self) -> None:
        extender = OpenLineageExtender()
        extender.priority = 7

        assert extender.priority == 7


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
    """An injected client is pickled as-is if it can be, else trial-pickle drops it with a warning;
    a self-built one is rebuilt by the copy."""

    def test_pickle_round_trip_keeps_config(self) -> None:
        _SinkTransport.events.clear()
        extender = OpenLineageExtender(
            client=OpenLineageClient(transport=_SinkTransport()),
            raise_on_error=True,
            job_namespace="custom-ns",
            dataset_namespace="custom-ds",
            root_job_name="custom.root",
            parent_id=_PARENT_ID,
            root_parent_id=_ROOT_PARENT_ID,
        )

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        assert copy.raise_on_error is True
        assert copy.job_namespace == "custom-ns"
        assert copy.dataset_namespace == "custom-ds"
        assert copy.root_job_name == "custom.root"
        copy.on_run_start(RunContext(run_id=_RUN_X, plan_id="plan-0001"), _plan(), ())
        parent = _parent_facet(_SinkTransport.events[0])
        assert parent.run.runId == _RUN_A
        assert (parent.job.namespace, parent.job.name) == ("airflow", "dag.task")
        assert parent.root is not None
        assert parent.root.run.runId == _RUN_B
        assert (parent.root.job.namespace, parent.root.job.name) == ("airflow", "dag")

    def test_pickled_copy_can_still_build_a_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        extender = OpenLineageExtender(use_sdk_defaults=True)

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        monkeypatch.setenv("OPENLINEAGE_DISABLED", "true")
        first = copy._get_client()
        second = copy._get_client()

        assert isinstance(first, OpenLineageClient)
        assert first is second

    def test_dropped_injected_client_copy_believes_it_owns_its_client(self) -> None:
        """After a drop, the copy must recognize it now owns/self-builds its client, not still think
        a client was injected - else a second pickle of the copy would wrongly treat its self-built
        client as "injected" and try (and fail) to pickle it as-is again."""
        extender = OpenLineageExtender(
            client=OpenLineageClient(transport=LockHoldingTransport()), use_sdk_defaults=True
        )

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        assert copy._owns_client is True

    def test_self_built_client_is_dropped_on_pickle_and_rebuilt_by_copy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _PicklableFakeClient)
        created = _spy_on_finalize(monkeypatch)
        extender = OpenLineageExtender(use_sdk_defaults=True)
        extender._get_client()

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        assert copy._client is None
        assert copy._finalizer is None
        assert isinstance(copy._get_client(), _PicklableFakeClient)
        assert len(created) == 2
        assert created[1] is copy
        assert copy._finalizer is not None
        assert copy._finalizer is not extender._finalizer
        assert copy._finalizer.alive
        assert copy._finalizer.peek()[0] is copy
        assert extender._finalizer is not None
        assert extender._finalizer.peek()[0] is extender
        assert extender._finalizer.alive

    def test_client_published_before_ownership_flag_is_still_dropped_on_pickle(self) -> None:
        """Ownership must follow from use_sdk_defaults with no injected client, not from a flag that
        could lag a lazy build publishing _client."""
        extender = OpenLineageExtender(use_sdk_defaults=True)
        extender._client = cast(OpenLineageClient, _PicklableFakeClient())

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        assert copy._client is None


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

            def close(self, timeout: float = -1.0) -> bool:
                return True

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FakeOpenLineageClient)
        created = _spy_on_finalize(monkeypatch)

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
        assert created == [extender]
        assert extender._finalizer is not None
        assert extender._finalizer.alive


class TestOpenLineageExtenderClose:
    """close() flushes the underlying OpenLineageClient/transport and ties a weakref.finalize
    only to a client this extender built itself; a caller-injected client is never touched by
    a finalizer, and closing before any client exists must not build one."""

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
        created = _spy_on_finalize(monkeypatch)
        extender = OpenLineageExtender()

        result = extender.close()

        assert result is True
        assert extender._client is None
        assert created == []
        assert extender._finalizer is None

    def test_finalizer_created_exactly_once_when_client_lazily_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeClient:
            def close(self, timeout: float = -1.0) -> bool:
                return True

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FakeClient)
        created = _spy_on_finalize(monkeypatch)
        extender = OpenLineageExtender(use_sdk_defaults=True)

        extender._get_client()
        extender._get_client()

        assert created == [extender]
        assert extender._finalizer is not None
        assert extender._finalizer.alive
        assert not extender._finalizer.atexit
        peeked = extender._finalizer.peek()
        assert peeked is not None
        assert peeked[0] is extender

    def test_finalizer_not_created_when_client_injected_via_constructor(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = ol_capture
        created = _spy_on_finalize(monkeypatch)
        extender = OpenLineageExtender(client=client)

        assert extender._get_client() is client
        extender.close()

        assert created == []
        assert extender._finalizer is None

    def test_building_many_self_built_extenders_does_not_grow_atexit_and_collection_closes_each_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closed: list[tuple[int, float]] = []
        refs: list[weakref.ref[Any]] = []

        class _ProbeClient:
            def __init__(self) -> None:
                self.index = len(refs)
                refs.append(weakref.ref(self))

            def emit(self, event: Any) -> None:
                pass

            def close(self, timeout: float = -1.0) -> bool:
                closed.append((self.index, timeout))
                return True

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _ProbeClient)

        class _Anchor:
            pass

        warm_up = _Anchor()
        weakref.finalize(warm_up, lambda: None)
        del warm_up
        gc.collect()
        callbacks_before = atexit._ncallbacks()
        count = 20

        for _ in range(count):
            extender = OpenLineageExtender(use_sdk_defaults=True)
            with make_hook_context().activate():
                extender(lambda: None)
        del extender
        gc.collect()

        assert atexit._ncallbacks() == callbacks_before
        assert sorted(closed) == [(index, OpenLineageExtender.close_timeout) for index in range(count)]
        assert len(refs) == count
        assert all(ref() is None for ref in refs)

    def test_finalizer_callback_never_raises_when_client_close_raises_and_logs_only_the_type(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _RaisingCloseClient:
            def close(self, timeout: float = -1.0) -> bool:
                raise RuntimeError(_CLOSE_ERROR_MESSAGE)

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _RaisingCloseClient)
        extender = OpenLineageExtender(use_sdk_defaults=True)
        extender._get_client()
        finalizer = extender._finalizer
        assert finalizer is not None

        with caplog.at_level(logging.WARNING, logger=openlineage_extender_module.__name__):
            finalizer()

        assert not finalizer.alive
        records = [
            r for r in caplog.records if r.name == openlineage_extender_module.__name__ and r.levelno == logging.WARNING
        ]
        assert any("OpenLineageExtender" in r.getMessage() and "RuntimeError" in r.getMessage() for r in records)
        assert all(_CLOSE_ERROR_MESSAGE not in r.getMessage() for r in caplog.records)


class TestOpenLineageExtenderCloseIdempotencyAndReuse:
    """close() must be idempotent, detach its own finalizer, use a bounded finalizer timeout,
    warn on incomplete flush, and reject reuse of a closed extender with a RuntimeError that the
    standard warning-only CompositeExtender fallback degrades gracefully instead of silently
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

    def test_close_detaches_finalizer_when_client_lazily_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeClient:
            def __init__(self) -> None:
                self.close_calls: list[float] = []

            def close(self, timeout: float = -1.0) -> bool:
                self.close_calls.append(timeout)
                return True

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FakeClient)
        extender = OpenLineageExtender(use_sdk_defaults=True)
        built: Any = extender._get_client()
        finalizer = extender._finalizer
        assert finalizer is not None
        assert finalizer.alive

        extender.close()

        assert not finalizer.alive
        del extender
        gc.collect()
        assert len(built.close_calls) == 1

    def test_finalizer_closes_client_with_bounded_timeout_not_blocking_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeClient:
            def __init__(self) -> None:
                self.close_calls: list[float] = []

            def close(self, timeout: float = -1.0) -> bool:
                self.close_calls.append(timeout)
                return True

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FakeClient)
        extender = OpenLineageExtender(use_sdk_defaults=True)
        built: Any = extender._get_client()
        assert extender._finalizer is not None
        assert not extender._finalizer.atexit

        extender._finalizer()

        assert built.close_calls == [OpenLineageExtender.close_timeout]
        timeout = built.close_calls[0]
        assert isinstance(timeout, float)
        assert timeout != -1.0
        assert 0 < timeout < float("inf")

    def test_finalizer_closes_with_the_instance_close_timeout_read_when_the_client_is_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeClient:
            def __init__(self) -> None:
                self.close_calls: list[float] = []

            def close(self, timeout: float = -1.0) -> bool:
                self.close_calls.append(timeout)
                return True

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FakeClient)
        extender = OpenLineageExtender(use_sdk_defaults=True)
        extender.close_timeout = 3.0
        built: Any = extender._get_client()
        extender.close_timeout = 99.0
        assert extender._finalizer is not None
        assert not extender._finalizer.atexit

        extender._finalizer()

        assert built.close_calls == [3.0]

    def test_exit_hook_closes_live_self_built_extenders_with_the_atexit_close_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeClient:
            def __init__(self) -> None:
                self.close_calls: list[float] = []

            def close(self, timeout: float = -1.0) -> bool:
                self.close_calls.append(timeout)
                return True

        class _ShortTimeoutExtender(OpenLineageExtender):
            _ATEXIT_CLOSE_TIMEOUT = 3.0

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FakeClient)
        default = OpenLineageExtender(use_sdk_defaults=True)
        short = _ShortTimeoutExtender(use_sdk_defaults=True)
        default_client: Any = default._get_client()
        short_client: Any = short._get_client()

        openlineage_extender_module._close_live_extenders_at_exit()

        assert default_client.close_calls == [OpenLineageExtender._ATEXIT_CLOSE_TIMEOUT]
        assert short_client.close_calls == [3.0]

    def test_exit_hook_skips_collected_and_closed_extenders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeClient:
            def __init__(self) -> None:
                self.close_calls: list[float] = []

            def close(self, timeout: float = -1.0) -> bool:
                self.close_calls.append(timeout)
                return True

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FakeClient)
        collected = OpenLineageExtender(use_sdk_defaults=True)
        collected_client: Any = collected._get_client()
        closed = OpenLineageExtender(use_sdk_defaults=True)
        closed_client: Any = closed._get_client()
        closed.close(timeout=1.0)
        del collected
        gc.collect()
        assert collected_client.close_calls == [OpenLineageExtender.close_timeout]

        openlineage_extender_module._close_live_extenders_at_exit()

        assert collected_client.close_calls == [OpenLineageExtender.close_timeout]
        assert closed_client.close_calls == [1.0]

    def test_exit_hook_never_raises_and_logs_only_the_type(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _RaisingCloseClient:
            def close(self, timeout: float = -1.0) -> bool:
                raise RuntimeError(_CLOSE_ERROR_MESSAGE)

        class _FakeClient:
            def __init__(self) -> None:
                self.close_calls: list[float] = []

            def close(self, timeout: float = -1.0) -> bool:
                self.close_calls.append(timeout)
                return True

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _RaisingCloseClient)
        raising = OpenLineageExtender(use_sdk_defaults=True)
        raising._get_client()
        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FakeClient)
        healthy = OpenLineageExtender(use_sdk_defaults=True)
        healthy_client: Any = healthy._get_client()

        with caplog.at_level(logging.WARNING, logger=openlineage_extender_module.__name__):
            openlineage_extender_module._close_live_extenders_at_exit()

        assert healthy_client.close_calls == [OpenLineageExtender._ATEXIT_CLOSE_TIMEOUT]
        assert all(_CLOSE_ERROR_MESSAGE not in r.getMessage() for r in caplog.records)
        assert any("RuntimeError" in r.getMessage() for r in caplog.records)

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
        composite = CompositeExtender([extender])
        sentinel = object()

        def func() -> object:
            return sentinel

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            with caplog.at_level(logging.WARNING):
                result = composite(func)

        assert result is sentinel
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OpenLineageExtender" in message for message in warnings)

    def test_repeat_close_of_injected_client_returns_the_recorded_false_result(self) -> None:
        """A repeat close() must return the real recorded flush result, not a blanket True."""
        transport = _IncompleteFlushTransport()
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))

        first = extender.close()
        second = extender.close()

        assert first is False
        assert second is False
        assert transport.close_calls == 1

    def test_repeat_close_of_self_built_client_returns_the_recorded_false_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A self-built client's repeat close() must return the recorded flush result, not a blanket True."""

        class _FailingFlushClient:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self, timeout: float = -1.0) -> bool:
                self.close_calls += 1
                return False

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FailingFlushClient)
        extender = OpenLineageExtender(use_sdk_defaults=True)
        built: Any = extender._get_client()

        first = extender.close()
        second = extender.close()

        assert first is False
        assert second is False
        assert built.close_calls == 1

    def test_concurrent_second_close_on_the_same_instance_waits_for_the_in_flight_flush(self) -> None:
        """A second close() on the same instance must wait for the first's in-flight flush, not
        short-circuit to True before the flush has even finished."""
        transport = _BlockingFlushTransport()
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        results: dict[str, bool] = {}

        def close_first() -> None:
            results["first"] = extender.close(timeout=-1)

        def close_second() -> None:
            results["second"] = extender.close(timeout=-1)

        thread_first = threading.Thread(target=close_first)
        thread_second = threading.Thread(target=close_second)
        try:
            thread_first.start()
            assert transport.entered.wait(timeout=5), "transport.close() was never entered"
            thread_second.start()

            thread_second.join(timeout=0.2)
            assert thread_second.is_alive(), (
                "the second close() must block on the in-flight flush, not return immediately"
            )
        finally:
            transport.release.set()
            thread_first.join(timeout=5)
            thread_second.join(timeout=5)

        assert not thread_first.is_alive()
        assert not thread_second.is_alive()
        assert results["first"] is False
        assert results["second"] is False
        assert transport.close_calls == 1

    def test_close_racing_an_in_flight_lazy_build_waits_for_it_and_closes_the_built_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """close() started while a lazy build is still in flight must wait for that build, not
        return True for "no client yet", then flush the client the build produced."""
        building = threading.Event()
        release = threading.Event()

        class _BuildRacingFakeClient:
            def __init__(self) -> None:
                self.close_calls = 0
                building.set()
                release.wait(timeout=5)

            def close(self, timeout: float = -1.0) -> bool:
                self.close_calls += 1
                return False

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _BuildRacingFakeClient)
        extender = OpenLineageExtender(use_sdk_defaults=True)
        results: dict[str, bool] = {}

        def build() -> None:
            extender._get_client()

        def closer() -> None:
            results["close"] = extender.close(timeout=-1)

        thread_build = threading.Thread(target=build)
        thread_close = threading.Thread(target=closer)
        try:
            thread_build.start()
            assert building.wait(timeout=5), "the fake client's __init__ was never entered"
            thread_close.start()

            thread_close.join(timeout=0.2)
            assert thread_close.is_alive(), (
                "close() must wait for the in-flight build, not return True for no client yet"
            )
        finally:
            release.set()
            thread_build.join(timeout=5)
            thread_close.join(timeout=5)

        assert not thread_build.is_alive()
        assert not thread_close.is_alive()
        assert results["close"] is False
        built: Any = extender._client
        assert built is not None
        assert built.close_calls == 1
        assert extender._finalizer is not None
        assert not extender._finalizer.alive

    def test_close_retries_after_a_flush_that_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A close() that raises must propagate the error and leave the extender retryable, not
        permanently closed as if the flush had recorded a result."""

        class _FlakyThenOkClient:
            def __init__(self) -> None:
                self.close_calls = 0

            def close(self, timeout: float = -1.0) -> bool:
                self.close_calls += 1
                if self.close_calls == 1:
                    raise RuntimeError("flush boom")
                return True

        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _FlakyThenOkClient)
        extender = OpenLineageExtender(use_sdk_defaults=True)
        built: Any = extender._get_client()

        with pytest.raises(RuntimeError, match="flush boom"):
            extender.close()

        second = extender.close()

        assert second is True
        assert built.close_calls == 2


class TestOpenLineageExtenderCloseTimeoutDefault:
    """close() with no explicit timeout uses the class-level close_timeout (defaulting to the shared
    CLOSE_TIMEOUT), instead of blocking forever."""

    def test_no_arg_close_passes_close_timeout_to_client_close(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mloda.community.extenders.shared.teardown import CLOSE_TIMEOUT

        client, _ = ol_capture
        extender = OpenLineageExtender(client=client)
        received: list[float] = []

        def fake_close(timeout: float = -1.0) -> bool:
            received.append(timeout)
            return True

        monkeypatch.setattr(client, "close", fake_close)

        extender.close()

        assert received == [CLOSE_TIMEOUT]

    def test_instance_close_timeout_override_is_honored(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = ol_capture
        extender = OpenLineageExtender(client=client)
        extender.close_timeout = 42.0
        received: list[float] = []

        def fake_close(timeout: float = -1.0) -> bool:
            received.append(timeout)
            return True

        monkeypatch.setattr(client, "close", fake_close)

        extender.close()

        assert received == [42.0]

    def test_instance_close_timeout_override_survives_pickling(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = ol_capture
        extender = OpenLineageExtender(client=client)
        extender.close_timeout = 42.0

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        received: list[float] = []

        def fake_close(timeout: float = -1.0) -> bool:
            received.append(timeout)
            return True

        monkeypatch.setattr(copy._client, "close", fake_close)

        copy.close()

        assert received == [42.0]

    def test_explicit_timeout_still_passes_through_unchanged(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = ol_capture
        extender = OpenLineageExtender(client=client)
        received: list[float] = []

        def fake_close(timeout: float = -1.0) -> bool:
            received.append(timeout)
            return True

        monkeypatch.setattr(client, "close", fake_close)

        extender.close(timeout=3.5)

        assert received == [3.5]

    def test_explicit_no_cap_close_in_active_close_context_becomes_remaining_budget(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = ol_capture
        extender = OpenLineageExtender(client=client)
        received: list[float] = []

        def fake_close(timeout: float = -1.0) -> bool:
            received.append(timeout)
            return True

        monkeypatch.setattr(client, "close", fake_close)

        with active_close_context(3.0):
            extender.close(-1)

        assert len(received) == 1
        assert 0 < received[0] <= 3.0


class TestOpenLineageExtenderSharedInjectedClientCloseState:
    """Two extenders built with the same injected client object share its close lifecycle in both directions."""

    def test_closing_a_makes_bs_get_client_raise_too(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        shared, _ = ol_capture
        extender_a = OpenLineageExtender(client=shared)
        extender_b = OpenLineageExtender(client=shared)

        extender_a.close()

        with pytest.raises(RuntimeError):
            extender_b._get_client()

    def test_closing_b_makes_as_get_client_raise_too(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        shared, _ = ol_capture
        extender_a = OpenLineageExtender(client=shared)
        extender_b = OpenLineageExtender(client=shared)

        extender_b.close()

        with pytest.raises(RuntimeError):
            extender_a._get_client()

    def test_close_state_tracked_by_identity_eq_without_hash(self) -> None:
        """__eq__ without __hash__ makes the client unhashable; a WeakSet cannot even check membership."""
        transport = RecordingTransport()
        client = _EqDefiningClient(transport=transport)
        extender = OpenLineageExtender(client=client)

        assert extender._get_client() is client
        assert extender.close() is True

    def test_close_state_tracked_by_identity_equal_but_distinct_instances(self) -> None:
        """Two different client objects that compare equal must not share close state."""
        transport_a = RecordingTransport()
        transport_b = RecordingTransport()
        client_a = _EqualityCollidingClient(transport=transport_a)
        client_b = _EqualityCollidingClient(transport=transport_b)
        extender_a = OpenLineageExtender(client=client_a)
        extender_b = OpenLineageExtender(client=client_b)

        extender_a.close()

        assert extender_b._get_client() is client_b

    def test_close_state_tracked_by_identity_slotted_duck_type_no_weakref(self) -> None:
        """A client with no __weakref__ slot cannot be added to a WeakSet at all."""
        client = _SlottedDuckTypeClient()
        extender = OpenLineageExtender(client=cast(OpenLineageClient, client))

        from mloda.community.extenders.shared.teardown import CLOSE_TIMEOUT

        result = extender.close()

        assert result is True
        assert client.close_calls == [CLOSE_TIMEOUT]

    def test_zero_budget_closer_leaves_the_shared_flush_to_a_later_sibling(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _SlottedDuckTypeClient()
        shared = cast(OpenLineageClient, client)
        extender_a = OpenLineageExtender(client=shared)
        extender_b = OpenLineageExtender(client=shared)

        with caplog.at_level(logging.WARNING):
            assert extender_a.close(0.0) is False
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "close skipped" in warnings[0]
        assert "OpenLineageExtender" in warnings[0]
        assert client.close_calls == []
        assert extender_b.close(-1) is True
        assert client.close_calls == [-1]

    def test_zero_budget_close_then_negative_budget_close_on_same_instance_flushes(self) -> None:
        client = _SlottedDuckTypeClient()
        extender = OpenLineageExtender(client=cast(OpenLineageClient, client))

        assert extender.close(0.0) is False
        assert client.close_calls == []
        assert extender.close(-1) is True
        assert client.close_calls == [-1]

    def test_zero_budget_closer_after_real_flush_gets_cached_result(self) -> None:
        client = _SlottedDuckTypeClient()
        shared = cast(OpenLineageClient, client)
        extender_a = OpenLineageExtender(client=shared)
        extender_b = OpenLineageExtender(client=shared)

        assert extender_b.close(5.0) is True
        assert extender_a.close(0.0) is True
        assert client.close_calls == [5.0]

    def test_shared_incomplete_flush_result_is_returned_to_every_closer(self) -> None:
        transport = _IncompleteFlushTransport()
        shared = OpenLineageClient(transport=transport)
        extender_a = OpenLineageExtender(client=shared)
        extender_b = OpenLineageExtender(client=shared)

        result_a = extender_a.close()
        result_b = extender_b.close()

        assert result_a is False
        assert result_b is False, "B must see the real recorded flush result, not a blanket True"
        assert transport.close_calls == 1

    def test_concurrent_close_of_shared_client_serializes_and_returns_the_same_result(self) -> None:
        transport = _BlockingFlushTransport()
        shared = OpenLineageClient(transport=transport)
        extender_a = OpenLineageExtender(client=shared)
        extender_b = OpenLineageExtender(client=shared)
        results: dict[str, bool] = {}

        def close_a() -> None:
            results["a"] = extender_a.close(timeout=-1)

        def close_b() -> None:
            results["b"] = extender_b.close(timeout=-1)

        thread_a = threading.Thread(target=close_a)
        thread_b = threading.Thread(target=close_b)
        try:
            thread_a.start()
            assert transport.entered.wait(timeout=5), "transport.close() was never entered"
            thread_b.start()

            thread_b.join(timeout=0.2)
            assert thread_b.is_alive(), "B.close() must block on A's in-flight flush, not return immediately"
        finally:
            transport.release.set()
            thread_a.join(timeout=5)
            thread_b.join(timeout=5)

        assert not thread_a.is_alive()
        assert not thread_b.is_alive()
        assert results["a"] == results["b"]
        assert results["a"] is False, "must be the transport's real recorded result, not a default True"
        assert results["b"] is False, "must be the transport's real recorded result, not a default True"
        assert transport.close_calls == 1

    def test_close_with_timeout_gives_up_on_an_in_flight_shared_flush(self) -> None:
        transport = _BlockingFlushTransport()
        shared = OpenLineageClient(transport=transport)
        extender_a = OpenLineageExtender(client=shared)
        extender_b = OpenLineageExtender(client=shared)
        result_b: list[bool] = []

        def close_a() -> None:
            extender_a.close(timeout=-1)

        thread_a = threading.Thread(target=close_a)
        try:
            thread_a.start()
            assert transport.entered.wait(timeout=5), "transport.close() was never entered"

            started = time.monotonic()
            result_b.append(extender_b.close(timeout=0.05))
            elapsed = time.monotonic() - started

            assert result_b == [False], "B must not fall back to a blanket True while A's flush is in flight"
            assert elapsed < 1.0, "B.close() must not block waiting for A's in-flight flush"
        finally:
            transport.release.set()
            thread_a.join(timeout=5)

    def test_closed_injected_client_is_evicted_from_the_shared_close_registry(self) -> None:
        """A closed extender must not keep its injected client alive forever in the shared-close registry."""
        client = OpenLineageClient(transport=RecordingTransport())
        extender = OpenLineageExtender(client=client)
        key = id(client)
        client_ref = weakref.ref(client)

        extender.close()
        del extender
        del client
        gc.collect()

        assert key not in openlineage_extender_module._shared_close_registry
        assert client_ref() is None

    def test_closed_slotted_duck_type_client_is_evicted_from_the_shared_close_registry(self) -> None:
        """A non-weakrefable duck-typed client must also not be retained forever by the shared-close registry."""
        client = _SlottedDuckTypeClient()
        extender = OpenLineageExtender(client=cast(OpenLineageClient, client))
        key = id(client)

        extender.close()
        del extender
        del client
        gc.collect()

        assert key not in openlineage_extender_module._shared_close_registry


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
    def test_pickled_copy_logs_its_own_inert_state(self, caplog: pytest.LogCaptureFixture) -> None:
        """The copy logs its own inert state instead of inheriting the original's fired guard."""
        extender = OpenLineageExtender()
        extender._inert_warning.warn_once(lambda: None)

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
        original_warning = openlineage_extender_module.logger.warning
        warning_calls: list[str] = []

        # Slow the one-shot warning to widen the check-then-log race window.
        def slow_warning(msg: str, *args: Any, **kwargs: Any) -> None:
            warning_calls.append(msg)
            time.sleep(0.05)
            original_warning(msg, *args, **kwargs)

        monkeypatch.setattr(openlineage_extender_module.logger, "warning", slow_warning)

        thread_count = 32
        barrier = threading.Barrier(thread_count)

        def worker() -> None:
            barrier.wait(timeout=5)
            with make_hook_context().activate():
                extender(lambda: None)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(thread_count)]
        with caplog.at_level(logging.WARNING):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        # Self-check: the slow_warning patch was actually hit.
        assert len(warning_calls) == 1, warning_calls

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
    """After a successful func: outputs (one per feature name), then RunEvent(COMPLETE).

    Schema facets are driven entirely by context.output_schema (mloda core's real seam:
    each compute framework's _extract_column_names/_extract_column_dtype, wired through
    instrument() in core's compute_framework.py); the extender no longer introspects the
    raw calculate-feature result at all. Duck-typing coverage for most pandas/spark/polars
    shapes moved to mloda core's own per-compute-framework test suite, not here.

    Note: test_schema_facet_type_not_garbage_for_duplicate_pandas_column_names was deleted along
    with the other duck-typing tests rather than transferred, because core's behavior for that
    exact scenario is NOT identical: a duplicate pandas column name now yields type=None instead
    of the old garbage-avoided "int64". That one case is the exception to the paragraph above: it
    was not picked up by core's own test suite either, so it is currently untested anywhere, not
    just moved out of this file. This is a real, minor fidelity change, not a pure transfer.
    """

    def test_schema_facet_present_from_context_output_schema(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = make_hook_context(
            feature_names=("value_int", "value_str"),
            output_schema=(("value_int", "int64"), ("value_str", "string")),
        )
        extender = OpenLineageExtender(client=client, dataset_namespace="custom-ds")

        with context.activate():
            extender(lambda: None)

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
        context = make_hook_context(
            feature_names=("value_int", "value_str"),
            output_schema=(("value_int", "int64"), ("value_str", "string"), ("internal_secret_col", "int64")),
        )
        extender = OpenLineageExtender(client=client)

        with context.activate():
            extender(lambda: None)

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

    def test_schema_facet_absent_when_output_schema_none_even_with_pyarrow_result(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        """A real pyarrow Table result must never be duck-typed for a schema facet; only
        context.output_schema may drive facet presence, and it defaults to None."""
        client, transport = ol_capture
        context = make_hook_context()
        extender = OpenLineageExtender(client=client)

        with context.activate():
            extender(lambda: pa.table({"value_int": [1, 2, 3]}))

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        output = complete_event.outputs[0]
        assert output.facets is not None
        assert "schema" not in output.facets

    def test_schema_facet_dtype_string_shape_unit_pin(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        """Fast, isolated unit pin of the dict-interchange dtype-string shape (e.g. "int"/"str", as
        core's `_dict_output_schema` produces), driven purely off a hand-set context.output_schema.
        The extender never reads `func`'s return value for schema purposes (only context.output_schema
        drives facet content), so `func` is a no-op here; distinct from
        test_run_all_complete_event_carries_real_schema_facet, which proves the same shape end-to-end
        through a real mloda.run_all."""
        client, transport = ol_capture
        context = make_hook_context(
            feature_names=("value_int", "value_str"),
            output_schema=(("value_int", "int"), ("value_str", "str")),
        )
        extender = OpenLineageExtender(client=client)

        with context.activate():
            extender(lambda: None)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        datasets_by_name = {ds.name: ds for ds in complete_event.outputs}

        int_dataset = datasets_by_name["value_int"]
        assert int_dataset.facets is not None
        int_schema = int_dataset.facets.get("schema")
        assert isinstance(int_schema, schema_dataset.SchemaDatasetFacet)
        assert int_schema.fields is not None
        assert [f.name for f in int_schema.fields] == ["value_int"]
        assert [f.type for f in int_schema.fields] == ["int"]

        str_dataset = datasets_by_name["value_str"]
        assert str_dataset.facets is not None
        str_schema = str_dataset.facets.get("schema")
        assert isinstance(str_schema, schema_dataset.SchemaDatasetFacet)
        assert str_schema.fields is not None
        assert [f.name for f in str_schema.fields] == ["value_str"]
        assert [f.type for f in str_schema.fields] == ["str"]

    def test_schema_facet_type_none_passthrough(self, ol_capture: tuple[OpenLineageClient, RecordingTransport]) -> None:
        """SchemaDatasetFacetFields(type=None) is a meaningful value (what a partial-dtype result,
        e.g. python_dict/duckdb, produces in practice), not a reason to drop the field or
        stringify it to the literal "None"."""
        client, transport = ol_capture
        context = make_hook_context(feature_names=("value_int",), output_schema=(("value_int", None),))
        extender = OpenLineageExtender(client=client)

        with context.activate():
            extender(lambda: None)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        output = complete_event.outputs[0]
        assert output.facets is not None
        schema_facet = output.facets.get("schema")
        assert isinstance(schema_facet, schema_dataset.SchemaDatasetFacet)
        assert schema_facet.fields is not None
        assert [f.name for f in schema_facet.fields] == ["value_int"]
        assert schema_facet.fields[0].type is None

    def test_schema_facet_uses_output_schema_populated_during_call(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        """context.output_schema is set by instrument() only once func returns; the extender must
        read it AFTER calling func, never a stale/empty value captured before the call completes."""
        client, transport = ol_capture
        context = make_hook_context(feature_names=("value_int",))
        extender = OpenLineageExtender(client=client)

        wrapped = instrument(
            context,
            lambda: pa.table({"value_int": [1, 2]}),
            output_schema=lambda result: (("value_int", "int64"),),
        )

        with context.activate():
            extender(wrapped)

        complete_event = transport.events[1]
        assert complete_event.outputs is not None
        output = complete_event.outputs[0]
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

    @pytest.mark.parametrize(
        ("parent_id", "root_parent_id", "expected_root"),
        [
            pytest.param(_PARENT_ID, _ROOT_PARENT_ID, ("airflow", "dag", _RUN_B), id="orchestrator-root"),
            pytest.param(None, None, None, id="unconfigured"),
        ],
    )
    def test_step_parent_facet_root(
        self,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        parent_id: str | None,
        root_parent_id: str | None,
        expected_root: tuple[str, str, str] | None,
    ) -> None:
        client, transport = ol_capture
        run_id = str(uuid.uuid4())
        extender = OpenLineageExtender(client=client, parent_id=parent_id, root_parent_id=root_parent_id)

        with make_hook_context(run_id=run_id).activate():
            extender(lambda: None)

        parent = _parent_facet(transport.events[0])
        assert parent.run.runId == run_id
        assert (parent.job.namespace, parent.job.name) == ("mloda", "mloda.run_all")
        if expected_root is None:
            assert parent.root is None
        else:
            assert parent.root is not None
            assert (parent.root.job.namespace, parent.root.job.name, parent.root.run.runId) == expected_root


_PLAN_ID = "plan-0001"


def _plan(structure_hash: str | None = None) -> PlanContext:
    return PlanContext(
        plan_id=_PLAN_ID,
        tenant_id=None,
        project_id=None,
        principal=None,
        created_at=datetime.now(timezone.utc),
        structure_hash=structure_hash,
    )


def _plan_facet(event: Any) -> dict[str, Any]:
    facet: dict[str, Any] = json.loads(Serde.to_json(event))["run"]["facets"]["mlodaPlan"]
    return facet


class TestOpenLineageExtenderParentRun:
    """on_run_start emits START for the run itself (job root_job_name, runId = run_id, mlodaPlan facet);
    on_run_complete emits COMPLETE, FAIL or ABORT, only when START was emitted."""

    def test_run_start_emits_a_parent_start_event(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        started_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        extender = OpenLineageExtender(client=client)

        extender.on_run_start(RunContext(run_id=_RUN_A, plan_id=_PLAN_ID, started_at=started_at), _plan(), ())

        assert len(transport.events) == 1
        event = transport.events[0]
        assert event.eventType == RunState.START
        assert event.run.runId == _RUN_A
        assert (event.job.namespace, event.job.name) == ("mloda", "mloda.run_all")
        assert datetime.fromisoformat(event.eventTime) == started_at
        assert _plan_facet(event)["planId"] == _PLAN_ID
        assert _plan_facet(event)["_producer"] == extender.producer
        assert event.producer == extender.producer
        assert "structureHash" not in _plan_facet(event)

    def test_run_start_uses_the_custom_namespace_and_root_job_name_and_now_without_started_at(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client, job_namespace="custom-ns", root_job_name="custom.root")
        before = datetime.now(timezone.utc)

        extender.on_run_start(RunContext(run_id=_RUN_A), _plan(), ())

        event = transport.events[0]
        assert (event.job.namespace, event.job.name) == ("custom-ns", "custom.root")
        assert datetime.fromisoformat(event.eventTime) >= before

    @pytest.mark.parametrize(
        ("status", "state"),
        [
            pytest.param("succeeded", RunState.COMPLETE, id="succeeded"),
            pytest.param("failed", RunState.FAIL, id="failed"),
            pytest.param("cancelled", RunState.ABORT, id="cancelled"),
        ],
    )
    def test_run_complete_emits_the_terminal_event_for_the_outcome(
        self, status: Any, state: RunState, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        run = RunContext(run_id=_RUN_A, plan_id=_PLAN_ID)

        extender.on_run_start(run, _plan(), ())
        extender.on_run_complete(run, LifecycleOutcome(status=status))

        assert [event.eventType for event in transport.events] == [RunState.START, state]
        terminal = transport.events[1]
        assert terminal.run.runId == _RUN_A
        assert terminal.job.name == "mloda.run_all"
        assert _plan_facet(terminal)["planId"] == _PLAN_ID
        assert "structureHash" not in _plan_facet(terminal)

    @pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
    def test_start_and_terminal_facets_carry_the_structure_hash(
        self, status: Any, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        run = RunContext(run_id=_RUN_A, plan_id=_PLAN_ID)

        extender.on_run_start(run, _plan("h" * 64), ())
        extender.on_run_complete(run, LifecycleOutcome(status=status))

        assert len(transport.events) == 2
        assert [_plan_facet(event)["structureHash"] for event in transport.events] == ["h" * 64, "h" * 64]

    def test_run_complete_without_an_emitted_start_emits_nothing(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)

        extender.on_run_complete(RunContext(run_id=_RUN_A), LifecycleOutcome(status="succeeded"))

        assert transport.events == []

    def test_a_terminal_event_is_emitted_once_per_started_run(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        run = RunContext(run_id=_RUN_A)

        extender.on_run_start(run, _plan(), ())
        extender.on_run_complete(run, LifecycleOutcome(status="succeeded"))
        extender.on_run_complete(run, LifecycleOutcome(status="succeeded"))

        assert [event.eventType for event in transport.events] == [RunState.START, RunState.COMPLETE]

    def test_inert_extender_emits_no_parent_events(self) -> None:
        extender = OpenLineageExtender()
        run = RunContext(run_id=_RUN_A)

        extender.on_run_start(run, _plan(), ())
        extender.on_run_complete(run, LifecycleOutcome(status="succeeded"))

    def test_run_without_a_run_id_emits_nothing(self, ol_capture: tuple[OpenLineageClient, RecordingTransport]) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        run = RunContext(run_id=None)

        extender.on_run_start(run, _plan(), ())
        extender.on_run_complete(run, LifecycleOutcome(status="succeeded"))

        assert transport.events == []

    def test_a_failed_parent_start_trips_the_breaker_for_the_run(self, caplog: pytest.LogCaptureFixture) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        composite = CompositeExtender([extender])
        sentinel = object()

        with caplog.at_level(logging.WARNING):
            extender.on_run_start(RunContext(run_id=_RUN_X), _plan(), ())
            assert transport.emit_attempts == 1
            with make_hook_context(run_id=_RUN_X).activate():
                assert composite(lambda: sentinel) is sentinel

        assert transport.emit_attempts == 1
        assert len(_breaker_records(caplog)) == 1

    def test_a_failed_parent_start_emits_no_terminal_event(self) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        run = RunContext(run_id=_RUN_X)

        extender.on_run_start(run, _plan(), ())
        extender.on_run_complete(run, LifecycleOutcome(status="succeeded"))

        assert transport.emit_attempts == 1

    def test_run_complete_clears_the_breaker_and_the_started_state(self) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        run = RunContext(run_id=_RUN_X)

        extender.on_run_start(run, _plan(), ())
        extender.on_run_complete(run, LifecycleOutcome(status="succeeded"))
        assert extender._tripped_runs == {}

        transport.failing = False
        extender.on_run_start(run, _plan(), ())
        extender.on_run_complete(run, LifecycleOutcome(status="succeeded"))
        assert transport.emit_attempts == 3

    def test_raise_on_error_true_propagates_a_failed_parent_start(self) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport), raise_on_error=True)

        with pytest.raises(ConnectionError):
            extender.on_run_start(RunContext(run_id=_RUN_X), _plan(), ())

    def test_raise_on_error_true_refuses_the_run_on_a_failed_parent_start(self) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport), raise_on_error=True)

        with pytest.raises(Exception):
            run_value_int(extender)

    def test_a_failed_parent_start_never_fails_the_run_by_default(self) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))

        assert run_value_int(extender) == expected_value_int()

    def test_step_runs_keep_only_the_parent_facet(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture

        run_value_int(OpenLineageExtender(client=client))

        steps = [event for event in transport.events if event.job.name != "mloda.run_all"]
        assert steps
        for event in steps:
            assert "mlodaPlan" not in (event.run.facets or {})

    def test_step_run_id_is_derived_from_the_run_job_features_and_framework(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        run_id = str(uuid.uuid4())
        step_uuid = uuid.uuid4()
        context = make_hook_context(
            run_id=run_id,
            feature_group_class="pkg.Group",
            feature_names=("b", "a"),
            compute_framework_name="PyArrowTable",
            step_uuid=step_uuid,
        )

        with context.activate():
            extender(lambda: None)

        assert transport.events[0].run.runId == step_run_id(run_id, "pkg.Group", ("a", "b"), "PyArrowTable", step_uuid)

    def test_step_run_id_falls_back_to_a_random_uuid_without_a_derivable_run_id(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)

        for _ in range(2):
            with make_hook_context(run_id=None).activate():
                extender(lambda: None)

        starts = [event.run.runId for event in transport.events if event.eventType == RunState.START]
        assert len(set(starts)) == 2
        for value in starts:
            uuid.UUID(value)

    @pytest.mark.parametrize(
        ("env_set", "parent_from_env"),
        [
            pytest.param(True, False, id="env-set-without-opt-in"),
            pytest.param(False, True, id="env-unset-with-opt-in"),
        ],
    )
    def test_no_parent_facet_on_the_root_run(
        self,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        env_set: bool,
        parent_from_env: bool,
    ) -> None:
        for name, value in (("OPENLINEAGE_PARENT_ID", _PARENT_ID), ("OPENLINEAGE_ROOT_PARENT_ID", _ROOT_PARENT_ID)):
            if env_set:
                monkeypatch.setenv(name, value)
            else:
                monkeypatch.delenv(name, raising=False)
        client, transport = ol_capture

        with caplog.at_level(logging.WARNING):
            extender = OpenLineageExtender(client=client, parent_from_env=parent_from_env)
        event = _start_root(extender, transport)

        assert event.run.facets is not None
        assert "parent" not in event.run.facets
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_start_and_terminal_events_carry_the_orchestrator_parent_and_root(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client, parent_id=_PARENT_ID, root_parent_id=_ROOT_PARENT_ID)
        run = RunContext(run_id=_RUN_X, plan_id=_PLAN_ID)

        extender.on_run_start(run, _plan(), ())
        extender.on_run_complete(run, LifecycleOutcome(status="succeeded"))

        assert [event.eventType for event in transport.events] == [RunState.START, RunState.COMPLETE]
        for event in transport.events:
            parent = _parent_facet(event)
            assert parent.run.runId == _RUN_A
            assert (parent.job.namespace, parent.job.name) == ("airflow", "dag.task")
            assert parent.root is not None
            assert parent.root.run.runId == _RUN_B
            assert (parent.root.job.namespace, parent.root.job.name) == ("airflow", "dag")
            assert parent._producer == extender.producer

    @pytest.mark.parametrize(
        ("parent_id", "expected_namespace", "expected_name", "via_env"),
        [
            pytest.param(_PARENT_ID, "airflow", "dag.task", False, id="missing-root-defaults-to-parent"),
            pytest.param(
                f"kafka://host:9092/my.job/{_RUN_A}", "kafka://host:9092", "my.job", False, id="namespace-with-port"
            ),
            pytest.param(f"airflow/dag.task/{_RUN_A.replace('-', '').upper()}", "airflow", "dag.task", False, id="hex"),
            pytest.param(f"airflow/dag.task/{{{_RUN_A.upper()}}}", "airflow", "dag.task", False, id="braces"),
            pytest.param(f"airflow/dag.task/urn:uuid:{_RUN_A}", "airflow", "dag.task", False, id="urn"),
            pytest.param(f"{_PARENT_ID}\n", "airflow", "dag.task", False, id="trailing-newline"),
            pytest.param(f" {_PARENT_ID}\n", "airflow", "dag.task", True, id="env-surrounding-whitespace"),
        ],
    )
    def test_parent_id_parses_and_the_root_defaults_to_it(
        self,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        monkeypatch: pytest.MonkeyPatch,
        parent_id: str,
        expected_namespace: str,
        expected_name: str,
        via_env: bool,
    ) -> None:
        client, transport = ol_capture
        if via_env:
            monkeypatch.setenv("OPENLINEAGE_PARENT_ID", parent_id)
            monkeypatch.delenv("OPENLINEAGE_ROOT_PARENT_ID", raising=False)
            extender = OpenLineageExtender(client=client, parent_from_env=True)
        else:
            extender = OpenLineageExtender(client=client, parent_id=parent_id)

        parent = _parent_facet(_start_root(extender, transport))

        assert (parent.job.namespace, parent.job.name, parent.run.runId) == (expected_namespace, expected_name, _RUN_A)
        assert parent.root is not None
        assert (parent.root.job.namespace, parent.root.job.name, parent.root.run.runId) == (
            expected_namespace,
            expected_name,
            _RUN_A,
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"parent_id": "no-slashes"}, id="parent-no-slashes"),
            pytest.param({"parent_id": f"ns/{_RUN_A}"}, id="parent-two-parts"),
            pytest.param({"parent_id": "ns/job/not-a-uuid"}, id="parent-bad-uuid"),
            pytest.param({"parent_id": f"/job/{_RUN_A}"}, id="parent-empty-namespace"),
            pytest.param({"parent_id": f"ns//{_RUN_A}"}, id="parent-empty-job"),
            pytest.param({"parent_id": "ns/job/"}, id="parent-empty-run"),
            pytest.param({"parent_id": _PARENT_ID, "root_parent_id": "ns/job/not-a-uuid"}, id="root-bad-uuid"),
            pytest.param({"parent_id": _PARENT_ID, "root_parent_id": "garbage"}, id="root-garbage"),
            pytest.param({"root_parent_id": _ROOT_PARENT_ID}, id="root-without-parent"),
        ],
    )
    def test_malformed_explicit_ids_raise_value_error(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            OpenLineageExtender(**kwargs)

    def test_env_is_read_when_opted_in(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENLINEAGE_PARENT_ID", _PARENT_ID)
        monkeypatch.setenv("OPENLINEAGE_ROOT_PARENT_ID", _ROOT_PARENT_ID)
        client, transport = ol_capture

        parent = _parent_facet(_start_root(OpenLineageExtender(client=client, parent_from_env=True), transport))

        assert (parent.job.namespace, parent.job.name, parent.run.runId) == ("airflow", "dag.task", _RUN_A)
        assert parent.root is not None
        assert (parent.root.job.namespace, parent.root.job.name, parent.root.run.runId) == ("airflow", "dag", _RUN_B)

    def test_env_is_resolved_once_in_init(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENLINEAGE_PARENT_ID", _PARENT_ID)
        monkeypatch.delenv("OPENLINEAGE_ROOT_PARENT_ID", raising=False)
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client, parent_from_env=True)
        monkeypatch.setenv("OPENLINEAGE_PARENT_ID", _OTHER_PARENT_ID)

        assert _parent_facet(_start_root(extender, transport)).run.runId == _RUN_A

    def test_explicit_parent_wins_over_env_and_env_is_not_read(
        self,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("OPENLINEAGE_PARENT_ID", "malformed")
        monkeypatch.setenv("OPENLINEAGE_ROOT_PARENT_ID", "malformed")
        client, transport = ol_capture

        with caplog.at_level(logging.WARNING):
            extender = OpenLineageExtender(client=client, parent_id=_PARENT_ID, parent_from_env=True)
        parent = _parent_facet(_start_root(extender, transport))

        assert parent.run.runId == _RUN_A
        assert parent.root is not None
        assert parent.root.run.runId == _RUN_A
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    @pytest.mark.parametrize(
        ("variable", "env", "has_parent"),
        [
            pytest.param(
                "OPENLINEAGE_PARENT_ID", {"OPENLINEAGE_PARENT_ID": "secret-malformed-value"}, False, id="parent"
            ),
            pytest.param(
                "OPENLINEAGE_ROOT_PARENT_ID",
                {"OPENLINEAGE_PARENT_ID": _PARENT_ID, "OPENLINEAGE_ROOT_PARENT_ID": "secret-malformed-value"},
                True,
                id="root",
            ),
            pytest.param(
                "OPENLINEAGE_ROOT_PARENT_ID",
                {"OPENLINEAGE_ROOT_PARENT_ID": "secret-malformed-value"},
                False,
                id="root-without-parent",
            ),
        ],
    )
    def test_malformed_env_warns_once_without_the_value(
        self,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        variable: str,
        env: dict[str, str],
        has_parent: bool,
    ) -> None:
        for name in ("OPENLINEAGE_PARENT_ID", "OPENLINEAGE_ROOT_PARENT_ID"):
            if name in env:
                monkeypatch.setenv(name, env[name])
            else:
                monkeypatch.delenv(name, raising=False)
        client, transport = ol_capture

        with caplog.at_level(logging.WARNING):
            extender = OpenLineageExtender(client=client, parent_from_env=True)
        event = _start_root(extender, transport)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert variable in warnings[0].getMessage()
        assert "secret-malformed-value" not in warnings[0].getMessage()
        assert len(transport.events) == 1
        if has_parent:
            parent = _parent_facet(event)
            assert parent.run.runId == _RUN_A
            assert parent.root is not None
            assert parent.root.run.runId == _RUN_A
        else:
            assert event.run.facets is not None
            assert "parent" not in event.run.facets


class TestOpenLineageExtenderInputDataLoadCorrelation:
    """INPUT_DATA_LOAD fires nested inside an already-open CALCULATE_FEATURE invocation."""

    def test_recorded_s3_input_is_mapped_to_bucket_namespace_and_key_with_the_identity_in_the_data_source_facet(
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
        assert (input_dataset.namespace, input_dataset.name) == ("s3://bucket", "key.parquet")

        from openlineage.client.facet_v2 import datasource_dataset

        assert input_dataset.facets is not None
        data_source_facet = input_dataset.facets["dataSource"]
        assert isinstance(data_source_facet, datasource_dataset.DatasourceDatasetFacet)
        assert data_source_facet.name == "s3://bucket/key.parquet"

    # (raw data_access passed as args[0], expected dataSource name (the core identity), expected input dataset
    # (namespace, name), secret markers absent from every event). Expected values are hardcoded literals, pinned
    # once against core's BaseInputData.data_access_identity.
    @pytest.mark.parametrize(
        ("raw", "expected_name", "expected_dataset", "secrets"),
        [
            pytest.param(
                "https://user:pw@host/p/a?sig=SECRET#frag",
                "https://host/p/a",
                ("mloda", "https://host/p/a"),
                ("user:pw", "SECRET", "frag"),
                id="userinfo_query_fragment",
            ),
            pytest.param(
                "s3://bucket/key.parquet?versionId=SECRET",
                "s3://bucket/key.parquet",
                ("s3://bucket", "key.parquet"),
                ("SECRET",),
                id="query",
            ),
            pytest.param(
                "https://host/p?email=a@b.com/x&sig=SECRET",
                "str",
                ("mloda", "str"),
                ("SECRET", "a@b.com"),
                id="at_sign_in_query_value_is_unparseable",
            ),
            pytest.param(
                "postgresql://user:pa?ss@host:5432/db",
                "str",
                ("mloda", "str"),
                ("user:pa", "pa?ss", "user:"),
                id="query_marker_inside_userinfo_is_unparseable",
            ),
            pytest.param(
                "host=db user=u password=hunter2",
                "str",
                ("mloda", "str"),
                ("hunter2",),
                id="keyword_dsn",
            ),
            pytest.param(
                "Server=x;Uid=u;Pwd=hunter2;",
                "str",
                ("mloda", "str"),
                ("hunter2",),
                id="odbc_connection_string",
            ),
        ],
    )
    def test_recorded_input_and_data_source_names_use_the_core_identity_never_the_raw_data_access(
        self,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        raw: str,
        expected_name: str,
        expected_dataset: tuple[str, str],
        secrets: tuple[str, ...],
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        context_identity = BaseInputData.data_access_identity(raw)
        inner_context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=context_identity)

        def inner_func(*_: Any) -> str:
            return "loaded-data"

        def outer_func() -> str:
            with inner_context.activate():
                extender(inner_func, raw)
            return "calculate-result"

        with make_hook_context().activate():
            extender(outer_func)

        complete_event = transport.events[1]
        assert complete_event.inputs is not None
        assert len(complete_event.inputs) == 1
        input_dataset = complete_event.inputs[0]
        assert (input_dataset.namespace, input_dataset.name) == expected_dataset
        assert input_dataset.facets is not None

        from openlineage.client.facet_v2 import datasource_dataset

        data_source_facet = input_dataset.facets["dataSource"]
        assert isinstance(data_source_facet, datasource_dataset.DatasourceDatasetFacet)
        assert data_source_facet.name == expected_name

        from openlineage.client.serde import Serde

        for event in transport.events:
            serialized = Serde.to_json(event)
            for secret in secrets:
                assert secret not in serialized

    @pytest.mark.parametrize(
        ("is_fallback", "expect_facet"),
        [(True, True), (False, False), (None, False)],
        ids=["fallback", "real", "unset"],
    )
    def test_data_access_facet_marks_only_fallback_identities(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], is_fallback: bool | None, expect_facet: bool
    ) -> None:
        import json

        from openlineage.client.serde import Serde

        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        inner_context = make_hook_context(
            hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="str", data_access_identity_is_fallback=is_fallback
        )

        def outer_func() -> str:
            with inner_context.activate():
                extender(lambda *_: "loaded-data", "host=db")
            return "calculate-result"

        with make_hook_context().activate():
            extender(outer_func)

        payload = json.loads(Serde.to_json(transport.events[1]))
        assert len(payload["inputs"]) == 1
        facets = payload["inputs"][0]["facets"]
        if expect_facet:
            assert facets["mlodaDataAccess"]["identityIsFallback"] is True
            assert facets["mlodaDataAccess"]["_producer"] == extender.producer
        else:
            assert "mlodaDataAccess" not in facets

    @pytest.mark.parametrize("flags", [(False, True), (True, False)], ids=["real_then_fallback", "fallback_then_real"])
    def test_any_fallback_load_of_an_identity_marks_the_one_input_dataset(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], flags: tuple[bool, bool]
    ) -> None:
        import json

        from openlineage.client.serde import Serde

        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)

        def outer_func() -> str:
            for flag in flags:
                inner_context = make_hook_context(
                    hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="str", data_access_identity_is_fallback=flag
                )
                with inner_context.activate():
                    extender(lambda *_: "loaded-data", "host=db")
            return "calculate-result"

        with make_hook_context().activate():
            extender(outer_func)

        payload = json.loads(Serde.to_json(transport.events[1]))
        assert len(payload["inputs"]) == 1
        assert payload["inputs"][0]["facets"]["mlodaDataAccess"]["identityIsFallback"] is True

    def test_context_identity_none_records_no_input_even_with_a_str_uri_args0(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        raw = "https://host/p?sig=SECRET"
        inner_context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=None)

        def outer_func() -> str:
            with inner_context.activate():
                extender(lambda *_: "loaded-data", raw)
            return "calculate-result"

        with make_hook_context().activate():
            extender(outer_func)

        complete_event = transport.events[1]
        assert complete_event.inputs == []

        from openlineage.client.serde import Serde

        for event in transport.events:
            assert "SECRET" not in Serde.to_json(event)

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

    def test_stack_is_restored_after_a_calculate_that_raises(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)

        def failing_body() -> None:
            raise RuntimeError("calculate boom")

        with make_hook_context().activate():
            with pytest.raises(RuntimeError, match="calculate boom"):
                extender(failing_body)
        events_after_failure = len(transport.events)

        with caplog.at_level(logging.DEBUG):
            with make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="standalone").activate():
                result = extender(lambda: "loaded")

        assert result == "loaded"
        assert len(transport.events) == events_after_failure

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
        composite = CompositeExtender([extender_a, extender_b])

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
            assert (inputs[0].namespace, inputs[0].name) == ("s3://bucket", "key.parquet")

    def test_nested_calculate_attributes_each_load_to_its_own_level(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        outer_class = "mloda.testing.OuterFeatureGroup"
        inner_class = "mloda.testing.InnerFeatureGroup"

        def load(identity: str) -> None:
            with make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=identity).activate():
                extender(lambda: "loaded")

        def inner_body() -> None:
            load("s3://bucket/inner.parquet")

        def outer_body() -> None:
            with make_hook_context(feature_group_class=inner_class).activate():
                extender(inner_body)
            load("s3://bucket/outer.parquet")

        with make_hook_context(feature_group_class=outer_class).activate():
            extender(outer_body)

        complete_by_job = {e.job.name: e for e in transport.events if e.eventType == RunState.COMPLETE}
        assert [(i.namespace, i.name) for i in complete_by_job[inner_class].inputs or []] == [
            ("s3://bucket", "inner.parquet")
        ]
        assert [(i.namespace, i.name) for i in complete_by_job[outer_class].inputs or []] == [
            ("s3://bucket", "outer.parquet")
        ]


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

    def test_identities_differing_only_in_query_produce_single_input(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        raw_one = "https://host/p?token=AAA"
        raw_two = "https://host/p?token=BBB"
        context_identity = BaseInputData.data_access_identity(raw_one)
        assert context_identity == BaseInputData.data_access_identity(raw_two)

        def load(raw: str) -> None:
            with make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=context_identity).activate():
                extender(lambda *_: "loaded", raw)

        def calculate_body() -> str:
            load(raw_one)
            load(raw_two)
            return "calculated"

        with make_hook_context().activate():
            extender(calculate_body)

        complete_event = transport.events[-1]
        assert complete_event.eventType == RunState.COMPLETE
        assert complete_event.inputs is not None
        assert [i.name for i in complete_event.inputs] == ["https://host/p"]

    def test_input_feature_matching_a_data_load_identity_is_reported_once(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client, dataset_namespace="custom-ds")
        inner_context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="shared")

        def inner_loader() -> str:
            return "loaded-data"

        def calculate_body() -> str:
            with inner_context.activate():
                extender(inner_loader)
            return "calculated"

        with make_hook_context(input_features=frozenset({"shared", "other"})).activate():
            extender(calculate_body)

        complete_event = transport.events[-1]
        assert complete_event.eventType == RunState.COMPLETE
        assert complete_event.inputs is not None
        assert sorted((i.namespace, i.name) for i in complete_event.inputs) == [
            ("custom-ds", "other"),
            ("custom-ds", "shared"),
        ]

    def test_uri_shaped_input_feature_is_published_as_given_and_not_merged_with_a_stripped_load(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        raw = "https://host/p?v=1"
        context_identity = BaseInputData.data_access_identity(raw)  # "https://host/p"
        inner_context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=context_identity)

        def inner_loader(*_: Any) -> str:
            return "loaded-data"

        def calculate_body() -> str:
            with inner_context.activate():
                extender(inner_loader, raw)
            return "calculated"

        with make_hook_context(input_features=frozenset({raw})).activate():
            extender(calculate_body)

        complete_event = transport.events[-1]
        assert complete_event.eventType == RunState.COMPLETE
        assert complete_event.inputs is not None
        assert sorted(i.name for i in complete_event.inputs) == sorted([context_identity, raw])

    def test_two_azure_containers_on_one_account_become_two_distinct_input_datasets(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = OpenLineageExtender(client=client)
        marker = "SECRET"
        raw_container = f"abfss://raw@acct.dfs.core.windows.net/p?sv=1&sig={marker}"
        curated_container = f"abfss://curated@acct.dfs.core.windows.net/p?sv=1&sig={marker}"

        def load(raw: str) -> None:
            identity = BaseInputData.data_access_identity(raw)
            with make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=identity).activate():
                extender(lambda *_: "loaded", raw)

        def calculate_body() -> str:
            load(raw_container)
            load(curated_container)
            return "calculated"

        with make_hook_context().activate():
            extender(calculate_body)

        complete_event = transport.events[-1]
        assert complete_event.eventType == RunState.COMPLETE
        assert complete_event.inputs is not None
        assert sorted((i.namespace, i.name) for i in complete_event.inputs) == sorted(
            [
                ("abfss://raw@acct.dfs.core.windows.net", "p"),
                ("abfss://curated@acct.dfs.core.windows.net", "p"),
            ]
        )

        from openlineage.client.serde import Serde

        for event in transport.events:
            assert marker not in Serde.to_json(event)


class TestOpenLineageExtenderRunAll:
    """End-to-end wiring through mloda.user.mloda.run_all: RunEvents carry the real feature group's job name."""

    def test_run_all_events_use_feature_group_qualified_name_as_job_name(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture

        run_value_int(OpenLineageExtender(client=client))

        expected_job_name = f"{PyArrowDataOpsTestDataCreator.__module__}.{PyArrowDataOpsTestDataCreator.__qualname__}"
        assert any(e.job.name == expected_job_name for e in transport.events)

    def test_job_name_falls_back_to_the_owning_feature_group_when_the_context_has_no_class(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture

        class OwnerFeatureGroup:
            @classmethod
            def calculate_feature(cls) -> None:
                return None

        with make_hook_context(feature_group_class=None).activate():
            OpenLineageExtender(client=client)(OwnerFeatureGroup.calculate_feature)

        assert transport.events
        assert {e.job.name for e in transport.events} == {"OwnerFeatureGroup"}

    def test_run_all_complete_event_carries_real_schema_facet(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        """End-to-end: `DataOperationsTestDataCreator.calculate_feature` (which `run_value_int` drives)
        returns a plain dict, so core's dict-interchange output_schema path (`_dict_output_schema` /
        `_python_dtype`) is what feeds `instrument()` -> `context.output_schema` -> the extender here,
        not the PyArrow arrow-schema path (`arrow_schema_output_schema` / `_extract_column_dtype`); that
        path is proven separately in `test_schema_facet_uses_output_schema_populated_during_call`. This
        test still proves the wiring is a real `mloda.run_all`, not a HookContext fixture mock."""
        client, transport = ol_capture

        run_value_int(OpenLineageExtender(client=client))

        complete_events = [e for e in transport.events if e.eventType == RunState.COMPLETE]
        assert complete_events

        schema_types: list[str] = []
        for event in complete_events:
            for output in event.outputs or []:
                if output.name != "value_int":
                    continue
                facets = output.facets or {}
                schema_facet = facets.get("schema")
                if isinstance(schema_facet, schema_dataset.SchemaDatasetFacet) and schema_facet.fields:
                    schema_types.extend(f.type for f in schema_facet.fields if f.type is not None)

        # value_int's raw fixture values are plain python ints, so `_python_dtype` (type(value).__name__)
        # yields exactly "int" via the dict-interchange path; a silent regression to the arrow-schema
        # path (e.g. "int64") or to a garbage/empty type must fail loudly, not pass on a loose substring.
        assert schema_types == ["int"]


def _breaker_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == openlineage_extender_module.__name__
        and r.levelno == logging.WARNING
        and "skips new steps" in r.getMessage()
    ]


class TestOpenLineageExtenderEmitBreaker:
    """After a transport failure in a run the extender skips new steps' emission for that run (one WARNING per
    trip), a different run retries, a run_id of None never trips it, on_run_complete resets it, and other
    errors or raise_on_error=True never trip it."""

    @staticmethod
    def _composite_call(composite: CompositeExtender, run_id: str | None, sentinel: object) -> object:
        with make_hook_context(run_id=run_id).activate():
            return composite(lambda: sentinel)

    def test_one_run_makes_one_emit_attempt_and_logs_one_breaker_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        composite = CompositeExtender([extender])
        sentinel = object()
        run_id = str(uuid.uuid4())

        with caplog.at_level(logging.WARNING):
            results = [self._composite_call(composite, run_id, sentinel) for _ in range(3)]

        assert all(result is sentinel for result in results)
        assert transport.emit_attempts == 1
        breaker = _breaker_records(caplog)
        assert len(breaker) == 1
        message = breaker[0].getMessage()
        assert "OpenLineageExtender" in message
        assert run_id in message
        assert "ConnectionError" in message
        module_records = [r for r in caplog.records if r.name == openlineage_extender_module.__name__]
        assert all(_EMIT_ERROR_MESSAGE not in r.getMessage() for r in module_records)

    def test_a_different_run_id_retries_and_trips_again(self, caplog: pytest.LogCaptureFixture) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        composite = CompositeExtender([extender])
        sentinel = object()

        with caplog.at_level(logging.WARNING):
            self._composite_call(composite, _RUN_A, sentinel)
            self._composite_call(composite, _RUN_A, sentinel)
            self._composite_call(composite, _RUN_B, sentinel)
            self._composite_call(composite, _RUN_B, sentinel)

        assert transport.emit_attempts == 2
        assert len(_breaker_records(caplog)) == 2

    def test_interleaved_runs_each_trip_once_and_on_run_complete_frees_only_its_run(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        composite = CompositeExtender([extender])
        sentinel = object()

        with caplog.at_level(logging.DEBUG, logger=openlineage_extender_module.__name__):
            for run_id in (_RUN_A, _RUN_B, _RUN_A, _RUN_B, _RUN_A, _RUN_B):
                self._composite_call(composite, run_id, sentinel)

            assert transport.emit_attempts == 2
            assert len(_breaker_records(caplog)) == 2
            skipped = [
                r
                for r in caplog.records
                if r.name == openlineage_extender_module.__name__
                and r.levelno == logging.DEBUG
                and "OpenLineageExtender" in r.getMessage()
                and _RUN_A in r.getMessage()
            ]
            assert skipped

            extender.on_run_complete(RunContext(run_id=_RUN_A), LifecycleOutcome(status="succeeded"))
            self._composite_call(composite, _RUN_B, sentinel)
            assert transport.emit_attempts == 2
            self._composite_call(composite, _RUN_A, sentinel)
            assert transport.emit_attempts == 3

    def test_run_id_none_never_trips_the_breaker(self, caplog: pytest.LogCaptureFixture) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        composite = CompositeExtender([extender])
        sentinel = object()

        with caplog.at_level(logging.WARNING):
            results = [self._composite_call(composite, None, sentinel) for _ in range(3)]

        assert all(result is sentinel for result in results)
        assert transport.emit_attempts == 3
        assert _breaker_records(caplog) == []

    def test_on_run_complete_for_the_failed_run_resets_the_breaker(self, caplog: pytest.LogCaptureFixture) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        composite = CompositeExtender([extender])
        sentinel = object()

        with caplog.at_level(logging.WARNING):
            self._composite_call(composite, _RUN_X, sentinel)
            self._composite_call(composite, _RUN_X, sentinel)
            assert transport.emit_attempts == 1

            extender.on_run_complete(RunContext(run_id=_RUN_X), LifecycleOutcome(status="succeeded"))
            self._composite_call(composite, _RUN_X, sentinel)

        assert transport.emit_attempts == 2
        assert len(_breaker_records(caplog)) == 2

    def test_on_run_complete_for_another_run_does_not_reset_the_breaker(self) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        composite = CompositeExtender([extender])
        sentinel = object()

        self._composite_call(composite, _RUN_X, sentinel)
        extender.on_run_complete(RunContext(run_id=_RUN_OTHER), LifecycleOutcome(status="succeeded"))
        self._composite_call(composite, _RUN_X, sentinel)

        assert transport.emit_attempts == 1

    def test_raise_on_error_raises_on_every_step_and_records_no_trip(self) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport), raise_on_error=True)
        sentinel = object()

        with make_hook_context(run_id=_RUN_X).activate():
            for _ in range(3):
                with pytest.raises(ConnectionError):
                    extender(lambda: sentinel)

        assert transport.emit_attempts == 3
        assert extender._tripped_runs == {}

    @pytest.mark.parametrize(
        ("error_factory", "expected_attempts", "expected_warnings"),
        [
            pytest.param(lambda: RuntimeError(_EMIT_ERROR_MESSAGE), 3, 0, id="runtime_error"),
            pytest.param(lambda: TypeError(_EMIT_ERROR_MESSAGE), 3, 0, id="type_error"),
            pytest.param(lambda: _http_error(400), 3, 0, id="http_400"),
            pytest.param(lambda: _http_error(503), 1, 1, id="http_503"),
            pytest.param(lambda: _http_error(408), 1, 1, id="http_408"),
            pytest.param(lambda: _http_error(429), 1, 1, id="http_429"),
            pytest.param(lambda: OSError(_EMIT_ERROR_MESSAGE), 1, 1, id="os_error"),
            pytest.param(_runtime_error_from_connection_error, 1, 1, id="runtime_error_from_connection_error"),
            pytest.param(_runtime_error_in_handler_of_connection_error, 1, 1, id="runtime_error_implicit_context"),
            pytest.param(_runtime_error_from_none_after_connection_error, 3, 0, id="runtime_error_from_none"),
        ],
    )
    def test_only_transport_errors_trip_the_breaker(
        self,
        error_factory: Callable[[], BaseException],
        expected_attempts: int,
        expected_warnings: int,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        transport = _FailingEmitTransport(error_factory)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        composite = CompositeExtender([extender])
        sentinel = object()

        with caplog.at_level(logging.WARNING):
            results = [self._composite_call(composite, _RUN_X, sentinel) for _ in range(3)]

        assert all(result is sentinel for result in results)
        assert transport.emit_attempts == expected_attempts
        assert len(_breaker_records(caplog)) == expected_warnings

    def test_a_skipped_start_still_runs_func_returns_its_result_and_emits_no_terminal_event(self) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        composite = CompositeExtender([extender])
        sentinel = object()
        self._composite_call(composite, _RUN_X, sentinel)
        assert transport.emit_attempts == 1
        func_calls = 0

        def func() -> object:
            nonlocal func_calls
            func_calls += 1
            return sentinel

        with make_hook_context(run_id=_RUN_X).activate():
            result = composite(func)

        assert result is sentinel
        assert func_calls == 1
        assert transport.emit_attempts == 1

    def test_a_skipped_start_propagates_funcs_exception_unchanged_without_a_terminal_event(self) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        composite = CompositeExtender([extender])
        self._composite_call(composite, _RUN_X, object())
        error = ValueError("func boom")

        def func() -> None:
            raise error

        with make_hook_context(run_id=_RUN_X).activate():
            with pytest.raises(ValueError) as exc_info:
                composite(func)

        assert exc_info.value is error
        assert transport.emit_attempts == 1

    def test_a_started_step_still_emits_its_terminal_event_after_another_step_tripped_the_run(self) -> None:
        transport = _FailingEmitTransport(_connection_error, succeed_first=1)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        composite = CompositeExtender([extender])
        sentinel = object()

        def step_b() -> object:
            return self._composite_call(composite, _RUN_X, sentinel)

        with make_hook_context(run_id=_RUN_X).activate():
            result = composite(step_b)

        assert result is sentinel
        assert transport.emit_attempts == 3

    def test_an_expired_trip_reprobes_and_a_failing_probe_trips_again(self, caplog: pytest.LogCaptureFixture) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        extender._BREAKER_RETRY_AFTER = 0.0
        composite = CompositeExtender([extender])
        sentinel = object()

        with caplog.at_level(logging.WARNING):
            self._composite_call(composite, _RUN_X, sentinel)
            self._composite_call(composite, _RUN_X, sentinel)

        assert transport.emit_attempts == 2
        assert len(_breaker_records(caplog)) == 2

    def test_an_expired_trip_reprobes_and_a_successful_probe_leaves_the_run_closed(self) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        extender._BREAKER_RETRY_AFTER = 0.0
        composite = CompositeExtender([extender])
        sentinel = object()
        self._composite_call(composite, _RUN_X, sentinel)
        assert transport.emit_attempts == 1

        transport.failing = False
        extender._BREAKER_RETRY_AFTER = 60.0
        extender._tripped_runs[_RUN_X] = time.monotonic() - 1000.0
        self._composite_call(composite, _RUN_X, sentinel)
        attempts_after_probe = transport.emit_attempts
        self._composite_call(composite, _RUN_X, sentinel)

        assert _RUN_X not in extender._tripped_runs
        assert attempts_after_probe == 3
        assert transport.emit_attempts == 5

    def test_a_trip_prunes_expired_entries(self) -> None:
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        composite = CompositeExtender([extender])
        stale = time.monotonic() - 1000.0
        extender._tripped_runs.update({f"stale-{index}": stale for index in range(5)})

        self._composite_call(composite, _RUN_A, object())

        assert set(extender._tripped_runs) == {_RUN_A}

    def test_copy_and_pickle_start_with_an_empty_trip_table_that_is_not_shared(self) -> None:
        extender = OpenLineageExtender()
        extender._tripped_runs[_RUN_A] = time.monotonic()

        shallow = copy.copy(extender)
        unpickled = pickle.loads(pickle.dumps(extender))  # nosec

        assert shallow._tripped_runs == {}
        assert unpickled._tripped_runs == {}
        assert shallow._tripped_runs is not extender._tripped_runs
        shallow._tripped_runs[_RUN_B] = time.monotonic()
        assert set(extender._tripped_runs) == {_RUN_A}

    def test_prepared_session_run_twice_attempts_emission_once_per_run(self) -> None:
        """Each run of a prepared session mints a fresh run_id, so the breaker never carries over. The chained
        feature group gives each run two calculate steps, so a stale or missing breaker shows as extra attempts."""
        transport = _FailingEmitTransport(_connection_error)
        extender = OpenLineageExtender(client=OpenLineageClient(transport=transport))
        feature_group = MlodaTestingValueIntPlusOne
        session = mloda.prepare(
            [feature_group.get_class_name()],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, feature_group}),
            function_extender={extender},
        )

        session.run()
        assert transport.emit_attempts == 1

        session.run()
        assert transport.emit_attempts == 2


class TestOpenLineageExtenderConsoleFallbackWarning:
    """use_sdk_defaults with no ambient OpenLineage config falls back to the SDK console transport; the SDK
    warns once per built client and the extender adds no warning of its own."""

    @pytest.fixture
    def clean_ambient_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        for name in list(os.environ):
            if name.startswith(_OPENLINEAGE_ENV_PREFIXES):
                monkeypatch.delenv(name)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))

    @staticmethod
    def _sdk_console_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
        return [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "will print events to console" in r.getMessage()
        ]

    @staticmethod
    def _extender_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
        return [
            r for r in caplog.records if r.name == openlineage_extender_module.__name__ and r.levelno == logging.WARNING
        ]

    def test_fallback_logs_the_sdk_warning_once_per_built_client_and_none_from_the_extender(
        self, clean_ambient_config: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        extender = OpenLineageExtender(use_sdk_defaults=True)

        with caplog.at_level(logging.WARNING):
            with make_hook_context().activate():
                extender(lambda: None)
            with make_hook_context().activate():
                extender(lambda: None)

        extender.close()
        assert len(self._sdk_console_records(caplog)) == 1
        assert self._extender_warnings(caplog) == []

    def test_injected_client_logs_no_console_warning(
        self,
        clean_ambient_config: None,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        client, _ = ol_capture
        extender = OpenLineageExtender(client=client, use_sdk_defaults=True)

        with caplog.at_level(logging.WARNING):
            with make_hook_context().activate():
                extender(lambda: None)

        assert self._sdk_console_records(caplog) == []
        assert self._extender_warnings(caplog) == []


class TestOpenLineageExtenderSubclassSeams:
    """The seams a subclass builds on (producer, _dispatch, _calculate_run_facets, _calculate_output_facets); the
    defaults leave the emitter's behavior unchanged."""

    def test_default_producer_is_the_community_package_url(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture

        with make_hook_context().activate():
            OpenLineageExtender(client=client)(lambda: None)

        assert OpenLineageExtender.producer == _DEFAULT_PRODUCER
        assert transport.events
        assert all(event.producer == _DEFAULT_PRODUCER for event in transport.events)

    def test_producer_override_reaches_events_and_every_facet_the_class_builds(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = _CustomProducerExtender(client=client)
        inner_context = make_hook_context(
            hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="s3://bucket/key.parquet"
        )

        def outer_func() -> None:
            with inner_context.activate():
                extender(lambda: "loaded-data")

        with make_hook_context(run_id=str(uuid.uuid4()), output_schema=(("value_int", "int64"),)).activate():
            extender(outer_func)

        start_payload, complete_payload = (json.loads(Serde.to_json(event)) for event in transport.events)
        assert start_payload["producer"] == _CUSTOM_PRODUCER
        assert complete_payload["producer"] == _CUSTOM_PRODUCER
        assert start_payload["run"]["facets"]["parent"]["_producer"] == _CUSTOM_PRODUCER
        assert complete_payload["run"]["facets"]["parent"]["_producer"] == _CUSTOM_PRODUCER
        assert complete_payload["inputs"][0]["facets"]["dataSource"]["_producer"] == _CUSTOM_PRODUCER
        assert complete_payload["outputs"][0]["facets"]["schema"]["_producer"] == _CUSTOM_PRODUCER

    def test_producer_override_reaches_the_fail_event(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = _CustomProducerExtender(client=client)

        def failing_body() -> None:
            raise RuntimeError("calculate boom")

        with make_hook_context(run_id=str(uuid.uuid4())).activate():
            with pytest.raises(RuntimeError, match="calculate boom"):
                extender(failing_body)

        fail_payload = json.loads(Serde.to_json(transport.events[-1]))
        assert fail_payload["eventType"] == "FAIL"
        assert fail_payload["producer"] == _CUSTOM_PRODUCER
        assert fail_payload["run"]["facets"]["parent"]["_producer"] == _CUSTOM_PRODUCER

    def test_dispatch_runs_only_after_the_inert_and_no_context_guards(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, _ = ol_capture
        inert = _RecordingDispatchExtender()
        live = _RecordingDispatchExtender(client=client)

        with make_hook_context().activate():
            assert inert(lambda: 1) == 1
        assert live(lambda: 2) == 2

        assert inert.dispatched == []
        assert live.dispatched == []

        with make_hook_context().activate():
            assert live(lambda: 3) == 3
        assert len(live.dispatched) == 1

    def test_dispatch_receives_the_ambient_context_and_the_call_unpacked(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, _ = ol_capture
        extender = _RecordingDispatchExtender(client=client)
        context = make_hook_context()

        def func(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return args, kwargs

        with context.activate():
            result = extender(func, "positional", keyword="value")

        assert result == (("positional",), {"keyword": "value"})
        assert extender.dispatched == [(context, func, ("positional",), {"keyword": "value"})]

    @pytest.mark.parametrize("hook", [hook for hook in ExtenderHook if hook != ExtenderHook.INPUT_DATA_LOAD])
    def test_default_dispatch_routes_every_hook_but_input_data_load_to_the_calculate_lifecycle(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], hook: ExtenderHook
    ) -> None:
        client, transport = ol_capture
        extender = _RecordingDispatchExtender(client=client)

        with make_hook_context(hook=hook).activate():
            extender(lambda: None)

        assert [context.hook for context, *_ in extender.dispatched] == [hook]
        assert [event.eventType for event in transport.events] == [RunState.START, RunState.COMPLETE]

    def test_default_dispatch_routes_input_data_load_to_the_open_calculate_invocation(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = _RecordingDispatchExtender(client=client)
        inner_context = make_hook_context(
            hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="s3://bucket/key.parquet"
        )

        def outer_func() -> None:
            with inner_context.activate():
                extender(lambda: "loaded-data")

        with make_hook_context().activate():
            extender(outer_func)

        hooks = [context.hook for context, *_ in extender.dispatched]
        assert hooks == [ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD]
        assert [event.eventType for event in transport.events] == [RunState.START, RunState.COMPLETE]
        assert [(i.namespace, i.name) for i in transport.events[-1].inputs or []] == [("s3://bucket", "key.parquet")]

    def test_calculate_run_facets_adds_a_facet_next_to_parent_on_start_and_terminal_events(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        run_id = str(uuid.uuid4())
        extender = _RunFacetExtender(client=client)
        context = make_hook_context(run_id=run_id)

        def func(*args: Any) -> str:
            return "result"

        with context.activate():
            result = extender(func, "positional")

        assert result == "result"
        assert extender.run_facet_calls == [(context, func, ("positional",))]
        assert [event.eventType for event in transport.events] == [RunState.START, RunState.COMPLETE]
        for event in transport.events:
            facets = event.run.facets or {}
            parent = facets.get("parent")
            assert isinstance(parent, parent_run.ParentRunFacet)
            assert parent.run.runId == run_id
            assert isinstance(facets.get("probe"), nominal_time_run.NominalTimeRunFacet)

    def test_calculate_output_facets_merge_next_to_schema_on_each_complete_output(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = _OutputFacetExtender(client=client)
        context = make_hook_context(
            feature_names=("value_int", "value_str"),
            output_schema=(("value_int", "int64"), ("value_str", "string")),
            input_features=frozenset({"src"}),
        )
        inner_context = make_hook_context(
            hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="s3://bucket/key.parquet"
        )

        def func(*args: Any) -> None:
            with inner_context.activate():
                extender(lambda: "loaded-data")

        with context.activate():
            extender(func, "positional")

        complete_event = transport.events[-1]
        assert complete_event.eventType == RunState.COMPLETE
        assert [(i.namespace, i.name) for i in complete_event.inputs or []] == [
            ("mloda", "src"),
            ("s3://bucket", "key.parquet"),
        ]
        # The seam gets the COMPLETE event's inputs: declared first, loaded second.
        assert extender.output_facet_calls == [
            (context, func, ("positional",), "value_int", complete_event.inputs),
            (context, func, ("positional",), "value_str", complete_event.inputs),
        ]
        assert complete_event.outputs is not None
        assert [output.name for output in complete_event.outputs] == ["value_int", "value_str"]
        for output in complete_event.outputs:
            facets = output.facets or {}
            assert isinstance(facets.get("schema"), schema_dataset.SchemaDatasetFacet)
            probe = facets.get("probe")
            assert isinstance(probe, documentation_dataset.DocumentationDatasetFacet)
            assert probe.description == f"probe:{output.name}"

    def test_output_facets_seam_failure_never_corrupts_the_result(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], caplog: pytest.LogCaptureFixture
    ) -> None:
        """A raising seam is logged, not fatal: COMPLETE is still emitted with the base (schema) facets only."""
        client, transport = ol_capture
        extender = _FailingOutputFacetExtender(client=client)
        context = make_hook_context(feature_names=("value_int",), output_schema=(("value_int", "int64"),))

        with context.activate():
            with caplog.at_level(logging.WARNING):
                result = extender(lambda: 42)

        assert result == 42
        assert [event.eventType for event in transport.events] == [RunState.START, RunState.COMPLETE]
        complete_event = transport.events[-1]
        assert [output.name for output in complete_event.outputs or []] == ["value_int"]
        for output in complete_event.outputs or []:
            assert set(output.facets or {}) == {"schema"}
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(type(extender).__name__ in message and "RuntimeError" in message for message in warnings)
        assert "output facet boom" not in caplog.text

    def test_post_call_instrumentation_failure_logs_only_the_type_and_keeps_the_result(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, _ = ol_capture
        extender = OpenLineageExtender(client=client)
        original_emit_event = extender._emit_event

        def emit_event(state: RunState, *args: Any) -> bool:
            if state == RunState.COMPLETE:
                raise RuntimeError("complete boom")
            return original_emit_event(state, *args)

        with patch.object(extender, "_emit_event", side_effect=emit_event):
            with make_hook_context().activate():
                with caplog.at_level(logging.WARNING):
                    result = extender(lambda: 42)

        assert result == 42
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("post-call instrumentation failed" in m and "RuntimeError" in m for m in warnings), warnings
        assert "complete boom" not in caplog.text

    def test_log_messages_name_the_subclass_not_the_base(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, _ = ol_capture
        name = _CustomProducerExtender.__name__

        def failing_body() -> None:
            raise RuntimeError("calculate boom")

        with caplog.at_level(logging.DEBUG, logger=openlineage_extender_module.logger.name):
            with make_hook_context().activate():
                _CustomProducerExtender()(lambda: None)
                with pytest.raises(RuntimeError, match="calculate boom"):
                    _CustomProducerExtender(client=client)(failing_body)
            with make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity="standalone").activate():
                _CustomProducerExtender(client=client)(lambda: "loaded")
            pickle.dumps(_CustomProducerExtender(client=OpenLineageClient(transport=LockHoldingTransport())))

        messages = [r.message for r in caplog.records if r.name == openlineage_extender_module.logger.name]
        assert any(name in m and "inert" in m.lower() for m in messages), messages
        assert any(name in m and "RuntimeError" in m for m in messages), messages
        assert "calculate boom" not in caplog.text
        assert any(name in m and "calculate" in m.lower() and ("enclosing" in m or "open" in m) for m in messages), (
            messages
        )
        assert any(name in m and "picklable" in m for m in messages), messages
        assert not [m for m in messages if "OpenLineageExtender" in m], messages

    def test_seam_table_names_exactly_the_documented_method_seams(self) -> None:
        assert set(OPENLINEAGE_EXTENDER_SEAMS) == {
            "_dispatch",
            "_call_input_data_load",
            "_call_calculate_feature",
            "_calculate_run_facets",
            "_calculate_output_facets",
            "_run_with_events",
        }

    def test_seam_checker_accepts_the_base_and_a_subclass_adding_new_private_methods(self) -> None:
        assert_openlineage_extender_seams(OpenLineageExtender, OpenLineageExtender)
        assert_openlineage_extender_seams(_AddsPrivateMethodExtender, OpenLineageExtender)
        assert_openlineage_extender_seams(_RunFacetExtender, OpenLineageExtender)
        assert_openlineage_extender_seams(_OutputFacetExtender, OpenLineageExtender)
        assert_openlineage_extender_seams(_RecordingDispatchExtender, OpenLineageExtender)

    @pytest.mark.parametrize(
        "extender_class",
        [
            _OverridesNonSeamExtender,
            _ReshapedSeamExtender,
            _ExtraSeamParameterExtender,
            _DropsDatasetNamespaceExtender,
            _StaticmethodOverrideExtender,
            _PropertyOverrideExtender,
            _DropsSeamDefaultExtender,
            _OverridesGetstateExtender,
            _CallsPrivateEmitExtender,
            _ReadsPrivateClientExtender,
            _UsesPrivateModuleGlobalExtender,
            _UsesPrivateModuleNameExtender,
            _DefinesDeepcopyExtender,
            _OverridesSetattrExtender,
            _CallsPrivateViaBaseClassExtender,
            _ReadsPrivateClassAttributeExtender,
            _OverridesPrivateConstantExtender,
        ],
        ids=[
            "non_seam_override",
            "renamed_parameter",
            "extra_parameter",
            "dropped_attribute",
            "staticmethod_override",
            "property_override",
            "dropped_default",
            "dunder_override",
            "private_call",
            "private_attribute",
            "private_module_attribute",
            "private_module_name",
            "deepcopy_override",
            "setattr_override",
            "private_via_base_class",
            "private_via_type",
            "private_constant_override",
        ],
    )
    def test_seam_checker_rejects_a_subclass_that_breaks_a_seam(
        self, extender_class: type[OpenLineageExtender]
    ) -> None:
        with pytest.raises(AssertionError):
            assert_openlineage_extender_seams(extender_class, OpenLineageExtender)
