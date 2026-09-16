"""Tests for OpenLineageExtender: contract compliance via OpenLineageExtenderTestMixin, plus
schema-facet, namespace-override, dedupe and per-instance-attribution checks not covered by the
mixin.

Direct __call__ tests below wrap calls in a manually built HookContext.activate() scope, mirroring
core's INPUT_DATA_LOAD nesting inside the enclosing CALCULATE_FEATURE HookContext.
"""

from __future__ import annotations

import atexit
import gc
import logging
import pickle  # nosec
import threading
import time
import uuid
import weakref
from collections.abc import Iterator
from typing import Any, cast

import pyarrow as pa
import pytest
from mloda.core.abstract_plugins.hook_context import instrument  # no public equivalent yet
from mloda.steward import CompositeExtender, ExtenderHook
from mloda.user import ParallelizationMode

from mloda.community.extenders.openlineage import openlineage_extender as openlineage_extender_module
from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.openlineage import (
    LockHoldingTransport,
    OpenLineageExtenderTestMixin,
    RecordingTransport,
    make_recording_client,
)
from mloda.testing.extenders.runners import expected_value_int, run_value_int
from openlineage.client.client import OpenLineageClient
from openlineage.client.event_v2 import RunState
from openlineage.client.facet_v2 import parent_run, schema_dataset
from openlineage.client.transport.transport import Config, Transport


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
    """An injected client is pickled as-is if it can be, else trial-pickle drops it with a warning;
    a self-built one is rebuilt by the copy."""

    def test_pickle_round_trip_keeps_config(self, ol_capture: tuple[OpenLineageClient, RecordingTransport]) -> None:
        client, _ = ol_capture
        extender = OpenLineageExtender(
            client=client,
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

    def test_unpicklable_injected_client_is_dropped_on_pickle_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        extender = OpenLineageExtender(client=OpenLineageClient(transport=LockHoldingTransport()))

        with caplog.at_level(logging.WARNING):
            copy = pickle.loads(pickle.dumps(extender))  # nosec

        assert copy._client is None
        warnings = [
            r.message for r in caplog.records if r.levelno == logging.WARNING and "OpenLineageExtender" in r.message
        ]
        assert any("client" in message.lower() for message in warnings), warnings
        # The generic "could not be pickled" sentence alone gives no clue why; the underlying
        # trial-pickle exception's type name must be present too (pickling LockHoldingTransport's
        # threading.Lock always raises TypeError).
        assert any("TypeError" in message for message in warnings), warnings

    def test_dropped_injected_client_copy_believes_it_owns_its_client(self) -> None:
        """After a drop, the copy must recognize it now owns/self-builds its client, not still think
        a client was injected - else a second pickle of the copy would wrongly treat its self-built
        client as "injected" and try (and fail) to pickle it as-is again."""
        extender = OpenLineageExtender(
            client=OpenLineageClient(transport=LockHoldingTransport()), use_sdk_defaults=True
        )

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        assert copy._owns_client is True

    def test_picklable_injected_client_survives_pickling_with_no_warning(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, _ = ol_capture
        extender = OpenLineageExtender(client=client)

        with caplog.at_level(logging.WARNING):
            copy = pickle.loads(pickle.dumps(extender))  # nosec

        assert copy._client is not None
        warnings = [
            r.message for r in caplog.records if r.levelno == logging.WARNING and "OpenLineageExtender" in r.message
        ]
        client_warnings = [message for message in warnings if "client" in message.lower()]
        assert client_warnings == [], client_warnings

    def test_self_built_client_is_dropped_on_pickle_and_rebuilt_by_copy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openlineage_extender_module, "OpenLineageClient", _PicklableFakeClient)
        registered: list[Any] = []
        monkeypatch.setattr(atexit, "register", lambda *args, **kwargs: registered.append(args[0]))
        extender = OpenLineageExtender(use_sdk_defaults=True)
        extender._get_client()

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        assert copy._client is None
        assert isinstance(copy._get_client(), _PicklableFakeClient)
        assert len(registered) == 2
        assert registered[1].__self__ is copy

    def test_client_published_before_ownership_flag_is_still_dropped_on_pickle(self) -> None:
        """Ownership must follow from use_sdk_defaults with no injected client, not from a flag that
        could lag a lazy build publishing _client."""
        extender = OpenLineageExtender(use_sdk_defaults=True)
        extender._client = cast(OpenLineageClient, _PicklableFakeClient())

        copy = pickle.loads(pickle.dumps(extender))  # nosec

        assert copy._client is None


class TestOpenLineageExtenderInProcessIdentityPreservation:
    """Core never pickles extenders under SYNC or THREADING (no real subprocess), so an injected
    client must receive real pipeline events over a genuine run_all, not just survive a manual
    pickle round trip."""

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    def test_run_all_emits_into_the_exact_injected_client(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], mode: ParallelizationMode
    ) -> None:
        client, transport = ol_capture

        values = run_value_int(OpenLineageExtender(client=client), parallelization_modes={mode})

        assert values == expected_value_int()
        assert transport.events, "no events reached the injected client's transport"
        assert transport.events[0].eventType == RunState.START
        assert transport.events[-1].eventType == RunState.COMPLETE


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
            results["first"] = extender.close()

        def close_second() -> None:
            results["second"] = extender.close()

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
        registered: list[Any] = []
        monkeypatch.setattr(atexit, "register", lambda *args, **kwargs: registered.append(args[0]))
        extender = OpenLineageExtender(use_sdk_defaults=True)
        results: dict[str, bool] = {}

        def build() -> None:
            extender._get_client()

        def closer() -> None:
            results["close"] = extender.close()

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

        result = extender.close()

        assert result is True
        assert client.close_calls == [-1.0]

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
            results["a"] = extender_a.close()

        def close_b() -> None:
            results["b"] = extender_b.close()

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
            extender_a.close()

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
        """The copy logs its own inert state instead of inheriting `_logged_inert` from the original."""
        extender = OpenLineageExtender()
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
