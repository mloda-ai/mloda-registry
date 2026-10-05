"""Self-tests for mloda.testing.extenders.otel helpers, plus a negative test proving the mixin's default
own_failure() detects a fault."""

from __future__ import annotations

import dataclasses
import pickle  # nosec
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("opentelemetry.sdk")

from mloda.steward import Extender, ExtenderHook, HookContext
from opentelemetry import propagate, trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import NonRecordingSpan, SpanContext, Status, StatusCode, TraceFlags, set_span_in_context

from mloda.community.extenders.otel import OtelExtender
from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.otel import (
    OtelExtenderTestMixin,
    RebuildingSpanCaptureProvider,
    assert_well_formed_trace,
    inject_parent_carrier,
    make_picklable_span_capture,
    make_span_capture,
    read_span_records,
    single_span,
    single_span_attributes,
)

_TRACEPARENT_PATTERN = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")

_SPAN_NAME = "probe.calculate"
# Fixed, nonzero placeholder span id: only the derived trace_id matters for a synthesized parent.
_PARENT_SPAN_ID = 0x0000000000000001


def _parent_context(context: HookContext | None) -> Context | None:
    """Carrier wins when truthy; else a non-recording span whose trace id derives from run_id; else None."""
    if context is None:
        return None
    if context.carrier:
        return propagate.extract(context.carrier)
    if context.run_id is not None:
        span_context = SpanContext(
            trace_id=uuid.UUID(context.run_id).int,
            span_id=_PARENT_SPAN_ID,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        return set_span_in_context(NonRecordingSpan(span_context))
    return None


class TestMakeSpanCapture:
    def test_finished_span_lands_in_returned_exporter(self) -> None:
        provider, exporter = make_span_capture()
        tracer = provider.get_tracer("test-extenders-otel")

        with tracer.start_as_current_span("probe-span"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "probe-span"


class TestSingleSpan:
    def test_raises_on_zero_spans(self) -> None:
        _, exporter = make_span_capture()
        with pytest.raises(AssertionError):
            single_span(exporter)

    def test_returns_the_span_when_exactly_one_exists(self) -> None:
        provider, exporter = make_span_capture()
        tracer = provider.get_tracer("test-extenders-otel")

        with tracer.start_as_current_span("only-span"):
            pass

        span = single_span(exporter)
        assert span.name == "only-span"


class TestSingleSpanAttributes:
    def test_returns_the_attributes_mapping(self) -> None:
        provider, exporter = make_span_capture()
        tracer = provider.get_tracer("test-extenders-otel")

        with tracer.start_as_current_span("attributed-span") as span:
            span.set_attribute("probe.key", "probe-value")

        attributes = single_span_attributes(exporter)
        assert attributes["probe.key"] == "probe-value"


class TestInjectParentCarrier:
    def test_carrier_traceparent_matches_returned_trace_and_span_ids(self) -> None:
        carrier, trace_id, span_id = inject_parent_carrier()

        traceparent = carrier["traceparent"]
        assert _TRACEPARENT_PATTERN.match(traceparent)
        _, hex_trace_id, hex_span_id, _ = traceparent.split("-")
        assert hex_trace_id == format(trace_id, "032x")
        assert hex_span_id == format(span_id, "016x")


class TestMakePicklableSpanCapture:
    def test_finished_span_lands_in_returned_list(self) -> None:
        provider, captured = make_picklable_span_capture()
        tracer = provider.get_tracer("test-extenders-otel")

        with tracer.start_as_current_span("probe-span"):
            pass

        assert captured == ["probe-span"]

    def test_unpickled_provider_still_records_into_the_same_list(self) -> None:
        provider, captured = make_picklable_span_capture()
        tracer = provider.get_tracer("test-extenders-otel")
        with tracer.start_as_current_span("before-pickle-span"):
            pass

        restored = pickle.loads(pickle.dumps(provider))  # nosec
        restored_tracer = restored.get_tracer("test-extenders-otel")
        with restored_tracer.start_as_current_span("after-pickle-span"):
            pass

        assert captured == ["before-pickle-span", "after-pickle-span"]

    def test_second_call_resets_the_captured_list(self) -> None:
        provider, first_captured = make_picklable_span_capture()
        tracer = provider.get_tracer("test-extenders-otel")
        with tracer.start_as_current_span("first-call-span"):
            pass
        assert first_captured == ["first-call-span"]

        _, second_captured = make_picklable_span_capture()
        assert second_captured == []


class TestRebuildingSpanCaptureProviderBatch:
    """batch=True wires a BatchSpanProcessor whose own scheduled flush never fires within a test; only
    an explicit force_flush() drains a buffered span, proving a real worker's close() actually flushed."""

    def test_force_flush_before_any_tracer_returns_true(self, tmp_path: Path) -> None:
        provider = RebuildingSpanCaptureProvider(marker_path=tmp_path / "marker.txt", batch=True)

        assert provider.force_flush() is True

    def test_batch_span_reaches_the_marker_only_after_force_flush(self, tmp_path: Path) -> None:
        marker_path = tmp_path / "marker.txt"
        provider = RebuildingSpanCaptureProvider(marker_path=marker_path, batch=True)
        tracer = provider.get_tracer("test-extenders-otel-batch")

        with tracer.start_as_current_span("buffered-span"):
            pass

        assert not marker_path.exists()

        assert provider.force_flush() is True

        assert marker_path.read_text().splitlines() == ["buffered-span"]

    def test_batch_survives_a_pickle_round_trip(self, tmp_path: Path) -> None:
        marker_path = tmp_path / "marker.txt"
        provider = RebuildingSpanCaptureProvider(marker_path=marker_path, batch=True)

        copy = pickle.loads(pickle.dumps(provider))  # nosec

        assert copy._batch is True
        assert copy._marker_path == marker_path


class TestRecordModeSpanCapture:
    """records=True: FileSpanExporter/RebuildingSpanCaptureProvider write one JSON record per finished span
    instead of a bare name; read_span_records parses those lines back."""

    def test_parent_and_child_span_records_round_trip_with_attributes(self, tmp_path: Path) -> None:
        marker_path = tmp_path / "spans.jsonl"
        provider = RebuildingSpanCaptureProvider(marker_path=marker_path, records=True)
        tracer = provider.get_tracer("test-extenders-otel-records")

        with tracer.start_as_current_span("parent-span") as parent_span:
            parent_span.set_attribute("probe.key", "probe-value")
            with tracer.start_as_current_span("child-span"):
                pass

        records = read_span_records(marker_path)
        assert {record["name"] for record in records} == {"parent-span", "child-span"}

        parent_record = next(record for record in records if record["name"] == "parent-span")
        child_record = next(record for record in records if record["name"] == "child-span")

        assert parent_record["trace_id"] == child_record["trace_id"]
        assert parent_record["parent_span_id"] is None
        assert child_record["parent_span_id"] == parent_record["span_id"]
        assert parent_record["attributes"]["probe.key"] == "probe-value"

    def test_records_survives_a_pickle_round_trip(self, tmp_path: Path) -> None:
        marker_path = tmp_path / "spans.jsonl"
        provider = RebuildingSpanCaptureProvider(marker_path=marker_path, records=True)

        copy = pickle.loads(pickle.dumps(provider))  # nosec
        tracer = copy.get_tracer("test-extenders-otel-records")
        with tracer.start_as_current_span("after-pickle-span"):
            pass

        records = read_span_records(marker_path)
        assert [record["name"] for record in records] == ["after-pickle-span"]


class _RealOtelExtenderHost(OtelExtenderTestMixin):
    """Wires the mixin to the real OtelExtender; reused by mixin self-tests below."""

    @classmethod
    def extender_class(cls) -> type[OtelExtender]:
        return OtelExtender

    def make_otel_extender(
        self, tracer_provider: TracerProvider, *, raise_on_error: bool | None = None
    ) -> OtelExtender:
        if raise_on_error is None:
            return OtelExtender(tracer_provider=tracer_provider)
        return OtelExtender(tracer_provider=tracer_provider, raise_on_error=raise_on_error)


class TestOtelExtenderTestMixinShape:
    def test_is_extender_contract_subclass(self) -> None:
        assert issubclass(OtelExtenderTestMixin, ExtenderContractTestMixin)

    def test_raise_on_error_default_is_false(self) -> None:
        assert OtelExtenderTestMixin.raise_on_error_default() is False

    def test_expected_span_names_defaults_to_none(self) -> None:
        assert OtelExtenderTestMixin.expected_span_names() is None

    def test_ambient_sink_captured_reads_finished_spans_from_the_spy_exporters(self) -> None:
        provider, exporter = make_span_capture()
        with provider.get_tracer("test-extenders-otel").start_as_current_span("probe-span"):
            pass

        mixin = OtelExtenderTestMixin()
        captured = mixin.ambient_sink_captured([exporter])
        assert captured is not None
        assert [span.name for span in captured] == ["probe-span"]
        assert mixin.ambient_sink_captured([]) == []

    def test_sdk_defaults_contract_fails_when_the_ambient_sink_captured_nothing(self) -> None:
        class _Host(_RealOtelExtenderHost):
            def ambient_sink_captured(self, spy: list[Any]) -> list[Any] | None:
                return []

        with pytest.raises(AssertionError):
            _Host().test_contract_sdk_defaults_resolves_sink()


class _CachedTracerProbeOtelExtender(Extender):
    """Minimal OTel probe that resolves its tracer once in __init__ and never calls get_tracer again."""

    def __init__(self, tracer_provider: TracerProvider | None = None, raise_on_error: bool = False) -> None:
        self.raise_on_error = raise_on_error
        self._tracer_provider = tracer_provider
        self._tracer = trace.get_tracer("mloda-testing-probe-otel-cached", tracer_provider=tracer_provider)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_tracer_provider"] = None
        state["_tracer"] = None
        return state

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        parent = _parent_context(context)
        with self._tracer.start_as_current_span(
            _SPAN_NAME, record_exception=False, context=parent, set_status_on_exception=False
        ) as span:
            try:
                return func(*args, **kwargs)
            except BaseException as exc:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", f"{type(exc).__module__}.{type(exc).__qualname__}")
                raise


class _ProbeHost(OtelExtenderTestMixin):
    @classmethod
    def extender_class(cls) -> type[Extender]:
        return _CachedTracerProbeOtelExtender

    def make_otel_extender(self, tracer_provider: TracerProvider, *, raise_on_error: bool | None = None) -> Extender:
        return _CachedTracerProbeOtelExtender(tracer_provider=tracer_provider)


class TestExpectedSpanName:
    def test_defaults_to_none_when_no_names_are_declared(self) -> None:
        assert OtelExtenderTestMixin.expected_span_name(make_hook_context()) is None

    def test_defaults_to_the_declared_name_for_the_context_hook(self) -> None:
        class _Host(_ProbeHost):
            @classmethod
            def expected_span_names(cls) -> dict[ExtenderHook, str] | None:
                return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "named"}

        assert _Host.expected_span_name(make_hook_context()) == "named"
        assert _Host.expected_span_name(make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD)) is None

    def test_span_names_per_hook_uses_the_override_not_the_static_name(self) -> None:
        class _Host(_ProbeHost):
            @classmethod
            def expected_span_names(cls) -> dict[ExtenderHook, str] | None:
                return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "static-name-never-emitted"}

            @classmethod
            def expected_span_name(cls, context: HookContext) -> str | None:
                return _SPAN_NAME

        _Host().test_otel_span_names_per_hook()

    def test_span_names_per_hook_fails_when_the_override_disagrees(self) -> None:
        class _Host(_ProbeHost):
            @classmethod
            def expected_span_names(cls) -> dict[ExtenderHook, str] | None:
                return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: _SPAN_NAME}

            @classmethod
            def expected_span_name(cls, context: HookContext) -> str | None:
                return "something else"

        with pytest.raises(AssertionError):
            _Host().test_otel_span_names_per_hook()

    def test_sink_probe_expected_content_uses_the_override(self) -> None:
        class _Host(_ProbeHost):
            @classmethod
            def expected_span_names(cls) -> dict[ExtenderHook, str] | None:
                return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "static"}

            @classmethod
            def expected_span_name(cls, context: HookContext) -> str | None:
                return "overridden"

        assert _Host().sink_probe_expected_content() == {"overridden"}

    def test_sink_probe_expected_content_context_names_the_data_creator(self) -> None:
        seen: list[str | None] = []

        class _Host(_ProbeHost):
            @classmethod
            def expected_span_names(cls) -> dict[ExtenderHook, str] | None:
                return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "static"}

            @classmethod
            def expected_span_name(cls, context: HookContext) -> str | None:
                seen.append(context.feature_group_class)
                return "x"

        _Host().sink_probe_expected_content()
        assert seen
        assert seen[0] is not None
        assert seen[0].endswith("PyArrowDataOpsTestDataCreator")


class TestAssertWellFormedTrace:
    @staticmethod
    def _spans(*edges: tuple[str, str | None]) -> list[Any]:
        """Finished spans from (name, parent name) edges; ids are derived per name."""
        provider, exporter = make_span_capture()
        tracer = provider.get_tracer("test-extenders-otel")
        started: dict[str, Any] = {}
        for name, parent in edges:
            context = set_span_in_context(started[parent]) if parent in started else None
            span = tracer.start_span(name, context=context)
            started[name] = span
        for span in started.values():
            span.end()
        return list(exporter.get_finished_spans())

    def test_accepts_one_root_with_children(self) -> None:
        spans = self._spans(("root", None), ("a", "root"), ("b", "a"))
        assert assert_well_formed_trace(spans).name == "root"

    def test_rejects_two_roots(self) -> None:
        with pytest.raises(AssertionError):
            assert_well_formed_trace(self._spans(("r1", None), ("r2", None)))

    def test_rejects_a_parent_that_is_not_among_the_spans(self) -> None:
        spans = [span for span in self._spans(("root", None), ("mid", "root"), ("leaf", "mid")) if span.name != "mid"]
        with pytest.raises(AssertionError):
            assert_well_formed_trace(spans)

    def test_accepts_a_root_whose_parent_is_the_caller_span(self) -> None:
        carrier, _, caller_span_id = inject_parent_carrier()
        provider, exporter = make_span_capture()
        tracer = provider.get_tracer("test-extenders-otel")
        with tracer.start_as_current_span("root", context=propagate.extract(carrier)):
            with tracer.start_as_current_span("child"):
                pass
        spans = exporter.get_finished_spans()
        assert assert_well_formed_trace(spans, caller_span_id=caller_span_id).name == "root"
        with pytest.raises(AssertionError):
            assert_well_formed_trace(spans)

    def test_rejects_two_trace_ids(self) -> None:
        provider, exporter = make_span_capture()
        tracer = provider.get_tracer("test-extenders-otel")
        with tracer.start_as_current_span("one"):
            pass
        with tracer.start_as_current_span("two"):
            pass
        with pytest.raises(AssertionError):
            assert_well_formed_trace(exporter.get_finished_spans())


class TestRunAllWellFormedTraceContract:
    def test_fails_for_an_extender_whose_spans_all_hang_off_a_missing_parent(self) -> None:
        with pytest.raises(AssertionError):
            _ProbeHost().test_otel_run_all_spans_share_one_trace_id()


class TestOwnFailureDefaultDetectsNoFault:
    """Proves the chained own_failure() test is no longer vacuous for a probe with an already-cached tracer."""

    def test_default_own_failure_fails_loudly_when_nothing_is_faulted(self, caplog: pytest.LogCaptureFixture) -> None:
        class _Host(OtelExtenderTestMixin):
            @classmethod
            def extender_class(cls) -> type[Extender]:
                return _CachedTracerProbeOtelExtender

            def make_otel_extender(
                self, tracer_provider: TracerProvider, *, raise_on_error: bool | None = None
            ) -> Extender:
                if raise_on_error is None:
                    return _CachedTracerProbeOtelExtender(tracer_provider=tracer_provider)
                return _CachedTracerProbeOtelExtender(tracer_provider=tracer_provider, raise_on_error=raise_on_error)

        with pytest.raises(AssertionError, match="own_failure"):
            _Host().test_contract_own_failure_does_not_stop_chained_extender(caplog)


class _RewritingIdentityProbeOtelExtender(OtelExtender):
    """An OtelExtender that rewrites the load identity through a caller-supplied callable before recording it."""

    rewrite_identity: Callable[[HookContext, tuple[Any, ...]], str | None]

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is not None and context.hook == ExtenderHook.INPUT_DATA_LOAD:
            rewritten_context = dataclasses.replace(context, data_access_identity=self.rewrite_identity(context, args))
            with rewritten_context.activate():
                return super().__call__(func, *args, **kwargs)
        return super().__call__(func, *args, **kwargs)


def _make_rewriting_identity_host(rewrite_identity: Callable[[HookContext, tuple[Any, ...]], str | None]) -> Any:
    class _Probe(_RewritingIdentityProbeOtelExtender):
        pass

    _Probe.rewrite_identity = staticmethod(rewrite_identity)

    class _Host(_RealOtelExtenderHost):
        @classmethod
        def extender_class(cls) -> type[OtelExtender]:
            return _Probe

        def make_otel_extender(
            self, tracer_provider: TracerProvider, *, raise_on_error: bool | None = None
        ) -> OtelExtender:
            if raise_on_error is None:
                return _Probe(tracer_provider=tracer_provider)
            return _Probe(tracer_provider=tracer_provider, raise_on_error=raise_on_error)

    return _Host()


class TestQueryStringIdentityContract:
    """Proves the identity contract test fails for an extender that records the raw data access in a span
    attribute, or its user information whole or with only the password masked."""

    @pytest.mark.parametrize(
        ("rewrite_identity", "match"),
        [
            pytest.param(lambda context, args: str(args[0]), "SENSITIVE_QUERY_VALUE_xyz123", id="records-raw-args0"),
            pytest.param(
                lambda context, args: str(args[0]).partition("?")[0],
                "URI user information reached a span attribute",
                id="strips-query-keeps-userinfo",
            ),
            pytest.param(
                lambda context, args: re.sub(r":[^:@/]+@", ":***@", str(args[0]).partition("?")[0]),
                "URI user information reached a span attribute",
                id="masks-password-keeps-username",
            ),
        ],
    )
    def test_non_compliant_identity_handling_is_detected(
        self, rewrite_identity: Callable[[HookContext, tuple[Any, ...]], str | None], match: str
    ) -> None:
        host = _make_rewriting_identity_host(rewrite_identity)
        with pytest.raises(AssertionError, match=match):
            host.test_otel_input_data_load_query_string_never_reaches_span_attributes()
