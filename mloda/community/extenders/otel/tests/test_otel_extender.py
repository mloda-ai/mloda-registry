"""Tests for OtelExtender.

Every direct __call__ in this file is wrapped in a manually built HookContext.activate()
scope (mirroring core's _run_hook), so the "no ambient HookContext" case is never exercised
here. Each test gets its own isolated (TracerProvider, InMemorySpanExporter) pair via the
otel_capture fixture; the global tracer provider is never touched, since tests run under
pytest-xdist alongside other test files in this same process group.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import uuid
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from mloda.core.abstract_plugins.hook_context import instrument  # no public equivalent yet
from mloda.steward import ExtenderHook
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from mloda.community.extenders.otel import OtelExtender
from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.runners import expected_value_int, run_value_int

# The one attribute key that MUST carry content preview.
_CONTENT_ATTRIBUTE = "mloda.content.preview"


@pytest.fixture
def otel_capture() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    """A fresh, isolated (provider, exporter) pair per test; never touches the global provider."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    provider.shutdown()


def _single_span_attributes(exporter: InMemorySpanExporter) -> Mapping[str, Any]:
    """The attributes of the one-and-only finished span; fails loudly if that's not the case."""
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, spans
    attributes = spans[0].attributes
    assert attributes is not None
    return attributes


def _inject_carrier_from_new_span() -> tuple[dict[str, str], int, int]:
    """Start a span in its own isolated (TracerProvider, InMemorySpanExporter) pair - simulating the
    parent process in a multiprocessing setup - inject its context into a W3C traceparent carrier
    dict, and return (carrier, parent_trace_id, parent_span_id) for assertions.

    Imports otel_multiprocessing.inject_carrier locally (not at module scope) so that, until Green
    implements that module, only the tests that call this helper fail (with ModuleNotFoundError)
    instead of breaking collection for this entire test file.
    """
    from mloda.community.extenders.otel.otel_multiprocessing import inject_carrier

    parent_exporter = InMemorySpanExporter()
    parent_provider = TracerProvider()
    parent_provider.add_span_processor(SimpleSpanProcessor(parent_exporter))
    parent_tracer = parent_provider.get_tracer("test-otel-extender-carrier-parent")

    with parent_tracer.start_as_current_span("parent-span") as parent_span:
        parent_span_context = parent_span.get_span_context()
        carrier = inject_carrier()

    return carrier, parent_span_context.trace_id, parent_span_context.span_id


class TestOtelExtenderImport:
    def test_import_from_package(self) -> None:
        from mloda.community.extenders.otel import OtelExtender

        assert OtelExtender is not None

    def test_class_is_accessible(self) -> None:
        from mloda.community.extenders.otel import OtelExtender

        assert isinstance(OtelExtender, type)


class TestOtelExtenderContract(ExtenderContractTestMixin):
    """OtelExtender must satisfy the shared Extender contract."""

    @classmethod
    def extender_class(cls) -> type[OtelExtender]:
        return OtelExtender

    @classmethod
    def raise_on_error_default(cls) -> bool:
        return False

    def make_extender(self, *, raise_on_error: bool | None = None) -> OtelExtender:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
        if raise_on_error is None:
            return OtelExtender(tracer_provider=provider)
        return OtelExtender(raise_on_error=raise_on_error, tracer_provider=provider)

    def own_failure(self) -> AbstractContextManager[Any]:
        return patch(
            "mloda.community.extenders.otel.otel_extender.trace.get_tracer",
            side_effect=RuntimeError("otel instrumentation boom"),
        )


class TestOtelExtenderModuleImports:
    """opentelemetry-sdk is a dev-only extra of this package (only opentelemetry-api is a real runtime
    dependency), so the module must not import anything under opentelemetry.sdk at the top level."""

    def test_no_top_level_opentelemetry_sdk_import(self) -> None:
        from mloda.community.extenders.otel import otel_extender

        source = Path(otel_extender.__file__).read_text()
        tree = ast.parse(source)

        sdk_imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                sdk_imports.extend(alias.name for alias in node.names if alias.name.startswith("opentelemetry.sdk"))
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.startswith("opentelemetry.sdk"):
                    sdk_imports.append(node.module)

        assert sdk_imports == [], (
            f"otel_extender.py has a top-level import of {sdk_imports} from opentelemetry.sdk, a dev-only "
            "extra (opentelemetry-sdk is declared only under this package's 'dev' extra, never as a "
            "runtime dependency); the module already has `from __future__ import annotations`, so any "
            "type reference needed purely for annotations must come from opentelemetry.trace (the "
            "API-only module) instead."
        )


class TestOtelExtenderErrorContract:
    """raise_on_error and wraps(): OtelExtender is observability-only, so it must default to warning-only."""

    def test_raise_on_error_can_be_enabled(self) -> None:
        assert OtelExtender(raise_on_error=True).raise_on_error is True

    def test_raise_on_error_explicit_false(self) -> None:
        assert OtelExtender(raise_on_error=False).raise_on_error is False

    def test_wraps_returns_all_three_hooks(self) -> None:
        expected = {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.VALIDATE_INPUT_FEATURE,
            ExtenderHook.VALIDATE_OUTPUT_FEATURE,
        }
        assert OtelExtender().wraps() == expected

    def test_wraps_is_unconditional_regardless_of_constructor_args(self) -> None:
        """opentelemetry-api is a hard dependency now: no "trace missing" degraded set exists anymore."""
        expected = {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.VALIDATE_INPUT_FEATURE,
            ExtenderHook.VALIDATE_OUTPUT_FEATURE,
        }
        assert OtelExtender(raise_on_error=True, capture_content=True).wraps() == expected


class TestOtelExtenderConstructorOptions:
    """tracer_provider injection: the seam that keeps tests off the global OTel provider."""

    def test_tracer_provider_is_stored_on_construction(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, _ = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        assert otel._tracer_provider is provider

    def test_default_tracer_provider_is_none_and_call_still_works(self) -> None:
        otel = OtelExtender()
        context = make_hook_context()

        with context.activate():
            result = otel(lambda: 42)

        assert result == 42


class TestOtelExtenderSpanNaming:
    """One span per invocation, named by hook."""

    def test_calculate_hook_span_name(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "mloda.calculate"

    def test_validate_input_hook_span_name(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.VALIDATE_INPUT_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "mloda.validate.input"

    def test_validate_output_hook_span_name(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.VALIDATE_OUTPUT_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "mloda.validate.output"


class TestOtelExtenderSpanAttributes:
    """Span attributes under the mloda.* namespace, populated from the ambient HookContext."""

    def test_operation_name_for_calculate_hook(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.operation.name"] == "calculate"

    def test_operation_name_for_validate_input_hook(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.VALIDATE_INPUT_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.operation.name"] == "validate"

    def test_operation_name_for_validate_output_hook(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.VALIDATE_OUTPUT_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.operation.name"] == "validate"

    def test_feature_group_name_and_version_attributes(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(feature_group_class="pkg.mod.MyFeatureGroup", feature_group_version="7")
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        attrs = _single_span_attributes(exporter)
        assert attrs["mloda.feature_group.name"] == "pkg.mod.MyFeatureGroup"
        assert attrs["mloda.feature_group.version"] == "7"

    def test_compute_framework_name_attribute(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(compute_framework_name="PyArrowTable")
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.compute_framework.name"] == "PyArrowTable"

    def test_rows_in_attribute(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(rows_in=42)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.rows.in"] == 42

    def test_rows_out_attribute_present_after_successful_calculate(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """rows_out is set by instrument() DURING the func call; the extender must read it AFTER, not before."""
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            result = otel(instrument(context, func))

        assert result == [1, 2, 3]
        assert _single_span_attributes(exporter)["mloda.rows.out"] == 3

    def test_rows_out_absent_for_validate_hooks_even_when_context_carries_a_value(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """Gate on hook == calculate explicitly, not merely on rows_out being non-None."""
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.VALIDATE_INPUT_FEATURE, rows_out=99)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.rows.out" not in _single_span_attributes(exporter)

    def test_feature_name_present_for_single_feature(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(feature_names=("value_int",))
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.feature.name"] == "value_int"

    def test_feature_name_absent_for_zero_features(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(feature_names=())
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.feature.name" not in _single_span_attributes(exporter)

    def test_feature_name_absent_for_multiple_features(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(feature_names=("a", "b"))
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.feature.name" not in _single_span_attributes(exporter)

    def test_plugin_version_attribute_present_when_known(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(plugin_version="1.2.3")
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.plugin.version"] == "1.2.3"

    def test_plugin_version_attribute_absent_when_unknown(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(plugin_version=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.plugin.version" not in _single_span_attributes(exporter)

    def test_run_id_attribute_absent_when_none(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        """Covers the explicit run_id=None case: core (mloda 0.11.3+) always mints a real run_id now, but
        a hand-built HookContext (as used throughout this file) can still pass None explicitly, and the
        attribute must stay absent rather than being set to a null/empty value."""
        provider, exporter = otel_capture
        context = make_hook_context(run_id=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.run.id" not in _single_span_attributes(exporter)


class TestOtelExtenderWorkerIndexAttribute:
    """mloda.subprocess.worker_index: identifies which spawned worker process emitted a span, present
    only when the context actually carries one (mloda 0.11.3+ threads worker_index through HookContext
    when a compute framework executes across multiple processes)."""

    def test_worker_index_attribute_present_when_set(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(worker_index=2)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert _single_span_attributes(exporter)["mloda.subprocess.worker_index"] == 2

    def test_worker_index_attribute_absent_when_none(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(worker_index=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.subprocess.worker_index" not in _single_span_attributes(exporter)


class TestOtelExtenderCarrierPropagation:
    """context.carrier: when present, the span OtelExtender creates must parent from it instead of
    always starting a fresh root trace, so spans emitted by a spawned worker process correlate with
    the parent process's trace. Mirrors the assertions in
    TestExtractCarrier.test_round_trip_child_span_shares_trace_id_with_new_span_id in
    test_otel_multiprocessing.py, but exercised through OtelExtender itself rather than the raw
    extract_carrier() primitive."""

    def test_span_shares_trace_id_and_parents_from_injected_carrier(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        carrier, parent_trace_id, parent_span_id = _inject_carrier_from_new_span()

        provider, exporter = otel_capture
        context = make_hook_context(carrier=carrier)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1, spans
        span = spans[0]

        assert span.context.trace_id == parent_trace_id
        assert span.parent is not None
        assert span.parent.span_id == parent_span_id

    def test_empty_dict_carrier_does_not_crash_and_falls_through_like_none(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """An empty carrier (no traceparent key) carries no parent to extract; it must be handled the
        same as carrier=None rather than raising or spuriously parenting the span to anything."""
        provider, exporter = otel_capture
        context = make_hook_context(carrier={})
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert len(exporter.get_finished_spans()) == 1


_DETERMINISTIC_RUN_ID = "018f1e4a-7c3b-7c3b-8c3b-1234567890ab"


class TestOtelExtenderDeterministicTraceId:
    """When no carrier is present but a run_id is, spans must still correlate across process
    boundaries: the trace_id is deterministically derived from run_id via the same uuid.UUID(run_id).int
    mapping as otel_multiprocessing.trace_id_from_run_id (the "Flyte-style" deterministic-trace-id
    trick), instead of each process minting an unrelated random trace_id for the same logical run."""

    def test_trace_id_derived_from_run_id_when_no_carrier(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(run_id=_DETERMINISTIC_RUN_ID, carrier=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1, spans
        assert spans[0].context.trace_id == uuid.UUID(_DETERMINISTIC_RUN_ID).int

    def test_two_calls_with_same_run_id_and_no_carrier_share_trace_id(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """Proves same-run correlation across two separate hook invocations (e.g. one calculate, one
        validate) that never explicitly exchange a carrier - exactly the case of two spans emitted by
        different processes of the same run_all() call."""
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)

        calculate_context = make_hook_context(
            hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, run_id=_DETERMINISTIC_RUN_ID, carrier=None
        )
        with calculate_context.activate():
            otel(lambda: None)

        validate_context = make_hook_context(
            hook=ExtenderHook.VALIDATE_INPUT_FEATURE, run_id=_DETERMINISTIC_RUN_ID, carrier=None
        )
        with validate_context.activate():
            otel(lambda: None)

        spans = exporter.get_finished_spans()
        assert len(spans) == 2, spans
        trace_ids = {span.context.trace_id for span in spans}
        assert len(trace_ids) == 1, trace_ids

    def test_carrier_wins_over_run_id_when_both_present(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        carrier, parent_trace_id, _parent_span_id = _inject_carrier_from_new_span()
        # Sanity check: the carrier's trace id and the run_id-derived trace id must actually differ, or
        # this test would pass by coincidence instead of proving the carrier takes priority.
        assert parent_trace_id != uuid.UUID(_DETERMINISTIC_RUN_ID).int

        provider, exporter = otel_capture
        context = make_hook_context(run_id=_DETERMINISTIC_RUN_ID, carrier=carrier)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1, spans
        assert spans[0].context.trace_id == parent_trace_id
        assert spans[0].context.trace_id != uuid.UUID(_DETERMINISTIC_RUN_ID).int

    def test_neither_carrier_nor_run_id_still_produces_exactly_one_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """Regression guard: this is the pre-existing default path used throughout the rest of this file
        (which never passes carrier/run_id) and must keep working unchanged; the resulting trace_id is
        random here, so it is deliberately not asserted."""
        provider, exporter = otel_capture
        context = make_hook_context(run_id=None, carrier=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert len(exporter.get_finished_spans()) == 1


class TestOtelExtenderFailureHandling:
    """When the WRAPPED FUNCTION raises: never swallow, always propagate, but observe first."""

    def test_call_sets_error_span_status_and_error_type_on_func_failure(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(tracer_provider=provider)

        def func() -> None:
            raise RuntimeError("inner boom")

        with context.activate():
            with pytest.raises(RuntimeError):
                otel(func)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes["error.type"] == "builtins.RuntimeError"

    def test_call_logs_warning_on_func_failure(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, _ = otel_capture
        context = make_hook_context()
        otel = OtelExtender(tracer_provider=provider)

        def func() -> None:
            raise RuntimeError("inner boom")

        with caplog.at_level(logging.WARNING):
            with context.activate():
                with pytest.raises(RuntimeError):
                    otel(func)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OtelExtender" in r.message and "inner boom" in r.message for r in warnings), warnings

    def test_call_logs_exception_type_and_span_name_for_a_message_less_exception(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
    ) -> None:
        """`raise ValueError()` carries no message, so `logger.warning("OtelExtender %s", exc)` logs the
        literal string "OtelExtender " (str(exc) == ""): no exception type, no span/hook name, nothing
        actionable. The log must name both, independent of whether exc happens to carry a message."""
        provider, _ = otel_capture
        context = make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        def func() -> None:
            raise ValueError()

        with caplog.at_level(logging.WARNING):
            with context.activate():
                with pytest.raises(ValueError):
                    otel(func)

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("ValueError" in message and "mloda.calculate" in message for message in warnings), warnings

    def test_func_exception_message_does_not_leak_into_span_events_by_default(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """OTel's start_as_current_span defaults to record_exception=True: an exception propagating out of
        the with-block gets an auto-attached 'exception' span EVENT carrying its message and full stack
        trace (exception.message, exception.stacktrace), independent of capture_content /
        MLODA_OTEL_TRACE_CONTENT. When the func's exception message embeds raw data (a common real-world
        validation-error pattern), that value must not leak onto the span while capture_content stays at
        its metadata-only default."""
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(tracer_provider=provider)  # capture_content left at its metadata-only default
        marker = "SENSITIVE_ROW_VALUE_xyz123"

        def func() -> None:
            raise ValueError(f"invalid value found: {marker}")

        with context.activate():
            with pytest.raises(ValueError):
                otel(func)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1, spans
        span = spans[0]

        for event in span.events:
            event_attrs = event.attributes or {}
            for value in event_attrs.values():
                assert marker not in str(value), (
                    f"span event {event.name!r} attribute leaked the func exception's message ({marker!r}) "
                    "even though capture_content is False; OTel's default record_exception=True must be "
                    "disabled (or the message scrubbed) so metadata-only stays metadata-only"
                )


class TestOtelExtenderContentCapture:
    """Metadata-only by default; capture_content=True or MLODA_OTEL_TRACE_CONTENT opts in, mask redacts."""

    def test_no_content_attribute_by_default(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MLODA_OTEL_TRACE_CONTENT", raising=False)
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            otel(instrument(context, func))

        attrs = _single_span_attributes(exporter)
        assert not any("content" in key for key in attrs), attrs

    def test_content_attribute_present_when_capture_content_true(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MLODA_OTEL_TRACE_CONTENT", raising=False)
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(capture_content=True, tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            otel(instrument(context, func))

        assert _CONTENT_ATTRIBUTE in _single_span_attributes(exporter)

    @pytest.mark.parametrize("value", ["true", "1"])
    def test_content_attribute_present_when_env_var_truthy(
        self,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        monkeypatch: pytest.MonkeyPatch,
        value: str,
    ) -> None:
        monkeypatch.setenv("MLODA_OTEL_TRACE_CONTENT", value)
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(tracer_provider=provider)  # capture_content constructor arg left at default False

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            otel(instrument(context, func))

        assert _CONTENT_ATTRIBUTE in _single_span_attributes(exporter)

    def test_content_attribute_absent_when_env_var_falsy(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLODA_OTEL_TRACE_CONTENT", "false")
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            otel(instrument(context, func))

        assert _CONTENT_ATTRIBUTE not in _single_span_attributes(exporter)

    def test_content_attribute_absent_on_validate_hooks_even_when_capture_enabled(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.VALIDATE_INPUT_FEATURE)
        otel = OtelExtender(capture_content=True, tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            otel(instrument(context, func))

        assert _CONTENT_ATTRIBUTE not in _single_span_attributes(exporter)

    def test_content_attribute_absent_on_func_failure(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(capture_content=True, tracer_provider=provider)

        def func() -> None:
            raise RuntimeError("inner boom")

        with context.activate():
            with pytest.raises(RuntimeError):
                otel(instrument(context, func))

        assert _CONTENT_ATTRIBUTE not in _single_span_attributes(exporter)

    def test_content_attribute_is_bounded_for_large_results(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(capture_content=True, tracer_provider=provider)
        raw = "x" * 10_000

        def func() -> str:
            return raw

        with context.activate():
            otel(instrument(context, func))

        preview = _single_span_attributes(exporter)[_CONTENT_ATTRIBUTE]
        assert len(str(preview)) < len(raw)

    def test_content_attribute_uses_mask_to_redact_raw_value(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context()
        secret = "SECRET_VALUE_12345"  # nosec
        masked = "***MASKED***"

        def mask(_value: Any) -> str:
            return masked

        otel = OtelExtender(capture_content=True, mask=mask, tracer_provider=provider)

        def func() -> str:
            return secret

        with context.activate():
            otel(instrument(context, func))

        attrs = _single_span_attributes(exporter)
        for value in attrs.values():
            assert secret not in str(value), attrs
        assert masked in str(attrs[_CONTENT_ATTRIBUTE])


class TestOtelExtenderPostCallInstrumentationFailure:
    """A bug in the extender's OWN post-call code (reading context.rows_out, then mask/str on a result
    that func already returned successfully) runs outside any try/except, inside the
    `start_as_current_span` block. If it raises, OTel's context manager marks the span ERROR and records
    an exception event, exactly as it would for a genuine func failure - even though func itself
    succeeded and the computed result is fine. That makes a successful run indistinguishable, from the
    trace backend's point of view, from a real pipeline failure."""

    def test_mask_failure_after_func_success_does_not_mark_span_error(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)

        def broken_mask(_value: Any) -> Any:
            raise RuntimeError("mask boom")

        otel = OtelExtender(capture_content=True, mask=broken_mask, tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        # Called directly (not through _CompositeExtender): raise_on_error has no bearing on this path,
        # since it only governs how core's _invoke_extender reacts to a raise from ANYWHERE inside
        # __call__, not whether the extender's own post-call code corrupts the span it already built.
        # func already succeeded by the time broken_mask runs, so whatever escapes here is the
        # extender's own bug, not func's; let it propagate and inspect the span it leaves behind.
        with context.activate():
            with contextlib.suppress(Exception):
                otel(instrument(context, func))

        spans = exporter.get_finished_spans()
        assert len(spans) == 1, spans
        span = spans[0]
        assert span.status.status_code != StatusCode.ERROR, (
            "func succeeded, yet the span's own post-call attribute-setting code (mask/str on the "
            "result) raising made the span look identical to a genuine func failure; a broken mask must "
            "not be indistinguishable, from the trace backend's point of view, from func itself failing"
        )


class TestOtelExtenderContentPreviewCost:
    """_content_preview must bound the cost of previewing a large result, not materialize str(result) in
    full before slicing to 200 chars."""

    def test_content_preview_does_not_repr_every_element_of_a_large_result(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        class _CountingItem:
            calls = 0

            def __repr__(self) -> str:
                _CountingItem.calls += 1
                return "item"

        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(capture_content=True, tracer_provider=provider)

        result = [_CountingItem() for _ in range(5000)]
        _CountingItem.calls = 0

        def func() -> list[_CountingItem]:
            return result

        with context.activate():
            otel(instrument(context, func))

        assert _CountingItem.calls < 50, (
            f"_content_preview repr'd {_CountingItem.calls} of 5000 elements while computing a 200-char "
            "preview; it must bound the cost of previewing a large result instead of materializing "
            "str(result) in full before truncating"
        )


class TestOtelExtenderRunAll:
    """End-to-end wiring through mloda.user.mloda.run_all: real spans, unmodified results."""

    def test_run_all_produces_expected_spans_and_leaves_result_unchanged(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture

        values = run_value_int(OtelExtender(tracer_provider=provider))
        assert values == expected_value_int()

        spans = exporter.get_finished_spans()
        span_names = {span.name for span in spans}
        # A root DataCreator-based feature group never triggers VALIDATE_INPUT_FEATURE (no data
        # exists to validate before calculate runs), so only these two hooks are expected here.
        assert {"mloda.calculate", "mloda.validate.output"} <= span_names

        for span in spans:
            assert span.attributes is not None
            assert span.attributes.get("mloda.feature.name") == "value_int"
            assert span.attributes.get("mloda.compute_framework.name") == "PyArrowTable"

        # Core (mloda 0.11.3+) always mints a real run_id and threads it through every HookContext for
        # this run_all() call, so all spans from this one run must correlate via a shared trace_id.
        trace_ids = {span.context.trace_id for span in spans}
        assert len(trace_ids) == 1, trace_ids
