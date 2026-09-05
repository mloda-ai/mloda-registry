"""Tests for OtelExtender: contract compliance via OtelExtenderTestMixin, plus otel-specific
attribute, content-capture, mask, preview-cost and sdk-import checks not covered by the mixin.

Direct __call__ tests below wrap calls in a manually built HookContext.activate() scope; each
gets its own isolated (TracerProvider, InMemorySpanExporter) pair via the otel_capture fixture.
"""

from __future__ import annotations

import ast
import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mloda.core.abstract_plugins.hook_context import instrument  # no public equivalent yet
from mloda.steward import ExtenderHook
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from mloda.community.extenders.otel import OtelExtender
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.otel import OtelExtenderTestMixin, make_span_capture, single_span_attributes
from mloda.testing.extenders.runners import expected_value_int, run_value_int

# The one attribute key that MUST carry content preview.
_CONTENT_ATTRIBUTE = "mloda.content.preview"


@pytest.fixture
def otel_capture() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    """A fresh, isolated (provider, exporter) pair per test; never touches the global provider."""
    provider, exporter = make_span_capture()
    yield provider, exporter
    provider.shutdown()


class TestOtelExtenderContract(OtelExtenderTestMixin):
    """OtelExtender must satisfy the shared Extender contract and the OTel span contract."""

    @classmethod
    def extender_class(cls) -> type[OtelExtender]:
        return OtelExtender

    def make_otel_extender(
        self, tracer_provider: TracerProvider, *, raise_on_error: bool | None = None
    ) -> OtelExtender:
        if raise_on_error is None:
            return OtelExtender(tracer_provider=tracer_provider)
        return OtelExtender(tracer_provider=tracer_provider, raise_on_error=raise_on_error)

    @classmethod
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        return {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.VALIDATE_INPUT_FEATURE,
            ExtenderHook.VALIDATE_OUTPUT_FEATURE,
        }

    @classmethod
    def expected_span_names(cls) -> dict[ExtenderHook, str] | None:
        return {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "mloda.calculate",
            ExtenderHook.VALIDATE_INPUT_FEATURE: "mloda.validate.input",
            ExtenderHook.VALIDATE_OUTPUT_FEATURE: "mloda.validate.output",
        }


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


class TestOtelExtenderSpanAttributes:
    """Span attributes under the mloda.* namespace, populated from the ambient HookContext."""

    def test_operation_name_for_calculate_hook(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert single_span_attributes(exporter)["mloda.operation.name"] == "calculate"

    def test_operation_name_for_validate_input_hook(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.VALIDATE_INPUT_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert single_span_attributes(exporter)["mloda.operation.name"] == "validate"

    def test_operation_name_for_validate_output_hook(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.VALIDATE_OUTPUT_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert single_span_attributes(exporter)["mloda.operation.name"] == "validate"

    def test_feature_group_name_and_version_attributes(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(feature_group_class="pkg.mod.MyFeatureGroup", feature_group_version="7")
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        attrs = single_span_attributes(exporter)
        assert attrs["mloda.feature_group.name"] == "pkg.mod.MyFeatureGroup"
        assert attrs["mloda.feature_group.version"] == "7"

    def test_compute_framework_name_attribute(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(compute_framework_name="PyArrowTable")
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert single_span_attributes(exporter)["mloda.compute_framework.name"] == "PyArrowTable"

    def test_rows_in_attribute(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(rows_in=42)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert single_span_attributes(exporter)["mloda.rows.in"] == 42

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
        assert single_span_attributes(exporter)["mloda.rows.out"] == 3

    def test_rows_out_absent_for_validate_hooks_even_when_context_carries_a_value(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """Gate on hook == calculate explicitly, not merely on rows_out being non-None."""
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.VALIDATE_INPUT_FEATURE, rows_out=99)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.rows.out" not in single_span_attributes(exporter)

    def test_feature_name_present_for_single_feature(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(feature_names=("value_int",))
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert single_span_attributes(exporter)["mloda.feature.name"] == "value_int"

    def test_feature_name_absent_for_zero_features(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(feature_names=())
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.feature.name" not in single_span_attributes(exporter)

    def test_feature_name_absent_for_multiple_features(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(feature_names=("a", "b"))
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.feature.name" not in single_span_attributes(exporter)

    def test_plugin_version_attribute_present_when_known(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(plugin_version="1.2.3")
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert single_span_attributes(exporter)["mloda.plugin.version"] == "1.2.3"

    def test_plugin_version_attribute_absent_when_unknown(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(plugin_version=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.plugin.version" not in single_span_attributes(exporter)

    def test_run_id_attribute_absent_when_none(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        """Covers the explicit run_id=None case: core (mloda 0.11.3+) always mints a real run_id now, but
        a hand-built HookContext (as used throughout this file) can still pass None explicitly, and the
        attribute must stay absent rather than being set to a null/empty value."""
        provider, exporter = otel_capture
        context = make_hook_context(run_id=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.run.id" not in single_span_attributes(exporter)


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

        assert single_span_attributes(exporter)["mloda.subprocess.worker_index"] == 2

    def test_worker_index_attribute_absent_when_none(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(worker_index=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.subprocess.worker_index" not in single_span_attributes(exporter)


class TestOtelExtenderFailureHandling:
    """A message-less exception must still produce an actionable warning."""

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

        attrs = single_span_attributes(exporter)
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

        assert _CONTENT_ATTRIBUTE in single_span_attributes(exporter)

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

        assert _CONTENT_ATTRIBUTE in single_span_attributes(exporter)

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

        assert _CONTENT_ATTRIBUTE not in single_span_attributes(exporter)

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

        assert _CONTENT_ATTRIBUTE not in single_span_attributes(exporter)

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

        assert _CONTENT_ATTRIBUTE not in single_span_attributes(exporter)

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

        preview = single_span_attributes(exporter)[_CONTENT_ATTRIBUTE]
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

        attrs = single_span_attributes(exporter)
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
