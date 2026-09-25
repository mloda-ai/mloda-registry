"""Tests for OtelExtender: contract compliance via OtelExtenderTestMixin, plus otel-specific
attribute, content-capture, mask, preview-cost, sdk-import and provider-resolution warning (use_sdk_defaults
with only the API default provider) checks not covered by the mixin.

Direct __call__ tests below wrap calls in a manually built HookContext.activate() scope; each
gets its own isolated (TracerProvider, InMemorySpanExporter) pair via the otel_capture fixture.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import pickle  # nosec
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import Mock

import pytest
from mloda.core.abstract_plugins.hook_context import instrument  # no public equivalent yet
from mloda.provider import FeatureGroup, FeatureSet
from mloda.steward import CompositeExtender, Extender, ExtenderHook
from mloda.user import ParallelizationMode
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from mloda.community.extenders.otel import OtelExtender
from mloda.community.extenders.otel import otel_extender as otel_extender_module
from mloda.testing.extenders.flush import blocking_flush_provider, call_with_join_timeout
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.otel import (
    OtelExtenderTestMixin,
    RebuildingSpanCaptureProvider,
    inject_parent_carrier,
    make_span_capture,
    read_span_records,
    single_span,
    single_span_attributes,
)
from mloda.testing.extenders.runners import CountingExtender, expected_value_int, run_csv_feature, run_value_int

# The one attribute key that MUST carry content preview.
_CONTENT_ATTRIBUTE = "mloda.content.preview"

_NO_SDK_PROVIDER_MARKER = "found no OpenTelemetry SDK tracer provider"


@pytest.fixture
def otel_capture() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    """A fresh, isolated (provider, exporter) pair per test; never touches the global provider."""
    provider, exporter = make_span_capture()
    yield provider, exporter
    provider.shutdown()


class _AmbientProvider:
    """What the patched opentelemetry.trace.get_tracer_provider returns; reassign .provider to switch mid-test."""

    def __init__(self) -> None:
        self.provider: trace.TracerProvider = trace.ProxyTracerProvider()


@pytest.fixture
def ambient_provider(monkeypatch: pytest.MonkeyPatch) -> _AmbientProvider:
    """Patch get_tracer_provider (span creation resolves through it too) so no test installs a real global provider."""
    holder = _AmbientProvider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: holder.provider)
    return holder


def _call_once(otel: OtelExtender) -> Any:
    with make_hook_context().activate():
        return otel(lambda: 42)


def _marker_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if _NO_SDK_PROVIDER_MARKER in r.getMessage()]


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
            ExtenderHook.INPUT_DATA_LOAD,
        }

    @classmethod
    def expected_span_names(cls) -> dict[ExtenderHook, str] | None:
        return {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "mloda.calculate",
            ExtenderHook.VALIDATE_INPUT_FEATURE: "mloda.validate.input",
            ExtenderHook.VALIDATE_OUTPUT_FEATURE: "mloda.validate.output",
            ExtenderHook.INPUT_DATA_LOAD: "mloda.load",
        }

    @classmethod
    def supports_real_worker_sink(cls) -> bool:
        return True

    def make_real_worker_extender_and_marker(self, tmp_path: Path) -> tuple[Extender, Path]:
        marker_path = tmp_path / "otel_real_worker_spans.txt"
        provider = RebuildingSpanCaptureProvider(marker_path=marker_path)
        extender = self.extender_class()(tracer_provider=provider)
        return extender, marker_path

    @classmethod
    def supports_real_worker_buffered_sink(cls) -> bool:
        return True

    def make_real_worker_buffered_extender_and_marker(self, tmp_path: Path) -> tuple[Extender, Path]:
        marker_path = tmp_path / "otel_real_worker_buffered_spans.txt"
        provider = RebuildingSpanCaptureProvider(marker_path=marker_path, batch=True)
        extender = self.extender_class()(tracer_provider=provider)
        return extender, marker_path


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

    def test_wraps_is_independent_of_raise_on_error_and_capture_content(self) -> None:
        assert OtelExtender(raise_on_error=True, capture_content=True).wraps() == OtelExtender().wraps()


class TestOtelExtenderPickledInertLogging:
    def test_pickled_copy_logs_its_own_inert_state(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, _ = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        otel._inert_warning.warn_once(lambda: None)

        copy = pickle.loads(pickle.dumps(otel))  # nosec

        with caplog.at_level(logging.WARNING):
            with make_hook_context().activate():
                result = copy(lambda: 42)

        assert result == 42
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        # The pickle-drop warning also says "inert"; the guidance substrings only appear in the copy's own warning.
        assert any(
            "OtelExtender" in r.message
            and "inert" in r.message.lower()
            and "use_sdk_defaults=True and configure" in r.message
            and "SDK tracer provider" in r.message
            for r in warning_records
        ), warning_records
        assert _marker_records(caplog) == [], caplog.records


class TestOtelExtenderConcurrentInertLogging:
    @pytest.mark.parametrize("use_sdk_defaults", [False, True], ids=["inert", "sdk_defaults"])
    def test_concurrent_first_calls_log_exactly_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        ambient_provider: _AmbientProvider,
        use_sdk_defaults: bool,
    ) -> None:
        # ambient_provider defaults to the API default provider, so the sdk_defaults case warns.
        otel = OtelExtender(use_sdk_defaults=use_sdk_defaults)
        original_warning = otel_extender_module.logger.warning
        warning_calls: list[str] = []

        # Slow the one-shot warning to widen the check-then-log race window.
        def slow_warning(msg: str, *args: Any, **kwargs: Any) -> None:
            warning_calls.append(msg)
            time.sleep(0.05)
            original_warning(msg, *args, **kwargs)

        monkeypatch.setattr(otel_extender_module.logger, "warning", slow_warning)

        thread_count = 32
        barrier = threading.Barrier(thread_count)

        def worker() -> None:
            barrier.wait(timeout=5)
            with make_hook_context().activate():
                otel(lambda: None)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(thread_count)]
        with caplog.at_level(logging.WARNING):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        if use_sdk_defaults:
            matching_records = [r for r in caplog.records if _NO_SDK_PROVIDER_MARKER in r.message]
        else:
            matching_records = [
                r for r in caplog.records if "OtelExtender" in r.message and "inert" in r.message.lower()
            ]
        assert len(matching_records) == 1, matching_records
        # Self-check: the slow_warning patch was actually hit.
        assert len(warning_calls) == 1, warning_calls


class TestOtelExtenderNoSdkProviderWarning:
    """Local to this extender, deliberately not in the published OtelExtenderTestMixin (binds third-party hosts)."""

    @pytest.mark.parametrize("default_provider_class", [trace.ProxyTracerProvider, trace.NoOpTracerProvider])
    def test_warns_once_per_instance_when_ambient_provider_is_the_api_default(
        self,
        ambient_provider: _AmbientProvider,
        caplog: pytest.LogCaptureFixture,
        default_provider_class: type[trace.TracerProvider],
    ) -> None:
        ambient_provider.provider = default_provider_class()
        otel = OtelExtender(use_sdk_defaults=True)

        with caplog.at_level(logging.WARNING):
            first_result = _call_once(otel)
            second_result = _call_once(otel)

            records = _marker_records(caplog)
            assert len(records) == 1, caplog.records
            assert records[0].levelno == logging.WARNING
            assert records[0].name == otel_extender_module.logger.name
            message = records[0].getMessage()
            assert "opentelemetry-sdk" in message
            assert "inert" not in message.lower()

            _call_once(OtelExtender(use_sdk_defaults=True))

        assert first_result == 42
        assert second_result == 42
        assert len(_marker_records(caplog)) == 2, caplog.records

    def test_warns_when_the_unpatched_api_default_provider_is_in_effect(self, caplog: pytest.LogCaptureFixture) -> None:
        if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
            pytest.skip("global provider already installed in this process")
        otel = OtelExtender(use_sdk_defaults=True)

        with caplog.at_level(logging.WARNING):
            _call_once(otel)

        assert len(_marker_records(caplog)) == 1, caplog.records

    @pytest.mark.parametrize(
        "install_before_construction", [True, False], ids=["before_construction", "after_construction"]
    )
    def test_real_ambient_provider_does_not_warn_and_receives_the_span(
        self,
        ambient_provider: _AmbientProvider,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        caplog: pytest.LogCaptureFixture,
        install_before_construction: bool,
    ) -> None:
        provider, exporter = otel_capture
        if install_before_construction:
            ambient_provider.provider = provider
        otel = OtelExtender(use_sdk_defaults=True)
        if not install_before_construction:
            ambient_provider.provider = provider

        with caplog.at_level(logging.WARNING):
            result = _call_once(otel)

        assert result == 42
        assert _marker_records(caplog) == []
        assert single_span(exporter).name == "mloda.calculate"

    def test_switching_to_a_real_provider_after_the_warning_emits_spans_without_warning_again(
        self,
        ambient_provider: _AmbientProvider,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(use_sdk_defaults=True)

        with caplog.at_level(logging.WARNING):
            _call_once(otel)
            assert len(_marker_records(caplog)) == 1, caplog.records

            ambient_provider.provider = provider
            _call_once(otel)

        assert len(_marker_records(caplog)) == 1, caplog.records
        assert single_span(exporter).name == "mloda.calculate"

    def test_pickled_copy_warns_again(
        self, ambient_provider: _AmbientProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        otel = OtelExtender(use_sdk_defaults=True)

        with caplog.at_level(logging.WARNING):
            _call_once(otel)
            assert len(_marker_records(caplog)) == 1, caplog.records

            copy = pickle.loads(pickle.dumps(otel))  # nosec
            _call_once(copy)

        assert len(_marker_records(caplog)) == 2, caplog.records

    def test_pickled_copy_that_dropped_its_injected_provider_warns(
        self,
        ambient_provider: _AmbientProvider,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        provider, _ = otel_capture
        # The SDK TracerProvider holds locks, so pickling drops it; the copy then resolves the ambient provider.
        otel = OtelExtender(tracer_provider=provider, use_sdk_defaults=True)

        with caplog.at_level(logging.WARNING):
            _call_once(otel)
            assert _marker_records(caplog) == [], caplog.records

            copy = pickle.loads(pickle.dumps(otel))  # nosec
            assert copy._tracer_provider is None
            _call_once(copy)

        assert len(_marker_records(caplog)) == 1, caplog.records

    def test_inert_instance_does_not_log_the_marker(
        self, ambient_provider: _AmbientProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        otel = OtelExtender()

        with caplog.at_level(logging.WARNING):
            result = _call_once(otel)

        assert result == 42
        assert any("inert" in r.getMessage().lower() for r in caplog.records), caplog.records
        assert _marker_records(caplog) == []


class TestOtelExtenderNonRecordingContentCapture:
    @pytest.mark.parametrize(
        ("use_sdk_defaults", "inject_provider", "carrier"),
        [
            (False, False, None),
            (True, False, None),
            # A remote parent with the sampled flag unset (-00): the default ParentBased sampler drops the span.
            (False, True, {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00"}),
        ],
        ids=["inert", "ambient_api_default", "unsampled_parent"],
    )
    def test_non_recording_span_never_calls_mask(
        self,
        ambient_provider: _AmbientProvider,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        use_sdk_defaults: bool,
        inject_provider: bool,
        carrier: dict[str, str] | None,
    ) -> None:
        provider, exporter = otel_capture
        calls = 0

        def mask(_value: Any) -> Any:
            nonlocal calls
            calls += 1
            return _value

        otel = OtelExtender(
            capture_content=True,
            mask=mask,
            tracer_provider=provider if inject_provider else None,
            use_sdk_defaults=use_sdk_defaults,
        )
        context = make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, carrier=carrier)

        with context.activate():
            result = otel(lambda: [1, 2, 3])

        assert result == [1, 2, 3]
        assert calls == 0
        assert exporter.get_finished_spans() == ()


class TestOtelExtenderPickling:
    def test_pickling_with_use_sdk_defaults_and_injected_unpicklable_provider_still_warns_about_tracer_provider(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, _ = otel_capture
        otel = OtelExtender(tracer_provider=provider, use_sdk_defaults=True)

        with caplog.at_level(logging.WARNING):
            pickle.dumps(otel)  # nosec

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("tracer_provider" in message for message in warnings), warnings
        # The generic "could not be pickled" sentence alone gives no clue why; the underlying
        # trial-pickle exception's type name must be present too (pickling the real SDK
        # TracerProvider's internal threading.Lock always raises TypeError).
        assert any("TypeError" in message for message in warnings), warnings


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

    @pytest.mark.parametrize(
        "hook",
        [ExtenderHook.VALIDATE_INPUT_FEATURE, ExtenderHook.VALIDATE_OUTPUT_FEATURE, ExtenderHook.INPUT_DATA_LOAD],
    )
    def test_content_attribute_absent_on_validate_and_load_hooks_even_when_capture_enabled(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], hook: ExtenderHook
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=hook)
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


class TestOtelExtenderPreCallInstrumentationFailure:
    """A bug in the extender's OWN pre-call code (_set_context_attributes) runs before the
    try/except around func; since the span is started with set_status_on_exception=False, only the
    explicit except block marks a span ERROR, so a pre-call failure must still leave the span ERROR."""

    def test_pre_call_instrumentation_failure_marks_span_error_and_propagates(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mloda.community.extenders.otel import otel_extender as otel_extender_module

        def broken_set_context_attributes(span: Any, context: Any) -> None:
            raise RuntimeError("attrs boom")

        monkeypatch.setattr(otel_extender_module, "_set_context_attributes", broken_set_context_attributes)

        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            with pytest.raises(RuntimeError, match="attrs boom"):
                otel(lambda: None)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1, spans
        assert spans[0].status.status_code == StatusCode.ERROR


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

        # Called directly (not through CompositeExtender): raise_on_error has no bearing on this path,
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


class TestOtelExtenderLoadSpanParenting:
    """INPUT_DATA_LOAD-only parent rule: an ambient active span wins over carrier/run_id fallback,
    but only when its trace id actually matches the trace the carrier/run_id would give."""

    def test_load_span_carrier_parents_when_no_active_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        from mloda.testing.extenders.otel import inject_parent_carrier

        carrier, trace_id, span_id = inject_parent_carrier()
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, carrier=carrier)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        span = single_span(exporter)
        assert span.context is not None
        assert span.context.trace_id == trace_id
        assert span.parent is not None
        assert span.parent.span_id == span_id

    def test_load_span_run_id_derives_trace_id_without_carrier_or_active_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        from mloda.community.extenders.otel.otel_multiprocessing import trace_id_from_run_id

        run_id = "018f1e4a-7c3b-7c3b-8c3b-1234567890ab"
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, run_id=run_id, carrier=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        span = single_span(exporter)
        assert span.context is not None
        assert span.context.trace_id == trace_id_from_run_id(run_id)

    def test_load_under_unrelated_ambient_span_with_different_trace_id_falls_back_to_run_id(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        from mloda.community.extenders.otel.otel_multiprocessing import trace_id_from_run_id

        run_id = "018f1e4a-7c3b-7c3b-8c3b-1234567890ab"
        provider, exporter = otel_capture
        ambient_tracer = provider.get_tracer("mloda-testing-ambient")
        otel = OtelExtender(tracer_provider=provider)

        with ambient_tracer.start_as_current_span("ambient"):
            with make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, run_id=run_id, carrier=None).activate():
                otel(lambda: None)

        spans = [span for span in exporter.get_finished_spans() if span.name == "mloda.load"]
        assert len(spans) == 1, spans
        assert spans[0].context is not None
        assert spans[0].context.trace_id == trace_id_from_run_id(run_id), (
            "an ambient span whose trace id does not match the run_id must not win; the load span "
            "should fall back to the run_id-derived trace id instead"
        )


class TestOtelExtenderLoadFailureHandling:
    def test_failing_load_marks_span_error_with_error_type_and_no_message(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD)
        otel = OtelExtender(tracer_provider=provider)
        marker = "SENSITIVE_LOAD_VALUE_xyz123"

        def func() -> None:
            raise RuntimeError(f"load boom: {marker}")

        with context.activate():
            with pytest.raises(RuntimeError, match="load boom"):
                otel(func)

        span = single_span(exporter)
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert "error.type" in span.attributes
        for value in span.attributes.values():
            assert marker not in str(value)
        assert marker not in (span.status.description or "")


class TestOtelExtenderCalculateSpanIgnoresAmbientSpan:
    """Characterization: calculate/validate hooks keep today's rule, ignoring an ambient active span
    whenever a carrier or run_id is present."""

    def test_calculate_span_with_run_id_ignores_ambient_active_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        from mloda.community.extenders.otel.otel_multiprocessing import trace_id_from_run_id

        run_id = "018f1e4a-7c3b-7c3b-8c3b-1234567890ab"
        provider, exporter = otel_capture
        ambient_tracer = provider.get_tracer("mloda-testing-ambient")
        otel = OtelExtender(tracer_provider=provider)

        with ambient_tracer.start_as_current_span("ambient"):
            with make_hook_context(
                hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, run_id=run_id, carrier=None
            ).activate():
                otel(lambda: None)

        spans = [span for span in exporter.get_finished_spans() if span.name == "mloda.calculate"]
        assert len(spans) == 1, spans
        assert spans[0].context is not None
        assert spans[0].context.trace_id == trace_id_from_run_id(run_id)

    def test_calculate_span_with_carrier_ignores_ambient_active_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        from mloda.testing.extenders.otel import inject_parent_carrier

        carrier, carrier_trace_id, _ = inject_parent_carrier()
        provider, exporter = otel_capture
        ambient_tracer = provider.get_tracer("mloda-testing-ambient")
        otel = OtelExtender(tracer_provider=provider)

        with ambient_tracer.start_as_current_span("ambient"):
            with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, carrier=carrier).activate():
                otel(lambda: None)

        spans = [span for span in exporter.get_finished_spans() if span.name == "mloda.calculate"]
        assert len(spans) == 1, spans
        assert spans[0].context is not None
        assert spans[0].context.trace_id == carrier_trace_id


class TestOtelExtenderLoadSpanAttributes:
    """Attributes set on the mloda.load span."""

    def test_operation_name_for_load_hook(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert single_span_attributes(exporter)["mloda.operation.name"] == "load"

    def test_load_identity_attribute_absent_for_keyword_dsn_while_format_is_unaffected(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        identity = "host=db user=u password=pw"
        context = make_hook_context(
            hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=identity, data_access_format="postgresql"
        )
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda *_: "loaded-data", identity)

        attrs = single_span_attributes(exporter)
        assert "mloda.data_access.identity" not in attrs, attrs
        assert attrs["mloda.data_access.format"] == "postgresql"

    def test_load_identity_attribute_is_sanitized(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        identity = "https://user:pw@host/path?sig=secret"
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=identity)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            # Core passes the raw data access as arg 0, matching _load_data_via_hook's call shape.
            otel(lambda *_: "loaded-data", identity)

        assert single_span_attributes(exporter)["mloda.data_access.identity"] == "https://host/path"

    def test_load_identity_attribute_absent_when_context_has_none(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.data_access.identity" not in single_span_attributes(exporter)

    def test_load_format_attribute(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_format="csv")
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert single_span_attributes(exporter)["mloda.data_access.format"] == "csv"

    def test_load_format_attribute_absent_when_none(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_format=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        assert "mloda.data_access.format" not in single_span_attributes(exporter)

    def test_load_rows_out_attribute_present_after_successful_load(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD)
        otel = OtelExtender(tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            result = otel(instrument(context, func))

        assert result == [1, 2, 3]
        assert single_span_attributes(exporter)["mloda.rows.out"] == 3


class _DeclaringFeatureGroup(FeatureGroup):
    """Declares span attributes via a declared_attributes classmethod; records the FeatureSet it receives."""

    received_feature_sets: ClassVar[list[FeatureSet | None]] = []

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return "calculated"

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, Any]:
        cls.received_feature_sets.append(features)
        return {"dataset": "orders"}


class _DeclaringFeatureGroupRaisingCall(FeatureGroup):
    """calculate_feature always raises; declared_attributes must still fire, before the call."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise RuntimeError("inner boom")

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, Any]:
        return {"dataset": "orders"}


class _RaisingDeclaration(FeatureGroup):
    """declared_attributes itself raises; must be contained, not break the call or the span."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return "ok"

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, Any]:
        raise ValueError("declaration boom")


class _NonMappingDeclaration(FeatureGroup):
    """declared_attributes returns a non-mapping; must be contained the same way as a raise."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return "ok"

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Any:
        return ["not", "a", "mapping"]


class _DeclaringReader:
    """Stand-in for a reader class: the owning class of an INPUT_DATA_LOAD call per class_attribute()."""

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return "loaded-data"

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, Any]:
        return {"table": "orders_raw"}


_DECLARED_ATTRIBUTES_CHAINS = ["direct", "otel-inner", "otel-outer"]


def _chained_call(otel: OtelExtender, chain: str) -> tuple[Extender, CountingExtender | None]:
    """direct: otel called alone. Otherwise CompositeExtender([otel, counting]), with otel inner or
    outer; the two unwrap a different number of wrapper levels before reaching the owning class."""
    if chain == "direct":
        return otel, None
    counting = CountingExtender()
    counting.priority = 50 if chain == "otel-inner" else 200
    return CompositeExtender([otel, counting]), counting


class TestOtelExtenderDeclaredAttributes:
    """mloda.declared.<key> attributes from a declared_attributes classmethod on the owning class."""

    @pytest.mark.parametrize("chain", _DECLARED_ATTRIBUTES_CHAINS)
    def test_declared_attributes_set_on_calculate_span_and_classmethod_receives_feature_set(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], chain: str
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        call, counting = _chained_call(otel, chain)
        features = FeatureSet()
        _DeclaringFeatureGroup.received_feature_sets = []

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            call(_DeclaringFeatureGroup.calculate_feature, None, features)

        assert single_span_attributes(exporter)["mloda.declared.dataset"] == "orders"
        assert _DeclaringFeatureGroup.received_feature_sets == [features]
        if counting is not None:
            assert counting.calls == 1

    @pytest.mark.parametrize("chain", _DECLARED_ATTRIBUTES_CHAINS)
    def test_declared_attributes_set_on_load_span_from_reader_classmethod(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], chain: str
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        call, counting = _chained_call(otel, chain)
        features = FeatureSet()

        with make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD).activate():
            call(_DeclaringReader.load_data, "s3://bucket/key.parquet", features)

        assert single_span_attributes(exporter)["mloda.declared.table"] == "orders_raw"
        if counting is not None:
            assert counting.calls == 1

    def test_declared_attributes_absent_on_validate_spans(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        features = FeatureSet()

        with make_hook_context(hook=ExtenderHook.VALIDATE_INPUT_FEATURE).activate():
            otel(_DeclaringFeatureGroup.calculate_feature, None, features)

        attrs = single_span_attributes(exporter)
        assert not any(key.startswith("mloda.declared.") for key in attrs), attrs

    def test_declared_attributes_set_even_when_the_wrapped_call_raises(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        features = FeatureSet()

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            with pytest.raises(RuntimeError, match="inner boom"):
                otel(_DeclaringFeatureGroupRaisingCall.calculate_feature, None, features)

        assert single_span_attributes(exporter)["mloda.declared.dataset"] == "orders"

    def test_raising_declaration_is_contained_result_returned_span_not_error(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        features = FeatureSet()

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            with caplog.at_level(logging.WARNING):
                result = otel(_RaisingDeclaration.calculate_feature, None, features)

        assert result == "ok"
        span = single_span(exporter)
        assert span.status.status_code != StatusCode.ERROR
        attrs = span.attributes or {}
        assert not any(key.startswith("mloda.declared.") for key in attrs), attrs

        extender_name = OtelExtender.__name__
        owner_name = _RaisingDeclaration.__name__
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            extender_name in message and owner_name in message and "ValueError" in message for message in warnings
        ), warnings
        assert not any("declaration boom" in message for message in warnings), warnings

    def test_non_mapping_declaration_is_contained_result_returned_span_not_error(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        features = FeatureSet()

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            with caplog.at_level(logging.WARNING):
                result = otel(_NonMappingDeclaration.calculate_feature, None, features)

        assert result == "ok"
        span = single_span(exporter)
        assert span.status.status_code != StatusCode.ERROR
        attrs = span.attributes or {}
        assert not any(key.startswith("mloda.declared.") for key in attrs), attrs

        extender_name = OtelExtender.__name__
        owner_name = _NonMappingDeclaration.__name__
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(extender_name in message and owner_name in message for message in warnings), warnings


class _RaisingIterationMapping(Mapping[str, Any]):
    """A Mapping whose iteration itself raises; declared_attributes may return one of these, so
    materializing the entries (not just calling declared_attributes) must be contained too."""

    def __getitem__(self, key: str) -> Any:
        raise KeyError(key)

    def __iter__(self) -> Any:
        raise RuntimeError("iteration boom")

    def __len__(self) -> int:
        return 1


class _RaisingIterationDeclaration(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return "ok"

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, Any]:
        return _RaisingIterationMapping()


class _InterruptingDeclaration(FeatureGroup):
    """declared_attributes raises a non-Exception BaseException (an interrupt), which must mark the
    span ERROR and propagate, like a failure in _set_context_attributes."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return "ok"

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, Any]:
        raise KeyboardInterrupt()


class _CountingDeclaration(FeatureGroup):
    calls: ClassVar[int] = 0

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return "ok"

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, Any]:
        cls.calls += 1
        return {"dataset": "orders"}


class _MixedTypeDeclaration(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return "ok"

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, Any]:
        return {
            "scalar": "kept",
            "flag": True,
            "num": 3.5,
            "listy": [1, 2, 3],
            "mapping": {"a": 1},
            "none": None,
        }


class _LongStringDeclaration(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return "ok"

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, Any]:
        return {"long": "x" * 500}


class _ManyKeysDeclaration(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return "ok"

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, Any]:
        return {f"k{i}": i for i in range(40)}


class TestOtelExtenderDeclaredAttributesContainment:
    """Declared-attribute build failures and value shaping (mloda.declared.*): a raising iteration
    or interrupt must be handled the same way as a raising or non-mapping declared_attributes call,
    and only bounded, scalar values ever reach the span."""

    def test_raising_mapping_iteration_is_contained_result_returned_span_not_error(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        features = FeatureSet()

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            with caplog.at_level(logging.WARNING):
                result = otel(_RaisingIterationDeclaration.calculate_feature, None, features)

        assert result == "ok"
        span = single_span(exporter)
        assert span.status.status_code != StatusCode.ERROR
        attrs = span.attributes or {}
        assert not any(key.startswith("mloda.declared.") for key in attrs), attrs

        extender_name = OtelExtender.__name__
        owner_name = _RaisingIterationDeclaration.__name__
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(extender_name in message and owner_name in message for message in warnings), warnings
        assert not any("iteration boom" in message for message in warnings), warnings

    def test_interrupt_from_declaration_marks_span_error_and_propagates(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        features = FeatureSet()

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            with pytest.raises(KeyboardInterrupt):
                otel(_InterruptingDeclaration.calculate_feature, None, features)

        span = single_span(exporter)
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert "error.type" in span.attributes

    def test_declaration_not_called_when_span_is_not_recording(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        features = FeatureSet()
        _CountingDeclaration.calls = 0
        # An unsampled remote parent: the default ParentBased sampler drops the span (not recording).
        carrier = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00"}

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, carrier=carrier).activate():
            result = otel(_CountingDeclaration.calculate_feature, None, features)

        assert result == "ok"
        assert exporter.get_finished_spans() == ()
        assert _CountingDeclaration.calls == 0

    def test_non_scalar_declared_values_are_dropped(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        features = FeatureSet()

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            otel(_MixedTypeDeclaration.calculate_feature, None, features)

        attrs = single_span_attributes(exporter)
        assert attrs["mloda.declared.scalar"] == "kept"
        assert attrs["mloda.declared.flag"] is True
        assert attrs["mloda.declared.num"] == 3.5
        assert "mloda.declared.listy" not in attrs
        assert "mloda.declared.mapping" not in attrs
        assert "mloda.declared.none" not in attrs

    def test_long_declared_string_is_truncated(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        features = FeatureSet()

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            otel(_LongStringDeclaration.calculate_feature, None, features)

        value = single_span_attributes(exporter)["mloda.declared.long"]
        assert len(value) == otel_extender_module._CONTENT_PREVIEW_MAX_LEN

    def test_declared_keys_are_capped_at_32(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        features = FeatureSet()

        with make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE).activate():
            otel(_ManyKeysDeclaration.calculate_feature, None, features)

        attrs = single_span_attributes(exporter)
        declared_keys = {key for key in attrs if key.startswith("mloda.declared.")}
        assert len(declared_keys) == 32, declared_keys
        assert declared_keys == {f"mloda.declared.k{i}" for i in range(32)}


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

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING])
    @pytest.mark.parametrize("parenting", ["run_id", "carrier"])
    def test_run_csv_feature_produces_a_load_span_child_of_the_calculate_span(
        self,
        tmp_path: Path,
        mode: ParallelizationMode,
        parenting: str,
        request: pytest.FixtureRequest,
    ) -> None:
        # RebuildingSpanCaptureProvider is picklable (unlike otel_capture's SDK TracerProvider), so it
        # survives into a real spawned MULTIPROCESSING worker.
        marker_path = tmp_path / "spans.jsonl"
        provider = RebuildingSpanCaptureProvider(marker_path=marker_path, records=True)
        flight_server = (
            request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
        )

        carrier: dict[str, str] | None = None
        carrier_trace_id: int | None = None
        carrier_span_id: int | None = None
        if parenting == "carrier":
            carrier, carrier_trace_id, carrier_span_id = inject_parent_carrier()

        result = run_csv_feature(
            tmp_path,
            OtelExtender(tracer_provider=provider),
            parallelization_modes={mode},
            flight_server=flight_server,
            carrier=carrier,
        )

        assert result == [1, 3]

        assert marker_path.exists(), "no span records written; the extender never emitted through the injected provider"
        records = read_span_records(marker_path)
        calculate_records = [record for record in records if record["name"] == "mloda.calculate"]
        load_records = [record for record in records if record["name"] == "mloda.load"]
        assert len(calculate_records) == 1, records
        assert len(load_records) == 1, records
        calculate_record, load_record = calculate_records[0], load_records[0]

        assert load_record["trace_id"] == calculate_record["trace_id"], records
        assert load_record["parent_span_id"] == calculate_record["span_id"], records

        if parenting == "run_id":
            run_id = calculate_record["attributes"]["mloda.run.id"]
            assert calculate_record["trace_id"] == uuid.UUID(run_id).int, records
        else:
            assert calculate_record["trace_id"] == carrier_trace_id, records
            assert calculate_record["parent_span_id"] == carrier_span_id, records

        if mode == ParallelizationMode.MULTIPROCESSING:
            assert "mloda.subprocess.worker_index" in calculate_record["attributes"], calculate_record
            assert "mloda.subprocess.worker_index" in load_record["attributes"], load_record

        load_attrs = load_record["attributes"]
        assert load_attrs.get("mloda.data_access.format") is not None
        identity = load_attrs.get("mloda.data_access.identity")
        assert isinstance(identity, str) and identity.endswith("data.csv"), load_attrs


class TestOtelExtenderClose:
    """close() flushes the resolved tracer_provider within close_timeout; never terminal, never raises,
    never calls shutdown() (core, not the extender, owns provider lifetime)."""

    def test_close_flushes_the_injected_provider_with_timeout_millis(self) -> None:
        from mloda.community.extenders.shared.teardown import CLOSE_TIMEOUT

        provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelExtender(tracer_provider=provider)

        otel.close()

        provider.force_flush.assert_called_once_with(timeout_millis=int(CLOSE_TIMEOUT * 1000))

    def test_close_flushes_the_global_provider_under_use_sdk_defaults(self, ambient_provider: _AmbientProvider) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        ambient_provider.provider = provider
        otel = OtelExtender(use_sdk_defaults=True)

        otel.close()

        provider.force_flush.assert_called_once()

    def test_inert_extender_close_touches_no_provider(self, ambient_provider: _AmbientProvider) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        ambient_provider.provider = provider
        otel = OtelExtender()  # no injected provider, use_sdk_defaults False: inert

        otel.close()

        provider.force_flush.assert_not_called()

    def test_close_never_calls_shutdown(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelExtender(tracer_provider=provider)

        otel.close()

        provider.shutdown.assert_not_called()

    def test_close_swallows_a_raising_force_flush_and_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = Mock(force_flush=Mock(side_effect=RuntimeError("flush boom")))
        otel = OtelExtender(tracer_provider=provider)

        with caplog.at_level(logging.WARNING):
            otel.close()  # must not raise

        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OtelExtender" in message for message in warnings), warnings

    def test_close_logs_a_warning_when_force_flush_returns_false(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = Mock(force_flush=Mock(return_value=False))
        otel = OtelExtender(tracer_provider=provider)

        with caplog.at_level(logging.WARNING):
            otel.close()

        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OtelExtender" in message for message in warnings), warnings

    def test_close_logs_nothing_when_provider_has_no_force_flush(self, caplog: pytest.LogCaptureFixture) -> None:
        class _NoFlushProvider:
            pass

        otel = OtelExtender(tracer_provider=_NoFlushProvider())  # type: ignore[arg-type]

        with caplog.at_level(logging.WARNING):
            otel.close()

        assert caplog.records == []

    def test_close_timeout_override_is_honored(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelExtender(tracer_provider=provider)
        otel.close_timeout = 5.0

        otel.close()

        provider.force_flush.assert_called_once_with(timeout_millis=5000)

    def test_close_bounds_a_blocking_force_flush_and_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """opentelemetry-sdk's BatchProcessor.force_flush(timeout_millis) currently ignores the timeout
        and exports synchronously; close() must still return well under a second."""
        with blocking_flush_provider() as provider:
            otel = OtelExtender(tracer_provider=provider)
            otel.close_timeout = 0.1

            start = time.monotonic()
            with caplog.at_level(logging.WARNING):
                still_running, outcome = call_with_join_timeout(otel.close, join_timeout=1.0)
            elapsed = time.monotonic() - start

        assert not still_running, "close() did not return within 1.0s while force_flush blocked past close_timeout"
        if "error" in outcome:
            raise outcome["error"]
        assert elapsed < 1.0, elapsed

        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OtelExtender" in message for message in warnings), warnings

    def test_negative_close_timeout_calls_force_flush_with_no_args(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelExtender(tracer_provider=provider)
        otel.close_timeout = -1.0

        otel.close()

        provider.force_flush.assert_called_once_with()
