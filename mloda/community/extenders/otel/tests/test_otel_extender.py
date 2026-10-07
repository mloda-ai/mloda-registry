"""Tests for OtelExtender: contract compliance via OtelExtenderTestMixin, plus otel-specific
attribute, content-capture, mask, preview-cost, sdk-import and provider-resolution warning (use_sdk_defaults
with only the API default provider) checks not covered by the mixin.

Direct __call__ tests below wrap calls in a manually built HookContext.activate() scope; each
gets its own isolated (TracerProvider, InMemorySpanExporter) pair via the otel_capture fixture.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
import datetime
import inspect
import logging
import pickle  # nosec
import threading
import time
import uuid
from collections import ChainMap, OrderedDict, UserDict, defaultdict, deque, namedtuple
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import Mock

import pytest
from mloda.core.abstract_plugins.hook_context import instrument  # no public equivalent yet
from mloda.provider import BaseInputData, FeatureSet
from mloda.steward import (
    AsOfJoinConfig,
    CompositeExtender,
    Extender,
    ExtenderHook,
    FeatureResolutionError,
    HookContext,
    LifecycleOutcome,
    PlanContext,
    RunContext,
    pickle_failure_reason,
    scrub_credentials,
)
from mloda.user import ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import Histogram, InMemoryMetricReader, Sum
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from mloda.community.extenders.otel import OtelExtender
from mloda.community.extenders.otel import otel_extender as otel_extender_module
from mloda.community.extenders.shared.step_run_id import owner_name, step_run_id
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.extenders.flush import active_close_context, blocking_flush_provider, call_with_join_timeout
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.otel import (
    OtelExtenderTestMixin,
    RebuildingSpanCaptureProvider,
    assert_well_formed_trace,
    inject_parent_carrier,
    make_metric_capture,
    make_span_capture,
    read_span_records,
    single_span,
    single_span_attributes,
)
from mloda.testing.extenders.runners import (
    CountingExtender,
    MlodaTestingFailingFeatureGroup,
    MlodaTestingJoinLeft,
    MlodaTestingJoinRight,
    expected_value_int,
    failing_feature_group,
    prepare_value_int,
    run_csv_feature,
    run_failing_feature,
    run_feature,
    run_joined_features,
    run_value_int,
)

# The one attribute key that MUST carry content preview.
_CONTENT_ATTRIBUTE = "mloda.content.preview"

_NO_SDK_PROVIDER_MARKER = "found no OpenTelemetry SDK tracer provider"

_RUN_DURATION = "mloda.run.duration"
_STEP_DURATION = "mloda.step.duration"
_ROWS_IN = "mloda.step.rows.in"
_ROWS_OUT = "mloda.step.rows.out"
_STEP_ATTRIBUTES = {"mloda.operation.name", "mloda.feature_group.name", "mloda.compute_framework.name"}
_METRIC_ATTRIBUTE_ALLOWLIST = _STEP_ATTRIBUTES | {"error.type", "mloda.run.status"}
_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)


@pytest.fixture
def otel_capture() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    """A fresh, isolated (provider, exporter) pair per test; never touches the global provider."""
    provider, exporter = make_span_capture()
    yield provider, exporter
    provider.shutdown()


@pytest.fixture
def metric_capture() -> Iterator[tuple[MeterProvider, InMemoryMetricReader]]:
    """A fresh, isolated (provider, reader) pair per test; never touches the global meter provider."""
    provider, reader = make_metric_capture()
    yield provider, reader
    provider.shutdown()


class _AmbientProvider:
    """What the patched opentelemetry.trace.get_tracer_provider and opentelemetry.metrics.get_meter_provider
    return; reassign .provider / .meter_provider to switch mid-test."""

    def __init__(self) -> None:
        self.provider: trace.TracerProvider = trace.ProxyTracerProvider()
        # The real API default (the proxy), captured before get_meter_provider is patched.
        self.meter_provider: metrics.MeterProvider = metrics.get_meter_provider()
        self.meter_provider_calls = 0

    def get_meter_provider(self) -> metrics.MeterProvider:
        self.meter_provider_calls += 1
        return self.meter_provider


@pytest.fixture
def ambient_provider(monkeypatch: pytest.MonkeyPatch) -> _AmbientProvider:
    """Patch get_tracer_provider (span creation resolves through it too) and get_meter_provider so no test
    installs a real global provider."""
    holder = _AmbientProvider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: holder.provider)
    monkeypatch.setattr(metrics, "get_meter_provider", holder.get_meter_provider)
    return holder


def _call_once(otel: OtelExtender, context: HookContext | None = None) -> Any:
    with (context or make_hook_context()).activate():
        return otel(lambda: 42)


class _FloatSubclass(float):
    """Stand-in for float subclasses such as numpy.float64."""


def _join_context(**kwargs: Any) -> HookContext:
    """Mirrors core's JOIN context shape: no feature group, no feature names."""
    return make_hook_context(
        hook=ExtenderHook.JOIN, feature_group_class=None, feature_group_version=None, feature_names=(), **kwargs
    )


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
            ExtenderHook.JOIN,
        }

    @classmethod
    def expected_span_names(cls) -> dict[ExtenderHook, str] | None:
        return {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "calculate",
            ExtenderHook.VALIDATE_INPUT_FEATURE: "mloda.validate.input",
            ExtenderHook.VALIDATE_OUTPUT_FEATURE: "mloda.validate.output",
            ExtenderHook.INPUT_DATA_LOAD: "mloda.load",
            ExtenderHook.JOIN: "join",
        }

    @classmethod
    def expected_span_name(cls, context: HookContext) -> str | None:
        if context.hook == ExtenderHook.JOIN:
            return "join" if context.join_type is None else f"join {context.join_type}"
        if context.hook == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE:
            if context.feature_group_class is None:
                return "calculate"
            return f"calculate {context.feature_group_class.rsplit('.', 1)[-1]}"
        return super().expected_span_name(context)

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

    def test_meter_provider_is_stored_on_construction(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, _ = metric_capture
        otel = OtelExtender(meter_provider=provider)
        assert otel._meter_provider is provider

    def test_default_meter_provider_is_none(self) -> None:
        assert OtelExtender()._meter_provider is None

    def test_meter_provider_is_the_last_constructor_parameter(self) -> None:
        parameters = list(inspect.signature(OtelExtender.__init__).parameters)
        assert parameters[-2:] == ["trace_scope", "meter_provider"]

    def test_default_tracer_provider_is_none_and_call_still_works(self) -> None:
        otel = OtelExtender()
        context = make_hook_context()

        with context.activate():
            result = otel(lambda: 42)

        assert result == 42

    def test_wraps_is_independent_of_raise_on_error_and_capture_content(self) -> None:
        assert (
            OtelExtender(raise_on_error=True, capture_content=True, mask=lambda v: v).wraps() == OtelExtender().wraps()
        )


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
        assert single_span(exporter).name == "calculate DummyFeatureGroup"

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
        assert single_span(exporter).name == "calculate DummyFeatureGroup"

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

    def test_unpicklable_meter_provider_is_dropped_with_a_warning_naming_it(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, _ = metric_capture
        otel = OtelExtender(meter_provider=provider)

        with caplog.at_level(logging.WARNING):
            copy = pickle.loads(pickle.dumps(otel))  # nosec

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("meter_provider" in message and "TypeError" in message for message in warnings), warnings
        assert copy._meter_provider is None

    def test_tracer_only_drop_keeps_the_original_warning_sentence(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, _ = otel_capture
        reason = pickle_failure_reason(provider)
        otel = OtelExtender(tracer_provider=provider)

        with caplog.at_level(logging.WARNING):
            pickle.dumps(otel)  # nosec

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [
            f"OtelExtender drops an injected tracer_provider when pickled or copied because it isn't picklable "
            f"({reason}); the copy is inert unless use_sdk_defaults=True, which lets it resolve a provider "
            "installed in its own process, e.g. via child_bootstrap under MULTIPROCESSING."
        ]

    def test_meter_only_drop_does_not_call_the_copy_inert_when_the_tracer_survives(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, _ = metric_capture
        otel = OtelExtender(tracer_provider=trace.NoOpTracerProvider(), meter_provider=provider)

        with caplog.at_level(logging.WARNING):
            copy = pickle.loads(pickle.dumps(otel))  # nosec

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert "meter_provider" in warnings[0] and "tracer_provider" not in warnings[0], warnings
        assert "inert" not in warnings[0].lower(), warnings
        assert "metrics" in warnings[0], warnings
        assert copy._meter_provider is None
        assert isinstance(copy._tracer_provider, trace.NoOpTracerProvider)

    def test_both_unpicklable_sinks_are_dropped_with_one_warning_naming_both(
        self,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        metric_capture: tuple[MeterProvider, InMemoryMetricReader],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        tracer_provider, _ = otel_capture
        meter_provider, _ = metric_capture
        otel = OtelExtender(tracer_provider=tracer_provider, meter_provider=meter_provider)

        with caplog.at_level(logging.WARNING):
            copy = pickle.loads(pickle.dumps(otel))  # nosec

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert "tracer_provider" in warnings[0] and "meter_provider" in warnings[0], warnings
        assert copy._tracer_provider is None
        assert copy._meter_provider is None

    @pytest.mark.parametrize(
        ("key", "provider_class"),
        [("tracer_provider", trace.NoOpTracerProvider), ("meter_provider", metrics.NoOpMeterProvider)],
    )
    def test_picklable_provider_is_kept_without_a_warning(
        self, caplog: pytest.LogCaptureFixture, key: str, provider_class: type[Any]
    ) -> None:
        otel = OtelExtender(**{key: provider_class()})

        with caplog.at_level(logging.WARNING):
            copy = pickle.loads(pickle.dumps(otel))  # nosec

        assert isinstance(getattr(copy, f"_{key}"), provider_class)
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


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

    def test_feature_group_and_framework_attributes_absent_when_unresolved(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(feature_group_class=None, feature_group_version=None, compute_framework_name=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda: None)

        attributes = single_span_attributes(exporter)
        for name in ("mloda.feature_group.name", "mloda.feature_group.version", "mloda.compute_framework.name"):
            assert name not in attributes

    @pytest.mark.parametrize(
        "hook",
        [
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.VALIDATE_INPUT_FEATURE,
            ExtenderHook.VALIDATE_OUTPUT_FEATURE,
            ExtenderHook.INPUT_DATA_LOAD,
        ],
    )
    def test_step_uuid_attribute_present_when_set(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], hook: ExtenderHook
    ) -> None:
        provider, exporter = otel_capture
        step_uuid = uuid.uuid4()
        context = make_hook_context(hook=hook, step_uuid=step_uuid)

        with context.activate():
            OtelExtender(tracer_provider=provider)(lambda: None)

        assert single_span_attributes(exporter)["mloda.step.uuid"] == str(step_uuid)

    def test_step_uuid_attribute_absent_when_none(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(step_uuid=None)

        with context.activate():
            OtelExtender(tracer_provider=provider)(lambda: None)

        assert "mloda.step.uuid" not in single_span_attributes(exporter)

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
        assert any("ValueError" in message and "calculate DummyFeatureGroup" in message for message in warnings), (
            warnings
        )

    def test_warnings_name_the_subclass_not_the_base(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Both the wrapped-failure and the post-call WARNING name type(self).__name__."""
        provider, _ = otel_capture

        class MyTracer(OtelExtender):
            pass

        def broken_mask(_value: Any) -> Any:
            raise RuntimeError("mask boom")

        def failing() -> None:
            raise ValueError()

        context = make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        with caplog.at_level(logging.WARNING):
            with context.activate():
                with pytest.raises(ValueError):
                    MyTracer(tracer_provider=provider)(failing)
                with contextlib.suppress(Exception):
                    MyTracer(tracer_provider=provider, capture_content=True, mask=broken_mask)(
                        instrument(context, lambda: [1])
                    )

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("MyTracer" in m and "ValueError" in m and "calculate DummyFeatureGroup" in m for m in warnings), (
            warnings
        )
        assert any("MyTracer" in m and "post-call instrumentation failed" in m for m in warnings), warnings
        assert not any(m.startswith("OtelExtender ") for m in warnings), warnings


class _CredentialRepr:
    def __repr__(self) -> str:
        return "ConnectionHandle(host=h, password=hunter2)"  # nosec


_HeaderPair = namedtuple("_HeaderPair", "name value")
_Creds = namedtuple("_Creds", "user password")


class _PairList(list[Any]):
    pass


class _PairSet(set[Any]):
    pass


class _PairFrozenSet(frozenset[Any]):
    pass


class _PairDeque(deque[Any]):
    pass


class _RedactedTuple(tuple[Any, ...]):
    def __repr__(self) -> str:
        return "_RedactedTuple(***)"


class _CaseInsensitiveMapping(Mapping[str, Any]):
    """Mirrors requests.structures.CaseInsensitiveDict."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._store: dict[str, tuple[str, Any]] = {key.lower(): (key, value) for key, value in data.items()}

    def __getitem__(self, key: str) -> Any:
        return self._store[key.lower()][1]

    def __iter__(self) -> Iterator[str]:
        return (original for original, _ in self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return str(dict(self.items()))


class _RedactedMapping(Mapping[str, Any]):
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return "_RedactedMapping(***)"


class _ReformattingMapping(_RedactedMapping):
    """Mapping whose repr reformats values into neither repr(v) nor str(v)."""

    _encoding = "latin-1"

    def _element(self, element: Any) -> str:
        return str(element)

    def _render(self, value: Any, seen: tuple[int, ...] = ()) -> str:
        if id(value) in seen:
            return "..."
        seen = (*seen, id(value))
        if isinstance(value, Mapping):
            return ", ".join(f"{k} -> {self._render(v, seen)}" for k, v in value.items())
        if isinstance(value, bytes):
            return value.decode(self._encoding)
        if isinstance(value, str):
            return value
        if isinstance(value, (tuple, list, set)):
            return "|".join(
                self._render(element, seen) if isinstance(element, (tuple, list, set)) else self._element(element)
                for element in value
            )
        return str(value)

    def __repr__(self) -> str:
        return "Headers(" + ", ".join(f"{key} -> {self._render(value)}" for key, value in self._data.items()) + ")"


class _Utf8ReformattingMapping(_ReformattingMapping):
    _encoding = "utf-8"


class _EscapingReformattingMapping(_ReformattingMapping):
    """Reformatting mapping that renders elements with repr(element), so escapes appear."""

    def _element(self, element: Any) -> str:
        return repr(element)


class _MultiDictReformattingMapping(_ReformattingMapping):
    """Multidict: iteration repeats a key, __getitem__ returns the first value, items() yields every pair."""

    def __init__(self, pairs: list[tuple[str, Any]]) -> None:
        super().__init__({})
        self._pairs = pairs

    def __getitem__(self, key: str) -> Any:
        return next(v for k, v in self._pairs if k == key)

    def __iter__(self) -> Iterator[str]:
        return (k for k, _ in self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def items(self) -> Any:
        return list(self._pairs)

    def __repr__(self) -> str:
        return "Headers(" + ", ".join(f"{k} -> {self._render(v)}" for k, v in self._pairs) + ")"


class _StrObj:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return "_StrObj(***)"


def _self_referential_list() -> list[Any]:
    items: list[Any] = []
    items.append(items)
    items.extend(["Bearer", "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg"])  # nosec
    return items


def _shared_value_plain_then_secret_key() -> Any:
    shared = ["u", "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg"]  # nosec
    return _ReformattingMapping({"a": shared, "password": shared})


class _SelfRedactingReformattingMapping(_ReformattingMapping):
    """Reformatting mapping whose repr keeps the scheme word but redacts the secrets."""

    def __repr__(self) -> str:
        return "Headers(X-Authorization -> Bearer|***, session -> ***)"


class _RaisingRepr:
    def __repr__(self) -> str:
        raise RuntimeError("repr boom")


class _RaisingDict(dict[Any, Any]):
    def __getitem__(self, key: Any) -> Any:
        raise RuntimeError("getitem boom")


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
        otel = OtelExtender(capture_content=True, mask=lambda v: v, tracer_provider=provider)

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
        otel = OtelExtender(mask=lambda v: v, tracer_provider=provider)  # capture_content left at default None

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
        otel = OtelExtender(mask=lambda v: v, tracer_provider=provider)

        def func() -> list[int]:
            return [1, 2, 3]

        with context.activate():
            otel(instrument(context, func))

        assert _CONTENT_ATTRIBUTE not in single_span_attributes(exporter)

    def test_explicit_false_beats_truthy_env_var(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLODA_OTEL_TRACE_CONTENT", "true")
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(capture_content=False, mask=lambda v: v, tracer_provider=provider)

        with context.activate():
            otel(instrument(context, lambda: [1, 2, 3]))

        assert _CONTENT_ATTRIBUTE not in single_span_attributes(exporter)

    def test_capture_content_true_without_mask_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError, match=r"OtelExtender.*capture_content.*mask|capture_content.*mask.*OtelExtender"
        ):
            OtelExtender(capture_content=True)

    @pytest.mark.parametrize("value", ["true", "1"])
    def test_env_truthy_without_mask_records_no_preview_and_warns_once(
        self,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        value: str,
    ) -> None:
        monkeypatch.setenv("MLODA_OTEL_TRACE_CONTENT", value)
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(tracer_provider=provider)

        with caplog.at_level(logging.WARNING):
            with context.activate():
                otel(instrument(context, lambda: [1, 2, 3]))
                otel(instrument(context, lambda: [1, 2, 3]))

        for span in exporter.get_finished_spans():
            assert _CONTENT_ATTRIBUTE not in (span.attributes or {})
        mask_warnings = [
            r for r in caplog.records if r.levelno >= logging.WARNING and "mask" in r.message and "content" in r.message
        ]
        assert len(mask_warnings) == 1, caplog.text

    def test_env_truthy_with_mask_records_masked_preview(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLODA_OTEL_TRACE_CONTENT", "true")
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(mask=lambda _v: "***MASKED***", tracer_provider=provider)

        with context.activate():
            otel(instrument(context, lambda: "SECRET_VALUE_12345"))

        attrs = single_span_attributes(exporter)
        assert "***MASKED***" in str(attrs[_CONTENT_ATTRIBUTE])
        assert "SECRET_VALUE_12345" not in str(attrs)

    def test_attribute_set_true_after_construction_without_mask_records_no_preview(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MLODA_OTEL_TRACE_CONTENT", raising=False)
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(tracer_provider=provider)
        otel.capture_content = True

        with context.activate():
            otel(instrument(context, lambda: [1, 2, 3]))

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
        otel = OtelExtender(capture_content=True, mask=lambda v: v, tracer_provider=provider)

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
        otel = OtelExtender(capture_content=True, mask=lambda v: v, tracer_provider=provider)

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
        otel = OtelExtender(capture_content=True, mask=lambda v: v, tracer_provider=provider)
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

    @pytest.mark.parametrize(
        ("value", "fragments"),
        [
            pytest.param(["postgresql://u:hunter2@h/db"], ["hunter2"], id="short-dsn-in-list"),  # nosec
            pytest.param(["postgresql://user:password@host/db"], ["sword", "password"], id="dsn-cut-mid-password"),  # nosec
            pytest.param(
                ["https://bucket.s3.amazonaws.com/k.csv?X-Amz-Signature=deadbeefcafe&X-Amz-Credential=AKIAXYZ123"],
                ["deadbeefcafe", "AKIAXYZ123", "XYZ123"],
                id="presigned-url",
            ),  # nosec
            pytest.param(_CredentialRepr(), ["hunter2"], id="instance-repr"),
            pytest.param({"password": "hunter2"}, ["hunter2"], id="key-based-secret"),  # nosec
            pytest.param(
                {"very_long_prefix_name_service_password": "hunter2"},  # nosec
                ["hunter2"],
                id="long-secret-key",
            ),
            pytest.param({"password": ("a", "hunter2")}, ["hunter2"], id="tuple-under-secret-key"),  # nosec
            pytest.param("password=hunter2", ["hunter2"], id="top-level-str"),  # nosec
            pytest.param({"Authorization": ("Basic", "dXNlcjpwYXNz")}, ["dXNlcjpwYXNz"], id="authorization-tuple"),  # nosec
            pytest.param({"Authorization": ["Bearer", "dXNlcjpwYXNz"]}, ["dXNlcjpwYXNz"], id="authorization-list"),  # nosec
            pytest.param({"Authorization": b"Basic dXNlcjpwYXNz"}, ["dXNlcjpwYXNz"], id="authorization-bytes"),  # nosec
            pytest.param(
                {"Proxy-Authorization": ("Basic", "dXNlcjpwYXNz")},  # nosec
                ["dXNlcjpwYXNz"],
                id="proxy-authorization-tuple",
            ),
            pytest.param(
                {"Proxy-Authorization": ["Bearer", "dXNlcjpwYXNz"]},  # nosec
                ["dXNlcjpwYXNz"],
                id="proxy-authorization-list",
            ),
            pytest.param(
                {"Proxy-Authorization": b"Basic dXNlcjpwYXNz"},  # nosec
                ["dXNlcjpwYXNz"],
                id="proxy-authorization-bytes",
            ),
            pytest.param({"X-Authorization": ("Basic", "dXNlcjpwYXNz")}, ["dXNlcjpwYXNz"], id="x-authorization"),  # nosec
            pytest.param({"HTTP_AUTHORIZATION": ("Basic", "dXNlcjpwYXNz")}, ["dXNlcjpwYXNz"], id="http-authorization"),  # nosec
            pytest.param({"proxy_authorization": ("Basic", "dXNlcjpwYXNz")}, ["dXNlcjpwYXNz"], id="proxy-underscore"),  # nosec
            pytest.param(
                {"authorization_header": ("Basic", "dXNlcjpwYXNz")},  # nosec
                ["dXNlcjpwYXNz"],
                id="authorization-header",
            ),
            pytest.param(
                {"X-Proxy-Authorization": ("Basic", "dXNlcjpwYXNz")},  # nosec
                ["dXNlcjpwYXNz"],
                id="x-proxy-authorization",
            ),
            pytest.param(
                {"headers": [("Authorization", "Basic dXNlcjpwYXNz")]},  # nosec
                ["dXNlcjpwYXNz"],
                id="authorization-pair-in-container",
            ),
            pytest.param(
                {"headers": [("Proxy-Authorization", "Basic dXNlcjpwYXNz")]},  # nosec
                ["dXNlcjpwYXNz"],
                id="proxy-authorization-pair-in-container",
            ),
            pytest.param(("Authorization", "Bearer dXNlcjpwYXNz"), ["dXNlcjpwYXNz"], id="top-level-pair"),  # nosec
            pytest.param(
                {"headers": [(b"authorization", b"Basic dXNlcjpwYXNz")]},  # nosec
                ["dXNlcjpwYXNz"],
                id="asgi-bytes-pair",
            ),
            pytest.param([("password", "hunter2")], ["hunter2"], id="secret-keyed-pair"),  # nosec
            pytest.param({b"password": "hunter2"}, ["hunter2"], id="bytes-secret-key"),  # nosec
            pytest.param({"Authorization": "Basic dXNlcjpwYXNz"}, ["dXNlcjpwYXNz"], id="authorization-basic"),  # nosec
            pytest.param(
                {"Proxy-Authorization": "Token abc123def456"}, ["abc123def456"], id="proxy-authorization-token"
            ),  # nosec
            pytest.param("Authorization: Bearer abcdef123456", ["abcdef123456"], id="top-level-bearer-str"),  # nosec
            pytest.param(
                {"Proxy-Authorization": {"v": "Token abcdef123456"}},
                ["abcdef123456"],
                id="dict-under-proxy-authorization-key",
            ),  # nosec
            pytest.param(
                {"HTTP_AUTHORIZATION": "Basic dXNlcjpwYXNz"}, ["dXNlcjpwYXNz"], id="http-authorization-meta-key"
            ),  # nosec
            pytest.param({"X-Authorization": "Basic dXNlcjpwYXNz"}, ["dXNlcjpwYXNz"], id="x-authorization-header-key"),  # nosec
            pytest.param(
                {"headers": [_HeaderPair("Authorization", "Basic dXNlcjpwYXNz")]},  # nosec
                ["dXNlcjpwYXNz"],
                id="namedtuple-authorization-pair",
            ),
            pytest.param([_HeaderPair("password", "hunter2")], ["hunter2"], id="namedtuple-secret-keyed-pair"),  # nosec
            pytest.param([_Creds("u", "hunter2")], ["hunter2"], id="namedtuple-secret-field-name"),  # nosec
            pytest.param(
                OrderedDict(Authorization=("Basic", "dXNlcjpwYXNz")),  # nosec
                ["dXNlcjpwYXNz", "NlcjpwYXNz"],
                id="ordered-dict-authorization",
            ),
            pytest.param(
                defaultdict(list, {"X-Authorization": ["Bearer", "dXNlcjpwYXNz"]}),  # nosec
                ["dXNlcjpwYXNz", "NlcjpwYXNz"],
                id="defaultdict-x-authorization",
            ),
            pytest.param(_PairList([("password", "hunter2")]), ["hunter2"], id="list-subclass-pair"),  # nosec
            pytest.param(_PairSet([("password", "hunter2")]), ["hunter2"], id="set-subclass-pair"),  # nosec
            pytest.param(_PairFrozenSet([("password", "hunter2")]), ["hunter2"], id="frozenset-subclass-pair"),  # nosec
            pytest.param(_PairDeque([("password", "hunter2")]), ["hunter2"], id="deque-subclass-pair"),  # nosec
            pytest.param(
                _RedactedTuple(("sk_live_abcdef123456",)),  # nosec
                ["sk_live_abcdef123456"],
                id="self-redacting-tuple-subclass",
            ),
            pytest.param(
                ChainMap({"X-Authorization": ("Bearer", "dXNlcjpwYXNz")}),  # nosec
                ["dXNlcjpwYXNz", "NlcjpwYXNz"],
                id="chainmap-x-authorization",
            ),
            pytest.param(
                UserDict({"X-Authorization": ("Bearer", "dXNlcjpwYXNz")}),  # nosec
                ["dXNlcjpwYXNz", "NlcjpwYXNz"],
                id="userdict-x-authorization",
            ),
            pytest.param(
                _CaseInsensitiveMapping({"Authorization": ("Basic", "dXNlcjpwYXNz")}),  # nosec
                ["dXNlcjpwYXNz", "NlcjpwYXNz"],
                id="case-insensitive-mapping-authorization",
            ),
            pytest.param(
                _RedactedMapping({"password": "hunter2", "note": "sk_live_abcdef123456"}),  # nosec
                ["hunter2", "sk_live_abcdef123456"],
                id="self-redacting-mapping",
            ),
            pytest.param(
                _RedactedMapping({"password": "", "note": "sk_live_abcdef123456"}),  # nosec
                ["sk_live_abcdef123456"],
                id="self-redacting-mapping-empty-secret",
            ),
            pytest.param(
                UserDict({"headers": {"X-Authorization": ("Bearer", "dXNlcjpwYXNz")}}),  # nosec
                ["dXNlcjpwYXNz", "NlcjpwYXNz", "cjpwYXNz"],
                id="userdict-nested-x-authorization",
            ),
            pytest.param(
                MappingProxyType({"headers": {"X-Authorization": ("Bearer", "dXNlcjpwYXNz")}}),  # nosec
                ["dXNlcjpwYXNz", "NlcjpwYXNz", "cjpwYXNz"],
                id="mappingproxy-nested-x-authorization",
            ),
            pytest.param(
                _CaseInsensitiveMapping({"headers": {"X-Authorization": ("Bearer", "dXNlcjpwYXNz")}}),  # nosec
                ["dXNlcjpwYXNz", "NlcjpwYXNz", "cjpwYXNz"],
                id="case-insensitive-mapping-nested-authorization",
            ),
            pytest.param(
                _RedactedMapping({"password": "***", "note": "sk_live_abcdef123456"}),  # nosec
                ["sk_live_abcdef123456"],
                id="self-redacting-mapping-short-value",
            ),
            pytest.param(
                _ReformattingMapping({"X-Authorization": ("Bearer", "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg")}),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg"],
                id="reformatting-mapping-x-authorization",
            ),
            pytest.param(
                _ReformattingMapping({"password": ("u", "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg")}),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg"],
                id="reformatting-mapping-password",
            ),
            pytest.param(
                _ReformattingMapping({"password": "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg".encode()}),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg"],
                id="reformatting-mapping-bytes",
            ),
            pytest.param(
                _ReformattingMapping(
                    {
                        **{f"k{i:02}": ("a", "b") for i in range(11)},
                        "zz-Authorization": ("Bearer", "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg"),  # nosec
                    }
                ),
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg"],
                id="reformatting-mapping-secret-past-maxdict",
            ),
            pytest.param(
                _SelfRedactingReformattingMapping(
                    {
                        "X-Authorization": ("Bearer", "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg"),  # nosec
                        "session": "sk_live_abcdef123456",  # nosec
                    }
                ),
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg", "sk_live_abcdef123456"],
                id="self-redacting-reformatting-mapping",
            ),
            pytest.param(
                _MultiDictReformattingMapping(
                    [("password", ("u", "first_secret_aaaa")), ("password", ("u", "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg"))]
                ),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg", "first_secret_aaaa"],
                id="reformatting-multidict-repeated-secret-key",
            ),
            pytest.param(
                _ReformattingMapping({"password": ["a", ["Bearer", "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg"]]}),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg"],
                id="reformatting-mapping-nested-list",
            ),
            pytest.param(
                _ReformattingMapping({"password": {"Bearer", "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg"}}),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg"],
                id="reformatting-mapping-set-value",
            ),
            pytest.param(
                _ReformattingMapping({"password": ("u", _StrObj("dXNlcjpwYXNzd29yZGxvbmd0b2tlbg"))}),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg"],
                id="reformatting-mapping-object-element",
            ),
            pytest.param(
                _ReformattingMapping({"password": ("Bearer", "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", 1234567890)}),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg", "1234567890"],
                id="reformatting-mapping-int-element",
            ),
            pytest.param(
                _ReformattingMapping({"cfg": {"password": "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg"}}),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg"],
                id="reformatting-mapping-nested-secret-key",
            ),
            pytest.param(
                _ReformattingMapping({"headers": [("password", "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg")]}),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg"],
                id="reformatting-mapping-secret-pair-under-plain-key",
            ),
            pytest.param(
                _Utf8ReformattingMapping({"password": ("\u00e9" + "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg").encode("utf-8")}),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg"],
                id="reformatting-mapping-utf8-bytes",
            ),
            pytest.param(
                _EscapingReformattingMapping({"password": ("u", "'q\\x" + "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg")}),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "xvbmd0b2tlbg"],
                id="reformatting-mapping-repr-escaped-element",
            ),
            pytest.param(
                _ReformattingMapping(
                    {
                        **{f"k{i:05}": "a" for i in range(5000)},
                        "zz-Authorization": ("Bearer", "dXNlcjpwYXNzd29yZGxvbmd0b2tlbg"),
                    }
                ),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg"],
                id="reformatting-mapping-over-scan-budget",
            ),
            pytest.param(
                _ReformattingMapping({"password": _self_referential_list()}),  # nosec
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "Gxvbmd0b2tlbg"],
                id="reformatting-mapping-self-referential",
            ),
            pytest.param(
                _shared_value_plain_then_secret_key(),
                ["dXNlcjpwYXNzd29yZGxvbmd0b2tlbg", "xvbmd0b2tlbg"],
                id="reformatting-mapping-shared-value-plain-then-secret-key",
            ),
        ],
    )
    def test_content_attribute_never_contains_credentials_with_identity_mask(
        self,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        value: Any,
        fragments: list[str],
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(capture_content=True, mask=lambda v: v, tracer_provider=provider)

        with context.activate():
            otel(instrument(context, lambda: value))

        attrs = single_span_attributes(exporter)
        assert _CONTENT_ATTRIBUTE in attrs
        preview = str(attrs[_CONTENT_ATTRIBUTE])
        for fragment in fragments:
            assert fragment not in preview, preview

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param(lambda key: {key: "visible"}, id="dict"),
            pytest.param(lambda key: [(key, "visible")], id="pair"),
            pytest.param(lambda key: {"col": [key, "visible"]}, id="list-column"),
            pytest.param(lambda key: [_HeaderPair(key, "visible")], id="namedtuple-pair"),
            pytest.param(lambda key: OrderedDict({key: "visible"}), id="ordered-dict"),
            pytest.param(lambda key: ChainMap({key: "visible"}), id="chainmap"),
            pytest.param(lambda key: _CaseInsensitiveMapping({key: "visible"}), id="case-insensitive-mapping"),
            pytest.param(lambda key: _ReformattingMapping({key: ("visible", "x")}), id="reformatting-mapping"),
        ],
    )
    @pytest.mark.parametrize("key", ["host", "name", "feature", "bearer", "author", "Authorization-Info"])
    def test_content_attribute_keeps_non_secret_key_values_visible(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], key: str, shape: Callable[[str], Any]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(capture_content=True, mask=lambda v: v, tracer_provider=provider)

        with context.activate():
            otel(instrument(context, lambda: shape(key)))

        preview = str(single_span_attributes(exporter)[_CONTENT_ATTRIBUTE])
        assert "visible" in preview, preview

    @pytest.mark.parametrize(
        "result",
        [
            pytest.param(_RaisingRepr(), id="raising-repr"),
            pytest.param(_RaisingDict({"k": "v"}), id="raising-dict-subclass"),
        ],
    )
    def test_raising_result_repr_still_emits_span_and_returns_result(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], result: Any
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(capture_content=True, mask=lambda v: v, tracer_provider=provider)

        with context.activate():
            returned = otel(instrument(context, lambda: result))

        assert returned is result
        assert len(exporter.get_finished_spans()) == 1
        assert _CONTENT_ATTRIBUTE in single_span_attributes(exporter)


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
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
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
            with caplog.at_level(logging.WARNING):
                with contextlib.suppress(Exception):
                    otel(instrument(context, func))

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("post-call instrumentation failed" in m and "RuntimeError" in m for m in warnings), warnings
        assert "mask boom" not in caplog.text

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
        otel = OtelExtender(capture_content=True, mask=lambda v: v, tracer_provider=provider)

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

    @pytest.mark.parametrize("kind", ["str", "list-of-str", "repr-object"])
    def test_scrub_credentials_inputs_stay_bounded(
        self,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        monkeypatch: pytest.MonkeyPatch,
        kind: str,
    ) -> None:
        real_scrub = scrub_credentials
        lengths: list[int] = []

        def recording_scrub(text: str, *args: Any, **kwargs: Any) -> Any:
            lengths.append(len(text))
            return real_scrub(text, *args, **kwargs)

        monkeypatch.setattr("mloda.community.extenders.otel.otel_extender.scrub_credentials", recording_scrub)

        big = "a" * 2_000_000

        class _BigRepr:
            def __repr__(self) -> str:
                return big

        results: dict[str, Any] = {"str": big, "list-of-str": [big], "repr-object": _BigRepr()}
        result = results[kind]
        provider, exporter = otel_capture
        context = make_hook_context()
        otel = OtelExtender(capture_content=True, mask=lambda v: v, tracer_provider=provider)

        with context.activate():
            otel(instrument(context, lambda: result))

        assert lengths, "scrub_credentials was never called"
        assert max(lengths) < 50_000, max(lengths)


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


class TestOtelExtenderStepFailureHandling:
    @pytest.mark.parametrize("hook", [ExtenderHook.INPUT_DATA_LOAD, ExtenderHook.JOIN])
    def test_failing_step_marks_span_error_with_error_type_and_no_message(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], hook: ExtenderHook
    ) -> None:
        provider, exporter = otel_capture
        context = _join_context() if hook == ExtenderHook.JOIN else make_hook_context(hook=hook)
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
    """Characterization: step spans (calculate, join) keep today's rule, ignoring an ambient active span
    whenever a carrier or run_id is present."""

    @pytest.mark.parametrize(
        ("hook", "name"),
        [(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, "calculate DummyFeatureGroup"), (ExtenderHook.JOIN, "join")],
    )
    def test_step_span_with_run_id_ignores_ambient_active_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], hook: ExtenderHook, name: str
    ) -> None:
        from mloda.community.extenders.otel.otel_multiprocessing import trace_id_from_run_id

        run_id = "018f1e4a-7c3b-7c3b-8c3b-1234567890ab"
        provider, exporter = otel_capture
        ambient_tracer = provider.get_tracer("mloda-testing-ambient")
        otel = OtelExtender(tracer_provider=provider)
        context = (
            _join_context(run_id=run_id, carrier=None)
            if hook == ExtenderHook.JOIN
            else make_hook_context(hook=hook, run_id=run_id, carrier=None)
        )

        with ambient_tracer.start_as_current_span("ambient"):
            with context.activate():
                otel(lambda: None)

        spans = [span for span in exporter.get_finished_spans() if span.name == name]
        assert len(spans) == 1, spans
        assert spans[0].context is not None
        assert spans[0].context.trace_id == trace_id_from_run_id(run_id)

    @pytest.mark.parametrize(
        ("hook", "name"),
        [(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, "calculate DummyFeatureGroup"), (ExtenderHook.JOIN, "join")],
    )
    def test_step_span_with_carrier_ignores_ambient_active_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], hook: ExtenderHook, name: str
    ) -> None:
        from mloda.testing.extenders.otel import inject_parent_carrier

        carrier, carrier_trace_id, _ = inject_parent_carrier()
        provider, exporter = otel_capture
        ambient_tracer = provider.get_tracer("mloda-testing-ambient")
        otel = OtelExtender(tracer_provider=provider)

        context = (
            _join_context(carrier=carrier)
            if hook == ExtenderHook.JOIN
            else make_hook_context(hook=hook, carrier=carrier)
        )

        with ambient_tracer.start_as_current_span("ambient"):
            with context.activate():
                otel(lambda: None)

        spans = [span for span in exporter.get_finished_spans() if span.name == name]
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

    def test_load_identity_attribute_is_core_type_name_for_keyword_dsn_while_format_is_unaffected(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        raw = "host=db user=u password=pw"
        context_identity = BaseInputData.data_access_identity(raw)  # "str"
        context = make_hook_context(
            hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=context_identity, data_access_format="postgresql"
        )
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            # Core passes the raw data access as arg 0; the extender must record the context identity, not it.
            otel(lambda *_: "loaded-data", raw)

        attrs = single_span_attributes(exporter)
        assert attrs["mloda.data_access.identity"] == "str"
        assert "pw" not in str(attrs)
        assert attrs["mloda.data_access.format"] == "postgresql"

    def test_load_identity_attribute_is_the_core_identity(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        raw = "https://user:pw@host/path?sig=secret"
        context_identity = BaseInputData.data_access_identity(raw)  # "https://host/path"
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=context_identity)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            # Core passes the raw data access as arg 0, matching _load_data_via_hook's call shape.
            otel(lambda *_: "loaded-data", raw)

        attrs = single_span_attributes(exporter)
        assert attrs["mloda.data_access.identity"] == "https://host/path"
        assert "pw" not in str(attrs)
        assert "secret" not in str(attrs)

    def test_load_identity_attribute_for_a_mapping_data_access_is_the_sorted_key_set(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        raw = {"host": "h", "port": 5432, "api_key": "SECRET"}
        context_identity = BaseInputData.data_access_identity(raw)  # "{api_key, host, port}"
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=context_identity)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda *_: "loaded-data", raw)

        attrs = single_span_attributes(exporter)
        assert attrs["mloda.data_access.identity"] == "{api_key, host, port}"
        assert "SECRET" not in str(attrs)

    def test_load_identity_attribute_absent_when_context_has_none(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, data_access_identity=None)
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            # Context None means nothing recorded, even with a str args[0] that looks like a URI.
            otel(lambda *_: "loaded-data", "https://host/p?sig=SECRET")

        assert "mloda.data_access.identity" not in single_span_attributes(exporter)

    @pytest.mark.parametrize(
        ("identity", "is_fallback"),
        [("str", True), ("/data/x.csv", False), ("/data/x.csv", None)],
        ids=["fallback", "real", "unset"],
    )
    def test_load_identity_is_fallback_attribute_follows_the_context_flag(
        self,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        identity: str,
        is_fallback: bool | None,
    ) -> None:
        provider, exporter = otel_capture
        context = make_hook_context(
            hook=ExtenderHook.INPUT_DATA_LOAD,
            data_access_identity=identity,
            data_access_identity_is_fallback=is_fallback,
        )
        otel = OtelExtender(tracer_provider=provider)

        with context.activate():
            otel(lambda *_: "loaded-data", "raw")

        attrs = single_span_attributes(exporter)
        if is_fallback is None:
            assert "mloda.data_access.identity_is_fallback" not in attrs
        else:
            assert attrs["mloda.data_access.identity_is_fallback"] is is_fallback

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


class TestOtelExtenderJoinSpanAttributes:
    """Attributes set on the join span."""

    def _join_attributes(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], **kwargs: Any
    ) -> Mapping[str, Any]:
        provider, exporter = otel_capture
        with _join_context(**kwargs).activate():
            OtelExtender(tracer_provider=provider)(lambda: None)
        return single_span_attributes(exporter)

    def test_plain_inner_join_attributes(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        attrs = self._join_attributes(
            otel_capture, join_type="inner", join_keys=("left_id=right_id",), compute_framework_name="PyArrowTable"
        )

        assert attrs["mloda.operation.name"] == "join"
        assert attrs["mloda.join.type"] == "inner"
        assert attrs["mloda.join.keys"] == ("left_id=right_id",)
        assert attrs["mloda.compute_framework.name"] == "PyArrowTable"
        for key in attrs:
            assert not key.startswith(("mloda.join.asof.", "mloda.feature_group.")), key
        assert "mloda.feature.name" not in attrs
        assert "mloda.rows.out" not in attrs

    def test_asof_join_records_time_columns_direction_and_exact_matches(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        config = AsOfJoinConfig(
            left_time_column="lt", right_time_column="rt", direction="nearest", allow_exact_matches=False
        )

        attrs = self._join_attributes(otel_capture, join_type="asof", join_keys=("k=k",), asof_config=config)

        assert attrs["mloda.join.type"] == "asof"
        assert attrs["mloda.join.asof.left_time_column"] == "lt"
        assert attrs["mloda.join.asof.right_time_column"] == "rt"
        assert attrs["mloda.join.asof.direction"] == "nearest"
        assert attrs["mloda.join.asof.allow_exact_matches"] is False

    @pytest.mark.parametrize(
        ("tolerance", "expected_key", "expected_value"),
        [
            (datetime.timedelta(minutes=2), "mloda.join.asof.tolerance_seconds", 120.0),
            (5, "mloda.join.asof.tolerance", 5),
            (2.5, "mloda.join.asof.tolerance", 2.5),
            (_FloatSubclass(1.5), "mloda.join.asof.tolerance", 1.5),
            (None, None, None),
            (True, None, None),
        ],
        ids=["timedelta", "int", "float", "float_subclass", "absent", "bool"],
    )
    def test_asof_join_tolerance_attribute(
        self,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        tolerance: Any,
        expected_key: str | None,
        expected_value: float | None,
    ) -> None:
        config = AsOfJoinConfig(left_time_column="lt", right_time_column="rt", tolerance=tolerance)

        attrs = self._join_attributes(otel_capture, join_type="asof", asof_config=config)

        assert attrs["mloda.join.asof.left_time_column"] == "lt"
        assert attrs["mloda.join.asof.allow_exact_matches"] is True
        for key in ("mloda.join.asof.tolerance", "mloda.join.asof.tolerance_seconds"):
            if key != expected_key:
                assert key not in attrs, key
        if expected_key is not None:
            assert attrs[expected_key] == expected_value
            assert type(attrs[expected_key]) is type(expected_value)

    def test_join_feature_group_attributes_present_when_set(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        attrs = self._join_attributes(
            otel_capture, join_type="inner", join_left_feature_group="pkg.Left", join_right_feature_group="pkg.Right"
        )

        assert attrs["mloda.join.left_feature_group"] == "pkg.Left"
        assert attrs["mloda.join.right_feature_group"] == "pkg.Right"

    def test_join_feature_group_attributes_absent_when_none(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        attrs = self._join_attributes(otel_capture, join_type="inner")

        assert "mloda.join.left_feature_group" not in attrs
        assert "mloda.join.right_feature_group" not in attrs

    def test_join_keys_absent_for_a_keyless_join(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        attrs = self._join_attributes(otel_capture, join_type="append", join_keys=None)

        assert attrs["mloda.join.type"] == "append"
        assert "mloda.join.keys" not in attrs

    def test_coerce_time_columns_is_never_recorded(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        config = AsOfJoinConfig(left_time_column="lt", right_time_column="rt", coerce_time_columns=True)

        attrs = self._join_attributes(otel_capture, join_type="asof", asof_config=config)

        assert attrs["mloda.join.asof.left_time_column"] == "lt"
        assert not any("coerce" in key for key in attrs), attrs

    def test_declared_attributes_are_not_set_on_a_join_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        attrs = self._join_attributes(otel_capture, join_type="inner", declared_attributes={"dataset": "orders"})

        assert not any(key.startswith("mloda.declared.") for key in attrs), attrs


class _Target:
    """Stand-in for a feature group or reader."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return "calculated"

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return "loaded-data"

    @classmethod
    def failing_calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise RuntimeError("inner boom")


_DECLARED_ATTRIBUTES_CHAINS = ["direct", "otel-inner", "otel-outer", "no-self-wrapper"]


def _chained_call(otel: OtelExtender, chain: str) -> tuple[Extender, CountingExtender | None]:
    """direct: otel called alone. Otherwise CompositeExtender([otel, counting]), with otel inner or
    outer. no-self-wrapper: otel called alone (see _target, which hides __self__)."""
    if chain in ("direct", "no-self-wrapper"):
        return otel, None
    counting = CountingExtender()
    counting.priority = 50 if chain == "otel-inner" else 200
    return CompositeExtender([otel, counting]), counting


def _target(chain: str, func: Any) -> Any:
    """For no-self-wrapper, a plain function that calls func but does not copy __self__."""
    if chain != "no-self-wrapper":
        return func

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


def _declared_span_attrs(exporter: InMemorySpanExporter) -> dict[str, Any]:
    return {k: v for k, v in single_span_attributes(exporter).items() if k.startswith("mloda.declared.")}


class TestOtelExtenderDeclaredAttributes:
    """mloda.declared.<key> attributes from HookContext.declared_attributes."""

    @pytest.mark.parametrize("chain", _DECLARED_ATTRIBUTES_CHAINS)
    def test_declared_attributes_set_on_calculate_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], chain: str
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        call, counting = _chained_call(otel, chain)
        context = make_hook_context(
            hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, declared_attributes={"dataset": "orders"}
        )

        with context.activate():
            call(_target(chain, _Target.calculate_feature), None, FeatureSet())

        assert single_span_attributes(exporter)["mloda.declared.dataset"] == "orders"
        if counting is not None:
            assert counting.calls == 1

    @pytest.mark.parametrize("chain", _DECLARED_ATTRIBUTES_CHAINS)
    def test_declared_attributes_set_on_load_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], chain: str
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        call, counting = _chained_call(otel, chain)
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD, declared_attributes={"table": "orders_raw"})

        with context.activate():
            call(_target(chain, _Target.load_data), "s3://bucket/key.parquet", FeatureSet())

        assert single_span_attributes(exporter)["mloda.declared.table"] == "orders_raw"
        if counting is not None:
            assert counting.calls == 1

    def test_declared_attributes_absent_on_validate_spans(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        context = make_hook_context(hook=ExtenderHook.VALIDATE_INPUT_FEATURE, declared_attributes={"dataset": "orders"})

        with context.activate():
            otel(_Target.calculate_feature, None, FeatureSet())

        attrs = single_span_attributes(exporter)
        assert not any(key.startswith("mloda.declared.") for key in attrs), attrs

    def test_declared_attributes_set_even_when_the_wrapped_call_raises(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        context = make_hook_context(
            hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, declared_attributes={"dataset": "orders"}
        )

        with context.activate():
            with pytest.raises(RuntimeError, match="inner boom"):
                otel(_Target.failing_calculate_feature, None, FeatureSet())

        assert single_span_attributes(exporter)["mloda.declared.dataset"] == "orders"

    @pytest.mark.parametrize("declared", [None, {}])
    def test_absent_or_empty_declared_attributes_set_nothing(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], declared: dict[str, Any] | None
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        context = make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, declared_attributes=declared)

        with context.activate():
            otel(_Target.calculate_feature, None, FeatureSet())

        assert _declared_span_attrs(exporter) == {}

    def test_non_scalar_declared_values_are_dropped(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        declared: dict[str, Any] = {
            "scalar": "kept",
            "flag": True,
            "num": 3.5,
            "listy": [1, 2, 3],
            "mapping": {"a": 1},
            "none": None,
        }
        context = make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, declared_attributes=declared)

        with context.activate():
            otel(_Target.calculate_feature, None, FeatureSet())

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
        context = make_hook_context(
            hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, declared_attributes={"long": "x" * 500}
        )

        with context.activate():
            otel(_Target.calculate_feature, None, FeatureSet())

        value = single_span_attributes(exporter)["mloda.declared.long"]
        assert len(value) == otel_extender_module._CONTENT_PREVIEW_MAX_LEN

    def test_declared_keys_are_capped_at_32(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        context = make_hook_context(
            hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            declared_attributes={f"k{i}": i for i in range(40)},
        )

        with context.activate():
            otel(_Target.calculate_feature, None, FeatureSet())

        declared_keys = set(_declared_span_attrs(exporter))
        assert len(declared_keys) == 32, declared_keys
        assert declared_keys == {f"mloda.declared.k{i}" for i in range(32)}


def _declaring_feature_group(
    declaration: Callable[[Any, FeatureSet | None], Any],
) -> type[MlodaTestingFailingFeatureGroup]:
    """Build a fresh succeeding feature group whose declared_attributes classmethod is declaration."""

    class _Declaring(MlodaTestingFailingFeatureGroup):
        feature_name = f"declared_{uuid.uuid4().hex}"

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return {cls.feature_name: [1, 2, 3]}

        @classmethod
        def declared_attributes(cls, features: FeatureSet | None) -> Any:
            return declaration(cls, features)

    return _Declaring


def _declare_orders(cls: Any, features: FeatureSet | None) -> Mapping[str, Any]:
    return {"dataset": "orders"}


def _declare_raises(cls: Any, features: FeatureSet | None) -> Mapping[str, Any]:
    raise ValueError("declaration boom")


def _declare_non_mapping(cls: Any, features: FeatureSet | None) -> Any:
    return ["not", "a", "mapping"]


class TestOtelExtenderRunAll:
    """End-to-end wiring through mloda.user.mloda.run_all: real spans, unmodified results."""

    def test_run_all_sets_declared_attributes_on_the_calculate_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        group = _declaring_feature_group(_declare_orders)

        results = run_feature(group, OtelExtender(tracer_provider=provider))

        assert results[0].to_pydict()[group.feature_name] == [1, 2, 3]
        calculate = [span for span in exporter.get_finished_spans() if span.name.startswith("calculate ")]
        assert len(calculate) == 1
        assert (calculate[0].attributes or {}).get("mloda.declared.dataset") == "orders"

    @pytest.mark.parametrize("declaration", [_declare_raises, _declare_non_mapping])
    def test_run_all_with_a_bad_declaration_succeeds_without_declared_attributes(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], declaration: Any
    ) -> None:
        provider, exporter = otel_capture
        group = _declaring_feature_group(declaration)

        results = run_feature(group, OtelExtender(tracer_provider=provider))

        assert results[0].to_pydict()[group.feature_name] == [1, 2, 3]
        calculate = [span for span in exporter.get_finished_spans() if span.name.startswith("calculate ")]
        assert len(calculate) == 1
        assert calculate[0].status.status_code != StatusCode.ERROR
        attrs = calculate[0].attributes or {}
        assert not any(key.startswith("mloda.declared.") for key in attrs), attrs

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
        assert "mloda.validate.output" in span_names
        assert any(name.startswith("calculate ") for name in span_names), span_names

        for span in spans:
            if span.name == "mloda.run":
                continue
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
        calculate_records = [record for record in records if record["name"].startswith("calculate ")]
        root_records = [record for record in records if record["name"] == "mloda.run"]
        assert len(root_records) == 1, records
        root_record = root_records[0]
        load_records = [record for record in records if record["name"] == "mloda.load"]
        assert len(calculate_records) == 1, records
        assert len(load_records) == 1, records
        calculate_record, load_record = calculate_records[0], load_records[0]

        assert load_record["trace_id"] == calculate_record["trace_id"], records
        assert load_record["parent_span_id"] == calculate_record["span_id"], records

        # The run root span now owns the trace; the step parents to it, in the parent process or a worker.
        assert calculate_record["trace_id"] == root_record["trace_id"], records
        assert calculate_record["parent_span_id"] == root_record["span_id"], records
        assert root_record["attributes"]["mloda.run.id"] == calculate_record["attributes"]["mloda.run.id"]
        assert calculate_record["attributes"].get("mloda.step.uuid"), calculate_record
        if parenting == "run_id":
            run_id = calculate_record["attributes"]["mloda.run.id"]
            assert root_record["trace_id"] != uuid.UUID(run_id).int, records
            assert root_record["parent_span_id"] is None, records
        else:
            assert root_record["trace_id"] == carrier_trace_id, records
            assert root_record["parent_span_id"] == carrier_span_id, records

        if mode == ParallelizationMode.MULTIPROCESSING:
            assert "mloda.subprocess.worker_index" in calculate_record["attributes"], calculate_record
            assert "mloda.subprocess.worker_index" in load_record["attributes"], load_record

        load_attrs = load_record["attributes"]
        assert load_attrs.get("mloda.data_access.format") is not None
        identity = load_attrs.get("mloda.data_access.identity")
        assert isinstance(identity, str) and identity.endswith("data.csv"), load_attrs
        assert load_attrs.get("mloda.data_access.identity_is_fallback") is False, load_attrs

    def test_plan_scope_multiprocessing_worker_parents_to_the_run_root_under_the_plan_span(
        self, tmp_path: Path, flight_server: Any
    ) -> None:
        marker_path = tmp_path / "spans.jsonl"
        provider = RebuildingSpanCaptureProvider(marker_path=marker_path, records=True)

        result = run_csv_feature(
            tmp_path,
            OtelExtender(tracer_provider=provider, trace_scope="plan"),
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
        )

        assert result == [1, 3]
        records = read_span_records(marker_path)
        plan_records = [record for record in records if record["name"] == "mloda.plan"]
        root_records = [record for record in records if record["name"] == "mloda.run"]
        calculate_records = [record for record in records if record["name"].startswith("calculate ")]
        assert len(plan_records) == 1, records
        assert len(root_records) == 1, records
        assert len(calculate_records) == 1, records
        plan_record, root_record, calculate_record = plan_records[0], root_records[0], calculate_records[0]
        assert "mloda.subprocess.worker_index" in calculate_record["attributes"], calculate_record
        assert root_record["parent_span_id"] == plan_record["span_id"], records
        assert calculate_record["parent_span_id"] == root_record["span_id"], records
        assert {plan_record["trace_id"], root_record["trace_id"], calculate_record["trace_id"]} == {
            plan_record["trace_id"]
        }, records


def _plan(plan_id: str, structure_hash: str | None = None) -> PlanContext:
    return PlanContext(
        structure_hash=structure_hash,
        plan_id=plan_id,
        tenant_id=None,
        project_id=None,
        principal=None,
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )


def _run(run_id: str, plan_id: str = "plan-1", carrier: dict[str, str] | None = None) -> RunContext:
    return RunContext(run_id=run_id, plan_id=plan_id, carrier=carrier)


def _start_run(
    otel: OtelExtender,
    run_id: str,
    plan_id: str = "plan-1",
    carrier: dict[str, str] | None = None,
    structure_hash: str | None = None,
) -> None:
    otel.on_run_start(_run(run_id, plan_id, carrier), Mock(plan_id=plan_id, structure_hash=structure_hash), ())


def _complete_run(
    otel: OtelExtender,
    run_id: str | None,
    status: Any = "succeeded",
    error_type: str | None = None,
    started_at: datetime.datetime | None = None,
) -> None:
    otel.on_run_complete(
        RunContext(run_id=run_id, started_at=started_at), LifecycleOutcome(status=status, error_type=error_type)
    )


def _step(otel: OtelExtender, run_id: str, **kwargs: Any) -> None:
    carrier = kwargs.pop("carrier", None)
    with make_hook_context(run_id=run_id, carrier=carrier, **kwargs).activate():
        otel(lambda: None)


def _named(exporter: InMemorySpanExporter, name: str) -> list[Any]:
    return [span for span in exporter.get_finished_spans() if span.name == name]


def _by_trace(exporter: InMemorySpanExporter) -> dict[int, list[Any]]:
    grouped: dict[int, list[Any]] = {}
    for span in exporter.get_finished_spans():
        assert span.context is not None
        grouped.setdefault(span.context.trace_id, []).append(span)
    return grouped


def _ids(span: Any) -> tuple[int, int]:
    assert span.context is not None
    return span.context.trace_id, span.context.span_id


class TestOtelExtenderTraceScopeOption:
    @pytest.mark.parametrize("value", ["bogus", "Run", "", None, "both"])
    def test_invalid_trace_scope_raises_value_error(self, value: Any) -> None:
        with pytest.raises(ValueError, match="trace_scope"):
            OtelExtender(trace_scope=value)

    @pytest.mark.parametrize("value", ["run", "plan"])
    def test_valid_trace_scope_is_accepted(self, value: str) -> None:
        OtelExtender(trace_scope=value)  # type: ignore[arg-type]

    def test_plan_map_is_bounded_at_1024_by_default(self) -> None:
        assert otel_extender_module._MAX_PLAN_SPANS == 1024


class TestOtelExtenderRunScopeFullRuns:
    """trace_scope="run" (default): one trace and one mloda.run root per run, no plan span unless the plan fails."""

    def test_run_all_is_one_well_formed_trace_rooted_at_mloda_run(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture

        run_value_int(OtelExtender(tracer_provider=provider))

        spans = exporter.get_finished_spans()
        root = assert_well_formed_trace(spans)
        assert root.name == "mloda.run"
        assert _named(exporter, "mloda.plan") == []
        assert len(_named(exporter, "mloda.run")) == 1
        attributes = root.attributes or {}
        assert attributes.get("mloda.run.id")
        assert attributes.get("mloda.plan.id")
        assert attributes.get("mloda.run.status") == "succeeded"
        for span in spans:
            assert (span.attributes or {}).get("mloda.plan.id") == attributes["mloda.plan.id"], span.name
            if span is not root:
                assert (span.attributes or {}).get("mloda.run.id") == attributes["mloda.run.id"]
        assert root.context is not None
        assert root.context.trace_id != uuid.UUID(attributes["mloda.run.id"]).int

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING])
    def test_run_all_with_a_link_puts_the_join_span_under_the_run_root(
        self, tmp_path: Path, mode: ParallelizationMode, request: pytest.FixtureRequest
    ) -> None:
        marker_path = tmp_path / "spans.jsonl"
        provider = RebuildingSpanCaptureProvider(marker_path=marker_path, records=True)
        flight_server = (
            request.getfixturevalue("flight_server") if mode == ParallelizationMode.MULTIPROCESSING else None
        )

        result = run_joined_features(
            OtelExtender(tracer_provider=provider), parallelization_modes={mode}, flight_server=flight_server
        )

        assert sorted(result) == [110, 220]
        records = read_span_records(marker_path)
        roots = [record for record in records if record["name"] == "mloda.run"]
        joins = [record for record in records if record["name"] == "join inner"]
        assert len(roots) == 1, records
        assert len(joins) == 1, records
        root, join = roots[0], joins[0]

        assert {record["trace_id"] for record in records} == {root["trace_id"]}, records
        assert [record for record in records if record["parent_span_id"] is None] == [root], records
        span_ids = {record["span_id"] for record in records}
        assert all(record["parent_span_id"] in span_ids for record in records if record is not root), records

        assert join["parent_span_id"] == root["span_id"], records
        attributes = join["attributes"]
        assert attributes["mloda.join.type"] == "inner"
        assert attributes["mloda.join.keys"] == ["mloda_testing_left_id=mloda_testing_right_id"]
        assert (
            attributes["mloda.join.left_feature_group"]
            == f"{MlodaTestingJoinLeft.__module__}.{MlodaTestingJoinLeft.__qualname__}"
        )
        assert (
            attributes["mloda.join.right_feature_group"]
            == f"{MlodaTestingJoinRight.__module__}.{MlodaTestingJoinRight.__qualname__}"
        )
        assert attributes["mloda.run.id"] == root["attributes"]["mloda.run.id"]
        if mode == ParallelizationMode.MULTIPROCESSING:
            assert "mloda.subprocess.worker_index" in attributes, join
        else:
            assert "mloda.subprocess.worker_index" not in attributes, join

    def test_prepared_plan_run_twice_gives_one_trace_and_one_root_per_run(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        session = prepare_value_int(OtelExtender(tracer_provider=provider))

        session.run()
        session.run()

        traces = _by_trace(exporter)
        assert len(traces) == 2
        roots = [assert_well_formed_trace(spans) for spans in traces.values()]
        assert [root.name for root in roots] == ["mloda.run", "mloda.run"]
        assert _named(exporter, "mloda.plan") == []
        run_ids = {(root.attributes or {})["mloda.run.id"] for root in roots}
        plan_ids = {(root.attributes or {})["mloda.plan.id"] for root in roots}
        assert len(run_ids) == 2
        assert len(plan_ids) == 1
        hashes = {(root.attributes or {}).get("mloda.plan.structure_hash") for root in roots}
        assert len(hashes) == 1
        assert all(hashes)

    def test_failed_run_marks_the_root_error(self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]) -> None:
        provider, exporter = otel_capture
        group = failing_feature_group("mloda_testing_otel_root_fail")

        with pytest.raises(Exception):
            run_failing_feature(group, OtelExtender(tracer_provider=provider))

        root = assert_well_formed_trace(exporter.get_finished_spans())
        assert root.name == "mloda.run"
        assert root.status.status_code == StatusCode.ERROR
        assert (root.attributes or {}).get("mloda.run.status") == "failed"
        assert "error.type" in (root.attributes or {})

    def test_caller_active_span_is_the_parent_of_the_root(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        caller_tracer = provider.get_tracer("mloda-testing-caller")

        with caller_tracer.start_as_current_span("caller") as caller:
            run_value_int(OtelExtender(tracer_provider=provider))

        spans = exporter.get_finished_spans()
        caller_trace_id, caller_span_id = _ids(next(span for span in spans if span.name == "caller"))
        assert caller.get_span_context().span_id == caller_span_id
        assert assert_well_formed_trace(spans).name == "caller"
        root = _named(exporter, "mloda.run")[0]
        assert root.parent is not None
        assert root.parent.span_id == caller_span_id
        assert _ids(root)[0] == caller_trace_id

    def test_run_carrier_is_the_parent_of_the_root(
        self, tmp_path: Path, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        carrier, carrier_trace_id, carrier_span_id = inject_parent_carrier()

        run_csv_feature(tmp_path, OtelExtender(tracer_provider=provider), carrier=carrier)

        spans = exporter.get_finished_spans()
        root = assert_well_formed_trace(spans, caller_span_id=carrier_span_id)
        assert root.name == "mloda.run"
        assert _ids(root)[0] == carrier_trace_id
        assert root.parent is not None
        assert root.parent.span_id == carrier_span_id

    def test_run_carrier_wins_over_the_caller_active_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        carrier, carrier_trace_id, carrier_span_id = inject_parent_carrier()
        otel = OtelExtender(tracer_provider=provider)
        run_id = str(uuid.uuid4())

        with provider.get_tracer("mloda-testing-caller").start_as_current_span("caller"):
            _start_run(otel, run_id, carrier=carrier)
            _complete_run(otel, run_id)

        root = _named(exporter, "mloda.run")[0]
        assert _ids(root)[0] == carrier_trace_id
        assert root.parent is not None
        assert root.parent.span_id == carrier_span_id

    def test_every_run_shares_no_trace_with_another_run(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)

        run_value_int(otel)
        run_value_int(otel)

        assert len(_by_trace(exporter)) == 2

    def test_one_shared_extender_used_by_two_threads_gives_each_run_its_own_trace(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                run_value_int(otel)
            except BaseException as exc:  # surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert errors == []
        traces = _by_trace(exporter)
        assert len(traces) == 2
        for spans in traces.values():
            assert assert_well_formed_trace(spans).name == "mloda.run"
        assert len(_named(exporter, "mloda.run")) == 2

    def test_prepare_failing_on_an_unresolvable_feature_leaves_one_error_plan_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator})

        with pytest.raises(FeatureResolutionError, match="mloda_testing_otel_unresolvable"):
            mloda.prepare(
                ["mloda_testing_otel_unresolvable"],
                compute_frameworks=[PyArrowTable],
                plugin_collector=plugin_collector,
                function_extender={OtelExtender(tracer_provider=provider)},
            )

        plan = assert_well_formed_trace(exporter.get_finished_spans())
        assert plan.name == "mloda.plan"
        assert _named(exporter, "mloda.run") == []
        assert plan.status.status_code == StatusCode.ERROR
        assert (plan.attributes or {})["error.type"] == "FeatureResolutionError"


class TestOtelExtenderPlanScopeFullRuns:
    """trace_scope="plan": one mloda.plan root span per plan, every run root a child of it."""

    def test_run_all_roots_the_trace_at_the_plan_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture

        run_value_int(OtelExtender(tracer_provider=provider, trace_scope="plan"))

        spans = exporter.get_finished_spans()
        root = assert_well_formed_trace(spans)
        assert root.name == "mloda.plan"
        plan = _named(exporter, "mloda.plan")
        runs = _named(exporter, "mloda.run")
        assert len(plan) == 1
        assert len(runs) == 1
        assert runs[0].parent is not None
        assert runs[0].parent.span_id == _ids(plan[0])[1]
        assert (plan[0].attributes or {}).get("mloda.plan.id") == (runs[0].attributes or {}).get("mloda.plan.id")

    def test_prepared_plan_run_twice_shares_one_trace_with_two_run_roots(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        session = prepare_value_int(OtelExtender(tracer_provider=provider, trace_scope="plan"))

        session.run()
        session.run()

        spans = exporter.get_finished_spans()
        root = assert_well_formed_trace(spans)
        assert root.name == "mloda.plan"
        runs = _named(exporter, "mloda.run")
        assert len(runs) == 2
        for run in runs:
            assert run.parent is not None
            assert run.parent.span_id == _ids(root)[1]
        assert len({(run.attributes or {})["mloda.run.id"] for run in runs}) == 2

    def test_caller_active_span_parents_the_plan_span_and_is_a_link_on_the_run_root(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture

        with provider.get_tracer("mloda-testing-caller").start_as_current_span("caller"):
            run_value_int(OtelExtender(tracer_provider=provider, trace_scope="plan"))

        spans = exporter.get_finished_spans()
        _, caller_span_id = _ids(next(span for span in spans if span.name == "caller"))
        assert assert_well_formed_trace(spans).name == "caller"
        plan = _named(exporter, "mloda.plan")[0]
        assert plan.parent is not None
        assert plan.parent.span_id == caller_span_id
        run = _named(exporter, "mloda.run")[0]
        assert run.parent is not None
        assert run.parent.span_id == _ids(plan)[1]
        assert caller_span_id in {link.context.span_id for link in run.links}

    def test_run_carrier_is_a_link_on_the_run_root_not_its_parent(
        self, tmp_path: Path, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        carrier, _, carrier_span_id = inject_parent_carrier()

        run_csv_feature(tmp_path, OtelExtender(tracer_provider=provider, trace_scope="plan"), carrier=carrier)

        run = _named(exporter, "mloda.run")[0]
        plan = _named(exporter, "mloda.plan")[0]
        assert run.parent is not None
        assert run.parent.span_id == _ids(plan)[1]
        assert carrier_span_id in {link.context.span_id for link in run.links}


class TestOtelExtenderRootSpanHooks:
    """Lifecycle hooks driven directly (all fire in the parent process)."""

    @pytest.mark.parametrize("structure_hash", ["abc123", None])
    def test_on_run_start_and_complete_emit_the_root_with_attributes(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], structure_hash: str | None
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        run_id = str(uuid.uuid4())

        _start_run(otel, run_id, plan_id="plan-x", structure_hash=structure_hash)
        assert exporter.get_finished_spans() == ()
        _complete_run(otel, run_id)

        span = single_span(exporter)
        assert span.name == "mloda.run"
        assert span.parent is None
        assert span.attributes is not None
        assert span.attributes["mloda.run.id"] == run_id
        assert span.attributes["mloda.plan.id"] == "plan-x"
        if structure_hash is None:
            assert "mloda.plan.structure_hash" not in span.attributes
        else:
            assert span.attributes["mloda.plan.structure_hash"] == structure_hash
        assert span.attributes["mloda.run.status"] == "succeeded"
        assert span.status.status_code != StatusCode.ERROR

    @pytest.mark.parametrize("status", ["failed", "cancelled"])
    def test_non_success_status_is_recorded(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], status: str
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        run_id = str(uuid.uuid4())

        _start_run(otel, run_id)
        _complete_run(otel, run_id, status=status, error_type="ValueError")

        span = single_span(exporter)
        assert (span.attributes or {})["mloda.run.status"] == status
        if status == "failed":
            assert span.status.status_code == StatusCode.ERROR
            assert (span.attributes or {})["error.type"] == "ValueError"

    def test_complete_for_an_unknown_run_id_is_a_noop(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)

        _complete_run(otel, str(uuid.uuid4()))

        assert exporter.get_finished_spans() == ()

    def test_complete_drops_the_root_so_later_steps_fall_back_to_the_run_id(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        run_id = str(uuid.uuid4())

        _start_run(otel, run_id)
        _complete_run(otel, run_id)
        _step(otel, run_id)

        step = next(span for span in exporter.get_finished_spans() if span.name != "mloda.run")
        assert _ids(step)[0] == uuid.UUID(run_id).int

    @pytest.mark.parametrize("hook", [ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.JOIN])
    def test_step_span_is_a_child_of_the_known_root(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], hook: ExtenderHook
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        run_id = str(uuid.uuid4())

        _start_run(otel, run_id)
        if hook == ExtenderHook.JOIN:
            with _join_context(run_id=run_id, carrier=None).activate():
                otel(lambda: None)
        else:
            _step(otel, run_id, hook=hook)
        _complete_run(otel, run_id)

        spans = exporter.get_finished_spans()
        assert assert_well_formed_trace(spans).name == "mloda.run"
        step = next(span for span in spans if span.name != "mloda.run")
        root = _named(exporter, "mloda.run")[0]
        assert step.parent is not None
        assert step.parent.span_id == _ids(root)[1]
        assert _ids(step)[0] == _ids(root)[0]

    def test_known_root_wins_over_the_hook_carrier(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        run_id = str(uuid.uuid4())
        carrier, carrier_trace_id, _ = inject_parent_carrier()

        _start_run(otel, run_id)
        _step(otel, run_id, carrier=carrier)
        _complete_run(otel, run_id)

        step = next(span for span in exporter.get_finished_spans() if span.name != "mloda.run")
        root = _named(exporter, "mloda.run")[0]
        assert _ids(step)[0] == _ids(root)[0] != carrier_trace_id
        assert step.parent is not None
        assert step.parent.span_id == _ids(root)[1]

    def test_load_under_an_ambient_span_of_the_root_trace_is_its_child(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        run_id = str(uuid.uuid4())

        _start_run(otel, run_id)
        with make_hook_context(run_id=run_id).activate():
            otel(lambda: _step(otel, run_id, hook=ExtenderHook.INPUT_DATA_LOAD))
        _complete_run(otel, run_id)

        spans = exporter.get_finished_spans()
        assert assert_well_formed_trace(spans).name == "mloda.run"
        load = _named(exporter, "mloda.load")[0]
        calculate = next(span for span in spans if span.name.startswith("calculate"))
        assert load.parent is not None
        assert load.parent.span_id == _ids(calculate)[1]

    def test_load_under_an_ambient_span_of_another_trace_falls_back_to_the_root(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)
        run_id = str(uuid.uuid4())

        _start_run(otel, run_id)
        with provider.get_tracer("mloda-testing-ambient").start_as_current_span("ambient"):
            _step(otel, run_id, hook=ExtenderHook.INPUT_DATA_LOAD)
        _complete_run(otel, run_id)

        load = _named(exporter, "mloda.load")[0]
        root = _named(exporter, "mloda.run")[0]
        assert _ids(load)[0] == _ids(root)[0]
        assert load.parent is not None
        assert load.parent.span_id == _ids(root)[1]

    @pytest.mark.parametrize(
        "make_otel",
        [OtelExtender, lambda: OtelExtender(tracer_provider=trace.NoOpTracerProvider())],
        ids=["inert", "no_op_provider"],
    )
    def test_extender_without_a_recording_provider_stores_no_root(self, make_otel: Callable[[], OtelExtender]) -> None:
        otel = make_otel()
        run_id = str(uuid.uuid4())

        _start_run(otel, run_id)
        assert otel._run_roots == {}
        _complete_run(otel, run_id)

        assert otel._run_roots == {}
        otel.on_plan_start(_plan("plan-1"))
        otel.on_plan_complete(_plan("plan-1"), LifecycleOutcome(status="failed", error_type="KeyError"))
        assert otel._plan_spans == {}
        assert not otel._plan_ints

    def test_no_op_provider_without_carrier_or_caller_span_stores_nothing(self) -> None:
        otel = OtelExtender(tracer_provider=trace.NoOpTracerProvider())

        _start_run(otel, str(uuid.uuid4()))

        assert otel._run_roots == {}

    def test_no_op_provider_with_a_carrier_stores_the_carrier_span_ints(self) -> None:
        otel = OtelExtender(tracer_provider=trace.NoOpTracerProvider())
        run_id = str(uuid.uuid4())
        carrier, carrier_trace_id, carrier_span_id = inject_parent_carrier()

        _start_run(otel, run_id, carrier=carrier)

        assert [roots[:2] for roots in otel._run_roots.values()] == [(carrier_trace_id, carrier_span_id)]

    def test_no_op_provider_with_a_caller_span_stores_the_caller_span_ints(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, _ = otel_capture
        otel = OtelExtender(tracer_provider=trace.NoOpTracerProvider())
        run_id = str(uuid.uuid4())

        with provider.get_tracer("mloda-testing-caller").start_as_current_span("caller") as caller:
            context = caller.get_span_context()
            _start_run(otel, run_id)

        assert [roots[:2] for roots in otel._run_roots.values()] == [(context.trace_id, context.span_id)]
        assert list(otel._run_roots) == [run_id]


class TestOtelExtenderPlanSpanHooks:
    def test_run_scope_emits_no_plan_span_for_a_succeeded_plan(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider)

        otel.on_plan_start(_plan("plan-1"))
        otel.on_plan_complete(_plan("plan-1"), LifecycleOutcome(status="succeeded"))

        assert exporter.get_finished_spans() == ()

    @pytest.mark.parametrize(
        ("trace_scope", "outcome", "is_error"),
        [
            pytest.param("plan", LifecycleOutcome(status="succeeded"), False, id="plan_succeeded"),
            pytest.param("plan", LifecycleOutcome(status="failed", error_type="KeyError"), True, id="plan_failed"),
            pytest.param("run", LifecycleOutcome(status="failed", error_type="KeyError"), True, id="run_failed"),
        ],
    )
    @pytest.mark.parametrize("structure_hash", ["abc123", None])
    def test_plan_span_has_attributes_and_status(
        self,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        trace_scope: Any,
        outcome: LifecycleOutcome,
        is_error: bool,
        structure_hash: str | None,
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider, trace_scope=trace_scope)
        completed = dataclasses.replace(
            _plan("plan-1", structure_hash),
            created_at=datetime.datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=datetime.timezone.utc),
        )

        otel.on_plan_start(_plan("plan-1"))
        assert exporter.get_finished_spans() == ()
        otel.on_plan_complete(completed, outcome)

        span = single_span(exporter)
        assert span.name == "mloda.plan"
        assert (span.attributes or {})["mloda.plan.id"] == "plan-1"
        if structure_hash is None:
            assert "mloda.plan.structure_hash" not in (span.attributes or {})
        else:
            assert (span.attributes or {})["mloda.plan.structure_hash"] == structure_hash
        if is_error:
            assert span.status.status_code == StatusCode.ERROR
            assert (span.attributes or {})["error.type"] == "KeyError"
        else:
            assert span.status.status_code != StatusCode.ERROR
        if trace_scope == "run":
            assert span.start_time == 1767323045678901000
            assert span.parent is None

    @pytest.mark.parametrize(
        ("trace_scope", "status"), [("plan", "succeeded"), ("run", "failed")], ids=["plan_succeeded", "run_failed"]
    )
    def test_plan_span_is_a_child_of_the_caller_active_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], trace_scope: Any, status: Any
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider, trace_scope=trace_scope)

        with provider.get_tracer("mloda-testing-caller").start_as_current_span("caller"):
            otel.on_plan_start(_plan("plan-1"))
            otel.on_plan_complete(_plan("plan-1"), LifecycleOutcome(status=status))

        spans = exporter.get_finished_spans()
        assert assert_well_formed_trace(spans).name == "caller"
        assert len(_named(exporter, "mloda.plan")) == 1

    @pytest.mark.parametrize("failed_plan", [False, True], ids=["never_planned", "failed_plan"])
    def test_run_with_an_unknown_plan_falls_back_to_run_scope_parenting(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], failed_plan: bool
    ) -> None:
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider, trace_scope="plan")
        run_id = str(uuid.uuid4())
        carrier, carrier_trace_id, carrier_span_id = inject_parent_carrier()
        if failed_plan:
            otel.on_plan_start(_plan("never-planned"))
            otel.on_plan_complete(_plan("never-planned"), LifecycleOutcome(status="failed", error_type="KeyError"))
            exporter.clear()

        _start_run(otel, run_id, plan_id="never-planned", carrier=carrier)
        _complete_run(otel, run_id)

        root = single_span(exporter)
        assert root.name == "mloda.run"
        assert _ids(root)[0] == carrier_trace_id
        assert root.parent is not None
        assert root.parent.span_id == carrier_span_id

    def test_plan_map_evicts_the_oldest_plan_when_the_bound_is_exceeded(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(otel_extender_module, "_MAX_PLAN_SPANS", 2)
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider, trace_scope="plan")
        for plan_id in ("p1", "p2", "p3"):
            otel.on_plan_start(_plan(plan_id))
            otel.on_plan_complete(_plan(plan_id), LifecycleOutcome(status="succeeded"))
        plan_spans = {(span.attributes or {})["mloda.plan.id"]: span for span in _named(exporter, "mloda.plan")}

        evicted, kept = str(uuid.uuid4()), str(uuid.uuid4())
        _start_run(otel, evicted, plan_id="p1")
        _complete_run(otel, evicted)
        _start_run(otel, kept, plan_id="p3")
        _complete_run(otel, kept)

        roots = {(span.attributes or {})["mloda.run.id"]: span for span in _named(exporter, "mloda.run")}
        assert roots[evicted].parent is None
        assert _ids(roots[evicted])[0] != _ids(plan_spans["p1"])[0]
        assert roots[kept].parent is not None
        assert roots[kept].parent.span_id == _ids(plan_spans["p3"])[1]

    def test_a_plan_whose_runs_keep_coming_is_not_evicted(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(otel_extender_module, "_MAX_PLAN_SPANS", 2)
        provider, exporter = otel_capture
        otel = OtelExtender(tracer_provider=provider, trace_scope="plan")
        for plan_id in ("pa", "pb"):
            otel.on_plan_start(_plan(plan_id))
            otel.on_plan_complete(_plan(plan_id), LifecycleOutcome(status="succeeded"))
        refresh = str(uuid.uuid4())
        _start_run(otel, refresh, plan_id="pa")
        _complete_run(otel, refresh)
        otel.on_plan_start(_plan("pc"))
        otel.on_plan_complete(_plan("pc"), LifecycleOutcome(status="succeeded"))
        plan_spans = {(span.attributes or {})["mloda.plan.id"]: span for span in _named(exporter, "mloda.plan")}

        kept, evicted = str(uuid.uuid4()), str(uuid.uuid4())
        _start_run(otel, kept, plan_id="pa")
        _complete_run(otel, kept)
        _start_run(otel, evicted, plan_id="pb")
        _complete_run(otel, evicted)

        roots = {(span.attributes or {})["mloda.run.id"]: span for span in _named(exporter, "mloda.run")}
        assert roots[kept].parent is not None
        assert roots[kept].parent.span_id == _ids(plan_spans["pa"])[1]
        assert roots[evicted].parent is None


class TestOtelExtenderStepSpanNaming:
    @pytest.mark.parametrize(
        ("feature_group_class", "expected"),
        [
            ("pkg.mod.MyFeatureGroup", "calculate MyFeatureGroup"),
            ("pkg.mod.Outer.Inner", "calculate Inner"),
            ("Bare", "calculate Bare"),
            (None, "calculate"),
        ],
    )
    def test_calculate_span_name_carries_the_short_feature_group_name(
        self,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        feature_group_class: str | None,
        expected: str,
    ) -> None:
        provider, exporter = otel_capture

        with make_hook_context(feature_group_class=feature_group_class).activate():
            OtelExtender(tracer_provider=provider)(lambda: None)

        assert single_span(exporter).name == expected

    @pytest.mark.parametrize(
        ("hook", "name"),
        [
            (ExtenderHook.VALIDATE_INPUT_FEATURE, "mloda.validate.input"),
            (ExtenderHook.VALIDATE_OUTPUT_FEATURE, "mloda.validate.output"),
            (ExtenderHook.INPUT_DATA_LOAD, "mloda.load"),
        ],
    )
    def test_other_hooks_keep_their_names(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], hook: ExtenderHook, name: str
    ) -> None:
        provider, exporter = otel_capture

        with make_hook_context(hook=hook).activate():
            OtelExtender(tracer_provider=provider)(lambda: None)

        assert single_span(exporter).name == name

    def test_calculate_span_has_the_shared_step_run_id(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture
        run_id = str(uuid.uuid4())
        step_uuid = uuid.uuid4()
        context = make_hook_context(run_id=run_id, feature_names=("b", "a"), step_uuid=step_uuid)

        with context.activate():
            OtelExtender(tracer_provider=provider)(lambda: None)

        expected = step_run_id(run_id, owner_name(context, lambda: None), ("a", "b"), "PyArrowTable", step_uuid)
        assert expected is not None
        assert single_span_attributes(exporter)["mloda.step.run_id"] == expected

    def test_calculate_span_has_no_step_run_id_without_a_uuid_run_id(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture

        with make_hook_context(run_id="not-a-uuid").activate():
            OtelExtender(tracer_provider=provider)(lambda: None)

        assert "mloda.step.run_id" not in single_span_attributes(exporter)

    @pytest.mark.parametrize(
        ("join_type", "expected"), [("inner", "join inner"), ("asof", "join asof"), (None, "join")]
    )
    def test_join_span_name_carries_the_join_type(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], join_type: str | None, expected: str
    ) -> None:
        provider, exporter = otel_capture

        with _join_context(join_type=join_type).activate():
            OtelExtender(tracer_provider=provider)(lambda: None)

        assert single_span(exporter).name == expected

    @pytest.mark.parametrize(
        "hook",
        [
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.VALIDATE_INPUT_FEATURE,
            ExtenderHook.VALIDATE_OUTPUT_FEATURE,
            ExtenderHook.INPUT_DATA_LOAD,
            ExtenderHook.JOIN,
        ],
    )
    def test_every_span_carries_the_plan_id_when_set(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter], hook: ExtenderHook
    ) -> None:
        provider, exporter = otel_capture

        with make_hook_context(hook=hook, plan_id="plan-x").activate():
            OtelExtender(tracer_provider=provider)(lambda: None)

        assert single_span_attributes(exporter)["mloda.plan.id"] == "plan-x"

    def test_plan_id_attribute_absent_when_none(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        provider, exporter = otel_capture

        with make_hook_context(plan_id=None).activate():
            OtelExtender(tracer_provider=provider)(lambda: None)

        assert "mloda.plan.id" not in single_span_attributes(exporter)


class TestOtelExtenderRootSpanPickling:
    def _marker_extender(self, tmp_path: Path, **kwargs: Any) -> tuple[OtelExtender, Path]:
        marker_path = tmp_path / "root_spans.jsonl"
        provider = RebuildingSpanCaptureProvider(marker_path=marker_path, records=True)
        return OtelExtender(tracer_provider=provider, **kwargs), marker_path

    def test_pickling_with_a_live_root_keeps_the_ints_so_the_copy_parents_to_the_root(self, tmp_path: Path) -> None:
        otel, marker_path = self._marker_extender(tmp_path)
        run_id = str(uuid.uuid4())

        _start_run(otel, run_id)
        copy = pickle.loads(pickle.dumps(otel))  # nosec
        _step(copy, run_id)
        _complete_run(otel, run_id)

        records = read_span_records(marker_path)
        root = next(record for record in records if record["name"] == "mloda.run")
        step = next(record for record in records if record["name"].startswith("calculate"))
        assert step["trace_id"] == root["trace_id"]
        assert step["parent_span_id"] == root["span_id"]

    def test_unpickled_copy_has_a_working_lock_and_empty_span_maps(self, tmp_path: Path) -> None:
        otel, marker_path = self._marker_extender(tmp_path, trace_scope="plan")
        otel.on_plan_start(_plan("plan-1"))
        run_id = str(uuid.uuid4())
        _start_run(otel, run_id)

        copy = pickle.loads(pickle.dumps(otel))  # nosec

        # The copy never owned the Span objects, so completing the run on it ends nothing.
        _complete_run(copy, run_id)
        otel.on_plan_complete(_plan("plan-1"), LifecycleOutcome(status="succeeded"))
        _complete_run(otel, run_id)
        # A fresh run on the copy works, proving the rebuilt lock.
        fresh = str(uuid.uuid4())
        _start_run(copy, fresh, plan_id="other")
        _complete_run(copy, fresh)

        names = [record["name"] for record in read_span_records(marker_path)]
        assert names.count("mloda.run") == 2
        assert names.count("mloda.plan") == 1

    def test_copy_does_not_share_the_run_roots_with_the_original(self, tmp_path: Path) -> None:
        otel, _ = self._marker_extender(tmp_path)
        run_id = str(uuid.uuid4())
        _start_run(otel, run_id)

        duplicate = copy.copy(otel)
        _complete_run(otel, run_id)

        assert run_id in duplicate._run_roots
        assert duplicate._run_roots is not otel._run_roots

    def test_pickled_state_run_roots_is_a_snapshot(self, tmp_path: Path) -> None:
        otel, _ = self._marker_extender(tmp_path)
        run_id = str(uuid.uuid4())
        _start_run(otel, run_id)

        state = otel.__getstate__()
        _complete_run(otel, run_id)

        assert run_id in state["_run_roots"]


class TestOtelExtenderClose:
    """close() flushes the resolved tracer_provider and meter_provider within one close_timeout; never terminal,
    never raises, never calls shutdown() (core, not the extender, owns provider lifetime)."""

    def test_close_flushes_the_injected_provider_with_timeout_millis(self) -> None:
        from mloda.community.extenders.shared.teardown import CLOSE_TIMEOUT

        provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelExtender(tracer_provider=provider)

        otel.close()

        provider.force_flush.assert_called_once_with(timeout_millis=int(CLOSE_TIMEOUT * 1000))

    def test_close_caps_flush_timeout_to_the_active_close_context(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelExtender(tracer_provider=provider)
        otel.close_timeout = 5.0

        with active_close_context(3.0):
            otel.close()

        provider.force_flush.assert_called_once()
        assert 0 < provider.force_flush.call_args.kwargs["timeout_millis"] <= 3000

    def test_close_flushes_the_global_provider_under_use_sdk_defaults(self, ambient_provider: _AmbientProvider) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        ambient_provider.provider = provider
        otel = OtelExtender(use_sdk_defaults=True)

        otel.close()

        provider.force_flush.assert_called_once()

    def test_close_swallows_a_raising_force_flush_and_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = Mock(force_flush=Mock(side_effect=RuntimeError("flush boom")))
        otel = OtelExtender(tracer_provider=provider)

        with caplog.at_level(logging.WARNING):
            otel.close()  # must not raise

        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OtelExtender" in message for message in warnings), warnings
        assert "flush boom" not in caplog.text

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

    def test_close_flushes_the_injected_meter_provider_with_timeout_millis(self) -> None:
        tracer_provider = Mock(force_flush=Mock(return_value=True))
        meter_provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelExtender(tracer_provider=tracer_provider, meter_provider=meter_provider)

        otel.close()

        tracer_provider.force_flush.assert_called_once()
        meter_provider.force_flush.assert_called_once()
        assert 0 < meter_provider.force_flush.call_args.kwargs["timeout_millis"] <= 1000

    def test_metrics_only_close_flushes_the_meter_provider(self) -> None:
        meter_provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelExtender(meter_provider=meter_provider)

        otel.close()

        meter_provider.force_flush.assert_called_once()

    def test_close_flushes_the_global_meter_provider_under_use_sdk_defaults(
        self, ambient_provider: _AmbientProvider
    ) -> None:
        meter_provider = Mock(force_flush=Mock(return_value=True))
        ambient_provider.meter_provider = meter_provider
        otel = OtelExtender(use_sdk_defaults=True)

        otel.close()

        meter_provider.force_flush.assert_called_once()

    @pytest.mark.parametrize("attribute", ["provider", "meter_provider"])
    def test_inert_extender_close_touches_no_provider(self, ambient_provider: _AmbientProvider, attribute: str) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        setattr(ambient_provider, attribute, provider)
        otel = OtelExtender()  # no injected provider, use_sdk_defaults False: inert

        otel.close()

        provider.force_flush.assert_not_called()
        assert ambient_provider.meter_provider_calls == 0

    @pytest.mark.parametrize("sink", ["tracer_provider", "meter_provider", "both"])
    def test_close_never_calls_shutdown(self, sink: str) -> None:
        providers = {key: Mock(force_flush=Mock(return_value=True)) for key in ("tracer_provider", "meter_provider")}
        injected = providers if sink == "both" else {sink: providers[sink]}
        otel = OtelExtender(**injected)

        otel.close()

        for provider in injected.values():
            provider.shutdown.assert_not_called()

    @pytest.mark.parametrize(
        ("failure", "wording"),
        [(RuntimeError("flush boom"), "meter_provider"), (False, "metrics")],
        ids=["raises", "returns_false"],
    )
    def test_close_logs_a_warning_naming_a_failing_meter_flush_and_never_raises(
        self, caplog: pytest.LogCaptureFixture, failure: Any, wording: str
    ) -> None:
        tracer_provider = Mock(force_flush=Mock(return_value=True))
        flush = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
        otel = OtelExtender(tracer_provider=tracer_provider, meter_provider=Mock(force_flush=flush))

        with caplog.at_level(logging.WARNING):
            otel.close()  # must not raise

        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OtelExtender" in message and wording in message for message in warnings), warnings
        assert not any("tracer_provider" in message for message in warnings), warnings
        assert "flush boom" not in caplog.text

    def test_a_failing_tracer_flush_does_not_skip_the_meter_flush(self) -> None:
        tracer_provider = Mock(force_flush=Mock(side_effect=RuntimeError("flush boom")))
        meter_provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelExtender(tracer_provider=tracer_provider, meter_provider=meter_provider)

        otel.close()

        meter_provider.force_flush.assert_called_once()

    def test_close_shares_one_budget_between_the_tracer_and_meter_flush(self) -> None:
        meter_provider = Mock(force_flush=Mock(return_value=True))
        with blocking_flush_provider() as tracer_provider:
            otel = OtelExtender(tracer_provider=tracer_provider, meter_provider=meter_provider)
            otel.close_timeout = 0.3

            start = time.monotonic()
            still_running, outcome = call_with_join_timeout(otel.close, join_timeout=2.0)
            elapsed = time.monotonic() - start

        assert not still_running
        if "error" in outcome:
            raise outcome["error"]
        assert elapsed < 0.55, elapsed  # one 0.3s budget, never 2x
        # The tracer flush used the whole budget; the meter flush is skipped or gets a ~0 budget.
        if meter_provider.force_flush.call_args is not None:
            assert meter_provider.force_flush.call_args.kwargs["timeout_millis"] <= 50

    @pytest.mark.parametrize("sink", ["tracer_provider", "meter_provider", "both"])
    def test_negative_close_timeout_calls_force_flush_with_no_args(self, sink: str) -> None:
        providers = {key: Mock(force_flush=Mock(return_value=True)) for key in ("tracer_provider", "meter_provider")}
        injected = providers if sink == "both" else {sink: providers[sink]}
        otel = OtelExtender(**injected)
        otel.close_timeout = -1.0

        otel.close()

        for provider in injected.values():
            provider.force_flush.assert_called_once_with()

    def test_close_warning_texts_name_what_failed_to_flush(self, caplog: pytest.LogCaptureFixture) -> None:
        tracer_false = OtelExtender(tracer_provider=Mock(force_flush=Mock(return_value=False)))
        meter_false = OtelExtender(meter_provider=Mock(force_flush=Mock(return_value=False)))
        both_raise = OtelExtender(
            tracer_provider=Mock(force_flush=Mock(side_effect=RuntimeError("boom"))),
            meter_provider=Mock(force_flush=Mock(side_effect=RuntimeError("boom"))),
        )

        with caplog.at_level(logging.WARNING):
            tracer_false.close()
            meter_false.close()
            both_raise.close()

        assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
            "OtelExtender did not flush all spans within its close budget",
            "OtelExtender did not flush all metrics within its close budget",
            "OtelExtender failed to flush tracer_provider: RuntimeError",
            "OtelExtender failed to flush meter_provider: RuntimeError",
        ]

    def test_close_skips_the_meter_flush_and_warns_when_the_budget_is_spent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        meter_provider = Mock(force_flush=Mock(return_value=True))
        with blocking_flush_provider() as tracer_provider:
            otel = OtelExtender(tracer_provider=tracer_provider, meter_provider=meter_provider)
            otel.close_timeout = 0.1

            with caplog.at_level(logging.WARNING):
                still_running, outcome = call_with_join_timeout(otel.close, join_timeout=2.0)

        assert not still_running
        if "error" in outcome:
            raise outcome["error"]
        meter_provider.force_flush.assert_not_called()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert "OtelExtender did not flush all metrics within its close budget" in warnings, warnings

    def test_inert_close_never_raises_even_with_a_nonsensical_close_timeout(
        self, ambient_provider: _AmbientProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        otel = OtelExtender()
        otel.close_timeout = None  # type: ignore[assignment]

        with caplog.at_level(logging.WARNING):
            otel.close()

        assert ambient_provider.meter_provider_calls == 0
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_close_with_a_nonsensical_close_timeout_never_raises_and_logs_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelExtender(tracer_provider=provider)
        otel.close_timeout = None  # type: ignore[assignment]

        with caplog.at_level(logging.WARNING):
            otel.close()  # must not raise

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("OtelExtender" in message and "Error" in message for message in warnings), warnings


def _collected(reader: InMemoryMetricReader) -> dict[str, Any]:
    data = reader.get_metrics_data()
    if data is None:
        return {}
    return {m.name: m for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics}


def _points(reader: InMemoryMetricReader, name: str) -> list[Any]:
    metric = _collected(reader).get(name)
    return [] if metric is None else list(metric.data.data_points)


def _single_point(reader: InMemoryMetricReader, name: str) -> Any:
    points = _points(reader, name)
    assert len(points) == 1, (name, points)
    return points[0]


def _failure_type(exc_type: type[BaseException]) -> str:
    return f"{exc_type.__module__}.{exc_type.__qualname__}"


class _RaisingMeterProvider(metrics.MeterProvider):
    """Picklable meter provider whose get_meter always raises; counts the attempts per instance."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.get_meter_calls = 0

    def get_meter(self, *args: Any, **kwargs: Any) -> metrics.Meter:
        self.get_meter_calls += 1
        raise self.error


class _StepBoom(Exception):
    pass


class _Halt(BaseException):
    pass


def _rows_context() -> HookContext:
    return make_hook_context(rows_in=3, rows_out=2)


def _failing_step(otel: OtelExtender, error: BaseException, context: HookContext | None = None) -> None:
    def func() -> None:
        raise error

    with (context or make_hook_context()).activate():
        with pytest.raises(type(error)) as caught:
            otel(func)
    assert caught.value is error


class TestOtelExtenderMetrics:
    """Metrics emitted through opentelemetry.metrics (API only): step and run durations plus row counters."""

    def test_exactly_four_instruments_with_kinds_units_and_bucket_advisory(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture

        _call_once(OtelExtender(meter_provider=provider), _rows_context())

        collected = _collected(reader)
        assert set(collected) == {_STEP_DURATION, _ROWS_IN, _ROWS_OUT}
        otel = OtelExtender(meter_provider=provider)
        _complete_run(otel, "r", started_at=datetime.datetime.now(datetime.timezone.utc), status="succeeded")
        collected = _collected(reader)
        assert set(collected) == {_RUN_DURATION, _STEP_DURATION, _ROWS_IN, _ROWS_OUT}
        for name in (_RUN_DURATION, _STEP_DURATION):
            assert isinstance(collected[name].data, Histogram), name
            assert collected[name].unit == "s", name
            assert tuple(collected[name].data.data_points[0].explicit_bounds) == _DURATION_BUCKETS, name
        for name in (_ROWS_IN, _ROWS_OUT):
            assert isinstance(collected[name].data, Sum), name
            assert collected[name].data.is_monotonic, name
            assert collected[name].unit == "{row}", name

    @pytest.mark.parametrize(
        ("context", "operation", "has_feature_group"),
        [
            (make_hook_context(hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE), "calculate", True),
            (make_hook_context(hook=ExtenderHook.VALIDATE_INPUT_FEATURE), "validate", True),
            (make_hook_context(hook=ExtenderHook.VALIDATE_OUTPUT_FEATURE), "validate", True),
            (make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD), "load", True),
            (_join_context(join_type="inner"), "join", False),
        ],
        ids=["calculate", "validate_input", "validate_output", "load", "join"],
    )
    def test_step_duration_attributes_per_hook(
        self,
        metric_capture: tuple[MeterProvider, InMemoryMetricReader],
        context: HookContext,
        operation: str,
        has_feature_group: bool,
    ) -> None:
        provider, reader = metric_capture

        _call_once(OtelExtender(meter_provider=provider), context)

        point = _single_point(reader, _STEP_DURATION)
        expected = {"mloda.operation.name": operation, "mloda.compute_framework.name": "PyArrowTable"}
        if has_feature_group:
            expected["mloda.feature_group.name"] = "mloda.testing.DummyFeatureGroup"
        assert dict(point.attributes) == expected
        assert point.count == 1
        assert point.sum >= 0

    def test_failure_adds_error_type_records_the_duration_and_propagates(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture
        error = _StepBoom("boom")

        _failing_step(OtelExtender(meter_provider=provider), error, make_hook_context(rows_in=3, rows_out=2))

        point = _single_point(reader, _STEP_DURATION)
        assert point.attributes["error.type"] == _failure_type(_StepBoom)
        assert point.attributes["mloda.operation.name"] == "calculate"
        assert _points(reader, _ROWS_IN) == []
        assert _points(reader, _ROWS_OUT) == []

    def test_base_exception_is_counted_and_propagates(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture

        _failing_step(OtelExtender(meter_provider=provider), _Halt("halt"))

        assert _single_point(reader, _STEP_DURATION).attributes["error.type"] == _failure_type(_Halt)

    def test_success_has_no_error_type(self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]) -> None:
        provider, reader = metric_capture

        _call_once(OtelExtender(meter_provider=provider), _rows_context())

        assert "error.type" not in _single_point(reader, _STEP_DURATION).attributes

    @pytest.mark.parametrize(
        ("hook", "rows_in", "rows_out", "expected_in", "expected_out"),
        [
            (ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, 7, 5, 7, 5),
            (ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, None, 5, None, 5),
            (ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, None, None, None, None),
            (ExtenderHook.INPUT_DATA_LOAD, None, 9, None, 9),
            (ExtenderHook.VALIDATE_INPUT_FEATURE, 7, 5, None, None),
            (ExtenderHook.VALIDATE_OUTPUT_FEATURE, 7, 5, None, None),
            (ExtenderHook.JOIN, 7, 5, None, None),
        ],
        ids=["calculate", "calculate_no_in", "calculate_no_rows", "load", "validate_in", "validate_out", "join"],
    )
    def test_rows_are_recorded_only_for_calculate_and_load_when_known(
        self,
        metric_capture: tuple[MeterProvider, InMemoryMetricReader],
        hook: ExtenderHook,
        rows_in: int | None,
        rows_out: int | None,
        expected_in: int | None,
        expected_out: int | None,
    ) -> None:
        provider, reader = metric_capture

        _call_once(
            OtelExtender(meter_provider=provider), make_hook_context(hook=hook, rows_in=rows_in, rows_out=rows_out)
        )

        for name, expected in ((_ROWS_IN, expected_in), (_ROWS_OUT, expected_out)):
            if expected is None:
                assert _points(reader, name) == [], name
            else:
                point = _single_point(reader, name)
                assert point.value == expected, name
                assert set(point.attributes) == _STEP_ATTRIBUTES, name

    def test_attributes_stay_within_the_allowlist_and_leak_no_identifiers(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture
        otel = OtelExtender(meter_provider=provider)
        step_uuid = uuid.uuid4()
        secrets = (
            "run-id-secret",
            "plan-id-secret",
            "identity-secret",
            "tenant-secret",
            "feature-secret",
            str(step_uuid),
        )
        for hook in (ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD, ExtenderHook.JOIN):
            context = make_hook_context(
                hook=hook,
                rows_in=1,
                rows_out=2,
                run_id="run-id-secret",
                plan_id="plan-id-secret",
                step_uuid=step_uuid,
                worker_index=3,
                data_access_identity="identity-secret",
                data_access_format="csv",
                feature_names=("feature-secret",),
                plugin_version="9.9.9",
                feature_group_version="7",
                tenant_id="tenant-secret",
                project_id="project-secret",
                principal="principal-secret",
                join_type="inner",
                join_keys=("key-secret",),
                declared_attributes={"declared": "declared-secret"},
            )
            _call_once(otel, context)
        _failing_step(
            otel, _StepBoom("boom"), make_hook_context(run_id="run-id-secret", feature_names=("feature-secret",))
        )
        _complete_run(
            otel,
            "run-id-secret",
            started_at=datetime.datetime.now(datetime.timezone.utc),
            status="failed",
            error_type="RuntimeError",
        )

        collected = _collected(reader)
        assert collected
        for name, metric in collected.items():
            for point in metric.data.data_points:
                assert set(point.attributes) <= _METRIC_ATTRIBUTE_ALLOWLIST, (name, dict(point.attributes))
                for value in point.attributes.values():
                    assert not any(secret in str(value) for secret in secrets), (name, dict(point.attributes))
                    assert str(value) not in ("9.9.9", "3", "csv", "key-secret"), (name, dict(point.attributes))

    def test_no_hook_context_records_no_metrics(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture

        assert OtelExtender(meter_provider=provider)(lambda: 42) == 42

        assert _collected(reader) == {}

    def test_pre_call_attribute_failure_records_no_metric(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, reader = metric_capture

        def broken_set_context_attributes(span: Any, context: Any) -> None:
            raise RuntimeError("attrs boom")

        monkeypatch.setattr(otel_extender_module, "_set_context_attributes", broken_set_context_attributes)
        ran = Mock()

        with make_hook_context().activate():
            with pytest.raises(RuntimeError, match="attrs boom"):
                OtelExtender(meter_provider=provider)(ran)

        ran.assert_not_called()
        assert _collected(reader) == {}

    def test_step_duration_times_only_the_wrapped_call(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture

        with make_hook_context().activate():
            OtelExtender(meter_provider=provider)(lambda: time.sleep(0.05))

        assert 0.05 <= _single_point(reader, _STEP_DURATION).sum < 5

    @pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
    def test_run_duration_is_recorded_with_the_run_status(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], status: Any
    ) -> None:
        provider, reader = metric_capture
        started_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=2)
        error_type = "ValueError" if status == "failed" else None

        _complete_run(
            OtelExtender(meter_provider=provider),
            str(uuid.uuid4()),
            started_at=started_at,
            status=status,
            error_type=error_type,
        )

        point = _single_point(reader, _RUN_DURATION)
        expected: dict[str, str] = {"mloda.run.status": status}
        if error_type is not None:
            expected["error.type"] = error_type
        assert dict(point.attributes) == expected
        assert 2 <= point.sum < 60

    def test_failed_run_without_an_error_type_omits_the_attribute(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture

        _complete_run(
            OtelExtender(meter_provider=provider),
            "r",
            started_at=datetime.datetime.now(datetime.timezone.utc),
            status="failed",
        )

        assert dict(_single_point(reader, _RUN_DURATION).attributes) == {"mloda.run.status": "failed"}

    def test_run_without_started_at_records_no_duration(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture

        _complete_run(OtelExtender(meter_provider=provider), "r")

        assert _points(reader, _RUN_DURATION) == []

    def test_run_without_run_id_still_records_the_duration(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture

        _complete_run(
            OtelExtender(meter_provider=provider),
            None,
            started_at=datetime.datetime.now(datetime.timezone.utc),
            status="succeeded",
        )

        assert _single_point(reader, _RUN_DURATION).count == 1

    def test_run_started_in_the_future_is_clamped_to_zero(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture
        started_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)

        _complete_run(OtelExtender(meter_provider=provider), "r", started_at=started_at, status="succeeded")

        assert _single_point(reader, _RUN_DURATION).sum == 0

    def test_use_sdk_defaults_resolves_the_global_meter_provider(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], ambient_provider: _AmbientProvider
    ) -> None:
        provider, reader = metric_capture
        ambient_provider.meter_provider = provider

        _call_once(OtelExtender(use_sdk_defaults=True), _rows_context())

        assert _single_point(reader, _STEP_DURATION).count == 1

    def test_injected_meter_provider_wins_over_the_global_one(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], ambient_provider: _AmbientProvider
    ) -> None:
        provider, reader = metric_capture
        global_provider, global_reader = make_metric_capture()
        ambient_provider.meter_provider = global_provider

        _call_once(OtelExtender(meter_provider=provider, use_sdk_defaults=True), _rows_context())

        assert _single_point(reader, _STEP_DURATION).count == 1
        assert _collected(global_reader) == {}

    def test_use_sdk_defaults_without_an_sdk_meter_provider_warns_nothing_about_metrics(
        self, ambient_provider: _AmbientProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            assert _call_once(OtelExtender(use_sdk_defaults=True), _rows_context()) == 42

        assert not [
            r for r in caplog.records if "meter" in r.getMessage().lower() or "metric" in r.getMessage().lower()
        ]

    def test_unconfigured_extender_never_resolves_the_meter_provider_and_records_nothing(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], ambient_provider: _AmbientProvider
    ) -> None:
        provider, reader = metric_capture
        ambient_provider.meter_provider = provider
        otel = OtelExtender()

        assert _call_once(otel, _rows_context()) == 42
        _complete_run(otel, "r", started_at=datetime.datetime.now(datetime.timezone.utc), status="succeeded")

        assert ambient_provider.meter_provider_calls == 0
        assert _collected(reader) == {}

    def test_metrics_only_injection_does_not_log_the_inert_warning(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, _ = metric_capture

        with caplog.at_level(logging.WARNING):
            _call_once(OtelExtender(meter_provider=provider), _rows_context())

        assert not [r for r in caplog.records if "inert" in r.getMessage().lower()], caplog.records

    def test_one_meter_is_created_per_provider_across_extender_instances(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, reader = metric_capture
        spy = Mock(wraps=provider.get_meter)
        monkeypatch.setattr(provider, "get_meter", spy)

        for _ in range(5):
            otel = OtelExtender(meter_provider=provider)
            _call_once(otel, _rows_context())
            _complete_run(otel, "r", started_at=datetime.datetime.now(datetime.timezone.utc), status="succeeded")

        spy.assert_called_once()
        assert spy.call_args.args[0] == "mloda_community_otel"
        assert _single_point(reader, _STEP_DURATION).count == 5

    @pytest.mark.parametrize("raise_on_error", [False, True])
    def test_a_raising_meter_provider_never_breaks_the_call_and_logs_a_warning(
        self, caplog: pytest.LogCaptureFixture, raise_on_error: bool
    ) -> None:
        meter_provider = Mock(get_meter=Mock(side_effect=RuntimeError("meter boom")))
        otel = OtelExtender(meter_provider=meter_provider, raise_on_error=raise_on_error)
        error = _StepBoom("step boom")

        with caplog.at_level(logging.WARNING):
            assert _call_once(otel, _rows_context()) == 42
            _failing_step(otel, error)
            _complete_run(otel, "r", started_at=datetime.datetime.now(datetime.timezone.utc), status="succeeded")

        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("OtelExtender metric recording failed: RuntimeError" in m for m in messages), messages
        assert "meter boom" not in caplog.text

    def test_a_type_error_while_building_instruments_is_one_attempt_and_one_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        meter_provider = _RaisingMeterProvider(TypeError("meter boom"))
        otel = OtelExtender(meter_provider=meter_provider)

        with caplog.at_level(logging.WARNING):
            assert _call_once(otel) == 42

        assert meter_provider.get_meter_calls == 1
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("OtelExtender metric recording failed: TypeError" in m for m in messages), messages

    def test_metric_recording_failures_warn_once_per_instance(self, caplog: pytest.LogCaptureFixture) -> None:
        otel = OtelExtender(meter_provider=_RaisingMeterProvider(RuntimeError("meter boom")))

        with caplog.at_level(logging.WARNING):
            _call_once(otel)
            _call_once(otel)
            _complete_run(otel, "r", started_at=datetime.datetime.now(datetime.timezone.utc))
            assert len(self._failure_records(caplog)) == 1, caplog.records

            copy = pickle.loads(pickle.dumps(otel))  # nosec
            _call_once(copy)
            _call_once(copy)

        assert len(self._failure_records(caplog)) == 2, caplog.records

    @staticmethod
    def _failure_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
        return [r for r in caplog.records if "metric recording failed" in r.getMessage()]

    def test_repeated_recordings_on_the_api_proxy_meter_provider_add_one_meter(
        self, ambient_provider: _AmbientProvider
    ) -> None:
        proxy: Any = ambient_provider.meter_provider
        if not hasattr(proxy, "_meters"):
            pytest.skip("global meter provider already installed in this process")
        before = len(proxy._meters)

        for _ in range(5):
            otel = OtelExtender(use_sdk_defaults=True)
            _call_once(otel, _rows_context())
            _complete_run(otel, "r", started_at=datetime.datetime.now(datetime.timezone.utc))

        assert len(proxy._meters) - before <= 1

    def test_a_raising_instrument_never_breaks_the_call(self, caplog: pytest.LogCaptureFixture) -> None:
        instrument_mock = Mock(
            record=Mock(side_effect=RuntimeError("record boom")), add=Mock(side_effect=RuntimeError("add boom"))
        )
        meter = Mock(
            create_histogram=Mock(return_value=instrument_mock), create_counter=Mock(return_value=instrument_mock)
        )
        otel = OtelExtender(meter_provider=Mock(get_meter=Mock(return_value=meter)), raise_on_error=True)

        with caplog.at_level(logging.WARNING):
            assert _call_once(otel, _rows_context()) == 42

        assert "OtelExtender metric recording failed: RuntimeError" in caplog.text

    def test_run_all_records_step_and_run_metrics(
        self,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        metric_capture: tuple[MeterProvider, InMemoryMetricReader],
    ) -> None:
        tracer_provider, _ = otel_capture
        meter_provider, reader = metric_capture

        values = run_value_int(OtelExtender(tracer_provider=tracer_provider, meter_provider=meter_provider))

        assert values == expected_value_int()
        durations = _points(reader, _STEP_DURATION)
        assert any(point.attributes["mloda.operation.name"] == "calculate" for point in durations), durations
        run_points = _points(reader, _RUN_DURATION)
        assert [point.attributes["mloda.run.status"] for point in run_points] == ["succeeded"]
