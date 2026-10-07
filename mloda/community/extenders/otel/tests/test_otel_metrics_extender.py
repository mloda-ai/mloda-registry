"""Tests for OtelMetricsExtender: contract compliance via ExtenderContractTestMixin, plus metric-specific
instrument, attribute, resolution, failure, pickling and close checks.

Direct __call__ tests wrap calls in a manually built HookContext.activate() scope; each gets its own
isolated (MeterProvider, InMemoryMetricReader) pair via the metric_capture fixture.
"""

from __future__ import annotations

import datetime
import inspect
import logging
import pickle  # nosec
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from mloda.steward import (
    CompositeExtender,
    Extender,
    ExtenderHook,
    HookContext,
    LifecycleOutcome,
    RunContext,
    pickle_failure_reason,
)
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import Histogram, InMemoryMetricReader, Sum
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mloda.community.extenders.otel import OtelExtender, OtelMetricsExtender
from mloda.community.extenders.otel import otel_extender as otel_extender_module
from mloda.community.extenders.otel import otel_metrics_extender as otel_metrics_extender_module
from mloda.community.extenders.shared.teardown import CLOSE_TIMEOUT
from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.flush import active_close_context, blocking_flush_provider, call_with_join_timeout
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.otel import (
    _meter_provider_resolution_spy,
    make_metric_capture,
    make_span_capture,
    single_span,
)
from mloda.testing.extenders.runners import expected_value_int, run_value_int

_RUN_DURATION = "mloda.run.duration"
_STEP_DURATION = "mloda.step.duration"
_ROWS_IN = "mloda.step.rows.in"
_ROWS_OUT = "mloda.step.rows.out"
_STEP_ATTRIBUTES = {"mloda.operation.name", "mloda.feature_group.name", "mloda.compute_framework.name"}
_METRIC_ATTRIBUTE_ALLOWLIST = _STEP_ATTRIBUTES | {"error.type", "mloda.run.status"}
_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)

_NO_SDK_PROVIDER_MARKER = "found no OpenTelemetry SDK meter provider"


@pytest.fixture
def metric_capture() -> Iterator[tuple[MeterProvider, InMemoryMetricReader]]:
    """A fresh, isolated (provider, reader) pair per test; never touches the global meter provider."""
    provider, reader = make_metric_capture()
    yield provider, reader
    provider.shutdown()


@pytest.fixture
def otel_capture() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    """A fresh, isolated (provider, exporter) pair per test; never touches the global provider."""
    provider, exporter = make_span_capture()
    yield provider, exporter
    provider.shutdown()


class _AmbientMeterProvider:
    """What the patched opentelemetry.metrics.get_meter_provider returns; reassign .meter_provider to switch."""

    def __init__(self) -> None:
        # The real API default (the proxy), captured before get_meter_provider is patched.
        self.meter_provider: metrics.MeterProvider = metrics.get_meter_provider()
        self.meter_provider_calls = 0

    def get_meter_provider(self) -> metrics.MeterProvider:
        self.meter_provider_calls += 1
        return self.meter_provider


@pytest.fixture
def ambient_provider(monkeypatch: pytest.MonkeyPatch) -> _AmbientMeterProvider:
    """Patch get_meter_provider so no test installs a real global meter provider."""
    holder = _AmbientMeterProvider()
    monkeypatch.setattr(metrics, "get_meter_provider", holder.get_meter_provider)
    return holder


def _call_once(otel: OtelMetricsExtender, context: HookContext | None = None) -> Any:
    with (context or make_hook_context()).activate():
        return otel(lambda: 42)


def _complete_run(
    otel: OtelMetricsExtender,
    run_id: str | None,
    status: Any = "succeeded",
    error_type: str | None = None,
    started_at: datetime.datetime | None = None,
) -> None:
    otel.on_run_complete(
        RunContext(run_id=run_id, started_at=started_at), LifecycleOutcome(status=status, error_type=error_type)
    )


def _join_context(**kwargs: Any) -> HookContext:
    """Mirrors core's JOIN context shape: no feature group, no feature names."""
    return make_hook_context(
        hook=ExtenderHook.JOIN, feature_group_class=None, feature_group_version=None, feature_names=(), **kwargs
    )


def _marker_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if _NO_SDK_PROVIDER_MARKER in r.getMessage()]


def _inert_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "inert" in r.getMessage().lower()]


def _metric_names(reader: InMemoryMetricReader) -> list[str]:
    data = reader.get_metrics_data()
    return (
        [] if data is None else [m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics]
    )


class TestOtelMetricsExtenderContract(ExtenderContractTestMixin):
    """OtelMetricsExtender must satisfy the shared Extender contract."""

    @classmethod
    def extender_class(cls) -> type[OtelMetricsExtender]:
        return OtelMetricsExtender

    def make_extender(self, *, raise_on_error: bool | None = None) -> OtelMetricsExtender:
        provider, _ = make_metric_capture()
        if raise_on_error is None:
            return OtelMetricsExtender(meter_provider=provider)
        return OtelMetricsExtender(meter_provider=provider, raise_on_error=raise_on_error)

    def own_failure(self) -> AbstractContextManager[Any]:
        return patch.object(MeterProvider, "get_meter", side_effect=RuntimeError("otel metrics boom"))

    @classmethod
    def raise_on_error_default(cls) -> bool:
        return False

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
    def has_backend_sink(cls) -> bool:
        return True

    def sink_resolution_spy(self) -> AbstractContextManager[list[Any]]:
        return _meter_provider_resolution_spy()

    def ambient_sink_captured(self, spy: list[Any]) -> list[Any] | None:
        return [name for reader in spy for name in _metric_names(reader)]

    def make_extender_with_sink_probe(self) -> tuple[Extender, Callable[[], Any]]:
        provider, reader = make_metric_capture()
        return OtelMetricsExtender(meter_provider=provider), lambda: _metric_names(reader)

    def sink_probe_expected_content(self) -> set[str] | None:
        return {_STEP_DURATION}

    def make_injected_and_sdk_defaults_extender(self) -> Extender:
        provider, _ = make_metric_capture()
        return OtelMetricsExtender(meter_provider=provider, use_sdk_defaults=True)

    @classmethod
    def supports_unpicklable_sink_degrade(cls) -> bool:
        return True

    def make_unpicklable_sink_extender(self) -> Extender:
        provider, _ = make_metric_capture()
        return OtelMetricsExtender(meter_provider=provider)

    @classmethod
    def sink_noun(cls) -> str | None:
        return "meter_provider"


class TestOtelMetricsExtenderModule:
    """The one-way dependency: the metrics module reads shared constants from otel_extender, never the reverse."""

    def test_shares_the_operation_names_and_scope_with_otel_extender(self) -> None:
        for name in ("_OPERATION_NAMES", "_DECLARABLE_HOOKS", "_TRACER_NAME"):
            assert getattr(otel_metrics_extender_module, name) is getattr(otel_extender_module, name), name

    def test_otel_extender_has_no_metrics_surface(self) -> None:
        assert "meter_provider" not in inspect.signature(OtelExtender.__init__).parameters
        assert "otel_metrics" not in Path(otel_extender_module.__file__ or "").read_text()


class TestOtelMetricsExtenderConstructorOptions:
    def test_meter_provider_is_stored_on_construction(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, _ = metric_capture
        assert OtelMetricsExtender(meter_provider=provider)._meter_provider is provider

    def test_default_meter_provider_is_none(self) -> None:
        assert OtelMetricsExtender()._meter_provider is None

    def test_constructor_parameters(self) -> None:
        parameters = list(inspect.signature(OtelMetricsExtender.__init__).parameters)
        assert parameters == ["self", "raise_on_error", "meter_provider", "use_sdk_defaults"]

    def test_defaults_match_otel_extender_priority_and_close_timeout(self) -> None:
        otel = OtelMetricsExtender()
        assert otel.priority == OtelExtender().priority
        assert otel.close_timeout == CLOSE_TIMEOUT
        assert otel.raise_on_error is False
        assert otel.use_sdk_defaults is False


class TestOtelMetricsExtenderInertWarning:
    def test_inert_extender_warns_once_per_instance_and_still_calls_through(
        self, ambient_provider: _AmbientMeterProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        otel = OtelMetricsExtender()

        with caplog.at_level(logging.WARNING):
            assert _call_once(otel) == 42
            assert _call_once(otel) == 42
            records = _inert_records(caplog)
            assert len(records) == 1, caplog.records
            message = records[0].getMessage()
            assert "OtelMetricsExtender" in message and "meter_provider" in message
            assert "use_sdk_defaults=True" in message

            _call_once(OtelMetricsExtender())

        assert len(_inert_records(caplog)) == 2, caplog.records
        assert ambient_provider.meter_provider_calls == 0

    def test_injected_meter_provider_is_not_inert(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, _ = metric_capture

        with caplog.at_level(logging.WARNING):
            _call_once(OtelMetricsExtender(meter_provider=provider))

        assert _inert_records(caplog) == [], caplog.records

    def test_pickled_copy_warns_for_its_own_inert_state(self, caplog: pytest.LogCaptureFixture) -> None:
        otel = OtelMetricsExtender()
        with caplog.at_level(logging.WARNING):
            _call_once(otel)
            copy = pickle.loads(pickle.dumps(otel))  # nosec
            _call_once(copy)

        assert len(_inert_records(caplog)) == 2, caplog.records


class TestOtelMetricsExtenderNoSdkProviderWarning:
    """use_sdk_defaults with only the API default meter provider: one warning per instance."""

    @pytest.mark.parametrize("default_kind", ["proxy", "noop"])
    def test_warns_once_per_instance_when_ambient_provider_is_the_api_default(
        self, ambient_provider: _AmbientMeterProvider, caplog: pytest.LogCaptureFixture, default_kind: str
    ) -> None:
        if default_kind == "noop":
            ambient_provider.meter_provider = metrics.NoOpMeterProvider()
        otel = OtelMetricsExtender(use_sdk_defaults=True)

        with caplog.at_level(logging.WARNING):
            assert _call_once(otel) == 42
            assert _call_once(otel) == 42

            records = _marker_records(caplog)
            assert len(records) == 1, caplog.records
            assert records[0].levelno == logging.WARNING
            message = records[0].getMessage()
            assert "OtelMetricsExtender" in message and "opentelemetry-sdk" in message
            assert "inert" not in message.lower()

            _call_once(OtelMetricsExtender(use_sdk_defaults=True))

        assert len(_marker_records(caplog)) == 2, caplog.records

    def test_real_ambient_provider_does_not_warn_and_receives_the_metrics(
        self,
        ambient_provider: _AmbientMeterProvider,
        metric_capture: tuple[MeterProvider, InMemoryMetricReader],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        provider, reader = metric_capture
        ambient_provider.meter_provider = provider

        with caplog.at_level(logging.WARNING):
            assert _call_once(OtelMetricsExtender(use_sdk_defaults=True)) == 42

        assert _marker_records(caplog) == []
        assert _STEP_DURATION in _metric_names(reader)

    def test_pickled_copy_warns_again(
        self, ambient_provider: _AmbientMeterProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        otel = OtelMetricsExtender(use_sdk_defaults=True)

        with caplog.at_level(logging.WARNING):
            _call_once(otel)
            assert len(_marker_records(caplog)) == 1, caplog.records

            _call_once(pickle.loads(pickle.dumps(otel)))  # nosec

        assert len(_marker_records(caplog)) == 2, caplog.records


class TestOtelMetricsExtenderPickling:
    def test_unpicklable_meter_provider_is_dropped_with_the_documented_warning(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, _ = metric_capture
        reason = pickle_failure_reason(provider)
        otel = OtelMetricsExtender(meter_provider=provider)

        with caplog.at_level(logging.WARNING):
            copy = pickle.loads(pickle.dumps(otel))  # nosec
            pickle.dumps(otel)  # nosec

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [
            f"OtelMetricsExtender drops an injected meter_provider when pickled or copied because it isn't "
            f"picklable ({reason}); the copy is inert unless use_sdk_defaults=True, which lets it resolve a "
            "provider installed in its own process, e.g. via child_bootstrap under MULTIPROCESSING."
        ]
        assert copy._meter_provider is None

    def test_picklable_provider_is_kept_without_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        otel = OtelMetricsExtender(meter_provider=metrics.NoOpMeterProvider())

        with caplog.at_level(logging.WARNING):
            copy = pickle.loads(pickle.dumps(otel))  # nosec

        assert isinstance(copy._meter_provider, metrics.NoOpMeterProvider)
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_pickling_leaves_the_original_provider_in_place(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, _ = metric_capture
        otel = OtelMetricsExtender(meter_provider=provider)

        pickle.dumps(otel)  # nosec

        assert otel._meter_provider is provider


class _RaisingMeterProvider(metrics.MeterProvider):
    """Picklable meter provider whose get_meter always raises; counts the attempts per instance."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.get_meter_calls = 0

    def get_meter(self, *args: Any, **kwargs: Any) -> metrics.Meter:
        self.get_meter_calls += 1
        raise self.error


class _BrokenInstrument:
    def record(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("record boom")

    add = record


class _BrokenMeter:
    def create_histogram(self, *args: Any, **kwargs: Any) -> _BrokenInstrument:
        return _BrokenInstrument()

    create_counter = create_histogram


class _BrokenInstrumentMeterProvider(metrics.MeterProvider):
    """Picklable meter provider whose instruments raise on every record, so only post-call recording fails."""

    def get_meter(self, *args: Any, **kwargs: Any) -> Any:
        return _BrokenMeter()


class _StepBoom(Exception):
    pass


class _Halt(BaseException):
    pass


def _rows_context() -> HookContext:
    return make_hook_context(rows_in=3, rows_out=2)


def _failing_step(otel: OtelMetricsExtender, error: BaseException, context: HookContext | None = None) -> None:
    def func() -> None:
        raise error

    with (context or make_hook_context()).activate():
        with pytest.raises(type(error)) as caught:
            otel(func)
    assert caught.value is error


class TestOtelMetricsExtenderFailureHandling:
    """A failure resolving the meter follows core's raise_on_error; post-call recording never changes the step."""

    @pytest.mark.parametrize("error_class", [RuntimeError, TypeError])
    def test_a_raising_meter_provider_falls_back_with_a_warning_when_raise_on_error_is_false(
        self, caplog: pytest.LogCaptureFixture, error_class: type[Exception]
    ) -> None:
        meter_provider = _RaisingMeterProvider(error_class("meter boom"))
        composite = CompositeExtender([OtelMetricsExtender(meter_provider=meter_provider, raise_on_error=False)])
        calls = 0

        def func() -> int:
            nonlocal calls
            calls += 1
            return 42

        with caplog.at_level(logging.WARNING):
            with make_hook_context().activate():
                assert composite(func) == 42

        assert calls == 1
        assert meter_provider.get_meter_calls == 1
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("OtelMetricsExtender" in message for message in warnings), warnings

    def test_a_raising_meter_provider_propagates_before_the_step_when_raise_on_error_is_true(self) -> None:
        meter_provider = _RaisingMeterProvider(RuntimeError("meter boom"))
        composite = CompositeExtender([OtelMetricsExtender(meter_provider=meter_provider, raise_on_error=True)])
        ran = Mock()

        with make_hook_context().activate():
            with pytest.raises(RuntimeError, match="meter boom"):
                composite(ran)

        ran.assert_not_called()

    @pytest.mark.parametrize("raise_on_error", [False, True])
    def test_a_raising_instrument_never_breaks_the_call_and_warns_once(
        self, caplog: pytest.LogCaptureFixture, raise_on_error: bool
    ) -> None:
        otel = OtelMetricsExtender(meter_provider=_BrokenInstrumentMeterProvider(), raise_on_error=raise_on_error)

        with caplog.at_level(logging.WARNING):
            assert _call_once(otel, _rows_context()) == 42
            assert _call_once(otel, _rows_context()) == 42
            _complete_run(otel, "r", started_at=datetime.datetime.now(datetime.timezone.utc))

        messages = [r.getMessage() for r in caplog.records if "metric recording failed" in r.getMessage()]
        assert messages == ["OtelMetricsExtender metric recording failed: RuntimeError"], messages
        assert "record boom" not in caplog.text

    def test_a_raising_instrument_never_replaces_the_steps_own_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        otel = OtelMetricsExtender(meter_provider=_BrokenInstrumentMeterProvider(), raise_on_error=True)
        error = _StepBoom("step boom")

        with caplog.at_level(logging.WARNING):
            _failing_step(otel, error)
            _failing_step(otel, _Halt("halt"))

        assert "OtelMetricsExtender metric recording failed: RuntimeError" in caplog.text

    def test_a_raising_instrument_in_on_run_complete_never_raises(self, caplog: pytest.LogCaptureFixture) -> None:
        otel = OtelMetricsExtender(meter_provider=_BrokenInstrumentMeterProvider(), raise_on_error=True)

        with caplog.at_level(logging.WARNING):
            _complete_run(otel, "r", started_at=datetime.datetime.now(datetime.timezone.utc))

        assert "OtelMetricsExtender metric recording failed: RuntimeError" in caplog.text

    def test_recording_failure_warns_once_per_instance(self, caplog: pytest.LogCaptureFixture) -> None:
        otel = OtelMetricsExtender(meter_provider=_BrokenInstrumentMeterProvider())

        with caplog.at_level(logging.WARNING):
            _call_once(otel)
            _call_once(otel)
            assert len(self._failure_records(caplog)) == 1, caplog.records

            copy = pickle.loads(pickle.dumps(otel))  # nosec
            _call_once(copy)
            _call_once(copy)

        assert len(self._failure_records(caplog)) == 2, caplog.records

    @staticmethod
    def _failure_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
        return [r for r in caplog.records if "metric recording failed" in r.getMessage()]


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


class TestOtelMetricsExtenderMetrics:
    """Metrics emitted through opentelemetry.metrics (API only): step and run durations plus row counters."""

    def test_exactly_four_instruments_with_kinds_units_and_bucket_advisory(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture

        _call_once(OtelMetricsExtender(meter_provider=provider), _rows_context())

        collected = _collected(reader)
        assert set(collected) == {_STEP_DURATION, _ROWS_IN, _ROWS_OUT}
        otel = OtelMetricsExtender(meter_provider=provider)
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

        _call_once(OtelMetricsExtender(meter_provider=provider), context)

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

        _failing_step(OtelMetricsExtender(meter_provider=provider), error, make_hook_context(rows_in=3, rows_out=2))

        point = _single_point(reader, _STEP_DURATION)
        assert point.attributes["error.type"] == _failure_type(_StepBoom)
        assert point.attributes["mloda.operation.name"] == "calculate"
        assert _points(reader, _ROWS_IN) == []
        assert _points(reader, _ROWS_OUT) == []

    def test_base_exception_is_counted_and_propagates(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture

        _failing_step(OtelMetricsExtender(meter_provider=provider), _Halt("halt"))

        assert _single_point(reader, _STEP_DURATION).attributes["error.type"] == _failure_type(_Halt)

    def test_success_has_no_error_type(self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]) -> None:
        provider, reader = metric_capture

        _call_once(OtelMetricsExtender(meter_provider=provider), _rows_context())

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
            OtelMetricsExtender(meter_provider=provider),
            make_hook_context(hook=hook, rows_in=rows_in, rows_out=rows_out),
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
        otel = OtelMetricsExtender(meter_provider=provider)
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

        assert OtelMetricsExtender(meter_provider=provider)(lambda: 42) == 42

        assert _collected(reader) == {}

    def test_step_duration_times_only_the_wrapped_call(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture

        with make_hook_context().activate():
            OtelMetricsExtender(meter_provider=provider)(lambda: time.sleep(0.05))

        assert 0.05 <= _single_point(reader, _STEP_DURATION).sum < 5

    @pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
    def test_run_duration_is_recorded_with_the_run_status(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], status: Any
    ) -> None:
        provider, reader = metric_capture
        started_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=2)
        error_type = "ValueError" if status == "failed" else None

        _complete_run(
            OtelMetricsExtender(meter_provider=provider),
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
            OtelMetricsExtender(meter_provider=provider),
            "r",
            started_at=datetime.datetime.now(datetime.timezone.utc),
            status="failed",
        )

        assert dict(_single_point(reader, _RUN_DURATION).attributes) == {"mloda.run.status": "failed"}

    def test_run_without_started_at_records_no_duration(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture

        _complete_run(OtelMetricsExtender(meter_provider=provider), "r")

        assert _points(reader, _RUN_DURATION) == []

    def test_run_without_run_id_still_records_the_duration(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        provider, reader = metric_capture

        _complete_run(
            OtelMetricsExtender(meter_provider=provider),
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

        _complete_run(OtelMetricsExtender(meter_provider=provider), "r", started_at=started_at, status="succeeded")

        assert _single_point(reader, _RUN_DURATION).sum == 0

    def test_use_sdk_defaults_resolves_the_global_meter_provider(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], ambient_provider: _AmbientMeterProvider
    ) -> None:
        provider, reader = metric_capture
        ambient_provider.meter_provider = provider

        _call_once(OtelMetricsExtender(use_sdk_defaults=True), _rows_context())

        assert _single_point(reader, _STEP_DURATION).count == 1

    def test_injected_meter_provider_wins_over_the_global_one(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], ambient_provider: _AmbientMeterProvider
    ) -> None:
        provider, reader = metric_capture
        global_provider, global_reader = make_metric_capture()
        ambient_provider.meter_provider = global_provider

        _call_once(OtelMetricsExtender(meter_provider=provider, use_sdk_defaults=True), _rows_context())

        assert _single_point(reader, _STEP_DURATION).count == 1
        assert _collected(global_reader) == {}

    def test_unconfigured_extender_never_resolves_the_meter_provider_and_records_nothing(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], ambient_provider: _AmbientMeterProvider
    ) -> None:
        provider, reader = metric_capture
        ambient_provider.meter_provider = provider
        otel = OtelMetricsExtender()

        assert _call_once(otel, _rows_context()) == 42
        _complete_run(otel, "r", started_at=datetime.datetime.now(datetime.timezone.utc), status="succeeded")

        assert ambient_provider.meter_provider_calls == 0
        assert _collected(reader) == {}

    def test_one_meter_is_created_per_provider_across_extender_instances(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, reader = metric_capture
        spy = Mock(wraps=provider.get_meter)
        monkeypatch.setattr(provider, "get_meter", spy)

        for _ in range(5):
            otel = OtelMetricsExtender(meter_provider=provider)
            _call_once(otel, _rows_context())
            _complete_run(otel, "r", started_at=datetime.datetime.now(datetime.timezone.utc), status="succeeded")

        spy.assert_called_once()
        assert spy.call_args.args[0] == "mloda_community_otel"
        assert _single_point(reader, _STEP_DURATION).count == 5

    def test_repeated_recordings_on_the_api_proxy_meter_provider_add_one_meter(
        self, ambient_provider: _AmbientMeterProvider
    ) -> None:
        proxy: Any = ambient_provider.meter_provider
        if not hasattr(proxy, "_meters"):
            pytest.skip("global meter provider already installed in this process")
        before = len(proxy._meters)

        for _ in range(5):
            otel = OtelMetricsExtender(use_sdk_defaults=True)
            _call_once(otel, _rows_context())
            _complete_run(otel, "r", started_at=datetime.datetime.now(datetime.timezone.utc))

        assert len(proxy._meters) - before <= 1


class _SpanProbingMeterProvider(MeterProvider):
    """Real SDK meter provider that records the active span at each get_meter (instrument resolution) call."""

    def __init__(self) -> None:
        self.reader = InMemoryMetricReader()
        super().__init__(metric_readers=[self.reader], shutdown_on_exit=False)
        self.active_spans: list[Any] = []

    def get_meter(self, *args: Any, **kwargs: Any) -> Any:
        span = trace.get_current_span()
        self.active_spans.append((span, span.is_recording()))
        return super().get_meter(*args, **kwargs)


class TestOtelMetricsExtenderRunAll:
    """End-to-end wiring through mloda.user.mloda.run_all, alone and next to OtelExtender."""

    def test_run_all_records_step_and_run_metrics(
        self, metric_capture: tuple[MeterProvider, InMemoryMetricReader]
    ) -> None:
        meter_provider, reader = metric_capture

        values = run_value_int(OtelMetricsExtender(meter_provider=meter_provider))

        assert values == expected_value_int()
        durations = _points(reader, _STEP_DURATION)
        assert any(point.attributes["mloda.operation.name"] == "calculate" for point in durations), durations
        run_points = _points(reader, _RUN_DURATION)
        assert [point.attributes["mloda.run.status"] for point in run_points] == ["succeeded"]

    def test_run_all_with_both_extenders_produces_spans_and_metrics(
        self,
        otel_capture: tuple[TracerProvider, InMemorySpanExporter],
        metric_capture: tuple[MeterProvider, InMemoryMetricReader],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        tracer_provider, exporter = otel_capture
        meter_provider, reader = metric_capture

        with caplog.at_level(logging.WARNING):
            values = run_value_int(
                OtelExtender(tracer_provider=tracer_provider), OtelMetricsExtender(meter_provider=meter_provider)
            )

        assert values == expected_value_int()
        assert any(span.name.startswith("calculate ") for span in exporter.get_finished_spans())
        assert any(point.attributes["mloda.operation.name"] == "calculate" for point in _points(reader, _STEP_DURATION))
        assert [point.attributes["mloda.run.status"] for point in _points(reader, _RUN_DURATION)] == ["succeeded"]
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_the_metrics_extender_runs_inside_the_otel_extender_span(
        self, otel_capture: tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        tracer_provider, exporter = otel_capture
        meter_provider = _SpanProbingMeterProvider()
        composite = CompositeExtender(
            [OtelMetricsExtender(meter_provider=meter_provider), OtelExtender(tracer_provider=tracer_provider)],
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
        )

        with make_hook_context().activate():
            assert composite(lambda: 42) == 42

        [(active, recording)] = meter_provider.active_spans
        assert active.get_span_context().is_valid and recording
        assert single_span(exporter).name == "calculate DummyFeatureGroup"
        assert active.get_span_context().span_id == single_span(exporter).context.span_id
        assert _STEP_DURATION in _metric_names(meter_provider.reader)


class TestOtelMetricsExtenderClose:
    """close() flushes the resolved meter_provider within its own close_timeout; never terminal, never raises,
    never calls shutdown() (core, not the extender, owns provider lifetime)."""

    def test_close_flushes_the_injected_provider_with_timeout_millis(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelMetricsExtender(meter_provider=provider)

        otel.close()

        provider.force_flush.assert_called_once_with(timeout_millis=int(CLOSE_TIMEOUT * 1000))

    def test_close_caps_flush_timeout_to_the_active_close_context(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelMetricsExtender(meter_provider=provider)
        otel.close_timeout = 5.0

        with active_close_context(3.0):
            otel.close()

        provider.force_flush.assert_called_once()
        assert 0 < provider.force_flush.call_args.kwargs["timeout_millis"] <= 3000

    def test_close_flushes_the_global_provider_under_use_sdk_defaults(
        self, ambient_provider: _AmbientMeterProvider
    ) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        ambient_provider.meter_provider = provider
        otel = OtelMetricsExtender(use_sdk_defaults=True)

        otel.close()

        provider.force_flush.assert_called_once()

    def test_inert_extender_close_touches_no_provider(self, ambient_provider: _AmbientMeterProvider) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        ambient_provider.meter_provider = provider
        otel = OtelMetricsExtender()  # no injected provider, use_sdk_defaults False: inert

        otel.close()

        provider.force_flush.assert_not_called()
        assert ambient_provider.meter_provider_calls == 0

    def test_close_never_calls_shutdown(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelMetricsExtender(meter_provider=provider)

        otel.close()

        provider.shutdown.assert_not_called()

    def test_close_swallows_a_raising_force_flush_and_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = Mock(force_flush=Mock(side_effect=RuntimeError("flush boom")))
        otel = OtelMetricsExtender(meter_provider=provider)

        with caplog.at_level(logging.WARNING):
            otel.close()  # must not raise

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == ["OtelMetricsExtender failed to flush meter_provider: RuntimeError"]
        assert "flush boom" not in caplog.text

    def test_close_logs_a_warning_when_force_flush_returns_false(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = Mock(force_flush=Mock(return_value=False))
        otel = OtelMetricsExtender(meter_provider=provider)

        with caplog.at_level(logging.WARNING):
            otel.close()

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == ["OtelMetricsExtender did not flush all metrics within its close budget"]

    def test_close_logs_nothing_when_provider_has_no_force_flush(self, caplog: pytest.LogCaptureFixture) -> None:
        class _NoFlushProvider:
            pass

        otel = OtelMetricsExtender(meter_provider=_NoFlushProvider())  # type: ignore[arg-type]

        with caplog.at_level(logging.WARNING):
            otel.close()

        assert caplog.records == []

    def test_close_timeout_override_is_honored(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelMetricsExtender(meter_provider=provider)
        otel.close_timeout = 5.0

        otel.close()

        provider.force_flush.assert_called_once_with(timeout_millis=5000)

    def test_close_bounds_a_blocking_force_flush_and_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """opentelemetry-sdk's BatchProcessor.force_flush(timeout_millis) currently ignores the timeout
        and exports synchronously; close() must still return well under a second."""
        with blocking_flush_provider() as provider:
            otel = OtelMetricsExtender(meter_provider=provider)
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
        assert any("OtelMetricsExtender" in message for message in warnings), warnings

    def test_negative_close_timeout_calls_force_flush_with_no_args(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelMetricsExtender(meter_provider=provider)
        otel.close_timeout = -1.0

        otel.close()

        provider.force_flush.assert_called_once_with()

    def test_close_with_a_nonsensical_close_timeout_never_raises_and_logs_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        otel = OtelMetricsExtender(meter_provider=provider)
        otel.close_timeout = None  # type: ignore[assignment]

        with caplog.at_level(logging.WARNING):
            otel.close()  # must not raise

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("OtelMetricsExtender" in message and "Error" in message for message in warnings), warnings

    def test_inert_close_never_raises_even_with_a_nonsensical_close_timeout(
        self, ambient_provider: _AmbientMeterProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        otel = OtelMetricsExtender()
        otel.close_timeout = None  # type: ignore[assignment]

        with caplog.at_level(logging.WARNING):
            otel.close()

        assert ambient_provider.meter_provider_calls == 0
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_close_flushes_a_real_meter_provider_and_exports_pending_metrics(self) -> None:
        provider, reader = make_metric_capture()
        otel = OtelMetricsExtender(meter_provider=provider)
        _call_once(otel)

        otel.close()

        assert _STEP_DURATION in _metric_names(reader)
