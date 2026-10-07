"""OtelMetricsExtender: emits OpenTelemetry metrics for mloda pipeline hooks (OTel API only)."""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from mloda.steward import (
    Extender,
    ExtenderHook,
    HookContext,
    LifecycleOutcome,
    RunContext,
    WarnOncePerInstance,
    pickle_failure_reason,
)
from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, MeterProvider

from mloda.community.extenders.otel.otel_extender import _DECLARABLE_HOOKS, _OPERATION_NAMES, _TRACER_NAME
from mloda.community.extenders.shared.teardown import (
    CLOSE_TIMEOUT,
    capped_close_timeout,
    force_flush,
    to_timeout_millis,
)

logger = logging.getLogger(__name__)

_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)


def _is_api_default(provider: object) -> bool:
    # The API proxy and no-op providers live in opentelemetry.metrics; the SDK provider does not (no private imports).
    return type(provider).__module__.startswith("opentelemetry.metrics")


_INERT_MESSAGE = (
    "OtelMetricsExtender is inert: no injected meter_provider and use_sdk_defaults is False; no metrics will be "
    "recorded. Inject a meter_provider, or pass use_sdk_defaults=True and configure an OpenTelemetry SDK "
    "meter provider, to enable recording."
)

_NO_SDK_PROVIDER_MESSAGE = (
    "OtelMetricsExtender found no OpenTelemetry SDK meter provider while use_sdk_defaults is True; no metrics "
    "will be exported. Configure an SDK MeterProvider with a reader via opentelemetry.metrics.set_meter_provider "
    "(install opentelemetry-sdk if missing; under MULTIPROCESSING, in each worker via child_bootstrap), "
    "or inject a meter_provider."
)


class Instruments:
    def __init__(self, provider: MeterProvider, meter_name: str) -> None:
        meter = metrics.get_meter(meter_name, meter_provider=provider)
        self.run_duration: Histogram = meter.create_histogram(
            "mloda.run.duration", unit="s", explicit_bucket_boundaries_advisory=_DURATION_BUCKETS
        )
        self.step_duration: Histogram = meter.create_histogram(
            "mloda.step.duration", unit="s", explicit_bucket_boundaries_advisory=_DURATION_BUCKETS
        )
        self.rows_in: Counter = meter.create_counter("mloda.step.rows.in", unit="{row}")
        self.rows_out: Counter = meter.create_counter("mloda.step.rows.out", unit="{row}")


# One meter and instrument set per provider object for the whole process (the API proxy provider
# would otherwise append a meter and proxy instruments on every get_meter call).
_INSTRUMENTS: weakref.WeakKeyDictionary[Any, Instruments] = weakref.WeakKeyDictionary()
_INSTRUMENTS_LOCK = threading.Lock()


def instruments_for(provider: MeterProvider, meter_name: str) -> Instruments:
    with _INSTRUMENTS_LOCK:
        try:
            cached = _INSTRUMENTS.get(provider)
        except TypeError:  # not weak-referenceable or not hashable: build uncached
            return Instruments(provider, meter_name)
        if cached is None:
            cached = _INSTRUMENTS[provider] = Instruments(provider, meter_name)
        return cached


def record_step(
    instruments: Instruments, context: HookContext, seconds: float, error_type: str | None, ok: bool
) -> None:
    """Step duration always; rows in/out only for calculate and load on success."""
    attributes: dict[str, str] = {"mloda.operation.name": _OPERATION_NAMES.get(context.hook, "unknown")}
    if context.feature_group_class is not None:
        attributes["mloda.feature_group.name"] = context.feature_group_class
    if context.compute_framework_name is not None:
        attributes["mloda.compute_framework.name"] = context.compute_framework_name
    duration_attributes = attributes if error_type is None else {**attributes, "error.type": error_type}
    instruments.step_duration.record(seconds, duration_attributes)
    if ok and context.hook in _DECLARABLE_HOOKS:
        if context.rows_in is not None:
            instruments.rows_in.add(context.rows_in, attributes)
        if context.rows_out is not None:
            instruments.rows_out.add(context.rows_out, attributes)


def record_run(instruments: Instruments, run: RunContext, outcome: LifecycleOutcome) -> None:
    """Run duration since started_at (clamped at 0); skipped when started_at is unknown."""
    started_at = run.started_at
    if started_at is None:
        return
    attributes: dict[str, str] = {"mloda.run.status": outcome.status}
    if outcome.status == "failed" and outcome.error_type is not None:
        attributes["error.type"] = outcome.error_type
    seconds = max(0.0, (datetime.now(timezone.utc) - started_at).total_seconds())
    instruments.run_duration.record(seconds, attributes)


class OtelMetricsExtender(Extender):
    """Records run and step durations and step row counts for wrapped hooks.

    Uses the injected meter_provider, else the global one with use_sdk_defaults, else is inert (warns once).
    close() flushes the provider and never calls shutdown(). Chains inside OtelExtender by default."""

    close_timeout: float = CLOSE_TIMEOUT

    def __init__(
        self,
        raise_on_error: bool = False,
        meter_provider: MeterProvider | None = None,
        use_sdk_defaults: bool = False,
    ) -> None:
        self.raise_on_error = raise_on_error
        self._meter_provider = meter_provider
        self.use_sdk_defaults = use_sdk_defaults
        self._inert_warning = WarnOncePerInstance()  # shared by the inert and no-SDK warnings
        self._pickle_drop_warning = WarnOncePerInstance()
        self._failure_warning = WarnOncePerInstance()

    def wraps(self) -> set[ExtenderHook]:
        return set(_OPERATION_NAMES)

    def _configured_provider(self) -> MeterProvider | None:
        """Injected provider wins, else the global meter provider when use_sdk_defaults, else None."""
        if self._meter_provider is not None:
            return self._meter_provider
        if self.use_sdk_defaults:
            return metrics.get_meter_provider()
        return None

    def _resolve_provider(self) -> MeterProvider | None:
        provider = self._configured_provider()
        if provider is None:
            self._warn_once(_INERT_MESSAGE)
        elif self.use_sdk_defaults and self._meter_provider is None and _is_api_default(provider):
            self._warn_once(_NO_SDK_PROVIDER_MESSAGE)
        return provider

    def _warn_once(self, message: str) -> None:
        self._inert_warning.warn_once(lambda: logger.warning(message))

    def _guarded(self, action: Callable[[], None]) -> None:
        """Best effort: never changes the wrapped call's result or exception."""
        try:
            action()
        except Exception as exc:
            error_name = type(exc).__name__
            self._failure_warning.warn_once(
                lambda: logger.warning("%s metric recording failed: %s", type(self).__name__, error_name)
            )

    def _record(self, instruments: Instruments | None, record: Callable[[Instruments], None]) -> None:
        if instruments is not None:
            self._guarded(lambda: record(instruments))

    def _instruments(self) -> Instruments | None:
        provider = self._resolve_provider()
        return None if provider is None else instruments_for(provider, _TRACER_NAME)

    def on_run_complete(self, run: RunContext, outcome: LifecycleOutcome) -> None:
        self._guarded(lambda: self._record(self._instruments(), lambda i: record_run(i, run, outcome)))

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is None:
            return func(*args, **kwargs)
        instruments = self._instruments()
        started = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            seconds = time.perf_counter() - started
            error_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
            self._record(instruments, lambda i: record_step(i, context, seconds, error_type, ok=False))
            raise
        seconds = time.perf_counter() - started
        self._record(instruments, lambda i: record_step(i, context, seconds, None, ok=True))
        return result

    # Core calls close() with no args on graceful MULTIPROCESSING worker exit and ignores the result.
    def close(self) -> None:
        """Best-effort flush of the provider within the close budget; never raises."""
        name = type(self).__name__
        try:
            provider = self._configured_provider()
            if provider is None:
                return
            millis = to_timeout_millis(capped_close_timeout(self.close_timeout))
            result = force_flush(provider, timeout_millis=millis)
        except Exception as exc:
            logger.warning("%s failed to flush meter_provider: %s", name, type(exc).__name__)
            return
        if result is False:
            logger.warning("%s did not flush all metrics within its close budget", name)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        sink = self._meter_provider
        reason = pickle_failure_reason(sink) if sink is not None else None
        if reason is not None:
            self._pickle_drop_warning.warn_once(
                lambda: logger.warning(
                    f"{type(self).__name__} drops an injected meter_provider when pickled or copied because it "
                    f"isn't picklable ({reason}); the copy is inert unless use_sdk_defaults=True, which lets it "
                    "resolve a provider installed in its own process, e.g. via child_bootstrap under "
                    "MULTIPROCESSING."
                )
            )
            state["_meter_provider"] = None
        return state
