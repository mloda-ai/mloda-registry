"""Metric instruments and recording for OtelExtender (OTel API only)."""

import threading
import weakref
from datetime import datetime, timezone
from typing import Any

from mloda.steward import ExtenderHook, HookContext, LifecycleOutcome, RunContext
from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, MeterProvider

# Duration histogram bucket advisory, in seconds.
_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)

OPERATION_NAMES: dict[ExtenderHook, str] = {
    ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "calculate",
    ExtenderHook.VALIDATE_INPUT_FEATURE: "validate",
    ExtenderHook.VALIDATE_OUTPUT_FEATURE: "validate",
    ExtenderHook.INPUT_DATA_LOAD: "load",
    ExtenderHook.JOIN: "join",
}

# Hooks that record the context's declared attributes and rows.out after the call: calculate and load only.
DECLARABLE_HOOKS = {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD}


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
    attributes: dict[str, str] = {"mloda.operation.name": OPERATION_NAMES.get(context.hook, "unknown")}
    if context.feature_group_class is not None:
        attributes["mloda.feature_group.name"] = context.feature_group_class
    if context.compute_framework_name is not None:
        attributes["mloda.compute_framework.name"] = context.compute_framework_name
    duration_attributes = attributes if error_type is None else {**attributes, "error.type": error_type}
    instruments.step_duration.record(seconds, duration_attributes)
    if ok and context.hook in DECLARABLE_HOOKS:
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
