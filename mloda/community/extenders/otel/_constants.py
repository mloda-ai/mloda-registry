"""Names shared by the OTel span and metrics extenders."""

from __future__ import annotations

from mloda.steward import ExtenderHook

TRACER_NAME = "mloda_community_otel"

OPERATION_NAMES: dict[ExtenderHook, str] = {
    ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "calculate",
    ExtenderHook.VALIDATE_INPUT_FEATURE: "validate",
    ExtenderHook.VALIDATE_OUTPUT_FEATURE: "validate",
    ExtenderHook.INPUT_DATA_LOAD: "load",
    ExtenderHook.JOIN: "join",
}

# Hooks that record declared attributes, result attributes and rows.out: calculate and load only.
DECLARABLE_HOOKS = {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD}

CANCEL_TYPE_KEY = "mloda.cancel.type"


def exception_type_name(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__qualname__}"


def is_interrupt(exc: BaseException) -> bool:
    """An interrupt is a BaseException that is not an Exception (core's failed vs cancelled rule)."""
    return not isinstance(exc, Exception)
