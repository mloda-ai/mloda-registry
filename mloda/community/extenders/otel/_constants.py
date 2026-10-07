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

# Hooks that record the context's declared attributes and rows.out after the call: calculate and load only.
DECLARABLE_HOOKS = {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD}
