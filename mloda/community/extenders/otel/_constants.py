"""Names shared by the OTel span and metrics extenders."""

from __future__ import annotations

from mloda.steward import ExtenderHook

from mloda.community.extenders.shared.exception_type import qualified_type_name

TRACER_NAME = "mloda_community_otel"
# Cap keys per declared/result mapping so span attributes leave room for core attributes.
MAX_DECLARED_KEYS = 32

OPERATION_NAMES: dict[ExtenderHook, str] = {
    ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "calculate",
    ExtenderHook.VALIDATE_INPUT_FEATURE: "validate",
    ExtenderHook.VALIDATE_OUTPUT_FEATURE: "validate",
    ExtenderHook.INPUT_DATA_LOAD: "load",
    ExtenderHook.JOIN: "join",
}

# Hooks that record declared attributes, result attributes and rows.out: calculate and load only.
DECLARABLE_HOOKS = {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD}

ERROR_TYPE_KEY = "error.type"
CANCEL_TYPE_KEY = "mloda.cancel.type"


def exception_attribute(exc: BaseException) -> tuple[str, str]:
    """(key, module.qualname): an interrupt (BaseException, not Exception) is mloda.cancel.type, else error.type."""
    key = ERROR_TYPE_KEY if isinstance(exc, Exception) else CANCEL_TYPE_KEY
    return key, qualified_type_name(exc)


def outcome_attribute_key(status: str) -> str | None:
    """Type attribute key for a failed or cancelled outcome, None for any other status."""
    return {"failed": ERROR_TYPE_KEY, "cancelled": CANCEL_TYPE_KEY}.get(status)
