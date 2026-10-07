"""Trace correlation for audit records; opentelemetry is optional, so it is imported lazily."""

from __future__ import annotations

import re
from collections.abc import Mapping

from mloda.community.extenders.shared.trace_context import active_span_ids

_TRACEPARENT = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}", re.IGNORECASE)


def _carrier_trace_id(carrier: Mapping[str, str] | None) -> str | None:
    traceparent = carrier.get("traceparent") if carrier else None
    match = _TRACEPARENT.match(traceparent) if isinstance(traceparent, str) else None
    if match is None or int(match.group(1), 16) == 0:
        return None
    return match.group(1).lower()


def trace_ids(carrier: Mapping[str, str] | None) -> tuple[str | None, str | None]:
    """(trace_id, span_id): the active valid span, else the carrier's trace id with no span id, else (None, None)."""
    active = active_span_ids()
    if active is not None:
        return active
    return _carrier_trace_id(carrier), None
