"""Bounded-flush primitives shared by registry extenders' close(); must not import opentelemetry so
extenders without an OTel dependency can still use it."""

from __future__ import annotations

from typing import Any

CLOSE_TIMEOUT = 1.0
"""Default per-extender flush cap, in seconds, applied on a MULTIPROCESSING worker's graceful exit."""


def force_flush(provider: Any, timeout_millis: int | None = None) -> bool | None:
    """Call provider.force_flush() if present and callable, returning its own boolean result.

    None means provider has no callable force_flush (duck-typed as "cannot be asked to flush"), distinct
    from a real False. A genuine exception raised by the provider's own force_flush() propagates unmodified.
    """
    flush = getattr(provider, "force_flush", None)
    if not callable(flush):
        return None
    if timeout_millis is not None:
        return bool(flush(timeout_millis=timeout_millis))
    return bool(flush())
