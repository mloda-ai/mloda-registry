"""Bounded-flush primitives shared by registry extenders' close(); must not import opentelemetry so
extenders without an OTel dependency can still use it."""

from __future__ import annotations

import math
import threading
from typing import Any

CLOSE_TIMEOUT = 1.0
"""Default per-extender flush cap, in seconds, applied on a MULTIPROCESSING worker's graceful exit."""


def to_timeout_millis(seconds: float) -> int | None:
    """Convert a close_timeout in seconds to milliseconds, or None for "no cap" (negative or infinite)."""
    if seconds < 0 or math.isinf(seconds):
        return None
    return int(seconds * 1000)


def force_flush(provider: Any, timeout_millis: int | None = None) -> bool | None:
    """Call provider.force_flush() if present and callable, returning its own boolean result.

    None means provider has no callable force_flush, distinct from a real False. A genuine exception
    raised by the provider's own force_flush() propagates unmodified.

    When timeout_millis is given, the call runs on a daemon thread joined for timeout_millis: the OTel
    SDK's batch processors ignore timeout_millis, so this is the only way to actually bound the wait.
    If the thread hasn't finished by then, the flush keeps running in the background and False is
    returned; timeout_millis=None keeps the direct, unbounded call.
    """
    flush = getattr(provider, "force_flush", None)
    if not callable(flush):
        return None
    if timeout_millis is None:
        return bool(flush())

    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["result"] = flush(timeout_millis=timeout_millis)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread below
            outcome["error"] = exc

    thread = threading.Thread(target=run, name="mloda-extender-flush", daemon=True)
    thread.start()
    thread.join(timeout_millis / 1000)
    if thread.is_alive():
        return False
    if "error" in outcome:
        raise outcome["error"]
    return bool(outcome["result"])
