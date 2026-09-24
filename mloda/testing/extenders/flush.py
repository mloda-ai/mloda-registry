"""Bounded-wait test helpers for a force_flush(timeout_millis=...) that may not honor its timeout.
Must not import opentelemetry: mloda-community-extenders-shared's own tests have no OTel dependency."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock


def call_with_join_timeout(func: Callable[[], Any], *, join_timeout: float) -> tuple[bool, dict[str, Any]]:
    """Run func() on a daemon thread, joined for join_timeout; returns (still_running, outcome with
    'result' or 'error'). A still-running thread is left running so the test never hangs."""
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["result"] = func()
        except BaseException as exc:  # noqa: BLE001 - captured to re-raise on the calling thread
            outcome["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(join_timeout)
    return thread.is_alive(), outcome


@contextmanager
def blocking_flush_provider() -> Iterator[Mock]:
    """Yield a Mock whose force_flush(timeout_millis=...) ignores the timeout and blocks until the
    context exits, like the OTel SDK's BatchProcessor.force_flush. The Event is always released on
    exit, so no thread outlives the test."""
    release = threading.Event()

    def blocking_force_flush(timeout_millis: int) -> bool:
        release.wait()  # ignores timeout_millis, exactly like the real SDK's BatchProcessor
        return True

    provider = Mock(force_flush=Mock(side_effect=blocking_force_flush))
    try:
        yield provider
    finally:
        release.set()
