"""Opt-in SIGTERM handler: flags the termination, caps the unwind with a watchdog, exits 143."""

from __future__ import annotations

import logging
import os
import signal
import threading
from types import FrameType
from typing import Any

logger = logging.getLogger(__name__)

_EXIT_CODE = 143

_terminating = False
_installed = False
_previous: Any = None
_grace = 25.0


def terminating() -> bool:
    return _terminating


def _watch(grace: float) -> None:
    threading.Event().wait(grace)
    os._exit(_EXIT_CODE)


def _handle(signum: int, frame: FrameType | None) -> None:
    global _terminating
    if _terminating:
        return
    _terminating = True
    # The hard cap, also for PID 1: a stuck unwind or atexit flush cannot outlive the grace.
    threading.Thread(target=_watch, args=(_grace,), daemon=True).start()
    if not threading.main_thread().is_alive():
        return  # Raising here would cut the running atexit flush short.
    if callable(_previous):
        try:
            _previous(signum, frame)
        except Exception as exc:
            logger.warning("The previous SIGTERM handler failed: %s", type(exc).__name__)
    raise SystemExit(_EXIT_CODE)


def install_sigterm_handler(grace: float = 25.0) -> None:
    """Call on the main thread. Idempotent: a repeat call only updates grace."""
    global _installed, _previous, _grace
    if not 0 < grace <= threading.TIMEOUT_MAX:
        raise ValueError(f"grace must be in (0, {threading.TIMEOUT_MAX}], got {grace!r}")
    _grace = grace
    if _installed:
        return
    _previous = signal.signal(signal.SIGTERM, _handle)
    _installed = True
