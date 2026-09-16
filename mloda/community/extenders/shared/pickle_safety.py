"""Trial-pickle helper shared by extenders that degrade gracefully when an injected sink can't pickle."""

from __future__ import annotations

import pickle  # nosec
from typing import Any


def pickle_failure_reason(value: Any) -> str | None:
    """None if value pickles cleanly, else the caught exception's type name."""
    try:
        pickle.dumps(value)
    except Exception as exc:
        return type(exc).__name__
    return None


def is_picklable(value: Any) -> bool:
    return pickle_failure_reason(value) is None
