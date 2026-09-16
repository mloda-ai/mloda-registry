"""Trial-pickle helper shared by extenders that degrade gracefully when an injected sink can't pickle."""

from __future__ import annotations

import pickle  # nosec
from typing import Any


def is_picklable(value: Any) -> bool:
    try:
        pickle.dumps(value)  # nosec
    except Exception:
        return False
    return True
