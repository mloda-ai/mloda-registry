"""Duplicated in mloda-community-openlineage: separate distributions, no shared runtime package."""

from __future__ import annotations

import os
import threading
import uuid
import weakref
from typing import Any

_lock = threading.Lock()
_table: dict[str, Any] = {}


def register(owner: object, handle: Any) -> str:
    """Return a token resolving back to `handle`, in this process only, while `owner` lives."""
    token = f"{os.getpid()}:{uuid.uuid4().hex}"
    with _lock:
        _table[token] = handle

    def _drop(token: str = token) -> None:
        with _lock:
            _table.pop(token, None)

    weakref.finalize(owner, _drop)
    return token


def resolve(token: str | None) -> Any | None:
    """The handle for `token`, or None when it was not minted by this process."""
    # A spawn child starts with an empty table and a fork child inherits it but has a different pid,
    # so neither can ever resolve a token minted by another process.
    if token is None:
        return None
    try:
        pid_str, _, _ = token.partition(":")
        if not pid_str or int(pid_str) != os.getpid():
            return None
    except (ValueError, AttributeError):
        return None
    with _lock:
        return _table.get(token)
