"""Exception type naming shared by the OTel and audit extenders."""

from __future__ import annotations


def qualified_type_name(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__qualname__}"
