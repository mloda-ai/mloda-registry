"""Owning-class helpers shared by extenders that read class-level declarations off the wrapped call."""

from __future__ import annotations

import inspect
from typing import Any


def bound_method(func: Any) -> Any:
    """The bound method behind func; core's wrappers copy __self__ but not __func__, so stop at the method type."""
    return inspect.unwrap(func, stop=inspect.ismethod)


def class_attribute(func: Any, name: str) -> Any:
    """The named attribute off func's owning class, or None when func has no owner or lacks the attribute."""
    owner = getattr(bound_method(func), "__self__", None)
    owning_class = owner if isinstance(owner, type) else type(owner)
    return getattr(owning_class, name, None)
