"""Declared-attribute helpers shared by the OpenLineage, audit and OTel extenders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def declared_scalars(declared: Mapping[str, Any] | None) -> dict[str, str | int | float | bool]:
    """A fresh dict of the declarations; core already dropped non-scalars, non-finite floats and big ints."""
    if declared is None:
        return {}
    return dict(declared)


def readable_declared(declared: Mapping[str, Any] | None) -> dict[str, str | int | float | bool] | None:
    """None when core could not read the declaration, else its scalars (possibly empty)."""
    return None if declared is None else declared_scalars(declared)


def same_scalar(left: object, right: object) -> bool:
    """True only when both have exactly the same type and are equal."""
    return type(left) is type(right) and left == right
