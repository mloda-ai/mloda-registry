"""Declared-attribute helpers shared by the OpenLineage, audit and OTel extenders."""

from __future__ import annotations

from collections.abc import Mapping


def declared_scalars(declared: Mapping[str, str | int | float | bool] | None) -> dict[str, str | int | float | bool]:
    """A fresh plain dict of the declarations, which core's HookContext already filtered to portable scalars."""
    return {} if declared is None else dict(declared)


def readable_declared(
    declared: Mapping[str, str | int | float | bool] | None,
) -> dict[str, str | int | float | bool] | None:
    """None when core could not read the declaration, else its portable scalars (possibly empty)."""
    return None if declared is None else declared_scalars(declared)


def same_scalar(left: object, right: object) -> bool:
    """True only when both have exactly the same type and are equal."""
    return type(left) is type(right) and left == right
