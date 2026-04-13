"""Shared utilities for collision-safe internal helper column names.

Several feature-group implementations materialise temporary helper columns
(row-number trackers, null-sort flags, ...) during computation. When a user
happens to ship a column with the same name as one of those hardcoded
helpers, the computation silently clobbers or drops the user column.

This module centralises the logic that derives a non-colliding name from
a preferred base name, so that the base name remains visible in logs and
generated SQL when no collision exists.
"""

from __future__ import annotations

from typing import Iterable


def unique_helper_name(base: str, existing: Iterable[str]) -> str:
    """Return a name derived from *base* that is not in *existing*.

    If *base* is absent it is returned unchanged; otherwise appends
    ``_1``, ``_2`` ... until a free name is found.
    """
    existing_set = set(existing)
    if base not in existing_set:
        return base
    i = 1
    while f"{base}_{i}" in existing_set:
        i += 1
    return f"{base}_{i}"
