"""SQLite "what counts as numeric" source-column check for arithmetic.

Shared by the point-arithmetic and scalar-arithmetic families so both reject
the same set of non-numeric SQLite source columns up-front.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.sqlite_helpers import (
    column_affinities,
    first_storage_class_outside,
)

_NUMERIC_STORAGE_CLASSES: frozenset[str] = frozenset({"integer", "real"})


def sqlite_non_numeric_descriptor(data: Any, source_col: str) -> str | None:
    """Return a descriptor (declared affinity or storage class) when NON-numeric, else ``None``.

    Uses ``PRAGMA table_info`` declared affinity. Returns ``None`` when the
    column is absent (presence is validated separately by the caller).
    A computed column (empty declared type) is probed by storage class instead.

    Caveat: ``SqliteRelation.from_arrow`` maps arrow booleans to SQLite
    ``INTEGER`` affinity (see ``mloda_plugins`` ``_arrow_type_to_sqlite``), so a
    boolean source column is indistinguishable from ``int64`` at the relation
    level. The shared boolean-source tests are therefore skipped for SQLite
    (``detects_non_numeric_source`` override); the 0/1 result is instead pinned by
    ``test_boolean_source_column_computed_when_undetected`` and documented in
    ``docs/guides/data-operation-patterns/known-divergences.md``.
    """
    affinity = column_affinities(data).get(source_col)
    if affinity is None:
        return None
    if affinity == "":
        offender = first_storage_class_outside(data, source_col, _NUMERIC_STORAGE_CLASSES)
        return None if offender is None else f"SQLite storage class {offender!r}"
    if "INT" in affinity or "REAL" in affinity or "FLOA" in affinity or "DOUB" in affinity or "NUMERIC" in affinity:
        return None
    return f"SQLite affinity {affinity!r}"
