"""SQLite "what counts as numeric" source-column check for arithmetic.

Shared by the point-arithmetic and scalar-arithmetic families so both reject
the same set of non-numeric SQLite source columns up-front.
"""

from __future__ import annotations

from typing import Any

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident


def sqlite_non_numeric_descriptor(data: Any, source_col: str) -> str | None:
    """Return ``f"SQLite affinity {affinity!r}"`` when NON-numeric, else ``None``.

    Uses ``PRAGMA table_info`` declared affinity. Returns ``None`` when the
    column is absent (presence is validated separately by the caller).

    Caveat: ``SqliteRelation.from_arrow`` maps arrow booleans to SQLite
    ``INTEGER`` affinity (see ``mloda_plugins`` ``_arrow_type_to_sqlite``), so a
    boolean source column is indistinguishable from ``int64`` at the relation
    level. The shared boolean-source tests are correspondingly skipped for
    SQLite via the ``detects_non_numeric_source`` test-class override.
    """
    rows = data.connection.execute(f"PRAGMA table_info({quote_ident(data.table_name)})").fetchall()
    affinity_by_column = {row[1]: (row[2] or "").upper() for row in rows}
    affinity = affinity_by_column.get(source_col)
    if affinity is None:
        return None
    if "INT" in affinity or "REAL" in affinity or "FLOA" in affinity or "DOUB" in affinity or "NUMERIC" in affinity:
        return None
    return f"SQLite affinity {affinity!r}"
