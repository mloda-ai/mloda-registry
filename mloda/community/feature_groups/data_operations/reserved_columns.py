"""Guard that keeps user-supplied columns out of the internal ``__mloda_`` namespace.

Several row-preserving implementations tag rows with a helper column (for
example to restore input order after a reordering window function). When the
helper name is hardcoded, an input already carrying a column with that name
would either overwrite user data or collide with the generated column silently.

The SQL backends (DuckDB, SQLite) no longer rely on this guard: they choose
collision-free helper-column names at runtime via
``pick_helper_column_name`` (in ``sql_utils``), which picks the lowest
``__mloda_rnN__`` name not already present in the input, so reserved-prefixed
USER columns are accepted and processed safely.

The pandas, polars and pyarrow plugins (and the ``scalar_arithmetic`` base)
still use hardcoded helper names, so they call the validator below at entry.
The check is a whole-prefix ban rather than a per-name check so that new
helpers added in the future are covered automatically. The prefix match is
case-insensitive because SQLite and DuckDB fold unquoted identifiers, so a
user column like ``__MLODA_RN__`` would silently collide with an internal
``__mloda_rn__`` helper.
"""

from __future__ import annotations

from collections.abc import Iterable

RESERVED_PREFIX = "__mloda_"


def assert_no_reserved_columns(
    columns: Iterable[str],
    *,
    framework: str | None = None,
    operation: str | None = None,
) -> None:
    """Raise ``ValueError`` if any input column name starts with the reserved prefix.

    Args:
        columns: All column names present on the input data.
        framework: Optional framework label (``"DuckDB"``, ``"SQLite"``,
            ``"Pandas"``, ``"Polars"``, ``"PyArrow"``) echoed in the message.
        operation: Optional operation qualifier (``"frame aggregate"``,
            ``"window aggregation"``, ...) echoed in the message.
    """
    collisions = sorted({c for c in columns if c.lower().startswith(RESERVED_PREFIX)})
    if not collisions:
        return

    prefix = "Input contains reserved helper column name"
    if len(collisions) > 1:
        prefix += "s"
    if framework is not None:
        prefix += f" for {framework}"
    if operation is not None:
        prefix += f" {operation}"

    collision_list = ", ".join(repr(c) for c in collisions)
    raise ValueError(
        f"{prefix}: {collision_list}. "
        f"Column names starting with {RESERVED_PREFIX!r} are reserved for internal use; "
        f"rename the input column."
    )
