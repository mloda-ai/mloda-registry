"""Backend aggregation-function tables shared between the aggregation and window-aggregation families.

Removes the duplication that previously existed between each family's per-backend modules,
following the same consolidation precedent as ``aggregation_base.py`` (cf. issue #246).
"""

from __future__ import annotations

# Aggregation types with direct PyArrow group_by support.
PYARROW_AGG_FUNCS: dict[str, str] = {
    "sum": "sum",
    "avg": "mean",
    "mean": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
    "nunique": "count_distinct",
}

# Variance/stddev operations mapped to (PyArrow func name, ddof).
PYARROW_VARIANCE_FUNCS: dict[str, tuple[str, int]] = {
    "std": ("stddev", 0),
    "var": ("variance", 0),
    "std_pop": ("stddev", 0),
    "std_samp": ("stddev", 1),
    "var_pop": ("variance", 0),
    "var_samp": ("variance", 1),
}

# Ordered aggregates need use_threads=False in PyArrow.
PYARROW_ORDERED_FUNCS: dict[str, str] = {
    "first": "first",
    "last": "last",
}

PYARROW_SUPPORTED_AGG_TYPES = {*PYARROW_AGG_FUNCS, *PYARROW_VARIANCE_FUNCS, *PYARROW_ORDERED_FUNCS}

# Aggregation types that SQLite supports natively.
SQLITE_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "mean": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}
