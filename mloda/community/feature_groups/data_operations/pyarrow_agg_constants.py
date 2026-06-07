"""Shared PyArrow aggregation-function tables.

The aggregation and window-aggregation families map the same aggregation
names to the same PyArrow ``group_by().aggregate()`` function names. These
tables were byte-for-byte duplicated across both backend modules; they live
here so a per-function change is made once.
"""

from __future__ import annotations

# Aggregation types with direct PyArrow group_by support.
PA_AGG_FUNCS: dict[str, str] = {
    "sum": "sum",
    "avg": "mean",
    "mean": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
    "nunique": "count_distinct",
}

# Variance/stddev operations mapped to (PyArrow func name, ddof).
PA_VARIANCE_FUNCS: dict[str, tuple[str, int]] = {
    "std": ("stddev", 0),
    "var": ("variance", 0),
    "std_pop": ("stddev", 0),
    "std_samp": ("stddev", 1),
    "var_pop": ("variance", 0),
    "var_samp": ("variance", 1),
}

# Ordered aggregates need use_threads=False in PyArrow.
PA_ORDERED_FUNCS: dict[str, str] = {
    "first": "first",
    "last": "last",
}

PA_SUPPORTED_AGG_TYPES = {*PA_AGG_FUNCS, *PA_VARIANCE_FUNCS, *PA_ORDERED_FUNCS}
