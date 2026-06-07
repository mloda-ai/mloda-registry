"""Shared DuckDB aggregation-function table.

The aggregation and window-aggregation families share 15 native DuckDB
function mappings. They deliberately diverge on ``first``/``last``: group
aggregation uses ``FIRST``/``LAST`` while window functions need
``FIRST_VALUE``/``LAST_VALUE``. Each backend module adds those two entries
itself, so only the shared 15 live here.
"""

from __future__ import annotations

# Aggregation types natively supported by DuckDB (excluding first/last, which
# differ between group and window contexts).
DUCKDB_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "mean": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
    "std": "STDDEV_POP",
    "var": "VAR_POP",
    "std_pop": "STDDEV_POP",
    "std_samp": "STDDEV_SAMP",
    "var_pop": "VAR_POP",
    "var_samp": "VAR_SAMP",
    "median": "MEDIAN",
    "mode": "MODE",
    "nunique": "COUNT_DISTINCT",  # handled specially: COUNT(DISTINCT col) syntax
}
