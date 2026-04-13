"""DuckDB relation helpers that avoid touching ``DuckdbRelation._relation``.

``DuckdbRelation`` (in ``mloda``) currently exposes only a subset of the
underlying ``DuckDBPyRelation`` operations as public API. The data
operation feature groups occasionally need ``aggregate()`` (GROUP BY on a
lazy relation) and ``query()`` (run SQL against the relation bound to a
table alias), neither of which has a public wrapper yet.

Rather than reach into the private ``_relation`` attribute from every
feature group, this module centralises that coupling in one place. When
``mloda`` adds public equivalents these thin helpers can be replaced by
direct calls without touching the feature group implementations.

Security note: callers must ensure that *agg_expr*, *group_expr* and
*sql* contain only trusted fragments (e.g. identifiers produced by
``quote_ident`` and hardcoded SQL functions). No user input is inlined.
"""

from __future__ import annotations

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation


def group_aggregate(data: DuckdbRelation, agg_expr: str, group_expr: str) -> DuckdbRelation:
    """Return a lazy relation computed as ``SELECT <agg_expr> GROUP BY <group_expr>``.

    Thin wrapper around ``DuckDBPyRelation.aggregate`` that preserves
    laziness (no temp views or eager SQL execution).
    """
    new_rel = data._relation.aggregate(agg_expr, group_expr)
    return DuckdbRelation(data.connection, new_rel)


def query_with_alias(data: DuckdbRelation, alias: str, sql: str) -> DuckdbRelation:
    """Run *sql* against *data* exposed under *alias*, returning a new relation.

    Thin wrapper around ``DuckDBPyRelation.query``. Used when a feature
    group needs full SELECT syntax (e.g. window functions combined with
    row-order tagging) that cannot be expressed via ``project`` alone.
    """
    new_rel = data._relation.query(alias, sql)
    return DuckdbRelation(data.connection, new_rel)
