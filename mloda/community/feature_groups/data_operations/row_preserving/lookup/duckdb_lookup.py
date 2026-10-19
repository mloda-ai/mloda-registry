"""DuckDB implementation of key-lookup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pyarrow as pa
from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident

from mloda.community.feature_groups.data_operations.duckdb_helpers import column_types
from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import (
    LookupFeatureGroup,
    assert_key_family,
    build_unmatched_error,
)

# UBIGINT, HUGEINT and UHUGEINT cannot be compared at int64 width and fall to "other".
_INT_TYPES = frozenset({"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER"})


def _source_family(duckdb_type: str) -> str:
    type_name = duckdb_type.upper()
    if type_name in ("VARCHAR", "TEXT", "STRING"):
        return "str"
    if type_name in _INT_TYPES:
        return "int"
    return "other"


class DuckdbLookup(LookupFeatureGroup):
    """DuckDB backend: the mapping is a registered Arrow relation, LEFT JOINed on the key."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _compute_lookup(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        source_col: str,
        mapping: Mapping[Any, Any],
        policy: str,
        reasons: Mapping[Any, str],
    ) -> DuckdbRelation:
        source_type = column_types(data)[source_col]
        assert_key_family(feature_name, source_col, mapping, _source_family(source_type), str(source_type))

        original_cols = list(data.columns)
        taken = set(original_cols) | {feature_name}
        rn = pick_helper_column_name(taken=taken)
        map_key = pick_helper_column_name(taken=taken | {rn}, prefix="__mloda_lk")
        map_val = pick_helper_column_name(taken=taken | {rn, map_key}, prefix="__mloda_lv")

        mapping_table = pa.table({map_key: pa.array(list(mapping.keys())), map_val: pa.array(list(mapping.values()))})
        mapping_rel = DuckdbRelation.from_arrow(data.connection, mapping_table).set_alias("lookup_map")

        tagged = data.with_row_number(rn).set_alias("lookup_src")
        quoted_source = quote_ident(source_col)
        quoted_val = quote_ident(map_val)
        joined = tagged.join(
            mapping_rel,
            f"lookup_src.{quoted_source} = lookup_map.{quote_ident(map_key)}",
            how="left",
        )

        if policy == "raise":
            unmatched = joined.filter(f"{quoted_source} IS NOT NULL AND {quoted_val} IS NULL")
            rows = len(unmatched)
            if rows:
                distinct_rel = unmatched.aggregate(quoted_source, quoted_source)
                distinct = distinct_rel.to_arrow_table().column(0).to_pylist()
                raise build_unmatched_error(feature_name, rows, distinct, reasons)

        # keep: unmatched non-null sources fall through; mapped values are never null.
        result_expr = quoted_val
        if policy == "keep":
            result_expr = f"COALESCE({quoted_val}, CAST({quoted_source} AS {_value_sql_type(mapping)}))"
        ordered = joined.order(quote_ident(rn))
        keep_cols = ", ".join(quote_ident(c) for c in original_cols)
        return ordered.project(f"{keep_cols}, {result_expr} AS {quote_ident(feature_name)}")


def _value_sql_type(mapping: Mapping[Any, Any]) -> str:
    return "VARCHAR" if isinstance(next(iter(mapping.values())), str) else "BIGINT"
